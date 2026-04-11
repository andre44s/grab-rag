import json
import os
import sys
import time
from pathlib import Path
import argparse
from tqdm import tqdm
import random

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.policies import gbnf, pols, format_messages
from src.detector import detect
from src.retrieval import Retriever
from src.context_quality import conds, ContradictoryGenerator, build_condition
from src.utils import exact_match, f1_score, contains_answer


model_gguf = {
    "phi": "microsoft_Phi-4-mini-instruct-Q4_K_M.gguf",
    "llama": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    "qwen": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
}

dataset_dirs = {
    'nq': "natural_questions",
    "hpqa": "hotpotqa",
}

repo_dir=Path(__file__).parent.parent
seed=42

def find_model_path(model_name):
    #print("find_model_path:", model_name)
    model_dir = repo_dir / "models" / model_name
    if model_name in model_gguf:
        path=model_dir / model_gguf[model_name]
        if path.exists():
            return path
    gguf_files=sorted(model_dir.glob("*.gguf"))
    if gguf_files:
        return gguf_files[0]
    raise FileNotFoundError(f"no gguf file found in {model_dir}")

def load_records(dataset_name):
    eval_file = repo_dir / "data" / dataset_dirs[dataset_name] / "eval.jsonl"
    if not eval_file.exists():
        raise FileNotFoundError(
            f"dataset not found: {eval_file}. run: python scripts/download_datasets.py"
        )
    required_keys={"id","question","answers","gold_passages"}
    records=[]
    n_bad=0
    n_schema=0
    with eval_file.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                n_bad += 1
                continue
            missing = required_keys - rec.keys()
            if missing:
                n_schema += 1
                continue
            records.append(rec)
    if n_bad:
        print(f"warn: {n_bad} corrupted lines in {eval_file.name} (skipped)")
    if n_schema:
        print(f"warn: {n_schema} records missing required keys in {eval_file.name} (skipped)")
    if not records:
        raise ValueError(f"no valid records in {eval_file}")
    return records

def build_context_string(passages, model=None, max_tokens=1500):
    parts=[]
    for i in range(len(passages)):
        p = passages[i]
        title = p.get('title', '').strip()
        text = p.get('text', '').strip()
        if title:
            parts.append(f"[{i + 1}] {title}: {text}")
        else:
            parts.append(f"[{i + 1}] {text}")
    context = "\n\n".join(parts)
    if model != None:
        tokens = model.tokenize(context.encode(), add_bos=False)
        if len(tokens) > max_tokens:
            context = model.detokenize(tokens[:max_tokens]).decode(errors='replace')
    return context

def generate_one(model, messages, grammar, max_tokens=512):
    #print("generate_one:", messages[1]['content'][:50])
    t0=time.perf_counter()
    try:
        result = model.create_chat_completion(
            messages=messages,
            grammar=grammar,
            temperature=0,
            max_tokens=max_tokens,
        )
        content=result['choices'][0]['message']['content']
        usage = result.get('usage', {})
        return {
            "content": content,
            "prompt_tokens": usage.get('prompt_tokens', 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            'wall_clock_s': time.perf_counter() - t0,
            "error": False,
        }
    except Exception as e:
        return {
            'content': f"[error: {type(e).__name__}: {e}]",
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'wall_clock_s': time.perf_counter() - t0,
            'error': True,
        }

def run_experiment(args):
    from llama_cpp import Llama, LlamaGrammar

    if any(c in args.output for c in '\r\n\x00'):
        sys.exit("error: --output path contains control characters")

    out_dir=Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    grammar=LlamaGrammar.from_string(gbnf)

    if "all" in args.models:
        models=list(model_gguf)
    else:
        models = args.models
    if "all" in args.datasets:
        datasets=list(dataset_dirs)
    else:
        datasets = args.datasets
    if "all" in args.policies:
        policies = [p for p in pols if p not in ("CB", "P3")]
    else:
        policies = args.policies
    if "all" in args.conditions:
        conditions=conds
    else:
        conditions = args.conditions

    for model_name in models:
        model_path=find_model_path(model_name)
        print(f"loading model: {model_name} ({model_path.name})")
        model = Llama(
            model_path=str(model_path),
            n_gpu_layers=args.n_gpu_layers,
            n_ctx=args.n_ctx,
            seed=args.seed,
            verbose=False,
        )

        sample_msgs = format_messages("P2", "x", "")
        sample_text = " ".join(m["content"] for m in sample_msgs)
        base_overhead = len(model.tokenize(sample_text.encode(), add_bos=True)) + 64
        if args.n_ctx - base_overhead - args.max_tokens - 128 < 256:
            sys.exit("error: n_ctx too small, increase --n-ctx or reduce --max-tokens")

        for dataset_name in datasets:
            out_file = out_dir / f"{model_name}_{dataset_name}.jsonl"

            records = load_records(dataset_name)

            done = set()
            if out_file.exists() and args.force == True:
                out_file.unlink()
                print("force: deleted existing results")
            elif out_file.exists():
                n_bad=0
                with out_file.open(encoding='utf-8') as f:
                    for line_no, line in enumerate(f, 1):
                        try:
                            r = json.loads(line)
                            done.add((r["question_id"], r['condition'], r["policy"]))
                        except (json.JSONDecodeError, KeyError, TypeError):
                            n_bad += 1
                msg = f"resuming: {len(done)} already computed"
                if n_bad:
                    msg += f" ({n_bad} corrupted lines ignored)"
                print(msg)

            print("building retriever")
            retriever = Retriever.from_records(records)

            contra_gen = None
            if "QC" in conditions:
                print("building contradictory generator")
                contra_gen = ContradictoryGenerator.from_records(records)

            n_resumed=0
            n_written=0
            n_qc_skip=0
            n_build_err=0
            n_gen_err=0
            n_json=0
            n_json_repair=0
            n_string_match=0
            n_needs_judge=0
            n_cap_hit=0
            warned_fallback=False
            warned_cap=False

            n_questions_done=0
            if done:
                done_qids=set()
                for key in done:
                    done_qids.add(key[0])
                for rec in records:
                    if rec["id"] in done_qids:
                        n_questions_done += 1
                    else:
                        break

            t_start=time.time()
            with out_file.open("a", encoding='utf-8') as fout:
                pbar = tqdm(
                    total=len(records),
                    desc=f"{model_name}+{dataset_name}",
                    unit='q',
                    initial=n_questions_done,
                )
                for qi in range(len(records)):
                    rec = records[qi]
                    qid = rec["id"]
                    question = rec['question']
                    answers = rec.get('answers', [])
                    gold_passages = rec.get("gold_passages", [])

                    cands = retriever.retrieve(question, top_k=80)
                    retrieved = cands[:5]
                    hard_negs = retriever.hard_negatives(question, gold_passages, answers, candidates=cands)
                    if retrieved:
                        max_score = retrieved[0]['score']
                    else:
                        max_score=0.0

                    for condition in conditions:
                        rng=random.Random(args.seed + qi)
                        try:
                            passages = build_condition(
                                gold_passages=gold_passages,
                                hard_negatives=hard_negs,
                                answers=answers,
                                condition=condition,
                                rng=rng,
                                generator=contra_gen,
                            )
                        except ValueError as exc:
                            n_build_err += len(policies)
                            if n_build_err <= 5 * len(policies):
                                tqdm.write(f"warn: build_condition failed q={qid} cond={condition}: {exc}")
                            continue
                        if passages == None:
                            n_qc_skip += len(policies)
                            continue

                        q_msgs = format_messages("P2", question, "")
                        q_text = " ".join(m["content"] for m in q_msgs)
                        q_overhead = len(model.tokenize(q_text.encode(), add_bos=True)) + 64
                        row_budget = max(0, args.n_ctx - q_overhead - args.max_tokens - 128)
                        if row_budget < 64:
                            tqdm.write(f"warn: context budget {row_budget} tokens for q={qid}")
                        context = build_context_string(passages, model, min(args.max_context_tokens, row_budget))
                        suf_text = "\n".join(p.get("text", "") for p in passages)
                        sufficient = contains_answer(suf_text, answers)

                        for policy in policies:
                            key=(qid,condition,policy)
                            if key in done:
                                n_resumed += 1
                                continue

                            #print("run:", qid, condition, policy)
                            if policy == "P3":
                                continue  #p3 is post-hoc, run metric/generate_p3.py instead

                            msgs = format_messages(policy, question, context)
                            result = generate_one(model, msgs, grammar, args.max_tokens)
                            is_error = result.get('error', False)
                            if is_error == True:
                                n_gen_err += 1
                                tqdm.write(f"error: generation failed q={qid} cond={condition} pol={policy}: {result['content'][:120]}")

                            det=detect(result['content'])

                            correct=(
                                not is_error
                                and det['decision'] == "answer"
                                and exact_match(det['answer'], answers)
                            )
                            if not is_error and det['decision'] == 'answer':
                                f1 = f1_score(det['answer'], answers)
                            else:
                                f1=0.0

                            if is_error:
                                dec_val="error"
                                ans_val=""
                                conf=-1
                                meth='error'
                            else:
                                dec_val=det["decision"]
                                ans_val=det['answer']
                                conf=det["confidence"]
                                meth=det['method']

                            if meth == 'json':
                                n_json += 1
                            elif meth == 'json_repair':
                                n_json_repair += 1
                            elif meth == 'string_match':
                                n_string_match += 1
                            elif meth == 'needs_judge':
                                n_needs_judge += 1

                            cap_hit = result.get('completion_tokens', 0) >= args.max_tokens
                            if cap_hit:
                                n_cap_hit += 1

                            row = {
                                "model": model_name,
                                "dataset": dataset_name,
                                "question_id": qid,
                                "question": question,
                                "condition": condition,
                                "policy": policy,
                                "sufficient": sufficient,
                                "raw_output": result['content'],
                                "decision": dec_val,
                                "answer": ans_val,
                                "confidence": conf,
                                'detection_method': meth,
                                "correct_em": correct,
                                'correct_f1': f1,
                                "prompt_tokens": result['prompt_tokens'],
                                "completion_tokens": result["completion_tokens"],
                                "wall_clock_s": result['wall_clock_s'],
                                'retrieval_score': max_score,
                                "cap_hit": cap_hit,
                                "error": is_error,
                            }
                            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                            fout.flush()
                            n_written += 1
                            if n_written % 100 == 0:
                                try:
                                    os.fsync(fout.fileno())
                                except OSError:
                                    pass

                    if qi >= n_questions_done:
                        pbar.update(1)
                    if qi % 5 == 0:
                        n_fb = n_string_match + n_needs_judge
                        pbar.set_postfix(written=n_written, resumed=n_resumed, fallback=n_fb)
                    if (qi + 1) % 100 == 0:
                        elapsed_m=(time.time() - t_start) / 60
                        n_fb=n_string_match + n_needs_judge
                        fb_rate=0.0
                        if n_written > 0:
                            fb_rate=100.0 * n_fb / n_written
                        tqdm.write(f"checkpoint q={qi+1}/{len(records)}: written={n_written} fallback={fb_rate:.1f}% elapsed={elapsed_m:.1f}m")
                        if not warned_fallback and n_written >= 200 and fb_rate > 5.0:
                            tqdm.write(f"warn: fallback >5% ({n_fb}/{n_written} rows), grammar may be broken")
                            warned_fallback=True
                        if not warned_cap and n_written > 0:
                            cap_rate=100.0 * n_cap_hit / n_written
                            if cap_rate > 5.0:
                                tqdm.write(f"warn: cap-hit {cap_rate:.1f}% ({n_cap_hit}/{n_written} rows)")
                                warned_cap=True

                pbar.set_postfix(written=n_written, resumed=n_resumed)
                pbar.close()

            total_done =n_resumed + n_written
            summary = f"done: {out_file} ({total_done} total: {n_written} written, {n_resumed} resumed)"
            if n_qc_skip:
                summary += f", {n_qc_skip} QC skipped"
            if n_build_err:
                summary += f", {n_build_err} build errors"
            if n_gen_err:
                summary += f", {n_gen_err} generation errors"
            if n_written > 0:
                n_fb=n_string_match + n_needs_judge
                fb_pct=100.0 * n_fb / n_written
                summary += f"\ndetection: json={n_json} repair={n_json_repair} string={n_string_match} needs_judge={n_needs_judge}"
                summary += f", fallback={fb_pct:.2f}% (this session)"
                if n_cap_hit > 0:
                    cap_pct=100.0 * n_cap_hit / n_written
                    summary += f", cap-hit={cap_pct:.2f}%"
            print(summary)
            if n_gen_err:
                print(f"warn: {n_gen_err} gen errors, decision='error', excluded from metrics")

        del model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["all"],
                        choices=[*model_gguf, "all"])
    parser.add_argument("--datasets", nargs="+", default=["all"],
                        choices=[*dataset_dirs, 'all'])
    parser.add_argument("--policies", nargs="+", default=['all'],
                        choices=[*pols, "all"])
    parser.add_argument('--conditions', nargs="+", default=["all"],
                        choices=[*conds, 'all'])
    parser.add_argument('--output', default='results/raw/')
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument('--max-tokens', type=int, default=512)
    parser.add_argument("--max-context-tokens", type=int, default=1500)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=seed)
    run_experiment(parser.parse_args())

if __name__ == "__main__":
    main()
