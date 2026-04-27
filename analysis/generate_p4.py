import json, sys, time, random
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

repo=Path(__file__).parent.parent
sys.path.insert(0, str(repo))
results_base=repo / "results"
nli_dir=repo / "models" / "nli"

from src.context_quality import ContradictoryGenerator, build_condition
from src.retrieval import Retriever
from src.utils import exact_match, f1_score

seed=42
cf_thresh=50
conditions=["Q100", "Q50", "Q0", "QC"]

files = {
    ("phi", "nq"): results_base / "phi-nq/phi_nq.jsonl",
    ("phi", "hpqa"): results_base / "phi-hpqa/phi_hpqa.jsonl",
    ("llama", "nq"): results_base / "llama-nq/llama_nq.jsonl",
    ("llama", "hpqa"): results_base / "llama-hpqa/llama_hpqa.jsonl",
    ("qwen", "nq"): results_base / "qwen-nq/qwen_nq.jsonl",
    ("qwen", "hpqa"): results_base / "qwen-hpqa/qwen_hpqa.jsonl",
}

eval_dirs = {
    "nq": repo / "data" / "natural_questions" / "eval.jsonl",
    "hpqa": repo / "data" / "hotpotqa" / "eval.jsonl",
}

def load_records(dataset):
    records=[]
    with eval_dirs[dataset].open(encoding='utf-8') as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records

def load_nli_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading nli model on {device}")
    tok = AutoTokenizer.from_pretrained(str(nli_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(nli_dir))
    model.eval()
    model.to(device)
    lbls = {v.lower() for v in model.config.id2label.values()}
    assert "contradiction" in lbls, f"unexpected label set: {model.config.id2label}"
    print(f"nli labels: {model.config.id2label}")
    return tok, model, device

def nli_infer(passages, hypothesis, tok, model, device, max_len=512):
    if not passages:
        return [], []
    premises=[]
    for p in passages:
        title=p.get('title', '').strip()
        text=p.get('text', '').strip()
        if title:
            premises.append(f"{title}: {text}")
        else:
            premises.append(text)
    inputs = tok(
        premises,
        [hypothesis] * len(premises),
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors='pt',
    ).to(device)
    with torch.no_grad():
        out = model(**inputs)
        probs = torch.softmax(out.logits, dim=-1).cpu().tolist()
    id2label=model.config.id2label
    n_cls=out.logits.shape[-1]
    lbl_order=[]
    for i in range(n_cls):
        lbl_order.append(id2label.get(i, id2label.get(str(i), f"cls{i}")).lower())
    pred_ids=out.logits.argmax(dim=-1).tolist()
    labels=[lbl_order[i] for i in pred_ids]
    prob_dicts=[]
    for row in range(len(probs)):
        d={}
        for j in range(n_cls):
            d[lbl_order[j]]=probs[row][j]
        prob_dicts.append(d)
    return labels, prob_dicts

def p4_decision(p1_row, cb_row, passages, question, tok, model, device):
    if p1_row['decision'] == 'abstain':
        return 'abstain', '', 0, 'p1_abstain', [], []
    #cb unsure, same as p3
    if cb_row == None or cb_row['decision'] == 'abstain' or cb_row['confidence'] < cf_thresh:
        return p1_row['decision'], p1_row['answer'], p1_row['confidence'], 'cb_unsure', [], []
    cb_ans=cb_row['answer']
    hypothesis=f"The answer to '{question}' is {cb_ans}."
    labels, prob_dicts=nli_infer(passages, hypothesis, tok, model, device)
    if any(lbl == 'contradiction' for lbl in labels):
        return 'abstain', '', 0, 'nli_contradiction', labels, prob_dicts
    return p1_row['decision'], p1_row['answer'], p1_row['confidence'], 'nli_clear', labels, prob_dicts

def process_file(key, path, records, retriever, contra_gen, gold_answers, tok, nli_model, device):
    model_name, dataset=key
    if not path.exists():
        print(f"skip {path}: not found")
        return
    p1_rows={}
    cb_rows={}
    existing_p4=set()
    with path.open(encoding='utf-8') as f:
        for line in f:
            try:
                r=json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get('error'):
                continue
            pol=r.get('policy', '')
            cond=r.get('condition', '')
            qid=r['question_id']
            if pol == 'P4':
                existing_p4.add((qid, cond))
            elif pol == 'P1':
                p1_rows[(qid, cond)]=r
            elif pol == 'CB' and cond == 'Qclosed':
                cb_rows[qid]=r
    if not p1_rows:
        print(f"{model_name}+{dataset}: no p1 rows, skipping")
        return
    if not cb_rows:
        print(f"{model_name}+{dataset}: no cb rows, run cb condition first")
        return
    qid_to_qi={rec['id']: qi for qi, rec in enumerate(records)}
    new_rows=[]
    n_new=n_skip=n_p1_abstain=n_cb_unsure=n_nli_contra=n_nli_clear=n_build_err=n_qc_skip=0
    for (qid, condition), p1_row in p1_rows.items():
        if condition not in conditions:
            continue
        if (qid, condition) in existing_p4:
            n_skip += 1
            continue
        qi=qid_to_qi.get(qid)
        if qi == None:
            continue
        rec=records[qi]
        gold_passages=rec.get('gold_passages', [])
        answers=rec.get('answers', [])
        question=rec['question']
        #same passages and rng as runner.py
        cands=retriever.retrieve(question, top_k=80)
        hard_negs=retriever.hard_negatives(question, gold_passages, answers, candidates=cands)
        rng=random.Random(seed + qi)
        try:
            passages=build_condition(
                gold_passages=gold_passages,
                hard_negatives=hard_negs,
                answers=answers,
                condition=condition,
                rng=rng,
                generator=contra_gen,
            )
        except Exception:
            n_build_err += 1
            continue
        if passages == None:
            n_qc_skip += 1
            continue
        cb_row=cb_rows.get(qid)
        t0=time.perf_counter()
        res=p4_decision(p1_row, cb_row, passages, question, tok, nli_model, device)
        elapsed=time.perf_counter() - t0
        dec=res[0]; ans=res[1]; conf=res[2]; reason=res[3]; nli_lbls=res[4]; nli_probs=res[5]
        if reason == 'p1_abstain': n_p1_abstain += 1
        elif reason == 'cb_unsure': n_cb_unsure += 1
        elif reason == 'nli_contradiction': n_nli_contra += 1
        elif reason == 'nli_clear': n_nli_clear += 1
        ans_list=gold_answers.get(qid, [])
        em=False
        f1=0.0
        if dec == 'answer' and ans:
            em=exact_match(ans, ans_list)
            f1=f1_score(ans, ans_list)
        p4_row=dict(p1_row)
        p4_row.update({
            'policy': 'P4',
            'decision': dec,
            'answer': ans,
            'confidence': conf,
            'correct_em': em,
            'correct_f1': f1,
            'raw_output': json.dumps({
                'decision': dec, 'answer': ans, 'confidence': conf,
                'reasoning': f'p4 {reason}',
                'nli_labels': nli_lbls, 'nli_probs': nli_probs,
            }),
            'detection_method': 'p4_nli_posthoc',
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'wall_clock_s': round(elapsed, 4),
        })
        new_rows.append(p4_row)
        n_new += 1
    with path.open('a', encoding='utf-8') as f:
        for r in new_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{model_name}+{dataset}: wrote {n_new} p4 rows, skipped {n_skip}")
    print(f"p1_abstain={n_p1_abstain}  cb_unsure={n_cb_unsure}  nli_contradiction={n_nli_contra}  nli_clear={n_nli_clear}  build_err={n_build_err}  qc_skip={n_qc_skip}")

def main():
    tok, nli_model, device=load_nli_model()
    #build retriever and generator once per dataset
    cache={}
    for key, path in files.items():
        _, dataset=key
        if dataset not in cache:
            print(f"building retriever for {dataset}")
            records=load_records(dataset)
            retriever=Retriever.from_records(records)
            contra_gen=ContradictoryGenerator.from_records(records)
            gold_answers={rec['id']: rec.get('answers', []) for rec in records}
            cache[dataset]=(records, retriever, contra_gen, gold_answers)
        tmp=cache[dataset]
        print(f"processing {key[0]}+{dataset}")
        process_file(key, path, tmp[0], tmp[1], tmp[2], tmp[3], tok, nli_model, device)
    print("done")

if __name__ == "__main__":
    main()
