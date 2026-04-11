import argparse
import json
import random
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import contains_answer

seed = 42
n_eval=500
ddir = Path(__file__).parent.parent / "data"

max_pass_chars = 2000 #skip passages longer than this
max_aliases = 10 #too many aliases inflates em

def filter_record(record):
    #skip anything out of bounds
    if len(record["answers"]) > max_aliases:
        return None
    if any(len(p.get("text", "")) > max_pass_chars for p in record["gold_passages"]):
        return None
    ctx = " ".join(p.get('text', '') for p in record.get('gold_passages', []))
    if not contains_answer(ctx, record.get("answers", [])):
        return None
    return record

def nq_passage(tokens, start, end):
    return " ".join(
        tokens["token"][i]
        for i in range(start, end)
        if not tokens["is_html"][i]
    ).strip()

def process_nq(ex):
    answers, passage = [],""
    n_ann = len(ex["annotations"]["id"])
    for i in range(n_ann):
        sa_texts = ex["annotations"]["short_answers"][i].get("text", [])
        for txt in sa_texts:
            if txt:
                answers.append(txt)
        la = ex["annotations"]["long_answer"][i]
        if not passage and la.get("candidate_index", -1) >= 0:
            passage = nq_passage(
                ex["document"]["tokens"], la["start_token"], la["end_token"]
            )
    if passage:
        gold_passages = [{"title": ex["document"]["title"], "text": passage}]
    else:
        gold_passages = []
    return {
        "id": str(ex["id"]),
        "question": ex["question"]["text"],
        "answers": list(dict.fromkeys(answers)),
        "gold_passages": gold_passages,
        "dataset": "natural_questions",
    }

def process_hotpotqa(ex):
    supporting = set(ex["supporting_facts"]["title"])
    return {
        "id": ex["id"],
        "question": ex["question"],
        "answers": [ex["answer"]],
        "gold_passages": [
            {"title": t, "text": " ".join(s)}
            for t, s in zip(ex["context"]["title"], ex["context"]["sentences"])
            if t in supporting
        ],
        "dataset": "hotpotqa",
    }

configs = {
    "nq": {"label": "natural_questions", "hf_path": "natural_questions", "hf_config": None, "split": "validation", "processor": process_nq, "keep": lambda r: bool(r["answers"] and r["gold_passages"]), "n": n_eval},
    "hpqa": {"label": "hotpotqa", "hf_path": "hotpot_qa", "hf_config": "distractor", "split": "validation", "processor": process_hotpotqa, "keep": lambda r: bool(r["answers"] and r["gold_passages"] and r["answers"][0].strip().lower() not in ("yes", "no")), "n": n_eval},
}

def download_one(name, cfg, force=False):
    out_dir=ddir / cfg["label"]
    out_file = out_dir / "eval.jsonl"
    tmp_file=out_dir / "eval.jsonl.tmp"

    if out_file.exists() and not force:
        n = sum(1 for _ in out_file.open(encoding='utf-8'))
        print(f"{name} already done {n} records")
        return True

    out_dir.mkdir(parents=True, exist_ok=True)

    load_kw={"path": cfg["hf_path"],"split": cfg["split"]}
    if cfg["hf_config"]:
        load_kw["name"] = cfg["hf_config"]

    print(f"{name} fetching from huggingface")
    try:
        ds = load_dataset(**load_kw)
    except Exception as e:
        print(f"{name} fetch failed {e}")
        return False

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    records=[]
    seen_ids=set()

    for idx in indices:
        if len(records) >= cfg["n"]:
            break
        raw = ds[idx]
        rec = cfg["processor"](raw)
        if not cfg["keep"](rec):
            continue
        if rec["id"] in seen_ids:
            continue

        rec = filter_record(rec)
        if rec is None:
            continue

        seen_ids.add(rec["id"])
        records.append(rec)

    if len(records) < cfg["n"]:
        print(f"{name} failed only {len(records)} records")
        return False

    records = sorted(records, key=lambda r: r["id"])

    with tmp_file.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp_file.replace(out_file)

    n_written= sum(1 for _ in out_file.open(encoding="utf-8"))
    if n_written != len(records):
        print(f"{name} write failed expected {len(records)} got {n_written}")
        return False

    print(f"{name} {len(records)} records {out_file}")
    return True

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", choices=[*configs, "all"], default=["all"])
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    targets = list(configs) if "all" in args.datasets else args.datasets
    print(f"downloading {' '.join(targets)}")

    ok=True
    for name in targets:
        ok &= download_one(name, configs[name], force=args.force)

    if not ok:
        print("one or more datasets failed")
        sys.exit(1)
    print("all datasets ready")

if __name__ == "__main__":
    main()
