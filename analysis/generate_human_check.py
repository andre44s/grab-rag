import csv
import json
import random
import sys
from pathlib import Path

script_dir=Path(__file__).parent
repo_dir = script_dir.parent
sys.path.insert(0, str(repo_dir))
from src.context_quality import gliner_labels, gliner_label_map, locate_answer
from src.utils import contains_answer
results_dir=repo_dir / "results"
data_dir = repo_dir / "data"
odir = results_dir / "human_check"
seed=42
sample_seed=100
datasets = ["nq", "hpqa"]
dataset_dirs = {"nq": "natural_questions", "hpqa": "hotpotqa"}

#gliner labels to swap type
ner_to_swap = {
    "PERSON": "PERSON",
    "GPE": "LOCATION", "LOC": "LOCATION",
    "ORG": "ORG", "NORP": "ORG",
    "EVENT": "OTHER", "WORK_OF_ART": "OTHER",
}

def load_records(dataset_name):
    path = data_dir / dataset_dirs[dataset_name] / "eval.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def answers_str(answers):
    return " | ".join(f'"{a}"' for a in answers[:5])

def classify_swap_type(nlp, span):
    if span and span.strip():
        ents = nlp.predict_entities(span.strip(), gliner_labels, threshold=0.5)
        lbl = None
        if ents:
            lbl = gliner_label_map.get(ents[0]["label"])
        if lbl and lbl in ner_to_swap:
            return ner_to_swap[lbl]
    return "OTHER"

def gen_qc_artifact_csv(recs, cgens):
    rng=random.Random(sample_seed + 1)
    out,n=[],0
    for ds in datasets:
        if ds == "nq":
            dsl = "NQ"
        else:
            dsl = "HPQA"
        candidates=[]
        for qi, rec in enumerate(recs[ds]):
            gp = rec.get("gold_passages", [])
            ans = rec.get("answers", [])
            swap_rng = random.Random(seed + qi)
            found = None
            for p in gp:
                text=p.get("text", "")
                try:
                    swapped, ok, replacement = cgens[ds].do_swap(text, ans, swap_rng)
                except Exception:
                    continue
                if ok and not contains_answer(swapped, ans):
                    title = p.get("title", "")
                    title_sw, _ = cgens[ds].swap(title, ans, swap_rng)
                    if contains_answer(title_sw, ans):
                        title_sw=title
                    if contains_answer(title_sw + " " + swapped, ans):
                        continue
                    span=locate_answer(text, ans)
                    swap_from = ""
                    if span:
                        swap_from = span[0]
                    found = {
                        "passage": p,
                        "swapped_text": swapped,
                        "swapped_title": title_sw,
                        "swap_from": swap_from,
                        "swap_to": replacement or "",
                    }
                    break
            if found:
                candidates.append((qi, rec, found))

        for qi, rec, info in rng.sample(candidates, min(100, len(candidates))):
            n += 1
            ans = rec.get("answers", [])
            stype = classify_swap_type(cgens[ds].nlp, info["swap_from"])
            out.append({
                "sample_id": f"{dsl}-QC-{n:03d}",
                "dataset": dsl, "question_id": rec["id"], "question": rec["question"],
                "gold_answers": answers_str(ans),
                "original_title": info["passage"].get("title", ""),
                "original_text": info["passage"].get("text", ""),
                "swapped_title": info["swapped_title"],
                "swapped_text": info["swapped_text"],
                "swap_from": info["swap_from"],
                "swap_to": info["swap_to"],
                "swap_type": stype, "notes": "",
            })
    return out

def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)

def main():
    odir.mkdir(parents=True, exist_ok=True)
    from src.context_quality import ContradictoryGenerator
    recs={ds: load_records(ds) for ds in datasets}
    cgens = {}
    for ds in datasets:
        print(f"building generator: {ds}")
        cgens[ds] = ContradictoryGenerator.from_records(recs[ds])
    qc_rows=gen_qc_artifact_csv(recs, cgens)
    qc_fields = ["sample_id", "dataset", "question_id", "question", "gold_answers",
                 "original_title", "original_text", "swapped_title", "swapped_text",
                 "swap_from", "swap_to", "swap_type", "notes"]
    out = odir / "audit_qc_artifacts.csv"
    write_csv(out, qc_rows, qc_fields)
    print(f"saved {len(qc_rows)} rows to {out}")

if __name__ == "__main__":
    main()
