import json
import sys
from pathlib import Path

repo=Path(__file__).parent.parent
sys.path.insert(0, str(repo))
results_base = repo / "results"

from src.utils import normalize, exact_match, f1_score

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

cf_thresh=50
f1_thresh = 0.8

def load_answers(dataset):
    answers={}
    with eval_dirs[dataset].open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            answers[rec["id"]] = rec["answers"]
    return answers

def p3_decision(p1_row, cb_row):
    #p1 abstain already
    if p1_row["decision"] == "abstain":
        return "abstain", "", 0, "p1_abstain"
    #cb unsure so trust context
    if cb_row["decision"]=="abstain" or cb_row['confidence'] < cf_thresh:
        return p1_row["decision"], p1_row["answer"], p1_row["confidence"], "cb_unsure"
    f1 = f1_score(p1_row["answer"], [cb_row["answer"]])
    if f1 >= f1_thresh:
        return p1_row["decision"], p1_row["answer"], p1_row['confidence'], "agree"
    return "abstain", "", 0, "conflict"

def process_file(key, path):
    model,dataset=key
    if not path.exists():
        print(f"skip {path}, not found")
        return
    gold = load_answers(dataset)
    p1_by_qc = {}
    cb_by_q={}
    existing_p3 = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["policy"]=="P3":
                existing_p3.add((r["question_id"], r["condition"]))
                continue
            if r["policy"]=="P1":
                p1_by_qc[(r["question_id"], r["condition"])]=r
            if r['policy']=="CB" and r["condition"]=="Qclosed":
                cb_by_q[r["question_id"]]=r
    if not cb_by_q:
        print(f"{model}+{dataset}: no CB rows, run --policies CB --conditions Qclosed first")
        return
    conds = ["Q100", "Q50", "Q0", "QC"]
    new_rows=[]
    n_new=n_skip=n_p1_abstain=n_cb_unsure=n_agree=n_conflict=n_no_cb=0
    for (qid, cond), p1_row in p1_by_qc.items():
        if cond not in conds:
            continue
        if (qid, cond) in existing_p3:
            n_skip += 1
            continue
        if qid not in cb_by_q:
            n_no_cb += 1
            continue
        cb_row=cb_by_q[qid]
        dec, ans, conf, reason = p3_decision(p1_row, cb_row)
        if reason=="p1_abstain": n_p1_abstain += 1
        elif reason=="cb_unsure": n_cb_unsure += 1
        elif reason=="agree": n_agree += 1
        elif reason=="conflict": n_conflict+=1
        ans_list=gold.get(qid, [])
        if dec=="answer" and ans:
            em=exact_match(ans, ans_list)
            f1 = f1_score(ans, ans_list)
        else:
            em=False
            f1 = 0.0
        p3_row = dict(p1_row)
        p3_row["policy"]="P3"
        p3_row["decision"] = dec
        p3_row["answer"]=ans
        p3_row["confidence"] = conf
        p3_row["correct_em"]=em
        p3_row["correct_f1"]=f1
        p3_row["raw_output"]=json.dumps({
            "decision": dec, "answer": ans,
            "confidence": conf, "reasoning": f"p3 {reason}",
        })
        p3_row["detection_method"]="p3_posthoc"
        p3_row["prompt_tokens"]=0
        p3_row["completion_tokens"]=0
        p3_row["wall_clock_s"]=0.0
        new_rows.append(p3_row)
        n_new += 1
    with path.open("a", encoding="utf-8") as f:
        for r in new_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{model}+{dataset}: wrote {n_new} new P3 rows, skipped {n_skip} existing")
    print(f"p1_abstain={n_p1_abstain}  cb_unsure={n_cb_unsure}  agree={n_agree}  conflict={n_conflict}  no_cb={n_no_cb}")

def main():
    for key, path in files.items():
        process_file(key, path)
    print("done")

if __name__ == "__main__":
    main()
