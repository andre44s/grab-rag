import json
import sys
from pathlib import Path

repo=Path(__file__).parent.parent
sys.path.insert(0, str(repo))

from src.utils import exact_match, f1_score
from analysis.generate_p3 import load_answers, files

f1s = [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]
cfs=[0, 20, 40, 50, 60, 80]

out_base = repo / "results" / "p3_sweep"
def_f1=0.8
def_cf = 50

def p3_decide(p1, cb, cf_t, f1_t):
    #p1 abstain already
    if p1["decision"] == "abstain":
        return "abstain", "", 0, "p1_abstain"
    #cb unsure so trust context
    if cb["decision"]=="abstain" or cb['confidence'] < cf_t:
        return p1['decision'], p1["answer"], p1["confidence"], "cb_unsure"
    f1 = f1_score(p1["answer"], [cb['answer']])
    if f1 >= f1_t:
        return p1["decision"], p1["answer"], p1['confidence'], "agree"
    return "abstain", "", 0, "conflict"

def build_cap_set(path):
    #p1-anchored: qids p1 got em-correct at q100
    cap=set()
    if not path.exists():
        return cap
    with path.open() as f:
        for line in f:
            r=json.loads(line)
            if r["policy"]=="P1" and r["condition"]=="Q100":
                if r["decision"]=="answer" and r.get("correct_em")==True:
                    cap.add(r["question_id"])
    return cap

def process_file(key, path, cf_t, f1_t, odir, cap):
    model,dataset=key
    if not path.exists():
        return []

    gold = load_answers(dataset)
    p1_by_qc={}
    cb_by_q = {}

    with path.open() as f:
        for line in f:
            r=json.loads(line)
            if r["policy"]=="P1":
                p1_by_qc[(r["question_id"], r["condition"])] = r
            if r['policy']=="CB" and r["condition"]=="Qclosed":
                cb_by_q[r["question_id"]]=r

    conds = ["Q100", "Q50", "Q0", "QC"]
    res=[]
    for (qid, cond), p1 in p1_by_qc.items():
        if cond not in conds:
            continue
        if qid not in cb_by_q:
            continue

        cb=cb_by_q[qid]
        dec, ans, conf, reason = p3_decide(p1, cb, cf_t, f1_t)

        gold_ans = gold.get(qid, [])
        if dec=="answer" and ans:
            em=exact_match(ans, gold_ans)
            f1 = f1_score(ans, gold_ans)
        else:
            em=False
            f1=0.0

        row = dict(p1)
        row["policy"]="P3"
        row["decision"] = dec
        row["answer"]=ans
        row["confidence"] = conf
        row["correct_em"]=em
        row["correct_f1"] = f1
        row['p3_cf_threshold']=cf_t
        row['p3_f1_threshold'] = f1_t
        row["detection_method"]="p3_posthoc"
        row["prompt_tokens"]=0
        row["completion_tokens"] = 0
        row["wall_clock_s"]=0.0
        row["_in_cap"]=qid in cap
        res.append(row)

    ofile = odir / f"{model}_{dataset}.jsonl"
    with ofile.open("w", encoding="utf-8") as f:
        for r in res:
            tmp={k:v for k,v in r.items() if not k.startswith("_")}
            f.write(json.dumps(tmp, ensure_ascii=False) + "\n")

    return res

def summarize(rows):
    qc_n=0; qc_ans=0; qc_em=0
    q0_n=0; q0_ans=0
    q100_n=0; q100_ans=0; q100_em=0
    q50_n = 0; q50_ans=0; q50_em=0
    #fac uses cap-set restriction
    fac100_n=0; fac100_ab=0
    fac50_n=0; fac50_ab=0

    for r in rows:
        cond = r["condition"]
        is_ans=r['decision']=="answer"
        is_ab = r["decision"]=="abstain"
        is_em = r.get("correct_em") == True
        in_cap = r.get("_in_cap", False)

        if cond=="QC":
            qc_n+=1
            if is_ans:
                qc_ans += 1
                if is_em:
                    qc_em+=1
        elif cond == "Q0":
            q0_n += 1
            if is_ans:
                q0_ans+=1
        elif cond=="Q100":
            q100_n+=1
            if is_ans:
                q100_ans += 1
            if is_em:
                q100_em += 1
            if in_cap:
                fac100_n += 1
                if is_ab: fac100_ab += 1
        elif cond=="Q50":
            q50_n += 1
            if is_ans:
                q50_ans += 1
            if is_em:
                q50_em+=1
            if in_cap:
                fac50_n += 1
                if is_ab: fac50_ab += 1

    return {
        "qc_hwsa": qc_ans/qc_n*100 if qc_n else 0,
        "q0_hwsa": q0_ans/q0_n*100 if q0_n else 0,
        'q100_em': q100_em/q100_n*100 if q100_n else 0,
        "q100_ans": q100_ans/q100_n*100 if q100_n else 0,
        "q50_em": q50_em/q50_n*100 if q50_n else 0,
        'q50_ans': q50_ans/q50_n*100 if q50_n else 0,
        "fac_q100": fac100_ab/fac100_n*100 if fac100_n else 0,
        "fac_q50": fac50_ab/fac50_n*100 if fac50_n else 0,
    }

def macro_avg(cell_summaries):
    #macro-avg across 6 cells (equal weight per cell), not pooled
    keys = ["qc_hwsa", "q0_hwsa", "q100_em", "q100_ans", "q50_em", "q50_ans", "fac_q100", "fac_q50"]
    n = len(cell_summaries)
    if n == 0:
        return {k: 0 for k in keys}
    return {k: sum(s[k] for s in cell_summaries) / n for k in keys}

def print_grid(title, mkey, results, lower_better=True):
    tag = "higher better"
    if lower_better:
        tag = "lower better"
    print(f"\n{title} ({tag})")
    header="f1/cf    " + "  ".join(f"{c:>6}" for c in cfs)
    print(header)
    for f1_t in f1s:
        row=[f"{f1_t:<7.1f} "]
        for cf_t in cfs:
            v = results[(cf_t, f1_t)][mkey]
            mark="*" if (cf_t==def_cf and f1_t==def_f1) else " "
            row.append(f"{v:>5.1f}%{mark}")
        print("  ".join(row))

def main():
    print("starting sweep")
    results={}

    #p1-anchored cap_set per cell, built once
    caps={k: build_cap_set(p) for k, p in files.items()}
    for k, c in caps.items():
        print(f"cap {k[0]}-{k[1]}: {len(c)}")

    #2d grid sweep
    for cf_t in cfs:
        for f1_t in f1s:
            tag=f"cf{cf_t:02d}_f1{int(f1_t*10):02d}"
            odir = out_base / tag
            odir.mkdir(parents=True, exist_ok=True)

            cell_summaries=[]
            for key, path in files.items():
                tmp = process_file(key, path, cf_t, f1_t, odir, caps[key])
                if tmp:
                    cell_summaries.append(summarize(tmp))

            results[(cf_t, f1_t)]=macro_avg(cell_summaries)

    print(f"sweep done, default cf {def_cf} f1 {def_f1} marked star")

    print_grid("qc hwsa, answer rate at qc", "qc_hwsa", results, lower_better=True)
    print_grid("q0 hwsa, answer rate at q0", "q0_hwsa", results, lower_better=True)
    print_grid("q100 em, utility at q100", "q100_em", results, lower_better=False)
    print_grid("q100 ans rate, coverage at q100", "q100_ans", results, lower_better=False)
    print_grid("q50 em, utility at q50", "q50_em", results, lower_better=False)
    print_grid("q50 ans rate, coverage at q50", "q50_ans", results, lower_better=False)
    print_grid("fac@q100 (p1-anchored cap)", "fac_q100", results, lower_better=True)
    print_grid("fac@q50 (p1-anchored cap)", "fac_q50", results, lower_better=True)

    #csv for figures later
    csv_path=out_base / "sweep_summary.csv"
    with csv_path.open("w") as f:
        f.write("cf_threshold,f1_threshold,qc_hwsa,q0_hwsa,q100_em,q100_ans,q50_em,q50_ans,fac_q100,fac_q50\n")
        for (cf_t, f1_t), m in results.items():
            f.write(f"{cf_t},{f1_t:.1f},{m['qc_hwsa']:.2f},{m['q0_hwsa']:.2f},"
                    f"{m['q100_em']:.2f},{m['q100_ans']:.2f},{m['q50_em']:.2f},{m['q50_ans']:.2f},"
                    f"{m['fac_q100']:.2f},{m['fac_q50']:.2f}\n")

    print(f"csv saved to {csv_path}")

if __name__=="__main__":
    main()
