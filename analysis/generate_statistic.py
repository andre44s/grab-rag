import csv, json, sys
import contextlib, random
import statistics
from pathlib import Path
from collections import defaultdict
import numpy as np

repo=Path(__file__).parent.parent
results_base=repo / "results"
data_base=repo / "data"
audit_path=results_base / "human_check" / "audit_qc_artifacts.csv"

sys.path.insert(0, str(repo))

files = {
    ("phi", "nq"): results_base / "phi-nq/phi_nq.jsonl",
    ("phi", "hpqa"): results_base / "phi-hpqa/phi_hpqa.jsonl",
    ("llama", "nq"): results_base / "llama-nq/llama_nq.jsonl",
    ("llama", "hpqa"): results_base / "llama-hpqa/llama_hpqa.jsonl",
    ("qwen", "nq"): results_base / "qwen-nq/qwen_nq.jsonl",
    ("qwen", "hpqa"): results_base / "qwen-hpqa/qwen_hpqa.jsonl",
}

eval_dirs = {
    "nq": data_base / "natural_questions" / "eval.jsonl",
    "hpqa": data_base / "hotpotqa" / "eval.jsonl",
}

models = ["phi", "llama", "qwen"]
datasets = ["nq", "hpqa"]
policies = ["P0", "P1", "P2", "P3", "P4"]
conditions=["Q100", "Q50", "Q0", "QC"]

n_boot=10000
rng=np.random.default_rng(42)
seed=42
n_bins=10

mlabels={"phi": "Phi-4-mini (3.8B)", "llama": "Llama-3.1-8B", "qwen": "Qwen2.5-7B"}
dslabels = {"nq": "NQ", "hpqa": "HotpotQA"}

class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, data):
        for s in self.streams: s.write(data)
    def flush(self):
        for s in self.streams: s.flush()

def load_usable_qids():
    usable = {"nq": set(), "hpqa": set()}
    with open(audit_path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            qual = row["QUALITY"].strip()
            if qual.upper() in ("FLUENT", "BORDERLINE"):
                ds_raw = row["dataset"].strip()
                if ds_raw == "NQ":
                    ds_key = "nq"
                else:
                    ds_key = "hpqa"
                usable[ds_key].add(row["question_id"].strip())
    return usable

def load_all_rows():
    rows, n_skip = [], 0
    for model in models:
        for dataset in datasets:
            with open(files[(model, dataset)], encoding='utf-8') as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        if not r.get("error"):
                            rows.append(r)
                        else:
                            n_skip += 1
                    except json.JSONDecodeError:
                        n_skip += 1
    #print(f"loaded {len(rows)} rows, skipped {n_skip}")
    return rows

def load_eval(dataset):
    records=[]
    with open(eval_dirs[dataset], encoding='utf-8') as f:
        for line in f:
            try: records.append(json.loads(line))
            except json.JSONDecodeError: pass
    return records

def build_index(rows):
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    for r in rows:
        data[r["model"]][r["dataset"]][r["question_id"]][r["condition"]][r["policy"]] = r
    return data

def bs_mean(arr, n=n_boot):
    arr = np.asarray(arr, dtype=float)
    if len(arr) == 0:
        return np.nan, (np.nan, np.nan)
    boot = np.array([np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n)])
    return float(np.mean(arr)), (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))

def bs_delta_paired(x, y, n=n_boot):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    assert len(x) == len(y)
    if len(x) == 0:
        return np.nan, (np.nan, np.nan)
    n_obs = len(x)
    boot=np.zeros(n)
    i=0
    while i < n:
        idx = rng.integers(0, n_obs, size=n_obs)
        boot[i] = np.mean(y[idx]) - np.mean(x[idx])
        i += 1
    delta = float(np.mean(y) - np.mean(x))
    return delta, (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))

def bs_diff_independent(a, b, n=n_boot):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    boot=[]
    for _ in range(n):
        boot.append(
            np.mean(rng.choice(a, size=len(a), replace=True))
            - np.mean(rng.choice(b, size=len(b), replace=True))
        )
    boot = np.array(boot)
    diff = float(np.mean(a) - np.mean(b))
    return diff, (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))

def compute_aurc(is_correct, confidence):
    conf = np.asarray(confidence, dtype=float)
    correct = np.asarray(is_correct, dtype=float)
    n = len(correct)
    if n == 0:
        return np.nan
    order = np.argsort(-conf)
    correct_sorted = correct[order]
    cum = np.cumsum(correct_sorted)
    risk = 1.0 - cum / np.arange(1, n+1)
    return float(np.mean(risk))

def pct(v): return f"{v*100:.1f}%"
def pp(v): return f"{v*100:+.1f}pp"
def ci_pct(lo, hi): return f"[{lo*100:.1f}, {hi*100:.1f}]"
def ci_pp(lo, hi): return f"[{lo*100:+.1f}, {hi*100:+.1f}]"

def compute_calibration(rows):
    groups = defaultdict(list)
    for r in rows:
        cond = r["condition"]
        pol = r["policy"]
        conf = max(0.0, float(r.get("confidence", 0)))
        if cond in ("Q0", "QC"):
            correct = int(r["decision"] == "abstain")
        else:
            correct = int(r["decision"] == "answer" and bool(r.get("correct_em", False)))
        groups[(pol, cond)].append((conf, correct))
    out = {}
    for key, pairs in groups.items():
        confs = np.array([c for c, _ in pairs])
        correct = np.array([v for _, v in pairs], dtype=float)
        bins, ece = [], 0.0
        n_total = len(pairs)
        for b in range(n_bins):
            lo = b * 10.0
            hi = (b + 1) * 10.0
            if b < n_bins - 1:
                mask = (confs >= lo) & (confs < hi)
            else:
                mask = confs >= lo
            n_b = int(mask.sum())
            mid = lo + 5.0
            if n_b == 0:
                bins.append({"lo": lo, "hi": hi, "n": 0, "acc": None, "mid": mid})
                continue
            acc = float(correct[mask].mean())
            bins.append({"lo": lo, "hi": hi, "n": n_b, "acc": acc, "mid": mid})
            ece += (n_b / n_total) * abs(acc - mid / 100.0)
        out[key] = {"bins": bins, "ece": ece,
                    "mean_conf": float(confs.mean()), "mean_correct": float(correct.mean()), "n": n_total}
    return out

def build_swap_map(records):
    from src.context_quality import ContradictoryGenerator
    from src.utils import contains_answer
    gen = ContradictoryGenerator.from_records(records)
    swap_map={}
    for qi in range(len(records)):
        rec = records[qi]
        qid = rec["id"]
        passages = rec.get("gold_passages", [])
        answers = rec.get("answers", [])
        r = random.Random(seed + qi)
        planted = None
        for p in passages:
            text = p.get("text", "")
            swapped, ok, replacement = gen.do_swap(text, answers, r)
            if ok and not contains_answer(swapped, answers):
                planted = replacement
                break
        swap_map[qid] = planted
    return swap_map

def is_echoing(model_ans, planted):
    if not planted or not model_ans:
        return False
    from src.utils import normalize
    m = normalize(model_ans).strip()
    p = normalize(planted).strip()
    if not m or not p:
        return False
    return m == p or m in p or p in m

def comparability_test(data):
    result = {}
    for ds in datasets:
        elig_em, nonel_em = [], []
        elig_f1, nonel_f1 = [], []
        elig_n, nonel_n = set(), set()
        for m in models:
            d = data[m][ds]
            qc_qids = {qid for qid, c in d.items() if "QC" in c}
            non_qids = set(d.keys()) - qc_qids
            elig_n |= qc_qids
            nonel_n |= non_qids
            for qid in qc_qids:
                for pol in policies:
                    r = d[qid].get("Q100", {}).get(pol)
                    if r:
                        elig_em.append(float(r["correct_em"]))
                        elig_f1.append(float(r["correct_f1"]))
            for qid in non_qids:
                for pol in policies:
                    r = d[qid].get("Q100", {}).get(pol)
                    if r:
                        nonel_em.append(float(r["correct_em"]))
                        nonel_f1.append(float(r["correct_f1"]))
        em_diff, em_ci = bs_diff_independent(elig_em, nonel_em)
        f1_diff, f1_ci = bs_diff_independent(elig_f1, nonel_f1)
        result[ds] = {
            "em_ok": em_ci[0] <= 0 <= em_ci[1], "f1_ok": f1_ci[0] <= 0 <= f1_ci[1],
            "em_diff": em_diff, "em_ci": em_ci, "f1_diff": f1_diff, "f1_ci": f1_ci,
            "elig_n_q": len(elig_n), "nonel_n_q": len(nonel_n),
            "em_elig": bs_mean(elig_em), "em_nonel": bs_mean(nonel_em),
            "f1_elig": bs_mean(elig_f1), "f1_nonel": bs_mean(nonel_f1),
        }
    return result

def compute_hwsa(data, qid_filter=None):
    hwsa = {}
    for m in models:
        for ds in datasets:
            d = data[m][ds]
            qc_qids = sorted(qid for qid, c in d.items() if "QC" in c)
            if qid_filter is not None:
                allowed = qid_filter.get(ds, set())
                qc_qids = [q for q in qc_qids if q in allowed]
            for pol in policies:
                q0_v, qc_v = [], []
                for qid in qc_qids:
                    r0 = d[qid].get("Q0", {}).get(pol)
                    rc = d[qid].get("QC", {}).get(pol)
                    if r0 and rc:
                        q0_v.append(1 if r0["decision"] == "answer" else 0)
                        qc_v.append(1 if rc["decision"] == "answer" else 0)
                q0_m, q0_ci = bs_mean(q0_v)
                qc_m, qc_ci = bs_mean(qc_v)
                delta, d_ci = bs_delta_paired(q0_v, qc_v)
                excl0 = not (d_ci[0] <= 0 <= d_ci[1])
                hwsa[(m, ds, pol)] = dict(
                    n=len(q0_v), hwsa_q0=q0_m, ci_q0=q0_ci,
                    hwsa_qc=qc_m, ci_qc=qc_ci, delta=delta, ci_delta=d_ci, excl0=excl0,
                )
    for k, v in hwsa.items():
        if v["excl0"] == False:
            print(f"warn: hwsa ci includes 0 for {k}: delta={pp(v['delta'])} {ci_pp(*v['ci_delta'])}")
    return hwsa

def compute_fac(data):
    #cap: p1-correct at q100
    fac={}
    for m in models:
        for ds in datasets:
            d = data[m][ds]
            cap = set()
            for qid in d:
                r100 = d[qid].get("Q100", {}).get("P1")
                if r100 == None:
                    continue
                if r100.get("decision") != "answer":
                    continue
                if r100.get("correct_em") != True:
                    continue
                cap.add(qid)
            for pol in policies:
                for cond in ["Q100", "Q50"]:
                    vals=[]
                    for qid, cd in d.items():
                        if qid not in cap:
                            continue
                        r = cd.get(cond, {}).get(pol)
                        if r:
                            v = 1 if r["decision"] == "abstain" else 0
                            vals.append(v)
                    fac[(m, ds, pol, cond)] = bs_mean(vals)
    return fac

def compute_ans_rate(data):
    #answer rate at q100/q50
    out={}
    for m in models:
        for ds in datasets:
            d = data[m][ds]
            for pol in policies:
                for cond in ["Q100", "Q50"]:
                    vals=[]
                    for qid, cd in d.items():
                        r = cd.get(cond, {}).get(pol)
                        if r:
                            vals.append(1 if r["decision"]=="answer" else 0)
                    out[(m, ds, pol, cond)]=bs_mean(vals)
    return out

def compute_aurc_all(data):
    aurc_res = {}
    for m in models:
        for ds in datasets:
            d = data[m][ds]
            for pol in policies:
                is_correct, conf_all = [], []
                for qid, cd in d.items():
                    for cond in conditions:
                        r = cd.get(cond, {}).get(pol)
                        if r is None:
                            continue
                        if cond in ("Q0", "QC"):
                            is_correct.append(1 if r["decision"] == "abstain" else 0)
                        else:
                            if r["decision"] == "answer" and r.get("correct_em", False):
                                is_correct.append(1)
                            else:
                                is_correct.append(0)
                        conf_all.append(max(0.0, float(r.get("confidence", 0))))
                aurc_res[(m, ds, pol)] = compute_aurc(is_correct, conf_all)
    return aurc_res

def print_hwsa_robustness(hwsa_full, hwsa_filt, usable):
    print(f"\nfiltered-qc robustness (fluent+borderline only: NQ={len(usable['nq'])} HPQA={len(usable['hpqa'])})")
    print(f"{'model':<6} {'ds':<5} {'pol':<4}  "
          f"{'full n':>6}  {'full HwSA@QC':>13}  "
          f"{'filt n':>6}  {'filt HwSA@QC':>13}  "
          f"{'shift':>8}  sign_ok")
    all_sign_ok = True
    for m in models:
        for ds in datasets:
            for pol in policies:
                f = hwsa_full[(m, ds, pol)]
                fi = hwsa_filt.get((m, ds, pol))
                if fi is None:
                    continue
                shift = (fi["hwsa_qc"] - f["hwsa_qc"]) * 100
                sign_ok = (f["delta"] > 0) == (fi["delta"] > 0)
                if not sign_ok:
                    all_sign_ok = False
                sign_str = "NO"
                if sign_ok:
                    sign_str = "YES"
                print(f"{m:<6} {ds:<5} {pol:<4}  "
                      f"{f['n']:>6}  {f['hwsa_qc']*100:>6.1f}% [{f['ci_delta'][0]*100:+.1f},{f['ci_delta'][1]*100:+.1f}]  "
                      f"{fi['n']:>6}  {fi['hwsa_qc']*100:>6.1f}% [{fi['ci_delta'][0]*100:+.1f},{fi['ci_delta'][1]*100:+.1f}]  "
                      f"{shift:>+7.1f}pp  {sign_str}")
    full_means = [hwsa_full[(m, ds, pol)]["hwsa_qc"] for m in models for ds in datasets for pol in policies]
    filt_means = [hwsa_filt[(m, ds, pol)]["hwsa_qc"] for m in models for ds in datasets for pol in policies]
    print(f"macro-avg HwSA@QC  full={np.mean(full_means)*100:.1f}%  "
          f"filtered={np.mean(filt_means)*100:.1f}%  "
          f"shift={( np.mean(filt_means) - np.mean(full_means))*100:+.1f}pp")
    all_str = "NO"
    if all_sign_ok:
        all_str = "YES"
    print(f"all 24 delta signs preserved: {all_str}")

def print_tables_main(hwsa, fac, ans_rate, aurc_res, cmp):
    print("hwsa")
    print(f"{'model':<6} {'ds':<5} {'pol':<4} {'n':>4}  {'hwsa_q0':>10}  {'hwsa_qc':>10}  {'delta':>12}  excl0")
    for m in models:
        for ds in datasets:
            for pol in policies:
                v = hwsa[(m, ds, pol)]
                lo_q0, hi_q0 = v["ci_q0"]
                lo_qc, hi_qc = v["ci_qc"]
                tmp = v["ci_delta"]
                lo_d = tmp[0]
                hi_d = tmp[1]
                excl = "Y" if v["excl0"] else "N"
                print(f"{m:<6} {ds:<5} {pol:<4} {v['n']:>4}  "
                      f"{v['hwsa_q0']*100:>5.1f}% [{lo_q0*100:.1f},{hi_q0*100:.1f}]  "
                      f"{v['hwsa_qc']*100:>5.1f}% [{lo_qc*100:.1f},{hi_qc*100:.1f}]  "
                      f"{v['delta']*100:>+6.1f}pp [{lo_d*100:+.1f},{hi_d*100:+.1f}]  {excl}")

    print("\nfac (p1-anchored cap_set)")
    print(f"{'model':<6} {'ds':<5} {'pol':<4}  {'fac@q100':>18}  {'fac@q50':>18}")
    for m in models:
        for ds in datasets:
            for pol in policies:
                fq1m, fq1ci = fac[(m, ds, pol, 'Q100')]
                tmp = fac[(m, ds, pol, 'Q50')]
                fq5m = tmp[0]
                fq5ci = tmp[1]
                print(f"{m:<6} {ds:<5} {pol:<4}  "
                      f"{fq1m*100:>5.1f}% [{fq1ci[0]*100:.1f},{fq1ci[1]*100:.1f}]  "
                      f"{fq5m*100:>5.1f}% [{fq5ci[0]*100:.1f},{fq5ci[1]*100:.1f}]")

    print("\nanswer rate (raw, p3 at cf=50 f1=0.8 default)")
    print(f"{'model':<6} {'ds':<5} {'pol':<4}  {'q100_ans':>18}  {'q50_ans':>18}")
    for m in models:
        for ds in datasets:
            for pol in policies:
                a1m, a1ci = ans_rate[(m, ds, pol, "Q100")]
                tmp = ans_rate[(m, ds, pol, "Q50")]
                a5m = tmp[0]
                a5ci = tmp[1]
                print(f"{m:<6} {ds:<5} {pol:<4}  "
                      f"{a1m*100:>5.1f}% [{a1ci[0]*100:.1f},{a1ci[1]*100:.1f}]  "
                      f"{a5m*100:>5.1f}% [{a5ci[0]*100:.1f},{a5ci[1]*100:.1f}]")
    print("\nmacro-avg ans rate across 6 cells")
    print(f"{'pol':<4}  {'q100_ans':>10}  {'q50_ans':>10}")
    for pol in policies:
        v100=[ans_rate[(m, ds, pol, "Q100")][0] for m in models for ds in datasets]
        v50=[ans_rate[(m, ds, pol, "Q50")][0] for m in models for ds in datasets]
        print(f"{pol:<4}  {np.mean(v100)*100:>9.1f}%  {np.mean(v50)*100:>9.1f}%")

    print("\naurc")
    print(f"{'model':<6} {'ds':<5}  " + "  ".join(f"{p:>8}" for p in policies))
    for m in models:
        for ds in datasets:
            vals = [f"{aurc_res[(m, ds, pol)]:.4f}" for pol in policies]
            print(f"{m:<6} {ds:<5}  {'  '.join(f'{v:>8}' for v in vals)}")
    means = []
    for pol in policies:
        means.append(np.mean([aurc_res[(m, ds, pol)] for m in models for ds in datasets]))
    print("mean        " + "  ".join(f"{v:>8.4f}" for v in means))

    print("\ncomparability")
    for ds in datasets:
        c = cmp[ds]
        em_m, em_ci = c["em_elig"]
        em_nm, em_nci = c["em_nonel"]
        f1_m, f1_ci = c["f1_elig"]
        f1_nm, f1_nci = c["f1_nonel"]
        lbl = "comparable" if c['em_ok'] else "DIFFERENT"
        print(f"{ds.upper()}: {c['elig_n_q']} eligible / {c['nonel_n_q']} non-eligible")
        print(f"  em  elig={pct(em_m)} {ci_pct(*em_ci)}  non-elig={pct(em_nm)} {ci_pct(*em_nci)}  "
              f"diff={pp(c['em_diff'])} {ci_pp(*c['em_ci'])}  {lbl}")
        lbl2 = "comparable" if c["f1_ok"] else "DIFFERENT"
        print(f"  f1  elig={f1_m:.3f} [{f1_ci[0]:.3f},{f1_ci[1]:.3f}]  "
              f"non-elig={f1_nm:.3f} [{f1_nci[0]:.3f},{f1_nci[1]:.3f}]  "
              f"diff={c['f1_diff']:+.3f} [{c['f1_ci'][0]:+.3f},{c['f1_ci'][1]:+.3f}]  {lbl2}")

def qc_source_analysis(all_rows, eval_records):
    try:
        swap_maps={}
        for ds in datasets:
            swap_maps[ds] = build_swap_map(eval_records[ds])

        counts = defaultdict(lambda: defaultdict(int))
        for r in all_rows:
            if r.get("condition") != "QC":
                continue
            m, ds, pol = r["model"], r["dataset"], r["policy"]
            key = (m, ds, pol)
            counts[key]["total_qc"] += 1
            if r["decision"] != "answer":
                continue
            counts[key]["answered"] += 1
            if r.get("correct_em", False) == True:
                counts[key]["correct"] += 1
                continue
            planted = swap_maps[ds].get(r["question_id"])
            if is_echoing(r.get("answer", ""), planted):
                counts[key]["echoed"] += 1
            else:
                counts[key]["hallucinated"] += 1

        print("\nqc answer source")
        print(f"{'model':<6} {'ds':<5} {'pol':<4}  {'qc rows':>8}  {'ans%':>6}  {'correct%':>9}  {'echoed%':>9}  {'halluc%':>9}")
        for m in models:
            for ds in datasets:
                for pol in policies:
                    v = counts[(m, ds, pol)]
                    nq = v["total_qc"]
                    na = v["answered"]
                    if na == 0:
                        print(f"{m:<6} {ds:<5} {pol:<4}  {nq:>8}  {'0':>6}  {'n/a':>9}  {'n/a':>9}  {'n/a':>9}")
                    else:
                        print(f"{m:<6} {ds:<5} {pol:<4}  {nq:>8}  {na/nq*100:>5.1f}%  "
                              f"{v['correct']/na*100:>8.1f}%  {v['echoed']/na*100:>8.1f}%  "
                              f"{v['hallucinated']/na*100:>8.1f}%")

        grand = defaultdict(int)
        for v in counts.values():
            for k, n in v.items():
                grand[k] += n
        na = grand["answered"]
        print(f"all: {na} answers, correct={grand['correct']/na*100:.1f}%  "
              f"echoed={grand['echoed']/na*100:.1f}%  hallucinated={grand['hallucinated']/na*100:.1f}%")
        return counts

    except (ImportError, RuntimeError) as e:
        print(f"qc_source_analysis skipped: {e}")
        return None

def calibration_analysis(rows):
    calib = compute_calibration(rows)

    print("\nece by policy x condition")
    print(f"{'':10}" + "".join(f"  {c:<8}" for c in conditions))
    for pol in policies:
        parts=[]
        for cond in conditions:
            if (pol, cond) in calib:
                parts.append(f"{calib[(pol,cond)]['ece']:.4f}  ")
            else:
                parts.append("  n/a")
        print(f"{pol:<10}" + "".join(parts))

    print("\nconfidence gap")
    print(f"{'pol':<5}  {'cond':<6}  {'mean_conf':>9}  {'mean_corr%':>10}  {'gap':>9}  verdict")
    for pol in policies:
        for cond in conditions:
            v = calib.get((pol, cond))
            if v == None:
                continue
            gap = v["mean_conf"] - v["mean_correct"] * 100
            if gap > 10:
                verdict = "overconfident"
            elif gap < -10:
                verdict = "underconfident"
            else:
                verdict = "calibrated"
            print(f"{pol:<5}  {cond:<6}  {v['mean_conf']:>9.1f}  {v['mean_correct']*100:>9.1f}%  {gap:>+8.1f}  {verdict}")

    return calib

def phi_overabstention(all_rows):
    #q100 answer capacity vs qc hwsa
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for r in all_rows:
        data[r["model"]][r["dataset"]][r["policy"]][r["condition"]].append(r["decision"])

    print("\nq100 ans rate / qc hwsa / norm (avg across policies)")
    print(f"  {'model':<20} {'dataset':<8} {'q100 ans%':>10} {'qc hwsa%':>10} {'norm%':>8}")
    msumm = defaultdict(list)
    for model in models:
        for dataset in datasets:
            q100_rates=[]
            qc_rates=[]
            for policy in policies:
                q100 = data[model][dataset][policy].get("Q100", [])
                qc = data[model][dataset][policy].get("QC", [])
                if q100:
                    q100_rates.append(sum(1 for d in q100 if d == "answer") / len(q100))
                if qc:
                    qc_rates.append(sum(1 for d in qc if d == "answer") / len(qc))
            q100_m = statistics.mean(q100_rates) if q100_rates else float("nan")
            qc_m = statistics.mean(qc_rates) if qc_rates else float("nan")
            if q100_m > 0:
                norm = qc_m / q100_m
            else:
                norm = float("nan")
            flag = " (*)" if model == "phi" else ""
            print(f"  {mlabels[model]:<20} {dslabels[dataset]:<8} {q100_m*100:>9.1f}% {qc_m*100:>9.1f}% {norm*100:>7.1f}%{flag}")
            msumm[model].append((q100_m, qc_m, norm))

    print(f"\n  {'model':<20} {'q100 ans%':>10} {'qc hwsa%':>10} {'norm%':>8}  note")
    for model in models:
        vals = msumm[model]
        q100_avg = statistics.mean(v[0] for v in vals)
        qc_avg = statistics.mean(v[1] for v in vals)
        norm_avg = statistics.mean(v[2] for v in vals)
        if q100_avg < 0.75:
            note = "low q100 capacity"
        else:
            note = "normal capacity"
        print(f"  {mlabels[model]:<20} {q100_avg*100:>9.1f}% {qc_avg*100:>9.1f}% {norm_avg*100:>7.1f}%  {note}")

    print(f"\nphi per-policy breakdown")
    print(f"  {'dataset':<8} {'policy':<6} {'q100 ans%':>10} {'qc hwsa%':>10} {'norm%':>8}")
    for dataset in datasets:
        for policy in policies:
            q100 = data["phi"][dataset][policy].get("Q100", [])
            qc = data["phi"][dataset][policy].get('QC', [])
            if q100:
                q100_r = sum(1 for d in q100 if d == "answer") / len(q100)
            else:
                q100_r = float("nan")
            if qc:
                qc_r = sum(1 for d in qc if d == "answer") / len(qc)
            else:
                qc_r = float("nan")
            norm = qc_r / q100_r if q100_r > 0 else float("nan")
            print(f"  {dslabels[dataset]:<8} {policy:<6} {q100_r*100:>9.1f}% {qc_r*100:>9.1f}% {norm*100:>7.1f}%")

def nq_hpqa_divergence(all_rows):
    qc_eligible = defaultdict(set)
    for r in all_rows:
        if r["condition"] == "QC":
            qc_eligible[r["dataset"]].add(r["question_id"])

    for dataset in datasets:
        print(f"qc-eligible: {dslabels[dataset]} = {len(qc_eligible[dataset])} questions")

    abs_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    conf_abs = defaultdict(list)
    conf_ans = defaultdict(list)
    span_data = defaultdict(list)

    for r in all_rows:
        if r["condition"] != "QC":
            continue
        if r["question_id"] not in qc_eligible[r["dataset"]]:
            continue
        model = r["model"]
        dataset = r["dataset"]
        policy = r["policy"]
        abs_data[dataset][model][policy].append(r["decision"] == "abstain")
        if policy in ("P0", "P1", "P2"):
            conf = float(r.get("confidence", 50))
            if r["decision"] == "abstain":
                conf_abs[dataset].append(conf)
            else:
                conf_ans[dataset].append(conf)
        if r["decision"] == "answer" and r.get("answer", "").strip():
            span_data[dataset].append(len(r["answer"].split()))

    print(f"\nqc abstention within eligible subset")
    print(f"  {'dataset':<8} {'model':<20} {'P0':>6} {'P1':>6} {'P2':>6} {'P3':>6} {'mean':>7}")
    ds_tot = {}
    for dataset in datasets:
        all_vals=[]
        for model in models:
            per_policy=[]
            row_parts=[]
            for policy in policies:
                vals = abs_data[dataset][model][policy]
                if vals:
                    r_pct = 100 * sum(vals) / len(vals)
                    per_policy.append(r_pct)
                    all_vals.extend(vals)
                    row_parts.append(f"{r_pct:>6.1f}")
                else:
                    row_parts.append(f"{'n/a':>6}")
            mean_r = statistics.mean(per_policy) if per_policy else float("nan")
            print(f"  {dslabels[dataset]:<8} {mlabels[model]:<20} " + " ".join(row_parts) + f" {mean_r:>7.1f}")
        overall = 100 * sum(all_vals) / len(all_vals) if all_vals else float("nan")
        ds_tot[dataset] = overall
        print(f"  {dslabels[dataset]} overall: {overall:.1f}%")

    gap = ds_tot.get("hpqa", 0) - ds_tot.get("nq", 0)
    print(f"  gap (hpqa - nq): {gap:+.1f} pp")

    print(f"\nanswer span length")
    print(f"  {'dataset':<8} {'mean words':>11} {'median':>8} {'n':>6}")
    for dataset in datasets:
        spans = span_data[dataset]
        if spans:
            print(f"  {dslabels[dataset]:<8} {statistics.mean(spans):>10.2f} {statistics.median(spans):>8.1f} {len(spans):>6}")
        else:
            print(f"  {dslabels[dataset]:<8}  no answered qc rows")

    print(f"\nconfidence at qc (P0/P1/P2)")
    print(f"  {'dataset':<8} {'abstainers':>16} {'n':>6}  {'answerers':>15} {'n':>6}")
    for dataset in datasets:
        ab = conf_abs[dataset]
        an = conf_ans[dataset]
        ab_m = f"{statistics.mean(ab):.1f}" if ab else "n/a"
        an_m = f"{statistics.mean(an):.1f}" if an else "n/a"
        print(f"  {dslabels[dataset]:<8} {ab_m:>16} {len(ab):>6}  {an_m:>15} {len(an):>6}")

    print(f"\nqc abstention per model")
    print(f"  {'model':<20} {'nq':>8} {'hpqa':>10} {'gap':>8}")
    for model in models:
        nq_vals=[]
        hpqa_vals=[]
        for p in policies:
            nq_vals.extend(abs_data["nq"][model][p])
            hpqa_vals.extend(abs_data["hpqa"][model][p])
        nq_r = 100 * sum(nq_vals) / len(nq_vals) if nq_vals else float("nan")
        hpqa_r = 100 * sum(hpqa_vals) / len(hpqa_vals) if hpqa_vals else float("nan")
        print(f"  {mlabels[model]:<20} {nq_r:>7.1f}% {hpqa_r:>9.1f}% {hpqa_r - nq_r:>+7.1f} pp")

def main():
    out_path = results_base / "statistics.txt"
    all_rows = load_all_rows()
    data = build_index(all_rows)
    eval_records = {ds: load_eval(ds) for ds in datasets}
    usable_qids = load_usable_qids()

    with open(out_path, 'w', encoding='utf-8') as f:
        tee = _Tee(sys.stdout, f)
        with contextlib.redirect_stdout(tee):
            print(f"rows: {len(all_rows):,}")

            cmp = comparability_test(data)
            hwsa_res = compute_hwsa(data)
            fac_res = compute_fac(data)
            ans_res = compute_ans_rate(data)
            aurc_res = compute_aurc_all(data)
            print_tables_main(hwsa_res, fac_res, ans_res, aurc_res, cmp)

            hwsa_filt = compute_hwsa(data, qid_filter=usable_qids)
            print_hwsa_robustness(hwsa_res, hwsa_filt, usable_qids)

            src_cnt = qc_source_analysis(all_rows, eval_records)
            calib = calibration_analysis(all_rows)

            phi_overabstention(all_rows)
            nq_hpqa_divergence(all_rows)

    print(f"\nstatistics written to: {out_path}")

if __name__ == "__main__":
    main()
