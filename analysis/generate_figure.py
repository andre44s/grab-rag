import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

script_dir=Path(__file__).parent
repo_dir = script_dir.parent
results_dir = repo_dir / "results"
fig_dir=results_dir / "figures"

models = ["phi", "llama", "qwen"]
datasets = ["nq", "hpqa"]
policies = ["P0", "P1", "P2", "P3"]
cond_order = ["Q100", "Q50", "Q0", "QC"]

mlabels = {"phi": "Phi-4-mini\n(3.8B)", "llama": "Llama-3.1-8B", "qwen": "Qwen2.5-7B"}
plabels = {"P0": "P0 (none)", "P1": "P1 (IDK)", "P2": "P2 (CoT)", "P3": "P3 (conflict)"}
pc = {"P0": "#999999", "P1": "#E69F00", "P2": "#56B4E9", "P3": "#009E73"}
pls = {"P0": "--", "P1": "-", "P2": "-", "P3": ":"}
pm = {"P0": "s", "P1": "o", "P2": "^", "P3": "D"}

rc = {
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
}

def load_data():
    rows=[]
    #print("loading from", results_dir)
    for model in models:
        for dataset in datasets:
            path = results_dir / f"{model}-{dataset}" / f"{model}_{dataset}.jsonl"
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
    return rows

def fig1_abstention_cliff(rows):
    #bucket by dataset so we can average rates across datasets, not pool rows
    abstain = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for r in rows:
        if r.get("error") or r["decision"] == "unknown":
            continue
        abstain[r["dataset"]][r["model"]][r["policy"]][r["condition"]].append(
            r["decision"] == "abstain"
        )

    n_total = sum(len(v) for ds in abstain.values() for m in ds.values()
                   for p in m.values() for v in p.values())

    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
        x=np.arange(len(cond_order))

        for ax, model in zip(axes, models):
            for policy in policies:
                rates,lo, hi = [],[],[]
                for cond in cond_order:
                    #pooled se across datasets
                    ds_pairs = [
                        (float(np.mean(abstain[ds][model][policy][cond])),
                         len(abstain[ds][model][policy][cond]))
                        for ds in datasets if abstain[ds][model][policy][cond]
                    ]
                    if ds_pairs:
                        k = len(ds_pairs)
                        p = np.mean([pi for pi, _ in ds_pairs])
                        se = np.sqrt(sum(pi * (1 - pi) / ni for pi, ni in ds_pairs)) / k
                    else:
                        p, se = np.nan, 0.0
                    rates.append(p * 100)
                    lo.append(p * 100 - 1.96 * se * 100)
                    hi.append(p * 100 + 1.96 * se * 100)

                ax.plot(x, rates, color=pc[policy], linestyle=pls[policy],
                    marker=pm[policy], markersize=5, linewidth=1.8,
                    label=plabels[policy], zorder=3)
                ax.fill_between(x, lo, hi, color=pc[policy], alpha=0.10, zorder=2)

            ax.axvspan(1.5, 3.5, color="#F4D7D7", alpha=0.35, zorder=0)
            ax.text(2.5, 2, "should abstain", ha="center", va="bottom",
                    fontsize=8, color="#A04040", fontstyle="italic")
            ax.set_title(mlabels[model], fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(cond_order)
            ax.set_ylim(-3, 108)
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
            ax.grid(axis="y")

        axes[0].set_ylabel("Abstention rate")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(policies),
            bbox_to_anchor=(0.5, 0.0), frameon=False,
            handlelength=2.2, columnspacing=2.0)
        fig.suptitle(
            "Abstention Rate by Context Condition  (shaded band = 95% CI; averaged NQ + HotpotQA)",
            fontsize=9, y=1.01,
        )
        fig.tight_layout(rect=[0, 0.05, 1, 1])
    return fig

def main():
    fig_dir.mkdir(parents=True, exist_ok=True)
    rows = load_data()
    fig = fig1_abstention_cliff(rows)
    out = fig_dir / "fig1_abstention_cliff.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")

if __name__ == "__main__":
    main()
