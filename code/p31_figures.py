"""P3.1 :  results/*.json 5 (PNG 300dpi + PDF) -> figures/

Fig1 (12 test mean QWK)  Fig2 (3)
Fig3 (cumul vs CE)          Fig4 (//)
Fig5 8()
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
FIG = PROJ / "figures"
FIG.mkdir(exist_ok=True)
LABEL_COLS = ["HeartSize", "PulmonaryCongestion", "PleuralEffusion_Right", "PleuralEffusion_Left",
              "PulmonaryOpacities_Right", "PulmonaryOpacities_Left", "Atelectasis_Right", "Atelectasis_Left"]
SHORT = {"HeartSize": "Heart", "PulmonaryCongestion": "Congest.", "PleuralEffusion_Right": "Effusion R",
         "PleuralEffusion_Left": "Effusion L", "PulmonaryOpacities_Right": "Opacities R",
         "PulmonaryOpacities_Left": "Opacities L", "Atelectasis_Right": "Atelect. R", "Atelectasis_Left": "Atelect. L"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}")


def fig1_matrix():
    d = json.load(open(PROJ / "results/matrix_summary_p4.json"))
    names = ["224bc", "224bm", "224lc", "224lm", "518bc", "518bm", "raddino", "biomedclip"]
    rows = ["B · 224 · cls", "B · 224 · mean", "L · 224 · cls", "L · 224 · mean",
            "B · 518 · cls", "B · 518 · mean", "RAD-DINO · cls", "BiomedCLIP · cls"]
    M = np.array([[d[f"{n}{w}"]["mean"] for w in ("_u", "_w")] for n in names])
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    im = ax.imshow(M, cmap="YlGnBu", vmin=0.39, vmax=0.54, aspect="auto")
    ax.set_xticks([0, 1], ["unweighted", "weighted"], fontsize=8)
    ax.set_yticks(range(len(names)), rows, fontsize=8)
    for i in range(len(names)):
        for j in range(2):
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > 0.50 else "black")
    ax.set_title("Frozen-feature linear probe: test mean QWK\n(mean over seeds 0-2; 518 = native resolution)", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.85)
    save(fig, "fig1_config_matrix")


def fig2_loss():
    d = json.load(open(PROJ / "results/loss_matrix_p22.json"))["loss_matrix"]
    order = ["ce", "softord", "coral_shared", "corn", "coral"]
    names = ["Plain CE", "Soft-Ordinal\nCE", "CORAL\n(shared-w)", "CORN\n(conditional)", "Unconstrained\ncumulative"]
    colors = ["#aaa", "#8ab", "#58a", "#7a7", "#246"]
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    for i, k in enumerate(order):
        t = d[k]
        ax.bar(i, t["mean"], yerr=t["std"], capsize=4, color=colors[i],
               edgecolor="black", linewidth=0.5)
        ax.text(i, t["mean"] + t["std"] + 0.002, f'{t["mean"]:.4f}', ha="center", fontsize=7.5)
    ax.set_xticks(range(5), names, fontsize=7.5)
    ax.set_ylabel("Test mean QWK (518-base-mean)")
    ax.set_ylim(0.38, 0.45)
    ax.set_title("Loss formulation (3 seeds, mean±std)", fontsize=9)
    save(fig, "fig2_loss_matrix")


def fig3_calibration():
    """ECE(p41). coverage(0.997), Table III."""
    d = json.load(open(PROJ / "results/p41_conformal_proper.json"))["heads"]
    tasks = LABEL_COLS
    x = np.arange(len(tasks))
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    cv = [d["cumul"]["ece"][t] for t in tasks]
    ce = [d["ce"]["ece"][t] for t in tasks]
    ax.bar(x - 0.18, cv, 0.36, label="cumulative", color="#246")
    ax.bar(x + 0.18, ce, 0.36, label="CE softmax", color="#c66")
    ax.set_xticks(x, [SHORT[t] for t in tasks], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("ECE (15 bins)")
    ax.set_title("Per-task calibration error (lower is better)", fontsize=9)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.95)
    fig.tight_layout()
    save(fig, "fig3_calibration_conformal")


def fig4_strata():
    d = json.load(open(PROJ / "results/p26_stratified.json"))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    ax = axes[0]
    groups = [("normal", "#8ab"), ("inverted_corrected", "#c66"), ("12bit_scale", "#7a7"), ("16bit_scale", "#2a6")]
    for i, (g, c) in enumerate(groups):
        t = d["inv_group" if g in ("normal", "inverted_corrected") else "bit_group"][g]
        ax.bar(i, t["mean_qwk"], color=c, edgecolor="black", linewidth=0.5)
        ax.text(i, t["mean_qwk"] + 0.008, f'{t["mean_qwk"]:.3f}\n(n={t["n"]:,})', ha="center", fontsize=7)
    ax.set_xticks(range(4), ["normal\nphotometry", "inverted\n(corrected)", "12-bit\nscale", "16-bit\nscale"], fontsize=8)
    ax.set_ylabel("Test mean QWK")
    ax.set_ylim(0, 0.52)
    ax.set_title("Scenario stratification (518-base-mean)", fontsize=9)
    ax = axes[1]
    for c, col in [("Atelectasis_Left", "#246"), ("HeartSize", "#c66"), ("PleuralEffusion_Right", "#7a7")]:
        rec = d["per_grade_recall"][c]
        gs = sorted(int(g) for g in rec)
        r = [rec[str(g)]["recall"] for g in gs]
        ax.plot(gs, r, "-o", ms=4, label=SHORT[c], color=col)
    ax.set_xlabel("True severity grade")
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-grade recall collapses at extremes", fontsize=9)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.95)
    fig.tight_layout()
    save(fig, "fig4_stratification")


def fig5_label_dist():
    d = json.load(open(PROJ / "results/p26_stratified.json"))["per_grade_recall"]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for ci, c in enumerate(LABEL_COLS):
        rec = d[c]
        gs = sorted(int(g) for g in rec)
        sup = [rec[str(g)]["support"] for g in gs]
        ax.plot(gs, np.array(sup) / 42928 * 100, "-o", ms=3, lw=1, alpha=0.75,
                color=plt.cm.viridis(ci / 8), label=SHORT[c])
    ax.set_xlabel("Severity grade")
    ax.set_ylabel("% of test set")
    ax.set_title("Ordinal label distribution (test, n=42,928)\nHeartSize has 4 grades, others 5", fontsize=9)
    ax.legend(fontsize=6, ncol=2)
    save(fig, "fig5_label_distribution")


if __name__ == "__main__":
    for fn in [fig1_matrix, fig2_loss, fig3_calibration, fig4_strata, fig5_label_dist]:
        try:
            fn()
        except Exception as e:
            print(f"  {fn.__name__} : {e}")
    print("P3.1 ")
