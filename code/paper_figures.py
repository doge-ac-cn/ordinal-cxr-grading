"""Paper figures: Nature-journal palette (muted, colorblind-safe, grayscale-distinguishable).
Palette: Navy #3C5488 | Teal #37827B | Orange #E18727 | L-Steel #9FB3D1 | D-Slate #33384A | Gray #9AA3AD
Text/axes #262626 | grid #D9D9D9. Font: Liberation Sans (Arial-metric compatible).
Output: figures/Fig{1,2,3}.pdf + .png + .tif (600 dpi, width <= 174 mm).
"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "code"))
from p31_figures import LABEL_COLS, SHORT

NAVY  = '#3C5488'
TEAL  = '#37827B'
ORANGE = '#E18727'
LSTEEL = '#9FB3D1'
DSLATE = '#33384A'
GRAY  = '#9AA3AD'
TEXT  = '#262626'

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": TEXT,
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "text.color": TEXT,
})

OUT = PROJ / "figures"
W_MAX_MM = 174

def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=600, bbox_inches="tight", facecolor='white')
    plt.close(fig)
    im = Image.open(OUT / f"{name}.png").convert("RGB")
    w_mm = im.width / 600 * 25.4
    if w_mm > W_MAX_MM:
        scale = W_MAX_MM / w_mm
        im = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
        im.save(OUT / f"{name}.png", dpi=(600, 600))
    im.save(OUT / f"{name}.tif", compression="tiff_lzw", dpi=(600, 600))
    print(f"{name}: {im.width}x{im.height}px = {im.width/600*25.4:.0f}mm")


# ---------- Fig1: per-grade label distribution (8 lines; hue = finding, linestyle = side) ----------
def fig1_dist():
    d = json.load(open(PROJ / "results/p26_stratified.json"))["per_grade_recall"]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    pathology_colors = {
        "HeartSize": ORANGE, "PulmonaryOpacities": LSTEEL,
        "PleuralEffusion": TEAL, "Atelectasis": NAVY,
        "PulmonaryCongestion": DSLATE,
    }
    lines = [
        ("HeartSize", "Heart", "HeartSize", True),
        ("PleuralEffusion_Right", "Effusion R", "PleuralEffusion", True),
        ("PleuralEffusion_Left", "Effusion L", "PleuralEffusion", False),
        ("PulmonaryOpacities_Right", "Opacities R", "PulmonaryOpacities", True),
        ("PulmonaryOpacities_Left", "Opacities L", "PulmonaryOpacities", False),
        ("Atelectasis_Right", "Atelect. R", "Atelectasis", True),
        ("Atelectasis_Left", "Atelect. L", "Atelectasis", False),
        ("PulmonaryCongestion", "Congest.", "PulmonaryCongestion", True),
    ]
    for col, label, ck, solid in lines:
        rec = d[col]
        gs = sorted(int(g) for g in rec)
        sup = [rec[str(g)]["support"] for g in gs]
        pct = np.array(sup) / 42928 * 100
        ls = '-' if solid else '--'
        marker = 'o' if solid else '^'
        ax.plot(gs, pct, ls, marker=marker, ms=4, lw=1.5,
                color=pathology_colors[ck], label=label, markerfacecolor=pathology_colors[ck] if solid else 'white',
                markeredgecolor=pathology_colors[ck])
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xlabel("Severity grade", fontsize=8, color=TEXT)
    ax.set_ylabel("% of test set", fontsize=8, color=TEXT)
    ax.legend(fontsize=6.5, ncol=2, frameon=False)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    save(fig, "Fig1")


# ---------- Fig2: ECE per task, cumulative vs CE ----------
def fig2_ece():
    d = json.load(open(PROJ / "results/p41_conformal_proper.json"))["heads"]
    tasks = LABEL_COLS
    x = np.arange(len(tasks))
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    cv = [d["cumul"]["ece"][t] for t in tasks]
    ce = [d["ce"]["ece"][t] for t in tasks]
    ax.bar(x - 0.18, cv, 0.36, label="cumulative", color=NAVY, edgecolor='white', linewidth=0.5)
    ax.bar(x + 0.18, ce, 0.36, label="CE softmax", color=GRAY, edgecolor='white', linewidth=0.5)
    ax.set_xticks(x, [SHORT[t] for t in tasks], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("ECE (15 bins)", fontsize=8, color=TEXT)
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    save(fig, "Fig2")


# ---------- Fig3: photometric/bit-depth strata (bars) + per-grade recall (lines) ----------
def fig3_strata():
    d = json.load(open(PROJ / "results/p26_stratified.json"))
    fig, axes = plt.subplots(1, 2, figsize=(6.84, 2.66))
    ax = axes[0]
    groups = [("normal", NAVY), ("inverted_corrected", ORANGE), ("12bit_scale", TEAL), ("16bit_scale", LSTEEL)]
    for i, (g, c) in enumerate(groups):
        t = d["inv_group" if g in ("normal", "inverted_corrected") else "bit_group"][g]
        ax.bar(i, t["mean_qwk"], color=c, edgecolor='white', linewidth=0.5)
        ax.text(i, t["mean_qwk"] + 0.008, f'{t["mean_qwk"]:.3f}\n(n={t["n"]:,})', ha="center", fontsize=6.5, color=TEXT)
    ax.set_xticks(range(4), ["normal\nphotometry", "inverted\n(corrected)", "12-bit\nscale", "16-bit\nscale"], fontsize=7)
    ax.set_ylabel("Test mean QWK", fontsize=8, color=TEXT)
    ax.set_ylim(0, 0.52)
    ax.tick_params(labelsize=7)
    ax = axes[1]
    for c, col in [("Atelectasis_Left", NAVY), ("HeartSize", ORANGE), ("PleuralEffusion_Right", TEAL)]:
        rec = d["per_grade_recall"][c]
        gs = sorted(int(g) for g in rec)
        r = [rec[str(g)]["recall"] for g in gs]
        ax.plot(gs, r, "-o", ms=4, lw=1.5, label=SHORT[c], color=col, markerfacecolor=col, markeredgecolor=col)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xlabel("True severity grade", fontsize=8, color=TEXT)
    ax.set_ylabel("Recall", fontsize=8, color=TEXT)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    save(fig, "Fig3")


if __name__ == "__main__":
    fig1_dist()
    fig2_ece()
    fig3_strata()
    print("Paper figures written to", OUT)
