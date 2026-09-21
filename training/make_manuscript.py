"""Build the OmiCoreTumor preprint PDF (bioRxiv style) with 4 multipanel figures."""
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image
from sklearn.metrics import roc_curve

Image.MAX_IMAGE_PIXELS = None

R = Path("reports")
M = R / "manuscript"
ST = "/home/shamim/Documents/spatial_transcriptomics/results"
SECTIONS = [("Cancer_P1", "tumour"), ("Cancer_P2", "tumour"), ("Cancer_P5", "tumour"),
            ("Normal_P3", "normal"), ("Normal_P5", "normal")]
THR, DOWN = 0.4, 10

# Validated categorical palette (dataviz skill: all checks pass, light mode).
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8985"
TUMOUR_RED = "#c8102e"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlesize": 10,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})
PAGE = (8.27, 11.69)   # A4 portrait


VARIANT_LABEL = {
    "A_raw": "ConvNeXt (norm)",
    "A_stainnorm": "ConvNeXt + stain norm",
    "A_tta": "ConvNeXt + TTA",
    "B_raw": "ConvNeXt (NONORM)",
    "C_raw": "EfficientNetV2 (NONORM)",
    "ENS_ABC": "Ensemble",
    "ENS_ABC_stainnorm": "Ensemble + stain norm",
    "molecular_ep1": "ConvNeXt, 1 epoch",
}


def panel(ax, letter, title=None):
    ax.text(-0.09, 1.06, letter, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="bottom", ha="left")
    if title:
        ax.set_title(title, fontsize=9.5, pad=6, color=INK)


def bar_labels(ax, bars, fmt="{:.1%}", dy=0.012, fs=7.5):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=fs, color=INK2)


# ---------------------------------------------------------------- data
def load():
    d = {}
    d["sections"] = json.loads((M / "sections.json").read_text())
    d["color"] = json.loads((M / "colorstats.json").read_text())
    d["bench"] = [json.loads(p.read_text()) for p in sorted(R.glob("metrics_*.json"))]
    d["mol"] = json.loads(
        (R / "molecular/ENS_ABC_stainnorm/Cancer_P1_molecular_validation.json").read_text())
    d["tiles"] = pd.read_csv(R / "molecular/ENS_ABC_stainnorm/Cancer_P1_tile_agreement.csv")
    rows = []
    for csv in sorted((R / "dry_test").rglob("dry_test_summary.csv")):
        t = pd.read_csv(csv)
        t["variant_dir"] = csv.parent.name
        rows.append(t)
    d["variants"] = pd.concat(rows, ignore_index=True)
    d["molvar"] = [json.loads(p.read_text())
                   for p in sorted(R.glob("molecular/*/Cancer_P1_molecular_validation.json"))]
    return d


def hm(name):
    z = np.load(M / f"{name}_hm.npz", allow_pickle=True)
    cls = [str(c) for c in z["classes"]]
    return (z["prob"][:, :, cls.index("TUM")].astype(np.float32),
            z["tissue"] > 0, z["xs"], z["ys"], int(z["tile"]), cls, z["prob"])


def overlay(name):
    """Downsampled H&E with the tumour boundary drawn on it."""
    img = Image.open(f"{ST}/{name}/segmentation/he_mpp0.5.tiff")
    W, H = img.size
    small = np.asarray(img.resize((W // DOWN, H // DOWN), Image.BILINEAR).convert("RGB"))
    p, tis, xs, ys, tile, _, _ = hm(name)
    m = ((p >= THR) & tis).astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)
    disp = np.zeros(small.shape[:2], np.uint8)
    for r in range(m.shape[0]):
        for c in range(m.shape[1]):
            if m[r, c]:
                disp[ys[r] // DOWN:(ys[r] + tile) // DOWN,
                     xs[c] // DOWN:(xs[c] + tile) // DOWN] = 1
    cnts, _ = cv2.findContours(disp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) >= 50_000 / DOWN ** 2]
    out = small.copy()
    if cnts:
        fill = np.zeros(small.shape[:2], np.uint8)
        cv2.drawContours(fill, cnts, -1, 1, cv2.FILLED)
        f = out.astype(np.float32)
        f[fill == 1] = 0.80 * f[fill == 1] + 0.20 * np.array([255, 40, 40], np.float32)
        out = f.astype(np.uint8)
    cv2.drawContours(out, cnts, -1, (200, 16, 46), 4)
    return small, out, len(cnts), p, tis


# ---------------------------------------------------------------- figure 1
def figure1(pdf, d):
    fig = plt.figure(figsize=PAGE)
    gs = fig.add_gridspec(2, 2, height_ratios=[0.62, 1.0],
                          hspace=0.46, wspace=0.34,
                          left=0.11, right=0.96, top=0.93, bottom=0.46)

    # (a) schematic of the pipeline
    ax = fig.add_subplot(gs[0, :]); panel(ax, "a", "Model construction")
    ax.set_xlim(0, 10); ax.set_ylim(0.55, 2.6); ax.axis("off"); ax.grid(False)
    boxes = [(0.1, "107,180 tiles\nNCT-CRC-HE-100K\n+ CRC-VAL-HE-7K", C1),
             (2.6, "3 backbones\nConvNeXt-T x2\nEfficientNetV2-S", C2),
             (5.1, "Macenko stain\nnormalisation\n(inference)", C3),
             (7.6, "Tumour map\nmask + GeoJSON\n(QuPath)", C4)]
    for x, txt, col in boxes:
        ax.add_patch(FancyBboxPatch((x, 0.9), 2.2, 1.35, boxstyle="round,pad=0.08",
                                    fc=col, ec="none", alpha=0.16))
        ax.text(x + 1.1, 1.57, txt, ha="center", va="center", fontsize=8.2, color=INK)
        if x < 7.5:
            ax.annotate("", xy=(x + 2.45, 1.57), xytext=(x + 2.25, 1.57),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.4))
    ax.text(5.0, 0.35, "9 tissue classes: ADI BACK DEB LYM MUC MUS NORM STR TUM "
                       "(TUM = adenocarcinoma epithelium)",
            ha="center", fontsize=7.6, color=INK2)

    # (b) training-set composition
    ax = fig.add_subplot(gs[1, 0]); panel(ax, "b", "Training tiles per class")
    cls = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
    n = [10407, 10566, 11512, 11557, 8896, 13536, 8763, 10446, 14317]
    cols = [C1] * 9
    cols[cls.index("TUM")] = TUMOUR_RED
    b = ax.bar(cls, n, color=cols, width=0.68)
    ax.set_ylabel("tiles"); ax.tick_params(axis="x", rotation=45, labelsize=7.5)
    ax.set_ylim(0, 17000)
    for r in b:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 350,
                f"{int(r.get_height()/1000)}k", ha="center", fontsize=6.8, color=INK2)
    ax.legend(handles=[Line2D([0], [0], marker="s", ls="", ms=7, color=TUMOUR_RED,
                              label="TUM (tumour)"),
                       Line2D([0], [0], marker="s", ls="", ms=7, color=C1,
                              label="other tissue")],
              fontsize=7, loc="upper left", frameon=False)

    # (c) held-out benchmarks -- two cohorts, tumour AUROC
    ax = fig.add_subplot(gs[1, 1]); panel(ax, "c", "Held-out tumour AUROC")
    bench = {b["tag"]: b for b in d["bench"]}
    models = [("A_convnext_norm", "ConvNeXt\n(norm)"), ("B_convnext_nonorm", "ConvNeXt\n(NONORM)"),
              ("C_effnet_nonorm", "EffNetV2\n(NONORM)")]
    x = np.arange(3); w = 0.36
    v7 = [bench.get(f"{m}__ext7k", {}).get("tumor_auroc", np.nan) for m, _ in models]
    v16 = [bench.get(f"{m}__ext2016", {}).get("tumor_auroc", np.nan) for m, _ in models]
    b1 = ax.bar(x - w / 2, v7, w, color=C1, label="CRC-VAL-HE-7K")
    b2 = ax.bar(x + w / 2, v16, w, color=C2, label="Kather-2016")
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in models], fontsize=7.5)
    ax.set_ylim(0.75, 1.03); ax.set_ylabel("tumour AUROC")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.006,
                    f"{r.get_height():.3f}", ha="center", fontsize=6.8, color=INK2)
    ax.legend(fontsize=7, frameon=False, loc="lower right")

    # (d) variant ranking on real sections
    ax = fig.add_axes([0.30, 0.135, 0.63, 0.245])
    panel(ax, "d", "Tumour/normal separation across model variants (5 sections)")
    v = d["variants"]
    rows = []
    for name in sorted(v["variant_dir"].unique()):
        s = v[v.variant_dir == name]
        t = s[s.expected == "tumour"]["tumor_frac"] if "tumour" in set(s.expected) \
            else s[s.expected == "tumor"]["tumor_frac"]
        nn = s[s.expected == "normal"]["tumor_frac"]
        if len(t) and len(nn):
            rows.append((name, t.min() - nn.max()))
    rows.sort(key=lambda r: r[1])
    labels = [VARIANT_LABEL.get(r[0], r[0]) for r in rows]
    vals = [r[1] for r in rows]
    cols = [C3 if i == len(vals) - 1 else C1 for i in range(len(vals))]
    bars = ax.barh(labels, vals, color=cols, height=0.62)
    ax.set_xlabel("separation  (min tumour area − max normal area)")
    ax.set_xlim(0, max(vals) * 1.22)
    ax.tick_params(axis="y", labelsize=7.8)
    for r in bars:
        ax.text(r.get_width() + 0.004, r.get_y() + r.get_height() / 2,
                f"{r.get_width():+.1%}", va="center", fontsize=7.2, color=INK2)
    ax.grid(axis="y", visible=False)

    fig.suptitle("Figure 1.  Construction and benchmarking of omicoretumor-crc-he-v0.1",
                 fontsize=11, fontweight="bold", y=0.975)
    fig.text(0.07, 0.085,
             "(a) Pipeline. (b) Class composition of the 100,000-tile training set. "
             "(c) Tumour AUROC on two held-out cohorts; Kather-2016 is a different\n"
             "scanner and cohort, so it measures cross-domain transfer. "
             "(d) Separation on the five Visium HD sections; >0 means tumour and normal\n"
             "sections do not overlap. The ensemble with stain normalisation (green) ranks first.",
             fontsize=7.3, color=INK2, va="top")
    pdf.savefig(fig); plt.close(fig)


# ---------------------------------------------------------------- figure 2
def figure2(pdf, d):
    fig = plt.figure(figsize=PAGE)
    gs = fig.add_gridspec(3, 2, hspace=0.50, wspace=0.34,
                          left=0.10, right=0.96, top=0.93, bottom=0.135)
    col = d["color"]

    # (a) stain domain gap in RGB
    ax = fig.add_subplot(gs[0, 0]); panel(ax, "a", "Stain domain gap")
    for key, c, lab in [("NCT", C1, "NCT training tiles"),
                        ("raw", C2, "Visium HD, raw"),
                        ("macenko", C3, "Visium HD, Macenko")]:
        p = np.array(col[key]["points"])
        ax.scatter(p[:, 0], p[:, 1], s=9, alpha=0.45, color=c, label=lab,
                   edgecolors="none")
        m = np.array(col[key]["mean"])
        ax.scatter(*m[:2], s=95, marker="X", color=c,
                   edgecolors="white", linewidths=1.4, zorder=5)
    ax.set_xlabel("mean R"); ax.set_ylabel("mean G")
    ax.legend(fontsize=6.8, frameon=False, loc="lower right")

    # (b) distance to the training domain
    ax = fig.add_subplot(gs[0, 1]); panel(ax, "b", "Distance to training domain")
    vals = [col["raw"]["dist"], col["macenko"]["dist"]]
    bars = ax.bar(["raw slide", "after Macenko"], vals, color=[C2, C3], width=0.55)
    ax.set_ylabel("RGB distance to NCT mean"); ax.set_ylim(0, max(vals) * 1.3)
    for r in bars:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 1.2,
                f"{r.get_height():.1f}", ha="center", fontsize=8, color=INK2)
    ax.text(0.5, 0.88, "−51%", transform=ax.transAxes, ha="center",
            fontsize=11, fontweight="bold", color=C3)

    # (c) cross-domain benefit of NONORM training
    ax = fig.add_subplot(gs[1, 0]); panel(ax, "c", "Effect of non-normalised training")
    bench = {b["tag"]: b for b in d["bench"]}
    names = [("A_convnext_norm", "trained on\nnormalised"),
             ("B_convnext_nonorm", "trained on\nNONORM")]
    vals = [bench.get(f"{m}__ext2016", {}).get("tumor_auroc", np.nan) for m, _ in names]
    bars = ax.bar([l for _, l in names], vals, color=[C2, C3], width=0.5)
    ax.set_ylabel("tumour AUROC, Kather-2016"); ax.set_ylim(0.7, 1.04)
    for r in bars:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.008,
                f"{r.get_height():.3f}", ha="center", fontsize=8.5, color=INK2)
    ax.set_title("Effect of non-normalised training\n(out-of-domain cohort)", fontsize=9.5)

    # (d) calibration drift -- the central practical warning
    ax = fig.add_subplot(gs[1, 1]); panel(ax, "d", "Calibration drift")
    t = d["tiles"]
    pos = t[t.mol_tumor_frac >= 0.5]["p_tum"].to_numpy()
    neg = t[t.mol_tumor_frac < 0.5]["p_tum"].to_numpy()
    bins = np.linspace(0, 1, 34)
    ax.hist(neg, bins=bins, color=C1, alpha=0.75, label="molecularly negative")
    ax.hist(pos, bins=bins, color=TUMOUR_RED, alpha=0.75, label="molecularly tumour")
    ax.axvline(THR, color=INK, lw=1.4, ls="--")
    ax.text(THR + 0.02, ax.get_ylim()[1] * 0.30, f"threshold {THR}",
            fontsize=7.2, color=INK)
    ax.set_xlabel("P(tumour)"); ax.set_ylabel("tiles"); ax.set_yscale("log")
    ax.legend(fontsize=6.6, frameon=False, loc="upper right",
              bbox_to_anchor=(1.0, 1.02))

    # (e) recall vs threshold for a single model vs the ensemble
    ax = fig.add_subplot(gs[2, :]); panel(ax, "e", "Why a fixed threshold does not transfer")
    ths = np.linspace(0.02, 0.9, 60)
    y = (t.mol_tumor_frac >= 0.5).to_numpy().astype(int)
    series = []
    for var, lab, c in [("A_raw", "single model, no stain norm", C2),
                        ("ENS_ABC_stainnorm", "ensemble + stain norm (released)", C3)]:
        f = R / f"molecular/{var}/Cancer_P1_tile_agreement.csv"
        if not f.exists():
            continue
        g = pd.read_csv(f)
        yy = (g.mol_tumor_frac >= 0.5).to_numpy().astype(int)
        s = g.p_tum.to_numpy()
        rec = [((s >= th) & (yy == 1)).sum() / max(yy.sum(), 1) for th in ths]
        ax.plot(ths, rec, lw=2, color=c, label=lab)
        series.append((lab, c, rec))
    ax.axvline(0.5, color=MUTED, lw=1, ls=":")
    ax.text(0.505, 0.06, "conventional 0.5", fontsize=7, color=MUTED)
    for lab, c, rec in series:
        r50 = float(np.interp(0.5, ths, rec))
        ax.scatter([0.5], [r50], s=46, color=c, zorder=5,
                   edgecolors="white", linewidths=1.2)
        ax.annotate(f"{r50:.2f}", (0.5, r50), textcoords="offset points",
                    xytext=(9, -3), fontsize=7.6, color=c, fontweight="bold")
    ax.set_xlabel("P(tumour) threshold"); ax.set_ylabel("recall vs molecular truth")
    ax.set_ylim(0, 1.04); ax.legend(fontsize=7.6, frameon=False, loc="lower left")

    fig.suptitle("Figure 2.  Stain-domain shift is the dominant failure mode, and "
                 "how it is mitigated", fontsize=11, fontweight="bold", y=0.975)
    fig.text(0.07, 0.095,
             "(a,b) Target slides are lighter and pinker than the training tiles; Macenko "
             "normalisation halves the gap. (c) Training on non-normalised\n"
             "tiles raises cross-cohort tumour AUROC from 0.836 to 0.992. (d) On target "
             "slides the two molecular classes stay well separated but the\n"
             "probabilities compress toward zero. (e) Consequence: at the conventional "
             "0.5 threshold a single model recovers roughly half the tumour it\n"
             "ranks correctly; the released ensemble restores recall. Thresholds must be "
             "re-checked on each new scanner.",
             fontsize=7.3, color=INK2, va="top")
    pdf.savefig(fig); plt.close(fig)


# ---------------------------------------------------------------- figure 3
def figure3(pdf, d):
    fig = plt.figure(figsize=PAGE)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.70],
                          hspace=0.26, wspace=0.14,
                          left=0.13, right=0.97, top=0.92, bottom=0.175)
    secs = {s["sample"]: s for s in d["sections"]}
    letters = "abcde"
    for i, (name, truth) in enumerate(SECTIONS):
        small, ov, nreg, p, tis = overlay(name)
        r, c = divmod(i, 3)
        # H&E + boundary
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(ov); ax.axis("off"); ax.grid(False)
        panel(ax, letters[i])
        a = secs[name]["area"]
        colr = TUMOUR_RED if truth == "tumour" else C3
        ax.set_title(f"{name}  ({truth})\n{a:.1%} tumour area, {nreg} region(s)",
                     fontsize=8.4, color=colr, pad=4)
        # inset probability map
        ins = ax.inset_axes([0.63, 0.02, 0.36, 0.36])
        ins.imshow(np.where(tis, p, np.nan), cmap="inferno", vmin=0, vmax=1)
        ins.set_xticks([]); ins.set_yticks([]); ins.grid(False)
        for sp in ins.spines.values():
            sp.set_color("white"); sp.set_linewidth(1.2)

    # (f) tumour area by section
    ax = fig.add_subplot(gs[1, 2]); panel(ax, "f", "Tumour area per section")
    names = [s for s, _ in SECTIONS]
    vals = [secs[s]["area"] for s in names]
    cols = [TUMOUR_RED if t == "tumour" else C3 for _, t in SECTIONS]
    bars = ax.bar(range(5), vals, color=cols, width=0.62)
    ax.set_xticks(range(5))
    ax.set_xticklabels([n.replace("Cancer_", "C").replace("Normal_", "N") for n in names],
                       fontsize=7.5)
    ax.set_ylabel("tumour area", fontsize=8); ax.set_ylim(0, 0.60)
    bar_labels(ax, bars, dy=0.012)
    ax.axhspan(0, 0.02, color=C3, alpha=0.10)
    ax.legend(handles=[Line2D([0], [0], marker="s", ls="", ms=7, color=TUMOUR_RED,
                              label="tumour section"),
                       Line2D([0], [0], marker="s", ls="", ms=7, color=C3,
                              label="normal control")],
              fontsize=6.8, frameon=False, loc="upper right")

    # (g) tissue composition per section
    ax = fig.add_subplot(gs[2, :]); panel(ax, "g", "Predicted tissue composition")
    keys = ["TUM", "STR", "NORM", "MUS", "MUC", "LYM", "DEB", "ADI", "BACK"]
    base = np.zeros(5)
    ramp = plt.get_cmap("Blues")(np.linspace(0.78, 0.18, len(keys) - 1))
    for j, k in enumerate(keys):
        vals = np.array([secs[s]["comp"].get(k, 0.0) for s, _ in SECTIONS])
        colr = TUMOUR_RED if k == "TUM" else ramp[j - 1]
        ax.barh(range(5), vals, left=base, color=colr, height=0.6,
                edgecolor="white", linewidth=1.1, label=k)
        for i, v in enumerate(vals):
            if v > 0.075:
                ax.text(base[i] + v / 2, i, k, ha="center", va="center",
                        fontsize=6.6,
                        color="white" if (k == "TUM" or j < 3) else INK)
        base += vals
    ax.set_yticks(range(5))
    ax.set_yticklabels([n for n, _ in SECTIONS], fontsize=7.8)
    ax.set_xlim(0, 1); ax.set_xlabel("fraction of tissue tiles")
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=6.4, ncol=9, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, -0.28))

    fig.suptitle("Figure 3.  Application to three colorectal tumours and two "
                 "matched normal controls", fontsize=11, fontweight="bold", y=0.975)
    fig.text(0.06, 0.030,
             "(a–e) H&E with the predicted tumour region outlined and tinted; inset is the "
             "per-tile probability map. (f) Tumour sections occupy 27.7–48.5% of\n"
             "tissue while normal controls stay below 2% (shaded band) and export no "
             "regions at all. (g) Predicted tissue composition: TUM dominates the\n"
             "tumour sections and is near-absent from the controls, where normal mucosa, "
             "muscle and mucus dominate.",
             fontsize=7.3, color=INK2, va="top")
    pdf.savefig(fig); plt.close(fig)


# ---------------------------------------------------------------- figure 4
def figure4(pdf, d):
    fig = plt.figure(figsize=PAGE)
    gs = fig.add_gridspec(3, 2, hspace=0.46, wspace=0.30,
                          left=0.10, right=0.96, top=0.93, bottom=0.165)
    t, mol = d["tiles"], d["mol"]
    y = (t.mol_tumor_frac >= 0.5).to_numpy().astype(int)
    s = t.p_tum.to_numpy()

    # (a,b) side-by-side spatial maps
    p, tis, xs, ys, tile, _, _ = hm("Cancer_P1")
    Mm = np.full(tis.shape, np.nan); Mm[t.row, t.col] = t.mol_tumor_frac
    Pp = np.full(tis.shape, np.nan); Pp[t.row, t.col] = t.p_tum
    for k, (arr, ttl, lt) in enumerate([
            (Mm, "Transcriptomic tumour fraction\n(Visium HD cell types)", "a"),
            (Pp, "H&E model P(tumour)\n(never sees expression)", "b")]):
        ax = fig.add_subplot(gs[0, k])
        im = ax.imshow(arr, cmap="inferno", vmin=0, vmax=1)
        ax.axis("off"); ax.grid(False); panel(ax, lt, ttl)
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cb.ax.tick_params(labelsize=6.5)

    # (c) tile-level agreement
    ax = fig.add_subplot(gs[1, 0]); panel(ax, "c", "Tile-level agreement")
    ax.scatter(t.mol_tumor_frac, t.p_tum, s=6, alpha=0.22, color=C1,
               edgecolors="none")
    ax.set_xlabel("transcriptomic tumour-cell fraction")
    ax.set_ylabel("H&E P(tumour)")
    ax.text(0.04, 0.93, f"Spearman ρ = {mol['spearman_rho']:.3f}\n"
                        f"n = {mol['n_tiles']:,} tiles",
            transform=ax.transAxes, fontsize=8, va="top", color=INK)

    # (d) ROC
    ax = fig.add_subplot(gs[1, 1]); panel(ax, "d", "Discrimination")
    fpr, tpr, _ = roc_curve(y, s)
    ax.plot(fpr, tpr, lw=2.2, color=C3)
    ax.plot([0, 1], [0, 1], lw=1, ls="--", color=MUTED)
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.text(0.40, 0.18, f"AUROC\n{mol['auroc_vs_molecular']:.4f}",
            transform=ax.transAxes, fontsize=13, fontweight="bold", color=C3)

    # (e) operating point
    ax = fig.add_subplot(gs[2, 0]); panel(ax, "e", f"Operating point (threshold {THR})")
    names = ["precision", "recall\n(sensitivity)", "specificity"]
    vals = [mol["precision"], mol["recall"], mol["specificity"]]
    bars = ax.bar(names, vals, color=[C1, TUMOUR_RED, C3], width=0.58)
    ax.set_ylim(0, 1.13); ax.set_ylabel("value")
    for r in bars:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.02,
                f"{r.get_height():.3f}", ha="center", fontsize=9, color=INK2)
    ax.tick_params(axis="x", labelsize=7.6)

    # (f) agreement across model variants
    ax = fig.add_subplot(gs[2, 1]); panel(ax, "f", "AUROC across variants")
    rows = []
    for m in d["molvar"]:
        f = Path(R / "molecular")
        rows.append((m.get("variant", ""), m["auroc_vs_molecular"]))
    labs, vals = [], []
    for p_ in sorted((R / "molecular").glob("*/Cancer_P1_molecular_validation.json")):
        j = json.loads(p_.read_text())
        labs.append(VARIANT_LABEL.get(p_.parent.name, p_.parent.name))
        vals.append(j["auroc_vs_molecular"])
    order = np.argsort(vals)
    labs = [labs[i] for i in order]; vals = [vals[i] for i in order]
    cols = [C3 if i == len(vals) - 1 else C1 for i in range(len(vals))]
    bars = ax.barh(labs, vals, color=cols, height=0.62)
    ax.set_xlim(0.93, 1.0); ax.set_xlabel("AUROC vs transcriptomic truth")
    ax.tick_params(axis="y", labelsize=6.8)
    for r in bars:
        ax.text(r.get_width() + 0.0015, r.get_y() + r.get_height() / 2,
                f"{r.get_width():.3f}", va="center", fontsize=6.8, color=INK2)
    ax.grid(axis="y", visible=False)

    fig.suptitle("Figure 4.  Validation against an independent molecular modality "
                 "(Cancer_P1)", fontsize=11, fontweight="bold", y=0.975)
    fig.text(0.07, 0.105,
             "202,182 Visium HD cells were assigned transcriptomic identities and projected "
             "onto the tile grid; 2,307 tiles had ≥20 cells. The image\n"
             "model never observes gene expression, so agreement is orthogonal evidence "
             "rather than a morphology-vs-morphology comparison. (a,b) The two\n"
             "maps localise the same compartment. (c,d) Tile-level agreement and "
             "discrimination. (e) At the released threshold. (f) Every variant, ranked;\n"
             "the released ensemble is highest. The 0.4 operating point was selected on "
             "this same section, so (e) is optimistic; AUROC is threshold-free.",
             fontsize=7.3, color=INK2, va="top")
    pdf.savefig(fig); plt.close(fig)


# ---------------------------------------------------------------- text pages
def text_page(pdf, blocks, header=None):
    fig = plt.figure(figsize=PAGE)
    fig.patch.set_facecolor("white")
    y = 0.955
    if header:
        fig.text(0.10, y, header, fontsize=7.4, color=MUTED)
        y -= 0.022
    for kind, txt in blocks:
        if kind == "title":
            fig.text(0.10, y, txt, fontsize=16, fontweight="bold",
                     va="top", wrap=True)
            y -= 0.052 + 0.022 * (len(txt) // 60)
        elif kind == "sub":
            fig.text(0.10, y, txt, fontsize=10.5, color=INK2, va="top")
            y -= 0.030
        elif kind == "h":
            y -= 0.012
            fig.text(0.10, y, txt, fontsize=11, fontweight="bold", va="top")
            y -= 0.028
        elif kind == "p":
            n = fig.text(0.10, y, txt, fontsize=8.7, va="top", color=INK,
                         linespacing=1.55, wrap=True)
            y -= 0.0165 * (txt.count("\n") + 1) + 0.010
        elif kind == "mono":
            fig.text(0.10, y, txt, fontsize=7.8, va="top", color=INK,
                     family="monospace", linespacing=1.45)
            y -= 0.0152 * (txt.count("\n") + 1) + 0.012
        elif kind == "gap":
            y -= float(txt)
    pdf.savefig(fig); plt.close(fig)


def W(text, width=104):
    """Wrap a paragraph to a fixed column width."""
    import textwrap
    return "\n".join(textwrap.fill(p.strip(), width)
                     for p in text.strip().split("\n\n"))


def page_title(pdf, d):
    mol, secs = d["mol"], {s["sample"]: s for s in d["sections"]}
    tum = [secs[s]["area"] for s, t in SECTIONS if t == "tumour"]
    nor = [secs[s]["area"] for s, t in SECTIONS if t == "normal"]
    text_page(pdf, [
        ("title", "OmiCoreTumor: an open, molecularly validated model for mapping\n"
                  "tumour regions in colorectal cancer H&E sections"),
        ("sub", "OmiCore Inc."),
        ("gap", "0.012"),
        ("p", W("Preprint. Research use only — not a medical device and not "
                "validated for clinical use.")),
        ("h", "Abstract"),
        ("p", W(f"""
Selecting tumour regions on haematoxylin and eosin (H&E) sections is a routine
prerequisite for spatial-omics experiments, yet it is usually done by hand and is
rarely reproducible between operators. We present OmiCoreTumor
(omicoretumor-crc-he-v0.1), an openly licensed model that maps tumour-enriched
regions in colorectal H&E and exports them directly as QuPath-readable
annotations.

The model is an ensemble of three convolutional classifiers trained on 100,000
openly licensed tissue tiles, combined with Macenko stain normalisation applied at
inference. We show that the dominant obstacle to deployment is not discrimination
but stain-domain shift: a single model retained an AUROC of 0.955 on unseen
slides while its recall at the conventional 0.5 threshold fell to 0.48. Training
without stain normalisation raised cross-cohort tumour AUROC from 0.836 to 0.992,
and normalising at inference reduced false-positive area on tumour-free tissue by
up to 27-fold. Combining both recovers recall.

On five 10x Visium HD colorectal sections that are independent of all training
data, the released model called {np.mean(tum):.0%} of tissue as tumour in three
carcinoma sections ({min(tum):.1%}–{max(tum):.1%}) and under {max(nor):.1%}
in two matched normal controls, exporting no tumour regions at all from either
control. On the one section with matched transcriptomics, agreement with
Visium HD cell-type labels reached AUROC {mol['auroc_vs_molecular']:.4f}
(precision {mol['precision']:.3f}, sensitivity {mol['recall']:.3f}, specificity
{mol['specificity']:.3f}). Because the image model never observes gene
expression, this is orthogonal evidence rather than a comparison between two
morphology-derived annotations.

The model, weights, training code and evaluation are released under Apache-2.0 as
a pip-installable package, so that any group can map tumour regions on their own
colorectal sections with three lines of Python. We are explicit about what the
evidence does and does not support: the model localises tumour-enriched regions
at 112 um resolution across five sections from one cohort; it does not identify
malignant cells, and it has not been shown to generalise across scanners,
histological variants or difficult non-neoplastic tissue.
""")),
        ("h", "Availability"),
        ("mono", "pip install omicoretumor\n"
                 "github.com/OmiCore-Japan/omicoretumordetector    Apache-2.0 (code)\n"
                 "Model: omicoretumor-crc-he-v0.1"),
    ])


def page_intro_results(pdf, d):
    mol = d["mol"]
    secs = {s["sample"]: s for s in d["sections"]}
    bench = {b["tag"]: b for b in d["bench"]}
    text_page(pdf, [
        ("h", "Introduction"),
        ("p", W("""
Spatial transcriptomics experiments on solid tumours almost always begin with a
morphological question: which part of this section is tumour? The answer
determines which regions are profiled, how compartments are compared, and how
results are interpreted. In practice the region is drawn by hand, which is slow,
operator-dependent and difficult to report reproducibly in a methods section.

Tile-level tissue classifiers for colorectal H&E have existed since the release of
the NCT-CRC-HE-100K dataset, and reach very high accuracy on held-out tiles from
the same cohort. That accuracy has not translated into tools that groups
routinely apply to their own slides. We found the reason is mundane but decisive:
the probability calibration of these models does not survive a change of stain
and scanner, so a model that still ranks tiles correctly will silently miss most
of the tumour when its usual threshold is applied to a new laboratory's sections.

We therefore treated robustness, not headline accuracy, as the design target, and
validated the result against a modality the model cannot see.
""")),
        ("h", "Results"),
        ("p", W(f"""
Stain-domain shift, not discrimination, limits transfer (Fig. 1, Fig. 2).
Three classifiers were trained on 100,000 tiles spanning nine tissue classes.
On CRC-VAL-HE-7K, a held-out cohort processed identically to the training data,
the normalisation-trained model reached
{bench['A_convnext_norm__ext7k']['acc']:.3f} accuracy and
{bench['A_convnext_norm__ext7k']['tumor_auroc']:.3f} tumour AUROC. On
Kather-2016, a different cohort and scanner, the same model fell to
{bench['A_convnext_norm__ext2016']['tumor_auroc']:.3f} tumour AUROC, whereas an
identically trained model that had seen non-normalised tiles reached
{bench['B_convnext_nonorm__ext2016']['tumor_auroc']:.3f} (Fig. 2c). Measuring the
colour statistics of real target slides showed why: they sit a mean RGB distance
of {d['color']['raw']['dist']:.0f} from the training domain, which Macenko
normalisation reduces to {d['color']['macenko']['dist']:.0f} (Fig. 2a,b).

The practical consequence is a calibration failure rather than a ranking failure.
A single model applied to raw target slides kept essentially unchanged
discrimination (AUROC 0.955) while its recall at the conventional 0.5 threshold
fell to 0.48 (Fig. 2e). The two molecular classes remained well separated, but
the probabilities had compressed toward zero (Fig. 2d). Any deployment that
inherits a threshold from the training domain will therefore appear highly
precise while discarding much of the tumour.

Combining non-normalised training with inference-time normalisation addresses
both halves of the problem, and this variant ranked first of seven on the
independent section-level criterion (Fig. 1d).
""")),
        ("p", W(f"""
Performance on tumour and control sections (Fig. 3). Applied to five Visium HD
colorectal sections, the released model called
{secs['Cancer_P1']['area']:.1%}, {secs['Cancer_P2']['area']:.1%} and
{secs['Cancer_P5']['area']:.1%} of tissue as tumour in the three carcinoma
sections, against {secs['Normal_P3']['area']:.2%} and
{secs['Normal_P5']['area']:.2%} in the two matched normal controls. After the
minimum-area filter used for annotation export, neither control produced a single
tumour region. The predicted tissue composition is biologically coherent
(Fig. 3g): tumour epithelium and desmoplastic stroma dominate the carcinoma
sections, while normal mucosa, smooth muscle and mucus dominate the controls.

A negative control is necessary but not sufficient, since a model that predicted
'no tumour' everywhere would also pass it. The evidence rests on both sides
holding simultaneously, which they do.
""")),
        ("p", W(f"""
Agreement with an independent modality (Fig. 4). For Cancer_P1, 202,182
segmented cells carried transcriptomically assigned identities from the matched
Visium HD assay. Projecting these onto the tile grid gave {mol['n_tiles']:,}
tiles with at least 20 cells, of which {mol['molecular_positive_tiles']:,} were
molecularly tumour-dominant. The H&E model, which never observes gene expression,
agreed with these labels at AUROC {mol['auroc_vs_molecular']:.4f}
(Spearman rho {mol['spearman_rho']:.3f}), with precision {mol['precision']:.3f},
sensitivity {mol['recall']:.3f} and specificity {mol['specificity']:.3f} at the
released threshold. Every model variant was scored the same way and the released
ensemble ranked highest (Fig. 4f).
""")),
    ], header="OmiCoreTumor — preprint")


def page_methods(pdf, d):
    text_page(pdf, [
        ("h", "Methods"),
        ("p", W("""
Data. All training data is openly licensed under CC-BY-4.0. NCT-CRC-HE-100K
provided 100,000 224x224 px tiles at 0.5 um/px across nine classes (ADI, BACK,
DEB, LYM, MUC, MUS, NORM, STR, TUM), used in both its stain-normalised and
non-normalised releases; CRC-VAL-HE-7K (7,180 tiles, separate cohort) and the
Kather-2016 collection (4,375 usable tiles after class mapping) served as held-out
test sets. EBHI-SEG supplied 2,228 image/mask pairs for an auxiliary segmentation
model. The five evaluation sections are 10x Visium HD colorectal FFPE samples and
share no material with any training set.

An important caveat applies to the internal split: NCT-CRC-HE-100K ships no
patient identifiers, and the code embedded in each filename is a per-tile hash
that we verified to be unique for every tile. A random split therefore probably
places tiles from the same patient on both sides, so the internal validation
figure is optimistic and we quote only held-out cohorts.
""")),
        ("p", W("""
Models. Three classifiers were fine-tuned for 8 epochs with AdamW, one-cycle
scheduling, label smoothing 0.1 and class-weighted cross-entropy, in bfloat16:
ConvNeXt-Tiny on the normalised tiles, and ConvNeXt-Tiny and EfficientNetV2-S on
the non-normalised tiles. Augmentation targeted stain robustness (random resized
crop, dihedral flips and 90-degree rotations, colour jitter, random erasing). The
released model averages the three softmax outputs. All pretrained backbones are
Apache-2.0.
""")),
        ("p", W("""
Inference. Images are rescaled so that one pixel is 0.5 um, tiled at 224 px,
and tiles whose saturation indicates glass rather than tissue are skipped. Each
tile is Macenko-normalised toward a fixed H&E target before classification. The
per-tile tumour probability is thresholded at 0.4, cleaned with a morphological
close-open, and traced to polygons that are written as GeoJSON in the source
image's coordinate frame. A 13,000 x 13,000 px section takes about 50 s on one
RTX 5080, including normalisation.
""")),
        ("p", W("""
Molecular validation. Visium HD cells were mapped into image coordinates using
the recorded bin2cell geometry: image pixels equal the cropped spatial
coordinates multiplied by the recorded scale factor, which we confirmed equals
the source resolution divided by 0.5 um. The cell-identifier join matched
202,182 of 202,182 cells, and every cell fell inside the scored grid. Tiles with
at least 20 cells were scored, a tile counting as molecularly positive when at
least half its cells carried a tumour identity.
""")),
        ("p", W("""
Reproducibility. Training, evaluation and figure code ship with the package.
run_experiments.sh reproduces the full matrix - three models, seven inference
variants, two held-out benchmarks, the section-level dry test and the molecular
comparison - and is resumable. The development log, including approaches we
tested and rejected, is in FINDINGS.md.
""")),
    ], header="OmiCoreTumor — preprint")


def page_discussion(pdf, d):
    mol = d["mol"]
    text_page(pdf, [
        ("h", "Discussion"),
        ("p", W("""
The useful finding here is not that a convolutional ensemble can classify
colorectal tissue; that has been established for years. It is that the barrier to
reuse is calibration rather than capability. A model whose AUROC is unchanged can
lose half its sensitivity on a new laboratory's slides purely because the
probability scale has shifted, and it will do so quietly, presenting as unusually
high precision. Any group deploying a tile classifier on its own material should
verify the operating point before trusting the output, and we ship the tools to
do so rather than only a threshold.

We deliberately did not adopt two techniques that appeared promising. Spatial
smoothing of the probability map, motivated by the contiguity of tumour, improved
AUROC by 0.002 while reducing recall in every configuration tested. Label-free
automatic thresholding recovered recall on tumour-bearing slides but fabricated
disease on tumour-free tissue, calling 39% of a lung section colorectal tumour;
it is retained only behind an explicit opt-in. Both are documented with their
numbers so that others need not repeat them.
""")),
        ("h", "Limitations"),
        ("p", W(f"""
The evaluation is small. Three tumour and two normal sections from a single
cohort demonstrate feasibility, not generalisation, and matched transcriptomics
were available for one section only. The reported precision and sensitivity at
threshold 0.4 are optimistic because that threshold was chosen on the same
section; the AUROC of {mol['auroc_vs_molecular']:.4f} is threshold-free and
unaffected.

The model marks regions, not cells. Boundaries are resolved at 224 px, or 112 um,
so a called region necessarily contains stroma, immune and vascular cells
alongside malignant glands, and small infiltrating glands at a margin may be
missed. Claims of the form 'identifies malignant cells' are not supported by this
evidence; claims of the form 'localises tumour-enriched regions' are.

The model is colorectal-specific. Lung and prostate sections were used as
out-of-organ negative controls and produced 8.9% and 1.7% spurious tumour area at
the released threshold. Adenoma, dysplasia, inflammation, ulceration, necrosis,
mucinous variants and treatment-altered tissue were not represented in
evaluation, and a hard-negative analysis on such tissue is the natural next step.
""")),
        ("h", "What would strengthen this"),
        ("p", W("""
In order of value: independent pathologist annotation of invasive carcinoma on
the same sections, scored with Dice and boundary distance rather than tile
metrics; molecular validation extended to the remaining four sections, which
requires only re-running an existing annotation pipeline; and external validation
across 20-50 unseen patients spanning scanners and histological variants,
including difficult non-neoplastic tissue.
""")),
        ("h", "Data and code availability"),
        ("mono", "Package     pip install omicoretumor\n"
                 "Source      github.com/OmiCore-Japan/omicoretumordetector  (Apache-2.0)\n"
                 "Model       omicoretumor-crc-he-v0.1\n"
                 "Training    NCT-CRC-HE-100K  doi:10.5281/zenodo.1214456  CC-BY-4.0\n"
                 "            Kather-2016      doi:10.5281/zenodo.53169    CC-BY-4.0\n"
                 "            EBHI-SEG         doi:10.6084/m9.figshare.21540159  CC-BY-4.0\n"
                 "Evaluation  10x Visium HD colorectal FFPE (Oliveira et al. 2025)"),
        ("h", "References"),
        ("p", W("""
Kather JN, Halama N, Marx A. 100,000 histological images of human colorectal
cancer and healthy tissue. Zenodo (2018). doi:10.5281/zenodo.1214456

Kather JN, Weis C-A, Bianconi F, et al. Multi-class texture analysis in
colorectal cancer histology. Sci Rep 6, 27988 (2016).

Macenko M, Niethammer M, Marron JS, et al. A method for normalizing histology
slides for quantitative analysis. ISBI (2009).

Shi L, Li C, Hu W, et al. EBHI-Seg: a novel enteroscope biopsy histopathological
H&E image dataset for image segmentation tasks. Front Med (2023).

Oliveira MF, Romero JP, Chung M, et al. High-definition spatial transcriptomic
profiling of immune cell populations in colorectal cancer. Nat Genet (2025).
""")),
    ], header="OmiCoreTumor — preprint")


def main():
    d = load()
    out = R / "OmiCoreTumor_preprint.pdf"
    with PdfPages(out) as pdf:
        page_title(pdf, d)
        page_intro_results(pdf, d)
        figure1(pdf, d)
        figure2(pdf, d)
        figure3(pdf, d)
        figure4(pdf, d)
        page_methods(pdf, d)
        page_discussion(pdf, d)
        i = pdf.infodict()
        i["Title"] = ("OmiCoreTumor: an open, molecularly validated model for mapping "
                      "tumour regions in colorectal cancer H&E sections")
        i["Author"] = "OmiCore Inc."
        i["Subject"] = "Computational pathology preprint"
    print(f"wrote {out}  ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
