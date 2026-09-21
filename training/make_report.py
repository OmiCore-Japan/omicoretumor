"""Build the PDF performance report with tumour boundaries drawn on the H&E."""
import argparse
import json
from datetime import date
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from PIL import Image

from crc_tumor_api import TumorDetector

Image.MAX_IMAGE_PIXELS = None
ST = "/home/shamim/Documents/spatial_transcriptomics/results"
SAMPLES = [("Cancer_P1", "tumour"), ("Cancer_P2", "tumour"), ("Cancer_P5", "tumour"),
           ("Normal_P3", "normal"), ("Normal_P5", "normal")]
DOWN = 8          # display downsample factor


def boundary_figure(pdf, name, truth, img, res, thr, stats):
    """One page: H&E, H&E with tumour outline, and the probability map."""
    W, H = img.size
    small = img.resize((W // DOWN, H // DOWN), Image.BILINEAR).convert("RGB")
    arr = np.asarray(small).copy()

    # Tile-grid mask -> display-resolution mask, then trace its outline.
    m = ((res["p_tumor"] >= thr) & res["tissue"]).astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)
    disp = np.zeros(arr.shape[:2], np.uint8)
    xs, ys, tile = res["xs"], res["ys"], res["tile"]
    for r in range(m.shape[0]):
        for c in range(m.shape[1]):
            if m[r, c]:
                y0, x0 = ys[r] // DOWN, xs[c] // DOWN
                disp[y0:y0 + tile // DOWN, x0:x0 + tile // DOWN] = 1
    cnts, _ = cv2.findContours(disp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) >= 50_000 / (DOWN ** 2)]
    # Translucent fill plus a heavy outline: the fill shows what was called,
    # the outline lets the underlying histology stay readable at the edge.
    outlined = arr.copy()
    if cnts:
        fill = np.zeros(arr.shape[:2], np.uint8)
        cv2.drawContours(fill, cnts, -1, 1, cv2.FILLED)
        tint = outlined.astype(np.float32)
        tint[fill == 1] = 0.75 * tint[fill == 1] + 0.25 * np.array([255, 40, 40], np.float32)
        outlined = tint.astype(np.uint8)
    cv2.drawContours(outlined, cnts, -1, (200, 0, 25), 5)

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 6.2))
    ax[0].imshow(arr)
    ax[0].set_title(f"{name} -- H&E  (ground truth: {truth})", fontsize=11)
    ax[1].imshow(outlined)
    ax[1].set_title(f"tumour boundary at P(TUM) >= {thr}  --  "
                    f"{len(cnts)} region(s)", fontsize=11)
    ax[1].legend(handles=[Line2D([0], [0], color=(200 / 255, 0, 25 / 255),
                                 lw=3, label="predicted tumour region")],
                 loc="lower right", fontsize=9, framealpha=0.9)
    hm = np.where(res["tissue"], res["p_tumor"], np.nan)
    im = ax[2].imshow(hm, cmap="inferno", vmin=0, vmax=1)
    ax[2].set_title("P(tumour) per 224 px tile", fontsize=11)
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    for a in ax:
        a.axis("off")
    verdict = ("correctly outlines tumour" if truth == "tumour" and stats["area"] > 0.05
               else "no tumour called -- correct" if truth == "normal" and stats["polygons"] == 0
               else "review")
    fig.suptitle(f"{name}   |   tumour area {stats['area']:.1%}   |   "
                 f"{stats['tiles']:,} tissue tiles   |   exported regions "
                 f"{stats['polygons']}   |   {verdict}",
                 fontsize=12.5, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, dpi=140)
    plt.close(fig)


def table_page(pdf, rows, thr, mol):
    fig = plt.figure(figsize=(16.5, 10.5))
    fig.text(0.5, 0.955, "CRC Tumour-Region Detection -- Performance Report",
             ha="center", fontsize=20, weight="bold")
    fig.text(0.5, 0.923,
             f"3-model ensemble + Macenko stain normalisation   |   "
             f"threshold {thr}   |   {date.today().isoformat()}",
             ha="center", fontsize=12, color="#444444")

    cols = ["section", "ground truth", "tissue tiles", "tumour area",
            "mean P(TUM)", "regions exported", "result"]
    cell = [[r["name"], r["truth"], f"{r['tiles']:,}", f"{r['area']:.2%}",
             f"{r['mean_p']:.3f}", str(r["polygons"]),
             "detected" if r["truth"] == "tumour" and r["area"] > 0.05
             else "clean" if r["truth"] == "normal" and r["polygons"] == 0
             else "review"] for r in rows]
    t = fig.add_axes([0.06, 0.60, 0.88, 0.26])
    t.axis("off")
    tab = t.table(cellText=cell, colLabels=cols, loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(11)
    tab.scale(1, 2.0)
    for j in range(len(cols)):
        tab[0, j].set_facecolor("#33475b")
        tab[0, j].set_text_props(color="w", weight="bold")
    for i, r in enumerate(rows, start=1):
        tab[i, 0].set_facecolor("#fdeaea" if r["truth"] == "tumour" else "#eaf5ea")
        tab[i, 6].set_text_props(weight="bold")

    tum = [r["area"] for r in rows if r["truth"] == "tumour"]
    nor = [r["area"] for r in rows if r["truth"] == "normal"]
    lines = [
        "Summary",
        f"   Tumour sections   mean tumour area {np.mean(tum):.1%}   (range {min(tum):.1%} - {max(tum):.1%})",
        f"   Normal sections   mean tumour area {np.mean(nor):.1%}   (range {min(nor):.1%} - {max(nor):.1%})",
        f"   Separation        {min(tum) - max(nor):+.1%}   (min tumour - max normal; > 0 = groups do not overlap)",
        f"   Negative control  {sum(1 for r in rows if r['truth']=='normal' and r['polygons']==0)}"
        f" of {len(nor)} healthy sections produced ZERO exported tumour regions",
        "",
        "Validation against transcriptomic ground truth (Cancer_P1, independent modality)",
    ]
    if mol:
        lines += [
            f"   {mol['n_tiles']:,} tiles scored against Visium HD cell-type labels "
            f"({mol.get('molecular_positive_tiles', 0):,} molecularly tumour-positive)",
            f"   AUROC {mol['auroc_vs_molecular']:.4f}  (threshold-free)",
            f"   At the same threshold {mol.get('threshold', '?')} used for the areas and boundaries above:",
            f"      precision {mol['precision']:.3f}    recall {mol['recall']:.3f}"
            f"    specificity {mol['specificity']:.3f}",
            "   The image model never sees gene expression, so this is an independent check.",
        ]
    lines += [
        "",
        "How to read this report",
        "   A negative control is necessary but not sufficient: a model predicting 'no tumour'",
        "   everywhere would also pass it. The evidence rests on both sides holding at once --",
        "   tumour found in the cancer sections AND nothing called in the healthy ones.",
        "   Boundaries are drawn at 224 px (112 um) tile resolution, so the red line is a",
        "   tumour-REGION estimate, not a histologically precise margin. Regions contain stroma,",
        "   immune and vascular cells as well as malignant glands.",
        "   The 0.4 operating point was chosen on this same Cancer_P1 molecular data, so the",
        "   precision/recall above are optimistic; the AUROC is threshold-free and unaffected.",
        "   Research pipeline, not a diagnostic device. Not validated for clinical use.",
    ]
    fig.text(0.06, 0.50, "\n".join(lines), fontsize=11.5, va="top", family="monospace")
    pdf.savefig(fig, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--out", default="../reports/CRC_tumour_detection_report.pdf")
    args = ap.parse_args()

    det = TumorDetector(["../models/A_convnext_norm/best.pt",
                         "../models/B_convnext_nonorm/best.pt",
                         "../models/C_effnet_nonorm/best.pt"], stain_norm=True)
    mol_path = Path("../reports/molecular/ENS_ABC_stainnorm/Cancer_P1_molecular_validation.json")
    mol = json.loads(mol_path.read_text()) if mol_path.exists() else None

    cached, rows = {}, []
    for name, truth in SAMPLES:
        img = Image.open(f"{ST}/{name}/segmentation/he_mpp0.5.tiff")
        res = det.predict(f"{ST}/{name}/segmentation/he_mpp0.5.tiff")
        p, t = res["p_tumor"], res["tissue"]
        n = det.to_geojson(res, f"../reports/{name}_tumor_regions.geojson",
                           threshold=args.threshold)
        st = {"area": float(((p >= args.threshold) & t).sum() / t.sum()),
              "tiles": int(t.sum()), "polygons": n, "mean_p": float(p[t].mean())}
        cached[name] = (img, res, st)
        rows.append({"name": name, "truth": truth, **st})
        print(f"{name:11s} area={st['area']:.2%} polygons={n}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        table_page(pdf, rows, args.threshold, mol)
        for name, truth in SAMPLES:
            img, res, st = cached[name]
            boundary_figure(pdf, name, truth, img, res, args.threshold, st)
        d = pdf.infodict()
        d["Title"] = "CRC Tumour-Region Detection -- Performance Report"
        d["Subject"] = "Ensemble + Macenko stain normalisation, 5 Visium HD sections"
    print(f"\nwrote {out}  ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
