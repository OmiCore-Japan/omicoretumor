"""Validate H&E tumor calls against transcriptomic cell-type labels (Cancer_P1).

The H&E model never sees gene expression, so Visium HD cell-type labels are an
independent modality -- agreement here is much stronger evidence of real tumor
localisation than any image-only metric.

Coordinate chain (verified against the bin2cell outputs):
    image_px = (obsm['spatial'] - offset) * tissue_0.5_mpp_150_buffer_scalef
where offset = spatial - spatial_cropped_150_buffer (constant) and the scalef
equals source microns_per_pixel / 0.5.
"""
import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve


def read_obs(f, key):
    o = f["obs"][key]
    if isinstance(o, h5py.Group):
        cats, codes = o["categories"][:], o["codes"][:]
        return np.array([cats[c] for c in codes])
    return o[:]


def load_cells(seg_h5ad):
    with h5py.File(seg_h5ad, "r") as f:
        cropped = f["obsm/spatial_cropped_150_buffer"][:]
        oid = read_obs(f, "object_id")
        scalef = None
        g = f["uns/spatial"]
        lib = list(g.keys())[0]
        for k, v in g[lib]["scalefactors"].items():
            if "0.5_mpp" in k:
                scalef = float(v[()])
    if scalef is None:
        raise SystemExit("no tissue_0.5_mpp_150_buffer_scalef in uns")
    return pd.DataFrame({"cell_id": np.asarray(oid).astype(str),
                         "x_px": cropped[:, 0] * scalef,
                         "y_px": cropped[:, 1] * scalef}), scalef


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="Cancer_P1")
    ap.add_argument("--st-root", default="/home/shamim/Documents/spatial_transcriptomics/results")
    ap.add_argument("--heatmap", default="reports/dry_test/Cancer_P1_heatmap.npz")
    ap.add_argument("--labels", default=None, help="labels_lr.parquet")
    ap.add_argument("--out", default="reports/molecular")
    ap.add_argument("--min-cells", type=int, default=20,
                    help="min cells per tile to score it")
    ap.add_argument("--tumor-frac", type=float, default=0.5,
                    help="molecular tumor fraction defining a positive tile")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="P(TUM) operating point; must match the threshold "
                         "used for the masks/areas being reported alongside")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.st_root) / args.sample
    labels_path = args.labels or root / "annotation" / "labels_lr.parquet"

    cells, scalef = load_cells(root / "segmentation" / "cells.h5ad")
    lab = pd.read_parquet(labels_path)[["cell_id", "lineage", "prob", "margin"]]
    lab["cell_id"] = lab["cell_id"].astype(str)
    df = cells.merge(lab, on="cell_id", how="inner")
    print(f"{args.sample}: {len(df):,} cells with coords + molecular labels "
          f"(scalef={scalef:.6f})")

    hm = np.load(args.heatmap, allow_pickle=True)
    prob, tissue = hm["prob"].astype(np.float32), hm["tissue"]
    classes = [str(c) for c in hm["classes"]]
    xs, ys, tile = hm["xs"], hm["ys"], int(hm["tile"])
    ti = classes.index("TUM")

    # Assign each cell to the grid tile whose extent contains it.
    col = np.searchsorted(xs, df["x_px"].to_numpy(), side="right") - 1
    row = np.searchsorted(ys, df["y_px"].to_numpy(), side="right") - 1
    ok = ((col >= 0) & (col < len(xs)) & (row >= 0) & (row < len(ys))
          & (df["x_px"].to_numpy() < xs[np.clip(col, 0, len(xs) - 1)] + tile)
          & (df["y_px"].to_numpy() < ys[np.clip(row, 0, len(ys) - 1)] + tile))
    df, col, row = df[ok], col[ok], row[ok]
    print(f"  {len(df):,} cells fall inside the scored grid")

    df = df.assign(row=row, col=col, is_tum=(df["lineage"] == "Tumor").astype(int))
    grp = df.groupby(["row", "col"]).agg(n_cells=("is_tum", "size"),
                                         mol_tumor_frac=("is_tum", "mean")).reset_index()
    grp["p_tum"] = prob[grp["row"], grp["col"], ti]
    grp["has_tissue"] = tissue[grp["row"], grp["col"]] > 0
    g = grp[(grp.n_cells >= args.min_cells) & grp.has_tissue].copy()
    print(f"  {len(g):,} tiles with >={args.min_cells} cells and predicted tissue")

    y = (g["mol_tumor_frac"] >= args.tumor_frac).astype(int).to_numpy()
    s = g["p_tum"].to_numpy()
    rho, pv = spearmanr(g["mol_tumor_frac"], g["p_tum"])
    res = {"sample": args.sample, "threshold": args.threshold,
           "n_tiles": int(len(g)),
           "n_cells": int(len(df)), "min_cells": args.min_cells,
           "molecular_positive_tiles": int(y.sum()),
           "spearman_rho": float(rho), "spearman_p": float(pv)}
    if 0 < y.sum() < len(y):
        res["auroc_vs_molecular"] = float(roc_auc_score(y, s))
        pred = (s >= args.threshold).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
        res.update(tp=tp, fp=fp, fn=fn, tn=tn,
                   precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                   specificity=tn / max(tn + fp, 1))

    print(json.dumps(res, indent=2))
    (out / f"{args.sample}_molecular_validation.json").write_text(json.dumps(res, indent=2))
    g.to_csv(out / f"{args.sample}_tile_agreement.csv", index=False)

    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
    ax[0].scatter(g["mol_tumor_frac"], g["p_tum"], s=5, alpha=0.25)
    ax[0].set_xlabel("molecular tumor-cell fraction")
    ax[0].set_ylabel("H&E P(TUM)")
    ax[0].set_title(f"tile agreement  rho={rho:.3f}")
    M = np.full(tissue.shape, np.nan)
    M[g["row"], g["col"]] = g["mol_tumor_frac"]
    P = np.full(tissue.shape, np.nan)
    P[g["row"], g["col"]] = g["p_tum"]
    for a, img, t in [(ax[1], M, "molecular tumor fraction"),
                      (ax[2], P, "H&E P(TUM)")]:
        im = a.imshow(img, cmap="inferno", vmin=0, vmax=1)
        a.set_title(t); a.axis("off")
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out / f"{args.sample}_molecular_vs_he.png", dpi=130)
    plt.close(fig)

    if "auroc_vs_molecular" in res:
        fpr, tpr, _ = roc_curve(y, s)
        fig, a = plt.subplots(figsize=(5.5, 5))
        a.plot(fpr, tpr, lw=2, label=f"AUROC={res['auroc_vs_molecular']:.4f}")
        a.plot([0, 1], [0, 1], "k--", lw=1)
        a.set_xlabel("FPR"); a.set_ylabel("TPR")
        a.set_title(f"{args.sample}: H&E vs molecular truth")
        a.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(out / f"{args.sample}_roc_molecular.png", dpi=130)
        plt.close(fig)
    print(f"wrote {out}/")


if __name__ == "__main__":
    main()
