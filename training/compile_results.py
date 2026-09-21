"""Aggregate every experiment into one leaderboard and pick an operating point.

Two questions it answers:
  1. Which model/variant separates tumour from normal sections best?
  2. What P(TUM) threshold should be used in production?

The threshold is chosen on the SAMPLE-level dry-test labels (Cancer_* vs
Normal_*), which is a 5-section sample -- treat the number as a starting
operating point, not a calibrated constant.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_dry_tests(root: Path) -> pd.DataFrame:
    frames = []
    for csv in sorted(root.rglob("dry_test_summary.csv")):
        d = pd.read_csv(csv)
        d["variant_dir"] = csv.parent.name
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_benchmarks(root: Path) -> pd.DataFrame:
    rows = []
    for j in sorted(root.rglob("metrics_*.json")):
        d = json.loads(j.read_text())
        d["file"] = str(j.relative_to(root))
        rows.append(d)
    return pd.DataFrame(rows)


def load_molecular(root: Path) -> pd.DataFrame:
    """Agreement with Visium HD cell-type labels -- the independent-modality check."""
    rows = []
    for j in sorted(root.rglob("*_molecular_validation.json")):
        d = json.loads(j.read_text())
        d["variant"] = j.parent.name
        rows.append(d)
    return pd.DataFrame(rows)


def sweep_threshold(root: Path, variant_dir: str):
    """Re-score saved heatmaps across thresholds to maximise tumour/normal gap."""
    d = root / variant_dir
    files = sorted(d.glob("*_heatmap.npz"))
    if not files:
        return None
    from dry_test import EXPECTED

    best = None
    for thr in np.arange(0.20, 0.96, 0.05):
        tum, nor = [], []
        for f in files:
            s = f.name.replace("_heatmap.npz", "")
            exp = EXPECTED.get(s)
            if exp not in ("tumor", "normal"):
                continue
            z = np.load(f, allow_pickle=True)
            classes = [str(c) for c in z["classes"]]
            if "TUM" not in classes:
                continue
            p = z["prob"][:, :, classes.index("TUM")].astype(np.float32)
            t = z["tissue"] > 0
            if t.sum() == 0:
                continue
            frac = float(((p >= thr) & t).sum() / t.sum())
            (tum if exp == "tumor" else nor).append(frac)
        if tum and nor:
            gap = min(tum) - max(nor)
            rec = {"variant": variant_dir, "threshold": round(float(thr), 2),
                   "mean_tumor_frac": float(np.mean(tum)),
                   "mean_normal_frac": float(np.mean(nor)),
                   "min_tumor": float(min(tum)), "max_normal": float(max(nor)),
                   "separation": float(gap)}
            if best is None or gap > best["separation"]:
                best = rec
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="../reports")
    ap.add_argument("--dry-root", default="../reports/dry_test")
    ap.add_argument("--out", default="../reports/LEADERBOARD.md")
    args = ap.parse_args()

    reports, dry_root = Path(args.reports), Path(args.dry_root)
    bench, dry = load_benchmarks(reports), load_dry_tests(dry_root)
    mol = load_molecular(reports)

    lines = ["# CRC tumour-detection: experiment leaderboard", ""]

    if not bench.empty:
        cols = [c for c in ["tag", "n", "acc", "balanced_acc", "macro_f1",
                            "tumor_auroc"] if c in bench.columns]
        lines += ["## Held-out benchmarks", "",
                  bench[cols].sort_values(cols[0]).to_markdown(index=False), ""]

    if not dry.empty:
        cols = [c for c in ["sample", "variant", "checkpoint", "expected",
                            "tissue_tiles", "tumor_frac", "mean_p_tum"]
                if c in dry.columns]
        lines += ["## Dry test on real Visium HD sections", "",
                  dry[cols].to_markdown(index=False), ""]

        rows = []
        for v in sorted(dry["variant_dir"].unique()):
            sub = dry[dry.variant_dir == v]
            t = sub[sub.expected == "tumor"]["tumor_frac"]
            n = sub[sub.expected == "normal"]["tumor_frac"]
            o = sub[sub.expected == "other_organ"]["tumor_frac"]
            if len(t) and len(n):
                rows.append({"variant": v, "mean_tumor": t.mean(),
                             "mean_normal": n.mean(),
                             "mean_other_organ": o.mean() if len(o) else np.nan,
                             "separation": t.min() - n.max()})
        if rows:
            summ = pd.DataFrame(rows).sort_values("separation", ascending=False)
            lines += ["## Variant ranking (by tumour/normal separation)", "",
                      summ.to_markdown(index=False), "",
                      "`separation` = min(tumour-section fraction) - "
                      "max(normal-section fraction); > 0 means the two groups "
                      "do not overlap at all.", ""]

        sweeps = [s for s in (sweep_threshold(dry_root, v)
                              for v in sorted(dry["variant_dir"].unique())) if s]
        if sweeps:
            lines += ["## Best operating threshold per variant", "",
                      pd.DataFrame(sweeps).sort_values(
                          "separation", ascending=False).to_markdown(index=False), "",
                      "Chosen on 5 sections only -- a starting operating point, "
                      "not a calibrated constant.", ""]

    if not mol.empty:
        cols = [c for c in ["variant", "sample", "n_tiles", "auroc_vs_molecular",
                            "spearman_rho", "precision", "recall", "specificity"]
                if c in mol.columns]
        lines += ["## Agreement with transcriptomic ground truth (independent modality)",
                  "", mol[cols].sort_values(
                      "auroc_vs_molecular", ascending=False).to_markdown(index=False), "",
                  "The H&E model never sees gene expression, so this is not a "
                  "circular check. Available for Cancer_P1 only -- the other "
                  "sections have no cell-type annotation.", ""]

    Path(args.out).write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
