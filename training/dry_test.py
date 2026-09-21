"""Dry-run a trained tile model over real Visium HD H&E sections.

Sample-level ground truth is the folder name (Cancer_* vs Normal_*), so this
measures whether tumor calls concentrate in the tumor sections and stay near
zero in normal-adjacent tissue -- i.e. specificity, not just accuracy.
Out-of-organ sections (Lung/Prostate) act as negative controls for a model
that is only supposed to recognise colorectal adenocarcinoma.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image

from predict_regions import heatmap

Image.MAX_IMAGE_PIXELS = None

# Expected truth per sample directory name; None = out-of-organ control.
EXPECTED = {
    "Cancer_P1": "tumor", "Cancer_P2": "tumor", "Cancer_P5": "tumor",
    "Normal_P3": "normal", "Normal_P5": "normal",
    "Lung_HD_Exp1": "other_organ", "Prostate_HD": "other_organ",
}


def load_model(ck_path, device):
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    model = timm.create_model(ck["arch"], pretrained=False,
                              num_classes=len(ck["classes"]))
    model.load_state_dict(ck["model"])
    return model.to(device, memory_format=torch.channels_last).eval(), ck


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", nargs="+", default=["models/run1/best.pt"],
                    help="one path, or several to ensemble")
    ap.add_argument("--st-root", default="/home/shamim/Documents/spatial_transcriptomics/results")
    ap.add_argument("--out", default="reports/dry_test")
    ap.add_argument("--samples", nargs="*", default=None)
    ap.add_argument("--stride", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--stain-norm", action="store_true",
                    help="Macenko-normalise each tile toward the NCT domain")
    ap.add_argument("--tta", action="store_true", help="4-way flip TTA")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    tag = args.tag or ("norm" if args.stain_norm else "raw") + ("_tta" if args.tta else "")
    out = Path(args.out) / tag
    out.mkdir(parents=True, exist_ok=True)
    print(f"variant: {tag}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if len(args.checkpoint) == 1:
        model, ck = load_model(args.checkpoint[0], device)
        classes = ck["classes"]
    else:
        from ensemble import ProbEnsemble
        model = ProbEnsemble(args.checkpoint, device)
        classes = model.classes
        print(f"ensemble of {len(args.checkpoint)} models")
    stain = None
    if args.stain_norm:
        from stain_norm import normalize as stain
        print("stain normalisation: ON")
    ti = classes.index("TUM") if "TUM" in classes else len(classes) - 1

    root = Path(args.st_root)
    samples = args.samples or sorted(
        d.name for d in root.iterdir()
        if (d / "segmentation" / "he_mpp0.5.tiff").exists()
    )

    rows = []
    for s in samples:
        img_path = root / s / "segmentation" / "he_mpp0.5.tiff"
        if not img_path.exists():
            print(f"skip {s}: no he_mpp0.5.tiff")
            continue
        print(f"\n=== {s} ===")
        img = Image.open(img_path)
        # he_mpp0.5.tiff is already 0.5 um/px, matching NCT training resolution.
        acc, seen, img, xs, ys = heatmap(model, img, classes, device, 224,
                                         args.stride, args.batch_size, 1.0,
                                         stain=stain, tta=args.tta)
        tissue = seen > 0
        n_tiles = int(tissue.sum())
        if n_tiles == 0:
            print(f"  no tissue detected")
            continue
        tum = acc[:, :, ti]
        pos = (tum >= args.threshold) & tissue

        # Composition over tissue tiles tells us WHAT it called instead of tumor.
        comp = {classes[c]: float(((acc.argmax(2) == c) & tissue).sum() / n_tiles)
                for c in range(len(classes))}
        rec = {
            "sample": s,
            "expected": EXPECTED.get(s, "unknown"),
            "tissue_tiles": n_tiles,
            "tumor_frac": float(pos.sum() / n_tiles),
            "mean_p_tum": float(tum[tissue].mean()),
            "p90_p_tum": float(np.percentile(tum[tissue], 90)),
            **{f"frac_{k}": v for k, v in comp.items()},
        }
        rows.append(rec)
        print(f"  tissue tiles {n_tiles:,}  tumor_frac {rec['tumor_frac']:.1%}  "
              f"mean P(TUM) {rec['mean_p_tum']:.3f}")
        print("  top calls: " + ", ".join(
            f"{k}={v:.1%}" for k, v in sorted(comp.items(), key=lambda x: -x[1])[:4]))

        np.savez_compressed(out / f"{s}_heatmap.npz", prob=acc.astype(np.float16),
                            tissue=seen.astype(np.uint8), tile=224,
                            classes=np.array(classes), stride=args.stride,
                            xs=xs, ys=ys)

        hm = np.where(tissue, tum, np.nan)
        fig, ax = plt.subplots(1, 3, figsize=(19, 6.5))
        ax[0].imshow(img.resize((img.size[0] // 6, img.size[1] // 6)))
        ax[0].set_title(f"{s} -- H&E ({EXPECTED.get(s,'?')})")
        m = ax[1].imshow(hm, cmap="inferno", vmin=0, vmax=1)
        ax[1].set_title(f"P(tumor)  mean={rec['mean_p_tum']:.3f}")
        fig.colorbar(m, ax=ax[1], fraction=0.046)
        ax[2].imshow(np.where(tissue, pos, np.nan), cmap="Reds", vmin=0, vmax=1)
        ax[2].set_title(f"P>={args.threshold}  area={rec['tumor_frac']:.1%}")
        for a in ax:
            a.axis("off")
        fig.tight_layout()
        fig.savefig(out / f"{s}_dry_test.png", dpi=120)
        plt.close(fig)

    if not rows:
        raise SystemExit("no samples processed")
    df = pd.DataFrame(rows)
    df.insert(1, "variant", tag)
    df.insert(2, "checkpoint", "|".join(Path(c).parent.name for c in args.checkpoint))
    df.to_csv(out / "dry_test_summary.csv", index=False)
    cols = ["sample", "expected", "tissue_tiles", "tumor_frac", "mean_p_tum", "p90_p_tum"]
    print("\n" + "=" * 70)
    print(df[cols].to_string(index=False))

    tum_s = df[df.expected == "tumor"]["tumor_frac"]
    nor_s = df[df.expected == "normal"]["tumor_frac"]
    if len(tum_s) and len(nor_s):
        print(f"\ntumor sections  mean tumor_frac = {tum_s.mean():.1%}")
        print(f"normal sections mean tumor_frac = {nor_s.mean():.1%}  <- want near 0")
        print(f"separation = {tum_s.min() - nor_s.max():+.1%} "
              f"(min tumor - max normal; >0 means fully separated)")
    (out / "dry_test_summary.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}/")


if __name__ == "__main__":
    main()
