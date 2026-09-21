"""Map the Kather-2016 8-class tiles onto the 9-class NCT scheme.

This is a SECOND external test set from a different cohort and a different
acquisition (150x150 @ 0.495 MPP vs 224x224 @ 0.5 MPP), so expect a real domain
gap -- it is a robustness probe, not a like-for-like benchmark.
"""
import argparse
from pathlib import Path

import pandas as pd

from build_manifest import CLASSES

# 03_COMPLEX is "stroma containing single tumor cells" -- genuinely mixed, with
# no clean counterpart in the 9-class scheme, so it is dropped rather than
# forced into TUM or STR.
MAP_2016 = {
    "01_TUMOR": "TUM",
    "02_STROMA": "STR",
    "03_COMPLEX": None,
    "04_LYMPHO": "LYM",
    "05_DEBRIS": "DEB",
    "06_MUCOSA": "NORM",
    "07_ADIPOSE": "ADI",
    "08_EMPTY": "BACK",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="data/manifests/test_2016.csv")
    args = ap.parse_args()

    rows, dropped = [], 0
    for folder, cls in MAP_2016.items():
        d = Path(args.root) / folder
        if not d.is_dir():
            raise SystemExit(f"missing folder {d}")
        imgs = sorted(d.glob("*.tif"))
        if cls is None:
            dropped += len(imgs)
            continue
        for img in imgs:
            rows.append({"path": str(img.resolve()), "label": cls,
                         "tile_id": img.stem, "src_class": folder})

    df = pd.DataFrame(rows)
    df["label_idx"] = df["label"].map({c: i for i, c in enumerate(CLASSES)}).astype(int)
    df["is_tumor"] = (df["label"] == "TUM").astype(int)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"wrote {out}  n={len(df):,}  (dropped {dropped:,} 03_COMPLEX tiles)")
    print(df["label"].value_counts().to_string())
    print("\nNOTE: MUC and MUS have no 2016 equivalent; those two classes are")
    print("absent here, so read per-class metrics for them as undefined.")


if __name__ == "__main__":
    main()
