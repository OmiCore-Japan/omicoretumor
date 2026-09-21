"""Build CSV manifests from the extracted Kather CRC tile folders.

Directory layout expected:  <root>/<CLASS>/<image>.tif
Produces train/val (from NCT-CRC-HE-100K) and test (CRC-VAL-HE-7K) manifests.
"""
import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
TUMOR_CLASS = "TUM"

# Tile names look like TUM-TCGA-AAHIDTWA.tif. The trailing code is a per-tile
# hash, NOT a patient identifier -- verified unique for every tile in both
# releases -- so patient-level grouping is not recoverable from filenames.
# It is kept only as a stable tile ID.
_CODE = re.compile(r"^[A-Z]+-(?:TCGA-)?([A-Z0-9]+)\.")


def tile_code(name: str) -> str:
    m = _CODE.match(name)
    return m.group(1) if m else "UNKNOWN"


def index_dir(root: Path) -> pd.DataFrame:
    rows = []
    for cls in sorted(p.name for p in root.iterdir() if p.is_dir()):
        for img in sorted(root.joinpath(cls).glob("*.tif")):
            rows.append(
                {
                    "path": str(img.resolve()),
                    "label": cls,
                    "tile_id": tile_code(img.name),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"no .tif tiles found under {root}")
    df["label_idx"] = df["label"].map({c: i for i, c in enumerate(CLASSES)})
    if df["label_idx"].isna().any():
        bad = sorted(df.loc[df["label_idx"].isna(), "label"].unique())
        raise SystemExit(f"unexpected class folders: {bad}")
    df["label_idx"] = df["label_idx"].astype(int)
    df["is_tumor"] = (df["label"] == TUMOR_CLASS).astype(int)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-root", required=True, help="NCT-CRC-HE-100K dir")
    ap.add_argument("--test-root", required=True, help="CRC-VAL-HE-7K dir")
    ap.add_argument("--out", default="data/manifests")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    full = index_dir(Path(args.train_root))
    test = index_dir(Path(args.test_root))

    # Stratify on class. NOTE: the 100K release carries no reliable patient IDs,
    # so this internal split may share patients across train/val -- the honest
    # generalization number is the CRC-VAL-HE-7K external test set.
    train, val = train_test_split(
        full,
        test_size=args.val_frac,
        stratify=full["label"],
        random_state=args.seed,
    )

    for name, df in [("train", train), ("val", val), ("test", test)]:
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"{name:6s} n={len(df):7,d}  tumor={df.is_tumor.sum():6,d}  "
              f"tile_ids={df.tile_id.nunique():5,d}")

    overlap = set(full.tile_id) & set(test.tile_id)
    print(f"\n100K/7K tile-id overlap: {len(overlap)} (expect 0)")
    print("\nclass distribution (train):")
    print(train["label"].value_counts().reindex(CLASSES).to_string())


if __name__ == "__main__":
    main()
