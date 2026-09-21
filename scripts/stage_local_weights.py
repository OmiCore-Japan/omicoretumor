"""Install locally trained checkpoints into the package cache.

Use this to test the package before publishing weights as release assets.

    python scripts/stage_local_weights.py ../models
"""
import sys
from pathlib import Path

from omicoretumordetector import install_local

MAP = {
    "omicore-tumordetector-crc-he-convnext-norm-v0.1": "A_convnext_norm/best.pt",
    "omicore-tumordetector-crc-he-convnext-nonorm-v0.1": "B_convnext_nonorm/best.pt",
    "omicore-tumordetector-crc-he-effnetv2-nonorm-v0.1": "C_effnet_nonorm/best.pt",
    "omicore-tumordetector-crc-he-seg-unet-v0.1": "seg_ebhi/best.pt",
}

root = Path(sys.argv[1] if len(sys.argv) > 1 else "models")
found = {k: root / v for k, v in MAP.items() if (root / v).exists()}
missing = sorted(set(MAP) - set(found))
if not found:
    raise SystemExit(f"no checkpoints found under {root}")
for p in install_local(found):
    print("staged", p)
if missing:
    print("missing (skipped):", ", ".join(missing))
