"""Run a folder of sections and save a heatmap + boundary overlay for each."""
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from omicoretumor import TumorDetector

Image.MAX_IMAGE_PIXELS = None
folder = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
mpp = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
THR, DOWN = 0.4, 8

det = TumorDetector.from_pretrained()
out = Path("omicoretumor_out")
out.mkdir(exist_ok=True)

for img_path in sorted(list(folder.glob("*.tif")) + list(folder.glob("*.tiff"))):
    res = det.predict(img_path, mpp=mpp)
    area = float(((res["p_tumor"] >= THR) & res["tissue"]).sum()
                 / max(res["tissue"].sum(), 1))
    img = Image.open(img_path)
    W, H = img.size
    small = np.asarray(img.resize((W // DOWN, H // DOWN), Image.BILINEAR).convert("RGB"))

    full = det.tumor_mask(res, threshold=THR, full_res=True)
    disp = cv2.resize(full, (small.shape[1], small.shape[0]), cv2.INTER_NEAREST)
    cnts, _ = cv2.findContours(disp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = small.copy()
    cv2.drawContours(overlay, cnts, -1, (200, 0, 25), 3)

    fig, ax = plt.subplots(1, 3, figsize=(16, 6))
    ax[0].imshow(small); ax[0].set_title(img_path.stem)
    ax[1].imshow(overlay); ax[1].set_title(f"tumour boundary  ({area:.1%})")
    m = ax[2].imshow(np.where(res["tissue"], res["p_tumor"], np.nan),
                     cmap="inferno", vmin=0, vmax=1)
    ax[2].set_title("P(tumour)")
    fig.colorbar(m, ax=ax[2], fraction=0.046)
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(out / f"{img_path.stem}.png", dpi=130)
    plt.close(fig)
    print(f"{img_path.stem}: {area:.1%} tumour -> {out / (img_path.stem + '.png')}")
