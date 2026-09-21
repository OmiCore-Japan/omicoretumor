"""Sliding-window inference over a large H&E image."""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

Image.MAX_IMAGE_PIXELS = None


def tile_grid(w, h, tile, stride):
    """Top-left corners of every tile, with the last one snapped to the edge.

    Returns empty lists when the image is smaller than one tile, since no
    full tile can be taken from it.
    """
    if w < tile or h < tile:
        return [], []
    xs = list(range(0, w - tile + 1, stride))
    ys = list(range(0, h - tile + 1, stride))
    if xs[-1] != w - tile:
        xs.append(w - tile)
    if ys[-1] != h - tile:
        ys.append(h - tile)
    return xs, ys


def is_background(patch_np, sat_thresh=0.10, frac=0.75):
    """Glass/empty slide has near-zero saturation -- skip it to save compute."""
    mx = patch_np.max(2).astype(np.float32)
    mn = patch_np.min(2).astype(np.float32)
    sat = np.where(mx > 0, (mx - mn) / np.clip(mx, 1, None), 0)
    return (sat < sat_thresh).mean() > frac


@torch.inference_mode()
def heatmap(model, img, classes, device, tile=224, stride=112, bs=128,
            mpp_scale=1.0, stain=None, tta=False):
    """stain: optional callable applied per tile (e.g. Macenko normalisation).
    tta: average logits over the 4 dihedral flips the tiles are agnostic to."""
    W, H = img.size
    if mpp_scale != 1.0:
        img = img.resize((int(W * mpp_scale), int(H * mpp_scale)), Image.BILINEAR)
        W, H = img.size
    arr = np.asarray(img.convert("RGB"))
    xs, ys = tile_grid(W, H, tile, stride)
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    acc = np.zeros((len(ys), len(xs), len(classes)), np.float32)
    seen = np.zeros((len(ys), len(xs)), np.float32)
    batch, coords = [], []

    def flush():
        if not batch:
            return
        x = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).float().div_(255)
        x = norm(x).to(device, memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if tta:
                # The model returns (B, n_classes) -- a class score, not a
                # spatial map -- so only the INPUT is flipped. Flipping the
                # output would index non-existent spatial dims.
                acc_l = model(x).float()
                for dims in ([2], [3], [2, 3]):
                    acc_l = acc_l + model(torch.flip(x, dims)).float()
                logit = acc_l / 4.0
            else:
                logit = model(x).float()
            p = F.softmax(logit, 1).cpu().numpy()
        for (r, c), pr in zip(coords, p):
            acc[r, c] = pr
            seen[r, c] = 1
        batch.clear()
        coords.clear()

    for r, y in enumerate(tqdm(ys, desc="rows")):
        for c, x0 in enumerate(xs):
            patch = arr[y:y + tile, x0:x0 + tile]
            if patch.shape[:2] != (tile, tile) or is_background(patch):
                continue
            batch.append(stain(patch) if stain is not None else patch)
            coords.append((r, c))
            if len(batch) >= bs:
                flush()
    flush()
    return acc, seen, img, np.asarray(xs), np.asarray(ys)
