"""Public API: load a model, map tumour regions, export masks and GeoJSON."""
import json
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
from PIL import Image

from .inference import heatmap
from .weights import resolve

Image.MAX_IMAGE_PIXELS = None

#: Resolution the models were trained at (microns per pixel).
TRAIN_MPP = 0.5


def auto_threshold(p_tumor, tissue, fallback=0.5):
    """Operating point from one slide's own P(TUM) distribution.

    !! Only for a slide already known to contain tumour. !! The method assumes
    a tumour mode exists; on tumour-free tissue it finds a threshold inside the
    noise and invents disease (measured: normal sections 12.6% vs 2.9% called
    area, lung 39.4% vs 5.6%). To decide *whether* a slide has tumour, use a
    fixed threshold.
    """
    try:
        from skimage.filters import threshold_triangle
    except ImportError as e:  # pragma: no cover
        raise ImportError("auto thresholding needs scikit-image") from e
    vals = p_tumor[tissue]
    vals = vals[np.isfinite(vals)]
    if vals.size < 50 or float(vals.max() - vals.min()) < 1e-6:
        return fallback
    return float(np.clip(threshold_triangle(vals), 1e-3, 0.95))


class TumorDetector:
    """Map tumour regions on a colorectal H&E image.

    RESEARCH USE ONLY -- not a diagnostic device.
    """

    def __init__(self, checkpoints, device=None, stain_norm=True, tta=False,
                 tumor_class="TUM", allow_auto_threshold=False):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        paths = resolve(checkpoints)
        if len(paths) == 1:
            ck = torch.load(paths[0], map_location="cpu", weights_only=False)
            m = timm.create_model(ck["arch"], pretrained=False,
                                  num_classes=len(ck["classes"]))
            m.load_state_dict(ck["model"])
            self.model = m.to(self.device, memory_format=torch.channels_last).eval()
            self.classes, self.tile = list(ck["classes"]), ck.get("size", 224)
        else:
            from .ensemble import ProbEnsemble
            self.model = ProbEnsemble([str(p) for p in paths], self.device)
            self.classes = list(self.model.classes)
            self.tile = self.model.size or 224
        if tumor_class not in self.classes:
            raise ValueError(f"{tumor_class!r} not in {self.classes}")
        self.ti = self.classes.index(tumor_class)
        self.tta = tta
        self.allow_auto_threshold = allow_auto_threshold
        self.stain = None
        if stain_norm:
            from .stain import macenko_normalize
            self.stain = macenko_normalize

    @classmethod
    def from_pretrained(cls, name=None, **kw):
        """Load released weights by name, downloading them on first use."""
        from .weights import DEFAULT_MODEL
        return cls(name or DEFAULT_MODEL, **kw)

    def predict(self, image, mpp=TRAIN_MPP, stride=None, batch_size=256):
        """Map tumour probability over a slide.

        Args:
            image: path or PIL image.
            mpp: microns per pixel of `image`. The models are trained at 0.5;
                anything else is rescaled to match, because magnification
                mismatch is the single biggest cause of silent failure.
            stride: tile step in pixels. Defaults to the tile size (no overlap);
                halve it for finer boundaries at ~4x the compute.
        """
        if mpp is None or not np.isfinite(mpp) or mpp <= 0:
            raise ValueError("mpp must be a positive number (microns per pixel)")
        img = Image.open(image) if isinstance(image, (str, Path)) else image
        scale = float(mpp) / TRAIN_MPP
        stride = stride or self.tile
        acc, seen, used, xs, ys = heatmap(
            self.model, img, self.classes, self.device, self.tile, stride,
            batch_size, scale, stain=self.stain, tta=self.tta)
        tissue = seen > 0
        if seen.size == 0:
            raise ValueError(
                f"image is {used.size[0]}x{used.size[1]} px after rescaling to "
                f"0.5 um/px, smaller than one {self.tile} px tile. Check that "
                f"mpp={mpp} is correct for this image.")
        p = acc[:, :, self.ti]
        return {
            "prob": acc, "p_tumor": p, "tissue": tissue, "classes": self.classes,
            "xs": xs, "ys": ys, "tile": self.tile, "stride": stride,
            # Geometry is in the RESCALED frame; `scale` maps it back to the
            # caller's original pixels so exported coordinates always match
            # the image they passed in.
            "scale": scale, "image_size": used.size,
            "source_size": img.size, "mpp": float(mpp),
            "n_tissue_tiles": int(tissue.sum()),
            "tumor_fraction": float(((p >= 0.5) & tissue).sum() / max(tissue.sum(), 1)),
        }

    def resolve_threshold(self, res, threshold):
        if isinstance(threshold, str):
            if threshold != "auto":
                raise ValueError('threshold must be a float or "auto"')
            if not self.allow_auto_threshold:
                raise ValueError(
                    'threshold="auto" assumes this slide contains tumour and '
                    "inflates false positives on tumour-free tissue. Construct "
                    "with allow_auto_threshold=True only to delineate a tumour "
                    "you have already confirmed is present.")
            t = auto_threshold(res["p_tumor"], res["tissue"])
        else:
            t = float(threshold)
        res["threshold_used"] = t
        return t

    def tumor_mask(self, res, threshold=0.4, full_res=True, smooth=3):
        """Binary mask on the tile grid, or upsampled to the source image."""
        threshold = self.resolve_threshold(res, threshold)
        m = ((res["p_tumor"] >= threshold) & res["tissue"]).astype(np.uint8)
        if smooth:
            k = np.ones((smooth, smooth), np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        if not full_res:
            return m
        W, H = res["source_size"]
        out = np.zeros((H, W), np.uint8)
        xs, ys, tile, s = res["xs"], res["ys"], res["tile"], res["scale"]
        for r in range(m.shape[0]):
            for c in range(m.shape[1]):
                if m[r, c]:
                    y0, x0 = int(ys[r] / s), int(xs[c] / s)
                    out[y0:y0 + int(tile / s), x0:x0 + int(tile / s)] = 1
        return out

    def to_geojson(self, res, path, threshold=0.4, min_area_px=50_000,
                   smooth=3, name="Tumor"):
        """Write tumour regions as GeoJSON polygons (QuPath-readable).

        Coordinates are in the source image's pixel space.
        """
        mask = self.tumor_mask(res, threshold, full_res=True, smooth=smooth)
        contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP,
                                          cv2.CHAIN_APPROX_SIMPLE)
        feats = []
        if hier is not None:
            hier = hier[0]
            for i, cnt in enumerate(contours):
                if hier[i][3] != -1 or cv2.contourArea(cnt) < min_area_px:
                    continue
                rings = [cnt.squeeze(1)]
                child = hier[i][2]
                while child != -1:
                    if cv2.contourArea(contours[child]) >= min_area_px / 10:
                        rings.append(contours[child].squeeze(1))
                    child = hier[child][0]
                coords = []
                for ring in rings:
                    if len(ring) < 3:
                        continue
                    pts = [[int(x), int(y)] for x, y in ring]
                    pts.append(pts[0])
                    coords.append(pts)
                if coords:
                    feats.append({
                        "type": "Feature",
                        "geometry": {"type": "Polygon", "coordinates": coords},
                        "properties": {
                            "objectType": "annotation",
                            "classification": {"name": name, "colorRGB": -3670016},
                            "area_px": float(cv2.contourArea(cnt)),
                            "model": "omicoretumordetector",
                            "threshold": res.get("threshold_used", threshold),
                        },
                    })
        Path(path).write_text(json.dumps(
            {"type": "FeatureCollection", "features": feats}))
        return len(feats)

    def tissue_composition(self, res):
        """Fraction of tissue tiles assigned to each class."""
        t = res["tissue"]
        n = max(int(t.sum()), 1)
        arg = res["prob"].argmax(2)
        return {c: float(((arg == i) & t).sum() / n)
                for i, c in enumerate(self.classes)}
