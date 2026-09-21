"""OmiCoreTumorDetector -- tumour-region mapping for colorectal cancer H&E.

RESEARCH USE ONLY. Not a diagnostic device; not validated for clinical use.

    from omicoretumordetector import TumorDetector

    det = TumorDetector.from_pretrained("omicoretumor-crc-he-v0.1")
    res = det.predict("slide.tiff", mpp=0.5)
    det.to_geojson(res, "tumor_regions.geojson", threshold=0.4)
"""
from .api import TumorDetector
from .inference import heatmap
from .stain import macenko_normalize
from .weights import CHECKPOINTS, DEFAULT_MODEL, ENSEMBLES, clear_cache, fetch, install_local

__version__ = "0.1.0"
__all__ = ["TumorDetector", "heatmap", "macenko_normalize", "CHECKPOINTS",
           "ENSEMBLES", "DEFAULT_MODEL", "fetch", "install_local", "clear_cache",
           "TISSUE_CLASSES", "__version__"]

#: 9-class NCT tissue scheme. TUM is colorectal adenocarcinoma epithelium.
TISSUE_CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
