"""OmiCoreTumorDetector -- tumour-region mapping for colorectal cancer H&E.

RESEARCH USE ONLY. Not a diagnostic device; not validated for clinical use.

    from omicoretumordetector import TumorDetector

    det = TumorDetector.from_pretrained("omicore-tumordetector-crc-he-v0.1")
    res = det.predict("slide.tiff", mpp=0.5)
    det.to_geojson(res, "tumor_regions.geojson", threshold=0.4)

If you use this package, please cite:

    Niwase S, Fujiyama A. OmiCoreTumorDetector: an open, molecularly validated
    model for mapping tumour regions in colorectal cancer H&E sections.
    bioRxiv 2026.09.21.753083 (2026). doi:10.64898/2026.09.21.753083

The citation is also available as ``omicoretumordetector.__citation__`` (text),
``omicoretumordetector.__bibtex__`` (BibTeX) and ``omicoretumordetector cite``.
"""
from .api import TumorDetector
from .inference import heatmap
from .stain import macenko_normalize
from .weights import CHECKPOINTS, DEFAULT_MODEL, ENSEMBLES, clear_cache, fetch, install_local

__version__ = "0.1.0"

__citation__ = (
    "Niwase S, Fujiyama A. OmiCoreTumorDetector: an open, molecularly validated model "
    "for mapping tumour regions in colorectal cancer H&E sections. "
    "bioRxiv 2026.09.21.753083 (2026). doi:10.64898/2026.09.21.753083")

__bibtex__ = """@article{niwase2026omicoretumordetector,
  title   = {{OmiCoreTumorDetector}: an open, molecularly validated model for mapping
             tumour regions in colorectal cancer {H\\&E} sections},
  author  = {Niwase, Shamim and Fujiyama, Akihisa},
  journal = {bioRxiv},
  year    = {2026},
  pages   = {2026.09.21.753083},
  doi     = {10.64898/2026.09.21.753083},
  url     = {https://doi.org/10.64898/2026.09.21.753083},
  note    = {Preprint}
}"""

__all__ = ["TumorDetector", "heatmap", "macenko_normalize", "CHECKPOINTS",
           "ENSEMBLES", "DEFAULT_MODEL", "fetch", "install_local", "clear_cache",
           "TISSUE_CLASSES", "__version__", "__citation__", "__bibtex__"]

#: 9-class NCT tissue scheme. TUM is colorectal adenocarcinoma epithelium.
TISSUE_CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
