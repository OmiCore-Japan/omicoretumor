"""Tests that run without model weights, plus opt-in weight-dependent checks."""
import numpy as np
import pytest

import omicoretumordetector as ot
from omicoretumordetector.inference import is_background, tile_grid
from omicoretumordetector.stain import macenko_normalize, stain_matrix


def test_exports_and_classes():
    assert ot.__version__
    assert ot.TISSUE_CLASSES[-1] == "TUM"
    assert len(ot.TISSUE_CLASSES) == 9
    assert ot.DEFAULT_MODEL in ot.ENSEMBLES


def test_tile_grid_covers_edges():
    xs, ys = tile_grid(1000, 500, 224, 224)
    assert xs[0] == 0 and ys[0] == 0
    # final tile is snapped so the right/bottom edge is never dropped
    assert xs[-1] == 1000 - 224
    assert ys[-1] == 500 - 224


def test_tile_grid_smaller_than_tile():
    xs, ys = tile_grid(100, 100, 224, 224)
    assert xs == [] and ys == []


def test_background_detection():
    white = np.full((224, 224, 3), 250, np.uint8)
    assert is_background(white)
    rng = np.random.default_rng(0)
    tissue = rng.integers(40, 200, (224, 224, 3), dtype=np.uint8)
    assert not is_background(tissue)


def test_stain_matrix_is_well_conditioned():
    rng = np.random.default_rng(0)
    tile = rng.integers(40, 210, (224, 224, 3), dtype=np.uint8)
    m = stain_matrix(tile)
    if m is not None:
        # the two stain vectors must not collapse onto each other
        cos = float(np.clip(m[:, 0] @ m[:, 1], -1, 1))
        assert cos < 0.999


@pytest.mark.parametrize("tile", [
    np.full((64, 64, 3), 250, np.uint8),
    np.zeros((64, 64, 3), np.uint8),
])
def test_normalize_survives_degenerate_tiles(tile):
    out = macenko_normalize(tile)
    assert out.shape == tile.shape and out.dtype == np.uint8


def test_mpp_must_be_positive():
    det = ot.TumorDetector.__new__(ot.TumorDetector)
    with pytest.raises(ValueError):
        ot.TumorDetector.predict(det, "x.tif", mpp=0)


def test_auto_threshold_is_gated():
    det = ot.TumorDetector.__new__(ot.TumorDetector)
    det.allow_auto_threshold = False
    res = {"p_tumor": np.random.rand(20, 20).astype(np.float32),
           "tissue": np.ones((20, 20), bool)}
    with pytest.raises(ValueError, match="tumour-free"):
        ot.TumorDetector.resolve_threshold(det, res, "auto")
    assert ot.TumorDetector.resolve_threshold(det, res, 0.4) == 0.4


def test_unknown_model_name_is_clear():
    with pytest.raises((FileNotFoundError, KeyError)):
        ot.weights.resolve("not-a-real-model")


def test_output_stems_never_collide():
    """Visium HD writes every sample to <sample>/segmentation/he_mpp0.5.tiff,
    so identical basenames must not overwrite each other's results."""
    from omicoretumordetector.cli import _unique_stems

    paths = ["/d/Cancer_P2/segmentation/he_mpp0.5.tiff",
             "/d/Normal_P5/segmentation/he_mpp0.5.tiff",
             "/d/other/slide.tif"]
    stems = _unique_stems(paths)
    assert len(set(stems.values())) == 3
    assert any("Cancer_P2" in v for v in stems.values())
    assert any("Normal_P5" in v for v in stems.values())


def test_single_file_keeps_plain_name():
    from omicoretumordetector.cli import _unique_stems
    assert _unique_stems(["/x/slide.tif"])["/x/slide.tif"] == "slide"
