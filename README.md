# OmiCoreTumor

**Tumour-region mapping for colorectal cancer H&E histopathology.**

`omicoretumor-crc-he-v0.1` takes an H&E image of colon tissue and returns a
per-tile tumour probability map, a binary tumour mask, and polygon annotations
that open directly in QuPath.

> ⚠️ **RESEARCH USE ONLY.** Not a medical device. Not cleared or approved by any
> regulatory authority. Must not be used for diagnosis or clinical decisions.

| | |
|---|---|
| Validation | 5 Visium HD colorectal sections, independent of all training data |
| Tumour sections | 27.7% – 48.5% of tissue called tumour |
| Normal controls | 0.08% and 1.82%; **zero** tumour regions exported |
| Molecular agreement | **AUROC 0.9855** vs Visium HD transcriptomics (Cancer_P1) |

Full write-up: [`OmiCoreTumor_preprint.pdf`](OmiCoreTumor_preprint.pdf) ·
[`MODEL_CARD.md`](MODEL_CARD.md) · [`FINDINGS.md`](FINDINGS.md)

---

## 1. Install

This package is **not on PyPI**. Install it from this repository.

### Option A — install directly from GitHub (simplest)

```bash
pip install "git+https://github.com/OmiCore-Japan/omicoretumor.git"
```

### Option B — clone, then install

```bash
git clone https://github.com/OmiCore-Japan/omicoretumor.git
cd omicoretumor
pip install -e .
```

### PyTorch

`pip` installs a default PyTorch build. Check it sees your GPU:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If it prints `False` but you have an NVIDIA GPU, install a CUDA build:

```bash
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
```

> **RTX 50-series (Blackwell, e.g. 5070/5080/5090) must use cu128 or newer.**
> The default `cu121` wheels do not support compute capability `sm_120` and will
> fail at runtime. This is the most common installation problem.

CPU works but is slow: a 13,000 × 13,000 px section takes minutes on CPU versus
about 50 s on one RTX 5080.

---

## 2. Get the model weights

Weights are ~400 MB and exceed GitHub's 100 MB per-file limit, so they are
**release assets**, not files in this repository.

**They download automatically the first time you load the model** — you do not
normally need to do anything:

```python
from omicoretumor import TumorDetector
det = TumorDetector.from_pretrained()   # downloads ~300 MB once, then cached
```

Cached in `~/.cache/omicoretumor/` (override with `OMICORETUMOR_HOME`).
Downloads are SHA-256 verified.

```bash
omicoretumor models        # list weights and show what is cached
omicoretumor clear-cache   # delete them
```

<details>
<summary>Offline or air-gapped machines</summary>

Download the four `.pt` files from the
[Releases page](https://github.com/OmiCore-Japan/omicoretumor/releases) on a connected
machine, copy them across, then place them in the cache directory:

```bash
mkdir -p ~/.cache/omicoretumor
cp omicoretumor-crc-he-*.pt ~/.cache/omicoretumor/
omicoretumor models        # should now show [cached]
```

Or point the package at any URL or local mirror:

```bash
export OMICORETUMOR_WEIGHTS_URL="file:///data/omicoretumor-weights"
```
</details>

---

## 3. Use it

### Python

```python
from omicoretumor import TumorDetector

det = TumorDetector.from_pretrained("omicoretumor-crc-he-v0.1")

# mpp = microns per pixel of YOUR image.
# The model expects 0.5; other values are rescaled automatically.
res = det.predict("colon_section.tiff", mpp=0.5)

print(f"tumour area  : {res['tumor_fraction']:.1%}")
print(f"tissue tiles : {res['n_tissue_tiles']:,}")

mask = det.tumor_mask(res, threshold=0.4)                    # HxW uint8
det.to_geojson(res, "tumor_regions.geojson", threshold=0.4)  # open in QuPath
```

### Command line

```bash
omicoretumor predict slide.tiff --mpp 0.5 -o results/
omicoretumor predict *.tiff --mpp 0.25 --stride 112 --save-heatmap -o results/
```

### Examples

```bash
python examples/quickstart.py  my_section.tiff 0.5
python examples/batch_and_plot.py  ./my_sections/  0.5   # writes PNG overlays
```

---

## 4. Getting `mpp` right — read this

`mpp` (microns per pixel) is the **one parameter you must supply correctly**.
Passing the wrong value feeds the model tissue at the wrong magnification and
results degrade silently — no error, just poor output.

| your image | pass |
|---|---|
| already 0.5 µm/px | `mpp=0.5` |
| 40× scan, typical | `mpp=0.25` |
| 20× scan, typical | `mpp=0.5` |
| Visium HD `he_mpp0.5.tiff` | `mpp=0.5` |

Find it in your slide metadata (`openslide` `mpp-x`, or the scanner's export
settings). If your image is smaller than one 224 px tile after rescaling, the
package raises an error naming `mpp` rather than returning an empty result.

---

## 5. Output

Nine tissue classes — `ADI BACK DEB LYM MUC MUS NORM STR TUM` — where **TUM** is
colorectal adenocarcinoma epithelium and **NORM** is normal colon mucosa.

| key | meaning |
|---|---|
| `p_tumor` | tumour probability per 224 px tile |
| `prob` | all nine class probabilities, `(rows, cols, 9)` |
| `tissue` | tiles containing tissue (glass is skipped) |
| `tumor_fraction` | fraction of tissue tiles above 0.5 |
| `xs`, `ys`, `scale` | tile geometry and the factor back to source pixels |

`det.tissue_composition(res)` returns the fraction of tissue assigned to each
class — useful as a sanity check (a colorectal tumour section should be rich in
`TUM` and `STR`).

GeoJSON coordinates are in **your input image's pixel space**, so annotations
line up when loaded over the original slide.

---

## 6. Choosing a threshold

Default **0.4**. Do not assume it transfers to your scanner.

A model's probability calibration shifts between stain domains. We measured a
single model whose AUROC was unchanged on new slides while its recall at the
conventional 0.5 threshold collapsed from 0.92 to **0.48** — the ranking was
intact, the threshold was simply wrong. It looked highly precise while
discarding half the tumour.

If you have any labelled tissue, sweep the threshold on it. If you have none,
start at 0.4, inspect the overlays, and adjust: lower catches more tumour,
higher is more conservative.

`threshold="auto"` calibrates from the slide's own distribution, but it assumes
tumour is present and **fabricates regions on tumour-free tissue** (it called
39% of a lung section colorectal tumour). It is gated behind
`TumorDetector(..., allow_auto_threshold=True)` and must not be used to decide
*whether* a slide has tumour.

---

## 7. Limitations — please read before using results

- **Regions, not cells.** Boundaries resolve at 224 px (112 µm), so a called
  region contains stroma, immune and vascular cells alongside malignant glands.
  Use `stride=112` for finer edges at ~4× the compute. This model supports
  "localises tumour-enriched regions"; it does **not** support "identifies
  malignant cells".
- **Validation is small.** Three tumour and two normal sections from one cohort,
  with matched transcriptomics for one. That is feasibility, not generalisation
  across patients, scanners or histological variants.
- **Colorectal only.** Lung and prostate were negative controls and produced
  8.9% and 1.7% spurious tumour area; other organs are unsupported.
- **Untested tissue types:** adenoma, dysplasia, inflammation, ulceration,
  necrosis, mucinous variants, treated tissue.
- The internal validation figure (99.8%) is optimistic — NCT-CRC-HE-100K carries
  no patient IDs, so a random split likely shares patients. Quote the held-out
  cohorts instead.

---

## 8. Repository contents

```
omicoretumor/          the installable package
  api.py               TumorDetector: predict, tumor_mask, to_geojson
  inference.py         sliding-window tiling
  stain.py             Macenko normalisation
  ensemble.py          probability averaging
  weights.py           weight resolution, download, checksum
  cli.py               `omicoretumor` command
examples/              runnable scripts
tests/                 pytest suite (runs without weights)
training/              full training + evaluation code, reproduces everything
docs/                  per-section result figures
scripts/               helper to stage locally trained weights
OmiCoreTumor_preprint.pdf          manuscript, 4 multipanel figures
CRC_tumour_detection_report.pdf    per-section performance report
MODEL_CARD.md          intended use, training data, metrics, failure modes
FINDINGS.md            development log: what worked, what was rejected, bugs found
NOTICE                 dataset attribution (CC-BY-4.0 requirements)
```

Reproduce the whole study (needs the public datasets, ~30 GB):

```bash
bash training/run_experiments.sh
```

---

## 9. Troubleshooting

| symptom | cause and fix |
|---|---|
| `CUDA error` / `no kernel image` on RTX 50-series | install the cu128 PyTorch build (§1) |
| `404` when downloading weights | no release published yet; see `PUBLISHING.md`, or stage weights manually (§2) |
| `checksum mismatch` | partial download; `omicoretumor clear-cache` and retry |
| Tumour found everywhere | `mpp` likely wrong, or threshold too low |
| Nothing found on an obvious tumour | `mpp` likely wrong, or calibration shift — lower the threshold and inspect (§6) |
| `image is NxN px ... smaller than one tile` | `mpp` wrong for this image |
| Very slow | running on CPU; check `torch.cuda.is_available()` |

---

## 10. Citation and licence

Code is Apache-2.0. Weights are derivative works of CC-BY-4.0 datasets — if you
use them you must also credit Kather et al. (2018, 2016) and, for the
segmentation model, Shi et al. (2022). Full attribution in [`NOTICE`](NOTICE);
citation metadata in [`CITATION.cff`](CITATION.cff).
