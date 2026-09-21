# Model card — omicore-tumordetector-crc-he-v0.1

## Overview

| | |
|---|---|
| Name | `omicore-tumordetector-crc-he-v0.1` |
| Task | Tumour-region mapping in colorectal H&E |
| Output | 9-class tissue probabilities per 224 px tile; tumour mask; GeoJSON polygons |
| Architecture | Ensemble of 3 classifiers, probability-averaged, with Macenko stain normalisation at inference |
| Input resolution | 0.5 µm/px (other resolutions rescaled via `mpp`) |
| Intended use | Research: locating tumour-enriched regions, e.g. for spatial-omics region selection |
| Out of scope | Any clinical or diagnostic use; non-colorectal tissue; cell-level malignancy calls |

### Ensemble members

| member | backbone | trained on |
|---|---|---|
| `...-convnext-norm-v0.1` | ConvNeXt-Tiny | NCT-CRC-HE-100K (Macenko-normalised) |
| `...-convnext-nonorm-v0.1` | ConvNeXt-Tiny | NCT-CRC-HE-100K-NONORM |
| `...-effnetv2-nonorm-v0.1` | EfficientNetV2-S | NCT-CRC-HE-100K-NONORM |

Auxiliary `...-seg-unet-v0.1` (U-Net/ResNet-34, EBHI-SEG, val IoU 0.866) refines
boundaries inside already-flagged regions. It is **not** a standalone detector:
EBHI-SEG masks mark epithelium rather than tumour, and only 76 normal images
are available against 981 positives.

## Training data

100,000 tiles across 9 tissue classes (`ADI BACK DEB LYM MUC MUS NORM STR TUM`)
from NCT-CRC-HE-100K, split 85k train / 15k validation. All CC-BY-4.0.

**Known limitation:** NCT-CRC-HE-100K ships no patient identifiers — the code in
each filename is a per-tile hash, verified unique for every tile. A random split
therefore probably shares patients between train and validation, so the internal
validation score (99.8%) is optimistic and should not be quoted. Held-out numbers
below are the meaningful ones.

## Evaluation

### Held-out tile benchmarks

| set | n | accuracy | balanced acc | tumour AUROC |
|---|---|---|---|---|
| CRC-VAL-HE-7K (different cohort) | 7,180 | 0.971 | 0.955 | 0.999 |
| Kather-2016 (different cohort + scanner) | 4,375 | 0.772 | 0.772 | 0.994 |

Kather-2016 lacks MUC and MUS tiles, so its macro AUROC is computed over the
7 classes present. Its lower accuracy reflects a genuine domain shift; tumour
AUROC stays high, which is what matters for this task.

### Whole-section performance (5 Visium HD sections, independent of training)

| section | truth | tumour area | regions |
|---|---|---|---|
| Cancer_P1 | tumour | 32.3% | 2 |
| Cancer_P2 | tumour | 48.5% | 1 |
| Cancer_P5 | tumour | 27.7% | 5 |
| Normal_P3 | normal | 0.08% | 0 |
| Normal_P5 | normal | 1.82% | 0 |

Separation +25.8% (min tumour − max normal). Both negative controls exported
zero regions.

### Independent-modality validation (Cancer_P1)

2,307 tiles with ≥20 segmented cells, scored against Visium HD transcriptomic
cell-type labels (202,182 cells, 859 molecularly tumour-positive tiles). The
image model never observes gene expression.

| metric | value |
|---|---|
| AUROC | **0.9855** |
| Spearman ρ | 0.769 |
| precision @ 0.4 | 0.935 |
| recall @ 0.4 | 0.921 |
| specificity @ 0.4 | 0.962 |

**Caveat:** the 0.4 operating point was selected on this same Cancer_P1 data, so
precision/recall are optimistic. AUROC is threshold-free and unaffected.

## Known failure modes

- **Calibration drift across stain domains.** A single model kept AUROC 0.955
  while recall at 0.5 fell to 0.481 on raw slides. The ensemble plus stain
  normalisation largely fixes this (recall 0.893 at 0.5), but re-check on a new
  scanner.
- **Well-differentiated tumour read as normal mucosa**, and **necrotic tumour
  read as debris** — the two dominant error modes in tile-level analysis.
- **Non-colorectal tissue.** Lung and prostate controls yielded 8.9% and 1.7%
  spurious tumour area at threshold 0.4.
- **Auto-thresholding fabricates disease** on tumour-free tissue; gated behind an
  explicit opt-in.

## Ethical and regulatory

Research use only. Not a medical device; no regulatory clearance. Outputs must
not inform patient care. Trained on public de-identified data; no patient
identifiers are present in the weights or this repository.

## Provenance

Datasets and pretrained backbone licences are listed in [`NOTICE`](NOTICE).
Full development log, including rejected approaches and bugs found, is in
[`FINDINGS.md`](FINDINGS.md).
