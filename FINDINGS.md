# Iteration log

Chronological record of what was tried, what worked, and what did not.
Numbers are reproducible from the scripts in `src/`.

## Setup facts established up front

- **GPU**: RTX 5080 is Blackwell (`sm_120`); it needs a CUDA 12.8+ build.
  The default `cu121` PyTorch wheels do not run on it. Env `crc_ml` has
  torch 2.11.0+cu128, verified to execute on device.
- **Patient IDs are not recoverable** from NCT-CRC-HE-100K filenames. The code
  in each name is a per-tile hash (5,026 distinct codes for 5,026 tiles in the
  7K set), not a patient ID. A random split therefore probably leaks patients
  between train and internal val, so the internal val score is optimistic and
  **CRC-VAL-HE-7K is the number to quote**.
- **EBHI-SEG masks are not tumour masks.** Mean mask foreground is 55-70% for
  *every* class including `Normal`; the masks delineate glandular/epithelial
  structure and the folder name carries the diagnosis.
- **Kather-2016 `03_COMPLEX`** ("stroma with single tumour cells") has no clean
  counterpart in the 9-class scheme; its 625 tiles are dropped rather than
  forced into TUM or STR. `MUC`/`MUS` are absent from that release entirely.

## Domain gap to the target slides (measured, not assumed)

Mean RGB over tissue crops:

| | mean RGB | distance to NCT |
|---|---|---|
| NCT-CRC-HE-100K (target domain) | [179.3, 119.8, 167.2] | -- |
| Cancer_P1 `he_mpp0.5.tiff` raw | [212.9, 134.9, 211.3] | 57.5 |
| Cancer_P1 after Macenko | [193.7, 145.7, 158.8] | **30.8** |

The user's slides are lighter and pinker than NCT tiles. Macenko normalisation
roughly halves the gap. Whether that helps *accuracy* is decided by dry test,
not by this number.

### Bug found and fixed in the Macenko implementation
The first version produced a near-singular stain matrix (its two columns were
almost identical: `[0.487,0.764,0.422]` vs `[0.476,0.770,0.425]`), so the
least-squares deconvolution returned concentrations of +-18 and normalised
tiles came out nearly white. Cause: eigenvectors were ordered largest-first
(`vecs[:,[2,1]]`); the angle convention requires `vecs[:,1:3]`. In the flipped
frame the alpha/100-alpha percentiles collapse across the `arctan2` branch cut.
After the fix the two stain vectors sit 15 degrees apart and concentrations are sane.

## Experiment 1 -- early probe, Model A epoch-1 checkpoint

ConvNeXt-Tiny, NCT-CRC-HE-100K (Macenko-normalised), no stain norm, no TTA.

| sample | expected | tumour area | mean P(TUM) | top tissue calls |
|---|---|---|---|---|
| Cancer_P1 | tumour | 29.4% | 0.272 | STR 34%, TUM 32%, NORM 24% |
| Normal_P3 | normal | **0.0%** | 0.057 | MUS 34%, STR 19%, NORM 19% |

Composition is biologically sensible: desmoplastic stroma dominates the tumour
section, muscularis dominates the normal colon wall.

### Validation against an independent modality
Visium HD transcriptomic cell-type labels for Cancer_P1 (202,182 cells,
78,110 labelled `Tumor`). The H&E model never sees gene expression.

Coordinate chain verified exactly: `microns_per_pixel 0.2738 / 0.5 = 0.547616`
equals the recorded `tissue_0.5_mpp_150_buffer_scalef`; crop offset is a
constant `[380, 12177]`; cell-ID join is 202,182 / 202,182; all cells land
inside the scored grid.

**AUROC 0.956**, precision 0.912, recall 0.824, specificity 0.953,
Spearman rho 0.749 (2,307 tiles with >=20 cells).

### Error analysis of the 151 false-negative tiles
| called instead | n |
|---|---|
| NORM | 72 |
| DEB | 50 |
| TUM (but P < 0.5) | 24 |
| STR / MUS | 5 |

Their median molecular tumour fraction is 0.82, so they are genuinely
tumour-rich. Two real error modes: well-differentiated tumour read as normal
mucosa, and necrotic tumour read as debris. The 24 TUM-argmax tiles are pure
threshold artefacts. All 68 false positives were tumour-adjacent partial tiles
(median molecular tumour fraction 0.28).

## Experiment 2 -- spatial smoothing: REJECTED

Hypothesis: tumour regions are contiguous, so isolated NORM/DEB tiles inside a
tumour mass are misses that smoothing should recover.

| variant | AUROC | precision | recall | F1 |
|---|---|---|---|---|
| baseline (raw) | 0.9556 | 0.912 | 0.824 | 0.866 |
| gaussian sigma=1 (tissue-masked) | 0.9575 | 0.915 | 0.804 | 0.856 |
| gaussian sigma=1.5 | 0.9512 | 0.889 | 0.790 | 0.837 |
| median 3x3 | 0.9435 | 0.894 | 0.811 | 0.851 |
| mean 3x3 | 0.9569 | 0.911 | 0.799 | 0.851 |

**The hypothesis was wrong.** Smoothing moves AUROC by +0.002 at best and
*reduces* recall in every configuration. Isolated tiles were not the main error
mode. Not adopted.

## Experiment 3 -- threshold tuning: ADOPTED

| threshold | precision | recall | specificity | F1 |
|---|---|---|---|---|
| 0.50 (default) | 0.912 | 0.824 | 0.953 | 0.866 |
| 0.40 | 0.890 | 0.854 | 0.937 | 0.872 |
| **0.35** | 0.879 | 0.866 | 0.930 | **0.873** |
| 0.30 | 0.858 | 0.878 | 0.914 | 0.868 |
| 0.25 | 0.842 | 0.889 | 0.901 | 0.865 |

Dropping the threshold to ~0.35 buys 4 points of recall for 3 of precision.
Tuned on one section, so treat it as a starting operating point.

## A caveat about benchmarking the NONORM models

`CRC-VAL-HE-7K` is only released **stain-normalised**. There is no
non-normalised counterpart. So Model B/C (trained on NONORM tiles) are tested
on out-of-domain data on that benchmark, while Model A (trained on normalised
tiles) is tested in-domain.

This means the external-test table is **biased in Model A's favour**, and a
lower 7K score for B/C is not by itself evidence that they are worse for the
actual deployment. The dry test on the user's raw `he_mpp0.5.tiff` sections is
the comparison that reflects real use, because those slides are raw. Read the
two tables together, not the benchmark alone.

## Experiment 4 -- normalised vs non-normalised training: NONORM WINS OUT-OF-DOMAIN

Three models, 8 epochs each, identical recipe apart from the training tiles.

| model | trained on | 7K acc | 7K TUM AUROC | 2016 acc | **2016 TUM AUROC** |
|---|---|---|---|---|---|
| A ConvNeXt-Tiny | NCT-100K (Macenko-normalised) | 0.9705 | 0.9990 | 0.5687 | **0.8361** |
| B ConvNeXt-Tiny | NCT-100K-NONORM | 0.7797 | 0.9875 | 0.7630 | **0.9920** |
| C EfficientNetV2-S | NCT-100K-NONORM | 0.8209 | 0.9944 | 0.7723 | **0.9940** |

Kather-2016 is a different cohort and scanner, so it is the honest
out-of-domain probe. Training without stain normalisation lifts cross-domain
tumour AUROC from **0.836 to 0.99+**.

The picture is consistent once you account for which test set is in-domain for
which model:
- `CRC-VAL-HE-7K` is stain-normalised, so it is in-domain for A and
  out-of-domain for B/C -- A's lead there is expected and not informative
  about deployment.
- `Kather-2016` is out-of-domain for all three, and there B/C win decisively.

The user's `he_mpp0.5.tiff` sections are raw, so B/C are the better prior for
deployment. The dry test decides it.

## Two bugs found while running the matrix

1. **`evaluate.py` crashed on Kather-2016.** `roc_auc_score(..., multi_class="ovr")`
   requires every class to appear in `y_true`; the 2016 set has no MUC or MUS
   tiles, so it raised `ValueError: Number of classes in y_true not equal to
   the number of columns in 'y_score'`. Fixed by computing the macro
   one-vs-rest AUROC over the classes actually present and recording which are
   absent.
2. **The failure was invisible.** `bench()` piped stderr into
   `grep -E "accuracy|AUROC|balanced"`, which filtered the traceback away, so
   the stage looked like it had succeeded while silently writing no JSON.
   Fixed: evaluation now writes its own log, non-zero exit is reported as
   `[FAIL]`, and only then is the log grepped for the summary lines.

## Experiment 5 -- the 0.5 threshold is wrong on out-of-domain slides

Model A, epoch 1 vs epoch 8, both scored against Cancer_P1 molecular truth:

| | epoch 1 | epoch 8 |
|---|---|---|
| AUROC | 0.9556 | 0.9553 |
| precision @ 0.5 | 0.912 | 0.981 |
| **recall @ 0.5** | **0.824** | **0.481** |

**AUROC is identical, recall halves.** More training did not damage the model's
ability to rank tiles; it changed its calibration. On these raw slides the
epoch-8 probabilities are compressed toward zero:

| | 10th | 25th | 50th | 75th | 90th |
|---|---|---|---|---|---|
| molecularly-tumour tiles | 0.028 | 0.090 | **0.461** | 0.834 | 0.873 |
| molecularly-negative tiles | 0.006 | 0.007 | 0.009 | 0.015 | 0.037 |

The two distributions are still well separated -- just not around 0.5.

Threshold sweep for epoch 8:

| threshold | precision | recall | specificity | F1 |
|---|---|---|---|---|
| 0.02 | 0.748 | 0.942 | 0.811 | 0.834 |
| **0.05** | 0.866 | 0.818 | 0.925 | **0.841** |
| 0.10 | 0.911 | 0.738 | 0.957 | 0.815 |
| 0.50 (default) | 0.981 | 0.481 | 0.994 | 0.645 |

**Practical consequence: the operating threshold must be calibrated per
deployment domain.** Shipping the usual 0.5 default on these slides would
silently discard half the tumour while looking highly precise.

Epoch 1 at its own best threshold (0.35, F1 0.873) still beats epoch 8 at its
best (0.05, F1 0.841), so the longer schedule traded some domain robustness for
in-domain fit. The NONORM models are the better bet; their dry tests decide.

## Experiment 6 -- label-free auto-thresholding: USEFUL BUT NOT AS A DEFAULT

Experiment 5 showed the threshold must be calibrated per domain, which is a
problem on a new slide with no labels. Tested three label-free methods against
the Cancer_P1 molecular truth, using only that slide's own P(TUM) values:

| method | threshold | F1 | recall |
|---|---|---|---|
| Otsu | 0.3951 | 0.689 | 0.533 |
| Li | 0.1884 | 0.764 | 0.640 |
| Otsu on log-odds | 0.1316 | 0.799 | 0.701 |
| **Triangle** | **0.0546** | **0.838** | 0.807 |
| oracle (swept with labels) | 0.0334 | 0.855 | 0.885 |

Triangle lands within 2% of the oracle F1 with no labels. Otsu fails because it
assumes two comparable modes; this distribution is a large near-zero mass with
a long tail.

### ...but it destroys specificity on tumour-free tissue

Mean tumour area per group, same model, different thresholding:

| thresholding | tumour sections | normal sections | out-of-organ | separation |
|---|---|---|---|---|
| fixed 0.5 | 22.2% | **2.9%** | **4.9%** | +11.5% |
| fixed 0.05 | 36.6% | 11.6% | 43.9% | +12.4% |
| auto (triangle) | 37.5% | **12.6%** | **27.2%** | +13.1% |

Per-slide auto thresholds: Lung_HD 0.0655 -> **39.4% of lung tissue called
colorectal tumour** (5.6% at a fixed 0.5). Normal_P5 0.0378 -> 18.4% (4.8%).

**Why:** the triangle method assumes a tumour mode exists. Given tumour-free
tissue it finds a threshold inside the noise and invents disease. The
separation metric barely moves because both groups inflate together.

**Decision:** `threshold="auto"` is gated behind
`TumorDetector(..., allow_auto_threshold=True)` and raises with this rationale
otherwise. Use it only to delineate a tumour already known to be present; use a
fixed threshold to decide *whether* a slide has tumour. The stated requirement
here is high specificity, so the fixed threshold is the default.

## Experiment 7 -- Macenko normalisation at inference: ADOPTED

Model A, same checkpoint, tiles Macenko-normalised toward the NCT domain
before classification. Mean tumour area per section:

| section | expected | raw | stain-norm |
|---|---|---|---|
| Cancer_P1 | tumour | 16.2% | 24.0% |
| Cancer_P2 | tumour | 29.8% | 18.4% |
| Cancer_P5 | tumour | 20.6% | 23.4% |
| Normal_P3 | normal | 1.1% | **0.04%** |
| Normal_P5 | normal | 4.8% | **0.30%** |
| Lung_HD | other organ | 5.6% | **0.67%** |
| Prostate_HD | other organ | 4.2% | **1.29%** |
| | **separation** | **+11.5%** | **+18.1%** |

False-positive area on tumour-free tissue drops by 3-27x while tumour sections
stay at 18-24%. Against Cancer_P1 molecular truth, AUROC is unchanged
(0.9548 vs 0.9553) but calibration improves markedly -- recall at the 0.5
default goes 0.481 -> 0.710, and the best F1 rises 0.841 -> **0.865** (at
threshold 0.10 rather than 0.05).

So normalisation does not make the model *discriminate* better; it puts the
slide back in the domain where the model's probabilities mean what they were
trained to mean. Cost is 9.6 ms/tile (see below), about 3 min for all seven
sections.

### Macenko implementation sped up 4x
`concentrations()` called `np.linalg.lstsq(m, od.T)`, solving a 3x2 system for
~50,000 right-hand sides per tile. Replaced with a precomputed `pinv(m)` and a
single small matmul -- identical least-squares answer -- plus a 20,000-pixel
cap when estimating the stain matrix, since the stain directions are a
population property. 38.5 -> 9.6 ms/tile, output unchanged.

## Bug 3 -- test-time augmentation crashed

`A_tta` died with `IndexError: Dimension out of range (expected to be in range
of [-2, 1], but got 2)`. The tile model returns `(B, n_classes)` -- a class
score, not a spatial map -- but the TTA code flipped the *output* back along
dims 2 and 3 as though it were segmentation. Only the input should be flipped.
Fixed and verified on the synthetic mosaic (tumour half 0.882, normal half
0.008, probabilities still summing to 1).

## Experiment 8 -- final variant matrix

Seven variants x seven real sections, ranked two independent ways.

### Tumour/normal separation on the user's sections

| variant | tumour | normal | other organ | separation |
|---|---|---|---|---|
| **ENS_ABC + stain-norm** | 34.1% | **0.57%** | 3.3% | **+24.6%** |
| A + stain-norm | 21.9% | 0.17% | 0.98% | +18.1% |
| ENS_ABC | 31.0% | 1.14% | 3.7% | +16.4% |
| B (NONORM) | 35.1% | 3.06% | 7.3% | +14.2% |
| A + TTA | 22.2% | 2.54% | 4.2% | +12.7% |
| A raw | 22.2% | 2.92% | 4.9% | +11.5% |
| C (NONORM) | 30.0% | 1.27% | 6.1% | +10.7% |

### Agreement with Cancer_P1 transcriptomic truth (independent modality)

| variant | AUROC | precision | recall | specificity |
|---|---|---|---|---|
| **ENS_ABC + stain-norm** | **0.9855** | 0.950 | **0.893** | 0.972 |
| ENS_ABC | 0.9831 | 0.960 | 0.875 | 0.979 |
| B (NONORM) | 0.9792 | 0.910 | 0.939 | 0.945 |
| C (NONORM) | 0.9712 | 0.929 | 0.880 | 0.960 |
| A + TTA | 0.9600 | 0.988 | 0.498 | 0.997 |
| A epoch-1 raw | 0.9556 | 0.912 | 0.824 | 0.953 |
| A raw | 0.9553 | 0.981 | 0.481 | 0.994 |
| A + stain-norm | 0.9548 | 0.971 | 0.710 | 0.988 |

**The same variant wins both rankings**, which is reassuring because the two
criteria are independent: one uses sample-level labels across seven sections,
the other uses per-cell transcriptomics on one section.

The two ingredients are complementary, which is why combining them helps:
- **NONORM training** raises sensitivity (tumour area 35% vs 22%) and
  cross-cohort AUROC (2016: 0.99 vs 0.84).
- **Macenko at inference** raises specificity (normal-section area 0.2% vs
  2.9%) by restoring the calibration the model was trained with.

### Recommended production configuration

    TumorDetector(["models/A_convnext_norm/best.pt",
                   "models/B_convnext_nonorm/best.pt",
                   "models/C_effnet_nonorm/best.pt"],
                  stain_norm=True)
    # threshold 0.35-0.40; stride 112 for finer boundaries

Threshold sweep for this configuration against molecular truth:

| threshold | precision | recall | specificity | F1 |
|---|---|---|---|---|
| 0.30 | 0.908 | 0.948 | 0.943 | **0.928** |
| 0.40 | 0.935 | 0.921 | 0.962 | **0.928** |
| 0.50 | 0.950 | 0.893 | 0.972 | 0.921 |

The calibration problem from Experiment 5 is largely gone: recall at the plain
0.5 default is back to 0.893 (it was 0.481 for the single raw model), so the
result is no longer sensitive to an exotic threshold choice. 0.35-0.40 is the
F1 optimum; raise it toward 0.5 if you want to favour precision.

## Segmentation model (EBHI-SEG)

U-Net, ResNet-34 encoder, positives = Adenocarcinoma + High-grade IN,
negatives = Normal (empty target). Best validation **IoU 0.866, Dice 0.928**
(precision 0.915, recall 0.941).

Caveat from the top of this file still applies: with only 76 Normal images
against 981 positives, the negative evidence is thin, so this model largely
learns "epithelium" and should be used to refine boundaries inside regions the
tile ensemble already flagged -- not as a standalone tumour detector.
