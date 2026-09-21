#!/usr/bin/env bash
# Full experiment matrix: train variants -> benchmark -> dry test -> molecular
# check -> leaderboard. Resumable: anything already produced is skipped.
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate crc_ml
cd "$(dirname "$0")/src"
M=../models; R=../reports; L=../logs; MAN=../data/manifests

train () {  # name arch manifests epochs
  local name=$1 arch=$2 man=$3 ep=${4:-8}
  # history.json is written only after the final epoch; best.pt appears from
  # epoch 1, so keying on it would skip a run that is still in progress.
  if [ -f "$M/$name/history.json" ]; then echo "[skip] train $name"; return; fi
  echo "[train] $name ($arch)"
  python train.py --manifests "$man" --arch "$arch" --epochs "$ep" \
    --batch-size 128 --workers 12 --out "$M/$name" > "$L/train_$name.log" 2>&1
  tr '\r' '\n' < "$L/train_$name.log" | grep -E "val_acc=|best val" | tail -3
}

bench () {  # name
  local name=$1
  for spec in "test.csv:${name}__ext7k" "test_2016.csv:${name}__ext2016" "val.csv:${name}__val"; do
    local mf="${spec%%:*}" tag="${spec##*:}"
    if [ -f "$R/metrics_$tag.json" ]; then echo "[skip] bench $tag"; continue; fi
    [ -f "$MAN/$mf" ] || continue
    # Do not pipe stderr into the grep: it would silently swallow tracebacks.
    if ! python evaluate.py --checkpoint "$M/$name/best.pt" --manifest "$MAN/$mf" \
         --tag "$tag" --out "$R" > "$L/eval_$tag.log" 2>&1; then
      echo "[FAIL] bench $tag -- see $L/eval_$tag.log"
      tail -5 "$L/eval_$tag.log"
      continue
    fi
    grep -E "^accuracy|^balanced accuracy|AUROC|^note:" "$L/eval_$tag.log" || true
  done
}

dry () {  # tag checkpoint... (extra flags via DRY_FLAGS)
  local tag=$1; shift
  if [ -f "$R/dry_test/$tag/dry_test_summary.csv" ]; then echo "[skip] dry $tag"; return; fi
  echo "[dry] $tag"
  # Run to its own log: with `set -o pipefail` a crash inside a pipeline would
  # otherwise abort the whole matrix, which is how one bad variant previously
  # took down every stage after it.
  if ! python dry_test.py --checkpoint "$@" --out "$R/dry_test" --tag "$tag" \
       --batch-size 256 --stride 224 ${DRY_FLAGS:-} > "$L/dry_$tag.log" 2>&1; then
    echo "[FAIL] dry $tag -- see $L/dry_$tag.log"
    tr '\r' '\n' < "$L/dry_$tag.log" | grep -vE "^rows:|^\s*$" | tail -12
    return 0
  fi
  tr '\r' '\n' < "$L/dry_$tag.log" | grep -vE "^rows:|^\s*$" | tail -20
}

mol () {  # tag
  local tag=$1
  local hm="$R/dry_test/$tag/Cancer_P1_heatmap.npz"
  [ -f "$hm" ] || { echo "[skip] molecular $tag (no heatmap)"; return; }
  if [ -f "$R/molecular/$tag/Cancer_P1_molecular_validation.json" ]; then echo "[skip] molecular $tag"; return; fi
  if ! python validate_molecular.py --sample Cancer_P1 --heatmap "$hm" \
       --out "$R/molecular/$tag" > "$L/mol_$tag.log" 2>&1; then
    echo "[FAIL] molecular $tag -- see $L/mol_$tag.log"
    tail -8 "$L/mol_$tag.log"
    return 0
  fi
  echo "[mol] $tag"
  grep -E "auroc_vs_molecular|precision|recall|specificity|spearman_rho" "$L/mol_$tag.log" || true
}

echo "================ TRAIN ================"
train A_convnext_norm   convnext_tiny.fb_in22k_ft_in1k  "$MAN"         8
if [ -d ../data/manifests_nonorm ]; then
  train B_convnext_nonorm convnext_tiny.fb_in22k_ft_in1k  ../data/manifests_nonorm 8
  train C_effnet_nonorm   tf_efficientnetv2_s.in21k_ft_in1k ../data/manifests_nonorm 8
else
  echo "[skip] NONORM models (manifests_nonorm not built yet)"
fi

echo "================ BENCHMARK ================"
for n in A_convnext_norm B_convnext_nonorm C_effnet_nonorm; do
  if [ -f "$M/$n/best.pt" ]; then bench "$n"; else echo "[skip] bench $n"; fi
done

echo "================ DRY TEST ================"
dry A_raw          "$M/A_convnext_norm/best.pt"
DRY_FLAGS="--stain-norm" dry A_stainnorm "$M/A_convnext_norm/best.pt"
DRY_FLAGS="--tta"        dry A_tta       "$M/A_convnext_norm/best.pt"
if [ -f "$M/B_convnext_nonorm/best.pt" ]; then dry B_raw "$M/B_convnext_nonorm/best.pt"; fi
if [ -f "$M/C_effnet_nonorm/best.pt" ];   then dry C_raw "$M/C_effnet_nonorm/best.pt"; fi
if [ -f "$M/B_convnext_nonorm/best.pt" ] && [ -f "$M/C_effnet_nonorm/best.pt" ]; then
  dry ENS_ABC "$M/A_convnext_norm/best.pt" "$M/B_convnext_nonorm/best.pt" "$M/C_effnet_nonorm/best.pt"
fi

echo "================ MOLECULAR ================"
for t in A_raw A_stainnorm A_tta B_raw C_raw ENS_ABC; do mol "$t"; done

echo "================ LEADERBOARD ================"
python compile_results.py --reports "$R" --dry-root "$R/dry_test" --out "$R/LEADERBOARD.md"
