#!/bin/bash
# Sec. 3.2-3.3: safety neurons -> co-activation and activation-propagation pathways ->
# monolingual safety pathways P_lambda and shared safety pathways P*_lambda.
#
#   bash scripts/identify_pathways.sh configs/llama31_8b.sh
#
# The GPU steps loop over $PATH_LANGS; on a cluster, run them as one job per language.
set -eo pipefail
source "${1:?usage: identify_pathways.sh <config>}"
PYTHON=${PYTHON:-$(command -v python || command -v python3)}
cd "$REPO"

for L in $PATH_LANGS; do
  # Step 0: probing contexts D^- / D^+ and FFN activations (GPU)
  $PYTHON pathway/extract_ffn_activations.py --languages $L \
      --train-data-dir "$TRAIN_DATA_DIR" --benign-dir "$BENIGN_DIR" --n-benign $N_BENIGN \
      --max-new-tokens $PROBE_MAX_NEW_TOKENS \
      --prepared-dir "$OUT_ROOT/prepared_data" --activations-dir "$OUT_ROOT/ffn_activations"

  # Step 1: safety neurons, Eq. 1-2 (GPU)
  $PYTHON pathway/attribute_safety_neurons.py --lang $L --top-pct $TOP_PCT \
      --prepared-dir "$OUT_ROOT/prepared_data" --output-dir "$OUT_ROOT/safety_neurons"
done

# Step 2: co-activation pathways, Eq. 3-5 (CPU)
$PYTHON pathway/build_coactivation_pathways.py --languages $PATH_LANGS \
    --neurons-dir "$OUT_ROOT/safety_neurons" --acts-dir "$OUT_ROOT/ffn_activations" \
    --output-dir "$OUT_ROOT/type_A" \
    --min-support $S_MIN --min-phi $PHI_MIN --min-safety-ratio $RHO \
    --n-permutations $N_PERMUTATIONS --perm-alpha $ALPHA

# Step 3: activation-propagation pathways, Eq. 6-7 (GPU)
for L in $PATH_LANGS; do
  $PYTHON pathway/build_propagation_pathways.py --languages $L \
      --neurons-dir "$OUT_ROOT/safety_neurons" --acts-dir "$OUT_ROOT/ffn_activations" \
      --prepared-dir "$OUT_ROOT/prepared_data" --output-dir "$OUT_ROOT/type_B" \
      --n-subsample $N_SUBSAMPLE --n-null $N_NULL --min-z-delta $Z_MIN --min-rel-delta $R_MIN
done

# Step 4: P_lambda = P^A ∪ P^B and P*_lambda = P_lambda ∩ P_HR
$PYTHON pathway/export_pathways.py --languages $PATH_LANGS --hr-lang $HR_LANG \
    --type-A-file "$OUT_ROOT/type_A/type_A_paths_by_lang.json" --type-B-dir "$OUT_ROOT/type_B" \
    --output-dir "$OUT_ROOT/pathways"
