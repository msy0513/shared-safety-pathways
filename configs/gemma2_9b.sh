# Gemma-2-9B-it
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
export MODEL_TAG=gemma2_9b
export MODEL_NAME=${MODEL_NAME:-google/gemma-2-9b-it}
export NUM_LAYERS=42 HIDDEN_DIM=3584 FFN_INTERMEDIATE_DIM=14336 MODEL_DTYPE=bfloat16
export OUT_ROOT=${OUT_ROOT:-$REPO/outputs/$MODEL_TAG}
