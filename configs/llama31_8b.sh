# Llama-3.1-8B-Instruct
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
export MODEL_TAG=llama31_8b
export MODEL_NAME=${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}
export NUM_LAYERS=32 HIDDEN_DIM=4096 FFN_INTERMEDIATE_DIM=14336 MODEL_DTYPE=float16
export OUT_ROOT=${OUT_ROOT:-$REPO/outputs/$MODEL_TAG}
