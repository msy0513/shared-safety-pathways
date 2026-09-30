#!/bin/bash
# Sec. 4: safety pathways-targeted cross-lingual alignment. Only the FFN parameters of the
# HR safety pathway are updated (Eq. 8); hyper-parameters as in Appendix Table 7.
#
#   bash scripts/train.sh configs/gemma2_9b.sh
set -eo pipefail
source "${1:?usage: train.sh <config>}"
PYTHON=${PYTHON:-$(command -v python || command -v python3)}
cd "$REPO"

$PYTHON training/pathway_sft.py \
    --type-A-file "$OUT_ROOT/type_A/type_A_paths_by_lang.json" --type-B-dir "$OUT_ROOT/type_B" \
    --train-data-dir "$TRAIN_DATA_DIR" --hr-lang $HR_LANG --target-langs $FT_LANGS \
    --output-dir "$OUT_ROOT/ckpt/pathway_sft" \
    --epochs $FT_EPOCHS --lr $FT_LR --batch-size $FT_BS --grad-accum-steps $FT_GA \
    --warmup-ratio 0.03 --weight-decay 0 --max-length 512 --save-every-epoch
