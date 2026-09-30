"""
Model configuration, read from environment variables so that the same scripts run on
Gemma-2-9B-it and Llama-3.1-8B-it.  Source one of the files in ``configs/``
before running any script, e.g. ``source configs/llama31_8b.sh``.

MODEL_NAME            HF repo id or local path of the model
NUM_LAYERS            number of transformer blocks
FFN_INTERMEDIATE_DIM  SwiGLU intermediate width (= number of FFN neurons per layer)
MODEL_DTYPE           float16 | bfloat16 (dtype of the attribution step)
"""

import os

import torch

MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
NUM_LAYERS = int(os.environ.get("NUM_LAYERS", "32"))
FFN_INTERMEDIATE_DIM = int(os.environ.get("FFN_INTERMEDIATE_DIM", "14336"))
MODEL_DTYPE = os.environ.get("MODEL_DTYPE", "float16")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def model_torch_dtype() -> torch.dtype:
    return torch.bfloat16 if MODEL_DTYPE == "bfloat16" else torch.float16


# Sub-directory name of each language inside data/train_data/
LANG_TO_TRAINDIR = {
    "EN": "en", "ZH": "zh-CN", "KO": "ko", "BN": "bn", "TH": "th", "SW": "sw",
    "HU": "hu", "AF": "af", "IT": "it", "NE": "ne",
}
