# Who Bridges Safety? Identifying and Targeting Cross-Lingual Shared Safety Pathways

This repository contains the official implementation of the paper
[*Who Bridges Safety? Identifying and Targeting Cross-Lingual Shared Safety Pathways*](https://arxiv.org/abs/2608.09095).

LLMs refuse harmful requests reliably in high-resource (HR) languages but much less so in
non-high-resource (NHR) languages. This repository implements three steps:

1. **Safety pathways.** Safety neurons are connected across layers into *monolingual safety
   pathways*.
2. **Shared safety pathways.** The part of each NHR pathway that it shares with the HR
   pathway is the bridge for safety transfer.
3. **Pathway-targeted alignment.** Fine-tuning updates only the parameters of the HR
   safety pathway, which is under 1% of the model.

Supported models: Gemma-2-9B-it, Llama-3.1-8B-Instruct.
HR language: EN. NHR languages: ZH, KO, BN, TH, SW, HU, AF, IT, NE.

> ⚠️ `data/` contains harmful prompts. It is released for safety research only.

## Repository structure

```
├── configs/                 configuration
│   ├── common.sh            data paths, languages and hyper-parameters (Appendix Tables 7-8)
│   ├── gemma2_9b.sh         model-specific settings
│   └── llama31_8b.sh
├── data/                    safety, benign and evaluation data + build scripts (see data/README.md)
├── lingualpath/             shared library (model config, pathway loading)
├── pathway/                 safety-pathway identification (Sec. 3.2-3.3)
│   ├── extract_ffn_activations.py
│   ├── attribute_safety_neurons.py
│   ├── build_coactivation_pathways.py
│   ├── build_propagation_pathways.py
│   └── export_pathways.py
├── training/
│   └── pathway_sft.py       pathway-targeted alignment (Sec. 4)
└── scripts/
    ├── identify_pathways.sh
    └── train.sh
```

## Installation

```bash
git clone <this-repo> && cd <this-repo>
conda create -n pathways python=3.10 -y && conda activate pathways
pip install -r requirements.txt
```

Every step runs on a single 80 GB GPU.

## Quick start

```bash
CONFIG=configs/llama31_8b.sh            # or configs/gemma2_9b.sh

# 0. Translate the benign queries D^+ into the NHR languages (once for all models)
bash data/scripts/translate_benign.sh

# 1. Identify safety pathways P_λ and shared safety pathways P*_λ
bash scripts/identify_pathways.sh $CONFIG

# 2. Pathway-targeted cross-lingual alignment
bash scripts/train.sh $CONFIG
```

The aligned model is written to `outputs/<model>/ckpt/pathway_sft/final` (a regular
Hugging Face checkpoint); a checkpoint is also saved after every epoch.

To use a local copy of a model, set `MODEL_NAME=/path/to/model` before running.
Configs are plain shell files. Every variable can be overridden from the environment,
e.g. `TOP_PCT=5 bash scripts/identify_pathways.sh $CONFIG`.

## Step by step

`scripts/identify_pathways.sh` runs the following steps for every language in `$PATH_LANGS`:

```bash
source configs/llama31_8b.sh
L=EN

# Step 0: probing contexts D^- (queries + safe refusals) and D^+ (benign queries + model
#         responses), and response-mean-pooled FFN activations
python pathway/extract_ffn_activations.py --languages $L \
    --train-data-dir $TRAIN_DATA_DIR --benign-dir $BENIGN_DIR \
    --prepared-dir $OUT_ROOT/prepared_data --activations-dir $OUT_ROOT/ffn_activations

# Step 1: safety neurons (Eq. 1-2)
python pathway/attribute_safety_neurons.py --lang $L \
    --prepared-dir $OUT_ROOT/prepared_data --output-dir $OUT_ROOT/safety_neurons

# Step 2: co-activation pathways (Eq. 3-5; CPU, all languages in one call)
python pathway/build_coactivation_pathways.py --languages $PATH_LANGS \
    --neurons-dir $OUT_ROOT/safety_neurons --acts-dir $OUT_ROOT/ffn_activations \
    --output-dir $OUT_ROOT/type_A

# Step 3: activation-propagation pathways (Eq. 6-7)
python pathway/build_propagation_pathways.py --languages $L \
    --neurons-dir $OUT_ROOT/safety_neurons --acts-dir $OUT_ROOT/ffn_activations \
    --prepared-dir $OUT_ROOT/prepared_data --output-dir $OUT_ROOT/type_B

# Step 4: P_λ and P*_λ
python pathway/export_pathways.py --languages $PATH_LANGS \
    --type-A-file $OUT_ROOT/type_A/type_A_paths_by_lang.json --type-B-dir $OUT_ROOT/type_B \
    --output-dir $OUT_ROOT/pathways
```

Steps 0, 1 and 3 are independent across languages. On a cluster, submit them as one job
per language. Step 3 is the most expensive one: every safety neuron is intervened on
separately.

Pathway-targeted alignment (`scripts/train.sh`):

```bash
python training/pathway_sft.py \
    --type-A-file $OUT_ROOT/type_A/type_A_paths_by_lang.json --type-B-dir $OUT_ROOT/type_B \
    --train-data-dir $TRAIN_DATA_DIR --target-langs $FT_LANGS \
    --output-dir $OUT_ROOT/ckpt/pathway_sft --save-every-epoch
```

## Configuration (`configs/common.sh`)

**Safety-pathway identification** (Appendix Table 8)

| Variable | Default | Meaning |
|---|---|---|
| `TOP_PCT` | 3.0 | per-layer top-k% of the unsafe / benign importance (Eq. 2) |
| `S_MIN` | 0.03 | minimum joint co-activation rate s_min (Eq. 5) |
| `PHI_MIN` | 0.02 | minimum phi coefficient φ_min (Eq. 5) |
| `RHO` | 1.5 | safety-specificity ρ: co-activation lift on D⁻ over lift on D⁺ (Eq. 5) |
| `ALPHA` | 0.05 | significance level α of the permutation test (Eq. 5) |
| `Z_MIN` | 2.0 | standardized intervention effect z_min (Eq. 7) |
| `R_MIN` | 0.05 | relative activation change r_min (Eq. 7) |
| `N_BENIGN` | 1000 | benign queries per language in D⁺ |
| `PROBE_MAX_NEW_TOKENS` | 128 | length of the generated benign responses |
| `N_PERMUTATIONS` | 100 | permutations of the permutation test |
| `N_SUBSAMPLE`, `N_NULL` | 50, 20 | intervention samples D̂⁻ / random reference neurons (Eq. 6-7) |

**Pathway-targeted alignment** (Appendix Table 7)

| Variable | Default | Meaning |
|---|---|---|
| `FT_LANGS` | EN ZH KO BN TH SW HU AF IT NE | languages of D_train |
| `FT_EPOCHS` | 3 | training epochs |
| `FT_LR` | 2e-5 | learning rate |
| `FT_BS` × `FT_GA` | 1 × 16 | batch size × gradient accumulation (effective batch size 16) |

Optimizer: Adafactor, warm-up ratio 0.03, maximum sequence length 512, weight decay 0.

## Outputs

```
outputs/<model>/
├── prepared_data/<LANG>/{safety,benign}.pt        tokenized D^- and D^+ sequences
├── ffn_activations/<lang>/{safety,benign}/layer_<l>.pt
├── safety_neurons/ffn_safety_neurons_<LANG>.json  safety neurons S_λ (+ raw importance scores)
├── type_A/type_A_paths_by_lang.json               co-activation pathways
├── type_B/type_B_paths_<LANG>.json                activation-propagation pathways
├── pathways/pathway_<LANG>.json                   P_λ   (layer -> neuron indices)
├── pathways/shared_pathway_<LANG>.json            P*_λ
├── pathways/summary.json                          pathway sizes per language
└── ckpt/pathway_sft/                              aligned model (final/, checkpoint-epoch*/, train_config.json)
```

## Implementation notes

- **FFN neuron.** An FFN neuron n_l^(i) is channel *i* of the gated FFN intermediate
  activation, i.e. the input of `model.model.layers[l].mlp.down_proj`. Activations are
  averaged over the response tokens.
- **Activation of a neuron (Eq. 3).** A neuron counts as activated on a sample
  (b_l^(i)(x) = 1) when its activation exceeds its median over the context.
- **Updated parameters (Eq. 8).** For every neuron *i* of the HR pathway, the mask m keeps
  row *i* of `gate_proj` and `up_proj` and column *i* of `down_proj`. All other parameters
  are frozen: attention, embeddings, norms and the remaining neurons.
  `train_config.json` records the pathway and the fraction of updated parameters.
- **Missing pathway files.** `lingualpath/pathways.py` raises an error if a Type-A or
  Type-B file is missing, so a pathway never silently degrades to a single edge type.

## License

The code is released under the MIT License (see `LICENSE`). The datasets in `data/`
are derived from third-party datasets and keep their original licences (see
`data/README.md`).

## Citation

If you find this work useful, please cite:

```bibtex
@article{miao2026whobridges,
  title   = {Who Bridges Safety? Identifying and Targeting Cross-Lingual Shared Safety Pathways},
  author  = {Miao, Shuyi and Qiu, Wangjie and Shao, Pengyang and Xiao, Canran and Shen, Fei and Zheng, Zhiming and Chua, Tat-Seng},
  journal = {arXiv preprint arXiv:2608.09095},
  year    = {2026}
}
```
