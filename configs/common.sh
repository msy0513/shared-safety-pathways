# Settings shared by all models. Sourced by the model configs; any variable can be
# overridden in the environment before sourcing.
export REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export DATA_ROOT=${DATA_ROOT:-$REPO/data}
export PYTHONPATH=$REPO:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# ---- Data (see data/README.md) ---------------------------------------------------
export TRAIN_DATA_DIR=${TRAIN_DATA_DIR:-$DATA_ROOT/train_data}  # <lang>/final_safe.jsonl: D^- and D_train
export BENIGN_DIR=${BENIGN_DIR:-$DATA_ROOT/benign}              # <lang>.json benign queries: D^+

# ---- Languages (Sec. 3.1) --------------------------------------------------------
export HR_LANG=${HR_LANG:-EN}
export PATH_LANGS=${PATH_LANGS:-"EN ZH KO BN TH SW HU AF IT NE"}   # HR + 9 NHR languages

# ---- Safety-pathway identification (Sec. 3.2, Appendix Table 8) -------------------
export N_BENIGN=${N_BENIGN:-1000}                 # benign queries per language in D^+
export PROBE_MAX_NEW_TOKENS=${PROBE_MAX_NEW_TOKENS:-128}  # length of the benign responses
export TOP_PCT=${TOP_PCT:-3.0}                    # per-layer top-k%
export S_MIN=${S_MIN:-0.03}                       # joint co-activation rate s_min
export PHI_MIN=${PHI_MIN:-0.02}                   # phi coefficient phi_min
export RHO=${RHO:-1.5}                            # safety-specificity rho
export ALPHA=${ALPHA:-0.05}                       # permutation-test significance level
export N_PERMUTATIONS=${N_PERMUTATIONS:-100}
export Z_MIN=${Z_MIN:-2.0}                        # standardized intervention effect z_min
export R_MIN=${R_MIN:-0.05}                       # relative activation change r_min
export N_SUBSAMPLE=${N_SUBSAMPLE:-50}             # |D^-_hat|, unsafe samples used for the interventions
export N_NULL=${N_NULL:-20}                       # random non-safety neurons for the reference

# ---- Pathway-targeted alignment (Sec. 4, Appendix Table 7) ------------------------
export FT_LANGS=${FT_LANGS:-"EN ZH KO BN TH SW HU AF IT NE"}
export FT_EPOCHS=${FT_EPOCHS:-3}
export FT_LR=${FT_LR:-2e-5}
export FT_BS=${FT_BS:-1}
export FT_GA=${FT_GA:-16}                          # effective batch size = FT_BS x FT_GA = 16
