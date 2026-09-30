#!/bin/bash
# Translate the English benign queries (data/benign/en.json, D^+) into the NHR languages.
#
#   bash data/scripts/translate_benign.sh                                   # Google Translate
#   BACKEND=llm LLM_API_KEY=... LLM_MODEL=... bash data/scripts/translate_benign.sh
# Re-running the script retries items whose translation failed.
set -eo pipefail
cd "$(dirname "$0")/../.."
PYTHON=${PYTHON:-$(command -v python || command -v python3)}
BACKEND=${BACKEND:-google}
LANGS=${LANGS:-"zh ko th bn ne hu af it sw"}

for l in $LANGS; do
  case $l in zh) code=zh-CN ;; *) code=$l ;; esac
  $PYTHON data/scripts/translate.py benign --src data/benign/en.json \
      --out data/benign/$l.json --lang $code --backend $BACKEND
done
