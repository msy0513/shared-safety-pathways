"""
Build the per-language AdvBench-x / MultiJail evaluation files (data/eval/).

  eval/advbench_x_json/<lang>.json   {"safety": [520 prompts]}
  eval/multijail_json/<lang>.json    {"safety": [315 prompts]}

Sources
  --advbench-x-csv   AdvBench-x CSV with one column per language (en, zh, ko, th, af,
                     ne, sw, bn, it, vi, ar, jv); 520 rows = AdvBench harmful behaviours.
  --multijail-csv    MultiJail.csv (DAMO-NLP-SG/MultiJail; columns id, source, tags,
                     en, zh, it, vi, ar, ko, th, bn, sw, jv); 315 rows.

Languages that have no column in the CSV are machine-translated from the English
column with Google Translate (data/scripts/translate.py).  For the released files
this applies to HU, JP (both benchmarks) and AF, NE (MultiJail).  AdvBench-x AF and
NE were also re-translated from English with Google Translate rather than taken from
the CSV column; pass --retranslate af ne to reproduce this.

Usage
  python data/scripts/build_eval_sets.py \
      --advbench-x-csv advbench_x.csv --multijail-csv MultiJail.csv \
      --out-dir data/eval --langs en zh ko bn th sw hu af it ne jp ar \
      --retranslate af ne
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from translate import translate_list  # noqa: E402

# release language code -> Google Translate code
GOOGLE_CODE = {"zh": "zh-CN", "jp": "ja", "jv": "jw"}


def read_columns(path: str):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def write_safety(path: str, prompts):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"safety": prompts}, f, ensure_ascii=False, indent=2)
    print(f"  {path}: {len(prompts)} prompts")


def build(rows, bench_dir, langs, retranslate, skip_empty, sleep):
    cols = set(rows[0].keys())
    en = [r["en"].strip() for r in rows if not skip_empty or r["en"].strip()]
    for lang in langs:
        out = os.path.join(bench_dir, f"{lang}.json")
        if lang in cols and lang not in retranslate:
            prompts = [r[lang].strip() for r in rows if not skip_empty or r[lang].strip()]
        elif lang == "en":
            prompts = en
        else:
            if os.path.exists(out):
                print(f"  {out}: exists, skip translation")
                continue
            print(f"  translating {os.path.basename(bench_dir)} EN -> {lang}")
            prompts = translate_list(en, GOOGLE_CODE.get(lang, lang), "google", sleep)
        write_safety(out, prompts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--advbench-x-csv", required=True)
    ap.add_argument("--multijail-csv", required=True)
    ap.add_argument("--out-dir", default="data/eval")
    ap.add_argument("--langs", nargs="+",
                    default=["en", "zh", "ko", "bn", "th", "sw", "hu", "af", "it", "ne", "jp", "ar"])
    ap.add_argument("--retranslate", nargs="*", default=[],
                    help="AdvBench-x languages to re-translate from EN even if a CSV column exists")
    ap.add_argument("--sleep", type=float, default=0.2)
    args = ap.parse_args()

    print("AdvBench-x")
    build(read_columns(args.advbench_x_csv), os.path.join(args.out_dir, "advbench_x_json"),
          args.langs, set(args.retranslate), skip_empty=False, sleep=args.sleep)
    print("MultiJail")
    build(read_columns(args.multijail_csv), os.path.join(args.out_dir, "multijail_json"),
          args.langs, set(), skip_empty=True, sleep=args.sleep)


if __name__ == "__main__":
    main()
