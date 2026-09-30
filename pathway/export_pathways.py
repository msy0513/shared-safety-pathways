"""
Step 4 — materialise monolingual and shared safety pathways (Sec. 3.2 / 3.3).

  P_lambda   = node set of P^A_lambda ∪ P^B_lambda
  P*_lambda  = P_lambda ∩ P_HR            (shared safety pathway, HR = EN)

Writes, for every language,
  <output-dir>/pathway_<LANG>.json          {layer: [neuron, ...]}
  <output-dir>/shared_pathway_<LANG>.json   {layer: [neuron, ...]}   (NHR languages)
  <output-dir>/summary.json                 node counts per language

Fine-tuning (training/pathway_sft.py) recomputes these sets from the Type-A / Type-B
files with the same loader; this script is for inspection and pathway statistics.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lingualpath.config import FFN_INTERMEDIATE_DIM, NUM_LAYERS
from lingualpath.pathways import (count_neurons, load_pathway_neurons,
                                  load_shared_pathway_neurons)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--languages", nargs="+", required=True)
    ap.add_argument("--type-A-file", required=True)
    ap.add_argument("--type-B-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--hr-lang", default="EN")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    total_ffn = NUM_LAYERS * FFN_INTERMEDIATE_DIM
    summary = {}
    for lang in args.languages:
        row = {}
        for mode in ("type_a", "type_b", "union"):
            nodes = load_pathway_neurons(args.type_A_file, args.type_B_dir, lang, mode)
            row[f"n_{mode}"] = count_neurons(nodes)
            if mode == "union":
                with open(os.path.join(args.output_dir, f"pathway_{lang}.json"), "w") as f:
                    json.dump({str(l): v for l, v in sorted(nodes.items())}, f)
        row["pct_of_ffn_neurons"] = round(100.0 * row["n_union"] / total_ffn, 4)
        if lang != args.hr_lang:
            shared = load_shared_pathway_neurons(args.type_A_file, args.type_B_dir, lang, args.hr_lang)
            row["n_shared_with_hr"] = count_neurons(shared)
            with open(os.path.join(args.output_dir, f"shared_pathway_{lang}.json"), "w") as f:
                json.dump({str(l): v for l, v in sorted(shared.items())}, f)
        summary[lang] = row
        print(lang, row)

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
