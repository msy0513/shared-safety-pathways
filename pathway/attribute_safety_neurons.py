"""
Step 1 — localizing safety neurons (Sec. 3.2 (1), Eq. 1-2).

For a (query, response) pair (x, y) and the FFN neuron n_l^(i) (channel i of the input of
``down_proj`` in layer l), the response-level gradient-activation attribution is

    G_l^(i)(x, y) = sum_{t in response} dL/da_{l,t}^(i) * a_{l,t}^(i),   L = -log p(y | x).

The unsafe importance of a neuron averages it over the unsafe context (Eq. 1)

    I^-_lambda(l, i) = - 1/|D^-| * sum_{(x,y) in D^-} G_l^(i)(x, y),

and the benign importance I^+_lambda(l, i) is computed in the same way on D^+.  With
T^-_{lambda,l} / T^+_{lambda,l} the per-layer top-k% neurons ranked by I^- / I^+, the
safety neurons of language lambda are (Eq. 2)

    S_lambda = U_l ( T^-_{lambda,l} \\ T^+_{lambda,l} ).

Inputs: <prepared_dir>/<LANG>/{safety,benign}.pt written by extract_ffn_activations.py
Outputs (in --output-dir)
  ffn_safety_neurons_<LANG>.json   {layer: {"indices": [...], "scores": {...}}}
  raw_scores_<LANG>.pt             {"unsafe": I^- (L, D), "benign": I^+ (L, D)}
"""

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lingualpath.config import (DEVICE, FFN_INTERMEDIATE_DIM, MODEL_NAME, NUM_LAYERS,
                                model_torch_dtype)


class FFNCapture:
    """Keeps the FFN intermediate activations (inputs of down_proj) with their gradients."""

    def __init__(self, model):
        self.model = model
        self.acts: Dict[int, torch.Tensor] = {}
        self.handles = []

    def __enter__(self):
        for l in range(NUM_LAYERS):
            def hook(module, inp, out, l=l):
                x = inp[0]
                x.retain_grad()
                self.acts[l] = x
            self.handles.append(self.model.model.layers[l].mlp.down_proj.register_forward_hook(hook))
        return self

    def __exit__(self, *args):
        for h in self.handles:
            h.remove()
        self.handles = []


def attribution(model, capture: FFNCapture, full_ids: torch.Tensor, resp_start: int) -> torch.Tensor:
    """G_l^(i)(x, y) for all layers and neurons: (NUM_LAYERS, FFN_INTERMEDIATE_DIM)."""
    model.zero_grad(set_to_none=True)
    capture.acts.clear()
    logits = model(input_ids=full_ids).logits
    T = logits.shape[1]
    loss = F.cross_entropy(logits[0, resp_start - 1:T - 1].float(), full_ids[0, resp_start:T])
    loss.backward()
    scores = torch.zeros(NUM_LAYERS, FFN_INTERMEDIATE_DIM)
    for l, a in capture.acts.items():
        if a.grad is not None:
            scores[l] = (a.grad * a).detach()[0, resp_start:T].float().sum(dim=0).cpu()
    return scores


def importance(model, capture, prepared: List[Dict], desc: str) -> torch.Tensor:
    """I(l, i) = - mean over the context of G_l^(i)(x, y)."""
    device = next(model.parameters()).device
    total = torch.zeros(NUM_LAYERS, FFN_INTERMEDIATE_DIM)
    n = 0
    for item in tqdm(prepared, desc=desc):
        ids, start = item["full_input_ids"].unsqueeze(0).to(device), int(item["response_start_idx"])
        if start >= ids.shape[1]:
            continue
        try:
            total += attribution(model, capture, ids, start)
            n += 1
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
    return -total / max(n, 1)


def select_safety_neurons(imp_unsafe: torch.Tensor, imp_benign: torch.Tensor,
                          top_pct: float = 3.0) -> Dict[str, Dict]:
    """Eq. 2: per layer, top-k% by I^- minus top-k% by I^+."""
    L, D = imp_unsafe.shape
    k = max(1, int(D * top_pct / 100))
    out = {}
    for l in range(L):
        u, b = imp_unsafe[l].numpy(), imp_benign[l].numpy()
        selected = sorted(set(np.argsort(u)[-k:].tolist()) - set(np.argsort(b)[-k:].tolist()))
        if selected:
            out[str(l)] = {"indices": selected, "scores": {str(i): float(u[i]) for i in selected}}
    return out


def main():
    ap = argparse.ArgumentParser(description="Safety-neuron localization (Eq. 1-2)")
    ap.add_argument("--lang", required=True)
    ap.add_argument("--prepared-dir", required=True, help="dir with <LANG>/{safety,benign}.pt")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--top-pct", type=float, default=3.0, help="per-layer top-k%%")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    from transformers import AutoModelForCausalLM

    print(f"[attribution] lang={args.lang} model={MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=model_torch_dtype(),
                                                 device_map=DEVICE)
    model.eval()
    # Gradients must reach the FFN activations while no parameter is trained: freeze
    # everything except the input embeddings.
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.get_input_embeddings().parameters():
        p.requires_grad_(True)

    load = lambda t: torch.load(os.path.join(args.prepared_dir, args.lang, f"{t}.pt"),
                                weights_only=False, map_location="cpu")
    unsafe, benign = load("safety"), load("benign")
    print(f"  |D^-| = {len(unsafe)}, |D^+| = {len(benign)}")

    with FFNCapture(model) as capture:
        imp_u = importance(model, capture, unsafe, f"  I^-[{args.lang}]")
        imp_b = importance(model, capture, benign, f"  I^+[{args.lang}]")

    neurons = select_safety_neurons(imp_u, imp_b, args.top_pct)
    print(f"  safety neurons (top {args.top_pct}%): {sum(len(v['indices']) for v in neurons.values())}")
    with open(os.path.join(args.output_dir, f"ffn_safety_neurons_{args.lang}.json"), "w") as f:
        json.dump(neurons, f, indent=2)
    torch.save({"unsafe": imp_u, "benign": imp_b, "n_unsafe": len(unsafe), "n_benign": len(benign)},
               os.path.join(args.output_dir, f"raw_scores_{args.lang}.pt"))


if __name__ == "__main__":
    main()
