"""
Step 3 — activation-propagation pathways P^B_lambda (Sec. 3.2 (2)-2, Eq. 6-7).

For every adjacent layer pair (l, l+1) and every safety neuron n_l^(i):
  1. on a subset D^-_hat of the unsafe context (--n-subsample samples, fixed seed), replace
     the activation of n_l^(i) by its mean activation over the benign samples;
  2. measure the change of every safety neuron n_{l+1}^(j) of the next layer (Eq. 6)
         Delta_{l,i->j} = E_{(x,y) in D^-_hat} [ a_{l+1}^(j)(x) - a~_{l+1}^(j)(x; l, i) ];
  3. apply the same intervention to --n-null randomly sampled NON-safety neurons of layer
     l; their changes form the reference distribution used to standardize Delta into z;
  4. keep the edge if (Eq. 7)
         z_{l,i->j} >= --min-z-delta (z_min)  and  Delta_{l,i->j} / M_{l+1,j} >= --min-rel-delta (r_min),
     where M_{l+1,j} is the mean activation of n_{l+1}^(j) over the unsafe samples.

Output (one file per language, so languages can run as parallel jobs)
  <output-dir>/type_B_paths_<LANG>.json      {LANG: {"edges": [...], ...}}

The final safety pathway is P_lambda = P^A_lambda ∪ P^B_lambda (see lingualpath/pathways.py).
"""

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lingualpath.config import FFN_INTERMEDIATE_DIM, MODEL_NAME, NUM_LAYERS
from lingualpath.pathways import load_safety_neurons

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_normal_acts(acts_dir: str, lang: str, layer: int) -> np.ndarray:
    """Load saved FFN intermediate activations for safety data at given layer."""
    path = os.path.join(acts_dir, lang.lower(), "safety", f"layer_{layer}.pt")
    t = torch.load(path, map_location="cpu", weights_only=False)
    return t.numpy() if isinstance(t, torch.Tensor) else np.asarray(t)


def load_benign_means(acts_dir: str, lang: str) -> Dict[int, torch.Tensor]:
    """Per-layer mean of benign FFN activations (for mean-ablation)."""
    means = {}
    for l in range(NUM_LAYERS):
        path = os.path.join(acts_dir, lang.lower(), "benign", f"layer_{l}.pt")
        if os.path.exists(path):
            t = torch.load(path, map_location="cpu", weights_only=False)
            means[l] = t.mean(dim=0)
    return means


def load_prepared_data(prepared_dir: str, lang: str) -> list:
    path = os.path.join(prepared_dir, lang, "safety.pt")
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# Single-neuron ablation: core routine
# ---------------------------------------------------------------------------
def ablate_single_neuron_and_capture(
    model,
    prepared_data: list,
    sample_indices: List[int],
    ablate_layer: int,
    ablate_neuron: int,
    capture_layer: int,
    capture_neurons: List[int],
    benign_mean: torch.Tensor,
) -> np.ndarray:
    """Ablate one neuron at ablate_layer, capture activations at capture_layer.

    Returns:
        captured_acts: (n_samples, n_capture_neurons) mean-pooled over response tokens.
    """
    idx_tensor = torch.tensor([ablate_neuron], dtype=torch.long, device=DEVICE)
    mean_vals = benign_mean.to(DEVICE)

    captured_acts = []

    for si in sample_indices:
        item = prepared_data[si]
        full_ids = item["full_input_ids"]
        resp_start = item["response_start_idx"]

        if resp_start >= full_ids.shape[0]:
            captured_acts.append(np.zeros(len(capture_neurons)))
            continue

        captured = {}

        def make_ablation_hook(idx_t=idx_tensor, mv=mean_vals):
            def hook(module, inputs):
                x = inputs[0]
                x[..., idx_t] = mv[idx_t].to(x.dtype)
                return (x,) + inputs[1:]
            return hook

        def make_capture_hook(store=captured, key="act"):
            def hook(module, inputs):
                store[key] = inputs[0].detach().clone()
            return hook

        h1 = model.model.layers[ablate_layer].mlp.down_proj.register_forward_pre_hook(
            make_ablation_hook()
        )
        h2 = model.model.layers[capture_layer].mlp.down_proj.register_forward_pre_hook(
            make_capture_hook()
        )

        with torch.no_grad():
            model(input_ids=full_ids.unsqueeze(0).to(DEVICE))

        h1.remove()
        h2.remove()

        if "act" in captured:
            act = captured["act"]
            resp_act = act[0, resp_start:, :]
            mean_act = resp_act.mean(dim=0).cpu().numpy()
            captured_acts.append(mean_act[capture_neurons])
        else:
            captured_acts.append(np.zeros(len(capture_neurons)))

    return np.stack(captured_acts, axis=0)  # (n_samples, n_capture_neurons)


# ---------------------------------------------------------------------------
# Discover causal edges for one language
# ---------------------------------------------------------------------------
def discover_causal_edges(
    model,
    lang: str,
    safety_neurons: Dict[int, List[int]],
    prepared_data: list,
    benign_means: Dict[int, torch.Tensor],
    acts_dir: str,
    n_subsample: int = 50,
    n_null_neurons: int = 20,
    min_z_delta: float = 2.0,
    min_rel_delta: float = 0.05,
) -> Dict:
    """Discover Type-B causal-influence edges via single-neuron ablation.

    For each layer l, ablate each safety neuron A individually, measure
    delta at each safety neuron B at layer l+1. Compare against null
    (ablating random non-safety neurons).
    """
    rng = np.random.default_rng(42)
    n_total = len(prepared_data)
    sample_indices = sorted(rng.choice(n_total, size=min(n_subsample, n_total), replace=False))

    all_edges = []
    layer_stats = {}

    safety_neuron_set_by_layer = {l: set(ns) for l, ns in safety_neurons.items()}

    for ablate_layer in tqdm(range(NUM_LAYERS - 1), desc=f"  [{lang}] causal edges"):
        if ablate_layer not in safety_neurons or not safety_neurons[ablate_layer]:
            continue
        capture_layer = ablate_layer + 1
        if capture_layer not in safety_neurons or not safety_neurons[capture_layer]:
            continue

        src_neurons = safety_neurons[ablate_layer]
        tgt_neurons = safety_neurons[capture_layer]

        # Load normal activations at capture layer for the subsampled prompts
        normal_all = load_normal_acts(acts_dir, lang, capture_layer)
        normal_sub = normal_all[sample_indices][:, tgt_neurons]  # (n_sub, n_tgt)
        # M_{l+1,j}: mean activation of each target neuron over the unsafe samples
        mean_act = normal_all[:, tgt_neurons].mean(axis=0)
        mean_act = np.where(np.abs(mean_act) > 1e-10, mean_act, np.nan)

        benign_mean = benign_means[ablate_layer]

        # --- Null distribution: ablate random NON-safety neurons ---
        all_neuron_indices = set(range(FFN_INTERMEDIATE_DIM))
        safety_set = safety_neuron_set_by_layer.get(ablate_layer, set())
        non_safety = sorted(all_neuron_indices - safety_set)
        null_neurons = sorted(rng.choice(non_safety, size=min(n_null_neurons, len(non_safety)), replace=False))

        null_deltas = []  # (n_null, n_tgt)
        for nn in null_neurons:
            ablated = ablate_single_neuron_and_capture(
                model, prepared_data, sample_indices,
                ablate_layer, nn, capture_layer, tgt_neurons, benign_mean,
            )
            delta = (normal_sub - ablated).mean(axis=0)  # (n_tgt,)
            null_deltas.append(delta)
        null_matrix = np.stack(null_deltas, axis=0)  # (n_null, n_tgt)
        null_mean = null_matrix.mean(axis=0)  # (n_tgt,)
        null_std = null_matrix.std(axis=0)  # (n_tgt,)
        # Avoid division by zero: where std==0, set to inf so z_delta becomes 0
        null_std = np.where(null_std > 0, null_std, np.inf)

        # --- Test each safety source neuron ---
        layer_edges = []
        for src_idx, src_neuron in enumerate(src_neurons):
            ablated = ablate_single_neuron_and_capture(
                model, prepared_data, sample_indices,
                ablate_layer, src_neuron, capture_layer, tgt_neurons, benign_mean,
            )
            # Per-target-neuron effect
            delta = (normal_sub - ablated).mean(axis=0)  # Eq. 6, (n_tgt,)
            rel_delta = delta / mean_act                  # Delta / M
            z_delta = (delta - null_mean) / null_std

            for tgt_idx, tgt_neuron in enumerate(tgt_neurons):
                if z_delta[tgt_idx] >= min_z_delta and rel_delta[tgt_idx] >= min_rel_delta:
                    layer_edges.append({
                        "source_layer": ablate_layer,
                        "source_neuron": int(src_neuron),
                        "target_layer": capture_layer,
                        "target_neuron": int(tgt_neuron),
                        "delta": round(float(delta[tgt_idx]), 6),
                        "rel_delta": round(float(rel_delta[tgt_idx]), 6),
                        "z_delta": round(float(z_delta[tgt_idx]), 4),
                        "null_mean": round(float(null_mean[tgt_idx]), 6),
                        "null_std": round(float(null_std[tgt_idx]), 6) if np.isfinite(null_std[tgt_idx]) else 999999.0,
                    })

        all_edges.extend(layer_edges)
        layer_stats[str(ablate_layer)] = {
            "n_src_neurons": len(src_neurons),
            "n_tgt_neurons": len(tgt_neurons),
            "n_null_neurons": len(null_neurons),
            "n_edges": len(layer_edges),
        }
        if layer_edges:
            print(f"    layer {ablate_layer}->{capture_layer}: "
                  f"{len(layer_edges)} causal edges "
                  f"(from {len(src_neurons)}×{len(tgt_neurons)} pairs)")

    # Sort by z_delta descending
    all_edges.sort(key=lambda e: -e["z_delta"])

    return {
        "total_edges": len(all_edges),
        "n_subsample": len(sample_indices),
        "n_null_neurons_per_layer": n_null_neurons,
        "thresholds": {"min_z_delta": min_z_delta, "min_rel_delta": min_rel_delta},
        "layer_stats": layer_stats,
        "edges": all_edges,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Activation-propagation (Type-B) pathways")
    parser.add_argument("--languages", nargs="+", required=True)
    parser.add_argument("--neurons-dir", required=True, help="output of attribute_safety_neurons.py")
    parser.add_argument("--acts-dir", required=True, help="output of extract_ffn_activations.py")
    parser.add_argument("--prepared-dir", required=True, help="output of extract_ffn_activations.py")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16",
                        help="model dtype for this step")
    parser.add_argument("--n-subsample", type=int, default=50,
                        help="Number of prompts to subsample for ablation")
    parser.add_argument("--n-null", type=int, default=20,
                        help="Number of random non-safety neurons for null distribution per layer")
    parser.add_argument("--min-z-delta", type=float, default=2.0)
    parser.add_argument("--min-rel-delta", type=float, default=0.05)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"{'='*60}\nACTIVATION-PROPAGATION (TYPE-B) PATHWAYS\n{'='*60}")
    print(f"Languages       : {args.languages}")
    print(f"Neurons dir     : {args.neurons_dir}")
    print(f"Acts dir        : {args.acts_dir}")
    print(f"Prepared dir    : {args.prepared_dir}")
    print(f"Output dir      : {args.output_dir}")
    print(f"n_subsample     : {args.n_subsample}")
    print(f"n_null          : {args.n_null}")
    print(f"Device          : {DEVICE}")
    print(f"min_z_delta     : {args.min_z_delta}")
    print(f"min_rel_delta   : {args.min_rel_delta}")
    print(f"{'='*60}\n")

    # Load model
    from transformers import AutoModelForCausalLM
    print(f"Loading model {MODEL_NAME} ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=getattr(torch, args.dtype), device_map=DEVICE,
    )
    model.eval()

    for lang in args.languages:
        print(f"\n[{lang}] loading data...")
        safety_neurons = load_safety_neurons(args.neurons_dir, lang)
        benign_means = load_benign_means(args.acts_dir, lang)
        prepared = load_prepared_data(args.prepared_dir, lang)
        total_n = sum(len(v) for v in safety_neurons.values())
        print(f"  safety neurons: {total_n}, prepared samples: {len(prepared)}")
        result = discover_causal_edges(
            model, lang, safety_neurons, prepared, benign_means, args.acts_dir,
            n_subsample=args.n_subsample,
            n_null_neurons=args.n_null,
            min_z_delta=args.min_z_delta,
            min_rel_delta=args.min_rel_delta,
        )
        print(f"[{lang}] total Type-B edges: {result['total_edges']}")

        lang_out = os.path.join(args.output_dir, f"type_B_paths_{lang}.json")
        with open(lang_out, "w") as f:
            json.dump({lang: result}, f, indent=2)
        print(f"Saved: {lang_out}")


if __name__ == "__main__":
    main()
