"""
Step 2 — co-activation pathways P^A_lambda (Sec. 3.2 (2)-1, Eq. 3-5).

Candidate edges are all pairs (n_l^(i) -> n_{l+1}^(j)) of safety neurons in ADJACENT layers.
For every sample the response-mean-pooled activation of a neuron is binarised at the
neuron's median over the context (b = 1 means "activated").  On the unsafe context D^-:

    pA, pB           marginal activation rates
    pAB              joint co-activation rate                                   (Eq. 3)
    lift  = pAB / (pA pB)
    phi   = (pAB - pA pB) / sqrt(pA(1-pA) pB(1-pB))                             (Eq. 4)
    perm_p           permutation-test p-value of pAB (target neuron's samples shuffled)

An edge is kept if (Eq. 5)
    pAB >= --min-support (s_min)          phi >= --min-phi (phi_min)
    perm_p < --perm-alpha (alpha)         lift_unsafe / lift_benign >= --min-safety-ratio (rho)
where lift_benign is the same statistic on the benign context D^+ (safety-specificity
ratio).

Output (in --output-dir; language entries are merged into an existing file)
  type_A_paths_by_lang.json   {LANG: {"edges": [...], "params": ..., "layer_stats": ...}}
"""

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lingualpath.config import NUM_LAYERS
from lingualpath.pathways import load_safety_neurons


def _load_layer_acts(acts_dir: str, lang: str, data_type: str, layer: int) -> np.ndarray:
    path = os.path.join(acts_dir, lang.lower(), data_type, f"layer_{layer}.pt")
    t = torch.load(path, map_location="cpu", weights_only=False)
    return t.numpy() if isinstance(t, torch.Tensor) else np.asarray(t)


def _compute_lift_phi(
    acts_l: np.ndarray,
    acts_l_next: np.ndarray,
    neurons_l: List[int],
    neurons_l_next: List[int],
):
    n_samples = acts_l.shape[0]
    cols_l = acts_l[:, neurons_l]
    cols_next = acts_l_next[:, neurons_l_next]
    thr_l = np.median(cols_l, axis=0)
    thr_next = np.median(cols_next, axis=0)
    high_l = cols_l > thr_l
    high_next = cols_next > thr_next

    pA = high_l.mean(axis=0)
    pB = high_next.mean(axis=0)
    pAB = (high_l.astype(np.float32).T @ high_next.astype(np.float32)) / n_samples
    independent = np.outer(pA, pB)
    eps = 1e-10
    lift = pAB / (independent + eps)
    denom_phi = np.sqrt(np.outer(pA * (1 - pA), pB * (1 - pB)) + eps)
    phi = (pAB - independent) / denom_phi
    return pAB, pA, pB, lift, phi, high_l, high_next


def _layer_pair_edges_with_benign(
    acts_l_u: np.ndarray, acts_next_u: np.ndarray,
    acts_l_b: np.ndarray, acts_next_b: np.ndarray,
    neurons_l: List[int], neurons_l_next: List[int],
    min_support: float, min_phi: float,
    min_safety_ratio: float,
    n_permutations: int, perm_alpha: float,
) -> List[dict]:
    if not neurons_l or not neurons_l_next:
        return []

    pAB_u, pA_u, pB_u, lift_u, phi_u, high_l_u, high_next_u = _compute_lift_phi(
        acts_l_u, acts_next_u, neurons_l, neurons_l_next
    )

    # Initial filter on unsafe
    mask = (pAB_u >= min_support) & (phi_u >= min_phi)
    if not mask.any():
        return []
    i_idx, j_idx = np.where(mask)
    observed = pAB_u[i_idx, j_idx]
    n_samples_u = acts_l_u.shape[0]

    # Permutation test
    if n_permutations > 0:
        rng = np.random.default_rng(42)
        high_l_f = high_l_u.astype(np.float32)
        exceed = np.zeros(len(i_idx), dtype=np.int32)
        for _ in range(n_permutations):
            shuffled = high_next_u[rng.permutation(n_samples_u)].astype(np.float32)
            null = (high_l_f.T @ shuffled) / n_samples_u
            exceed += (null[i_idx, j_idx] >= observed).astype(np.int32)
        perm_p = (exceed + 1).astype(np.float64) / (n_permutations + 1)
    else:
        perm_p = np.zeros(len(i_idx))

    # Benign contrast: compute lift on the SAME (i,j) pairs using benign acts
    pAB_b, pA_b, pB_b, lift_b, phi_b, _, _ = _compute_lift_phi(
        acts_l_b, acts_next_b, neurons_l, neurons_l_next
    )

    edges = []
    for k, (i, j) in enumerate(zip(i_idx, j_idx)):
        if n_permutations > 0 and perm_p[k] >= perm_alpha:
            continue
        l_u = float(lift_u[i, j])
        l_b = float(lift_b[i, j])
        ratio = l_u / max(l_b, 1e-6)
        if ratio < min_safety_ratio:
            continue
        edges.append({
            "source_neuron": int(neurons_l[i]),
            "target_neuron": int(neurons_l_next[j]),
            "pAB": round(float(pAB_u[i, j]), 5),
            "pA": round(float(pA_u[i]), 5),
            "pB": round(float(pB_u[j]), 5),
            "lift": round(l_u, 4),
            "lift_benign": round(l_b, 4),
            "safety_ratio": round(ratio, 4),
            "phi": round(float(phi_u[i, j]), 5),
            "perm_p": round(float(perm_p[k]), 5) if n_permutations > 0 else None,
        })
    return edges


def build_paths_for_language(
    lang: str, neurons_by_layer: Dict[int, List[int]],
    acts_dir: str,
    min_support: float, min_phi: float,
    min_safety_ratio: float,
    n_permutations: int, perm_alpha: float,
) -> Dict:
    all_edges = []
    layer_stats = {}
    for l in range(NUM_LAYERS - 1):
        if not neurons_by_layer.get(l) or not neurons_by_layer.get(l + 1):
            continue
        try:
            acts_l_u = _load_layer_acts(acts_dir, lang, "safety", l)
            acts_next_u = _load_layer_acts(acts_dir, lang, "safety", l + 1)
            acts_l_b = _load_layer_acts(acts_dir, lang, "benign", l)
            acts_next_b = _load_layer_acts(acts_dir, lang, "benign", l + 1)
        except FileNotFoundError as e:
            print(f"  [{lang}] missing acts: {e}; skipping pair {l}->{l+1}")
            continue

        pair_edges = _layer_pair_edges_with_benign(
            acts_l_u, acts_next_u, acts_l_b, acts_next_b,
            neurons_by_layer[l], neurons_by_layer[l + 1],
            min_support, min_phi,
            min_safety_ratio,
            n_permutations, perm_alpha,
        )
        for e in pair_edges:
            e["source_layer"] = l
            e["target_layer"] = l + 1
            all_edges.append(e)
        layer_stats[str(l)] = {
            "n_src_neurons": len(neurons_by_layer[l]),
            "n_tgt_neurons": len(neurons_by_layer[l + 1]),
            "n_edges": len(pair_edges),
        }

    all_edges.sort(key=lambda e: -e["safety_ratio"])
    return {
        "neurons_per_layer_counts": {str(l): len(v) for l, v in neurons_by_layer.items()},
        "total_neurons": sum(len(v) for v in neurons_by_layer.values()),
        "total_edges": len(all_edges),
        "params": {
            "min_support": min_support,
            "min_phi": min_phi,
            "min_safety_ratio": min_safety_ratio,
            "n_permutations": n_permutations,
            "perm_alpha": perm_alpha,
        },
        "layer_stats": layer_stats,
        "edges": all_edges,
    }


def run(
    languages, neurons_dir, acts_dir, output_dir,
    min_support, min_phi, min_safety_ratio,
    n_permutations, perm_alpha,
):
    os.makedirs(output_dir, exist_ok=True)
    print(f"=" * 60)
    print(f"CO-ACTIVATION (TYPE-A) PATHWAYS WITH BENIGN CONTRAST")
    print(f"=" * 60)
    print(f"Languages={languages}  min_safety_ratio={min_safety_ratio}")

    paths_by_lang = {}
    for lang in languages:
        neurons = load_safety_neurons(neurons_dir, lang)
        total = sum(len(v) for v in neurons.values())
        print(f"[{lang}] safety neurons: {total}")
        result = build_paths_for_language(
            lang, neurons, acts_dir,
            min_support, min_phi, min_safety_ratio,
            n_permutations, perm_alpha,
        )
        print(f"[{lang}] total Type-A edges (after benign filter): {result['total_edges']}")
        paths_by_lang[lang] = result

    out = os.path.join(output_dir, "type_A_paths_by_lang.json")
    if os.path.exists(out):
        with open(out) as f:
            existing = json.load(f)
        existing.update(paths_by_lang)
        paths_by_lang = existing
    with open(out, "w") as f:
        json.dump(paths_by_lang, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--languages", nargs="+", required=True)
    p.add_argument("--neurons-dir", required=True)
    p.add_argument("--acts-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--min-support", type=float, default=0.03)
    p.add_argument("--min-phi", type=float, default=0.02)
    p.add_argument("--min-safety-ratio", type=float, default=1.5)
    p.add_argument("--n-permutations", type=int, default=100)
    p.add_argument("--perm-alpha", type=float, default=0.05)
    args = p.parse_args()

    run(
        languages=args.languages,
        neurons_dir=args.neurons_dir, acts_dir=args.acts_dir,
        output_dir=args.output_dir,
        min_support=args.min_support,
        min_phi=args.min_phi, min_safety_ratio=args.min_safety_ratio,
        n_permutations=args.n_permutations, perm_alpha=args.perm_alpha,
    )
