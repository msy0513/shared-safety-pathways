"""
Loading safety neurons and safety pathways.

File formats (produced by the scripts in ``pathway/``)
-----------------------------------------------------
safety neurons   <neurons_dir>/ffn_safety_neurons_<LANG>.json
                 {"<layer>": {"indices": [...], "scores": {...}}, ...}
co-activation    <type_A_file>  (type_A_paths_by_lang.json)
  pathways (P^A) {"<LANG>": {"edges": [{"source_layer", "source_neuron",
                                        "target_layer", "target_neuron", ...}]}}
activation-      <type_B_dir>/type_B_paths_<LANG>.json
  propagation    {"<LANG>": {"edges": [...]}}
  pathways (P^B) (a nested layout <type_B_dir>/<LANG>/type_B_paths_<LANG>.json is
                 also accepted)

A safety pathway P_lambda = P^A_lambda ∪ P^B_lambda is a set of cross-layer edges;
fine-tuning operates on its node set, i.e. the (layer, neuron) endpoints of all edges.
Node sets are returned as ``{layer: sorted([neuron, ...])}``.

A missing Type-A / Type-B file raises an error unless ``strict=False`` is passed
explicitly (otherwise "A ∪ B" would silently become "A only").
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

NeuronsByLayer = Dict[int, List[int]]


def load_safety_neurons(neurons_dir: str, lang: str) -> NeuronsByLayer:
    """Safety neurons S_lambda (Eq. 2) written by pathway/attribute_safety_neurons.py."""
    path = os.path.join(neurons_dir, f"ffn_safety_neurons_{lang}.json")
    with open(path) as f:
        data = json.load(f)
    out: NeuronsByLayer = {}
    for layer_str, entry in data.items():
        idx = entry["indices"] if isinstance(entry, dict) else entry
        if idx:
            out[int(layer_str)] = list(idx)
    return out


def resolve_type_b_file(type_B_dir: str, lang: str, strict: bool = True) -> Optional[str]:
    """Return the Type-B file of ``lang`` (flat or nested layout)."""
    candidates = [
        os.path.join(type_B_dir, f"type_B_paths_{lang}.json"),
        os.path.join(type_B_dir, lang, f"type_B_paths_{lang}.json"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    if strict:
        raise FileNotFoundError(
            f"No activation-propagation (Type-B) file for {lang} under {type_B_dir}; "
            f"tried {candidates}.")
    return None


def load_type_a_edges(type_A_file: str, lang: str, strict: bool = True) -> List[dict]:
    if not os.path.exists(type_A_file):
        if strict:
            raise FileNotFoundError(f"Co-activation (Type-A) file not found: {type_A_file}")
        return []
    with open(type_A_file) as f:
        data = json.load(f)
    if strict and lang not in data:
        raise KeyError(f"{lang} not in {type_A_file} (has {sorted(data)})")
    return data.get(lang, {}).get("edges", [])


def load_type_b_edges(type_B_dir: str, lang: str, strict: bool = True) -> List[dict]:
    path = resolve_type_b_file(type_B_dir, lang, strict=strict)
    if path is None:
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get(lang, {}).get("edges", [])


def _add_edge_nodes(store: Dict[int, Set[int]], edges: List[dict]) -> None:
    for e in edges:
        store[e["source_layer"]].add(e["source_neuron"])
        store[e["target_layer"]].add(e["target_neuron"])


def load_pathway_neurons(type_A_file: str, type_B_dir: str, lang: str,
                         mode: str = "union", strict: bool = True) -> NeuronsByLayer:
    """Node set of the safety pathway of ``lang``.

    mode: "union" = P^A ∪ P^B (the pathway P_lambda), "type_a" = P^A, "type_b" = P^B.
    """
    if mode not in ("union", "type_a", "type_b"):
        raise ValueError(f"unknown pathway mode: {mode}")
    nodes: Dict[int, Set[int]] = defaultdict(set)
    if mode in ("union", "type_a"):
        _add_edge_nodes(nodes, load_type_a_edges(type_A_file, lang, strict))
    if mode in ("union", "type_b"):
        _add_edge_nodes(nodes, load_type_b_edges(type_B_dir, lang, strict))
    return {l: sorted(v) for l, v in nodes.items() if v}


def pathway_node_set(type_A_file: str, type_B_dir: str, lang: str,
                     strict: bool = True) -> Set[Tuple[int, int]]:
    """Node set of P_lambda as (layer, neuron) tuples."""
    nodes = load_pathway_neurons(type_A_file, type_B_dir, lang, "union", strict)
    return {(l, n) for l, ns in nodes.items() for n in ns}


def load_shared_pathway_neurons(type_A_file: str, type_B_dir: str, target_lang: str,
                                hr_lang: str = "EN", strict: bool = True) -> NeuronsByLayer:
    """Shared safety pathway P*_lambda = P_lambda ∩ P_HR (Sec. 3.3)."""
    shared = (pathway_node_set(type_A_file, type_B_dir, hr_lang, strict)
              & pathway_node_set(type_A_file, type_B_dir, target_lang, strict))
    out: Dict[int, List[int]] = defaultdict(list)
    for layer, neuron in sorted(shared):
        out[layer].append(neuron)
    return dict(out)


def count_neurons(neurons: NeuronsByLayer) -> int:
    return sum(len(v) for v in neurons.values())
