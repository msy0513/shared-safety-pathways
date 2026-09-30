"""
Safety pathways-targeted cross-lingual alignment (Sec. 4, Eq. 8).

    theta_{t+1} = theta_t - eta * ( m ⊙ grad_theta L(D_train) )

m is a binary mask that keeps gradients only for the parameters associated with the HR
safety pathway P_HR = P^A_HR ∪ P^B_HR: for neuron i of layer l, row i of ``gate_proj`` and
``up_proj`` and column i of ``down_proj``.  Attention, embeddings, norms and all other FFN
neurons stay frozen.

D_train is the cross-lingual safety corpus {(x_HR, y_HR), (x_NHR, y_NHR)}: harmful queries
with safe responses in the HR language and their translations into the NHR languages
(data/train_data/<lang>/final_safe.jsonl).  L is the negative log-likelihood of the safe
responses.

Usage
  python training/pathway_sft.py \
      --type-A-file $OUT_ROOT/type_A/type_A_paths_by_lang.json --type-B-dir $OUT_ROOT/type_B \
      --train-data-dir data/train_data --target-langs EN ZH KO BN TH SW HU AF IT NE \
      --output-dir $OUT_ROOT/ckpt/pathway_sft
"""

import argparse
import json
import os
import random
import sys
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lingualpath.config import LANG_TO_TRAINDIR, MODEL_NAME
from lingualpath.pathways import count_neurons, load_pathway_neurons


# ---------------------------------------------------------------------------
# Gradient mask m
# ---------------------------------------------------------------------------
def build_gradient_masks(model, path_neurons: Dict[int, List[int]]) -> Dict[int, torch.Tensor]:
    """{id(weight): 0/1 mask}: rows of gate/up_proj and columns of down_proj of the pathway neurons."""
    masks = {}
    for layer_idx, neuron_indices in path_neurons.items():
        mlp = model.model.layers[layer_idx].mlp
        idx = torch.tensor(neuron_indices, dtype=torch.long)
        for proj in (mlp.gate_proj, mlp.up_proj):
            m = torch.zeros_like(proj.weight)
            m[idx, :] = 1.0
            masks[id(proj.weight)] = m
        m = torch.zeros_like(mlp.down_proj.weight)
        m[:, idx] = 1.0
        masks[id(mlp.down_proj.weight)] = m
    return masks


def apply_gradient_mask(model, masks) -> None:
    for p in model.parameters():
        if p.grad is not None and id(p) in masks:
            p.grad.mul_(masks[id(p)])


# ---------------------------------------------------------------------------
# D_train
# ---------------------------------------------------------------------------
class SafetyCorpus(Dataset):
    """(harmful query, safe response) pairs of the training languages; loss on the response."""

    def __init__(self, train_data_dir, target_langs, tokenizer, max_length=512, seed=42):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        for lang in target_langs:
            path = os.path.join(train_data_dir, LANG_TO_TRAINDIR.get(lang.upper(), lang.lower()),
                                "final_safe.jsonl")
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        x, y = row.get("instruction", "").strip(), row.get("response", "").strip()
                        if x and y:
                            self.samples.append((x, y))
        random.Random(seed).shuffle(self.samples)
        print(f"  D_train: {len(self.samples)} pairs from {len(target_langs)} languages")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": x}, {"role": "assistant", "content": y}],
            tokenize=False, add_generation_prompt=False)
        input_ids = self.tokenizer(text, max_length=self.max_length, truncation=True,
                                   return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": x}], tokenize=False, add_generation_prompt=True)
        prompt_len = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        return {"input_ids": input_ids, "labels": labels}


def collate_fn(batch, pad_token_id):
    max_len = max(b["input_ids"].size(0) for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, b in enumerate(batch):
        n = b["input_ids"].size(0)
        input_ids[i, :n] = b["input_ids"]
        labels[i, :n] = b["labels"]
        attention_mask[i, :n] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


# ---------------------------------------------------------------------------
def train(args):
    from transformers import (Adafactor, AutoModelForCausalLM, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    torch.manual_seed(args.seed)
    path_neurons = load_pathway_neurons(args.type_A_file, args.type_B_dir, args.hr_lang, "union")
    n_path = count_neurons(path_neurons)
    print(f"Safety pathway P_{args.hr_lang}: {n_path} neurons in {len(path_neurons)} layers")
    if n_path == 0:
        raise SystemExit("The safety pathway is empty; check the pathway files and thresholds.")

    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze the FFN projections of pathway layers; the gradient mask m restricts the
    # update to the pathway neurons.
    masked_params = 0
    for layer_idx, neuron_indices in path_neurons.items():
        mlp = model.model.layers[layer_idx].mlp
        for proj in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
            proj.weight.requires_grad = True
        masked_params += len(neuron_indices) * (2 * mlp.gate_proj.weight.shape[1] + mlp.down_proj.weight.shape[0])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Updated parameters: {masked_params:,} / {total_params:,} ({100 * masked_params / total_params:.2f}%)")
    masks = build_gradient_masks(model, path_neurons)

    target_langs = [l.upper() for l in args.target_langs]
    dataset = SafetyCorpus(args.train_data_dir, target_langs, tokenizer, args.max_length, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id))

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adafactor(trainable, lr=args.lr, scale_parameter=False, relative_step=False,
                          warmup_init=False, weight_decay=args.weight_decay)
    ga = args.grad_accum_steps
    total_steps = (len(loader) + ga - 1) // ga * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)
    print(f"Training: {args.epochs} epochs, batch {args.batch_size} x {ga}, {total_steps} optimizer steps, "
          f"lr {args.lr}")

    model.train()
    step_count = 0
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader):
            loss = model(**{k: v.to(model.device) for k, v in batch.items()}).loss
            (loss / ga).backward()
            apply_gradient_mask(model, masks)
            epoch_loss += loss.item()

            if (step + 1) % ga == 0 or (step + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step_count += 1
                if step_count % args.log_interval == 0:
                    print(f"  epoch {epoch + 1} step {step_count}/{total_steps} "
                          f"loss {epoch_loss / (step + 1):.4f} lr {scheduler.get_last_lr()[0]:.2e}")

        print(f"Epoch {epoch + 1}: mean loss {epoch_loss / len(loader):.4f}")
        if args.save_every_epoch:
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-epoch{epoch + 1}")
            model.save_pretrained(ckpt_dir, safe_serialization=True, max_shard_size="5GB")
            tokenizer.save_pretrained(ckpt_dir)

    final_dir = os.path.join(args.output_dir, "final")
    model.gradient_checkpointing_disable()
    model = model.to("cpu", dtype=torch.bfloat16)
    model.save_pretrained(final_dir, safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(final_dir)
    print(f"Saved -> {final_dir}")

    config = dict(vars(args))
    config.update({
        "model": MODEL_NAME,
        "target_langs": target_langs,
        "pathway_neurons": {str(l): list(map(int, ns)) for l, ns in path_neurons.items()},
        "n_pathway_neurons": n_path,
        "updated_params": masked_params,
        "total_params": total_params,
        "updated_ratio": masked_params / total_params,
        "n_train_samples": len(dataset),
    })
    with open(os.path.join(args.output_dir, "train_config.json"), "w") as f:
        json.dump(config, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description="Safety pathways-targeted cross-lingual alignment")
    ap.add_argument("--type-A-file", required=True, help="type_A_paths_by_lang.json")
    ap.add_argument("--type-B-dir", required=True, help="dir with type_B_paths_<LANG>.json")
    ap.add_argument("--train-data-dir", required=True, help="dir with <lang>/final_safe.jsonl")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--target-langs", nargs="+", required=True, help="languages of D_train")
    ap.add_argument("--hr-lang", default="EN", help="language whose safety pathway is updated")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum-steps", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-every-epoch", action="store_true",
                    help="also save a checkpoint after every epoch (for checkpoint selection)")
    ap.add_argument("--log-interval", type=int, default=10, help="in optimizer steps")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()
