"""
Machine translation of the English data into the target languages.

Back-ends:

  google   Google Translate web endpoint (no key).
  llm      An OpenAI-compatible chat model translating batches of 20 sentences
           (set LLM_API_KEY, LLM_BASE_URL, LLM_MODEL).  When a batch fails after all
           retries, its English sentences are kept, so check the output.

Sub-commands
------------
  train   EN  train_data/en/final_safe.jsonl  -> <out>/<lang>/final_safe.jsonl
          (translates both "instruction" and "response"; resumable)
  benign  EN  benign/en.json (list[str])      -> benign/<lang>.json
  eval    EN  eval/<bench>/en.json {"safety"} -> eval/<bench>/<lang>.json

Examples
--------
  python data/scripts/translate.py train  --src data/train_data/en/final_safe.jsonl \
      --out data/train_data/sw/final_safe.jsonl --lang sw
  python data/scripts/translate.py benign --src data/benign/en.json \
      --out data/benign/sw.json --lang sw
  python data/scripts/translate.py eval   --src data/eval/multijail_json/en.json \
      --out data/eval/multijail_json/ne.json --lang ne
"""

import argparse
import json
import os
import sys
import time
from typing import List

# Language names for the LLM back-end (Google uses ISO codes directly).
LANG_NAMES = {
    "zh-CN": "Chinese (Simplified)", "ko": "Korean", "bn": "Bengali", "th": "Thai",
    "sw": "Swahili", "hu": "Hungarian", "af": "Afrikaans", "it": "Italian",
    "ne": "Nepali", "ja": "Japanese", "ar": "Arabic",
}


# ---------------------------------------------------------------------------
# Back-ends
# ---------------------------------------------------------------------------
def google_translate(text: str, to_lang: str, retries: int = 3, src: str = "en") -> str:
    """Google Translate web endpoint; returns the input unchanged after `retries` failures."""
    import requests

    url = "https://translate.googleapis.com/translate_a/single"
    params = {"client": "gtx", "sl": src, "tl": to_lang, "dt": "t", "q": text}
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=20)
            res = json.loads(r.text)
            return "".join(p[0] for p in res[0] if p[0])
        except Exception as e:  # noqa: BLE001
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                print(f"  FAIL ({to_lang}): {e}", file=sys.stderr)
                return text


def llm_translate_batch(client, model: str, texts: List[str], lang_name: str,
                        retries: int = 3) -> List[str]:
    """Translate a batch with an OpenAI-compatible chat model."""
    prompt = (
        f"Translate the following English sentences to {lang_name}. "
        f"Return ONLY a JSON array of translated strings, one per input sentence. "
        f"Do not add explanations.\n\n" + json.dumps(texts, ensure_ascii=False)
    )
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}], temperature=0.3,
            )
            content = resp.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0]
            result = json.loads(content)
            if isinstance(result, list) and len(result) == len(texts):
                return result
            print(f"  batch returned {len(result)} items, expected {len(texts)}; retrying")
        except Exception as e:  # noqa: BLE001
            print(f"  attempt {attempt + 1} failed: {e}")
            time.sleep(5 * (attempt + 1))
    print("  batch FAILED - keeping the English originals", file=sys.stderr)
    return list(texts)


def translate_list(texts: List[str], lang: str, backend: str, sleep: float = 0.2,
                   batch_size: int = 20, done: List[str] = None) -> List[str]:
    """Translate ``texts``. Items of a previous run (``done``) are reused, except items
    that are still identical to the English source (failed translations), which are
    translated again."""
    out = list(done or [])[: len(texts)]
    out += texts[len(out):]
    todo = [i for i, (src, cur) in enumerate(zip(texts, out))
            if i >= len(done or []) or cur.strip() == src.strip()]
    print(f"  {len(todo)} / {len(texts)} items to translate")

    if backend == "google":
        for k, i in enumerate(todo):
            out[i] = google_translate(texts[i], lang)
            time.sleep(sleep)
            if (k + 1) % 50 == 0:
                print(f"  {k + 1}/{len(todo)}", flush=True)
    else:
        from openai import OpenAI

        api_key, model = os.environ.get("LLM_API_KEY"), os.environ.get("LLM_MODEL")
        if not api_key or not model:
            raise SystemExit("backend=llm needs LLM_API_KEY and LLM_MODEL (and optionally LLM_BASE_URL)")
        client = OpenAI(api_key=api_key, base_url=os.environ.get("LLM_BASE_URL"))
        lang_name = LANG_NAMES.get(lang, lang)
        for start in range(0, len(todo), batch_size):
            idx = todo[start:start + batch_size]
            for i, t in zip(idx, llm_translate_batch(client, model, [texts[i] for i in idx], lang_name)):
                out[i] = t
            time.sleep(sleep)
            print(f"  {min(start + batch_size, len(todo))}/{len(todo)}", flush=True)

    failed = sum(1 for src, cur in zip(texts, out) if cur.strip() == src.strip())
    if failed:
        print(f"  WARNING: {failed} items are still identical to English; run the same "
              f"command again to retry them", file=sys.stderr)
    return out


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------
def cmd_train(args):
    """Translate (instruction, response) pairs line by line (resumable)."""
    items = [json.loads(l) for l in open(args.src, encoding="utf-8") if l.strip()]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    done = [json.loads(l) for l in open(args.out, encoding="utf-8")] if os.path.exists(args.out) else []
    if done:
        print(f"resume from {len(done)}")
    with open(args.out, "a" if done else "w", encoding="utf-8") as f:
        for i, it in enumerate(items[len(done):], start=len(done)):
            ins = google_translate(it["instruction"], args.lang)
            time.sleep(args.sleep)
            res = google_translate(it["response"], args.lang)
            time.sleep(args.sleep)
            f.write(json.dumps({"instruction": ins, "response": res}, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(items)}", flush=True)
    out = [json.loads(l) for l in open(args.out, encoding="utf-8")]
    failed = sum(1 for a, b in zip(items, out)
                 if a["instruction"].strip() == b["instruction"].strip()
                 or a["response"].strip() == b["response"].strip())
    if failed:
        print(f"WARNING: {failed} pairs are still identical to English", file=sys.stderr)
    print("done", args.out)


def cmd_benign(args):
    texts = json.load(open(args.src, encoding="utf-8"))
    done = json.load(open(args.out, encoding="utf-8")) if os.path.exists(args.out) else []
    out = translate_list(texts, args.lang, args.backend, args.sleep, done=done)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"saved {len(out)} -> {args.out}")


def cmd_eval(args):
    texts = json.load(open(args.src, encoding="utf-8"))["safety"]
    out = translate_list(texts, args.lang, args.backend, args.sleep)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"safety": out}, f, ensure_ascii=False, indent=2)
    print(f"saved {len(out)} -> {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("train", cmd_train), ("benign", cmd_benign), ("eval", cmd_eval)):
        p = sub.add_parser(name)
        p.add_argument("--src", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--lang", required=True, help="Google language code, e.g. zh-CN, ja, sw")
        p.add_argument("--backend", choices=["google", "llm"], default="google")
        p.add_argument("--sleep", type=float, default=0.2)
        p.set_defaults(func=fn)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
