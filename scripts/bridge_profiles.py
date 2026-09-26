#!/usr/bin/env python3
"""
BRIDGE: a knowledge bridge from the LLM's world knowledge to the recommender.

ReDial is a single domain, so BRIDGE's original job -- adapting across domains --
has nothing to adapt between. The gap that does exist is between what the
recommender knows about an item (a title and whatever co-occurrence signal
training provided; 23% of the catalogue is never a training answer) and what a
large LLM knows about the same movie. This script writes that knowledge down
once: a one-line profile per catalogue item (genre, era, tone, premise),
generated greedily by an instruct model and told to say "unknown" rather than
guess. Downstream agents (STAR) read the profiles, so they can judge movies the
retriever has barely seen.

    python scripts/bridge_profiles.py --model /tmp/harpo_rebuild/models/Qwen2.5-7B-Instruct \\
        --catalog .../catalog.json --out .../bridge_profiles.json
"""

import argparse
import json
import os
import time

import torch

PROMPT = ("Describe the movie {title} in one short line: genre, era, tone and what it is "
          "about. If you do not recognise it, answer exactly: unknown")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    local = os.path.isdir(args.model)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=local)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        local_files_only=local).to(device).eval()

    with open(args.catalog) as f:
        titles = json.load(f)
    if args.limit:
        titles = titles[:args.limit]
    profiles = {}
    if os.path.exists(args.out):  # resumable
        with open(args.out) as f:
            profiles = json.load(f)
    todo = [t for t in titles if t not in profiles]
    print(f"{len(titles)} titles, {len(todo)} to generate", flush=True)

    start = time.time()
    for s in range(0, len(todo), args.batch):
        chunk = todo[s:s + args.batch]
        prompts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(title=t)}],
                                           add_generation_prompt=True, tokenize=False)
                   for t in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        texts = tok.batch_decode(out[:, enc["input_ids"].size(1):], skip_special_tokens=True)
        for t, text in zip(chunk, texts):
            line = " ".join(text.strip().split())
            profiles[t] = "" if line.lower().startswith("unknown") else line
        if (s // args.batch) % 10 == 0:
            done = s + len(chunk)
            rate = (time.time() - start) / done
            print(f"  {done}/{len(todo)}  ({rate * (len(todo) - done) / 60:.1f} min left)  "
                  f"e.g. {chunk[0]!r} -> {profiles[chunk[0]]!r}", flush=True)
            with open(args.out + ".tmp", "w") as f:
                json.dump(profiles, f)
            os.replace(args.out + ".tmp", args.out)
    with open(args.out + ".tmp", "w") as f:
        json.dump(profiles, f, indent=0)
    os.replace(args.out + ".tmp", args.out)
    known = sum(1 for v in profiles.values() if v)
    print(f"wrote {args.out}: {known}/{len(profiles)} profiled, "
          f"{len(profiles) - known} unknown, in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
