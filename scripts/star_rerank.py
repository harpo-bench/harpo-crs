#!/usr/bin/env python3
"""
STAR scores for a saved shortlist (see harpo/star.py).

Reads a rerank_eval.py raw dump, rebuilds its rows, and for each distinct
dialogue runs --branches branches of "reason, then choose among the top-M"
with an instruct model, using BRIDGE profiles as candidate descriptions.
Writes ``[N, K]`` scores aligned to the dump: the first M columns are STAR's
aggregated log choice-probabilities; the rest get the row mean, so STAR is
neutral about candidates it did not see.

    python scripts/star_rerank.py --model /tmp/harpo_rebuild/models/Qwen2.5-7B-Instruct \\
        --raw .../rerank_top100_raw.pt --profiles .../bridge/profiles.json --out .../star_test.pt
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--profiles", default=None)
    parser.add_argument("--data", default=os.path.expanduser("~/Desktop/harpo-work/redial_data"))
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--charm-fraction", type=float, default=0.25)
    parser.add_argument("--top-m", type=int, default=20)
    parser.add_argument("--branches", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="dialogues (smoke tests)")
    parser.add_argument("--adapter", default=None,
                        help="trained selector (scripts/train_star.py); implies --direct")
    parser.add_argument("--direct", action="store_true",
                        help="read the choice straight away, without generated reasoning")
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from run_experiment import expand_mentions, load_split, split_conversations
    from harpo.star import (ANSWER_CUE, LABELS, aggregate, branch_order, build_messages,
                            clean_dialogue, direct_prompt, trim_reasoning)

    assert args.top_m <= len(LABELS)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw = torch.load(args.raw)
    catalog = raw["catalog"]
    index = {t.lower(): i for i, t in enumerate(catalog)}
    if args.split == "test":
        rows = expand_mentions(load_split(os.path.join(args.data, "test_sft.json"), 0, args.seed))
    else:
        _, val_raw, _ = split_conversations(
            load_split(os.path.join(args.data, "sft_data.json"), 0, args.seed),
            args.val_fraction, args.charm_fraction)
        rows = expand_mentions(val_raw)
    rows = [r for r in rows if str(r["ground_truth_item"]).lower() in index]
    targets = torch.tensor([index[str(r["ground_truth_item"]).lower()] for r in rows])
    if not torch.equal(targets, raw["targets"].cpu()):
        raise ValueError("rows do not line up with the raw dump (different data or split)")
    profiles = {}
    if args.profiles and os.path.exists(args.profiles):
        with open(args.profiles) as f:
            profiles = json.load(f)
    print(f"device={device}  rows={len(rows)}  profiles={sum(1 for v in profiles.values() if v)}",
          flush=True)

    local = os.path.isdir(args.model)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=local)
    tok.padding_side = "left"
    tok.truncation_side = "left"
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    lm = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        local_files_only=local).to(device).eval()
    direct = args.direct or bool(args.adapter)
    if args.adapter:
        from peft import PeftModel
        lm = PeftModel.from_pretrained(lm, args.adapter).eval()
        print(f"trained selector: {args.adapter}", flush=True)
    label_ids = torch.tensor([tok(" " + L, add_special_tokens=False)["input_ids"][0]
                              for L in LABELS[:args.top_m]], device=device)

    first = {}
    for i, r in enumerate(rows):
        first.setdefault(r["input"], i)
    dialogues = list(first)
    if args.limit:
        dialogues = dialogues[:args.limit]
    m = args.top_m
    top_idx = raw["top_idx"][:, :m]
    per_dialogue = torch.zeros(len(dialogues), args.branches, m)
    examples = []
    start = time.time()

    for b in range(args.branches):
        for s in range(0, len(dialogues), args.batch):
            chunk = dialogues[s:s + args.batch]
            orders, prompts = [], []
            for d in chunk:
                cands = [catalog[c] for c in top_idx[first[d]].tolist()]
                order = branch_order(m, b, d)
                orders.append(order)
                shown = [(cands[j], profiles.get(cands[j], "")) for j in order]
                prompts.append(direct_prompt(tok, d, shown) if direct else
                               tok.apply_chat_template(build_messages(clean_dialogue(d), shown),
                                                       add_generation_prompt=True, tokenize=False))
            if direct:
                reasons, full = [""] * len(prompts), prompts
            else:
                enc = tok(prompts, return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    gen = lm.generate(**enc, max_new_tokens=args.max_new_tokens, pad_token_id=pad_id,
                                      do_sample=b > 0, temperature=args.temperature if b > 0 else None,
                                      top_p=0.95 if b > 0 else None)
                reasons = [trim_reasoning(t) for t in tok.batch_decode(
                    gen[:, enc["input_ids"].size(1):], skip_special_tokens=True)]
                full = [p + r + ANSWER_CUE for p, r in zip(prompts, reasons)]
            enc2 = tok(full, return_tensors="pt", padding=True, truncation=True,
                       max_length=args.max_length).to(device)
            with torch.no_grad():
                logits = lm(**enc2, logits_to_keep=1).logits[:, -1].float()
            logp = F.log_softmax(logits[:, label_ids], dim=-1).cpu()        # [b, m] by label
            for n, order in enumerate(orders):
                # label position p showed original candidate order[p]
                per_dialogue[s + n, b, torch.tensor(order)] = logp[n]
            if b == 0 and len(examples) < 5:
                examples.append({"dialogue_tail": clean_dialogue(chunk[0])[-300:],
                                 "reasoning": reasons[0],
                                 "choice": catalog[top_idx[first[chunk[0]]][int(logp[0].argmax())]]})
            done = b * len(dialogues) + s + len(chunk)
            if (s // args.batch) % 20 == 0:
                rate = (time.time() - start) / done
                print(f"  branch {b + 1}/{args.branches}  {s + len(chunk)}/{len(dialogues)}  "
                      f"(~{rate * (args.branches * len(dialogues) - done) / 60:.0f} min left)",
                      flush=True)

    agg = torch.stack([aggregate(per_dialogue[i]) for i in range(len(dialogues))])   # [D, m]
    slot = {d: n for n, d in enumerate(dialogues)}
    k = raw["top_idx"].size(1)
    scores = torch.zeros(len(rows), k)
    covered = torch.zeros(len(rows), dtype=torch.bool)
    for i, r in enumerate(rows):
        if r["input"] in slot:
            row = agg[slot[r["input"]]]
            scores[i, :m] = row
            scores[i, m:] = row.mean()
            covered[i] = True
    # Quick diagnostic: STAR's own top-1 among the top-M, on the covered rows.
    hit = top_idx[covered] == targets[covered, None]
    in_m = hit.any(1)
    star_top1 = (scores[covered][:, :m].argmax(1) == hit.float().argmax(1)) & in_m
    retr_top1 = hit[:, 0]
    print(f"STAR in {(time.time() - start) / 60:.1f} min over {int(covered.sum())} rows; "
          f"top-1 accuracy (target in top-{m}: {100 * float(in_m.float().mean()):.1f}%): "
          f"STAR {100 * float(star_top1.float().mean()):.2f}%  vs retriever "
          f"{100 * float(retr_top1.float().mean()):.2f}%", flush=True)
    for e in examples[:3]:
        print(f"  reasoning: {e['reasoning']!r} -> {e['choice']!r}")
    torch.save({"scores": scores, "targets": targets, "covered": covered, "top_m": m,
                "branches": args.branches, "examples": examples}, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
