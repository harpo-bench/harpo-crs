#!/usr/bin/env python3
"""
Score a saved re-ranking shortlist with a trained CHARM cross-encoder.

The cross-encoder scores (dialogue, candidate) pairs, so it can re-rank any
retriever's shortlist -- e.g. the 7B system's, although it was trained on the
0.5B retriever's hard negatives. Reads the ``rerank_top*_raw.pt`` dump written by
rerank_eval.py, rebuilds the same rows, and writes ``[N, K]`` scores aligned to it.

    python scripts/score_shortlist_ce.py --adapter .../charm_ce_7b/charm_ce_adapter \\
        --base-model /tmp/harpo_rebuild/models/Qwen2.5-7B-Instruct --dtype bfloat16 \\
        --raw .../7b_6ep/rerank_top50_raw.pt --out .../7b_6ep/charm_ce_7b_scores.pt
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--raw", required=True, help="rerank_eval.py raw dump")
    parser.add_argument("--data", default=os.path.expanduser("~/Desktop/harpo-work/redial_data"))
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--charm-fraction", type=float, default=0.25)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--score-dialogues", type=int, default=0,
                        help="dialogues per batch; default: --pair-budget / shortlist size")
    parser.add_argument("--pair-budget", type=int, default=800,
                        help="candidate sequences per batch: each shares its dialogue's KV "
                             "cache, so memory grows with dialogues x candidates (32 x 100 "
                             "ran a 7B out of memory on an 80 GB A100)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from run_experiment import expand_mentions, load_split, split_conversations
    from harpo.charm_ce import CrossEncoderCHARM

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

    ce = CrossEncoderCHARM(args.base_model, device, max_length=args.max_length,
                           dtype=getattr(torch, args.dtype))
    ce.load_adapter(args.adapter)

    first = {}
    for i, r in enumerate(rows):
        first.setdefault(r["input"], i)
    dialogues = list(first)
    top_idx = raw["top_idx"]
    groups = [[catalog[c] for c in top_idx[first[d]].tolist()] for d in dialogues]
    per_batch = args.score_dialogues or max(1, args.pair_budget // top_idx.size(1))
    print(f"scoring {len(dialogues)} dialogues x {top_idx.size(1)} candidates, "
          f"{per_batch} dialogues per batch", flush=True)
    start = time.time()
    per_dialogue = ce.score_groups(dialogues, groups, per_batch).cpu()
    slot = {d: n for n, d in enumerate(dialogues)}
    scores = per_dialogue[[slot[r["input"]] for r in rows]]
    print(f"scored {len(dialogues)} dialogues x {top_idx.size(1)} candidates "
          f"in {(time.time() - start) / 60:.1f} min")
    # Write then rename, so a queue polling for the file never reads half of it.
    torch.save({"scores": scores, "targets": targets, "adapter": args.adapter}, args.out + ".partial")
    os.replace(args.out + ".partial", args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
