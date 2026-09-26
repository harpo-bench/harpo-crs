#!/usr/bin/env python3
"""
Cold-start probe: re-score a saved retriever with rebuilt vectors for rarely
recommended items (harpo/coldstart.py), without retraining.

Every variant is evaluated on validation and test, but chosen on validation
only: the variant that puts the most validation targets in the top --top
(the shortlist the re-ranking agents see), provided validation R@10 falls by at
most --max-r10-drop points. The choice is written to --out for
``rerank_eval.py --coldstart``.

    python scripts/coldstart_probe.py --checkpoint .../7b_val/checkpoints/sft_final \\
        --val-fraction 0.05 --charm-fraction 0.0 --profiles .../bridge/profiles.json --out probe.json
"""

import argparse
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--catalog-file", default=None)
    parser.add_argument("--data", default=os.path.expanduser("~/Desktop/harpo-work/redial_data"))
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--charm-fraction", type=float, default=0.25,
                        help="as the backbone was trained (its training rows give the counts)")
    parser.add_argument("--profiles", default=None, help="BRIDGE profiles (title -> text)")
    parser.add_argument("--top", type=int, default=200)
    parser.add_argument("--max-r10-drop", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from run_experiment import (build_eval_catalog, build_model, coldstart_parts,
                                encode_test_contexts, expand_mentions, load_split,
                                split_conversations, write_json)
    from harpo.coldstart import ColdStart, cold_items, training_counts
    from harpo.ranking import mentioned_in_context
    from harpo.training import HARPOMTv2Trainer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    catalog_file = args.catalog_file or os.path.join(args.checkpoint, "..", "..", "catalog.json")
    with open(catalog_file) as f:
        catalog_list = json.load(f)
    index = {t.lower(): i for i, t in enumerate(catalog_list)}
    n_items = len(catalog_list)

    def usable(rows):
        return [r for r in rows if str(r["ground_truth_item"]).lower() in index]

    train_raw, val_raw, _ = split_conversations(
        load_split(os.path.join(args.data, "sft_data.json"), 0, args.seed),
        args.val_fraction, args.charm_fraction)
    counts = training_counts(expand_mentions(train_raw), index, n_items)
    rows = {"val": usable(expand_mentions(val_raw)),
            "test": usable(expand_mentions(load_split(os.path.join(args.data, "test_sft.json"),
                                                      0, args.seed)))}
    profiles = {}
    if args.profiles:
        with open(args.profiles) as f:
            profiles = json.load(f)
    print(f"catalogue={n_items}  never recommended in training: {int((counts == 0).sum())} items  "
          + "  ".join(f"{k}={len(v)}" for k, v in rows.items()), flush=True)

    model, training_config = build_model(os.path.join(args.checkpoint, "base_model"),
                                         device, args.seq_len, 16, False)
    HARPOMTv2Trainer(model, training_config, device=device).load_checkpoint(args.checkpoint)
    model.eval()

    parts = coldstart_parts(model, catalog_list, device, profiles)
    _, trained = build_eval_catalog(model, catalog_list, device)
    vecs, _ = cold_items(parts, counts, ColdStart())
    gap = float((vecs - trained.embeddings.float()).abs().max())
    print(f"identity config vs trained catalogue: max |diff| {gap:.2e}", flush=True)
    assert gap < 1e-3, "the rebuilt catalogue does not reproduce the trained one"

    temp = float(model.retriever.temperature)
    split = {}
    for name, rs in rows.items():
        pooled, _ = encode_test_contexts(model, rs, device, args.seq_len)
        with torch.no_grad():
            ctx = model.retriever.encode_context(pooled).float()
        tgt = torch.tensor([index[str(r["ground_truth_item"]).lower()] for r in rs], device=device)
        c = counts.to(device)[tgt]
        split[name] = {"ctx": ctx, "target": tgt,
                       "subsets": {"all": torch.ones_like(c, dtype=torch.bool),
                                   "never": c == 0, "rare_1_5": (c >= 1) & (c <= 5),
                                   "dedup": torch.tensor(
                                       [not mentioned_in_context(str(r["ground_truth_item"]),
                                                                 r["input"]) for r in rs],
                                       device=device)}}
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    def evaluate(v, b):
        out = {}
        for name, s in split.items():
            scores = s["ctx"] @ v.T / temp + b[None, :]
            t = scores.gather(1, s["target"][:, None])
            rank = ((scores > t).sum(1) + 1 + ((scores == t).sum(1) - 1) / 2).float()
            out[name] = {sub: {"R@1": 100 * float((rank[m] <= 1).float().mean()),
                               "R@10": 100 * float((rank[m] <= 10).float().mean()),
                               "R@50": 100 * float((rank[m] <= 50).float().mean()),
                               "R@100": 100 * float((rank[m] <= 100).float().mean()),
                               f"R@{args.top}": 100 * float((rank[m] <= args.top).float().mean()),
                               "MRR": float((1 / rank[m]).mean()), "n": int(m.sum())}
                         for sub, m in s["subsets"].items()}
        return out

    grid = [ColdStart()]
    for mc, mode, q, prof in itertools.product((1, 3, 6), ("keep", "drop", "map"),
                                               (None, 0.25, 0.5, 0.75), (False, True)):
        cfg = ColdStart(mc, mode, q, prof and bool(profiles))
        if not cfg.is_identity() and cfg not in grid:
            grid.append(cfg)
    results = []
    for cfg in grid:
        v, b = cold_items(parts, counts, cfg)
        results.append({"config": cfg.to_dict(), "metrics": evaluate(v, b)})

    top = f"R@{args.top}"
    base = results[0]["metrics"]["val"]["all"]
    ok = [r for r in results
          if r["metrics"]["val"]["all"]["R@10"] >= base["R@10"] - args.max_r10_drop]
    chosen = max(ok, key=lambda r: (r["metrics"]["val"]["all"][top],
                                    r["metrics"]["val"]["all"]["MRR"]))

    def row(r):
        v, t = r["metrics"]["val"], r["metrics"]["test"]
        c = r["config"]
        return (f"{c['min_count']:>3} {c['id_mode']:<5}{str(c['bias_q']):>5} {str(c['profiles']):<6}"
                f"| val R@10 {v['all']['R@10']:5.2f} {top} {v['all'][top]:5.1f} MRR {v['all']['MRR']:.4f}"
                f" never {top} {v['never'][top]:5.1f} "
                f"| test R@10 {t['all']['R@10']:5.2f} R@100 {t['all']['R@100']:5.1f} {top} {t['all'][top]:5.1f}"
                f" never {top} {t['never'][top]:5.1f} rare R@10 {t['rare_1_5']['R@10']:5.1f}")

    print(f"\nmin id    biasq prof  | (chosen on validation: max {top}, R@10 drop <= {args.max_r10_drop})")
    for r in sorted(results, key=lambda r: -r["metrics"]["val"]["all"][top])[:25]:
        print(("* " if r is chosen else "  ") + row(r))
    print("\nbaseline (trained scoring):\n  " + row(results[0]))
    print("chosen:\n  " + row(chosen), flush=True)
    write_json(args.out, {"args": vars(args), "temperature": temp,
                          "chosen": chosen["config"], "baseline": results[0],
                          "variants": results})
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
