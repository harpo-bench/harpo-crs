#!/usr/bin/env python3
"""
LM-likelihood re-ranking of retriever shortlists, evaluated on a saved checkpoint.

For each test dialogue the retriever's top-K candidates are re-scored by the
fine-tuned LM's log-probability of writing each title (see harpo/lm_rerank.py)
and fused with the retriever score:

    score = (1 - alpha) * z(retriever) + alpha * z(lm)        within the top-K

alpha = 0 reproduces the retriever exactly (a built-in consistency check against
the training run's own evaluation); alpha = 1 is the LM alone.

Choosing alpha on the test set would be tuning on test. The headline therefore
uses 2-fold cross-fitting over conversations: each half is scored with the alpha
that maximises MRR on the *other* half, so no case is scored with an alpha
chosen by looking at it. The full alpha sweep is printed as well, labelled as a
test-set sweep, for transparency only.

    python scripts/rerank_eval.py --checkpoint /tmp/harpo_rebuild/outputs/7b_6ep/checkpoints/sft_final
"""

import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help=".../checkpoints/sft_final")
    parser.add_argument("--catalog-file", default=None,
                        help="default: catalog.json two levels above the checkpoint")
    parser.add_argument("--data", default=os.path.expanduser("~/Desktop/harpo-work/redial_data"))
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--alphas", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1")
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--no-item-bias", action="store_true",
                        help="set when the checkpoint was trained without the item bias")
    parser.add_argument("--limit", type=int, default=0,
                        help="score only the first N dialogues (smoke test)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=["test", "val"], default="test",
                        help="val: the held-out validation conversations (same hash split "
                             "as run_experiment.py), for training fusions without test data")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--charm-fraction", type=float, default=0.25)
    parser.add_argument("--coldstart", default=None,
                        help="coldstart_probe.py output: rebuild rarely recommended items' "
                             "vectors with its chosen config (harpo/coldstart.py)")
    parser.add_argument("--profiles", default=None, help="BRIDGE profiles, for --coldstart")
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()

    from run_experiment import (build_eval_catalog, build_model, encode_test_contexts,
                                expand_mentions, load_split, split_conversations, write_json)
    from harpo.lm_rerank import LikelihoodReranker
    from harpo.ranking import metrics_from_ranks, mentioned_in_context
    from harpo.training import HARPOMTv2Trainer

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    alphas = [float(a) for a in args.alphas.split(",")]

    catalog_file = args.catalog_file or os.path.join(args.checkpoint, "..", "..", "catalog.json")
    with open(catalog_file) as f:
        catalog_list = json.load(f)
    index = {t.lower(): i for i, t in enumerate(catalog_list)}

    if args.split == "test":
        test_rows = expand_mentions(load_split(os.path.join(args.data, "test_sft.json"),
                                               0, args.seed))
    else:
        _, val_raw, _ = split_conversations(
            load_split(os.path.join(args.data, "sft_data.json"), 0, args.seed),
            args.val_fraction, args.charm_fraction)
        test_rows = expand_mentions(val_raw)
    if args.limit:
        keep = set(list(dict.fromkeys(r["input"] for r in test_rows))[:args.limit])
        test_rows = [r for r in test_rows if r["input"] in keep]
    test_rows = [r for r in test_rows if str(r["ground_truth_item"]).lower() in index]
    print(f"device={device}  checkpoint={args.checkpoint}")
    print(f"catalogue={len(catalog_list)} from {catalog_file}   test cases={len(test_rows)}")

    model, training_config = build_model(os.path.join(args.checkpoint, "base_model"),
                                         device, args.seq_len, 16, False,
                                         item_bias=not args.no_item_bias)
    trainer = HARPOMTv2Trainer(model, training_config, device=device)
    trainer.load_checkpoint(args.checkpoint)
    model.eval()

    # ---- retriever: exactly the scoring the training run evaluated
    items, catalog = build_eval_catalog(model, catalog_list, device)
    coldstart = None
    if args.coldstart:
        from run_experiment import coldstart_parts
        from harpo.coldstart import ColdStart, cold_items, training_counts

        with open(args.coldstart) as f:
            coldstart = ColdStart(**json.load(f)["chosen"])
        profiles = {}
        if coldstart.profiles:
            with open(args.profiles) as f:
                profiles = json.load(f)
        # Counts from the backbone's own training rows (same split fractions).
        train_raw, _, _ = split_conversations(
            load_split(os.path.join(args.data, "sft_data.json"), 0, args.seed),
            args.val_fraction, args.charm_fraction)
        counts = training_counts(expand_mentions(train_raw), index, len(catalog_list))
        vecs, bias = cold_items(coldstart_parts(model, catalog_list, device, profiles),
                                counts, coldstart)
        catalog.override(vecs, bias)
        print(f"cold-start items: {coldstart} ({int((counts < coldstart.min_count).sum())} items)")
    ctx_all, n_dialogues = encode_test_contexts(model, test_rows, device, args.seq_len)
    scores = catalog.scores(ctx_all).float()                       # [N, C]
    targets = torch.tensor([index[str(r["ground_truth_item"]).lower()] for r in test_rows],
                           device=scores.device)
    tgt = scores.gather(1, targets[:, None])
    retr_rank = ((scores > tgt).sum(1) + 1 + ((scores == tgt).sum(1) - 1) / 2).tolist()
    top_scores, top_idx = scores.topk(args.top_k, dim=1)

    # ---- LM likelihood of each shortlisted title, once per distinct dialogue
    reranker = LikelihoodReranker(model.base_model, model.tokenizer, device)
    first_row = {}
    for i, r in enumerate(test_rows):
        first_row.setdefault(r["input"], i)
    lm_scores = {}
    start = time.time()
    for n, (context, i) in enumerate(first_row.items()):
        lm_scores[context] = reranker.score(
            context, [catalog_list[j] for j in top_idx[i].tolist()]).float()
        if (n + 1) % 500 == 0:
            rate = (time.time() - start) / (n + 1)
            print(f"  LM-scored {n + 1}/{len(first_row)} dialogues "
                  f"({rate:.2f}s each, ~{rate * (len(first_row) - n - 1) / 60:.0f} min left)",
                  flush=True)
    print(f"LM scoring: {len(first_row)} dialogues x {args.top_k} candidates "
          f"in {(time.time() - start) / 60:.1f} min")

    # ---- per-candidate features, [N, K], shared by every fusion variant
    from harpo.lm_rerank import NEUTRAL_CONTEXT, score_many
    from harpo.ranking import _normalise

    dev = top_scores.device
    lm = torch.stack([lm_scores[r["input"]].to(dev) for r in test_rows])
    union = torch.unique(top_idx)
    start = time.time()
    prior_vals = score_many(reranker, NEUTRAL_CONTEXT, [catalog_list[j] for j in union.tolist()])
    prior_full = torch.zeros(len(catalog_list), device=dev)
    prior_full[union] = prior_vals.to(dev)
    prior = prior_full[top_idx]
    print(f"title priors for {len(union)} shortlisted items in {time.time() - start:.0f}s")

    norm_titles = [_normalise(t) for t in catalog_list]
    rows_of = {}
    for k, r in enumerate(test_rows):
        rows_of.setdefault(r["input"], []).append(k)
    mention = torch.zeros_like(top_scores)
    for context, i in first_row.items():
        norm_ctx = _normalise(context)
        flags = torch.tensor([1.0 if norm_titles[j] and norm_titles[j] in norm_ctx else 0.0
                              for j in top_idx[i].tolist()], device=dev)
        mention[rows_of[context]] = flags

    def zrow(x):
        return (x - x.mean(1, keepdim=True)) / x.std(1, keepdim=True).clamp(min=1e-6)

    feats = {"retriever": zrow(top_scores), "lm": zrow(lm), "prior": zrow(prior),
             "mention": mention}
    hit = top_idx == targets[:, None]
    in_top = hit.any(1)
    tpos = hit.float().argmax(1)
    retr_rank_t = torch.tensor(retr_rank, device=dev, dtype=torch.float)

    def ranks(w):
        fused = sum(wt * feats[name] for name, wt in w.items() if wt)
        if not torch.is_tensor(fused):
            fused = torch.zeros_like(top_scores)
        t = fused.gather(1, tpos[:, None])
        r = (fused > t).sum(1) + 1 + ((fused == t).sum(1) - 1) / 2
        return torch.where(in_top, r.float(), retr_rank_t).tolist()

    all_idx = list(range(len(test_rows)))
    dedup_idx = [i for i in all_idx if not mentioned_in_context(
        str(test_rows[i]["ground_truth_item"]), test_rows[i]["input"])]
    subsets = {"standard": all_idx, "dedup": dedup_idx}
    pool = len(catalog_list)

    def metrics(rk, idx):
        return metrics_from_ranks([rk[i] for i in idx], pool, (1, 10, 50))

    def fold_of(i):
        conv = str(test_rows[i].get("conversation_id", test_rows[i]["input"]))
        return int(hashlib.md5(conv.encode()).hexdigest(), 16) % args.folds

    folds = [fold_of(i) for i in all_idx]
    betas = [0.0, 0.25, 0.5, 0.75, 1.0]
    mentions = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    grids = {
        "A retriever only": [{"retriever": 1.0}],
        "B LM only (top-K)": [{"lm": 1.0}],
        "C retriever + LM": [{"retriever": 1 - a, "lm": a} for a in alphas],
        "D retriever + calibrated LM": [{"retriever": 1 - a, "lm": a, "prior": -a * b}
                                        for a in alphas for b in betas],
        "E D + already-mentioned flag": [{"retriever": 1 - a, "lm": a, "prior": -a * b,
                                          "mention": m}
                                         for a in alphas for b in betas for m in mentions],
    }

    # Headline per variant: 2-fold cross-fitting, weights chosen on the other
    # fold by standard-set MRR (the published protocol), never on the scored cases.
    cache = {}

    def ranks_cached(w):
        key = tuple(sorted(w.items()))
        if key not in cache:
            cache[key] = ranks(w)
        return cache[key]

    report = {}
    print(f"\n{'=' * 78}\nRE-RANKING top-{args.top_k}: {args.folds}-fold cross-fitted "
          f"(weights chosen on the other fold)\n{'=' * 78}")
    print(f"{'variant':<32}{'std R@1':>8}{'R@10':>7}{'MRR':>8}{'ddp R@1':>9}{'R@10':>7}{'MRR':>8}")
    for name, grid in grids.items():
        final = [None] * len(all_idx)
        chosen = {}
        for f in range(args.folds):
            tune = [i for i in all_idx if folds[i] != f]
            best = max(grid, key=lambda w: metrics(ranks_cached(w), tune)["mrr"])
            chosen[f] = best
            rk = ranks_cached(best)
            for i in all_idx:
                if folds[i] == f:
                    final[i] = rk[i]
        res = {sub: metrics(final, idx) for sub, idx in subsets.items()}
        report[name] = {"metrics": res, "chosen": {str(f): w for f, w in chosen.items()}}
        s_, d_ = res["standard"], res["dedup"]
        print(f"{name:<32}{100 * s_['recall_at_1']:>7.2f}%{100 * s_['recall_at_10']:>6.2f}%"
              f"{s_['mrr']:>8.4f}{100 * d_['recall_at_1']:>8.2f}%{100 * d_['recall_at_10']:>6.2f}%"
              f"{d_['mrr']:>8.4f}")
    print("  (R@50 is unchanged by re-ranking inside the top-50)")
    for name, r in report.items():
        if len(grids[name]) > 1:
            print(f"  {name}: chosen weights per fold {r['chosen']}")

    print(f"\n--- alpha sweep, variant C, on the test set (transparency only)")
    for a in alphas:
        rk = ranks_cached({"retriever": 1 - a, "lm": a})
        s_, d_ = metrics(rk, subsets["standard"]), metrics(rk, subsets["dedup"])
        print(f"  alpha {a:.1f}: std R@1 {100 * s_['recall_at_1']:5.2f} R@10 {100 * s_['recall_at_10']:5.2f} "
              f"MRR {s_['mrr']:.4f} | dedup R@1 {100 * d_['recall_at_1']:5.2f} "
              f"R@10 {100 * d_['recall_at_10']:5.2f} MRR {d_['mrr']:.4f}")

    results_dir = args.results_dir or os.path.join(args.checkpoint, "..", "..")
    tag = f"rerank_top{args.top_k}" + ("" if args.split == "test" else f"_{args.split}")
    write_json(os.path.join(results_dir, f"{tag}.json"),
               {"args": vars(args), "device": device, "n_dialogues": n_dialogues,
                "coldstart": coldstart.to_dict() if coldstart else None,
                "variants": report})
    # Raw features so fusions can be re-analysed offline without a GPU.
    raw_path = os.path.join(results_dir, f"{tag}_raw.pt")
    torch.save({"top_idx": top_idx.cpu(), "retriever": top_scores.cpu(), "lm": lm.cpu(),
                "prior": prior.cpu(), "mention": mention.cpu(), "targets": targets.cpu(),
                "retriever_rank": retr_rank_t.cpu(), "folds": torch.tensor(folds),
                "dedup_rows": torch.tensor(dedup_idx),
                "conversation_ids": [str(r.get("conversation_id", "")) for r in test_rows],
                "catalog": catalog_list}, raw_path)
    print(f"wrote {raw_path}")


if __name__ == "__main__":
    main()
