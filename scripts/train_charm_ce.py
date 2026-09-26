#!/usr/bin/env python3
"""
Train the cross-encoder CHARM re-ranker (harpo/charm_ce.py).

  1. Shortlists: the SFT backbone's retriever ranks the full catalogue for every
     training, validation and test case; the top-K are cached to disk.
  2. Training: every training conversation except validation. Each step scores
     groups of [positive + (G-1) hard negatives sampled from the shortlist]
     with a listwise softmax; the other movies recommended in the same turn are
     never used as negatives.
  3. Selection: every --eval-every steps the validation shortlists are re-scored
     and fused with the retriever at each weight in a grid; the best
     (checkpoint, weight) by validation MRR is kept. The retriever alone
     (weight 0) is always a candidate.
  4. Test: evaluated once, with the selected checkpoint and weight.

    python scripts/train_charm_ce.py --backbone .../half_charm/checkpoints/sft_final \\
        --base-model /tmp/harpo_rebuild/models/Qwen2.5-0.5B-Instruct
"""

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F

WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def compute_shortlists(model, catalog, rows, index, device, top_k, seq_len):
    """Retriever top-K for each row, plus what evaluation needs."""
    from run_experiment import encode_test_contexts
    from harpo.ranking import mentioned_in_context

    ctx, _ = encode_test_contexts(model, rows, device, seq_len)
    target = torch.tensor([index[str(r["ground_truth_item"]).lower()] for r in rows],
                          device=device)
    top_idx, top_scores, rank = [], [], []
    for s in range(0, len(rows), 2048):
        sc = catalog.scores(ctx[s:s + 2048]).float()
        t = sc.gather(1, target[s:s + 2048, None])
        rank.append((sc > t).sum(1) + 1 + ((sc == t).sum(1) - 1) / 2)
        v, i = sc.topk(top_k, dim=1)
        top_idx.append(i)
        top_scores.append(v)
    top_idx = torch.cat(top_idx)
    hit = top_idx == target[:, None]
    return {
        "top_idx": top_idx.cpu(), "retriever": torch.cat(top_scores).cpu(),
        "retr_rank": torch.cat(rank).float().cpu(), "target": target.cpu(),
        "target_pos": torch.where(hit.any(1), hit.float().argmax(1),
                                  torch.full_like(target, -1)).cpu(),
        "dedup": torch.tensor([not mentioned_in_context(str(r["ground_truth_item"]), r["input"])
                               for r in rows]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", required=True, help="SFT checkpoint providing shortlists")
    parser.add_argument("--base-model", required=True, help="plain model the cross-encoder starts from")
    parser.add_argument("--catalog-file", default=None)
    parser.add_argument("--data", default=os.path.expanduser("~/Desktop/harpo-work/redial_data"))
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--charm-fraction", type=float, default=0.25,
                        help="the backbone's split; the cross-encoder trains on all "
                             "non-validation conversations either way")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--group", type=int, default=16)
    parser.add_argument("--batch-groups", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--max-train-cases", type=int, default=0,
                        help="cap on training cases (budget for large models)")
    parser.add_argument("--eval-top-k", type=int, default=0,
                        help="re-score only the first K of each shortlist at evaluation "
                             "(default: --top-k); the rest keep the retriever order")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32",
                        help="backbone weights; bfloat16 for 7B")
    parser.add_argument("--shortlists", default=None,
                        help="reuse a shortlist cache from another run (same data and splits)")
    parser.add_argument("--score-dialogues", type=int, default=32,
                        help="dialogues per scoring batch at evaluation")
    parser.add_argument("--val-limit", type=int, default=0, help="smoke tests only")
    parser.add_argument("--init-adapter", default=None,
                        help="continue training from a saved charm_ce_adapter")
    parser.add_argument("--novel-only", action="store_true",
                        help="train only on cases whose answer is not already named in the "
                             "dialogue, so the model cannot learn to copy titles")
    parser.add_argument("--no-test", action="store_true",
                        help="skip the test evaluation (the test shortlist is scored "
                             "separately, e.g. by scripts/score_shortlist_ce.py)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()

    from run_experiment import (build_eval_catalog, build_model, expand_mentions,
                                load_split, split_conversations, write_json)
    from harpo.charm_ce import CrossEncoderCHARM, fused_ranks, sample_group
    from harpo.ranking import metrics_from_ranks
    from harpo.training import HARPOMTv2Trainer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(args.results_dir, exist_ok=True)

    catalog_file = args.catalog_file or os.path.join(args.backbone, "..", "..", "catalog.json")
    with open(catalog_file) as f:
        catalog_list = json.load(f)
    index = {t.lower(): i for i, t in enumerate(catalog_list)}

    def usable(rows):
        return [r for r in rows if str(r["ground_truth_item"]).lower() in index]

    backbone_raw, val_raw, charm_raw = split_conversations(
        load_split(os.path.join(args.data, "sft_data.json"), 0, args.seed),
        args.val_fraction, args.charm_fraction)
    rows = {
        "train": usable(expand_mentions(backbone_raw + charm_raw)),
        "val": usable(expand_mentions(val_raw)),
        "test": usable(expand_mentions(
            load_split(os.path.join(args.data, "test_sft.json"), 0, args.seed))),
    }
    if args.val_limit:
        keep = set(list(dict.fromkeys(r["input"] for r in rows["val"]))[:args.val_limit])
        rows["val"] = [r for r in rows["val"] if r["input"] in keep]
    print("cases: " + "  ".join(f"{k}={len(v)}" for k, v in rows.items()), flush=True)

    # ---- 1. shortlists (cached)
    cache = args.shortlists or os.path.join(args.results_dir, f"shortlists_top{args.top_k}.pt")
    lists = torch.load(cache) if os.path.exists(cache) else None
    if lists is None or any(lists[k]["top_idx"].size(0) != len(rows[k]) for k in rows):
        model, training_config = build_model(os.path.join(args.backbone, "base_model"),
                                             device, args.seq_len, 16, False)
        HARPOMTv2Trainer(model, training_config, device=device).load_checkpoint(args.backbone)
        model.eval()
        _, catalog = build_eval_catalog(model, catalog_list, device)
        lists = {k: compute_shortlists(model, catalog, v, index, device, args.top_k, args.seq_len)
                 for k, v in rows.items()}
        torch.save(lists, cache)
        del model, catalog
        if device == "cuda":
            torch.cuda.empty_cache()
    for k, v in lists.items():
        print(f"  {k}: target in retriever top-{args.top_k}: "
              f"{100 * float((v['target_pos'] >= 0).float().mean()):.1f}%", flush=True)

    if args.novel_only:
        from harpo.ranking import mentioned_in_context
        keep = [i for i, r in enumerate(rows["train"])
                if not mentioned_in_context(str(r["ground_truth_item"]), r["input"])]
        print(f"novel-only: training on {len(keep)} of {len(rows['train'])} cases "
              f"(dropped answers already named in the dialogue)", flush=True)
        rows["train"] = [rows["train"][i] for i in keep]
        idx = torch.tensor(keep)
        lists["train"] = {k: v[idx] for k, v in lists["train"].items()}

    # Evaluation re-scores only the first eval_k candidates (large models are
    # expensive per pair); outside them the retriever's ranking stands.
    eval_k = args.eval_top_k or args.top_k
    eval_lists = {}
    for k, v in lists.items():
        tp = v["target_pos"]
        eval_lists[k] = {**v, "top_idx": v["top_idx"][:, :eval_k],
                         "retriever": v["retriever"][:, :eval_k],
                         "target_pos": torch.where(tp < eval_k, tp, torch.full_like(tp, -1))}

    # ---- 2. cross-encoder
    ce = CrossEncoderCHARM(args.base_model, device, lora_r=args.lora_r,
                           lora_alpha=2 * args.lora_r, max_length=args.max_length,
                           dtype=getattr(torch, args.dtype))
    if args.init_adapter:
        ce.load_adapter(args.init_adapter)
        print(f"continuing from {args.init_adapter}", flush=True)
    opt = torch.optim.AdamW([{"params": ce.adapter_parameters(), "lr": args.lr},
                             {"params": ce.head_parameters(), "lr": args.head_lr}],
                            weight_decay=0.01)
    n_train = len(rows["train"]) if not args.max_train_cases else min(args.max_train_cases,
                                                                        len(rows["train"]))
    steps_per_epoch = (n_train + args.batch_groups - 1) // args.batch_groups
    total = steps_per_epoch * args.epochs
    warmup = min(100, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(warmup, 1)) * max(0.0, 1 - s / max(total, 1)))

    turn_pos = {}
    for r in rows["train"]:
        turn_pos.setdefault(r["input"], set()).add(index[str(r["ground_truth_item"]).lower()])

    def evaluate(split, retriever_only=False):
        feats, rs = eval_lists[split], rows[split]
        pool = len(catalog_list)
        if retriever_only:  # weight 0 needs no cross-encoder scores at all
            ranks = fused_ranks(feats["retriever"], feats["retriever"], 0.0,
                                feats["target_pos"], feats["retr_rank"])
            return {0.0: {"standard": metrics_from_ranks(ranks.tolist(), pool, (1, 10, 50)),
                          "dedup": metrics_from_ranks(ranks[feats["dedup"]].tolist(), pool,
                                                      (1, 10, 50))}}
        first = {}
        for i, r in enumerate(rs):
            first.setdefault(r["input"], i)
        dialogues = list(first)
        groups = [[catalog_list[c] for c in feats["top_idx"][first[d]].tolist()] for d in dialogues]
        per_dialogue = ce.score_groups(dialogues, groups, args.score_dialogues).cpu()
        slot = {d: n for n, d in enumerate(dialogues)}
        ce_scores = per_dialogue[[slot[r["input"]] for r in rs]]
        out = {}
        for w in WEIGHTS:
            ranks = fused_ranks(feats["retriever"], ce_scores, w, feats["target_pos"],
                                feats["retr_rank"])
            out[w] = {"standard": metrics_from_ranks(ranks.tolist(), pool, (1, 10, 50)),
                      "dedup": metrics_from_ranks(ranks[feats["dedup"]].tolist(), pool, (1, 10, 50))}
        return out

    def line(tag, m):
        s_, d_ = m["standard"], m["dedup"]
        return (f"{tag:<34} std R@1 {100 * s_['recall_at_1']:5.2f} R@10 {100 * s_['recall_at_10']:5.2f} "
                f"MRR {s_['mrr']:.4f} | dedup R@1 {100 * d_['recall_at_1']:5.2f} "
                f"R@10 {100 * d_['recall_at_10']:5.2f} MRR {d_['mrr']:.4f}")

    # Retriever alone is the baseline every checkpoint must beat on validation.
    base_val = evaluate("val", retriever_only=True)[0.0]
    print(line("VAL retriever only", base_val), flush=True)
    best = {"mrr": base_val["standard"]["mrr"], "step": 0, "weight": 0.0, "state": None}
    history = []
    if args.init_adapter:
        # The starting adapter is itself a candidate: continuing must beat it.
        val0 = evaluate("val")
        w0 = max(WEIGHTS, key=lambda w: val0[w]["standard"]["mrr"])
        print(line(f"start VAL (w={w0:.1f})", val0[w0]), flush=True)
        if w0 > 0 and val0[w0]["standard"]["mrr"] > best["mrr"]:
            best.update(mrr=val0[w0]["standard"]["mrr"], step=0, weight=w0,
                        state=ce.trainable_state())

    def select(step):
        val = evaluate("val")
        w_best = max(WEIGHTS, key=lambda w: val[w]["standard"]["mrr"])
        m = val[w_best]
        history.append({"step": step, "best_weight": w_best, "val": {str(w): v for w, v in val.items()}})
        print(line(f"step {step} VAL (w={w_best:.1f})", m)
              + f"   CE only MRR {val[1.0]['standard']['mrr']:.4f}", flush=True)
        if w_best > 0 and m["standard"]["mrr"] > best["mrr"]:
            best.update(mrr=m["standard"]["mrr"], step=step, weight=w_best,
                        state=ce.trainable_state())

    # ---- 3. training
    rng = random.Random(args.seed)
    step, start = 0, time.time()
    run_loss, run_acc, run_n = 0.0, 0.0, 0
    ce.model.train()
    for epoch in range(args.epochs):
        order = list(range(len(rows["train"])))
        rng.shuffle(order)
        order = order[:n_train]
        for s in range(0, len(order), args.batch_groups):
            batch = order[s:s + args.batch_groups]
            dialogues, groups = [], []
            for i in batch:
                r = rows["train"][i]
                tgt = int(lists["train"]["target"][i])
                group = sample_group(tgt, lists["train"]["top_idx"][i].tolist(),
                                     turn_pos[r["input"]] - {tgt}, args.group,
                                     len(catalog_list), rng)
                dialogues.append(r["input"])
                groups.append([catalog_list[c] for c in group])
            # One encoding of each dialogue, shared by its whole group.
            scores = ce.grouped_scores(dialogues, groups)
            loss = F.cross_entropy(scores, torch.zeros(len(batch), dtype=torch.long,
                                                       device=scores.device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ce.adapter_parameters() + ce.head_parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            run_loss += float(loss) * len(batch)
            run_acc += float((scores.argmax(1) == 0).float().sum())
            run_n += len(batch)
            if step % 100 == 0:
                rate = (time.time() - start) / step
                print(f"  step {step}/{total}  loss {run_loss / run_n:.3f}  "
                      f"group top-1 {100 * run_acc / run_n:.1f}%  "
                      f"({rate:.2f}s/step, ~{rate * (total - step) / 60:.0f} min left)", flush=True)
                run_loss, run_acc, run_n = 0.0, 0.0, 0
            if step % args.eval_every == 0:
                select(step)
    if step % args.eval_every:
        select(step)

    # ---- 4. test, once
    print(f"\nselected: step {best['step']}, weight {best['weight']:.1f} "
          f"(validation MRR {best['mrr']:.4f} vs retriever {base_val['standard']['mrr']:.4f})")
    if best["state"] is None:
        print("no checkpoint beat the retriever on validation: CHARM-CE is disabled")
        test = {0.0: None}
        chosen = None
    else:
        ce.load_trainable_state(best["state"])
        ce.model.save_pretrained(os.path.join(args.results_dir, "charm_ce_adapter"))
        if args.no_test:
            test, chosen = {0.0: None}, None
        else:
            test = evaluate("test")
            chosen = test[best["weight"]]
    print(f"\n{'=' * 78}\nTEST (full catalogue of {len(catalog_list)}; re-ranking the top-{eval_k})\n{'=' * 78}")
    retr_test = test[0.0] if chosen is not None else None
    if chosen is not None:
        print(line("retriever only", retr_test))
        print(line(f"retriever + CHARM-CE (w={best['weight']:.1f})", chosen))
        print(line("CHARM-CE only (within top-K)", test[1.0]))
    write_json(os.path.join(args.results_dir, "charm_ce_results.json"),
               {"args": vars(args), "device": device, "selected_step": best["step"],
                "selected_weight": best["weight"], "val_retriever_only": base_val,
                "history": history,
                "test": ({str(w): m for w, m in test.items()} if chosen is not None else None)})


if __name__ == "__main__":
    main()
