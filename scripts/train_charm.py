#!/usr/bin/env python3
"""
Train the CHARM re-ranker on a finished SFT/retriever checkpoint.

Everything upstream is frozen: dialogue states, title states and retriever
scores are computed once, so CHARM trains in minutes. Each training case is the
retriever's own top-K for that dialogue, used only when the retriever found the
positive -- exactly the situation CHARM faces at test time. (Inserting missed
positives, as a first version did, put the answer at the bottom of the list with
the lowest retriever score in over half the lists; CHARM learned to prefer
exactly that, and test R@10 fell 18.4 -> 10.8.) The loss is a listwise softmax
that ignores the other movies recommended in the same turn.

Discipline:
  * the epoch is chosen on the held-out validation conversations (the same
    hash split run_experiment.py uses, so the checkpoint never trained on them),
    and "no training" -- which equals the retriever alone -- is one of the
    candidates, so CHARM cannot ship worse than the retriever on validation
  * test is evaluated once, on the chosen epoch
  * each head's contribution is measured by removing it at evaluation

    python scripts/train_charm.py --checkpoint .../checkpoints/sft_final
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--catalog-file", default=None)
    parser.add_argument("--data", default=os.path.expanduser("~/Desktop/harpo-work/redial_data"))
    parser.add_argument("--val-fraction", type=float, default=0.05,
                        help="must match the SFT run, so validation stays unseen")
    parser.add_argument("--charm-fraction", type=float, default=0.25,
                        help="must match the SFT run's --charm-fraction: CHARM trains on "
                             "these conversations, which the backbone never saw")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--proj-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no-item-bias", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()

    from run_experiment import (build_eval_catalog, build_model, encode_pooled,
                                encode_test_contexts, expand_mentions, load_split,
                                split_conversations, write_json)
    from harpo.charm import (HEADS, CHARMReranker, candidate_features, listwise_loss,
                             mentioned_items)
    from harpo.ranking import mentioned_in_context, metrics_from_ranks
    from harpo.training import HARPOMTv2Trainer

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    catalog_file = args.catalog_file or os.path.join(args.checkpoint, "..", "..", "catalog.json")
    with open(catalog_file) as f:
        catalog_list = json.load(f)
    index = {t.lower(): i for i, t in enumerate(catalog_list)}

    def in_catalogue(rows):
        return [r for r in rows if str(r["ground_truth_item"]).lower() in index]

    backbone_raw, val_raw, charm_raw = split_conversations(
        load_split(os.path.join(args.data, "sft_data.json"), 0, args.seed),
        args.val_fraction, args.charm_fraction)
    if not charm_raw:
        print("WARNING: --charm-fraction 0: CHARM trains on conversations the backbone "
              "saw; expect it to learn memorised patterns that do not transfer")
        charm_raw = backbone_raw
    # Popularity comes from the backbone's training conversations only, never
    # from the rows CHARM learns on: counting a row's own label leaks it.
    backbone_targets = [str(m).lower() for r in expand_mentions(backbone_raw)
                        for m in [r["ground_truth_item"]]]
    splits = {
        "train": in_catalogue(expand_mentions(charm_raw)),
        "val": in_catalogue(expand_mentions(val_raw)),
        "test": in_catalogue(expand_mentions(
            load_split(os.path.join(args.data, "test_sft.json"), 0, args.seed))),
    }
    print(f"device={device}  checkpoint={args.checkpoint}")
    print("cases: " + "  ".join(f"{k}={len(v)}" for k, v in splits.items()))

    # ---- frozen upstream
    model, training_config = build_model(os.path.join(args.checkpoint, "base_model"),
                                         device, args.seq_len, 16, False,
                                         item_bias=not args.no_item_bias)
    HARPOMTv2Trainer(model, training_config, device=device).load_checkpoint(args.checkpoint)
    model.eval()
    _, catalog = build_eval_catalog(model, catalog_list, device)
    item_emb = catalog.embeddings.to(device).float()
    item_hidden = encode_pooled(model, catalog_list, device, 32,
                                batch_size=256 if device == "cuda" else 64).float()
    counts = torch.zeros(len(catalog_list), device=device)
    for t in backbone_targets:
        if t in index:
            counts[index[t]] += 1
    log_pop = torch.log1p(counts)

    feats = {}
    start = time.time()
    for name, rows in splits.items():
        unique = list(dict.fromkeys(r["input"] for r in rows))
        pos = {c: i for i, c in enumerate(unique)}
        ctx_rows, _ = encode_test_contexts(model, rows, device, args.seq_len)
        ctx_unique = torch.zeros(len(unique), ctx_rows.size(1), device=device)
        row_ctx = torch.tensor([pos[r["input"]] for r in rows], device=device)
        ctx_unique[row_ctx] = ctx_rows.float()
        target = torch.tensor([index[str(r["ground_truth_item"]).lower()] for r in rows],
                              device=device)

        top_idx, top_scores, retr_rank = [], [], []
        for s in range(0, len(rows), 2048):
            sc = catalog.scores(ctx_rows[s:s + 2048]).float()
            t = sc.gather(1, target[s:s + 2048, None])
            retr_rank.append((sc > t).sum(1) + 1 + ((sc == t).sum(1) - 1) / 2)
            v, i = sc.topk(args.top_k, dim=1)
            top_idx.append(i)
            top_scores.append(v)
        top_idx, top_scores = torch.cat(top_idx), torch.cat(top_scores)
        retr_rank = torch.cat(retr_rank).float()

        hit = top_idx == target[:, None]
        target_pos = torch.where(hit.any(1), hit.float().argmax(1), torch.full_like(target, -1))
        # Other movies recommended in the same turn are positives too.
        turn_targets = {}
        for r in rows:
            turn_targets.setdefault(r["input"], set()).add(index[str(r["ground_truth_item"]).lower()])
        others = torch.zeros_like(top_idx, dtype=torch.bool)
        for n, r in enumerate(rows):
            extra = turn_targets[r["input"]] - {int(target[n])}
            if extra:
                others[n] = torch.isin(top_idx[n], torch.tensor(sorted(extra), device=device))

        mentioned = [mentioned_items(r["input"], index) for r in rows]
        mention, pref = candidate_features(top_idx, mentioned, item_emb)
        dedup = torch.tensor([not mentioned_in_context(str(r["ground_truth_item"]), r["input"])
                              for r in rows], device=device)
        feats[name] = dict(ctx=ctx_unique, row_ctx=row_ctx, top_idx=top_idx,
                           retriever=top_scores, target_pos=target_pos, others=others,
                           mention=mention, pref=pref, retr_rank=retr_rank, dedup=dedup)
        print(f"  {name}: {len(rows)} cases, {len(unique)} dialogues, target in top-{args.top_k}: "
              f"{100 * float(hit.any(1).float().mean()):.1f}%")
    print(f"features in {(time.time() - start) / 60:.1f} min")

    charm = CHARMReranker(item_hidden.size(1), item_emb.size(1), args.proj_dim,
                          args.dropout).to(device)
    opt = torch.optim.AdamW(charm.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def forward(f, rows_idx):
        ti = f["top_idx"][rows_idx]
        return charm(f["ctx"][f["row_ctx"][rows_idx]], item_hidden[ti], item_emb[ti],
                     f["retriever"][rows_idx], f["mention"][rows_idx], f["pref"][rows_idx],
                     log_pop[ti])

    @torch.no_grad()
    def evaluate(f, drop=None, use_charm=True):
        charm.eval()
        ranks = []
        n = f["top_idx"].size(0)
        for s in range(0, n, 1024):
            idx = torch.arange(s, min(s + 1024, n), device=device)
            if use_charm:
                out = forward(f, idx)
                score = out["score"]
                if drop:
                    w = out["weights"][:, HEADS.index(drop)]
                    score = score - w[:, None] * out[drop]
            else:
                score = f["retriever"][idx]
            tp = f["target_pos"][idx]
            t = score.gather(1, tp.clamp(min=0)[:, None])
            r = (score > t).sum(1) + 1 + ((score == t).sum(1) - 1) / 2
            ranks.append(torch.where(tp >= 0, r.float(), f["retr_rank"][idx]))
        ranks = torch.cat(ranks)
        pool = len(catalog_list)
        return {"standard": metrics_from_ranks(ranks.tolist(), pool, (1, 10, 50)),
                "dedup": metrics_from_ranks(ranks[f["dedup"]].tolist(), pool, (1, 10, 50))}

    def line(tag, m):
        s_, d_ = m["standard"], m["dedup"]
        return (f"{tag:<30} std R@1 {100 * s_['recall_at_1']:5.2f} R@10 {100 * s_['recall_at_10']:5.2f} "
                f"MRR {s_['mrr']:.4f} | dedup R@1 {100 * d_['recall_at_1']:5.2f} "
                f"R@10 {100 * d_['recall_at_10']:5.2f} MRR {d_['mrr']:.4f}")

    base_val = evaluate(feats["val"], use_charm=False)
    print("\n" + line("VAL retriever only", base_val))

    # Train only on lists that contain their positive (see the module docstring).
    tr = feats["train"]
    usable = torch.nonzero(tr["target_pos"] >= 0).flatten()
    n_train = usable.numel()
    print(f"training lists with the positive in the top-{args.top_k}: {n_train} "
          f"of {tr['top_idx'].size(0)}")
    # Epoch 0 is the untrained CHARM, i.e. the retriever alone.
    best, best_epoch, bad = base_val["standard"]["mrr"], 0, 0
    best_state = {k: v.detach().clone() for k, v in charm.state_dict().items()}
    history = []
    for epoch in range(args.epochs):
        charm.train()
        perm = usable[torch.randperm(n_train, device=device)]
        total = 0.0
        for s in range(0, n_train, args.batch):
            idx = perm[s:s + args.batch]
            loss = listwise_loss(forward(tr, idx)["score"], tr["target_pos"][idx], tr["others"][idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * idx.numel()
        val = evaluate(feats["val"])
        mrr = val["standard"]["mrr"]
        history.append({"epoch": epoch + 1, "train_loss": total / n_train, "val": val})
        print(line(f"epoch {epoch + 1:>2} loss {total / n_train:.3f}  VAL", val))
        if mrr > best:
            best, best_epoch, bad = mrr, epoch + 1, 0
            best_state = {k: v.detach().clone() for k, v in charm.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    charm.load_state_dict(best_state)
    print(f"\nepoch selected on validation MRR: {best_epoch}"
          + ("  (no epoch beat the retriever alone: CHARM is the identity)"
             if best_epoch == 0 else ""))

    # ---- test, once, on the selected model
    test_base = evaluate(feats["test"], use_charm=False)
    test_charm = evaluate(feats["test"])
    ablations = {h: evaluate(feats["test"], drop=h) for h in HEADS}
    print(f"\n{'=' * 78}\nTEST (full catalogue of {len(catalog_list)}; re-ranking the top-{args.top_k})\n{'=' * 78}")
    print(line("retriever only", test_base))
    print(line("retriever + CHARM", test_charm))
    for h, m in ablations.items():
        print(line(f"  CHARM without {h}", m))

    with torch.no_grad():
        charm.eval()
        f = feats["test"]
        idx = torch.arange(f["top_idx"].size(0), device=device)
        out = forward(f, idx)
        mean_w = out["weights"].mean(0).tolist()
        nov_w = (charm.novelty(charm.ctx_proj(f["ctx"][f["row_ctx"]])).squeeze(-1)
                 * out["weights"][:, HEADS.index("novelty")])
    print("\nmean head weights on test: " + "  ".join(f"{h} {w:.2f}" for h, w in zip(HEADS, mean_w)))
    print(f"novelty effect per dialogue (weighted): mean {float(nov_w.mean()):+.3f}, "
          f"share favouring repeats {100 * float((nov_w > 0).float().mean()):.1f}%")

    results_dir = args.results_dir or os.path.join(args.checkpoint, "..", "..")
    os.makedirs(results_dir, exist_ok=True)
    torch.save({"state_dict": charm.state_dict(), "args": vars(args),
                "hidden_size": item_hidden.size(1), "item_dim": item_emb.size(1)},
               os.path.join(results_dir, "charm.pt"))
    write_json(os.path.join(results_dir, "charm_results.json"),
               {"args": vars(args), "device": device, "selected_epoch": best_epoch,
                "val_retriever_only": base_val, "history": history,
                "test_retriever_only": test_base, "test_charm": test_charm,
                "test_head_ablations": ablations,
                "mean_head_weights": dict(zip(HEADS, mean_w))})


if __name__ == "__main__":
    main()
