#!/usr/bin/env python3
"""
Train and evaluate MAVEN consensus over agent scores on a shared shortlist.

Inputs are the rerank_eval.py raw dump (retriever + LM likelihood + mentioned
flag per candidate) and, optionally, CHARM cross-encoder scores for the same
rows (score_shortlist_ce.py). Two ways to fit, never on the rows being scored:

  --fit-on val       fit on the validation dump, evaluate test once
                     (requires a backbone that never trained on validation)
  --fit-on crossfit  2-fold cross-fitting over test conversations: each half is
                     scored by a model fitted on the other half (for backbones
                     trained on every training conversation)

Reports every agent alone, the static consensus (one global weighting) and
MAVEN (per-dialogue weighting), on the standard and deduplicated test sets.
Variants that add the already-mentioned flag are reported separately: it helps
the standard protocol only through the repetition shortcut.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import torch


def load(raw_path, ce_path, extras=()):
    raw = torch.load(raw_path)
    k = raw["top_idx"].size(1)
    agents = {"retriever": raw["retriever"].float(), "lm": raw["lm"].float()}
    for name, path in ([("charm", ce_path)] if ce_path else []) + list(extras):
        d = torch.load(path)
        if not torch.equal(d["targets"], raw["targets"].cpu()):
            raise ValueError(f"{path} does not line up with {raw_path}")
        scores = d["scores"].float()
        if scores.size(1) < k:  # agent saw fewer candidates: neutral elsewhere
            fill = scores.mean(1, keepdim=True).expand(-1, k - scores.size(1))
            scores = torch.cat([scores, fill], 1)
        agents[name] = scores[:, :k]
    hit = raw["top_idx"] == raw["targets"][:, None]
    target_pos = torch.where(hit.any(1), hit.float().argmax(1),
                             torch.full_like(raw["targets"], -1))
    dedup = torch.zeros(raw["top_idx"].size(0), dtype=torch.bool)
    dedup[raw["dedup_rows"]] = True
    return {"agents": agents, "mention": raw["mention"].float(), "target_pos": target_pos,
            "retr_rank": raw["retriever_rank"].float(), "folds": raw["folds"],
            "dedup": dedup, "pool": len(raw["catalog"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-raw", required=True)
    parser.add_argument("--test-ce", default=None)
    parser.add_argument("--val-raw", default=None)
    parser.add_argument("--val-ce", default=None)
    parser.add_argument("--agent", action="append", default=[],
                        help="extra agent: name=test_scores.pt[,val_scores.pt] (e.g. STAR)")
    parser.add_argument("--fit-on", choices=["val", "crossfit"], required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from harpo.maven import MAVENConsensus, agreement_features, fit, ranks_from_scores, zrow
    from harpo.ranking import metrics_from_ranks

    torch.manual_seed(0)
    extras = []
    for spec in args.agent:
        name, paths = spec.split("=", 1)
        parts = paths.split(",")
        extras.append((name, parts[0], parts[1] if len(parts) > 1 else None))
    test = load(args.test_raw, args.test_ce, [(n, t) for n, t, _ in extras])
    val = (load(args.val_raw, args.val_ce, [(n, v) for n, _, v in extras])
           if args.fit_on == "val" else None)
    names = list(test["agents"])

    def tensors(d, with_mention):
        cols = [zrow(d["agents"][n]) for n in names]
        if with_mention:
            cols.append(d["mention"] - d["mention"].mean(1, keepdim=True))
        z = torch.stack(cols, -1)
        return z, agreement_features(z)

    def metrics(ranks, d):
        return {"standard": metrics_from_ranks(ranks.tolist(), d["pool"], (1, 10, 50)),
                "dedup": metrics_from_ranks(ranks[d["dedup"]].tolist(), d["pool"], (1, 10, 50))}

    def run(gated, with_mention):
        z_t, f_t = tensors(test, with_mention)
        a = z_t.size(-1)
        if args.fit_on == "val":
            z_v, f_v = tensors(val, with_mention)
            model = MAVENConsensus(a, f_v.size(1), gated=gated)
            fit(model, z_v, f_v, val["target_pos"], epochs=args.epochs)
            with torch.no_grad():
                fused, w = model(z_t, f_t)
            weights = w.mean(0).tolist()
        else:
            fused = torch.zeros(z_t.shape[:2])
            weights = []
            for fold in test["folds"].unique().tolist():
                train, held = test["folds"] != fold, test["folds"] == fold
                model = MAVENConsensus(a, f_t.size(1), gated=gated)
                fit(model, z_t[train], f_t[train], test["target_pos"][train], epochs=args.epochs)
                with torch.no_grad():
                    fused_h, w = model(z_t[held], f_t[held])
                fused[held] = fused_h
                weights.append(w.mean(0).tolist())
        ranks = ranks_from_scores(fused, test["target_pos"], test["retr_rank"])
        return metrics(ranks, test), weights

    report = {}
    for n in names:  # each agent alone, inside the shortlist
        ranks = ranks_from_scores(test["agents"][n], test["target_pos"], test["retr_rank"])
        report[f"agent: {n}"] = {"metrics": metrics(ranks, test)}
    for label, gated, mention in [("static consensus", False, False),
                                  ("MAVEN (per-dialogue)", True, False),
                                  ("static + mentioned flag", False, True),
                                  ("MAVEN + mentioned flag", True, True)]:
        m, w = run(gated, mention)
        report[label] = {"metrics": m, "mean_weights": w}

    agent_names = names + ["mentioned"]
    print(f"agents: {names}   fitted on: {args.fit_on}")
    print(f"{'':<28}{'std R@1':>8}{'R@10':>7}{'R@50':>7}{'MRR':>8}{'ddp R@1':>9}{'R@10':>7}{'MRR':>8}")
    for label, r in report.items():
        s_, d_ = r["metrics"]["standard"], r["metrics"]["dedup"]
        print(f"{label:<28}{100 * s_['recall_at_1']:>7.2f}%{100 * s_['recall_at_10']:>6.2f}%"
              f"{100 * s_['recall_at_50']:>6.2f}%{s_['mrr']:>8.4f}{100 * d_['recall_at_1']:>8.2f}%"
              f"{100 * d_['recall_at_10']:>6.2f}%{d_['mrr']:>8.4f}")
    for label, r in report.items():
        if "mean_weights" in r:
            print(f"  {label} mean weights: {r['mean_weights']}  ({agent_names})")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "agents": names, "report": report}, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
