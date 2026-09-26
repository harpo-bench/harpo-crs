#!/usr/bin/env python3
"""
Train STAR's listwise selector: dialogue + top-M candidates (with BRIDGE
profiles) -> the letter of the movie the recommender actually suggested.

Zero-shot, the instruct model reasons fluently but does not know what ReDial
users pick (top-1 among the top-20: 3.5% vs the retriever's 4.3%). Trained on the
retriever's own hard negatives it learns exactly that choice. Candidate order is
shuffled per example, so no position carries information; other movies
recommended in the same turn are removed from the list rather than counted
wrong. The loss is a softmax over the M label tokens at the answer position.

Rows and shortlists are the CHARM cross-encoder's cache (same data, splits and
catalogue), checked for alignment. The adapter with the best validation MRR
within the list is kept; step 0 (zero-shot) is a candidate.

    python scripts/train_star.py --base-model .../Qwen2.5-7B-Instruct \\
        --shortlists .../charm_ce/shortlists_top50.pt --catalog .../half_charm/catalog.json \\
        --profiles .../bridge/profiles.json --out-dir .../star_trained
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--shortlists", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--profiles", default=None)
    parser.add_argument("--data", default=os.path.expanduser("~/Desktop/harpo-work/redial_data"))
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--charm-fraction", type=float, default=0.25)
    parser.add_argument("--top-m", type=int, default=20)
    parser.add_argument("--max-train", type=int, default=12000)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--val-limit", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from run_experiment import expand_mentions, load_split, split_conversations
    from harpo.star import LABELS, direct_prompt

    assert args.top_m <= len(LABELS)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.catalog) as f:
        catalog = json.load(f)
    index = {t.lower(): i for i, t in enumerate(catalog)}
    profiles = {}
    if args.profiles and os.path.exists(args.profiles):
        with open(args.profiles) as f:
            profiles = json.load(f)

    def usable(rows):
        return [r for r in rows if str(r["ground_truth_item"]).lower() in index]

    backbone_raw, val_raw, charm_raw = split_conversations(
        load_split(os.path.join(args.data, "sft_data.json"), 0, args.seed),
        args.val_fraction, args.charm_fraction)
    rows = {"train": usable(expand_mentions(backbone_raw + charm_raw)),
            "val": usable(expand_mentions(val_raw))}
    lists = torch.load(args.shortlists)
    for k, rs in rows.items():
        t = torch.tensor([index[str(r["ground_truth_item"]).lower()] for r in rs])
        if not torch.equal(t, lists[k]["target"]):
            raise ValueError(f"{k} rows do not line up with {args.shortlists}")
    print(f"train={len(rows['train'])}  val={len(rows['val'])}  profiles="
          f"{sum(1 for v in profiles.values() if v)}", flush=True)

    local = os.path.isdir(args.base_model)
    tok = AutoTokenizer.from_pretrained(args.base_model, local_files_only=local)
    tok.padding_side = "left"
    tok.truncation_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        local_files_only=local)
    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=2 * args.lora_r,
                     lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                        "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(base, cfg).to(device)
    label_ids = torch.tensor([tok(" " + L, add_special_tokens=False)["input_ids"][0]
                              for L in LABELS[:args.top_m]], device=device)
    m = args.top_m

    turn_pos = {}
    for r in rows["train"]:
        turn_pos.setdefault(r["input"], set()).add(index[str(r["ground_truth_item"]).lower()])

    def cands_of(r):
        return [(catalog[c], profiles.get(catalog[c], "")) for c in r]

    def label_logits(prompts):
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=args.max_length).to(device)
        out = model(**enc, logits_to_keep=1)
        return out.logits[:, -1].float()[:, label_ids]

    @torch.no_grad()
    def evaluate():
        """Within-list quality on validation, candidates in retriever order."""
        model.eval()
        v = lists["val"]
        first = {}
        for i, r in enumerate(rows["val"]):
            first.setdefault(r["input"], i)
        idx = [i for i in first.values() if bool((v["top_idx"][i][:m] == v["target"][i]).any())]
        idx = idx[:args.val_limit]
        ranks = []
        for s in range(0, len(idx), 16):
            chunk = idx[s:s + 16]
            prompts = [direct_prompt(tok, rows["val"][i]["input"],
                                     cands_of(v["top_idx"][i][:m].tolist())) for i in chunk]
            logits = label_logits(prompts)
            for n, i in enumerate(chunk):
                pos = int((v["top_idx"][i][:m] == v["target"][i]).nonzero()[0])
                ranks.append(int((logits[n] > logits[n, pos]).sum()) + 1)
        model.train()
        ranks = torch.tensor(ranks, dtype=torch.float)
        retr = torch.tensor([float(int((v["top_idx"][i][:m] == v["target"][i]).nonzero()[0]) + 1)
                             for i in idx])
        return {"n": len(idx), "top1": float((ranks == 1).float().mean()),
                "mrr": float((1 / ranks).mean()), "retriever_top1": float((retr == 1).float().mean()),
                "retriever_mrr": float((1 / retr).mean())}

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    order = list(range(len(rows["train"])))
    random.Random(args.seed).shuffle(order)
    order = order[:args.max_train]
    total = (len(order) + args.batch - 1) // args.batch
    warmup = min(100, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(warmup, 1)) * max(0.0, 1 - s / max(total, 1)))

    ev = evaluate()
    print(f"step 0 (zero-shot) VAL within top-{m}: top-1 {100 * ev['top1']:.2f}%  MRR {ev['mrr']:.4f}"
          f"  | retriever order: top-1 {100 * ev['retriever_top1']:.2f}%  MRR {ev['retriever_mrr']:.4f}"
          f"  (n={ev['n']})", flush=True)
    best = {"mrr": ev["mrr"], "step": 0, "state": None}
    history = [{"step": 0, **ev}]
    rng = random.Random(args.seed + 1)
    tl = rows["train"]
    tr = lists["train"]
    model.train()
    start, run_loss, run_acc, run_n = time.time(), 0.0, 0.0, 0
    for step in range(1, total + 1):
        batch = order[(step - 1) * args.batch: step * args.batch]
        prompts, labels = [], []
        for i in batch:
            t = int(tr["target"][i])
            banned = turn_pos[tl[i]["input"]] - {t}
            cands = [c for c in tr["top_idx"][i].tolist() if c not in banned][:m]
            if t not in cands:
                cands[-1] = t
            rng.shuffle(cands)                      # no position carries information
            prompts.append(direct_prompt(tok, tl[i]["input"], cands_of(cands)))
            labels.append(cands.index(t))
        logits = label_logits(prompts)
        target = torch.tensor(labels, device=device)
        loss = F.cross_entropy(logits, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        run_loss += float(loss) * len(batch)
        run_acc += float((logits.argmax(1) == target).float().sum())
        run_n += len(batch)
        if step % 50 == 0:
            rate = (time.time() - start) / step
            print(f"  step {step}/{total}  loss {run_loss / run_n:.3f}  top-1 {100 * run_acc / run_n:.1f}%"
                  f"  ({rate:.2f}s/step, ~{rate * (total - step) / 60:.0f} min left)", flush=True)
            run_loss, run_acc, run_n = 0.0, 0.0, 0
        if step % args.eval_every == 0 or step == total:
            ev = evaluate()
            history.append({"step": step, **ev})
            print(f"step {step} VAL within top-{m}: top-1 {100 * ev['top1']:.2f}%  MRR {ev['mrr']:.4f}",
                  flush=True)
            if ev["mrr"] > best["mrr"]:
                best = {"mrr": ev["mrr"], "step": step,
                        "state": {n: p.detach().clone() for n, p in model.named_parameters()
                                  if p.requires_grad}}

    print(f"\nselected step {best['step']} (validation MRR within list {best['mrr']:.4f})", flush=True)
    if best["state"] is not None:
        named = dict(model.named_parameters())
        with torch.no_grad():
            for n, v in best["state"].items():
                named[n].copy_(v)
        model.save_pretrained(os.path.join(args.out_dir, "star_adapter"))
        print(f"saved {os.path.join(args.out_dir, 'star_adapter')}")
    with open(os.path.join(args.out_dir, "star_train.json"), "w") as f:
        json.dump({"args": vars(args), "selected_step": best["step"], "history": history}, f, indent=2)


if __name__ == "__main__":
    main()
