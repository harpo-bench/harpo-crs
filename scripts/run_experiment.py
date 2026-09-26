#!/usr/bin/env python3
"""
Train on real data and evaluate under the honest ranking protocols.

Answers the question the unit tests cannot: does the retrieval objective actually
improve ranking, measured against the full catalogue and against the baselines a
real model has to beat?

Every evaluation reports two test sets:

  * standard -- one case per (recommender turn, mentioned movie), nothing
                removed: how KBRD / KGSF / UniCRS report ReDial, so these are
                the numbers to put beside theirs
  * dedup    -- the same cases minus those whose movie is already named in the
                dialogue (the repetition shortcut)

and for each: full-catalogue Recall@K, the legacy sampled-100 protocol (for
reference only), and the repetition, popularity and random baselines. A model
that does not beat those baselines has not shown recommendation ability.

    python scripts/run_experiment.py --probe
    python scripts/run_experiment.py --train-size 600 --epochs 1 --catalog-size 400
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch


def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split(path, size, seed, require_item=True):
    with open(path) as f:
        rows = json.load(f)
    if require_item:
        rows = [r for r in rows if r.get("ground_truth_item")]
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:size] if size else rows


def split_conversations(rows, val_fraction, charm_fraction=0.0, salt="harpo-val"):
    """Split training data by whole conversations: (train, validation, charm).

    Turns of one conversation share most of their context, so a turn-level split
    would leak. The assignment hashes the conversation id: stable across runs,
    machines and --train-size, and independent of the shuffle seed.
    Validation takes buckets [0, val); CHARM's own training conversations take
    [val, val + charm), so validation is identical whatever charm_fraction is.

    CHARM must train on conversations the backbone never saw: on seen ones the
    retriever has memorised the answer (target in top-50: 79% vs 38% unseen)
    and a re-ranker learns to trust patterns that do not exist at test time.
    """
    train, val, charm = [], [], []
    for r in rows:
        key = f"{salt}:{r.get('conversation_id', r['input'])}"
        bucket = int(hashlib.md5(key.encode()).hexdigest(), 16) % 10000
        if bucket < val_fraction * 10000:
            val.append(r)
        elif bucket < (val_fraction + charm_fraction) * 10000:
            charm.append(r)
        else:
            train.append(r)
    return train, val, charm


def expand_mentions(rows):
    """One row per (recommender turn, mentioned movie).

    The converter keeps only the first movie of each turn as
    ``ground_truth_item``; the standard ReDial protocol scores every movie the
    recommender names. Item metadata is cleared so the in-batch item text is the
    bare title, the same text the catalogue and evaluation encode.
    """
    out = []
    for r in rows:
        movies = [m for m in (r.get("movies_mentioned") or []) if m] or [r["ground_truth_item"]]
        for m in dict.fromkeys(movies):  # dedupe within a turn, keep order
            out.append({**r, "ground_truth_item": m, "item_metadata": {}})
    return out


def build_model(model_name, device, seq_len, lora_r, train_embeddings, item_bias=True):
    from harpo.config import ModelConfig, TrainingConfig
    from harpo.model import HARPOMTv2

    model_config = ModelConfig(model_name=model_name, use_flash_attention=False,
                               lora_r=lora_r, lora_alpha=2 * lora_r)
    training_config = TrainingConfig(
        max_seq_length=seq_len, bf16=False, fp16=False,
        gradient_checkpointing=False, use_accelerate=False,
        dataloader_num_workers=0)
    training_config.retrieval_config.item_bias = item_bias

    model = HARPOMTv2(model_config, training_config)
    if not train_embeddings:
        # modules_to_save=['embed_tokens','lm_head'] makes ~40% of the model
        # trainable and dominates wall-clock on memory-bound devices.
        model._skip_modules_to_save = True
    model.load_base_model(device)
    return model, training_config


@torch.no_grad()
def encode_pooled(model, texts, device, max_length, batch_size=16):
    from harpo.pooling import masked_mean_pool

    chunks = []
    for start in range(0, len(texts), batch_size):
        enc = model.tokenizer(texts[start:start + batch_size], return_tensors="pt",
                              padding=True, truncation=True,
                              max_length=max_length).to(device)
        # logits_to_keep=1: only hidden states are used, and full logits at a
        # 151k vocabulary cost more memory than the whole 7B forward.
        out = model.base_model(input_ids=enc["input_ids"],
                              attention_mask=enc["attention_mask"],
                              output_hidden_states=True, logits_to_keep=1)
        chunks.append(masked_mean_pool(out.hidden_states[-1], enc["attention_mask"]))
    return torch.cat(chunks, dim=0)


def build_eval_catalog(model, movie_list, device, item_repr="model"):
    """Encode the catalogue exactly as training scores it (fusion + item bias)."""
    from harpo.retrieval import Item, ItemCatalog

    cuda = str(device).startswith("cuda")
    items = [Item(t.lower(), t) for t in movie_list]
    # "model" scores items with the fusion training used; "content" is the text
    # tower alone, kept only to diagnose what the id table contributes.
    catalog = ItemCatalog(model.retriever).build(
        items, lambda ts: encode_pooled(model, ts, device, 32,
                                        batch_size=256 if cuda else 64),
        batch_size=256 if cuda else 64, show_progress=True,
        item_encoder=model.encode_item_embeddings if item_repr == "model" else None,
        item_indices=list(range(len(items))),
        item_bias=(model.item_bias.weight[:len(items), 0]
                   if item_repr == "model" and getattr(model, "item_bias", None) is not None
                   else None))
    return items, catalog


def coldstart_parts(model, movie_list, device, profiles=None):
    """The pieces harpo.coldstart rebuilds item vectors from: text tower, id table, bias.

    With ``profiles`` (title -> BRIDGE profile) the text tower also encodes
    "title: profile" for every item that has one.
    """
    import torch.nn.functional as F

    n = len(movie_list)
    _, content = build_eval_catalog(model, movie_list, device, item_repr="content")
    parts = {"content": content.embeddings.float(),
             "id": F.normalize(model.item_id_embedding.weight[:n].detach().float(), dim=-1),
             "bias": model.item_bias.weight[:n, 0].detach().float(),
             "id_weight": model.retrieval_config.id_embedding_weight}
    if profiles:
        from harpo.retrieval import Item, ItemCatalog

        cuda = str(device).startswith("cuda")
        items = [Item(t.lower(), f"{t}: {profiles[t]}" if profiles.get(t) else t)
                 for t in movie_list]
        cat = ItemCatalog(model.retriever).build(
            items, lambda ts: encode_pooled(model, ts, device, 64,
                                            batch_size=256 if cuda else 64),
            batch_size=256 if cuda else 64)
        parts["profile_content"] = cat.embeddings.float()
    return parts


def encode_test_contexts(model, test_rows, device, seq_len):
    """Pooled context per test row; each distinct dialogue is encoded once."""
    cuda = str(device).startswith("cuda")
    unique = list(dict.fromkeys(r["input"] for r in test_rows))
    position = {c: i for i, c in enumerate(unique)}
    # The same span the training objective pools: dialogue + reply cue.
    encoded = encode_pooled(model, [c + "\nAssistant: " for c in unique], device,
                            seq_len, batch_size=64 if cuda else 8)
    return encoded[[position[r["input"]] for r in test_rows]], len(unique)


def evaluate(model, test_rows, movie_list, device, seq_len, label, seed,
             train_targets=(), item_repr="model"):
    """Evaluate on the standard and deduplicated test sets.

    Returns ``{"standard": [...], "dedup": [...]}``, each a list of
    ProtocolResult: full catalogue, sampled-100, repetition, popularity, random.
    """
    from harpo.ranking import (
        FullCatalogProtocol, PopularityBaseline, RandomBaseline, RepetitionBaseline,
        SampledProtocol, format_comparison, mentioned_in_context)
    print(f"\n{'=' * 78}\nEVALUATION: {label}\n{'=' * 78}")
    # After train_sft the model is still in train mode: LoRA dropout would make
    # every encoded context stochastic and the metrics unrepeatable.
    was_training = model.training
    model.eval()
    try:
        items, catalog = build_eval_catalog(model, movie_list, device, item_repr)
        ctx_all, n_dialogues = encode_test_contexts(model, test_rows, device, seq_len)
        contexts = [r["input"] for r in test_rows]
        titles = [str(r["ground_truth_item"]) for r in test_rows]
        keep = [i for i, (c, t) in enumerate(zip(contexts, titles))
                if not mentioned_in_context(t, c)]
        print(f"item representation: {item_repr}   test cases: {len(test_rows)} "
              f"from {n_dialogues} dialogues   already-mentioned: "
              f"{len(test_rows) - len(keep)} ({100 * (1 - len(keep) / max(len(test_rows), 1)):.1f}%)")

        out = {}
        for name, idx in (("standard", list(range(len(test_rows)))), ("dedup", keep)):
            ctx = ctx_all[idx]
            ctx_texts = [contexts[i] for i in idx]
            targets = [titles[i].lower() for i in idx]
            out[name] = [
                FullCatalogProtocol(catalog).evaluate(ctx, targets),
                SampledProtocol(catalog, num_negatives=99, seed=seed).evaluate(ctx, targets),
                RepetitionBaseline(items).evaluate(ctx_texts, targets),
                PopularityBaseline(items, train_targets).evaluate(targets),
                RandomBaseline(len(items), seed=seed).evaluate(len(targets)),
            ]
            title = ("STANDARD (all cases; comparable to published baselines)"
                     if name == "standard" else "DEDUP (already-mentioned movies removed)")
            print(f"\n--- {title}")
            print(format_comparison(out[name]))
    finally:
        model.train(was_training)
    return out


def to_dict(results):
    return {k: [r.to_dict() for r in v] for k, v in results.items()} if results else None


def headline(results):
    for name in ("standard", "dedup"):
        m = results[name][0].metrics
        print(f"  {name:<9} R@1 {100 * m.get('recall_at_1', 0):5.2f}  "
              f"R@10 {100 * m.get('recall_at_10', 0):5.2f}  "
              f"R@50 {100 * m.get('recall_at_50', 0):5.2f}  MRR {m.get('mrr', 0):.4f}  "
              f"(n={results[name][0].n_evaluated}, pool={results[name][0].pool_size})")


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)  # never leave a half-written file behind
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=os.path.expanduser("~/Desktop/harpo-work/redial_data"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default=None)
    parser.add_argument("--train-size", type=int, default=600)
    parser.add_argument("--test-size", type=int, default=120)
    parser.add_argument("--catalog-size", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=192)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train-embeddings", action="store_true")
    parser.add_argument("--catalog-refresh", type=int, default=100)
    parser.add_argument("--no-catalog-loss", action="store_true")
    parser.add_argument("--no-item-bias", action="store_true",
                        help="ablation: no log-popularity item bias")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--eval-only", default=None, metavar="CHECKPOINT",
                        help="evaluate a saved checkpoint (e.g. .../checkpoints/sft_final)")
    parser.add_argument("--eval-item-repr", choices=["model", "content"], default="model")
    parser.add_argument("--catalog-file", default=None,
                        help="JSON list fixing the catalogue order; defaults to the "
                             "catalog.json saved beside an --eval-only checkpoint")
    parser.add_argument("--no-expand-train", action="store_true",
                        help="train on the first movie per turn only (the converter default)")
    parser.add_argument("--val-fraction", type=float, default=0.05,
                        help="share of training conversations held out for model selection")
    parser.add_argument("--charm-fraction", type=float, default=0.0,
                        help="share of training conversations reserved for training the "
                             "CHARM re-ranker (excluded from this run's training)")
    parser.add_argument("--no-epoch-eval", action="store_true",
                        help="skip the evaluation after each epoch")
    parser.add_argument("--output", default=os.path.expanduser("~/Desktop/harpo-work/results"),
                        help="checkpoints and results.json")
    parser.add_argument("--results-dir", default=None,
                        help="where per-epoch and final JSON go (default: --output)")
    args = parser.parse_args()

    set_seeds(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")
    if device.startswith("cuda"):
        # float32 heads (retriever, id table, bias) run on tensor cores via TF32.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    results_dir = args.results_dir or args.output
    print(f"device={device}  model={args.model}  seq_len={args.seq_len}")
    print(f"train={args.train_size}  test={args.test_size}  epochs={args.epochs}")

    train_rows = load_split(os.path.join(args.data, "sft_data.json"),
                            args.train_size, args.seed)
    test_rows = expand_mentions(load_split(os.path.join(args.data, "test_sft.json"),
                                           args.test_size, args.seed))
    train_rows, val_rows, charm_rows = split_conversations(
        train_rows, args.val_fraction, args.charm_fraction)
    if charm_rows:
        print(f"reserved for CHARM (not trained on): {len(charm_rows)} turns from "
              f"{len({r.get('conversation_id') for r in charm_rows})} conversations")
    val_rows = expand_mentions(val_rows)
    if not args.no_expand_train:
        train_rows = expand_mentions(train_rows)
    else:
        train_rows = [{**r, "item_metadata": {}} for r in train_rows]
    print(f"train cases={len(train_rows)}  validation cases={len(val_rows)} "
          f"(held-out conversations)  test cases={len(test_rows)} "
          f"(one per recommender turn and mentioned movie)")
    with open(os.path.join(args.data, "movie_list.json")) as f:
        movies = json.load(f)
    if isinstance(movies, dict):
        movies = list(movies.values())
    # ReDial's movie table contains nulls for unresolved ids.
    movies = [m for m in movies if isinstance(m, str) and m.strip()]

    # Catalogue: every test target plus filler. Pool size is reported with every
    # metric, so a trimmed catalogue cannot be mistaken for the full one.
    targets = {str(r["ground_truth_item"]) for r in test_rows if r.get("ground_truth_item")}
    # The order matters: position i indexes the id table and the item bias, so a
    # checkpoint is only meaningful with the exact catalogue it was trained on.
    catalog_file = args.catalog_file
    if catalog_file is None and args.eval_only:
        saved = os.path.join(args.eval_only, "..", "..", "catalog.json")
        catalog_file = saved if os.path.exists(saved) else None
    if catalog_file:
        with open(catalog_file) as f:
            catalog = json.load(f)
        print(f"catalogue order from {catalog_file}")
    else:
        if args.eval_only:
            print("WARNING: no catalog.json beside the checkpoint; rebuilding the order, "
                  "which only matches if the data and arguments are identical")
        filler = [m for m in movies if m not in targets]
        random.Random(args.seed).shuffle(filler)
        catalog = sorted(targets) + filler[:max(0, args.catalog_size - len(targets))]
    missing = targets - set(catalog)
    print(f"catalog={len(catalog)} items ({len(targets) - len(missing)} test targets"
          f"{f', {len(missing)} missing' if missing else ''})")
    if not args.eval_only:
        write_json(os.path.join(args.output, "catalog.json"), catalog)

    # A checkpoint's adapter records its base model, which load_base_model reads.
    model_path = os.path.join(args.eval_only, "base_model") if args.eval_only else args.model
    model, training_config = build_model(model_path, device, args.seq_len,
                                         args.lora_r, args.train_embeddings,
                                         item_bias=not args.no_item_bias)
    training_config.batch_size = args.batch_size
    training_config.gradient_accumulation_steps = args.grad_accum
    training_config.sft_epochs = args.epochs
    training_config.output_dir = args.output
    model.model_config.sft_learning_rate = args.lr

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable: {trainable:,}/{total:,} ({100 * trainable / total:.1f}%)")

    from harpo.training import HARPOMTv2Trainer, SFTDataset

    # Index the *evaluation* catalogue, so the softmax denominator and the
    # metric's candidate set are the same. Training on a different (tiny)
    # candidate set is the mismatch that left in-batch accuracy at 0.66 while
    # full-catalogue R@10 sat at chance.
    item_index = {t.lower(): i for i, t in enumerate(catalog)}
    dataset = SFTDataset(train_rows, model.tokenizer, max_length=args.seq_len,
                         item_index=item_index, item_max_length=32)
    covered = sum(1 for r in train_rows
                  if str(r["ground_truth_item"]).lower() in item_index)
    print(f"train targets in catalogue: {covered}/{len(train_rows)} "
          f"({100 * covered / max(len(train_rows), 1):.1f}%)")

    # Training-split target frequency per catalogue item: the item-bias prior
    # and the popularity baseline both come from here, never from test data.
    item_counts = [0] * len(catalog)
    for r in train_rows:
        idx = item_index.get(str(r["ground_truth_item"]).lower())
        if idx is not None:
            item_counts[idx] += 1

    trainer = HARPOMTv2Trainer(model, training_config, device=device)
    if not args.no_catalog_loss:
        trainer.attach_catalog(catalog, item_index, refresh_every=args.catalog_refresh,
                               item_counts=item_counts)

    train_targets = [str(r["ground_truth_item"]).lower() for r in train_rows]

    if args.eval_only:
        trainer.load_checkpoint(args.eval_only)
        result = evaluate(model, test_rows, catalog, device, args.seq_len,
                          f"CHECKPOINT {args.eval_only}", args.seed,
                          train_targets, args.eval_item_repr)
        print("\nHEADLINE (full catalogue)")
        headline(result)
        write_json(os.path.join(results_dir, f"eval_{args.eval_item_repr}.json"),
                   {"args": vars(args), "device": device, "results": to_dict(result)})
        return

    if args.probe:
        from torch.utils.data import DataLoader
        from harpo.training import trim_padding_collate
        model.freeze_for_sft()
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            collate_fn=trim_padding_collate)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=args.lr)
        times = []
        for i, batch in enumerate(loader):
            if i >= 6:
                break
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            start = time.time()
            out = model(input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"], domain=batch["domain_idx"],
                        vto_labels=batch["vto_labels"], training_stage="sft")
            out.loss.backward()
            opt.step()
            opt.zero_grad()
            if device == "mps":
                torch.mps.synchronize()
            times.append(time.time() - start)
            print(f"  step {i}: {times[-1]:.2f}s  loss={out.loss.item():.3f}")
        steady = sum(times[2:]) / max(len(times[2:]), 1)
        steps = (len(dataset) // args.batch_size) * args.epochs
        print(f"\nsteady-state: {steady:.2f}s/step -> ~{steady * steps / 60:.0f} min "
              f"for {args.epochs} epoch(s), excluding evaluation")
        if device.startswith("cuda"):
            print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB "
                  f"of {torch.cuda.get_device_properties(0).total_memory / 2**30:.0f}")
        return

    before = evaluate(model, test_rows, catalog, device, args.seq_len,
                      "BEFORE training", args.seed, train_targets, args.eval_item_repr)
    print("\nHEADLINE before training (full catalogue)")
    headline(before)
    write_json(os.path.join(results_dir, "before.json"),
               {"args": vars(args), "device": device, "results": to_dict(before)})

    per_epoch, per_epoch_val = {}, {}

    def on_epoch_end(epoch, stats):
        # Results survive even if the job is killed before the last epoch.
        if args.no_epoch_eval and epoch + 1 < args.epochs:
            return
        val_res = None
        if val_rows:
            val_res = evaluate(model, val_rows, catalog, device, args.seq_len,
                               f"VALIDATION after epoch {epoch + 1}/{args.epochs}", args.seed,
                               train_targets, args.eval_item_repr)
            per_epoch_val[epoch] = val_res
            print(f"\nVALIDATION after epoch {epoch + 1} (full catalogue)")
            headline(val_res)
        res = evaluate(model, test_rows, catalog, device, args.seq_len,
                       f"TEST after epoch {epoch + 1}/{args.epochs}", args.seed,
                       train_targets, args.eval_item_repr)
        per_epoch[epoch] = res
        print(f"\nTEST after epoch {epoch + 1} (full catalogue)")
        headline(res)
        write_json(os.path.join(results_dir, f"epoch{epoch + 1}.json"),
                   {"args": vars(args), "device": device, "epoch": epoch + 1,
                    "train_stats": stats, "results": to_dict(res),
                    "validation": to_dict(val_res)})

    print(f"\n{'=' * 78}\nTRAINING: {args.epochs} epoch(s) on {len(dataset)} examples\n{'=' * 78}")
    start = time.time()
    losses = trainer.train_sft(dataset, epoch_end_callback=on_epoch_end)
    print(f"\ntraining took {(time.time() - start) / 60:.1f} min (including per-epoch evaluation)")
    for e in [x for x in losses if x["stage"] == "sft"]:
        line = f"  epoch {e['epoch']}: loss={e['loss']:.4f} in_batch_acc={e['retrieval_acc']:.3f}"
        if e.get("catalog_rank") is not None:
            line += (f" catalog_rank={e['catalog_rank']:.1f}"
                     f" catalog_top1={100 * e['catalog_acc']:.2f}%")
        if e.get("nan_steps"):
            line += f" SKIPPED {e['nan_steps']} non-finite"
        print(line)

    # Model selection on validation only; the last epoch is kept alongside so
    # the choice can be audited.
    last = max(per_epoch)
    if per_epoch_val:
        best = max(per_epoch_val,
                   key=lambda e: per_epoch_val[e]["standard"][0].metrics.get("mrr", 0.0))
        print(f"\nepoch selected on validation MRR: {best + 1} (last: {last + 1})")
    else:
        best = last
    after = per_epoch[best]
    write_json(os.path.join(results_dir, "results.json"),
               {"args": vars(args), "device": device, "trainable_params": trainable,
                "losses": losses, "before": to_dict(before), "after": to_dict(after),
                "selected_epoch": best + 1, "last_epoch": last + 1,
                "after_last_epoch": to_dict(per_epoch[last]),
                "per_epoch": {e + 1: to_dict(r) for e, r in per_epoch.items()},
                "per_epoch_validation": {e + 1: to_dict(r) for e, r in per_epoch_val.items()}})

    print(f"\n{'=' * 78}\nSUMMARY (full catalogue)\n{'=' * 78}")
    for name in ("standard", "dedup"):
        b, a = before[name][0].metrics, after[name][0].metrics
        print(f"\n  [{name}]")
        for k in ("recall_at_1", "recall_at_10", "recall_at_50", "mrr"):
            if k in b and k in a:
                print(f"  {k:<14} {b[k]:.4f} -> {a[k]:.4f}  "
                      f"({'+' if a[k] >= b[k] else ''}{100 * (a[k] - b[k]):.2f} pts)")
        rep = after[name][2].metrics.get("recall_at_10", 0.0)
        pop = after[name][3].metrics.get("recall_at_10", 0.0)
        print(f"  baselines R@10: repetition {rep:.4f}  popularity {pop:.4f}")
        if a.get("recall_at_10", 0.0) <= max(rep, pop):
            print("  WARNING: does not beat the repetition/popularity baselines.")


if __name__ == "__main__":
    main()
