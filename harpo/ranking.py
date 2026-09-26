"""
Ranking evaluation protocols for HARPO.

Two problems made the original ranking numbers non-comparable to the baselines
printed beside them.

**1. Candidate set size.** The published ReDial protocol ranks the ground truth
against the full catalogue -- roughly 6.5k movies. The baseline rows in the paper
(KBRD 2.9/16.7/36.2, KGSF 3.8/18.1/37.4, UniCRS 4.8/21.2/40.8) match published
full-catalogue figures closely. But ``evaluation.RankingEvaluator`` scored against
1 positive + 99 sampled negatives, a ~65x easier task. Simulated, a sampled-100
cutoff corresponds to a full-catalogue cutoff roughly 65x larger, so the reported
R@10 of 29.8 is closer to a full-catalogue R@649, and R@50 of 50.2 is exactly
chance for a pool of 100.

**2. Repetition shortcut.** The SIGIR 2026 standardized ReDial re-evaluation found
roughly half of reported accuracy comes from recommending items already present in
the dialogue context, and that a naive repeat-mentioned baseline reaches
R@1 = 0.043 -- above several trained systems. The original evaluator did no
deduplication.
"""

import math
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .retrieval import Item, ItemCatalog, _inference_mode


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metrics_from_ranks(ranks: Sequence[float],
                       pool_size: int,
                       cutoffs: Sequence[int] = (1, 5, 10, 20, 50),
                       max_chance_rate: float = 0.2) -> Dict[str, float]:
    """Recall/NDCG/MRR from 1-based ranks, omitting cutoffs that are at chance.

    A cutoff K over a pool of N has a random baseline of K/N. At K=50, N=100 that
    is 50%, so the metric says nothing about the model; such cutoffs are skipped.
    ``random_baseline_at_k`` is reported so any surviving number can be read
    against its floor.
    """
    ranks = [r for r in ranks if r == r]  # drop NaN
    n = len(ranks)
    if n == 0:
        return {"n": 0}

    out: Dict[str, float] = {"n": n, "pool_size": pool_size}
    for k in cutoffs:
        if pool_size > 0 and k / pool_size > max_chance_rate:
            continue
        out[f"recall_at_{k}"] = sum(1 for r in ranks if r <= k) / n
        out[f"ndcg_at_{k}"] = sum(1.0 / math.log2(r + 1) for r in ranks if r <= k) / n
        out[f"random_baseline_at_{k}"] = k / pool_size if pool_size else 0.0

    out["mrr"] = sum(1.0 / r for r in ranks) / n
    out["mrr_at_10"] = sum(1.0 / r for r in ranks if r <= 10) / n
    out["mean_rank"] = sum(ranks) / n
    return out


def rank_with_ties(scores: torch.Tensor, target_index: int) -> float:
    """1-based rank of ``target_index``, averaging over ties.

    Counting only strictly-greater candidates (as the original did) hands rank 1
    to any scorer that emits a constant -- which is exactly what the old
    recommendation head, regressed against one scalar per batch, would produce.
    """
    target = scores[target_index]
    better = int((scores > target).sum())
    tied = int((scores == target).sum()) - 1
    return better + 1.0 + tied / 2.0


def rank_with_tie_handling(scores: List[float], gt_index: int = 0) -> float:
    """List-based variant of :func:`rank_with_ties`."""
    if not scores:
        return 0.0
    gt_score = scores[gt_index]
    better = sum(1 for i, s in enumerate(scores) if i != gt_index and s > gt_score)
    tied = sum(1 for i, s in enumerate(scores) if i != gt_index and s == gt_score)
    return better + 1.0 + tied / 2.0


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _normalise(title: str) -> str:
    """Lowercase, strip a trailing year, collapse punctuation and whitespace."""
    title = title.lower().strip()
    title = re.sub(r"\s*\(\s*(19|20)\d{2}\s*\)\s*$", "", title)
    title = re.sub(r"[^\w\s]", " ", title)
    return re.sub(r"\s+", " ", title).strip()


def mentioned_in_context(item_title: str, context: str) -> bool:
    """Whether ``item_title`` already appears in the dialogue context."""
    norm_title = _normalise(item_title)
    if not norm_title:
        return False
    return norm_title in _normalise(context)


def deduplicate(test_data: List[Dict],
                context_key: str = "input",
                gt_key: str = "ground_truth_item") -> Tuple[List[Dict], int]:
    """Drop instances whose ground truth already appears in the context.

    Recommending something the user just named is not recommendation, and roughly
    half of headline ReDial accuracy has been shown to come from exactly that.
    """
    kept, removed = [], 0
    for item in test_data:
        gt = item.get(gt_key)
        if gt and mentioned_in_context(str(gt), item.get(context_key, "")):
            removed += 1
            continue
        kept.append(item)
    return kept, removed


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

@dataclass
class ProtocolResult:
    """Metrics plus the provenance needed to interpret them."""
    protocol: str
    metrics: Dict[str, float]
    n_evaluated: int
    n_skipped: int = 0
    n_deduplicated: int = 0
    pool_size: int = 0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "protocol": self.protocol,
            "n_evaluated": self.n_evaluated,
            "n_skipped": self.n_skipped,
            "n_deduplicated": self.n_deduplicated,
            "pool_size": self.pool_size,
            "notes": self.notes,
            **self.metrics,
        }


class FullCatalogProtocol:
    """Rank the ground truth against the entire catalogue.

    This is what the published baselines were measured under, so it is the only
    setting in which a comparison against them means anything.
    """

    name = "full_catalog"

    def __init__(self, catalog: ItemCatalog):
        self.catalog = catalog

    def evaluate(self, context_embeddings: torch.Tensor,
                 target_ids: Sequence[str],
                 cutoffs: Sequence[int] = (1, 10, 50)) -> ProtocolResult:
        ranks = self.catalog.rank_of(context_embeddings, target_ids)
        valid = [r for r in ranks if r == r]
        return ProtocolResult(
            protocol=self.name,
            metrics=metrics_from_ranks(valid, len(self.catalog), cutoffs),
            n_evaluated=len(valid),
            n_skipped=len(ranks) - len(valid),
            pool_size=len(self.catalog),
            notes=[f"ranked against full catalogue of {len(self.catalog)} items"],
        )


class SampledProtocol:
    """Legacy 1-positive + N-negatives protocol.

    Retained so results stay comparable to the currently published table, but it
    is *not* comparable to the baselines in that table, which were scored against
    the full catalogue. Always report it beside ``FullCatalogProtocol``.
    """

    name = "sampled"

    def __init__(self, catalog: ItemCatalog, num_negatives: int = 99, seed: int = 0):
        self.catalog = catalog
        self.num_negatives = num_negatives
        self.seed = seed

    def evaluate(self, context_embeddings: torch.Tensor,
                 target_ids: Sequence[str],
                 cutoffs: Sequence[int] = (1, 10, 50)) -> ProtocolResult:
        rng = random.Random(self.seed)  # seeded: the original sampled unseeded
        pool_size = self.num_negatives + 1

        # eval mode: encoding outside it leaves dropout live, making scores
        # stochastic and the seed meaningless.
        all_scores = self.catalog.scores(context_embeddings)

        ranks, skipped = [], 0
        n_items = len(self.catalog)
        for row, target in zip(all_scores, target_ids):
            idx = self.catalog._index.get(target)
            if idx is None:
                skipped += 1
                continue
            choices = [i for i in range(n_items) if i != idx]
            negatives = rng.sample(choices, min(self.num_negatives, len(choices)))
            subset = row[[idx] + negatives]
            ranks.append(rank_with_ties(subset, 0))

        return ProtocolResult(
            protocol=f"{self.name}_{pool_size}",
            metrics=metrics_from_ranks(ranks, pool_size, cutoffs),
            n_evaluated=len(ranks),
            n_skipped=skipped,
            pool_size=pool_size,
            notes=[
                f"1 positive + {self.num_negatives} sampled negatives",
                "NOT comparable to published baselines, which use the full catalogue",
            ],
        )


# ---------------------------------------------------------------------------
# Sanity baselines
# ---------------------------------------------------------------------------

class RepetitionBaseline:
    """Rank items by whether they already appear in the dialogue context.

    The floor any real model must clear. On ReDial this naive rule reaches
    R@1 ~ 0.043, above several trained systems.
    """

    name = "repetition_baseline"

    def __init__(self, items: Sequence[Item]):
        self.items = list(items)
        # Normalise every title once. Calling mentioned_in_context per
        # (context, item) pair normalises *both* sides each time -- at 4k test
        # instances against a 6.6k catalogue that is 27.8M pairs and does not
        # finish in usable time.
        self._norm_titles = [_normalise(it.title) for it in self.items]
        self._index = {it.item_id: i for i, it in enumerate(self.items)}

    def evaluate(self, contexts: Sequence[str], target_ids: Sequence[str],
                 cutoffs: Sequence[int] = (1, 10, 50)) -> ProtocolResult:
        ranks, skipped = [], 0
        for context, target in zip(contexts, target_ids):
            if target not in self._index:
                skipped += 1
                continue
            norm_context = _normalise(context)  # once per instance, not per item
            scores = torch.tensor(
                [1.0 if t and t in norm_context else 0.0 for t in self._norm_titles]
            )
            ranks.append(rank_with_ties(scores, self._index[target]))

        return ProtocolResult(
            protocol=self.name,
            metrics=metrics_from_ranks(ranks, len(self.items), cutoffs),
            n_evaluated=len(ranks),
            n_skipped=skipped,
            pool_size=len(self.items),
            notes=["recommends items already mentioned in context"],
        )


class PopularityBaseline:
    """Rank items by how often they were the training target.

    Needs no context at all, so a model that cannot beat it is not using the
    dialogue. ReDial is heavily skewed towards a few titles, which makes this
    floor far higher than random.
    """

    name = "popularity_baseline"

    def __init__(self, items: Sequence[Item], train_targets: Sequence[str]):
        counts: Dict[str, int] = {}
        for t in train_targets:
            counts[t] = counts.get(t, 0) + 1
        self.items = list(items)
        self._index = {it.item_id: i for i, it in enumerate(self.items)}
        self._scores = torch.tensor([float(counts.get(it.item_id, 0)) for it in self.items])

    def evaluate(self, target_ids: Sequence[str],
                 cutoffs: Sequence[int] = (1, 10, 50)) -> ProtocolResult:
        ranks, skipped = [], 0
        for target in target_ids:
            if target not in self._index:
                skipped += 1
                continue
            ranks.append(rank_with_ties(self._scores, self._index[target]))
        return ProtocolResult(
            protocol=self.name,
            metrics=metrics_from_ranks(ranks, len(self.items), cutoffs),
            n_evaluated=len(ranks),
            n_skipped=skipped,
            pool_size=len(self.items),
            notes=["ranks by training-target frequency; ignores the dialogue"],
        )


class RandomBaseline:
    """Uniformly random ranking -- the absolute floor."""

    name = "random_baseline"

    def __init__(self, num_items: int, seed: int = 0):
        self.num_items = num_items
        self.seed = seed

    def evaluate(self, n_examples: int,
                 cutoffs: Sequence[int] = (1, 10, 50)) -> ProtocolResult:
        rng = random.Random(self.seed)
        ranks = [rng.randint(1, self.num_items) for _ in range(n_examples)]
        return ProtocolResult(
            protocol=self.name,
            metrics=metrics_from_ranks(ranks, self.num_items, cutoffs),
            n_evaluated=n_examples,
            pool_size=self.num_items,
            notes=["uniform random ranking"],
        )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_comparison(results: Sequence[ProtocolResult],
                      cutoffs: Sequence[int] = (1, 10, 50)) -> str:
    """Render protocols side by side, so a reader cannot conflate them."""
    lines = []
    header = f"{'protocol':<24}{'n':>7}{'pool':>8}"
    for k in cutoffs:
        header += f"{'R@' + str(k):>9}"
    header += f"{'MRR':>9}"
    lines.append(header)
    lines.append("-" * len(header))

    for res in results:
        row = f"{res.protocol:<24}{res.n_evaluated:>7}{res.pool_size:>8}"
        for k in cutoffs:
            value = res.metrics.get(f"recall_at_{k}")
            row += f"{value * 100:>8.1f}%" if value is not None else f"{'--':>9}"
        row += f"{res.metrics.get('mrr', 0.0):>9.4f}"
        lines.append(row)

    notes = [f"  {r.protocol}: {n}" for r in results for n in r.notes]
    if notes:
        lines.append("")
        lines.extend(notes)
    return "\n".join(lines)
