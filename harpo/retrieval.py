"""
Item retrieval for HARPO.

Why this module exists
----------------------
Recall@K, MRR@K and NDCG@K are the headline metrics, but nothing in the original
four-stage curriculum ever taught the model to rank *items*. The only signal
reaching ``recommendation_head`` was ``torch.exp(-outputs.loss)`` -- the batch-mean
language-modelling loss, one scalar broadcast across the batch. A head regressed
against a per-batch constant can only learn to emit a constant.

There were no item embeddings anywhere and no positive/negative contrast at any
stage. That is a missing objective, not a broken one, and it explains ranking
numbers that sat at chance.
"""

import contextlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@contextlib.contextmanager
def _inference_mode(module: nn.Module) -> Iterator[None]:
    """Temporarily put ``module`` in eval mode.

    Both towers contain dropout. Encoding or ranking in training mode makes
    embeddings stochastic, so the catalogue is built from noisy vectors and the
    same query can return different orderings. Ranking must be deterministic.
    """
    was_training = module.training
    module.eval()
    try:
        yield
    finally:
        module.train(was_training)


@dataclass
class Item:
    """One recommendable item.

    ``text`` is what the item tower encodes. Metadata beyond the title matters:
    titles alone are ambiguous ("Star Wars" vs "Star Wars (1977)"), and the
    original negative sampler happily served such pairs as each other's negatives.
    """
    item_id: str
    title: str
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def text(self) -> str:
        parts = [self.title]
        for key in ("year", "genre", "director", "description", "category", "brand"):
            value = self.metadata.get(key)
            if value:
                parts.append(f"{key}: {value}")
        return " | ".join(parts)


class TwoTowerRetriever(nn.Module):
    """Dual-encoder retriever trained with InfoNCE.

    Both towers consume pooled hidden states and project into a shared
    L2-normalised space, so scoring is a dot product and the whole catalogue can
    be ranked with one matmul.
    """

    def __init__(self, hidden_size: int, embed_dim: int = 256,
                 dropout: float = 0.1, init_temperature: float = 0.07):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim

        def tower() -> nn.Module:
            return nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, embed_dim),
            )

        self.context_tower = tower()
        self.item_tower = tower()
        self.log_temperature = nn.Parameter(torch.tensor(init_temperature).log())

    @property
    def temperature(self) -> torch.Tensor:
        # Clamped: an unbounded temperature can drive the softmax to a one-hot
        # and stall the gradient.
        return self.log_temperature.exp().clamp(min=0.01, max=1.0)

    def encode_context(self, pooled: torch.Tensor) -> torch.Tensor:
        pooled = pooled.to(self.context_tower[0].weight.dtype)  # fp32 head, bf16 backbone
        return F.normalize(self.context_tower(pooled), dim=-1)

    def encode_item(self, pooled: torch.Tensor) -> torch.Tensor:
        pooled = pooled.to(self.item_tower[0].weight.dtype)
        return F.normalize(self.item_tower(pooled), dim=-1)

    def score(self, context_embeds: torch.Tensor,
              item_embeds: torch.Tensor) -> torch.Tensor:
        """``[n_ctx, n_items]`` similarities."""
        return context_embeds @ item_embeds.t()

    def contrastive_loss(self, context_pooled: torch.Tensor,
                         positive_pooled: torch.Tensor,
                         negative_pooled: Optional[torch.Tensor] = None,
                         positive_ids: Optional[Sequence[str]] = None
                         ) -> Dict[str, torch.Tensor]:
        """InfoNCE with in-batch negatives and optional mined hard negatives."""
        ctx = self.encode_context(context_pooled)
        pos = self.encode_item(positive_pooled)
        batch = ctx.size(0)
        device = ctx.device

        logits = (ctx @ pos.t()) / self.temperature

        if positive_ids is not None:
            ids = list(positive_ids)
            same = torch.tensor(
                [[a == b for b in ids] for a in ids], device=device, dtype=torch.bool
            )
            # Keep the diagonal (true positive), drop other copies of it: a few
            # titles dominate ReDial, and without this a repeated item is trained
            # to push itself away.
            same &= ~torch.eye(batch, dtype=torch.bool, device=device)
            logits = logits.masked_fill(same, torch.finfo(logits.dtype).min)

        if negative_pooled is not None and negative_pooled.numel() > 0:
            n_neg = negative_pooled.size(1)
            neg = self.encode_item(negative_pooled.reshape(-1, self.hidden_size))
            neg = neg.reshape(batch, n_neg, self.embed_dim)
            neg_logits = torch.einsum("bd,bnd->bn", ctx, neg) / self.temperature
            logits = torch.cat([logits, neg_logits], dim=1)

        targets = torch.arange(batch, device=device)
        loss = F.cross_entropy(logits, targets)
        return {
            "loss": loss,
            "accuracy": (logits.argmax(dim=1) == targets).float().mean(),
            "temperature": self.temperature.detach(),
        }


class CatalogSoftmax(nn.Module):
    """Full-catalogue softmax: score the context against *every* item.

    Why this exists
    ---------------
    In-batch InfoNCE optimises a far easier task than the one measured. At batch 4
    it supplies 3 negatives; evaluation then ranks against thousands. A model can
    reach 66% in-batch accuracy while sitting at chance on the real task -- exactly
    what the first ReDial run showed: in-batch accuracy 0.664, full-catalogue R@10
    0.43% against a 0.6% random floor.

    This makes the training objective the evaluation objective: softmax over all N
    items, cross-entropy against the true index.

    Item representations combine a **learnable id table**, which receives gradient
    for the positive and every negative on every step, with a **cached content
    matrix** re-encoded from item text periodically (detached; the in-batch loss is
    what trains the item tower).
    """

    def __init__(self, retriever: TwoTowerRetriever,
                 item_id_embedding: Optional[nn.Embedding],
                 num_items: int,
                 id_weight: float = 0.5,
                 label_smoothing: float = 0.0,
                 item_bias: Optional[nn.Embedding] = None):
        super().__init__()
        self.retriever = retriever
        self.item_id_embedding = item_id_embedding
        self.item_bias = item_bias
        self.num_items = num_items
        self.id_weight = id_weight
        self.label_smoothing = label_smoothing
        self.register_buffer(
            "content", torch.zeros(num_items, retriever.embed_dim), persistent=False
        )
        self._has_content = False

    @torch.no_grad()
    def set_content(self, content_embeds: torch.Tensor) -> None:
        """Install a freshly encoded, L2-normalised content matrix ``[N, d]``."""
        if content_embeds.shape[0] != self.num_items:
            raise ValueError(
                f"expected {self.num_items} item embeddings, got {content_embeds.shape[0]}"
            )
        self.content = F.normalize(
            content_embeds.detach().to(self.content.device, self.content.dtype), dim=-1
        )
        self._has_content = True

    def item_matrix(self) -> torch.Tensor:
        """``[N, d]`` item embeddings, id table fused with cached content."""
        if self.item_id_embedding is None:
            return self.content

        ids = torch.arange(self.num_items, device=self.content.device)
        id_embeds = F.normalize(self.item_id_embedding(ids), dim=-1)
        if not self._has_content:
            return id_embeds
        return F.normalize(
            self.id_weight * id_embeds + (1.0 - self.id_weight) * self.content, dim=-1
        )

    def forward(self, context_pooled: torch.Tensor,
                target_indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Cross-entropy over the full catalogue.

        Rows whose target is -1 (item outside the catalogue) are skipped, since
        they have no softmax target.
        """
        valid = (target_indices >= 0) & (target_indices < self.num_items)
        if not bool(valid.any()):
            zero = context_pooled.new_zeros(())
            return {"loss": zero, "accuracy": zero, "rank": zero, "n": 0}

        ctx = self.retriever.encode_context(context_pooled[valid])
        items = self.item_matrix().to(ctx.dtype)
        targets = target_indices[valid]

        logits = (ctx @ items.t()) / self.retriever.temperature
        if self.item_bias is not None:
            logits = logits + self.item_bias.weight[:self.num_items, 0].to(logits.dtype)
        loss = F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)

        with torch.no_grad():
            gold = logits.gather(1, targets.unsqueeze(1))
            # Mean rank over the whole catalogue -- directly comparable to the
            # eval protocol, unlike in-batch accuracy.
            rank = (logits > gold).sum(dim=1).float().mean() + 1.0
            accuracy = (logits.argmax(dim=1) == targets).float().mean()

        return {"loss": loss, "accuracy": accuracy, "rank": rank,
                "n": int(valid.sum())}


class ItemCatalog:
    """Encode-once, rank-many item embedding cache.

    Holding the catalogue in memory is what makes full-catalogue ranking
    practical. The original evaluator ran a separate backbone forward pass *per
    candidate* -- roughly 100k for a 1k test set -- which is why ``max_samples``
    defaulted to 100.
    """

    def __init__(self, retriever: TwoTowerRetriever):
        self.retriever = retriever
        self.items: List[Item] = []
        self._embeddings: Optional[torch.Tensor] = None
        self._bias: Optional[torch.Tensor] = None
        self._index: Dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.items)

    @property
    def embeddings(self) -> torch.Tensor:
        if self._embeddings is None:
            raise RuntimeError("call build() before ranking")
        return self._embeddings

    def build(self, items: Sequence[Item],
              encode_fn: Callable[[List[str]], torch.Tensor],
              batch_size: int = 64,
              show_progress: bool = False,
              item_encoder: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
              item_indices: Optional[Sequence[int]] = None,
              item_bias: Optional[torch.Tensor] = None) -> "ItemCatalog":
        """Encode the catalogue once.

        ``item_encoder(pooled, indices)`` replaces the plain text tower -- pass
        the model's id/content fusion so items are scored exactly as training
        scored them. Encoding text alone discards the learned id table, and the
        context tower, trained against fused vectors, then ranks at chance.
        """
        self.items = list(items)
        self._index = {item.item_id: i for i, item in enumerate(self.items)}
        self._bias = None if item_bias is None else item_bias.detach().float()

        chunks = []
        iterator = range(0, len(self.items), batch_size)
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Encoding catalog")

        with _inference_mode(self.retriever), torch.no_grad():
            for start in iterator:
                texts = [it.text for it in self.items[start:start + batch_size]]
                pooled = encode_fn(texts)
                if item_encoder is None:
                    chunks.append(self.retriever.encode_item(pooled))
                else:
                    ids = (item_indices[start:start + batch_size] if item_indices is not None
                           else range(start, start + len(texts)))
                    ids = torch.tensor(list(ids), dtype=torch.long, device=pooled.device)
                    chunks.append(item_encoder(pooled, ids))

        self._embeddings = torch.cat(chunks, dim=0) if chunks else torch.empty(0)
        return self

    def override(self, embeddings: torch.Tensor, bias: Optional[torch.Tensor]) -> "ItemCatalog":
        """Replace the encoded vectors and bias (e.g. harpo.coldstart), keeping the items."""
        if embeddings.shape[0] != len(self.items):
            raise ValueError(f"expected {len(self.items)} item vectors, got {embeddings.shape[0]}")
        self._embeddings = embeddings.detach()
        self._bias = None if bias is None else bias.detach().float()
        return self

    def scores(self, context_pooled: torch.Tensor) -> torch.Tensor:
        """``[n_ctx, N]`` scores, exactly as the training softmax computed them.

        With an item bias the similarity is put on the logit scale first (divided
        by the temperature); without one that division cannot change a ranking.
        """
        with _inference_mode(self.retriever), torch.no_grad():
            ctx = self.retriever.encode_context(context_pooled)
            scores = self.retriever.score(ctx, self.embeddings.to(ctx.device, ctx.dtype))
            if self._bias is not None:
                scores = (scores / self.retriever.temperature
                          + self._bias.to(scores.device, scores.dtype))
        return scores

    def rank(self, context_pooled: torch.Tensor, top_k: Optional[int] = None
             ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(scores, indices)`` over the whole catalogue, best first."""
        with torch.no_grad():
            scores = self.scores(context_pooled)
            k = min(top_k or scores.size(1), scores.size(1))
            return scores.topk(k, dim=1)

    def rank_of(self, context_pooled: torch.Tensor,
                target_ids: Sequence[str]) -> List[float]:
        """1-based rank of each target over the full catalogue, ties averaged."""
        scores = self.scores(context_pooled)

        ranks = []
        for row, item_id in zip(scores, target_ids):
            idx = self._index.get(item_id)
            if idx is None:
                ranks.append(float("nan"))
                continue
            target = row[idx]
            better = int((row > target).sum())
            tied = int((row == target).sum()) - 1
            ranks.append(better + 1.0 + tied / 2.0)
        return ranks


class NegativeQueue:
    """Cross-batch memory bank of item embeddings (MoCo-style).

    At batch 4 the in-batch objective supplies **3** negatives per positive, which
    is far too few for a catalogue of thousands. A queue decouples the number of
    negatives from the batch size at the cost of one enqueue per step. Entries are
    detached, so no gradient flows through stale embeddings.
    """

    def __init__(self, embed_dim: int, capacity: int = 4096):
        self.capacity = capacity
        self.embed_dim = embed_dim
        self._embeds = torch.zeros(capacity, embed_dim)
        self._indices = torch.full((capacity,), -1, dtype=torch.long)
        self._size = 0
        self._ptr = 0

    def __len__(self) -> int:
        return self._size

    @torch.no_grad()
    def enqueue(self, embeds: torch.Tensor,
                item_indices: Optional[torch.Tensor] = None) -> None:
        embeds = embeds.detach().to(self._embeds.device, self._embeds.dtype)
        n = embeds.size(0)
        if n == 0:
            return
        if n >= self.capacity:
            embeds, n = embeds[-self.capacity:], self.capacity
            if item_indices is not None:
                item_indices = item_indices[-self.capacity:]

        end = self._ptr + n
        if end <= self.capacity:
            self._embeds[self._ptr:end] = embeds
            if item_indices is not None:
                self._indices[self._ptr:end] = item_indices.detach().cpu()
        else:
            first = self.capacity - self._ptr
            self._embeds[self._ptr:] = embeds[:first]
            self._embeds[:end - self.capacity] = embeds[first:]
            if item_indices is not None:
                idx = item_indices.detach().cpu()
                self._indices[self._ptr:] = idx[:first]
                self._indices[:end - self.capacity] = idx[first:]

        self._ptr = end % self.capacity
        self._size = min(self._size + n, self.capacity)

    @torch.no_grad()
    def sample_hard(self, context_embeds: torch.Tensor, num_negatives: int,
                    exclude: Optional[torch.Tensor] = None,
                    popularity: Optional[torch.Tensor] = None,
                    debias: float = 0.0) -> Optional[torch.Tensor]:
        """``[batch, num_negatives, embed_dim]`` hard negatives, or None if empty."""
        if self._size == 0 or num_negatives <= 0:
            return None

        bank = self._embeds[:self._size].to(context_embeds.device)
        bank_idx = self._indices[:self._size].to(context_embeds.device)
        scores = context_embeds.detach().to(bank.dtype) @ bank.t()

        if exclude is not None:
            same = bank_idx.unsqueeze(0) == exclude.unsqueeze(1)
            same &= bank_idx.unsqueeze(0) >= 0
            scores = scores.masked_fill(same, torch.finfo(scores.dtype).min)

        if popularity is not None and debias > 0:
            valid = bank_idx.clamp(min=0)
            pop = popularity.to(scores.device)[valid]
            pop = torch.where(bank_idx >= 0, pop, torch.ones_like(pop))
            scores = scores - debias * pop.log().unsqueeze(0)

        k = min(num_negatives, self._size)
        top = scores.topk(k, dim=1).indices
        negatives = bank[top]

        if k < num_negatives:
            pad = negatives[:, -1:, :].expand(-1, num_negatives - k, -1)
            negatives = torch.cat([negatives, pad], dim=1)
        return negatives


def build_item_index(data: Sequence[Dict],
                     gt_key: str = "ground_truth_item",
                     max_size: int = 20000) -> Dict[str, int]:
    """Map item title -> catalogue index, ordered by frequency.

    Without this the training pipeline passes an empty index, every item resolves
    to -1, and the id half of the hybrid representation never receives a gradient.
    """
    counts = Counter()
    for row in data:
        gt = row.get(gt_key)
        if gt and str(gt).strip():
            counts[str(gt).strip().lower()] += 1
    return {title: i for i, (title, _) in enumerate(counts.most_common(max_size))}


def popularity_weights(data: Sequence[Dict],
                       item_index: Dict[str, int],
                       debias: float = 0.5,
                       gt_key: str = "ground_truth_item") -> torch.Tensor:
    """Sampling weights over the catalogue, flattened toward uniform.

    ReDial is dominated by a handful of titles. Sampling negatives in proportion
    to raw popularity over-represents exactly those, making the contrastive task
    easier than the real one.
    """
    counts = torch.ones(len(item_index))
    for row in data:
        gt = row.get(gt_key)
        idx = item_index.get(str(gt).strip().lower()) if gt else None
        if idx is not None:
            counts[idx] += 1

    log_p = counts.log()
    log_uniform = torch.full_like(log_p, float(counts.sum().log() / len(counts)))
    mixed = (1.0 - debias) * log_p + debias * log_uniform
    return (mixed - mixed.max()).exp()


class CrossEncoderReranker(nn.Module):
    """Rerank a shortlist by scoring context and item jointly.

    A dual encoder must compress the context before it sees any item, so it cannot
    represent interactions between them. Reranking the top-k recovers that at a
    cost linear in k rather than in the catalogue.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, context_pooled: torch.Tensor,
                item_pooled: torch.Tensor) -> torch.Tensor:
        if item_pooled.dim() == 2:
            item_pooled = item_pooled.unsqueeze(1)
        k = item_pooled.size(1)
        ctx = context_pooled.unsqueeze(1).expand(-1, k, -1)
        features = torch.cat(
            [ctx, item_pooled, ctx * item_pooled, (ctx - item_pooled).abs()], dim=-1
        )
        return self.scorer(features).squeeze(-1)

    def listwise_loss(self, scores: torch.Tensor,
                      positive_index: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(scores, positive_index)
