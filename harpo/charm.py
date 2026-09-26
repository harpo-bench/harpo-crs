"""
CHARM as a re-ranker: a hierarchical reward over (dialogue, candidate) pairs.

The retriever compresses a dialogue into one vector and ranks the catalogue by a
dot product. That gets the right movie into the top 50 about 40% of the time
but orders the very top poorly (R@1 ~3%), and zero-shot LM likelihood does not
fix it: on ReDial it mostly re-recommends titles already named in the dialogue.

CHARM re-scores the retriever's shortlist with four reward heads, each looking
at information the dot product cannot use:

  relevance   a learned interaction between the dialogue and the candidate
              ([c, i, c*i, |c-i|] -> MLP), far richer than one inner product
  novelty     whether the candidate was already mentioned, with a weight the
              dialogue decides -- some users ask for a repeat, most want new
  preference  similarity of the candidate to the movies the user mentioned
              (excluding itself, so it does not duplicate novelty)
  popularity  training-set popularity, again with a dialogue-dependent weight

A meta-learner weights the heads per dialogue, and the result is added to the
retriever score. Every output layer starts at zero, so an untrained CHARM
reproduces the retriever's ranking exactly and can only depart from it where
training data says it should.

Trained listwise against the retriever's own top-K -- hard negatives, not the
random movies the original preference pairs used.
"""

import re
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

HEADS = ("relevance", "novelty", "preference", "popularity")
PREF_FEATURES = 3  # max similarity, mean similarity, log(1 + #mentioned)

# The converter replaces ReDial's @movieId links with "Title (Year)".
_QUOTED = re.compile(r'"([^"]+)"')


def mentioned_items(context: str, title_index: Dict[str, int]) -> List[int]:
    """Catalogue indices of the movies linked in a dialogue, in order of mention."""
    out: List[int] = []
    for title in _QUOTED.findall(context):
        idx = title_index.get(title.strip().lower())
        if idx is not None and idx not in out:
            out.append(idx)
    return out


@torch.no_grad()
def candidate_features(top_idx: torch.Tensor, mentioned: Sequence[Sequence[int]],
                       item_embeddings: torch.Tensor, chunk: int = 2048):
    """Per-candidate novelty flag and preference-similarity features.

    Args:
        top_idx: ``[N, K]`` catalogue indices of each row's shortlist.
        mentioned: per row, the catalogue indices linked in its dialogue.
        item_embeddings: ``[C, E]`` L2-normalised retriever item embeddings.

    Returns:
        ``mention [N, K]`` (1.0 if the candidate is in the dialogue) and
        ``pref [N, K, 3]``. Similarities exclude the candidate itself.
    """
    device = top_idx.device
    n, k = top_idx.shape
    width = max([len(m) for m in mentioned] + [1])
    ment = torch.full((n, width), -1, dtype=torch.long, device=device)
    for r, m in enumerate(mentioned):
        if m:
            ment[r, :len(m)] = torch.tensor(list(m), device=device)

    mention = torch.zeros(n, k, device=device)
    pref = torch.zeros(n, k, PREF_FEATURES, device=device)
    emb = item_embeddings.float()
    for s in range(0, n, chunk):
        ti, mi = top_idx[s:s + chunk], ment[s:s + chunk]
        valid = mi >= 0                                              # [b, M]
        same = ti[:, :, None] == mi[:, None, :]                     # [b, K, M]
        mention[s:s + chunk] = same.any(-1).float()
        sims = torch.einsum("bke,bme->bkm", emb[ti], emb[mi.clamp(min=0)])
        others = valid[:, None, :] & ~same                          # exclude itself
        count = others.sum(-1)
        has = count > 0
        mx = sims.masked_fill(~others, -1.0).amax(-1)
        mean = (sims * others).sum(-1) / count.clamp(min=1)
        pref[s:s + chunk, :, 0] = torch.where(has, mx, torch.zeros_like(mx))
        pref[s:s + chunk, :, 1] = torch.where(has, mean, torch.zeros_like(mean))
        pref[s:s + chunk, :, 2] = torch.log1p(valid.sum(-1).float())[:, None]
    return mention, pref


class CHARMReranker(nn.Module):
    """Hierarchical reward heads over a retriever shortlist."""

    def __init__(self, hidden_size: int, item_dim: int, proj_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.ctx_proj = nn.Sequential(
            nn.Linear(hidden_size, proj_dim), nn.LayerNorm(proj_dim), nn.GELU(),
            nn.Dropout(dropout))
        self.item_proj = nn.Sequential(
            nn.Linear(hidden_size + item_dim, proj_dim), nn.LayerNorm(proj_dim), nn.GELU(),
            nn.Dropout(dropout))
        self.relevance = nn.Sequential(
            nn.Linear(4 * proj_dim, proj_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(proj_dim, 1))
        # The dialogue sets the sign and strength of these two signals.
        self.novelty = nn.Linear(proj_dim, 1)
        self.popularity = nn.Linear(proj_dim, 1)
        self.preference = nn.Sequential(
            nn.Linear(PREF_FEATURES + proj_dim, 64), nn.GELU(), nn.Linear(64, 1))
        self.meta = nn.Linear(proj_dim, len(HEADS))
        self.retriever_scale = nn.Parameter(torch.tensor(1.0))

        # Untrained CHARM == the retriever's ranking.
        for layer in (self.relevance[-1], self.novelty, self.popularity,
                      self.preference[-1], self.meta):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, ctx_hidden: torch.Tensor, cand_hidden: torch.Tensor,
                cand_embed: torch.Tensor, retriever: torch.Tensor,
                mention: torch.Tensor, pref: torch.Tensor,
                log_pop: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Scores for ``[B, K]`` candidates.

        Args:
            ctx_hidden: ``[B, H]`` pooled dialogue states.
            cand_hidden: ``[B, K, H]`` pooled title states.
            cand_embed: ``[B, K, E]`` retriever item embeddings.
            retriever: ``[B, K]`` retriever logits.
            mention: ``[B, K]`` already-mentioned flags.
            pref: ``[B, K, 3]`` preference-similarity features.
            log_pop: ``[B, K]`` log(1 + training frequency).
        """
        dtype = self.retriever_scale.dtype
        c = self.ctx_proj(ctx_hidden.to(dtype))                             # [B, P]
        i = self.item_proj(torch.cat([cand_hidden.to(dtype), cand_embed.to(dtype)], -1))
        cc = c[:, None, :].expand_as(i)

        heads = {
            "relevance": self.relevance(torch.cat([cc, i, cc * i, (cc - i).abs()], -1)).squeeze(-1),
            "novelty": self.novelty(c) * mention.to(dtype),
            "preference": self.preference(torch.cat([pref.to(dtype), cc], -1)).squeeze(-1),
            "popularity": self.popularity(c) * log_pop.to(dtype),
        }
        weights = torch.softmax(self.meta(c), -1) * len(HEADS)            # mean weight 1
        stacked = torch.stack([heads[h] for h in HEADS], -1)               # [B, K, 4]
        reward = (stacked * weights[:, None, :]).sum(-1)
        score = self.retriever_scale * retriever.to(dtype) + reward
        return {"score": score, "reward": reward, "weights": weights, **heads}


def listwise_loss(score: torch.Tensor, target: torch.Tensor,
                  other_positives: torch.Tensor) -> torch.Tensor:
    """Softmax cross-entropy over the shortlist.

    ``other_positives`` masks the *other* movies recommended in the same turn:
    they are correct too, so they must not act as negatives.
    """
    score = score.masked_fill(other_positives, torch.finfo(score.dtype).min)
    return F.cross_entropy(score, target)
