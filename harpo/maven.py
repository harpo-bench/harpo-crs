"""
MAVEN: per-dialogue consensus among recommendation agents.

Several agents score the same retriever shortlist, each from different
information:

  recommender  the two-tower retriever (collaborative + popularity signal)
  explainer    the LM's likelihood of naming the title in its reply
  critic       the CHARM cross-encoder, reading dialogue and candidate together

No single agent is best everywhere -- on ReDial the LM is strong when the
dialogue names concrete preferences and the retriever when it is vague -- so a
fixed blend leaves accuracy on the table. MAVEN decides, per dialogue, how much
to trust each agent from how they behave on *that* dialogue: how far they agree
with each other (the cosine agreement of the paper's Eq. 16) and how confident
each is (the margin between its top two candidates).

The model is deliberately tiny (a linear gate over those statistics), so it can
be fitted on a few thousand held-out cases without overfitting, and it starts as
an equal-weight average of the agents.
"""

import itertools
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def zrow(x: torch.Tensor) -> torch.Tensor:
    """Standardise each row (over its K candidates)."""
    x = x.float()
    return (x - x.mean(1, keepdim=True)) / x.std(1, keepdim=True).clamp(min=1e-6)


def agreement_features(z: torch.Tensor) -> torch.Tensor:
    """Per-dialogue statistics of how the agents behave.

    Args:
        z: ``[N, K, A]`` standardised agent scores over each shortlist.

    Returns:
        ``[N, A*(A-1)/2 + A]``: pairwise cosine agreement between agents, then
        each agent's top-1 vs top-2 margin (its confidence).
    """
    n, k, a = z.shape
    unit = F.normalize(z, dim=1)
    pairs = [(unit[:, :, i] * unit[:, :, j]).sum(1)
             for i, j in itertools.combinations(range(a), 2)]
    top2 = z.topk(2, dim=1).values                                   # [N, 2, A]
    margins = top2[:, 0, :] - top2[:, 1, :]
    return torch.cat([torch.stack(pairs, 1) if pairs else z.new_zeros(n, 0), margins], 1)


class MAVENConsensus(nn.Module):
    """Per-dialogue softmax weights over agents, from agreement features."""

    def __init__(self, num_agents: int, num_features: int, gated: bool = True):
        super().__init__()
        self.gated = gated
        self.prior = nn.Parameter(torch.zeros(num_agents))      # global trust
        self.gate = nn.Linear(num_features, num_agents)          # per-dialogue shift
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("feat_mean", torch.zeros(num_features))
        self.register_buffer("feat_std", torch.ones(num_features))

    def set_feature_stats(self, feats: torch.Tensor) -> None:
        self.feat_mean = feats.mean(0)
        self.feat_std = feats.std(0).clamp(min=1e-6)

    def weights(self, feats: torch.Tensor) -> torch.Tensor:
        logits = self.prior.expand(feats.size(0), -1)
        if self.gated:
            logits = logits + self.gate((feats - self.feat_mean) / self.feat_std)
        return torch.softmax(logits, -1)

    def forward(self, z: torch.Tensor, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        w = self.weights(feats)
        return self.scale * (z * w[:, None, :]).sum(-1), w


def fit(model: MAVENConsensus, z: torch.Tensor, feats: torch.Tensor,
        target_pos: torch.Tensor, epochs: int = 300, lr: float = 0.05,
        weight_decay: float = 1e-3) -> Dict[str, float]:
    """Full-batch listwise training on the rows whose target is in the shortlist."""
    keep = target_pos >= 0
    z, feats, tp = z[keep], feats[keep], target_pos[keep]
    model.set_feature_stats(feats)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    first = last = None
    for _ in range(epochs):
        fused, _ = model(z, feats)
        loss = F.cross_entropy(fused, tp)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = float(loss) if first is None else first
        last = float(loss)
    return {"first_loss": first, "last_loss": last, "rows": int(keep.sum())}


def ranks_from_scores(fused: torch.Tensor, target_pos: torch.Tensor,
                      retriever_rank: torch.Tensor) -> torch.Tensor:
    """1-based target rank within the shortlist; retriever rank outside it."""
    t = fused.gather(1, target_pos.clamp(min=0)[:, None])
    r = (fused > t).sum(1) + 1 + ((fused == t).sum(1) - 1) / 2
    return torch.where(target_pos >= 0, r.float(), retriever_rank.float())
