"""
Sequence pooling for HARPO.

Every downstream module (BRIDGE, CHARM, STAR, MAVEN, the ranking head) consumes a
fixed-size summary of the encoder's last hidden state. The original implementation
used a bare ``hidden_states[-1].mean(dim=1)``, which averages over padding as well
as content. With ``padding_side="left"`` and ``padding="max_length"`` that means a
short sequence is mostly padding, so the pooled vector is dominated by pad
embeddings and its scale varies with sequence length.

That is not cosmetic for preference learning: a chosen/rejected pair almost never
has the same length, so a reward model fed unmasked means can separate the two by
padding fraction alone, without reading either response.
"""

from typing import Optional

import torch
import torch.nn as nn


def masked_mean_pool(hidden_states: torch.Tensor,
                     attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mean over the sequence axis, ignoring padded positions.

    Args:
        hidden_states: ``[batch, seq, hidden]``, or ``[batch, hidden]`` (returned
            unchanged, so callers can stay agnostic about what they were handed).
        attention_mask: ``[batch, seq]`` with 1 for real tokens. When ``None`` this
            degrades to an unmasked mean -- only correct if nothing is padded.

    Returns:
        ``[batch, hidden]``
    """
    if hidden_states.dim() == 2:
        return hidden_states
    if hidden_states.dim() != 3:
        raise ValueError(f"expected [batch, seq, hidden], got {tuple(hidden_states.shape)}")

    if attention_mask is None:
        return hidden_states.mean(dim=1)

    if attention_mask.dim() != 2 or attention_mask.shape[:2] != hidden_states.shape[:2]:
        raise ValueError(
            f"attention_mask {tuple(attention_mask.shape)} does not match "
            f"hidden_states {tuple(hidden_states.shape[:2])}"
        )

    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    # An all-padding row would divide by zero; clamp so it yields zeros not NaN.
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def last_token_pool(hidden_states: torch.Tensor,
                    attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Take the final real token of each sequence.

    For a causal LM this is the only position that has attended to the whole
    sequence, which makes it the natural summary for scoring. Handles left- and
    right-padding, since it indexes the last position where the mask is 1.
    """
    if hidden_states.dim() == 2:
        return hidden_states
    if attention_mask is None:
        return hidden_states[:, -1, :]

    idx = attention_mask.to(torch.int64).cumsum(dim=1).argmax(dim=1)
    batch = torch.arange(hidden_states.size(0), device=hidden_states.device)
    return hidden_states[batch, idx, :]


class AttentionPool(nn.Module):
    """Single-query attention pooling over the unpadded sequence.

    A learned query attends over token states, so the summary can concentrate on
    the positions that matter (a mentioned title, a rejection) instead of averaging
    them away. Padded positions are masked before the softmax, so they receive
    exactly zero weight regardless of what the encoder put there.
    """

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size {hidden_size} not divisible by num_heads {num_heads}")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.query = nn.Parameter(torch.randn(1, 1, hidden_size) * hidden_size ** -0.5)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``[batch, seq, hidden]`` -> ``[batch, hidden]``."""
        if hidden_states.dim() == 2:
            return hidden_states

        b, s, _ = hidden_states.shape
        q = self.query.expand(b, -1, -1)

        def split(x, length):
            return x.view(b, length, self.num_heads, self.head_dim).transpose(1, 2)

        qh = split(q, 1)
        kh = split(self.k_proj(hidden_states), s)
        vh = split(self.v_proj(hidden_states), s)

        scores = (qh @ kh.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if attention_mask is not None:
            pad = (attention_mask == 0).view(b, 1, 1, s)
            # finfo.min rather than -inf: an all-padding row would softmax to NaN.
            scores = scores.masked_fill(pad, torch.finfo(scores.dtype).min)

        weights = self.dropout(scores.softmax(dim=-1))
        pooled = (weights @ vh).transpose(1, 2).reshape(b, self.hidden_size)
        return self.norm(self.out_proj(pooled))
