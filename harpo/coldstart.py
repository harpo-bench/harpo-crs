"""
Cold-start item vectors for the two-tower retriever, applied at inference.

The catalogue softmax gives every item an id vector and a popularity bias. An
item that is never a training target only ever appears as a negative, so both
are pushed away from every context and the item cannot be recommended: none of
the ReDial test cases whose answer was never recommended in training (8.8%)
reach the 7B retriever's top-100.

For items recommended fewer than ``min_count`` times in training the vector is
rebuilt from what can be trusted -- the text tower -- and the bias is lifted to
a floor taken from rarely recommended items:

- ``id_mode="drop"``: the text-tower vector alone;
- ``id_mode="map"``: the id part predicted from the text vector by a ridge
  regression fitted on frequently recommended items (content -> id space);
- ``bias_q``: cold biases are raised to the q-quantile of the bias of items
  recommended 1-5 times (None keeps the trained bias);
- ``profiles``: cold items' text is "title: BRIDGE profile" instead of the title.
"""

from dataclasses import asdict, dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ColdStart:
    min_count: int = 1
    id_mode: str = "keep"          # "keep" | "drop" | "map"
    bias_q: Optional[float] = None
    profiles: bool = False

    def is_identity(self) -> bool:
        return self.id_mode == "keep" and self.bias_q is None and not self.profiles

    def to_dict(self) -> Dict:
        return asdict(self)


def training_counts(rows, index: Dict[str, int], num_items: int) -> torch.Tensor:
    """How often each catalogue item is a target in ``rows`` (the backbone's training rows)."""
    counts = torch.zeros(num_items)
    for r in rows:
        i = index.get(str(r["ground_truth_item"]).lower())
        if i is not None:
            counts[i] += 1
    return counts


def fit_content_to_id(content: torch.Tensor, id_embeds: torch.Tensor, mask: torch.Tensor,
                      ridge: float = 1.0) -> torch.Tensor:
    """Ridge map ``W`` with ``content @ W ~ id_embeds``, fitted on the ``mask`` items."""
    x, y = content[mask].double(), id_embeds[mask].double()
    eye = torch.eye(x.size(1), dtype=x.dtype, device=x.device)
    return torch.linalg.solve(x.T @ x + ridge * eye, x.T @ y).to(content.dtype)


def cold_items(parts: Dict, counts: torch.Tensor, cfg: ColdStart, warm_min: int = 10):
    """``(vectors [N, d], bias [N])`` with the cold items rebuilt per ``cfg``.

    ``parts`` holds the catalogue as the retriever scores it: ``content`` (text
    tower, L2-normalised), ``id`` (id table, L2-normalised), ``bias`` and
    ``id_weight``; ``profile_content`` is needed only when ``cfg.profiles``.
    The identity config reproduces the trained scoring exactly.
    """
    content, ids, bias, w = parts["content"], parts["id"], parts["bias"], parts["id_weight"]
    vecs = F.normalize((1.0 - w) * content + w * ids, dim=-1)
    counts = counts.to(content.device)
    cold = counts < cfg.min_count
    if cfg.is_identity() or not bool(cold.any()):
        return vecs, bias
    text = parts["profile_content"] if cfg.profiles else content
    if cfg.id_mode == "keep":
        new = (1.0 - w) * text[cold] + w * ids[cold]
    elif cfg.id_mode == "drop":
        new = text[cold]
    elif cfg.id_mode == "map":
        W = fit_content_to_id(content, ids, counts >= warm_min)
        new = (1.0 - w) * text[cold] + w * F.normalize(text[cold] @ W, dim=-1)
    else:
        raise ValueError(f"unknown id_mode {cfg.id_mode!r}")
    vecs = vecs.clone()
    vecs[cold] = F.normalize(new, dim=-1)
    if cfg.bias_q is not None:
        ref = bias[(counts >= 1) & (counts <= 5)].float()
        floor = torch.quantile(ref, cfg.bias_q).to(bias.dtype)
        bias = bias.clone()
        bias[cold] = torch.maximum(bias[cold], floor)
    return vecs, bias
