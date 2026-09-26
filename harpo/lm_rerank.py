"""
Re-rank retriever candidates by the fine-tuned LM's likelihood of naming them.

The retriever compresses a dialogue into one vector, which is good at getting the
right movie into the top 50 (R@50 ~38%) and coarse at ordering the very top
(R@1 ~3%). The fine-tuned LM already reads the whole dialogue token by token and
was trained to write the recommended title: 85% of ReDial SFT replies name it in
the same tool call,

    <|think|>explain_choice<|/think|>
    <|tool_start|>[{"tool": "get_info", "args": {"movie": "<title>", ...

so ``log p(title | dialogue, that prefix)`` is a second, much finer-grained
score for each shortlisted candidate. No extra training is involved.

Tokenisation must reproduce training exactly. Qwen's pre-tokeniser glues a run
of punctuation into one piece, so in training text the closing quote after a
title merges with the following comma (``)",``); scoring ``title"`` alone would
ask for a token sequence the model never saw. The candidate continuation is
therefore the JSON-escaped title, its closing quote and the comma, exactly the
bytes ``json.dumps`` wrote into the training data.
"""

import json
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

# The dominant training template, up to the opening quote of the title.
DEFAULT_PREFIX = ('<|think|>explain_choice<|/think|>\n'
                  '<|tool_start|>[{"tool": "get_info", "args": {"movie": "')


def title_continuation(title: str) -> str:
    """The bytes that follow the prefix in training data for ``title``."""
    # json.dumps with ensure_ascii, as the converter wrote it; drop the opening
    # quote (it ends the prefix) and keep the closing quote plus the comma.
    return json.dumps(title)[1:] + ","


@torch.no_grad()
def continuation_logprobs(lm, prompt_ids: List[int], continuations: Sequence[List[int]],
                          device, pad_id: int = 0) -> torch.Tensor:
    """``[len(continuations)]`` total log-probability of each continuation.

    The prompt is encoded once and its KV cache shared by every continuation.
    Right padding is harmless under causal attention: real tokens never attend
    to the pads after them, and pad positions are masked out of the sum.
    """
    prompt = torch.tensor([prompt_ids], device=device)
    out = lm(input_ids=prompt, use_cache=True, logits_to_keep=1)
    past = out.past_key_values
    first_logp = F.log_softmax(out.logits[0, -1].float(), dim=-1)

    k = len(continuations)
    length = torch.tensor([len(c) for c in continuations], device=device)
    width = int(length.max())
    ids = torch.full((k, width), pad_id, dtype=torch.long, device=device)
    for i, c in enumerate(continuations):
        ids[i, :len(c)] = torch.tensor(c, device=device)

    past.batch_repeat_interleave(k)
    attn = torch.ones((k, prompt.size(1) + width), dtype=torch.long, device=device)
    logits = lm(input_ids=ids, past_key_values=past, attention_mask=attn,
                use_cache=False).logits
    logp = F.log_softmax(logits.float(), dim=-1)

    total = first_logp[ids[:, 0]]
    if width > 1:
        nxt = logp[:, :-1].gather(-1, ids[:, 1:, None]).squeeze(-1)
        live = torch.arange(1, width, device=device)[None, :] < length[:, None]
        total = total + (nxt * live).sum(dim=1)
    return total


class LikelihoodReranker:
    """Scores candidate titles as continuations of a fixed reply prefix."""

    def __init__(self, lm, tokenizer, device, prefix: str = DEFAULT_PREFIX,
                 max_prompt_tokens: int = 232):
        # 232 + a ~20-token title stays inside the 256-token training window.
        self.lm = lm
        self.tokenizer = tokenizer
        self.device = device
        self.prefix = prefix
        self.max_prompt_tokens = max_prompt_tokens
        self._cand_cache: Dict[str, List[int]] = {}
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def prompt_ids(self, context: str) -> List[int]:
        """Dialogue + reply cue + prefix, truncated from the left like training."""
        text = context + "\nAssistant: " + self.prefix
        side = self.tokenizer.truncation_side
        self.tokenizer.truncation_side = "left"
        try:
            ids = self.tokenizer(text, truncation=True, max_length=self.max_prompt_tokens,
                                 add_special_tokens=True)["input_ids"]
        finally:
            self.tokenizer.truncation_side = side
        return ids

    def candidate_ids(self, title: str) -> List[int]:
        ids = self._cand_cache.get(title)
        if ids is None:
            ids = self.tokenizer(title_continuation(title),
                                 add_special_tokens=False)["input_ids"]
            self._cand_cache[title] = ids
        return ids

    @torch.no_grad()
    def score(self, context: str, titles: Sequence[str]) -> torch.Tensor:
        """``[len(titles)]`` total log-probability of each title's continuation.

        The prompt is encoded once and its KV cache shared by every candidate.
        """
        return continuation_logprobs(self.lm, self.prompt_ids(context),
                                     [self.candidate_ids(t) for t in titles],
                                     self.device, self.pad_id)


# A dialogue-free prompt in the training input format. log p(title | this) is the
# title's prior under the model -- its popularity and length -- which contextual
# calibration subtracts so the score reflects the dialogue, not the title.
NEUTRAL_CONTEXT = "<|domain:movies|>\n\nUser: Hi! Can you recommend a movie?"


def score_many(reranker: "LikelihoodReranker", context: str, titles: Sequence[str],
               chunk: int = 256) -> torch.Tensor:
    """:meth:`LikelihoodReranker.score` over many titles, in memory-bounded chunks."""
    parts = [reranker.score(context, titles[i:i + chunk]).float()
             for i in range(0, len(titles), chunk)]
    return torch.cat(parts) if parts else torch.empty(0)


def fused_order(retriever_scores: torch.Tensor, lm_scores: torch.Tensor,
                alpha: float) -> torch.Tensor:
    """Candidate order under ``(1-alpha)*z(retriever) + alpha*z(lm)``.

    Both signals are z-normalised within the shortlist: their raw scales
    (logits over temperature vs. summed log-probabilities) are unrelated.
    """
    def z(x):
        x = x.float()
        return (x - x.mean()) / x.std().clamp(min=1e-6)
    fused = (1 - alpha) * z(retriever_scores) + alpha * z(lm_scores)
    return fused.argsort(descending=True)


def rank_after_rerank(target: int, shortlist: torch.Tensor, order: torch.Tensor,
                      retriever_rank: float) -> float:
    """1-based rank of ``target``: re-ranked inside the shortlist, unchanged outside."""
    hits = (shortlist == target).nonzero()
    if hits.numel() == 0:
        return retriever_rank  # outside the top-K, already > K
    pos = int((order == int(hits[0])).nonzero()[0])
    return float(pos + 1)
