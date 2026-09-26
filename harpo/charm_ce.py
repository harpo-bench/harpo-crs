"""
CHARM relevance as a cross-encoder: the LLM reads the dialogue and one candidate
together and outputs a relevance score.

Why a cross-encoder. The retriever squeezes the dialogue and each title into one
vector apiece before comparing them, so it cannot check a candidate against the
specifics of the request. A feature-level re-ranker on those same frozen vectors
found nothing new on ReDial (test R@10 18.4 -> 18.7, all of it from boosting
already-mentioned titles). A cross-encoder attends from every candidate token to
every dialogue token -- the information the two-tower model throws away.

Two design choices follow from what went wrong before:

  * It starts from the plain instruct model with a fresh LoRA adapter, not from
    the SFT backbone. The backbone has memorised the training answers (its reply
    names the target in every ReDial example), and a re-ranker built on it learns
    patterns that exist only for seen conversations.
  * It never sees the retriever's score, so inserting a missed positive into a
    training group is harmless: there is no "lowest-scored item is the answer"
    artefact to learn.

At evaluation its score is fused with the retriever's, with the weight chosen on
validation (weight 0 -- the retriever alone -- is always a candidate).
"""

import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import torch

SUFFIX = '\nAssistant: How about "{title}"?'
# Token budget reserved for the candidate suffix. Fixing it (rather than sizing
# the dialogue to each title) makes the dialogue prefix identical for every
# candidate, which is what lets them share one encoding of it.
CAND_BUDGET = 32


def sample_group(positive: int, candidates: Sequence[int], exclude: Sequence[int],
                 group: int, num_items: int, rng: random.Random) -> List[int]:
    """``[positive] + (group - 1)`` distinct negatives, hard ones first.

    Negatives come from the retriever's shortlist minus every positive of the
    turn; if the shortlist runs short, random catalogue items fill the rest.
    """
    banned = set(exclude) | {positive}
    hard = [c for c in dict.fromkeys(candidates) if c not in banned]
    need = group - 1
    negs = rng.sample(hard, need) if len(hard) >= need else list(hard)
    taken = banned | set(negs)
    while len(negs) < need:
        c = rng.randrange(num_items)
        if c not in taken:
            negs.append(c)
            taken.add(c)
    return [positive] + negs


def fused_ranks(retriever: torch.Tensor, other: torch.Tensor, weight: float,
                target_pos: torch.Tensor, retriever_rank: torch.Tensor) -> torch.Tensor:
    """1-based target ranks under ``(1-w) z(retriever) + w z(other)`` within each list.

    Rows whose target is outside the shortlist keep their retriever rank (> K).
    """
    def z(x):
        x = x.float()
        return (x - x.mean(1, keepdim=True)) / x.std(1, keepdim=True).clamp(min=1e-6)
    fused = (1 - weight) * z(retriever) + weight * z(other)
    t = fused.gather(1, target_pos.clamp(min=0)[:, None])
    r = (fused > t).sum(1) + 1 + ((fused == t).sum(1) - 1) / 2
    return torch.where(target_pos >= 0, r.float(), retriever_rank.float())


class CrossEncoderCHARM:
    """LLM + LoRA + scalar head scoring (dialogue, candidate) pairs."""

    def __init__(self, model_path: str, device: str, lora_r: int = 16, lora_alpha: int = 32,
                 max_length: int = 256, dtype: torch.dtype = torch.float32):
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        local = os.path.isdir(model_path)
        self.tok = AutoTokenizer.from_pretrained(model_path, local_files_only=local)
        # Left padding and truncation: the candidate sits at the end of every
        # sequence and the oldest turns are the ones dropped.
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        base = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=1, dtype=dtype, local_files_only=local)
        base.config.pad_token_id = self.tok.pad_token_id
        cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=lora_r, lora_alpha=lora_alpha,
                         lora_dropout=0.05,
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                         "gate_proj", "up_proj", "down_proj"])
        self.model = get_peft_model(base, cfg).to(device)
        # A bf16 backbone would leave the freshly initialised score head in bf16,
        # where AdamW steps of ~lr fall below its resolution. Keep the head in
        # float32; autocast handles the mixed-precision matmul.
        for p in self.head_parameters():
            p.data = p.data.float()
        self.device = device
        self.max_length = max_length
        self._ctx: Dict[str, List[int]] = {}
        self._cand: Dict[str, List[int]] = {}

    def head_parameters(self):
        return [p for n, p in self.model.named_parameters() if p.requires_grad and "score" in n]

    def adapter_parameters(self):
        return [p for n, p in self.model.named_parameters() if p.requires_grad and "score" not in n]

    def _prefix(self, dialogue: str) -> List[int]:
        ctx = self._ctx.get(dialogue)
        if ctx is None:
            ids = self.tok(dialogue, add_special_tokens=False)["input_ids"]
            ctx = ids[-max(self.max_length - CAND_BUDGET, 1):]  # keep the latest turns
            self._ctx[dialogue] = ctx
        return ctx

    def _suffix(self, title: str) -> List[int]:
        cand = self._cand.get(title)
        if cand is None:
            cand = self.tok(SUFFIX.format(title=title), add_special_tokens=False)["input_ids"]
            self._cand[title] = cand
        return cand

    def _ids(self, dialogue: str, title: str) -> List[int]:
        return self._prefix(dialogue) + self._suffix(title)

    def _batch(self, pairs: Sequence[Tuple[str, str]]):
        seqs = [self._ids(d, t) for d, t in pairs]
        width = max(len(s) for s in seqs)
        pad = self.tok.pad_token_id
        ids = torch.full((len(seqs), width), pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, width - len(s):] = torch.tensor(s)
            mask[i, width - len(s):] = 1
        return ids.to(self.device), mask.to(self.device)

    def _autocast(self):
        if str(self.device).startswith("cuda"):
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return torch.autocast("cpu", enabled=False)

    def scores(self, pairs: Sequence[Tuple[str, str]]) -> torch.Tensor:
        """Differentiable ``[len(pairs)]`` scores (training)."""
        ids, mask = self._batch(pairs)
        with self._autocast():
            return self.model(input_ids=ids, attention_mask=mask).logits.squeeze(-1).float()

    @torch.no_grad()
    def score_many(self, pairs: Sequence[Tuple[str, str]], batch_size: int = 256) -> torch.Tensor:
        was = self.model.training
        self.model.eval()
        try:
            out = [self.scores(pairs[i:i + batch_size]) for i in range(0, len(pairs), batch_size)]
        finally:
            self.model.train(was)
        return torch.cat(out) if out else torch.empty(0)

    def grouped_scores(self, dialogues: Sequence[str],
                       groups: Sequence[Sequence[str]]) -> torch.Tensor:
        """``[B, G]`` scores for ``G`` candidates per dialogue, sharing the dialogue.

        Each dialogue is encoded once and its KV cache is shared by all its
        candidates, so a group costs one dialogue plus G short suffixes instead
        of G full sequences -- identical scores, ~G x less prefix compute, and
        differentiable (the cache tensors carry gradients back to the prefix).
        """
        g = len(groups[0])
        assert all(len(x) == g for x in groups), "groups must have equal size"
        pad = self.tok.pad_token_id

        prefixes = [self._prefix(d) for d in dialogues]
        width = max(len(p) for p in prefixes)
        p_ids = torch.full((len(prefixes), width), pad, dtype=torch.long)
        p_mask = torch.zeros((len(prefixes), width), dtype=torch.long)
        for i, p in enumerate(prefixes):          # left-pad: dialogue ends at the join
            p_ids[i, width - len(p):] = torch.tensor(p)
            p_mask[i, width - len(p):] = 1
        p_ids, p_mask = p_ids.to(self.device), p_mask.to(self.device)

        suffixes = [self._suffix(t) for grp in groups for t in grp]
        s_width = max(len(x) for x in suffixes)
        s_ids = torch.full((len(suffixes), s_width), pad, dtype=torch.long)
        s_mask = torch.zeros((len(suffixes), s_width), dtype=torch.long)
        for i, x in enumerate(suffixes):          # right-pad: pads follow the candidate
            s_ids[i, :len(x)] = torch.tensor(x)
            s_mask[i, :len(x)] = 1
        s_ids, s_mask = s_ids.to(self.device), s_mask.to(self.device)

        with self._autocast():
            past = self.model(input_ids=p_ids, attention_mask=p_mask,
                              use_cache=True).past_key_values
            past.batch_repeat_interleave(g)
            attn = torch.cat([p_mask.repeat_interleave(g, dim=0), s_mask], dim=1)
            # Sequence-classification pooling reads the last non-pad token of the
            # *suffix*, i.e. the end of each candidate.
            logits = self.model(input_ids=s_ids, attention_mask=attn,
                                past_key_values=past, use_cache=False).logits
        return logits.view(len(dialogues), g).float()

    @torch.no_grad()
    def score_groups(self, dialogues: Sequence[str], groups: Sequence[Sequence[str]],
                     dialogues_per_batch: int = 16) -> torch.Tensor:
        """Evaluation-mode :meth:`grouped_scores` over many dialogues."""
        was = self.model.training
        self.model.eval()
        try:
            out = [self.grouped_scores(dialogues[i:i + dialogues_per_batch],
                                       groups[i:i + dialogues_per_batch])
                   for i in range(0, len(dialogues), dialogues_per_batch)]
        finally:
            self.model.train(was)
        return torch.cat(out) if out else torch.empty(0, 0)

    def load_adapter(self, path: str) -> None:
        """Load a trained adapter + score head saved with ``save_pretrained``."""
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        weights = load_file(os.path.join(path, "adapter_model.safetensors"))
        result = set_peft_model_state_dict(self.model, weights)
        missing = [k for k in getattr(result, "unexpected_keys", []) or []]
        if missing:
            raise ValueError(f"adapter keys not in the model: {missing[:5]}")
        for p in self.head_parameters():
            p.data = p.data.float()

    def trainable_state(self) -> Dict[str, torch.Tensor]:
        return {n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad}

    def load_trainable_state(self, state: Dict[str, torch.Tensor]) -> None:
        params = dict(self.model.named_parameters())
        with torch.no_grad():
            for n, v in state.items():
                params[n].copy_(v)
