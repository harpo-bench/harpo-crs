"""
STAR: reason about the user, then choose among the shortlist -- over several
branches of thought.

Each branch shows the LLM the dialogue and the top candidates (labelled A-T,
each with its BRIDGE knowledge profile), lets it state in one sentence what the
user wants, and then reads its choice as a distribution over the labels. Scores
come from the choice distribution, not from free text, so every candidate gets a
graded score and nothing depends on parsing.

Branches differ in candidate order (retriever order, reversed, shuffled) and in
the reasoning (greedy for the first, sampled for the rest). Averaging them is
the "tree" in STAR: it marginalises over thoughts and cancels the LLM's
well-known bias toward whatever is listed first.
"""

import random
import re
from typing import List, Sequence, Tuple

import torch

LABELS = [chr(ord("A") + i) for i in range(20)]
SYSTEM = "You are an expert movie recommender talking with a user."
ANSWER_CUE = "\nBest recommendation:"
_DOMAIN_TAG = re.compile(r"<\|domain:[a-z]+\|>")


def clean_dialogue(text: str, max_chars: int = 2000) -> str:
    """Drop training-format tags and keep the most recent turns."""
    text = _DOMAIN_TAG.sub("", text).strip()
    if len(text) > max_chars:
        text = text[-max_chars:]
        cut = text.find("\n")
        if 0 <= cut < 200:
            text = text[cut + 1:]
    return text


def build_messages(dialogue: str, candidates: Sequence[Tuple[str, str]], direct: bool = False):
    """Chat messages asking for a label, with one sentence of reasoning first
    unless ``direct`` (the trained selector answers straight away)."""
    lines = [f"{LABELS[i]}. {title}" + (f" -- {profile}" if profile else "")
             for i, (title, profile) in enumerate(candidates)]
    ask = ("Give the letter of the single best movie to recommend next." if direct else
           "In one sentence, say what the user is looking for right now. Then give the "
           "letter of the single best movie to recommend next, as: Best recommendation: <letter>")
    user = (f"Conversation so far:\n{dialogue}\n\n"
            f"Candidate movies:\n" + "\n".join(lines) + "\n\n" + ask)
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def direct_prompt(tok, dialogue: str, candidates: Sequence[Tuple[str, str]]) -> str:
    """Prompt whose next token is the chosen label: the same text in training
    and scoring, so the trained selector is read exactly where it learned."""
    return tok.apply_chat_template(build_messages(clean_dialogue(dialogue), candidates, direct=True),
                                   add_generation_prompt=True, tokenize=False) + "Best recommendation:"


def branch_order(k: int, branch: int, key: str) -> List[int]:
    """Candidate order for a branch: retriever, reversed, then seeded shuffles."""
    order = list(range(k))
    if branch == 1:
        order.reverse()
    elif branch >= 2:
        random.Random(f"{key}:{branch}").shuffle(order)
    return order


def trim_reasoning(text: str) -> str:
    """Keep the model's reasoning, dropping any answer it already started."""
    text = text.split("Best recommendation")[0]
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return " ".join(lines[:2])


def aggregate(branch_logps: torch.Tensor) -> torch.Tensor:
    """``[B, K] -> [K]``: log of the mean choice probability across branches."""
    return torch.logsumexp(branch_logps, 0) - torch.log(torch.tensor(float(branch_logps.size(0))))
