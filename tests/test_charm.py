"""Unit tests for the CHARM re-ranker (CPU, no model download)."""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from harpo.charm import (HEADS, CHARMReranker, candidate_features, listwise_loss,
                         mentioned_items)


def _inputs(b=3, k=5, h=16, e=8):
    torch.manual_seed(0)
    return dict(
        ctx_hidden=torch.randn(b, h), cand_hidden=torch.randn(b, k, h),
        cand_embed=torch.randn(b, k, e), retriever=torch.randn(b, k),
        mention=(torch.rand(b, k) > 0.7).float(), pref=torch.rand(b, k, 3),
        log_pop=torch.rand(b, k))


def test_mentioned_items_parses_linked_titles_in_order():
    index = {"inception (2010)": 4, "up (2009)": 7}
    ctx = 'User: loved "Inception (2010)"\nAssistant: try "Up (2009)" or "Inception (2010)"?'
    assert mentioned_items(ctx, index) == [4, 7]
    assert mentioned_items('User: "Unknown (1999)"', index) == []


def test_untrained_charm_reproduces_retriever_ranking():
    model = CHARMReranker(16, 8).eval()
    x = _inputs()
    out = model(**x)
    assert torch.allclose(out["score"], x["retriever"])
    assert torch.equal(out["score"].argsort(1), x["retriever"].argsort(1))


def test_every_head_receives_gradient_after_one_step():
    model = CHARMReranker(16, 8)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    x = _inputs()
    for _ in range(2):  # zero-initialised output layers need one step to open up
        loss = listwise_loss(model(**x)["score"], torch.tensor([0, 1, 2]),
                             torch.zeros(3, 5, dtype=torch.bool))
        opt.zero_grad()
        loss.backward()
        opt.step()
    out = model(**x)
    for head in HEADS:
        assert out[head].abs().sum() > 0, head


def test_listwise_loss_masks_other_positives():
    score = torch.tensor([[0.0, 5.0, 1.0]])
    target = torch.tensor([0])
    masked = listwise_loss(score, target, torch.tensor([[False, True, False]]))
    plain = F.cross_entropy(score, target)
    assert masked < plain
    assert torch.isclose(masked, F.cross_entropy(torch.tensor([[0.0, 1.0]]), target))


def test_candidate_features_exclude_the_candidate_itself():
    emb = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), dim=-1)
    top_idx = torch.tensor([[0, 1, 2]])
    mention, pref = candidate_features(top_idx, [[0]], emb)
    assert mention.tolist() == [[1.0, 0.0, 0.0]]
    # candidate 0 is the only mentioned movie: no *other* mentions -> zeros
    assert pref[0, 0, 0] == 0 and pref[0, 0, 1] == 0
    # candidate 2 is at 45 degrees to movie 0
    assert torch.isclose(pref[0, 2, 0], torch.tensor(2 ** -0.5), atol=1e-6)
    assert torch.isclose(pref[0, 1, 0], torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(pref[0, :, 2], torch.log1p(torch.tensor(1.0)))


def test_rows_without_mentions_get_zero_features():
    emb = F.normalize(torch.randn(4, 3), dim=-1)
    mention, pref = candidate_features(torch.tensor([[0, 1], [2, 3]]), [[], [1]], emb)
    assert mention[0].sum() == 0 and pref[0].abs().sum() == 0


# ---------------------------------------------------------------------------
# Cross-encoder CHARM helpers
# ---------------------------------------------------------------------------

def test_sample_group_puts_positive_first_and_excludes_turn_positives():
    import random
    from harpo.charm_ce import sample_group
    rng = random.Random(0)
    g = sample_group(5, [5, 1, 2, 3, 4, 9], exclude=[9], group=4, num_items=100, rng=rng)
    assert g[0] == 5 and len(g) == 4 and len(set(g)) == 4
    assert 9 not in g and all(c in {1, 2, 3, 4} for c in g[1:])


def test_sample_group_tops_up_with_random_items_when_shortlist_is_short():
    import random
    from harpo.charm_ce import sample_group
    g = sample_group(0, [0, 1], exclude=[], group=6, num_items=50, rng=random.Random(1))
    assert g[0] == 0 and g[1] == 1 and len(set(g)) == 6


def test_fused_ranks_weight_zero_reproduces_retriever():
    from harpo.charm_ce import fused_ranks
    retr = torch.tensor([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]])
    other = torch.tensor([[0.0, 0.0, 9.0], [9.0, 0.0, 0.0]])
    tpos = torch.tensor([2, 0])
    rank = torch.tensor([99.0, 99.0])
    assert fused_ranks(retr, other, 0.0, tpos, rank).tolist() == [3.0, 3.0]
    assert fused_ranks(retr, other, 1.0, tpos, rank).tolist() == [1.0, 1.0]
    # target outside the shortlist keeps its retriever rank
    assert fused_ranks(retr, other, 0.5, torch.tensor([-1, 0]), torch.tensor([77.0, 5.0]))[0] == 77.0
