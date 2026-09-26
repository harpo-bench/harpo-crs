"""Unit tests for MAVEN consensus (CPU)."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from harpo.maven import (MAVENConsensus, agreement_features, fit, ranks_from_scores,
                         zrow)


def test_agreement_features_shape_and_values():
    z = zrow(torch.randn(4, 6, 3))
    f = agreement_features(z)
    assert f.shape == (4, 3 + 3)
    same = zrow(torch.randn(2, 5, 1)).repeat(1, 1, 2)       # two identical agents
    assert torch.allclose(agreement_features(same)[:, 0], torch.ones(2), atol=1e-5)


def test_untrained_maven_is_the_equal_weight_average():
    z = zrow(torch.randn(3, 5, 3))
    model = MAVENConsensus(3, agreement_features(z).size(1))
    fused, w = model(z, agreement_features(z))
    assert torch.allclose(w, torch.full_like(w, 1 / 3))
    assert torch.allclose(fused, z.mean(-1))


def test_fit_learns_to_trust_the_informative_agent():
    torch.manual_seed(0)
    n, k = 400, 10
    target = torch.randint(0, k, (n,))
    good = torch.randn(n, k)
    good[torch.arange(n), target] += 3.0                    # agent 0 knows the answer
    z = zrow(torch.stack([good, torch.randn(n, k), torch.randn(n, k)], -1))
    feats = agreement_features(z)
    model = MAVENConsensus(3, feats.size(1), gated=False)
    stats = fit(model, z, feats, target, epochs=200)
    assert stats["last_loss"] < stats["first_loss"]
    w = model.weights(feats)[0]
    assert int(w.argmax()) == 0


def test_ranks_keep_retriever_rank_outside_the_shortlist():
    fused = torch.tensor([[0.1, 0.9, 0.5]])
    assert ranks_from_scores(fused, torch.tensor([1]), torch.tensor([40.0])).item() == 1.0
    assert ranks_from_scores(fused, torch.tensor([-1]), torch.tensor([40.0])).item() == 40.0
