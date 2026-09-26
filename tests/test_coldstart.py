import torch
import torch.nn.functional as F

from harpo.coldstart import ColdStart, cold_items, fit_content_to_id, training_counts


def _parts(n=40, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"content": F.normalize(torch.randn(n, d, generator=g), dim=-1),
            "id": F.normalize(torch.randn(n, d, generator=g), dim=-1),
            "bias": torch.randn(n, generator=g),
            "id_weight": 0.5,
            "profile_content": F.normalize(torch.randn(n, d, generator=g), dim=-1)}


def test_training_counts():
    rows = [{"ground_truth_item": "A"}, {"ground_truth_item": "a"},
            {"ground_truth_item": "B"}, {"ground_truth_item": "missing"}]
    assert training_counts(rows, {"a": 0, "b": 1, "c": 2}, 3).tolist() == [2, 1, 0]


def test_identity_reproduces_fusion():
    p = _parts()
    counts = torch.arange(40.0) % 4
    vecs, bias = cold_items(p, counts, ColdStart())
    assert torch.allclose(vecs, F.normalize(0.5 * p["content"] + 0.5 * p["id"], dim=-1))
    assert torch.equal(bias, p["bias"])


def test_drop_touches_only_cold_items():
    p = _parts()
    counts = torch.arange(40.0) % 4                      # 0,1,2,3,...
    vecs, bias = cold_items(p, counts, ColdStart(min_count=1, id_mode="drop", bias_q=0.5))
    base, _ = cold_items(p, counts, ColdStart())
    cold = counts < 1
    assert torch.allclose(vecs[cold], p["content"][cold])
    assert torch.equal(vecs[~cold], base[~cold])
    floor = torch.quantile(p["bias"][(counts >= 1) & (counts <= 5)], 0.5)
    assert torch.all(bias[cold] >= floor - 1e-6)
    assert torch.equal(bias[~cold], p["bias"][~cold])
    assert torch.equal(p["bias"], _parts()["bias"])     # inputs are not modified in place


def test_map_recovers_a_linear_id_space():
    g = torch.Generator().manual_seed(1)
    content = F.normalize(torch.randn(200, 6, generator=g), dim=-1)
    true_w = torch.randn(6, 6, generator=g)
    ids = content @ true_w
    w = fit_content_to_id(content, ids, torch.ones(200, dtype=torch.bool), ridge=1e-6)
    assert torch.allclose(w, true_w, atol=1e-3)


def test_profiles_change_cold_text_only():
    p = _parts()
    counts = torch.arange(40.0) % 4
    vecs, _ = cold_items(p, counts, ColdStart(min_count=2, id_mode="drop", profiles=True))
    cold = counts < 2
    assert torch.allclose(vecs[cold], p["profile_content"][cold])
