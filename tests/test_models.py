import torch

from wtpf.config import load_config
from wtpf.models.factory import MODEL_REGISTRY, build_model

# Exact parameter counts reported in Table 6 of the paper.
PAPER_PARAM_COUNTS = {
    "cnn": 18_433,
    "lstm": 202_881,
    "bilstm": 536_833,
    "cnn_bilstm": 608_385,
    "attention_bilstm": 537_090,
    "cnn_attention_bilstm": 624_898,
}


def _default_config():
    return load_config("configs/config.yaml")


def test_every_registered_model_builds_and_forward_passes():
    cfg = _default_config()
    batch, T, F = 4, cfg.model.sequence_length, cfg.model.input_features
    x = torch.randn(batch, T, F)

    for name in MODEL_REGISTRY:
        model = build_model(name, cfg.model)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (batch,), f"{name} produced unexpected output shape {out.shape}"


def test_model_parameter_counts_match_paper_table6():
    cfg = _default_config()
    for name, expected in PAPER_PARAM_COUNTS.items():
        model = build_model(name, cfg.model)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params == expected, f"{name}: expected {expected:,} params, got {n_params:,}"


def test_attention_bilstm_returns_valid_attention_weights():
    cfg = _default_config()
    model = build_model("attention_bilstm", cfg.model)
    model.eval()
    x = torch.randn(2, cfg.model.sequence_length, cfg.model.input_features)
    with torch.no_grad():
        out, weights = model(x, return_attention=True)

    assert out.shape == (2,)
    assert weights.shape == (2, cfg.model.sequence_length)
    # softmax weights must sum to 1 across the time axis
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2), atol=1e-5)
    assert (weights >= 0).all()


def test_cnn_attention_bilstm_attention_weights_shape():
    cfg = _default_config()
    model = build_model("cnn_attention_bilstm", cfg.model)
    model.eval()
    x = torch.randn(3, cfg.model.sequence_length, cfg.model.input_features)
    with torch.no_grad():
        out, weights = model(x, return_attention=True)
    assert out.shape == (3,)
    assert weights.shape == (3, cfg.model.sequence_length)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(3), atol=1e-5)


def test_ablation_variants_build_and_run():
    cfg = _default_config()
    x = torch.randn(2, cfg.model.sequence_length, cfg.model.input_features)
    for name in ("no_cnn", "no_attention", "uni_lstm", "no_soft_pool"):
        model = build_model(name, cfg.model)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2,)


def test_unknown_model_name_raises():
    cfg = _default_config()
    import pytest

    with pytest.raises(ValueError):
        build_model("not_a_real_model", cfg.model)
