"""
Model factory.

Single entry point that builds any of the six comparative architectures
(Sec. III-D / III-E) or the four ablation variants (Sec. IV-I) from the
project's `config.yaml`, so scripts never need to import model classes
directly.
"""

from __future__ import annotations

import torch.nn as nn

from wtpf.models.ablation_variants import ABLATION_REGISTRY
from wtpf.models.attention_bilstm import AttentionBiLSTMForecaster
from wtpf.models.bilstm import BiLSTMForecaster
from wtpf.models.cnn import CNNForecaster
from wtpf.models.cnn_attention_bilstm import CNNAttentionBiLSTM
from wtpf.models.cnn_bilstm import CNNBiLSTMForecaster
from wtpf.models.lstm import LSTMForecaster

MODEL_REGISTRY = {
    "cnn": CNNForecaster,
    "lstm": LSTMForecaster,
    "bilstm": BiLSTMForecaster,
    "cnn_bilstm": CNNBiLSTMForecaster,
    "attention_bilstm": AttentionBiLSTMForecaster,
    "cnn_attention_bilstm": CNNAttentionBiLSTM,
    **ABLATION_REGISTRY,
}


def build_model(name: str, model_cfg: dict) -> nn.Module:
    """Construct a model by registry name using flattened hyperparameters
    pulled from `configs/config.yaml`'s `model:` section.

    Parameters
    ----------
    name:
        One of MODEL_REGISTRY's keys, e.g. "cnn_attention_bilstm".
    model_cfg:
        The `model` section of the loaded Config (dict-like).
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY.keys())}"
        )

    cls = MODEL_REGISTRY[name]
    kwargs = dict(
        input_features=model_cfg["input_features"],
        cnn_channels=model_cfg["cnn"]["channels"],
        kernel_size=model_cfg["cnn"]["kernel_size"],
        cnn_dropout=model_cfg["cnn"]["dropout"],
        bilstm_hidden=model_cfg["bilstm"]["hidden_size"],
        bilstm_layers=model_cfg["bilstm"]["num_layers"],
        rnn_dropout=model_cfg["bilstm"]["dropout"],
        attn_dropout=model_cfg["attention"]["dropout"],
        fc_hidden_dim=model_cfg["fc_head"]["hidden_dim"],
        fc_dropout=model_cfg["fc_head"]["dropout"],
        # aliases used by some baseline constructors
        hidden_size=model_cfg["bilstm"]["hidden_size"],
        num_layers=model_cfg["bilstm"]["num_layers"],
        lstm_hidden=model_cfg["bilstm"]["hidden_size"],
        lstm_layers=model_cfg["bilstm"]["num_layers"],
    )

    # Filter kwargs to those accepted by the target constructor.
    import inspect

    sig = inspect.signature(cls.__init__)
    accepted = set(sig.parameters.keys()) - {"self"}
    has_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if has_var_kwargs:
        filtered = kwargs
    else:
        filtered = {k: v for k, v in kwargs.items() if k in accepted}

    return cls(**filtered)
