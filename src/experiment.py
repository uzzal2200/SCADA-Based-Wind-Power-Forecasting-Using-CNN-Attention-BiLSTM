"""
High-level single-model experiment orchestration.

`run_experiment` trains one named architecture (any key in
`wtpf.models.MODEL_REGISTRY`) end-to-end on an already-prepared dataset
and returns everything downstream scripts need: training history, test
predictions (physical units, kW), the full metric bundle, and basic
complexity/profiling stats (Table 6). Used by `scripts/02_train.py`,
`scripts/03_train_all_baselines.py` and `scripts/05_ablation_study.py`
so the core train/evaluate logic lives in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from wtpf.config import Config
from wtpf.data.dataset import PreparedData, make_dataloaders
from wtpf.evaluation.metrics import MetricBundle, compute_all_metrics
from wtpf.models.factory import build_model
from wtpf.training.profiling import count_parameters, measure_inference_latency_ms, model_size_mb
from wtpf.training.trainer import Trainer, TrainingHistory


@dataclass
class ExperimentResult:
    model_name: str
    history: TrainingHistory
    y_true_kw: np.ndarray
    y_pred_kw: np.ndarray
    metrics: MetricBundle
    n_params: int
    size_mb: float
    inference_latency_ms: float


def run_experiment(
    model_name: str,
    cfg: Config,
    prepared: PreparedData,
    device: torch.device,
    checkpoint_dir: Optional[str | Path] = None,
    logger=None,
) -> ExperimentResult:
    """Build, train and evaluate a single architecture end-to-end.

    Parameters
    ----------
    model_name:
        Key into `wtpf.models.MODEL_REGISTRY` (e.g. "cnn_attention_bilstm").
    cfg:
        Loaded Config (uses `cfg.model`, `cfg.training`, `cfg.evaluation`).
    prepared:
        Output of `wtpf.pipeline.build_datasets` — shared across all
        models so every architecture trains/evaluates on identical
        windows and scalers.
    """
    log = logger or (lambda msg: print(msg))

    train_loader, val_loader, test_loader = make_dataloaders(
        prepared.train_ds,
        prepared.val_ds,
        prepared.test_ds,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
    )

    model = build_model(model_name, cfg.model)
    n_params = count_parameters(model)
    size_mb = model_size_mb(model)
    log(f"[{model_name}] {n_params:,} trainable parameters, {size_mb:.3f} MB on disk")

    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        optimizer_name=cfg.training.optimizer,
        huber_delta=cfg.training.huber_delta,
        grad_clip_max_norm=cfg.training.grad_clip_max_norm,
        scheduler_patience=cfg.training.lr_scheduler.patience,
        scheduler_factor=cfg.training.lr_scheduler.factor,
        early_stopping_patience=cfg.training.early_stopping.patience,
        seed=cfg.project.seed,
        logger=log,
    )

    ckpt_path = None
    if checkpoint_dir is not None:
        ckpt_path = Path(checkpoint_dir) / f"{model_name}.pt"

    history = trainer.fit(train_loader, val_loader, max_epochs=cfg.training.max_epochs, checkpoint_path=ckpt_path)

    y_true_scaled, y_pred_scaled = trainer.predict(test_loader)
    y_true_kw = prepared.target_scaler.inverse_transform(y_true_scaled)
    y_pred_kw = prepared.target_scaler.inverse_transform(y_pred_scaled)

    metrics = compute_all_metrics(
        y_true_kw,
        y_pred_kw,
        power_threshold_kw=cfg.evaluation.mape_power_threshold_kw,
        smape_eps=cfg.evaluation.smape_eps,
        psnr_eps=cfg.evaluation.psnr_eps,
    )
    log(f"[{model_name}] Test metrics: {metrics}")

    sample_input = prepared.test_ds[0][0].unsqueeze(0)
    latency_ms = measure_inference_latency_ms(model, sample_input, device)

    return ExperimentResult(
        model_name=model_name,
        history=history,
        y_true_kw=y_true_kw,
        y_pred_kw=y_pred_kw,
        metrics=metrics,
        n_params=n_params,
        size_mb=size_mb,
        inference_latency_ms=latency_ms,
    )
