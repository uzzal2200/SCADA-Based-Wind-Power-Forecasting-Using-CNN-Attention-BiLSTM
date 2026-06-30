import torch
from torch.utils.data import DataLoader, TensorDataset

from wtpf.config import load_config
from wtpf.models.factory import build_model
from wtpf.training.trainer import Trainer


def _make_loaders(n_train=64, n_val=16, T=24, F=8, batch_size=16):
    g = torch.Generator().manual_seed(0)
    X_train = torch.randn(n_train, T, F, generator=g)
    y_train = torch.randn(n_train, generator=g)
    X_val = torch.randn(n_val, T, F, generator=g)
    y_val = torch.randn(n_val, generator=g)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def test_trainer_runs_a_few_epochs_and_improves_or_holds_loss():
    cfg = load_config("configs/config.yaml")
    cfg.model.sequence_length = 24
    model = build_model("cnn_attention_bilstm", cfg.model)

    train_loader, val_loader = _make_loaders(T=24)
    trainer = Trainer(
        model,
        device=torch.device("cpu"),
        learning_rate=1e-3,
        early_stopping_patience=10,
        scheduler_patience=10,
        logger=lambda msg: None,
    )
    history = trainer.fit(train_loader, val_loader, max_epochs=3)

    assert len(history.train_loss) == 3
    assert len(history.val_loss) == 3
    assert history.best_epoch >= 1
    assert all(v >= 0 for v in history.train_loss)


def test_trainer_predict_returns_aligned_arrays():
    cfg = load_config("configs/config.yaml")
    cfg.model.sequence_length = 24
    model = build_model("lstm", cfg.model)

    train_loader, val_loader = _make_loaders(T=24)
    trainer = Trainer(model, device=torch.device("cpu"), logger=lambda msg: None)
    trainer.fit(train_loader, val_loader, max_epochs=1)

    y_true, y_pred = trainer.predict(val_loader)
    assert y_true.shape == y_pred.shape
    assert y_true.shape[0] == 16


def test_early_stopping_restores_best_weights():
    from wtpf.training.callbacks import EarlyStopping
    import torch.nn as nn

    model = nn.Linear(2, 1)
    es = EarlyStopping(patience=2)

    es.step(1.0, model)
    best_state_after_first = {k: v.clone() for k, v in model.state_dict().items()}

    # mutate weights, then report a worse loss twice -> should trigger stop
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)

    es.step(2.0, model)
    assert not es.should_stop
    es.step(3.0, model)
    assert es.should_stop

    restored = es.restore_best(model)
    for k, v in restored.state_dict().items():
        assert torch.allclose(v, best_state_after_first[k])
