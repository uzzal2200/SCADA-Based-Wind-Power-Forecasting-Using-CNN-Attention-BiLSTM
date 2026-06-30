from wtpf.config import load_config


def test_config_loads_expected_top_level_sections():
    cfg = load_config("configs/config.yaml")
    for section in ("project", "data", "features", "sequence", "model", "training", "baselines", "ablation", "evaluation"):
        assert section in cfg


def test_config_dot_access_matches_dict_access():
    cfg = load_config("configs/config.yaml")
    assert cfg.training.batch_size == cfg["training"]["batch_size"]
    assert cfg.model.sequence_length == cfg["model"]["sequence_length"]


def test_config_nested_mutation_persists():
    """Regression test: attribute-style mutation of a nested section must
    persist (previously, accessing a nested dict via dot-notation
    returned a throwaway shallow copy, silently dropping writes)."""
    cfg = load_config("configs/config.yaml")
    cfg.training.max_epochs = 7
    assert cfg.training.max_epochs == 7
    assert cfg["training"]["max_epochs"] == 7

    # repeated access should hit the *same* underlying object
    cfg.training.batch_size = 999
    assert cfg.training.batch_size == 999


def test_config_selected_features_has_eight_entries():
    cfg = load_config("configs/config.yaml")
    assert len(cfg.features.selected_features) == 8
    assert cfg.model.input_features == 8
