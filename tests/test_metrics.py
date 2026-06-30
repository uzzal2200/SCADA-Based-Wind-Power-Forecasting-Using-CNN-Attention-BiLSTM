import numpy as np

from wtpf.evaluation.metrics import compute_all_metrics, mae, mape, psnr, r2, rmse, smape
from wtpf.evaluation.power_band_analysis import power_band_breakdown
from wtpf.evaluation.seasonal_analysis import seasonal_breakdown
from wtpf.evaluation.statistical_tests import run_bonferroni_wilcoxon_suite, wilcoxon_test


def test_metrics_perfect_prediction():
    y = np.array([10.0, 20.0, 30.0, 40.0])
    bundle = compute_all_metrics(y, y.copy())
    assert np.isclose(bundle.rmse, 0.0)
    assert np.isclose(bundle.mae, 0.0)
    assert np.isclose(bundle.r2, 1.0)
    assert np.isclose(bundle.mape, 0.0)
    assert np.isclose(bundle.smape, 0.0)
    assert bundle.psnr > 100  # near-infinite for a perfect match (eps-bounded)


def test_rmse_mae_known_values():
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([3.0, 4.0, 0.0])
    assert np.isclose(rmse(y_true, y_pred), np.sqrt((9 + 16 + 0) / 3))
    assert np.isclose(mae(y_true, y_pred), (3 + 4 + 0) / 3)


def test_mape_respects_power_threshold():
    y_true = np.array([1.0, 100.0])  # first sample below 5kW threshold
    y_pred = np.array([2.0, 110.0])
    value = mape(y_true, y_pred, power_threshold_kw=5.0)
    # only the second sample (100 -> 110) should be evaluated: |10/100|*100=10%
    assert np.isclose(value, 10.0)


def test_smape_symmetric():
    """sMAPE's symmetry property is smape(y_true, y_pred) == smape(y_pred,
    y_true) (swapping which series is "truth" doesn't change the score) —
    not that equal-magnitude over/under-prediction errors yield equal
    scores (they don't, since the denominator |y|+|yhat| differs)."""
    y_true = np.array([100.0, 50.0, 30.0])
    y_pred = np.array([120.0, 40.0, 35.0])
    assert np.isclose(smape(y_true, y_pred), smape(y_pred, y_true), atol=1e-9)


def test_r2_constant_target_returns_nan():
    y_true = np.array([5.0, 5.0, 5.0])
    y_pred = np.array([5.0, 6.0, 4.0])
    assert np.isnan(r2(y_true, y_pred))


def test_psnr_decreases_with_more_noise():
    rng = np.random.default_rng(0)
    y_true = np.abs(rng.normal(200, 50, 1000))
    pred_low_noise = y_true + rng.normal(0, 5, 1000)
    pred_high_noise = y_true + rng.normal(0, 50, 1000)
    assert psnr(y_true, pred_low_noise) > psnr(y_true, pred_high_noise)


def test_power_band_breakdown_bins_correctly():
    y_true = np.array([50.0, 150.0, 450.0, 750.0])
    y_pred = y_true.copy()
    bands = {"Low": (0, 100), "Medium": (100, 300), "High": (300, 600), "Rated": (600, 900)}
    results = power_band_breakdown(y_true, y_pred, bands=bands)
    assert set(results.keys()) == {"Low", "Medium", "High", "Rated"}
    for bundle in results.values():
        assert np.isclose(bundle.rmse, 0.0)


def test_seasonal_breakdown_groups_by_month():
    import pandas as pd

    idx = pd.date_range("2012-01-01", periods=4, freq="45D")  # spans 4 different seasons
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = y_true.copy()
    results = seasonal_breakdown(idx, y_true, y_pred)
    assert len(results) >= 1
    for bundle in results.values():
        assert np.isclose(bundle.rmse, 0.0)


def test_wilcoxon_test_detects_systematic_improvement():
    rng = np.random.default_rng(0)
    proposed_errors = np.abs(rng.normal(20, 5, 2000))
    baseline_errors = np.abs(rng.normal(40, 5, 2000))  # systematically worse

    result = wilcoxon_test(proposed_errors, baseline_errors, "proposed_vs_baseline", alpha=0.01)
    assert result.significant
    assert result.z < 0  # proposed has systematically lower ranks (smaller errors)
    assert 0.0 <= result.effect_size_r <= 1.0


def test_bonferroni_suite_runs_for_multiple_baselines():
    rng = np.random.default_rng(0)
    errors = {
        "cnn_attention_bilstm": np.abs(rng.normal(20, 5, 1000)),
        "cnn": np.abs(rng.normal(50, 5, 1000)),
        "lstm": np.abs(rng.normal(40, 5, 1000)),
    }
    results = run_bonferroni_wilcoxon_suite(errors, proposed_key="cnn_attention_bilstm")
    assert set(results.keys()) == {"cnn", "lstm"}
    for r in results.values():
        assert r.significant
