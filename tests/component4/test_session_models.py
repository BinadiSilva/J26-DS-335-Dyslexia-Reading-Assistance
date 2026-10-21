"""Leakage, chronology, artifact round trips and every proposed model family."""
from copy import deepcopy
import numpy as np
import pytest

from backend.app.component4.models import prepare_examples, split_groups, Predictor, SessionModel, train_comparison, load_model
from backend.app.component4.schema import FEATURES
from backend.app.component4.synthetic import generate_cohort


def test_sequences_preserve_history_and_keep_participants_disjoint():
    frame = generate_cohort(18, 8)
    x, y, groups, dates, data = prepare_examples(frame)
    assert x.shape == (72, 4, len(FEATURES)) and y.shape == (72, 4)
    assert all(item["input_end"] < item["target_date"] for item in dates)
    splits = split_groups(groups)
    assert not (set(splits["fit"]) & set(splits["validation"]))
    assert not (set(splits["fit"]) & set(splits["test"]))
    assert not (set(splits["validation"]) & set(splits["test"]))
    assert set(sum(splits.values(), [])) == set(groups)
    assert "reference_stagnation" not in FEATURES and "learner_id" not in FEATURES


def test_missing_week_is_not_imputed_or_labelled_as_one_week():
    frame = generate_cohort(1, 6).drop(index=2)
    with pytest.raises(ValueError, match="five weekly"):
        prepare_examples(frame)


def test_imported_derived_values_are_recomputed_and_sources_cannot_mix():
    frame = generate_cohort(18, 8)
    frame["reading_accuracy"] = 999
    x, _, _, _, _ = prepare_examples(frame)
    assert x[:, :, 1].max() <= 100
    frame.loc[0, "source"] = "collected"
    with pytest.raises(ValueError, match="separately"):
        prepare_examples(frame)


def test_scaler_fits_only_training_and_predictions_repeat():
    x, y, *_ = prepare_examples(generate_cohort(18, 8))
    estimator = Predictor("Linear Regression").fit(x[:20], y[:20], (x[20:] + 1000, y[20:] + 1000))
    np.testing.assert_allclose(estimator.x_scale.mean_, x[:20].reshape(-1, len(FEATURES)).mean(axis=0))
    np.testing.assert_allclose(estimator.predict(x[20:]), estimator.predict(x[20:]))


def test_forecast_refuses_source_mismatch_irregularity_and_out_of_range():
    frame = generate_cohort(18, 8)
    x, y, *_ = prepare_examples(frame)
    predictor = Predictor("Linear Regression").fit(x, y)
    bundle = SessionModel(predictor, "synthetic", (x.min(axis=(0, 1)), x.max(axis=(0, 1))), {})
    history = frame[frame.learner_id == "SYN0001"].head(4).to_dict("records")
    forecast = bundle.forecast(history, "synthetic")
    assert forecast["status"] == "available" and forecast["date"] == "2025-02-03"
    assert forecast["confidence"] is None
    assert bundle.forecast(history, "collected")["status"] == "unavailable"
    assert bundle.forecast(history[:2], "synthetic")["status"] == "unavailable"
    bad = deepcopy(history)
    bad[-1]["assessed_on"] = "2025-01-28"
    assert bundle.forecast(bad, "synthetic")["status"] == "unavailable"
    bad = deepcopy(history)
    bad[-1]["reading_speed_wpm"] = 999
    assert bundle.forecast(bad, "synthetic")["status"] == "unavailable"


@pytest.mark.parametrize("kind", ["Random Forest", "XGBoost", "LSTM", "GRU"])
def test_real_model_family_fits_and_roundtrips(kind, tmp_path):
    import joblib
    if kind in {"LSTM", "GRU"}:
        pytest.importorskip("torch", reason="Use the documented Component 4 virtual environment for full model coverage")
    if kind == "XGBoost":
        pytest.importorskip("xgboost")
    x, y, *_ = prepare_examples(generate_cohort(18, 8))
    model = Predictor(kind, epochs=2).fit(x[:40], y[:40], (x[40:], y[40:]))
    prediction = model.predict(x[40:])
    assert prediction.shape == y[40:].shape and np.isfinite(prediction).all()
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    np.testing.assert_allclose(joblib.load(path).predict(x[40:]), prediction)


def test_experiment_saves_manifest_preserves_artifacts_and_loads(tmp_path):
    destination = tmp_path / "experiment"
    model, report = train_comparison(generate_cohort(18, 8), destination, names=("Persistence", "Linear Regression"))
    assert report["participants"]["test"] > 0
    assert report["stagnation"]["reference_source"].startswith("generator")
    assert report["intervention_effectiveness"]["status"] == "not_established"
    assert (destination / "split_manifest.json").exists()
    assert load_model(destination).predictor.name == model.predictor.name
    with pytest.raises(FileExistsError):
        train_comparison(generate_cohort(18, 8), destination)
    with (destination / "model.joblib").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_model(destination)
