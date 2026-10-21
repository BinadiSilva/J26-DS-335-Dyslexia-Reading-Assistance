"""Participant-separated, one-week, multi-metric model comparison.

All models receive the same four weekly sessions. Selection uses validation
only. No learner IDs, reference labels or future intervention outcomes are
features. Models predict changes from the latest assessment.
"""

from datetime import date, timedelta
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from .analytics import monitor
from .schema import FEATURES, TARGETS, WINDOW, HORIZON_DAYS, validate_session

MODEL_NAMES = ("Persistence", "Linear Regression", "Random Forest", "XGBoost", "LSTM", "GRU")
SEED = 335


def prepare_examples(frame):
    required = {"learner_id", "source", "assessed_on", *FEATURES}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing session columns: {sorted(required - set(frame.columns))}")
    if frame.empty or frame.learner_id.isna().any() or frame.learner_id.astype(str).str.strip().eq("").any():
        raise ValueError("Nonempty learner IDs and sessions are required.")
    if frame.source.nunique() != 1 or frame.source.iloc[0] not in {"synthetic", "collected"}:
        raise ValueError("Train synthetic and collected datasets separately.")
    if frame.duplicated(["learner_id", "assessed_on"]).any():
        raise ValueError("Duplicate learner assessment dates.")
    validated = []
    for row in frame.to_dict("records"):
        # Derived values are recomputed, never trusted from an imported CSV.
        validated.append({**row, **validate_session(row)})
    data = pd.DataFrame(validated).sort_values(["learner_id", "assessed_on"])
    xs, ys, groups, metadata = [], [], [], []
    for learner_id, history in data.groupby("learner_id", sort=True):
        rows = history.to_dict("records")
        for end in range(WINDOW, len(rows)):
            sequence = rows[end - WINDOW:end + 1]
            days = [date.fromisoformat(s["assessed_on"]) for s in sequence]
            # The declared forecast horizon is seven days at unchanged difficulty.
            if any((b - a).days != HORIZON_DAYS for a, b in zip(days, days[1:])):
                continue
            if len({s["difficulty_level"] for s in sequence}) != 1:
                continue
            xs.append([[s[k] for k in FEATURES] for s in sequence[:-1]])
            ys.append([sequence[-1][k] for k in TARGETS])
            groups.append(str(learner_id))
            metadata.append({"learner_id": str(learner_id), "input_end": days[-2].isoformat(), "target_date": days[-1].isoformat()})
    if not xs:
        raise ValueError("Need at least five weekly assessments at unchanged difficulty per eligible learner.")
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64), np.array(groups), metadata, data


def split_groups(groups):
    ids = np.unique(groups)
    if len(ids) < 15:
        raise ValueError("At least 15 eligible learners are required for separate training, validation and test groups.")
    development, test = train_test_split(ids, test_size=0.2, random_state=SEED)
    fit, validation = train_test_split(development, test_size=0.2, random_state=SEED + 1)
    return {"fit": sorted(fit.tolist()), "validation": sorted(validation.tolist()), "test": sorted(test.tolist())}


def _network(kind, inputs, hidden=24):
    import torch

    class RecurrentNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            cell = torch.nn.LSTM if kind == "LSTM" else torch.nn.GRU
            self.recurrent = cell(inputs, hidden, batch_first=True)
            self.output = torch.nn.Linear(hidden, len(TARGETS))

        def forward(self, x):
            states, _ = self.recurrent(x)
            return self.output(states[:, -1, :])

    return RecurrentNetwork()


class SequenceRegressor:
    """Small CPU LSTM/GRU with validation early stopping and persisted weights."""
    def __init__(self, kind, epochs=80):
        self.kind, self.epochs = kind, epochs

    def fit(self, x, y, validation=None):
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(SEED)
        torch.use_deterministic_algorithms(True)
        net = _network(self.kind, x.shape[-1])
        optimizer = torch.optim.Adam(net.parameters(), lr=0.003)
        loss_fn = torch.nn.MSELoss()
        tx, ty = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        generator = torch.Generator().manual_seed(SEED)
        best_loss, stale, best_state = float("inf"), 0, None
        self.best_epoch = self.epochs
        for epoch in range(self.epochs):
            net.train()
            for batch in torch.randperm(len(tx), generator=generator).split(128):
                optimizer.zero_grad()
                loss = loss_fn(net(tx[batch]), ty[batch])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
                optimizer.step()
            if validation is not None:
                net.eval()
                with torch.no_grad():
                    val_loss = loss_fn(net(torch.tensor(validation[0], dtype=torch.float32)), torch.tensor(validation[1], dtype=torch.float32)).item()
                if val_loss < best_loss - 1e-6:
                    best_loss, stale, self.best_epoch = val_loss, 0, epoch + 1
                    best_state = {key: value.detach().clone() for key, value in net.state_dict().items()}
                else:
                    stale += 1
                if stale >= 10:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        self.weights = {key: value.detach().numpy().copy() for key, value in net.state_dict().items()}
        self._net = net
        return self

    def predict(self, x):
        import torch
        if getattr(self, "_net", None) is None:
            self._net = _network(self.kind, x.shape[-1])
            self._net.load_state_dict({key: torch.tensor(value) for key, value in self.weights.items()})
            self._net.eval()
        with torch.no_grad():
            return self._net(torch.tensor(x, dtype=torch.float32)).numpy()

    def __getstate__(self):
        return {key: value for key, value in self.__dict__.items() if key != "_net"}


class Predictor:
    def __init__(self, name, epochs=80):
        self.name, self.epochs = name, epochs

    def fit(self, x, y, validation=None):
        if self.name == "Persistence":
            return self
        self.x_scale = StandardScaler().fit(x.reshape(-1, x.shape[-1]))
        self.y_scale = StandardScaler().fit(y - x[:, -1, :len(TARGETS)])
        sx = self._features(x)
        sy = self.y_scale.transform(y - x[:, -1, :len(TARGETS)])
        if self.name == "Linear Regression":
            self.estimator = LinearRegression()
        elif self.name == "Random Forest":
            self.estimator = RandomForestRegressor(n_estimators=160, min_samples_leaf=4, random_state=SEED, n_jobs=1)
        elif self.name == "XGBoost":
            from xgboost import XGBRegressor
            self.estimator = MultiOutputRegressor(XGBRegressor(
                n_estimators=140, max_depth=3, learning_rate=0.05, objective="reg:squarederror",
                random_state=SEED, n_jobs=1, subsample=1.0, colsample_bytree=1.0,
            ))
        elif self.name in {"LSTM", "GRU"}:
            self.estimator = SequenceRegressor(self.name, self.epochs)
        else:
            raise ValueError("Unknown model.")
        if self.name in {"LSTM", "GRU"}:
            transformed_validation = None if validation is None else (
                self._features(validation[0]),
                self.y_scale.transform(validation[1] - validation[0][:, -1, :len(TARGETS)]),
            )
            self.estimator.fit(sx, sy, transformed_validation)
        else:
            self.estimator.fit(sx, sy)
        return self

    def _features(self, x):
        scaled = self.x_scale.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
        return scaled if self.name in {"LSTM", "GRU"} else scaled.reshape(len(x), -1)

    def predict(self, x):
        base = x[:, -1, :len(TARGETS)].copy()
        if self.name != "Persistence":
            base += self.y_scale.inverse_transform(self.estimator.predict(self._features(x)))
        base[:, 0] = np.maximum(base[:, 0], 0)
        base[:, 1:3] = np.clip(base[:, 1:3], 0, 100)
        base[:, 3] = np.maximum(base[:, 3], 0)
        return base


def metrics(y, prediction, groups):
    return {key: {
        "mae": float(mean_absolute_error(y[:, index], prediction[:, index])),
        "rmse": float(np.sqrt(mean_squared_error(y[:, index], prediction[:, index]))),
        "r2": float(r2_score(y[:, index], prediction[:, index])),
        "participant_mae": float(np.mean([np.abs(y[groups == g, index] - prediction[groups == g, index]).mean() for g in np.unique(groups)])),
    } for index, key in enumerate(TARGETS)}


def evaluate_stagnation(data, test_ids):
    """Use independent reference annotations, never the detector's own outputs."""
    if "reference_stagnation" not in data:
        return {"status": "unavailable", "reason": "Supply independent reference_stagnation annotations to evaluate detection."}
    truth, predicted, abstained = [], [], 0
    for learner_id, history in data.groupby("learner_id"):
        if str(learner_id) not in test_ids:
            continue
        row = history.iloc[-1]
        label = row.reference_stagnation
        if pd.isna(label):
            continue
        if label not in (True, False, 0, 1, "True", "False", "true", "false", "0", "1"):
            raise ValueError("Reference stagnation labels must be boolean (or 0/1).")
        reference = str(label).lower() in {"true", "1"}
        risk = monitor(history.to_dict("records"))
        if risk["level"] not in {"High", "Medium", "Low"}:
            abstained += 1
            continue
        truth.append(reference)
        predicted.append(risk["level"] in {"High", "Medium"})
    if not truth:
        return {"status": "unavailable", "reason": "No comparable labelled test learners.", "abstained": abstained}
    precision, recall, f1, _ = precision_recall_fscore_support(truth, predicted, average="binary", zero_division=0)
    return {"status": "evaluated", "reference_source": "generator latent scenario; synthetic only" if data.source.iloc[0] == "synthetic" else "supplied independent annotations",
            "learners": len(truth), "abstained": abstained, "accuracy": float(accuracy_score(truth, predicted)),
            "precision": float(precision), "recall": float(recall), "f1": float(f1),
            "confusion_matrix_labels": ["progress", "stagnation"],
            "confusion_matrix": confusion_matrix(truth, predicted, labels=[False, True]).tolist(),
            "majority_baseline_accuracy": max(sum(truth), len(truth) - sum(truth)) / len(truth)}


class SessionModel:
    def __init__(self, predictor, source, bounds, report):
        self.predictor, self.source, self.bounds, self.report = predictor, source, bounds, report

    def forecast(self, sessions, source):
        def unavailable(reason):
            return {"status": "unavailable", "reason": reason}
        if source != self.source:
            return unavailable("This model was trained on a different data source. Synthetic models are not used for collected learner records.")
        history = sorted(sessions, key=lambda s: s["assessed_on"])[-WINDOW:]
        if len(history) < WINDOW:
            return unavailable(f"Record {WINDOW} weekly assessments before forecasting.")
        dates = [date.fromisoformat(s["assessed_on"]) for s in history]
        if any((b - a).days != HORIZON_DAYS for a, b in zip(dates, dates[1:])):
            return unavailable("This model needs four assessments seven days apart; irregular sessions are still stored and reported.")
        if len({s["difficulty_level"] for s in history}) != 1:
            return unavailable("The four assessments must use the same passage difficulty.")
        x = np.array([[[s[k] for k in FEATURES] for s in history]])
        low, high = np.array(self.bounds[0]), np.array(self.bounds[1])
        if not np.isfinite(x).all() or np.any(x < low) or np.any(x > high):
            return unavailable("One or more measurements lie outside the model's training range; collect suitable training evidence.")
        prediction = self.predictor.predict(x)[0]
        return {"status": "available", "values": {key: round(float(value), 3) for key, value in zip(TARGETS, prediction)},
                "date": (dates[-1] + timedelta(days=HORIZON_DAYS)).isoformat(), "model": self.predictor.name,
                "data_source": self.source, "confidence": None,
                "reason": "Experimental one-week estimate at unchanged difficulty. Synthetic performance is not real-world validation." if source == "synthetic" else "Experimental one-week estimate at unchanged difficulty; no individual confidence percentage is established."}


def train_comparison(frame, output_dir, epochs=80, names=MODEL_NAMES):
    """Persist a versioned experiment; refuse to overwrite previous evidence."""
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("Choose a new output directory to preserve previous experiments.")
    x, y, groups, example_dates, data = prepare_examples(frame)
    splits = split_groups(groups)
    masks = {key: np.isin(groups, ids) for key, ids in splits.items()}
    fit, val, test = (masks[key] for key in ("fit", "validation", "test"))
    # Fixed training-only scales avoid averaging WPM and WER in incompatible units.
    scales = np.maximum(y[fit].std(axis=0), 1e-6)
    fitted, evaluations, selection = {}, {}, {}
    for name in names:
        print(f"Fitting {name}...", flush=True)
        started = time.perf_counter()
        estimator = Predictor(name, epochs).fit(x[fit], y[fit], (x[val], y[val]))
        prediction = estimator.predict(x[val])
        # Equal weight per participant, then per standardized target.
        selection[name] = float(np.mean([np.mean(np.abs(y[val][groups[val] == g] - prediction[groups[val] == g]) / scales) for g in np.unique(groups[val])]))
        fitted[name] = estimator
        evaluations[name] = {"validation": metrics(y[val], prediction, groups[val]),
                             "validation_selection_score": selection[name], "training_seconds": time.perf_counter() - started}
    chosen = min(selection, key=selection.get)
    # The choice is locked before reading any test metric.
    for name, estimator in fitted.items():
        evaluations[name]["test"] = metrics(y[test], estimator.predict(x[test]), groups[test])
    epochs_selected = fitted[chosen].estimator.best_epoch if chosen in {"LSTM", "GRU"} else epochs
    development = fit | val
    selected = Predictor(chosen, epochs_selected).fit(x[development], y[development])
    report = {
        "schema_version": 1, "source": str(data.source.iloc[0]), "seed": SEED,
        "history_sessions": WINDOW, "horizon_days": HORIZON_DAYS,
        "features": list(FEATURES), "targets": list(TARGETS),
        "dataset_sha256": hashlib.sha256(data.to_csv(index=False).encode()).hexdigest(),
        "participants": {key: len(ids) for key, ids in splits.items()},
        "examples": {key: int(mask.sum()) for key, mask in masks.items()},
        "selection_rule": "Lowest validation equal-participant mean absolute error divided by training target standard deviations.",
        "selection_scales": scales.tolist(), "selected_model": chosen,
        "models": evaluations, "selected_refit_test": metrics(y[test], selected.predict(x[test]), groups[test]),
        "stagnation": evaluate_stagnation(data, set(splits["test"])),
        "intervention_effectiveness": {"status": "not_established", "reason": "Dashboard tracks observed before/after changes. No causal outcome experiment has been performed."},
        "limitations": ["Synthetic data test software and assumed patterns, not real children.",
                        "Only weekly, unchanged-difficulty sequences are eligible.",
                        "No individual confidence percentage or intervention benefit is inferred.",
                        "Reference labels need independent educator annotation for collected-data validation."],
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "joblib")},
    }
    for dependency in ("torch", "xgboost"):
        try:
            report["versions"][dependency] = importlib.metadata.version(dependency)
        except importlib.metadata.PackageNotFoundError:
            pass
    bounds = (x[development].min(axis=(0, 1)).tolist(), x[development].max(axis=(0, 1)).tolist())
    bundle = SessionModel(selected, report["source"], bounds, report)
    output.mkdir(parents=True)
    # Joblib files are local trusted artifacts only; the web app has no model upload.
    joblib.dump(bundle, output / "model.joblib")
    report["model_sha256"] = hashlib.sha256((output / "model.joblib").read_bytes()).hexdigest()
    (output / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "split_manifest.json").write_text(json.dumps(splits, indent=2) + "\n")
    (output / "example_dates.json").write_text(json.dumps(example_dates, indent=2) + "\n")
    data.to_csv(output / "training_sessions.csv", index=False)
    return bundle, report


def load_model(directory):
    directory = Path(directory)
    report = json.loads((directory / "evaluation.json").read_text())
    model_path = directory / "model.joblib"
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != report["model_sha256"]:
        raise ValueError("Model checksum mismatch; use the original trusted experiment artifact.")
    return joblib.load(model_path)
