"""Reproduce aggregate Component 4 results without replacing existing outputs.

Run as a module from the repository root. An optional --output path creates a
new JSON file exclusively; no participant-level scores or identifiers are saved.
"""

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import sklearn

from .data_preparation import LIBS_DATA_PATH, load_libs_reading_data
from .progress_prediction import (
    PARTICIPANT_COLUMN, RANDOM_STATE, evaluate_prediction_intervals,
    evaluate_repeated_group_validation, train_libs_progress_model,
)
from .progress_reporting import generate_progress_report
from .stagnation_detection import STUDY_AGES


def summarize_errors(data, predictions):
    """Distinguish record-weighted error from equal weight per participant."""
    actual = data['target_score'].to_numpy(dtype=float)
    predicted = np.asarray(predictions, dtype=float)
    if actual.shape != predicted.shape or not len(actual):
        raise ValueError('Predictions must match a nonempty target array.')
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError('Scores and predictions must be finite.')
    errors = actual - predicted
    per_participant = pd.DataFrame({
        'participant': data[PARTICIPANT_COLUMN].to_numpy(),
        'absolute_error': np.abs(errors),
    }).groupby('participant').absolute_error.mean()
    total_variation = np.square(actual - actual.mean()).sum()
    return {
        'records': len(data),
        'participants': len(per_participant),
        'mae': float(np.abs(errors).mean()),
        'rmse': float(np.sqrt(np.square(errors).mean())),
        'r2': float(1 - np.square(errors).sum() / total_variation) if total_variation > 0 else None,
        'participant_macro_mae': float(per_participant.mean()),
    }


def time_point_mean_predictions(fit_data, evaluation_data):
    """A simple comparator using fitting targets at each CURRENT time point.

    The target assessment time is not an input: it is unavailable for a
    next-observed-score forecast. Unseen current time points use the fit mean.
    """
    means = fit_data.groupby('time_point').target_score.mean()
    return evaluation_data.time_point.map(means).fillna(fit_data.target_score.mean()).to_numpy()


def evaluation_diagnostics(trained):
    """Add diagnostic comparisons without changing model selection."""
    test = trained['test_data']
    features = trained['features']
    predictions = {'Persistence Baseline': test.assessment_score.to_numpy()}
    for name, model in trained['models'].items():
        predictions[name] = model.predict(test[features])
    predictions['Current-time-point Mean'] = time_point_mean_predictions(trained['train_data'], test)
    selected = trained['best_model_name']
    test_predictions = predictions[selected]
    test_work = test.copy()
    test_work['prediction'] = test_predictions
    horizons = []
    for (current, target), group in test_work.groupby(['time_point', 'target_time_point']):
        horizons.append({
            'current_time_point': int(current), 'target_time_point': int(target),
            'approximate_study_year_gap': STUDY_AGES[target] - STUDY_AGES[current],
            **summarize_errors(group, group.prediction.to_numpy()),
        })
    validation = trained['validation_data']
    # This comparator is fitted on the inner fit group, never validation targets.
    comparator = time_point_mean_predictions(trained['fit_data'], validation)
    return {
        'test_models': {name: summarize_errors(test, pred) for name, pred in predictions.items()},
        'time_point_mean_validation': summarize_errors(validation, comparator),
        'selected_model_by_observed_transition': horizons,
        'note': 'The added time-point comparator is diagnostic, not a new selection candidate. No parameters were tuned on these results.',
    }


def _participant_digest(ids):
    # A digest permits checking reproducibility without publishing participant IDs.
    return hashlib.sha256(json.dumps(sorted(float(x) for x in ids)).encode()).hexdigest()


def build_evaluation_report():
    """Run existing workflows and return aggregate, provenance-labelled results."""
    reading = load_libs_reading_data()
    with contextlib.redirect_stdout(io.StringIO()):
        trained = train_libs_progress_model(save_results=False)
        repeated = evaluate_repeated_group_validation()
        intervals = evaluate_prediction_intervals()
        report = generate_progress_report()
    test_ids = set(trained['test_data'][PARTICIPANT_COLUMN])
    if not test_ids == repeated['test_participants'] == intervals['test_participants']:
        raise RuntimeError('Evaluation workflows do not share test participants.')
    interval_groups = []
    for target, group in intervals['test_results'].groupby('target_time_point'):
        interval_groups.append({
            'target_time_point': int(target), 'records': len(group),
            'participants': int(group[PARTICIPANT_COLUMN].nunique()),
            'record_coverage': float(group.covered.mean()),
            'mean_interval_width': float(group.interval_width.mean()),
        })
    result = {
        'measured_at_utc': datetime.now(timezone.utc).isoformat(),
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'pandas': pd.__version__, 'scikit_learn': sklearn.__version__},
        'provenance': {
            'dataset_sha256': hashlib.sha256(LIBS_DATA_PATH.read_bytes()).hexdigest(),
            'outer_split_seed': RANDOM_STATE,
            'development_participants': int(trained['train_data'][PARTICIPANT_COLUMN].nunique()),
            'test_participants': len(test_ids),
            'test_participant_digest': _participant_digest(test_ids),
            'status': 'Exploratory: earlier results have already been inspected.',
            'source_sha256': {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(Path(__file__).parent.glob('*.py'))
            },
        },
        'data_quality': reading.attrs['quality_report'],
        'dataset': {
            'observed_records': len(reading),
            'participants': int(reading[PARTICIPANT_COLUMN].nunique()),
            'prediction_examples': len(trained['data']),
            'prediction_participants': int(trained['data'][PARTICIPANT_COLUMN].nunique()),
            'assessment_counts': {str(k): int(v) for k, v in reading.groupby(PARTICIPANT_COLUMN).size().value_counts().sort_index().items()},
        },
        'selected_model': trained['best_model_name'],
        'validation_metrics': trained['validation_results'],
        'diagnostics': evaluation_diagnostics(trained),
        'repeated_validation': repeated['summary'].to_dict('records'),
        'repeated_validation_details': repeated['detailed_results'].to_dict('records'),
        'intervals': intervals['summary'],
        'intervals_by_target_time_point': interval_groups,
        'monitoring': {
            'participants': len(report),
            'screening_categories': report.stagnation_risk.value_counts().to_dict(),
            'latest_declines': int(report.recent_decline.sum()),
            'missing_intermediate_time_points': int(report.has_missing_time_points.sum()),
            'forecast_statuses': report.forecast_status.value_counts().to_dict(),
            'support_priorities': report.support_priority.value_counts().to_dict(),
        },
    }
    workbook = LIBS_DATA_PATH.parents[2] / 'LIBS_DataSet_UnLinked_July2024.xlsx'
    if workbook.exists():
        source = pd.read_excel(workbook)
        scores = source[[f'T{i}BAS_word_reading' for i in range(1, 5)]].apply(pd.to_numeric, errors='coerce')
        scores = scores.where(scores.ge(0) & scores.le(90))
        result['dataset']['source_rows'] = len(source)
        result['dataset']['source_valid_bas_count_distribution'] = {
            str(k): int(v) for k, v in scores.notna().sum(axis=1).value_counts().sort_index().items()
        }
        result['provenance']['workbook_sha256'] = hashlib.sha256(workbook.read_bytes()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='New JSON path; existing files are never overwritten')
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error('Output already exists; choose a new filename.')
    result = build_evaluation_report()
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as output:
            output.write(text + '\n')
        print(f'Aggregate evaluation saved to {args.output}')
    else:
        print(text)


if __name__ == '__main__':
    main()
