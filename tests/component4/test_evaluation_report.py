"""Verify aggregate metrics and a complete fictional reporting workflow."""

import numpy as np
import pandas as pd
import pytest

from backend.app.services.learning_progress import evaluation_report as evaluation
from backend.app.services.learning_progress import progress_prediction as prediction
from backend.app.services.learning_progress import progress_reporting as reporting
from backend.app.services.learning_progress import stagnation_detection as screening


def test_macro_mae_weights_participants_equally():
    data = pd.DataFrame({'UnlinkedParticipantNumber': [1, 1, 2],
                         'target_score': [0, 0, 9]})
    metrics = evaluation.summarize_errors(data, [0, 0, 0])
    assert metrics['mae'] == 3
    assert metrics['participant_macro_mae'] == 4.5
    assert metrics['rmse'] == pytest.approx(np.sqrt(27))


def test_one_record_r2_is_explicitly_unavailable():
    data = pd.DataFrame({'UnlinkedParticipantNumber': [1], 'target_score': [20]})
    assert evaluation.summarize_errors(data, [10])['r2'] is None


def test_time_point_baseline_uses_only_fitting_targets():
    fitting = pd.DataFrame({'time_point': [2, 2, 3], 'target_score': [10, 30, 50]})
    validation = pd.DataFrame({'time_point': [2, 3, 4], 'target_score': [90, 90, 90]})
    expected = [20, 50, 30]  # Unseen time point falls back to fitting mean.
    np.testing.assert_equal(evaluation.time_point_mean_predictions(fitting, validation), expected)
    validation['target_score'] = -1000
    np.testing.assert_equal(evaluation.time_point_mean_predictions(fitting, validation), expected)


def test_complete_report_preserves_missing_forecast_and_decline(monkeypatch):
    rows = [(1, 1, 10), (1, 2, 50), (1, 3, 40), (1, 4, 30), (2, 1, 20)]
    rows += [(person, time, 10 + person + 5 * time)
             for person in range(3, 21) for time in range(1, 5)]
    data = pd.DataFrame(rows, columns=['UnlinkedParticipantNumber', 'time_point', 'reading_score'])
    for module in (prediction, reporting, screening):
        monkeypatch.setattr(module, 'load_libs_reading_data', lambda: data.copy())
    report = reporting.generate_progress_report().set_index('UnlinkedParticipantNumber')
    assert len(report) == 20
    assert report.loc[1, 'recent_decline']
    assert report.loc[1, 'support_priority'] == 'Educator review'
    assert report.loc[1, 'forecast_status'] == 'unseen_training_time_point'
    assert report.loc[1, 'forecast_use'].startswith('Not used')
    assert report.loc[2, 'forecast_status'] == 'no_forecast'
    assert report.loc[2, 'support_priority'] == 'Collect evidence'
    assert report.review_required.all()


def test_report_command_refuses_existing_output(tmp_path, monkeypatch):
    path = tmp_path / 'result.json'
    path.write_text('preserve')
    monkeypatch.setattr('sys.argv', ['evaluation', '--output', str(path)])
    with pytest.raises(SystemExit) as error:
        evaluation.main()
    assert error.value.code == 2
    assert path.read_text() == 'preserve'
