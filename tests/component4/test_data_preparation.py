"""Data quality checks using fictional CSVs and workbooks only."""

import numpy as np
import pandas as pd
import pytest

from backend.app.services.learning_progress import data_preparation as prep
from backend.app.services.learning_progress import prepare_libs_longitudinal as wide


def write_data(tmp_path, monkeypatch, rows):
    path = tmp_path / 'reading.csv'
    pd.DataFrame(rows, columns=prep.LIBS_REQUIRED_COLUMNS).to_csv(path, index=False)
    monkeypatch.setattr(prep, 'LIBS_DATA_PATH', path)


def test_loader_keeps_zero_sorts_and_reports_exclusions(tmp_path, monkeypatch):
    write_data(tmp_path, monkeypatch, [
        (1, 'T2BAS_word_reading', 10, 2),
        (1, 'T1BAS_word_reading', 0, 1),
        (1, 'T1BAS_word_reading', 0, 1),
        (2, 'T1BAS_word_reading', '.', 1),
        (3, 'T1BAS_word_reading', 91, 1),
        (np.inf, 'T1BAS_word_reading', 10, 1),
        (4, 'T5BAS_word_reading', 10, 5),
    ])
    result = prep.load_libs_reading_data()
    assert result.reading_score.tolist() == [0, 10]
    assert result.reading_score_change.iloc[1] == 10
    assert result.attrs['quality_report'] == {
        'source_rows': 7, 'invalid_rows_removed': 4,
        'duplicate_rows_removed': 1, 'retained_rows': 2,
    }


@pytest.mark.parametrize('name', ['T2BAS_word_reading', 'T1DTWRPnonwords', None, ''])
def test_loader_rejects_mismatched_assessment(tmp_path, monkeypatch, name):
    write_data(tmp_path, monkeypatch, [(1, name, 10, 1)])
    with pytest.raises(ValueError, match='Assessment names'):
        prep.load_libs_reading_data()


def test_loader_rejects_conflicting_duplicate(tmp_path, monkeypatch):
    write_data(tmp_path, monkeypatch, [(1, 'T1BAS_word_reading', score, 1) for score in (10, 20)])
    with pytest.raises(ValueError, match='Conflicting'):
        prep.load_libs_reading_data()


def test_loader_rejects_no_valid_records(tmp_path, monkeypatch):
    write_data(tmp_path, monkeypatch, [(1, 'T1BAS_word_reading', -1, 1)])
    with pytest.raises(ValueError, match='No valid LIBS'):
        prep.load_libs_reading_data()


def test_loader_reports_missing_file_and_columns(tmp_path, monkeypatch):
    path = tmp_path / 'missing.csv'
    monkeypatch.setattr(prep, 'LIBS_DATA_PATH', path)
    with pytest.raises(FileNotFoundError):
        prep.load_libs_reading_data()
    pd.DataFrame({'wrong_column': [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match='missing required columns'):
        prep.load_libs_reading_data()


def test_workbook_preparation_counts_only_valid_scores(tmp_path):
    frame = pd.DataFrame({
        'UnlinkedParticipantNumber': [1, 2, 3],
        'T1BAS_word_reading': [0, 10, 10],
        'T2BAS_word_reading': [10, 100, '.'],
        'T3BAS_word_reading': ['.', '.', 30],
        'T4BAS_word_reading': [30, '.', '.'],
    })
    path = tmp_path / 'fictional.xlsx'
    frame.to_excel(path, index=False)
    result = wide.prepare_longitudinal_data(path)
    assert result.UnlinkedParticipantNumber.tolist() == [1, 1, 1, 3, 3]
    assert result.time_point.tolist() == [1, 2, 4, 1, 3]
    assert result.reading_score.tolist() == [0, 10, 30, 10, 30]
    assert not (tmp_path / 'libs_reading_assessments_long.csv').exists()


def test_preparation_refuses_to_overwrite_existing_file(tmp_path, monkeypatch):
    path = tmp_path / 'existing.csv'
    path.write_text('preserve me')
    monkeypatch.setattr('sys.argv', ['prepare', '--output', str(path)])
    with pytest.raises(SystemExit) as error:
        wide.main()
    assert error.value.code == 2
    assert path.read_text() == 'preserve me'
