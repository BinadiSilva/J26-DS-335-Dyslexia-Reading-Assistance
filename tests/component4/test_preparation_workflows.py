"""Verify the older preparation workflow cannot replace the current BAS CSV."""

import pandas as pd

from backend.app.services.learning_progress import (
    analyze_libs_progress,
    data_preparation,
    prepare_bas_prediction_data,
    prepare_libs_dataset,
)


def test_multi_assessment_workflow_preserves_bas_data(tmp_path, monkeypatch):
    """Exercise Excel preparation and both consumers using temporary files."""

    bas_path = tmp_path / data_preparation.LIBS_DATA_PATH.name
    all_path = tmp_path / prepare_libs_dataset.OUTPUT_FILE.name
    assert all_path != bas_path
    assert prepare_bas_prediction_data.INPUT_FILE == prepare_libs_dataset.OUTPUT_FILE
    assert analyze_libs_progress.INPUT_FILE == prepare_libs_dataset.OUTPUT_FILE

    # An existing current-format BAS dataset must survive the older workflow.
    original_bas = (
        "UnlinkedParticipantNumber,assessment,reading_score,time_point\n"
        "101,T1BAS_word_reading,0,1\n"
        "101,T2BAS_word_reading,10,2\n"
    )
    bas_path.write_text(original_bas, encoding="utf-8")

    # Include a genuine zero, an unavailable score, and a different assessment.
    workbook = tmp_path / "fictional.xlsx"
    pd.DataFrame(
        {
            "UnlinkedParticipantNumber": [101],
            "T1BAS_word_reading": [0],
            "T2BAS_word_reading": [10],
            "T3BAS_word_reading": [20],
            "T4BAS_word_reading": ["."],
            "T1DTWRPnonwords": [5],
        }
    ).to_excel(workbook, index=False)

    monkeypatch.setattr(prepare_libs_dataset, "INPUT_FILE", workbook)
    monkeypatch.setattr(prepare_libs_dataset, "OUTPUT_FILE", all_path)
    prepare_libs_dataset.main()
    prepared = pd.read_csv(all_path)
    assert len(prepared) == 4
    assert set(prepared["assessment_name"]) == {"BAS_word_reading", "DTWRPnonwords"}
    assert prepared["assessment_score"].eq(0).sum() == 1

    prediction_path = tmp_path / "prediction.csv"
    monkeypatch.setattr(prepare_bas_prediction_data, "INPUT_FILE", all_path)
    monkeypatch.setattr(prepare_bas_prediction_data, "OUTPUT_FILE", prediction_path)
    prepare_bas_prediction_data.main()
    predictions = pd.read_csv(prediction_path)
    assert len(predictions) == 1
    assert predictions.iloc[0]["previous_score"] == 0
    assert predictions.iloc[0]["assessment_score"] == 10
    assert predictions.iloc[0]["target_score"] == 20

    changes_path = tmp_path / "changes.csv"
    monkeypatch.setattr(analyze_libs_progress, "INPUT_FILE", all_path)
    monkeypatch.setattr(analyze_libs_progress, "OUTPUT_FILE", changes_path)
    analyze_libs_progress.main()
    changes = pd.read_csv(changes_path)
    assert changes["score_change"].dropna().tolist() == [10, 10]

    assert bas_path.read_text(encoding="utf-8") == original_bas
    monkeypatch.setattr(data_preparation, "LIBS_DATA_PATH", bas_path)
    assert data_preparation.load_libs_reading_data()["reading_score"].tolist() == [0, 10]
