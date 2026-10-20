
import json
import sys
from pathlib import Path

from audio_preprocessing import preprocess_audio
from speech_recognition import load_whisper_model, transcribe_audio
from acoustic_features import extract_acoustic_features

from reading_metrics import (
    align_words,
    normalize_text,
    calculate_reading_duration,
    calculate_wpm,
    detect_repetitions,
    detect_hesitations,
    find_mps_record
)


# --------------------------------------------------
# Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[4]

DEFAULT_RECORDING_ID = "3b15g_EN-OL-RC-234_2"

METADATA_PATH = (
    PROJECT_ROOT / "datasets/c1/mps/selected_20.json"
)

AUDIO_FOLDER = (
    PROJECT_ROOT / "datasets/c1/mps/audios"
)

OUTPUT_FOLDER = (
    PROJECT_ROOT / "datasets/c1/test_audio"
)


# --------------------------------------------------
# Process one reading recording
# --------------------------------------------------

def process_recording(recording_id, model, records=None):

    recording_id = Path(str(recording_id)).stem

    audio_path = AUDIO_FOLDER / f"{recording_id}.wav"
    output_path = OUTPUT_FOLDER / f"{recording_id}_output.json"

    print("\nC1 - Reading Speech Analysis & Feature Extraction")
    print("================================================")
    print("Recording:", recording_id)

    # --------------------------------------------------
    # 1. Load reference passage
    # --------------------------------------------------

    if records is None:
        with open(
            METADATA_PATH,
            "r",
            encoding="utf-8"
        ) as file:
            records = json.load(file)

    record = find_mps_record(records, recording_id)

    if record is None:
        raise ValueError(
            f"Recording not found in metadata: {recording_id}"
        )

    reference_text = record.get("promptText")

    if not reference_text:
        raise ValueError(
            f"Reference passage not found for: {recording_id}"
        )

    if not audio_path.is_file():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    # --------------------------------------------------
    # 2. Audio preprocessing
    # --------------------------------------------------

    print("\n[1/4] Preprocessing audio...")

    preprocessed = preprocess_audio(
        audio_path,
        apply_noise_reduction=False
    )

    recording_duration = preprocessed["duration_seconds"]
    silence_intervals = preprocessed["silence_intervals"]

    reading_duration = calculate_reading_duration(
        audio_duration=recording_duration,
        silence_intervals=silence_intervals
    )

    # --------------------------------------------------
    # 3. Whisper transcription
    # --------------------------------------------------

    print("\n[2/4] Running Whisper transcription...")

    # Reuse the model passed into this function.
    # Do not load Whisper again here.

    transcription_result = transcribe_audio(
        model,
        audio_path
    )

    transcribed_text = transcription_result["transcribed_text"]

    # --------------------------------------------------
    # 4. Reading-performance metrics
    # --------------------------------------------------

    print("\n[3/4] Extracting reading-performance metrics...")

    reading_metrics = align_words(
        reference_text,
        transcribed_text
    )

    spoken_word_count = len(
        normalize_text(transcribed_text)
    )

    estimated_wpm = calculate_wpm(
        spoken_word_count,
        reading_duration
    )

    repetitions = detect_repetitions(
        transcribed_text
    )

    hesitations = detect_hesitations(
        silence_intervals=silence_intervals,
        recording_duration=recording_duration,
        min_pause_duration=0.5
    )

    # --------------------------------------------------
    # 5. Acoustic feature extraction
    # --------------------------------------------------

    print("\n[4/4] Extracting acoustic features...")

    acoustic_features = extract_acoustic_features(
        audio=preprocessed["audio"],
        sr=preprocessed["sample_rate"],
        silence_intervals=silence_intervals
    )

    # --------------------------------------------------
    # 6. Combine structured outputs
    # --------------------------------------------------

    session_output = {
        "recording_id": recording_id,
        "audio_file_path": str(audio_path),
        "reference_text": reference_text,

        "reading_duration_sec": round(
            reading_duration, 2
        ),

        "words_total": reading_metrics["words_total"],
        "words_correct": reading_metrics["words_correct"],
        "words_incorrect": reading_metrics["words_incorrect"],

        "reading_speed_wpm": (
            round(estimated_wpm, 2)
            if estimated_wpm is not None
            else None
        ),

        "reading_accuracy": reading_metrics["reading_accuracy"],

        # Fluency scoring is not yet validated.
        "fluency_score": None,

        "word_error_rate": reading_metrics["word_error_rate"],

        "hesitation_count": hesitations["hesitation_count"],
        "repetition_count": repetitions["repetition_count"],

        "omission_count": reading_metrics["omission_count"],
        "substitution_count": reading_metrics["substitution_count"],
        "insertion_count": reading_metrics["insertion_count"],

        "transcribed_text": transcribed_text,

        "pitch_mean": acoustic_features["pitch_mean_hz"],
        "energy_mean": acoustic_features["rms_mean"],
        "avg_pause_duration": hesitations["avg_pause_duration"],

        "mfcc_mean": acoustic_features["mfcc_mean"],
        "mfcc_std": acoustic_features["mfcc_std"],

        "pitch_std": acoustic_features["pitch_std_hz"],
        "energy_std": acoustic_features["rms_std"],

        "recording_duration_sec": recording_duration,

        "internal_quiet_interval_count": (
            acoustic_features["pause_count"]
        ),

        "alignment": reading_metrics["alignment"],
        "flagged_pauses": hesitations["flagged_pauses"],

        "measurement_status": {
            "reading_metrics": "ASR-based estimates",
            "hesitations": (
                "Candidate pauses, not confirmed hesitations"
            ),
            "repetitions": "Consecutive-word heuristic",
            "fluency_score": "Not yet validated"
        }
    }

    # --------------------------------------------------
    # 7. Save structured output
    # --------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            session_output,
            file,
            indent=4,
            ensure_ascii=False
        )

    # --------------------------------------------------
    # 8. Display summary
    # --------------------------------------------------

    print("\nC1 - Processing Completed")
    print("=========================")

    print("Recording:", recording_id)

    print(
        "Reading duration:",
        session_output["reading_duration_sec"],
        "seconds"
    )

    print(
        "Reading speed:",
        session_output["reading_speed_wpm"],
        "WPM"
    )

    print(
        "Estimated reading accuracy:",
        round(session_output["reading_accuracy"], 2),
        "%"
    )

    print(
        "Passage-relative WER:",
        round(session_output["word_error_rate"], 4)
    )

    print(
        "Candidate hesitations:",
        session_output["hesitation_count"]
    )

    print(
        "Detected repetitions:",
        session_output["repetition_count"]
    )

    print(
        "Mean pitch:",
        session_output["pitch_mean"],
        "Hz"
    )

    print(
        "Mean RMS energy:",
        session_output["energy_mean"]
    )

    print("Fluency score: Pending validation")

    print("\nStructured output saved to:")
    print(output_path)

    return session_output


# --------------------------------------------------
# Single-recording execution
# --------------------------------------------------

def main():

    if len(sys.argv) > 2:
        raise SystemExit(
            "Usage: python main.py [recording_id]"
        )

    recording_id = (
        sys.argv[1]
        if len(sys.argv) == 2
        else DEFAULT_RECORDING_ID
    )

    model = load_whisper_model()

    return process_recording(
        recording_id=recording_id,
        model=model
    )


if __name__ == "__main__":
    main()
