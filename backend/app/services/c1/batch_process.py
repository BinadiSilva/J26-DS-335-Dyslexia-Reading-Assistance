
import json
from pathlib import Path

from main import (
    METADATA_PATH,
    AUDIO_FOLDER,
    OUTPUT_FOLDER,
    process_recording,
)
from speech_recognition import load_whisper_model


def is_completed(recording_id):
    """
    Check whether a valid output already exists.
    This prevents overwriting completed recordings.
    """
    output_path = OUTPUT_FOLDER / f"{recording_id}_output.json"

    if not output_path.is_file():
        return False

    try:
        with open(output_path, "r", encoding="utf-8") as file:
            result = json.load(file)

        required_fields = [
            "recording_id",
            "transcribed_text",
            "reading_accuracy",
            "word_error_rate",
            "reading_speed_wpm",
            "mfcc_mean",
            "mfcc_std",
            "pitch_mean",
            "energy_mean",
        ]

        return (
            isinstance(result, dict)
            and result.get("recording_id") == recording_id
            and all(field in result for field in required_fields)
            and isinstance(result.get("mfcc_mean"), list)
            and len(result["mfcc_mean"]) == 13
        )

    except (OSError, ValueError, TypeError):
        return False


def main():

    print("\nC1 - Automatic Batch Processing")
    print("================================")

    # 1. Load selected MPS metadata once.
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError("Expected a list of MPS records.")

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    pending = []
    skipped = []
    missing_audio = []

    # 2. Identify completed and remaining recordings.
    for record in records:

        audio_path_text = record.get("audioPath")

        if not audio_path_text:
            raise ValueError(
                "A metadata record has no audioPath."
            )

        filename = Path(
            str(audio_path_text).replace("\\", "/")
        ).name

        recording_id = Path(filename).stem
        audio_path = AUDIO_FOLDER / filename

        if is_completed(recording_id):
            skipped.append(recording_id)

        elif not audio_path.is_file():
            missing_audio.append(recording_id)

        else:
            pending.append(recording_id)

    print("Selected recordings:", len(records))
    print("Already completed:", len(skipped))
    print("Pending processing:", len(pending))
    print("Missing audio:", len(missing_audio))

    if not pending:
        print("\nNo recordings need processing.")
        return

    # 3. Load Whisper only once.
    print("\nLoading Whisper for batch processing...")
    model = load_whisper_model()

    successful = []
    failed = []

    # 4. Process each remaining recording.
    for index, recording_id in enumerate(pending, start=1):

        print("\n--------------------------------")
        print(f"Processing {index}/{len(pending)}")
        print("Recording:", recording_id)
        print("--------------------------------")

        try:
            process_recording(
                recording_id=recording_id,
                model=model,
                records=records
            )

            successful.append(recording_id)

        except Exception as error:
            failed.append((recording_id, str(error)))

            print("\nERROR processing:", recording_id)
            print("Reason:", error)

    # 5. Display final summary.
    print("\n================================")
    print("C1 - Batch Processing Summary")
    print("================================")

    print("Selected:", len(records))
    print("Previously completed:", len(skipped))
    print("Newly processed:", len(successful))
    print("Missing audio:", len(missing_audio))
    print("Failed:", len(failed))
    print("Total completed:", len(skipped) + len(successful))

    if missing_audio:
        print("\nMissing audio files:")
        for recording_id in missing_audio:
            print("-", recording_id)

    if failed:
        print("\nFailed recordings:")
        for recording_id, error in failed:
            print(f"- {recording_id}: {error}")

    print("\nOutput folder:")
    print(OUTPUT_FOLDER)


if __name__ == "__main__":
    main()
