
import json
import re
from pathlib import Path


# ==================================================
# 1. TEXT NORMALIZATION
# ==================================================

def normalize_text(text):
    """
    Convert text to lowercase, remove punctuation,
    and split it into words.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


# ==================================================
# 2. WORD ALIGNMENT AND READING METRICS
# ==================================================

def align_words(reference_text, transcribed_text):
    """
    Compare the reference passage with Whisper output.

    Uses word-level edit distance and prefers exact
    word matches when multiple alignments are possible.

    These are ASR-based estimates, not confirmed
    child reading errors.
    """

    reference = normalize_text(reference_text)
    hypothesis = normalize_text(transcribed_text)

    n = len(reference)
    m = len(hypothesis)

    # Each entry stores:
    # (edit cost, negative exact matches,
    #  substitution count, operations)
    dp = [[None] * (m + 1) for _ in range(n + 1)]

    dp[0][0] = (0, 0, 0, [])

    # Initialize deletions
    for i in range(1, n + 1):
        previous = dp[i - 1][0]

        dp[i][0] = (
            previous[0] + 1,
            previous[1],
            previous[2],
            previous[3] + [
                (reference[i - 1], None, "deletion")
            ]
        )

    # Initialize insertions
    for j in range(1, m + 1):
        previous = dp[0][j - 1]

        dp[0][j] = (
            previous[0] + 1,
            previous[1],
            previous[2],
            previous[3] + [
                (None, hypothesis[j - 1], "insertion")
            ]
        )

    # Fill alignment table
    for i in range(1, n + 1):
        for j in range(1, m + 1):

            candidates = []
            previous = dp[i - 1][j - 1]

            # Exact match
            if reference[i - 1] == hypothesis[j - 1]:

                candidates.append((
                    previous[0],
                    previous[1] - 1,
                    previous[2],
                    previous[3] + [
                        (
                            reference[i - 1],
                            hypothesis[j - 1],
                            "correct"
                        )
                    ]
                ))

            # Substitution
            else:

                candidates.append((
                    previous[0] + 1,
                    previous[1],
                    previous[2] + 1,
                    previous[3] + [
                        (
                            reference[i - 1],
                            hypothesis[j - 1],
                            "substitution"
                        )
                    ]
                ))

            # Deletion / omission
            previous = dp[i - 1][j]

            candidates.append((
                previous[0] + 1,
                previous[1],
                previous[2],
                previous[3] + [
                    (reference[i - 1], None, "deletion")
                ]
            ))

            # Insertion
            previous = dp[i][j - 1]

            candidates.append((
                previous[0] + 1,
                previous[1],
                previous[2],
                previous[3] + [
                    (None, hypothesis[j - 1], "insertion")
                ]
            ))

            # Prefer:
            # 1. Minimum edit distance
            # 2. Maximum exact word matches
            # 3. Minimum substitutions
            dp[i][j] = min(
                candidates,
                key=lambda x: (x[0], x[1], x[2])
            )

    operations = dp[n][m][3]

    alignment = [
        {
            "reference": ref,
            "transcribed": hyp,
            "operation": operation
        }
        for ref, hyp, operation in operations
    ]

    correct = sum(
        item["operation"] == "correct"
        for item in alignment
    )

    substitutions = sum(
        item["operation"] == "substitution"
        for item in alignment
    )

    deletions = sum(
        item["operation"] == "deletion"
        for item in alignment
    )

    insertions = sum(
        item["operation"] == "insertion"
        for item in alignment
    )

    word_error_rate = (
        (substitutions + deletions + insertions) / n
        if n > 0 else None
    )

    reading_accuracy = (
        (correct / n) * 100
        if n > 0 else None
    )

    return {
        "words_total": n,
        "words_correct": correct,
        "words_incorrect": substitutions + deletions,
        "substitution_count": substitutions,
        "omission_count": deletions,
        "insertion_count": insertions,
        "word_error_rate": word_error_rate,
        "reading_accuracy": reading_accuracy,
        "alignment": alignment
    }


# ==================================================
# 3. READING SPEED (WPM)
# ==================================================

def calculate_wpm(word_count, duration_seconds):
    """
    Calculate estimated words per minute.
    """

    if duration_seconds <= 0:
        return None

    return (word_count / duration_seconds) * 60


# ==================================================
# 4. READING DURATION
# ==================================================

def calculate_reading_duration(
    audio_duration,
    silence_intervals,
    tolerance=0.02
):
    """
    Estimate reading duration by excluding
    leading and trailing quiet intervals.

    Internal pauses remain included.
    """

    if audio_duration <= 0:
        return 0.0

    leading_silence = 0.0
    trailing_silence = 0.0

    for interval in silence_intervals:

        # Leading silence
        if interval["start"] <= tolerance:
            leading_silence = max(
                leading_silence,
                interval["end"]
            )

        # Trailing silence
        if interval["end"] >= audio_duration - tolerance:
            trailing_silence = max(
                trailing_silence,
                audio_duration - interval["start"]
            )

    reading_duration = (
        audio_duration
        - leading_silence
        - trailing_silence
    )

    return max(0.0, reading_duration)


# ==================================================
# 5. REPETITION DETECTION
# ==================================================

def detect_repetitions(transcribed_text):
    """
    Identify consecutive repeated words
    in Whisper's transcription.

    This is an initial heuristic.
    """

    words = normalize_text(transcribed_text)

    repetition_count = 0
    repeated_words = []

    for i in range(1, len(words)):

        if words[i] == words[i - 1]:
            repetition_count += 1
            repeated_words.append(words[i])

    return {
        "repetition_count": repetition_count,
        "repeated_words": repeated_words
    }


# ==================================================
# 6. CANDIDATE HESITATION DETECTION
# ==================================================

def detect_hesitations(
    silence_intervals,
    recording_duration,
    min_pause_duration=0.5,
    tolerance=0.02
):
    """
    Flag internal quiet intervals lasting at least
    min_pause_duration seconds.

    These are candidate hesitation pauses,
    not confirmed reading hesitations.
    """

    flagged_pauses = []

    for interval in silence_intervals:

        # Exclude leading quiet interval
        if interval["start"] <= tolerance:
            continue

        # Exclude trailing quiet interval
        if interval["end"] >= recording_duration - tolerance:
            continue

        # Flag long internal quiet intervals
        if interval["duration"] >= min_pause_duration:
            flagged_pauses.append(interval)

    durations = [
        pause["duration"]
        for pause in flagged_pauses
    ]

    return {
        "hesitation_count": len(flagged_pauses),
        "avg_pause_duration": (
            sum(durations) / len(durations)
            if durations else 0.0
        ),
        "flagged_pauses": flagged_pauses
    }


def calculate_fluency_score(
    accuracy_score,
    speed_score,
    pause_score,
    accuracy_weight=0.5,
    speed_weight=0.3,
    pause_weight=0.2
):
    """
    Calculate an experimental composite fluency score.

    All three inputs must already be normalized
    to a 0-100 scale.

    This is not a clinically validated score.
    """

    scores = (
        accuracy_score,
        speed_score,
        pause_score
    )

    if any(score is None for score in scores):
        return None

    if any(not 0 <= score <= 100 for score in scores):
        raise ValueError(
            "All fluency inputs must be between 0 and 100."
        )

    total_weight = (
        accuracy_weight
        + speed_weight
        + pause_weight
    )

    if abs(total_weight - 1.0) > 1e-9:
        raise ValueError(
            "Fluency weights must add up to 1."
        )

    fluency_score = (
        accuracy_weight * accuracy_score
        + speed_weight * speed_score
        + pause_weight * pause_score
    )

    return round(fluency_score, 2)

# ==================================================
# 7. FIND RECORDING IN MPS DATASET
# ==================================================

def find_mps_record(records, recording_id):
    """
    Find a recording using filename or recording ID.

    Does not assume that the metadata identifier
    is stored in a field named 'id'.
    """

    for record in records:

        if not isinstance(record, dict):
            continue

        for key, value in record.items():

            key_lower = key.lower()

            if not any(
                term in key_lower
                for term in (
                    "id", "file", "audio", "name", "path"
                )
            ):
                continue

            if isinstance(value, str):

                filename = Path(
                    value.replace("\\", "/")
                ).name

                if (
                    value == recording_id
                    or filename == recording_id
                    or filename == f"{recording_id}.wav"
                ):
                    return record

    return None


# ==================================================
# 8. MAIN - TEST USING REAL MPS RECORDING
# ==================================================

if __name__ == "__main__":

    from audio_preprocessing import preprocess_audio

    from speech_recognition import (
        load_whisper_model,
        transcribe_audio
    )

    # Grade 3 reading recording
    recording_id = "3b30f_EN-OL-RC-234_2"

    metadata_path = Path(
        "datasets/c1/mps/selected_20.json"
    )

    audio_path = Path(
        f"datasets/c1/mps/audios/{recording_id}.wav"
    )

    # --------------------------------------------------
    # Load MPS metadata
    # --------------------------------------------------

    with open(
        metadata_path,
        "r",
        encoding="utf-8"
    ) as file:
        records = json.load(file)

    record = find_mps_record(
        records,
        recording_id
    )

    if record is None:

        print("Available metadata fields:")

        if records:
            print(list(records[0].keys()))

        raise ValueError(
            "Recording not found in MPS metadata."
        )

    reference_text = record.get("promptText")

    if not reference_text:
        raise ValueError(
            "Reference passage (promptText) not found."
        )

    # --------------------------------------------------
    # Audio preprocessing
    # --------------------------------------------------

    print("\nPreprocessing audio...")

    preprocessing_result = preprocess_audio(
        audio_path,
        apply_noise_reduction=False
    )

    duration = preprocessing_result[
        "duration_seconds"
    ]

    silence_intervals = preprocessing_result[
        "silence_intervals"
    ]

    reading_duration = calculate_reading_duration(
        audio_duration=duration,
        silence_intervals=silence_intervals
    )

    # --------------------------------------------------
    # Whisper transcription
    # --------------------------------------------------

    print("\nLoading Whisper model...")

    model = load_whisper_model()

    print("\nTranscribing MPS recording...")

    transcription_result = transcribe_audio(
        model,
        audio_path
    )

    transcribed_text = transcription_result[
        "transcribed_text"
    ]

    # --------------------------------------------------
    # Reading metrics
    # --------------------------------------------------

    metrics = align_words(
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

    repetition_results = detect_repetitions(
        transcribed_text
    )

    hesitation_results = detect_hesitations(
        silence_intervals=silence_intervals,
        recording_duration=duration,
        min_pause_duration=0.5
    )

    # --------------------------------------------------
    # Display results
    # --------------------------------------------------

    print("\nC1 - MPS Reading Metrics")
    print("------------------------")

    print("\nReference Passage:")
    print(reference_text)

    print("\nWhisper Transcription:")
    print(transcribed_text)

    print("\nDuration:")
    print(
        "Full Recording Duration:",
        round(duration, 2),
        "seconds"
    )

    print(
        "Estimated Reading Duration:",
        round(reading_duration, 2),
        "seconds"
    )

    print(
        "Estimated Spoken WPM:",
        round(estimated_wpm, 2)
        if estimated_wpm is not None
        else None
    )

    print("\nReading Metrics:")

    for key, value in metrics.items():
        if key != "alignment":
            print(f"{key}: {value}")

    print("\nWord Alignment:")

    for item in metrics["alignment"]:
        print(item)

    print("\nRepetition Detection:")
    print(
        "Estimated Repetition Count:",
        repetition_results["repetition_count"]
    )
    print(
        "Repeated Words:",
        repetition_results["repeated_words"]
    )

    print("\nCandidate Hesitation Detection:")
    print(
        "Count:",
        hesitation_results["hesitation_count"]
    )
    print(
        "Average flagged pause duration:",
        round(
            hesitation_results["avg_pause_duration"],
            3
        ),
        "seconds"
    )

    
    print("\nFlagged Pause Intervals:")

    for pause in hesitation_results["flagged_pauses"]:
        print(
            f"{pause['start']:.2f}s - "
            f"{pause['end']:.2f}s "
            f"({pause['duration']:.3f}s)"
        )

    