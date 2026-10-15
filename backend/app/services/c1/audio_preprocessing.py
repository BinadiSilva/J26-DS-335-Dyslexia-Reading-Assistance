
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


TARGET_SAMPLE_RATE = 16000


def load_audio(audio_path, target_sr=TARGET_SAMPLE_RATE):
    """
    Load audio, convert it to mono, and resample to 16 kHz.
    """

    audio_path = Path(audio_path)

    if not audio_path.is_file():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    audio, sr = librosa.load(
        str(audio_path),
        sr=target_sr,
        mono=True
    )

    if len(audio) == 0:
        raise ValueError("The audio recording is empty.")

    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains invalid sample values.")

    return audio, sr


def normalize_audio(audio, peak_target=0.95):
    """
    Normalize the peak amplitude without removing pauses.
    """

    peak = np.max(np.abs(audio))

    if peak > 0:
        audio = (audio / peak) * peak_target

    return audio


def preprocess_audio(audio_path):
    """
    Main preprocessing pipeline for C1.
    """

    # Step 1: Load and standardize audio
    audio, sr = load_audio(audio_path)

    # Step 2: Record duration before further processing
    duration_seconds = len(audio) / sr

    # Step 3: Normalize audio
    processed_audio = normalize_audio(audio)

    return {
        "audio": processed_audio,
        "sample_rate": sr,
        "duration_seconds": duration_seconds,
        "number_of_samples": len(processed_audio)
    }


def save_processed_audio(audio, sr, output_path):
    """
    Save processed audio as a 16-bit WAV file.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    sf.write(
        str(output_path),
        audio,
        sr,
        subtype="PCM_16"
    )
