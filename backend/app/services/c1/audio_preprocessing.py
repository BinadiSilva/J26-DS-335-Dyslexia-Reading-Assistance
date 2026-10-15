
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr


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


def reduce_background_noise(audio, sr):
    """
    Apply mild noise reduction to audio.

    Uses non-stationary noise reduction without
    requiring a separate noise-only recording.
    """

    reduced_audio = nr.reduce_noise(
        y=audio,
        sr=sr,
        stationary=False,
        prop_decrease=0.6
    )

    return reduced_audio.astype(np.float32)


def detect_silence(audio, sr, top_db=30, min_silence_duration=0.2):
    """
    Estimate quiet intervals in an audio recording.

    This function does not remove audio.
    """

    # Find intervals containing non-silent audio
    non_silent_intervals = librosa.effects.split(
        audio,
        top_db=top_db
    )

    total_samples = len(audio)
    silence_intervals = []

    previous_end = 0

    # Find silent gaps between non-silent intervals
    for start, end in non_silent_intervals:

        if start > previous_end:
            duration = (start - previous_end) / sr

            if duration >= min_silence_duration:
                silence_intervals.append({
                    "start": previous_end / sr,
                    "end": start / sr,
                    "duration": duration
                })

        previous_end = end

    # Check silence at the end of the recording
    if previous_end < total_samples:
        duration = (total_samples - previous_end) / sr

        if duration >= min_silence_duration:
            silence_intervals.append({
                "start": previous_end / sr,
                "end": total_samples / sr,
                "duration": duration
            })

    return silence_intervals
 


def preprocess_audio(audio_path, apply_noise_reduction=False):
    """
    Main preprocessing pipeline for C1.

    Noise reduction is optional because it can
    change speech and acoustic characteristics.
    """

    # Step 1: Load audio, convert to mono and resample
    audio, sr = load_audio(audio_path)

    # Step 2: Calculate original duration
    duration_seconds = len(audio) / sr

    # Step 3: Normalize the audio
    normalized_audio = normalize_audio(audio)

    # Step 4: Detect quiet intervals before noise reduction
    silence_intervals = detect_silence(
        normalized_audio,
        sr
    )

    # Step 5: Apply optional noise reduction
    if apply_noise_reduction:
        processed_audio = reduce_background_noise(
            normalized_audio,
            sr
        )
    else:
        processed_audio = normalized_audio.copy()

    total_silence_duration = sum(
        interval["duration"]
        for interval in silence_intervals
    )

    return {
        "audio": processed_audio,
        "sample_rate": sr,
        "duration_seconds": duration_seconds,
        "number_of_samples": len(processed_audio),
        "silence_intervals": silence_intervals,
        "silence_count": len(silence_intervals),
        "total_silence_duration": total_silence_duration,
        "noise_reduction_applied": apply_noise_reduction
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
