
import librosa
import numpy as np


def extract_mfcc(audio, sr, n_mfcc=13):
    """
    Extract MFCC features from speech audio.
    Return mean and standard deviation of each coefficient.
    """

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=1024,
        hop_length=256
    )

    return {
        "mfcc_mean": np.mean(mfcc, axis=1).tolist(),
        "mfcc_std": np.std(mfcc, axis=1).tolist()
    }


def extract_pitch(audio, sr):
    """
    Estimate fundamental frequency (F0) using librosa.pyin.

    Unvoiced frames are excluded from pitch statistics.
    """

    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio,
        sr=sr,
        fmin=65,
        fmax=1000,
        frame_length=1024,
        hop_length=256
    )

    valid_pitch = f0[np.isfinite(f0)]

    if len(valid_pitch) == 0:
        return {
            "pitch_mean_hz": None,
            "pitch_std_hz": None
        }

    return {
        "pitch_mean_hz": float(np.mean(valid_pitch)),
        "pitch_std_hz": float(np.std(valid_pitch))
    }


def extract_energy(audio, sr):
    """
    Calculate Root Mean Square (RMS) energy.
    """

    rms = librosa.feature.rms(
        y=audio,
        frame_length=1024,
        hop_length=256
    )[0]

    return {
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms))
    }



def extract_pause_features(
    silence_intervals,
    recording_duration
):
    """
    Summarize internal quiet intervals.

    Exclude leading and trailing quiet intervals.
    """

    tolerance = 0.02

    internal_intervals = [
        interval
        for interval in silence_intervals
        if interval["start"] > tolerance
        and interval["end"] < recording_duration - tolerance
    ]

    durations = [
        interval["duration"]
        for interval in internal_intervals
    ]

    if not durations:
        return {
            "pause_count": 0,
            "mean_pause_sec": 0.0,
            "max_pause_sec": 0.0,
            "total_pause_sec": 0.0
        }

    return {
        "pause_count": len(durations),
        "mean_pause_sec": float(np.mean(durations)),
        "max_pause_sec": float(np.max(durations)),
        "total_pause_sec": float(np.sum(durations))
    }



def extract_acoustic_features(
    audio,
    sr,
    silence_intervals
):
    """
    Combine acoustic feature extraction functions.
    """

    features = {}

    features.update(extract_mfcc(audio, sr))
    features.update(extract_pitch(audio, sr))
    features.update(extract_energy(audio, sr))

    recording_duration = len(audio) / sr

    features.update(
        extract_pause_features(
            silence_intervals,
            recording_duration
        )
    )

    return features
