
from pathlib import Path

from audio_preprocessing import preprocess_audio
from acoustic_features import extract_acoustic_features


def main():

    # Grade 3 reading recording
    audio_path = Path(
        "datasets/c1/mps/audios/3b30f_EN-OL-RC-234_2.wav"
    )

    print("Loading and preprocessing audio...")

    # Use standardized audio without noise reduction
    result = preprocess_audio(
        audio_path,
        apply_noise_reduction=False
    )

    print("Extracting acoustic features...")

    features = extract_acoustic_features(
        audio=result["audio"],
        sr=result["sample_rate"],
        silence_intervals=result["silence_intervals"]
    )

    print("\nC1 - Acoustic Feature Results")
    print("-----------------------------")

    print("Audio:", audio_path.name)

    print("\nMFCC:")
    print("Mean:", features["mfcc_mean"])
    print("Std:", features["mfcc_std"])

    print("\nPitch:")
    print("Mean (Hz):", features["pitch_mean_hz"])
    print("Std (Hz):", features["pitch_std_hz"])

    print("\nEnergy:")
    print("RMS Mean:", features["rms_mean"])
    print("RMS Std:", features["rms_std"])

    print("\nPause Information:")
    print("Count:", features["pause_count"])
    print("Mean duration:", features["mean_pause_sec"])
    print("Maximum duration:", features["max_pause_sec"])
    print("Total duration:", features["total_pause_sec"])


if __name__ == "__main__":
    main()
