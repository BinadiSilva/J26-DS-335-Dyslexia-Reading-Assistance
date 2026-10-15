
from pathlib import Path

from audio_preprocessing import (
    preprocess_audio,
    save_processed_audio
)


def main():
    audio_path = Path(
          "datasets/c1/mps/audios/3b30f_EN-OL-RC-234_2.wav"
    )

    result = preprocess_audio(audio_path)

    print("\nC1 - Audio Preprocessing Results")
    print("--------------------------------")
    print("Sample rate:", result["sample_rate"], "Hz")
    print(
        "Duration:",
        round(result["duration_seconds"], 2),
        "seconds"
    )
    print("Number of samples:", result["number_of_samples"])

    output_path = Path(
         "datasets/c1/test_audio/processed_3b30f.wav"
    )

    save_processed_audio(
        result["audio"],
        result["sample_rate"],
        output_path
    )

    print("Processed audio saved:", output_path)


if __name__ == "__main__":
    main()
