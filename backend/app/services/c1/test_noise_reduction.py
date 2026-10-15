
from pathlib import Path

from audio_preprocessing import (
    preprocess_audio,
    save_processed_audio
)


def main():
    audio_path = Path(
        "datasets/c1/mps/audios/3b30f_EN-OL-RC-234_2.wav"
    )

    # Process audio with noise reduction disabled
    original_result = preprocess_audio(
        audio_path,
        apply_noise_reduction=False
    )

    # Process the same audio with noise reduction enabled
    reduced_result = preprocess_audio(
        audio_path,
        apply_noise_reduction=True
    )

    output_folder = Path("datasets/c1/test_audio")

    original_output = output_folder / "normalized_3b30f.wav"
    reduced_output = output_folder / "noise_reduced_3b30f.wav"

    save_processed_audio(
        original_result["audio"],
        original_result["sample_rate"],
        original_output
    )

    save_processed_audio(
        reduced_result["audio"],
        reduced_result["sample_rate"],
        reduced_output
    )

    print("\nC1 - Noise Reduction Test")
    print("-------------------------")
    print("Original duration:", original_result["duration_seconds"])
    print("Reduced duration:", reduced_result["duration_seconds"])

    print("\nWithout noise reduction:", original_output)
    print("With noise reduction:", reduced_output)

    print(
        "\nNoise reduction applied:",
        reduced_result["noise_reduction_applied"]
    )


if __name__ == "__main__":
    main()
