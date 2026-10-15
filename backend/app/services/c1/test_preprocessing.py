
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

    
    print("\nSilence Detection Results")
    print("-------------------------")

    print("Silence count:", result["silence_count"])

    print(
        "Total silence duration:",
        round(result["total_silence_duration"], 2),
        "seconds"
    )

    print("\nDetected quiet intervals:")

    for interval in result["silence_intervals"]:
        print(
            f'Start: {interval["start"]:.2f}s | '
            f'End: {interval["end"]:.2f}s | '
            f'Duration: {interval["duration"]:.2f}s'
        )


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
