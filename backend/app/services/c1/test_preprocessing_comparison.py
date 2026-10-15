
from pathlib import Path

from speech_recognition import (
    load_whisper_model,
    transcribe_audio
)


def main():
    audio_files = {
        "Normalized": Path(
            "datasets/c1/test_audio/normalized_3b30f.wav"
        ),
        "Noise Reduced": Path(
            "datasets/c1/test_audio/noise_reduced_3b30f.wav"
        )
    }

    model = load_whisper_model()

    for name, audio_path in audio_files.items():

        print(f"\n--- {name} Audio ---")

        result = transcribe_audio(
            model,
            audio_path
        )

        print("Transcription:")
        print(result["transcribed_text"])


if __name__ == "__main__":
    main()
