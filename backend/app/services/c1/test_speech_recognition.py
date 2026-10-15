
from pathlib import Path

from speech_recognition import (
    load_whisper_model,
    transcribe_audio
)


def main():

    audio_path = Path(
        "datasets/c1/mps/audios/3b30f_EN-OL-RC-234_2.wav"
    )

    model = load_whisper_model()

    print("\nTranscribing audio...")

    result = transcribe_audio(
        model,
        audio_path
    )

    print("\nC1 - Whisper Transcription Results")
    print("---------------------------------")

    print("Audio:", audio_path.name)
    print("Duration:", round(result["duration_seconds"], 2), "seconds")
    print("Sample rate:", result["sample_rate"], "Hz")

    print("\nWhisper Transcription:")
    print(result["transcribed_text"])


if __name__ == "__main__":
    main()
