
import numpy as np
import whisper

from audio_preprocessing import preprocess_audio


MODEL_NAME = "base.en"


def load_whisper_model():
    """
    Load a pretrained English Whisper model.
    """
    print("Loading Whisper model...")

    model = whisper.load_model(
        MODEL_NAME,
        device="cpu"
    )

    print("Whisper model loaded successfully!")

    return model


def transcribe_audio(model, audio_path):
    """
    Transcribe a reading recording using Whisper.
    """

    # Step 1: Preprocess the audio
    result = preprocess_audio(audio_path)

    # Step 2: Get the processed audio
    audio = result["audio"].astype(np.float32)

    # Step 3: Convert speech to text
    transcription = model.transcribe(
        audio,
        language="en",
        task="transcribe",
        fp16=False,
        verbose=False
    )

    return {
        "transcribed_text": transcription["text"].strip(),
        "duration_seconds": result["duration_seconds"],
        "sample_rate": result["sample_rate"]
    }
