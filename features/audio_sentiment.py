"""
features/audio_sentiment.py
===========================
Ingests live audio streams (e.g. Fed press conferences) and converts speech to text
using OpenAI's Whisper model, then routes the text into the FinBERT/VADER pipelines
to generate tradable sentiment signals before official transcripts are published.
"""

import logging
from typing import Any

try:
    import whisper  # pyright: ignore[reportMissingImports]
except ImportError:
    whisper = None

WhisperModule = Any


class AudioSentimentPipeline:
    def __init__(self, model_size: str = "base"):
        self.logger = logging.getLogger(__name__)
        self.model_size = model_size
        self.model: WhisperModule | None = None

    def load_model(self) -> bool:
        if whisper is None:
            self.logger.error("Whisper library is not installed. Run `pip install openai-whisper`")
            return False

        self.logger.info(f"Loading Whisper model ({self.model_size})...")
        self.model = whisper.load_model(self.model_size)
        return True

    def process_audio(self, audio_path: str) -> str:
        """Transcribe an audio file to text."""
        if self.model is None and not self.load_model():
            raise RuntimeError("Whisper model could not be loaded")

        if self.model is None:
            raise RuntimeError("Whisper model was not initialized")

        self.logger.info(f"Transcribing audio: {audio_path}")
        result = self.model.transcribe(audio_path)
        transcript = result["text"].strip()
        self.logger.info(f"Transcription complete (Length: {len(transcript)})")
        return transcript

    def generate_sentiment_signal(self, audio_path: str) -> float:
        """
        Transcribes the audio and routes the text through the FinBERT pipeline.
        Returns a float between -1.0 (Hawkish/Bearish) and 1.0 (Dovish/Bullish).
        """
        transcript = self.process_audio(audio_path)
        if not transcript:
            self.logger.warning("Empty transcript - returning neutral sentiment")
            return 0.0

        self.logger.info("Routing transcript to FinBERT/SentimentPipeline...")
        try:
            from features.finbert_sentiment import SentimentPipeline

            pipe = SentimentPipeline()
            score = float(pipe.score_headlines([transcript]))
            return float(max(-1.0, min(1.0, score)))
        except Exception as exc:
            self.logger.error("Sentiment scoring failed (%s) - returning neutral", exc)
            return 0.0
