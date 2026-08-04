import argparse
import json
import logging
import os
import sys
from datetime import UTC, datetime

# Ensure the root directory is in sys.path so we can import 'features'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from kafka import KafkaConsumer, KafkaProducer
    KAFKA = True
except ImportError:
    KAFKA = False

# Import our existing FinBERT/Ollama sentiment pipeline
from features.finbert_sentiment import SentimentPipeline

# Zero-shot classification for event tagging
try:
    from transformers import pipeline
    TRANSFORMERS = True
except ImportError:
    TRANSFORMERS = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class StreamingNewsEnricher:
    """
    Consumes raw news, computes FinBERT sentiment and DistilBERT event classification,
    and publishes the enriched payload to be ingested by the TimescaleDB consumer.
    """
    def __init__(self, raw_topic="forex.raw_news", enriched_topic="forex.news", servers="localhost:9092"):
        self.raw_topic = raw_topic
        self.enriched_topic = enriched_topic
        self.servers = servers

        self.sentiment_pipeline = SentimentPipeline(prefer_backend="ollama", ollama_model="gemma4:e2b")

        if TRANSFORMERS:
            logging.info("Loading Zero-Shot event classifier...")
            self.event_classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-3")
        else:
            logging.warning("transformers not installed. Event classification will be stubbed.")
            self.event_classifier = None

        if KAFKA:
            self.consumer = KafkaConsumer(
                self.raw_topic,
                bootstrap_servers=self.servers,
                group_id="news_enricher",
                value_deserializer=lambda x: json.loads(x.decode('utf-8'))
            )
            self.producer = KafkaProducer(
                bootstrap_servers=self.servers,
                value_serializer=lambda x: json.dumps(x).encode('utf-8')
            )
        else:
            logging.warning("kafka-python not installed. Running in mock mode.")

    def is_high_impact(self, headline: str) -> bool:
        """Determines if the headline refers to a high impact macroeconomic event."""
        # Fast path keyword match
        keywords = ["nfp", "nonfarm payrolls", "cpi", "inflation", "fomc", "rate decision", "fed", "ecb"]
        lower_head = headline.lower()
        if any(k in lower_head for k in keywords):
            return True

        if self.event_classifier:
            try:
                res = self.event_classifier(headline, candidate_labels=["macroeconomic policy decision", "routine market news"])
                if res['labels'][0] == "macroeconomic policy decision" and res['scores'][0] > 0.6:
                    return True
            except Exception as e:
                logging.error(f"Classification error: {e}")
        return False

    def process_message(self, msg_dict: dict):
        headline = msg_dict.get("headline", "")
        if not headline:
            return

        # 1. FinBERT / Ollama Sentiment & Embedding
        sentiment_score = 0.0
        embedding = []
        try:
            score = self.sentiment_pipeline.score_headlines([headline])
            sentiment_score = float(score)
            # Emulate an embedding output for now (dim=8)
            embedding = [sentiment_score] * 8
        except Exception as e:
            logging.error(f"Sentiment error: {e}")

        # 2. DistilBERT Event Classification
        high_impact = self.is_high_impact(headline)

        # 3. Publish Enriched Payload
        enriched_msg = {
            "time": msg_dict.get("time", datetime.now(UTC).isoformat()),
            "pair": msg_dict.get("pair", "EURUSD"),
            "headline": headline,
            "sentiment_score": sentiment_score,
            "finbert_embedding": embedding,
            "is_high_impact": high_impact
        }

        if KAFKA:
            self.producer.send(self.enriched_topic, enriched_msg)
        else:
            logging.info(f"Mock Output: {enriched_msg}")

    def run(self):
        logging.info(f"Starting News Enricher: {self.raw_topic} -> {self.enriched_topic}")
        if not KAFKA:
            # Mock run
            self.process_message({"headline": "US Nonfarm Payrolls smash expectations, rising 300k", "pair": "EURUSD"})
            return

        for message in self.consumer:
            self.process_message(message.value)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Run without Kafka")
    args = parser.parse_args()

    if args.mock:
        KAFKA = False

    enricher = StreamingNewsEnricher()
    enricher.run()
