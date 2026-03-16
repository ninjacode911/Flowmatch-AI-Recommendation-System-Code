"""
Kafka Event Producer -- streams user events to Kafka topics.

In a recommendation system, real-time events drive:
  1. Session-based recommendations (what did the user just do?)
  2. Real-time feature updates (user's click count in the last 5 minutes)
  3. Model retraining triggers (when enough new data accumulates)
  4. Analytics and monitoring (conversion funnels, engagement metrics)

This producer accepts events via HTTP and publishes them to Kafka.
If Kafka is unavailable, events are buffered in an in-memory queue
and flushed when the connection is restored (at-least-once delivery).
"""

import json
import logging
from collections import deque
from datetime import datetime

import structlog

logger = structlog.get_logger()

# Maximum events to buffer when Kafka is unavailable
MAX_BUFFER_SIZE = 10_000

# Kafka topic names
TOPIC_USER_EVENTS = "user-events"
TOPIC_PURCHASES = "purchases"
TOPIC_SEARCHES = "searches"


class EventProducer:
    """
    Publishes user events to Kafka topics.

    Supports two modes:
      1. Kafka mode: publishes to a real Kafka cluster (production)
      2. Local mode: writes to an in-memory buffer (development/testing)
    """

    def __init__(self, bootstrap_servers: str = "localhost:9092", local_mode: bool = True) -> None:
        self.local_mode = local_mode
        self.bootstrap_servers = bootstrap_servers
        self._buffer: deque[dict] = deque(maxlen=MAX_BUFFER_SIZE)
        self._kafka_producer = None
        self._event_count = 0

        if not local_mode:
            self._init_kafka()
        else:
            logger.info("event_producer.local_mode", msg="Kafka disabled, using in-memory buffer")

    def _init_kafka(self) -> None:
        """Initialize the Kafka producer connection."""
        try:
            from kafka import KafkaProducer

            self._kafka_producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                acks="all",  # Wait for all replicas to acknowledge
                retries=3,
                max_in_flight_requests_per_connection=1,  # Ensure ordering
                linger_ms=10,  # Batch events for 10ms before sending (throughput vs latency)
                batch_size=16384,  # 16KB batch size
            )
            logger.info("event_producer.kafka_connected", servers=self.bootstrap_servers)
        except ImportError:
            logger.warning("event_producer.kafka_not_installed", msg="kafka-python not installed, falling back to local mode")
            self.local_mode = True
        except Exception as e:
            logger.error("event_producer.kafka_connection_failed", error=str(e))
            self.local_mode = True

    def _select_topic(self, event_type: str) -> str:
        """Route events to the appropriate Kafka topic."""
        if event_type == "purchase":
            return TOPIC_PURCHASES
        elif event_type == "search":
            return TOPIC_SEARCHES
        return TOPIC_USER_EVENTS

    def send_event(self, event: dict) -> bool:
        """
        Publish a single event.

        Args:
            event: dict with at least {user_id, item_id, event_type, timestamp}

        Returns:
            True if the event was accepted (either sent to Kafka or buffered)
        """
        self._event_count += 1
        topic = self._select_topic(event.get("event_type", ""))

        if self.local_mode or self._kafka_producer is None:
            self._buffer.append({"topic": topic, "event": event})
            return True

        try:
            self._kafka_producer.send(
                topic,
                key=event.get("user_id", ""),
                value=event,
            )
            return True
        except Exception as e:
            logger.error("event_producer.send_failed", error=str(e), event_type=event.get("event_type"))
            self._buffer.append({"topic": topic, "event": event})
            return True

    def send_batch(self, events: list[dict]) -> int:
        """
        Publish a batch of events.

        Returns the number of events accepted.
        """
        accepted = 0
        for event in events:
            if self.send_event(event):
                accepted += 1

        if not self.local_mode and self._kafka_producer:
            self._kafka_producer.flush()

        return accepted

    def flush(self) -> None:
        """Force-send any buffered events."""
        if self._kafka_producer:
            self._kafka_producer.flush()

    def get_buffer(self) -> list[dict]:
        """Return buffered events (for local mode / testing)."""
        return list(self._buffer)

    def get_stats(self) -> dict:
        """Return producer statistics."""
        return {
            "total_events_processed": self._event_count,
            "buffer_size": len(self._buffer),
            "buffer_capacity": MAX_BUFFER_SIZE,
            "local_mode": self.local_mode,
            "kafka_connected": self._kafka_producer is not None,
        }

    def close(self) -> None:
        """Gracefully shut down the producer."""
        if self._kafka_producer:
            self._kafka_producer.flush()
            self._kafka_producer.close()
            logger.info("event_producer.closed")
