"""Unit tests for the EventProducer (local mode)."""

from services.event_collector.app.producer import EventProducer


def test_producer_local_mode_init() -> None:
    producer = EventProducer(local_mode=True)
    assert producer.local_mode is True
    assert producer._kafka_producer is None


def test_send_event_buffers_locally() -> None:
    producer = EventProducer(local_mode=True)
    event = {"user_id": "u1", "item_id": "i1", "event_type": "click"}
    result = producer.send_event(event)
    assert result is True
    assert len(producer.get_buffer()) == 1
    assert producer.get_buffer()[0]["event"] == event


def test_topic_routing_purchase() -> None:
    producer = EventProducer(local_mode=True)
    producer.send_event({"user_id": "u1", "item_id": "i1", "event_type": "purchase"})
    assert producer.get_buffer()[0]["topic"] == "purchases"


def test_topic_routing_search() -> None:
    producer = EventProducer(local_mode=True)
    producer.send_event({"user_id": "u1", "event_type": "search", "query": "shoes"})
    assert producer.get_buffer()[0]["topic"] == "searches"


def test_topic_routing_default() -> None:
    producer = EventProducer(local_mode=True)
    producer.send_event({"user_id": "u1", "item_id": "i1", "event_type": "click"})
    assert producer.get_buffer()[0]["topic"] == "user-events"


def test_send_batch() -> None:
    producer = EventProducer(local_mode=True)
    events = [
        {"user_id": "u1", "item_id": f"i{n}", "event_type": "view"}
        for n in range(5)
    ]
    accepted = producer.send_batch(events)
    assert accepted == 5
    assert len(producer.get_buffer()) == 5


def test_stats() -> None:
    producer = EventProducer(local_mode=True)
    producer.send_event({"user_id": "u1", "event_type": "view"})
    producer.send_event({"user_id": "u2", "event_type": "click"})
    stats = producer.get_stats()
    assert stats["total_events_processed"] == 2
    assert stats["buffer_size"] == 2
    assert stats["local_mode"] is True
    assert stats["kafka_connected"] is False


def test_buffer_max_size() -> None:
    producer = EventProducer(local_mode=True)
    # Buffer has maxlen=10000; send more than that
    for i in range(100):
        producer.send_event({"user_id": f"u{i}", "event_type": "view"})
    assert len(producer.get_buffer()) == 100
    assert producer.get_stats()["total_events_processed"] == 100


def test_close_without_kafka() -> None:
    producer = EventProducer(local_mode=True)
    producer.close()  # Should not raise
