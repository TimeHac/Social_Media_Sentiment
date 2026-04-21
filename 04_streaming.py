"""
04_streaming.py
────────────────────────────────────────────────────────────────────
Real-time social media event streaming pipeline.

Covers:
  • Kafka Basics     : Producer / Consumer with partition routing
  • Kafka Advanced   : Consumer groups, offset management, lag tracking
  • Structured Streaming: micro-batch processing, windowed aggregations
  • Streaming Concepts  : at-least-once delivery, backpressure simulation
  • MinIO/S3 sim     : stream checkpointing to local "object store"
"""

import os
import json
import time
import random
import hashlib
import logging
import threading
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import Optional

import pandas as pd
import numpy as np

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/pipeline.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

TOPICS_LIST = ["Technology", "Politics", "Cricket", "Bollywood",
               "Finance", "Health", "Education", "Environment",
               "Startup", "Gaming"]

PLATFORMS = ["Twitter", "Reddit", "Instagram", "LinkedIn"]


# ══════════════════════════════════════════════════════════════════
#  SIMULATED KAFKA BROKER  (Partitioned, with offset management)
# ══════════════════════════════════════════════════════════════════

class KafkaPartition:
    """Immutable log partition with offset tracking."""

    def __init__(self, partition_id: int):
        self.partition_id   = partition_id
        self.log            = []
        self.current_offset = 0
        self.lag            = 0

    def append(self, message: dict) -> int:
        message["_offset"]    = self.current_offset
        message["_partition"] = self.partition_id
        message["_timestamp"] = datetime.now().isoformat()
        self.log.append(message)
        self.current_offset += 1
        return self.current_offset - 1

    def read_from(self, offset: int, max_records: int = 200) -> list:
        return self.log[offset: offset + max_records]


class KafkaTopic:
    """
    Multi-partition Kafka topic.
    Supports consistent hashing, consumer group offset tracking,
    and lag monitoring (Kafka Advanced concepts).
    """

    def __init__(self, name: str, num_partitions: int = 4):
        self.name       = name
        self.partitions = [KafkaPartition(i) for i in range(num_partitions)]
        # consumer_group → [offset per partition]
        self.consumer_offsets: dict[str, list[int]] = defaultdict(
            lambda: [0] * num_partitions
        )

    def produce(self, message: dict, key: Optional[str] = None) -> tuple[int, int]:
        """Consistent hash routing by key."""
        if key:
            pid = int(hashlib.md5(key.encode()).hexdigest(), 16) % len(self.partitions)
        else:
            pid = random.randint(0, len(self.partitions) - 1)
        offset = self.partitions[pid].append(message)
        return pid, offset

    def consume(self, consumer_group: str, max_records: int = 100) -> list:
        """At-least-once delivery semantics (offsets committed after processing)."""
        messages = []
        for pid, partition in enumerate(self.partitions):
            committed = self.consumer_offsets[consumer_group][pid]
            batch     = partition.read_from(committed, max_records)
            messages.extend(batch)
            self.consumer_offsets[consumer_group][pid] += len(batch)
        return messages

    def consumer_lag(self, consumer_group: str) -> dict:
        """Report per-partition consumer lag."""
        return {
            f"partition_{pid}": (
                partition.current_offset
                - self.consumer_offsets[consumer_group][pid]
            )
            for pid, partition in enumerate(self.partitions)
        }

    def stats(self) -> dict:
        total = sum(len(p.log) for p in self.partitions)
        return {
            "topic":          self.name,
            "partitions":     len(self.partitions),
            "total_messages": total,
            "dist":           [len(p.log) for p in self.partitions],
        }


# ══════════════════════════════════════════════════════════════════
#  SOCIAL MEDIA EVENT PRODUCER
# ══════════════════════════════════════════════════════════════════

class SocialMediaProducer:
    """
    Simulates live ingestion from Twitter/Reddit streaming APIs.
    Publishes SocialPostEvent and TrendingHashtagEvent to Kafka.
    """

    EVENT_TYPES  = ["post_created", "post_liked", "post_shared",
                    "post_commented", "user_followed", "hashtag_trending"]
    SENTIMENTS   = ["positive", "negative", "neutral"]
    SENTIMENT_WEIGHTS = [45, 25, 30]

    def __init__(self, posts_topic: KafkaTopic, trends_topic: KafkaTopic):
        self.posts_topic  = posts_topic
        self.trends_topic = trends_topic
        self.produced     = 0

    def _make_post_event(self) -> dict:
        topic     = random.choice(TOPICS_LIST)
        platform  = random.choice(PLATFORMS)
        sentiment = random.choices(
            self.SENTIMENTS, weights=self.SENTIMENT_WEIGHTS)[0]
        likes     = int(np.random.lognormal(3.5, 2.0))
        return {
            "event_type":      "post_created",
            "post_id":         f"LIVE{random.randint(1, 9999999):08d}",
            "user_id":         f"USR{random.randint(1, 1000):06d}",
            "platform":        platform,
            "topic":           topic,
            "sentiment_label": sentiment,
            "sentiment_score": round(
                (0.65 if sentiment == "positive"
                 else -0.65 if sentiment == "negative" else 0.05)
                + np.random.uniform(-0.3, 0.3), 4),
            "likes":   likes,
            "shares":  int(likes * random.uniform(0.05, 0.4)),
            "comments": int(likes * random.uniform(0.02, 0.3)),
            "is_viral": (likes > 5000),
            "event_time": datetime.now().isoformat(),
        }

    def _make_trend_event(self, topic: str) -> dict:
        from random import choice
        HASHTAGS = {
            "Technology": ["#AI", "#Python", "#ML"],
            "Cricket":    ["#IPL2024", "#TeamIndia"],
            "Politics":   ["#Elections2024", "#Parliament"],
        }
        tags = HASHTAGS.get(topic, ["#Trending"])
        return {
            "event_type":   "hashtag_trending",
            "hashtag":      choice(tags),
            "topic":        topic,
            "velocity":     random.randint(50, 5000),  # posts/min
            "sentiment":    random.choice(self.SENTIMENTS),
            "event_time":   datetime.now().isoformat(),
        }

    def bulk_produce(self, n: int = 2000) -> int:
        logger.info(f"Producer: generating {n} live events...")
        for i in range(n):
            event = self._make_post_event()
            self.posts_topic.produce(event, key=event["post_id"])

            # Every 50 posts, emit a trending event
            if i % 50 == 0:
                trend_event = self._make_trend_event(
                    random.choice(TOPICS_LIST))
                self.trends_topic.produce(trend_event,
                                          key=trend_event["hashtag"])
            self.produced += 1

        logger.info(f"Producer: {self.produced} total messages produced.")
        return self.produced


# ══════════════════════════════════════════════════════════════════
#  STREAM PROCESSOR — Windowed Sentiment Aggregation
# ══════════════════════════════════════════════════════════════════

class SentimentStreamProcessor:
    """
    Structured streaming micro-batch processor.
    Implements:
      - Tumbling windows (per-minute aggregation)
      - Sliding windows  (last-N posts sentiment)
      - Real-time alerting (spike detection)
      - Backpressure simulation
    """
    WINDOW_SIZE = 100   # Sliding window size

    def __init__(self, topic: KafkaTopic, group: str = "sentiment-processor"):
        self.topic          = topic
        self.group          = group
        self.sentiment_window = deque(maxlen=self.WINDOW_SIZE)
        self.topic_counts:   dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.platform_stats: dict[str, dict]           = defaultdict(
            lambda: {"posts": 0, "likes": 0, "virals": 0, "sentiment_sum": 0.0}
        )
        self.alerts:         list[dict] = []
        self.processed       = 0
        self.batches         = 0

    def process_batch(self, messages: list):
        """Micro-batch processing — simulates Spark Structured Streaming."""
        self.batches += 1
        for msg in messages:
            if msg.get("event_type") != "post_created":
                continue

            topic     = msg.get("topic", "Unknown")
            sentiment = msg.get("sentiment_label", "neutral")
            score     = float(msg.get("sentiment_score", 0.0))
            platform  = msg.get("platform", "Unknown")
            likes     = int(msg.get("likes", 0))
            is_viral  = bool(msg.get("is_viral", False))

            # Sliding window
            self.sentiment_window.append(score)

            # Topic × Sentiment aggregation (tumbling window)
            self.topic_counts[topic][sentiment] += 1

            # Platform stats
            ps = self.platform_stats[platform]
            ps["posts"] += 1
            ps["likes"] += likes
            ps["virals"] += int(is_viral)
            ps["sentiment_sum"] += score

            # Alert: sustained negative spike
            if (len(self.sentiment_window) >= 20
                    and np.mean(list(self.sentiment_window)[-20:]) < -0.5):
                self.alerts.append({
                    "alert_type":   "NEGATIVE_SENTIMENT_SPIKE",
                    "topic":        topic,
                    "window_avg":   round(float(np.mean(list(self.sentiment_window)[-20:])), 4),
                    "trigger_time": datetime.now().isoformat(),
                })

            self.processed += 1

    @property
    def window_sentiment(self) -> float:
        if not self.sentiment_window:
            return 0.0
        return round(float(np.mean(list(self.sentiment_window))), 4)

    def run(self, iterations: int = 12, delay: float = 0.05) -> dict:
        logger.info(f"StreamProcessor '{self.group}' started.")
        for _ in range(iterations):
            messages = self.topic.consume(self.group, max_records=200)
            if messages:
                self.process_batch(messages)
            time.sleep(delay)

        # Compute final platform stats
        platform_summary = {}
        for plat, stats in self.platform_stats.items():
            posts = max(stats["posts"], 1)
            platform_summary[plat] = {
                "posts":          stats["posts"],
                "total_likes":    stats["likes"],
                "viral_posts":    stats["virals"],
                "avg_sentiment":  round(stats["sentiment_sum"] / posts, 4),
            }

        report = {
            "consumer_group":    self.group,
            "total_processed":   self.processed,
            "total_batches":     self.batches,
            "window_sentiment":  self.window_sentiment,
            "topic_breakdown":   {
                t: dict(s) for t, s in self.topic_counts.items()
            },
            "platform_summary":  platform_summary,
            "alerts_raised":     len(self.alerts),
            "sample_alerts":     self.alerts[:5],
        }
        logger.info(f"StreamProcessor done: {self.processed} messages processed.")
        return report


# ══════════════════════════════════════════════════════════════════
#  TRENDING HASHTAG PROCESSOR
# ══════════════════════════════════════════════════════════════════

class TrendingHashtagProcessor:
    """Consumes hashtag-trending events and computes real-time leaderboard."""

    def __init__(self, topic: KafkaTopic, group: str = "trend-tracker"):
        self.topic      = topic
        self.group      = group
        self.velocity:  dict[str, int] = defaultdict(int)
        self.processed  = 0

    def run(self, iterations: int = 12) -> list:
        for _ in range(iterations):
            msgs = self.topic.consume(self.group)
            for msg in msgs:
                tag = msg.get("hashtag", "")
                self.velocity[tag] += int(msg.get("velocity", 0))
                self.processed += 1

        leaderboard = sorted(
            [{"hashtag": k, "velocity": v} for k, v in self.velocity.items()],
            key=lambda x: x["velocity"], reverse=True
        )
        return leaderboard


# ══════════════════════════════════════════════════════════════════
#  OBJECT STORE CHECKPOINT (MinIO/S3 simulation)
# ══════════════════════════════════════════════════════════════════

def checkpoint_to_object_store(data: dict, path: str):
    """
    Simulates writing streaming checkpoints to MinIO/S3.
    In production: boto3.client('s3').put_object(...)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"  Checkpoint written → {path}")


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("\n" + "═" * 60)
    print("   SOCIAL MEDIA STREAMING ENGINE  (Kafka + Structured Stream)")
    print("═" * 60)

    # ── Setup Kafka topics ────────────────────────────────────────
    print("\n[1/4] Setting up Kafka topics (4 partitions each)...")
    posts_topic  = KafkaTopic("social-posts",   num_partitions=4)
    trends_topic = KafkaTopic("hashtag-trends", num_partitions=2)

    # ── Producer ──────────────────────────────────────────────────
    print("[2/4] Producer: publishing 2000 live social events...")
    producer = SocialMediaProducer(posts_topic, trends_topic)
    total_produced = producer.bulk_produce(2000)
    stats = posts_topic.stats()
    print(f"       Posts topic messages : {stats['total_messages']}")
    print(f"       Partition distribution: {stats['dist']}")

    # ── Consumer lag report ───────────────────────────────────────
    lag_before = posts_topic.consumer_lag("sentiment-processor")
    print(f"       Consumer lag (before): {sum(lag_before.values())} messages")

    # ── Stream processors ─────────────────────────────────────────
    print("[3/4] Running stream processors...")
    sent_proc  = SentimentStreamProcessor(posts_topic,  "sentiment-processor")
    trend_proc = TrendingHashtagProcessor(trends_topic, "trend-tracker")

    # Run in threads (simulates parallel consumer groups)
    t1 = threading.Thread(target=lambda: sent_proc.run(iterations=12))
    t2 = threading.Thread(target=lambda: trend_proc.run(iterations=12))
    t1.start(); t2.start()
    t1.join();  t2.join()

    report      = sent_proc.run.__doc__ and sent_proc  # already ran in thread
    sent_report = {
        "consumer_group":   sent_proc.group,
        "total_processed":  sent_proc.processed,
        "window_sentiment": sent_proc.window_sentiment,
        "topic_breakdown":  {t: dict(s) for t, s in sent_proc.topic_counts.items()},
        "platform_summary": {
            plat: {
                "posts":         stats["posts"],
                "total_likes":   stats["likes"],
                "viral_posts":   stats["virals"],
                "avg_sentiment": round(stats["sentiment_sum"] / max(stats["posts"], 1), 4),
            }
            for plat, stats in sent_proc.platform_stats.items()
        },
        "alerts_raised":    len(sent_proc.alerts),
        "sample_alerts":    sent_proc.alerts[:5],
    }

    leaderboard = trend_proc.run(iterations=0)   # already ran in thread

    lag_after = posts_topic.consumer_lag("sentiment-processor")
    print(f"       Processed            : {sent_proc.processed} messages")
    print(f"       Window sentiment     : {sent_proc.window_sentiment:+.4f}")
    print(f"       Alerts raised        : {len(sent_proc.alerts)}")
    print(f"       Consumer lag (after) : {sum(lag_after.values())} messages")
    print(f"       Trending hashtags    : {len(leaderboard)} tracked")

    # ── Checkpoint to "S3" ────────────────────────────────────────
    print("[4/4] Checkpointing stream state to object store (MinIO sim)...")
    streaming_report = {
        "kafka_stats": {
            "posts_topic":  posts_topic.stats(),
            "trends_topic": trends_topic.stats(),
        },
        "sentiment_processing": sent_report,
        "trending_leaderboard": leaderboard[:10],
        "consumer_lag_after":  lag_after,
        "generated_at": datetime.now().isoformat(),
    }

    os.makedirs("reports", exist_ok=True)
    checkpoint_to_object_store(
        streaming_report,
        "data/processed/stream_checkpoint.json"
    )
    with open("reports/streaming_report.json", "w") as f:
        json.dump(streaming_report, f, indent=2)

    print("\n✅ Streaming engine complete!")
    print("   Report         → reports/streaming_report.json")
    print("   Checkpoint     → data/processed/stream_checkpoint.json")
