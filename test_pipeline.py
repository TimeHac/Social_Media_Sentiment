"""
test_pipeline.py
────────────────────────────────────────────────────────────────────
Unit tests for the Social Media Sentiment Pipeline.
Run with:  python test_pipeline.py
"""

import os
import sys
import json
import unittest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ── Tests for Data Generation ────────────────────────────────────

class TestDataGeneration(unittest.TestCase):

    def setUp(self):
        from importlib import import_module
        # Import helpers directly from script via exec
        self.gen_src = open("01_generate_data.py").read()
        ns = {}
        exec(compile(self.gen_src, "01_generate_data.py", "exec"), ns)
        self.ns = ns

    def test_mask_email(self):
        mask = self.ns["mask_email"]
        result = mask("testuser@gmail.com")
        self.assertTrue(result.startswith("t"))
        self.assertIn("@", result)
        self.assertIn("***", result)

    def test_sentiment_score_range(self):
        ss = self.ns["sentiment_score"]
        for label in ["positive", "negative", "neutral"]:
            score = ss(label)
            self.assertGreaterEqual(score, -1.0)
            self.assertLessEqual(score, 1.0)

    def test_generate_post_text_returns_tuple(self):
        fn = self.ns["generate_post_text"]
        text, tags = fn("Technology", "positive")
        self.assertIsInstance(text, str)
        self.assertIsInstance(tags, list)
        self.assertGreater(len(text), 5)

    def test_users_csv_exists(self):
        self.assertTrue(os.path.exists("data/raw/users.csv"))

    def test_posts_csv_row_count(self):
        if not os.path.exists("data/raw/posts.csv"):
            self.skipTest("posts.csv not generated yet")
        df = pd.read_csv("data/raw/posts.csv")
        self.assertGreater(len(df), 1000)

    def test_no_unmasked_email_in_users(self):
        if not os.path.exists("data/raw/users.csv"):
            self.skipTest("users.csv not generated yet")
        df = pd.read_csv("data/raw/users.csv")
        # All emails should contain *** (PII masked)
        email_col = df["email_masked"].dropna()
        self.assertTrue(email_col.str.contains(r"\*\*\*").all())


# ── Tests for ETL Pipeline ────────────────────────────────────────

class TestETLPipeline(unittest.TestCase):

    def test_bronze_files_exist(self):
        for name in ["users", "posts", "hashtag_trends", "api_logs"]:
            self.assertTrue(
                os.path.exists(f"data/bronze/{name}.csv"),
                f"Bronze file missing: {name}.csv"
            )

    def test_bronze_metadata_columns(self):
        if not os.path.exists("data/bronze/posts.csv"):
            self.skipTest("Bronze posts.csv missing")
        df = pd.read_csv("data/bronze/posts.csv")
        for col in ["_ingested_at", "_source_file", "_pipeline_version"]:
            self.assertIn(col, df.columns)

    def test_silver_engagement_score(self):
        if not os.path.exists("data/silver/posts.csv"):
            self.skipTest("Silver posts.csv missing")
        df = pd.read_csv("data/silver/posts.csv")
        self.assertIn("engagement_score", df.columns)
        self.assertTrue((df["engagement_score"] >= 0).all())

    def test_silver_sentiment_clipped(self):
        if not os.path.exists("data/silver/posts.csv"):
            self.skipTest("Silver posts.csv missing")
        df = pd.read_csv("data/silver/posts.csv")
        self.assertTrue((df["sentiment_score"] >= -1.0).all())
        self.assertTrue((df["sentiment_score"] <= 1.0).all())

    def test_gold_topic_sentiment_exists(self):
        self.assertTrue(os.path.exists("data/gold/topic_sentiment.csv"))

    def test_gold_platform_perf_exists(self):
        self.assertTrue(os.path.exists("data/gold/platform_perf.csv"))

    def test_warehouse_db_exists(self):
        self.assertTrue(os.path.exists("data/warehouse/sentiment_warehouse.db"))

    def test_warehouse_tables(self):
        if not os.path.exists("data/warehouse/sentiment_warehouse.db"):
            self.skipTest("Warehouse DB missing")
        import sqlite3
        conn = sqlite3.connect("data/warehouse/sentiment_warehouse.db")
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()
        for t in ["dim_users", "fact_posts", "fact_hashtag_trends", "stg_api_logs"]:
            self.assertIn(t, tables, f"Table missing: {t}")


# ── Tests for Analytics ───────────────────────────────────────────

class TestAnalytics(unittest.TestCase):

    def test_dashboard_png_exists(self):
        self.assertTrue(os.path.exists("reports/analytics_dashboard.png"))

    def test_data_quality_report_exists(self):
        self.assertTrue(os.path.exists("reports/data_quality_report.txt"))

    def test_data_quality_report_content(self):
        if not os.path.exists("reports/data_quality_report.txt"):
            self.skipTest("DQ report missing")
        with open("reports/data_quality_report.txt") as f:
            content = f.read()
        self.assertIn("DATA QUALITY REPORT", content)
        self.assertIn("POSTS", content)

    def test_nlp_inference_csv_exists(self):
        self.assertTrue(os.path.exists("data/processed/nlp_inference_sample.csv"))

    def test_nlp_labels_valid(self):
        path = "data/processed/nlp_inference_sample.csv"
        if not os.path.exists(path):
            self.skipTest("NLP inference CSV missing")
        df = pd.read_csv(path)
        valid = {"positive", "negative", "neutral"}
        self.assertTrue(set(df["nlp_label"].unique()).issubset(valid))

    def test_search_index_report_exists(self):
        self.assertTrue(os.path.exists("reports/search_index_report.json"))

    def test_trending_hashtags_csv_exists(self):
        self.assertTrue(os.path.exists("data/warehouse/trending_hashtags.csv"))


# ── Tests for Streaming ───────────────────────────────────────────

class TestStreaming(unittest.TestCase):

    def test_streaming_report_exists(self):
        self.assertTrue(os.path.exists("reports/streaming_report.json"))

    def test_streaming_report_structure(self):
        path = "reports/streaming_report.json"
        if not os.path.exists(path):
            self.skipTest("Streaming report missing")
        with open(path) as f:
            report = json.load(f)
        self.assertIn("kafka_stats", report)
        self.assertIn("sentiment_processing", report)
        self.assertIn("trending_leaderboard", report)

    def test_streaming_processed_count(self):
        path = "reports/streaming_report.json"
        if not os.path.exists(path):
            self.skipTest("Streaming report missing")
        with open(path) as f:
            report = json.load(f)
        processed = report["sentiment_processing"].get("total_processed", 0)
        self.assertGreater(processed, 0)

    def test_kafka_partition_offset_monotonic(self):
        """Unit test: Kafka partition offset increments correctly."""
        # Import directly from 04_streaming
        ns = {}
        exec(compile(open("04_streaming.py").read(), "04_streaming.py", "exec"), ns)
        KafkaPartition = ns["KafkaPartition"]
        p = KafkaPartition(0)
        for i in range(5):
            off = p.append({"data": i})
            self.assertEqual(off, i)
        self.assertEqual(p.current_offset, 5)


# ── Tests for DAG Orchestrator ────────────────────────────────────

class TestDAG(unittest.TestCase):

    def test_dag_log_exists(self):
        logs = [f for f in os.listdir("logs") if f.startswith("dag_run_")]
        self.assertGreater(len(logs), 0, "No DAG run logs found in logs/")

    def test_pipeline_task_retry(self):
        """Unit test: PipelineTask retries on failure then succeeds."""
        ns = {}
        exec(compile(open("dag_runner.py").read(), "dag_runner.py", "exec"), ns)
        PipelineTask = ns["PipelineTask"]

        call_count = {"n": 0}

        def flaky():
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise ValueError("Simulated failure")

        task = PipelineTask("flaky_task", flaky, retries=3)
        result = task.run()
        self.assertTrue(result)
        self.assertEqual(call_count["n"], 3)

    def test_pipeline_task_exhausts_retries(self):
        """Unit test: PipelineTask fails after all retries."""
        ns = {}
        exec(compile(open("dag_runner.py").read(), "dag_runner.py", "exec"), ns)
        PipelineTask  = ns["PipelineTask"]
        TaskStatus    = ns["TaskStatus"]

        def always_fails():
            raise RuntimeError("Always fails")

        task = PipelineTask("bad_task", always_fails, retries=2, retry_delay_sec=0)
        result = task.run()
        self.assertFalse(result)
        self.assertEqual(task.status, TaskStatus.FAILED)


# ── Runner ────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("🧪 Running Social Media Sentiment Pipeline Tests...\n")
    loader  = unittest.TestLoader()
    suite   = unittest.TestSuite()

    for cls in [TestDataGeneration, TestETLPipeline,
                TestAnalytics, TestStreaming, TestDAG]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
