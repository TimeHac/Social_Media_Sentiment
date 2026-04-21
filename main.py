#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║   SOCIAL MEDIA SENTIMENT & TREND PIPELINE                       ║
║   End-to-End Data Engineering Project                           ║
║                                                                  ║
║   Tech Stack: Python · Pandas · NumPy · SQLite · Matplotlib     ║
║               NLP (rule-based transformer) · Kafka (sim)         ║
║               Medallion Architecture · Elasticsearch (sim)       ║
╚══════════════════════════════════════════════════════════════════╝

Run the complete pipeline:
  python main.py

Or run individual stages:
  python 01_generate_data.py
  python 02_etl_pipeline.py
  python 03_analytics.py
  python 04_streaming.py

Or run the DAG orchestrator:
  python dag_runner.py
"""

import os
import sys
import time
import logging
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

for d in ["data/raw", "data/processed", "data/bronze", "data/silver",
          "data/gold", "data/warehouse", "logs", "reports"]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    filename="logs/pipeline.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║      SOCIAL MEDIA SENTIMENT & TREND PIPELINE                    ║
║      End-to-End Data Engineering Project                        ║
║                                                                  ║
║  Stages:                                                         ║
║   [1] Data Generation    — Twitter/Reddit API sim, Faker        ║
║   [2] ETL / Medallion    — Bronze → Silver → Gold               ║
║   [3] NLP Analytics      — Transformer sim, Elasticsearch sim   ║
║   [4] Streaming Engine   — Kafka sim, windowed aggregations     ║
║                                                                  ║
║  Architecture:                                                   ║
║   Raw CSV → Bronze (ingest) → Silver (clean/enrich)             ║
║          → Gold (aggregate) → SQLite Warehouse → Dashboard      ║
║          + Kafka Stream → Real-time Sentiment Monitor            ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""


def run_stage(stage_name: str, script_path: str) -> bool:
    print(f"\n{'═'*65}")
    print(f"  STAGE: {stage_name}")
    print(f"{'═'*65}")
    start = time.time()
    try:
        with open(script_path) as f:
            code = f.read()
        exec(compile(code, script_path, "exec"),
             {"__name__": "__main__", "__file__": os.path.abspath(script_path)})
        elapsed = time.time() - start
        print(f"\n  ✅ {stage_name} complete in {elapsed:.1f}s")
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n  ❌ {stage_name} FAILED after {elapsed:.1f}s: {e}")
        logging.error(f"Stage '{stage_name}' failed: {e}")
        return False


def print_summary():
    import pandas as pd

    print("\n" + "█" * 65)
    print("  PIPELINE OUTPUT SUMMARY")
    print("█" * 65)

    outputs = {
        "Raw Data":       [
            "data/raw/users.csv", "data/raw/posts.csv",
            "data/raw/hashtag_trends.csv", "data/raw/api_logs.csv",
        ],
        "Bronze Layer":   [
            "data/bronze/users.csv", "data/bronze/posts.csv",
            "data/bronze/hashtag_trends.csv", "data/bronze/api_logs.csv",
        ],
        "Silver Layer":   [
            "data/silver/users.csv", "data/silver/posts.csv",
            "data/silver/hashtag_trends.csv", "data/silver/api_logs.csv",
        ],
        "Gold Layer":     [
            "data/gold/topic_sentiment.csv",
            "data/gold/daily_trend.csv",
            "data/gold/platform_perf.csv",
            "data/gold/top_influencers.csv",
            "data/gold/hourly_pattern.csv",
        ],
        "Data Warehouse": [
            "data/warehouse/sentiment_warehouse.db",
            "data/warehouse/trending_hashtags.csv",
            "data/warehouse/user_segment_analysis.csv",
            "data/warehouse/primetime_analysis.csv",
        ],
        "Reports":        [
            "reports/analytics_dashboard.png",
            "reports/data_quality_report.txt",
            "reports/streaming_report.json",
            "reports/search_index_report.json",
        ],
    }

    total_kb = 0.0
    for section, files in outputs.items():
        print(f"\n  [{section}]")
        for f in files:
            if os.path.exists(f):
                kb = os.path.getsize(f) / 1024
                total_kb += kb
                extra = ""
                if f.endswith(".csv"):
                    try:
                        rows = len(pd.read_csv(f))
                        extra = f"  ({rows:,} rows)"
                    except Exception:
                        pass
                print(f"    ✅ {f:<60} {kb:7.1f} KB{extra}")
            else:
                print(f"    ❌ {f}  (missing)")

    print(f"\n  Total output size : {total_kb/1024:.2f} MB")
    print("█" * 65)


if __name__ == "__main__":
    pipeline_start = datetime.now()
    print(BANNER)
    print(f"  Started: {pipeline_start.strftime('%Y-%m-%d %H:%M:%S')}")

    stages = [
        ("Data Generation",         "01_generate_data.py"),
        ("ETL / Medallion Pipeline", "02_etl_pipeline.py"),
        ("NLP Analytics Engine",     "03_analytics.py"),
        ("Kafka Streaming Engine",   "04_streaming.py"),
    ]

    results = []
    for name, script in stages:
        ok = run_stage(name, script)
        results.append((name, ok))

    print_summary()

    elapsed = (datetime.now() - pipeline_start).total_seconds()
    passed  = sum(1 for _, ok in results if ok)
    failed  = sum(1 for _, ok in results if not ok)

    print(f"\n  Pipeline finished in {elapsed:.1f}s")
    print(f"  Stages: {passed} passed  |  {failed} failed")

    if failed == 0:
        print("\n  🎉 ALL STAGES PASSED — Pipeline is production-ready!")
    else:
        print("\n  ⚠  Some stages failed. Check logs/pipeline.log for details.")
