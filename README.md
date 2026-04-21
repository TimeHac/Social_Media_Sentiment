# 🐦 Social Media Sentiment & Trend Pipeline

> End-to-End Data Engineering Project — Advanced version of the E-Commerce Pipeline

## 🏗 Architecture

```
SIMULATED APIs  (Twitter / Reddit)
       │
       ▼
┌──────────────┐    ┌─────────────────────────┐    ┌────────────────────────┐
│ 01_generate  │───▶│   02_etl_pipeline        │───▶│  03_analytics          │
│ _data.py     │    │  ┌──────────────────┐    │    │  NLP Transformer sim   │
│ Faker, NumPy │    │  │ BRONZE (raw)     │    │    │  Elasticsearch sim     │
│ Dataclasses  │    │  │ SILVER (clean)   │    │    │  Advanced SQL          │
│ 1K users     │    │  │ GOLD   (agg.)    │    │    │  8-panel Dashboard     │
│ 50K posts    │    │  └──────────────────┘    │    └────────────────────────┘
└──────────────┘    │  SQLite + Star Schema    │
                    └─────────────────────────┘
                               │
                               ▼
               ┌────────────────────────────┐
               │  04_streaming.py           │
               │  Kafka sim (4 partitions)  │
               │  Windowed aggregations     │
               │  Sentiment alerts          │
               │  MinIO/S3 checkpoint sim   │
               └────────────────────────────┘
                               │
                               ▼
               ┌────────────────────────────┐
               │  dag_runner.py             │
               │  Airflow-like DAG          │
               │  Topological sort          │
               │  Retry + dependency skip   │
               └────────────────────────────┘
```

## 📦 Tech Stack

| Layer             | Technology                                 |
|-------------------|--------------------------------------------|
| Language          | Python 3.10+                               |
| Data Manipulation | Pandas, NumPy                              |
| Database          | SQLite (star schema warehouse)             |
| Architecture      | **Medallion (Bronze → Silver → Gold)**     |
| NLP               | Rule-based Transformer sim (HuggingFace-style) |
| Search            | **Elasticsearch-style inverted index**     |
| Visualization     | Matplotlib, Seaborn                        |
| Data Generation   | Faker (en_IN), NumPy distributions         |
| Streaming         | **Kafka simulation (4 partitions)**        |
| Orchestration     | **Custom Airflow DAG with topological sort** |
| Object Store      | MinIO/S3 checkpoint simulation             |
| Logging           | Python logging module + JSON run logs      |
| Testing           | unittest                                   |

## 🆕 What's New vs E-Commerce Pipeline

| Feature                     | E-Commerce (original)     | This Project               |
|-----------------------------|---------------------------|----------------------------|
| Architecture                | Flat ETL                  | **Medallion (Bronze/Silver/Gold)** |
| NLP / ML                    | ❌ None                   | ✅ Transformer-style NLP   |
| Search Engine               | ❌ None                   | ✅ Elasticsearch inverted index |
| Kafka Partitions            | 3                         | **4** + consumer lag report |
| Streaming checkpointing     | ❌ None                   | ✅ MinIO/S3 object store sim |
| PII Masking                 | ❌ None                   | ✅ Email masking on ingest  |
| Data Generator style        | Plain dicts               | **Dataclasses + generators** |
| Consumer Groups             | 1                         | **2 parallel groups**       |

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (all 4 stages)
python main.py

# 3. Or run individual stages
python 01_generate_data.py
python 02_etl_pipeline.py
python 03_analytics.py
python 04_streaming.py

# 4. Run with DAG orchestrator
python dag_runner.py

# 5. Run tests
python test_pipeline.py
```

## 📁 Project Structure

```
Social-Media-Sentiment-Pipeline/
├── main.py                      # Master runner — all stages
├── dag_runner.py                # Airflow-like DAG orchestrator
├── requirements.txt
├── test_pipeline.py             # Unit tests (unittest)
│
├── 01_generate_data.py          # Twitter/Reddit API simulation
├── 02_etl_pipeline.py           # Bronze → Silver → Gold ETL
├── 03_analytics.py              # NLP + Elasticsearch + Dashboard
├── 04_streaming.py              # Kafka streaming + alerts
│
├── data/
│   ├── raw/                     # Simulated API responses (CSV)
│   ├── bronze/                  # Raw ingest + metadata columns
│   ├── silver/                  # Cleaned, enriched data
│   ├── gold/                    # Aggregated BI-ready tables
│   ├── processed/               # NLP inference, stream analysis
│   └── warehouse/               # SQLite DB + analytical views
│
├── reports/
│   ├── analytics_dashboard.png  # 8-panel dark-theme dashboard
│   ├── data_quality_report.txt  # Bronze-layer quality checks
│   ├── streaming_report.json    # Kafka + windowed aggregations
│   └── search_index_report.json # Elasticsearch demo results
│
└── logs/
    ├── pipeline.log             # Full execution log
    └── dag_run_*.json           # Per-run DAG execution logs
```

## 📊 Data Model (Star Schema)

```
dim_users ─────────────────────────────────────────────┐
  user_id PK                                            │
  username, platform, followers, follower_tier          │
  is_influencer, verified, location                     │
                                                        ▼
                                                 fact_posts
dim_date ──────────────────────────────────────► post_id PK
  date_id PK                                     user_id FK
  year, month, day, weekday, is_weekend          sentiment_label, sentiment_score
                                                 engagement_score, is_viral
                                                 platform, topic, post_date

fact_hashtag_trends                              stg_api_logs
  date, hashtag, topic                           log_id PK
  post_count, viral_count                        source, endpoint
  engagement_index                               status_code, response_time_ms
```

## 🔍 Key Features

- **Medallion Architecture** — Bronze (raw) → Silver (clean) → Gold (aggregated)
- **NLP Sentiment Engine** — Simulates HuggingFace transformers pipeline with batch inference
- **Elasticsearch Inverted Index** — Full-text search with BM25-simplified scoring
- **Kafka Simulation** — 4 partitions, consistent-hash routing, consumer lag tracking
- **Parallel Consumer Groups** — `sentiment-processor` + `trend-tracker` run in threads
- **Windowed Aggregations** — Sliding + tumbling windows for real-time sentiment
- **Sentiment Alerts** — Auto-raises alert when window sentiment drops below -0.5
- **PII Masking** — Email addresses masked at ingest (Bronze layer)
- **MinIO/S3 Checkpointing** — Stream state persisted to simulated object store
- **8-Panel Dark Dashboard** — Sentiment trend, heatmap, NLP confidence histogram

## 📈 Sample KPIs

- 1,000 users · 50,000 posts · 15,000 API log events
- Medallion layers: Bronze → Silver → Gold → Warehouse
- NLP model agreement rate: ~65%+ with stored labels
- Kafka: 4 partitions, ~2,000 live events per run
- Search index: 10,000 documents, BM25-style relevance scoring
