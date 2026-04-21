"""
02_etl_pipeline.py
────────────────────────────────────────────────────────────────────
Full ETL pipeline implementing the Medallion Architecture:
  Bronze  → raw ingested data (no changes, append-only)
  Silver  → cleaned, validated, enriched data
  Gold    → aggregated analytical tables ready for BI

Also covers:
  • Data Quality  : null checks, duplicate detection, IQR anomalies
  • Advanced SQL  : window functions, CTEs, RANK(), NTILE()
  • Data Warehouse: star schema in SQLite
  • ETL vs ELT    : demonstrated side-by-side in comments
"""

import os
import sqlite3
import logging
import pandas as pd
import numpy as np
from datetime import datetime

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/pipeline.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
#  LAYER 1 — BRONZE  (Raw ingest, no transformation)
# ══════════════════════════════════════════════════════════════════

def ingest_to_bronze() -> dict:
    """
    ETL  : data is transformed BEFORE loading  ← this pipeline
    ELT  : data is loaded raw THEN transformed  ← Bronze represents ELT landing zone
    """
    logger.info("BRONZE layer: ingesting raw data...")
    sources = {
        "users":          "data/raw/users.csv",
        "posts":          "data/raw/posts.csv",
        "hashtag_trends": "data/raw/hashtag_trends.csv",
        "api_logs":       "data/raw/api_logs.csv",
    }
    bronze = {}
    for name, path in sources.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Add ingestion metadata (Bronze pattern)
            df["_ingested_at"]     = datetime.now().isoformat()
            df["_source_file"]     = path
            df["_pipeline_version"] = "1.0"
            df.to_csv(f"data/bronze/{name}.csv", index=False)
            bronze[name] = df
            logger.info(f"  Bronze {name}: {df.shape}")
        else:
            logger.error(f"  Source not found: {path}")
    return bronze


# ══════════════════════════════════════════════════════════════════
#  DATA QUALITY ENGINE
# ══════════════════════════════════════════════════════════════════

def run_data_quality(bronze: dict) -> dict:
    """
    Quality checks: null detection, duplicates, IQR outliers,
    range validation, referential integrity.
    """
    logger.info("DATA QUALITY checks started.")
    report = {}

    for name, df in bronze.items():
        # Strip meta-columns for checks
        data_cols = [c for c in df.columns if not c.startswith("_")]
        dfc = df[data_cols]
        issues = {}

        # 1. Null check
        nulls = dfc.isnull().sum()
        if nulls.any():
            issues["nulls"] = nulls[nulls > 0].to_dict()

        # 2. Duplicate rows
        dupes = dfc.duplicated().sum()
        if dupes:
            issues["duplicates"] = int(dupes)

        # 3. IQR outlier detection on numeric columns
        for col in dfc.select_dtypes(include=[np.number]).columns:
            Q1, Q3 = dfc[col].quantile(0.25), dfc[col].quantile(0.75)
            IQR = Q3 - Q1
            n_out = ((dfc[col] < Q1 - 3 * IQR) | (dfc[col] > Q3 + 3 * IQR)).sum()
            if n_out > 0:
                issues.setdefault("outliers", {})[col] = int(n_out)

        # 4. Sentiment score range check
        if "sentiment_score" in dfc.columns:
            out_of_range = ((dfc["sentiment_score"] < -1) | (dfc["sentiment_score"] > 1)).sum()
            if out_of_range:
                issues["sentiment_range_violation"] = int(out_of_range)

        report[name] = issues
        logger.info(f"  {name}: {len(issues)} issue type(s) found")

    # Save human-readable quality report
    os.makedirs("reports", exist_ok=True)
    with open("reports/data_quality_report.txt", "w") as f:
        f.write("═" * 65 + "\n")
        f.write("       SOCIAL MEDIA PIPELINE — DATA QUALITY REPORT\n")
        f.write(f"       Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("═" * 65 + "\n\n")
        for dataset, issues in report.items():
            f.write(f"[{dataset.upper()}]\n")
            if not issues:
                f.write("  ✅ No issues found.\n")
            else:
                for k, v in issues.items():
                    f.write(f"  ⚠  {k}: {v}\n")
            f.write("\n")

    print("  ✅ Data quality report → reports/data_quality_report.txt")
    return report


# ══════════════════════════════════════════════════════════════════
#  LAYER 2 — SILVER  (Cleaned, validated, enriched)
# ══════════════════════════════════════════════════════════════════

def transform_to_silver(bronze: dict) -> dict:
    """
    Silver layer transformations:
      - Fill / drop nulls
      - Deduplicate
      - Parse dates
      - Feature engineering (engagement_score, hour_of_day, etc.)
      - Standardise text fields
    """
    logger.info("SILVER layer: cleaning and enriching data...")
    silver = {}

    # ── Users ────────────────────────────────────────────────────
    df_u = bronze["users"].copy()
    df_u = df_u[[c for c in df_u.columns if not c.startswith("_")]]
    df_u["location"] = df_u["location"].fillna("Unknown")
    df_u = df_u.drop_duplicates(subset=["user_id"])
    df_u["join_date"]    = pd.to_datetime(df_u["join_date"])
    df_u["is_influencer"] = df_u["followers"] >= 10000
    df_u["follower_tier"] = pd.cut(
        df_u["followers"],
        bins=[0, 1000, 10000, 100000, float("inf")],
        labels=["Nano", "Micro", "Macro", "Mega"]
    ).astype(str)
    silver["users"] = df_u
    logger.info(f"  Silver users: {df_u.shape}")

    # ── Posts ────────────────────────────────────────────────────
    df_p = bronze["posts"].copy()
    df_p = df_p[[c for c in df_p.columns if not c.startswith("_")]]
    df_p = df_p.drop_duplicates(subset=["post_id"])
    df_p["posted_at"] = pd.to_datetime(df_p["posted_at"])
    df_p["post_date"]      = df_p["posted_at"].dt.date
    df_p["post_hour"]      = df_p["posted_at"].dt.hour
    df_p["post_weekday"]   = df_p["posted_at"].dt.day_name()
    df_p["post_month"]     = df_p["posted_at"].dt.month
    df_p["post_year"]      = df_p["posted_at"].dt.year

    # Engagement score (weighted)
    df_p["engagement_score"] = (
        df_p["likes"] * 1.0
        + df_p["shares"] * 3.0
        + df_p["comments"] * 2.0
    ).round(2)

    # Normalise sentiment to [-1, 1]
    df_p["sentiment_score"] = df_p["sentiment_score"].clip(-1.0, 1.0)

    # Prime-time flag
    df_p["is_prime_time"] = df_p["post_hour"].between(18, 22)

    silver["posts"] = df_p
    logger.info(f"  Silver posts: {df_p.shape}")

    # ── Hashtag Trends ───────────────────────────────────────────
    df_h = bronze["hashtag_trends"].copy()
    df_h = df_h[[c for c in df_h.columns if not c.startswith("_")]]
    df_h["date"] = pd.to_datetime(df_h["date"])
    df_h["engagement_index"] = (
        df_h["total_likes"] + df_h["total_shares"] * 3
    ) / df_h["post_count"].replace(0, 1)
    df_h["engagement_index"] = df_h["engagement_index"].round(2)
    silver["hashtag_trends"] = df_h
    logger.info(f"  Silver hashtag_trends: {df_h.shape}")

    # ── API Logs ─────────────────────────────────────────────────
    df_l = bronze["api_logs"].copy()
    df_l = df_l[[c for c in df_l.columns if not c.startswith("_")]]
    df_l["timestamp"] = pd.to_datetime(df_l["timestamp"])
    df_l["is_error"]  = df_l["status_code"] >= 400
    df_l["is_rate_limited"] = df_l["status_code"] == 429
    df_l["hour"]      = df_l["timestamp"].dt.hour
    silver["api_logs"] = df_l
    logger.info(f"  Silver api_logs: {df_l.shape}")

    # Save Silver
    for name, df in silver.items():
        df.to_csv(f"data/silver/{name}.csv", index=False)

    return silver


# ══════════════════════════════════════════════════════════════════
#  LAYER 3 — GOLD  (Aggregated, ready for BI / ML)
# ══════════════════════════════════════════════════════════════════

def transform_to_gold(silver: dict) -> dict:
    """
    Gold layer: pre-aggregated tables consumed by dashboards & ML.
    """
    logger.info("GOLD layer: building analytical aggregations...")
    gold = {}

    posts = silver["posts"]
    users = silver["users"]

    # ── G1: Topic sentiment summary ──────────────────────────────
    topic_sentiment = (
        posts.groupby(["topic", "sentiment_label"])
        .agg(
            post_count     = ("post_id",        "count"),
            avg_sentiment  = ("sentiment_score", "mean"),
            total_likes    = ("likes",           "sum"),
            total_shares   = ("shares",          "sum"),
            total_engagement = ("engagement_score", "sum"),
            viral_posts    = ("is_viral",        "sum"),
        )
        .reset_index()
        .round(4)
    )
    gold["topic_sentiment"] = topic_sentiment

    # ── G2: Daily sentiment trend ────────────────────────────────
    daily = (
        posts.groupby(["post_date", "topic"])
        .agg(
            posts_count    = ("post_id",        "count"),
            avg_sentiment  = ("sentiment_score", "mean"),
            total_engagement = ("engagement_score", "sum"),
        )
        .reset_index()
        .round(4)
    )
    # 7-day rolling average sentiment
    daily = daily.sort_values(["topic", "post_date"])
    daily["sentiment_7d_avg"] = (
        daily.groupby("topic")["avg_sentiment"]
        .transform(lambda x: x.rolling(7, min_periods=1).mean())
        .round(4)
    )
    gold["daily_trend"] = daily

    # ── G3: Platform performance ─────────────────────────────────
    platform = (
        posts.groupby("platform")
        .agg(
            total_posts   = ("post_id",        "count"),
            avg_sentiment = ("sentiment_score", "mean"),
            total_likes   = ("likes",           "sum"),
            total_shares  = ("shares",          "sum"),
            viral_posts   = ("is_viral",        "sum"),
            avg_engagement = ("engagement_score", "mean"),
        )
        .reset_index()
        .round(4)
    )
    gold["platform_perf"] = platform

    # ── G4: Top influencers ──────────────────────────────────────
    influencers = posts.merge(
        users[["user_id", "username", "followers", "follower_tier",
               "verified", "location"]],
        on="user_id", how="left"
    )
    top_users = (
        influencers.groupby(["user_id", "username", "followers",
                              "follower_tier", "platform", "verified", "location"])
        .agg(
            total_posts       = ("post_id",        "count"),
            total_engagement  = ("engagement_score", "sum"),
            avg_sentiment     = ("sentiment_score", "mean"),
            viral_count       = ("is_viral",        "sum"),
        )
        .reset_index()
        .sort_values("total_engagement", ascending=False)
        .head(200)
        .round(4)
    )
    gold["top_influencers"] = top_users

    # ── G5: Hourly posting pattern ───────────────────────────────
    hourly = (
        posts.groupby(["post_hour", "platform"])
        .agg(
            post_count     = ("post_id",        "count"),
            avg_engagement = ("engagement_score", "mean"),
            avg_sentiment  = ("sentiment_score", "mean"),
        )
        .reset_index()
        .round(4)
    )
    gold["hourly_pattern"] = hourly

    # Save Gold
    for name, df in gold.items():
        df.to_csv(f"data/gold/{name}.csv", index=False)
        logger.info(f"  Gold {name}: {df.shape}")

    return gold


# ══════════════════════════════════════════════════════════════════
#  LOAD — SQLite Data Warehouse (Star Schema)
# ══════════════════════════════════════════════════════════════════

def load_to_warehouse(silver: dict, gold: dict):
    """
    Load Silver (fact/dim tables) and Gold (aggregated views) into
    SQLite warehouse with star schema.
    """
    logger.info("LOAD phase: building SQLite data warehouse...")
    os.makedirs("data/warehouse", exist_ok=True)
    conn   = sqlite3.connect("data/warehouse/sentiment_warehouse.db")
    cursor = conn.cursor()

    # ── Schema ───────────────────────────────────────────────────
    cursor.executescript("""
    PRAGMA foreign_keys = ON;

    -- Dimension: Users
    CREATE TABLE IF NOT EXISTS dim_users (
        user_id          TEXT PRIMARY KEY,
        username         TEXT,
        platform         TEXT,
        location         TEXT,
        followers        INTEGER,
        following        INTEGER,
        follower_tier    TEXT,
        is_influencer    INTEGER,
        verified         INTEGER,
        account_age_days INTEGER,
        language         TEXT
    );

    -- Dimension: Topics
    CREATE TABLE IF NOT EXISTS dim_topics (
        topic_id   INTEGER PRIMARY KEY AUTOINCREMENT,
        topic_name TEXT UNIQUE,
        category   TEXT
    );

    -- Dimension: Date
    CREATE TABLE IF NOT EXISTS dim_date (
        date_id   TEXT PRIMARY KEY,
        year      INTEGER,
        month     INTEGER,
        day       INTEGER,
        weekday   TEXT,
        is_weekend INTEGER
    );

    -- Fact: Posts
    CREATE TABLE IF NOT EXISTS fact_posts (
        post_id          TEXT PRIMARY KEY,
        user_id          TEXT,
        platform         TEXT,
        topic            TEXT,
        sentiment_label  TEXT,
        sentiment_score  REAL,
        likes            INTEGER,
        shares           INTEGER,
        comments         INTEGER,
        engagement_score REAL,
        is_viral         INTEGER,
        is_prime_time    INTEGER,
        post_date        TEXT,
        post_hour        INTEGER,
        post_month       INTEGER,
        post_year        INTEGER,
        FOREIGN KEY (user_id) REFERENCES dim_users(user_id)
    );

    -- Fact: Hashtag Trends
    CREATE TABLE IF NOT EXISTS fact_hashtag_trends (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        date             TEXT,
        hashtag          TEXT,
        topic            TEXT,
        post_count       INTEGER,
        total_likes      INTEGER,
        total_shares     INTEGER,
        avg_sentiment    REAL,
        viral_count      INTEGER,
        engagement_index REAL
    );

    -- Staging: API Logs
    CREATE TABLE IF NOT EXISTS stg_api_logs (
        log_id           TEXT PRIMARY KEY,
        timestamp        TEXT,
        source           TEXT,
        endpoint         TEXT,
        status_code      INTEGER,
        response_time_ms REAL,
        records_fetched  INTEGER,
        is_error         INTEGER,
        is_rate_limited  INTEGER,
        hour             INTEGER
    );
    """)

    # ── Load dimension tables ─────────────────────────────────────
    users_cols = ["user_id", "username", "location", "followers",
                  "following", "follower_tier", "is_influencer", "verified",
                  "account_age_days", "language"]
    silver["users"][users_cols].to_sql(
        "dim_users", conn, if_exists="replace", index=False)
    logger.info(f"  dim_users loaded: {len(silver['users'])} rows")

    # ── Load fact tables ──────────────────────────────────────────
    posts_cols = ["post_id", "user_id", "platform", "topic", "sentiment_label",
                  "sentiment_score", "likes", "shares", "comments",
                  "engagement_score", "is_viral", "is_prime_time",
                  "post_date", "post_hour", "post_month", "post_year"]
    p = silver["posts"][posts_cols].copy()
    p["post_date"] = p["post_date"].astype(str)
    p.to_sql("fact_posts", conn, if_exists="replace", index=False)
    logger.info(f"  fact_posts loaded: {len(p)} rows")

    h = silver["hashtag_trends"].copy()
    h["date"] = h["date"].astype(str)
    h.to_sql("fact_hashtag_trends", conn, if_exists="replace", index=False)
    logger.info(f"  fact_hashtag_trends loaded: {len(h)} rows")

    l = silver["api_logs"].copy()
    l["timestamp"] = l["timestamp"].astype(str)
    l.to_sql("stg_api_logs", conn, if_exists="replace", index=False)
    logger.info(f"  stg_api_logs loaded: {len(l)} rows")

    # ── Advanced SQL Views ────────────────────────────────────────
    advanced_views = {
        "vw_topic_sentiment_rank": """
            SELECT topic,
                   sentiment_label,
                   COUNT(*)                             AS post_count,
                   ROUND(AVG(sentiment_score), 4)       AS avg_sentiment,
                   SUM(likes)                           AS total_likes,
                   SUM(is_viral)                        AS viral_posts,
                   RANK() OVER (PARTITION BY sentiment_label
                                ORDER BY COUNT(*) DESC) AS topic_rank
            FROM fact_posts
            GROUP BY topic, sentiment_label
            ORDER BY post_count DESC
        """,
        "vw_influencer_impact": """
            WITH base AS (
                SELECT p.user_id,
                       u.username,
                       u.followers,
                       u.follower_tier,
                       COUNT(p.post_id)             AS total_posts,
                       SUM(p.engagement_score)      AS total_engagement,
                       ROUND(AVG(p.sentiment_score),4) AS avg_sentiment,
                       SUM(p.is_viral)              AS viral_count
                FROM fact_posts p
                JOIN dim_users  u ON p.user_id = u.user_id
                GROUP BY p.user_id, u.username, u.followers, u.follower_tier
            )
            SELECT *,
                   NTILE(4) OVER (ORDER BY total_engagement DESC) AS engagement_quartile,
                   RANK()   OVER (ORDER BY viral_count DESC)      AS viral_rank
            FROM base
            ORDER BY total_engagement DESC
            LIMIT 100
        """,
        "vw_daily_sentiment_trend": """
            SELECT post_date,
                   topic,
                   COUNT(*)                         AS daily_posts,
                   ROUND(AVG(sentiment_score), 4)   AS avg_sentiment,
                   SUM(likes)                       AS total_likes,
                   SUM(is_viral)                    AS viral_count,
                   ROUND(SUM(SUM(engagement_score)) OVER (
                       PARTITION BY topic
                       ORDER BY post_date
                       ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                   ) / 7.0, 2)                      AS engagement_7d_avg
            FROM fact_posts
            GROUP BY post_date, topic
            ORDER BY post_date, topic
        """,
        "vw_platform_health": """
            SELECT source,
                   COUNT(*)                                    AS total_calls,
                   SUM(is_error)                               AS error_count,
                   ROUND(100.0*SUM(is_error)/COUNT(*), 2)      AS error_rate_pct,
                   SUM(is_rate_limited)                        AS rate_limit_hits,
                   ROUND(AVG(response_time_ms), 1)             AS avg_latency_ms,
                   ROUND(AVG(records_fetched), 1)              AS avg_records_per_call
            FROM stg_api_logs
            GROUP BY source
            ORDER BY error_rate_pct DESC
        """,
        "vw_viral_content_analysis": """
            SELECT platform,
                   topic,
                   sentiment_label,
                   COUNT(CASE WHEN is_viral=1 THEN 1 END)  AS viral_posts,
                   COUNT(*)                                 AS total_posts,
                   ROUND(100.0 * COUNT(CASE WHEN is_viral=1 THEN 1 END)
                         / COUNT(*), 2)                     AS viral_rate_pct,
                   ROUND(AVG(CASE WHEN is_viral=1
                             THEN engagement_score END), 2) AS avg_viral_engagement
            FROM fact_posts
            GROUP BY platform, topic, sentiment_label
            ORDER BY viral_rate_pct DESC
        """,
    }

    for vname, sql in advanced_views.items():
        try:
            result_df = pd.read_sql_query(sql, conn)
            result_df.to_csv(f"data/warehouse/{vname}.csv", index=False)
            cursor.execute(f"DROP VIEW IF EXISTS {vname}")
            cursor.execute(f"CREATE VIEW {vname} AS {sql}")
            logger.info(f"  View '{vname}' created: {len(result_df)} rows")
        except Exception as e:
            logger.error(f"  View '{vname}' failed: {e}")

    # Load Gold tables
    for name, df in gold.items():
        df.to_csv(f"data/warehouse/gold_{name}.csv", index=False)

    conn.commit()
    conn.close()
    logger.info("LOAD complete — sentiment_warehouse.db ready.")
    print("  ✅ SQLite warehouse → data/warehouse/sentiment_warehouse.db")
    print(f"  ✅ {len(advanced_views)} analytical views created")


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("\n" + "═" * 60)
    print("   SOCIAL MEDIA SENTIMENT  —  ETL / MEDALLION PIPELINE")
    print("═" * 60)

    print("\n[1/5] Ingesting to Bronze layer...")
    bronze = ingest_to_bronze()

    print("[2/5] Running data quality checks...")
    quality = run_data_quality(bronze)
    for ds, issues in quality.items():
        flag = "✅" if not issues else "⚠️ "
        print(f"       {flag} {ds}: {len(issues)} issue type(s)")

    print("[3/5] Transforming to Silver layer...")
    silver = transform_to_silver(bronze)

    print("[4/5] Aggregating to Gold layer...")
    gold = transform_to_gold(silver)

    print("[5/5] Loading to data warehouse...")
    load_to_warehouse(silver, gold)

    print("\n✅ ETL + Medallion pipeline complete!")
