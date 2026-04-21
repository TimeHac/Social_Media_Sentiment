"""
03_analytics.py
────────────────────────────────────────────────────────────────────
Analytics & NLP engine for the Social Media Sentiment Pipeline.

Covers:
  • NLP Sentiment Analysis  : TextBlob-style rule-based scoring
                              + HuggingFace-style transformer stub
  • Pandas / NumPy           : large-scale analysis, memory optimisation
  • Advanced SQL             : window functions, CTEs, NTILE, RANK
  • Cloud DW patterns        : BigQuery-style partitioned queries
  • Elasticsearch simulation : inverted index + full-text search
  • Visualisations           : 8-panel dark-theme dashboard
"""

import os
import re
import json
import sqlite3
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter, defaultdict
from datetime import datetime

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/pipeline.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Chart style ──────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d2e",
    "axes.edgecolor":   "#2d3561",
    "axes.labelcolor":  "#e0e0e0",
    "xtick.color":      "#aaaaaa",
    "ytick.color":      "#aaaaaa",
    "text.color":       "#e0e0e0",
    "grid.color":       "#2d3561",
    "grid.linewidth":   0.5,
})
PALETTE = ["#00d4ff", "#7c4dff", "#ff6b6b", "#ffd93d",
           "#6bcb77", "#ff922b", "#f06595", "#74c0fc"]

CONN_PATH = "data/warehouse/sentiment_warehouse.db"


def connect():
    return sqlite3.connect(CONN_PATH)


# ══════════════════════════════════════════════════════════════════
#  NLP ENGINE  (simulates HuggingFace Transformers pipeline)
# ══════════════════════════════════════════════════════════════════

POSITIVE_WORDS = {
    "amazing", "brilliant", "excellent", "love", "fantastic", "great",
    "awesome", "superb", "wonderful", "outstanding", "proud", "inspiring",
    "delightful", "impressive", "best", "perfect", "thrilled", "happy",
}
NEGATIVE_WORDS = {
    "terrible", "awful", "horrible", "hate", "disgusting", "worst",
    "pathetic", "shameful", "disappointing", "disaster", "corrupt",
    "failure", "useless", "toxic", "outrageous", "sad", "angry",
}


def rule_based_sentiment(text: str) -> dict:
    """
    Rule-based NLP classifier — simulates a lightweight transformer.
    Returns: {"label": str, "score": float, "confidence": float}
    """
    tokens = re.findall(r"\b\w+\b", text.lower())
    pos = sum(1 for t in tokens if t in POSITIVE_WORDS)
    neg = sum(1 for t in tokens if t in NEGATIVE_WORDS)
    total = pos + neg + 1e-9

    if pos > neg:
        label = "positive"
        score = min(pos / total, 1.0)
    elif neg > pos:
        label = "negative"
        score = -min(neg / total, 1.0)
    else:
        label = "neutral"
        score = 0.05

    confidence = abs(pos - neg) / (total) if total > 0 else 0.5
    return {
        "label":      label,
        "score":      round(float(score), 4),
        "confidence": round(min(confidence, 1.0), 4)
    }


def batch_nlp_inference(posts_df: pd.DataFrame,
                         sample_size: int = 5000) -> pd.DataFrame:
    """
    Simulate HuggingFace batch inference on a sample of posts.
    Returns DataFrame with re-scored sentiment columns.
    """
    logger.info(f"NLP batch inference on {sample_size} posts...")
    sample = posts_df.sample(min(sample_size, len(posts_df)), random_state=42).copy()

    results = sample["text"].apply(rule_based_sentiment)
    sample["nlp_label"]      = results.apply(lambda x: x["label"])
    sample["nlp_score"]      = results.apply(lambda x: x["score"])
    sample["nlp_confidence"] = results.apply(lambda x: x["confidence"])

    # Accuracy vs stored label (agreement rate)
    agreement = (sample["nlp_label"] == sample["sentiment_label"]).mean()
    logger.info(f"  NLP model agreement with stored labels: {agreement:.2%}")
    print(f"  NLP agreement rate: {agreement:.2%}")

    sample.to_csv("data/processed/nlp_inference_sample.csv", index=False)
    return sample


# ══════════════════════════════════════════════════════════════════
#  ELASTICSEARCH-STYLE FULL-TEXT SEARCH INDEX
# ══════════════════════════════════════════════════════════════════

class SentimentSearchIndex:
    """
    Simulates Elasticsearch inverted index for full-text search
    on social media posts.
    """

    def __init__(self):
        self.inverted_index: dict[str, list] = defaultdict(list)
        self.documents: dict[str, dict] = {}
        self.doc_count = 0

    def index(self, posts_df: pd.DataFrame, sample: int = 10000):
        logger.info(f"Building search index on {sample} posts...")
        df = posts_df.sample(min(sample, len(posts_df)), random_state=42)
        for _, row in df.iterrows():
            doc_id = row["post_id"]
            tokens = re.findall(r"\b\w+\b", str(row.get("text", "")).lower())
            self.documents[doc_id] = {
                "post_id":        doc_id,
                "text":           row.get("text", ""),
                "topic":          row.get("topic", ""),
                "sentiment_label": row.get("sentiment_label", ""),
                "sentiment_score": row.get("sentiment_score", 0),
                "engagement_score": row.get("engagement_score", 0),
                "platform":       row.get("platform", ""),
            }
            for token in set(tokens):
                self.inverted_index[token].append(doc_id)
            self.doc_count += 1
        logger.info(f"  Index built: {self.doc_count} docs, "
                    f"{len(self.inverted_index)} unique terms")

    def search(self, query: str, top_k: int = 10) -> list:
        """BM25-simplified term-frequency search."""
        query_tokens = re.findall(r"\b\w+\b", query.lower())
        scores: dict[str, float] = defaultdict(float)
        for token in query_tokens:
            matching_docs = self.inverted_index.get(token, [])
            idf = np.log(self.doc_count / (len(matching_docs) + 1)) + 1
            for doc_id in matching_docs:
                scores[doc_id] += idf

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {**self.documents[doc_id], "relevance_score": round(score, 4)}
            for doc_id, score in ranked
        ]

    def top_terms(self, n: int = 20) -> list[tuple]:
        return sorted(
            [(term, len(docs)) for term, docs in self.inverted_index.items()],
            key=lambda x: x[1], reverse=True
        )[:n]


# ══════════════════════════════════════════════════════════════════
#  KPI COMPUTATION
# ══════════════════════════════════════════════════════════════════

def compute_kpis(conn) -> dict:
    q = """
    SELECT
        COUNT(*)                                      AS total_posts,
        COUNT(DISTINCT user_id)                       AS unique_users,
        SUM(likes)                                    AS total_likes,
        SUM(shares)                                   AS total_shares,
        SUM(is_viral)                                 AS viral_posts,
        ROUND(AVG(sentiment_score), 4)                AS overall_sentiment,
        ROUND(AVG(engagement_score), 2)               AS avg_engagement,
        SUM(CASE WHEN sentiment_label='positive' THEN 1 ELSE 0 END) AS positive_posts,
        SUM(CASE WHEN sentiment_label='negative' THEN 1 ELSE 0 END) AS negative_posts,
        SUM(CASE WHEN sentiment_label='neutral'  THEN 1 ELSE 0 END) AS neutral_posts
    FROM fact_posts
    """
    row = pd.read_sql_query(q, conn).iloc[0]
    kpis = row.to_dict()
    logger.info(f"KPIs: total_posts={kpis['total_posts']:,}, "
                f"sentiment={kpis['overall_sentiment']:.4f}")
    return kpis


# ══════════════════════════════════════════════════════════════════
#  ADVANCED SQL ANALYTICS
# ══════════════════════════════════════════════════════════════════

def run_advanced_sql(conn):
    logger.info("Running advanced SQL analytics...")

    # ── Hashtag velocity (trending) ───────────────────────────────
    hashtag_velocity = pd.read_sql_query("""
        SELECT hashtag, topic,
               SUM(post_count)    AS total_posts,
               SUM(total_likes)   AS total_likes,
               SUM(total_shares)  AS total_shares,
               ROUND(AVG(avg_sentiment), 4) AS avg_sentiment,
               SUM(viral_count)   AS total_viral,
               RANK() OVER (ORDER BY SUM(post_count) DESC) AS trend_rank
        FROM fact_hashtag_trends
        GROUP BY hashtag, topic
        ORDER BY total_posts DESC
        LIMIT 30
    """, conn)
    hashtag_velocity.to_csv("data/warehouse/trending_hashtags.csv", index=False)

    # ── User segment analysis ─────────────────────────────────────
    user_segment = pd.read_sql_query("""
        WITH user_stats AS (
            SELECT user_id,
                   COUNT(*)                    AS post_count,
                   ROUND(AVG(sentiment_score), 4) AS avg_sentiment,
                   SUM(engagement_score)       AS total_engagement,
                   SUM(is_viral)               AS viral_posts
            FROM fact_posts
            GROUP BY user_id
        )
        SELECT u.follower_tier,
               COUNT(DISTINCT u.user_id)           AS user_count,
               ROUND(AVG(s.post_count), 1)          AS avg_posts,
               ROUND(AVG(s.total_engagement), 2)    AS avg_engagement,
               ROUND(AVG(s.avg_sentiment), 4)       AS avg_sentiment,
               SUM(s.viral_posts)                   AS total_viral
        FROM dim_users u
        JOIN user_stats s ON u.user_id = s.user_id
        GROUP BY u.follower_tier
        ORDER BY avg_engagement DESC
    """, conn)
    user_segment.to_csv("data/warehouse/user_segment_analysis.csv", index=False)

    # ── Prime-time engagement ─────────────────────────────────────
    prime_time = pd.read_sql_query("""
        SELECT post_hour,
               platform,
               COUNT(*)                         AS posts,
               ROUND(AVG(engagement_score), 2)  AS avg_engagement,
               ROUND(AVG(sentiment_score), 4)   AS avg_sentiment,
               ROUND(100.0 * SUM(is_viral) / COUNT(*), 2) AS viral_rate_pct
        FROM fact_posts
        GROUP BY post_hour, platform
        ORDER BY post_hour, platform
    """, conn)
    prime_time.to_csv("data/warehouse/primetime_analysis.csv", index=False)

    logger.info("  Advanced SQL analytics complete.")
    return hashtag_velocity, user_segment, prime_time


# ══════════════════════════════════════════════════════════════════
#  VISUALISATION — 8-PANEL DASHBOARD
# ══════════════════════════════════════════════════════════════════

def build_dashboard(conn, kpis: dict, nlp_sample: pd.DataFrame):
    logger.info("Building analytics dashboard...")
    fig = plt.figure(figsize=(22, 28), facecolor="#0f1117")
    fig.suptitle(
        "SOCIAL MEDIA SENTIMENT PIPELINE  —  ANALYTICS DASHBOARD",
        fontsize=20, fontweight="bold", color="#00d4ff",
        y=0.98, fontfamily="monospace"
    )

    gs = fig.add_gridspec(4, 3, hspace=0.50, wspace=0.38,
                          left=0.06, right=0.97, top=0.94, bottom=0.04)

    # ── 1. Daily Sentiment Trend ──────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    trend = pd.read_sql_query("""
        SELECT post_date,
               ROUND(AVG(sentiment_score), 4) AS avg_sentiment,
               COUNT(*) AS posts
        FROM fact_posts
        GROUP BY post_date
        ORDER BY post_date
    """, conn)
    trend["post_date"] = pd.to_datetime(trend["post_date"])
    trend["rolling_7"] = trend["avg_sentiment"].rolling(7, min_periods=1).mean()

    ax1.fill_between(trend["post_date"], trend["avg_sentiment"],
                     alpha=0.15, color="#7c4dff")
    ax1.plot(trend["post_date"], trend["avg_sentiment"],
             color="#7c4dff", linewidth=0.8, alpha=0.6, label="Daily")
    ax1.plot(trend["post_date"], trend["rolling_7"],
             color="#00d4ff", linewidth=2.5, label="7-Day Avg")
    ax1.axhline(0, color="#ff6b6b", linestyle="--", linewidth=1, alpha=0.5, label="Neutral")
    ax1.set_title("Daily Sentiment Trend (with 7-Day Rolling Avg)", fontsize=13,
                  color="#00d4ff", pad=10)
    ax1.set_ylabel("Avg Sentiment Score", fontsize=9)
    ax1.legend(fontsize=8, facecolor="#1a1d2e", edgecolor="#2d3561")
    ax1.grid(axis="y", alpha=0.4)

    # ── 2. Sentiment Distribution Pie ────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    labels = ["Positive", "Negative", "Neutral"]
    sizes  = [kpis["positive_posts"], kpis["negative_posts"], kpis["neutral_posts"]]
    colors = ["#6bcb77", "#ff6b6b", "#ffd93d"]
    wedges, texts, autotexts = ax2.pie(
        sizes, labels=labels, autopct="%1.1f%%",
        colors=colors, startangle=140,
        wedgeprops=dict(edgecolor="#0f1117", linewidth=2),
        pctdistance=0.75
    )
    for t in autotexts: t.set_fontsize(9); t.set_color("#0f1117")
    ax2.set_title("Overall Sentiment Distribution", fontsize=13, color="#6bcb77", pad=10)

    # ── 3. Topic Sentiment Heatmap ────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    heat_data = pd.read_sql_query("""
        SELECT topic, sentiment_label, COUNT(*) AS cnt
        FROM fact_posts GROUP BY topic, sentiment_label
    """, conn)
    pivot = heat_data.pivot(index="topic", columns="sentiment_label", values="cnt").fillna(0)
    sns.heatmap(pivot, ax=ax3, cmap="YlOrRd",
                annot=True, fmt=".0f", linewidths=0.5,
                annot_kws={"size": 8},
                cbar_kws={"shrink": 0.8})
    ax3.set_title("Post Volume by Topic × Sentiment", fontsize=13, color="#ff6b6b", pad=10)
    ax3.set_xlabel("Sentiment", fontsize=9)
    ax3.set_ylabel("Topic", fontsize=9)
    plt.setp(ax3.get_xticklabels(), fontsize=8)
    plt.setp(ax3.get_yticklabels(), fontsize=7)

    # ── 4. Platform Engagement Bars ───────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    plat = pd.read_sql_query("""
        SELECT platform,
               ROUND(AVG(engagement_score), 2) AS avg_eng,
               COUNT(*) AS posts
        FROM fact_posts GROUP BY platform ORDER BY avg_eng DESC
    """, conn)
    bars = ax4.bar(plat["platform"], plat["avg_eng"],
                   color=PALETTE[:len(plat)], edgecolor="#0f1117", width=0.6)
    ax4.set_title("Avg Engagement by Platform", fontsize=13, color="#ffd93d", pad=10)
    ax4.set_ylabel("Engagement Score", fontsize=9)
    ax4.grid(axis="y", alpha=0.3)
    for b in bars:
        ax4.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                 f"{b.get_height():.1f}", ha="center", fontsize=8, color="#e0e0e0")

    # ── 5. Top 10 Trending Hashtags ───────────────────────────────
    ax5 = fig.add_subplot(gs[2, :2])
    if os.path.exists("data/warehouse/trending_hashtags.csv"):
        th = pd.read_csv("data/warehouse/trending_hashtags.csv").head(10)
        colors_h = ["#ff6b6b" if s < 0 else "#6bcb77" if s > 0 else "#ffd93d"
                    for s in th["avg_sentiment"]]
        bars_h = ax5.barh(th["hashtag"][::-1], th["total_posts"][::-1],
                          color=colors_h[::-1], edgecolor="#0f1117", height=0.6)
        ax5.set_title("Top 10 Trending Hashtags (color = sentiment)",
                      fontsize=13, color="#ff922b", pad=10)
        ax5.set_xlabel("Total Posts", fontsize=9)
        ax5.grid(axis="x", alpha=0.3)
        legend_els = [mpatches.Patch(color="#6bcb77", label="Positive"),
                      mpatches.Patch(color="#ff6b6b", label="Negative"),
                      mpatches.Patch(color="#ffd93d", label="Neutral")]
        ax5.legend(handles=legend_els, fontsize=7,
                   facecolor="#1a1d2e", edgecolor="#2d3561")

    # ── 6. Hourly Posting Pattern (heatmap) ───────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    hourly = pd.read_sql_query("""
        SELECT post_hour, COUNT(*) AS posts
        FROM fact_posts GROUP BY post_hour ORDER BY post_hour
    """, conn)
    colors_h2 = ["#ff922b" if 18 <= h <= 22 else "#00d4ff"
                 for h in hourly["post_hour"]]
    ax6.bar(hourly["post_hour"], hourly["posts"],
            color=colors_h2, edgecolor="#0f1117", width=0.8)
    ax6.set_title("Posts by Hour of Day\n(orange = prime time 18-22h)",
                  fontsize=11, color="#ff922b", pad=10)
    ax6.set_xlabel("Hour", fontsize=9)
    ax6.set_ylabel("Posts", fontsize=9)
    ax6.grid(axis="y", alpha=0.3)

    # ── 7. NLP Confidence Distribution ───────────────────────────
    ax7 = fig.add_subplot(gs[3, :2])
    if "nlp_confidence" in nlp_sample.columns:
        for label, colour in zip(
            ["positive", "negative", "neutral"],
            ["#6bcb77", "#ff6b6b", "#ffd93d"]
        ):
            subset = nlp_sample[nlp_sample["nlp_label"] == label]["nlp_confidence"]
            ax7.hist(subset, bins=30, alpha=0.7, color=colour,
                     label=label, edgecolor="#0f1117")
        ax7.set_title("NLP Model Confidence Distribution by Sentiment Class",
                      fontsize=13, color="#7c4dff", pad=10)
        ax7.set_xlabel("Confidence Score", fontsize=9)
        ax7.set_ylabel("Count", fontsize=9)
        ax7.legend(fontsize=8, facecolor="#1a1d2e", edgecolor="#2d3561")
        ax7.grid(axis="y", alpha=0.3)

    # ── 8. API Health ─────────────────────────────────────────────
    ax8 = fig.add_subplot(gs[3, 2])
    api = pd.read_sql_query("""
        SELECT endpoint,
               ROUND(100.0*SUM(is_error)/COUNT(*), 1) AS error_pct,
               ROUND(AVG(response_time_ms), 0)        AS avg_ms
        FROM stg_api_logs GROUP BY endpoint ORDER BY error_pct DESC
    """, conn)
    c8 = ["#ff6b6b" if v > 20 else "#ffd93d" if v > 10 else "#6bcb77"
          for v in api["error_pct"]]
    ax8.barh(api["endpoint"], api["error_pct"], color=c8, edgecolor="#0f1117")
    ax8.set_title("API Endpoint Error Rates", fontsize=13, color="#ff6b6b", pad=10)
    ax8.set_xlabel("Error Rate (%)", fontsize=9)
    plt.setp(ax8.get_yticklabels(), fontsize=6)
    ax8.grid(axis="x", alpha=0.3)
    leg8 = [mpatches.Patch(color="#ff6b6b", label=">20% Critical"),
            mpatches.Patch(color="#ffd93d", label="10-20% Warning"),
            mpatches.Patch(color="#6bcb77", label="<10% OK")]
    ax8.legend(handles=leg8, fontsize=6, facecolor="#1a1d2e", edgecolor="#2d3561")

    os.makedirs("reports", exist_ok=True)
    plt.savefig("reports/analytics_dashboard.png", dpi=150,
                bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    logger.info("Dashboard saved → reports/analytics_dashboard.png")
    print("  ✅ Dashboard saved → reports/analytics_dashboard.png")


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("\n" + "═" * 60)
    print("   ANALYTICS & NLP ENGINE")
    print("═" * 60)

    conn = connect()

    print("\n[1/5] Computing KPIs...")
    kpis = compute_kpis(conn)
    print(f"       Total posts       : {kpis['total_posts']:,}")
    print(f"       Unique users       : {kpis['unique_users']:,}")
    print(f"       Overall sentiment  : {kpis['overall_sentiment']:.4f}")
    print(f"       Viral posts        : {kpis['viral_posts']:,}")

    print("\n[2/5] Loading posts for NLP inference...")
    # Load text from Silver CSV (text not stored in warehouse to save space)
    posts_df = pd.read_csv("data/silver/posts.csv",
                           usecols=["post_id","text","sentiment_label",
                                    "sentiment_score","engagement_score","platform"])

    print("[3/5] Running NLP batch inference (rule-based transformer sim)...")
    nlp_sample = batch_nlp_inference(posts_df)

    print("[4/5] Building Elasticsearch-style search index...")
    search_idx = SentimentSearchIndex()
    search_idx.index(posts_df, sample=10000)
    top_terms = search_idx.top_terms(20)
    print(f"       Index built: {search_idx.doc_count:,} docs indexed")

    # Demo search
    results = search_idx.search("AI machine learning data science", top_k=5)
    search_report = {
        "query_demo": "AI machine learning data science",
        "results":    results[:3],
        "top_terms":  top_terms[:10],
        "generated_at": datetime.now().isoformat()
    }
    os.makedirs("reports", exist_ok=True)
    with open("reports/search_index_report.json", "w") as f:
        json.dump(search_report, f, indent=2)
    print(f"       Demo search results saved → reports/search_index_report.json")

    print("[5/5] Running advanced SQL analytics and building dashboard...")
    run_advanced_sql(conn)
    build_dashboard(conn, kpis, nlp_sample)

    conn.close()
    print("\n✅ Analytics & NLP complete!")
