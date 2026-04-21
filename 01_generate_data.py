"""
01_generate_data.py
────────────────────────────────────────────────────────────────────
Simulates live ingestion from Twitter/Reddit APIs.
Generates synthetic social media posts, user profiles, hashtag trends,
and API server logs — all India-flavoured via Faker(en_IN).

Concepts covered:
  • Advanced Python  : dataclasses, generators, type hints
  • Pandas / NumPy   : large-scale DataFrame creation, memory optimisation
  • Data Ingestion   : multi-source batch ingest (simulated REST API)
  • PII Handling     : email / phone masking before storage
"""

import os
import json
import random
import logging
import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Generator, List

# ── Directories & Logging ────────────────────────────────────────
for d in ["data/raw", "data/processed", "data/bronze",
          "data/silver", "data/gold", "logs", "reports"]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    filename="logs/pipeline.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

fake = Faker("en_IN")
random.seed(42)
np.random.seed(42)

# ── Constants ────────────────────────────────────────────────────
PLATFORMS   = ["Twitter", "Reddit", "Instagram", "LinkedIn"]
LANGUAGES   = ["en", "hi", "ta", "te", "bn"]
TOPICS      = ["Technology", "Politics", "Cricket", "Bollywood",
               "Finance", "Health", "Education", "Environment",
               "Startup", "Gaming"]

HASHTAGS = {
    "Technology":   ["#AI", "#MachineLearning", "#Python", "#DataScience", "#CloudComputing"],
    "Politics":     ["#Elections2024", "#Democracy", "#Parliament", "#BJP", "#Congress"],
    "Cricket":      ["#IPL2024", "#TeamIndia", "#ViratKohli", "#RohitSharma", "#CricketFever"],
    "Bollywood":    ["#Bollywood", "#BoxOffice", "#NewMovie", "#OTT", "#BollywoodNews"],
    "Finance":      ["#Sensex", "#Nifty50", "#StockMarket", "#MutualFunds", "#Crypto"],
    "Health":       ["#MentalHealth", "#Yoga", "#FitIndia", "#Healthcare", "#Wellness"],
    "Education":    ["#NEP2020", "#OnlineLearning", "#IIT", "#EdTech", "#Scholarship"],
    "Environment":  ["#ClimateChange", "#GoGreen", "#Pollution", "#SaveEarth", "#EV"],
    "Startup":      ["#Startup", "#Unicorn", "#VC", "#Founders", "#Entrepreneurship"],
    "Gaming":       ["#BGMIIndia", "#Esports", "#StreamerLife", "#GamingCommunity", "#FreeFireIndia"],
}

POSITIVE_WORDS = ["amazing", "brilliant", "excellent", "love", "fantastic",
                  "great", "awesome", "superb", "wonderful", "outstanding",
                  "proud", "inspiring", "delightful", "impressive", "best"]
NEGATIVE_WORDS = ["terrible", "awful", "horrible", "hate", "disgusting",
                  "worst", "pathetic", "shameful", "disappointing", "disaster",
                  "corrupt", "failure", "useless", "toxic", "outrageous"]
NEUTRAL_WORDS  = ["update", "news", "report", "today", "latest",
                  "happening", "according", "statement", "announced", "released"]

SENTIMENT_TEMPLATES = {
    "positive": [
        "This is {adj}! #{tag1} #{tag2}",
        "Really {adj} to see this development in {topic}. Great work! #{tag1}",
        "Absolutely {adj}! {topic} is heading in the right direction. #{tag1} #{tag2}",
        "Loving the {adj} vibes from {topic} lately! #{tag1}",
    ],
    "negative": [
        "This is absolutely {adj}! #{tag1} #{tag2}",
        "Can't believe how {adj} this situation with {topic} has become. #{tag1}",
        "Feeling {adj} about the state of {topic}. Something must change. #{tag1}",
        "The {adj} truth about {topic} nobody is talking about. #{tag1} #{tag2}",
    ],
    "neutral": [
        "Latest {topic} {adj}: New developments announced. #{tag1}",
        "Breaking: {topic} {adj} - here's what you need to know. #{tag1} #{tag2}",
        "{topic} {adj} released today. Full details below. #{tag1}",
        "Official {adj} on {topic} from authorities. #{tag1}",
    ]
}


# ── Dataclasses (Advanced Python) ───────────────────────────────

@dataclass
class UserProfile:
    user_id:        str
    username:       str
    display_name:   str
    email_masked:   str          # PII masked
    platform:       str
    location:       str
    followers:      int
    following:      int
    verified:       bool
    account_age_days: int
    language:       str
    join_date:      str


@dataclass
class SocialPost:
    post_id:        str
    user_id:        str
    platform:       str
    topic:          str
    text:           str
    hashtags:       str          # pipe-separated
    likes:          int
    shares:         int
    comments:       int
    sentiment_label: str         # positive / negative / neutral
    sentiment_score: float       # -1.0 to +1.0
    language:       str
    posted_at:      str
    is_viral:       bool


# ── Utility helpers ──────────────────────────────────────────────

def mask_email(email: str) -> str:
    """PII masking: keep first char + domain."""
    local, domain = email.split("@")
    return local[0] + "***@" + domain


def sentiment_score(label: str) -> float:
    """Convert label to numeric score with some noise."""
    base = {"positive": 0.65, "negative": -0.65, "neutral": 0.05}[label]
    return round(base + np.random.uniform(-0.3, 0.3), 4)


def generate_post_text(topic: str, sentiment: str) -> tuple[str, list]:
    """Generate post text and extract hashtags."""
    tags = HASHTAGS.get(topic, ["#Trending"])
    chosen_tags = random.sample(tags, k=min(2, len(tags)))
    word_pool   = (POSITIVE_WORDS if sentiment == "positive"
                   else NEGATIVE_WORDS if sentiment == "negative"
                   else NEUTRAL_WORDS)
    adj = random.choice(word_pool)
    template = random.choice(SENTIMENT_TEMPLATES[sentiment])
    text = template.format(adj=adj, topic=topic,
                           tag1=chosen_tags[0],
                           tag2=chosen_tags[-1])
    return text, chosen_tags


# ── Generators ──────────────────────────────────────────────────

def user_generator(n: int = 1000) -> Generator[UserProfile, None, None]:
    """Lazy generator for user profiles."""
    cities = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai",
              "Pune", "Kolkata", "Ahmedabad", "Jaipur", "Lucknow",
              "Surat", "Nagpur", "Indore", "Bhopal", "Patna"]
    for i in range(1, n + 1):
        platform = random.choice(PLATFORMS)
        followers = int(np.random.lognormal(mean=7.0, sigma=2.0))
        yield UserProfile(
            user_id        = f"USR{i:06d}",
            username       = fake.user_name(),
            display_name   = fake.name(),
            email_masked   = mask_email(fake.email()),
            platform       = platform,
            location       = random.choice(cities),
            followers      = followers,
            following      = int(followers * random.uniform(0.3, 2.0)),
            verified       = random.choices([True, False], weights=[5, 95])[0],
            account_age_days = random.randint(30, 3650),
            language       = random.choice(LANGUAGES),
            join_date      = fake.date_between(start_date="-10y", end_date="-1m").isoformat(),
        )


def post_generator(users_df: pd.DataFrame,
                   n: int = 50000) -> Generator[SocialPost, None, None]:
    """Lazy generator for social media posts."""
    user_ids  = users_df["user_id"].tolist()
    platforms = users_df.set_index("user_id")["platform"].to_dict()
    start_dt  = datetime(2023, 1, 1)

    sentiments = random.choices(
        ["positive", "negative", "neutral"],
        weights=[45, 25, 30],
        k=n
    )

    for i in range(1, n + 1):
        uid       = random.choice(user_ids)
        topic     = random.choice(TOPICS)
        sentiment = sentiments[i - 1]
        text, tags = generate_post_text(topic, sentiment)
        score      = sentiment_score(sentiment)
        likes      = int(np.random.lognormal(mean=3.5, sigma=2.0))
        shares     = int(likes * random.uniform(0.05, 0.4))
        comments   = int(likes * random.uniform(0.02, 0.3))

        posted_at = (start_dt + timedelta(
            days=random.randint(0, 540),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )).strftime("%Y-%m-%d %H:%M:%S")

        yield SocialPost(
            post_id        = f"POST{i:08d}",
            user_id        = uid,
            platform       = platforms.get(uid, "Twitter"),
            topic          = topic,
            text           = text,
            hashtags       = "|".join(tags),
            likes          = likes,
            shares         = shares,
            comments       = comments,
            sentiment_label = sentiment,
            sentiment_score = score,
            language       = random.choice(LANGUAGES),
            posted_at      = posted_at,
            is_viral       = (likes + shares) > 5000,
        )


# ── Generate Functions ───────────────────────────────────────────

def generate_users(n: int = 1000) -> pd.DataFrame:
    logger.info(f"Generating {n} user profiles...")
    df = pd.DataFrame([asdict(u) for u in user_generator(n)])
    # Introduce nulls for data quality testing
    df.loc[df.sample(frac=0.02).index, "location"] = None
    df.to_csv("data/raw/users.csv", index=False)
    logger.info(f"users.csv saved → {df.shape}")
    return df


def generate_posts(users_df: pd.DataFrame, n: int = 50000) -> pd.DataFrame:
    logger.info(f"Generating {n} social media posts...")
    records = []
    for idx, post in enumerate(post_generator(users_df, n)):
        records.append(asdict(post))
        if (idx + 1) % 10000 == 0:
            logger.info(f"  Generated {idx+1}/{n} posts...")

    df = pd.DataFrame(records)
    # Memory optimise
    for col in df.select_dtypes("int64").columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    df.to_csv("data/raw/posts.csv", index=False)
    logger.info(f"posts.csv saved → {df.shape}")
    return df


def generate_hashtag_trends(posts_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hashtag performance over time (trend table)."""
    logger.info("Generating hashtag trend aggregations...")
    # Explode pipe-separated hashtags
    exploded = posts_df.copy()
    exploded["hashtag"] = exploded["hashtags"].str.split("|")
    exploded = exploded.explode("hashtag")
    exploded["posted_at"] = pd.to_datetime(exploded["posted_at"])
    exploded["date"]      = exploded["posted_at"].dt.date

    trend = (
        exploded.groupby(["date", "hashtag", "topic"])
        .agg(
            post_count      = ("post_id",  "count"),
            total_likes     = ("likes",    "sum"),
            total_shares    = ("shares",   "sum"),
            avg_sentiment   = ("sentiment_score", "mean"),
            viral_count     = ("is_viral", "sum"),
        )
        .reset_index()
    )
    trend["avg_sentiment"] = trend["avg_sentiment"].round(4)
    trend.to_csv("data/raw/hashtag_trends.csv", index=False)
    logger.info(f"hashtag_trends.csv saved → {trend.shape}")
    return trend


def generate_api_logs(n: int = 15000) -> pd.DataFrame:
    """Simulate Twitter/Reddit API ingestion logs."""
    logger.info(f"Generating {n} API log entries...")
    endpoints   = ["/api/v2/tweets/search", "/api/v2/users/lookup",
                   "/r/india/hot.json",     "/r/cricket/new.json",
                   "/api/v2/trending",      "/api/v2/hashtags"]
    status_codes = [200, 200, 200, 200, 429, 404, 500]  # 429 = rate-limit
    sources      = ["Twitter", "Reddit"]
    base_time    = datetime(2024, 1, 1)

    logs = []
    for i in range(n):
        ts = base_time + timedelta(seconds=i * random.uniform(1, 8))
        logs.append({
            "log_id":          f"LOG{i:07d}",
            "timestamp":       ts.strftime("%Y-%m-%d %H:%M:%S"),
            "source":          random.choice(sources),
            "endpoint":        random.choice(endpoints),
            "status_code":     random.choice(status_codes),
            "response_time_ms": round(np.random.exponential(scale=180), 1),
            "records_fetched": random.randint(0, 100),
            "rate_limited":    random.choice(status_codes) == 429,
        })
    df = pd.DataFrame(logs)
    df.to_csv("data/raw/api_logs.csv", index=False)
    logger.info(f"api_logs.csv saved → {df.shape}")
    return df


# ── Main ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("🚀 Generating synthetic social media data...\n")

    users = generate_users(1000)
    posts = generate_posts(users, 50000)
    trends = generate_hashtag_trends(posts)
    logs  = generate_api_logs(15000)

    # Quick NumPy stats
    scores = posts["sentiment_score"].values
    print(f"📊 Post Stats:")
    print(f"   Total posts      : {len(posts):,}")
    print(f"   Viral posts      : {posts['is_viral'].sum():,}")
    print(f"   Sentiment mean   : {np.mean(scores):.4f}")
    print(f"   Sentiment std    : {np.std(scores):.4f}")
    print(f"   Positive posts   : {(posts['sentiment_label']=='positive').sum():,}")
    print(f"   Negative posts   : {(posts['sentiment_label']=='negative').sum():,}")
    print(f"\n✅ Data generation complete!")
    print(f"   Users            : {len(users):,}")
    print(f"   Posts            : {len(posts):,}")
    print(f"   Hashtag trends   : {len(trends):,}")
    print(f"   API logs         : {len(logs):,}")
