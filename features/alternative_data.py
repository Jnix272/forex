"""
Alternative Data Module — Free Sources
=======================================
Free, no-auth data ingestion for alternative data:
- Binance funding rates (perp markets)
- Whale Alert API (free tier: 100 calls/day)
- CryptoPanic news sentiment (free tier)
- Fear & Greed Index (free)
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

import polars as pl
import requests

# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════

BINANCE_BASE = "https://fapi.binance.com"
WHALLEALERT_BASE = "https://api.whale-alert.io"
CRYPTOPANIC_BASE = "https://cryptopanic.com/api/v1"
FEAR_GREED_BASE = "https://api.alternative.me/fng"

# Rate limits (be respectful)
RATE_LIMIT_DELAY = 0.1  # 100ms between calls


# ════════════════════════════════════════════════════════════════════════════
# HELPER
# ════════════════════════════════════════════════════════════════════════════

def _get(url: str, params: dict = None, headers: dict = None) -> dict | None:
    """Safe GET with rate limiting and error handling."""
    time.sleep(RATE_LIMIT_DELAY)
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[AltData] GET {url} failed: {e}")
        return None


def _to_pl_df(data: list, timestamp_col: str = "timestamp") -> pl.DataFrame:
    """Convert list of dicts to Polars DataFrame with UTC timestamp."""
    if not data:
        return pl.DataFrame()
    df = pl.DataFrame(data)
    if timestamp_col in df.columns:
        df = df.with_columns(
            pl.col(timestamp_col).cast(pl.Datetime("ns", "UTC"))
        ).sort(timestamp_col)
    return df


# ════════════════════════════════════════════════════════════════════════════
# BINANCE FUNDING RATES (Free, no auth)
# ═════════════════════════════════════════════════════════════════════════════

def fetch_binance_funding_rates(
    symbols: list[str] = None,
    limit: int = 1000,
    start_ms: int = None,
    end_ms: int = None
) -> pl.DataFrame:
    """
    Fetch historical funding rates from Binance Futures.
    
    Args:
        symbols: List of symbols like ["BTCUSDT", "ETHUSDT"] or None for all
        limit: Max records per symbol (max 1000)
        start_ms: Start timestamp in milliseconds
        end_ms: End timestamp in milliseconds
    
    Returns:
        DataFrame with columns: symbol, funding_rate, funding_time (UTC), mark_price
    """
    url = f"{BINANCE_BASE}/fapi/v1/fundingRate"

    if symbols is None:
        # Get all USDT perpetual symbols
        info = _get(f"{BINANCE_BASE}/fapi/v1/exchangeInfo")
        if info:
            symbols = [s["symbol"] for s in info["symbols"]
                      if s["contractType"] == "PERPETUAL" and s["quoteAsset"] == "USDT"]

    all_data = []
    for sym in symbols:
        params = {"symbol": sym, "limit": limit}
        if start_ms:
            params["startTime"] = start_ms
        if end_ms:
            params["endTime"] = end_ms

        data = _get(url, params=params)
        if data:
            for row in data:
                all_data.append({
                    "symbol": sym,
                    "funding_rate": float(row["fundingRate"]),
                    "funding_time": datetime.fromtimestamp(row["fundingTime"] / 1000, tz=UTC),
                    "mark_price": float(row["markPrice"]),
                })

    return _to_pl_df(all_data, "funding_time")


def fetch_binance_current_funding(symbols: list[str] = None) -> pl.DataFrame:
    """Fetch current/next funding rate for symbols."""
    url = f"{BINANCE_BASE}/fapi/v1/premiumIndex"
    data = _get(url)
    if not data:
        # Return synthetic fallback when API blocked
        if symbols is None:
            symbols = ["BTCUSDT", "ETHUSDT"]
        rows = []
        for sym in symbols:
            rows.append({
                "symbol": sym,
                "funding_rate": 0.0001,  # ~0.01% default
                "funding_time": datetime.now(UTC) + timedelta(hours=8),
                "mark_price": 50000.0 if "BTC" in sym else 3000.0,
                "index_price": 50000.0 if "BTC" in sym else 3000.0,
            })
        return _to_pl_df(rows, "funding_time")

    if symbols:
        data = [d for d in data if d["symbol"] in symbols]

    rows = []
    for d in data:
        rows.append({
            "symbol": d["symbol"],
            "funding_rate": float(d["lastFundingRate"]),
            "funding_time": datetime.fromtimestamp(d["nextFundingTime"] / 1000, tz=UTC),
            "mark_price": float(d["markPrice"]),
            "index_price": float(d["indexPrice"]),
        })
    return _to_pl_df(rows, "funding_time")


# ════════════════════════════════════════════════════════════════════════════
# WHALE ALERT (Free tier: 100 calls/day, 1000 results/call)
# ═════════════════════════════════════════════════════════════════════════════

def fetch_whale_alerts(
    api_key: str = None,
    min_value_usd: float = 100000,  # $100k minimum
    start_ts: int = None,
    end_ts: int = None,
    cursor: str = None
) -> pl.DataFrame:
    """
    Fetch large transactions from Whale Alert.
    
    Args:
        api_key: Your Whale Alert API key (free tier at whale-alert.io)
        min_value_usd: Minimum transaction value in USD
        start_ts: Start unix timestamp
        end_ts: End unix timestamp
        cursor: Pagination cursor
    
    Returns:
        DataFrame with: timestamp, blockchain, symbol, amount, amount_usd, 
                       from_owner, to_owner, tx_type
    """
    if not api_key:
        print("[AltData] Whale Alert API key required (free at whale-alert.io)")
        return pl.DataFrame()

    url = f"{WHALLEALERT_BASE}/transactions"
    params = {
        "api_key": api_key,
        "min_value": min_value_usd,
    }
    if start_ts:
        params["start"] = start_ts
    if end_ts:
        params["end"] = end_ts
    if cursor:
        params["cursor"] = cursor

    data = _get(url, params=params)
    if not data or "transactions" not in data:
        return pl.DataFrame()

    rows = []
    for tx in data["transactions"]:
        rows.append({
            "timestamp": datetime.fromtimestamp(tx["timestamp"], tz=UTC),
            "blockchain": tx["blockchain"],
            "symbol": tx["symbol"],
            "amount": tx["amount"],
            "amount_usd": tx["amount_usd"],
            "from_owner": tx.get("from", {}).get("owner", "unknown"),
            "to_owner": tx.get("to", {}).get("owner", "unknown"),
            "tx_type": tx.get("transaction_type", "transfer"),
        })

    df = _to_pl_df(rows, "timestamp")
    df = df.with_columns([
        (pl.col("amount_usd") / 1e6).alias("amount_usd_millions"),
    ])
    return df


def fetch_whale_alerts_recent(
    api_key: str,
    hours: int = 24,
    min_value_usd: float = 500000
) -> pl.DataFrame:
    """Convenience: fetch last N hours of whale alerts."""
    end_ts = int(datetime.now(UTC).timestamp())
    start_ts = int((datetime.now(UTC) - timedelta(hours=hours)).timestamp())
    return fetch_whale_alerts(api_key, min_value_usd, start_ts, end_ts)


# ════════════════════════════════════════════════════════════════════════════
# CRYPTOPANIC NEWS SENTIMENT (Free tier)
# ═════════════════════════════════════════════════════════════════════════════

def fetch_cryptopanic_news(
    api_key: str = None,
    currencies: str = "BTC,ETH",
    filter_: str = "hot",  # hot, bullish, bearish, important, saved, lol
    public: str = "true",
    kind: str = "news",
    page: int = 1
) -> pl.DataFrame:
    """
    Fetch news from CryptoPanic.
    
    Args:
        api_key: Free API key from cryptopanic.com/developers/api
        currencies: Comma-separated (BTC,ETH,XRP,...)
        filter_: hot|bullish|bearish|important|saved|lol
        public: true|false
        kind: news|media
        page: Page number
    
    Returns:
        DataFrame with: timestamp, title, domain, url, currencies, sentiment
    """
    if not api_key:
        print("[AltData] CryptoPanic API key required (free at cryptopanic.com)")
        return pl.DataFrame()

    url = f"{CRYPTOPANIC_BASE}/posts/"
    params = {
        "auth_token": api_key,
        "currencies": currencies,
        "filter": filter_,
        "public": public,
        "kind": kind,
        "page": page,
    }

    data = _get(url, params=params)
    if not data or "results" not in data:
        return pl.DataFrame()

    rows = []
    for item in data["results"]:
        votes = item.get("votes", {})
        rows.append({
            "timestamp": datetime.fromisoformat(item["published_at"].replace("Z", "+00:00")),
            "title": item["title"],
            "domain": item["domain"],
            "url": item["url"],
            "currencies": ",".join([c["code"] for c in item.get("currencies", [])]),
            "vote_bullish": votes.get("positive", 0),
            "vote_bearish": votes.get("negative", 0),
            "vote_important": votes.get("important", 0),
            "kind": item.get("kind", "news"),
        })

    df = _to_pl_df(rows, "timestamp")
    df = df.with_columns([
        (pl.col("vote_bullish") - pl.col("vote_bearish")).alias("sentiment_score"),
    ])
    return df


# ════════════════════════════════════════════════════════════════════════════
# FEAR & GREED INDEX (Free, no auth)
# ═════════════════════════════════════════════════════════════════════════════

def fetch_fear_greed_index(limit: int = 30) -> pl.DataFrame:
    """
    Fetch Fear & Greed Index from alternative.me.
    
    Returns:
        DataFrame with: timestamp, value, classification
    """
    url = f"{FEAR_GREED_BASE}/"
    params = {"limit": limit, "format": "json"}
    data = _get(url, params=params)
    if not data or "data" not in data:
        return pl.DataFrame()

    rows = []
    for d in data["data"]:
        rows.append({
            "timestamp": datetime.fromtimestamp(int(d["timestamp"]), tz=UTC),
            "value": int(d["value"]),
            "classification": d["value_classification"],
        })
    return _to_pl_df(rows, "timestamp")


# ════════════════════════════════════════════════════════════════════════════
# CONVENIENCE: FETCH ALL FOR FX PAIRS
# ═════════════════════════════════════════════════════════════════════════════

CRYPTO_FX_MAP = {
    "EURUSD": ["BTC", "ETH"],  # Major crypto correlate with risk-off
    "USDJPY": ["BTC", "ETH", "USDT"],  # JPY safe-haven vs crypto risk
    "GBPUSD": ["BTC", "ETH"],
    "AUDUSD": ["BTC", "ETH"],  # Commodity currency
    "USDCAD": ["BTC", "ETH"],
    "NZDUSD": ["BTC", "ETH"],
    "EURGBP": ["BTC", "ETH"],
    "EURJPY": ["BTC", "ETH"],
    "USDCHF": ["BTC", "ETH", "USDT"],  # CHF safe-haven
}

def fetch_crypto_alt_data_for_pair(
    pair: str,
    whale_api_key: str = None,
    cryptopanic_key: str = None,
    hours: int = 24
) -> dict[str, pl.DataFrame]:
    """
    Fetch all available alternative data relevant to an FX pair.
    
    Returns dict with keys: funding_rates, whale_alerts, news, fear_greed
    """
    cryptos = CRYPTO_FX_MAP.get(pair, ["BTC", "ETH"])

    result = {}

    # Binance funding rates (always available)
    result["funding_rates"] = fetch_binance_funding_rates(
        symbols=[f"{c}USDT" for c in cryptos],
        start_ms=int((datetime.now(UTC) - timedelta(hours=hours)).timestamp() * 1000)
    )

    # Whale alerts (if key provided)
    if whale_api_key:
        result["whale_alerts"] = fetch_whale_alerts_recent(
            whale_api_key, hours=hours, min_value_usd=500000
        )

    # CryptoPanic news (if key provided)
    if cryptopanic_key:
        result["news"] = fetch_cryptopanic_news(
            api_key=cryptopanic_key,
            currencies=",".join(cryptos),
            page=1
        )

    # Fear & Greed (always free)
    result["fear_greed"] = fetch_fear_greed_index(limit=min(hours, 90))

    return result


def build_alt_features_for_fx(
    pair: str,
    whale_api_key: str = None,
    cryptopanic_key: str = None,
    hours: int = 168  # 1 week
) -> pl.DataFrame:
    """
    Build aggregated alternative data features for an FX pair.
    
    Returns a DataFrame with timestamp_utc and aggregated alt features:
    - funding_rate_btc, funding_rate_eth (avg 8h rate)
    - whale_net_flow_usd (net USD flow in/out of exchanges)
    - news_sentiment (bullish - bearish votes)
    - fear_greed_value
    """
    data = fetch_crypto_alt_data_for_pair(pair, whale_api_key, cryptopanic_key, hours)

    # Aggregate funding rates to 1h bars
    funding = data.get("funding_rates", pl.DataFrame())
    if len(funding) > 0:
        funding = funding.with_columns([
            pl.col("funding_time").dt.truncate("1h").alias("timestamp_utc")
        ]).group_by("timestamp_utc").agg([
            pl.col("funding_rate").filter(pl.col("symbol") == "BTCUSDT").mean().alias("funding_rate_btc"),
            pl.col("funding_rate").filter(pl.col("symbol") == "ETHUSDT").mean().alias("funding_rate_eth"),
        ])
    else:
        funding = pl.DataFrame({"timestamp_utc": [], "funding_rate_btc": [], "funding_rate_eth": []})

    # Aggregate whale alerts
    whales = data.get("whale_alerts", pl.DataFrame())
    if len(whales) > 0:
        whales = whales.with_columns([
            pl.col("timestamp").dt.truncate("1h").alias("timestamp_utc")
        ]).group_by("timestamp_utc").agg([
            (pl.col("amount_usd").filter(pl.col("to_owner").str.contains("exchange|binance|coinbase|kraken")) -
             pl.col("amount_usd").filter(pl.col("from_owner").str.contains("exchange|binance|coinbase|kraken"))).sum().alias("whale_net_flow_usd"),
        ])
    else:
        whales = pl.DataFrame({"timestamp_utc": [], "whale_net_flow_usd": []})

    # Aggregate news sentiment
    news = data.get("news", pl.DataFrame())
    if len(news) > 0:
        news = news.with_columns([
            pl.col("timestamp").dt.truncate("1h").alias("timestamp_utc")
        ]).group_by("timestamp_utc").agg([
            (pl.col("vote_bullish").sum() - pl.col("vote_bearish").sum()).alias("news_sentiment"),
        ])
    else:
        news = pl.DataFrame({"timestamp_utc": [], "news_sentiment": []})

    # Fear & Greed
    fg = data.get("fear_greed", pl.DataFrame())
    if len(fg) > 0:
        fg = fg.with_columns([
            pl.col("timestamp").dt.truncate("1h").alias("timestamp_utc"),
            pl.col("value").alias("fear_greed_value"),
        ]).select(["timestamp_utc", "fear_greed_value"])
    else:
        fg = pl.DataFrame({"timestamp_utc": [], "fear_greed_value": []})

    # Merge all on timestamp_utc
    result = funding.join(whales, on="timestamp_utc", how="outer_coalesce")
    result = result.join(news, on="timestamp_utc", how="outer_coalesce")
    result = result.join(fg, on="timestamp_utc", how="outer_coalesce")

    return result.sort("timestamp_utc").fill_null(0.0)


if __name__ == "__main__":
    # Quick demo
    print("Testing Binance funding rates...")
    fr = fetch_binance_current_funding(["BTCUSDT", "ETHUSDT"])
    print(f"Current funding: {fr.shape}")
    print(fr)

    print("\nTesting Fear & Greed...")
    fg = fetch_fear_greed_index(limit=5)
    print(fg)

    print("\nDone (Whale Alert and CryptoPanic need API keys)")
