"""
Multi-Modal Sentiment (Improvement #3)
======================================
Fuses news + social + COT sentiment modalities, extracts topic structure and
financial named entities from headlines, and produces bar-aligned features.

All computations are causal and deterministic:
  - financial NER is a pure regex extractor over curated FX/macro categories,
  - topic modelling uses sklearn Tfidf + NMF with a fixed seed,
  - sentiment scoring uses a bundled domain lexicon (self-contained, offline),
  - bar aggregation uses a per-modality exponentially-weighted recurrence.

Features emitted per bar (all 0 when no events):
  sent_news / sent_news_count          : decayed news sentiment & event count
  sent_social / sent_social_count      : decayed social sentiment & event count
  sent_cot                             : clipped COT z-score (if provided)
  sent_fused                           : count-weighted combination of modalities
  sent_agreement                       : 1 - cross-modality dispersion (>=2 mods)
  sent_dispersion                      : std of active modalities
  sent_modalities                      : number of active modalities
  topic_{k}                            : topic-k weight from the fitted NMF model
  ner_rate_hike/ner_rate_cut/...       : decayed NER event counts per category
  ner_total_events                     : decayed total event count
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import polars as pl

# ════════════════════════════════════════════════════════════════════════════
# 1. Financial NER — lightweight pattern extractor
# ════════════════════════════════════════════════════════════════════════════

NER_CATEGORIES = [
    "rate_hike", "rate_cut", "rate_hold", "cpi", "nfp",
    "gdp", "dovish", "hawkish", "pair_mentions", "cb_mentions",
]

_CURRENCY_PAIRS = [
    "EUR/USD", "EURUSD", "GBP/USD", "GBPUSD", "USD/JPY", "USDJPY",
    "USD/CHF", "USDCHF", "AUD/USD", "AUDUSD", "NZD/USD", "NZDUSD",
    "USD/CAD", "USDCAD", "EUR/JPY", "EURJPY", "EUR/GBP", "EURGBP",
    "GBP/JPY", "GBPJPY", "AUD/JPY", "AUDJPY", "CAD/JPY", "CADJPY",
]
_CENTRAL_BANKS = [
    r"\bFed\b", r"\bFederal Reserve\b", r"\bFOMC\b", r"\bECB\b",
    r"\bEuropean Central Bank\b", r"\bBOJ\b", r"\bBank of Japan\b",
    r"\bBOE\b", r"\bBank of England\b", r"\bSNB\b", r"\bSwiss National Bank\b",
    r"\bRBA\b", r"\bReserve Bank of Australia\b", r"\bRBNZ\b",
    r"\bBank of Canada\b", r"\bBOC\b", r"\bPBOC\b", r"\bPeople's Bank of China\b",
]

_PATTERNS: dict[str, list[re.Pattern]] = {
    "rate_hike": [re.compile(r"\b(?:rate|interest)\s+hikes?\b", re.I),
                  re.compile(r"\bhikes?\s+(?:interest\s+)?rates\b", re.I),
                  re.compile(r"\b(?:hiked|raising|raise|raises)\s+(?:interest\s+)?rates\b", re.I),
                  re.compile(r"\btighten(?:ing)?\b", re.I)],
    "rate_cut": [re.compile(r"\b(?:rate|interest)\s+cuts?\b", re.I),
                 re.compile(r"\bcuts?\s+(?:interest\s+)?rates\b", re.I),
                 re.compile(r"\b(?:cut|lower|lowered|lowering|trims?)\s+(?:interest\s+)?rates\b", re.I),
                 re.compile(r"\bloosen(?:ing)?\b", re.I)],
    "rate_hold": [re.compile(r"\bhold(?:ing|s)?\s+(?:interest\s+)?rates\b", re.I),
                  re.compile(r"\bon\s+hold\b", re.I),
                  re.compile(r"\bno\s+change\b", re.I)],
    "cpi": [re.compile(r"\bCPI\b", re.I), re.compile(r"\binflation\b", re.I),
            re.compile(r"\bprice\s+index\b", re.I)],
    "nfp": [re.compile(r"\bNFP\b", re.I), re.compile(r"\b(non[- ]farm|payroll|jobs)\b", re.I),
            re.compile(r"\bunemployment\b", re.I)],
    "gdp": [re.compile(r"\bGDP\b", re.I), re.compile(r"\beconomic\s+growth\b", re.I)],
    "dovish": [re.compile(r"\bdovish\b", re.I), re.compile(r"\b(dove|accommodative)\b", re.I),
               re.compile(r"\bstimulus\b", re.I), re.compile(r"\bquantitative\s+easing\b", re.I)],
    "hawkish": [re.compile(r"\bhawkish\b", re.I), re.compile(r"\b(taper|tapering)\b", re.I),
                re.compile(r"\bquantitative\s+tightening\b", re.I)],
}

_PAIR_RE = re.compile(r"\b(?:EUR|GBP|USD|JPY|CHF|AUD|NZD|CAD)/(?:EUR|GBP|USD|JPY|CHF|AUD|NZD|CAD)\b", re.I)


def _compile_banks() -> list[re.Pattern]:
    return [re.compile(p, re.I) for p in _CENTRAL_BANKS]


_BANK_RES = _compile_banks()


def financial_ner_counts(texts: Sequence[str]) -> np.ndarray:
    """Count NER category hits per document.

    Returns an (n_docs, n_categories) int matrix; column order matches
    ``NER_CATEGORIES``.
    """
    n_cat = len(NER_CATEGORIES)
    out = np.zeros((len(texts), n_cat), dtype=np.int32)
    for d, text in enumerate(texts):
        t = str(text)
        for c, pats in _PATTERNS.items():
            hit = 0
            for p in pats:
                hit += len(p.findall(t))
            out[d, NER_CATEGORIES.index(c)] += hit
        # pair mentions (union of currency-pair tokens)
        pairs = len(_PAIR_RE.findall(t))
        for raw in _CURRENCY_PAIRS:
            pairs += len(re.findall(r"\b" + re.escape(raw) + r"\b", t))
        out[d, NER_CATEGORIES.index("pair_mentions")] += pairs
        # central-bank mentions
        cb = 0
        for p in _BANK_RES:
            cb += len(p.findall(t))
        out[d, NER_CATEGORIES.index("cb_mentions")] += cb
    return out


# ════════════════════════════════════════════════════════════════════════════
# 2. Domain lexicon sentiment (offline, deterministic, in [-1, +1])
# ════════════════════════════════════════════════════════════════════════════

_BULLISH = [
    r"\b(?:surges?|soars?|jumps?|rallies?|climbs?|advances?|gains?|strengthens?|rises?)\b",
    r"\b(?:beats?|beat)\s+(?:expectations?|estimates?|forecasts?)\b",
    r"\b(?:strong|strength|robust|expansion|recovery|growth)\b",
    r"\b(?:bullish|hawkish|dove)\b",
    r"\b(?:upgrade|positive|outlook|stimulus|easing)\b",
    r"\b(?:supports?|sparks?|boosts?|lifts?|firms?)\b",
]
_BEARISH = [
    r"\b(?:falls?|falls|plunges?|tumbles?|slumps?|drops?|slides?|weakens?|declines?|dips?)\b",
    r"\b(?:misses?|miss)\s+(?:expectations?|estimates?|forecasts?)\b",
    r"\b(?:weak|weakness|slowdown|recession|contraction|deflation)\b",
    r"\b(?:bearish|dovish|hawk)\b",
    r"\b(?:downgrade|negative|outlook|worries?|fears?|sanctions?)\b",
    r"\b(?:weighs?|pressures?|weighs?\s+on|drags?|hurts?)\b",
]


def _compile_lexicon() -> list[re.Pattern]:
    return [re.compile(p, re.I) for p in _BULLISH], [re.compile(p, re.I) for p in _BEARISH]


_BULL, _BEAR = _compile_lexicon()


def lexicon_score(texts: Sequence[str]) -> np.ndarray:
    """Lexicon sentiment in [-1, +1] per document.

    Uses the ratio of bullish vs bearish keyword hits with a small prior so the
    score stays well-conditioned for short headlines.
    """
    out = np.zeros(len(texts), dtype=float)
    for d, text in enumerate(texts):
        t = str(text)
        b = sum(len(p.findall(t)) for p in _BULL)
        be = sum(len(p.findall(t)) for p in _BEAR)
        total = b + be + 2.0
        out[d] = (b - be) / total
    return out


# ════════════════════════════════════════════════════════════════════════════
# 3. Topic model (sklearn Tfidf + NMF, deterministic)
# ════════════════════════════════════════════════════════════════════════════

def fit_topic_model(
    texts: Sequence[str],
    n_topics: int = 4,
    max_features: int = 500,
    seed: int = 0,
    top_k_words: int = 8,
) -> tuple[np.ndarray, list[list[str]]]:
    """Fit a Tfidf+NMF topic model on the corpus.

    Returns (doc_topic_weights [n_docs, n_topics], top_words [n_topics, list]).
    Each document's weights sum to 1. With fewer than ``n_topics`` documents,
    returns a uniform-weight placeholder and empty top-word lists.
    """
    from sklearn.decomposition import NMF
    from sklearn.feature_extraction.text import TfidfVectorizer

    n_docs = len(texts)
    if n_docs == 0:
        return np.zeros((0, n_topics)), [[] for _ in range(n_topics)]

    n_topics = max(1, min(int(n_topics), n_docs))
    vec = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        lowercase=True,
        ngram_range=(1, 2),
    )
    tfidf = vec.fit_transform([str(t) for t in texts])
    if tfidf.shape[1] < 1:
        w = np.full((n_docs, n_topics), 1.0 / n_topics)
        return w, [[] for _ in range(n_topics)]

    n_comp = min(n_topics, tfidf.shape[1])
    nmf = NMF(n_components=n_comp, init="nndsvda", random_state=seed, max_iter=400)
    W = nmf.fit_transform(tfidf)
    H = nmf.components_  # (n_comp, n_features)

    # normalize doc weights to sum to 1
    row_sum = W.sum(axis=1, keepdims=True)
    W = W / np.where(row_sum == 0, 1.0, row_sum)

    feature_names = np.asarray(vec.get_feature_names_out())
    top_words = []
    for k in range(n_comp):
        idx = np.argsort(H[k])[::-1][:top_k_words]
        top_words.append([str(feature_names[i]) for i in idx if i < len(feature_names)])

    if n_comp < n_topics:
        # pad remaining topic columns to a constant, keeping column count stable
        padded = np.zeros((n_docs, n_topics))
        padded[:, :n_comp] = W
        remaining = 1.0 - W.sum(axis=1)
        padded[:, n_comp:] = remaining[:, None] / max(1, n_topics - n_comp)
        W = padded
        top_words.extend([[] for _ in range(n_topics - n_comp)])
    return W, top_words


# ════════════════════════════════════════════════════════════════════════════
# 4. Bar aggregation helpers
# ════════════════════════════════════════════════════════════════════════════

def _ewma(E: np.ndarray, lam: float, dt_sec: float) -> np.ndarray:
    """Per-column exponentially-weighted aggregate: A[t] = A[t-1]*g + E[t].

    ``g = exp(-lam * dt_sec)`` and ``E`` is the raw per-bar contribution.
    """
    g = float(np.exp(-lam * dt_sec))
    out = np.zeros_like(E, dtype=float)
    prev = np.zeros(E.shape[1], dtype=float)
    for t in range(E.shape[0]):
        prev = prev * g + E[t]
        out[t] = prev
    return out


def _events_to_bars(
    events: pl.DataFrame,
    bar_ns: np.ndarray,
    time_col: str = "timestamp_utc",
) -> np.ndarray:
    """Map each event to the latest bar at/before its timestamp (ns)."""
    ev_ns = events[time_col].to_numpy().astype("datetime64[ns]").astype(np.int64)
    idx = np.searchsorted(bar_ns, ev_ns, side="right") - 1
    return idx


# ════════════════════════════════════════════════════════════════════════════
# 5. Orchestrator
# ════════════════════════════════════════════════════════════════════════════

def build_sentiment_features(
    events: pl.DataFrame,
    bar_ts: Sequence,
    source_col: str = "source",
    text_col: str = "text",
    sent_col: str | None = "sentiment",
    cot_z: pl.Series | None = None,
    lam: float = 0.05,
    dt_sec: float = 1800.0,
    n_topics: int = 4,
    seed: int = 0,
) -> pl.DataFrame:
    """Build multi-modal sentiment features aligned to ``bar_ts``.

    ``events`` : Polars DataFrame with ``timestamp_utc``, ``text`` (headline) and
    optionally ``source`` ("news"/"social") and ``sentiment`` (pre-scored). If no
    ``sentiment`` column is present, the bundled lexicon scorer is used.

    ``bar_ts`` : sequence of bar datetimes (tz-aware or naive UTC), used for
    alignment. Returns a Polars DataFrame with one row per bar and the feature
    columns described in the module docstring.
    """
    bars = list(bar_ts)
    n = len(bars)
    if n == 0:
        return pl.DataFrame()

    def _to_ns(items: Sequence) -> np.ndarray:
        out = np.empty(len(items), dtype=np.int64)
        for i, it in enumerate(items):
            if hasattr(it, "value"):          # pd.Timestamp / polars scalar
                out[i] = int(it.value)
            else:                              # datetime / date
                out[i] = int(it.timestamp() * 1e9)
        return out

    ts = _to_ns(bars)

    def _col(name: str, dtype) -> pl.Series:
        return pl.Series(name, [dtype() for _ in range(n)])

    if events is None or len(events) == 0:
        cols = _empty_feature_cols(n, n_topics)
        return cols

    texts = [str(t) for t in events[text_col].to_numpy().tolist()]
    if sent_col and sent_col in events.columns:
        scores = np.asarray(events[sent_col].to_numpy(), dtype=float)
    else:
        scores = lexicon_score(texts)

    # split by source
    if source_col and source_col in events.columns:
        src = np.asarray(events[source_col].to_numpy()).astype(str)
    else:
        src = np.full(len(events), "news", dtype=object)

    eidx = _events_to_bars(events, ts)
    valid = (eidx >= 0) & (eidx < n)

    def _agg(mask: np.ndarray, vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        E_sum = np.zeros(n)
        E_cnt = np.zeros(n)
        m = mask & valid
        if m.any():
            ii = eidx[m]
            np.add.at(E_sum, ii, vals[m])
            np.add.at(E_cnt, ii, 1.0)
        return _ewma(E_sum[:, None], lam, dt_sec)[:, 0], _ewma(E_cnt[:, None], lam, dt_sec)[:, 0]

    news_s, news_c = _agg(src == "news", scores)
    soc_s, soc_c = _agg(src == "social", scores)

    # clamp decayed means to [-1, 1]
    news_mean = np.where(news_c > 0, news_s / np.maximum(news_c, 1e-9), 0.0)
    soc_mean = np.where(soc_c > 0, soc_s / np.maximum(soc_c, 1e-9), 0.0)
    news_mean = np.clip(news_mean, -1.0, 1.0)
    soc_mean = np.clip(soc_mean, -1.0, 1.0)

    # COT modality (normalized z-score, clipped to [-2, 2] then /2)
    if cot_z is not None:
        cot = np.clip(np.asarray(cot_z.to_numpy(), dtype=float), -2.0, 2.0) / 2.0
    else:
        cot = np.zeros(n)

    # ---- fusion ----
    w_news = np.where(news_c > 0, np.clip(news_c, 0.5, 50.0), 0.0)
    w_soc = np.where(soc_c > 0, np.clip(soc_c, 0.5, 50.0), 0.0)
    w_cot = np.ones(n) if cot_z is not None else np.zeros(n)
    wtot = w_news + w_soc + w_cot
    fused = np.divide(w_news * news_mean + w_soc * soc_mean + w_cot * cot,
                      wtot, out=np.zeros_like(wtot), where=wtot > 0)
    fused = np.clip(fused, -1.0, 1.0)

    mods = (news_c > 0).astype(float) + (soc_c > 0).astype(float) + float(cot_z is not None)
    arr = np.stack([news_mean, soc_mean, cot], axis=1)
    disp = np.where(mods >= 2, arr.std(axis=1), 0.0)
    agreement = np.clip(1.0 - disp * 2.0, 0.0, 1.0)

    # ---- NER event counts (decayed) ----
    ner = np.zeros((n, len(NER_CATEGORIES)))
    if valid.any():
        counts = financial_ner_counts(texts)
        for c in range(len(NER_CATEGORIES)):
            E = np.zeros(n)
            np.add.at(E, eidx[valid], counts[valid, c])
            ner[:, c] = _ewma(E[:, None], lam, dt_sec)[:, 0]
    ner_total = ner.sum(axis=1)

    # ---- topics ----
    W, _ = fit_topic_model(texts, n_topics=n_topics, seed=seed)
    topic = np.zeros((n, n_topics))
    if valid.any():
        E = np.zeros((n, n_topics))
        np.add.at(E, eidx[valid], W[valid])
        topic = _ewma(E, lam, dt_sec)

    frames = {
        "sent_news": news_mean, "sent_news_count": news_c,
        "sent_social": soc_mean, "sent_social_count": soc_c,
        "sent_cot": cot,
        "sent_fused": fused, "sent_agreement": agreement,
        "sent_dispersion": disp, "sent_modalities": mods,
    }
    for k in range(n_topics):
        frames[f"topic_{k}"] = topic[:, k]
    for i, cat in enumerate(NER_CATEGORIES):
        frames[f"ner_{cat}"] = ner[:, i]
    frames["ner_total_events"] = ner_total

    return pl.DataFrame(frames)


def _empty_feature_cols(n: int, n_topics: int) -> pl.DataFrame:
    """Zeroed schema for the no-events case (stable column names)."""
    cols: dict[str, pl.Series] = {
        "sent_news": np.zeros(n), "sent_news_count": np.zeros(n),
        "sent_social": np.zeros(n), "sent_social_count": np.zeros(n),
        "sent_cot": np.zeros(n), "sent_fused": np.zeros(n),
        "sent_agreement": np.zeros(n), "sent_dispersion": np.zeros(n),
        "sent_modalities": np.zeros(n),
    }
    for k in range(n_topics):
        cols[f"topic_{k}"] = np.zeros(n)
    for cat in NER_CATEGORIES:
        cols[f"ner_{cat}"] = np.zeros(n)
    cols["ner_total_events"] = np.zeros(n)
    return pl.DataFrame(cols)


# ════════════════════════════════════════════════════════════════════════════
# Convenience: append features to an existing bar frame
# ════════════════════════════════════════════════════════════════════════════

def add_sentiment_features(
    bars: pl.DataFrame,
    events: pl.DataFrame,
    time_col: str = "timestamp_utc",
    **kwargs,
) -> pl.DataFrame:
    """Append multi-modal sentiment columns to a bar DataFrame in place."""
    feat = build_sentiment_features(events, bars[time_col].to_list(), **kwargs)
    new_cols = [c for c in feat.columns if c not in bars.columns]
    if new_cols:
        bars = pl.concat([bars, feat.select(new_cols)], how="horizontal_extend")
    return bars
