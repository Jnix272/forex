"""
risk/portfolio_monitor.py — Portfolio-level risk aggregation

Aggregates exposure across pairs, computes net currency exposure,
correlation-aware exposure, and liquidity tiering. Designed to sit on top of
RiskEngine.position/exposure snapshots and feed the alerting entry point.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    from config.settings import LIVE_RISK as _LR
except ImportError:
    _LR = {}


# Liquidity tiers by average daily volume (lots/day). Tier 1 is the deepest.
DEFAULT_LIQUIDITY_TIERS: Dict[str, int] = {
    "EURUSD": 1, "GBPUSD": 1, "USDJPY": 1,
    "AUDUSD": 2, "USDCAD": 2, "USDCHF": 2,
    "NZDUSD": 2, "EURGBP": 2, "EURJPY": 3,
    "GBPJPY": 3, "AUDJPY": 3, "EURAUD": 3,
    "CADJPY": 3, "CHFJPY": 3, "AUDNZD": 4,
    "EURCHF": 4, "GBPCHF": 4, "NZDJPY": 4,
}


@dataclass
class ExposureSnapshot:
    """Snapshot of positions + returns history used for aggregation."""
    positions: Dict[str, Dict]                    # pair -> {lots, entry_price, direction}
    returns: Optional[Dict[str, np.ndarray]] = None  # pair -> return series (optional)


class PortfolioMonitor:
    """Aggregate exposure, net currency exposure, correlation-aware exposure
    and liquidity tiering across a multi-pair book."""

    def __init__(
        self,
        liquidity_tiers: Optional[Dict[str, int]] = None,
        corr_threshold: float = 0.60,
        liquidity_penalty: float = 0.5,
    ):
        self.liquidity_tiers = dict(DEFAULT_LIQUIDITY_TIERS)
        if liquidity_tiers:
            self.liquidity_tiers.update(liquidity_tiers)
        self.corr_threshold = corr_threshold
        self.liquidity_penalty = liquidity_penalty

    # ── exposure aggregation ──────────────────────────────────────────────

    def aggregate_exposure(self, positions: Dict[str, Dict]) -> Dict:
        """Total / net notional exposure by pair and currency."""
        total_lots = 0.0
        by_pair: Dict[str, float] = {}
        by_currency: Dict[str, float] = {}
        notional_usd = 0.0
        for pair, pos in positions.items():
            lots = abs(float(pos.get("lots", 0.0)))
            price = float(pos.get("entry_price", 1.0))
            direction = pos.get("direction", "long")
            sign = 1.0 if direction == "long" else -1.0
            total_lots += lots
            notional_usd += lots * 100_000 * price
            by_pair[pair] = round(sign * lots, 4)
            if len(pair) == 6:
                base, quote = pair[:3], pair[3:]
                by_currency[base] = by_currency.get(base, 0.0) + sign * lots * 100_000
                by_currency[quote] = by_currency.get(quote, 0.0) - sign * lots * 100_000
        return {
            "total_lots": round(total_lots, 4),
            "notional_usd": round(notional_usd, 2),
            "by_pair_lots": by_pair,
            "net_currency_notional": {k: round(v, 2) for k, v in by_currency.items()},
            "n_pairs": len([p for p in positions if abs(float(positions[p].get("lots", 0.0))) > 1e-9]),
        }

    def liquidity_exposure(self, positions: Dict[str, Dict]) -> Dict:
        """Lots split by liquidity tier, plus illiquid exposure penalty."""
        tier_lots: Dict[int, float] = {}
        for pair, pos in positions.items():
            lots = abs(float(pos.get("lots", 0.0)))
            tier = self.liquidity_tiers.get(pair.upper(), 4)
            tier_lots[tier] = tier_lots.get(tier, 0.0) + lots
        total = float(sum(tier_lots.values()))
        illiquid = tier_lots.get(3, 0.0) + tier_lots.get(4, 0.0)
        return {
            "tier_lots": {int(k): round(v, 4) for k, v in sorted(tier_lots.items())},
            "total_lots": round(total, 4),
            "illiquid_lots": round(illiquid, 4),
            "illiquid_pct": round(illiquid / total, 4) if total > 0 else 0.0,
            "liquidity_adjusted_lots": round(total - self.liquidity_penalty * illiquid, 4),
        }

    def correlation_exposure(self, positions: Dict[str, Dict], returns: Dict[str, np.ndarray]) -> Dict:
        """Correlation-aware exposure: identifies pairs whose returns are highly
        correlated (same-direction risk concentration) and flags net directional
        exposure in the correlated cluster."""
        pairs = [p for p, pos in positions.items() if abs(float(pos.get("lots", 0.0))) > 1e-9]
        rows: List[np.ndarray] = []
        valid: List[str] = []
        min_len = None
        for p in pairs:
            r = returns.get(p)
            if r is None or len(r) == 0:
                continue
            arr = np.asarray(r, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if arr.size < 20:
                continue
            valid.append(p)
            rows.append(arr)
            min_len = arr.size if min_len is None else min(min_len, arr.size)
        if not valid or min_len is None:
            return {"high_corr_clusters": [], "correlation_avg": 0.0, "max_pair_corr": 0.0}
        rows = np.array([row[-min_len:] for row in rows])
        corr = np.corrcoef(rows) if len(valid) > 1 else np.array([[1.0]])

        edges = []
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                if abs(float(corr[i, j])) >= self.corr_threshold:
                    edges.append((valid[i], valid[j], round(float(corr[i, j]), 4)))

        triu = np.triu_indices(len(valid), k=1)
        tri_values = corr[triu] if len(valid) > 1 else np.array([])
        avg_corr = float(tri_values.mean()) if tri_values.size else 0.0

        # Greedy clustering on high-corr edges
        clusters: List[List[str]] = []
        for a, b, _ in edges:
            merged = False
            for cl in clusters:
                if a in cl or b in cl:
                    cl.extend([x for x in (a, b) if x not in cl])
                    merged = True
                    break
            if not merged:
                clusters.append([a, b])

        cluster_lots = []
        for cl in clusters:
            lots = sum(abs(float(positions[p].get("lots", 0.0))) for p in cl)
            cluster_lots.append({"pairs": cl, "lots": round(lots, 4)})

        max_corr = float(np.max(np.abs(tri_values))) if tri_values.size else 0.0
        return {
            "high_corr_clusters": cluster_lots,
            "correlation_avg": round(avg_corr, 4),
            "max_pair_corr": round(max_corr, 4),
            "n_high_corr_edges": len(edges),
        }

    # ── one-call report ───────────────────────────────────────────────────

    def report(
        self,
        positions: Dict[str, Dict],
        returns: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict:
        """Full portfolio risk report (exposure + liquidity + correlation)."""
        report = {
            "exposure": self.aggregate_exposure(positions),
            "liquidity": self.liquidity_exposure(positions),
        }
        if returns:
            report["correlation"] = self.correlation_exposure(positions, returns)
        return report
