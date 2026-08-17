"""
risk/fx_greeks.py - FX Greeks for option-like exposures

Computes delta, gamma, theta, vega, rho for standard FX option models on an
underlying spot and a strike, plus position-adjusted portfolio Greeks.

Conventions
-----------
  * FX delta is expressed in base-currency units (amount of base currency).
  * JPY-quoted pairs (USDJPY, EURJPY, ...) flip the usual delta formula: the
    notional is quoted in JPY, so per-1bp-of-spot delta scales by 1/(spot^2)
    and vega is scaled by 1/spot. The ``is_jpy`` flag handles this.
  * Model: Black-76 with an FX-style forward and normal vol per sqr-year.

These helpers are also used by portfolio_monitor to size correlation-aware
option-like exposure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _norm_cdf(x: float) -> float:
    try:
        from scipy.stats import norm

        return float(norm.cdf(x))
    except ImportError:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    try:
        from scipy.stats import norm

        return float(norm.pdf(x))
    except ImportError:
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


@dataclass
class FxGreeks:
    """Greeks for a single FX option-like position."""

    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    price: float
    spot: float
    strike: float
    tenor_years: float
    is_jpy: bool = False

    def to_dict(self) -> dict:
        return {
            "delta": round(self.delta, 6),
            "gamma": round(self.gamma, 8),
            "theta": round(self.theta, 6),
            "vega": round(self.vega, 6),
            "rho": round(self.rho, 6),
            "price": round(self.price, 6),
            "spot": self.spot,
            "strike": self.strike,
            "tenor_years": self.tenor_years,
            "is_jpy": self.is_jpy,
        }


def _is_jpy_pair(pair: str) -> bool:
    """USDJPY, EURJPY, GBPJPY ... are quoted in JPY (100 JPY = 1 pip)."""
    p = (pair or "").upper()
    return p.endswith("JPY")


class FxOptionGreeks:
    """Black-76 greeks for FX options with base-currency delta and JPY quoting.

    Parameters
    ----------
    spot, strike : spot and strike price in quote units.
    tenor_years  : time to expiry in years.
    vol          : annualised implied volatility (0.10 == 10%).
    rate_dom     : domestic (base) interest rate.
    rate_for     : foreign (quote) interest rate.
    is_jpy       : True when the pair is JPY-quoted (USDJPY, EURJPY, ...).
    """

    def __init__(
        self,
        spot: float,
        strike: float,
        tenor_years: float = 1.0,
        vol: float = 0.10,
        rate_dom: float = 0.02,
        rate_for: float = 0.01,
        is_jpy: bool = False,
    ):
        self.spot = float(spot)
        self.strike = float(strike)
        self.t = float(tenor_years)
        self.vol = float(vol)
        self.rd = float(rate_dom)
        self.rf = float(rate_for)
        self.is_jpy = bool(is_jpy)

    # ── Black-76 pricing on the FX forward ────────────────────────────────

    def forward(self) -> float:
        return self.spot * math.exp((self.rd - self.rf) * self.t)

    def _d1d2(self) -> tuple:
        t, v, _s, k = self.t, self.vol, self.spot, self.strike
        if t <= 0 or v <= 1e-9:
            raise ValueError("tenor must be > 0 and vol > 0 to price greeks")
        f = self.forward()
        vt = v * math.sqrt(t)
        d1 = (math.log(f / k) + 0.5 * v * v * t) / vt
        d2 = d1 - vt
        return d1, d2

    def price(self, call: bool = True) -> float:
        f = self.forward()
        d1, d2 = self._d1d2()
        if call:
            return math.exp(-self.rd * self.t) * (f * _norm_cdf(d1) - self.strike * _norm_cdf(d2))
        return math.exp(-self.rd * self.t) * (self.strike * _norm_cdf(-d2) - f * _norm_cdf(-d1))

    # ── greeks ────────────────────────────────────────────────────────────

    def greeks(self, call: bool = True) -> FxGreeks:
        """Return greeks for a single option on 1.0 unit of base currency.

        Delta for a standard (non-JPY) pair is 1 unit of base per 1.0 move in
        spot (i.e. a delta of 1.0 == 1.0 base notional). For JPY-quoted pairs
        delta is scaled by 1/spot so a 1 bp move maps to ~1 base unit, and
        vega is scaled by 1/spot to keep the *value* per unit vol consistent.
        """
        d1, d2 = self._d1d2()
        t, v, s = self.t, self.vol, self.spot
        phi = 1.0 if call else -1.0

        delta_raw = math.exp(-self.rf * t) * _norm_cdf(phi * d1)
        gamma_raw = math.exp(-self.rf * t) * _norm_pdf(d1) / (s * v * math.sqrt(t))
        vega_raw = s * math.exp(-self.rf * t) * _norm_pdf(d1) * math.sqrt(t)
        theta_raw = -(
            s * math.exp(-self.rf * t) * _norm_pdf(d1) * v / (2.0 * math.sqrt(t))
            + phi * self.rf * s * math.exp(-self.rf * t) * _norm_cdf(phi * d1)
            - phi * self.rd * self.strike * math.exp(-self.rd * t) * _norm_cdf(phi * d2)
        )
        rho_raw = phi * self.t * self.strike * math.exp(-self.rd * t) * _norm_cdf(phi * d2)

        # JPY quoting adjustments (base-currency consistent)
        if self.is_jpy:
            delta = delta_raw / s
            gamma = gamma_raw / s
            vega = vega_raw / s
        else:
            delta = delta_raw
            gamma = gamma_raw
            vega = vega_raw

        price = self.price(call)
        return FxGreeks(
            delta=phi * delta,
            gamma=gamma,
            theta=-theta_raw,
            vega=vega,
            rho=rho_raw,
            price=price,
            spot=s,
            strike=self.strike,
            tenor_years=t,
            is_jpy=self.is_jpy,
        )


def compute_greeks(
    pair: str,
    spot: float,
    strike: float,
    tenor_years: float = 1.0,
    vol: float = 0.10,
    rate_dom: float = 0.02,
    rate_for: float = 0.01,
    call: bool = True,
    quantity: float = 1.0,
) -> FxGreeks:
    """One-call helper: greeks for a position of ``quantity`` base units."""
    g = FxOptionGreeks(
        spot=spot,
        strike=strike,
        tenor_years=tenor_years,
        vol=vol,
        rate_dom=rate_dom,
        rate_for=rate_for,
        is_jpy=_is_jpy_pair(pair),
    ).greeks(call=call)
    g.delta *= quantity
    g.gamma *= quantity
    g.vega *= quantity
    g.theta *= quantity
    g.rho *= quantity
    g.price *= quantity
    return g


class PortfolioGreeks:
    """Position-adjusted aggregate greeks across a book of FX option-like
    exposures.

    Each leg: dict with keys ``pair, spot, strike, tenor_years, vol,
    quantity, call, rate_dom, rate_for``.
    """

    def __init__(self, legs: list[dict] | None = None):
        self.legs: list[dict] = list(legs or [])

    def add(self, leg: dict) -> None:
        self.legs.append(leg)

    def aggregate(self) -> dict:
        total = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}
        per_leg: list[dict] = []
        for leg in self.legs:
            g = compute_greeks(
                pair=leg["pair"],
                spot=float(leg["spot"]),
                strike=float(leg["strike"]),
                tenor_years=float(leg.get("tenor_years", 1.0)),
                vol=float(leg.get("vol", 0.10)),
                rate_dom=float(leg.get("rate_dom", 0.02)),
                rate_for=float(leg.get("rate_for", 0.01)),
                call=bool(leg.get("call", True)),
                quantity=float(leg.get("quantity", 1.0)),
            )
            per_leg.append(g.to_dict())
            for k in total:
                total[k] += getattr(g, k)
        return {
            "delta": round(total["delta"], 6),
            "gamma": round(total["gamma"], 8),
            "theta": round(total["theta"], 6),
            "vega": round(total["vega"], 6),
            "rho": round(total["rho"], 6),
            "legs": per_leg,
            "n_legs": len(self.legs),
        }

    def net_delta_by_currency(self) -> dict[str, float]:
        """Net base-currency delta across legs, keyed by the 3-letter base code."""
        out: dict[str, float] = {}
        for leg in self.legs:
            pair = (leg.get("pair") or "").upper()
            if len(pair) != 6:
                continue
            g = compute_greeks(
                pair=pair,
                spot=float(leg["spot"]),
                strike=float(leg["strike"]),
                tenor_years=float(leg.get("tenor_years", 1.0)),
                vol=float(leg.get("vol", 0.10)),
                rate_dom=float(leg.get("rate_dom", 0.02)),
                rate_for=float(leg.get("rate_for", 0.01)),
                call=bool(leg.get("call", True)),
                quantity=float(leg.get("quantity", 1.0)),
            )
            out[pair[:3]] = out.get(pair[:3], 0.0) + g.delta
        return {k: round(v, 6) for k, v in out.items()}
