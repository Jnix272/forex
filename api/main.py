"""
Proposed api/main.py
FastAPI application interface for Forex Scaling Model risk & kelly sizing.
"""

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sizing.kelly_criterion import PositionSizer, vol_target_scalar

app = FastAPI(
    title="Forex Scaling Model API",
    description="Milestone 2 Risk and Sizing API Contract",
    version="1.0.0",
)


class KellySizingRequest(BaseModel):
    win_prob: float = Field(..., description="Winning probability, float between 0.0 and 1.0", ge=0.0, le=1.0)
    win_loss_ratio: float = Field(..., description="Win/loss ratio, positive float", gt=0.0)
    returns: list[float] = Field(..., description="List of historical returns (floats)")
    price: float = Field(..., description="Current asset price, positive float", gt=0.0)
    current_atr: float = Field(..., description="Current Average True Range (ATR), positive float", gt=0.0)
    equity: float = Field(..., description="Current portfolio equity, positive float", gt=0.0)
    lot_size: float = Field(10000.0, description="Standard lot contract size (e.g. 10000 or 100000)", gt=0.0)

    # Optional PositionSizer configuration parameter overrides
    kelly_fraction: float | None = Field(
        None, description="Fractional Kelly multiplier (default: 0.25)", gt=0.0, le=1.0
    )
    max_position_pct: float | None = Field(
        None, description="Maximum position risk percentage of equity (default: 0.05)", gt=0.0, le=1.0
    )
    target_vol: float | None = Field(None, description="Target annual volatility (default: 0.10)", gt=0.0)
    pip_risk: float | None = Field(None, description="Minimum stop loss in pips (default: 20.0)", gt=0.0)


class KellySizingResponse(BaseModel):
    lots: float = Field(..., description="Calculated position size in lots")
    full_kelly: float = Field(..., description="Unscaled/full Kelly fraction")
    frac_kelly: float = Field(..., description="Scaled/fractional Kelly fraction")
    vol_scalar: float = Field(..., description="Volatility scaling factor")
    risk_usd: float = Field(..., description="Total capital risk in USD")
    impact_usd: float = Field(..., description="Estimated market impact in USD")


class VolatilityBoundsRequest(BaseModel):
    returns: list[float] = Field(..., description="List of historical returns (floats)")
    target_vol: float = Field(0.10, description="Target annual volatility, positive float", gt=0.0)
    lookback: int = Field(20, description="Lookback window size, positive integer", gt=1)


class VolatilityBoundsResponse(BaseModel):
    vol_scalar: float = Field(..., description="Calculated volatility scaling multiplier")
    realized_vol: float = Field(..., description="Annualized realized volatility computed over the lookback window")


@app.post("/kelly_sizing", response_model=KellySizingResponse)
def kelly_sizing(payload: KellySizingRequest):
    """
    Computes position sizing based on fractional Kelly criterion, volatility targeting,
    and market impact estimation. Maps inputs to the PositionSizer class.
    """
    import math

    # Validate all float parameters are finite
    for val, name in [
        (payload.win_prob, "win_prob"),
        (payload.win_loss_ratio, "win_loss_ratio"),
        (payload.price, "price"),
        (payload.current_atr, "current_atr"),
        (payload.equity, "equity"),
        (payload.lot_size, "lot_size"),
    ]:
        if not math.isfinite(val):
            raise HTTPException(status_code=400, detail=f"Parameter {name} must be a finite float")

    for val, name in [
        (payload.kelly_fraction, "kelly_fraction"),
        (payload.max_position_pct, "max_position_pct"),
        (payload.target_vol, "target_vol"),
        (payload.pip_risk, "pip_risk"),
    ]:
        if val is not None and not math.isfinite(val):
            raise HTTPException(status_code=400, detail=f"Parameter {name} must be a finite float")

    # Validate all returns are finite floats
    for i, r in enumerate(payload.returns):
        if r is None or not math.isfinite(r):
            raise HTTPException(status_code=400, detail=f"Item at index {i} in returns must be a finite float")

    # In /kelly_sizing, if win_prob == 0 (since Pydantic ge=0.0 prevents < 0), ensure position size is lots = 0.0
    if payload.win_prob <= 0:
        return KellySizingResponse(
            lots=0.0,
            full_kelly=0.0,
            frac_kelly=0.0,
            vol_scalar=0.0,
            risk_usd=0.0,
            impact_usd=0.0,
        )

    try:
        # Construct PositionSizer with request parameters or defaults
        sizer_args = {"equity": payload.equity}
        if payload.kelly_fraction is not None:
            sizer_args["kelly_fraction"] = payload.kelly_fraction
        if payload.max_position_pct is not None:
            sizer_args["max_position_pct"] = payload.max_position_pct
        if payload.target_vol is not None:
            sizer_args["target_vol"] = payload.target_vol
        if payload.pip_risk is not None:
            sizer_args["pip_risk"] = payload.pip_risk

        sizer = PositionSizer(**sizer_args)

        # Clean returns list to filter out non-finite float values (NaN, Inf, -Inf)
        clean_returns = [r for r in payload.returns if r is not None and np.isfinite(r)]

        # Call position sizing logic
        result = sizer.size_position(
            win_prob=payload.win_prob,
            win_loss_ratio=payload.win_loss_ratio,
            returns=clean_returns,
            price=payload.price,
            current_atr=payload.current_atr,
            lot_size=payload.lot_size,
        )

        return KellySizingResponse(
            lots=result["lots"],
            full_kelly=result["full_kelly"],
            frac_kelly=result["frac_kelly"],
            vol_scalar=result["vol_scalar"],
            risk_usd=result["risk_usd"],
            impact_usd=result["impact_usd"],
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Sizing calculation failed: {e!s}")


@app.post("/volatility_bounds", response_model=VolatilityBoundsResponse)
def volatility_bounds(payload: VolatilityBoundsRequest):
    """
    Computes realized volatility and the volatility scaling factor.
    Calls vol_target_scalar inside sizing/kelly_criterion.py.
    """
    import math

    # Validate target_vol is finite
    if not math.isfinite(payload.target_vol):
        raise HTTPException(status_code=400, detail="Parameter target_vol must be a finite float")

    # Validate returns list contains only finite floats
    for i, r in enumerate(payload.returns):
        if r is None or not math.isfinite(r):
            raise HTTPException(status_code=400, detail=f"Item at index {i} in returns must be a finite float")

    try:
        # Clean returns list to filter out non-finite float values (NaN, Inf, -Inf)
        clean_returns = [r for r in payload.returns if r is not None and np.isfinite(r)]
        returns_np = np.array(clean_returns)

        # Guard against empty or single-element arrays
        if len(returns_np) < 2:
            return VolatilityBoundsResponse(vol_scalar=1.0, realized_vol=0.0)

        # Compute realized vol to align with vol_target_scalar logic:
        # realized_vol = std(recent) * sqrt(252)
        lookback = payload.lookback
        recent = returns_np[-lookback:] if len(returns_np) >= lookback else returns_np
        realized_vol = float(np.std(recent) * np.sqrt(252))

        # Call the canonical sizing function
        scalar = vol_target_scalar(
            returns=returns_np,
            target_vol=payload.target_vol,
            lookback=payload.lookback,
        )

        return VolatilityBoundsResponse(
            vol_scalar=scalar,
            realized_vol=realized_vol,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Volatility bounds calculation failed: {e!s}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
