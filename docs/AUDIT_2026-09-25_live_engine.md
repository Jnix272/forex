# Audit: Live Trading Engine (2026-09-25)

**Scope:** `trading/live_engine.py` (3,847 lines), mainly `LiveTradingEngine._on_new_bar`, the broker adapters (OANDA, Paper, LMAX, Bridge), `LiveSafetyGate`, the multi-pair observation assembly and the CLI gate. Guards in `trading/live_guards.py` and the sizing in `risk/execution.py` were read where the engine calls them.

**Method:** static code review against `main` @ `cc344ab`. Nothing was run against a broker.

**Bottom line:**
- The engine has many layers of guards.
- A few paths can still leave a real position with no stop, or trade on an account without a promoted model.
- The live inputs don't match what the models were trained on, so live predictions can't be trusted even with a good model.

---

## P0: money at risk

### L1. `--demo` with a real broker trades random actions and skips the promotion gate
- The gate check is `if args.broker != "paper" and not args.demo:`.
- `--demo --broker oanda` therefore skips the gate. With `OANDA_ENV=live` it sends orders to `api-fxtrade`.
- The demo agent's `select_action` returns `randint(0, 2)`.
- **Fix:** refuse `--demo` with any broker except `paper`, or force `OANDA_ENV=practice` in demo mode.

### L2. OANDA positions have no broker-side stop
- For OANDA, `_place` sends orders without `stopLossOnFill` or `takeProfitOnFill` (`attach_stops = with_stops and not is_oanda`), citing the NFA FIFO rule.
- The only stop is a software check that runs **once per bar**, on the mid price, and only while the process is running.
- If the process crashes, the network drops or the machine sleeps, the position has no stop. Between bars a 5-minute move can blow far past the stop.
- With one position per instrument, `stopLossOnFill` does not break FIFO. If the account rejects it, place a separate STOP order, or a guaranteed stop.

### L3. An order can be live at OANDA while the engine thinks it failed
- `market_order` returns `network_error` on a timeout, and the POST is not idempotent (no `clientExtensions.id`).
- A timed-out order may still have filled.
- The engine treats it as failed. The next bar's `_reconcile_positions` then adopts the broker size, but with `_entry_price = 0`.
- The software stop requires `self._entry_price > 0` (L4), so that position is never stopped out.
- **Fix:** send a client order id. On timeout, query the order by that id before deciding. Always take the entry price from the broker (`averagePrice` or the fill price).

### L4. Adopted or resized positions have no entry price, so they're never stopped
- This happens at startup ("Adopted pre-existing broker position") and in `_reconcile_positions` on a size mismatch.
- Both set `_position` but leave `_entry_price` at 0 or stale.
- The software SL/TP runs only `if ... self._entry_price > 0`.
- **Fix:** read the average price from OANDA `/positions` and set `_entry_price`. Refuse to trade the pair until it's known.

### L5. The minimum lot size overrides the risk sizer
- The confidence scaling ends with `np.clip(lots * ..., 0.02, self.max_lots)`.
- When Kelly, VaR, drawdown or regime sizing returns 0 lots (meaning don't trade), this raises it to 0.02 and the engine trades anyway.
- **Fix:** keep 0 as 0: `lots = 0 if lots <= 0 else clip(..., min, max)`.

### L6. Flatten paths ignore failed closes
- These paths all call `close_position` and then set `_position = 0` without checking the result:
  - drawdown guard (`dae` FLATTEN / HALT)
  - calendar flatten
  - weekend square-off
  - risk circuit breaker, which checks the result but returns either way
- A failed close leaves a real position that the engine believes is flat.
- **Fix:** keep the position until the broker confirms, and retry or alert.

### L7. Live features are in the wrong units and the wrong basis (partly caused by my audit fixes)
- **`ret_5` units:** the engine feeds `ret_5` into parametric VaR and `rck.size` as a *price-fraction* return. After the 2026-09-25 feature fix, `ret_5` is in **basis points** (×1e4). VaR is inflated about 10,000×, so `size_adj` is always 0.5 and the Kelly volatility input is wrong. **Fixed:** the engine now computes returns from `close`.
- **Basis flip:** "currency basis normalization" flips BUY and SELL for USDJPY and USDCAD, on the assumption that the model predicts a USD-neutral consensus. With per-pair heads (now on in `run.yaml`), each head already predicts its own pair, so the flip would invert correct signals. **Fixed:** the flip is skipped when the model has per-pair heads.

---

## P1: live inputs don't match training

### L8. Features are built from a short buffer of polled quotes
- **History too short:** training features used 14 days of warm-up and 5-day windows of Dukascopy ticks. Live builds them from `LiveTickBuffer(max_bars=500)`, seeded with only **120** historical candles.
  - Anything with a lookback above about 500 bars is wrong or empty live: the 7-day volatility clock (2,016 bars), 200/240-bar regime and volume-profile windows, and HTF 1h features.
  - Training starts at `len(bars) >= 70`, while the model's `seq_len` is 120.
- **Fake ticks:** live "ticks" are `get_bid_ask` polls every 0.1 s with `volume=1`. Training used real tick counts and volumes, so volume-based features differ in scale and meaning live: amihud, VPIN, OFI, VWAP, `volume` itself.
- **Fix:** seed at least 14 days of M5 candles. Rebuild features with the same windows as training. Either drop tick-volume features or use OANDA's candle volume consistently in both training and live.

### L9. Hard-coded feature order and pair slots
- `_feature_columns` uses a hard-coded `CANONICAL_PAIR_146` list, unless `_expected_features` is loaded.
- `_format_obs` places pairs in the hard-coded order `["EURUSD","USDJPY","GBPUSD","USDCAD"]`.
- Training's order comes from the cache's `_feature_schema.json` and the `pair_ticks` order.
- Any mismatch silently feeds one pair's features into another pair's slot. Nothing checks names against the checkpoint's schema; `schema_hash` is "unknown" on every checkpoint.
- **Fix:** store the ordered feature names and pair order in each checkpoint and assert them live.

### L10. The multi-pair view is stale or zero-filled
- The other pairs' slots come from `shared_pair_features`, which holds whatever that pair's engine computed last. That may be the previous bar, or nothing: missing pairs are zero-filled silently.
- Warm-up rows fill the other pairs' slots with their *current* observation for every historical step.
- **Fix:** build all pairs' features for the same bar timestamp before any pair decides, and block if any pair is missing or stale.

### L11. Live sentiment differs from training sentiment
- Live `finbert_sentiment` scores the 12 latest headlines on every bar.
- Training used per-timestamp averages of archived FinBERT scores, and nothing at all before 2021.
- The live input's distribution differs from anything the model saw.

### L12. Stops and sizing use made-up numbers
- `rck.size(equity, 0.55, 1.5, ...)` hard-codes a 55% win rate and a 1.5 payoff. Kelly sizing on invented edge is not sizing.
- The stop is `stop_loss_atr × atr_6`. On 5-minute bars, ATR(6) is a few pips, so with SL 0.8 × ATR the stop sits close to the spread.
- The entry price is the pre-trade **mid**, not the fill price, so PnL, SL and TP are off by at least half the spread.

---

## P2: correctness and operations

- **L13.** `pnl = equity − prev_equity` runs every bar, including unrealised mark-to-market. It feeds `demotion.on_trade_closed` as if a trade had closed, which pollutes the demotion stats (win rate and Sharpe).
- **L14.** `LiveSafetyGate` resets `halted` every UTC day. Max daily loss is 5% and it's measured against equity at the day's first bar. There's no weekly or total drawdown stop at this layer.
- **L15.** The promotion gate is checked only at startup. Staleness is checked only for the ensemble certification file, and per-model `promotion_gate.json` files are never checked against checkpoint mtimes. Hot reload (`_maybe_hot_reload`) can swap in a model that was never promoted.
- **L16.** Scale-out and scale-in orders don't update the entry price or the risk engine consistently, and they skip the safety gate's rate limit and spread check.
- **L17.** The main loop polls `get_bid_ask` every 0.1 s. On OANDA REST that's 10 requests/s per pair. Use the streaming price endpoint, which the ZMQ subscriber partly does.
- **L18.** `preflight_check.py` still isn't wired into startup (open item M4 from the earlier audit).
- **L19.** Many exceptions are swallowed silently: hedge update, position probe, `_other.select_action`, the rate-limit record. At minimum, count them and alert.

---

## Recommended order

1. **Before any real-money run:** L1, L2, L3, L4, L5 and L6.
2. **Before trusting live predictions:** L8, L9 and L10. Add a replay test that feeds a day of historical ticks through the live feature path and compares the result to the training cache, column by column.
3. **Then:** L12 (sizing from measured win rate and payoff), then L13–L19.

---

## Fix log

| # | Fix |
|---|---|
| L1 | `--demo` is refused with any broker other than `paper`. |
| L2 | OANDA orders request broker-side SL/TP. If the brackets are rejected (FIFO), the plain order is placed and SL/TP are attached to the opened trade (`set_trade_stops`). A failed attach logs `broker_stops_missing`. |
| L3 | Orders carry a client id. On a timeout the order is looked up by that id (FILLED, CANCELLED or unknown) instead of being reported as failed. |
| L4 | Adopted and reconciled positions take the broker's `averagePrice` as entry, or the mid as a fallback. Fills record the fill price. |
| L5 | A sizing decision of 0 stays 0; the 0.02-lot minimum applies only to positive sizes. |
| L6 | `_flatten()` clears state only after the broker confirms the close. A failed close halts new orders and is retried every bar. |
| L7 | Returns come from `close` (`ret_5` is now in bps). The basis flip is skipped for per-pair-head models. |
| L8 | Startup seeds about 14 days of candles into a 5,000-bar buffer. Volume counts quote changes. Entries wait for `seq_len + 60` bars. Absolute-volume features are neutralised in the training scaler. |
| L9 | Checkpoints save `<stem>_features.json` (copied on promotion). Live adopts that column order and refuses a pair-order mismatch. |
| L10 | Multi-pair bars run in two phases (publish, then decide). A pair blocks on stale or missing peer features, and multi-pair entries wait for a full live window. |
| L11 | Live per-bar sentiment is passed to `fe.build(sentiment=...)` as a timestamped series, as in training. |
| L12 | Kelly inputs come from the engine's own closed trades, shrunk toward a no-edge prior. The sizer uses per-pair pip size and USD pip value, and annualises at the bar frequency. Stops have a floor of 3× the spread. |
| L13 | Demotion and Prometheus trade stats update per closed trade, not per bar. |
| L14 | `LiveSafetyGate` has a 15% drawdown-from-peak halt that survives the daily reset. |
| L15 | `check_promotion()` requires the gate artifact to be newer than the checkpoints it authorises. It's used at startup and on hot reload (real brokers). Hot reload also re-checks the feature contract. |
| L16 | Scale-in and scale-out go through the safety gate and the rate limit. Scale-in keeps a weighted-average entry. |
| L17 | Quote polling runs every 0.5 s on REST and every 0.1 s only with the ZMQ stream or the paper broker. |
| L18 | `run_preflight` runs before real-broker starts: feed, broker, risk limits, model, and a schema hash taken from `_features.json`. |
| L17b | OANDABroker has a pure-Python v20 pricing stream (on by default when no ZMQ; `OANDA_STREAM=0` disables it). Heartbeats count as liveness, it reconnects with backoff, and REST falls back only while the cached quote is under 10 s old. |
| L19 | A broker position error blocks the order. Errors from the peer model are logged. |

**Still open:**
- Warm-up rows for multi-pair models: entries are gated instead of rebuilding historical peer rows.
- A replay test of the live feature path against the training cache.

**Note:** real-broker runs now require a checkpoint trained after this change (it needs `_features.json`), so preflight will block the current checkpoints.
