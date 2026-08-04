"""
monitoring/discord_alerts.py
==============================
Discord webhook alerter for the Forex Scaling Model live engine.

Sends structured Discord embeds for 6 alert types:
  🔴 circuit_breaker  — DrawdownAwareExitManager fires close_all
  🌊 drift_detected   — DriftDetector fires
  🔄 emergency_retrain — Retrain DAG triggered by demotion monitor
  ✅ model_promoted   — PromotionGate.evaluate() passes
  ⬇️  model_demoted   — DemotionMonitor triggers rollback
  💸 tca_breach       — Slippage/cost metrics exceed policy limit

Falls back to print() when DISCORD_WEBHOOK_URL is not set.

Usage:
    from monitoring.discord_alerts import DiscordAlerter
    alerter = DiscordAlerter()

    alerter.send("circuit_breaker", {
        "drawdown": "10.5%",
        "equity":   "$8,950",
        "action":   "close_all",
    })

    alerter.send("model_promoted", {
        "model":   "haelt_v4",
        "sharpe":  "1.82",
        "git":     "a3f9c1d",
    })
"""

import json
import os
import time
from datetime import UTC, datetime
from typing import Any

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    import urllib.error
    import urllib.request
    REQUESTS_AVAILABLE = False

WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
USER_ID     = os.getenv("DISCORD_USER_ID", "")

# ── alert definitions ────────────────────────────────────────────────────────

ALERT_CONFIG: dict[str, dict[str, Any]] = {
    "circuit_breaker": {
        "emoji":       "🔴",
        "title":       "Circuit Breaker Triggered",
        "description": "Drawdown limit breached. All positions closed.",
        "color":       0xFF0000,  # red
    },
    "drift_detected": {
        "emoji":       "🌊",
        "title":       "Feature Drift Detected",
        "description": "Model input distribution has shifted significantly.",
        "color":       0xFF8C00,  # orange
    },
    "retrain_started": {
        "emoji":       "🔄",
        "title":       "Emergency Retrain Started",
        "description": "Demotion triggered automatic retraining DAG.",
        "color":       0xFFD700,  # gold
    },
    "production_deploy_completed": {
        "emoji":       "✅",
        "title":       "Production Deploy Completed",
        "description": "New model passed all promotion gates and is now live.",
        "color":       0x00CC44,  # green
    },
    "production_deploy_failed": {
        "emoji":       "❌",
        "title":       "Production Deploy Failed",
        "description": "Critical failure occurred while copying production checkpoints.",
        "color":       0xFF0000,  # red
    },
    "model_demoted": {
        "emoji":       "⬇️",
        "title":       "Model Demoted — Rolling Back",
        "description": "Live performance fell below policy thresholds.",
        "color":       0xAA00FF,  # purple
    },
    "tca_breach": {
        "emoji":       "💸",
        "title":       "TCA Policy Breach",
        "description": "Transaction costs exceeded allowed % of gross P&L.",
        "color":       0xFF4444,  # light red
    },
    "training_started": {
        "emoji":       "🚀",
        "title":       "Training Started",
        "description": "A new model training run has begun.",
        "color":       0x3498DB,  # blue
    },
    "training_epoch": {
        "emoji":       "[TRAIN]",
        "title":       "Training Epoch Metrics",
        "description": "Full end-of-epoch training metrics.",
        "color":       0x5865F2,
    },
    "training_completed": {
        "emoji":       "🏁",
        "title":       "Training Completed",
        "description": "Model training fold loop finished.",
        "color":       0x3498DB,  # blue
    },
    "fold_selected": {
        "emoji":       "🎯",
        "title":       "Best Fold Selected",
        "description": "Cross-validation best fold selected for evaluation.",
        "color":       0x57F287,
    },
    "promotion_gate_passed": {
        "emoji":       "🔓",
        "title":       "Promotion Gate Passed",
        "description": "Candidate model passed all policy checks.",
        "color":       0x00CC44,  # green
    },
    "promotion_gate_failed": {
        "emoji":       "🚧",
        "title":       "Promotion Gate Failed",
        "description": "Candidate model did not pass policy checks.",
        "color":       0xFFFF00,  # yellow
    },
    "training_warning": {
        "emoji":       "⚠️",
        "title":       "Training Warning",
        "description": "A background training job encountered an anomaly.",
        "color":       0xFFFF00,  # yellow
    },
    "backtest_result": {
        "emoji":       "[BACKTEST]",
        "title":       "Backtest Result",
        "description": "Full backtest result summary.",
        "color":       0x57F287,
    },
}

# Alert types that always bypass the rate limiter — never suppress these
CRITICAL_ALERT_TYPES = {
    "circuit_breaker",
    "production_deploy_failed",
    "promotion_gate_failed",
    "model_demoted",
}


# ── alerter class ─────────────────────────────────────────────────────────────

class DiscordAlerter:
    """
    Discord webhook alerter with rate limiting and fallback to print().

    Parameters
    ----------
    webhook_url    : Discord webhook URL. Falls back to DISCORD_WEBHOOK_URL env var.
    min_interval_s : Minimum seconds between two identical alert types
                     (prevents spam on rapid re-triggers).
    environment    : Tag shown in footers ("production", "staging", etc.).
    verbose        : If True, always prints alerts to stdout as well.
    """

    def __init__(
        self,
        webhook_url:     str | None = None,
        min_interval_s:  float = 300.0,  # 5-minute cooldown per alert type
        environment:     str   = "production",
        verbose:         bool  = True,
        user_id:         str | None = None,
    ):
        self._url         = webhook_url or WEBHOOK_URL
        self._user_id     = user_id or USER_ID
        self._min_ivl     = min_interval_s
        self._env         = environment
        self._verbose     = verbose
        self._last_sent:  dict[str, float] = {}   # alert_type → timestamp

        if not self._url:
            print("[Discord] No webhook URL set — alerts will only print to console. "
                  "Set DISCORD_WEBHOOK_URL env var to enable Discord delivery.")

    # ── public API ──────────────────────────────────────────────────────────

    def send(
        self,
        alert_type: str,
        fields:     dict[str, str] | None = None,
        force:      bool = False,
        ping_user:  bool = False,
        image_path: str | None = None,
        rate_key:   str | None = None,
    ) -> bool:
        """
        Send a Discord alert embed.

        Parameters
        ----------
        alert_type : One of the 6 alert types (see ALERT_CONFIG).
        fields     : Dict of field_name → value pairs shown in the embed.
        force      : If True, bypass rate-limit cooldown.
        ping_user  : If True and DISCORD_USER_ID is set, ping the user.
        image_path : Path to an image to attach to the embed.

        Returns True if the message was sent (or printed), False if throttled.
        """
        if alert_type not in ALERT_CONFIG:
            print(f"[Discord] Unknown alert type: {alert_type}")
            return False

        # Rate limiting — critical alerts always bypass
        rk = rate_key or alert_type
        if not force and alert_type not in CRITICAL_ALERT_TYPES:
            last = self._last_sent.get(rk, 0.0)
            if time.time() - last < self._min_ivl:
                return False

        cfg       = ALERT_CONFIG[alert_type]
        timestamp = datetime.now(UTC).isoformat()

        # Add content ping if requested
        content = f"<@{self._user_id}>" if ping_user and self._user_id else ""

        embed     = self._build_embed(cfg, fields or {}, timestamp)

        if image_path and os.path.exists(image_path) and REQUESTS_AVAILABLE:
            embed["embeds"][0]["image"] = {"url": f"attachment://{os.path.basename(image_path)}"}

        payload = embed
        if content:
            payload["content"] = content

        self._print_alert(cfg, fields or {}, timestamp)
        self._post_webhook(payload, image_path)
        self._last_sent[rk] = time.time()
        return True

    def send_circuit_breaker(self, drawdown: float, equity: float,
                              action: str, pair: str = "EURUSD"):
        self.send("circuit_breaker", {
            "Pair":     pair,
            "Drawdown": f"{drawdown:.2%}",
            "Equity":   f"${equity:,.2f}",
            "Action":   action.upper(),
        })

    def send_drift(self, psi_max: float, reasons: list):
        self.send("drift_detected", {
            "PSI Max":  f"{psi_max:.4f}",
            "Reasons":  ", ".join(str(r) for r in reasons[:3]),
        })

    def send_retrain_started(self, triggers: list, model: str = "unknown"):
        self.send("retrain_started", {
            "Model":    model,
            "Triggers": "\n".join(triggers[:3]),
        })

    def send_retrain(self, triggers: list, model: str = "unknown"):
        """Backward-compatible alias for older call sites."""
        self.send_retrain_started(triggers=triggers, model=model)

    def send_training_started(self, model: str, run_name: str = "unknown", pairs: list = None, data_window: str = "unknown"):
        from pathlib import Path
        base = Path('.').absolute().as_posix()
        self.send("training_started", {
            "Model": model,
            "Run Name": run_name,
            "Pairs": ", ".join(pairs) if pairs else "EURUSD",
            "Data Window": data_window,
            "Log": f"file:///{base}/logs/{run_name}_{model}_cv.json"
        }, rate_key=f"training_started_{model}_{run_name}")

    def send_production_deploy_completed(self, model: str, onnx_path: str = "", schema_path: str = "", fields: dict = None):
        from pathlib import Path
        f_dict = {
            "Model":  model,
            "ONNX":   f"file:///{Path(onnx_path).absolute().as_posix()}" if onnx_path else "—",
            "Schema": f"file:///{Path(schema_path).absolute().as_posix()}" if schema_path else "—",
        }
        if fields:
            f_dict.update(fields)
        self.send("production_deploy_completed", f_dict)

    def send_promotion(self, model: str, sharpe: float, git_hash: str = "", fields: dict = None):
        """Backward-compatible alias for older promotion call sites."""
        f_dict = {
            "Sharpe": f"{sharpe:.3f}",
            "Git": git_hash or "-",
        }
        if fields:
            f_dict.update(fields)
        self.send_production_deploy_completed(model=model, fields=f_dict)

    def send_fold_selected(self, model: str, fold: int, metrics: dict):
        from pathlib import Path
        base = Path('.').absolute().as_posix()
        self.send("fold_selected", {
            "Model": model,
            "Fold": str(fold),
            "Sharpe": f"{metrics.get('sharpe', 0):.4f}",
            "Loss": f"{metrics.get('val_loss', 0):.4f}",
            "Fold JSON": f"file:///{base}/checkpoints/{model}/fold_selection.json"
        }, rate_key=f"fold_selected_{model}_{fold}")

    def send_training_completed(self, model: str, fold: int, metric: str, score: float):
        self.send("training_completed", {
            "Model": model,
            "Best Fold": str(fold),
            "Score": f"{score:.4f}",
            f"Best {metric.capitalize()}": f"{score:.4f}",
        })

    def send_promotion_gate_passed(self, model: str, sharpe: float, details: dict = None):
        self.send("promotion_gate_passed", {
            "Model": model,
            "Sharpe": f"{sharpe:.4f}",
        })

    def send_gate_failed(self, model: str, reasons: list, profit_factor: float = 0.0, psr: float = 0.0):
        self.send("promotion_gate_failed", {
            "Model": model,
            "Reason": "\n".join(reasons[:3]),
            "Profit Factor": f"{profit_factor:.2f}",
            "PSR": f"{psr:.1f}%",
        })

    # Alias used by new code paths
    def send_promotion_gate_failed(self, model: str, reasons: list, profit_factor: float = 0.0, psr: float = 0.0):
        self.send_gate_failed(model=model, reasons=reasons, profit_factor=profit_factor, psr=psr)

    def send_production_deploy_failed(self, model: str, error_msg: str):
        self.send("production_deploy_failed", {
            "Model":  model,
            "Error":  str(error_msg)[:200],
        }, force=True)

    def send_demotion(self, triggers: list, rolled_back: bool):
        self.send("model_demoted", {
            "Triggers":  "\n".join(triggers[:3]),
            "Rollback":  "✓ Previous model restored" if rolled_back else "✗ No backup found",
            "Retrain":   "DAG triggered",
        })

    def send_tca_breach(self, cost_pct: float, limit_pct: float,
                         gross_pnl: float):
        self.send("tca_breach", {
            "Cost %":     f"{cost_pct:.1%}",
            "Policy Limit": f"{limit_pct:.1%}",
            "Gross P&L":  f"${gross_pnl:,.0f}",
        })

    # ── internal ────────────────────────────────────────────────────────────

    def _build_embed(self, cfg: dict, fields: dict[str, str],
                     timestamp: str) -> dict:
        embed_fields = [
            {"name": k, "value": v, "inline": True}
            for k, v in fields.items()
        ]
        embed_fields.append({"name": "Environment", "value": self._env, "inline": True})
        return {
            "embeds": [{
                "title":       f"{cfg['emoji']}  {cfg['title']}",
                "description": cfg["description"],
                "color":       cfg["color"],
                "fields":      embed_fields,
                "footer":      {"text": f"Forex Scaling Model  •  {timestamp}"},
                "timestamp":   timestamp,
            }]
        }

    def _print_alert(self, cfg: dict, fields: dict, timestamp: str) -> bool:
        if not self._verbose:
            return True
        try:
            sep = "─" * 55
            print(f"\n{sep}")
            print(f"  {cfg['emoji']}  {cfg['title']}")
            print(f"  {cfg['description']}")
            if fields:
                for k, v in fields.items():
                    print(f"    {k}: {v}")
            print(f"  {timestamp[:19]} UTC")
            print(f"{sep}")
        except UnicodeEncodeError:
            # Fallback for Windows cp1252 consoles that crash on emojis/lines
            sep = "-" * 55
            print(f"\n{sep}")
            print(f"  [ALERT] {cfg['title']}")
            print(f"  {cfg['description']}")
            if fields:
                for k, v in fields.items():
                    try:
                        print(f"    {k}: {v}")
                    except UnicodeEncodeError:
                        pass
            print(f"  {timestamp[:19]} UTC")
            print(f"{sep}")
        return True

    def _post_webhook(self, payload: dict, image_path: str | None = None) -> bool:
        if not self._url:
            return False

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        }

        try:
            if REQUESTS_AVAILABLE:
                if image_path and os.path.exists(image_path):
                    with open(image_path, "rb") as f:
                        files = {"file": (os.path.basename(image_path), f, "image/png")}
                        res = requests.post(
                            self._url,
                            data={"payload_json": json.dumps(payload)},
                            files=files,
                            headers=headers,
                            timeout=10
                        )
                else:
                    headers["Content-Type"] = "application/json"
                    res = requests.post(self._url, json=payload, headers=headers, timeout=10)
                res.raise_for_status()
                return True
            else:
                # Fallback to urllib if requests is somehow not installed (no image support)
                if image_path:
                    print("[Discord] urllib fallback does not support image attachments; sending embed only.")
                headers["Content-Type"] = "application/json"
                body = json.dumps(payload).encode("utf-8")
                headers["Content-Length"] = str(len(body))
                req  = urllib.request.Request(self._url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=10):
                    pass
                return True

        except Exception as e:
            if not REQUESTS_AVAILABLE and isinstance(e, urllib.error.HTTPError): # type: ignore
                body_snippet = e.read().decode(errors="replace")[:200]
                print(f"[Discord] HTTP {e.code}: {body_snippet}")
            else:
                print(f"[Discord] Send failed: {e}")
        return False


# ── smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("DiscordAlerter — smoke test (no webhook required)")
    alerter = DiscordAlerter(verbose=True, min_interval_s=0)

    alerter.send_circuit_breaker(0.105, 8_950.0, "close_all")
    alerter.send_drift(0.312, ["PSI > 0.2 on vol_20", "KS p < 0.05 on rsi_14"])
    alerter.send_promotion("haelt_v5", 1.82, "a3f9c1d")
    alerter.send_demotion(
        ["Sharpe 0.42 < floor 0.50", "WinRate 42.1% < floor 45%"],
        rolled_back=True,
    )
    alerter.send_tca_breach(0.36, 0.30, 12_500.0)
    print("\nOK [SUCCESS]")
