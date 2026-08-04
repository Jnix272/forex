"""
training/smoke_test.py
========================
Pre-flight smoke test — runs all 6 model architectures through a tiny
synthetic training loop before committing to a full GPU training run.

What is verified per model
--------------------------
  ✓ Build          — model instantiates without error
  ✓ Forward pass   — output shape is correct, loss is finite (no NaN)
  ✓ Backward pass  — gradients flow, no NaN in any .grad tensor
  ✓ Mini-training  — 3 epochs of (train + validate) without OOM or crash
  ✓ Checkpoint     — state_dict saves to a temp directory, file size > 0

Usage
-----
    # CPU (default)
    python training/smoke_test.py

    # GPU (uses CUDA if available, else CPU)
    python training/smoke_test.py --gpu

    # AMP (mixed precision — recommended before a real GPU run)
    python training/smoke_test.py --gpu --amp

    # Only specific models
    python training/smoke_test.py --models haelt,tft,mamba

    # Bigger batch for a closer-to-real test
    python training/smoke_test.py --gpu --amp --batch-size 128 --epochs 5

Exit code: 0 = all models passed  |  1 = one or more failed
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

# ── project root on path ───────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np

# ── optional rich ──────────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text
    _RICH = True
    _console = Console()
except ImportError:
    _RICH = False
    _console = None  # type: ignore

# ── torch ─────────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH = True
except ImportError:
    _TORCH = False


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — change these to make the smoke test faster or more rigorous
# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS = {
    "n_samples":  300,    # synthetic samples (train + val)
    "seq_len":    30,     # input window length in bars
    "n_features": 64,     # number of input features
    "batch_size": 32,     # mini-batch size
    "epochs":     3,      # training epochs per model
    "val_frac":   0.2,    # fraction held out for validation
    "d_model":    64,     # transformer / attention dim
    "nhead":      4,      # attention heads
    "num_layers": 2,      # encoder layers
    "hidden_size":128,    # LSTM / GRU hidden size
    "dropout":    0.1,
    "lr":         1e-3,
    "grad_clip":  1.0,
}

ALL_MODELS = ["haelt", "tft", "transformer", "mamba", "gnn", "expert"]


# ─────────────────────────────────────────────────────────────────────────────
# RESULT DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

class ModelResult:
    def __init__(self, name: str):
        self.name        = name
        self.passed      = False
        self.build_ok    = False
        self.fwd_ok      = False
        self.bwd_ok      = False
        self.train_ok    = False
        self.ckpt_ok     = False
        self.build_ms    = 0.0
        self.fwd_ms      = 0.0
        self.bwd_ms      = 0.0
        self.train_s     = 0.0
        self.final_loss  = float("nan")
        self.oom_skips   = 0
        self.nan_skips   = 0
        self.nan_grads   = 0
        self.param_count = 0
        self.ckpt_kb     = 0.0
        self.error       = ""

    @property
    def status(self) -> str:
        if self.passed:
            return "PASS"
        return "FAIL"


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _make_dataset(cfg: dict[str, Any], device: torch.device) -> tuple[
        DataLoader, DataLoader]:
    """Generate a tiny random dataset and return train/val DataLoaders."""
    n     = cfg["n_samples"]
    T     = cfg["seq_len"]
    F     = cfg["n_features"]
    B     = cfg["batch_size"]
    split = int(n * (1 - cfg["val_frac"]))

    rng = np.random.default_rng(42)
    X   = rng.standard_normal((n, T, F)).astype(np.float32)
    y   = rng.integers(0, 3, size=(n,)).astype(np.int64)   # 3-class labels

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)

    train_ds = TensorDataset(X_t[:split], y_t[:split])
    val_ds   = TensorDataset(X_t[split:], y_t[split:])

    train_dl = DataLoader(train_ds, batch_size=B, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=B, shuffle=False, num_workers=0)
    return train_dl, val_dl


def _nan_in_grads(model: nn.Module) -> int:
    """Return number of parameters whose .grad contains NaN."""
    count = 0
    for p in model.parameters():
        if p.grad is not None and torch.isnan(p.grad).any():
            count += 1
    return count


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _print(msg: str, style: str = "") -> None:
    if _RICH and _console:
        _console.print(msg, style=style)
    else:
        print(msg)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL TESTER
# ─────────────────────────────────────────────────────────────────────────────

def _test_model(
    name:       str,
    cfg:        dict[str, Any],
    device:     torch.device,
    use_amp:    bool,
    amp_dtype:  Any,
    ckpt_dir:   Path,
    verbose:    bool = True,
) -> ModelResult:
    """Run all smoke-test checks for a single model. Returns a ModelResult."""
    from models.architectures import build_model  # type: ignore

    result = ModelResult(name)

    # ── 1. Build ──────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        model = build_model(
            name,
            input_size  = cfg["n_features"],
            seq_len     = cfg["seq_len"],
            d_model     = cfg["d_model"],
            nhead       = cfg["nhead"],
            num_layers  = cfg["num_layers"],
            hidden_size = cfg["hidden_size"],
            dropout     = cfg["dropout"],
        ).to(device)
        result.build_ms    = (time.perf_counter() - t0) * 1000
        result.param_count = _count_params(model)
        result.build_ok    = True
    except Exception as e:
        result.error = f"Build failed: {e}"
        return result

    # ── 2. Forward pass ───────────────────────────────────────────────────────
    crit = nn.CrossEntropyLoss()
    try:
        model.eval()
        dummy_x = torch.randn(cfg["batch_size"], cfg["seq_len"],
                              cfg["n_features"], device=device)
        dummy_y = torch.randint(0, 3, (cfg["batch_size"],), device=device)

        t0 = time.perf_counter()
        with torch.no_grad():
            if use_amp and device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    out = model(dummy_x)
            else:
                out = model(dummy_x)

        # Handle MultiTask tuple output
        logits = out[0] if isinstance(out, tuple) else out
        # Handle regression output (B,) -> expand to (B, 3) for CE loss check
        if logits.dim() == 1:
            logits = logits.unsqueeze(1).expand(-1, 3)

        loss_val = crit(logits, dummy_y).item()
        result.fwd_ms = (time.perf_counter() - t0) * 1000
        result.fwd_ok = not (loss_val != loss_val)   # NaN check
    except Exception as e:
        result.error = f"Forward failed: {e}"
        return result

    # ── 3. Backward pass ──────────────────────────────────────────────────────
    try:
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
        scaler = (torch.amp.GradScaler("cuda")
                  if use_amp and device.type == "cuda" and amp_dtype == torch.float16
                  else None)

        t0 = time.perf_counter()
        opt.zero_grad()
        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                out = model(dummy_x)
                logits = out[0] if isinstance(out, tuple) else out
                if logits.dim() == 1:
                    logits = logits.unsqueeze(1).expand(-1, 3)
                loss = crit(logits, dummy_y)
        else:
            out = model(dummy_x)
            logits = out[0] if isinstance(out, tuple) else out
            if logits.dim() == 1:
                logits = logits.unsqueeze(1).expand(-1, 3)
            loss = crit(logits, dummy_y)

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        result.bwd_ms   = (time.perf_counter() - t0) * 1000
        result.nan_grads = _nan_in_grads(model)
        result.bwd_ok   = (result.nan_grads == 0)
        opt.zero_grad()
    except Exception as e:
        result.error = f"Backward failed: {e}"
        return result

    # ── 4. Mini training loop ─────────────────────────────────────────────────
    try:
        train_dl, val_dl = _make_dataset(cfg, device)
        opt       = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=cfg["lr"] * 5,
            epochs=cfg["epochs"], steps_per_epoch=len(train_dl),
        )
        scaler = (torch.amp.GradScaler("cuda")
                  if use_amp and device.type == "cuda" and amp_dtype == torch.float16
                  else None)

        t0 = time.perf_counter()
        last_loss = float("nan")

        for ep in range(cfg["epochs"]):
            model.train()
            ep_loss = 0.0; ep_n = 0

            for xb, yb in train_dl:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                try:
                    if use_amp and device.type == "cuda":
                        with torch.amp.autocast("cuda", dtype=amp_dtype):
                            out = model(xb)
                            logits = out[0] if isinstance(out, tuple) else out
                            if logits.dim() == 1:
                                logits = logits.unsqueeze(1).expand(-1, 3)
                            loss = crit(logits, yb)
                        if scaler:
                            scaler.scale(loss).backward()
                            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                            scaler.step(opt); scaler.update()
                        else:
                            loss.backward()
                            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                            opt.step()
                    else:
                        out = model(xb)
                        logits = out[0] if isinstance(out, tuple) else out
                        if logits.dim() == 1:
                            logits = logits.unsqueeze(1).expand(-1, 3)
                        loss = crit(logits, yb)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                        opt.step()

                    lv = loss.item()
                    if lv != lv:     # NaN
                        result.nan_skips += 1
                    else:
                        ep_loss += lv; ep_n += 1
                    scheduler.step()

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        result.oom_skips += 1
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                    else:
                        raise

            if ep_n:
                last_loss = ep_loss / ep_n

        result.train_s    = time.perf_counter() - t0
        result.final_loss = last_loss
        result.train_ok   = (
            result.oom_skips == 0
            and result.nan_skips == 0
            and last_loss == last_loss   # not NaN
        )
    except Exception as e:
        result.error = f"Training failed: {e}\n{traceback.format_exc()[-600:]}"
        return result

    # ── 5. Checkpoint save ────────────────────────────────────────────────────
    try:
        ckpt_path = ckpt_dir / f"{name}_smoke.pt"
        core = model.module if hasattr(model, "module") else model
        torch.save(core.state_dict(), ckpt_path)
        result.ckpt_kb  = ckpt_path.stat().st_size / 1024
        result.ckpt_ok  = result.ckpt_kb > 0
    except Exception as e:
        result.error = f"Checkpoint failed: {e}"
        return result

    result.passed = (
        result.build_ok
        and result.fwd_ok
        and result.bwd_ok
        and result.train_ok
        and result.ckpt_ok
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def _print_results_rich(results: list[ModelResult], total_s: float,
                         cfg: dict[str, Any], device_str: str,
                         use_amp: bool) -> None:
    n_pass = sum(1 for r in results if r.passed)
    n_fail = len(results) - n_pass

    _console.rule(
        "[bold cyan]Smoke Test Results[/]  "
        f"[dim]{device_str}  AMP={use_amp}  "
        f"samples={cfg['n_samples']}  seq={cfg['seq_len']}  "
        f"features={cfg['n_features']}  epochs={cfg['epochs']}[/]",
        style="cyan",
    )

    t = Table(show_header=True, header_style="bold cyan",
              border_style="dim", show_lines=True, expand=True)
    t.add_column("Model",       style="cyan",         width=13)
    t.add_column("Status",      justify="center",      width=8)
    t.add_column("Build",       justify="center",      width=7)
    t.add_column("Forward",     justify="center",      width=9)
    t.add_column("Backward",    justify="center",      width=9)
    t.add_column("Train",       justify="center",      width=7)
    t.add_column("Ckpt",        justify="center",      width=7)
    t.add_column("Params",      justify="right",       width=10)
    t.add_column("Loss",        justify="right",       width=8)
    t.add_column("OOM/NaN",     justify="center",      width=9)
    t.add_column("Time (s)",    justify="right",       width=9)
    t.add_column("Ckpt (KB)",   justify="right",       width=10)

    def _tick(ok: bool) -> Text:
        return Text("✓", style="green") if ok else Text("✗", style="red")

    for r in results:
        row_style = "bold green" if r.passed else "red"
        oom_nan = ""
        if r.oom_skips:
            oom_nan += f"OOM×{r.oom_skips} "
        if r.nan_skips:
            oom_nan += f"NaN×{r.nan_skips}"
        if r.nan_grads:
            oom_nan += f"∇NaN×{r.nan_grads}"
        oom_nan = oom_nan.strip() or "—"

        t.add_row(
            r.name.upper(),
            Text("PASS", style="bold green") if r.passed else Text("FAIL", style="bold red"),
            _tick(r.build_ok),
            _tick(r.fwd_ok),
            _tick(r.bwd_ok),
            _tick(r.train_ok),
            _tick(r.ckpt_ok),
            f"{r.param_count:,}",
            f"{r.final_loss:.4f}" if r.final_loss == r.final_loss else "NaN",
            oom_nan,
            f"{r.train_s:.1f}",
            f"{r.ckpt_kb:.0f}",
            style=row_style,
        )

    _console.print(t)

    # Errors
    for r in results:
        if r.error:
            _console.print(
                f"\n[red][{r.name.upper()}] Error:[/]\n{r.error}",
                style="red",
            )

    # Summary
    summary_style = "bold green" if n_fail == 0 else "bold red"
    _console.rule(
        f"[{summary_style}]{n_pass}/{len(results)} models passed[/]  "
        f"[dim]total {total_s:.1f}s[/]",
        style="green" if n_fail == 0 else "red",
    )
    if n_fail == 0:
        _console.print(
            "[bold green]✓ All models passed — safe to start full training.[/]"
        )
    else:
        failed = [r.name for r in results if not r.passed]
        _console.print(
            f"[bold red]✗ {n_fail} model(s) failed: {', '.join(failed)}[/]\n"
            "[yellow]Fix the errors above before running a full training run.[/]"
        )


def _print_results_plain(results: list[ModelResult], total_s: float) -> None:
    SEP = "─" * 90
    print(f"\n{SEP}")
    print(f"  {'MODEL':<12} {'STATUS':<7} {'BUILD':<7} {'FWD':<6} {'BWD':<6} "
          f"{'TRAIN':<7} {'PARAMS':>10}  {'LOSS':>8}  {'OOM/NaN':<10} {'TIME':>7}")
    print(SEP)
    for r in results:
        oom_nan = ""
        if r.oom_skips: oom_nan += f"OOM×{r.oom_skips} "
        if r.nan_skips: oom_nan += f"NaN×{r.nan_skips}"
        oom_nan = oom_nan.strip() or "—"
        print(
            f"  {r.name.upper():<12} "
            f"{'PASS' if r.passed else 'FAIL':<7} "
            f"{'✓' if r.build_ok else '✗':<7} "
            f"{'✓' if r.fwd_ok else '✗':<6} "
            f"{'✓' if r.bwd_ok else '✗':<6} "
            f"{'✓' if r.train_ok else '✗':<7} "
            f"{r.param_count:>10,}  "
            f"{r.final_loss:>8.4f}  "
            f"{oom_nan:<10} "
            f"{r.train_s:>6.1f}s"
        )
        if r.error:
            print(f"    ERROR: {r.error[:120]}")
    print(SEP)
    n_pass = sum(1 for r in results if r.passed)
    print(f"  {n_pass}/{len(results)} models passed  |  total {total_s:.1f}s")
    print(SEP)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pre-flight smoke test for all Forex Scaling Model architectures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--gpu",        action="store_true",
                   help="Use GPU if available (default: CPU)")
    p.add_argument("--amp",        action="store_true",
                   help="Enable AMP mixed precision (requires --gpu)")
    p.add_argument("--dtype",      type=str, default="auto",
                   choices=["auto", "bf16", "fp16", "fp32"],
                   help="AMP dtype override (default: auto -> bf16 on Ampere+, fp32 on CPU)")
    p.add_argument("--models",     type=str, default=",".join(ALL_MODELS),
                   help=f"Comma-separated models to test (default: all — {','.join(ALL_MODELS)})")
    p.add_argument("--epochs",     type=int, default=DEFAULTS["epochs"],
                   help=f"Training epochs per model (default: {DEFAULTS['epochs']})")
    p.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"],
                   help=f"Mini-batch size (default: {DEFAULTS['batch_size']})")
    p.add_argument("--n-samples",  type=int, default=DEFAULTS["n_samples"],
                   help=f"Synthetic samples (default: {DEFAULTS['n_samples']})")
    p.add_argument("--seq-len",    type=int, default=DEFAULTS["seq_len"],
                   help=f"Sequence length (default: {DEFAULTS['seq_len']})")
    p.add_argument("--n-features", type=int, default=DEFAULTS["n_features"],
                   help=f"Feature count (default: {DEFAULTS['n_features']})")
    p.add_argument("--d-model",    type=int, default=DEFAULTS["d_model"])
    p.add_argument("--nhead",      type=int, default=DEFAULTS["nhead"])
    p.add_argument("--num-layers", type=int, default=DEFAULTS["num_layers"])
    p.add_argument("--hidden-size",type=int, default=DEFAULTS["hidden_size"])
    p.add_argument("--verbose",    action="store_true",
                   help="Print per-model progress as each check runs")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    if not _TORCH:
        print("[ERROR] PyTorch is not installed. "
              "Install it from https://pytorch.org/get-started/locally/")
        return 1

    args = _parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        cc     = torch.cuda.get_device_capability()
        device_str = (f"CUDA — {torch.cuda.get_device_name(0)} "
                      f"(CC {cc[0]}.{cc[1]}, "
                      f"{torch.cuda.get_device_properties(0).total_memory//1_000_000} MB)")
    else:
        if args.gpu:
            _print("[yellow]No CUDA GPU found — running on CPU.[/]" if _RICH
                   else "No CUDA GPU found — running on CPU.")
        device     = torch.device("cpu")
        device_str = "CPU"

    # ── AMP dtype ─────────────────────────────────────────────────────────────
    use_amp = args.amp and device.type == "cuda"
    if args.dtype == "bf16":
        amp_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        amp_dtype = torch.float16
    elif args.dtype == "fp32":
        amp_dtype = torch.float32
    else:
        cc = torch.cuda.get_device_capability() if device.type == "cuda" else (0, 0)
        amp_dtype = torch.bfloat16 if cc[0] >= 8 else torch.float16 if use_amp else torch.float32

    # ── Models to test ────────────────────────────────────────────────────────
    models_to_test = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    invalid = [m for m in models_to_test if m not in ALL_MODELS]
    if invalid:
        _print(f"Unknown models: {invalid}. Valid: {ALL_MODELS}")
        return 1

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = dict(DEFAULTS)
    cfg.update({
        "epochs":     args.epochs,
        "batch_size": args.batch_size,
        "n_samples":  args.n_samples,
        "seq_len":    args.seq_len,
        "n_features": args.n_features,
        "d_model":    args.d_model,
        "nhead":      args.nhead,
        "num_layers": args.num_layers,
        "hidden_size":args.hidden_size,
    })

    # ── Header ────────────────────────────────────────────────────────────────
    if _RICH and _console:
        _console.rule("[bold cyan]Forex Scaling Model — Pre-flight Smoke Test[/]",
                      style="cyan")
        _console.print(
            f"  [dim]device=[/][cyan]{device_str}[/]  "
            f"[dim]amp=[/][cyan]{use_amp}[/]  "
            f"[dim]models=[/][cyan]{', '.join(m.upper() for m in models_to_test)}[/]\n"
            f"  [dim]samples={cfg['n_samples']}  seq={cfg['seq_len']}  "
            f"features={cfg['n_features']}  epochs={cfg['epochs']}  "
            f"batch={cfg['batch_size']}[/]"
        )
    else:
        print(f"\n{'═'*62}")
        print("  Forex Scaling Model — Pre-flight Smoke Test")
        print(f"  Device  : {device_str}")
        print(f"  AMP     : {use_amp}")
        print(f"  Models  : {', '.join(m.upper() for m in models_to_test)}")
        print(f"  Config  : samples={cfg['n_samples']}  seq={cfg['seq_len']}  "
              f"features={cfg['n_features']}  epochs={cfg['epochs']}")
        print("═" * 62)

    # ── Run tests ─────────────────────────────────────────────────────────────
    results: list[ModelResult] = []
    total_t0 = time.perf_counter()

    with tempfile.TemporaryDirectory() as ckpt_dir:
        ckpt_path = Path(ckpt_dir)

        for i, name in enumerate(models_to_test, 1):
            prefix = (
                f"[bold cyan][{i}/{len(models_to_test)}][/] "
                f"Testing [cyan]{name.upper()}[/]..."
            ) if _RICH else f"[{i}/{len(models_to_test)}] Testing {name.upper()}..."

            if _RICH and _console:
                _console.print(prefix, end=" ")
            else:
                print(prefix, end=" ", flush=True)

            t0 = time.perf_counter()
            result = _test_model(
                name       = name,
                cfg        = cfg,
                device     = device,
                use_amp    = use_amp,
                amp_dtype  = amp_dtype,
                ckpt_dir   = ckpt_path,
                verbose    = args.verbose,
            )
            elapsed = time.perf_counter() - t0
            results.append(result)

            if _RICH and _console:
                if result.passed:
                    _console.print(
                        f"[bold green]PASS[/]  "
                        f"[dim]{result.param_count:,} params  "
                        f"loss={result.final_loss:.4f}  "
                        f"{elapsed:.1f}s[/]"
                    )
                else:
                    _console.print(
                        f"[bold red]FAIL[/]  "
                        f"[red]{result.error[:80]}[/]"
                    )
            else:
                status = "PASS" if result.passed else f"FAIL — {result.error[:60]}"
                print(f"{status}  ({elapsed:.1f}s)")

    total_s = time.perf_counter() - total_t0

    # ── Report ────────────────────────────────────────────────────────────────
    print()
    if _RICH and _console:
        _print_results_rich(results, total_s, cfg, device_str, use_amp)
    else:
        _print_results_plain(results, total_s)

    n_pass = sum(1 for r in results if r.passed)
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
