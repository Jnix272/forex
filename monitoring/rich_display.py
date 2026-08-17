"""
monitoring/rich_display.py
===========================
Real-time rich terminal dashboard for GPU training.

Replaces the plain print() epoch table with a live-updating panel that shows:
  • Epoch progress bar with ETA and elapsed time
  • Per-epoch metrics table (all completed epochs; best row highlighted green)
  • GPU stats sidebar (temperature, memory, OOM skips, NaN skips)
  • Early-stop countdown
  • Per-batch progress bar with loss, GPU temp, OOM/NaN counts in postfix

Falls back gracefully to the existing plain-print table when `rich` is not installed.

Usage
-----
    from monitoring.rich_display import RichTrainingDisplay

    display = RichTrainingDisplay(
        model_name   = "haelt",
        total_epochs = 100,
        patience     = 10,
        metric_name  = "val_sharpe",
        higher_is_better = True,
    )

    with display:
        for ep in range(total_epochs):
            train_pbar = display.batch_progress("Train", n_batches=len(train_dl))
            # ... training loop, call train_pbar.update() each batch ...
            train_pbar.stop()

            val_pbar = display.batch_progress("Val", n_batches=len(val_dl))
            # ... val loop ...
            val_pbar.stop()

            display.end_epoch(ep, metrics={
                "train_loss": tl, "val_loss": vl, "dir_acc": da,
                "val_sharpe": v_sh, "lr": lr,
                "gpu_mb": gm, "gpu_temp_c": temp,
                "oom_skips": n_oom, "nan_skips": n_nan,
                "elapsed_s": el,
            }, is_best=improved, no_improve=no_improve)

    display.finish(best_epoch=22, best_metric=1.34)
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# RICH IMPORT - graceful fallback when not installed
# ─────────────────────────────────────────────────────────────────────────────

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table
    from rich.text import Text

    _RICH = True
except ImportError:
    _RICH = False


# ─────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────────────────

_C = {
    "header": "bold cyan",
    "best": "bold green",
    "warn": "bold yellow",
    "danger": "bold red",
    "dim": "dim white",
    "metric": "bright_white",
    "epoch": "bright_cyan",
    "improved": "green",
    "normal": "white",
}


# ─────────────────────────────────────────────────────────────────────────────
# BATCH PROGRESS WRAPPER (used inside train_epoch / validate_epoch)
# ─────────────────────────────────────────────────────────────────────────────


class _BatchBar:
    """
    Thin wrapper that updates the rich Progress bar for a single phase
    (train or val). Call .update(loss, oom, nan, temp) each batch.
    """

    def __init__(self, progress: Any, task_id: Any, phase: str):
        self._progress = progress
        self._task_id = task_id
        self._phase = phase
        self._oom = 0
        self._nan = 0

    def update(self, loss: float | None = None, oom_skips: int = 0, nan_skips: int = 0, gpu_temp: int = -1) -> None:
        self._oom = oom_skips
        self._nan = nan_skips
        desc_parts = []
        if loss is not None:
            desc_parts.append(f"loss={loss:.5f}")
        if oom_skips:
            desc_parts.append(f"[yellow]OOMx{oom_skips}[/]")
        if nan_skips:
            desc_parts.append(f"[red]NaNx{nan_skips}[/]")
        if gpu_temp > 0:
            colour = "red" if gpu_temp >= 80 else "yellow" if gpu_temp >= 70 else "green"
            desc_parts.append(f"[{colour}]{gpu_temp}°C[/]")
        if self._progress is not None:
            self._progress.update(
                self._task_id,
                advance=1,
                description=f"  {self._phase} {'  '.join(desc_parts)}",
            )

    def stop(self) -> None:
        if self._progress is not None:
            self._progress.stop_task(self._task_id)

    # Allow use as a no-op tqdm-like object in non-rich mode
    def set_postfix(self, **kw) -> None:
        pass

    def close(self) -> None:
        self.stop()


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK - plain print table (used when rich is not installed)
# ─────────────────────────────────────────────────────────────────────────────


class _PlainDisplay:
    """No-op display used when rich is not installed."""

    def __init__(self, model_name: str, total_epochs: int, patience: int, metric_name: str, higher_is_better: bool):
        self.model_name = model_name
        self.total_epochs = total_epochs
        self.patience = patience
        self.metric_name = metric_name

    def __enter__(self):
        print(f"\n{'-' * 96}")
        print(f"  Training: {self.model_name.upper()} | {self.total_epochs} epochs")
        print(f"{'-' * 96}")
        print(
            f"{'Epoch':>9} {'Train':>11} {'Val':>11} {'DirAcc':>8} "
            f"{self.metric_name[:10]:>10} {'LR':>10} {'Time':>8} "
            f"{'GPU_MB':>8} {'OOM':>4} {'NaN':>4} {'Best':>5}"
        )
        print("-" * 96)
        return self

    def __exit__(self, *_):
        pass

    def batch_progress(self, phase: str, n_batches: int) -> _BatchBar:
        return _BatchBar(None, None, phase)

    def end_epoch(self, epoch: int, metrics: dict[str, Any], is_best: bool = False, no_improve: int = 0) -> None:
        tl = metrics.get("train_loss", float("nan"))
        vl = metrics.get("val_loss", float("nan"))
        da = metrics.get("dir_acc", float("nan"))
        sh = metrics.get("val_sharpe", float("nan"))
        lr = metrics.get("lr", 0.0)
        el = metrics.get("elapsed_s", 0.0)
        gm = metrics.get("gpu_mb", 0)
        oom = metrics.get("oom_skips", 0)
        nan = metrics.get("nan_skips", 0)
        marker = "yes" if is_best else ""
        print(
            f"{epoch + 1:>4}/{self.total_epochs:<4} {tl:>11.6f} {vl:>11.6f} "
            f"{da:>8.4f} {sh:>10.4f} {lr:>10.2e} {el:>7.1f}s "
            f"{gm:>8.0f} {oom:>4} {nan:>4} {marker:>5}"
        )
        return

    def finish(self, best_epoch: int = 0, best_metric: float = 0.0) -> None:
        print(f"\n{'-' * 96}")
        print(f"  Training complete | best epoch {best_epoch + 1} | best metric {best_metric:.4f}")
        print("-" * 96)


# ─────────────────────────────────────────────────────────────────────────────
# RICH DISPLAY
# ─────────────────────────────────────────────────────────────────────────────


class _RichDisplay:
    """Live rich terminal dashboard."""

    def __init__(self, model_name: str, total_epochs: int, patience: int, metric_name: str, higher_is_better: bool):
        self.model_name = model_name
        self.total_epochs = total_epochs
        self.patience = patience
        self.metric_name = metric_name
        self.higher_is_better = higher_is_better

        self._console = Console()
        self._history: list[dict[str, Any]] = []
        self._best_ep = -1
        self._best_val = float("-inf") if higher_is_better else float("inf")
        self._no_improve = 0
        self._start_ts = time.monotonic()
        self._live: Live | None = None

        # Epoch-level progress bar (shown above the table)
        self._ep_progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=28),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self._console,
            transient=False,
        )
        self._ep_task = self._ep_progress.add_task(
            f"[bold cyan]{model_name.upper()}",
            total=total_epochs,
        )

        # Batch-level progress (transient, shown during each epoch)
        self._batch_progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(bar_width=32),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self._console,
            transient=True,
        )

    # ── context manager ──────────────────────────────────────────────────────

    def __enter__(self) -> _RichDisplay:
        self._console.rule(
            f"[bold cyan]Training  ·  {self.model_name.upper()}  ·  {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
            style="cyan",
        )
        self._live = Live(
            self._build_layout(),
            console=self._console,
            refresh_per_second=4,
            vertical_overflow="visible",
        )
        self._live.__enter__()
        return self

    def __exit__(self, *args) -> None:
        if self._live:
            self._live.__exit__(*args)

    # ── batch progress bar ───────────────────────────────────────────────────

    def batch_progress(self, phase: str, n_batches: int) -> _BatchBar:
        task_id = self._batch_progress.add_task(
            f"  {phase}",
            total=n_batches,
        )
        if self._live:
            self._live.update(self._build_layout())
        return _BatchBar(self._batch_progress, task_id, phase)

    # ── epoch end ────────────────────────────────────────────────────────────

    def end_epoch(self, epoch: int, metrics: dict[str, Any], is_best: bool = False, no_improve: int = 0) -> None:
        self._no_improve = no_improve
        if is_best:
            self._best_ep = epoch
            self._best_val = metrics.get(
                "val_sharpe" if self.higher_is_better else "val_loss",
                self._best_val,
            )

        self._history.append({"epoch": epoch + 1, "is_best": is_best, **metrics})
        self._ep_progress.update(self._ep_task, advance=1)

        if self._live:
            self._live.update(self._build_layout())

    # ── finish ───────────────────────────────────────────────────────────────

    def finish(self, best_epoch: int = 0, best_metric: float = 0.0) -> None:
        if self._live:
            self._live.update(self._build_layout())
        elapsed = time.monotonic() - self._start_ts
        mins, secs = int(elapsed // 60), int(elapsed % 60)
        self._console.rule(style="green")
        self._console.print(
            f"[bold green]  ✓ Training complete[/]  "
            f"model=[cyan]{self.model_name}[/]  "
            f"best_epoch=[cyan]{best_epoch + 1}[/]  "
            f"best_{self.metric_name}=[cyan]{best_metric:.4f}[/]  "
            f"duration=[cyan]{mins}m {secs}s[/]"
        )
        self._console.rule(style="green")

    # ── layout builder ───────────────────────────────────────────────────────

    def _build_layout(self) -> Any:
        return Panel(
            self._build_content(),
            title=f"[bold cyan]{self.model_name.upper()}[/]  [dim]epoch {len(self._history)}/{self.total_epochs}[/]",
            border_style="cyan",
            padding=(0, 1),
        )

    def _build_content(self) -> Any:
        from rich.columns import Columns

        left = self._build_metrics_table()
        right = self._build_gpu_panel()

        # Stack: epoch progress bar, then columns of table + stats
        from rich.console import Group

        return Group(
            self._ep_progress,
            self._batch_progress,
            Columns([left, right], equal=False, expand=True),
        )

    def _build_metrics_table(self) -> Table:
        t = Table(
            show_header=True,
            header_style=_C["header"],
            border_style="dim",
            expand=True,
            show_lines=False,
        )
        t.add_column("Ep", style=_C["epoch"], justify="right", width=5)
        t.add_column("TrainLoss", justify="right", width=11)
        t.add_column("ValLoss", justify="right", width=11)
        t.add_column("DirAcc", justify="right", width=8)
        t.add_column("Sharpe", justify="right", width=9)
        t.add_column("LR", justify="right", width=10)
        t.add_column("GPU MB", justify="right", width=8)
        t.add_column("Time", justify="right", width=7)
        t.add_column("★", justify="center", width=3)

        # Show last 20 epochs so the table stays readable
        for rec in self._history[-20:]:
            is_best = rec.get("is_best", False)
            style = _C["best"] if is_best else _C["normal"]
            oom = rec.get("oom_skips", 0)
            nan = rec.get("nan_skips", 0)
            warn = ""
            if oom:
                warn += f"[yellow]O{oom}[/]"
            if nan:
                warn += f"[red]N{nan}[/]"

            t.add_row(
                str(rec["epoch"]),
                f"{rec.get('train_loss', 0):.6f}",
                f"{rec.get('val_loss', 0):.6f}",
                f"{rec.get('dir_acc', 0):.4f}",
                f"{rec.get('val_sharpe', 0):.4f}",
                f"{rec.get('lr', 0):.2e}",
                str(int(rec.get("gpu_mb", 0))),
                f"{rec.get('elapsed_s', 0):.1f}s",
                "★" if is_best else warn,
                style=style,
            )
        return t

    def _build_gpu_panel(self) -> Panel:
        lines: list[Text] = []

        # Latest GPU stats
        if self._history:
            last = self._history[-1]
            temp = last.get("gpu_temp_c", -1)
            mem = last.get("gpu_mb", -1)
            oom = last.get("oom_skips", 0)
            nan = last.get("nan_skips", 0)

            if temp >= 0:
                colour = "red" if temp >= 83 else "yellow" if temp >= 70 else "green"
                lines.append(Text.assemble("GPU Temp : ", (f"{temp}°C", colour)))
            if mem >= 0:
                lines.append(Text(f"GPU Mem  : {mem} MB"))
            if oom:
                lines.append(Text.assemble("OOM Skips: ", (str(oom), "yellow")))
            if nan:
                lines.append(Text.assemble("NaN Skips: ", (str(nan), "red")))

        lines.append(Text(""))

        # Early-stop countdown
        remain = self.patience - self._no_improve
        if self._no_improve > 0:
            colour = "red" if remain <= 2 else "yellow" if remain <= 5 else "green"
            lines.append(
                Text.assemble(
                    "Early Stop: ",
                    (f"{remain} left", colour),
                    f" / {self.patience}",
                )
            )
        else:
            lines.append(Text(f"Early Stop: {self.patience} left / {self.patience}", style="green"))

        lines.append(Text(""))

        # Best so far
        if self._best_ep >= 0:
            lines.append(Text(f"Best Epoch: {self._best_ep + 1}", style="bold green"))
            lines.append(Text(f"Best {self.metric_name[:8]}: {self._best_val:.4f}", style="bold green"))

        # Elapsed
        elapsed = time.monotonic() - self._start_ts
        mins, secs = int(elapsed // 60), int(elapsed % 60)
        lines.append(Text(""))
        lines.append(Text(f"Elapsed   : {mins}m {secs}s", style="dim"))

        from rich.console import Group

        content = Group(*lines)
        return Panel(content, title="[dim]Stats[/]", border_style="dim", width=28, padding=(0, 1))


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC FACTORY - returns the appropriate display object
# ─────────────────────────────────────────────────────────────────────────────


class RichTrainingDisplay:
    """
    Public factory. Returns a rich live display if the `rich` package is
    available, otherwise a plain-print fallback - same API in both cases.

    Parameters
    ----------
    model_name       : Architecture name shown in the header.
    total_epochs     : Total number of training epochs.
    patience         : Early-stop patience (used for countdown display).
    metric_name      : Name of the primary tracked metric.
    higher_is_better : True for Sharpe/accuracy, False for loss.
    """

    def __new__(
        cls,
        model_name: str = "model",
        total_epochs: int = 100,
        patience: int = 10,
        metric_name: str = "val_sharpe",
        higher_is_better: bool = True,
    ):
        kwargs = {
            "model_name": model_name,
            "total_epochs": total_epochs,
            "patience": patience,
            "metric_name": metric_name,
            "higher_is_better": higher_is_better,
        }
        if _RICH:
            return _RichDisplay(**kwargs)
        return _PlainDisplay(**kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random

    print(f"rich available: {_RICH}")
    N_EPOCHS = 12
    PATIENCE = 5

    display = RichTrainingDisplay(
        model_name="haelt",
        total_epochs=N_EPOCHS,
        patience=PATIENCE,
        metric_name="val_sharpe",
        higher_is_better=True,
    )

    best_sharpe = float("-inf")
    no_improve = 0

    with display:
        for ep in range(N_EPOCHS):
            # Simulate batch loop
            n_batches = 40
            train_bar = display.batch_progress("Train", n_batches)
            oom_count = 0
            nan_count = 0
            for b in range(n_batches):
                time.sleep(0.01)
                oom = 1 if b == 7 and ep == 3 else 0
                nan = 1 if b == 15 and ep == 5 else 0
                oom_count += oom
                nan_count += nan
                train_bar.update(
                    loss=0.45 - ep * 0.02 + random.gauss(0, 0.01),
                    oom_skips=oom_count,
                    nan_skips=nan_count,
                    gpu_temp=65 + ep,
                )
            train_bar.stop()

            val_bar = display.batch_progress("Val", 10)
            for _ in range(10):
                time.sleep(0.01)
                val_bar.update(loss=0.38 - ep * 0.015 + random.gauss(0, 0.005))
            val_bar.stop()

            tl = 0.45 - ep * 0.02
            vl = 0.38 - ep * 0.015
            sh = 0.70 + ep * 0.05 + random.gauss(0, 0.03)
            is_best = sh > best_sharpe
            if is_best:
                best_sharpe = sh
                no_improve = 0
            else:
                no_improve += 1

            display.end_epoch(
                ep,
                {
                    "train_loss": tl,
                    "val_loss": vl,
                    "dir_acc": 0.55 + ep * 0.01,
                    "val_sharpe": sh,
                    "lr": 1e-4 * (0.9**ep),
                    "gpu_mb": 4800 + ep * 50,
                    "gpu_temp_c": 65 + ep,
                    "oom_skips": oom_count,
                    "nan_skips": nan_count,
                    "elapsed_s": 0.5 + ep * 0.1,
                },
                is_best=is_best,
                no_improve=no_improve,
            )

    display.finish(best_epoch=ep, best_metric=best_sharpe)
    print("\nOK ✓")
