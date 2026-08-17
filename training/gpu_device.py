"""Device setup, thermal throttle, and training preflight helpers.\n\nSee docs/CONTINUE.md."""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn

from training.core import _GPU_CFG, _log_info, _log_warn

# -----------------------------------------------------------------------------
# THERMAL THROTTLE  (laptop-safe GPU temperature guard)
# -----------------------------------------------------------------------------


def _gpu_temp_celsius() -> int:
    """Return current GPU 0 temperature in ┬░C, or -1 if pynvml unavailable."""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return int(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
    except Exception:
        return -1


def _thermal_check(limit: int = 83, pause_secs: float = 2.0) -> None:
    """
    Pause training if GPU temperature exceeds limit.
    Called every N batches in the training loop.
    limit=0 disables the check (desktop with active cooling).
    """
    if limit <= 0:
        return
    temp = _gpu_temp_celsius()
    if temp < 0:
        return  # pynvml unavailable ΓÇö skip silently
    if temp >= limit:
        msg = f"[Thermal] GPU {temp}┬░C >= limit {limit}┬░C ΓÇö pausing {pause_secs}s"
        print(msg)
        _log_warn(msg)
        import time as _t

        _t.sleep(pause_secs)


# -----------------------------------------------------------------------------
# GPU SETUP
# -----------------------------------------------------------------------------


def resolve_amp_dtype(preference: str = "auto") -> torch.dtype:
    """Select AMP dtype from ``auto`` / ``bf16`` / ``fp16`` / ``fp32``.

    Ampere+ (compute capability ≥ 8.0) always gets BF16 under ``auto``: same
    dynamic range as FP32 and no GradScaler. Pre-Ampere falls back to FP16.
    Explicit ``bf16`` falls back to FP16 with a warning if unsupported.
    """
    pref = str(preference or "auto").strip().lower()
    if pref == "fp32":
        return torch.float32
    if pref == "fp16":
        return torch.float16

    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        torch.backends.cudnn.enabled = False
        return torch.float32

    cc_major = int(torch.cuda.get_device_capability(0)[0])
    bf16_ok = bool(torch.cuda.is_bf16_supported()) and cc_major >= 8

    if pref == "bf16":
        if bf16_ok:
            return torch.bfloat16
        print(f"[GPU] BF16 requested but unsupported (CC {cc_major}.x) - falling back to FP16")
        return torch.float16

    # auto: force BF16 on all Ampere+ GPUs
    if bf16_ok:
        return torch.bfloat16
    return torch.float16


def setup_device(dtype_override: str = "auto", deterministic: bool = False) -> tuple[torch.device, int, torch.dtype]:
    """
    Detect GPU, configure cuDNN/TF32 flags, and select the optimal AMP dtype.

    Returns
    -------
    (device, n_gpus, amp_dtype)

    AMP dtype selection (override with --dtype or GPU["amp_dtype"] in settings):
    - BF16 - Ampere+ (CC >= 8.0): full FP32 range, no GradScaler, preferred.
    - FP16 - pre-Ampere: needs GradScaler to prevent underflow.
    - FP32 - CPU fallback or --dtype fp32 for debugging.

    Tensor Cores activate automatically when:
      1. Mixed precision is enabled (FP16 or BF16 via autocast).
      2. Standard layers are used (nn.Linear, nn.Conv*).
      3. Feature/hidden dims are multiples of 8 (all arch defaults satisfy this).
    """
    if not torch.cuda.is_available():
        print("[GPU] No CUDA detected - running on CPU (very slow for 20M ticks)")
        _log_warn("[GPU] CUDA unavailable - CPU mode")
        return torch.device("cpu"), 1, torch.float32

    n = torch.cuda.device_count()
    dev = torch.device("cuda:0")

    # -- Linux-specific: set multiprocessing start method to fork ---------------
    # fork() is the Linux default but setting it explicitly prevents edge cases
    # where PyTorch internally triggers spawn (e.g. nested multiprocessing calls).
    # Must be set before any DataLoader workers are created.
    if os.name != "nt":
        try:
            import torch.multiprocessing as _tmp

            _tmp.set_start_method("fork", force=True)
        except RuntimeError:
            pass  # already set elsewhere - fine

    # -- Linux-specific: pin CPU threads so DataLoader workers don't fight ------
    # Without this, 4 workers each try to use all CPU cores via OpenMP/MKL,
    # causing cache thrashing. One thread per worker is optimal for I/O-bound
    # zarr decompression (Blosc already uses internal multi-threading per chunk).
    if os.name != "nt":
        _n_cpu = os.cpu_count() or 4
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", str(min(4, _n_cpu)))

    # -- cuDNN / TF32 flags ----------------------------------------------------
    allow_tf32 = bool(_GPU_CFG.get("allow_tf32", True))
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = bool(_GPU_CFG.get("cudnn_benchmark", True))
        torch.backends.cudnn.deterministic = False  # max speed
    # Unlock TF32 Tensor Cores for all FP32 matmuls (~3x faster on Ada/Ampere).
    # 'high' = TF32 matmul + BF16 AMP - recommended for RTX 40-series.
    torch.set_float32_matmul_precision("high")

    # -- CUDA memory: use 95% of VRAM, leave 5% for driver/display overhead ----
    # PyTorch default reserves ~20% for its caching allocator which causes
    # premature OOM on 8 GB cards. 0.95 gives ~8.5 GB usable on RTX 4060.
    try:
        torch.cuda.set_per_process_memory_fraction(0.95, dev)
    except Exception:
        pass  # older PyTorch versions don't support this

    # -- AMP dtype selection ---------------------------------------------------
    cfg_dtype = _GPU_CFG.get("amp_dtype", "auto")
    effective_override = dtype_override if dtype_override != "auto" else cfg_dtype
    amp_dtype = resolve_amp_dtype(str(effective_override))

    dtype_name = {torch.bfloat16: "BF16", torch.float16: "FP16", torch.float32: "FP32"}.get(amp_dtype, "?")
    cc = "N/A"
    for i in range(n):
        g = torch.cuda.get_device_properties(i)
        vram = g.total_memory / 1e9
        cc = f"{g.major}.{g.minor}"
        print(f"[GPU {i}] {g.name} | {vram:.0f} GB VRAM | CC {cc} | CUDA {torch.version.cuda}")
        print(
            f"         AMP dtype: {dtype_name} | TF32: {allow_tf32} | cuDNN benchmark: {torch.backends.cudnn.benchmark}"
        )
        if vram < 12:
            print(
                f"         NOTE: low VRAM ({vram:.0f} GB). Use --hardware-profile "
                "rtx_4060_16gb_ram or ubuntu_rtx_laptop and --batch-size 384–512."  # noqa: RUF001
            )
    _log_info(f"[GPU] device={dev} n_gpus={n} amp_dtype={dtype_name} CC={cc}")
    return dev, n, amp_dtype


_RNN_MODULE_TYPES = (nn.LSTM, nn.GRU, nn.RNN)


def _compiler_disable():
    """Return ``torch.compiler.disable`` (or Dynamo fallback), else ``None``."""
    disable = getattr(getattr(torch, "compiler", None), "disable", None)
    if disable is not None:
        return disable
    try:
        import torch._dynamo as dynamo

        return dynamo.disable
    except Exception:
        return None


def disable_compile_on_rnn_modules(model: nn.Module) -> int:
    """Keep LSTM/GRU/RNN cells eager under ``torch.compile``.

    Recurrent cells have dynamic internal shapes that break CUDA-graph modes
    and can NaN when compiled. Marking their ``forward`` with
    ``torch.compiler.disable`` graph-breaks only those layers so the rest of
    the model can still use inductor (including ``reduce-overhead``).
    """
    disable = _compiler_disable()
    if disable is None:
        return 0
    n = 0
    for mod in model.modules():
        if not isinstance(mod, _RNN_MODULE_TYPES):
            continue
        fwd = mod.forward
        if getattr(fwd, "_forex_compiler_disabled", False):
            continue
        wrapped = disable(fwd)
        try:
            wrapped._forex_compiler_disabled = True  # type: ignore[attr-defined]
        except Exception:
            pass
        mod.forward = wrapped  # type: ignore[method-assign]
        n += 1
    return n


def maybe_torch_compile(model: nn.Module, device: torch.device, gpu_cfg: dict | None = None):
    """Compile ``model`` when CUDA + Triton + ``GPU.torch_compile`` allow it.

    Defaults to enabled (``torch_compile=True``). RNN modules are left eager
    via :func:`disable_compile_on_rnn_modules` rather than skipping compile for
    the whole network.
    """
    cfg = gpu_cfg if gpu_cfg is not None else (_GPU_CFG or {})
    if device.type != "cuda" or not hasattr(torch, "compile"):
        return model
    if not bool(cfg.get("torch_compile", True)):
        print("[Model] torch.compile disabled via GPU.torch_compile=false")
        return model

    triton_ok = False
    try:
        import triton  # noqa: F401

        triton_ok = True
    except ImportError:
        pass
    if not triton_ok:
        print("[Model] torch.compile skipped (Triton not available - running eager mode)")
        _log_info("[Model] torch.compile skipped - eager mode")
        return model

    n_rnn = disable_compile_on_rnn_modules(model)
    if n_rnn:
        print(f"[Model] Left {n_rnn} LSTM/GRU/RNN module(s) eager (torch.compiler.disable); compiling the rest")

    mode = str(cfg.get("torch_compile_mode", "reduce-overhead"))
    try:
        compiled = torch.compile(model, mode=mode)
        print(f"[Model] torch.compile ON (backend=inductor, mode={mode})")
        _log_info(f"[Model] torch.compile inductor mode={mode}")
        return compiled
    except Exception as exc:
        print(f"[Model] torch.compile skipped: {exc}")
        _log_warn(f"[Model] torch.compile skipped: {exc}")
        return model


def build_adamw(
    params,
    *,
    lr: float,
    weight_decay: float = 0.0,
    fused: bool | None = None,
    foreach: bool | None = None,
    **kwargs: Any,
) -> torch.optim.Optimizer:
    """AdamW preferring a fused CUDA kernel; falls back to eager AdamW.

    Preference order:
      1. ``torch.optim.AdamW(..., fused=True)`` (PyTorch ≥2.0, CUDA)
      2. ``apex.optimizers.FusedAdamW`` or ``FusedAdam(adam_w_mode=True)``
      3. Standard ``torch.optim.AdamW``

    ``fused=None`` enables fused mode when CUDA is available. Pass
    ``fused=False`` to force the eager path (CPU / unsupported param layouts).
    """
    param_list = list(params)
    want_fused = bool(torch.cuda.is_available()) if fused is None else bool(fused)

    def _log(msg: str, *, warn: bool = False) -> None:
        fn = globals().get("_log_warn" if warn else "_log_info")
        if callable(fn):
            try:
                fn(msg)
                return
            except Exception:
                pass
        print(msg)

    if want_fused:
        try:
            opt_kwargs = dict(kwargs)
            # fused and foreach are mutually exclusive in torch.optim.AdamW
            opt_kwargs["foreach"] = False if foreach is None else bool(foreach)
            opt = torch.optim.AdamW(
                param_list,
                lr=lr,
                weight_decay=weight_decay,
                fused=True,
                **opt_kwargs,
            )
            _log("[Optim] Using fused torch.optim.AdamW")
            return opt
        except (RuntimeError, TypeError, ValueError) as exc:
            _log(f"[Optim] fused torch.optim.AdamW unavailable ({exc})", warn=True)

        try:
            from apex.optimizers import FusedAdamW as _ApexFusedAdamW  # type: ignore

            _log("[Optim] Using apex.optimizers.FusedAdamW")
            return _ApexFusedAdamW(param_list, lr=lr, weight_decay=weight_decay, **kwargs)
        except Exception:
            pass
        try:
            from apex.optimizers import FusedAdam as _ApexFusedAdam  # type: ignore

            _log("[Optim] Using apex.optimizers.FusedAdam (adam_w_mode=True)")
            return _ApexFusedAdam(
                param_list,
                lr=lr,
                weight_decay=weight_decay,
                adam_w_mode=True,
                **kwargs,
            )
        except Exception:
            pass
        _log("[Optim] Fused AdamW unavailable - falling back to eager AdamW", warn=True)

    opt_kwargs = dict(kwargs)
    if foreach is not None:
        opt_kwargs["foreach"] = bool(foreach)
    return torch.optim.AdamW(param_list, lr=lr, weight_decay=weight_decay, **opt_kwargs)
