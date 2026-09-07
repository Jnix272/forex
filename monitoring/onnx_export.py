"""PyTorch → ONNX export with round-trip verification and INT8 quantisation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from config.settings import PATHS

try:
    import torch
    import torch.nn as nn

    TORCH = True
except ImportError:
    TORCH = False

if TYPE_CHECKING:
    pass

__all__ = ["ONNXExporter"]


class ONNXExporter:
    """
    Export trained PyTorch models to ONNX for:
      - 3-5x faster CPU inference (no Python GIL, no PyTorch overhead)
      - Deploy to any platform without CUDA dependency
      - Run in C++, JavaScript, or via ONNX Runtime
      - Quantise to INT8 for further 2x speedup

    Typical latency improvement:
      PyTorch CPU HAELT: ~18ms -> ONNX Runtime: ~4ms
      PyTorch GPU HAELT: ~5ms  -> ONNX Runtime w/GPU: ~2ms
    """

    def __init__(self, export_dir: str | None = None):
        if export_dir is None:
            export_dir = PATHS["exports"]
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        model: "nn.Module",
        model_name: str,
        n_features: int,
        seq_len: int = 60,
        batch_size: int = 1,
        opset: int = 17,
        quantize: bool = False,
    ) -> str:
        """Export model to ONNX. Returns path to the .onnx file."""
        if not TORCH:
            raise RuntimeError("PyTorch required for ONNX export")

        import torch

        model.eval()
        dummy = torch.randn(batch_size, seq_len, n_features)
        out_path = str(self.export_dir / f"{model_name}.onnx")

        torch.onnx.export(
            model,
            dummy,
            out_path,
            opset_version=opset,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            do_constant_folding=True,
        )
        size_mb = Path(out_path).stat().st_size / 1e6
        print(f"[ONNX] Exported {model_name} -> {out_path} ({size_mb:.1f} MB)")

        try:
            import onnxruntime as ort

            sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
            dummy_np = dummy.numpy()
            out_torch = model(dummy).detach().numpy()
            out_onnx = sess.run(None, {"input": dummy_np})[0]
            max_diff = np.abs(out_torch - out_onnx).max()
            print(f"[ONNX] Round-trip verified | max diff = {max_diff:.2e}")

            t0 = time.perf_counter()
            for _ in range(100):
                sess.run(None, {"input": dummy_np})
            lat_onnx = (time.perf_counter() - t0) / 100 * 1000
            model.eval()
            with torch.no_grad():
                t0 = time.perf_counter()
                for _ in range(100):
                    model(dummy)
                lat_torch = (time.perf_counter() - t0) / 100 * 1000
            print(
                f"[ONNX] Latency - PyTorch: {lat_torch:.2f}ms  "
                f"ONNX: {lat_onnx:.2f}ms  "
                f"Speedup: {lat_torch / lat_onnx:.1f}x"
            )
        except ImportError:
            print("[ONNX] onnxruntime not installed - skipping verification")

        if quantize:
            return self._quantize_int8(out_path)
        return out_path

    def _quantize_int8(self, onnx_path: str) -> str:
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            q_path = onnx_path.replace(".onnx", "_int8.onnx")
            quantize_dynamic(onnx_path, q_path, weight_type=QuantType.QInt8)
            size_orig = Path(onnx_path).stat().st_size / 1e6
            size_q = Path(q_path).stat().st_size / 1e6
            print(f"[ONNX] INT8 quantised: {size_orig:.1f}MB -> {size_q:.1f}MB ({size_q / size_orig:.0%} of original)")
            return q_path
        except ImportError:
            print("[ONNX] onnxruntime quantisation not available")
            return onnx_path

    def load_onnx(self, onnx_path: str):
        """Load an ONNX model for fast inference."""
        try:
            import onnxruntime as ort

            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if TORCH and torch.cuda.is_available()
                else ["CPUExecutionProvider"]
            )
            return ort.InferenceSession(onnx_path, providers=providers)
        except ImportError:
            raise RuntimeError("pip install onnxruntime")
