"""
scripts/generate_parity_data.py
===============================
Generates deterministic test vectors (.bin) for verifying numerical parity
between Python PyTorch and C++ ONNX Runtime.
"""

from pathlib import Path
import numpy as np
import onnxruntime as ort

def generate_parity_data():
    out_dir = Path("tests/data")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Stacking Ensemble Meta-Learner (1, 120, 584)
    ens_path = "checkpoints/ensemble/ensemble_meta_best.onnx"
    if Path(ens_path).exists():
        np.random.seed(42)
        x = np.random.randn(1, 120, 584).astype(np.float32)
        sess = ort.InferenceSession(ens_path, providers=["CPUExecutionProvider"])
        y = sess.run(None, {"features": x})[0].astype(np.float32)

        x.tofile(out_dir / "ensemble_test_input.bin")
        y.tofile(out_dir / "ensemble_test_output.bin")
        print(f"Generated ensemble parity data: input {x.shape}, output {y.shape}")

    # 2. RL Ensemble Consensus Policy (1, 1, 591) -> (1, 591)
    rl_path = "checkpoints/ensemble/rl_ensemble_best.onnx"
    if Path(rl_path).exists():
        np.random.seed(42)
        x_rl = np.random.randn(1, 591).astype(np.float32)
        sess_rl = ort.InferenceSession(rl_path, providers=["CPUExecutionProvider"])
        y_rl = sess_rl.run(None, {"observation": x_rl})[0].astype(np.float32)

        x_rl.tofile(out_dir / "rl_test_input.bin")
        y_rl.tofile(out_dir / "rl_test_output.bin")
        print(f"Generated RL parity data: input {x_rl.shape}, output {y_rl.shape}")

if __name__ == "__main__":
    generate_parity_data()
