import sys
from pathlib import Path
import tempfile
import torch
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.architectures import MambaScalper
from inference.pytorch_inference import PyTorchInferenceEngine
from inference.onnx_inference import DirectMLInferenceEngine, core_onnx_export
from inference.rl_inference import RLInferenceAgent

pytest.importorskip("onnx")
pytest.importorskip("onnxscript")
pytest.importorskip("onnxruntime")


def test_inference_consistency():
    with tempfile.TemporaryDirectory() as td:
        ckpt_dir = Path(td)
        n_features = 32
        seq_len = 60
        model_name = "mamba"
        
        # 1. Create a dummy model and save its checkpoint
        model = MambaScalper(input_size=n_features, d_model=128, num_layers=2, num_classes=3)
        model.eval()
        ckpt_path = ckpt_dir / f"{model_name}_best.pt"
        torch.save(model.state_dict(), ckpt_path)
        
        # Create dummy config
        import json
        cfg_path = ckpt_dir / f"{model_name}_config.json"
        cfg_path.write_text(json.dumps({
            "model": "mamba",
            "n_features": n_features,
            "seq_len": seq_len,
            "d_model": 128,
            "num_layers": 2,
            "num_classes": 3
        }))
        
        # 2. PyTorch Engine
        pt_engine = PyTorchInferenceEngine(
            checkpoint_path=str(ckpt_path),
            model_name=model_name,
            seq_len=seq_len,
            n_features=n_features,
            device="cpu"
        )
        assert pt_engine.seq_len == seq_len
        assert pt_engine.n_features == n_features
        
        # 3. ONNX Engine
        onnx_path = ckpt_dir / f"{model_name}_best.onnx"
        core_onnx_export(model, n_features, seq_len, str(onnx_path))
        
        onnx_engine = DirectMLInferenceEngine(
            onnx_path=str(onnx_path),
            seq_len=seq_len,
            prefer_cpu=True
        )
        assert onnx_engine.seq_len == seq_len
        
        # Check action interfaces
        dummy_obs = np.random.randn(n_features).astype(np.float32)
        
        # Both should accept the identical 1D array
        pt_act = pt_engine.select_action(dummy_obs)
        assert pt_act in [0, 1, 2]
        
        onnx_act = onnx_engine.select_action(dummy_obs)
        assert onnx_act in [0, 1, 2]
        
        # 4. RL Engine
        # Create dummy RL checkpoint
        rl_path = ckpt_dir / "rl_dqn_best.pt"
        from models.rl_agents import DQNAgent
        
        # obs_size = encoder out + 5 state vars
        obs_size = model(torch.randn(1, seq_len, n_features)).shape[-1] + 5
        agent = DQNAgent(obs_size=obs_size, device="cpu")
        torch.save(agent.policy_net.state_dict(), rl_path)
        
        rl_engine = RLInferenceAgent(
            rl_checkpoint=str(rl_path),
            supervised_checkpoint=str(ckpt_path),
            model_name=model_name,
            seq_len=seq_len,
            n_features=n_features,
            device="cpu"
        )
        assert rl_engine.seq_len == seq_len
        assert rl_engine.n_features == n_features
        
        rl_act = rl_engine.select_action(dummy_obs)
        assert rl_act in range(10)
        
        print("Consistency check passed for PyTorch, ONNX, and RL engines.")
        print(f"Validated seq_len={seq_len}, n_features={n_features}")

if __name__ == "__main__":
    test_inference_consistency()
