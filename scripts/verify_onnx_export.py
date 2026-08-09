import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from inference.pytorch_inference import load_pytorch_model


def generate_test_data(checkpoint_path: str, model_name: str, batch_size: int = 1, seq_len: int = 60, n_features: int = 64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load PyTorch model
    model, out_features, out_seq, arch, _scaler = load_pytorch_model(
        checkpoint_path, model_name, seq_len=seq_len, n_features=n_features, device=device
    )
    model.eval()

    # Generate deterministic pseudo-random input
    torch.manual_seed(42)
    np.random.seed(42)

    # Values between -3.0 and 3.0 to simulate normalized financial data
    input_tensor = (torch.rand(batch_size, seq_len, n_features, device=device) * 6.0) - 3.0

    with torch.no_grad():
        output = model(input_tensor)

    if isinstance(output, tuple):
        output = output[0] # Take logits/continuous output

    # Convert to numpy
    input_np = input_tensor.cpu().numpy().astype(np.float32)
    output_np = output.cpu().numpy().astype(np.float32)

    # Save binaries
    out_dir = Path("tests/data")
    out_dir.mkdir(parents=True, exist_ok=True)

    input_np.tofile(out_dir / "test_input.bin")
    output_np.tofile(out_dir / "test_output.bin")

    print(f"Saved test_input.bin ({input_np.shape})")
    print(f"Saved test_output.bin ({output_np.shape})")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python verify_onnx_export.py <checkpoint_path> <model_name>")
        sys.exit(1)

    generate_test_data(sys.argv[1], sys.argv[2])
