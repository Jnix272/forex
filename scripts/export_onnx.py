import argparse
import sys
from pathlib import Path

# Add the project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from inference.onnx_inference import export_to_onnx as _export_to_onnx


def export_to_onnx(
    checkpoint_path: str,
    model_name: str,
    output_path: str,
    seq_len: int = 60,
    n_features: int | None = None,
    opset: int = 17,
):
    """
    Delegate to the canonical exporter used by training/live deployment.
    """
    return _export_to_onnx(
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        seq_len=seq_len,
        output_path=output_path,
        opset=opset,
        n_features=n_features,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch models to ONNX.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pt checkpoint file")
    parser.add_argument("--model-name", type=str, default="haelt", help="Architecture name (e.g., haelt, tft)")
    parser.add_argument("--output", type=str, default="models/haelt.onnx", help="Output path for the .onnx file")
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--n-feat", type=int, default=None)
    parser.add_argument("--opset", type=int, default=17)

    args = parser.parse_args()

    # Ensure the output directory exists
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    export_to_onnx(
        args.checkpoint,
        args.model_name,
        args.output,
        seq_len=args.seq_len,
        n_features=args.n_feat,
        opset=args.opset,
    )
