import subprocess
import argparse
import sys
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run diverse model recipes sequentially.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--data-source", type=str, default="synthetic")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/diverse_test")
    args = parser.parse_args()

    models = ["haelt", "mamba", "tft"]
    base_cmd = [
        sys.executable, "training/train_gpu.py",
        "--epochs", str(args.epochs),
        "--data-source", args.data_source,
        "--checkpoint-dir", args.checkpoint_dir,
        "--force-rebuild"
    ]

    results = {}

    for model in models:
        print(f"\n{'='*50}\nTraining Model Recipe: {model.upper()}\n{'='*50}")
        cmd = base_cmd + ["--model", model]
        
        # Run training
        try:
            subprocess.run(cmd, check=True)
            print(f"[Success] {model} finished training.")
            
            # Read back the deployment.json or train_summary.json
            summary_path = Path(args.checkpoint_dir) / model / "train_summary.json"
            if summary_path.exists():
                with open(summary_path, "r") as f:
                    summary = json.load(f)
                    results[model] = summary.get("best_val_sharpe", 0.0)
            else:
                results[model] = "N/A"
                
        except subprocess.CalledProcessError as e:
            print(f"[Failed] {model} failed with exit code {e.returncode}")
            results[model] = "Failed"
            
    print(f"\n{'='*50}\nDiverse Recipes Results\n{'='*50}")
    for m, score in results.items():
        print(f"Model: {m.ljust(10)} | Val Sharpe: {score}")

if __name__ == "__main__":
    main()
