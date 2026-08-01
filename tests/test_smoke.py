import subprocess
import json

def test_train_gpu_synthetic_smoke(tmp_path):
    """
    Run train_gpu.py end-to-end with synthetic data for 2 epochs.
    Validates that:
      - The script completes successfully.
      - Checkpoint directory is created.
      - Manifest, train_summary, and deployment files are written.
    """
    import sys
    checkpoint_dir = tmp_path / "checkpoints"
    
    cmd = [
        sys.executable, "training/train_gpu.py",
        "--epochs", "2",
        "--force-rebuild",
        "--no-wandb",
        "--data-source", "synthetic",
        "--model", "haelt",
        "--checkpoint-dir", str(checkpoint_dir),
        "--amp",
        "--quick",
        "--n-ticks", "10000",
        "--chunk-size", "5000",
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
    
    assert result.returncode == 0, "train_gpu.py failed with synthetic data"
    
    model_dir = checkpoint_dir / "haelt"
    assert model_dir.exists(), "Model checkpoint directory was not created"
    
    # Check that artifact files are written
    assert (model_dir / "train_summary.json").exists(), "train_summary.json missing"
    assert (model_dir / "manifest.json").exists(), "manifest.json missing"
    
    # Since we train and evaluate, it should attempt deployment
    assert (model_dir / "deployment.json").exists(), "deployment.json missing"
    
    with open(model_dir / "train_summary.json", "r") as f:
        summary = json.load(f)
        assert summary["model_name"] == "haelt"
        assert summary["epochs_completed"] >= 1
        
    with open(model_dir / "deployment.json", "r") as f:
        deploy = json.load(f)
        assert deploy["model_name"] == "haelt"
