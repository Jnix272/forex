import json

# Note: this requires that `training.train_gpu` can be imported
from training.train_gpu import _atomic_copy, _promote_best_fold


def test_atomic_copy_success(tmp_path):
    """Test that atomic_copy creates the destination and removes temporary files."""
    src = tmp_path / "src.pt"
    src.write_text("dummy data")

    dst_dir = tmp_path / "nested" / "dest"
    dst = dst_dir / "target.pt"

    # Run the atomic copy
    _atomic_copy(src, dst)

    assert dst.exists()
    assert dst.read_text() == "dummy data"

    # Check that no hidden tmp files remain
    files = list(dst_dir.glob(".*.tmp"))
    assert len(files) == 0

def test_promote_best_fold_flat_dir(tmp_path):
    """Test _promote_best_fold correctly selects the best fold and writes fold_selection.json."""
    model_name = "test_model"

    # Create flat checkpoints
    for fold in range(3):
        ckpt = tmp_path / f"{model_name}_fold{fold}_best.pt"
        ckpt.write_text(f"weights fold {fold}")

    cv_hist = [
        {"fold": 0, "best_metric": 1.0},
        {"fold": 1, "best_metric": 2.5},  # best!
        {"fold": 2, "best_metric": 0.5},
    ]

    _promote_best_fold(model_name, str(tmp_path), cv_hist, early_stop_metric="sharpe")

    # Assert promoted file exists
    promoted = tmp_path / f"{model_name}_best.pt"
    assert promoted.exists()
    assert promoted.read_text() == "weights fold 1"

    # Assert fold_selection.json
    selection_file = tmp_path / "fold_selection.json"
    assert selection_file.exists()
    data = json.loads(selection_file.read_text())
    assert data["selected_fold"] == 1
    assert data["metric_value"] == 2.5

def test_promote_best_fold_nested_dir(tmp_path):
    """Test that _promote_best_fold searches nested directories if flat ones don't exist."""
    model_name = "nested_model"
    model_dir = tmp_path / model_name
    model_dir.mkdir()

    # Create nested checkpoints
    ckpt = model_dir / f"{model_name}_fold0_best.pt"
    ckpt.write_text("nested weights")

    cv_hist = [
        {"fold": 0, "best_metric": 1.5},
    ]

    _promote_best_fold(model_name, str(tmp_path), cv_hist, early_stop_metric="sharpe")

    promoted = tmp_path / f"{model_name}_best.pt"
    assert promoted.exists()
    assert promoted.read_text() == "nested weights"

def test_promote_best_fold_tie_breaker(tmp_path):
    """Test that tie breaker (val_loss) is used when sharpe is identical."""
    model_name = "tie_model"

    for fold in range(2):
        ckpt = tmp_path / f"{model_name}_fold{fold}_best.pt"
        ckpt.write_text(f"weights fold {fold}")

        cfg = tmp_path / f"{model_name}_fold{fold}_config.json"
        cfg.write_text(json.dumps({
            "best_val_sharpe_proxy": 2.0,
            "best_val_loss": 1.0 if fold == 0 else 0.5  # Fold 1 has lower loss!
        }))

    cv_hist = [
        {"fold": 0, "best_metric": 2.0},
        {"fold": 1, "best_metric": 2.0},
    ]

    _promote_best_fold(model_name, str(tmp_path), cv_hist, early_stop_metric="sharpe")

    selection_file = tmp_path / "fold_selection.json"
    assert selection_file.exists()
    data = json.loads(selection_file.read_text())

    # Fold 1 should win because of lower loss
    assert data["selected_fold"] == 1

    cand = data["candidates"]
    assert len(cand) == 2
    assert cand[0]["tie_breaker"] == -1.0
    assert cand[1]["tie_breaker"] == -0.5
