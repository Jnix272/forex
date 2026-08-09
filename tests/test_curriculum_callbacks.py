"""Tests for curriculum callbacks (Composer, PyTorch Lightning, Custom)."""
import pytest
import numpy as np

# Skip if torch not available (sandbox environment)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestBaseCurriculum:
    """Test the framework-agnostic base curriculum logic."""
    
    def test_linear_pace(self):
        from training.curriculum_callbacks import BaseCurriculum, CurriculumConfig
        
        difficulty = np.linspace(0, 1, 100)
        config = CurriculumConfig(pace_function="linear", total_epochs=10, min_fraction=0.1)
        curriculum = BaseCurriculum(config, difficulty)
        
        # Epoch 0: should be at min_fraction
        info = curriculum.update(0)
        assert abs(info["pace"] - 0.1) < 0.01
        assert info["n_included"] == 10  # 10% of 100
        
        # Epoch 5: halfway
        info = curriculum.update(5)
        assert abs(info["pace"] - 0.55) < 0.02
        assert info["n_included"] == 55
        
        # Epoch 9: near end
        info = curriculum.update(9)
        assert abs(info["pace"] - 0.91) < 0.02
        assert info["n_included"] == 91
    
    def test_exp_pace(self):
        from training.curriculum_callbacks import BaseCurriculum, CurriculumConfig
        
        difficulty = np.linspace(0, 1, 100)
        config = CurriculumConfig(pace_function="exp", total_epochs=10, min_fraction=0.1, exp_rate=5.0)
        curriculum = BaseCurriculum(config, difficulty)
        
        info = curriculum.update(0)
        assert abs(info["pace"] - 0.1) < 0.01
        
        # Exponential should advance faster early on
        info = curriculum.update(3)
        # exp pace with rate=5 at epoch 3 (e=0.3): 0.1 + 0.9*(1-exp(-1.5)) ≈ 0.1 + 0.9*0.777 = 0.799
        assert info["pace"] > 0.5
    
    def test_sqrt_pace(self):
        from training.curriculum_callbacks import BaseCurriculum, CurriculumConfig
        
        difficulty = np.linspace(0, 1, 100)
        config = CurriculumConfig(pace_function="sqrt", total_epochs=10, min_fraction=0.1)
        curriculum = BaseCurriculum(config, difficulty)
        
        info = curriculum.update(0)
        assert abs(info["pace"] - 0.1) < 0.01
        
        # sqrt advances slower early, faster later
        info = curriculum.update(5)
        # sqrt pace at epoch 5 (e=0.5): 0.1 + 0.9*sqrt(0.5) ≈ 0.1 + 0.9*0.707 = 0.736
        assert info["pace"] > 0.6
    
    def test_step_pace(self):
        from training.curriculum_callbacks import BaseCurriculum, CurriculumConfig
        
        difficulty = np.linspace(0, 1, 100)
        config = CurriculumConfig(pace_function="step", total_epochs=10, min_fraction=0.1, n_steps=5)
        curriculum = BaseCurriculum(config, difficulty)
        
        # Step should have discrete jumps
        info = curriculum.update(0)
        pace_0 = info["pace"]
        
        info = curriculum.update(1)
        pace_1 = info["pace"]
        
        # With 5 steps over 10 epochs, steps at epochs 0, 2, 4, 6, 8
        info = curriculum.update(2)
        pace_2 = info["pace"]
        assert pace_2 > pace_0  # Should have advanced
    
    def test_log_pace(self):
        from training.curriculum_callbacks import BaseCurriculum, CurriculumConfig
        
        difficulty = np.linspace(0, 1, 100)
        config = CurriculumConfig(pace_function="log", total_epochs=10, min_fraction=0.1)
        curriculum = BaseCurriculum(config, difficulty)
        
        info = curriculum.update(0)
        assert abs(info["pace"] - 0.1) < 0.01
        
        info = curriculum.update(9)
        # log pace: 0.1 + 0.9*log(1+9*0.9)/log(10) ≈ 0.1 + 0.9*log(9.1)/log(10) ≈ 0.1 + 0.9*0.959 = 0.963
        assert info["pace"] > 0.9
    
    def test_inclusion_mask_sorted_by_difficulty(self):
        from training.curriculum_callbacks import BaseCurriculum, CurriculumConfig
        
        # Create difficulty where first 50 are easy (0.0), last 50 are hard (1.0)
        difficulty = np.concatenate([np.zeros(50), np.ones(50)])
        config = CurriculumConfig(pace_function="linear", total_epochs=10, min_fraction=0.1)
        curriculum = BaseCurriculum(config, difficulty)
        
        # At 10% pace, should include 10 easiest samples
        curriculum.update(0)
        mask = curriculum.get_inclusion_mask()
        assert mask.sum() == 10
        # All included should be from first 50 (easy)
        included_indices = np.where(mask)[0]
        assert all(idx < 50 for idx in included_indices)
        
        # At epoch 4: pace = 0.1 + 0.9 * (4/10) = 0.46, so 46 samples
        curriculum.update(4)
        mask = curriculum.get_inclusion_mask()
        assert mask.sum() == 46
        included_indices = np.where(mask)[0]
        assert all(idx < 50 for idx in included_indices)
        
        # At epoch 5: pace = 0.1 + 0.9 * (5/10) = 0.55, so 55 samples (includes some hard)
        curriculum.update(5)
        mask = curriculum.get_inclusion_mask()
        assert mask.sum() == 55
        included_indices = np.where(mask)[0]
        assert any(idx >= 50 for idx in included_indices)
    
    def test_loss_weighting(self):
        from training.curriculum_callbacks import BaseCurriculum, CurriculumConfig
        
        difficulty = np.linspace(0, 1, 100)
        config = CurriculumConfig(
            pace_function="linear", 
            total_epochs=10, 
            min_fraction=0.1,
            use_loss_weighting=True,
            loss_temperature=1.0
        )
        curriculum = BaseCurriculum(config, difficulty)
        
        # Without losses, should use uniform weights
        info = curriculum.update(0)
        weights = curriculum.get_sample_weights()
        assert weights.mean() == 0.1  # Only 10% included, all weight 1.0
        
        # With losses, included samples get weighted by exp(-loss)
        losses = np.linspace(0, 5, 100)  # Easy samples have low loss
        info = curriculum.update(0, losses)
        weights = curriculum.get_sample_weights()
        
        # Included samples (first 10) should have higher weights (lower loss)
        included_weights = weights[weights > 0]
        assert len(included_weights) == 10
        # First sample (loss=0) should have weight=1.0
        assert abs(included_weights[0] - 1.0) < 0.01
        # Last included sample (loss≈0.45) should have lower weight
        assert included_weights[-1] < 1.0
    
    def test_state_dict_checkpointing(self):
        from training.curriculum_callbacks import BaseCurriculum, CurriculumConfig
        
        difficulty = np.linspace(0, 1, 100)
        config = CurriculumConfig(pace_function="linear", total_epochs=10, min_fraction=0.1)
        curriculum = BaseCurriculum(config, difficulty)
        
        curriculum.update(5)
        state = curriculum.get_state()
        
        assert state["current_epoch"] == 5
        assert state["config"]["pace_function"] == "linear"
        
        # Create new curriculum and load state
        curriculum2 = BaseCurriculum(config, difficulty)
        curriculum2.load_state(state)
        
        assert curriculum2.current_epoch == 5


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
class TestPLCurriculumCallback:
    """Test PyTorch Lightning curriculum callback."""
    
    def test_pl_callback_creation(self):
        from training.curriculum_callbacks import PLCurriculumCallback
        
        difficulty = np.linspace(0, 1, 100)
        callback = PLCurriculumCallback(difficulty, pace_function="linear", total_epochs=10)
        
        assert callback is not None
        assert callback.curriculum is not None
        assert callback.verbose is True
    
    def test_pl_callback_state_dict(self):
        from training.curriculum_callbacks import PLCurriculumCallback
        
        difficulty = np.linspace(0, 1, 100)
        callback = PLCurriculumCallback(difficulty, pace_function="linear", total_epochs=10)
        
        # Simulate a few epochs
        callback.curriculum.update(3)
        state = callback.state_dict()
        
        assert "curriculum_state" in state
        assert state["curriculum_state"]["current_epoch"] == 3
        
        # Load state into new callback
        callback2 = PLCurriculumCallback(difficulty, pace_function="linear", total_epochs=10)
        callback2.load_state_dict(state)
        
        assert callback2.curriculum.current_epoch == 3


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
class TestComposerCurriculumCallback:
    """Test MosaicML Composer curriculum callback."""
    
    def test_composer_callback_creation(self):
        from training.curriculum_callbacks import ComposerCurriculumCallback
        
        difficulty = np.linspace(0, 1, 100)
        callback = ComposerCurriculumCallback(difficulty, pace_function="linear", total_epochs=10)
        
        assert callback is not None
        assert callback.curriculum is not None
        assert callback.verbose is True
    
    def test_composer_callback_match_apply(self):
        from training.curriculum_callbacks import ComposerCurriculumCallback
        
        difficulty = np.linspace(0, 1, 100)
        callback = ComposerCurriculumCallback(difficulty, pace_function="linear", total_epochs=10)
        
        # Test match
        assert callback.match("EPOCH_START", None)
        assert callback.match("BATCH_END", None)
        assert not callback.match("EPOCH_END", None)
        
        # Test state_dict
        callback.curriculum.update(3)
        state = callback.state_dict()
        assert "curriculum_state" in state


class TestCustomCurriculumAdapter:
    """Test adapter for existing custom curriculum implementations."""
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available - required by custom curriculum")
    def test_adapter_difficulty_mode(self):
        from training.curriculum_callbacks import CustomCurriculumAdapter
        
        difficulty = np.linspace(0, 1, 100)
        adapter = CustomCurriculumAdapter(difficulty, mode="difficulty", 
                                          pace_function="linear", advance_rate=0.1)
        
        info = adapter.update(0)
        assert info["current_epoch"] == 0
        assert info["inclusion_rate"] > 0
        
        mask = adapter.get_inclusion_mask()
        assert mask.sum() > 0
        
        weights = adapter.get_sample_weights()
        assert len(weights) == 100
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available - required by custom curriculum")
    def test_adapter_state_dict(self):
        from training.curriculum_callbacks import CustomCurriculumAdapter
        
        difficulty = np.linspace(0, 1, 100)
        adapter = CustomCurriculumAdapter(difficulty, mode="difficulty",
                                          pace_function="linear", advance_rate=0.1)
        
        adapter.update(5)
        state = adapter.state_dict()
        
        assert state["mode"] == "difficulty"
        assert state["current_epoch"] == 5
        
        # Load state
        adapter2 = CustomCurriculumAdapter(difficulty, mode="difficulty",
                                           pace_function="linear", advance_rate=0.1)
        adapter2.load_state_dict(state)
        
        assert adapter2._curriculum.current_epoch == 5


class TestFactoryFunction:
    """Test the create_curriculum_callback factory."""
    
    def test_create_pytorch_lightning(self):
        from training.curriculum_callbacks import create_curriculum_callback
        
        difficulty = np.linspace(0, 1, 100)
        callback = create_curriculum_callback(
            framework="pytorch_lightning",
            difficulty_scores=difficulty,
            pace_function="linear",
            total_epochs=10
        )
        
        from training.curriculum_callbacks import PLCurriculumCallback
        assert isinstance(callback, PLCurriculumCallback)
    
    def test_create_composer(self):
        from training.curriculum_callbacks import create_curriculum_callback
        
        difficulty = np.linspace(0, 1, 100)
        callback = create_curriculum_callback(
            framework="composer",
            difficulty_scores=difficulty,
            pace_function="linear",
            total_epochs=10
        )
        
        from training.curriculum_callbacks import ComposerCurriculumCallback
        assert isinstance(callback, ComposerCurriculumCallback)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available - required by custom curriculum")
    def test_create_custom(self):
        from training.curriculum_callbacks import create_curriculum_callback
        
        difficulty = np.linspace(0, 1, 100)
        callback = create_curriculum_callback(
            framework="custom",
            difficulty_scores=difficulty,
            mode="difficulty",
            pace_function="linear",
            advance_rate=0.1
        )
        
        from training.curriculum_callbacks import CustomCurriculumAdapter
        assert isinstance(callback, CustomCurriculumAdapter)
    
    def test_create_invalid_framework(self):
        from training.curriculum_callbacks import create_curriculum_callback
        
        difficulty = np.linspace(0, 1, 100)
        with pytest.raises(ValueError):
            create_curriculum_callback(
                framework="invalid",
                difficulty_scores=difficulty
            )


class TestIntegrationHelpers:
    """Test DataLoader integration helpers."""
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
    def test_integrate_curriculum_with_dataloader(self):
        from training.curriculum_callbacks import integrate_curriculum_with_dataloader
        import torch.utils.data as torch_data
        
        dataset = torch_data.TensorDataset(torch.randn(100, 10), torch.randint(0, 2, (100,)))
        loader = torch_data.DataLoader(dataset, batch_size=10, shuffle=True)
        
        weights = np.ones(100) * 0.5
        weights[:50] = 1.0  # First 50 samples weighted more
        
        new_loader = integrate_curriculum_with_dataloader(loader, weights)
        
        assert new_loader.batch_size == 10
        assert hasattr(new_loader, "sampler")
        assert isinstance(new_loader.sampler, torch_data.WeightedRandomSampler)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
    def test_make_curriculum_datamodule(self):
        from training.curriculum_callbacks import make_curriculum_datamodule, PLCurriculumCallback
        import torch.utils.data as torch_data
        
        class SimpleDataModule:
            def __init__(self):
                self.dataset = torch_data.TensorDataset(torch.randn(100, 10), torch.randint(0, 2, (100,)))
            
            def train_dataloader(self):
                return torch_data.DataLoader(self.dataset, batch_size=10, shuffle=True)
        
        base_dm = SimpleDataModule()
        difficulty = np.linspace(0, 1, 100)
        callback = PLCurriculumCallback(difficulty, pace_function="linear", total_epochs=10)
        
        CurriculumDM = make_curriculum_datamodule(base_dm, callback)
        curriculum_dm = CurriculumDM()
        
        assert hasattr(curriculum_dm, "set_curriculum_weights")
        assert hasattr(curriculum_dm, "train_dataloader")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])