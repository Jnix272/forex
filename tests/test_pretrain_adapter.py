"""Tests for pretraining adapters."""
import pytest
import numpy as np

# Skip if torch not available (sandbox environment)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestPretrainConfig:
    """Test PretrainConfig dataclass."""
    
    def test_default_config(self):
        from training.pretrain_adapter import PretrainConfig
        
        config = PretrainConfig()
        assert config.input_dims == 1
        assert config.output_dims == 320
        assert config.batch_size == 16
        assert config.lr == 1e-3
    
    def test_custom_config(self):
        from training.pretrain_adapter import PretrainConfig
        
        config = PretrainConfig(
            input_dims=10,
            output_dims=128,
            batch_size=32,
            lr=1e-4,
            max_epochs=50,
        )
        assert config.input_dims == 10
        assert config.output_dims == 128
        assert config.batch_size == 32
        assert config.lr == 1e-4
        assert config.max_epochs == 50


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
class TestTNCAdapter:
    """Test TNC (Temporal Neighborhood Coding) adapter."""
    
    def test_tnc_adapter_creation(self):
        from training.pretrain_adapter import TNCAdapter, PretrainConfig
        
        config = PretrainConfig(input_dims=5, output_dims=64, max_epochs=2)
        adapter = TNCAdapter(config)
        
        assert adapter is not None
        assert adapter.config.input_dims == 5
        assert not adapter.is_fitted
    
    def test_tnc_fit_encode(self):
        from training.pretrain_adapter import TNCAdapter, PretrainConfig
        
        config = PretrainConfig(
            input_dims=3,
            output_dims=32,
            hidden_dims=32,
            max_epochs=2,
            batch_size=8,
            lr=1e-3,
            verbose=False,
        )
        adapter = TNCAdapter(config)
        
        # Create dummy time series data
        np.random.seed(42)
        train_data = np.random.randn(100, 20, 3).astype(np.float32)
        
        # Train
        history = adapter.fit(train_data, neighborhood_size=5)
        
        assert adapter.is_fitted
        assert "losses" in history
        assert len(history["losses"]) == 2
        
        # Encode
        test_data = np.random.randn(10, 20, 3).astype(np.float32)
        reprs = adapter.encode(test_data)
        
        assert reprs.shape == (10, 32)
    
    def test_tnc_save_load(self):
        from training.pretrain_adapter import TNCAdapter, PretrainConfig
        import tempfile
        import os
        
        config = PretrainConfig(
            input_dims=3,
            output_dims=16,
            hidden_dims=16,
            max_epochs=1,
            verbose=False,
        )
        adapter = TNCAdapter(config)
        
        train_data = np.random.randn(50, 10, 3).astype(np.float32)
        adapter.fit(train_data, neighborhood_size=3)
        
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        
        try:
            adapter.save(path)
            assert os.path.exists(path)
            
            # Load into new adapter
            adapter2 = TNCAdapter(config)
            adapter2.load(path)
            
            assert adapter2.is_fitted
            
            # Test encode works after load
            test_data = np.random.randn(5, 10, 3).astype(np.float32)
            reprs = adapter2.encode(test_data)
            assert reprs.shape == (5, 16)
        finally:
            if os.path.exists(path):
                os.remove(path)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
class TestLightlySoloAdapter:
    """Test lightly-ssl / solo-learn adapter."""
    
    def test_lightly_adapter_creation(self):
        from training.pretrain_adapter import LightlySoloAdapter, PretrainConfig
        
        config = PretrainConfig(input_dims=4, output_dims=64, max_epochs=1)
        adapter = LightlySoloAdapter(config, framework="lightly", method="simclr", backbone="resnet18")
        
        assert adapter is not None
        assert adapter.framework == "lightly"
        assert adapter.method == "simclr"
    
    @pytest.mark.skipif(True, reason="lightly not installed in test environment")
    def test_lightly_fit_encode(self):
        from training.pretrain_adapter import LightlySoloAdapter, PretrainConfig
        
        config = PretrainConfig(
            input_dims=2,
            output_dims=16,
            hidden_dims=16,
            max_epochs=1,
            batch_size=4,
            lr=1e-3,
            verbose=False,
        )
        adapter = LightlySoloAdapter(config, framework="lightly", method="simclr")
        
        # Create dummy time series data
        np.random.seed(42)
        train_data = np.random.randn(20, 10, 2).astype(np.float32)
        
        # Train
        history = adapter.fit(train_data)
        
        assert adapter.is_fitted
        assert "losses" in history
        assert len(history["losses"]) == 1
        
        # Encode
        test_data = np.random.randn(5, 10, 2).astype(np.float32)
        reprs = adapter.encode(test_data)
        
        assert reprs.shape == (5, 16)
    
    @pytest.mark.skipif(True, reason="lightly not installed in test environment")
    def test_lightly_invalid_method(self):
        from training.pretrain_adapter import LightlySoloAdapter, PretrainConfig
        
        config = PretrainConfig(max_epochs=1)
        adapter = LightlySoloAdapter(config, framework="lightly", method="invalid_method")
        
        train_data = np.random.randn(10, 5, 2).astype(np.float32)
        
        with pytest.raises(NotImplementedError):
            adapter.fit(train_data)


class TestCustomPretrainAdapter:
    """Test custom pretraining adapter (existing BYOL, Masked, etc.)."""
    
    def test_custom_adapter_creation(self):
        from training.pretrain_adapter import CustomPretrainAdapter, PretrainConfig
        
        config = PretrainConfig(max_epochs=10)
        adapter = CustomPretrainAdapter(config, method="byol")
        
        assert adapter is not None
        assert adapter.method == "byol"
        assert not adapter.is_fitted
    
    def test_custom_adapter_methods(self):
        from training.pretrain_adapter import CustomPretrainAdapter, PretrainConfig
        
        for method in ["byol", "masked", "vae", "forecast", "drift", "cluster", "tscl"]:
            config = PretrainConfig(max_epochs=1)
            adapter = CustomPretrainAdapter(config, method=method)
            assert adapter.method == method


class TestFactoryFunction:
    """Test create_pretrain_adapter factory function."""
    
    def test_create_ts2vec(self):
        from training.pretrain_adapter import create_pretrain_adapter, PretrainConfig
        from training.pretrain_adapter import TS2VecAdapter
        
        config = PretrainConfig()
        adapter = create_pretrain_adapter("ts2vec", config)
        
        assert isinstance(adapter, TS2VecAdapter)
    
    def test_create_tnc(self):
        from training.pretrain_adapter import create_pretrain_adapter, PretrainConfig
        from training.pretrain_adapter import TNCAdapter
        
        config = PretrainConfig()
        adapter = create_pretrain_adapter("tnc", config)
        
        assert isinstance(adapter, TNCAdapter)
    
    def test_create_custom(self):
        from training.pretrain_adapter import create_pretrain_adapter, PretrainConfig
        from training.pretrain_adapter import CustomPretrainAdapter
        
        config = PretrainConfig()
        adapter = create_pretrain_adapter("custom", config, method="byol")
        
        assert isinstance(adapter, CustomPretrainAdapter)
    
    def test_create_lightly(self):
        from training.pretrain_adapter import create_pretrain_adapter, PretrainConfig
        from training.pretrain_adapter import LightlySoloAdapter
        
        config = PretrainConfig()
        adapter = create_pretrain_adapter("lightly", config)
        
        assert isinstance(adapter, LightlySoloAdapter)
        assert adapter.framework == "lightly"
    
    def test_create_solo(self):
        from training.pretrain_adapter import create_pretrain_adapter, PretrainConfig
        from training.pretrain_adapter import LightlySoloAdapter
        
        config = PretrainConfig()
        adapter = create_pretrain_adapter("solo", config)
        
        assert isinstance(adapter, LightlySoloAdapter)
        assert adapter.framework == "solo"
    
    def test_create_invalid(self):
        from training.pretrain_adapter import create_pretrain_adapter, PretrainConfig
        
        config = PretrainConfig()
        with pytest.raises(ValueError):
            create_pretrain_adapter("invalid", config)


class TestRunPretrainWithAdapter:
    """Test integration with existing data pipeline."""
    
    def test_integration_import(self):
        """Test that the integration function can be imported."""
        from training.pretrain_adapter import run_pretrain_with_adapter
        
        assert run_pretrain_with_adapter is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])