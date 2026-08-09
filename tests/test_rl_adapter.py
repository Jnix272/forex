"""Tests for RL adapters (CleanRL, Stable-Baselines3, Custom)."""
import pytest
import numpy as np

# Skip if torch not available (sandbox environment)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestRLConfig:
    """Test RLConfig dataclass."""
    
    def test_default_config(self):
        from training.rl_adapter import RLConfig
        
        config = RLConfig()
        assert config.algorithm == "ppo"
        assert config.learning_rate == 3e-4
        assert config.gamma == 0.99
        assert config.total_timesteps == 100000
    
    def test_custom_config(self):
        from training.rl_adapter import RLConfig
        
        config = RLConfig(
            algorithm="dqn",
            learning_rate=1e-4,
            hidden_dims=(128, 128),
            total_timesteps=50000,
        )
        assert config.algorithm == "dqn"
        assert config.learning_rate == 1e-4
        assert config.hidden_dims == (128, 128)
        assert config.total_timesteps == 50000


class TestGymEnvWrapper:
    """Test Gymnasium environment wrapper."""
    
    @pytest.fixture
    def mock_env(self):
        class MockEnv:
            def __init__(self):
                self.obs_size = 10
                self.n_actions = 5
                self._done = False
            
            def reset(self, valid_starts=None):
                self._done = False
                return np.zeros(10, dtype=np.float32)
            
            def step(self, action):
                self._done = True
                return np.zeros(10, dtype=np.float32), 1.0, True, {}
            
            def action_mask(self):
                return np.ones(5, dtype=bool)
        return MockEnv()
    
    def test_wrapper_creation(self, mock_env):
        # Skip if gymnasium not available
        try:
            import gymnasium
        except ImportError:
            pytest.skip("gymnasium not available")
        
        from training.rl_adapter import GymEnvWrapper
        wrapper = GymEnvWrapper(mock_env)
        
        assert wrapper.observation_space.shape == (10,)
        assert wrapper.action_space.n == 5
    
    def test_wrapper_reset_step(self, mock_env):
        # Skip if gymnasium not available
        try:
            import gymnasium
        except ImportError:
            pytest.skip("gymnasium not available")
        
        from training.rl_adapter import GymEnvWrapper
        
        class MockEnv2:
            def __init__(self):
                self.obs_size = 4
                self.n_actions = 2
                self._step_count = 0
            
            def reset(self, valid_starts=None):
                self._step_count = 0
                return np.ones(4, dtype=np.float32)
            
            def step(self, action):
                self._step_count += 1
                done = self._step_count >= 3
                return np.ones(4, dtype=np.float32), 1.0, done, {"steps": self._step_count}
            
            def action_mask(self):
                return np.array([True, True])
        
        env = MockEnv2()
        wrapper = GymEnvWrapper(env)
        
        obs, info = wrapper.reset()
        assert obs.shape == (4,)
        assert np.all(obs == 1.0)
        
        obs, reward, terminated, truncated, info = wrapper.step(0)
        assert not terminated
        assert not truncated
        assert reward == 1.0
        
        # Step until done
        for _ in range(2):
            obs, reward, terminated, truncated, info = wrapper.step(0)
        
        assert terminated
        assert not truncated


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
class TestCustomRLAdapter:
    """Test custom RL adapter (existing rl_agents.py)."""
    
    def test_custom_adapter_creation(self):
        from training.rl_adapter import CustomRLAdapter, RLConfig
        
        config = RLConfig(algorithm="ppo", total_timesteps=1000)
        adapter = CustomRLAdapter(config)
        
        assert adapter is not None
        assert adapter.config.algorithm == "ppo"
        assert not adapter.is_trained
    
    def test_custom_adapter_dqn(self):
        from training.rl_adapter import CustomRLAdapter, RLConfig
        
        config = RLConfig(algorithm="dqn", total_timesteps=1000)
        adapter = CustomRLAdapter(config)
        
        assert adapter.config.algorithm == "dqn"
    
    def test_custom_adapter_invalid_algorithm(self):
        from training.rl_adapter import CustomRLAdapter, RLConfig
        
        config = RLConfig(algorithm="sac")  # Not supported in custom
        adapter = CustomRLAdapter(config)
        
        # Should fail when trying to train
        class MockEnv:
            obs_size = 4
            n_actions = 2
            def reset(self, valid_starts=None): return np.zeros(4)
            def step(self, action): return np.zeros(4), 0, True, {}
            def action_mask(self): return np.array([True, True])
        
        with pytest.raises(NotImplementedError):
            adapter.train(MockEnv())


class TestFactoryFunction:
    """Test create_rl_adapter factory function."""
    
    def test_create_cleanrl(self):
        from training.rl_adapter import create_rl_adapter, RLConfig
        from training.rl_adapter import CleanRLAdapter
        
        config = RLConfig()
        adapter = create_rl_adapter("cleanrl", "ppo", config)
        
        assert isinstance(adapter, CleanRLAdapter)
    
    def test_create_sb3(self):
        from training.rl_adapter import create_rl_adapter, RLConfig
        from training.rl_adapter import SB3Adapter
        
        config = RLConfig()
        adapter = create_rl_adapter("sb3", "ppo", config)
        
        assert isinstance(adapter, SB3Adapter)
    
    def test_create_stable_baselines3(self):
        from training.rl_adapter import create_rl_adapter, RLConfig
        from training.rl_adapter import SB3Adapter
        
        config = RLConfig()
        adapter = create_rl_adapter("stable_baselines3", "ppo", config)
        
        assert isinstance(adapter, SB3Adapter)
    
    def test_create_custom(self):
        from training.rl_adapter import create_rl_adapter, RLConfig
        from training.rl_adapter import CustomRLAdapter
        
        config = RLConfig()
        adapter = create_rl_adapter("custom", "ppo", config)
        
        assert isinstance(adapter, CustomRLAdapter)
    
    def test_create_invalid_framework(self):
        from training.rl_adapter import create_rl_adapter, RLConfig
        
        config = RLConfig()
        with pytest.raises(ValueError):
            create_rl_adapter("invalid", "ppo", config)


class TestRunRLWithAdapter:
    """Test integration function."""
    
    def test_integration_import(self):
        """Test that the integration function can be imported."""
        from training.rl_adapter import run_rl_with_adapter
        
        assert run_rl_with_adapter is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])