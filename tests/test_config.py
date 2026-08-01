import yaml
from pathlib import Path

def test_run_yaml_parses():
    config_path = Path("config/run.yaml")
    assert config_path.exists(), "config/run.yaml does not exist"
    
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        
    assert isinstance(data, dict), "Config should be a dictionary"
    
    # Required sections
    required_sections = ["data", "model", "training", "paths", "execution"]
    for sec in required_sections:
        assert sec in data, f"Missing required section: {sec}"
        
    # Checkpoint dir resolves
    checkpoint_dir = data.get("paths", {}).get("checkpoint_dir")
    assert checkpoint_dir is not None, "Missing checkpoint_dir in paths"
    
    resolved_path = Path(checkpoint_dir).expanduser().resolve()
    assert resolved_path.parent.exists() or resolved_path.exists() or str(resolved_path).startswith(str(Path.home())), "Checkpoint dir parent should exist or be under home"
