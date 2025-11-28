import yaml
from pathlib import Path

class Config:
    def __init__(self, cfg_dict):
        self._cfg = cfg_dict

    def __getitem__(self, item):
        return self._cfg[item]

    def get(self, item, default=None):
        return self._cfg.get(item, default)

    def __repr__(self):
        return f"Config({self._cfg})"

def load_config(path: str):
    """
    Load a YAML config file and return a Config object.
    """
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open(cfg_path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    cfg = Config(cfg_dict)
    print(f"[Config] Loaded config from: {cfg_path}")
    return cfg