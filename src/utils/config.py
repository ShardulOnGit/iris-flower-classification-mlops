from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(config_path: str = "params.yaml") -> Dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    path = Path(config_path)
    if not path.is_file():
        # Fallback to configs/params.yaml if root params.yaml not found, and vice versa
        alt_path = Path("configs/params.yaml") if config_path == "params.yaml" else Path("params.yaml")
        if alt_path.is_file():
            path = alt_path
        else:
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config
