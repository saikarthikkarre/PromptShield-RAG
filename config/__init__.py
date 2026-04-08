"""
PromptShiels: Configuration Loader
==================================
Handles loading and validation of configuration settings.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class Tier1Config:
    """Tier 1: Dynamic Outlier Filtration configuration."""
    top_k_retrieval: int = 10
    min_k_clusters: int = 2
    max_k_clusters: int = 3
    silhouette_threshold: float = 0.3
    min_cluster_ratio: float = 0.3


@dataclass
class Tier2Config:
    """Tier 2: Latent Activation Analysis configuration."""
    target_layers: list = field(default_factory=lambda: [12, 13, 14, 15])
    classifier_path: str = "models/tier2_classifier.pkl"
    activation_dim: int = 4096
    confidence_threshold: float = 0.7


@dataclass
class IAOConfig:
    """IAO Pipeline configuration."""
    base_dataset: str = "ms_marco"
    num_iterations: int = 5
    similarity_threshold: float = 0.8
    max_samples: int = 3000
    poison_ratio: float = 0.5


@dataclass
class ModelConfig:
    """Model configuration."""
    victim_llm: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class PathConfig:
    """Path configuration."""
    data_dir: str = "data"
    models_dir: str = "models"
    logs_dir: str = "logs"
    cache_dir: str = "cache"


class Config:
    """Main configuration class for PromptShiels."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration from YAML file.
        
        Args:
            config_path: Path to config YAML file. If None, uses default.
        """
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        
        self.config_path = Path(config_path)
        self._raw_config: Dict[str, Any] = {}
        
        # Load configuration
        self._load_config()
        
        # Initialize sub-configs
        self.models = ModelConfig(**self._raw_config.get("models", {}))
        self.tier1 = Tier1Config(**self._raw_config.get("tier1", {}))
        self.tier2 = Tier2Config(**self._raw_config.get("tier2", {}))
        self.iao = IAOConfig(**self._raw_config.get("iao", {}))
        self.paths = PathConfig(**self._raw_config.get("paths", {}))
        
        # Create directories
        self._create_directories()
    
    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                self._raw_config = yaml.safe_load(f) or {}
        else:
            print(f"Warning: Config file not found at {self.config_path}. Using defaults.")
            self._raw_config = {}
    
    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        base_path = self.config_path.parent.parent
        
        for dir_name in [self.paths.data_dir, self.paths.models_dir, 
                         self.paths.logs_dir, self.paths.cache_dir]:
            dir_path = base_path / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def get_absolute_path(self, relative_path: str) -> Path:
        """Convert relative path to absolute path based on project root."""
        base_path = self.config_path.parent.parent
        return base_path / relative_path
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "models": {
                "victim_llm": self.models.victim_llm,
                "embedding_model": self.models.embedding_model,
            },
            "tier1": {
                "top_k_retrieval": self.tier1.top_k_retrieval,
                "min_k_clusters": self.tier1.min_k_clusters,
                "max_k_clusters": self.tier1.max_k_clusters,
                "silhouette_threshold": self.tier1.silhouette_threshold,
                "min_cluster_ratio": self.tier1.min_cluster_ratio,
            },
            "tier2": {
                "target_layers": self.tier2.target_layers,
                "classifier_path": self.tier2.classifier_path,
                "activation_dim": self.tier2.activation_dim,
                "confidence_threshold": self.tier2.confidence_threshold,
            },
            "iao": {
                "base_dataset": self.iao.base_dataset,
                "num_iterations": self.iao.num_iterations,
                "similarity_threshold": self.iao.similarity_threshold,
                "max_samples": self.iao.max_samples,
                "poison_ratio": self.iao.poison_ratio,
            },
            "paths": {
                "data_dir": self.paths.data_dir,
                "models_dir": self.paths.models_dir,
                "logs_dir": self.paths.logs_dir,
                "cache_dir": self.paths.cache_dir,
            }
        }


# Global config instance
_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get the global configuration instance.
    
    Args:
        config_path: Optional path to config file. Only used on first call.
    
    Returns:
        Config instance.
    """
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config


def reload_config(config_path: Optional[str] = None) -> Config:
    """
    Reload configuration from file.
    
    Args:
        config_path: Optional path to config file.
    
    Returns:
        New Config instance.
    """
    global _config
    _config = Config(config_path)
    return _config
