"""
PromptShiels: Main Defense Pipeline
====================================
Integrates Tier 1 (Outlier Filtration) and Tier 2 (Activation Analysis)
into a unified defense system.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path

from .tier1 import Tier1Filter, FilterResult
from .tier2 import Tier2Analyzer, Tier2Result

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DefenseResult:
    """Complete result from PromptShiels defense pipeline."""
    # Final output
    safe_docs: List[Dict[str, Any]]
    safe_indices: List[int]
    
    # Tier 1 results
    tier1_result: FilterResult
    tier1_dropped: int
    
    # Tier 2 results  
    tier2_result: Optional[Tier2Result]
    tier2_blocked: int
    
    # Statistics
    total_input: int
    total_output: int
    total_filtered: int
    defense_summary: str


class RAGSentinel:
    """
    PromptShiels: Hybrid Tiered Defense System
    
    A defense-in-depth architecture that protects RAG systems
    from indirect prompt injection attacks using two tiers:
    
    Tier 1 (Speed Trap): Fast outlier detection using clustering
    Tier 2 (Brain Scan): Deep activation analysis for stealthy attacks
    """
    
    def __init__(
        self,
        # Tier 1 settings
        tier1_enabled: bool = True,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        min_k: int = 2,
        max_k: int = 3,
        silhouette_threshold: float = 0.3,
        
        # Tier 2 settings
        tier2_enabled: bool = True,
        target_layers: List[int] = [12, 13, 14, 15],
        classifier_path: Optional[str] = None,
        confidence_threshold: float = 0.7,
        
        # General settings
        device: str = "cpu"
    ):
        """
        Initialize PromptShiels defense system.
        
        Args:
            tier1_enabled: Whether to enable Tier 1 filtering
            embedding_model: Sentence transformer model for Tier 1
            min_k: Minimum K for clustering
            max_k: Maximum K for clustering
            silhouette_threshold: Threshold for applying clustering filter
            tier2_enabled: Whether to enable Tier 2 analysis
            target_layers: LLM layers to probe for Tier 2
            classifier_path: Path to trained Tier 2 classifier
            confidence_threshold: Classification confidence threshold
            device: Device to run on ('cpu' or 'cuda')
        """
        self.tier1_enabled = tier1_enabled
        self.tier2_enabled = tier2_enabled
        self.device = device
        
        # Initialize Tier 1
        if tier1_enabled:
            logger.info("Initializing Tier 1: Dynamic Outlier Filtration")
            self.tier1 = Tier1Filter(
                embedding_model=embedding_model,
                min_k=min_k,
                max_k=max_k,
                silhouette_threshold=silhouette_threshold,
                device=device
            )
        else:
            self.tier1 = None
        
        # Initialize Tier 2
        if tier2_enabled:
            logger.info("Initializing Tier 2: Latent Activation Analysis")
            self.tier2 = Tier2Analyzer(
                target_layers=target_layers,
                classifier_path=classifier_path,
                confidence_threshold=confidence_threshold,
                device=device
            )
        else:
            self.tier2 = None
        
        logger.info(f"PromptShiels initialized. Tier 1: {tier1_enabled}, Tier 2: {tier2_enabled}")
    
    def set_llm(self, model: Any, tokenizer: Any) -> None:
        """
        Set the LLM model for Tier 2 analysis.
        
        Args:
            model: The transformer model
            tokenizer: Associated tokenizer
        """
        if self.tier2 is not None:
            self.tier2.set_model(model, tokenizer)
            logger.info("LLM model set for Tier 2 analysis")
    
    def defend(
        self,
        documents: List[Dict[str, Any]],
        query: Optional[str] = None,
        text_key: str = "text"
    ) -> DefenseResult:
        """
        Apply full defense pipeline to retrieved documents.
        
        Args:
            documents: List of retrieved document dictionaries
            query: Original user query
            text_key: Key to access document text
            
        Returns:
            DefenseResult with safe documents and analysis details
        """
        total_input = len(documents)
        logger.info(f"PromptShiels processing {total_input} documents...")
        
        # Track documents through pipeline
        current_docs = documents
        original_indices = list(range(total_input))
        
        # === TIER 1: Outlier Filtration ===
        tier1_result = None
        tier1_dropped = 0
        
        if self.tier1_enabled and self.tier1 is not None:
            logger.info("Running Tier 1: Dynamic Outlier Filtration")
            tier1_result = self.tier1.filter(current_docs, query, text_key)
            
            if tier1_result.filtering_applied:
                tier1_dropped = len(tier1_result.dropped_docs)
                current_docs = tier1_result.passed_docs
                original_indices = [original_indices[i] for i in tier1_result.passed_indices]
                logger.info(f"Tier 1 dropped {tier1_dropped} outlier documents")
            else:
                logger.info(f"Tier 1 passed all documents: {tier1_result.reason}")
        else:
            # Create dummy result
            tier1_result = FilterResult(
                passed_docs=current_docs,
                dropped_docs=[],
                passed_indices=list(range(len(current_docs))),
                dropped_indices=[],
                optimal_k=1,
                silhouette_score=0.0,
                cluster_labels=np.zeros(len(current_docs), dtype=int),
                majority_cluster=0,
                filtering_applied=False,
                reason="Tier 1 disabled"
            )
        
        # === TIER 2: Activation Analysis ===
        tier2_result = None
        tier2_blocked = 0
        
        if self.tier2_enabled and self.tier2 is not None and len(current_docs) > 0:
            logger.info("Running Tier 2: Latent Activation Analysis")
            tier2_result = self.tier2.analyze(current_docs, query, text_key)
            
            if tier2_result.analysis_applied:
                tier2_blocked = len(tier2_result.blocked_docs)
                current_docs = tier2_result.passed_docs
                original_indices = [original_indices[i] for i in tier2_result.passed_indices]
                logger.info(f"Tier 2 blocked {tier2_blocked} suspicious documents")
            else:
                logger.info(f"Tier 2 passed all documents: {tier2_result.reason}")
        else:
            tier2_result = Tier2Result(
                passed_docs=current_docs,
                blocked_docs=[],
                passed_indices=list(range(len(current_docs))),
                blocked_indices=[],
                classifications=[],
                analysis_applied=False,
                reason="Tier 2 disabled or no model set"
            )
        
        # === Compile Results ===
        total_filtered = tier1_dropped + tier2_blocked
        total_output = len(current_docs)
        
        summary = (
            f"PromptShiels Defense Complete:\n"
            f"  Input: {total_input} documents\n"
            f"  Tier 1 (Outlier Filter): Dropped {tier1_dropped}\n"
            f"  Tier 2 (Activation Analysis): Blocked {tier2_blocked}\n"
            f"  Output: {total_output} safe documents"
        )
        
        logger.info(summary)
        
        return DefenseResult(
            safe_docs=current_docs,
            safe_indices=original_indices,
            tier1_result=tier1_result,
            tier1_dropped=tier1_dropped,
            tier2_result=tier2_result,
            tier2_blocked=tier2_blocked,
            total_input=total_input,
            total_output=total_output,
            total_filtered=total_filtered,
            defense_summary=summary
        )
    
    def get_statistics(self, result: DefenseResult) -> Dict[str, Any]:
        """
        Get detailed statistics from a defense result.
        
        Args:
            result: DefenseResult from defend()
            
        Returns:
            Dictionary of statistics
        """
        stats = {
            "input_documents": result.total_input,
            "output_documents": result.total_output,
            "total_filtered": result.total_filtered,
            "filter_rate": result.total_filtered / result.total_input if result.total_input > 0 else 0,
            "tier1": {
                "enabled": self.tier1_enabled,
                "dropped": result.tier1_dropped,
                "applied": result.tier1_result.filtering_applied if result.tier1_result else False,
                "silhouette_score": result.tier1_result.silhouette_score if result.tier1_result else 0,
                "optimal_k": result.tier1_result.optimal_k if result.tier1_result else 0,
            },
            "tier2": {
                "enabled": self.tier2_enabled,
                "blocked": result.tier2_blocked,
                "applied": result.tier2_result.analysis_applied if result.tier2_result else False,
            }
        }
        
        # Add Tier 2 classification details if available
        if result.tier2_result and result.tier2_result.classifications:
            confidences = [c.confidence for c in result.tier2_result.classifications]
            stats["tier2"]["avg_confidence"] = np.mean(confidences)
            stats["tier2"]["min_confidence"] = np.min(confidences)
            stats["tier2"]["max_confidence"] = np.max(confidences)
        
        return stats
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.tier2 is not None:
            self.tier2.cleanup()


# Convenience function
def create_sentinel(
    config_path: Optional[str] = None,
    tier1_only: bool = False,
    tier2_only: bool = False,
    device: str = "cpu"
) -> RAGSentinel:
    """
    Factory function to create a PromptShiels instance.
    
    Args:
        config_path: Path to configuration file
        tier1_only: Only enable Tier 1
        tier2_only: Only enable Tier 2
        device: Device to run on
        
    Returns:
        Configured RAGSentinel instance
    """
    from config import get_config
    
    config = get_config(config_path)
    
    tier1_enabled = not tier2_only
    tier2_enabled = not tier1_only
    
    return RAGSentinel(
        tier1_enabled=tier1_enabled,
        embedding_model=config.models.embedding_model,
        min_k=config.tier1.min_k_clusters,
        max_k=config.tier1.max_k_clusters,
        silhouette_threshold=config.tier1.silhouette_threshold,
        tier2_enabled=tier2_enabled,
        target_layers=config.tier2.target_layers,
        classifier_path=str(config.get_absolute_path(config.tier2.classifier_path)),
        confidence_threshold=config.tier2.confidence_threshold,
        device=device
    )
