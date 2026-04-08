"""
PromptShiels: A Hybrid Tiered Defense System Against
Indirect Prompt Injection in Retrieval-Augmented Generation
============================================================
"""

from .promptshield import RAGSentinel, DefenseResult, create_sentinel
from .tier1 import Tier1Filter, FilterResult, filter_documents
from .tier2 import Tier2Analyzer, Tier2Result, PoisonClassifier
from .data import IAOPipeline, DatasetBuilder, create_pipeline

__version__ = "1.0.0"
__author__ = "PromptShiels Team"

__all__ = [
    # Main
    "RAGSentinel",
    "DefenseResult",
    "create_sentinel",
    
    # Tier 1
    "Tier1Filter",
    "FilterResult",
    "filter_documents",
    
    # Tier 2
    "Tier2Analyzer",
    "Tier2Result", 
    "PoisonClassifier",
    
    # Data
    "IAOPipeline",
    "DatasetBuilder",
    "create_pipeline",
]
