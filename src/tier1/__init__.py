"""
PromptShiels: Tier 1 Module
===========================
Dynamic Outlier Filtration - The "Speed Trap"
"""
from .outlier_filter import (
    Tier1Filter,
    FilterResult,
    filter_documents
)

__all__ = [
    "Tier1Filter",
    "FilterResult", 
    "filter_documents"
]
