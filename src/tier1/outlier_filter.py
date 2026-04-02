"""
RAG-Sentinel: Tier 1 - Dynamic Outlier Filtration
=================================================
The "Speed Trap" - Instantly discards mathematically suspicious documents
using Dynamic Semantic Consensus Clustering.

Algorithm Steps:
1. Retrieval: Fetch top N documents for a query
2. Vectorization: Encode all docs using a fast embedding model
3. L2 Normalization: Normalize vectors to unit length
4. Dynamic K-Selection: Use Silhouette Analysis to find optimal K
5. Filtration: Keep majority cluster, discard outliers
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    """Result of Tier 1 filtering."""
    passed_docs: List[Dict[str, Any]]
    dropped_docs: List[Dict[str, Any]]
    passed_indices: List[int]
    dropped_indices: List[int]
    optimal_k: int
    silhouette_score: float
    cluster_labels: np.ndarray
    majority_cluster: int
    filtering_applied: bool
    reason: str


class Tier1Filter:
    """
    Tier 1: Dynamic Outlier Filtration
    
    Uses semantic clustering to identify and remove potentially
    malicious documents that are semantically dissimilar from
    the consensus (majority cluster).
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        min_k: int = 2,
        max_k: int = 3,
        silhouette_threshold: float = 0.3,
        min_cluster_ratio: float = 0.3,
        device: str = "cpu"
    ):
        """
        Initialize Tier 1 Filter.
        
        Args:
            embedding_model: Name/path of the sentence transformer model
            min_k: Minimum number of clusters to try
            max_k: Maximum number of clusters to try
            silhouette_threshold: Minimum silhouette score to apply filtering
            min_cluster_ratio: Minimum ratio of docs in majority cluster
            device: Device to run embeddings on ('cpu' or 'cuda')
        """
        self.min_k = min_k
        self.max_k = max_k
        self.silhouette_threshold = silhouette_threshold
        self.min_cluster_ratio = min_cluster_ratio
        self.device = device
        
        logger.info(f"Loading embedding model: {embedding_model}")
        self.encoder = SentenceTransformer(embedding_model, device=device)
        logger.info("Tier 1 Filter initialized successfully")
    
    def encode_documents(self, documents: List[str]) -> np.ndarray:
        """
        Encode documents into dense vectors.
        
        Args:
            documents: List of document texts
            
        Returns:
            Document embeddings as numpy array
        """
        embeddings = self.encoder.encode(
            documents,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False  # We'll normalize separately
        )
        return embeddings
    
    def l2_normalize(self, vectors: np.ndarray) -> np.ndarray:
        """
        Apply L2 normalization to vectors.
        
        Scientific Justification:
        Normalizing to unit length ensures clustering is based
        purely on semantic angle (meaning), removing bias from
        document length variations.
        
        Args:
            vectors: Input vectors
            
        Returns:
            L2-normalized vectors
        """
        return normalize(vectors, norm='l2', axis=1)
    
    def find_optimal_k(
        self,
        vectors: np.ndarray,
        random_state: int = 42
    ) -> Tuple[int, float, np.ndarray]:
        """
        Find optimal K using Silhouette Analysis.
        
        Scientific Justification:
        Silhouette score measures how similar an object is to its own
        cluster compared to other clusters. A high score means well-defined
        clusters exist (potential anomaly detected).
        
        Args:
            vectors: Normalized document vectors
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (optimal_k, best_silhouette_score, cluster_labels)
        """
        n_samples = len(vectors)
        
        # Need at least 2 samples for clustering
        if n_samples < 2:
            return 1, 0.0, np.zeros(n_samples, dtype=int)
        
        # Cannot have more clusters than samples
        actual_max_k = min(self.max_k, n_samples)
        actual_min_k = min(self.min_k, actual_max_k)
        
        if actual_min_k == actual_max_k or n_samples <= actual_min_k:
            # Not enough samples for meaningful clustering
            kmeans = KMeans(n_clusters=actual_min_k, random_state=random_state, n_init=10)
            labels = kmeans.fit_predict(vectors)
            if n_samples > actual_min_k:
                score = silhouette_score(vectors, labels)
            else:
                score = 0.0
            return actual_min_k, score, labels
        
        best_k = actual_min_k
        best_score = -1.0
        best_labels = None
        
        for k in range(actual_min_k, actual_max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = kmeans.fit_predict(vectors)
            
            # Silhouette score requires at least 2 clusters with at least 1 sample each
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                continue
                
            score = silhouette_score(vectors, labels)
            
            logger.debug(f"K={k}, Silhouette Score={score:.4f}")
            
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels.copy()
        
        if best_labels is None:
            # Fallback: just use min_k
            kmeans = KMeans(n_clusters=actual_min_k, random_state=random_state, n_init=10)
            best_labels = kmeans.fit_predict(vectors)
            best_score = 0.0
        
        return best_k, best_score, best_labels
    
    def identify_majority_cluster(self, labels: np.ndarray) -> int:
        """
        Identify the majority cluster (consensus truth).
        
        Args:
            labels: Cluster labels for each document
            
        Returns:
            Label of the majority cluster
        """
        unique, counts = np.unique(labels, return_counts=True)
        majority_idx = np.argmax(counts)
        return unique[majority_idx]
    
    def filter(
        self,
        documents: List[Dict[str, Any]],
        query: Optional[str] = None,
        text_key: str = "text"
    ) -> FilterResult:
        """
        Apply Tier 1 filtering to retrieved documents.
        
        Args:
            documents: List of document dictionaries
            query: Optional query string (for logging/debugging)
            text_key: Key to access document text in dictionary
            
        Returns:
            FilterResult containing passed and dropped documents
        """
        n_docs = len(documents)
        
        # Edge case: too few documents to filter
        if n_docs <= 2:
            logger.info(f"Too few documents ({n_docs}) for Tier 1 filtering, passing all")
            return FilterResult(
                passed_docs=documents,
                dropped_docs=[],
                passed_indices=list(range(n_docs)),
                dropped_indices=[],
                optimal_k=1,
                silhouette_score=0.0,
                cluster_labels=np.zeros(n_docs, dtype=int),
                majority_cluster=0,
                filtering_applied=False,
                reason="Too few documents for clustering"
            )
        
        # Step 1: Extract text from documents
        texts = [doc[text_key] for doc in documents]
        
        # Step 2: Vectorization
        logger.info(f"Encoding {n_docs} documents...")
        embeddings = self.encode_documents(texts)
        
        # Step 3: L2 Normalization
        normalized_embeddings = self.l2_normalize(embeddings)
        
        # Step 4: Dynamic K-Selection using Silhouette Analysis
        optimal_k, sil_score, labels = self.find_optimal_k(normalized_embeddings)
        
        logger.info(f"Optimal K={optimal_k}, Silhouette Score={sil_score:.4f}")
        
        # Check if we should apply filtering
        if sil_score < self.silhouette_threshold:
            logger.info(f"Silhouette score ({sil_score:.4f}) below threshold ({self.silhouette_threshold}), "
                       "documents appear homogeneous, passing all")
            return FilterResult(
                passed_docs=documents,
                dropped_docs=[],
                passed_indices=list(range(n_docs)),
                dropped_indices=[],
                optimal_k=optimal_k,
                silhouette_score=sil_score,
                cluster_labels=labels,
                majority_cluster=0,
                filtering_applied=False,
                reason="Low silhouette score - documents are homogeneous"
            )
        
        # Step 5: Identify majority cluster and filter
        majority_cluster = self.identify_majority_cluster(labels)
        
        passed_indices = []
        dropped_indices = []
        passed_docs = []
        dropped_docs = []
        
        for i, (doc, label) in enumerate(zip(documents, labels)):
            if label == majority_cluster:
                passed_indices.append(i)
                passed_docs.append(doc)
            else:
                dropped_indices.append(i)
                dropped_docs.append(doc)
        
        # Safety check: ensure we keep minimum ratio of documents
        passed_ratio = len(passed_docs) / n_docs
        if passed_ratio < self.min_cluster_ratio:
            logger.warning(f"Majority cluster too small ({passed_ratio:.2%}), "
                          "reverting to pass all documents")
            return FilterResult(
                passed_docs=documents,
                dropped_docs=[],
                passed_indices=list(range(n_docs)),
                dropped_indices=[],
                optimal_k=optimal_k,
                silhouette_score=sil_score,
                cluster_labels=labels,
                majority_cluster=majority_cluster,
                filtering_applied=False,
                reason="Majority cluster too small"
            )
        
        logger.info(f"Tier 1 Filtering: Passed {len(passed_docs)}/{n_docs} documents, "
                   f"Dropped {len(dropped_docs)} outliers")
        
        return FilterResult(
            passed_docs=passed_docs,
            dropped_docs=dropped_docs,
            passed_indices=passed_indices,
            dropped_indices=dropped_indices,
            optimal_k=optimal_k,
            silhouette_score=sil_score,
            cluster_labels=labels,
            majority_cluster=majority_cluster,
            filtering_applied=True,
            reason="Outlier documents detected and filtered"
        )
    
    def get_cluster_analysis(
        self,
        documents: List[Dict[str, Any]],
        text_key: str = "text"
    ) -> Dict[str, Any]:
        """
        Get detailed cluster analysis for visualization.
        
        Args:
            documents: List of document dictionaries
            text_key: Key to access document text
            
        Returns:
            Dictionary containing detailed analysis
        """
        texts = [doc[text_key] for doc in documents]
        embeddings = self.encode_documents(texts)
        normalized_embeddings = self.l2_normalize(embeddings)
        
        # Get clustering results for different K values
        results = {}
        for k in range(self.min_k, self.max_k + 1):
            if k >= len(documents):
                continue
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(normalized_embeddings)
            
            if len(np.unique(labels)) >= 2:
                score = silhouette_score(normalized_embeddings, labels)
            else:
                score = 0.0
                
            results[f"k_{k}"] = {
                "labels": labels.tolist(),
                "silhouette_score": score,
                "cluster_sizes": dict(zip(*np.unique(labels, return_counts=True)))
            }
        
        return {
            "n_documents": len(documents),
            "embedding_dim": embeddings.shape[1],
            "clustering_results": results,
            "embeddings": normalized_embeddings.tolist()
        }


# Convenience function for quick filtering
def filter_documents(
    documents: List[Dict[str, Any]],
    query: Optional[str] = None,
    text_key: str = "text",
    **kwargs
) -> FilterResult:
    """
    Quick function to filter documents using Tier 1.
    
    Args:
        documents: List of document dictionaries
        query: Optional query string
        text_key: Key to access document text
        **kwargs: Additional arguments for Tier1Filter
        
    Returns:
        FilterResult
    """
    filter_instance = Tier1Filter(**kwargs)
    return filter_instance.filter(documents, query, text_key)

if __name__ == "__main__":
    docs = [
        # AI cluster
        {"text": "Machine learning models are trained on datasets"},
        {"text": "Deep learning uses neural networks"},
        {"text": "Transformers are used in NLP tasks"},
        {"text": "PyTorch is a deep learning framework"},
        {"text": "Artificial intelligence automates tasks"},
        {"text": "Neural networks learn patterns"},

        # very different cluster
        {"text": "Cells contain DNA and proteins"},
        {"text": "Photosynthesis occurs in plants"},

        # random/noise
        {"text": "@@@ ### $$$ %%%"},
        {"text": "!!!! ???? ////"}
    ]

    print("\n🚀 Running Tier 1...\n")

    t1 = Tier1Filter(device="cpu", silhouette_threshold=0.1)
    result = t1.filter(docs)

    print("\n✅ PASSED:\n")
    for d in result.passed_docs:
        print("-", d["text"])

    print("\n❌ DROPPED:\n")
    for d in result.dropped_docs:
        print("-", d["text"])

    print("\n📊 STATS:")
    print("K:", result.optimal_k)
    print("Score:", result.silhouette_score)
    print("Applied:", result.filtering_applied)
    print("Reason:", result.reason)