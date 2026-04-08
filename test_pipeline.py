"""
PromptShiels: Comprehensive Test Pipeline
==========================================
Tests both Tier 1 (Outlier Filtering) and Tier 2 (Activation Analysis)
against the super_poison_dataset and PoisonBench.
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.tier1 import Tier1Filter, FilterResult
from src.tier2 import Tier2Result, ClassificationResult


def load_super_poison_dataset(path: str = "data/super_poison_dataset.json") -> List[Dict]:
    """Load the super poison dataset."""
    full_path = Path(__file__).parent / path
    with open(full_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both formats: direct list or dict with 'samples' key
    if isinstance(data, dict):
        if 'samples' in data:
            samples = data['samples']
            metadata = data.get('metadata', {})
            print(f"✅ Loaded {len(samples)} samples from super_poison_dataset.json")
            print(f"   Metadata: {metadata.get('total', 'N/A')} total, {metadata.get('poison', 'N/A')} poison, {metadata.get('clean', 'N/A')} clean")
            return samples
        else:
            # Single sample as dict
            return [data]
    elif isinstance(data, list):
        print(f"✅ Loaded {len(data)} samples from super_poison_dataset.json")
        return data
    else:
        print(f"⚠️ Unexpected data format: {type(data)}")
        return []


def load_poison_bench(path: str = None) -> List[Dict]:
    """
    Load PoisonBench dataset.
    If path is provided, load from file. Otherwise, try Hugging Face.
    """
    if path and Path(path).exists():
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Loaded {len(data)} samples from PoisonBench file")
        return data
    
    try:
        from datasets import load_dataset
        print("Loading PoisonBench from Hugging Face...")
        dataset = load_dataset("TrustAIRLab/PoisonBench", split="test")
        return list(dataset)
    except Exception as e:
        print(f"⚠️ Could not load PoisonBench: {e}")
        return []


def prepare_test_scenarios(dataset: List[Dict]) -> List[Dict]:
    """
    Prepare test scenarios from the dataset.
    Each scenario has a mix of clean and poison documents.
    """
    scenarios = []
    
    # Group samples by query
    by_query = defaultdict(list)
    for sample in dataset:
        query = sample.get('query', '')
        by_query[query].append(sample)
    
    # Create scenarios with mixed documents
    for query, samples in by_query.items():
        documents = []
        labels = []
        
        for sample in samples:
            # Add clean version
            if sample.get('clean_text'):
                documents.append({
                    "id": f"{sample['id']}_clean",
                    "text": sample['clean_text'],
                    "source": "Clean"
                })
                labels.append(0)
            
            # Add poison version (if it's a poison sample)
            if sample.get('label', 0) == 1 and sample.get('poison_text'):
                documents.append({
                    "id": f"{sample['id']}_poison",
                    "text": sample['poison_text'],
                    "source": "Poison"
                })
                labels.append(1)
        
        if len(documents) >= 3:  # Need at least 3 docs for meaningful clustering
            scenarios.append({
                "query": query,
                "documents": documents,
                "labels": labels,
                "target_output": samples[0].get('target_output', '')
            })
    
    return scenarios


def create_mixed_test_batch(dataset: List[Dict], batch_size: int = 10) -> Tuple[List[Dict], List[int]]:
    """
    Create a mixed batch of clean and poison documents for testing.
    """
    documents = []
    labels = []
    
    # Separate clean and poison samples
    clean_samples = [s for s in dataset if s.get('label', 0) == 0]
    poison_samples = [s for s in dataset if s.get('label', 0) == 1]
    
    # Select half clean, half poison
    n_clean = batch_size // 2
    n_poison = batch_size - n_clean
    
    np.random.seed(42)
    selected_clean = np.random.choice(len(clean_samples), min(n_clean, len(clean_samples)), replace=False)
    selected_poison = np.random.choice(len(poison_samples), min(n_poison, len(poison_samples)), replace=False)
    
    for idx in selected_clean:
        sample = clean_samples[idx]
        documents.append({
            "id": sample.get('id', f'clean_{idx}'),
            "text": sample.get('clean_text', sample.get('text', '')),
            "source": "Clean"
        })
        labels.append(0)
    
    for idx in selected_poison:
        sample = poison_samples[idx]
        documents.append({
            "id": sample.get('id', f'poison_{idx}'),
            "text": sample.get('poison_text', sample.get('text', '')),
            "source": "Poison"
        })
        labels.append(1)
    
    # Shuffle
    combined = list(zip(documents, labels))
    np.random.shuffle(combined)
    documents, labels = zip(*combined) if combined else ([], [])
    
    return list(documents), list(labels)


def test_tier1(documents: List[Dict], labels: List[int], verbose: bool = True) -> Dict[str, Any]:
    """
    Test Tier 1 (Outlier Filtering) on a batch of documents.
    
    Returns metrics dictionary.
    """
    print("\n" + "="*60)
    print("🎯 TIER 1 TEST: Dynamic Outlier Filtration")
    print("="*60)
    
    # Initialize Tier 1
    tier1 = Tier1Filter(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        min_k=2,
        max_k=3,
        silhouette_threshold=0.3,
        device="cpu"
    )
    
    # Run filtering
    start_time = time.time()
    result = tier1.filter(documents, text_key="text")
    elapsed = time.time() - start_time
    
    # Calculate metrics
    dropped_indices = set(result.dropped_indices)
    passed_indices = set(result.passed_indices)
    
    # True positives: poison docs correctly dropped
    # False positives: clean docs incorrectly dropped
    # True negatives: clean docs correctly passed
    # False negatives: poison docs incorrectly passed
    
    tp = sum(1 for i in dropped_indices if labels[i] == 1)  # Poison correctly dropped
    fp = sum(1 for i in dropped_indices if labels[i] == 0)  # Clean incorrectly dropped
    tn = sum(1 for i in passed_indices if labels[i] == 0)   # Clean correctly passed
    fn = sum(1 for i in passed_indices if labels[i] == 1)   # Poison incorrectly passed
    
    total = len(documents)
    total_poison = sum(labels)
    total_clean = total - total_poison
    
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        "total_documents": total,
        "total_clean": total_clean,
        "total_poison": total_poison,
        "dropped": len(dropped_indices),
        "passed": len(passed_indices),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "filtering_applied": result.filtering_applied,
        "optimal_k": result.optimal_k,
        "silhouette_score": result.silhouette_score,
        "latency_ms": elapsed * 1000
    }
    
    if verbose:
        print(f"\n📊 Clustering Analysis:")
        print(f"   Optimal K: {result.optimal_k}")
        print(f"   Silhouette Score: {result.silhouette_score:.4f}")
        print(f"   Filtering Applied: {'✅ Yes' if result.filtering_applied else '❌ No'}")
        print(f"   Reason: {result.reason}")
        
        print(f"\n📈 Results:")
        print(f"   Total Documents: {total} ({total_clean} clean, {total_poison} poison)")
        print(f"   Dropped: {len(dropped_indices)}")
        print(f"   Passed: {len(passed_indices)}")
        
        print(f"\n🎯 Detection Performance:")
        print(f"   True Positives (poison dropped): {tp}")
        print(f"   False Positives (clean dropped): {fp}")
        print(f"   True Negatives (clean passed): {tn}")
        print(f"   False Negatives (poison passed): {fn}")
        
        print(f"\n📉 Metrics:")
        print(f"   Accuracy: {accuracy:.2%}")
        print(f"   Precision: {precision:.2%}")
        print(f"   Recall: {recall:.2%}")
        print(f"   F1 Score: {f1:.2%}")
        print(f"   Latency: {elapsed*1000:.2f} ms")
    
    return metrics


def test_tier2_simulation(documents: List[Dict], labels: List[int], verbose: bool = True) -> Dict[str, Any]:
    """
    Test Tier 2 using keyword-based simulation (without GPU/LLM).
    This simulates what the activation-based classifier would do.
    
    For actual Tier 2 testing with the trained classifier, use test_tier2_full().
    """
    print("\n" + "="*60)
    print("🧠 TIER 2 TEST: Activation Analysis (Simulated)")
    print("="*60)
    
    # Poison keywords that indicate injection attempts
    poison_indicators = [
        "system", "override", "ignore", "admin", "bypass",
        "security", "delete", "password", "urgent", "deprecated",
        "previous instructions", "reveal", "confidential", 
        "assistant\n", "user\n", "<!--", "-->",
        "cannot provide", "i cannot", "safety guidelines"
    ]
    
    start_time = time.time()
    
    predictions = []
    confidences = []
    
    for doc in documents:
        text = doc.get("text", "").lower()
        
        # Count poison indicators
        indicator_count = sum(1 for ind in poison_indicators if ind.lower() in text)
        
        # Calculate poison probability
        poison_prob = min(indicator_count * 0.15, 0.95)
        
        # Prediction
        pred = 1 if poison_prob >= 0.5 else 0
        confidence = poison_prob if pred == 1 else (1 - poison_prob)
        
        predictions.append(pred)
        confidences.append(confidence)
    
    elapsed = time.time() - start_time
    
    # Calculate metrics
    tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
    tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)
    fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)
    
    total = len(documents)
    total_poison = sum(labels)
    total_clean = total - total_poison
    
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        "total_documents": total,
        "total_clean": total_clean,
        "total_poison": total_poison,
        "blocked": sum(predictions),
        "passed": total - sum(predictions),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "avg_confidence": np.mean(confidences),
        "latency_ms": elapsed * 1000
    }
    
    if verbose:
        print(f"\n📊 Classification Analysis:")
        print(f"   Mode: Keyword-based simulation")
        print(f"   Avg Confidence: {np.mean(confidences):.2%}")
        
        print(f"\n📈 Results:")
        print(f"   Total Documents: {total} ({total_clean} clean, {total_poison} poison)")
        print(f"   Blocked: {sum(predictions)}")
        print(f"   Passed: {total - sum(predictions)}")
        
        print(f"\n🎯 Detection Performance:")
        print(f"   True Positives (poison blocked): {tp}")
        print(f"   False Positives (clean blocked): {fp}")
        print(f"   True Negatives (clean passed): {tn}")
        print(f"   False Negatives (poison passed): {fn}")
        
        print(f"\n📉 Metrics:")
        print(f"   Accuracy: {accuracy:.2%}")
        print(f"   Precision: {precision:.2%}")
        print(f"   Recall: {recall:.2%}")
        print(f"   F1 Score: {f1:.2%}")
        print(f"   Latency: {elapsed*1000:.2f} ms")
    
    return metrics


def test_combined_pipeline(documents: List[Dict], labels: List[int], verbose: bool = True) -> Dict[str, Any]:
    """
    Test the complete Tier 1 + Tier 2 pipeline.
    """
    print("\n" + "="*60)
    print("🛡️ COMBINED PIPELINE TEST: Tier 1 → Tier 2")
    print("="*60)
    
    total = len(documents)
    total_poison = sum(labels)
    total_clean = total - total_poison
    
    # === TIER 1 ===
    tier1 = Tier1Filter(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        min_k=2,
        max_k=3,
        silhouette_threshold=0.3,
        device="cpu"
    )
    
    start_time = time.time()
    tier1_result = tier1.filter(documents, text_key="text")
    tier1_time = time.time() - start_time
    
    # Track what Tier 1 dropped
    tier1_dropped_indices = set(tier1_result.dropped_indices)
    tier1_passed_indices = tier1_result.passed_indices
    tier1_passed_docs = tier1_result.passed_docs
    tier1_passed_labels = [labels[i] for i in tier1_passed_indices]
    
    # === TIER 2 (on remaining docs) ===
    poison_indicators = [
        "system", "override", "ignore", "admin", "bypass",
        "security", "delete", "password", "urgent", "deprecated",
        "previous instructions", "reveal", "confidential", 
        "assistant\n", "user\n", "<!--", "-->",
        "cannot provide", "i cannot", "safety guidelines"
    ]
    
    start_time = time.time()
    tier2_predictions = []
    
    for doc in tier1_passed_docs:
        text = doc.get("text", "").lower()
        indicator_count = sum(1 for ind in poison_indicators if ind.lower() in text)
        poison_prob = min(indicator_count * 0.15, 0.95)
        tier2_predictions.append(1 if poison_prob >= 0.5 else 0)
    
    tier2_time = time.time() - start_time
    
    # Calculate final results
    # After Tier 2, which docs are blocked?
    tier2_blocked_indices = [tier1_passed_indices[i] for i, p in enumerate(tier2_predictions) if p == 1]
    tier2_passed_indices = [tier1_passed_indices[i] for i, p in enumerate(tier2_predictions) if p == 0]
    
    # All blocked = Tier 1 dropped + Tier 2 blocked
    all_blocked = tier1_dropped_indices | set(tier2_blocked_indices)
    all_passed = set(tier2_passed_indices)
    
    # Calculate metrics
    tp = sum(1 for i in all_blocked if labels[i] == 1)  # Poison blocked
    fp = sum(1 for i in all_blocked if labels[i] == 0)  # Clean blocked (false alarm)
    tn = sum(1 for i in all_passed if labels[i] == 0)   # Clean passed
    fn = sum(1 for i in all_passed if labels[i] == 1)   # Poison passed (missed)
    
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Attack success rate = poison that got through / total poison
    attack_success_rate = fn / total_poison if total_poison > 0 else 0
    
    metrics = {
        "total_documents": total,
        "total_clean": total_clean,
        "total_poison": total_poison,
        "tier1_dropped": len(tier1_dropped_indices),
        "tier2_blocked": len(tier2_blocked_indices),
        "total_blocked": len(all_blocked),
        "total_passed": len(all_passed),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "attack_success_rate": attack_success_rate,
        "total_latency_ms": (tier1_time + tier2_time) * 1000
    }
    
    if verbose:
        print(f"\n📊 Pipeline Flow:")
        print(f"   Input: {total} documents ({total_clean} clean, {total_poison} poison)")
        print(f"   → Tier 1 Dropped: {len(tier1_dropped_indices)}")
        print(f"   → Tier 2 Blocked: {len(tier2_blocked_indices)}")
        print(f"   → Final Output: {len(all_passed)} documents")
        
        print(f"\n🎯 Detection Performance:")
        print(f"   True Positives (attacks blocked): {tp}")
        print(f"   False Positives (clean blocked): {fp}")
        print(f"   True Negatives (clean passed): {tn}")
        print(f"   False Negatives (attacks missed): {fn}")
        
        print(f"\n📉 Overall Metrics:")
        print(f"   Accuracy: {accuracy:.2%}")
        print(f"   Precision: {precision:.2%}")
        print(f"   Recall: {recall:.2%}")
        print(f"   F1 Score: {f1:.2%}")
        print(f"   Attack Success Rate: {attack_success_rate:.2%} (lower is better)")
        print(f"   Total Latency: {(tier1_time + tier2_time)*1000:.2f} ms")
    
    return metrics


def run_comprehensive_test():
    """Run all tests."""
    print("\n" + "="*70)
    print("🛡️  PROMPTSHIELS COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    # Load dataset
    print("\n📂 Loading datasets...")
    dataset = load_super_poison_dataset()
    
    if not dataset:
        print("❌ Failed to load dataset. Please check data/super_poison_dataset.json")
        return
    
    # Create test batch
    print("\n📝 Creating test batch...")
    documents, labels = create_mixed_test_batch(dataset, batch_size=20)
    
    print(f"   Created batch with {len(documents)} documents:")
    print(f"   - Clean: {sum(1 for l in labels if l == 0)}")
    print(f"   - Poison: {sum(1 for l in labels if l == 1)}")
    
    # Test Tier 1
    tier1_metrics = test_tier1(documents, labels)
    
    # Test Tier 2 (simulation)
    tier2_metrics = test_tier2_simulation(documents, labels)
    
    # Test combined pipeline
    combined_metrics = test_combined_pipeline(documents, labels)
    
    # Summary
    print("\n" + "="*70)
    print("📋 TEST SUMMARY")
    print("="*70)
    
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│                        TEST RESULTS                             │")
    print("├─────────────────┬────────────┬────────────┬────────────────────┤")
    print("│     Metric      │   Tier 1   │   Tier 2   │     Combined       │")
    print("├─────────────────┼────────────┼────────────┼────────────────────┤")
    print(f"│ Accuracy        │  {tier1_metrics['accuracy']:>7.2%}   │  {tier2_metrics['accuracy']:>7.2%}   │     {combined_metrics['accuracy']:>7.2%}         │")
    print(f"│ Precision       │  {tier1_metrics['precision']:>7.2%}   │  {tier2_metrics['precision']:>7.2%}   │     {combined_metrics['precision']:>7.2%}         │")
    print(f"│ Recall          │  {tier1_metrics['recall']:>7.2%}   │  {tier2_metrics['recall']:>7.2%}   │     {combined_metrics['recall']:>7.2%}         │")
    print(f"│ F1 Score        │  {tier1_metrics['f1_score']:>7.2%}   │  {tier2_metrics['f1_score']:>7.2%}   │     {combined_metrics['f1_score']:>7.2%}         │")
    print(f"│ Latency (ms)    │  {tier1_metrics['latency_ms']:>7.1f}   │  {tier2_metrics['latency_ms']:>7.1f}   │     {combined_metrics['total_latency_ms']:>7.1f}         │")
    print("└─────────────────┴────────────┴────────────┴────────────────────┘")
    
    print(f"\n🛡️ Attack Success Rate: {combined_metrics['attack_success_rate']:.2%}")
    print("   (Percentage of poison documents that bypassed both tiers)")
    
    print("\n✅ Test complete!")
    
    return {
        "tier1": tier1_metrics,
        "tier2": tier2_metrics,
        "combined": combined_metrics
    }


if __name__ == "__main__":
    run_comprehensive_test()
