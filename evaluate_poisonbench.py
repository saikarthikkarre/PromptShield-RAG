"""
PromptShiels: PoisonBench Evaluation
=====================================
Evaluates PromptShiels on the official PoisonBench benchmark.

Usage:
    python evaluate_poisonbench.py --file path/to/poisonbench.json
    python evaluate_poisonbench.py --huggingface  # Load from HuggingFace
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.tier1 import Tier1Filter, FilterResult


def load_poisonbench_from_file(file_path: str) -> List[Dict]:
    """Load PoisonBench from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different formats
    if isinstance(data, dict):
        if 'data' in data:
            samples = data['data']
        elif 'samples' in data:
            samples = data['samples']
        elif 'test' in data:
            samples = data['test']
        else:
            samples = list(data.values())[0] if data else []
    elif isinstance(data, list):
        samples = data
    else:
        samples = []
    
    print(f"✅ Loaded {len(samples)} samples from {file_path}")
    return samples


def load_poisonbench_from_huggingface() -> List[Dict]:
    """Load PoisonBench from HuggingFace."""
    try:
        from datasets import load_dataset
        print("Loading PoisonBench from HuggingFace...")
        dataset = load_dataset("TrustAIRLab/PoisonBench", split="test")
        samples = list(dataset)
        print(f"✅ Loaded {len(samples)} samples from HuggingFace")
        return samples
    except Exception as e:
        print(f"❌ Failed to load from HuggingFace: {e}")
        return []


def analyze_poisonbench_structure(samples: List[Dict]):
    """Analyze and display the structure of PoisonBench samples."""
    if not samples:
        print("❌ No samples to analyze")
        return
    
    print("\n📊 PoisonBench Structure Analysis")
    print("="*50)
    
    # Get first sample to show structure
    sample = samples[0]
    print(f"\nSample keys: {list(sample.keys())}")
    print(f"\nSample structure:")
    for key, value in sample.items():
        if isinstance(value, str):
            print(f"  {key}: (str) '{value[:100]}...' " if len(str(value)) > 100 else f"  {key}: (str) '{value}'")
        elif isinstance(value, list):
            print(f"  {key}: (list) {len(value)} items")
            if value:
                print(f"    First item type: {type(value[0])}")
        elif isinstance(value, dict):
            print(f"  {key}: (dict) {list(value.keys())}")
        else:
            print(f"  {key}: ({type(value).__name__}) {value}")
    
    # Count labels if present
    labels = [s.get('label', s.get('is_poison', None)) for s in samples]
    if labels[0] is not None:
        poison_count = sum(1 for l in labels if l == 1 or l == True or l == 'poison')
        clean_count = sum(1 for l in labels if l == 0 or l == False or l == 'clean')
        print(f"\n📈 Label distribution: {clean_count} clean, {poison_count} poison")


def convert_to_standard_format(samples: List[Dict]) -> List[Dict]:
    """
    Convert PoisonBench samples to our standard format:
    {
        "id": str,
        "query": str,
        "clean_text": str,
        "poison_text": str,
        "label": int (0=clean, 1=poison),
        "target_output": str
    }
    """
    converted = []
    
    for i, sample in enumerate(samples):
        converted_sample = {
            "id": sample.get('id', sample.get('idx', f'sample_{i}')),
            "query": sample.get('query', sample.get('question', sample.get('instruction', ''))),
            "clean_text": sample.get('clean_text', sample.get('context', sample.get('original', ''))),
            "poison_text": sample.get('poison_text', sample.get('poisoned_context', sample.get('perturbed', ''))),
            "label": 1 if sample.get('label', sample.get('is_poison', 0)) in [1, True, 'poison'] else 0,
            "target_output": sample.get('target_output', sample.get('target', sample.get('malicious_output', '')))
        }
        
        # Handle case where there's a "documents" list
        if 'documents' in sample:
            docs = sample['documents']
            labels = sample.get('labels', [0] * len(docs))
            for j, (doc, label) in enumerate(zip(docs, labels)):
                doc_sample = converted_sample.copy()
                doc_sample['id'] = f"{converted_sample['id']}_{j}"
                doc_sample['clean_text'] = doc if label == 0 else ''
                doc_sample['poison_text'] = doc if label == 1 else ''
                doc_sample['label'] = label
                converted.append(doc_sample)
            continue
        
        converted.append(converted_sample)
    
    return converted


def evaluate_tier1(samples: List[Dict], batch_size: int = 10, num_batches: int = 10) -> Dict[str, Any]:
    """
    Evaluate Tier 1 on PoisonBench samples.
    Creates batches mixing clean and poison documents.
    """
    print("\n" + "="*60)
    print("🎯 TIER 1 EVALUATION: Dynamic Outlier Filtration")
    print("="*60)
    
    # Initialize Tier 1
    tier1 = Tier1Filter(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        min_k=2,
        max_k=3,
        silhouette_threshold=0.3,
        device="cpu"
    )
    
    # Separate clean and poison samples
    clean_samples = [s for s in samples if s.get('label', 0) == 0]
    poison_samples = [s for s in samples if s.get('label', 0) == 1]
    
    print(f"\nDataset: {len(clean_samples)} clean, {len(poison_samples)} poison")
    
    all_tp, all_fp, all_tn, all_fn = 0, 0, 0, 0
    filtering_applied_count = 0
    total_latency = 0
    
    np.random.seed(42)
    
    for batch_idx in range(num_batches):
        # Create mixed batch
        n_clean = batch_size // 2
        n_poison = batch_size - n_clean
        
        batch_clean = [clean_samples[i] for i in np.random.choice(len(clean_samples), min(n_clean, len(clean_samples)), replace=False)]
        batch_poison = [poison_samples[i] for i in np.random.choice(len(poison_samples), min(n_poison, len(poison_samples)), replace=False)]
        
        documents = []
        labels = []
        
        for s in batch_clean:
            text = s.get('clean_text') or s.get('text', '')
            if text:
                documents.append({"id": s['id'], "text": text, "source": "Clean"})
                labels.append(0)
        
        for s in batch_poison:
            text = s.get('poison_text') or s.get('text', '')
            if text:
                documents.append({"id": s['id'], "text": text, "source": "Poison"})
                labels.append(1)
        
        if len(documents) < 3:
            continue
        
        # Shuffle
        combined = list(zip(documents, labels))
        np.random.shuffle(combined)
        documents, labels = zip(*combined)
        documents, labels = list(documents), list(labels)
        
        # Run Tier 1
        start_time = time.time()
        result = tier1.filter(documents, text_key="text")
        total_latency += time.time() - start_time
        
        if result.filtering_applied:
            filtering_applied_count += 1
        
        # Calculate metrics
        dropped_indices = set(result.dropped_indices)
        passed_indices = set(result.passed_indices)
        
        all_tp += sum(1 for i in dropped_indices if labels[i] == 1)
        all_fp += sum(1 for i in dropped_indices if labels[i] == 0)
        all_tn += sum(1 for i in passed_indices if labels[i] == 0)
        all_fn += sum(1 for i in passed_indices if labels[i] == 1)
    
    # Overall metrics
    total = all_tp + all_fp + all_tn + all_fn
    accuracy = (all_tp + all_tn) / total if total > 0 else 0
    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n📈 Results over {num_batches} batches:")
    print(f"   Filtering Applied: {filtering_applied_count}/{num_batches} batches")
    print(f"   Total Documents Evaluated: {total}")
    print(f"\n🎯 Detection Performance:")
    print(f"   True Positives: {all_tp}")
    print(f"   False Positives: {all_fp}")
    print(f"   True Negatives: {all_tn}")
    print(f"   False Negatives: {all_fn}")
    print(f"\n📉 Metrics:")
    print(f"   Accuracy: {accuracy:.2%}")
    print(f"   Precision: {precision:.2%}")
    print(f"   Recall: {recall:.2%}")
    print(f"   F1 Score: {f1:.2%}")
    print(f"   Avg Latency: {total_latency/num_batches*1000:.2f} ms/batch")
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "filtering_applied_rate": filtering_applied_count / num_batches,
        "avg_latency_ms": total_latency / num_batches * 1000
    }


def evaluate_tier2_simulation(samples: List[Dict], batch_size: int = 100) -> Dict[str, Any]:
    """
    Evaluate Tier 2 (simulated) on PoisonBench samples.
    """
    print("\n" + "="*60)
    print("🧠 TIER 2 EVALUATION: Activation Analysis (Simulated)")
    print("="*60)
    
    poison_indicators = [
        "system", "override", "ignore", "admin", "bypass",
        "security", "delete", "password", "urgent", "deprecated",
        "previous instructions", "reveal", "confidential", 
        "assistant\n", "user\n", "<!--", "-->",
        "cannot provide", "i cannot", "safety guidelines",
        "disregard", "forget", "new instructions", "important update"
    ]
    
    all_tp, all_fp, all_tn, all_fn = 0, 0, 0, 0
    
    for sample in samples[:batch_size]:
        label = sample.get('label', 0)
        text = (sample.get('poison_text') if label == 1 else sample.get('clean_text')) or sample.get('text', '')
        text_lower = text.lower()
        
        # Count indicators
        indicator_count = sum(1 for ind in poison_indicators if ind.lower() in text_lower)
        poison_prob = min(indicator_count * 0.15, 0.95)
        prediction = 1 if poison_prob >= 0.5 else 0
        
        if prediction == 1 and label == 1:
            all_tp += 1
        elif prediction == 1 and label == 0:
            all_fp += 1
        elif prediction == 0 and label == 0:
            all_tn += 1
        else:
            all_fn += 1
    
    total = all_tp + all_fp + all_tn + all_fn
    accuracy = (all_tp + all_tn) / total if total > 0 else 0
    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n📈 Results on {total} samples:")
    print(f"   True Positives: {all_tp}")
    print(f"   False Positives: {all_fp}")
    print(f"   True Negatives: {all_tn}")
    print(f"   False Negatives: {all_fn}")
    print(f"\n📉 Metrics:")
    print(f"   Accuracy: {accuracy:.2%}")
    print(f"   Precision: {precision:.2%}")
    print(f"   Recall: {recall:.2%}")
    print(f"   F1 Score: {f1:.2%}")
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate PromptShiels on PoisonBench")
    parser.add_argument("--file", type=str, help="Path to PoisonBench JSON file")
    parser.add_argument("--huggingface", action="store_true", help="Load from HuggingFace")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for evaluation")
    parser.add_argument("--num-batches", type=int, default=10, help="Number of batches")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze structure")
    
    args = parser.parse_args()
    
    # Load data
    if args.file:
        samples = load_poisonbench_from_file(args.file)
    elif args.huggingface:
        samples = load_poisonbench_from_huggingface()
    else:
        # Default: try to load from data directory
        default_paths = [
            "data/poisonbench.json",
            "data/PoisonBench.json",
            "data/poison_bench.json"
        ]
        for path in default_paths:
            if Path(path).exists():
                samples = load_poisonbench_from_file(path)
                break
        else:
            # Fall back to our super_poison_dataset
            print("No PoisonBench file found, using super_poison_dataset.json")
            samples = load_poisonbench_from_file("data/super_poison_dataset.json")
            if isinstance(samples, dict) and 'samples' in samples:
                samples = samples['samples']
    
    if not samples:
        print("❌ No data loaded. Please provide a file with --file or use --huggingface")
        return
    
    # Analyze structure
    analyze_poisonbench_structure(samples)
    
    if args.analyze_only:
        return
    
    # Convert to standard format if needed
    if samples and 'label' not in samples[0] and 'is_poison' not in samples[0]:
        print("\n🔄 Converting to standard format...")
        samples = convert_to_standard_format(samples)
    
    # Run evaluations
    print("\n" + "="*70)
    print("🛡️  PROMPTSHIELS POISONBENCH EVALUATION")
    print("="*70)
    
    tier1_metrics = evaluate_tier1(samples, args.batch_size, args.num_batches)
    tier2_metrics = evaluate_tier2_simulation(samples, args.batch_size * args.num_batches)
    
    # Summary
    print("\n" + "="*70)
    print("📋 EVALUATION SUMMARY")
    print("="*70)
    print(f"\n{'Metric':<20} {'Tier 1':<15} {'Tier 2 (Sim)':<15}")
    print("-" * 50)
    print(f"{'Accuracy':<20} {tier1_metrics['accuracy']:>12.2%} {tier2_metrics['accuracy']:>12.2%}")
    print(f"{'Precision':<20} {tier1_metrics['precision']:>12.2%} {tier2_metrics['precision']:>12.2%}")
    print(f"{'Recall':<20} {tier1_metrics['recall']:>12.2%} {tier2_metrics['recall']:>12.2%}")
    print(f"{'F1 Score':<20} {tier1_metrics['f1_score']:>12.2%} {tier2_metrics['f1_score']:>12.2%}")
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
