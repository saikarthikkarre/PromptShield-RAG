"""
PromptShiels: Tier 2 - Latent Activation Analysis
=================================================
The "Brain Scan" - Detects stealthy attacks by analyzing the LLM's
internal state using Latent Linear Probing (Based on RevPRAG).

Algorithm Steps:
1. Input: Feed documents into the Victim LLM (Llama-3-8B)
2. Forward Hook: Attach probes to reasoning layers (12-15)
3. Extraction: Capture activation vectors at last token position
4. Classification: Use trained Logistic Regression to classify
   - Safe Pattern: Model processing facts → Pass
   - Poison Pattern: Model processing coercive instructions → Block
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import pickle
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ActivationResult:
    """Result of activation extraction."""
    activations: torch.Tensor  # Shape: (batch_size, hidden_dim)
    layer_activations: Dict[int, torch.Tensor]  # Per-layer activations
    token_positions: List[int]  # Position of extracted tokens


@dataclass 
class ClassificationResult:
    """Result of Tier 2 classification."""
    is_safe: bool
    confidence: float
    prediction: int  # 0 = safe, 1 = poisoned
    activation_norm: float
    details: Dict[str, Any]


@dataclass
class Tier2Result:
    """Complete Tier 2 analysis result."""
    passed_docs: List[Dict[str, Any]]
    blocked_docs: List[Dict[str, Any]]
    passed_indices: List[int]
    blocked_indices: List[int]
    classifications: List[ClassificationResult]
    analysis_applied: bool
    reason: str


class ActivationExtractor:
    """
    Extracts hidden state activations from transformer layers.
    
    Uses PyTorch forward hooks to capture internal representations
    during inference without modifying the model.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layers: List[int],
        extraction_position: str = "last"  # "last", "first", or "mean"
    ):
        """
        Initialize the activation extractor.
        
        Args:
            model: The transformer model to probe
            target_layers: List of layer indices to extract from
            extraction_position: Which token position to extract
        """
        self.model = model
        self.target_layers = target_layers
        self.extraction_position = extraction_position
        self.hooks: List[Any] = []
        self.activations: Dict[int, torch.Tensor] = {}
        
    def _get_hook(self, layer_idx: int) -> Callable:
        """Create a forward hook for a specific layer."""
        def hook(module, input, output):
            # Handle different output formats
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            self.activations[layer_idx] = hidden_states.detach()
        return hook
    
    def register_hooks(self) -> None:
        """Register forward hooks on target layers."""
        self.clear_hooks()
        
        # Navigate to the model's layers
        # This works for Llama-style models
        if hasattr(self.model, 'model'):
            layers = self.model.model.layers
        elif hasattr(self.model, 'transformer'):
            layers = self.model.transformer.h
        elif hasattr(self.model, 'layers'):
            layers = self.model.layers
        else:
            raise ValueError("Cannot find model layers. Unsupported architecture.")
        
        for layer_idx in self.target_layers:
            if layer_idx < len(layers):
                hook = layers[layer_idx].register_forward_hook(self._get_hook(layer_idx))
                self.hooks.append(hook)
                logger.debug(f"Registered hook on layer {layer_idx}")
            else:
                logger.warning(f"Layer {layer_idx} not found in model with {len(layers)} layers")
    
    def clear_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}
    
    def extract(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> ActivationResult:
        """
        Extract activations from a forward pass.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            
        Returns:
            ActivationResult containing extracted activations
        """
        self.activations = {}
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Determine extraction positions
        batch_size = input_ids.shape[0]
        if attention_mask is not None:
            # Get last non-padding position for each sequence
            seq_lengths = attention_mask.sum(dim=1)
            token_positions = (seq_lengths - 1).tolist()
        else:
            # Use last position
            token_positions = [input_ids.shape[1] - 1] * batch_size
        
        # Extract activations at specified position
        layer_activations = {}
        extracted_activations = []
        
        for layer_idx, hidden_states in self.activations.items():
            # hidden_states shape: (batch_size, seq_len, hidden_dim)
            if self.extraction_position == "last":
                # Extract at last token position
                batch_extracts = []
                for b in range(batch_size):
                    pos = token_positions[b]
                    batch_extracts.append(hidden_states[b, pos, :])
                layer_act = torch.stack(batch_extracts)
            elif self.extraction_position == "first":
                layer_act = hidden_states[:, 0, :]
            else:  # mean
                layer_act = hidden_states.mean(dim=1)
            
            layer_activations[layer_idx] = layer_act
            extracted_activations.append(layer_act)
        
        # Concatenate or average activations from multiple layers
        if extracted_activations:
            # Average across layers for final activation
            combined = torch.stack(extracted_activations).mean(dim=0)
        else:
            combined = torch.zeros(batch_size, 1)
        
        return ActivationResult(
            activations=combined,
            layer_activations=layer_activations,
            token_positions=token_positions
        )


class PoisonClassifier:
    """
    Logistic Regression classifier for detecting poisoned activations.
    
    Trained on activation patterns from clean vs poisoned documents.
    """
    
    def __init__(self, input_dim: int = 4096, confidence_threshold: float = 0.7):
        """
        Initialize the classifier.
        
        Args:
            input_dim: Dimension of activation vectors
            confidence_threshold: Minimum confidence for classification
        """
        self.input_dim = input_dim
        self.confidence_threshold = confidence_threshold
        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[float] = None
        self.is_trained = False
        
    def train(
        self,
        activations: np.ndarray,
        labels: np.ndarray,
        regularization: float = 1.0
    ) -> Dict[str, float]:
        """
        Train the logistic regression classifier.
        
        Args:
            activations: Training activations (n_samples, hidden_dim)
            labels: Binary labels (0=safe, 1=poisoned)
            regularization: L2 regularization strength
            
        Returns:
            Training metrics dictionary
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # Split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            activations, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Train logistic regression
        clf = LogisticRegression(
            C=regularization,
            max_iter=1000,
            random_state=42,
            solver='lbfgs'
        )
        clf.fit(X_train, y_train)
        
        # Store weights
        self.weights = clf.coef_[0]
        self.bias = clf.intercept_[0]
        self.is_trained = True
        self._sklearn_model = clf
        
        # Evaluate on validation set
        val_preds = clf.predict(X_val)
        
        metrics = {
            "accuracy": accuracy_score(y_val, val_preds),
            "precision": precision_score(y_val, val_preds),
            "recall": recall_score(y_val, val_preds),
            "f1": f1_score(y_val, val_preds),
            "train_size": len(X_train),
            "val_size": len(X_val)
        }
        
        logger.info(f"Training complete. Validation Accuracy: {metrics['accuracy']:.4f}")
        
        return metrics
    
    def predict(self, activations: np.ndarray) -> List[ClassificationResult]:
        """
        Predict whether activations indicate poisoning.
        
        Args:
            activations: Activation vectors (n_samples, hidden_dim)
            
        Returns:
            List of ClassificationResult for each sample
        """
        if not self.is_trained:
            raise ValueError("Classifier not trained. Call train() first.")
        
        # Get probabilities
        probs = self._sklearn_model.predict_proba(activations)
        predictions = self._sklearn_model.predict(activations)
        
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probs)):
            confidence = prob[pred]  # Confidence in the predicted class
            poison_prob = prob[1]  # Probability of being poisoned
            
            # Apply confidence threshold
            if confidence < self.confidence_threshold:
                # Low confidence - default to safe to avoid false positives
                is_safe = True
                final_pred = 0
            else:
                is_safe = (pred == 0)
                final_pred = pred
            
            results.append(ClassificationResult(
                is_safe=is_safe,
                confidence=confidence,
                prediction=final_pred,
                activation_norm=np.linalg.norm(activations[i]),
                details={
                    "poison_probability": poison_prob,
                    "safe_probability": prob[0],
                    "raw_prediction": int(pred)
                }
            ))
        
        return results
    
    def save(self, path: str) -> None:
        """Save classifier to disk."""
        save_dict = {
            "weights": self.weights,
            "bias": self.bias,
            "input_dim": self.input_dim,
            "confidence_threshold": self.confidence_threshold,
            "sklearn_model": self._sklearn_model if hasattr(self, '_sklearn_model') else None
        }
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
        logger.info(f"Classifier saved to {path}")
    
    def load(self, path: str) -> None:
        """Load classifier from disk."""
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
        
        self.weights = save_dict["weights"]
        self.bias = save_dict["bias"]
        self.input_dim = save_dict["input_dim"]
        self.confidence_threshold = save_dict["confidence_threshold"]
        self._sklearn_model = save_dict["sklearn_model"]
        self.is_trained = True
        logger.info(f"Classifier loaded from {path}")


class Tier2Analyzer:
    """
    Tier 2: Latent Activation Analysis
    
    Complete system for detecting stealthy prompt injections
    by analyzing LLM internal representations.
    """
    
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
        target_layers: List[int] = [12, 13, 14, 15],
        classifier_path: Optional[str] = None,
        confidence_threshold: float = 0.7,
        device: str = "cuda"
    ):
        """
        Initialize Tier 2 Analyzer.
        
        Args:
            model: The Llama model to analyze
            tokenizer: The tokenizer for the model
            target_layers: Layers to probe (default: Llama-3 reasoning layers)
            classifier_path: Path to pre-trained classifier
            confidence_threshold: Classification confidence threshold
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.target_layers = target_layers
        self.device = device
        self.confidence_threshold = confidence_threshold
        
        # Initialize components
        self.extractor: Optional[ActivationExtractor] = None
        self.classifier = PoisonClassifier(
            input_dim=4096,  # Llama-3-8B hidden dim
            confidence_threshold=confidence_threshold
        )
        
        if classifier_path and Path(classifier_path).exists():
            self.classifier.load(classifier_path)
            logger.info("Loaded pre-trained classifier")
        
        if model is not None:
            self._setup_extractor()
    
    def _setup_extractor(self) -> None:
        """Set up activation extractor with hooks."""
        self.extractor = ActivationExtractor(
            model=self.model,
            target_layers=self.target_layers,
            extraction_position="last"
        )
        self.extractor.register_hooks()
    
    def set_model(self, model: nn.Module, tokenizer: Any) -> None:
        """
        Set or update the model and tokenizer.
        
        Args:
            model: Transformer model
            tokenizer: Associated tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        self._setup_extractor()
    
    def extract_activations(
        self,
        texts: List[str],
        query: Optional[str] = None,
        max_length: int = 512
    ) -> np.ndarray:
        """
        Extract activations for given texts.
        
        Args:
            texts: List of document texts
            query: Optional query to prepend
            max_length: Maximum sequence length
            
        Returns:
            Activation matrix (n_texts, hidden_dim)
        """
        if self.extractor is None:
            raise ValueError("Model not set. Call set_model() first.")
        
        all_activations = []
        
        for text in texts:
            # Format input
            if query:
                input_text = f"Query: {query}\nDocument: {text}"
            else:
                input_text = text
            
            # Tokenize
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Extract activations
            result = self.extractor.extract(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            
            all_activations.append(result.activations.cpu().numpy())
        
        return np.vstack(all_activations)
    
    def analyze(
        self,
        documents: List[Dict[str, Any]],
        query: Optional[str] = None,
        text_key: str = "text"
    ) -> Tier2Result:
        """
        Analyze documents for potential prompt injections.
        
        Args:
            documents: List of document dictionaries
            query: Optional query for context
            text_key: Key to access document text
            
        Returns:
            Tier2Result with analysis details
        """
        if not documents:
            return Tier2Result(
                passed_docs=[],
                blocked_docs=[],
                passed_indices=[],
                blocked_indices=[],
                classifications=[],
                analysis_applied=False,
                reason="No documents to analyze"
            )
        
        if not self.classifier.is_trained:
            logger.warning("Classifier not trained. Passing all documents.")
            return Tier2Result(
                passed_docs=documents,
                blocked_docs=[],
                passed_indices=list(range(len(documents))),
                blocked_indices=[],
                classifications=[],
                analysis_applied=False,
                reason="Classifier not trained"
            )
        
        # Extract texts
        texts = [doc[text_key] for doc in documents]
        
        # Extract activations
        logger.info(f"Extracting activations for {len(texts)} documents...")
        activations = self.extract_activations(texts, query)
        
        # Classify
        logger.info("Running poison classification...")
        classifications = self.classifier.predict(activations)
        
        # Separate passed and blocked
        passed_docs = []
        blocked_docs = []
        passed_indices = []
        blocked_indices = []
        
        for i, (doc, clf_result) in enumerate(zip(documents, classifications)):
            if clf_result.is_safe:
                passed_docs.append(doc)
                passed_indices.append(i)
            else:
                blocked_docs.append(doc)
                blocked_indices.append(i)
        
        logger.info(f"Tier 2 Analysis: Passed {len(passed_docs)}/{len(documents)}, "
                   f"Blocked {len(blocked_docs)}")
        
        return Tier2Result(
            passed_docs=passed_docs,
            blocked_docs=blocked_docs,
            passed_indices=passed_indices,
            blocked_indices=blocked_indices,
            classifications=classifications,
            analysis_applied=True,
            reason="Activation analysis complete"
        )
    
    def train_classifier(
        self,
        clean_texts: List[str],
        poison_texts: List[str],
        query: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Train the classifier on labeled data.
        
        Args:
            clean_texts: List of clean document texts
            poison_texts: List of poisoned document texts
            query: Optional query context
            save_path: Path to save trained classifier
            
        Returns:
            Training metrics
        """
        logger.info("Extracting activations for training...")
        
        # Extract activations for both classes
        clean_activations = self.extract_activations(clean_texts, query)
        poison_activations = self.extract_activations(poison_texts, query)
        
        # Combine data
        all_activations = np.vstack([clean_activations, poison_activations])
        labels = np.array([0] * len(clean_texts) + [1] * len(poison_texts))
        
        # Train
        metrics = self.classifier.train(all_activations, labels)
        
        if save_path:
            self.classifier.save(save_path)
        
        return metrics
    
    def cleanup(self) -> None:
        """Clean up resources (remove hooks)."""
        if self.extractor:
            self.extractor.clear_hooks()
