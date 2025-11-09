"""Utility functions for evaluation metrics."""
import numpy as np
from typing import List, Dict, Tuple


def calculate_precision_recall_f1(
    true_positives: int,
    false_positives: int,
    false_negatives: int
) -> Dict[str, float]:
    """Calculate precision, recall, and F1 score."""
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate (WER) between reference and hypothesis."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    # Dynamic programming for edit distance
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
    
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
        
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion = d[i][j-1] + 1
                deletion = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
    
    wer = d[len(ref_words)][len(hyp_words)] / len(ref_words) if len(ref_words) > 0 else 0.0
    return wer


def calculate_correlation(x: List[float], y: List[float]) -> float:
    """Calculate Pearson correlation coefficient."""
    if len(x) != len(y) or len(x) == 0:
        return 0.0
    
    return np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0.0


def normalize_score(score: float, min_val: float = 0.0, max_val: float = 10.0) -> float:
    """Normalize score to 0-10 scale."""
    return max(min_val, min(max_val, score))
