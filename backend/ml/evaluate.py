"""Evaluation metrics for LaTeX generation: BLEU score, exact-match accuracy, CER/WER. (Pranav)"""

import math
from collections import Counter


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def bleu_score(predictions: list[str], references: list[str], max_n: int = 4) -> float:
    """Compute corpus-level BLEU score between predicted and reference LaTeX strings."""
    match_counts = [0] * max_n
    total_counts = [0] * max_n
    pred_len_total = 0
    ref_len_total = 0

    for prediction, reference in zip(predictions, references):
        pred_tokens = prediction.split()
        ref_tokens = reference.split()
        pred_len_total += len(pred_tokens)
        ref_len_total += len(ref_tokens)

        for n in range(1, max_n + 1):
            pred_ngrams = _ngrams(pred_tokens, n)
            ref_ngrams = _ngrams(ref_tokens, n)
            match_counts[n - 1] += sum((pred_ngrams & ref_ngrams).values())
            total_counts[n - 1] += sum(pred_ngrams.values())

    if pred_len_total == 0:
        return 0.0

    # Epsilon-smooth zero-count precisions rather than collapsing the whole score to 0 —
    # formulas are short enough that higher-order n-grams often have no matches at all.
    precisions = [
        (match_counts[n] / total_counts[n]) if total_counts[n] > 0 else 0.0 for n in range(max_n)
    ]
    precisions = [p if p > 0 else 1e-9 for p in precisions]

    log_precision_mean = sum(math.log(p) for p in precisions) / max_n
    brevity_penalty = (
        1.0 if pred_len_total > ref_len_total else math.exp(1 - ref_len_total / max(pred_len_total, 1))
    )

    return brevity_penalty * math.exp(log_precision_mean)


def exact_match_accuracy(predictions: list[str], references: list[str]) -> float:
    """Compute the fraction of predictions that exactly match their reference string."""
    if not predictions:
        return 0.0
    matches = sum(p.strip() == r.strip() for p, r in zip(predictions, references))
    return matches / len(predictions)


def _edit_distance(a: list, b: list) -> int:
    """Levenshtein distance between two sequences."""
    prev = list(range(len(b) + 1))
    for i, a_item in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, b_item in enumerate(b, start=1):
            cost = 0 if a_item == b_item else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def character_error_rate(predictions: list[str], references: list[str]) -> float:
    """Compute the average character error rate (CER) between predictions and references."""
    total_distance = 0
    total_chars = 0
    for prediction, reference in zip(predictions, references):
        total_distance += _edit_distance(list(prediction), list(reference))
        total_chars += len(reference)
    return total_distance / total_chars if total_chars > 0 else 0.0


def word_error_rate(predictions: list[str], references: list[str]) -> float:
    """Compute the average word error rate (WER) between predictions and references."""
    total_distance = 0
    total_words = 0
    for prediction, reference in zip(predictions, references):
        pred_words = prediction.split()
        ref_words = reference.split()
        total_distance += _edit_distance(pred_words, ref_words)
        total_words += len(ref_words)
    return total_distance / total_words if total_words > 0 else 0.0
