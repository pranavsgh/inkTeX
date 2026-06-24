"""Evaluation metrics for LaTeX generation: BLEU score, exact-match accuracy, CER/WER. (Shared)"""


def bleu_score(predictions: list[str], references: list[str]) -> float:
    """Compute corpus-level BLEU score between predicted and reference LaTeX strings."""
    raise NotImplementedError()


def exact_match_accuracy(predictions: list[str], references: list[str]) -> float:
    """Compute the fraction of predictions that exactly match their reference string."""
    raise NotImplementedError()


def character_error_rate(predictions: list[str], references: list[str]) -> float:
    """Compute the average character error rate (CER) between predictions and references."""
    raise NotImplementedError()


def word_error_rate(predictions: list[str], references: list[str]) -> float:
    """Compute the average word error rate (WER) between predictions and references."""
    raise NotImplementedError()
