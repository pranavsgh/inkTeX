"""LaTeX tokenizer: builds a vocabulary over LaTeX formulas and encodes/decodes token sequences. (Pranav)"""


class LatexTokenizer:
    """Tokenizes LaTeX strings into integer ids using a vocabulary built from the training corpus."""

    def __init__(self, vocab_path: str | None = None) -> None:
        """Load an existing vocabulary file, or initialize an empty one to be built via build_vocab."""
        raise NotImplementedError()

    def build_vocab(self, formulas: list[str], min_freq: int = 1) -> None:
        """Build the token vocabulary from a list of LaTeX formula strings."""
        raise NotImplementedError()

    def encode(self, formula: str) -> list[int]:
        """Convert a LaTeX formula string into a list of token ids."""
        raise NotImplementedError()

    def decode(self, token_ids: list[int]) -> str:
        """Convert a list of token ids back into a LaTeX formula string."""
        raise NotImplementedError()

    def save(self, path: str) -> None:
        """Persist the vocabulary to disk."""
        raise NotImplementedError()

    @classmethod
    def load(cls, path: str) -> "LatexTokenizer":
        """Load a tokenizer with a previously saved vocabulary."""
        raise NotImplementedError()
