"""LaTeX tokenizer: builds a vocabulary over LaTeX formulas and encodes/decodes token sequences. (Pranav)"""

import json
from collections import Counter

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


class LatexTokenizer:
    """Tokenizes LaTeX strings into integer ids using a vocabulary built from the training corpus.

    Formulas are expected to already be whitespace-tokenized (as im2latex-100k's
    normalized formulas are, e.g. "x ^ { 2 } + y ^ { 2 } = z ^ { 2 }"), so tokenization
    here is a simple whitespace split rather than BPE/command parsing.
    """

    def __init__(self, vocab_path: str | None = None) -> None:
        """Load an existing vocabulary file, or initialize an empty one to be built via build_vocab."""
        if vocab_path is not None:
            loaded = self.load(vocab_path)
            self.token_to_id = loaded.token_to_id
            self.id_to_token = loaded.id_to_token
        else:
            self.token_to_id = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
            self.id_to_token = {idx: tok for tok, idx in self.token_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def __len__(self) -> int:
        return self.vocab_size

    def build_vocab(self, formulas: list[str], min_freq: int = 1) -> None:
        """Build the token vocabulary from a list of LaTeX formula strings."""
        counts = Counter()
        for formula in formulas:
            counts.update(formula.split())

        kept_tokens = sorted(
            (tok for tok, freq in counts.items() if freq >= min_freq),
            key=lambda tok: (-counts[tok], tok),
        )

        self.token_to_id = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
        for tok in kept_tokens:
            if tok not in self.token_to_id:
                self.token_to_id[tok] = len(self.token_to_id)
        self.id_to_token = {idx: tok for tok, idx in self.token_to_id.items()}

    def encode(self, formula: str) -> list[int]:
        """Convert a LaTeX formula string into a list of token ids, wrapped with <sos>/<eos>."""
        unk_id = self.token_to_id[UNK_TOKEN]
        ids = [self.token_to_id.get(tok, unk_id) for tok in formula.split()]
        return [self.token_to_id[SOS_TOKEN], *ids, self.token_to_id[EOS_TOKEN]]

    def decode(self, token_ids: list[int]) -> str:
        """Convert a list of token ids back into a LaTeX formula string."""
        tokens = []
        for token_id in token_ids:
            token = self.id_to_token.get(token_id, UNK_TOKEN)
            if token == EOS_TOKEN:
                break
            if token in (PAD_TOKEN, SOS_TOKEN):
                continue
            tokens.append(token)
        return " ".join(tokens)

    def save(self, path: str) -> None:
        """Persist the vocabulary to disk."""
        with open(path, "w") as f:
            json.dump(self.token_to_id, f)

    @classmethod
    def load(cls, path: str) -> "LatexTokenizer":
        """Load a tokenizer with a previously saved vocabulary."""
        tokenizer = cls.__new__(cls)
        with open(path) as f:
            tokenizer.token_to_id = json.load(f)
        tokenizer.id_to_token = {idx: tok for tok, idx in tokenizer.token_to_id.items()}
        return tokenizer
