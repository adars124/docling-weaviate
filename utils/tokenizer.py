from tiktoken import get_encoding
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from constants import MAX_TOKENS


class OpenAITokenizerWrapper(PreTrainedTokenizerBase):
    """Minimal wrapper for OpenAI tokenizer."""

    def __init__(
        self, model_name: str = "cl100k_base", max_length: int = MAX_TOKENS, **kwargs
    ):
        """Initialize the tokenizer.

        Args:
            model_name: The name of the OpenAI encoding to use
            max_length: Maximum sequence length
        """
        super().__init__(model_max_length=max_length, **kwargs)
        self.tokenizer = get_encoding(model_name)
        self._vocab_size = self.tokenizer.max_token_value

    def tokenize(self, text: str, *args) -> list[str]:
        """Main method used by HybridTokenizer."""
        return [str(t) for t in self.tokenizer.encode(text)]

    def _tokenize(self, text: str) -> list[str]:
        return self.tokenize(text)

    def _convert_token_to_id(self, token: str) -> int:
        return int(token)

    def _convert_id_to_token(self, index: int) -> str:
        return str(index)

    def get_vocab(self) -> dict[str, int]:
        return dict(enumerate(range(self.vocab_size)))

    def save_vocabulary(self, *args) -> tuple[str]:
        return tuple()

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def __len__(self) -> int:
        """Return the size of vocabulary."""
        return self._vocab_size

    @property
    def from_pretrained(cls, *args, **kwargs) -> "OpenAITokenizerWrapper":
        """Class method to match HuggingFace's interface."""
        return cls()


if __name__ == "__main__":
    tokenizer = OpenAITokenizerWrapper()
    print(tokenizer.tokenize("Hello, world!"))
