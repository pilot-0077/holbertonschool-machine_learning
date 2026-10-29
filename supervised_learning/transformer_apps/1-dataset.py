#!/usr/bin/env python3
"""Load, tokenize, and encode the Portuguese-English dataset."""
import transformers
from setup import load_pt2en


class Dataset:
    """Prepare a machine-translation dataset and encode sentence pairs."""

    def __init__(self):
        """Load the train/validation splits and train both tokenizers."""
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """Train Portuguese and English tokenizers from the training data.

        Args:
            data: Dataset of pairs of Portuguese and English string tensors.

        Returns:
            The Portuguese tokenizer and the English tokenizer.
        """
        pt_sentences = []
        en_sentences = []

        for pt, en in data.as_numpy_iterator():
            pt_sentences.append(pt.decode('utf-8'))
            en_sentences.append(en.decode('utf-8'))

        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased'
        )

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            pt_sentences, vocab_size=2 ** 13
        )
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_sentences, vocab_size=2 ** 13
        )

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """Encode a Portuguese-English sentence pair with boundary tokens.

        Args:
            pt: Tensor containing a Portuguese sentence.
            en: Tensor containing the corresponding English sentence.

        Returns:
            A pair of integer lists: Portuguese tokens and English tokens.
            Each list starts with its tokenizer's vocab_size and ends with
            vocab_size + 1.
        """
        pt_text = pt.numpy().decode('utf-8')
        en_text = en.numpy().decode('utf-8')

        pt_vocab_size = self.tokenizer_pt.vocab_size
        en_vocab_size = self.tokenizer_en.vocab_size

        pt_tokens = self.tokenizer_pt.encode(
            pt_text, add_special_tokens=False
        )
        en_tokens = self.tokenizer_en.encode(
            en_text, add_special_tokens=False
        )

        pt_tokens = [pt_vocab_size] + pt_tokens + [pt_vocab_size + 1]
        en_tokens = [en_vocab_size] + en_tokens + [en_vocab_size + 1]

        return pt_tokens, en_tokens
