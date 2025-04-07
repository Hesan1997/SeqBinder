"""
Encoders module for converting sequences (e.g., SMILES or protein sequences) into tokenized representations.

Contains:
- Tokenizer: Abstract base class for encoders.
- CharTokenizer: Encoder for protein sequences.
- SMILESTokenizer: Encoder for SMILES strings.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable

import re


class Tokenizer(ABC):
    """
    Abstract base class for tokenizing sequences into symbols.
    
    Provides a framework for converting input sequences into numerical representations using a defined vocabulary.
    """

    def __init__(self, sequences: Iterable[str], maxLen: int, pad: bool = True, unk: bool = False):
        """
        Initialize the tokenizer by building a vocabulary from input sequences.
        
        :param sequences: Collection of input sequences (e.g., SMILES or protein sequences).
        :param maxLen: Maximum length of a tuple after padding.
        :param pad: Whether to add a 'PAD' token to the vocabulary.
        :param unk: Whether to add an 'UNK' token for unknown symbols.
        """
        self.maxLen = maxLen
        self.pad = pad
        self.unk = unk

        # Create vocabulary with a default indexing function
        self.vocabulary = defaultdict(lambda: len(self.vocabulary))

        # Optionally include padding and unknown tokens in the vocabulary
        if pad:
            self.vocabulary['PAD'] = 0    # Padding token
        if unk:
            self.vocabulary['UNK'] = 1    # Unknown token

        for seq in sequences:
            for symbol in self.parse(seq):
                self.vocabulary[symbol]

    def __call__(self, sequence: str) -> tuple:
        """
        Convert a sequence into a tuple of token indexes, with optional padding.
        
        :param sequence: The input sequence to tokenize.
        :return: A tuple of token indexes.
        """
        # Tokenize the input sequence
        symbols = self.parse(sequence)

        # Handle unknown tokens if specified
        indexes = [
            self.vocabulary.get(symbol, self.vocabulary['UNK']) if self.unk else self.vocabulary[symbol]
            for symbol in symbols
        ]

        # Apply padding if necessary
        if self.pad:
            padded_sequence = indexes + [self.vocabulary['PAD']] * (self.maxLen - len(indexes))
            return tuple(padded_sequence[:self.maxLen])  # Truncate if necessary

        return tuple(indexes)

    @abstractmethod
    def parse(self, sequence: str):
        """
        Tokenize the given sequence into a tuple of tokens.
        """
        pass


class CharTokenizer(Tokenizer):
    """
    Tokenizer for protein sequences.

    Examples
    --------
    >>> encoder = CharTokenizer(["ACDEFGHIKLMNPQRSTVWY"], maxLen=10)
    >>> encoder.parse("ACDEFGHIKLMNPQRSTVWY")
    ('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y')
    """

    def parse(self, sequence: str):
        """
        Tokenize a protein sequence.
        """
        return tuple(sequence)


class SMILESTokenizer(Tokenizer):
    """
    Tokenizer for SMILES strings using a regex pattern from Schwaller et al.

    Examples
    --------
    >>> encoder = SMILESTokenizer(["CC(=O)OC1=CC=CC=C1C(=O)O"], maxLen=25)
    >>> encoder.parse("CC(=O)OC1=CC=CC=C1C(=O)O")
    ('C', 'C', '(', '=', 'O', ')', 'O', 'C', '1', '=', 'C', 'C', '=', 'C', 'C', '=', 'C', '1', 'C', '(', '=', 'O', ')', 'O')

    References
    ----------
    Schwaller, P., et al. "Molecular Transformer: A Model for Uncertainty-Calibrated Chemical Reaction Prediction." 
    ACS Central Science, 2019, 5(9), 1572-1583. DOI: 10.1021/acscentsci.9b00576
    """
    
    REGEX_PATTERN = re.compile(
        r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
    )

    def parse(self, sequence: str):
        """
        Tokenize a SMILES string.
        """
        return tuple(self.REGEX_PATTERN.findall(sequence))


if __name__ == '__main__':
    import doctest
    doctest.testmod()