import itertools
from typing import Sequence, Tuple, List
import torch


STD_CHARS = (
    'L','A','G','V','S','E','R','T','I',
    'D','P','K','Q','N','F','Y','M','H',
    'W','C','X','B','U','Z','O','.','-'
)


class Tokenizer:
    """Tokenizes protein sequences into fixed-length ID tensors."""

    PREFIX_CHARS = ("<cls>", "<pad>", "<eos>", "<unk>")
    SUFFIX_CHARS = ("<mask>",)

    def __init__(self, padded_size: int):
        self.padded_size = padded_size

        # Build vocabulary: prefix → standard → padding nulls → suffix
        self.all_toks = list(self.PREFIX_CHARS)
        self.all_toks.extend(STD_CHARS)
        padding_needed = (8 - (len(self.all_toks) % 8)) % 8
        for i in range(padding_needed):
            self.all_toks.append(f"<null_{i+1}>")
        self.all_toks.extend(self.SUFFIX_CHARS)

        # Create mapping
        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}
        self.unk_idx     = self.tok_to_idx["<unk>"]
        self.padding_idx = self.tok_to_idx["<pad>"]
        self.cls_idx     = self.tok_to_idx["<cls>"]
        self.eos_idx     = self.tok_to_idx["<eos>"]
        self.mask_idx    = self.tok_to_idx["<mask>"]

        self.unique_no_split_tokens = set(self.all_toks)

    def __len__(self) -> int:
        return len(self.all_toks)

    def get_idx(self, tok: str) -> int:
        return self.tok_to_idx.get(tok, self.unk_idx)

    def get_tok(self, idx: int) -> str:
        return self.all_toks[idx]

    def to_dict(self) -> dict:
        return dict(self.tok_to_idx)

    def __call__(self, sequences: Sequence[str]) -> torch.Tensor:
        """
        Encode a batch of raw sequences into a tensor of shape
        (batch_size, padded_size+2), with <cls> at [*,0] and <eos> at [*,len+1].
        """
        batch_size = len(sequences)
        # +2 for <cls> and <eos>
        out = torch.full((batch_size, self.padded_size + 2),
                         self.padding_idx, dtype=torch.int64)

        for i, seq in enumerate(sequences):
            seq_ids = self.encode(seq)[: self.padded_size]
            # BOS
            out[i, 0] = self.cls_idx
            # main tokens
            out[i, 1 : 1 + len(seq_ids)] = torch.tensor(seq_ids, dtype=torch.int64)
            # EOS (if space remains)
            if 1 + len(seq_ids) < out.size(1):
                out[i, 1 + len(seq_ids)] = self.eos_idx

        return out

    def encode(self, text: str) -> List[int]:
        return [self.get_idx(tok) for tok in self.tokenize(text)]

    def tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Split `text` into tokens, preserving any special tokens from
        `self.unique_no_split_tokens`.
        """
        def split_on_token(tok: str, t: str) -> List[str]:
            parts = t.split(tok)
            res = []
            for idx, p in enumerate(parts):
                if idx < len(parts)-1:
                    p = p.rstrip()
                if idx > 0:
                    p = p.lstrip()
                if idx == 0 and not p:
                    res.append(tok)
                elif idx == len(parts)-1:
                    if p:
                        res.append(p)
                else:
                    if p:
                        res.append(p)
                    res.append(tok)
            return res

        def split_iter(tokens: List[str], t: str) -> List[str]:
            if not t.strip():
                return []
            curr = [t]
            for tok in tokens:
                new = []
                for segment in curr:
                    if segment in self.unique_no_split_tokens:
                        new.append(segment)
                    else:
                        new.extend(split_on_token(tok, segment))
                curr = new
            # final whitespace split for any non-special
            final = []
            for seg in curr:
                if seg in self.unique_no_split_tokens:
                    final.append(seg)
                else:
                    final.extend(seg.split())
            return final

        return split_iter(list(self.unique_no_split_tokens), text)
