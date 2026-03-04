# %%
from typing import Iterable, Iterator
from functools import lru_cache
from copy import copy

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
C_PAT = re.compile(PAT)

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = copy(vocab)
        self.vocab_to_idx = {v:idx for idx, v in vocab.items()}
        self.merges = merges
        self.merges_idx = {merge:idx for idx, merge in enumerate(merges)}
        self.lru = []
        if special_tokens is not None:
            self.special_tokens = sorted(special_tokens, reverse=True)
            vocab_set = set(vocab)
            idx = len(vocab)
            for token in special_tokens:
                if token not in vocab_set:
                    self.vocab[idx] = token.encode()
                    idx += 1

            del vocab_set
        else:
            self.special_tokens: list[str] = []

    @lru_cache(maxsize=80000)
    def _encode_pretoken(self, ptoken: str) -> list[int]:
        b = [bytes([b]) for b in ptoken.encode()]
        while True:
            min_merge = len(self.merges_idx)
            min_pair = (b'', b'')
            for p, q in zip(b[:-1], b[1:]):
                curr_rank = self.merges_idx.get((p, q))
                if curr_rank is not None and curr_rank < min_merge:
                    min_merge = curr_rank
                    min_pair = p, q

            if min_merge >= len(self.merges):
                break

            merged_pair = min_pair[0] + min_pair[1]
            idx = 0
            b_new = []
            while idx < len(b):
                if idx < len(b) - 1 and (b[idx], b[idx+1]) == min_pair:
                    b_new.append(merged_pair)
                    idx += 2
                else:
                    b_new.append(b[idx])
                    idx += 1

            b = b_new
            
        return [self.vocab_to_idx[p] for p in b]


    def encode(self, text: str) -> list[int]:
        text_chunks = [text]
        for token in self.special_tokens:
            new_chunks = []
            for chunk in text_chunks:
                match chunk:
                    case int(a):
                        new_chunks.append(a)
                    case str(b):
                        spl_str = b.split(token)
                        token_idx = self.vocab_to_idx[token.encode()]
                        new_spl_str = []
                        for e in spl_str[:-1]:
                            new_spl_str.append(e)
                            new_spl_str.append(token_idx)
                        new_spl_str.append(spl_str[-1])
                        new_chunks.extend(new_spl_str)
            text_chunks = new_chunks
        
        tokens = []
        for chunk in text_chunks:
            match chunk:
                case int(a):
                    tokens.append(a)
                case str(b):
                    ptok = re.finditer(C_PAT, b)
                    for pretokens in ptok:
                        tokens.extend(self._encode_pretoken(pretokens.group()))

        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for s in iterable:
            yield from self.encode(s)

    def decode(self, ids: list[int]) -> str:
        return (b''.join(map(lambda x: self.vocab.get(x, b''), ids))).decode(errors='replace')
