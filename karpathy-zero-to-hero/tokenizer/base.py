import unicodedata
from collections import defaultdict

def get_stats(ids: list[int] , counts=None):
    counts = counts or defaultdict(int)
    for pair in zip(ids , ids[1:]):  # iterating consecutive elements
        counts[pair] += 1
    return counts

def merge(ids: list[int] , pair , index: int):
    new_ids , i = [] , 0

    while i < len(ids):
        # if not at the last position AND the pair matches, replace
        if i < len(ids)-1 and (ids[i] == pair[0] and ids[i+1] == pair[1]):
            new_ids.append(index)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
        
    return new_ids

class Tokenizer:
    def __init__(self):
        # default vocab size is 256, no merges, and no patterns
        self.merges = {}  # (int,int) -> int  # tuple of ints to int
        self.pattern = ''
        self.special_tokens = {}  # str -> int  ({'<|endoftext|>': 100257})
        self.vocab = self._build_vocab()  # int -> bytes

    def train(self , text: str , vocab_size: int = 256 , verbose: bool = False):
        ids = list(map(int , text.encode('utf-8')))
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = max(stats , key=stats.get)
            index = 256 + len(self.merges)
            self.merges[pair] = index
            ids = merge(ids , pair , index)
            if verbose:
                print(f'merging {pair} into {index}')

    def encode(self , text: str):
        return list(map(self.vocab.get , text.encode('utf-8')))

    def decode(self , ids: list[int]):
        return ''.join(map(self.vocab.get , ids))

    def _build_vocab(self):
        # vocab is derived from merges, special tokens, and patterns
        vocab = {index:bytes([index]) for index in range(256)}
        for (p0,p1) , index in self.merges.items():
            vocab[index] = vocab[p0] + vocab[p1]
        for special_token , index in self.special_tokens.items():
            vocab[index] = special_token.encode('utf-8')
        return vocab