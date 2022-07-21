import collections, re
from typing import Iterable, NamedTuple
import nltk
import json
from tqdm import tqdm
import numpy as np 


def create_seed_vocab(corpus: str):
    """ 
    Create initial vocab from a corpus.

    Returns the initial vocabulary + returns probs

    Currently just finds every substring from a vocab - super inefficient

    Ex:
    >>> create_seed_vocab(corpus = 'abc ab')
    {'a': 1.5040773967762742, 'ab': 1.5040773967762742, 'abc': 2.1972245773362196, 
    'b': 1.5040773967762742, 'bc': 2.1972245773362196, 'c': 2.1972245773362196}
    """

    word_counts = collections.defaultdict(int)

    # First, just count the occurence of each word in the vocab
    for word in corpus.split(' '):
        word_counts[word] += 1
    
    substring_counts = collections.defaultdict(int)
    total_sum = 0
    for word, freq in word_counts.items():
        for idx_start in range(len(word)):
            for idx_end in range(idx_start+1,len(word)+1):            
                substring_counts[word[idx_start:idx_end]] += freq
                total_sum += freq 

    # log probs for every substring         
    substring_probs = {substr: -np.log(freq/total_sum) for substr, freq in substring_counts.items()}
    return substring_probs

if __name__ == '__main__':

    print(create_seed_vocab(corpus = 'abc ab'))
            





