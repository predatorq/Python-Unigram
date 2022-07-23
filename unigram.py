import collections
from tqdm import tqdm
import numpy as np
import logging
from typing import Iterable, Mapping, Tuple


logging.basicConfig(
    level=logging.INFO, filename="log/unigram.log", filemode="w"
)


def create_seed_vocab(corpus: str) -> Mapping[str, float]:
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
    for word in corpus.split(" "):
        word_counts[word] += 1

    # Store counts for all substrings
    substring_counts = collections.defaultdict(int)
    total_sum = 0
    for word, freq in word_counts.items():
        for idx_start in range(len(word)):
            for idx_end in range(idx_start + 1, len(word) + 1):
                substring_counts[word[idx_start:idx_end]] += freq
                total_sum += freq

    # log probs for every substring
    substring_probs = {
        substr: -np.log(freq / total_sum)
        for substr, freq in substring_counts.items()
        if freq > 1 or len(substr) == 1
    }
    return substring_probs


def viterbi_forward(
    word: str, vocab: Mapping[str, float]
) -> Tuple[list, np.array]:
    """
    Viterbi forward step. Get all tokenizations for a word
    """

    best_subword_slices_arr = [None] * (len(word) + 1)
    neg_loglik = np.zeros(len(word) + 1)

    for i in range(1, len(word) + 1):
        # iterate up to current char
        neg_loglik[i] = np.inf

        for j in range(i):
            # indexing through everything up to the current character

            if word[j:i] in vocab.keys():

                # compute log prob of the token
                logp = vocab[word[j:i]]

                # get logp of best probability up to this point
                logp_prev = neg_loglik[j]

                # Compute the new subword probability
                s = logp_prev + logp
                if s < neg_loglik[i]:
                    neg_loglik[i] = s
                    best_subword_slices_arr[i] = (j, i)

    return best_subword_slices_arr, neg_loglik


def viterbi_backward(
    word: str, subword_slices: Iterable[Tuple], subword_losses: Iterable[float]
) -> Tuple[Iterable[str], float]:

    tokenized_word = []
    curr_slice = subword_slices[-1]
    tokenized_loss = 0

    while curr_slice is not None:
        tokenized_word.append(word[curr_slice[0] : curr_slice[1]])
        tokenized_loss += subword_losses[curr_slice[1]]
        curr_slice = subword_slices[curr_slice[0]]

    return tokenized_word[::-1], tokenized_loss


def tokenize_word(
    word: str, vocab: Mapping[str, float]
) -> Tuple[Iterable[str], float]:
    """
    Given a word, and current vocabulary, performs Viterbi forward and backward
    pass and returns the tokenized word along with its losses
    """

    subword_slices_arr, neg_loglik_arr = viterbi_forward(word, vocab=vocab)
    return viterbi_backward(word, subword_slices_arr, neg_loglik_arr)


if __name__ == "__main__":

    corpus = "hug pug pun bun hugs"

    vocab = create_seed_vocab(corpus=corpus)

    for word in corpus.split(" "):
        print(tokenize_word(word, vocab))
