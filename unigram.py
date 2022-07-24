import collections
import numpy as np
import logging
from typing import Iterable, Mapping, Tuple
from tqdm import tqdm


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

    # This block of code removes all substrings that only appear once. 
    # Since we are removing them, total_sum has to be adjusted too
    for substr, freq in substring_counts.items():
        if freq == 1 and len(substr) > 1:
            total_sum -= 1

    substring_probs = {
        substr: -np.log(freq / total_sum)
        for substr, freq in substring_counts.items()
        if freq > 1 or len(substr) == 1
    }
    return substring_probs, word_counts

def compute_vocab_probs(corpus, vocab) -> Tuple[Mapping[str, float], Mapping[str,int]]:
    """
    From a given vocab and corpus, finds all the current probs

    Ex:
    >>> compute_substring_probs(corpus = 'abc ab', vocab = ['a','b','c','ab'])
    {'a': 1.252762968495368, 'ab': 1.252762968495368, 'b': 1.252762968495368, 'c': 1.9459101490553135}
    """

    word_counts = collections.defaultdict(int)

    # First, just count the occurence of each word in the vocab
    for word in corpus.split(" "):
        word_counts[word] += 1


    substring_counts = collections.defaultdict(int)
    total_sum = 0
    for word, freq in tqdm(word_counts.items(), desc = 'Computing Substring Frequences'):
        for idx_start in range(len(word)):
            for idx_end in range(idx_start + 1, len(word) + 1):
                substr = word[idx_start:idx_end]
                if substr in vocab:
                    substring_counts[substr] += freq
                    total_sum += freq

    substring_probs = {
        substr: -np.log(freq / total_sum)
        for substr, freq in substring_counts.items()
    }

    return substring_probs, word_counts


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

    subword_slices_arr, neg_loglik_arr = viterbi_forward(word, vocab)
    return viterbi_backward(word, subword_slices_arr, neg_loglik_arr)



if __name__ == "__main__":

    with open('text.txt', 'r', encoding='UTF-8') as f:
        corpus = f.read()

    vocab, word_counts = create_seed_vocab(corpus=corpus)
    
    base_corpus_loss = 0
    for word, freq in word_counts.items():
        base_corpus_loss += freq*tokenize_word(word, vocab)[1]
    
    print(f"Base corpus loss: {base_corpus_loss} - Number of tokens: {len(vocab.keys())}")

    with tqdm() as pbar:
        while len(vocab.keys()) > 1000:

            eta = 0.1
            scores_diff = {}
            for key in tqdm(vocab.keys(), desc = 'Computing token removals'):

                # Technically this keeps punctuation...
                if len(key) > 1:
                    vocab_complement = list(set(vocab.keys()) - set([key]))
                    vocab_complement, word_counts = compute_vocab_probs(corpus, vocab_complement)
                    corpus_loss = 0

                    for word, freq in word_counts.items():
                        corpus_loss += freq*tokenize_word(word, vocab_complement)[1]
                    
                    # should be negative. We expect loss to go up if we remove tokens, since we have done the optimal tokenization?
                    # maybe not though? Since there will be redundant tokenizations of words that are never used
                    # loss_diff = base_corpus_loss - corpus_loss
                    # print(f"Increase in loss from removing token {key} : {loss_diff}")
                    scores_diff[key] = base_corpus_loss - corpus_loss
            
            # Sort in descending order by diff in loss
            sorted_scores = sorted(scores_diff.items(), key=lambda x: x[1], reverse=True)

            for i in range(int(len(vocab) * eta)):
                _ = vocab.pop(sorted_scores[i][0])
            pbar.update(1)


    final_vocab =  compute_vocab_probs(corpus, vocab)

    print(final_vocab)
        

