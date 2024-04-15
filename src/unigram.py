import collections
import numpy as np
import logging
from typing import Iterable, Mapping, Tuple, NamedTuple, Union
from tqdm import tqdm
import json
import regex as re
from src.bytes import bytes_to_unicode


logging.basicConfig(
    level=logging.INFO, filename="log/unigram.log", filemode="w"
)


class UnigramToken(NamedTuple):
    token: Iterable[str]
    probs: Iterable[int]


class TokenizerError(Exception):
    pass


class UnigramTokenizer:
    def __init__(
        self,
        corpus_path: Union[str, None] = None,
        vocab_path: Union[str, None] = None,
    ) -> None:
        """
        Initializes a byte-level Unigram tokenizer as described in:

        `Subword Regularization: Improving Neural Network Translation Models
        with Multiple Subword Candidates`
        <https://arxiv.org/abs/1804.10959>

        """

        if corpus_path is None and vocab_path is None:
            raise TokenizerError(
                "Must provide either a corpus or a saved vocabulary path"
            )

        self.vocab = None
        self.corpus = None

        try:
            if corpus_path is not None:
                with open(corpus_path, encoding="utf-8") as f:
                    self.corpus = f.read()
            else:
                self.vocab = self.load_saved_tokenizer(save_path=vocab_path)
        except Exception as e:
            raise TokenizerError(str(e))

    def train_tokenizer(
        self,
        save_path: str,
        proportion_to_remove: float = 0.1,
        min_vocab_size: int = 32000,
    ) -> Mapping[int, UnigramToken]:
        """
        Trains a byte-level unigram tokenizer on a given corpus
        """

        assert (
            self.vocab is None
        ), "Tokenizer has already been instantiated with a trained vocabulary."
        print("Start create seed Vocab. ")
        vocab, word_counts = self.create_seed_vocab()
        print("Seed Vocab Completed. ")
        base_corpus_loss = 0
        for word, freq in word_counts.items():
            base_corpus_loss += freq * self.tokenize_word(word, vocab)[1]
        print(vocab)

        with tqdm() as pbar:
            while len(vocab.keys()) > min_vocab_size:

                scores_diff = {}
                for key in tqdm(vocab.keys(), desc="Computing token removals"):

                    if len(key) > 1:
                        vocab_complement = list(set(vocab.keys()) - set([key]))
                        (
                            vocab_complement,
                            word_counts,
                        ) = self.compute_vocab_probs(vocab_complement)
                        corpus_loss = 0

                        for word, freq in word_counts.items():
                            corpus_loss += (
                                freq
                                * self.tokenize_word(word, vocab_complement)[1]
                            )

                        scores_diff[key] = base_corpus_loss - corpus_loss

                # Sort in descending order by diff in loss
                sorted_scores = sorted(
                    scores_diff.items(), key=lambda x: x[1], reverse=True
                )

                for i in range(int(len(vocab) * proportion_to_remove)):
                    _ = vocab.pop(sorted_scores[i][0])
                pbar.update(1)

            final_vocab, _ = self.compute_vocab_probs(vocab)

            tokenized_vocab = {}
            for i, (key, value) in enumerate(final_vocab.items()):
                tokenized_vocab[i] = UnigramToken(key, value)

            if save_path is not None:
                with open(save_path, "w") as j:
                    json.dump(tokenized_vocab, j, indent=4)

            self.vocab = final_vocab
            return final_vocab

    def load_saved_tokenizer(self, save_path: str) -> Mapping[str, float]:
        """
        Loads a saved tokenizer.
        """
        try:
            with open(save_path) as f:
                raw_dict = json.load(f)

            vocab_dict = {}
            for _, value in raw_dict.items():
                vocab_dict[value[0]] = [value[1]]

            self.vocab = vocab_dict
            return vocab_dict
        except Exception as e:
            raise TokenizerError(str(e))

    def create_seed_vocab(self) -> Mapping[str, float]:
        """
        Create initial vocab from a corpus.

        Returns the initial vocabulary + returns probs

        Currently just finds every substring from a vocab - inefficient

        Example::
            >>> create_seed_vocab(corpus = 'abc ab')
            {'a': 1.5040773967762742, 'ab': 1.5040773967762742, 'abc': 2.1972245773362196,
            'b': 1.5040773967762742, 'bc': 2.1972245773362196, 'c': 2.1972245773362196}
        """

        word_counts = collections.defaultdict(int)

        pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        tokens = re.findall(pat, self.corpus)
        print("start word count")
        for word in tokens:
            token_byte_shifted = self.byte_encode_word(word)
            word_counts[token_byte_shifted] += 1

        # Store counts for all substrings
        print("start substring count")
        byte_encoder = bytes_to_unicode()
        substring_counts = collections.defaultdict(int)
        total_sum = 0
        for word, freq in word_counts.items():
            for idx_start in range(len(word)):
                for idx_end in range(idx_start + 1, len(word) + 1):
                    # a bit hacky, but we don't want the individual space token to be
                    # a part of our vocabulary.
                    if word[idx_start:idx_end] != byte_encoder[32]:
                        substring_counts[word[idx_start:idx_end]] += freq
                        total_sum += freq

        # In a 'real' corpus this probably wouldn't happen but we just want to make
        # sure all of the raw UTF-8 bytes are included in the corpus. If they don't appear
        # just log them with a frequency of 1 and update the total sum accordingly

        for _, unicode_bytes in byte_encoder.items():
            if unicode_bytes not in substring_counts:
                substring_counts[unicode_bytes] += 1
                total_sum += 1

        # This block of code removes all substrings that only appear once.
        # Since we are removing them, total_sum has to be adjusted too
        for substr, freq in substring_counts.items():
            if freq == 1 and len(substr) > 1:
                total_sum -= 1
        print(-np.log(1 / total_sum))
        substring_probs = {
            substr: -np.log(freq / total_sum)
            for substr, freq in substring_counts.items()
            if freq > 1 or len(substr) == 1
        }
        return substring_probs, word_counts

    def compute_vocab_probs(
        self, vocab
    ) -> Tuple[Mapping[str, float], Mapping[str, int]]:
        """
        From a given vocab and corpus, gets the negative log probs and returns as a dict

        Example::
            >>> compute_vocab_probs(vocab = ['a','b','c','ab'])
            {'a': 1.252762968495368, 'ab': 1.252762968495368, 'b': 1.252762968495368, 'c': 1.9459101490553135}
        """

        word_counts = collections.defaultdict(int)

        pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        tokens = re.findall(pat, self.corpus)
        for word in tokens:
            token_byte_shifted = self.byte_encode_word(word)
            word_counts[token_byte_shifted] += 1

        substring_counts = collections.defaultdict(int)
        total_sum = 0
        for word, freq in word_counts.items():
            for idx_start in range(len(word)):
                for idx_end in range(idx_start + 1, len(word) + 1):
                    substr = word[idx_start:idx_end]
                    if substr in vocab:
                        substring_counts[substr] += freq
                        total_sum += freq

        # In a 'real' corpus this probably wouldn't happen but we just want
        # to make sure all of the raw UTF-8 bytes are included in the
        # corpus. If they don't appear just log them with a frequency of 1
        # and update the total sum accordingly
        byte_encoder = bytes_to_unicode()
        for _, unicode_bytes in byte_encoder.items():
            if unicode_bytes not in substring_counts:
                substring_counts[unicode_bytes] += 1
                total_sum += 1

        substring_probs = {
            substr: -np.log(freq / total_sum)
            for substr, freq in substring_counts.items()
        }

        return substring_probs, word_counts

    def viterbi_forward(
        self, word: str, vocab: Mapping[str, float]
    ) -> Tuple[list, np.array]:
        """
        Viterbi forward step. Given a vocabulary with token probabilities returns
        all the tokenizations for the word.
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
        self,
        word: str,
        subword_slices: Iterable[Tuple],
        subword_losses: Iterable[float],
    ) -> Tuple[Iterable[str], float]:
        """
        Viterbi backward step. Given the word tokenizations up to every character,
        returns the optimal tokenization and its loss.
        """

        tokenized_word = []
        curr_slice = subword_slices[-1]
        tokenized_loss = 0

        while curr_slice is not None:
            tokenized_word.append(word[curr_slice[0] : curr_slice[1]])
            tokenized_loss += subword_losses[curr_slice[1]]
            curr_slice = subword_slices[curr_slice[0]]

        return tokenized_word[::-1], tokenized_loss

    def tokenize_word(
        self, word: str, vocab: Mapping[str, float]
    ) -> Tuple[Iterable[str], float]:
        """
        Given a (byte-encoded) word, and current vocabulary, performs Viterbi forward and backward
        pass and returns the tokenized word along with its loss.

        Example::
            >>> tokenizer_vocab = load_saved_tokenizer("tokenizers/example_tokenizer.json")
            >>> word = "unigram"
            >>> print(tokenize_word(word, tokenizer_vocab))
            (['un', 'ig', 'ra', 'm'], 70.90316975550996)

        """

        subword_slices_arr, neg_loglik_arr = self.viterbi_forward(word, vocab)
        return self.viterbi_backward(word, subword_slices_arr, neg_loglik_arr)

    def byte_encode_word(self, word):
        """
        Performs proper byte encoding of a word using the bytes_to_unicode function
        """
        token_bytes = word.encode("utf-8")
        byte_encoder = bytes_to_unicode()
        token_byte_shifted = "".join(byte_encoder[b] for b in token_bytes)
        return token_byte_shifted

    def tokenize_inference(self, string: str) -> Iterable[str]:
        """
        Tokenizes a string of text given a trained vocabulary

        Example::
            >>> tokenizer_vocab = load_saved_tokenizer("tokenizers/example_tokenizer.json")
            >>> mystr = "this is a test string"
            >>> print(tokenize_inference(mystr, tokenizer_vocab))
            ['this', 'Ġis','Ġa','Ġte', 'st', 'Ġst', 'ring']


        """
        # split string with regex
        pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        tokens = re.findall(pat, string)
        tokenized_sentence = []
        for word in tokens:
            token_byte_shifted = self.byte_encode_word(word)
            tokenized_sentence += self.tokenize_word(
                token_byte_shifted, self.vocab
            )[0]
        return tokenized_sentence

    def init_word_count(self):
        """
        Trains a byte-level unigram tokenizer on a given corpus
        """

        assert (
            self.vocab is None
        ), "Tokenizer has already been instantiated with a trained vocabulary."
        print("Start create seed Vocab. ")
        vocab, wordcount = self.create_seed_vocab()
        print("Seed Vocab Completed. ")
        return vocab, wordcount

    def __call__(self, string: str) -> Iterable[str]:
        return self.tokenize_inference(string)
