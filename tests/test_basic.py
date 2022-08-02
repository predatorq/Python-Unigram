import unittest
import pickle
from src.unigram import UnigramTokenizer, TokenizerError
import json


class TestUnigramTrainFunctions(unittest.TestCase):

    """
    Unittests for all individual functions used in tokenizer training.
    """

    def setUp(self) -> None:
        pass

    def test_seed_vocab(self):
        """
        Test to ensure we can create an original seed corpus
        """

        tokenizer = UnigramTokenizer(
            corpus_path="tests/data/text/unittest_text.txt"
        )

        probs, word_counts = tokenizer.create_seed_vocab()

        with open("tests/data/text/expected/probs.pickle", "rb") as f:
            expected_probs = pickle.load(f)

        with open("tests/data/text/expected/word_counts.pickle", "rb") as f:
            expected_word_counts = pickle.load(f)

        self.assertDictEqual(expected_probs, probs)

        self.assertDictEqual(expected_word_counts, word_counts)

    def test_compute_vocab_probs(self):
        """
        Ensure we can compute token probs from a vocabulary
        """

        tokenizer = UnigramTokenizer(
            corpus_path="tests/data/text/small_corpus.txt"
        )

        probs, _ = tokenizer.compute_vocab_probs(
            vocab=["a", "b", "c", "d", "ab"]
        )

        with open("tests/data/text/expected/probs_small.pickle", "rb") as f:
            expected_probs = pickle.load(f)
        self.assertDictEqual(expected_probs, probs)

    def test_tokenize_string(self):
        """
        Ensure we can tokenize a sentence given a trained vocab
        """
        tokenizer = UnigramTokenizer(
            vocab_path="tests/data/vocab/sample_tokenizer.json"
        )
        test_sentences = [
            "This is the first test sentence",
            "Another test sentence",
            "A third test sentence 🥰",
        ]

        expected_sentences = [
            [
                "T",
                "h",
                "i",
                "s",
                "Ġis",
                "Ġthe",
                "Ġ",
                "first",
                "Ġ",
                "t",
                "e",
                "st",
                "Ġs",
                "en",
                "t",
                "e",
                "nce",
            ],
            [
                "A",
                "not",
                "her",
                "Ġ",
                "t",
                "e",
                "st",
                "Ġs",
                "en",
                "t",
                "e",
                "nce",
            ],
            [
                "A",
                "Ġ",
                "t",
                "h",
                "i",
                "r",
                "d",
                "Ġ",
                "t",
                "e",
                "st",
                "Ġs",
                "en",
                "t",
                "e",
                "nce",
                "Ġ",
                "ð",
                "Ł",
                "¥",
                "°",
            ],
        ]

        for test, expected in zip(test_sentences, expected_sentences):
            self.assertEqual(tokenizer.tokenize_inference(test), expected)
            self.assertEqual(tokenizer(test), expected)

    def test_load_tokenizer(self):
        """
        Ensure we can load a saved tokenization
        """

    def test_errors(self):
        """
        Ensure errors are raised in correct spots
        """

        # Passing in no vocab or corpus
        self.assertRaises(TokenizerError, UnigramTokenizer)

        # Passing in invalid corpus
        self.assertRaises(
            TokenizerError, UnigramTokenizer, "tests/data/text/invalid_path.txt"
        )

        # Passing in invalid vocab
        self.assertRaises(
            TokenizerError,
            UnigramTokenizer,
            None,
            "tests/data/vocab/invalid_vocab.json",
        )

    def test_train_tokenizer(self):
        """
        Ensure we can train a tokenizer starting from a corpus
        """

        tokenizer = UnigramTokenizer(
            corpus_path="tests/data/text/unittest_text.txt"
        )

        tokenizer.train_tokenizer(
            save_path="tests/data/vocab/result_unittest_tokenizer.json",
            min_vocab_size=300,
        )

        with open("tests/data/vocab/result_unittest_tokenizer.json") as f:
            raw_dict = json.load(f)

        vocab_dict = {}
        for _, value in raw_dict.items():
            vocab_dict[value[0]] = [value[1]]

        with open("tests/data/vocab/expected_unittest_tokenizer.json") as f:
            raw_dict = json.load(f)

        expected_vocab_dict = {}
        for _, value in raw_dict.items():
            expected_vocab_dict[value[0]] = [value[1]]

        self.assertDictEqual(expected_vocab_dict, vocab_dict)
