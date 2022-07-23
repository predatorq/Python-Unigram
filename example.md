corpus = "hug pug pun bun hugs"

# Corpus Creation Step (Step 0)
Our basic corpus (all characters or tokens appearing more than once):
{'h': 2.833, 'hu': 2.833, 'hug': 2.833, 'u': 1.916, 'ug': 2.427, 'g': 2.427, 'p': 2.833, 'pu': 2.833, 'un': 2.833, 'n': 2.833, 'b': 3.526, 's': 3.526}


# Tokenization using full corpus (Step 1)

Naively Tokenizing hug:

Has a few tokenizations. Evaluate the neg log probs for each (just the sum of the token values)
(h u g): 7.176
(hug): 2.833
(h ug): 5.26
(hu g): 5.26

We are looking for tokenization with the smallest loss. In that case it is (hug), L = 2.833

In general however, computing *every* tokenization for a word would be terribly slow, especially if the word
is long and the vocabulary is huge (as there would be many ways to split the word). Instead, we use the Viterbi Algorithm. Doing this for "hug" would go as follows:

1. (up to) First char: only one way to tokenize it: (h) - 2.833
2. (up to) Second char: Two ways: (h u) - 4.749 and (hu) - 2.833
3. (up to) Third char: All ways are (h u g), (hu g), (h ug), (hug). However, we discard (h u g) since we have ruled out (h u) to be the best tokenization.

Final tokenization is then (hug)!

