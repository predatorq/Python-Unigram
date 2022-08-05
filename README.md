# Python Implementation of Unigram

Pedagogical implementation of the Unigram Tokenization algorithm as described in [Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates](https://arxiv.org/abs/1804.10959).

# Example Use

```tokenizers/sample_tokenizer.json``` is a tokenized vocabulary of size 977 trained on the first chapter
of Alice's Adventures in Wonderland sourced from [Project Gutenburg](https://www.gutenberg.org/ebooks/11). Tokenizer was trained on UTF-8 bytes as is [common](https://openai.com/blog/better-language-models/) for many tokenizers.

```python 
# Training from scratch (takes a long time!)
tokenizer = UnigramTokenizer(corpus_path='mycorpus.txt')
tokenizer.train_tokenizer(save_path='myvocab.json', min_vocab_size=16384)
```


```python
# Loading pretrained tokenizer
tokenizer = UnigramTokenizer(vocab_path='tokenizers\sample_tokenizer.json')

sentence = 'This is a short sentence.'
tokenized_sentence = tokenizer.tokenize_inference(sentence)
>>> print(tokenized_sentence)
['T', 'h', 'i', 's', 'Ġis', 'Ġa', 'Ġsh', 'or', 't', 'Ġs', 'en', 't', 'e', 'nce', '.']
```

# Testing
```
python -m pytest
```

# Citations

```bibtex
@article{DBLP:journals/corr/abs-1804-10959,
  author    = {Taku Kudo},
  title     = {Subword Regularization: Improving Neural Network Translation Models
               with Multiple Subword Candidates},
  journal   = {CoRR},
  volume    = {abs/1804.10959},
  year      = {2018}
}
```

```bibtex
@misc{chung_2019, 
  author    = {Chung, Kyle},
  title     = {On Subword units}, 
  url       = {https://everdark.github.io/k9/notebooks/ml/natural_language_understanding/subword_units/subword_units.nb.html},  
  year      = {2019}, 
} 
```

```bibtex
@misc{Summary of the Tokenizers, 
  title     = {Summary of the tokenizers}, 
  url       = {https://huggingface.co/docs/transformers/tokenizer_summary#unigram}, 

```
