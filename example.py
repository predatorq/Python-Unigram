from src.unigram import *

if __name__ == "__main__":
    tokenizer = UnigramTokenizer(vocab_path='tokenizers/sample_tokenizer.json')
    mystr = "Alice could perhaps be in Wonderland"
    print(tokenizer(mystr))

