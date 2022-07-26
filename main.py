from src.unigram import *

if __name__ == "__main__":

    with open("text.txt", "r", encoding="UTF-8") as f:
        corpus = f.read()

    # final_vocab =  train_tokenizer(corpus, save_path='tokenization.json', min_num_tokens=2100)
    final_vocab = load_saved_tokenizer("tokenizers/example_tokenizer.json")

    word = "unigram"
    # print(tokenize_inference(words, final_vocab))
    print(tokenize_word(word, final_vocab))
