from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset, load_metric
import torch
import numpy as np

def generate_summary(model, tokenizer, input_text, max_length=150,
                     num_beams=4):
    # Tokenize the input text
    input_ids = tokenizer.encode("summarize: " + input_text,
                                 return_tensors="pt", add_special_tokens=True)
    # Generate summary IDs
    summary_ids = model.generate(input_ids, max_length=max_length,
                                 num_beams=num_beams, early_stopping=True)
    # Decode and return the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def main():
    # Load the model and tokenizer
    model_name = "t5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Load the X-Sum dataset
    dataset = load_dataset("xsum", split="test")

    # Load the Rouge metric
    rouge = load_metric("rouge")

    # Lists to store predictions and references
    predictions = []
    references = []

    # Process the dataset in batches
    batch_size = 8  # Adjust based on your machine's capabilities
    for batched_examples in batch(dataset, n=batch_size):
        input_texts = [ex["document"] for ex in batched_examples]
        reference_summaries = [ex["summary"] for ex in batched_examples]

        # Generate summaries
        batch_predictions = [generate_summary(model, tokenizer, input_text) for
                             input_text in input_texts]
        predictions.extend(batch_predictions)
        references.extend(reference_summaries)

    # Compute ROUGE scores
    scores = rouge.compute(predictions=predictions, references=references,
                           use_stemmer=True)
    # Print aggregated ROUGE scores
    for key in scores.keys():
        print(f"{key}: {np.mean(scores[key])}")


if __name__ == "__main__":
    main()
