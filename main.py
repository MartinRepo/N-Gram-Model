"""
ANLP Assignment 1
Authors: Chi Xing,
Email: s2682783@ed.ac.uk
Date: 2024-10-05
|====================WARM UP=======================|
| Calculate the perplexity of the sequence ##abaab#|
| Perplexity = 2^(- logP / N)                      |
| Result = 3.14                                    |
|==================================================|
"""
import math
import random
import re
import sys
from collections import defaultdict, Counter

"""
1. Preprocessing each line
"""


def preprocess_line(text):
    text = text.lower()
    text = re.sub(r"\d", '0', text)
    text = re.sub(r"[^a-z0. ]", "", text)
    text = "##" + text + "#"  # Add start and end markers
    return text


"""
2. Examining a pre-trained model [NO CODE | PASS]
"""

"""
3. Implementing a model
Basic Workflow:
Read file -> Collect counts -> Estimate probabilities -> Write model probabilities to result.file
"""


def model_training(input_file, output_file):
    bigram_counts = defaultdict(Counter)
    trigram_counts = defaultdict(Counter)

    with open(input_file) as f:
        for line in f:
            line = preprocess_line(line)
            for i in range(len(line) - 2):
                bigram = (line[i], line[i + 1])
                trigram = (line[i], line[i + 1], line[i + 2])
                trigram_counts[bigram][trigram[2]] += 1
                bigram_counts[bigram][trigram[2]] += 1

    with open(output_file, 'w') as f_out:
        for bigram in trigram_counts:
            total_bigram_count = sum(bigram_counts[bigram].values())
            for char, count in trigram_counts[bigram].items():
                prob = count / total_bigram_count
                f_out.write(f"{bigram[0]}{bigram[1]}{char} {prob:.6f}\n")


"""
4. Generating from models
"""
def load_language_model(model_file):
    language_model = {}
    with open(model_file) as f:
        for line in f:
            trigram = line[:3]
            probability = line[4:]
            history = trigram[:2]
            next_char = trigram[2]
            if history not in language_model:
                language_model[history] = []
            language_model[history].append((next_char, probability))
    return language_model

def generate_from_LM(model_file, sequence_length=300):
    language_model = load_language_model(model_file)
    generated_sequence = '##'
    while len(generated_sequence) < sequence_length:
        # Get the last two characters (bigram history)
        bigram = generated_sequence[-2:]

        # If the bigram exists in the model, use the probabilities to pick the next character
        if bigram in language_model:
            next_chars, probabilities = zip(*language_model[bigram])
            probabilities = [float(p) for p in probabilities]
            next_char = random.choices(next_chars, probabilities)[0]
            generated_sequence += next_char
        else:
            # If the bigram is not in the model, (Use Backoff to try smaller n?) (here just break early)
            break

    return generated_sequence

"""
5. Computing perplexity
"""


def compute_perplexity(test_file, model):
    language_model = load_language_model(model)
    total_log_prob = 0
    total_trigrams = 0

    with open(test_file, 'r') as f:
        for line in f:
            line = preprocess_line(line)
            line = '##' + line + '#'

            for i in range(len(line) - 2):
                bigram = line[i:i + 2]
                next_char = line[i + 2]

                if bigram in language_model:
                    # Look up the probability of the next character given the bigram
                    candidates = language_model[bigram]
                    probabilities = {char: prob for char, prob in candidates}
                    prob = float(probabilities.get(next_char, 0))
                else:
                    # If bigram not found, assume a small probability (smoothing)
                    prob = 1e-6

                if prob > 0:
                    total_log_prob += math.log2(prob)
                else:
                    total_log_prob += math.log2(1e-6)  # Handle unseen trigrams

                total_trigrams += 1

    # Perplexity formula
    if total_trigrams == 0:
        return float('inf')  # Prevent division by zero if no trigrams are found
    avg_log_prob = total_log_prob / total_trigrams
    perplexity = 2 ** (-avg_log_prob)

    return perplexity


"""
|====================MAIN SCRIPT=======================|
| Usage: python main.py <training_file> <output_file>  |
|======================================================|
"""
# Test model_training()
model_training("assignment1-data/training.en", "assignment1-data/output_model.en")

# Test generate_from_LM()
print(generate_from_LM("assignment1-data/model-br.en", 300))
print(generate_from_LM("output.en", 300))

# Test compute_perplexity()
print(compute_perplexity("assignment1-data/test", "assignment1-data/model-br.en"))
print(compute_perplexity("assignment1-data/test", "output.en"))