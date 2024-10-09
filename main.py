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
import itertools
import math
import random
import re
import string
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


def simple_probability_estimation(trigram_counts):
    trigram_probs = defaultdict(dict)
    for bigram, counts in trigram_counts.items():
        total = sum(counts.values())
        for char, count in counts.items():
            trigram_probs[bigram][char] = count / total
    return trigram_probs


def add_alpha_smoothing(trigram_counts, bigram_counts, alpha=0.01):
    trigram_probs = defaultdict(dict)
    for bigram, counts in trigram_counts.items():
        total = sum(counts.values())
        for char, count in counts.items():
            trigram_probs[bigram][char] = (count + alpha) / (total + alpha * len(bigram_counts[bigram]))
    return trigram_probs

def good_turing_smoothing(trigram_counts, bigram_counts):
    # Step 1: Count frequencies of frequencies (N_k)
    freq_of_freqs = Counter()
    for bigram in trigram_counts:
        for count in trigram_counts[bigram].values():
            freq_of_freqs[count] += 1

    # Step 2: Compute adjusted counts using Good-Turing formula
    adjusted_trigram_probs = defaultdict(dict)

    for bigram in trigram_counts:
        total_bigram_count = sum(bigram_counts[bigram].values())  # Total count of the bigram

        for char, count in trigram_counts[bigram].items():
            adjusted_count = (count + 1) * (freq_of_freqs[count + 1] / freq_of_freqs[count])
            prob = adjusted_count / total_bigram_count

            adjusted_trigram_probs[bigram][char] = prob

    # Step 3: Handle unseen trigrams (those with count 0)
    unseen_prob = freq_of_freqs[1] / sum(freq_of_freqs.values())

    for bigram in bigram_counts:
        total_bigram_count = sum(bigram_counts[bigram].values())

        for char in bigram_counts[bigram]:
            if char not in trigram_counts[bigram]:
                adjusted_trigram_probs[bigram][char] = unseen_prob / total_bigram_count

    return adjusted_trigram_probs


def model_training(input_file, output_file):
    bigram_counts = defaultdict(Counter)
    trigram_counts = defaultdict(Counter)
    charset = list(string.ascii_lowercase) + ['.', '0', '#', ' ']
    all_trigrams = sorted([''.join(trigram) for trigram in itertools.product(charset, repeat=3) if
                           not (trigram[0] == '#' and trigram[2] == '#') and not (
                                   trigram[1] == '#' and not (trigram[0] == '#' and trigram[1] == '#'))])

    with open(input_file) as f:
        for line in f:
            line = preprocess_line(line)
            for i in range(len(line) - 2):
                bigram = (line[i], line[i + 1])
                trigram = (line[i], line[i + 1], line[i + 2])
                trigram_counts[bigram][trigram[2]] += 1
                bigram_counts[bigram][trigram[2]] += 1

    # trigram_probs = simple_probability_estimation(trigram_counts)  # pp = 7.089956
    # trigram_probs = add_alpha_smoothing(trigram_counts, bigram_counts, alpha=0.017)  # pp = 7.087658
    trigram_probs = good_turing_smoothing(trigram_counts, bigram_counts)  # pp = 5.890559

    with open(output_file, 'w') as f_out:
        for trigram in all_trigrams:
            bigram = (trigram[0], trigram[1])
            char = trigram[2]
            prob = trigram_probs.get(bigram, {}).get(char, 0.0)
            f_out.write(f"{trigram} {prob:.6f}\n")


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

    with open(test_file) as f:
        test_text = f.read()
        test_text = preprocess_line(test_text)
    N = len(test_text)
    logP = 0
    for i in range(len(test_text) - 2):
        bigram = test_text[i:i + 2]
        next_char = test_text[i + 2]
        if bigram in language_model:
            next_chars, probabilities = zip(*language_model[bigram])
            if next_char in next_chars:
                prob = float(probabilities[next_chars.index(next_char)])
            else:
                prob = 0
        else:
            prob = 0
        logP += -1 * (prob and math.log2(prob))
    perplexity = 2 ** (logP / N)
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
print(generate_from_LM("assignment1-data/output_model.en", 300))

# Test compute_perplexity()
print(compute_perplexity("assignment1-data/test", "assignment1-data/model-br.en"))
print(compute_perplexity("assignment1-data/test", "assignment1-data/output_model.en"))
