import re
import math
import matplotlib.pyplot as plt
import random
from collections import defaultdict


def preprocess_line(text):
    text = text.lower()
    text = re.sub(r"\d", '0', text)
    text = re.sub(r"[^a-z0. ]", "", text)
    text = "##" + text + "#"  # Add start and end markers
    return text


def split_corpus(input_file, train_ratio=0.8, dev_ratio=0.1, seed=42):
    # Helper function to split the corpus into training, development, and test sets
    with open(input_file, 'r') as f:
        lines = f.readlines()
    random.seed(seed)
    random.shuffle(lines)

    total_lines = len(lines)
    train_end = int(total_lines * train_ratio)
    dev_end = int(total_lines * (train_ratio + dev_ratio))

    training_set = lines[:train_end]
    development_set = lines[train_end:dev_end]
    test_set = lines[dev_end:]

    return training_set, development_set, test_set


def write_model_to_file(all_trigrams, trigram_probs, output_file):
    # Helper function to write the model details to the output file
    with open(output_file, 'w') as f_out:
        for trigram in all_trigrams:
            bigram = (trigram[0], trigram[1])
            char = trigram[2]
            prob = trigram_probs.get(bigram, {}).get(char, 0.0)
            f_out.write(f"{trigram} {prob:.3e}\n")


def load_language_model(model_file):
    # Helper function to load the language model from the model file
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


def generate_from_LM(model_file, sequence_length=300, iterations=20):
    language_model = load_language_model(model_file)
    generated_sequences = []
    for i in range(iterations):
        generated_sequence = '##'
        while len(generated_sequence) < sequence_length:
            bigram = generated_sequence[-2:]

            if bigram in language_model:
                next_chars, probabilities = zip(*language_model[bigram])
                probabilities = [float(p) for p in probabilities]
                next_char = random.choices(next_chars, probabilities)[0]
                generated_sequence += next_char
            else:
                break
        generated_sequences.append(generated_sequence)

    return generated_sequences


def compute_perplexity(test_file, model):
    language_model = load_language_model(model)
    text_length = 0
    logP = 0
    with open(test_file) as f:
        for test_text in f:
            test_text = preprocess_line(test_text)
            N = len(test_text)
            text_length += N - 2
            prob = 0
            for i in range(N - 2):
                bigram = test_text[i:i + 2]
                next_char = test_text[i + 2]
                if bigram in language_model:
                    next_chars, probabilities = zip(*language_model[bigram])
                    prob = float(probabilities[next_chars.index(next_char)])
                logP += -1 * math.log2(prob)
    perplexity = 2 ** (logP / text_length)
    return perplexity


def frange(start, stop, step):
    # Helper function to generate float range
    while start < stop:
        yield start
        start += step


def plot_distribution(model_file, bigram_history):
    # Helper function to plot the distribution of trigrams starting with a given bigram
    language_model = load_language_model(model_file)
    if bigram_history in language_model:
        next_chars, probabilities = zip(*language_model[bigram_history])
        probabilities = [float(p) for p in probabilities]
        prob_sum = sum(probabilities)
        print(f"Total Probability: {prob_sum}")
        plt.figure(figsize=(10, 6))
        plt.bar(next_chars, probabilities, color='blue')

        plt.xlabel(f'Trigrams (Starting with "{bigram_history}")')
        plt.ylabel('Probability Values')
        plt.title(f'Probability Distribution of Trigrams Starting with "{bigram_history}"')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()
