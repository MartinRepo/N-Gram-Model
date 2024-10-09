import re
import random
import math


def preprocess_line(text):
    text = text.lower()
    text = re.sub(r"\d", '0', text)
    text = re.sub(r"[^a-z0. ]", "", text)
    text = "##" + text + "#"  # Add start and end markers
    return text


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


def compute_perplexity(test_file, model):
    language_model = load_language_model(model)

    with open(test_file) as f:
        test_text = f.read()
        test_text = preprocess_line(test_text)
    N = len(test_text)
    logP = 0
    prob = 0
    for i in range(len(test_text) - 2):
        bigram = test_text[i:i + 2]
        next_char = test_text[i + 2]
        if bigram in language_model:
            next_chars, probabilities = zip(*language_model[bigram])
            if next_char in next_chars:
                prob = float(probabilities[next_chars.index(next_char)])
        logP += -1 * (prob and math.log2(prob))
    perplexity = 2 ** (logP / N)
    return perplexity