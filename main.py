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


def model_training(infile, outfile):
    bigram_counts = defaultdict(Counter)
    trigram_counts = defaultdict(Counter)

    with open(infile) as f:
        for line in f:
            line = preprocess_line(line)
            for i in range(len(line) - 2):
                bigram = (line[i], line[i + 1])
                trigram = (line[i], line[i + 1], line[i + 2])
                trigram_counts[bigram][trigram[2]] += 1
                bigram_counts[bigram][trigram[2]] += 1

    with open(outfile, 'w') as f_out:
        for bigram in trigram_counts:
            total_bigram_count = sum(bigram_counts[bigram].values())
            for char, count in trigram_counts[bigram].items():
                prob = count / total_bigram_count
                f_out.write(f"{bigram[0]}{bigram[1]} -> {char}: {prob:.6f}\n")


"""
4. Generating from models
"""
def generate_from_LM(model_file, sequence_length=300):
    language_model = {}
    with open(model_file) as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) != 2:
                    continue
                trigram = parts[0].strip()
                probability = parts[1].strip()
                if len(trigram) != 3:
                    continue
                history = trigram[:2]
                next_char = trigram[2]
                if history not in language_model:
                    language_model[history] = []
                language_model[history].append((next_char, probability))

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


"""
|====================MAIN SCRIPT=======================|
| Usage: python main.py <training_file> <output_file>  |
|======================================================|
"""
# Test model_training()
# if len(sys.argv) != 3:
#     print("Usage: ", sys.argv[0], "<training_file> <output_file>")
#     sys.exit(1)
#
# infile = sys.argv[1]
# outfile = sys.argv[2]
# model_training(infile, outfile)

# Test generate_from_LM()
print(generate_from_LM("assignment1-data/model-br.en", 300))
