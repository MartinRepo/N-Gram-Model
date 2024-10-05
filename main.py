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
def generate_from_LM(model_file, seed, n):
    pass
"""
5. Computing perplexity
"""


"""
|====================MAIN SCRIPT=======================|
| Usage: python main.py <training_file> <output_file>  |
|======================================================|
"""
if len(sys.argv) != 3:
    print("Usage: ", sys.argv[0], "<training_file> <output_file>")
    sys.exit(1)

infile = sys.argv[1]
outfile = sys.argv[2]
model_training(infile, outfile)
