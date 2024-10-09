import itertools
import string
from utils import preprocess_line
from collections import defaultdict, Counter


def backoff_model_training(input_file, output_file):
    bigram_counts = defaultdict(Counter)
    trigram_counts = defaultdict(Counter)
    charset = list(string.ascii_lowercase) + ['.', '0', '#', ' ']

    # Generate all possible trigrams, sorted
    all_trigrams = sorted([''.join(trigram) for trigram in itertools.product(charset, repeat=3)
                           if not (trigram[0] == '#' and trigram[2] == '#') and not (
                    trigram[1] == '#' and not (trigram[0] == '#' and trigram[1] == '#'))])

    # Initialize all trigram counts to 1 for smoothing
    for trigram in all_trigrams:
        bigram = (trigram[0], trigram[1])
        trigram_counts[bigram][trigram[2]] = 1
        bigram_counts[bigram][trigram[2]] = 1

    # Read input file and populate bigram_counts and trigram_counts
    with open(input_file) as f:
        for line in f:
            line = preprocess_line(line)  # You need to define preprocess_line function
            for i in range(len(line) - 2):
                bigram = (line[i], line[i + 1])
                trigram = (line[i], line[i + 1], line[i + 2])
                trigram_counts[bigram][trigram[2]] += 1
                bigram_counts[bigram][trigram[2]] += 1

    # Calculate backoff probabilities
    trigram_probs = {}
    for bigram, counter in trigram_counts.items():
        trigram_probs[bigram] = {}
        total_count = sum(counter.values())
        for char, count in counter.items():
            trigram_probs[bigram][char] = count / total_count

    bigram_probs = {}
    for bigram, counter in bigram_counts.items():
        bigram_probs[bigram] = {}
        total_count = sum(counter.values())
        for char, count in counter.items():
            bigram_probs[bigram][char] = count / total_count

    # Write backoff probabilities to output file
    with open(output_file, 'w') as f_out:
        for trigram in all_trigrams:
            bigram = (trigram[0], trigram[1])
            char = trigram[2]
            prob = trigram_probs.get(bigram, {}).get(char, 0.0)
            if prob == 0.0:
                prob = bigram_probs.get(bigram, {}).get(char, 0.0)
            f_out.write(f"{trigram} {prob:.3e}\n")

# Note: You need to implement preprocess_line function.