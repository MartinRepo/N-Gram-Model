import itertools
import string
from collections import defaultdict, Counter
from smoothing import simple_probability_estimation, add_alpha_smoothing, good_turing_smoothing
from utils import preprocess_line

def smoothing_model_training(input_file, output_file, smoothingType, alpha=0):
    bigram_counts = defaultdict(Counter)
    trigram_counts = defaultdict(Counter)
    charset = list(string.ascii_lowercase) + ['.', '0', '#', ' ']

    # Generate all possible trigrams, sorted
    all_trigrams = sorted([''.join(trigram) for trigram in itertools.product(charset, repeat=3)
                           if not (trigram[0] == '#' and trigram[2] == '#') and not (
                    trigram[1] == '#' and not (trigram[0] == '#' and trigram[1] == '#'))])

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

    # Choose smoothing method based on smoothingType
    if smoothingType == "simple":
        trigram_probs = simple_probability_estimation(trigram_counts)

    elif smoothingType == "alpha":
        trigram_probs = add_alpha_smoothing(trigram_counts, bigram_counts, alpha)

    elif smoothingType == "goodTuring":
        trigram_probs = good_turing_smoothing(trigram_counts, bigram_counts)

    # Write trigram probabilities to output file
    with open(output_file, 'w') as f_out:
        for trigram in all_trigrams:
            bigram = (trigram[0], trigram[1])
            char = trigram[2]
            prob = trigram_probs.get(bigram, {}).get(char, 0.0)
            f_out.write(f"{trigram} {prob:.3e}\n")