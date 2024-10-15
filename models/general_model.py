import itertools
import string
from collections import defaultdict, Counter
from smoothing_methods import simple_probability_estimation, add_alpha_smoothing, good_turing_smoothing, back_off_smoothing, interpolation_smoothing
from utils import preprocess_line, split_corpus, write_model_to_file


def model_training(input_file, output_file, smoothingType, alpha=0):
    train_set, dev_set, test_set = split_corpus(input_file)
    with open("temp/test_set.out", 'w') as f:
        for line in test_set:
            f.write(line)

    unigram_counts = Counter()
    bigram_counts = defaultdict(Counter)
    trigram_counts = defaultdict(Counter)
    charset = list(string.ascii_lowercase) + ['.', '0', '#', ' ']

    for char in charset:
        unigram_counts[char] = 0

    # Generate all possible trigrams, sorted
    all_trigrams = sorted([''.join(trigram) for trigram in itertools.product(charset, repeat=3)
                           if not (trigram[0] == '#' and trigram[2] == '#') and not (
                    trigram[1] == '#' and not (trigram[0] == '#' and trigram[1] == '#'))])

    for trigram in all_trigrams:
        bigram = (trigram[0], trigram[1])
        trigram_counts[bigram][trigram[2]] = 0
        bigram_counts[bigram][trigram[2]] = 0

    # Read train_set and populate bigram_counts and trigram_counts
    for line in train_set:
        line = preprocess_line(line)
        for i in range(len(line) - 2):
            unigram_counts[line[i]] += 1
            bigram = (line[i], line[i + 1])
            trigram = (line[i], line[i + 1], line[i + 2])
            trigram_counts[bigram][trigram[2]] += 1
            bigram_counts[bigram][trigram[2]] += 1

    # Choose smoothing method based on smoothingType
    if smoothingType == "simple":
        trigram_probs = simple_probability_estimation(trigram_counts)

    elif smoothingType == "add-alpha":
        trigram_probs = add_alpha_smoothing(trigram_counts, bigram_counts, alpha)

    elif smoothingType == "good-turing":
        trigram_probs = good_turing_smoothing(trigram_counts, bigram_counts)

    elif smoothingType == "backoff":
        trigram_probs = back_off_smoothing(trigram_counts, bigram_counts, unigram_counts)

    elif smoothingType == "interpolation":
        trigram_probs = interpolation_smoothing(trigram_counts, bigram_counts, unigram_counts)

    # Write trigram probabilities to output file
    write_model_to_file(all_trigrams, trigram_probs, output_file)