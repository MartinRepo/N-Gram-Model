import string
import itertools
from collections import defaultdict, Counter
from utils import split_corpus, write_model_to_file, compute_perplexity, preprocess_line
from smoothing_methods import add_alpha_smoothing
from matplotlib import pyplot as plt


def model_training_with_add_alpha(input_file, output_file):
    train_set, dev_set, test_set = split_corpus(input_file)
    # write dev_set to file
    with open("temp/dev_set.out", 'w') as f:
        for line in dev_set:
            f.write(line)
    bigram_counts = defaultdict(Counter)
    trigram_counts = defaultdict(Counter)
    charset = list(string.ascii_lowercase) + ['.', '0', '#', ' ']

    # Generate all possible trigrams, sorted
    all_trigrams = sorted([''.join(trigram) for trigram in itertools.product(charset, repeat=3)
                           if not (trigram[0] == '#' and trigram[2] == '#') and not (
                    trigram[1] == '#' and not (trigram[0] == '#' and trigram[1] == '#'))])
    for trigram in all_trigrams:
        bigram = (trigram[0], trigram[1])
        trigram_counts[bigram][trigram[2]] = 0
        bigram_counts[bigram][trigram[2]] = 0

    for line in train_set:
        line = preprocess_line(line)
        for i in range(len(line) - 2):
            bigram = (line[i], line[i + 1])
            trigram = (line[i], line[i + 1], line[i + 2])
            trigram_counts[bigram][trigram[2]] += 1
            bigram_counts[bigram][trigram[2]] += 1

    alpha_values = [round(a * 0.001, 3) for a in range(1, 1001)]
    perplexity_set = []
    best_alpha = None
    best_perplexity = float('inf')
    best_trigram_probs = None
    for alpha in alpha_values:
        trigram_probs = add_alpha_smoothing(trigram_counts, bigram_counts, alpha)
        write_model_to_file(all_trigrams, trigram_probs, "temp/temp_alpha.out")
        # Compute perplexity
        perplexity = compute_perplexity("temp/dev_set.out", "temp/temp_alpha.out")
        perplexity_set.append(perplexity)
        if perplexity < best_perplexity:
            best_perplexity = perplexity
            best_alpha = alpha
            best_trigram_probs = trigram_probs
    plt.plot(alpha_values, perplexity_set)
    plt.title("Perplexities within different alpha values")
    plt.xlabel("Alpha")
    plt.ylabel("Perplexity")
    plt.scatter(best_alpha, best_perplexity, color='red', zorder=5)
    plt.annotate(f'Best alpha: {best_alpha}\nBest perplexity: {best_perplexity}',
                 xy=(best_alpha, best_perplexity),
                 xytext=(best_alpha + 0.05, best_perplexity + 0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.show()
    print(f"Best alpha: {best_alpha}, Best Perplexity: {best_perplexity}")
    # Write the best model to output file
    write_model_to_file(all_trigrams, best_trigram_probs, output_file)