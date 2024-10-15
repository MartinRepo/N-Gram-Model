import string
import itertools
import pandas as pd
from collections import defaultdict, Counter
from utils import split_corpus, write_model_to_file, compute_perplexity, preprocess_line, frange
from smoothing_methods import interpolation_smoothing
from matplotlib import pyplot as plt

def model_training_with_interpolation(input_file, output_file):
    train_set, dev_set, test_set = split_corpus(input_file)
    # write dev_set to file
    with open("temp/dev_set.out", 'w') as f:
        for line in dev_set:
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

    for line in train_set:
        line = preprocess_line(line)
        for i in range(len(line) - 2):
            unigram_counts[line[i]] += 1
            bigram = (line[i], line[i + 1])
            trigram = (line[i], line[i + 1], line[i + 2])
            trigram_counts[bigram][trigram[2]] += 1
            bigram_counts[bigram][trigram[2]] += 1

    # Generate lambda values for grid search optimal lambda
    lambda_combinations = []
    for lambda_1 in [round(x, 4) for x in frange(0.9701, 1, 0.0001)]:
        remaining_sum = 1 - lambda_1
        for lambda_2 in [round(x, 4) for x in frange(0, remaining_sum, 0.0001)]:
            lambda_3 = round(remaining_sum - lambda_2, 4)
            if lambda_2 > lambda_3 > 0:
                lambda_combinations.append((lambda_1, lambda_2, lambda_3))

    # Grid search for optimal lambda values
    perplexity_set = []
    best_lambdas = None
    best_perplexity = float('inf')
    best_trigram_probs = None
    for lambdas in lambda_combinations:
        trigram_probs = interpolation_smoothing(trigram_counts, bigram_counts, unigram_counts, lambdas[0], lambdas[1], lambdas[2])
        write_model_to_file(all_trigrams, trigram_probs, "temp/temp_interpolation.out")
        perplexity = compute_perplexity("temp/dev_set.out", "temp/temp_interpolation.out")
        perplexity_set.append(perplexity)
        if perplexity < best_perplexity:
            best_perplexity = perplexity
            best_lambdas = lambdas
            best_trigram_probs = trigram_probs
        print(f"Lambda: {lambdas}, Perplexity: {perplexity}")

    print(f"Best lambdas: {best_lambdas}, Best Perplexity: {best_perplexity}")
    # Write the best model to output file
    write_model_to_file(all_trigrams, best_trigram_probs, output_file)

    # Write lambda values and perplexity to csv file, for plotting
    with open("temp/lambda_perplexity.csv", 'w') as f:
        f.write("lambda_1,lambda_2,lambda_3,perplexity\n")
        for (l1, l2, l3), p in zip(lambda_combinations, perplexity_set):
            f.write(f"{l1},{l2},{l3},{p}\n")
    data = pd.read_csv("temp/lambda_perplexity.csv")

    # Plot lambda values vs perplexity
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # Lambda 1 vs Perplexity
    axs[0].scatter(data['lambda_1'], data['perplexity'], c='blue', alpha=0.6)
    axs[0].set_xlabel('Lambda 1')
    axs[0].set_ylabel('Perplexity')
    axs[0].set_title('Lambda 1 vs Perplexity')

    # Lambda 2 vs Perplexity
    axs[1].scatter(data['lambda_2'], data['perplexity'], c='green', alpha=0.6)
    axs[1].set_xlabel('Lambda 2')
    axs[1].set_ylabel('Perplexity')
    axs[1].set_title('Lambda 2 vs Perplexity')

    # Lambda 3 vs Perplexity
    axs[2].scatter(data['lambda_3'], data['perplexity'], c='red', alpha=0.6)
    axs[2].set_xlabel('Lambda 3')
    axs[2].set_ylabel('Perplexity')
    axs[2].set_title('Lambda 3 vs Perplexity')

    plt.tight_layout()
    plt.show()
    best_row = data.loc[data['perplexity'].idxmin()]

    plt.figure(figsize=(10, 6))
    plt.scatter(data['lambda_1'], data['perplexity'], label='All Combinations')
    plt.scatter(best_row['lambda_1'], best_row['perplexity'], color='red', label='Best Lambda', s=100)
    plt.xlabel('Lambda 1')
    plt.ylabel('Perplexity')
    plt.title('Lambda 1 vs Perplexity with Best Combination Highlighted')
    plt.legend()
    plt.show()