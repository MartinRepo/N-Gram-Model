from collections import defaultdict, Counter

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
    # Step 1: Counting the number of occurrences of each frequency (N_k)
    count_of_counts = Counter()
    for bigram in trigram_counts:
        for count in trigram_counts[bigram].values():
            count_of_counts[count] += 1

    # Step 2: Compute adjusted counts using Good-Turing formula
    adjusted_trigram_probs = defaultdict(dict)

    for bigram in trigram_counts:
        total_bigram_count = sum(bigram_counts[bigram].values())  # Total count of the bigram

        if total_bigram_count == 0:
            continue  # Skip if no counts for this bigram

        for char, count in trigram_counts[bigram].items():
            adjusted_count = (count + 1) * (count_of_counts[count + 1] / count_of_counts[count]) if count_of_counts[count+1] > 0 else count
            adjusted_trigram_probs[bigram][char] = adjusted_count

    # Step 3: Handle unseen trigrams (those with count 0)
    if count_of_counts[0] > 0:
        unseen_prob = count_of_counts[1] / count_of_counts[0]
        for bigram in bigram_counts:
            for char in bigram_counts[bigram]:
                if char not in adjusted_trigram_probs[bigram] or adjusted_trigram_probs[bigram][char] == 0:
                    adjusted_trigram_probs[bigram][char] = unseen_prob


    # Normalize the probabilities for each bigram
    for bigram in adjusted_trigram_probs:
        total_prob = sum(adjusted_trigram_probs[bigram].values())
        for char in adjusted_trigram_probs[bigram]:
            adjusted_trigram_probs[bigram][char] /= total_prob
    return adjusted_trigram_probs


def interpolation_smoothing(trigram_counts, bigram_counts, unigram_counts, lambda1=0.9997, lambda2=0.0002, lambda3=0.0001):
    trigram_probs = {}
    total_unigrams = sum(unigram_counts.values())

    for bigram, following_chars in trigram_counts.items():
        trigram_probs[bigram] = {}
        bigram_count = sum(bigram_counts[bigram].values())
        for char, trigram_count in following_chars.items():
            unigram_count = unigram_counts[char]
            if trigram_count > 0:
                trigram_probs[bigram][char] = lambda1*(trigram_count / bigram_count) + lambda2*(bigram_count / unigram_count) + lambda3*(unigram_count / total_unigrams)
            elif bigram_count > 0:
                trigram_probs[bigram][char] = lambda2*(bigram_count / unigram_count) + lambda3*(unigram_count / total_unigrams)
            else:
                trigram_probs[bigram][char] = lambda3*(unigram_count / total_unigrams)

    for bigram in trigram_probs:
        total_prob = sum(trigram_probs[bigram].values())
        if total_prob > 0:
            for char in trigram_probs[bigram]:
                trigram_probs[bigram][char] /= total_prob

    return trigram_probs


def back_off_smoothing(trigram_counts, bigram_counts, unigram_counts):
    trigram_probs = {}
    total_unigrams = sum(unigram_counts.values())

    for bigram, following_chars in trigram_counts.items():
        trigram_probs[bigram] = {}
        bigram_count = sum(bigram_counts[bigram].values())
        for char, trigram_count in following_chars.items():
            unigram_count = unigram_counts[char]
            if trigram_count > 0:
                prob = trigram_count / bigram_count
            elif bigram_count > 0:
                prob = 0.001 * (bigram_count / unigram_count)
            else:
                prob = 0.001 * (0.001 * (unigram_count / total_unigrams))

            trigram_probs[bigram][char] = prob

    for bigram in trigram_probs:
        total_prob = sum(trigram_probs[bigram].values())
        if total_prob > 0:
            for char in trigram_probs[bigram]:
                trigram_probs[bigram][char] /= total_prob

    return trigram_probs

