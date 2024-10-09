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
    # Step 1: Count frequencies of frequencies (N_k)
    freq_of_freqs = Counter()
    for bigram in trigram_counts:
        for count in trigram_counts[bigram].values():
            freq_of_freqs[count] += 1

    # Step 2: Compute adjusted counts using Good-Turing formula
    adjusted_trigram_probs = defaultdict(dict)

    for bigram in trigram_counts:
        total_bigram_count = sum(bigram_counts[bigram].values())  # Total count of the bigram

        if total_bigram_count == 0:
            continue  # Skip if no counts for this bigram

        for char, count in trigram_counts[bigram].items():
            # Handle cases where count + 1 exceeds the available frequency counts
            adjusted_count = (count + 1) * (freq_of_freqs[count + 1] / freq_of_freqs[count]) if freq_of_freqs[
                                                                                                    count] > 0 else 0

            adjusted_trigram_probs[bigram][char] = adjusted_count

    # Step 3: Handle unseen trigrams (those with count 0)
    unseen_prob = freq_of_freqs[1] / sum(freq_of_freqs.values())

    for bigram in bigram_counts:
        total_bigram_count = sum(bigram_counts[bigram].values())

        if total_bigram_count == 0:
            continue  # Skip if no counts for this bigram

        for char in bigram_counts[bigram]:
            if char not in adjusted_trigram_probs[bigram]:
                adjusted_trigram_probs[bigram][char] = unseen_prob / total_bigram_count

    # Normalize the probabilities for each bigram
    for bigram in adjusted_trigram_probs:
        total_prob = sum(adjusted_trigram_probs[bigram].values())

        if total_prob > 0:  # Ensure the total is not zero before normalization
            for char in adjusted_trigram_probs[bigram]:
                adjusted_trigram_probs[bigram][char] /= total_prob

    return adjusted_trigram_probs
