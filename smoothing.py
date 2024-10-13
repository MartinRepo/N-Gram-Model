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


"""
1. 计数从1开始，模型的混淆度更小 47.2
2. 计数从0开始，即真实计数，并且根据goodTuring的方法给unseen数据一个概率，即N1/N0。那么模型的混淆度要更大 47.9
3. 但不管怎么样，混淆度都要大于已给的模型。
4. 之前混淆度在5-6是因为混淆度的计算还存在问题，对于概率为0的值，依然log值为1
5. 现在的混淆度计算函数已经完全解决了这个问题，每个模型中都不存在概率为0的值了。（概率为0本身就是不应该存在的，因为会无法计算混淆度，也表示模型的训练数据过于稀疏）
"""
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
            adjusted_count = (count + 1) * (freq_of_freqs[count + 1] / freq_of_freqs[count]) if freq_of_freqs[count+1] > 0 else count/total_bigram_count
            adjusted_trigram_probs[bigram][char] = adjusted_count

    # Step 3: Handle unseen trigrams (those with count 0)
    if freq_of_freqs[0] > 0:
        unseen_prob = freq_of_freqs[1] / freq_of_freqs[0]
        for bigram in bigram_counts:
            for char in bigram_counts[bigram]:
                if char not in adjusted_trigram_probs[bigram] or adjusted_trigram_probs[bigram][char] == 0:
                    adjusted_trigram_probs[bigram][char] = unseen_prob


    # Normalize the probabilities for each bigram
    for bigram in adjusted_trigram_probs:
        total_prob = sum(adjusted_trigram_probs[bigram].values())
        if total_prob > 0:  # Ensure the total is not zero before normalization
            for char in adjusted_trigram_probs[bigram]:
                adjusted_trigram_probs[bigram][char] /= total_prob
    return adjusted_trigram_probs


def interpolation_smoothing(trigram_counts, bigram_counts, unigram_counts, lambda1=0.998, lambda2=0.001, lambda3=0.001):
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

def kneser_ney_smoothing(trigram_counts, bigram_counts, discount=0.75):
    continuation_counts = Counter()
    bigram_continuation_counts = Counter()
    total_bigrams = 0

    # Calculate continuation counts
    for bigram, following_chars in trigram_counts.items():
        for char in following_chars:
            continuation_counts[char] += 1
        bigram_continuation_counts[bigram] = len(following_chars)
        total_bigrams += len(following_chars)

    # Calculate probabilities with Kneser-Ney smoothing
    trigram_probs = defaultdict(dict)
    for bigram, following_chars in trigram_counts.items():
        bigram_count = sum(following_chars.values())
        for char, count in following_chars.items():
            discounted_count = max(count - discount, 0)
            lambda_factor = (discount / bigram_count) * bigram_continuation_counts[bigram]
            continuation_prob = continuation_counts[char] / total_bigrams
            prob = (discounted_count / bigram_count) + (lambda_factor * continuation_prob)
            trigram_probs[bigram][char] = prob

    return trigram_probs