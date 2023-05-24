import os
from collections import defaultdict, Counter
import math

def word_tokenize(line):
    return line.strip().split()

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [['<START>'] + word_tokenize(line) + ['<STOP>'] for line in f.readlines()]
    return data

def calculate_freq_dists(data):
    unigram_freqdist = Counter()
    bigram_freqdist = defaultdict(lambda: defaultdict(int))
    trigram_freqdist = defaultdict(lambda: defaultdict(int))

    for sentence in data:
        unigram_freqdist.update(sentence)
        for i in range(len(sentence) - 1):
            bigram_freqdist[sentence[i]][sentence[i + 1]] += 1
        for i in range(len(sentence) - 2):
            trigram_freqdist[(sentence[i], sentence[i + 1])][sentence[i + 2]] += 1

    return unigram_freqdist, bigram_freqdist, trigram_freqdist

def calculate_unigram_perplexity(unigram_freqdist, data):
    total_logprob = 0
    N = sum(unigram_freqdist.values())
    for word, count in unigram_freqdist.items():
        word_prob = count / N
        total_logprob += count * math.log(word_prob)
    return math.exp(-total_logprob/N)

def calculate_bigram_perplexity(bigram_freqdist, unigram_freqdist, data):
    total_logprob = 0
    N = 0
    V = len(unigram_freqdist)
    for sentence in data:
        prev_word = None
        for word in sentence:
            if prev_word is not None:
                N += 1
                word_prob = (bigram_freqdist[prev_word][word] + 1) / (unigram_freqdist[prev_word] + V)
                total_logprob += math.log(word_prob)
            prev_word = word
    return math.exp(-total_logprob / N)

def calculate_trigram_perplexity(trigram_freqdist, bigram_freqdist, data):
    total_logprob = 0
    N = 0
    V = len(bigram_freqdist)
    for sentence in data:
        prev_word1 = None
        prev_word2 = None
        for word in sentence:
            if prev_word1 is not None and prev_word2 is not None:
                N += 1
                if prev_word2 == '<START>':
                    word_prob = (bigram_freqdist[prev_word2][word] + 1) / (unigram_freqdist[prev_word2] + V)
                else:
                    word_prob = (trigram_freqdist.get((prev_word1, prev_word2), {}).get(word, 0) + 1) / (bigram_freqdist.get(prev_word1, {}).get(prev_word2, 0) + V)
                total_logprob += math.log(word_prob)
            prev_word1, prev_word2 = prev_word2, word
    return math.exp(-total_logprob / N)

if __name__ == "__main__":
    train_data = load_data(os.path.join("A2-data", "1b_benchmark.train.tokens"))
    dev_data = load_data(os.path.join("A2-data", "1b_benchmark.dev.tokens"))
    test_data = load_data(os.path.join("A2-data", "1b_benchmark.test.tokens"))
    
    unigram_freqdist, bigram_freqdist, trigram_freqdist = calculate_freq_dists(train_data)

    unigram_train_perplexity = calculate_unigram_perplexity(unigram_freqdist, train_data)
    bigram_train_perplexity = calculate_bigram_perplexity(bigram_freqdist, unigram_freqdist, train_data)
    trigram_train_perplexity = calculate_trigram_perplexity(trigram_freqdist, bigram_freqdist, train_data)

    unigram_dev_perplexity = calculate_unigram_perplexity(unigram_freqdist, dev_data)
    bigram_dev_perplexity = calculate_bigram_perplexity(bigram_freqdist, unigram_freqdist, dev_data)
    trigram_dev_perplexity = calculate_trigram_perplexity(trigram_freqdist, bigram_freqdist, dev_data)

    unigram_test_perplexity = calculate_unigram_perplexity(unigram_freqdist, test_data)
    bigram_test_perplexity = calculate_bigram_perplexity(bigram_freqdist, unigram_freqdist, test_data)
    trigram_test_perplexity = calculate_trigram_perplexity(trigram_freqdist, bigram_freqdist, test_data)

    print(f'Train Data Perplexity: Unigram={unigram_train_perplexity}, Bigram={bigram_train_perplexity}, Trigram={trigram_train_perplexity}')
    print(f'Dev Data Perplexity: Unigram={unigram_dev_perplexity}, Bigram={bigram_dev_perplexity}, Trigram={trigram_dev_perplexity}')
    print(f'Test Data Perplexity: Unigram={unigram_test_perplexity}, Bigram={bigram_test_perplexity}, Trigram={trigram_test_perplexity}')
