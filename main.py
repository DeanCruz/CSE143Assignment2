import nltk
import numpy as np
from nltk import FreqDist
from nltk.util import ngrams
from collections import defaultdict
import os
from collections import defaultdict

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [['<START>'] + nltk.word_tokenize(line) + ['<STOP>'] for line in f.readlines()]
    return data

def calculate_freq_dists(data):
    unigram_freqdist = nltk.FreqDist()
    bigram_freqdist = defaultdict(lambda: defaultdict(int))
    trigram_freqdist = defaultdict(lambda: defaultdict(int))
    
    for sentence in data:
        unigram_freqdist.update(ngrams(sentence, 1))
        for i in range(len(sentence) - 1):
            bigram_freqdist[sentence[i]][sentence[i + 1]] += 1
        for i in range(len(sentence) - 2):
            trigram_freqdist[(sentence[i], sentence[i + 1])][sentence[i + 2]] += 1

    return unigram_freqdist, bigram_freqdist, trigram_freqdist

def calculate_perplexity(unigram_freqdist, bigram_freqdist, trigram_freqdist, data):
    unigram_pplx = calculate_unigram_perplexity(unigram_freqdist, data)
    bigram_pplx = calculate_bigram_perplexity(bigram_freqdist, unigram_freqdist, data)
    trigram_pplx = calculate_trigram_perplexity(trigram_freqdist, bigram_freqdist, data)
    
    return unigram_pplx, bigram_pplx, trigram_pplx

def calculate_unigram_perplexity(unigram_freqdist, data):
    total_logprob = 0
    N = sum(unigram_freqdist.values())
    for unigram in unigram_freqdist:
        word_prob = unigram_freqdist.freq(unigram)
        total_logprob += unigram_freqdist[unigram] * np.log(word_prob)

    return np.exp(-total_logprob/N)

def calculate_bigram_perplexity(bigram_freqdist, unigram_freqdist, data):
    total_logprob = 0
    N = 0
    V = len(unigram_freqdist)  # size of vocabulary
    for sentence in data:
        prev_word = None
        for word in sentence:
            if prev_word is not None:
                N += 1
                # Add one for smoothing and calculate the probability of the word
                word_prob = (bigram_freqdist[prev_word][word] + 1) / (unigram_freqdist[prev_word] + V)
                # Add the log probability of the word to the total
                total_logprob += np.log(word_prob)
            prev_word = word

    # Calculate perplexity
    perplexity = np.exp(-total_logprob / N)

    return perplexity

def calculate_trigram_perplexity(trigram_freqdist, bigram_freqdist, data):
    total_logprob = 0
    N = 0
    V = len(bigram_freqdist)  # size of vocabulary
    for sentence in data:
        prev_word1 = None
        prev_word2 = None
        for word in sentence:
            if prev_word1 is not None and prev_word2 is not None:
                N += 1
                # Add one for smoothing and calculate the probability of the word
                if prev_word2 == '<START>':
                    # Use bigram probability for the word immediately after <START>
                    word_prob = (bigram_freqdist[prev_word2][word] + 1) / (unigram_freqdist[prev_word2] + V)
                else:
                    word_prob = (trigram_freqdist.get(prev_word1, {}).get(prev_word2, {}).get(word, 0) + 1) / (bigram_freqdist.get(prev_word1, {}).get(prev_word2, 0) + V)
                # Add the log probability of the word to the total
                total_logprob += np.log(word_prob)
            prev_word1, prev_word2 = prev_word2, word

    # Calculate perplexity
    perplexity = np.exp(-total_logprob / N)

    return perplexity

if __name__ == "__main__":
    nltk.download('punkt')
    
    train_data = load_data(os.path.join("A2-data", "1b_benchmark.train.tokens"))
    dev_data = load_data(os.path.join("A2-data", "1b_benchmark.dev.tokens"))
    test_data = load_data(os.path.join("A2-data", "1b_benchmark.test.tokens"))
    
    unigram_freqdist, bigram_freqdist, trigram_freqdist = calculate_freq_dists(train_data)
    
    unigram_train_perplexity, bigram_train_perplexity, trigram_train_perplexity = calculate_perplexity(unigram_freqdist, bigram_freqdist, trigram_freqdist, train_data)
    unigram_dev_perplexity, bigram_dev_perplexity, trigram_dev_perplexity = calculate_perplexity(unigram_freqdist, bigram_freqdist, trigram_freqdist, dev_data)
    unigram_test_perplexity, bigram_test_perplexity, trigram_test_perplexity = calculate_perplexity(unigram_freqdist, bigram_freqdist, trigram_freqdist, test_data)

    print(f'Train Data Perplexity: Unigram={unigram_train_perplexity}, Bigram={bigram_train_perplexity}, Trigram={trigram_train_perplexity}')
    print(f'Dev Data Perplexity: Unigram={unigram_dev_perplexity}, Bigram={bigram_dev_perplexity}, Trigram={trigram_dev_perplexity}')
    print(f'Test Data Perplexity: Unigram={unigram_test_perplexity}, Bigram={bigram_test_perplexity}, Trigram={trigram_test_perplexity}')
