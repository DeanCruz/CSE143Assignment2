import nltk
import numpy as np
from nltk import FreqDist
from nltk.util import ngrams
from collections import defaultdict
import os
from collections import defaultdict


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [nltk.word_tokenize(line) for line in f.readlines()]
    return data

def count_bigrams(data):
    bigram_freqdist = defaultdict(lambda: defaultdict(int))

    for sentence in data:
        for i in range(len(sentence) - 1):
            bigram_freqdist[sentence[i]][sentence[i + 1]] += 1

    return bigram_freqdist

def calculate_freq_dists(data):
    unigrams = [gram for line in data for gram in ngrams(line, 1)]
    bigrams = count_bigrams(data) 
    trigrams = [gram for line in data for gram in ngrams(line, 3, pad_left=True, pad_right=True)]

    unigram_freqdist = nltk.FreqDist(unigrams)
    trigram_freqdist = nltk.FreqDist(trigrams)

    return unigram_freqdist, bigrams, trigram_freqdist

def calculate_perplexity(unigram_freqdist, bigram_freqdist, trigram_freqdist, data):
    unigram_pplx = calculate_unigram_perplexity(unigram_freqdist, data)
    bigram_pplx = calculate_bigram_perplexity(bigram_freqdist, unigram_freqdist, data)
    trigram_pplx = calculate_trigram_perplexity(trigram_freqdist, bigram_freqdist, data)
    
    return unigram_pplx, bigram_pplx, trigram_pplx

def calculate_unigram_perplexity(unigram_freqdist, data):
    total_logprob = 0
    N = 0
    for sentence in data:
        for word in sentence:
            N += 1
            try:
                # Calculate the probability of the word
                word_prob = unigram_freqdist.freq(word)
                # Add the log probability of the word to the total
                total_logprob += np.log(word_prob)
            except ValueError:
                # If the word is not in the unigram_freqdist, it is considered as <UNK>
                word_prob = unigram_freqdist.freq('<UNK>')
                total_logprob += np.log(word_prob)

    # Calculate perplexity
    perplexity = np.exp(-total_logprob/N)

    return perplexity

def calculate_bigram_perplexity(bigram_freqdist, unigram_freqdist, data):
    total_logprob = 0
    N = 0
    for sentence in data:
        prev_word = None
        for word in sentence:
            if prev_word is not None:
                N += 1
                try:
                    # Calculate the probability of the word
                    if unigram_freqdist[prev_word] == 0:
                        if '<UNK>' in unigram_freqdist:
                            word_prob = bigram_freqdist[prev_word]['<UNK>'] / unigram_freqdist['<UNK>']
                        else:
                            word_prob = 0  # Assign a zero probability if '<UNK>' is not present
                    else:
                        word_prob = bigram_freqdist[prev_word][word] / unigram_freqdist[prev_word]
                    # Add the log probability of the word to the total
                    if word_prob != 0:  # Check for zero probability
                        total_logprob += np.log(word_prob)
                    else:
                        total_logprob += np.log(1e-10)  # Assign a small value for zero probability
                except (KeyError, ZeroDivisionError):
                    # If the bigram or unigram frequency is missing, use <UNK> probability
                    if '<UNK>' in bigram_freqdist[prev_word]:
                        word_prob = bigram_freqdist[prev_word]['<UNK>'] / unigram_freqdist['<UNK>']
                    else:
                        word_prob = 0  # Assign a zero probability if '<UNK>' is not present
                    total_logprob += np.log(word_prob)
            prev_word = word

    # Calculate perplexity
    perplexity = np.exp(-total_logprob / N)

    return perplexity


def calculate_trigram_perplexity(trigram_freqdist, bigram_freqdist, data):
    total_logprob = 0
    N = 0
    for sentence in data:
        prev_word1 = None
        prev_word2 = None
        for word in sentence:
            if prev_word1 is not None and prev_word2 is not None:
                N += 1
                try:
                    # Check if the trigram exists in the frequency distribution
                    if word in trigram_freqdist[(prev_word1, prev_word2)]:
                        # Calculate the probability of the word
                        word_prob = trigram_freqdist[(prev_word1, prev_word2)][word] / bigram_freqdist[prev_word1][prev_word2]
                        # Add the log probability of the word to the total
                        if word_prob != 0:  # Check for zero denominator
                            total_logprob += np.log(word_prob)
                        else:
                            total_logprob += np.log(1e-10)  # Assign a small value for zero probability
                    else:
                        # If the trigram is not in the trigram_freqdist, it is considered as <UNK>
                        word_prob = trigram_freqdist[(prev_word1, prev_word2)]['<UNK>'] / bigram_freqdist[prev_word1][prev_word2]
                        if word_prob != 0:  # Check for zero denominator
                            total_logprob += np.log(word_prob)
                        else:
                            total_logprob += np.log(1e-10)  # Assign a small value for zero probability
                except KeyError:
                    # If the trigram is not found, assign a small probability value
                    total_logprob += np.log(1e-10)
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
