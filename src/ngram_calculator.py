from struct import calcsize

import nltk
import numpy as np
import matplotlib.pyplot as plt
import re

from collections import defaultdict

WORD_BOUNDARY = '#'

def generate_bigrams(tokens):
    bigrams = [
        x for token in tokens
        for x in nltk.ngrams(
            [WORD_BOUNDARY] + token + [WORD_BOUNDARY],
            2
        )
    ]

    return bigrams

def get_token_freqs(tokens, sound_idx):
    # Note that this returns log frequencies
    num_sounds = len(sound_idx)
    unigram_weighted_freqs = defaultdict(int)
    # This isn't really correct... 
    bigram_weighted_freqs = np.ones((num_sounds, num_sounds))
    
    for token, freq in tokens:
        for unigram in token:
            unigram_weighted_freqs[unigram] += np.log(freq)

        bigrams = generate_bigrams([token])

        for s1, s2 in bigrams:
            bigram_weighted_freqs[sound_idx.index(s2)][sound_idx.index(s1)] += np.log(freq)
    
    return unigram_weighted_freqs, bigram_weighted_freqs 

def fit_unigrams(tokens):
    flat_tokens = [segment for word in tokens for segment in word]
    unigram_freq = nltk.FreqDist(flat_tokens)
    total_sounds = sum(unigram_freq.values())
    unigram_probs = {key: (np.log(value/total_sounds)) for key, value in unigram_freq.items()}
    return unigram_probs

def fit_bigrams(tokens, sound_idx):
    bigrams = generate_bigrams(tokens)
    num_sounds = len(sound_idx)
    count_matrix = np.ones((num_sounds, num_sounds))

    for s1, s2 in bigrams:
        count_matrix[sound_idx.index(s2)][sound_idx.index(s1)] += 1

    bigram_probs = np.log(count_matrix / np.sum(count_matrix, 0))
    return bigram_probs

def fit_token_ngrams(unigram_token_freq, bigram_token_freq):
    # Unigram
    total_weighted_sounds = sum(unigram_token_freq.values())
    unigram_token_probs = {
        key: np.log(value/total_weighted_sounds) 
        for key, value in unigram_token_freq.items()
    }
    # Bigrams
    bigram_token_probs = np.log(bigram_token_freq / np.sum(bigram_token_freq, 0))

    return unigram_token_probs, bigram_token_probs

def read_tokens(dataset):
    with open(dataset, 'r') as f:
        tokens = f.read()

    token_freqs = tokens.split('\n')
    token_freqs = [re.split(',|\t', t) for t in token_freqs if t]

    # Tidy this up
    try:
        # If we have frequencies
        token_freqs = [(key.split(' '), float(val)) for (key, val) in token_freqs]
    except:
        # If we don't
        token_freqs = [(key.split(' '), 0) for (key,) in token_freqs]

    tokens = [t[0] for t in token_freqs]
    return tokens, token_freqs

def build_ngram_models(dataset):
    tokens, token_freqs = read_tokens(dataset)

    unique_sounds = set(
        [item for sublist in tokens for item in sublist]
    )
    sound_idx = sorted(list(unique_sounds)) + ['#']

    # Get type-weighted probabilities
    unigram_probs = fit_unigrams(tokens)
    bigram_probs = fit_bigrams(tokens, sound_idx)

    # Get token-weighted probabilities
    unigram_token_freq, bigram_token_freq = get_token_freqs(token_freqs, sound_idx)
    unigram_token_probs, bigram_token_probs = fit_token_ngrams(unigram_token_freq, bigram_token_freq)

    return unigram_probs, bigram_probs, sound_idx, unigram_token_probs, bigram_token_probs

def normalize_positional_counts(counts):
    for idx in counts.keys():
        total = sum(counts[idx].values())
        for gram in counts[idx].keys():
            counts[idx][gram] /= total

    return counts

def build_positional_models(dataset):
    _, token_freqs = read_tokens(dataset)

    pos_unigram_freqs = defaultdict(lambda: defaultdict(int))
    pos_bigram_freqs = defaultdict(lambda: defaultdict(int))

    pos_unigram_tok_freqs = defaultdict(lambda: defaultdict(int))
    pos_bigram_tok_freqs = defaultdict(lambda: defaultdict(int))

    for token, freq in token_freqs:
        for idx, sound in enumerate(token):
            pos_unigram_freqs[idx][sound] += 1
            pos_unigram_tok_freqs[idx][sound] += np.log(freq)
            if idx < len(token) - 1:
                pos_bigram_freqs[(idx, idx + 1)][(sound, token[idx + 1])] += 1
                pos_bigram_tok_freqs[(idx, idx + 1)][(sound, token[idx + 1])] += np.log(freq)

    pos_unigram_freqs = normalize_positional_counts(pos_unigram_freqs)
    pos_bigram_freqs = normalize_positional_counts(pos_bigram_freqs)
    pos_unigram_tok_freqs = normalize_positional_counts(pos_unigram_tok_freqs)
    pos_bigram_tok_freqs = normalize_positional_counts(pos_bigram_tok_freqs)

    return pos_unigram_freqs, pos_bigram_freqs, pos_unigram_tok_freqs, pos_bigram_tok_freqs

def get_unigram_prob(word, unigram_probs):
    prob = 0
    for sound in word:
        # Add basic smoothing for sounds that appear in test data but
        #   not training data. Use float('-inf') because we are
        #   using log probabilities --> log(-inf) = 0
        prob += unigram_probs.get(sound, float('-inf'))

    return prob

def get_bigram_prob(word, bigram_probs, sound_idx):
    bigrams = generate_bigrams([word])
    prob = 0
    for s1, s2 in bigrams:
        try:
            prob += bigram_probs[sound_idx.index(s2), sound_idx.index(s1)]
        except:
            # If bigram contains symbol we haven't seen, assign 0 probability
            prob += float('-inf')

    return prob

def get_pos_unigram_score(word, pos_uni_freqs):
    score = 1

    for idx, sound in enumerate(word):
        score += pos_uni_freqs[idx][sound]

    return score

def get_pos_bigram_score(word, pos_bi_freqs):
    score = 1

    for idx, sound in enumerate(word):
        if idx < len(word) - 1:
            score += pos_bi_freqs[(idx, idx + 1)][sound, word[idx + 1]]

    return score

def get_unigram_token_score(word, unigram_token_probs):
    if not unigram_token_probs:
        return "N/A"
    score = 0
    for sound in word:
        score += unigram_token_probs.get(sound, float('-inf'))
    return score

def get_bigram_token_score(word, bigram_token_probs):
    if not bigram_token_probs:
        return "N/A"
    score = 0
    expanded_word = ['#'] + word + ['#']
    for i in range(len(expanded_word) - 1):
        bigram = (expanded_word[i], expanded_word[i+1])
        score += bigram_token_probs.get(bigram, float('-inf'))
    return score

def get_pos_unigram_tok_score(word, pos_uni_tok_freqs):
    score = 1

    for idx, sound in enumerate(word):
        score += pos_uni_tok_freqs[idx][sound]

    return score

def get_pos_bigram_tok_score(word, pos_bi_tok_freqs):
    score = 1

    for idx, sound in enumerate(word):
        if idx < len(word) - 1:
            score += pos_bi_tok_freqs[(idx, idx + 1)][sound, word[idx + 1]]

    return score

def calculate_rsquared(actual, pred):
    assert len(actual) == len(pred), "Actual and predicted lists need same length"
    # Pearson correlation coefficients between every pair of variables (2 variable: 2x2 matrix)
    #   Values along diagonal are 1 since every variable is perfectly related to itself
    #   R = correlation between variable at indices 0 and 1
    corr_matrix = np.corrcoef(actual, pred)
    R = corr_matrix[0,1]
    return R**2

def plot(X, Y, labelX, labelY, file_name='plot.png'):
    #fig=plt.figure()
    plt.scatter(X,Y,s=5)

    m, b = np.polyfit(X, Y, 1)
    pred_points = [m*x + b for x in X]
    plt.plot(X, pred_points, '-')
    
    plt.xlabel(labelX)
    plt.ylabel(labelY)
    plt.title(f'{labelX} vs {labelY}')

    r_sq = calculate_rsquared(Y, pred_points)

    # place r^2 as annotation
    plt.annotate(f'R^2 = {r_sq:.3f}', xy=(0.75,0.1), xycoords='axes fraction')
    # place regression equation as annotation
    plt.annotate(f'y = {m:.2f}x + {b:.2f}', xy=(0.75,0.05), xycoords='axes fraction')

    plt.savefig(f'media/uploads/{file_name}')

def score_corpus(dataset, outfile, unigram_probs, bigram_probs, pos_uni_freqs, 
                 pos_bi_freqs, sound_idx, unigram_token_probs,
                 bigram_token_probs, pos_uni_tok_freqs, pos_bi_tok_freqs):
    tokens, _ = read_tokens(dataset)

    uni_prob_list = []
    bi_prob_list = []
    pos_uni_score_list = []
    pos_bi_score_list = []
    uni_tok_prob_list = []
    bi_tok_prob_list = []
    pos_uni_tok_score_list = []
    pos_bi_tok_score_list = []

    with open(outfile, 'w') as f:
        f.write(','.join([
            'word',
            'word_len',
            'uni_prob',
            'bi_prob',
            'pos_uni_freq',
            'pos_bi_freq',
            'uni_tok_prob',
            'bi_tok_prob',
            'pos_uni_tok',
            'pos_bi_tok\n'
        ]))

        for token in tokens:
            word_len = len(token)
            unigram_prob = get_unigram_prob(token, unigram_probs)
            bigram_prob = get_bigram_prob(token, bigram_probs, sound_idx)
            pos_uni_score = get_pos_unigram_score(token, pos_uni_freqs)
            pos_bi_score = get_pos_bigram_score(token, pos_bi_freqs)
            uni_tok_prob = get_unigram_prob(token, unigram_token_probs)
            bi_tok_prob = get_bigram_prob(token, bigram_token_probs, sound_idx)
            pos_uni_tok_score = get_pos_unigram_tok_score(token, pos_uni_tok_freqs)
            pos_bi_tok_score = get_pos_bigram_tok_score(token, pos_bi_tok_freqs)

            uni_prob_list.append(unigram_prob)
            bi_prob_list.append(bigram_prob)
            pos_uni_score_list.append(pos_uni_score)
            pos_bi_score_list.append(pos_bi_score)
            uni_tok_prob_list.append(uni_tok_prob)
            bi_tok_prob_list.append(bi_tok_prob)
            pos_uni_tok_score_list.append(pos_uni_tok_score)
            pos_bi_tok_score_list.append(pos_bi_tok_score)

            f.write('{}\n'.format(','.join([
                ' '.join(token), 
                str(word_len),
                str(unigram_prob), 
                str(bigram_prob), 
                str(pos_uni_score), 
                str(pos_bi_score),
                str(uni_tok_prob),
                str(bi_tok_prob),
                str(pos_uni_tok_score),
                str(pos_bi_tok_score)
            ])))

    # plot(uni_prob_list, uni_tok_prob_list, 'Unigram Prob', 'Unigram Tok Prob')

def run(train, test, out):
    unigram_probs, bigram_probs, sound_idx, unigram_token_probs, bigram_token_probs = build_ngram_models(train)
    pos_uni_freqs, pos_bi_freqs, pos_uni_tok_freqs, pos_bi_tok_freqs = build_positional_models(train)

    score_corpus(
        test, out, 
        unigram_probs, bigram_probs, 
        pos_uni_freqs, pos_bi_freqs,
        sound_idx, unigram_token_probs,
        bigram_token_probs, pos_uni_tok_freqs,
        pos_bi_tok_freqs
    )

if __name__ == "__main__":
    train_dataset = 'data/small_cmu_frequencies.txt'
    test_dataset = 'data/english_test_data.csv'
    outfile = 'media/uploads/english_output.csv'

    run(train_dataset, test_dataset, outfile)

    