import nltk
import numpy as np

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

def build_ngram_models(dataset):
    with open(dataset, 'r') as f:
        tokens = f.read()

    tokens = set([tuple(s.split(" ")) for s in tokens.split("\n") if s])
    tokens = [list(token) for token in tokens]

    unique_sounds = set(
        [item for sublist in tokens for item in sublist]
    )
    sound_idx = sorted(list(unique_sounds)) + ['#']

    # Get unigram probabilities
    flat_tokens = [segment for word in tokens for segment in word]
    unigram_freq = nltk.FreqDist(flat_tokens)
    total_sounds = sum(unigram_freq.values())
    unigram_probs = {key: (np.log(value/total_sounds)) for key, value in unigram_freq.items()}

    # Get bigram probabilities
    bigrams = generate_bigrams(tokens)
    num_sounds = len(sound_idx)
    count_matrix = np.ones((num_sounds, num_sounds))

    for s1, s2 in bigrams:
        count_matrix[sound_idx.index(s2)][sound_idx.index(s1)] += 1

    bigram_probs = np.log(count_matrix / np.sum(count_matrix, 0))

    return unigram_probs, bigram_probs, sound_idx

def build_positional_models(dataset):
    with open(dataset, 'r') as f:
        tokens = f.read()

    tokens = set([tuple(s.split(" ")) for s in tokens.split("\n") if s])
    tokens = [list(token) for token in tokens]

    pos_unigram_freqs = defaultdict(lambda: defaultdict(int))
    pos_bigram_freqs = defaultdict(lambda: defaultdict(int))

    for token in tokens:
        for idx, sound in enumerate(token):
            pos_unigram_freqs[idx][sound] += 1
            if idx < len(token) - 1:
                pos_bigram_freqs[(idx, idx + 1)][(sound, token[idx + 1])] += 1
    
    for idx in pos_unigram_freqs.keys():
        total = sum(pos_unigram_freqs[idx].values())
        for sound in pos_unigram_freqs[idx].keys():
            pos_unigram_freqs[idx][sound] /= total
    
    for idx in pos_bigram_freqs.keys():
        total = sum(pos_bigram_freqs[idx].values())
        for sounds in pos_bigram_freqs[idx].keys():
            pos_bigram_freqs[idx][sounds] /= total

    return pos_unigram_freqs, pos_bigram_freqs

def get_unigram_prob(word, unigram_probs):
    prob = 0
    for sound in word:
        prob += unigram_probs[sound]

    return prob

def get_bigram_prob(word, bigram_probs, sound_idx):
    bigrams = generate_bigrams([word])
    prob = 0
    for s1, s2 in bigrams:
        prob += bigram_probs[sound_idx.index(s2), sound_idx.index(s1)]
    return prob

def get_pos_unigram_score(word, pos_uni_freqs):
    score = 0

    for idx, sound in enumerate(word):
        score += pos_uni_freqs[idx][sound]

    return score

def get_pos_bigram_score(word, pos_bi_freqs):
    score = 0

    for idx, sound in enumerate(word):
        if idx < len(word) - 1:
            score += pos_bi_freqs[(idx, idx + 1)][sound, word[idx + 1]]

    return score

def score_corpus(dataset, outfile, unigram_probs, bigram_probs, pos_uni_freqs, pos_bi_freqs, sound_idx):
    with open(dataset, 'r') as f:
        tokens = f.read()

    tokens = set([tuple(s.split(" ")) for s in tokens.split("\n") if s])
    tokens = [list(token) for token in tokens]

    with open(outfile, 'w') as f:
        f.write('word,word_len,uni_prob,bi_prob,pos_uni_freq,pos_bi_freq\n')
        for token in tokens:
            word_len = len(token)
            unigram_prob = get_unigram_prob(token, unigram_probs)
            bigram_prob = get_bigram_prob(token, bigram_probs, sound_idx)
            pos_uni_score = 1 + get_pos_unigram_score(token, pos_uni_freqs)
            pos_bi_score = 1 + get_pos_bigram_score(token, pos_bi_freqs)
            f.write('{}\n'.format(','.join([
                ' '.join(token), 
                str(word_len),
                str(unigram_prob), 
                str(bigram_prob), 
                str(pos_uni_score), 
                str(pos_bi_score)
            ])))

def run(train, test, out):
    unigram_probs, bigram_probs, sound_idx = build_ngram_models(train)
    pos_uni_freqs, pos_bi_freqs = build_positional_models(train)

    score_corpus(
        test, out, 
        unigram_probs, bigram_probs, 
        pos_uni_freqs, pos_bi_freqs,
        sound_idx
    )

if __name__ == "__main__":
    train_dataset = '../data/english_training_data.txt'
    test_dataset = '../data/english_test_data.csv'
    outfile = '../english_output.csv'

    run(train_dataset, test_dataset, outfile)

    