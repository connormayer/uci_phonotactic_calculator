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

    # split using default e.g. s.split() not split(" ")
    # if last part of split is a float, we have frequencies and can
    #   do log-prob with tokens
    #   - map word to frequency
    #   - loop through words and add freq of unigrams to dict
    #   - calculate log prob as below (unigram_probs)
    line_splits = [s.split() for s in tokens.split("\n") if s]
    unigram_token_freqs = defaultdict(float)
    bigram_token_freqs = defaultdict(float)
    try:
        tokens = []
        for line in line_splits:
            freq = float(line[-1]) # might not work if frequencies are not given
            word = line[:-1]
            for segment in word:
                unigram_token_freqs[segment] += np.log(freq)
            word = ['#'] + word + ['#']
            for i in range(len(word)-1):
                bigram = (word[i], word[i+1])
                bigram_token_freqs[bigram] += np.log(freq)
            tokens.append(word)
        total_count_uni = sum(unigram_token_freqs.values())
        total_count_bi = sum(bigram_token_freqs.values())
        unigram_token_probs = {key : np.log(value / total_count_uni) for key, value in unigram_token_freqs.items()}
        bigram_token_probs = {key : np.log(value / (unigram_token_freqs[key[0]] if key[0] != '#' else total_count_uni)) \
            for key, value in bigram_token_freqs.items() }
    except:
        unigram_token_probs = dict()
        bigram_token_probs = dict()
        tokens = line_splits




    #tokens = [s.split(" ") for s in tokens.split("\n") if s]

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

    return unigram_probs, bigram_probs, sound_idx, unigram_token_probs, bigram_token_probs

def build_positional_models(dataset):
    with open(dataset, 'r') as f:
        tokens = f.read()

    tokens = [s.split() for s in tokens.split("\n") if s]
    try:
        new_tokens = []
        for t in tokens:
            freq = float(t[-1])
            new_tokens.append((t[:-1], freq))
        tokens = new_tokens
    except:
        pass

    pos_unigram_freqs = defaultdict(lambda: defaultdict(int))
    pos_bigram_freqs = defaultdict(lambda: defaultdict(int))

    pos_unigram_tok_freqs = defaultdict(lambda: defaultdict(int))
    pos_bigram_tok_freqs = defaultdict(lambda: defaultdict(int))

    for element in tokens:
        if len(element) == 2:
            token,freq = element
        else:
            token = element
            freq = None
        for idx, sound in enumerate(token):
            pos_unigram_freqs[idx][sound] += 1
            if freq:
                pos_unigram_tok_freqs[idx][sound] += np.log(freq)
            if idx < len(token) - 1:
                pos_bigram_freqs[(idx, idx + 1)][(sound, token[idx + 1])] += 1
                if freq:
                    pos_bigram_tok_freqs[(idx, idx + 1)][(sound, token[idx + 1])] += np.log(freq)
    
    for idx in pos_unigram_freqs.keys():
        total = sum(pos_unigram_freqs[idx].values())
        for sound in pos_unigram_freqs[idx].keys():
            pos_unigram_freqs[idx][sound] /= total
    
    for idx in pos_bigram_freqs.keys():
        total = sum(pos_bigram_freqs[idx].values())
        for sounds in pos_bigram_freqs[idx].keys():
            pos_bigram_freqs[idx][sounds] /= total

    if len(pos_unigram_tok_freqs) != 0:
        for idx in pos_unigram_tok_freqs.keys():
            total = sum(pos_unigram_tok_freqs[idx].values())
            for sound in pos_unigram_tok_freqs[idx].keys():
                pos_unigram_tok_freqs[idx][sound] /= total

    if len(pos_bigram_tok_freqs) != 0:
        for idx in pos_bigram_tok_freqs.keys():
            total = sum(pos_bigram_tok_freqs[idx].values())
            for sound in pos_bigram_tok_freqs[idx].keys():
                pos_bigram_tok_freqs[idx][sound] /= total

    return pos_unigram_freqs, pos_bigram_freqs, pos_unigram_tok_freqs, pos_bigram_tok_freqs

def get_unigram_prob(word, unigram_probs):
    prob = 0
    for sound in word:
        '''
        if sound in unigram_probs:
            prob += unigram_probs[sound]
        else:
            prob += float('-inf')
        '''
        # Add basic smoothing for sounds that appear in test data but
        #   not training data. Use float('-inf') because we are
        #   using log probabilities --> log(-inf) = 0
        prob += unigram_probs.get(sound, float('-inf'))

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
    score = 0

    for idx, sound in enumerate(word):
        score += pos_uni_tok_freqs[idx][sound]

    return score

def get_pos_bigram_tok_score(word, pos_bi_tok_freqs):
    score = 0

    for idx, sound in enumerate(word):
        if idx < len(word) - 1:
            score += pos_bi_tok_freqs[(idx, idx + 1)][sound, word[idx + 1]]

    return score


def score_corpus(dataset, outfile, unigram_probs, bigram_probs, pos_uni_freqs, pos_bi_freqs, sound_idx, unigram_token_probs, \
    bigram_token_probs, pos_uni_tok_freqs, pos_bi_tok_freqs):
    with open(dataset, 'r') as f:
        tokens = f.read()

    tokens = [s.split(" ") for s in tokens.split("\n") if s]

    with open(outfile, 'w') as f:
        f.write('word,word_len,uni_prob,bi_prob,pos_uni_freq,pos_bi_freq,uni_tok_prob,bi_tok_prob,pos_uni_tok,pos_bi_tok\n')
        for token in tokens:
            word_len = len(token)
            unigram_prob = get_unigram_prob(token, unigram_probs)
            bigram_prob = get_bigram_prob(token, bigram_probs, sound_idx)
            pos_uni_score = 1 + get_pos_unigram_score(token, pos_uni_freqs)
            pos_bi_score = 1 + get_pos_bigram_score(token, pos_bi_freqs)
            uni_tok_prob = get_unigram_token_score(token, unigram_token_probs)
            bi_tok_prob = get_bigram_token_score(token, bigram_token_probs)
            pos_uni_tok_score = 1 + get_pos_unigram_tok_score(token, pos_uni_tok_freqs)
            pos_bi_tok_score = 1 + get_pos_bigram_tok_score(token, pos_bi_tok_freqs)
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

    