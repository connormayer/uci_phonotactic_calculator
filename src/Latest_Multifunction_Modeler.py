import os
from collections import defaultdict
import itertools
import csv

class LanguageModel:
    def __init__(self, data, lexicon, model_type='conditional', use_word_boundaries=True, positional=True):
        self.data = data
        self.lexicon = lexicon
        self.model_type = model_type  # 'joint' or 'conditional'
        self.use_word_boundaries = use_word_boundaries
        self.positional = positional
        
        if use_word_boundaries:
            # Split text into words
            self.tokens = data.split()
        else:
            # Treat text as a continuous sequence (character-level n-grams)
            self.tokens = list(data)
    
    def build_model(self):
        if self.model_type == 'joint':
            return self._build_joint_model()
        elif self.model_type == 'conditional':
            return self._build_conditional_model()
    
    def _build_joint_model(self):
        """Build a joint probability model with normalized probabilities."""
        ngram_freq = defaultdict(int)
        unigram_freq = defaultdict(int)
        
        # Generate unigrams, bigrams, and trigrams
        total_tokens = len(self.tokens)
        lexicon_hits = defaultdict(int)
        
        for i in range(total_tokens):
            unigram = self.tokens[i]
            unigram_freq[unigram] += 1
            
            # Check if unigram is in lexicon
            if unigram in self.lexicon:
                lexicon_hits[unigram] += 1
            
            if i < total_tokens - 1:
                bigram = (self.tokens[i], self.tokens[i + 1]) if self.positional else tuple(sorted([self.tokens[i], self.tokens[i + 1]]))
                ngram_freq[bigram] += 1
            if i < total_tokens - 2:
                trigram = (self.tokens[i], self.tokens[i + 1], self.tokens[i + 2]) if self.positional else tuple(sorted([self.tokens[i], self.tokens[i + 1], self.tokens[i + 2]]))
                ngram_freq[trigram] += 1

        # Total counts for normalization
        total_unigrams = sum(unigram_freq.values())
        total_bigrams = sum(freq for ngram, freq in ngram_freq.items() if len(ngram) == 2)
        total_trigrams = sum(freq for ngram, freq in ngram_freq.items() if len(ngram) == 3)

        # Prepare data for CSV output
        rows = []
        rows.append(['Model Type', 'Joint'])
        rows.append(['Use Word Boundaries', self.use_word_boundaries])
        rows.append(['Positional', self.positional])
        rows.append(['Total Tokens', total_tokens])

        # Lexicon Hits
        rows.append([])
        rows.append(['Lexicon Hits'])
        for lex_word, count in lexicon_hits.items():
            rows.append([lex_word, count])

        # Unigram Probabilities
        rows.append([])
        rows.append(['Unigram Probabilities'])
        for unigram, freq in unigram_freq.items():
            prob = freq / total_unigrams if total_unigrams > 0 else 0
            rows.append([f"P({unigram})", f"{prob:.6f}"])

        # Bigram Probabilities
        rows.append([])
        rows.append(['Bigram Probabilities'])
        for bigram, freq in ngram_freq.items():
            if len(bigram) == 2:
                prob = freq / total_bigrams if total_bigrams > 0 else 0
                rows.append([f"P({bigram[0]}, {bigram[1]})", f"{prob:.6f}"])

        # Trigram Probabilities
        rows.append([])
        rows.append(['Trigram Probabilities'])
        for trigram, freq in ngram_freq.items():
            if len(trigram) == 3:
                prob = freq / total_trigrams if total_trigrams > 0 else 0
                rows.append([f"P({trigram[0]}, {trigram[1]}, {trigram[2]})", f"{prob:.6f}"])

        return rows

    def _build_conditional_model(self):
        """Build a conditional probability model with normalized probabilities."""
        ngram_freq = defaultdict(int)
        unigram_freq = defaultdict(int)
        lexicon_hits = defaultdict(int)
        total_tokens = len(self.tokens)
        
        for i in range(total_tokens - 1):
            unigram = self.tokens[i]
            bigram = (self.tokens[i], self.tokens[i + 1]) if self.positional else tuple(sorted([self.tokens[i], self.tokens[i + 1]]))
            unigram_freq[unigram] += 1
            ngram_freq[bigram] += 1
            
            # Check if unigram is in lexicon
            if unigram in self.lexicon:
                lexicon_hits[unigram] += 1

        # Include the last token in unigram frequencies
        if total_tokens > 0:
            last_token = self.tokens[-1]
            unigram_freq[last_token] += 1
            if last_token in self.lexicon:
                lexicon_hits[last_token] += 1

        # Prepare data for CSV output
        rows = []
        rows.append(['Model Type', 'Conditional'])
        rows.append(['Use Word Boundaries', self.use_word_boundaries])
        rows.append(['Positional', self.positional])
        rows.append(['Total Tokens', total_tokens])

        # Lexicon Hits
        rows.append([])
        rows.append(['Lexicon Hits'])
        for lex_word, count in lexicon_hits.items():
            rows.append([lex_word, count])

        # Unigram Frequencies
        rows.append([])
        rows.append(['Unigram Frequencies'])
        for unigram, freq in unigram_freq.items():
            rows.append([unigram, freq])

        # Conditional Probabilities
        rows.append([])
        rows.append(['Conditional Probabilities (Bigram)'])
        for bigram, freq in ngram_freq.items():
            unigram = bigram[0]
            prob = freq / unigram_freq[unigram] if unigram_freq[unigram] > 0 else 0
            rows.append([f"P({bigram[1]}|{bigram[0]})", f"{prob:.6f}"])

        return rows

    def generate_output(self, output_file, rows):
        """Write the output to the specified CSV file."""
        with open(output_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
            writer.writerow(['='*80])
        print(f"Output has been written to {output_file}")

def read_input_file(input_file):
    """Reads data from a given text file."""
    with open(input_file, 'r', encoding='utf-8') as file:
        return file.read()

def read_lexicon(lexicon_file):
    """Reads lexicon words from a file, returning a set of lexicon words."""
    with open(lexicon_file, 'r', encoding='utf-8') as file:
        return set(line.strip() for line in file)

def main():
    # File paths relative to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, '..', 'data', 'sample_test_data', 'sample_informational_text.txt')
    lexicon_file = os.path.join(script_dir, '..', 'data', 'english.csv')
    output_file = os.path.join(script_dir, 'output.csv')

    # Read input from the file
    data = read_input_file(input_file)
    
    # Read lexicon from lexicon file
    lexicon = read_lexicon(lexicon_file)
    
    # Possible options
    model_types = ['joint', 'conditional']
    use_word_boundaries_options = [True, False]
    positional_options = [True, False]
    
    # Clear the output file if it exists
    open(output_file, 'w', encoding='utf-8').close()
    
    # Iterate over all combinations
    for model_type, use_word_boundaries, positional in itertools.product(
        model_types, use_word_boundaries_options, positional_options):
        
        # Create and build the language model
        lm = LanguageModel(
            data, lexicon, 
            model_type=model_type, 
            use_word_boundaries=use_word_boundaries, 
            positional=positional)
        
        # Generate result
        rows = lm.build_model()
        
        # Write result to the file
        lm.generate_output(output_file, rows)
        
if __name__ == "__main__":
    main()
