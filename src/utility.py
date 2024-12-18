from os import path

def valid_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        tokens = f.read()

    tokens = [s.split(',') for s in tokens.split("\n") if s]
    tokens_no_freq = [token[0].split(' ') for token in tokens]

    if any(map(lambda x: '\t' in x, tokens)):
        return (False, 'Files must be comma delimited.')
    
    if all(map(lambda x: len(x) == 1, tokens_no_freq)):
        return (False, 'Phonemes must be separated by spaces.')

    return (True, "")

def get_filename(test_file, timestamp):
    timestamp_str = str(timestamp).replace('.', '_')
    base_name, ext = path.splitext(test_file)
    outfile = base_name + "_scores" + ext
    folder_name = outfile[:4] + '_' + timestamp_str
    return path.join(folder_name, outfile)
