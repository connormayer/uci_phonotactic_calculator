# src\validate.py

def valid_file(file_path):
    with open(file_path, 'r') as f:
        tokens = f.read()

    tokens = [s.split(" ") for s in tokens.split("\n") if s]

    for token_list in tokens:
        if len(token_list) == 1 and len(token_list[0]) > 2:
            return False
    return True