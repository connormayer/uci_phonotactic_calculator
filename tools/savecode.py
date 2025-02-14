
"""
savecode.py - Save Python code from directories and files into one output file.
Version: 1.2.1
"""

import os
import argparse

def gather_py_files(root_dir, skip_dirs=None):
    """
    Recursively gather all .py files under root_dir,
    skipping any directories listed in skip_dirs.
    """
    skip_dirs = set(skip_dirs or [])
    py_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Remove directories that should be skipped so os.walk won’t traverse them.
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for fname in filenames:
            if fname.endswith(".py"):
                py_files.append(os.path.join(dirpath, fname))
    return py_files

def save_code(py_files, output_file):
    """
    Save the code from each Python file in py_files to output_file.
    Each file's code is preceded by a header (with the file path)
    and separated by blank lines.
    """
    with open(output_file, 'w', encoding='utf-8') as out:
        for file in py_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    header = f"\nFile: {file}\n\n"
                    out.write(header)
                    out.write(f.read())
                    out.write("\n\n")
            except Exception as e:
                print(f"Error reading {file}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Save the full code from Python files in specified directories and individual files to a single output file."
    )
    parser.add_argument(
        '-r', '--roots',
        nargs='*',
        default=[],
        help="One or more root directories to search for Python files."
    )
    parser.add_argument(
        '-f', '--files',
        nargs='*',
        default=[],
        help="One or more individual Python file paths to include."
    )
    parser.add_argument(
        '-o', '--output',
        default="./temp.txt",
        help="Output file path. Defaults to './temp.txt'."
    )
    parser.add_argument(
        '--skip',
        nargs='*',
        default=['rnn_src'],
        help="Subdirectory names to skip (default: ['rnn_src'])."
    )
    args = parser.parse_args()

    all_py_files = []

    # Gather Python files from specified directories.
    for root in args.roots:
        all_py_files.extend(gather_py_files(root, args.skip))

    # Add individual Python files, ensuring they exist and have the correct extension.
    for file in args.files:
        if os.path.isfile(file) and file.endswith(".py"):
            all_py_files.append(file)
        else:
            print(f"Warning: {file} is not a valid Python file.")

    # Use the provided output file or default to './temp.txt'.
    output_file = args.output

    save_code(all_py_files, output_file)

    # Print the list of saved files with colors.
    green = "\033[1;32m"
    blue = "\033[1;34m"
    cyan = "\033[1;36m"
    white = "\033[1;37m"
    reset = "\033[0m"
    print(f"\n{cyan}Saved code from {len(all_py_files)} files to {output_file}{reset}")
    print(f"\n{green}Files saved:{reset}")
    for f in all_py_files:
        print(f"{blue}- {f}{reset}")
    print ("\n")

if __name__ == "__main__":
    main()