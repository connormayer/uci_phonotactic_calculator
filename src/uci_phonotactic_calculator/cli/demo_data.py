"""
CLI helpers for demo data: --list-data and --use-demo-data
"""

import importlib.resources as pkg_resources


def list_data_files():
    base = "uci_phonotactic_calculator.data"
    # List top-level files
    for f in pkg_resources.files(base).iterdir():
        if f.is_file():
            print(f"{f}")
    # List files in subdirs
    for sub in pkg_resources.files(base).iterdir():
        if sub.is_dir():
            for f in sub.iterdir():
                if f.is_file():
                    print(f"{f}")


def get_demo_paths():
    base = "uci_phonotactic_calculator.data"
    train = pkg_resources.files(base).joinpath("english.csv")
    test = pkg_resources.files(f"{base}.sample_test_data").joinpath(
        "english_test_data.csv"
    )
    return str(train), str(test)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Demo data utilities")
    parser.add_argument(
        "--list-data",
        action="store_true",
        help="List all available packaged data files",
    )
    parser.add_argument(
        "--use-demo-data",
        action="store_true",
        help="Print paths to demo train/test files "
        "(english.csv, english_test_data.csv)",
    )
    args = parser.parse_args()
    if args.list_data:
        list_data_files()
    elif args.use_demo_data:
        train, test = get_demo_paths()
        print(train)
        print(test)
    else:
        parser.print_help()
