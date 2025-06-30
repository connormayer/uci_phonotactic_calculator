"""
CLI helpers for demo data: --list-data and --use-demo-data
"""

import importlib.resources as pkg_resources


def list_data_files() -> None:
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


def get_demo_paths(dataset_name: str) -> tuple[str, str]:
    """
    Get paths to a demo dataset.

    Args:
        dataset_name: The name of the dataset file (e.g., "english.csv").

    Returns:
        A tuple containing the absolute paths to the training and test files.
    """
    base = "uci_phonotactic_calculator.data"
    train = pkg_resources.files(base).joinpath(dataset_name)

    # e.g., "english.csv" -> "english_test_data.csv"
    test_filename = dataset_name.replace(".csv", "_test_data.csv")
    test = pkg_resources.files(f"{base}.sample_test_data").joinpath(test_filename)
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
        train, test = get_demo_paths("english.csv")
        print(train)
        print(test)
    else:
        parser.print_help()
