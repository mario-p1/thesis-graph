import argparse
from pathlib import Path

import pandas as pd

from mentor_finder.data import load_raw_committee_csv
from mentor_finder.embedding import embed_text


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split dataset into train, validation, and test sets"
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("data/committee.csv"),
        help="Path to the input file",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=Path("data"),
        help="Folder to save the output files (default: data)",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.8,
        help="Fraction of data for training (default: 0.8)",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Fraction of data for validation (default: 0.1)",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.1,
        help="Fraction of data for testing (default: 0.1)",
    )
    return parser.parse_args()


def split_train_test_val(
    df, train_fraction, val_fraction, test_fraction, random_state=42
):
    if train_fraction + val_fraction + test_fraction != 1.0:
        raise ValueError("Train, validation, and test fractions must sum to 1.0")

    train_df = df.sample(frac=train_fraction, random_state=random_state)
    test_val_df = df.drop(train_df.index)
    val_df = test_val_df.sample(
        frac=val_fraction / (val_fraction + test_fraction), random_state=random_state
    )
    test_df = test_val_df.drop(val_df.index)
    return train_df, val_df, test_df


def main():
    args = parse_args()

    input_file = args.input_file
    output_folder = args.output_folder

    train_frac = args.train_frac
    val_frac = args.val_frac
    test_frac = args.test_frac

    df = load_raw_committee_csv(input_file)

    train_df, val_df, test_df = split_train_test_val(
        df, train_fraction=train_frac, val_fraction=val_frac, test_fraction=test_frac
    )

    input_file_name = input_file.stem

    train_path = output_folder / f"{input_file_name}_train.csv"
    val_path = output_folder / f"{input_file_name}_val.csv"
    test_path = output_folder / f"{input_file_name}_test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"{'=' * 60}")
    print("Dataset split completed successfully!")
    print(f"{'=' * 60}")
    print("")
    print(f"Input file: {input_file}")
    print(f"Total records: {len(df)}")
    print("")
    print("Split ratios:")
    print(f"  Train: {train_frac:.1%} ({len(train_df)} records)")
    print(f"  Validation: {val_frac:.1%} ({len(val_df)} records)")
    print(f"  Test: {test_frac:.1%} ({len(test_df)} records)")
    print("")
    print(f"Output files saved to: {output_folder.resolve()}")
    print(f"  - {train_path.name}")
    print(f"  - {val_path.name}")
    print(f"  - {test_path.name}")


if __name__ == "__main__":
    main()
