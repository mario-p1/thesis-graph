from pathlib import Path

import pandas as pd


def main():
    pd.options.display.max_rows = 20
    pd.options.display.max_columns = 20

    df = pd.read_csv(Path(__file__).parent.parent / "data" / "committee_train.csv")

    print(df.head())


if __name__ == "__main__":
    main()
