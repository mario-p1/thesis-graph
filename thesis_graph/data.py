from pathlib import Path

import pandas as pd

from thesis_graph import config


def load_thesis_csv(path: Path = config.THESIS_CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [
        "thesis_title_mk",
        "student",
        "mentor",
        "c1",
        "c2",
        "application_date",
        "thesis_status",
        "thesis_desc_mk",
        "thesis_desc_en",
        "thesis_title_en",
    ]
    df["application_date"] = pd.to_datetime(df["application_date"], format="%d.%m.%Y")
    df = df.sort_values(by="application_date", ascending=True)
    df = df.reset_index(drop=True)
    return df


def filter_thesis_df(df: pd.DataFrame) -> pd.DataFrame:
    # TODO
    # df = df[df["thesis_status"] == "Одбрана"].reset_index(drop=True)
    df = df.drop_duplicates(subset=["student"], keep="last")
    return df


def load_researchers_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        converters={
            "interests": lambda x: x.split("|") if x else [],
            "articles": lambda x: x.split("|") if x else [],
        },
    )


def train_test_split_thesis_df(df: pd.DataFrame, train_ratio: float, val_ratio: float):
    train_cut = int(df.shape[0] * train_ratio)
    val_cut = int(df.shape[0] * (train_ratio + val_ratio))

    train_df, val_df, test_df = (
        df.iloc[:train_cut],
        df.iloc[train_cut:val_cut],
        df.iloc[val_cut:],
    )

    return train_df, val_df, test_df
