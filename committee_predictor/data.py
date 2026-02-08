from pathlib import Path

import numpy as np
import pandas as pd

from committee_predictor import config


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


def prepare_thesis_data_splits(
    thesis_df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    thesis_filter: int = 0,
) -> tuple[dict[str, int], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    thesis_df = filter_thesis_df(thesis_df)

    if thesis_filter < 0:
        thesis_df = thesis_df[thesis_filter:]
    elif thesis_filter > 0:
        thesis_df = thesis_df[:thesis_filter]

    train_df, val_df, test_df = _train_test_split_df(
        thesis_df, train_ratio=train_ratio, val_ratio=val_ratio
    )
    professors_lookup = _build_professors_lookup(train_df)

    train_df = _map_professor_ids(train_df, professors_lookup)
    val_df = _map_professor_ids(val_df, professors_lookup)
    test_df = _map_professor_ids(test_df, professors_lookup)

    return professors_lookup, train_df, val_df, test_df


def _train_test_split_df(df: pd.DataFrame, train_ratio: float, val_ratio: float):
    train_cut = int(df.shape[0] * train_ratio)
    val_cut = int(df.shape[0] * (train_ratio + val_ratio))

    train_df, val_df, test_df = (
        df.iloc[:train_cut],
        df.iloc[train_cut:val_cut],
        df.iloc[val_cut:],
    )

    return train_df.copy(), val_df.copy(), test_df.copy()


def _build_professors_lookup(thesis_df: pd.DataFrame) -> dict[str, int]:
    mentors = sorted(thesis_df["mentor"].unique().tolist())
    mentors_dict = {mentor: index for index, mentor in enumerate(mentors)}
    return mentors_dict


def _map_professor_ids(
    df: pd.DataFrame, professor_lookup: dict[str, int]
) -> pd.DataFrame:

    # Apply professor name -> professor id mapping
    df["mentor_id"] = df["mentor"].apply(
        lambda mentor: professor_lookup.get(mentor, np.nan)
    )
    df["c1_id"] = df["c1"].apply(lambda mentor: professor_lookup.get(mentor, np.nan))
    df["c2_id"] = df["c2"].apply(lambda mentor: professor_lookup.get(mentor, np.nan))

    # Remove rows with missing professor ids
    df = df.dropna(subset=["mentor_id"]).reset_index(drop=True)

    return df
