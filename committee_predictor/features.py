from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from committee_predictor.embedding import embed_text


def get_researchers_features(
    researchers: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    names = []
    descriptions = []

    for researcher in researchers:
        names.append(researcher["name"])
        interests = researcher.get("interests", [])
        # articles = researcher.get("articles", [])

        desc = f"Interests of research: {','.join(interests)}"
        descriptions.append(desc)

    mlflow.log_param("researchers_features", "Interests")
    embeddings = embed_text(descriptions)
    return {name: embedding for name, embedding in zip(names, embeddings)}


def get_researchers_features_v2(researchers_df: pd.DataFrame) -> dict[str, np.ndarray]:
    interests_encoder = OneHotEncoder(sparse_output=False)

    exploded_interests_df = researchers_df.explode(column="interests")

    encoded_interest_features = interests_encoder.fit_transform(
        exploded_interests_df[["interests"]]
    )

    interest_features_df = pd.DataFrame(
        encoded_interest_features,
        columns=interests_encoder.get_feature_names_out(["interests"]),
    )

    interest_features_df = pd.concat(
        [exploded_interests_df[["name"]].reset_index(drop=True), interest_features_df],
        axis=1,
    ).pivot_table(index="name", aggfunc="sum")

    result_df = pd.merge(
        researchers_df, interest_features_df, left_on="name", right_index=True
    )
    result_df = result_df.drop(columns=["interests", "articles"])
    result_df = result_df.set_index("name")

    return {name: row.to_numpy(dtype=np.float32) for name, row in result_df.iterrows()}


def build_mentors_features_matrix(
    researchers: pd.DataFrame, mentors_df: dict[str, int]
):
    # features = get_researchers_features(researchers.to_dict(orient="records"))

    features = get_researchers_features_v2(researchers)

    if len(features) != len(mentors_df):
        raise ValueError("Some mentors are missing in researchers features.")

    matrix = np.zeros(
        (len(mentors_df), next(iter(features.values())).shape[0]), dtype=np.float32
    )

    for mentor_name, index in mentors_df.items():
        matrix[index] = features[mentor_name]

    return matrix
