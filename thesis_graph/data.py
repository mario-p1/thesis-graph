from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch_geometric.data import HeteroData

from thesis_graph.embedding import embed_text
from thesis_graph.features import build_mentors_features_matrix


def load_thesis_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [
        "thesis_title_mk",
        "student",
        "mentor",
        "c1",
        "c2",
        "thesis_application_date",
        "thesis_status",
        "thesis_desc_mk",
        "thesis_desc_en",
        "thesis_title_en",
    ]
    return df


def load_researchers_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        converters={
            "interests": lambda x: x.split("|") if x else [],
            "articles": lambda x: x.split("|") if x else [],
        },
    )


def build_graph(
    thesis_df: pd.DataFrame, researchers_df: pd.DataFrame
) -> tuple[HeteroData, dict[str, Any]]:
    # Build Mentors features
    mentors = sorted(thesis_df["mentor"].unique().tolist())
    mentors_dict = {mentor: index for index, mentor in enumerate(mentors)}

    # researchers_features = build_mentors_features_matrix(researchers_df, mentors_dict)
    # researchers_features = torch.from_numpy(researchers_features)

    # Build thesis features
    desc_embeddings = embed_text(thesis_df["thesis_desc_en"].tolist())
    thesis_features = torch.from_numpy(desc_embeddings)

    # Supervises relation
    supervises_mentor = thesis_df["mentor"].apply(lambda mentor: mentors_dict[mentor])
    supervises_thesis = thesis_df.index.tolist()
    supervises_features = torch.vstack(
        [
            torch.LongTensor(supervises_thesis),
            torch.LongTensor(supervises_mentor),
        ]
    )

    # Build graph
    graph = HeteroData()
    graph["thesis"].node_id = torch.arange(len(thesis_df))
    graph["thesis"].x = thesis_features

    graph["mentor"].node_id = torch.arange(len(mentors))
    # graph["mentor"].x = researchers_features

    graph["thesis", "supervised_by", "mentor"].edge_index = supervises_features
    graph["mentor", "supervises", "thesis"].edge_index = supervises_features.flip(0)

    # Validate the constructed graph
    validate_result = graph.validate()
    print("Graph validation result:", validate_result)

    return graph, {"mentors_dict": mentors_dict}
