from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch_geometric.data import HeteroData


from mentor_finder.embedding import embed_text


def load_raw_committee_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [
        "thesis_title_mk",
        "student",
        "mentor",
        "c1",
        "c2",
        "thesis_application_date",
        "thesis_status",
        "graduation_thesis_desc_mk",
        "thesis_desc_en",
        "thesis_title_en",
    ]
    return df


def build_graph(df: pd.DataFrame) -> tuple[HeteroData, dict[str, Any]]:
    # Build thesis features
    desc_embeddings = embed_text(df["thesis_desc_en"].tolist())
    thesis_features = torch.from_numpy(desc_embeddings)

    # Mentors
    mentors = sorted(df["mentor"].unique().tolist())
    mentors_dict = {mentor: index for index, mentor in enumerate(mentors)}

    # Supervises relation
    supervises_mentor = df["mentor"].apply(lambda mentor: mentors_dict[mentor])
    supervises_thesis = df.index.tolist()
    supervises_features = torch.vstack(
        [
            torch.LongTensor(supervises_thesis),
            torch.LongTensor(supervises_mentor),
        ]
    )

    # Build graph
    graph = HeteroData()
    graph["thesis"].node_id = torch.arange(len(df))
    graph["thesis"].x = thesis_features
    graph["mentor"].node_id = torch.arange(len(mentors))

    graph["thesis", "supervised_by", "mentor"].edge_index = supervises_features
    graph["mentor", "supervises", "thesis"].edge_index = supervises_features.flip(0)

    # Validate the constructed graph
    graph.validate()

    return graph, {"mentors_dict": mentors_dict}
