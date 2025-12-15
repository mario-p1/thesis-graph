from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit

from thesis_graph.config import THESIS_CSV_PATH
from thesis_graph.data import load_thesis_csv, train_test_split_thesis_df
from thesis_graph.embedding import embed_text


def build_single_graph(
    thesis_df: pd.DataFrame,
    mentors_dict: dict[str, int],
    researchers_df: pd.DataFrame | None = None,
    add_edge_labels: bool = False,
) -> HeteroData:
    # Build graph
    graph = HeteroData()

    # => Thesis nodes
    desc_embeddings = embed_text(thesis_df["thesis_desc_en"].tolist())
    thesis_features = torch.from_numpy(desc_embeddings)

    graph["thesis"].node_id = torch.arange(len(thesis_df))
    graph["thesis"].x = thesis_features

    # => Mentor nodes
    graph["mentor"].node_id = torch.tensor(list(mentors_dict.values()))
    # graph["mentor"].x = researchers_features

    # => Thesis supervised by mentor links
    mentor_indices = list(
        thesis_df["mentor"].apply(lambda mentor: mentors_dict[mentor])
    )
    thesis_indices = torch.arange(len(thesis_df))

    thesis_supervised_by_mentor_indices = torch.vstack(
        [
            torch.LongTensor(thesis_indices),
            torch.LongTensor(mentor_indices),
        ]
    )

    # ==> Message passing
    graph[
        "thesis", "supervised_by", "mentor"
    ].edge_index = thesis_supervised_by_mentor_indices
    graph[
        "mentor", "supervises", "thesis"
    ].edge_index = thesis_supervised_by_mentor_indices.flip(0)

    # ==> Labels
    if add_edge_labels:
        edge_labels = torch.ones(
            thesis_supervised_by_mentor_indices.size(1), dtype=torch.float
        )
        graph["thesis", "supervised_by", "mentor"].edge_label = edge_labels
        graph["mentor", "supervises", "thesis"].edge_label = edge_labels
        graph["thesis", "supervised_by", "mentor"].edge_label_index = graph[
            "thesis", "supervised_by", "mentor"
        ].edge_index
        graph["mentor", "supervises", "thesis"].edge_label_index = graph[
            "mentor", "supervises", "thesis"
        ].edge_index

    graph.validate(raise_on_error=False)

    return graph


def build_mentors_dict(thesis_df: pd.DataFrame) -> dict[str, int]:
    mentors = sorted(thesis_df["mentor"].unique().tolist())
    mentors_dict = {mentor: index for index, mentor in enumerate(mentors)}
    return mentors_dict


def build_graphs(
    thesis_path: Path = THESIS_CSV_PATH,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    disjoint_train_ratio: float = 0.7,
):
    thesis_df = load_thesis_csv(thesis_path)

    train_df, val_df, test_df = train_test_split_thesis_df(
        thesis_df, train_ratio=train_ratio, val_ratio=val_ratio
    )

    mentors_dict = build_mentors_dict(train_df)

    train_data = build_single_graph(train_df, mentors_dict=mentors_dict)

    # Split train links into message passing and supervision links
    train_splitter = RandomLinkSplit(
        num_test=0,
        num_val=0,
        disjoint_train_ratio=disjoint_train_ratio,
        edge_types=[("thesis", "supervised_by", "mentor")],
        rev_edge_types=[("mentor", "supervises", "thesis")],
        neg_sampling_ratio=0,
    )

    train_data = train_splitter(train_data)[0]

    val_data = build_single_graph(
        val_df, mentors_dict=mentors_dict, add_edge_labels=True
    )
    test_data = build_single_graph(
        test_df, mentors_dict=mentors_dict, add_edge_labels=True
    )

    return mentors_dict, train_data, val_data, test_data
