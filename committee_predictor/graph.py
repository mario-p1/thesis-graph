import random

import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit

from committee_predictor.embedding import embed_text


def build_single_graph(
    thesis_df: pd.DataFrame,
    professors_lookup: dict[str, int],
    add_edge_labels: bool = False,
) -> HeteroData:
    # Empty graph
    graph = HeteroData()

    # Reset thesis_df index to ensure its index can be used as thesis node ids
    thesis_df = thesis_df.reset_index(drop=True)

    # Add Thesis nodes
    torch_thesis_ids = torch.arange(thesis_df.shape[0])
    desc_embeddings = embed_text(thesis_df["thesis_desc_en"].tolist())
    thesis_features = torch.from_numpy(desc_embeddings)

    graph["thesis"].node_id = torch_thesis_ids
    graph["thesis"].x = thesis_features

    # Add Professor nodes
    graph["professor"].node_id = torch.arange(len(professors_lookup))

    # Add Thesis supervised_by Professor edges
    torch_mentor_ids = torch.LongTensor(thesis_df["mentor_id"].tolist())

    thesis_mentor_edges = torch.vstack(
        [
            torch_thesis_ids,
            torch_mentor_ids,
        ]
    )

    graph["thesis", "supervised_by", "professor"].edge_index = thesis_mentor_edges
    graph["professor", "supervises", "thesis"].edge_index = thesis_mentor_edges.flip(0)

    # Add Thesis has_committee_member Professor edges
    thesis_with_c1 = thesis_df.dropna(subset=["c1_id"])
    thesis_with_c2 = thesis_df.dropna(subset=["c2_id"])

    comission_edges = torch.hstack(
        [
            torch.vstack(
                [
                    torch.LongTensor(thesis_with_c1.index.tolist()),
                    torch.LongTensor(thesis_with_c1["c1_id"].tolist()),
                ]
            ),
            torch.vstack(
                [
                    torch.LongTensor(thesis_with_c2.index.tolist()),
                    torch.LongTensor(thesis_with_c2["c2_id"].tolist()),
                ]
            ),
        ]
    )

    graph["thesis", "has_committee_member", "professor"].edge_index = comission_edges
    graph[
        "professor", "is_committee_member_of", "thesis"
    ].edge_index = comission_edges.flip(0)

    if add_edge_labels:
        # Add edge labels for the "has_committee_member" edges
        num_positive_edges = comission_edges.shape[1]
        graph["thesis", "has_committee_member", "professor"].edge_label = torch.ones(
            num_positive_edges, dtype=torch.long
        )
        graph[
            "thesis", "has_committee_member", "professor"
        ].edge_label_index = comission_edges

    # Validate graph
    graph.validate()

    return graph


def add_negatives_to_edge_labels(
    graph: HeteroData, link: tuple[str], negative_rate: int
):
    negative_links = []

    positive_links = graph[link].edge_label_index.t().tolist()

    dest_possible_choices = graph[link[-1]].num_nodes

    for _ in range(negative_rate):
        for source, dest in positive_links:
            rnd_dest = (
                dest + random.randint(1, dest_possible_choices - 1)
            ) % dest_possible_choices

            negative_links.append([source, rnd_dest])

    # Set negative edge labels
    negative_links_tensor = torch.LongTensor(negative_links).t()
    all_edge_index = torch.cat(
        [graph[link].edge_label_index, negative_links_tensor], dim=1
    )
    graph[link].edge_label_index = all_edge_index

    positive_edge_labels = graph[link].edge_label
    negative_edge_labels = torch.zeros(
        negative_links_tensor.shape[1], dtype=positive_edge_labels.dtype
    )
    all_edge_labels = torch.cat([positive_edge_labels, negative_edge_labels], dim=0)
    graph[link].edge_label = all_edge_labels


def build_graphs(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame | None,
    professors_lookup: dict[str, int],
    disjoint_train_ratio: float,
    neg_train_ratio: int,
    neg_val_test_ratio: int,
) -> tuple[HeteroData, HeteroData | None, HeteroData | None]:
    # Build train graph
    orig_train_data = build_single_graph(train_df, professors_lookup=professors_lookup)

    # Split train links into message passing and supervision links
    train_splitter = RandomLinkSplit(
        num_test=0,
        num_val=0,
        disjoint_train_ratio=disjoint_train_ratio,
        edge_types=[("thesis", "has_committee_member", "professor")],
        rev_edge_types=[("professor", "is_committee_member_of", "thesis")],
        neg_sampling_ratio=0,
    )

    # Split train data into train MESSAGE PASSING and train CLASSIFICATION
    train_data = train_splitter(orig_train_data)[0]

    add_negatives_to_edge_labels(
        train_data,
        ("thesis", "has_committee_member", "professor"),
        neg_train_ratio,
    )

    val_data = build_single_graph(
        val_df, professors_lookup=professors_lookup, add_edge_labels=True
    )
    add_negatives_to_edge_labels(
        val_data, ("thesis", "has_committee_member", "professor"), neg_val_test_ratio
    )

    test_data = build_single_graph(
        test_df, professors_lookup=professors_lookup, add_edge_labels=True
    )
    add_negatives_to_edge_labels(
        test_data, ("thesis", "has_committee_member", "professor"), neg_val_test_ratio
    )

    return train_data, val_data, test_data
