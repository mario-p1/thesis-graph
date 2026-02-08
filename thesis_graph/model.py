import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import to_hetero
from torch_geometric.nn.models import GraphSAGE
from torch_geometric.typing import EdgeType, NodeType


class Classifier(torch.nn.Module):
    def __init__(self, thesis_dim: int, mentor_dim: int):
        super().__init__()

        self.lin = torch.nn.Linear(thesis_dim + mentor_dim, 1)

    def forward(self, x_thesis: Tensor, x_mentor: Tensor) -> Tensor:
        return self.lin(torch.cat([x_thesis, x_mentor], dim=-1)).view(-1)
        return (x_thesis * x_mentor).sum(dim=-1)


class Model(torch.nn.Module):
    def __init__(
        self,
        num_mentors: int,
        thesis_features_dim: int,
        node_embedding_channels: int,
        hidden_channels: int,
        gnn_num_layers: int,
        metadata: tuple[list[NodeType], list[EdgeType]],
    ):
        super().__init__()

        # Thesis features space -> node embedding space
        self.thesis_lin = torch.nn.Linear(thesis_features_dim, node_embedding_channels)

        # Professor embeddings
        self.professor_emb = torch.nn.Embedding(num_mentors, node_embedding_channels)

        # Graph Neural Network
        self.gnn = GraphSAGE(
            in_channels=node_embedding_channels,
            hidden_channels=hidden_channels,
            num_layers=gnn_num_layers,
        )
        self.gnn = to_hetero(self.gnn, metadata=metadata)

        # Classifier (mentor professor + thesis + professor) -> score
        self.classifier = Classifier(hidden_channels, hidden_channels)

    def forward(self, data: HeteroData) -> Tensor:
        thesis_node_repr = self.thesis_lin(data["thesis"].x)

        mentor_node_repr = self.professor_emb(data["mentor"].node_id)

        x_dict = {
            "thesis": thesis_node_repr,
            "mentor": mentor_node_repr,
        }

        x_dict = self.gnn(x_dict, data.edge_index_dict)

        eli = data["thesis", "supervised_by", "mentor"].edge_label_index
        # breakpoint()
        pred = self.classifier(
            x_dict["thesis"][eli[0]],
            x_dict["mentor"][eli[1]],
        )

        return pred

    def get_prediction_new_thesis(self, thesis_features: Tensor) -> Tensor:
        thesis_node_repr = self.thesis_lin(thesis_features.unsqueeze(0))

        mentor_node_repr = self.professor_emb.weight

        scores = self.classifier(
            thesis_node_repr.repeat(mentor_node_repr.size(0), 1),
            mentor_node_repr,
        )

        return scores
