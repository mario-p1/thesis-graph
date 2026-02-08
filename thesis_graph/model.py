import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import to_hetero
from torch_geometric.nn.models import GraphSAGE
from torch_geometric.typing import EdgeType, NodeType


class Classifier(torch.nn.Module):
    def __init__(self, thesis_dim: int, professor_dim: int):
        super().__init__()
        self.lin = torch.nn.Linear(thesis_dim + professor_dim + professor_dim, 32)
        self.lin2 = torch.nn.Linear(32, 1)

    def forward(
        self, x_thesis: Tensor, x_mentor: Tensor, x_committee_member: Tensor
    ) -> Tensor:
        x = self.lin(torch.cat([x_thesis, x_mentor, x_committee_member], dim=-1))
        x = torch.relu(x)
        x = self.lin2(x)
        x = x.view(-1)
        return x


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
            aggr="sum",
        )
        self.gnn = to_hetero(self.gnn, metadata=metadata)

        # Classifier (mentor professor + thesis + professor) -> score
        self.classifier = Classifier(hidden_channels, hidden_channels)

    def forward(self, data: HeteroData) -> Tensor:
        thesis_node_repr = self.thesis_lin(data["thesis"].x)

        professor_node_repr = self.professor_emb(data["professor"].node_id)

        x_dict = {
            "thesis": thesis_node_repr,
            "professor": professor_node_repr,
        }

        x_dict = self.gnn(x_dict, data.edge_index_dict)

        com_member_label_index = data[
            "thesis", "has_committee_member", "professor"
        ].edge_label_index

        sup_by_edge_index = data["thesis", "supervised_by", "professor"].edge_index
        sup_by_dict = sup_by_edge_index.cpu()
        sup_by_dict = {
            thesis.item(): mentor.item()
            for thesis, mentor in zip(sup_by_dict[0], sup_by_dict[1])
        }

        # Mentor id for each thesis in the com_member_label_index
        mentors_indices = [
            sup_by_dict[thesis.item()] for thesis in com_member_label_index[0].cpu()
        ]

        pred = self.classifier(
            x_dict["thesis"][com_member_label_index[0]],
            x_dict["professor"][torch.LongTensor(mentors_indices)],
            x_dict["professor"][com_member_label_index[1]],
        )

        return pred
