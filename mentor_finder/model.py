import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels: int):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class Classifier(torch.nn.Module):
    def forward(
        self, x_mentor: Tensor, x_thesis: Tensor, edge_label_index: Tensor
    ) -> Tensor:
        edge_feat_thesis = x_thesis[edge_label_index[0]]
        edge_feat_mentor = x_mentor[edge_label_index[1]]

        return (edge_feat_thesis * edge_feat_mentor).sum(dim=-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels: int, data: HeteroData):
        super().__init__()
        self.thesis_lin = torch.nn.Linear(384, hidden_channels)
        self.thesis_emb = torch.nn.Embedding(data["thesis"].num_nodes, hidden_channels)
        self.mentor_emb = torch.nn.Embedding(data["mentor"].num_nodes, hidden_channels)

        self.gnn = GNN(hidden_channels)
        self.gnn = to_hetero(self.gnn, data.metadata())

        self.classifier = Classifier()

    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
            "thesis": self.thesis_lin(data["thesis"].x)
            + self.thesis_emb(data["thesis"].node_id),
            "mentor": self.mentor_emb(data["mentor"].node_id),
        }

        x_dict = self.gnn(x_dict, data.edge_index_dict)

        pred = self.classifier(
            x_dict["mentor"],
            x_dict["thesis"],
            data["thesis", "supervised_by", "mentor"].edge_label_index,
        )

        return pred
