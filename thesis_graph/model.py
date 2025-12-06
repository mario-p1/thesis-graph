import mlflow
import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import to_hetero
from torch_geometric.nn.models import GraphSAGE


class Classifier(torch.nn.Module):
    def forward(
        self, x_thesis: Tensor, x_mentor: Tensor, edge_label_index: Tensor
    ) -> Tensor:
        edge_feat_thesis = x_thesis[edge_label_index[0]]
        edge_feat_mentor = x_mentor[edge_label_index[1]]

        return (edge_feat_thesis * edge_feat_mentor).sum(dim=-1)


class Model(torch.nn.Module):
    def __init__(
        self,
        node_embedding_channels: int,
        hidden_channels: int,
        gnn_num_layers: int,
        data: HeteroData,
    ):
        super().__init__()

        self.thesis_lin = torch.nn.Linear(384, node_embedding_channels)

        self.thesis_emb = torch.nn.Embedding(
            data["thesis"].num_nodes, node_embedding_channels
        )

        self.mentor_lin = torch.nn.Linear(117, node_embedding_channels)

        self.mentor_emb = torch.nn.Embedding(
            data["mentor"].num_nodes, node_embedding_channels
        )

        self.gnn = GraphSAGE(
            in_channels=node_embedding_channels,
            hidden_channels=hidden_channels,
            num_layers=gnn_num_layers,
        )
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())

        self.classifier = Classifier()

    def forward(self, data: HeteroData) -> Tensor:
        thesis_node_repr = self.thesis_lin(data["thesis"].x) + self.thesis_emb(
            data["thesis"].node_id
        )
        mlflow.log_param("using_thesis_node_features", "abstract_embeding")

        # mentor_node_repr = self.mentor_lin(data["mentor"].x) + self.mentor_emb(
        #     data["mentor"].node_id
        # )
        # mlflow.log_param("using_mentor_node_features", "interests(one-hot)")

        mentor_node_repr = self.mentor_emb(data["mentor"].node_id)
        mlflow.log_param("using_mentor_node_features", "none")

        x_dict = {
            "thesis": thesis_node_repr,
            "mentor": mentor_node_repr,
        }

        x_dict = self.gnn(x_dict, data.edge_index_dict)

        pred = self.classifier(
            x_dict["thesis"],
            x_dict["mentor"],
            data["thesis", "supervised_by", "mentor"].edge_label_index,
        )

        return pred
