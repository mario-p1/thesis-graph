from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import tqdm
from torch_geometric.loader import LinkNeighborLoader
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero

from mentor_finder.data import build_graph, load_raw_committee_csv


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
        edge_feat_mentor = x_mentor[edge_label_index[0]]
        edge_feat_thesis = x_thesis[edge_label_index[1]]

        return (edge_feat_mentor * edge_feat_thesis).sum(dim=-1)


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
            data["mentor", "supervises", "thesis"].edge_label_index,
        )

        return pred


def main():
    pd.options.display.max_rows = 20
    pd.options.display.max_columns = 20

    df = load_raw_committee_csv(Path(__file__).parent.parent / "data" / "committee.csv")
    df = df.head(10)

    data, metadata = build_graph(df)

    print("=> Data")
    print(data)
    print(metadata)

    transform = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        edge_types=[("mentor", "supervises", "thesis")],
        rev_edge_types=[("thesis", "supervised_by", "mentor")],
    )

    train_data, val_data, test_data = transform(data)

    print("=> Data after RandomLinkSplit")
    print(train_data)
    print(val_data)

    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=[20, 10],
        neg_sampling_ratio=2.0,
        batch_size=64,
        edge_label_index=(
            ("mentor", "supervises", "thesis"),
            train_data["mentor", "supervises", "thesis"].edge_label_index,
        ),
        edge_label=train_data["mentor", "supervises", "thesis"].edge_label,
        shuffle=True,
    )

    model = Model(hidden_channels=32, data=train_data)
    print("=> Model")
    print(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=> Using device: {device}")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 6):
        total_loss = total_examples = 0
        for sampled_data in tqdm.tqdm(train_loader):
            optimizer.zero_grad()

            sampled_data = sampled_data.to(device)

            pred = model.forward(sampled_data)

            loss = F.binary_cross_entropy_with_logits(
                pred,
                sampled_data["mentor", "supervises", "thesis"].edge_label.float(),
            )

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * pred.numel()
            total_examples += pred.numel()
        print(f"Epoch {epoch:03d}, Loss: {total_loss / total_examples:.4f}")


if __name__ == "__main__":
    main()
