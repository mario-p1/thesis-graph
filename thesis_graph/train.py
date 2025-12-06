import pickle
import mlflow
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import tqdm
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch_geometric import seed_everything
from torch_geometric.loader import LinkNeighborLoader

from thesis_graph.data import build_graph, load_researchers_csv, load_thesis_csv
from thesis_graph.metrics import add_prefix_to_metrics, get_metrics
from thesis_graph.model import Model
from thesis_graph.utils import base_data_path


def train_epoch(
    model: Model,
    loader: LinkNeighborLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    model.train()
    total_loss = total_examples = 0
    for sampled_data in tqdm.tqdm(loader):
        optimizer.zero_grad()

        sampled_data = sampled_data.to(device)
        y = sampled_data["thesis", "supervised_by", "mentor"].edge_label

        pred = model(sampled_data)
        loss = F.binary_cross_entropy_with_logits(pred, y)

        # breakpoint()

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * pred.numel()
        total_examples += pred.numel()

    return total_loss / total_examples


def validate(
    model: Model, loader: LinkNeighborLoader, device: torch.device
) -> tuple[float, list, list]:
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = total_examples = 0
    with torch.no_grad():
        for sampled_data in loader:
            sampled_data = sampled_data.to(device)

            pred = model.forward(sampled_data)
            labels = sampled_data["thesis", "supervised_by", "mentor"].edge_label.cpu()

            all_preds.append((pred > 0).cpu().int())
            all_labels.append(labels)

            loss = F.binary_cross_entropy_with_logits(
                pred,
                labels.float(),
            )
            total_loss += loss.item() * pred.numel()
            total_examples += pred.numel()

    return (
        total_loss / total_examples,
        torch.cat(all_preds),
        torch.cat(all_labels),
    )


def main():
    # Hyperparameters
    disjoint_train_ratio = 0.7
    neg_sampling_train_ratio = 1
    neg_sampling_val_test_ratio = 1.0
    num_epochs = 16
    node_embedding_channels = 64
    hidden_channels = 32
    learning_rate = 0.0001
    gnn_num_layers = 5

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    seed_everything(42)
    pd.options.display.max_rows = 20
    pd.options.display.max_columns = 20

    df = load_thesis_csv(base_data_path / "committee.csv")
    # df = df.head(20)

    researchers_df = load_researchers_csv(base_data_path / "researchers.csv")

    # data, metadata = build_graph(df, researchers_df)
    # pickle.dump((data, metadata), open("graph_data.pkl", "wb"))

    data, metadata = pickle.load(open("graph_data.pkl", "rb"))

    print("=> Data")
    print(data)
    print(metadata)

    mlflow.log_param("disjoint_train_ratio", disjoint_train_ratio)
    mlflow.log_param("neg_sampling_train_ratio", neg_sampling_train_ratio)
    mlflow.log_param("neg_sampling_val_test_ratio", neg_sampling_val_test_ratio)
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("node_embedding_channels", node_embedding_channels)
    mlflow.log_param("hidden_channels", hidden_channels)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("gnn_num_layers", gnn_num_layers)

    transform = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        disjoint_train_ratio=disjoint_train_ratio,
        neg_sampling_ratio=neg_sampling_val_test_ratio,
        add_negative_train_samples=False,
        edge_types=[("thesis", "supervised_by", "mentor")],
        rev_edge_types=[("mentor", "supervises", "thesis")],
    )

    train_data, val_data, test_data = transform(data)

    print("=> Data after RandomLinkSplit")
    print(train_data)
    print(val_data)

    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=[20, 10],
        batch_size=256,
        edge_label_index=(
            ("thesis", "supervised_by", "mentor"),
            train_data["thesis", "supervised_by", "mentor"].edge_label_index,
        ),
        edge_label=train_data["thesis", "supervised_by", "mentor"].edge_label,
        shuffle=True,
        neg_sampling_ratio=neg_sampling_train_ratio,
    )

    val_loader = LinkNeighborLoader(
        data=val_data,
        num_neighbors=[20, 10],
        batch_size=64,
        edge_label_index=(
            ("thesis", "supervised_by", "mentor"),
            val_data["thesis", "supervised_by", "mentor"].edge_label_index,
        ),
        edge_label=val_data["thesis", "supervised_by", "mentor"].edge_label,
        shuffle=False,
    )

    model = Model(
        node_embedding_channels=node_embedding_channels,
        hidden_channels=hidden_channels,
        gnn_num_layers=gnn_num_layers,
        data=train_data,
    )
    print("=> Model")
    print(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=> Using device: {device}")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    val_best_epoch = -1
    val_best_metrics = {}
    val_best_report = ""

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)

        val_loss, val_preds, val_labels = validate(model, val_loader, device)
        _, train_preds, train_labels = validate(model, train_loader, device)

        val_metrics = get_metrics(val_labels, val_preds)
        train_metrics = get_metrics(train_labels, train_preds)

        if val_metrics["f1"] > val_best_metrics.get("f1", 0):
            val_best_epoch = epoch
            val_best_metrics = val_metrics

            val_best_report = classification_report(
                val_labels,
                val_preds,
            )

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)

        mlflow.log_metrics(add_prefix_to_metrics(val_metrics, "val"), step=epoch)
        mlflow.log_metrics(add_prefix_to_metrics(train_metrics, "train"), step=epoch)

        print(
            f"Epoch {epoch:02d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

    mlflow.log_metrics(add_prefix_to_metrics(val_best_metrics, "val_best"))

    print(f"=> Val best metrics (epoch: {val_best_epoch}):")
    print(val_best_report)

    _, final_preds, final_labels = validate(model, val_loader, device)
    print("=> Latest epoch metrics:")
    print(classification_report(final_labels, final_preds))


if __name__ == "__main__":
    main()
