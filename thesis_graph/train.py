import pickle

import mlflow
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from sklearn.metrics import classification_report
from torch_geometric import seed_everything
from torch_geometric.loader import LinkNeighborLoader

from thesis_graph.graph import build_graphs
from thesis_graph.metrics import add_prefix_to_metrics, get_metrics, get_ranking_metrics
from thesis_graph.model import Model


def train_epoch(
    model: Model,
    loader: LinkNeighborLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pos_weight: float,
):
    model.train()
    total_loss = total_examples = 0
    pos_weight_tensor = torch.tensor(pos_weight, device=device)
    for sampled_data in tqdm.tqdm(loader):
        optimizer.zero_grad()

        sampled_data = sampled_data.to(device)
        y = sampled_data["thesis", "supervised_by", "mentor"].edge_label

        pred = model(sampled_data)
        loss = F.binary_cross_entropy_with_logits(pred, y, pos_weight=pos_weight_tensor)

        # breakpoint()

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * pred.numel()
        total_examples += pred.numel()

    return total_loss / total_examples


def validate(
    model: Model, loader: LinkNeighborLoader, device: torch.device
) -> tuple[float, list, list, list]:
    # Return: loss, scores, preds, labels
    model.eval()
    all_scores = []
    all_preds = []
    all_labels = []

    total_loss = total_examples = 0
    with torch.no_grad():
        for sampled_data in loader:
            sampled_data = sampled_data.to(device)

            pred = model.forward(sampled_data)
            labels = sampled_data["thesis", "supervised_by", "mentor"].edge_label.cpu()

            all_scores.append(pred.cpu())
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
        torch.cat(all_scores),
        torch.cat(all_preds),
        torch.cat(all_labels),
    )


def main():
    # Hyperparameters
    disjoint_train_ratio = 0.3
    neg_sampling_train_ratio = 5
    pos_weight = neg_sampling_train_ratio
    neg_sampling_val_test_ratio = 10
    num_epochs = 100
    node_embedding_channels = 256
    hidden_channels = 32
    learning_rate = 0.001
    gnn_num_layers = 3

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    seed_everything(42)
    pd.options.display.max_rows = 20
    pd.options.display.max_columns = 20

    mlflow.log_param("disjoint_train_ratio", disjoint_train_ratio)
    mlflow.log_param("neg_sampling_train_ratio", neg_sampling_train_ratio)
    mlflow.log_param("pos_weight", pos_weight)
    mlflow.log_param("neg_sampling_val_test_ratio", neg_sampling_val_test_ratio)
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("node_embedding_channels", node_embedding_channels)
    mlflow.log_param("hidden_channels", hidden_channels)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("gnn_num_layers", gnn_num_layers)

    # Build and save graph data
    graphs_data = build_graphs(disjoint_train_ratio=disjoint_train_ratio)
    pickle.dump(graphs_data, open("graph_data.pkl", "wb"))

    # Load saved graph data from disk
    graphs_data = pickle.load(open("graph_data.pkl", "rb"))

    mentors_dict, train_data, val_data, test_data = graphs_data

    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=[-1],
        batch_size=4096,
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
        num_neighbors=[-1],
        batch_size=4096,
        edge_label_index=(
            ("thesis", "supervised_by", "mentor"),
            val_data["thesis", "supervised_by", "mentor"].edge_label_index,
        ),
        edge_label=val_data["thesis", "supervised_by", "mentor"].edge_label,
        shuffle=False,
        neg_sampling_ratio=neg_sampling_val_test_ratio,
    )

    test_loader = LinkNeighborLoader(
        data=test_data,
        num_neighbors=[-1],
        batch_size=4096,
        edge_label_index=(
            ("thesis", "supervised_by", "mentor"),
            test_data["thesis", "supervised_by", "mentor"].edge_label_index,
        ),
        edge_label=test_data["thesis", "supervised_by", "mentor"].edge_label,
        shuffle=False,
        neg_sampling_ratio=neg_sampling_val_test_ratio,
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

    for epoch in range(0, num_epochs + 1):
        if epoch > 0:
            train_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                pos_weight=pos_weight,
            )

        train_loss, train_scores, train_preds, train_labels = validate(
            model, train_loader, device
        )
        val_loss, val_scores, val_preds, val_labels = validate(
            model, val_loader, device
        )

        val_metrics = get_metrics(val_labels, val_scores, val_preds)
        train_metrics = get_metrics(train_labels, train_scores, train_preds)

        train_ranking_scores = get_ranking_metrics(
            train_data, model, device, mentors_dict
        )
        val_ranking_scores = get_ranking_metrics(val_data, model, device, mentors_dict)

        if val_metrics["pr_auc"] > val_best_metrics.get("pr_auc", 0):
            val_best_epoch = epoch
            val_best_metrics = val_metrics

            val_best_report = classification_report(
                val_labels,
                val_preds,
            )

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metrics(add_prefix_to_metrics(train_metrics, "train"), step=epoch)
        mlflow.log_metrics(
            add_prefix_to_metrics(train_ranking_scores, "train"), step=epoch
        )

        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metrics(add_prefix_to_metrics(val_metrics, "val"), step=epoch)
        mlflow.log_metrics(add_prefix_to_metrics(val_ranking_scores, "val"), step=epoch)

        print(
            f"Epoch {epoch:02d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

    mlflow.log_metric("val_best_epoch", val_best_epoch)
    mlflow.log_metrics(add_prefix_to_metrics(val_best_metrics, "val_best"))

    print(f"=> Val best metrics (epoch: {val_best_epoch}):")
    print(val_best_report)

    _, _, last_epoch_preds, last_epoch_labels = validate(model, val_loader, device)
    print("=> Latest epoch metrics:")
    print(classification_report(last_epoch_labels, last_epoch_preds))

    # _, test_scores, test_preds, test_labels = validate(model, test_loader, device)
    # print("=> Test metrics:")
    # test_metrics = get_metrics(test_labels, test_scores, test_preds)
    # print(classification_report(test_labels, test_preds))
    # print(pd.DataFrame({"test": test_metrics}))


if __name__ == "__main__":
    main()
