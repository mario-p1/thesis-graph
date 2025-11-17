from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import tqdm
from torch_geometric.loader import LinkNeighborLoader
import mlflow
from mentor_finder.data import build_graph, load_raw_committee_csv
from mentor_finder.model import Model


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

        pred = model(sampled_data)

        loss = F.binary_cross_entropy_with_logits(
            pred,
            sampled_data["thesis", "supervised_by", "mentor"].edge_label,
        )

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
    disjoint_train_ratio = 0.5
    neg_sampling_ratio = 1.0
    add_negative_train_samples = True

    pd.options.display.max_rows = 20
    pd.options.display.max_columns = 20

    df = load_raw_committee_csv(Path(__file__).parent.parent / "data" / "committee.csv")
    # df = df.head(20)

    data, metadata = build_graph(df)

    print("=> Data")
    print(data)
    print(metadata)

    mlflow.log_param("disjoint_train_ratio", disjoint_train_ratio)
    mlflow.log_param("neg_sampling_ratio", neg_sampling_ratio)
    mlflow.log_param("add_negative_train_samples", add_negative_train_samples)

    transform = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        disjoint_train_ratio=disjoint_train_ratio,
        neg_sampling_ratio=neg_sampling_ratio,
        add_negative_train_samples=add_negative_train_samples,
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
        batch_size=64,
        edge_label_index=(
            ("thesis", "supervised_by", "mentor"),
            train_data["thesis", "supervised_by", "mentor"].edge_label_index,
        ),
        edge_label=train_data["thesis", "supervised_by", "mentor"].edge_label,
        shuffle=True,
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

    model = Model(hidden_channels=64, data=train_data)
    print("=> Model")
    print(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=> Using device: {device}")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(1, 16):
        train_loss = train_epoch(model, train_loader, optimizer, device)

        val_loss, val_preds, val_labels = validate(model, val_loader, device)

        # breakpoint()
        val_accuracy = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)
        precision = precision_score(val_labels, val_preds)
        recall = recall_score(val_labels, val_preds)

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
        mlflow.log_metric("val_f1", val_f1, step=epoch)
        mlflow.log_metric("val_precision", precision, step=epoch)
        mlflow.log_metric("val_recall", recall, step=epoch)

        print(
            f"Epoch {epoch:02d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

    _, final_preds, final_labels = validate(model, val_loader, device)
    print("Validation Classification Report:")
    print(
        classification_report(
            final_labels,
            final_preds,
            target_names=["no mentor", "mentor"],
        )
    )


if __name__ == "__main__":
    main()
