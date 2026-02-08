import argparse
import pickle
import random

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric import seed_everything
from torch_geometric.data import HeteroData

from committee_predictor.data import load_thesis_csv, prepare_thesis_data_splits
from committee_predictor.graph import build_graphs
from committee_predictor.metrics import (
    add_prefix_to_metrics,
    calculate_metrics,
    log_metrics_tb,
)
from committee_predictor.model import Model

writer = SummaryWriter()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the thesis graph model.")

    # Data
    parser.add_argument("--disjoint-train-ratio", type=float, required=True)
    parser.add_argument("--neg-sampling-train-ratio", type=int, required=True)
    parser.add_argument("--neg-sampling-val-test-ratio", type=int, required=True)
    parser.add_argument(
        "--thesis-filter",
        type=int,
        default=0,
        help="Negative number to use only the latest N thesis, "
        "positive number to use only the first N thesis, 0 to use all thesis",
    )
    parser.add_argument("--train-ratio", type=float, required=True)
    parser.add_argument("--val-ratio", type=float, required=True)

    # Training
    parser.add_argument("--num-epochs", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)

    # Model embedding
    parser.add_argument("--node-embedding-dim", type=int, required=True)

    # Model GNN
    parser.add_argument("--gnn-dim", type=int, required=True)
    parser.add_argument("--gnn-num-layers", type=int, required=True)

    # Model classifier
    parser.add_argument("--classifier-dim", type=int, required=True)
    parser.add_argument("--threshold", type=float, required=True)

    return parser.parse_args()


def train_epoch(
    model: Model,
    data: HeteroData,
    optimizer: torch.optim.Optimizer,
) -> float:
    # Returns: training loss
    model.train()

    optimizer.zero_grad()

    y = data["thesis", "has_committee_member", "professor"].edge_label.float()

    pred = model(data)
    loss = F.binary_cross_entropy_with_logits(pred, y)

    loss.backward()
    optimizer.step()

    return loss.item()


def validate(
    model: Model, data: HeteroData, threshold: float
) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Returns: loss, pred probabilities, pred categories, ground truth labels
    model.eval()
    all_pred_probs = []
    all_pred_cats = []
    all_labels = []

    with torch.no_grad():
        pred = model.forward(data)
        labels = data["thesis", "has_committee_member", "professor"].edge_label.float()

        loss = F.binary_cross_entropy_with_logits(
            pred,
            labels.float(),
        )

        pred_probs = torch.sigmoid(pred)

        all_pred_probs.append(pred_probs.cpu())
        all_pred_cats.append((pred_probs > threshold).cpu().int())
        all_labels.append(labels.cpu())

    all_pred_probs = torch.cat(all_pred_probs)
    all_pred_cats = torch.cat(all_pred_cats)
    all_labels = torch.cat(all_labels)

    return (loss.item(), all_pred_probs, all_pred_cats, all_labels)


def validate_and_calculate_metrics(
    model: Model,
    data: HeteroData,
    threshold: float,
) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    loss, pred_probs, pred_cats, labels = validate(
        model=model, data=data, threshold=threshold
    )
    metrics = calculate_metrics(labels, pred_probs, pred_cats)
    return (loss, pred_probs, pred_cats, labels, metrics)


def main():
    args = parse_args()

    # Data
    disjoint_train_ratio = args.disjoint_train_ratio
    neg_sampling_train_ratio = args.neg_sampling_train_ratio
    neg_sampling_val_test_ratio = args.neg_sampling_val_test_ratio
    thesis_filter = args.thesis_filter
    train_ratio = args.train_ratio
    val_ratio = args.val_ratio

    # Training
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate

    # Model embedding
    node_embedding_dim = args.node_embedding_dim

    # Model classifier
    classifier_dim = args.classifier_dim
    threshold = args.threshold

    # Model GNN
    gnn_dim = args.gnn_dim
    gnn_num_layers = args.gnn_num_layers

    seed_everything(42)
    random.seed(42)
    pd.options.display.max_rows = 20
    pd.options.display.max_columns = 20

    # Build and save graph data
    thesis_df = load_thesis_csv()
    professors_lookup, train_df, val_df, test_df = prepare_thesis_data_splits(
        thesis_df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        thesis_filter=thesis_filter,
    )

    graphs_data = build_graphs(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        professors_lookup=professors_lookup,
        disjoint_train_ratio=disjoint_train_ratio,
        neg_train_ratio=neg_sampling_train_ratio,
        neg_val_test_ratio=neg_sampling_val_test_ratio,
    )
    pickle.dump((professors_lookup, *graphs_data), open("graph_data.pkl", "wb"))

    # Load saved graph data from disk
    graphs_data = pickle.load(open("graph_data.pkl", "rb"))

    professors_lookup, train_data, val_data, test_data = graphs_data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=> Using device: {device}")

    model = Model(
        num_mentors=train_data["professor"].num_nodes,
        thesis_features_dim=train_data["thesis"].x.shape[1],
        node_embedding_dim=node_embedding_dim,
        gnn_dim=gnn_dim,
        gnn_num_layers=gnn_num_layers,
        classifier_dim=classifier_dim,
        metadata=train_data.metadata(),
    )

    print("=> Model")
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Move data to target device
    model = model.to(device)
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)

    # TODO: Implement torch checkpointer

    for epoch in range(0, num_epochs + 1):
        if epoch > 0:
            train_epoch(
                model=model,
                data=train_data,
                optimizer=optimizer,
            )

        # Loss and predictions
        train_loss, train_pred_probs, _, train_labels, train_metrics = (
            validate_and_calculate_metrics(
                model=model, data=train_data, threshold=threshold
            )
        )
        writer.add_scalar("Loss/train", train_loss, epoch)
        log_metrics_tb(writer, train_metrics, "train", epoch)

        val_loss, val_pred_probs, _, val_labels, val_metrics = (
            validate_and_calculate_metrics(
                model=model, data=val_data, threshold=threshold
            )
        )
        writer.add_scalar("Loss/val", val_loss, epoch)
        log_metrics_tb(writer, val_metrics, "val", epoch)

        if epoch % 5 == 0:
            writer.add_pr_curve("PR Curve/train", train_labels, train_pred_probs, epoch)
            writer.add_pr_curve("PR Curve/val", val_labels, val_pred_probs, epoch)
            writer.add_embedding(
                model.professor_emb.weight.cpu(),
                metadata=professors_lookup,
                global_step=epoch,
                tag="Mentor Embeddings",
            )

        print(
            f"Epoch {epoch:02d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        writer.flush()

    _, _, _, _, train_metrics = validate_and_calculate_metrics(
        model=model, data=train_data, threshold=threshold
    )
    _, _, _, _, val_metrics = validate_and_calculate_metrics(
        model=model, data=val_data, threshold=threshold
    )
    _, _, _, _, test_metrics = validate_and_calculate_metrics(
        model=model, data=test_data, threshold=threshold
    )

    writer.add_hparams(
        {
            "disjoint_train_ratio": disjoint_train_ratio,
            "neg_sampling_train_ratio": neg_sampling_train_ratio,
            "neg_sampling_val_test_ratio": neg_sampling_val_test_ratio,
            "num_epochs": num_epochs,
            "node_embedding_dim": node_embedding_dim,
            "gnn_dim": gnn_dim,
            "learning_rate": learning_rate,
            "gnn_num_layers": gnn_num_layers,
            "thesis_filter": thesis_filter,
            "classifier_dim": classifier_dim,
            "threshold": threshold,
        },
        add_prefix_to_metrics(val_metrics, prefix="fval"),
    )

    print("=> Metrics at the last epoch:")
    print(
        pd.DataFrame({"train": train_metrics, "val": val_metrics, "test": test_metrics})
    )

    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
