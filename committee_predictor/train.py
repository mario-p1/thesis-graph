import argparse
import pickle
import random

import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from sklearn.metrics import classification_report
from torch.utils.tensorboard import SummaryWriter
from torch_geometric import seed_everything
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader

from committee_predictor.data import load_thesis_csv, prepare_thesis_data_splits
from committee_predictor.graph import build_graphs
from committee_predictor.metrics import calculate_metrics, log_metrics_tb
from committee_predictor.model import Model

writer = SummaryWriter()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the thesis graph model.")

    # Data
    parser.add_argument("--disjoint-train-ratio", type=float)
    parser.add_argument("--neg-sampling-train-ratio", type=int)
    parser.add_argument("--neg-sampling-val-test-ratio", type=int)
    parser.add_argument(
        "--thesis-filter",
        type=int,
        default=0,
        help="Negative number to use only the latest N thesis, "
        "positive number to use only the first N thesis, 0 to use all thesis",
    )

    # Training
    parser.add_argument("--num-epochs", type=int)
    parser.add_argument("--learning-rate", type=float)

    # Model embedding
    parser.add_argument("--node-embedding-channels", type=int)

    # Model GNN
    parser.add_argument("--hidden-channels", type=int)
    parser.add_argument("--gnn-num-layers", type=int)

    return parser.parse_args()


def train_epoch(
    model: Model,
    loader: LinkNeighborLoader | None,
    data: HeteroData | None,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    model.train()
    total_loss = total_examples = 0

    if loader is not None:
        data_iter = loader
    else:
        data_iter = [data]

    for sampled_data in tqdm.tqdm(data_iter):
        optimizer.zero_grad()

        sampled_data = sampled_data.to(device)
        y = sampled_data["thesis", "has_committee_member", "professor"].edge_label

        pred = model(sampled_data)
        loss = F.binary_cross_entropy_with_logits(pred, y)

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * pred.numel()
        total_examples += pred.numel()

    return total_loss / total_examples


def validate(
    model: Model,
    loader: LinkNeighborLoader | None,
    data: HeteroData | None,
    device: torch.device,
) -> tuple[float, list, list, list]:
    # Return: loss, scores, preds, labels
    model.eval()
    all_scores = []
    all_preds = []
    all_labels = []

    total_loss = total_examples = 0
    with torch.no_grad():
        if loader is not None:
            data_iter = loader
        else:
            data_iter = [data]
        for sampled_data in data_iter:
            sampled_data = sampled_data.to(device)

            pred = model.forward(sampled_data)
            labels = sampled_data[
                "thesis", "has_committee_member", "professor"
            ].edge_label

            all_scores.append(pred.cpu())
            all_preds.append((pred > 0).cpu().int())
            all_labels.append(labels.cpu())

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
    args = parse_args()

    # Data
    disjoint_train_ratio = args.disjoint_train_ratio
    neg_sampling_train_ratio = args.neg_sampling_train_ratio
    neg_sampling_val_test_ratio = args.neg_sampling_val_test_ratio
    thesis_filter = args.thesis_filter

    # Training
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate

    # Model embedding
    node_embedding_channels = args.node_embedding_channels

    # Model GNN
    hidden_channels = args.hidden_channels
    gnn_num_layers = args.gnn_num_layers

    seed_everything(42)
    random.seed(42)
    pd.options.display.max_rows = 20
    pd.options.display.max_columns = 20

    writer.add_hparams(
        {
            "disjoint_train_ratio": disjoint_train_ratio,
            "neg_sampling_train_ratio": neg_sampling_train_ratio,
            "neg_sampling_val_test_ratio": neg_sampling_val_test_ratio,
            "num_epochs": num_epochs,
            "node_embedding_channels": node_embedding_channels,
            "hidden_channels": hidden_channels,
            "learning_rate": learning_rate,
            "gnn_num_layers": gnn_num_layers,
            "thesis_filter": thesis_filter,
        },
        {},
    )

    # Build and save graph data
    thesis_df = load_thesis_csv()
    professors_lookup, train_df, val_df, test_df = prepare_thesis_data_splits(
        thesis_df, train_ratio=0.8, val_ratio=0.1, thesis_filter=thesis_filter
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
        node_embedding_channels=node_embedding_channels,
        hidden_channels=hidden_channels,
        gnn_num_layers=gnn_num_layers,
        metadata=train_data.metadata(),
    )
    model = model.to(device)
    print("=> Model")
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # TODO: Implement torch checkpointer

    for epoch in range(0, num_epochs + 1):
        if epoch > 0:
            train_epoch(
                model=model,
                loader=None,
                data=train_data,
                optimizer=optimizer,
                device=device,
            )

        # Loss and predictions
        train_loss, train_pred_prob, train_pred_cat, train_labels = validate(
            model, None, train_data, device
        )
        writer.add_scalar("Loss/train", train_loss, epoch)

        val_loss, val_pred_prob, val_pred_cat, val_labels = validate(
            model, None, val_data, device
        )
        writer.add_scalar("Loss/val", val_loss, epoch)

        # Metrics
        train_metrics = calculate_metrics(train_labels, train_pred_prob, train_pred_cat)
        log_metrics_tb(writer, train_metrics, "train", epoch)

        val_metrics = calculate_metrics(val_labels, val_pred_prob, val_pred_cat)
        log_metrics_tb(writer, val_metrics, "val", epoch)

        if epoch % 5 == 0:
            writer.add_pr_curve("PR Curve/train", train_labels, train_pred_prob, epoch)
            writer.add_pr_curve("PR Curve/val", val_labels, val_pred_prob, epoch)
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

    _, _, last_epoch_pred_cat, last_epoch_labels = validate(
        model, None, val_data, device
    )
    print("=> Last epoch metrics:")
    print(classification_report(last_epoch_labels, last_epoch_pred_cat))

    print("=> Final test metrics:")
    _, test_pred_prob, test_pred_cat, test_labels = validate(
        model, None, test_data, device
    )
    test_metrics = calculate_metrics(test_labels, test_pred_prob, test_pred_cat)
    print(classification_report(test_labels, test_pred_cat))
    print(test_metrics)

    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
