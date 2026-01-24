import pickle

import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from sklearn.metrics import classification_report
from torch.utils.tensorboard import SummaryWriter
from torch_geometric import seed_everything
from torch_geometric.loader import LinkNeighborLoader

from thesis_graph.graph import build_graphs
from thesis_graph.metrics import calculate_metrics, log_metrics_tb
from thesis_graph.model import Model

writer = SummaryWriter()


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
    ## Data
    disjoint_train_ratio = 0.3
    neg_sampling_train_ratio = 1
    neg_sampling_val_test_ratio = 1

    ## Training
    num_epochs = 100
    learning_rate = 0.0003

    ## Model embedding
    node_embedding_channels = 128

    ## Model GNN
    hidden_channels = 64
    gnn_num_layers = 2

    seed_everything(42)
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
        },
        {},
    )

    # Build and save graph data
    graphs_data = build_graphs(
        disjoint_train_ratio=disjoint_train_ratio,
        neg_train_ratio=neg_sampling_train_ratio,
        neg_val_test_ratio=neg_sampling_val_test_ratio,
    )
    pickle.dump(graphs_data, open("graph_data.pkl", "wb"))

    # Load saved graph data from disk
    graphs_data = pickle.load(open("graph_data.pkl", "rb"))

    mentors_dict, train_data, val_data, _ = graphs_data

    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=[-1],
        batch_size=len(
            train_data["thesis", "supervised_by", "mentor"]["edge_index"][0]
        ),
        edge_label_index=(
            ("thesis", "supervised_by", "mentor"),
            train_data["thesis", "supervised_by", "mentor"].edge_label_index,
        ),
        edge_label=train_data["thesis", "supervised_by", "mentor"].edge_label,
        shuffle=False,
        neg_sampling_ratio=0,
    )

    val_loader = LinkNeighborLoader(
        data=val_data,
        num_neighbors=[-1],
        batch_size=len(val_data["thesis", "supervised_by", "mentor"]["edge_index"][0]),
        edge_label_index=(
            ("thesis", "supervised_by", "mentor"),
            val_data["thesis", "supervised_by", "mentor"].edge_label_index,
        ),
        edge_label=val_data["thesis", "supervised_by", "mentor"].edge_label,
        shuffle=False,
        neg_sampling_ratio=0,
    )

    model = Model(
        num_mentors=train_data["mentor"].num_nodes,
        thesis_features_dim=train_data["thesis"].x.shape[1],
        node_embedding_channels=node_embedding_channels,
        hidden_channels=hidden_channels,
        gnn_num_layers=gnn_num_layers,
        metadata=train_data.metadata(),
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
            )

        # Loss and predictions
        train_loss, train_scores, train_preds, train_labels = validate(
            model, train_loader, device
        )
        writer.add_scalar("Loss/train", train_loss, epoch)

        val_loss, val_scores, val_preds, val_labels = validate(
            model, val_loader, device
        )
        writer.add_scalar("Loss/val", val_loss, epoch)

        # Metrics
        train_metrics = calculate_metrics(train_labels, train_scores, train_preds)
        log_metrics_tb(writer, train_metrics, "train", epoch)

        val_metrics = calculate_metrics(val_labels, val_scores, val_preds)
        log_metrics_tb(writer, val_metrics, "val", epoch)

        if epoch % 5 == 0:
            writer.add_pr_curve("PR Curve/train", train_labels, train_scores, epoch)
            writer.add_pr_curve("PR Curve/val", val_labels, val_scores, epoch)
            writer.add_embedding(
                model.mentor_emb.weight.cpu(),
                metadata=mentors_dict,
                global_step=epoch,
                tag="Mentor Embeddings",
            )

        if val_metrics["pr_auc"] > val_best_metrics.get("pr_auc", 0):
            val_best_epoch = epoch
            val_best_metrics = val_metrics
            val_best_report = classification_report(val_labels, val_preds)

        print(
            f"Epoch {epoch:02d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        writer.flush()

    print(f"=> Val best metrics (epoch: {val_best_epoch}):")
    print(val_best_report)

    _, _, last_epoch_preds, last_epoch_labels = validate(model, val_loader, device)
    print("=> Last epoch metrics:")
    print(classification_report(last_epoch_labels, last_epoch_preds))

    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
