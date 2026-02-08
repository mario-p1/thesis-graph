from collections import defaultdict
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
)
import torch
import torch_geometric

from committee_predictor.model import Model
from committee_predictor.utils import reverse_dict


def calculate_metrics(y_true, y_pred_prob, y_pred_cat) -> dict[str, float]:
    f1 = f1_score(y_true, y_pred_cat)
    accuracy = accuracy_score(y_true, y_pred_cat)
    precision = precision_score(y_true, y_pred_cat)
    recall = recall_score(y_true, y_pred_cat)
    specificity = recall_score(y_true, y_pred_cat, pos_label=0)
    roc_auc = roc_auc_score(y_true, y_pred_prob)
    pr_auc = average_precision_score(y_true, y_pred_prob)
    mcc = matthews_corrcoef(y_true, y_pred_cat)

    return {
        "f1": f1,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "sensitivity": recall,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "mcc": mcc,
    }


def add_prefix_to_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def log_metrics_tb(writer, metrics: dict[str, float], split: str = "", epoch=None):
    for key, value in metrics.items():
        logkey = f"{key.capitalize()}/{split}" if split else key.capitalize()
        writer.add_scalar(logkey, value, epoch)


# TODO: Revise the correctness and usefulness of this function
def get_ranking_metrics(
    data: torch_geometric.data.HeteroData,
    model: Model,
    device: torch.device,
    mentors_dict: dict[int, str],
) -> torch.Tensor:
    mentors_dict = reverse_dict(mentors_dict)

    model.eval()

    ranks_sum = 0
    reciprocal_ranks_sum = 0.0
    n = 0

    average_mentor_rank = defaultdict(lambda: 0)
    ranked_first_count = defaultdict(lambda: 0)

    with torch.no_grad():
        for thesis_id, real_mentor in zip(
            data[("thesis", "supervised_by", "mentor")].edge_label_index[0],
            data[("thesis", "supervised_by", "mentor")].edge_label_index[1],
        ):
            thesis_features = data["thesis"].x[thesis_id].to(device)
            real_mentor = real_mentor.to("cpu").item()

            scores = model.get_prediction_new_thesis(thesis_features)

            rankings = scores.argsort(descending=True).to("cpu").numpy().tolist()

            ranked_first_count[rankings[0]] += 1

            for rank, mentor_id in enumerate(rankings):
                average_mentor_rank[mentor_id] += rank + 1

            rank_of_real_mentor = rankings.index(real_mentor) + 1
            reciprocal_rank = 1.0 / rank_of_real_mentor

            ranks_sum += rank_of_real_mentor
            reciprocal_ranks_sum += reciprocal_rank
            n += 1

    for mentor_id in average_mentor_rank.keys():
        average_mentor_rank[mentor_id] /= n

    return {
        "mean_rank": ranks_sum / n,
        "mean_reciprocal_rank": reciprocal_ranks_sum / n,
        **{
            f"average_mentor_rank_{mentors_dict[mentor_id]}": avg_rank
            for mentor_id, avg_rank in average_mentor_rank.items()
        },
        **{
            f"ranked_first_count_{mentors_dict[mentor_id]}": count
            for mentor_id, count in ranked_first_count.items()
        },
    }
