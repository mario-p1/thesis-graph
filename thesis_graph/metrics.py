from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
)


def get_metrics(y_true, y_scores, y_preds) -> dict[str, float]:
    f1 = f1_score(y_true, y_preds, average="weighted")
    accuracy = accuracy_score(y_true, y_preds)
    precision = precision_score(y_true, y_preds)
    recall = recall_score(y_true, y_preds)
    specificity = recall_score(y_true, y_preds, pos_label=0)
    roc_auc = roc_auc_score(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)

    return {
        "f1": f1,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "sensitivity": recall,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def add_prefix_to_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}
