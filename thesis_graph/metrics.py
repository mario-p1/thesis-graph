from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def get_metrics(y_true, y_preds) -> dict[str, float]:
    f1 = f1_score(y_true, y_preds, average="weighted")
    accuracy = accuracy_score(y_true, y_preds)
    precision = precision_score(y_true, y_preds)
    recall = recall_score(y_true, y_preds)
    specificity = recall_score(y_true, y_preds, pos_label=0)

    return {
        "f1": f1,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "sensitivity": recall,
    }


def add_prefix_to_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}
