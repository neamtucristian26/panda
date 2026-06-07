import numpy as np
import torch
import torch.nn.functional as F
from sklearn.covariance import LedoitWolf
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_energy_score(
    logits: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:

    probs = F.softmax(logits / temperature, dim=1)
    return -temperature * torch.logsumexp(probs, dim=1)


def compute_entropy_score(
    logits: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    
    probs = F.softmax(logits / temperature, dim=1)
    return -torch.sum(probs * torch.log(probs + 1e-10), dim=1)


def compute_max_proxy_similarity_score(
    embeddings: torch.Tensor, proxies: torch.Tensor
) -> torch.Tensor:
    
    similarities = torch.matmul(embeddings, proxies.t())
    max_sim, _ = similarities.max(dim=1)
    return 1.0 - max_sim


def compute_mahalanobis_score(
    embeddings: torch.Tensor,
    class_centroids: torch.Tensor,
    precision_matrix: torch.Tensor,
) -> torch.Tensor:
    
    batch_size = embeddings.shape[0]
    num_classes = class_centroids.shape[0]

    distances = torch.zeros(batch_size, num_classes, device=embeddings.device)
    for c in range(num_classes):
        diff = embeddings - class_centroids[c].unsqueeze(0)
        transformed = torch.matmul(diff, precision_matrix)
        distances[:, c] = torch.sum(diff * transformed, dim=1)

    min_distances, _ = distances.min(dim=1)
    return min_distances


def compute_mahalanobis_params(embeddings: np.ndarray, labels: np.ndarray) -> tuple:
    
    unique_labels = np.sort(np.unique(labels))
    embedding_dim = embeddings.shape[1]

    centroids = np.zeros((len(unique_labels), embedding_dim))
    for i, label in enumerate(unique_labels):
        centroids[i] = embeddings[labels == label].mean(axis=0)

    all_centered = np.concatenate(
        [embeddings[labels == label] - centroids[i] for i, label in enumerate(unique_labels)],
        axis=0,
    )

    lw = LedoitWolf()
    lw.fit(all_centered)
    print(f"Ledoit-Wolf shrinkage coefficient: {lw.shrinkage_:.4f}")

    return centroids, lw.precision_


def calibrate_threshold(
    energy_scores: np.ndarray,
    target_recall: float = 0.95,
    calibration_mode: str = "ood",
) -> float:
    if calibration_mode == "id":
        raise ValueError(
            "ID calibration mode is no longer supported. "
            "Pass OOD calibration samples instead."
        )
    elif calibration_mode != "ood":
        raise ValueError(f"Unknown calibration_mode: {calibration_mode}. Must be 'ood'")

    sorted_scores = np.sort(energy_scores)
    threshold_idx = int(len(sorted_scores) * (1.0 - target_recall))
    return sorted_scores[threshold_idx]


def evaluate_ood_detection(
    id_energy_scores: np.ndarray, ood_energy_scores: np.ndarray, threshold: float = None
) -> dict:

    all_scores = np.concatenate([id_energy_scores, ood_energy_scores])
    all_labels = np.concatenate(
        [np.zeros(len(id_energy_scores)), np.ones(len(ood_energy_scores))]
    )

    auroc = roc_auc_score(all_labels, all_scores)
    aupr = average_precision_score(all_labels, all_scores)

    if threshold is None:
        threshold = calibrate_threshold(id_energy_scores, target_recall=0.95)

    ood_pred_as_ood = (ood_energy_scores > threshold).astype(int)
    id_pred_as_ood = (id_energy_scores > threshold).astype(int)

    tp = np.sum(ood_pred_as_ood == 1)
    fn = np.sum(ood_pred_as_ood == 0)
    fp = np.sum(id_pred_as_ood == 1)
    tn = np.sum(id_pred_as_ood == 0)

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "auroc": auroc,
        "aupr": aupr,
        "fpr": fpr,
        "threshold": threshold,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def detect_ood(energy_score: float, threshold: float) -> bool:
    return energy_score > threshold
