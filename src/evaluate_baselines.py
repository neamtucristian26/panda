import argparse
import os
import pickle
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import top_k_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

from data_loader import create_data_loaders
from ood_detection import evaluate_ood_detection
from utils import (
    load_model_to_label_mapping,
    load_split_config,
    load_split_metadata,
    set_seed,
)

warnings.filterwarnings("ignore")


def extract_features_and_labels(data_loader, desc="Extracting features"):
    X_list = []
    y_list = []

    for embeddings, labels, _ in tqdm(data_loader, desc=desc):
        X_list.append(embeddings.numpy())
        y_list.append(labels.numpy())

    X = np.vstack(X_list)
    y = np.hstack(y_list)

    return X, y


def train_logistic_regression(X_train, y_train):
    print("\nTraining Logistic Regression...")
    clf = LogisticRegression(solver="lbfgs", max_iter=1000, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)
    print("Training complete.")
    return clf


def train_knn(X_train, y_train):
    print("\nTraining k-Nearest Neighbors (k=21, cosine)...")
    clf = KNeighborsClassifier(n_neighbors=21, metric="cosine", n_jobs=-1)
    clf.fit(X_train, y_train)
    print("Training complete.")
    return clf


def evaluate_classification(model, X_test, y_test, num_classes, model_name="Baseline"):
    print(f"\nEvaluating {model_name} classification...")

    y_pred = model.predict(X_test)
    acc_top1 = 100.0 * (y_pred == y_test).sum() / len(y_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
        acc_top5 = 100.0 * top_k_accuracy_score(
            y_test, y_proba, k=5, labels=range(num_classes)
        )
    else:
        acc_top5 = acc_top1

    return {
        "acc_top1": acc_top1,
        "acc_top5": acc_top5,
        "predictions": y_pred,
        "labels": y_test,
    }


def compute_ood_scores(model, X, method="entropy"):
    if method == "entropy":
        proba = model.predict_proba(X)
        return -np.sum(proba * np.log(proba + 1e-10), axis=1)
    elif method == "distance":
        distances, _ = model.kneighbors(X)
        return distances.mean(axis=1)
    else:
        raise ValueError(f"Unknown OOD scoring method: {method}")


def evaluate_ood(model, X_id, X_ood, ood_method, model_name="Baseline"):
    print(f"\nEvaluating {model_name} OOD detection using {ood_method}...")

    id_scores = compute_ood_scores(model, X_id, method=ood_method)
    ood_scores = compute_ood_scores(model, X_ood, method=ood_method)

    sorted_ood_scores = np.sort(ood_scores)
    threshold = sorted_ood_scores[int(0.05 * len(sorted_ood_scores))]
    print(f"Calibrated threshold (95% OOD recall): {threshold:.4f}")

    ood_metrics = evaluate_ood_detection(id_scores, ood_scores, threshold)
    return ood_metrics, id_scores, ood_scores, threshold


def main(args):
    set_seed(args.seed)

    print("Loading split metadata...")
    id_models, ood_cal_models, ood_test_models = load_split_metadata(args.metadata_dir)
    split_config = load_split_config(args.metadata_dir)
    model_to_label = load_model_to_label_mapping(
        os.path.join(args.metadata_dir, "model_to_label.json")
    )
    num_classes = len(set(model_to_label.values()))
    print(f"Number of classes: {num_classes}")

    print("Creating data loaders...")
    train_loader, val_loader, id_test_loader, ood_cal_loader, ood_test_loader = (
        create_data_loaders(
            embeddings_dir=args.embeddings_dir,
            id_models=id_models,
            ood_cal_models=ood_cal_models,
            ood_test_models=ood_test_models,
            model_to_label=model_to_label,
            train_ratio=split_config["train_ratio"],
            val_ratio=split_config["val_ratio"],
            test_ratio=split_config["test_ratio"],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=split_config["seed"],
            max_samples_per_model=split_config["max_samples_per_model"],
        )
    )
    print("\nExtracting features from data loaders...")
    X_train, y_train = extract_features_and_labels(
        train_loader, "Extracting training features"
    )
    X_test, y_test = extract_features_and_labels(
        id_test_loader, "Extracting test features"
    )
    X_ood_test, _ = extract_features_and_labels(
        ood_test_loader, "Extracting OOD test features"
    )

    print(f"Training Data: {X_train.shape}")
    print(f"Test Data:     {X_test.shape}")
    print(f"OOD Data:      {X_ood_test.shape}")

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("LOGISTIC REGRESSION BASELINE")
    print("=" * 80)

    lr_model = train_logistic_regression(X_train, y_train)

    with open(
        os.path.join(args.output_dir, "logistic_regression_model.pkl"), "wb"
    ) as f:
        pickle.dump(lr_model, f)

    lr_metrics = evaluate_classification(
        lr_model, X_test, y_test, num_classes, "Logistic Regression"
    )
    print(f'  Top-1 Accuracy: {lr_metrics["acc_top1"]:.2f}%')
    print(f'  Top-5 Accuracy: {lr_metrics["acc_top5"]:.2f}%')

    lr_ood_metrics, lr_id_scores, lr_ood_scores, lr_thresh = evaluate_ood(
        lr_model,
        X_test,
        X_ood_test,
        ood_method="entropy",
        model_name="Logistic Regression",
    )
    print(f'  AUROC: {lr_ood_metrics["auroc"]:.4f}')
    print(f'  FPR: {lr_ood_metrics["fpr"]:.4f}')

    print("\n" + "=" * 80)
    print("k-NEAREST NEIGHBORS BASELINE")
    print("=" * 80)

    knn_model = train_knn(X_train, y_train)

    with open(os.path.join(args.output_dir, "knn_model.pkl"), "wb") as f:
        pickle.dump(knn_model, f)

    knn_metrics = evaluate_classification(
        knn_model, X_test, y_test, num_classes, "k-NN"
    )
    print(f'  Top-1 Accuracy: {knn_metrics["acc_top1"]:.2f}%')
    print(f'  Top-5 Accuracy: {knn_metrics["acc_top5"]:.2f}%')

    knn_ood_metrics, knn_id_scores, knn_ood_scores, knn_thresh = evaluate_ood(
        knn_model, X_test, X_ood_test, ood_method="distance", model_name="k-NN"
    )
    print(f'  AUROC: {knn_ood_metrics["auroc"]:.4f}')
    print(f'  FPR: {knn_ood_metrics["fpr"]:.4f}')

    results_path = os.path.join(args.output_dir, "baseline_results.txt")
    with open(results_path, "w") as f:
        f.write("BASELINE COMPARISON RESULTS\n")
        f.write("=" * 40 + "\n")
        f.write("LOGISTIC REGRESSION:\n")
        f.write(f"  Acc:   {lr_metrics['acc_top1']:.2f}%\n")
        f.write(f"  AUROC: {lr_ood_metrics['auroc']:.4f}\n")
        f.write(f"  FPR: {lr_ood_metrics['fpr']:.4f}\n\n")

        f.write("k-NEAREST NEIGHBORS:\n")
        f.write(f"  Acc:   {knn_metrics['acc_top1']:.2f}%\n")
        f.write(f"  AUROC: {knn_ood_metrics['auroc']:.4f}\n")
        f.write(f"  FPR: {knn_ood_metrics['fpr']:.4f}\n")

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings_dir", type=str, default="data/w2vbert_embeddings/layer4"
    )
    parser.add_argument("--metadata_dir", type=str, default="metadata/non_merged")
    parser.add_argument(
        "--output_dir", type=str, default="results/non_merged/baselines"
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--merge_map",
        type=str,
        default=None,
        help="Path to merge_map.json (unused directly; merged labels are loaded from metadata)",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
