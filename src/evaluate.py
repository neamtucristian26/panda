import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from data_loader import create_data_loaders
from model import create_model
from ood_detection import (
    compute_energy_score,
    compute_entropy_score,
    compute_mahalanobis_score,
    compute_max_proxy_similarity_score,
    evaluate_ood_detection,
)
from utils import (
    apply_config_to_parser,
    get_device,
    load_merge_map,
    load_model_to_label_mapping,
    load_split_config,
    load_split_metadata,
    set_seed,
)


def evaluate_classification(model, data_loader, device, num_classes):
    model.eval()

    all_predictions = []
    all_labels = []
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for embeddings, labels, _ in tqdm(
            data_loader, desc="Evaluating classification"
        ):
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            embeddings_proj, _ = model(embeddings)
            logits = model.get_logits(embeddings_proj)

            _, pred = logits.max(1)
            correct_top1 += pred.eq(labels).sum().item()

            _, pred_top5 = logits.topk(5, 1, True, True)
            pred_top5 = pred_top5.t()
            correct_top5 += (
                pred_top5.eq(labels.view(1, -1).expand_as(pred_top5)).sum().item()
            )

            total += labels.size(0)
            all_predictions.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    per_class_acc = {}
    for class_idx in range(num_classes):
        mask = all_labels == class_idx
        if mask.sum() > 0:
            per_class_acc[class_idx] = (
                100.0 * (all_predictions[mask] == class_idx).sum() / mask.sum()
            )

    return {
        "acc_top1": 100.0 * correct_top1 / total,
        "acc_top5": 100.0 * correct_top5 / total,
        "per_class_acc": per_class_acc,
        "confusion_matrix": confusion_matrix(all_labels, all_predictions),
        "predictions": all_predictions,
        "labels": all_labels,
    }


def evaluate_ood(
    model, id_loader, ood_loader, device, thresholds, mahalanobis_params=None
):
    model.eval()

    if isinstance(thresholds, (int, float)):
        thresholds = {"energy": float(thresholds)}

    methods = list(thresholds.keys())
    has_mahalanobis = "mahalanobis" in methods and mahalanobis_params is not None

    if has_mahalanobis:
        centroids_tensor = torch.from_numpy(mahalanobis_params[0]).float().to(device)
        precision_tensor = torch.from_numpy(mahalanobis_params[1]).float().to(device)

    id_scores = {m: [] for m in methods}
    ood_scores = {m: [] for m in methods}

    def collect_scores(loader, scores_dict, desc):
        with torch.no_grad():
            for embeddings, _, _ in tqdm(loader, desc=desc):
                embeddings = embeddings.to(device)
                embeddings_proj, proxies = model(embeddings)
                logits = model.get_logits(embeddings_proj)

                if "energy" in methods:
                    scores_dict["energy"].extend(
                        compute_energy_score(logits).cpu().numpy().tolist()
                    )
                if "entropy" in methods:
                    scores_dict["entropy"].extend(
                        compute_entropy_score(logits).cpu().numpy().tolist()
                    )
                if "max_proxy_similarity" in methods:
                    scores_dict["max_proxy_similarity"].extend(
                        compute_max_proxy_similarity_score(embeddings_proj, proxies)
                        .cpu()
                        .numpy()
                        .tolist()
                    )
                if has_mahalanobis:
                    scores_dict["mahalanobis"].extend(
                        compute_mahalanobis_score(
                            embeddings_proj, centroids_tensor, precision_tensor
                        )
                        .cpu()
                        .numpy()
                        .tolist()
                    )

    collect_scores(id_loader, id_scores, "Computing ID scores")
    collect_scores(ood_loader, ood_scores, "Computing OOD scores")

    for m in methods:
        id_scores[m] = np.array(id_scores[m])
        ood_scores[m] = np.array(ood_scores[m])

    all_metrics = {
        m: evaluate_ood_detection(id_scores[m], ood_scores[m], thresholds[m])
        for m in methods
        if not (m == "mahalanobis" and not has_mahalanobis)
    }

    return all_metrics, id_scores, ood_scores


def analyze_confusion_pairs(conf_matrix, model_to_label, save_path, top_k=20):
    label_to_model = {v: k for k, v in model_to_label.items()}

    confusion_pairs = []
    num_classes = conf_matrix.shape[0]

    for true_label in range(num_classes):
        for pred_label in range(num_classes):
            if true_label == pred_label:
                continue
            count = conf_matrix[true_label, pred_label]
            if count > 0:
                total_true = conf_matrix[true_label, :].sum()
                confusion_pairs.append(
                    {
                        "true_model": label_to_model.get(
                            true_label, f"Class_{true_label}"
                        ),
                        "pred_model": label_to_model.get(
                            pred_label, f"Class_{pred_label}"
                        ),
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "count": count,
                        "percentage": (
                            (count / total_true * 100) if total_true > 0 else 0
                        ),
                        "total_true": total_true,
                    }
                )

    confusion_pairs.sort(key=lambda x: x["count"], reverse=True)

    with open(save_path, "w") as f:
        f.write("=" * 100 + "\n")
        f.write("CONFUSION MATRIX ANALYSIS - Most Confused Deepfake System Pairs\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"Total number of confusion pairs: {len(confusion_pairs)}\n")
        f.write(
            f"Showing top {min(top_k, len(confusion_pairs))} most confused pairs\n\n"
        )

        f.write("-" * 100 + "\n")
        f.write(
            f"{'Rank':<6} {'True System':<25} {'Predicted As':<25} {'Count':<8} {'% of True':<12} {'Total':<8}\n"
        )
        f.write("-" * 100 + "\n")

        for rank, pair in enumerate(confusion_pairs[:top_k], 1):
            f.write(
                f"{rank:<6} {pair['true_model']:<25} {pair['pred_model']:<25} "
                f"{pair['count']:<8} {pair['percentage']:<12.2f} {pair['total_true']:<8}\n"
            )

        f.write("-" * 100 + "\n\n")

        f.write("=" * 100 + "\n")
        f.write("SYSTEMS WITH MOST MISCLASSIFICATIONS (Hardest to Classify)\n")
        f.write("=" * 100 + "\n\n")

        misclass_per_system = {}
        for pair in confusion_pairs:
            true_model = pair["true_model"]
            if true_model not in misclass_per_system:
                misclass_per_system[true_model] = {
                    "total_misclassified": 0,
                    "total_samples": pair["total_true"],
                    "confused_with": [],
                }
            misclass_per_system[true_model]["total_misclassified"] += pair["count"]
            misclass_per_system[true_model]["confused_with"].append(
                (pair["pred_model"], pair["count"], pair["percentage"])
            )

        sorted_systems = sorted(
            misclass_per_system.items(),
            key=lambda x: x[1]["total_misclassified"],
            reverse=True,
        )

        for rank, (system, stats) in enumerate(sorted_systems[:top_k], 1):
            total = stats["total_samples"]
            misclass = stats["total_misclassified"]
            error_rate = (misclass / total * 100) if total > 0 else 0

            f.write(f"\n{rank}. {system}\n")
            f.write(
                f"   Total samples: {total}, Misclassified: {misclass}, Error rate: {error_rate:.2f}%\n"
            )
            f.write("   Most confused with:\n")

            stats["confused_with"].sort(key=lambda x: x[1], reverse=True)
            for pred_model, count, pct in stats["confused_with"][:5]:
                f.write(f"      - {pred_model}: {count} times ({pct:.2f}%)\n")

        f.write("\n" + "=" * 100 + "\n")
        f.write("BIDIRECTIONAL CONFUSION PAIRS (Systems confused with each other)\n")
        f.write("=" * 100 + "\n\n")

        bidirectional_pairs = []
        processed = set()

        for pair1 in confusion_pairs:
            if (pair1["true_label"], pair1["pred_label"]) in processed:
                continue

            for pair2 in confusion_pairs:
                if (
                    pair1["true_label"] == pair2["pred_label"]
                    and pair1["pred_label"] == pair2["true_label"]
                ):

                    total_confusion = pair1["count"] + pair2["count"]
                    bidirectional_pairs.append(
                        {
                            "system1": pair1["true_model"],
                            "system2": pair1["pred_model"],
                            "count1to2": pair1["count"],
                            "count2to1": pair2["count"],
                            "total": total_confusion,
                        }
                    )

                    processed.add((pair1["true_label"], pair1["pred_label"]))
                    processed.add((pair1["pred_label"], pair1["true_label"]))
                    break

        bidirectional_pairs.sort(key=lambda x: x["total"], reverse=True)

        f.write(f"Found {len(bidirectional_pairs)} bidirectional confusion pairs\n")
        f.write(f"Showing top {min(top_k, len(bidirectional_pairs))} pairs\n\n")

        f.write("-" * 100 + "\n")
        f.write(
            f"{'Rank':<6} {'System 1':<25} {'System 2':<25} {'1->2':<8} {'2->1':<8} {'Total':<8}\n"
        )
        f.write("-" * 100 + "\n")

        for rank, pair in enumerate(bidirectional_pairs[:top_k], 1):
            f.write(
                f"{rank:<6} {pair['system1']:<25} {pair['system2']:<25} "
                f"{pair['count1to2']:<8} {pair['count2to1']:<8} {pair['total']:<8}\n"
            )

        f.write("-" * 100 + "\n")

    print(f"Confusion analysis saved to {save_path}")

    return confusion_pairs, misclass_per_system, bidirectional_pairs


def main(args):
    set_seed(args.seed)

    device = get_device()
    print(f"Using device: {device}")

    print("Loading split metadata...")
    id_models, ood_cal_models, ood_test_models = load_split_metadata(args.metadata_dir)
    split_config = load_split_config(args.metadata_dir)

    model_to_label = load_model_to_label_mapping(
        os.path.join(args.metadata_dir, "model_to_label.json")
    )
    num_classes = len(set(model_to_label.values()))
    print(f"Number of classes: {num_classes}")

    print("Creating data loaders...")
    _, _, id_test_loader, _, ood_test_loader = create_data_loaders(
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

    print("Creating model...")
    model = create_model(
        input_dim=args.input_dim,
        hidden_dims=args.hidden_dims,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        num_classes=num_classes,
    )
    model = model.to(device)

    checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint_file)
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f'Loaded checkpoint from epoch {checkpoint["epoch"]}')

    threshold_metadata_path = os.path.join(
        args.checkpoint_dir, "threshold_metadata.json"
    )
    mahalanobis_params = None

    if os.path.exists(threshold_metadata_path):
        with open(threshold_metadata_path, "r") as f:
            threshold_metadata = json.load(f)
        if "thresholds" in threshold_metadata:
            ood_thresholds = threshold_metadata["thresholds"]
            print("OOD thresholds (multi-method):")
            for method, thresh in ood_thresholds.items():
                print(f"  {method}: {thresh:.4f}")

            mahalanobis_path = os.path.join(
                args.checkpoint_dir, "mahalanobis_params.npz"
            )
            if "mahalanobis" in ood_thresholds and os.path.exists(mahalanobis_path):
                data = np.load(mahalanobis_path)
                mahalanobis_params = (data["centroids"], data["precision_matrix"])
                print(
                    f"  Loaded Mahalanobis params: {mahalanobis_params[0].shape[0]} centroids"
                )
        else:
            ood_thresholds = {"energy": threshold_metadata["threshold_value"]}
            print(f"OOD threshold (energy): {ood_thresholds['energy']:.4f}")
    else:
        threshold_path = os.path.join(args.checkpoint_dir, "ood_threshold.txt")
        with open(threshold_path, "r") as f:
            ood_thresholds = {"energy": float(f.read().strip())}
        print(f"OOD threshold (energy): {ood_thresholds['energy']:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("Evaluating classification on ID test set...")
    print("=" * 80)
    id_test_metrics = evaluate_classification(
        model, id_test_loader, device, num_classes
    )

    print("\nID Test Classification Metrics:")
    print(f'  Top-1 Accuracy: {id_test_metrics["acc_top1"]:.2f}%')
    print(f'  Top-5 Accuracy: {id_test_metrics["acc_top5"]:.2f}%')

    # When using merged labels, show architecture names in confusion analysis
    confusion_label_map = model_to_label
    if args.merge_map and os.path.exists(args.merge_map):
        merge_map = load_merge_map(args.merge_map)
        arch_to_label = {}
        for model_name, label in model_to_label.items():
            arch_name = merge_map.get(model_name, model_name)
            arch_to_label[arch_name] = label
        confusion_label_map = arch_to_label

    print("\nAnalyzing confusion matrix...")
    confusion_analysis_path = os.path.join(args.output_dir, "confusion_analysis.txt")
    analyze_confusion_pairs(
        id_test_metrics["confusion_matrix"],
        confusion_label_map,
        confusion_analysis_path,
        top_k=30,
    )

    print("\n" + "=" * 80)
    print("Evaluating OOD detection (ID test vs OOD test)...")
    print("=" * 80)
    all_ood_metrics, id_scores, ood_scores = evaluate_ood(
        model,
        id_test_loader,
        ood_test_loader,
        device,
        ood_thresholds,
        mahalanobis_params=mahalanobis_params,
    )

    print("\n" + "=" * 80)
    print("OOD Detection Method Comparison")
    print("=" * 80)
    header = f"{'Method':<22} {'AUROC':>8} {'AUPR':>8} {'FPR':>8} {'Recall':>8} {'Prec':>8} {'F1':>8} {'Acc':>8}"
    print(header)
    print("-" * len(header))

    best_method = min(all_ood_metrics, key=lambda m: all_ood_metrics[m]["fpr"])

    for method, metrics in all_ood_metrics.items():
        marker = " *" if method == best_method else ""
        print(
            f"{method:<22} {metrics['auroc']:>8.4f} {metrics['aupr']:>8.4f} "
            f"{metrics['fpr']:>8.4f} {metrics['recall']:>8.4f} "
            f"{metrics['precision']:>8.4f} {metrics['f1']:>8.4f} "
            f"{metrics['accuracy']:>8.4f}{marker}"
        )
    print(f"\n* Best method (lowest FPR): {best_method}")

    ood_metrics = all_ood_metrics[best_method]
    print(f"\nBest Method ({best_method}) Detailed Metrics:")
    print(f'  AUROC: {ood_metrics["auroc"]:.4f}')
    print(f'  AUPR: {ood_metrics["aupr"]:.4f}')
    print(f'  FPR: {ood_metrics["fpr"]:.4f}')
    print(f'  Threshold: {ood_metrics["threshold"]:.4f}')
    print(f'  Recall: {ood_metrics["recall"]:.4f}')
    print(f'  Precision: {ood_metrics["precision"]:.4f}')
    print(f'  F1-score: {ood_metrics["f1"]:.4f}')
    print(f'  Accuracy: {ood_metrics["accuracy"]:.4f}')
    print("\nConfusion Matrix:")
    print(f'  True Positives (OOD as OOD): {ood_metrics["tp"]}')
    print(f'  False Negatives (OOD as ID): {ood_metrics["fn"]}')
    print(f'  False Positives (ID as OOD): {ood_metrics["fp"]}')
    print(f'  True Negatives (ID as ID): {ood_metrics["tn"]}')

    results_path = os.path.join(args.output_dir, "evaluation_results.txt")
    with open(results_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Classification Metrics (ID Test Set)\n")
        f.write("=" * 80 + "\n")
        f.write(f'Top-1 Accuracy: {id_test_metrics["acc_top1"]:.2f}%\n')
        f.write(f'Top-5 Accuracy: {id_test_metrics["acc_top5"]:.2f}%\n')
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("OOD Detection Method Comparison\n")
        f.write("=" * 80 + "\n")
        f.write(
            f"{'Method':<22} {'AUROC':>8} {'AUPR':>8} {'FPR':>8} {'Recall':>8} {'Prec':>8} {'F1':>8} {'Acc':>8}\n"
        )
        f.write("-" * 86 + "\n")
        for method, metrics in all_ood_metrics.items():
            marker = " *" if method == best_method else ""
            f.write(
                f"{method:<22} {metrics['auroc']:>8.4f} {metrics['aupr']:>8.4f} "
                f"{metrics['fpr']:>8.4f} {metrics['recall']:>8.4f} "
                f"{metrics['precision']:>8.4f} {metrics['f1']:>8.4f} "
                f"{metrics['accuracy']:>8.4f}{marker}\n"
            )
        f.write(f"\n* Best method (lowest FPR): {best_method}\n")
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write(f"Best Method ({best_method}) Detailed Metrics\n")
        f.write("=" * 80 + "\n")
        f.write(f'AUROC: {ood_metrics["auroc"]:.4f}\n')
        f.write(f'AUPR: {ood_metrics["aupr"]:.4f}\n')
        f.write(f'FPR: {ood_metrics["fpr"]:.4f}\n')
        f.write(f'Threshold: {ood_metrics["threshold"]:.4f}\n')
        f.write(f'Recall: {ood_metrics["recall"]:.4f}\n')
        f.write(f'Precision: {ood_metrics["precision"]:.4f}\n')
        f.write(f'F1-score: {ood_metrics["f1"]:.4f}\n')
        f.write(f'Accuracy: {ood_metrics["accuracy"]:.4f}\n')
        f.write("\nConfusion Matrix:\n")
        f.write(f'TP: {ood_metrics["tp"]}, FN: {ood_metrics["fn"]}\n')
        f.write(f'FP: {ood_metrics["fp"]}, TN: {ood_metrics["tn"]}\n')

    print(f"\nResults saved to {results_path}")
    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate audio deepfake source tracing model"
    )

    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default="data/w2vbert_embeddings/layer4",
        help="Path to embeddings directory",
    )
    parser.add_argument(
        "--metadata_dir",
        type=str,
        default="metadata/non_merged",
        help="Directory with split metadata",
    )

    parser.add_argument("--input_dim", type=int, default=1024, help="Input dimension")
    parser.add_argument(
        "--hidden_dims",
        type=int,
        nargs="*",
        default=[],
        help="Hidden layer dimensions (omit values for no hidden layers)",
    )
    parser.add_argument(
        "--embedding_dim", type=int, default=1024, help="Output embedding dimension"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.3, help="Dropout probability"
    )

    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/non_merged",
        help="Directory with saved checkpoints",
    )
    parser.add_argument(
        "--checkpoint_file",
        type=str,
        default="best_model.pth",
        help="Checkpoint file to load",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/non_merged/proxy_anchor",
        help="Directory to save evaluation results",
    )

    parser.add_argument(
        "--merge_map",
        type=str,
        default=None,
        help="Path to merge_map.json (for display purposes in confusion analysis)",
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML config file (CLI arguments override config values)",
    )

    preliminary_args, _ = parser.parse_known_args()
    apply_config_to_parser(parser, preliminary_args.config)
    args = parser.parse_args()

    main(args)
