import argparse
import json
import os

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from data_loader import create_data_loaders
from losses import ProxyAnchorLossWithProxies
from model import create_model
from ood_detection import (
    calibrate_threshold,
    compute_energy_score,
    compute_entropy_score,
    compute_mahalanobis_params,
    compute_mahalanobis_score,
    compute_max_proxy_similarity_score,
)
from utils import (
    AverageMeter,
    accuracy,
    apply_config_to_parser,
    create_merged_model_to_label_mapping,
    create_model_to_label_mapping,
    get_all_models,
    get_device,
    load_merge_map,
    load_split_metadata,
    save_checkpoint,
    save_model_to_label_mapping,
    save_split_config,
    save_split_metadata,
    set_seed,
    split_models,
    split_models_by_architecture,
)


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch_idx, (embeddings, labels, _) in enumerate(progress_bar):
        embeddings = embeddings.to(device)
        labels = labels.to(device)

        embeddings_proj, proxies = model(embeddings)
        loss = criterion(embeddings_proj, labels, proxies)
        logits = model.get_logits(embeddings_proj)
        acc1 = accuracy(logits, labels, topk=(1,))[0]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        losses.update(loss.item(), embeddings.size(0))
        top1.update(acc1, embeddings.size(0))
        progress_bar.set_postfix({"loss": f"{losses.avg:.4f}", "acc": f"{top1.avg:.2f}%"})

    return losses.avg, top1.avg


def validate(
    model, val_loader, criterion, device, epoch, compute_energy_scores=False
):
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    energy_scores = []

    with torch.no_grad():
        for embeddings, labels, _ in tqdm(val_loader, desc="Validation"):
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            embeddings_proj, proxies = model(embeddings)
            loss = criterion(embeddings_proj, labels, proxies)
            logits = model.get_logits(embeddings_proj)
            acc1, acc5 = accuracy(logits, labels, topk=(1, 5))

            if compute_energy_scores:
                energy_scores.extend(compute_energy_score(logits).cpu().numpy().tolist())

            losses.update(loss.item(), embeddings.size(0))
            top1.update(acc1, embeddings.size(0))
            top5.update(acc5, embeddings.size(0))

    print(f"Validation - Loss: {losses.avg:.4f}, Acc@1: {top1.avg:.2f}%, Acc@5: {top5.avg:.2f}%")

    if compute_energy_scores:
        return losses.avg, top1.avg, top5.avg, np.array(energy_scores)
    return losses.avg, top1.avg, top5.avg


def main(args):
    set_seed(args.seed)

    device = get_device()
    print(f"Using device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.metadata_dir, exist_ok=True)

    merge_map = None
    if args.merge_map:
        print(f"Loading merge map from {args.merge_map}...")
        merge_map = load_merge_map(args.merge_map)

    if args.use_existing_split:
        print("Loading existing split from metadata...")
        id_models, ood_cal_models, ood_test_models = load_split_metadata(args.metadata_dir)
    else:
        print("Loading and splitting models...")
        all_models = get_all_models(args.embeddings_dir)
        print(f"Total models: {len(all_models)}")

        if merge_map:
            from utils import get_unique_architectures
            unique_archs = get_unique_architectures(all_models, merge_map)
            num_unique = len(unique_archs)
            num_id = num_unique - args.num_ood_cal_models - args.num_ood_test_models
            print(f"Merging reduces {len(all_models)} models to {num_unique} unique architectures")
            print(f"Architecture split: {num_id} ID, {args.num_ood_cal_models} OOD-cal, "
                  f"{args.num_ood_test_models} OOD-test")
            id_models, ood_cal_models, ood_test_models = split_models_by_architecture(
                all_models, merge_map,
                num_id_architectures=num_id,
                num_ood_cal_architectures=args.num_ood_cal_models,
                num_ood_test_architectures=args.num_ood_test_models,
                seed=args.seed,
            )
        else:
            id_models, ood_cal_models, ood_test_models = split_models(
                all_models,
                num_id_models=args.num_id_models,
                num_ood_cal_models=args.num_ood_cal_models,
                num_ood_test_models=args.num_ood_test_models,
                seed=args.seed,
            )

        save_split_metadata(id_models, ood_cal_models, ood_test_models, args.metadata_dir)

    if merge_map:
        model_to_label = create_merged_model_to_label_mapping(id_models, merge_map)
        import shutil
        shutil.copy2(args.merge_map, os.path.join(args.metadata_dir, "merge_map.json"))
    else:
        model_to_label = create_model_to_label_mapping(id_models)
    save_model_to_label_mapping(
        model_to_label, os.path.join(args.metadata_dir, "model_to_label.json")
    )

    num_classes = len(set(model_to_label.values()))
    print(f"Number of training classes: {num_classes}")

    print("Creating data loaders...")

    train_loader, val_loader, id_test_loader, ood_cal_loader, ood_test_loader = (
        create_data_loaders(
            embeddings_dir=args.embeddings_dir,
            id_models=id_models,
            ood_cal_models=ood_cal_models,
            ood_test_models=ood_test_models,
            model_to_label=model_to_label,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            max_samples_per_model=args.max_samples_per_model,
        )
    )

    save_split_config(
        args.metadata_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        max_samples_per_model=args.max_samples_per_model,
    )

    print("Creating model...")
    model = create_model(
        input_dim=args.input_dim,
        hidden_dims=args.hidden_dims,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        num_classes=num_classes,
        temperature=args.temperature,
    )
    model = model.to(device)

    criterion = ProxyAnchorLossWithProxies(margin=args.margin, alpha=args.alpha)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    best_acc = 0.0
    best_loss = float("inf")
    best_acc_epoch = 0
    best_loss_epoch = 0

    print(f"\nStarting training for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f'Learning rate: {optimizer.param_groups[0]["lr"]:.6f}')

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_acc1, val_acc5 = validate(
            model, val_loader, criterion, device, epoch
        )
        scheduler.step()

        if val_acc1 > best_acc:
            best_acc = val_acc1
            best_acc_epoch = epoch
            save_checkpoint(model, optimizer, epoch, best_acc, args.checkpoint_dir,
                            filename="best_model.pth")
            print(f"✓ New best accuracy! Acc@1: {best_acc:.2f}% (epoch {epoch})")

        if val_loss < best_loss:
            best_loss = val_loss
            best_loss_epoch = epoch
            save_checkpoint(model, optimizer, epoch, best_acc, args.checkpoint_dir,
                            filename="best_model_by_loss.pth")
            print(f"✓ New best loss! Loss: {best_loss:.4f} (epoch {epoch})")

        if epoch % args.save_freq == 0:
            save_checkpoint(model, optimizer, epoch, best_acc, args.checkpoint_dir,
                            filename=f"checkpoint_epoch_{epoch}.pth")

    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_acc:.2f}% at epoch {best_acc_epoch}")
    print(f"Best validation loss: {best_loss:.4f} at epoch {best_loss_epoch}")

    print("\nComputing Mahalanobis parameters from training data...")
    model.eval()
    train_embeddings_list = []
    train_labels_list = []

    with torch.no_grad():
        for embeddings, labels, _ in tqdm(train_loader, desc="Extracting training embeddings"):
            embeddings = embeddings.to(device)
            embeddings_proj, _ = model(embeddings)
            train_embeddings_list.append(embeddings_proj.cpu().numpy())
            train_labels_list.append(labels.numpy())

    train_embeddings_np = np.concatenate(train_embeddings_list, axis=0)
    train_labels_np = np.concatenate(train_labels_list, axis=0)

    centroids, precision_matrix = compute_mahalanobis_params(train_embeddings_np, train_labels_np)

    mahalanobis_path = os.path.join(args.checkpoint_dir, "mahalanobis_params.npz")
    np.savez(mahalanobis_path, centroids=centroids, precision_matrix=precision_matrix)
    print(f"Mahalanobis parameters saved to {mahalanobis_path}")

    print("\nCalibrating OOD thresholds on OOD calibration set...")
    ood_cal_energy_scores = []
    ood_cal_entropy_scores = []
    ood_cal_proxy_sim_scores = []
    ood_cal_mahalanobis_scores = []

    centroids_tensor = torch.from_numpy(centroids).float().to(device)
    precision_tensor = torch.from_numpy(precision_matrix).float().to(device)

    with torch.no_grad():
        for embeddings, labels, _ in tqdm(ood_cal_loader, desc="Computing OOD cal scores"):
            embeddings = embeddings.to(device)
            embeddings_proj, proxies = model(embeddings)
            logits = model.get_logits(embeddings_proj)

            ood_cal_energy_scores.extend(compute_energy_score(logits).cpu().numpy().tolist())
            ood_cal_entropy_scores.extend(compute_entropy_score(logits).cpu().numpy().tolist())
            ood_cal_proxy_sim_scores.extend(
                compute_max_proxy_similarity_score(embeddings_proj, proxies).cpu().numpy().tolist()
            )
            ood_cal_mahalanobis_scores.extend(
                compute_mahalanobis_score(embeddings_proj, centroids_tensor, precision_tensor)
                .cpu().numpy().tolist()
            )

    scoring_methods = {
        "energy": np.array(ood_cal_energy_scores),
        "entropy": np.array(ood_cal_entropy_scores),
        "max_proxy_similarity": np.array(ood_cal_proxy_sim_scores),
        "mahalanobis": np.array(ood_cal_mahalanobis_scores),
    }

    thresholds = {}
    for method_name, scores in scoring_methods.items():
        threshold = calibrate_threshold(scores, target_recall=args.ood_target_recall,
                                        calibration_mode="ood")
        thresholds[method_name] = float(threshold)
        print(f"  {method_name} threshold ({args.ood_target_recall*100:.0f}% recall): {threshold:.4f}")

    threshold_path = os.path.join(args.checkpoint_dir, "ood_threshold.txt")
    with open(threshold_path, "w") as f:
        f.write(f"{thresholds['energy']}\n")

    threshold_metadata = {
        "calibration_type": "ood",
        "target_recall": args.ood_target_recall,
        "num_ood_cal_models": len(ood_cal_models),
        "thresholds": thresholds,
        "threshold_value": thresholds["energy"],
    }
    with open(os.path.join(args.checkpoint_dir, "threshold_metadata.json"), "w") as f:
        json.dump(threshold_metadata, f, indent=2)

    print(f"\nThresholds saved to {threshold_path} and threshold_metadata.json")

    print("\nTraining script completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train audio deepfake source tracing model"
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
        help="Directory to save split metadata",
    )

    parser.add_argument(
        "--input_dim",
        type=int,
        default=1024,
        help="Input dimension (wav2vec2-bert layer 4)",
    )
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
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for logit scaling (lower = sharper predictions)",
    )

    parser.add_argument(
        "--margin", type=float, default=0.1, help="Proxy-anchor loss margin"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=32.0,
        help="Proxy-anchor loss alpha scaling factor",
    )

    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--num_workers", type=int, default=5, help="Number of data loading workers"
    )

    parser.add_argument(
        "--num_id_models",
        type=int,
        default=120,
        help="Number of ID models to use",
    )
    parser.add_argument(
        "--num_ood_cal_models",
        type=int,
        default=10,
        help="Number of OOD calibration models",
    )
    parser.add_argument(
        "--num_ood_test_models",
        type=int,
        default=10,
        help="Number of OOD test models",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.70,
        help="Training samples ratio per ID model",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Validation samples ratio per ID model",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Test samples ratio per ID model",
    )
    parser.add_argument(
        "--max_samples_per_model",
        type=int,
        default=2000,
        help="Maximum samples per model (default: 2000, use None for no cap)",
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/non_merged",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--save_freq", type=int, default=10, help="Save checkpoint every N epochs"
    )

    parser.add_argument(
        "--ood_target_recall",
        type=float,
        default=0.95,
        help="Target recall for OOD threshold calibration (default: 0.95)",
    )

    parser.add_argument(
        "--merge_map",
        type=str,
        default=None,
        help="Path to merge_map.json for architecture-level merging (optional)",
    )

    parser.add_argument(
        "--use_existing_split",
        action="store_true",
        help="Use existing split from metadata_dir instead of generating a new one",
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
