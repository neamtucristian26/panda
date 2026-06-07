import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml


def load_config(config_path: str) -> dict:

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    flat = {}
    for section_key, section_val in config.items():
        if isinstance(section_val, dict):
            flat.update(section_val)
        else:
            flat[section_key] = section_val

    return flat


def apply_config_to_parser(parser, config_path: Optional[str] = None):

    if config_path and os.path.exists(config_path):
        config = load_config(config_path)
        existing_dests = {
            action.dest for action in parser._actions if action.dest != "help"
        }
        parser.set_defaults(**{k: v for k, v in config.items() if k in existing_dests})
        return True
    return False


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_acc: float,
    checkpoint_dir: str,
    filename: str = "checkpoint.pth",
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_acc": best_acc,
        },
        checkpoint_path,
    )
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[int, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    best_acc = checkpoint["best_acc"]
    print(
        f"Checkpoint loaded from {checkpoint_path} (epoch {epoch}, best_acc {best_acc:.4f})"
    )
    return epoch, best_acc


def get_all_models(embeddings_dir: str) -> List[str]:

    models = set()
    for lang_dir in Path(embeddings_dir).iterdir():
        if lang_dir.is_dir():
            for model_dir in lang_dir.iterdir():
                if model_dir.is_dir():
                    models.add(model_dir.name)
    return sorted(models)


def split_models(
    all_models: List[str],
    num_id_models: int = 100,
    num_ood_cal_models: int = 20,
    num_ood_test_models: int = 19,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    total_needed = num_id_models + num_ood_cal_models + num_ood_test_models
    assert (
        len(all_models) >= total_needed
    ), f"Need {total_needed} models, have {len(all_models)}"

    random.seed(seed)
    models = all_models.copy()
    random.shuffle(models)

    id_models = models[:num_id_models]
    ood_cal_models = models[num_id_models : num_id_models + num_ood_cal_models]
    ood_test_models = models[num_id_models + num_ood_cal_models : total_needed]

    print(
        f"Model split: {len(id_models)} ID, "
        f"{len(ood_cal_models)} OOD-cal, {len(ood_test_models)} OOD-test"
    )
    return id_models, ood_cal_models, ood_test_models


def save_split_metadata(
    id_models: List[str],
    ood_cal_models: List[str],
    ood_test_models: List[str],
    output_dir: str,
):
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "id_models.txt"), "w") as f:
        f.write("\n".join(id_models))
    with open(os.path.join(output_dir, "ood_cal_models.txt"), "w") as f:
        f.write("\n".join(ood_cal_models))
    with open(os.path.join(output_dir, "ood_test_models.txt"), "w") as f:
        f.write("\n".join(ood_test_models))

    metadata = {
        "split_type": "sample_level_id",
        "num_id_models": len(id_models),
        "num_ood_cal_models": len(ood_cal_models),
        "num_ood_test_models": len(ood_test_models),
    }
    with open(os.path.join(output_dir, "split_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(
        f"Saved split metadata: {len(id_models)} ID, "
        f"{len(ood_cal_models)} OOD-cal, {len(ood_test_models)} OOD-test"
    )


def save_split_config(
    metadata_dir: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    max_samples_per_model: int,
):

    config = {
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "seed": seed,
        "max_samples_per_model": max_samples_per_model,
    }
    config_path = os.path.join(metadata_dir, "split_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Split config saved to {config_path}")


def load_split_config(metadata_dir: str) -> dict:
    config_path = os.path.join(metadata_dir, "split_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Missing split_config.json in {metadata_dir}. Run train.py first."
        )
    with open(config_path, "r") as f:
        config = json.load(f)
    print(
        f"Loaded split config: {config['train_ratio']}/{config['val_ratio']}/{config['test_ratio']} "
        f"(seed={config['seed']}, max_samples={config['max_samples_per_model']})"
    )
    return config


def load_split_metadata(metadata_dir: str) -> Tuple[List[str], List[str], List[str]]:
    def load_models(filename):
        path = os.path.join(metadata_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {filename}. Run train.py first.")
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]

    id_models = load_models("id_models.txt")
    ood_cal_models = load_models("ood_cal_models.txt")
    ood_test_models = load_models("ood_test_models.txt")

    print(
        f"Loaded split: {len(id_models)} ID, "
        f"{len(ood_cal_models)} OOD-cal, {len(ood_test_models)} OOD-test"
    )
    return id_models, ood_cal_models, ood_test_models


def create_model_to_label_mapping(train_models: List[str]) -> Dict[str, int]:
    return {model: idx for idx, model in enumerate(sorted(train_models))}


def save_model_to_label_mapping(model_to_label: Dict[str, int], output_path: str):
    with open(output_path, "w") as f:
        json.dump(model_to_label, f, indent=2)
    print(f"Model-to-label mapping saved to {output_path}")


def load_model_to_label_mapping(mapping_path: str) -> Dict[str, int]:
    with open(mapping_path, "r") as f:
        return json.load(f)


def load_merge_map(path: str) -> Dict[str, str]:

    with open(path, "r") as f:
        merge_map = json.load(f)
    print(
        f"Loaded merge map with {len(merge_map)} entries "
        f"({len(set(merge_map.values()))} unique architectures)"
    )
    return merge_map


def apply_merge_map(model_name: str, merge_map: Dict[str, str]) -> str:
    return merge_map.get(model_name, model_name)


def get_unique_architectures(models: List[str], merge_map: Dict[str, str]) -> List[str]:
    return sorted({apply_merge_map(m, merge_map) for m in models})


def create_merged_model_to_label_mapping(
    id_models: List[str], merge_map: Dict[str, str]
) -> Dict[str, int]:
    architectures = get_unique_architectures(id_models, merge_map)
    arch_to_label = {arch: idx for idx, arch in enumerate(architectures)}

    model_to_label = {
        model: arch_to_label[apply_merge_map(model, merge_map)] for model in id_models
    }

    print(
        f"Merged {len(id_models)} models into {len(architectures)} architecture classes"
    )
    return model_to_label


def split_models_by_architecture(
    all_models: List[str],
    merge_map: Dict[str, str],
    num_id_architectures: int,
    num_ood_cal_architectures: int,
    num_ood_test_architectures: int,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:

    arch_to_models: Dict[str, List[str]] = {}
    for m in all_models:
        arch = apply_merge_map(m, merge_map)
        arch_to_models.setdefault(arch, []).append(m)

    unique_architectures = sorted(arch_to_models.keys())
    total_needed = (
        num_id_architectures + num_ood_cal_architectures + num_ood_test_architectures
    )
    assert (
        len(unique_architectures) >= total_needed
    ), f"Need {total_needed} unique architectures, have {len(unique_architectures)}."

    random.seed(seed)
    shuffled_archs = unique_architectures.copy()
    random.shuffle(shuffled_archs)

    id_archs = shuffled_archs[:num_id_architectures]
    ood_cal_archs = shuffled_archs[
        num_id_architectures : num_id_architectures + num_ood_cal_architectures
    ]
    ood_test_archs = shuffled_archs[
        num_id_architectures + num_ood_cal_architectures : total_needed
    ]

    id_models = [m for arch in id_archs for m in arch_to_models[arch]]
    ood_cal_models = [m for arch in ood_cal_archs for m in arch_to_models[arch]]
    ood_test_models = [m for arch in ood_test_archs for m in arch_to_models[arch]]

    print(
        f"Architecture-level split: {len(id_archs)} ID archs ({len(id_models)} models), "
        f"{len(ood_cal_archs)} OOD-cal archs ({len(ood_cal_models)} models), "
        f"{len(ood_test_archs)} OOD-test archs ({len(ood_test_models)} models)"
    )
    return id_models, ood_cal_models, ood_test_models


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res
