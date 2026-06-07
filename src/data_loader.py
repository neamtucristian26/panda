import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


class AudioEmbeddingDataset(Dataset):
    def __init__(
        self,
        embeddings_dir: str,
        models: List[str],
        model_to_label: Dict[str, int],
        is_training: bool = True,
        sample_indices: Optional[List[int]] = None,
        max_samples_per_model: Optional[int] = None,
    ):
        self.embeddings_dir = Path(embeddings_dir)
        self.models = set(models)
        self.model_to_label = model_to_label
        self.is_training = is_training
        self.max_samples_per_model = max_samples_per_model

        self.samples = []
        self._collect_samples()

        if sample_indices is not None:
            self.samples = [self.samples[i] for i in sample_indices]

        print(f"Loaded {len(self.samples)} samples from {len(self.models)} models")

    def _collect_samples(self):
        model_samples_dict = {}

        for lang_dir in sorted(self.embeddings_dir.iterdir()):
            if not lang_dir.is_dir():
                continue
            for model_dir in sorted(lang_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                model_name = model_dir.name
                if model_name not in self.models:
                    continue

                label = self.model_to_label.get(model_name, -1)

                if model_name not in model_samples_dict:
                    model_samples_dict[model_name] = []

                for emb_file in sorted(model_dir.glob("*.pt")):
                    model_samples_dict[model_name].append(
                        {
                            "path": str(emb_file),
                            "label": label,
                            "model_name": model_name,
                        }
                    )

        capped_models = []
        for model_name in sorted(model_samples_dict.keys()):
            samples = model_samples_dict[model_name]
            original_count = len(samples)

            if (
                self.max_samples_per_model
                and original_count > self.max_samples_per_model
            ):
                samples = random.sample(samples, self.max_samples_per_model)
                capped_models.append(
                    (model_name, original_count, self.max_samples_per_model)
                )

            self.samples.extend(samples)

        if capped_models:
            print(
                f"\nCapped {len(capped_models)} models to {self.max_samples_per_model} samples:"
            )
            for model_name, original, capped in sorted(
                capped_models, key=lambda x: x[1], reverse=True
            )[:10]:
                print(
                    f"  {model_name}: {original} → {capped} ({100*(capped/original):.1f}% retained)"
                )
            if len(capped_models) > 10:
                print(f"  ... and {len(capped_models) - 10} more models")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        embedding = torch.load(sample["path"])

        if embedding.dim() == 1:
            pooled = embedding
        elif embedding.dim() == 2:
            if embedding.shape[0] == 1:
                pooled = embedding.squeeze(0)
            else:
                pooled = embedding.mean(dim=0)
        else:
            raise ValueError(f"Unexpected embedding shape: {embedding.shape}")

        assert (
            pooled.shape[0] == 1024
        ), f"Expected 1024-dim embedding, got {pooled.shape}"

        return pooled, sample["label"], sample["model_name"]


def split_samples_per_model(
    embeddings_dir: str,
    models: List[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    max_samples_per_model: Optional[int] = None,
) -> Tuple[List[int], List[int], List[int]]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    embeddings_path = Path(embeddings_dir)
    random.seed(seed)

    model_samples: Dict[str, List[int]] = {}
    global_index = 0

    for lang_dir in sorted(embeddings_path.iterdir()):
        if not lang_dir.is_dir():
            continue
        for model_dir in sorted(lang_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name
            if model_name not in models:
                continue
            if model_name not in model_samples:
                model_samples[model_name] = []
            for _ in sorted(model_dir.glob("*.pt")):
                model_samples[model_name].append(global_index)
                global_index += 1

    capped_models = []
    if max_samples_per_model is not None:
        for model_name in model_samples:
            original_count = len(model_samples[model_name])
            if original_count > max_samples_per_model:
                model_samples[model_name] = random.sample(
                    model_samples[model_name], max_samples_per_model
                )
                capped_models.append(
                    (model_name, original_count, max_samples_per_model)
                )

    if capped_models:
        print(
            f"\nCapped {len(capped_models)} models to {max_samples_per_model} samples:"
        )
        for model_name, original, capped in sorted(
            capped_models, key=lambda x: x[1], reverse=True
        )[:10]:
            print(
                f"  {model_name}: {original} → {capped} ({100*(capped/original):.1f}% retained)"
            )
        if len(capped_models) > 10:
            print(f"  ... and {len(capped_models) - 10} more models")

    train_indices, val_indices, test_indices = [], [], []

    for model_name in sorted(model_samples.keys()):
        samples = model_samples[model_name]
        n = len(samples)
        shuffled = samples.copy()
        random.shuffle(shuffled)

        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_indices.extend(shuffled[:n_train])
        val_indices.extend(shuffled[n_train : n_train + n_val])
        test_indices.extend(shuffled[n_train + n_val :])

    print(
        f"Split {len(train_indices)} train, {len(val_indices)} val, "
        f"{len(test_indices)} test samples across {len(model_samples)} models"
    )
    return train_indices, val_indices, test_indices


def create_data_loaders(
    embeddings_dir: str,
    id_models: List[str],
    ood_cal_models: List[str],
    ood_test_models: List[str],
    model_to_label: Dict[str, int],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    batch_size: int = 256,
    num_workers: int = 4,
    seed: int = 42,
    max_samples_per_model: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, DataLoader]:
    """Create 5 data loaders: ID train/val/test + OOD cal/test."""
    train_indices, val_indices, test_indices = split_samples_per_model(
        embeddings_dir=embeddings_dir,
        models=id_models,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        max_samples_per_model=max_samples_per_model,
    )

    id_train_dataset = AudioEmbeddingDataset(
        embeddings_dir,
        id_models,
        model_to_label,
        is_training=True,
        sample_indices=train_indices,
        max_samples_per_model=None,
    )
    id_val_dataset = AudioEmbeddingDataset(
        embeddings_dir,
        id_models,
        model_to_label,
        is_training=False,
        sample_indices=val_indices,
        max_samples_per_model=None,
    )
    id_test_dataset = AudioEmbeddingDataset(
        embeddings_dir,
        id_models,
        model_to_label,
        is_training=False,
        sample_indices=test_indices,
        max_samples_per_model=None,
    )
    ood_cal_dataset = AudioEmbeddingDataset(
        embeddings_dir,
        ood_cal_models,
        model_to_label,
        is_training=False,
        max_samples_per_model=max_samples_per_model,
    )
    ood_test_dataset = AudioEmbeddingDataset(
        embeddings_dir,
        ood_test_models,
        model_to_label,
        is_training=False,
        max_samples_per_model=max_samples_per_model,
    )

    loader_kwargs = dict(num_workers=num_workers, pin_memory=True)

    id_train_loader = DataLoader(
        id_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    id_val_loader = DataLoader(
        id_val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs
    )
    id_test_loader = DataLoader(
        id_test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs
    )
    ood_cal_loader = DataLoader(
        ood_cal_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs
    )
    ood_test_loader = DataLoader(
        ood_test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs
    )

    print("Created 5 data loaders:")
    print(f"  ID train:  {len(id_train_dataset)} samples")
    print(f"  ID val:    {len(id_val_dataset)} samples")
    print(f"  ID test:   {len(id_test_dataset)} samples")
    print(f"  OOD cal:   {len(ood_cal_dataset)} samples")
    print(f"  OOD test:  {len(ood_test_dataset)} samples")

    return (
        id_train_loader,
        id_val_loader,
        id_test_loader,
        ood_cal_loader,
        ood_test_loader,
    )


def collate_fn(batch):
    embeddings, labels, model_names = zip(*batch)
    return (
        torch.stack(embeddings),
        torch.tensor(labels, dtype=torch.long),
        list(model_names),
    )
