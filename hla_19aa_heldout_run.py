"""
HLA-only 19-amino-acid held-out binding experiment.

This run uses hla_only.txt, selects one HLA allele plus one held-out
standard amino acid, trains linear binding predictors on peptides that do
not contain the held-out amino acid, and evaluates the five-fold ensemble on
peptides from the same allele that do contain it.

The three compared encoders are:
  - AE residue centroids from peptide_autoencoder_v3_512.pth
  - 20-symbol one-hot encoding
  - BLOSUM62 row encoding

Run split preparation only:
    python hla_19aa_heldout_run.py --prepare-only

Run full training:
    python hla_19aa_heldout_run.py
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Optional


PROJECT_DIR = Path(".")
DEFAULT_HLA_FILE = PROJECT_DIR / "hla_only.txt"
DEFAULT_AE_WEIGHTS = PROJECT_DIR / "peptide_autoencoder_v3_512.pth"
DEFAULT_IMAGE_DIR = PROJECT_DIR / "peptide_rotamers"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "hla_19aa_heldout_results"

STANDARD_AA = tuple("ACDEFGHIKLMNPQRSTVWY")
STANDARD_AA_SET = set(STANDARD_AA)
AA_TO_INDEX_20 = {aa: idx for idx, aa in enumerate(STANDARD_AA)}

RANDOM_SEED = 7
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
FOLDS = 5


@dataclass(frozen=True)
class Record:
    peptide: str
    label: int
    allele: str
    fold: Optional[int] = None


def safe_token(value: str) -> str:
    return value.replace(":", "_").replace("/", "_").replace("\\", "_")


def read_hla_only(path: Path) -> tuple[dict[str, list[Record]], Counter]:
    grouped: dict[str, dict[str, int]] = defaultdict(dict)
    stats = Counter()

    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 3:
                stats["malformed"] += 1
                continue

            peptide, label_text, allele = parts[0].strip(), parts[1].strip(), parts[2].strip()
            try:
                label = int(label_text)
            except ValueError:
                stats["bad_label"] += 1
                continue

            if label not in {0, 1}:
                stats["bad_label"] += 1
                continue
            if len(peptide) != 9:
                stats["not_9mer"] += 1
                continue
            if any(aa not in STANDARD_AA_SET for aa in peptide):
                stats["non_standard_residue"] += 1
                continue

            previous = grouped[allele].get(peptide)
            if previous is not None and previous != label:
                stats["label_conflict_collapsed_to_positive"] += 1
            grouped[allele][peptide] = max(previous, label) if previous is not None else label
            stats["valid_rows"] += 1

    records_by_allele = {
        allele: [Record(peptide=peptide, label=label, allele=allele) for peptide, label in sorted(peptides.items())]
        for allele, peptides in sorted(grouped.items())
    }
    stats["alleles"] = len(records_by_allele)
    stats["unique_allele_peptides"] = sum(len(rows) for rows in records_by_allele.values())
    return records_by_allele, stats


def label_counts(records: Iterable[Record]) -> Counter:
    return Counter(row.label for row in records)


def split_candidate_row(
    allele: str,
    held_out_aa: str,
    train_records: list[Record],
    eval_records: list[Record],
    min_train_rows: int,
    min_eval_rows: int,
    min_train_per_class: int,
    min_eval_per_class: int,
) -> dict[str, object]:
    train_counts = label_counts(train_records)
    eval_counts = label_counts(eval_records)
    valid = (
        len(train_records) >= min_train_rows
        and len(eval_records) >= min_eval_rows
        and train_counts.get(0, 0) >= min_train_per_class
        and train_counts.get(1, 0) >= min_train_per_class
        and eval_counts.get(0, 0) >= min_eval_per_class
        and eval_counts.get(1, 0) >= min_eval_per_class
    )
    return {
        "Allele": allele,
        "Held_Out_AA": held_out_aa,
        "Allele_Total": len(train_records) + len(eval_records),
        "Train_Rows": len(train_records),
        "Train_Positive": train_counts.get(1, 0),
        "Train_Negative": train_counts.get(0, 0),
        "Eval_Rows": len(eval_records),
        "Eval_Positive": eval_counts.get(1, 0),
        "Eval_Negative": eval_counts.get(0, 0),
        "Valid": valid,
    }


def compute_split_candidates(
    records_by_allele: dict[str, list[Record]],
    min_train_rows: int,
    min_eval_rows: int,
    min_train_per_class: int,
    min_eval_per_class: int,
) -> list[dict[str, object]]:
    candidates = []
    for allele, records in records_by_allele.items():
        for held_out_aa in STANDARD_AA:
            train_records = [row for row in records if held_out_aa not in row.peptide]
            eval_records = [row for row in records if held_out_aa in row.peptide]
            candidates.append(
                split_candidate_row(
                    allele,
                    held_out_aa,
                    train_records,
                    eval_records,
                    min_train_rows,
                    min_eval_rows,
                    min_train_per_class,
                    min_eval_per_class,
                )
            )
    return candidates


def candidate_sort_key(row: dict[str, object]) -> tuple[object, ...]:
    return (
        int(row["Valid"]),
        int(row["Allele_Total"]),
        int(row["Train_Rows"]),
        int(row["Eval_Rows"]),
        int(row["Train_Positive"]),
        int(row["Eval_Positive"]),
        str(row["Allele"]),
        str(row["Held_Out_AA"]),
    )


def choose_candidate(
    candidates: list[dict[str, object]],
    allele: Optional[str],
    held_out_aa: Optional[str],
) -> dict[str, object]:
    filtered = candidates
    if allele:
        filtered = [row for row in filtered if row["Allele"] == allele]
    if held_out_aa:
        filtered = [row for row in filtered if row["Held_Out_AA"] == held_out_aa]

    valid = [row for row in filtered if row["Valid"]]
    if not valid:
        filters = []
        if allele:
            filters.append(f"allele={allele}")
        if held_out_aa:
            filters.append(f"held_out_aa={held_out_aa}")
        suffix = f" for {' and '.join(filters)}" if filters else ""
        raise ValueError(f"No valid held-out split found{suffix}. Lower thresholds or inspect the candidate CSV.")

    return max(valid, key=candidate_sort_key)


def build_selected_split(
    records_by_allele: dict[str, list[Record]],
    selected: dict[str, object],
    folds: int,
    seed: int,
) -> tuple[list[Record], list[Record]]:
    allele = str(selected["Allele"])
    held_out_aa = str(selected["Held_Out_AA"])
    allele_records = records_by_allele[allele]
    train_records = [row for row in allele_records if held_out_aa not in row.peptide]
    eval_records = [row for row in allele_records if held_out_aa in row.peptide]
    cv_records = assign_stratified_folds(train_records, folds=folds, seed=seed)

    assert all(row.allele == allele for row in cv_records)
    assert all(row.allele == allele for row in eval_records)
    assert all(held_out_aa not in row.peptide for row in cv_records)
    assert all(held_out_aa in row.peptide for row in eval_records)
    assert not ({row.peptide for row in cv_records} & {row.peptide for row in eval_records})
    return cv_records, eval_records


def assign_stratified_folds(records: list[Record], folds: int, seed: int) -> list[Record]:
    by_label: dict[int, list[Record]] = defaultdict(list)
    for row in sorted(records, key=lambda item: item.peptide):
        by_label[row.label].append(row)

    for label in (0, 1):
        if len(by_label[label]) < folds:
            raise ValueError(f"Need at least {folds} rows for class {label} in the train/test split.")

    rng = random.Random(seed)
    folded = []
    for label, label_records in sorted(by_label.items()):
        shuffled = list(label_records)
        rng.shuffle(shuffled)
        for idx, row in enumerate(shuffled):
            folded.append(replace(row, fold=idx % folds))

    return sorted(folded, key=lambda item: (item.fold if item.fold is not None else -1, item.peptide))


def write_split_candidates(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "Allele",
        "Held_Out_AA",
        "Allele_Total",
        "Train_Rows",
        "Train_Positive",
        "Train_Negative",
        "Eval_Rows",
        "Eval_Positive",
        "Eval_Negative",
        "Valid",
    ]
    sorted_rows = sorted(rows, key=candidate_sort_key, reverse=True)
    save_rows_csv(sorted_rows, path, fieldnames)


def write_records_csv(records: list[Record], path: Path, include_fold: bool) -> None:
    fieldnames = ["Peptide", "Label", "Allele"]
    if include_fold:
        fieldnames.append("Fold")
    rows = []
    for row in records:
        out = {"Peptide": row.peptide, "Label": row.label, "Allele": row.allele}
        if include_fold:
            out["Fold"] = row.fold
        rows.append(out)
    save_rows_csv(rows, path, fieldnames)


def write_labeled_txt(records: list[Record], path: Path, include_fold: bool) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        for row in records:
            if include_fold:
                handle.write(f"{row.peptide} {row.label} {row.allele} {row.fold}\n")
            else:
                handle.write(f"{row.peptide} {row.label} {row.allele}\n")


def save_rows_csv(rows: list[dict[str, object]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def one_hot_20_featurizer(sequence: str):
    import numpy as np

    matrix = np.zeros((len(sequence), len(STANDARD_AA)), dtype=np.float32)
    for position, aa in enumerate(sequence):
        matrix[position, AA_TO_INDEX_20[aa]] = 1.0
    return matrix.reshape(-1)


BLOSUM62_ORDER = tuple("ARNDCQEGHILKMFPSTWYV")
BLOSUM62_VALUES = [
    [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],
    [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],
    [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],
    [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],
    [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
    [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],
    [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],
    [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],
    [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],
    [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],
    [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],
    [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],
    [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],
    [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],
    [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],
    [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],
    [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],
    [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],
    [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],
    [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],
]
BLOSUM62_INDEX = {aa: idx for idx, aa in enumerate(BLOSUM62_ORDER)}


def blosum62_20_featurizer(sequence: str):
    import numpy as np

    rows = [BLOSUM62_VALUES[BLOSUM62_INDEX[aa]] for aa in sequence]
    return np.array(rows, dtype=np.float32).reshape(-1)


AA_TO_IMAGE_CLASS = {
    "alanine": "A",
    "arginine": "R",
    "asparagine": "N",
    "aspartic_acid": "D",
    "cysteine": "C",
    "glutamic_acid": "E",
    "glutamine": "Q",
    "glycine": "G",
    "histidine": "H",
    "isoleucine": "I",
    "leucine": "L",
    "lysine": "K",
    "methionine": "M",
    "phenylalanine": "F",
    "proline": "P",
    "serine": "S",
    "threonine": "T",
    "tryptophan": "W",
    "tyrosine": "Y",
    "valine": "V",
}


def load_autoencoder_centroids(weights_path: Path, image_dir: Path, device):
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing AE weights: {weights_path}")
    if not image_dir.exists():
        raise FileNotFoundError(
            f"Missing AE image directory: {image_dir}. "
            "The AE encoder needs the same residue rotamer image folders used by SIHA_ex.ipynb."
        )

    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    class SEBlock(nn.Module):
        def __init__(self, channels: int, reduction: int = 16):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(channels, channels // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channels // reduction, channels),
                nn.Sigmoid(),
            )

        def forward(self, x):
            batch, channels, _, _ = x.size()
            y = self.pool(x).view(batch, channels)
            y = self.fc(y).view(batch, channels, 1, 1)
            return x * y

    class AutoEncoderV3(nn.Module):
        def __init__(self, bottleneck_size: int = 512):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.1),
                nn.MaxPool2d(2),
                SEBlock(32),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.1),
                nn.MaxPool2d(2),
                SEBlock(64),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.1),
                nn.MaxPool2d(2),
                SEBlock(128),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1),
                nn.MaxPool2d(2),
                SEBlock(256),
                nn.Flatten(),
                nn.Linear(256 * 15 * 15, bottleneck_size),
            )

        def forward(self, x):
            return self.encoder(x)

    print(f"Loading AE weights: {weights_path}")
    model = AutoEncoderV3(bottleneck_size=512).to(device)
    try:
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    dataset = datasets.ImageFolder(str(image_dir), transform=transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    latent_vectors = []
    labels = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            if device.type == "cuda":
                with torch.autocast(device_type="cuda"):
                    vectors = model.encoder(inputs)
            else:
                vectors = model.encoder(inputs)
            latent_vectors.append(vectors.cpu().numpy())
            labels.extend(targets.numpy())

    latent_matrix = np.concatenate(latent_vectors, axis=0)
    labels = np.array(labels)
    centroids = {}
    for idx, class_name in enumerate(dataset.classes):
        letter = AA_TO_IMAGE_CLASS.get(class_name)
        if letter is None:
            continue
        centroids[letter] = latent_matrix[labels == idx].mean(axis=0)

    missing = sorted(STANDARD_AA_SET - set(centroids))
    if missing:
        raise ValueError(f"Missing AE centroid(s) for standard residues: {missing}")
    print(f"Loaded AE centroids for {len(centroids)} standard residues.")
    return centroids


def make_autoencoder_featurizer(centroid_dict: dict[str, object]) -> Callable[[str], object]:
    def featurize(sequence: str):
        import numpy as np

        return np.concatenate([centroid_dict[aa] for aa in sequence]).astype(np.float32)

    return featurize


def records_to_tensors(records: list[Record], featurize: Callable[[str], object]):
    import numpy as np
    import torch

    features = [featurize(row.peptide) for row in records]
    labels = [row.label for row in records]
    x = torch.tensor(np.array(features), dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    return x, y


def run_fivefold_cv(
    records: list[Record],
    featurize: Callable[[str], object],
    experiment: dict[str, object],
    device,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> list[object]:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    class TorchLinearNN(nn.Module):
        def __init__(self, input_size: int):
            super().__init__()
            self.fc = nn.Linear(input_size, 1)
            # Important for one-hot held-out residues: columns absent during
            # training remain neutral instead of retaining random weights.
            nn.init.zeros_(self.fc.weight)
            nn.init.zeros_(self.fc.bias)

        def forward(self, x):
            return self.fc(x)

    all_results = []
    trained_models = []
    for fold_idx in range(int(experiment["folds"])):
        train_records = [row for row in records if row.fold != fold_idx]
        test_records = [row for row in records if row.fold == fold_idx]
        print(
            f"--- {experiment['id']} fold {fold_idx}: "
            f"train={len(train_records)} test={len(test_records)} ---"
        )

        x_train, y_train = records_to_tensors(train_records, featurize)
        x_test, y_test = records_to_tensors(test_records, featurize)

        generator = torch.Generator()
        generator.manual_seed(seed + fold_idx)
        train_loader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )

        model = TorchLinearNN(input_size=x_train.shape[1]).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        model.train()
        for _ in range(epochs):
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()

        model_path = Path(experiment["output_dir"]) / f"{experiment['model_prefix']}_fold_{fold_idx}.pth"
        torch.save(model.state_dict(), model_path)
        trained_models.append(model)

        model.eval()
        probabilities = []
        with torch.no_grad():
            for start in range(0, len(x_test), batch_size):
                batch_x = x_test[start : start + batch_size].to(device)
                probabilities.extend(torch.sigmoid(model(batch_x)).cpu().numpy().flatten().tolist())

        for row, true_value, probability in zip(test_records, y_test.numpy().flatten(), probabilities):
            all_results.append(
                {
                    "Experiment": experiment["id"],
                    "Encoder": experiment["encoder_id"],
                    "Allele": experiment["allele"],
                    "Held_Out_AA": experiment["held_out_aa"],
                    "Fold": fold_idx,
                    "Peptide": row.peptide,
                    "True_Class": int(true_value),
                    "Probability": float(probability),
                }
            )

    fieldnames = [
        "Experiment",
        "Encoder",
        "Allele",
        "Held_Out_AA",
        "Fold",
        "Peptide",
        "True_Class",
        "Probability",
    ]
    save_rows_csv(all_results, Path(experiment["cv_csv"]), fieldnames)
    print(f"Saved CV predictions: {experiment['cv_csv']} | rows={len(all_results)}")
    return trained_models


def predict_heldout_evaluation(
    records: list[Record],
    featurize: Callable[[str], object],
    models: list[object],
    experiment: dict[str, object],
    device,
    batch_size: int,
) -> list[dict[str, object]]:
    import numpy as np
    import torch

    x, y = records_to_tensors(records, featurize)
    predictions_matrix = np.zeros((len(x), len(models)), dtype=np.float32)
    x = x.to(device)

    for model_idx, model in enumerate(models):
        model.eval()
        fold_probabilities = []
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                batch_x = x[start : start + batch_size]
                fold_probabilities.extend(torch.sigmoid(model(batch_x)).cpu().numpy().flatten().tolist())
        predictions_matrix[:, model_idx] = fold_probabilities

    ensemble = predictions_matrix.mean(axis=1)
    rows = []
    for idx, (row, true_value, mean_probability) in enumerate(zip(records, y.numpy().flatten(), ensemble)):
        out = {
            "Experiment": experiment["id"],
            "Encoder": experiment["encoder_id"],
            "Allele": experiment["allele"],
            "Held_Out_AA": experiment["held_out_aa"],
            "Index": idx,
            "Peptide": row.peptide,
            "True_Class": int(true_value),
            "Ensemble_Probability": float(mean_probability),
        }
        for model_idx in range(len(models)):
            out[f"Fold_{model_idx}_Probability"] = float(predictions_matrix[idx, model_idx])
        rows.append(out)

    fieldnames = [
        "Experiment",
        "Encoder",
        "Allele",
        "Held_Out_AA",
        "Index",
        "Peptide",
        "True_Class",
        "Ensemble_Probability",
    ] + [f"Fold_{idx}_Probability" for idx in range(len(models))]
    save_rows_csv(rows, Path(experiment["eval_csv"]), fieldnames)
    print(f"Saved held-out evaluation predictions: {experiment['eval_csv']} | rows={len(rows)}")
    return rows


def rank_auc(y_true: list[int], y_score: list[float]) -> float:
    positives = sum(1 for value in y_true if value == 1)
    negatives = sum(1 for value in y_true if value == 0)
    if positives == 0 or negatives == 0:
        return math.nan

    pairs = sorted(zip(y_score, y_true), key=lambda item: item[0])
    rank_sum_positive = 0.0
    idx = 0
    while idx < len(pairs):
        end = idx + 1
        while end < len(pairs) and pairs[end][0] == pairs[idx][0]:
            end += 1
        average_rank = (idx + 1 + end) / 2.0
        rank_sum_positive += average_rank * sum(1 for _, label in pairs[idx:end] if label == 1)
        idx = end

    return (rank_sum_positive - positives * (positives + 1) / 2.0) / (positives * negatives)


def threshold_metrics(y_true: list[int], y_score: list[float], threshold: float = 0.5) -> dict[str, float]:
    predicted = [1 if value >= threshold else 0 for value in y_score]
    tp = sum(1 for truth, pred in zip(y_true, predicted) if truth == 1 and pred == 1)
    tn = sum(1 for truth, pred in zip(y_true, predicted) if truth == 0 and pred == 0)
    fp = sum(1 for truth, pred in zip(y_true, predicted) if truth == 0 and pred == 1)
    fn = sum(1 for truth, pred in zip(y_true, predicted) if truth == 1 and pred == 0)
    total = len(y_true)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "Accuracy": (tp + tn) / total if total else math.nan,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Predicted_Positive": tp + fp,
    }


def summarize_cv_rows(rows: list[dict[str, object]], experiment: dict[str, object]) -> dict[str, object]:
    y_true = [int(row["True_Class"]) for row in rows]
    y_score = [float(row["Probability"]) for row in rows]
    fold_aucs = []
    for fold_idx in sorted({int(row["Fold"]) for row in rows}):
        fold_rows = [row for row in rows if int(row["Fold"]) == fold_idx]
        fold_aucs.append(
            rank_auc(
                [int(row["True_Class"]) for row in fold_rows],
                [float(row["Probability"]) for row in fold_rows],
            )
        )
    valid_aucs = [value for value in fold_aucs if not math.isnan(value)]
    metrics = threshold_metrics(y_true, y_score)
    return {
        "Experiment": experiment["id"],
        "Encoder": experiment["encoder_id"],
        "Allele": experiment["allele"],
        "Held_Out_AA": experiment["held_out_aa"],
        "Rows": len(rows),
        "Positives": sum(y_true),
        "Negatives": len(y_true) - sum(y_true),
        "Mean_AUC": statistics.mean(valid_aucs) if valid_aucs else math.nan,
        "Std_AUC": statistics.pstdev(valid_aucs) if len(valid_aucs) > 1 else 0.0,
        **metrics,
    }


def summarize_eval_rows(rows: list[dict[str, object]], experiment: dict[str, object]) -> dict[str, object]:
    y_true = [int(row["True_Class"]) for row in rows]
    y_score = [float(row["Ensemble_Probability"]) for row in rows]
    metrics = threshold_metrics(y_true, y_score)
    return {
        "Experiment": experiment["id"],
        "Encoder": experiment["encoder_id"],
        "Allele": experiment["allele"],
        "Held_Out_AA": experiment["held_out_aa"],
        "Rows": len(rows),
        "Positives": sum(y_true),
        "Negatives": len(y_true) - sum(y_true),
        "AUC": rank_auc(y_true, y_score),
        "Mean_Probability": statistics.mean(y_score) if y_score else math.nan,
        "Median_Probability": statistics.median(y_score) if y_score else math.nan,
        **metrics,
    }


def build_encoder_specs(args, device) -> list[dict[str, object]]:
    specs = []
    requested = set(args.encoders)

    if "ae" in requested:
        centroids = load_autoencoder_centroids(args.ae_weights, args.image_dir, device)
        specs.append(
            {
                "id": "ae",
                "label": "AE",
                "expected_dim": 9 * 512,
                "featurizer": make_autoencoder_featurizer(centroids),
            }
        )

    if "onehot20" in requested:
        specs.append(
            {
                "id": "onehot20",
                "label": "One-hot 20",
                "expected_dim": 9 * len(STANDARD_AA),
                "featurizer": one_hot_20_featurizer,
            }
        )

    if "blosum62_20" in requested:
        specs.append(
            {
                "id": "blosum62_20",
                "label": "BLOSUM62 20",
                "expected_dim": 9 * len(BLOSUM62_ORDER),
                "featurizer": blosum62_20_featurizer,
            }
        )

    return specs


def make_experiment(
    encoder_spec: dict[str, object],
    selected: dict[str, object],
    output_dir: Path,
    folds: int,
) -> dict[str, object]:
    allele = str(selected["Allele"])
    held_out_aa = str(selected["Held_Out_AA"])
    encoder_id = str(encoder_spec["id"])
    split_token = f"{safe_token(allele)}_holdout_{held_out_aa}"
    experiment_id = f"{split_token}__{encoder_id}"
    return {
        "id": experiment_id,
        "encoder_id": encoder_id,
        "encoder_label": encoder_spec["label"],
        "allele": allele,
        "held_out_aa": held_out_aa,
        "folds": folds,
        "output_dir": output_dir,
        "model_prefix": f"hla_19aa_{split_token}_{encoder_id}",
        "cv_csv": output_dir / f"HLA_19AA_CV_Binding_Predictions_{experiment_id}.csv",
        "eval_csv": output_dir / f"HLA_19AA_Heldout_Evaluation_{experiment_id}.csv",
        "featurizer": encoder_spec["featurizer"],
        "expected_dim": encoder_spec["expected_dim"],
    }


def assert_feature_dimensions(experiments: list[dict[str, object]], sample_record: Record) -> None:
    for experiment in experiments:
        observed = len(experiment["featurizer"](sample_record.peptide))
        expected = int(experiment["expected_dim"])
        if observed != expected:
            raise AssertionError(f"{experiment['id']} dim mismatch: expected {expected}, observed {observed}")
    print(f"Feature dimension checks passed for {len(experiments)} encoders.")


def load_prediction_rows(path: Path) -> list[dict[str, object]]:
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_fold_distribution(records: list[Record]) -> list[dict[str, object]]:
    rows = []
    for fold_idx in sorted({row.fold for row in records}):
        fold_records = [row for row in records if row.fold == fold_idx]
        counts = label_counts(fold_records)
        rows.append(
            {
                "Fold": fold_idx,
                "Rows": len(fold_records),
                "Positive": counts.get(1, 0),
                "Negative": counts.get(0, 0),
            }
        )
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Run HLA 19-AA held-out amino-acid binding comparison.")
    parser.add_argument("--hla-file", type=Path, default=DEFAULT_HLA_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ae-weights", type=Path, default=DEFAULT_AE_WEIGHTS)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--allele", default=None, help="Optional exact HLA allele to use, e.g. HLA-A29:02.")
    parser.add_argument("--held-out-aa", default=None, help="Optional held-out standard amino acid, e.g. W.")
    parser.add_argument("--min-train-rows", type=int, default=1000)
    parser.add_argument("--min-eval-rows", type=int, default=100)
    parser.add_argument("--min-train-per-class", type=int, default=25)
    parser.add_argument("--min-eval-per-class", type=int, default=10)
    parser.add_argument("--folds", type=int, default=FOLDS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--encoders",
        nargs="+",
        default=["ae", "onehot20", "blosum62_20"],
        choices=["ae", "onehot20", "blosum62_20"],
    )
    parser.add_argument("--prepare-only", action="store_true", help="Build and write the split files without training.")
    args = parser.parse_args()
    if args.held_out_aa:
        args.held_out_aa = args.held_out_aa.upper()
        if args.held_out_aa not in STANDARD_AA_SET:
            raise ValueError(f"--held-out-aa must be one of {''.join(STANDARD_AA)}")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records_by_allele, load_stats = read_hla_only(args.hla_file)
    print(f"Loaded {load_stats['valid_rows']} valid rows from {args.hla_file}")
    print(f"Collapsed to {load_stats['unique_allele_peptides']} unique allele-peptide records across {load_stats['alleles']} alleles")
    if load_stats["label_conflict_collapsed_to_positive"]:
        print(f"Collapsed label conflicts to positive: {load_stats['label_conflict_collapsed_to_positive']}")

    candidates = compute_split_candidates(
        records_by_allele,
        min_train_rows=args.min_train_rows,
        min_eval_rows=args.min_eval_rows,
        min_train_per_class=args.min_train_per_class,
        min_eval_per_class=args.min_eval_per_class,
    )
    candidate_csv = args.output_dir / "HLA_19AA_Split_Candidates.csv"
    write_split_candidates(candidates, candidate_csv)
    print(f"Saved split candidate report: {candidate_csv}")

    selected = choose_candidate(candidates, allele=args.allele, held_out_aa=args.held_out_aa)
    cv_records, eval_records = build_selected_split(records_by_allele, selected, folds=args.folds, seed=args.seed)

    split_token = f"{safe_token(str(selected['Allele']))}_holdout_{selected['Held_Out_AA']}"
    train_csv = args.output_dir / f"HLA_19AA_{split_token}_TrainTest_9mers_Labeled.csv"
    train_txt = args.output_dir / f"HLA_19AA_{split_token}_TrainTest_9mers_Labeled.txt"
    eval_csv = args.output_dir / f"HLA_19AA_{split_token}_Heldout_Evaluation_9mers_Labeled.csv"
    eval_txt = args.output_dir / f"HLA_19AA_{split_token}_Heldout_Evaluation_9mers_Labeled.txt"
    fold_csv = args.output_dir / f"HLA_19AA_{split_token}_Fold_Distribution.csv"

    write_records_csv(cv_records, train_csv, include_fold=True)
    write_labeled_txt(cv_records, train_txt, include_fold=True)
    write_records_csv(eval_records, eval_csv, include_fold=False)
    write_labeled_txt(eval_records, eval_txt, include_fold=False)
    save_rows_csv(summarize_fold_distribution(cv_records), fold_csv, ["Fold", "Rows", "Positive", "Negative"])

    print("Selected split:")
    print(
        f"  allele={selected['Allele']} held_out_aa={selected['Held_Out_AA']} "
        f"train/test={len(cv_records)} eval={len(eval_records)}"
    )
    print(
        f"  train positives={selected['Train_Positive']} negatives={selected['Train_Negative']} | "
        f"eval positives={selected['Eval_Positive']} negatives={selected['Eval_Negative']}"
    )
    print(f"Saved selected train/test split: {train_csv}")
    print(f"Saved selected held-out evaluation split: {eval_csv}")

    if args.prepare_only:
        print("Preparation complete. Re-run without --prepare-only to train the models.")
        return

    import numpy as np
    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"System hardware: {device}")

    encoder_specs = build_encoder_specs(args, device)
    experiments = [make_experiment(spec, selected, args.output_dir, args.folds) for spec in encoder_specs]
    assert_feature_dimensions(experiments, cv_records[0])

    cv_summary_rows = []
    eval_summary_rows = []
    for experiment in experiments:
        print("\n" + "=" * 90)
        print(f"Running {experiment['id']} | encoder={experiment['encoder_label']}")
        print("=" * 90)
        models = run_fivefold_cv(
            cv_records,
            experiment["featurizer"],
            experiment,
            device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            seed=args.seed,
        )
        eval_rows = predict_heldout_evaluation(
            eval_records,
            experiment["featurizer"],
            models,
            experiment,
            device,
            batch_size=max(args.batch_size, 128),
        )
        cv_rows = load_prediction_rows(Path(experiment["cv_csv"]))
        cv_summary_rows.append(summarize_cv_rows(cv_rows, experiment))
        eval_summary_rows.append(summarize_eval_rows(eval_rows, experiment))

    cv_summary_path = args.output_dir / f"HLA_19AA_{split_token}_CV_Summary.csv"
    eval_summary_path = args.output_dir / f"HLA_19AA_{split_token}_Heldout_Evaluation_Summary.csv"
    summary_fields = [
        "Experiment",
        "Encoder",
        "Allele",
        "Held_Out_AA",
        "Rows",
        "Positives",
        "Negatives",
        "Mean_AUC",
        "Std_AUC",
        "AUC",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Predicted_Positive",
        "Mean_Probability",
        "Median_Probability",
    ]
    cv_fields = [field for field in summary_fields if any(field in row for row in cv_summary_rows)]
    eval_fields = [field for field in summary_fields if any(field in row for row in eval_summary_rows)]
    save_rows_csv(cv_summary_rows, cv_summary_path, cv_fields)
    save_rows_csv(eval_summary_rows, eval_summary_path, eval_fields)
    print(f"Saved CV summary: {cv_summary_path}")
    print(f"Saved held-out evaluation summary: {eval_summary_path}")


if __name__ == "__main__":
    main()
