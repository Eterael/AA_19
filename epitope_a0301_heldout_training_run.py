"""Train held-out-AA binding models on new epitope positives plus A03:01 negatives.

The new XLSX epitope export is treated as an all-binder positive set for
HLA-A03:01. Non-binders are taken from hla_only.txt for the same allele. The
combined labeled dataset is then split by held-out amino acid and trained with
the same AE, one-hot, and BLOSUM62 linear networks as the HLA-only runs.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

from hla_19aa_heldout_run import (
    DEFAULT_AE_WEIGHTS,
    DEFAULT_HLA_FILE,
    DEFAULT_IMAGE_DIR,
    FOLDS,
    LEARNING_RATE,
    BATCH_SIZE,
    EPOCHS,
    RANDOM_SEED,
    STANDARD_AA,
    STANDARD_AA_SET,
    Record,
    assert_feature_dimensions,
    build_encoder_specs,
    build_selected_split,
    choose_candidate,
    compute_split_candidates,
    label_counts,
    load_prediction_rows,
    predict_heldout_evaluation,
    run_fivefold_cv,
    safe_token,
    save_rows_csv,
    summarize_cv_rows,
    summarize_eval_rows,
    summarize_fold_distribution,
    write_labeled_txt,
)


PROJECT_DIR = Path(".")
DEFAULT_EPITOPE_CSV = PROJECT_DIR / "epitope_table_new_data_results" / "Epitope_Table_Canonical_9mers.csv"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "epitope_a0301_heldout_results"
DEFAULT_ALLELE = "HLA-A03:01"
DEFAULT_HELD_OUT_AA = "W"


def normalize_allele(value: str) -> str:
    value = value.strip()
    if value == "HLA-A*03:01":
        return "HLA-A03:01"
    return value


def read_epitope_positive_peptides(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing canonical epitope CSV: {path}. "
            "Run Epitope_Table_Heldout_Run.ipynb first to prepare the XLSX."
        )

    peptides = set()
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            peptide = row["Peptide"].strip().upper()
            if len(peptide) == 9 and set(peptide) <= STANDARD_AA_SET:
                peptides.add(peptide)
    if not peptides:
        raise ValueError(f"No canonical 9-mer epitope positives were found in {path}")
    return peptides


def read_hla_only_negatives(
    path: Path,
    allele: str,
    positive_peptides: set[str],
    allow_fallback_negatives: bool,
    min_negative_rows: int,
) -> tuple[set[str], Counter]:
    stats = Counter()
    primary_negatives = set()
    fallback_negatives = set()
    target_allele = normalize_allele(allele)

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 3:
                stats["malformed"] += 1
                continue
            peptide, label_text, row_allele = parts[0].upper(), parts[1], normalize_allele(parts[2])
            if label_text != "0":
                continue
            if len(peptide) != 9 or set(peptide) - STANDARD_AA_SET:
                stats["invalid_negative_peptide"] += 1
                continue
            if peptide in positive_peptides:
                stats["negative_overlaps_epitope_positive_excluded"] += 1
                continue
            if row_allele == target_allele:
                primary_negatives.add(peptide)
            elif allow_fallback_negatives:
                fallback_negatives.add(peptide)

    selected = set(primary_negatives)
    stats["primary_same_allele_negatives"] = len(primary_negatives)
    if allow_fallback_negatives and len(selected) < min_negative_rows:
        needed = min_negative_rows - len(selected)
        selected.update(sorted(fallback_negatives)[:needed])
        stats["fallback_negatives_used"] = len(selected) - len(primary_negatives)
    else:
        stats["fallback_negatives_used"] = 0
    stats["selected_negatives"] = len(selected)
    return selected, stats


def build_combined_records(positive_peptides: set[str], negative_peptides: set[str], allele: str) -> list[Record]:
    records = [Record(peptide=peptide, label=1, allele=allele) for peptide in sorted(positive_peptides)]
    records.extend(Record(peptide=peptide, label=0, allele=allele) for peptide in sorted(negative_peptides))
    return sorted(records, key=lambda row: (row.label, row.peptide))


def write_records_with_source_csv(
    records: Iterable[Record],
    positive_peptides: set[str],
    path: Path,
    include_fold: bool,
) -> None:
    rows = []
    for row in records:
        out = {
            "Peptide": row.peptide,
            "Label": row.label,
            "Allele": row.allele,
            "Source": "epitope_xlsx_binder" if row.peptide in positive_peptides else "hla_only_a0301_nonbinder",
        }
        if include_fold:
            out["Fold"] = row.fold
        rows.append(out)
    fieldnames = ["Peptide", "Label", "Allele", "Source"] + (["Fold"] if include_fold else [])
    save_rows_csv(rows, path, fieldnames)


def make_epitope_experiment(encoder_spec: dict[str, object], selected: dict[str, object], output_dir: Path, folds: int) -> dict[str, object]:
    allele = str(selected["Allele"])
    held_out_aa = str(selected["Held_Out_AA"])
    encoder_id = str(encoder_spec["id"])
    split_token = f"{safe_token(allele)}_epitope_bind_pos_hla_neg_holdout_{held_out_aa}"
    experiment_id = f"{split_token}__{encoder_id}"
    return {
        "id": experiment_id,
        "encoder_id": encoder_id,
        "encoder_label": encoder_spec["label"],
        "allele": allele,
        "held_out_aa": held_out_aa,
        "folds": folds,
        "output_dir": output_dir,
        "model_prefix": f"epitope_a0301_19aa_{split_token}_{encoder_id}",
        "cv_csv": output_dir / f"Epitope_A0301_19AA_CV_Binding_Predictions_{experiment_id}.csv",
        "eval_csv": output_dir / f"Epitope_A0301_19AA_Heldout_Evaluation_{experiment_id}.csv",
        "featurizer": encoder_spec["featurizer"],
        "expected_dim": encoder_spec["expected_dim"],
    }


def run_workflow(args: SimpleNamespace) -> dict[str, object]:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    positives = read_epitope_positive_peptides(args.epitope_csv)
    negatives, negative_stats = read_hla_only_negatives(
        args.hla_file,
        allele=args.allele,
        positive_peptides=positives,
        allow_fallback_negatives=args.allow_fallback_negatives,
        min_negative_rows=args.min_negative_rows,
    )
    if len(negatives) < args.min_negative_rows:
        raise ValueError(
            f"Only {len(negatives)} usable negatives were found for {args.allele}. "
            "Lower MIN_NEGATIVE_ROWS or enable ALLOW_FALLBACK_NEGATIVES."
        )

    combined_records = build_combined_records(positives, negatives, args.allele)
    records_by_allele = {args.allele: combined_records}
    candidates = compute_split_candidates(
        records_by_allele,
        min_train_rows=args.min_train_rows,
        min_eval_rows=args.min_eval_rows,
        min_train_per_class=args.min_train_per_class,
        min_eval_per_class=args.min_eval_per_class,
    )

    candidate_csv = args.output_dir / "Epitope_A0301_19AA_Split_Candidates.csv"
    sorted_candidates = sorted(candidates, key=lambda row: (row["Valid"], row["Train_Rows"], row["Eval_Rows"]), reverse=True)
    save_rows_csv(
        sorted_candidates,
        candidate_csv,
        ["Allele", "Held_Out_AA", "Allele_Total", "Train_Rows", "Train_Positive", "Train_Negative", "Eval_Rows", "Eval_Positive", "Eval_Negative", "Valid"],
    )

    selected = choose_candidate(candidates, allele=args.allele, held_out_aa=args.held_out_aa)
    cv_records, eval_records = build_selected_split(records_by_allele, selected, folds=args.folds, seed=args.seed)
    split_token = f"{safe_token(str(selected['Allele']))}_epitope_bind_pos_hla_neg_holdout_{selected['Held_Out_AA']}"

    train_csv = args.output_dir / f"Epitope_A0301_19AA_{split_token}_TrainTest_9mers_Labeled.csv"
    train_txt = args.output_dir / f"Epitope_A0301_19AA_{split_token}_TrainTest_9mers_Labeled.txt"
    eval_csv = args.output_dir / f"Epitope_A0301_19AA_{split_token}_Heldout_Evaluation_9mers_Labeled.csv"
    eval_txt = args.output_dir / f"Epitope_A0301_19AA_{split_token}_Heldout_Evaluation_9mers_Labeled.txt"
    fold_csv = args.output_dir / f"Epitope_A0301_19AA_{split_token}_Fold_Distribution.csv"
    source_summary_csv = args.output_dir / "Epitope_A0301_19AA_Source_Summary.csv"

    write_records_with_source_csv(cv_records, positives, train_csv, include_fold=True)
    write_labeled_txt(cv_records, train_txt, include_fold=True)
    write_records_with_source_csv(eval_records, positives, eval_csv, include_fold=False)
    write_labeled_txt(eval_records, eval_txt, include_fold=False)
    save_rows_csv(summarize_fold_distribution(cv_records), fold_csv, ["Fold", "Rows", "Positive", "Negative"])

    source_rows = [
        {"Metric": "Positive epitope binders from XLSX", "Value": len(positives)},
        {"Metric": f"Same-allele HLA-only non-binders for {args.allele}", "Value": negative_stats["primary_same_allele_negatives"]},
        {"Metric": "Fallback non-binders used", "Value": negative_stats["fallback_negatives_used"]},
        {"Metric": "Selected non-binders", "Value": len(negatives)},
        {"Metric": "Total labeled peptides", "Value": len(combined_records)},
        {"Metric": "Held-out amino acid", "Value": selected["Held_Out_AA"]},
        {"Metric": "Train/CV rows", "Value": len(cv_records)},
        {"Metric": "Train/CV positives", "Value": selected["Train_Positive"]},
        {"Metric": "Train/CV negatives", "Value": selected["Train_Negative"]},
        {"Metric": "Held-out evaluation rows", "Value": len(eval_records)},
        {"Metric": "Held-out evaluation positives", "Value": selected["Eval_Positive"]},
        {"Metric": "Held-out evaluation negatives", "Value": selected["Eval_Negative"]},
    ]
    save_rows_csv(source_rows, source_summary_csv, ["Metric", "Value"])

    print("Epitope A03:01 labeled held-out run")
    print(f"  positives from XLSX: {len(positives)}")
    print(f"  negatives from hla_only.txt ({args.allele}): {negative_stats['primary_same_allele_negatives']}")
    print(f"  fallback negatives used: {negative_stats['fallback_negatives_used']}")
    print(
        f"  selected split: allele={selected['Allele']} held_out_aa={selected['Held_Out_AA']} "
        f"train/test={len(cv_records)} eval={len(eval_records)}"
    )
    print(
        f"  train positives={selected['Train_Positive']} negatives={selected['Train_Negative']} | "
        f"eval positives={selected['Eval_Positive']} negatives={selected['Eval_Negative']}"
    )
    print(f"  saved train/test split: {train_csv}")
    print(f"  saved held-out evaluation split: {eval_csv}")

    if args.prepare_only:
        print("Preparation complete. Set PREPARE_ONLY = False in the notebook to train the models.")
        return {
            "selected": selected,
            "cv_records": cv_records,
            "eval_records": eval_records,
            "cv_summary_path": None,
            "eval_summary_path": None,
        }

    import numpy as np
    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"System hardware: {device}")

    encoder_specs = build_encoder_specs(args, device)
    experiments = [make_epitope_experiment(spec, selected, args.output_dir, args.folds) for spec in encoder_specs]
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

    cv_summary_path = args.output_dir / f"Epitope_A0301_19AA_{split_token}_CV_Summary.csv"
    eval_summary_path = args.output_dir / f"Epitope_A0301_19AA_{split_token}_Heldout_Evaluation_Summary.csv"
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
    return {
        "selected": selected,
        "cv_records": cv_records,
        "eval_records": eval_records,
        "cv_summary_path": cv_summary_path,
        "eval_summary_path": eval_summary_path,
    }


def default_args() -> SimpleNamespace:
    return SimpleNamespace(
        epitope_csv=DEFAULT_EPITOPE_CSV,
        hla_file=DEFAULT_HLA_FILE,
        output_dir=DEFAULT_OUTPUT_DIR,
        ae_weights=DEFAULT_AE_WEIGHTS,
        image_dir=DEFAULT_IMAGE_DIR,
        allele=DEFAULT_ALLELE,
        held_out_aa=DEFAULT_HELD_OUT_AA,
        min_negative_rows=100,
        allow_fallback_negatives=False,
        min_train_rows=1000,
        min_eval_rows=100,
        min_train_per_class=25,
        min_eval_per_class=10,
        folds=FOLDS,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        seed=RANDOM_SEED,
        encoders=["ae", "onehot20", "blosum62_20"],
        prepare_only=False,
    )


def parse_args() -> SimpleNamespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epitope-csv", type=Path, default=DEFAULT_EPITOPE_CSV)
    parser.add_argument("--hla-file", type=Path, default=DEFAULT_HLA_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ae-weights", type=Path, default=DEFAULT_AE_WEIGHTS)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--allele", default=DEFAULT_ALLELE)
    parser.add_argument("--held-out-aa", default=DEFAULT_HELD_OUT_AA)
    parser.add_argument("--min-negative-rows", type=int, default=100)
    parser.add_argument("--allow-fallback-negatives", action="store_true")
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
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(namespace=default_args())
    args.allele = normalize_allele(args.allele)
    args.held_out_aa = args.held_out_aa.upper()
    if args.held_out_aa not in STANDARD_AA_SET:
        raise ValueError(f"--held-out-aa must be one of {''.join(STANDARD_AA)}")
    return args


if __name__ == "__main__":
    run_workflow(parse_args())
