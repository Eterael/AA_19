"""Analyze the balanced epitope A03:01 held-out training run.

This script reads the split and prediction CSVs produced by
Epitope_A0301_Heldout_Training_Run.ipynb and writes dataset summaries,
performance summaries, ROC curves, confusion matrices, sequence-logo SVGs, and
top-prediction logo summaries.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_DIR = Path(".")
OUTPUT_DIR = PROJECT_DIR / "epitope_a0301_heldout_results"
NOTEBOOK_PATH = PROJECT_DIR / "Epitope_A0301_Heldout_Analysis.ipynb"
STANDARD_AA = tuple("ACDEFGHIKLMNPQRSTVWY")
THRESHOLD = 0.5
TOP_PREDICTION_SLICES = [
    ("Top 1%", 0.00, 0.01),
    ("Top 1-10%", 0.01, 0.10),
    ("Top 10%", 0.00, 0.10),
]
ENCODER_ORDER = ["onehot20", "blosum62_20", "ae"]
ENCODER_COLORS = {
    "onehot20": "#1f77b4",
    "blosum62_20": "#2ca02c",
    "ae": "#d62728",
}
AA_COLORS = {
    "A": "#4C78A8",
    "V": "#4C78A8",
    "L": "#4C78A8",
    "I": "#4C78A8",
    "M": "#4C78A8",
    "F": "#4C78A8",
    "W": "#4C78A8",
    "Y": "#4C78A8",
    "P": "#4C78A8",
    "S": "#59A14F",
    "T": "#59A14F",
    "N": "#59A14F",
    "Q": "#59A14F",
    "C": "#59A14F",
    "D": "#E15759",
    "E": "#E15759",
    "K": "#B07AA1",
    "R": "#B07AA1",
    "H": "#B07AA1",
    "G": "#9C755F",
}


def safe_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, object]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def infer_split_token(output_dir: Path, split_token: str | None = None) -> str:
    if split_token:
        return split_token
    prefix = "Epitope_A0301_19AA_"
    suffix = "_TrainTest_9mers_Labeled.csv"
    candidates = sorted(output_dir.glob(f"{prefix}*{suffix}"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No train/test split files found in {output_dir}")
    return candidates[0].name[len(prefix) : -len(suffix)]


def load_split(output_dir: Path, split_token: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    train_path = output_dir / f"Epitope_A0301_19AA_{split_token}_TrainTest_9mers_Labeled.csv"
    eval_path = output_dir / f"Epitope_A0301_19AA_{split_token}_Heldout_Evaluation_9mers_Labeled.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train/test split: {train_path}")
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing held-out evaluation split: {eval_path}")
    train_rows = read_csv(train_path)
    eval_rows = read_csv(eval_path)
    for row in train_rows + eval_rows:
        row["True_Class"] = row.get("Label", row.get("True_Class", ""))
    return train_rows, eval_rows


def load_prediction_outputs(output_dir: Path, split_token: str) -> tuple[dict[str, list[dict[str, str]]], dict[str, list[dict[str, str]]]]:
    cv_outputs = {}
    eval_outputs = {}
    cv_prefix = "Epitope_A0301_19AA_CV_Binding_Predictions_"
    eval_prefix = "Epitope_A0301_19AA_Heldout_Evaluation_"
    for path in sorted(output_dir.glob(f"{cv_prefix}{split_token}__*.csv")):
        encoder = path.stem.split("__")[-1]
        cv_outputs[encoder] = read_csv(path)
    for path in sorted(output_dir.glob(f"{eval_prefix}{split_token}__*.csv")):
        encoder = path.stem.split("__")[-1]
        eval_outputs[encoder] = read_csv(path)
    return cv_outputs, eval_outputs


def int_label(value: object) -> int:
    return int(float(value))


def float_score(value: object) -> float:
    return float(value)


def rank_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    positives = sum(1 for value in y_true if value == 1)
    negatives = sum(1 for value in y_true if value == 0)
    if positives == 0 or negatives == 0:
        return math.nan

    pairs = sorted(zip(y_score, y_true), key=lambda item: item[0])
    rank_sum_positive = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        rank_sum_positive += average_rank * sum(1 for _, label in pairs[index:end] if label == 1)
        index = end
    return (rank_sum_positive - positives * (positives + 1) / 2.0) / (positives * negatives)


def roc_points(y_true: Sequence[int], y_score: Sequence[float]) -> list[dict[str, float]]:
    positives = sum(1 for value in y_true if value == 1)
    negatives = sum(1 for value in y_true if value == 0)
    if positives == 0 or negatives == 0:
        return [{"FPR": 0.0, "TPR": 0.0}]
    pairs = sorted(zip(y_score, y_true), key=lambda item: item[0], reverse=True)
    points = [{"FPR": 0.0, "TPR": 0.0, "Threshold": 1.0}]
    tp = 0
    fp = 0
    index = 0
    while index < len(pairs):
        threshold = pairs[index][0]
        while index < len(pairs) and pairs[index][0] == threshold:
            label = pairs[index][1]
            if label == 1:
                tp += 1
            else:
                fp += 1
            index += 1
        points.append({"FPR": fp / negatives, "TPR": tp / positives, "Threshold": threshold})
    if points[-1]["FPR"] != 1.0 or points[-1]["TPR"] != 1.0:
        points.append({"FPR": 1.0, "TPR": 1.0, "Threshold": 0.0})
    return points


def threshold_metrics(y_true: Sequence[int], y_score: Sequence[float], threshold: float = THRESHOLD) -> dict[str, object]:
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
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Accuracy": (tp + tn) / total if total else math.nan,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Predicted_Positive": tp + fp,
    }


def performance_row(encoder: str, split: str, rows: Sequence[dict[str, str]], score_col: str) -> dict[str, object]:
    y_true = [int_label(row["True_Class"]) for row in rows]
    y_score = [float_score(row[score_col]) for row in rows]
    metrics = threshold_metrics(y_true, y_score)
    return {
        "Encoder": encoder,
        "Split": split,
        "Rows": len(rows),
        "Positives": sum(y_true),
        "Negatives": len(y_true) - sum(y_true),
        "AUC": rank_auc(y_true, y_score),
        **metrics,
    }


def dataset_summary_row(name: str, rows: Sequence[dict[str, str]]) -> dict[str, object]:
    labels = [int_label(row["True_Class"]) for row in rows]
    source_counts = Counter(row.get("Source_Group", "") for row in rows)
    return {
        "Dataset": name,
        "Rows": len(rows),
        "Positives": sum(labels),
        "Negatives": len(labels) - sum(labels),
        "Positive_Rate": sum(labels) / len(labels) if labels else math.nan,
        "Unique_Peptides": len({row["Peptide"] for row in rows}),
        "Epitope_XLSX_Positives": source_counts.get("positive_epitope_xlsx", 0),
        "Same_Allele_Negatives": source_counts.get("negative_same_allele_hla_only", 0),
        "Other_Allele_Negatives": source_counts.get("negative_other_allele_hla_only", 0),
    }


def position_counts(peptides: Iterable[str]) -> list[Counter]:
    counts = [Counter() for _ in range(9)]
    for peptide in peptides:
        for index, aa in enumerate(peptide):
            if index < 9:
                counts[index][aa] += 1
    return counts


def position_frequency_rows(peptides: Sequence[str], dataset_name: str) -> list[dict[str, object]]:
    counts = position_counts(peptides)
    total = len(peptides)
    rows = []
    for position, counter in enumerate(counts, start=1):
        for aa in STANDARD_AA:
            count = counter.get(aa, 0)
            rows.append(
                {
                    "Dataset": dataset_name,
                    "Position": position,
                    "AA": aa,
                    "Count": count,
                    "Frequency": count / total if total else 0.0,
                }
            )
    return rows


def aa_composition_rows(rows: Sequence[dict[str, str]], dataset_name: str) -> list[dict[str, object]]:
    peptides = [row["Peptide"] for row in rows]
    counts = Counter("".join(peptides))
    total = sum(counts.values())
    return [
        {
            "Dataset": dataset_name,
            "AA": aa,
            "Count": counts.get(aa, 0),
            "Frequency": counts.get(aa, 0) / total if total else 0.0,
        }
        for aa in STANDARD_AA
    ]


def logo_panel_svg(peptides: Sequence[str], title: str, panel_width: int = 760, panel_height: int = 300) -> str:
    peptides = list(peptides)
    max_bits = math.log2(len(STANDARD_AA))
    left = 52
    top = 42
    chart_height = 204
    col_width = 68
    chart_width = 9 * col_width
    height_per_bit = chart_height / max_bits
    counts = position_counts(peptides)
    lines = [
        f'<text x="{left}" y="22" font-size="16" font-family="Arial, sans-serif" font-weight="700" fill="#222222">{title} (n={len(peptides)})</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#222222" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}" stroke="#222222" stroke-width="1"/>',
    ]
    for tick in range(5):
        y = top + chart_height - tick * height_per_bit
        lines.append(f'<line x1="{left - 4}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#777777" stroke-width="1"/>')
        lines.append(f'<text x="{left - 28}" y="{y + 4:.2f}" font-size="10" font-family="Arial, sans-serif" fill="#555555">{tick}</text>')

    for pos_index, counter in enumerate(counts):
        x = left + pos_index * col_width + 10
        bottom = top + chart_height
        total = sum(counter.values())
        if total:
            entropy = -sum((count / total) * math.log2(count / total) for count in counter.values() if count)
            information = max_bits - entropy
            contributions = []
            for aa in STANDARD_AA:
                freq = counter.get(aa, 0) / total
                if freq:
                    contributions.append((aa, freq * information))
            for aa, bits in sorted(contributions, key=lambda item: item[1]):
                block_height = bits * height_per_bit
                if block_height < 1:
                    continue
                y = bottom - block_height
                color = AA_COLORS.get(aa, "#555555")
                lines.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="46" height="{block_height:.2f}" fill="{color}" opacity="0.92"/>')
                if block_height >= 9:
                    lines.append(
                        f'<text x="{x + 23:.2f}" y="{y + block_height / 2 + 4:.2f}" text-anchor="middle" '
                        f'font-size="11" font-family="Arial, sans-serif" font-weight="700" fill="#ffffff">{aa}</text>'
                    )
                bottom = y
        lines.append(
            f'<text x="{x + 23:.2f}" y="{top + chart_height + 24}" text-anchor="middle" font-size="12" '
            f'font-family="Arial, sans-serif" fill="#222222">P{pos_index + 1}</text>'
        )
    return "\n".join(lines)


def logo_collection_svg(logo_sets: Sequence[tuple[str, Sequence[str]]], title: str) -> str:
    panel_width = 760
    panel_height = 300
    width = panel_width
    height = 52 + panel_height * len(logo_sets)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="28" font-size="20" font-family="Arial, sans-serif" font-weight="700" fill="#111111">{title}</text>',
    ]
    for index, (set_title, peptides) in enumerate(logo_sets):
        y_offset = 48 + index * panel_height
        lines.append(f'<g transform="translate(0,{y_offset})">')
        lines.append(logo_panel_svg(list(peptides), set_title, panel_width, panel_height))
        lines.append("</g>")
    lines.append("</svg>")
    return "\n".join(lines)


def roc_svg(roc_by_encoder: dict[str, list[dict[str, float]]], auc_by_encoder: dict[str, float], title: str) -> str:
    width = 720
    height = 540
    left = 72
    top = 44
    plot = 420
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="26" font-size="18" font-family="Arial, sans-serif" font-weight="700">{title}</text>',
        f'<rect x="{left}" y="{top}" width="{plot}" height="{plot}" fill="#ffffff" stroke="#222222" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top + plot}" x2="{left + plot}" y2="{top}" stroke="#bbbbbb" stroke-width="1" stroke-dasharray="6 6"/>',
    ]
    for tick in range(6):
        value = tick / 5
        x = left + value * plot
        y = top + plot - value * plot
        lines.append(f'<line x1="{x:.2f}" y1="{top + plot}" x2="{x:.2f}" y2="{top + plot + 5}" stroke="#555555"/>')
        lines.append(f'<line x1="{left - 5}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#555555"/>')
        lines.append(f'<text x="{x:.2f}" y="{top + plot + 22}" text-anchor="middle" font-size="11" font-family="Arial, sans-serif">{value:.1f}</text>')
        lines.append(f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-size="11" font-family="Arial, sans-serif">{value:.1f}</text>')
    lines.append(f'<text x="{left + plot / 2}" y="{top + plot + 46}" text-anchor="middle" font-size="13" font-family="Arial, sans-serif">False positive rate</text>')
    lines.append(f'<text x="18" y="{top + plot / 2}" text-anchor="middle" font-size="13" font-family="Arial, sans-serif" transform="rotate(-90,18,{top + plot / 2})">True positive rate</text>')
    legend_y = top + 18
    for idx, encoder in enumerate(sorted(roc_by_encoder, key=lambda enc: ENCODER_ORDER.index(enc) if enc in ENCODER_ORDER else 99)):
        color = ENCODER_COLORS.get(encoder, "#333333")
        path = []
        for point in roc_by_encoder[encoder]:
            x = left + point["FPR"] * plot
            y = top + plot - point["TPR"] * plot
            path.append(f"{x:.2f},{y:.2f}")
        if path:
            lines.append(f'<polyline points="{" ".join(path)}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        ly = legend_y + idx * 24
        lines.append(f'<line x1="{left + plot + 34}" y1="{ly}" x2="{left + plot + 64}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        lines.append(
            f'<text x="{left + plot + 72}" y="{ly + 4}" font-size="12" font-family="Arial, sans-serif">{encoder} AUC={auc_by_encoder.get(encoder, math.nan):.4f}</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines)


def confusion_collection_svg(rows: Sequence[dict[str, object]], title: str) -> str:
    panel_width = 300
    panel_height = 250
    width = panel_width * len(rows)
    height = panel_height + 54
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="20" y="28" font-size="18" font-family="Arial, sans-serif" font-weight="700">{title}</text>',
    ]
    for idx, row in enumerate(rows):
        x0 = idx * panel_width + 42
        y0 = 74
        encoder = str(row["Encoder"])
        cells = [
            ("TN", int(row["TN"]), 0, 0, "#d7ebff"),
            ("FP", int(row["FP"]), 1, 0, "#ffd6d6"),
            ("FN", int(row["FN"]), 0, 1, "#ffd6d6"),
            ("TP", int(row["TP"]), 1, 1, "#d9f2d9"),
        ]
        lines.append(f'<text x="{x0}" y="{y0 - 28}" font-size="14" font-family="Arial, sans-serif" font-weight="700">{encoder}</text>')
        lines.append(f'<text x="{x0 + 86}" y="{y0 - 8}" text-anchor="middle" font-size="11" font-family="Arial, sans-serif">Pred 0</text>')
        lines.append(f'<text x="{x0 + 184}" y="{y0 - 8}" text-anchor="middle" font-size="11" font-family="Arial, sans-serif">Pred 1</text>')
        lines.append(f'<text x="{x0 - 12}" y="{y0 + 56}" text-anchor="end" font-size="11" font-family="Arial, sans-serif">True 0</text>')
        lines.append(f'<text x="{x0 - 12}" y="{y0 + 154}" text-anchor="end" font-size="11" font-family="Arial, sans-serif">True 1</text>')
        for label, count, cx, cy, color in cells:
            x = x0 + cx * 98
            y = y0 + cy * 98
            lines.append(f'<rect x="{x}" y="{y}" width="92" height="92" fill="{color}" stroke="#333333"/>')
            lines.append(f'<text x="{x + 46}" y="{y + 36}" text-anchor="middle" font-size="15" font-family="Arial, sans-serif" font-weight="700">{label}</text>')
            lines.append(f'<text x="{x + 46}" y="{y + 62}" text-anchor="middle" font-size="18" font-family="Arial, sans-serif">{count}</text>')
        lines.append(f'<text x="{x0}" y="{y0 + 214}" font-size="11" font-family="Arial, sans-serif">Acc={float(row["Accuracy"]):.3f} F1={float(row["F1"]):.3f}</text>')
    lines.append("</svg>")
    return "\n".join(lines)


def prediction_slice(rows: Sequence[dict[str, str]], score_col: str, lower: float, upper: float) -> list[dict[str, str]]:
    sorted_rows = sorted(rows, key=lambda row: float_score(row[score_col]), reverse=True)
    start = int(math.floor(len(sorted_rows) * lower))
    end = int(math.ceil(len(sorted_rows) * upper))
    return sorted_rows[start:end]


def fmt(value: object) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.4f}"
    return str(value)


def markdown_table(rows: Sequence[dict[str, object]], columns: Sequence[str], max_rows: int | None = None) -> str:
    visible = list(rows[:max_rows]) if max_rows else list(rows)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in visible:
        lines.append("| " + " | ".join(fmt(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def build_notebook(summary: dict[str, object], performance_rows: list[dict[str, object]], dataset_rows: list[dict[str, object]], top_rows: list[dict[str, object]]) -> dict[str, object]:
    analysis_dir = summary["analysis_dir"]
    split_token = summary["split_token"]
    perf_cols = ["Encoder", "Split", "Rows", "Positives", "Negatives", "AUC", "Accuracy", "Precision", "Recall", "F1"]
    dataset_cols = ["Dataset", "Rows", "Positives", "Negatives", "Positive_Rate", "Unique_Peptides", "Epitope_XLSX_Positives", "Same_Allele_Negatives", "Other_Allele_Negatives"]
    top_cols = ["Encoder", "Slice", "Rows", "Positives", "Negatives", "Mean_Probability", "Min_Probability", "Max_Probability"]
    run_output = (
        f"Split token: {split_token}\n"
        f"Analysis directory: {analysis_dir}\n"
        f"Dataset summary rows: {len(dataset_rows)}\n"
        f"Performance rows: {len(performance_rows)}\n"
        f"Top prediction slice rows: {len(top_rows)}\n"
        "Saved sequence-logo SVGs, ROC SVGs, confusion-matrix SVGs, and CSV summaries.\n"
    )
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": (
                "# Epitope A03:01 Held-Out Analysis\n\n"
                "This notebook analyzes the balanced epitope A03:01 held-out-AA run. It reads the train/CV split, held-out evaluation split, and model prediction CSVs from `epitope_a0301_heldout_results/`.\n\n"
                "It reports dataset/source composition, AUC, ROC curves, confusion matrices, and sequence-logo analyses for the dataset, binders, non-binders, evaluation set, ensemble predicted binders, top 1%, top 1-10%, and top 10% predictions."
            ),
        },
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [{"name": "stdout", "output_type": "stream", "text": run_output}],
            "source": "# Recreate the analysis outputs.\nfrom epitope_a0301_heldout_analysis import run_analysis\nsummary = run_analysis(create_notebook=False)\n",
        },
        {"cell_type": "markdown", "metadata": {}, "source": "## Dataset Summary\n\n" + markdown_table(dataset_rows, dataset_cols)},
        {"cell_type": "markdown", "metadata": {}, "source": "## Performance Summary\n\n" + markdown_table(performance_rows, perf_cols)},
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": (
                "## ROC Curves\n\n"
                f"CV out-of-fold ROC:\n\n![CV ROC]({analysis_dir}/{split_token}_CV_ROC.svg)\n\n"
                f"Held-out ensemble ROC:\n\n![Held-out ROC]({analysis_dir}/{split_token}_Heldout_Ensemble_ROC.svg)"
            ),
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": (
                "## Confusion Matrices\n\n"
                f"CV out-of-fold confusion matrices:\n\n![CV confusion]({analysis_dir}/{split_token}_CV_Confusion_Matrices.svg)\n\n"
                f"Held-out ensemble confusion matrices:\n\n![Held-out confusion]({analysis_dir}/{split_token}_Heldout_Ensemble_Confusion_Matrices.svg)"
            ),
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": (
                "## Dataset Sequence Logos\n\n"
                f"![Dataset logos]({analysis_dir}/{split_token}_Dataset_Sequence_Logos.svg)"
            ),
        },
        {"cell_type": "markdown", "metadata": {}, "source": "## Top Prediction Slices\n\n" + markdown_table(top_rows, top_cols)},
    ]
    for encoder in sorted({row["Encoder"] for row in top_rows}, key=lambda enc: ENCODER_ORDER.index(enc) if enc in ENCODER_ORDER else 99):
        cells.append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": (
                    f"## Prediction Sequence Logos: {encoder}\n\n"
                    f"![{encoder} prediction logos]({analysis_dir}/{split_token}_{encoder}_Prediction_Sequence_Logos.svg)"
                ),
            }
        )
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def run_analysis(split_token: str | None = None, create_notebook: bool = True) -> dict[str, object]:
    split_token = infer_split_token(OUTPUT_DIR, split_token)
    analysis_dir = OUTPUT_DIR / f"analysis_{split_token}"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    train_rows, eval_rows = load_split(OUTPUT_DIR, split_token)
    all_rows = train_rows + eval_rows
    cv_outputs, eval_outputs = load_prediction_outputs(OUTPUT_DIR, split_token)

    dataset_sets = [
        ("All labeled peptides", all_rows),
        ("Train/CV all", train_rows),
        ("Train/CV binders", [row for row in train_rows if int_label(row["True_Class"]) == 1]),
        ("Train/CV non-binders", [row for row in train_rows if int_label(row["True_Class"]) == 0]),
        ("Held-out evaluation all", eval_rows),
        ("Held-out evaluation binders", [row for row in eval_rows if int_label(row["True_Class"]) == 1]),
        ("Held-out evaluation non-binders", [row for row in eval_rows if int_label(row["True_Class"]) == 0]),
        ("Held-out eval same-allele negatives", [row for row in eval_rows if row.get("Source_Group") == "negative_same_allele_hla_only"]),
        ("Held-out eval other-allele negatives", [row for row in eval_rows if row.get("Source_Group") == "negative_other_allele_hla_only"]),
    ]
    dataset_rows = [dataset_summary_row(name, rows) for name, rows in dataset_sets]
    write_csv(analysis_dir / f"{split_token}_Dataset_Summary.csv", dataset_rows)
    write_csv(
        analysis_dir / f"{split_token}_AA_Composition.csv",
        [row for name, rows in dataset_sets for row in aa_composition_rows(rows, name)],
    )
    write_csv(
        analysis_dir / f"{split_token}_Position_Frequencies_For_Logos.csv",
        [row for name, rows in dataset_sets for row in position_frequency_rows([item["Peptide"] for item in rows], name)],
    )
    write_text(
        analysis_dir / f"{split_token}_Dataset_Sequence_Logos.svg",
        logo_collection_svg([(name, [row["Peptide"] for row in rows]) for name, rows in dataset_sets], "Dataset sequence logos"),
    )

    performance_rows = []
    fold_rows = []
    roc_rows = []
    confusion_rows = []
    cv_roc_by_encoder = {}
    cv_auc_by_encoder = {}
    eval_roc_by_encoder = {}
    eval_auc_by_encoder = {}

    for encoder, rows in cv_outputs.items():
        perf = performance_row(encoder, "CV out-of-fold", rows, "Probability")
        performance_rows.append(perf)
        confusion_rows.append({**perf, "Split": "CV out-of-fold"})
        y_true = [int_label(row["True_Class"]) for row in rows]
        y_score = [float_score(row["Probability"]) for row in rows]
        cv_roc_by_encoder[encoder] = roc_points(y_true, y_score)
        cv_auc_by_encoder[encoder] = perf["AUC"]
        for point in cv_roc_by_encoder[encoder]:
            roc_rows.append({"Encoder": encoder, "Split": "CV out-of-fold", **point})
        for fold in sorted({row["Fold"] for row in rows}, key=lambda value: int(float(value))):
            fold_subset = [row for row in rows if row["Fold"] == fold]
            fold_perf = performance_row(encoder, "CV fold", fold_subset, "Probability")
            fold_perf["Fold"] = fold
            fold_rows.append(fold_perf)

    for encoder, rows in eval_outputs.items():
        perf = performance_row(encoder, "Held-out ensemble", rows, "Ensemble_Probability")
        performance_rows.append(perf)
        confusion_rows.append({**perf, "Split": "Held-out ensemble"})
        y_true = [int_label(row["True_Class"]) for row in rows]
        y_score = [float_score(row["Ensemble_Probability"]) for row in rows]
        eval_roc_by_encoder[encoder] = roc_points(y_true, y_score)
        eval_auc_by_encoder[encoder] = perf["AUC"]
        for point in eval_roc_by_encoder[encoder]:
            roc_rows.append({"Encoder": encoder, "Split": "Held-out ensemble", **point})

    performance_rows = sorted(
        performance_rows,
        key=lambda row: (row["Split"], -float(row["AUC"]), ENCODER_ORDER.index(row["Encoder"]) if row["Encoder"] in ENCODER_ORDER else 99),
    )
    write_csv(analysis_dir / f"{split_token}_Detailed_Performance.csv", performance_rows)
    write_csv(analysis_dir / f"{split_token}_Fold_Performance.csv", fold_rows)
    write_csv(analysis_dir / f"{split_token}_ROC_Points.csv", roc_rows)
    write_csv(analysis_dir / f"{split_token}_Confusion_Matrices.csv", confusion_rows)
    write_text(analysis_dir / f"{split_token}_CV_ROC.svg", roc_svg(cv_roc_by_encoder, cv_auc_by_encoder, "CV out-of-fold ROC"))
    write_text(analysis_dir / f"{split_token}_Heldout_Ensemble_ROC.svg", roc_svg(eval_roc_by_encoder, eval_auc_by_encoder, "Held-out ensemble ROC"))
    write_text(
        analysis_dir / f"{split_token}_CV_Confusion_Matrices.svg",
        confusion_collection_svg([row for row in confusion_rows if row["Split"] == "CV out-of-fold"], "CV out-of-fold confusion matrices"),
    )
    write_text(
        analysis_dir / f"{split_token}_Heldout_Ensemble_Confusion_Matrices.svg",
        confusion_collection_svg([row for row in confusion_rows if row["Split"] == "Held-out ensemble"], "Held-out ensemble confusion matrices"),
    )

    top_rows = []
    top_frequency_rows = []
    for encoder, rows in eval_outputs.items():
        score_col = "Ensemble_Probability"
        predicted_binders = [row for row in rows if float_score(row[score_col]) >= THRESHOLD]
        predicted_nonbinders = [row for row in rows if float_score(row[score_col]) < THRESHOLD]
        logo_sets = [
            ("Evaluation all", [row["Peptide"] for row in rows]),
            ("True binders", [row["Peptide"] for row in rows if int_label(row["True_Class"]) == 1]),
            ("True non-binders", [row["Peptide"] for row in rows if int_label(row["True_Class"]) == 0]),
            (f"Ensemble predicted binders >= {THRESHOLD}", [row["Peptide"] for row in predicted_binders]),
            (f"Ensemble predicted non-binders < {THRESHOLD}", [row["Peptide"] for row in predicted_nonbinders]),
        ]
        for label, lower, upper in TOP_PREDICTION_SLICES:
            slice_rows = prediction_slice(rows, score_col, lower, upper)
            scores = [float_score(row[score_col]) for row in slice_rows]
            y_true = [int_label(row["True_Class"]) for row in slice_rows]
            top_row = {
                "Encoder": encoder,
                "Slice": label,
                "Lower_Quantile": lower,
                "Upper_Quantile": upper,
                "Rows": len(slice_rows),
                "Positives": sum(y_true),
                "Negatives": len(y_true) - sum(y_true),
                "Mean_Probability": sum(scores) / len(scores) if scores else math.nan,
                "Min_Probability": min(scores) if scores else math.nan,
                "Max_Probability": max(scores) if scores else math.nan,
            }
            top_rows.append(top_row)
            set_name = f"{encoder} {label}"
            logo_sets.append((set_name, [row["Peptide"] for row in slice_rows]))
            top_frequency_rows.extend(position_frequency_rows([row["Peptide"] for row in slice_rows], set_name))

        write_text(
            analysis_dir / f"{split_token}_{encoder}_Prediction_Sequence_Logos.svg",
            logo_collection_svg(logo_sets, f"{encoder} prediction sequence logos"),
        )
    slice_order = {label: index for index, (label, _, _) in enumerate(TOP_PREDICTION_SLICES)}
    top_rows = sorted(
        top_rows,
        key=lambda row: (
            ENCODER_ORDER.index(row["Encoder"]) if row["Encoder"] in ENCODER_ORDER else 99,
            slice_order.get(row["Slice"], 99),
        ),
    )
    write_csv(analysis_dir / f"{split_token}_Top_Prediction_Slices.csv", top_rows)
    write_csv(analysis_dir / f"{split_token}_Top_Prediction_Position_Frequencies.csv", top_frequency_rows)

    summary = {
        "split_token": split_token,
        "analysis_dir": str(analysis_dir).replace("\\", "/"),
        "dataset_rows": len(dataset_rows),
        "performance_rows": len(performance_rows),
        "top_rows": len(top_rows),
    }
    write_text(analysis_dir / f"{split_token}_Analysis_Summary.json", json.dumps(summary, indent=2))
    if create_notebook:
        NOTEBOOK_PATH.write_text(json.dumps(build_notebook(summary, performance_rows, dataset_rows, top_rows), indent=1), encoding="utf-8")

    print(f"Split token: {split_token}")
    print(f"Analysis directory: {analysis_dir}")
    print(f"Dataset summary rows: {len(dataset_rows)}")
    print(f"Performance rows: {len(performance_rows)}")
    print(f"Top prediction slice rows: {len(top_rows)}")
    print("Saved sequence-logo SVGs, ROC SVGs, confusion-matrix SVGs, and CSV summaries.")
    return summary


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-token", default=None)
    parser.add_argument("--no-notebook", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_analysis(split_token=args.split_token, create_notebook=not args.no_notebook)
