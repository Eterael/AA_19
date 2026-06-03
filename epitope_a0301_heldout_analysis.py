"""Anchor-style analysis for the epitope A03:01 held-out run.

This is a script export of Epitope_A0301_Heldout_Analysis.ipynb. It requires numpy, pandas, matplotlib, and IPython display, matching HLA_19AA_Anchor_Heldout_Analysis.ipynb.
"""
from pathlib import Path
from collections import Counter, defaultdict
import math

try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display
except ImportError as exc:
    raise ImportError(
        "This analysis notebook needs numpy, pandas, matplotlib, and IPython display. "
        "Run it in the same Python/Jupyter environment used for training."
    ) from exc

from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D

PROJECT_DIR = Path(".")
HLA_FILE = PROJECT_DIR / "hla_only.txt"
OUTPUT_DIR = PROJECT_DIR / "epitope_a0301_heldout_results"

# Leave SPLIT_TOKEN as None to use the newest prepared split in OUTPUT_DIR.
# Example: SPLIT_TOKEN = "HLA-A03_01_epitope_bind_pos_hla_neg_holdout_R"
SPLIT_TOKEN = None

THRESHOLD = 0.5
STANDARD_AA = tuple("ACDEFGHIKLMNPQRSTVWY")
STANDARD_AA_SET = set(STANDARD_AA)

# Includes both interpretations: strict top 1%, the next 1-10% band, and top 10% overall.
TOP_PREDICTION_SLICES = [
    ("Top 1%", 0.00, 0.01),
    ("Top 1-10%", 0.01, 0.10),
    ("Top 10%", 0.00, 0.10),
]


def infer_split_token(output_dir, split_token=None):
    output_dir = Path(output_dir)
    if split_token:
        return split_token
    prefix = "Epitope_A0301_19AA_"
    suffix = "_TrainTest_9mers_Labeled.csv"
    candidates = sorted(
        output_dir.glob(f"{prefix}*{suffix}"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No prepared split files found in {output_dir}. Run Epitope_A0301_Heldout_Training_Run.ipynb through the prepare-split cell first."
        )
    print("Available prepared splits:")
    for path in candidates:
        token = path.name[len(prefix):-len(suffix)]
        print(f"  {token}")
    selected_token = candidates[0].name[len(prefix):-len(suffix)]
    print(f"Using newest split: {selected_token}")
    return selected_token


def load_selected_split(output_dir, split_token):
    output_dir = Path(output_dir)
    train_path = output_dir / f"Epitope_A0301_19AA_{split_token}_TrainTest_9mers_Labeled.csv"
    eval_path = output_dir / f"Epitope_A0301_19AA_{split_token}_Heldout_Evaluation_9mers_Labeled.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train/test split file: {train_path}")
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing held-out evaluation split file: {eval_path}")
    train_df = pd.read_csv(train_path)
    eval_df = pd.read_csv(eval_path)
    train_df = train_df.rename(columns={"Label": "True_Class"})
    eval_df = eval_df.rename(columns={"Label": "True_Class"})
    train_df["Split"] = "Train/test"
    eval_df["Split"] = "Held-out evaluation"
    if "Fold" not in eval_df.columns:
        eval_df["Fold"] = np.nan
    return train_df, eval_df, train_path, eval_path


def parse_held_out_aa(split_token):
    marker = "_holdout_"
    if marker in split_token:
        return split_token.rsplit(marker, 1)[1].split("_", 1)[0]
    return None


SPLIT_TOKEN = infer_split_token(OUTPUT_DIR, SPLIT_TOKEN)
ANALYSIS_DIR = OUTPUT_DIR / f"analysis_{SPLIT_TOKEN}"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

train_df, evaluation_df, train_path, eval_path = load_selected_split(OUTPUT_DIR, SPLIT_TOKEN)
selected_df = pd.concat([train_df, evaluation_df], ignore_index=True)
ALLELE = selected_df["Allele"].iloc[0] if "Allele" in selected_df.columns and len(selected_df) else None
HELD_OUT_AA = parse_held_out_aa(SPLIT_TOKEN)

print(f"Loaded train/test split: {train_path}")
print(f"Loaded held-out evaluation split: {eval_path}")
print(f"Analysis output directory: {ANALYSIS_DIR}")
print(f"Allele: {ALLELE} | held-out amino acid: {HELD_OUT_AA}")


AA_COLORS = {
    "A": "#4C78A8", "V": "#4C78A8", "L": "#4C78A8", "I": "#4C78A8", "M": "#4C78A8",
    "F": "#4C78A8", "W": "#4C78A8", "Y": "#4C78A8", "P": "#4C78A8",
    "S": "#59A14F", "T": "#59A14F", "N": "#59A14F", "Q": "#59A14F", "C": "#59A14F",
    "D": "#E15759", "E": "#E15759",
    "K": "#B07AA1", "R": "#B07AA1", "H": "#B07AA1",
    "G": "#9C755F",
}
LOGO_FONT = FontProperties(family="DejaVu Sans", weight="bold")


def read_hla_only_summary(path):
    rows = []
    skipped = Counter()
    if not Path(path).exists():
        print(f"Raw HLA file not found: {path}")
        return pd.DataFrame(), skipped
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 3:
                skipped["malformed"] += 1
                continue
            peptide, label_text, allele = parts[:3]
            try:
                label = int(label_text)
            except ValueError:
                skipped["bad_label"] += 1
                continue
            if len(peptide) != 9 or any(aa not in STANDARD_AA_SET for aa in peptide):
                skipped["not_standard_9mer"] += 1
                continue
            rows.append({"Peptide": peptide, "True_Class": label, "Allele": allele})
    return pd.DataFrame(rows).drop_duplicates(["Allele", "Peptide"]), skipped


def dataset_summary_row(name, df):
    positives = int((df["True_Class"] == 1).sum()) if "True_Class" in df.columns else 0
    negatives = int((df["True_Class"] == 0).sum()) if "True_Class" in df.columns else 0
    return {
        "Set": name,
        "Rows": int(len(df)),
        "Unique_Peptides": int(df["Peptide"].nunique()) if "Peptide" in df.columns else 0,
        "Positives": positives,
        "Negatives": negatives,
        "Positive_Rate": positives / len(df) if len(df) else np.nan,
    }


def amino_acid_composition(peptides, name):
    counts = {aa: 0 for aa in STANDARD_AA}
    total = 0
    for peptide in peptides:
        for aa in str(peptide):
            if aa in counts:
                counts[aa] += 1
                total += 1
    return pd.DataFrame([
        {"Set": name, "AA": aa, "Count": counts[aa], "Frequency": counts[aa] / total if total else 0.0}
        for aa in STANDARD_AA
    ])


def position_count_df(peptides):
    counts = pd.DataFrame(0.0, index=range(1, 10), columns=list(STANDARD_AA))
    for peptide in peptides:
        peptide = str(peptide)
        if len(peptide) != 9:
            continue
        for pos, aa in enumerate(peptide, start=1):
            if aa in counts.columns:
                counts.loc[pos, aa] += 1
    return counts


def position_frequency_df(peptides):
    counts = position_count_df(peptides)
    totals = counts.sum(axis=1).replace(0, np.nan)
    return counts.div(totals, axis=0).fillna(0.0)


def logo_height_df(peptides, mode="information"):
    freq = position_frequency_df(peptides)
    if mode == "probability":
        return freq
    nonzero = freq.replace(0.0, np.nan)
    entropy = -(nonzero * np.log2(nonzero)).sum(axis=1).fillna(0.0)
    information = np.log2(len(STANDARD_AA)) - entropy
    return freq.mul(information, axis=0)


def _draw_logo_letter(ax, letter, x, y, height, color):
    if height <= 0:
        return
    path = TextPath((0, 0), letter, size=1, prop=LOGO_FONT)
    bbox = path.get_extents()
    if bbox.width == 0 or bbox.height == 0:
        return
    width = 0.82
    transform = (
        Affine2D()
        .translate(-bbox.x0, -bbox.y0)
        .scale(width / bbox.width, height / bbox.height)
        .translate(x + (1.0 - width) / 2.0, y)
    )
    ax.add_patch(PathPatch(path, lw=0, color=color, transform=transform + ax.transData))


def plot_sequence_logo(peptides, title, ax=None, mode="information", min_height=0.015):
    peptides = [str(peptide) for peptide in peptides if isinstance(peptide, str) and len(str(peptide)) == 9]
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3))
    if not peptides:
        ax.text(0.5, 0.5, "No peptides", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        ax.set_title(title)
        return ax
    heights = logo_height_df(peptides, mode=mode)
    for pos_idx, pos in enumerate(heights.index):
        y_base = 0.0
        for aa, height in heights.loc[pos].sort_values().items():
            height = float(height)
            if height >= min_height:
                _draw_logo_letter(ax, aa, pos_idx, y_base, height, AA_COLORS.get(aa, "#333333"))
            y_base += height
    if mode == "information":
        ymax = max(float(heights.sum(axis=1).max()) * 1.12, 1.0)
        ylabel = "Information (bits)"
    else:
        ymax = 1.05
        ylabel = "Frequency"
    ax.set_xlim(0, 9)
    ax.set_ylim(0, ymax)
    ax.set_xticks(np.arange(9) + 0.5)
    ax.set_xticklabels(range(1, 10))
    ax.set_xlabel("Position")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} (n={len(peptides)})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def plot_logo_collection(logo_sets, title, output_path=None, mode="information", ncols=2):
    logo_sets = [(name, list(peptides)) for name, peptides in logo_sets]
    nrows = int(np.ceil(len(logo_sets) / ncols)) if logo_sets else 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(9 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, peptides) in zip(axes, logo_sets):
        plot_sequence_logo(peptides, name, ax=ax, mode=mode)
    for ax in axes[len(logo_sets):]:
        ax.set_axis_off()
    fig.suptitle(title, y=1.01, fontsize=14)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure: {output_path}")
    return fig


def rank_auc(y_true, y_score):
    y_true = [int(value) for value in y_true]
    y_score = [float(value) for value in y_score]
    positives = sum(1 for value in y_true if value == 1)
    negatives = sum(1 for value in y_true if value == 0)
    if positives == 0 or negatives == 0:
        return np.nan
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


def binary_confusion(y_true, y_score, threshold=THRESHOLD):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = (np.asarray(y_score, dtype=float) >= threshold).astype(int)
    return {
        "TN": int(((y_true == 0) & (y_pred == 0)).sum()),
        "FP": int(((y_true == 0) & (y_pred == 1)).sum()),
        "FN": int(((y_true == 1) & (y_pred == 0)).sum()),
        "TP": int(((y_true == 1) & (y_pred == 1)).sum()),
    }


def performance_row(encoder, split_name, df, score_col, threshold=THRESHOLD):
    y_true = df["True_Class"].astype(int).tolist()
    y_score = df[score_col].astype(float).tolist()
    cm = binary_confusion(y_true, y_score, threshold=threshold)
    tp, tn, fp, fn = cm["TP"], cm["TN"], cm["FP"], cm["FN"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "Encoder": encoder,
        "Split": split_name,
        "Rows": int(len(df)),
        "Positives": int(sum(y_true)),
        "Negatives": int(len(y_true) - sum(y_true)),
        "AUC": rank_auc(y_true, y_score),
        "Accuracy": (tp + tn) / len(df) if len(df) else np.nan,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1": f1,
        **cm,
    }


def roc_points(y_true, y_score):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    positives = y_true.sum()
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return np.array([0, 1]), np.array([0, 1])
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    score_sorted = y_score[order]
    tpr = [0.0]
    fpr = [0.0]
    tp = 0
    fp = 0
    last_score = None
    for label, score in zip(y_sorted, score_sorted):
        if last_score is not None and score != last_score:
            tpr.append(tp / positives)
            fpr.append(fp / negatives)
        if label == 1:
            tp += 1
        else:
            fp += 1
        last_score = score
    tpr.append(tp / positives)
    fpr.append(fp / negatives)
    return np.asarray(fpr), np.asarray(tpr)


def plot_confusion_matrix(ax, cm, title):
    matrix = np.array([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]])
    ax.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=12, color="black")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["True 0", "True 1"])
    ax.set_title(title)


def plot_performance_panels(outputs, split_name, score_col, output_prefix):
    if not outputs:
        print(f"No {split_name} prediction files found yet.")
        return None
    n = len(outputs)
    fig_roc, axes_roc = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    fig_cm, axes_cm = plt.subplots(1, n, figsize=(4.2 * n, 4), squeeze=False)
    for ax_roc, ax_cm, (encoder, df) in zip(axes_roc.ravel(), axes_cm.ravel(), outputs.items()):
        y_true = df["True_Class"].astype(int).tolist()
        y_score = df[score_col].astype(float).tolist()
        auc_value = rank_auc(y_true, y_score)
        fpr, tpr = roc_points(y_true, y_score)
        ax_roc.plot(fpr, tpr, lw=2)
        ax_roc.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
        ax_roc.set_xlim(0, 1)
        ax_roc.set_ylim(0, 1)
        ax_roc.set_xlabel("False positive rate")
        ax_roc.set_ylabel("True positive rate")
        ax_roc.set_title(f"{encoder} {split_name}\nAUC={auc_value:.3f}")
        cm = binary_confusion(y_true, y_score, threshold=THRESHOLD)
        plot_confusion_matrix(ax_cm, cm, f"{encoder} {split_name}\nthreshold={THRESHOLD}")
    fig_roc.tight_layout()
    fig_cm.tight_layout()
    roc_path = ANALYSIS_DIR / f"{output_prefix}_ROC.png"
    cm_path = ANALYSIS_DIR / f"{output_prefix}_Confusion_Matrices.png"
    fig_roc.savefig(roc_path, dpi=200, bbox_inches="tight")
    fig_cm.savefig(cm_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure: {roc_path}")
    print(f"Saved figure: {cm_path}")
    return fig_roc, fig_cm


def load_prediction_outputs(output_dir, split_token):
    output_dir = Path(output_dir)
    cv_outputs = {}
    eval_outputs = {}
    for path in sorted(output_dir.glob(f"Epitope_A0301_19AA_CV_Binding_Predictions_{split_token}__*.csv")):
        encoder = path.stem.split("__")[-1]
        cv_outputs[encoder] = pd.read_csv(path)
    for path in sorted(output_dir.glob(f"Epitope_A0301_19AA_Heldout_Evaluation_{split_token}__*.csv")):
        encoder = path.stem.split("__")[-1]
        eval_outputs[encoder] = pd.read_csv(path)
    return cv_outputs, eval_outputs


def prediction_slice(df, score_col, lower_fraction, upper_fraction):
    ranked = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    n = len(ranked)
    start = int(np.floor(n * lower_fraction))
    stop = int(np.ceil(n * upper_fraction))
    if n and stop <= start:
        stop = start + 1
    return ranked.iloc[start:stop].copy()


source_summary_path = OUTPUT_DIR / "Epitope_A0301_19AA_Source_Summary.csv"
source_summary_df = pd.read_csv(source_summary_path) if source_summary_path.exists() else pd.DataFrame()

summary_df = pd.DataFrame([
    dataset_summary_row("All labeled peptides", selected_df),
    dataset_summary_row("Train/test all", train_df),
    dataset_summary_row("Train/test binders", train_df[train_df["True_Class"] == 1]),
    dataset_summary_row("Train/test non-binders", train_df[train_df["True_Class"] == 0]),
    dataset_summary_row("Held-out evaluation all", evaluation_df),
    dataset_summary_row("Held-out evaluation binders", evaluation_df[evaluation_df["True_Class"] == 1]),
    dataset_summary_row("Held-out evaluation non-binders", evaluation_df[evaluation_df["True_Class"] == 0]),
])

fold_summary_df = (
    train_df.groupby("Fold")
    .agg(Rows=("Peptide", "count"), Positives=("True_Class", "sum"))
    .reset_index()
)
fold_summary_df["Negatives"] = fold_summary_df["Rows"] - fold_summary_df["Positives"]

source_group_summary_df = (
    selected_df.groupby(["Split", "Source_Group"])
    .agg(Rows=("Peptide", "count"), Unique_Peptides=("Peptide", "nunique"))
    .reset_index()
    if "Source_Group" in selected_df.columns
    else pd.DataFrame()
)

candidate_path = OUTPUT_DIR / "Epitope_A0301_19AA_Split_Candidates.csv"
candidate_df = pd.read_csv(candidate_path) if candidate_path.exists() else pd.DataFrame()

summary_path = ANALYSIS_DIR / f"{SPLIT_TOKEN}_Dataset_Summary.csv"
fold_summary_path = ANALYSIS_DIR / f"{SPLIT_TOKEN}_Fold_Summary.csv"
source_group_path = ANALYSIS_DIR / f"{SPLIT_TOKEN}_Source_Group_Summary.csv"
summary_df.to_csv(summary_path, index=False)
fold_summary_df.to_csv(fold_summary_path, index=False)
if len(source_group_summary_df):
    source_group_summary_df.to_csv(source_group_path, index=False)
print(f"Saved dataset summary: {summary_path}")
print(f"Saved fold summary: {fold_summary_path}")
if len(source_group_summary_df):
    print(f"Saved source group summary: {source_group_path}")

print("Selected split summary")
display(summary_df)
print("Fold summary")
display(fold_summary_df)
if len(source_summary_df):
    print("Negative source summary")
    display(source_summary_df)
if len(source_group_summary_df):
    print("Source groups by split")
    display(source_group_summary_df)
if len(candidate_df):
    print("Held-out split candidates for the current negative selection")
    display(candidate_df.head(20))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
summary_df.set_index("Set")[["Positives", "Negatives"]].plot(kind="bar", stacked=True, ax=axes[0], color=["#4C78A8", "#E15759"])
axes[0].set_title("Label composition")
axes[0].set_ylabel("Peptides")
axes[0].tick_params(axis="x", rotation=45)
fold_summary_df.set_index("Fold")[["Positives", "Negatives"]].plot(kind="bar", stacked=True, ax=axes[1], color=["#4C78A8", "#E15759"])
axes[1].set_title("Five-fold train/test label balance")
axes[1].set_ylabel("Peptides")
fig.tight_layout()
fig_path = ANALYSIS_DIR / f"{SPLIT_TOKEN}_Dataset_Label_Balance.png"
fig.savefig(fig_path, dpi=200, bbox_inches="tight")
print(f"Saved figure: {fig_path}")

if len(source_group_summary_df):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    source_pivot = source_group_summary_df.pivot(index="Split", columns="Source_Group", values="Rows").fillna(0)
    source_pivot.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Peptide source groups")
    ax.set_ylabel("Peptides")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    source_fig_path = ANALYSIS_DIR / f"{SPLIT_TOKEN}_Source_Group_Balance.png"
    fig.savefig(source_fig_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure: {source_fig_path}")

composition_sets = [
    amino_acid_composition(selected_df["Peptide"], "All labeled peptides"),
    amino_acid_composition(train_df["Peptide"], "Train/test all"),
    amino_acid_composition(train_df.loc[train_df["True_Class"] == 1, "Peptide"], "Train/test binders"),
    amino_acid_composition(train_df.loc[train_df["True_Class"] == 0, "Peptide"], "Train/test non-binders"),
    amino_acid_composition(evaluation_df["Peptide"], "Held-out evaluation all"),
    amino_acid_composition(evaluation_df.loc[evaluation_df["True_Class"] == 1, "Peptide"], "Held-out evaluation binders"),
    amino_acid_composition(evaluation_df.loc[evaluation_df["True_Class"] == 0, "Peptide"], "Held-out evaluation non-binders"),
]
composition_df = pd.concat(composition_sets, ignore_index=True)
composition_path = ANALYSIS_DIR / f"{SPLIT_TOKEN}_Amino_Acid_Composition.csv"
composition_df.to_csv(composition_path, index=False)
print(f"Saved amino-acid composition table: {composition_path}")

display(composition_df.pivot(index="AA", columns="Set", values="Frequency"))


logo_sets = [
    ("All labeled peptides", selected_df["Peptide"]),
    ("Train/test all", train_df["Peptide"]),
    ("Train/test binders", train_df.loc[train_df["True_Class"] == 1, "Peptide"]),
    ("Train/test non-binders", train_df.loc[train_df["True_Class"] == 0, "Peptide"]),
    ("Held-out evaluation all", evaluation_df["Peptide"]),
    ("Held-out evaluation binders", evaluation_df.loc[evaluation_df["True_Class"] == 1, "Peptide"]),
    ("Held-out evaluation non-binders", evaluation_df.loc[evaluation_df["True_Class"] == 0, "Peptide"]),
]
plot_logo_collection(
    logo_sets,
    "Sequence logos: dataset, binders, non-binders, and held-out evaluation set",
    output_path=ANALYSIS_DIR / f"{SPLIT_TOKEN}_Dataset_Sequence_Logos.png",
)

frequency_rows = []
for set_name, peptides in logo_sets:
    freq = position_frequency_df(peptides).reset_index(names="Position")
    freq.insert(0, "Set", set_name)
    frequency_rows.append(freq)
position_frequency_df_all = pd.concat(frequency_rows, ignore_index=True)
position_frequency_path = ANALYSIS_DIR / f"{SPLIT_TOKEN}_Position_Frequencies_For_Logos.csv"
position_frequency_df_all.to_csv(position_frequency_path, index=False)
print(f"Saved position-frequency table for logos: {position_frequency_path}")


cv_outputs, eval_outputs = load_prediction_outputs(OUTPUT_DIR, SPLIT_TOKEN)
print(f"Found CV prediction files for encoders: {list(cv_outputs)}")
print(f"Found held-out ensemble prediction files for encoders: {list(eval_outputs)}")

if not cv_outputs and not eval_outputs:
    print("No prediction CSVs found yet. Run the training notebook first, then re-run this cell.")
else:
    performance_rows = []
    fold_rows = []
    for encoder, df in cv_outputs.items():
        performance_rows.append(performance_row(encoder, "CV out-of-fold", df, "Probability"))
        for fold, fold_df in df.groupby("Fold"):
            row = performance_row(encoder, f"CV fold {fold}", fold_df, "Probability")
            row["Fold"] = fold
            fold_rows.append(row)
    for encoder, df in eval_outputs.items():
        performance_rows.append(performance_row(encoder, "Held-out ensemble", df, "Ensemble_Probability"))

    performance_df = pd.DataFrame(performance_rows).sort_values(["Split", "AUC"], ascending=[True, False])
    performance_path = ANALYSIS_DIR / f"{SPLIT_TOKEN}_Detailed_Performance.csv"
    performance_df.to_csv(performance_path, index=False)
    print(f"Saved detailed performance table: {performance_path}")
    display(performance_df)

    if fold_rows:
        fold_performance_df = pd.DataFrame(fold_rows).sort_values(["Encoder", "Fold"])
        fold_performance_path = ANALYSIS_DIR / f"{SPLIT_TOKEN}_CV_Fold_Performance.csv"
        fold_performance_df.to_csv(fold_performance_path, index=False)
        print(f"Saved fold-level performance table: {fold_performance_path}")
        display(fold_performance_df)


if "cv_outputs" not in globals() or "eval_outputs" not in globals():
    cv_outputs, eval_outputs = load_prediction_outputs(OUTPUT_DIR, SPLIT_TOKEN)

plot_performance_panels(cv_outputs, "CV out-of-fold", "Probability", f"{SPLIT_TOKEN}_CV")
plot_performance_panels(eval_outputs, "Held-out ensemble", "Ensemble_Probability", f"{SPLIT_TOKEN}_Heldout_Ensemble")


if "eval_outputs" not in globals():
    _, eval_outputs = load_prediction_outputs(OUTPUT_DIR, SPLIT_TOKEN)

if not eval_outputs:
    print("No held-out ensemble prediction CSVs found yet. Run the training notebook first, then re-run this cell.")
else:
    all_slice_rows = []
    for encoder, pred_df in eval_outputs.items():
        score_col = "Ensemble_Probability"
        pred_df = pred_df.sort_values(score_col, ascending=False).reset_index(drop=True)
        predicted_binders = pred_df[pred_df[score_col] >= THRESHOLD]
        predicted_nonbinders = pred_df[pred_df[score_col] < THRESHOLD]

        logo_sets = [
            ("Evaluation all", pred_df["Peptide"]),
            ("True binders", pred_df.loc[pred_df["True_Class"] == 1, "Peptide"]),
            ("True non-binders", pred_df.loc[pred_df["True_Class"] == 0, "Peptide"]),
            (f"Ensemble predicted binders >= {THRESHOLD}", predicted_binders["Peptide"]),
            (f"Ensemble predicted non-binders < {THRESHOLD}", predicted_nonbinders["Peptide"]),
        ]

        for label, lower, upper in TOP_PREDICTION_SLICES:
            slice_df = prediction_slice(pred_df, score_col, lower, upper)
            logo_sets.append((label, slice_df["Peptide"]))
            all_slice_rows.append({
                "Encoder": encoder,
                "Slice": label,
                "Rank_Start_Percent": lower * 100,
                "Rank_End_Percent": upper * 100,
                "Rows": int(len(slice_df)),
                "Positives": int((slice_df["True_Class"] == 1).sum()),
                "Negatives": int((slice_df["True_Class"] == 0).sum()),
                "Positive_Rate": float((slice_df["True_Class"] == 1).mean()) if len(slice_df) else np.nan,
                "Mean_Ensemble_Probability": float(slice_df[score_col].mean()) if len(slice_df) else np.nan,
                "Min_Ensemble_Probability": float(slice_df[score_col].min()) if len(slice_df) else np.nan,
                "Max_Ensemble_Probability": float(slice_df[score_col].max()) if len(slice_df) else np.nan,
            })

        print(f"{encoder}: top 20 held-out evaluation peptides by ensemble probability")
        display(pred_df[["Peptide", "True_Class", score_col]].head(20))
        safe_encoder = encoder.replace("/", "_")
        plot_logo_collection(
            logo_sets,
            f"{encoder}: held-out ensemble and top prediction sequence logos",
            output_path=ANALYSIS_DIR / f"{SPLIT_TOKEN}_{safe_encoder}_Ensemble_And_Top_Prediction_Logos.png",
        )

    top_slice_df = pd.DataFrame(all_slice_rows)
    top_slice_path = ANALYSIS_DIR / f"{SPLIT_TOKEN}_Top_Prediction_Slices.csv"
    top_slice_df.to_csv(top_slice_path, index=False)
    print(f"Saved top prediction slice summary: {top_slice_path}")
    display(top_slice_df)


