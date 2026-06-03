"""Prepare a 19-AA held-out split from the IEDB epitope table export.

This workflow prepares canonical epitope positives only. The downstream
training run treats these epitopes as HLA-A03:01 binders and adds same-allele
label-0 peptides from hla_only.txt as non-binders.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence
from xml.etree import ElementTree as ET


STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
STANDARD_AA_SET = set(STANDARD_AA)
PEPTIDE_LENGTH = 9
FOLDS = 5
MIN_TRAIN_ROWS = 1000
MIN_EVAL_ROWS = 100
DEFAULT_HELD_OUT_AA = "W"

WORKBOOK_NAME = "epitope_table_export_1779969300.xlsx"
OUTPUT_DIR = Path("epitope_table_new_data_results")
NOTEBOOK_PATH = Path("Epitope_Table_Heldout_Run.ipynb")
DESCRIPTION_PATH = Path("Epitope_Table_Heldout_File_Description.txt")

NS_MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def latest_workbook() -> Path:
    configured = Path(WORKBOOK_NAME)
    if configured.exists():
        return configured
    candidates = sorted(Path(".").glob("epitope_table*.xlsx"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError("No epitope_table*.xlsx workbook was found in the workspace.")
    return candidates[-1]


def col_letters(cell_ref: str) -> str:
    match = re.match(r"([A-Z]+)", cell_ref)
    if not match:
        raise ValueError(f"Could not parse Excel cell reference: {cell_ref!r}")
    return match.group(1)


def read_shared_strings(zf: zipfile.ZipFile) -> List[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    values = []
    for item in root.findall(f"{NS_MAIN}si"):
        values.append("".join(text.text or "" for text in item.findall(f".//{NS_MAIN}t")))
    return values


def cell_text(cell: ET.Element, shared_strings: Sequence[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(text.text or "" for text in cell.findall(f".//{NS_MAIN}t")).strip()
    value = cell.find(f"{NS_MAIN}v")
    if value is None or value.text is None:
        return ""
    raw_value = value.text.strip()
    if cell_type == "s":
        return shared_strings[int(raw_value)].strip()
    return raw_value


def read_sheet_rows(workbook: Path) -> Dict[int, Dict[str, str]]:
    with zipfile.ZipFile(workbook) as zf:
        shared_strings = read_shared_strings(zf)
        sheet_path = "xl/worksheets/sheet1.xml"
        root = ET.fromstring(zf.read(sheet_path))

    rows: Dict[int, Dict[str, str]] = {}
    for row in root.findall(f".//{NS_MAIN}sheetData/{NS_MAIN}row"):
        row_index = int(row.attrib["r"])
        values: Dict[str, str] = {}
        for cell in row.findall(f"{NS_MAIN}c"):
            ref = cell.attrib.get("r", "")
            values[col_letters(ref)] = cell_text(cell, shared_strings)
        if any(value for value in values.values()):
            rows[row_index] = values
    return rows


def clean_peptide(value: str) -> str:
    return re.sub(r"\s+", "", value.upper().strip())


def split_positions(peptide: str, aa: str) -> str:
    positions = [str(i + 1) for i, residue in enumerate(peptide) if residue == aa]
    return ";".join(positions)


def first_nonempty(values: Iterable[str]) -> str:
    for value in values:
        if value:
            return value
    return ""


def parse_epitope_rows(rows: Dict[int, Dict[str, str]]) -> List[dict]:
    records = []
    for row_index in sorted(index for index in rows if index >= 3):
        row = rows[row_index]
        records.append(
            {
                "Row": row_index,
                "IEDB_IRI": row.get("A", "").strip(),
                "Object_Type": row.get("B", "").strip(),
                "Raw_Name": row.get("C", "").strip(),
                "Peptide": clean_peptide(row.get("C", "")),
                "Modified_Residues": row.get("D", "").strip(),
                "Modifications": row.get("E", "").strip(),
                "Source_Molecule": row.get("J", "").strip(),
                "Molecule_Parent": row.get("L", "").strip(),
                "Source_Organism": row.get("N", "").strip(),
                "Species": row.get("P", "").strip(),
                "Related_Epitope_Relation": row.get("R", "").strip(),
                "Related_Object_Name": row.get("T", "").strip(),
            }
        )
    return records


def filter_records(records: Sequence[dict]) -> tuple[List[dict], Counter, List[str]]:
    retained = []
    exclusions: Counter = Counter()
    bad_examples = []
    canonical_pattern = re.compile(rf"^[{STANDARD_AA}]+$")

    for record in records:
        peptide = record["Peptide"]
        object_type = record["Object_Type"]
        if object_type and object_type != "Linear peptide":
            exclusions["not_linear_peptide"] += 1
            continue
        if not peptide:
            exclusions["missing_epitope_name"] += 1
            continue
        if record["Modified_Residues"] or record["Modifications"]:
            exclusions["modified_or_uncertain_annotation"] += 1
            continue
        if not canonical_pattern.fullmatch(peptide):
            exclusions["noncanonical_sequence_characters"] += 1
            if len(bad_examples) < 10:
                bad_examples.append(record["Raw_Name"])
            continue
        if len(peptide) != PEPTIDE_LENGTH:
            exclusions["not_9mer"] += 1
            continue
        retained.append(record)
    return retained, exclusions, bad_examples


def collapse_unique(records: Sequence[dict]) -> List[dict]:
    grouped: dict[str, dict] = {}
    aggregate_fields = [
        "IEDB_IRI",
        "Object_Type",
        "Source_Molecule",
        "Molecule_Parent",
        "Source_Organism",
        "Species",
        "Related_Epitope_Relation",
        "Related_Object_Name",
    ]

    for record in records:
        peptide = record["Peptide"]
        if peptide not in grouped:
            grouped[peptide] = {
                "Peptide": peptide,
                "IEDB_IRIs": set(),
                "Duplicate_Row_Count": 0,
                "Rows": [],
                **{field: [] for field in aggregate_fields},
            }
        entry = grouped[peptide]
        entry["Duplicate_Row_Count"] += 1
        entry["Rows"].append(record["Row"])
        for field in aggregate_fields:
            value = record[field]
            if value:
                entry[field].append(value)
        if record["IEDB_IRI"]:
            entry["IEDB_IRIs"].add(record["IEDB_IRI"])

    collapsed = []
    for peptide in sorted(grouped):
        entry = grouped[peptide]
        source_orgs = sorted(set(entry["Source_Organism"]))
        species_all = sorted(set(entry["Species"]))
        collapsed.append(
            {
                "Peptide": peptide,
                "IEDB_IRIs": ";".join(sorted(entry["IEDB_IRIs"])),
                "Duplicate_Row_Count": entry["Duplicate_Row_Count"],
                "Contains_W": "yes" if "W" in peptide else "no",
                "W_Positions": split_positions(peptide, "W"),
                "Object_Type": first_nonempty(entry["Object_Type"]),
                "Source_Molecule": first_nonempty(entry["Source_Molecule"]),
                "Molecule_Parent": first_nonempty(entry["Molecule_Parent"]),
                "Source_Organism": first_nonempty(entry["Source_Organism"]),
                "Species": first_nonempty(entry["Species"]),
                "Source_Organisms": ";".join(source_orgs),
                "Species_All": ";".join(species_all),
                "Related_Epitope_Relation": first_nonempty(entry["Related_Epitope_Relation"]),
                "Related_Object_Name": first_nonempty(entry["Related_Object_Name"]),
            }
        )
    return collapsed


def candidate_rows(peptides: Sequence[str]) -> List[dict]:
    rows = []
    total = len(peptides)
    for aa in STANDARD_AA:
        train = [peptide for peptide in peptides if aa not in peptide]
        evaluation = [peptide for peptide in peptides if aa in peptide]
        position_counts = [sum(peptide[pos] == aa for peptide in evaluation) for pos in range(PEPTIDE_LENGTH)]
        anchor_contains = [
            peptide
            for peptide in evaluation
            if peptide[1] == aa or peptide[8] == aa
        ]
        anchor_only = [
            peptide
            for peptide in anchor_contains
            if all((i in (1, 8)) or residue != aa for i, residue in enumerate(peptide))
        ]
        rows.append(
            {
                "Held_Out_AA": aa,
                "Total_Unique_9mers": total,
                "Train_Count_No_AA": len(train),
                "Evaluation_Count_With_AA": len(evaluation),
                "Evaluation_Rate": round(len(evaluation) / total, 6) if total else 0,
                "Anchor_P2_or_P9_Count": len(anchor_contains),
                "Anchor_Only_P2_or_P9_Count": len(anchor_only),
                "Valid_By_Minimums": "yes"
                if len(train) >= MIN_TRAIN_ROWS and len(evaluation) >= MIN_EVAL_ROWS
                else "no",
                **{f"Position_{pos + 1}_Count": position_counts[pos] for pos in range(PEPTIDE_LENGTH)},
            }
        )

    by_train = sorted(rows, key=lambda row: (-row["Train_Count_No_AA"], -row["Evaluation_Count_With_AA"]))
    by_eval = sorted(rows, key=lambda row: (-row["Evaluation_Count_With_AA"], -row["Train_Count_No_AA"]))
    train_rank = {row["Held_Out_AA"]: index + 1 for index, row in enumerate(by_train)}
    eval_rank = {row["Held_Out_AA"]: index + 1 for index, row in enumerate(by_eval)}
    for row in rows:
        row["Rank_By_Train_Count"] = train_rank[row["Held_Out_AA"]]
        row["Rank_By_Evaluation_Count"] = eval_rank[row["Held_Out_AA"]]
    return sorted(rows, key=lambda row: row["Rank_By_Train_Count"])


def fold_for_peptide(peptide: str, index: int) -> int:
    # There are no labels in this file, so the folds are deterministic round-robin splits.
    return index % FOLDS + 1


def split_unique_rows(unique_rows: Sequence[dict], held_out_aa: str) -> tuple[List[dict], List[dict]]:
    train = []
    evaluation = []
    for row in unique_rows:
        peptide = row["Peptide"]
        base = dict(row)
        base["Held_Out_AA"] = held_out_aa
        base["Held_Out_Positions"] = split_positions(peptide, held_out_aa)
        if held_out_aa in peptide:
            evaluation.append(base)
        else:
            train.append(base)

    for index, row in enumerate(sorted(train, key=lambda item: item["Peptide"])):
        row["Fold"] = fold_for_peptide(row["Peptide"], index)
    return sorted(train, key=lambda item: item["Peptide"]), sorted(evaluation, key=lambda item: item["Peptide"])


def position_counts(peptides: Sequence[str]) -> List[Counter]:
    counts = [Counter() for _ in range(PEPTIDE_LENGTH)]
    for peptide in peptides:
        for index, aa in enumerate(peptide):
            counts[index][aa] += 1
    return counts


def frequency_rows(peptides: Sequence[str], dataset_name: str) -> List[dict]:
    counts = position_counts(peptides)
    total = len(peptides)
    rows = []
    for pos_index, counter in enumerate(counts, start=1):
        for aa in STANDARD_AA:
            count = counter.get(aa, 0)
            rows.append(
                {
                    "Dataset": dataset_name,
                    "Position": pos_index,
                    "AA": aa,
                    "Count": count,
                    "Frequency": round(count / total, 8) if total else 0.0,
                }
            )
    return rows


def consensus(peptides: Sequence[str]) -> str:
    if not peptides:
        return ""
    counts = position_counts(peptides)
    return "".join(counter.most_common(1)[0][0] if counter else "X" for counter in counts)


def top_position_table(peptides: Sequence[str], dataset_name: str, top_n: int = 3) -> List[dict]:
    counts = position_counts(peptides)
    total = len(peptides)
    rows = []
    for pos_index, counter in enumerate(counts, start=1):
        for rank, (aa, count) in enumerate(counter.most_common(top_n), start=1):
            rows.append(
                {
                    "Dataset": dataset_name,
                    "Position": pos_index,
                    "Rank": rank,
                    "AA": aa,
                    "Count": count,
                    "Frequency": round(count / total, 6) if total else 0.0,
                }
            )
    return rows


def aa_composition(peptides: Sequence[str], dataset_name: str) -> List[dict]:
    counts = Counter("".join(peptides))
    total = sum(counts.values())
    return [
        {
            "Dataset": dataset_name,
            "AA": aa,
            "Count": counts.get(aa, 0),
            "Frequency": round(counts.get(aa, 0) / total, 8) if total else 0.0,
        }
        for aa in STANDARD_AA
    ]


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


AA_COLORS = {
    "D": "#d73027",
    "E": "#d73027",
    "K": "#4575b4",
    "R": "#4575b4",
    "H": "#4575b4",
    "S": "#1a9850",
    "T": "#1a9850",
    "N": "#1a9850",
    "Q": "#1a9850",
    "C": "#b8a500",
    "G": "#777777",
    "P": "#984ea3",
    "A": "#222222",
    "V": "#222222",
    "I": "#222222",
    "L": "#222222",
    "M": "#222222",
    "F": "#222222",
    "W": "#222222",
    "Y": "#222222",
}


def logo_svg(peptides: Sequence[str], title: str) -> str:
    counts = position_counts(peptides)
    max_bits = math.log2(len(STANDARD_AA))
    chart_height = 230
    chart_width = 9 * 70
    left = 52
    top = 38
    bottom = 44
    right = 24
    height_per_bit = chart_height / max_bits
    width = left + chart_width + right
    height = top + chart_height + bottom
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="22" font-size="16" font-family="Arial, sans-serif" font-weight="700" fill="#222222">{title}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#222222" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}" stroke="#222222" stroke-width="1"/>',
        f'<text x="8" y="{top + 10}" font-size="11" font-family="Arial, sans-serif" fill="#555555">bits</text>',
    ]
    for tick in range(5):
        y = top + chart_height - tick * height_per_bit
        lines.append(f'<line x1="{left - 4}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#555555" stroke-width="1"/>')
        lines.append(f'<text x="{left - 28}" y="{y + 4:.2f}" font-size="10" font-family="Arial, sans-serif" fill="#555555">{tick}</text>')

    for pos_index, counter in enumerate(counts):
        x = left + pos_index * 70 + 11
        column_bottom = top + chart_height
        total = sum(counter.values())
        if total == 0:
            continue
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
            y = column_bottom - block_height
            color = AA_COLORS.get(aa, "#444444")
            lines.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="48" height="{block_height:.2f}" fill="{color}" opacity="0.9"/>'
            )
            if block_height >= 9:
                label_y = y + block_height / 2 + 4
                lines.append(
                    f'<text x="{x + 24:.2f}" y="{label_y:.2f}" text-anchor="middle" font-size="11" '
                    f'font-family="Arial, sans-serif" font-weight="700" fill="#ffffff">{aa}</text>'
                )
            column_bottom = y
        lines.append(
            f'<text x="{x + 24:.2f}" y="{top + chart_height + 24}" text-anchor="middle" font-size="12" '
            f'font-family="Arial, sans-serif" fill="#222222">P{pos_index + 1}</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines)


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def markdown_table(rows: Sequence[dict], columns: Sequence[str], max_rows: int | None = None) -> str:
    visible = list(rows[:max_rows]) if max_rows else list(rows)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in visible:
        body.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join([header, divider, *body])


def notebook_code_cell(source: str, output: str | None = None, execution_count: int | None = None) -> dict:
    outputs = []
    if output is not None:
        outputs.append({"name": "stdout", "output_type": "stream", "text": output})
    return {
        "cell_type": "code",
        "execution_count": execution_count,
        "metadata": {},
        "outputs": outputs,
        "source": source,
    }


def notebook_markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def build_notebook(summary: dict, candidates: Sequence[dict], train_preview: Sequence[dict], eval_preview: Sequence[dict]) -> dict:
    candidate_columns = [
        "Held_Out_AA",
        "Train_Count_No_AA",
        "Evaluation_Count_With_AA",
        "Anchor_P2_or_P9_Count",
        "Anchor_Only_P2_or_P9_Count",
        "Rank_By_Train_Count",
        "Rank_By_Evaluation_Count",
    ]
    split_columns = ["Peptide", "Fold", "Held_Out_AA", "Held_Out_Positions", "Source_Organism", "Species"]
    eval_columns = ["Peptide", "Held_Out_AA", "Held_Out_Positions", "Source_Organism", "Species"]
    run_output = (
        f"Workbook: {summary['workbook']}\n"
        f"Raw epitope rows: {summary['raw_rows']}\n"
        f"Canonical retained rows before dedupe: {summary['retained_rows_before_dedupe']}\n"
        f"Unique canonical 9-mers: {summary['unique_canonical_9mers']}\n"
        f"Default held-out amino acid: {summary['held_out_aa']}\n"
        f"Training/CV peptides without {summary['held_out_aa']}: {summary['train_rows']}\n"
        f"Evaluation peptides with {summary['held_out_aa']}: {summary['evaluation_rows']}\n"
        "Network training is handled by Epitope_A0301_Heldout_Training_Run.ipynb after adding A03:01 non-binders.\n"
    )
    report_rows = [
        {"Metric": "Raw epitope rows", "Value": summary["raw_rows"]},
        {"Metric": "Retained canonical 9-mer rows before dedupe", "Value": summary["retained_rows_before_dedupe"]},
        {"Metric": "Unique canonical 9-mers", "Value": summary["unique_canonical_9mers"]},
        {"Metric": f"Train/CV peptides without {summary['held_out_aa']}", "Value": summary["train_rows"]},
        {"Metric": f"Evaluation peptides with {summary['held_out_aa']}", "Value": summary["evaluation_rows"]},
        {"Metric": "Consensus, all canonical 9-mers", "Value": summary["consensus_all"]},
        {"Metric": f"Consensus, no {summary['held_out_aa']} train/CV", "Value": summary["consensus_train"]},
        {"Metric": f"Consensus, with {summary['held_out_aa']} evaluation", "Value": summary["consensus_evaluation"]},
    ]
    for reason, count in summary["exclusions"].items():
        report_rows.append({"Metric": f"Excluded: {reason}", "Value": count})

    cells = [
        notebook_markdown_cell(
            "# Epitope-Only Held-Out Preparation Workflow\n\n"
            "This notebook prepares the new IEDB epitope table export for the epitope-only held-out workflow. "
            "Rows are kept only when the epitope is a linear, unmodified, canonical 9-mer using the 20 standard amino acids. "
            "Any peptide name with non-canonical characters such as `+`, `X`, `B`, `Z`, `J`, brackets, or other uncertainty/modification notation is excluded. "
            "Rows with populated `Modified Residue(s)` or `Modifications` fields are also excluded before deduplication.\n\n"
            "The downstream training notebook is `Epitope_A0301_Heldout_Training_Run.ipynb`, where these epitopes are treated as HLA-A03:01 binders and same-allele label-0 peptides from `hla_only.txt` are introduced as non-binders."
        ),
        notebook_code_cell(
            "# Recreate all generated CSV/SVG/TXT outputs from the workbook.\n%run epitope_table_heldout_run.py --no-notebook\n",
            output=run_output,
            execution_count=1,
        ),
        notebook_markdown_cell(
            "## Current Saved Summary\n\n" + markdown_table(report_rows, ["Metric", "Value"])
        ),
        notebook_markdown_cell(
            "## Held-Out Amino-Acid Candidates\n\n"
            "The default split is held-out `W` so this epitope-table preparation stays comparable with the earlier HLA W-heldout runs. "
            "The candidate table below also shows that the workbook can support other held-out choices; set `DEFAULT_HELD_OUT_AA` in `epitope_table_heldout_run.py` and rerun if you want a different residue.\n\n"
            + markdown_table(candidates, candidate_columns)
        ),
        notebook_markdown_cell(
            f"## Default {summary['held_out_aa']} Split Preview\n\n"
            f"Training/CV peptides contain no `{summary['held_out_aa']}` anywhere. Evaluation peptides contain `{summary['held_out_aa']}` at one or more positions.\n\n"
            "Train/CV preview:\n\n"
            + markdown_table(train_preview, split_columns, max_rows=8)
            + "\n\nEvaluation preview:\n\n"
            + markdown_table(eval_preview, eval_columns, max_rows=8)
        ),
        notebook_markdown_cell(
            "## Sequence-Logo Style Composition\n\n"
            "The files below are information-content stack plots generated from position-wise amino-acid frequencies. "
            "The underlying frequency tables are saved as CSVs in `epitope_table_new_data_results/`.\n\n"
            "All canonical 9-mers:\n\n"
            "![All canonical 9-mer logo](epitope_table_new_data_results/Epitope_Table_Logo_All.svg)\n\n"
            f"Train/CV peptides with no `{summary['held_out_aa']}`:\n\n"
            f"![Train logo](epitope_table_new_data_results/Epitope_Table_Logo_Train_No_{summary['held_out_aa']}.svg)\n\n"
            f"Evaluation peptides with `{summary['held_out_aa']}`:\n\n"
            f"![Evaluation logo](epitope_table_new_data_results/Epitope_Table_Logo_Evaluation_With_{summary['held_out_aa']}.svg)"
        ),
        notebook_markdown_cell(
            "## Downstream Network Training\n\n"
            "Run `Epitope_A0301_Heldout_Training_Run.ipynb` after this preparation notebook. "
            "That training notebook adds HLA-A03:01 label-0 non-binders from `hla_only.txt`, prepares the held-out-AA split, trains AE/one-hot/BLOSUM62 models, and reports CV plus held-out evaluation summaries."
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def make_description(summary: dict, candidates: Sequence[dict]) -> str:
    candidate_columns = [
        "Held_Out_AA",
        "Train_Count_No_AA",
        "Evaluation_Count_With_AA",
        "Anchor_P2_or_P9_Count",
        "Anchor_Only_P2_or_P9_Count",
        "Rank_By_Train_Count",
        "Rank_By_Evaluation_Count",
    ]
    return (
        "EPITOPE TABLE 19-AA HELD-OUT PREPARATION: FILE DESCRIPTION\n"
        "===========================================================\n\n"
        "Purpose\n"
        "-------\n"
        "This workflow prepares the new IEDB epitope table XLSX as an epitope-only\n"
        "held-out preparation run. It keeps canonical unmodified 9-mer epitopes,\n"
        "then creates a train/CV set with one amino acid absent and an evaluation set\n"
        "where that amino acid is present.\n\n"
        "The epitopes in this workbook are treated as binders for HLA-A03:01. The\n"
        "workbook does not include non-binders, so the supervised training run is\n"
        "handled separately in Epitope_A0301_Heldout_Training_Run.ipynb, where\n"
        "same-allele label-0 peptides from hla_only.txt are introduced as non-binders.\n\n"
        "Input file\n"
        "----------\n"
        f"{summary['workbook']}\n"
        "  IEDB epitope table export. The workflow reads Sheet1 and uses the epitope\n"
        "  fields from the first two header rows. The epitope sequence is the Epitope\n"
        "  Name column.\n\n"
        "Filtering decisions\n"
        "-------------------\n"
        "Rows are retained only if all of the following are true:\n"
        "  Object Type is blank or Linear peptide.\n"
        "  Epitope Name is present.\n"
        "  Modified Residue(s) and Modifications are both empty.\n"
        "  The cleaned epitope sequence contains only ACDEFGHIKLMNPQRSTVWY.\n"
        "  The cleaned epitope sequence is exactly 9 residues long.\n\n"
        "The non-canonical sequence rule removes entries with +, X, B, Z, J,\n"
        "brackets, gaps, or other uncertainty/modification notation in the sequence.\n"
        "Rows that carry modification/uncertainty annotations in the workbook fields\n"
        "are excluded before the sequence regex check, so exclusion counts are not\n"
        "double-counted.\n\n"
        "Current data summary\n"
        "--------------------\n"
        f"Raw epitope rows: {summary['raw_rows']}\n"
        f"Canonical retained rows before dedupe: {summary['retained_rows_before_dedupe']}\n"
        f"Unique canonical 9-mers: {summary['unique_canonical_9mers']}\n"
        + "".join(f"Excluded: {reason}: {count}\n" for reason, count in summary["exclusions"].items())
        + f"\nDefault held-out amino acid: {summary['held_out_aa']}\n"
        f"Training/CV peptides without {summary['held_out_aa']}: {summary['train_rows']}\n"
        f"Evaluation peptides with {summary['held_out_aa']}: {summary['evaluation_rows']}\n"
        f"Consensus, all canonical 9-mers: {summary['consensus_all']}\n"
        f"Consensus, train/CV no {summary['held_out_aa']}: {summary['consensus_train']}\n"
        f"Consensus, evaluation with {summary['held_out_aa']}: {summary['consensus_evaluation']}\n\n"
        "Held-out decision\n"
        "-----------------\n"
        "The default split holds out W for continuity with the earlier held-out-AA\n"
        "experiments. The candidate table is still written for all 20 standard amino\n"
        "acids so a different held-out amino acid can be selected without changing\n"
        "the filtering logic.\n\n"
        + markdown_table(candidates, candidate_columns)
        + "\n\n"
        "Generated files\n"
        "---------------\n"
        "Epitope_Table_Heldout_Run.ipynb\n"
        "  Notebook with visible saved summary, candidate table, split preview, and\n"
        "  sequence-logo style SVGs.\n\n"
        "Epitope_A0301_Heldout_Training_Run.ipynb\n"
        "  Downstream network-training notebook. It treats these canonical epitopes as\n"
        "  HLA-A03:01 binders and adds HLA-A03:01 label-0 peptides from hla_only.txt as\n"
        "  non-binders.\n\n"
        "epitope_table_heldout_run.py\n"
        "  Reproducible stdlib-only runner used by the notebook.\n\n"
        "epitope_table_new_data_results/Epitope_Table_Canonical_9mers.csv\n"
        "  Unique retained canonical 9-mer peptides and aggregated metadata.\n\n"
        "epitope_table_new_data_results/Epitope_Table_Filter_Report.csv\n"
        "  Non-overlapping filtering counts and selected split summary.\n\n"
        "epitope_table_new_data_results/Epitope_Table_Heldout_Candidates.csv\n"
        "  Train/evaluation support for holding out each standard amino acid.\n\n"
        f"epitope_table_new_data_results/Epitope_Table_Heldout_{summary['held_out_aa']}_Train_9mers.csv\n"
        f"  Peptides with no {summary['held_out_aa']}; includes a deterministic 1-5 fold assignment.\n\n"
        f"epitope_table_new_data_results/Epitope_Table_Heldout_{summary['held_out_aa']}_Evaluation_9mers.csv\n"
        f"  Peptides containing {summary['held_out_aa']} and the positions where it appears.\n\n"
        "epitope_table_new_data_results/Epitope_Table_Position_Frequencies_*.csv\n"
        "  Position-wise amino-acid frequencies for all, train, and evaluation sets.\n\n"
        "epitope_table_new_data_results/Epitope_Table_Logo_*.svg\n"
        "  Sequence-logo style information-content plots for all, train, and evaluation sets.\n"
    )


def main(create_notebook: bool = True) -> dict:
    workbook = latest_workbook()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = read_sheet_rows(workbook)
    records = parse_epitope_rows(rows)
    retained, exclusions, bad_examples = filter_records(records)
    unique_rows = collapse_unique(retained)
    peptides = [row["Peptide"] for row in unique_rows]
    candidates = candidate_rows(peptides)

    train_rows, eval_rows = split_unique_rows(unique_rows, DEFAULT_HELD_OUT_AA)
    train_peptides = [row["Peptide"] for row in train_rows]
    eval_peptides = [row["Peptide"] for row in eval_rows]

    unique_fields = [
        "Peptide",
        "IEDB_IRIs",
        "Duplicate_Row_Count",
        "Contains_W",
        "W_Positions",
        "Object_Type",
        "Source_Molecule",
        "Molecule_Parent",
        "Source_Organism",
        "Species",
        "Source_Organisms",
        "Species_All",
        "Related_Epitope_Relation",
        "Related_Object_Name",
    ]
    split_fields = unique_fields + ["Held_Out_AA", "Held_Out_Positions", "Fold"]
    eval_fields = unique_fields + ["Held_Out_AA", "Held_Out_Positions"]

    write_csv(OUTPUT_DIR / "Epitope_Table_Canonical_9mers.csv", unique_rows, unique_fields)
    write_csv(OUTPUT_DIR / "Epitope_Table_Heldout_Candidates.csv", candidates)
    write_csv(OUTPUT_DIR / f"Epitope_Table_Heldout_{DEFAULT_HELD_OUT_AA}_Train_9mers.csv", train_rows, split_fields)
    write_csv(OUTPUT_DIR / f"Epitope_Table_Heldout_{DEFAULT_HELD_OUT_AA}_Evaluation_9mers.csv", eval_rows, eval_fields)

    all_frequency = frequency_rows(peptides, "all_canonical_9mers")
    train_frequency = frequency_rows(train_peptides, f"train_no_{DEFAULT_HELD_OUT_AA}")
    eval_frequency = frequency_rows(eval_peptides, f"evaluation_with_{DEFAULT_HELD_OUT_AA}")
    write_csv(OUTPUT_DIR / "Epitope_Table_Position_Frequencies_All.csv", all_frequency)
    write_csv(OUTPUT_DIR / f"Epitope_Table_Position_Frequencies_Train_No_{DEFAULT_HELD_OUT_AA}.csv", train_frequency)
    write_csv(OUTPUT_DIR / f"Epitope_Table_Position_Frequencies_Evaluation_With_{DEFAULT_HELD_OUT_AA}.csv", eval_frequency)
    write_csv(
        OUTPUT_DIR / "Epitope_Table_Top_Position_Residues.csv",
        top_position_table(peptides, "all_canonical_9mers")
        + top_position_table(train_peptides, f"train_no_{DEFAULT_HELD_OUT_AA}")
        + top_position_table(eval_peptides, f"evaluation_with_{DEFAULT_HELD_OUT_AA}"),
    )
    write_csv(
        OUTPUT_DIR / "Epitope_Table_Amino_Acid_Composition.csv",
        aa_composition(peptides, "all_canonical_9mers")
        + aa_composition(train_peptides, f"train_no_{DEFAULT_HELD_OUT_AA}")
        + aa_composition(eval_peptides, f"evaluation_with_{DEFAULT_HELD_OUT_AA}"),
    )

    write_text(
        OUTPUT_DIR / "Epitope_Table_Logo_All.svg",
        logo_svg(peptides, f"All canonical 9-mers, n={len(peptides)}"),
    )
    write_text(
        OUTPUT_DIR / f"Epitope_Table_Logo_Train_No_{DEFAULT_HELD_OUT_AA}.svg",
        logo_svg(train_peptides, f"Train/CV without {DEFAULT_HELD_OUT_AA}, n={len(train_peptides)}"),
    )
    write_text(
        OUTPUT_DIR / f"Epitope_Table_Logo_Evaluation_With_{DEFAULT_HELD_OUT_AA}.svg",
        logo_svg(eval_peptides, f"Evaluation with {DEFAULT_HELD_OUT_AA}, n={len(eval_peptides)}"),
    )

    summary = {
        "workbook": workbook.name,
        "raw_rows": len(records),
        "retained_rows_before_dedupe": len(retained),
        "unique_canonical_9mers": len(unique_rows),
        "duplicate_rows_removed": len(retained) - len(unique_rows),
        "held_out_aa": DEFAULT_HELD_OUT_AA,
        "train_rows": len(train_rows),
        "evaluation_rows": len(eval_rows),
        "consensus_all": consensus(peptides),
        "consensus_train": consensus(train_peptides),
        "consensus_evaluation": consensus(eval_peptides),
        "exclusions": dict(sorted(exclusions.items())),
        "bad_sequence_examples_after_annotation_filter": bad_examples,
    }

    report_rows = [
        {"Metric": "Workbook", "Value": summary["workbook"]},
        {"Metric": "Raw epitope rows", "Value": summary["raw_rows"]},
        {"Metric": "Canonical retained rows before dedupe", "Value": summary["retained_rows_before_dedupe"]},
        {"Metric": "Canonical unique 9-mers", "Value": summary["unique_canonical_9mers"]},
        {"Metric": "Duplicate retained rows collapsed", "Value": summary["duplicate_rows_removed"]},
        {"Metric": "Default held-out amino acid", "Value": summary["held_out_aa"]},
        {"Metric": f"Train/CV peptides without {DEFAULT_HELD_OUT_AA}", "Value": summary["train_rows"]},
        {"Metric": f"Evaluation peptides with {DEFAULT_HELD_OUT_AA}", "Value": summary["evaluation_rows"]},
        {"Metric": "Consensus, all canonical 9-mers", "Value": summary["consensus_all"]},
        {"Metric": f"Consensus, train no {DEFAULT_HELD_OUT_AA}", "Value": summary["consensus_train"]},
        {"Metric": f"Consensus, evaluation with {DEFAULT_HELD_OUT_AA}", "Value": summary["consensus_evaluation"]},
    ]
    report_rows.extend({"Metric": f"Excluded: {reason}", "Value": count} for reason, count in summary["exclusions"].items())
    if bad_examples:
        report_rows.append({"Metric": "Noncanonical examples after annotation filter", "Value": "; ".join(bad_examples)})
    else:
        report_rows.append({"Metric": "Noncanonical examples after annotation filter", "Value": "none"})
    write_csv(OUTPUT_DIR / "Epitope_Table_Filter_Report.csv", report_rows, ["Metric", "Value"])

    write_text(DESCRIPTION_PATH, make_description(summary, candidates))
    if create_notebook:
        notebook = build_notebook(summary, candidates, train_rows, eval_rows)
        NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1), encoding="utf-8")

    print(f"Workbook: {summary['workbook']}")
    print(f"Raw epitope rows: {summary['raw_rows']}")
    print(f"Canonical retained rows before dedupe: {summary['retained_rows_before_dedupe']}")
    print(f"Unique canonical 9-mers: {summary['unique_canonical_9mers']}")
    print(f"Default held-out amino acid: {DEFAULT_HELD_OUT_AA}")
    print(f"Training/CV peptides without {DEFAULT_HELD_OUT_AA}: {len(train_rows)}")
    print(f"Evaluation peptides with {DEFAULT_HELD_OUT_AA}: {len(eval_rows)}")
    print("Network training is handled by Epitope_A0301_Heldout_Training_Run.ipynb after adding A03:01 non-binders.")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-notebook", action="store_true", help="Do not regenerate the notebook file.")
    args = parser.parse_args()
    main(create_notebook=not args.no_notebook)
