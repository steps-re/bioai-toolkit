#!/usr/bin/env python3
"""
process_datasets.py — Process raw downloaded datasets into clean CSVs
for the Bio-AI Toolkit Streamlit app.

Inputs:  data/merops/, data/peplife/, data/dppiv/
Outputs: data/processed/*.csv
"""

import json
import os
import re
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
OUT = os.path.join(DATA, "processed")
os.makedirs(OUT, exist_ok=True)

# --- Exopeptidase family codes ---
EXOPEPTIDASE_FAMILIES = {
    "M01", "M14", "M17", "M18", "M24", "M28", "M32",
    "S09", "S10", "S28",
    "C01",  # partial — cathepsin H has aminopeptidase activity
}


def strip_quotes(val):
    """Strip surrounding single quotes and handle NULL."""
    if val is None or val == "NULL":
        return None
    s = str(val).strip()
    if s.startswith("'") and s.endswith("'"):
        return s[1:-1]
    return s


def extract_family(merops_code):
    """Extract family code from MEROPS code like 'A01.001' -> 'A01'."""
    if not merops_code:
        return None
    m = re.match(r"^([A-Z]\d+)", merops_code)
    return m.group(1) if m else None


# ============================================================
# A. MEROPS exopeptidase cleavages
# ============================================================
def process_merops():
    print("=" * 60)
    print("Processing MEROPS Substrate_search.txt ...")
    path = os.path.join(DATA, "merops", "Substrate_search.txt")

    # Column mapping by position (0-indexed), based on inspection:
    # 0=cleavage_id, 1=merops_code, 2=substrate_name, 3=cleavage_notation,
    # 4=P4, 5=P3, 6=P2, 7=P1, 8=P1prime, 9=P2prime, 10=P3prime, 11=P4prime,
    # 12=reference, 13=uniprot, 14=position, 15=organism, 16=protease_name,
    # 17-21=unknown, 22=cleavage_type
    col_names = [
        "cleavage_id", "merops_code", "substrate_name", "cleavage_notation",
        "P4", "P3", "P2", "P1", "P1prime", "P2prime", "P3prime", "P4prime",
        "reference", "uniprot", "position", "organism", "protease_name",
        "col17", "col18", "col19", "col20", "col21", "cleavage_type",
        "_trailing",  # trailing tab creates empty 24th column
    ]

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=col_names,
        dtype=str,
        on_bad_lines="skip",
        encoding="latin-1",
    )
    print(f"  Raw rows loaded: {len(df):,}")

    # Strip single quotes from all string columns
    for col in df.columns:
        df[col] = df[col].apply(strip_quotes)

    # Extract protease family
    df["protease_family"] = df["merops_code"].apply(extract_family)

    # --- A. Exopeptidase cleavages ---
    exo = df[df["protease_family"].isin(EXOPEPTIDASE_FAMILIES)].copy()
    exo_cols = [
        "cleavage_id", "merops_code", "protease_family", "protease_name",
        "substrate_name", "P4", "P3", "P2", "P1", "P1prime", "P2prime",
        "P3prime", "P4prime", "uniprot", "position", "organism", "cleavage_type",
    ]
    exo = exo[exo_cols]
    exo_path = os.path.join(OUT, "merops_exopeptidase_cleavages.csv")
    exo.to_csv(exo_path, index=False)
    print(f"  Exopeptidase cleavages: {len(exo):,} rows -> {exo_path}")
    print(f"    Families: {sorted(exo['protease_family'].unique())}")

    # --- B. Summary stats per protease family ---
    summary = (
        df.groupby("protease_family")
        .agg(
            protease_name_example=("protease_name", "first"),
            total_cleavages=("cleavage_id", "count"),
            organisms_count=("organism", "nunique"),
        )
        .reset_index()
    )
    summary["is_exopeptidase"] = summary["protease_family"].isin(EXOPEPTIDASE_FAMILIES)
    summary = summary.sort_values("total_cleavages", ascending=False)
    summary_path = os.path.join(OUT, "merops_all_cleavages_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"  Cleavage summary: {len(summary)} families -> {summary_path}")
    print(f"    Top 5 families by cleavage count:")
    for _, row in summary.head(5).iterrows():
        tag = " [EXO]" if row["is_exopeptidase"] else ""
        print(f"      {row['protease_family']}: {row['total_cleavages']:,} cleavages{tag}")

    return len(exo), len(summary)


# ============================================================
# C. PEPlife2 combined
# ============================================================
def process_peplife():
    print("=" * 60)
    print("Processing PEPlife2 JSON files ...")

    keep_cols = [
        "id", "seq", "name", "length", "half_life", "units_half",
        "protease", "assay", "test_sample", "vivo_vitro",
        "lin_cyc", "chiral", "chem_mod", "nter", "cter",
        "origin", "nature", "pmid", "year",
    ]

    frames = []
    for fname in ["peplife2_api_natural.json", "peplife2_api_modified.json"]:
        path = os.path.join(DATA, "peplife", fname)
        with open(path) as f:
            raw = json.load(f)
        records = raw.get("data", [])
        df = pd.DataFrame(records)
        print(f"  {fname}: {len(df):,} records")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # Keep only desired columns (some may be missing)
    available = [c for c in keep_cols if c in combined.columns]
    combined = combined[available]

    # Convert half_life to numeric
    combined["half_life"] = pd.to_numeric(combined["half_life"], errors="coerce")
    combined["length"] = pd.to_numeric(combined["length"], errors="coerce")

    # Drop exact duplicates on id
    before = len(combined)
    combined = combined.drop_duplicates(subset=["id"])
    dupes = before - len(combined)

    out_path = os.path.join(OUT, "peplife2_combined.csv")
    combined.to_csv(out_path, index=False)
    print(f"  Combined: {len(combined):,} records ({dupes} dupes dropped) -> {out_path}")
    print(f"    With half_life: {combined['half_life'].notna().sum():,}")
    print(f"    Vivo/vitro split: {combined['vivo_vitro'].value_counts().to_dict()}")

    return len(combined)


# ============================================================
# D. iDPPIV benchmark
# ============================================================
def parse_fasta_like(path):
    """Parse iDPPIV FASTA-like format: >Label N\\nSEQUENCE"""
    sequences = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            sequences.append(line)
    return sequences


def process_dppiv_benchmark():
    print("=" * 60)
    print("Processing iDPPIV benchmark ...")

    base = os.path.join(DATA, "dppiv", "idppiv-benchmark", "iDPPIV", "data")
    rows = []

    for split in ["train", "test"]:
        for label_name, label_val in [("positive", 1), ("negative", 0)]:
            fname = f"{split}_{label_name}.txt"
            path = os.path.join(base, fname)
            seqs = parse_fasta_like(path)
            for seq in seqs:
                rows.append({"sequence": seq, "label": label_val, "split": split})
            print(f"  {fname}: {len(seqs):,} sequences")

    df = pd.DataFrame(rows)
    out_path = os.path.join(OUT, "dppiv_benchmark.csv")
    df.to_csv(out_path, index=False)
    print(f"  Combined: {len(df):,} rows -> {out_path}")
    print(f"    Label distribution: {df['label'].value_counts().to_dict()}")
    print(f"    Split distribution: {df['split'].value_counts().to_dict()}")

    return len(df)


# ============================================================
# E. ChEMBL DPP-IV IC50
# ============================================================
def process_chembl():
    print("=" * 60)
    print("Processing ChEMBL DPP-IV activities ...")

    path = os.path.join(DATA, "dppiv", "chembl", "chembl284_dpp4_activities.csv")
    df = pd.read_csv(path)
    print(f"  Raw rows: {len(df):,}")
    print(f"  Standard types: {df['standard_type'].value_counts().head().to_dict()}")

    # Filter to IC50 only
    ic50 = df[df["standard_type"] == "IC50"].copy()
    print(f"  IC50 rows: {len(ic50):,}")

    # Drop rows without pchembl_value
    ic50 = ic50.dropna(subset=["pchembl_value"])
    print(f"  With pchembl_value: {len(ic50):,}")

    # Rename and select columns
    ic50 = ic50.rename(columns={
        "molecule_chembl_id": "molecule_id",
        "canonical_smiles": "smiles",
        "standard_value": "ic50_nm",
        "pchembl_value": "pchembl",
        "assay_description": "assay_description",
        "document_journal": "journal",
        "document_year": "year",
    })

    keep = ["molecule_id", "smiles", "ic50_nm", "pchembl", "assay_description", "journal", "year"]
    ic50 = ic50[keep]
    ic50["ic50_nm"] = pd.to_numeric(ic50["ic50_nm"], errors="coerce")
    ic50["pchembl"] = pd.to_numeric(ic50["pchembl"], errors="coerce")

    out_path = os.path.join(OUT, "dppiv_chembl_ic50.csv")
    ic50.to_csv(out_path, index=False)
    print(f"  Output: {len(ic50):,} rows -> {out_path}")
    print(f"    pChEMBL range: {ic50['pchembl'].min():.2f} - {ic50['pchembl'].max():.2f}")
    print(f"    Unique molecules: {ic50['molecule_id'].nunique():,}")

    return len(ic50)


# ============================================================
# F. Dataset summary
# ============================================================
def write_summary(merops_exo, merops_fam, peplife_n, dppiv_bench, chembl_n):
    print("=" * 60)
    print("Writing dataset_summary.csv ...")

    rows = [
        {
            "dataset": "merops_exopeptidase_cleavages",
            "source": "MEROPS (merops.sanger.ac.uk)",
            "records": merops_exo,
            "description": "Exopeptidase cleavage sites from MEROPS substrate search",
            "exopeptidase_relevant": True,
        },
        {
            "dataset": "merops_all_cleavages_summary",
            "source": "MEROPS (merops.sanger.ac.uk)",
            "records": merops_fam,
            "description": "Summary statistics per protease family",
            "exopeptidase_relevant": True,
        },
        {
            "dataset": "peplife2_combined",
            "source": "PEPlife2 (peplife2.iith.ac.in)",
            "records": peplife_n,
            "description": "Peptide half-life data (natural + modified)",
            "exopeptidase_relevant": True,
        },
        {
            "dataset": "dppiv_benchmark",
            "source": "iDPPIV (Lin et al.)",
            "records": dppiv_bench,
            "description": "DPP-IV inhibitory peptide benchmark (train+test, binary labels)",
            "exopeptidase_relevant": True,
        },
        {
            "dataset": "dppiv_chembl_ic50",
            "source": "ChEMBL (target CHEMBL284)",
            "records": chembl_n,
            "description": "DPP-IV small-molecule IC50 values with pChEMBL scores",
            "exopeptidase_relevant": False,
        },
    ]

    df = pd.DataFrame(rows)
    out_path = os.path.join(OUT, "dataset_summary.csv")
    df.to_csv(out_path, index=False)
    print(f"  Summary: {len(df)} datasets -> {out_path}")
    print(f"  Total records across all datasets: {df['records'].sum():,}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("Bio-AI Toolkit — Dataset Processing Pipeline")
    print(f"Base directory: {BASE}")
    print(f"Output directory: {OUT}")
    print()

    merops_exo, merops_fam = process_merops()
    print()
    peplife_n = process_peplife()
    print()
    dppiv_bench = process_dppiv_benchmark()
    print()
    chembl_n = process_chembl()
    print()
    write_summary(merops_exo, merops_fam, peplife_n, dppiv_bench, chembl_n)

    print()
    print("=" * 60)
    print("All datasets processed successfully.")
    print(f"Output files in: {OUT}")
    for f in sorted(os.listdir(OUT)):
        if f.endswith(".csv"):
            size = os.path.getsize(os.path.join(OUT, f))
            print(f"  {f} ({size:,} bytes)")
