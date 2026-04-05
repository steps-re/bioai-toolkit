"""
ExoPred Data Normalization Pipeline
====================================
Normalizes heterogeneous peptide cleavage datasets into a unified training
format with three task-specific outputs (binary cleavage, half-life regression,
kinetic curves).

Usage:
    python -m exopred.data_pipeline          # from bioai-toolkit/
    python exopred/data_pipeline.py          # from bioai-toolkit/

Idempotent — safe to re-run. Overwrites previous outputs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "processed"
TRAIN_DIR = REPO_ROOT / "data" / "training"

MEROPS_PATH = DATA_DIR / "merops_exopeptidase_cleavages.csv"
PEPLIFE2_PATH = DATA_DIR / "peplife2_combined.csv"
DPPIV_PATH = DATA_DIR / "dppiv_benchmark.csv"
ROZANS_TEMPLATE_PATH = DATA_DIR / "rozans_template.csv"

# Reproducibility
SEED = 42

# ---------------------------------------------------------------------------
# Unified record schema
# ---------------------------------------------------------------------------

@dataclass
class PeptideRecord:
    sequence: str                               # canonical AA string (one-letter codes)
    n_terminal_mod: str = "none"                # NH2, Ac, Fmoc, PEG, etc.
    c_terminal_mod: str = "none"                # COOH, amide, etc.
    enzyme_ec: str = "unknown"                  # EC number like "3.4.11.2"
    enzyme_name: str = "unknown"                # APN, CPA, DPP-IV, etc.
    enzyme_family: str = "unknown"              # MEROPS family code like M01, S09
    measurement_type: str = "binary"            # "binary", "half_life", "kinetic_curve"
    value: float = np.nan                       # 1/0 for binary, t1/2 in min for half_life
    curve_values: list = field(default_factory=list)      # time-course values
    curve_timepoints: list = field(default_factory=list)  # corresponding timepoints (min)
    conditions: dict = field(default_factory=dict)        # pH, temp, cell_type, etc.
    source: str = ""                            # "merops", "peplife2", "dppiv_benchmark", "rozans"
    confidence: float = 1.0                     # 1.0=direct, 0.5=inferred


# ---------------------------------------------------------------------------
# Amino-acid code mappings
# ---------------------------------------------------------------------------

THREE_TO_ONE: dict[str, str] = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
    # Common non-standard with reasonable one-letter mappings
    "Nle": "L",   # norleucine ~ leucine
    "Nva": "V",   # norvaline ~ valine
    "Orn": "K",   # ornithine ~ lysine
    "Cit": "Q",   # citrulline ~ glutamine (uncharged arg)
    "Hyp": "P",   # hydroxyproline ~ proline
    "Sar": "G",   # sarcosine ~ glycine
    "Glp": "Q",   # pyroglutamate ~ glutamine
    "Abu": "A",   # aminobutyric acid ~ alanine
    "Xaa": "X",   # any amino acid
    "Tyn": "Y",   # nitro-tyrosine ~ tyrosine
    "Tys": "Y",   # sulfo-tyrosine ~ tyrosine
    "HSe": "S",   # homoserine ~ serine
    "HCy": "C",   # homocysteine ~ cysteine
}

# Residues that are chromophores, protecting groups, or linkers (not amino acids)
NON_RESIDUE_CODES: set[str] = {
    "-", "Abz", "Ac", "AMC", "Boc", "Bz", "Dnp", "Dns", "Dpa", "Dpm",
    "FA", "For", "Mca", "MCA", "MNA", "NAP", "NH2", "NPh", "OBz", "OEt",
    "OMe", "pNA", "Suc", "Tos", "Z", "Bio", "Sal", "Moc", "INH", "HYD",
    "FMC", "MFA", "NAN", "NPG", "ONA", "ONe", "ONl", "OPr", "PAG", "SBz",
    "pAb", "pGl", "\\'-\\'",
}

# MEROPS family -> (enzyme_name, EC number)
MEROPS_FAMILY_MAP: dict[str, tuple[str, str]] = {
    "M01": ("APN/CD13",               "3.4.11.2"),
    "M14": ("CPA/CPB",                "3.4.17.1"),
    "M17": ("LAP",                    "3.4.11.1"),
    "M18": ("aspartyl aminopeptidase", "3.4.11.21"),
    "M24": ("MetAP",                  "3.4.11.18"),
    "M28": ("aminopeptidase",         "3.4.11.-"),
    "M32": ("carboxypeptidase Taq",   "3.4.17.-"),
    "S09": ("DPP-IV/DPP-9",          "3.4.14.5"),
    "S10": ("serine carboxypeptidase", "3.4.16.5"),
    "S28": ("prolyl carboxypeptidase", "3.4.16.2"),
    "C01": ("cathepsin",              "3.4.22.-"),
}


def _three_to_one(code: str) -> Optional[str]:
    """Convert a 3-letter code to 1-letter. Returns None for non-residues."""
    if code in NON_RESIDUE_CODES:
        return None
    return THREE_TO_ONE.get(code, None)


def _build_sequence_from_p_sites(row: pd.Series) -> Optional[str]:
    """Build an 8-residue sequence from P4..P4' columns.

    Skips non-residue positions (chromophores, etc.). Returns None if
    fewer than 2 canonical residues are present (not useful for training).
    """
    cols = ["P4", "P3", "P2", "P1", "P1prime", "P2prime", "P3prime", "P4prime"]
    residues: list[str] = []
    for col in cols:
        val = row[col]
        if pd.isna(val) or val == "-":
            continue
        aa = _three_to_one(str(val).strip())
        if aa is not None:
            residues.append(aa)
    if len(residues) < 2:
        return None
    return "".join(residues)


# ---------------------------------------------------------------------------
# 1. Normalize MEROPS
# ---------------------------------------------------------------------------

def normalize_merops() -> list[PeptideRecord]:
    """Load MEROPS exopeptidase cleavages and produce positive + negative records."""
    print(f"[MEROPS] Loading {MEROPS_PATH.name} ...")
    df = pd.read_csv(MEROPS_PATH)
    print(f"[MEROPS] {len(df):,} raw rows")

    records: list[PeptideRecord] = []
    skipped = 0

    for _, row in df.iterrows():
        seq = _build_sequence_from_p_sites(row)
        if seq is None:
            skipped += 1
            continue

        family = str(row["protease_family"]).strip()
        enz_name, enz_ec = MEROPS_FAMILY_MAP.get(family, ("unknown", "unknown"))

        rec = PeptideRecord(
            sequence=seq,
            enzyme_ec=enz_ec,
            enzyme_name=enz_name,
            enzyme_family=family,
            measurement_type="binary",
            value=1.0,
            conditions={
                "cleavage_type": str(row.get("cleavage_type", "")),
                "organism": str(row.get("organism", "")),
            },
            source="merops",
            confidence=0.5,
        )
        records.append(rec)

    print(f"[MEROPS] {len(records):,} positive records ({skipped:,} skipped — non-standard)")

    # --- Negative sampling: shuffle P1/P1' to create non-cleavage examples ---
    rng = random.Random(SEED)
    positives = [r for r in records]
    negatives: list[PeptideRecord] = []

    for rec in positives:
        seq = list(rec.sequence)
        if len(seq) < 4:
            continue
        # Identify the cleavage site (positions 3,4 in 0-indexed for P1/P1')
        # For variable-length sequences, swap the middle two residues with random AAs
        mid = len(seq) // 2
        shuffled = seq.copy()
        # Swap P1 and P1' with random standard amino acids
        standard_aa = "ACDEFGHIKLMNPQRSTVWY"
        shuffled[mid - 1] = rng.choice(standard_aa)
        shuffled[mid] = rng.choice(standard_aa)
        neg_seq = "".join(shuffled)

        if neg_seq == rec.sequence:
            continue  # extremely unlikely but skip if identical

        neg = PeptideRecord(
            sequence=neg_seq,
            enzyme_ec=rec.enzyme_ec,
            enzyme_name=rec.enzyme_name,
            enzyme_family=rec.enzyme_family,
            measurement_type="binary",
            value=0.0,
            conditions=rec.conditions.copy(),
            source="merops",
            confidence=0.5,
        )
        negatives.append(neg)

    records.extend(negatives)
    print(f"[MEROPS] {len(negatives):,} negative samples generated")
    print(f"[MEROPS] {len(records):,} total records")
    return records


# ---------------------------------------------------------------------------
# 2. Normalize PEPlife2
# ---------------------------------------------------------------------------

# Unit -> multiplier to convert to minutes
UNIT_TO_MINUTES: dict[str, float] = {
    "minutes": 1.0, "minute": 1.0, "min": 1.0, "Minutes": 1.0,
    "hours": 60.0, "hour": 60.0, "Hours": 60.0, "h": 60.0,
    "seconds": 1.0 / 60.0, "Seconds": 1.0 / 60.0, "s": 1.0 / 60.0,
    "days": 1440.0, "day": 1440.0, "Days": 1440.0,
    "month": 43200.0, "Months": 43200.0, "Month": 43200.0,
    "Week": 10080.0, "week": 10080.0,
}

# PEPlife2 terminal modification mapping
NTER_MAP: dict[str, str] = {
    "Free": "NH2",
}

CTER_MAP: dict[str, str] = {
    "Free": "COOH",
    "Amidation": "amide",
    "Pegylation": "PEG",
}

# Rough protease name -> family mapping for PEPlife2
PROTEASE_FAMILY_MAP: dict[str, tuple[str, str, str]] = {
    # pattern substring -> (enzyme_name, enzyme_family, enzyme_ec)
    "DPP IV": ("DPP-IV", "S09", "3.4.14.5"),
    "DPP-IV": ("DPP-IV", "S09", "3.4.14.5"),
    "dipeptidyl peptidase": ("DPP-IV", "S09", "3.4.14.5"),
    "aminopeptidase N": ("APN/CD13", "M01", "3.4.11.2"),
    "neprilysin": ("NEP", "M13", "3.4.24.11"),
    "ACE": ("ACE", "M02", "3.4.15.1"),
    "angiotensin": ("ACE", "M02", "3.4.15.1"),
    "trypsin": ("trypsin", "S01", "3.4.21.4"),
    "chymotrypsin": ("chymotrypsin", "S01", "3.4.21.1"),
    "pepsin": ("pepsin", "A01", "3.4.23.1"),
    "cathepsin": ("cathepsin", "C01", "3.4.22.-"),
    "elastase": ("elastase", "S01", "3.4.21.36"),
    "plasmin": ("plasmin", "S01", "3.4.21.7"),
    "thrombin": ("thrombin", "S01", "3.4.21.5"),
}


def _map_protease(name: str) -> tuple[str, str, str]:
    """Map a PEPlife2 protease string to (enzyme_name, family, EC)."""
    if pd.isna(name):
        return ("unknown", "unknown", "unknown")
    name_lower = name.lower()
    for pattern, (enz, fam, ec) in PROTEASE_FAMILY_MAP.items():
        if pattern.lower() in name_lower:
            return (enz, fam, ec)
    return (str(name), "unknown", "unknown")


def normalize_peplife2() -> list[PeptideRecord]:
    """Load PEPlife2 dataset and convert to PeptideRecords."""
    print(f"\n[PEPlife2] Loading {PEPLIFE2_PATH.name} ...")
    df = pd.read_csv(PEPLIFE2_PATH)
    print(f"[PEPlife2] {len(df):,} raw rows")

    # Filter to rows with numeric half_life
    df = df[pd.to_numeric(df["half_life"], errors="coerce").notna()].copy()
    df["half_life"] = pd.to_numeric(df["half_life"])
    print(f"[PEPlife2] {len(df):,} rows with numeric half_life")

    records: list[PeptideRecord] = []
    unit_failures = 0

    for _, row in df.iterrows():
        seq = str(row["seq"]).strip().upper()
        # Validate: only keep sequences with standard amino acids
        if not seq or not all(c in "ACDEFGHIKLMNPQRSTVWY" for c in seq):
            continue

        # Convert half-life to minutes
        unit = str(row.get("units_half", "")).strip()
        multiplier = UNIT_TO_MINUTES.get(unit, None)
        if multiplier is None:
            unit_failures += 1
            continue
        half_life_min = float(row["half_life"]) * multiplier

        # Terminal modifications
        nter = str(row.get("nter", "Free")).strip()
        cter = str(row.get("cter", "Free")).strip()
        n_mod = NTER_MAP.get(nter, nter if nter != "nan" else "none")
        c_mod = CTER_MAP.get(cter, cter if cter != "nan" else "none")

        # Protease
        enz_name, enz_family, enz_ec = _map_protease(row.get("protease"))

        rec = PeptideRecord(
            sequence=seq,
            n_terminal_mod=n_mod,
            c_terminal_mod=c_mod,
            enzyme_ec=enz_ec,
            enzyme_name=enz_name,
            enzyme_family=enz_family,
            measurement_type="half_life",
            value=half_life_min,
            conditions={
                "vivo_vitro": str(row.get("vivo_vitro", "")),
                "test_sample": str(row.get("test_sample", "")),
                "assay": str(row.get("assay", "")),
                "lin_cyc": str(row.get("lin_cyc", "")),
            },
            source="peplife2",
            confidence=0.8,
        )
        records.append(rec)

    if unit_failures:
        print(f"[PEPlife2] {unit_failures} rows skipped (unrecognized units)")
    print(f"[PEPlife2] {len(records):,} total records")
    return records


# ---------------------------------------------------------------------------
# 3. Normalize DPP-IV benchmark
# ---------------------------------------------------------------------------

def normalize_dppiv() -> list[PeptideRecord]:
    """Load DPP-IV benchmark dataset and convert to PeptideRecords."""
    print(f"\n[DPP-IV] Loading {DPPIV_PATH.name} ...")
    df = pd.read_csv(DPPIV_PATH)
    print(f"[DPP-IV] {len(df):,} raw rows")

    records: list[PeptideRecord] = []
    for _, row in df.iterrows():
        seq = str(row["sequence"]).strip().upper()
        if not seq:
            continue

        rec = PeptideRecord(
            sequence=seq,
            enzyme_name="DPP-IV",
            enzyme_ec="3.4.14.5",
            enzyme_family="S09",
            measurement_type="binary",
            value=float(row["label"]),
            conditions={"split_original": str(row.get("split", ""))},
            source="dppiv_benchmark",
            confidence=0.9,
        )
        records.append(rec)

    print(f"[DPP-IV] {len(records):,} total records")
    return records


# ---------------------------------------------------------------------------
# 4. Rozans template slot
# ---------------------------------------------------------------------------

def prepare_rozans_slot() -> None:
    """Create a template CSV showing the exact format Sam's 80K data needs."""
    print(f"\n[Rozans] Generating template at {ROZANS_TEMPLATE_PATH.name} ...")

    template_rows = [
        {
            "sequence": "YGGFL",
            "n_terminal_mod": "none",
            "c_terminal_mod": "amide",
            "enzyme_ec": "3.4.14.5",
            "enzyme_name": "DPP-IV",
            "enzyme_family": "S09",
            "measurement_type": "kinetic_curve",
            "value": "",
            "curve_values": "100.0;82.3;61.5;34.2;12.1;3.8",
            "curve_timepoints": "0;5;15;30;60;120",
            "conditions_ph": "7.4",
            "conditions_temp_c": "37",
            "conditions_matrix": "human_plasma",
            "source": "rozans",
            "confidence": "1.0",
        },
        {
            "sequence": "DRVYIHPF",
            "n_terminal_mod": "Ac",
            "c_terminal_mod": "COOH",
            "enzyme_ec": "3.4.11.2",
            "enzyme_name": "APN/CD13",
            "enzyme_family": "M01",
            "measurement_type": "half_life",
            "value": "42.5",
            "curve_values": "100.0;75.0;50.0;25.0;10.0",
            "curve_timepoints": "0;15;42;90;180",
            "conditions_ph": "7.4",
            "conditions_temp_c": "37",
            "conditions_matrix": "buffer_PBS",
            "source": "rozans",
            "confidence": "1.0",
        },
        {
            "sequence": "RPKPQQFFGLM",
            "n_terminal_mod": "none",
            "c_terminal_mod": "none",
            "enzyme_ec": "unknown",
            "enzyme_name": "human_serum_mix",
            "enzyme_family": "unknown",
            "measurement_type": "kinetic_curve",
            "value": "",
            "curve_values": "100.0;95.1;88.7;79.3;65.4;48.2;30.1;15.6;7.2",
            "curve_timepoints": "0;2;5;10;20;40;60;120;240",
            "conditions_ph": "7.4",
            "conditions_temp_c": "37",
            "conditions_matrix": "human_serum",
            "source": "rozans",
            "confidence": "1.0",
        },
    ]

    template_df = pd.DataFrame(template_rows)
    template_df.to_csv(ROZANS_TEMPLATE_PATH, index=False)
    print(f"[Rozans] Template saved with {len(template_rows)} example rows")
    print("[Rozans] Columns:")
    for col in template_df.columns:
        print(f"         - {col}")


# ---------------------------------------------------------------------------
# 5. Build task-specific training sets
# ---------------------------------------------------------------------------

def _records_to_dataframe(records: list[PeptideRecord]) -> pd.DataFrame:
    """Convert list of PeptideRecords to a flat DataFrame."""
    rows = []
    for r in records:
        d = asdict(r)
        # Flatten conditions dict
        conds = d.pop("conditions", {})
        for k, v in conds.items():
            d[f"cond_{k}"] = v
        # Serialize lists
        d["curve_values"] = ";".join(str(x) for x in d["curve_values"]) if d["curve_values"] else ""
        d["curve_timepoints"] = ";".join(str(x) for x in d["curve_timepoints"]) if d["curve_timepoints"] else ""
        rows.append(d)
    return pd.DataFrame(rows)


def _stratified_split(
    df: pd.DataFrame,
    stratify_cols: list[str],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = SEED,
) -> pd.DataFrame:
    """Add a 'split' column with values train/val/test (80/10/10).

    Stratified by source and enzyme_family where possible.
    Falls back to random split for very small strata.
    """
    rng = np.random.RandomState(seed)

    # Build stratification key (handle missing values)
    strat_key = df[stratify_cols].fillna("_NA_").apply(lambda x: "|".join(x), axis=1)
    df = df.copy()
    df["_strat"] = strat_key
    df["split"] = ""

    for _, group_df in df.groupby("_strat"):
        idx = group_df.index.tolist()
        rng.shuffle(idx)
        n = len(idx)
        n_train = max(1, int(n * train_frac))
        n_val = max(1, int(n * val_frac)) if n > 2 else 0
        # Assign splits
        df.loc[idx[:n_train], "split"] = "train"
        df.loc[idx[n_train:n_train + n_val], "split"] = "val"
        df.loc[idx[n_train + n_val:], "split"] = "test"

    # Any remaining empties go to train
    df.loc[df["split"] == "", "split"] = "train"
    df.drop(columns=["_strat"], inplace=True)
    return df


def build_training_sets(
    merops_records: list[PeptideRecord],
    peplife2_records: list[PeptideRecord],
    dppiv_records: list[PeptideRecord],
) -> None:
    """Build and save task-specific training sets."""
    print("\n[Training] Building task-specific datasets ...")
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    # --- Task A: Binary cleavage prediction ---
    # MEROPS (already binary) + DPP-IV (already binary) + PEPlife2 binarized
    task_a_records: list[PeptideRecord] = []
    task_a_records.extend(merops_records)
    task_a_records.extend(dppiv_records)

    # Binarize PEPlife2: t1/2 < 60 min = cleaved (1), else intact (0)
    for rec in peplife2_records:
        binary_rec = PeptideRecord(
            sequence=rec.sequence,
            n_terminal_mod=rec.n_terminal_mod,
            c_terminal_mod=rec.c_terminal_mod,
            enzyme_ec=rec.enzyme_ec,
            enzyme_name=rec.enzyme_name,
            enzyme_family=rec.enzyme_family,
            measurement_type="binary",
            value=1.0 if rec.value < 60.0 else 0.0,
            conditions=rec.conditions.copy(),
            source=rec.source,
            confidence=rec.confidence * 0.9,  # slight penalty for binarization
        )
        task_a_records.append(binary_rec)

    df_a = _records_to_dataframe(task_a_records)
    df_a = _stratified_split(df_a, stratify_cols=["source", "enzyme_family"])

    path_a = TRAIN_DIR / "task_a_binary.csv"
    df_a.to_csv(path_a, index=False)
    print(f"[Task A] Binary cleavage: {len(df_a):,} records")
    print(f"         Positive: {(df_a['value'] == 1.0).sum():,}  Negative: {(df_a['value'] == 0.0).sum():,}")
    print(f"         Split: train={len(df_a[df_a['split']=='train']):,}  val={len(df_a[df_a['split']=='val']):,}  test={len(df_a[df_a['split']=='test']):,}")
    print(f"         Saved to {path_a}")

    # --- Task B: Half-life regression ---
    df_b = _records_to_dataframe(peplife2_records)
    df_b = _stratified_split(df_b, stratify_cols=["source", "enzyme_family"])

    path_b = TRAIN_DIR / "task_b_halflife.csv"
    df_b.to_csv(path_b, index=False)
    print(f"\n[Task B] Half-life regression: {len(df_b):,} records")
    print(f"         Value range: {df_b['value'].min():.2f} — {df_b['value'].max():.2f} minutes")
    print(f"         Median: {df_b['value'].median():.2f} min")
    print(f"         Split: train={len(df_b[df_b['split']=='train']):,}  val={len(df_b[df_b['split']=='val']):,}  test={len(df_b[df_b['split']=='test']):,}")
    print(f"         Saved to {path_b}")

    # --- Task C: Kinetic curves (placeholder for Rozans data) ---
    path_c = TRAIN_DIR / "task_c_kinetic.csv"
    # Create empty file with correct headers
    cols = [
        "sequence", "n_terminal_mod", "c_terminal_mod", "enzyme_ec",
        "enzyme_name", "enzyme_family", "measurement_type", "value",
        "curve_values", "curve_timepoints", "source", "confidence", "split",
    ]
    pd.DataFrame(columns=cols).to_csv(path_c, index=False)
    print(f"\n[Task C] Kinetic curves: 0 records (awaiting Rozans data)")
    print(f"         Saved to {path_c}")


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full normalization pipeline."""
    print("=" * 60)
    print("ExoPred Data Normalization Pipeline")
    print("=" * 60)

    # Verify inputs exist
    for path in [MEROPS_PATH, PEPLIFE2_PATH, DPPIV_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Missing input: {path}")
        print(f"  Found: {path.name}")

    print()

    # Normalize each source
    merops_records = normalize_merops()
    peplife2_records = normalize_peplife2()
    dppiv_records = normalize_dppiv()

    # Rozans template
    prepare_rozans_slot()

    # Build training sets
    build_training_sets(merops_records, peplife2_records, dppiv_records)

    # Summary
    total = len(merops_records) + len(peplife2_records) + len(dppiv_records)
    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print(f"  Total unified records: {total:,}")
    print(f"  Sources: MEROPS ({len(merops_records):,}), PEPlife2 ({len(peplife2_records):,}), DPP-IV ({len(dppiv_records):,})")
    print(f"  Outputs in: {TRAIN_DIR}")
    print(f"  Rozans template: {ROZANS_TEMPLATE_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
