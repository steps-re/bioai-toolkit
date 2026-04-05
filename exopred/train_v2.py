"""
ExoPred v2 Training — MEROPS features + Sam's calibration labels.

Approach:
  - Training labels: Sam's Paper 1 (234 peptides × 4 cell types = 936 samples)
    with fraction_remaining_48h computed from published calibration model.
  - Features: physicochemical (18) + MEROPS cleavage frequency (44) +
    terminal modification encoding (6) + cell type (4) = ~72 features.
  - Models: GradientBoostingRegressor (primary) + Ridge (baseline).
  - Validation: Spearman correlation against PEPlife2 half-lives.

Usage:
    python3 -m exopred.train_v2              # Train + evaluate
    python3 -m exopred.train_v2 --validate   # Just validate against PEPlife2
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from exopred.config import CHECKPOINT_DIR, DATA_DIR, PROCESSED_DIR
from exopred.data_pipeline import THREE_TO_ONE, NON_RESIDUE_CODES, _three_to_one
from exopred.features import KD, ENZYME_FAMILIES, AMINO_FAMILIES, CARBOXY_FAMILIES

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STANDARD_AAS = "ACDEFGHIKLMNPQRSTVWY"

# Sam's calibration model parameters (ACS Biomater 2024)
N_MOD_BASE = {"NH2": 0.20, "N-\u03b2A": 0.62, "NH2-\u03b2A": 0.62, "Ac": 0.82, "Ac-\u03b2A": 0.81}
C_MOD_BASE = {"COOH": 0.36, "amide": 0.73, "\u03b2A": 0.83}

# AA effects on N-terminal stability
N_AA_EFFECT = {
    "P": +0.15, "D": +0.10, "E": +0.10,
    "K": -0.10, "R": -0.10, "H": -0.20, "W": -0.10,
    "F": -0.05, "L": -0.05,
}

# AA effects on C-terminal stability
C_AA_EFFECT = {
    "P": +0.20, "D": +0.12, "E": +0.12,
    "H": -0.15, "W": -0.08, "F": -0.10,
}

# Cell type aggressiveness (lower = more degradation)
CELL_TYPES = {
    "hMSC": 0.85,
    "hUVEC": 0.45,
    "Macrophage": 0.20,
    "THP-1": 0.30,
}

# N-terminal modification protection level (0 = unprotected, 1 = fully protected)
N_MOD_PROTECTION = {
    "NH2": 0.0, "N-\u03b2A": 0.5, "NH2-\u03b2A": 0.5,
    "Ac": 1.0, "Ac-\u03b2A": 1.0,
    "cyclo": 1.0, "N3": 0.8,
}

# C-terminal modification protection level
C_MOD_PROTECTION = {
    "COOH": 0.0, "amide": 0.5, "\u03b2A": 1.0,
}


# ---------------------------------------------------------------------------
# MEROPS cleavage frequency table
# ---------------------------------------------------------------------------

def build_merops_cleavage_freq() -> pd.DataFrame:
    """Build a table: rows = 1-letter AA, columns = family, values = P(AA | family cleavage).

    Computes frequency for each position (P1, P1prime, P2, P2prime) separately,
    then returns a dict of DataFrames keyed by position name.
    """
    merops_path = PROCESSED_DIR / "merops_exopeptidase_cleavages.csv"
    df = pd.read_csv(merops_path)

    # Convert 3-letter to 1-letter for position columns
    pos_cols = ["P1", "P2", "P1prime", "P2prime"]
    for col in pos_cols:
        df[col + "_1"] = df[col].apply(
            lambda x: _three_to_one(str(x).strip()) if pd.notna(x) else None
        )

    families = sorted(df["protease_family"].unique())
    freq_tables = {}

    for pos in pos_cols:
        col_1 = pos + "_1"
        freq = pd.DataFrame(0.0, index=list(STANDARD_AAS), columns=families)

        for fam in families:
            sub = df[df["protease_family"] == fam]
            valid = sub[col_1].dropna()
            valid = valid[valid.isin(set(STANDARD_AAS))]
            if len(valid) == 0:
                continue
            counts = valid.value_counts()
            for aa, ct in counts.items():
                if aa in freq.index:
                    freq.loc[aa, fam] = ct / len(valid)

        freq_tables[pos] = freq

    return freq_tables


# ---------------------------------------------------------------------------
# Label computation: Sam's calibration model
# ---------------------------------------------------------------------------

def compute_fraction_remaining(
    n_terminal: str,
    c_terminal: str,
    n_term_aa: str,
    c_term_aa: str,
    cell_type: str,
) -> float:
    """Compute fraction_remaining_48h from Sam's published calibration.

    predicted = (n_base + c_base) / 2 + n_aa_effect + c_aa_effect
    Then scaled by cell aggressiveness:
        final = predicted * cell_aggressiveness + (1 - cell_aggressiveness) * (1 - predicted)
    More aggressive cells (lower value) degrade more peptide.
    """
    n_base = N_MOD_BASE.get(n_terminal, 0.20)  # default to NH2 (unprotected)
    c_base = C_MOD_BASE.get(c_terminal, 0.36)  # default to COOH (unprotected)

    n_aa = N_AA_EFFECT.get(n_term_aa, 0.0)
    c_aa = C_AA_EFFECT.get(c_term_aa, 0.0)

    base = (n_base + c_base) / 2.0 + n_aa + c_aa

    # Cell aggressiveness scaling:
    # High aggressiveness (hMSC=0.85) means cell environment is mild -> peptide survives more
    # Low aggressiveness (Macrophage=0.20) means cell is aggressive -> peptide degrades
    agg = CELL_TYPES.get(cell_type, 0.50)
    # Scale: fraction_remaining = base * aggressiveness_factor
    # When agg=1.0 (very mild), fraction ~ base
    # When agg=0.0 (very aggressive), fraction ~ 0
    final = base * agg

    return float(np.clip(final, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

# Physicochemical features (reusing constants from features.py)
_AA_MW = {
    "A": 89.09, "R": 174.20, "N": 132.12, "D": 133.10, "C": 121.16,
    "Q": 146.15, "E": 147.13, "G": 75.03, "H": 155.16, "I": 131.17,
    "L": 131.17, "K": 146.19, "M": 149.21, "F": 165.19, "P": 115.13,
    "S": 105.09, "T": 119.12, "W": 204.23, "Y": 181.19, "V": 117.15,
}

HYDROPHOBIC = set("AVILMFWP")
POLAR = set("STNQ")
CHARGED_POS = set("KRH")
CHARGED_NEG = set("DE")
AROMATIC = set("FWY")

# Instability index dipeptide weights (Guruprasad 1990) — import from features
from exopred.features import _DIWV


def physicochemical_features(seq: str) -> dict[str, float]:
    """Compute 18 physicochemical features for a peptide sequence."""
    n = len(seq)
    if n == 0:
        return {f"phys_{i}": 0.0 for i in range(18)}

    # 1. Length
    length = n

    # 2. Molecular weight
    mw = sum(_AA_MW.get(aa, 110.0) for aa in seq) - (n - 1) * 18.02

    # 3. Net charge at pH 7
    charge = 0.0
    for aa in seq:
        if aa in CHARGED_POS:
            charge += 1.0
        elif aa in CHARGED_NEG:
            charge -= 1.0

    # 4. GRAVY (grand average of hydropathy)
    gravy = np.mean([KD.get(aa, 0.0) for aa in seq])

    # 5. Instability index
    instability = 0.0
    for i in range(n - 1):
        dp = seq[i] + seq[i + 1]
        instability += _DIWV.get(dp, 1.0)
    instability = instability * 10.0 / n if n > 0 else 0.0

    # 6. Aromaticity
    aromaticity = sum(1 for aa in seq if aa in AROMATIC) / n

    # 7-10. Composition fractions
    frac_hydrophobic = sum(1 for aa in seq if aa in HYDROPHOBIC) / n
    frac_polar = sum(1 for aa in seq if aa in POLAR) / n
    frac_charged_pos = sum(1 for aa in seq if aa in CHARGED_POS) / n
    frac_charged_neg = sum(1 for aa in seq if aa in CHARGED_NEG) / n

    # 11-12. Terminal hydrophobicity
    n_term_hydro = KD.get(seq[0], 0.0)
    c_term_hydro = KD.get(seq[-1], 0.0)

    # 13-14. Proline flags
    has_n_proline = 1.0 if seq[0] == "P" else 0.0
    has_c_proline = 1.0 if seq[-1] == "P" else 0.0

    # 15. Fraction branched (VILM)
    frac_branched = sum(1 for aa in seq if aa in set("VILM")) / n

    # 16. Fraction small (GAP)
    frac_small = sum(1 for aa in seq if aa in set("GAP")) / n

    # 17. Absolute net charge
    abs_charge = abs(charge)

    # 18. Helix propensity (Chou-Fasman simplified)
    helix_formers = set("AELM")
    frac_helix = sum(1 for aa in seq if aa in helix_formers) / n

    return {
        "length": length,
        "mw": mw,
        "charge": charge,
        "gravy": gravy,
        "instability": instability,
        "aromaticity": aromaticity,
        "frac_hydrophobic": frac_hydrophobic,
        "frac_polar": frac_polar,
        "frac_charged_pos": frac_charged_pos,
        "frac_charged_neg": frac_charged_neg,
        "n_term_hydro": n_term_hydro,
        "c_term_hydro": c_term_hydro,
        "has_n_proline": has_n_proline,
        "has_c_proline": has_c_proline,
        "frac_branched": frac_branched,
        "frac_small": frac_small,
        "abs_charge": abs_charge,
        "frac_helix": frac_helix,
    }


def merops_features(
    seq: str,
    freq_tables: dict[str, pd.DataFrame],
) -> dict[str, float]:
    """Compute 44 MEROPS cleavage frequency features.

    For a sequence, look up:
      - N-terminal AA (pos 0) -> P1 cleavage frequencies across 11 families
      - Position 2 AA (pos 1) -> P2 cleavage frequencies across 11 families
      - Position N-1 AA (pos -2) -> P2prime frequencies across 11 families
      - C-terminal AA (pos -1) -> P1prime frequencies across 11 families
    """
    families = sorted(freq_tables["P1"].columns)
    feats = {}

    # N-terminal AA -> aminopeptidase P1 preferences
    n_aa = seq[0] if len(seq) > 0 else "A"
    for fam in families:
        val = freq_tables["P1"].loc[n_aa, fam] if n_aa in freq_tables["P1"].index else 0.0
        feats[f"merops_nterm_P1_{fam}"] = val

    # Position 2 -> P2 preferences
    p2_aa = seq[1] if len(seq) > 1 else "A"
    for fam in families:
        val = freq_tables["P2"].loc[p2_aa, fam] if p2_aa in freq_tables["P2"].index else 0.0
        feats[f"merops_pos2_P2_{fam}"] = val

    # Position N-1 -> P2prime preferences (for carboxypeptidases, penultimate position)
    pn1_aa = seq[-2] if len(seq) > 1 else "A"
    for fam in families:
        val = freq_tables["P2prime"].loc[pn1_aa, fam] if pn1_aa in freq_tables["P2prime"].index else 0.0
        feats[f"merops_posN1_P2p_{fam}"] = val

    # C-terminal AA -> carboxypeptidase P1prime preferences
    c_aa = seq[-1] if len(seq) > 0 else "A"
    for fam in families:
        val = freq_tables["P1prime"].loc[c_aa, fam] if c_aa in freq_tables["P1prime"].index else 0.0
        feats[f"merops_cterm_P1p_{fam}"] = val

    return feats


def terminal_mod_features(n_terminal: str, c_terminal: str) -> dict[str, float]:
    """Encode terminal modifications (6 features)."""
    n_prot = N_MOD_PROTECTION.get(n_terminal, 0.0)
    c_prot = C_MOD_PROTECTION.get(c_terminal, 0.0)

    return {
        "n_protection_level": n_prot,
        "c_protection_level": c_prot,
        "has_n_protection": 1.0 if n_prot > 0 else 0.0,
        "has_c_protection": 1.0 if c_prot > 0 else 0.0,
        "n_mod_ordinal": {"NH2": 0, "N-\u03b2A": 1, "NH2-\u03b2A": 1, "Ac": 2, "Ac-\u03b2A": 3, "cyclo": 4, "N3": 2}.get(n_terminal, 0),
        "c_mod_ordinal": {"COOH": 0, "amide": 1, "\u03b2A": 2}.get(c_terminal, 0),
    }


def cell_type_features(cell_type: str) -> dict[str, float]:
    """One-hot encode cell type (4 features)."""
    return {
        f"cell_{ct}": 1.0 if cell_type == ct else 0.0
        for ct in CELL_TYPES
    }


def featurize_one(
    seq: str,
    n_terminal: str,
    c_terminal: str,
    cell_type: str,
    freq_tables: dict[str, pd.DataFrame],
) -> dict[str, float]:
    """Compute full feature vector for one sample."""
    feats = {}
    feats.update(physicochemical_features(seq))
    feats.update(merops_features(seq, freq_tables))
    feats.update(terminal_mod_features(n_terminal, c_terminal))
    feats.update(cell_type_features(cell_type))
    return feats


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def build_training_data() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Build training dataset: 234 peptides × 4 cell types = 936 samples.

    Returns (X_df, y, feature_names).
    """
    print("[DATA] Loading Sam's Paper 1 data...")
    rozans_path = DATA_DIR / "rozans-618-enriched.csv"
    df = pd.read_csv(rozans_path)
    paper1 = df[df["paper"] == "Paper 1 (ACS Biomater 2024)"].copy()
    print(f"[DATA] Paper 1: {len(paper1)} peptides")

    # Filter to peptides with valid terminal mods that exist in calibration
    # (cyclo and N3 don't have calibration base values — exclude from training)
    valid_n = set(N_MOD_BASE.keys())
    valid_c = set(C_MOD_BASE.keys())
    paper1 = paper1[paper1["n_terminal"].isin(valid_n) & paper1["c_terminal"].isin(valid_c)]
    print(f"[DATA] After filtering valid mods: {len(paper1)} peptides")

    print("[DATA] Building MEROPS cleavage frequency tables...")
    freq_tables = build_merops_cleavage_freq()
    for pos, tbl in freq_tables.items():
        print(f"  {pos}: {tbl.shape}")

    print("[DATA] Generating 936 training samples (peptides × cell types)...")
    rows = []
    targets = []

    for _, row in paper1.iterrows():
        seq = str(row["clean_sequence"])
        n_mod = str(row["n_terminal"])
        c_mod = str(row["c_terminal"])
        n_aa = seq[0] if len(seq) > 0 else "A"
        c_aa = seq[-1] if len(seq) > 0 else "A"

        for cell_type in CELL_TYPES:
            # Compute label
            y = compute_fraction_remaining(n_mod, c_mod, n_aa, c_aa, cell_type)
            targets.append(y)

            # Compute features
            feats = featurize_one(seq, n_mod, c_mod, cell_type, freq_tables)
            rows.append(feats)

    X_df = pd.DataFrame(rows)
    y = np.array(targets)
    feature_names = list(X_df.columns)

    print(f"[DATA] Dataset: {X_df.shape[0]} samples, {X_df.shape[1]} features")
    print(f"[DATA] Target range: [{y.min():.3f}, {y.max():.3f}], mean={y.mean():.3f}")

    return X_df, y, feature_names


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_models(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
) -> dict:
    """Train GBR (primary) and Ridge (baseline), report 5-fold CV and test metrics."""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # --- Gradient Boosting Regressor ---
    print("\n" + "=" * 60)
    print("MODEL 1: GradientBoostingRegressor")
    print("=" * 60)

    gbr = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42,
    )

    gbr_r2_cv = cross_val_score(gbr, X, y, cv=kf, scoring="r2")
    gbr_rmse_cv = -cross_val_score(gbr, X, y, cv=kf, scoring="neg_root_mean_squared_error")
    gbr_mae_cv = -cross_val_score(gbr, X, y, cv=kf, scoring="neg_mean_absolute_error")

    print(f"  5-fold CV R2:   {gbr_r2_cv.mean():.4f} +/- {gbr_r2_cv.std():.4f}")
    print(f"  5-fold CV RMSE: {gbr_rmse_cv.mean():.4f} +/- {gbr_rmse_cv.std():.4f}")
    print(f"  5-fold CV MAE:  {gbr_mae_cv.mean():.4f} +/- {gbr_mae_cv.std():.4f}")

    # Train final model on all data
    gbr.fit(X, y)
    y_pred_gbr = gbr.predict(X)
    train_r2 = r2_score(y, y_pred_gbr)
    print(f"  Train R2 (full): {train_r2:.4f}")

    # Feature importance
    importances = gbr.feature_importances_
    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    print("\n  Top 20 features:")
    for i, row in fi.head(20).iterrows():
        print(f"    {row['feature']:40s} {row['importance']:.4f}")

    # --- Ridge Regression (baseline) ---
    print("\n" + "=" * 60)
    print("MODEL 2: Ridge Regression (baseline)")
    print("=" * 60)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ridge = Ridge(alpha=1.0)
    ridge_r2_cv = cross_val_score(ridge, X_scaled, y, cv=kf, scoring="r2")
    ridge_rmse_cv = -cross_val_score(ridge, X_scaled, y, cv=kf, scoring="neg_root_mean_squared_error")
    ridge_mae_cv = -cross_val_score(ridge, X_scaled, y, cv=kf, scoring="neg_mean_absolute_error")

    print(f"  5-fold CV R2:   {ridge_r2_cv.mean():.4f} +/- {ridge_r2_cv.std():.4f}")
    print(f"  5-fold CV RMSE: {ridge_rmse_cv.mean():.4f} +/- {ridge_rmse_cv.std():.4f}")
    print(f"  5-fold CV MAE:  {ridge_mae_cv.mean():.4f} +/- {ridge_mae_cv.std():.4f}")

    ridge.fit(X_scaled, y)
    y_pred_ridge = ridge.predict(X_scaled)
    train_r2_ridge = r2_score(y, y_pred_ridge)
    print(f"  Train R2 (full): {train_r2_ridge:.4f}")

    return {
        "gbr": gbr,
        "ridge": ridge,
        "scaler": scaler,
        "feature_importance": fi,
        "metrics": {
            "gbr": {
                "cv_r2_mean": float(gbr_r2_cv.mean()),
                "cv_r2_std": float(gbr_r2_cv.std()),
                "cv_rmse_mean": float(gbr_rmse_cv.mean()),
                "cv_rmse_std": float(gbr_rmse_cv.std()),
                "cv_mae_mean": float(gbr_mae_cv.mean()),
                "cv_mae_std": float(gbr_mae_cv.std()),
                "train_r2": float(train_r2),
            },
            "ridge": {
                "cv_r2_mean": float(ridge_r2_cv.mean()),
                "cv_r2_std": float(ridge_r2_cv.std()),
                "cv_rmse_mean": float(ridge_rmse_cv.mean()),
                "cv_rmse_std": float(ridge_rmse_cv.std()),
                "cv_mae_mean": float(ridge_mae_cv.mean()),
                "cv_mae_std": float(ridge_mae_cv.std()),
                "train_r2": float(train_r2_ridge),
            },
        },
    }


# ---------------------------------------------------------------------------
# PEPlife2 validation
# ---------------------------------------------------------------------------

def validate_peplife2(
    gbr: GradientBoostingRegressor,
    freq_tables: dict[str, pd.DataFrame],
) -> dict:
    """Validate trained model against PEPlife2 half-lives (independent data).

    PEPlife2 has heterogeneous conditions (various proteases, assays, species),
    so we predict fraction_remaining using a default 'hUVEC' cell type and
    map terminal mods as best we can. The Spearman correlation is a sanity
    check — not a training metric.
    """
    print("\n" + "=" * 60)
    print("VALIDATION: PEPlife2 (independent)")
    print("=" * 60)

    peplife_path = PROCESSED_DIR / "peplife2_combined.csv"
    df = pd.read_csv(peplife_path)
    print(f"  PEPlife2 total: {len(df)} rows")

    # Filter to rows with numeric half-life in consistent units
    df = df[df["half_life"].notna()].copy()
    # Normalize units to minutes
    unit_map = {
        "minutes": 1.0, "minute": 1.0, "Minutes": 1.0,
        "hours": 60.0, "hour": 60.0, "Hours": 60.0,
        "seconds": 1.0 / 60.0, "Seconds": 1.0 / 60.0,
        "days": 1440.0, "day": 1440.0, "Days": 1440.0,
    }
    df = df[df["units_half"].isin(unit_map.keys())]
    df["half_life_min"] = df["half_life"] * df["units_half"].map(unit_map)
    print(f"  After filtering valid units: {len(df)} rows")

    # Filter to linear, L-amino acid peptides with standard sequences
    df = df[df["lin_cyc"] == "Linear"]
    df = df[df["chiral"] == "L"]
    print(f"  After linear + L filter: {len(df)} rows")

    # Keep only sequences with standard AAs
    aa_set = set(STANDARD_AAS)
    df = df[df["seq"].apply(lambda s: all(c in aa_set for c in str(s)) if pd.notna(s) else False)]
    print(f"  After standard AA filter: {len(df)} rows")

    if len(df) < 10:
        print("  Too few valid PEPlife2 entries for meaningful validation.")
        return {"spearman_r": None, "spearman_p": None, "n_samples": len(df)}

    # Map PEPlife2 terminal mods to our encoding
    def map_n_mod(nter):
        nter = str(nter).strip()
        if nter in ("Free", "free", "H"):
            return "NH2"
        if nter in ("Ac", "Acetyl", "acetyl"):
            return "Ac"
        return "NH2"  # default

    def map_c_mod(cter):
        cter = str(cter).strip()
        if cter in ("Free", "free", "OH", "COOH"):
            return "COOH"
        if cter in ("NH2", "amide", "Amide"):
            return "amide"
        return "COOH"  # default

    # Predict for each PEPlife2 peptide using default hUVEC cell type
    default_cell = "hUVEC"
    feats_list = []
    for _, row in df.iterrows():
        seq = str(row["seq"])
        n_mod = map_n_mod(row.get("nter", "Free"))
        c_mod = map_c_mod(row.get("cter", "Free"))
        feats = featurize_one(seq, n_mod, c_mod, default_cell, freq_tables)
        feats_list.append(feats)

    X_peplife = pd.DataFrame(feats_list)

    # Ensure columns match training features
    # (handle any missing columns by filling 0)
    train_feature_path = CHECKPOINT_DIR / "v2_feature_importance.csv"
    if train_feature_path.exists():
        fi = pd.read_csv(train_feature_path)
        expected_cols = list(fi["feature"])
        for col in expected_cols:
            if col not in X_peplife.columns:
                X_peplife[col] = 0.0
        X_peplife = X_peplife[expected_cols]

    y_pred = gbr.predict(X_peplife.values)

    # Convert half-life to approximate fraction_remaining_48h for correlation
    # Using first-order decay: fraction = exp(-ln2 * t / t_half)
    # t = 48 hours = 2880 minutes
    t_48h = 2880.0  # minutes
    df["frac_remaining_48h_approx"] = np.exp(-np.log(2) * t_48h / df["half_life_min"].clip(lower=0.01))

    # Spearman correlation between predicted stability and PEPlife2 stability
    rho, p_val = stats.spearmanr(y_pred, df["frac_remaining_48h_approx"].values)

    print(f"\n  PEPlife2 validation ({len(df)} peptides):")
    print(f"    Spearman rho:   {rho:.4f}")
    print(f"    Spearman p-val: {p_val:.2e}")
    print(f"    Predicted frac range: [{y_pred.min():.3f}, {y_pred.max():.3f}]")
    print(f"    PEPlife2 frac range:  [{df['frac_remaining_48h_approx'].min():.3f}, "
          f"{df['frac_remaining_48h_approx'].max():.3f}]")

    return {
        "spearman_r": float(rho),
        "spearman_p": float(p_val),
        "n_samples": len(df),
    }


# ---------------------------------------------------------------------------
# Save artifacts
# ---------------------------------------------------------------------------

def save_artifacts(results: dict, feature_names: list[str]) -> None:
    """Save model, feature importance, and metrics."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Model
    model_path = CHECKPOINT_DIR / "exopred_v2_gbr.pkl"
    joblib.dump({
        "gbr": results["gbr"],
        "ridge": results["ridge"],
        "scaler": results["scaler"],
        "feature_names": feature_names,
    }, model_path)
    print(f"\n[SAVE] Model -> {model_path}")

    # Feature importance
    fi_path = CHECKPOINT_DIR / "v2_feature_importance.csv"
    results["feature_importance"].to_csv(fi_path, index=False)
    print(f"[SAVE] Feature importance -> {fi_path}")

    # Metrics
    metrics_path = CHECKPOINT_DIR / "v2_metrics.json"
    metrics = results["metrics"]
    if "peplife2" in results:
        metrics["peplife2_validation"] = results["peplife2"]
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVE] Metrics -> {metrics_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ExoPred v2 training")
    parser.add_argument("--validate", action="store_true",
                        help="Only run PEPlife2 validation (requires trained model)")
    args = parser.parse_args()

    if args.validate:
        # Load saved model and validate
        model_path = CHECKPOINT_DIR / "exopred_v2_gbr.pkl"
        if not model_path.exists():
            print(f"[ERROR] No trained model at {model_path}. Run training first.")
            sys.exit(1)
        bundle = joblib.load(model_path)
        gbr = bundle["gbr"]
        freq_tables = build_merops_cleavage_freq()
        peplife2_results = validate_peplife2(gbr, freq_tables)
        return

    # Full training pipeline
    print("=" * 60)
    print("ExoPred v2 — MEROPS features + Sam's calibration labels")
    print("=" * 60)

    # 1. Build dataset
    X_df, y, feature_names = build_training_data()
    X = X_df.values

    # 2. Train models
    results = train_models(X, y, feature_names)

    # 3. Validate against PEPlife2
    freq_tables = build_merops_cleavage_freq()
    peplife2_results = validate_peplife2(results["gbr"], freq_tables)
    results["peplife2"] = peplife2_results

    # 4. Save
    save_artifacts(results, feature_names)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    gbr_m = results["metrics"]["gbr"]
    ridge_m = results["metrics"]["ridge"]
    print(f"  GBR   5-fold CV R2: {gbr_m['cv_r2_mean']:.4f} +/- {gbr_m['cv_r2_std']:.4f}")
    print(f"  Ridge 5-fold CV R2: {ridge_m['cv_r2_mean']:.4f} +/- {ridge_m['cv_r2_std']:.4f}")
    if peplife2_results.get("spearman_r") is not None:
        print(f"  PEPlife2 Spearman:  {peplife2_results['spearman_r']:.4f} "
              f"(n={peplife2_results['n_samples']})")
    print(f"\n  Artifacts saved to {CHECKPOINT_DIR}/")


if __name__ == "__main__":
    main()
