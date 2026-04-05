"""
ExoPred v4 Training — Turk MMP susceptibility features + Bottger external validation.

Key insight from previous iterations:
  - v2 (72 hand-crafted features): R2=0.997 random CV, R2=0.948 leave-sequence-out
  - v3 (+30 ESM-2 PCA features): R2=0.918 leave-sequence-out — WORSE (overfitting)
  - The bottleneck is generalization to novel sequences, not training fit.

New in v4:
  - 6 Turk-derived MMP susceptibility features from 18,583 peptides x 18 MMPs
  - Bottger 2017 external validation (55 stability measurements, different peptide class)
  - Same leave-sequence-out CV methodology as v3 for apples-to-apples comparison

Usage:
    python3 -m exopred.train_v4
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GroupKFold

from exopred.config import CHECKPOINT_DIR, DATA_DIR, PROCESSED_DIR
from exopred.train_v2 import (
    CELL_TYPES,
    N_MOD_BASE,
    C_MOD_BASE,
    STANDARD_AAS,
    build_merops_cleavage_freq,
    compute_fraction_remaining,
    featurize_one,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Turk 2015 data paths
# ---------------------------------------------------------------------------

TURK_PATH = DATA_DIR / "turk2015" / "mmc2-table-S1.xlsx"
BOTTGER_PATH = DATA_DIR / "external_validation" / "extracted_data.csv"
BOTTGER_CLEAVAGE_PATH = DATA_DIR / "external_validation" / "cleavage_sites.csv"

MMP_COLS = [
    "MMP1", "MMP2", "MMP3", "MMP7", "MMP8", "MMP9", "MMP10", "MMP11",
    "MMP12", "MMP13", "MMP14", "MMP15", "MMP16", "MMP17", "MMP19",
    "MMP20", "MMP24", "MMP25",
]


# ---------------------------------------------------------------------------
# Part 1: Build Turk-derived MMP susceptibility lookup tables
# ---------------------------------------------------------------------------

def build_turk_lookup() -> dict:
    """Build per-position, per-AA MMP susceptibility lookup from Turk et al. 2015.

    The Turk library contains 18,583 10-mer peptides with Z-scores for 18 MMPs.
    The cleavage site is between positions 4 and 5 (0-indexed), i.e.:
        P5 P4 P3 P2 P1 | P1' P2' P3' P4' P5'
        0  1  2  3  4     5   6   7   8   9

    Returns dict with keys:
        'position_aa_all':  {pos: {AA: mean_z_across_all_MMPs}}
        'position_aa_mmp14': {pos: {AA: mean_z_for_MMP14}}
        'turk_df': the loaded DataFrame (for debugging)
    """
    print("[TURK] Loading Turk et al. 2015 (18,583 peptides x 18 MMPs)...")
    turk_df = pd.read_excel(TURK_PATH, engine="openpyxl")
    turk_df.rename(columns={"Unnamed: 0": "sequence"}, inplace=True)

    # Validate
    print(f"[TURK] Loaded {len(turk_df)} peptides, {len(MMP_COLS)} MMPs")
    assert len(turk_df) > 18000, f"Expected ~18,583 peptides, got {len(turk_df)}"

    # Filter to peptides that are exactly 10 residues of standard AAs
    aa_set = set(STANDARD_AAS)
    valid_mask = turk_df["sequence"].apply(
        lambda s: isinstance(s, str) and len(s) == 10 and all(c in aa_set for c in s)
    )
    turk_valid = turk_df[valid_mask].copy()
    print(f"[TURK] Valid 10-mer standard-AA peptides: {len(turk_valid)}")

    # Build lookup: for each of 10 positions, compute average Z-score per AA
    # across all 18 MMPs (mean of means)
    position_aa_all = {}
    position_aa_mmp14 = {}

    for pos in range(10):
        position_aa_all[pos] = {}
        position_aa_mmp14[pos] = {}

        for aa in STANDARD_AAS:
            mask = turk_valid["sequence"].str[pos] == aa
            n_match = mask.sum()
            if n_match > 0:
                # Mean Z-score across all MMPs for this AA at this position
                z_all = turk_valid.loc[mask, MMP_COLS].mean().mean()
                position_aa_all[pos][aa] = float(z_all)

                # MMP-14 specific Z-score
                z_mmp14 = turk_valid.loc[mask, "MMP14"].mean()
                position_aa_mmp14[pos][aa] = float(z_mmp14)
            else:
                position_aa_all[pos][aa] = 0.0
                position_aa_mmp14[pos][aa] = 0.0

    # Print sample lookup values for verification
    print("[TURK] Sample lookup (position 4 = P1, position 5 = P1'):")
    for aa in "RGPK":
        p1_val = position_aa_all[4].get(aa, 0.0)
        p1p_val = position_aa_all[5].get(aa, 0.0)
        print(f"  {aa}: P1 z={p1_val:.3f}, P1' z={p1p_val:.3f}")

    return {
        "position_aa_all": position_aa_all,
        "position_aa_mmp14": position_aa_mmp14,
        "turk_df": turk_valid,
    }


def turk_features_for_peptide(
    seq: str,
    turk_lookup: dict,
) -> dict[str, float]:
    """Compute 6 Turk-derived MMP susceptibility features for a peptide.

    Turk positions (0-indexed in the 10-mer):
        P1  = position 4 (last residue before cleavage)
        P1' = position 5 (first residue after cleavage)

    For Sam's peptides, we interpret:
        - C-terminal AA -> maps to P1 (the residue carboxypeptidases release)
        - N-terminal AA -> maps to P1' (the residue aminopeptidases release)
        - Internal positions -> scan all internal dipeptide junctions as potential
          endopeptidase sites

    Features:
        1. turk_nterm_mmp_avg: avg MMP Z-score for N-terminal AA at P1'
        2. turk_cterm_mmp_avg: avg MMP Z-score for C-terminal AA at P1
        3. turk_nterm_mmp14: MMP-14 Z-score for N-terminal AA at P1'
        4. turk_cterm_mmp14: MMP-14 Z-score for C-terminal AA at P1
        5. turk_max_internal_mmp14: max MMP-14 Z-score for any internal cleavage site
        6. turk_mean_mmp_susceptibility: mean Z across all positions x all MMPs
    """
    pos_all = turk_lookup["position_aa_all"]
    pos_mmp14 = turk_lookup["position_aa_mmp14"]

    n_aa = seq[0] if len(seq) > 0 else "A"
    c_aa = seq[-1] if len(seq) > 0 else "A"

    # Feature 1: N-terminal AA mapped to P1' (position 5) — avg across all MMPs
    turk_nterm_mmp_avg = pos_all[5].get(n_aa, 0.0)

    # Feature 2: C-terminal AA mapped to P1 (position 4) — avg across all MMPs
    turk_cterm_mmp_avg = pos_all[4].get(c_aa, 0.0)

    # Feature 3: N-terminal AA at P1' — MMP-14 specific
    turk_nterm_mmp14 = pos_mmp14[5].get(n_aa, 0.0)

    # Feature 4: C-terminal AA at P1 — MMP-14 specific
    turk_cterm_mmp14 = pos_mmp14[4].get(c_aa, 0.0)

    # Feature 5: Max MMP-14 Z-score for any internal cleavage site
    # For each internal position i (1 to len-2), the AA at position i is
    # the P1 residue if a cut happens between i and i+1.
    # We look up P1 (position 4 in Turk) scores for each internal AA.
    internal_mmp14_scores = []
    for i in range(1, len(seq) - 1):
        aa = seq[i]
        # This AA as P1 of an internal cleavage
        score = pos_mmp14[4].get(aa, 0.0)
        internal_mmp14_scores.append(score)
    turk_max_internal_mmp14 = max(internal_mmp14_scores) if internal_mmp14_scores else 0.0

    # Feature 6: Mean MMP susceptibility across the full sequence
    # For each residue, average its P1 and P1' scores (it could be on either side
    # of a cleavage), then average across the whole peptide and all MMPs.
    all_scores = []
    for i, aa in enumerate(seq):
        p1_score = pos_all[4].get(aa, 0.0)
        p1p_score = pos_all[5].get(aa, 0.0)
        all_scores.append((p1_score + p1p_score) / 2.0)
    turk_mean_mmp_susceptibility = np.mean(all_scores) if all_scores else 0.0

    return {
        "turk_nterm_mmp_avg": turk_nterm_mmp_avg,
        "turk_cterm_mmp_avg": turk_cterm_mmp_avg,
        "turk_nterm_mmp14": turk_nterm_mmp14,
        "turk_cterm_mmp14": turk_cterm_mmp14,
        "turk_max_internal_mmp14": turk_max_internal_mmp14,
        "turk_mean_mmp_susceptibility": turk_mean_mmp_susceptibility,
    }


# ---------------------------------------------------------------------------
# Part 2: Build expanded training set (72 v2 features + 6 Turk features = 78)
# ---------------------------------------------------------------------------

def build_dataset(turk_lookup: dict) -> dict:
    """Build training dataset: 228 peptides x 4 cell types = 912 samples.

    Returns dict with X_v2, X_v4, y, groups, feature names.
    """
    print("\n[DATA] Loading Paper 1 data...")
    rozans_path = DATA_DIR / "rozans-618-enriched.csv"
    df = pd.read_csv(rozans_path)
    paper1 = df[df["paper"] == "Paper 1 (ACS Biomater 2024)"].copy()

    valid_n = set(N_MOD_BASE.keys())
    valid_c = set(C_MOD_BASE.keys())
    paper1 = paper1[paper1["n_terminal"].isin(valid_n) & paper1["c_terminal"].isin(valid_c)]
    print(f"[DATA] Paper 1 peptides (valid mods): {len(paper1)}")

    unique_seqs = sorted(paper1["clean_sequence"].unique())
    print(f"[DATA] Unique base sequences: {len(unique_seqs)}")

    # Build MEROPS tables
    print("[DATA] Building MEROPS cleavage tables...")
    freq_tables = build_merops_cleavage_freq()

    # Pre-compute Turk features per unique sequence (they don't depend on mods/cell)
    print("[DATA] Computing Turk MMP features per sequence...")
    seq_to_turk = {}
    for seq in unique_seqs:
        seq_to_turk[seq] = turk_features_for_peptide(seq, turk_lookup)

    # Build training samples
    print("[DATA] Building training samples (peptides x cell types)...")
    v2_rows = []
    turk_rows = []
    targets = []
    groups = []

    seq_to_group = {seq: i for i, seq in enumerate(unique_seqs)}

    for _, row in paper1.iterrows():
        seq = str(row["clean_sequence"])
        n_mod = str(row["n_terminal"])
        c_mod = str(row["c_terminal"])
        n_aa = seq[0] if len(seq) > 0 else "A"
        c_aa = seq[-1] if len(seq) > 0 else "A"

        turk_feats = seq_to_turk[seq]

        for cell_type in CELL_TYPES:
            y = compute_fraction_remaining(n_mod, c_mod, n_aa, c_aa, cell_type)
            targets.append(y)
            groups.append(seq_to_group[seq])

            v2_feats = featurize_one(seq, n_mod, c_mod, cell_type, freq_tables)
            v2_rows.append(v2_feats)
            turk_rows.append(turk_feats)

    X_v2_df = pd.DataFrame(v2_rows)
    X_turk_df = pd.DataFrame(turk_rows)
    X_v4_df = pd.concat([X_v2_df, X_turk_df], axis=1)

    y = np.array(targets)
    groups = np.array(groups)

    v2_names = list(X_v2_df.columns)
    turk_names = list(X_turk_df.columns)
    v4_names = v2_names + turk_names

    print(f"[DATA] Samples: {len(y)}")
    print(f"[DATA] v2 features: {X_v2_df.shape[1]}, Turk features: {X_turk_df.shape[1]}, "
          f"v4 total: {X_v4_df.shape[1]}")
    print(f"[DATA] Target range: [{y.min():.3f}, {y.max():.3f}], mean={y.mean():.3f}")
    print(f"[DATA] Unique groups (sequences): {len(np.unique(groups))}")

    return {
        "X_v2": X_v2_df.values,
        "X_v4": X_v4_df.values,
        "y": y,
        "groups": groups,
        "feature_names_v2": v2_names,
        "feature_names_turk": turk_names,
        "feature_names_v4": v4_names,
        "freq_tables": freq_tables,
    }


# ---------------------------------------------------------------------------
# Part 3: Leave-sequence-out cross-validation
# ---------------------------------------------------------------------------

def leave_sequence_out_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    label: str,
) -> dict:
    """Run leave-sequence-out CV with GBR. Returns metrics dict."""
    n_groups = len(np.unique(groups))
    gkf = GroupKFold(n_splits=n_groups)

    y_pred_all = np.full_like(y, np.nan)

    gbr_params = dict(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42,
    )

    fold_r2s = []
    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        gbr = GradientBoostingRegressor(**gbr_params)
        gbr.fit(X_train, y_train)
        preds = gbr.predict(X_test)
        y_pred_all[test_idx] = preds

        fold_r2 = r2_score(y_test, preds) if len(y_test) > 1 else float("nan")
        fold_r2s.append(fold_r2)

    overall_r2 = r2_score(y, y_pred_all)
    overall_rmse = np.sqrt(mean_squared_error(y, y_pred_all))
    fold_r2s = np.array(fold_r2s)

    print(f"\n  {label}:")
    print(f"    Leave-seq-out R2:   {overall_r2:.4f}")
    print(f"    Leave-seq-out RMSE: {overall_rmse:.4f}")
    print(f"    Per-fold R2 range:  [{np.nanmin(fold_r2s):.4f}, {np.nanmax(fold_r2s):.4f}]")
    print(f"    Per-fold R2 mean:   {np.nanmean(fold_r2s):.4f} +/- {np.nanstd(fold_r2s):.4f}")

    return {
        "overall_r2": float(overall_r2),
        "overall_rmse": float(overall_rmse),
        "fold_r2_mean": float(np.nanmean(fold_r2s)),
        "fold_r2_std": float(np.nanstd(fold_r2s)),
        "fold_r2_min": float(np.nanmin(fold_r2s)),
        "fold_r2_max": float(np.nanmax(fold_r2s)),
        "n_folds": n_groups,
    }


# ---------------------------------------------------------------------------
# Part 4: Bottger external validation
# ---------------------------------------------------------------------------

def validate_bottger(
    gbr: GradientBoostingRegressor,
    feature_names: list[str],
    freq_tables: dict,
    turk_lookup: dict,
) -> dict:
    """Validate against Bottger 2017 stability data (55 measurements).

    Bottger's peptides are therapeutic antimicrobials (not adhesion peptides),
    measured in blood/plasma/serum (not cell culture). Even weak correlation
    here means the model is learning real biology.
    """
    print("\n" + "=" * 60)
    print("EXTERNAL VALIDATION: Bottger 2017")
    print("=" * 60)

    df = pd.read_csv(BOTTGER_PATH)
    print(f"  Loaded {len(df)} stability measurements for {df['peptide_name'].nunique()} peptides")
    print(f"  Matrices: {sorted(df['matrix'].unique())}")

    # Filter to rows with valid pct_intact (numeric)
    df = df[df["pct_intact"].notna()].copy()
    df["pct_intact"] = pd.to_numeric(df["pct_intact"], errors="coerce")
    df = df[df["pct_intact"].notna()].copy()
    print(f"  After filtering valid pct_intact: {len(df)} rows")

    if len(df) < 5:
        print("  Too few valid measurements for external validation.")
        return {"spearman_r": None, "spearman_p": None, "n_samples": len(df)}

    # Extract clean sequences: strip modification notation
    # Bottger sequences look like: "gu-ONNRPVYIPRPRPPHPRL-NH2"
    # We need to: remove prefix (gu-), remove suffix (-NH2/-OH),
    # and convert non-standard AAs (O=ornithine) to closest standard
    aa_set = set(STANDARD_AAS)
    NON_STANDARD_MAP = {
        "O": "K",  # Ornithine -> Lysine (closest: both basic, similar structure)
        "U": "C",  # Selenocysteine -> Cysteine
        "B": "N",  # Asn or Asp -> Asn
        "Z": "Q",  # Gln or Glu -> Gln
        "J": "L",  # Leu or Ile -> Leu
    }

    def clean_bottger_sequence(raw_seq: str) -> str | None:
        """Extract clean standard-AA sequence from Bottger notation."""
        if not isinstance(raw_seq, str):
            return None

        seq = raw_seq.strip()

        # Remove common prefixes
        for prefix in ["gu-", "Gu-", "GU-"]:
            if seq.startswith(prefix):
                seq = seq[len(prefix):]

        # Remove common suffixes
        for suffix in ["-NH2", "-OH", "-nh2", "-oh", "-COOH", "-amide"]:
            if seq.endswith(suffix):
                seq = seq[:-len(suffix)]

        # Map non-standard AAs
        cleaned = []
        for c in seq:
            if c in aa_set:
                cleaned.append(c)
            elif c in NON_STANDARD_MAP:
                cleaned.append(NON_STANDARD_MAP[c])
            elif c.upper() in aa_set:
                cleaned.append(c.upper())
            elif c.upper() in NON_STANDARD_MAP:
                cleaned.append(NON_STANDARD_MAP[c.upper()])
            # Skip lowercase d-amino acid markers and other non-AA characters

        result = "".join(cleaned)
        return result if len(result) >= 3 else None

    df["clean_seq"] = df["sequence"].apply(clean_bottger_sequence)
    df = df[df["clean_seq"].notna()].copy()
    print(f"  After sequence cleaning: {len(df)} rows")

    if len(df) < 5:
        print("  Too few valid sequences for external validation.")
        return {"spearman_r": None, "spearman_p": None, "n_samples": len(df)}

    # Determine terminal modifications from Bottger notation
    def get_bottger_mods(mod_notes: str, raw_seq: str) -> tuple[str, str]:
        """Infer N-terminal and C-terminal modification from Bottger data."""
        mod = str(mod_notes).lower() if pd.notna(mod_notes) else ""
        seq = str(raw_seq).strip() if pd.notna(raw_seq) else ""

        # N-terminal
        n_mod = "NH2"  # default: free N-terminus
        if "guanidinylated" in mod or seq.startswith("gu-") or seq.startswith("Gu-"):
            n_mod = "Ac"  # guanidinylation protects N-terminus, similar to acetylation
        elif "acetyl" in mod or "ac-" in mod:
            n_mod = "Ac"

        # C-terminal
        c_mod = "COOH"  # default: free C-terminus
        if "c-term amide" in mod or seq.endswith("-NH2"):
            c_mod = "amide"
        elif seq.endswith("-OH"):
            c_mod = "COOH"

        return n_mod, c_mod

    # Predict for each row
    feats_list = []
    for _, row in df.iterrows():
        seq = row["clean_seq"]
        n_mod, c_mod = get_bottger_mods(row.get("modification_notes"), row.get("sequence"))

        # Use hUVEC as default cell type (moderate protease environment,
        # closest to the serum/plasma conditions in Bottger)
        cell_type = "hUVEC"

        # v2 features
        v2_feats = featurize_one(seq, n_mod, c_mod, cell_type, freq_tables)

        # Turk features
        turk_feats = turk_features_for_peptide(seq, turk_lookup)

        combined = {**v2_feats, **turk_feats}
        feats_list.append(combined)

    X_bottger = pd.DataFrame(feats_list)

    # Ensure columns match training feature order
    for col in feature_names:
        if col not in X_bottger.columns:
            X_bottger[col] = 0.0
    X_bottger = X_bottger[feature_names]

    y_pred = gbr.predict(X_bottger.values)

    # Compute correlation: predicted stability vs actual pct_intact
    actual = df["pct_intact"].values
    rho, p_val = stats.spearmanr(y_pred, actual)

    # Also compute Pearson
    pearson_r, pearson_p = stats.pearsonr(y_pred, actual)

    print(f"\n  Bottger external validation ({len(df)} measurements):")
    print(f"    Spearman rho:     {rho:.4f} (p={p_val:.2e})")
    print(f"    Pearson r:        {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"    Predicted range:  [{y_pred.min():.3f}, {y_pred.max():.3f}]")
    print(f"    Actual range:     [{actual.min():.1f}%, {actual.max():.1f}%]")

    # Per-peptide breakdown
    print("\n  Per-peptide summary:")
    for pname in sorted(df["peptide_name"].unique()):
        sub = df[df["peptide_name"] == pname]
        idx = sub.index
        pred_mean = y_pred[df.index.get_indexer(idx)].mean()
        actual_mean = sub["pct_intact"].mean()
        print(f"    {pname:15s}: predicted={pred_mean:.3f}, actual={actual_mean:.1f}% intact "
              f"({len(sub)} measurements)")

    return {
        "spearman_r": float(rho),
        "spearman_p": float(p_val),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "n_samples": len(df),
        "n_peptides": int(df["peptide_name"].nunique()),
        "pred_range": [float(y_pred.min()), float(y_pred.max())],
        "actual_range": [float(actual.min()), float(actual.max())],
    }


# ---------------------------------------------------------------------------
# Part 5: Feature importance + final model
# ---------------------------------------------------------------------------

def train_final_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
) -> tuple[GradientBoostingRegressor, pd.DataFrame]:
    """Train final GBR on all data and report feature importance."""
    print("\n" + "=" * 60)
    print(f"FEATURE IMPORTANCE ({len(feature_names)} features, full training set)")
    print("=" * 60)

    gbr = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42,
    )
    gbr.fit(X, y)

    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": gbr.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print("\n  Top 20 features:")
    for i, row in fi.head(20).iterrows():
        marker = " [TURK]" if row["feature"].startswith("turk_") else ""
        print(f"    {i+1:2d}. {row['feature']:40s} {row['importance']:.4f}{marker}")

    turk_in_top20 = fi.head(20)["feature"].str.startswith("turk_").sum()
    turk_total_importance = fi[fi["feature"].str.startswith("turk_")]["importance"].sum()
    print(f"\n  Turk MMP features in top 20: {turk_in_top20}")
    print(f"  Turk MMP total importance:   {turk_total_importance:.4f} "
          f"({turk_total_importance * 100:.1f}%)")

    return gbr, fi


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("ExoPred v4 — Turk MMP Features + Bottger External Validation")
    print("=" * 60)

    # Part 1: Build Turk lookup tables
    turk_lookup = build_turk_lookup()

    # Part 2: Build dataset
    data = build_dataset(turk_lookup)
    X_v2 = data["X_v2"]
    X_v4 = data["X_v4"]
    y = data["y"]
    groups = data["groups"]

    # Part 3: Leave-sequence-out CV — compare v2 vs v4
    print("\n" + "=" * 60)
    print("LEAVE-SEQUENCE-OUT CROSS-VALIDATION")
    print(f"({len(np.unique(groups))} unique sequences, {len(y)} total samples)")
    print("=" * 60)

    results_v2 = leave_sequence_out_cv(
        X_v2, y, groups,
        "Model v2: 72 hand-crafted features (baseline)",
    )

    results_v4 = leave_sequence_out_cv(
        X_v4, y, groups,
        "Model v4: 72 hand-crafted + 6 Turk MMP features",
    )

    # Feature importance + final model
    gbr_final, fi = train_final_model(
        X_v4, y, data["feature_names_v4"],
    )

    # Part 4: Bottger external validation
    bottger_results = validate_bottger(
        gbr_final,
        data["feature_names_v4"],
        data["freq_tables"],
        turk_lookup,
    )

    # Part 5: Summary + save
    print("\n" + "=" * 60)
    print("SUMMARY — v4 vs v2 Comparison")
    print("=" * 60)

    delta_r2 = results_v4["overall_r2"] - results_v2["overall_r2"]

    print(f"\n  {'Model':<50s} {'LSO R2':>8s} {'RMSE':>8s}")
    print(f"  {'-'*50} {'-'*8} {'-'*8}")
    print(f"  {'v2: 72 hand-crafted features':<50s} "
          f"{results_v2['overall_r2']:8.4f} {results_v2['overall_rmse']:8.4f}")
    print(f"  {'v4: 72 hand-crafted + 6 Turk MMP':<50s} "
          f"{results_v4['overall_r2']:8.4f} {results_v4['overall_rmse']:8.4f}")
    print(f"\n  Turk MMP delta R2 (v4 - v2): {delta_r2:+.4f}")

    if delta_r2 > 0.01:
        print("  --> Turk MMP features IMPROVE generalization to novel peptides.")
    elif delta_r2 > -0.01:
        print("  --> Turk MMP features have MARGINAL effect on generalization.")
    else:
        print("  --> Turk MMP features HURT generalization (possible overfitting).")

    if bottger_results.get("spearman_r") is not None:
        print(f"\n  Bottger external validation:")
        print(f"    Spearman rho: {bottger_results['spearman_r']:.4f} "
              f"(p={bottger_results['spearman_p']:.2e})")
        print(f"    Pearson r:    {bottger_results['pearson_r']:.4f} "
              f"(p={bottger_results['pearson_p']:.2e})")
        print(f"    n={bottger_results['n_samples']} measurements, "
              f"{bottger_results['n_peptides']} peptides")

        if bottger_results["spearman_p"] < 0.05:
            print("    --> SIGNIFICANT correlation with independent therapeutic peptide data!")
        elif bottger_results["spearman_p"] < 0.10:
            print("    --> Marginally significant — suggestive but not conclusive.")
        else:
            print("    --> Not significant — model may not generalize to this peptide class.")

    # Save artifacts
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Model
    model_path = CHECKPOINT_DIR / "exopred_v4_gbr.pkl"
    joblib.dump({
        "gbr": gbr_final,
        "feature_names": data["feature_names_v4"],
        "turk_lookup": {
            "position_aa_all": turk_lookup["position_aa_all"],
            "position_aa_mmp14": turk_lookup["position_aa_mmp14"],
        },
    }, model_path)
    print(f"\n[SAVE] Model -> {model_path}")

    # Feature importance
    fi_path = CHECKPOINT_DIR / "v4_feature_importance.csv"
    fi.to_csv(fi_path, index=False)
    print(f"[SAVE] Feature importance -> {fi_path}")

    # Metrics
    metrics = {
        "model_v2_baseline": results_v2,
        "model_v4_turk": results_v4,
        "turk_delta_r2": float(delta_r2),
        "bottger_external_validation": bottger_results,
        "n_samples": len(y),
        "n_sequences": len(np.unique(groups)),
        "n_features": {
            "v2": int(X_v2.shape[1]),
            "turk": 6,
            "v4": int(X_v4.shape[1]),
        },
    }
    metrics_path = CHECKPOINT_DIR / "v4_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVE] Metrics -> {metrics_path}")

    print(f"\n  All artifacts saved to {CHECKPOINT_DIR}/")


if __name__ == "__main__":
    main()
