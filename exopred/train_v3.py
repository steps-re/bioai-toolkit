"""
ExoPred v3 Training — ESM-2 embeddings + leave-sequence-out generalization test.

Hypothesis: ESM-2 protein language model embeddings capture biochemical context
beyond hand-crafted lookup tables and should improve generalization to truly
novel peptides (not just random CV splits).

Approach:
  - Step 1: Generate ESM-2 (35M param) embeddings for all peptide sequences.
    Extract mean-pooled, N-terminal, and C-terminal residue embeddings (480-dim each).
    PCA reduce each to 10 components -> 30 ESM features total.
  - Step 2: Combine with v2's 72 hand-crafted features -> 102 total.
  - Step 3: Leave-sequence-out CV (the real generalization test).
    Hold out ALL samples for one base sequence, train on the rest.
    Compare: Model A (72 features), Model B (72+30), Model C (30 ESM only).
  - Step 4: Feature importance analysis on the full 102-feature model.

Usage:
    python3 -m exopred.train_v3
"""

from __future__ import annotations

import json
import pickle
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GroupKFold

from exopred.config import CHECKPOINT_DIR, DATA_DIR, PROCESSED_DIR
from exopred.train_v2 import (
    CELL_TYPES,
    N_MOD_BASE,
    C_MOD_BASE,
    build_merops_cleavage_freq,
    compute_fraction_remaining,
    featurize_one,
)

warnings.filterwarnings("ignore", category=FutureWarning)

ESM_CACHE_PATH = PROCESSED_DIR / "esm2_embeddings.pkl"
ESM_MODEL_NAME = "facebook/esm2_t12_35M_UR50D"
N_PCA_COMPONENTS = 10  # per embedding type (mean, first, last)


# ---------------------------------------------------------------------------
# Step 1: ESM-2 Embeddings
# ---------------------------------------------------------------------------

def generate_esm2_embeddings(sequences: list[str]) -> dict[str, np.ndarray]:
    """Generate ESM-2 embeddings for a list of peptide sequences.

    Returns dict with keys: 'mean', 'first', 'last' — each shape (n_seqs, 480).
    Caches to disk so we don't recompute.
    """
    if ESM_CACHE_PATH.exists():
        print(f"[ESM2] Loading cached embeddings from {ESM_CACHE_PATH}")
        with open(ESM_CACHE_PATH, "rb") as f:
            cached = pickle.load(f)
        # Verify cache covers all sequences
        cached_seqs = set(cached.get("sequences", []))
        needed = set(sequences)
        if needed.issubset(cached_seqs):
            # Build arrays in the right order
            seq_to_idx = {s: i for i, s in enumerate(cached["sequences"])}
            idxs = [seq_to_idx[s] for s in sequences]
            return {
                "mean": cached["mean"][idxs],
                "first": cached["first"][idxs],
                "last": cached["last"][idxs],
            }
        print(f"[ESM2] Cache has {len(cached_seqs)} seqs, need {len(needed)}. Regenerating.")

    print(f"[ESM2] Loading model: {ESM_MODEL_NAME}")
    import torch
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
    model = AutoModel.from_pretrained(ESM_MODEL_NAME)
    model.eval()

    unique_seqs = sorted(set(sequences))
    print(f"[ESM2] Generating embeddings for {len(unique_seqs)} unique sequences...")

    mean_embs = []
    first_embs = []
    last_embs = []

    for i, seq in enumerate(unique_seqs):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(unique_seqs)}] {seq[:20]}...")

        inputs = tokenizer(seq, return_tensors="pt", add_special_tokens=True)
        with torch.no_grad():
            outputs = model(**inputs)
            # Shape: [1, seq_len+2, 480] — skip CLS (0) and EOS (-1)
            hidden = outputs.last_hidden_state[0]
            residue_emb = hidden[1:-1, :]  # [seq_len, 480]

            mean_embs.append(residue_emb.mean(dim=0).numpy())
            first_embs.append(residue_emb[0].numpy())
            last_embs.append(residue_emb[-1].numpy())

    cache_data = {
        "sequences": unique_seqs,
        "mean": np.array(mean_embs),
        "first": np.array(first_embs),
        "last": np.array(last_embs),
        "model": ESM_MODEL_NAME,
    }

    ESM_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ESM_CACHE_PATH, "wb") as f:
        pickle.dump(cache_data, f)
    print(f"[ESM2] Cached {len(unique_seqs)} embeddings to {ESM_CACHE_PATH}")

    # Return in requested order
    seq_to_idx = {s: i for i, s in enumerate(unique_seqs)}
    idxs = [seq_to_idx[s] for s in sequences]
    return {
        "mean": cache_data["mean"][idxs],
        "first": cache_data["first"][idxs],
        "last": cache_data["last"][idxs],
    }


def pca_reduce_embeddings(
    embeddings: dict[str, np.ndarray],
    n_components: int = N_PCA_COMPONENTS,
) -> tuple[np.ndarray, dict[str, PCA]]:
    """PCA reduce each embedding type to n_components. Returns (features, pca_models).

    Output shape: (n_samples, 3 * n_components).
    """
    pca_models = {}
    reduced = []

    for key in ["mean", "first", "last"]:
        emb = embeddings[key]
        pca = PCA(n_components=n_components, random_state=42)
        r = pca.fit_transform(emb)
        pca_models[key] = pca
        reduced.append(r)
        var_explained = pca.explained_variance_ratio_.sum()
        print(f"[PCA] {key}: {emb.shape[1]}d -> {n_components}d "
              f"({var_explained:.1%} variance explained)")

    return np.hstack(reduced), pca_models


# ---------------------------------------------------------------------------
# Step 2: Build enhanced dataset
# ---------------------------------------------------------------------------

def build_dataset() -> dict:
    """Build training dataset with hand-crafted + ESM-2 features.

    Returns dict with:
        X_v2: hand-crafted features (n_samples, 72)
        X_esm: ESM-2 PCA features (n_samples, 30)
        X_combined: all features (n_samples, 102)
        y: targets
        groups: sequence group labels for leave-sequence-out CV
        feature_names_v2, feature_names_esm, feature_names_all
    """
    print("[DATA] Loading Paper 1 data...")
    rozans_path = DATA_DIR / "rozans-618-enriched.csv"
    df = pd.read_csv(rozans_path)
    paper1 = df[df["paper"] == "Paper 1 (ACS Biomater 2024)"].copy()

    valid_n = set(N_MOD_BASE.keys())
    valid_c = set(C_MOD_BASE.keys())
    paper1 = paper1[paper1["n_terminal"].isin(valid_n) & paper1["c_terminal"].isin(valid_c)]
    print(f"[DATA] Paper 1 peptides (valid mods): {len(paper1)}")

    # Get unique sequences for ESM-2
    unique_seqs = sorted(paper1["clean_sequence"].unique())
    print(f"[DATA] Unique base sequences: {len(unique_seqs)}")

    # Build MEROPS tables
    print("[DATA] Building MEROPS cleavage tables...")
    freq_tables = build_merops_cleavage_freq()

    # Generate ESM-2 embeddings (one per unique sequence)
    embeddings_raw = generate_esm2_embeddings(unique_seqs)

    # PCA reduce
    esm_reduced, pca_models = pca_reduce_embeddings(embeddings_raw)

    # Map sequence -> ESM PCA features
    seq_to_esm = {seq: esm_reduced[i] for i, seq in enumerate(unique_seqs)}

    # Build training samples: peptide x cell_type
    print("[DATA] Building training samples...")
    v2_rows = []
    esm_rows = []
    targets = []
    groups = []  # sequence index for leave-sequence-out

    seq_to_group = {seq: i for i, seq in enumerate(unique_seqs)}

    for _, row in paper1.iterrows():
        seq = str(row["clean_sequence"])
        n_mod = str(row["n_terminal"])
        c_mod = str(row["c_terminal"])
        n_aa = seq[0] if len(seq) > 0 else "A"
        c_aa = seq[-1] if len(seq) > 0 else "A"

        esm_feats = seq_to_esm[seq]

        for cell_type in CELL_TYPES:
            y = compute_fraction_remaining(n_mod, c_mod, n_aa, c_aa, cell_type)
            targets.append(y)
            groups.append(seq_to_group[seq])

            v2_feats = featurize_one(seq, n_mod, c_mod, cell_type, freq_tables)
            v2_rows.append(v2_feats)
            esm_rows.append(esm_feats)

    X_v2_df = pd.DataFrame(v2_rows)
    X_v2 = X_v2_df.values
    X_esm = np.array(esm_rows)
    y = np.array(targets)
    groups = np.array(groups)

    # ESM feature names
    esm_names = (
        [f"esm_mean_pc{i}" for i in range(N_PCA_COMPONENTS)]
        + [f"esm_nterm_pc{i}" for i in range(N_PCA_COMPONENTS)]
        + [f"esm_cterm_pc{i}" for i in range(N_PCA_COMPONENTS)]
    )

    v2_names = list(X_v2_df.columns)
    all_names = v2_names + esm_names

    X_combined = np.hstack([X_v2, X_esm])

    print(f"[DATA] Samples: {len(y)}")
    print(f"[DATA] v2 features: {X_v2.shape[1]}, ESM features: {X_esm.shape[1]}, "
          f"combined: {X_combined.shape[1]}")
    print(f"[DATA] Target range: [{y.min():.3f}, {y.max():.3f}], mean={y.mean():.3f}")
    print(f"[DATA] Unique groups (sequences): {len(np.unique(groups))}")

    return {
        "X_v2": X_v2,
        "X_esm": X_esm,
        "X_combined": X_combined,
        "y": y,
        "groups": groups,
        "feature_names_v2": v2_names,
        "feature_names_esm": esm_names,
        "feature_names_all": all_names,
        "pca_models": pca_models,
        "unique_sequences": unique_seqs,
    }


# ---------------------------------------------------------------------------
# Step 3: Leave-sequence-out cross-validation
# ---------------------------------------------------------------------------

def leave_sequence_out_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    label: str,
) -> dict:
    """Run leave-sequence-out CV with GBR. Returns metrics dict."""
    n_groups = len(np.unique(groups))
    gkf = GroupKFold(n_splits=n_groups)  # one fold per sequence = leave-one-out

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
# Step 4: Feature importance
# ---------------------------------------------------------------------------

def feature_importance_analysis(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Train GBR on all data with 102 features and report top 20 by importance."""
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE (102 features, full training set)")
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
        marker = " [ESM]" if row["feature"].startswith("esm_") else ""
        print(f"    {i+1:2d}. {row['feature']:40s} {row['importance']:.4f}{marker}")

    esm_in_top20 = fi.head(20)["feature"].str.startswith("esm_").sum()
    esm_total_importance = fi[fi["feature"].str.startswith("esm_")]["importance"].sum()
    print(f"\n  ESM-2 features in top 20: {esm_in_top20}")
    print(f"  ESM-2 total importance:   {esm_total_importance:.4f} "
          f"({esm_total_importance * 100:.1f}%)")

    return fi


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("ExoPred v3 — ESM-2 Embeddings + Leave-Sequence-Out CV")
    print("=" * 60)

    # Build dataset
    data = build_dataset()
    X_v2 = data["X_v2"]
    X_esm = data["X_esm"]
    X_combined = data["X_combined"]
    y = data["y"]
    groups = data["groups"]

    # Leave-sequence-out CV for all 3 models
    print("\n" + "=" * 60)
    print("LEAVE-SEQUENCE-OUT CROSS-VALIDATION")
    print(f"({len(np.unique(groups))} unique sequences, "
          f"{len(y)} total samples)")
    print("=" * 60)

    results_a = leave_sequence_out_cv(
        X_v2, y, groups,
        "Model A: 72 hand-crafted features (v2 baseline)",
    )

    results_b = leave_sequence_out_cv(
        X_combined, y, groups,
        "Model B: 72 hand-crafted + 30 ESM-2 features",
    )

    results_c = leave_sequence_out_cv(
        X_esm, y, groups,
        "Model C: 30 ESM-2 features ONLY",
    )

    # Feature importance on full combined set
    fi = feature_importance_analysis(
        X_combined, y, data["feature_names_all"],
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY — Leave-Sequence-Out Generalization Test")
    print("=" * 60)
    print(f"  {'Model':<45s} {'R2':>8s} {'RMSE':>8s}")
    print(f"  {'-'*45} {'-'*8} {'-'*8}")
    print(f"  {'A: 72 hand-crafted (v2)':<45s} {results_a['overall_r2']:8.4f} {results_a['overall_rmse']:8.4f}")
    print(f"  {'B: 72 hand-crafted + 30 ESM-2':<45s} {results_b['overall_r2']:8.4f} {results_b['overall_rmse']:8.4f}")
    print(f"  {'C: 30 ESM-2 only':<45s} {results_c['overall_r2']:8.4f} {results_c['overall_rmse']:8.4f}")

    delta_r2 = results_b["overall_r2"] - results_a["overall_r2"]
    print(f"\n  ESM-2 delta R2 (B - A): {delta_r2:+.4f}")
    if delta_r2 > 0.01:
        print("  --> ESM-2 embeddings IMPROVE generalization to novel peptides.")
    elif delta_r2 > -0.01:
        print("  --> ESM-2 embeddings have MARGINAL effect on generalization.")
    else:
        print("  --> ESM-2 embeddings HURT generalization (possible overfitting).")

    # Save artifacts
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {
        "model_a_v2_only": results_a,
        "model_b_v2_plus_esm": results_b,
        "model_c_esm_only": results_c,
        "esm_delta_r2": float(delta_r2),
        "n_samples": len(y),
        "n_sequences": len(np.unique(groups)),
        "n_features": {"v2": X_v2.shape[1], "esm": X_esm.shape[1], "combined": X_combined.shape[1]},
    }
    metrics_path = CHECKPOINT_DIR / "v3_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[SAVE] Metrics -> {metrics_path}")

    fi_path = CHECKPOINT_DIR / "v3_feature_importance.csv"
    fi.to_csv(fi_path, index=False)
    print(f"[SAVE] Feature importance -> {fi_path}")


if __name__ == "__main__":
    main()
