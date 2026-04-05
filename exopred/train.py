"""
ExoPred training script.

Usage:
    python3 -m exopred.train --phase 1              # Train on public data only
    python3 -m exopred.train --phase 2 --rozans PATH # Fine-tune with Sam's data
    python3 -m exopred.train --eval                   # Evaluate saved model

Phase 1 uses physicochemical + enzyme + modification features (dim=40).
Phase 2 adds ESM-2 embeddings and Sam's kinetic-curve data.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split

from exopred.config import CHECKPOINT_DIR, PROCESSED_DIR, TRAINING_DIR
from exopred.model import ExoPredDataset, ExoPredModel, exopred_collate_fn

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

CONFIG = {
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "max_epochs": 100,
    "patience": 10,
    "task_weights": {"binary": 0.3, "halflife": 0.7, "curve": 0.0},  # Phase 1
    "val_fraction": 0.1,
    "test_fraction": 0.1,
    "random_seed": 42,
}

# Known amino-acid properties for physicochemical features
AA_PROPERTIES = {
    # aa: (mol_weight, pI, hydrophobicity_KD, charge_pH7, hbond_donors, hbond_acceptors)
    "A": (89.09, 6.00, 1.8, 0, 1, 1),
    "R": (174.20, 10.76, -4.5, 1, 4, 1),
    "N": (132.12, 5.41, -3.5, 0, 2, 2),
    "D": (133.10, 2.77, -3.5, -1, 1, 3),
    "C": (121.16, 5.07, 2.5, 0, 1, 1),
    "E": (147.13, 3.22, -3.5, -1, 1, 3),
    "Q": (146.15, 5.65, -3.5, 0, 2, 2),
    "G": (75.03, 5.97, -0.4, 0, 1, 1),
    "H": (155.16, 7.59, -3.2, 0, 2, 2),
    "I": (131.17, 6.02, 4.5, 0, 1, 1),
    "L": (131.17, 5.98, 3.8, 0, 1, 1),
    "K": (146.19, 9.74, -3.9, 1, 2, 1),
    "M": (149.21, 5.74, 1.9, 0, 1, 1),
    "F": (165.19, 5.48, 2.8, 0, 1, 1),
    "P": (115.13, 6.30, -1.6, 0, 0, 1),
    "S": (105.09, 5.68, -0.8, 0, 2, 2),
    "T": (119.12, 5.60, -0.7, 0, 2, 2),
    "W": (204.23, 5.89, -0.9, 0, 2, 1),
    "Y": (181.19, 5.66, -1.3, 0, 2, 2),
    "V": (117.15, 5.96, 4.2, 0, 1, 1),
}

# Exopeptidase families for one-hot encoding (11 families from MEROPS)
ENZYME_FAMILIES = [
    "C01", "M01", "M14", "M17", "M18", "M24", "M28",
    "S09", "S28", "other", "unknown",
]

# ---------------------------------------------------------------------------
# Featurization helpers
# ---------------------------------------------------------------------------


def physicochemical_features(seq: str) -> np.ndarray:
    """Compute 18 physicochemical features for a peptide sequence.

    Features (18 total):
      0-5:   mean of (mw, pI, hydro, charge, hbd, hba) across residues
      6-11:  std of same 6 properties
      12:    sequence length
      13:    net charge at pH 7
      14:    grand average hydropathy (GRAVY)
      15:    fraction hydrophobic residues
      16:    molecular weight of full peptide (sum - (n-1)*18.015)
      17:    isoelectric point estimate (mean of residue pIs)
    """
    props = []
    for aa in seq.upper():
        if aa in AA_PROPERTIES:
            props.append(AA_PROPERTIES[aa])
    if not props:
        return np.zeros(18, dtype=np.float32)

    arr = np.array(props, dtype=np.float32)
    means = arr.mean(axis=0)  # 6
    stds = arr.std(axis=0)    # 6

    length = len(seq)
    net_charge = arr[:, 3].sum()
    gravy = arr[:, 2].mean()
    frac_hydro = np.mean(arr[:, 2] > 0)
    mw_full = arr[:, 0].sum() - (length - 1) * 18.015
    pI_est = arr[:, 1].mean()

    return np.concatenate([
        means, stds,
        np.array([length, net_charge, gravy, frac_hydro, mw_full, pI_est], dtype=np.float32),
    ])


def enzyme_features(protease_family: Optional[str] = None) -> np.ndarray:
    """Compute 16 enzyme features: 11 one-hot family + 5 engineered.

    Engineered features:
      0: is_aminopeptidase (M01, M17, M18, M24)
      1: is_carboxypeptidase (M14, M28)
      2: is_dipeptidylpeptidase (S09)
      3: is_cysteine_protease (C01)
      4: is_serine_protease (S09, S28)
    """
    # One-hot for family
    onehot = np.zeros(len(ENZYME_FAMILIES), dtype=np.float32)
    fam = str(protease_family).strip() if protease_family else "unknown"
    if fam in ENZYME_FAMILIES:
        onehot[ENZYME_FAMILIES.index(fam)] = 1.0
    elif fam != "unknown":
        onehot[ENZYME_FAMILIES.index("other")] = 1.0
    else:
        onehot[ENZYME_FAMILIES.index("unknown")] = 1.0

    # Engineered
    eng = np.zeros(5, dtype=np.float32)
    eng[0] = float(fam in ("M01", "M17", "M18", "M24"))
    eng[1] = float(fam in ("M14", "M28"))
    eng[2] = float(fam == "S09")
    eng[3] = float(fam == "C01")
    eng[4] = float(fam in ("S09", "S28"))

    return np.concatenate([onehot, eng])


def modification_features(
    lin_cyc: str = "Linear",
    chiral: str = "L",
    chem_mod: str = "None",
    nter: str = "Free",
    cter: str = "Free",
    nature: str = "Natural",
) -> np.ndarray:
    """Compute 6 binary modification features.

    0: is_cyclic
    1: is_d_amino_acids
    2: has_chemical_modification
    3: nter_modified
    4: cter_modified
    5: is_synthetic
    """
    return np.array([
        float(str(lin_cyc).lower() == "cyclic"),
        float(str(chiral).upper() == "D"),
        float(str(chem_mod).lower() not in ("none", "nan", "")),
        float(str(nter).lower() not in ("free", "nan", "")),
        float(str(cter).lower() not in ("free", "nan", "")),
        float(str(nature).lower() in ("synthetic", "designed")),
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Data loading & featurization
# ---------------------------------------------------------------------------


THREE_TO_ONE = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
}


def load_merops_binary() -> pd.DataFrame:
    """Load MEROPS exopeptidase cleavages as binary-labeled feature matrix.

    Every MEROPS entry is a positive cleavage example (label=1).
    We generate synthetic negatives by shuffling the P1-P1' context.
    """
    path = PROCESSED_DIR / "merops_exopeptidase_cleavages.csv"
    if not path.exists():
        print(f"[WARN] MEROPS data not found at {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    rows = []

    for _, r in df.iterrows():
        # Build a short peptide from the cleavage context (P4..P4')
        context_cols = ["P4", "P3", "P2", "P1", "P1prime", "P2prime", "P3prime", "P4prime"]
        residues = []
        for c in context_cols:
            val = str(r.get(c, "")).strip()
            # Convert 3-letter codes to 1-letter
            if val in THREE_TO_ONE:
                residues.append(THREE_TO_ONE[val])
            elif len(val) == 1 and val.upper() in AA_PROPERTIES:
                residues.append(val.upper())
        seq = "".join(residues) if residues else ""
        if len(seq) < 2:
            continue

        fam = str(r.get("protease_family", "unknown"))

        feat = np.concatenate([
            physicochemical_features(seq),
            enzyme_features(fam),
            modification_features(),  # defaults (linear, L, no mods)
        ])
        row = {f"f{i}": feat[i] for i in range(len(feat))}
        row["label_binary"] = 1.0
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    # Generate same number of negatives by shuffling sequences
    rng = np.random.RandomState(CONFIG["random_seed"])
    neg_rows = []
    for pos in rows:
        neg = dict(pos)
        neg["label_binary"] = 0.0
        # Shuffle physicochemical features (first 18) to break real signal
        phys = np.array([neg[f"f{i}"] for i in range(18)])
        rng.shuffle(phys)
        for i in range(18):
            neg[f"f{i}"] = phys[i]
        neg_rows.append(neg)

    return pd.DataFrame(rows + neg_rows)


def load_dppiv_binary() -> pd.DataFrame:
    """Load DPP-IV benchmark dataset as binary-labeled feature matrix."""
    path = PROCESSED_DIR / "dppiv_benchmark.csv"
    if not path.exists():
        print(f"[WARN] DPP-IV data not found at {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    rows = []

    for _, r in df.iterrows():
        seq = str(r.get("sequence", ""))
        if len(seq) < 2:
            continue
        label = float(r.get("label", 0))

        feat = np.concatenate([
            physicochemical_features(seq),
            enzyme_features("S09"),  # DPP-IV is family S09
            modification_features(),
        ])
        row = {f"f{i}": feat[i] for i in range(len(feat))}
        row["label_binary"] = label
        # Preserve original split if available
        row["_split"] = str(r.get("split", "train"))
        rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def load_peplife_halflife() -> pd.DataFrame:
    """Load PEPlife2 half-life data as regression-labeled feature matrix."""
    path = PROCESSED_DIR / "peplife2_combined.csv"
    if not path.exists():
        print(f"[WARN] PEPlife2 data not found at {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    rows = []

    for _, r in df.iterrows():
        seq = str(r.get("seq", ""))
        if len(seq) < 2:
            continue
        hl = r.get("half_life")
        if pd.isna(hl) or hl is None:
            continue
        try:
            hl = float(hl)
        except (ValueError, TypeError):
            continue
        if hl <= 0:
            continue

        # Determine enzyme family from protease column
        protease = str(r.get("protease", "unknown")).lower()
        if "dpp" in protease:
            fam = "S09"
        elif "aminopeptidase" in protease:
            fam = "M01"
        elif "carboxypeptidase" in protease:
            fam = "M14"
        else:
            fam = "unknown"

        feat = np.concatenate([
            physicochemical_features(seq),
            enzyme_features(fam),
            modification_features(
                lin_cyc=str(r.get("lin_cyc", "Linear")),
                chiral=str(r.get("chiral", "L")),
                chem_mod=str(r.get("chem_mod", "None")),
                nter=str(r.get("nter", "Free")),
                cter=str(r.get("cter", "Free")),
                nature=str(r.get("nature", "Natural")),
            ),
        ])
        row = {f"f{i}": feat[i] for i in range(len(feat))}
        row["label_halflife"] = np.log10(hl)  # log10(minutes)
        rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Train / eval helpers
# ---------------------------------------------------------------------------


def split_dataset(
    df: pd.DataFrame,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    split_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a feature DataFrame into train / val / test.

    If *split_col* exists and contains 'train'/'val'/'test', use that.
    Otherwise do a random split.
    """
    if split_col and split_col in df.columns:
        train = df[df[split_col] == "train"].drop(columns=[split_col])
        val = df[df[split_col] == "val"].drop(columns=[split_col])
        test = df[df[split_col] == "test"].drop(columns=[split_col])
        # If val or test is empty, carve from train
        if len(val) == 0 or len(test) == 0:
            rng = np.random.RandomState(seed)
            idx = rng.permutation(len(train))
            n_val = max(1, int(len(train) * val_frac))
            n_test = max(1, int(len(train) * test_frac))
            val = train.iloc[idx[:n_val]]
            test = train.iloc[idx[n_val:n_val + n_test]]
            train = train.iloc[idx[n_val + n_test:]]
        return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

    # Random split
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(df))
    n_val = max(1, int(len(df) * val_frac))
    n_test = max(1, int(len(df) * test_frac))
    val = df.iloc[idx[:n_val]]
    test = df.iloc[idx[n_val:n_val + n_test]]
    train = df.iloc[idx[n_val + n_test:]]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def train_one_epoch(
    model: ExoPredModel,
    loaders: Dict[str, DataLoader],
    optimizer: torch.optim.Optimizer,
    task_weights: Dict[str, float],
) -> Dict[str, float]:
    """Train for one epoch, alternating mini-batches across task loaders."""
    model.train()
    epoch_losses: Dict[str, list] = {"total": []}

    # Build iterators for each available loader
    iters = {name: iter(loader) for name, loader in loaders.items()}
    active = set(iters.keys())

    while active:
        for name in list(active):
            try:
                batch = next(iters[name])
            except StopIteration:
                active.discard(name)
                continue

            x = batch["features"]
            targets = batch["targets"]

            optimizer.zero_grad()
            preds = model(x)
            loss, loss_dict = model.compute_loss(preds, targets, task_weights)

            if loss.requires_grad:
                loss.backward()
                optimizer.step()

            epoch_losses["total"].append(loss_dict["total"])
            for k, v in loss_dict.items():
                epoch_losses.setdefault(k, []).append(v)

    return {k: float(np.mean(v)) for k, v in epoch_losses.items() if v}


@torch.no_grad()
def evaluate(
    model: ExoPredModel,
    loaders: Dict[str, DataLoader],
    task_weights: Dict[str, float],
) -> Dict[str, float]:
    """Evaluate model on validation/test loaders."""
    model.eval()
    all_losses: Dict[str, list] = {"total": []}

    for name, loader in loaders.items():
        for batch in loader:
            x = batch["features"]
            targets = batch["targets"]
            preds = model(x)
            _, loss_dict = model.compute_loss(preds, targets, task_weights)
            all_losses["total"].append(loss_dict["total"])
            for k, v in loss_dict.items():
                all_losses.setdefault(k, []).append(v)

    return {k: float(np.mean(v)) for k, v in all_losses.items() if v}


@torch.no_grad()
def compute_metrics(
    model: ExoPredModel,
    loader: DataLoader,
) -> Dict[str, float]:
    """Compute task-specific metrics on a DataLoader.

    Head A: accuracy, AUC-ROC, F1
    Head B: R2, RMSE, MAE on log10(t_half)
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, mean_absolute_error,
        mean_squared_error, r2_score, roc_auc_score,
    )

    model.eval()
    all_binary_pred, all_binary_true = [], []
    all_hl_pred, all_hl_true = [], []

    for batch in loader:
        x = batch["features"]
        preds = model(x)
        targets = batch["targets"]

        if targets.get("binary") is not None:
            prob = torch.sigmoid(preds["binary"]).cpu().numpy()
            true = targets["binary"].cpu().numpy()
            all_binary_pred.append(prob)
            all_binary_true.append(true)

        if targets.get("halflife") is not None:
            pred_hl = preds["halflife"].cpu().numpy()
            true_hl = targets["halflife"].cpu().numpy()
            all_hl_pred.append(pred_hl)
            all_hl_true.append(true_hl)

    metrics: Dict[str, float] = {}

    # Head A metrics
    if all_binary_pred:
        y_prob = np.concatenate(all_binary_pred)
        y_true = np.concatenate(all_binary_true)
        y_pred = (y_prob >= 0.5).astype(int)
        metrics["binary_acc"] = float(accuracy_score(y_true, y_pred))
        metrics["binary_f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        try:
            metrics["binary_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["binary_auc"] = float("nan")

    # Head B metrics
    if all_hl_pred:
        y_pred_hl = np.concatenate(all_hl_pred)
        y_true_hl = np.concatenate(all_hl_true)
        metrics["halflife_r2"] = float(r2_score(y_true_hl, y_pred_hl))
        metrics["halflife_rmse"] = float(np.sqrt(mean_squared_error(y_true_hl, y_pred_hl)))
        metrics["halflife_mae"] = float(mean_absolute_error(y_true_hl, y_pred_hl))

    return metrics


# ---------------------------------------------------------------------------
# Phase 1 training
# ---------------------------------------------------------------------------


def train_phase1() -> None:
    """Phase 1: train on public data (MEROPS + DPP-IV + PEPlife2)."""
    torch.manual_seed(CONFIG["random_seed"])
    np.random.seed(CONFIG["random_seed"])

    print("=" * 60)
    print("ExoPred Phase 1 Training")
    print("=" * 60)

    # --- Load and featurize data ---
    print("\n[1/6] Loading MEROPS binary data...")
    merops_df = load_merops_binary()
    print(f"  MEROPS samples: {len(merops_df)}")

    print("[2/6] Loading DPP-IV binary data...")
    dppiv_df = load_dppiv_binary()
    print(f"  DPP-IV samples: {len(dppiv_df)}")

    print("[3/6] Loading PEPlife2 half-life data...")
    peplife_df = load_peplife_halflife()
    print(f"  PEPlife2 samples: {len(peplife_df)}")

    # --- Build feature columns list ---
    # All feature DataFrames should have f0..f39 columns
    n_features = 18 + 16 + 6  # physicochemical + enzyme + modification = 40
    feat_cols = [f"f{i}" for i in range(n_features)]
    input_dim = n_features
    print(f"\n  Feature dimension: {input_dim}")

    # --- Combine binary data ---
    binary_dfs = []
    if len(merops_df) > 0:
        binary_dfs.append(merops_df)
    if len(dppiv_df) > 0:
        # Use DPP-IV's original split if available
        split_col = "_split" if "_split" in dppiv_df.columns else None
        binary_dfs.append(dppiv_df)

    if not binary_dfs and len(peplife_df) == 0:
        print("\n[ERROR] No training data found. Ensure processed CSVs exist in:")
        print(f"  {PROCESSED_DIR}")
        sys.exit(1)

    # --- Split datasets ---
    print("[4/6] Splitting datasets...")
    train_loaders: Dict[str, DataLoader] = {}
    val_loaders: Dict[str, DataLoader] = {}
    test_loaders: Dict[str, DataLoader] = {}

    if binary_dfs:
        binary_all = pd.concat(binary_dfs, ignore_index=True)
        # Drop _split column if present (we'll do our own split on combined)
        if "_split" in binary_all.columns:
            binary_all = binary_all.drop(columns=["_split"])
        bin_train, bin_val, bin_test = split_dataset(
            binary_all,
            val_frac=CONFIG["val_fraction"],
            test_frac=CONFIG["test_fraction"],
            seed=CONFIG["random_seed"],
        )
        print(f"  Binary — train: {len(bin_train)}, val: {len(bin_val)}, test: {len(bin_test)}")

        ds_train = ExoPredDataset(bin_train, task="binary", feature_columns=feat_cols)
        ds_val = ExoPredDataset(bin_val, task="binary", feature_columns=feat_cols)
        ds_test = ExoPredDataset(bin_test, task="binary", feature_columns=feat_cols)

        train_loaders["binary"] = DataLoader(
            ds_train, batch_size=CONFIG["batch_size"], shuffle=True,
            collate_fn=exopred_collate_fn,
        )
        val_loaders["binary"] = DataLoader(
            ds_val, batch_size=CONFIG["batch_size"], shuffle=False,
            collate_fn=exopred_collate_fn,
        )
        test_loaders["binary"] = DataLoader(
            ds_test, batch_size=CONFIG["batch_size"], shuffle=False,
            collate_fn=exopred_collate_fn,
        )

    if len(peplife_df) > 0:
        hl_train, hl_val, hl_test = split_dataset(
            peplife_df,
            val_frac=CONFIG["val_fraction"],
            test_frac=CONFIG["test_fraction"],
            seed=CONFIG["random_seed"],
        )
        print(f"  Half-life — train: {len(hl_train)}, val: {len(hl_val)}, test: {len(hl_test)}")

        ds_hl_train = ExoPredDataset(hl_train, task="halflife", feature_columns=feat_cols)
        ds_hl_val = ExoPredDataset(hl_val, task="halflife", feature_columns=feat_cols)
        ds_hl_test = ExoPredDataset(hl_test, task="halflife", feature_columns=feat_cols)

        train_loaders["halflife"] = DataLoader(
            ds_hl_train, batch_size=CONFIG["batch_size"], shuffle=True,
            collate_fn=exopred_collate_fn,
        )
        val_loaders["halflife"] = DataLoader(
            ds_hl_val, batch_size=CONFIG["batch_size"], shuffle=False,
            collate_fn=exopred_collate_fn,
        )
        test_loaders["halflife"] = DataLoader(
            ds_hl_test, batch_size=CONFIG["batch_size"], shuffle=False,
            collate_fn=exopred_collate_fn,
        )

    # --- Initialize model ---
    print(f"[5/6] Initializing ExoPredModel(input_dim={input_dim})...")
    # Compute pos_weight for binary task (handle imbalance)
    pos_weight = None
    if binary_dfs:
        n_pos = binary_all["label_binary"].sum()
        n_neg = len(binary_all) - n_pos
        if n_pos > 0 and n_neg > 0:
            pos_weight = float(n_neg / n_pos)

    model = ExoPredModel(input_dim=input_dim, pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    task_weights = CONFIG["task_weights"]

    # --- Training loop ---
    print(f"[6/6] Training (max {CONFIG['max_epochs']} epochs, patience {CONFIG['patience']})...\n")
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    no_improve = 0
    history = []

    t0 = time.time()

    for epoch in range(1, CONFIG["max_epochs"] + 1):
        # Train
        train_loss = train_one_epoch(model, train_loaders, optimizer, task_weights)

        # Validate
        val_loss = evaluate(model, val_loaders, task_weights)

        # Scheduler step
        scheduler.step(val_loss.get("total", 0.0))

        # Logging
        lr_now = optimizer.param_groups[0]["lr"]
        parts = [f"Epoch {epoch:3d}"]
        parts.append(f"train_loss={train_loss.get('total', 0):.4f}")
        parts.append(f"val_loss={val_loss.get('total', 0):.4f}")
        for k in ("binary", "halflife"):
            if k in train_loss:
                parts.append(f"train_{k}={train_loss[k]:.4f}")
            if k in val_loss:
                parts.append(f"val_{k}={val_loss[k]:.4f}")
        parts.append(f"lr={lr_now:.1e}")
        print("  " + " | ".join(parts))

        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in train_loss.items()},
                         **{f"val_{k}": v for k, v in val_loss.items()}, "lr": lr_now})

        # Early stopping
        vl = val_loss.get("total", 0.0)
        if vl < best_val_loss:
            best_val_loss = vl
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= CONFIG["patience"]:
                print(f"\n  Early stopping at epoch {epoch} (best epoch: {best_epoch})")
                break

    elapsed = time.time() - t0
    print(f"\n  Training completed in {elapsed:.1f}s")

    # --- Restore best model & evaluate on test set ---
    if best_state is not None:
        model.load_state_dict(best_state)

    print("\n--- Test Set Evaluation ---")
    test_metrics = compute_metrics(model, DataLoader(
        # Combine all test loaders into one pass
        torch.utils.data.ConcatDataset(
            [loader.dataset for loader in test_loaders.values()]
        ),
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        collate_fn=exopred_collate_fn,
    ))

    for k, v in sorted(test_metrics.items()):
        print(f"  {k}: {v:.4f}")

    # Check iDPPIV baseline
    if "binary_acc" in test_metrics:
        baseline = 0.867
        status = "PASS" if test_metrics["binary_acc"] >= baseline else "BELOW BASELINE"
        print(f"\n  iDPPIV baseline (ACC >= {baseline}): {status}")

    # --- Save checkpoint ---
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / "exopred_phase1.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": CONFIG,
        "input_dim": input_dim,
        "pos_weight": pos_weight,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }, ckpt_path)
    print(f"\n  Model saved to {ckpt_path}")

    # Save metrics
    metrics_path = CHECKPOINT_DIR / "phase1_metrics.json"
    metrics_out = {
        "test_metrics": test_metrics,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "training_time_s": elapsed,
        "num_epochs": len(history),
        "config": CONFIG,
        "history": history,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2, default=str)
    print(f"  Metrics saved to {metrics_path}")


# ---------------------------------------------------------------------------
# Phase 2 (stub)
# ---------------------------------------------------------------------------


def train_phase2(rozans_path: str) -> None:
    """Phase 2: fine-tune with Sam's kinetic curve data + optional ESM-2.

    Activated when Sam provides LC-MS degradation curves via
    rozans_template.csv format.
    """
    torch.manual_seed(CONFIG["random_seed"])
    np.random.seed(CONFIG["random_seed"])

    print("=" * 60)
    print("ExoPred Phase 2 Training (Fine-tune)")
    print("=" * 60)

    # Load Phase 1 checkpoint
    ckpt_path = CHECKPOINT_DIR / "exopred_phase1.pt"
    if not ckpt_path.exists():
        print(f"[ERROR] Phase 1 checkpoint not found at {ckpt_path}")
        print("  Run --phase 1 first.")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    phase1_dim = ckpt["input_dim"]

    # Load Sam's data
    rozans_df = pd.read_csv(rozans_path)
    print(f"  Loaded {len(rozans_df)} samples from {rozans_path}")

    # TODO: Featurize Sam's data (sequence -> physicochemical + enzyme + mod)
    # TODO: Optionally compute ESM-2 embeddings to expand input_dim
    #   from transformers import EsmModel, EsmTokenizer
    #   esm = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")  # 320-dim
    #   new_input_dim = phase1_dim + 320

    # For now, use same feature dim as Phase 1
    input_dim = phase1_dim

    # Initialize model and load Phase 1 weights
    model = ExoPredModel(input_dim=input_dim, pos_weight=ckpt.get("pos_weight"))
    model.load_state_dict(ckpt["model_state_dict"])

    # Unfreeze all layers for full fine-tuning
    for param in model.parameters():
        param.requires_grad = True

    # Phase 2 task weights (all heads active)
    phase2_weights = {"binary": 0.2, "halflife": 0.3, "curve": 0.5}

    # TODO: Build Task C dataset from Sam's kinetic curve columns
    # Expected columns: sequence, protease, t0, t15, t30, t60, t120, t240
    # Each t_X is fraction remaining at X minutes

    # TODO: Train all 3 heads jointly
    # optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    # ... training loop similar to Phase 1 ...

    print("\n  [STUB] Phase 2 training not yet implemented.")
    print("  Waiting for Sam's LC-MS data in rozans_template.csv format.")
    print(f"  Phase 1 model is ready at {ckpt_path}")

    # Save Phase 2 checkpoint (placeholder)
    # ckpt2_path = CHECKPOINT_DIR / "exopred_phase2.pt"
    # torch.save({...}, ckpt2_path)


# ---------------------------------------------------------------------------
# Eval-only mode
# ---------------------------------------------------------------------------


def eval_only() -> None:
    """Evaluate a saved model checkpoint on test data."""
    torch.manual_seed(CONFIG["random_seed"])
    np.random.seed(CONFIG["random_seed"])

    print("=" * 60)
    print("ExoPred Evaluation")
    print("=" * 60)

    # Try Phase 2 first, fall back to Phase 1
    ckpt_path = CHECKPOINT_DIR / "exopred_phase2.pt"
    if not ckpt_path.exists():
        ckpt_path = CHECKPOINT_DIR / "exopred_phase1.pt"
    if not ckpt_path.exists():
        print(f"[ERROR] No checkpoint found in {CHECKPOINT_DIR}")
        sys.exit(1)

    print(f"  Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    input_dim = ckpt["input_dim"]

    model = ExoPredModel(input_dim=input_dim, pos_weight=ckpt.get("pos_weight"))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"  Best epoch: {ckpt.get('best_epoch')}")
    print(f"  Best val loss: {ckpt.get('best_val_loss', 'N/A')}")

    # Reload test data
    n_features = input_dim
    feat_cols = [f"f{i}" for i in range(n_features)]

    test_datasets = []

    # Binary test
    merops_df = load_merops_binary()
    dppiv_df = load_dppiv_binary()
    binary_dfs = []
    if len(merops_df) > 0:
        binary_dfs.append(merops_df)
    if len(dppiv_df) > 0:
        if "_split" in dppiv_df.columns:
            dppiv_df = dppiv_df.drop(columns=["_split"])
        binary_dfs.append(dppiv_df)
    if binary_dfs:
        binary_all = pd.concat(binary_dfs, ignore_index=True)
        _, _, bin_test = split_dataset(
            binary_all, val_frac=CONFIG["val_fraction"],
            test_frac=CONFIG["test_fraction"], seed=CONFIG["random_seed"],
        )
        test_datasets.append(ExoPredDataset(bin_test, task="binary", feature_columns=feat_cols))

    # Half-life test
    peplife_df = load_peplife_halflife()
    if len(peplife_df) > 0:
        _, _, hl_test = split_dataset(
            peplife_df, val_frac=CONFIG["val_fraction"],
            test_frac=CONFIG["test_fraction"], seed=CONFIG["random_seed"],
        )
        test_datasets.append(ExoPredDataset(hl_test, task="halflife", feature_columns=feat_cols))

    if not test_datasets:
        print("[ERROR] No test data available.")
        sys.exit(1)

    combined_test = torch.utils.data.ConcatDataset(test_datasets)
    test_loader = DataLoader(
        combined_test, batch_size=CONFIG["batch_size"],
        shuffle=False, collate_fn=exopred_collate_fn,
    )

    print("\n--- Test Set Metrics ---")
    metrics = compute_metrics(model, test_loader)
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}")

    if "binary_acc" in metrics:
        baseline = 0.867
        status = "PASS" if metrics["binary_acc"] >= baseline else "BELOW BASELINE"
        print(f"\n  iDPPIV baseline (ACC >= {baseline}): {status}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="ExoPred training script")
    parser.add_argument("--phase", type=int, choices=[1, 2], default=None,
                        help="Training phase (1=public data, 2=fine-tune with Sam's data)")
    parser.add_argument("--rozans", type=str, default=None,
                        help="Path to Sam's data CSV (required for --phase 2)")
    parser.add_argument("--eval", action="store_true",
                        help="Evaluate saved model on test data")
    args = parser.parse_args()

    if args.eval:
        eval_only()
    elif args.phase == 1:
        train_phase1()
    elif args.phase == 2:
        if args.rozans is None:
            parser.error("--rozans PATH is required for Phase 2")
        train_phase2(args.rozans)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
