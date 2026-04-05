"""
ExoPred multi-task model for exopeptidase degradation prediction.

Architecture: Shared backbone with 3 task-specific heads.
  - Head A: Binary cleavage classification (sigmoid)
  - Head B: Half-life regression (log-scale)
  - Head C: Kinetic curve prediction (multi-output sigmoid)
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ExoPredModel(nn.Module):
    """Multi-task model for exopeptidase degradation prediction.

    Shared backbone -> 3 task heads:
      - Head A: Binary cleavage classification (sigmoid)
      - Head B: Half-life regression (log-scale)
      - Head C: Kinetic curve prediction (multi-output sigmoid)

    Parameters
    ----------
    input_dim : int
        Number of input features.  40 (physico+enzyme+mod) without ESM-2,
        360 with ESM-2 t6, 1320 with ESM-2 t33.
    num_timepoints : int
        Number of kinetic-curve timepoints for Head C (default 6,
        representing 0, 15, 30, 60, 120, 240 min).
    pos_weight : float or None
        Positive-class weight for BCEWithLogitsLoss (Head A) to handle
        class imbalance.  If None, uniform weighting is used.
    """

    def __init__(
        self,
        input_dim: int = 40,
        num_timepoints: int = 6,
        pos_weight: Optional[float] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_timepoints = num_timepoints

        # ---- Shared backbone ----
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # ---- Head A: Binary cleavage ----
        self.head_binary = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            # No sigmoid — BCEWithLogitsLoss expects raw logits
        )

        # ---- Head B: Half-life regression ----
        self.head_halflife = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            # No activation — predicts log10(t_half in minutes)
        )

        # ---- Head C: Kinetic curve ----
        self.head_curve = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_timepoints),
            nn.Sigmoid(),  # fraction remaining at each timepoint
        )

        # ---- Losses ----
        pw = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.loss_binary = nn.BCEWithLogitsLoss(pos_weight=pw)
        self.loss_halflife = nn.MSELoss()
        self.loss_curve = nn.MSELoss()

    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through backbone + all heads.

        Returns dict with keys 'binary', 'halflife', 'curve'.
        """
        shared = self.backbone(x)
        return {
            "binary": self.head_binary(shared).squeeze(-1),
            "halflife": self.head_halflife(shared).squeeze(-1),
            "curve": self.head_curve(shared),
        }

    # ------------------------------------------------------------------

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, Optional[torch.Tensor]],
        task_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute multi-task loss, skipping tasks with None targets.

        Parameters
        ----------
        predictions : dict
            Keys 'binary', 'halflife', 'curve' from forward().
        targets : dict
            Same keys. Values are tensors or None (missing labels).
        task_weights : dict or None
            Per-task scalar weights.  Defaults to
            {'binary': 0.2, 'halflife': 0.3, 'curve': 0.5}.

        Returns
        -------
        total_loss : Tensor  (scalar, differentiable)
        losses_dict : dict   (detached floats for logging)
        """
        if task_weights is None:
            task_weights = {"binary": 0.2, "halflife": 0.3, "curve": 0.5}

        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        losses_dict: Dict[str, float] = {}

        # Head A — binary cleavage
        if targets.get("binary") is not None:
            loss_a = self.loss_binary(
                predictions["binary"], targets["binary"].float()
            )
            total_loss = total_loss + task_weights["binary"] * loss_a
            losses_dict["binary"] = loss_a.item()

        # Head B — half-life regression
        if targets.get("halflife") is not None:
            loss_b = self.loss_halflife(
                predictions["halflife"], targets["halflife"].float()
            )
            total_loss = total_loss + task_weights["halflife"] * loss_b
            losses_dict["halflife"] = loss_b.item()

        # Head C — kinetic curve
        if targets.get("curve") is not None:
            loss_c = self.loss_curve(
                predictions["curve"], targets["curve"].float()
            )
            total_loss = total_loss + task_weights["curve"] * loss_c
            losses_dict["curve"] = loss_c.item()

        losses_dict["total"] = total_loss.item()
        return total_loss, losses_dict

    # ------------------------------------------------------------------

    def predict(
        self, x: torch.Tensor, enforce_monotonic: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Inference-mode forward pass with post-processing.

        - Head A: applies sigmoid to return probabilities.
        - Head B: returns 10^output (predicted t_half in minutes).
        - Head C: optionally enforces monotonically decreasing curve.
        """
        self.eval()
        with torch.no_grad():
            preds = self.forward(x)

        out: Dict[str, torch.Tensor] = {}

        # Binary probability
        out["binary_prob"] = torch.sigmoid(preds["binary"])

        # Half-life in minutes
        out["halflife_min"] = torch.pow(10.0, preds["halflife"])

        # Kinetic curve (fraction remaining)
        curve = preds["curve"]
        if enforce_monotonic and curve.dim() == 2 and curve.shape[1] > 1:
            # Enforce: each timepoint <= previous timepoint
            mono = torch.zeros_like(curve)
            mono[:, 0] = curve[:, 0]
            for t in range(1, curve.shape[1]):
                mono[:, t] = torch.min(curve[:, t], mono[:, t - 1])
            curve = mono
        out["curve"] = curve

        return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ExoPredDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for ExoPred training.

    Handles multi-task labels: each sample may have labels for 1, 2, or 3
    tasks.  Missing task labels are stored as NaN and surfaced as None in
    the collate step.

    Parameters
    ----------
    features_df : pd.DataFrame
        Feature matrix.  Must contain numeric feature columns.  May also
        contain label columns:
          - ``label_binary`` (0/1)       for Head A
          - ``label_halflife`` (float)   for Head B  (log10 minutes)
          - ``label_curve_0`` ... ``label_curve_N``  for Head C
    task : str
        Which task labels to expose: "binary", "halflife", "curve", or
        "all" (default).
    feature_columns : list[str] or None
        Explicit list of feature column names.  If None, inferred as all
        numeric columns that don't start with ``label_``.
    """

    # Column prefixes used for labels
    _LABEL_BINARY = "label_binary"
    _LABEL_HALFLIFE = "label_halflife"
    _LABEL_CURVE_PREFIX = "label_curve_"

    def __init__(
        self,
        features_df: pd.DataFrame,
        task: str = "all",
        feature_columns: Optional[list] = None,
    ):
        super().__init__()
        assert task in ("binary", "halflife", "curve", "all")
        self.task = task

        # Identify feature columns
        if feature_columns is not None:
            self.feature_cols = list(feature_columns)
        else:
            self.feature_cols = [
                c for c in features_df.columns
                if not c.startswith("label_") and features_df[c].dtype.kind in "iufb"
            ]

        # Store features as float32 tensor
        self.X = torch.tensor(
            features_df[self.feature_cols].values, dtype=torch.float32
        )

        # Extract label columns when present
        self.y_binary = None
        self.y_halflife = None
        self.y_curve = None

        if task in ("binary", "all") and self._LABEL_BINARY in features_df.columns:
            vals = features_df[self._LABEL_BINARY].values.astype(np.float32)
            self.y_binary = torch.tensor(vals, dtype=torch.float32)

        if task in ("halflife", "all") and self._LABEL_HALFLIFE in features_df.columns:
            vals = features_df[self._LABEL_HALFLIFE].values.astype(np.float32)
            self.y_halflife = torch.tensor(vals, dtype=torch.float32)

        if task in ("curve", "all"):
            curve_cols = sorted(
                [c for c in features_df.columns if c.startswith(self._LABEL_CURVE_PREFIX)]
            )
            if curve_cols:
                vals = features_df[curve_cols].values.astype(np.float32)
                self.y_curve = torch.tensor(vals, dtype=torch.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> dict:
        item = {"features": self.X[idx]}

        if self.y_binary is not None:
            val = self.y_binary[idx]
            item["binary"] = val if not torch.isnan(val) else None
        else:
            item["binary"] = None

        if self.y_halflife is not None:
            val = self.y_halflife[idx]
            item["halflife"] = val if not torch.isnan(val) else None
        else:
            item["halflife"] = None

        if self.y_curve is not None:
            val = self.y_curve[idx]
            item["curve"] = val if not torch.any(torch.isnan(val)) else None
        else:
            item["curve"] = None

        return item


def exopred_collate_fn(batch: list) -> dict:
    """Custom collate that stacks features and groups task labels.

    For each task, if *all* samples in the batch have a label the result is
    a stacked tensor; otherwise it is None (the task is skipped for that
    batch).
    """
    features = torch.stack([b["features"] for b in batch])

    targets: Dict[str, Optional[torch.Tensor]] = {}
    for key in ("binary", "halflife", "curve"):
        vals = [b[key] for b in batch]
        if all(v is not None for v in vals):
            targets[key] = torch.stack(vals)
        else:
            targets[key] = None

    return {"features": features, "targets": targets}
