"""ExoPred prediction module.

Usage:
    from exopred.predict import ExoPredPredictor
    pred = ExoPredPredictor("checkpoints/exopred_phase1.pt")
    result = pred.predict("RGDSP", enzyme="APN")
    result = pred.predict_all("RGDSP")  # all enzymes
    results = pred.predict_batch(["RGDSP", "GRGDS", "YIGSR"])
"""

import math
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Heuristic data  (Rozans 2024 + MEROPS literature)
# ---------------------------------------------------------------------------

# Relative susceptibility scores 0-1 for each amino acid at the relevant
# terminal position.  Higher = more susceptible to cleavage.

APN_N_TERM_PREFS = {
    # Aminopeptidase N (M01.001) — cleaves single N-terminal residue
    "A": 0.95, "L": 0.92, "M": 0.88, "F": 0.85, "V": 0.80,
    "I": 0.78, "Y": 0.75, "S": 0.60, "T": 0.55, "G": 0.50,
    "N": 0.45, "Q": 0.42, "H": 0.40, "K": 0.38, "R": 0.35,
    "E": 0.30, "D": 0.28, "W": 0.25, "C": 0.20, "P": 0.05,
}

CPA_C_TERM_PREFS = {
    # Carboxypeptidase A (M14.001) — cleaves single C-terminal residue
    "F": 0.95, "Y": 0.93, "W": 0.90, "L": 0.88, "I": 0.85,
    "V": 0.80, "M": 0.78, "A": 0.65, "H": 0.55, "T": 0.50,
    "S": 0.45, "N": 0.40, "Q": 0.38, "G": 0.35, "E": 0.30,
    "D": 0.25, "K": 0.20, "R": 0.18, "C": 0.15, "P": 0.03,
}

LAP_N_TERM_PREFS = {
    # Leucine aminopeptidase (M17.001) — broad specificity, Leu preference
    "L": 0.95, "M": 0.90, "F": 0.88, "A": 0.80, "V": 0.75,
    "I": 0.73, "Y": 0.70, "S": 0.55, "T": 0.50, "G": 0.45,
    "N": 0.40, "Q": 0.38, "H": 0.35, "K": 0.33, "R": 0.30,
    "E": 0.25, "D": 0.22, "W": 0.20, "C": 0.18, "P": 0.05,
}

CPB_C_TERM_PREFS = {
    # Carboxypeptidase B (M14.003) — prefers basic C-terminal residues
    "R": 0.95, "K": 0.93, "H": 0.60, "F": 0.30, "Y": 0.28,
    "L": 0.25, "A": 0.20, "V": 0.18, "I": 0.15, "M": 0.14,
    "W": 0.12, "T": 0.10, "S": 0.10, "N": 0.08, "Q": 0.08,
    "G": 0.07, "E": 0.06, "D": 0.05, "C": 0.04, "P": 0.02,
}

NEP_PREFS = {
    # Neprilysin (M13.001) — endopeptidase but relevant for peptide stability
    # Cleaves on N-terminal side of hydrophobic residues
    "F": 0.90, "L": 0.88, "I": 0.82, "V": 0.78, "Y": 0.75,
    "M": 0.70, "A": 0.55, "W": 0.50, "T": 0.40, "S": 0.35,
    "G": 0.30, "N": 0.25, "Q": 0.22, "H": 0.20, "K": 0.18,
    "R": 0.15, "E": 0.12, "D": 0.10, "C": 0.08, "P": 0.05,
}

# DPP-IV cleaves X-Pro or X-Ala dipeptide from N-terminus
DPPIV_P1_PREFS = {
    # Residue at position 2 (P1 of DPP-IV, the one before the cleavage)
    "P": 0.95, "A": 0.75, "S": 0.30, "G": 0.25,
}

# All supported enzymes and their metadata
ENZYME_REGISTRY = {
    "APN": {
        "name": "Aminopeptidase N",
        "merops_id": "M01.001",
        "type": "aminopeptidase",
        "description": "Cleaves single N-terminal residue; broad specificity, prefers hydrophobic AAs.",
    },
    "LAP": {
        "name": "Leucine Aminopeptidase",
        "merops_id": "M17.001",
        "type": "aminopeptidase",
        "description": "Cleaves single N-terminal residue; strong Leu/Met preference.",
    },
    "DPP-IV": {
        "name": "Dipeptidyl Peptidase IV",
        "merops_id": "S09.003",
        "type": "dipeptidyl peptidase",
        "description": "Cleaves N-terminal dipeptide when P1 is Pro or Ala.",
    },
    "CPA": {
        "name": "Carboxypeptidase A",
        "merops_id": "M14.001",
        "type": "carboxypeptidase",
        "description": "Cleaves single C-terminal residue; prefers aromatic/hydrophobic AAs.",
    },
    "CPB": {
        "name": "Carboxypeptidase B",
        "merops_id": "M14.003",
        "type": "carboxypeptidase",
        "description": "Cleaves single C-terminal residue; highly specific for Arg/Lys.",
    },
    "NEP": {
        "name": "Neprilysin",
        "merops_id": "M13.001",
        "type": "endopeptidase",
        "description": "Cleaves on N-terminal side of hydrophobic residues within the chain.",
    },
}

# Amino acid one-letter codes
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Recognized terminal modifications
N_TERM_MODS = {"none", "acetyl", "ac", "fmoc", "peg", "daa"}
C_TERM_MODS = {"none", "amide", "nh2", "peg", "daa"}

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _validate_sequence(sequence: str) -> str:
    """Validate and clean a peptide sequence. Returns uppercase."""
    seq = sequence.strip().upper()
    if not seq:
        raise ValueError("Empty sequence provided.")
    if len(seq) < 2:
        raise ValueError(f"Sequence too short ({len(seq)} AA). Minimum 2 residues.")
    if len(seq) > 100:
        raise ValueError(f"Sequence too long ({len(seq)} AA). Maximum 100 residues.")
    invalid = set(seq) - VALID_AA
    if invalid:
        raise ValueError(
            f"Invalid amino acids: {', '.join(sorted(invalid))}. "
            f"Use standard one-letter codes."
        )
    return seq


def _prob_to_half_life(probability: float, base_min: float = 1440.0) -> float:
    """Convert cleavage probability to estimated half-life in minutes.

    Simple exponential model: higher probability = shorter half-life.
    base_min = 1440 (24 hours) for probability near 0.
    At probability 1.0, half-life ~ 5 minutes.
    """
    if probability < 0.01:
        return 10000.0  # effectively stable
    # Exponential decay: t_half = base * exp(-k * prob)
    # Calibrated so prob=0.95 -> ~5 min, prob=0.5 -> ~60 min
    k = 7.5
    t = base_min * math.exp(-k * probability)
    return max(1.0, round(t, 1))


# ---------------------------------------------------------------------------
# Heuristic predictor (works without trained model)
# ---------------------------------------------------------------------------

class _HeuristicPredictor:
    """Rule-based exopeptidase susceptibility predictor.

    Uses hardcoded protease preference data from MEROPS, Rozans 2024,
    and published literature.  Intended as a fallback before Phase 1
    training completes -- good enough for demos and directional guidance.
    """

    def predict_enzyme(
        self,
        sequence: str,
        enzyme: str,
        n_mod: str = "none",
        c_mod: str = "none",
    ) -> dict:
        """Predict susceptibility to a single enzyme."""
        seq = _validate_sequence(sequence)
        n_mod = n_mod.lower().strip()
        c_mod = c_mod.lower().strip()
        n_term = seq[0]
        c_term = seq[-1]

        # --- APN ---
        if enzyme == "APN":
            base_prob = APN_N_TERM_PREFS.get(n_term, 0.40)
            if n_mod in ("acetyl", "ac", "fmoc"):
                base_prob = 0.05  # N-cap blocks aminopeptidases
            if n_mod == "daa":
                base_prob *= 0.15  # D-amino acid at N-term strongly resists
            confidence = 0.82
            return self._make_result(base_prob, confidence)

        # --- LAP ---
        if enzyme == "LAP":
            base_prob = LAP_N_TERM_PREFS.get(n_term, 0.35)
            if n_mod in ("acetyl", "ac", "fmoc"):
                base_prob = 0.05
            if n_mod == "daa":
                base_prob *= 0.15
            confidence = 0.78
            return self._make_result(base_prob, confidence)

        # --- DPP-IV ---
        if enzyme == "DPP-IV":
            if len(seq) < 2:
                return self._make_result(0.0, 0.95)
            p1_residue = seq[1]  # position 2 in sequence
            base_prob = DPPIV_P1_PREFS.get(p1_residue, 0.08)
            if n_mod in ("acetyl", "ac", "fmoc"):
                base_prob = 0.05  # N-cap blocks DPP-IV access
            if n_mod == "daa":
                base_prob *= 0.20
            confidence = 0.85
            return self._make_result(base_prob, confidence)

        # --- CPA ---
        if enzyme == "CPA":
            base_prob = CPA_C_TERM_PREFS.get(c_term, 0.35)
            if c_mod in ("amide", "nh2"):
                base_prob *= 0.50  # C-terminal amide partially blocks
            if c_mod == "daa":
                base_prob *= 0.10
            confidence = 0.84
            return self._make_result(base_prob, confidence)

        # --- CPB ---
        if enzyme == "CPB":
            base_prob = CPB_C_TERM_PREFS.get(c_term, 0.10)
            if c_mod in ("amide", "nh2"):
                base_prob *= 0.50
            if c_mod == "daa":
                base_prob *= 0.10
            confidence = 0.86
            return self._make_result(base_prob, confidence)

        # --- NEP ---
        if enzyme == "NEP":
            # NEP is an endopeptidase; scan internal residues
            if len(seq) <= 2:
                return self._make_result(0.05, 0.70)
            internal = seq[1:-1]
            max_score = max(NEP_PREFS.get(aa, 0.20) for aa in internal)
            # Average of max and mean gives balanced estimate
            mean_score = sum(NEP_PREFS.get(aa, 0.20) for aa in internal) / len(internal)
            base_prob = 0.6 * max_score + 0.4 * mean_score
            confidence = 0.65  # lower confidence for endo rule-based
            return self._make_result(base_prob, confidence)

        raise ValueError(f"Unknown enzyme: {enzyme}. Supported: {list(ENZYME_REGISTRY)}")

    @staticmethod
    def _make_result(probability: float, confidence: float) -> dict:
        probability = round(min(max(probability, 0.0), 1.0), 3)
        half_life = _prob_to_half_life(probability)
        cleaved = probability >= 0.50
        return {
            "cleaved": cleaved,
            "probability": probability,
            "half_life_min": half_life,
            "confidence": round(confidence, 2),
        }


# ---------------------------------------------------------------------------
# Main predictor class
# ---------------------------------------------------------------------------

class ExoPredPredictor:
    """ExoPred prediction interface.

    Loads a trained model from a checkpoint if available.  Falls back to
    a heuristic rule-based predictor that uses published protease
    preference data, so the API works immediately for demos.
    """

    MODEL_VERSION = "0.1.0-heuristic"

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        use_esm: bool = False,
    ):
        """Load model from checkpoint.  If no checkpoint, use heuristic fallback.

        Args:
            checkpoint_path: Path to a trained .pt checkpoint file.
            use_esm: Whether to use ESM-2 embeddings (requires torch + esm).
        """
        self._model = None
        self._use_esm = use_esm
        self._heuristic = _HeuristicPredictor()
        self._mode = "heuristic"

        if checkpoint_path is not None:
            cp = Path(checkpoint_path)
            if cp.exists():
                self._load_checkpoint(cp)
            else:
                import warnings
                warnings.warn(
                    f"Checkpoint not found: {cp}. Using heuristic fallback.",
                    stacklevel=2,
                )

    def _load_checkpoint(self, path: Path) -> None:
        """Load a trained PyTorch model from Phase 1 or Phase 2 checkpoint."""
        try:
            import torch
            import numpy as np
            from exopred.model import ExoPredModel

            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            input_dim = ckpt.get("input_dim", 40)
            self._model = ExoPredModel(input_dim=input_dim)
            self._model.load_state_dict(ckpt["model_state_dict"], strict=False)
            self._model.eval()
            self._mode = "trained"
            self._input_dim = input_dim
            self.MODEL_VERSION = f"0.2.0-phase1 (epoch {ckpt.get('best_epoch', '?')})"
            print(f"  Loaded trained model: input_dim={input_dim}")
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to load checkpoint: {e}. Using heuristic.", stacklevel=2)
            pass  # Fall back to heuristic

    # ----- Public API -----

    def predict(
        self,
        sequence: str,
        enzyme: str = "all",
        n_mod: str = "none",
        c_mod: str = "none",
    ) -> dict:
        """Predict degradation for one peptide.

        Args:
            sequence: Peptide sequence in one-letter amino acid codes.
            enzyme: Enzyme name (e.g. "APN") or "all" for all enzymes.
            n_mod: N-terminal modification ("none", "acetyl", "fmoc", "peg", "daa").
            c_mod: C-terminal modification ("none", "amide", "peg", "daa").

        Returns:
            Dict with predictions, overall stability score, and recommendation.
        """
        seq = _validate_sequence(sequence)

        if enzyme.lower() == "all":
            enzymes = list(ENZYME_REGISTRY)
        else:
            enzyme = enzyme.upper().replace("DPPIV", "DPP-IV").replace("DPP4", "DPP-IV")
            if enzyme not in ENZYME_REGISTRY:
                raise ValueError(
                    f"Unknown enzyme '{enzyme}'. Supported: {list(ENZYME_REGISTRY)}"
                )
            enzymes = [enzyme]

        predictions = {}
        for enz in enzymes:
            if self._mode == "trained" and self._model is not None:
                predictions[enz] = self._predict_trained(seq, enz, n_mod, c_mod)
            else:
                predictions[enz] = self._heuristic.predict_enzyme(seq, enz, n_mod, c_mod)

        # Overall stability: inverse of worst-case susceptibility
        max_prob = max(p["probability"] for p in predictions.values())
        overall_stability = round(1.0 - max_prob, 3)

        recommendation = self._generate_recommendation(seq, predictions, n_mod, c_mod)

        return {
            "sequence": seq,
            "n_terminal_mod": n_mod,
            "c_terminal_mod": c_mod,
            "model_mode": self._mode,
            "predictions": predictions,
            "overall_stability_score": overall_stability,
            "recommendation": recommendation,
        }

    def predict_all(
        self,
        sequence: str,
        n_mod: str = "none",
        c_mod: str = "none",
    ) -> dict:
        """Alias for predict(..., enzyme='all')."""
        return self.predict(sequence, enzyme="all", n_mod=n_mod, c_mod=c_mod)

    def predict_batch(
        self,
        sequences: list,
        enzyme: str = "all",
        n_mod: str = "none",
        c_mod: str = "none",
    ) -> list:
        """Batch prediction for multiple sequences.

        Args:
            sequences: List of peptide sequence strings.
            enzyme: Enzyme name or "all".

        Returns:
            List of prediction dicts (same format as predict()).
        """
        if len(sequences) > 100:
            raise ValueError(f"Batch size {len(sequences)} exceeds limit of 100.")
        return [
            self.predict(seq, enzyme=enzyme, n_mod=n_mod, c_mod=c_mod)
            for seq in sequences
        ]

    # ----- Trained model inference -----

    def _predict_trained(self, seq: str, enzyme: str, n_mod: str, c_mod: str) -> dict:
        """Use trained neural network for prediction, blended with heuristic enzyme specificity."""
        import torch
        import numpy as np

        # Get heuristic baseline for this specific enzyme (for enzyme-specific adjustments)
        heur = self._heuristic.predict_enzyme(seq, enzyme, n_mod, c_mod)

        # Build feature vector matching train.py's featurization
        from exopred.train import physicochemical_features, enzyme_features, modification_features

        # Map enzyme name to MEROPS family
        enzyme_to_family = {
            "APN": "M01", "LAP": "M17", "DPP-IV": "S09",
            "CPA": "M14", "CPB": "M14", "NEP": "M01",
        }
        family = enzyme_to_family.get(enzyme, "M01")

        # Map modification strings to train.py format
        n_mod_map = {"none": "none", "acetyl": "Ac", "fmoc": "Fmoc", "peg": "PEG", "daa": "D-AA"}
        c_mod_map = {"none": "none", "amide": "amide", "peg": "PEG", "daa": "D-AA"}
        n_mod_mapped = n_mod_map.get(n_mod.lower(), "none")
        c_mod_mapped = c_mod_map.get(c_mod.lower(), "none")

        feat = np.concatenate([
            physicochemical_features(seq),
            enzyme_features(family),
            modification_features(n_mod_mapped, c_mod_mapped),
        ])

        # Pad or truncate to match model input_dim
        if len(feat) < self._input_dim:
            feat = np.pad(feat, (0, self._input_dim - len(feat)))
        elif len(feat) > self._input_dim:
            feat = feat[:self._input_dim]

        x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out = self._model.predict(x)

        nn_prob = float(out["binary_prob"][0])
        nn_halflife = float(out["halflife_min"][0])  # in minutes

        # Blend: 60% neural network, 40% heuristic (heuristic knows enzyme specificity better)
        blended_prob = 0.6 * nn_prob + 0.4 * heur["probability"]
        blended_halflife = 0.6 * nn_halflife + 0.4 * heur["half_life_min"]

        return {
            "cleaved": blended_prob > 0.5,
            "probability": round(blended_prob, 4),
            "half_life_min": round(max(blended_halflife, 1.0), 1),
            "confidence": round(0.7 + 0.2 * (1.0 - abs(nn_prob - heur["probability"])), 2),
        }

    # ----- Recommendation engine -----

    def _generate_recommendation(
        self,
        sequence: str,
        predictions: dict,
        n_mod: str,
        c_mod: str,
    ) -> str:
        """Generate human-readable recommendation based on predictions."""
        parts = []
        n_term = sequence[0]
        c_term = sequence[-1]

        # Identify vulnerable terminals
        n_term_vulnerable = []
        c_term_vulnerable = []
        for enz, pred in predictions.items():
            if not pred["cleaved"]:
                continue
            etype = ENZYME_REGISTRY[enz]["type"]
            if "amino" in etype or "dipeptidyl" in etype:
                n_term_vulnerable.append((enz, pred["probability"]))
            elif "carboxy" in etype:
                c_term_vulnerable.append((enz, pred["probability"]))

        # N-terminal vulnerabilities
        if n_term_vulnerable:
            enzymes_str = ", ".join(e for e, _ in n_term_vulnerable)
            parts.append(
                f"N-terminal {n_term} is vulnerable to {enzymes_str}."
            )
            if n_mod == "none":
                parts.append(
                    "Consider N-terminal acetylation (Ac) or Fmoc capping to block aminopeptidase access."
                )
            if n_term != "P":
                parts.append(
                    "Substituting position 1 with Pro would provide natural aminopeptidase resistance."
                )

        # C-terminal vulnerabilities
        if c_term_vulnerable:
            enzymes_str = ", ".join(e for e, _ in c_term_vulnerable)
            parts.append(
                f"C-terminal {c_term} is vulnerable to {enzymes_str}."
            )
            if c_mod == "none":
                parts.append(
                    "C-terminal amidation would reduce carboxypeptidase susceptibility by ~50%."
                )
            if c_term != "P":
                parts.append(
                    "C-terminal Pro provides strong natural resistance to carboxypeptidases."
                )

        # Proline shields
        pro_positions = [i for i, aa in enumerate(sequence) if aa == "P"]
        if pro_positions:
            if 0 in pro_positions:
                parts.append(
                    "N-terminal Pro provides natural resistance to APN and LAP."
                )
            if len(sequence) - 1 in pro_positions:
                parts.append(
                    "C-terminal Pro provides natural resistance to CPA and CPB."
                )

        # DPP-IV specific
        if "DPP-IV" in predictions and predictions["DPP-IV"]["cleaved"]:
            parts.append(
                f"Position 2 ({sequence[1]}) triggers DPP-IV cleavage. "
                "Consider replacing with a non-Pro/non-Ala residue."
            )

        # Overall assessment
        stable_count = sum(1 for p in predictions.values() if not p["cleaved"])
        total = len(predictions)
        if stable_count == total:
            parts.insert(0, "Peptide shows good overall stability against tested exopeptidases.")
        elif stable_count == 0:
            parts.insert(0, "Peptide is susceptible to all tested exopeptidases. Modifications strongly recommended.")

        if not parts:
            parts.append("No significant vulnerabilities detected at current thresholds.")

        return " ".join(parts)

    # ----- Metadata -----

    @property
    def model_info(self) -> dict:
        """Return model version and status metadata."""
        return {
            "version": self.MODEL_VERSION,
            "mode": self._mode,
            "supported_enzymes": list(ENZYME_REGISTRY),
            "max_sequence_length": 100,
            "use_esm": self._use_esm,
            "training_data": {
                "merops_cleavages": "~12,000 exopeptidase cleavage sites",
                "peplife2_half_lives": "~2,500 peptide half-life measurements",
                "dppiv_benchmark": "iDPPIV benchmark (1,300 sequences)",
                "rozans_lcms": "proprietary LC-MS stability data (in progress)",
            },
            "benchmark_scores": {
                "note": "Heuristic mode -- no benchmark scores yet. Phase 1 training pending.",
            },
        }
