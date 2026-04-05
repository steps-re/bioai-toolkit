"""
Feature engineering for ExoPred exopeptidase cleavage prediction.

Three feature groups:
  1. Physicochemical — always available, fast (no external deps beyond numpy/pandas)
  2. ESM-2 embeddings — optional (requires torch + transformers)
  3. Enzyme family features — one-hot + kinetics
  4. Modification features — terminal protection signals

Usage:
    from exopred.features import featurize_dataset, get_feature_names
    df = featurize_dataset(df, use_esm=False)
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
_HAS_BIOPYTHON = False
try:
    from Bio.SeqUtils import molecular_weight as _bio_mw
    _HAS_BIOPYTHON = True
except ImportError:
    pass

_HAS_TORCH = False
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    _HAS_TORCH = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KD: dict[str, float] = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Average residue molecular weights (monoisotopic-ish, daltons)
_AA_MW: dict[str, float] = {
    "A": 89.09, "R": 174.20, "N": 132.12, "D": 133.10, "C": 121.16,
    "Q": 146.15, "E": 147.13, "G": 75.03, "H": 155.16, "I": 131.17,
    "L": 131.17, "K": 146.19, "M": 149.21, "F": 165.19, "P": 115.13,
    "S": 105.09, "T": 119.12, "W": 204.23, "Y": 181.19, "V": 117.15,
}

# pKa values for charge calculation (N-term, C-term, sidechains)
_PKA_NTERM = 9.69
_PKA_CTERM = 2.34
_PKA_SIDE: dict[str, tuple[float, int]] = {
    # (pKa, +1 if protonated is positive, -1 if deprotonated is negative)
    "D": (3.65, -1),
    "E": (4.25, -1),
    "C": (8.18, -1),
    "Y": (10.07, -1),
    "H": (6.00, 1),
    "K": (10.54, 1),
    "R": (12.48, 1),
}

HYDROPHOBIC = set("AVILMFWP")
POLAR = set("STNQ")
CHARGED = set("DEKRH")
AROMATIC = set("FWY")

# Guruprasad instability dipeptide weights (DIWV table, subset — full 400 entries)
# Using the standard Guruprasad et al. (1990) table.
_DIWV: dict[str, float] = {}
_DIWV_RAW = {
    "WW": 1.0, "WC": 1.0, "WM": 24.68, "WH": 24.68, "WF": 1.0,
    "WP": 1.0, "WS": 1.0, "WT": -14.03, "WY": 1.0, "WA": -14.03,
    "WR": 1.0, "WN": 13.34, "WD": 1.0, "WQ": 1.0, "WE": 1.0,
    "WG": -9.37, "WI": 1.0, "WL": 13.34, "WK": 1.0, "WV": -7.49,
    "CW": 24.68, "CC": 1.0, "CM": 33.6, "CH": 33.6, "CF": 1.0,
    "CP": 20.26, "CS": 1.0, "CT": 33.6, "CY": 1.0, "CA": 1.0,
    "CR": 1.0, "CN": 1.0, "CD": 20.26, "CQ": -6.54, "CE": 1.0,
    "CG": 1.0, "CI": 1.0, "CL": 20.26, "CK": 1.0, "CV": -6.54,
    "MW": 1.0, "MC": 1.0, "MM": -1.88, "MH": 58.28, "MF": 1.0,
    "MP": 44.94, "MS": 44.94, "MT": -1.88, "MY": 24.68, "MA": 13.34,
    "MR": -6.54, "MN": 1.0, "MD": 1.0, "MQ": -6.54, "ME": 1.0,
    "MG": 1.0, "MI": 1.0, "ML": 1.0, "MK": 1.0, "MV": 1.0,
    "HW": -1.88, "HC": 1.0, "HM": 1.0, "HH": 1.0, "HF": -9.37,
    "HP": -1.88, "HS": 1.0, "HT": -6.54, "HY": 44.94, "HA": 1.0,
    "HR": 1.0, "HN": 24.68, "HD": 1.0, "HQ": 1.0, "HE": 1.0,
    "HG": -9.37, "HI": 44.94, "HL": 1.0, "HK": 24.68, "HV": -6.54,
    "FW": 1.0, "FC": 1.0, "FM": 1.0, "FH": 1.0, "FF": 1.0,
    "FP": 20.26, "FS": 1.0, "FT": 13.34, "FY": 33.6, "FA": 1.0,
    "FR": 1.0, "FN": 1.0, "FD": 13.34, "FQ": 1.0, "FE": 1.0,
    "FG": 1.0, "FI": 1.0, "FL": 1.0, "FK": -14.03, "FV": 1.0,
    "PW": -1.88, "PC": -6.54, "PM": -6.54, "PH": 1.0, "PF": 20.26,
    "PP": 20.26, "PS": 20.26, "PT": 1.0, "PY": 1.0, "PA": 20.26,
    "PR": -6.54, "PN": 1.0, "PD": -6.54, "PQ": 20.26, "PE": 18.38,
    "PG": 1.0, "PI": 1.0, "PL": 1.0, "PK": 1.0, "PV": 20.26,
    "SW": 1.0, "SC": 33.6, "SM": 1.0, "SH": 1.0, "SF": 1.0,
    "SP": 44.94, "SS": 20.26, "ST": 1.0, "SY": 1.0, "SA": 1.0,
    "SR": 20.26, "SN": 1.0, "SD": 1.0, "SQ": 20.26, "SE": 20.26,
    "SG": 1.0, "SI": 1.0, "SL": 1.0, "SK": 1.0, "SV": 1.0,
    "TW": -14.03, "TC": 1.0, "TM": 1.0, "TH": 1.0, "TF": 13.34,
    "TP": 1.0, "TS": 1.0, "TT": 1.0, "TY": 1.0, "TA": 1.0,
    "TR": 1.0, "TN": -14.03, "TD": 1.0, "TQ": -6.54, "TE": 20.26,
    "TG": -7.49, "TI": 1.0, "TL": 1.0, "TK": 1.0, "TV": 1.0,
    "YW": -9.37, "YC": 1.0, "YM": 44.94, "YH": 13.34, "YF": 1.0,
    "YP": 13.34, "YS": 1.0, "YT": -7.49, "YY": 13.34, "YA": 24.68,
    "YR": -15.91, "YN": 1.0, "YD": 24.68, "YQ": 1.0, "YE": -6.54,
    "YG": -7.49, "YI": 1.0, "YL": 1.0, "YK": 1.0, "YV": 1.0,
    "AW": 1.0, "AC": 44.94, "AM": 1.0, "AH": -7.49, "AF": 1.0,
    "AP": 20.26, "AS": 1.0, "AT": 1.0, "AY": 1.0, "AA": 1.0,
    "AR": 1.0, "AN": 1.0, "AD": -7.49, "AQ": 1.0, "AE": 1.0,
    "AG": 1.0, "AI": 1.0, "AL": 1.0, "AK": 1.0, "AV": 1.0,
    "RW": 58.28, "RC": 1.0, "RM": 1.0, "RH": 20.26, "RF": 1.0,
    "RP": 20.26, "RS": 44.94, "RT": 1.0, "RY": -6.54, "RA": 1.0,
    "RR": 58.28, "RN": 13.34, "RD": 1.0, "RQ": 20.26, "RE": 1.0,
    "RG": -7.49, "RI": 1.0, "RL": 1.0, "RK": 1.0, "RV": 1.0,
    "NW": 13.34, "NC": -1.88, "NM": 1.0, "NH": 1.0, "NF": -14.03,
    "NP": 1.0, "NS": 1.0, "NT": -7.49, "NY": 1.0, "NA": 1.0,
    "NR": 1.0, "NN": 1.0, "ND": 1.0, "NQ": -6.54, "NE": 1.0,
    "NG": -14.03, "NI": 44.94, "NL": 1.0, "NK": 24.68, "NV": 1.0,
    "DW": 1.0, "DC": 1.0, "DM": 1.0, "DH": 1.0, "DF": -6.54,
    "DP": 1.0, "DS": 20.26, "DT": -14.03, "DY": 1.0, "DA": 1.0,
    "DR": -6.54, "DN": 1.0, "DD": 1.0, "DQ": 1.0, "DE": 1.0,
    "DG": 1.0, "DI": 1.0, "DL": 1.0, "DK": -7.49, "DV": 1.0,
    "QW": 1.0, "QC": -6.54, "QM": 1.0, "QH": 1.0, "QF": -6.54,
    "QP": 20.26, "QS": 44.94, "QT": 1.0, "QY": 1.0, "QA": 1.0,
    "QR": 1.0, "QN": 1.0, "QD": 20.26, "QQ": 20.26, "QE": 1.0,
    "QG": 1.0, "QI": 1.0, "QL": 1.0, "QK": 1.0, "QV": -6.54,
    "EW": -14.03, "EC": 44.94, "EM": 1.0, "EH": 1.0, "EF": 1.0,
    "EP": 20.26, "ES": 20.26, "ET": 1.0, "EY": 1.0, "EA": 1.0,
    "ER": 1.0, "EN": 1.0, "ED": 20.26, "EQ": 20.26, "EE": 33.6,
    "EG": 1.0, "EI": 20.26, "EL": 1.0, "EK": 1.0, "EV": 1.0,
    "GW": 13.34, "GC": 1.0, "GM": 1.0, "GH": 1.0, "GF": 1.0,
    "GP": 1.0, "GS": 1.0, "GT": -7.49, "GY": -7.49, "GA": -7.49,
    "GR": 1.0, "GN": -7.49, "GD": 1.0, "GQ": 1.0, "GE": -6.54,
    "GG": 13.34, "GI": -7.49, "GL": 1.0, "GK": -7.49, "GV": 1.0,
    "IW": 1.0, "IC": 1.0, "IM": 1.0, "IH": 13.34, "IF": 1.0,
    "IP": -1.88, "IS": 1.0, "IT": 1.0, "IY": 1.0, "IA": 1.0,
    "IR": 1.0, "IN": 1.0, "ID": 1.0, "IQ": 1.0, "IE": 44.94,
    "IG": 1.0, "II": 1.0, "IL": 20.26, "IK": -7.49, "IV": -7.49,
    "LW": 24.68, "LC": 1.0, "LM": 1.0, "LH": 1.0, "LF": 1.0,
    "LP": 20.26, "LS": 1.0, "LT": 1.0, "LY": 1.0, "LA": 1.0,
    "LR": 20.26, "LN": 1.0, "LD": 1.0, "LQ": 33.6, "LE": 1.0,
    "LG": 1.0, "LI": 1.0, "LL": 1.0, "LK": -7.49, "LV": 1.0,
    "KW": 1.0, "KC": 1.0, "KM": 33.6, "KH": 1.0, "KF": 1.0,
    "KP": -6.54, "KS": 1.0, "KT": 1.0, "KY": 1.0, "KA": 1.0,
    "KR": 33.6, "KN": 1.0, "KD": 1.0, "KQ": 24.68, "KE": 1.0,
    "KG": -7.49, "KI": -7.49, "KL": -7.49, "KK": 1.0, "KV": -7.49,
    "VW": 1.0, "VC": 1.0, "VM": 1.0, "VH": 1.0, "VF": 1.0,
    "VP": 20.26, "VS": 1.0, "VT": -7.49, "VY": -6.54, "VA": 1.0,
    "VR": 1.0, "VN": 1.0, "VD": -14.03, "VQ": 1.0, "VE": 1.0,
    "VG": -7.49, "VI": 1.0, "VL": 1.0, "VK": -1.88, "VV": 1.0,
}
_DIWV.update(_DIWV_RAW)

# Enzyme families
ENZYME_FAMILIES = [
    "M01", "M17", "M18", "M24", "M28",  # aminopeptidases
    "M14", "M32", "S10", "S28",          # carboxypeptidases
    "S09",                                 # DPP
    "C01",                                 # cathepsins
]

AMINO_FAMILIES = {"M01", "M17", "M18", "M24", "M28"}
CARBOXY_FAMILIES = {"M14", "M32", "S10", "S28"}
DPP_FAMILIES = {"S09"}

# Average Km (uM) and kcat (s^-1) from BRENDA — representative values
ENZYME_KM: dict[str, float] = {
    "M01": 120.0, "M17": 250.0, "M18": 180.0, "M24": 300.0, "M28": 200.0,
    "M14": 150.0, "M32": 90.0, "S10": 400.0, "S28": 350.0,
    "S09": 80.0, "C01": 500.0,
}

ENZYME_KCAT: dict[str, float] = {
    "M01": 45.0, "M17": 30.0, "M18": 25.0, "M24": 15.0, "M28": 35.0,
    "M14": 60.0, "M32": 20.0, "S10": 10.0, "S28": 12.0,
    "S09": 55.0, "C01": 8.0,
}

# N-terminal modification encoding
N_MOD_MAP: dict[str, int] = {
    "none": 0, "NH2": 1, "Ac": 2, "Fmoc": 3, "PEG": 4, "beta_alanine": 5,
}

# C-terminal modification encoding
C_MOD_MAP: dict[str, int] = {
    "none": 0, "COOH": 1, "amide": 2, "beta_alanine": 3,
}

# Modifications that protect against aminopeptidases
N_PROTECTIVE = {"Ac", "Fmoc", "PEG"}
# Modifications that protect against carboxypeptidases
C_PROTECTIVE = {"amide"}


# ---------------------------------------------------------------------------
# Helper: clean sequence
# ---------------------------------------------------------------------------
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def _clean_sequence(seq: str) -> str:
    """Uppercase, strip whitespace, replace non-standard AAs with X."""
    seq = seq.strip().upper()
    return "".join(ch if ch in STANDARD_AA else "X" for ch in seq)


# ---------------------------------------------------------------------------
# 1. Physicochemical features
# ---------------------------------------------------------------------------

def _molecular_weight(seq: str) -> float:
    """Molecular weight in daltons.  Uses Biopython if available, else lookup table."""
    if _HAS_BIOPYTHON:
        try:
            return _bio_mw(seq, seq_type="protein")
        except Exception:
            pass
    # Fallback: sum of residue weights minus water lost per peptide bond
    water = 18.015
    mw = sum(_AA_MW.get(aa, 118.0) for aa in seq)  # 118 = avg for unknowns
    mw -= water * max(0, len(seq) - 1)
    return mw


def _net_charge_ph7(seq: str, ph: float = 7.4) -> float:
    """Henderson-Hasselbalch net charge at given pH."""
    charge = 0.0
    # N-terminal amino group (positive when protonated)
    charge += 1.0 / (1.0 + 10 ** (ph - _PKA_NTERM))
    # C-terminal carboxyl (negative when deprotonated)
    charge -= 1.0 / (1.0 + 10 ** (_PKA_CTERM - ph))
    # Side chains
    for aa in seq:
        if aa in _PKA_SIDE:
            pka, sign = _PKA_SIDE[aa]
            if sign > 0:  # positive when protonated
                charge += 1.0 / (1.0 + 10 ** (ph - pka))
            else:  # negative when deprotonated
                charge -= 1.0 / (1.0 + 10 ** (pka - ph))
    return charge


def _gravy(seq: str) -> float:
    """Grand average of hydropathy (Kyte-Doolittle)."""
    if not seq:
        return 0.0
    return sum(KD.get(aa, 0.0) for aa in seq) / len(seq)


def _instability_index(seq: str) -> float:
    """Guruprasad instability index."""
    if len(seq) < 2:
        return 0.0
    total = 0.0
    for i in range(len(seq) - 1):
        dipep = seq[i : i + 2]
        total += _DIWV.get(dipep, 1.0)
    return (10.0 / len(seq)) * total


def _aromaticity(seq: str) -> float:
    if not seq:
        return 0.0
    return sum(1 for aa in seq if aa in AROMATIC) / len(seq)


def _fraction(seq: str, aa_set: set) -> float:
    if not seq:
        return 0.0
    return sum(1 for aa in seq if aa in aa_set) / len(seq)


def physicochemical_features(seq: str) -> dict[str, float]:
    """Compute all physicochemical features for a single cleaned sequence."""
    seq = _clean_sequence(seq)
    n = len(seq)
    if n == 0:
        # Return zeros for empty sequences
        return {k: 0.0 for k in [
            "length", "mw", "charge_ph7", "gravy", "instability_index",
            "aromaticity", "frac_hydrophobic", "frac_polar", "frac_charged",
            "n_term_aa", "c_term_aa",
            "n_term_hydrophobicity", "c_term_hydrophobicity",
            "proline_at_1", "proline_at_2", "proline_at_n", "proline_at_n1",
        ]}

    first = seq[0]
    last = seq[-1]

    return {
        "length": float(n),
        "mw": _molecular_weight(seq),
        "charge_ph7": _net_charge_ph7(seq),
        "gravy": _gravy(seq),
        "instability_index": _instability_index(seq),
        "aromaticity": _aromaticity(seq),
        "frac_hydrophobic": _fraction(seq, HYDROPHOBIC),
        "frac_polar": _fraction(seq, POLAR),
        "frac_charged": _fraction(seq, CHARGED),
        "n_term_aa": first,
        "c_term_aa": last,
        "n_term_hydrophobicity": KD.get(first, 0.0),
        "c_term_hydrophobicity": KD.get(last, 0.0),
        "proline_at_1": 1.0 if first == "P" else 0.0,
        "proline_at_2": 1.0 if (n >= 2 and seq[1] == "P") else 0.0,
        "proline_at_n": 1.0 if last == "P" else 0.0,
        "proline_at_n1": 1.0 if (n >= 2 and seq[-2] == "P") else 0.0,
    }


# ---------------------------------------------------------------------------
# 2. ESM-2 embeddings
# ---------------------------------------------------------------------------

def extract_esm2_embeddings(
    sequences: list[str],
    model_name: str = "facebook/esm2_t6_8M_UR50D",
    batch_size: int = 32,
) -> np.ndarray:
    """Extract ESM-2 embeddings for a list of peptide sequences.

    Returns shape (n_sequences, embed_dim * 4) where features are:
      [CLS] token | mean-pool | first-residue | last-residue

    Raises ImportError if torch/transformers unavailable.
    """
    if not _HAS_TORCH:
        raise ImportError(
            "ESM-2 embeddings require torch and transformers. "
            "Install with: pip install torch transformers"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Clean sequences: replace non-standard with X, ensure min length
    cleaned = []
    for seq in sequences:
        s = _clean_sequence(seq)
        if len(s) < 3:
            s = s + "G" * (3 - len(s))  # pad short sequences
        cleaned.append(s)

    all_embeddings = []

    for start in range(0, len(cleaned), batch_size):
        batch = cleaned[start : start + batch_size]
        original_lengths = [len(s) for s in batch]

        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=1024
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # outputs.last_hidden_state: (batch, seq_len, hidden)
        hidden = outputs.last_hidden_state

        for i in range(len(batch)):
            h = hidden[i]  # (seq_len, hidden)
            # Token layout: [CLS] res1 res2 ... resN [EOS] [PAD]...
            cls_emb = h[0]  # CLS token
            orig_len = original_lengths[i]
            # Residue tokens are positions 1..orig_len
            residue_tokens = h[1 : orig_len + 1]
            mean_emb = residue_tokens.mean(dim=0)
            first_emb = h[1]
            last_emb = h[orig_len]  # last residue token

            feat = torch.cat([cls_emb, mean_emb, first_emb, last_emb])
            all_embeddings.append(feat.cpu().numpy())

    return np.stack(all_embeddings, axis=0)


# ---------------------------------------------------------------------------
# 3. Enzyme features
# ---------------------------------------------------------------------------

def enzyme_features(enzyme_family: Optional[str]) -> dict[str, float]:
    """One-hot encode enzyme family + kinetic parameters."""
    feats: dict[str, float] = {}

    # One-hot for each family
    for fam in ENZYME_FAMILIES:
        feats[f"enzyme_{fam}"] = 1.0 if enzyme_family == fam else 0.0

    fam = enzyme_family or ""
    feats["enzyme_is_amino"] = 1.0 if fam in AMINO_FAMILIES else 0.0
    feats["enzyme_is_carboxy"] = 1.0 if fam in CARBOXY_FAMILIES else 0.0
    feats["enzyme_is_dipeptidyl"] = 1.0 if fam in DPP_FAMILIES else 0.0
    feats["enzyme_km"] = ENZYME_KM.get(fam, 0.0)
    feats["enzyme_kcat"] = ENZYME_KCAT.get(fam, 0.0)

    return feats


# ---------------------------------------------------------------------------
# 4. Modification features
# ---------------------------------------------------------------------------

def modification_features(
    n_mod: Optional[str] = None,
    c_mod: Optional[str] = None,
) -> dict[str, float]:
    """Encode terminal modifications."""
    n_mod = n_mod or "none"
    c_mod = c_mod or "none"

    n_code = N_MOD_MAP.get(n_mod, 6)  # 6 = other
    c_code = C_MOD_MAP.get(c_mod, 4)  # 4 = other

    return {
        "n_mod_type": float(n_code),
        "c_mod_type": float(c_code),
        "has_n_protection": 1.0 if n_mod in N_PROTECTIVE else 0.0,
        "has_c_protection": 1.0 if c_mod in C_PROTECTIVE else 0.0,
    }


# ---------------------------------------------------------------------------
# Feature name lists
# ---------------------------------------------------------------------------

_PHYSCHEM_NUMERIC = [
    "length", "mw", "charge_ph7", "gravy", "instability_index",
    "aromaticity", "frac_hydrophobic", "frac_polar", "frac_charged",
    "n_term_hydrophobicity", "c_term_hydrophobicity",
    "proline_at_1", "proline_at_2", "proline_at_n", "proline_at_n1",
]

_ENZYME_NAMES = (
    [f"enzyme_{fam}" for fam in ENZYME_FAMILIES]
    + ["enzyme_is_amino", "enzyme_is_carboxy", "enzyme_is_dipeptidyl",
       "enzyme_km", "enzyme_kcat"]
)

_MOD_NAMES = ["n_mod_type", "c_mod_type", "has_n_protection", "has_c_protection"]


def get_feature_names(use_esm: bool = False, esm_model: str = "facebook/esm2_t6_8M_UR50D") -> list[str]:
    """Return list of all feature column names for model input.

    Note: n_term_aa and c_term_aa are categorical and need one-hot encoding
    before model training — they are NOT included here. Use pd.get_dummies
    on the featurized DataFrame for those.
    """
    names = list(_PHYSCHEM_NUMERIC) + list(_ENZYME_NAMES) + list(_MOD_NAMES)

    if use_esm:
        # ESM-2 hidden dims: t6_8M = 320, t33_650M = 1280
        if "t6_8M" in esm_model:
            dim = 320
        elif "t33_650M" in esm_model:
            dim = 1280
        else:
            dim = 320  # default
        for prefix in ["esm_cls", "esm_mean", "esm_first", "esm_last"]:
            names += [f"{prefix}_{i}" for i in range(dim)]

    return names


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def featurize_dataset(
    df: pd.DataFrame,
    use_esm: bool = False,
    esm_model: str = "facebook/esm2_t6_8M_UR50D",
) -> pd.DataFrame:
    """Add all feature columns to a DataFrame with 'sequence' column.

    Expected columns (all optional except 'sequence'):
      - sequence: str — peptide sequence
      - enzyme_family: str — e.g. "M01", "S09"
      - n_mod: str — N-terminal modification
      - c_mod: str — C-terminal modification

    Returns DataFrame with original columns + all feature columns.
    """
    df = df.copy()

    # --- Physicochemical ---
    physchem_rows = []
    for seq in df["sequence"]:
        physchem_rows.append(physicochemical_features(seq))
    physchem_df = pd.DataFrame(physchem_rows, index=df.index)
    df = pd.concat([df, physchem_df], axis=1)

    # --- Enzyme ---
    if "enzyme_family" in df.columns:
        enz_rows = [enzyme_features(fam) for fam in df["enzyme_family"]]
    else:
        enz_rows = [enzyme_features(None)] * len(df)
    enz_df = pd.DataFrame(enz_rows, index=df.index)
    df = pd.concat([df, enz_df], axis=1)

    # --- Modifications ---
    n_mods = df["n_mod"] if "n_mod" in df.columns else [None] * len(df)
    c_mods = df["c_mod"] if "c_mod" in df.columns else [None] * len(df)
    mod_rows = [modification_features(n, c) for n, c in zip(n_mods, c_mods)]
    mod_df = pd.DataFrame(mod_rows, index=df.index)
    df = pd.concat([df, mod_df], axis=1)

    # --- ESM-2 embeddings ---
    if use_esm:
        if not _HAS_TORCH:
            warnings.warn(
                "ESM-2 requested but torch/transformers not installed. "
                "Skipping embeddings — using physicochemical features only.",
                UserWarning,
                stacklevel=2,
            )
        else:
            sequences = df["sequence"].tolist()
            embeddings = extract_esm2_embeddings(sequences, model_name=esm_model)
            # Determine hidden dim
            embed_dim = embeddings.shape[1] // 4
            esm_cols = []
            for prefix in ["esm_cls", "esm_mean", "esm_first", "esm_last"]:
                esm_cols += [f"{prefix}_{i}" for i in range(embed_dim)]
            esm_df = pd.DataFrame(embeddings, columns=esm_cols, index=df.index)
            df = pd.concat([df, esm_df], axis=1)

    return df


# ---------------------------------------------------------------------------
# Quick self-test (physicochemical only — no ESM-2)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== ExoPred Feature Engineering Self-Test ===\n")

    test_data = pd.DataFrame({
        "sequence": [
            "DRVYIHPFHL",   # Angiotensin I fragment
            "YGGFL",         # Leu-enkephalin
            "PP",            # Diproline (minimal, proline-rich)
            "ACDEFGHIKLMNPQRSTVWY",  # All 20 standard AAs
            "A",             # Single residue
        ],
        "enzyme_family": ["M01", "S09", "M14", None, "C01"],
        "n_mod": ["none", "Ac", "Fmoc", "none", "PEG"],
        "c_mod": ["none", "amide", "none", "COOH", "beta_alanine"],
    })

    result = featurize_dataset(test_data, use_esm=False)

    print(f"Input rows: {len(test_data)}")
    print(f"Output columns: {len(result.columns)}")
    print(f"Feature columns (numeric): {len(get_feature_names(use_esm=False))}")
    print()

    # Print a few key features per sequence
    for i, row in result.iterrows():
        seq = row["sequence"]
        print(f"  {seq[:20]:<22s} len={row['length']:.0f}  MW={row['mw']:.1f}  "
              f"charge={row['charge_ph7']:+.2f}  GRAVY={row['gravy']:.2f}  "
              f"instab={row['instability_index']:.1f}  "
              f"N={row['n_term_aa']}  C={row['c_term_aa']}")

    print()

    # Validate shapes
    feat_names = get_feature_names(use_esm=False)
    missing = [f for f in feat_names if f not in result.columns]
    if missing:
        print(f"WARNING: missing columns: {missing}")
    else:
        print("All expected feature columns present.")

    # Check no NaN in numeric features
    numeric_cols = [c for c in feat_names if c in result.columns]
    nan_count = result[numeric_cols].isna().sum().sum()
    print(f"NaN values in numeric features: {nan_count}")

    # Verify proline signals
    pp_row = result.iloc[2]
    assert pp_row["proline_at_1"] == 1.0, "PP should have proline_at_1"
    assert pp_row["proline_at_n"] == 1.0, "PP should have proline_at_n"
    print("Proline signal checks passed.")

    # Verify modification features
    assert result.iloc[1]["has_n_protection"] == 1.0, "Ac should be N-protective"
    assert result.iloc[1]["has_c_protection"] == 1.0, "amide should be C-protective"
    assert result.iloc[0]["has_n_protection"] == 0.0, "none should not be N-protective"
    print("Modification feature checks passed.")

    # Verify enzyme features
    assert result.iloc[0]["enzyme_M01"] == 1.0, "First row should be M01"
    assert result.iloc[0]["enzyme_is_amino"] == 1.0, "M01 is aminopeptidase"
    assert result.iloc[1]["enzyme_is_dipeptidyl"] == 1.0, "S09 is DPP"
    assert result.iloc[2]["enzyme_is_carboxy"] == 1.0, "M14 is carboxypeptidase"
    print("Enzyme feature checks passed.")

    if _HAS_TORCH:
        print("\ntorch + transformers available — ESM-2 embeddings can be used.")
    else:
        print("\ntorch/transformers not installed — ESM-2 disabled (physicochemical only).")

    print("\n=== All tests passed. ===")
