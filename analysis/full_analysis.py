"""
Full physicochemical analysis of 618 Rozans peptide sequences.
Outputs: enriched CSV + markdown report + plots.
"""

import pandas as pd
import numpy as np
from Bio.SeqUtils import molecular_weight
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from collections import Counter
from pathlib import Path
import json

DATA_PATH = Path(__file__).parent.parent / "data" / "rozans-peptide-library.csv"
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

# Amino acid property tables
KD_HYDROPHOBICITY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
}

# Exopeptidase susceptibility by residue (approximate, from literature)
# Higher = more susceptible to aminopeptidase (N-terminal) or carboxypeptidase (C-terminal)
# Based on broad-spectrum aminopeptidase/carboxypeptidase substrate preferences
AMINOPEPTIDASE_SUSCEPTIBILITY = {
    'A': 0.9, 'R': 0.6, 'N': 0.5, 'D': 0.3, 'E': 0.3,
    'Q': 0.5, 'G': 0.7, 'H': 0.4, 'I': 0.7, 'L': 0.9,
    'K': 0.7, 'M': 0.8, 'F': 0.8, 'P': 0.1, 'S': 0.6,
    'T': 0.6, 'V': 0.7, 'W': 0.5, 'Y': 0.6,
}

CARBOXYPEPTIDASE_SUSCEPTIBILITY = {
    'A': 0.8, 'R': 0.5, 'N': 0.4, 'D': 0.2, 'E': 0.2,
    'Q': 0.4, 'G': 0.6, 'H': 0.3, 'I': 0.7, 'L': 0.8,
    'K': 0.6, 'M': 0.7, 'F': 0.9, 'P': 0.05, 'S': 0.5,
    'T': 0.5, 'V': 0.7, 'W': 0.6, 'Y': 0.7,
}

# MMP cleavage preferences (for Paper 3 crosslinker analysis)
# P1 position (before scissile bond) preferences for MMPs
MMP_P1_PREFERENCE = {
    'G': 0.3, 'A': 0.5, 'V': 0.4, 'L': 0.8, 'I': 0.7,
    'P': 0.2, 'F': 0.6, 'W': 0.5, 'M': 0.6, 'S': 0.4,
    'T': 0.4, 'N': 0.5, 'Q': 0.5, 'D': 0.3, 'E': 0.4,
    'K': 0.4, 'R': 0.4, 'H': 0.3, 'Y': 0.5,
}
# P1' position (after scissile bond) preferences for MMPs
MMP_P1P_PREFERENCE = {
    'G': 0.3, 'A': 0.5, 'V': 0.4, 'L': 0.8, 'I': 0.8,
    'P': 0.1, 'F': 0.6, 'W': 0.5, 'M': 0.6, 'S': 0.5,
    'T': 0.5, 'N': 0.4, 'Q': 0.4, 'D': 0.3, 'E': 0.3,
    'K': 0.4, 'R': 0.4, 'H': 0.3, 'Y': 0.5,
}

AA_CATEGORIES = {
    'Hydrophobic': set('AILMFWVP'),
    'Polar': set('STNQ'),
    'Charged+': set('KRH'),
    'Charged-': set('DE'),
    'Aromatic': set('FWY'),
    'Small': set('GAS'),
    'Branched': set('VIL'),
}

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def analyze_sequence(seq):
    """Compute all properties for a single sequence."""
    clean = "".join(c for c in seq.upper() if c in VALID_AA)
    if len(clean) < 2:
        return None

    try:
        analysis = ProteinAnalysis(clean)
        mw = molecular_weight(clean, "protein")
        pi = analysis.isoelectric_point()
        gravy = analysis.gravy()
        instability = analysis.instability_index()
        aromaticity = analysis.aromaticity()
        ss = analysis.secondary_structure_fraction()

        # Amino acid composition
        aa_percent = analysis.amino_acids_percent

        # Category fractions
        cat_fracs = {}
        for cat, aas in AA_CATEGORIES.items():
            cat_fracs[f"frac_{cat.lower().replace('+','pos').replace('-','neg')}"] = (
                sum(1 for c in clean if c in aas) / len(clean)
            )

        # Net charge at pH 7 (approximate)
        pos_count = sum(1 for c in clean if c in 'KR') + sum(0.1 for c in clean if c == 'H')
        neg_count = sum(1 for c in clean if c in 'DE')
        net_charge_ph7 = pos_count - neg_count

        # Terminal residue properties
        n_term_aa = clean[0]
        c_term_aa = clean[-1]
        n_term_hydro = KD_HYDROPHOBICITY.get(n_term_aa, 0)
        c_term_hydro = KD_HYDROPHOBICITY.get(c_term_aa, 0)

        # Exopeptidase susceptibility scores
        amino_susc = AMINOPEPTIDASE_SUSCEPTIBILITY.get(n_term_aa, 0.5)
        carboxy_susc = CARBOXYPEPTIDASE_SUSCEPTIBILITY.get(c_term_aa, 0.5)
        total_exo_susc = (amino_susc + carboxy_susc) / 2

        return {
            "clean_sequence": clean,
            "length": len(clean),
            "mw_da": round(mw, 2),
            "pI": round(pi, 3),
            "gravy": round(gravy, 4),
            "instability_index": round(instability, 2),
            "aromaticity": round(aromaticity, 4),
            "helix_frac": round(ss[0], 4),
            "turn_frac": round(ss[1], 4),
            "sheet_frac": round(ss[2], 4),
            "net_charge_ph7": round(net_charge_ph7, 1),
            "n_term_aa": n_term_aa,
            "c_term_aa": c_term_aa,
            "n_term_hydrophobicity": n_term_hydro,
            "c_term_hydrophobicity": c_term_hydro,
            "aminopeptidase_susceptibility": amino_susc,
            "carboxypeptidase_susceptibility": carboxy_susc,
            "total_exopeptidase_susceptibility": round(total_exo_susc, 3),
            **cat_fracs,
            **{f"aa_{aa}": round(aa_percent.get(aa, 0), 4) for aa in sorted(VALID_AA)},
        }
    except Exception as e:
        print(f"  ERROR on {seq}: {e}")
        return {"error": str(e)}


def analyze_crosslinker(seq, var_residues):
    """Additional analysis for KLVAD-X1X2-ASAE crosslinkers."""
    if len(var_residues) != 2:
        return {}

    x1, x2 = var_residues[0], var_residues[1]
    mmp_p1 = MMP_P1_PREFERENCE.get(x1, 0.5)
    mmp_p1p = MMP_P1P_PREFERENCE.get(x2, 0.5)
    mmp_score = (mmp_p1 + mmp_p1p) / 2

    # Steric bulk at cleavage site
    bulk = {
        'G': 0.1, 'A': 0.3, 'V': 0.6, 'L': 0.7, 'I': 0.7,
        'P': 0.5, 'F': 0.9, 'W': 1.0, 'M': 0.7, 'S': 0.3,
        'T': 0.4, 'N': 0.4, 'Q': 0.5, 'D': 0.4, 'E': 0.5,
        'K': 0.6, 'R': 0.7, 'H': 0.6, 'Y': 0.8,
    }
    avg_bulk = (bulk.get(x1, 0.5) + bulk.get(x2, 0.5)) / 2

    # Dipeptide hydrophobicity
    dp_hydro = (KD_HYDROPHOBICITY.get(x1, 0) + KD_HYDROPHOBICITY.get(x2, 0)) / 2

    return {
        "mmp_p1_score": mmp_p1,
        "mmp_p1p_score": mmp_p1p,
        "mmp_cleavage_score": round(mmp_score, 3),
        "cleavage_site_bulk": round(avg_bulk, 3),
        "dipeptide_hydrophobicity": round(dp_hydro, 2),
    }


# ---- MAIN ----
print("Loading peptide library...")
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} peptides")

# Analyze all sequences
print("Computing physicochemical properties...")
props_list = []
for i, row in df.iterrows():
    props = analyze_sequence(row["sequence"])
    if props and "error" not in props:
        # Add crosslinker analysis for Paper 3
        if row["scaffold"] == "KLVAD-XX-ASAE" and len(str(row["variable_residue"])) == 2:
            xlink_props = analyze_crosslinker(row["sequence"], row["variable_residue"])
            props.update(xlink_props)
        props_list.append(props)
    else:
        # Fill with NaN so columns align
        print(f"  SKIP row {i}: {row['sequence']} -> props={props}")
        props_list.append({"clean_sequence": row["sequence"], "length": len(row["sequence"])})

props_df = pd.DataFrame(props_list, index=df.index)
enriched = pd.concat([df, props_df], axis=1)

# Save enriched CSV
enriched_path = OUT_DIR / "rozans-618-enriched.csv"
enriched.to_csv(enriched_path, index=False)
print(f"Enriched CSV saved: {enriched_path}")

# ============================================================
# GENERATE REPORT
# ============================================================
report = []
report.append("# Rozans Peptide Library — Full Analysis")
report.append(f"\n**618 peptides** from 3 publications (Pashuck Lab, Lehigh University)")
report.append(f"Analysis date: 2026-04-03\n")

# ---- Overview ----
report.append("## 1. Overview\n")
report.append(f"| Metric | Value |")
report.append(f"|--------|-------|")
report.append(f"| Total peptides | {len(enriched)} |")
report.append(f"| Unique sequences | {enriched['sequence'].nunique()} |")
report.append(f"| Papers | {enriched['paper'].nunique()} |")
report.append(f"| Libraries | {enriched['library'].nunique()} |")
report.append(f"| Length range | {enriched['length'].min()}-{enriched['length'].max()} aa |")
report.append(f"| MW range | {enriched['mw_da'].min():.0f}-{enriched['mw_da'].max():.0f} Da |")
report.append(f"| pI range | {enriched['pI'].min():.2f}-{enriched['pI'].max():.2f} |")
report.append("")

# By paper
report.append("### Peptides by Paper\n")
for paper, group in enriched.groupby("paper"):
    report.append(f"- **{paper}**: {len(group)} peptides, {group['library'].nunique()} libraries")
report.append("")

# ---- Property Distributions ----
report.append("## 2. Property Distributions\n")

for prop, label, unit in [
    ("mw_da", "Molecular Weight", "Da"),
    ("pI", "Isoelectric Point", ""),
    ("gravy", "GRAVY (Hydrophobicity)", ""),
    ("instability_index", "Instability Index", ""),
    ("net_charge_ph7", "Net Charge (pH 7)", ""),
    ("aromaticity", "Aromaticity", ""),
]:
    col = enriched[prop].dropna()
    if len(col) == 0:
        continue
    report.append(f"### {label}\n")
    report.append(f"| Stat | Value |")
    report.append(f"|------|-------|")
    report.append(f"| Mean | {col.mean():.3f} {unit} |")
    report.append(f"| Median | {col.median():.3f} {unit} |")
    report.append(f"| Std | {col.std():.3f} {unit} |")
    report.append(f"| Min | {col.min():.3f} {unit} |")
    report.append(f"| Max | {col.max():.3f} {unit} |")
    report.append("")

# ---- Stability classification ----
report.append("### Stability Classification (Instability Index)\n")
report.append("Instability Index < 40 = predicted stable; >= 40 = predicted unstable.\n")
stable = enriched[enriched["instability_index"] < 40]
unstable = enriched[enriched["instability_index"] >= 40]
report.append(f"- **Stable (II < 40):** {len(stable)} peptides ({100*len(stable)/len(enriched):.1f}%)")
report.append(f"- **Unstable (II >= 40):** {len(unstable)} peptides ({100*len(unstable)/len(enriched):.1f}%)")
report.append("")

# ---- Terminal Residue Analysis ----
report.append("## 3. Terminal Residue Analysis\n")
report.append("Critical for exopeptidase degradation — the exposed N-terminal and C-terminal residues determine how fast aminopeptidases and carboxypeptidases degrade the peptide.\n")

# N-terminal
report.append("### N-terminal Residue Distribution\n")
n_counts = enriched["n_term_aa"].value_counts().sort_index()
report.append("| Residue | Count | Aminopeptidase Susceptibility |")
report.append("|---------|-------|------------------------------|")
for aa, count in n_counts.items():
    susc = AMINOPEPTIDASE_SUSCEPTIBILITY.get(aa, "?")
    bar = "█" * int(susc * 10)
    report.append(f"| {aa} | {count} | {susc:.1f} {bar} |")
report.append("")

# C-terminal
report.append("### C-terminal Residue Distribution\n")
c_counts = enriched["c_term_aa"].value_counts().sort_index()
report.append("| Residue | Count | Carboxypeptidase Susceptibility |")
report.append("|---------|-------|---------------------------------|")
for aa, count in c_counts.items():
    susc = CARBOXYPEPTIDASE_SUSCEPTIBILITY.get(aa, "?")
    bar = "█" * int(susc * 10)
    report.append(f"| {aa} | {count} | {susc:.1f} {bar} |")
report.append("")

# ---- Exopeptidase Susceptibility Ranking ----
report.append("## 4. Exopeptidase Susceptibility Ranking\n")
report.append("Combined aminopeptidase + carboxypeptidase susceptibility score (0 = resistant, 1 = highly susceptible).\n")

# Protection effect of terminal modifications
report.append("### Effect of Terminal Modifications on Degradation\n")
report.append("The RGEFV libraries systematically test 4 N-terminal and 3 C-terminal modifications:\n")
report.append("- **NH2** (free amine) = unprotected, maximally susceptible to aminopeptidases")
report.append("- **Ac** (acetyl) = blocks aminopeptidase recognition")
report.append("- **βA** (beta-alanine spacer) = non-natural AA, resists all peptidases")
report.append("- **Ac-βA** (acetyl + beta-alanine) = maximum N-terminal protection")
report.append("- **COOH** (free acid) = unprotected C-terminus")
report.append("- **amide** = partial C-terminal protection")
report.append("- **C-βA** = maximum C-terminal protection\n")

# Protection hierarchy
report.append("**Expected protection hierarchy:**")
report.append("- N-terminal: NH2 < Ac < NH2-βA < Ac-βA (increasing protection)")
report.append("- C-terminal: COOH < amide < βA (increasing protection)")
report.append("")

# Top 20 most susceptible
report.append("### Top 20 Most Susceptible to Exopeptidase Degradation\n")
susc_sorted = enriched.dropna(subset=["total_exopeptidase_susceptibility"]).sort_values(
    "total_exopeptidase_susceptibility", ascending=False
)
report.append("| Rank | Sequence | Notation | N-term AA | C-term AA | Score |")
report.append("|------|----------|----------|-----------|-----------|-------|")
for i, (_, row) in enumerate(susc_sorted.head(20).iterrows()):
    report.append(
        f"| {i+1} | {row['sequence']} | {row['full_notation'][:50]} | "
        f"{row['n_term_aa']} | {row['c_term_aa']} | {row['total_exopeptidase_susceptibility']:.3f} |"
    )
report.append("")

# Top 20 most resistant
report.append("### Top 20 Most Resistant to Exopeptidase Degradation\n")
report.append("| Rank | Sequence | Notation | N-term AA | C-term AA | Score |")
report.append("|------|----------|----------|-----------|-----------|-------|")
for i, (_, row) in enumerate(susc_sorted.tail(20).iloc[::-1].iterrows()):
    report.append(
        f"| {i+1} | {row['sequence']} | {row['full_notation'][:50]} | "
        f"{row['n_term_aa']} | {row['c_term_aa']} | {row['total_exopeptidase_susceptibility']:.3f} |"
    )
report.append("")

# ---- RGEFV Library Analysis ----
report.append("## 5. RGEFV Library Deep Dive (Papers 1 & 2)\n")

rgefv = enriched[enriched["scaffold"].isin(["RGEFV-X", "X-RGEFV"])]
report.append(f"**{len(rgefv)} peptides** across {rgefv['library'].nunique()} libraries\n")

# Variable residue effect on properties
report.append("### Variable Residue Effect on Properties\n")
report.append("Each library tests 19 amino acids at the variable position. This shows how the variable residue affects peptide properties.\n")

rgefv_single = rgefv[rgefv["variable_residue"].str.len() == 1].copy()
if len(rgefv_single) > 0:
    var_stats = rgefv_single.groupby("variable_residue").agg({
        "mw_da": "mean",
        "pI": "mean",
        "gravy": "mean",
        "instability_index": "mean",
        "total_exopeptidase_susceptibility": "mean",
    }).round(3)

    report.append("| Variable AA | Avg MW (Da) | Avg pI | Avg GRAVY | Avg Instability | Avg Exo Susceptibility |")
    report.append("|-------------|-------------|--------|-----------|-----------------|----------------------|")
    for aa, row in var_stats.sort_index().iterrows():
        report.append(
            f"| {aa} | {row['mw_da']:.1f} | {row['pI']:.2f} | {row['gravy']:.3f} | "
            f"{row['instability_index']:.1f} | {row['total_exopeptidase_susceptibility']:.3f} |"
        )
    report.append("")

    # N-terminal vs C-terminal variable position (Paper 2)
    report.append("### N-terminal vs C-terminal Variable Position\n")
    report.append("Paper 2 tests RGEFV-X (C-terminal variable) vs X-RGEFV (N-terminal variable).\n")

    for scaffold_name in ["RGEFV-X", "X-RGEFV"]:
        subset = rgefv_single[rgefv_single["scaffold"] == scaffold_name]
        if len(subset) > 0:
            report.append(f"**{scaffold_name}** ({len(subset)} peptides):")
            report.append(f"- MW: {subset['mw_da'].mean():.1f} +/- {subset['mw_da'].std():.1f} Da")
            report.append(f"- pI: {subset['pI'].mean():.2f} +/- {subset['pI'].std():.2f}")
            report.append(f"- GRAVY: {subset['gravy'].mean():.3f} +/- {subset['gravy'].std():.3f}")
            report.append(f"- Exopeptidase susceptibility: {subset['total_exopeptidase_susceptibility'].mean():.3f}")
            report.append("")

# ---- Crosslinker Library Analysis ----
report.append("## 6. Crosslinker Library Deep Dive (Paper 3)\n")

xlink = enriched[enriched["scaffold"] == "KLVAD-XX-ASAE"].copy()
report.append(f"**{len(xlink)} peptides** — KLVAD-X1X2-ASAE combinatorial library\n")
report.append("These are MMP-cleavable crosslinker peptides for cell-responsive hydrogels. ")
report.append("The X1X2 dipeptide at the cleavage site determines how fast cells can degrade the gel.\n")

if "mmp_cleavage_score" in xlink.columns:
    xlink_valid = xlink.dropna(subset=["mmp_cleavage_score"])

    report.append("### MMP Cleavage Site Preferences\n")
    report.append(f"- Mean MMP cleavage score: {xlink_valid['mmp_cleavage_score'].mean():.3f}")
    report.append(f"- Score range: {xlink_valid['mmp_cleavage_score'].min():.3f} - {xlink_valid['mmp_cleavage_score'].max():.3f}")
    report.append("")

    # Top 20 most cleavable
    report.append("### Top 20 Most MMP-Cleavable Variants\n")
    report.append("| Rank | Dipeptide | Sequence | MMP Score | Bulk | Hydrophobicity |")
    report.append("|------|-----------|----------|-----------|------|----------------|")
    top_mmp = xlink_valid.sort_values("mmp_cleavage_score", ascending=False).head(20)
    for i, (_, row) in enumerate(top_mmp.iterrows()):
        report.append(
            f"| {i+1} | {row['variable_residue']} | {row['sequence']} | "
            f"{row['mmp_cleavage_score']:.3f} | {row['cleavage_site_bulk']:.2f} | "
            f"{row['dipeptide_hydrophobicity']:.2f} |"
        )
    report.append("")

    # Top 20 most resistant
    report.append("### Top 20 Most MMP-Resistant Variants\n")
    report.append("| Rank | Dipeptide | Sequence | MMP Score | Bulk | Hydrophobicity |")
    report.append("|------|-----------|----------|-----------|------|----------------|")
    bot_mmp = xlink_valid.sort_values("mmp_cleavage_score", ascending=True).head(20)
    for i, (_, row) in enumerate(bot_mmp.iterrows()):
        report.append(
            f"| {i+1} | {row['variable_residue']} | {row['sequence']} | "
            f"{row['mmp_cleavage_score']:.3f} | {row['cleavage_site_bulk']:.2f} | "
            f"{row['dipeptide_hydrophobicity']:.2f} |"
        )
    report.append("")

    # Key finding: KLVADLMASAE (the lead from Paper 3)
    lead = xlink_valid[xlink_valid["variable_residue"] == "LM"]
    if len(lead) > 0:
        lead_row = lead.iloc[0]
        report.append("### Key Finding: KLVADLMASAE (Paper 3 Optimized Lead)\n")
        report.append(f"- **Dipeptide:** LM (Leu-Met)")
        report.append(f"- **MMP cleavage score:** {lead_row['mmp_cleavage_score']:.3f}")
        report.append(f"- **Rank:** {(xlink_valid['mmp_cleavage_score'] >= lead_row['mmp_cleavage_score']).sum()} / {len(xlink_valid)}")
        report.append(f"- **Cleavage site bulk:** {lead_row['cleavage_site_bulk']:.2f}")
        report.append(f"- **Dipeptide hydrophobicity:** {lead_row['dipeptide_hydrophobicity']:.2f}")
        report.append(f"- **MW:** {lead_row['mw_da']:.1f} Da")
        report.append(f"- **pI:** {lead_row['pI']:.2f}")
        report.append(f"- **GRAVY:** {lead_row['gravy']:.3f}")
        report.append("")
        report.append("The LM dipeptide was identified via split-and-pool screening as optimal for ")
        report.append("cell-mediated hydrogel degradation — balancing MMP accessibility with gel stability.\n")

    # PanMMP benchmark comparison
    panmmp = enriched[enriched["sequence"] == "GPQGIWGQ"]
    if len(panmmp) > 0:
        pm = panmmp.iloc[0]
        report.append("### Benchmark: GPQGIWGQ (PanMMP crosslinker)\n")
        report.append(f"- **MW:** {pm['mw_da']:.1f} Da")
        report.append(f"- **pI:** {pm['pI']:.2f}")
        report.append(f"- **GRAVY:** {pm['gravy']:.3f}")
        report.append(f"- **Length:** {pm['length']} aa")
        report.append(f"- This is the standard MMP-cleavable crosslinker used in most hydrogel literature.")
        report.append(f"- Paper 3's KLVADLMASAE was designed to improve on this benchmark.\n")

# ---- Amino Acid Composition Across Libraries ----
report.append("## 7. Amino Acid Composition Patterns\n")

aa_cols = [f"aa_{aa}" for aa in sorted(VALID_AA)]
existing_aa_cols = [c for c in aa_cols if c in enriched.columns]

if existing_aa_cols:
    # Overall composition
    overall = enriched[existing_aa_cols].mean()
    report.append("### Average Amino Acid Composition (all 618 peptides)\n")
    report.append("| Amino Acid | Avg Fraction | Category |")
    report.append("|------------|-------------|----------|")
    for col in sorted(existing_aa_cols, key=lambda c: -overall[c]):
        aa = col.replace("aa_", "")
        cats = [cat for cat, aas in AA_CATEGORIES.items() if aa in aas]
        cat_str = ", ".join(cats) if cats else ""
        report.append(f"| {aa} | {overall[col]:.4f} | {cat_str} |")
    report.append("")

    # Enrichment in crosslinker vs RGEFV libraries
    report.append("### Composition: RGEFV Libraries vs Crosslinker Library\n")
    rgefv_comp = enriched[enriched["scaffold"].isin(["RGEFV-X", "X-RGEFV"])][existing_aa_cols].mean()
    xlink_comp = enriched[enriched["scaffold"] == "KLVAD-XX-ASAE"][existing_aa_cols].mean()

    report.append("| AA | RGEFV Libraries | Crosslinker | Enrichment (XL/RGEFV) |")
    report.append("|----|-----------------|-------------|----------------------|")
    for col in sorted(existing_aa_cols):
        aa = col.replace("aa_", "")
        r_val = rgefv_comp[col]
        x_val = xlink_comp[col]
        enrichment = x_val / r_val if r_val > 0 else float('inf')
        marker = " **" if enrichment > 1.5 or enrichment < 0.67 else ""
        report.append(f"| {aa} | {r_val:.4f} | {x_val:.4f} | {enrichment:.2f}x{marker} |")
    report.append("")

# ---- Category Composition ----
report.append("## 8. Physicochemical Category Analysis\n")
cat_cols = [c for c in enriched.columns if c.startswith("frac_")]
if cat_cols:
    report.append("### Average Category Fractions by Library Type\n")

    for scaffold_name in ["RGEFV-X", "X-RGEFV", "KLVAD-XX-ASAE"]:
        subset = enriched[enriched["scaffold"] == scaffold_name]
        if len(subset) > 0:
            report.append(f"\n**{scaffold_name}** ({len(subset)} peptides):\n")
            report.append("| Category | Mean Fraction |")
            report.append("|----------|---------------|")
            for col in sorted(cat_cols):
                cat_name = col.replace("frac_", "").replace("chargedpos", "Charged+").replace("chargedneg", "Charged-").title()
                report.append(f"| {cat_name} | {subset[col].mean():.3f} |")

    report.append("")

# ---- Summary Statistics Table ----
report.append("## 9. Summary Statistics by Library\n")
report.append("| Library | n | Avg MW | Avg pI | Avg GRAVY | Avg II | Avg Exo Susc |")
report.append("|---------|---|--------|--------|-----------|--------|-------------|")

for lib, group in enriched.groupby("library"):
    n = len(group)
    report.append(
        f"| {lib[:35]} | {n} | {group['mw_da'].mean():.0f} | {group['pI'].mean():.2f} | "
        f"{group['gravy'].mean():.3f} | {group['instability_index'].mean():.1f} | "
        f"{group['total_exopeptidase_susceptibility'].mean():.3f} |"
    )
report.append("")

# ---- Correlations ----
report.append("## 10. Property Correlations\n")
num_cols = ["mw_da", "pI", "gravy", "instability_index", "aromaticity", "net_charge_ph7",
            "total_exopeptidase_susceptibility"]
existing_num = [c for c in num_cols if c in enriched.columns]
corr = enriched[existing_num].corr().round(3)

report.append("| | " + " | ".join(existing_num) + " |")
report.append("|" + "---|" * (len(existing_num) + 1))
for idx in existing_num:
    vals = " | ".join(str(corr.loc[idx, c]) for c in existing_num)
    report.append(f"| {idx} | {vals} |")
report.append("")

# Notable correlations
report.append("### Notable Correlations\n")
for i, col1 in enumerate(existing_num):
    for col2 in existing_num[i+1:]:
        r = corr.loc[col1, col2]
        if abs(r) > 0.5:
            direction = "positive" if r > 0 else "negative"
            report.append(f"- **{col1}** vs **{col2}**: r = {r:.3f} ({direction})")
report.append("")

# ---- Methods ----
report.append("## Methods\n")
report.append("- Molecular weight, pI, GRAVY, instability index, aromaticity, and secondary structure fractions computed using BioPython ProteinAnalysis")
report.append("- Hydrophobicity: Kyte-Doolittle scale")
report.append("- Exopeptidase susceptibility: literature-derived scores for aminopeptidase (N-terminal) and carboxypeptidase (C-terminal) substrate preferences")
report.append("- MMP cleavage scores: based on published MMP substrate specificity profiles (P1/P1' position preferences)")
report.append("- Net charge: approximate at pH 7 (K, R = +1; H = +0.1; D, E = -1)")
report.append("- Instability Index < 40 = predicted stable (Guruprasad et al., 1990)")
report.append("")

report.append("## Data Sources\n")
report.append("1. Rozans SJ et al. ACS Biomater Sci Eng 2024; 10:4916-4926 (RGEFV degradation libraries)")
report.append("2. Rozans SJ et al. J Biomed Mater Res A 2025; e37864 (LC-MS assay optimization)")
report.append("3. Wu Y, Rozans SJ et al. Adv Healthcare Mater 2025; e2501932 (crosslinker optimization)")
report.append("")

# Write report
report_path = OUT_DIR / "rozans-618-analysis.md"
with open(report_path, "w") as f:
    f.write("\n".join(report))

print(f"\nReport saved: {report_path}")
print(f"Enriched CSV saved: {enriched_path}")
print(f"\nDone! {len(enriched)} peptides analyzed with {len(props_df.columns)} computed properties each.")

# Quick summary to stdout
print("\n" + "=" * 70)
print("QUICK SUMMARY")
print("=" * 70)
print(f"Total peptides: {len(enriched)}")
print(f"MW range: {enriched['mw_da'].min():.0f} - {enriched['mw_da'].max():.0f} Da")
print(f"pI range: {enriched['pI'].min():.2f} - {enriched['pI'].max():.2f}")
print(f"GRAVY range: {enriched['gravy'].min():.3f} - {enriched['gravy'].max():.3f}")
print(f"Stable (II<40): {len(stable)} | Unstable: {len(unstable)}")
print(f"Most susceptible to exopeptidases: {susc_sorted.iloc[0]['sequence']} ({susc_sorted.iloc[0]['full_notation']})")
print(f"Most resistant: {susc_sorted.iloc[-1]['sequence']} ({susc_sorted.iloc[-1]['full_notation']})")
if "mmp_cleavage_score" in xlink.columns:
    top1 = top_mmp.iloc[0]
    print(f"Top MMP-cleavable crosslinker: KLVAD-{top1['variable_residue']}-ASAE (score: {top1['mmp_cleavage_score']:.3f})")
    bot1 = bot_mmp.iloc[0]
    print(f"Most MMP-resistant crosslinker: KLVAD-{bot1['variable_residue']}-ASAE (score: {bot1['mmp_cleavage_score']:.3f})")
