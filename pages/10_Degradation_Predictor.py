import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from Bio.SeqUtils import molecular_weight
from Bio.SeqUtils.ProtParam import ProteinAnalysis

st.set_page_config(page_title="Peptide Degradation Predictor", page_icon="⚗️", layout="wide")
st.title("Peptide Degradation Predictor")
st.markdown(
    "Predict exopeptidase susceptibility based on terminal chemistry and residue identity. "
    "Calibrated against published data from Rozans et al. (ACS Biomater Sci Eng 2024)."
)

# ============================================================
# CALIBRATION DATA (approximate, extracted from Rozans 2024 figures)
# NOTE: These values were read from bar charts in Paper 1, not from
# published tables. The relative ordering is reliable but exact
# numeric values are approximate. Std devs represent variation across
# the 19-AA variable residue library, not measurement uncertainty.
# ============================================================

# N-terminal modification: fraction remaining at 48h (approximate)
NTERM_PROTECTION = {
    "NH2": {"fraction_remaining": 0.20, "std": 0.23, "rank": 4, "description": "Free amine — no protection"},
    "N-βA": {"fraction_remaining": 0.62, "std": 0.17, "rank": 3, "description": "Beta-alanine spacer — moderate protection"},
    "Ac": {"fraction_remaining": 0.82, "std": 0.15, "rank": 1, "description": "Acetyl cap — strong protection (tied with Ac-βA)"},
    "Ac-βA": {"fraction_remaining": 0.81, "std": 0.12, "rank": 2, "description": "Acetyl + beta-alanine — strong protection"},
}

# C-terminal modification: fraction remaining at 48h
CTERM_PROTECTION = {
    "COOH": {"fraction_remaining": 0.36, "std": 0.15, "rank": 3, "description": "Free carboxylic acid — poor protection"},
    "amide": {"fraction_remaining": 0.73, "std": 0.11, "rank": 2, "description": "Amide cap — moderate protection"},
    "C-βA": {"fraction_remaining": 0.83, "std": 0.07, "rank": 1, "description": "Beta-alanine spacer — best protection"},
}

# Cell-type degradation profiles (averaged across all chemistries, 48h)
CELL_TYPE_PROFILES = {
    "hMSC": {"aggressiveness": 0.85, "description": "Human mesenchymal stem cells — most degradative", "avg_remaining": 0.18},
    "hUVEC": {"aggressiveness": 0.45, "description": "Human umbilical vein endothelial cells — moderate", "avg_remaining": 0.57},
    "Macrophage": {"aggressiveness": 0.20, "description": "Primary macrophages — least degradative", "avg_remaining": 0.80},
    "THP-1": {"aggressiveness": 0.30, "description": "THP-1 monocyte cell line", "avg_remaining": 0.70},
}

# Amino acid-specific effects on degradation (relative to average)
# Based on Paper 1 findings: effect of variable residue at N-terminus
AA_NTERM_EFFECT = {
    'G': 0.0, 'A': 0.0, 'V': 0.0, 'L': -0.05, 'I': 0.0,
    'P': 0.15,  # Proline slows N-terminal degradation
    'F': -0.05, 'W': -0.10,  # Trp: faster degradation noted
    'M': 0.0, 'S': 0.0, 'T': 0.0,
    'N': 0.0, 'Q': 0.0,
    'D': 0.10, 'E': 0.10,  # Negative charge slows degradation
    'K': -0.10, 'R': -0.10,  # Positive charge accelerates
    'H': -0.20,  # HISTIDINE EXCEPTION: degraded even when acetylated
    'Y': 0.0,
}

# C-terminal effects
AA_CTERM_EFFECT = {
    'G': 0.0, 'A': 0.0, 'V': 0.0, 'L': -0.05, 'I': 0.0,
    'P': 0.20,  # Proline strongly resists carboxypeptidases
    'F': -0.08, 'W': -0.10,
    'M': -0.05, 'S': 0.0, 'T': 0.0,
    'N': 0.0, 'Q': 0.0,
    'D': 0.12, 'E': 0.12,  # Acidic residues resist carboxypeptidases
    'K': -0.08, 'R': -0.05,
    'H': -0.15,  # Histidine exception
    'Y': -0.05,
}


def predict_degradation(sequence, n_mod, c_mod, cell_type, concentration_uM=500):
    """Predict fraction remaining at 48h based on calibrated model."""
    clean = "".join(c for c in sequence.upper() if c in set("ACDEFGHIKLMNPQRSTVWY"))
    if len(clean) < 2:
        return None

    # Base prediction from terminal chemistry
    n_base = NTERM_PROTECTION.get(n_mod, NTERM_PROTECTION["NH2"])["fraction_remaining"]
    c_base = CTERM_PROTECTION.get(c_mod, CTERM_PROTECTION["COOH"])["fraction_remaining"]

    # Average N and C terminal contributions
    base_remaining = (n_base + c_base) / 2

    # Amino acid effects
    n_term_aa = clean[0]
    c_term_aa = clean[-1]
    aa_n_effect = AA_NTERM_EFFECT.get(n_term_aa, 0)
    aa_c_effect = AA_CTERM_EFFECT.get(c_term_aa, 0)

    # Cell type scaling
    cell = CELL_TYPE_PROFILES.get(cell_type, CELL_TYPE_PROFILES["hMSC"])
    cell_factor = cell["aggressiveness"]

    # Concentration effect (lower conc = faster degradation)
    # Based on Paper 1: 19.5 uM degrades much faster than 5000 uM
    conc_factor = min(1.0, 0.5 + 0.5 * (concentration_uM / 1000))

    # Combined prediction
    predicted = base_remaining + aa_n_effect + aa_c_effect
    # Scale by cell aggressiveness (more aggressive = lower remaining)
    predicted = predicted * (1 - cell_factor * 0.3) * conc_factor

    # PEGylation bonus (if peptide is long enough to have PEG context)
    if len(clean) > 10:
        predicted *= 1.05  # PEGylation slightly protects

    # Clamp to [0, 1]
    predicted = max(0.0, min(1.0, predicted))

    return {
        "fraction_remaining_48h": round(predicted, 3),
        "n_term_contribution": round(n_base, 3),
        "c_term_contribution": round(c_base, 3),
        "n_term_aa_effect": round(aa_n_effect, 3),
        "c_term_aa_effect": round(aa_c_effect, 3),
        "cell_aggressiveness": cell_factor,
        "stability_class": "Stable" if predicted > 0.7 else "Moderate" if predicted > 0.4 else "Susceptible",
        "warnings": [],
    }


# ============================================================
# UI
# ============================================================

tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Analysis", "Calibration Data"])

with tab1:
    st.markdown("### Predict degradation for a single peptide")

    col1, col2 = st.columns(2)
    with col1:
        sequence = st.text_input("Peptide sequence", value="RGEFVL", placeholder="e.g. RGEFVL, GRGDS, KLVADLMASAE")
        n_mod = st.selectbox("N-terminal modification", list(NTERM_PROTECTION.keys()), index=0)
        c_mod = st.selectbox("C-terminal modification", list(CTERM_PROTECTION.keys()), index=0)
    with col2:
        cell_type = st.selectbox("Cell type", list(CELL_TYPE_PROFILES.keys()), index=0)
        concentration = st.slider("Peptide concentration (uM)", 10, 5000, 500, step=10)

    if st.button("Predict", type="primary"):
        result = predict_degradation(sequence, n_mod, c_mod, cell_type, concentration)
        if result:
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                color = "green" if result["fraction_remaining_48h"] > 0.7 else "orange" if result["fraction_remaining_48h"] > 0.4 else "red"
                st.metric("Predicted Fraction Remaining (48h)", f"{result['fraction_remaining_48h']:.1%}")
            with col_b:
                st.metric("Stability Class", result["stability_class"])
            with col_c:
                st.metric("Cell Aggressiveness", f"{result['cell_aggressiveness']:.0%}")

            st.markdown("#### Contribution Breakdown")
            breakdown = pd.DataFrame([
                {"Factor": f"N-terminal ({n_mod})", "Contribution": result["n_term_contribution"],
                 "Info": NTERM_PROTECTION[n_mod]["description"]},
                {"Factor": f"C-terminal ({c_mod})", "Contribution": result["c_term_contribution"],
                 "Info": CTERM_PROTECTION[c_mod]["description"]},
                {"Factor": f"N-term AA ({sequence[0] if sequence else '?'})", "Contribution": result["n_term_aa_effect"],
                 "Info": "Positive = protective, Negative = accelerates degradation"},
                {"Factor": f"C-term AA ({sequence[-1] if sequence else '?'})", "Contribution": result["c_term_aa_effect"],
                 "Info": "Positive = protective, Negative = accelerates degradation"},
            ])
            st.dataframe(breakdown, hide_index=True)

            # Compare across all cell types
            st.markdown("#### Prediction Across Cell Types")
            cell_results = []
            for ct in CELL_TYPE_PROFILES:
                r = predict_degradation(sequence, n_mod, c_mod, ct, concentration)
                cell_results.append({
                    "Cell Type": ct,
                    "Fraction Remaining": r["fraction_remaining_48h"],
                    "Stability": r["stability_class"],
                })
            cell_df = pd.DataFrame(cell_results)
            fig = px.bar(cell_df, x="Cell Type", y="Fraction Remaining", color="Stability",
                         color_discrete_map={"Stable": "green", "Moderate": "orange", "Susceptible": "red"})
            fig.update_layout(height=350, margin=dict(t=20), yaxis=dict(range=[0, 1]))
            st.plotly_chart(fig, use_container_width=True)

            # Compare across terminal modifications
            st.markdown("#### Effect of Terminal Chemistry")
            mod_results = []
            for nm in NTERM_PROTECTION:
                for cm in CTERM_PROTECTION:
                    r = predict_degradation(sequence, nm, cm, cell_type, concentration)
                    mod_results.append({
                        "N-terminal": nm, "C-terminal": cm,
                        "Fraction Remaining": r["fraction_remaining_48h"],
                    })
            mod_df = pd.DataFrame(mod_results)
            pivot = mod_df.pivot(index="N-terminal", columns="C-terminal", values="Fraction Remaining")
            fig = px.imshow(pivot, color_continuous_scale="RdYlGn", zmin=0, zmax=1,
                            labels={"color": "Fraction Remaining"},
                            aspect="auto")
            fig.update_layout(height=350, margin=dict(t=20))
            st.plotly_chart(fig, use_container_width=True)

            # Histidine warning
            clean = "".join(c for c in sequence.upper() if c in set("ACDEFGHIKLMNPQRSTVWY"))
            if clean and (clean[0] == 'H' or clean[-1] == 'H'):
                st.warning(
                    "**Histidine Exception:** Rozans et al. found that histidine-containing peptides "
                    "are substantially degraded by all three cell types, even when acetylated. "
                    "This is a notable exception to the general protection hierarchy. "
                    "Our model accounts for this but the actual degradation may be even faster than predicted."
                )

with tab2:
    st.markdown("### Batch predictions")
    st.markdown("Paste sequences (one per line) with optional terminal chemistry.")

    batch_input = st.text_area(
        "Sequences",
        value="GRGDS\nRGEFVL\nRGEFVP\nRGEFVH\nRGEFVD\nIVKVA\nKLVADLMASAE",
        height=200,
    )
    batch_nmod = st.selectbox("N-terminal (all)", list(NTERM_PROTECTION.keys()), index=0, key="batch_n")
    batch_cmod = st.selectbox("C-terminal (all)", list(CTERM_PROTECTION.keys()), index=0, key="batch_c")
    batch_cell = st.selectbox("Cell type (all)", list(CELL_TYPE_PROFILES.keys()), index=0, key="batch_cell")

    if st.button("Run Batch", type="primary"):
        lines = [l.strip().upper() for l in batch_input.strip().split("\n") if l.strip()]
        results = []
        for seq in lines:
            r = predict_degradation(seq, batch_nmod, batch_cmod, batch_cell)
            if r:
                analysis = ProteinAnalysis("".join(c for c in seq if c in set("ACDEFGHIKLMNPQRSTVWY")))
                results.append({
                    "Sequence": seq,
                    "Fraction Remaining (48h)": r["fraction_remaining_48h"],
                    "Stability": r["stability_class"],
                    "N-term AA": seq[0],
                    "C-term AA": seq[-1],
                    "MW (Da)": round(molecular_weight("".join(c for c in seq if c in set("ACDEFGHIKLMNPQRSTVWY")), "protein"), 1),
                    "pI": round(analysis.isoelectric_point(), 2),
                })
        if results:
            result_df = pd.DataFrame(results).sort_values("Fraction Remaining (48h)")
            st.dataframe(result_df, hide_index=True)

            fig = px.bar(result_df, x="Sequence", y="Fraction Remaining (48h)",
                         color="Stability",
                         color_discrete_map={"Stable": "green", "Moderate": "orange", "Susceptible": "red"})
            fig.update_layout(height=400, margin=dict(t=20), yaxis=dict(range=[0, 1]))
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("### Calibration Data (from Rozans et al. 2024)")
    st.markdown("All values are fraction remaining at 48h in PEG hydrogels.")

    st.markdown("#### N-terminal Protection Hierarchy (hMSCs)")
    n_df = pd.DataFrame([
        {"Modification": k, "Fraction Remaining": v["fraction_remaining"], "Std": v["std"], "Description": v["description"]}
        for k, v in NTERM_PROTECTION.items()
    ]).sort_values("Fraction Remaining", ascending=False)
    st.dataframe(n_df, hide_index=True)

    fig = px.bar(n_df, x="Modification", y="Fraction Remaining", error_y="Std",
                 color="Fraction Remaining", color_continuous_scale="RdYlGn")
    fig.update_layout(height=300, margin=dict(t=20), yaxis=dict(range=[0, 1]))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### C-terminal Protection Hierarchy (hMSCs)")
    c_df = pd.DataFrame([
        {"Modification": k, "Fraction Remaining": v["fraction_remaining"], "Std": v["std"], "Description": v["description"]}
        for k, v in CTERM_PROTECTION.items()
    ]).sort_values("Fraction Remaining", ascending=False)
    st.dataframe(c_df, hide_index=True)

    fig = px.bar(c_df, x="Modification", y="Fraction Remaining", error_y="Std",
                 color="Fraction Remaining", color_continuous_scale="RdYlGn")
    fig.update_layout(height=300, margin=dict(t=20), yaxis=dict(range=[0, 1]))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Cell Type Degradation Profiles")
    cell_df = pd.DataFrame([
        {"Cell Type": k, "Aggressiveness": v["aggressiveness"], "Avg Remaining": v["avg_remaining"],
         "Description": v["description"]}
        for k, v in CELL_TYPE_PROFILES.items()
    ]).sort_values("Aggressiveness", ascending=False)
    st.dataframe(cell_df, hide_index=True)

    st.markdown("#### Key Findings from Published Data")
    st.markdown("""
    1. **Terminal chemistry dominates over amino acid identity.** The difference between NH2 (0.20) and Ac-βA (0.81) is far larger than the difference between any two amino acids at the variable position.

    2. **Histidine is a notable exception.** Even acetylated His peptides degrade substantially across all cell types — likely due to imidazole ring recognition by specific proteases.

    3. **Proline at either terminus resists degradation.** Its rigid pyrrolidine ring blocks exopeptidase access.

    4. **Charged residues matter:** Positive (K, R) accelerate degradation; negative (D, E) slow it. Electrostatic interactions with aminopeptidase/carboxypeptidase active sites.

    5. **Cell type matters enormously.** hMSCs are 4x more degradative than macrophages for the same peptide.

    6. **Concentration matters.** Lower peptide concentrations degrade faster (Michaelis-Menten kinetics).

    7. **PEGylation provides modest additional protection** by sterically shielding terminal residues.
    """)

    st.markdown("---")
    st.caption(
        "Calibration data from: Rozans SJ et al. 'Quantifying and Controlling the Proteolytic Degradation "
        "of Cell Adhesion Peptides.' ACS Biomater Sci Eng 2024; 10:4916-4926."
    )
