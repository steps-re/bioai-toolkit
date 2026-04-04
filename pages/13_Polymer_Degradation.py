import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Polymer Degradation Calculator", page_icon="🧱", layout="wide")
st.title("Tyrosine-Derived Polymer Degradation Calculator")
st.markdown(
    "Predict Tg and degradation rate for tyrosine-derived polycarbonate terpolymers. "
    "Equations from Kohn Lab (Rutgers / NJCBM)."
)

# ============================================================
# KOHN LAB QSPR MODELS
# ============================================================

# Source: Srinivasan et al., J Mater Sci: Mater Med (2013), PMC3809329
# DoE model for poly(DTE-co-DT-co-PEG2K carbonate) terpolymers

def predict_tg_doe(peg_pct, dt_pct):
    """Predict Tg (C) from %PEG2K and %DT using DoE polynomial."""
    return (57.8
            - 20.8 * peg_pct
            + 1.7 * peg_pct**2
            + 5.3 * dt_pct
            - 0.1 * dt_pct**2
            - 0.8 * peg_pct * dt_pct)


def predict_mw_retention(peg_pct, dt_pct):
    """Predict fractional MW retention at 24h from %PEG2K and %DT."""
    return (0.575
            - 0.061 * peg_pct
            + 0.011 * peg_pct**2
            - 0.122 * dt_pct
            + 0.005 * dt_pct**2
            + 0.015 * peg_pct * dt_pct)


# Source: Bateman et al., Polymer 48 (2007), PMC2203329
# Mass-per-flexible-bond Tg prediction

MPFB_CLASSES = {
    "I: Polycarbonates + polyarylates": {"A": 7.7505, "C": 116.17, "r2": 0.905},
    "II: Poly(DTE-co-DT-co-PEG carbonate)s": {"A": 13.512, "C": -66.374, "r2": 0.989},
    "III: Poly(I2DTE-co-I2DT-co-PEG carbonate)s": {"A": 8.8031, "C": -130.02, "r2": 0.978},
}

REPEAT_UNITS = {
    "DTE carbonate": {"M": 383.4, "f": 12, "Mf": 31.95},
    "DT carbonate": {"M": 355.4, "f": 10, "Mf": 35.54},
    "CTE carbonate": {"M": 381.4, "f": 9.5, "Mf": 40.15},
    "DTD dodecanedioate": {"M": 692.0, "f": 33, "Mf": 20.97},
    "I2DTE carbonate": {"M": 635.2, "f": 10.5, "Mf": 60.49},
    "I2DT carbonate": {"M": 607.1, "f": 8.5, "Mf": 71.43},
    "PEG1000 carbonate": {"M": 1073.2, "f": 71, "Mf": 15.12},
    "PEG2000 carbonate": {"M": 2086.3, "f": 140, "Mf": 14.90},
}

# Poly(peptide-ester) data from Fung/Pashuck/Kohn 2023
PPE_DATA = [
    {"Polymer": "HTyGlu", "Diacid": "Glutaric", "Mw_kDa": 157.6, "Mn_kDa": 90.9,
     "Tg_C": 33, "Tm_C": 138, "Crystallinity_pct": 32,
     "Diacid_solubility_mg_mL": 1600, "Notes": "Most hydrolytically sensitive"},
    {"Polymer": "HTyAz", "Diacid": "Azelaic", "Mw_kDa": 172.6, "Mn_kDa": 83.6,
     "Tg_C": 4, "Tm_C": 72, "Crystallinity_pct": 60,
     "Diacid_solubility_mg_mL": 2.41, "Notes": "Highest crystallinity"},
    {"Polymer": "HTyPDA", "Diacid": "1,4-Phenylenediacetic", "Mw_kDa": 156.0, "Mn_kDa": 82.4,
     "Tg_C": 51, "Tm_C": 133, "Crystallinity_pct": 0,
     "Diacid_solubility_mg_mL": 0.16, "Notes": "Amorphous — highest Tg"},
    {"Polymer": "HTyDD", "Diacid": "Dodecanedioic", "Mw_kDa": 157.9, "Mn_kDa": 86.9,
     "Tg_C": 6, "Tm_C": 89, "Crystallinity_pct": 36,
     "Diacid_solubility_mg_mL": 0.041, "Notes": "Base polymer for peptide incorporation"},
    {"Polymer": "HTyDD+2%Pep", "Diacid": "Dodecanedioic", "Mw_kDa": 112.8, "Mn_kDa": 76.7,
     "Tg_C": 7, "Tm_C": 90, "Crystallinity_pct": 28,
     "Diacid_solubility_mg_mL": 0.041, "Notes": "2% MMP-cleavable peptide crosslinker"},
    {"Polymer": "HTyDD+8%Pep", "Diacid": "Dodecanedioic", "Mw_kDa": 140.4, "Mn_kDa": 63.0,
     "Tg_C": 12, "Tm_C": 79, "Crystallinity_pct": None,
     "Diacid_solubility_mg_mL": 0.041, "Notes": "8% peptide — broadest dispersity"},
]

# Validation data for DoE model
DOE_VALIDATION = [
    {"Polymer": "E1502", "PEG_pct": 2, "DT_pct": 15, "Tg_meas": 74.0, "MW_ret_meas": 0.801},
    {"Polymer": "E1504", "PEG_pct": 4, "DT_pct": 15, "Tg_meas": 54.0, "MW_ret_meas": 0.718},
    {"Polymer": "E1506", "PEG_pct": 6, "DT_pct": 15, "Tg_meas": 34.0, "MW_ret_meas": 0.618},
    {"Polymer": "E2502", "PEG_pct": 2, "DT_pct": 25, "Tg_meas": 80.0, "MW_ret_meas": 0.645},
    {"Polymer": "E2504", "PEG_pct": 4, "DT_pct": 25, "Tg_meas": 60.0, "MW_ret_meas": 0.550},
    {"Polymer": "E2506", "PEG_pct": 6, "DT_pct": 25, "Tg_meas": 40.0, "MW_ret_meas": 0.480},
    {"Polymer": "E3502", "PEG_pct": 2, "DT_pct": 35, "Tg_meas": 86.0, "MW_ret_meas": 0.538},
    {"Polymer": "E3504", "PEG_pct": 4, "DT_pct": 35, "Tg_meas": 64.0, "MW_ret_meas": 0.440},
    {"Polymer": "E3506", "PEG_pct": 6, "DT_pct": 35, "Tg_meas": 43.0, "MW_ret_meas": 0.415},
]

# ============================================================
# UI
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "DoE Calculator", "MPFB Tg Model", "Poly(peptide-ester) Data", "112-Polymer Library"
])

# ---- DoE Calculator ----
with tab1:
    st.markdown("### Terpolymer Tg + Degradation Predictor")
    st.markdown(
        "Poly(DTE-co-DT-co-PEG2K carbonate) terpolymers. "
        "Equations from Srinivasan et al. (2013)."
    )

    col1, col2 = st.columns(2)
    with col1:
        peg_pct = st.slider("PEG2K content (mol%)", 0.0, 10.0, 4.0, 0.5)
    with col2:
        dt_pct = st.slider("DT content (mol%)", 0.0, 50.0, 25.0, 1.0)

    tg_pred = predict_tg_doe(peg_pct, dt_pct)
    mw_ret = predict_mw_retention(peg_pct, dt_pct)
    mw_loss = (1 - mw_ret) * 100

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Predicted Tg", f"{tg_pred:.1f} C")
    with col_b:
        st.metric("MW Retained (24h)", f"{mw_ret:.1%}")
    with col_c:
        st.metric("MW Loss (24h)", f"{mw_loss:.1f}%")

    st.markdown("#### Design Rules")
    st.markdown("""
    - **More PEG** = lower Tg (plasticizer) + faster degradation (hydrophilic)
    - **More DT** = higher Tg (rigid backbone) + faster degradation (free carboxyl)
    - PEG effect on Tg is ~4x stronger than DT effect
    - Interaction term (PEG x DT) is small — effects are mostly additive
    """)

    # Heatmap
    st.markdown("#### Property Landscape")
    peg_range = np.linspace(1, 8, 30)
    dt_range = np.linspace(10, 40, 30)

    col_h1, col_h2 = st.columns(2)

    with col_h1:
        st.markdown("**Tg (C)**")
        tg_grid = np.array([[predict_tg_doe(p, d) for p in peg_range] for d in dt_range])
        fig = px.imshow(tg_grid, x=np.round(peg_range, 1), y=np.round(dt_range, 1),
                        color_continuous_scale="RdBu_r", aspect="auto",
                        labels={"x": "PEG2K (mol%)", "y": "DT (mol%)", "color": "Tg (C)"})
        fig.update_layout(height=400, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_h2:
        st.markdown("**MW Retention at 24h**")
        mw_grid = np.array([[predict_mw_retention(p, d) for p in peg_range] for d in dt_range])
        fig = px.imshow(mw_grid, x=np.round(peg_range, 1), y=np.round(dt_range, 1),
                        color_continuous_scale="RdYlGn", aspect="auto", zmin=0.2, zmax=1.0,
                        labels={"x": "PEG2K (mol%)", "y": "DT (mol%)", "color": "Fraction Retained"})
        fig.update_layout(height=400, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    # Validation
    st.markdown("#### Model Validation (9 polymers)")
    val_df = pd.DataFrame(DOE_VALIDATION)
    val_df["Tg_pred"] = val_df.apply(lambda r: predict_tg_doe(r["PEG_pct"], r["DT_pct"]), axis=1).round(1)
    val_df["MW_ret_pred"] = val_df.apply(lambda r: predict_mw_retention(r["PEG_pct"], r["DT_pct"]), axis=1).round(3)
    val_df["Tg_error"] = (val_df["Tg_pred"] - val_df["Tg_meas"]).abs().round(1)
    val_df["MW_error"] = (val_df["MW_ret_pred"] - val_df["MW_ret_meas"]).abs().round(3)

    st.dataframe(val_df[["Polymer", "PEG_pct", "DT_pct", "Tg_meas", "Tg_pred", "Tg_error",
                          "MW_ret_meas", "MW_ret_pred", "MW_error"]], hide_index=True)

    avg_tg_err = val_df["Tg_error"].mean()
    avg_mw_err = val_df["MW_error"].mean()
    st.markdown(f"**Mean Tg error:** {avg_tg_err:.1f} C | **Mean MW retention error:** {avg_mw_err:.3f}")

# ---- MPFB Model ----
with tab2:
    st.markdown("### Mass-per-Flexible-Bond Tg Prediction")
    st.markdown(
        "Universal Tg prediction from polymer repeat unit structure. "
        "From Bateman et al. (2007). Works across 132+ polymers."
    )

    st.markdown("#### Method")
    st.latex(r"T_g = A \cdot \left(\frac{M}{f}\right)_p + C")
    st.markdown(
        "Where M = repeat unit molecular weight (g/mol), f = number of flexible bonds, "
        "and A, C are class-specific regression coefficients."
    )

    # Class selector
    polymer_class = st.selectbox("Polymer class", list(MPFB_CLASSES.keys()))
    cls = MPFB_CLASSES[polymer_class]

    st.markdown(f"**A** = {cls['A']}, **C** = {cls['C']} K, **R2** = {cls['r2']}")

    # Copolymer composition
    st.markdown("#### Define copolymer composition")
    st.markdown("Add up to 3 repeat units with mass fractions (must sum to 1.0).")

    available_units = list(REPEAT_UNITS.keys())

    col1, col2, col3 = st.columns(3)
    with col1:
        unit1 = st.selectbox("Repeat unit 1", available_units, index=0)
        frac1 = st.number_input("Mass fraction 1", 0.0, 1.0, 0.7, 0.05)
    with col2:
        unit2 = st.selectbox("Repeat unit 2", ["None"] + available_units, index=2)
        frac2 = st.number_input("Mass fraction 2", 0.0, 1.0, 0.2, 0.05)
    with col3:
        unit3 = st.selectbox("Repeat unit 3", ["None"] + available_units, index=7)
        frac3 = st.number_input("Mass fraction 3", 0.0, 1.0, 0.1, 0.05)

    total_frac = frac1
    mf_weighted = frac1 * REPEAT_UNITS[unit1]["Mf"]

    if unit2 != "None":
        total_frac += frac2
        mf_weighted += frac2 * REPEAT_UNITS[unit2]["Mf"]
    if unit3 != "None":
        total_frac += frac3
        mf_weighted += frac3 * REPEAT_UNITS[unit3]["Mf"]

    if abs(total_frac - 1.0) > 0.01:
        st.warning(f"Mass fractions sum to {total_frac:.2f} — should be 1.0")

    if st.button("Predict Tg", type="primary"):
        tg_k = cls["A"] * mf_weighted + cls["C"]
        tg_c = tg_k - 273.15

        st.metric("Predicted Tg", f"{tg_c:.1f} C ({tg_k:.1f} K)")
        st.markdown(f"**Weighted M/f:** {mf_weighted:.2f} g/(mol*bond)")

    # Reference table
    st.markdown("#### Repeat Unit Parameters")
    ru_df = pd.DataFrame([
        {"Repeat Unit": k, "M (g/mol)": v["M"], "f (flexible bonds)": v["f"], "M/f": v["Mf"]}
        for k, v in REPEAT_UNITS.items()
    ])
    st.dataframe(ru_df, hide_index=True)

# ---- Poly(peptide-ester) Data ----
with tab3:
    st.markdown("### Poly(peptide-ester) Block Copolymers")
    st.markdown(
        "Data from Fung, Cohen, Pashuck, Kohn (J Mater Chem B, 2023). "
        "Enzyme-specific surface resorption via incorporated peptide crosslinkers."
    )

    ppe_df = pd.DataFrame(PPE_DATA)
    st.dataframe(ppe_df, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(ppe_df, x="Polymer", y="Tg_C", color="Diacid",
                     labels={"Tg_C": "Tg (C)"})
        fig.update_layout(height=350, margin=dict(t=20), title="Glass Transition Temperature")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(ppe_df, x="Tg_C", y="Tm_C", color="Diacid", size="Mw_kDa",
                         text="Polymer",
                         labels={"Tg_C": "Tg (C)", "Tm_C": "Tm (C)", "Mw_kDa": "Mw (kDa)"})
        fig.update_traces(textposition="top center")
        fig.update_layout(height=350, margin=dict(t=20), title="Tg vs Tm")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Key Findings")
    st.markdown("""
    - **2% peptide incorporation** is sufficient for enzyme-specific surface resorption
    - **Proteinase K** was the ONLY enzyme that caused film thickness reduction below starting value
    - Diacid solubility correlates with hydrolytic sensitivity (glutaric >> dodecanedioic)
    - Peptide sequence: `mercaptopropionate-GGPMGPWGGC` (MMP-cleavable after Met and Trp)
    - Control peptide: `mercaptopropionate-GGPGGPGGGC` (non-degradable)
    """)

# ---- 112-Polymer Library ----
with tab4:
    st.markdown("### Brocchini-Kohn 112-Polymer Polyarylate Library")
    st.markdown(
        "14 tyrosine-derived diphenols x 8 aliphatic diacids = 112 systematically varied polymers. "
        "From Brocchini et al. (JACS, 1997)."
    )

    st.markdown("""
    #### Library Design

    **14 Diphenols** (tyrosine-derived, vary pendent chain):
    - DTE (ethyl), DTB (butyl), DTH (hexyl), DTO (octyl)
    - Plus branched, oxygenated, and aromatic variants

    **8 Diacids** (vary methylene chain length):
    - Succinic (C2), Glutaric (C3), Adipic (C4), Suberic (C6)
    - Azelaic (C7), Sebacic (C8), Dodecanedioic (C10)
    - Plus oxygenated variants

    #### Property Ranges Across Library

    | Property | Min | Max |
    |----------|-----|-----|
    | Tg | 2 C | 91 C |
    | Contact angle (air-water) | 64 deg | 101 deg |
    | Tensile strength | ~6 MPa | ~45 MPa |
    | Young's modulus | ~0.3 GPa | ~1.7 GPa |

    #### Key Finding
    Cell proliferation correlates **inversely** with surface hydrophobicity — EXCEPT for polymers
    with oxygen-containing diacids, which are uniformly good substrates regardless of contact angle.
    This exception breaks simple QSPR models and highlights the importance of backbone chemistry
    beyond surface energy.
    """)

    st.markdown("---")
    st.caption(
        "Sources: Srinivasan et al. J Mater Sci Mater Med 2013 (PMC3809329) | "
        "Bateman et al. Polymer 2007 (PMC2203329) | "
        "Fung et al. J Mater Chem B 2023 (PMC10519181) | "
        "Brocchini et al. JACS 1997"
    )
