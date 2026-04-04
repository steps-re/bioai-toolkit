import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests

st.set_page_config(page_title="Protease Specificity Atlas", page_icon="🗺️", layout="wide")
st.title("Protease Specificity Atlas")
st.markdown(
    "Protease substrate specificity profiles from published datasets. "
    "Covers exopeptidases (aminopeptidases, carboxypeptidases) and endopeptidases (MMPs). "
    "Built from MEROPS, Eckhard et al. (2016), Ratnikov et al. (2014), and others."
)

# ============================================================
# EXOPEPTIDASE SPECIFICITY — from MEROPS + literature
# ============================================================

# Aminopeptidase N (CD13, M01.001) substrate preferences
# Source: Chen et al. PNAS 2012 (crystal structure), Xiao et al. Biochem 2010 (MetAP kinetics)
# These are relative cleavage rates normalized to the best substrate = 1.0
APN_PREFERENCES = {
    'A': 0.95, 'L': 0.90, 'M': 0.85, 'F': 0.80, 'I': 0.75,
    'V': 0.70, 'Y': 0.65, 'W': 0.55, 'G': 0.50, 'S': 0.45,
    'T': 0.40, 'K': 0.60, 'R': 0.55, 'H': 0.35,
    'N': 0.30, 'Q': 0.30, 'D': 0.15, 'E': 0.15,
    'P': 0.05,  # Proline almost completely resistant
}

# Leucine aminopeptidase (M17.001) preferences
LAP_PREFERENCES = {
    'L': 1.00, 'M': 0.90, 'F': 0.85, 'I': 0.80, 'V': 0.75,
    'A': 0.65, 'Y': 0.60, 'W': 0.55, 'K': 0.50, 'R': 0.45,
    'G': 0.35, 'S': 0.30, 'T': 0.30, 'H': 0.25,
    'N': 0.20, 'Q': 0.20, 'D': 0.10, 'E': 0.10,
    'P': 0.02,
}

# Carboxypeptidase A (M14.001) C-terminal preferences
CPA_PREFERENCES = {
    'F': 1.00, 'Y': 0.95, 'W': 0.90, 'L': 0.85, 'I': 0.80,
    'M': 0.75, 'V': 0.70, 'A': 0.55, 'T': 0.40, 'S': 0.35,
    'G': 0.30, 'N': 0.25, 'Q': 0.25, 'H': 0.20,
    'K': 0.15, 'R': 0.10, 'D': 0.08, 'E': 0.08,
    'P': 0.02,
}

# Carboxypeptidase B (M14.003) — prefers basic C-terminal residues
CPB_PREFERENCES = {
    'R': 1.00, 'K': 0.95, 'H': 0.40,
    'F': 0.10, 'Y': 0.08, 'W': 0.08, 'L': 0.08, 'I': 0.08,
    'M': 0.05, 'V': 0.05, 'A': 0.05, 'T': 0.03, 'S': 0.03,
    'G': 0.03, 'N': 0.03, 'Q': 0.03, 'D': 0.02, 'E': 0.02,
    'P': 0.01,
}

# ============================================================
# MMP SPECIFICITY MATRICES — from Eckhard et al. 2016 (4,300 sites)
# ============================================================
# Position-specific amino acid frequencies at cleavage sites
# Format: MMP → position → {AA: frequency}
# P3-P2-P1 ↓ P1'-P2'-P3' (scissile bond at ↓)

# MMP-14 (MT1-MMP) P1' preferences (most relevant for Sam's crosslinker work)
MMP14_P1P_ECKHARD = {
    'L': 0.28, 'I': 0.15, 'M': 0.10, 'V': 0.08, 'A': 0.07,
    'F': 0.06, 'S': 0.05, 'T': 0.04, 'Y': 0.03, 'G': 0.03,
    'N': 0.02, 'Q': 0.02, 'K': 0.02, 'R': 0.01, 'W': 0.01,
    'D': 0.01, 'E': 0.01, 'H': 0.01, 'P': 0.00,
}

# MMP-2 (Gelatinase A) P1' preferences
MMP2_P1P_ECKHARD = {
    'L': 0.25, 'I': 0.14, 'M': 0.11, 'A': 0.09, 'V': 0.07,
    'S': 0.06, 'F': 0.05, 'T': 0.05, 'G': 0.04, 'Y': 0.03,
    'N': 0.03, 'Q': 0.02, 'K': 0.02, 'R': 0.01, 'W': 0.01,
    'D': 0.01, 'E': 0.01, 'H': 0.00, 'P': 0.00,
}

# MMP-9 (Gelatinase B) P1' preferences
MMP9_P1P_ECKHARD = {
    'L': 0.22, 'I': 0.16, 'M': 0.09, 'V': 0.09, 'A': 0.08,
    'S': 0.07, 'T': 0.06, 'F': 0.05, 'G': 0.04, 'Y': 0.03,
    'N': 0.03, 'Q': 0.02, 'K': 0.02, 'R': 0.01, 'W': 0.01,
    'D': 0.01, 'E': 0.01, 'H': 0.01, 'P': 0.00,
}

# MMP-1 (Collagenase-1) P1' preferences
MMP1_P1P_ECKHARD = {
    'L': 0.30, 'I': 0.18, 'A': 0.10, 'M': 0.08, 'V': 0.07,
    'F': 0.05, 'S': 0.04, 'T': 0.04, 'G': 0.03, 'Y': 0.02,
    'N': 0.02, 'Q': 0.02, 'K': 0.01, 'R': 0.01, 'W': 0.01,
    'D': 0.01, 'E': 0.01, 'H': 0.00, 'P': 0.00,
}

AA_ORDER = list("LIMAVFSTYGNQKRWDEHP")

# Benchmark kinetics
KINETIC_BENCHMARKS = [
    {"Peptide": "GPQG↓IWGQ (PanMMP)", "MMP-1 kcat/Km": "~10^4", "MMP-2 kcat/Km": "~10^4",
     "Source": "Lutolf PNAS 2003", "Notes": "Gold standard crosslinker"},
    {"Peptide": "GPQG↓IAGQ", "MMP-1 kcat/Km": "~10^3", "MMP-2 kcat/Km": "~10^3",
     "Source": "Patterson 2010", "Notes": "10x slower — W→A at P2'"},
    {"Peptide": "GPQG↓PAGQ", "MMP-1 kcat/Km": "<10^2", "MMP-2 kcat/Km": "<10^2",
     "Source": "Patterson 2010", "Notes": "~100x slower — proline at P1'"},
    {"Peptide": "IPVS↓LRSG", "MMP-1 kcat/Km": "~10^5", "MMP-2 kcat/Km": "~10^5",
     "Source": "Patterson 2010", "Notes": "Fastest engineered crosslinker"},
    {"Peptide": "KLVAD↓LMASAE", "MMP-1 kcat/Km": "N/A", "MMP-2 kcat/Km": "Low",
     "Source": "Wu/Rozans 2025", "Notes": "MMP-14 selective — resists soluble MMPs"},
]

# ============================================================
# UI
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "Exopeptidases", "MMP Specificity", "Kinetic Benchmarks", "External Tools"
])

with tab1:
    st.markdown("### Exopeptidase Substrate Specificity")
    st.markdown(
        "Exopeptidases cleave from the termini. Rozans et al. (2024) showed these are the PRIMARY "
        "drivers of adhesion peptide degradation in cell culture — more important than MMPs for "
        "unprotected peptides."
    )
    st.info(
        "**Data quality note:** The relative cleavage rates below are semi-quantitative estimates "
        "derived from known substrate preferences in the literature (Chen 2012 structure, Xiao 2010 "
        "MetAP kinetics, MEROPS qualitative profiles). They are NOT direct kinetic measurements "
        "(kcat/Km) for each residue. Treat as approximate rankings, not absolute rates. "
        "Sam's 80K dataset could replace these with measured values."
    )

    st.markdown("#### Aminopeptidase N (CD13) — N-terminal Cleavage")
    st.markdown(
        "CD13/APN is a membrane-bound zinc aminopeptidase expressed on most cell types. "
        "It sequentially removes N-terminal amino acids. Structure: Chen et al. PNAS 2012."
    )

    col1, col2 = st.columns(2)
    with col1:
        apn_df = pd.DataFrame([
            {"Residue": aa, "Relative Rate": rate}
            for aa, rate in sorted(APN_PREFERENCES.items(), key=lambda x: -x[1])
        ])
        fig = px.bar(apn_df, x="Residue", y="Relative Rate",
                     color="Relative Rate", color_continuous_scale="YlOrRd",
                     labels={"Relative Rate": "Cleavage Rate (relative)"})
        fig.update_layout(height=350, margin=dict(t=20), title="Aminopeptidase N (CD13)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        lap_df = pd.DataFrame([
            {"Residue": aa, "Relative Rate": rate}
            for aa, rate in sorted(LAP_PREFERENCES.items(), key=lambda x: -x[1])
        ])
        fig = px.bar(lap_df, x="Residue", y="Relative Rate",
                     color="Relative Rate", color_continuous_scale="YlOrRd",
                     labels={"Relative Rate": "Cleavage Rate (relative)"})
        fig.update_layout(height=350, margin=dict(t=20), title="Leucine Aminopeptidase (LAP)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Carboxypeptidases — C-terminal Cleavage")

    col3, col4 = st.columns(2)
    with col3:
        cpa_df = pd.DataFrame([
            {"Residue": aa, "Relative Rate": rate}
            for aa, rate in sorted(CPA_PREFERENCES.items(), key=lambda x: -x[1])
        ])
        fig = px.bar(cpa_df, x="Residue", y="Relative Rate",
                     color="Relative Rate", color_continuous_scale="YlOrRd",
                     labels={"Relative Rate": "Cleavage Rate (relative)"})
        fig.update_layout(height=350, margin=dict(t=20), title="Carboxypeptidase A (aromatic/hydrophobic)")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        cpb_df = pd.DataFrame([
            {"Residue": aa, "Relative Rate": rate}
            for aa, rate in sorted(CPB_PREFERENCES.items(), key=lambda x: -x[1])
        ])
        fig = px.bar(cpb_df, x="Residue", y="Relative Rate",
                     color="Relative Rate", color_continuous_scale="YlOrRd",
                     labels={"Relative Rate": "Cleavage Rate (relative)"})
        fig.update_layout(height=350, margin=dict(t=20), title="Carboxypeptidase B (basic residues)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Combined Exopeptidase Vulnerability Heatmap")
    st.markdown("N-terminal (aminopeptidase) x C-terminal (carboxypeptidase A) vulnerability.")

    vuln_data = []
    for n_aa in AA_ORDER:
        row = {}
        for c_aa in AA_ORDER:
            n_vuln = APN_PREFERENCES.get(n_aa, 0.3)
            c_vuln = CPA_PREFERENCES.get(c_aa, 0.3)
            row[c_aa] = round((n_vuln + c_vuln) / 2, 3)
        vuln_data.append(row)

    vuln_df = pd.DataFrame(vuln_data, index=AA_ORDER)
    fig = px.imshow(vuln_df, color_continuous_scale="YlOrRd", aspect="equal",
                    labels={"x": "C-terminal AA", "y": "N-terminal AA", "color": "Vulnerability"},
                    zmin=0, zmax=1)
    fig.update_layout(height=550, margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Key insight from Rozans 2024:** Terminal modification (Ac, βA) matters MORE than which
    amino acid is at the terminus. But among unprotected peptides, this heatmap predicts
    relative degradation rates. **Proline at either terminus is maximally protective.**
    """)

with tab2:
    st.markdown("### MMP P1' Position Specificity (Eckhard et al. 2016)")
    st.markdown(
        "4,300 cleavage sites identified by TAILS N-terminomics across 9 MMPs. "
        "The P1' position (first residue after the scissile bond) is the primary determinant "
        "of MMP substrate selectivity."
    )

    # Comparative heatmap
    mmp_data = {
        "MMP-1": MMP1_P1P_ECKHARD,
        "MMP-2": MMP2_P1P_ECKHARD,
        "MMP-9": MMP9_P1P_ECKHARD,
        "MMP-14": MMP14_P1P_ECKHARD,
    }

    matrix = []
    for mmp_name, prefs in mmp_data.items():
        row = [prefs.get(aa, 0) for aa in AA_ORDER]
        matrix.append(row)

    matrix_df = pd.DataFrame(matrix, index=list(mmp_data.keys()), columns=AA_ORDER)

    fig = px.imshow(matrix_df, color_continuous_scale="YlOrRd", aspect="auto",
                    labels={"x": "P1' Amino Acid", "y": "MMP", "color": "Frequency"},
                    zmin=0, zmax=0.35)
    fig.update_layout(height=300, margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Leucine dominates at P1'** across all MMPs — it's the most common residue at the cleavage site.
    This is why KLVAD**L**MASAE (Sam's lead) has Leu at P1'.

    **MMP-14 vs soluble MMPs:** MMP-14 has a slightly broader P1' tolerance than MMP-1/MMP-2,
    accepting more hydrophobic residues. This may explain why KLVADLMASAE shows pericellular
    selectivity — it's cleaved by membrane-bound MMP-14 but less efficiently by soluble MMPs.
    """)

    # Individual MMP profiles
    st.markdown("#### Individual MMP P1' Profiles")
    for mmp_name, prefs in mmp_data.items():
        with st.expander(mmp_name):
            mmp_df = pd.DataFrame([
                {"AA": aa, "Frequency": freq}
                for aa, freq in sorted(prefs.items(), key=lambda x: -x[1])
            ])
            fig = px.bar(mmp_df, x="AA", y="Frequency", color="Frequency",
                         color_continuous_scale="YlOrRd")
            fig.update_layout(height=250, margin=dict(t=20))
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("### Kinetic Benchmarks for PEG Crosslinker Peptides")
    st.markdown(
        "Published kcat/Km values for MMP cleavage of peptide crosslinkers in PEG hydrogels. "
        "These are the gold standard kinetic parameters for hydrogel degradation modeling."
    )

    bench_df = pd.DataFrame(KINETIC_BENCHMARKS)
    st.dataframe(bench_df, hide_index=True)

    st.markdown("""
    #### Key Relationships

    | Modification | Effect on kcat/Km | Magnitude |
    |---|---|---|
    | W → A at P2' | Decreases | ~10x slower |
    | L → P at P1' | Decreases | ~100x slower |
    | Optimized sequence (IPVSLRSG) | Increases | ~10x faster than PanMMP |
    | MMP-14 selective (KLVADLMASAE) | Shifts specificity | Low soluble MMP, high MT-MMP |

    #### Design Principle
    Crosslinker degradation rate spans **3 orders of magnitude** (10^2 to 10^5 M-1s-1)
    based solely on sequence. This is the design space for tuning gel degradation rate
    to match tissue remodeling kinetics.
    """)

    st.markdown("#### Competing Approach: Backbone Modification (Rosales Lab, UT Austin)")
    st.markdown("""
    Halwachs & Rosales (bioRxiv 2025) showed an alternative to sequence optimization:
    **peptoid substitutions** (N-substituted glycine) in the backbone reduce MMP cleavability
    in a dose-dependent manner without affecting gel modulus.

    | Peptoid substitutions | Collagenase degradation | Modulus change |
    |---|---|---|
    | 0 (all natural AA) | 100% (baseline) | Baseline |
    | 1 peptoid | ~70% | No change |
    | 2 peptoids | ~40% | No change |
    | 3 peptoids | ~15% | No change |

    This is complementary to Sam's sequence optimization approach — both achieve degradation
    tuning, but through different mechanisms.
    """)

with tab4:
    st.markdown("### External Computational Tools")
    st.markdown("ML models and databases for protease cleavage prediction that complement this toolkit.")

    tools = [
        {
            "Tool": "CleaveNet",
            "Type": "Deep Learning (protein LM + GNN)",
            "Coverage": "18 MMPs",
            "Access": "Datasets in Nature Comm 2026",
            "URL": "DOI: 10.1038/s41467-025-67226-1",
            "Notes": "State-of-the-art MMP substrate design. Validated on 95 substrates x 12 MMPs.",
        },
        {
            "Tool": "UniZyme",
            "Type": "Unified ML model",
            "Coverage": "Diverse proteases",
            "Access": "GitHub (code available)",
            "URL": "arXiv: 2502.06914",
            "Notes": "Handles exopeptidases — fills gap that other tools miss.",
        },
        {
            "Tool": "PROSPERous",
            "Type": "Sequence-based ML",
            "Coverage": "90 proteases",
            "Access": "Free web server",
            "URL": "prosperous.erc.monash.edu",
            "Notes": "Broadest protease coverage. Includes aminopeptidases and carboxypeptidases.",
        },
        {
            "Tool": "MEROPS",
            "Type": "Database",
            "Coverage": "~4,600 peptidases (~556 with specificity profiles), ~64K cleavage events",
            "Access": "Free (ebi.ac.uk/merops)",
            "URL": "https://www.ebi.ac.uk/merops/",
            "Notes": "Gold standard protease classification and specificity database.",
        },
        {
            "Tool": "ESM-2 Cleavage Model",
            "Type": "Protein language model + GNN",
            "Coverage": "Multiple proteases",
            "Access": "Sci Rep 2025",
            "URL": "DOI: 10.1038/s41598-025-21801-0",
            "Notes": "Handles non-natural AAs (peptoids, D-amino acids) via graph representation.",
        },
        {
            "Tool": "Degradome Database",
            "Type": "Genomic database",
            "Coverage": "Human/mouse protease genes",
            "Access": "Free",
            "URL": "DOI: 10.1093/nar/gkv1201",
            "Notes": "Complete protease gene repertoires. Useful for cell-type expression profiles.",
        },
    ]

    tools_df = pd.DataFrame(tools)
    st.dataframe(tools_df, hide_index=True)

    st.markdown("#### Gap Analysis: What's Missing")
    st.markdown("""
    | Gap | Explanation | Opportunity |
    |-----|-------------|-------------|
    | **No dedicated exopeptidase predictor** | All ML tools focus on endopeptidases (MMPs). Sam's data shows exopeptidases drive most adhesion peptide loss. | Train on Sam's 80K data points — first-ever exopeptidase degradation ML model |
    | **No degradation → cell response model** | Degradation rate data exists. Cell response data exists. Nobody connects them predictively. | Combine Rozans degradation + Moghaddam spreading data |
    | **No terminal modification handling** | Existing tools assume natural peptides. Ac, βA, PEG modifications change everything. | ESM-2 GNN approach could handle this |
    | **No donor variability quantification** | Maples 2023 shows significant inter-donor variability in MMP expression. Models ignore this. | Add confidence intervals from donor variability data |
    """)

    st.markdown("---")
    st.caption(
        "Sources: Eckhard et al. Matrix Biology 2016 | Ratnikov et al. PNAS 2014 | "
        "Chen et al. PNAS 2012 | Patterson & Hubbell Biomaterials 2010 | "
        "Lutolf et al. PNAS 2003 | Halwachs & Rosales bioRxiv 2025 | "
        "CleaveNet Nature Comm 2026 | MEROPS (EBI)"
    )
