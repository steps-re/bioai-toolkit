import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import py3Dmol
from stmol import showmol

st.set_page_config(page_title="Rozans Analysis", page_icon="🔬", layout="wide")
st.title("Rozans Analysis")

# Hero enzyme visualization
col_hero1, col_hero2 = st.columns([1, 2])
with col_hero1:
    view = py3Dmol.view(query="pdb:4FYS", width=350, height=280)
    view.setStyle({}, {"cartoon": {"colorscheme": "spectrum"}})
    view.setStyle({"hetflag": True}, {"stick": {"colorscheme": "greenCarbon", "radius": 0.3}})
    view.setStyle({"elem": "Zn"}, {"sphere": {"radius": 0.8, "color": "gray"}})
    view.setStyle({"elem": "ZN"}, {"sphere": {"radius": 0.8, "color": "gray"}})
    view.setBackgroundColor("white")
    view.zoomTo()
    showmol(view, height=280, width=350)
    st.caption("Aminopeptidase N (CD13) — the enzyme degrading NH2-terminated peptides")
with col_hero2:
    st.markdown(
        "618 peptides from 3 Pashuck Lab publications, analyzed through every tool in this toolkit. "
        "Each tab applies a different analytical lens to the same peptide library."
    )

DATA_PATH = Path(__file__).parent.parent / "data" / "rozans-618-enriched.csv"

# ============================================================
# DATA + MODELS (imported from other pages' logic)
# ============================================================

# Protease specificity profiles (from page 13)
APN_PREFS = {
    'A': 0.95, 'L': 0.90, 'M': 0.85, 'F': 0.80, 'I': 0.75,
    'V': 0.70, 'Y': 0.65, 'W': 0.55, 'G': 0.50, 'S': 0.45,
    'T': 0.40, 'K': 0.60, 'R': 0.55, 'H': 0.35,
    'N': 0.30, 'Q': 0.30, 'D': 0.15, 'E': 0.15, 'P': 0.05,
}
LAP_PREFS = {
    'L': 1.00, 'M': 0.90, 'F': 0.85, 'I': 0.80, 'V': 0.75,
    'A': 0.65, 'Y': 0.60, 'W': 0.55, 'K': 0.50, 'R': 0.45,
    'G': 0.35, 'S': 0.30, 'T': 0.30, 'H': 0.25,
    'N': 0.20, 'Q': 0.20, 'D': 0.10, 'E': 0.10, 'P': 0.02,
}
CPA_PREFS = {
    'F': 1.00, 'Y': 0.95, 'W': 0.90, 'L': 0.85, 'I': 0.80,
    'M': 0.75, 'V': 0.70, 'A': 0.55, 'T': 0.40, 'S': 0.35,
    'G': 0.30, 'N': 0.25, 'Q': 0.25, 'H': 0.20,
    'K': 0.15, 'R': 0.10, 'D': 0.08, 'E': 0.08, 'P': 0.02,
}

# Degradation predictor calibration (from page 8)
NTERM_PROTECTION = {
    "NH2": 0.20, "N-βA": 0.62, "Ac": 0.82, "Ac-βA": 0.81,
}
CTERM_PROTECTION = {
    "COOH": 0.36, "amide": 0.73, "C-βA": 0.83,
}
CELL_AGGRESSIVENESS = {
    "hMSC": 0.85, "hUVEC": 0.45, "Macrophage": 0.20, "THP-1": 0.30,
}
AA_NTERM_EFFECT = {
    'G': 0.0, 'A': 0.0, 'V': 0.0, 'L': -0.05, 'I': 0.0,
    'P': 0.15, 'F': -0.05, 'W': -0.10, 'M': 0.0, 'S': 0.0, 'T': 0.0,
    'N': 0.0, 'Q': 0.0, 'D': 0.10, 'E': 0.10,
    'K': -0.10, 'R': -0.10, 'H': -0.20, 'Y': 0.0,
}
AA_CTERM_EFFECT = {
    'G': 0.0, 'A': 0.0, 'V': 0.0, 'L': -0.05, 'I': 0.0,
    'P': 0.20, 'F': -0.08, 'W': -0.10, 'M': -0.05, 'S': 0.0, 'T': 0.0,
    'N': 0.0, 'Q': 0.0, 'D': 0.12, 'E': 0.12,
    'K': -0.08, 'R': -0.05, 'H': -0.15, 'Y': -0.05,
}

# MMP Eckhard P1' frequencies (from page 13)
MMP_P1P = {
    "MMP-14": {'L': 0.28, 'I': 0.15, 'M': 0.10, 'V': 0.08, 'A': 0.07, 'F': 0.06, 'S': 0.05, 'T': 0.04, 'Y': 0.03, 'G': 0.03, 'N': 0.02, 'Q': 0.02, 'K': 0.02, 'R': 0.01, 'W': 0.01, 'D': 0.01, 'E': 0.01, 'H': 0.01, 'P': 0.00},
    "MMP-2": {'L': 0.25, 'I': 0.14, 'M': 0.11, 'A': 0.09, 'V': 0.07, 'S': 0.06, 'F': 0.05, 'T': 0.05, 'G': 0.04, 'Y': 0.03, 'N': 0.03, 'Q': 0.02, 'K': 0.02, 'R': 0.01, 'W': 0.01, 'D': 0.01, 'E': 0.01, 'H': 0.00, 'P': 0.00},
    "MMP-9": {'L': 0.22, 'I': 0.16, 'M': 0.09, 'V': 0.09, 'A': 0.08, 'S': 0.07, 'T': 0.06, 'F': 0.05, 'G': 0.04, 'Y': 0.03, 'N': 0.03, 'Q': 0.02, 'K': 0.02, 'R': 0.01, 'W': 0.01, 'D': 0.01, 'E': 0.01, 'H': 0.01, 'P': 0.00},
    "MMP-1": {'L': 0.30, 'I': 0.18, 'A': 0.10, 'M': 0.08, 'V': 0.07, 'F': 0.05, 'S': 0.04, 'T': 0.04, 'G': 0.03, 'Y': 0.02, 'N': 0.02, 'Q': 0.02, 'K': 0.01, 'R': 0.01, 'W': 0.01, 'D': 0.01, 'E': 0.01, 'H': 0.00, 'P': 0.00},
}

# Self-assembly residue propensity (from page 12)
RESIDUE_AP = {
    'F': 0.90, 'W': 0.88, 'Y': 0.65, 'I': 0.55, 'L': 0.50,
    'V': 0.45, 'M': 0.40, 'A': 0.25, 'P': 0.20, 'G': 0.15,
    'S': 0.15, 'T': 0.18, 'N': 0.12, 'Q': 0.12,
    'D': 0.08, 'E': 0.08, 'K': 0.10, 'R': 0.10, 'H': 0.20,
}

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


@st.cache_data
def load_and_enrich():
    df = pd.read_csv(DATA_PATH)

    # Add protease atlas scores
    df["apn_score"] = df["n_term_aa"].map(APN_PREFS)
    df["lap_score"] = df["n_term_aa"].map(LAP_PREFS)
    df["cpa_score"] = df["c_term_aa"].map(CPA_PREFS)
    df["combined_exo_atlas"] = ((df["apn_score"].fillna(0) + df["lap_score"].fillna(0)) / 2 +
                                 df["cpa_score"].fillna(0)) / 2

    # Add degradation predictions for each cell type + terminal chemistry combo
    for cell_name, cell_agg in CELL_AGGRESSIVENESS.items():
        for n_mod, n_base in NTERM_PROTECTION.items():
            col_name = f"deg_{cell_name}_{n_mod}_COOH"
            preds = []
            for _, row in df.iterrows():
                clean = "".join(c for c in str(row["sequence"]).upper() if c in VALID_AA)
                if len(clean) < 2:
                    preds.append(None)
                    continue
                base = (n_base + CTERM_PROTECTION["COOH"]) / 2
                aa_n = AA_NTERM_EFFECT.get(clean[0], 0)
                aa_c = AA_CTERM_EFFECT.get(clean[-1], 0)
                pred = (base + aa_n + aa_c) * (1 - cell_agg * 0.3)
                preds.append(round(max(0, min(1, pred)), 3))
            df[col_name] = preds

    # Add self-assembly propensity (average residue AP)
    sa_scores = []
    for seq in df["sequence"]:
        clean = "".join(c for c in str(seq).upper() if c in VALID_AA)
        if len(clean) < 2:
            sa_scores.append(None)
            continue
        avg = sum(RESIDUE_AP.get(c, 0.2) for c in clean) / len(clean)
        sa_scores.append(round(avg, 4))
    df["self_assembly_propensity"] = sa_scores

    # Add MMP-14 P1' score for crosslinkers
    mmp14_scores = []
    for _, row in df.iterrows():
        if row.get("scaffold") == "KLVAD-XX-ASAE" and isinstance(row.get("variable_residue"), str) and len(row["variable_residue"]) == 2:
            x1 = row["variable_residue"][0]
            p1p = MMP_P1P["MMP-14"].get(x1, 0.02)
            mmp14_scores.append(round(p1p, 3))
        else:
            mmp14_scores.append(None)
    df["mmp14_p1p_eckhard"] = mmp14_scores

    return df


df = load_and_enrich()

# ---- Sidebar filters ----
st.sidebar.markdown("### Filters")
papers = ["All"] + sorted(df["paper"].unique().tolist())
selected_paper = st.sidebar.selectbox("Paper", papers)
scaffolds = ["All"] + sorted(df["scaffold"].dropna().unique().tolist())
selected_scaffold = st.sidebar.selectbox("Scaffold", scaffolds)

filtered = df.copy()
if selected_paper != "All":
    filtered = filtered[filtered["paper"] == selected_paper]
if selected_scaffold != "All":
    filtered = filtered[filtered["scaffold"] == selected_scaffold]

st.sidebar.markdown(f"**Showing {len(filtered)} / {len(df)} peptides**")

# ============================================================
# TABS
# ============================================================
tab_overview, tab_biophys, tab_exo, tab_protease, tab_degrade, tab_mmp, tab_selfasm, tab_corr, tab_table, tab_next = st.tabs([
    "Overview",
    "Biophysical Properties",
    "Exopeptidase (Literature)",
    "Protease Atlas (Eckhard/MEROPS)",
    "Degradation Predictor (Rozans '24)",
    "MMP Crosslinker (Wu '25)",
    "Self-Assembly Propensity",
    "Cross-Tool Correlations",
    "Full Data",
    "With 80K Data Points",
])

# ============================================================
# TAB: OVERVIEW
# ============================================================
with tab_overview:
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Peptides", len(filtered))
    with col2:
        st.metric("Unique Sequences", filtered["sequence"].nunique())
    with col3:
        st.metric("MW Range", f"{filtered['mw_da'].min():.0f}-{filtered['mw_da'].max():.0f} Da")
    with col4:
        st.metric("pI Range", f"{filtered['pI'].min():.2f}-{filtered['pI'].max():.2f}")
    with col5:
        stable = (filtered["instability_index"] < 40).sum()
        st.metric("Stable (II<40)", f"{stable}/{len(filtered)} ({100*stable/len(filtered):.0f}%)")

    st.markdown("### Analysis Tools Applied to All 618 Peptides")
    st.markdown("""
    | Tool | Source | What it scores | Tab |
    |------|--------|---------------|-----|
    | **BioPython** | Local computation | MW, pI, GRAVY, instability, aromaticity, SS fractions | Biophysical Properties |
    | **Literature exopeptidase scores** | Rozans 2024 design rules | Aminopeptidase + carboxypeptidase susceptibility | Exopeptidase (Literature) |
    | **Protease Atlas** | Eckhard 2016 (4,300 sites), MEROPS, Chen 2012 | APN, LAP, CPA vulnerability per residue | Protease Atlas |
    | **Degradation Predictor** | Rozans 2024 calibration data | Predicted fraction remaining at 48h per cell type + terminal mod | Degradation Predictor |
    | **MMP-14 Eckhard** | Eckhard 2016 TAILS data | P1' frequency for crosslinker X1 position | MMP Crosslinker |
    | **Self-Assembly** | Frederix 2011 AP scores | Average residue aggregation propensity | Self-Assembly |
    """)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### By Paper")
        paper_counts = filtered["paper"].value_counts().reset_index()
        paper_counts.columns = ["Paper", "Count"]
        fig = px.pie(paper_counts, names="Paper", values="Count", hole=0.4)
        fig.update_layout(height=300, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        st.markdown("### By Scaffold")
        sc_counts = filtered["scaffold"].value_counts().reset_index()
        sc_counts.columns = ["Scaffold", "Count"]
        fig = px.pie(sc_counts, names="Scaffold", values="Count", hole=0.4)
        fig.update_layout(height=300, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB: BIOPHYSICAL PROPERTIES
# ============================================================
with tab_biophys:
    st.markdown("### Biophysical Properties (BioPython)")
    st.markdown("Computed locally: MW, pI, GRAVY, instability index, aromaticity, secondary structure fractions.")

    props = [
        ("mw_da", "Molecular Weight (Da)"),
        ("pI", "Isoelectric Point"),
        ("gravy", "GRAVY (Hydrophobicity)"),
        ("instability_index", "Instability Index"),
        ("net_charge_ph7", "Net Charge (pH 7)"),
        ("aromaticity", "Aromaticity"),
    ]

    fig = make_subplots(rows=3, cols=2, subplot_titles=[p[1] for p in props])
    for i, (col, label) in enumerate(props):
        row = i // 2 + 1
        c = i % 2 + 1
        data = filtered[col].dropna()
        if len(data) > 0:
            fig.add_trace(go.Histogram(x=data, nbinsx=30, name=label,
                                        marker_color="#0E4D92", showlegend=False), row=row, col=c)
    fig.update_layout(height=800, margin=dict(t=40))
    st.plotly_chart(fig, use_container_width=True)

    # Variable residue effect
    st.markdown("### Variable Residue Effect (RGEFV Libraries)")
    rgefv = filtered[(filtered["scaffold"].isin(["RGEFV-X", "X-RGEFV"])) &
                     (filtered["variable_residue"].str.len() == 1)]
    if len(rgefv) > 0:
        var_stats = rgefv.groupby("variable_residue").agg(
            mw=("mw_da", "mean"), pI=("pI", "mean"),
            gravy=("gravy", "mean"),
        ).round(3).sort_index()
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(var_stats.reset_index(), x="variable_residue", y="gravy",
                         color="gravy", color_continuous_scale="RdBu_r", color_continuous_midpoint=0)
            fig.update_layout(height=300, margin=dict(t=20), xaxis_title="Variable AA", yaxis_title="Avg GRAVY")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.bar(var_stats.reset_index(), x="variable_residue", y="pI",
                         color="pI", color_continuous_scale="Viridis")
            fig.update_layout(height=300, margin=dict(t=20), xaxis_title="Variable AA", yaxis_title="Avg pI")
            st.plotly_chart(fig, use_container_width=True)

    # Secondary structure
    ss_cols = ["helix_frac", "turn_frac", "sheet_frac"]
    if all(c in filtered.columns for c in ss_cols):
        st.markdown("### Secondary Structure Propensity")
        ss_means = filtered[ss_cols].mean()
        fig = px.bar(x=["Helix", "Turn", "Sheet"], y=ss_means.values, color=["Helix", "Turn", "Sheet"])
        fig.update_layout(height=300, margin=dict(t=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB: EXOPEPTIDASE (LITERATURE)
# ============================================================
with tab_exo:
    st.markdown("### Exopeptidase Susceptibility (Literature-Derived)")
    st.markdown("Scores from our initial analysis based on published exopeptidase substrate preferences.")

    col1, col2 = st.columns(2)
    with col1:
        n_data = filtered.dropna(subset=["n_term_aa", "aminopeptidase_susceptibility"])
        if len(n_data) > 0:
            n_avg = n_data.groupby("n_term_aa")["aminopeptidase_susceptibility"].mean().sort_values(ascending=False)
            fig = px.bar(x=n_avg.index, y=n_avg.values, color=n_avg.values, color_continuous_scale="YlOrRd",
                         labels={"x": "N-terminal AA", "y": "Aminopeptidase Susceptibility"})
            fig.update_layout(height=350, margin=dict(t=20), title="N-terminal (Aminopeptidase)")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        c_data = filtered.dropna(subset=["c_term_aa", "carboxypeptidase_susceptibility"])
        if len(c_data) > 0:
            c_avg = c_data.groupby("c_term_aa")["carboxypeptidase_susceptibility"].mean().sort_values(ascending=False)
            fig = px.bar(x=c_avg.index, y=c_avg.values, color=c_avg.values, color_continuous_scale="YlOrRd",
                         labels={"x": "C-terminal AA", "y": "Carboxypeptidase Susceptibility"})
            fig.update_layout(height=350, margin=dict(t=20), title="C-terminal (Carboxypeptidase)")
            st.plotly_chart(fig, use_container_width=True)

    # Heatmap
    exo_data = filtered.dropna(subset=["n_term_aa", "c_term_aa", "total_exopeptidase_susceptibility"])
    if len(exo_data) > 10:
        st.markdown("### N-term x C-term Susceptibility Heatmap")
        pivot = exo_data.pivot_table(values="total_exopeptidase_susceptibility",
                                      index="n_term_aa", columns="c_term_aa", aggfunc="mean")
        fig = px.imshow(pivot, color_continuous_scale="YlOrRd", aspect="auto",
                        labels={"x": "C-terminal AA", "y": "N-terminal AA", "color": "Score"})
        fig.update_layout(height=450, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB: PROTEASE ATLAS
# ============================================================
with tab_protease:
    st.markdown("### Protease Atlas Scores (Eckhard/MEROPS/Chen)")
    st.markdown(
        "Each peptide scored against 3 exopeptidase profiles (APN, LAP, CPA) derived from "
        "published structural and kinetic data. These are independent from the literature scores above."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**APN (CD13) — N-terminal**")
        apn_data = filtered.dropna(subset=["apn_score"])
        if len(apn_data) > 0:
            apn_by_aa = apn_data.groupby("n_term_aa")["apn_score"].mean().sort_values(ascending=False)
            fig = px.bar(x=apn_by_aa.index, y=apn_by_aa.values, color=apn_by_aa.values,
                         color_continuous_scale="YlOrRd")
            fig.update_layout(height=300, margin=dict(t=20), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**LAP — N-terminal**")
        lap_data = filtered.dropna(subset=["lap_score"])
        if len(lap_data) > 0:
            lap_by_aa = lap_data.groupby("n_term_aa")["lap_score"].mean().sort_values(ascending=False)
            fig = px.bar(x=lap_by_aa.index, y=lap_by_aa.values, color=lap_by_aa.values,
                         color_continuous_scale="YlOrRd")
            fig.update_layout(height=300, margin=dict(t=20), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col3:
        st.markdown("**CPA — C-terminal**")
        cpa_data = filtered.dropna(subset=["cpa_score"])
        if len(cpa_data) > 0:
            cpa_by_aa = cpa_data.groupby("c_term_aa")["cpa_score"].mean().sort_values(ascending=False)
            fig = px.bar(x=cpa_by_aa.index, y=cpa_by_aa.values, color=cpa_by_aa.values,
                         color_continuous_scale="YlOrRd")
            fig.update_layout(height=300, margin=dict(t=20), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # Combined atlas score distribution
    st.markdown("### Combined Exopeptidase Atlas Score")
    atlas_data = filtered["combined_exo_atlas"].dropna()
    if len(atlas_data) > 0:
        fig = px.histogram(atlas_data, nbins=40, labels={"value": "Combined Score"},
                           color_discrete_sequence=["#0E4D92"])
        fig.update_layout(height=300, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    # Compare atlas vs literature scores
    st.markdown("### Atlas vs Literature Score Comparison")
    compare = filtered.dropna(subset=["combined_exo_atlas", "total_exopeptidase_susceptibility"])
    if len(compare) > 10:
        fig = px.scatter(compare, x="total_exopeptidase_susceptibility", y="combined_exo_atlas",
                         color="scaffold", hover_data=["sequence"],
                         labels={"total_exopeptidase_susceptibility": "Literature Exo Score",
                                 "combined_exo_atlas": "Protease Atlas Score"})
        fig.update_layout(height=400, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)
        r = compare[["total_exopeptidase_susceptibility", "combined_exo_atlas"]].corr().iloc[0, 1]
        st.markdown(f"**Correlation:** r = {r:.3f} — {'strong agreement' if r > 0.7 else 'moderate agreement' if r > 0.4 else 'weak agreement'}")

# ============================================================
# TAB: DEGRADATION PREDICTOR
# ============================================================
with tab_degrade:
    st.markdown("### Degradation Predictions (Calibrated: Rozans 2024)")
    st.markdown(
        "Predicted fraction remaining at 48h for each peptide, across 4 cell types and 4 N-terminal modifications. "
        "Calibrated against published experimental data (NH2=0.20, Ac=0.82, etc.)."
    )

    cell_type = st.selectbox("Cell type", list(CELL_AGGRESSIVENESS.keys()), key="deg_cell")

    # Show predictions for selected cell type across N-term mods
    deg_cols = [c for c in filtered.columns if c.startswith(f"deg_{cell_type}_") and c.endswith("_COOH")]
    if deg_cols:
        st.markdown(f"### {cell_type} — Fraction Remaining by N-terminal Modification")

        # Box plots by N-terminal mod
        melt_data = []
        for col in deg_cols:
            n_mod = col.replace(f"deg_{cell_type}_", "").replace("_COOH", "")
            for val in filtered[col].dropna():
                melt_data.append({"N-terminal": n_mod, "Fraction Remaining": val})

        if melt_data:
            melt_df = pd.DataFrame(melt_data)
            fig = px.box(melt_df, x="N-terminal", y="Fraction Remaining",
                         color="N-terminal", points="outliers")
            fig.update_layout(height=400, margin=dict(t=20), yaxis=dict(range=[0, 1]))
            st.plotly_chart(fig, use_container_width=True)

        # Heatmap: scaffold x N-mod
        st.markdown("### Mean Prediction by Scaffold x N-terminal Mod")
        heat_data = []
        for col in deg_cols:
            n_mod = col.replace(f"deg_{cell_type}_", "").replace("_COOH", "")
            for scaffold, group in filtered.groupby("scaffold"):
                mean_val = group[col].mean()
                if not np.isnan(mean_val):
                    heat_data.append({"Scaffold": scaffold, "N-terminal": n_mod, "Fraction Remaining": round(mean_val, 3)})

        if heat_data:
            heat_df = pd.DataFrame(heat_data)
            pivot = heat_df.pivot(index="Scaffold", columns="N-terminal", values="Fraction Remaining")
            fig = px.imshow(pivot, color_continuous_scale="RdYlGn", zmin=0, zmax=1, aspect="auto",
                            labels={"color": "Fraction Remaining"})
            fig.update_layout(height=350, margin=dict(t=20))
            st.plotly_chart(fig, use_container_width=True)

        # Most vulnerable peptides
        best_col = f"deg_{cell_type}_NH2_COOH"
        if best_col in filtered.columns:
            st.markdown(f"### Most Vulnerable Peptides ({cell_type}, NH2/COOH — worst case)")
            vuln = filtered.dropna(subset=[best_col]).sort_values(best_col).head(20)
            st.dataframe(vuln[["sequence", "full_notation", "n_term_aa", "c_term_aa", best_col]].rename(
                columns={best_col: "Fraction Remaining"}
            ), hide_index=True)

# ============================================================
# TAB: MMP CROSSLINKER
# ============================================================
with tab_mmp:
    st.markdown("### MMP Crosslinker Analysis (KLVAD-X1X2-ASAE)")

    xlink = filtered[filtered["scaffold"] == "KLVAD-XX-ASAE"].copy()

    if len(xlink) == 0:
        st.info("Filter to Paper 3 / KLVAD-XX-ASAE scaffold to see crosslinker analysis.")
    else:
        xlink_valid = xlink.dropna(subset=["mmp_cleavage_score"])

        # Side-by-side: our generic score vs Eckhard P1' frequency
        st.markdown("### Our Score vs Eckhard MMP-14 P1' Frequency")
        st.markdown("Comparing two independent analyses of the same crosslinkers.")

        xlink_cmp = xlink_valid.dropna(subset=["mmp14_p1p_eckhard"]).copy()
        if len(xlink_cmp) > 10:
            col1, col2 = st.columns(2)
            with col1:
                xlink_cmp["x1"] = xlink_cmp["variable_residue"].str[0]
                aa_order = list("LIMAVFSTYGNQKRWDEHP")

                # Our MMP score heatmap
                xlink_cmp["x2"] = xlink_cmp["variable_residue"].str[1]
                pivot1 = xlink_cmp.pivot_table(values="mmp_cleavage_score", index="x1", columns="x2", aggfunc="mean")
                pivot1 = pivot1.reindex(index=aa_order, columns=aa_order)
                fig = px.imshow(pivot1, color_continuous_scale="RdYlGn", aspect="equal", zmin=0, zmax=1,
                                labels={"color": "Our Score"})
                fig.update_layout(height=450, margin=dict(t=30), title="Our Generic MMP Score")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Eckhard P1' frequency (X1 only — X2 is P2')
                st.markdown("**Eckhard MMP-14 P1' frequency by X1 residue**")
                eckhard_by_x1 = xlink_cmp.groupby("x1")["mmp14_p1p_eckhard"].mean().reindex(aa_order)
                fig = px.bar(x=eckhard_by_x1.index, y=eckhard_by_x1.values,
                             color=eckhard_by_x1.values, color_continuous_scale="YlOrRd",
                             labels={"x": "X1 (P1') Residue", "y": "Eckhard MMP-14 Frequency"})
                fig.update_layout(height=350, margin=dict(t=20))
                st.plotly_chart(fig, use_container_width=True)

                # Correlation
                r = xlink_cmp[["mmp_cleavage_score", "mmp14_p1p_eckhard"]].corr().iloc[0, 1]
                st.metric("Score vs Eckhard Correlation", f"r = {r:.3f}")

        # Key variants
        st.markdown("### Key Crosslinker Variants")
        key_vars = ["LM", "LI", "LL", "NY", "PP", "GG"]
        key_data = xlink_valid[xlink_valid["variable_residue"].isin(key_vars)]
        if len(key_data) > 0:
            display_cols = ["variable_residue", "sequence", "mmp_cleavage_score", "mmp14_p1p_eckhard",
                            "mw_da", "pI", "gravy"]
            display_cols = [c for c in display_cols if c in key_data.columns]
            st.dataframe(key_data[display_cols].sort_values("mmp_cleavage_score", ascending=False).rename(columns={
                "variable_residue": "Dipeptide", "mmp_cleavage_score": "Our Score",
                "mmp14_p1p_eckhard": "Eckhard P1'",
            }), hide_index=True)

# ============================================================
# TAB: SELF-ASSEMBLY
# ============================================================
with tab_selfasm:
    st.markdown("### Self-Assembly Propensity (Frederix AP Method)")
    st.markdown(
        "Average residue aggregation propensity score for each peptide. "
        "Higher = more aromatic residues = greater tendency to self-assemble. "
        "Based on Frederix et al. (J Phys Chem Lett 2011) dipeptide screening."
    )

    sa_data = filtered["self_assembly_propensity"].dropna()
    if len(sa_data) > 0:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(sa_data, nbins=40, labels={"value": "Avg Residue AP"},
                               color_discrete_sequence=["#0E4D92"])
            fig.update_layout(height=300, margin=dict(t=20))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # By scaffold
            sa_by_scaffold = filtered.groupby("scaffold")["self_assembly_propensity"].mean().sort_values(ascending=False)
            fig = px.bar(x=sa_by_scaffold.index, y=sa_by_scaffold.values,
                         color=sa_by_scaffold.values, color_continuous_scale="Viridis",
                         labels={"x": "Scaffold", "y": "Avg AP Score"})
            fig.update_layout(height=300, margin=dict(t=20))
            st.plotly_chart(fig, use_container_width=True)

    # Top self-assemblers
    st.markdown("### Peptides Most Likely to Self-Assemble")
    sa_sorted = filtered.dropna(subset=["self_assembly_propensity"]).sort_values(
        "self_assembly_propensity", ascending=False
    )
    st.dataframe(sa_sorted.head(20)[["sequence", "full_notation", "scaffold", "self_assembly_propensity",
                                      "aromaticity", "gravy"]].rename(columns={
        "self_assembly_propensity": "AP Score"
    }), hide_index=True)

    st.markdown("""
    **Note:** These are short peptides (5-13 AA), so self-assembly propensity is low compared to
    Fmoc-dipeptides. The AP score here identifies which Rozans peptides have the HIGHEST aromatic
    content and thus greatest potential for pi-pi stacking interactions.
    """)

# ============================================================
# TAB: CROSS-TOOL CORRELATIONS
# ============================================================
with tab_corr:
    st.markdown("### Cross-Tool Correlation Matrix")
    st.markdown("How do the different analytical tools agree on peptide properties?")

    corr_cols = [
        "mw_da", "pI", "gravy", "instability_index", "aromaticity", "net_charge_ph7",
        "total_exopeptidase_susceptibility", "combined_exo_atlas",
        "self_assembly_propensity",
    ]
    # Add a degradation col if available
    deg_col = f"deg_hMSC_NH2_COOH"
    if deg_col in filtered.columns:
        corr_cols.append(deg_col)

    existing = [c for c in corr_cols if c in filtered.columns]
    corr = filtered[existing].corr().round(3)

    # Rename for readability
    rename_map = {
        "total_exopeptidase_susceptibility": "Exo (Literature)",
        "combined_exo_atlas": "Exo (Atlas)",
        "self_assembly_propensity": "Self-Assembly AP",
        "deg_hMSC_NH2_COOH": "Degrad (hMSC/NH2)",
    }
    corr_display = corr.rename(index=rename_map, columns=rename_map)

    fig = px.imshow(corr_display, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="equal",
                    labels={"color": "r"})
    fig.update_layout(height=600, margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

    # Notable cross-tool correlations
    st.markdown("### Key Cross-Tool Findings")
    notable = []
    for i, c1 in enumerate(existing):
        for c2 in existing[i+1:]:
            r = corr.loc[c1, c2]
            if abs(r) > 0.4:
                n1 = rename_map.get(c1, c1)
                n2 = rename_map.get(c2, c2)
                notable.append({"Tool 1": n1, "Tool 2": n2, "r": r,
                                "Agreement": "Strong" if abs(r) > 0.7 else "Moderate"})
    if notable:
        st.dataframe(pd.DataFrame(notable).sort_values("r", key=abs, ascending=False), hide_index=True)

    # Interactive scatter
    st.markdown("### Interactive Cross-Tool Scatter")
    col1, col2, col3 = st.columns(3)
    with col1:
        x_prop = st.selectbox("X axis", existing, index=0)
    with col2:
        y_idx = min(1, len(existing) - 1)
        y_prop = st.selectbox("Y axis", existing, index=y_idx)
    with col3:
        color_by = st.selectbox("Color", ["paper", "scaffold", "library"] + existing, index=0)

    fig = px.scatter(filtered, x=x_prop, y=y_prop, color=color_by,
                     hover_data=["sequence", "full_notation"], opacity=0.7)
    fig.update_layout(height=500, margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB: FULL DATA
# ============================================================
with tab_table:
    st.markdown("### Full Enriched Dataset")
    st.markdown(f"{len(filtered)} peptides x {len(filtered.columns)} columns")

    all_cols = list(filtered.columns)
    default_cols = ["sequence", "full_notation", "scaffold", "variable_residue", "paper",
                    "mw_da", "pI", "gravy", "total_exopeptidase_susceptibility",
                    "combined_exo_atlas", "self_assembly_propensity"]
    if deg_col in all_cols:
        default_cols.append(deg_col)
    default_cols = [c for c in default_cols if c in all_cols]
    selected_cols = st.multiselect("Columns to display", all_cols, default=default_cols)

    if selected_cols:
        st.dataframe(filtered[selected_cols], hide_index=True, height=600)

    csv = filtered.to_csv(index=False)
    st.download_button("Download full enriched CSV", csv, "rozans-618-all-tools.csv", "text/csv")

# ============================================================
# TAB: WITH 80K DATA POINTS
# ============================================================
with tab_next:
    st.markdown("### What This Preview Shows")
    st.markdown("""
    The 618-peptide analysis above is a **proof of concept** built entirely from published
    sequence data. Every score is a *prediction* — we have no measured degradation rates to
    validate against.

    **Sam's 80,000+ LC-MS data points change everything.** They contain the actual measured
    concentrations at 0, 1, 4, 8, 24, and 48 hours for each peptide across multiple cell types,
    terminal modifications, and concentrations. That data turns predictions into validated models.
    """)

    st.markdown("---")

    st.markdown("### What Becomes Possible with the Full Dataset")

    st.markdown("#### 1. Train the First Exopeptidase ML Model")
    st.markdown("""
    **The gap:** CleaveNet, UniZyme, PROSPERous — every existing ML protease predictor focuses on
    endopeptidases (MMPs, serine proteases). **Nobody has built a dedicated exopeptidase degradation model.**
    Sam's data is the largest published exopeptidase substrate dataset in existence.

    **With 80K data points we could:**
    - Train a sequence-to-degradation-rate model (random forest, XGBoost, or fine-tuned ESM-2)
    - Validate against held-out peptides (cross-validation across libraries)
    - Publish as the first exopeptidase-specific prediction tool
    - Deploy as a page in this toolkit with live predictions

    **Current state (618 peptides):** We use literature-derived scores. Correlation with Sam's
    actual data is unknown.

    **With 80K points:** Replace literature scores with data-driven model. Expected accuracy
    improvement: substantial (literature scores are generic; Sam's data is peptide-specific).
    """)

    st.markdown("#### 2. Validate Every Prediction in This Toolkit")
    st.markdown("""
    | Tool | Current (618 sequences) | With 80K Data Points |
    |------|------------------------|---------------------|
    | **Exopeptidase scores** | Literature-derived, unvalidated | Validate against measured degradation; recalibrate |
    | **Protease Atlas (APN/LAP/CPA)** | Published specificity profiles | Compare profile predictions vs measured rates; identify which protease dominates |
    | **Degradation Predictor** | Calibrated on paper-level averages | Calibrate on individual peptide measurements; per-residue correction factors |
    | **MMP crosslinker scores** | Generic P1/P1' preferences | Validate against split-and-pool screening results; rank accuracy |
    | **Terminal modification hierarchy** | Published: NH2 < N-βA < Ac ≈ Ac-βA | Quantify exact protection factors per amino acid x modification combo |
    | **Histidine exception** | Flagged qualitatively | Quantify: how much worse is His vs other residues? Dose-response? |
    """)

    st.markdown("#### 3. Discover New Structure-Activity Relationships")
    st.markdown("""
    **Questions only the 80K dataset can answer:**

    - **Which amino acid pairs are most vulnerable?** The 618 sequences test single variable
      residues. With degradation rates, we can find synergistic effects between N-terminal
      and C-terminal residues that aren't predictable from individual scores.

    - **Is there a degradation cliff?** Does protection increase linearly with modification,
      or is there a threshold (e.g., Ac alone gives 80% of Ac-βA's protection)?

    - **Cell-type-specific protease fingerprints.** hMSCs degrade 4x faster than macrophages.
      With per-peptide resolution, we can identify which proteases each cell type uses
      (aminopeptidase N vs leucine aminopeptidase vs carboxypeptidase A vs B).

    - **Concentration-dependent kinetics.** Sam tested 19.5 to 5,000 uM. Michaelis-Menten
      fitting would give Km and Vmax for exopeptidases against each peptide sequence —
      the first such dataset for synthetic peptides.

    - **Time-course modeling.** The 6-timepoint kinetics (0-48h) enable degradation rate
      fitting (first-order, Michaelis-Menten, or biphasic models) for every peptide.
    """)

    st.markdown("#### 4. Build a Degradation Design Tool")
    st.markdown("""
    The end goal: a tool where you input a peptide sequence + terminal modifications + cell type
    and get back:

    - **Predicted half-life** (hours) with confidence interval
    - **Recommended protection strategy** (which N/C-terminal mod to use)
    - **Cell-type-specific risk assessment**
    - **Comparison to benchmarks** (GRGDS, cyclic RGD, GPQGIWGQ)

    This doesn't exist anywhere. It would be the first tool of its kind.
    """)

    st.markdown("#### 5. Publishable Outputs")
    st.markdown("""
    | Output | Target Journal | Impact |
    |--------|---------------|--------|
    | Exopeptidase ML model + web tool | Bioinformatics or NAR Web Server | First exopeptidase-specific predictor |
    | Comprehensive SAR analysis (80K points) | ACS Biomater Sci Eng or Biomaterials | Largest peptide degradation SAR study |
    | Open-source Bio-AI toolkit | JOSS or SoftwareX | Community tool for biomaterials design |
    | Validated degradation design rules | Nature Protocols | Practical guide for hydrogel engineers |
    """)

    st.markdown("---")

    st.markdown("### Data Integration Checklist")
    st.markdown("""
    To unlock the above, we need Sam's data in a structured format:

    - [ ] **Raw LC-MS peak areas** for each peptide at each timepoint (CSV or Excel)
    - [ ] **Internal standard normalization** (AUC ratios, not raw areas)
    - [ ] **Metadata:** peptide identity, terminal modification, cell type, donor ID, concentration, replicate number
    - [ ] **Timepoints:** 0, 1, 4, 8, 24, 48 hours
    - [ ] **Cell types:** hMSC, hUVEC, macrophage, THP-1 (which are included?)
    - [ ] **Quality flags:** any wells excluded, column failure runs, etc.

    **Format:** One row per measurement. Columns: peptide_id, sequence, n_terminal_mod,
    c_terminal_mod, cell_type, donor_id, concentration_uM, timepoint_h, replicate,
    normalized_auc, fraction_remaining.

    **Size estimate:** 618 peptides x ~6 timepoints x ~3 replicates x ~3 cell types
    = ~33K rows minimum. Sam's 80K+ suggests additional conditions (concentrations,
    PEGylation variants, etc.).
    """)

    st.markdown("---")
    st.markdown(
        "*This page is a preview for Sam Rozans showing the analytical pipeline. "
        "The 618-peptide analysis demonstrates the tools; the 80K dataset would validate "
        "and train them into something publishable.*"
    )
