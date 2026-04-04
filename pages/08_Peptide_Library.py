import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from Bio.SeqUtils import molecular_weight
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from pathlib import Path

st.set_page_config(page_title="Peptide Library", page_icon="📊", layout="wide")
st.title("Rozans Peptide Library")
st.markdown(
    "618 peptide sequences reconstructed from Rozans et al. (Pashuck Lab, Lehigh University). "
    "Sources: ACS Biomater Sci Eng 2024, J Biomed Mater Res A 2025, Adv Healthcare Mater 2025."
)

# Load data
DATA_PATH = Path(__file__).parent.parent / "data" / "rozans-peptide-library.csv"


@st.cache_data
def load_library():
    df = pd.read_csv(DATA_PATH)
    # Compute properties for each unique sequence
    mws, pis, gravys, lengths = [], [], [], []
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    for seq in df["sequence"]:
        clean = "".join(c for c in seq.upper() if c in valid_aa)
        if len(clean) >= 2:
            try:
                mws.append(molecular_weight(clean, "protein"))
                analysis = ProteinAnalysis(clean)
                pis.append(analysis.isoelectric_point())
                gravys.append(analysis.gravy())
                lengths.append(len(clean))
            except Exception:
                mws.append(None)
                pis.append(None)
                gravys.append(None)
                lengths.append(len(clean))
        else:
            mws.append(None)
            pis.append(None)
            gravys.append(None)
            lengths.append(len(clean))

    df["mw_da"] = mws
    df["pI"] = pis
    df["gravy"] = gravys
    df["seq_length"] = lengths
    return df


df = load_library()

# Filters
st.sidebar.markdown("### Filters")
papers = ["All"] + sorted(df["paper"].unique().tolist())
selected_paper = st.sidebar.selectbox("Paper", papers)
libraries = ["All"] + sorted(df["library"].unique().tolist())
selected_lib = st.sidebar.selectbox("Library", libraries)

filtered = df.copy()
if selected_paper != "All":
    filtered = filtered[filtered["paper"] == selected_paper]
if selected_lib != "All":
    filtered = filtered[filtered["library"] == selected_lib]

st.markdown(f"**Showing {len(filtered)} / {len(df)} peptides**")

# Summary stats
tab1, tab2, tab3 = st.tabs(["Overview", "Property Distributions", "Full Table"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Peptides", len(filtered))
    with col2:
        st.metric("Papers", filtered["paper"].nunique())
    with col3:
        st.metric("Libraries", filtered["library"].nunique())
    with col4:
        avg_len = filtered["seq_length"].mean()
        st.metric("Avg Length", f"{avg_len:.1f} aa")

    # By paper
    st.markdown("### Peptides by Paper")
    paper_counts = filtered["paper"].value_counts().reset_index()
    paper_counts.columns = ["Paper", "Count"]
    fig = px.bar(paper_counts, x="Paper", y="Count", color="Paper")
    fig.update_layout(height=300, margin=dict(t=20), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # By library
    st.markdown("### Peptides by Library")
    lib_counts = filtered["library"].value_counts().reset_index()
    lib_counts.columns = ["Library", "Count"]
    fig2 = px.bar(lib_counts, x="Count", y="Library", orientation="h", color="Library")
    fig2.update_layout(height=500, margin=dict(t=20), showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

    # Variable residue distribution (for library peptides)
    var_res = filtered[filtered["variable_residue"].str.len() == 1]["variable_residue"]
    if len(var_res) > 0:
        st.markdown("### Variable Residue Distribution (single AA libraries)")
        res_counts = var_res.value_counts().sort_index().reset_index()
        res_counts.columns = ["Amino Acid", "Count"]
        fig3 = px.bar(res_counts, x="Amino Acid", y="Count")
        fig3.update_layout(height=300, margin=dict(t=20))
        st.plotly_chart(fig3, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Molecular Weight Distribution")
        mw_data = filtered["mw_da"].dropna()
        if len(mw_data) > 0:
            fig = px.histogram(mw_data, nbins=40, labels={"value": "MW (Da)"})
            fig.update_layout(height=350, margin=dict(t=20), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Isoelectric Point Distribution")
        pi_data = filtered["pI"].dropna()
        if len(pi_data) > 0:
            fig = px.histogram(pi_data, nbins=30, labels={"value": "pI"})
            fig.update_layout(height=350, margin=dict(t=20), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### GRAVY (Hydrophobicity) Distribution")
        gravy_data = filtered["gravy"].dropna()
        if len(gravy_data) > 0:
            fig = px.histogram(gravy_data, nbins=30, labels={"value": "GRAVY Score"})
            fig.update_layout(height=350, margin=dict(t=20), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown("### MW vs pI")
        scatter_data = filtered.dropna(subset=["mw_da", "pI"])
        if len(scatter_data) > 0:
            fig = px.scatter(
                scatter_data, x="mw_da", y="pI",
                color="paper",
                hover_data=["sequence", "library"],
                labels={"mw_da": "MW (Da)", "pI": "Isoelectric Point"},
            )
            fig.update_layout(height=350, margin=dict(t=20))
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("### Full Peptide Library")
    display_cols = ["sequence", "full_notation", "scaffold", "variable_residue",
                    "library", "paper", "type", "mw_da", "pI", "gravy", "seq_length"]
    st.dataframe(
        filtered[display_cols].rename(columns={
            "mw_da": "MW (Da)", "pI": "pI", "gravy": "GRAVY", "seq_length": "Length"
        }),
        hide_index=True,
        height=600,
    )

    csv = filtered.to_csv(index=False)
    st.download_button("Download filtered CSV", csv, "peptide_library_filtered.csv", "text/csv")
