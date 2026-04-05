import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json
import re

st.set_page_config(page_title="ExoPred Training Data", page_icon="🧠", layout="wide")
st.title("ExoPred Training Data Explorer")
st.markdown(
    "Integrated public datasets for training exopeptidase degradation prediction models. "
    "Each tab lets you explore the raw data that feeds into ExoPred's feature matrix "
    "-- MEROPS cleavage sites, PEPlife2 half-lives, DPP-IV benchmarks, and Sam's proprietary LC-MS data."
)

DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# ============================================================
# HELPER: safe data loading
# ============================================================

def load_merops_cleavages():
    """Load and parse MEROPS cleavage.txt (tab-delimited, quoted fields)."""
    processed = PROCESSED_DIR / "merops_exopeptidase_cleavages.csv"
    if processed.exists():
        return pd.read_csv(processed)

    raw = DATA_DIR / "merops" / "cleavage.txt"
    if not raw.exists():
        return None

    # Raw MEROPS cleavage.txt: tab-separated, 13 columns, quoted strings
    # Cols: protease_id, substrate_uniprot, cleavage_pos, evidence, context,
    #       col5, col6, col7, substrate_range, col9, reference, method, col12
    cols = [
        "protease_id", "substrate_uniprot", "cleavage_pos", "evidence",
        "context", "col5", "col6", "col7", "substrate_range", "col9",
        "reference", "method", "col12"
    ]
    df = pd.read_csv(
        raw, sep="\t", header=None, names=cols,
        quotechar='"', na_values=["\\N"], on_bad_lines="warn",
        encoding="latin-1", low_memory=False,
    )
    # Clean protease_id
    df["protease_id"] = df["protease_id"].astype(str).str.strip('"').str.strip()
    # Extract protease family (letter + digits before dot)
    df["protease_family"] = df["protease_id"].str.extract(r"^([A-Z]\d+)", expand=False)
    # Filter to exopeptidases: families starting with M14 (carboxypeptidases),
    # M01 (aminopeptidases), S09 (DPP-IV family), S28, C01, M17, M18, M24, M28
    exo_families = ["M14", "M01", "M17", "M18", "M24", "M28", "S09", "S28", "C01"]
    df_exo = df[df["protease_family"].isin(exo_families)].copy()
    if len(df_exo) == 0:
        # If strict exo filter is empty, return all with family parsed
        return df
    return df_exo


def load_merops_substrate_search():
    """Load MEROPS Substrate_search.txt for richer cleavage site context."""
    raw = DATA_DIR / "merops" / "Substrate_search.txt"
    if not raw.exists():
        return None
    cols = [
        "cleavage_id", "protease_id", "substrate_name", "cleavage_desc",
        "P3", "P2", "P1", "P1p", "P2p", "P3p", "P4p",  # err — let's be safe
    ]
    # This file has variable columns; read raw
    df = pd.read_csv(
        raw, sep="\t", header=None, quotechar="'",
        na_values=["NULL", "-", ""], on_bad_lines="warn",
        encoding="latin-1", low_memory=False,
    )
    # Assign known columns
    col_names = [
        "cleavage_id", "protease_id", "substrate_name", "cleavage_desc",
        "P3", "P2", "P1", "P1p", "P2p", "P3p", "P4p", "reference",
        "substrate_uniprot", "cleavage_pos", "organism", "protease_name",
        "col16", "col17", "substrate_range", "col19", "col20", "context",
    ]
    for i, name in enumerate(col_names):
        if i < len(df.columns):
            df.rename(columns={i: name}, inplace=True)

    df["protease_family"] = df["protease_id"].astype(str).str.extract(r"^([A-Z]\d+)", expand=False)
    return df


def load_peplife2():
    """Load PEPlife2 data from JSON API dumps."""
    processed = PROCESSED_DIR / "peplife2_combined.csv"
    if processed.exists():
        return pd.read_csv(processed)

    peplife_dir = DATA_DIR / "peplife"
    records = []
    for fname in ["peplife2_api_natural.json", "peplife2_api_linear.json",
                   "peplife2_api_cyclic.json", "peplife2_api_modified.json"]:
        fpath = peplife_dir / fname
        if fpath.exists():
            with open(fpath) as f:
                d = json.load(f)
                if "data" in d:
                    for rec in d["data"]:
                        rec["source_file"] = fname.replace("peplife2_api_", "").replace(".json", "")
                        records.append(rec)
    if not records:
        return None
    df = pd.DataFrame(records)
    return df


def load_dppiv_benchmark():
    """Load iDPPIV benchmark FASTA-like files."""
    processed = PROCESSED_DIR / "dppiv_benchmark.csv"
    if processed.exists():
        return pd.read_csv(processed)

    base = DATA_DIR / "dppiv" / "idppiv-benchmark" / "iDPPIV" / "data"
    if not base.exists():
        return None

    records = []
    for split in ["train", "test"]:
        for label in ["positive", "negative"]:
            fpath = base / f"{split}_{label}.txt"
            if not fpath.exists():
                continue
            with open(fpath) as f:
                lines = f.read().strip().split("\n")
            for i in range(0, len(lines) - 1, 2):
                header = lines[i]
                seq = lines[i + 1].strip()
                records.append({
                    "sequence": seq,
                    "label": 1 if label == "positive" else 0,
                    "split": split,
                    "length": len(seq),
                })
    if not records:
        return None
    return pd.DataFrame(records)


def load_dppiv_chembl():
    """Load ChEMBL DPP-IV IC50 data."""
    processed = PROCESSED_DIR / "dppiv_chembl_ic50.csv"
    if processed.exists():
        return pd.read_csv(processed)

    fpath = DATA_DIR / "dppiv" / "chembl" / "chembl284_dpp4_activities.csv"
    if not fpath.exists():
        return None
    df = pd.read_csv(fpath)
    return df


def parse_half_life_minutes(val):
    """Convert half-life string to numeric minutes."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    # Handle ranges like "<30", ">120", "~60"
    s = s.replace("<", "").replace(">", "").replace("~", "").replace("≈", "")
    try:
        return float(s)
    except ValueError:
        return np.nan


# ============================================================
# TABS
# ============================================================

tab_overview, tab_merops, tab_peplife, tab_dppiv, tab_blueprint = st.tabs([
    "Overview",
    "MEROPS Exopeptidases",
    "Peptide Half-Lives (PEPlife2)",
    "DPP-IV Intelligence",
    "ExoPred Model Blueprint",
])


# ============================================================
# TAB 1: OVERVIEW
# ============================================================
with tab_overview:
    st.header("Dataset Summary")

    # Load all datasets to get counts
    merops_df = load_merops_cleavages()
    merops_sub = load_merops_substrate_search()
    peplife_df = load_peplife2()
    dppiv_bench = load_dppiv_benchmark()
    dppiv_chembl = load_dppiv_chembl()

    merops_cleavage_count = len(merops_df) if merops_df is not None else 0
    merops_sub_count = len(merops_sub) if merops_sub is not None else 0
    peplife_count = len(peplife_df) if peplife_df is not None else 0
    dppiv_bench_count = len(dppiv_bench) if dppiv_bench is not None else 0
    dppiv_chembl_count = len(dppiv_chembl) if dppiv_chembl is not None else 0

    datasets = pd.DataFrame([
        {
            "Dataset": "MEROPS Cleavage DB",
            "Source": "merops.sanger.ac.uk",
            "Records": f"{merops_cleavage_count:,}",
            "Exopeptidase Relevance": "Direct -- cleavage sites by protease family",
            "Status": "Loaded" if merops_cleavage_count > 0 else "Missing",
        },
        {
            "Dataset": "MEROPS Substrate Search",
            "Source": "merops.sanger.ac.uk",
            "Records": f"{merops_sub_count:,}",
            "Exopeptidase Relevance": "P1-P1' residue context at each cleavage",
            "Status": "Loaded" if merops_sub_count > 0 else "Missing",
        },
        {
            "Dataset": "PEPlife2 Half-Lives",
            "Source": "webs.iiitd.edu.in/peplife2",
            "Records": f"{peplife_count:,}",
            "Exopeptidase Relevance": "Experimental half-lives under protease exposure",
            "Status": "Loaded" if peplife_count > 0 else "Missing",
        },
        {
            "Dataset": "iDPPIV Benchmark",
            "Source": "Charoenkwan et al. 2022",
            "Records": f"{dppiv_bench_count:,}",
            "Exopeptidase Relevance": "DPP-IV substrate classification (pos/neg)",
            "Status": "Loaded" if dppiv_bench_count > 0 else "Missing",
        },
        {
            "Dataset": "ChEMBL DPP-IV IC50",
            "Source": "ChEMBL (Target 284)",
            "Records": f"{dppiv_chembl_count:,}",
            "Exopeptidase Relevance": "Small-molecule inhibitor potency data",
            "Status": "Loaded" if dppiv_chembl_count > 0 else "Missing",
        },
        {
            "Dataset": "Rozans LC-MS (proprietary)",
            "Source": "Pashuck Lab, Lehigh",
            "Records": "~80,000",
            "Exopeptidase Relevance": "Time-resolved degradation by cell type",
            "Status": "Proprietary",
        },
    ])

    st.dataframe(datasets, hide_index=True, use_container_width=True)

    # Key metrics
    st.subheader("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Cleavage Sites", f"{merops_cleavage_count + merops_sub_count:,}")
    col2.metric("Half-Life Measurements", f"{peplife_count:,}")
    col3.metric("DPP-IV Peptides", f"{dppiv_bench_count:,}")
    col4.metric("ChEMBL IC50 Values", f"{dppiv_chembl_count:,}")

    # Bar chart of dataset sizes
    st.subheader("Dataset Size Comparison")
    size_df = pd.DataFrame({
        "Dataset": [
            "MEROPS Cleavage", "MEROPS Substrate", "PEPlife2",
            "iDPPIV Benchmark", "ChEMBL DPP-IV", "Rozans LC-MS"
        ],
        "Records": [
            merops_cleavage_count, merops_sub_count, peplife_count,
            dppiv_bench_count, dppiv_chembl_count, 80000
        ],
        "Type": [
            "Public", "Public", "Public",
            "Public", "Public", "Proprietary"
        ],
    })
    fig_sizes = px.bar(
        size_df, x="Dataset", y="Records", color="Type",
        title="Records per Dataset",
        text="Records",
    )
    fig_sizes.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig_sizes.update_layout(yaxis_type="log", yaxis_title="Records (log scale)")
    st.plotly_chart(fig_sizes, use_container_width=True)

    # Data advantage section
    st.subheader("Sam's Data Advantage")
    total_public = merops_cleavage_count + merops_sub_count + peplife_count + dppiv_bench_count
    st.markdown(f"""
**Public exopeptidase-relevant data: ~{total_public:,} records**
across MEROPS, PEPlife2, and iDPPIV. Most of this is categorical (cleaved/not) or
single-timepoint (one half-life value per peptide).

**Sam's 80,000 LC-MS data points** are fundamentally different:
- **Time-resolved**: degradation curves at 0, 1, 4, 8, 24, 48 hours
- **Multi-cell-type**: hMSC, hUVEC, macrophage, THP-1
- **Systematic variation**: 19 amino acids x 4 N-terminal x 3 C-terminal chemistries
- **Quantitative**: exact fraction remaining, not just binary classification

This is 70x more data than the best published benchmark (ENZ-XGBoost used 1,119 peptides).
When combined with the public datasets above, ExoPred will have the largest training
corpus ever assembled for exopeptidase degradation prediction.
""")

    st.subheader("Data Completeness Matrix")
    completeness = pd.DataFrame([
        {"Feature": "N-terminal cleavage rates", "MEROPS": "Yes", "PEPlife2": "Partial", "iDPPIV": "No", "Rozans LC-MS": "Yes (systematic)"},
        {"Feature": "C-terminal cleavage rates", "MEROPS": "Yes", "PEPlife2": "Partial", "iDPPIV": "No", "Rozans LC-MS": "Yes (systematic)"},
        {"Feature": "DPP-IV (internal exo)", "MEROPS": "Yes", "PEPlife2": "Some", "iDPPIV": "Yes", "Rozans LC-MS": "No"},
        {"Feature": "Time-resolved kinetics", "MEROPS": "No", "PEPlife2": "Single t1/2", "iDPPIV": "No", "Rozans LC-MS": "Yes (6 timepoints)"},
        {"Feature": "Cell-type specificity", "MEROPS": "No", "PEPlife2": "No", "iDPPIV": "No", "Rozans LC-MS": "Yes (4 cell types)"},
        {"Feature": "Chemical modification effects", "MEROPS": "No", "PEPlife2": "Yes", "iDPPIV": "No", "Rozans LC-MS": "Yes (12 combos)"},
        {"Feature": "Sequence-level features", "MEROPS": "P1-P1' only", "PEPlife2": "Full seq", "iDPPIV": "Full seq", "Rozans LC-MS": "Full seq"},
    ])
    st.dataframe(completeness, hide_index=True, use_container_width=True)
    st.markdown(
        "*Sam's data fills the two biggest gaps: time-resolved kinetics and cell-type specificity. "
        "No public dataset covers these dimensions systematically.*"
    )


# ============================================================
# TAB 2: MEROPS EXOPEPTIDASES
# ============================================================
with tab_merops:
    st.header("MEROPS Exopeptidase Cleavage Data")

    merops_sub = load_merops_substrate_search()
    if merops_sub is None or len(merops_sub) == 0:
        st.warning(
            "MEROPS Substrate_search.txt not found or empty. "
            "Place the file in data/merops/ or run process_datasets.py."
        )
    else:
        # Filter to exopeptidase families
        exo_families = ["M14", "M01", "M17", "M18", "M24", "M28", "S09", "S28", "C01"]
        exo_mask = merops_sub["protease_family"].isin(exo_families)
        has_exo = exo_mask.sum() > 0

        # Sidebar-style filters
        col_f1, col_f2, col_f3 = st.columns(3)
        available_families = sorted(merops_sub["protease_family"].dropna().unique())
        with col_f1:
            default_fams = [f for f in exo_families if f in available_families]
            sel_families = st.multiselect(
                "Protease Family",
                options=available_families,
                default=default_fams[:6] if default_fams else available_families[:6],
            )
        with col_f2:
            organisms = ["All"] + sorted(merops_sub["organism"].dropna().unique().tolist())[:50]
            sel_organism = st.selectbox("Organism", organisms)
        with col_f3:
            contexts = ["All"] + sorted(merops_sub["context"].dropna().unique().tolist())[:20]
            sel_context = st.selectbox("Context (physiological/non-physiological)", contexts)

        filtered = merops_sub.copy()
        if sel_families:
            filtered = filtered[filtered["protease_family"].isin(sel_families)]
        if sel_organism != "All":
            filtered = filtered[filtered["organism"] == sel_organism]
        if sel_context != "All":
            filtered = filtered[filtered["context"] == sel_context]

        # Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Cleavages", f"{len(filtered):,}")
        m2.metric("Unique Proteases", filtered["protease_id"].nunique())
        m3.metric("Unique Substrates", filtered["substrate_uniprot"].nunique())

        # Bar chart: cleavages per protease family
        st.subheader("Cleavages per Protease Family")
        family_counts = (
            filtered.groupby("protease_family").size()
            .reset_index(name="cleavages")
            .sort_values("cleavages", ascending=False)
        )
        # Add family descriptions
        family_labels = {
            "M14": "M14 (Carboxypeptidases A/B)",
            "M01": "M01 (Aminopeptidase N)",
            "M17": "M17 (Leucine aminopeptidase)",
            "M18": "M18 (Aspartyl aminopeptidase)",
            "M24": "M24 (Methionine aminopeptidase)",
            "M28": "M28 (Aminopeptidase Y)",
            "S09": "S09 (DPP-IV family)",
            "S28": "S28 (Lysosomal Pro-X carboxypeptidase)",
            "C01": "C01 (Cathepsins)",
            "A01": "A01 (Pepsin/Cathepsin D)",
        }
        family_counts["label"] = family_counts["protease_family"].map(
            lambda x: family_labels.get(x, x)
        )
        fig_fam = px.bar(
            family_counts, x="label", y="cleavages",
            title="Cleavage Sites by Protease Family",
            labels={"label": "Protease Family", "cleavages": "Cleavage Sites"},
            text="cleavages",
        )
        fig_fam.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig_fam.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig_fam, use_container_width=True)

        # Heatmap: P1 amino acid frequency by protease family
        st.subheader("P1 Amino Acid Frequency by Protease Family")
        st.markdown("Which residues are cleaved most often at the P1 position (immediately N-terminal to the scissile bond)?")

        if "P1" in filtered.columns:
            # Clean P1 values to single-letter codes where possible
            aa_3to1 = {
                "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
                "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
                "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
                "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
            }
            filtered_p1 = filtered.copy()
            filtered_p1["P1_clean"] = filtered_p1["P1"].map(aa_3to1)
            filtered_p1 = filtered_p1.dropna(subset=["P1_clean", "protease_family"])

            if len(filtered_p1) > 0:
                pivot = (
                    filtered_p1.groupby(["protease_family", "P1_clean"]).size()
                    .reset_index(name="count")
                    .pivot(index="protease_family", columns="P1_clean", values="count")
                    .fillna(0)
                )
                # Normalize per family
                pivot_norm = pivot.div(pivot.sum(axis=1), axis=0)

                fig_heat = go.Figure(data=go.Heatmap(
                    z=pivot_norm.values,
                    x=pivot_norm.columns.tolist(),
                    y=[family_labels.get(f, f) for f in pivot_norm.index.tolist()],
                    colorscale="YlOrRd",
                    colorbar=dict(title="Frequency"),
                    text=np.round(pivot_norm.values, 3),
                    texttemplate="%{text:.2f}",
                    hovertemplate="Family: %{y}<br>AA: %{x}<br>Freq: %{z:.3f}<extra></extra>",
                ))
                fig_heat.update_layout(
                    title="P1 Residue Preference by Protease Family (normalized)",
                    xaxis_title="P1 Amino Acid",
                    yaxis_title="Protease Family",
                    height=max(350, len(pivot_norm) * 50 + 150),
                )
                st.plotly_chart(fig_heat, use_container_width=True)
            else:
                st.info("No P1 residue data available for the selected filters.")
        else:
            st.info("P1 column not found in Substrate_search data.")

        # Browsable table
        st.subheader("Cleavage Data Browser")
        display_cols = [c for c in [
            "protease_id", "protease_family", "substrate_name", "cleavage_desc",
            "P3", "P2", "P1", "P1p", "P2p", "P3p", "organism", "context"
        ] if c in filtered.columns]
        st.dataframe(
            filtered[display_cols].head(2000),
            hide_index=True,
            use_container_width=True,
        )
        if len(filtered) > 2000:
            st.caption(f"Showing 2,000 of {len(filtered):,} records. Apply filters to narrow.")


# ============================================================
# TAB 3: PEPTIDE HALF-LIVES (PEPlife2)
# ============================================================
with tab_peplife:
    st.header("PEPlife2 Peptide Half-Life Database")

    peplife_df = load_peplife2()
    if peplife_df is None or len(peplife_df) == 0:
        st.warning(
            "PEPlife2 data not found. Place peplife2_api_*.json files in data/peplife/."
        )
    else:
        # Parse half-life to numeric
        if "half_life" in peplife_df.columns:
            peplife_df["half_life_min"] = peplife_df["half_life"].apply(parse_half_life_minutes)

        # Compute length from sequence
        if "seq" in peplife_df.columns:
            peplife_df["seq_length"] = peplife_df["seq"].str.len()

        # Filters
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        with col_f1:
            proteases = ["All"] + sorted(peplife_df["protease"].dropna().unique().tolist())[:40]
            sel_protease = st.selectbox("Protease", proteases, key="peplife_protease")
        with col_f2:
            vv_opts = ["All"] + sorted(peplife_df["vivo_vitro"].dropna().unique().tolist())
            sel_vv = st.selectbox("In vivo / In vitro", vv_opts, key="peplife_vv")
        with col_f3:
            lc_opts = ["All"] + sorted(peplife_df["lin_cyc"].dropna().unique().tolist())
            sel_lc = st.selectbox("Linear / Cyclic", lc_opts, key="peplife_lc")
        with col_f4:
            mod_opts = ["All"] + sorted(peplife_df["chem_mod"].dropna().unique().tolist())[:30]
            sel_mod = st.selectbox("Chemical Modification", mod_opts, key="peplife_mod")

        filt = peplife_df.copy()
        if sel_protease != "All":
            filt = filt[filt["protease"] == sel_protease]
        if sel_vv != "All":
            filt = filt[filt["vivo_vitro"] == sel_vv]
        if sel_lc != "All":
            filt = filt[filt["lin_cyc"] == sel_lc]
        if sel_mod != "All":
            filt = filt[filt["chem_mod"] == sel_mod]

        # Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Entries", f"{len(filt):,}")
        median_hl = filt["half_life_min"].median() if "half_life_min" in filt.columns else None
        m2.metric("Median Half-Life", f"{median_hl:.1f} min" if median_hl and not np.isnan(median_hl) else "N/A")
        m3.metric("Protease Coverage", filt["protease"].nunique())

        # Histogram of half-life distribution
        st.subheader("Half-Life Distribution")
        hl_data = filt.dropna(subset=["half_life_min"])
        hl_data = hl_data[hl_data["half_life_min"] > 0]
        if len(hl_data) > 0:
            fig_hist = px.histogram(
                hl_data, x="half_life_min",
                nbins=50,
                title="Distribution of Peptide Half-Lives",
                labels={"half_life_min": "Half-Life (minutes)"},
                log_x=True,
            )
            fig_hist.update_layout(
                xaxis_title="Half-Life (minutes, log scale)",
                yaxis_title="Count",
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("No numeric half-life values available for the selected filters.")

        # Scatter: length vs half_life
        st.subheader("Sequence Length vs Half-Life")
        scatter_data = filt.dropna(subset=["half_life_min", "seq_length"])
        scatter_data = scatter_data[scatter_data["half_life_min"] > 0]
        if len(scatter_data) > 0:
            fig_scatter = px.scatter(
                scatter_data, x="seq_length", y="half_life_min",
                title="Peptide Length vs Half-Life",
                labels={"seq_length": "Sequence Length (aa)", "half_life_min": "Half-Life (min)"},
                opacity=0.5,
                log_y=True,
                hover_data=["seq", "protease", "lin_cyc"],
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Insufficient data for scatter plot.")

        # Box plot: half_life by protease
        st.subheader("Half-Life by Protease Type")
        box_data = filt.dropna(subset=["half_life_min"])
        box_data = box_data[box_data["half_life_min"] > 0]
        if len(box_data) > 0:
            # Show top 15 proteases by count
            top_proteases = (
                box_data["protease"].value_counts().head(15).index.tolist()
            )
            box_filtered = box_data[box_data["protease"].isin(top_proteases)]
            if len(box_filtered) > 0:
                fig_box = px.box(
                    box_filtered, x="protease", y="half_life_min",
                    title="Half-Life Distribution by Protease (top 15)",
                    labels={"protease": "Protease", "half_life_min": "Half-Life (min)"},
                    log_y=True,
                )
                fig_box.update_layout(xaxis_tickangle=-30)
                st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("Insufficient data for box plot.")

        # Browsable table
        st.subheader("Data Browser")
        display_cols = [c for c in [
            "seq", "name", "half_life", "units_half", "protease",
            "lin_cyc", "chem_mod", "vivo_vitro", "assay", "source_file"
        ] if c in filt.columns]
        st.dataframe(
            filt[display_cols].head(2000),
            hide_index=True,
            use_container_width=True,
        )
        if len(filt) > 2000:
            st.caption(f"Showing 2,000 of {len(filt):,} records.")


# ============================================================
# TAB 4: DPP-IV INTELLIGENCE
# ============================================================
with tab_dppiv:
    st.header("DPP-IV Intelligence")

    st.markdown("""
**Why DPP-IV matters for ExoPred:**
Dipeptidyl peptidase IV (DPP-IV, CD26) is the exopeptidase that degrades GLP-1, the
hormone behind semaglutide (Ozempic/Wegovy). DPP-IV cleaves dipeptides from the
N-terminus when the penultimate residue is proline or alanine. Understanding its
substrate preferences is critical for designing protease-resistant therapeutic peptides
-- a &#36;50B+ market driven by the GLP-1 agonist class.
""")

    subtab_bench, subtab_chembl = st.tabs(["Peptide Benchmark", "ChEMBL Inhibitors"])

    with subtab_bench:
        dppiv_bench = load_dppiv_benchmark()
        if dppiv_bench is None:
            st.warning(
                "iDPPIV benchmark data not found. Place train/test files in "
                "data/dppiv/idppiv-benchmark/iDPPIV/data/."
            )
        else:
            st.subheader("iDPPIV Benchmark Dataset")
            st.markdown(
                "Binary classification: is this peptide a DPP-IV substrate (positive) or not (negative)? "
                "From Charoenkwan et al. (2022)."
            )

            # Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Peptides", f"{len(dppiv_bench):,}")
            m2.metric("Positive (substrates)", f"{dppiv_bench['label'].sum():,}")
            m3.metric("Negative (non-substrates)", f"{(1 - dppiv_bench['label']).sum():,.0f}")

            # Class balance
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Class Balance")
                balance = dppiv_bench.groupby(["split", "label"]).size().reset_index(name="count")
                balance["class"] = balance["label"].map({1: "Positive (substrate)", 0: "Negative"})
                fig_bal = px.bar(
                    balance, x="split", y="count", color="class",
                    barmode="group",
                    title="Train/Test Split by Class",
                    labels={"split": "Split", "count": "Peptides"},
                    text="count",
                )
                fig_bal.update_traces(textposition="outside")
                st.plotly_chart(fig_bal, use_container_width=True)

            with col_b:
                st.subheader("Sequence Length Distribution")
                fig_len = px.histogram(
                    dppiv_bench, x="length", color=dppiv_bench["label"].map({1: "Positive", 0: "Negative"}),
                    barmode="overlay", nbins=30,
                    title="Length Distribution: Positive vs Negative",
                    labels={"length": "Sequence Length", "color": "Class"},
                    opacity=0.7,
                )
                st.plotly_chart(fig_len, use_container_width=True)

            # Amino acid composition comparison
            st.subheader("Amino Acid Composition: Positive vs Negative")
            standard_aa = list("ACDEFGHIKLMNPQRSTVWY")

            def aa_freq(seqs):
                total = sum(len(s) for s in seqs)
                if total == 0:
                    return {aa: 0 for aa in standard_aa}
                counts = {aa: 0 for aa in standard_aa}
                for s in seqs:
                    for c in s.upper():
                        if c in counts:
                            counts[c] += 1
                return {aa: cnt / total for aa, cnt in counts.items()}

            pos_seqs = dppiv_bench[dppiv_bench["label"] == 1]["sequence"].tolist()
            neg_seqs = dppiv_bench[dppiv_bench["label"] == 0]["sequence"].tolist()
            pos_freq = aa_freq(pos_seqs)
            neg_freq = aa_freq(neg_seqs)

            aa_comp = pd.DataFrame({
                "Amino Acid": standard_aa,
                "Positive (substrate)": [pos_freq[aa] for aa in standard_aa],
                "Negative": [neg_freq[aa] for aa in standard_aa],
            })
            aa_comp["Difference"] = aa_comp["Positive (substrate)"] - aa_comp["Negative"]

            fig_aa = go.Figure()
            fig_aa.add_trace(go.Bar(
                x=aa_comp["Amino Acid"], y=aa_comp["Positive (substrate)"],
                name="Positive (substrate)", marker_color="#636EFA",
            ))
            fig_aa.add_trace(go.Bar(
                x=aa_comp["Amino Acid"], y=aa_comp["Negative"],
                name="Negative", marker_color="#EF553B",
            ))
            fig_aa.update_layout(
                title="Amino Acid Frequency: DPP-IV Substrates vs Non-Substrates",
                xaxis_title="Amino Acid",
                yaxis_title="Frequency",
                barmode="group",
            )
            st.plotly_chart(fig_aa, use_container_width=True)

            # Highlight enriched/depleted
            enriched = aa_comp.nlargest(5, "Difference")
            depleted = aa_comp.nsmallest(5, "Difference")
            col_e, col_d = st.columns(2)
            with col_e:
                st.markdown("**Enriched in DPP-IV substrates:**")
                for _, row in enriched.iterrows():
                    st.markdown(f"- **{row['Amino Acid']}**: +{row['Difference']:.4f}")
            with col_d:
                st.markdown("**Depleted in DPP-IV substrates:**")
                for _, row in depleted.iterrows():
                    st.markdown(f"- **{row['Amino Acid']}**: {row['Difference']:.4f}")

    with subtab_chembl:
        dppiv_chembl = load_dppiv_chembl()
        if dppiv_chembl is None:
            st.warning("ChEMBL DPP-IV data not found. Place chembl284_dpp4_activities.csv in data/dppiv/chembl/.")
        else:
            st.subheader("ChEMBL DPP-IV Inhibitor Activities")
            st.markdown(
                "Small-molecule inhibitors of DPP-IV from ChEMBL (target CHEMBL284). "
                "This data contextualizes the competitive landscape -- sitagliptin, saxagliptin, etc."
            )

            # Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Bioactivities", f"{len(dppiv_chembl):,}")
            ic50_data = dppiv_chembl[dppiv_chembl["standard_type"] == "IC50"]
            m2.metric("IC50 Measurements", f"{len(ic50_data):,}")
            m3.metric("Unique Compounds", dppiv_chembl["molecule_chembl_id"].nunique())

            # IC50 distribution
            st.subheader("IC50 Distribution")
            ic50_numeric = ic50_data.dropna(subset=["standard_value"])
            ic50_numeric = ic50_numeric[ic50_numeric["standard_value"] > 0]
            if len(ic50_numeric) > 0:
                fig_ic50 = px.histogram(
                    ic50_numeric, x="standard_value",
                    nbins=60, log_x=True,
                    title="DPP-IV IC50 Distribution (nM)",
                    labels={"standard_value": "IC50 (nM)"},
                )
                fig_ic50.update_layout(xaxis_title="IC50 (nM, log scale)", yaxis_title="Count")
                st.plotly_chart(fig_ic50, use_container_width=True)

            # Top journals
            if "document_journal" in dppiv_chembl.columns:
                st.subheader("Top Publishing Journals")
                journal_counts = (
                    dppiv_chembl["document_journal"].value_counts().head(15)
                    .reset_index()
                )
                journal_counts.columns = ["Journal", "Publications"]
                fig_journals = px.bar(
                    journal_counts, x="Publications", y="Journal",
                    orientation="h",
                    title="Top 15 Journals for DPP-IV Inhibitor Data",
                )
                fig_journals.update_layout(yaxis=dict(autorange="reversed"), height=500)
                st.plotly_chart(fig_journals, use_container_width=True)

            # Year trend
            if "document_year" in dppiv_chembl.columns:
                st.subheader("Publication Year Trend")
                year_data = dppiv_chembl.dropna(subset=["document_year"])
                year_data = year_data[year_data["document_year"] > 1990]
                year_counts = (
                    year_data.groupby("document_year").size()
                    .reset_index(name="bioactivities")
                )
                fig_year = px.line(
                    year_counts, x="document_year", y="bioactivities",
                    title="DPP-IV Research Activity Over Time",
                    labels={"document_year": "Year", "bioactivities": "Bioactivities Published"},
                    markers=True,
                )
                st.plotly_chart(fig_year, use_container_width=True)


# ============================================================
# TAB 5: EXOPRED MODEL BLUEPRINT
# ============================================================
with tab_blueprint:
    st.header("ExoPred Model Blueprint")
    st.markdown("*Strategic architecture for Sam's exopeptidase degradation prediction model.*")

    st.subheader("1. Model Architecture")
    st.markdown("""
**Input layer: ESM-2 Protein Language Model Embeddings**
- Use Meta's ESM-2 (650M parameter variant) to generate per-residue embeddings
- Each peptide sequence becomes a 1280-dimensional feature vector per residue
- Append terminal chemistry features: N-terminal modification (one-hot, 4 categories),
  C-terminal modification (one-hot, 3 categories), sequence length, net charge, hydrophobicity

**Prediction head options (train all three, benchmark independently):**

1. **Binary classifier** -- cleaved or not by a given exopeptidase
   - Architecture: ESM-2 [CLS] embedding -> 2-layer MLP (512 -> 128 -> 1, sigmoid)
   - Loss: binary cross-entropy with class weights
   - Training data: iDPPIV benchmark + MEROPS exo cleavage sites (positive) + random non-cleaved pairs (negative)

2. **Kinetic regressor** -- predict half-life or rate of degradation
   - Architecture: ESM-2 mean-pool -> 3-layer MLP (512 -> 256 -> 64 -> 1, ReLU)
   - Loss: MSE on log(half-life) to handle the wide dynamic range
   - Training data: PEPlife2 half-lives + Sam's 80K LC-MS time-course data

3. **Multi-enzyme router** -- which exopeptidase(s) will attack this peptide
   - Architecture: ESM-2 [CLS] -> multi-label classifier (one output per enzyme family)
   - Enzyme families: APN (M01), LAP (M17), CPA (M14), CPB (M14), DPP-IV (S09), cathepsins (C01)
   - Loss: binary cross-entropy per label
   - Training data: MEROPS cleavage sites (enzyme-labeled) + Sam's cell-type data (deconvolved)
""")

    st.subheader("2. Training Data Pipeline")
    st.markdown("""
**Stage 1: Public data pre-training**
- MEROPS exopeptidase cleavages: ~10K+ sites with enzyme labels and P1-P1' context
- PEPlife2: ~8K peptides with experimental half-lives under various proteases
- iDPPIV: 1,328 labeled peptides for DPP-IV substrate classification

**Stage 2: Fine-tuning on Sam's proprietary data**
- 80,000 LC-MS data points across 618 peptides x 4 cell types x 6 timepoints x ~5 replicates
- This is the competitive moat -- no one else has systematic, time-resolved, multi-cell degradation data
- Fine-tune the kinetic regressor and multi-enzyme router on this data
- Use the cell-type dimension to learn enzyme mixture signatures (hMSC = high APN, macrophage = high cathepsin)

**Stage 3: Unified feature matrix**
```
[ESM-2 embedding (1280d)] + [N-mod one-hot (4d)] + [C-mod one-hot (3d)]
+ [length (1d)] + [charge (1d)] + [hydrophobicity (1d)] + [P1 identity (20d)]
= 1310-dimensional input per peptide
```
""")

    st.subheader("3. Benchmark Plan")
    st.markdown("""
**Target to beat: ENZ-XGBoost (Pande et al., 2023)**
- Current SOTA for enzyme-peptide interaction prediction
- R-squared = 0.84, trained on 1,119 peptides
- Uses hand-crafted features (AAC, DPC, CTD) -- no protein language model

**ExoPred advantages:**
- **70x more training data** (80K vs 1,119 data points)
- **ESM-2 embeddings** capture evolutionary and structural information that hand-crafted features miss
- **Time-resolved labels** enable kinetic prediction, not just binary classification
- **Multi-cell-type** training reveals enzyme-specific signatures

**Benchmark protocol:**
1. Reproduce ENZ-XGBoost results on their published dataset (baseline)
2. Train ExoPred binary classifier on same data -- expect R-squared > 0.90 from ESM-2 alone
3. Add PEPlife2 + MEROPS pre-training -- expect R-squared > 0.93
4. Fine-tune on Sam's 80K -- target R-squared > 0.95
5. Publish the kinetic regressor as a novel contribution (no existing benchmark)
""")

    st.subheader("4. Three Output Modes")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
**Mode 1: Binary**
- Input: peptide sequence + enzyme
- Output: cleaved / not cleaved
- Use case: screening peptide libraries
- Metric: AUC-ROC, F1
""")
    with col2:
        st.markdown("""
**Mode 2: Kinetic**
- Input: peptide + modifications + cell type
- Output: predicted half-life (minutes)
- Use case: optimizing therapeutic peptides
- Metric: R-squared, MAE on log(t1/2)
""")
    with col3:
        st.markdown("""
**Mode 3: Multi-Enzyme**
- Input: peptide sequence
- Output: vulnerability profile across 6 enzyme families
- Use case: rational design of resistant peptides
- Metric: per-label AUC, Hamming loss
""")

    st.subheader("5. Business Model")
    st.markdown("""
**API pricing tiers (projected):**

| Tier | Queries/month | Price | Target customer |
|------|--------------|-------|-----------------|
| Academic | 1,000 | Free | University labs |
| Startup | 10,000 | &#36;99/mo | Biotech startups |
| Enterprise | 100,000 | &#36;499/mo | Pharma peptide programs |
| Unlimited | Unlimited | &#36;2,499/mo | Large pharma, CROs |

**Revenue model assumptions:**
- 50 academic users (free, builds community + citations)
- 20 startup users x &#36;99 = &#36;1,980/mo
- 5 enterprise users x &#36;499 = &#36;2,495/mo
- 1 unlimited user x &#36;2,499 = &#36;2,499/mo
- **Total ARR: ~&#36;84K at modest adoption**

**Competitive moat:** Sam's 80K LC-MS dataset is proprietary and cannot be replicated without
running the same 618-peptide x 4-cell-type experiment (~6 months of lab work, ~&#36;200K in reagents).
The model improves with every new peptide Sam's lab tests, creating a data flywheel.

**Path to &#36;1M ARR:** Partner with 2-3 pharma companies doing GLP-1 analog or peptide drug programs.
A single pharma seat at &#36;2,499/mo plus custom model fine-tuning at &#36;50K/engagement gets to
&#36;1M within 18 months of launch.
""")

    st.divider()
    st.markdown(
        "*This blueprint was designed for Sam Rozans' ExoPred project. "
        "All public datasets referenced above are freely available. "
        "Sam's 80K LC-MS dataset is proprietary to the Pashuck Lab at Lehigh University.*"
    )
