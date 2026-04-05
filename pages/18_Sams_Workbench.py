import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import shutil
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="Sam's Workbench", page_icon="🔬", layout="wide")
st.title("Sam's Workbench")
st.markdown(
    "Your personal data platform for ExoPred development. "
    "Ingest data, run analyses, explore results."
)

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
TOOLKIT_DIR = Path(__file__).parent.parent
DATA_DIR = TOOLKIT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
TRAINING_DIR = DATA_DIR / "training"
CHECKPOINT_DIR = TOOLKIT_DIR / "exopred" / "checkpoints"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CATEGORY_PATTERNS = {
    "Your Data": ["rozans_*", "rozans-*"],
    "Public Protease": ["merops_*"],
    "Half-Life": ["peplife*"],
    "MMP": ["turk*", "mmp*"],
    "DPP-IV": ["dppiv_*"],
    "Validation": ["*validation*", "*benchmark*", "*bottger*"],
    "Training": ["task_*"],
}


def _categorize(filename: str) -> str:
    """Assign a dataset file to a human-readable category."""
    fn = filename.lower()
    for cat, patterns in CATEGORY_PATTERNS.items():
        for pat in patterns:
            pat_clean = pat.replace("*", "")
            if pat.startswith("*") and pat.endswith("*"):
                if pat_clean in fn:
                    return cat
            elif pat.startswith("*"):
                if fn.endswith(pat_clean):
                    return cat
            elif pat.endswith("*"):
                if fn.startswith(pat_clean):
                    return cat
            else:
                if fn == pat_clean:
                    return cat
    return "Other"


def _status_badge(filepath: Path) -> str:
    """Return a status string for a dataset file."""
    try:
        df = pd.read_csv(filepath, nrows=2)
        if len(df.columns) < 2:
            return "Needs Processing"
        return "Ready"
    except Exception:
        suffix = filepath.suffix.lower()
        if suffix in (".xlsx", ".xls"):
            return "Raw"
        return "Needs Processing"


@st.cache_data(ttl=60)
def _scan_datasets():
    """Walk data directories and build an inventory."""
    results = []
    search_dirs = [PROCESSED_DIR, TRAINING_DIR, DATA_DIR / "turk2015",
                   DATA_DIR / "dppiv", DATA_DIR / "external_validation",
                   DATA_DIR / "merops", DATA_DIR / "peplife",
                   DATA_DIR / "brenda", DATA_DIR / "rozans_si"]
    seen = set()
    for d in search_dirs:
        if not d.exists():
            continue
        for f in sorted(d.iterdir()):
            if f.is_file() and f.suffix.lower() in (".csv", ".xlsx", ".xls", ".tsv", ".txt", ".pkl"):
                if f.name in seen:
                    continue
                seen.add(f.name)
                stat = f.stat()
                row_count = None
                if f.suffix.lower() == ".csv":
                    try:
                        with open(f, "r", errors="replace") as fh:
                            row_count = sum(1 for _ in fh) - 1
                    except Exception:
                        pass
                results.append({
                    "file": f.name,
                    "path": str(f),
                    "size_kb": round(stat.st_size / 1024, 1),
                    "rows": row_count,
                    "category": _categorize(f.name),
                    "status": _status_badge(f) if f.suffix.lower() == ".csv" else "Raw",
                })
    return results


def _has_80k():
    """Check if Sam's 80K LC-MS dataset has been ingested."""
    if (PROCESSED_DIR / "rozans_80k.csv").exists():
        return True
    return any(PROCESSED_DIR.glob("rozans_8*"))


def _load_checkpoint_metrics():
    """Load all checkpoint metric JSON files."""
    metrics = []
    if not CHECKPOINT_DIR.exists():
        return metrics
    for f in sorted(CHECKPOINT_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            data["_file"] = f.name
            metrics.append(data)
        except Exception:
            pass
    return metrics


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_inv, tab_ingest, tab_rec, tab_pred, tab_log, tab_tools = st.tabs([
    "Data Inventory",
    "Ingest Data",
    "Analysis Recommender",
    "Quick Predict",
    "Experiment Log",
    "Tools & Resources",
])

# ===== TAB 1: Data Inventory =============================================
with tab_inv:
    datasets = _scan_datasets()

    # Top-level metrics
    total_datasets = len(datasets)
    total_records = sum(d["rows"] or 0 for d in datasets)
    your_data_records = sum(d["rows"] or 0 for d in datasets if d["category"] == "Your Data")
    public_records = total_records - your_data_records

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total datasets", total_datasets)
    c2.metric("Total records", f"{total_records:,}")
    c3.metric("Your data", f"{your_data_records:,}")
    c4.metric("Public data", f"{public_records:,}")

    if _has_80k():
        c5.markdown("**Sam's 80K**")
        c5.success("LOADED")
    else:
        c5.markdown("**Sam's 80K**")
        c5.error("NOT YET INGESTED")

    st.divider()

    # Group by category
    cats = {}
    for d in datasets:
        cats.setdefault(d["category"], []).append(d)

    for cat in ["Your Data", "Public Protease", "Half-Life", "MMP", "DPP-IV",
                "Validation", "Training", "Other"]:
        items = cats.get(cat, [])
        if not items:
            continue
        cat_rows = sum(d["rows"] or 0 for d in items)
        st.subheader(f"{cat}  ({len(items)} files, {cat_rows:,} records)")
        for d in items:
            status_color = {"Ready": "🟢", "Raw": "🟡", "Needs Processing": "🔴"}.get(d["status"], "⚪")
            row_str = f"{d['rows']:,} rows" if d["rows"] is not None else "N/A rows"
            col_a, col_b = st.columns([3, 1])
            col_a.markdown(f"{status_color} **{d['file']}** — {d['size_kb']} KB, {row_str}")
            col_b.markdown(f"`{d['status']}`")
            with st.expander(f"Quick Look: {d['file']}", expanded=False):
                try:
                    if d["file"].endswith(".csv"):
                        preview = pd.read_csv(d["path"], nrows=5)
                        st.dataframe(preview, use_container_width=True)
                    elif d["file"].endswith((".xlsx", ".xls")):
                        preview = pd.read_excel(d["path"], nrows=5)
                        st.dataframe(preview, use_container_width=True)
                    elif d["file"].endswith(".pkl"):
                        st.info("Pickle file — cannot preview as table.")
                    else:
                        with open(d["path"], "r", errors="replace") as fh:
                            lines = [fh.readline() for _ in range(6)]
                        st.code("".join(lines))
                except Exception as e:
                    st.warning(f"Could not preview: {e}")

# ===== TAB 2: Ingest Data ================================================
with tab_ingest:
    st.subheader("Bring data into the workbench")
    ingest_mode = st.radio(
        "How do you want to add data?",
        ["Upload CSV", "Point to file path", "Grab from URL"],
        horizontal=True,
    )

    # --- Option A: Upload ---
    if ingest_mode == "Upload CSV":
        uploaded = st.file_uploader("Upload your LC-MS data (CSV)", type=["csv"])
        if uploaded is not None:
            try:
                df_up = pd.read_csv(uploaded)
                st.success(f"Loaded **{uploaded.name}** — {len(df_up):,} rows, {len(df_up.columns)} columns")
                st.dataframe(df_up.head(10), use_container_width=True)

                # Check template match
                template_cols = [
                    "sequence", "n_terminal_mod", "c_terminal_mod", "enzyme_ec",
                    "enzyme_name", "enzyme_family", "measurement_type", "value",
                    "curve_values", "curve_timepoints", "conditions_ph",
                    "conditions_temp_c", "conditions_matrix", "source", "confidence",
                ]
                matched = set(df_up.columns) & set(template_cols)
                missing = set(template_cols) - set(df_up.columns)
                if len(missing) == 0:
                    st.success("Matches rozans_template.csv format exactly.")
                elif len(matched) >= 5:
                    st.warning(f"Partial match. Missing columns: {', '.join(sorted(missing))}")
                else:
                    st.info("Custom format detected. You may need to map columns after saving.")

                # Stats
                st.markdown("**Quick stats:**")
                sc1, sc2, sc3 = st.columns(3)
                if "sequence" in df_up.columns:
                    sc1.metric("Unique sequences", df_up["sequence"].nunique())
                if "enzyme_name" in df_up.columns:
                    sc2.metric("Unique enzymes", df_up["enzyme_name"].nunique())
                sc3.metric("Total rows", f"{len(df_up):,}")

                # Save
                save_name = st.text_input("Save as (filename):", value=uploaded.name)
                dest = PROCESSED_DIR / save_name
                if dest.exists():
                    st.warning(f"`{save_name}` already exists in data/processed/. Saving will overwrite.")
                    confirm = st.checkbox("I confirm overwrite", value=False)
                else:
                    confirm = True

                if st.button("Save to data/processed/", type="primary") and confirm:
                    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
                    df_up.to_csv(dest, index=False)
                    st.toast(f"Saved {save_name} to data/processed/")
                    st.cache_data.clear()

                    # Delta comparison
                    before_rows = sum(d["rows"] or 0 for d in _scan_datasets())
                    st.markdown("---")
                    st.markdown(f"**After ingestion:** +{len(df_up):,} records added to your corpus.")
            except Exception as e:
                st.error(f"Error reading file: {e}")

    # --- Option B: File path ---
    elif ingest_mode == "Point to file path":
        filepath_str = st.text_input("Enter a file path on this machine:")
        if filepath_str:
            fp = Path(filepath_str.strip())
            if not fp.exists():
                st.error(f"File not found: {fp}")
            else:
                try:
                    if fp.suffix.lower() in (".xlsx", ".xls"):
                        df_fp = pd.read_excel(fp)
                    else:
                        df_fp = pd.read_csv(fp)
                    st.success(f"Loaded **{fp.name}** — {len(df_fp):,} rows, {len(df_fp.columns)} columns")
                    st.dataframe(df_fp.head(10), use_container_width=True)

                    # Column mapper
                    st.markdown("**Column Mapping** — map your columns to the ExoPred schema:")
                    expected_cols = [
                        "sequence", "n_terminal_mod", "c_terminal_mod", "enzyme_name",
                        "measurement_type", "value", "curve_values", "curve_timepoints",
                    ]
                    source_cols = ["(none)"] + list(df_fp.columns)
                    mapping = {}
                    cols_row = st.columns(4)
                    for i, ec in enumerate(expected_cols):
                        with cols_row[i % 4]:
                            best_guess = ec if ec in df_fp.columns else "(none)"
                            mapping[ec] = st.selectbox(
                                f"→ {ec}", source_cols,
                                index=source_cols.index(best_guess),
                                key=f"map_{ec}",
                            )

                    # Stats
                    seq_col = mapping.get("sequence")
                    enz_col = mapping.get("enzyme_name")
                    if seq_col and seq_col != "(none)" and seq_col in df_fp.columns:
                        st.metric("Unique sequences", df_fp[seq_col].nunique())
                    if enz_col and enz_col != "(none)" and enz_col in df_fp.columns:
                        st.metric("Unique enzymes", df_fp[enz_col].nunique())

                    save_name_fp = st.text_input("Save as:", value=fp.stem + "_ingested.csv", key="fp_save")
                    dest_fp = PROCESSED_DIR / save_name_fp
                    if dest_fp.exists():
                        st.warning(f"`{save_name_fp}` already exists. Saving will overwrite.")
                        confirm_fp = st.checkbox("I confirm overwrite", value=False, key="fp_confirm")
                    else:
                        confirm_fp = True

                    if st.button("Save to data/processed/", key="fp_btn", type="primary") and confirm_fp:
                        # Rename columns per mapping
                        rename_map = {v: k for k, v in mapping.items() if v != "(none)"}
                        df_save = df_fp.rename(columns=rename_map)
                        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
                        df_save.to_csv(dest_fp, index=False)
                        st.toast(f"Saved {save_name_fp}")
                        st.cache_data.clear()
                except Exception as e:
                    st.error(f"Error reading file: {e}")

    # --- Option C: URL ---
    elif ingest_mode == "Grab from URL":
        st.markdown("**Common shortcuts:**")
        bc1, bc2, bc3 = st.columns(3)
        bc1.link_button("MEROPS Latest", "https://ftp.ebi.ac.uk/pub/databases/merops/current_release/")
        bc2.link_button("PEPlife2 API", "https://webs.iiitd.edu.in/raghava/peplife/")
        bc3.link_button("ChEMBL DPP-IV", "https://www.ebi.ac.uk/chembl/target_report_card/CHEMBL284/")

        url_str = st.text_input("Paste a URL to a CSV/Excel file:")
        if url_str:
            with st.spinner("Downloading..."):
                try:
                    if url_str.endswith((".xlsx", ".xls")):
                        df_url = pd.read_excel(url_str)
                    else:
                        df_url = pd.read_csv(url_str)
                    st.success(f"Downloaded — {len(df_url):,} rows, {len(df_url.columns)} columns")
                    st.dataframe(df_url.head(10), use_container_width=True)

                    url_fname = url_str.split("/")[-1].split("?")[0] or "downloaded_data.csv"
                    save_name_url = st.text_input("Save as:", value=url_fname, key="url_save")
                    dest_url = PROCESSED_DIR / save_name_url
                    if dest_url.exists():
                        st.warning(f"`{save_name_url}` already exists. Saving will overwrite.")
                        confirm_url = st.checkbox("Confirm overwrite", value=False, key="url_confirm")
                    else:
                        confirm_url = True

                    if st.button("Save to data/processed/", key="url_btn", type="primary") and confirm_url:
                        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
                        df_url.to_csv(dest_url, index=False)
                        st.toast(f"Saved {save_name_url}")
                        st.cache_data.clear()
                except Exception as e:
                    st.error(f"Error downloading/reading: {e}")

    st.divider()
    st.markdown("**Expected schema** (rozans_template.csv):")
    st.code(
        "sequence, n_terminal_mod, c_terminal_mod, enzyme_ec, enzyme_name, "
        "enzyme_family, measurement_type, value, curve_values, curve_timepoints, "
        "conditions_ph, conditions_temp_c, conditions_matrix, source, confidence",
        language=None,
    )

# ===== TAB 3: Analysis Recommender ========================================
with tab_rec:
    st.subheader("What should you work on next?")

    has_80k = _has_80k()
    has_merops = (PROCESSED_DIR / "merops_exopeptidase_cleavages.csv").exists()
    has_turk = (DATA_DIR / "turk2015" / "mmc2-table-S1.xlsx").exists()
    has_peplife = (PROCESSED_DIR / "peplife2_combined.csv").exists()
    has_dppiv = (PROCESSED_DIR / "dppiv_benchmark.csv").exists() or (PROCESSED_DIR / "dppiv_chembl_ic50.csv").exists()
    has_training = (TRAINING_DIR / "task_a_binary.csv").exists()
    has_v2 = (CHECKPOINT_DIR / "exopred_v2_gbr.pkl").exists()
    has_v4 = (CHECKPOINT_DIR / "exopred_v4_gbr.pkl").exists()
    has_esm = (PROCESSED_DIR / "esm2_embeddings.pkl").exists()

    recs = []

    # --- Priority 1: 80K data ---
    if not has_80k:
        recs.append({
            "priority": "PRIORITY 1",
            "title": "Export your 80K LC-MS data",
            "why": (
                "Your 80K dataset is the single most valuable asset for ExoPred. "
                "Every model trained so far uses public data only. Adding your proprietary LC-MS "
                "measurements will transform prediction accuracy for real therapeutic peptides."
            ),
            "data": "Your LC-MS export from the mass spec pipeline",
            "outcome": "A rozans_80k.csv file in data/processed/ ready for training",
            "action": "template",
        })
        recs.append({
            "priority": "WHILE WAITING",
            "title": "Explore MEROPS exopeptidase profiles",
            "why": (
                "Familiarize yourself with MEROPS P1/P1' cleavage site preferences. "
                "This is the ground truth for protease specificity that ExoPred builds on."
            ),
            "data": "merops_exopeptidase_cleavages.csv" if has_merops else "Not yet processed — run process_datasets.py",
            "outcome": "Understanding of which terminal residues are most vulnerable per enzyme family",
            "action": "notebook_1",
        })
        recs.append({
            "priority": "WHILE WAITING",
            "title": "Cross-reference Paper 3 crosslinkers against Turk MMP-14",
            "why": (
                "Your crosslinker peptides may have MMP-14 cleavage sites embedded. "
                "Turk et al. (2015) mapped 846 MMP-14 substrates — check for overlaps."
            ),
            "data": "turk2015/mmc2-table-S1.xlsx" if has_turk else "Missing Turk data",
            "outcome": "List of your peptides with MMP-14 susceptibility annotations",
            "action": "notebook_2",
        })
    else:
        recs.append({
            "priority": "PRIORITY 1",
            "title": "Train ExoPred v2 on your real data",
            "why": (
                "Your 80K data is loaded. Train a new model version that incorporates "
                "your proprietary LC-MS measurements alongside the public data."
            ),
            "data": "rozans_80k.csv + existing training data",
            "outcome": "New checkpoint with dramatically improved R-squared on real peptides",
            "action": "train_v2",
        })
        recs.append({
            "priority": "HIGH",
            "title": "Benchmark against ENZ-XGBoost (SOTA)",
            "why": "Compare your trained model's R-squared against the published state of the art.",
            "data": "Trained checkpoint + benchmark datasets",
            "outcome": "Side-by-side R-squared, RMSE, and Spearman correlation",
            "action": "benchmark",
        })
        recs.append({
            "priority": "HIGH",
            "title": "Generate predictions for novel peptides",
            "why": "Use the trained model to predict stability for sequences you haven't tested yet.",
            "data": "Trained model + your input sequences",
            "outcome": "Ranked list of peptides by predicted stability",
            "action": "predict",
        })
        recs.append({
            "priority": "MEDIUM",
            "title": "Run DPP-IV analysis on GLP-1 variants",
            "why": (
                "DPP-IV degradation is the key barrier for GLP-1 peptide therapeutics. "
                "Analyze how your stabilization strategies compare."
            ),
            "data": "dppiv_benchmark.csv + your 80K data",
            "outcome": "DPP-IV susceptibility scores for GLP-1 sequence variants",
            "action": "dppiv",
        })

    # --- Always available ---
    recs.append({
        "priority": "AVAILABLE",
        "title": "Explore MEROPS P1/P1' preferences (interactive heatmap)",
        "why": "Visualize which amino acids are preferred at each subsite for each protease family.",
        "data": "merops_exopeptidase_cleavages.csv",
        "outcome": "Interactive heatmap of cleavage site preferences",
        "action": "heatmap",
    })
    recs.append({
        "priority": "AVAILABLE",
        "title": "Compare susceptibility scores against MEROPS ground truth",
        "why": "Validate ExoPred's heuristic scores against real MEROPS cleavage frequencies.",
        "data": "MEROPS cleavage data + ExoPred heuristic tables",
        "outcome": "Correlation plot between predicted and observed cleavage rates",
        "action": "validate",
    })
    if has_turk:
        recs.append({
            "priority": "AVAILABLE",
            "title": "Run Turk MMP-14 cross-reference for Paper 3 peptides",
            "why": "Check if any of your designed peptides contain known MMP-14 cleavage motifs.",
            "data": "turk2015/mmc2-table-S1.xlsx",
            "outcome": "Annotated list of peptides with MMP-14 risk flags",
            "action": "notebook_2",
        })
    if has_peplife:
        recs.append({
            "priority": "AVAILABLE",
            "title": "Browse PEPlife2 for similar peptides to yours",
            "why": "Find published half-life data for peptides that resemble your sequences.",
            "data": "peplife2_combined.csv",
            "outcome": "Nearest-neighbor matches with measured half-lives",
            "action": "peplife_search",
        })
    recs.append({
        "priority": "AVAILABLE",
        "title": "Check if your peptides appear in any public database",
        "why": "Cross-reference your sequences against MEROPS, PEPlife2, ChEMBL, and DPP-IV benchmarks.",
        "data": "All processed datasets",
        "outcome": "Overlap report showing which of your peptides have public data",
        "action": "overlap",
    })

    # Display recommendations
    for r in recs:
        priority_colors = {
            "PRIORITY 1": "🔴",
            "HIGH": "🟠",
            "MEDIUM": "🟡",
            "WHILE WAITING": "🔵",
            "AVAILABLE": "🟢",
        }
        icon = priority_colors.get(r["priority"], "⚪")

        with st.container(border=True):
            st.markdown(f"### {icon} {r['priority']}: {r['title']}")
            st.markdown(f"**Why:** {r['why']}")
            st.markdown(f"**Data:** `{r['data']}`")
            st.markdown(f"**Expected outcome:** {r['outcome']}")

            action = r["action"]
            if action == "template":
                st.markdown("**Next step:** Export your data using this template format:")
                st.code(
                    "sequence,n_terminal_mod,c_terminal_mod,enzyme_ec,enzyme_name,"
                    "enzyme_family,measurement_type,value,curve_values,curve_timepoints,"
                    "conditions_ph,conditions_temp_c,conditions_matrix,source,confidence",
                    language=None,
                )
                st.info(
                    "Save as `rozans_80k.csv` and drop it into data/processed/, "
                    "or use the **Ingest Data** tab to upload it."
                )
            elif action == "notebook_1":
                nb_path = TOOLKIT_DIR / "notebooks" / "01_data_landscape.ipynb"
                st.markdown(f"**Notebook:** `{nb_path}`")
                st.code(f"jupyter notebook {nb_path}", language="bash")
            elif action == "notebook_2":
                nb_path = TOOLKIT_DIR / "notebooks" / "02_your_data_deep_dive.ipynb"
                st.markdown(f"**Notebook:** `{nb_path}`")
                st.code(f"jupyter notebook {nb_path}", language="bash")
            elif action == "train_v2":
                st.code("python -m exopred.train_v2 --data data/processed/rozans_80k.csv", language="bash")
            elif action == "benchmark":
                st.code("python -m exopred.train_v3 --benchmark", language="bash")
            elif action == "predict":
                st.info("Use the **Quick Predict** tab to run predictions interactively.")
            elif action == "dppiv":
                st.code("python -m exopred.train_v2 --enzyme DPP-IV --data data/processed/rozans_80k.csv", language="bash")
            elif action == "heatmap":
                st.info("Go to **page 15 (Protease Specificity)** for the interactive heatmap, or notebook 01.")
            elif action == "validate":
                nb_path = TOOLKIT_DIR / "notebooks" / "03_model_experiments.ipynb"
                st.markdown(f"**Notebook:** `{nb_path}`")
            elif action == "peplife_search":
                st.info("Go to **page 16 (ExoPred Data)** → PEPlife2 tab for sequence search.")
            elif action == "overlap":
                st.info("Go to **page 16 (ExoPred Data)** → Dataset Summary tab.")

# ===== TAB 4: Quick Predict ===============================================
with tab_pred:
    st.subheader("Predict peptide degradation")

    # Load predictor
    @st.cache_resource
    def _get_predictor():
        import sys
        sys.path.insert(0, str(TOOLKIT_DIR))
        try:
            from exopred.predict import ExoPredPredictor
            # Try trained model first
            v2_path = CHECKPOINT_DIR / "exopred_v2_gbr.pkl"
            phase1_path = CHECKPOINT_DIR / "exopred_phase1.pt"
            if v2_path.exists():
                return ExoPredPredictor(str(v2_path))
            elif phase1_path.exists():
                return ExoPredPredictor(str(phase1_path))
            else:
                return ExoPredPredictor()
        except Exception as e:
            st.warning(f"Could not load trained model: {e}. Using heuristic.")
            from exopred.predict import ExoPredPredictor
            return ExoPredPredictor()

    try:
        predictor = _get_predictor()
        model_info = predictor.model_info
        st.caption(f"Model: {model_info['version']} | Mode: {model_info['mode']}")
    except Exception as e:
        st.error(f"Failed to initialize predictor: {e}")
        predictor = None

    seq_input = st.text_area(
        "Enter peptide sequence(s) — one per line or comma-separated:",
        placeholder="YGGFL\nRGDSP\nDRVYIHPF",
        height=100,
    )

    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        n_mod = st.selectbox("N-terminal modification", ["none", "acetyl", "fmoc", "peg", "daa"])
    with pc2:
        c_mod = st.selectbox("C-terminal modification", ["none", "amide", "peg", "daa"])
    with pc3:
        cell_type = st.selectbox("Cell type / matrix", ["human_plasma", "human_serum", "buffer_PBS", "cell_lysate", "other"])

    if st.button("Predict", type="primary") and predictor is not None and seq_input.strip():
        # Parse sequences
        raw_seqs = seq_input.replace(",", "\n").strip().split("\n")
        sequences = [s.strip().upper() for s in raw_seqs if s.strip()]

        if not sequences:
            st.warning("No valid sequences entered.")
        elif len(sequences) > 100:
            st.error("Maximum 100 sequences per batch.")
        else:
            with st.spinner("Running predictions..."):
                results_rows = []
                errors = []
                for seq in sequences:
                    try:
                        result = predictor.predict(seq, enzyme="all", n_mod=n_mod, c_mod=c_mod)
                        row = {
                            "sequence": result["sequence"],
                            "stability_score": result["overall_stability_score"],
                        }
                        # Stability class
                        s = result["overall_stability_score"]
                        if s >= 0.7:
                            row["stability_class"] = "Stable"
                        elif s >= 0.4:
                            row["stability_class"] = "Moderate"
                        else:
                            row["stability_class"] = "Vulnerable"

                        # Per-enzyme probabilities
                        for enz, pred in result["predictions"].items():
                            row[f"{enz}_prob"] = pred["probability"]
                            row[f"{enz}_t½(min)"] = pred["half_life_min"]

                        row["recommendation"] = result["recommendation"]
                        results_rows.append(row)
                    except Exception as e:
                        errors.append(f"{seq}: {e}")

                if errors:
                    for err in errors:
                        st.warning(err)

                if results_rows:
                    df_results = pd.DataFrame(results_rows)

                    # Display main table (without long recommendation text)
                    display_cols = [c for c in df_results.columns if c != "recommendation"]
                    st.dataframe(
                        df_results[display_cols].style.background_gradient(
                            subset=["stability_score"], cmap="RdYlGn", vmin=0, vmax=1,
                        ),
                        use_container_width=True,
                    )

                    # Recommendations
                    for _, row in df_results.iterrows():
                        with st.expander(f"Recommendations for {row['sequence']}"):
                            st.markdown(row["recommendation"])

                    # MEROPS comparison
                    if has_merops:
                        with st.expander("Compare to MEROPS cleavage data"):
                            st.info(
                                "Terminal amino acids for your sequences vs MEROPS frequency data. "
                                "Go to **Protease Specificity** page for the full interactive heatmap."
                            )
                            for _, row in df_results.iterrows():
                                seq = row["sequence"]
                                st.markdown(
                                    f"**{seq}** — N-term: `{seq[0]}`, C-term: `{seq[-1]}` | "
                                    f"Stability: {row['stability_score']}"
                                )

                    # Export
                    csv_out = df_results.to_csv(index=False)
                    st.download_button(
                        "Export results as CSV",
                        csv_out,
                        file_name=f"exopred_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                    )

# ===== TAB 5: Experiment Log ==============================================
with tab_log:
    st.subheader("Model training history")

    metrics_files = _load_checkpoint_metrics()

    if not metrics_files:
        st.info("No checkpoint metrics found. Train a model to see results here.")
    else:
        # Build summary table
        log_rows = []

        # Phase 1
        for m in metrics_files:
            fname = m.get("_file", "")
            if fname == "phase1_metrics.json":
                tm = m.get("test_metrics", {})
                log_rows.append({
                    "Version": "Phase 1 (neural net)",
                    "File": fname,
                    "R-squared (CV)": f"{tm.get('binary_auc', 'N/A'):.4f}" if isinstance(tm.get("binary_auc"), float) else "N/A",
                    "R-squared (leave-seq)": f"{tm.get('halflife_r2', 'N/A'):.4f}" if isinstance(tm.get("halflife_r2"), float) else "N/A",
                    "Features": "physicochemical + enzyme + mod",
                    "Notes": f"Epoch {m.get('best_epoch', '?')}, binary AUC {tm.get('binary_auc', 'N/A'):.3f}" if isinstance(tm.get("binary_auc"), float) else "",
                })
            elif fname == "v2_metrics.json":
                gbr = m.get("gbr", {})
                log_rows.append({
                    "Version": "v2 (GBR)",
                    "File": fname,
                    "R-squared (CV)": f"{gbr.get('cv_r2_mean', 'N/A'):.4f}" if isinstance(gbr.get("cv_r2_mean"), float) else "N/A",
                    "R-squared (leave-seq)": "N/A",
                    "Features": "physicochemical + enzyme + mod",
                    "Notes": f"Train R-squared={gbr.get('train_r2', 0):.4f}" if isinstance(gbr.get("train_r2"), float) else "",
                })
            elif fname == "v3_metrics.json":
                ma = m.get("model_a_v2_only", {})
                log_rows.append({
                    "Version": "v3 (leave-seq-out)",
                    "File": fname,
                    "R-squared (CV)": f"{ma.get('overall_r2', 'N/A'):.4f}" if isinstance(ma.get("overall_r2"), float) else "N/A",
                    "R-squared (leave-seq)": f"{ma.get('fold_r2_mean', 'N/A'):.4f}" if isinstance(ma.get("fold_r2_mean"), float) else "N/A",
                    "Features": "physicochemical + enzyme + mod (no ESM)",
                    "Notes": f"{ma.get('n_folds', '?')} folds, min R-squared={ma.get('fold_r2_min', 0):.3f}" if isinstance(ma.get("fold_r2_min"), float) else "",
                })
            elif fname == "v4_metrics.json":
                mv4 = m.get("model_v4_turk", {})
                log_rows.append({
                    "Version": "v4 (+ Turk MMP-14)",
                    "File": fname,
                    "R-squared (CV)": f"{mv4.get('overall_r2', 'N/A'):.4f}" if isinstance(mv4.get("overall_r2"), float) else "N/A",
                    "R-squared (leave-seq)": f"{mv4.get('fold_r2_mean', 'N/A'):.4f}" if isinstance(mv4.get("fold_r2_mean"), float) else "N/A",
                    "Features": "physicochemical + enzyme + mod + Turk",
                    "Notes": f"Turk delta R-squared = {m.get('turk_delta_r2', 0):.4f}" if isinstance(m.get("turk_delta_r2"), float) else "",
                })

        if log_rows:
            st.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)

        # Feature importance for best model
        st.divider()
        st.subheader("Feature importance (best model)")

        fi_files = sorted(CHECKPOINT_DIR.glob("*feature_importance*.csv"), reverse=True)
        if fi_files:
            # Show latest
            fi_path = fi_files[0]
            try:
                df_fi = pd.read_csv(fi_path)
                if "feature" in df_fi.columns and "importance" in df_fi.columns:
                    df_fi = df_fi.sort_values("importance", ascending=False).head(20)
                    import plotly.express as px
                    fig = px.bar(
                        df_fi, x="importance", y="feature", orientation="h",
                        title=f"Top 20 features ({fi_path.name})",
                    )
                    fig.update_layout(yaxis=dict(autorange="reversed"), height=500)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.dataframe(df_fi.head(20), use_container_width=True)
            except Exception as e:
                st.warning(f"Could not load feature importance: {e}")
        else:
            st.info("No feature importance files found.")

        # Current best recommendation
        st.divider()
        best_r2 = 0.948  # v3 leave-seq-out
        st.info(
            f"**Current best:** v3 with leave-sequence-out R-squared = {best_r2:.3f}. "
            "To improve, add Sam's 80K proprietary LC-MS data and retrain."
        )

# ===== TAB 6: Tools & Resources ===========================================
with tab_tools:
    st.subheader("Bioinformatics tools for ExoPred development")

    tools_data = {
        "Protein Language Models": [
            ("ESM-2", "Meta's protein language model — state-of-the-art sequence embeddings", "pip install fair-esm", "https://github.com/facebookresearch/esm"),
            ("ProtTrans", "Pretrained transformer models for protein sequences (ProtBERT, ProtT5)", "pip install transformers sentencepiece", "https://github.com/agemagician/ProtTrans"),
            ("PeptideBERT", "BERT fine-tuned for peptide property prediction", "pip install transformers", "https://huggingface.co/ChatterjeeLab/PeptideBERT"),
        ],
        "Protease Prediction": [
            ("PROSPERous", "Protease substrate specificity prediction server", "Web-based", "https://prosperous.erc.monash.edu/"),
            ("DeepCleave", "Deep learning for protease cleavage site prediction", "pip install deepcleave", "https://github.com/arontier/deepcleave"),
        ],
        "Peptide Properties": [
            ("peptides", "Physicochemical properties for peptide sequences", "pip install peptides", "https://github.com/althonos/peptides.py"),
            ("modlAMP", "Antimicrobial peptide descriptors and ML features", "pip install modlamp", "https://github.com/alexarnimueller/modlAMP"),
        ],
        "Structure": [
            ("ESMFold", "Fast protein structure prediction from sequence", "pip install fair-esm", "https://github.com/facebookresearch/esm"),
            ("biotite", "Structural bioinformatics library (PDB parsing, DSSP, contacts)", "pip install biotite", "https://www.biotite-python.org/"),
            ("FreeSASA", "Solvent-accessible surface area computation", "pip install freesasa", "https://freesasa.github.io/"),
        ],
        "ML / Data Science": [
            ("XGBoost", "Gradient boosting — baseline for tabular peptide data", "pip install xgboost", "https://xgboost.readthedocs.io/"),
            ("SHAP", "Model interpretability — explain which features drive predictions", "pip install shap", "https://github.com/shap/shap"),
            ("Optuna", "Hyperparameter optimization framework", "pip install optuna", "https://optuna.org/"),
        ],
        "Database Access": [
            ("brendapy", "BRENDA enzyme database client", "pip install brendapy", "https://github.com/matthiaskoenig/brendapy"),
            ("ChEMBL client", "Python client for ChEMBL bioactivity database", "pip install chembl-webresource-client", "https://github.com/chembl/chembl_webresource_client"),
            ("bioservices", "Access to biological web services (UniProt, KEGG, etc.)", "pip install bioservices", "https://bioservices.readthedocs.io/"),
        ],
        "Visualization": [
            ("logomaker", "Publication-quality sequence logos", "pip install logomaker", "https://logomaker.readthedocs.io/"),
            ("nglview", "3D molecular visualization in Jupyter", "pip install nglview", "https://github.com/nglviewer/nglview"),
        ],
    }

    for category, tools in tools_data.items():
        st.markdown(f"#### {category}")
        for name, desc, install, url in tools:
            tc1, tc2 = st.columns([3, 1])
            tc1.markdown(f"**[{name}]({url})** — {desc}")
            tc2.code(install, language="bash")
        st.markdown("")

    st.divider()
    st.subheader("Quick install (all recommended packages)")
    st.code("pip install xgboost shap optuna logomaker freesasa biotite lmfit modlamp", language="bash")

    st.divider()
    st.subheader("Jupyter notebooks")

    notebooks = [
        ("01_data_landscape.ipynb", "Survey all available datasets — MEROPS, PEPlife2, Turk, DPP-IV, and your data"),
        ("02_your_data_deep_dive.ipynb", "Deep dive into Sam's proprietary LC-MS data — quality checks, coverage analysis, sequence clustering"),
        ("03_model_experiments.ipynb", "Model training experiments — feature engineering, cross-validation, benchmarks"),
    ]
    for nb_name, nb_desc in notebooks:
        nb_path = TOOLKIT_DIR / "notebooks" / nb_name
        exists = nb_path.exists()
        status = "Ready" if exists else "Not found"
        st.markdown(f"{'🟢' if exists else '🔴'} **{nb_name}** — {nb_desc} (`{status}`)")

    st.divider()
    st.subheader("Full guide")
    workbench_md = TOOLKIT_DIR / "SAM_WORKBENCH.md"
    if workbench_md.exists():
        st.markdown(f"Read the complete workbench guide: `{workbench_md}`")
        with st.expander("Show SAM_WORKBENCH.md"):
            try:
                content = workbench_md.read_text()
                # Escape dollar signs to prevent LaTeX rendering
                content = content.replace("$", "&#36;")
                st.markdown(content)
            except Exception as e:
                st.warning(f"Could not read: {e}")
    else:
        st.info("SAM_WORKBENCH.md not found in toolkit root.")
