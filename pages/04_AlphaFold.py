import streamlit as st
import requests
import py3Dmol
from stmol import showmol

st.set_page_config(page_title="AlphaFold DB Lookup", page_icon="🔬", layout="wide")
st.title("AlphaFold DB Lookup")
st.markdown("Predicted protein structures from DeepMind's AlphaFold (200M+ proteins). Enter a UniProt accession ID.")

API_BASE = "https://alphafold.ebi.ac.uk/api"

# Common examples
examples = {
    "EGFR (P00533)": "P00533",
    "p53 (P04637)": "P04637",
    "Insulin (P01308)": "P01308",
    "GFP (P42212)": "P42212",
    "Hemoglobin alpha (P69905)": "P69905",
    "BRCA1 (P38398)": "P38398",
}

col1, col2 = st.columns([2, 1])
with col1:
    uniprot_id = st.text_input("UniProt Accession ID", value="P00533", placeholder="e.g. P00533")
with col2:
    example = st.selectbox("Or pick an example", [""] + list(examples.keys()))
    if example:
        uniprot_id = examples[example]

if st.button("Fetch Structure", type="primary") and uniprot_id:
    uniprot_id = uniprot_id.strip().upper()

    with st.spinner(f"Fetching AlphaFold prediction for {uniprot_id}..."):
        try:
            resp = requests.get(f"{API_BASE}/prediction/{uniprot_id}", timeout=15)
            if resp.status_code == 404:
                st.error(f"No AlphaFold prediction found for {uniprot_id}. Check the UniProt ID.")
                st.stop()
            resp.raise_for_status()
            entry = resp.json()[0]
        except requests.RequestException as e:
            st.error(f"API error: {e}")
            st.stop()

    # Metadata
    st.markdown("---")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("UniProt ID", entry.get("uniprotAccession", ""))
        st.markdown(f"**Organism:** {entry.get('organismScientificName', 'N/A')}")
    with col_b:
        st.metric("Sequence Length", entry.get("uniprotEnd", "N/A"))
        st.markdown(f"**Gene:** {entry.get('gene', 'N/A')}")
    with col_c:
        st.metric("Model Version", entry.get("modelCreatedDate", "N/A"))
        st.markdown(f"**DB Version:** {entry.get('latestVersion', 'N/A')}")

    # Download PDB
    pdb_url = entry.get("pdbUrl", "")
    if pdb_url:
        with st.spinner("Downloading PDB..."):
            pdb_resp = requests.get(pdb_url, timeout=30)
            pdb_data = pdb_resp.text

        st.markdown("### 3D Structure (colored by pLDDT confidence)")
        st.markdown("**Blue** = high confidence (pLDDT > 90) | **Red** = low confidence (pLDDT < 50)")

        # 3D viewer
        view = py3Dmol.view(width=800, height=600)
        view.addModel(pdb_data, "pdb")
        view.setStyle({
            "cartoon": {
                "colorscheme": {
                    "prop": "b",
                    "gradient": "roygb",
                    "min": 30,
                    "max": 100
                }
            }
        })
        view.zoomTo()
        view.setBackgroundColor("white")
        showmol(view, height=600, width=800)

        # pLDDT distribution
        bfactors = []
        for line in pdb_data.split("\n"):
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                try:
                    bf = float(line[60:66].strip())
                    bfactors.append(bf)
                except ValueError:
                    pass

        if bfactors:
            import plotly.graph_objects as go

            st.markdown("### pLDDT Confidence per Residue")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(1, len(bfactors) + 1)),
                y=bfactors,
                mode="lines",
                line=dict(color="#0E4D92", width=1.5),
                name="pLDDT"
            ))
            fig.add_hrect(y0=90, y1=100, fillcolor="blue", opacity=0.08, line_width=0,
                          annotation_text="Very high", annotation_position="top left")
            fig.add_hrect(y0=70, y1=90, fillcolor="cyan", opacity=0.08, line_width=0,
                          annotation_text="Confident", annotation_position="top left")
            fig.add_hrect(y0=50, y1=70, fillcolor="yellow", opacity=0.08, line_width=0,
                          annotation_text="Low", annotation_position="top left")
            fig.add_hrect(y0=0, y1=50, fillcolor="red", opacity=0.08, line_width=0,
                          annotation_text="Very low", annotation_position="top left")
            fig.update_layout(
                xaxis_title="Residue Position",
                yaxis_title="pLDDT Score",
                yaxis=dict(range=[0, 105]),
                height=350,
                margin=dict(t=20)
            )
            st.plotly_chart(fig, use_container_width=True)

            avg_plddt = sum(bfactors) / len(bfactors)
            high_conf = sum(1 for b in bfactors if b >= 90)
            st.markdown(
                f"**Average pLDDT:** {avg_plddt:.1f} | "
                f"**Residues >= 90:** {high_conf}/{len(bfactors)} ({100*high_conf/len(bfactors):.0f}%)"
            )

        # Download links
        st.markdown("### Downloads")
        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            st.download_button("Download PDB", pdb_data, file_name=f"AF-{uniprot_id}.pdb", mime="chemical/x-pdb")
        with col_d2:
            cif_url = entry.get("cifUrl", "")
            if cif_url:
                st.markdown(f"[Download mmCIF]({cif_url})")
        with col_d3:
            pae_url = entry.get("paeDocUrl", "")
            if pae_url:
                st.markdown(f"[Download PAE (JSON)]({pae_url})")
