import streamlit as st
import py3Dmol
from stmol import showmol

st.set_page_config(
    page_title="Bio-AI Toolkit",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Bio-AI Toolkit")
st.markdown("Open-source biological intelligence tools for peptide engineering and biomaterials design.")
st.markdown("*A [Steps Ventures](https://stepsventures.com) platform — built by Sam Rozans, Venture Associate*")

# ---- Hero: Enzyme 3D visualization ----
st.markdown("---")

col_hero1, col_hero2 = st.columns([2, 1])
with col_hero1:
    view = py3Dmol.view(query="pdb:3CPA", width=700, height=400)
    view.setStyle({}, {"cartoon": {"colorscheme": "spectrum"}})
    view.setStyle({"hetflag": True}, {"stick": {"colorscheme": "greenCarbon", "radius": 0.3}})
    view.setStyle({"elem": "Zn"}, {"sphere": {"radius": 0.8, "color": "gray"}})
    view.setStyle({"elem": "ZN"}, {"sphere": {"radius": 0.8, "color": "gray"}})
    view.setBackgroundColor("white")
    view.zoomTo()
    showmol(view, height=400, width=700)

with col_hero2:
    st.markdown("### Carboxypeptidase A")
    st.markdown("*with Gly-Tyr substrate in active site*")
    st.markdown(
        "This zinc metalloprotease clips C-terminal residues from peptides — "
        "one of the key enzymes degrading adhesion peptides in biomaterial hydrogels. "
        "Rozans et al. (2024) showed C-terminal proline resists this enzyme, "
        "while Phe/Tyr/Trp are cleaved fastest."
    )
    st.markdown("[PDB: 3CPA](https://www.rcsb.org/structure/3CPA) | 2.0 A resolution")
    st.page_link("pages/02_Enzyme_Visualization.py", label="Explore all enzyme structures", icon="🎨")

st.markdown("---")

# ---- Rozans Analysis + Enzyme Viz + Commercial (top section) ----
st.markdown("## Rozans Peptide Intelligence")

col_top1, col_top2, col_top3 = st.columns(3)

with col_top1:
    st.markdown("### Rozans Analysis")
    st.markdown("618 peptides through every tool: BioPython, Protease Atlas, Degradation Predictor, MMP-14, Self-Assembly. Preview for 80K dataset.")
    st.page_link("pages/01_Rozans_Analysis.py", label="Open Rozans Analysis", icon="🔬")

with col_top2:
    st.markdown("### Enzyme-Peptide 3D Viewer")
    st.markdown("Interactive 3D structures of CPA, APN/CD13, MMP-14, MMP-2 with substrates bound in active sites.")
    st.page_link("pages/02_Enzyme_Visualization.py", label="Open 3D Viewer", icon="🎨")

with col_top3:
    st.markdown("### Commercial Opportunities")
    st.markdown("Startup concepts, licensing pathways, market analysis, and competitive moat for Rozans peptide degradation IP.")
    st.page_link("pages/03_Commercial_Opportunities.py", label="Open Commercial Analysis", icon="💼")

st.markdown("---")

# ---- Public Databases ----
st.markdown("## Public Databases")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### AlphaFold DB")
    st.markdown("Predicted protein structures by UniProt ID. 3D viewer with pLDDT confidence.")
    st.page_link("pages/04_AlphaFold.py", label="Open AlphaFold Lookup", icon="🔬")

with col2:
    st.markdown("### UniProt")
    st.markdown("Protein knowledge base. Function, domains, GO terms.")
    st.page_link("pages/05_UniProt.py", label="Open UniProt Search", icon="🧫")

with col3:
    st.markdown("### PubChem")
    st.markdown("Compound search by name/CID. Properties, 2D structures, Lipinski Ro5.")
    st.page_link("pages/06_PubChem.py", label="Open PubChem Search", icon="💊")

st.markdown("---")

# ---- Sequence Analysis ----
st.markdown("## Sequence Analysis")
col4, col5, col6 = st.columns(3)

with col4:
    st.markdown("### Sequence Tools")
    st.markdown("MW, pI, GRAVY, composition, hydrophobicity, pairwise alignment.")
    st.page_link("pages/07_Sequence_Tools.py", label="Open Sequence Tools", icon="🧮")

with col5:
    st.markdown("### BLAST Search")
    st.markdown("Remote blastp against NCBI (nr, swissprot, pdb, refseq).")
    st.page_link("pages/09_BLAST.py", label="Open BLAST", icon="🎯")

with col6:
    st.markdown("### Peptide Library")
    st.markdown("618 Rozans peptides (Pashuck Lab, Lehigh). MW, pI, GRAVY for all.")
    st.page_link("pages/08_Peptide_Library.py", label="Open Peptide Library", icon="📊")

st.markdown("---")

# ---- Pashuck Lab Suite ----
st.markdown("## Pashuck Lab Analysis Suite")
st.markdown("Calibrated against published data from Rozans, Wu, Moghaddam et al. (Lehigh University, 2024-2026)")

col7, col8 = st.columns(2)

with col7:
    st.markdown("### Degradation Predictor")
    st.markdown("Predict peptide stability at 48h by terminal chemistry, residue, and cell type. Calibrated against Rozans 2024 experimental data.")
    st.page_link("pages/10_Degradation_Predictor.py", label="Open Degradation Predictor", icon="⚗️")

    st.markdown("### MMP-14 Cleavage Predictor")
    st.markdown("19x19 heatmap of KLVAD-X1X2-ASAE crosslinker cleavability. Calibrated against Wu/Rozans 2025 screening data.")
    st.page_link("pages/11_MMP14_Predictor.py", label="Open MMP-14 Predictor", icon="✂️")

with col8:
    st.markdown("### Hydrogel Designer")
    st.markdown("Configure PEG-peptide hydrogels: backbone, adhesion ligand, crosslinker. Design rules from 6 years of Pashuck Lab publications.")
    st.page_link("pages/12_Hydrogel_Designer.py", label="Open Hydrogel Designer", icon="🧪")

    st.markdown("### Protease Specificity Atlas")
    st.markdown("Exopeptidase (APN, LAP, CPA, CPB) + MMP (1, 2, 9, 14) substrate profiles. Eckhard 4,300 sites. Kinetic benchmarks.")
    st.page_link("pages/15_Protease_Specificity.py", label="Open Protease Atlas", icon="🗺️")

st.markdown("---")

# ---- Materials Science ----
st.markdown("## Materials Science")
st.markdown("Polymer degradation models and peptide self-assembly prediction from collaborating labs")

col9, col10 = st.columns(2)

with col9:
    st.markdown("### Polymer Degradation Calculator")
    st.markdown("QSPR models for tyrosine-derived polycarbonates: Tg + MW retention from composition. Kohn Lab (Rutgers/NJCBM).")
    st.page_link("pages/13_Polymer_Degradation.py", label="Open Polymer Calculator", icon="🧱")

with col10:
    st.markdown("### Self-Assembly Predictor")
    st.markdown("Dipeptide AP heatmap (Frederix/Ulijn), Stupp PA database (43 entries), Fmoc interaction energies (Zanuy QM).")
    st.page_link("pages/14_Self_Assembly.py", label="Open Self-Assembly", icon="🔗")

st.markdown("---")
st.markdown("#### Data Sources")
st.markdown(
    "AlphaFold DB (EBI) | UniProt | PubChem (NCBI) | NCBI BLAST | MEROPS | "
    "Rozans et al. ACS Biomater 2024 | Rozans et al. JBMR-A 2025 | "
    "Wu/Rozans et al. Adv Healthcare Mater 2025 | Moghaddam et al. JACS 2026 | "
    "Moghaddam et al. Acta Biomaterialia 2025 | "
    "Srinivasan/Kohn J Mater Sci 2013 | Bateman/Kohn Polymer 2007 | "
    "Frederix/Ulijn J Phys Chem Lett 2011 | Frederix Nature Chemistry 2015 | "
    "Zanuy et al. PCCP 2016 | Pashuck & Stupp JACS 2010 | "
    "Eckhard et al. Matrix Biology 2016 | Ratnikov et al. PNAS 2014 | "
    "Chen et al. PNAS 2012 | Patterson & Hubbell Biomaterials 2010 | "
    "CleaveNet Nature Comm 2026 | MEROPS (EBI)"
)
st.caption("Built by Steps Ventures | Sam Rozans, Venture Associate | Mike German, Ph.D., P.E., Managing Partner")
