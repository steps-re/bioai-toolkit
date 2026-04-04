import streamlit as st
import pandas as pd
import py3Dmol
from stmol import showmol
import requests

st.set_page_config(page_title="Enzyme-Peptide Visualization", page_icon="🎨", layout="wide")
st.title("Enzyme-Peptide Cleavage Visualization")
st.markdown(
    "3D structures of the proteases that degrade Sam's peptide libraries — "
    "aminopeptidases, carboxypeptidases, and MMPs — with substrates bound in their active sites."
)

# ============================================================
# PDB STRUCTURES
# ============================================================

STRUCTURES = {
    "Carboxypeptidase A + Gly-Tyr substrate (3CPA)": {
        "pdb_id": "3CPA",
        "description": "Bovine Carboxypeptidase A with dipeptide substrate Gly-Tyr bound in the active site. "
                       "This is the enzyme that clips C-terminal residues from Sam's unprotected peptides. "
                       "The zinc ion (gray sphere) activates the scissile bond for hydrolysis.",
        "resolution": "2.0 A",
        "enzyme": "Carboxypeptidase A (CPA)",
        "bound": "Gly-Tyr dipeptide (true substrate complex)",
        "relevance": "CPA degrades C-terminal Phe, Tyr, Trp, Leu fastest — exactly what Sam's Paper 1 confirmed. "
                     "C-terminal Pro is resistant because its ring can't fit the S1' pocket.",
        "style": "cartoon_ligand",
        "highlight_residues": None,
    },
    "Aminopeptidase N (CD13) + Bestatin (4FYS)": {
        "pdb_id": "4FYS",
        "description": "E. coli Aminopeptidase N with bestatin (a transition-state analog) bound in the active site. "
                       "APN/CD13 is the membrane-bound aminopeptidase responsible for N-terminal degradation "
                       "of Sam's unprotected (NH2-) peptides.",
        "resolution": "1.8 A",
        "enzyme": "Aminopeptidase N (APN/CD13)",
        "bound": "Bestatin (peptidomimetic inhibitor)",
        "relevance": "Sam's Paper 1 showed NH2-terminated peptides lose 80% in 48h. APN is the likely culprit. "
                     "Acetylation (Ac-) blocks APN recognition by eliminating the free alpha-amine.",
        "style": "cartoon_ligand",
        "highlight_residues": None,
    },
    "MMP-14 catalytic domain + inhibitor (3MA2)": {
        "pdb_id": "3MA2",
        "description": "MMP-14 (MT1-MMP) catalytic domain with a hydroxamate inhibitor coordinating the active-site zinc. "
                       "This is the membrane-anchored protease that cleaves Sam's KLVADLMASAE crosslinker pericellularly.",
        "resolution": "1.7 A",
        "enzyme": "MMP-14 (MT1-MMP)",
        "bound": "Hydroxamate inhibitor (zinc chelator)",
        "relevance": "Paper 3's key finding: KLVADLMASAE is cleaved by MMP-14 at the cell surface, NOT by soluble MMPs. "
                     "The Leu at P1' fits perfectly in MMP-14's deep hydrophobic S1' pocket.",
        "style": "cartoon_ligand",
        "highlight_residues": None,
    },
    "MMP-2 pro-form — peptide in active site cleft (1HOV)": {
        "pdb_id": "1HOV",
        "description": "Full-length pro-MMP-2 with its pro-domain peptide threading through the entire active site cleft. "
                       "This is the closest view of what a peptide substrate looks like inside an MMP — "
                       "the pro-domain mimics a substrate running through the S3-S3' subsites.",
        "resolution": "2.8 A",
        "enzyme": "MMP-2 (Gelatinase A, pro-form)",
        "bound": "Pro-domain peptide (occupies active site groove)",
        "relevance": "MMP-2 is a soluble MMP — Sam's KLVADLMASAE crosslinker was designed to RESIST cleavage "
                     "by MMP-2 while remaining sensitive to membrane-bound MMP-14.",
        "style": "cartoon_chain",
        "highlight_residues": None,
    },
    "Human CD13 ectodomain + Bestatin (4FYQ)": {
        "pdb_id": "4FYQ",
        "description": "Human CD13/Aminopeptidase N ectodomain — the actual human enzyme that degrades peptides "
                       "in Sam's cell culture experiments. Bestatin bound in active site.",
        "resolution": "2.5 A",
        "enzyme": "Human CD13/APN",
        "bound": "Bestatin",
        "relevance": "This is the human version of the enzyme degrading Sam's peptides in hMSC, hUVEC, "
                     "and macrophage cultures. CD13 is expressed on all three cell types.",
        "style": "cartoon_ligand",
        "highlight_residues": None,
    },
    "CPA high-resolution with TS analog (1M4L)": {
        "pdb_id": "1M4L",
        "description": "Carboxypeptidase A at 1.38 A resolution with a phosphonate transition-state analog. "
                       "Shows the exact geometry of the catalytic zinc center during peptide bond hydrolysis.",
        "resolution": "1.38 A",
        "enzyme": "Carboxypeptidase A",
        "bound": "Phosphonate transition-state analog",
        "relevance": "Ultra-high resolution view of CPA's catalytic mechanism. The zinc ion polarizes the "
                     "carbonyl of the scissile peptide bond, making it susceptible to nucleophilic attack by water.",
        "style": "cartoon_ligand",
        "highlight_residues": None,
    },
}

# Sam's peptide sequences for context
SAMS_PEPTIDES = {
    "RGEFVL (most common scaffold)": "RGEFVL",
    "RGEFVP (most resistant — Pro)": "RGEFVP",
    "RGEFVF (most susceptible — Phe)": "RGEFVF",
    "RGEFVH (Histidine exception)": "RGEFVH",
    "GRGDS (linear RGD)": "GRGDS",
    "KLVADLMASAE (optimized crosslinker)": "KLVADLMASAE",
    "GPQGIWGQ (PanMMP benchmark)": "GPQGIWGQ",
}

# ============================================================
# UI
# ============================================================

tab1, tab2 = st.tabs(["Enzyme Structures", "Peptide-Enzyme Context"])

with tab1:
    selected = st.selectbox("Select enzyme structure", list(STRUCTURES.keys()))
    struct = STRUCTURES[selected]

    col1, col2 = st.columns([2, 1])

    with col2:
        st.markdown(f"**PDB:** [{struct['pdb_id']}](https://www.rcsb.org/structure/{struct['pdb_id']})")
        st.markdown(f"**Resolution:** {struct['resolution']}")
        st.markdown(f"**Enzyme:** {struct['enzyme']}")
        st.markdown(f"**Bound:** {struct['bound']}")
        st.markdown(f"**Relevance to Rozans data:**")
        st.markdown(f"> {struct['relevance']}")

    with col1:
        st.markdown(f"### {struct['enzyme']}")
        st.markdown(struct['description'])

        # Visualization options
        vis_col1, vis_col2, vis_col3 = st.columns(3)
        with vis_col1:
            style = st.selectbox("Protein style", ["cartoon", "surface", "stick", "sphere"], key=f"style_{selected}")
        with vis_col2:
            color_scheme = st.selectbox("Color", ["spectrum", "chain", "ssColormap", "white"], key=f"color_{selected}")
        with vis_col3:
            show_ligand = st.checkbox("Highlight ligand/substrate", value=True, key=f"lig_{selected}")

        with st.spinner(f"Loading {struct['pdb_id']} from RCSB..."):
            view = py3Dmol.view(query=f"pdb:{struct['pdb_id']}", width=750, height=550)

            # Protein
            if color_scheme == "chain":
                view.setStyle({"chain": "A"}, {style: {"color": "#0E4D92"}})
                view.setStyle({"chain": "B"}, {style: {"color": "#E8792B"}})
            else:
                view.setStyle({}, {style: {"colorscheme": color_scheme if color_scheme != "white" else {"prop": "b", "gradient": "roygb", "min": 0, "max": 100}}})

            # Ligand/substrate highlighting
            if show_ligand:
                view.setStyle({"hetflag": True}, {"stick": {"colorscheme": "greenCarbon", "radius": 0.3}})
                # Zinc ions
                view.setStyle({"elem": "Zn"}, {"sphere": {"radius": 0.8, "color": "gray"}})
                view.setStyle({"elem": "ZN"}, {"sphere": {"radius": 0.8, "color": "gray"}})

            view.setBackgroundColor("white")
            view.zoomTo()
            showmol(view, height=550, width=750)

    st.markdown("---")
    st.markdown("### All Available Structures")
    struct_df = pd.DataFrame([
        {"Structure": k, "PDB": v["pdb_id"], "Resolution": v["resolution"],
         "Enzyme": v["enzyme"], "Bound": v["bound"]}
        for k, v in STRUCTURES.items()
    ])
    st.dataframe(struct_df, hide_index=True)

with tab2:
    st.markdown("### How These Enzymes Degrade Sam's Peptides")

    st.markdown("""
    #### The Degradation Pathway

    When Sam places a peptide like **NH2-RGEFV-L-COOH** in a PEG hydrogel with cells,
    here's what happens at the molecular level:

    ```
    Step 1: Cell secretes / displays proteases
        hMSC → high APN (CD13) + CPA + some MMPs
        hUVEC → moderate APN + CPA
        Macrophage → low APN + CPA

    Step 2: Aminopeptidase N (CD13) attacks the N-terminus
        NH2-R-GEFVL → NH2-G-EFVL → NH2-E-FVL → ...
        (Sequential removal of N-terminal residues)
        BLOCKED by: Ac- or Ac-βA- modification

    Step 3: Carboxypeptidase A attacks the C-terminus
        RGEFV-L-COOH → RGEFV-COOH → RGEF-COOH → ...
        (Sequential removal of C-terminal residues)
        BLOCKED by: -amide or -βA modification

    Step 4: Both ends degrade simultaneously
        After 48h with hMSCs:
        - NH2/COOH: only 20% remaining (both ends exposed)
        - Ac-βA/C-βA: 83% remaining (both ends protected)
    ```
    """)

    st.markdown("#### Sam's Peptides in Context")

    import pandas as pd

    context_data = [
        {"Peptide": "NH2-RGEFVL-COOH", "N-term enzyme": "APN (CD13)", "C-term enzyme": "CPA",
         "N-term vulnerability": "High (Arg)", "C-term vulnerability": "High (Leu)",
         "Predicted 48h (hMSC)": "~20%", "Protection needed": "Ac- and -amide minimum"},
        {"Peptide": "NH2-RGEFVP-COOH", "N-term enzyme": "APN (CD13)", "C-term enzyme": "CPA",
         "N-term vulnerability": "High (Arg)", "C-term vulnerability": "Very Low (Pro)",
         "Predicted 48h (hMSC)": "~45%", "Protection needed": "Only N-term (Pro protects C-term)"},
        {"Peptide": "NH2-RGEFVH-COOH", "N-term enzyme": "APN (CD13)", "C-term enzyme": "CPA",
         "N-term vulnerability": "High (Arg)", "C-term vulnerability": "Moderate (His)",
         "Predicted 48h (hMSC)": "~15%", "Protection needed": "HISTIDINE EXCEPTION — Ac doesn't fully protect"},
        {"Peptide": "KLVADLMASAE", "N-term enzyme": "APN + MMP-14", "C-term enzyme": "CPA",
         "N-term vulnerability": "Moderate (Lys)", "C-term vulnerability": "Low (Glu)",
         "Predicted 48h (hMSC)": "Crosslinker — cleaved at D|L by MMP-14", "Protection needed": "Designed to be cleaved pericellularly"},
        {"Peptide": "Ac-βA-GRGDS-βA", "N-term enzyme": "Blocked", "C-term enzyme": "Blocked",
         "N-term vulnerability": "None (Ac-βA)", "C-term vulnerability": "None (βA)",
         "Predicted 48h (hMSC)": "~83%", "Protection needed": "Already optimally protected"},
    ]

    st.dataframe(pd.DataFrame(context_data), hide_index=True)

    st.markdown("""
    #### Why This Matters for Biomaterials

    Every tissue engineering hydrogel uses adhesion peptides (RGD, IKVAV, YIGSR) to support
    cell attachment. **Most researchers don't know these peptides degrade during culture.**

    Sam's work quantified this problem for the first time and showed that simple terminal
    modifications can prevent 80% of the degradation. The structures above show *why* those
    modifications work — they physically block the enzyme's active site from recognizing the peptide.
    """)
