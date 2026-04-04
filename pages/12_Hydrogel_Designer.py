import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Hydrogel Designer", page_icon="🧪", layout="wide")
st.title("PEG-Peptide Hydrogel Designer")
st.markdown(
    "Design cell-responsive PEG hydrogels with peptide adhesion ligands and crosslinkers. "
    "Based on Pashuck Lab methodologies (Lehigh University, 2020-2026)."
)

# ============================================================
# KNOWLEDGE BASE — from Pashuck Lab publications
# ============================================================

ADHESION_LIGANDS = {
    "GRGDS (linear RGD)": {
        "sequence": "GRGDS",
        "target": "αvβ3 and α5β1 integrins",
        "spreading_um2": 746,  # Ac-βA protected, from Paper 1
        "best_protection": "Ac-βA (N) + C-βA (C)",
        "notes": "Gold standard. Protected version (Ac-βA) matches cyclic RGD performance.",
        "source": "Rozans 2024",
    },
    "cyclo-GRGDSK (cyclic RGD)": {
        "sequence": "cGRGDSK",
        "target": "αvβ3 integrin (selective)",
        "spreading_um2": 769,
        "best_protection": "Cyclic — inherently protease resistant",
        "notes": "Best cell spreading but expensive and harder to synthesize.",
        "source": "Rozans 2024",
    },
    "IKVAV (laminin mimetic)": {
        "sequence": "IKVAV",
        "target": "α6β1 integrin / syndecan",
        "spreading_um2": None,
        "best_protection": "Ac-βA (N) + C-βA (C)",
        "notes": "Neural tissue applications. Promotes neurite outgrowth. Test as IVKVA variant.",
        "source": "Pashuck 2012 (Stupp lab)",
    },
    "Multiple RGD presentation": {
        "sequence": "Multi-arm PEG-GRGDS",
        "target": "αvβ3 / α5β1 (clustered)",
        "spreading_um2": None,
        "best_protection": "Ac-βA each arm",
        "notes": "Moghaddam 2025 (Acta Biomaterialia): multiple RGDs per junction increase spreading. "
                 "4-arm PEG with 2+ RGDs per arm outperforms single RGD.",
        "source": "Moghaddam 2025",
    },
}

CROSSLINKERS = {
    "GPQGIWGQ (PanMMP)": {
        "sequence": "GPQGIWGQ",
        "target_protease": "All MMPs (broad spectrum)",
        "degradation_mode": "Bulk + pericellular",
        "gel_stability": "Low — G' increases only 81% over 14 days",
        "d0_stiffness": 78,
        "d14_stiffness": 141,
        "notes": "Standard crosslinker. Cleaved by secreted MMPs → bulk gel degradation.",
        "source": "Wu/Rozans 2025",
    },
    "KLVADLMASAE (MMP-14 optimized)": {
        "sequence": "KLVADLMASAE",
        "target_protease": "MMP-14 (MT1-MMP, membrane-bound)",
        "degradation_mode": "Pericellular only",
        "gel_stability": "High — G' increases 181% over 14 days",
        "d0_stiffness": 89,
        "d14_stiffness": 250,
        "notes": "Optimized lead from split-and-pool screen. Pericellular degradation preserves bulk gel.",
        "source": "Wu/Rozans 2025",
    },
    "KLVADNYASAE (initial hit)": {
        "sequence": "KLVADNYASAE",
        "target_protease": "MMP-14",
        "degradation_mode": "Pericellular",
        "gel_stability": "Moderate",
        "d0_stiffness": None,
        "d14_stiffness": None,
        "notes": "Identified via MMP-14 proteomic screen. Replaced by LM variant.",
        "source": "Wu/Rozans 2025",
    },
    "Non-degradable (KLVADPPASAE)": {
        "sequence": "KLVADPPASAE",
        "target_protease": "None (resistant)",
        "degradation_mode": "No degradation",
        "gel_stability": "Maximum",
        "d0_stiffness": None,
        "d14_stiffness": None,
        "notes": "Proline-proline at cleavage site blocks all MMPs. Use as negative control.",
        "source": "Predicted",
    },
}

# Pashuck Lab 2026 JACS finding: sequence controls gel mechanics
SELF_ASSEMBLY_MECHANICS = {
    "Valine-rich": {"stiffness_effect": "Increases G'", "mechanism": "Strong H-bonding along fiber axis",
                    "source": "Moghaddam 2026 JACS"},
    "Alanine-rich": {"stiffness_effect": "Decreases G'", "mechanism": "Weaker H-bonding, more flexible fibers",
                     "source": "Moghaddam 2026 JACS"},
    "Alternating V/A": {"stiffness_effect": "Tunable", "mechanism": "Intermediate H-bond strength",
                        "source": "Moghaddam 2026 JACS"},
}

PEG_OPTIONS = {
    "4-arm PEG-Mal (10 kDa)": {"arms": 4, "mw": 10000, "chemistry": "Maleimide-thiol", "stiffness_range": "50-500 Pa"},
    "4-arm PEG-Mal (20 kDa)": {"arms": 4, "mw": 20000, "chemistry": "Maleimide-thiol", "stiffness_range": "100-2000 Pa"},
    "8-arm PEG-Mal (20 kDa)": {"arms": 8, "mw": 20000, "chemistry": "Maleimide-thiol", "stiffness_range": "500-5000 Pa"},
    "4-arm PEG-Norbornene (20 kDa)": {"arms": 4, "mw": 20000, "chemistry": "Thiol-ene (UV)", "stiffness_range": "200-3000 Pa"},
    "PEG-diacrylate (6 kDa)": {"arms": 2, "mw": 6000, "chemistry": "Radical (UV)", "stiffness_range": "1000-50000 Pa"},
}

# ============================================================
# UI
# ============================================================

tab1, tab2, tab3 = st.tabs(["Design a Gel", "Component Library", "Pashuck Lab Methods"])

with tab1:
    st.markdown("### Configure your hydrogel")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### PEG Backbone")
        peg_choice = st.selectbox("PEG type", list(PEG_OPTIONS.keys()))
        peg = PEG_OPTIONS[peg_choice]
        st.markdown(f"**Arms:** {peg['arms']} | **MW:** {peg['mw']/1000:.0f} kDa")
        st.markdown(f"**Chemistry:** {peg['chemistry']}")
        st.markdown(f"**Stiffness range:** {peg['stiffness_range']}")

        peg_conc = st.slider("PEG concentration (wt%)", 1.0, 15.0, 5.0, 0.5)

    with col2:
        st.markdown("#### Adhesion Ligand")
        ligand_choice = st.selectbox("Adhesion ligand", list(ADHESION_LIGANDS.keys()))
        ligand = ADHESION_LIGANDS[ligand_choice]
        st.markdown(f"**Sequence:** `{ligand['sequence']}`")
        st.markdown(f"**Target:** {ligand['target']}")
        if ligand["spreading_um2"]:
            st.markdown(f"**Cell spreading:** {ligand['spreading_um2']} um2 (Day 7, hMSC)")
        st.markdown(f"**Best protection:** {ligand['best_protection']}")

        ligand_conc = st.slider("Ligand concentration (mM)", 0.1, 5.0, 1.0, 0.1)

    with col3:
        st.markdown("#### Crosslinker")
        xlink_choice = st.selectbox("Crosslinker", list(CROSSLINKERS.keys()))
        xlink = CROSSLINKERS[xlink_choice]
        st.markdown(f"**Sequence:** `{xlink['sequence']}`")
        st.markdown(f"**Target protease:** {xlink['target_protease']}")
        st.markdown(f"**Degradation mode:** {xlink['degradation_mode']}")
        st.markdown(f"**Gel stability:** {xlink['gel_stability']}")

    st.markdown("---")
    st.markdown("### Predicted Gel Properties")

    col_a, col_b, col_c, col_d = st.columns(4)

    # Rough stiffness estimate based on PEG concentration
    base_stiffness = peg_conc * 20 * (peg["arms"] / 4)
    with col_a:
        st.metric("Est. Initial G' (Pa)", f"{base_stiffness:.0f}")
    with col_b:
        if xlink.get("d14_stiffness") and xlink.get("d0_stiffness"):
            ratio = xlink["d14_stiffness"] / xlink["d0_stiffness"]
            d14_est = base_stiffness * ratio
            st.metric("Est. Day 14 G' (Pa)", f"{d14_est:.0f}")
        else:
            st.metric("Est. Day 14 G' (Pa)", "N/A")
    with col_c:
        if ligand["spreading_um2"]:
            st.metric("Expected Cell Spreading", f"~{ligand['spreading_um2']} um2")
        else:
            st.metric("Expected Cell Spreading", "No data")
    with col_d:
        st.metric("Degradation Mode", xlink["degradation_mode"])

    # Recommendations
    st.markdown("### Design Recommendations")
    recs = []
    if "NH2" in ligand.get("best_protection", ""):
        recs.append("Consider Ac-βA terminal protection for your adhesion ligand — free amine loses 80% at 48h")
    if "PanMMP" in xlink_choice:
        recs.append("PanMMP crosslinker: expect bulk gel softening. Consider KLVADLMASAE for better gel stability")
    if peg_conc < 3:
        recs.append("Low PEG concentration — gel may be very soft. Consider increasing to 4-5 wt%")
    if peg_conc > 10:
        recs.append("High PEG concentration — may limit cell spreading. Consider 4-6 wt% for 3D culture")
    if ligand_conc < 0.5:
        recs.append("Low ligand concentration — cells may not spread. Use >= 1 mM for hMSC spreading")
    if ligand_conc > 3:
        recs.append("Very high ligand concentration — diminishing returns above 2 mM. Watch for peptide degradation at lower effective concentrations")

    if recs:
        for r in recs:
            st.markdown(f"- {r}")
    else:
        st.success("Configuration looks reasonable based on published data.")

with tab2:
    st.markdown("### Adhesion Ligands")
    lig_df = pd.DataFrame([
        {"Ligand": k, "Sequence": v["sequence"], "Target": v["target"],
         "Spreading (um2)": v["spreading_um2"] or "N/A",
         "Protection": v["best_protection"], "Source": v["source"]}
        for k, v in ADHESION_LIGANDS.items()
    ])
    st.dataframe(lig_df, hide_index=True)

    st.markdown("### Crosslinkers")
    xlink_df = pd.DataFrame([
        {"Crosslinker": k, "Sequence": v["sequence"],
         "Target Protease": v["target_protease"],
         "Degradation": v["degradation_mode"],
         "G' Day 0": v.get("d0_stiffness", "N/A"),
         "G' Day 14": v.get("d14_stiffness", "N/A"),
         "Source": v["source"]}
        for k, v in CROSSLINKERS.items()
    ])
    st.dataframe(xlink_df, hide_index=True)

    if any(v.get("d0_stiffness") and v.get("d14_stiffness") for v in CROSSLINKERS.values()):
        st.markdown("### Gel Stiffness Comparison")
        gel_data = []
        for k, v in CROSSLINKERS.items():
            if v.get("d0_stiffness") and v.get("d14_stiffness"):
                gel_data.append({"Crosslinker": k, "Day": 0, "G' (Pa)": v["d0_stiffness"]})
                gel_data.append({"Crosslinker": k, "Day": 14, "G' (Pa)": v["d14_stiffness"]})
        gel_df = pd.DataFrame(gel_data)
        fig = px.bar(gel_df, x="Crosslinker", y="G' (Pa)", color="Day", barmode="group",
                     color_discrete_map={0: "#0E4D92", 14: "#E8792B"})
        fig.update_layout(height=350, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Self-Assembly Mechanics (JACS 2026)")
    st.markdown(
        "Moghaddam et al. showed that amino acid sequence controls hydrogel stiffness via "
        "hydrogen bond alignment along self-assembled peptide fibers."
    )
    sa_df = pd.DataFrame([
        {"Sequence Type": k, "Effect": v["stiffness_effect"], "Mechanism": v["mechanism"], "Source": v["source"]}
        for k, v in SELF_ASSEMBLY_MECHANICS.items()
    ])
    st.dataframe(sa_df, hide_index=True)

with tab3:
    st.markdown("### Pashuck Lab Experimental Methods")

    st.markdown("""
    #### Hydrogel Fabrication (from published protocols)

    1. **PEG-maleimide gels:**
       - Dissolve 4-arm PEG-Mal in PBS at desired wt%
       - Add adhesion peptide (Cys-terminated) at 1 mM
       - Add crosslinker peptide (bis-Cys) at stoichiometric ratio
       - Mix quickly, pipette into mold, gel in 5-10 min at 37C

    2. **Cell encapsulation:**
       - Trypsinize cells, resuspend at 5-10M cells/mL in gel precursor
       - Mix with crosslinker to initiate gelation
       - Culture in growth medium, change every 2-3 days

    #### LC-MS Degradation Assay (Rozans 2025)

    - **Setup:** 75,000 cells/well, 24-well plates, PEG gels
    - **Timepoints:** 0, 1, 4, 8, 24, 48 hours
    - **Sampling:** Collect conditioned medium, add internal standard
    - **LC-MS:** <5 min run time per sample, C18 column
    - **Quantification:** AUC ratio to internal standard (NH2-βF-(βA)6-amide)
    - **Replicates:** 3 biological x 3 technical = 9 per condition

    #### Rheology (Wu/Rozans 2025)

    - AR-G2 rheometer, 20mm parallel plate
    - 1% strain, 1 Hz frequency sweep
    - Measure G' and G'' at Day 0, 7, 14
    - Temperature: 37C

    #### Cell Spreading (Rozans 2024)

    - Phalloidin-DAPI staining at Day 7
    - Confocal microscopy, maximum intensity projections
    - ImageJ quantification of projected cell area
    - n >= 50 cells per condition
    """)

    st.markdown("---")
    st.markdown("### Complete Pashuck Lab Publication List")
    st.markdown("""
    | Year | Key Paper | Topic |
    |------|-----------|-------|
    | 2026 | Moghaddam — JACS | Sequence-controlled gel mechanics via H-bonding |
    | 2025 | Wu/Rozans — Adv Healthcare Mater | MMP-14 crosslinker optimization (LM lead) |
    | 2025 | Rozans — J Biomed Mater Res A | High-throughput LC-MS degradation assay |
    | 2025 | Moghaddam — Acta Biomaterialia | Multiple RGD presentations increase spreading |
    | 2025 | Jensen — Biomater Sci | 4D bioprinting composite hydrogels |
    | 2024 | Rozans — ACS Biomater Sci Eng | Terminal chemistry controls peptide degradation |
    | 2023 | Fung — J Mater Chem B | Poly(peptide-ester) surface resorption |
    | 2021 | Webber/Pashuck — Adv Drug Deliv Rev | Self-assembly for drug delivery (review) |
    | 2021 | Lin — ACS Nano | High-throughput peptide derivatization |
    """)
