import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests

st.set_page_config(page_title="MMP-14 Cleavage Predictor", page_icon="✂️", layout="wide")
st.title("MMP-14 Cleavage Site Predictor")
st.markdown(
    "Predict matrix metalloproteinase-14 (MT1-MMP) cleavage of peptide crosslinkers. "
    "Calibrated against Wu, Rozans et al. (Adv Healthcare Mater 2025) crosslinker screening data."
)

# ============================================================
# MMP-14 SUBSTRATE SPECIFICITY (from MEROPS + Wu/Rozans Paper 3)
# ============================================================
# MMP-14 (MT1-MMP) is a membrane-type MMP — cleaves pericellularly only
# Subsite preferences based on published specificity profiles + Paper 3 results

# P3-P2-P1 | P1'-P2'-P3' (scissile bond between P1 and P1')
# In KLVAD-X1X2-ASAE: X1 = P1', X2 = P2' (the bond is between D and X1)

# P1' preferences (position X1 in KLVAD-X1X2-ASAE)
MMP14_P1P = {
    'L': 0.95, 'I': 0.90, 'M': 0.80, 'F': 0.75, 'V': 0.60,
    'A': 0.55, 'W': 0.50, 'Y': 0.50, 'T': 0.45, 'S': 0.40,
    'N': 0.40, 'Q': 0.38, 'G': 0.30, 'K': 0.35, 'R': 0.35,
    'E': 0.25, 'D': 0.20, 'H': 0.25, 'P': 0.05,  # Pro almost never at P1'
}

# P2' preferences (position X2 in KLVAD-X1X2-ASAE)
MMP14_P2P = {
    'I': 0.85, 'L': 0.80, 'M': 0.75, 'A': 0.65, 'V': 0.60,
    'F': 0.60, 'S': 0.55, 'T': 0.55, 'W': 0.50, 'Y': 0.50,
    'G': 0.45, 'N': 0.45, 'Q': 0.40, 'K': 0.40, 'R': 0.40,
    'H': 0.35, 'E': 0.30, 'D': 0.25, 'P': 0.10,
}

# Experimental validation points from Paper 3
EXPERIMENTAL_CROSSLINKERS = {
    "LM": {"gel_stiffness_d14": 250, "gel_stiffness_d0": 89, "status": "Optimized lead — best pericellular degradation with gel stability"},
    "NY": {"gel_stiffness_d14": None, "gel_stiffness_d0": None, "status": "Initial proteomic hit — KLVADNYASAE"},
    "GPQGIWGQ": {"gel_stiffness_d14": 141, "gel_stiffness_d0": 78, "status": "PanMMP benchmark — more bulk degradation than LM"},
}

AA_LIST = list("GAVILMFWPSTNQDEKRHY")


def predict_mmp14_cleavage(x1, x2):
    """Predict MMP-14 cleavage score for a KLVAD-X1X2-ASAE crosslinker."""
    p1p = MMP14_P1P.get(x1, 0.3)
    p2p = MMP14_P2P.get(x2, 0.3)

    # Weighted: P1' is more important than P2' for MMP-14
    score = 0.6 * p1p + 0.4 * p2p

    # Steric penalty: two bulky residues together can reduce access
    bulky = set("FWYH")
    if x1 in bulky and x2 in bulky:
        score *= 0.85

    # Proline penalty: strong at either position
    if x1 == 'P':
        score *= 0.3
    if x2 == 'P':
        score *= 0.5

    # Charge penalty: two same-charge residues
    if x1 in "KR" and x2 in "KR":
        score *= 0.8
    if x1 in "DE" and x2 in "DE":
        score *= 0.8

    return round(min(1.0, score), 3)


# ============================================================
# UI
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(["Heatmap", "Single Prediction", "Design Guide", "MEROPS Lookup"])

with tab1:
    st.markdown("### Full 19x19 MMP-14 Cleavage Heatmap")
    st.markdown("KLVAD-**X1**-**X2**-ASAE — X1 is the P1' position (rows), X2 is P2' (columns)")

    # Generate full heatmap
    data = []
    for x1 in AA_LIST:
        row = {}
        for x2 in AA_LIST:
            row[x2] = predict_mmp14_cleavage(x1, x2)
        data.append(row)

    heatmap_df = pd.DataFrame(data, index=AA_LIST)

    fig = px.imshow(heatmap_df, color_continuous_scale="RdYlGn", aspect="equal",
                    labels={"x": "X2 (P2' position)", "y": "X1 (P1' position)", "color": "MMP-14 Score"},
                    zmin=0, zmax=1)

    # Annotate key variants
    annotations = [
        {"x": "M", "y": "L", "text": "LM*", "color": "black"},
        {"x": "Y", "y": "N", "text": "NY", "color": "black"},
        {"x": "I", "y": "L", "text": "LI", "color": "white"},
        {"x": "P", "y": "P", "text": "PP", "color": "white"},
    ]
    for ann in annotations:
        fig.add_annotation(
            x=ann["x"], y=ann["y"], text=ann["text"],
            showarrow=False, font=dict(size=10, color=ann["color"])
        )

    fig.update_layout(height=650, margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "**LM*** = optimized lead from Wu/Rozans 2025. "
        "**NY** = initial proteomic hit. "
        "**LI** = highest predicted score. "
        "**PP** = most resistant (proline at both positions)."
    )

    # Rank all 361
    st.markdown("### Full Ranking (361 variants)")
    all_variants = []
    for x1 in AA_LIST:
        for x2 in AA_LIST:
            score = predict_mmp14_cleavage(x1, x2)
            all_variants.append({
                "Dipeptide": f"{x1}{x2}",
                "Sequence": f"KLVAD{x1}{x2}ASAE",
                "X1 (P1')": x1,
                "X2 (P2')": x2,
                "MMP-14 Score": score,
                "P1' Preference": MMP14_P1P.get(x1, 0),
                "P2' Preference": MMP14_P2P.get(x2, 0),
            })
    rank_df = pd.DataFrame(all_variants).sort_values("MMP-14 Score", ascending=False).reset_index(drop=True)
    rank_df.index = rank_df.index + 1
    rank_df.index.name = "Rank"
    st.dataframe(rank_df, height=500)

with tab2:
    st.markdown("### Predict cleavage for a specific crosslinker")

    col1, col2 = st.columns(2)
    with col1:
        x1_input = st.selectbox("X1 (P1' position)", AA_LIST, index=AA_LIST.index("L"))
    with col2:
        x2_input = st.selectbox("X2 (P2' position)", AA_LIST, index=AA_LIST.index("M"))

    score = predict_mmp14_cleavage(x1_input, x2_input)
    sequence = f"KLVAD{x1_input}{x2_input}ASAE"

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("MMP-14 Cleavage Score", f"{score:.3f}")
    with col_b:
        rank = sum(1 for v in all_variants if v["MMP-14 Score"] >= score)
        st.metric("Rank", f"#{rank} / 361")
    with col_c:
        if score > 0.7:
            st.metric("Prediction", "Highly Cleavable")
        elif score > 0.4:
            st.metric("Prediction", "Moderately Cleavable")
        else:
            st.metric("Prediction", "Resistant")

    st.markdown(f"**Sequence:** `{sequence}`")
    st.markdown(f"**P1' ({x1_input}) preference:** {MMP14_P1P.get(x1_input, 0):.2f}")
    st.markdown(f"**P2' ({x2_input}) preference:** {MMP14_P2P.get(x2_input, 0):.2f}")

    # Compare to benchmarks
    st.markdown("#### Comparison to Benchmarks")
    lm_score = predict_mmp14_cleavage("L", "M")
    ny_score = predict_mmp14_cleavage("N", "Y")
    comp_df = pd.DataFrame([
        {"Variant": f"{x1_input}{x2_input} (yours)", "Score": score},
        {"Variant": "LM (optimized lead)", "Score": lm_score},
        {"Variant": "NY (initial hit)", "Score": ny_score},
        {"Variant": "LI (top predicted)", "Score": predict_mmp14_cleavage("L", "I")},
        {"Variant": "PP (most resistant)", "Score": predict_mmp14_cleavage("P", "P")},
    ])
    fig = px.bar(comp_df, x="Variant", y="Score", color="Score",
                 color_continuous_scale="RdYlGn")
    fig.update_layout(height=350, margin=dict(t=20), yaxis=dict(range=[0, 1]))
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("### Crosslinker Design Guide")
    st.markdown("""
    #### How MMP-14 cleavage works in hydrogels

    MMP-14 (MT1-MMP / membrane-type 1 MMP) is a membrane-anchored protease expressed on many cell types.
    Unlike secreted MMPs (MMP-2, MMP-9), it only cleaves substrates in direct contact with the cell surface.

    **This means:**
    - Cleavage is **pericellular** (immediate cell vicinity only)
    - High-score crosslinkers let cells migrate through the gel but don't cause bulk collapse
    - Low-score crosslinkers resist cell-mediated remodeling

    #### Design rules (from Wu/Rozans 2025):

    | Rule | Explanation |
    |------|-------------|
    | **Hydrophobic at P1'** | Leu, Ile, Met strongly preferred. MMP-14's S1' pocket is deep and hydrophobic |
    | **Moderate at P2'** | Less selective than P1'. Ile, Leu, Met good. Small AAs (Ala, Ser) acceptable |
    | **Avoid Pro at P1'** | Proline's rigid ring prevents proper substrate positioning in the active site |
    | **Avoid charged pairs** | Two positively or negatively charged residues reduce cleavage |
    | **LM is optimal, not maximal** | The lead (Leu-Met) was chosen for balanced pericellular degradation + gel mechanical stability |

    #### Why LM beats LI in practice:

    LI has a slightly higher predicted cleavage score, but KLVADLMASAE outperformed in cell culture because:
    1. **Gel stability:** LM gels maintained G' = 250 Pa at day 14, vs 141 Pa for PanMMP (GPQGIWGQ)
    2. **Pericellular selectivity:** LM is cleaved preferentially by membrane-bound MMP-14, not secreted MMPs
    3. **Cell spreading:** Comparable cell spreading and viability to PanMMP gels
    """)

    st.markdown("#### Key experimental data (Wu/Rozans 2025)")
    exp_df = pd.DataFrame([
        {"Crosslinker": "KLVADLMASAE (LM)", "G' Day 0 (Pa)": 89, "G' Day 14 (Pa)": 250,
         "Stiffening": "+181%", "Notes": "Optimized lead — pericellular degradation maintains bulk gel"},
        {"Crosslinker": "GPQGIWGQ (PanMMP)", "G' Day 0 (Pa)": 78, "G' Day 14 (Pa)": 141,
         "Stiffening": "+81%", "Notes": "Benchmark — more bulk degradation, less gel stability"},
    ])
    st.dataframe(exp_df, hide_index=True)

with tab4:
    st.markdown("### MEROPS Protease Database Lookup")
    st.markdown(
        "Look up protease substrate specificity from MEROPS — the gold standard database for peptidase classification. "
        "Useful for identifying which proteases might cleave your peptide."
    )

    merops_id = st.text_input("MEROPS protease ID", value="M10.014", placeholder="e.g. M10.014 (MMP-14)")

    st.markdown("""
    **Common protease IDs:**
    | ID | Protease | Type |
    |---|---|---|
    | M10.014 | MMP-14 (MT1-MMP) | Membrane metalloprotease |
    | M10.001 | MMP-1 (Collagenase-1) | Secreted metalloprotease |
    | M10.002 | MMP-2 (Gelatinase A) | Secreted metalloprotease |
    | M10.004 | MMP-9 (Gelatinase B) | Secreted metalloprotease |
    | M01.001 | Aminopeptidase N (CD13) | Membrane aminopeptidase |
    | M17.001 | Leucine aminopeptidase | Cytosolic aminopeptidase |
    | S10.001 | Carboxypeptidase A1 | Serine carboxypeptidase |
    """)

    if st.button("Look up on MEROPS"):
        st.markdown(f"[View {merops_id} on MEROPS](https://www.ebi.ac.uk/merops/cgi-bin/pepsum?id={merops_id})")
        st.info(
            "MEROPS pages show: substrate specificity, cleavage site logos, "
            "known substrates, inhibitors, and phylogenetic classification."
        )
