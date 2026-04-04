import streamlit as st
from Bio.Seq import Seq
from Bio.SeqUtils import molecular_weight
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="Sequence Tools", page_icon="🧮", layout="wide")
st.title("Protein Sequence Tools")
st.markdown("Analyze protein sequences: molecular weight, composition, hydrophobicity, pI, and pairwise alignment.")

tab1, tab2, tab3 = st.tabs(["Sequence Analysis", "Pairwise Alignment", "Batch MW Calculator"])

# ---- Tab 1: Single Sequence Analysis ----
with tab1:
    seq_input = st.text_area(
        "Protein sequence (one-letter code, no spaces/headers)",
        value="MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVL",
        height=100,
    )

    if st.button("Analyze", type="primary", key="analyze"):
        seq_clean = "".join(seq_input.upper().split())
        seq_clean = seq_clean.replace("*", "")

        # Validate
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        invalid = set(seq_clean) - valid_aa
        if invalid:
            st.error(f"Invalid amino acid characters: {invalid}")
            st.stop()

        analysis = ProteinAnalysis(seq_clean)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            mw = molecular_weight(seq_clean, "protein")
            st.metric("Molecular Weight", f"{mw:.1f} Da")
        with col2:
            st.metric("Length", f"{len(seq_clean)} aa")
        with col3:
            pi = analysis.isoelectric_point()
            st.metric("Isoelectric Point (pI)", f"{pi:.2f}")
        with col4:
            gravy = analysis.gravy()
            label = "hydrophobic" if gravy > 0 else "hydrophilic"
            st.metric("GRAVY", f"{gravy:.3f} ({label})")

        st.markdown("---")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("### Amino Acid Composition")
            aa_comp = analysis.get_amino_acids_percent()
            aa_df = pd.DataFrame([
                {"Amino Acid": aa, "Fraction": f"{frac:.3f}", "Count": seq_clean.count(aa)}
                for aa, frac in sorted(aa_comp.items(), key=lambda x: -x[1])
            ])
            st.dataframe(aa_df, hide_index=True, height=400)

        with col_b:
            st.markdown("### Composition Chart")
            sorted_aa = sorted(aa_comp.items(), key=lambda x: -x[1])
            fig = go.Figure(go.Bar(
                x=[a[0] for a in sorted_aa],
                y=[a[1] * 100 for a in sorted_aa],
                marker_color="#0E4D92",
            ))
            fig.update_layout(
                xaxis_title="Amino Acid",
                yaxis_title="Frequency (%)",
                height=400,
                margin=dict(t=20),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Hydrophobicity profile (Kyte-Doolittle)
        st.markdown("### Hydrophobicity Profile (Kyte-Doolittle, window=7)")
        kd_scale = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
        }
        window = 7
        half_w = window // 2
        hydro = []
        for i in range(half_w, len(seq_clean) - half_w):
            window_seq = seq_clean[i - half_w:i + half_w + 1]
            score = sum(kd_scale.get(aa, 0) for aa in window_seq) / window
            hydro.append(score)

        if hydro:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=list(range(half_w + 1, len(seq_clean) - half_w + 1)),
                y=hydro,
                mode="lines",
                line=dict(color="#0E4D92", width=1.5),
            ))
            fig2.add_hline(y=0, line_dash="dash", line_color="gray")
            fig2.update_layout(
                xaxis_title="Residue Position",
                yaxis_title="Hydrophobicity Score",
                height=300,
                margin=dict(t=20),
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Secondary structure propensity
        st.markdown("### Secondary Structure Fraction (Chou-Fasman)")
        ss = analysis.secondary_structure_fraction()
        ss_df = pd.DataFrame([
            {"Structure": "Helix", "Fraction": f"{ss[0]:.3f}"},
            {"Structure": "Turn", "Fraction": f"{ss[1]:.3f}"},
            {"Structure": "Sheet", "Fraction": f"{ss[2]:.3f}"},
        ])
        st.dataframe(ss_df, hide_index=True)

# ---- Tab 2: Pairwise Alignment ----
with tab2:
    st.markdown("Global pairwise alignment of two protein sequences.")
    col1, col2 = st.columns(2)
    with col1:
        seq1_input = st.text_area("Sequence 1", value="MKTVRQERLKSIV", height=100, key="seq1")
    with col2:
        seq2_input = st.text_area("Sequence 2", value="MKTVRQGRLKSIV", height=100, key="seq2")

    if st.button("Align", type="primary", key="align"):
        s1 = "".join(seq1_input.upper().split())
        s2 = "".join(seq2_input.upper().split())

        alignments = pairwise2.align.globalxx(s1, s2, one_alignment_only=True)
        if alignments:
            aln = alignments[0]
            st.markdown(f"**Score:** {aln.score}")
            st.code(format_alignment(*aln))

            # Identity calculation
            matches = sum(1 for a, b in zip(aln.seqA, aln.seqB) if a == b and a != "-")
            aligned_len = max(len(aln.seqA), len(aln.seqB))
            identity = 100 * matches / aligned_len if aligned_len else 0
            st.metric("Sequence Identity", f"{identity:.1f}%")
        else:
            st.warning("No alignment found.")

# ---- Tab 3: Batch MW Calculator ----
with tab3:
    st.markdown("Paste multiple sequences (one per line) to calculate molecular weights.")
    batch_input = st.text_area(
        "Sequences (one per line)",
        value="GRGDS\nIVKVA\nLIAANK\nGPQGIWGQ\nKLVADLMASAE",
        height=200,
        key="batch",
    )

    if st.button("Calculate", type="primary", key="batch_calc"):
        lines = [l.strip().upper() for l in batch_input.strip().split("\n") if l.strip()]
        results = []
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        for seq in lines:
            invalid = set(seq) - valid_aa
            if invalid:
                results.append({"Sequence": seq, "MW (Da)": "INVALID", "Length": len(seq), "pI": ""})
            else:
                mw = molecular_weight(seq, "protein")
                analysis = ProteinAnalysis(seq)
                pi = analysis.isoelectric_point()
                results.append({
                    "Sequence": seq,
                    "MW (Da)": f"{mw:.1f}",
                    "Length": len(seq),
                    "pI": f"{pi:.2f}",
                })

        st.dataframe(pd.DataFrame(results), hide_index=True)
