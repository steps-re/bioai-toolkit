import streamlit as st
from Bio.Blast import NCBIWWW, NCBIXML
from Bio import Entrez
import pandas as pd
import time

st.set_page_config(page_title="BLAST Search", page_icon="🎯", layout="wide")
st.title("NCBI BLAST Search")
st.markdown("Run remote protein BLAST (blastp) against NCBI databases. Results typically take 30-120 seconds.")

Entrez.email = "mike@stepsventures.com"

# Input
seq_input = st.text_area(
    "Protein sequence (amino acids only)",
    value="MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVL",
    height=100,
)

col1, col2, col3 = st.columns(3)
with col1:
    database = st.selectbox("Database", ["nr", "swissprot", "pdb", "refseq_protein"], index=1)
with col2:
    max_hits = st.selectbox("Max hits", [5, 10, 25, 50], index=1)
with col3:
    evalue = st.selectbox("E-value threshold", [0.001, 0.01, 0.1, 1.0, 10.0], index=2)

st.warning("BLAST queries run on NCBI servers and can take 30-120+ seconds. Please be patient.")

if st.button("Run BLAST", type="primary"):
    seq_clean = "".join(seq_input.upper().split())

    valid_aa = set("ACDEFGHIKLMNPQRSTVWYX*")
    invalid = set(seq_clean) - valid_aa
    if invalid:
        st.error(f"Invalid characters: {invalid}")
        st.stop()

    progress = st.empty()
    progress.info("Submitting BLAST query to NCBI... this may take 30-120 seconds.")

    start = time.time()
    try:
        result_handle = NCBIWWW.qblast(
            "blastp",
            database,
            seq_clean,
            expect=evalue,
            hitlist_size=max_hits,
        )
        elapsed = time.time() - start
        progress.success(f"BLAST completed in {elapsed:.0f} seconds.")
    except Exception as e:
        progress.error(f"BLAST failed: {e}")
        st.stop()

    blast_records = NCBIXML.parse(result_handle)
    record = next(blast_records)

    if not record.alignments:
        st.warning("No hits found. Try a longer sequence or less restrictive E-value.")
        st.stop()

    st.markdown(f"### {len(record.alignments)} hits found")

    results = []
    for i, alignment in enumerate(record.alignments):
        hsp = alignment.hsps[0]
        title = alignment.title[:120]
        identity = 100 * hsp.identities / hsp.align_length if hsp.align_length else 0
        coverage = 100 * hsp.align_length / len(seq_clean) if len(seq_clean) else 0

        results.append({
            "#": i + 1,
            "Hit": title,
            "Score": hsp.score,
            "E-value": f"{hsp.expect:.2e}",
            "Identity": f"{identity:.1f}%",
            "Coverage": f"{coverage:.1f}%",
            "Gaps": hsp.gaps,
            "Length": alignment.length,
        })

    st.dataframe(pd.DataFrame(results), hide_index=True)

    # Show alignments in expandable sections
    for i, alignment in enumerate(record.alignments):
        hsp = alignment.hsps[0]
        with st.expander(f"Hit {i+1}: {alignment.title[:80]}"):
            st.code(
                f"Score: {hsp.score}, E-value: {hsp.expect:.2e}\n"
                f"Identities: {hsp.identities}/{hsp.align_length}, "
                f"Positives: {hsp.positives}/{hsp.align_length}, "
                f"Gaps: {hsp.gaps}/{hsp.align_length}\n\n"
                f"Query:   {hsp.query[:80]}...\n"
                f"         {hsp.match[:80]}...\n"
                f"Subject: {hsp.sbjct[:80]}...",
                language=None,
            )
