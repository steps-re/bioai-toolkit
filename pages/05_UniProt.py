import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="UniProt Search", page_icon="🧫", layout="wide")
st.title("UniProt Protein Search")
st.markdown("Search the UniProt knowledge base for protein function, domains, and annotations.")

API_BASE = "https://rest.uniprot.org"


def search_uniprot(query, organism="", limit=10):
    """Search UniProt with optional organism filter."""
    q = query
    if organism:
        q += f" AND organism_name:{organism}"
    params = {
        "query": q,
        "format": "json",
        "fields": "accession,protein_name,gene_names,organism_name,length,cc_function,ft_domain,go_p,go_f,go_c",
        "size": limit,
    }
    resp = requests.get(f"{API_BASE}/uniprotkb/search", params=params, timeout=15)
    resp.raise_for_status()
    return resp.json().get("results", [])


def get_entry(accession):
    """Get full UniProt entry."""
    resp = requests.get(f"{API_BASE}/uniprotkb/{accession}.json", timeout=15)
    resp.raise_for_status()
    return resp.json()


# Search interface
col1, col2, col3 = st.columns([3, 2, 1])
with col1:
    query = st.text_input("Search query", value="", placeholder="e.g. EGFR, insulin receptor, kinase...")
with col2:
    organism = st.text_input("Organism filter (optional)", placeholder="e.g. Human, Homo sapiens")
with col3:
    limit = st.selectbox("Max results", [5, 10, 25, 50], index=1)

if st.button("Search", type="primary") and query:
    with st.spinner("Searching UniProt..."):
        try:
            results = search_uniprot(query, organism, limit)
        except requests.RequestException as e:
            st.error(f"API error: {e}")
            st.stop()

    if not results:
        st.warning("No results found.")
        st.stop()

    st.markdown(f"### {len(results)} results")

    for entry in results:
        accession = entry.get("primaryAccession", "")
        protein_desc = entry.get("proteinDescription", {})
        rec_name = protein_desc.get("recommendedName", {})
        full_name = rec_name.get("fullName", {}).get("value", "") if rec_name else ""
        if not full_name:
            sub_names = protein_desc.get("submissionNames", [])
            if sub_names:
                full_name = sub_names[0].get("fullName", {}).get("value", "Unknown")

        genes = entry.get("genes", [])
        gene_name = genes[0].get("geneName", {}).get("value", "") if genes else ""
        org = entry.get("organism", {}).get("scientificName", "")
        length = entry.get("sequence", {}).get("length", "")

        with st.expander(f"**{accession}** — {full_name} ({gene_name}) | {org} | {length} aa"):
            # Function
            comments = entry.get("comments", [])
            functions = [c for c in comments if c.get("commentType") == "FUNCTION"]
            if functions:
                st.markdown("**Function:**")
                for f in functions:
                    for text in f.get("texts", []):
                        st.markdown(f"> {text.get('value', '')}")

            # Domains
            features = entry.get("features", [])
            domains = [f for f in features if f.get("type") == "Domain"]
            if domains:
                st.markdown("**Domains:**")
                domain_data = []
                for d in domains:
                    loc = d.get("location", {})
                    start = loc.get("start", {}).get("value", "")
                    end = loc.get("end", {}).get("value", "")
                    desc = d.get("description", "")
                    domain_data.append({"Domain": desc, "Start": start, "End": end})
                st.dataframe(pd.DataFrame(domain_data), hide_index=True)

            # GO terms
            go_terms = entry.get("uniProtKBCrossReferences", [])
            go_entries = [x for x in go_terms if x.get("database") == "GO"]
            if go_entries:
                st.markdown("**GO Annotations:**")
                go_data = []
                for g in go_entries:
                    props = {p["key"]: p["value"] for p in g.get("properties", [])}
                    go_data.append({
                        "GO ID": g.get("id", ""),
                        "Term": props.get("GoTerm", ""),
                        "Source": props.get("GoEvidenceType", ""),
                    })
                if go_data:
                    st.dataframe(pd.DataFrame(go_data), hide_index=True)

            # Links
            st.markdown(
                f"[View on UniProt](https://www.uniprot.org/uniprotkb/{accession}) | "
                f"[AlphaFold](https://alphafold.ebi.ac.uk/entry/{accession}) | "
                f"[PDB](https://www.rcsb.org/search?q=rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession:{accession})"
            )
