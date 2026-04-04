import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="PubChem Search", page_icon="💊", layout="wide")
st.title("PubChem Compound Search")
st.markdown("Search compounds by name, CID, or SMILES. View properties and 2D structures.")

PUG_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


def search_by_name(name):
    url = f"{PUG_BASE}/compound/name/{requests.utils.quote(name)}/property/IUPACName,MolecularFormula,MolecularWeight,CanonicalSMILES,IsomericSMILES,XLogP,TPSA,HBondDonorCount,HBondAcceptorCount,RotatableBondCount,ExactMass,Charge/JSON"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return resp.json()["PropertyTable"]["Properties"]


def search_by_cid(cid):
    url = f"{PUG_BASE}/compound/cid/{cid}/property/IUPACName,MolecularFormula,MolecularWeight,CanonicalSMILES,IsomericSMILES,XLogP,TPSA,HBondDonorCount,HBondAcceptorCount,RotatableBondCount,ExactMass,Charge/JSON"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return resp.json()["PropertyTable"]["Properties"]


def get_description(cid):
    url = f"{PUG_BASE}/compound/cid/{cid}/description/JSON"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        descs = resp.json().get("InformationList", {}).get("Information", [])
        for d in descs:
            if d.get("Description"):
                return d["Description"]
    except Exception:
        pass
    return None


def get_2d_image_url(cid):
    return f"{PUG_BASE}/compound/cid/{cid}/PNG?image_size=400x300"


# Search interface
search_type = st.radio("Search by", ["Compound name", "CID"], horizontal=True)

col1, col2 = st.columns([3, 1])
with col1:
    if search_type == "Compound name":
        query = st.text_input("Compound name", placeholder="e.g. aspirin, caffeine, glucose, penicillin")
    else:
        query = st.text_input("PubChem CID", placeholder="e.g. 2244, 2519")

# RDKit for SMILES rendering
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

if st.button("Search", type="primary") and query:
    with st.spinner("Searching PubChem..."):
        try:
            if search_type == "Compound name":
                results = search_by_name(query.strip())
            else:
                results = search_by_cid(query.strip())
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                st.error("Compound not found.")
            else:
                st.error(f"API error: {e}")
            st.stop()
        except requests.RequestException as e:
            st.error(f"API error: {e}")
            st.stop()

    for compound in results:
        cid = compound.get("CID", "")
        iupac = compound.get("IUPACName", "N/A")
        formula = compound.get("MolecularFormula", "")
        mw = compound.get("MolecularWeight", "")
        smiles = compound.get("CanonicalSMILES", "")

        st.markdown(f"---")
        st.markdown(f"### CID {cid} — {iupac}")

        col_a, col_b = st.columns([1, 2])

        with col_a:
            # 2D structure image
            st.markdown("**2D Structure**")
            if HAS_RDKIT and smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    img = Draw.MolToImage(mol, size=(400, 300))
                    st.image(img)
                else:
                    st.image(get_2d_image_url(cid), width=400)
            else:
                st.image(get_2d_image_url(cid), width=400)

        with col_b:
            st.markdown("**Properties**")
            props = {
                "Molecular Formula": formula,
                "Molecular Weight (g/mol)": mw,
                "Exact Mass": compound.get("ExactMass", ""),
                "SMILES": smiles,
                "XLogP (lipophilicity)": compound.get("XLogP", "N/A"),
                "TPSA (polar surface area)": compound.get("TPSA", "N/A"),
                "H-Bond Donors": compound.get("HBondDonorCount", ""),
                "H-Bond Acceptors": compound.get("HBondAcceptorCount", ""),
                "Rotatable Bonds": compound.get("RotatableBondCount", ""),
                "Charge": compound.get("Charge", ""),
            }
            for k, v in props.items():
                st.markdown(f"**{k}:** {v}")

        # Lipinski's Rule of Five
        try:
            mw_val = float(mw)
            logp = compound.get("XLogP")
            hbd = compound.get("HBondDonorCount", 0)
            hba = compound.get("HBondAcceptorCount", 0)
            if logp is not None:
                violations = sum([
                    mw_val > 500,
                    float(logp) > 5,
                    int(hbd) > 5,
                    int(hba) > 10,
                ])
                ro5 = "PASS" if violations <= 1 else "FAIL"
                st.markdown(f"**Lipinski Rule of Five:** {ro5} ({violations} violation{'s' if violations != 1 else ''})")
        except (ValueError, TypeError):
            pass

        # Description
        desc = get_description(cid)
        if desc:
            st.markdown(f"**Description:** {desc[:500]}")

        st.markdown(f"[View on PubChem](https://pubchem.ncbi.nlm.nih.gov/compound/{cid})")
