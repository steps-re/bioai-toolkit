import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Commercial Opportunities", page_icon="💼", layout="wide")
st.title("Commercial Opportunities")
st.markdown(
    "Market analysis and commercialization pathways for peptide degradation IP "
    "and the Bio-AI Toolkit platform. Based on Rozans et al. (Pashuck Lab, Lehigh University)."
)

# ============================================================
# MARKET DATA
# ============================================================

MARKET_SEGMENTS = [
    {"Segment": "Peptide Therapeutics", "Market Size (2025)": "&#36;50B+", "CAGR": "9.1%",
     "Relevance": "Peptide stability is the #1 formulation challenge. Sam's degradation data directly addresses this.",
     "Key Players": "Novo Nordisk, Eli Lilly, Sanofi, AstraZeneca"},
    {"Segment": "3D Cell Culture / Organoids", "Market Size (2025)": "&#36;1.3-2.0B", "CAGR": "12-15%",
     "Relevance": "PEG hydrogels with peptide ligands are the standard. Nobody tests for degradation during culture.",
     "Key Players": "Corning, Merck (Matrigel), Cellendes, QGel"},
    {"Segment": "Biomaterials / Tissue Engineering", "Market Size (2025)": "&#36;186-238B", "CAGR": "8-15%",
     "Relevance": "Every implantable hydrogel uses adhesion peptides. Degradation affects long-term performance.",
     "Key Players": "Integra LifeSciences, Stryker, Smith+Nephew, Organogenesis"},
    {"Segment": "Contract Research (CRO)", "Market Size (2025)": "&#36;80B+", "CAGR": "7.8%",
     "Relevance": "Peptide stability testing is a standard CRO service. Sam's high-throughput LC-MS assay is 10x faster.",
     "Key Players": "Eurofins, SGS, WuXi, Charles River"},
    {"Segment": "AI Drug Discovery", "Market Size (2025)": "&#36;2-5B", "CAGR": "15-30%",
     "Relevance": "ML models for peptide stability prediction are a gap in the market. First-mover advantage.",
     "Key Players": "Insilico Medicine, Recursion, Exscientia, Isomorphic Labs"},
]

# ============================================================
# UI
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "IP Portfolio", "Startup Concepts", "Licensing Opportunities", "Market Analysis", "Competitive Moat", "Next Steps"
])

with tab1:
    st.markdown("### Intellectual Property Portfolio")
    st.markdown("Assets derived from 3 publications + ~80,000 LC-MS data points (Sam's estimate of unpublished dataset size) + Bio-AI Toolkit.")

    ip_assets = [
        {"Asset": "High-Throughput LC-MS Degradation Assay",
         "Type": "Method / Trade Secret",
         "Protection": "Published protocol (Paper 2) — trade secret in optimization details",
         "Value": "10x faster than competing assays. <5 min/sample. Standardized internal standard.",
         "Status": "Published — freely available. Competitive advantage in implementation speed + know-how."},

        {"Asset": "Terminal Modification Protection Library",
         "Type": "Dataset / Know-How",
         "Protection": "Published (Paper 1) — but the raw 80K dataset is unpublished",
         "Value": "Only systematic dataset of terminal modification effects across 3 cell types. "
                  "No competitor has this breadth.",
         "Status": "Published findings, unpublished raw data. Raw data enables ML model training."},

        {"Asset": "KLVADLMASAE Crosslinker Sequence",
         "Type": "Composition of Matter (potential patent)",
         "Protection": "Published (Paper 3) — patent window may still be open (filed <1 year from publication?)",
         "Value": "First MMP-14-selective crosslinker for pericellular hydrogel degradation. "
                  "Outperforms GPQGIWGQ (the industry standard) in gel stability.",
         "Status": "Check with Lehigh OTL — may be patented or patent-pending."},

        {"Asset": "Exopeptidase Degradation Predictor (ML Model)",
         "Type": "Software / Algorithm",
         "Protection": "Not yet built — first-mover opportunity. Would be patentable as a method.",
         "Value": "No competing tool exists. CleaveNet, UniZyme, PROSPERous all focus on endopeptidases. "
                  "This would be the first exopeptidase-specific prediction tool.",
         "Status": "Requires Sam's 80K dataset to train. Could be SaaS product."},

        {"Asset": "Bio-AI Toolkit Platform",
         "Type": "Software Platform",
         "Protection": "Open-source (value in network effects + data moat)",
         "Value": "13-page integrated analysis platform. Demonstrates full-stack bio + data science + ML capability.",
         "Status": "Deployed and live. Portfolio piece + foundation for commercial products."},

        {"Asset": "618-Peptide Enriched Database",
         "Type": "Curated Dataset",
         "Protection": "Derived from public data — freely shareable",
         "Value": "50 computed properties per peptide. Cross-referenced across 6 analytical tools. "
                  "No equivalent curated dataset exists.",
         "Status": "Complete. Available for download in the Rozans Analysis page."},
    ]

    for asset in ip_assets:
        with st.expander(f"**{asset['Asset']}** — {asset['Type']}"):
            st.markdown(f"**Protection:** {asset['Protection']}")
            st.markdown(f"**Value:** {asset['Value']}")
            st.markdown(f"**Status:** {asset['Status']}")

with tab2:
    st.markdown("### Startup Concepts")

    st.markdown("---")

    st.markdown("#### Concept 1: PepStable — Peptide Stability-as-a-Service")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **Problem:** Pharma companies spend 6-12 months testing peptide drug stability.
        Biomaterial companies don't test at all (and their products degrade in vivo).

        **Solution:** SaaS platform that predicts peptide degradation rates in any biological
        environment (blood, cell culture, tissue) and recommends stabilization strategies.

        **How it works:**
        1. Customer inputs peptide sequence + target environment
        2. ML model (trained on Sam's 80K dataset) predicts half-life
        3. Platform recommends terminal modifications, backbone substitutions, or formulation changes
        4. Optional: wet-lab validation service using Sam's high-throughput LC-MS assay

        **Revenue model:**
        - Freemium: 10 predictions/month free (with Bio-AI Toolkit as lead gen)
        - Pro: &#36;500/month for unlimited predictions + API access
        - Enterprise: &#36;5K/month for custom models + wet-lab validation
        - CRO service: &#36;200-500/peptide for LC-MS stability testing

        **Serviceable market:** &#36;5-20M (peptide stability SaaS is niche — ~500 pharma/biotech peptide programs x &#36;10-40K/yr)
        """)
    with col2:
        st.markdown("**Key Metrics**")
        st.metric("Prediction speed", "<1 second")
        st.metric("Competing assay time", "2-4 weeks")
        st.metric("Dataset advantage", "80K+ points")
        st.metric("Competitor models", "0 (whitespace)")

    st.markdown("---")

    st.markdown("#### Concept 2: SmartGel — Intelligent Hydrogel Design Platform")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **Problem:** Hydrogel design is trial-and-error. Researchers test 5-10 formulations
        over months. Most don't account for peptide degradation or protease specificity.

        **Solution:** Design platform that optimizes hydrogel formulations computationally
        before any wet-lab work. Integrates peptide stability, crosslinker kinetics,
        mechanical properties, and cell response prediction.

        **How it works:**
        1. Specify cell type, target stiffness, desired degradation profile
        2. Platform recommends: PEG backbone, adhesion peptide + protection, crosslinker
        3. Predicts gel behavior over 14 days (stiffness, degradation, cell spreading)
        4. Exports formulation protocol ready for the lab

        **Revenue model:**
        - Academic license: &#36;2K/year per lab
        - Biotech license: &#36;20K/year
        - Pharma enterprise: &#36;100K+/year with custom cell-type models

        **Serviceable market:** &#36;2-5M (niche academic/biotech tool — ~1,000 hydrogel labs x &#36;2-5K/yr)
        """)
    with col2:
        st.markdown("**Key Metrics**")
        st.metric("Design iterations saved", "5-10x")
        st.metric("Time to first gel", "Hours vs weeks")
        st.metric("Tools integrated", "13 pages")
        st.metric("Data sources", "20+ publications")

    st.markdown("---")

    st.markdown("#### Concept 3: ExoPred — Exopeptidase Intelligence Platform")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **Problem:** Every protease prediction tool (CleaveNet, PROSPERous, UniZyme) focuses
        on endopeptidases. Exopeptidases are ignored despite being the primary degradation
        pathway for therapeutic peptides and biomaterial ligands.

        **Solution:** The first ML platform specifically for exopeptidase substrate prediction.
        Trained on Sam's unique dataset — no competitor has equivalent training data.

        **How it works:**
        1. Input any peptide sequence
        2. Model predicts vulnerability to aminopeptidase N, leucine aminopeptidase,
           carboxypeptidase A, and carboxypeptidase B
        3. Recommends optimal terminal protection strategy
        4. Predicts cell-type-specific degradation rates

        **Competitive advantage:** First-mover in a genuine whitespace. Sam's 80K data points
        are the only training dataset that exists for this problem.

        **Revenue model:** API-based pricing (&#36;0.01/prediction), SaaS dashboard, pharma partnerships

        **Serviceable market:** &#36;1-5M initially (novel category — needs market creation;
        could grow if adopted as standard tool by peptide drug programs)
        """)
    with col2:
        st.markdown("**Key Metrics**")
        st.metric("Competing exo tools", "0")
        st.metric("Training data points", "80,000+")
        st.metric("Proteases covered", "4+")
        st.metric("Cell types modeled", "4")

with tab3:
    st.markdown("### Licensing Opportunities")
    st.markdown("Revenue without building a company — license the IP to existing players.")

    licensing = [
        {"Licensee Type": "Hydrogel Suppliers (Cellendes, QGel, Advanced BioMatrix)",
         "What They License": "Terminal modification recommendations for their peptide products",
         "Value Prop": "Their customers' experiments fail because peptides degrade. "
                       "Licensing Sam's data lets them sell 'degradation-resistant' peptide kits.",
         "Deal Structure": "Royalty: 3-5% on peptide product sales; or flat &#36;50-100K/year",
         "Estimated Revenue": "&#36;10-50K/year per licensee"},

        {"Licensee Type": "Pharma / Biotech (Peptide Drug Programs)",
         "What They License": "Stability prediction API + LC-MS assay protocol",
         "Value Prop": "Accelerate peptide drug formulation by 3-6 months. "
                       "De-risk stability failures before expensive clinical batches.",
         "Deal Structure": "Per-program fee: &#36;25-100K; or API subscription &#36;10-25K/year",
         "Estimated Revenue": "&#36;50-200K/year across 2-5 licensees"},

        {"Licensee Type": "CRO / CDMO (WuXi, Eurofins, Bachem)",
         "What They License": "High-throughput LC-MS assay as a service offering",
         "Value Prop": "Sam's assay is 10x faster than standard stability testing. "
                       "CROs can offer 'rapid peptide stability screening' as a premium service.",
         "Deal Structure": "Technology transfer fee: &#36;25-75K + running royalty 3-5%",
         "Estimated Revenue": "&#36;25-75K upfront + &#36;20-50K/year ongoing"},

        {"Licensee Type": "AI Drug Discovery (Insilico, Recursion, Isomorphic)",
         "What They License": "80K-point training dataset + exopeptidase model weights",
         "Value Prop": "Unique training data for peptide stability prediction. "
                       "No equivalent dataset exists. Integrates into their platforms.",
         "Deal Structure": "Data license: &#36;50-200K one-time; or co-development partnership with equity",
         "Estimated Revenue": "&#36;50-200K (or equity in co-development)"},

        {"Licensee Type": "Lehigh University OTL (if patent filed on KLVADLMASAE)",
         "What They License": "Composition of matter patent on MMP-14-selective crosslinker",
         "Value Prop": "KLVADLMASAE outperforms GPQGIWGQ — the 20-year industry standard. "
                       "Every hydrogel company would want this sequence.",
         "Deal Structure": "Non-exclusive license: &#36;10-50K per licensee + royalty; "
                          "exclusive to one company: &#36;100-500K upfront",
         "Estimated Revenue": "&#36;50-200K (depends on patent status — check Lehigh OTL)"},
    ]

    for lic in licensing:
        with st.expander(f"**{lic['Licensee Type']}**"):
            st.markdown(f"**What they license:** {lic['What They License']}")
            st.markdown(f"**Value proposition:** {lic['Value Prop']}")
            st.markdown(f"**Deal structure:** {lic['Deal Structure']}")
            st.metric("Estimated Revenue", lic["Estimated Revenue"])

with tab4:
    st.markdown("### Market Analysis")

    mkt_df = pd.DataFrame(MARKET_SEGMENTS)
    st.dataframe(mkt_df, hide_index=True)

    st.markdown("### Market Map")

    fig = go.Figure()

    # Bubble chart: market size vs CAGR vs relevance
    sizes = [50, 1.6, 210, 80, 3.5]
    cagrs = [9.1, 13.5, 12.0, 7.8, 22.0]
    names = ["Peptide\nTherapeutics", "3D Cell\nCulture", "Biomaterials /\nTissue Eng", "CRO", "AI Drug\nDiscovery"]
    relevance = [90, 95, 70, 80, 100]  # % relevance to Sam's IP

    fig.add_trace(go.Scatter(
        x=cagrs, y=[s if s < 100 else 50 for s in sizes],  # Cap display size
        mode="markers+text",
        text=names,
        textposition="top center",
        marker=dict(
            size=[r * 0.5 for r in relevance],
            color=relevance,
            colorscale="YlOrRd",
            showscale=True,
            colorbar=dict(title="Relevance"),
        ),
    ))
    fig.update_layout(
        xaxis_title="CAGR (%)",
        yaxis_title="Market Size (&#36;B)",
        height=450,
        margin=dict(t=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Key insight:** AI Drug Discovery has the highest growth rate (28.5% CAGR) and the
    strongest relevance to Sam's IP — but the peptide therapeutics market is 10x larger in
    absolute terms. The optimal strategy targets both: sell AI tools to drug discovery companies
    AND stability data to peptide therapeutics programs.
    """)

with tab5:
    st.markdown("### Competitive Moat")
    st.markdown("What makes this IP defensible and hard to replicate?")

    moat_data = [
        {"Moat": "Data Moat (80K LC-MS points)",
         "Strength": "Very Strong",
         "Explanation": "Sam's dataset took 3+ years of PhD work to generate. No shortcut — "
                        "each data point requires cell culture, sample prep, and LC-MS measurement. "
                        "A competitor would need &#36;500K+ and 2-3 years to replicate.",
         "Durability": "10+ years (dataset doesn't expire; only grows more valuable with additions)"},

        {"Moat": "First-Mover in Exopeptidase ML",
         "Strength": "Strong",
         "Explanation": "Every protease prediction tool focuses on endopeptidases. "
                        "Being first to market with an exopeptidase model creates category ownership. "
                        "Competitors would need their own exopeptidase training data — which doesn't exist elsewhere.",
         "Durability": "3-5 years (until someone generates comparable data)"},

        {"Moat": "Domain Expertise Moat",
         "Strength": "Strong",
         "Explanation": "PhD (biomaterials) + MBA (business) + this toolkit (data science) = "
                        "rare combination. Most ML researchers don't understand peptide chemistry. "
                        "Most biochemists can't build ML models. Sam can do both.",
         "Durability": "Career-long"},

        {"Moat": "Publication Record",
         "Strength": "Moderate",
         "Explanation": "3 published papers in top biomaterials journals. 7 total publications. "
                        "Cited by CleaveNet (Nature Comm) and other high-profile work. "
                        "Establishes credibility with customers and investors.",
         "Durability": "Permanent (citations only grow)"},

        {"Moat": "Assay Speed Advantage",
         "Strength": "Moderate",
         "Explanation": "Sam's LC-MS assay runs in <5 min/sample with standardized internal standards. "
                        "Competing assays take 30-60 min. This enables high-throughput screening "
                        "that competitors can't match.",
         "Durability": "5+ years (until someone develops a faster assay)"},
    ]

    for moat in moat_data:
        strength_color = {"Very Strong": "green", "Strong": "blue", "Moderate": "orange"}.get(moat["Strength"], "gray")
        with st.expander(f"**{moat['Moat']}** — :{strength_color}[{moat['Strength']}]"):
            st.markdown(f"{moat['Explanation']}")
            st.markdown(f"**Durability:** {moat['Durability']}")

    st.markdown("---")

    st.markdown("### Sam's Unique Position")
    st.markdown("""
    **Sam Rozans** — Venture Associate at Steps Ventures (2025-present), PhD Bioengineering (Lehigh),
    MBA (Lehigh). 7 publications, 80K+ LC-MS data points, 3 years peptide degradation research.

    | Capability | Sam | Typical PhD | Typical MBA | Typical Data Scientist |
    |-----------|-----|------------|------------|----------------------|
    | Peptide chemistry expertise | Deep (3 yrs PhD) | Varies | No | No |
    | LC-MS assay development | Invented the assay | Maybe | No | No |
    | 80K-point proprietary dataset | Yes | Maybe | No | No |
    | ML / data science | Learning (portfolio) | Rare | Maybe | Yes |
    | Business / market analysis | MBA | No | Yes | No |
    | Full-stack web deployment | This toolkit | No | No | Maybe |
    | Venture capital / due diligence | Steps Ventures | No | Maybe | No |

    **The combination is extremely rare.** Most people have 1-2 of these columns.
    Sam credibly fills 6 of 7 — and this toolkit demonstrates the 7th.
    """)

    st.markdown("---")
    st.caption(
        "This analysis is for strategic planning purposes. Market sizes are ranges from multiple "
        "analyst reports (Grand View Research, MarketsandMarkets, IMARC, Global Market Insights) — "
        "estimates vary significantly by scope definition. Revenue estimates for startup/licensing "
        "concepts are illustrative assumptions, not validated by customer discovery. "
        "All numbers should be treated as starting points for further research."
    )

with tab6:
    st.markdown("### Next Steps — From Portfolio Piece to Product")

    st.markdown("#### Phase 1: Data + Model (Weeks 1-4)")
    st.markdown("""
    - [ ] **Get Sam's 80K LC-MS dataset** in structured CSV format (see Rozans Analysis > "With 80K Data Points" tab for format spec)
    - [ ] **Clean and validate** the data — remove outliers, normalize to internal standards, flag quality issues
    - [ ] **Train v1 exopeptidase ML model** — start with random forest or XGBoost on sequence features vs measured degradation rate
    - [ ] **Validate** using leave-one-library-out cross-validation (train on 11 libraries, test on 12th)
    - [ ] **Benchmark** against the literature-derived scores currently in this toolkit — quantify improvement
    - [ ] **Integrate** trained model into the Rozans Analysis page as a new subtab: "ML Predictions"
    """)

    st.markdown("#### Phase 2: MVP Product (Weeks 5-8)")
    st.markdown("""
    - [ ] **Add API endpoint** — FastAPI service that accepts a peptide sequence and returns predicted half-life
    - [ ] **User accounts** — basic auth so usage can be tracked (freemium model)
    - [ ] **Landing page** — position as "the first exopeptidase degradation predictor" with Sam's publication record as credibility
    - [ ] **Benchmark page** — show model accuracy vs CleaveNet, PROSPERous, UniZyme (which don't cover exopeptidases)
    - [ ] **Deploy on custom domain** (e.g., pepstable.io or exopred.com)
    """)

    st.markdown("#### Phase 3: Validation + Outreach (Weeks 9-16)")
    st.markdown("""
    - [ ] **Submit paper** on the ML model + web tool to Bioinformatics or NAR Web Server issue
    - [ ] **Beta testers** — reach out to 5-10 hydrogel labs (start with Pashuck Lab collaborators: Stevens, Stupp, Webber, Chow)
    - [ ] **Conference presentation** — SFB (Society for Biomaterials) or BMES annual meeting
    - [ ] **Customer discovery** — 20 interviews with peptide therapeutics companies to validate pricing/willingness to pay
    - [ ] **Check patent status** of KLVADLMASAE with Lehigh OTL — if unpatented, evaluate provisional filing
    """)

    st.markdown("#### Phase 4: Revenue (Months 4-12)")
    st.markdown("""
    - [ ] **First paying customers** — target 3-5 academic labs at &#36;2-5K/year
    - [ ] **CRO partnership** — pitch 1-2 CROs on licensing the LC-MS assay as a service line
    - [ ] **SBIR/STTR grant** — NIH R41/R42 for "ML-driven peptide stability prediction" (&#36;275K Phase I)
    - [ ] **NSF I-Corps** — &#36;50K for customer discovery (Sam's MBA + this toolkit = strong application)
    - [ ] **Expand dataset** — collaborate with 2-3 labs to test predictions on their peptide systems (expands training data + validates model generalizability)
    """)

    st.markdown("---")

    st.markdown("#### What Sam Needs to Get Started")
    st.markdown("""
    | Item | Status | Action |
    |------|--------|--------|
    | 80K LC-MS dataset | Sam has it (unpublished) | Export from lab notebook to CSV |
    | Bio-AI Toolkit | Live (this site) | Share with potential employers / collaborators |
    | ML training pipeline | Not built yet | 2-3 days with scikit-learn / XGBoost |
    | Custom domain | Not set up | &#36;12/year for domain + free Cloud Run hosting |
    | Publication draft | Not started | 2-4 weeks writing after model validation |
    | Patent check | Unknown | Email Lehigh OTL re: KLVADLMASAE |

    **Minimum viable next step:** Export the 80K dataset to CSV and share it. Everything else
    builds on that data.
    """)

    st.markdown("---")

    st.markdown("#### For Employers Viewing This")
    st.markdown("""
    This toolkit demonstrates:
    - **Data science:** 618 peptides x 50+ computed properties, cross-tool correlation analysis
    - **ML readiness:** Model architecture defined, training data identified, validation strategy planned
    - **Domain expertise:** Deep understanding of peptide biochemistry, protease biology, and biomaterials
    - **Full-stack deployment:** Cloud Run, Streamlit, 15 interactive pages, live API integrations
    - **Business acumen:** Market sizing, IP strategy, licensing models, competitive moat analysis (MBA, Lehigh)
    - **Venture experience:** Venture Associate at Steps Ventures — due diligence, deal sourcing, portfolio support across climate-tech and deep-tech
    - **Scientific communication:** Published in ACS, Wiley, and bioRxiv; data presented in accessible visualizations

    Sam built the experimental foundation (3 papers, 80K data points) and then applied venture
    and business development skills to identify the commercial opportunity. This toolkit shows
    what happens when deep science meets entrepreneurial execution.
    """)
