import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="GLP-1 Market Intelligence", page_icon="💰", layout="wide")
st.title("GLP-1 Biosimilar Market Intelligence")
st.markdown(
    "The commercial opportunity for peptide stability prediction "
    "— driven by the largest patent cliff in pharmaceutical history."
)

# ============================================================
# TAB LAYOUT
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "Market Overview",
    "Why DPP-IV Matters",
    "Competitive Landscape",
    "CRO Pricing & ExoPred Value",
])

# ============================================================
# TAB 1: MARKET OVERVIEW
# ============================================================

with tab1:
    st.header("GLP-1 Market at a Glance")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("GLP-1 Market 2025", "&#36;64.4B")
    col2.metric("GLP-1 Market 2030", "&#36;268.4B")
    col3.metric("CAGR", "13%")
    col4.metric("Active Biosimilar Developers", "50+")

    st.subheader("Patent Cliff Timeline")

    patent_data = pd.DataFrame([
        {"Market": "Canada", "Drug": "Semaglutide", "Patent Expiry": "Jan 2026",
         "Status": "Expired — Sandoz, Apotex, Teva filed"},
        {"Market": "China", "Drug": "Semaglutide", "Patent Expiry": "Mar 2026",
         "Status": "17+ generics in Phase 3"},
        {"Market": "India", "Drug": "Semaglutide", "Patent Expiry": "Mar 2026",
         "Status": "50+ brands preparing"},
        {"Market": "US", "Drug": "Semaglutide", "Patent Expiry": "2031-2032",
         "Status": "Settlements with Mylan, Dr. Reddy's"},
        {"Market": "Global", "Drug": "Tirzepatide", "Patent Expiry": "2036-2038",
         "Status": "Longest runway"},
    ])
    st.dataframe(patent_data, hide_index=True, use_container_width=True)

    st.info(
        "The semaglutide patent cliff is already underway outside the US. "
        "Biosimilar developers in Canada, China, and India need analytical tools NOW "
        "to demonstrate equivalent stability profiles."
    )

    st.subheader("GLP-1 Market Size by Year")

    market_size_data = pd.DataFrame({
        "Year": [2024, 2025, 2026, 2027, 2028, 2029, 2030],
        "Market Size (&#36;B)": [53.7, 64.4, 101.4, 130.0, 165.0, 210.0, 268.4],
    })

    fig_market = px.bar(
        market_size_data,
        x="Year",
        y="Market Size (&#36;B)",
        text="Market Size (&#36;B)",
        color_discrete_sequence=["#2563eb"],
    )
    fig_market.update_traces(texttemplate="&#36;%{text:.1f}B", textposition="outside")
    fig_market.update_layout(
        xaxis_title="Year",
        yaxis_title="Market Size (&#36;B)",
        xaxis=dict(tickmode="linear"),
        showlegend=False,
        height=450,
    )
    st.plotly_chart(fig_market, use_container_width=True)

    st.success(
        "The GLP-1 market is projected to grow from &#36;53.7B (2024) to &#36;268.4B (2030). "
        "Every biosimilar entrant needs to prove equivalent DPP-IV resistance — "
        "the exact capability ExoPred provides."
    )

# ============================================================
# TAB 2: WHY DPP-IV MATTERS
# ============================================================

with tab2:
    st.header("Why DPP-IV Matters")
    st.markdown(
        "Understanding DPP-IV cleavage is the single most important factor "
        "in GLP-1 drug design and biosimilar approval."
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("The Problem")
        st.markdown(
            "- Native GLP-1 has a half-life of **~2 minutes** in the body\n"
            "- DPP-IV (dipeptidyl peptidase-4) cleaves the N-terminal His-Ala dipeptide\n"
            "- The cleaved peptide is biologically inactive\n"
            "- This makes native GLP-1 useless as a drug"
        )

    with col_b:
        st.subheader("The Solution")
        st.markdown(
            "- Semaglutide substitutes **Aib (alpha-aminoisobutyric acid)** at position 8\n"
            "- Aib sterically blocks DPP-IV from cleaving the peptide\n"
            "- This extends the half-life to **~7 days** (enabling weekly injection)\n"
            "- Combined with a fatty acid chain for albumin binding"
        )

    st.divider()

    st.subheader("Two Paths, Two Outcomes")

    st.markdown(
        """
```
Native GLP-1  ──▶  DPP-IV cleaves in ~2 min  ──▶  Inactive fragment  ──▶  No therapeutic value

Semaglutide   ──▶  Aib blocks DPP-IV         ──▶  7-day half-life    ──▶  &#36;64B market
```
"""
    )

    st.warning(
        "Every semaglutide biosimilar must demonstrate equivalent DPP-IV resistance. "
        "This is not optional — it is the core mechanism of action. "
        "FDA analytical similarity guidance (Sep 2025) means computational prediction "
        "of DPP-IV resistance can REPLACE expensive clinical trials."
    )

    st.subheader("Why This Matters for ExoPred")
    st.markdown(
        "- **50+ biosimilar developers** need to prove DPP-IV resistance\n"
        "- Traditional approach: synthesize candidates, run in-vitro DPP-IV assays (weeks, &#36;&#36;&#36;)\n"
        "- ExoPred approach: predict DPP-IV cleavage computationally in seconds\n"
        "- FDA's Sep 2025 guidance on analytical similarity makes computational evidence more valuable\n"
        "- First-mover advantage: no other AI tool specifically predicts exopeptidase degradation"
    )

    st.info(
        "The Aib substitution at position 8 is the &#36;64 billion insight. "
        "ExoPred is the only AI tool that can predict whether a given peptide modification "
        "will achieve equivalent DPP-IV resistance."
    )

# ============================================================
# TAB 3: COMPETITIVE LANDSCAPE
# ============================================================

with tab3:
    st.header("Competitive Landscape")
    st.markdown(
        "The peptide AI space has attracted over &#36;1B in funding — "
        "but **nobody** is doing degradation prediction."
    )

    competitors = pd.DataFrame([
        {"Company": "Peptilogics", "Focus": "De novo peptide design",
         "Funding": 205, "Does Degradation Prediction?": "No"},
        {"Company": "Generate:Biomedicines", "Focus": "Generative protein design",
         "Funding": 370, "Does Degradation Prediction?": "No"},
        {"Company": "Pinnacle Medicines", "Focus": "Oral peptide design",
         "Funding": 134, "Does Degradation Prediction?": "No"},
        {"Company": "Nuritas", "Focus": "Bioactive peptide discovery",
         "Funding": 130, "Does Degradation Prediction?": "No"},
        {"Company": "Peptone", "Focus": "Disordered proteins",
         "Funding": 42, "Does Degradation Prediction?": "No"},
        {"Company": "ProteinQure", "Focus": "Quantum + AI peptide design",
         "Funding": 16, "Does Degradation Prediction?": "No"},
        {"Company": "Chai Discovery", "Focus": "Structure prediction",
         "Funding": 230, "Does Degradation Prediction?": "No"},
        {"Company": "EvolutionaryScale", "Focus": "Protein language models",
         "Funding": 142, "Does Degradation Prediction?": "No"},
        {"Company": "ExoPred (Sam)", "Focus": "Exopeptidase degradation",
         "Funding": 0, "Does Degradation Prediction?": "YES — only one"},
    ])

    # Display table with funding formatted
    display_df = competitors.copy()
    display_df["Funding (&#36;M)"] = display_df["Funding"].apply(
        lambda x: f"&#36;{x}M" if x > 0 else "&#36;0"
    )
    st.dataframe(
        display_df[["Company", "Focus", "Funding (&#36;M)", "Does Degradation Prediction?"]],
        hide_index=True,
        use_container_width=True,
    )

    st.success(
        "Over &#36;1.2B has been invested in peptide AI companies. "
        "None of them predict enzymatic degradation. "
        "ExoPred owns this whitespace entirely."
    )

    st.subheader("Funding Comparison")

    fig_funding = px.bar(
        competitors.sort_values("Funding", ascending=True),
        x="Funding",
        y="Company",
        orientation="h",
        text="Funding",
        color="Does Degradation Prediction?",
        color_discrete_map={"No": "#94a3b8", "YES — only one": "#22c55e"},
    )
    fig_funding.update_traces(texttemplate="&#36;%{text}M", textposition="outside")
    fig_funding.update_layout(
        xaxis_title="Funding (&#36;M)",
        yaxis_title="",
        showlegend=True,
        legend_title="Degradation Prediction",
        height=450,
    )
    st.plotly_chart(fig_funding, use_container_width=True)

    st.warning(
        "ExoPred's competitive moat is Sam's proprietary LC-MS exopeptidase degradation dataset — "
        "the only one of its kind. Competitors would need 2-3 years of wet lab work to replicate."
    )

# ============================================================
# TAB 4: CRO PRICING & EXOPRED VALUE
# ============================================================

with tab4:
    st.header("CRO Pricing & ExoPred Value Proposition")
    st.markdown(
        "Pharma companies currently spend &#36;50K-200K per molecule on stability testing. "
        "ExoPred can deliver equivalent predictive insight at a fraction of the cost and time."
    )

    st.subheader("Current CRO Stability Testing Costs")

    cro_data = pd.DataFrame([
        {"Service": "Forced degradation study",
         "Cost per Molecule": "&#36;3K - &#36;15K", "Timeline": "4 - 8 weeks"},
        {"Service": "Full ICH stability program",
         "Cost per Molecule": "&#36;50K - &#36;200K", "Timeline": "6 - 36 months"},
        {"Service": "Method development",
         "Cost per Molecule": "&#36;10K - &#36;50K", "Timeline": "2 - 6 weeks"},
    ])
    st.dataframe(cro_data, hide_index=True, use_container_width=True)

    st.info(
        "A single biosimilar candidate can cost &#36;60K-265K in stability testing alone. "
        "Most developers screen 5-20 candidates, meaning &#36;300K-5M in CRO fees "
        "before a single candidate reaches clinical trials."
    )

    st.divider()

    st.subheader("ExoPred Pricing Model")

    pricing_data = pd.DataFrame([
        {"Tier": "API Query", "What": "Single peptide stability prediction",
         "Price": "&#36;50 - &#36;200 / query", "Margin": "~95%"},
        {"Tier": "Platform License", "What": "Unlimited queries, 1 year",
         "Price": "&#36;25K - &#36;100K / yr", "Margin": "~90%"},
        {"Tier": "Consulting", "What": "Custom modeling + interpretation",
         "Price": "&#36;15K - &#36;50K / project", "Margin": "~70%"},
        {"Tier": "Data License", "What": "Training data access",
         "Price": "&#36;100K - &#36;500K one-time", "Margin": "~95%"},
    ])
    st.dataframe(pricing_data, hide_index=True, use_container_width=True)

    st.divider()

    st.subheader("Revenue Scenarios")

    revenue_data = pd.DataFrame([
        {"Scenario": "Conservative", "Clients": "10 biosimilar cos",
         "Avg Revenue / Client": "&#36;25K", "Annual Revenue": "&#36;250K"},
        {"Scenario": "Base", "Clients": "25 pharma + biotech",
         "Avg Revenue / Client": "&#36;40K", "Annual Revenue": "&#36;1M"},
        {"Scenario": "Bull", "Clients": "50 clients + data licensing",
         "Avg Revenue / Client": "&#36;60K", "Annual Revenue": "&#36;3M"},
    ])
    st.dataframe(revenue_data, hide_index=True, use_container_width=True)

    col_x, col_y, col_z = st.columns(3)
    col_x.metric("Conservative", "&#36;250K / yr", help="10 biosimilar companies at &#36;25K avg")
    col_y.metric("Base Case", "&#36;1M / yr", help="25 pharma + biotech at &#36;40K avg")
    col_z.metric("Bull Case", "&#36;3M / yr", help="50 clients + data licensing at &#36;60K avg")

    st.success(
        "Even the conservative scenario (&#36;250K/yr) is achievable with 10 biosimilar clients "
        "paying for platform access. The bull case (&#36;3M/yr) requires penetrating the broader "
        "peptide therapeutics market and licensing Sam's proprietary degradation dataset."
    )

    st.divider()

    st.subheader("Unit Economics")
    st.markdown(
        "- **Cost to serve:** Near-zero marginal cost per API query (inference on existing model)\n"
        "- **CAC:** Low — biosimilar developers are a concentrated, identifiable market\n"
        "- **Retention:** High — stability data is integrated into development workflows\n"
        "- **Expansion:** Each client screens multiple candidates per year\n"
        "- **Moat:** Proprietary LC-MS dataset + first-mover in exopeptidase prediction"
    )

    st.warning(
        "Key risk: ExoPred must demonstrate predictive accuracy on DPP-IV cleavage "
        "that matches or exceeds in-vitro assay reproducibility (~85-90%). "
        "Sam's dataset is uniquely positioned to achieve this."
    )
