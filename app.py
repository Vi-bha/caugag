import streamlit as st
import networkx as nx
import sys
sys.path.insert(0, ".")

from caugag.claim_parser import ClaimParser
from caugag.graph_verifier import GraphVerifier, Verdict
from caugag.gate import GenerationGate
from caugag.benchmark.sachs_benchmark import get_sachs_dag, SACHS_EDGES
from caugag.benchmark.german_credit_benchmark import get_german_dag, GERMAN_EDGES

st.set_page_config(page_title="CauGAG Demo", page_icon="🔬", layout="wide")

st.title("CauGAG: Causal-Graph-Augmented Generation")
st.markdown("**Ask causal questions — CauGAG verifies claims mathematically before answering**")

# Sidebar
with st.sidebar:
    st.header("Settings")
    groq_key = st.text_input("Groq API Key", type="password")
    dataset = st.selectbox("Select Dataset", [
        "Sachs Protein Signaling (Biology)",
        "German Credit (FinTech)"
    ])
    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("1. Parse your question")
    st.markdown("2. Verify against causal DAG")
    st.markdown("3. Gate the LLM response")
    st.markdown("4. Show mathematical proof")

# Load DAG
if "Sachs" in dataset:
    dag = get_sachs_dag()
    edges = SACHS_EDGES
    example_questions = [
        "Does Mek directly cause Erk?",
        "Does Erk cause PKC?",
        "What happens to Akt if we intervene on PKA?",
        "Is the relationship between Raf and Akt confounded?",
        "Does PKC cause Mek?",
    ]
else:
    dag = get_german_dag()
    edges = GERMAN_EDGES
    example_questions = [
        "Does CreditHistory directly cause LoanDefault?",
        "Does LoanDefault cause Income?",
        "If Income increases, what happens to LoanDefault?",
        "Is the relationship between Employment and LoanDefault confounded?",
        "Does Age directly cause LoanDefault?",
    ]

# Show DAG
col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("Causal DAG")
    edge_text = "\n".join([f"**{u}** → {v}" for u,v in edges[:10]])
    st.markdown(edge_text)
    if len(edges) > 10:
        st.markdown(f"*...and {len(edges)-10} more edges*")

with col1:
    st.subheader("Ask a Causal Question")

    # Example questions
    st.markdown("**Example questions:**")
    for eq in example_questions:
        if st.button(eq, key=eq):
            st.session_state["question"] = eq

    question = st.text_input(
        "Your question:",
        value=st.session_state.get("question", ""),
        placeholder="e.g. Does X directly cause Y?"
    )

    if st.button("Ask CauGAG", type="primary") and question and groq_key:
        with st.spinner("Verifying causal claim..."):
            try:
                parser = ClaimParser(groq_api_key=groq_key)
                verifier = GraphVerifier(dag)
                gate = GenerationGate(groq_api_key=groq_key)

                # Parse
                parsed = parser.parse(question)
                # Verify
                verified = verifier.verify(parsed)
                # Gate
                output = gate.generate(question, verified)

                # Show results
                st.markdown("---")

                # Verdict badge
                verdict_colors = {
                    "VERIFIED": "🟢",
                    "UNVERIFIABLE": "🔴",
                    "CONFOUNDED": "🟡",
                    "ASSOCIATED": "🔵",
                }
                emoji = verdict_colors.get(verified.verdict.value, "⚪")

                col_v1, col_v2, col_v3 = st.columns(3)
                with col_v1:
                    st.metric("Verdict", f"{emoji} {verified.verdict.value}")
                with col_v2:
                    st.metric("Source", parsed.source_var)
                with col_v3:
                    st.metric("Target", parsed.target_var)

                # Proof object
                with st.expander("📐 Mathematical Proof", expanded=True):
                    st.markdown(f"**Explanation:** {verified.explanation}")
                    if verified.directed_path:
                        st.markdown(f"**Causal path:** {' → '.join(verified.directed_path)}")
                    if verified.adjustment_set:
                        st.markdown(f"**Adjustment set:** {verified.adjustment_set}")

                # Response
                st.markdown("### CauGAG Response")
                if output.was_refused:
                    st.error(output.response)
                else:
                    st.success(output.response)

                # Gate decision
                st.markdown("---")
                if output.was_refused:
                    st.warning("🚫 **Gate decision: BLOCKED** — LLM was not allowed to assert causation")
                else:
                    st.info("✅ **Gate decision: ALLOWED** — Causal claim is mathematically verified")

            except Exception as e:
                st.error(f"Error: {e}")

    elif st.button("Ask CauGAG", type="primary") and not groq_key:
        st.warning("Please enter your Groq API key in the sidebar")
