import gradio as gr
import sys
sys.path.insert(0, ".")
import os

from caugag.claim_parser import ClaimParser
from caugag.graph_verifier import GraphVerifier, Verdict
from caugag.gate import GenerationGate
from caugag.benchmark.sachs_benchmark import get_sachs_dag
from caugag.benchmark.german_credit_benchmark import get_german_dag

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

DATASETS = {
    "German Credit (FinTech)": get_german_dag(),
    "Sachs Protein Signaling (Biology)": get_sachs_dag(),
}

EXAMPLES = {
    "German Credit (FinTech)": [
        "Does CreditHistory directly cause LoanDefault?",
        "Does LoanDefault cause Income?",
        "If Income increases, what happens to LoanDefault?",
        "Is the relationship between Employment and LoanDefault confounded?",
        "Does Age directly cause LoanDefault?",
    ],
    "Sachs Protein Signaling (Biology)": [
        "Does Mek directly cause Erk?",
        "Does Erk cause PKC?",
        "What happens to Akt if we intervene on PKA?",
        "Is the relationship between Raf and Akt confounded?",
        "Does PKC cause Mek?",
    ],
}

def run_caugag(question, dataset_name, api_key):
    key = api_key or GROQ_API_KEY
    if not key:
        return "❌ Please enter your Groq API key", "", "", ""
    if not question.strip():
        return "❌ Please enter a question", "", "", ""

    try:
        dag = DATASETS[dataset_name]
        parser = ClaimParser(groq_api_key=key)
        verifier = GraphVerifier(dag)
        gate = GenerationGate(groq_api_key=key)

        parsed = parser.parse(question)
        verified = verifier.verify(parsed)
        output = gate.generate(question, verified)

        verdict_emoji = {
            "VERIFIED": "🟢 VERIFIED",
            "UNVERIFIABLE": "🔴 UNVERIFIABLE",
            "CONFOUNDED": "🟡 CONFOUNDED",
            "ASSOCIATED": "🔵 ASSOCIATED",
        }.get(verified.verdict.value, verified.verdict.value)

        gate_decision = (
            "🚫 BLOCKED — LLM was not allowed to assert causation"
            if output.was_refused
            else "✅ ALLOWED — Causal claim is mathematically verified"
        )

        proof = verified.explanation
        if verified.directed_path:
            proof += f"\nCausal path: {' → '.join(verified.directed_path)}"
        if verified.adjustment_set:
            proof += f"\nAdjustment set: {verified.adjustment_set}"

        return verdict_emoji, gate_decision, proof, output.response

    except Exception as e:
        return f"Error: {e}", "", "", ""

with gr.Blocks(title="CauGAG Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔬 CauGAG: Causal-Graph-Augmented Generation
    **Ask causal questions — CauGAG verifies claims mathematically before answering**

    *Paper: CauGAG: Grounding Causal Claims in LLMs via Graph-Verified Retrieval*
    """)

    with gr.Row():
        with gr.Column(scale=2):
            api_key = gr.Textbox(
                label="Groq API Key (get free at console.groq.com)",
                type="password",
                placeholder="gsk_..."
            )
            dataset = gr.Dropdown(
                choices=list(DATASETS.keys()),
                value="German Credit (FinTech)",
                label="Select Dataset"
            )
            question = gr.Textbox(
                label="Your causal question",
                placeholder="e.g. Does CreditHistory directly cause LoanDefault?",
                lines=2
            )
            submit = gr.Button("Ask CauGAG", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### Example Questions")
            gr.Markdown("""
            **German Credit:**
            - Does CreditHistory directly cause LoanDefault?
            - Does LoanDefault cause Income?
            - If Income increases, what happens to LoanDefault?

            **Sachs Biology:**
            - Does Mek directly cause Erk?
            - Does Erk cause PKC?
            - What happens to Akt if we intervene on PKA?
            """)

    gr.Markdown("---")
    gr.Markdown("### Results")

    with gr.Row():
        verdict_out = gr.Textbox(label="🎯 Verdict", interactive=False)
        gate_out = gr.Textbox(label="🚦 Gate Decision", interactive=False)

    proof_out = gr.Textbox(label="📐 Mathematical Proof (d-separation + backdoor criterion)", interactive=False, lines=3)
    response_out = gr.Textbox(label="💬 CauGAG Response", interactive=False, lines=5)

    submit.click(
        fn=run_caugag,
        inputs=[question, dataset, api_key],
        outputs=[verdict_out, gate_out, proof_out, response_out]
    )

    gr.Markdown("""
    ---
    ### How CauGAG works
    1. **Claim Parser** — converts your question into a structured causal query
    2. **Graph Verifier** — checks the causal DAG using d-separation and backdoor criterion
    3. **Generation Gate** — VERIFIED → LLM answers | UNVERIFIABLE → LLM refuses
    4. **DoWhy Injector** — injects verified causal effect estimates

    **Key result:** CauGAG achieves 94.4% correct refusal rate vs 12.3% for vanilla LLM across 4 datasets
    """)

demo.launch()
