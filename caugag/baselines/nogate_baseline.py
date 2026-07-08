from groq import Groq
import networkx as nx

class NoGateBaseline:
    """
    Ablation: inject raw graph edges as context but NO gate.
    LLM sees edge list only — not the verifier verdict or explanation.
    This isolates the gate contribution from graph context contribution.
    """
    def __init__(self, groq_api_key: str, dag: nx.DiGraph,
                 model: str = "llama-3.1-8b-instant"):
        self.client = Groq(api_key=groq_api_key)
        self.model = model
        self.triples = [f"{u} -> {v}" for u,v in dag.edges()]

    def answer(self, question: str, source: str, target: str) -> str:
        # Give raw edge list only — no verdict, no explanation
        relevant = [t for t in self.triples
                    if source.lower() in t.lower()
                    or target.lower() in t.lower()]
        context = "\n".join(relevant) if relevant else "No relevant edges found."

        system = f"""You are a causal inference assistant.
Known causal edges from the graph:
{context}

Answer the question. You may make causal claims based on these edges."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ],
            temperature=0.1, max_tokens=512,
        )
        return response.choices[0].message.content.strip()
