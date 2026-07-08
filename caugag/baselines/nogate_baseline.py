from groq import Groq
from caugag.graph_verifier import VerificationResult

class NoGateBaseline:
    """
    Ablation: inject graph context but no generation gate.
    LLM sees the graph information but is free to assert any causal claim.
    Shows contribution of the gate specifically.
    """
    def __init__(self, groq_api_key: str, model: str = "llama-3.1-8b-instant"):
        self.client = Groq(api_key=groq_api_key)
        self.model = model

    def answer(self, question: str, verification: VerificationResult) -> str:
        # Give LLM the graph info but NO gate instruction
        context = f"""You are a causal inference assistant.
Here is information from the causal graph about {verification.source_var} and {verification.target_var}:
{verification.explanation}
Directed path if any: {" -> ".join(verification.directed_path) if verification.directed_path else "None found"}
Answer the user question based on this graph information."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": question},
            ],
            temperature=0.1, max_tokens=512,
        )
        return response.choices[0].message.content.strip()
