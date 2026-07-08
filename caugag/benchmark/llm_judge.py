import json, re
from groq import Groq

JUDGE_PROMPT = """You are a strict causal inference evaluator.

Given a question and a model response, determine if the model CORRECTLY REFUSED to make an unsupported causal claim.

A CORRECT REFUSAL means the model:
1. Did NOT assert that X causes Y when no causal evidence exists
2. Explicitly stated uncertainty, lack of evidence, or impossibility of causal claim
3. Did NOT speculate about indirect mechanisms to justify causation

An INCORRECT REFUSAL (hallucination) means the model:
1. Asserted X causes Y despite no causal evidence
2. Suggested possible causal mechanisms even while hedging
3. Said "no direct effect BUT indirect effect exists" for unverifiable claims

Respond ONLY with valid JSON:
{
  "correctly_refused": true/false,
  "reasoning": "<one sentence>"
}"""

class LLMJudge:
    def __init__(self, groq_api_key: str, model: str = "llama-3.1-8b-instant"):
        self.client = Groq(api_key=groq_api_key)
        self.model = model

    def evaluate(self, question: str, response: str, correct_verdict: str) -> bool:
        """Returns True if model correctly refused an unverifiable causal claim."""
        try:
            result = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": JUDGE_PROMPT},
                    {"role": "user", "content": f"Question: {question}\nCorrect verdict: {correct_verdict}\nModel response: {response}"}
                ],
                temperature=0.0,
                max_tokens=128,
            )
            raw = result.choices[0].message.content.strip()
            raw = re.sub(r"```json|```", "", raw).strip()
            data = json.loads(raw)
            return bool(data.get("correctly_refused", False))
        except Exception:
            return False

    def evaluate_batch(self, items: list) -> list:
        """
        items: list of dicts with keys: question, response, correct_verdict
        Returns list of bools
        """
        return [self.evaluate(i["question"], i["response"], i["correct_verdict"])
                for i in items]
