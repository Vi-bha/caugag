import re
from dataclasses import dataclass
from typing import Optional
from groq import Groq
from caugag.graph_verifier import Verdict, VerificationResult


GATE_PROMPTS = {
    Verdict.VERIFIED: """You are a causal inference assistant.
The causal graph has VERIFIED that {source} causally affects {target}.
Graph proof: {explanation}
Directed path: {path}
{dowhy_context}
Answer the question clearly. State the causal relationship.
If a numeric effect is provided, report it with its confidence interval.
Do NOT speculate beyond what the graph tells you.""",

    Verdict.CONFOUNDED: """You are a causal inference assistant.
The causal graph found the relationship between {source} and {target} is CONFOUNDED.
Graph proof: {explanation}
Adjustment set needed: {adjustment_set}
You MUST:
1. Say the relationship is confounded
2. Name the confounders: {adjustment_set}
3. NOT assert that {source} directly causes {target}""",

    Verdict.UNVERIFIABLE: """You are a causal inference assistant.
The causal graph does NOT support a causal claim between {source} and {target}.
Graph proof: {explanation}
You MUST:
1. Clearly state the graph provides NO evidence of {source} causing {target}
2. REFUSE to assert causation
3. Note that correlation does not imply causation""",

    Verdict.ASSOCIATED: """You are a causal inference assistant.
The causal graph shows a statistical association between {source} and {target} only.
Graph proof: {explanation}
Association path: {path}
You MUST:
1. Use "associated" or "correlated" — NOT "causes"
2. Explicitly state this is NOT a causal relationship
3. Note that intervention study would be needed for causation"""
}


@dataclass
class GateOutput:
    verdict: Verdict
    response: str
    source_var: str
    target_var: str
    was_allowed: bool
    was_refused: bool
    dowhy_estimate: Optional[float] = None
    verification_explanation: str = ""


class GenerationGate:
    def __init__(self, groq_api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.client = Groq(api_key=groq_api_key)
        self.model = model

    def generate(self, question: str, verification: VerificationResult,
                 dowhy_result: Optional[dict] = None) -> GateOutput:
        verdict = verification.verdict
        source = verification.source_var
        target = verification.target_var

        dowhy_context = ""
        dowhy_estimate = None
        if dowhy_result and verdict == Verdict.VERIFIED:
            est = dowhy_result.get("estimate")
            ci_lower = dowhy_result.get("ci_lower")
            ci_upper = dowhy_result.get("ci_upper")
            if est is not None:
                dowhy_estimate = est
                dowhy_context = f"DoWhy causal effect estimate: beta={est:.4f} [95% CI: {ci_lower:.4f}, {ci_upper:.4f}]"

        prompt = GATE_PROMPTS[verdict].format(
            source=source,
            target=target,
            explanation=verification.explanation,
            path=" -> ".join(verification.directed_path) if verification.directed_path else "N/A",
            adjustment_set=verification.adjustment_set or [],
            dowhy_context=dowhy_context,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.1,
            max_tokens=512,
        )
        text = response.choices[0].message.content.strip()

        return GateOutput(
            verdict=verdict,
            response=text,
            source_var=source,
            target_var=target,
            was_allowed=verdict in (Verdict.VERIFIED, Verdict.ASSOCIATED),
            was_refused=verdict in (Verdict.UNVERIFIABLE, Verdict.CONFOUNDED),
            dowhy_estimate=dowhy_estimate,
            verification_explanation=verification.explanation,
        )

    def generate_vanilla(self, question: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant knowledgeable about causal inference. Answer the user causal question directly."},
                {"role": "user", "content": question},
            ],
            temperature=0.1,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()
