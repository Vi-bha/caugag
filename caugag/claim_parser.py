import json, re
from dataclasses import dataclass
from enum import Enum
from groq import Groq


class QueryType(str, Enum):
    ASSOCIATION = "association"
    INTERVENTION = "intervention"
    COUNTERFACTUAL = "counterfactual"
    UNKNOWN = "unknown"


@dataclass
class CausalQuery:
    query_type: QueryType
    source_var: str
    target_var: str
    conditioning_set: list
    raw_question: str
    confidence: float = 1.0


SYSTEM_PROMPT = """You are a causal query parser.
Your ONLY job is to extract the source and target variables from the question.
DO NOT use any prior knowledge about causal relationships.
Extract variables EXACTLY as they appear in the question syntax.

Rules:
- source_var: the variable being intervened on, manipulated, or asked about as a cause
- target_var: the variable whose outcome is being asked about
- The question syntax determines direction, NOT your knowledge of biology or causation
- "If we intervene on X, what happens to Y?" -> source=X, target=Y always
- "What happens to Y if X changes?" -> source=X, target=Y always
- "Does X cause Y?" -> source=X, target=Y always
- "Does X directly affect Y?" -> source=X, target=Y always
- NEVER swap source and target based on what you think the causal direction should be

query_type:
- association: correlation/relationship questions
- intervention: causation, effect of X on Y, what happens if X changes
- counterfactual: past tense hypothetical (would have, had been)

If you need to think through the problem, do so, but ALWAYS end your response
with the final JSON object on its own, with no other text after it.
Output ONLY valid JSON as the FINAL thing in your response:
{
  "query_type": "association"|"intervention"|"counterfactual"|"unknown",
  "source_var": "<variable being manipulated/cause — from question syntax>",
  "target_var": "<variable whose outcome is asked — from question syntax>",
  "conditioning_set": [],
  "confidence": <0.0-1.0>
}

Examples:
Q: Does PKC cause Mek? -> {"query_type":"intervention","source_var":"PKC","target_var":"Mek","conditioning_set":[],"confidence":0.95}
Q: Does Mek cause PKC? -> {"query_type":"intervention","source_var":"Mek","target_var":"PKC","conditioning_set":[],"confidence":0.95}
Q: If we intervene on Mek, what happens to PKC? -> {"query_type":"intervention","source_var":"Mek","target_var":"PKC","conditioning_set":[],"confidence":0.95}
Q: What happens to PKC if we intervene on Erk? -> {"query_type":"intervention","source_var":"Erk","target_var":"PKC","conditioning_set":[],"confidence":0.95}
Q: Is Raf correlated with Erk? -> {"query_type":"association","source_var":"Raf","target_var":"Erk","conditioning_set":[],"confidence":0.95}
Q: If PKA had been higher, would Raf have changed? -> {"query_type":"counterfactual","source_var":"PKA","target_var":"Raf","conditioning_set":[],"confidence":0.92}
"""


class ClaimParser:
    def __init__(self, groq_api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.client = Groq(api_key=groq_api_key)
        self.model = model

    def parse(self, question: str) -> CausalQuery:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Q: {question}"}
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            raw = response.choices[0].message.content.strip()

            # Strip reasoning blocks (e.g. Qwen/DeepSeek-style <think>...</think>)
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            raw = re.sub(r"```json|```", "", raw).strip()

            # Extract the last complete JSON object in the text
            # (handles cases where reasoning models leave trailing commentary)
            matches = re.findall(r"\{[^{}]*\}", raw, flags=re.DOTALL)
            json_str = matches[-1] if matches else raw

            data = json.loads(json_str)
            return CausalQuery(
                query_type=QueryType(data.get("query_type", "unknown")),
                source_var=data.get("source_var", ""),
                target_var=data.get("target_var", ""),
                conditioning_set=data.get("conditioning_set", []),
                raw_question=question,
                confidence=float(data.get("confidence", 1.0)),
            )
        except Exception:
            return CausalQuery(QueryType.UNKNOWN, "", "", [], question, 0.0)

    def parse_batch(self, questions: list) -> list:
        return [self.parse(q) for q in questions]
