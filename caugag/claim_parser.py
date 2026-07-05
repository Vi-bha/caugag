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
Convert the user question into JSON with these fields:
{
  "query_type": "association" | "intervention" | "counterfactual" | "unknown",
  "source_var": "<cause variable, exact name from question>",
  "target_var": "<effect variable, exact name from question>",
  "conditioning_set": [],
  "confidence": <0.0-1.0>
}
Rules:
- association: question asks about correlation/relationship
- intervention: question asks about causation, effect of X on Y
- counterfactual: past tense hypothetical (would have, had been)
- Output ONLY valid JSON, no extra text.

Examples:
Q: Does PKC cause Mek? -> {"query_type":"intervention","source_var":"PKC","target_var":"Mek","conditioning_set":[],"confidence":0.95}
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
                max_tokens=256,
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r"```json|```", "", raw).strip()
            data = json.loads(raw)
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
