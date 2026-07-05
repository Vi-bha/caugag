from groq import Groq

CAUSAL_CORPUS = [
    {
        "id": "backdoor_criterion",
        "text": "The backdoor criterion: A set Z satisfies the backdoor criterion relative to (X,Y) if no variable in Z is a descendant of X, and Z blocks every path between X and Y that contains an arrow into X.",
        "keywords": ["backdoor","confounding","adjustment","criterion","causal effect"]
    },
    {
        "id": "d_separation",
        "text": "D-separation: Variables X and Y are d-separated by Z if every path between X and Y is blocked by Z. D-separation implies conditional independence.",
        "keywords": ["d-separation","independence","blocked","path","collider"]
    },
    {
        "id": "interventional",
        "text": "The do-calculus: P(Y|do(X=x)) is the interventional distribution when X is set by external intervention. The do-operator removes all incoming edges to X in the causal graph.",
        "keywords": ["do-calculus","intervention","do-operator","interventional"]
    },
    {
        "id": "confounding",
        "text": "Confounding: A confounder causally affects both treatment and outcome, creating spurious association. Confounding makes it impossible to infer causation from correlation without adjustment.",
        "keywords": ["confounding","confounder","spurious","correlation","causation"]
    },
    {
        "id": "sachs_signaling",
        "text": "Protein signaling: PKC and PKA phosphorylate downstream proteins Raf, Mek, Erk in cascade pathways. The MAPK cascade Raf->Mek->Erk is a well-characterized causal chain. PKA regulates Raf, Mek, Akt, JNK, P38.",
        "keywords": ["PKC","PKA","Raf","Mek","Erk","Akt","signaling","MAPK","kinase","PIP3","Plcg"]
    },
    {
        "id": "correlation_causation",
        "text": "Correlation does not imply causation. Two variables can be correlated because one causes the other, a common cause exists, or by coincidence. Without randomization or causal identification, correlation cannot establish causation.",
        "keywords": ["correlation","causation","spurious","randomized","identification"]
    },
    {
        "id": "causal_discovery",
        "text": "Causal discovery algorithms like PC and FCI learn causal structure from observational data using conditional independence tests. PC outputs a CPDAG representing a Markov equivalence class.",
        "keywords": ["PC algorithm","causal discovery","conditional independence","CPDAG","DAG"]
    },
]

def bm25_score(query, doc):
    query_words = set(query.lower().split())
    doc_text = (doc["text"] + " " + " ".join(doc["keywords"])).lower()
    doc_words = set(doc_text.split())
    return len(query_words & doc_words) / (len(query_words) + 1)


class RAGBaseline:
    def __init__(self, groq_api_key: str, model: str = "llama-3.3-70b-versatile", top_k: int = 3):
        self.client = Groq(api_key=groq_api_key)
        self.model = model
        self.top_k = top_k

    def retrieve(self, question: str):
        scores = [(doc, bm25_score(question, doc)) for doc in CAUSAL_CORPUS]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scores[:self.top_k]]

    def answer(self, question: str) -> str:
        retrieved = self.retrieve(question)
        context = "\n\n".join([f"[{d['id']}]: {d['text']}" for d in retrieved])
        system_prompt = f"""You are a causal inference assistant.
Use the following retrieved context to answer the question:

{context}

Answer based on this context and your knowledge."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.1,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()


class GraphRAGBaseline:
    def __init__(self, groq_api_key: str, dag, model: str = "llama-3.3-70b-versatile"):
        self.client = Groq(api_key=groq_api_key)
        self.model = model
        self.dag = dag
        self.triples = [f"{u} directly causes {v}" for u, v in dag.edges()]

    def retrieve(self, source: str, target: str):
        return [t for t in self.triples
                if source.lower() in t.lower() or target.lower() in t.lower()][:10]

    def answer(self, question: str, source: str, target: str) -> str:
        triples = self.retrieve(source, target)
        context = "\n".join(triples) if triples else "No relevant graph triples found."
        system_prompt = f"""You are a causal inference assistant.
Known causal relationships from the graph:

{context}

Use this to answer the question. You may assert causal relationships if they appear above."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.1,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()
