from dataclasses import dataclass, field
import networkx as nx
import pandas as pd

SACHS_NODES = ["PKC","PKA","Raf","Mek","Erk","Akt","PIP2","PIP3","JNK","P38","Plcg"]

SACHS_EDGES = [
    ("PKC","Mek"),("PKC","Raf"),("PKC","PIP2"),("PKC","JNK"),("PKC","P38"),
    ("PKA","Akt"),("PKA","Erk"),("PKA","Raf"),("PKA","Mek"),("PKA","JNK"),("PKA","P38"),
    ("Raf","Mek"),("Mek","Erk"),("Erk","Akt"),
    ("PIP3","PIP2"),("PIP3","Akt"),("Plcg","PIP3"),("Plcg","PIP2"),
]

def get_sachs_dag():
    dag = nx.DiGraph()
    dag.add_nodes_from(SACHS_NODES)
    dag.add_edges_from(SACHS_EDGES)
    return dag

@dataclass
class CausalQAPair:
    question_id: str
    question: str
    query_type: str
    source_var: str
    target_var: str
    correct_answer: str
    correct_verdict: str
    ground_truth_path: list = field(default_factory=list)
    notes: str = ""

class SachsBenchmark:
    def __init__(self):
        self.dag = get_sachs_dag()
        self.qa_pairs = []

    def generate(self):
        self.qa_pairs = []
        self._direction_true()
        self._direction_false()
        self._confounding()
        self._intervention_true()
        self._intervention_false()
        return self.qa_pairs

    def _direction_true(self):
        cases = [
            ("PKC","Raf","Does PKC directly cause Raf?"),
            ("Raf","Mek","Does Raf directly cause Mek?"),
            ("Mek","Erk","Does Mek directly cause Erk?"),
            ("Erk","Akt","Does Erk directly cause Akt?"),
            ("PKA","Akt","Does PKA directly cause Akt?"),
            ("PKA","Raf","Does PKA directly cause Raf?"),
            ("PIP3","Akt","Does PIP3 directly cause Akt?"),
            ("Plcg","PIP3","Does Plcg directly cause PIP3?"),
            ("PKC","JNK","Does PKC directly cause JNK?"),
            ("PKA","Mek","Does PKA directly cause Mek?"),
        ]
        for src,tgt,q in cases:
            self.qa_pairs.append(CausalQAPair(
                f"DIR_TRUE_{src}_{tgt}", q, "direction",
                src, tgt, "yes", "VERIFIED", [src,tgt]))

    def _direction_false(self):
        reversed_cases = [
            ("Mek","Raf","Does Mek directly cause Raf?"),
            ("Erk","Mek","Does Erk directly cause Mek?"),
            ("Akt","Erk","Does Akt directly cause Erk?"),
            ("Akt","PKA","Does Akt directly cause PKA?"),
        ]
        for src,tgt,q in reversed_cases:
            self.qa_pairs.append(CausalQAPair(
                f"DIR_REV_{src}_{tgt}", q, "direction",
                src, tgt, "no", "UNVERIFIABLE", notes="Reversed edge"))
        non_edges = [
            ("JNK","Erk","Does JNK directly cause Erk?"),
            ("P38","Akt","Does P38 directly cause Akt?"),
            ("PIP2","Mek","Does PIP2 directly cause Mek?"),
        ]
        for src,tgt,q in non_edges:
            self.qa_pairs.append(CausalQAPair(
                f"DIR_NONE_{src}_{tgt}", q, "direction",
                src, tgt, "no", "UNVERIFIABLE", notes="No edge"))

    def _confounding(self):
        cases = [
            ("Raf","Akt","Is the relationship between Raf and Akt confounded?","VERIFIED"),
            ("Mek","Akt","Is the relationship between Mek and Akt confounded?","VERIFIED"),
        ]
        for src,tgt,q,v in cases:
            self.qa_pairs.append(CausalQAPair(
                f"CONF_{src}_{tgt}", q, "confounding",
                src, tgt, "yes_confounded", v))

    def _intervention_true(self):
        cases = [
            ("Mek","Erk","What happens to Erk if we intervene on Mek?"),
            ("PKA","Akt","If we intervene on PKA, what happens to Akt?"),
            ("PKC","Erk","What is the causal effect of PKC on Erk?"),
            ("PIP3","Akt","What happens to Akt if PIP3 is experimentally increased?"),
            ("Raf","Erk","What is the causal effect of Raf on Erk?"),
        ]
        for src,tgt,q in cases:
            path = nx.shortest_path(self.dag, src, tgt)
            self.qa_pairs.append(CausalQAPair(
                f"INTERV_{src}_{tgt}", q, "intervention",
                src, tgt, "yes_causal_effect", "VERIFIED", path))

    def _intervention_false(self):
        cases = [
            ("Erk","PKC","What happens to PKC if we intervene on Erk?"),
            ("Akt","Raf","What is the effect of intervening on Akt on Raf?"),
            ("JNK","Mek","If we intervene on JNK, what happens to Mek?"),
        ]
        for src,tgt,q in cases:
            self.qa_pairs.append(CausalQAPair(
                f"INTERV_FALSE_{src}_{tgt}", q, "intervention",
                src, tgt, "no_causal_effect", "UNVERIFIABLE",
                notes="No causal path"))

    def to_dataframe(self):
        if not self.qa_pairs:
            self.generate()
        return pd.DataFrame([vars(qa) for qa in self.qa_pairs])

    def summary(self):
        if not self.qa_pairs:
            self.generate()
        df = self.to_dataframe()
        return {
            "total": len(df),
            "by_type": df["query_type"].value_counts().to_dict(),
            "by_verdict": df["correct_verdict"].value_counts().to_dict(),
        }
