from dataclasses import dataclass, field
import networkx as nx
import pandas as pd

GERMAN_NODES = [
    "Age", "Gender", "Employment", "Housing",
    "Savings", "Income", "CreditHistory",
    "LoanAmount", "LoanDuration", "LoanPurpose",
    "LoanDefault"
]

GERMAN_EDGES = [
    ("Age", "Employment"),("Age", "Income"),("Age", "CreditHistory"),
    ("Gender", "Income"),("Gender", "Employment"),
    ("Employment", "Income"),("Employment", "LoanDefault"),
    ("Housing", "LoanDefault"),("Housing", "Savings"),
    ("Income", "Savings"),("Income", "LoanAmount"),("Income", "LoanDefault"),
    ("Savings", "LoanDefault"),("CreditHistory", "LoanDefault"),
    ("LoanAmount", "LoanDefault"),("LoanDuration", "LoanDefault"),
    ("LoanPurpose", "LoanAmount"),
]

def get_german_dag():
    dag = nx.DiGraph()
    dag.add_nodes_from(GERMAN_NODES)
    dag.add_edges_from(GERMAN_EDGES)
    assert nx.is_directed_acyclic_graph(dag)
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

class GermanCreditBenchmark:
    def __init__(self):
        self.dag = get_german_dag()
        self.qa_pairs = []

    def generate(self):
        self.qa_pairs = []
        self._direction_true()
        self._direction_false_reversed()
        self._direction_false_nonedge()
        self._confounding()
        self._intervention_true()
        self._intervention_false()
        self._multihop()
        return self.qa_pairs

    def _direction_true(self):
        cases = [
            ("Age","Employment","Does Age directly affect Employment status?"),
            ("Income","LoanDefault","Does Income directly cause LoanDefault?"),
            ("CreditHistory","LoanDefault","Does CreditHistory directly cause LoanDefault?"),
            ("LoanAmount","LoanDefault","Does LoanAmount directly cause LoanDefault?"),
            ("Savings","LoanDefault","Does Savings directly affect LoanDefault?"),
            ("Employment","Income","Does Employment directly cause Income?"),
            ("LoanDuration","LoanDefault","Does LoanDuration directly cause LoanDefault?"),
            ("Housing","Savings","Does Housing directly affect Savings?"),
            ("Age","Income","Does Age directly affect Income?"),
            ("LoanPurpose","LoanAmount","Does LoanPurpose directly determine LoanAmount?"),
        ]
        for src,tgt,q in cases:
            self.qa_pairs.append(CausalQAPair(
                f"GER_TRUE_{src}_{tgt}",q,"direction",
                src,tgt,"yes","VERIFIED",[src,tgt]))

    def _direction_false_reversed(self):
        cases = [
            ("LoanDefault","Income","Does LoanDefault directly cause Income?"),
            ("LoanDefault","CreditHistory","Does LoanDefault directly cause CreditHistory?"),
            ("Employment","Age","Does Employment directly cause Age?"),
            ("Income","Age","Does Income directly cause Age?"),
            ("LoanAmount","Income","Does LoanAmount directly cause Income?"),
            ("LoanDefault","Savings","Does LoanDefault directly cause Savings?"),
        ]
        for src,tgt,q in cases:
            self.qa_pairs.append(CausalQAPair(
                f"GER_REV_{src}_{tgt}",q,"direction",
                src,tgt,"no","UNVERIFIABLE",notes="Reversed edge"))

    def _direction_false_nonedge(self):
        cases = [
            ("Gender","LoanDefault","Does Gender directly cause LoanDefault?"),
            ("Housing","Income","Does Housing directly cause Income?"),
            ("LoanDuration","Savings","Does LoanDuration directly affect Savings?"),
            ("LoanPurpose","LoanDefault","Does LoanPurpose directly cause LoanDefault?"),
            ("Age","Savings","Does Age directly cause Savings?"),
        ]
        for src,tgt,q in cases:
            self.qa_pairs.append(CausalQAPair(
                f"GER_NONE_{src}_{tgt}",q,"direction",
                src,tgt,"no","UNVERIFIABLE",notes="No direct edge"))

    def _confounding(self):
        cases = [
            ("Employment","LoanDefault","Is the relationship between Employment and LoanDefault confounded?"),
            ("Income","CreditHistory","Is the relationship between Income and CreditHistory confounded by Age?"),
            ("Savings","LoanAmount","Is the relationship between Savings and LoanAmount confounded?"),
        ]
        for src,tgt,q in cases:
            parents_src = set(self.dag.predecessors(src))
            parents_tgt = set(self.dag.predecessors(tgt))
            common = parents_src & parents_tgt
            verdict = "CONFOUNDED" if common else "VERIFIED"
            self.qa_pairs.append(CausalQAPair(
                f"GER_CONF_{src}_{tgt}",q,"confounding",
                src,tgt,"yes_confounded",verdict,
                notes=f"Common causes: {common}"))

    def _intervention_true(self):
        cases = [
            ("Income","LoanDefault","If we increase Income, what happens to LoanDefault risk?"),
            ("CreditHistory","LoanDefault","What happens to LoanDefault if CreditHistory improves?"),
            ("LoanAmount","LoanDefault","If we intervene on LoanAmount, what happens to LoanDefault?"),
            ("Savings","LoanDefault","If Savings increases, what is the causal effect on LoanDefault?"),
            ("LoanDuration","LoanDefault","What happens to LoanDefault if LoanDuration is extended?"),
            ("Employment","LoanDefault","What happens to LoanDefault if Employment status changes?"),
        ]
        for src,tgt,q in cases:
            if nx.has_path(self.dag,src,tgt):
                path = nx.shortest_path(self.dag,src,tgt)
                self.qa_pairs.append(CausalQAPair(
                    f"GER_INTERV_{src}_{tgt}",q,"intervention",
                    src,tgt,"yes_causal_effect","VERIFIED",path))

    def _intervention_false(self):
        cases = [
            ("LoanDefault","Income","If LoanDefault increases, what happens to Income?"),
            ("LoanDefault","CreditHistory","What happens to CreditHistory if we intervene on LoanDefault?"),
            ("LoanDefault","Savings","If we intervene on LoanDefault, what happens to Savings?"),
            ("Income","Age","If Income increases, what happens to Age?"),
            ("LoanAmount","Employment","What happens to Employment if LoanAmount changes?"),
        ]
        for src,tgt,q in cases:
            self.qa_pairs.append(CausalQAPair(
                f"GER_FALSE_{src}_{tgt}",q,"intervention",
                src,tgt,"no_causal_effect","UNVERIFIABLE",
                notes="Reversed or no path"))

    def _multihop(self):
        cases = [
            ("Age","LoanDefault","Does Age indirectly affect LoanDefault through Income?"),
            ("Employment","LoanDefault","Does Employment indirectly affect LoanDefault through Income?"),
            ("Gender","LoanDefault","Does Gender indirectly affect LoanDefault through Income?"),
            ("Housing","LoanDefault","Does Housing indirectly affect LoanDefault through Savings?"),
        ]
        for src,tgt,q in cases:
            if nx.has_path(self.dag,src,tgt):
                path = nx.shortest_path(self.dag,src,tgt)
                self.qa_pairs.append(CausalQAPair(
                    f"GER_MULTI_{src}_{tgt}",q,"intervention",
                    src,tgt,"yes_indirect","VERIFIED",path,
                    notes="Multi-hop"))

    def to_dataframe(self):
        if not self.qa_pairs: self.generate()
        return pd.DataFrame([vars(qa) for qa in self.qa_pairs])

    def summary(self):
        if not self.qa_pairs: self.generate()
        df = self.to_dataframe()
        return {"total":len(df),
                "by_type":df["query_type"].value_counts().to_dict(),
                "by_verdict":df["correct_verdict"].value_counts().to_dict()}
