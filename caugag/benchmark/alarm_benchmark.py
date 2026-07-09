from dataclasses import dataclass, field
import networkx as nx
import pandas as pd

# Alarm network — 37 nodes, 46 edges
# Source: Beinlich et al. (1989), standard BN repository
ALARM_NODES = [
    "HYPOVOLEMIA","LVEDVOLUME","STROKEVOLUME","ERRLOWOUTPUT",
    "HRBP","HREKG","HRSAT","ERRCAUTER","HR","CATECHOL",
    "SAO2","EXPCO2","ARTCO2","VENTALV","VENTLUNG","INTUBATION",
    "KINKEDTUBE","MINVOLSET","VENTMACH","DISCONNECT","MINVOL",
    "FIO2","PVSAT","VENTTUBE","PRESS","ANAPHYLAXIS","TPR",
    "INSUFFANESTH","PULMEMBOLUS","PAP","SHUNT","LVFAILURE",
    "HISTORY","CVP","PCWP","CO","BP"
]

ALARM_EDGES = [
    ("HYPOVOLEMIA","LVEDVOLUME"),("HYPOVOLEMIA","STROKEVOLUME"),
    ("LVEDVOLUME","STROKEVOLUME"),("STROKEVOLUME","CO"),
    ("ERRLOWOUTPUT","HRBP"),("ERRLOWOUTPUT","HREKG"),
    ("ERRLOWOUTPUT","HRSAT"),("ERRCAUTER","HREKG"),
    ("ERRCAUTER","HRSAT"),("HR","HRBP"),("HR","HREKG"),
    ("HR","HRSAT"),("CATECHOL","HR"),("SAO2","CATECHOL"),
    ("EXPCO2","CATECHOL"),("ARTCO2","CATECHOL"),
    ("ARTCO2","EXPCO2"),("VENTALV","ARTCO2"),("VENTALV","SAO2"),
    ("VENTLUNG","VENTALV"),("INTUBATION","VENTLUNG"),
    ("INTUBATION","VENTALV"),("KINKEDTUBE","VENTLUNG"),
    ("MINVOLSET","VENTMACH"),("VENTMACH","VENTTUBE"),
    ("DISCONNECT","VENTTUBE"),("VENTTUBE","VENTLUNG"),
    ("VENTTUBE","PRESS"),("MINVOL","VENTALV"),
    ("FIO2","PVSAT"),("PVSAT","SAO2"),
    ("ANAPHYLAXIS","TPR"),("TPR","CATECHOL"),("TPR","BP"),
    ("INSUFFANESTH","CATECHOL"),("PULMEMBOLUS","PAP"),
    ("PULMEMBOLUS","SHUNT"),("PAP","CATECHOL"),
    ("SHUNT","SAO2"),("LVFAILURE","LVEDVOLUME"),
    ("LVFAILURE","STROKEVOLUME"),("LVFAILURE","HISTORY"),
    ("LVFAILURE","CVP"),("LVFAILURE","PCWP"),
    ("CO","BP"),("CO","CVP"),
]

def get_alarm_dag():
    dag = nx.DiGraph()
    dag.add_nodes_from(ALARM_NODES)
    dag.add_edges_from(ALARM_EDGES)
    # Verify DAG
    assert nx.is_directed_acyclic_graph(dag), "Alarm graph has cycles"
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

class AlarmBenchmark:
    def __init__(self):
        self.dag = get_alarm_dag()
        self.edges = list(self.dag.edges())
        self.qa_pairs = []

    def generate(self):
        self.qa_pairs = []
        self._direction_true()
        self._direction_false_reversed()
        self._direction_false_nonedge()
        self._intervention_true()
        self._intervention_false()
        self._multihop()
        return self.qa_pairs

    def _direction_true(self):
        templates = [
            "Does {s} directly cause {t}?",
            "Is there a direct causal effect of {s} on {t}?",
            "Does {s} directly affect {t}?",
        ]
        for i,(src,tgt) in enumerate(self.edges[:15]):
            q = templates[i%3].format(s=src,t=tgt)
            self.qa_pairs.append(CausalQAPair(
                f"ALARM_TRUE_{i}",q,"direction",
                src,tgt,"yes","VERIFIED",[src,tgt]))

    def _direction_false_reversed(self):
        reversed_edges = [(v,u) for u,v in self.edges[:12]]
        for i,(src,tgt) in enumerate(reversed_edges):
            self.qa_pairs.append(CausalQAPair(
                f"ALARM_REV_{i}",
                f"Does {src} directly cause {tgt}?",
                "direction",src,tgt,"no","UNVERIFIABLE",
                notes="Reversed edge"))

    def _direction_false_nonedge(self):
        # Find non-edges
        edge_set = set(self.edges)
        non_edges = [
            (u,v) for u in ALARM_NODES[:10]
            for v in ALARM_NODES[:10]
            if u!=v and (u,v) not in edge_set
            and (v,u) not in edge_set
        ][:10]
        for i,(src,tgt) in enumerate(non_edges):
            self.qa_pairs.append(CausalQAPair(
                f"ALARM_NONE_{i}",
                f"Does {src} directly cause {tgt}?",
                "direction",src,tgt,"no","UNVERIFIABLE",
                notes="No edge"))

    def _intervention_true(self):
        cases = [
            ("LVFAILURE","CO","What happens to CO if we intervene on LVFAILURE?"),
            ("STROKEVOLUME","CO","What is the causal effect of STROKEVOLUME on CO?"),
            ("CATECHOL","HR","What happens to HR if CATECHOL is experimentally increased?"),
            ("VENTALV","SAO2","What is the interventional effect of VENTALV on SAO2?"),
            ("HYPOVOLEMIA","CO","If we intervene on HYPOVOLEMIA, what happens to CO?"),
            ("TPR","BP","What happens to BP if we intervene on TPR?"),
            ("FIO2","SAO2","What is the causal effect of FIO2 on SAO2?"),
            ("PULMEMBOLUS","SAO2","If PULMEMBOLUS occurs, what happens to SAO2?"),
        ]
        for src,tgt,q in cases:
            if nx.has_path(self.dag,src,tgt):
                path = nx.shortest_path(self.dag,src,tgt)
                self.qa_pairs.append(CausalQAPair(
                    f"ALARM_INTERV_{src}_{tgt}",q,"intervention",
                    src,tgt,"yes_causal_effect","VERIFIED",path))

    def _intervention_false(self):
        cases = [
            ("CO","HYPOVOLEMIA","What happens to HYPOVOLEMIA if we intervene on CO?"),
            ("HR","CATECHOL","What is the effect of intervening on HR on CATECHOL?"),
            ("SAO2","FIO2","If we intervene on SAO2, what happens to FIO2?"),
            ("BP","TPR","What happens to TPR if we experimentally increase BP?"),
            ("CVP","LVFAILURE","What is the causal effect of CVP on LVFAILURE?"),
            ("EXPCO2","VENTALV","If we intervene on EXPCO2, what happens to VENTALV?"),
            ("PRESS","VENTTUBE","What happens to VENTTUBE if we intervene on PRESS?"),
        ]
        for src,tgt,q in cases:
            self.qa_pairs.append(CausalQAPair(
                f"ALARM_FALSE_{src}_{tgt}",q,"intervention",
                src,tgt,"no_causal_effect","UNVERIFIABLE",
                notes="No causal path — should refuse"))

    def _multihop(self):
        cases = [
            ("HYPOVOLEMIA","BP","Does HYPOVOLEMIA indirectly affect BP?"),
            ("LVFAILURE","BP","What is the indirect effect of LVFAILURE on BP?"),
            ("INTUBATION","SAO2","Does INTUBATION indirectly affect SAO2?"),
            ("PULMEMBOLUS","CATECHOL","Does PULMEMBOLUS indirectly affect CATECHOL?"),
            ("MINVOLSET","ARTCO2","What is the indirect causal effect of MINVOLSET on ARTCO2?"),
        ]
        for src,tgt,q in cases:
            if nx.has_path(self.dag,src,tgt):
                path = nx.shortest_path(self.dag,src,tgt)
                self.qa_pairs.append(CausalQAPair(
                    f"ALARM_MULTI_{src}_{tgt}",q,"intervention",
                    src,tgt,"yes_indirect","VERIFIED",path,
                    notes="Multi-hop"))

    def to_dataframe(self):
        if not self.qa_pairs: self.generate()
        return pd.DataFrame([vars(qa) for qa in self.qa_pairs])

    def summary(self):
        if not self.qa_pairs: self.generate()
        df = self.to_dataframe()
        return {
            "total": len(df),
            "by_type": df["query_type"].value_counts().to_dict(),
            "by_verdict": df["correct_verdict"].value_counts().to_dict(),
        }
