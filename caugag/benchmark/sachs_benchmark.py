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
        self._direction_false_reversed()
        self._direction_false_nonedge()
        self._confounding()
        self._intervention_true()
        self._intervention_false()
        self._multihop()
        self._association()
        self._counterfactual()
        return self.qa_pairs

    def _direction_true(self):
        templates = [
            "Does {s} directly cause {t}?",
            "Is there a direct causal effect of {s} on {t}?",
            "Does {s} directly affect {t}?",
        ]
        for i,(src,tgt) in enumerate(SACHS_EDGES):
            q = templates[i%len(templates)].format(s=src,t=tgt)
            self.qa_pairs.append(CausalQAPair(
                f"DIR_TRUE_{src}_{tgt}",q,"direction",
                src,tgt,"yes","VERIFIED",[src,tgt]))

    def _direction_false_reversed(self):
        templates = [
            "Does {s} directly cause {t}?",
            "Is there a direct causal effect of {s} on {t}?",
            "Does {s} directly affect {t}?",
        ]
        for i,(src,tgt) in enumerate([(v,u) for u,v in SACHS_EDGES]):
            q = templates[i%len(templates)].format(s=src,t=tgt)
            self.qa_pairs.append(CausalQAPair(
                f"DIR_REV_{src}_{tgt}",q,"direction",
                src,tgt,"no","UNVERIFIABLE",notes="Reversed edge"))

    def _direction_false_nonedge(self):
        non_edges = [
            ("JNK","Erk"),("P38","Akt"),("PIP2","Mek"),("Akt","Plcg"),
            ("JNK","PIP3"),("P38","Raf"),("Erk","Plcg"),("Akt","PIP2"),
            ("JNK","Akt"),("P38","Mek"),
        ]
        for src,tgt in non_edges:
            self.qa_pairs.append(CausalQAPair(
                f"DIR_NONE_{src}_{tgt}",
                f"Does {src} directly cause {tgt}?",
                "direction",src,tgt,"no","UNVERIFIABLE",notes="No edge"))

    def _confounding(self):
        cases = [
            ("Raf","Akt","Is the relationship between Raf and Akt confounded?"),
            ("Mek","Akt","Is the relationship between Mek and Akt confounded?"),
            ("Raf","Erk","Is the naive association between Raf and Erk confounded?"),
            ("Mek","JNK","Is there confounding between Mek and JNK?"),
            ("Raf","P38","Is the relationship between Raf and P38 confounded?"),
        ]
        for src,tgt,q in cases:
            parents_src = set(self.dag.predecessors(src))
            parents_tgt = set(self.dag.predecessors(tgt))
            common = parents_src & parents_tgt
            verdict = "CONFOUNDED" if common else "VERIFIED"
            self.qa_pairs.append(CausalQAPair(
                f"CONF_{src}_{tgt}",q,"confounding",
                src,tgt,"yes_confounded",verdict,
                notes=f"Common causes: {common}"))

    def _intervention_true(self):
        cases = [
            ("Mek","Erk","What happens to Erk if we intervene on Mek?"),
            ("PKA","Akt","If we intervene on PKA, what happens to Akt?"),
            ("PKC","Erk","What is the causal effect of PKC on Erk?"),
            ("PIP3","Akt","What happens to Akt if PIP3 is experimentally increased?"),
            ("Raf","Erk","What is the causal effect of Raf on Erk?"),
            ("PKA","Erk","What happens to Erk when PKA is experimentally manipulated?"),
            ("PKC","Mek","What is the interventional effect of PKC on Mek?"),
            ("Plcg","PIP2","If Plcg is experimentally increased, what happens to PIP2?"),
            ("PKA","Mek","What is the causal effect of PKA on Mek?"),
            ("PKC","Raf","What happens to Raf if we do(PKC=high)?"),
        ]
        for src,tgt,q in cases:
            if nx.has_path(self.dag,src,tgt):
                path = nx.shortest_path(self.dag,src,tgt)
                self.qa_pairs.append(CausalQAPair(
                    f"INTERV_{src}_{tgt}",q,"intervention",
                    src,tgt,"yes_causal_effect","VERIFIED",path))

    def _intervention_false(self):
        cases = [
            ("Erk","PKC","What happens to PKC if we intervene on Erk?"),
            ("Akt","Raf","What is the effect of intervening on Akt on Raf?"),
            ("JNK","Mek","If we intervene on JNK, what happens to Mek?"),
            ("P38","Erk","What happens to Erk if we experimentally increase P38?"),
            ("Akt","PKA","What is the causal effect of Akt on PKA?"),
            ("Mek","PKC","If we intervene on Mek, what happens to PKC?"),
            ("Erk","Raf","What happens to Raf when Erk is experimentally manipulated?"),
        ]
        for src,tgt,q in cases:
            self.qa_pairs.append(CausalQAPair(
                f"INTERV_FALSE_{src}_{tgt}",q,"intervention",
                src,tgt,"no_causal_effect","UNVERIFIABLE",
                notes="No causal path"))

    def _multihop(self):
        cases = [
            ("PKC","Erk","Does PKC indirectly cause Erk through Mek?"),
            ("Plcg","Akt","Does Plcg indirectly cause Akt through PIP3?"),
            ("PKA","Akt","Does PKA indirectly affect Akt via Erk?"),
            ("Plcg","PIP2","Does Plcg cause PIP2 through PIP3?"),
            ("PKC","Akt","What is the indirect causal effect of PKC on Akt?"),
        ]
        for src,tgt,q in cases:
            if nx.has_path(self.dag,src,tgt):
                path = nx.shortest_path(self.dag,src,tgt)
                self.qa_pairs.append(CausalQAPair(
                    f"MULTIHOP_{src}_{tgt}",q,"intervention",
                    src,tgt,"yes_indirect","VERIFIED",path,
                    notes="Multi-hop"))

    def _association(self):
        cases = [
            ("Raf","Erk","Is Raf statistically associated with Erk?"),
            ("JNK","P38","Is JNK correlated with P38?"),
            ("PIP2","Akt","Is PIP2 associated with Akt?"),
            ("Mek","Akt","Is there a statistical association between Mek and Akt?"),
        ]
        for src,tgt,q in cases:
            skeleton = self.dag.to_undirected()
            connected = nx.has_path(skeleton,src,tgt)
            self.qa_pairs.append(CausalQAPair(
                f"ASSOC_{src}_{tgt}",q,"association",
                src,tgt,
                "yes_associated" if connected else "no",
                "ASSOCIATED" if connected else "UNVERIFIABLE"))

    def _counterfactual(self):
        cases = [
            ("PKA","Raf","If PKA had been higher, would Raf have changed?"),
            ("Mek","Erk","If Mek had been inhibited, would Erk have been lower?"),
            ("PKC","JNK","Had PKC been absent, would JNK have been affected?"),
            ("PIP3","Akt","If PIP3 had been blocked, would Akt have changed?"),
            ("Erk","PKC","If Erk had been higher, would PKC have changed?"),
        ]
        for src,tgt,q in cases:
            has_path = nx.has_path(self.dag,src,tgt)
            self.qa_pairs.append(CausalQAPair(
                f"CF_{src}_{tgt}",q,"counterfactual",
                src,tgt,
                "yes_counterfactual" if has_path else "no",
                "VERIFIED" if has_path else "UNVERIFIABLE",
                nx.shortest_path(self.dag,src,tgt) if has_path else []))

    def to_dataframe(self):
        if not self.qa_pairs: self.generate()
        return pd.DataFrame([vars(qa) for qa in self.qa_pairs])

    def summary(self):
        if not self.qa_pairs: self.generate()
        df = self.to_dataframe()
        return {"total":len(df),
                "by_type":df["query_type"].value_counts().to_dict(),
                "by_verdict":df["correct_verdict"].value_counts().to_dict()}
