from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import networkx as nx

class Verdict(str, Enum):
    VERIFIED = "VERIFIED"
    CONFOUNDED = "CONFOUNDED"
    UNVERIFIABLE = "UNVERIFIABLE"
    ASSOCIATED = "ASSOCIATED"

@dataclass
class VerificationResult:
    verdict: Verdict
    source_var: str
    target_var: str
    directed_path: list = field(default_factory=list)
    backdoor_paths: list = field(default_factory=list)
    adjustment_set: list = field(default_factory=list)
    identifiable: bool = False
    explanation: str = ""

class GraphVerifier:
    def __init__(self, dag: nx.DiGraph):
        if not nx.is_directed_acyclic_graph(dag):
            raise ValueError("Input graph is not a DAG.")
        self.dag = dag
        self.nodes = list(dag.nodes())

    def verify(self, query) -> VerificationResult:
        from caugag.claim_parser import QueryType
        source, target = query.source_var, query.target_var
        if source not in self.nodes:
            return VerificationResult(Verdict.UNVERIFIABLE, source, target,
                explanation=f"Variable {source} not in DAG.")
        if target not in self.nodes:
            return VerificationResult(Verdict.UNVERIFIABLE, source, target,
                explanation=f"Variable {target} not in DAG.")
        if query.query_type == QueryType.ASSOCIATION:
            if self._has_common_cause(source, target):
                return self._check_confounding(source, target)
            return self._check_association(source, target)
        elif query.query_type in (QueryType.INTERVENTION, QueryType.COUNTERFACTUAL):
            return self._check_causal_identifiability(source, target)
        else:
            return VerificationResult(Verdict.UNVERIFIABLE, source, target,
                explanation="Unknown query type.")

    def _has_common_cause(self, source, target):
        parents_src = set(self.dag.predecessors(source))
        parents_tgt = set(self.dag.predecessors(target))
        return bool(parents_src & parents_tgt)

    def _check_confounding(self, source, target):
        parents_src = set(self.dag.predecessors(source))
        parents_tgt = set(self.dag.predecessors(target))
        common = list(parents_src & parents_tgt)
        return VerificationResult(Verdict.CONFOUNDED, source, target,
            identifiable=False, adjustment_set=common,
            explanation=f"CONFOUNDED. Common causes: {common}.")

    def _check_association(self, source, target):
        skeleton = self.dag.to_undirected()
        if nx.has_path(skeleton, source, target):
            path = nx.shortest_path(skeleton, source, target)
            return VerificationResult(Verdict.ASSOCIATED, source, target,
                directed_path=path, identifiable=True,
                explanation=f"Association path: {str(path)}")
        return VerificationResult(Verdict.UNVERIFIABLE, source, target,
            explanation=f"No path between {source} and {target}.")

    def _check_causal_identifiability(self, source, target):
        if not nx.has_path(self.dag, source, target):
            return VerificationResult(Verdict.UNVERIFIABLE, source, target,
                explanation=f"No directed path from {source} to {target} in DAG.")
        directed_path = nx.shortest_path(self.dag, source, target)
        backdoor_paths = self._find_backdoor_paths(source, target)
        adjustment_set = self._find_adjustment_set(source, target)
        if not backdoor_paths:
            return VerificationResult(Verdict.VERIFIED, source, target,
                directed_path=directed_path, identifiable=True,
                explanation=f"VERIFIED. Path: {str(directed_path)}. No confounders.")
        elif adjustment_set is not None:
            return VerificationResult(Verdict.VERIFIED, source, target,
                directed_path=directed_path, backdoor_paths=backdoor_paths,
                adjustment_set=adjustment_set, identifiable=True,
                explanation=f"VERIFIED (adjustable). Path: {str(directed_path)}. Adjust for: {adjustment_set}.")
        else:
            return VerificationResult(Verdict.CONFOUNDED, source, target,
                directed_path=directed_path, backdoor_paths=backdoor_paths,
                identifiable=False,
                explanation=f"CONFOUNDED. Backdoor paths: {str(backdoor_paths)}.")

    def _find_backdoor_paths(self, source, target):
        parents = list(self.dag.predecessors(source))
        backdoor = []
        for parent in parents:
            tmp = self.dag.copy()
            for s in list(self.dag.successors(source)):
                tmp.remove_edge(source, s)
            if nx.has_path(tmp, parent, target):
                path = nx.shortest_path(tmp, parent, target)
                backdoor.append([source] + path)
        return backdoor

    def _find_adjustment_set(self, source, target):
        parents = list(self.dag.predecessors(source))
        backdoor = self._find_backdoor_paths(source, target)
        if not backdoor:
            return []
        if parents:
            return parents
        return None
