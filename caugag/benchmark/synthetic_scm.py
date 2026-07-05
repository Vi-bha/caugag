from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd
import networkx as nx


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


class SyntheticSCM:
    def __init__(self, n_nodes=15, edge_density=0.3, seed=42):
        self.n_nodes = n_nodes
        self.edge_density = edge_density
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.nodes = [f"V{i}" for i in range(1, n_nodes+1)]
        self.dag = None
        self.beta = None
        self.data = None

    def generate_dag(self):
        dag = nx.DiGraph()
        dag.add_nodes_from(self.nodes)
        n = self.n_nodes
        for i in range(n):
            for j in range(i+1, n):
                if self.rng.random() < self.edge_density:
                    dag.add_edge(self.nodes[i], self.nodes[j])
        assert nx.is_directed_acyclic_graph(dag)
        self.dag = dag
        return dag

    def assign_coefficients(self):
        beta = {}
        for u, v in self.dag.edges():
            coef = self.rng.uniform(0.2, 1.0)
            if self.rng.random() < 0.5:
                coef = -coef
            beta[(u, v)] = coef
        self.beta = beta
        return beta

    def generate_data(self, n_samples=1000):
        topo_order = list(nx.topological_sort(self.dag))
        node_idx = {node: i for i, node in enumerate(self.nodes)}
        data = np.zeros((n_samples, self.n_nodes))
        for node in topo_order:
            idx = node_idx[node]
            parents = list(self.dag.predecessors(node))
            noise = self.rng.normal(0, 1, n_samples)
            value = noise.copy()
            for parent in parents:
                p_idx = node_idx[parent]
                value += self.beta[(parent, node)] * data[:, p_idx]
            data[:, idx] = value
        self.data = pd.DataFrame(data, columns=self.nodes)
        return self.data

    def get_true_total_effect(self, source, target):
        if not nx.has_path(self.dag, source, target):
            return 0.0
        all_paths = list(nx.all_simple_paths(self.dag, source, target))
        total = 0.0
        for path in all_paths:
            effect = 1.0
            for i in range(len(path)-1):
                effect *= self.beta.get((path[i], path[i+1]), 0.0)
            total += effect
        return total

    def generate_qa_pairs(self, n_pairs=40):
        qa_pairs = []
        edges = list(self.dag.edges())
        non_edges = [
            (u, v) for u in self.nodes for v in self.nodes
            if u != v and not self.dag.has_edge(u, v) and not self.dag.has_edge(v, u)
        ]
        reversed_edges = [(v, u) for u, v in edges]

        # TRUE causal — direct edges
        for i, (src, tgt) in enumerate(edges[:10]):
            qa_pairs.append(CausalQAPair(
                f"SYN_TRUE_{i}", f"Does {src} directly cause {tgt}?",
                "direction", src, tgt, "yes", "VERIFIED", [src, tgt],
                notes=f"beta={self.beta.get((src,tgt),0):.3f}"))

        # FALSE — reversed edges
        self.rng.shuffle(reversed_edges)
        for i, (src, tgt) in enumerate(reversed_edges[:8]):
            qa_pairs.append(CausalQAPair(
                f"SYN_REV_{i}", f"Does {src} directly cause {tgt}?",
                "direction", src, tgt, "no", "UNVERIFIABLE",
                notes="Reversed edge"))

        # FALSE — non edges
        self.rng.shuffle(non_edges)
        for i, (src, tgt) in enumerate(non_edges[:8]):
            qa_pairs.append(CausalQAPair(
                f"SYN_NONE_{i}", f"What is the causal effect of {src} on {tgt}?",
                "intervention", src, tgt, "no_causal_effect", "UNVERIFIABLE",
                notes="No edge"))

        # Intervention with true effect
        for i, (src, tgt) in enumerate(edges[:8]):
            effect = self.get_true_total_effect(src, tgt)
            qa_pairs.append(CausalQAPair(
                f"SYN_EFFECT_{i}", f"What is the causal effect of {src} on {tgt}?",
                "intervention", src, tgt, f"effect={effect:.3f}", "VERIFIED",
                [src, tgt], notes=f"True effect={effect:.3f}"))

        return qa_pairs

    def get_dag_dot(self):
        edges_str = "; ".join([f"{u} -> {v}" for u, v in self.dag.edges()])
        return f"digraph {{ {edges_str} }}"

    @classmethod
    def build(cls, n_nodes=15, edge_density=0.3, n_samples=1000, seed=42):
        scm = cls(n_nodes=n_nodes, edge_density=edge_density, seed=seed)
        scm.generate_dag()
        scm.assign_coefficients()
        scm.generate_data(n_samples)
        return scm
