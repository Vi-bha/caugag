# CauGAG — Causal Graph-Augmented Generation

CauGAG pairs an LLM with a formal causal-graph verifier (d-separation, backdoor
criterion) and a generation gate that conditions the model's response on the
verifier's verdict — so the model correctly **refuses** to assert causation
that isn't supported by the underlying causal graph, instead of hallucinating
a confident answer from surface-level correlation.

## PipelineThe verifier returns one of: `VERIFIED`, `UNVERIFIABLE`, `CONFOUNDED`,
`ASSOCIATED`. The gate uses this verdict to decide whether the LLM is allowed
to assert a causal claim or must refuse.

## Repo layout

- `caugag/` - core pipeline (`claim_parser.py`, `graph_verifier.py`, `gate.py`)
- `caugag/baselines/` - RAG and no-gate ablation baselines
- `caugag/benchmark/` - per-dataset benchmark generation + LLM-judge scoring
- `data/` - source data (e.g. Sachs protein-signaling dataset)
- `results/` - benchmark outputs, LLM-judge scores, statistical tests
- `figures/` - generated paper figures
- `app.py` - Gradio demo

## Benchmarks

Evaluated on four causal-graph datasets: Sachs protein signaling, a synthetic
economics SCM, the Alarm Bayesian network, and German Credit. See
`results/master_results.json` for headline numbers and
`results/reproducibility_config.json` for run configuration.

## Status

Experiments complete; paper in active writing/submission preparation.
