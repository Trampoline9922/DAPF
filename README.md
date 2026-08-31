# DAPF
# DAPF: Dual-View Adaptive Path Framework for Self-Supervised Heterogeneous Graph Representation Learning

This repository provides the source code and reproducibility materials for the paper:

**DAPF: A Dual-View Adaptive Path Framework for Self-Supervised Heterogeneous Graph Representation Learning**

DAPF is a self-supervised heterogeneous graph representation learning framework that jointly models multi-relational structural patterns and fine-grained meta-path instance semantics. The framework consists of an Adaptive Structure View (AS-View), a Semantic Path View (SP-View), and dual-view contrastive learning objectives.

The released code is intended to reproduce the node classification, node clustering, ablation, and sensitivity experiments reported in the paper.

---

1. Repository Structure

A recommended directory structure is shown below:

```text
DAPF/
├── main.py
├── model.py / DAPF.py
├── transformer_model.py
├── path_sampling.py
├── params.py
├── utils/
│   └── ...
├── scripts/
│   ├── run_acm.ps1
│   ├── run_dblp.ps1
│   ├── run_aminer.ps1
│   └── run_all_seeds.ps1
├── data/
│   └── README.md
├── results/
│   └── ...
├── requirements.txt
└── README.md

2. Requirements
The experiments are implemented in Python using PyTorch.
The exact software versions used for the final experiments will be provided in requirements.txt.
Main dependencies include:
Python
PyTorch
NumPy
SciPy
scikit-learn
pandas

To install the required packages, run:
pip install -r requirements.txt
The exact Python, PyTorch, CUDA, and GPU configurations used in the experiments will be reported in the final reproducibility package.

3. Datasets
Experiments are conducted on three widely used heterogeneous information network benchmarks:
ACM
DBLP
AMiner
The processed datasets follow the settings used in the GTC repository and related heterogeneous graph representation learning studies.
ACM
Target node type: Paper
Meta-paths:
PAP
PSP
The full-model path-instance budget is:
K = 9
The hidden and embedding dimensions are:
128
DBLP
Target node type: Author
Meta-paths:
APA
APCPA
APTPA
The full-model path-instance budget is:
K = 15
The hidden and embedding dimensions are:
64
AMiner
Target node type: Paper
Meta-paths:
PAP
PRP
The full-model path-instance budget is:
K = 9
The hidden and embedding dimensions are:
64
Since the raw AMiner benchmark does not provide initial node attributes under the adopted preprocessing setting, the provided processed input features follow the same feature construction used in the reference benchmark.

The exact processed data source, repository commit, and file checksums will be reported in the final release.
4. Meta-Path Instance Sampling
DAPF performs online schema-constrained meta-path instance sampling during representation learning.
For each target node and each meta-path:
At each traversal hop, at most three neighboring nodes are uniformly sampled without replacement.
Only node sequences satisfying the corresponding meta-path schema are retained.
The resulting schema-valid paths form the candidate path-instance set.
At most $K$ path instances are uniformly retained without replacement.
If fewer than $K$ valid instances are available, all available instances are retained without duplication.
If no valid instance is available, the aggregated path contribution of that semantic view is set to zero while the residual target-node representation is retained.
Path instances are dynamically resampled at every training epoch.
The common sampling settings are:
max_neighbors = 3
sample_ratio = 1.0
max_total_paths = None
The full-model path budgets are:
ACM:    K = 9
DBLP:   K = 15
AMiner: K = 9
5. Randomness and Reproducibility
Several stages of the experimental pipeline involve stochastic operations. The released implementation explicitly controls and records the corresponding random seeds.

5.1 Representation Learning
The main results are obtained from 10 independent representation-learning runs:
representation seeds = 0, 1, ..., 9
Each representation-learning seed controls the stochastic components associated with model initialization, neural optimization, and online path-instance sampling.
Reported classification and clustering results are finally summarized across these 10 representation-learning runs.

5.2 Online Path Sampling
Meta-path instances are not fixed before training. They are dynamically resampled at each epoch.
The random sampling process is controlled by the experiment seed to ensure that each complete experimental run can be reproduced.

5.3 Prototype Clustering
The semantic prototypes are updated during representation learning using MiniBatchKMeans on detached node embeddings.
The prototype clustering configuration is:
batch_size = 1024
max_iter = 20
n_init = 5
initialization = k-means++
reassignment_ratio = 0.01
Prototype-clustering randomness is explicitly seeded in the implementation and recorded together with the experiment configuration.
