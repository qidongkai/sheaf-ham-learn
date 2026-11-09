# sheaf-ham-learn
Sheaf-Cohomological Learning (Reference Implementation)
Reference implementation for the paper:

Qi, Huang, Ye. Sheaf-Cohomological Learning: Gluing Local Hamiltonian Fragments into Global Hamiltonians.

This repository provides the code, datasets, and experiment scripts needed to reproduce the results reported in the paper. It implements the cellular-sheaf formulation of Hamiltonian fragment gluing, the strict Čech-cohomological criterion, the differentiable cohomological proxy used for gradient training, the cellular equivariant graph neural network, and the numerical experiments on one- and two-dimensional non-Hermitian lattice systems.

Installation
bash
pip install -r requirements.txt
Quick Start
bash
python scripts/run_experiment.py --config configs/ising_1d.yaml
python scripts/run_noise_scan.py --config configs/ising_1d.yaml
python scripts/run_delta_scan.py --config configs/ising_1d.yaml
python scripts/plot_results.py --outdir results/
The four commands above run, in order: the main reconstruction experiment, the noise-robustness scan, the incompatibility (delta) scan on artificially constructed incompatible datasets, and the result plotting utility.

Repository Layout and Correspondence to the Paper
The table below maps each module in this repository to the corresponding section, theorem, or equation in the paper.

Module	Paper location
sheaf_ham/models.py	Sec. 5.1.1, Table 5.1
sheaf_ham/complex.py	Sec. 3.1.1
sheaf_ham/sheaf.py	Sec. 3.1.2
sheaf_ham/cech.py	Sec. 3.2.2, Sec. 4.2.1
sheaf_ham/cohomology.py	Sec. 3.3, Theorem 1, Eq. (27)–(29)
sheaf_ham/gnn.py	Sec. 4.1
sheaf_ham/losses.py	Sec. 4.3.1, Eq. (33)–(34)
sheaf_ham/datasets.py	Sec. 5.1.2–5.1.3, Table 5.3–5.4
sheaf_ham/train.py	Sec. 4.3.3
scripts/run_*.py	Sec. 5.2–5.5
tests/	Assumption 1–2, Theorem 1
Package Structure
text
sheaf_ham/
├── data/
│   ├── __init__.py
│   ├── fragments.py      # Slice global H into {H_σ} (0/1/2-cells)
│   ├── observables.py    # Trajectories / scattering / local measurements
│   └── generators.py     # Constructors for the five dataset families in Table 5.3
└── datasets.py           # Legacy convenience entry point; delegates internally to data/
Notes on the data subpackage:

data/fragments.py implements the decomposition of a global Hamiltonian into local operator fragments attached to 0-, 1-, and 2-cells of the cellular complex, matching the fragment construction used throughout the paper.

data/observables.py generates the three observation modalities used as network inputs: dynamical trajectories, scattering S-matrices, and local measurement data.

data/generators.py constructs the five dataset families described in Table 5.3.

sheaf_ham/datasets.py is retained as a legacy convenience entry point; it delegates internally to the data/ subpackage and is kept for backward compatibility with older scripts.

Explicit Limitations
The following limitations are stated explicitly in Sec. 6.3 of the paper. They are reproduced here so that users of this code do not mistake the proxy for the strict invariant.

The proxy dist(omega, B^1) is not equal to the strict H^1 criterion.

Small-amplitude non-trivial cocycles can be missed (Table 5.8).

False positives occur under high noise (Table 5.8).

The physical screening condition G_phys is not a subsheaf; it can only be checked on global sections.

Reproducing the Paper's Results
Install dependencies with pip install -r requirements.txt.

Run the main experiment for a given configuration, e.g. python scripts/run_experiment.py --config configs/ising_1d.yaml.

Run the noise-robustness scan with scripts/run_noise_scan.py and the incompatibility scan with scripts/run_delta_scan.py, both taking the same configuration file.

Aggregate and plot results with python scripts/plot_results.py --outdir results/.

The tests/ directory contains tests corresponding to Assumptions 1–2 and Theorem 1 of the paper; these can be run to verify that the implementation satisfies the stated mathematical assumptions.

Scope of This Release
This is a reference implementation intended for reproducibility of the published results. It is not a general-purpose library for sheaf-cohomological learning, and it does not claim that the differentiable proxy is equivalent to the strict sheaf-cohomological invariant. As stated in the paper, the proxy is positioned as a differentiable, computable auxiliary diagnostic tool.
