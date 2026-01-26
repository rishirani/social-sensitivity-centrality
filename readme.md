# Social Sensitivity Centrality — code for L‑CSS letter

Code accompanying the L‑CSS letter:
"Social Sensitivity Centrality: Dynamical Centrality Metrics For Friedkin‑Johnson Social Networks via Riccati Perturbation Analysis".

Purpose
-------
Reproduce the numerical experiments and figures comparing standard network centralities with the proposed social‑sensitivity (SSC) metrics derived from Riccati perturbation analysis on Friedkin‑Johnson opinion dynamics. The Zachary's Karate Club graph is used as the running example.

Repository layout
-----------------
- main.py — primary script: constructs graphs, computes traditional centralities and SSC metrics, and generates figures.
- README.md — this file.
- ssc_comparison.png — saved comparison figure (traditional vs SSC metrics).
- topology.png — annotated topology figure used in the letter.

Quick start
-----------
Requirements:
- Python 3.7+
- numpy
- scipy
- networkx
- matplotlib

Install:
```sh
pip install numpy scipy networkx matplotlib
```

Run:
```sh
python main.py
```
Expected output (saved to repository):
- ssc_comparison.png
- topology.png

Key functions and files
-----------------------
- compute_social_sensitivities(...) in main.py — computes influence, stubbornness, resilience, actuation via Riccati and Lyapunov solves.
- normalize(...) in main.py — min‑max normalization helper used for plotting.
- Final plotting section in main.py — generates the figures used in the L‑CSS letter.

Reproducibility notes
---------------------
- The SSC metrics are computed for a directed version of the Karate Club graph inside main.py.
- Key parameters (e.g., self‑reliance lambda, Riccati tolerances, Lyapunov settings) are defined in main.py and can be adjusted.
- Set RNG seeds in main.py before layout/visualization calls for deterministic figures.

Citation
--------
If you use this code or reproduce results, please cite the L‑CSS letter:
"Social Sensitivity Centrality: Dynamical Centrality Metrics For Friedkin‑Johnson Social Networks via Riccati Perturbation Analysis" (L‑CSS letter).

License
-------
No license file included. Add a LICENSE if you intend to permit public reuse or redistribution.
```// filepath: /Users/rishir/Desktop/Work/PhD_code_repos/SSC_code/README.md
// ...existing code...
# Social Sensitivity Centrality — code for L‑CSS letter

Code accompanying the L‑CSS letter:
"Social Sensitivity Centrality: Dynamical Centrality Metrics For Friedkin‑Johnson Social Networks via Riccati Perturbation Analysis".

Purpose
-------
Reproduce the numerical experiments and figures comparing standard network centralities with the proposed social‑sensitivity (SSC) metrics derived from Riccati perturbation analysis on Friedkin‑Johnson opinion dynamics. The Zachary's Karate Club graph is used as the running example.

Repository layout
-----------------
- main.py — primary script: constructs graphs, computes traditional centralities and SSC metrics, and generates figures.
- README.md — this file.
- ssc_comparison.png — saved comparison figure (traditional vs SSC metrics).
- topology.png — annotated topology figure used in the letter.

Quick start
-----------
Requirements:
- Python 3.7+
- numpy
- scipy
- networkx
- matplotlib

Install:
```sh
pip install numpy scipy networkx matplotlib
```

Run:
```sh
python main.py
```
Expected output (saved to repository):
- ssc_comparison.png
- topology.png

Key functions and files
-----------------------
- compute_social_sensitivities(...) in main.py — computes influence, stubbornness, resilience, actuation via Riccati and Lyapunov solves.
- normalize(...) in main.py — min‑max normalization helper used for plotting.
- Final plotting section in main.py — generates the figures used in the L‑CSS letter.

Reproducibility notes
---------------------
- The SSC metrics are computed for a directed version of the Karate Club graph inside main.py.
- Key parameters (e.g., self‑reliance lambda, Riccati tolerances, Lyapunov settings) are defined in main.py and can be adjusted.
- Set RNG seeds in main.py before layout/visualization calls for deterministic figures.

Citation
--------
If you use this code or reproduce results, please cite the L‑CSS letter:
"Social Sensitivity Centrality: Dynamical Centrality Metrics For Friedkin‑Johnson Social Networks via Riccati Perturbation Analysis" (L‑CSS letter).

License
-------
No license file included. Add a LICENSE if you intend to permit public reuse or redistribution.