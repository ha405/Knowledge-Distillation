# Knowledge-Distillation — Project Overview

This repository contains code and notebooks used for experimenting with knowledge distillation techniques on a vision model (VGG variants). The notebooks contain the experiments, discussions, and plots. The main scripts and modules provide utilities, model definitions, training/evaluation loops, and utilities for distillation.

## Task to Notebook mapping

- Task 1 — Logit Matching: `Logit_Matching.ipynb`
  - Implements logit-matching distillation experiments.
  - Contains discussion of results and in-notebook plots.

- Task 2 — Limitations of Logit Matching: `LogitMatching_Limitations.ipynb`
  - Explores limitations and failure modes of logit-matching.
  - Includes discussion and diagnostic plots.

- Task 3 — Feature Distillation: `Feature_Distillation.ipynb`
  - Experiments and discussion on distillation via intermediate feature matching.
  - Plots and comparisons are included in the notebook.

- Task 4 — Multiple Students: `Multiple_Students.ipynb`
  - Investigates distilling into multiple student networks and ensemble effects.
  - Discussions and visualizations live inside the notebook.

Each notebook contains prose discussion, experiment code, and result plots — see the notebooks for full context and figures.

## Repository structure (top-level files)

- `data_utils.py` — Data loading and preprocessing helper functions.
- `feature_kd.py` — Utilities and loss functions for feature-based knowledge distillation.
- `model.py` — Model definitions and helper constructors (student/teacher definitions used in experiments).
- `vgg.py` — VGG model definitions and any VGG-specific variants used in experiments.
- `train_kd.py` — Training loop for knowledge distillation experiments (handles distillation losses / schedules).
- `train_eval.py` — Training and evaluation utilities; metrics and evaluation loops.
- `logit_matching` notebooks: see notebooks listed above.

## How to run

1. Open the desired notebook in Jupyter or VS Code (e.g. open `Logit_Matching.ipynb`). Each notebook contains code cells to (re)run experiments and reproduce plots.
2. Ensure you have the required Python environment installed (common libs: `torch`, `torchvision`, `numpy`, `matplotlib`, `pandas`, etc.). If you keep an environment file, install dependencies with your usual tool.

Quick start (example): open a terminal in this folder and start Jupyter:

```powershell
python -m notebook
```

Then open the notebook file from the browser or VS Code and run the cells.

## Notes

- The notebooks are self-contained: they include both the experimental code and written discussion of the results, plus plots for visualization.

