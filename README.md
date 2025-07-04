# AMN2 — Hybrid Neural Network for Dynamical FBA

**AMN2** is a hybrid machine learning model that combines:
- A neural network for metabolic flux inference, and
- A dynamical FBA to simulate metabolite and biomass evolution over time.

It is designed to predict **time-course biomass growth** under various **media conditions**, integrating stoichiometry and transport constraints from genome-scale metabolic models (GEMs).

---

## Project Structure

| File/Folder            | Description |
|------------------------|-------------|
| `train_AMN2.py`   | Script to train the model on a given dataset |
| `test_AMN2.py`    | Script to evaluate trained models |
| `utils.py`             | Core logic: model, data preprocessing, training, plotting |
| `data/`                | Input datasets: media, OD, and metabolic model (SBML) |
| `model/`               | Folder where trained models and validation arrays are stored |
| `figure/`              | Plots for training and testing curves |
---

## Setup and Environment

This project uses **Python ≥ 3.8**, TensorFlow 2.x, and COBRApy.

### Create the Conda environment (recommended)
To create the required environment from the `environment.yml` file:

```bash
conda env create -n amn2_env -f environment.yml
conda activate amn2_env
```