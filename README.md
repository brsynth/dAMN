# dAMN — Hybrid Neural Network for Dynamical FBA

**dAMN** is a hybrid machine learning model that combines:
- A neural network for metabolic flux inference;
- A dynamical FBA to simulate metabolite and biomass evolution over time

It is designed to predict **time-course biomass growth** under various **media conditions**, integrating stoichiometry and transport constraints from genome-scale metabolic models (GEMs).

Applied on [Millard et al. 2021](https://elifesciences.org/articles/63661) dataset in *E.coli*.

---

## Project Structure

| File/Folder            | Description |
|------------------------|-------------|
| `dAMN.ipynb`   | Notebook to parametrize and run the model with a given dataset |
| `train_dAMN.py`   | Script to train the model on a given dataset |
| `utils.py`             | Core functions: model, data preprocessing, training, plotting |
| `data/`                | Input datasets: media, OD, and metabolic model (SBML) |
| `model/`               | Folder where trained models and validation arrays are stored |
| `figure/`              | Plots for training and testing curves |
---

## Setup and Environment

This project uses **Python ≥ 3.8**, TensorFlow 2.x, and COBRApy.

### Create the Conda environment (recommended)
To recreate the required environment from the `environment.yml` file:

```bash
conda env create -n dAMN_env -f environment.yml
conda activate dAMN_env
```