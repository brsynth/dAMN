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
| `train_AMN2_{Paul/Millard}.py`   | Script to train the model on a given dataset |
| `test_AMN2_{Paul/Millard}.py`    | Script to evaluate trained models |
| `utils.py`             | Core logic: model, data preprocessing, training, plotting |
| `data/`                | Input datasets: media, OD, and metabolic model (SBML) |
| `model/`               | Folder where trained models and validation arrays are stored |
| `figure/`              | Plots for training and testing curves |
|`library/` | contaning the needed py file(s) to run dAMN |
|`scripts/` | contaning the test and the training scripts for both available datasets|

---

## Setup and Environment

This project uses **Python ≥ 3.8**, TensorFlow 2.x, and COBRApy.

### Create the Conda environment (recommended)
To recreate the required environment from the `environment.yml` file:

```bash
conda env create -n amn2_env -f environment.yml
conda activate amn2_env
```

## Contributor(s)
**Jean-Loup Faulon**: Implementation

Email: jean-loup.faulon@inrae.fr

**Danilo Dursoniah**: Maintenance

Email: danilo.dursoniah@inrae.fr