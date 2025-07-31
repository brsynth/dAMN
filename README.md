# dAMN — Hybrid Neural Network for Dynamical FBA


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16635533.svg)](https://doi.org/10.5281/zenodo.16635533)

> First usable release of the dAMN software.  
> This version corresponds to the implementation used to generate results for the associated publication.

## Overview 

**dAMN** is a hybrid machine learning model that combines:
- A neural network for metabolic flux inference;
- A dynamical FBA to simulate metabolite and biomass evolution over time

It is designed to predict **time-course biomass growth** under various **media conditions**, integrating stoichiometry and transport constraints from genome-scale metabolic models (GEMs).

Applied on E.coli dataset.

---

## Project Structure

| File/Folder               | Description                                                    |
|---------------------------|----------------------------------------------------------------|
| `dAMN.ipynb`              | Notebook to train, test and  parametrize the model with a given dataset |
| `dAMN_train.py`           | Script to train the model on a given dataset                   |
| `dAMN_test.ipynb`         | Notebook to test and visualize the prediction                  |
| `dAMN_parameter_search.py`| Script for model parametrization                            |
| `utils.py`                | Core functions for data preprocessing, model training, testing and plotting  |
| `data/`                   | Input datasets: media, OD, and metabolic model (SBML)          |
| `model/`                  | Folder where trained models and validation arrays are stored   |
| `figure/`                 | Plots for training and testing curves                          |
| `environment.yml` | conda environment configuration file |
---

## Setup and Environment

This project uses **Python ≥ 3.8**, TensorFlow 2.19.0, and COBRApy.

### Create the Conda environment (recommended)
To recreate the required environment from the `environment.yml` file:

```bash
conda env create -n dAMN_env -f environment.yml
conda activate dAMN_env
```

### Contributors

**Pr. Jean-Loup Faulon (jean-loup.faulon@inrae.fr)**: conceptualization, coding, modeling
 
**Danilo Dursoniah, Postdoc (danilo.dursoniah@inrae.fr)**: testing, maintenance