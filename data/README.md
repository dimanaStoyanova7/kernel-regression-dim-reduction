# Superconductor Dataset

## Dataset Description

This folder contains the **Superconductor dataset** downloaded from Kaggle:

https://www.kaggle.com/datasets/munumbutt/superconductor-dataset

The dataset consists of a CSV file containing numerical descriptors of superconducting materials and a continuous target variable representing the **superconducting critical temperature (Tc)**.

This dataset is used for a **regression task**, where the goal is to predict the critical temperature from material features.

---

## Files

- `train.csv`  
  Main dataset containing:
  - One row per superconducting material
  - Multiple numerical feature columns
  - One target column representing the critical temperature

No dimensionality reduction or feature extraction has been applied to this dataset.

---

## Target Variable

- **Critical temperature (Tc)**  
  A continuous numerical value indicating the temperature (in Kelvin) below which the material becomes superconducting.

This variable is used as the regression target.

---

## Features

The remaining columns represent numerical material descriptors derived from elemental, physical, and chemical properties of the superconductors.

All features are numeric and suitable for:
- Standardization
- PCA (dimensionality reduction)
- Kernel-based regression methods

---

## Usage Notes

- The dataset should be loaded directly from the CSV file.
- Feature scaling is required before applying PCA or kernel methods.
- PCA will be applied during preprocessing to create a reduced feature space for comparison with the raw feature space.
- No train/test split is provided and must be created during preprocessing.

---

## Provenance

- Source: Kaggle – Superconductor Dataset (munumbutt)
- Original data derived from published superconductivity databases.
