# pka-ml-prediction
Machine learning models for pKa classification and prediction using RDKit descriptors, Gradient Boosting and XGBoost.

# Ordinal Classification and Regression of pKa Values using Machine Learning

This repository contains the computational framework and notebooks developed in the context of an MBA project in Data Science, focused on the prediction and classification of acid dissociation constants (pKa) of organic compounds using machine learning techniques.

The study investigates structure–property relationships by combining physicochemical molecular descriptors with tree-based models and ordinal classification approaches, aiming to support compound selection for CO₂ conversion and sustainable chemistry applications.

---

## Project Overview

The rational prediction of pKa values remains a challenging task due to the complex interplay between molecular structure, electronic effects, and solvation phenomena. In this project, two chemically distinct datasets were analyzed:

- **IUPAC Digitized pKa Dataset**
- **SAMPL6 pKa Challenge Dataset**

Molecular descriptors were generated using **RDKit**, followed by exploratory statistical analysis, dimensionality reduction, and supervised learning.

The work compares:
- **Ordinal classification** of pKa ranges
- **Direct regression** of continuous pKa values

Results indicate that ordinal classification provides more robust and chemically interpretable predictions than regression using global molecular descriptors.

---

## Datasets

- **IUPAC Digitized pKa Dataset**
- **SAMPL6 pKa Challenge**

Both datasets include experimentally measured pKa values and molecular structures represented as SMILES strings.

> Note: Original datasets are not redistributed here. Please refer to the respective official sources.

---

## Methods and Models

### Molecular Descriptors
Generated using **RDKit**, including:
- TPSA
- LogP
- Number of rotatable bonds
- Ring count
- Hydrogen bond donors and acceptors
- Additional physicochemical descriptors

### Machine Learning Models

#### Classification (Ordinal)
- Gradient Boosting Classifier (GB)
- eXtreme Gradient Boosting (XGBoost)
- Ordinal Logistic Regression (RLO)

#### Regression (Benchmarking)
- Gradient Boosting Regressor (GBR)
- XGBoost Regressor
- Decision Tree Regressor

---

## Evaluation Metrics

### Classification
- Multiclass ROC-AUC (One-vs-Rest)
- Weighted F1-score
- Matthews Correlation Coefficient (MCC)

### Regression
- RMSE
- MAE
- R²

---

## Key Findings

- Ordinal classification consistently outperformed direct regression models.
- Feature importance analysis revealed chemically meaningful predictors (e.g., TPSA, flexibility, ring count).
- Regression models were strongly affected by outliers, especially in the SAMPL6 dataset.
- Tree-based regression models failed to outperform naive baselines.
- Ordinal classification mitigated the influence of extreme pKa values by discretization.

---

## Dimensionality Reduction

- Principal Component Analysis (PCA)
- Kernel PCA (Appendix)

Kernel PCA was used for exploratory visualization only, as it does not provide a direct explained variance ratio.

---

## Technologies

- Python 3
- RDKit
- scikit-learn
- XGBoost
- NumPy / Pandas
- Matplotlib / Seaborn

---

## Repository Structure (in construction)

```text
.
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   ├── classification_models.ipynb
│   ├── regression_models.ipynb
│   └── pca_kernel_pca.ipynb
├── figures/
│   ├── classification_results/
│   ├── regression_results/
│   └── pca_plots/
├── requirements.txt
└── README.md
