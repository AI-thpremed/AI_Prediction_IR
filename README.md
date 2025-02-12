# AI-Driven Insulin Resistance Prediction

This repository contains the code and data for the research paper titled "AI-driven Prediction of Insulin Resistance in Normal Populations: Comparing Models and Criteria" . The project aims to develop AI-based models for predicting insulin resistance using minimal invasive tests and easily accessible features, focusing on non-diabetic populations.

## Repository Structure

- **`catboost_nhanes-homa`**: Checkpoints for CatBoost models trained on the NHANES dataset for HOMA-IR classification.
- **`catboost_nhanes-mets-loose`**: Checkpoints for CatBoost models trained on the NHANES dataset for METS-IR classification with a loose threshold (41.33).
- **`catboost_nhanes-mets-strict`**: Checkpoints for CatBoost models trained on the NHANES dataset for METS-IR classification with a strict threshold (51.13).
- **`catboost_nhanes-tyg`**: Checkpoints for CatBoost models trained on the NHANES dataset for TyG index classification.
- **`data`**: A sample of test data used for validation and testing of the models.
- **`tabkanet`**: Source code for the TabKANet model implementation.
- **`tabkanet_reg`**: Checkpoints for TabKANet models trained for numerical prediction of METS-IR values.


## Data and Validation
The models were developed using data from the National Health and Nutrition Examination Survey (NHANES) from 1999 to 2020. Additionally, the models were validated using cross-national and cross-database validation on the China Health and Retirement Longitudinal Study (CHARLS) dataset. This validation ensures the robustness and generalizability of the models across diverse populations.

## Deployment

The provided checkpoints can be easily deployed to predict different insulin resistance assessment criteria for the general population. We hope these models will facilitate the early detection and management of insulin resistance, thereby improving health outcomes and reducing the risk of diabetes and cardiovascular diseases.




## About the Models

### CatBoost Models
CatBoost is a powerful gradient boosting framework that achieved excellent performance in binary classification tasks for insulin resistance assessment. The models in this repository are trained on the NHANES dataset and validated on both internal and external datasets (CHARLS). The checkpoints provided can be directly used for deployment.

### TabKANet Models
TabKANet is a neural network-based model that integrates numerical features using a Kolmogorov-Arnold Network and Transformer architecture. It demonstrated superior performance in predicting METS-IR values, achieving high R2 and low RMSE scores. The model code and trained checkpoints are included for both internal and external validation datasets.

## Usage

### Requirements
- Python 3.8+
- CatBoost
- PyTorch
- Scikit-learn
- Pandas
- NumPy

### Installation
```bash
pip install catboost torch scikit-learn pandas numpy


