### baseline  kaggle submit Accuracy: 0.76794


================================================================================
### MODEL COMPARISON (18:56 14-06-2026)
================================================================================

**Run Configuration:**
- **Seed:** `1415926536`
- **CV Folds:** `2`
- **Ensemble Type:** `voting`
- **Optuna Trials (Default):** `100`
- **Optuna Trials (Complex):** `300`

| Rank | Model | CV Accuracy Mean | Delta from Best | CV Accuracy Std | Test Accuracy | Test Precision | Test Recall | Test F1 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **XGBoost** | 0.8618 | +0.0000 | 0.0066 | 0.7873 | 0.7500 | 0.6699 | 0.7077 |
| 2 | CatBoost | 0.8570 | -0.0048 | 0.0034 | 0.7985 | 0.7753 | 0.6699 | 0.7188 |
| 3 | Ensemble (voting) | 0.8555 | -0.0063 | 0.0099 | 0.8060 | 0.7684 | 0.7087 | 0.7374 |
| 4 | LightGBM | 0.8530 | -0.0088 | 0.0018 | 0.7873 | 0.7556 | 0.6602 | 0.7047 |
| 5 | Random Forest | 0.8466 | -0.0153 | 0.0019 | 0.8209 | 0.8235 | 0.6796 | 0.7447 |
| 6 | KNN | 0.8450 | -0.0169 | 0.0115 | 0.7910 | 0.8133 | 0.5922 | 0.6854 |
| 7 | RBF SVM | 0.8442 | -0.0177 | 0.0035 | 0.7948 | 0.7791 | 0.6505 | 0.7090 |
| 8 | Logistic Regression | 0.8434 | -0.0185 | 0.0083 | 0.7985 | 0.7379 | 0.7379 | 0.7379 |
| 9 | Decision Tree | 0.8404 | -0.0214 | 0.0014 | 0.7836 | 0.7528 | 0.6505 | 0.6979 |
| 10 | PyTorch NN | 0.8394 | -0.0225 | 0.0067 | 0.7873 | 0.7556 | 0.6602 | 0.7047 |
| 11 | Gaussian Naive Bayes | 0.8027 | -0.0591 | 0.0029 | 0.7687 | 0.6814 | 0.7476 | 0.7130 |
| 12 | Baseline Logistic Regression (non-equal CV, 5-fold full train) | 0.7924 | -0.0694 | 0.0174 |          |          |          |          |

### ensemble_voting(XGBoost, RBF SVM, Logistic Regression) kaggle submit Accuracy: 0.77272

### XGBoost kaggle submit Accuracy: 0.75598

================================================================================
### MODEL COMPARISON (19:23 14-06-2026)
================================================================================

**Run Configuration:**
- **Seed:** `1415926536`
- **CV Folds:** `4`
- **Ensemble Type:** `voting`
- **Optuna Trials (Default):** `100`
- **Optuna Trials (Complex):** `300`

| Rank | Model | CV Accuracy Mean | Delta from Best | CV Accuracy Std | Test Accuracy | Test Precision | Test Recall | Test F1 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **CatBoost** | 0.8584 | +0.0000 | 0.0071 | 0.8022 | 0.7976 | 0.6505 | 0.7166 |
| 2 | XGBoost | 0.8573 | -0.0011 | 0.0124 | 0.7985 | 0.7634 | 0.6893 | 0.7245 |
| 3 | LightGBM | 0.8561 | -0.0023 | 0.0116 | 0.7910 | 0.7527 | 0.6796 | 0.7143 |
| 4 | Ensemble (voting) | 0.8539 | -0.0045 | 0.0132 | 0.8060 | 0.7684 | 0.7087 | 0.7374 |
| 5 | Random Forest | 0.8511 | -0.0073 | 0.0121 | 0.8134 | 0.8118 | 0.6699 | 0.7340 |
| 6 | KNN | 0.8434 | -0.0150 | 0.0082 | 0.7985 | 0.8182 | 0.6117 | 0.7000 |
| 7 | RBF SVM | 0.8414 | -0.0170 | 0.0122 | 0.7985 | 0.7816 | 0.6602 | 0.7158 |
| 8 | Logistic Regression | 0.8403 | -0.0182 | 0.0081 | 0.7985 | 0.7379 | 0.7379 | 0.7379 |
| 9 | PyTorch NN | 0.8352 | -0.0232 | 0.0182 | 0.7873 | 0.7447 | 0.6796 | 0.7107 |
| 10 | Decision Tree | 0.8350 | -0.0234 | 0.0122 | 0.7761 | 0.7263 | 0.6699 | 0.6970 |
| 11 | Gaussian Naive Bayes | 0.7999 | -0.0585 | 0.0118 | 0.7687 | 0.6847 | 0.7379 | 0.7103 |
| 12 | Baseline Logistic Regression (non-equal CV, 5-fold full train) | 0.7924 | -0.0660 | 0.0174 |          |          |          |          |

### ensemble_voting (CatBoost, RBF SVM, Logistic Regression) kaggle submit Accuracy: 0.77511

### CatBoost kaggle submit Accuracy: 0.76794


================================================================================
### MODEL COMPARISON (20:25 14-06-2026)
================================================================================

**Run Configuration:**
- **Seed:** `1415926536`
- **CV Folds:** `5`
- **Ensemble Type:** `voting`
- **Optuna Trials (Default):** `100`
- **Optuna Trials (Complex):** `300`

| Rank | Model | CV Accuracy Mean | Delta from Best | CV Accuracy Std | Test Accuracy | Test Precision | Test Recall | Test F1 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **XGBoost** | 0.8625 | +0.0000 | 0.0086 | 0.7948 | 0.7727 | 0.6602 | 0.7120 |
| 2 | Ensemble (voting) | 0.8555 | -0.0070 | 0.0147 | 0.8022 | 0.7717 | 0.6893 | 0.7282 |
| 3 | LightGBM | 0.8512 | -0.0113 | 0.0119 | 0.7985 | 0.7952 | 0.6408 | 0.7097 |
| 4 | CatBoost | 0.8469 | -0.0155 | 0.0140 | 0.8097 | 0.8095 | 0.6602 | 0.7273 |
| 5 | Random Forest | 0.8417 | -0.0207 | 0.0148 | 0.8246 | 0.8415 | 0.6699 | 0.7460 |
| 6 | RBF SVM | 0.8403 | -0.0221 | 0.0143 | 0.8022 | 0.7778 | 0.6796 | 0.7254 |
| 7 | Logistic Regression | 0.8377 | -0.0248 | 0.0165 | 0.7948 | 0.7353 | 0.7282 | 0.7317 |
| 8 | KNN | 0.8357 | -0.0267 | 0.0075 | 0.7836 | 0.7778 | 0.6117 | 0.6848 |
| 9 | PyTorch NN | 0.8357 | -0.0268 | 0.0204 | 0.7799 | 0.7245 | 0.6893 | 0.7065 |
| 10 | Decision Tree | 0.8225 | -0.0399 | 0.0275 | 0.7761 | 0.7263 | 0.6699 | 0.6970 |
| 11 | Gaussian Naive Bayes | 0.7927 | -0.0697 | 0.0196 | 0.7687 | 0.6847 | 0.7379 | 0.7103 |
| 12 | Baseline Logistic Regression (non-equal CV, 5-fold full train) | 0.7924 | -0.0701 | 0.0174 |          |          |          |          |

### ensemble_voting ("XGBoost", "Logistic Regression", "RBF SVM") kaggle submit Accuracy: 0.77033

### ensemble_voting ("CatBoost", "Logistic Regression", "KNN") kaggle submit Accuracy: 0.77751
