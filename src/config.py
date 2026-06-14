"""Конфигурация проекта для обучения, подбора параметров и инференса."""

from omegaconf import OmegaConf


config = {
    "general": {
        "seed": 1415926536,
        # Режимы: "TUNE" - подбор параметров, "TRAIN" - обучение по JSON,
        # "PREDICT" - инференс по сохраненным моделям.
        "mode": "TUNE",
    },
    "data": {
        "train_path": "data/train.csv",
        "test_path": "data/test.csv",
        "target_column": "Survived",
        "test_size": 0.30,
        "categorical_features": ["Sex", "Deck", "Pclass", "Honorific", "Embarked"],
        "binary_features": ["IsAlone"],
    },
    "model": {
        "cv_folds": 5,
        "positive_label": 1,
        "negative_label": 0,
        "task_type": "CPU",
        "active_models": [
            "Logistic Regression",
            "RBF SVM",
            "KNN",
            "Gaussian Naive Bayes",
            "Decision Tree",
            "Random Forest",
            "CatBoost",
            "XGBoost",
            "LightGBM",
            "PyTorch NN"
        ],
        "ensemble": "voting",
        "ensemble_models": ["CatBoost", "Logistic Regression", "KNN"],
    },
    "optuna": {
        "complex_models": ["CatBoost", "Random Forest", "XGBoost", "LightGBM"],
        "n_trials_default": 100,
        "n_trials_complex": 300,
    },
    "output": {
        "log_file": "log_final_Table.md",
        "best_model_submission_file": "subs/submission.csv",
        "ensemble_submission_file": "subs/submission_ensemble.csv",
        "params_file": "models/best_params.json",
        "baseline_results_file": "models/baseline_results.json",
        "folder_for_joblib": "models",
    },
}

config = OmegaConf.create(config)
OmegaConf.set_readonly(config, True)
