# config.py
from omegaconf import OmegaConf


config = {
    "general":{
        "seed":42,
        # Режимы: "TUNE" (поиск гиперпар.), "TRAIN" (быстрое обучение по .json), "PREDICT" (инференс)
        "mode":"TUNE",
    },
    "data": {
        "train_path": "data/train.csv",
        "test_path": "data/test.csv",
        "target_column": "Survived",
        "test_size": 0.30,
        "categorical_features":['Sex', 'Deck', 'Pclass'],
        "binary_features":['IsAlone'],
    },
    "model": {
        "cv_folds": 2,
        "positive_label": 1,
        "negative_label": 0,
        "task_type": "CPU", # Или "GPU", если доступно
        "active_models": [
            "Logistic Regression",
            "RBF SVM",
            "KNN",
            "Gaussian Naive Bayes",
            "Decision Tree",
            "Random Forest",
            "CatBoost",
            "XGBoost",
            "LightGBM"
        ],
        "ensemble":"voting",
        #other: ["averaging", "voting", "stacking_lr", "stacking_ridge"]
        "ensemble_models":["LightGBM", "Logistic Regression", "RBF SVM"]
    },

    "optuna": {
        "complex_models":["CatBoost", "Random Forest", "XGBoost", "LightGBM"],
        "n_trials_default": 10,
        "n_trials_complex": 3, # Для CatBoost и Random Forest
    },
    "output": {
        "log_file": "log_final_Table.md",

        "best_model_submission_file": "subs/submission.csv",
        "ensemble_submission_file": "subs/submission_ensemble.csv",

        "params_file":       "models/best_params.json",
        "folder_for_joblib": "models"
    }
}

config = OmegaConf.create(config)


OmegaConf.set_readonly(config, True)