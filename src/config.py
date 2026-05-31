# config.py
from omegaconf import OmegaConf


config = {
    "general":{
        "seed":42,
    },
    "data": {
        "train_path": "data/train.csv",
        "test_path": "data/test.csv",
        "target_column": "Survived",
        "test_size": 0.30,
    },
    "model": {
        "cv_folds": 4,
        "positive_label": 1,
        "negative_label": 0,
        "task_type": "CPU", # Или "GPU", если доступно
        "active_models": [
            "Logistic Regression",
            "RBF SVM",
            #"KNN",
            #"Gaussian Naive Bayes",
            #"Decision Tree",
            #"Random Forest",
            "CatBoost"
        ]
    },
    "optuna": {
        "n_trials_default": 150,
        "n_trials_complex": 300, # Для CatBoost и Random Forest
    },
    "output": {
        "log_file": "log_final_Table.md",
        "submission_file": "submission_ensemble.csv",
        "model_file": "ensemble_model.joblib"
    }
}

config = OmegaConf.create(config)


OmegaConf.set_readonly(config, True)