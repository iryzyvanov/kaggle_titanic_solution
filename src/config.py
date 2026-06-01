# config.py
from omegaconf import OmegaConf


config = {
    "general":{
        "seed":42,
        # Режимы: "TUNE" (поиск гиперпар.), "TRAIN" (быстрое обучение по .json), "PREDICT" (инференс)
        "mode":"TRAIN",
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
        ],
        "ensemble":"stacking_lr"
        #other: ["none", "averaging", "voting", "stacking_lr", "stacking_ridge"]
    },

    "optuna": {
        "n_trials_default": 10,
        "n_trials_complex": 3, # Для CatBoost и Random Forest
    },
    "output": {
        "log_file": "log_final_Table.md",

        "best_model_submission_file": "subs/submission.csv",
        "ensemble_submission_file": "subs/submission_ensemble.csv",

        "params_file":       "models/best_params.json",
        "best_model_joblib": "models/best_model.joblib",
        "ensemble_joblib":   "models/ensemble.joblib",
    }
}

config = OmegaConf.create(config)


OmegaConf.set_readonly(config, True)