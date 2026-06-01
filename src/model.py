

from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional, Tuple
import json
import os
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline


from sklearn.model_selection import StratifiedKFold
from sklearn import metrics, svm
from sklearn.base import clone

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

from config import config
from preprocessing import get_preprocessor
from utils import take_Optuna_with_modify_logs
optuna = take_Optuna_with_modify_logs()

from optuna.trial import FixedTrial

def build_model_by_trial(trial, model_name, random_state=config.general.seed):
    """Строит классификатор на основе гиперпараметров из trial."""
    if model_name == "Logistic Regression":
        penalty_choice = trial.suggest_categorical("penalty", ["l1", "l2"])
        l1_ratio_value = 1.0 if penalty_choice == "l1" else 0.0
        return LogisticRegression(
            # Расширен диапазон силы регуляризации
            C=trial.suggest_float("C", 1e-4, 100, log=True), 
            l1_ratio=l1_ratio_value,
            solver="liblinear",
            class_weight=trial.suggest_categorical("class_weight", [None, "balanced"]),
            max_iter=5000,
            random_state=random_state,
        )
        
    elif model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 700),
            # Добавлен выбор критерия разбиения
            criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]), 
            max_depth=trial.suggest_int("max_depth", 2, 10),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            # Убрали None, чтобы не превращать алгоритм в обычный бэггинг
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]), 
            random_state=random_state,
        )
        
    elif model_name == "RBF SVM":
        # Комбинированный подход для gamma: встроенные эвристики или точное число
        gamma_choice = trial.suggest_categorical("gamma_choice", ["scale", "auto", "float"])
        if gamma_choice == "float":
            gamma_val = trial.suggest_float("gamma_float", 1e-4, 1, log=True)
        else:
            gamma_val = gamma_choice
            
        return svm.SVC(
            # Расширен диапазон C для построения более сложных гиперплоскостей
            C=trial.suggest_float("C", 1e-3, 1000, log=True), 
            gamma=gamma_val,
            probability=True,
            kernel="rbf",
            random_state=random_state,
        )
        
    elif model_name == "KNN":
        return KNeighborsClassifier(
            n_neighbors=trial.suggest_int("n_neighbors", 3, 25),
            weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
            p=trial.suggest_int("p", 1, 2),
        )
        
    elif model_name == "Decision Tree":
        return DecisionTreeClassifier(
            # Добавлен выбор критерия разбиения
            criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]), 
            max_depth=trial.suggest_int("max_depth", 2, 10),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            random_state=random_state,
        )
        
    elif model_name == "Gaussian Naive Bayes":
        return GaussianNB(
            # Существенно расширен диапазон для лучшего сглаживания перекошенных признаков
            var_smoothing=trial.suggest_float("var_smoothing", 1e-10, 1e-2, log=True) 
        )
        
    elif model_name == "CatBoost" and CatBoostClassifier is not None:
        return CatBoostClassifier(
            iterations=trial.suggest_int("iterations", 100, 1000),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            # Сужен диапазон глубины для защиты от переобучения
            depth=trial.suggest_int("depth", 3, 7), 
            # Усилена регуляризация
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-1, 30, log=True), 
            # Добавлен бэггинг для случайной подвыборки строк
            subsample=trial.suggest_float("subsample", 0.5, 1.0), 
            bootstrap_type="Bernoulli", 
            # Аналог min_samples_leaf
            min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 1, 10), 
            verbose=False,
            random_seed=random_state,
            task_type=config.model.task_type,
        )
        
    else:
        raise ValueError(f"Unknown model: {model_name}")

def _take_rows(data, indices):
    """Скроем разницу в синтаксисе с помощью адаптера"""
    if hasattr(data, "iloc"):
        return data.iloc[indices]
    return data[indices]




def optimize_model(model_name, X, y, n_trials, cv_folds=5, random_state=config.general.seed):
    def objective(trial):
        # 1. Задаем модель
        model = build_model_by_trial(trial, model_name=model_name, random_state=random_state)
        
        # 2. Собираем пайплайн (Скейлер -> Модель)
        pipeline = Pipeline([
            ("preprocessor", get_preprocessor()),
            ("classifier", model)
        ])
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scores = []
        
        for train_idx, valid_idx in cv.split(X, y):
            X_train_fold, X_valid_fold = _take_rows(X, train_idx), _take_rows(X, valid_idx)
            y_train_fold, y_valid_fold = _take_rows(y, train_idx), _take_rows(y, valid_idx)
            
            # Клонируем и обучаем весь пайплайн (скейлинг произойдет ТОЛЬКО на train_fold)
            fold_model = clone(pipeline)
            fold_model.fit(X_train_fold, y_train_fold)
            
            valid_pred = fold_model.predict(X_valid_fold)
            score = metrics.accuracy_score(y_valid_fold, valid_pred)
            scores.append(score)
            
        mean_score = np.mean(scores)
        trial.set_user_attr("cv_std", np.std(scores))
        
        return mean_score

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    print(f"\nBEST PARAMS [{model_name}]")
    print(study.best_params)
    print(f"BEST SCORE: {study.best_value:.4f}")

    best_model = build_model_by_trial(study.best_trial, model_name=model_name, random_state=random_state)

    return best_model, study

def fit_evaluate_model(
    model_name: str,
    estimator,
    cv_mean: float,
    cv_std: float,
    train_X,
    train_Y,
    test_X,
    test_Y,) -> Tuple[dict, object]:
    
    # Собираем финальный пайплайн с лучшими параметрами
    final_pipeline = Pipeline([
            ("preprocessor", get_preprocessor()),
            ("classifier", clone(estimator))
        ])
    
    # Обучаем пайплайн (он сам отмасштабирует train_X)
    final_pipeline.fit(train_X, train_Y)

    # Предсказываем (он сам применит скейлинг к test_X)
    test_pred = final_pipeline.predict(test_X)

    result = {
        "model": model_name,
        "cv_accuracy_mean": cv_mean,
        "cv_accuracy_std": cv_std,
        "test_accuracy": metrics.accuracy_score(test_Y, test_pred),
        "test_precision": metrics.precision_score(test_Y, test_pred, zero_division=0),
        "test_recall": metrics.recall_score(test_Y, test_pred, zero_division=0),
        "test_f1": metrics.f1_score(test_Y, test_pred, zero_division=0),
    }

    return result, final_pipeline

def run_experiments(
    train_X, train_Y, test_X, test_Y, cv_folds: int = config.model.cv_folds
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Запускает обучение и сравнение всех моделей с учетом режима TUNE/TRAIN."""

    mode = config.general.mode
    params_path = config.output.params_file
    
    results = []
    trained_models = {}

    # Если режим TRAIN, пытаемся загрузить готовые параметры
    if mode == "TRAIN":
        if not os.path.exists(params_path):
            print(f"Файл {params_path} не найден! Принудительно включаем режим TUNE.")
            mode = "TUNE"
        else:
            print(f"\n[INFO] Загрузка гиперпараметров из {params_path}...")
            with open(params_path, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
            best_params_dict = saved_data["params"]
            saved_results = saved_data["results"]

    if mode == "TUNE":
        best_params_dict = {}
        saved_results = []

    for model_name in config.model.active_models:
        if mode == "TUNE":
            print(f"\nOPTIMIZING: {model_name}")
            n_trials = config.optuna.n_trials_complex if model_name in ["CatBoost", "Random Forest"] else config.optuna.n_trials_default
            best_model, study = optimize_model(model_name, X=train_X, y=train_Y, n_trials=n_trials, cv_folds=cv_folds)
            
            # Сохраняем найденное
            best_params_dict[model_name] = study.best_params
            cv_mean = study.best_value
            cv_std = study.best_trial.user_attrs.get("cv_std", np.nan)
            
        elif mode == "TRAIN":
            print(f"\nFAST TRAINING: {model_name} (Используем сохраненные параметры)")
            # берем ранее полученные параметры
            trial = FixedTrial(best_params_dict[model_name])
            best_model = build_model_by_trial(trial, model_name=model_name)
            
            # Восстанавливаем метрики кросс-валидации из файла
            res_info = next((item for item in saved_results if item["model"] == model_name), None)
            cv_mean = res_info["cv_accuracy_mean"] if res_info else 0.0
            cv_std = res_info["cv_accuracy_std"] if res_info else 0.0

        # Обучение на полных данных (train_X) и оценка на отложенном тесте (test_X)
        result, fitted_model = fit_evaluate_model(
            model_name=model_name, estimator=best_model, cv_mean=cv_mean, cv_std=cv_std,
            train_X=train_X, train_Y=train_Y, test_X=test_X, test_Y=test_Y,
        )
        results.append(result)
        trained_models[model_name] = fitted_model

    results_df = pd.DataFrame(results).round(5).sort_values(by=["cv_accuracy_mean", "cv_accuracy_std"], ascending=[False, True])

    # Если мы искали параметры, сохраняем их в JSON на будущее
    if mode == "TUNE":
        save_data = {
            "general_params":{"cv_folds":config.model.cv_folds},
            "params": best_params_dict,
            "results": results_df.to_dict(orient="records")
        }
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=4, ensure_ascii=False)
        print(f"\nВсе гиперпараметры сохранены в файл: {params_path}")

    return results_df, trained_models


