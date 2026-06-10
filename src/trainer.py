

from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional, Tuple
import json
import os
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.base import clone


from config import config
from preprocessing import get_preprocessor
from model_builder import build_model_by_trial
from utils import take_Optuna_with_modify_logs
optuna = take_Optuna_with_modify_logs()

from optuna.trial import FixedTrial


def _take_rows(data, indices):
    """Скроем разницу в синтаксисе с помощью адаптера"""
    if hasattr(data, "iloc"):
        return data.iloc[indices]
    return data[indices]

def _create_pipeline(estimator):
    return Pipeline([
            ("preprocessor", get_preprocessor()),
            ("classifier", estimator)
        ])

def optimize_model(model_name, X, y, n_trials, cv_folds=5, random_state=config.general.seed):
    def objective(trial):
        # 1. Задаем модель
        model = build_model_by_trial(trial, model_name=model_name, random_state=random_state)
        
        pipeline = _create_pipeline(model)
        
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
    final_pipeline = _create_pipeline(clone(estimator))
    
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
    train_X, train_Y, test_X, test_Y, 
    cv_folds: int = config.model.cv_folds,
    saved_data: Optional[dict] = None
    ) -> Tuple[pd.DataFrame, Dict[str, object], Dict[str, dict]]:
    """
    Запускает обучение и оценку всех активных моделей.
    
    Если в saved_data переданы параметры для конкретной модели — берет их (Fast Train).
    Если параметров для модели нет — автоматически включает Optuna (Tune).
    
    """
    results = []
    trained_models = {}
    
    # Безопасно извлекаем исторические параметры и результаты, если они есть
    existing_params = saved_data.get("params", {}) if saved_data else {}
    saved_results   = saved_data.get("results", []) if saved_data else {}
    
    # В этот словарь мы соберем финальные параметры (и старые сохраненные, и новые от Optuna)
    best_params_dict = dict(existing_params)

    for model_name in config.model.active_models:
        
        # Сценарий 1: Для модели ЕСТЬ сохраненные параметры -> Быстрое обучение
        if model_name in existing_params:
            print(f"\nFAST TRAINING: {model_name} (Используем сохраненные параметры)")
            trial = FixedTrial(existing_params[model_name])
            best_model = build_model_by_trial(trial, model_name=model_name)
            
            # Восстанавливаем метрики кросс-валидации из сохраненной истории
            res_info = next((item for item in saved_results if item["model"] == model_name), None)
            cv_mean = res_info["cv_accuracy_mean"] if res_info else 0.0
            cv_std = res_info["cv_accuracy_std"] if res_info else 0.0
            
        # Сценарий 2: Для модели НЕТ параметров -> Оптимизация через Optuna
        else:
            print(f"\nOPTIMIZING: {model_name} (Параметры не найдены, запускаем Optuna)")
            
            # Определяем количество триалов в зависимости от сложности модели
            complex_models = list(config.optuna.complex_models)
            n_trials = (config.optuna.n_trials_complex 
                        if model_name in complex_models 
                        else config.optuna.n_trials_default)
            
            # Запуск Optuna (эта функция остается неизменной, она возвращает модель и study)
            best_model, study = optimize_model(
                model_name, X=train_X, y=train_Y, n_trials=n_trials, cv_folds=cv_folds
            )
            
            # Запоминаем новые подобранные параметры
            best_params_dict[model_name] = study.best_params
            cv_mean = study.best_value
            cv_std = study.best_trial.user_attrs.get("cv_std", np.nan)

        # Обучение пайплайна на train_X и оценка на отложенном test_X
        result, fitted_model = fit_evaluate_model(
            model_name=model_name, estimator=best_model, cv_mean=cv_mean, cv_std=cv_std,
            train_X=train_X, train_Y=train_Y, test_X=test_X, test_Y=test_Y,
        )
        results.append(result)
        trained_models[model_name] = fitted_model

    # Сортируем итоговый DataFrame по качеству
    results_df = (pd.DataFrame(results)
                  .round(5)
                  .sort_values(by=["cv_accuracy_mean", "cv_accuracy_std"], ascending=[False, True]))

    # Возвращаем ТРИ вещи: датафрейм, обученные модели и актуальный словарь параметров
    return results_df, trained_models, best_params_dict


