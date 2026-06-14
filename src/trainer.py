"""Оптимизация, обучение и оценка моделей."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from optuna.trial import FixedTrial
from sklearn import metrics
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from config import config
from model_builder import build_model_by_trial
from preprocessing import GroupAgeImputer, TitanicFeatureEngineer, get_preprocessor
from utils import configure_optuna_logging

optuna = configure_optuna_logging()


def _take_rows(data: Any, indices: np.ndarray) -> Any:
    """Возвращает строки pandas или numpy объекта по позиционным индексам."""
    if hasattr(data, "iloc"):
        return data.iloc[indices]
    return data[indices]


def _create_pipeline(estimator: BaseEstimator) -> Pipeline:
    """Оборачивает классификатор feature engineering и общим предобработчиком."""
    return Pipeline(
        [
            ("feature_engineering", TitanicFeatureEngineer()),
            ("age_imputer", GroupAgeImputer()),
            ("preprocessor", get_preprocessor()),
            ("classifier", estimator),
        ]
    )


def cross_validate_estimator(
    estimator: BaseEstimator,
    x: pd.DataFrame,
    y: pd.Series,
    cv_folds: int,
    random_state: int,
) -> list[float]:
    """Оценивает pipeline на стратифицированных фолдах и возвращает accuracy."""
    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )
    scores = []

    for train_idx, valid_idx in cv.split(x, y):
        x_train_fold = _take_rows(x, train_idx)
        x_valid_fold = _take_rows(x, valid_idx)
        y_train_fold = _take_rows(y, train_idx)
        y_valid_fold = _take_rows(y, valid_idx)

        fold_model = clone(estimator)
        fold_model.fit(x_train_fold, y_train_fold)

        valid_pred = fold_model.predict(x_valid_fold)
        scores.append(metrics.accuracy_score(y_valid_fold, valid_pred))

    return scores


def optimize_model(
    model_name: str,
    x: pd.DataFrame,
    y: pd.Series,
    n_trials: int,
    cv_folds: int = 5,
    random_state: int = config.general.seed,
) -> tuple[BaseEstimator, Any]:
    """Подбирает параметры через Optuna и возвращает модель и исследование."""

    def objective(trial: Any) -> float:
        model = build_model_by_trial(
            trial,
            model_name=model_name,
            random_state=random_state,
        )
        pipeline = _create_pipeline(model)
        scores = cross_validate_estimator(
            estimator=pipeline,
            x=x,
            y=y,
            cv_folds=cv_folds,
            random_state=random_state,
        )
        cv_mean = np.mean(scores)
        cv_std  = np.std(scores)

        trial.set_user_attr("cv_std", cv_std)
        # Штрафуем модель за высокий разброс на фолдах
        # Коэффициент 0.5 в будущем нужно вынести в конфиг
        return float(cv_mean - (0.5 * cv_std))

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    print(f"\nBEST PARAMS [{model_name}]")
    print(study.best_params)
    print(f"BEST SCORE: {study.best_value:.4f}")

    best_model = build_model_by_trial(
        study.best_trial,
        model_name=model_name,
        random_state=random_state,
    )
    return best_model, study


def fit_evaluate_model(
    model_name: str,
    estimator: BaseEstimator,
    cv_mean: float,
    cv_std: float,
    train_x: pd.DataFrame,
    train_y: pd.Series,
    test_x: pd.DataFrame,
    test_y: pd.Series,
) -> tuple[dict[str, float | str], Pipeline]:
    """Обучает один конвейер модели и возвращает метрики отложенной выборки."""
    final_pipeline = _create_pipeline(clone(estimator))
    final_pipeline.fit(train_x, train_y)

    test_pred = final_pipeline.predict(test_x)
    result = {
        "model": model_name,
        "cv_accuracy_mean": cv_mean,
        "cv_accuracy_std": cv_std,
        "test_accuracy": metrics.accuracy_score(test_y, test_pred),
        "test_precision": metrics.precision_score(test_y, test_pred, zero_division=0),
        "test_recall": metrics.recall_score(test_y, test_pred, zero_division=0),
        "test_f1": metrics.f1_score(test_y, test_pred, zero_division=0),
    }

    return result, final_pipeline


def refit_models_on_full_data(
    trained_models: dict[str, Pipeline],
    full_x: pd.DataFrame,
    full_y: pd.Series,
) -> dict[str, Pipeline]:
    """Переобучает все модели на полном train.csv перед сохранением."""
    return {
        name: clone(model).fit(full_x, full_y)
        for name, model in trained_models.items()
    }


def _saved_cv_metrics(
    saved_results: list[dict[str, Any]],
    model_name: str,
) -> tuple[float, float]:
    """Читает сохраненные CV-метрики для режима быстрого обучения."""
    result_info = next(
        (item for item in saved_results if item["model"] == model_name),
        None,
    )
    if result_info is None:
        return 0.0, 0.0

    return result_info["cv_accuracy_mean"], result_info["cv_accuracy_std"]


def _n_trials_for_model(model_name: str) -> int:
    """Выбирает число испытаний Optuna для простых и сложных моделей."""
    complex_models = list(config.optuna.complex_models)
    if model_name in complex_models:
        return config.optuna.n_trials_complex
    return config.optuna.n_trials_default


def _build_from_saved_params(
    model_name: str,
    existing_params: dict[str, dict[str, Any]],
    saved_results: list[dict[str, Any]],
) -> tuple[BaseEstimator, float, float]:
    """Создает модель из сохраненных Optuna-параметров и CV-метрик."""
    print(f"\nFAST TRAINING: {model_name} (Используем сохраненные параметры)")

    trial = FixedTrial(existing_params[model_name])
    best_model = build_model_by_trial(trial, model_name=model_name)
    cv_mean, cv_std = _saved_cv_metrics(saved_results, model_name)
    return best_model, cv_mean, cv_std


def _optimize_missing_params(
    model_name: str,
    train_x: pd.DataFrame,
    train_y: pd.Series,
    cv_folds: int,
) -> tuple[BaseEstimator, dict[str, Any], float, float]:
    """Подбирает параметры модели, для которой еще нет сохраненных настроек."""
    print(f"\nOPTIMIZING: {model_name} (Параметры не найдены, запускаем Optuna)")

    best_model, study = optimize_model(
        model_name,
        x=train_x,
        y=train_y,
        n_trials=_n_trials_for_model(model_name),
        cv_folds=cv_folds,
    )
    cv_std = study.best_trial.user_attrs.get("cv_std", np.nan)
    return best_model, study.best_params, study.best_value, cv_std


def run_experiments(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    test_x: pd.DataFrame,
    test_y: pd.Series,
    cv_folds: int = config.model.cv_folds,
    saved_data: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Pipeline], dict[str, dict[str, Any]]]:
    """Обучает и оценивает все активные модели из конфига проекта."""
    results = []
    trained_models = {}

    existing_params = saved_data.get("params", {}) if saved_data else {}
    saved_results = saved_data.get("results", []) if saved_data else []
    best_params_dict = dict(existing_params)

    for model_name in config.model.active_models:
        if model_name in existing_params:
            best_model, cv_mean, cv_std = _build_from_saved_params(
                model_name,
                existing_params=existing_params,
                saved_results=saved_results,
            )
        else:
            best_model, best_params, cv_mean, cv_std = _optimize_missing_params(
                model_name,
                train_x=train_x,
                train_y=train_y,
                cv_folds=cv_folds,
            )
            best_params_dict[model_name] = best_params

        result, fitted_model = fit_evaluate_model(
            model_name=model_name,
            estimator=best_model,
            cv_mean=cv_mean,
            cv_std=cv_std,
            train_x=train_x,
            train_y=train_y,
            test_x=test_x,
            test_y=test_y,
        )
        results.append(result)
        trained_models[model_name] = fitted_model

    results_df = (
        pd.DataFrame(results)
        .round(5)
        .sort_values(
            by=["cv_accuracy_mean", "cv_accuracy_std"],
            ascending=[False, True],
        )
    )
    return results_df, trained_models, best_params_dict
