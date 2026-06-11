"""Оркестрация обучения для модельного конвейера Titanic."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import train_test_split

from config import config
from preprocessing import preprocessing_data
from trainer import run_experiments
from utils import (
    append_results_to_markdown_log,
    print_results,
    reduce_mem_usage,
    save_all_models,
)


TrainSplit = tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
]


def make_train_test_split(
    data: pd.DataFrame,
    target_column: str = config.data.target_column,
    test_size: float = config.data.test_size,
    random_state: int = config.general.seed,
) -> TrainSplit:
    """Делит данные на стратифицированные признаки и целевую переменную."""
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' was not found in train_data")

    train, test = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        stratify=data[target_column],
    )

    train_x = train.drop(columns=target_column)
    train_y = train[target_column]
    test_x = test.drop(columns=target_column)
    test_y = test[target_column]

    print("=" * 80)
    print(f"Train shape: {train_x.shape}")
    print(f"Test shape:  {test_x.shape}")
    print("=" * 80)

    return train, test, train_x, train_y, test_x, test_y


def get_ensemble_model(estimators: list[tuple[str, Any]]) -> Any:
    """Создает настроенный ансамбль из обученных базовых моделей."""
    ensemble_type = config.model.ensemble

    if ensemble_type == "averaging":
        print("Сборка ансамбля: Усреднение предсказаний (Soft Voting)...")
        return VotingClassifier(estimators=estimators, voting="soft")

    if ensemble_type == "voting":
        print("Сборка ансамбля: Голосование большинства (Hard Voting)...")
        return VotingClassifier(estimators=estimators, voting="hard")

    if ensemble_type == "stacking_lr":
        print("Сборка ансамбля: Стекинг (Мета-модель: Логистическая Регрессия)...")
        return StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(random_state=config.general.seed),
            cv=config.model.cv_folds,
            n_jobs=-1,
        )

    if ensemble_type == "stacking_ridge":
        print("Сборка ансамбля: Стекинг (Мета-модель: Ridge Classifier)...")
        return StackingClassifier(
            estimators=estimators,
            final_estimator=RidgeClassifier(random_state=config.general.seed),
            cv=config.model.cv_folds,
            n_jobs=-1,
        )

    raise ValueError(f"Неизвестный тип ансамблирования: {ensemble_type}")


def _preprocess_features(features: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Применяет инженерию признаков Titanic и уменьшение памяти."""
    return reduce_mem_usage(preprocessing_data(features), verbose=verbose)


def _select_models_for_ensemble(results_df: pd.DataFrame) -> list[str]:
    """Возвращает модели ансамбля из конфига или топ-3 по метрикам."""
    models_to_ensemble = config.model.ensemble_models
    if models_to_ensemble == "top3":
        return results_df["model"].head(3).tolist()
    return list(models_to_ensemble)


def _build_experiment_results(
    best_params_dict: dict[str, dict[str, Any]],
    results_df: pd.DataFrame,
) -> dict[str, Any]:
    """Создает JSON-сериализуемую сводку обучения."""
    return {
        "general_params": {"cv_folds": config.model.cv_folds},
        "params": best_params_dict,
        "results": results_df.to_dict(orient="records"),
    }


def build_and_train_ensemble(
    saved_data: dict[str, Any] | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Запускает эксперименты, обучает финальные модели и сохраняет их."""
    train_data = pd.read_csv(config.data.train_path)

    _, _, train_x, train_y, test_x, test_y = make_train_test_split(train_data)

    train_processed = _preprocess_features(train_x, verbose=True)
    test_processed = _preprocess_features(test_x, verbose=True)

    results_df, trained_models, best_params_dict = run_experiments(
        train_x=train_processed,
        train_y=train_y,
        test_x=test_processed,
        test_y=test_y,
        saved_data=saved_data,
    )

    print_results(results_df)
    append_results_to_markdown_log(results_df, config.output.log_file)

    best_model_name = results_df.iloc[0]["model"]
    best_single_model = trained_models[best_model_name]
    ensemble_model_names = _select_models_for_ensemble(results_df)

    print(f"\nModels for ensemble: {ensemble_model_names}")
    estimators = [(name, trained_models[name]) for name in ensemble_model_names]
    ensemble_model = get_ensemble_model(estimators)

    full_x = train_data.drop(columns=config.data.target_column)
    full_y = train_data[config.data.target_column]
    full_train_processed = _preprocess_features(full_x)

    print("\nFitting on full training data...")
    best_single_model.fit(full_train_processed, full_y)
    ensemble_model.fit(full_train_processed, full_y)

    save_all_models(trained_models, ensemble_model)
    return (
        best_single_model,
        ensemble_model,
        _build_experiment_results(best_params_dict, results_df),
    )
