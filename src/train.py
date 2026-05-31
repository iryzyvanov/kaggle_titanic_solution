from __future__ import annotations

from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier

from config import config
from preprocessing import preprocessing_data
from model2         import run_experiments
from utils          import append_results_to_markdown_log, print_results, reduce_mem_usage

def make_train_test_split(
    data: pd.DataFrame,
    target_column: str = config.data.target_column,
    test_size: float = config.data.test_size,
    random_state: int = config.general.seed,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Делит данные на train/test с сохранением баланса классов.
    """
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' was not found in train_data")

    train, test = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        stratify=data[target_column],
    )

    train_X = train.drop(columns=target_column)
    train_Y = train[target_column]

    test_X = test.drop(columns=target_column)
    test_Y = test[target_column]

    X = data.drop(columns=target_column)
    Y = data[target_column]

    print("=" * 80)
    print("TRAIN/TEST SPLIT")
    print("=" * 80)
    print(f"Train shape: {train_X.shape}")
    print(f"Test shape:  {test_X.shape}")
    print()

    return train, test, train_X, train_Y, test_X, test_Y

# Функция автоматической сборки ансамбля на основе конфигурации
def get_ensemble_model(estimators):
    """
    estimators: список кортежей вида [('catboost', model1), ('rf', model2), ...]
    """
    ensemble_type = config.model.ensemble
    
    if ensemble_type == "averaging":
        print("Сборка ансамбля: Усреднение предсказаний (Soft Voting)...")
        return VotingClassifier(estimators=estimators, voting='soft')
        
    elif ensemble_type == "voting":
        print("Сборка ансамбля: Голосование большинства (Hard Voting)...")
        return VotingClassifier(estimators=estimators, voting='hard')
        
    elif ensemble_type == "stacking_lr":
        print("Сборка ансамбля: Стекинг (Мета-модель: Логистическая Регрессия)...")
        return StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(random_state=config.RANDOM_STATE),
            cv=config.N_SPLITS,
            n_jobs=-1
        )
        
    elif ensemble_type == "stacking_ridge":
        print("Сборка ансамбля: Стекинг (Мета-модель: Ridge Classifier)...")
        return StackingClassifier(
            estimators=estimators,
            final_estimator=RidgeClassifier(random_state=config.RANDOM_STATE),
            cv=config.N_SPLITS,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Неизвестный тип ансамблирования: {ensemble_type}")

def build_and_train_ensemble():
    """Проводит эксперименты, собирает и обучает ансамбль, возвращает готовую модель."""
    print("=" * 80)
    print("STARTING TRAINING PIPELINE")
    print("=" * 80)
    
    train_data = pd.read_csv(config.data.train_path)

    train, test, train_X, train_Y, test_X, test_Y = make_train_test_split(train_data)

    train_processed = reduce_mem_usage(preprocessing_data(train_X), verbose=True)
    test_processed  = reduce_mem_usage(preprocessing_data(test_X), verbose=True)

    results_df, trained_models = run_experiments(
        train_X=train_processed,
        train_Y=train_Y,
        test_X=test_processed,
        test_Y=test_Y,
    )

    print_results(results_df)
    append_results_to_markdown_log(results_df, config.output.log_file)

    best_model_name = results_df.iloc[0]["model"]
    best_single_model = trained_models[best_model_name]
    

    # Собираем ансамбль из топ-3
    top_3_names = results_df["model"].head(3).tolist()
    print(f"\nTop 3 models for ensemble: {top_3_names}")
    
    estimators = [(name, trained_models[name]) for name in top_3_names]
    ensemble_model = get_ensemble_model(estimators)

    # Обучаем ансамбль на полных данных
    full_X = train_data.drop(columns=config.data.target_column)
    full_Y = train_data[config.data.target_column]
    
    full_train_processed = reduce_mem_usage(preprocessing_data(full_X))
    
    print("\nFitting on full training data...")
    best_single_model.fit(full_train_processed, full_Y)
    ensemble_model.fit(full_train_processed, full_Y)
    
    # ВОЗВРАЩАЕМ ГОТОВУЮ МОДЕЛЬ
    return best_single_model, ensemble_model
