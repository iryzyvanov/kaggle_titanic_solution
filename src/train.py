from __future__ import annotations

from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier

from preprocessing import preprocessing_data
from model2         import run_experiments
from utils          import append_results_to_markdown_log, print_results, reduce_mem_usage


TARGET_COLUMN = "Survived"
TEST_SIZE = 0.30
RANDOM_STATE = 42




def make_train_test_split(
    data: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Делит данные на train/test с сохранением баланса классов.

    Returns:
        train, test, train_X, train_Y, test_X, test_Y, X, Y
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

    return train, test, train_X, train_Y, test_X, test_Y, X, Y





def main():
    test_data = pd.read_csv("data/test.csv")
    train_data= pd.read_csv("data/train.csv")

    train, test, train_X, train_Y, test_X, test_Y, X, Y = make_train_test_split(
            train_data,
            target_column=TARGET_COLUMN,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )

    train_processed, test_processed = preprocessing_data(train_X), preprocessing_data(test_X)
    train_processed = reduce_mem_usage(train_processed)
    test_processed  = reduce_mem_usage(test_processed)

    print("=" * 80)
    print("PREPROCESSING")
    print("=" * 80)
    print(f"Train processed shape: {train_processed.shape}")
    print(f"Test processed shape:  {test_processed.shape}")
    print()

    results_df, trained_models = run_experiments(
        train_X=train_processed,
        train_Y=train_Y,
        test_X=test_processed,
        test_Y=test_Y,
    )

    print_results(results_df)
    append_results_to_markdown_log(results_df, "log_final_Table.md")
    # ==========================================================
    # KAGGLE SUBMISSION (ENSEMBLE)
    # ==========================================================

    print("=" * 80)
    print("BUILDING ENSEMBLE")
    print("=" * 80)

    # 1. Выбираем названия Топ-3 лучших моделей из отсортированного results_df
    top_3_names = results_df["model"].head(3).tolist()
    print(f"Top 3 models for ensemble: {top_3_names}\n")

    # 2. Достаем их обученные пайплайны из словаря trained_models
    # estimators - это список кортежей вида [('имя', модель), ...]
    estimators = [(name, trained_models[name]) for name in top_3_names]

    # 3. Создаем ансамбль. 
    # voting='soft' означает, что модели будут усреднять свои вероятности, 
    # а не просто жестко голосовать 0 или 1. Это дает лучшую точность.
    ensemble_model = VotingClassifier(estimators=estimators, voting='soft')

    # ========= PREPROCESS FULL TRAIN =========
    full_X = train_data.drop(columns=TARGET_COLUMN)
    full_Y = train_data[TARGET_COLUMN]

    full_train_processed = preprocessing_data(full_X)
    full_train_processed = reduce_mem_usage(full_train_processed)
    # ========= PREPROCESS KAGGLE TEST =========
    kaggle_test_processed = preprocessing_data(test_data)
    kaggle_test_processed = reduce_mem_usage(kaggle_test_processed)

    # ========= FIT ENSEMBLE ON FULL TRAIN =========
    # При вызове fit, ансамбль автоматически обучит все 3 модели внутри себя 
    # (и их скейлеры, так как мы завернули их в пайплайны) на полных данных!
    ensemble_model.fit(full_train_processed, full_Y)

    # ========= PREDICT =========
    # Ансамбль сам прогонит данные через 3 модели, усреднит вероятности 
    # и выдаст итоговые классы (с порогом 0.5)
    test_predictions = ensemble_model.predict(kaggle_test_processed)

    # ========= SAVE SUBMISSION =========
    submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": test_predictions,
    })
    
    submission.to_csv("submission_ensemble.csv", index=False)


if __name__ == "__main__":
    main()

