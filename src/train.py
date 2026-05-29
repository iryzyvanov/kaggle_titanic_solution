"""
train.py

Главный файл подготовки данных для ML-pipeline.

Что делает:
1. проверяет наличие целевой колонки;
2. показывает короткую сводку по данным;
3. делает stratified train/test split;
4. создаёт переменные, которые использует model.py:
   train_X, train_Y, test_X, test_Y, X, Y.

Ожидается, что DataFrame `train_data` уже загружен до запуска этого файла.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


from preprocessing import preprocessing_data
from model2         import run_experiments, print_results

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




def prepare_features(
    train_X: pd.DataFrame,
    test_X: pd.DataFrame,
):
    """
    Запускает препроцессинг и выравнивает колонки test относительно train.
    """
    train_processed, scaler = preprocessing_data(
        train_X,
        fit_robust_sc=True,
    )

    test_processed, _ = preprocessing_data(
        test_X,
        fit_robust_sc=False,
        scaler=scaler,
    )

    # Выровняем test по колонкам train.
    test_processed = test_processed.reindex(columns=train_processed.columns, fill_value=0)

    return train_processed, test_processed, scaler

def reduce_mem_usage(df):
    """Перебирает все столбцы датафрейма и изменяет тип данных для экономии памяти."""
    start_mem = df.memory_usage().sum() / 1024**2
    print(f'Исходный размер памяти: {start_mem:.4f} MB')
    
    for col in df.columns:
        col_type = df[col].dtype
        
        # Пропускаем текстовые/категориальные колонки
        if col_type != object and col_type.name != 'category':
            c_min = df[col].min()
            c_max = df[col].max()
            
            # Обработка целочисленных типов
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            # Обработка типов с плавающей точкой (float)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                    
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Размер памяти после оптимизации: {end_mem:.4f} MB')
    
    return df

def main():
    test_data = pd.read_csv("data/test.csv")
    train_data= pd.read_csv("data/train.csv")

    train, test, train_X, train_Y, test_X, test_Y, X, Y = make_train_test_split(
            train_data,
            target_column=TARGET_COLUMN,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )

    train_processed, test_processed, scaler = prepare_features(train_X, test_X)
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
    # return {
    #     "train_data": train_data,
    #     "train": train,
    #     "test": test,
    #     "train_X": train_X,
    #     "train_Y": train_Y,
    #     "test_X": test_X,
    #     "test_Y": test_Y,
    #     "X": X,
    #     "Y": Y,
    #     "train_processed": train_processed,
    #     "test_processed": test_processed,
    #     "scaler": scaler,
    #     "results_df": results_df,
    #     "trained_models": trained_models,
    #     "threshold_details": threshold_details,
    # }
    # ==========================================================
    # KAGGLE SUBMISSION
    # ==========================================================

    best_model_name = results_df.iloc[0]["model"]

    print("=" * 80)
    print("BEST MODEL")
    print("=" * 80)
    print(best_model_name)
    print()

    best_model = trained_models[best_model_name]

    # ========= PREPROCESS FULL TRAIN =========

    full_X = train_data.drop(columns=TARGET_COLUMN)
    full_Y = train_data[TARGET_COLUMN]

    full_train_processed, full_scaler = preprocessing_data(
        full_X,
        fit_robust_sc=True,
    )

    # ========= PREPROCESS KAGGLE TEST =========

    kaggle_test_processed, _ = preprocessing_data(
        test_data,
        fit_robust_sc=False,
        scaler=full_scaler,
    )

    kaggle_test_processed = kaggle_test_processed.reindex(
        columns=full_train_processed.columns,
        fill_value=0,
    )

    # ========= FIT BEST MODEL ON FULL TRAIN =========

    best_model.fit(full_train_processed, full_Y)

    # ========= PREDICT =========
    #print(kaggle_test_processed.isna().sum())
    
    test_predictions = best_model.predict(
        kaggle_test_processed
    )

    # ========= SAVE SUBMISSION =========

    submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": test_predictions,
    })

    submission.to_csv(
        "submission.csv",
        index=False
    )

    print("=" * 80)
    print("SUBMISSION SAVED")
    print("=" * 80)
    print("submission.csv")
    print()


if __name__ == "__main__":
    main()

