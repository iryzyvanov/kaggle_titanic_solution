"""Простой бейзлайн логистической регрессии для Titanic без утечки данных."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
TARGET = "Survived"
RANDOM_STATE = 42


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Загружает обучающий и тестовый файлы Kaggle для бейзлайна."""
    try:
        train_data = pd.read_csv("data/train.csv")
        test_data = pd.read_csv("data/test.csv")
    except FileNotFoundError:
        print("Файлы данных не найдены! Проверьте путь data/train.csv и data/test.csv")
        return None

    return train_data, test_data


def _prepare_features(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Готовит минимальный набор признаков без статистик из тестовых данных."""
    train_x = train_data[FEATURES].copy()
    train_y = train_data[TARGET]
    test_x = test_data[FEATURES].copy()
    test_ids = test_data["PassengerId"]

    age_median = train_x["Age"].median()
    fare_median = train_x["Fare"].median()
    embarked_mode = train_x["Embarked"].mode()[0]

    for frame in (train_x, test_x):
        frame["Age"] = frame["Age"].fillna(age_median)
        frame["Fare"] = frame["Fare"].fillna(fare_median)
        frame["Embarked"] = frame["Embarked"].fillna(embarked_mode)

    train_x = pd.get_dummies(
        train_x,
        columns=["Sex", "Embarked"],
        drop_first=True,
    )
    test_x = pd.get_dummies(
        test_x,
        columns=["Sex", "Embarked"],
        drop_first=True,
    )
    train_x, test_x = train_x.align(test_x, join="left", axis=1, fill_value=0)
    return train_x, train_y, test_x, test_ids


def _print_scores(scores: np.ndarray) -> None:
    """Печатает метрики кросс-валидации бейзлайна."""
    print("Модель: Logistic Regression (Default)")
    print("Фичи: Сырые числа + OHE для Sex и Embarked")
    print("-" * 60)
    print(f"CV Accuracy Mean: {np.mean(scores):.4f}")
    print(f"CV Accuracy Std:  {np.std(scores):.4f}")
    print("=" * 60)


def _save_submission(test_ids: pd.Series, predictions: np.ndarray) -> None:
    """Сохраняет предсказания бейзлайна в формате отправки Kaggle."""
    output_path = Path("subs/baseline_submission.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    submission = pd.DataFrame(
        {
            "PassengerId": test_ids,
            "Survived": predictions,
        }
    )
    submission.to_csv(output_path, index=False)
    print(f"Готово! Файл для отправки сохранен по пути: {output_path}")


def run_baseline() -> None:
    """Обучает и оценивает простой бейзлайн логистической регрессии."""
    print("=" * 60)
    print("СТАРТ РАСЧЕТА БЕЙЗЛАЙНА (Logistic Regression)")
    print("=" * 60)

    loaded_data = _load_data()
    if loaded_data is None:
        return

    train_data, test_data = loaded_data
    train_x, train_y, test_x, test_ids = _prepare_features(train_data, test_data)

    baseline_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        baseline_model,
        train_x,
        train_y,
        cv=cv,
        scoring="accuracy",
    )

    _print_scores(scores)

    print("\nОбучение финальной модели на всем объеме train...")
    baseline_model.fit(train_x, train_y)

    print("Формирование предсказаний...")
    test_preds = baseline_model.predict(test_x)
    _save_submission(test_ids, test_preds)


if __name__ == "__main__":
    run_baseline()
