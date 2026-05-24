"""
model.py

Обучение и сравнение моделей.

train fold
   ↓
fit model
   ↓
predict_proba(valid fold)
   ↓
select threshold
   ↓
fit on full train
   ↓
evaluate on test
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

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
except ImportError:  # CatBoost может быть не установлен в окружении
    CatBoostClassifier = None


RANDOM_STATE = 42
CV_FOLDS = 5
THRESHOLDS = np.round(np.arange(0.00, 1.01, 0.01), 2)
POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0


def get_models(random_state: int = RANDOM_STATE) -> Dict[str, object]:
    """Возвращает модели в одном месте, чтобы их было удобно менять."""
    models: Dict[str, object] = {
        "Logistic Regression": LogisticRegression(max_iter=1_000, random_state=random_state),
        "RBF SVM": svm.SVC(kernel="rbf", C=1, gamma=0.1, probability=True, random_state=random_state),
        "KNN": KNeighborsClassifier(),
        "Gaussian Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
    }

    if CatBoostClassifier is not None:
        models["CatBoost"] = CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=4,
            loss_function="Logloss",
            eval_metric="Accuracy",
            random_seed=random_state,
            verbose=False,
        )

    return models


def _take_rows(data, indices):
    """Скроем разницу с синтаксисе с помощью адаптера"""
    if hasattr(data, "iloc"):
        return data.iloc[indices]
    return data[indices]


def _positive_class_proba(model, X, positive_label: int = POSITIVE_LABEL) -> np.ndarray:
    """Возвращаем вероятность положительного класса из predict_proba."""
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"Model {type(model).__name__} does not support predict_proba")

    classes = list(model.classes_)
    if positive_label not in classes:
        raise ValueError(f"Positive label {positive_label} was not found in model classes: {classes}")

    positive_class_index = classes.index(positive_label)
    return model.predict_proba(X)[:, positive_class_index]


def _predict_by_threshold(
    proba: np.ndarray,
    threshold: float,
    positive_label: int = POSITIVE_LABEL,
    negative_label: int = NEGATIVE_LABEL,
) -> np.ndarray:
    """Возвращает классы по вероятности исходя из выбранного threshold."""
    return np.where(proba >= threshold, positive_label, negative_label)


def select_threshold_kfold(
    estimator,
    X,
    y,
    thresholds: Iterable[float] = THRESHOLDS,
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
    metric: Callable = metrics.accuracy_score,
    positive_label: int = POSITIVE_LABEL,
    negative_label: int = NEGATIVE_LABEL,
) -> Tuple[float, pd.DataFrame, pd.DataFrame]:
    """
    Подбирает threshold через Stratified K-Fold CV.

    Важно: test-set здесь не используется. Threshold выбирается только по validation-fold.

    Returns:
        best_threshold:
            threshold с максимальным средним validation score;
        threshold_scores:
            средний score и std по каждому threshold;
        fold_scores:
            лучший threshold на каждом fold для диагностики.
    """
    threshold_grid = [float(t) for t in thresholds]
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    scores_by_threshold = {threshold: [] for threshold in threshold_grid}
    best_by_fold = []

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        fold_model = clone(estimator)

        X_train_fold = _take_rows(X, train_idx)
        y_train_fold = _take_rows(y, train_idx)
        X_valid_fold = _take_rows(X, valid_idx)
        y_valid_fold = _take_rows(y, valid_idx)

        fold_model.fit(X_train_fold, y_train_fold)
        valid_proba = _positive_class_proba(fold_model, X_valid_fold, positive_label)

        fold_threshold_scores = []
        for threshold in threshold_grid:
            valid_pred = _predict_by_threshold(
                valid_proba,
                threshold,
                positive_label=positive_label,
                negative_label=negative_label,
            )
            score = metric(y_valid_fold, valid_pred)
            scores_by_threshold[threshold].append(score)
            fold_threshold_scores.append((threshold, score))

        fold_best_threshold, fold_best_score = max(
            fold_threshold_scores,
            key=lambda item: (item[1], -abs(item[0] - 0.5)),
        )
        best_by_fold.append(
            {
                "fold": fold,
                "best_threshold": fold_best_threshold,
                "best_validation_score": fold_best_score,
            }
        )

    threshold_scores = pd.DataFrame(
        [
            {
                "threshold": threshold,
                "mean_validation_score": np.mean(scores),
                "std_validation_score": np.std(scores),
            }
            for threshold, scores in scores_by_threshold.items()
        ]
    ).sort_values("threshold")

    best_row = threshold_scores.sort_values(
        by=["mean_validation_score", "threshold"],
        ascending=[False, True],
    ).iloc[0]
    best_threshold = float(best_row["threshold"])

    fold_scores = pd.DataFrame(best_by_fold)
    return best_threshold, threshold_scores, fold_scores


def fit_evaluate_model(
    model_name: str,
    estimator,
    train_X,
    train_Y,
    test_X,
    test_Y,
    cv_folds: int = CV_FOLDS,
    thresholds: Iterable[float] = THRESHOLDS,
) -> Tuple[dict, object, pd.DataFrame, pd.DataFrame]:
    """Подбирает threshold на CV, обучает модель на train и оценивает на test."""
    best_threshold, threshold_scores, fold_scores = select_threshold_kfold(
        estimator=estimator,
        X=train_X,
        y=train_Y,
        thresholds=thresholds,
        cv_folds=cv_folds,
    )

    final_model = clone(estimator)
    final_model.fit(train_X, train_Y)

    test_proba = _positive_class_proba(final_model, test_X, POSITIVE_LABEL)
    test_pred = _predict_by_threshold(test_proba, best_threshold, POSITIVE_LABEL, NEGATIVE_LABEL)

    best_cv_row = threshold_scores.loc[
        threshold_scores["threshold"].eq(best_threshold)
    ].iloc[0]

    result = {
        "model": model_name,
        "best_threshold": best_threshold,
        "cv_accuracy_mean": best_cv_row["mean_validation_score"],
        "cv_accuracy_std": best_cv_row["std_validation_score"],
        "test_accuracy": metrics.accuracy_score(test_Y, test_pred),
        "test_precision": metrics.precision_score(test_Y, test_pred, zero_division=0),
        "test_recall": metrics.recall_score(test_Y, test_pred, zero_division=0),
        "test_f1": metrics.f1_score(test_Y, test_pred, zero_division=0),
    }

    return result, final_model, threshold_scores, fold_scores


def run_experiments(
    train_X,
    train_Y,
    test_X,
    test_Y,
    models: Optional[Dict[str, object]] = None,
    cv_folds: int = CV_FOLDS,
    thresholds: Iterable[float] = THRESHOLDS,
) -> Tuple[pd.DataFrame, Dict[str, object], Dict[str, dict]]:
    """Запускает обучение и сравнение всех моделей."""
    if models is None:
        models = get_models()

    results = []
    trained_models = {}
    threshold_details = {}

    for model_name, estimator in models.items():
        print(f"Training: {model_name}")
        result, fitted_model, threshold_scores, fold_scores = fit_evaluate_model(
            model_name=model_name,
            estimator=estimator,
            train_X=train_X,
            train_Y=train_Y,
            test_X=test_X,
            test_Y=test_Y,
            cv_folds=cv_folds,
            thresholds=thresholds,
        )

        results.append(result)
        trained_models[model_name] = fitted_model
        threshold_details[model_name] = {
            "threshold_scores": threshold_scores,
            "fold_scores": fold_scores,
        }

    results_df = pd.DataFrame(results).sort_values(
        by=["cv_accuracy_mean", "test_accuracy"],
        ascending=False,
    )

    return results_df, trained_models, threshold_details


def print_results(results_df: pd.DataFrame) -> None:
    """Печатает итоговую таблицу в компактном виде."""
    columns = [
        "model",
        "best_threshold",
        "cv_accuracy_mean",
        "cv_accuracy_std",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
    ]

    print("\n" + "=" * 100)
    print("MODEL COMPARISON")
    print("=" * 100)
    print(results_df[columns].round(4).to_string(index=False))
    print()


