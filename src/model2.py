"""
model.py

Обучение и сравнение моделей. (Обновлено: Threshold подбирается внутри Optuna)

train + optuna (cv) -> [best params + best threshold]
   ↓
fit on full train (с лучшими параметрами)
   ↓
evaluate on test (с лучшим порогом)
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
except ImportError:
    CatBoostClassifier = None

import optuna

RANDOM_STATE = 42
CV_FOLDS = 5
POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0



def _get_catboost_task_type() -> str:
    """Определяет, доступна ли видеокарта для обучения CatBoost."""
    if CatBoostClassifier is None:
        return "CPU"
    try:
        from catboost.utils import get_gpu_device_count
        # Если найдена хотя бы одна видеокарта с поддержкой CUDA
        if get_gpu_device_count() > 0:
            return "GPU"
    except Exception:
        # Если библиотека скомпилирована без GPU или возникла ошибка драйверов
        pass
    return "CPU"

#в Титанике небольшой датасет, поэтому переход на GPU замедляет моделирование в целом.
TASK_TYPE = "CPU"
#_get_catboost_task_type()

def get_models(random_state: int = RANDOM_STATE) -> Dict[str, object]:
    """Возвращает базовые модели."""
    models: Dict[str, object] = {
        "Logistic Regression": LogisticRegression(
            max_iter=5_000, random_state=random_state, solver="liblinear"
        ),
        "RBF SVM": svm.SVC(
            kernel="rbf", probability=True, random_state=random_state
        ),
        "KNN": KNeighborsClassifier(),
        "Gaussian Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(random_state=random_state),
    }

    if CatBoostClassifier is not None:
        models["CatBoost"] = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="Accuracy",
            random_seed=random_state,
            task_type=TASK_TYPE,
            verbose=False,
            border_count = 64
        )

    return models

def build_model_by_trial(trial, model_name, random_state=RANDOM_STATE):
    """Строит классификатор на основе гиперпараметров из trial."""
    if model_name == "Logistic Regression":
        penalty_choice = trial.suggest_categorical("penalty", ["l1", "l2"])
        l1_ratio_value = 1.0 if penalty_choice == "l1" else 0.0
        return LogisticRegression(
            C=trial.suggest_float("C", 1e-3, 10, log=True),
            l1_ratio=l1_ratio_value,
            solver="liblinear",
            class_weight=trial.suggest_categorical("class_weight", [None, "balanced"]),
            max_iter=5000,
            random_state=random_state,
        )
    elif model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 700),
            max_depth=trial.suggest_int("max_depth", 2, 10),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            random_state=random_state,
        )
    elif model_name == "RBF SVM":
        return svm.SVC(
            C=trial.suggest_float("C", 1e-3, 100, log=True),
            gamma=trial.suggest_float("gamma", 1e-4, 1, log=True),
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
            max_depth=trial.suggest_int("max_depth", 2, 10),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            random_state=random_state,
        )
    elif model_name == "Gaussian Naive Bayes":
        return GaussianNB(
            var_smoothing=trial.suggest_float("var_smoothing", 1e-12, 1e-6, log=True)
        )
    elif model_name == "CatBoost" and CatBoostClassifier is not None:
        return CatBoostClassifier(
            iterations=trial.suggest_int("iterations", 100, 1000),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            depth=trial.suggest_int("depth", 3, 10),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-3, 10, log=True),
            verbose=False,
            random_seed=random_state,
            task_type=TASK_TYPE,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

def _take_rows(data, indices):
    """Скроем разницу в синтаксисе с помощью адаптера"""
    if hasattr(data, "iloc"):
        return data.iloc[indices]
    return data[indices]

# def _positive_class_proba(model, X, positive_label: int = POSITIVE_LABEL) -> np.ndarray:
#     """Возвращаем вероятность положительного класса из predict_proba."""
#     if not hasattr(model, "predict_proba"):
#         raise TypeError(f"Model {type(model).__name__} does not support predict_proba")

#     classes = list(model.classes_)
#     if positive_label not in classes:
#         raise ValueError(f"Positive label {positive_label} was not found in model classes: {classes}")

#     positive_class_index = classes.index(positive_label)
#     return model.predict_proba(X)[:, positive_class_index]

# def _predict_by_threshold(
#     proba: np.ndarray,
#     threshold: float,
#     positive_label: int = POSITIVE_LABEL,
#     negative_label: int = NEGATIVE_LABEL,
# ) -> np.ndarray:
#     """Возвращает классы по вероятности исходя из выбранного threshold."""
#     return np.where(proba >= threshold, positive_label, negative_label)


def optimize_model(
    model_name,
    X,
    y,
    n_trials,
    cv_folds=5,
    random_state=RANDOM_STATE
):
    """
    Подбор гиперпараметров с помощью Optuna (без порога).
    """
    def objective(trial):
        model = build_model_by_trial(trial, model_name=model_name, random_state=random_state)
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scores = []
        
        for train_idx, valid_idx in cv.split(X, y):
            X_train_fold, X_valid_fold = _take_rows(X, train_idx), _take_rows(X, valid_idx)
            y_train_fold, y_valid_fold = _take_rows(y, train_idx), _take_rows(y, valid_idx)
            
            fold_model = clone(model)
            fold_model.fit(X_train_fold, y_train_fold)
            
            # Используем стандартный predict (порог 0.5 "под капотом")
            valid_pred = fold_model.predict(X_valid_fold)
            
            # Считаем метрику
            score = metrics.accuracy_score(y_valid_fold, valid_pred)
            scores.append(score)
            
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Сохраняем std в trial, чтобы достать его позже для таблицы
        trial.set_user_attr("cv_std", std_score)
        
        return mean_score

    # Запускаем оптимизацию с фиксированным seed для воспроизводимости
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    print(f"\nBEST PARAMS [{model_name}]")
    print(study.best_params)
    print(f"BEST SCORE: {study.best_value:.4f}")

    # Создаем лучшую модель на основе найденных параметров
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
    test_Y,
) -> Tuple[dict, object]:
    """Обучает финальную модель на всем train и оценивает на test."""
    
    final_model = clone(estimator)
    final_model.fit(train_X, train_Y)

    # Стандартный predict
    test_pred = final_model.predict(test_X)

    result = {
        "model": model_name,
        "cv_accuracy_mean": cv_mean,
        "cv_accuracy_std": cv_std,
        "test_accuracy": metrics.accuracy_score(test_Y, test_pred),
        "test_precision": metrics.precision_score(test_Y, test_pred, zero_division=0),
        "test_recall": metrics.recall_score(test_Y, test_pred, zero_division=0),
        "test_f1": metrics.f1_score(test_Y, test_pred, zero_division=0),
    }

    return result, final_model


def run_experiments(
    train_X,
    train_Y,
    test_X,
    test_Y,
    models: Optional[Dict[str, object]] = None,
    cv_folds: int = CV_FOLDS,
    n_trials: int = 100,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Запускает обучение и сравнение всех моделей."""
    if models is None:
        models = get_models()

    results = []
    trained_models = {}

    for model_name, estimator in models.items():

        print(f"\nOPTIMIZING: {model_name}")

        best_model, study = optimize_model(
            model_name=model_name,
            X=train_X,
            y=train_Y,
            n_trials=n_trials,
            cv_folds=cv_folds,
        )
        
        cv_mean = study.best_value
        cv_std = study.best_trial.user_attrs.get("cv_std", np.nan)

        # Обучение на полных данных и оценка на тесте
        result, fitted_model = fit_evaluate_model(
            model_name=model_name,
            estimator=best_model,
            cv_mean=cv_mean,
            cv_std=cv_std,
            train_X=train_X,
            train_Y=train_Y,
            test_X=test_X,
            test_Y=test_Y,
        )

        results.append(result)
        trained_models[model_name] = fitted_model

    # results_df = pd.DataFrame(results).sort_values(
    #     by=["cv_accuracy_mean", "test_accuracy"],
    #     ascending=False,
    # )
    # Сортируем сначала по среднему, затем по отклонению
    # Mean - по убыванию (False), Std - по возрастанию (True)
    results_df = pd.DataFrame(results).sort_values(
            by=["cv_accuracy_mean", "cv_accuracy_std"], 
            ascending=[False, True], 
        )
    return results_df, trained_models


def print_results(results_df: pd.DataFrame) -> None:
    """Печатает итоговую таблицу в компактном виде."""
    columns = [
        "model",
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