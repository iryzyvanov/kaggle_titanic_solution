

from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler

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
from utils import take_Optuna_with_modify_logs
optuna = take_Optuna_with_modify_logs()




def get_models(random_state: int = config.general.seed) -> Dict[str, object]:
    """Возвращает базовые модели."""
    all_available_models: Dict[str, object] = {
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
        all_available_models["CatBoost"] = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="Accuracy",
            random_seed=random_state,
            task_type=config.model.task_type,
            verbose=False,
            border_count = 64
        )
    active_models = {}
    for model_name in config.model.active_models:
        if model_name in all_available_models:
            active_models[model_name] = all_available_models[model_name]
        else:
            print(f"⚠️ Предупреждение: Модель '{model_name}' запрошена в конфиге, но не найдена или не поддерживается.")
    return active_models

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


def get_preprocessor():
    """Создает трансформер: OHE для категорий, RobustScaler для всего остального."""
    categorical_features = ['Sex', 'Deck']
    
    return ColumnTransformer(
        transformers=[
            (
                'cat', 
                OneHotEncoder(
                    drop='first',             # Избегаем ловушки фиктивных переменных
                    handle_unknown='ignore',  # Если в test попадется новая палуба — ставим нули
                    sparse_output=False       # Возвращаем обычный массив (нужно для деревьев)
                ), 
                categorical_features
            )
        ],
        # К остальным (числовым) колонкам применяем масштабирование
        remainder=RobustScaler() 
    )

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
    test_Y,
) -> Tuple[dict, object]:
    
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
    train_X,
    train_Y,
    test_X,
    test_Y,
    cv_folds: int = config.model.cv_folds
    ) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Запускает обучение и сравнение всех моделей."""

    results = []
    trained_models = {}

    for model_name in config.model.active_models:

        print(f"\nOPTIMIZING: {model_name}")
        
        n_trials = config.optuna.n_trials_complex if model_name in ["CatBoost", "Random Forest"] else config.optuna.n_trials_default
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

    # Сортируем сначала по среднему, затем по отклонению
    # Mean - по убыванию (False), Std - по возрастанию (True)
    results_df = pd.DataFrame(results).sort_values(
            by=["cv_accuracy_mean", "cv_accuracy_std"], 
            ascending=[False, True], 
        )
    return results_df, trained_models


