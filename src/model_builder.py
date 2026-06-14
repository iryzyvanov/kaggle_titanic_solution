"""Фабрики для всех поддерживаемых классификаторов Titanic."""

from collections.abc import Callable
from typing import Any

from sklearn import svm
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from config import config

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    import torch
    from nn_model import PyTorchTitanicClassifier
except ImportError:
    PyTorchTitanicClassifier = None

ModelBuilder = Callable[[Any, int], BaseEstimator]



def _build_pytorch_nn(trial: Any, random_state: int) -> BaseEstimator:
    # Заставляем Optuna искать лучшие параметры для нейросети
    return PyTorchTitanicClassifier(
        epochs=trial.suggest_int("epochs", 50, 200),
        lr=trial.suggest_float("lr", 1e-4, 1e-1, log=True),
        batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
        random_state=random_state
    )


def _build_logistic_regression(trial: Any, random_state: int) -> BaseEstimator:
    penalty_choice = trial.suggest_categorical("penalty", ["l1", "l2"])
    l1_ratio_value = 1.0 if penalty_choice == "l1" else 0.0

    return LogisticRegression(
        C=trial.suggest_float("C", 1e-4, 100, log=True),
        l1_ratio=l1_ratio_value,
        solver="liblinear",
        class_weight=trial.suggest_categorical("class_weight", [None, "balanced"]),
        max_iter=5000,
        random_state=random_state,
    )


def _build_random_forest(trial: Any, random_state: int) -> BaseEstimator:
    return RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 700),
        criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]),
        max_depth=trial.suggest_int("max_depth", 2, 10),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        random_state=random_state,
    )


def _build_rbf_svm(trial: Any, random_state: int) -> BaseEstimator:
    gamma_choice = trial.suggest_categorical(
        "gamma_choice",
        ["scale", "auto", "float"],
    )
    gamma_value = (
        trial.suggest_float("gamma_float", 1e-4, 1, log=True)
        if gamma_choice == "float"
        else gamma_choice
    )

    return svm.SVC(
        C=trial.suggest_float("C", 1e-3, 1000, log=True),
        gamma=gamma_value,
        probability=True,
        kernel="rbf",
        random_state=random_state,
    )


def _build_knn(trial: Any, random_state: int) -> BaseEstimator:
    return KNeighborsClassifier(
        n_neighbors=trial.suggest_int("n_neighbors", 3, 25),
        weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
        p=trial.suggest_int("p", 1, 2),
    )


def _build_decision_tree(trial: Any, random_state: int) -> BaseEstimator:
    return DecisionTreeClassifier(
        criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]),
        max_depth=trial.suggest_int("max_depth", 2, 10),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
        random_state=random_state,
    )


def _build_gaussian_naive_bayes(trial: Any, random_state: int) -> BaseEstimator:
    return GaussianNB(
        var_smoothing=trial.suggest_float(
            "var_smoothing",
            1e-10,
            1e-2,
            log=True,
        )
    )


def _build_catboost(trial: Any, random_state: int) -> BaseEstimator:
    return CatBoostClassifier(
        iterations=trial.suggest_int("iterations", 100, 1000),
        learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        depth=trial.suggest_int("depth", 3, 7),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-1, 30, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        bootstrap_type="Bernoulli",
        min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 1, 10),
        verbose=False,
        random_seed=random_state,
        task_type=config.model.task_type,
    )


def _build_xgboost(trial: Any, random_state: int) -> BaseEstimator:
    return XGBClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 700),
        learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 7),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        gamma=trial.suggest_float("gamma", 0, 5),
        random_state=random_state,
        eval_metric="logloss",
        n_jobs=-1,
    )


def _build_lightgbm(trial: Any, random_state: int) -> BaseEstimator:
    max_depth = trial.suggest_int("max_depth", 3, 7)
    max_leaves = (2**max_depth) - 1

    return LGBMClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 700),
        learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        max_depth=max_depth,
        num_leaves=trial.suggest_int("num_leaves", 2, max_leaves),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_samples=trial.suggest_int("min_child_samples", 2, 20),
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


MODEL_BUILDERS: dict[str, ModelBuilder] = {
    "Logistic Regression": _build_logistic_regression,
    "Random Forest": _build_random_forest,
    "RBF SVM": _build_rbf_svm,
    "KNN": _build_knn,
    "Decision Tree": _build_decision_tree,
    "Gaussian Naive Bayes": _build_gaussian_naive_bayes,
    "PyTorch NN": (PyTorchTitanicClassifier, "torch", _build_pytorch_nn)
    
}

OPTIONAL_MODEL_BUILDERS: dict[str, tuple[Any, str, ModelBuilder]] = {
    "CatBoost": (CatBoostClassifier, "catboost", _build_catboost),
    "XGBoost": (XGBClassifier, "xgboost", _build_xgboost),
    "LightGBM": (LGBMClassifier, "lightgbm", _build_lightgbm),
    
}


def build_model_by_trial(
    trial: Any,
    model_name: str,
    random_state: int = config.general.seed,
) -> BaseEstimator:
    """Создает классификатор по гиперпараметрам из испытания Optuna."""
    if model_name in OPTIONAL_MODEL_BUILDERS:
        model_class, package_name, builder = OPTIONAL_MODEL_BUILDERS[model_name]
        if model_class is None:
            raise ImportError(
                f"Model '{model_name}' requires optional package '{package_name}'."
            )
        return builder(trial, random_state)

    try:
        builder = MODEL_BUILDERS[model_name]
    except KeyError as error:
        raise ValueError(f"Unknown model: {model_name}") from error

    return builder(trial, random_state)
