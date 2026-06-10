from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
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


from config import config

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
    elif model_name == "XGBoost" and XGBClassifier is not None:
        return XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 700),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            # Сужаем глубину, как делали для CatBoost, чтобы не переобучиться
            max_depth=trial.suggest_int("max_depth", 3, 7),
            # Бэггинг для случайной подвыборки строк и колонок
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            # Гамма контролирует "жадность" отсечения листьев (важно для XGBoost)
            gamma=trial.suggest_float("gamma", 0, 5),
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=-1
        )
        
    elif model_name == "LightGBM" and LGBMClassifier is not None:
        max_depth = trial.suggest_int("max_depth", 3, 7)
        # Вычисляем динамическую верхнюю границу для листьев
        max_leaves = int(2**max_depth) - 1
        
        return LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 700),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            max_depth=max_depth,
            # Изменили нижнюю границу с 10 на 2, чтобы избежать конфликта с max_depth=3
            num_leaves=trial.suggest_int("num_leaves", 2, max_leaves),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_samples=trial.suggest_int("min_child_samples", 2, 20),
            random_state=random_state,
            n_jobs=-1,
            verbose=-1 
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
