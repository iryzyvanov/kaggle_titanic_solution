"""Инженерия признаков и вспомогательная предобработка для Titanic."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from config import config

REPLACEMENT_HONORIFICS = {
    "Mlle": "Miss",
    "Mme": "Miss",
    "Ms": "Miss",
    "Dr": "Mr",
    "Major": "Mr",
    "Lady": "Mrs",
    "Countess": "Mrs",
    "Jonkheer": "Other",
    "Col": "Other",
    "Rev": "Other",
    "Capt": "Mr",
    "Sir": "Mr",
    "Don": "Mr",
    "Dona": "Mrs",
}
# AGE_BY_HONORIFIC = {
#     "Mr": 32.739609,
#     "Mrs": 35.981818,
#     "Master": 4.574167,
#     "Miss": 21.860,
#     "Other": 45.888889,
# }

FEATURES_TO_DROP = ["PassengerId", "Name", "Ticket", "Cabin", "SibSp", "Parch", "Fare"]
RARE_DECKS = ["T", "G", "F", "A"]

REQUIRED_COLUMNS = {
    "PassengerId",
    "Name",
    "Ticket",
    "Cabin",
    "SibSp",
    "Parch",
    "Fare",
    "Age",
    "Sex",
    "Pclass",
    "Embarked",
}


def _require_dataframe(X: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise ValueError("TitanicFeatureEngineer expects a pandas DataFrame.")
    return X


def extract_and_set_honorifics(df: pd.DataFrame) -> pd.DataFrame:
    """Извлекает обращения пассажиров из имен в столбец `Honorific`."""
    df = df.copy()
    df["Honorific"] = df["Name"].str.extract(r"([A-Za-z]+)\.", expand=False)
    return df


def normalize_honorifics(df: pd.DataFrame) -> pd.DataFrame:
    """Сводит редкие и похожие обращения к компактному набору значений."""
    df = df.copy()
    df["Honorific"] = df["Honorific"].replace(REPLACEMENT_HONORIFICS)
    return df


def create_family_size(df: pd.DataFrame) -> pd.DataFrame:
    """Создает общий размер семьи из SibSp и Parch."""
    df = df.copy()
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    return df


def create_is_alone(df: pd.DataFrame) -> pd.DataFrame:
    """Создает бинарный признак пассажиров, путешествующих без семьи."""
    df = df.copy()
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    return df


def create_fare_log(df: pd.DataFrame) -> pd.DataFrame:
    """Создает логарифмированный признак Fare с корректной обработкой нулей."""
    df = df.copy()
    df["FareLog"] = np.log1p(df["Fare"])
    return df


def preprocess_deck(df: pd.DataFrame) -> pd.DataFrame:
    """Извлекает палубу из Cabin и группирует редкие значения как `Rare`."""
    df = df.copy()
    df["Deck"] = df["Cabin"].fillna("U").str[0]
    df["Deck"] = df["Deck"].replace(RARE_DECKS, "Rare")
    return df


def drop_features(df: pd.DataFrame) -> pd.DataFrame:
    """Удаляет столбцы, которые больше не нужны после инженерии признаков."""
    return df.drop(columns=FEATURES_TO_DROP)


class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    """Sklearn-совместимый трансформер для инженерии признаков Titanic."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "TitanicFeatureEngineer":
        _require_dataframe(X)
        missing = REQUIRED_COLUMNS.difference(X.columns)
        if missing:
            raise ValueError(f"Отсутствуют необходимые колонки: {sorted(missing)}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = _require_dataframe(X)
        missing = REQUIRED_COLUMNS.difference(X_df.columns)
        if missing:
            raise ValueError(f"Отсутствуют необходимые колонки: {sorted(missing)}")

        df_processed = X_df.copy()
        df_processed = extract_and_set_honorifics(df_processed)
        df_processed = normalize_honorifics(df_processed)
        df_processed = create_family_size(df_processed)
        df_processed = create_is_alone(df_processed)
        df_processed = create_fare_log(df_processed)
        df_processed = preprocess_deck(df_processed)
        df_processed = drop_features(df_processed)
        
        return df_processed


class GroupAgeImputer(BaseEstimator, TransformerMixin):
    """Заполняет Age медианой по Honorific."""

    def fit(self, X: pd.DataFrame, y=None) -> "GroupAgeImputer":
        if not isinstance(X, pd.DataFrame):
            raise ValueError("GroupAgeImputer expects a pandas DataFrame.")

        required_columns = {"Age", "Honorific"}
        missing = required_columns.difference(X.columns)
        if missing:
            raise ValueError(f"Отсутствуют необходимые колонки: {sorted(missing)}")

        self.medians_ = X.groupby("Honorific")["Age"].median().to_dict()
        self.global_median_ = X["Age"].median()

        if pd.isna(self.global_median_):
            raise ValueError("Невозможно рассчитать общую медиану Age.")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("GroupAgeImputer expects a pandas DataFrame.")

        required_columns = {"Age", "Honorific"}
        missing = required_columns.difference(X.columns)
        if missing:
            raise ValueError(f"Отсутствуют необходимые колонки: {sorted(missing)}")

        X_out = X.copy()
        ages_by_group = X_out["Honorific"].map(self.medians_)
        X_out["Age"] = X_out["Age"].fillna(ages_by_group).fillna(self.global_median_)
        return X_out


def preprocessing_data(df: pd.DataFrame) -> pd.DataFrame:
    """Сохраненная совместимая обертка над feature engineering."""
    return TitanicFeatureEngineer().fit_transform(df)


def get_preprocessor() -> ColumnTransformer:
    """Создает sklearn-трансформер для использования внутри конвейера модели."""
    categorical_features = list(config.data.categorical_features)
    binary_features      = list(config.data.binary_features)

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
            ("bin", "passthrough", binary_features),
        ],
        remainder=numeric_transformer,
        verbose_feature_names_out=False,
    )
    preprocessor.set_output(transform="pandas")
    return preprocessor
