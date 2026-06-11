"""Инженерия признаков и вспомогательная предобработка для Titanic."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin

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
    'Dona': 'Mrs'
}

# AGE_BY_HONORIFIC = {
#     "Mr": 32.739609,
#     "Mrs": 35.981818,
#     "Master": 4.574167,
#     "Miss": 21.860,
#     "Other": 45.888889,
# }

class GroupAgeImputer(BaseEstimator, TransformerMixin):
    """
    Заполняет Age медианой по Honorific.

    Медианы рассчитываются во время fit(), поэтому при использовании
    внутри Pipeline они вычисляются только на обучающей части CV-фолда.
    """
    def fit(self, X: pd.DataFrame, y=None):
        required_columns = {"Age", "Honorific"}
        missing = required_columns.difference(X.columns)

        if missing:
            raise ValueError(
                f"Отсутствуют необходимые колонки: {sorted(missing)}"
            )

        self.medians_ = (
            X.groupby("Honorific")["Age"]
            .median()
            .to_dict()
        )
        self.global_median_ = X["Age"].median()

        if pd.isna(self.global_median_):
            raise ValueError("Невозможно рассчитать общую медиану Age.")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()

        ages_by_group = X_out["Honorific"].map(self.medians_)

        X_out["Age"] = (
            X_out["Age"]
            .fillna(ages_by_group)
            .fillna(self.global_median_)
        )

        return X_out

FEATURES_TO_DROP = ["PassengerId", "Name", "Ticket", "Cabin", "SibSp", "Parch"]
RARE_DECKS = ["T", "G", "F", "A"]


def extract_and_set_honorifics(df: pd.DataFrame) -> pd.DataFrame:
    """Извлекает обращения пассажиров из имен в столбец ``Honorific``."""
    df["Honorific"] = df["Name"].str.extract(r"([A-Za-z]+)\.")
    return df


def normalize_honorifics(df: pd.DataFrame) -> pd.DataFrame:
    """Сводит редкие и похожие обращения к компактному набору значений."""
    df["Honorific"] = df["Honorific"].replace(REPLACEMENT_HONORIFICS)
    return df


def fill_age_by_honorifics(df: pd.DataFrame) -> pd.DataFrame:
    """Заполняет пропуски возраста медианой для соответствующего обращения."""
    for honorific, age in AGE_BY_HONORIFIC.items():
        missing_age = df["Age"].isna() & (df["Honorific"] == honorific)
        df.loc[missing_age, "Age"] = age

    return df#.drop(columns="Honorific")

def fill_embarked(df: pd.DataFrame) -> pd.DataFrame:
    """Заполняет пропуски в порту посадки самой частой категорией."""
    if 'Embarked' in df.columns:
        df['Embarked'] = df['Embarked'].fillna('S')
    return df

def create_family_size(df: pd.DataFrame) -> pd.DataFrame:
    """Создает общий размер семьи из SibSp и Parch."""
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    return df


def create_is_alone(df: pd.DataFrame) -> pd.DataFrame:
    """Создает бинарный признак пассажиров, путешествующих без семьи."""
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    return df


def create_fare_log(df: pd.DataFrame) -> pd.DataFrame:
    """Создает логарифмированный признак Fare с корректной обработкой нулей."""
    df["FareLog"] = np.log1p(df["Fare"])
    return df


def preprocess_deck(df: pd.DataFrame) -> pd.DataFrame:
    """Извлекает палубу из Cabin и группирует редкие значения как ``Rare``."""
    df["Deck"] = df["Cabin"].fillna("U").str[0]
    df["Deck"] = df["Deck"].replace(RARE_DECKS, "Rare")
    return df


def drop_features(df: pd.DataFrame) -> pd.DataFrame:
    """Удаляет столбцы, которые больше не нужны после инженерии признаков."""
    return df.drop(columns=FEATURES_TO_DROP)


def preprocessing_data(df: pd.DataFrame) -> pd.DataFrame:
    """Возвращает признаки Titanic без изменения входного DataFrame."""
    df_processed = df.copy()

    preprocessing_steps = (
        extract_and_set_honorifics,
        normalize_honorifics,
        #fill_age_by_honorifics,
        create_family_size,
        create_is_alone,
        create_fare_log,
        preprocess_deck,
        drop_features,
    )

    for step in preprocessing_steps:
        df_processed = step(df_processed)
    print(df_processed.columns)
    return df_processed


def get_preprocessor() -> ColumnTransformer:
    """Создает sklearn-трансформер для использования внутри конвейера модели."""
    categorical_features = list(config.data.categorical_features)
    binary_features = list(config.data.binary_features)

    categorical_transformer = Pipeline(
        steps=[("imputer",
                SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False))] )
    
    numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')), 
            ('scaler', RobustScaler())
        ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat",
                categorical_transformer,
                categorical_features,
            ),
            ("bin",
                "passthrough",
                binary_features,),
        ],
        remainder=numeric_transformer,
        verbose_feature_names_out=False,
    )
    preprocessor.set_output(transform="pandas")
    return preprocessor
