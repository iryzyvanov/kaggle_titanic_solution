# 👉 здесь вся логика подготовки данных

# 💡 Что внутри:
# очистка данных
# создание новых признаков
# кодирование категорий

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler

import pandas as pd
import numpy as np
from config import config

REPLACEMENT_HONORIFICS = {
    'Mlle': 'Miss',
    'Mme': 'Miss',
    'Ms': 'Miss',
    'Dr': 'Mr',
    'Major': 'Mr',
    'Lady': 'Mrs',
    'Countess': 'Mrs',
    'Jonkheer': 'Other',
    'Col': 'Other',
    'Rev': 'Other',
    'Capt': 'Mr',
    'Sir': 'Mr',
    'Don': 'Mr'
}
def normalize_honorifics(df):
    df['Honorific'] = df['Honorific'].replace(REPLACEMENT_HONORIFICS)

AGE_BY_INITIAL_DOUBLE = {
    'Mr'    : 32.739609,
    'Mrs'   : 35.981818,
    'Master': 4.574167,
    'Miss'  : 21.860,
    'Other' : 45.888889
}

FEATURES_TO_DROP = ['PassengerId','Name','Ticket','Cabin','Embarked']

def extract_and_set_honorifics(df):
    df['Honorific'] = df.Name.str.extract(r'([A-Za-z]+)\.')


def fill_age_by_honorifics(df):

    for honorific in AGE_BY_INITIAL_DOUBLE:
        mask = (df.Age.isnull()
            & (df.Honorific == honorific))

        df.loc[mask, 'Age'] = (AGE_BY_INITIAL_DOUBLE[honorific])

    df.drop('Honorific', axis=1, inplace=True)

def create_FamilySize(df):
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

def create_IsAlone(df):
    df["IsAlone"] = (
        df["FamilySize"] == 1).astype(int)

def create_FareLog(df):
    df['FareLog'] = np.log1p(df['Fare'])

def drop_features(df):
    df.drop(FEATURES_TO_DROP,axis=1,inplace=True)


def preprocess_Deck(df):
    df['Deck'] = df['Cabin'].fillna('U').str[0]
    df['Deck'] = df['Deck'].replace(['T', 'G', 'F', 'A'], 'Rare')

def preprocessing_data(df):
    df_copy =  df.copy()
    
    # ========= FEATURE ENGINEERING =========

    extract_and_set_honorifics(df_copy)

    normalize_honorifics(df_copy)

    fill_age_by_honorifics(df_copy)

    create_FamilySize(df_copy)

    create_IsAlone(df_copy)

    create_FareLog(df_copy)

    preprocess_Deck(df_copy)

    drop_features(df_copy)

    return df_copy

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def get_preprocessor():
    """Создает трансформер: OHE для категорий, пропускаем бинарные, RobustScaler для всего остального."""
    categorical_features = list(config.data.categorical_features)
    binary_features      = list(config.data.binary_features)

    # Создаем конвейер для числовых фичей: честное заполнение пропусков -> масштабирование
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    preprocessor = ColumnTransformer(transformers=
            [('cat', OneHotEncoder(
                    drop='first',             # Избегаем ловушки фиктивных переменных
                    handle_unknown='ignore',  # Если в test попадется новая палуба — ставим нули
                    sparse_output=False),     # Возвращаем обычный массив (нужно для деревьев)
                categorical_features ),
                ('bin', 'passthrough', binary_features )
            ],
        # К остальным (числовым) колонкам применяем масштабирование
        remainder=numeric_transformer)
    # Заставляем трансформер возвращать DataFrame
    preprocessor.set_output(transform="pandas")
    return preprocessor