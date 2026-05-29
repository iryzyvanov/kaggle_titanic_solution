# 👉 здесь вся логика подготовки данных

# 💡 Что внутри:
# очистка данных
# создание новых признаков
# кодирование категорий

from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np


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
NUMERIC_FEATURES = [
    'Age',
    'FamilySize',
    'FareLog',
    'Pclass'
]
FEATURES_TO_DROP = ['PassengerId','Name','Ticket','SibSp','Parch','Fare','Cabin','Embarked']

def extract_and_set_honorifics(df):
    df['Honorific'] = df.Name.str.extract(r'([A-Za-z]+)\.')

# def fill_age_by_honorifics(df):
#     for honorific in AGE_BY_INITIAL_DOUBLE:
#         df.loc[(df.Age.isnull()) & (df.Honorific == honorific), 'Age'] = AGE_BY_INITIAL_DOUBLE[honorific]
#     df.drop('Honorific',axis=1,inplace=True)

def fill_age_by_honorifics(df):

    for honorific in AGE_BY_INITIAL_DOUBLE:
        mask = (df.Age.isnull()
            & (df.Honorific == honorific))

        df.loc[mask, 'Age'] = (AGE_BY_INITIAL_DOUBLE[honorific])

    # fallback
    df['Age'] = df['Age'].fillna(df['Age'].median())

    df.drop('Honorific', axis=1, inplace=True)

def create_FamilySize(df):
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

def create_IsAlone(df):
    df["IsAlone"] = (
        df["FamilySize"] == 1).astype(int)

# def create_FareLog(df):
#     df['FareLog'] = np.log1p(df['Fare'])
def create_FareLog(df):
    df['Fare'] = df['Fare'].fillna(
        df['Fare'].median()
    )
    df['FareLog'] = np.log1p(df['Fare'])

def drop_feauteres(df):
    df.drop(FEATURES_TO_DROP,axis=1,inplace=True)

def encode_categorical(df):
    df = pd.get_dummies(df, columns=['Sex'], drop_first=True)
    df = pd.get_dummies(df, columns=['Deck'], prefix='Deck', drop_first=True)
    return df

def fit_robust_scaler(df):
    scaler = RobustScaler()
    scaler.fit(df[NUMERIC_FEATURES])
    return scaler

def transform_robust_scaler(df, scaler):
    df[NUMERIC_FEATURES] = scaler.transform(
        df[NUMERIC_FEATURES]
    )
    return df

def preprocess_Deck(df):
    df['Deck'] = df['Cabin'].fillna('U').str[0]
    df['Deck'] = df['Deck'].replace(['T', 'G', 'F', 'A'], 'Rare')

def preprocessing_data(
    df,
    fit_robust_sc=False,
    scaler=None,
    train_columns=None
):
    df_copy =  df.copy()
    
    # ========= FEATURE ENGINEERING =========

    extract_and_set_honorifics(df_copy)

    normalize_honorifics(df_copy)

    fill_age_by_honorifics(df_copy)

    create_FamilySize(df_copy)

    create_IsAlone(df_copy)

    create_FareLog(df_copy)

    preprocess_Deck(df_copy)

    drop_feauteres(df_copy)

    # ========= ENCODING =========

    df_copy = encode_categorical(df_copy)

    # ========= SCALING =========

    if fit_robust_sc:

        scaler = fit_robust_scaler(df_copy)

    df_copy = transform_robust_scaler(df_copy, scaler)

    # ========= ALIGN TEST COLUMNS =========
    # Выравниваем тестовую выборку по обучающей
    if train_columns is not None:

        df = df.reindex(
            columns=train_columns,
            fill_value=0
        )

    return df_copy, scaler