# Kaggle Titanic Solution

Репозиторий с решением задачи **Titanic: Machine Learning from Disaster** на Kaggle. Проект содержит полный пайплайн: предобработку данных, подбор гиперпараметров, сравнение нескольких моделей, сборку ансамбля и генерацию submission-файлов для Kaggle.

## Что внутри

- feature engineering для табличных данных Titanic;
- stratified train/test split для внутренней оценки;
- подбор гиперпараметров через **Optuna**;
- сравнение нескольких моделей;
- ансамблирование лучших моделей;
- сохранение результатов в Markdown-лог и в `submission.csv`;
- отдельные ноутбуки для EDA и анализа признаков.

## Структура проекта

```text
kaggle_titanic_solution/
├── notebooks/
│   ├── EDA.ipynb
│   └── preprocess.ipynb
├── src/
│   ├── baseline.py
│   ├── config.py
│   ├── main.py
│   ├── model_builder.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── trainer.py
│   ├── nn_model.py
│   └── utils.py
└── README.md
```

## Используемые данные

Ожидается стандартная структура датасета Kaggle Titanic:

- `data/train.csv`
- `data/test.csv`

В конфиге пути заданы именно так.

## Как запустить

### 1. Обучение и генерация submission

Основной входной файл — `src/main.py`.

```bash
python src/main.py
```

Режим запуска задается в `src/config.py`:

- `TUNE` — подбор гиперпараметров через Optuna и обучение;
- `TRAIN` — быстрое обучение по сохраненным параметрам, если они есть;
- `PREDICT` — инференс по сохраненным моделям.

После запуска скрипт:

1. читает `data/train.csv`;
2. делает train/test split;
3. выполняет предобработку;
4. оптимизирует модели через Optuna;
5. сравнивает метрики;
6. обучает лучшую модель и ансамбль на полном train-датасете;
7. создает файлы:
   - `submission.csv`
   - `submission_ensemble.csv`

### 2. Логирование результатов

Итоги экспериментов добавляются в файл:

```text
log_final_Table.md
```

Там сохраняются метрики моделей и конфигурация запуска.

## Предобработка признаков
Основная логика находится в `preprocessing.py`.

`TitanicFeatureEngineer` создает признаки:

| Признак | Описание |
| :--- | :--- |
| `Honorific` | Обращение из имени пассажира: `Mr`, `Mrs`, `Miss`, `Master`, `Other` и т.д. |
| `FamilySize` | Размер семьи: `SibSp + Parch + 1`. |
| `IsAlone` | Бинарный признак одиночного путешествия. |
| `FareLog` | Логарифмированный тариф `log1p(Fare)`. |
| `Deck` | Первая буква `Cabin`; пропуски заменяются на `U`, редкие палубы группируются в `Rare`. |

После генерации признаков удаляются:

```python
["PassengerId", "Name", "Ticket", "Cabin", "SibSp", "Parch", "Fare"]
```

Отдельный трансформер `GroupAgeImputer` заполняет `Age` медианой по `Honorific`, а если группы нет — общей медианой train-части.

## Предобработка для моделей

`get_preprocessor()` собирает `ColumnTransformer`:

- категориальные признаки:
  - `Sex`
  - `Deck`
  - `Pclass`
  - `Honorific`
  - `Embarked`
- бинарные признаки:
  - `IsAlone`
- остальные числовые признаки:
  - заполнение медианой;
  - масштабирование через `RobustScaler`.

Категориальные признаки проходят:

```text
SimpleImputer(strategy="most_frequent")
OneHotEncoder(drop="first", handle_unknown="ignore")
```

## Модели

Фабрики моделей находятся в `model_builder.py`.

Поддерживаемые модели:

- Logistic Regression
- RBF SVM
- KNN
- Gaussian Naive Bayes
- Decision Tree
- Random Forest
- CatBoost
- XGBoost
- LightGBM
- PyTorch NN

Список реально запускаемых моделей задается в `config.py`:

```python
config.model.active_models
```

## Ансамбли

Тип ансамбля задается в `config.py`:

```python
config.model.ensemble = "voting"
```

Поддерживаются:

| Значение | Описание |
| :--- | :--- |
| `averaging` | Soft Voting: усреднение вероятностей. |
| `voting` | Hard Voting: голосование классов. |
| `stacking_lr` | Stacking с `LogisticRegression` как мета-моделью. |
| `stacking_ridge` | Stacking с `RidgeClassifier` как мета-моделью. |

Список моделей для ансамбля задается так:

```python
config.model.ensemble_models = ["CatBoost", "Logistic Regression", "KNN"]
```

## Конфигурация

Все основные параметры собраны в `src/config.py`:

- seed;
- пути к данным;
- размер hold-out выборки;
- число fold-ов для CV;
- список активных моделей;
- число trials для Optuna;
- имена выходных файлов.

## Логи и артефакты

| Файл | Что хранит |
| :--- | :--- |
| `log_final_Table.md` | Markdown-таблицы сравнений моделей по запускам. |
| `models/best_params.json` | Лучшие параметры Optuna и результаты моделей. |
| `models/baseline_results.json` | Метрики отдельного бейзлайна. |
| `models/*.joblib` | Сохраненные обученные pipeline-модели. |
| `models/ensemble_<type>.joblib` | Сохраненный ансамбль. |
| `subs/baseline_submission.csv` | Submission от бейзлайна. |
| `subs/submission.csv` | Submission от лучшей одиночной модели. |
| `subs/submission_ensemble.csv` | Submission от ансамбля. |


## Результаты моделей

Финальное сравнение моделей по последнему запуску эксперимента:

- **Seed:** `1415926536`
- **CV Folds:** `5`
- **Ensemble Type:** `voting`
- **Optuna Trials Default:** `100`
- **Optuna Trials Complex:** `300`

| Rank | Model | CV Accuracy | Test Accuracy | Precision | Recall | F1 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | **XGBoost** | **0.8625 ± 0.0086** | 0.7948 | 0.7727 | 0.6602 | 0.7120 |
| 2 | Ensemble (voting) | 0.8555 ± 0.0147 | 0.8022 | 0.7717 | 0.6893 | 0.7282 |
| 3 | LightGBM | 0.8512 ± 0.0119 | 0.7985 | 0.7952 | 0.6408 | 0.7097 |
| 4 | CatBoost | 0.8469 ± 0.0140 | 0.8097 | 0.8095 | 0.6602 | 0.7273 |
| 5 | Random Forest | 0.8417 ± 0.0148 | **0.8246** | **0.8415** | 0.6699 | **0.7460** |
| 6 | RBF SVM | 0.8403 ± 0.0143 | 0.8022 | 0.7778 | 0.6796 | 0.7254 |
| 7 | Logistic Regression | 0.8377 ± 0.0165 | 0.7948 | 0.7353 | 0.7282 | 0.7317 |
| 8 | KNN | 0.8357 ± 0.0075 | 0.7836 | 0.7778 | 0.6117 | 0.6848 |
| 9 | PyTorch NN | 0.8357 ± 0.0204 | 0.7799 | 0.7245 | 0.6893 | 0.7065 |
| 10 | Decision Tree | 0.8225 ± 0.0275 | 0.7761 | 0.7263 | 0.6699 | 0.6970 |
| 11 | Gaussian Naive Bayes | 0.7927 ± 0.0196 | 0.7687 | 0.6847 | **0.7379** | 0.7103 |
| 12 | Baseline Logistic Regression | 0.7924 ± 0.0174 | - | - | - | - |

## Kaggle Submit Scores

| Submission | Kaggle Accuracy |
| :--- | :---: |
| **Ensemble voting CV=5 CatBoost + Logistic Regression + KNN** | **0.77751** |
| Ensemble voting CV=4 (CatBoost, RBF SVM, Logistic Regression)| 0.77511 |
| Ensemble voting CV=2 (XGBoost, RBF SVM, Logistic Regression)| 0.77272 |
| XGBoost  CV=5 | 0.77272 |
| Ensemble voting CV=5 ("XGBoost", "Logistic Regression", "RBF SVM") | 0.77033 |
| Baseline Logistic Regression | 0.76794 |
| CatBoost CV=4 | 0.76794 |
| XGBoost  CV=2 | 0.75598 |

## Автор

github.com/iryzyvanov