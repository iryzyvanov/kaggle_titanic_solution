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
│   ├── config.py
│   ├── main.py
│   ├── model.py
│   ├── preprocessing.py
│   ├── train.py
│   └── utils.py
└── README.md
```

## Используемые данные

Ожидается стандартная структура датасета Kaggle Titanic:

- `data/train.csv`
- `data/test.csv`

В конфиге пути заданы именно так.

## Установка

Проект использует Python-библиотеки:

- `pandas`
- `numpy`
- `scikit-learn`
- `optuna`
- `omegaconf`
- `catboost` *(опционально, если установлен)*

Пример установки:

```bash
pip install pandas numpy scikit-learn optuna omegaconf catboost
```

## Как запустить

### 1. Обучение и генерация submission

Основной входной файл — `src/main.py`.

```bash
python src/main.py
```

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

Логика предобработки находится в `src/preprocessing.py`.

Что делается:

- извлечение обращения из имени пассажира (`Honorific`);
- нормализация редких обращений;
- заполнение пропусков в возрасте по обращению;
- создание признака `FamilySize`;
- создание бинарного признака `IsAlone`;
- логарифмирование `Fare` через `FareLog`;
- извлечение палубы `Deck` из `Cabin`;
- удаление лишних столбцов.

## Модели

В проекте используются несколько классических моделей машинного обучения:

- Logistic Regression
- RBF SVM
- KNN
- Gaussian Naive Bayes
- Decision Tree
- Random Forest
- CatBoost *(если установлен)*

Подбор параметров реализован через Optuna, а список активных моделей задается в `src/config.py`.

## Ансамбли

Поддерживаются разные варианты ансамблирования:

- `averaging` — soft voting / усреднение вероятностей;
- `voting` — hard voting;
- `stacking_lr` — stacking с Logistic Regression;
- `stacking_ridge` — stacking с Ridge Classifier.

Тип ансамбля задается в конфиге:

```python
config.model.ensemble
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

## Примечания

- CatBoost используется только если библиотека установлена.
- Для ускорения и экономии памяти применяется `reduce_mem_usage`.
- Код ориентирован на табличный классификационный пайплайн для Titanic, но отдельные части можно переиспользовать и в других задачах.

## Автор

github.com/iryzyvanov