"""Общие утилиты для обучения, логирования и сохранения артефактов."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd

from config import config

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None


class OptunaLogFormatter(logging.Filter):
    """Сжимает подробные сообщения Optuna для вывода в консоль."""

    def __init__(self, digits: int = 4) -> None:
        super().__init__()
        self.digits = digits

    def filter(self, record: logging.LogRecord) -> bool:
        msg = str(record.msg)
        msg = re.sub(
            r"(\d+\.\d{5,})",
            lambda match: f"{float(match.group(1)):.{self.digits}f}",
            msg,
        )
        msg = msg.replace("finished with value:", "->")
        msg = msg.replace("and parameters:", "|")
        msg = msg.replace("with value:", "=")

        record.msg = msg
        return True


def configure_optuna_logging() -> Any:
    """Настраивает логирование Optuna и возвращает импортированный модуль."""
    optuna.logging.set_verbosity(optuna.logging.INFO)
    logger = optuna.logging.get_logger("optuna")

    if logger.handlers:
        handler = logger.handlers[0]
        handler.setFormatter(
            logging.Formatter("[I %(asctime)s] %(message)s", datefmt="%H:%M:%S")
        )
        handler.addFilter(OptunaLogFormatter())

    return optuna


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Понижает типы числовых столбцов, чтобы уменьшить расход памяти."""
    start_mem = df.memory_usage().sum() / 1024**2

    for column in df.columns:
        column_type = df[column].dtype
        if (
            column_type == object
            or column_type.name == "category"
            or column_type == "bool"
        ):
            continue

        column_min = df[column].min()
        column_max = df[column].max()

        if str(column_type).startswith("int"):
            if (
                column_min > np.iinfo(np.int8).min
                and column_max < np.iinfo(np.int8).max
            ):
                df[column] = df[column].astype(np.int8)
            elif (
                column_min > np.iinfo(np.int16).min
                and column_max < np.iinfo(np.int16).max
            ):
                df[column] = df[column].astype(np.int16)
            elif (
                column_min > np.iinfo(np.int32).min
                and column_max < np.iinfo(np.int32).max
            ):
                df[column] = df[column].astype(np.int32)
        elif (
            column_min > np.finfo(np.float16).min
            and column_max < np.finfo(np.float16).max
        ):
            df[column] = df[column].astype(np.float16)
        elif (
            column_min > np.finfo(np.float32).min
            and column_max < np.finfo(np.float32).max
        ):
            df[column] = df[column].astype(np.float32)

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f"Исходный размер памяти: {start_mem:.4f} MB")
        print(f"Размер памяти после оптимизации: {end_mem:.4f} MB")

    return df


def _get_catboost_task_type() -> str:
    """Возвращает тип задачи GPU для CatBoost, если доступна CUDA."""
    if CatBoostClassifier is None:
        return "CPU"

    try:
        from catboost.utils import get_gpu_device_count

        if get_gpu_device_count() > 0:
            return "GPU"
    except Exception:
        pass

    return "CPU"


def print_results(results_df: pd.DataFrame) -> None:
    """Печатает компактную таблицу сравнения моделей."""
    columns = [
        "model",
        "cv_accuracy_mean",
        "cv_accuracy_std",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
    ]

    print("\n" + "=" * 120)
    print("MODEL COMPARISON")
    print("=" * 100)
    print(results_df[columns].round(4).to_string(index=False))
    print()


def _build_markdown_log_lines(results_df: pd.DataFrame) -> list[str]:
    """Формирует строки Markdown для одного запуска эксперимента."""
    best_model_name = results_df.iloc[0]["model"]
    current_time = datetime.now().strftime("%H:%M %d-%m-%Y")
    headers = [
        "Model",
        "CV Accuracy Mean",
        "CV Accuracy Std",
        "Test Accuracy",
        "Test Precision",
        "Test Recall",
        "Test F1",
        "Submit Accuracy",
    ]

    lines = [
        "\n" + "=" * 80,
        f"### MODEL COMPARISON ({current_time})",
        "=" * 80 + "\n",
        "**Run Configuration:**",
        f"- **Seed:** `{config.general.seed}`",
        f"- **CV Folds:** `{config.model.cv_folds}`",
        f"- **Ensemble Type:** `{config.model.ensemble}`",
        f"- **Optuna Trials (Default):** `{config.optuna.n_trials_default}`",
        f"- **Optuna Trials (Complex):** `{config.optuna.n_trials_complex}`\n",
        "| " + " | ".join(headers) + " |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |:---: |",
    ]

    for _, row in results_df.iterrows():
        model_name = row["model"]
        model_display = (
            f"**{model_name}**" if model_name == best_model_name else model_name
        )
        row_values = [
            model_display,
            f"{row['cv_accuracy_mean']:.4f}",
            f"{row['cv_accuracy_std']:.4f}",
            f"{row['test_accuracy']:.4f}",
            f"{row['test_precision']:.4f}",
            f"{row['test_recall']:.4f}",
            f"{row['test_f1']:.4f}",
            "        ",
        ]
        lines.append("| " + " | ".join(row_values) + " |")

    lines.extend(
        [
            "\n" + "-" * 40,
            "### BEST MODEL",
            f"**{best_model_name}**",
            "-" * 40 + "\n",
        ]
    )
    return lines


def append_results_to_markdown_log(
    results_df: pd.DataFrame,
    file_path: str | Path = config.output.log_file,
) -> None:
    """Добавляет Markdown-сводку эксперимента в настроенный лог-файл."""
    if results_df.empty:
        print("DataFrame с результатами пуст. Запись отменена.")
        return

    path = Path(file_path)
    with path.open("a", encoding="utf-8") as file:
        file.write("\n".join(_build_markdown_log_lines(results_df)) + "\n")

    print(f"Результаты успешно добавлены в конец файла: {path}")


def _safe_model_name(model_name: str) -> str:
    """Преобразует отображаемое имя модели в стабильное имя файла."""
    return model_name.replace(" ", "_")


def save_all_models(
    trained_models: dict[str, Any],
    ensemble_model: Any,
) -> None:
    """Сохраняет обученные базовые модели и настроенный ансамбль."""
    models_dir = Path(config.output.folder_for_joblib)
    models_dir.mkdir(parents=True, exist_ok=True)

    for name, fitted_model in trained_models.items():
        model_path = models_dir / f"{_safe_model_name(name)}.joblib"
        joblib.dump(fitted_model, model_path)

    ensemble_path = models_dir / f"ensemble_{config.model.ensemble}.joblib"
    joblib.dump(ensemble_model, ensemble_path)
    print(f"Все модели и ансамбль успешно сохранены в папку {models_dir}")


def save_submission_file(
    predictions: np.ndarray,
    passenger_ids: pd.Series,
    output_path: str | Path,
) -> None:
    """Создает и сохраняет файл отправки Kaggle."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "PassengerId": passenger_ids,
            "Survived": predictions,
        }
    ).to_csv(output_path, index=False)
    print(f"[SUCCESS] Submission saved to: {output_path}")


def get_loaded_models(model_names: list[str]) -> list[Any]:
    """Загружает обученные базовые модели по именам из папки models."""
    loaded_models = []
    models_dir = Path(config.output.folder_for_joblib)

    for model_name in model_names:
        model_path = models_dir / f"{_safe_model_name(model_name)}.joblib"

        if model_path.exists():
            loaded_models.append(joblib.load(model_path))
            print(f"  [+] {model_name} -> Успешно загружена")
        else:
            print(f" Модель '{model_name}' не найдена по пути {model_path}!")

    return loaded_models


def load_json_file(file_path: str | Path) -> dict[str, Any] | None:
    """Загружает JSON-файл, если он существует."""
    path = Path(file_path)
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json_file(file_path: str | Path, data: dict[str, Any]) -> None:
    """Сохраняет словарь в JSON, создавая родительские папки при необходимости."""
    path = Path(file_path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def load_ensemble_model() -> Any:
    """Загружает сохраненный ансамбль для чистого режима инференса."""
    model_path = (
        Path(config.output.folder_for_joblib)
        / f"ensemble_{config.model.ensemble}.joblib"
    )

    if not model_path.exists():
        raise FileNotFoundError(f"Файл ансамбля не найден: {model_path}")

    print(f"  [+] Ансамбль -> Успешно загружен из {model_path}")
    return joblib.load(model_path)
