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
    display_df = _format_results_for_output(_prepare_results_for_display(results_df))

    print("\n" + "=" * 120)
    print("MODEL COMPARISON")
    print("=" * 100)
    print(display_df.to_string(index=False))
    print()


def _prepare_results_for_display(
    results_df: pd.DataFrame,
    extra_rows: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Сортирует результаты и добавляет rank/delta относительно лучшего."""
    records: list[dict[str, Any]] = []
    if not results_df.empty:
        records.extend(results_df.to_dict(orient="records"))
    if extra_rows:
        records.extend(extra_rows)

    display_df = pd.DataFrame.from_records(records)
    if display_df.empty:
        return display_df

    display_df = (
        display_df.sort_values(
            by=["cv_accuracy_mean", "cv_accuracy_std"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
        .copy()
    )
    display_df["rank"] = np.arange(1, len(display_df) + 1)
    best_cv_accuracy_mean = float(display_df.iloc[0]["cv_accuracy_mean"])
    display_df["delta_from_best"] = (
        display_df["cv_accuracy_mean"] - best_cv_accuracy_mean
    )
    return display_df


def _display_model_label(row: pd.Series | dict[str, Any]) -> str:
    """Возвращает человекочитаемую подпись модели для таблицы."""
    display_name = row.get("display_model")
    if display_name is not None and not pd.isna(display_name):
        return str(display_name)

    model_name = row.get("model")
    if model_name is None or pd.isna(model_name):
        return ""

    if model_name == "Baseline Logistic Regression":
        return "Baseline Logistic Regression (non-equal CV, 5-fold full train)"
    return str(model_name)


def _format_metric(value: Any, signed: bool = False) -> str:
    """Форматирует метрику для Markdown, оставляя пустыми неизвестные значения."""
    if value is None or pd.isna(value):
        return "        "
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def _format_results_for_output(results_df: pd.DataFrame) -> pd.DataFrame:
    """Приводит таблицу результатов к удобному для печати виду."""
    if results_df.empty:
        return results_df

    formatted_df = pd.DataFrame(
        {
            "Rank": results_df["rank"],
            "Model": results_df.apply(_display_model_label, axis=1),
            "CV Accuracy Mean": results_df["cv_accuracy_mean"].map(_format_metric),
            "Delta from Best": results_df["delta_from_best"].map(
                lambda value: _format_metric(value, signed=True)
            ),
            "CV Accuracy Std": results_df["cv_accuracy_std"].map(_format_metric),
            "Test Accuracy": results_df["test_accuracy"].map(_format_metric),
            "Test Precision": results_df["test_precision"].map(_format_metric),
            "Test Recall": results_df["test_recall"].map(_format_metric),
            "Test F1": results_df["test_f1"].map(_format_metric),
        }
    )
    return formatted_df


def _load_baseline_log_result() -> dict[str, Any] | None:
    """Возвращает последнюю сохраненную строку бейзлайна для итогового лога."""
    baseline_data = load_json_file(config.output.baseline_results_file)
    if not baseline_data:
        return None

    result = baseline_data.get("result")
    if isinstance(result, dict):
        return result

    saved_results = baseline_data.get("results")
    if isinstance(saved_results, list) and saved_results:
        last_result = saved_results[-1]
        if isinstance(last_result, dict):
            return last_result

    return None


def _iter_markdown_rows(results_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Объединяет строки текущего запуска и сохраненный baseline."""
    baseline_result = _load_baseline_log_result()
    extra_rows = [baseline_result] if baseline_result is not None else None
    display_df = _prepare_results_for_display(results_df, extra_rows=extra_rows)
    return display_df.to_dict(orient="records")


def _build_markdown_log_lines(results_df: pd.DataFrame) -> list[str]:
    """Формирует строки Markdown для одного запуска эксперимента."""
    display_rows = _iter_markdown_rows(results_df)
    best_row = display_rows[0]
    current_time = datetime.now().strftime("%H:%M %d-%m-%Y")
    headers = [
        "Rank",
        "Model",
        "CV Accuracy Mean",
        "Delta from Best",
        "CV Accuracy Std",
        "Test Accuracy",
        "Test Precision",
        "Test Recall",
        "Test F1",
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
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for row in display_rows:
        model_display = _display_model_label(row)
        if row["rank"] == 1:
            model_display = f"**{model_display}**"
        row_values = [
            str(int(row["rank"])),
            model_display,
            _format_metric(row.get("cv_accuracy_mean")),
            _format_metric(row.get("delta_from_best"), signed=True),
            _format_metric(row.get("cv_accuracy_std")),
            _format_metric(row.get("test_accuracy")),
            _format_metric(row.get("test_precision")),
            _format_metric(row.get("test_recall")),
            _format_metric(row.get("test_f1")),
        ]
        lines.append("| " + " | ".join(row_values) + " |")

    lines.extend(
        [
            "\n" + "-" * 40,
            "### BEST MODEL",
            f"**{_display_model_label(best_row)}**",
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
