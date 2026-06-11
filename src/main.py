"""Точка входа командной строки для обучения и инференса."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config import config
from preprocessing import preprocessing_data
from train import build_and_train_ensemble
from utils import (
    get_loaded_models,
    load_json_file,
    reduce_mem_usage,
    save_json_file,
    save_submission_file,
)


def _load_and_preprocess_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Загружает сырые тестовые данные Kaggle и возвращает их с признаками."""
    print(f"[I/O] Reading test data from {config.data.test_path}...")
    test_data = pd.read_csv(config.data.test_path)

    print("[PREPROCESS] Preprocessing test data through pipeline...")
    test_processed = reduce_mem_usage(preprocessing_data(test_data))
    return test_data, test_processed


def generate_submission(best_single_model: Any, trained_ensemble: Any) -> None:
    """Создает файлы отправки Kaggle из обученных моделей в памяти."""
    print("\n" + "=" * 80)
    print("GENERATING KAGGLE SUBMISSION FROM MEMORY (TRAIN/TUNE MODE)")
    print("=" * 80)

    test_data, test_processed = _load_and_preprocess_test()

    print("Making predictions...")
    output_pairs = [
        (trained_ensemble, config.output.ensemble_submission_file),
        (best_single_model, config.output.best_model_submission_file),
    ]
    for model, output_path in output_pairs:
        predictions = model.predict(test_processed)
        save_submission_file(predictions, test_data["PassengerId"], output_path)


def _predict_by_averaging(
    loaded_models: list[Any],
    test_processed: pd.DataFrame,
) -> np.ndarray:
    """Предсказывает классы через усреднение вероятностей положительного класса."""
    probabilities = [
        model.predict_proba(test_processed)[:, 1] for model in loaded_models
    ]
    mean_probs = np.mean(probabilities, axis=0)
    return (mean_probs >= 0.5).astype(int)


def _predict_by_voting(
    loaded_models: list[Any],
    test_processed: pd.DataFrame,
) -> np.ndarray:
    """Предсказывает классы голосованием большинства загруженных моделей."""
    from scipy import stats

    class_predictions = [model.predict(test_processed) for model in loaded_models]
    test_predictions, _ = stats.mode(class_predictions, axis=0, keepdims=False)
    return test_predictions


def predict_inference() -> None:
    """Запускает инференс по моделям, уже сохраненным на диске."""
    print("\n" + "=" * 80)
    print("RUNNING INFERENCE (PREDICT MODE)")
    print("=" * 80)

    test_data, test_processed = _load_and_preprocess_test()
    ensemble_type = config.model.ensemble

    if "stacking" in ensemble_type:
        print(
            "На текущий момент stacking доступен в режиме TRAIN. "
            "Сначала запустите TRAIN."
        )
        return

    loaded_models = get_loaded_models(config.model.ensemble_models)
    if not loaded_models:
        print("Ошибка: Не удалось загрузить ни одной модели!")
        return

    if ensemble_type == "averaging":
        print(f"Выбран {ensemble_type}. Cборка вероятностей из базовых моделей...")
        test_predictions = _predict_by_averaging(loaded_models, test_processed)
    elif ensemble_type == "voting":
        test_predictions = _predict_by_voting(loaded_models, test_processed)
    else:
        raise ValueError(f"Не поддерживаемый тип ансамбля: {ensemble_type}")

    save_submission_file(
        predictions=test_predictions,
        passenger_ids=test_data["PassengerId"],
        output_path=config.output.ensemble_submission_file,
    )


def main() -> None:
    """Выбирает обучение, подбор параметров или инференс по режиму конфига."""
    mode = config.general.mode
    params_path = config.output.params_file

    print("=" * 80)
    print(f"СТАРТ ПРОГРАММЫ В РЕЖИМЕ: {mode}")
    print("=" * 80)

    if mode in ["TRAIN", "TUNE"]:
        saved_data = None

        if mode == "TRAIN":
            saved_data = load_json_file(params_path)
            if not saved_data:
                print(
                    f"Файл {params_path} не найден! "
                    "Принудительно запускаем поиск параметров (TUNE)."
                )

        best_single_model, trained_ensemble, exp_results = build_and_train_ensemble(
            saved_data=saved_data
        )

        if saved_data is None:
            save_json_file(params_path, exp_results)
            print(f"\n[INFO] Все новые гиперпараметры сохранены в файл: {params_path}")

        generate_submission(best_single_model, trained_ensemble)
    elif mode == "PREDICT":
        predict_inference()


if __name__ == "__main__":
    main()
