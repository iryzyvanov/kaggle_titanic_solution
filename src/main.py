"""Точка входа командной строки для обучения и инференса."""

from __future__ import annotations

from typing import Any

import pandas as pd

from config import config
from train import build_and_train_ensemble
from utils import (
    load_ensemble_model,
    load_json_file,
    save_json_file,
    save_submission_file,
)


def _load_test_data() -> pd.DataFrame:
    """Загружает исходные тестовые данные Kaggle без предобработки."""
    print(f"[I/O] Reading test data from {config.data.test_path}...")
    return pd.read_csv(config.data.test_path)


def generate_submission(best_single_model: Any, trained_ensemble: Any) -> None:
    """Создает файлы отправки Kaggle из обученных моделей в памяти."""
    print("\n" + "=" * 80)
    print("GENERATING KAGGLE SUBMISSION FROM MEMORY (TRAIN/TUNE MODE)")
    print("=" * 80)

    test_data = _load_test_data()

    print("Making predictions...")
    output_pairs = [
        (trained_ensemble, config.output.ensemble_submission_file),
        (best_single_model, config.output.best_model_submission_file),
    ]
    for model, output_path in output_pairs:
        predictions = model.predict(test_data)
        save_submission_file(predictions, test_data["PassengerId"], output_path)


def predict_inference() -> None:
    """Запускает инференс по моделям, уже сохраненным на диске."""
    print("\n" + "=" * 80)
    print("RUNNING INFERENCE (PREDICT MODE)")
    print("=" * 80)

    test_data = _load_test_data()
    ensemble_model = load_ensemble_model()
    test_predictions = ensemble_model.predict(test_data)

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
