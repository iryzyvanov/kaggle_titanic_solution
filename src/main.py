import os
import joblib
import pandas as pd
import numpy as np

from config import config
from preprocessing import preprocessing_data
from train import build_and_train_ensemble
from utils import load_json_file, save_json_file, reduce_mem_usage, get_loaded_models, save_submission_file


def _load_and_preprocess_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
     Загружает сырой тест конвейера, берет статистику и 
    возвращает исходный DataFrame и обработанную матрицу признаков.
    """
    print(f"[I/O] Reading test data from {config.data.test_path}...")
    test_data = pd.read_csv(config.data.test_path)
    
    print("[PREPROCESS] Preprocessing test data through pipeline...")
    test_processed = preprocessing_data(test_data)
    test_processed = reduce_mem_usage(test_processed)
    
    return test_data, test_processed





def generate_submission(best_single_model, trained_ensemble) -> None:
    """Принимает обученные в памяти модели и генерирует файлы для Kaggle."""
    print("\n" + "=" * 80)
    print("GENERATING KAGGLE SUBMISSION FROM MEMORY (TRAIN/TUNE MODE)")
    print("=" * 80)

    # Используем наш DRY-помощник для подготовки данных
    test_data, test_processed = _load_and_preprocess_test()

    print("Making predictions...")
    for model, out_file in [
        (trained_ensemble, config.output.ensemble_submission_file),
        (best_single_model, config.output.best_model_submission_file)
    ]:
        predictions = model.predict(test_processed)
        # Используем DRY-помощник для сохранения на диск
        save_submission_file(predictions, test_data["PassengerId"], out_file)



def predict_inference():
    print("\n" + "=" * 80)
    print("RUNNING INFERENCE (PREDICT MODE)")
    print("=" * 80)
    
    test_data, test_processed = _load_and_preprocess_test()
    ensemble_type = config.model.ensemble 
    
    # СЦЕНАРИЙ 1: Стэкинг (Требует загрузки заранее обученного мета-алгоритма)
    if "stacking" in ensemble_type:
        print("На текущий момент stacking доступен в режиме TRAIN. Сначала запустите TRAIN.")
        return
    
    loaded_models = get_loaded_models(config.model.ensemble_models)
    if not loaded_models:
        print(" Ошибка: Не удалось загрузить ни одной модели!")
        return
    
    if ensemble_type == "averaging":
        print(f"Выбран {ensemble_type}. Cборка вероятностей из базовых моделей...")

        # Собираем вероятности класс '1' от всех загруженных моделей
        probabilities = [model.predict_proba(test_processed)[:, 1] for model in loaded_models]
        
        # Усредняем и переводим в бинарные классы
        mean_probs = np.mean(probabilities, axis=0)
        test_predictions = (mean_probs >= 0.5).astype(int)
    elif ensemble_type == "voting":
        from scipy import stats
        # Собираем жесткие классы (0 или 1) от всех загруженных моделей
        class_predictions = [model.predict(test_processed) for model in loaded_models]
        
        # Находим самое частое значение (моду) для каждого пассажира
        test_predictions, _ = stats.mode(class_predictions, axis=0, keepdims=False)
    else:
        raise ValueError(f"Не поддерживаемый тип ансамбля: {ensemble_type}")
    # Cохранение файла (работает для обоих сценариев)
    save_submission_file(
        predictions=test_predictions, 
        passenger_ids=test_data["PassengerId"], 
        output_path=config.output.ensemble_submission_file
    )

def main() -> None:
    mode = config.general.mode
    params_path = config.output.params_file
    
    print("=" * 80)
    print(f"СТАРТ ПРОГРАММЫ В РЕЖИМЕ: {mode}")
    print("=" * 80)

    # Объединенный конвейер для поиска параметров и быстрого обучения
    if mode in ["TRAIN", "TUNE"]:
        saved_data = None
        
        if mode == "TRAIN":
            saved_data = load_json_file(params_path)
            if not saved_data:
                print(f"⚠️ Файл {params_path} не найден! Принудительно запускаем поиск параметров (TUNE).")
        
        # Запускаем конвейер обучения (все вычисления в памяти)
        best_single_model, trained_ensemble, exp_results = build_and_train_ensemble(saved_data=saved_data)
        
        # Если это был TUNE, сохраняем новые лучшие параметры на диск
        if saved_data is None:
            save_json_file(params_path, exp_results)
            print(f"\n[INFO] Все новые гиперпараметры сохранены в файл: {params_path}")
            
        # Генерируем предсказания из моделей в памяти
        generate_submission(best_single_model, trained_ensemble)

    # Чистый инференс из сохраненных на диск файлов .joblib
    elif mode == "PREDICT":
        predict_inference()


if __name__ == "__main__":
    main()