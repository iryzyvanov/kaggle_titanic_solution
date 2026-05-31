from train import build_and_train_ensemble

import pandas as pd

from config import config
from preprocessing import preprocessing_data
from utils import reduce_mem_usage

def generate_submission(best_single_model, trained_ensemble):
    """Принимает обученную модель и генерирует файл для Kaggle."""
    print("\n" + "=" * 80)
    print("GENERATING KAGGLE SUBMISSION")
    print("=" * 80)

    print(f"Reading test data from {config.data.test_path}...")
    test_data = pd.read_csv(config.data.test_path)
    
    print("Preprocessing test data...")
    test_processed = preprocessing_data(test_data)
    test_processed = reduce_mem_usage(test_processed)

    print("Making predictions...")
    

    for model, out_file in [
        (trained_ensemble, config.output.ensemble_submission_file),
        (best_single_model, config.output.best_model_submission_file)
    ]:
        pd.DataFrame({
            "PassengerId": test_data["PassengerId"],
            "Survived": model.predict(test_processed)
        }).to_csv(out_file, index=False)
    print("Sub. saved.")


def main():
    # 1. Запускаем весь цикл обучения и получаем готовую модель
    best_single_model, trained_ensemble = build_and_train_ensemble()
    
    # 2. Передаем модель в функцию предсказания (без сохранения на диск)
    generate_submission(best_single_model, trained_ensemble)

if __name__ == "__main__":
    main()