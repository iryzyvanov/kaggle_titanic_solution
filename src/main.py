from train import build_and_train_ensemble

import pandas as pd

from config import config
from preprocessing import preprocessing_data
from utils import reduce_mem_usage

def generate_submission(model):
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
    # Так как мы убрали подбор threshold из Optuna, .predict() использует стандартные 0.5
    test_predictions = model.predict(test_processed) 

    submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": test_predictions,
    })
    
    submission.to_csv(config.output.submission_file, index=False)
    print(f"✅ Submission successfully saved to: {config.output.submission_file}")


def main():
    # 1. Запускаем весь цикл обучения и получаем готовую модель
    trained_ensemble = build_and_train_ensemble()
    
    # 2. Передаем модель в функцию предсказания (без сохранения на диск)
    generate_submission(trained_ensemble)

if __name__ == "__main__":
    main()