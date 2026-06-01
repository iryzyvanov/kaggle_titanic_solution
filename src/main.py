from train import build_and_train_ensemble
import joblib
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
    mode = config.general.mode
    print("=" * 80)
    print(f"СТАРТ ПРОГРАММЫ В РЕЖИМЕ: {mode}")
    print("=" * 80)

    if mode in ["TUNE", "TRAIN"]:
        # TUNE: запускаем Optuna (долго)
        # TRAIN: загружаем параметры из JSON и быстро обучаем (средне)
        best_single_model, trained_ensemble = build_and_train_ensemble()
        generate_submission(best_single_model, trained_ensemble)

    elif mode == "PREDICT":
            print("\nЗагрузка данных для предсказания...")
            test_data = pd.read_csv(config.data.test_path)
            test_processed = reduce_mem_usage(preprocessing_data(test_data))

            print("\nЗагрузка обученных моделей с диска...")
            
            # 1. Если в конфиге Stacking - берем готовый собранный ансамбль
            if "stacking" in config.model.ensemble:
                print("Используем готовый обученный StackingClassifier...")
                trained_ensemble = joblib.load(config.output.ensemble_joblib)
                test_predictions = trained_ensemble.predict(test_processed)
                
            # 2. Если Voting/Averaging - собираем на лету из сохраненных моделей!
            else:
                print("Собираем ансамбль (Voting) на лету из одиночных моделей...")
                # Допустим, мы хотим усреднить Топ-3 (можно вынести эти имена в config)
                models_to_ensemble = ["CatBoost", "Logistic_Regression", "RBF_SVM"]
                
                probabilities = []
                for model_name in models_to_ensemble:
                    model_path = rf"models\{model_name}.joblib"
                    if os.path.exists(model_path):
                        loaded_model = joblib.load(model_path)
                        # Собираем вероятности предсказания класса "1" (Выжил)
                        probabilities.append(loaded_model.predict_proba(test_processed)[:, 1])
                    else:
                        print(f"⚠️ Модель {model_name} не найдена!")
                
                # Усредняем вероятности (Soft Voting)
                mean_probs = np.mean(probabilities, axis=0)
                # Переводим вероятности в жесткие классы по стандартному порогу 0.5
                test_predictions = (mean_probs >= 0.5).astype(int)

            # Сохраняем сабмит
            pd.DataFrame({
                "PassengerId": test_data["PassengerId"],
                "Survived": test_predictions
            }).to_csv(config.output.ensemble_submission_file, index=False)
            
            print(f"Sub. saved to {config.output.ensemble_submission_file}")

if __name__ == "__main__":
    main()