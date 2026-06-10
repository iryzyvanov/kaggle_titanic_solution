import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

def run_baseline():
    print("=" * 60)
    print("🚢 СТАРТ РАСЧЕТА БЕЙЗЛАЙНА (Logistic Regression)")
    print("=" * 60)

    # 1. Загружаем данные
    try:
        train_data = pd.read_csv('data/train.csv')
        test_data = pd.read_csv('data/test.csv')
    except FileNotFoundError:
        print("❌ Файлы данных не найдены! Проверьте путь data/train.csv и data/test.csv")
        return

    # 2. Выбираем только самые базовые фичи
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    target = 'Survived'

    X_train = train_data[features].copy()
    y_train = train_data[target]
    
    X_test = test_data[features].copy()
    test_ids = test_data['PassengerId']

    # =========================================================
    # 3. ПРИМИТИВНАЯ ПРЕДОБРАБОТКА (БЕЗ УТЕЧКИ ДАННЫХ)
    # =========================================================
    
    # Считаем статистики СТРОГО по train
    age_median = X_train['Age'].median()
    fare_median = X_train['Fare'].median()
    embarked_mode = X_train['Embarked'].mode()[0]

    # Заполняем пропуски в Train
    X_train['Age'] = X_train['Age'].fillna(age_median)
    X_train['Fare'] = X_train['Fare'].fillna(fare_median)
    X_train['Embarked'] = X_train['Embarked'].fillna(embarked_mode)

    # Заполняем пропуски в Test ТЕМИ ЖЕ значениями из Train
    X_test['Age'] = X_test['Age'].fillna(age_median)
    X_test['Fare'] = X_test['Fare'].fillna(fare_median)
    X_test['Embarked'] = X_test['Embarked'].fillna(embarked_mode)

    # Кодируем текст (Пол и Порт)
    X_train = pd.get_dummies(X_train, columns=['Sex', 'Embarked'], drop_first=True)
    X_test = pd.get_dummies(X_test, columns=['Sex', 'Embarked'], drop_first=True)

    # Выравниваем колонки (на случай, если в test не попала какая-то категория)
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    # =========================================================
    # 4. КРОСС-ВАЛИДАЦИЯ БЕЙЗЛАЙНА
    # =========================================================
    
    baseline_model = LogisticRegression(max_iter=1000, random_state=42)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(baseline_model, X_train, y_train, cv=cv, scoring='accuracy')

    print(f"Модель: Logistic Regression (Default)")
    print(f"Фичи: Сырые числа + OHE для Sex и Embarked")
    print("-" * 60)
    print(f"CV Accuracy Mean: {np.mean(scores):.4f}")
    print(f"CV Accuracy Std:  {np.std(scores):.4f}")
    print("=" * 60)

    # =========================================================
    # 5. ОБУЧЕНИЕ НА ВСЕХ ДАННЫХ И ГЕНЕРАЦИЯ SUBMISSION
    # =========================================================
    
    print("\nОбучение финальной модели на всем объеме train...")
    baseline_model.fit(X_train, y_train)

    print("Формирование предсказаний...")
    test_preds = baseline_model.predict(X_test)

    # Создаем DataFrame для Kaggle
    submission = pd.DataFrame({
        "PassengerId": test_ids,
        "Survived": test_preds
    })

    # Убеждаемся, что папка subs существует
    os.makedirs("subs", exist_ok=True)
    sub_path = "subs/baseline_submission.csv"
    
    # Сохраняем сабмит
    submission.to_csv(sub_path, index=False)
    print(f"✅ Готово! Файл для отправки сохранен по пути: {sub_path}")

if __name__ == "__main__":
    run_baseline()