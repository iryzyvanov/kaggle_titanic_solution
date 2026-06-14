import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from torch.utils.data import DataLoader, TensorDataset

from config import config

# ==========================================
# 1. Архитектура самой нейросети
# ==========================================
class SimpleTitanicNN(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),  # Немного защищаем от переобучения на малых данных
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # Один выход, так как задача бинарной классификации
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 2. Обёртка для интеграции в Pipeline
# ==========================================
class PyTorchTitanicClassifier(BaseEstimator, ClassifierMixin):
    """
    Scikit-Learn совместимая обёртка для PyTorch модели.
    Позволяет использовать нейросеть внутри Pipeline и Optuna.
    """
    def __init__(self, epochs=100, lr=0.01, batch_size=32):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = config.general.seed
        self.model_ = None
        self.classes_ = np.array([0, 1]) # Обязательно для sklearn

    def fit(self, X, y):
        # 1. Фиксируем seed для воспроизводимости
        torch.manual_seed(self.random_state)

        # 2. Безопасная конвертация данных (pandas/numpy -> тензоры)
        X_array = X.values if hasattr(X, 'values') else X
        y_array = y.values if hasattr(y, 'values') else y
        
        X_tensor = torch.FloatTensor(X_array)
        y_tensor = torch.FloatTensor(y_array).view(-1, 1)

        # 3. Инициализация модели, функции потерь и оптимизатора
        input_dim = X_tensor.shape[1]
        self.model_ = SimpleTitanicNN(input_dim)
        
        # BCEWithLogitsLoss объединяет Sigmoid и BCELoss (более стабильно математически)
        criterion = nn.BCEWithLogitsLoss() 
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)

        # 4. Подготовка DataLoader
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # 5. Цикл обучения
        self.model_.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model_(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict_proba(self, X):
        """Возвращает вероятности [P(Class=0), P(Class=1)]"""
        if self.model_ is None:
            raise ValueError("Модель еще не обучена. Вызовите fit() перед predict_proba().")

        self.model_.eval()
        X_array = X.values if hasattr(X, 'values') else X
        X_tensor = torch.FloatTensor(X_array)
        
        with torch.no_grad():
            logits = self.model_(X_tensor)
            probs_class_1 = torch.sigmoid(logits).numpy()

        # sklearn ожидает вероятности для обоих классов в виде матрицы (N, 2)
        probs_class_0 = 1.0 - probs_class_1
        return np.hstack([probs_class_0, probs_class_1])

    def predict(self, X):
        """Возвращает жесткие метки классов (0 или 1)"""
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)