import os
from datetime import datetime 

import pandas as pd
import numpy as np


import logging
import re
import optuna

from catboost import CatBoostClassifier

from config import config

# 1. Создаем фильтр для форматирования чисел и времени
class OptunaLogFormatter(logging.Filter):
    def __init__(self, digits=4):
        super().__init__()
        self.digits = digits
    def filter(self, record):
        msg = str(record.msg)
        
        # 1. Округляем все длинные числа с плавающей точкой
        msg = re.sub(r'(\d+\.\d{5,})', lambda m: f"{float(m.group(1)):.{self.digits}f}", msg)
        
        # 2. Убираем "finished with value:" -> заменяем на "->" или "="
        msg = msg.replace("finished with value:", "->")
        
        # 3. Убираем "and parameters:" -> заменяем на "|"
        msg = msg.replace("and parameters:", "|")
        
        msg = msg.replace("with value:", "=")
        
        record.msg = msg
        return True

def take_Optuna_with_modify_logs():
    # 2. Настраиваем формат времени (убираем дату, оставляем только ЧЧ:ММ:СС)
    optuna.logging.set_verbosity(optuna.logging.INFO)
    logger = optuna.logging.get_logger("optuna")

    # Меняем формат обработчика (handler)
    if logger.handlers:
        handler = logger.handlers[0]
        # %s в конце — это сообщение, которое уже обработано нашим фильтром
        handler.setFormatter(logging.Formatter('[I %(asctime)s] %(message)s', datefmt='%H:%M:%S'))
        handler.addFilter(OptunaLogFormatter())
    return optuna

def reduce_mem_usage(df, verbose = False):
    """Перебирает все столбцы датафрейма и изменяет тип данных для экономии памяти."""
    start_mem = df.memory_usage().sum() / 1024**2
    
    
    for col in df.columns:
        col_type = df[col].dtype
        
        # Пропускаем текстовые/категориальные колонки
        if col_type != object and col_type.name != 'category' and col_type != 'bool':
            c_min = df[col].min()
            c_max = df[col].max()
            
            # Обработка целочисленных типов
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            # Обработка типов с плавающей точкой (float)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                    
    end_mem = df.memory_usage().sum() / 1024**2
    if (verbose):
        print(f'Исходный размер памяти: {start_mem:.4f} MB')
        print(f'Размер памяти после оптимизации: {end_mem:.4f} MB')
    
    return df

def _get_catboost_task_type() -> str:
    """Определяет, доступна ли видеокарта для обучения CatBoost."""
    if CatBoostClassifier is None:
        return "CPU"
    try:
        from catboost.utils import get_gpu_device_count
        # Если найдена хотя бы одна видеокарта с поддержкой CUDA
        if get_gpu_device_count() > 0:
            return "GPU"
    except Exception:
        # Если библиотека скомпилирована без GPU или возникла ошибка драйверов
        pass
    return "CPU"

def print_results(results_df: pd.DataFrame) -> None:
    """Печатает итоговую таблицу в компактном виде."""
    columns = [
        "model",
        "cv_accuracy_mean",
        "cv_accuracy_std",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
    ]

    print("\n" + "=" * 100)
    print("MODEL COMPARISON")
    print("=" * 100)
    print(results_df[columns].round(4).to_string(index=False))
    print()

def append_results_to_markdown_log(results_df: pd.DataFrame, file_path: str = "log_final_Table.md"):
    """
    Преобразует results_df в Markdown-таблицу и добавляет её в конец указанного файла.
    В заголовок автоматически добавляется текущая дата и время.
    """
    if results_df.empty:
        print("DataFrame с результатами пуст. Запись отменена.")
        return

    # Извлекаем имя лучшей модели (первая строчка в отсортированном df)
    best_model_name = results_df.iloc[0]["model"]
    
    # Получаем текущее время
    current_time = datetime.now().strftime("%H:%M %d-%m-%Y")
    
    # Формируем строки для Markdown
    md_lines = []
    md_lines.append("\n" + "=" * 80)
    md_lines.append(f"### MODEL COMPARISON ({current_time})")  # Метка времени теперь здесь
    md_lines.append("=" * 80 + "\n")

    
    md_lines.append("**Run Configuration:**")
    md_lines.append(f"- **Seed:** `{config.general.seed}`")
    md_lines.append(f"- **CV Folds:** `{config.model.cv_folds}`")
    md_lines.append(f"- **Ensemble Type:** `{config.model.ensemble}`")
    md_lines.append(f"- **Optuna Trials (Default):** `{config.optuna.n_trials_default}`")
    md_lines.append(f"- **Optuna Trials (Complex):** `{config.optuna.n_trials_complex}`\n")

    # Заголовки таблицы
    headers = [
        "Model", "CV Accuracy Mean", "CV Accuracy Std", 
        "Test Accuracy", "Test Precision", "Test Recall", "Test F1", "Submit Accuracy"
    ]
    md_lines.append("| " + " | ".join(headers) + " |")
    md_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |:---: |")
    
    # Наполняем таблицу данными с округлением до 4 знаков
    for _, row in results_df.iterrows():
        model_name = row["model"]
        model_display = f"**{model_name}**" if model_name == best_model_name else model_name
        
        line_parts = [
            model_display,
            f"{row['cv_accuracy_mean']:.4f}",
            f"{row['cv_accuracy_std']:.4f}",
            f"{row['test_accuracy']:.4f}",
            f"{row['test_precision']:.4f}",
            f"{row['test_recall']:.4f}",
            f"{row['test_f1']:.4f}",
            "        "
        ]
        md_lines.append("| " + " | ".join(line_parts) + " |")
        
    # Добавляем блок BEST MODEL
    md_lines.append("\n" + "-" * 40)
    md_lines.append("### BEST MODEL")
    md_lines.append(f"**{best_model_name}**")
    md_lines.append("-" * 40 + "\n")
    
    # Открываем файл в режиме добавления ('a')
    with open(file_path, "a", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
        
    print(f"Результаты успешно добавлены в конец файла: {file_path}")