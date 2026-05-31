import os
import pandas as pd
from datetime import datetime 

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