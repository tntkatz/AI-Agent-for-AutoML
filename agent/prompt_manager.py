import pandas as pd
import os

def build_dataset_summary(task_dir: str) -> str:
    dict_path = os.path.join(task_dir, 'data_dictionary.csv')
    train_path = os.path.join(task_dir, 'train.csv')
    
    if not os.path.exists(dict_path) or not os.path.exists(train_path):
        return "Ошибка: Файлы данных не найдены."

    # Читаем словарь и сэмпл обучающей выборки для подсчета пропусков
    data_dict = pd.read_csv(dict_path)
    train_sample = pd.read_csv(train_path, nrows=5000) 
    missing_pct = (train_sample.isnull().mean() * 100).round(1).to_dict()
    
    lines = ["Краткая сводка по признакам (EDA Summary):"]
    for _, row in data_dict.iterrows():
        col = row.get('feature', 'unknown')
        desc = row.get('description_ru', '')
        dtype = row.get('dtype', 'unknown')
        miss = missing_pct.get(col, 0.0)
        
        lines.append(f"- {col} ({dtype}): {desc} | Пропуски: {miss}%")
        
    return "\n".join(lines)[:4000] # Защита от переполнения контекста

SYSTEM_PROMPT_KAGGLE_GRANDMASTER = """Ты — Kaggle Grandmaster по табличным данным. Твоя цель: сгенерировать оптимальный пайплайн и Feature Engineering для ML-задачи.
Действуй по шагам:
1. Проанализируй предоставленный EDA Summary.
2. Сформулируй 2-3 гипотезы (агрегации, разницы показателей, группировки редких категорий).
3. Напиши полный, исполняемый Python-код для кросс-валидации модели.
4. В конце выполнения скрипт ОБЯЗАТЕЛЬНО должен выводить метрику в stdout в формате: VALIDATION_SCORE: <число>.

Возвращай ТОЛЬКО Python-код, без пояснительного текста до или после кода."""

PROMPT_SELF_CORRECTION = """При выполнении твоего кода возникла ошибка.
Вывод stderr:
{stderr}

Проанализируй трейсбек, исправь ошибку и верни полный обновленный код. Формат вывода VALIDATION_SCORE: <число> должен быть сохранен."""