import os
import re
import pandas as pd

def get_system_prompt(target_col: str) -> str:
    return f"""Ты — опытный Data Scientist и AI-инженер признаков (Feature Engineer). 
Твоя задача — написать функцию `add_features(df)`, которая принимает pandas DataFrame, генерирует новые признаки и возвращает обновленный DataFrame.

КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО использовать, упоминать, преобразовывать или удалять колонку таргета '{target_col}' внутри функции `add_features`. В тестовой выборке (test.csv) этой колонки не будет, и скрипт упадет с ошибкой KeyError, что приведет к дисквалификации!
Колонка '{target_col}' должна остаться в датафрейме в первозданном виде для последующей валидации.

Верни ТОЛЬКО готовый код функции на Python без лишних рассуждений."""

def get_self_correction_prompt(target_col: str, error_trace: str) -> str:
    return f"""Твой предыдущий код функции `add_features(df)` завершился с ошибкой или показал ухудшение метрики.
Детали ошибки / вывода:
{error_trace}

Исправь код и верни обновленную функцию `add_features(df)`.
НАПОМИНАНИЕ: КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО использовать или как-либо взаимодействовать с колонкой таргета '{target_col}' для генерации признаков!"""

def extract_python_code(text: str) -> str:
    match = re.search(r'```(?:python|py)?\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()

def build_experiment_code(generated_function: str, target_col: str, train_path: str = "train.csv") -> str:
    wrapper_script = f"""
import pandas as pd
import warnings
from validator import evaluate_model

warnings.filterwarnings('ignore')

# === LLM GENERATED CODE ===
{generated_function}
# ==========================

def main():
    # 1. Load the dataset
    df = pd.read_csv('{train_path}')
    
    # 2. Apply feature engineering
    try:
        df = add_features(df)
    except Exception as e:
        print(f"EXPERIMENT_FAILED: Error in add_features - {{str(e)}}")
        return

    # 3. Separate features and target
    if '{target_col}' not in df.columns:
        print(f"EXPERIMENT_FAILED: KeyError - Target column '{target_col}' was dropped or modified by add_features.")
        return
        
    y = df['{target_col}']
    X = df.drop(columns=['{target_col}'])
    
    # 4. Evaluate the model
    try:
        score = evaluate_model(X, y)
        # 5. Output strictly in English for the agent to parse
        print(f"EXPERIMENT_SCORE: {{score}}")
    except Exception as e:
        print(f"EXPERIMENT_FAILED: Error during evaluation - {{str(e)}}")

if __name__ == "__main__":
    main()
"""
    return wrapper_script.strip()

def build_dataset_summary(train_path: str = "train.csv", dict_path: str = "data_dictionary.csv") -> str:
    if os.path.exists(dict_path):
        try:
            with open(dict_path, 'r', encoding='utf-8') as f:
                return f"Data Dictionary (from file):\n{f.read()}"
        except Exception as e:
            print(f"Warning: Could not read {dict_path} ({e}). Falling back to train.csv scanning.")

    try:
        df_sample = pd.read_csv(train_path, nrows=5000)
        summary_lines = [
            "Data Dictionary (Auto-generated fallback from train.csv sample):",
            f"{'Column Name'.ljust(25)} | {'Data Type'.ljust(15)} | {'Missing (%)'}"
        ]
        summary_lines.append("-" * 65)
        
        missing_percentages = (df_sample.isnull().sum() / len(df_sample)) * 100
        
        for col in df_sample.columns:
            dtype_str = str(df_sample[col].dtype)
            missing_val = missing_percentages[col]
            summary_lines.append(f"{col.ljust(25)} | {dtype_str.ljust(15)} | {missing_val:.2f}%")
            
        return "\n".join(summary_lines)
        
    except Exception as e:
        return f"Error: Could not generate dataset summary. Ensure {train_path} exists. Details: {str(e)}"