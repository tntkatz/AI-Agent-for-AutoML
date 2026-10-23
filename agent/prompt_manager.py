import pandas as pd
import os
import re

def extract_python_code(text: str) -> str:
    match = re.search(r'```python\s*(.*?)\s*```', text, re.DOTALL)
    return match.group(1) if match else text

def build_dataset_summary(task_dir: str) -> str:
    dict_path = os.path.join(task_dir, 'data_dictionary.csv')
    train_path = os.path.join(task_dir, 'train.csv')
    
    if not os.path.exists(dict_path) or not os.path.exists(train_path):
        return "Error: Data files not found."

    data_dict = pd.read_csv(dict_path)
    train_sample = pd.read_csv(train_path, nrows=5000) 
    missing_pct = (train_sample.isnull().mean() * 100).round(1).to_dict()
    
    lines = ["EDA Summary:"]
    cols = data_dict.columns
    
    feature_col = 'feature' if 'feature' in cols else cols[0]
    desc_col = 'description_ru' if 'description_ru' in cols else cols[-1]
    type_col = 'dtype' if 'dtype' in cols else (cols[2] if len(cols) > 2 else cols[1])
    
    for _, row in data_dict.iterrows():
        col_name = row.get(feature_col, 'unknown')
        desc = row.get(desc_col, '')
        dtype = row.get(type_col, 'unknown')
        miss = missing_pct.get(col_name, 0.0)
        
        lines.append(f"- {col_name} ({dtype}): {desc} | Missing: {miss}%")
            
    return "\n".join(lines)[:4000]

SYSTEM_PROMPT_KAGGLE_GRANDMASTER = """You are a Kaggle Grandmaster in tabular data. Your goal is to generate optimal Feature Engineering.

Task Type: {task_type}
Target Column: {target_col}
Target Metric: {metric}

EDA Summary:
{eda_summary}

Formulate 2-3 hypotheses (aggregations, differences, rare category groupings).
Write ONLY a Python function named `add_features(df)` that takes a pandas DataFrame `df` and returns the transformed DataFrame.
Do not write the full pipeline or cross-validation code. Your function will be integrated and evaluated automatically.

Return ONLY the Python code inside ```python ... ``` block, without any explanations."""

PROMPT_SELF_CORRECTION = """An error occurred while executing your code.
stderr:
{stderr}

Analyze the traceback, fix the error, and return the updated `add_features(df)` function inside a ```python ... ``` block."""