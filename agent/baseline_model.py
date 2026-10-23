import os
import re
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder


def parse_task_md(task_md_path: Path) -> Dict[str, Optional[str]]:
    if not task_md_path.exists():
        return {"task_type": None, "target": None, "metric": None}

    text = task_md_path.read_text(encoding="utf-8")

    task_type = None
    if re.search(r"классификац", text, re.IGNORECASE):
        task_type = "classification"
    elif re.search(r"регресси", text, re.IGNORECASE):
        task_type = "regression"

    target = None
    target_patterns = [
        r"```(?:csv)?\s+id\s*,\s*([a-zA-Z0-9_]+)\s*```",
        r"(?:таргет|целевая\s+переменная)\s*[—:-]?\s*`([a-zA-Z0-9_]+)`",
        r"предсказать\s+(?:класс\s+)?`([a-zA-Z0-9_]+)`",
    ]
    for pattern in target_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            target = match.group(1).strip()
            break

    metric = None
    metric_patterns = [
        r"(?:основная\s+метрика|метрика\s+оценки|основной\s+метрикой\s+выбран[а]?)\s*[—:-]?\s*\**\s*([a-zA-Z0-9_ \-]+?)\s*\**(?:\s+на\b|\s*,|\s*:|\s*\n|\s*$)",
        r"\b(Macro\s*F1|Weighted\s*F1|Micro\s*F1|RMSE|RMSLE|MAE|MAPE|ROC[\s\-_]*AUC|LogLoss|Accuracy|F1|R2)\b",
    ]
    for pattern in metric_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            raw_m = re.sub(r"\s+", " ", match.group(1).strip()).lower()
            if "macro" in raw_m and "f1" in raw_m:
                metric = "macro_f1"
            elif "rmse" in raw_m:
                metric = "rmse"
            elif "acc" in raw_m:
                metric = "accuracy"
            else:
                metric = raw_m.replace(" ", "_")
            break

    return {
        "task_type": task_type,
        "target": target,
        "metric": metric,
    }


def find_data_files(task_dir: Path):
    train_files = list(task_dir.rglob("train.csv"))
    test_files = list(task_dir.rglob("test.csv"))
    sample_files = list(task_dir.rglob("sample_submission.csv"))

    if not train_files or not test_files or not sample_files:
        raise FileNotFoundError(f"Не удалось найти все необходимые CSV в {task_dir}.")

    return train_files[0], test_files[0], sample_files[0]


def run_fast_baseline(task_dir: Path, task_file: Optional[Path] = None) -> Path:
    train_path, test_path, sample_path = find_data_files(task_dir)
    sub_path = task_dir / "submission.csv"

    if task_file is None:
        md_files = list(task_dir.glob("*.md"))
        task_file = md_files[0] if md_files else task_dir / "task.md"

    task_meta = parse_task_md(task_file)

    print(f"Загрузка данных из: {task_dir}")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample_sub = pd.read_csv(sample_path)

    id_col = sample_sub.columns[0]
    target_col = (
        task_meta["target"]
        if task_meta["target"] in sample_sub.columns
        else sample_sub.columns[1]
    )
    print(f"ID: '{id_col}', Target: '{target_col}', Metric: '{task_meta['metric']}'")

    if task_meta["task_type"] in ("classification", "regression"):
        task_type = task_meta["task_type"]
        is_classification = task_type == "classification"
    else:
        is_classification = train[target_col].dtype == "object" or train[target_col].nunique() < 50
        task_type = "classification" if is_classification else "regression"

    print(f"Определен тип задачи: {task_type}")

    y_train = train[target_col].copy()
    X_train = train.drop(columns=[target_col])
    X_test = test.copy()

    if id_col in X_train.columns:
        X_train = X_train.drop(columns=[id_col])
    if id_col in X_test.columns:
        X_test = X_test.drop(columns=[id_col])

    for col in X_train.columns:
        if X_train[col].dtype == "object":
            le = LabelEncoder()
            train_vals = X_train[col].fillna("__missing__").astype(str)
            test_vals = X_test[col].fillna("__missing__").astype(str)
            all_vals = pd.concat([train_vals, test_vals], axis=0)
            le.fit(all_vals)

            X_train[col] = le.transform(train_vals)
            X_test[col] = le.transform(test_vals)

    target_le = None
    if is_classification:
        target_le = LabelEncoder()
        y_train = target_le.fit_transform(y_train)

    print("Обучение модели LightGBM")
    lgb_params = {"n_jobs": 8, "verbose": -1, "random_state": 42}

    if is_classification:
        num_classes = len(np.unique(y_train))
        objective = "multiclass" if num_classes > 2 else "binary"
        model = lgb.LGBMClassifier(objective=objective, n_estimators=100, **lgb_params)
    else:
        model = lgb.LGBMRegressor(objective="regression", n_estimators=100, **lgb_params)

    model.fit(X_train, y_train)

    print("Генерация предсказаний")
    preds = model.predict(X_test)
    if is_classification and target_le is not None:
        preds = target_le.inverse_transform(preds)

    submission = pd.DataFrame({
        id_col: test[id_col],
        target_col: preds
    })
    submission = sample_sub[[id_col]].merge(submission, on=id_col, how="left")

    submission.to_csv(sub_path, index=False)
    print(f"Сабмит успешно создан: {sub_path} ({len(submission)} строк)")
    return sub_path