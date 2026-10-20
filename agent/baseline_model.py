import os
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder


def find_data_files(task_dir: Path):
    """
    Ищет train.csv, test.csv и sample_submission.csv внутри директории задачи
    или ее поддиректорий.
    """
    train_files = list(task_dir.rglob("train.csv"))
    test_files = list(task_dir.rglob("test.csv"))
    sample_files = list(task_dir.rglob("sample_submission.csv"))

    if not train_files or not test_files or not sample_files:
        raise FileNotFoundError(
            f"Не удалось найти все необходимые CSV в {task_dir}. "
            f"Найдено: train={len(train_files)}, test={len(test_files)}, sample={len(sample_files)}"
        )

    return train_files[0], test_files[0], sample_files[0]


def run_fast_baseline(task_dir: Path) -> Path:
    """
    Обучает базовый LightGBM и сохраняет гарантированный submission.csv.
    """
    train_path, test_path, sample_path = find_data_files(task_dir)
    sub_path = task_dir / "submission.csv"

    print(f"[Baseline] Загрузка данных из: {task_dir}")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample_sub = pd.read_csv(sample_path)

    id_col = sample_sub.columns[0]
    target_col = sample_sub.columns[1]
    print(f"[Baseline] ID: '{id_col}', Target: '{target_col}'")

    target_series = train[target_col]
    is_classification = target_series.dtype == "object" or target_series.nunique() < 50
    task_type = "classification" if is_classification else "regression"
    print(f"[Baseline] Определен тип задачи: {task_type}")

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

    print("[Baseline] Обучение модели LightGBM...")
    lgb_params = {
        "n_jobs": 8,
        "verbose": -1,
        "random_state": 42
    }

    if is_classification:
        num_classes = len(np.unique(y_train))
        objective = "multiclass" if num_classes > 2 else "binary"
        model = lgb.LGBMClassifier(objective=objective, n_estimators=100, **lgb_params)
    else:
        model = lgb.LGBMRegressor(objective="regression", n_estimators=100, **lgb_params)

    model.fit(X_train, y_train)

    print("[Baseline] Генерация предсказаний...")
    preds = model.predict(X_test)
    if is_classification and target_le is not None:
        preds = target_le.inverse_transform(preds)

    submission = pd.DataFrame({
        id_col: test[id_col],
        target_col: preds
    })
    submission = sample_sub[[id_col]].merge(submission, on=id_col, how="left")

    submission.to_csv(sub_path, index=False)
    print(f"[Baseline] Сабмит успешно создан: {sub_path} ({len(submission)} строк)")
    return sub_path