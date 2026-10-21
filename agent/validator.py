from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import root_mean_squared_error, f1_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb


def compute_metric(y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> float:
    if task_type == "regression":
        return float(root_mean_squared_error(y_true, y_pred))
    elif task_type == "classification":
        return float(f1_score(y_true, y_pred, average="macro"))
    else:
        raise ValueError(f"Неизвестный тип задачи: {task_type}")


def evaluate_model(
    model_factory,
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    n_splits: int = 5,
    random_state: int = 42
):
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    if task_type == "classification":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        target_encoder = None
        if y.dtype == "object" or isinstance(y.iloc[0], str):
            target_encoder = LabelEncoder()
            y_encoded = target_encoder.fit_transform(y)
        else:
            y_encoded = y.values
        oof_preds = np.empty(len(y), dtype=object if target_encoder else y.dtype)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        target_encoder = None
        y_encoded = y.values.astype(np.float64)
        oof_preds = np.zeros(len(y), dtype=np.float64)

    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y_encoded)):
        X_train, y_train = X.iloc[train_idx], y_encoded[train_idx]
        X_val, y_val = X.iloc[val_idx], y_encoded[val_idx]

        model = model_factory()
        model.fit(X_train, y_train)

        val_preds = model.predict(X_val)
        fold_score = compute_metric(y_val, val_preds, task_type)
        fold_scores.append(fold_score)

        if target_encoder is not None:
            oof_preds[val_idx] = target_encoder.inverse_transform(val_preds)
        else:
            oof_preds[val_idx] = val_preds

    final_score = compute_metric(y.values, oof_preds, task_type)
    return final_score, oof_preds, fold_scores


if __name__ == "__main__":
    print("1. ПРОВЕРКА НА РЕГРЕССИИ")
    reg_train = Path("data/regression/train.csv")
    if reg_train.exists():
        df_reg = pd.read_csv(reg_train)
        y_reg = df_reg["site_eui"]
        X_reg = df_reg.drop(columns=["site_eui", "id"], errors="ignore")

        for col in X_reg.select_dtypes(include=["object"]).columns:
            le = LabelEncoder()
            X_reg[col] = le.fit_transform(X_reg[col].fillna("__missing__").astype(str))

        def make_lgb_reg():
            return lgb.LGBMRegressor(n_estimators=50, n_jobs=8, random_state=42, verbose=-1)

        score_reg, _, folds_reg = evaluate_model(make_lgb_reg, X_reg, y_reg, task_type="regression")
        print(f"Скор по фолдам (RMSE): {[round(s, 3) for s in folds_reg]}")
        print(f"Итоговый OOF RMSE: {score_reg:.4f}")
    else:
        print(f"Файл {reg_train} не найден.")

    print("\n2. ПРОВЕРКА НА КЛАССИФИКАЦИИ")
    cls_train = Path("data/classification/train.csv")
    if cls_train.exists():
        df_cls = pd.read_csv(cls_train)
        y_cls = df_cls["attack_cat"]
        X_cls = df_cls.drop(columns=["attack_cat", "id"], errors="ignore")

        for col in X_cls.select_dtypes(include=["object"]).columns:
            le = LabelEncoder()
            X_cls[col] = le.fit_transform(X_cls[col].fillna("__missing__").astype(str))

        def make_lgb_cls():
            return lgb.LGBMClassifier(n_estimators=50, n_jobs=8, random_state=42, verbose=-1)

        score_cls, _, folds_cls = evaluate_model(make_lgb_cls, X_cls, y_cls, task_type="classification")
        print(f"Скор по фолдам (Macro F1): {[round(s, 3) for s in folds_cls]}")
        print(f"Итоговый OOF Macro F1: {score_cls:.4f}")
    else:
        print(f"Файл {cls_train} не найден.")