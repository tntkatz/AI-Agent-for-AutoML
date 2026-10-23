import gc
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    log_loss,
)
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb


def is_higher_better(metric: str, task_type: str) -> bool:
    m = (metric or "").lower().strip()
    if m in ("rmse", "rmsle", "mae", "mape", "logloss", "log_loss", "mse"):
        return False
    if m in ("macro_f1", "weighted_f1", "micro_f1", "f1", "accuracy", "roc_auc", "r2"):
        return True
    return task_type == "classification"


def compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str,
    metric: Optional[str] = None,
    y_prob: Optional[np.ndarray] = None,
) -> float:
    m = (metric or "").lower().strip()

    if task_type == "regression":
        y_true_f = np.asarray(y_true, dtype=np.float64)
        med = float(np.nanmedian(y_true_f)) if len(y_true_f) > 0 else 0.0
        y_pred_f = np.nan_to_num(
            np.asarray(y_pred, dtype=np.float64),
            nan=med,
            posinf=med,
            neginf=med,
        )
        if m == "mae":
            return float(mean_absolute_error(y_true_f, y_pred_f))
        if m == "mape":
            return float(mean_absolute_percentage_error(y_true_f, y_pred_f))
        if m == "rmsle":
            y_t_clip = np.clip(y_true_f, 0.0, None)
            y_p_clip = np.clip(y_pred_f, 0.0, None)
            return float(root_mean_squared_error(np.log1p(y_t_clip), np.log1p(y_p_clip)))
        if m == "r2":
            return float(r2_score(y_true_f, y_pred_f))
        return float(root_mean_squared_error(y_true_f, y_pred_f))

    elif task_type == "classification":
        if m == "roc_auc" and y_prob is not None:
            try:
                if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                    return float(roc_auc_score(y_true, y_prob[:, 1]))
                return float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
            except Exception:
                pass
        if m in ("logloss", "log_loss") and y_prob is not None:
            try:
                return float(log_loss(y_true, np.clip(y_prob, 1e-15, 1.0 - 1e-15)))
            except Exception:
                pass
        if m == "accuracy":
            return float(accuracy_score(y_true, y_pred))
        if m == "weighted_f1":
            return float(f1_score(y_true, y_pred, average="weighted"))
        if m == "micro_f1":
            return float(f1_score(y_true, y_pred, average="micro"))
        return float(f1_score(y_true, y_pred, average="macro"))

    raise ValueError(f"Unknown task type: {task_type}")


def optimize_class_multipliers(
    oof_probs: np.ndarray,
    y_true: np.ndarray,
    metric: str = "macro_f1",
    max_classes: int = 25,
) -> np.ndarray:
    n_samples, n_classes = oof_probs.shape
    m = (metric or "macro_f1").lower().strip()
    if m not in ("macro_f1", "f1", "weighted_f1") or n_classes > max_classes or n_classes < 2:
        return np.ones(n_classes, dtype=np.float64)

    avg_mode = "weighted" if m == "weighted_f1" else "macro"
    base_preds = np.argmax(oof_probs, axis=1)
    base_score = f1_score(y_true, base_preds, average=avg_mode)

    weights = np.ones(n_classes, dtype=np.float64)
    grid = np.linspace(0.35, 2.5, 35)
    for _ in range(2):
        improved = False
        for c in range(n_classes):
            best_w = weights[c]
            best_s = base_score
            scaled = oof_probs * weights[None, :]
            for cand in grid:
                scaled[:, c] = oof_probs[:, c] * cand
                preds = np.argmax(scaled, axis=1)
                s = f1_score(y_true, preds, average=avg_mode) - 0.0002 * abs(cand - 1.0)
                if s > best_s + 1e-5:
                    best_s = s
                    best_w = cand
                    improved = True
            weights[c] = best_w
            base_score = best_s
        if not improved:
            break

    shrunk_weights = 0.75 * weights + 0.25 * np.ones(n_classes, dtype=np.float64)
    final_preds = np.argmax(oof_probs * shrunk_weights[None, :], axis=1)
    if f1_score(y_true, final_preds, average=avg_mode) > f1_score(y_true, base_preds, average=avg_mode) + 1e-4:
        return shrunk_weights
    return np.ones(n_classes, dtype=np.float64)


def sanitize_features(
    X_train: pd.DataFrame,
    X_test: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    X_tr = X_train.copy()
    X_te = X_test.copy() if X_test is not None else None

    if X_te is not None:
        common_cols = [c for c in X_tr.columns if c in X_te.columns]
        X_tr = X_tr[common_cols].copy()
        X_te = X_te[common_cols].copy()

    clean_cols: List[str] = []
    seen: Dict[str, int] = {}
    for col in X_tr.columns:
        c_clean = re.sub(r'[\[\]{}:,"\\<>]+', "_", str(col)).strip()
        if not c_clean:
            c_clean = "feat"
        if c_clean in seen:
            seen[c_clean] += 1
            c_clean = f"{c_clean}_{seen[c_clean]}"
        else:
            seen[c_clean] = 0
        clean_cols.append(c_clean)

    X_tr.columns = clean_cols
    if X_te is not None:
        X_te.columns = clean_cols

    for col in X_tr.columns:
        dtype_name = str(X_tr[col].dtype)
        if dtype_name in ("object", "string", "category", "bool"):
            tr_s = X_tr[col].astype(str).fillna("__missing__")
            if X_te is not None:
                te_s = X_te[col].astype(str).fillna("__missing__")
                combined = pd.concat([tr_s, te_s], axis=0)
                _, uniques = pd.factorize(combined, sort=True)
                mapping = {val: idx for idx, val in enumerate(uniques)}
                X_tr[col] = tr_s.map(mapping).fillna(-1).astype(np.int32)
                X_te[col] = te_s.map(mapping).fillna(-1).astype(np.int32)
            else:
                codes, _ = pd.factorize(tr_s, sort=True)
                X_tr[col] = codes.astype(np.int32)
        else:
            X_tr[col] = pd.to_numeric(X_tr[col], errors="coerce")
            X_tr[col] = X_tr[col].replace([np.inf, -np.inf], np.nan).astype(np.float32)
            if X_te is not None:
                X_te[col] = pd.to_numeric(X_te[col], errors="coerce")
                X_te[col] = X_te[col].replace([np.inf, -np.inf], np.nan).astype(np.float32)

    return X_tr, X_te


def default_model_factory(
    task_type: str,
    num_classes: int = 2,
    n_estimators: int = 350,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    min_child_samples: int = 20,
    colsample_bytree: float = 0.8,
    subsample: float = 0.85,
    random_state: int = 42,
    class_weight: Optional[str] = None,
):
    if task_type == "classification":
        obj = "multiclass" if num_classes > 2 else "binary"
        extra = {"num_class": num_classes} if num_classes > 2 else {}
        return lgb.LGBMClassifier(
            objective=obj,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            subsample_for_bin=50000,
            colsample_bytree=colsample_bytree,
            subsample=subsample,
            subsample_freq=1,
            class_weight=class_weight,
            n_jobs=8,
            random_state=random_state,
            verbose=-1,
            **extra,
        )
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        subsample_for_bin=50000,
        colsample_bytree=colsample_bytree,
        subsample=subsample,
        subsample_freq=1,
        n_jobs=8,
        random_state=random_state,
        verbose=-1,
    )


def evaluate_model(
    model_factory: Callable,
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    n_splits: int = 5,
    random_state: int = 42,
    metric: Optional[str] = None,
    X_test: Optional[pd.DataFrame] = None,
    return_extra: bool = False,
):
    X_clean, X_test_clean = sanitize_features(X, X_test)
    X_clean = X_clean.reset_index(drop=True)
    y_clean = y.reset_index(drop=True)

    if task_type == "classification":
        target_encoder = None
        if (
            y_clean.dtype == "object"
            or str(y_clean.dtype) in ("string", "category", "bool")
            or isinstance(y_clean.iloc[0], str)
        ):
            target_encoder = LabelEncoder()
            y_encoded = target_encoder.fit_transform(y_clean.fillna("__missing__").astype(str))
        else:
            target_encoder = LabelEncoder()
            y_encoded = target_encoder.fit_transform(y_clean.values)

        num_classes = len(np.unique(y_encoded))
        min_class_count = int(pd.Series(y_encoded).value_counts().min())
        actual_splits = max(2, min(n_splits, min_class_count))
        if min_class_count >= actual_splits:
            splitter = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=random_state)
        else:
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        oof_probs = np.zeros((len(y_clean), num_classes), dtype=np.float64)
        test_probs = (
            np.zeros((len(X_test_clean), num_classes), dtype=np.float64)
            if X_test_clean is not None
            else None
        )
        oof_preds = None
        test_preds = None
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        target_encoder = None
        y_encoded = pd.to_numeric(y_clean, errors="coerce").fillna(y_clean.median()).values.astype(np.float64)
        oof_preds = np.zeros(len(y_clean), dtype=np.float64)
        test_preds = (
            np.zeros(len(X_test_clean), dtype=np.float64)
            if X_test_clean is not None
            else None
        )
        oof_probs = None
        test_probs = None

    fold_scores: List[float] = []
    feature_importances = np.zeros(X_clean.shape[1], dtype=np.float64)
    n_actual_splits = splitter.get_n_splits()

    for fold, (train_idx, val_idx) in enumerate(splitter.split(X_clean, y_encoded)):
        X_tr, y_tr = X_clean.iloc[train_idx], y_encoded[train_idx]
        X_va, y_va = X_clean.iloc[val_idx], y_encoded[val_idx]

        model = model_factory()
        model_cls_name = type(model).__name__.lower()

        try:
            if "lgbm" in model_cls_name:
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_va, y_va)],
                    callbacks=[lgb.early_stopping(40, verbose=False)],
                )
            elif "catboost" in model_cls_name:
                X_tr_cb = X_tr.fillna(-999.0)
                X_va_cb = X_va.fillna(-999.0)
                model.fit(X_tr_cb, y_tr, eval_set=(X_va_cb, y_va), early_stopping_rounds=40, verbose=False)
            elif "xgb" in model_cls_name:
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            else:
                model.fit(X_tr, y_tr)
        except Exception:
            model.fit(X_tr, y_tr)

        try:
            if hasattr(model, "booster_"):
                imp = model.booster_.feature_importance(importance_type="gain")
                feature_importances += np.asarray(imp, dtype=np.float64) / n_actual_splits
            elif hasattr(model, "feature_importances_"):
                imp = model.feature_importances_
                feature_importances += np.asarray(imp, dtype=np.float64) / n_actual_splits
        except Exception:
            pass

        X_va_eval = X_va.fillna(-999.0) if "catboost" in model_cls_name else X_va
        X_te_eval = (
            (X_test_clean.fillna(-999.0) if "catboost" in model_cls_name else X_test_clean)
            if X_test_clean is not None
            else None
        )

        if task_type == "classification":
            if hasattr(model, "predict_proba"):
                va_prob = np.asarray(model.predict_proba(X_va_eval), dtype=np.float64)
                if va_prob.ndim == 1:
                    va_prob = np.column_stack([1.0 - va_prob, va_prob])
                oof_probs[val_idx] = va_prob
                va_pred = np.argmax(va_prob, axis=1)
                if X_te_eval is not None and test_probs is not None:
                    te_prob = np.asarray(model.predict_proba(X_te_eval), dtype=np.float64)
                    if te_prob.ndim == 1:
                        te_prob = np.column_stack([1.0 - te_prob, te_prob])
                    test_probs += te_prob / n_actual_splits
            else:
                va_pred = np.asarray(model.predict(X_va_eval)).ravel().astype(np.int64)
                va_prob = None
                oof_probs[val_idx, va_pred] = 1.0
            fold_score = compute_metric(y_va, va_pred, task_type, metric=metric, y_prob=va_prob)
            fold_scores.append(fold_score)
        else:
            va_pred = np.asarray(model.predict(X_va_eval), dtype=np.float64).ravel()
            oof_preds[val_idx] = va_pred
            fold_score = compute_metric(y_va, va_pred, task_type, metric=metric)
            fold_scores.append(fold_score)
            if X_te_eval is not None and test_preds is not None:
                test_preds += np.asarray(model.predict(X_te_eval), dtype=np.float64).ravel() / n_actual_splits

    gc.collect()

    if task_type == "classification":
        oof_pred_idx = np.argmax(oof_probs, axis=1)
        final_score = compute_metric(y_encoded, oof_pred_idx, task_type, metric=metric, y_prob=oof_probs)
        if target_encoder is not None:
            oof_preds_out = target_encoder.inverse_transform(oof_pred_idx)
            test_preds_out = (
                target_encoder.inverse_transform(np.argmax(test_probs, axis=1))
                if test_probs is not None
                else None
            )
        else:
            oof_preds_out = oof_pred_idx
            test_preds_out = np.argmax(test_probs, axis=1) if test_probs is not None else None
    else:
        final_score = compute_metric(y_encoded, oof_preds, task_type, metric=metric)
        oof_preds_out = oof_preds
        test_preds_out = test_preds

    if return_extra:
        imp_series = pd.Series(feature_importances, index=X_clean.columns).sort_values(ascending=False)
        extra: Dict[str, Any] = {
            "test_preds": test_preds_out,
            "test_probs": test_probs,
            "oof_probs": oof_probs,
            "oof_raw": oof_preds if task_type == "regression" else oof_probs,
            "y_encoded": y_encoded,
            "target_encoder": target_encoder,
            "feature_importances": imp_series,
        }
        return final_score, oof_preds_out, fold_scores, extra

    return final_score, oof_preds_out, fold_scores


if __name__ == "__main__":
    print("1. Regression testing")
    reg_train = Path("data/regression/train.csv")
    if reg_train.exists():
        df_reg = pd.read_csv(reg_train)
        y_reg = df_reg["site_eui"]
        X_reg = df_reg.drop(columns=["site_eui", "id"], errors="ignore")

        def make_lgb_reg():
            return default_model_factory("regression")

        score_reg, _, folds_reg = evaluate_model(make_lgb_reg, X_reg, y_reg, task_type="regression")
        print(f"RMSE: {[round(s, 3) for s in folds_reg]}")
        print(f"OOF RMSE: {score_reg:.4f}")

    print("\n2. Classification testing")
    cls_train = Path("data/classification/train.csv")
    if cls_train.exists():
        df_cls = pd.read_csv(cls_train)
        y_cls = df_cls["attack_cat"]
        X_cls = df_cls.drop(columns=["attack_cat", "id"], errors="ignore")

        def make_lgb_cls():
            return default_model_factory("classification", num_classes=y_cls.nunique())

        score_cls, _, folds_cls = evaluate_model(make_lgb_cls, X_cls, y_cls, task_type="classification")
        print(f"Macro F1: {[round(s, 3) for s in folds_cls]}")
        print(f"OOF Macro F1: {score_cls:.4f}")