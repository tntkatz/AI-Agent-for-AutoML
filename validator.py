import gc
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
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


def _fit_coordinate_ascent_multipliers(
    probs: np.ndarray,
    y_true: np.ndarray,
    avg_mode: str = "macro",
) -> np.ndarray:
    n_classes = probs.shape[1]
    counts = np.bincount(y_true, minlength=n_classes).astype(np.float64)
    priors = counts / max(float(len(y_true)), 1.0)

    best_w = np.ones(n_classes, dtype=np.float64)
    best_score = f1_score(y_true, np.argmax(probs, axis=1), average=avg_mode)

    for gamma in (-0.12, -0.22, -0.32, -0.42, -0.52):
        cand_w = np.power(np.maximum(priors, 1e-5), gamma)
        cand_w = np.clip(cand_w / np.mean(cand_w), 0.15, 6.0)
        sc = f1_score(y_true, np.argmax(probs * cand_w[None, :], axis=1), average=avg_mode)
        if sc > best_score + 1e-5:
            best_score = sc
            best_w = cand_w.copy()

    grid = np.concatenate([
        np.linspace(0.12, 0.95, 18),
        np.array([1.0]),
        np.linspace(1.05, 4.50, 35),
    ])

    class_order = np.argsort(counts)
    for _ in range(3):
        improved = False
        scaled = probs * best_w[None, :]
        for c in class_order:
            orig_col = probs[:, c]
            cur_best_c = best_w[c]
            for cand in grid:
                scaled[:, c] = orig_col * cand
                preds = np.argmax(scaled, axis=1)
                sc = f1_score(y_true, preds, average=avg_mode) - 0.00015 * abs(cand - 1.0)
                if sc > best_score + 1e-5:
                    best_score = sc
                    cur_best_c = cand
                    improved = True
            best_w[c] = cur_best_c
            scaled[:, c] = orig_col * cur_best_c
        if not improved:
            break

    best_w /= np.mean(best_w)
    return best_w


def optimize_class_multipliers(
    oof_probs: np.ndarray,
    y_true: np.ndarray,
    metric: str = "macro_f1",
) -> np.ndarray:
    n_samples, n_classes = oof_probs.shape
    m = (metric or "macro_f1").lower().strip()
    if m not in ("macro_f1", "f1", "weighted_f1") or n_classes < 2 or n_classes > 30:
        return np.ones(n_classes, dtype=np.float64)

    avg_mode = "weighted" if m == "weighted_f1" else "macro"
    base_preds = np.argmax(oof_probs, axis=1)
    base_score = f1_score(y_true, base_preds, average=avg_mode)

    min_cnt = int(pd.Series(y_true).value_counts().min())
    if min_cnt >= 5:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(skf.split(oof_probs, y_true))
    else:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(kf.split(oof_probs))

    cv_oof_preds = np.zeros(n_samples, dtype=np.int64)
    fold_weights = []

    for tr_idx, va_idx in splits:
        w_fold = _fit_coordinate_ascent_multipliers(oof_probs[tr_idx], y_true[tr_idx], avg_mode=avg_mode)
        fold_weights.append(w_fold)
        cv_oof_preds[va_idx] = np.argmax(oof_probs[va_idx] * w_fold[None, :], axis=1)

    cv_score = f1_score(y_true, cv_oof_preds, average=avg_mode)
    if cv_score > base_score + 0.0002:
        mean_w = np.mean(fold_weights, axis=0)
        final_w = 0.90 * mean_w + 0.10 * np.ones(n_classes, dtype=np.float64)
        return final_w

    return np.ones(n_classes, dtype=np.float64)


def compute_balanced_sample_weights(y: np.ndarray, power: float = 0.45) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    n_total = float(len(y))
    n_classes = float(len(classes))
    class_weights = {}
    for c, cnt in zip(classes, counts):
        raw_w = n_total / max(n_classes * float(cnt), 1.0)
        class_weights[c] = float(np.clip(raw_w ** power, 0.4, 20.0))
    sw = np.array([class_weights[val] for val in y], dtype=np.float32)
    sw /= np.mean(sw)
    return sw


def _Identify_te_columns(df: pd.DataFrame, max_cols: int = 10) -> List[str]:
    priority_cols = []
    candidates = []
    n_rows = len(df)
    max_card = max(600, int(n_rows * 0.70))

    for col in df.columns:
        if col.startswith("__te_"):
            continue
        c_low = str(col).lower()
        dtype_str = str(df[col].dtype)
        n_uniq = df[col].nunique(dropna=True)

        if "entity_id" in c_low or "bldg" in c_low or "net_pss" in c_low or c_low.startswith("__comp_"):
            if 2 <= n_uniq <= max_card:
                priority_cols.append(col)
                continue

        if dtype_str in ("object", "string", "category"):
            if 2 <= n_uniq <= max_card:
                candidates.append((col, n_uniq))
        elif 3 <= n_uniq <= 300 and (
            "int" in dtype_str
            or "cat" in c_low
            or "type" in c_low
            or "id" in c_low
            or "state" in c_low
            or "service" in c_low
            or "proto" in c_low
            or "factor" in c_low
        ):
            candidates.append((col, n_uniq))

    candidates.sort(key=lambda x: x[1], reverse=True)
    out = priority_cols + [c[0] for c in candidates if c[0] not in priority_cols]
    return out[:max_cols]


def sanitize_features(
    X_train: pd.DataFrame,
    X_test: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    X_tr = X_train.copy()
    X_te = X_test.copy() if X_test is not None else None

    drop_patterns = re.compile(r"^(id|unnamed:\s*0|index)$", re.IGNORECASE)
    cols_to_drop = [c for c in X_tr.columns if drop_patterns.match(str(c).strip())]
    if cols_to_drop:
        X_tr = X_tr.drop(columns=cols_to_drop, errors="ignore")
        if X_te is not None:
            X_te = X_te.drop(columns=cols_to_drop, errors="ignore")

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
            s_tr = pd.to_numeric(X_tr[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            X_tr[col] = s_tr.clip(-1e15, 1e15).astype(np.float32)
            if X_te is not None:
                s_te = pd.to_numeric(X_te[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
                X_te[col] = s_te.clip(-1e15, 1e15).astype(np.float32)

    return X_tr, X_te


def _apply_fold_target_encoding(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    X_te: Optional[pd.DataFrame],
    te_cols: List[str],
    task_type: str,
    num_classes: int = 1,
    smoothing: float = 10.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    if not te_cols:
        return X_tr, X_va, X_te

    X_tr_out = X_tr.copy()
    X_va_out = X_va.copy()
    X_te_out = X_te.copy() if X_te is not None else None

    inner_kf = KFold(n_splits=4, shuffle=True, random_state=42)
    inner_splits = list(inner_kf.split(X_tr))

    if task_type == "regression":
        global_mean = float(np.mean(y_tr))
        global_std = float(np.std(y_tr)) if len(y_tr) > 1 else 0.0
        y_s = pd.Series(y_tr, index=X_tr.index, dtype=np.float64)

        for idx_c, col in enumerate(te_cols):
            c_low = str(col).lower()
            sm = 3.0 if ("entity_id" in c_low or "bldg" in c_low) else smoothing
            out_col = f"__te_mean_{col}"
            tr_encoded = np.full(len(X_tr), global_mean, dtype=np.float32)

            for in_tr_idx, in_va_idx in inner_splits:
                s_col = X_tr[col].iloc[in_tr_idx]
                s_y = y_s.iloc[in_tr_idx]
                stats = s_y.groupby(s_col).agg(["count", "mean"])
                smooth_vals = (stats["count"] * stats["mean"] + sm * global_mean) / (stats["count"] + sm)
                mapped = X_tr[col].iloc[in_va_idx].map(smooth_vals).fillna(global_mean).values
                tr_encoded[in_va_idx] = mapped.astype(np.float32)

            X_tr_out[out_col] = tr_encoded

            full_stats = y_s.groupby(X_tr[col]).agg(["count", "mean"])
            full_smooth = (full_stats["count"] * full_stats["mean"] + sm * global_mean) / (full_stats["count"] + sm)
            X_va_out[out_col] = X_va[col].map(full_smooth).fillna(global_mean).astype(np.float32).values
            if X_te_out is not None:
                X_te_out[out_col] = X_te[col].map(full_smooth).fillna(global_mean).astype(np.float32).values

            if idx_c < 4:
                std_col = f"__te_std_{col}"
                tr_std = np.full(len(X_tr), global_std, dtype=np.float32)
                for in_tr_idx, in_va_idx in inner_splits:
                    s_col = X_tr[col].iloc[in_tr_idx]
                    s_y = y_s.iloc[in_tr_idx]
                    st_s = s_y.groupby(s_col).std().fillna(global_std)
                    tr_std[in_va_idx] = X_tr[col].iloc[in_va_idx].map(st_s).fillna(global_std).values.astype(np.float32)
                X_tr_out[std_col] = tr_std
                full_std_s = y_s.groupby(X_tr[col]).std().fillna(global_std)
                X_va_out[std_col] = X_va[col].map(full_std_s).fillna(global_std).astype(np.float32).values
                if X_te_out is not None:
                    X_te_out[std_col] = X_te[col].map(full_std_s).fillna(global_std).astype(np.float32).values

    elif task_type == "classification" and 2 <= num_classes <= 15:
        use_cols = te_cols[:5] if num_classes > 2 else te_cols[:7]
        classes_to_encode = [1] if num_classes == 2 else list(range(num_classes))

        for col in use_cols:
            s_full_col = X_tr[col]
            counts_full = s_full_col.value_counts()

            for k in classes_to_encode:
                out_col = f"__te_c{k}_{col}"
                y_bin = (y_tr == k).astype(np.float64)
                prior_k = float(np.mean(y_bin))
                y_bin_s = pd.Series(y_bin, index=X_tr.index)

                tr_encoded = np.full(len(X_tr), prior_k, dtype=np.float32)
                for in_tr_idx, in_va_idx in inner_splits:
                    s_col = X_tr[col].iloc[in_tr_idx]
                    s_y = y_bin_s.iloc[in_tr_idx]
                    stats = s_y.groupby(s_col).agg(["count", "mean"])
                    smooth_p = (stats["count"] * stats["mean"] + smoothing * prior_k) / (stats["count"] + smoothing)
                    mapped = X_tr[col].iloc[in_va_idx].map(smooth_p).fillna(prior_k).values
                    tr_encoded[in_va_idx] = mapped.astype(np.float32)

                X_tr_out[out_col] = tr_encoded

                sums_full = y_bin_s.groupby(s_full_col).sum()
                smooth_full = (counts_full * (sums_full / np.maximum(counts_full, 1)) + smoothing * prior_k) / (counts_full + smoothing)
                X_va_out[out_col] = X_va[col].map(smooth_full).fillna(prior_k).astype(np.float32).values
                if X_te_out is not None:
                    X_te_out[out_col] = X_te[col].map(smooth_full).fillna(prior_k).astype(np.float32).values

    return X_tr_out, X_va_out, X_te_out


def default_model_factory(
    task_type: str,
    num_classes: int = 2,
    n_estimators: Optional[int] = None,
    learning_rate: Optional[float] = None,
    num_leaves: Optional[int] = None,
    min_child_samples: int = 25,
    colsample_bytree: float = 0.75,
    subsample: float = 0.85,
    reg_alpha: float = 0.5,
    reg_lambda: float = 1.5,
    extra_trees: bool = False,
    random_state: int = 42,
    class_weight: Optional[str] = None,
):
    if task_type == "classification":
        obj = "multiclass" if num_classes > 2 else "binary"
        extra = {"num_class": num_classes} if num_classes > 2 else {}
        n_est = n_estimators or (220 if num_classes > 2 else 550)
        lr = learning_rate or (0.080 if num_classes > 2 else 0.045)
        n_leaves = num_leaves or (40 if num_classes > 2 else 63)
        return lgb.LGBMClassifier(
            objective=obj,
            n_estimators=n_est,
            learning_rate=lr,
            num_leaves=n_leaves,
            min_child_samples=min_child_samples,
            subsample_for_bin=50000,
            colsample_bytree=colsample_bytree,
            subsample=subsample,
            subsample_freq=1,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            extra_trees=extra_trees,
            class_weight=class_weight,
            n_jobs=8,
            random_state=random_state,
            verbose=-1,
            **extra,
        )

    n_est = n_estimators or 650
    lr = learning_rate or 0.042
    n_leaves = num_leaves or 63
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=n_est,
        learning_rate=lr,
        num_leaves=n_leaves,
        min_child_samples=min_child_samples,
        subsample_for_bin=50000,
        colsample_bytree=colsample_bytree,
        subsample=subsample,
        subsample_freq=1,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        extra_trees=extra_trees,
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
    use_Wait_sample_weights: Optional[bool] = None,
):
    if use_Wait_sample_weights is None:
        use_Wait_sample_weights = (task_type == "classification")

    te_candidate_cols = _Identify_te_columns(X, max_cols=8)
    X_clean, X_test_clean = sanitize_features(X, X_test)
    X_clean = X_clean.reset_index(drop=True)
    y_clean = y.reset_index(drop=True)

    te_cols = [c for c in te_candidate_cols if c in X_clean.columns]

    if task_type == "classification":
        target_encoder = LabelEncoder()
        if (
            y_clean.dtype == "object"
            or str(y_clean.dtype) in ("string", "category", "bool")
            or isinstance(y_clean.iloc[0], str)
        ):
            y_encoded = target_encoder.fit_transform(y_clean.fillna("__missing__").astype(str))
        else:
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
        num_classes = 1
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
    base_feature_names = list(X_clean.columns)
    feature_importances = np.zeros(len(base_feature_names), dtype=np.float64)
    n_actual_splits = splitter.get_n_splits()

    for fold, (train_idx, val_idx) in enumerate(splitter.split(X_clean, y_encoded)):
        X_tr_raw, y_tr = X_clean.iloc[train_idx], y_encoded[train_idx]
        X_va_raw, y_va = X_clean.iloc[val_idx], y_encoded[val_idx]

        X_tr, X_va, X_te_fold = _apply_fold_target_encoding(
            X_tr_raw,
            y_tr,
            X_va_raw,
            X_test_clean,
            te_cols=te_cols,
            task_type=task_type,
            num_classes=num_classes,
        )

        model = model_factory()
        model_cls_name = type(model).__name__.lower()

        sw_tr = None
        if use_Wait_sample_weights and task_type == "classification":
            sw_tr = compute_balanced_sample_weights(y_tr, power=0.45)

        X_tr_clean = X_tr.replace([np.inf, -np.inf], np.nan).clip(-1e15, 1e15)
        X_va_clean = X_va.replace([np.inf, -np.inf], np.nan).clip(-1e15, 1e15)
        X_te_clean_fold = (
            X_te_fold.replace([np.inf, -np.inf], np.nan).clip(-1e15, 1e15)
            if X_te_fold is not None
            else None
        )

        es_rounds = 25 if (task_type == "classification" and num_classes > 2) else 35

        if "lgbm" in model_cls_name:
            try:
                model.fit(
                    X_tr_clean,
                    y_tr,
                    sample_weight=sw_tr,
                    eval_set=[(X_va_clean, y_va)],
                    callbacks=[lgb.early_stopping(es_rounds, verbose=False)],
                )
            except Exception:
                model.fit(X_tr_clean, y_tr)
            X_va_eval = X_va_clean
            X_te_eval = X_te_clean_fold

        elif "catboost" in model_cls_name:
            X_tr_cb = X_tr_clean.fillna(-999.0).astype(np.float32)
            X_va_cb = X_va_clean.fillna(-999.0).astype(np.float32)
            X_te_cb = (
                X_te_clean_fold.fillna(-999.0).astype(np.float32)
                if X_te_clean_fold is not None
                else None
            )
            try:
                model.fit(
                    X_tr_cb,
                    y_tr,
                    sample_weight=sw_tr,
                    eval_set=(X_va_cb, y_va),
                    early_stopping_rounds=es_rounds,
                    verbose=False,
                )
            except Exception:
                model.fit(X_tr_cb, y_tr, verbose=False)
            X_va_eval = X_va_cb
            X_te_eval = X_te_cb

        elif "xgb" in model_cls_name:
            X_tr_xgb = X_tr_clean.astype(np.float32)
            X_va_xgb = X_va_clean.astype(np.float32)
            X_te_xgb = (
                X_te_clean_fold.astype(np.float32)
                if X_te_clean_fold is not None
                else None
            )
            try:
                model.fit(
                    X_tr_xgb,
                    y_tr,
                    sample_weight=sw_tr,
                    eval_set=[(X_va_xgb, y_va)],
                    verbose=False,
                )
            except Exception:
                try:
                    model.set_params(early_stopping_rounds=None)
                except Exception:
                    pass
                model.fit(X_tr_xgb, y_tr, verbose=False)
            X_va_eval = X_va_xgb
            X_te_eval = X_te_xgb

        else:
            model.fit(X_tr_clean.fillna(0.0), y_tr)
            X_va_eval = X_va_clean.fillna(0.0)
            X_te_eval = X_te_clean_fold.fillna(0.0) if X_te_clean_fold is not None else None

        try:
            if hasattr(model, "booster_"):
                imp = model.booster_.feature_importance(importance_type="gain")
                feature_importances += np.asarray(imp[: len(base_feature_names)], dtype=np.float64) / n_actual_splits
            elif hasattr(model, "feature_importances_"):
                imp = model.feature_importances_
                feature_importances += np.asarray(imp[: len(base_feature_names)], dtype=np.float64) / n_actual_splits
        except Exception:
            pass

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
        mults = optimize_class_multipliers(oof_probs, y_encoded, metric=metric or "macro_f1")
        oof_pred_idx = np.argmax(oof_probs * mults[None, :], axis=1)
        final_score = compute_metric(y_encoded, oof_pred_idx, task_type, metric=metric, y_prob=oof_probs)
        if target_encoder is not None:
            oof_preds_out = target_encoder.inverse_transform(oof_pred_idx)
            test_preds_out = (
                target_encoder.inverse_transform(np.argmax(test_probs * mults[None, :], axis=1))
                if test_probs is not None
                else None
            )
        else:
            oof_preds_out = oof_pred_idx
            test_preds_out = np.argmax(test_probs * mults[None, :], axis=1) if test_probs is not None else None
    else:
        final_score = compute_metric(y_encoded, oof_preds, task_type, metric=metric)
        oof_preds_out = oof_preds
        test_preds_out = test_preds

    if return_extra:
        imp_series = pd.Series(feature_importances, index=base_feature_names).sort_values(ascending=False)
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
