import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np

from validator import evaluate_model, default_model_factory
from sandbox import validate_submission


def parse_task_md(task_md_path: Optional[Path], task_dir: Optional[Path] = None) -> Dict[str, Optional[str]]:
    task_type = None
    target = None
    metric = None

    if task_md_path and task_md_path.exists() and task_md_path.is_file():
        text = task_md_path.read_text(encoding="utf-8", errors="ignore")

        if re.search(r"классификац|classification", text, re.IGNORECASE):
            task_type = "classification"
        elif re.search(r"регресси|regression", text, re.IGNORECASE):
            task_type = "regression"

        target_patterns = [
            r"```(?:csv)?\s+id\s*,\s*([a-zA-Z0-9_]+)\s*```",
            r"(?:таргет|целевая\s+переменная|target)\s*[—:-]?\s*[`'\"]?([a-zA-Z0-9_]+)[`'\"]?",
            r"предсказать\s+(?:класс\s+)?[`'\"]?([a-zA-Z0-9_]+)[`'\"]?",
        ]
        for pattern in target_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                target = match.group(1).strip()
                break

        metric_patterns = [
            r"(?:основная\s+метрика|метрика\s+оценки|метрика|основной\s+метрикой\s+выбран[а]?)\s*[—:-]?\s*\**\s*(Macro\s*F1|Weighted\s*F1|Micro\s*F1|RMSE|RMSLE|MAE|MAPE|ROC[\s\-_]*AUC|LogLoss|Accuracy|F1|R2)\b",
            r"\b(Macro\s*F1|Weighted\s*F1|Micro\s*F1|RMSE|RMSLE|MAE|MAPE|ROC[\s\-_]*AUC|LogLoss|Accuracy|R2)\b",
        ]
        for pattern in metric_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                raw_m = re.sub(r"\s+", " ", match.group(1).strip()).lower()
                if "macro" in raw_m and "f1" in raw_m:
                    metric = "macro_f1"
                elif "weighted" in raw_m and "f1" in raw_m:
                    metric = "weighted_f1"
                elif "micro" in raw_m and "f1" in raw_m:
                    metric = "micro_f1"
                elif "roc" in raw_m and "auc" in raw_m:
                    metric = "roc_auc"
                elif "rmse" in raw_m:
                    metric = "rmse"
                elif "rmsle" in raw_m:
                    metric = "rmsle"
                elif "mae" in raw_m:
                    metric = "mae"
                elif "mape" in raw_m:
                    metric = "mape"
                elif "log" in raw_m:
                    metric = "logloss"
                elif "acc" in raw_m:
                    metric = "accuracy"
                elif "r2" in raw_m:
                    metric = "r2"
                else:
                    metric = raw_m.replace(" ", "_")
                break

    search_dir = task_dir or (task_md_path.parent if task_md_path else Path.cwd())
    dict_files = list(search_dir.rglob("data_dictionary.csv"))
    if dict_files and not target:
        try:
            dd = pd.read_csv(dict_files[0])
            if "role" in dd.columns and "feature" in dd.columns:
                target_rows = dd[dd["role"].astype(str).str.lower().str.contains("target|таргет|целев")]
                if not target_rows.empty:
                    target = str(target_rows.iloc[0]["feature"]).strip()
        except Exception:
            pass

    return {
        "task_type": task_type,
        "target": target,
        "metric": metric,
    }


def find_data_files(task_dir: Path) -> Tuple[Path, Path, Path]:
    train_files = sorted(task_dir.rglob("train.csv"), key=lambda p: len(p.parts))
    test_files = sorted(task_dir.rglob("test.csv"), key=lambda p: len(p.parts))
    sample_files = sorted(task_dir.rglob("sample_submission.csv"), key=lambda p: len(p.parts))

    if not train_files or not test_files or not sample_files:
        raise FileNotFoundError(f"Missing required CSV files in {task_dir}.")

    return train_files[0], test_files[0], sample_files[0]


def atomic_save_submission(
    submission_df: pd.DataFrame,
    sub_path: Path,
    sample_path: Path,
) -> bool:
    tmp_path = sub_path.with_suffix(".csv.tmp")
    try:
        submission_df.to_csv(tmp_path, index=False)
        is_valid, _ = validate_submission(tmp_path, sample_path)
        if not is_valid:
            if tmp_path.exists():
                tmp_path.unlink()
            return False

        os.replace(tmp_path, sub_path)

        cwd_sub = Path.cwd() / "submission.csv"
        if cwd_sub.resolve() != sub_path.resolve():
            try:
                cwd_tmp = cwd_sub.with_suffix(".csv.tmp")
                submission_df.to_csv(cwd_tmp, index=False)
                os.replace(cwd_tmp, cwd_sub)
            except Exception:
                pass
        return True
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        return False


def run_fast_baseline(task_dir: Path, task_file: Optional[Path] = None) -> Tuple[Path, float, str, str]:
    train_path, test_path, sample_path = find_data_files(task_dir)
    sub_path = task_dir / "submission.csv"

    if task_file is None or not task_file.exists() or task_file.is_dir():
        md_files = [p for p in task_dir.glob("*.md") if p.name.lower() != "readme.md"]
        if not md_files:
            md_files = list(task_dir.glob("*.md"))
        task_file = md_files[0] if md_files else task_dir / "task.md"

    task_meta = parse_task_md(task_file, task_dir=task_dir)

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample_sub = pd.read_csv(sample_path)

    id_col = sample_sub.columns[0]
    if task_meta["target"] and task_meta["target"] in sample_sub.columns:
        target_col = task_meta["target"]
    elif len(sample_sub.columns) > 1:
        target_col = sample_sub.columns[1]
    else:
        diff_cols = [c for c in train.columns if c not in test.columns and c != id_col]
        target_col = diff_cols[0] if diff_cols else train.columns[-1]

    if task_meta["task_type"] in ("classification", "regression"):
        task_type = task_meta["task_type"]
    else:
        is_cls = (
            train[target_col].dtype == "object"
            or str(train[target_col].dtype) in ("string", "category", "bool")
            or train[target_col].nunique() <= 30
        )
        task_type = "classification" if is_cls else "regression"

    metric = task_meta["metric"] or ("rmse" if task_type == "regression" else "macro_f1")

    y_train = train[target_col].copy()
    X_train = train.drop(columns=[target_col, id_col], errors="ignore")
    X_test = test.drop(columns=[target_col, id_col], errors="ignore")

    num_classes = int(y_train.nunique()) if task_type == "classification" else 1

    def model_factory():
        return default_model_factory(task_type=task_type, num_classes=num_classes)

    base_score, _, _, extra = evaluate_model(
        model_factory,
        X_train,
        y_train,
        task_type=task_type,
        metric=metric,
        X_test=X_test,
        return_extra=True,
    )

    test_preds = extra["test_preds"]
    if task_type == "regression":
        y_min, y_max = float(y_train.min()), float(y_train.max())
        margin = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
        test_preds = np.clip(test_preds, y_min - margin, y_max + margin)

    submission = pd.DataFrame({id_col: test[id_col], target_col: test_preds})
    submission = sample_sub[[id_col]].merge(submission, on=id_col, how="left")

    if submission[target_col].isna().any():
        fill_val = y_train.mode().iloc[0] if task_type == "classification" else y_train.median()
        submission[target_col] = submission[target_col].fillna(fill_val)

    atomic_save_submission(submission, sub_path, sample_path)
    return sub_path, base_score, target_col, task_type