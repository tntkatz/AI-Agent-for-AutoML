import ast
import os
import re
from typing import List, Optional
import numpy as np
import pandas as pd


def get_system_prompt(target_col: str, task_type: str, metric: str, eda_summary: str) -> str:
    return f"""Ты — Principal Kaggle Grandmaster и AI-инженер признаков.
Твоя задача — написать компактную функцию `add_features(df: pd.DataFrame) -> pd.DataFrame`, которая добавляет ТОЛЬКО 5-7 НОВЫХ высокоинформативных табличных признаков для градиентного бустинга.

Тип задачи: {task_type}
Целевая метрика: {metric}
Целевая переменная (ТАРГЕТ): '{target_col}'

Сводка по данным (EDA Summary):
{eda_summary}

ЖЕСТКИЕ ПРАВИЛА:
1. КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО обращаться к колонке таргета '{target_col}' или `id` внутри `add_features(df)`. В `test.csv` таргета нет!
2. ЗАПРЕЩЕНО менять число строк в `df` (`dropna`, фильтрация) или удалять существующие колонки.
3. ЗАПРЕЩЕНО использовать `.apply(..., axis=1)`, циклы `for` по строкам или медленные строковые манипуляции. Используй только быстрые векторизованные операции pandas/numpy!
4. Все ранее найденные лучшие признаки УЖЕ автоматически добавлены в `df` до вызова твоей функции — НЕ переписывай старые признаки, пиши ТОЛЬКО 5-7 новых!
5. Функция должна быть короткой (не более 25-30 строк кода), начинаться с `df = df.copy()` и ОБЯЗАТЕЛЬНО заканчиваться строкой `return df`.
6. Используй быстрые векторизованные операции pandas/numpy с сильным доменным смыслом:
   - Нелинейные взаимодействия топ-признаков: удельные отношения `df['a'] / (df['b'].replace(0, np.nan) + 1e-5)`, асимметрии `(df['a'] - df['b']) / (df['a'].abs() + df['b'].abs() + 1e-5)`, полиномиальные произведения.
   - Внутригрупповые агрегации по важным категориальным или дискретным колонкам: `grp_m = df.groupby('cat_col')['num_col'].transform('mean')`, `grp_s = df.groupby('cat_col')['num_col'].transform('std')`, Z-score `(df['num_col'] - grp_m) / (grp_s + 1e-5)`.
   - Квантильные/ранговые трансформации внутри групп: `df.groupby('cat_col')['num_col'].rank(pct=True)`.
   - Горизонтальные агрегации по семействам однородных колонок (`mean(axis=1)`, `std(axis=1)`, `max(axis=1) - min(axis=1)`).

НЕ ПИШИ никаких рассуждений или пояснений. Верни ТОЛЬКО короткий блок ```python ... ``` с функцией `def add_features(df):`. /no_think"""


def get_iteration_prompt(
    active_features: List[str],
    best_score: float,
    history_text: str,
    target_col: str,
    feature_importance_feedback: str = "",
) -> str:
    fi_block = f"\nВажность признаков (Gain):\n{feature_importance_feedback}\n" if feature_importance_feedback else ""
    active_str = ", ".join(active_features[-35:]) if active_features else "базовые признаки + seed features"

    return f"""Текущий лучший CV скор: {best_score:.5f}.
Уже автоматически активные сгенерированные признаки в датасете: [{active_str}]
{fi_block}
История экспериментов:
{history_text}

ЗАДАНИЕ:
Напиши короткую функцию `def add_features(df):` (максимум 25-30 строк), которая создает 5-7 НОВЫХ признаков из других семейств (например, Z-score по группам, ранги внутри категорий `groupby(...).rank(pct=True)`, отношения/разности топ-признаков из списка Gain), которых еще нет в списке активных.
Старые признаки переписывать НЕ НУЖНО — твоя функция вызывается поверх них.
Обязательно заверши функцию строкой `return df`. Не трогай '{target_col}'! Только код в ```python ... ```. /no_think"""


def get_self_correction_prompt(
    target_col: str,
    broken_code: str,
    error_trace: str,
    best_code: str = "",
) -> str:
    return f"""Предыдущая функция `add_features(df)` не создала рабочих признаков!

Код:
```python
{broken_code[:1200]}
```

Причина ошибки:
{error_trace[-800:]}

Напиши исправленную короткую функцию `def add_features(df):` (15-25 строк), использующую только реально существующие колонки из словаря данных, без обращения к '{target_col}', и обязательно заканчивающуюся на `return df`. Только код в ```python ... ```. /no_think"""


class _FaultTolerantTransformer(ast.NodeTransformer):
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        if node.name != "add_features":
            return node

        new_body: List[ast.stmt] = []
        for stmt in node.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom, ast.Return)):
                new_body.append(stmt)
                continue
            if isinstance(stmt, ast.Expr) and isinstance(getattr(stmt, "value", None), ast.Constant):
                new_body.append(stmt)
                continue
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                tgt = stmt.targets[0]
                if isinstance(tgt, ast.Name) and tgt.id == "df":
                    new_body.append(stmt)
                    continue

            handler = ast.ExceptHandler(
                type=ast.Name(id="Exception", ctx=ast.Load()),
                name="_e",
                body=[
                    ast.Expr(
                        value=ast.Call(
                            func=ast.Name(id="print", ctx=ast.Load()),
                            args=[
                                ast.JoinedStr(
                                    values=[
                                        ast.Constant(value="FEATURE_WARN: "),
                                        ast.FormattedValue(
                                            value=ast.Name(id="_e", ctx=ast.Load()),
                                            conversion=-1,
                                        ),
                                    ]
                                )
                            ],
                            keywords=[],
                        )
                    )
                ],
            )
            try_node = ast.Try(body=[stmt], handlers=[handler], orelse=[], finalbody=[])
            new_body.append(try_node)

        if not new_body or not isinstance(new_body[-1], ast.Return):
            new_body.append(ast.Return(value=ast.Name(id="df", ctx=ast.Load())))
        else:
            ret_stmt = new_body[-1]
            if ret_stmt.value is None:
                ret_stmt.value = ast.Name(id="df", ctx=ast.Load())

        node.body = new_body
        return node


def extract_python_code(text: str) -> str:
    if not text:
        return ""

    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    if "<think>" in cleaned_text and "def add_features" in cleaned_text:
        cleaned_text = cleaned_text[cleaned_text.find("def add_features") :]

    blocks = re.findall(r"```(?:python|py)?\s*(.*?)(?:```|$)", cleaned_text, re.DOTALL | re.IGNORECASE)
    candidates = [b.strip() for b in reversed(blocks) if "def add_features" in b]

    if not candidates:
        idx = cleaned_text.find("def add_features")
        if idx != -1:
            candidates = [cleaned_text[idx:].strip()]
        elif blocks:
            candidates = [blocks[-1].strip()]
        else:
            candidates = [cleaned_text.strip()]

    raw_cand = candidates[0]
    raw_cand = re.sub(r"```+.*$", "", raw_cand, flags=re.DOTALL).strip()

    if "def add_features" not in raw_cand and ("df[" in raw_cand or "df." in raw_cand):
        indented = "\n".join("    " + ln for ln in raw_cand.splitlines() if ln.strip())
        raw_cand = f"def add_features(df):\n    df = df.copy()\n{indented}\n    return df"

    lines = raw_cand.splitlines()

    parsed_tree = None
    for trim in range(min(len(lines), 25)):
        sub_lines = lines[: len(lines) - trim] if trim > 0 else lines
        attempt = "\n".join(sub_lines).rstrip()
        if not attempt:
            break
        for suffix in ("", "\n    return df"):
            try:
                parsed_tree = ast.parse(attempt + suffix)
                break
            except SyntaxError:
                continue
        if parsed_tree is not None:
            break

    if parsed_tree is None:
        return "def add_features(df):\n    return df"

    try:
        transformer = _FaultTolerantTransformer()
        safe_tree = transformer.visit(parsed_tree)
        ast.fix_missing_locations(safe_tree)
        return ast.unparse(safe_tree)
    except Exception:
        return "def add_features(df):\n    return df"


def compose_feature_pipeline(
    accepted_blocks: List[str],
    candidate_code: Optional[str] = None,
    dropped_cols: Optional[List[str]] = None,
) -> str:
    all_blocks = [b for b in accepted_blocks if b and b.strip()]
    if candidate_code and candidate_code.strip():
        all_blocks.append(candidate_code.strip())

    if not all_blocks:
        return "def add_features(df):\n    return df\n"

    parts = ["import math", "import numpy as np", "import pandas as pd", ""]
    fn_names = []

    for idx, block in enumerate(all_blocks):
        fn_name = f"_feat_step_{idx}"
        renamed = re.sub(r"\bdef\s+add_features\s*\(", f"def {fn_name}(", block, count=1)
        parts.append(renamed)
        parts.append("")
        fn_names.append(fn_name)

    drop_list_repr = repr(dropped_cols or [])
    fn_tuple_str = ", ".join(fn_names) + ("," if len(fn_names) == 1 else "")

    runner = f"""
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n_rows = len(df)
    for _fn in ({fn_tuple_str}):
        try:
            _in = df.copy()
            _out = _fn(_in)
            if isinstance(_out, pd.DataFrame) and len(_out) == n_rows:
                df = _out
            elif len(_in) == n_rows and len(_in.columns) > len(df.columns):
                df = _in
        except Exception as _step_err:
            print(f"PIPELINE_STEP_WARN: {{_step_err}}")
    _drop = {drop_list_repr}
    if _drop:
        df = df.drop(columns=[c for c in _drop if c in df.columns], errors='ignore')
    return df
"""
    parts.append(runner.strip())
    return "\n".join(parts)


def build_experiment_code(
    composed_pipeline_code: str,
    target_col: str,
    task_type: str,
    train_path: str = "train.csv",
    test_path: Optional[str] = None,
    metric: Optional[str] = None,
    prev_col_count: int = 0,
) -> str:
    wrapper_script = f"""
import sys
import warnings
import math
import numpy as np
import pandas as pd
from validator import evaluate_model, default_model_factory

warnings.filterwarnings('ignore')

# === COMPOSED FEATURE PIPELINE ===
{composed_pipeline_code}
# =================================

def main():
    try:
        train_df = pd.read_csv({repr(train_path)})
    except Exception as e:
        print(f"EXPERIMENT_FAILED: Failed to load train.csv - {{str(e)}}")
        sys.exit(1)

    n_train = len(train_df)

    if {repr(target_col)} not in train_df.columns:
        print("EXPERIMENT_FAILED: Target column missing in train.csv")
        sys.exit(1)

    y = train_df[{repr(target_col)}].copy()
    X_raw = train_df.drop(columns=[{repr(target_col)}], errors='ignore')
    base_cols = set(X_raw.columns)

    try:
        X_feat = add_features(X_raw.copy())
    except Exception as e:
        print(f"EXPERIMENT_FAILED: Error in add_features - {{type(e).__name__}}: {{str(e)}}")
        sys.exit(1)

    if not isinstance(X_feat, pd.DataFrame):
        print("EXPERIMENT_FAILED: add_features must return a pandas DataFrame.")
        sys.exit(1)

    if len(X_feat) != n_train:
        print(f"EXPERIMENT_FAILED: Row count changed ({{len(X_feat)}} != {{n_train}}). Never filter rows!")
        sys.exit(1)

    new_cols_all = [c for c in X_feat.columns if c not in base_cols]
    if {repr(prev_col_count)} > 0 and len(X_feat.columns) <= {repr(prev_col_count)}:
        print("EXPERIMENT_FAILED: Candidate function did not create any valid new columns (check column names and types).")
        sys.exit(1)

    X = X_feat.drop(columns=[{repr(target_col)}, 'id', 'ID', 'Id'], errors='ignore')
    num_classes = int(y.nunique()) if {repr(task_type)} == 'classification' else 1

    def model_factory():
        return default_model_factory(task_type={repr(task_type)}, num_classes=num_classes)

    try:
        score, _, _, extra = evaluate_model(
            model_factory,
            X,
            y,
            task_type={repr(task_type)},
            metric={repr(metric)},
            return_extra=True,
        )
        imp = extra.get("feature_importances")
        if imp is not None and len(imp) > 0:
            top_str = ", ".join([f"{{k}}({{v:.0f}})" for k, v in imp.head(15).items()])
            new_cols = [c for c in imp.index if c not in base_cols]
            zero_new = [c for c in new_cols if imp[c] <= 1e-5]
            print(f"TOP_FEATURES: {{top_str}}")
            if zero_new:
                print(f"ZERO_IMPORTANCE_NEW_FEATURES: {{', '.join(zero_new[:20])}}")
        print(f"TOTAL_COLS: {{len(X_feat.columns)}}")
        print(f"ACTIVE_NEW_COLS: {{', '.join(new_cols_all[-30:])}}")
        print(f"EXPERIMENT_SCORE: {{score:.6f}}")
    except Exception as e:
        print(f"EXPERIMENT_FAILED: Error during evaluation - {{type(e).__name__}}: {{str(e)}}")
        sys.exit(1)

if __name__ == "__main__":
    main()
"""
    return wrapper_script.strip()


def build_dataset_summary(
    train_path: str = "train.csv",
    dict_path: str = "data_dictionary.csv",
    target_col: Optional[str] = None,
) -> str:
    desc_map = {}
    if os.path.exists(dict_path):
        try:
            dd = pd.read_csv(dict_path)
            for _, row in dd.iterrows():
                feat = str(row.get("feature", "")).strip()
                desc = str(row.get("description_ru", "")).strip()[:55]
                if feat:
                    desc_map[feat] = desc
        except Exception:
            pass

    if not os.path.exists(train_path):
        return "Dataset file not found."

    try:
        df = pd.read_csv(train_path, nrows=10000)
        lines = [f"Строк в сэмпле: {len(df)}, Колонок: {len(df.columns)}"]

        if target_col and target_col in df.columns:
            if df[target_col].dtype == "object" or df[target_col].nunique() <= 25:
                vc = df[target_col].value_counts(normalize=True).head(10)
                dist_str = ", ".join([f"{k}:{v*100:.1f}%" for k, v in vc.items()])
                lines.append(f"Таргет '{target_col}' (классов={df[target_col].nunique()}): {dist_str}")
            else:
                t_s = pd.to_numeric(df[target_col], errors="coerce")
                lines.append(
                    f"Таргет '{target_col}': min={t_s.min():.2f}, med={t_s.median():.2f}, mean={t_s.mean():.2f}, max={t_s.max():.2f}"
                )

        lines.append("Колонки (имя | тип | пропуски | уник | диапазон | описание):")
        for col in df.columns:
            if col == target_col:
                continue
            miss_pct = df[col].isna().mean() * 100
            n_uniq = df[col].nunique(dropna=True)
            dtype_str = str(df[col].dtype)
            d_ru = desc_map.get(col, "")
            if dtype_str in ("object", "string", "category", "bool"):
                top_vals = df[col].dropna().astype(str).unique()[:3].tolist()
                info = f"vals={top_vals}"
            else:
                s_num = pd.to_numeric(df[col], errors="coerce")
                info = f"[{s_num.min():.2g}..{s_num.max():.2g}]"
            lines.append(f"- {col} | {dtype_str} | {miss_pct:.0f}% | u={n_uniq} | {info} | {d_ru}")

        if target_col and target_col in df.columns:
            num_cols = [
                c for c in df.select_dtypes(include=[np.number]).columns
                if c != target_col and c.lower() != "id"
            ]
            if num_cols:
                if df[target_col].dtype == "object" or str(df[target_col].dtype) in ("string", "category") or df[target_col].nunique() <= 25:
                    scores = {}
                    for nc in num_cols:
                        s_v = pd.to_numeric(df[nc], errors="coerce")
                        tot_var = float(s_v.var())
                        if tot_var and tot_var > 1e-12:
                            grp_var = float(s_v.groupby(df[target_col]).transform("mean").var())
                            scores[nc] = float(np.sqrt(max(0.0, grp_var / tot_var)))
                    corrs = pd.Series(scores).sort_values(ascending=False).head(18)
                else:
                    y_num = pd.to_numeric(df[target_col], errors="coerce")
                    p_corr = df[num_cols].apply(lambda x: x.corr(y_num)).abs().fillna(0)
                    s_corr = df[num_cols].apply(lambda x: x.corr(y_num, method="spearman")).abs().fillna(0)
                    corrs = np.maximum(p_corr, s_corr).sort_values(ascending=False).head(18)
                corr_str = ", ".join([f"{k}({v:.2f})" for k, v in corrs.items()])
                lines.append(f"\nТоп-18 признаков по связи с таргетом: {corr_str}")

        return "\n".join(lines)[:4800]
    except Exception as e:
        return f"Warning scanning train.csv: {str(e)}"