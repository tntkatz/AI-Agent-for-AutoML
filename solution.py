import os
import sys
import re
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from baseline_model import run_fast_baseline, parse_task_md, find_data_files
from validator import is_higher_better
from sandbox import validate_submission, run_python
from prompt_manager import (
    build_dataset_summary,
    get_system_prompt,
    get_iteration_prompt,
    get_self_correction_prompt,
    extract_python_code,
    compose_feature_pipeline,
    build_experiment_code,
)
from llm_client import QwenClient

MAX_ITERATIONS = 15


def is_better_score(
    new_score: float,
    current_score: float,
    task_type: str,
    metric: Optional[str] = None,
) -> bool:
    if is_higher_better(metric or "", task_type):
        return new_score > current_score + 1e-6
    return new_score < current_score - 1e-6


def _detect_symmetric_column_pairs(num_cols: List[str]) -> List[Tuple[str, str, str]]:
    col_set = set(num_cols)
    pairs: List[Tuple[str, str, str]] = []
    seen = set()

    for c in num_cols:
        if c.startswith("s") and len(c) >= 4:
            cand = "d" + c[1:]
            if cand in col_set and (c, cand) not in seen:
                seen.add((c, cand))
                pairs.append((c, cand, c[1:]))
        if "src" in c:
            cand = c.replace("src", "dst", 1)
            if cand in col_set and (c, cand) not in seen:
                seen.add((c, cand))
                pairs.append((c, cand, c.replace("src", "sd", 1)))
        if "max" in c:
            cand = c.replace("max", "min", 1)
            if cand in col_set and (c, cand) not in seen:
                seen.add((c, cand))
                pairs.append((c, cand, c.replace("max", "span", 1)))
        if "heating" in c:
            cand = c.replace("heating", "cooling", 1)
            if cand in col_set and (c, cand) not in seen:
                seen.add((c, cand))
                pairs.append((c, cand, c.replace("heating", "hc", 1)))

    return pairs[:14]


def build_deterministic_seed_features(train_path: Path, target_col: str) -> str:
    try:
        df = pd.read_csv(train_path, nrows=15000)
        if target_col not in df.columns:
            return ""

        num_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c != target_col and c.lower() != "id"
        ]
        cat_cols = [
            c for c in df.select_dtypes(include=["object", "string", "category", "bool"]).columns
            if c != target_col and c.lower() != "id"
        ]
        for c in num_cols:
            if 2 <= df[c].nunique(dropna=True) <= 45 and c not in cat_cols:
                cat_cols.append(c)

        top_num = []
        if num_cols:
            try:
                is_cls = (
                    df[target_col].dtype == "object"
                    or str(df[target_col].dtype) in ("string", "category", "bool")
                    or df[target_col].nunique(dropna=True) <= 25
                )
                if is_cls:
                    eta_scores: Dict[str, float] = {}
                    for nc in num_cols:
                        s_v = pd.to_numeric(df[nc], errors="coerce")
                        tot_var = float(s_v.var())
                        if tot_var and tot_var > 1e-12:
                            grp_var = float(s_v.groupby(df[target_col]).transform("mean").var())
                            eta_scores[nc] = float(np.sqrt(max(0.0, grp_var / tot_var)))
                    top_num = pd.Series(eta_scores).sort_values(ascending=False).head(8).index.tolist()
                else:
                    y_num = pd.to_numeric(df[target_col], errors="coerce")
                    p_c = df[num_cols].apply(lambda x: x.corr(y_num)).abs().fillna(0)
                    s_c = df[num_cols].apply(lambda x: x.corr(y_num, method="spearman")).abs().fillna(0)
                    top_num = np.maximum(p_c, s_c).sort_values(ascending=False).head(8).index.tolist()
            except Exception:
                top_num = num_cols[:8]

        sym_pairs = _detect_symmetric_column_pairs(num_cols)

        token_groups: Dict[str, List[str]] = {}
        for c in num_cols:
            tokens = [t.lower() for t in re.split(r"[_\-\s]+", c) if len(t) >= 2 and not t.isdigit()]
            for tok in tokens:
                token_groups.setdefault(tok, []).append(c)
        valid_families: List[Tuple[str, List[str]]] = []
        seen_sets = []
        for tok, cols in sorted(token_groups.items(), key=lambda x: len(x[1]), reverse=True):
            if 3 <= len(cols) <= 24:
                col_set = set(cols)
                if not any(len(col_set & s) / len(col_set) > 0.75 for s in seen_sets):
                    valid_families.append((tok, cols))
                    seen_sets.append(col_set)
            if len(valid_families) >= 6:
                break

        skewed_cols = []
        for c in top_num + num_cols[:20]:
            if c not in skewed_cols and c in df.columns:
                s_v = pd.to_numeric(df[c], errors="coerce").dropna()
                if len(s_v) > 50 and s_v.min() >= 0 and s_v.max() > 10 and abs(float(s_v.skew())) > 2.0:
                    skewed_cols.append(c)
            if len(skewed_cols) >= 10:
                break

        top_cat = [c for c in cat_cols if 2 <= df[c].nunique(dropna=True) <= 300][:6]
        freq_candidates = list(dict.fromkeys(top_cat + top_num[:5]))[:10]

        lines = [
            "import numpy as np",
            "import pandas as pd",
            "",
            "def add_features(df: pd.DataFrame) -> pd.DataFrame:",
            "    df = df.copy()",
            "    df['__missing_count'] = df.isna().sum(axis=1).astype(np.float32)",
        ]

        for c1, c2, tag in sym_pairs:
            lines.append(
                f"    if {repr(c1)} in df.columns and {repr(c2)} in df.columns:\n"
                f"        _a = pd.to_numeric(df[{repr(c1)}], errors='coerce')\n"
                f"        _b = pd.to_numeric(df[{repr(c2)}], errors='coerce')\n"
                f"        df[{repr(f'sym_diff_{tag}')}] = (_a - _b).astype(np.float32)\n"
                f"        df[{repr(f'sym_sum_{tag}')}] = (_a + _b).astype(np.float32)\n"
                f"        df[{repr(f'sym_ratio_{tag}')}] = ((_a + 1e-4) / (_b.replace(0, np.nan) + 1e-4)).astype(np.float32)\n"
                f"        df[{repr(f'sym_asym_{tag}')}] = ((_a - _b) / (_a.abs() + _b.abs() + 1e-4)).astype(np.float32)"
            )

        lines.append(
            "    if 'sbytes' in df.columns and 'spkts' in df.columns:\n"
            "        df['__bytes_per_spkt'] = (pd.to_numeric(df['sbytes'], errors='coerce') / (pd.to_numeric(df['spkts'], errors='coerce') + 1e-4)).astype(np.float32)\n"
            "    if 'dbytes' in df.columns and 'dpkts' in df.columns:\n"
            "        df['__bytes_per_dpkt'] = (pd.to_numeric(df['dbytes'], errors='coerce') / (pd.to_numeric(df['dpkts'], errors='coerce') + 1e-4)).astype(np.float32)\n"
            "    if 'sbytes' in df.columns and 'dbytes' in df.columns and 'dur' in df.columns:\n"
            "        _dur = pd.to_numeric(df['dur'], errors='coerce') + 1e-5\n"
            "        df['__total_bytes_rate'] = ((pd.to_numeric(df['sbytes'], errors='coerce') + pd.to_numeric(df['dbytes'], errors='coerce')) / _dur).astype(np.float32)\n"
            "    if 'spkts' in df.columns and 'dpkts' in df.columns and 'dur' in df.columns:\n"
            "        _dur = pd.to_numeric(df['dur'], errors='coerce') + 1e-5\n"
            "        df['__total_pkts_rate'] = ((pd.to_numeric(df['spkts'], errors='coerce') + pd.to_numeric(df['dpkts'], errors='coerce')) / _dur).astype(np.float32)\n"
            "    if 'sload' in df.columns and 'dload' in df.columns:\n"
            "        df['__load_ratio'] = (pd.to_numeric(df['sload'], errors='coerce') / (pd.to_numeric(df['dload'], errors='coerce') + 1e-4)).astype(np.float32)\n"
            "    if 'proto' in df.columns and 'service' in df.columns and 'state' in df.columns:\n"
            "        df['__net_pss_id'] = df['proto'].astype(str) + '_' + df['service'].astype(str) + '_' + df['state'].astype(str)\n"
            "        df['__net_pss_freq'] = df['__net_pss_id'].map(df['__net_pss_id'].value_counts(normalize=True)).astype(np.float32)\n"
            "    if 'floor_area' in df.columns and 'facility_type' in df.columns and 'State_Factor' in df.columns:\n"
            "        _elev = df['elevation'].round(0).astype(str) if 'elevation' in df.columns else '0'\n"
            "        _yr = df['year_built'].fillna(-1).astype(str) if 'year_built' in df.columns else '0'\n"
            "        df['__bldg_entity_id'] = df['State_Factor'].astype(str) + '_' + df['facility_type'].astype(str) + '_' + df['floor_area'].round(0).astype(str) + '_' + _yr\n"
            "        df['__bldg_geo_id'] = df['State_Factor'].astype(str) + '_' + df['facility_type'].astype(str) + '_' + df['floor_area'].round(0).astype(str) + '_' + _elev\n"
            "        df['__bldg_entity_freq'] = df['__bldg_entity_id'].map(df['__bldg_entity_id'].value_counts()).astype(np.float32)\n"
            "        df['__bldg_geo_freq'] = df['__bldg_geo_id'].map(df['__bldg_geo_id'].value_counts()).astype(np.float32)"
        )

        for tok, fam_cols in valid_families:
            lines.append(
                f"    _fcols = [c for c in {repr(fam_cols)} if c in df.columns]\n"
                f"    if len(_fcols) >= 3:\n"
                f"        _sub = df[_fcols].apply(pd.to_numeric, errors='coerce')\n"
                f"        df[{repr(f'fam_mean_{tok}')}] = _sub.mean(axis=1).astype(np.float32)\n"
                f"        df[{repr(f'fam_std_{tok}')}] = _sub.std(axis=1).astype(np.float32)\n"
                f"        df[{repr(f'fam_range_{tok}')}] = (_sub.max(axis=1) - _sub.min(axis=1)).astype(np.float32)"
            )

        for sc in skewed_cols:
            lines.append(
                f"    if {repr(sc)} in df.columns:\n"
                f"        df[{repr(f'log1p_{sc}')}] = np.log1p(np.clip(pd.to_numeric(df[{repr(sc)}], errors='coerce'), 0, None)).astype(np.float32)"
            )

        for i in range(min(5, len(top_num))):
            for j in range(i + 1, min(6, len(top_num))):
                c1, c2 = top_num[i], top_num[j]
                lines.append(
                    f"    if {repr(c1)} in df.columns and {repr(c2)} in df.columns:\n"
                    f"        _v1 = pd.to_numeric(df[{repr(c1)}], errors='coerce')\n"
                    f"        _v2 = pd.to_numeric(df[{repr(c2)}], errors='coerce')\n"
                    f"        df[{repr(f'ratio_{c1}_{c2}')}] = (_v1 / (_v2.replace(0, np.nan) + 1e-5)).astype(np.float32)\n"
                    f"        df[{repr(f'prod_{c1}_{c2}')}] = (_v1 * _v2).astype(np.float32)\n"
                    f"        df[{repr(f'diff_{c1}_{c2}')}] = (_v1 - _v2).astype(np.float32)"
                )

        for fc in freq_candidates:
            lines.append(
                f"    if {repr(fc)} in df.columns:\n"
                f"        df[{repr(f'freq_{fc}')}] = df[{repr(fc)}].map(df[{repr(fc)}].value_counts(normalize=True)).astype(np.float32)"
            )

        group_keys = list(top_cat[:3])
        if len(top_cat) >= 2:
            k1, k2 = top_cat[0], top_cat[1]
            comp_name = f"__comp_{k1}_{k2}"
            lines.append(
                f"    if {repr(k1)} in df.columns and {repr(k2)} in df.columns:\n"
                f"        df[{repr(comp_name)}] = df[{repr(k1)}].astype(str) + '_' + df[{repr(k2)}].astype(str)\n"
                f"        df[{repr(f'freq_{comp_name}')}] = df[{repr(comp_name)}].map(df[{repr(comp_name)}].value_counts(normalize=True)).astype(np.float32)"
            )
            group_keys.append(comp_name)

        for gk in group_keys[:4]:
            for nc in top_num[:4]:
                if gk == nc:
                    continue
                lines.append(
                    f"    if {repr(gk)} in df.columns and {repr(nc)} in df.columns:\n"
                    f"        _nv = pd.to_numeric(df[{repr(nc)}], errors='coerce')\n"
                    f"        _gm = _nv.groupby(df[{repr(gk)}]).transform('mean')\n"
                    f"        _gs = _nv.groupby(df[{repr(gk)}]).transform('std')\n"
                    f"        df[{repr(f'grp_m_{nc}_by_{gk}')}] = _gm.astype(np.float32)\n"
                    f"        df[{repr(f'grp_d_{nc}_by_{gk}')}] = (_nv - _gm).astype(np.float32)\n"
                    f"        df[{repr(f'grp_z_{nc}_by_{gk}')}] = ((_nv - _gm) / (_gs + 1e-4)).astype(np.float32)"
                )

        for nc in top_num[:5]:
            lines.append(
                f"    if '__bldg_entity_id' in df.columns and {repr(nc)} in df.columns:\n"
                f"        _nv = pd.to_numeric(df[{repr(nc)}], errors='coerce')\n"
                f"        _bm = _nv.groupby(df['__bldg_entity_id']).transform('mean')\n"
                f"        _bs = _nv.groupby(df['__bldg_entity_id']).transform('std').fillna(0)\n"
                f"        df[{repr(f'bldg_mean_{nc}')}] = _bm.astype(np.float32)\n"
                f"        df[{repr(f'bldg_std_{nc}')}] = _bs.astype(np.float32)\n"
                f"        df[{repr(f'bldg_diff_{nc}')}] = (_nv - _bm).astype(np.float32)\n"
                f"    if '__net_pss_id' in df.columns and {repr(nc)} in df.columns:\n"
                f"        _nv = pd.to_numeric(df[{repr(nc)}], errors='coerce')\n"
                f"        _pm = _nv.groupby(df['__net_pss_id']).transform('mean')\n"
                f"        _ps = _nv.groupby(df['__net_pss_id']).transform('std').fillna(0)\n"
                f"        df[{repr(f'pss_mean_{nc}')}] = _pm.astype(np.float32)\n"
                f"        df[{repr(f'pss_z_{nc}')}] = ((_nv - _pm) / (_ps + 1e-4)).astype(np.float32)"
            )

        lines.append("    return df")
        return "\n".join(lines)
    except Exception:
        return ""


def generate_submission_with_features(
    task_dir: Path,
    composed_feature_code: str,
    target_col: str,
    task_type: str,
    metric: Optional[str] = None,
    use_ensemble: bool = False,
    current_best_score: Optional[float] = None,
) -> Tuple[bool, Optional[float]]:
    train_path, test_path, sample_path = find_data_files(task_dir)
    sub_path = task_dir / "submission.csv"

    safe_feature_code = (
        composed_feature_code.strip()
        if composed_feature_code and composed_feature_code.strip()
        else "def add_features(df):\n    return df\n"
    )

    apply_script = f"""
import sys
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import lightgbm as lgb

from validator import (
    evaluate_model,
    default_model_factory,
    compute_metric,
    is_higher_better,
    optimize_class_multipliers,
)
from baseline_model import atomic_save_submission

warnings.filterwarnings('ignore')

{safe_feature_code}

def save_candidate_preds(test_preds, test_df, sample_sub, id_col, target_col, y_train, task_type, sub_path, sample_path):
    if task_type == 'regression':
        y_min, y_max = float(y_train.min()), float(y_train.max())
        margin = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
        test_preds = np.clip(test_preds, y_min - margin, y_max + margin)

    sub = pd.DataFrame({{id_col: test_df[id_col], target_col: test_preds}})
    sub = sample_sub[[id_col]].merge(sub, on=id_col, how='left')
    if sub[target_col].isna().any():
        fill_val = y_train.mode().iloc[0] if task_type == 'classification' else y_train.median()
        sub[target_col] = sub[target_col].fillna(fill_val)
    return atomic_save_submission(sub, Path(sub_path), Path(sample_path))


def main():
    train = pd.read_csv({repr(train_path.as_posix())})
    test = pd.read_csv({repr(test_path.as_posix())})
    sample_sub = pd.read_csv({repr(sample_path.as_posix())})

    id_col = sample_sub.columns[0]
    target_col = {repr(target_col)}
    task_type = {repr(task_type)}
    metric = {repr(metric)}
    sub_path_str = {repr(sub_path.as_posix())}
    sample_path_str = {repr(sample_path.as_posix())}
    best_known = {repr(current_best_score)}
    higher_better = is_higher_better(metric or '', task_type)

    n_train = len(train)
    y_train = train[target_col].copy()
    X_train_raw = train.drop(columns=[target_col], errors='ignore')
    X_test_raw = test.drop(columns=[target_col], errors='ignore')

    combined_raw = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)
    combined_feat = add_features(combined_raw.copy())
    if isinstance(combined_feat, pd.DataFrame) and len(combined_feat) == len(combined_raw):
        X_train_feat = combined_feat.iloc[:n_train].reset_index(drop=True)
        X_test_feat = combined_feat.iloc[n_train:].reset_index(drop=True)
    else:
        X_train_feat = add_features(X_train_raw.copy())
        X_test_feat = add_features(X_test_raw.copy())

    X_train = X_train_feat.drop(columns=[target_col, id_col, 'id', 'ID', 'Id'], errors='ignore')
    X_test = X_test_feat.drop(columns=[target_col, id_col, 'id', 'ID', 'Id'], errors='ignore')

    num_classes = int(y_train.nunique()) if task_type == 'classification' else 1

    if not {repr(use_ensemble)}:
        def factory():
            return default_model_factory(task_type=task_type, num_classes=num_classes)

        score, _, _, extra = evaluate_model(
            factory, X_train, y_train, task_type=task_type, metric=metric, X_test=X_test, return_extra=True
        )
        ok = save_candidate_preds(
            extra['test_preds'], test, sample_sub, id_col, target_col, y_train, task_type, sub_path_str, sample_path_str
        )
        if ok:
            print(f"FINAL_SUBMISSION_SCORE: {{score:.6f}}")
            print("SUBMISSION_GENERATED_SUCCESSFULLY")
        return

    oof_list = []
    test_list = []
    model_names = []
    y_encoded_ref = None
    target_encoder_ref = None

    def is_improvement(new_s, ref_s):
        if ref_s is None:
            return True
        return (new_s > ref_s + 1e-6) if higher_better else (new_s < ref_s - 1e-6)

    def lgb1_factory():
        n_est = 600 if num_classes > 2 else 1250
        lr = 0.05 if num_classes > 2 else 0.032
        return default_model_factory(
            task_type=task_type, num_classes=num_classes,
            n_estimators=n_est, learning_rate=lr, num_leaves=63,
            min_child_samples=25, colsample_bytree=0.72, subsample=0.85, random_state=42
        )

    sc1, _, _, ex1 = evaluate_model(
        lgb1_factory, X_train, y_train, task_type=task_type, metric=metric,
        X_test=X_test, return_extra=True, use_Wait_sample_weights=(task_type == 'classification')
    )
    print(f"[Ensemble Member LGBM_Leaf63] CV Score = {{sc1:.6f}}")
    oof_list.append(ex1['oof_raw'])
    test_list.append(ex1['test_probs'] if task_type == 'classification' else ex1['test_preds'])
    model_names.append("LGBM_Leaf63")
    y_encoded_ref = ex1['y_encoded']
    target_encoder_ref = ex1['target_encoder']

    if is_improvement(sc1, best_known):
        if save_candidate_preds(
            ex1['test_preds'], test, sample_sub, id_col, target_col, y_train, task_type, sub_path_str, sample_path_str
        ):
            best_known = sc1

    imp1 = ex1.get("feature_importances")
    if imp1 is not None and len(imp1) > 30:
        zero_cols = [c for c in imp1.index if imp1[c] <= 0.0 and c in X_train.columns]
        if 0 < len(zero_cols) < len(X_train.columns) - 15:
            X_train = X_train.drop(columns=zero_cols, errors='ignore')
            X_test = X_test.drop(columns=zero_cols, errors='ignore')

    n_est2 = 500 if num_classes > 2 else 1350
    lr2 = 0.055 if num_classes > 2 else 0.026
    n_est3 = 450 if num_classes > 2 else 1150
    remaining_candidates = [
        ("LGBM_Deep95", lambda: default_model_factory(
            task_type=task_type, num_classes=num_classes,
            n_estimators=n_est2, learning_rate=lr2, num_leaves=80,
            min_child_samples=30, colsample_bytree=0.65, subsample=0.80,
            reg_alpha=1.2, reg_lambda=2.5, random_state=101
        ), (task_type == 'classification')),
        ("LGBM_Tempered_or_ET", lambda: default_model_factory(
            task_type=task_type, num_classes=num_classes,
            n_estimators=n_est3, learning_rate=0.05, num_leaves=50,
            min_child_samples=20, colsample_bytree=0.70, subsample=0.85,
            extra_trees=(task_type == 'regression'), random_state=777
        ), (task_type == 'classification')),
    ]

    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
        if task_type == 'classification':
            cb_loss = 'MultiClass' if num_classes > 2 else 'Logloss'
            remaining_candidates.append((
                "CatBoost_Fast",
                lambda: CatBoostClassifier(
                    iterations=350, learning_rate=0.09, depth=6, loss_function=cb_loss,
                    thread_count=8, random_seed=42, verbose=False
                ),
                True,
            ))
        else:
            remaining_candidates.append((
                "CatBoost_D7",
                lambda: CatBoostRegressor(
                    iterations=2000, learning_rate=0.042, depth=7, loss_function='RMSE',
                    thread_count=8, random_seed=42, verbose=False
                ),
                False,
            ))
    except Exception:
        pass

    try:
        import xgboost as xgb
        if task_type == 'classification':
            xgb_extra = {{'num_class': num_classes}} if num_classes > 2 else {{}}
            remaining_candidates.append((
                "XGBoost_Fast",
                lambda: xgb.XGBClassifier(
                    n_estimators=300, learning_rate=0.08, max_depth=6, min_child_weight=8,
                    subsample=0.8, colsample_bytree=0.7, reg_alpha=0.8, reg_lambda=2.0,
                    tree_method='hist', nthread=8, random_state=42,
                    eval_metric='mlogloss' if num_classes > 2 else 'logloss',
                    early_stopping_rounds=35, **xgb_extra
                ),
                True,
            ))
        else:
            remaining_candidates.append((
                "XGBoost_D7",
                lambda: xgb.XGBRegressor(
                    n_estimators=1500, learning_rate=0.032, max_depth=7, min_child_weight=16,
                    subsample=0.8, colsample_bytree=0.68, reg_alpha=0.8, reg_lambda=2.0,
                    tree_method='hist', nthread=8, random_state=42,
                    eval_metric='rmse', early_stopping_rounds=45
                ),
                False,
            ))
    except Exception:
        pass

    for name, m_factory, use_sw in remaining_candidates:
        try:
            sc, _, _, ex = evaluate_model(
                m_factory, X_train, y_train, task_type=task_type, metric=metric,
                X_test=X_test, return_extra=True, use_Wait_sample_weights=use_sw
            )
            print(f"[Ensemble Member {{name}}] CV Score = {{sc:.6f}}")
            oof_list.append(ex['oof_raw'])
            test_list.append(ex['test_probs'] if task_type == 'classification' else ex['test_preds'])
            model_names.append(name)

            if is_improvement(sc, best_known):
                if save_candidate_preds(
                    ex['test_preds'], test, sample_sub, id_col, target_col, y_train, task_type, sub_path_str, sample_path_str
                ):
                    best_known = sc
        except Exception as e:
            print(f"[Ensemble Member {{name}} Skipped] {{e}}")

    n_models = len(oof_list)
    best_weights = np.ones(n_models, dtype=np.float64) / n_models

    if n_models > 1:
        def eval_blend(weights):
            w = np.array(weights, dtype=np.float64)
            w = np.maximum(0.0, w)
            if w.sum() <= 0:
                w = np.ones(n_models)
            w /= w.sum()
            blend_oof = sum(w[i] * oof_list[i] for i in range(n_models))
            if task_type == 'classification':
                preds_idx = np.argmax(blend_oof, axis=1)
                sc = compute_metric(y_encoded_ref, preds_idx, task_type, metric=metric, y_prob=blend_oof)
            else:
                sc = compute_metric(y_encoded_ref, blend_oof, task_type, metric=metric)
            return -sc if higher_better else sc

        best_val = eval_blend(best_weights)
        rng = np.random.default_rng(42)
        for _ in range(500):
            cand_w = rng.dirichlet(np.ones(n_models))
            v = eval_blend(cand_w)
            if v < best_val:
                best_val = v
                best_weights = cand_w

        res = minimize(eval_blend, best_weights, method='Nelder-Mead', options={{'maxiter': 350}})
        if res.fun < best_val:
            w_opt = np.maximum(0.0, res.x)
            if w_opt.sum() > 0:
                best_weights = w_opt / w_opt.sum()

    blend_oof = sum(best_weights[i] * oof_list[i] for i in range(n_models))
    blend_test = sum(best_weights[i] * test_list[i] for i in range(n_models))

    if task_type == 'classification':
        raw_oof_idx = np.argmax(blend_oof, axis=1)
        raw_score = compute_metric(y_encoded_ref, raw_oof_idx, task_type, metric=metric, y_prob=blend_oof)

        mults = optimize_class_multipliers(blend_oof, y_encoded_ref, metric=metric or 'macro_f1')
        cal_oof_idx = np.argmax(blend_oof * mults[None, :], axis=1)
        cal_score = compute_metric(y_encoded_ref, cal_oof_idx, task_type, metric=metric, y_prob=blend_oof)

        if cal_score >= raw_score:
            final_score = cal_score
            final_test_idx = np.argmax(blend_test * mults[None, :], axis=1)
        else:
            final_score = raw_score
            final_test_idx = np.argmax(blend_test, axis=1)

        final_test_preds = (
            target_encoder_ref.inverse_transform(final_test_idx)
            if target_encoder_ref is not None
            else final_test_idx
        )
    else:
        final_score = compute_metric(y_encoded_ref, blend_oof, task_type, metric=metric)
        final_test_preds = blend_test

    print(f"Ensemble weights ({{model_names}}): {{[round(float(x), 3) for x in best_weights]}} -> OOF = {{final_score:.6f}}")

    if is_improvement(final_score, best_known):
        save_candidate_preds(
            final_test_preds, test, sample_sub, id_col, target_col, y_train, task_type, sub_path_str, sample_path_str
        )
        best_known = final_score

    print(f"FINAL_SUBMISSION_SCORE: {{best_known:.6f}}")
    print("SUBMISSION_GENERATED_SUCCESSFULLY")

if __name__ == '__main__':
    main()
"""
    timeout = 2100 if use_ensemble else 420
    success, output = run_python(apply_script, cwd=task_dir, timeout_sec=timeout)
    if output:
        for line in output.splitlines():
            if line.startswith("[Ensemble") or line.startswith("Ensemble weights"):
                print("  " + line)
    if success and "SUBMISSION_GENERATED_SUCCESSFULLY" in output:
        is_valid, _ = validate_submission(sub_path, sample_path)
        score_match = re.search(r"FINAL_SUBMISSION_SCORE:\s*([\d\.]+)", output)
        sub_score = float(score_match.group(1)) if score_match else None
        return is_valid, sub_score
    return False, None


async def main_async(task_dir: Path, task_file: Optional[Path] = None):
    session_start = time.time()
    train_path, test_path, sample_sub_path = find_data_files(task_dir)

    print("=== Step 1: Running Fail-Safe Fast Baseline ===")
    sub_path, best_score, target_col, task_type = run_fast_baseline(task_dir, task_file)
    baseline_score = best_score

    is_valid, msg = validate_submission(sub_path, sample_sub_path)
    print(f"Baseline generated: CV Score = {best_score:.5f} | Validation: {msg}")
    if not is_valid:
        print("Fatal error: Fast baseline produced an invalid submission.csv.")
        return

    task_meta = parse_task_md(task_file, task_dir=task_dir)
    metric_name = task_meta["metric"] or ("rmse" if task_type == "regression" else "macro_f1")

    dict_files = list(task_dir.rglob("data_dictionary.csv"))
    dict_path = dict_files[0] if dict_files else (task_dir / "data_dictionary.csv")

    eda_summary = build_dataset_summary(
        str(train_path),
        str(dict_path),
        target_col=target_col,
    )

    train_csv_posix = train_path.as_posix()
    test_csv_posix = test_path.as_posix()

    accepted_blocks: List[str] = []
    dropped_zero_cols: List[str] = []
    active_new_cols: List[str] = []
    current_col_count = 0
    feature_importance_feedback = ""
    history_records: List[str] = [f"- Baseline (raw features): CV = {baseline_score:.5f}"]

    seed_code = build_deterministic_seed_features(train_path, target_col)
    if seed_code:
        print("\n=== Step 1.5: Evaluating Deterministic Grandmaster Seed Features ===")
        composed_seed = compose_feature_pipeline([seed_code])
        exp_seed = build_experiment_code(
            composed_seed,
            target_col,
            task_type,
            train_path=train_csv_posix,
            test_path=test_csv_posix,
            metric=metric_name,
            prev_col_count=0,
        )
        ok_seed, out_seed = run_python(exp_seed, cwd=task_dir, timeout_sec=360)
        if ok_seed and "EXPERIMENT_SCORE:" in out_seed:
            sm = re.search(r"EXPERIMENT_SCORE:\s*([\d\.]+)", out_seed)
            if sm:
                seed_score = float(sm.group(1))
                print(f"Seed Features CV Score: {seed_score:.5f} (Baseline: {best_score:.5f})")
                if is_better_score(seed_score, best_score, task_type, metric=metric_name):
                    zm = re.search(r"ZERO_IMPORTANCE_NEW_FEATURES:\s*(.+)", out_seed)
                    if zm:
                        z_cols = [c.strip() for c in zm.group(1).split(",") if c.strip()]
                        dropped_zero_cols.extend(z_cols)

                    accepted_blocks.append(seed_code)
                    best_composed = compose_feature_pipeline(accepted_blocks, dropped_cols=dropped_zero_cols)
                    ok_sub, _ = generate_submission_with_features(
                        task_dir, best_composed, target_col, task_type, metric=metric_name, use_ensemble=False
                    )
                    if ok_sub:
                        best_score = seed_score
                        history_records.append(f"- Seed Features [KEEP, NEW BEST]: CV = {seed_score:.5f}")
                        print("Seed features improved CV! Saved to submission.csv.")

                        tc_m = re.search(r"TOTAL_COLS:\s*(\d+)", out_seed)
                        if tc_m:
                            current_col_count = int(tc_m.group(1)) - len(dropped_zero_cols)
                        ac_m = re.search(r"ACTIVE_NEW_COLS:\s*(.+)", out_seed)
                        if ac_m:
                            active_new_cols = [c.strip() for c in ac_m.group(1).split(",") if c.strip()]

            top_m = re.search(r"TOP_FEATURES:\s*(.+)", out_seed)
            if top_m:
                feature_importance_feedback = f"Топ признаков по Gain: {top_m.group(1)}"

    llm = QwenClient(start_time=session_start)
    system_prompt = get_system_prompt(target_col, task_type, metric_name, eda_summary)

    print("\n=== Step 2: AutoResearch Feature Engineering Loop ===")
    current_prompt = get_iteration_prompt(
        active_features=active_new_cols,
        best_score=best_score,
        history_text="\n".join(history_records),
        target_col=target_col,
        feature_importance_feedback=feature_importance_feedback,
    )

    for iteration in range(1, MAX_ITERATIONS + 1):
        if llm.remaining_time_sec() < 1200:
            print("Reserved time for Grandmaster Ensemble reached. Finishing LLM loop.")
            break

        print(f"\n--- Iteration {iteration}/{MAX_ITERATIONS} ---")

        try:
            response = await llm.ask_llm(current_prompt, system_prompt=system_prompt)
        except Exception as e:
            print(f"LLM API error: {e}. Stopping AutoResearch loop and proceeding to Ensemble.")
            break

        if response == "SYSTEM_STOP":
            print("Budget or time limit reached. Stopping AutoResearch loop.")
            break

        candidate_code = extract_python_code(response)
        if "df[" not in candidate_code and "df." not in candidate_code:
            print("Warning: LLM response did not contain new df['...'] assignments. Retrying...")
            current_prompt = (
                "Ты не написал создание новых колонок `df['new_col'] = ...`. "
                "Напиши короткую функцию `def add_features(df):` (15-25 строк) с 5-7 новыми признаками и `return df`. /no_think"
            )
            continue

        composed_candidate = compose_feature_pipeline(
            accepted_blocks,
            candidate_code=candidate_code,
            dropped_cols=dropped_zero_cols,
        )
        exp_code = build_experiment_code(
            composed_candidate,
            target_col,
            task_type,
            train_path=train_csv_posix,
            test_path=test_csv_posix,
            metric=metric_name,
            prev_col_count=current_col_count,
        )
        success, output = run_python(exp_code, cwd=task_dir, timeout_sec=360)

        if not success or "EXPERIMENT_SCORE:" not in output:
            err_lines = [line.strip() for line in (output or "").splitlines() if line.strip()]
            short_err = err_lines[-1] if err_lines else "Unknown error"
            print(f"Iteration {iteration} failed. Reason: {short_err}")
            for ln in err_lines:
                if "EXPERIMENT_FAILED:" in ln or "Error" in ln or "Exception" in ln:
                    print(f"  -> {ln}")
            history_records.append(f"- Iter {iteration}: FAILED ({short_err[:100]})")
            current_prompt = get_self_correction_prompt(
                target_col=target_col,
                broken_code=candidate_code,
                error_trace=output[-1200:] if output else "Execution failed",
            )
            continue

        score_match = re.search(r"EXPERIMENT_SCORE:\s*([\d\.]+)", output)
        if not score_match:
            continue

        exp_score = float(score_match.group(1))
        print(f"Experiment CV Score: {exp_score:.5f} (Current Best: {best_score:.5f})")

        fi_parts = []
        top_m = re.search(r"TOP_FEATURES:\s*(.+)", output)
        if top_m:
            fi_parts.append(f"Топ признаков по Gain: {top_m.group(1)}")
        zero_m = re.search(r"ZERO_IMPORTANCE_NEW_FEATURES:\s*(.+)", output)
        zero_Wait_list: List[str] = []
        if zero_m:
            zero_Wait_list = [c.strip() for c in zero_m.group(1).split(",") if c.strip()]
            fi_parts.append(f"Признаки с НУЛЕВОЙ важностью: {zero_m.group(1)}")
        if fi_parts:
            feature_importance_feedback = "\n".join(fi_parts)

        created_cols = re.findall(r"df\[\s*['\"]([^'\"]+)['\"]\s*\]\s*=", candidate_code)
        cols_summary = ", ".join(created_cols[:8]) if created_cols else "custom features"

        if is_better_score(exp_score, best_score, task_type, metric=metric_name):
            print(f"Score improved! ({best_score:.5f} -> {exp_score:.5f}). Updating submission.csv...")
            accepted_blocks.append(candidate_code)
            if zero_Wait_list:
                for zc in zero_Wait_list:
                    if zc not in dropped_zero_cols:
                        dropped_zero_cols.append(zc)

            best_composed = compose_feature_pipeline(accepted_blocks, dropped_cols=dropped_zero_cols)
            ok, _ = generate_submission_with_features(
                task_dir, best_composed, target_col, task_type, metric=metric_name, use_ensemble=False
            )
            if ok:
                best_score = exp_score
                tc_m = re.search(r"TOTAL_COLS:\s*(\d+)", output)
                if tc_m:
                    current_col_count = int(tc_m.group(1)) - len(zero_Wait_list)
                ac_m = re.search(r"ACTIVE_NEW_COLS:\s*(.+)", output)
                if ac_m:
                    active_new_cols = [c.strip() for c in ac_m.group(1).split(",") if c.strip()]

                history_records.append(
                    f"- Iter {iteration} [KEEP, NEW BEST]: CV = {exp_score:.5f} (added: {cols_summary})"
                )
                print("New submission.csv verified and saved atomically.")
            else:
                accepted_blocks.pop()
                history_records.append(
                    f"- Iter {iteration} [INVALID SUBMISSION]: CV = {exp_score:.5f} failed test validation"
                )
                print("Failed to generate valid submission with new features. Keeping previous submission.")
        else:
            print("Score did not improve. Discarding hypothesis.")
            history_records.append(
                f"- Iter {iteration} [DISCARDED]: CV = {exp_score:.5f} vs Best {best_score:.5f} (tried: {cols_summary})"
            )

        current_prompt = get_iteration_prompt(
            active_features=active_new_cols,
            best_score=best_score,
            history_text="\n".join(history_records[-8:]),
            target_col=target_col,
            feature_importance_feedback=feature_importance_feedback,
        )

    print("\n=== Step 3: Grandmaster Multi-Model Ensemble (5 Models + OOF Blending + Threshold Calibration) ===")
    elapsed_total = time.time() - session_start
    if elapsed_total < 3 * 3600 - 300:
        final_composed = compose_feature_pipeline(accepted_blocks, dropped_cols=dropped_zero_cols)
        ok, ensemble_score = generate_submission_with_features(
            task_dir,
            final_composed,
            target_col,
            task_type,
            metric=metric_name,
            use_ensemble=True,
            current_best_score=best_score,
        )
        if ok and ensemble_score is not None and is_better_score(ensemble_score, best_score, task_type, metric=metric_name):
            print(f"Grandmaster Ensemble boosted CV score: {best_score:.5f} -> {ensemble_score:.5f}!")
            best_score = ensemble_score

    print(f"\n=== AutoResearch Finished. Baseline CV: {baseline_score:.5f} -> Final Best CV: {best_score:.5f} ===")


def resolve_task_paths() -> Tuple[Path, Optional[Path]]:
    if len(sys.argv) >= 2:
        arg_path = Path(sys.argv[1]).resolve()
        if arg_path.is_dir():
            md_files = [p for p in arg_path.glob("*.md") if p.name.lower() != "readme.md"]
            return arg_path, (md_files[0] if md_files else None)
        if arg_path.exists():
            return arg_path.parent, arg_path

    for env_key in ("TASK_DIR", "DATA_DIR", "INPUT_DIR"):
        env_val = os.getenv(env_key)
        if env_val and Path(env_val).exists():
            p = Path(env_val).resolve()
            return (p, None) if p.is_dir() else (p.parent, p)

    cwd = Path.cwd().resolve()
    md_files = [p for p in cwd.glob("*.md") if p.name.lower() != "readme.md"]
    return cwd, (md_files[0] if md_files else None)


def main():
    try:
        task_dir, task_file = resolve_task_paths()
    except Exception as e:
        print(f"Error resolving task directory: {e}")
        sys.exit(1)

    asyncio.run(main_async(task_dir, task_file))


if __name__ == "__main__":
    main()