import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd

MAX_LOG_CHARS = 8000
DEFAULT_TIMEOUT_SEC = 900 

def _truncate_log(text: str, max_chars: int = MAX_LOG_CHARS) -> str:
    if not text:
        return ""
    if len(text) > max_chars:
        return (
            f"[Лог обрезан, показаны последние {max_chars} символов]\n"
            + text[-max_chars:]
        )
    return text

def run_python(
    code: str,
    cwd: Optional[Union[str, Path]] = None,
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
) -> Tuple[bool, str]:
    working_dir = Path(cwd).resolve() if cwd else Path.cwd()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    project_root = str(Path(__file__).resolve().parent.parent)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{project_root}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else project_root
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        encoding="utf-8",
        dir=str(working_dir),
    ) as tmp_file:
        tmp_file.write(code)
        tmp_path = Path(tmp_file.name)

    cmd = [sys.executable, str(tmp_path)]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(working_dir),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=env,
        )

        output_parts = []
        if result.stdout:
            output_parts.append(f"--- STDOUT ---\n{result.stdout}")
        if result.stderr:
            output_parts.append(f"--- STDERR ---\n{result.stderr}")

        combined_output = "\n".join(output_parts)

        if result.returncode != 0:
            combined_output += f"\n[Process exited with code {result.returncode}]"
            return False, _truncate_log(combined_output)

        return True, _truncate_log(combined_output)

    except subprocess.TimeoutExpired as exc:
        stdout_str = (
            exc.stdout.decode("utf-8", errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        stderr_str = (
            exc.stderr.decode("utf-8", errors="replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )

        error_msg = (
            f"TimeoutError: Выполнение кода прервано по таймауту ({timeout_sec} сек.). "
            f"Возможно, код содержит бесконечный цикл или слишком долгое обучение.\n"
        )
        if stdout_str:
            error_msg += f"--- STDOUT до таймаута ---\n{stdout_str}\n"
        if stderr_str:
            error_msg += f"--- STDERR до таймаута ---\n{stderr_str}\n"

        return False, _truncate_log(error_msg)

    except Exception as exc:
        return False, f"SandboxExecutionError: Не удалось запустить процесс: {str(exc)}"

    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def validate_submission(
    sub_path: Union[str, Path],
    sample_sub_path: Union[str, Path],
) -> Tuple[bool, str]:
    sub_path = Path(sub_path)
    sample_sub_path = Path(sample_sub_path)

    if not sub_path.exists():
        return False, f"Ошибка валидации: Файл сабмита не найден ({sub_path})."

    if not sample_sub_path.exists():
        return False, f"Ошибка валидации: Эталонный файл не найден ({sample_sub_path})."

    if sub_path.stat().st_size == 0:
        return False, f"Ошибка валидации: Файл {sub_path.name} полностью пуст."

    try:
        sub_df = pd.read_csv(sub_path)
        sample_df = pd.read_csv(sample_sub_path)
    except Exception as exc:
        return False, f"Ошибка чтения CSV файла: {str(exc)}"

    if list(sub_df.columns) != list(sample_df.columns):
        return (
            False,
            f"Несовпадение колонок. Ожидалось: {list(sample_df.columns)}, получено: {list(sub_df.columns)}.",
        )

    if len(sub_df) != len(sample_df):
        return (
            False,
            f"Несовпадение количества строк. Ожидалось: {len(sample_df)}, получено: {len(sub_df)}.",
        )

    id_col = sample_df.columns[0]
    if not sub_df[id_col].equals(sample_df[id_col]):
        if (sub_df[id_col].values != sample_df[id_col].values).any():
            return (
                False,
                f"Колонка идентификатора '{id_col}' не совпадает с sample_submission.csv (нарушен порядок или значения).",
            )

    if sub_df.isna().any().any():
        nan_cols = sub_df.columns[sub_df.isna().any()].tolist()
        return False, f"Обнаружены пустые значения (NaN/None) в колонках: {nan_cols}."

    object_cols = sub_df.select_dtypes(include=["object", "string"]).columns
    for col in object_cols:
        if (sub_df[col].astype(str).str.strip() == "").any():
            return False, f"Обнаружены пустые строки в текстовой колонке '{col}'."

    numeric_cols = sub_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.isinf(sub_df[col]).any():
            return (
                False,
                f"Обнаружены бесконечные значения (+inf / -inf) в колонке '{col}'.",
            )

    return True, "Файл submission.csv успешно прошел все проверки."