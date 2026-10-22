import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd

MAX_LOG_CHARS = 8000
DEFAULT_TIMEOUT_SEC = 180


def _truncate_log(text: str, max_chars: int = MAX_LOG_CHARS) -> str:
    if not text:
        return ""
    if len(text) > max_chars:
        return (
            f"... [Log truncated, showing last {max_chars} characters] ...\n"
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

    current_dir = Path(__file__).resolve().parent
    project_root = str(
        current_dir.parent if current_dir.name == "agent" else current_dir
    )

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
            f"TimeoutError: Execution timed out after {timeout_sec} seconds. "
            f"The script may contain an infinite loop or excessive training time.\n"
        )
        if stdout_str:
            error_msg += f"--- STDOUT before timeout ---\n{stdout_str}\n"
        if stderr_str:
            error_msg += f"--- STDERR before timeout ---\n{stderr_str}\n"

        return False, _truncate_log(error_msg)

    except Exception as exc:
        return False, f"SandboxExecutionError: Failed to launch process: {str(exc)}"

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
        return False, f"Validation error: Submission file not found ({sub_path})."

    if not sample_sub_path.exists():
        return False, f"Validation error: Sample submission file not found ({sample_sub_path})."

    if sub_path.stat().st_size == 0:
        return False, f"Validation error: File {sub_path.name} is empty."

    try:
        sub_df = pd.read_csv(sub_path)
        sample_df = pd.read_csv(sample_sub_path)
    except Exception as exc:
        return False, f"CSV read error: {str(exc)}"

    if list(sub_df.columns) != list(sample_df.columns):
        return (
            False,
            f"Column mismatch. Expected: {list(sample_df.columns)}, got: {list(sub_df.columns)}.",
        )

    if len(sub_df) != len(sample_df):
        return (
            False,
            f"Row count mismatch. Expected: {len(sample_df)}, got: {len(sub_df)}.",
        )

    if sub_df.isna().any().any():
        nan_cols = sub_df.columns[sub_df.isna().any()].tolist()
        return False, f"Found missing values (NaN/None) in columns: {nan_cols}."

    id_col = sample_df.columns[0]
    if not (sub_df[id_col].astype(str).values == sample_df[id_col].astype(str).values).all():
        return (
            False,
            f"Identifier column '{id_col}' does not match sample_submission.csv (order or values differ).",
        )

    object_cols = sub_df.select_dtypes(include=["object", "string"]).columns
    for col in object_cols:
        if (sub_df[col].astype(str).str.strip() == "").any():
            return False, f"Found empty string values in column: '{col}'."

    numeric_cols = sub_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.isinf(sub_df[col]).any():
            return (
                False,
                f"Found infinite values (+inf / -inf) in column: '{col}'.",
            )

    return True, "submission.csv successfully passed all validation checks."