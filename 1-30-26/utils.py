"""
Entity Resolution Pipeline Utilities (Minimal)
"""

import re
import time
import json
import functools
from datetime import datetime
from typing import Optional, Dict, Any, Callable
from pathlib import Path

import pandas as pd
from pyathena import connect
from pyathena.pandas.cursor import PandasCursor

from config import config


class DatabaseManager:
    """Manages Athena database connections."""

    def __init__(self, cfg=None):
        self.config = cfg or config.database
        self._connection = None
        self._cursor = None

    def get_connection(self):
        if self._connection is None:
            import os
            aws_key = os.getenv("AWS_ACCESS_KEY_ID") or self.config.AWS_ACCESS_KEY_ID
            aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY") or self.config.AWS_SECRET_ACCESS_KEY
            region = os.getenv("AWS_REGION") or self.config.AWS_REGION
            s3_dir = os.getenv("S3_STAGING_DIR") or self.config.S3_STAGING_DIR

            self._connection = connect(
                aws_access_key_id=aws_key,
                aws_secret_access_key=aws_secret,
                s3_staging_dir=s3_dir,
                region_name=region,
                cursor_class=PandasCursor
            )
            self._cursor = self._connection.cursor()
        return self._connection, self._cursor

    def execute_query(self, query: str, description: str = "") -> pd.DataFrame:
        conn, cursor = self.get_connection()
        log_step(f"Executing: {description}")
        start = time.time()
        result = cursor.execute(query).as_pandas()
        elapsed = time.time() - start
        log_step(f"  Returned {len(result):,} rows in {elapsed:.1f}s")
        return result

    def close(self):
        if self._connection:
            self._connection.close()
            self._connection = None
            self._cursor = None


def parse_java_map(metadata_str: str) -> Dict[str, Any]:
    """Parse Java-style map string {key=value, ...} into dict."""
    if metadata_str is None or not isinstance(metadata_str, str):
        return {}

    s = metadata_str.strip()
    if s.startswith('{') and s.endswith('}'):
        s = s[1:-1]

    if not s:
        return {}

    result = {}
    key_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*='
    matches = list(re.finditer(key_pattern, s))

    for i, match in enumerate(matches):
        key = match.group(1)
        value_start = match.end()
        if i + 1 < len(matches):
            value_end = matches[i + 1].start()
            value = s[value_start:value_end].rstrip(', ')
        else:
            value = s[value_start:]
        value = value.strip()
        if value.lower() == 'null':
            value = None
        result[key] = value

    return result


def safe_float(val: Any) -> Optional[float]:
    if val is None or pd.isna(val):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def log_step(message: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix = {"INFO": "", "WARN": "[!]", "ERROR": "[X]"}.get(level, "")
    print(f"[{timestamp}] {prefix} {message}")


def timed(description: str = None):
    """Decorator to time function execution."""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            desc = description or func.__name__
            log_step(f"Starting: {desc}")
            start = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start
                log_step(f"Completed: {desc} ({elapsed:.1f}s)")
                return result
            except Exception as e:
                elapsed = time.time() - start
                log_step(f"Failed: {desc} ({elapsed:.1f}s) - {str(e)}", "ERROR")
                raise
        return wrapper
    return decorator


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, description: str):
        self.description = description
        self.start = None
        self.elapsed = None

    def __enter__(self):
        log_step(f"Starting: {self.description}")
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.time() - self.start
        if exc_type is None:
            log_step(f"Completed: {self.description} ({self.elapsed:.1f}s)")
        else:
            log_step(f"Failed: {self.description} ({self.elapsed:.1f}s)", "ERROR")
        return False


def save_checkpoint(df: pd.DataFrame, path: str, description: str = ""):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    log_step(f"Saving checkpoint: {description or path}")
    df.to_parquet(path, index=False)
    log_step(f"  Saved {len(df):,} rows to {path}")


def load_checkpoint(path: str, description: str = "") -> Optional[pd.DataFrame]:
    if not Path(path).exists():
        log_step(f"Checkpoint not found: {path}", "WARN")
        return None
    log_step(f"Loading checkpoint: {description or path}")
    df = pd.read_parquet(path)
    log_step(f"  Loaded {len(df):,} rows")
    return df


def load_json(path: str, description: str = "") -> Optional[Dict[str, Any]]:
    if not Path(path).exists():
        log_step(f"JSON not found: {path}", "WARN")
        return None
    log_step(f"Loading JSON: {description or path}")
    with open(path, 'r') as f:
        return json.load(f)


def describe_dataframe(df: pd.DataFrame, name: str = "DataFrame"):
    print(f"\n{name}")
    print("=" * 60)
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"\nColumn types:")
    for col in df.columns:
        dtype = df[col].dtype
        null_pct = 100 * df[col].isna().sum() / len(df)
        print(f"  {col:<30} {str(dtype):<15} {null_pct:5.1f}% null")

