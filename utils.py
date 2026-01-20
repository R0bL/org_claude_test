"""
Entity Resolution Pipeline Utilities
=====================================

Reusable utilities for database access, parsing, timing, and checkpointing.
"""

import re
import time
import json
import functools
from datetime import datetime
from typing import Optional, Dict, Any, Callable
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pyathena import connect
from pyathena.pandas.cursor import PandasCursor

from config import config


# =============================================================================
# DATABASE MANAGER
# =============================================================================

class DatabaseManager:
    """
    Manages Athena database connections with lazy initialization.
    
    Usage:
        db = DatabaseManager()
        df = db.execute_query("SELECT * FROM table LIMIT 10", "Fetching sample")
    """
    
    def __init__(self, cfg=None):
        self.config = cfg or config.database
        self._connection = None
        self._cursor = None
    
    def get_connection(self):
        """Lazy connection initialization"""
        if self._connection is None:
            # Read env vars directly at connection time (not from cached config)
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
        """
        Execute SQL query and return DataFrame.
        
        Args:
            query: SQL query string
            description: Human-readable description for logging
            
        Returns:
            pandas DataFrame with results
        """
        conn, cursor = self.get_connection()
        log_step(f"Executing: {description}")
        
        start = time.time()
        result = cursor.execute(query).as_pandas()
        elapsed = time.time() - start
        
        log_step(f"  Returned {len(result):,} rows in {elapsed:.1f}s")
        return result
    
    def close(self):
        """Close database connection"""
        if self._connection:
            self._connection.close()
            self._connection = None
            self._cursor = None


# =============================================================================
# METADATA PARSING
# =============================================================================

def parse_java_map(metadata_str: str) -> Dict[str, Any]:
    """
    Parse a Java/Scala-style map string into a Python dictionary.
    
    Handles format like: {key1=value1, key2=value2, ...}
    
    Args:
        metadata_str: String in Java map format
        
    Returns:
        Dictionary with parsed key-value pairs
        
    Example:
        >>> parse_java_map("{name=John, age=30}")
        {'name': 'John', 'age': '30'}
    """
    if metadata_str is None or not isinstance(metadata_str, str):
        return {}
    
    s = metadata_str.strip()
    if s.startswith('{') and s.endswith('}'):
        s = s[1:-1]
    
    if not s:
        return {}
    
    result = {}
    
    # Find all key=value pairs using regex
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
    """
    Safely convert value to float.
    
    Args:
        val: Value to convert
        
    Returns:
        Float value or None if conversion fails
    """
    if val is None or pd.isna(val):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def safe_int(val: Any) -> Optional[int]:
    """
    Safely convert value to int.
    
    Args:
        val: Value to convert
        
    Returns:
        Int value or None if conversion fails
    """
    if val is None or pd.isna(val):
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


# =============================================================================
# TIMING AND LOGGING
# =============================================================================

def log_step(message: str, level: str = "INFO"):
    """
    Log a pipeline step with timestamp.
    
    Args:
        message: Log message
        level: Log level (INFO, WARN, ERROR)
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix = {"INFO": "", "WARN": "[!]", "ERROR": "[X]"}.get(level, "")
    print(f"[{timestamp}] {prefix} {message}")


def timed(description: str = None):
    """
    Decorator to time function execution.
    
    Args:
        description: Optional description (defaults to function name)
        
    Usage:
        @timed("Loading data")
        def load_data():
            ...
    """
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
    """
    Context manager for timing code blocks.
    
    Usage:
        with Timer("Processing data"):
            process_data()
    """
    
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


# =============================================================================
# CHECKPOINTING
# =============================================================================

def save_checkpoint(df: pd.DataFrame, path: str, description: str = ""):
    """
    Save DataFrame checkpoint to Parquet.
    
    Args:
        df: DataFrame to save
        path: Output file path
        description: Description for logging
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    log_step(f"Saving checkpoint: {description or path}")
    df.to_parquet(path, index=False)
    log_step(f"  Saved {len(df):,} rows to {path}")


def load_checkpoint(path: str, description: str = "") -> Optional[pd.DataFrame]:
    """
    Load DataFrame checkpoint from Parquet.
    
    Args:
        path: Input file path
        description: Description for logging
        
    Returns:
        DataFrame or None if file doesn't exist
    """
    if not Path(path).exists():
        log_step(f"Checkpoint not found: {path}", "WARN")
        return None
    
    log_step(f"Loading checkpoint: {description or path}")
    df = pd.read_parquet(path)
    log_step(f"  Loaded {len(df):,} rows")
    return df


def save_json(data: Dict[str, Any], path: str, description: str = ""):
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        path: Output file path
        description: Description for logging
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    log_step(f"Saving JSON: {description or path}")
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    log_step(f"  Saved to {path}")


def load_json(path: str, description: str = "") -> Optional[Dict[str, Any]]:
    """
    Load dictionary from JSON file.
    
    Args:
        path: Input file path
        description: Description for logging
        
    Returns:
        Dictionary or None if file doesn't exist
    """
    if not Path(path).exists():
        log_step(f"JSON not found: {path}", "WARN")
        return None
    
    log_step(f"Loading JSON: {description or path}")
    with open(path, 'r') as f:
        return json.load(f)


# =============================================================================
# PROGRESS TRACKING
# =============================================================================

class ProgressTracker:
    """
    Track progress through multi-step operations.
    
    Usage:
        tracker = ProgressTracker(total_steps=5)
        for i in range(5):
            tracker.step(f"Processing batch {i}")
            process_batch(i)
        tracker.complete()
    """
    
    def __init__(self, total_steps: int, description: str = "Pipeline"):
        self.total_steps = total_steps
        self.description = description
        self.current_step = 0
        self.start_time = time.time()
        self.step_times = []
    
    def step(self, step_description: str):
        """Mark start of a new step"""
        self.current_step += 1
        pct = 100 * self.current_step / self.total_steps
        log_step(f"[{self.current_step}/{self.total_steps}] ({pct:.0f}%) {step_description}")
        self.step_times.append(time.time())
    
    def complete(self):
        """Mark pipeline complete"""
        elapsed = time.time() - self.start_time
        log_step(f"{self.description} complete in {elapsed:.1f}s")
    
    def eta(self) -> str:
        """Estimate time remaining"""
        if self.current_step == 0:
            return "Unknown"
        
        elapsed = time.time() - self.start_time
        avg_per_step = elapsed / self.current_step
        remaining_steps = self.total_steps - self.current_step
        eta_seconds = avg_per_step * remaining_steps
        
        if eta_seconds < 60:
            return f"{eta_seconds:.0f}s"
        elif eta_seconds < 3600:
            return f"{eta_seconds/60:.1f}m"
        else:
            return f"{eta_seconds/3600:.1f}h"


# =============================================================================
# DATA INSPECTION HELPERS
# =============================================================================

def describe_dataframe(df: pd.DataFrame, name: str = "DataFrame"):
    """
    Print summary statistics for a DataFrame.
    
    Args:
        df: DataFrame to describe
        name: Name for display
    """
    print(f"\n{name}")
    print("=" * 60)
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"\nColumn types:")
    for col in df.columns:
        dtype = df[col].dtype
        null_count = df[col].isna().sum()
        null_pct = 100 * null_count / len(df)
        print(f"  {col:<30} {str(dtype):<15} {null_pct:5.1f}% null")


def field_coverage_table(df: pd.DataFrame, fields: list) -> pd.DataFrame:
    """
    Create a field coverage summary table.
    
    Args:
        df: DataFrame to analyze
        fields: List of fields to check
        
    Returns:
        DataFrame with coverage statistics
    """
    coverage = []
    for field in fields:
        if field in df.columns:
            count = df[field].notna().sum()
            pct = 100 * count / len(df)
            coverage.append({
                'field': field,
                'non_null_count': count,
                'coverage_pct': round(pct, 1),
                'status': 'available'
            })
        else:
            coverage.append({
                'field': field,
                'non_null_count': 0,
                'coverage_pct': 0.0,
                'status': 'missing'
            })
    
    return pd.DataFrame(coverage)


if __name__ == "__main__":
    print("Entity Resolution Pipeline Utilities")
    print("=" * 50)
    
    # Test parse_java_map
    test_str = "{name=Test Corp, city=Boston, latitude=42.3}"
    parsed = parse_java_map(test_str)
    print(f"\nparse_java_map test:")
    print(f"  Input: {test_str}")
    print(f"  Output: {parsed}")
    
    # Test Timer
    print(f"\nTimer test:")
    with Timer("Sleep 0.1s"):
        time.sleep(0.1)

