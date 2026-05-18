"""
Utility functions for ML Algorithm Comparator.
Includes caching, model persistence, report generation, and helpers.
"""

import os
import pickle
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional
from functools import wraps

import pandas as pd
import streamlit as st


# ─────────────────────────────────────────────
# Caching utilities
# ─────────────────────────────────────────────

def get_data_hash(df: pd.DataFrame) -> str:
    """Generate a hash of a DataFrame for cache invalidation."""
    return hashlib.md5(str(df.values).encode()).hexdigest()


def cache_model_result(func):
    """
    Decorator to cache model training results in session state.
    Invalidate cache when data or parameters change.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Generate cache key from arguments
        cache_key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"

        if cache_key not in st.session_state:
            st.session_state[cache_key] = func(*args, **kwargs)

        return st.session_state[cache_key]

    return wrapper


# ─────────────────────────────────────────────
# Model persistence
# ─────────────────────────────────────────────

MODEL_DIR = "saved_models"


def save_model(
    model: Any,
    scaler: Optional[Any] = None,
    metadata: Optional[Dict] = None,
    model_name: str = "model"
) -> str:
    """
    Save a trained model to disk.

    Args:
        model: Trained sklearn model
        scaler: Fitted scaler (optional)
        metadata: Additional metadata to save
        model_name: Base name for the model file

    Returns:
        Path to saved model
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{timestamp}.pkl"
    filepath = os.path.join(MODEL_DIR, filename)

    save_data = {
        "model": model,
        "scaler": scaler,
        "metadata": metadata or {},
        "saved_at": timestamp,
    }

    with open(filepath, "wb") as f:
        pickle.dump(save_data, f)

    return filepath


def load_model(filepath: str) -> Dict[str, Any]:
    """
    Load a saved model from disk.

    Args:
        filepath: Path to .pkl file

    Returns:
        Dictionary with model, scaler, and metadata
    """
    with open(filepath, "rb") as f:
        return pickle.load(f)


def list_saved_models() -> List[str]:
    """List all saved model files."""
    if not os.path.exists(MODEL_DIR):
        return []

    files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]
    return sorted(files, reverse=True)  # Newest first


def delete_model(filepath: str) -> bool:
    """Delete a saved model file."""
    try:
        os.remove(filepath)
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────

def generate_html_report(
    dataset_name: str,
    problem_type: str,
    results: Dict[str, Dict[str, float]],
    best_model: str,
    timestamp: Optional[str] = None
) -> str:
    """
    Generate an HTML report of model comparison results.

    Args:
        dataset_name: Name of the dataset
        problem_type: 'Classification' or 'Regression'
        results: Model results dictionary
        best_model: Name of best model
        timestamp: Optional timestamp

    Returns:
        HTML string
    """
    timestamp = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build results table
    table_rows = ""
    for model_name, metrics in results.items():
        is_best = "style='background-color: #d4edda; font-weight: bold'" if model_name == best_model else ""
        table_rows += f"<tr {is_best}>"
        table_rows += f"<td>{model_name}</td>"
        for metric, value in metrics.items():
            table_rows += f"<td>{value}</td>"
        table_rows += "</tr>"

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ML Model Comparison Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #3498db; color: white; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .highlight {{ background-color: #d4edda; }}
            .meta {{ color: #7f8c8d; font-size: 0.9em; }}
        </style>
    </head>
    <body>
        <h1>📊 ML Model Comparison Report</h1>
        <p class="meta">Generated: {timestamp}</p>

        <h2>Dataset Information</h2>
        <p><strong>Dataset:</strong> {dataset_name}</p>
        <p><strong>Problem Type:</strong> {problem_type}</p>

        <h2>Model Comparison Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Model</th>
                    {"".join(f"<th>{m}</th>" for m in list(results.values())[0].keys())}
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>

        <h2>🏆 Best Model</h2>
        <p style="font-size: 1.2em; color: #27ae60;">
            <strong>{best_model}</strong>
        </p>

        <hr>
        <p class="meta">Generated by ML Algorithm Comparator</p>
    </body>
    </html>
    """

    return html


def generate_markdown_summary(
    results: Dict[str, Dict[str, float]],
    best_model: str,
    include_timestamp: bool = True
) -> str:
    """
    Generate a markdown summary of results.

    Args:
        results: Model results dictionary
        best_model: Best model name
        include_timestamp: Whether to include timestamp

    Returns:
        Markdown string
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") if include_timestamp else ""

    md = "## 📊 Model Comparison Results\n\n"

    # Get all metric names
    all_metrics = list(list(results.values())[0].keys())

    # Create table header
    md += "| Model | " + " | ".join(all_metrics) + " |\n"
    md += "|" + "|".join(["---"] * (len(all_metrics) + 1)) + "|\n"

    # Add rows
    for model_name, metrics in results.items():
        marker = "🏆 " if model_name == best_model else ""
        row = f"| {marker}{model_name} |"
        for metric in all_metrics:
            row += f" {metrics.get(metric, 'N/A')} |"
        md += row + "\n"

    md += f"\n**Best Model:** {best_model}\n"
    if include_timestamp:
        md += f"*Generated: {timestamp}*\n"

    return md


def export_results_csv(results: Dict[str, Dict[str, float]], filepath: str) -> None:
    """
    Export model results to CSV.

    Args:
        results: Model results dictionary
        filepath: Output file path
    """
    df = pd.DataFrame(results).T
    df.insert(0, "Model", df.index)
    df.to_csv(filepath, index=False)


# ─────────────────────────────────────────────
# Data utilities
# ─────────────────────────────────────────────

def format_bytes(size: int) -> str:
    """Format byte size to human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def format_number(n: int) -> str:
    """Format large numbers with K/M suffixes."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def get_dataset_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get summary statistics for a dataset.

    Args:
        df: DataFrame

    Returns:
        Dictionary of statistics
    """
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(df.isnull().sum().sum()),
        "missing_pct": round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2),
        "numeric_cols": len(df.select_dtypes(include=["number"]).columns),
        "categorical_cols": len(df.select_dtypes(include=["object", "category"]).columns),
        "duplicate_rows": df.duplicated().sum(),
    }


def sample_data_if_large(
    df: pd.DataFrame,
    max_rows: int = 5000,
    stratify_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Sample data if it exceeds max_rows.

    Args:
        df: Input DataFrame
        max_rows: Maximum rows to keep
        stratify_col: Column for stratified sampling

    Returns:
        Sampled DataFrame
    """
    if len(df) <= max_rows:
        return df

    if stratify_col and stratify_col in df.columns:
        # Manual stratified sampling using groupby
        frac = max_rows / len(df)
        sampled = df.groupby(stratify_col, group_keys=False).apply(
            lambda x: x.sample(max(1, int(len(x) * frac)), random_state=42)
        )
        # Trim or pad to exactly max_rows
        if len(sampled) > max_rows:
            sampled = sampled.sample(n=max_rows, random_state=42)
        return sampled.reset_index(drop=True)
    else:
        return df.sample(n=max_rows, random_state=42)


# ─────────────────────────────────────────────
# Streamlit UI helpers
# ─────────────────────────────────────────────

def show_metric_cards(metrics: Dict[str, float], cols: int = 4) -> None:
    """
    Display metrics as Streamlit metric cards.

    Args:
        metrics: Dictionary of metric name -> value
        cols: Number of columns
    """
    import streamlit as st

    chunk_size = (len(metrics) + cols - 1) // cols
    columns = st.columns(cols)

    for i, (name, value) in enumerate(metrics.items()):
        col_idx = i % cols
        with columns[col_idx]:
            delta = None
            if "R²" in name or "Accuracy" in name or "F1" in name:
                delta = f"{value:.2%}" if value < 1 else f"{value:.2f}"
            elif "RMSE" in name or "MAE" in name:
                delta = f"{value:.4f}"
            else:
                delta = f"{value:.4f}" if isinstance(value, float) else str(value)

            st.metric(label=name, value=delta, delta=None)


def create_collapsible_section(title: str, content: callable) -> None:
    """
    Create a collapsible section in Streamlit.

    Args:
        title: Section title
        content: Callable that renders content
    """
    with st.expander(title, expanded=False):
        content()


# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

def setup_logging(log_file: str = "ml_app.log") -> None:
    """
    Setup basic logging configuration.

    Args:
        log_file: Path to log file
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)
