"""
Data preprocessing module for ML Algorithm Comparator.
Handles data loading, cleaning, encoding, and feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, Dict, Any, Optional, List
import io


def detect_problem_type(series: pd.Series) -> str:
    """
    Determine if the target variable suggests classification or regression.

    Args:
        series: The target column as a pandas Series

    Returns:
        'classification' or 'regression'
    """
    if series.dtype == "object" or series.dtype.name == "category":
        return "classification"

    nunique = series.nunique()
    if nunique <= 20 and nunique / len(series) < 0.05:
        return "classification"

    return "regression"


def auto_detect_target(df: pd.DataFrame) -> str:
    """
    Pick a sensible default target column (last column by default).

    Args:
        df: Input DataFrame

    Returns:
        Name of the target column
    """
    return df.columns[-1]


def read_csv_safe(source: io.BytesIO) -> pd.DataFrame:
    """
    Try multiple encodings to read a CSV file successfully.

    Args:
        source: File-like object containing CSV data

    Returns:
        Loaded DataFrame
    """
    encodings = ["utf-8", "latin-1", "iso-8859-1", "cp1252", "utf-16"]

    for enc in encodings:
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            return pd.read_csv(source, encoding=enc)
        except (UnicodeDecodeError, Exception):
            continue

    # Last resort: read with latin-1 (accepts all byte values)
    if hasattr(source, "seek"):
        source.seek(0)
    return pd.read_csv(source, encoding="latin-1", on_bad_lines="skip")


def load_dataset(source: str, file_type: str = "auto") -> pd.DataFrame:
    """
    Load a dataset from file path or uploaded file.

    Args:
        source: Path to file or file-like object
        file_type: 'csv', 'excel', or 'auto' for auto-detection

    Returns:
        Loaded DataFrame
    """
    if file_type == "auto":
        if source.endswith(".csv"):
            file_type = "csv"
        elif source.endswith((".xlsx", ".xls")):
            file_type = "excel"

    if file_type == "excel":
        return pd.read_excel(source)
    else:
        return read_csv_safe(source)


def get_column_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get detailed information about all columns in a DataFrame.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with column statistics
    """
    return pd.DataFrame({
        "Column": df.columns,
        "Type": df.dtypes.astype(str).values,
        "Non-Null": df.notnull().sum().values,
        "Unique": df.nunique().values,
        "Nulls": df.isnull().sum().values,
        "Null_%": (df.isnull().sum() / len(df) * 100).round(2).values
    })


def preprocess_data(
    df: pd.DataFrame,
    target_col: str,
    impute_strategy: str = "auto",
    handle_outliers: bool = False
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, LabelEncoder], str, pd.DataFrame]:
    """
    Preprocess the dataset for machine learning.

    Args:
        df: Input DataFrame
        target_col: Name of the target column
        impute_strategy: 'auto', 'mean', 'median', 'mode', or 'constant'
        handle_outliers: Whether to handle outliers using IQR method

    Returns:
        Tuple of (X, y, feature_names, label_encoders, problem_type, processed_df)
    """
    df_work = df.copy()

    # Drop columns that are entirely NaN
    df_work.dropna(axis=1, how="all", inplace=True)

    if target_col not in df_work.columns:
        raise ValueError(f"Target column '{target_col}' was dropped because it contained only NaN values.")

    # Drop rows where target is NaN
    df_work.dropna(subset=[target_col], inplace=True)

    # Fill missing values
    for col in df_work.columns:
        if not pd.api.types.is_numeric_dtype(df_work[col]):
            # Categorical/String: fill with mode, fallback to "Unknown"
            mode_vals = df_work[col].mode()
            fill_val = mode_vals.iloc[0] if len(mode_vals) > 0 else "Unknown"
            df_work[col] = df_work[col].fillna(fill_val)
        else:
            # Numeric: replace inf with NaN, then fill with median/mean
            df_work[col] = df_work[col].replace([np.inf, -np.inf], np.nan)

            if impute_strategy == "mean":
                fill_val = df_work[col].mean()
            elif impute_strategy == "median":
                fill_val = df_work[col].median()
            else:  # auto or default
                fill_val = df_work[col].median()
                if pd.isna(fill_val):
                    fill_val = 0

            df_work[col] = df_work[col].fillna(fill_val)

    # Handle outliers using IQR method if requested
    if handle_outliers:
        for col in df_work.columns:
            if col != target_col and df_work[col].dtype != "object":
                q1 = df_work[col].quantile(0.25)
                q3 = df_work[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                df_work[col] = df_work[col].clip(lower=lower_bound, upper=upper_bound)

    problem_type = detect_problem_type(df_work[target_col])

    label_encoders = {}
    for col in df_work.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        df_work[col] = le.fit_transform(df_work[col].astype(str))
        label_encoders[col] = le

    # Convert everything to float and do a final NaN/inf safety check
    df_work = df_work.apply(pd.to_numeric, errors="coerce")
    df_work = df_work.replace([np.inf, -np.inf], np.nan)
    df_work = df_work.fillna(0)

    y = df_work[target_col].values
    feature_cols = [c for c in df_work.columns if c != target_col]
    X = df_work[feature_cols].values

    return X, y, feature_cols, label_encoders, problem_type, df_work


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Get all numeric column names."""
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    """Get all categorical column names."""
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


class DataPreprocessor:
    """
    Stateful preprocessor for consistent data transformation.
    Useful for applying same transformations to train/test/new data.
    """

    def __init__(self, target_col: str, impute_strategy: str = "auto"):
        self.target_col = target_col
        self.impute_strategy = impute_strategy
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_cols: List[str] = []
        self.problem_type: str = ""
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        """
        Fit the preprocessor on a DataFrame.

        Args:
            df: Training DataFrame

        Returns:
            Self for method chaining
        """
        _, _, self.feature_cols, self.label_encoders, self.problem_type, _ = preprocess_data(
            df, self.target_col, self.impute_strategy
        )
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform a DataFrame using fitted parameters.

        Args:
            df: DataFrame to transform

        Returns:
            Tuple of (X, y) as numpy arrays
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")

        df_work = df.copy()

        # Apply same encodings as fitted
        for col in df_work.select_dtypes(include=["object", "category"]).columns:
            if col in self.label_encoders:
                # Handle unseen categories
                df_work[col] = df_work[col].astype(str)
                known_classes = set(self.label_encoders[col].classes_)
                unknown_mask = ~df_work[col].isin(known_classes)
                if unknown_mask.any():
                    df_work.loc[unknown_mask, col] = self.label_encoders[col].classes_[0]
                df_work[col] = self.label_encoders[col].transform(df_work[col])
            else:
                le = LabelEncoder()
                df_work[col] = le.fit_transform(df_work[col].astype(str))

        df_work = df_work.apply(pd.to_numeric, errors="coerce").fillna(0)

        X = df_work[self.feature_cols].values
        y = df_work[self.target_col].values if self.target_col in df_work.columns else None

        return X, y

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)
