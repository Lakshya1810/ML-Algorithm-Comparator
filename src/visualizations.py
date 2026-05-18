"""
Visualization module for ML Algorithm Comparator.
Includes plotting functions for model evaluation and data exploration.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    silhouette_samples, silhouette_score
)
from sklearn.model_selection import learning_curve, validation_curve


# Set style defaults
sns.set_style("whitegrid")
plt.rcParams["figure.facecolor"] = "white"


def plot_metric_comparison(
    results: Dict[str, Dict[str, float]],
    metric: str,
    title: str,
    color: str = "#3498db",
    xlim: Tuple[float, float] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Create a horizontal bar chart comparing models on a single metric.

    Args:
        results: Dictionary of model_name -> metrics dict
        metric: Metric name to plot
        title: Chart title
        color: Bar color
        xlim: X-axis limits
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    model_names = list(results.keys())
    values = [results[name][metric] for name in model_names]

    bars = ax.barh(model_names, values, color=color)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(metric)

    if xlim:
        ax.set_xlim(xlim)

    # Add value labels
    for i, (name, val) in enumerate(zip(model_names, values)):
        ax.text(val, i, f" {val:.4f}", va="center", fontsize=10)

    plt.tight_layout()
    return fig


def plot_multi_metric_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: List[str],
    colors: List[str],
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """
    Create multiple bar charts for different metrics.

    Args:
        results: Model results dictionary
        metrics: List of metrics to plot
        colors: Color for each metric
        figsize: Figure size

    Returns:
        Matplotlib figure with subplots
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric, color in zip(axes, metrics, colors):
        model_names = list(results.keys())
        values = [results[name][metric] for name in model_names]

        ax.barh(model_names, values, color=color)
        ax.set_title(metric, fontsize=12, fontweight="bold")

        if metric in ["Accuracy", "Precision", "Recall", "F1 Score", "R² Score"]:
            ax.set_xlim(0, 1.05)

        for i, val in enumerate(values):
            ax.text(val, i, f" {val:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    figsize: Tuple[int, int] = (6, 5)
) -> plt.Figure:
    """
    Plot a confusion matrix heatmap.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional class names for axes
        title: Chart title
        cmap: Matplotlib colormap
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    cm = confusion_matrix(y_true, y_pred)
    x_labels = class_names if class_names is not None else "auto"
    y_labels = class_names if class_names is not None else "auto"
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=cmap,
        ax=ax, cbar=True,
        xticklabels=x_labels,
        yticklabels=y_labels
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_xlabel("Predicted", fontsize=12)

    plt.tight_layout()
    return fig


def plot_confusion_matrices(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (18, 5)
) -> plt.Figure:
    """
    Plot confusion matrices for multiple models side by side.

    Args:
        models: Dictionary of trained models
        X_test, y_test: Test data
        class_names: Optional class names
        figsize: Figure size

    Returns:
        Matplotlib figure with subplots
    """
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    if n_models == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        x_labels = class_names if class_names is not None else "auto"
        y_labels = class_names if class_names is not None else "auto"
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            ax=ax, cbar=False,
            xticklabels=x_labels,
            yticklabels=y_labels
        )

        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")

    plt.tight_layout()
    return fig


def plot_roc_curve(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    title: str = "ROC Curve",
    figsize: Tuple[int, int] = (6, 5)
) -> plt.Figure:
    """
    Plot ROC curve for a classifier.

    Args:
        model: Trained classifier
        X_test, y_test: Test data
        title: Chart title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    try:
        y_prob = model.predict_proba(X_test)

        if y_prob.shape[1] == 2:
            fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
            auc_score = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.4f})", linewidth=2)
        else:
            # Multi-class: use one-vs-rest for first class
            fpr, tpr, _ = roc_curve(y_test == 0, y_prob[:, 0])
            auc_score = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.4f})", linewidth=2)

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="lower right")

    except Exception as e:
        ax.text(0.5, 0.5, f"ROC not available\n{str(e)}", ha="center", va="center")
        ax.set_title(title)

    plt.tight_layout()
    return fig


def plot_learning_curve(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    title: str = "Learning Curve",
    cv: int = 5,
    scoring: str = "accuracy",
    train_sizes: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (8, 5)
) -> plt.Figure:
    """
    Plot learning curve showing model performance vs training size.

    Args:
        model: Scikit-learn model
        X, y: Data
        title: Chart title
        cv: Cross-validation folds
        scoring: Scoring metric
        train_sizes: Relative or absolute numbers of training examples
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    from sklearn.model_selection import learning_curve

    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)

    train_sizes_abs, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, train_sizes=train_sizes,
        scoring=scoring, n_jobs=-1, random_state=42
    )

    fig, ax = plt.subplots(figsize=figsize)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    ax.plot(train_sizes_abs, train_mean, "o-", color="blue", linewidth=2, label="Training score")
    ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")

    ax.plot(train_sizes_abs, test_mean, "o-", color="red", linewidth=2, label="Cross-validation score")
    ax.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std, alpha=0.1, color="red")

    ax.set_xlabel("Training Examples", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_validation_curve(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    param_name: str,
    param_range: List,
    title: str = "Validation Curve",
    cv: int = 5,
    scoring: str = "accuracy",
    figsize: Tuple[int, int] = (8, 5)
) -> plt.Figure:
    """
    Plot validation curve for a hyperparameter.

    Args:
        model: Scikit-learn model
        X, y: Data
        param_name: Parameter name to vary
        param_range: Values to try
        title: Chart title
        cv: Cross-validation folds
        scoring: Scoring metric
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    from sklearn.model_selection import validation_curve

    train_scores, test_scores = validation_curve(
        model, X, y, param_name=param_name,
        param_range=param_range, cv=cv,
        scoring=scoring, n_jobs=-1
    )

    fig, ax = plt.subplots(figsize=figsize)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    ax.plot(param_range, train_mean, "o-", color="blue", linewidth=2, label="Training score")
    ax.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")

    ax.plot(param_range, test_mean, "o-", color="red", linewidth=2, label="Cross-validation score")
    ax.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.1, color="red")

    ax.set_xlabel(param_name.replace("_", " ").title(), fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best")
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    title: str = "Feature Importance",
    top_n: int = 15,
    figsize: Tuple[int, int] = (8, 6),
    color: str = "#3498db"
) -> plt.Figure:
    """
    Plot feature importance from a trained model.

    Args:
        model: Trained model with feature_importances_ or coef_
        feature_names: List of feature names
        title: Chart title
        top_n: Number of top features to show
        figsize: Figure size
        color: Bar color

    Returns:
        Matplotlib figure
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
        if len(importances.shape) > 1:
            importances = importances.mean(axis=0)
    else:
        raise ValueError("Model does not have feature importance attributes")

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=figsize)

    ax.barh(importance_df["feature"], importance_df["importance"], color=color)
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.invert_yaxis()  # Highest at top

    plt.tight_layout()
    return fig


def plot_elbow_curve(
    inertias: List[float],
    k_range: range,
    title: str = "Elbow Method",
    figsize: Tuple[int, int] = (6, 4)
) -> plt.Figure:
    """
    Plot elbow curve for K-Means clustering.

    Args:
        inertias: List of inertia values
        k_range: Range of K values
        title: Chart title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(list(k_range), inertias, "bo-", linewidth=2, markersize=8)
    ax.set_xlabel("Number of Clusters (K)", fontsize=11)
    ax.set_ylabel("Inertia", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_silhouette_analysis(
    X: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    title: str = "Silhouette Analysis",
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot silhouette analysis for clustering evaluation.

    Args:
        X: Data (not used directly, kept for API consistency)
        labels: Cluster labels
        n_clusters: Number of clusters
        title: Chart title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import silhouette_samples

    silhouette_vals = silhouette_samples(X, labels)

    fig, ax = plt.subplots(figsize=figsize)

    y_lower = 10
    for i in range(n_clusters):
        cluster_vals = np.sort(silhouette_vals[labels == i])
        cluster_size = len(cluster_vals)

        y_upper = y_lower + cluster_size
        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        ax.barh(range(y_lower, y_upper), cluster_vals, height=1.0, color=color, alpha=0.7)

        y_lower = y_upper + 10

    ax.set_xlabel("Silhouette Coefficient", fontsize=11)
    ax.set_ylabel("Cluster", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim([0, y_upper])
    ax.axvline(x=silhouette_vals.mean(), color="red", linestyle="--", label="Average")
    ax.legend()

    plt.tight_layout()
    return fig


def plot_scatter_clusters(
    X: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str],
    title: str = "Cluster Visualization",
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Scatter plot of data colored by cluster labels.

    Args:
        X: Feature matrix (uses first 2 columns)
        labels: Cluster labels
        feature_names: Names of features
        title: Chart title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    scatter = ax.scatter(
        X[:, 0], X[:, 1],
        c=labels,
        cmap="Set2",
        alpha=0.7,
        edgecolors="k",
        linewidths=0.5
    )

    ax.set_xlabel(feature_names[0] if len(feature_names) > 0 else "Feature 1", fontsize=11)
    ax.set_ylabel(feature_names[1] if len(feature_names) > 1 else "Feature 2", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.colorbar(scatter, ax=ax, label="Cluster")
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    title: str = "Correlation Heatmap",
    figsize: Tuple[int, int] = (10, 8),
    annot: bool = True,
    cmap: str = "coolwarm"
) -> plt.Figure:
    """
    Plot correlation heatmap for numeric columns.

    Args:
        df: DataFrame
        title: Chart title
        figsize: Figure size
        annot: Whether to show values
        cmap: Colormap

    Returns:
        Matplotlib figure
    """
    num_df = df.select_dtypes(include=[np.number])

    if len(num_df.columns) < 2:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Need ≥ 2 numeric columns", ha="center", va="center")
        return fig

    corr = num_df.corr()

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr, annot=annot, cmap=cmap,
        ax=ax, fmt=".2f", linewidths=0.5,
        square=True, cbar=True
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    return fig


def plot_distribution(
    df: pd.DataFrame,
    column: str,
    title: Optional[str] = None,
    kind: str = "hist",
    figsize: Tuple[int, int] = (8, 5)
) -> plt.Figure:
    """
    Plot distribution of a column.

    Args:
        df: DataFrame
        column: Column to plot
        title: Chart title
        kind: 'hist' for histogram, 'kde' for density, 'box' for boxplot
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if kind == "hist":
        sns.histplot(data=df, x=column, bins=30, kde=True, ax=ax, color="#3498db")
    elif kind == "kde":
        sns.kdeplot(data=df, x=column, ax=ax, fill=True, color="#3498db")
    elif kind == "box":
        sns.boxplot(data=df, y=column, ax=ax, color="#3498db")

    ax.set_title(title or f"Distribution: {column}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_pair_comparison(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    plot_type: str = "scatter",
    hue_col: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot relationship between two columns.

    Args:
        df: DataFrame
        x_col, y_col: Columns to compare
        plot_type: 'scatter', 'line', 'bar', 'box'
        hue_col: Optional grouping column
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if plot_type == "scatter":
        if hue_col and hue_col in df.columns:
            sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax, alpha=0.7)
        else:
            sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax, alpha=0.7)

    elif plot_type == "line":
        if df[x_col].dtype in ["float64", "int64"]:
            df_sorted = df.sort_values(x_col)
            ax.plot(df_sorted[x_col], df_sorted[y_col], marker="o", markersize=4)
        else:
            ax.plot(df[x_col], df[y_col], marker="o")

    elif plot_type == "bar":
        if df[x_col].dtype == "object":
            grouped = df.groupby(x_col)[y_col].mean()
            grouped.plot(kind="bar", ax=ax, color=sns.color_palette("Set2", len(grouped)))
        else:
            ax.bar(range(min(50, len(df))), df[y_col].head(50), color="#3498db")
            ax.set_xlabel("Index")

    elif plot_type == "box":
        if df[x_col].dtype == "object":
            sns.boxplot(data=df, x=x_col, y=y_col, ax=ax, palette="Set2")
        else:
            sns.boxplot(data=df[[x_col, y_col]], ax=ax, palette="Set2")

    ax.set_title(f"{plot_type.title()}: {x_col} vs {y_col}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def create_results_summary_table(
    results: Dict[str, Dict[str, float]],
    highlight_best: bool = True,
    metric_order: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create a summary DataFrame of model results.

    Args:
        results: Model results dictionary
        highlight_best: Whether to format best values
        metric_order: Optional column order

    Returns:
        DataFrame with results
    """
    df = pd.DataFrame(results).T

    if metric_order:
        df = df[metric_order]

    if highlight_best:
        # Add column indicating if row has best value for any metric
        for col in df.columns:
            if "R²" in col or "Accuracy" in col or "F1" in col:
                df[f"Best_{col}"] = df[col] == df[col].max()
            elif "RMSE" in col or "MAE" in col or "MSE" in col:
                df[f"Best_{col}"] = df[col] == df[col].min()

    return df
