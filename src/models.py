"""
Machine Learning models module for ML Algorithm Comparator.
Includes model definitions, training, evaluation, and hyperparameter tuning.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, field

from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR, SVC, LinearSVR, LinearSVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    VotingRegressor, VotingClassifier, StackingClassifier, StackingRegressor
)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, silhouette_score,
    roc_curve, auc, roc_auc_score
)


# Default model configurations
REGRESSION_MODELS = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Elastic Net": ElasticNet(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "SVR (Linear)": LinearSVR(max_iter=5000, random_state=42),
    "SVR (RBF)": SVR(kernel="rbf"),
    "K-Neighbors": KNeighborsRegressor(),
    "MLP": MLPRegressor(max_iter=1000, random_state=42),
}

CLASSIFICATION_MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Ridge Classifier": LogisticRegression(penalty="l2", solver="lbfgs", max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "SVC (Linear)": LinearSVC(max_iter=5000, random_state=42, dual="auto"),
    "SVC (RBF)": SVC(kernel="rbf", random_state=42),
    "K-Neighbors": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "MLP": MLPClassifier(max_iter=1000, random_state=42),
}

# Hyperparameter grids for tuning
PARAM_GRIDS = {
    "Linear Regression": {},
    "Ridge Regression": {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "Lasso Regression": {"alpha": [0.01, 0.1, 1.0, 10.0]},
    "Elastic Net": {"alpha": [0.01, 0.1, 1.0], "l1_ratio": [0.2, 0.5, 0.8]},
    "Decision Tree": {
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    },
    "Random Forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    },
    "Gradient Boosting": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7]
    },
    "SVR (Linear)": {"C": [0.1, 1.0, 10.0], "epsilon": [0.1, 0.2, 0.5]},
    "SVR (RBF)": {"C": [0.1, 1.0, 10.0], "gamma": ["scale", "auto"], "epsilon": [0.1, 0.2]},
    "K-Neighbors": {"n_neighbors": [3, 5, 7, 10], "weights": ["uniform", "distance"]},
    "Logistic Regression": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "penalty": ["l2", None],
        "solver": ["lbfgs", "newton-cg"]
    },
    "SVC (Linear)": {"C": [0.1, 1.0, 10.0]},
    "SVC (RBF)": {"C": [0.1, 1.0, 10.0], "gamma": ["scale", "auto"]},
    "Naive Bayes": {},
    "MLP": {
        "hidden_layer_sizes": [(50,), (100,), (50, 25)],
        "alpha": [0.0001, 0.001, 0.01]
    },
}


@dataclass
class ModelResult:
    """Container for model evaluation results."""
    name: str
    metrics: Dict[str, float]
    model: Any
    cv_scores: Optional[np.ndarray] = None
    training_time: float = 0.0
    best_params: Dict[str, Any] = field(default_factory=dict)
    is_tuned: bool = False


def get_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate regression metrics."""
    return {
        "R² Score": round(r2_score(y_true, y_pred), 4),
        "MAE": round(mean_absolute_error(y_true, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "MSE": round(mean_squared_error(y_true, y_pred), 4),
    }


def get_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, average: str = "weighted"
) -> Dict[str, float]:
    """Calculate classification metrics."""
    zero_div = 0
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, average=average, zero_division=zero_div), 4),
        "Recall": round(recall_score(y_true, y_pred, average=average, zero_division=zero_div), 4),
        "F1 Score": round(f1_score(y_true, y_pred, average=average, zero_division=zero_div), 4),
    }


def train_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cv_folds: int = 5,
    problem_type: str = "regression",
    return_predictions: bool = True
) -> ModelResult:
    """
    Train a single model and evaluate it.

    Args:
        model: Scikit-learn model instance
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        cv_folds: Number of cross-validation folds
        problem_type: 'regression' or 'classification'
        return_predictions: Whether to return predictions

    Returns:
        ModelResult dataclass with metrics
    """
    import time
    start_time = time.time()

    model.fit(X_train, y_train)
    training_time = round(time.time() - start_time, 2)

    y_pred = model.predict(X_test) if return_predictions else None

    # Cross-validation
    scoring = "r2" if problem_type == "regression" else "accuracy"
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring=scoring)

    # Calculate metrics
    if problem_type == "regression":
        metrics = get_regression_metrics(y_test, y_pred) if return_predictions else {}
    else:
        n_classes = len(np.unique(y_test))
        avg = "weighted" if n_classes > 2 else "binary"
        metrics = get_classification_metrics(y_test, y_pred, avg) if return_predictions else {}

    metrics["CV Mean"] = round(cv_scores.mean(), 4)
    metrics["CV Std"] = round(cv_scores.std(), 4)

    return ModelResult(
        name=model.__class__.__name__,
        metrics=metrics,
        model=model,
        cv_scores=cv_scores,
        training_time=training_time
    )


def train_multiple_models(
    models: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cv_folds: int = 5,
    problem_type: str = "regression",
    progress_callback: Optional[callable] = None
) -> Dict[str, ModelResult]:
    """
    Train multiple models and return results.

    Args:
        models: Dictionary of model name -> model instance
        X_train, y_train, X_test, y_test: Data splits
        cv_folds: CV folds
        problem_type: Task type
        progress_callback: Optional callback for progress updates

    Returns:
        Dictionary of model_name -> ModelResult
    """
    results = {}
    total = len(models)

    for idx, (name, model) in enumerate(models.items()):
        if progress_callback:
            progress_callback(idx + 1, total, name)

        result = train_model(
            model, X_train, y_train, X_test, y_test,
            cv_folds=cv_folds, problem_type=problem_type
        )
        result.name = name  # Use display name
        results[name] = result

    return results


def tune_hyperparameters(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    param_grid: Dict[str, List],
    cv: int = 5,
    method: str = "grid",
    n_iter: int = 10,
    problem_type: str = "regression",
    verbose: int = 1
) -> Tuple[Any, Dict[str, Any], float]:
    """
    Perform hyperparameter tuning.

    Args:
        model: Base model to tune
        X, y: Training data
        param_grid: Parameter grid for tuning
        cv: Cross-validation folds
        method: 'grid' or 'random'
        n_iter: Number of iterations for random search
        problem_type: Task type
        verbose: Verbosity level

    Returns:
        Tuple of (best_model, best_params, best_score)
    """
    scoring = "r2" if problem_type == "regression" else "accuracy"

    if method == "grid":
        search = GridSearchCV(
            model, param_grid, cv=cv, scoring=scoring,
            n_jobs=-1, verbose=verbose
        )
    else:
        search = RandomizedSearchCV(
            model, param_grid, n_iter=n_iter, cv=cv,
            scoring=scoring, n_jobs=-1, verbose=verbose,
            random_state=42
        )

    search.fit(X, y)

    return search.best_estimator_, search.best_params_, search.best_score_


def create_ensemble_classifier(
    estimators: List[Tuple[str, Any]],
    ensemble_type: str = "voting",
    voting: str = "hard"
) -> Union[VotingClassifier, StackingClassifier]:
    """
    Create an ensemble classifier.

    Args:
        estimators: List of (name, estimator) tuples
        ensemble_type: 'voting' or 'stacking'
        voting: 'hard' or 'soft' for voting classifier

    Returns:
        Ensemble model instance
    """
    if ensemble_type == "voting":
        return VotingClassifier(estimators=estimators, voting=voting)
    else:
        # Stacking with logistic regression as final estimator
        final_estimator = LogisticRegression(max_iter=1000)
        return StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=5
        )


def create_ensemble_regressor(
    estimators: List[Tuple[str, Any]],
    ensemble_type: str = "voting"
) -> Union[VotingRegressor, StackingRegressor]:
    """
    Create an ensemble regressor.

    Args:
        estimators: List of (name, estimator) tuples
        ensemble_type: 'voting' or 'stacking'

    Returns:
        Ensemble model instance
    """
    if ensemble_type == "voting":
        return VotingRegressor(estimators=estimators)
    else:
        final_estimator = LinearRegression()
        return StackingRegressor(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=5
        )


def get_feature_importance(model: Any, feature_names: List[str]) -> pd.DataFrame:
    """
    Extract feature importance from a trained model.

    Args:
        model: Trained model
        feature_names: List of feature column names

    Returns:
        DataFrame with feature names and importance scores
    """
    import pandas as pd

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
        if len(importances.shape) > 1 and importances.shape[0] > 1:
            importances = importances.mean(axis=0)
    else:
        return None

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    return importance_df


def calculate_roc_auc(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int
) -> Optional[Dict[str, Any]]:
    """
    Calculate ROC curve and AUC score.

    Args:
        model: Trained classifier
        X_test: Test features
        y_test: Test target
        n_classes: Number of classes

    Returns:
        Dictionary with fpr, tpr, auc, or None for non-classifiers
    """
    if n_classes < 2:
        return None

    try:
        y_prob = model.predict_proba(X_test)

        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
            auc_score = roc_auc_score(y_test, y_prob[:, 1])
        else:
            # Multi-class: compute one-vs-rest
            from sklearn.preprocessing import label_binarize
            y_test_bin = label_binarize(y_test, classes=range(n_classes))
            fpr, tpr, _ = roc_curve(y_test_bin[:, 0], y_prob[:, 0])
            auc_score = roc_auc_score(y_test_bin, y_prob, average="macro")

        return {"fpr": fpr, "tpr": tpr, "auc": round(auc_score, 4)}
    except Exception:
        return None


class ModelTrainer:
    """
    Stateful model trainer with consistent preprocessing.
    """

    def __init__(self, problem_type: str = "regression"):
        self.problem_type = problem_type
        self.trained_models: Dict[str, Any] = {}
        self.results: Dict[str, ModelResult] = {}
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []

    def fit_scaler(self, X: np.ndarray) -> np.ndarray:
        """Fit and return scaled features."""
        self.scaler = StandardScaler()
        return self.scaler.fit_transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using fitted scaler."""
        if self.scaler is None:
            raise RuntimeError("Scaler must be fitted first")
        return self.scaler.transform(X)

    def train(
        self,
        name: str,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        cv_folds: int = 5
    ) -> ModelResult:
        """Train a single model."""
        result = train_model(
            model, X_train, y_train, X_test, y_test,
            cv_folds=cv_folds, problem_type=self.problem_type
        )
        result.name = name
        self.trained_models[name] = result.model
        self.results[name] = result
        return result

    def train_all(
        self,
        models: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        cv_folds: int = 5
    ) -> Dict[str, ModelResult]:
        """Train multiple models."""
        self.results = train_multiple_models(
            models, X_train, y_train, X_test, y_test,
            cv_folds=cv_folds, problem_type=self.problem_type
        )
        self.trained_models = {
            name: result.model for name, result in self.results.items()
        }
        return self.results

    def get_best_model(self, metric: str = "Accuracy") -> Tuple[str, Any, float]:
        """
        Get the best model based on a metric.

        Args:
            metric: Metric name to optimize

        Returns:
            Tuple of (name, model, score)
        """
        if not self.results:
            raise ValueError("No models trained yet")

        best_name = max(
            self.results.keys(),
            key=lambda k: self.results[k].metrics.get(metric, 0)
        )
        return best_name, self.trained_models[best_name], self.results[best_name].metrics[metric]
