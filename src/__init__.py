"""
ML Algorithm Comparator - Modular Package

A Streamlit-based ML model comparison tool with:
- Automatic data preprocessing
- Multiple supervised and unsupervised algorithms
- Hyperparameter tuning
- Advanced visualizations
- Model persistence and reporting
"""

from .preprocessing import (
    detect_problem_type,
    auto_detect_target,
    read_csv_safe,
    load_dataset,
    get_column_info,
    preprocess_data,
    get_numeric_columns,
    get_categorical_columns,
    DataPreprocessor,
)

from .models import (
    REGRESSION_MODELS,
    CLASSIFICATION_MODELS,
    PARAM_GRIDS,
    ModelResult,
    get_regression_metrics,
    get_classification_metrics,
    train_model,
    train_multiple_models,
    tune_hyperparameters,
    create_ensemble_classifier,
    create_ensemble_regressor,
    get_feature_importance,
    calculate_roc_auc,
    ModelTrainer,
)

from .visualizations import (
    plot_metric_comparison,
    plot_multi_metric_comparison,
    plot_confusion_matrix,
    plot_confusion_matrices,
    plot_roc_curve,
    plot_learning_curve,
    plot_validation_curve,
    plot_feature_importance,
    plot_elbow_curve,
    plot_silhouette_analysis,
    plot_scatter_clusters,
    plot_correlation_heatmap,
    plot_distribution,
    plot_pair_comparison,
    create_results_summary_table,
)

from .utils import (
    get_data_hash,
    save_model,
    load_model,
    list_saved_models,
    delete_model,
    generate_html_report,
    generate_markdown_summary,
    export_results_csv,
    format_bytes,
    format_number,
    get_dataset_stats,
    sample_data_if_large,
    show_metric_cards,
    setup_logging,
    get_logger,
)

__version__ = "2.0.0"
__author__ = "ML Algorithm Comparator"
