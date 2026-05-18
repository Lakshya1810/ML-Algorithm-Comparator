"""
📊 ML Algorithm Comparator Pro
A powerful Streamlit web application for testing, visualizing, and comparing ML models.

Features:
- Modular architecture with separate preprocessing, models, and visualization modules
- AutoML-style hyperparameter tuning (GridSearch/RandomSearch)
- Ensemble models (Voting & Stacking)
- Advanced visualizations (ROC, Learning Curves, Feature Importance)
- Model persistence and exportable reports
- Performance caching for faster execution
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
import time
from pathlib import Path

# Import directly from modules to avoid circular imports
from src.preprocessing import (
    detect_problem_type,
    auto_detect_target,
    read_csv_safe,
    preprocess_data,
    get_numeric_columns,
    get_categorical_columns,
    get_column_info
)
from src.models import (
    PARAM_GRIDS,
    tune_hyperparameters,
    create_ensemble_classifier,
    create_ensemble_regressor
)
from src.visualizations import (
    plot_multi_metric_comparison,
    plot_confusion_matrices,
    plot_elbow_curve,
    plot_scatter_clusters,
    plot_correlation_heatmap,
    plot_pair_comparison,
    plot_feature_importance
)
from src.utils import (
    save_model,
    list_saved_models,
    generate_html_report,
    format_number,
    get_dataset_stats
)

# sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.svm import SVR, SVC, LinearSVR, LinearSVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier
)
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    silhouette_score
)

# Suppress warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🚀 ML Algorithm Comparator Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    [data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.1);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
        padding: 10px;
    }
    .main-header {
        font-size: 2.5rem; 
        font-weight: bold;
        color: #3498db;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem; 
        opacity: 0.8;
        margin-top: -10px;
        margin-bottom: 20px;
    }
    .success-box {
        padding: 1rem; 
        border-radius: 8px; 
        background-color: rgba(40, 167, 69, 0.1); 
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }
    .info-box {
        padding: 1rem; 
        border-radius: 8px; 
        background-color: rgba(23, 162, 184, 0.1); 
        border-left: 5px solid #17a2b8;
        margin: 10px 0;
    }
    .stButton>button {
        border-radius: 5px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "df" not in st.session_state:
    st.session_state.df = None
if "target_col" not in st.session_state:
    st.session_state.target_col = None
if "problem_type" not in st.session_state:
    st.session_state.problem_type = None
if "feature_cols" not in st.session_state:
    st.session_state.feature_cols = None
if "le_dict" not in st.session_state:
    st.session_state.le_dict = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "sup_models" not in st.session_state:
    st.session_state.sup_models = None
if "sup_results" not in st.session_state:
    st.session_state.sup_results = None
if "df_pred" not in st.session_state:
    st.session_state.df_pred = None
if "km_labels" not in st.session_state:
    st.session_state.km_labels = None


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown('<p class="main-header">📊 ML Algorithm Comparator Pro</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Compare, tune, and visualize ML models with ease</p>', unsafe_allow_html=True)
st.markdown("---")


# ─────────────────────────────────────────────
# Sidebar: Data Loading
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Data Loading")

    data_source = st.radio(
        "Data source",
        ["Upload file", "Built-in dataset"],
        help="Choose between uploading your own file or using built-in sample datasets"
    )

    # Cache data loading
    @st.cache_resource
    def load_data_cached(source, file_type):
        if file_type == "excel":
            return pd.read_excel(source)
        else:
            return read_csv_safe(source)

    df = None

    if data_source == "Upload file":
        uploaded = st.file_uploader(
            "Upload CSV or Excel",
            type=["csv", "xlsx", "xls"],
            help="Supported formats: CSV, XLS, XLSX"
        )

        if uploaded is not None:
            try:
                file_type = "excel" if uploaded.name.endswith((".xlsx", ".xls")) else "csv"
                df = load_data_cached(uploaded, file_type)
                st.session_state.data_loaded = True
            except Exception as e:
                st.error(f"Error reading file: {e}")
    else:
        DATASET_DIR = Path(__file__).parent / "datasets"
        builtin_files = list(DATASET_DIR.glob("*.csv")) if DATASET_DIR.exists() else []

        if builtin_files:
            chosen = st.selectbox(
                "Select dataset",
                [f.name for f in builtin_files],
                help="Built-in sample datasets"
            )
            df = load_data_cached(str(DATASET_DIR / chosen), "csv")
            st.session_state.data_loaded = True
        else:
            st.warning("No built-in datasets found in /datasets folder.")

    # Show dataset info
    if df is not None:
        st.session_state.df = df
        stats = get_dataset_stats(df)

        with st.expander("📊 Dataset Statistics", expanded=False):
            st.metric("Rows", format_number(stats["rows"]))
            st.metric("Columns", stats["columns"])
            st.metric("Missing Values", stats["missing_values"])
            st.write(f"**Missing %:** {stats['missing_pct']}%")
            st.write(f"**Numeric columns:** {stats['numeric_cols']}")
            st.write(f"**Categorical columns:** {stats['categorical_cols']}")

    st.markdown("---")
    if st.button("🔄 Reset Application", use_container_width=True, help="Clear all data and results"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ─────────────────────────────────────────────
# Main Application Flow
# ─────────────────────────────────────────────
if st.session_state.data_loaded and st.session_state.df is not None:
    df = st.session_state.df

    # ─────────────────────────────────────
    # Data Preview Section
    # ─────────────────────────────────────
    st.header("📋 Dataset Preview")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", format_number(len(df)))
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())

    st.dataframe(df.head(20), use_container_width=True)

    with st.expander("🔍 Column Info & Statistics"):
        st.dataframe(get_column_info(df), use_container_width=True)
        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe(include="all").T, use_container_width=True)

    # ─────────────────────────────────────
    # Target Selection
    # ─────────────────────────────────────
    st.sidebar.header("🎯 Target Selection")

    default_target = auto_detect_target(df)
    target_col = st.sidebar.selectbox(
        "Select target column",
        df.columns.tolist(),
        index=df.columns.tolist().index(default_target) if default_target in df.columns else 0,
        help="The column you want to predict"
    )

    problem_type = detect_problem_type(df[target_col])
    st.sidebar.info(f"Detected: **{problem_type.upper()}**")

    st.session_state.target_col = target_col
    st.session_state.problem_type = problem_type

    # ─────────────────────────────────────
    # Model Configuration
    # ─────────────────────────────────────
    st.sidebar.header("⚙️ Model Configuration")

    test_size = st.sidebar.slider(
        "Test size (%)",
        10, 50, 20, 5,
        help="Percentage of data to use for testing"
    ) / 100

    cv_folds = st.sidebar.slider(
        "Cross-validation folds",
        2, 10, 5,
        help="Number of folds for cross-validation"
    )

    enable_tuning = st.sidebar.checkbox(
        "Enable Hyperparameter Tuning",
        value=False,
        help="Use GridSearchCV for automatic hyperparameter optimization"
    )

    use_ensemble = st.sidebar.checkbox(
        "Include Ensemble Models",
        value=True,
        help="Add Voting and Stacking ensemble models"
    )

    # Sampling for large datasets
    MAX_FAST = 5000
    if len(df) > MAX_FAST:
        st.sidebar.warning(f"⚠ Dataset has **{len(df):,}** rows. Consider sampling for faster training.")
        sample_n = st.sidebar.slider(
            "Max rows to use",
            min_value=1000,
            max_value=len(df),
            value=min(10000, len(df)),
            step=1000
        )
    else:
        sample_n = len(df)

    # ─────────────────────────────────────
    # Tabs Layout
    # ─────────────────────────────────────
    tabs = st.tabs([
        "🔬 Supervised",
        "🧩 Unsupervised",
        "🎯 Prediction",
        "📈 Visualizations",
        "💾 Models"
    ])

    # ═══════════════════════════════════════
    # TAB 1: Supervised Learning
    # ═══════════════════════════════════════
    with tabs[0]:
        st.header("🔬 Supervised Learning Comparison")
        st.markdown(f"**Target:** `{target_col}` | **Type:** `{problem_type}`")

        if st.button("▶ Run Model Comparison", type="primary", key="run_sup"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            start_time = time.time()

            # Preprocess data
            status_text.text("Preprocessing data...")
            X, y, feature_cols, le_dict, ptype, df_encoded = preprocess_data(df, target_col)

            # Sample if large
            if sample_n < len(X):
                from sklearn.utils import resample
                if ptype == "classification":
                    X, y = resample(X, y, n_samples=sample_n, stratify=y, random_state=42)
                else:
                    X, y = resample(X, y, n_samples=sample_n, random_state=42)
                st.info(f"Sampled {sample_n:,} rows from {len(df):,}")

            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            # Scale features
            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc = scaler.transform(X_test)

            # Store in session
            st.session_state.X_train = X_train_sc
            st.session_state.X_test = X_test_sc
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.feature_cols = feature_cols
            st.session_state.le_dict = le_dict
            st.session_state.scaler = scaler
            st.session_state.df_encoded = df_encoded
            st.session_state.problem_type = ptype

            # Select models based on problem type
            is_large = len(X_train) > MAX_FAST

            if ptype == "regression":
                models = {
                    "Linear Regression": LinearRegression(),
                    "Ridge": Ridge(),
                    "Decision Tree": DecisionTreeRegressor(random_state=42),
                    "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42),
                    "SVR": LinearSVR(max_iter=1000, random_state=42) if is_large else SVR(),
                }

                if use_ensemble:
                    models["Voting Ensemble"] = create_ensemble_regressor([
                        ("lr", LinearRegression()),
                        ("rf", RandomForestRegressor(n_estimators=30, random_state=42)),
                        ("gb", GradientBoostingRegressor(n_estimators=30, random_state=42))
                    ])
            else:
                models = {
                    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                    "Decision Tree": DecisionTreeClassifier(random_state=42),
                    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
                    "SVC": LinearSVC(max_iter=1000, random_state=42) if is_large else SVC(),
                    "Naive Bayes": GaussianNB(),
                }

                if use_ensemble:
                    models["Voting Ensemble"] = create_ensemble_classifier([
                        ("lr", LogisticRegression(max_iter=1000, random_state=42)),
                        ("rf", RandomForestClassifier(n_estimators=30, random_state=42)),
                        ("gnb", GaussianNB())
                    ])

            # Train models
            status_text.text("Training models...")
            results = {}
            trained_models = {}

            for idx, (name, model) in enumerate(models.items()):
                progress_bar.progress((idx + 1) / len(models), text=f"Training {name}...")

                if enable_tuning and name in PARAM_GRIDS and PARAM_GRIDS[name]:
                    status_text.text(f"Tuning {name}...")
                    best_model, best_params, _ = tune_hyperparameters(
                        model, X_train_sc, y_train,
                        PARAM_GRIDS[name], cv=3,
                        method="random" if is_large else "grid",
                        problem_type=ptype
                    )
                    model = best_model
                    status_text.text(f"Training {name} with best params...")

                model.fit(X_train_sc, y_train)
                y_pred = model.predict(X_test_sc)

                if ptype == "regression":
                    metrics = {
                        "R²": round(r2_score(y_test, y_pred), 4),
                        "MAE": round(mean_absolute_error(y_test, y_pred), 4),
                        "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
                    }
                else:
                    avg = "weighted" if len(np.unique(y_test)) > 2 else "binary"
                    metrics = {
                        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
                        "Precision": round(precision_score(y_test, y_pred, average=avg, zero_division=0), 4),
                        "Recall": round(recall_score(y_test, y_pred, average=avg, zero_division=0), 4),
                        "F1": round(f1_score(y_test, y_pred, average=avg, zero_division=0), 4),
                    }

                results[name] = metrics
                trained_models[name] = model

            training_time = round(time.time() - start_time, 2)
            progress_bar.empty()
            status_text.empty()

            # Display results
            st.success(f"✅ Training completed in {training_time}s")

            # Results table
            st.subheader("📊 Results")
            results_df = pd.DataFrame(results).T
            st.dataframe(results_df, use_container_width=True)

            # Metric comparison charts
            if ptype == "regression":
                metrics_to_plot = ["R²", "MAE", "RMSE"]
                colors = ["#2ecc71", "#e74c3c", "#3498db"]
            else:
                metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1"]
                colors = ["#2ecc71", "#9b59b6", "#e67e22", "#3498db"]

            fig = plot_multi_metric_comparison(results, metrics_to_plot, colors)
            st.pyplot(fig)

            # Best model
            if ptype == "regression":
                best_metric = "R²"
                best_model_name = results_df[best_metric].idxmax()
                st.success(f"🏆 Best: **{best_model_name}** ({best_metric} = {results_df.loc[best_model_name, best_metric]})")
            else:
                best_model_name = results_df["Accuracy"].idxmax()
                st.success(f"🏆 Best: **{best_model_name}** (Accuracy = {results_df.loc[best_model_name, 'Accuracy']})")

            # Confusion matrices for classification
            if ptype == "classification":
                st.subheader("Confusion Matrices")
                fig_cm = plot_confusion_matrices(trained_models, X_test_sc, y_test)
                st.pyplot(fig_cm)

            # Store results
            st.session_state.sup_results = results
            st.session_state.sup_models = trained_models

    # ═══════════════════════════════════════
    # TAB 2: Unsupervised Learning
    # ═══════════════════════════════════════
    with tabs[1]:
        st.header("🧩 Unsupervised Learning - Clustering")

        num_cols = get_numeric_columns(df)
        cat_cols = get_categorical_columns(df)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Numeric columns", len(num_cols))
        with col2:
            st.metric("Categorical columns", len(cat_cols))

        k_clusters = st.slider("Number of clusters (K)", 2, 10, 3)

        if st.button("▶ Run Clustering", key="run_unsup"):
            unsup_results = {}

            # K-Means
            if len(num_cols) >= 2:
                st.subheader("K-Means Clustering")
                X_num = df[num_cols].fillna(df[num_cols].median())
                scaler_km = StandardScaler()
                X_scaled = scaler_km.fit_transform(X_num)

                kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
                km_labels = kmeans.fit_predict(X_scaled)
                sil = silhouette_score(X_scaled, km_labels)

                unsup_results["K-Means"] = {
                    "Silhouette Score": round(sil, 4),
                    "Inertia": round(kmeans.inertia_, 2)
                }

                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Silhouette Score", round(sil, 4))
                with col_b:
                    st.metric("Inertia", round(kmeans.inertia_, 2))

                # Visualization
                fig_scatter = plot_scatter_clusters(X_scaled, km_labels, num_cols, "K-Means Clusters")
                st.pyplot(fig_scatter)

                # Elbow plot
                inertias = []
                K_range = range(2, min(11, len(X_scaled)))
                for k in K_range:
                    km_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
                    km_temp.fit(X_scaled)
                    inertias.append(km_temp.inertia_)

                fig_elbow = plot_elbow_curve(inertias, K_range)
                st.pyplot(fig_elbow)

                st.session_state.km_labels = km_labels

            # K-Modes
            if len(cat_cols) >= 1:
                st.subheader("K-Modes Clustering")
                try:
                    from kmodes.kmodes import KModes

                    X_cat = df[cat_cols].fillna("Unknown")
                    kmode = KModes(n_clusters=k_clusters, init="Huang", n_init=5, random_state=42)
                    kmode_labels = kmode.fit_predict(X_cat)

                    unsup_results["K-Modes"] = {
                        "Cost": round(kmode.cost_, 2)
                    }

                    st.metric("Cost", round(kmode.cost_, 2))
                    st.dataframe(pd.DataFrame(kmode.cluster_centroids_, columns=cat_cols))

                    st.session_state.kmode_labels = kmode_labels
                except ImportError:
                    st.error("Install kmodes: `pip install kmodes`")

            if unsup_results:
                st.json(unsup_results)

    # ═══════════════════════════════════════
    # TAB 3: Prediction
    # ═══════════════════════════════════════
    with tabs[2]:
        st.header("🎯 Prediction")

        if st.session_state.sup_models is None:
            st.warning("⚠ Run Supervised Comparison first to train models.")
        else:
            models = st.session_state.sup_models

            chosen_model = st.selectbox("Select model", list(models.keys()), key="prediction_select_model")
            model = models[chosen_model]

            # Predict on dataset
            if st.button("Predict on Full Dataset"):
                try:
                    # Re-encode the full dataset the same way training did
                    df_full_work = df.copy()
                    for col, le in st.session_state.le_dict.items():
                        if col in df_full_work.columns:
                            df_full_work[col] = df_full_work[col].astype(str)
                            known = set(le.classes_)
                            df_full_work[col] = df_full_work[col].apply(
                                lambda x: x if x in known else le.classes_[0]
                            )
                            df_full_work[col] = le.transform(df_full_work[col])
                    df_full_work = df_full_work.apply(pd.to_numeric, errors="coerce").fillna(0)
                    feature_cols = st.session_state.feature_cols
                    X_full = df_full_work[feature_cols].values
                    X_full_sc = st.session_state.scaler.transform(X_full)
                    predictions = model.predict(X_full_sc)

                    # Reverse-map predictions back to original labels for classification
                    if target_col in st.session_state.le_dict and st.session_state.problem_type == "classification":
                        le_target = st.session_state.le_dict[target_col]
                        predictions = le_target.inverse_transform(predictions.astype(int))

                    pred_col = f"Predicted_{target_col}"
                    df_pred = df.copy()
                    df_pred[pred_col] = predictions

                    st.dataframe(df_pred.head(20), use_container_width=True)
                    st.session_state.df_pred = df_pred
                except Exception as e:
                    st.error(f"Prediction error: {e}")

            # Predict on new values
            if st.session_state.feature_cols:
                st.subheader("Predict on New Values")
                input_cols = st.columns(min(4, len(st.session_state.feature_cols)))
                new_vals = {}

                for i, feat in enumerate(st.session_state.feature_cols):
                    with input_cols[i % 4]:
                        if feat in st.session_state.le_dict:
                            options = list(st.session_state.le_dict[feat].classes_)
                            new_vals[feat] = st.selectbox(feat, options, key=f"new_{feat}")
                        else:
                            try:
                                default = float(df[feat].median()) if feat in df.columns else 0.0
                            except (TypeError, ValueError):
                                default = 0.0
                            new_vals[feat] = st.number_input(feat, value=default, key=f"num_{feat}")

                if st.button("Predict"):
                    row = []
                    for feat in st.session_state.feature_cols:
                        if feat in st.session_state.le_dict:
                            row.append(st.session_state.le_dict[feat].transform([str(new_vals[feat])])[0])
                        else:
                            row.append(float(new_vals[feat]))

                    row_sc = st.session_state.scaler.transform([row])
                    pred = model.predict(row_sc)

                    # Display result appropriately
                    if st.session_state.problem_type == "classification":
                        pred_val = pred[0]
                        if target_col in st.session_state.le_dict:
                            pred_val = st.session_state.le_dict[target_col].inverse_transform([int(pred_val)])[0]
                        st.success(f"Prediction: **{pred_val}**")
                    else:
                        st.success(f"Prediction: **{pred[0]:.4f}**")

            # Download results
            if st.session_state.df_pred is not None:
                csv = st.session_state.df_pred.to_csv(index=False)
                st.download_button(
                    label="⬇ Download Results CSV",
                    data=csv,
                    file_name="ml_predictions.csv",
                    mime="text/csv"
                )

    # ═══════════════════════════════════════
    # TAB 4: Visualizations
    # ═══════════════════════════════════════
    with tabs[3]:
        st.header("📈 Data Visualizations")

        viz_col1, viz_col2 = st.columns(2)
        with viz_col1:
            x_col = st.selectbox("X-axis", df.columns.tolist(), key="viz_x")
        with viz_col2:
            y_col = st.selectbox("Y-axis", df.columns.tolist(), key="viz_y")

        chart_type = st.selectbox(
            "Chart type",
            ["Scatter", "Line", "Bar", "Box", "Histogram", "Correlation Heatmap"],
            key="viz_chart_type"
        )

        if st.button("📊 Generate Chart"):
            if chart_type == "Correlation Heatmap":
                fig = plot_correlation_heatmap(df)
            else:
                fig = plot_pair_comparison(df, x_col, y_col, chart_type.lower())
            st.pyplot(fig)

        # Feature importance (if models trained)
        if st.session_state.sup_models and st.session_state.feature_cols:
            st.subheader("Feature Importance")
            model_name = st.selectbox(
                "Select model",
                list(st.session_state.sup_models.keys()),
                key="viz_select_model"
            )
            model = st.session_state.sup_models[model_name]

            if hasattr(model, "feature_importances_") or hasattr(model, "coef_"):
                fig_fi = plot_feature_importance(model, st.session_state.feature_cols)
                st.pyplot(fig_fi)

    # ═══════════════════════════════════════
    # TAB 5: Model Management
    # ═══════════════════════════════════════
    with tabs[4]:
        st.header("💾 Model Management")

        # Save model
        if st.session_state.sup_models:
            st.subheader("Save Trained Model")

            model_to_save = st.selectbox(
                "Select model to save",
                list(st.session_state.sup_models.keys()),
                key="save_select_model"
            )

            if st.button("💾 Save Model"):
                model = st.session_state.sup_models[model_to_save]
                filepath = save_model(
                    model,
                    st.session_state.scaler,
                    metadata={
                        "target": st.session_state.target_col,
                        "features": st.session_state.feature_cols,
                        "problem_type": st.session_state.problem_type
                    },
                    model_name=model_to_save.replace(" ", "_")
                )
                st.success(f"Model saved to: {filepath}")

        # List saved models
        st.subheader("Saved Models")
        saved = list_saved_models()
        if saved:
            for model_file in saved:
                st.write(f"📦 {model_file}")
        else:
            st.info("No saved models yet.")

        # Export report
        if st.session_state.sup_results:
            st.subheader("Export Report")

            results = st.session_state.sup_results
            best = pd.DataFrame(results).T
            if "Accuracy" in best.columns:
                best_model = best["Accuracy"].idxmax()
            elif "R²" in best.columns:
                best_model = best["R²"].idxmax()
            else:
                best_model = list(results.keys())[0]

            if st.button("Generate HTML Report"):
                html_report = generate_html_report(
                    dataset_name="Custom Dataset",
                    problem_type=st.session_state.problem_type,
                    results=results,
                    best_model=best_model
                )
                st.download_button(
                    label="⬇ Download HTML Report",
                    data=html_report,
                    file_name="ml_report.html",
                    mime="text/html"
                )


else:
    st.info("👈 Load a dataset from the sidebar to get started.")


# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #7f8c8d;">
        Built with Streamlit • ML Algorithm Comparator Pro v2.0
    </div>
    """,
    unsafe_allow_html=True
)
