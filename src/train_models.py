"""
Model Training — Event Impact & Response Intelligence Platform
===============================================================
Trains three LightGBM models per spec Section 6:
  Model A: priority_target (binary classification — High/Low)
  Model B: closure_target (binary classification)
  Model C: clearance_time_minutes (regression)

Uses RandomizedSearchCV (max 20 combos per model) for hyperparameter tuning.
Reports: precision, recall, F1 per class, ROC-AUC for classifiers; MAE, RMSE
and error breakdown by event_cause for regressor. Feature importance exported.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_fscore_support,
    mean_absolute_error, mean_squared_error
)

import lightgbm as lgb

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_TUNING_ITER = 20  # Max 20 parameter combinations per model per spec

# Hyperparameter search space (bounded per spec Section 6)
PARAM_SPACE = {
    "n_estimators": [100, 200, 300, 500, 800],
    "max_depth": [3, 5, 7, 10, 15, -1],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "num_leaves": [31, 50, 80, 127],
    "min_child_samples": [5, 10, 20, 50],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
}


# ============================================================================
# DATA PREPARATION
# ============================================================================
def load_data():
    """Load the feature-engineered datasets."""
    full_path = os.path.join(DATA_DIR, "features_full.csv")
    config_path = os.path.join(DATA_DIR, "feature_config.json")

    if not os.path.exists(full_path):
        print(f"❌ ERROR: {full_path} not found. Run data_pipeline.py first.")
        sys.exit(1)

    df = pd.read_csv(full_path, low_memory=False)
    with open(config_path, "r") as f:
        feature_config = json.load(f)

    print(f"Loaded {len(df)} rows from features_full.csv")
    return df, feature_config


def prepare_features(df, feature_cols, target_col, is_regression=False):
    """
    Prepare feature matrix X and target vector y.
    Encodes categoricals with LabelEncoder for LightGBM.
    Returns X, y, label_encoders, feature_names.
    """
    # Filter to rows where target is not null
    if is_regression:
        mask = df[target_col].notna()
        # VERIFY: no active rows
        if "status" in df.columns:
            active_in_subset = (df.loc[mask, "status"] == "active").sum()
            assert active_in_subset == 0, \
                f"CRITICAL: {active_in_subset} active rows in clearance-time subset!"
            print(f"  ✅ Verified: 0 active-status rows in regression subset")
    else:
        mask = df[target_col].notna()

    df_subset = df.loc[mask].copy()
    print(f"  Subset for {target_col}: {len(df_subset)} rows")

    # Only use feature columns that exist
    available_features = [f for f in feature_cols if f in df_subset.columns]

    X = df_subset[available_features].copy()
    y = df_subset[target_col].copy()

    # Encode categorical features
    label_encoders = {}
    categorical_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()

    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = X[col].astype(str).fillna("__MISSING__")
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    # Encode binary target for classifiers
    target_le = None
    if not is_regression:
        target_le = LabelEncoder()
        y = pd.Series(target_le.fit_transform(y.astype(str)), index=y.index)
        label_encoders["__target__"] = target_le
        print(f"  Target classes: {list(target_le.classes_)}")

    # Handle remaining NaN in numeric features
    X = X.fillna(0)

    # Convert boolean columns to int
    for col in X.select_dtypes(include=["bool"]).columns:
        X[col] = X[col].astype(int)

    return X, y, label_encoders, available_features


# ============================================================================
# MODEL TRAINING
# ============================================================================
def train_classifier(X, y, model_name, label_encoders):
    """
    Train a LightGBM classifier with RandomizedSearchCV.
    Uses class weighting (is_unbalance=True) per spec — no SMOTE.
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")

    # Stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train: {len(X_train)} rows, Test: {len(X_test)} rows")
    print(f"  Train class distribution: {pd.Series(y_train).value_counts().to_dict()}")
    print(f"  Test class distribution: {pd.Series(y_test).value_counts().to_dict()}")

    # LightGBM with class weighting
    base_model = lgb.LGBMClassifier(
        random_state=RANDOM_STATE,
        is_unbalance=True,  # Per spec: class weighting, no SMOTE
        verbose=-1,
        n_jobs=-1,
    )

    # RandomizedSearchCV with max 20 iterations
    search = RandomizedSearchCV(
        base_model,
        PARAM_SPACE,
        n_iter=MAX_TUNING_ITER,
        cv=5,
        scoring="roc_auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )

    print("  Running RandomizedSearchCV (20 combos, 5-fold CV)...")
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_params = search.best_params_
    print(f"  Best params: {best_params}")
    print(f"  Best CV ROC-AUC: {search.best_score_:.4f}")

    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    target_le = label_encoders.get("__target__")
    target_names = list(target_le.classes_) if target_le else None

    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    print(f"  ROC-AUC: {roc_auc:.4f}")

    # Feature importance
    importance = dict(zip(X.columns, best_model.feature_importances_))
    importance_sorted = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    print(f"\n  Top 10 Feature Importances:")
    for i, (feat, imp) in enumerate(list(importance_sorted.items())[:10]):
        print(f"    {i+1}. {feat}: {imp}")

    # Build results dict
    results = {
        "model_name": model_name,
        "algorithm": "LightGBM Classifier",
        "best_params": best_params,
        "search_space": {k: [str(v) for v in vals] for k, vals in PARAM_SPACE.items()},
        "tuning_iterations": MAX_TUNING_ITER,
        "best_cv_roc_auc": round(search.best_score_, 4),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "classification_report": report,
        "roc_auc": round(roc_auc, 4),
        "feature_importance": {k: int(v) for k, v in importance_sorted.items()},
    }

    return best_model, results, {"X_test": X_test, "y_test": y_test}


def train_regressor(X, y, df_full, feature_cols):
    """
    Train a LightGBM regressor for clearance_time_minutes.
    Uses RandomizedSearchCV (max 20 combos).
    Reports MAE, RMSE, and error breakdown by event_cause.
    """
    print(f"\n{'='*60}")
    print(f"Training Model C: Clearance Time Regressor")
    print(f"{'='*60}")

    # 80/20 random split (no stratification for regression)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"  Train: {len(X_train)} rows, Test: {len(X_test)} rows")
    print(f"  Target stats (train): mean={y_train.mean():.1f}, median={y_train.median():.1f}, "
          f"std={y_train.std():.1f} minutes")

    # LightGBM regressor
    base_model = lgb.LGBMRegressor(
        random_state=RANDOM_STATE,
        verbose=-1,
        n_jobs=-1,
    )

    # RandomizedSearchCV
    search = RandomizedSearchCV(
        base_model,
        PARAM_SPACE,
        n_iter=MAX_TUNING_ITER,
        cv=5,
        scoring="neg_mean_absolute_error",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )

    print("  Running RandomizedSearchCV (20 combos, 5-fold CV)...")
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_params = search.best_params_
    print(f"  Best params: {best_params}")
    print(f"  Best CV MAE: {-search.best_score_:.2f} minutes")

    # Evaluate on test set
    y_pred = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"\n  Test MAE: {mae:.2f} minutes")
    print(f"  Test RMSE: {rmse:.2f} minutes")

    # Error breakdown by event_cause
    # Get event_cause for test set rows
    clearance_mask = df_full["clearance_time_minutes"].notna()
    df_eligible = df_full.loc[clearance_mask].copy()
    test_indices = X_test.index
    event_causes_test = df_eligible.loc[test_indices, "event_cause"] if "event_cause" in df_eligible.columns else None

    error_by_cause = {}
    if event_causes_test is not None:
        residuals = np.abs(y_test.values - y_pred)
        cause_error_df = pd.DataFrame({
            "event_cause": event_causes_test.values,
            "abs_error": residuals,
            "actual": y_test.values,
            "predicted": y_pred
        })
        for cause, group in cause_error_df.groupby("event_cause"):
            error_by_cause[cause] = {
                "count": int(len(group)),
                "mae": round(float(group["abs_error"].mean()), 2),
                "mean_actual": round(float(group["actual"].mean()), 2),
                "mean_predicted": round(float(group["predicted"].mean()), 2),
            }
        print(f"\n  Error breakdown by event_cause:")
        for cause, metrics in sorted(error_by_cause.items(), key=lambda x: x[1]["count"], reverse=True):
            print(f"    {cause}: MAE={metrics['mae']:.1f} min (n={metrics['count']})")

    # Feature importance
    importance = dict(zip(X.columns, best_model.feature_importances_))
    importance_sorted = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    print(f"\n  Top 10 Feature Importances:")
    for i, (feat, imp) in enumerate(list(importance_sorted.items())[:10]):
        print(f"    {i+1}. {feat}: {imp}")

    results = {
        "model_name": "Model C: Clearance Time Regressor",
        "algorithm": "LightGBM Regressor",
        "best_params": best_params,
        "search_space": {k: [str(v) for v in vals] for k, vals in PARAM_SPACE.items()},
        "tuning_iterations": MAX_TUNING_ITER,
        "best_cv_mae": round(-search.best_score_, 4),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "test_mae_minutes": round(mae, 2),
        "test_rmse_minutes": round(rmse, 2),
        "error_by_event_cause": error_by_cause,
        "feature_importance": {k: int(v) for k, v in importance_sorted.items()},
    }

    return best_model, results, {"X_test": X_test, "y_test": y_test}


# ============================================================================
# MAIN
# ============================================================================
def run_training():
    """Execute the full model training pipeline."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    df, feature_config = load_data()
    clf_features = feature_config["classification_features"]
    reg_features = feature_config["regression_features"]

    all_results = {}

    # ---- MODEL A: Priority Classification ----
    X_a, y_a, le_a, feats_a = prepare_features(
        df, clf_features, "priority_target", is_regression=False
    )
    model_a, results_a, test_data_a = train_classifier(X_a, y_a, "Model A: Priority Classifier", le_a)
    all_results["model_a"] = results_a

    # Save model A
    joblib.dump(model_a, os.path.join(MODELS_DIR, "model_a_priority.pkl"))
    joblib.dump(le_a, os.path.join(MODELS_DIR, "model_a_encoders.pkl"))
    joblib.dump(feats_a, os.path.join(MODELS_DIR, "model_a_features.pkl"))

    # ---- MODEL B: Closure Classification ----
    X_b, y_b, le_b, feats_b = prepare_features(
        df, clf_features, "closure_target", is_regression=False
    )
    model_b, results_b, test_data_b = train_classifier(X_b, y_b, "Model B: Closure Classifier", le_b)
    all_results["model_b"] = results_b

    # Save model B
    joblib.dump(model_b, os.path.join(MODELS_DIR, "model_b_closure.pkl"))
    joblib.dump(le_b, os.path.join(MODELS_DIR, "model_b_encoders.pkl"))
    joblib.dump(feats_b, os.path.join(MODELS_DIR, "model_b_features.pkl"))

    # ---- MODEL C: Clearance Time Regression ----
    X_c, y_c, le_c, feats_c = prepare_features(
        df, reg_features, "clearance_time_minutes", is_regression=True
    )
    model_c, results_c, test_data_c = train_regressor(X_c, y_c, df, reg_features)
    all_results["model_c"] = results_c

    # Save model C
    joblib.dump(model_c, os.path.join(MODELS_DIR, "model_c_clearance.pkl"))
    joblib.dump(le_c, os.path.join(MODELS_DIR, "model_c_encoders.pkl"))
    joblib.dump(feats_c, os.path.join(MODELS_DIR, "model_c_features.pkl"))

    # Save all results
    results_path = os.path.join(OUTPUTS_DIR, "model_evaluation_report.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n✅ All evaluation results saved to: {results_path}")

    # Save split sizes summary
    split_summary = {
        "model_a": {"train": results_a["train_size"], "test": results_a["test_size"]},
        "model_b": {"train": results_b["train_size"], "test": results_b["test_size"]},
        "model_c": {"train": results_c["train_size"], "test": results_c["test_size"]},
    }
    split_path = os.path.join(OUTPUTS_DIR, "split_sizes.json")
    with open(split_path, "w") as f:
        json.dump(split_summary, f, indent=2)

    print(f"\n{'='*60}")
    print("✅ MODEL TRAINING COMPLETE")
    print(f"{'='*60}")

    return all_results


if __name__ == "__main__":
    run_training()
