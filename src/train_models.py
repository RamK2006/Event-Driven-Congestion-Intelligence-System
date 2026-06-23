"""Leakage-safe Bayesian tuning and training for all production models."""

import argparse
import json
import os
import warnings

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", ".matplotlib"),
)

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import optuna
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from model_bundle import ModelBundle

if os.name == "nt":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
DIAGNOSTICS_DIR = os.path.join(OUTPUTS_DIR, "model_diagnostics")
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "features_full.csv"), low_memory=False)
    with open(os.path.join(DATA_DIR, "feature_config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    return df, config


def prepare_features(df, feature_cols, target_col, is_regression=False):
    mask = df[target_col].notna()
    if is_regression and "status" in df:
        assert not (df.loc[mask, "status"] == "active").any(), "Active rows leaked into clearance target"
    subset = df.loc[mask].copy()
    features = [name for name in feature_cols if name in subset and name != "status"]
    X = subset[features].copy()
    # Convert bool columns to int first (avoids dtype issues with select_dtypes)
    for col in X.select_dtypes(include=["bool"]).columns:
        X[col] = X[col].astype(int)
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    for column in categorical:
        X[column] = X[column].fillna("__MISSING__").astype(str)
    numeric = [column for column in features if column not in categorical]
    X[numeric] = X[numeric].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = subset[target_col].copy()
    target_encoder = None
    if not is_regression:
        target_encoder = LabelEncoder()
        y = pd.Series(target_encoder.fit_transform(y.astype(str)), index=y.index)
    return X, y.astype(float if is_regression else int), categorical, features, target_encoder


def suggest_params(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 150, 1600, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "max_depth": trial.suggest_categorical("max_depth", [-1, 3, 5, 7, 10, 15]),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.55, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        "max_bin": trial.suggest_int("max_bin", 63, 255),
    }


def make_encoder(categorical):
    return TargetEncoder(
        cols=categorical, smoothing=12, min_samples_leaf=20,
        handle_missing="value", handle_unknown="value", return_df=True,
    )


def make_estimator(params, is_regression, scale_pos_weight=1.0):
    common = dict(params, random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)
    if is_regression:
        return lgb.LGBMRegressor(objective="regression_l1", **common)
    return lgb.LGBMClassifier(
        objective="binary", scale_pos_weight=scale_pos_weight, **common
    )


def tune_model(X, y, categorical, is_regression, trials, model_key):
    splitter = (
        KFold(5, shuffle=True, random_state=RANDOM_STATE)
        if is_regression else
        StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    )

    def objective(trial):
        params = suggest_params(trial)
        scores = []
        for train_idx, valid_idx in splitter.split(X, y):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
            fit_target = np.log1p(y_train) if is_regression else y_train
            encoder = make_encoder(categorical)
            encoded_train = encoder.fit_transform(X_train, fit_target)
            encoded_valid = encoder.transform(X_valid)
            ratio = 1.0
            if not is_regression:
                counts = y_train.value_counts()
                ratio = float(counts.get(0, 1) / max(counts.get(1, 1), 1))
            estimator = make_estimator(params, is_regression, ratio)
            estimator.fit(
                encoded_train,
                fit_target,
                eval_set=[(encoded_valid, np.log1p(y_valid) if is_regression else y_valid)],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            prediction = estimator.predict(encoded_valid)
            if is_regression:
                scores.append(mean_absolute_error(y_valid, np.expm1(prediction).clip(min=0)))
            else:
                scores.append(roc_auc_score(y_valid, estimator.predict_proba(encoded_valid)[:, 1]))
        trial.set_user_attr("fold_scores", [float(score) for score in scores])
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="minimize" if is_regression else "maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=15),
        study_name=model_key,
    )
    study.optimize(objective, n_trials=trials, show_progress_bar=False, gc_after_trial=True)
    return study


def save_optimization_plot(study, model_key):
    completed = [trial for trial in study.trials if trial.value is not None]
    values = [trial.value for trial in completed]
    best = np.minimum.accumulate(values) if study.direction.name == "MINIMIZE" else np.maximum.accumulate(values)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(range(1, len(values) + 1), values, s=14, alpha=.45, label="CV score")
    ax.plot(range(1, len(best) + 1), best, color="#00a8cc", linewidth=2, label="Best so far")
    ax.set(xlabel="Optuna trial", ylabel="CV MAE (minutes)" if study.direction.name == "MINIMIZE" else "CV ROC-AUC")
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(DIAGNOSTICS_DIR, f"{model_key}_optimization.png"), dpi=160)
    plt.close(fig)


def save_learning_curve(evals_result, model_key, metric):
    if not evals_result:
        return
    validation = next(iter(evals_result.values()))
    values = next(iter(validation.values()))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(values, color="#00a8cc", linewidth=2)
    ax.set(xlabel="Boosting iteration", ylabel=metric, title=f"{model_key.upper()} validation learning curve")
    ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(os.path.join(DIAGNOSTICS_DIR, f"{model_key}_learning_curve.png"), dpi=160)
    plt.close(fig)


def save_shap_plot(bundle, X, model_key):
    try:
        import shap
        sample = X.sample(min(500, len(X)), random_state=RANDOM_STATE)
        encoded = bundle._transform(sample)
        explainer = shap.TreeExplainer(bundle.estimator)
        values = explainer.shap_values(encoded)
        if isinstance(values, list):
            values = values[-1]
        shap.summary_plot(values, encoded, feature_names=bundle.feature_names, show=False, max_display=18)
        plt.tight_layout()
        plt.savefig(os.path.join(DIAGNOSTICS_DIR, f"{model_key}_shap.png"), dpi=160, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        print(f"  SHAP diagnostic skipped for {model_key}: {exc}")


def train_final(X, y, categorical, study, is_regression, model_key, target_encoder=None):
    stratify = None if is_regression else y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify
    )
    inner_stratify = None if is_regression else y_train
    X_fit, X_valid, y_fit, y_valid = train_test_split(
        X_train, y_train, test_size=.15, random_state=RANDOM_STATE, stratify=inner_stratify
    )
    fit_target = np.log1p(y_fit) if is_regression else y_fit
    valid_target = np.log1p(y_valid) if is_regression else y_valid
    probe_encoder = make_encoder(categorical)
    fit_encoded = probe_encoder.fit_transform(X_fit, fit_target)
    valid_encoded = probe_encoder.transform(X_valid)
    counts = y_fit.value_counts() if not is_regression else None
    ratio = 1.0 if is_regression else float(counts.get(0, 1) / max(counts.get(1, 1), 1))
    probe = make_estimator(study.best_params, is_regression, ratio)
    probe.fit(
        fit_encoded, fit_target, eval_set=[(valid_encoded, valid_target)],
        callbacks=[lgb.early_stopping(60, verbose=False), lgb.record_evaluation({})],
    )
    best_iteration = max(int(probe.best_iteration_ or study.best_params["n_estimators"]), 20)
    evals_result = probe.evals_result_

    full_encoder = make_encoder(categorical)
    full_target = np.log1p(y_train) if is_regression else y_train
    train_encoded = full_encoder.fit_transform(X_train, full_target)
    final_params = dict(study.best_params, n_estimators=best_iteration)
    full_counts = y_train.value_counts() if not is_regression else None
    full_ratio = 1.0 if is_regression else float(full_counts.get(0, 1) / max(full_counts.get(1, 1), 1))
    estimator = make_estimator(final_params, is_regression, full_ratio)
    estimator.fit(train_encoded, full_target)
    bundle = ModelBundle(full_encoder, estimator, list(X.columns), categorical, log_target=is_regression)
    prediction = bundle.predict(X_test)
    importance = dict(sorted(zip(X.columns, estimator.feature_importances_), key=lambda pair: pair[1], reverse=True))
    common = {
        "algorithm": "LightGBM + leakage-safe target encoding",
        "best_params": final_params,
        "tuning_trials": len(study.trials),
        "train_size": len(X_train), "test_size": len(X_test),
        "feature_importance": {name: int(value) for name, value in importance.items()},
        "target_encoding": "Fold-fitted regularized target encoding",
        "early_stopping_best_iteration": best_iteration,
    }
    if is_regression:
        mae = mean_absolute_error(y_test, prediction)
        rmse = float(np.sqrt(mean_squared_error(y_test, prediction)))
        results = dict(common, model_name="Model C: Clearance Time Regressor", target_transform="log1p/expm1",
                       best_cv_mae_minutes=round(float(study.best_value), 2), test_mae_minutes=round(float(mae), 2),
                       test_rmse_minutes=round(rmse, 2), median_absolute_error_minutes=round(float(np.median(np.abs(y_test - prediction))), 2))
    else:
        probability = bundle.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probability)
        report = classification_report(y_test, prediction, target_names=list(target_encoder.classes_), output_dict=True, zero_division=0)
        results = dict(common, model_name="Model A: Priority Classifier" if model_key == "model_a" else "Model B: Closure Classifier",
                       best_cv_roc_auc=round(float(study.best_value), 4), roc_auc=round(float(auc), 4), classification_report=report)
        ConfusionMatrixDisplay.from_predictions(y_test, prediction, display_labels=list(target_encoder.classes_), cmap="Blues", colorbar=False)
        plt.tight_layout(); plt.savefig(os.path.join(DIAGNOSTICS_DIR, f"{model_key}_confusion_matrix.png"), dpi=160); plt.close()
    save_learning_curve(evals_result, model_key, "L1" if is_regression else "binary logloss")
    save_optimization_plot(study, model_key)
    save_shap_plot(bundle, X_test, model_key)
    return bundle, results, target_encoder


def run_training(trials=100):
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(DIAGNOSTICS_DIR, exist_ok=True)
    df, config = load_data()
    specifications = [
        ("model_a", "priority_target", config["classification_features"], False, "model_a_priority.pkl"),
        ("model_b", "closure_target", config["classification_features"], False, "model_b_closure.pkl"),
        ("model_c", "clearance_time_minutes", config["regression_features"], True, "model_c_clearance.pkl"),
    ]
    all_results = {}
    for model_key, target, features, is_regression, filename in specifications:
        print(f"\n{'=' * 68}\nTraining {model_key} with {trials} Optuna trials\n{'=' * 68}")
        X, y, categorical, feature_names, target_encoder = prepare_features(df, features, target, is_regression)
        study = tune_model(X, y, categorical, is_regression, trials, model_key)
        bundle, results, target_encoder = train_final(X, y, categorical, study, is_regression, model_key, target_encoder)
        all_results[model_key] = results
        joblib.dump(bundle, os.path.join(MODELS_DIR, filename))
        prefix = model_key.replace("model_", "model_")
        joblib.dump({"__target__": target_encoder, "__bundle__": True}, os.path.join(MODELS_DIR, f"{prefix}_encoders.pkl"))
        joblib.dump(feature_names, os.path.join(MODELS_DIR, f"{prefix}_features.pkl"))
        print(f"  Best CV score: {study.best_value:.4f}")
    with open(os.path.join(OUTPUTS_DIR, "model_evaluation_report.json"), "w", encoding="utf-8") as handle:
        json.dump(all_results, handle, indent=2, default=str)
    with open(os.path.join(OUTPUTS_DIR, "split_sizes.json"), "w", encoding="utf-8") as handle:
        json.dump({key: {"train": value["train_size"], "test": value["test_size"]} for key, value in all_results.items()}, handle, indent=2)
    print("\nModel training complete.")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=int(os.environ.get("OPTUNA_TRIALS", "100")))
    args = parser.parse_args()
    run_training(max(args.trials, 1))
