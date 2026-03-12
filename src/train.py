"""
train.py
--------
XGBoost training pipeline for fintech churn prediction.

Design decisions:
- scale_pos_weight handles class imbalance natively in XGBoost
- Optuna for hyperparameter tuning (Bayesian optimization)
- StratifiedKFold inside Optuna objective to prevent data leakage
- Early stopping prevents overfitting
"""

import numpy as np
import joblib
import optuna
import xgboost as xgb

from sklearn.pipeline    import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics     import roc_auc_score

optuna.logging.set_verbosity(optuna.logging.WARNING)


def compute_scale_pos_weight(y) -> float:
    """
    Computes scale_pos_weight for XGBoost imbalance handling.
    Formula: count(negative class) / count(positive class)
    This tells XGBoost to penalize missing a churner more heavily.
    """
    n_negative = (y == 0).sum()
    n_positive = (y == 1).sum()
    ratio = n_negative / n_positive
    print(f"scale_pos_weight = {ratio:.4f} "
          f"({n_negative} non-churners / {n_positive} churners)")
    return ratio


def build_xgb_pipeline(preprocessor, params: dict, scale_pos_weight: float) -> Pipeline:
    """
    Builds a full sklearn Pipeline with preprocessor + XGBoost classifier.
    """
    model = xgb.XGBClassifier(
        **params,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="auc",
        random_state=42,
        verbosity=0,
    )
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])


def optuna_objective(trial, X_train, y_train,
                     preprocessor, scale_pos_weight: float) -> float:
    """
    Optuna objective function.
    Each trial suggests a set of hyperparameters and evaluates
    them via 5-fold cross-validation ROC-AUC on the training set.
    No test set is touched during tuning.
    """
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 2),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 3),
    }

    pipeline = build_xgb_pipeline(preprocessor, params, scale_pos_weight)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=cv, scoring="roc_auc", n_jobs=-1
    )
    return scores.mean()


def run_optuna_study(X_train, y_train, preprocessor,
                     scale_pos_weight: float,
                     n_trials: int = 50) -> optuna.Study:
    """
    Runs the full Optuna hyperparameter search.
    n_trials=50 is a good balance between search quality and runtime.
    """
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: optuna_objective(
            trial, X_train, y_train, preprocessor, scale_pos_weight
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    print(f"\nBest ROC-AUC (CV): {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")
    return study


def save_model(pipeline: Pipeline, path: str) -> None:
    joblib.dump(pipeline, path)
    print(f"Model saved to {path}")


def load_model(path: str) -> Pipeline:
    return joblib.load(path)