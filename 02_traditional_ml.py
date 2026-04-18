# %% [markdown]
# Credit Card Fraud Detection - End-to-End Notebook Style
# Models: PyTorch CNN, PyTorch LSTM, Autoencoder, Logistic Regression, Decision Tree, Random Forest, XGBoost

# %%
import os, random, math, json, warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.base import clone

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Optional XGBoost
try:
    from xgboost import XGBClassifier

    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# %% [markdown]
# ## 4) Traditional Machine Learning Models


# %%
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

required_files = [
    "X_train_no_outliers_scaled.pkl",
    "y_train_no_outliers.pkl",
    "X_val_scaled.pkl",
    "y_val.pkl",
    "X_test_scaled.pkl",
    "y_test.pkl",
]
missing_files = [
    f for f in required_files if not os.path.exists(os.path.join(BASE_DIR, f))
]
if missing_files:
    raise FileNotFoundError(
        "Missing preprocessed dataset files: "
        + ", ".join(missing_files)
        + ".\nPlease run 01_data_preprocessing.py first to generate these files."
    )

X_train_no_outliers_scaled = pd.read_pickle(
    os.path.join(BASE_DIR, "X_train_no_outliers_scaled.pkl")
)
y_train_no_outliers = pd.read_pickle(os.path.join(BASE_DIR, "y_train_no_outliers.pkl"))
X_val_scaled = pd.read_pickle(os.path.join(BASE_DIR, "X_val_scaled.pkl"))
y_val = pd.read_pickle(os.path.join(BASE_DIR, "y_val.pkl"))
X_test_scaled = pd.read_pickle(os.path.join(BASE_DIR, "X_test_scaled.pkl"))
y_test = pd.read_pickle(os.path.join(BASE_DIR, "y_test.pkl"))

smote = SMOTE(random_state=RANDOM_STATE)
X_train_smote, y_train_smote = smote.fit_resample(
    X_train_no_outliers_scaled, y_train_no_outliers
)

print("Train shape:", X_train_smote.shape)
print("Validation shape:", X_val_scaled.shape)
print("Test shape:", X_test_scaled.shape)


def eval_binary(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "confusion_matrix": np.array2string(cm),
    }


def plot_roc_pr(y_true, y_prob, title_prefix, out_prefix):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(fpr, tpr, label=f"AUC={roc_auc_score(y_true, y_prob):.4f}")
    ax[0].plot([0, 1], [0, 1], "--")
    ax[0].set_title(f"{title_prefix} ROC")
    ax[0].set_xlabel("FPR")
    ax[0].set_ylabel("TPR")
    ax[0].legend()
    ax[1].plot(rec, prec, label=f"AP={average_precision_score(y_true, y_prob):.4f}")
    ax[1].set_title(f"{title_prefix} PR Curve")
    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(BASE_DIR, f"{out_prefix}_roc_pr.png"), dpi=200, bbox_inches="tight"
    )
    plt.close(fig)


def plot_combined_models(
    preds, y_true, title_prefix="All Models", out_prefix="all_models"
):
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    for name, y_prob in preds.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ax[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_true, y_prob):.4f})")
        ax[1].plot(
            rec,
            prec,
            label=f"{name} (AP={average_precision_score(y_true, y_prob):.4f})",
        )
    ax[0].plot([0, 1], [0, 1], "--", color="gray")
    ax[0].set_title(f"{title_prefix} ROC Comparison")
    ax[0].set_xlabel("FPR")
    ax[0].set_ylabel("TPR")
    ax[0].legend(loc="lower right", fontsize="small")
    ax[1].set_title(f"{title_prefix} Precision-Recall Comparison")
    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")
    ax[1].legend(loc="lower left", fontsize="small")
    plt.tight_layout()
    plt.savefig(
        os.path.join(BASE_DIR, f"{out_prefix}_comparison.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_metrics_summary(results, out_prefix="model_metrics_summary"):
    df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    csv_path = os.path.join(BASE_DIR, f"{out_prefix}.csv")
    png_path = os.path.join(BASE_DIR, f"{out_prefix}.png")
    html_path = os.path.join(BASE_DIR, f"{out_prefix}.html")
    df.to_csv(csv_path, index=False)
    df.to_html(html_path, index=False, justify="center")
    fig, ax = plt.subplots(figsize=(10, len(df) * 0.6 + 1))
    ax.axis("off")
    table = ax.table(
        cellText=df.round(4).values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    plt.tight_layout()
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return df


def make_smote_pipeline(model):
    return ImbPipeline([("smote", SMOTE(random_state=RANDOM_STATE)), ("clf", model)])


def print_cross_val_scores(model, X, y, cv=5):
    cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    model_for_cv = clone(model)
    if isinstance(model_for_cv, VotingClassifier):
        model_for_cv.n_jobs = 1
    if isinstance(model_for_cv, SVC) and hasattr(model_for_cv, "probability"):
        model_for_cv.probability = False
    pipeline = make_smote_pipeline(model_for_cv)
    cv_jobs = 1 if isinstance(model_for_cv, VotingClassifier) else -1
    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="roc_auc",
        n_jobs=cv_jobs,
    )
    print(f"{model.__class__.__name__} CV ROC AUC scores: {scores.round(4)}")
    print(f"Mean ROC AUC: {scores.mean():.4f} ± {scores.std():.4f}")
    return scores


def plot_cv_roc_auc(model, X, y, cv=5, title_prefix="CV", out_prefix="cv"):
    cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    fig, ax = plt.subplots(figsize=(8, 6))
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]
        model_for_cv = clone(model)
        if isinstance(model_for_cv, VotingClassifier):
            model_for_cv.n_jobs = 1
        if isinstance(model_for_cv, SVC) and hasattr(model_for_cv, "probability"):
            model_for_cv.probability = False
        pipeline = make_smote_pipeline(model_for_cv)
        pipeline.fit(X_train_fold, y_train_fold)
        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_val_fold)[:, 1]
        else:
            y_prob = pipeline.decision_function(X_val_fold)
        fpr, tpr, _ = roc_curve(y_val_fold, y_prob)
        auc_score = roc_auc_score(y_val_fold, y_prob)
        aucs.append(auc_score)
        ax.plot(fpr, tpr, lw=1, alpha=0.4, label=f"Fold {fold} AUC={auc_score:.4f}")
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        lw=2,
        label=f"Mean ROC AUC = {mean_auc:.4f} ± {std_auc:.4f}",
    )
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_title(f"{title_prefix} Cross-Validated ROC")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(
        os.path.join(BASE_DIR, f"{out_prefix}_cv_roc_auc.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)
    return aucs


Xtr = X_train_no_outliers_scaled
Xv = X_val_scaled
Xt = X_test_scaled

models = {}
models["LogisticRegression"] = LogisticRegression(
    max_iter=2000, class_weight=None, n_jobs=-1
)
models["DecisionTree"] = DecisionTreeClassifier(
    max_depth=8, random_state=RANDOM_STATE, class_weight=None
)
models["RandomForest"] = RandomForestClassifier(
    n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
)
models["SVM"] = SVC(
    kernel="rbf",
    probability=True,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    cache_size=200,
)
if XGB_AVAILABLE:
    models["XGBoost"] = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
voting_estimators = [(name.lower(), clone(model)) for name, model in models.items()]
models["Voting"] = VotingClassifier(
    estimators=voting_estimators,
    voting="soft",
    n_jobs=1,
)

results = []
preds = {}
for name, model in models.items():
    cv_splits = 3 if name == "SVM" else 5
    print(f"\n=== {name} Cross-Validation ===")
    print_cross_val_scores(model, Xtr, y_train_no_outliers, cv=cv_splits)
    plot_cv_roc_auc(
        model,
        Xtr,
        y_train_no_outliers,
        cv=cv_splits,
        title_prefix=name,
        out_prefix=f"{name.lower()}_train",
    )
    model.fit(X_train_smote, y_train_smote)
    prob = model.predict_proba(Xt)[:, 1]
    preds[name] = prob
    m = eval_binary(y_test, prob)
    m["model"] = name
    results.append(m)
    plot_roc_pr(y_test, prob, name, name.lower())

plot_combined_models(
    preds, y_test, title_prefix="All Traditional Models", out_prefix="all_models"
)
summary_df = save_metrics_summary(results, out_prefix="model_metrics_summary")
print("\nModel metrics summary:")
print(summary_df.to_string(index=False))
