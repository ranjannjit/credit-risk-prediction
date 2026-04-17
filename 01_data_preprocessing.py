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

from sklearn.model_selection import train_test_split
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
from sklearn.ensemble import RandomForestClassifier

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

# %%
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def save_plot(filename):
    plt.savefig(os.path.join(BASE_DIR, filename), dpi=200, bbox_inches="tight")


# %% [markdown]
# ## 1) Load data
# Kaggle dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# The dataset has 284,807 transactions and 492 fraud cases.

# %%
# C:\Users\ranja\Ranjan_Doc\Ranjan\NJIT\ms\Deep Learning\Project\Project-credit_card_fraud_detection\credit_card_fraud_detection\creditcard.csv
DATA_PATH = "creditcard.csv"  # put the Kaggle CSV in the same folder or update path
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
else:
    raise FileNotFoundError(
        "Please download Kaggle creditcard.csv and place it at DATA_PATH."
    )

print(df.shape)
print(df["Class"].value_counts())
print(df["Class"].value_counts(normalize=True))
print(df.head())
print(df.info())

# %% [markdown]
# ## 2) Quick EDA

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
sns.countplot(x="Class", data=df, ax=axes[0])
axes[0].set_title("Class Distribution")
axes[0].set_xticklabels(["Legit", "Fraud"])

sns.histplot(df["Amount"], bins=60, kde=True, ax=axes[1], color="tomato")
axes[1].set_title("Amount Distribution")

sns.histplot(df["Time"], bins=60, kde=True, ax=axes[2], color="steelblue")
axes[2].set_title("Time Distribution")
plt.tight_layout()
save_plot("eda_overview.png")
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.heatmap(
    df[["Time", "Amount", "Class"]].corr(), annot=True, cmap="coolwarm", fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
save_plot("correlation_heatmap.png")
plt.show()

# %% [markdown]
# ## 3) Preprocessing

# %%
X = df.drop(columns=["Class"])
y = df["Class"].astype(int)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
)

print("Train/Val/Test shapes:", X_train.shape, X_val.shape, X_test.shape)
for split_name, y_split in [
    ("Train", y_train),
    ("Validation", y_val),
    ("Test", y_test),
]:
    counts = y_split.value_counts()
    props = (counts / len(y_split)).rename("proportion")
    print(f"\n{split_name} class counts:\n{counts}")
    print(f"{split_name} class proportions:\n{props.round(4)}")

imbalance_df = pd.concat(
    [
        y_train.value_counts()
        .rename_axis("class")
        .reset_index(name="count")
        .assign(split="Train"),
        y_val.value_counts()
        .rename_axis("class")
        .reset_index(name="count")
        .assign(split="Validation"),
        y_test.value_counts()
        .rename_axis("class")
        .reset_index(name="count")
        .assign(split="Test"),
    ],
    ignore_index=True,
)

plt.figure(figsize=(10, 5))
sns.barplot(data=imbalance_df, x="split", y="count", hue="class", palette="coolwarm")
plt.title("Class Imbalance by Split")
plt.ylabel("Transaction Count")
plt.tight_layout()
save_plot("class_imbalance_by_split.png")
plt.show()

corr = df.corr()
plt.figure(figsize=(16, 14))
sns.heatmap(
    corr, cmap="coolwarm", center=0, fmt=".2f", square=True, cbar_kws={"shrink": 0.5}
)
plt.title("Correlation Matrix")
plt.tight_layout()
save_plot("correlation_matrix.png")
plt.show()

corr_with_class = corr["Class"].sort_values(ascending=False)
print("\nCorrelation with Class label:\n", corr_with_class)

# Outlier detection and removal for fraudulent transactions in the training set only
train_df = X_train.copy()
train_df["Class"] = y_train
fraud_train = train_df[train_df["Class"] == 1]

iqr_cols = ["Time", "Amount"]
iqr_bounds = {}
for col in iqr_cols:
    q1 = fraud_train[col].quantile(0.25)
    q3 = fraud_train[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    iqr_bounds[col] = (lower, upper)
    outlier_count = fraud_train[
        (fraud_train[col] < lower) | (fraud_train[col] > upper)
    ].shape[0]
    print(
        f"{col} outlier bounds for fraud class: ({lower:.2f}, {upper:.2f}), count={outlier_count}"
    )

outlier_mask = pd.Series(False, index=X_train.index)
for col, (lower, upper) in iqr_bounds.items():
    col_outliers = (X_train[col] < lower) | (X_train[col] > upper)
    outlier_mask |= col_outliers & (y_train == 1)

print("\nFraudulent train outliers removed:", outlier_mask.sum())
X_train_no_outliers = X_train.loc[~outlier_mask].copy()
y_train_no_outliers = y_train.loc[~outlier_mask].copy()

print("Train shape before outlier removal:", X_train.shape)
print("Train shape after outlier removal:", X_train_no_outliers.shape)
print(
    "Fraud class distribution in train after outlier removal:\n",
    y_train_no_outliers.value_counts(),
)
print(
    "Train class proportions after outlier removal:\n",
    (y_train_no_outliers.value_counts(normalize=True)).round(4),
)

plt.figure(figsize=(12, 4))
for i, col in enumerate(iqr_cols, 1):
    plt.subplot(1, len(iqr_cols), i)
    sns.boxplot(x="Class", y=col, data=train_df)
    plt.title(f"{col} Distribution by Class")
plt.tight_layout()
save_plot("outlier_boxplots.png")
plt.show()

scaler = StandardScaler()
X_train_no_outliers_scaled = X_train_no_outliers.copy()
X_val_scaled = X_val.copy()
X_test_scaled = X_test.copy()

X_train_no_outliers_scaled[["Time", "Amount"]] = scaler.fit_transform(
    X_train_no_outliers[["Time", "Amount"]]
)
X_val_scaled[["Time", "Amount"]] = scaler.transform(X_val[["Time", "Amount"]])
X_test_scaled[["Time", "Amount"]] = scaler.transform(X_test[["Time", "Amount"]])

print(
    "Train/Val/Test shapes after scaling:",
    X_train_no_outliers_scaled.shape,
    X_val_scaled.shape,
    X_test_scaled.shape,
)
print("Train class distribution before SMOTE:\n", y_train_no_outliers.value_counts())

# Save preprocessed datasets for downstream scripts
X_train_no_outliers_scaled.to_pickle(
    os.path.join(BASE_DIR, "X_train_no_outliers_scaled.pkl")
)
y_train_no_outliers.to_pickle(os.path.join(BASE_DIR, "y_train_no_outliers.pkl"))
X_val_scaled.to_pickle(os.path.join(BASE_DIR, "X_val_scaled.pkl"))
y_val.to_pickle(os.path.join(BASE_DIR, "y_val.pkl"))
X_test_scaled.to_pickle(os.path.join(BASE_DIR, "X_test_scaled.pkl"))
y_test.to_pickle(os.path.join(BASE_DIR, "y_test.pkl"))

# %%
smote = SMOTE(random_state=RANDOM_STATE)
X_train_smote, y_train_smote = smote.fit_resample(
    X_train_no_outliers_scaled, y_train_no_outliers
)
print(
    "Train class distribution after SMOTE:\n", pd.Series(y_train_smote).value_counts()
)

# Save split sizes
split_sizes = pd.DataFrame(
    {
        "split": ["train", "train_no_outliers", "val", "test", "train_smote"],
        "rows": [
            len(X_train),
            len(X_train_no_outliers),
            len(X_val),
            len(X_test),
            len(X_train_smote),
        ],
    }
)
split_sizes.to_csv("split_sizes.csv", index=False)

plt.figure(figsize=(8, 4))
sns.barplot(data=split_sizes, x="split", y="rows", palette="viridis")
plt.title("Dataset Split Sizes")
plt.tight_layout()
save_plot("split_sizes.png")
plt.show()
