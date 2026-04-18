# %% [markdown]
# ## 6) Autoencoder anomaly detection

# %%
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results = []
preds = {}

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
        + ". Please run 01_data_preprocessing.py first to generate these files."
    )

X_train_no_outliers_scaled = pd.read_pickle(
    os.path.join(BASE_DIR, "X_train_no_outliers_scaled.pkl")
)
y_train_no_outliers = pd.read_pickle(os.path.join(BASE_DIR, "y_train_no_outliers.pkl"))
X_val_scaled = pd.read_pickle(os.path.join(BASE_DIR, "X_val_scaled.pkl"))
y_val = pd.read_pickle(os.path.join(BASE_DIR, "y_val.pkl"))
X_test_scaled = pd.read_pickle(os.path.join(BASE_DIR, "X_test_scaled.pkl"))
y_test = pd.read_pickle(os.path.join(BASE_DIR, "y_test.pkl"))

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(
    X_train_no_outliers_scaled, y_train_no_outliers
)

print("Train shape:", X_train_smote.shape)
print("Validation shape:", X_val_scaled.shape)
print("Test shape:", X_test_scaled.shape)


class FraudDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(np.array(y), dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class AutoEncoder(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, n_features),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


train_ds = FraudDataset(X_train_smote, y_train_smote)
val_ds = FraudDataset(X_val_scaled, y_val.values)
test_ds = FraudDataset(X_test_scaled, y_test.values)

train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False)


def train_autoencoder(model, train_loader, val_loader, epochs=6, lr=1e-3):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    history = {"train_loss": [], "val_loss": []}
    best = None
    best_val = float("inf")
    for epoch in range(epochs):
        model.train()
        tl = 0.0
        for xb, _ in train_loader:
            xb = xb.to(DEVICE)
            opt.zero_grad()
            recon = model(xb)
            loss = criterion(recon, xb)
            loss.backward()
            opt.step()
            tl += loss.item() * len(xb)
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(DEVICE)
                recon = model(xb)
                loss = criterion(recon, xb)
                vl += loss.item() * len(xb)
        tl /= len(train_loader.dataset)
        vl /= len(val_loader.dataset)
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        if vl < best_val:
            best_val = vl
            best = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"AE Epoch {epoch + 1}: train={tl:.4f}, val={vl:.4f}")
    model.load_state_dict(best)
    return model, history


def plot_loss(history, name):
    plt.figure(figsize=(6, 4))
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.title(f"{name} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{name.lower()}_loss.png", dpi=200, bbox_inches="tight")
    plt.show()


def ae_scores(model, loader):
    model.eval()
    errs = []
    ys = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            recon = model(xb)
            err = torch.mean((recon - xb) ** 2, dim=1).cpu().numpy()
            errs.extend(err)
            ys.extend(yb.numpy())
    return np.array(ys), np.array(errs)


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


ae = AutoEncoder(X_train_smote.shape[1])
ae, ae_hist = train_autoencoder(ae, train_loader, val_loader, epochs=6, lr=1e-3)
plot_loss(ae_hist, "Autoencoder")

y_ae, err_test = ae_scores(ae, test_loader)
threshold = np.percentile(err_test, 95)
ae_prob = (err_test - err_test.min()) / (err_test.max() - err_test.min() + 1e-8)
ae_pred = (err_test >= threshold).astype(int)
results.append(
    {
        "model": "Autoencoder",
        "accuracy": accuracy_score(y_test, ae_pred),
        "precision": precision_score(y_test, ae_pred, zero_division=0),
        "recall": recall_score(y_test, ae_pred, zero_division=0),
        "f1": f1_score(y_test, ae_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, ae_prob),
        "pr_auc": average_precision_score(y_test, ae_prob),
    }
)
preds["Autoencoder"] = ae_prob
plot_roc_pr(y_test, ae_prob, "Autoencoder", "autoencoder")

# %% [markdown]
# ## 7) Compare all models

# %%
results_df = pd.DataFrame(results).sort_values(by="pr_auc", ascending=False)
results_df.to_csv("model_comparison.csv", index=False)
results_df.to_csv(os.path.join(BASE_DIR, "autoencoder_metrics.csv"), index=False)
print(results_df)
print("Saved autoencoder metrics to autoencoder_metrics.csv")

plt.figure(figsize=(12, 5))
sns.barplot(
    data=results_df,
    x="model",
    y="pr_auc",
    hue="model",
    dodge=False,
    palette="mako",
    legend=False,
)
plt.xticks(rotation=30, ha="right")
plt.title("Model Comparison by PR-AUC")
plt.tight_layout()
plt.savefig("model_pr_auc_comparison.png", dpi=200, bbox_inches="tight")
plt.show()

plt.figure(figsize=(12, 5))
sns.barplot(
    data=results_df,
    x="model",
    y="f1",
    hue="model",
    dodge=False,
    palette="rocket",
    legend=False,
)
plt.xticks(rotation=30, ha="right")
plt.title("Model Comparison by F1-score")
plt.tight_layout()
plt.savefig("model_f1_comparison.png", dpi=200, bbox_inches="tight")
plt.show()

# %%
for name, prob in preds.items():
    y_pred = (prob >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4.8, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"cm_{name.lower()}.png", dpi=200, bbox_inches="tight")
    plt.show()
