# %% [markdown]
# ## 5) PyTorch datasets and models

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


train_ds = FraudDataset(X_train_smote, y_train_smote)
val_ds = FraudDataset(X_val_scaled, y_val.values)
test_ds = FraudDataset(X_test_scaled, y_test.values)

train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False)


class CNN1D(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.net(x).squeeze(1)


class LSTMNet(nn.Module):
    def __init__(self, n_features, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=hidden, batch_first=True, num_layers=1
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden, 16), nn.ReLU(), nn.Dropout(0.2), nn.Linear(16, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze(1)


class AutoEncoder(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(), nn.Linear(64, 16), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, n_features)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


# %%
def train_classifier(model, train_loader, val_loader, epochs=6, lr=1e-3):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [len(y_train_smote) / max(y_train_smote.sum(), 1) - 1], device=DEVICE
        )
    )
    history = {"train_loss": [], "val_loss": []}
    best = None
    best_val = float("inf")
    for epoch in range(epochs):
        model.train()
        tl = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            tl += loss.item() * len(xb)
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                loss = criterion(logits, yb)
                vl += loss.item() * len(xb)
        tl /= len(train_loader.dataset)
        vl /= len(val_loader.dataset)
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        if vl < best_val:
            best_val = vl
            best = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"Epoch {epoch + 1}: train={tl:.4f}, val={vl:.4f}")
    model.load_state_dict(best)
    return model, history


def predict_proba_torch(model, loader):
    model.eval()
    probs = []
    ys = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            p = torch.sigmoid(logits).cpu().numpy()
            probs.extend(p)
            ys.extend(yb.numpy())
    return np.array(ys), np.array(probs)


# %%
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


# %%
cnn, cnn_hist = train_classifier(
    CNN1D(X_train_smote.shape[1]), train_loader, val_loader, epochs=6, lr=1e-3
)
plot_loss(cnn_hist, "CNN")
y_true_cnn, cnn_prob = predict_proba_torch(cnn, test_loader)
results.append({"model": "PyTorch_CNN", **eval_binary(y_true_cnn, cnn_prob)})
preds["PyTorch_CNN"] = cnn_prob
plot_roc_pr(y_test, cnn_prob, "PyTorch CNN", "pytorch_cnn")

# %%
lstm, lstm_hist = train_classifier(
    LSTMNet(X_train_smote.shape[1]), train_loader, val_loader, epochs=6, lr=1e-3
)
plot_loss(lstm_hist, "LSTM")
y_true_lstm, lstm_prob = predict_proba_torch(lstm, test_loader)
results.append({"model": "PyTorch_LSTM", **eval_binary(y_true_lstm, lstm_prob)})
preds["PyTorch_LSTM"] = lstm_prob
plot_roc_pr(y_test, lstm_prob, "PyTorch LSTM", "pytorch_lstm")

results_df = pd.DataFrame(results).sort_values(by="pr_auc", ascending=False)
results_df.to_csv(os.path.join(BASE_DIR, "pytorch_model_metrics.csv"), index=False)
print("Saved PyTorch metric results to pytorch_model_metrics.csv")

results_df = pd.DataFrame(results).sort_values(by="pr_auc", ascending=False)
results_df.to_csv(os.path.join(BASE_DIR, "pytorch_model_metrics.csv"), index=False)
print("Saved PyTorch metric results to pytorch_model_metrics.csv")
