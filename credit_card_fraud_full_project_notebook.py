# %% [markdown]
# Credit Card Fraud Detection - End-to-End Notebook Style
# Models: PyTorch CNN, PyTorch LSTM, Autoencoder, Logistic Regression, Decision Tree, Random Forest, XGBoost

# %%
import os, random, math, json, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, precision_recall_curve, roc_curve,
    f1_score, precision_score, recall_score, accuracy_score
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
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE)

# %% [markdown]
# ## 1) Load data
# Kaggle dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# The dataset has 284,807 transactions and 492 fraud cases.

# %%
DATA_PATH = 'creditcard.csv'  # put the Kaggle CSV in the same folder or update path
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
else:
    raise FileNotFoundError('Please download Kaggle creditcard.csv and place it at DATA_PATH.')

print(df.shape)
print(df['Class'].value_counts())
print(df['Class'].value_counts(normalize=True))

# %% [markdown]
# ## 2) Quick EDA

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
sns.countplot(x='Class', data=df, ax=axes[0])
axes[0].set_title('Class Distribution')
axes[0].set_xticklabels(['Legit', 'Fraud'])

sns.histplot(df['Amount'], bins=60, kde=True, ax=axes[1], color='tomato')
axes[1].set_title('Amount Distribution')

sns.histplot(df['Time'], bins=60, kde=True, ax=axes[2], color='steelblue')
axes[2].set_title('Time Distribution')
plt.tight_layout()
plt.savefig('eda_overview.png', dpi=200, bbox_inches='tight')
plt.show()

# %%
plt.figure(figsize=(8,6))
sns.heatmap(df[['Time','Amount','Class']].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=200, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 3) Preprocessing

# %%
X = df.drop(columns=['Class'])
y = df['Class'].astype(int)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
)

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()
X_test_scaled = X_test.copy()

X_train_scaled[['Time','Amount']] = scaler.fit_transform(X_train[['Time','Amount']])
X_val_scaled[['Time','Amount']] = scaler.transform(X_val[['Time','Amount']])
X_test_scaled[['Time','Amount']] = scaler.transform(X_test[['Time','Amount']])

print('Train/Val/Test:', X_train_scaled.shape, X_val_scaled.shape, X_test_scaled.shape)
print('Train class distribution before SMOTE:\n', y_train.value_counts())

# %%
smote = SMOTE(random_state=RANDOM_STATE)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
print('Train class distribution after SMOTE:\n', pd.Series(y_train_smote).value_counts())

# Save split sizes
split_sizes = pd.DataFrame({
    'split': ['train','val','test','train_smote'],
    'rows': [len(X_train), len(X_val), len(X_test), len(X_train_smote)]
})
split_sizes.to_csv('split_sizes.csv', index=False)

plt.figure(figsize=(8,4))
sns.barplot(data=split_sizes, x='split', y='rows', palette='viridis')
plt.title('Dataset Split Sizes')
plt.tight_layout()
plt.savefig('split_sizes.png', dpi=200, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 4) Traditional Machine Learning Models

# %%
def eval_binary(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'pr_auc': average_precision_score(y_true, y_prob)
    }

def plot_roc_pr(y_true, y_prob, title_prefix, out_prefix):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(1,2,figsize=(12,5))
    ax[0].plot(fpr, tpr, label=f'AUC={roc_auc_score(y_true, y_prob):.4f}')
    ax[0].plot([0,1],[0,1],'--')
    ax[0].set_title(f'{title_prefix} ROC')
    ax[0].set_xlabel('FPR'); ax[0].set_ylabel('TPR'); ax[0].legend()
    ax[1].plot(rec, prec, label=f'AP={average_precision_score(y_true, y_prob):.4f}')
    ax[1].set_title(f'{title_prefix} PR Curve')
    ax[1].set_xlabel('Recall'); ax[1].set_ylabel('Precision'); ax[1].legend()
    plt.tight_layout()
    plt.savefig(f'{out_prefix}_roc_pr.png', dpi=200, bbox_inches='tight')
    plt.show()

Xtr = X_train_smote.drop(columns=['Time','Amount'])
Xv = X_val_scaled.drop(columns=['Time','Amount'])
Xt = X_test_scaled.drop(columns=['Time','Amount'])
# use all features, already standardized Time/Amount retained by scaled versions
Xtr = X_train_smote
Xv = X_val_scaled
Xt = X_test_scaled

models = {}
models['LogisticRegression'] = LogisticRegression(max_iter=2000, class_weight=None, n_jobs=None)
models['DecisionTree'] = DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE, class_weight=None)
models['RandomForest'] = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
if XGB_AVAILABLE:
    models['XGBoost'] = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective='binary:logistic', eval_metric='logloss',
        random_state=RANDOM_STATE, n_jobs=-1
    )

results = []
preds = {}
for name, model in models.items():
    model.fit(Xtr, y_train_smote)
    prob = model.predict_proba(Xt)[:,1]
    preds[name] = prob
    m = eval_binary(y_test, prob)
    m['model'] = name
    results.append(m)
    plot_roc_pr(y_test, prob, name, name.lower())

# %% [markdown]
# ## 5) PyTorch datasets and models

# %%
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
            nn.Linear(16, 1)
        )
    def forward(self, x):
        x = x.unsqueeze(1)
        return self.net(x).squeeze(1)

class LSTMNet(nn.Module):
    def __init__(self, n_features, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True, num_layers=1)
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
            nn.Linear(n_features, 64), nn.ReLU(),
            nn.Linear(64, 16), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64), nn.ReLU(),
            nn.Linear(64, n_features)
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# %%
def train_classifier(model, train_loader, val_loader, epochs=6, lr=1e-3):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([len(y_train_smote)/max(y_train_smote.sum(),1)-1], device=DEVICE))
    history = {'train_loss':[], 'val_loss':[]}
    best = None
    best_val = float('inf')
    for epoch in range(epochs):
        model.train(); tl = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward(); opt.step()
            tl += loss.item() * len(xb)
        model.eval(); vl = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                loss = criterion(logits, yb)
                vl += loss.item() * len(xb)
        tl /= len(train_loader.dataset); vl /= len(val_loader.dataset)
        history['train_loss'].append(tl); history['val_loss'].append(vl)
        if vl < best_val:
            best_val = vl
            best = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        print(f'Epoch {epoch+1}: train={tl:.4f}, val={vl:.4f}')
    model.load_state_dict(best)
    return model, history

def predict_proba_torch(model, loader):
    model.eval(); probs=[]; ys=[]
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
    plt.figure(figsize=(6,4))
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'], label='val')
    plt.title(f'{name} Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.tight_layout()
    plt.savefig(f'{name.lower()}_loss.png', dpi=200, bbox_inches='tight')
    plt.show()

# %%
cnn, cnn_hist = train_classifier(CNN1D(X_train_smote.shape[1]), train_loader, val_loader, epochs=6, lr=1e-3)
plot_loss(cnn_hist, 'CNN')
y_true_cnn, cnn_prob = predict_proba_torch(cnn, test_loader)
results.append({'model':'PyTorch_CNN', **eval_binary(y_true_cnn, cnn_prob)})
preds['PyTorch_CNN'] = cnn_prob
plot_roc_pr(y_test, cnn_prob, 'PyTorch CNN', 'pytorch_cnn')

# %%
lstm, lstm_hist = train_classifier(LSTMNet(X_train_smote.shape[1]), train_loader, val_loader, epochs=6, lr=1e-3)
plot_loss(lstm_hist, 'LSTM')
y_true_lstm, lstm_prob = predict_proba_torch(lstm, test_loader)
results.append({'model':'PyTorch_LSTM', **eval_binary(y_true_lstm, lstm_prob)})
preds['PyTorch_LSTM'] = lstm_prob
plot_roc_pr(y_test, lstm_prob, 'PyTorch LSTM', 'pytorch_lstm')

# %% [markdown]
# ## 6) Autoencoder anomaly detection

# %%
def train_autoencoder(model, train_loader, val_loader, epochs=6, lr=1e-3):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    history = {'train_loss':[], 'val_loss':[]}
    best = None; best_val = float('inf')
    for epoch in range(epochs):
        model.train(); tl=0
        for xb, _ in train_loader:
            xb = xb.to(DEVICE)
            opt.zero_grad()
            recon = model(xb)
            loss = criterion(recon, xb)
            loss.backward(); opt.step()
            tl += loss.item() * len(xb)
        model.eval(); vl=0
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(DEVICE)
                recon = model(xb)
                loss = criterion(recon, xb)
                vl += loss.item() * len(xb)
        tl /= len(train_loader.dataset); vl /= len(val_loader.dataset)
        history['train_loss'].append(tl); history['val_loss'].append(vl)
        if vl < best_val:
            best_val = vl
            best = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        print(f'AE Epoch {epoch+1}: train={tl:.4f}, val={vl:.4f}')
    model.load_state_dict(best)
    return model, history

ae = AutoEncoder(X_train_smote.shape[1])
ae, ae_hist = train_autoencoder(ae, train_loader, val_loader, epochs=6, lr=1e-3)
plot_loss(ae_hist, 'Autoencoder')

def ae_scores(model, loader):
    model.eval(); errs=[]; ys=[]
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            recon = model(xb)
            err = torch.mean((recon - xb)**2, dim=1).cpu().numpy()
            errs.extend(err)
            ys.extend(yb.numpy())
    return np.array(ys), np.array(errs)

y_ae, err_test = ae_scores(ae, test_loader)
threshold = np.percentile(err_test, 95)
ae_prob = (err_test - err_test.min()) / (err_test.max() - err_test.min() + 1e-8)
ae_pred = (err_test >= threshold).astype(int)
results.append({
    'model':'Autoencoder',
    'accuracy': accuracy_score(y_test, ae_pred),
    'precision': precision_score(y_test, ae_pred, zero_division=0),
    'recall': recall_score(y_test, ae_pred, zero_division=0),
    'f1': f1_score(y_test, ae_pred, zero_division=0),
    'roc_auc': roc_auc_score(y_test, ae_prob),
    'pr_auc': average_precision_score(y_test, ae_prob)
})
preds['Autoencoder'] = ae_prob
plot_roc_pr(y_test, ae_prob, 'Autoencoder', 'autoencoder')

# %% [markdown]
# ## 7) Compare all models

# %%
results_df = pd.DataFrame(results).sort_values(by='pr_auc', ascending=False)
results_df.to_csv('model_comparison.csv', index=False)
print(results_df)

plt.figure(figsize=(12,5))
sns.barplot(data=results_df, x='model', y='pr_auc', palette='mako')
plt.xticks(rotation=30, ha='right')
plt.title('Model Comparison by PR-AUC')
plt.tight_layout()
plt.savefig('model_pr_auc_comparison.png', dpi=200, bbox_inches='tight')
plt.show()

plt.figure(figsize=(12,5))
sns.barplot(data=results_df, x='model', y='f1', palette='rocket')
plt.xticks(rotation=30, ha='right')
plt.title('Model Comparison by F1-score')
plt.tight_layout()
plt.savefig('model_f1_comparison.png', dpi=200, bbox_inches='tight')
plt.show()

# %%
for name, prob in preds.items():
    y_pred = (prob >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4.8,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'cm_{name.lower()}.png', dpi=200, bbox_inches='tight')
    plt.show()

# %% [markdown]
# ## 8) Save a final report

# %%
with open('final_notes.txt', 'w') as f:
    f.write(results_df.to_string(index=False))

print('Done. Files created: model_comparison.csv, plots, and final_notes.txt')
