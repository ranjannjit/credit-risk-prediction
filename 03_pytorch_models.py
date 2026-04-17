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