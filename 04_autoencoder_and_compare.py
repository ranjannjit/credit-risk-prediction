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
