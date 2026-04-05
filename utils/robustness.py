# robustness uses logistic baseline for drop/noise evaluations
import os
import numpy as np

try:
    import torch
except ImportError:
    torch = None
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score


def _predict_proba(model, X, model_type="sklearn", device=None):
    if model_type == "sklearn":
        return model.predict_proba(X)[:, 1]

    if model_type == "rnn":
        if torch is None:
            raise ImportError("torch is required for RNN model_type")
        device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        model.eval()
        model.to(device)

        with torch.no_grad():
            xt = torch.tensor(X, dtype=torch.float32).to(device)
            logits = model(xt)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    raise ValueError(f"Unknown model type: {model_type}")


def adversarial_test(
    model, X_test, y_test, noise_std=0.02, model_type="sklearn", device=None
):
    """Evaluate model AUC on test set with additive Gaussian noise."""
    noise = np.random.normal(0, noise_std, X_test.shape)
    X_noisy = X_test + noise

    y_prob = _predict_proba(model, X_noisy, model_type=model_type, device=device)
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "roc_auc": roc_auc_score(y_test, y_prob),
        "accuracy": accuracy_score(y_test, y_pred),
    }


def noise_robustness_curve(
    model,
    X_test,
    y_test,
    std_multiples=[0.01, 0.02, 0.05, 0.1],
    model_type="sklearn",
    device=None,
):
    """Compute AUC/accuracy for varying levels of gaussian input noise."""
    results = {}
    for std in std_multiples:
        out = adversarial_test(
            model,
            X_test,
            y_test,
            noise_std=std,
            model_type=model_type,
            device=device,
        )
        results[std] = out
    return results


def feature_dropout_test(
    model,
    X_test,
    y_test,
    drop_fraction=0.1,
    n_rounds=10,
    random_state=42,
    model_type="sklearn",
    device=None,
):
    """Randomly zero out a portion of input features and evaluate accuracy/AUC."""
    rng = np.random.default_rng(random_state)
    n_features = X_test.shape[1]
    n_drop = max(1, int(n_features * drop_fraction))
    auc_scores = []
    acc_scores = []

    for _ in range(n_rounds):
        dropped = rng.choice(n_features, size=n_drop, replace=False)
        X_dropped = X_test.copy()
        X_dropped[:, dropped] = 0

        y_prob = _predict_proba(model, X_dropped, model_type=model_type, device=device)
        y_pred = (y_prob >= 0.5).astype(int)

        auc_scores.append(roc_auc_score(y_test, y_prob))
        acc_scores.append(accuracy_score(y_test, y_pred))

    return {
        "avg_roc_auc": float(np.mean(auc_scores)),
        "avg_accuracy": float(np.mean(acc_scores)),
        "std_roc_auc": float(np.std(auc_scores)),
        "std_accuracy": float(np.std(acc_scores)),
    }


def plot_noise_robustness(noise_curve, out_dir="."):
    """Plot ROC AUC vs noise standard deviation."""
    stds = sorted(noise_curve.keys())
    aucs = [noise_curve[s]["roc_auc"] for s in stds]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(stds, aucs, marker="o", linestyle="-", color="blue")
    ax.set_xlabel("Noise Standard Deviation")
    ax.set_ylabel("ROC AUC")
    ax.set_title("Robustness to Input Noise")
    ax.set_ylim(0, 1)
    ax.grid(True)
    fig.tight_layout()

    fig_path = os.path.join(out_dir, "noise_robustness_plot.png")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Saved noise robustness plot: {fig_path}")


def plot_dropout_robustness(dropout_stats, out_dir="."):
    """Plot average ROC AUC and accuracy for feature dropout."""
    metrics = ["avg_roc_auc", "avg_accuracy"]
    values = [dropout_stats[m] for m in metrics]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(metrics, values, color=["blue", "green"])
    ax.set_ylabel("Value")
    ax.set_title("Feature Dropout Robustness")
    ax.set_ylim(0, 1)
    fig.tight_layout()

    fig_path = os.path.join(out_dir, "dropout_robustness_plot.png")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Saved dropout robustness plot: {fig_path}")


if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier

    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    print("Adversarial test:", adversarial_test(model, X, y))
    noise_curve = noise_robustness_curve(model, X, y)
    print("Noise robustness:", noise_curve)
    dropout_stats = feature_dropout_test(model, X, y)
    print("Feature dropout:", dropout_stats)

    # Plot
    plot_noise_robustness(noise_curve)
    plot_dropout_robustness(dropout_stats)
