# python -m utils.evaluation
"""# Default: 100,000 rows
python -m utils.evaluation

# Custom number of rows
python -m utils.evaluation --nrows 50000

# From preprocessing script
python utils/preprocessing.py --nrows 200000
"""

"""Evaluation pipeline for model performance, fairness, and robustness analyses."""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from utils.preprocessing import load_and_preprocess
from utils.fairness import summarize_fairness
from utils.robustness import noise_robustness_curve, feature_dropout_test

if TORCH_AVAILABLE:

    class TabularRNN(nn.Module):
        """Very simple RNN-based binary classifier over feature sequences."""

        def __init__(
            self,
            n_features,
            hidden_size=64,
            num_layers=1,
            bidirectional=False,
            dropout=0.2,
        ):
            super(TabularRNN, self).__init__()
            self.n_features = n_features
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.bidirectional = bidirectional

            self.rnn = nn.LSTM(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            direction_factor = 2 if bidirectional else 1
            self.fc = nn.Linear(hidden_size * direction_factor, 1)

        def forward(self, x):
            # x shape: [batch, n_features]
            x = x.unsqueeze(-1).float()  # [batch, seq_len, 1]
            output, _ = self.rnn(x)
            last_output = output[:, -1, :]
            out = self.fc(last_output)
            return out.squeeze(1)

    def train_rnn_model(
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        epochs=20,
        batch_size=256,
        lr=1e-3,
        device=None,
    ):
        # sklearn / pandas compatibility: convert DataFrame/Series to np.ndarray
        def _to_numpy(x):
            if x is None:
                return None
            if hasattr(x, "values"):
                return np.asarray(x.values, dtype=np.float32)
            return np.asarray(x, dtype=np.float32)

        X_train = _to_numpy(X_train)
        y_train = _to_numpy(y_train).ravel()
        X_val = _to_numpy(X_val)
        y_val = _to_numpy(y_val).ravel() if y_val is not None else None

        device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        train_tensor = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)

        n_features = X_train.shape[1]
        model = TabularRNN(n_features=n_features).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_val_auc = 0.0
        best_model = None

        val_loader = None
        if X_val is not None and y_val is not None:
            val_tensor = TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.float32),
            )
            val_loader = DataLoader(val_tensor, batch_size=batch_size, shuffle=False)

        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}/{epochs} - Training...")
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            if val_loader is not None:
                print(f"Epoch {epoch}/{epochs} - Validating...")
                model.eval()
                all_logits, all_targets = [], []
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        logits = model(xb)
                        all_logits.append(logits.detach().cpu())
                        all_targets.append(yb.detach().cpu())
                    all_logits = torch.cat(all_logits)
                    all_targets = torch.cat(all_targets)
                    all_probs = torch.sigmoid(all_logits).numpy()
                    val_auc = roc_auc_score(all_targets.numpy(), all_probs)
                    if val_auc > best_val_auc:
                        best_val_auc = val_auc
                        best_model = model.state_dict().copy()
                print(f"Epoch {epoch}/{epochs} - Val AUC: {val_auc:.4f}")

        print("Training complete. Evaluating on training set...")

        # final eval on train set for consistency
        model.eval()
        all_targets = []
        all_probs = []
        all_preds = []
        with torch.no_grad():
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs >= 0.5).astype(int)
                all_probs.extend(probs.tolist())
                all_preds.extend(preds.tolist())
                all_targets.extend(yb.cpu().numpy().tolist())

        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)

        return {
            "model": model,
            "accuracy": accuracy_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_prob),
            "y_pred": y_pred,
            "y_prob": y_prob,
            "y_true": y_true,
        }


def evaluate_rnn_model(model, X_test, y_test, batch_size=256, device=None):
    """Evaluate trained RNN model on test data."""

    def _to_numpy(x):
        if hasattr(x, "values"):
            return np.asarray(x.values, dtype=np.float32)
        return np.asarray(x, dtype=np.float32)

    X_test = _to_numpy(X_test)
    y_test = _to_numpy(y_test).ravel()

    device = device or (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    test_tensor = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )
    test_loader = DataLoader(test_tensor, batch_size=batch_size, shuffle=False)

    model.to(device)
    model.eval()

    all_targets = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_targets.extend(yb.cpu().numpy().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    return {
        "model": model,
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "y_pred": y_pred,
        "y_prob": y_prob,
        "y_true": y_true,
    }


def run_baseline_logistic(X_train, y_train, X_test, y_test):
    lr = LogisticRegression(max_iter=100, class_weight="balanced", solver="liblinear")
    lr.fit(X_train, y_train)

    y_prob = lr.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "model": lr,
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True
        ),
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def create_report_csv(results, report_path):
    data = []

    # collect in long format
    for group, metrics in results.items():
        row = {"category": group}
        row.update(metrics)
        data.append(row)

    report_df = pd.DataFrame(data)
    report_df.to_csv(report_path, index=False)


def pipeline(
    csv_path,
    sensitive_feature="home_ownership",
    report_path="evaluation_report.csv",
    nrows=1000,
    epochs=25,
):
    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        features,
        all_columns,
        numeric_cols,
        categorical_cols,
        pre_dummies_df,
        train_idx,
        test_idx,
    ) = load_and_preprocess(csv_path, nrows=nrows)

    if TORCH_AVAILABLE:
        gpu_available = torch.cuda.is_available()
        print(f"Hardware status: PyTorch installed")
        print(f"  CUDA available: {gpu_available}")
        if gpu_available:
            print(f"  GPU count: {torch.cuda.device_count()}")
            print(f"  GPU name: {torch.cuda.get_device_name(0)}")
        else:
            print("  Running on CPU")
    else:
        print(
            "Hardware status: PyTorch not installed; running only LogisticRegression on CPU"
        )

    print("Pipeline: baseline LogisticRegression")
    lr_stats = run_baseline_logistic(X_train, y_train, X_test, y_test)

    if TORCH_AVAILABLE:
        print("Pipeline: training TabularRNN")
        # For RNN, keep 10% of train as internal validation
        X_subtrain, X_val, y_subtrain, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42
        )
        rnn_train_stats = train_rnn_model(
            X_subtrain, y_subtrain, X_val, y_val, epochs=epochs
        )
        rnn_model = rnn_train_stats["model"]
        print("Pipeline: evaluating TabularRNN on test set")
        rnn_stats = evaluate_rnn_model(rnn_model, X_test, y_test)
    else:
        print("Pipeline: PyTorch not available, skipping RNN")
        rnn_stats = None
        rnn_model = None

    # Fairness analysis on aligned raw subset
    raw_test = pre_dummies_df.iloc[test_idx]
    feature_names = raw_test.drop(columns=["loan_status"]).columns.tolist()

    print("Pipeline: fairness summary")
    sensitive_vals = raw_test[sensitive_feature].values

    if rnn_stats is not None:
        fairness_summary = summarize_fairness(
            raw_test.drop(columns=["loan_status"]).values,
            raw_test["loan_status"].values,
            rnn_stats["y_pred"],
            sensitive_vals,
            feature_names,
        )
    else:
        fairness_summary = summarize_fairness(
            raw_test.drop(columns=["loan_status"]).values,
            raw_test["loan_status"].values,
            lr_stats["y_pred"],
            sensitive_vals,
            feature_names,
        )

    print("Pipeline: robustness analysis")
    robustness_stats = {
        "logistic": {
            "noise_curve": noise_robustness_curve(
                lr_stats["model"], X_test, y_test, model_type="sklearn"
            ),
            "feature_dropout": feature_dropout_test(
                lr_stats["model"], X_test, y_test, model_type="sklearn"
            ),
        },
    }
    if rnn_stats is not None:
        robustness_stats["rnn"] = {
            "noise_curve": noise_robustness_curve(
                rnn_model, X_test, y_test, model_type="rnn"
            ),
            "feature_dropout": feature_dropout_test(
                rnn_model, X_test, y_test, model_type="rnn"
            ),
        }

    # create report rows
    report = {
        "logistic": {
            "accuracy": lr_stats["accuracy"],
            "roc_auc": lr_stats["roc_auc"],
        },
    }
    if rnn_stats is not None:
        report["rnn"] = {
            "accuracy": rnn_stats["accuracy"],
            "roc_auc": rnn_stats["roc_auc"],
        }

    report["fairness"] = {
        "statistical_parity_difference": fairness_summary[
            "statistical_parity_difference"
        ],
        "disparate_impact": fairness_summary["disparate_impact"],
        "equal_opportunity_difference": fairness_summary[
            "equal_opportunity_difference"
        ],
    }
    report["robustness"] = robustness_stats

    # Save a simplified CSV for top-level metrics
    scalar_records = [
        {
            "model": "baseline_logistic",
            "accuracy": report["logistic"]["accuracy"],
            "roc_auc": report["logistic"]["roc_auc"],
        },
    ]
    if rnn_stats is not None:
        scalar_records.append(
            {
                "model": "tabular_rnn",
                "accuracy": report["rnn"]["accuracy"],
                "roc_auc": report["rnn"]["roc_auc"],
            }
        )
    pd.DataFrame(scalar_records).to_csv(report_path, index=False)

    # Save full fairness report for group values into a detailed CSV too
    fairness_group = fairness_summary["group_metrics"]
    if fairness_group:
        df_group = (
            pd.DataFrame.from_dict(fairness_group, orient="index")
            .reset_index()
            .rename(columns={"index": sensitive_feature})
        )
        df_group.to_csv("fairness_group_report.csv", index=False)

    print(f"Report files written: {report_path}, fairness_group_report.csv")

    try:
        plot_model_metrics(lr_stats, rnn_stats, ".")
        plot_robustness_curves(robustness_stats, ".")
    except Exception as ex:
        print(f"Plotting failed: {ex}")

    return {
        "logistic": lr_stats,
        "rnn": rnn_stats,
        "fairness": fairness_summary,
        "robustness": robustness_stats,
        "report": report,
    }


def plot_model_metrics(lr_stats, rnn_stats, out_dir="."):
    labels = ["Logistic"]
    accuracy = [lr_stats["accuracy"]]
    auc = [lr_stats["roc_auc"]]

    if rnn_stats is not None:
        labels.append("RNN")
        accuracy.append(rnn_stats["accuracy"])
        auc.append(rnn_stats["roc_auc"])

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, accuracy, width, label="Accuracy")
    ax.bar(x + width / 2, auc, width, label="ROC AUC")

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()

    fig_path = os.path.join(out_dir, "model_performance_comparison.png")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Saved model comparison plot: {fig_path}")


def plot_robustness_curves(robustness_stats, out_dir="."):
    # noise curve for logistic and rnn
    fig, ax = plt.subplots(figsize=(8, 5))

    model_types = list(robustness_stats.keys())
    for model_type in model_types:
        noise_curve = robustness_stats[model_type]["noise_curve"]
        stds = sorted(noise_curve.keys())
        aucs = [noise_curve[s]["roc_auc"] for s in stds]
        ax.plot(stds, aucs, marker="o", label=f"{model_type} ROC AUC")

    ax.set_xlabel("Noise std")
    ax.set_ylabel("ROC AUC")
    ax.set_title("Robustness to input noise")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig_path_noise = os.path.join(out_dir, "robustness_noise_curve.png")
    fig.savefig(fig_path_noise)
    plt.close(fig)
    print(f"Saved noise robustness plot: {fig_path_noise}")

    # dropout curve for logistic and rnn
    fig, ax = plt.subplots(figsize=(8, 5))
    model_names = list(robustness_stats.keys())
    avg_aucs = [
        robustness_stats[m]["feature_dropout"]["avg_roc_auc"] for m in model_names
    ]
    avg_accs = [
        robustness_stats[m]["feature_dropout"]["avg_accuracy"] for m in model_names
    ]

    x = np.arange(len(model_names))
    width = 0.35
    ax.bar(x - width / 2, avg_aucs, width, label="Avg ROC AUC")
    ax.bar(x + width / 2, avg_accs, width, label="Avg Accuracy")

    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in model_names])
    ax.set_ylabel("Value")
    ax.set_title("Feature dropout robustness comparison")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig_path_drop = os.path.join(out_dir, "robustness_dropout_comparison.png")
    fig.savefig(fig_path_drop)
    plt.close(fig)
    print(f"Saved dropout robustness plot: {fig_path_drop}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run model + fairness + robustness evaluation pipeline"
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "lending_club.csv"
        ),
    )
    parser.add_argument("--sensitive_feature", type=str, default="home_ownership")
    parser.add_argument("--report_path", type=str, default="evaluation_report.csv")
    parser.add_argument("--nrows", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=25)

    args = parser.parse_args()

    results = pipeline(
        args.csv_path,
        sensitive_feature=args.sensitive_feature,
        report_path=args.report_path,
        nrows=args.nrows,
        epochs=args.epochs,
    )
    print("Done.")
