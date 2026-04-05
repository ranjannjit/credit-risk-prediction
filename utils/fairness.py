# uses these outputs to compute equality metrics by sensitive axes
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import recall_score, confusion_matrix


def fairness_by_group(X, y_true, y_pred, sensitive_feature, feature_names):
    """Compute group-level recall and approval rate by sensitive group."""
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y_true
    df["pred"] = y_pred

    groups = df.groupby(sensitive_feature)
    results = {}

    for group, data in groups:
        recall = recall_score(data["target"], data["pred"])
        approval_rate = (data["pred"] == 0).mean()
        results[group] = {
            "recall": recall,
            "approval_rate": approval_rate,
            "support": len(data),
        }

    return results


def statistical_parity_difference(y_pred, sensitive_vals, positive_label=1):
    """Difference in selection rate between unprivileged and privileged groups."""
    groups = pd.Series(sensitive_vals)
    y = pd.Series(y_pred)

    privileged = y[groups == groups.mode().iloc[0]]
    unprivileged = y[groups != groups.mode().iloc[0]]

    p_priv = (privileged == positive_label).mean()
    p_unpriv = (unprivileged == positive_label).mean()
    return p_unpriv - p_priv


def disparate_impact(y_pred, sensitive_vals, positive_label=1):
    """Ratio of positive outcome rates (unprivileged/privileged)."""
    groups = pd.Series(sensitive_vals)
    y = pd.Series(y_pred)

    privileged = y[groups == groups.mode().iloc[0]]
    unprivileged = y[groups != groups.mode().iloc[0]]

    p_priv = (privileged == positive_label).mean()
    p_unpriv = (unprivileged == positive_label).mean()

    if p_priv == 0:
        return np.nan
    return p_unpriv / p_priv


def equal_opportunity_difference(y_true, y_pred, sensitive_vals, positive_label=1):
    """Difference in true positive rates between groups."""
    df = pd.DataFrame({"target": y_true, "pred": y_pred, "group": sensitive_vals})

    def tpr(group_df):
        cm = confusion_matrix(group_df["target"], group_df["pred"], labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        if tp + fn == 0:
            return np.nan
        return tp / (tp + fn)

    groups = df["group"].unique()
    if len(groups) < 2:
        return 0.0

    tprs = [tpr(df[df["group"] == g]) for g in groups]
    return max(tprs) - min(tprs)


def summarize_fairness(X, y_true, y_pred, sensitive_feature, feature_names):
    """Full fairness summary with group metrics and classic fairness stats."""
    group_metrics = fairness_by_group(
        X, y_true, y_pred, sensitive_feature, feature_names
    )

    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y_true
    df["pred"] = y_pred

    sensitive_vals = sensitive_feature
    spd = statistical_parity_difference(y_pred, sensitive_vals, positive_label=1)
    di = disparate_impact(y_pred, sensitive_vals, positive_label=1)
    eod = equal_opportunity_difference(y_true, y_pred, sensitive_vals, positive_label=1)

    return {
        "group_metrics": group_metrics,
        "statistical_parity_difference": spd,
        "disparate_impact": di,
        "equal_opportunity_difference": eod,
    }


def plot_group_fairness(group_metrics, out_dir="."):
    """Plot group-level recall and approval rate."""
    groups = list(group_metrics.keys())
    recall = [group_metrics[g]["recall"] for g in groups]
    approval = [group_metrics[g]["approval_rate"] for g in groups]

    x = np.arange(len(groups))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, recall, width, label="Recall")
    ax.bar(x + width / 2, approval, width, label="Approval Rate")

    ax.set_xlabel("Group")
    ax.set_ylabel("Value")
    ax.set_title("Group-level Fairness Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()

    fig_path = os.path.join(out_dir, "group_fairness_plot.png")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Saved group fairness plot: {fig_path}")


def plot_overall_fairness(fairness_summary, out_dir="."):
    """Plot overall fairness metrics: SPD, DI, EOD."""
    metrics = [
        "statistical_parity_difference",
        "disparate_impact",
        "equal_opportunity_difference",
    ]
    values = [fairness_summary[m] for m in metrics]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(metrics, values, color=["blue", "green", "red"])

    ax.set_ylabel("Value")
    ax.set_title("Overall Fairness Metrics")
    ax.set_ylim(-1, 1)  # SPD can be negative, DI >0, EOD >=0
    fig.tight_layout()

    fig_path = os.path.join(out_dir, "overall_fairness_plot.png")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Saved overall fairness plot: {fig_path}")


if __name__ == "__main__":
    # Quick demo using synthetic data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    sensitive_feature = np.random.choice(["A", "B"], size=100)
    feature_names = [f"f{i}" for i in range(5)]
    result = fairness_by_group(X, y_true, y_pred, sensitive_feature, feature_names)
    print("fairness_by_group summary:", result)
    print(
        "statistical_parity_difference:",
        statistical_parity_difference(y_pred, sensitive_feature),
    )
    print("disparate_impact:", disparate_impact(y_pred, sensitive_feature))
    print(
        "equal_opportunity_difference:",
        equal_opportunity_difference(y_true, y_pred, sensitive_feature),
    )

    # Full summary
    summary = summarize_fairness(X, y_true, y_pred, sensitive_feature, feature_names)
    print("Full fairness summary:", summary)

    # Plot
    plot_group_fairness(summary["group_metrics"])
    plot_overall_fairness(summary)
