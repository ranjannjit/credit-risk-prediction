# %% [markdown]
# ## 8) Save a final report

# %%
import os
import pandas as pd
from pathlib import Path

base_dir = Path(__file__).parent
sources = []
for filename in [
    "model_metrics_summary.csv",
    "pytorch_model_metrics.csv",
    "autoencoder_metrics.csv",
    "model_comparison.csv",
]:
    path = base_dir / filename
    if path.exists():
        sources.append(path)

if not sources:
    raise FileNotFoundError(
        "No model metric CSV files found. Please run 02_traditional_ml.py, 03_pytorch_models.py, and 04_autoencoder_and_compare.py first."
    )

frames = []
for path in sources:
    df = pd.read_csv(path)
    if "model" not in df.columns:
        raise ValueError(f"Expected 'model' column in {path}")
    frames.append(df)

results_df = pd.concat(frames, ignore_index=True)
results_df = results_df.drop_duplicates(subset=["model"], keep="first")


def explain_best_model(df: pd.DataFrame) -> str:
    if df.empty:
        return "No model comparison data available."

    lines = []
    lines.append("Model comparison summary:")
    lines.append("-------------------------")

    primary_metric = "pr_auc"
    if primary_metric not in df.columns:
        primary_metric = "roc_auc" if "roc_auc" in df.columns else df.columns[0]

    top_pr = df.sort_values(primary_metric, ascending=False).iloc[0]
    lines.append(
        f"Best model by {primary_metric.upper()}: {top_pr['model']} ({top_pr[primary_metric]:.4f})"
    )

    if "roc_auc" in df.columns:
        top_roc = df.sort_values("roc_auc", ascending=False).iloc[0]
        lines.append(
            f"Best model by ROC-AUC: {top_roc['model']} ({top_roc['roc_auc']:.4f})"
        )
    else:
        top_roc = None

    if "f1" in df.columns:
        top_f1 = df.sort_values("f1", ascending=False).iloc[0]
        lines.append(f"Best model by F1 score: {top_f1['model']} ({top_f1['f1']:.4f})")
    else:
        top_f1 = None

    lines.append("")
    lines.append("Why this matters:")
    lines.append(
        "- PR-AUC is the most important metric for fraud detection because the dataset is highly imbalanced."
    )
    lines.append(
        "  It focuses on the model's ability to keep precision high while identifying positive cases."
    )
    if top_roc is not None:
        lines.append(
            "- ROC-AUC measures overall ranking quality. A strong ROC-AUC means the model generally separates fraud from normal cases well."
        )
    if top_f1 is not None:
        lines.append(
            "- F1 score balances precision and recall, which helps assess whether the model is a good compromise between false positives and false negatives."
        )

    if top_pr["model"] == top_roc["model"] if top_roc is not None else False:
        lines.append(
            f"\nThe best model overall is {top_pr['model']}, because it leads on both {primary_metric.upper()} and ROC-AUC."
        )
    elif top_f1 is not None and top_pr["model"] == top_f1["model"]:
        lines.append(
            f"\nThe most consistent model is {top_pr['model']}, because it performs best on both {primary_metric.upper()} and F1 score."
        )
    else:
        lines.append(
            f"\nThe recommendation is to use {top_pr['model']} as the primary model for fraud detection based on {primary_metric.upper()}."
        )

    if "precision" in df.columns and "recall" in df.columns:
        selected = top_pr
        precision = selected["precision"]
        recall = selected["recall"]
        lines.append(
            f"It has precision={precision:.4f} and recall={recall:.4f}, "
            f"so it {('leans toward high precision' if precision > recall else 'leans toward high recall')} "
            "for fraud detection."
        )

    lines.append("")
    lines.append("Detailed model ranking:")
    for _, row in df.sort_values(primary_metric, ascending=False).iterrows():
        metric_line = (
            f"  - {row['model']}: {primary_metric.upper()}={row[primary_metric]:.4f}"
        )
        if "roc_auc" in df.columns:
            metric_line += f", ROC-AUC={row['roc_auc']:.4f}"
        if "f1" in df.columns:
            metric_line += f", F1={row['f1']:.4f}"
        lines.append(metric_line)

    return "\n".join(lines)


explanation_text = explain_best_model(results_df)

with open("final_notes.txt", "w", encoding="utf-8") as f:
    f.write("MODEL COMPARISON:\n")
    f.write(results_df.to_string(index=False))
    f.write("\n\n")
    f.write(explanation_text)

print("Done. Files created: model_comparison.csv, plots, and final_notes.txt")
