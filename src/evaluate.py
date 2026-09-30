import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)
from src.preprocessing import get_feature_names_out

REPORTS_DIR = "reports"
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")


def evaluate_models(models: dict, X_test, y_test, preprocessor=None):
    """
    Evaluates all trained models, prints classification metrics, and exports comparison summary.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)

    metrics_list = []
    roc_data = {}
    cm_dict = {}

    print("\n" + "=" * 70)
    print(f"{'MODEL EVALUATION RESULTS':^70}")
    print("=" * 70)

    for name, model in models.items():
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.decision_function(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)

        metrics_list.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc
        })

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = (fpr, tpr, auc)
        cm_dict[name] = confusion_matrix(y_test, y_pred)

        print(f"\n--- Model: {name} ---")
        print(f"Accuracy:  {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["On-Time (0)", "Delayed (1)"]))

    summary_df = pd.DataFrame(metrics_list)
    csv_path = os.path.join(REPORTS_DIR, "metrics_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSaved metrics summary to '{csv_path}'.")

    # Visualizations
    plot_roc_curves(roc_data)
    plot_confusion_matrices(cm_dict)
    plot_metrics_comparison(summary_df)

    if preprocessor is not None:
        plot_feature_importances(models, preprocessor)

    return summary_df


def plot_roc_curves(roc_data: dict):
    """
    Plots overlaid ROC curves for all models.
    """
    plt.figure(figsize=(8, 6))
    colors = ["#2b5c8f", "#2ca02c", "#d62728"]

    for (name, (fpr, tpr, auc)), color in zip(roc_data.items(), colors):
        plt.plot(fpr, tpr, lw=2.2, label=f"{name} (AUC = {auc:.3f})", color=color)

    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.50)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11, fontweight="bold")
    plt.title("ROC Curves Comparison - Flight Delay Classification", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    out_file = os.path.join(FIGURES_DIR, "roc_curves.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved ROC curves figure to '{out_file}'.")


def plot_confusion_matrices(cm_dict: dict):
    """
    Plots confusion matrix side-by-side heatmaps for all models.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    labels = ["On-Time (0)", "Delayed (1)"]

    for ax, (name, cm) in zip(axes, cm_dict.items()):
        cm_percent = cm.astype('float') / cm.sum() * 100
        annot = np.array([
            [f"{count:,}\n({pct:.1f}%)" for count, pct in zip(row_c, row_p)]
            for row_c, row_p in zip(cm, cm_percent)
        ])

        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=labels,
            yticklabels=labels,
            annot_kws={"size": 11, "weight": "bold"}
        )
        ax.set_title(f"{name}", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted Label", fontsize=10, fontweight="bold")
        ax.set_ylabel("Actual Label", fontsize=10, fontweight="bold")

    plt.suptitle("Confusion Matrix Comparison Across Models", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_file = os.path.join(FIGURES_DIR, "confusion_matrices.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrices figure to '{out_file}'.")


def plot_metrics_comparison(summary_df: pd.DataFrame):
    """
    Plots bar chart comparing metrics across all models.
    """
    melted_df = summary_df.melt(
        id_vars=["Model"],
        value_vars=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
        var_name="Metric",
        value_name="Score"
    )

    plt.figure(figsize=(10, 6))
    palette = ["#1f77b4", "#2ca02c", "#d62728"]
    ax = sns.barplot(
        data=melted_df,
        x="Metric",
        y="Score",
        hue="Model",
        palette=palette,
        edgecolor="black",
        linewidth=0.8
    )

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.2f}",
                (p.get_x() + p.get_width() / 2., height),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                xytext=(0, 3),
                textcoords="offset points"
            )

    plt.ylim(0, 1.15)
    plt.title("Performance Metrics Comparison (Logistic Regression vs Random Forest vs Gradient Boosting)", fontsize=12, fontweight="bold", pad=12)
    plt.ylabel("Score (0.0 to 1.0)", fontsize=11, fontweight="bold")
    plt.xlabel("Evaluation Metric", fontsize=11, fontweight="bold")
    plt.legend(title="Algorithm", loc="upper left", frameon=True)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()

    out_file = os.path.join(FIGURES_DIR, "metrics_comparison.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved metrics comparison figure to '{out_file}'.")


def plot_feature_importances(models: dict, preprocessor):
    """
    Plots top feature importances for Random Forest and Gradient Boosting.
    """
    try:
        feature_names = get_feature_names_out(preprocessor)
        clean_names = [f.replace("num__", "").replace("cat__", "") for f in feature_names]

        tree_models = {
            k: v for k, v in models.items()
            if hasattr(v, "feature_importances_")
        }

        if not tree_models:
            return

        fig, axes = plt.subplots(1, len(tree_models), figsize=(7 * len(tree_models), 7), sharey=False)
        if len(tree_models) == 1:
            axes = [axes]

        for ax, (name, model) in zip(axes, tree_models.items()):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:15]

            top_names = [clean_names[i] for i in indices]
            top_scores = importances[indices]

            sns.barplot(
                x=top_scores,
                y=top_names,
                ax=ax,
                hue=top_names,
                palette="viridis",
                legend=False,
                edgecolor="black",
                linewidth=0.5
            )
            ax.set_title(f"{name} Top 15 Features", fontsize=12, fontweight="bold")
            ax.set_xlabel("Relative Importance", fontsize=10, fontweight="bold")
            ax.grid(axis="x", linestyle=":", alpha=0.6)

        plt.suptitle("Feature Importance Ranking", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()

        out_file = os.path.join(FIGURES_DIR, "feature_importance.png")
        plt.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved feature importances figure to '{out_file}'.")
    except Exception as e:
        print(f"Warning: Could not plot feature importance: {e}")
