import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

models = {
    "Logistic Regression": "metrics/lr_metrics.json",
    "Linear SVM": "metrics/svm_metrics.json",
    "Naive Bayes": "metrics/nb_metrics.json"
}

metrics_data = {}
for name, path in models.items():
    with open(path) as f:
        metrics_data[name] = json.load(f)

# --------------------------------------------------
# BAR CHART: Accuracy / F1
# --------------------------------------------------
labels = list(metrics_data.keys())
accuracy = [metrics_data[m]["accuracy"] for m in labels]
f1 = [metrics_data[m]["f1"] for m in labels]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(7, 5))
plt.bar(x - width/2, accuracy, width, label="Accuracy")
plt.bar(x + width/2, f1, width, label="F1-score")

plt.xticks(x, labels)
plt.ylabel("Score")
plt.title("Model Comparison")
plt.legend()

plt.tight_layout()
plt.savefig("compare_img/model_comparison.png", dpi=300)
plt.close()

# --------------------------------------------------
# CONFUSION MATRIX COMPARISON
# --------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for ax, (name, data) in zip(axes, metrics_data.items()):
    sns.heatmap(
        np.array(data["confusion_matrix"]),
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig("compare_img/confusion_matrix_comparison.png", dpi=300)
plt.close()
