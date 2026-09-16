import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

np.random.seed(42)
os.makedirs("figuras", exist_ok=True)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# datos
# ---------------------------------------------------------------------------
df = pd.read_csv("telco_churn_processed.csv")

y = df["Churn"].to_numpy(dtype=float)
X = df.drop(columns=["Churn"]).to_numpy(dtype=float)
feature_names = df.drop(columns=["Churn"]).columns.tolist()

# ---------------------------------------------------------------------------
# train / validation / test split (60% / 20% / 20%), idéntico a
# telco_framework.py, misma semilla, para que los modelos sean comparables.
# ---------------------------------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)


# ---------------------------------------------------------------------------
# métricas manuales (mismas funciones que telco_framework.py)
# ---------------------------------------------------------------------------
def confusion_matrix_manual(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp, tn, fp, fn


def compute_metrics(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix_manual(y_true, y_pred)
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def metrics_report(y_true, y_pred, label=""):
    m = compute_metrics(y_true, y_pred)
    print(f"Matriz de confusión{f' ({label})' if label else ''}:")
    print(f"                pred_0   pred_1")
    print(f"  real_0        {m['tn']:6d}   {m['fp']:6d}")
    print(f"  real_1        {m['fn']:6d}   {m['tp']:6d}")
    print(f"\nAccuracy : {m['accuracy']:.4f}")
    print(f"Precision: {m['precision']:.4f}")
    print(f"Recall   : {m['recall']:.4f}")
    print(f"F1-score : {m['f1']:.4f}")
    return m


# ---------------------------------------------------------------------------
# entrenamiento de las tres variantes
# ---------------------------------------------------------------------------
configs = {
    "Sin ponderar (default)": None,
    "class_weight=balanced": "balanced",
    "class_weight=balanced_subsample": "balanced_subsample",
}

models = {}
for name, cw in configs.items():
    section(f"ENTRENAMIENTO — {name}")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        criterion="entropy",
        class_weight=cw,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    models[name] = model
    print(f"Número de árboles: {model.n_estimators}")
    print(f"Profundidad máxima: {model.max_depth}")
    print(f"class_weight: {cw}")

# ---------------------------------------------------------------------------
# evaluación de las tres variantes sobre train / validation / test
# ---------------------------------------------------------------------------
results = {"train": [], "val": [], "test": []}
splits = {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}

for name, model in models.items():
    for split_name, (X_split, y_split) in splits.items():
        y_pred = model.predict(X_split)
        m = compute_metrics(y_split, y_pred)
        results[split_name].append({"modelo": name, **m})

for split_name in ["train", "val", "test"]:
    section(f"RESUMEN COMPARATIVO — {split_name.upper()}")
    summary = pd.DataFrame(results[split_name])[
        ["modelo", "accuracy", "precision", "recall", "f1"]
    ]
    print(summary.to_string(index=False))

section("DETALLE — EVALUACIÓN EN TEST (matrices de confusión)")
for name, model in models.items():
    y_pred_test = model.predict(X_test)
    metrics_report(y_test, y_pred_test, label=name)
    print()

# ---------------------------------------------------------------------------
# gráfica: comparación de métricas en test entre las tres variantes
# ---------------------------------------------------------------------------
test_summary = pd.DataFrame(results["test"]).set_index("modelo")
metric_cols = ["accuracy", "precision", "recall", "f1"]

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(metric_cols))
width = 0.25
colors = ["steelblue", "darkorange", "firebrick"]

for i, (name, row) in enumerate(test_summary.iterrows()):
    ax.bar(x + (i - 1) * width, row[metric_cols], width, label=name, color=colors[i])

ax.set_xticks(x)
ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1-score"])
ax.set_ylim(0, 1)
ax.set_title("Comparación de métricas en test — efecto de class_weight (Random Forest)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("figuras/class_weight_comparacion_metricas.png", dpi=150, bbox_inches="tight")

# ---------------------------------------------------------------------------
# gráfica: matrices de confusión lado a lado (default vs. balanced vs.
# balanced_subsample)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, name in zip(axes, ["Sin ponderar (default)", "class_weight=balanced", "class_weight=balanced_subsample"]):
    y_pred_test = models[name].predict(X_test)
    tp, tn, fp, fn = confusion_matrix_manual(y_test, y_pred_test)
    cm = np.array([[tn, fp], [fn, tp]])

    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred: No Churn", "Pred: Churn"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Real: No Churn", "Real: Churn"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    ax.set_title(name, fontsize=10)

fig.suptitle("Matriz de confusión (test) — efecto de class_weight")
fig.tight_layout()
fig.savefig("figuras/class_weight_matrices_confusion.png", dpi=150, bbox_inches="tight")

plt.show()
