import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree

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
# train / validation / test split (60% / 20% / 20%), misma proporción y
# misma semilla que telco_modelo_manual.py, usando train_test_split de sklearn
# ---------------------------------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# ---------------------------------------------------------------------------
# Random Forest con scikit-learn
# ---------------------------------------------------------------------------
section("ENTRENAMIENTO")
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=5,
    criterion="entropy",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)
print(f"Número de árboles: {model.n_estimators}")
print(f"Profundidad máxima: {model.max_depth}")
print(f"Criterio de división: {model.criterion}")

# ---------------------------------------------------------------------------
# métricas manuales (mismas funciones que telco_modelo_manual.py, para
# comparar resultados en igualdad de condiciones)
# ---------------------------------------------------------------------------
def confusion_matrix_manual(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp, tn, fp, fn


def metrics_report(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix_manual(y_true, y_pred)
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"Matriz de confusión:")
    print(f"                pred_0   pred_1")
    print(f"  real_0        {tn:6d}   {fp:6d}")
    print(f"  real_1        {fn:6d}   {tp:6d}")
    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    return accuracy, precision, recall, f1


section("EVALUACIÓN EN TEST")
y_pred_test = model.predict(X_test)
test_accuracy, test_precision, test_recall, test_f1 = metrics_report(y_test, y_pred_test)

section("EVALUACIÓN EN VALIDATION")
y_pred_val = model.predict(X_val)
metrics_report(y_val, y_pred_val)

section("EVALUACIÓN EN TRAIN (referencia)")
y_pred_train = model.predict(X_train)
metrics_report(y_train, y_pred_train)

# ---------------------------------------------------------------------------
# importancia de variables (feature_importances_: reducción promedio de
# impureza aportada por cada variable en todos los árboles del bosque)
# ---------------------------------------------------------------------------
section("IMPORTANCIA DE VARIABLES (ordenadas por magnitud)")
importances = model.feature_importances_

imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
imp_df = imp_df.sort_values("importance", ascending=False)
print(imp_df.to_string(index=False))

# ---------------------------------------------------------------------------
# gráfica: matriz de confusión (conjunto de test)
# ---------------------------------------------------------------------------
tp, tn, fp, fn = confusion_matrix_manual(y_test, y_pred_test)
cm = np.array([[tn, fp], [fn, tp]])

fig, ax = plt.subplots(figsize=(5, 5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Pred: No Churn", "Pred: Churn"])
ax.set_yticks([0, 1])
ax.set_yticklabels(["Real: No Churn", "Real: Churn"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
ax.set_title("Matriz de confusión (test) - Random Forest")
fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig("figuras/matriz_confusion_rf.png", dpi=150, bbox_inches="tight")

# ---------------------------------------------------------------------------
# gráfica: importancia de variables
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 10))
ax.barh(imp_df["feature"], imp_df["importance"], color="steelblue")
ax.invert_yaxis()
ax.set_title("Importancia de variables - Random Forest")
fig.tight_layout()
fig.savefig("figuras/importancia_rf.png", dpi=150, bbox_inches="tight")

# ---------------------------------------------------------------------------
# gráfica: visualización de un árbol individual del bosque
# ---------------------------------------------------------------------------
# Se grafica solo el primer árbol del ensamble, limitando la profundidad
# mostrada (max_depth=3) para que la figura sea legible; el árbol real
# entrenado tiene max_depth=10.
fig, ax = plt.subplots(figsize=(24, 12))
plot_tree(
    model.estimators_[0],
    max_depth=3,        
    feature_names=feature_names,
    class_names=["No Churn", "Churn"],
    filled=True,
    rounded=True,
    fontsize=8,
    ax=ax,
)
ax.set_title("Primer árbol del Random Forest (mostrando solo los primeros 3 niveles)")
fig.tight_layout()
fig.savefig("figuras/arbol_ejemplo_rf.png", dpi=150, bbox_inches="tight")
