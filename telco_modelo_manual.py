import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
# train / validation / test split manual (60% / 20% / 20%)
# ---------------------------------------------------------------------------
n = len(y)
idx = np.random.permutation(n)
test_size = int(n * 0.2)
val_size = int(n * 0.2)

test_idx = idx[:test_size]
val_idx = idx[test_size:test_size + val_size]
train_idx = idx[test_size + val_size:]

X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

# ---------------------------------------------------------------------------
# estandarización (fit solo con train)
# ---------------------------------------------------------------------------
mu = X_train.mean(axis=0)
sigma = X_train.std(axis=0)
sigma[sigma == 0] = 1.0  # evita división por cero en columnas constantes

X_train = (X_train - mu) / sigma
X_val = (X_val - mu) / sigma
X_test = (X_test - mu) / sigma


# ---------------------------------------------------------------------------
# regresión logística desde cero (descenso de gradiente)
# ---------------------------------------------------------------------------
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def bce_loss(y, y_hat):
    eps = 1e-12
    return -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))


def train_logreg(X, y, X_val, y_val, epochs, lr):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        z = X @ w + b
        y_hat = sigmoid(z)
        error = y_hat - y

        grad_w = (X.T @ error) / n_samples
        grad_b = error.mean()

        w -= lr * grad_w
        b -= lr * grad_b

        train_loss = bce_loss(y, y_hat)
        val_loss = bce_loss(y_val, sigmoid(X_val @ w + b))
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % 200 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:5d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    return w, b, train_losses, val_losses


def predict_proba(X, w, b):
    return sigmoid(X @ w + b)


def predict(X, w, b, threshold=0.5):
    return (predict_proba(X, w, b) >= threshold).astype(int)


# ---------------------------------------------------------------------------
# entrenamiento
# ---------------------------------------------------------------------------
section("ENTRENAMIENTO")
epochs = 5000
w, b, train_losses, val_losses = train_logreg(
    X_train, y_train, X_val, y_val, lr=0.1, epochs=epochs
)

# ---------------------------------------------------------------------------
# métricas manuales
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
y_pred_test = predict(X_test, w, b)
test_accuracy, test_precision, test_recall, test_f1 = metrics_report(y_test, y_pred_test)

section("EVALUACIÓN EN VALIDATION")
y_pred_val = predict(X_val, w, b)
metrics_report(y_val, y_pred_val)

section("EVALUACIÓN EN TRAIN (referencia)")
y_pred_train = predict(X_train, w, b)
metrics_report(y_train, y_pred_train)

# ---------------------------------------------------------------------------
# importancia de variables (coeficientes sobre datos estandarizados)
# ---------------------------------------------------------------------------
section("COEFICIENTES (ordenados por magnitud)")
coef_df = pd.DataFrame({"feature": feature_names, "coef": w})
coef_df = coef_df.reindex(coef_df["coef"].abs().sort_values(ascending=False).index)
print(coef_df.to_string(index=False))
print(f"\nintercepto (bias): {b:.4f}")

# ---------------------------------------------------------------------------
# gráfica: evolución del loss (train vs validation)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(epochs), train_losses, label="train loss", color="steelblue")
ax.plot(range(epochs), val_losses, label="validation loss", color="firebrick")
ax.set_xlabel("epoch")
ax.set_ylabel("binary cross-entropy loss")
ax.set_title("Evolución del loss durante el entrenamiento")
ax.legend()
fig.tight_layout()
fig.savefig("figuras/loss_curve.png", dpi=150, bbox_inches="tight")

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
ax.set_title("Matriz de confusión (test)")
fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig("figuras/matriz_confusion.png", dpi=150, bbox_inches="tight")

# ---------------------------------------------------------------------------
# gráfica: coeficientes del modelo (importancia de variables)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 10))
colors = ["firebrick" if v > 0 else "steelblue" for v in coef_df["coef"]]
ax.barh(coef_df["feature"], coef_df["coef"], color=colors)
ax.invert_yaxis()
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Coeficientes del modelo (ordenados por magnitud)")
fig.tight_layout()
fig.savefig("figuras/coeficientes.png", dpi=150, bbox_inches="tight")

plt.show()
