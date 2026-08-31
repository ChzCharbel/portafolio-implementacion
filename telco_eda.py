import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", None)

os.makedirs("figuras", exist_ok=True)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# estructura general
# ---------------------------------------------------------------------------
section("ESTRUCTURA GENERAL")

df = pd.read_csv("telco_churn.csv")
df = df.drop(columns=["customerID"])

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["Churn"] = (df["Churn"] == "Yes").astype(int)

print(df.shape)
print("\nTipos de dato:")
print(df.dtypes)
print("\nPrimeras filas:")
print(df.head())

binary_cols = ["SeniorCitizen"]
num_cols = df.select_dtypes(include=np.number).columns.tolist()
for c in ["Churn"] + binary_cols:
    if c in num_cols:
        num_cols.remove(c)
cat_cols = df.select_dtypes(include="str").columns.tolist()

print(f"\nColumnas numéricas continuas: {num_cols}")
print(f"Columnas binarias (0/1): {binary_cols}")
print(f"Columnas categóricas: {cat_cols}")

# ---------------------------------------------------------------------------
# nulos
# ---------------------------------------------------------------------------
section("VALORES NULOS")

nulls = df.isnull().sum()
nulls_pct = (nulls / len(df) * 100).round(2)
print("Valores nulos por columna:")
print(pd.DataFrame({"nulos": nulls, "%": nulls_pct}))

# TotalCharges queda NaN cuando tenure=0 (clientes nuevos, aún no facturados)
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# ---------------------------------------------------------------------------
# descriptive stats
# ---------------------------------------------------------------------------
section("stats - númericas")
print(df[num_cols].describe().T)

section("stats - categóricas")
print(df[cat_cols + binary_cols].describe().T)

for c in cat_cols + binary_cols:
    print(df[c].value_counts(dropna=False))

# ---------------------------------------------------------------------------
# Variable objetivo: Churn
# ---------------------------------------------------------------------------
section("DISTRIBUCIÓN DE LA VARIABLE OBJETIVO")

churn_counts = df["Churn"].value_counts().sort_index()
churn_pct = (churn_counts / len(df) * 100).round(2)
print(pd.DataFrame({"conteo": churn_counts, "%": churn_pct}))

# ---------------------------------------------------------------------------
# correlación entre numéricas
# ---------------------------------------------------------------------------
section("MATRIZ DE CORRELACIÓN")

corr = df[num_cols + binary_cols + ["Churn"]].corr(numeric_only=True)
print(corr.round(2))

# ---------------------------------------------------------------------------
# Encoding de variables categóricas
# ---------------------------------------------------------------------------
section("ENCODING DE VARIABLES CATEGÓRICAS")

binary_map_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
one_hot_cols = [c for c in cat_cols if c not in binary_map_cols]

df_encoded = df.copy()
df_encoded["gender"] = (df_encoded["gender"] == "Male").astype(int)
df_encoded["Partner"] = (df_encoded["Partner"] == "Yes").astype(int)
df_encoded["Dependents"] = (df_encoded["Dependents"] == "Yes").astype(int)
df_encoded["PhoneService"] = (df_encoded["PhoneService"] == "Yes").astype(int)
df_encoded["PaperlessBilling"] = (df_encoded["PaperlessBilling"] == "Yes").astype(int)

df_encoded = pd.get_dummies(df_encoded, columns=one_hot_cols, prefix=one_hot_cols, dtype=int)

print(f"Binarias (mapeo manual): {binary_map_cols}")
print(f"One-hot: {one_hot_cols}")
print(f"\nForma original: {df.shape}  ->  Forma codificada: {df_encoded.shape}")
print("\nColumnas resultantes:")
print(df_encoded.columns.tolist())
print("\nPrimeras filas:")
print(df_encoded.head())

corr_encoded = df_encoded.corr(numeric_only=True)["Churn"].drop("Churn").sort_values()

# ===========================================================================
# GRÁFICOS
# ===========================================================================

# Histogramas de variables numéricas continuas
fig, axes = plt.subplots(1, len(num_cols), figsize=(5 * len(num_cols), 5))
for ax, c in zip(axes, num_cols):
    ax.hist(df[c].dropna(), bins=25, color="steelblue", edgecolor="white")
    ax.set_title(c)
fig.suptitle("Distribución de variables numéricas continuas")
fig.tight_layout()
fig.savefig("figuras/histogramas_numericas.png", dpi=150, bbox_inches="tight")

# Barras de variables categóricas y binarias
cat_all = cat_cols + binary_cols
fig, axes = plt.subplots(4, 4, figsize=(20, 16))
for ax, c in zip(axes.ravel(), cat_all):
    counts = df[c].value_counts()
    ax.bar(counts.index.astype(str), counts.values, color="darkorange")
    ax.set_title(c)
    ax.tick_params(axis="x", rotation=45)
for ax in axes.ravel()[len(cat_all):]:
    ax.axis("off")
fig.suptitle("Distribución de variables categóricas y binarias")
fig.tight_layout()
fig.savefig("figuras/barras_categoricas.png", dpi=150, bbox_inches="tight")

# Balance de la variable objetivo
fig, ax = plt.subplots(figsize=(5, 5))
ax.bar(["No churn", "Churn"], churn_counts.values, color=["seagreen", "firebrick"])
for i, v in enumerate(churn_counts.values):
    ax.text(i, v + 20, str(v), ha="center")
ax.set_title("Balance de clases: Churn")
fig.tight_layout()
fig.savefig("figuras/balance_churn.png", dpi=150, bbox_inches="tight")

# Matriz de correlación
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticks(range(len(corr.columns)))
ax.set_yticklabels(corr.columns)
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(im, ax=ax)
ax.set_title("Matriz de correlación")
fig.tight_layout()
fig.savefig("figuras/matriz_correlacion.png", dpi=150, bbox_inches="tight")

# Tasa de churn por antigüedad (bins) y por tipo de contrato
tenure_bins = pd.cut(df["tenure"], bins=[-1, 12, 24, 48, 60, 72])
rate_by_tenure = df.groupby(tenure_bins, observed=True)["Churn"].mean() * 100

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(rate_by_tenure.index.astype(str), rate_by_tenure.values, color="teal")
axes[0].set_title("% churn por antigüedad (meses)")
axes[0].tick_params(axis="x", rotation=45)

rate_by_contract = df.groupby("Contract")["Churn"].mean().sort_values(ascending=False) * 100
axes[1].bar(rate_by_contract.index.astype(str), rate_by_contract.values, color="purple")
axes[1].set_title("% churn por tipo de contrato")
axes[1].tick_params(axis="x", rotation=15)
fig.tight_layout()
fig.savefig("figuras/tasa_churn_tenure_contrato.png", dpi=150, bbox_inches="tight")

# Correlación de las variables codificadas con Churn
fig, ax = plt.subplots(figsize=(8, 10))
colors = ["firebrick" if v > 0 else "steelblue" for v in corr_encoded.values]
ax.barh(corr_encoded.index.astype(str), corr_encoded.values, color=colors)
ax.set_title("Correlación de variables codificadas con Churn")
ax.axvline(0, color="black", linewidth=0.8)
fig.tight_layout()
fig.savefig("figuras/correlacion_codificada.png", dpi=150, bbox_inches="tight")

plt.show()

# guardamos el df para usarlo en el modelo
df_encoded.to_csv("telco_churn_processed.csv", index=False)
