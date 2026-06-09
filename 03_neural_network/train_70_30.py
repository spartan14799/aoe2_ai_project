import os
import sys
import json
import time
import pickle
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

warnings.filterwarnings("ignore")

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE  = os.path.join(MODULE_DIR, "datasets", "clear_dataset.csv")
OUT_DIR    = os.path.join(MODULE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)


def cargar_datos():
    if not os.path.exists(DATA_FILE):
        print(f"No se encontro el dataset en {DATA_FILE}")
        sys.exit(1)
    df = pd.read_csv(DATA_FILE)
    y = df["winner"].astype(int).values
    X = df.drop(columns=["match_id", "winner"], errors="ignore")
    return X.values, y, X.columns.tolist()


def build_mlp(seed=42):
    return MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate="adaptive",
        learning_rate_init=1e-3,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=seed,
        verbose=False,
    )


def main():
    X, y, features = cargar_datos()
    print(f"Dataset cargado: {len(X):,} partidas\n")

    # Usamos ShuffleSplit para hacer la division 70/30, 10 veces para comparar bien
    rs = ShuffleSplit(n_splits=10, test_size=0.30, random_state=42)
    acc_list, auc_list = [], []
    best_acc, best_model, best_scaler = 0.0, None, None

    print("=== Entrenamiento 70/30 (Repetido 10 veces) ===")
    for split_idx, (tr_idx, te_idx) in enumerate(rs.split(X, y), start=1):
        t0 = time.time()

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_te = scaler.transform(X[te_idx])

        model = build_mlp(seed=42 + split_idx)
        model.fit(X_tr, y[tr_idx])

        acc = accuracy_score(y[te_idx], model.predict(X_te))
        auc = roc_auc_score(y[te_idx], model.predict_proba(X_te)[:, 1])
        acc_list.append(acc)
        auc_list.append(auc)

        print(f"  Split {split_idx:02d}/10 | Acc: {acc*100:.2f}% | AUC: {auc:.4f} | ({time.time()-t0:.1f}s)")

        if acc > best_acc:
            best_acc, best_model, best_scaler = acc, model, scaler

    mean_acc = np.mean(acc_list)
    mean_auc = np.mean(auc_list)
    
    print("-" * 40)
    print(f"Promedio 70/30 | Acc: {mean_acc*100:.2f}% | AUC: {mean_auc:.4f}\n")

    # Guardar modelo
    with open(os.path.join(OUT_DIR, "best_modelo_70_30.pkl"), "wb") as f:
        pickle.dump(best_model, f)
    with open(os.path.join(OUT_DIR, "best_scaler_70_30.pkl"), "wb") as f:
        pickle.dump(best_scaler, f)
    with open(os.path.join(OUT_DIR, "features_clear.json"), "w") as f:
        json.dump(features, f, indent=2)

    resultados = {
        "acc_por_split": [round(float(v), 6) for v in acc_list],
        "auc_por_split": [round(float(v), 6) for v in auc_list],
        "acc_media": round(float(mean_acc), 6),
        "auc_media": round(float(mean_auc), 6),
    }
    with open(os.path.join(OUT_DIR, "resultados_70_30.json"), "w") as f:
        json.dump(resultados, f, indent=2)

    print(f"Mejor modelo guardado en: {OUT_DIR}/best_modelo_70_30.pkl")


if __name__ == "__main__":
    main()
