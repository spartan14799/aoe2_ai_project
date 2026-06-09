import os
import sys
import json
import pickle
import argparse
import pandas as pd

MODEL_DIR     = os.path.dirname(os.path.abspath(__file__))
OUT_DIR       = os.path.join(MODEL_DIR, "output")
MODEL_FILE    = os.path.join(OUT_DIR, "best_modelo_90_10.pkl")
SCALER_FILE   = os.path.join(OUT_DIR, "best_scaler_90_10.pkl")
FEATURES_FILE = os.path.join(OUT_DIR, "features_clear.json")
DATA_FILE     = os.path.join(MODEL_DIR, "datasets", "clear_dataset.csv")

def cargar_artefactos():
    for path in (MODEL_FILE, SCALER_FILE, FEATURES_FILE):
        if not os.path.exists(path):
            print("Primero hay que entrenar el modelo: py 03_neural_network/train_90_10.py")
            sys.exit(1)

    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_FILE, "rb") as f:
        scaler = pickle.load(f)
    with open(FEATURES_FILE, "r") as f:
        features = json.load(f)

    return model, scaler, features


def predecir_y_mostrar(model, scaler, features, row_dict, winner_real=None):
    df = pd.DataFrame([row_dict])[features]
    X_sc = scaler.transform(df.values)  # .values evita warning de feature names

    pred = model.predict(X_sc)[0]
    prob = model.predict_proba(X_sc)[0]

    ganador = "Jugador 1" if pred == 0 else "Jugador 2"
    print(f"\nGanador predicho : {ganador}")
    print(f"Probabilidad P1  : {prob[0]*100:.1f}%")
    print(f"Probabilidad P2  : {prob[1]*100:.1f}%")

    if winner_real is not None:
        real_str = "Jugador 1" if winner_real == 0 else "Jugador 2"
        acierto  = "si" if pred == winner_real else "no"
        print(f"Resultado real   : {real_str}  (acierto: {acierto})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--match_id",      type=int,   help="ID de la partida en el dataset")
    parser.add_argument("--file",          type=str,   help="CSV con varias partidas para predecir")
    parser.add_argument("--avg_elo",       type=float)
    parser.add_argument("--diff_villagers",type=float)
    parser.add_argument("--diff_apm",      type=float)
    parser.add_argument("--diff_queues",   type=float)
    parser.add_argument("--diff_builds",   type=float)
    parser.add_argument("--diff_orders",   type=float)
    args = parser.parse_args()

    model, scaler, features = cargar_artefactos()

    # buscar por ID en el dataset limpio
    if args.match_id is not None:
        if not os.path.exists(DATA_FILE):
            print(f"Dataset no encontrado en {DATA_FILE}")
            sys.exit(1)

        df = pd.read_csv(DATA_FILE)
        fila = df[df["match_id"] == args.match_id]

        if fila.empty:
            print(f"No existe ninguna partida con match_id={args.match_id}")
            sys.exit(1)

        row = fila.iloc[0]
        print(f"match_id        : {args.match_id}")
        print(f"avg_elo         : {row['avg_elo']}")
        print(f"diff_villagers  : {row['diff_villagers_snapshot']}")
        print(f"diff_apm        : {row['diff_apm']:.2f}")
        print(f"diff_queues     : {row['diff_queues']}")
        print(f"diff_builds     : {row['diff_builds']}")
        print(f"diff_orders     : {row['diff_orders']}")

        datos = {
            "avg_elo":                  float(row["avg_elo"]),
            "diff_villagers_snapshot":  float(row["diff_villagers_snapshot"]),
            "diff_apm":                 float(row["diff_apm"]),
            "diff_queues":              float(row["diff_queues"]),
            "diff_builds":              float(row["diff_builds"]),
            "diff_orders":              float(row["diff_orders"]),
        }
        predecir_y_mostrar(model, scaler, features, datos, winner_real=int(row["winner"]))

    # prediccion en lote desde CSV
    elif args.file:
        if not os.path.exists(args.file):
            print(f"Archivo no encontrado: {args.file}")
            sys.exit(1)

        df = pd.read_csv(args.file)
        df = df.rename(columns={"diff_villagers": "diff_villagers_snapshot"})
        X_sc = scaler.transform(df[features])

        df["pred_winner"]  = model.predict(X_sc)
        df["prob_p2"]      = model.predict_proba(X_sc)[:, 1].round(4)

        out = args.file.replace(".csv", "_predicciones.csv")
        df.to_csv(out, index=False)
        print(f"Predicciones guardadas en {out}")

        cols = [c for c in ("match_id", "pred_winner", "prob_p2") if c in df.columns]
        print(df[cols].head(10).to_string(index=False))

    # prediccion manual con valores pasados como argumentos
    else:
        campos = [args.avg_elo, args.diff_villagers, args.diff_apm,
                  args.diff_queues, args.diff_builds, args.diff_orders]
        if any(v is None for v in campos):
            print("Usa --match_id, --file, o pasa los 6 argumentos manualmente:")
            print("  py predict_clear.py --avg_elo 1200 --diff_villagers 5 "
                  "--diff_apm 12.5 --diff_queues 2 --diff_builds 1 --diff_orders 14")
            sys.exit(1)

        datos = {
            "avg_elo":                 args.avg_elo,
            "diff_villagers_snapshot": args.diff_villagers,
            "diff_apm":                args.diff_apm,
            "diff_queues":             args.diff_queues,
            "diff_builds":             args.diff_builds,
            "diff_orders":             args.diff_orders,
        }
        predecir_y_mostrar(model, scaler, features, datos)


if __name__ == "__main__":
    main()
