import os
import sys
import json
import pickle
import pandas as pd
import numpy as np

MODEL_DIR     = os.path.dirname(os.path.abspath(__file__))
OUT_DIR       = os.path.join(MODEL_DIR, "output")

MODEL_FILE    = os.path.join(OUT_DIR, "best_modelo_90_10.pkl")
SCALER_FILE   = os.path.join(OUT_DIR, "best_scaler_90_10.pkl")
FEATURES_FILE = os.path.join(OUT_DIR, "features_clear.json")

CLEAR_DATA_FILE = os.path.join(MODEL_DIR, "datasets", "clear_dataset.csv")
AOE_DATA_FILE   = os.path.join(MODEL_DIR, "datasets", "aoe_data.csv")


def main():
    print("Cargando modelo y datos...")
    if not os.path.exists(MODEL_FILE):
        print("Falta el modelo entrenado. Corre train_90_10.py primero.")
        sys.exit(1)

    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_FILE, "rb") as f:
        scaler = pickle.load(f)
    with open(FEATURES_FILE, "r") as f:
        features = json.load(f)

    # Cargar datos
    df_clear = pd.read_csv(CLEAR_DATA_FILE)
    df_aoe = pd.read_csv(AOE_DATA_FILE, usecols=["match_id", "duration", "p1_civ", "p2_civ"])

    # Hacer merge para tener las naciones y la duración
    df = pd.merge(df_clear, df_aoe, on="match_id", how="inner")

    print(f"Partidas combinadas: {len(df):,}")

    # Predecir probabilidades de victoria en el minuto 20
    X_sc = scaler.transform(df[features].values)
    probs = model.predict_proba(X_sc)
    
    df["prob_p1_win"] = probs[:, 0]
    df["prob_p2_win"] = probs[:, 1]
    
    # Desdoblar el dataframe para analizar cada civilizacion individualmente
    # Para el jugador 1
    df_p1 = pd.DataFrame({
        "civ": df["p1_civ"],
        "duration": df["duration"],
        "is_winner": (df["winner"] == 0).astype(int),
        "predicted_win_prob_at_20m": df["prob_p1_win"]
    })

    # Para el jugador 2
    df_p2 = pd.DataFrame({
        "civ": df["p2_civ"],
        "duration": df["duration"],
        "is_winner": (df["winner"] == 1).astype(int),
        "predicted_win_prob_at_20m": df["prob_p2_win"]
    })

    # Juntar ambos
    df_civs = pd.concat([df_p1, df_p2], ignore_index=True)

    # Definir "Late game" como partidas que duran más de 40 minutos (2400 segundos)
    # Nota: El snapshot se toma al min 20, así que todo esto es a partir de ahí
    late_game_threshold = 2400
    df_late = df_civs[df_civs["duration"] >= late_game_threshold]

    print(f"Analizando {len(df_late):,} casos de Late Game (>40 min)...\n")

    # Agrupar por civilización
    stats = []
    civs = df_late["civ"].unique()
    
    for civ in civs:
        if pd.isna(civ):
            continue
            
        civ_late = df_late[df_late["civ"] == civ]
        n_matches = len(civ_late)
        
        if n_matches < 100:  # Ignorar si hay muy pocos datos para ser representativo
            continue

        actual_winrate = civ_late["is_winner"].mean() * 100
        predicted_winrate = civ_late["predicted_win_prob_at_20m"].mean() * 100
        
        # El "Indice Late Game" es qué tanto superan las expectativas.
        # Si al min 20 la red predecía que ganaban el 45% de las veces, pero en partidas largas 
        # terminan ganando el 55%, significa que escalan súper bien y remontan.
        lategame_index = actual_winrate - predicted_winrate

        stats.append({
            "Nacion": civ,
            "Partidas_Late": n_matches,
            "Expectativa_Min20": round(predicted_winrate, 1),
            "Winrate_Real_Late": round(actual_winrate, 1),
            "Indice_LateGame": round(lategame_index, 2)
        })

    # Ordenar por el índice Late Game (los que mejor remontan / mejor lategame tienen)
    df_stats = pd.DataFrame(stats).sort_values("Indice_LateGame", ascending=False)

    print("=== TOP NACIONES DE LATE GAME (Ganan más de lo que la IA esperaba al min 20) ===")
    print(df_stats.head(10).to_string(index=False))
    print("\n=== NACIONES QUE DECAEN EN LATE GAME (Tenían ventaja al min 20 pero terminan perdiendo) ===")
    print(df_stats.tail(10).to_string(index=False))

    # Guardar a CSV
    out_csv = os.path.join(OUT_DIR, "analisis_lategame_naciones.csv")
    df_stats.to_csv(out_csv, index=False)
    print(f"\nReporte completo guardado en {out_csv}")


if __name__ == "__main__":
    main()
