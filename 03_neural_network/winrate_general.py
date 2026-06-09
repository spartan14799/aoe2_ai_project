import os
import pandas as pd

MODULE_DIR    = os.path.dirname(os.path.abspath(__file__))
AOE_DATA_FILE = os.path.join(MODULE_DIR, "datasets", "aoe_data.csv")
OUT_DIR       = os.path.join(MODULE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    print("Cargando datos generales...")
    # Cargar solo las columnas necesarias para que sea rápido
    df = pd.read_csv(AOE_DATA_FILE, usecols=["p1_civ", "p2_civ", "winner"])

    print(f"Total de partidas en el dataset original: {len(df):,}\n")

    # Separar en dos dataframes: uno para el jugador 1 y otro para el jugador 2
    # winner == 0 significa que ganó P1, winner == 1 significa que ganó P2
    df_p1 = pd.DataFrame({
        "civ": df["p1_civ"],
        "is_winner": (df["winner"] == 0).astype(int)
    })

    df_p2 = pd.DataFrame({
        "civ": df["p2_civ"],
        "is_winner": (df["winner"] == 1).astype(int)
    })

    # Juntar todo (cada fila de df_civs es "una civilización jugada en una partida")
    df_civs = pd.concat([df_p1, df_p2], ignore_index=True)

    # Agrupar y calcular estadísticas
    stats = []
    civs = df_civs["civ"].unique()

    for civ in civs:
        if pd.isna(civ):
            continue
            
        civ_data = df_civs[df_civs["civ"] == civ]
        partidas = len(civ_data)
        
        # Ignorar si hay muy pocas partidas (aunque en el dataset entero no debería pasar)
        if partidas < 100:
            continue
            
        winrate = civ_data["is_winner"].mean() * 100
        
        stats.append({
            "Nacion": civ,
            "Partidas_Jugadas": partidas,
            "Winrate_General": round(winrate, 2)
        })

    # Ordenar por Winrate (las que más ganan arriba)
    df_stats = pd.DataFrame(stats).sort_values("Winrate_General", ascending=False)

    print("=== TOP 10 NACIONES CON MEJOR WINRATE GENERAL ===")
    print(df_stats.head(10).to_string(index=False))
    
    print("\n=== TOP 10 NACIONES CON PEOR WINRATE GENERAL ===")
    print(df_stats.tail(10).to_string(index=False))

    # Guardar a CSV
    out_csv = os.path.join(OUT_DIR, "winrate_general_naciones.csv")
    df_stats.to_csv(out_csv, index=False)
    print(f"\nReporte completo guardado en {out_csv}")


if __name__ == "__main__":
    main()
