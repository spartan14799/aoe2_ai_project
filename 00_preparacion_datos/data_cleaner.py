import pandas as pd
import numpy as np
import os

# ==========================================
# CONFIGURACIÓN DE RUTAS
# ==========================================
RUTA_DATOS = "../data/raw/"
RUTA_SALIDA = "../data/processed/"

ARCHIVO_SNAPSHOT = os.path.join(RUTA_DATOS, "sample_snapshots_t_1200.csv")
ARCHIVO_UNIDADES = os.path.join(RUTA_DATOS, "unit_masterdata.csv")
ARCHIVO_EDIFICIOS = os.path.join(RUTA_DATOS, "building_masterdata.csv")
ARCHIVO_TECNOLOGIAS = os.path.join(RUTA_DATOS, "research_masterdata.csv")


def cargar_y_limpiar_masterdatas():
    """Carga los CSV y asigna las nuevas categorías en inglés corto"""
    df_unidades = pd.read_csv(ARCHIVO_UNIDADES, sep=";")
    df_edificios = pd.read_csv(ARCHIVO_EDIFICIOS, sep=";")
    df_tecnologias = pd.read_csv(ARCHIVO_TECNOLOGIAS, sep=";")

    # Limpiar espacios en blanco de los nombres
    df_unidades["Unit"] = df_unidades["Unit"].astype(str).str.strip()
    df_edificios["Building"] = df_edificios["Building"].astype(str).str.strip()
    df_tecnologias["Technology"] = df_tecnologias["Technology"].astype(str).str.strip()

    # Asegurarnos de que existan las 4 columnas de recursos (rellenar con 0)
    for df in [df_unidades, df_edificios, df_tecnologias]:
        for recurso in ["Food", "Wood", "Gold", "Stone"]:
            if recurso not in df.columns:
                df[recurso] = 0
            df[recurso] = df[recurso].fillna(0).astype(float)

    # ----- CLASIFICACIÓN (NOMBRES CORTOS EN INGLÉS) -----

    # 1. Edificios (Buildings)
    df_edificios["Category"] = np.where(
        df_edificios["Type"].str.contains("Economic", case=False, na=False),
        "eco_bldg",
        "mil_bldg",
    )

    # 2. Unidades (Units)
    unidades_eco = ["Villager", "Fishing Ship", "Trade Cart", "Trade Cog"]
    df_unidades["Category"] = np.where(
        df_unidades["Unit"].isin(unidades_eco), "eco_unit", "mil_unit"
    )

    # 3. Tecnologías (Techs)
    df_tecnologias["Category"] = np.where(
        df_tecnologias["Type"].str.contains("Economic", case=False, na=False),
        "eco_tech",
        "mil_tech",
    )

    # ----- CREAR DICCIONARIOS -----
    diccionario_items = {}

    def agregar_a_diccionario(df, col_nombre):
        for _, row in df.iterrows():
            diccionario_items[row[col_nombre]] = {
                "food": row["Food"],
                "wood": row["Wood"],
                "gold": row["Gold"],
                "stone": row["Stone"],
                "category": row["Category"],
            }

    agregar_a_diccionario(df_unidades, "Unit")
    agregar_a_diccionario(df_edificios, "Building")
    agregar_a_diccionario(df_tecnologias, "Technology")

    return diccionario_items


def procesar_snapshots():
    print("1. Cargando datos maestros...")
    diccionario_items = cargar_y_limpiar_masterdatas()

    print("2. Cargando base de datos de partidas...")
    df_snap = pd.read_csv(ARCHIVO_SNAPSHOT)

    # Columnas base a conservar
    columnas_base = [
        "match_id",
        "avg_elo",
        "time",
        "map",
        "map_size",
        "duration",
        "p1_civ",
        "p2_civ",
        "winner",
        "p1 Feudal Age Time",
        "p1 Castle Age Time",
        "p1 Imperial Age Time",
        "p2 Feudal Age Time",
        "p2 Castle Age Time",
        "p2 Imperial Age Time",
    ]

    df_procesado = df_snap[columnas_base].copy()

    # Limpiar tiempos de edades (rellenando nulos con -1)
    for col in df_procesado.columns:
        if "Age Time" in col:
            df_procesado[col] = df_procesado[col].fillna(-1)

    print("3. Calculando recursos...")

    jugadores = ["p1", "p2"]
    categorias = [
        "eco_bldg",
        "mil_bldg",
        "eco_unit",
        "mil_unit",
        "eco_tech",
        "mil_tech",
    ]
    recursos = ["food", "wood", "gold", "stone"]

    # Inicializar TODAS las columnas numéricas en 0.0 para evitar KeyErrors
    for jugador in jugadores:
        for cat in categorias:
            for rec in recursos:
                col_name = f"{jugador}_{cat}_{rec}"
                df_procesado[col_name] = 0.0

        # Conteos
        df_procesado[f"{jugador}_villager_count"] = 0
        df_procesado[f"{jugador}_mil_unit_count"] = 0

    # Iterar y sumar
    for col in df_snap.columns:
        if col.startswith("p1_") or col.startswith("p2_"):
            jugador = col[:2]
            item_nombre = col[3:]

            if "Age Time" in col or item_nombre in ["civ", "None", ""]:
                continue

            if item_nombre in diccionario_items:
                info_item = diccionario_items[item_nombre]
                categoria = info_item["category"]

                cantidad = df_snap[col].fillna(0)

                # Multiplicar costo por cantidad para cada recurso
                for rec in recursos:
                    costo_unitario = info_item[rec]
                    col_name = f"{jugador}_{categoria}_{rec}"
                    df_procesado[col_name] += cantidad * costo_unitario

                # Conteos especiales
                if categoria == "eco_unit" and item_nombre == "Villager":
                    df_procesado[f"{jugador}_villager_count"] += cantidad
                elif categoria == "mil_unit":
                    df_procesado[f"{jugador}_mil_unit_count"] += cantidad

    print("4. Guardando CSV procesado...")
    os.makedirs(RUTA_SALIDA, exist_ok=True)
    ruta_guardado = os.path.join(RUTA_SALIDA, "clear_dataset.csv")
    df_procesado.to_csv(ruta_guardado, index=False)

    print(f"¡Éxito! Guardado en: {ruta_guardado}")


if __name__ == "__main__":
    procesar_snapshots()
