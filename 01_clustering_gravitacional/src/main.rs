// src/main.rs

mod particle;
mod physics;
mod union_find;

use csv::ReaderBuilder;
use particle::Particle;
use physics::run_gravitacional_clustering;
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;

// 1. Función para leer el CSV y transformarlo en el Universo de Partículas
fn cargar_particulas_desde_csv(ruta_archivo: &str) -> Result<Vec<Particle>, Box<dyn Error>> {
    println!("Cargando datos desde: {}...", ruta_archivo);

    let archivo = File::open(ruta_archivo)?;
    let mut lector = ReaderBuilder::new().from_reader(archivo);
    let cabeceras = lector.headers()?.clone();

    let mut particulas = Vec::new();
    let mut id_contador = 0;

    for resultado in lector.records() {
        let fila = resultado?;
        let mut posicion = Vec::new();

        // Recorremos cada columna de la fila
        for (i, campo) in fila.iter().enumerate() {
            let nombre_columna = &cabeceras[i];

            // Ignoramos las columnas que son texto o IDs para que no arruinen la distancia matemática
            if nombre_columna == "match_id"
                || nombre_columna == "map"
                || nombre_columna == "map_size"
                || nombre_columna == "p1_civ"
                || nombre_columna == "p2_civ"
                || nombre_columna == "winner"
            {
                continue;
            }

            // Intentamos convertir el texto a número decimal (f64). Si falla o está vacío, ponemos 0.0
            let valor: f64 = campo.parse().unwrap_or(0.0);
            posicion.push(valor);
        }

        // Creamos la partícula y la agregamos a nuestro universo
        particulas.push(Particle::new(id_contador, posicion));
        id_contador += 1;
    }

    Ok(particulas)
}

fn main() {
    println!("=== MOTOR DE CLUSTERING GRAVITACIONAL ===");

    // Ajusta la ruta a donde guardaste el dataset procesado por Python
    let ruta_dataset = "../data/processed/clear_dataset.csv";

    // 1. Cargar las partículas
    let mut universo = match cargar_particulas_desde_csv(ruta_dataset) {
        Ok(datos) => datos,
        Err(e) => {
            eprintln!("Error al leer el CSV: {}", e);
            return;
        }
    };

    println!(
        "Universo creado con {} partículas (partidas).",
        universo.len()
    );
    println!("Dimensiones por partícula: {}", universo[0].dimensions());

    // 2. Parámetros del algoritmo de tu profesor (AQUÍ ES DONDE DEBES EXPERIMENTAR PARA TU ENSAYO)
    let g_inicial = 100.0; // Fuerza gravitacional inicial
    let delta_g = 0.01; // Tasa de decaimiento (enfriamiento)
    let iteraciones = 500; // M: Número de veces que se repite la simulación
    let epsilon = 100.0; // Si la distancia al cuadrado es menor a esto, se fusionan

    // 3. Ejecutar la simulación gravitacional
    let mut union_find_resultado =
        run_gravitacional_clustering(&mut universo, g_inicial, delta_g, iteraciones, epsilon);

    // 4. Analizar los resultados (Contar cuántas partidas hay en cada Clúster)
    let mut tamano_clusters: HashMap<usize, usize> = HashMap::new();

    for i in 0..universo.len() {
        let raiz = union_find_resultado.find(i); // ¿A qué clúster pertenece la partida i?
        *tamano_clusters.entry(raiz).or_insert(0) += 1;
    }

    // Ordenar y mostrar los clústeres encontrados
    let mut clusters_ordenados: Vec<_> = tamano_clusters.into_iter().collect();
    // Ordenamos de mayor a menor tamaño
    clusters_ordenados.sort_by(|a, b| b.1.cmp(&a.1));

    println!("\n=== RESULTADOS DE LA CLUSTERIZACIÓN ===");
    println!(
        "Se encontraron {} estrategias (clústeres) distintas.",
        clusters_ordenados.len()
    );

    for (id_cluster, cantidad_partidas) in clusters_ordenados.iter().take(10) {
        println!("Clúster ID {}: {} partidas", id_cluster, cantidad_partidas);
    }

    if clusters_ordenados.len() > 10 {
        println!(
            "... y {} clústeres más pequeños.",
            clusters_ordenados.len() - 10
        );
    }
}
