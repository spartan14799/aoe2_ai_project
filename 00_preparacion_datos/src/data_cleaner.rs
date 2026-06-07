use std::error::Error;
use std::fs::File;
use std::path::Path;
use csv::ReaderBuilder;
use rayon::prelude::*; // Paralelismo multi-hilo

// Constantes de filtrado según especificaciones
const ELO_MINIMO: f64 = 900.0;
const DURACION_MINIMA_SEGUNDOS: i32 = 1200; // 20 minutos de duración mínima de juego
const TIEMPO_SNAPSHOT_SEGUNDOS: i32 = 1200; // Límite de procesamiento de acciones (Minuto 20)

struct PartidaFiltrada {
    match_id: String,
    avg_elo: f64,
    winner_target: f64, // 1.0 o 0.0 según el CSV indexado
    p1_villagers_snapshot: f64,
    p2_villagers_snapshot: f64,
}

struct FilaLimpia {
    match_id: String,
    avg_elo: f64,
    winner: f64,
    diff_villagers_snapshot: f64, 
    diff_apm: f64,                
    diff_queues: f64,             
    diff_builds: f64,             
    diff_orders: f64,             
}

pub fn ejecutar_limpieza() -> Result<(), Box<dyn Error>> {
    println!("=== Iniciando Pipeline de Limpieza ===");

    // Configuración de rutas de archivos
    let ruta_snapshots = "../data/raw/sample_snapshots_t_1200.csv"; 
    let carpeta_inputs = "../data/raw/inputs/inputs"; 
    let ruta_salida = "../data/processed/clear_dataset.csv";

    // Validar que el archivo maestro exista antes de continuar
    if !Path::new(ruta_snapshots).exists() {
        return Err(format!("No se encontró el archivo maestro en: {}. Asegúrate de moverlo a esa ubicación.", ruta_snapshots).into());
    }

    let mut rdr = ReaderBuilder::new().from_path(ruta_snapshots)?;
    
    // Lectura dinámica de encabezados del Snapshot para evitar problemas con columnas vacías
    let headers = rdr.headers()?.clone();
    let idx_match_id = headers.iter().position(|h| h == "match_id").ok_or("Falta columna match_id")?;
    let idx_elo = headers.iter().position(|h| h == "avg_elo").ok_or("Falta columna avg_elo")?;
    let idx_map = headers.iter().position(|h| h == "map").ok_or("Falta columna map")?;
    let idx_duration = headers.iter().position(|h| h == "duration").ok_or("Falta columna duration")?;
    let idx_winner = headers.iter().position(|h| h == "winner").ok_or("Falta columna winner")?;
    let idx_p1_vil = headers.iter().position(|h| h == "p1_Villager").ok_or("Falta columna p1_Villager")?;
    let idx_p2_vil = headers.iter().position(|h| h == "p2_Villager").ok_or("Falta columna p2_Villager")?;

    let mut partidas_candidatas = Vec::new();

    println!("Filtrando partidas (Mapa: Arabia, ELO >= {}, Duración >= 20 min)...", ELO_MINIMO);
    
    for result in rdr.records() {
        let record = result?;
        
        // Extracción de variables de control para el filtrado masivo
        let elo: f64 = record.get(idx_elo).unwrap_or("0").parse().unwrap_or(0.0);
        let mapa = record.get(idx_map).unwrap_or("");
        let duracion: i32 = record.get(idx_duration).unwrap_or("0").parse().unwrap_or(0);
        
        // FILTROS EXIGIDOS POR EL PAPER Y TU CONSULTA
        if elo >= ELO_MINIMO && mapa == "Arabia" && duracion >= DURACION_MINIMA_SEGUNDOS {
            let match_id = record.get(idx_match_id).unwrap_or("").to_string();
            let winner_str = record.get(idx_winner).unwrap_or("0");
            
            // SOLUCIÓN AL BUG DEL GANADOR: Parseo directo del target original numérico (0.0 o 1.0)
            let winner_target: f64 = winner_str.parse().unwrap_or(0.0);

            let p1_vil: f64 = record.get(idx_p1_vil).unwrap_or("0").parse().unwrap_or(0.0);
            let p2_vil: f64 = record.get(idx_p2_vil).unwrap_or("0").parse().unwrap_or(0.0);

            partidas_candidatas.push(PartidaFiltrada {
                match_id,
                avg_elo: elo,
                winner_target,
                p1_villagers_snapshot: p1_vil,
                p2_villagers_snapshot: p2_vil,
            });
        }
    }
    println!("Se identificaron {} partidas que cumplen con los criterios estructurales.", partidas_candidatas.len());

    // 2. EXTRACCIÓN MÚLTI-HILO DE LOS LOGS DE INPUTS (Hasta minuto 20)
    println!("Procesando logs de acciones detalladas en paralelo...");
    
    let registros_limpios: Vec<FilaLimpia> = partidas_candidatas.par_iter().filter_map(|partida| {
        let ruta_input_match = format!("{}/{}_inputs.csv", carpeta_inputs, partida.match_id);
        
        if !Path::new(&ruta_input_match).exists() {
            return None; // Salto seguro si el archivo individual de inputs no se encuentra
        }

        let mut input_rdr = match ReaderBuilder::new().from_path(&ruta_input_match) {
            Ok(r) => r,
            Err(_) => return None,
        };

        let in_headers = match input_rdr.headers() {
            Ok(h) => h,
            Err(_) => return None,
        };
        
        let idx_ts = in_headers.iter().position(|h| h == "ts_seconds").unwrap_or(1);
        let idx_type = in_headers.iter().position(|h| h == "type").unwrap_or(3);
        let idx_player = in_headers.iter().position(|h| h == "player").unwrap_or(6);

        let (mut p1_act, mut p2_act) = (0.0, 0.0);
        let (mut p1_queues, mut p2_queues) = (0.0, 0.0);
        let (mut p1_builds, mut p2_builds) = (0.0, 0.0);
        let (mut p1_orders, mut p2_orders) = (0.0, 0.0);

        for row_res in input_rdr.records() {
            if let Ok(row) = row_res {
                let ts: i32 = row.get(idx_ts).unwrap_or("0").parse().unwrap_or(0);
                
                // Restricción temporal estricta: No procesar acciones ocurridas tras el minuto 20 (1200s)
                if ts > TIEMPO_SNAPSHOT_SEGUNDOS {
                    break; 
                }

                let jugador = row.get(idx_player).unwrap_or("");
                let tipo_accion = row.get(idx_type).unwrap_or("");

                if jugador == "p1" {
                    p1_act += 1.0;
                    match tipo_accion {
                        "Queue" => p1_queues += 1.0,
                        "Build" => p1_builds += 1.0,
                        "Order" | "Target" => p1_orders += 1.0,
                        _ => {}
                    }
                } else if jugador == "p2" {
                    p2_act += 1.0;
                    match tipo_accion {
                        "Queue" => p2_queues += 1.0,
                        "Build" => p2_builds += 1.0,
                        "Order" | "Target" => p2_orders += 1.0,
                        _ => {}
                    }
                }
            }
        }

        // Computación de Atributos Diferenciales Simétricos
        let diff_villagers_snapshot = partida.p1_villagers_snapshot - partida.p2_villagers_snapshot;
        let diff_apm = (p1_act - p2_act) / 20.0; // Acciones netas por minuto promedio en los primeros 20 min
        let diff_queues = p1_queues - p2_queues;
        let diff_builds = p1_builds - p2_builds;
        let diff_orders = p1_orders - p2_orders;

        Some(FilaLimpia {
            match_id: partida.match_id.clone(),
            avg_elo: partida.avg_elo,
            winner: partida.winner_target,
            diff_villagers_snapshot,
            diff_apm,
            diff_queues,
            diff_builds,
            diff_orders,
        })
    }).collect();

    // 3. ESCRITURA DEL DATASET CENTRALIZADO LIMPIO
    let mut wtr = csv::Writer::from_path(ruta_salida)?;

    wtr.write_record(&[
        "match_id",
        "avg_elo",
        "winner",
        "diff_villagers_snapshot",
        "diff_apm",
        "diff_queues",
        "diff_builds",
        "diff_orders",
    ])?;

    for reg in registros_limpios {
        wtr.write_record(&[
            reg.match_id,
            reg.avg_elo.to_string(),
            reg.winner.to_string(),
            reg.diff_villagers_snapshot.to_string(),
            reg.diff_apm.to_string(),
            reg.diff_queues.to_string(),
            reg.diff_builds.to_string(),
            reg.diff_orders.to_string(),
        ])?;
    }

    wtr.flush()?;
    println!("¡Pipeline finalizado con éxito! Datos guardados en: {}", ruta_salida);
    Ok(())
}
