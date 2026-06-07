use csv::{Reader, Writer};
use std::collections::{HashMap, HashSet};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== FASE 0: INGENIERÍA DE DATOS, CRUCE Y LIMPIEZA ===");

    let ruta_eras_crudas = "data_cruda.csv";
    let ruta_nuevas_metricas = "nuevas_metricas.csv";
    let ruta_salida = "data_preparada.csv";

    // 1. LEER METRICAS NUEVAS EN MEMORIA
    println!("1. Leyendo '{}'...", ruta_nuevas_metricas);
    let mut rdr_metrics = Reader::from_path(ruta_nuevas_metricas)?;
    let metrics_headers = rdr_metrics.headers()?.clone();

    let metrics_match_id_idx = metrics_headers
        .iter()
        .position(|h| h == "match_id")
        .ok_or("No 'match_id' column found in nuevas_metricas.csv")?;

    let mut extra_metrics_indices = Vec::new();
    let mut extra_metrics_headers = Vec::new();
    for (idx, name) in metrics_headers.iter().enumerate() {
        if name != "match_id" && name != "avg_elo" && name != "winner" {
            extra_metrics_indices.push(idx);
            extra_metrics_headers.push(name.to_string());
        }
    }

    let mut mapa_nuevas_metricas: HashMap<String, Vec<String>> = HashMap::new();
    for result in rdr_metrics.records() {
        let record = result?;
        let match_id = record.get(metrics_match_id_idx).unwrap_or("").to_string();
        let mut row_metrics = Vec::new();
        for &idx in &extra_metrics_indices {
            row_metrics.push(record.get(idx).unwrap_or("").to_string());
        }
        mapa_nuevas_metricas.insert(match_id, row_metrics);
    }
    println!(
        "   -> Se cargaron {} registros de métricas.",
        mapa_nuevas_metricas.len()
    );

    // 2. ANALIZAR ESTRUCTURA DE DATA CRUDA
    println!("2. Analizando estructura de '{}'...", ruta_eras_crudas);
    let mut rdr_cruda = Reader::from_path(ruta_eras_crudas)?;
    let cruda_headers = rdr_cruda.headers()?.clone();

    let cruda_match_id_idx = cruda_headers
        .iter()
        .position(|h| h == "match_id")
        .ok_or("No 'match_id' column found in data_cruda.csv")?;

    // Columnas preservadas de la data cruda
    let mut preserved_cols = Vec::new();
    for col_name in &[
        "winner", "avg_elo", "time", "map", "map_size", "duration", "p1_civ", "p2_civ",
    ] {
        if let Some(idx) = cruda_headers.iter().position(|h| h == *col_name) {
            preserved_cols.push((idx, col_name.to_string()));
        }
    }

    // Identificar eras
    let p1_f_idx = cruda_headers
        .iter()
        .position(|h| h == "p1 Feudal Age Time")
        .ok_or("No 'p1 Feudal Age Time' column found")?;
    let p2_f_idx = cruda_headers
        .iter()
        .position(|h| h == "p2 Feudal Age Time")
        .ok_or("No 'p2 Feudal Age Time' column found")?;
    let p1_c_idx = cruda_headers
        .iter()
        .position(|h| h == "p1 Castle Age Time")
        .ok_or("No 'p1 Castle Age Time' column found")?;
    let p2_c_idx = cruda_headers
        .iter()
        .position(|h| h == "p2 Castle Age Time")
        .ok_or("No 'p2 Castle Age Time' column found")?;
    let p1_i_idx = cruda_headers
        .iter()
        .position(|h| h == "p1 Imperial Age Time")
        .ok_or("No 'p1 Imperial Age Time' column found")?;
    let p2_i_idx = cruda_headers
        .iter()
        .position(|h| h == "p2 Imperial Age Time")
        .ok_or("No 'p2 Imperial Age Time' column found")?;

    // Identificar pares de jugadores (p1_ y p2_) para diferencias
    let mut player_diff_pairs = Vec::new();
    for (idx, name) in cruda_headers.iter().enumerate() {
        if name.starts_with("p1_") {
            let suffix = &name[3..];
            if suffix == "civ" {
                continue; // Saltar civilización ya que no es numérica
            }
            let p2_name = format!("p2_{}", suffix);
            if let Some(p2_idx) = cruda_headers.iter().position(|h| h == p2_name) {
                player_diff_pairs.push((idx, p2_idx, format!("diff_{}", suffix)));
            }
        }
    }

    // Construir la cabecera combinada temporal
    let mut headers = Vec::new();
    headers.push("match_id".to_string());

    for (_, name) in &preserved_cols {
        headers.push(name.clone());
    }

    headers.push("diff_feudal_time".to_string());
    headers.push("diff_castle_time".to_string());
    headers.push("diff_imperial_time".to_string());

    for (_, _, name) in &player_diff_pairs {
        headers.push(name.clone());
    }

    for name in &extra_metrics_headers {
        headers.push(name.clone());
    }

    // 3. PROCESAR Y CRUZAR LAS TABLAS (INNER JOIN)
    println!("3. Cruzando tablas y calculando diferencias...");
    let mut dataset_combinado = Vec::new();

    for result in rdr_cruda.records() {
        let record = result?;
        let match_id = record.get(cruda_match_id_idx).unwrap_or("").to_string();

        if let Some(extra_row) = mapa_nuevas_metricas.get(&match_id) {
            let mut fila = Vec::new();
            fila.push(match_id);

            // Valores preservados
            for &(idx, _) in &preserved_cols {
                fila.push(record.get(idx).unwrap_or("").to_string());
            }

            // Diffs de eras
            let p1_f: f64 = record.get(p1_f_idx).unwrap_or("0").parse().unwrap_or(0.0);
            let p2_f: f64 = record.get(p2_f_idx).unwrap_or("0").parse().unwrap_or(0.0);
            let p1_c: f64 = record.get(p1_c_idx).unwrap_or("0").parse().unwrap_or(0.0);
            let p2_c: f64 = record.get(p2_c_idx).unwrap_or("0").parse().unwrap_or(0.0);
            let p1_i: f64 = record.get(p1_i_idx).unwrap_or("0").parse().unwrap_or(0.0);
            let p2_i: f64 = record.get(p2_i_idx).unwrap_or("0").parse().unwrap_or(0.0);

            fila.push((p1_f - p2_f).to_string());
            fila.push((p1_c - p2_c).to_string());
            fila.push((p1_i - p2_i).to_string());

            // Diffs de economía/militares
            for &(p1_idx, p2_idx, _) in &player_diff_pairs {
                let p1_val: f64 = record.get(p1_idx).unwrap_or("0").parse().unwrap_or(0.0);
                let p2_val: f64 = record.get(p2_idx).unwrap_or("0").parse().unwrap_or(0.0);
                fila.push((p1_val - p2_val).to_string());
            }

            // Métricas extra de nuevas_metricas
            for val in extra_row {
                fila.push(val.clone());
            }

            dataset_combinado.push(fila);
        }
    }
    println!(
        "   -> Se obtuvieron {} filas en el join inner.",
        dataset_combinado.len()
    );

    // 4. IDENTIFICAR COLUMNAS ÚTILES (FILTRAR VARIANZA 0 Y MATCH_ID)
    println!("4. Analizando varianza y quitando columnas constantes (varianza 0 o todo ceros)...");
    let mut indices_a_conservar = Vec::new();

    for col_idx in 0..headers.len() {
        if col_idx == 0 {
            println!("   ✂ Descartando '{}' (ID único)", headers[col_idx]);
            continue;
        }

        let mut valores_unicos = HashSet::new();
        for fila in &dataset_combinado {
            let val = &fila[col_idx];
            if let Ok(num) = val.parse::<f64>() {
                valores_unicos.insert(format!("{:.6}", num));
            } else {
                valores_unicos.insert(val.clone());
            }
        }

        if valores_unicos.len() > 1 {
            indices_a_conservar.push(col_idx);
        } else {
            println!(
                "   ⚠ Descartando '{}' (Constante: contiene un solo valor o puros ceros)",
                headers[col_idx]
            );
        }
    }

    // 5. GUARDAR ARCHIVO FINAL LIMPIO
    println!("5. Escribiendo archivo final '{}'...", ruta_salida);
    let mut wtr = Writer::from_path(ruta_salida)?;

    let mut headers_filtrados = Vec::new();
    for &idx in &indices_a_conservar {
        headers_filtrados.push(headers[idx].clone());
    }
    wtr.write_record(&headers_filtrados)?;

    for fila in dataset_combinado {
        let mut fila_filtrada = Vec::new();
        for &idx in &indices_a_conservar {
            fila_filtrada.push(fila[idx].clone());
        }
        wtr.write_record(&fila_filtrada)?;
    }

    wtr.flush()?;
    println!(
        "✔ ¡Limpieza terminada! Pasamos de {} columnas a {} columnas útiles.",
        headers.len(),
        indices_a_conservar.len()
    );
    Ok(())
}
