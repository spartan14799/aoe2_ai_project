use csv::{Reader, Writer};
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::Write;

#[derive(Debug)]
struct DatoDiscretizacion {
    valor: f64,
    ganador: String,
}

#[derive(Debug, Clone)]
struct Intervalo {
    min_valor: f64,
    max_valor: f64,
    conteos: HashMap<String, usize>,
}

fn main() -> std::result::Result<(), Box<dyn Error>> {
    println!("=== FASE 1: DISCRETIZACIÓN DE DATOS ===");

    let ruta_archivo = "data_preparada.csv";
    let ruta_salida = "data_discreta.csv";
    let ruta_reporte = "reporte_rangos.txt";

    let mut rdr_columnas = Reader::from_path(ruta_archivo)?;
    let headers = rdr_columnas.headers()?.clone();
    let total_columnas = headers.len();

    let columnas_ignoradas = vec!["winner", "avg_elo", "p1_civ", "p2_civ"];

    let indice_winner = headers
        .iter()
        .position(|h| h == "winner")
        .ok_or("No 'winner' column found in data_preparada.csv")?;

    let mut columnas_a_discretizar = Vec::new();
    for col in 0..total_columnas {
        let nombre_columna = &headers[col];

        // 2. Si el nombre de la columna NO está en la lista negra, la discretizamos
        if !columnas_ignoradas.contains(&nombre_columna) {
            columnas_a_discretizar.push(col);
        } else {
            println!(
                "⏭ Saltando discretización para la columna categórica/ignorada: '{}'",
                nombre_columna
            );
        }
    }

    let umbral_chi = 3.841;
    let min_partidas = 200;
    let max_partidas = 50000;

    let mut mapa_de_cortes: HashMap<usize, Vec<f64>> = HashMap::new();
    let mut archivo_reporte = File::create(ruta_reporte)?;
    writeln!(
        archivo_reporte,
        "=== DICCIONARIO DE RANGOS DE DISCRETIZACIÓN ===\n"
    )?;

    for &col_idx in &columnas_a_discretizar {
        if let Ok((cortes, intervalos_finales)) = procesar_columna(
            ruta_archivo,
            col_idx,
            indice_winner,
            umbral_chi,
            min_partidas,
            max_partidas,
        ) {
            mapa_de_cortes.insert(col_idx, cortes);

            // Escribir el detalle en el reporte
            writeln!(archivo_reporte, "-----------------------------------------")?;
            writeln!(
                archivo_reporte,
                "COLUMNA NOMBRE: '{}' (ÍNDICE: {})",
                &headers[col_idx], col_idx
            )?;
            for (i, intervalo) in intervalos_finales.iter().enumerate() {
                writeln!(
                    archivo_reporte,
                    "  Rango {:02}: [{} a {}]",
                    i, intervalo.min_valor, intervalo.max_valor
                )?;
            }
        }
    }

    println!("✔ Reporte de rangos generado en: {}", ruta_reporte);
    println!("Generando nuevo archivo '{}'...", ruta_salida);

    // --- LÓGICA DE TRANSFORMACIÓN DE CSV ---
    let mut rdr_transform = Reader::from_path(ruta_archivo)?;
    let mut wtr = Writer::from_path(ruta_salida)?;
    wtr.write_record(&rdr_transform.headers()?.clone())?;

    for result in rdr_transform.records() {
        let record = result?;
        let mut nueva_fila = Vec::new();

        for (idx, campo) in record.iter().enumerate() {
            if let Some(cortes) = mapa_de_cortes.get(&idx) {
                if let Ok(valor_num) = campo.parse::<f64>() {
                    let rango_id = transformar_a_rango(valor_num, cortes);
                    nueva_fila.push(rango_id.to_string());
                } else {
                    nueva_fila.push(campo.to_string());
                }
            } else {
                nueva_fila.push(campo.to_string());
            }
        }
        wtr.write_record(&nueva_fila)?;
    }
    wtr.flush()?;
    println!("✔ Archivo discretizado generado con éxito.");

    Ok(())
}

fn transformar_a_rango(valor: f64, puntos_de_corte: &[f64]) -> usize {
    for (indice, &corte) in puntos_de_corte.iter().enumerate() {
        if valor <= corte {
            return indice;
        }
    }
    puntos_de_corte.len()
}

fn procesar_columna(
    ruta: &str,
    indice_col: usize,
    indice_winner: usize,
    umbral: f64,
    min: usize,
    max: usize,
) -> Result<(Vec<f64>, Vec<Intervalo>), Box<dyn Error>> {
    let mut rdr = Reader::from_path(ruta)?;
    let mut muestras = Vec::new();

    for result in rdr.records() {
        let record = result?;

        if let (Some(valor_str), Some(ganador_str)) =
            (record.get(indice_col), record.get(indice_winner))
        {
            if let Ok(valor_num) = valor_str.parse::<f64>() {
                muestras.push(DatoDiscretizacion {
                    valor: valor_num,
                    ganador: ganador_str.to_string(),
                });
            }
        }
    }

    if muestras.is_empty() {
        return Err("No numeric data found".into());
    }

    muestras.sort_by(|a, b| a.valor.partial_cmp(&b.valor).unwrap());
    let iniciales = inicializar_intervalos(&muestras);
    let finales = ejecutar_chimerge(iniciales, umbral, min, max);

    let mut cortes = Vec::new();
    if !finales.is_empty() {
        for i in 0..finales.len() - 1 {
            cortes.push(finales[i].max_valor);
        }
    }

    Ok((cortes, finales))
}

fn inicializar_intervalos(muestras: &[DatoDiscretizacion]) -> Vec<Intervalo> {
    let mut intervalos: Vec<Intervalo> = Vec::new();
    if muestras.is_empty() {
        return intervalos;
    }

    let mut intervalo_actual = Intervalo {
        min_valor: muestras[0].valor,
        max_valor: muestras[0].valor,
        conteos: HashMap::new(),
    };
    intervalo_actual
        .conteos
        .insert(muestras[0].ganador.clone(), 1);

    for muestra in &muestras[1..] {
        if muestra.valor == intervalo_actual.max_valor {
            *intervalo_actual
                .conteos
                .entry(muestra.ganador.clone())
                .or_insert(0) += 1;
        } else {
            intervalos.push(intervalo_actual);
            intervalo_actual = Intervalo {
                min_valor: muestra.valor,
                max_valor: muestra.valor,
                conteos: HashMap::new(),
            };
            intervalo_actual.conteos.insert(muestra.ganador.clone(), 1);
        }
    }
    intervalos.push(intervalo_actual);
    intervalos
}

fn calcular_chi_cuadrado(i1: &Intervalo, i2: &Intervalo) -> f64 {
    let o1_0 = *i1.conteos.get("0").unwrap_or(&0) as f64;
    let o1_1 = *i1.conteos.get("1").unwrap_or(&0) as f64;
    let o2_0 = *i2.conteos.get("0").unwrap_or(&0) as f64;
    let o2_1 = *i2.conteos.get("1").unwrap_or(&0) as f64;

    let total_i1 = o1_0 + o1_1;
    let total_i2 = o2_0 + o2_1;
    let total_general = total_i1 + total_i2;

    if total_general == 0.0 {
        return 0.0;
    }

    let total_clase_0 = o1_0 + o2_0;
    let total_clase_1 = o1_1 + o2_1;
    let mut chi_cuadrado = 0.0;

    let e1_0 = (total_i1 * total_clase_0) / total_general;
    if e1_0 > 0.0 {
        chi_cuadrado += (o1_0 - e1_0).powi(2) / e1_0;
    }
    let e1_1 = (total_i1 * total_clase_1) / total_general;
    if e1_1 > 0.0 {
        chi_cuadrado += (o1_1 - e1_1).powi(2) / e1_1;
    }
    let e2_0 = (total_i2 * total_clase_0) / total_general;
    if e2_0 > 0.0 {
        chi_cuadrado += (o2_0 - e2_0).powi(2) / e2_0;
    }
    let e2_1 = (total_i2 * total_clase_1) / total_general;
    if e2_1 > 0.0 {
        chi_cuadrado += (o2_1 - e2_1).powi(2) / e2_1;
    }

    chi_cuadrado
}

fn ejecutar_chimerge(
    mut intervalos: Vec<Intervalo>,
    umbral: f64,
    min_partidas: usize,
    max_partidas: usize,
) -> Vec<Intervalo> {
    loop {
        if intervalos.len() < 2 {
            break;
        }

        let mut menor_chi = f64::INFINITY;
        let mut indice_a_fusionar = None;
        let mut forzar_fusion_por_tamano = false;

        for i in 0..intervalos.len() {
            let total_partidas: usize = intervalos[i].conteos.values().sum();

            if total_partidas < min_partidas {
                forzar_fusion_por_tamano = true;
                let mut menor_chi_p1 = f64::INFINITY;
                let mut mejor_vecino = None;

                let mut evaluar_vecino = |vecino_idx: usize| {
                    let tamano_vecino: usize = intervalos[vecino_idx].conteos.values().sum();
                    let mut chi = calcular_chi_cuadrado(&intervalos[i], &intervalos[vecino_idx]);
                    if total_partidas + tamano_vecino > max_partidas {
                        chi += 1_000_000.0;
                    }
                    if chi < menor_chi_p1 {
                        menor_chi_p1 = chi;
                        mejor_vecino = Some(vecino_idx);
                    }
                };

                if i > 0 {
                    evaluar_vecino(i - 1);
                }
                if i < intervalos.len() - 1 {
                    evaluar_vecino(i + 1);
                }

                if let Some(vecino) = mejor_vecino {
                    if vecino < i {
                        indice_a_fusionar = Some(vecino);
                    } else {
                        indice_a_fusionar = Some(i);
                    }
                }
                break;
            }
        }

        if !forzar_fusion_por_tamano {
            for i in 0..intervalos.len() - 1 {
                let tamano_i: usize = intervalos[i].conteos.values().sum();
                let tamano_siguiente: usize = intervalos[i + 1].conteos.values().sum();
                if tamano_i + tamano_siguiente > max_partidas {
                    continue;
                }

                let chi = calcular_chi_cuadrado(&intervalos[i], &intervalos[i + 1]);
                if chi < menor_chi {
                    menor_chi = chi;
                    indice_a_fusionar = Some(i);
                }
            }
            if (menor_chi >= umbral && intervalos.len() <= 6) || indice_a_fusionar.is_none() {
                break;
            }
        }

        if let Some(idx) = indice_a_fusionar {
            let siguiente = intervalos.remove(idx + 1);
            let actual = &mut intervalos[idx];
            actual.max_valor = siguiente.max_valor;
            for (clase, conteo) in siguiente.conteos {
                *actual.conteos.entry(clase).or_insert(0) += conteo;
            }
        }
    }
    intervalos
}
