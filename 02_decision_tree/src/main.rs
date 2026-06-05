use csv::Reader;
use std::collections::HashMap;
use std::error::Error;

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
    println!("Iniciando lector para discretización...");

    let mut rdr = Reader::from_path("data.csv")?;
    let indice_elo = 1;
    let indice_winner = 8;

    let mut muestras: Vec<DatoDiscretizacion> = Vec::new();

    for result in rdr.records() {
        let record = result?;
        if let (Some(elo_str), Some(ganador_str)) =
            (record.get(indice_elo), record.get(indice_winner))
        {
            if let Ok(elo_num) = elo_str.parse::<f64>() {
                muestras.push(DatoDiscretizacion {
                    valor: elo_num,
                    ganador: ganador_str.to_string(),
                });
            }
        }
    }

    muestras.sort_by(|a, b| a.valor.partial_cmp(&b.valor).unwrap());

    let intervalos_iniciales = inicializar_intervalos(&muestras);
    println!(
        "Intervalos iniciales únicos creados: {}",
        intervalos_iniciales.len()
    );

    let umbral_chi = 3.841;
    // REGLA DE SEGURIDAD: Cada rango debe tener al menos 200 partidas totales
    let min_partidas_por_rango = 200;

    println!(
        "Ejecutando ChiMerge Avanzado (Umbral: {}, Min Partidas: {})...",
        umbral_chi, min_partidas_por_rango
    );

    let intervalos_finales =
        ejecutar_chimerge(intervalos_iniciales, umbral_chi, min_partidas_por_rango);

    println!("\n================ CONFIGURACIÓN FINAL ================");
    println!(
        "Cantidad de clases/rangos distintos creados: \x1b[1;32m{}\x1b[0m",
        intervalos_finales.len()
    );
    println!("=====================================================");

    for (i, intervalo) in intervalos_finales.iter().enumerate() {
        let total: usize = intervalo.conteos.values().sum();
        println!(
            "  Rango {:02}: [{} a {}] -> Total partidas: {} {:?}",
            i + 1,
            intervalo.min_valor,
            intervalo.max_valor,
            total,
            intervalo.conteos
        );
    }

    Ok(())
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
) -> Vec<Intervalo> {
    loop {
        if intervalos.len() < 2 {
            break;
        }

        let mut menor_chi = f64::INFINITY;
        let mut indice_a_fusionar = None;
        let mut forzar_fusion_por_tamano = false;

        // PRIORIDAD 1: Buscar si hay algún intervalo que viole la regla del tamaño mínimo
        for i in 0..intervalos.len() {
            let total_partidas: usize = intervalos[i].conteos.values().sum();

            if total_partidas < min_partidas {
                forzar_fusion_por_tamano = true;

                // Evaluamos si es mejor fusionarlo con el vecino izquierdo o el derecho
                if i > 0 {
                    let chi = calcular_chi_cuadrado(&intervalos[i - 1], &intervalos[i]);
                    if chi < menor_chi {
                        menor_chi = chi;
                        indice_a_fusionar = Some(i - 1);
                    }
                }
                if i < intervalos.len() - 1 {
                    let chi = calcular_chi_cuadrado(&intervalos[i], &intervalos[i + 1]);
                    if chi < menor_chi {
                        menor_chi = chi;
                        indice_a_fusionar = Some(i);
                    }
                }
                // Rompemos el recorrido interno para resolver este intervalo pequeño de inmediato
                break;
            }
        }

        // PRIORIDAD 2: Si todos los intervalos son grandes, usamos el ChiMerge estándar
        if !forzar_fusion_por_tamano {
            for i in 0..intervalos.len() - 1 {
                let chi = calcular_chi_cuadrado(&intervalos[i], &intervalos[i + 1]);
                if chi < menor_chi {
                    menor_chi = chi;
                    indice_a_fusionar = Some(i);
                }
            }

            // Si el par más parecido ya supera el umbral estadístico, terminamos
            if menor_chi >= umbral {
                break;
            }
        }

        // Ejecutar la fusión física en el vector
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

fn calcular_entropia(datos: &[String]) -> f64 {
    let total_elementos = datos.len() as f64;
    if total_elementos == 0.0 {
        return 0.0;
    }

    // La etiqueta es el mismo dato
    let mut conteos = HashMap::new();
    for etiqueta in datos {
        // Primero consigue el dato de cuantos, 0 o 1
        let tracker = conteos.entry(etiqueta).or_insert(0);
        *tracker += 1;
    }

    // Fórmula de la entropía
    let mut entropia = 0.0;
    for &conteo in conteos.values() {
        let probabilidad = conteo as f64 / total_elementos;
        // Entropía = 0.0 - probabilidad * log_2(probabilidad)
        entropia -= probabilidad * probabilidad.log2();
    }
    entropia
}
