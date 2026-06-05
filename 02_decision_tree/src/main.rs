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
    let indice_elo = 1; // Columna: avg_elo
    let indice_winner = 8; // Columna: winner

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

    // 1. Ordenar muestras
    muestras.sort_by(|a, b| a.valor.partial_cmp(&b.valor).unwrap());

    // 2. Crear los intervalos iniciales por cada valor único
    let intervalos_iniciales = inicializar_intervalos(&muestras);
    println!(
        "Intervalos iniciales únicos creados: {}",
        intervalos_iniciales.len()
    );

    // 3. Definir el umbral estadístico (Chi-Square para 1 df al 95% de confianza)
    let umbral_chi = 3.841;
    println!(
        "Ejecutando ChiMerge con un umbral estadístico de {}...",
        umbral_chi
    );

    // 4. Ejecutar el algoritmo de fusión
    let intervalos_finales = ejecutar_chimerge(intervalos_iniciales, umbral_chi);

    // 5. Mostrar resultados en la terminal
    println!("\n================ CONFIGURACIÓN FINAL ================");
    println!("La columna 'avg_elo' se discretizó con éxito.");
    println!(
        "Cantidad de clases/rangos distintos creados: \x1b[1;32m{}\x1b[0m",
        intervalos_finales.len()
    );
    println!("=====================================================");

    println!("\nRangos estadísticos calculados para el Árbol de Decisión:");
    for (i, intervalo) in intervalos_finales.iter().enumerate() {
        println!(
            "  Rango {}: [{} a {}] -> Partidas analizadas: {:?}",
            i + 1,
            intervalo.min_valor,
            intervalo.max_valor,
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

fn ejecutar_chimerge(mut intervalos: Vec<Intervalo>, umbral: f64) -> Vec<Intervalo> {
    loop {
        // Si nos quedamos con un solo intervalo enorme, no podemos fusionar más
        if intervalos.len() < 2 {
            break;
        }

        let mut menor_chi = f64::INFINITY;
        let mut indice_a_fusionar = None;

        // Evaluamos todos los pares de intervalos adyacentes
        for i in 0..intervalos.len() - 1 {
            let chi = calcular_chi_cuadrado(&intervalos[i], &intervalos[i + 1]);
            if chi < menor_chi {
                menor_chi = chi;
                indice_a_fusionar = Some(i);
            }
        }

        // CONCEPTO RUST: Criterio de parada estadística.
        // Si el par con menor diferencia supera el umbral, significa que todos los intervalos
        // actuales ya son significativamente distintos entre sí. ¡Terminamos!
        if menor_chi >= umbral {
            break;
        }

        // Ejecutamos la fusión del intervalo `idx` y el `idx + 1`
        if let Some(idx) = indice_a_fusionar {
            // .remove() elimina el elemento del vector y desplaza los demás. Es costoso pero seguro.
            let siguiente = intervalos.remove(idx + 1);
            let actual = &mut intervalos[idx];

            // Expandimos el rango del intervalo actual para que absorba al siguiente
            actual.max_valor = siguiente.max_valor;

            // Combinamos los HashMap de conteos de victorias
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
