use csv::Reader;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::HashMap;
use std::error::Error;

// --- ESTRUCTURAS DE DATOS ---

#[derive(Clone, Debug)]
struct Fila {
    caracteristicas: Vec<String>,
    ganador: String,
}

#[derive(Debug, Clone)]
enum Nodo {
    Hoja(String),
    Decision {
        col_idx: usize,
        hijos: HashMap<String, Box<Nodo>>,
        prediccion_por_defecto: String,
    },
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("Iniciando Motor de Árbol de Decisión...\n");

    // 1. CARGAR DATOS
    let mut rdr = Reader::from_path("data_discreta.csv")?;
    let headers = rdr.headers()?.clone();
    let indice_winner = 8; // Asegúrate de que este es el índice correcto en tu CSV

    let mut dataset: Vec<Fila> = Vec::new();
    for result in rdr.records() {
        let record = result?;
        let mut caracteristicas = Vec::new();
        let mut ganador = String::new();

        for (i, campo) in record.iter().enumerate() {
            if i == indice_winner {
                ganador = campo.to_string();
            } else {
                caracteristicas.push(campo.to_string());
            }
        }
        dataset.push(Fila {
            caracteristicas,
            ganador,
        });
    }

    // 2. MEZCLAR Y DIVIDIR (70% Train / 30% Test)
    let mut rng = thread_rng();
    dataset.shuffle(&mut rng);

    let split_idx = (dataset.len() as f64 * 0.7) as usize;
    let (train_set, test_set) = dataset.split_at(split_idx);
    let train_set = train_set.to_vec();
    let test_set = test_set.to_vec();

    println!("Dataset total: {} partidas", dataset.len());
    println!(
        "  -> Set de Entrenamiento (70%): {} partidas",
        train_set.len()
    );
    println!("  -> Set de Testeo (30%): {} partidas\n", test_set.len());

    // 3. 10-FOLD CROSS VALIDATION (VARIANDO PARÁMETROS)
    println!("================ CROSS VALIDATION (10-FOLD) ================");
    let parametros_a_probar = vec![5, 10]; // Probaremos Profundidad Máxima = 5 y 10
    let mut mejor_parametro = 0;
    let mut mejor_precision_cv = 0.0;

    for &max_profundidad in &parametros_a_probar {
        let precision_promedio = k_fold_cv(&train_set, 10, max_profundidad);
        println!(
            "Profundidad Máxima: {:02} -> Precisión Promedio CV: {:.2}%",
            max_profundidad,
            precision_promedio * 100.0
        );

        if precision_promedio > mejor_precision_cv {
            mejor_precision_cv = precision_promedio;
            mejor_parametro = max_profundidad;
        }
    }

    // 4. ENTRENAMIENTO FINAL Y TESTEO REAL
    println!("\n================ PRUEBA FINAL EN TEST SET ================");
    println!(
        "Entrenando modelo final con todo el 70% usando el mejor parámetro (Profundidad: {})...",
        mejor_parametro
    );

    let num_caracteristicas = train_set[0].caracteristicas.len();
    let mut columnas_disponibles = Vec::new();

    for col in 0..num_caracteristicas {
        if col == 0 {
            println!(
                "✂ Columna índice {} excluida permanentemente (ID único detectado)",
                col
            );
            continue;
        }
        // Usamos un HashMap para contar cuántos valores distintos tiene esta columna
        let mut valores_unicos = HashMap::new();
        for fila in &train_set {
            valores_unicos.insert(&fila.caracteristicas[col], true);
        }

        // Si la columna tiene más de un valor único, es útil para el árbol
        if valores_unicos.len() > 1 {
            if valores_unicos.len() > 20 {
                println!(
                    "Columna índice {} descartada por alta cardinalidad ({} valores únicos). ¡Peligro de sobreajuste!",
                    col,
                    valores_unicos.len()
                );
                continue;
            }
            columnas_disponibles.push(col);
        } else {
            println!(
                "⚠ Columna índice {} descartada (Solo contiene ceros o un único valor)",
                col
            );
        }
    }
    println!(
        "Columnas útiles conservadas para el entrenamiento: {} de {}\n",
        columnas_disponibles.len(),
        num_caracteristicas
    );

    let modelo_final = construir_arbol(&train_set, columnas_disponibles, mejor_parametro, 0);
    println!("Estructura del Nodo Raíz: {:#?}", modelo_final);

    // Evaluar contra el 30% que nunca ha visto el modelo
    let mut aciertos = 0;
    for fila in &test_set {
        let prediccion = predecir(&modelo_final, fila);
        if prediccion == fila.ganador {
            aciertos += 1;
        }
    }

    let precision_final = aciertos as f64 / test_set.len() as f64;
    println!(
        "Precisión en datos desconocidos (30% Test): \x1b[1;32m{:.2}%\x1b[0m",
        precision_final * 100.0
    );
    println!("==========================================================");

    Ok(())
}

// --- FUNCIONES MATEMÁTICAS ---

/// Calcula la Entropía (H) de un conjunto de datos
fn calcular_entropia(filas: &[Fila]) -> f64 {
    let total = filas.len() as f64;
    if total == 0.0 {
        return 0.0;
    }

    let mut conteos = HashMap::new();
    for fila in filas {
        *conteos.entry(&fila.ganador).or_insert(0) += 1;
    }

    let mut entropia = 0.0;
    for &conteo in conteos.values() {
        let p = conteo as f64 / total;
        entropia -= p * p.log2();
    }
    entropia
}

/// Calcula la Ganancia de Información (IG) para una columna específica
fn calcular_ganancia(filas: &[Fila], col_idx: usize) -> f64 {
    let entropia_total = calcular_entropia(filas);
    let total_filas = filas.len() as f64;

    // Agrupar filas por el valor que tienen en la columna evaluada
    let mut subconjuntos: HashMap<&String, Vec<Fila>> = HashMap::new();
    for fila in filas {
        let valor_caracteristica = &fila.caracteristicas[col_idx];
        subconjuntos
            .entry(valor_caracteristica)
            .or_insert_with(Vec::new)
            .push(fila.clone());
    }

    let mut entropia_subconjuntos = 0.0;
    for subconjunto in subconjuntos.values() {
        let peso = subconjunto.len() as f64 / total_filas;
        entropia_subconjuntos += peso * calcular_entropia(subconjunto);
    }

    entropia_total - entropia_subconjuntos
}

/// Determina la clase más frecuente (para las hojas del árbol)
fn clase_mayoritaria(filas: &[Fila]) -> String {
    let mut conteos = HashMap::new();
    for fila in filas {
        *conteos.entry(fila.ganador.clone()).or_insert(0) += 1;
    }
    conteos
        .into_iter()
        .max_by_key(|&(_, count)| count)
        .map(|(val, _)| val)
        .unwrap_or_else(|| "0".to_string())
}

// --- CONSTRUCCIÓN DEL ÁRBOL (ID3) ---

fn construir_arbol(
    filas: &[Fila],
    columnas_disponibles: Vec<usize>,
    max_prof: usize,
    prof_actual: usize,
) -> Nodo {
    let clase_mayoria = clase_mayoritaria(filas);

    // Casos base de parada
    if prof_actual >= max_prof || columnas_disponibles.is_empty() || calcular_entropia(filas) == 0.0
    {
        return Nodo::Hoja(clase_mayoria);
    }

    // Buscar la mejor columna (Mayor Information Gain)
    let mut mejor_ganancia = -1.0;
    let mut mejor_columna = 0;

    for &col_idx in &columnas_disponibles {
        let ganancia = calcular_ganancia(filas, col_idx);
        if ganancia > mejor_ganancia {
            mejor_ganancia = ganancia;
            mejor_columna = col_idx;
        }
    }

    if mejor_ganancia <= 0.0 {
        return Nodo::Hoja(clase_mayoria.clone());
    }

    // Agrupar los datos dividiéndolos por la mejor columna
    let mut particiones: HashMap<String, Vec<Fila>> = HashMap::new();
    for fila in filas {
        let valor = fila.caracteristicas[mejor_columna].clone();
        particiones
            .entry(valor)
            .or_insert_with(Vec::new)
            .push(fila.clone());
    }

    // Quitar la columna usada para no volver a evaluar
    let mut nuevas_columnas = columnas_disponibles.clone();
    nuevas_columnas.retain(|&c| c != mejor_columna);

    // Construir ramas (hijos) recursivamente
    let mut hijos = HashMap::new();
    for (valor, sub_filas) in particiones {
        let nodo_hijo = construir_arbol(
            &sub_filas,
            nuevas_columnas.clone(),
            max_prof,
            prof_actual + 1,
        );
        hijos.insert(valor, Box::new(nodo_hijo));
    }

    Nodo::Decision {
        col_idx: mejor_columna,
        hijos,
        prediccion_por_defecto: clase_mayoria,
    }
}

// --- EVALUACIÓN Y PREDICCIÓN ---

fn predecir(nodo: &Nodo, fila: &Fila) -> String {
    match nodo {
        Nodo::Hoja(prediccion) => prediccion.clone(),
        Nodo::Decision {
            col_idx,
            hijos,
            prediccion_por_defecto,
        } => {
            let valor_fila = &fila.caracteristicas[*col_idx];
            // Si el árbol nunca vio este valor en el entrenamiento, retorna la mayoría
            if let Some(hijo) = hijos.get(valor_fila) {
                predecir(hijo, fila)
            } else {
                prediccion_por_defecto.clone()
            }
        }
    }
}

/// Ejecuta K-Fold Cross Validation y retorna la precisión promedio
fn k_fold_cv(filas: &[Fila], k: usize, max_profundidad: usize) -> f64 {
    let tamano_fold = filas.len() / k;
    let mut precision_total = 0.0;

    for i in 0..k {
        let inicio_test = i * tamano_fold;
        let fin_test = if i == k - 1 {
            filas.len()
        } else {
            (i + 1) * tamano_fold
        };

        let mut train_fold = Vec::new();
        let mut test_fold = Vec::new();

        for (idx, fila) in filas.iter().enumerate() {
            if idx >= inicio_test && idx < fin_test {
                test_fold.push(fila.clone());
            } else {
                train_fold.push(fila.clone());
            }
        }

        let num_caract = train_fold[0].caracteristicas.len();
        let arbol = construir_arbol(&train_fold, (0..num_caract).collect(), max_profundidad, 0);

        let mut aciertos = 0;
        for test_fila in &test_fold {
            if predecir(&arbol, test_fila) == test_fila.ganador {
                aciertos += 1;
            }
        }
        precision_total += aciertos as f64 / test_fold.len() as f64;
    }

    precision_total / k as f64
}
