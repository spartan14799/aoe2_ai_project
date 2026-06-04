mod data_loader; // Asumimos que conserva la lógica de lectura base
mod model;

use std::error::Error;
use csv::ReaderBuilder;
use rand::seq::SliceRandom;
use rand::thread_rng;

/// Estructura para definir las configuraciones de parámetros que vamos a evaluar (Ajuste de Hiperparámetros)
#[derive(Debug, Clone, Copy)]
struct Hiperparametros {
    alpha: f64,     // Tasa de aprendizaje
    epocas: usize,  // Iteraciones
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("====================================================");
    println!("=== EXPERIMENTO DE ML: PREDICCIÓN AOE2 (ARABIA) ===");
    println!("====================================================\n");

    let ruta_dataset_limpio = "../data/processed/clear_dataset.csv";

    // 1. CARGA COMPLETA Y NORMALIZACIÓN DE DATOS
    let (x_completo, y_completo) = cargar_y_normalizar_completo(ruta_dataset_limpio)?;
    let num_muestras = x_completo.len();
    let num_features = x_completo[0].len();

    println!("Dataset cargado con éxito.");
    println!(" -> Total de partidas: {}", num_muestras);
    println!(" -> Atributos por partida: {}\n", num_features);

    // Mezclar los índices de todo el dataset para garantizar que los pliegues del Cross-Validation sean aleatorios
    let mut indices: Vec<usize> = (0..num_muestras).collect();
    let mut rng = thread_rng();
    indices.shuffle(&mut rng);

    // 2. CONFIGURACIÓN DEL BARRIDO DE PARÁMETROS (Mínimo 2 ejecuciones diferentes como pidió el profesor)
    // Definimos 3 configuraciones diferentes para encontrar la óptima (Grid Sweep / Selección Evolutiva Básica)
    let configuraciones = vec![
        Hiperparametros { alpha: 0.2, epocas: 400 },  // Configuración 1: Aprendizaje rápido, pocas épocas
        Hiperparametros { alpha: 0.1, epocas: 800 },  // Configuración 2: Aprendizaje moderado
        Hiperparametros { alpha: 0.05, epocas: 1200 }, // Configuración 3: Aprendizaje lento pero profundo
    ];

    let mut mejor_config = configuraciones[0];
    let mut mejor_precision_cv = 0.0;

    println!("=== FASE 1: OPTIMIZACIÓN MEDIANTE 10-FOLD CROSS-VALIDATION ===");
    
    for (i, config) in configuraciones.iter().enumerate() {
        println!("\n[Ejecución #{}] Evaluando Parámetros: Alpha = {}, Épocas = {}", i + 1, config.alpha, config.epocas);
        
        // Ejecutar Validación Cruzada de 10 Pliegues para esta configuración
        let precision_cv = evaluar_10_fold(&x_completo, &y_completo, &indices, *config);
        println!(" -> Precisión Promedio (10-Fold CV): {:.2}%", precision_cv * 100.0);

        // Guardar la mejor configuración basada en el rendimiento del Cross-Validation
        if precision_cv > mejor_precision_cv {
            mejor_precision_cv = precision_cv;
            mejor_config = *config;
        }
    }

    println!("\n====================================================");
    println!("¡Optimización terminada! Ganador: {:?}", mejor_config);
    println!("Mejor rendimiento en Cross-Validation: {:.2}%", mejor_precision_cv * 100.0);
    println!("====================================================\n");

    // 3. FASE FINAL: ENTRENAMIENTO CON SPLIT 70/30 USANDO LOS MEJORES PARÁMETROS
    println!("=== FASE 2: EVALUACIÓN FINAL (SPLIT 70/30) ===");
    
    let punto_corte = ((num_muestras as f64) * 0.70).round() as usize;

    let mut x_train = Vec::with_capacity(punto_corte);
    let mut y_train = Vec::with_capacity(punto_corte);
    let mut x_test = Vec::with_capacity(num_muestras - punto_corte);
    let mut y_test = Vec::with_capacity(num_muestras - punto_corte);

    for (contador, &idx) in indices.iter().enumerate() {
        if contador < punto_corte {
            x_train.push(x_completo[idx].clone());
            y_train.push(y_completo[idx]);
        } else {
            x_test.push(x_completo[idx].clone());
            y_test.push(y_completo[idx]);
        }
    }

    println!(" -> Set de Entrenamiento Final (70%): {} partidas", x_train.len());
    println!(" -> Set de Prueba Final (30%):        {} partidas", x_test.len());

    // Instanciar y entrenar el modelo definitivo
    let mut modelo_final = model::RegresionLogistica::new(num_features);
    modelo_final.entrenar(&x_train, &y_train, mejor_config.alpha, mejor_config.epocas);

    // Evaluar la precisión final con el 30% de datos ocultos (Test Set)
    let mut aciertos_test = 0.0;
    for i in 0..x_test.len() {
        let prediccion = modelo_final.predecir(&x_test[i]);
        if prediccion == y_test[i] {
            aciertos_test += 1.0;
        }
    }
    let precision_final = aciertos_test / (x_test.len() as f64);

    println!("\n====================================================");
    println!("=== CONCLUSIÓN DEL EXPERIMENTO ===");
    println!("Precisión del modelo en datos reales nunca vistos (Test Set): {:.2}%", precision_final * 100.0);
    println!("Pesos Finales de los Atributos: {:?}", modelo_final.pesos);
    println!("====================================================");

    Ok(())
}

/// Implementación estricta de Validación Cruzada de 10 Pliegues (10-Fold Cross-Validation) desde cero
fn evaluar_10_fold(x: &[Vec<f64>], y: &[f64], indices_mezclados: &[usize], config: Hiperparametros) -> f64 {
    let k = 10;
    let num_muestras = x.len();
    let tamano_pliegue = num_muestras / k;
    let mut precisiones_pliegues = Vec::new();

    for fold in 0..k {
        // Determinar qué rango de índices mezclados será el set de validación en este turno
        let inicio_val = fold * tamano_pliegue;
        // El último pliegue se lleva los residuos sobrantes si la división no es exacta
        let fin_val = if fold == k - 1 { num_muestras } else { (fold + 1) * tamano_pliegue };

        let mut x_train_fold = Vec::new();
        let mut y_train_fold = Vec::new();
        let mut x_val_fold = Vec::new();
        let mut y_val_fold = Vec::new();

        // Construir los conjuntos Train y Validation para el pliegue actual
        for (contador, &idx) in indices_mezclados.iter().enumerate() {
            if contador >= inicio_val && contador < fin_val {
                x_val_fold.push(x[idx].clone());
                y_val_fold.push(y[idx]);
            } else {
                x_train_fold.push(x[idx].clone());
                y_train_fold.push(y[idx]);
            }
        }

        // Entrenar un modelo temporal para este pliegue
        let mut modelo_fold = model::RegresionLogistica::new(x[0].len());
        // Desactivamos logs internos extensos para no saturar la consola durante el CV
        modelo_fold.entrenar(&x_train_fold, &y_train_fold, config.alpha, config.epocas);

        // Evaluar el rendimiento en el pliegue de validación
        let mut aciertos = 0.0;
        for i in 0..x_val_fold.len() {
            if modelo_fold.predecir(&x_val_fold[i]) == y_val_fold[i] {
                aciertos += 1.0;
            }
        }
        let precision_fold = aciertos / (x_val_fold.len() as f64);
        precisiones_pliegues.push(precision_fold);
    }

    // Calcular el promedio matemático de los 10 pliegues
    let suma_precisiones: f64 = precisiones_pliegues.iter().sum();
    suma_precisiones / (k as f64)
}

/// Función auxiliar para cargar el archivo limpio completo y aplicar Z-Score unificado
fn cargar_y_normalizar_completo(ruta: &str) -> Result<(Vec<Vec<f64>>, Vec<f64>), Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new().from_path(ruta)?;
    let headers = rdr.headers()?.clone();
    
    let idx_winner = headers.iter().position(|h| h == "winner").ok_or("Falta winner")?;
    let idx_match_id = headers.iter().position(|h| h == "match_id").ok_or("Falta match_id")?;
    
    let idx_features: Vec<usize> = headers.iter().enumerate()
        .filter(|&(i, _)| i != idx_winner && i != idx_match_id)
        .map(|(i, _)| i)
        .collect();

    let mut x_crudo = Vec::new();
    let mut y = Vec::new();

    for result in rdr.records() {
        let record = result?;
        let objetivo: f64 = record.get(idx_winner).unwrap_or("0").parse().unwrap_or(0.0);
        y.push(objetivo);

        let mut fila = Vec::with_capacity(idx_features.len());
        for &idx in &idx_features {
            let val: f64 = record.get(idx).unwrap_or("0").parse().unwrap_or(0.0);
            fila.push(val);
        }
        x_crudo.push(fila);
    }

    // Z-Score unificado
    let n_muestras = x_crudo.len();
    let n_features = idx_features.len();
    let mut medias = vec![0.0; n_features];
    let mut desviaciones = vec![0.0; n_features];

    for j in 0..n_features {
        let mut suma = 0.0;
        for i in 0..n_muestras { suma += x_crudo[i][j]; }
        medias[j] = suma / (n_muestras as f64);
    }

    for j in 0..n_features {
        let mut suma_var = 0.0;
        for i in 0..n_muestras {
            let d = x_crudo[i][j] - medias[j];
            suma_var += d * d;
        }
        desviaciones[j] = (suma_var / (n_muestras as f64)).sqrt() + 1e-8;
    }

    let mut x_norm = vec![vec![0.0; n_features]; n_muestras];
    for i in 0..n_muestras {
        for j in 0..n_features {
            x_norm[i][j] = (x_crudo[i][j] - medias[j]) / desviaciones[j];
        }
    }

    Ok((x_norm, y))
}
