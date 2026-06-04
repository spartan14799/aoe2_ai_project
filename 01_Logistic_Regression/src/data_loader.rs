use std::error::Error;
use csv::ReaderBuilder;
use rand::seq::SliceRandom; // Para el mezclado aleatorio
use rand::thread_rng;

/// Estructura contenedora de los datos listos para el entrenamiento de Machine Learning
pub struct Dataset {
    pub x_train: Vec<Vec<f64>>, // Matriz de características de entrenamiento
    pub y_train: Vec<f64>,      // Vector de etiquetas de entrenamiento (0 o 1)
    pub x_test: Vec<Vec<f64>>,  // Matriz de características de prueba
    pub y_test: Vec<f64>,       // Vector de etiquetas de prueba
}

/// Carga el dataset limpio, aplica normalización Z-Score y realiza el split (Train/Test)
pub fn cargar_y_preparar_datos(ruta_csv: &str, proporcion_entrenamiento: f64) -> Result<Dataset, Box<dyn Error>> {
    println!("=== Cargando y Preparando Matrices Matemáticas ===");

    let mut rdr = ReaderBuilder::new().from_path(ruta_csv)?;
    
    // Identificar posiciones de columnas dinámicamente
    let headers = rdr.headers()?.clone();
    let idx_winner = headers.iter().position(|h| h == "winner").ok_or("Falta columna winner")?;
    let idx_match_id = headers.iter().position(|h| h == "match_id").ok_or("Falta columna match_id")?;

    // Guardaremos los índices de las columnas que SÍ son características (Features)
    let indices_caracteristicas: Vec<usize> = headers.iter().enumerate()
        .filter(|&(i, _)| i != idx_winner && i != idx_match_id)
        .map(|(i, _)| i)
        .collect();
    let num_features = indices_caracteristicas.len();

    let mut x_crudo: Vec<Vec<f64>> = Vec::new();
    let mut y_crudo: Vec<f64> = Vec::new();

    // 1. SEPARACIÓN DE VARIABLES (X, y)
    for result in rdr.records() {
        let record = result?;
        
        // Extraer la variable objetivo (y)
        let objetivo: f64 = record.get(idx_winner).unwrap_or("0").parse().unwrap_or(0.0);
        y_crudo.push(objetivo);

        // Extraer la fila de características (X)
        let mut fila_caracteristicas = Vec::with_capacity(num_features);
        for &idx in &indices_caracteristicas {
            let valor: f64 = record.get(idx).unwrap_or("0").parse().unwrap_or(0.0);
            fila_caracteristicas.push(valor);
        }
        x_crudo.push(fila_caracteristicas);
    }

    let num_muestras = x_crudo.len();
    if num_muestras == 0 {
        return Err("El archivo CSV no contiene registros válidos.".into());
    }
    println!("Registros totales cargados: {}", num_muestras);

    // 2. NORMALIZACIÓN Z-SCORE (Columna por Columna)
    // Inicializamos vectores para guardar la media (mu) y desviación estándar (sigma) de cada característica
    let mut medias = vec![0.0; num_features];
    let mut desviaciones = vec![0.0; num_features];

    // Calcular la Media (μ) para cada columna
    for j in 0..num_features {
        let mut suma = 0.0;
        for i in 0..num_muestras {
            suma += x_crudo[i][j];
        }
        medias[j] = suma / (num_muestras as f64);
    }

    // Calcular la Desviación Estándar (σ) para cada columna
    for j in 0..num_features {
        let mut suma_varianza = 0.0;
        for i in 0..num_muestras {
            let dif = x_crudo[i][j] - medias[j];
            suma_varianza += dif * dif;
        }
        // Usamos desviación estándar poblacional. Añadimos un épsilon pequeño (1e-8) para evitar división por cero
        desviaciones[j] = (suma_varianza / (num_muestras as f64)).sqrt() + 1e-8;
    }

    // Aplicar la fórmula: x_norm = (x - mu) / sigma
    let mut x_normalizado = vec![vec![0.0; num_features]; num_muestras];
    for i in 0..num_muestras {
        for j in 0..num_features {
            x_normalizado[i][j] = (x_crudo[i][j] - medias[j]) / desviaciones[j];
        }
    }
    println!("Escalado Z-Score completado de forma exitosa.");

    // 3. SHUFFLE Y SPLIT (Entrenamiento / Prueba)
    // Para no perder la sincronía entre X e y, mezclamos un vector de índices
    let mut indices: Vec<usize> = (0..num_muestras).collect();
    let mut rng = thread_rng();
    indices.shuffle(&mut rng); // Desordena aleatoriamente las partidas

    // Calcular el punto de corte (ej. 70%)
    let tamano_entrenamiento = ((num_muestras as f64) * proporcion_entrenamiento).round() as usize;

    let mut x_train = Vec::with_capacity(tamano_entrenamiento);
    let mut y_train = Vec::with_capacity(tamano_entrenamiento);
    let mut x_test = Vec::with_capacity(num_muestras - tamano_entrenamiento);
    let mut y_test = Vec::with_capacity(num_muestras - tamano_entrenamiento);

    // Distribuir los datos usando los índices mezclados
    for (contador, &idx) in indices.iter().enumerate() {
        if contador < tamano_entrenamiento {
            x_train.push(x_normalizado[idx].clone());
            y_train.push(y_crudo[idx]);
        } else {
            x_test.push(x_normalizado[idx].clone());
            y_test.push(y_crudo[idx]);
        }
    }

    println!("Split completado:");
    println!(" -> Set de Entrenamiento (Train): {} partidas", x_train.len());
    println!(" -> Set de Validación (Test):     {} partidas\n", x_test.len());

    Ok(Dataset {
        x_train,
        y_train,
        x_test,
        y_test,
    })
}
