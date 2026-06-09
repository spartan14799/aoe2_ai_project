use std::error::Error;
use csv::ReaderBuilder;
use rand::seq::SliceRandom; 
use rand::thread_rng;

pub struct Dataset {
    pub x_train: Vec<Vec<f64>>, 
    pub y_train: Vec<f64>,     
    pub x_test: Vec<Vec<f64>>,  
    pub y_test: Vec<f64>,      }

pub fn cargar_y_preparar_datos(ruta_csv: &str, proporcion_entrenamiento: f64) -> Result<Dataset, Box<dyn Error>> {
    println!("=== Cargando y Preparando Matrices Matemáticas ===");

    let mut rdr = ReaderBuilder::new().from_path(ruta_csv)?;
    let headers = rdr.headers()?.clone();

    let idx_winner = headers.iter().position(|h| h == "winner").ok_or("Falta columna winner")?;
    let idx_match_id = headers.iter().position(|h| h == "match_id").ok_or("Falta columna match_id")?;

    let indices_caracteristicas: Vec<usize> = headers.iter().enumerate()
        .filter(|&(i, _)| i != idx_winner && i != idx_match_id)
        .map(|(i, _)| i)
        .collect();
    let num_features = indices_caracteristicas.len();

    let mut x_crudo: Vec<Vec<f64>> = Vec::new();
    let mut y_crudo: Vec<f64> = Vec::new();

    for result in rdr.records() {
        let record = result?;
        let objetivo: f64 = record.get(idx_winner).unwrap_or("0").parse().unwrap_or(0.0);
        y_crudo.push(objetivo);
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
    
    //Normalización (Z-score)
    let mut medias = vec![0.0; num_features];
    let mut desviaciones = vec![0.0; num_features];

    //Media
    for j in 0..num_features {
        let mut suma = 0.0;
        for i in 0..num_muestras {
            suma += x_crudo[i][j];
        }
        medias[j] = suma / (num_muestras as f64);
    }
    //Desviacion
    for j in 0..num_features {
        let mut suma_varianza = 0.0;
        for i in 0..num_muestras {
            let dif = x_crudo[i][j] - medias[j];
            suma_varianza += dif * dif;
        }
        desviaciones[j] = (suma_varianza / (num_muestras as f64)).sqrt() + 1e-8;
    }

    let mut x_normalizado = vec![vec![0.0; num_features]; num_muestras];
    for i in 0..num_muestras {
        for j in 0..num_features {
            x_normalizado[i][j] = (x_crudo[i][j] - medias[j]) / desviaciones[j];
        }
    }
    println!("Escalado Z-Score completado de forma exitosa.");
    
    //Spliteo de datos aleatorios
    let mut indices: Vec<usize> = (0..num_muestras).collect();
    let mut rng = thread_rng();
    indices.shuffle(&mut rng); 

   
    let tamano_entrenamiento = ((num_muestras as f64) * proporcion_entrenamiento).round() as usize;

    let mut x_train = Vec::with_capacity(tamano_entrenamiento);
    let mut y_train = Vec::with_capacity(tamano_entrenamiento);
    let mut x_test = Vec::with_capacity(num_muestras - tamano_entrenamiento);
    let mut y_test = Vec::with_capacity(num_muestras - tamano_entrenamiento);


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
