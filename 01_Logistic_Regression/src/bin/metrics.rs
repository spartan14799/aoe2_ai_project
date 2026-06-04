use std::error::Error;
use csv::ReaderBuilder;

fn main() -> Result<(), Box<dyn Error>> {
    println!("====================================================");
    println!("=== CALCULADORA DE MÉTRICAS DE MACHINE LEARNING ===");
    println!("====================================================\n");

    let ruta_predicciones = "../data/test_predictions.csv";
    
    // Intentar abrir el archivo de salida del main
    let mut rdr = match ReaderBuilder::new().from_path(ruta_predicciones) {
        Ok(reader) => reader,
        Err(_) => {
            eprintln!("Error: No se encontró el archivo '{}'.", ruta_predicciones);
            eprintln!("Por favor, ejecuta primero 'cargo run' para entrenar el modelo y generar los datos.");
            return Ok(());
        }
    };

    // Inicializadores de la Matriz de Confusión (Corregidos)
    let mut tp = 0.0;     // True Positives (Verdaderos Positivos)
    let mut tn = 0.0;     // True Negatives (Verdaderos Negativos)
    let mut fp = 0.0;     // False Positives (Falsos Positivos)
    let mut fn_neg = 0.0; // False Negatives (Falsos Negativos)

    // Leer fila por fila el output del main
    for result in rdr.records() {
        let record = result?;
        let actual: f64 = record.get(0).ok_or("Falta columna actual")?.parse()?;
        let predicho: f64 = record.get(1).ok_or("Falta columna predicho")?.parse()?;

        if actual == 1.0 && predicho == 1.0 {
            tp += 1.0;
        } else if actual == 0.0 && predicho == 0.0 {
            tn += 1.0;
        } else if actual == 0.0 && predicho == 1.0 {
            fp += 1.0;
        } else if actual == 1.0 && predicho == 0.0 {
            fn_neg += 1.0;
        }
    }

    let total_muestras = tp + tn + fp + fn_neg;
    if total_muestras == 0.0 {
        return Err("El archivo de predicciones está vacío.".into());
    }

    // Cálculos Matemáticos de Diagnóstico
    let accuracy = (tp + tn) / total_muestras;
    let sensibilidad = tp / (tp + fn_neg + 1e-9); // 1e-9 evita división por cero
    let especificidad = tn / (tn + fp + 1e-9);
    let f1_score = (2.0 * tp) / (2.0 * tp + fp + fn_neg + 1e-9);

    // IMPRESIÓN DE RESULTADOS EN FORMATO ACADÉMICO
    println!("Muestras analizadas en el Test Set: {}\n", total_muestras);
    
    println!("=== 1. MATRIZ DE CONFUSIÓN ===");
    println!("                      PREDICHO (IA)");
    println!("                      Gana P1     Gana P2");
    println!("ACTUAL  Gana P1    |  {:>7}  |  {:>7}  | (Verdaderos Positivos / Falsos Negativos)", tp, fn_neg);
    println!("REAL    Gana P2    |  {:>7}  |  {:>7}  | (Falsos Positivos / Verdaderos Negativos)", fp, tn);
    println!("----------------------------------------------------");

    println!("\n=== 2. MÉTRICAS DE RENDIMIENTO ===");
    println!(" -> Exactitud (Accuracy):     {:.2}%", accuracy * 100.0);
    println!(" -> Sensibilidad (Recall):    {:.2}% (Capacidad de predecir victorias de P1)", sensibilidad * 100.0);
    println!(" -> Especificidad:            {:.2}% (Capacidad de predecir victorias de P2)", especificidad * 100.0);
    println!(" -> F1-Score:                 {:.4}  (Balance de precisión y sensibilidad)", f1_score);
    println!("====================================================");

    Ok(())
}
