use std::error::Error;
use csv::ReaderBuilder;
use plotters::prelude::*;

fn main() -> Result<(), Box<dyn Error>> {
    println!("====================================================");
    println!("=== MÓDULO DE MÉTRICAS AVANZADAS Y GENERACIÓN GRÁFICA ===");
    println!("====================================================\n");

    let ruta_predicciones = "../data/test_predictions.csv";
    let ruta_grafico = "../data/reporte_metricas_aoe2.png";
    
    // 1. LECTURA DE DATOS
    let mut rdr = match ReaderBuilder::new().from_path(ruta_predicciones) {
        Ok(reader) => reader,
        Err(_) => {
            eprintln!("Error: No se encontró el archivo '{}'.", ruta_predicciones);
            eprintln!("Por favor, ejecuta primero 'cargo run' para entrenar el modelo.");
            return Ok(());
        }
    };

    let mut tp = 0.0;     // True Positives
    let mut tn = 0.0;     // True Negatives
    let mut fp = 0.0;     // False Positives
    let mut fn_neg = 0.0; // False Negatives

    for result in rdr.records() {
        let record = result?;
        let actual: f64 = record.get(0).ok_or("Falta columna actual")?.parse()?;
        let predicho: f64 = record.get(1).ok_or("Falta columna predicho")?.parse()?;

        if actual == 1.0 && predicho == 1.0 { tp += 1.0; }
        else if actual == 0.0 && predicho == 0.0 { tn += 1.0; }
        else if actual == 0.0 && predicho == 1.0 { fp += 1.0; }
        else if actual == 1.0 && predicho == 0.0 { fn_neg += 1.0; }
    }

    let total_muestras = tp + tn + fp + fn_neg;
    if total_muestras == 0.0 {
        return Err("El archivo de predicciones está vacío.".into());
    }

    // 2. CÁLCULO DE MÉTRICAS ESENCIALES
    let accuracy = (tp + tn) / total_muestras;
    let sensibilidad = tp / (tp + fn_neg + 1e-9);
    let especificidad = tn / (tn + fp + 1e-9);
    let f1_score = (2.0 * tp) / (2.0 * tp + fp + fn_neg + 1e-9);

    let acc_pct = accuracy * 100.0;
    let sens_pct = sensibilidad * 100.0;
    let esp_pct = especificidad * 100.0;

    // ====================================================
    // 3. GENERACIÓN DEL GRÁFICO PROFESIONAL (PNG)
    // ====================================================
    println!("[i] Generando gráfico de barras estadístico...");
    
    let root = BitMapBackend::new(ruta_grafico, (800, 500)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Métricas Globales de Clasificación (Regresión Logística - Min 20)", ("sans-serif", 22).into_font())
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0..40, 0.0..105.0)?;

    // Configuración de la cuadrícula y las etiquetas personalizadas
    chart.configure_mesh()
        .y_desc("Porcentaje (%)")
        .x_desc("Métricas Evaluadas")
        .axis_desc_style(("sans-serif", 14))
        // CORRECCIÓN CRÍTICA: Cambiado de 0 a 9 para habilitar marcas exactas cada 5 unidades (0, 5, 10, 15...)
        .x_labels(9) 
        .x_label_formatter(&|x| {
            match *x {
                5 => "Exactitud (Acc)".to_string(),
                15 => "Sensibilidad (Recall)".to_string(),
                25 => "Especificidad".to_string(),
                35 => "F1-Score (Escala x100)".to_string(),
                _ => "".to_string(), // Las marcas intermedias (0, 10, 20...) no muestran texto
            }
        })
        .draw()?;

    // Datos y colores de las barras (SteelBlue, SeaGreen, DarkGoldenrod, Crimson)
    let valores = [acc_pct, sens_pct, esp_pct, f1_score * 100.0];
    let colores = [
        RGBColor(70, 130, 180),  
        RGBColor(46, 139, 87),  
        RGBColor(218, 165, 32), 
        RGBColor(178, 34, 34)
    ];

    // Dibujar las barras utilizando un espacio coordinado limpio
    for i in 0..4 {
        let inicio_x = (i * 10 + 2) as i32;
        let fin_x = (i * 10 + 8) as i32;
        
        chart.draw_series(std::iter::once(Rectangle::new(
            [(inicio_x, 0.0), (fin_x, valores[i])],
            colores[i].filled(),
        )))?;

        // Añadir el porcentaje flotando arriba de cada barra
        chart.draw_series(std::iter::once(Text::new(
            format!("{:.2}%", valores[i]),
            (inicio_x + 1, valores[i] + 2.0),
            ("sans-serif", 13).into_font(),
        )))?;
    }

    root.present()?;
    println!("[✓] ¡Gráfico exportado con éxito a: {}!\n", ruta_grafico);

    // ====================================================
    // 4. REPORTES ACADÉMICOS EN TEXTO (LISTOS PARA COPIAR)
    // ====================================================
    println!("====================================================");
    println!("=== RECURSOS PARA TU TRABAJO ESCRITO (COPIAR) ===");
    println!("====================================================");

    // FORMATO 1: TABLA MARKDOWN
    println!("\n");
    println!("| Métrica | Valor Obtenido | Descripción en el Contexto de AoE2 |");
    println!("| :--- | :---: | :--- |");
    println!("| **Exactitud (Accuracy)** | {:.2}% | Porcentaje total de predicciones correctas del ganador. |", acc_pct);
    println!("| **Sensibilidad (Recall)** | {:.2}% | Capacidad de identificar correctamente las victorias del Jugador 1. |", sens_pct);
    println!("| **Especificidad** | {:.2}% | Capacidad de identificar correctamente las victorias del Jugador 2. |", esp_pct);
    println!("| **F1-Score** | {:.4} | Balance armónico entre precisión y exhaustividad del algoritmo. |", f1_score);

    // FORMATO 2: TABLA LATEX PROFESIONAL
    println!("\n%% TABLA EN FORMATO LATEX (ENTORNO TABLE)");
    println!("\\begin{{table}}[h!]");
    println!("\\centering");
    println!("\\caption{{Resultados del Modelo Matemático de Predicción al Minuto 20}}");
    println!("\\label{{tab:resultados_ml}}");
    println!("\\begin{{tabular}}{{lcr}}");
    println!("\\hline");
    println!("\\textbf{{Métrica}} & \\textbf{{Valor}} & \\textbf{{Muestras de Test}} \\\\");
    println!("\\hline");
    println!("Exactitud (Accuracy) & {:.2}\\% & {:.0} \\\\", acc_pct, total_muestras);
    println!("Sensibilidad (Recall) & {:.2}\\% & -- \\\\", sens_pct);
    println!("Especificidad & {:.2}\\% & -- \\\\", esp_pct);
    println!("F1-Score & {:.4} & -- \\\\", f1_score);
    println!("\\hline");
    println!("\\end{{tabular}}");
    println!("\\end{{table}}");

    // FORMATO 3: MATRIZ DE CONFUSIÓN ACADÉMICA
    println!("\n%% MATRIZ DE CONFUSIÓN PARA EL TEXTO INDEPENDIENTE");
    println!(" -> Verdaderos Positivos (P1 Gana y se predijo P1): {:.0}", tp);
    println!(" -> Verdaderos Negativos (P2 Gana y se predijo P2): {:.0}", tn);
    println!(" -> Falsos Positivos (P2 Gana pero se predijo P1):  {:.0}", fp);
    println!(" -> Falsos Negativos (P1 Gana pero se predijo P2):  {:.0}", fn_neg);
    println!("====================================================");

    Ok(())
}
