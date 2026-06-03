use csv::Reader;
use std::collections::HashMap;
use std::error::Error;

fn main() -> std::result::Result<(), Box<dyn Error>> {
    println!("Iniciando lector de datos...");

    let mut rdr = Reader::from_path("data.csv")?;

    let mut vector_ganadores: Vec<String> = Vec::new();

    for result in rdr.records() {
        let record = result?;

        if let Some(ganador) = record.get(8) {
            vector_ganadores.push(ganador.to_string());
        }
    }

    println!(
        "Se cargaron {} registros de la columna objetivo.",
        vector_ganadores.len()
    );

    let entropia_total = calcular_entropia(&vector_ganadores);
    println!("La entropía total de los datos es: {:.4}", entropia_total);

    Ok(())
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

fn calcular_ganancia() -> &str {
    return "hola";
}
