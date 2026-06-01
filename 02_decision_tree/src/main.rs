use csv::Reader;
use std::error::Error;

fn main() -> std::result::Result<(), Box<dyn Error>> {
    println!("Iniciando lector de datos...");

    let mut rdr = Reader::from_path("data.csv")?;

    let headers = rdr.headers()?;
    println!("Hay {} columnas.", headers.len());

    if let Some(result) = rdr.records().next() {
        let record = result?;
        println!("Primera partida (fila 1): {:?}", record);
    }

    println!("¡Lectura exitosa!");

    Ok(())
}
