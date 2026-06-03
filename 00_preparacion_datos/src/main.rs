mod data_cleaner;

fn main() {
    if let Err(e) = data_cleaner::ejecutar_limpieza() {
        eprintln!("Ocurrió un error crítico procesando los datos: {}", e);
    }
}
