/// Estructura que representa el modelo de Regresión Logística
pub struct RegresionLogistica {
    pub pesos: Vec<f64>, // Vector de pesos (w), uno para cada característica (APM, aldeanos, etc.)
    pub sesgo: f64,      // Sesgo o intersección (b)
}

impl RegresionLogistica {
    /// Inicializa un nuevo modelo con pesos en cero basándose en el número de características
    pub fn new(num_caracteristicas: usize) -> Self {
        RegresionLogistica {
            pesos: vec![0.0; num_caracteristicas],
            sesgo: 0.0,
        }
    }

    /// Función de activación Sigmoide estable para evitar desbordamiento numérico
    fn sigmoide(&self, z: f64) -> f64 {
        // Clampeamos z para evitar valores extremos que generen NaN con el método .exp()
        let z_clamped = z.clamp(-20.0, 20.0);
        1.0 / (1.0 + (-z_clamped).exp())
    }

    /// Calcula la probabilidad de victoria para una única partida (X)
    pub fn predecir_probabilidad(&self, x: &[f64]) -> f64 {
        let mut z = self.sesgo;
        // Producto punto: w * x
        for i in 0..self.pesos.len() {
            z += self.pesos[i] * x[i];
        }
        self.sigmoide(z)
    }

    /// Clasifica una partida de forma binaria: 1.0 (Gana P1) o 0.0 (Gana P2)
    pub fn predecir(&self, x: &[f64]) -> f64 {
        if self.predecir_probabilidad(x) >= 0.5 {
            1.0
        } else {
            0.0
        }
    }

    /// Entrena el modelo utilizando el algoritmo de Gradiente Descendente
    pub fn entrenar(
        &mut self,
        x: &[Vec<f64>],
        y: &[f64],
        tasa_aprendizaje: f64,
        epocas: usize,
    ) {
        let m = x.len() as f64; // Número de muestras (partidas) en el set de entrenamiento
        let n = self.pesos.len(); // Número de características

        println!("Iniciando entrenamiento del modelo matemático...");
        
        for epoca in 1..=epocas {
            // Vectores para acumular los gradientes de esta época
            let mut dw = vec![0.0; n];
            let mut db = 0.0;
            let mut costo_total = 0.0;

            // 1. FORWARD PROPAGATION Y CÁLCULO DE GRADIENTES
            for i in 0..x.len() {
                let prediccion = self.predecir_probabilidad(&x[i]);
                let error = prediccion - y[i];

                // Acumular gradiente del sesgo (db)
                db += error;

                // Acumular gradiente de cada peso (dw)
                for j in 0..n {
                    dw[j] += error * x[i][j];
                }

                // Calcular la función de pérdida (Binary Cross-Entropy Loss)
                // Clampeamos la predicción para evitar log(0) que daría infinito
                let pred_segura = prediccion.clamp(1e-15, 1.0 - 1e-15);
                let loss = - (y[i] * pred_segura.ln() + (1.0 - y[i]) * (1.0 - pred_segura).ln());
                costo_total += loss;
            }

            // Promediar los gradientes y el costo dividiendo por la cantidad de muestras (m)
            let costo_promedio = costo_total / m;
            db /= m;
            for j in 0..n {
                dw[j] /= m;
            }

            // 2. BACKWARD PROPAGATION (Actualización de parámetros)
            self.sesgo -= tasa_aprendizaje * db;
            for j in 0..n {
                self.pesos[j] -= tasa_aprendizaje * dw[j];
            }

            // Mostrar el progreso del error cada 100 épocas
            if epoca == 1 || epoca % 100 == 0 || epoca == epocas {
                println!(" -> Época {:>4}/{} | Log-Loss (Error): {:.6}", epoca, epocas, costo_promedio);
            }
        }
        println!("¡Entrenamiento finalizado con éxito!");
    }
}
