pub struct RegresionLogistica {
    pub pesos: Vec<f64>, 
    pub sesgo: f64,      
}

impl RegresionLogistica {
    pub fn new(num_caracteristicas: usize) -> Self {
        RegresionLogistica {
            pesos: vec![0.0; num_caracteristicas],
            sesgo: 0.0,
        }
    }

    fn sigmoide(&self, z: f64) -> f64 {
        let z_clamped = z.clamp(-20.0, 20.0);
        1.0 / (1.0 + (-z_clamped).exp())
    }

    pub fn predecir_probabilidad(&self, x: &[f64]) -> f64 {
        let mut z = self.sesgo;
        for i in 0..self.pesos.len() {
            z += self.pesos[i] * x[i];
        }
        self.sigmoide(z)
    }

    pub fn predecir(&self, x: &[f64]) -> f64 {
        if self.predecir_probabilidad(x) >= 0.5 {
            1.0
        } else {
            0.0
        }
    }

    pub fn entrenar(
        &mut self,
        x: &[Vec<f64>],
        y: &[f64],
        tasa_aprendizaje: f64,
        epocas: usize,
    ) {
        let m = x.len() as f64; 
        let n = self.pesos.len(); 

        println!("Iniciando entrenamiento del modelo matemático...");
        
        for epoca in 1..=epocas {
            let mut dw = vec![0.0; n];
            let mut db = 0.0;
            let mut costo_total = 0.0;

            //FORWARD PROPAGATION Y CÁLCULO DE GRADIENTES
            for i in 0..x.len() {
                let prediccion = self.predecir_probabilidad(&x[i]);
                let error = prediccion - y[i];
                db += error;
                for j in 0..n {
                    dw[j] += error * x[i][j];
                }
                let pred_segura = prediccion.clamp(1e-15, 1.0 - 1e-15);
                let loss = - (y[i] * pred_segura.ln() + (1.0 - y[i]) * (1.0 - pred_segura).ln());
                costo_total += loss;
            }

            let costo_promedio = costo_total / m;
            db /= m;
            for j in 0..n {
                dw[j] /= m;
            }

            // BACKWARD PROPAGATION 
            self.sesgo -= tasa_aprendizaje * db;
            for j in 0..n {
                self.pesos[j] -= tasa_aprendizaje * dw[j];
            }

            if epoca == 1 || epoca % 100 == 0 || epoca == epocas {
                println!(" -> Época {:>4}/{} | Log-Loss (Error): {:.6}", epoca, epocas, costo_promedio);
            }
        }
        println!("¡Entrenamiento finalizado con éxito!");
    }
}
