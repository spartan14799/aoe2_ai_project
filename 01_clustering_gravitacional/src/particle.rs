// src/particle.rs

#[derive(Debug, Clone)]
pub struct Particle {
    pub id: usize,
    pub position: Vec<f64>,
    pub velocity: Vec<f64>,
    pub mass: f64,
    pub is_active: bool,
}

impl Particle{
    pub fn new(id: usize, position: Vec<f64>) -> Self{
        let dimensions = position.len();
        Self {
            id,
            position,
            velocity: vec![0.0; dimensions],
            mass: 1.0,
            is_active: true,
        }
    }
    pub fn dimensions(&self) -> usize{
        self.position.len()
    }
}
