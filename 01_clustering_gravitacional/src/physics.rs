// src/physics.rs

use crate::particle::{self, Particle};
use crate::union_find::UnionFind;
use rand::Rng;

pub fn euclidian_distance(p1: &Particle, p2: &Particle) -> f64 {
    let mut sum = 0.0;
    for i in 0..p1.dimensions() {
        let diff = p1.position[i] - p2.position[i];
        sum += diff * diff;
    }
    sum.sqrt()
}

pub fn run_gravitacional_clustering(
    particle: &mut Vec<Particle>,
    mut g: f64,
    delta_g: f64,
    iterations: usize,
    epsilon: f64,
) -> UnionFind {
    let n = particle.len();
    let mut uf = UnionFind::new(n);
    let mut rng = rand::thread_rng();

    println!(
        "Iniciando simulación con {} iteraciones y {} partículas...",
        iterations, n
    );

    for iter in 0..iterations {
        for j in 0..n {
            let mut k = rng.gen_range(0..n);
            while k == j {
                k = rng.gen_range(0..n);
            }

            let dist = euclidian_distance(&particle[j], &particle[k]);
            if dist * dist <= epsilon {
                uf.union(j, k);
            }

            if dist > 1e-7 {
                let dist_cubed = dist * dist * dist;
                let factor = g / dist_cubed;
                for dim in 0..particle[j].dimensions() {
                    let d_vector = particle[k].position[dim] - particle[j].position[dim];
                    let movement = d_vector * factor;
                    particle[j].position[dim] += movement;
                    particle[k].position[dim] -= movement;
                }
            }
        }
        g = g * (1.0 - delta_g);

        if iter % 50 == 0 {
            println!("Iteración {}/{} completada", iter, iterations);
        }
    }
    for i in 0..n {
        uf.find(i);
    }

    uf
}
