//! SIMD comparison: scalar vs auto-vectorized float ops on tile tensors.
//!
//! The paper claims 16× throughput on AVX-512. This bench measures
//! what LLVM actually delivers for simple loops that *should* auto-vectorize.
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

/// Flat tensor data for 10,000 tiles (Thick = 25 f32s each, Thin = 24 f32s each).
/// We use a single contiguous buffer of f32 to maximize auto-vectorization.
const N_TILES: usize = 10_000;
const ELEMENTS_PER_TILE_THICK: usize = 25; // 5×5
const ELEMENTS_PER_TILE_THIN: usize = 24;  // 3×8

fn generate_flat_tensors(n: usize, elements: usize) -> Vec<f32> {
    let mut data = Vec::with_capacity(n * elements);
    for tile_idx in 0..n {
        for i in 0..elements {
            let i_f = i as f32;
            let m = elements as f32;
            // Same formula as fill_from_source's mode_a + mode_b + mode_c modes.
            let val = (tile_idx as f32 * 0.1)
                + (i_f / m)
                + (2.0 * std::f32::consts::PI * (tile_idx as f32 % 100.0) * i_f / m).sin()
                + (((tile_idx.wrapping_mul(7) ^ i) as u32) as f32 / u32::MAX as f32);
            data.push(val);
        }
    }
    data
}

// ─── Scalar threshold: naive branch per element ──────────────────────

#[inline(never)]
fn scalar_threshold(data: &mut [f32], threshold: f32) {
    for v in data.iter_mut() {
        if *v < threshold {
            *v = 0.0;
        }
    }
}

// ─── Auto-vectorized threshold: hint-friendly loop ──────────────────
// LLVM should turn this into packed compares on AVX-512 (16 floats/op)
// or AVX2 (8 floats/op).

#[inline(always)]
fn auto_vec_threshold(data: &mut [f32], threshold: f32) {
    let len = data.len();
    // Ensure the loop bound is known to the optimizer.
    for i in 0..len {
        // Branchless: mask + blend — LLVM prefers this for auto-vec.
        let v = data[i];
        data[i] = if v < threshold { 0.0 } else { v };
    }
}

// ─── Scalar L1 norm ─────────────────────────────────────────────────

#[inline(never)]
fn scalar_l1_norm(data: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for &v in data {
        sum += v.abs();
    }
    sum
}

// ─── Auto-vectorized L1 norm ────────────────────────────────────────

#[inline(always)]
fn auto_vec_l1_norm(data: &[f32]) -> f32 {
    data.iter().map(|v| v.abs()).sum()
}

// ─── Scalar fill (mimics fill_from_source inner loop) ───────────────

#[inline(never)]
fn scalar_fill(data: &mut [f32], tile_count: usize, rows: usize, cols: usize) {
    let elements = rows * cols;
    for t in 0..tile_count {
        let base = t * elements;
        let a = ((t as i32).wrapping_mul(7) % 100).unsigned_abs() as f32 / 100.0;
        for i in 0..rows {
            for j in 0..cols {
                let idx = base + i * cols + j;
                let val = a + (i as f32 / rows as f32)
                    + (2.0 * std::f32::consts::PI * j as f32 / cols as f32).sin();
                data[idx] = val;
            }
        }
    }
}

// ─── Auto-vectorized fill ───────────────────────────────────────────

#[inline(always)]
fn auto_vec_fill(data: &mut [f32], tile_count: usize, rows: usize, cols: usize) {
    let elements = rows * cols;
    for t in 0..tile_count {
        let base = t * elements;
        let a = ((t as i32).wrapping_mul(7) % 100).unsigned_abs() as f32 / 100.0;
        for i in 0..rows {
            for j in 0..cols {
                let idx = base + i * cols + j;
                let val = a + (i as f32 / rows as f32)
                    + (2.0 * std::f32::consts::PI * j as f32 / cols as f32).sin();
                data[idx] = val;
            }
        }
    }
}

fn bench_threshold_scalar_vs_auto(c: &mut Criterion) {
    let mut group = c.benchmark_group("threshold_scalar_vs_auto");

    let total = N_TILES * ELEMENTS_PER_TILE_THICK;

    // Scalar
    let mut data_s = generate_flat_tensors(N_TILES, ELEMENTS_PER_TILE_THICK);
    group.bench_function("scalar", |b| {
        b.iter(|| {
            scalar_threshold(&mut data_s, black_box(0.5));
            black_box(&data_s);
        });
    });

    // Auto-vectorized
    let mut data_v = generate_flat_tensors(N_TILES, ELEMENTS_PER_TILE_THICK);
    group.bench_function("auto_vec", |b| {
        b.iter(|| {
            auto_vec_threshold(&mut data_v, black_box(0.5));
            black_box(&data_v);
        });
    });

    group.finish();
}

fn bench_l1_norm_scalar_vs_auto(c: &mut Criterion) {
    let mut group = c.benchmark_group("l1_norm_scalar_vs_auto");

    let total = N_TILES * ELEMENTS_PER_TILE_THICK;
    let data = generate_flat_tensors(N_TILES, ELEMENTS_PER_TILE_THICK);

    group.bench_function("scalar", |b| {
        b.iter(|| {
            black_box(scalar_l1_norm(&data))
        });
    });

    group.bench_function("auto_vec", |b| {
        b.iter(|| {
            black_box(auto_vec_l1_norm(&data))
        });
    });

    group.finish();
}

fn bench_fill_scalar_vs_auto(c: &mut Criterion) {
    let mut group = c.benchmark_group("fill_scalar_vs_auto");

    let elements = ELEMENTS_PER_TILE_THICK;
    let rows = 5;
    let cols = 5;

    let mut data_s = vec![0.0f32; N_TILES * elements];
    group.bench_function("scalar", |b| {
        b.iter(|| {
            scalar_fill(&mut data_s, N_TILES, rows, cols);
            black_box(&data_s);
        });
    });

    let mut data_v = vec![0.0f32; N_TILES * elements];
    group.bench_function("auto_vec", |b| {
        b.iter(|| {
            auto_vec_fill(&mut data_v, N_TILES, rows, cols);
            black_box(&data_v);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_threshold_scalar_vs_auto,
    bench_l1_norm_scalar_vs_auto,
    bench_fill_scalar_vs_auto,
);
criterion_main!(benches);
