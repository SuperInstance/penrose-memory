//! TensorTile benchmarks — fill, threshold, norm, and constraint check.
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use penrose_memory::tensor_tile::{TensorTile, TensorTiling};
use penrose_memory::cut_and_project::TileType;

/// Generate N tensor tiles with random-ish source coords.
fn make_tiles(n: usize) -> Vec<TensorTile> {
    (0..n)
        .map(|i| {
            let coords: [i32; 5] = [
                (i as i32).wrapping_mul(7) % 100,
                (i as i32).wrapping_mul(13) % 100,
                (i as i32).wrapping_mul(17) % 100,
                (i as i32).wrapping_mul(23) % 100,
                (i as i32).wrapping_mul(29) % 100,
            ];
            let tile_type = if i % 2 == 0 {
                TileType::Thick
            } else {
                TileType::Thin
            };
            TensorTile::new(
                coords,
                tile_type,
                0.0,
                [(i as f64 * 1.618).sin() * 50.0, (i as f64 * 2.618).cos() * 50.0],
            )
        })
        .collect()
}

fn tensor_fill_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_fill");
    for size in [100, 500, 1000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &n| {
            let mut tiles = make_tiles(n);
            b.iter(|| {
                for tile in &mut tiles {
                    tile.fill_from_source();
                }
                black_box(&tiles);
            });
        });
    }
    group.finish();
}

fn tensor_threshold_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_threshold");
    for size in [100, 500, 1000] {
        let mut tiles = make_tiles(size);
        for tile in &mut tiles {
            tile.fill_from_source();
        }
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &n| {
            b.iter(|| {
                for tile in &mut tiles {
                    tile.apply_threshold(black_box(0.5));
                }
                black_box(&tiles);
            });
        });
    }
    group.finish();
}

fn tensor_norm_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_norm");
    for size in [100, 500, 1000] {
        let mut tiles = make_tiles(size);
        for tile in &mut tiles {
            tile.fill_from_source();
        }
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            let mut total = 0.0f32;
            b.iter(|| {
                for tile in &tiles {
                    total += tile.l1_norm();
                }
                black_box(total)
            });
        });
    }
    group.finish();
}

fn tensortiling_constraint_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensortiling_constraint");
    for size in [100, 500, 1000] {
        // Place tiles close together so adjacency is detected.
        let tiles: Vec<TensorTile> = (0..size)
            .map(|i| {
                let coords: [i32; 5] = [
                    (i as i32).wrapping_mul(7) % 100,
                    (i as i32).wrapping_mul(13) % 100,
                    (i as i32).wrapping_mul(17) % 100,
                    (i as i32).wrapping_mul(23) % 100,
                    (i as i32).wrapping_mul(29) % 100,
                ];
                let tile_type = if i % 2 == 0 {
                    TileType::Thick
                } else {
                    TileType::Thin
                };
                let mut tt = TensorTile::new(
                    coords,
                    tile_type,
                    0.0,
                    [
                        (i as f64 % 20.0) + (i as f64 * 0.3).sin(),
                        ((i as f64 / 20.0).floor()) * 1.5,
                    ],
                );
                tt.fill_from_source();
                tt
            })
            .collect();

        let tiling = TensorTiling::new(tiles);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                black_box(tiling.constraint_check())
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    tensor_fill_bench,
    tensor_threshold_bench,
    tensor_norm_bench,
    tensortiling_constraint_bench,
);
criterion_main!(benches);
