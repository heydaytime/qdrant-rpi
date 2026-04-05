use std::hint::black_box;

use common::bitvec::BitVec;
use criterion::{Criterion, criterion_group, criterion_main};
use gridstore::bitmask::Bitmask;
use gridstore::config::DEFAULT_REGION_SIZE_BLOCKS;
use rand::RngExt;

pub fn bench_bitmask_ops(c: &mut Criterion) {
    let distr = rand::distr::StandardUniform;
    let rng = rand::rng();
    let random_bitvec = rng
        .sample_iter::<bool, _>(distr)
        .take(1000 * DEFAULT_REGION_SIZE_BLOCKS)
        .collect::<BitVec>();

    let bit_slices = random_bitvec
        .chunks_exact(DEFAULT_REGION_SIZE_BLOCKS)
        .collect::<Vec<_>>();
    let mut bitslice_idx = 0usize;

    c.bench_function("calculate_gaps", |b| {
        b.iter(|| {
            let bitslice = bit_slices[bitslice_idx % bit_slices.len()];
            bitslice_idx = bitslice_idx.wrapping_add(1);
            Bitmask::calculate_gaps(black_box(bitslice), DEFAULT_REGION_SIZE_BLOCKS)
        })
    });

    c.bench_function("find_available_blocks_in_slice", |b| {
        let mut rng = rand::rng();
        b.iter(|| {
            let bitslice = bit_slices[bitslice_idx % bit_slices.len()];
            bitslice_idx = bitslice_idx.wrapping_add(1);
            let num_blocks = rng.random_range(1..10);
            Bitmask::find_available_blocks_in_slice(black_box(bitslice), num_blocks, |_| (0, 0))
        })
    });
}

criterion_group!(benches, bench_bitmask_ops);
criterion_main!(benches);
