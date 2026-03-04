//! Benchmarks for sumcheck with split-eq delayed modular reduction.
//!
//! Run with: cargo bench --bench sumcheck
//! Filter:   cargo bench --bench sumcheck -- "/20"
//! Sizes:    BENCH_SIZES=24,26 cargo bench --bench sumcheck
//! Field:    BENCH_FIELD=pallas cargo bench --bench sumcheck
//!           Options: bn254 (default), pallas, vesta, t256
//! Mode:     BENCH_MODE=standard cargo bench --bench sumcheck
//!           Options: both (default), standard, small
//! L0:       BENCH_L0=3,6 cargo bench --bench sumcheck
//!           Options: 3, 6, 9, 12 (default: all). Comma-separated.
//!
//! Filter:   cargo bench --bench sumcheck -- "small/3"
//!           Use criterion's filter to select specific benchmarks

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use ff::Field;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use spartan2::{
  polys::multilinear::MultilinearPolynomial,
  provider::{Bn254Engine, PallasHyraxEngine, T256HyraxEngine, VestaHyraxEngine},
  small_field::{DelayedReduction, SmallValueField},
  small_sumcheck::prove_cubic_small_value,
  sumcheck::SumcheckProof,
  traits::{Engine, transcript::TranscriptEngineTrait},
};
use std::time::Duration;

criterion_group! {
    name = sumcheck;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(3))
        .measurement_time(Duration::from_secs(20))
        .sample_size(10);
    targets = bench_sumcheck
}

criterion_main!(sumcheck);

fn bench_standard_sumcheck<E: Engine>(c: &mut Criterion, field_name: &str, sizes: &[usize]) {
  let max_vars = *sizes.iter().max().unwrap_or(&26);
  let max_len = 1 << max_vars;

  // Pre-generate test data at maximum size
  let az: Vec<E::Scalar> = (0..max_len)
    .into_par_iter()
    .map(|i| E::Scalar::from(i as u64 * 3 + 1))
    .collect();

  let bz: Vec<E::Scalar> = (0..max_len)
    .into_par_iter()
    .map(|i| E::Scalar::from(i as u64 * 7 + 2))
    .collect();

  let cz: Vec<E::Scalar> = (0..max_len)
    .into_par_iter()
    .map(|i| E::Scalar::from(i as u64 * 11 + 3))
    .collect();

  let taus: Vec<E::Scalar> = (0..max_vars)
    .map(|i| E::Scalar::from(i as u64 + 1))
    .collect();

  let group_name = format!("Sumcheck/{field_name}");
  let mut group = c.benchmark_group(&group_name);

  for &num_vars in sizes {
    let len = 1 << num_vars;

    group.throughput(Throughput::Elements(len as u64));

    group.bench_with_input(
      BenchmarkId::from_parameter(num_vars),
      &num_vars,
      |b, &num_vars| {
        b.iter_batched(
          || {
            (
              MultilinearPolynomial::new(az[..len].to_vec()),
              MultilinearPolynomial::new(bz[..len].to_vec()),
              MultilinearPolynomial::new(cz[..len].to_vec()),
              taus[..num_vars].to_vec(),
              E::TE::new(b"bench"),
            )
          },
          |(mut poly_az, mut poly_bz, mut poly_cz, tau_vec, mut transcript)| {
            SumcheckProof::<E>::prove_cubic_with_three_inputs(
              &E::Scalar::ZERO,
              tau_vec,
              &mut poly_az,
              &mut poly_bz,
              &mut poly_cz,
              &mut transcript,
            )
            .unwrap()
          },
          BatchSize::LargeInput,
        );
      },
    );
  }

  group.finish();
}

/// Helper macro to generate bench function for a specific LB const generic
macro_rules! bench_small_value_with_lb {
  ($func_name:ident, $lb:expr) => {
    fn $func_name<E: Engine>(
      c: &mut Criterion,
      field_name: &str,
      sizes: &[usize],
      az: &[i64],
      bz: &[i64],
      cz: &[i64],
      taus: &[E::Scalar],
    ) where
      E::Scalar: SmallValueField<i64>
        + DelayedReduction<i64>
        + DelayedReduction<i128>
        + DelayedReduction<E::Scalar>,
    {
      let group_name = format!("Sumcheck/{}/small/{}", field_name, $lb);
      let mut group = c.benchmark_group(&group_name);

      for &num_vars in sizes {
        let len = 1 << num_vars;

        group.throughput(Throughput::Elements(len as u64));

        group.bench_with_input(
          BenchmarkId::from_parameter(num_vars),
          &num_vars,
          |b, &num_vars| {
            let az = az[..len].to_vec();
            let bz = bz[..len].to_vec();
            let cz = cz[..len].to_vec();
            let taus = taus[..num_vars].to_vec();
            b.iter_batched(
              || {
                (
                  MultilinearPolynomial::new(az.clone()),
                  MultilinearPolynomial::new(bz.clone()),
                  MultilinearPolynomial::new(cz.clone()),
                  taus.clone(),
                  E::TE::new(b"bench"),
                )
              },
              |(poly_az, poly_bz, poly_cz, tau_vec, mut transcript)| {
                prove_cubic_small_value::<E, i64, $lb>(
                  &E::Scalar::ZERO,
                  tau_vec,
                  &poly_az,
                  &poly_bz,
                  &poly_cz,
                  &mut transcript,
                )
                .unwrap()
              },
              BatchSize::LargeInput,
            );
          },
        );
      }

      group.finish();
    }
  };
}

// Generate bench functions for different L0 values
bench_small_value_with_lb!(bench_small_l0_3, 3);
bench_small_value_with_lb!(bench_small_l0_6, 6);
bench_small_value_with_lb!(bench_small_l0_9, 9);
bench_small_value_with_lb!(bench_small_l0_12, 12);

fn bench_small_value_sumcheck<E: Engine>(
  c: &mut Criterion,
  field_name: &str,
  sizes: &[usize],
  l0_values: &[usize],
) where
  E::Scalar: SmallValueField<i64>
    + DelayedReduction<i64>
    + DelayedReduction<i128>
    + DelayedReduction<E::Scalar>,
{
  let max_vars = *sizes.iter().max().unwrap_or(&26);
  let max_len = 1 << max_vars;

  // Pre-generate small-value test data (i64 range for products)
  let az: Vec<i64> = (0..max_len)
    .into_par_iter()
    .map(|i| (i % 100 + 1) as i64)
    .collect();

  let bz: Vec<i64> = (0..max_len)
    .into_par_iter()
    .map(|i| ((i * 7) % 100 + 1) as i64)
    .collect();

  // Cz = Az * Bz for satisfying R1CS
  let cz: Vec<i64> = az.iter().zip(&bz).map(|(&a, &b)| a * b).collect();

  let taus: Vec<E::Scalar> = (0..max_vars)
    .map(|i| E::Scalar::from(i as u64 + 1))
    .collect();

  // Run benchmarks for each l0 value
  for &l0 in l0_values {
    match l0 {
      3 => bench_small_l0_3::<E>(c, field_name, sizes, &az, &bz, &cz, &taus),
      6 => bench_small_l0_6::<E>(c, field_name, sizes, &az, &bz, &cz, &taus),
      9 => bench_small_l0_9::<E>(c, field_name, sizes, &az, &bz, &cz, &taus),
      12 => bench_small_l0_12::<E>(c, field_name, sizes, &az, &bz, &cz, &taus),
      _ => eprintln!("Unsupported l0 value '{}'. Supported: 3, 6, 9, 12", l0),
    }
  }
}

fn bench_sumcheck(c: &mut Criterion) {
  let field = std::env::var("BENCH_FIELD").unwrap_or_else(|_| "bn254".to_string());
  let mode = std::env::var("BENCH_MODE").unwrap_or_else(|_| "both".to_string());

  // Read sizes from env, default to 16..=26
  let sizes: Vec<usize> = std::env::var("BENCH_SIZES")
    .map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect())
    .unwrap_or_else(|_| (16..=26).collect());

  // Read l0 values for small-value mode (default: all supported values)
  let l0_values: Vec<usize> = std::env::var("BENCH_L0")
    .map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect())
    .unwrap_or_else(|_| vec![3, 6, 9, 12]);

  let run_standard = matches!(mode.to_lowercase().as_str(), "standard" | "both");
  let run_small = matches!(mode.to_lowercase().as_str(), "small" | "both");

  match field.to_lowercase().as_str() {
    "bn254" => {
      if run_standard {
        bench_standard_sumcheck::<Bn254Engine>(c, "bn254", &sizes);
      }
      if run_small {
        bench_small_value_sumcheck::<Bn254Engine>(c, "bn254", &sizes, &l0_values);
      }
    }
    "pallas" => {
      if run_standard {
        bench_standard_sumcheck::<PallasHyraxEngine>(c, "pallas", &sizes);
      }
      if run_small {
        bench_small_value_sumcheck::<PallasHyraxEngine>(c, "pallas", &sizes, &l0_values);
      }
    }
    "vesta" => {
      if run_standard {
        bench_standard_sumcheck::<VestaHyraxEngine>(c, "vesta", &sizes);
      }
      if run_small {
        bench_small_value_sumcheck::<VestaHyraxEngine>(c, "vesta", &sizes, &l0_values);
      }
    }
    "t256" => {
      if run_standard {
        bench_standard_sumcheck::<T256HyraxEngine>(c, "t256", &sizes);
      }
      if run_small {
        bench_small_value_sumcheck::<T256HyraxEngine>(c, "t256", &sizes, &l0_values);
      }
    }
    _ => {
      eprintln!("Unknown field '{}'. Using bn254.", field);
      if run_standard {
        bench_standard_sumcheck::<Bn254Engine>(c, "bn254", &sizes);
      }
      if run_small {
        bench_small_value_sumcheck::<Bn254Engine>(c, "bn254", &sizes, &l0_values);
      }
    }
  }
}
