//! Timing utilities for benchmarks and examples.
//!
//! This module provides:
//! - [`TimingLayer`]: A tracing layer that captures `elapsed_ms` and `constraints` fields
//! - Phase constants for Spartan and NeutronNova prove phases
//! - Helper functions for collecting and displaying timing data

use std::{
  collections::HashMap,
  sync::{Arc, Mutex},
};
use tracing::{
  Subscriber,
  field::{Field, Visit},
};
use tracing_subscriber::{Layer, layer::Context, registry::LookupSpan};

/// Shared timing data storage (phase name -> elapsed milliseconds).
pub type TimingData = Arc<Mutex<HashMap<String, u64>>>;

/// Shared constraint count storage.
pub type ConstraintsData = Arc<Mutex<Option<u64>>>;

/// A tracing layer that captures timing events with `elapsed_ms` fields.
pub struct TimingLayer {
  data: TimingData,
  constraints: ConstraintsData,
}

impl TimingLayer {
  /// Create a new timing layer and return handles to the collected data.
  pub fn new() -> (Self, TimingData, ConstraintsData) {
    let data: TimingData = Arc::new(Mutex::new(HashMap::new()));
    let constraints: ConstraintsData = Arc::new(Mutex::new(None));
    (
      Self {
        data: data.clone(),
        constraints: constraints.clone(),
      },
      data,
      constraints,
    )
  }
}

impl Default for TimingLayer {
  fn default() -> Self {
    Self::new().0
  }
}

struct ElapsedVisitor {
  elapsed_ms: Option<u64>,
  constraints: Option<u64>,
  message: Option<String>,
}

impl Visit for ElapsedVisitor {
  fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
    match field.name() {
      "elapsed_ms" => {
        let s = format!("{:?}", value);
        self.elapsed_ms = s.parse().ok();
      }
      "constraints" => {
        let s = format!("{:?}", value);
        self.constraints = s.parse().ok();
      }
      "message" => {
        self.message = Some(format!("{:?}", value));
      }
      _ => {}
    }
  }

  fn record_str(&mut self, field: &Field, value: &str) {
    if field.name() == "message" {
      self.message = Some(value.to_string());
    }
  }

  fn record_u64(&mut self, field: &Field, value: u64) {
    match field.name() {
      "elapsed_ms" => self.elapsed_ms = Some(value),
      "constraints" => self.constraints = Some(value),
      _ => {}
    }
  }
}

impl<S: Subscriber + for<'a> LookupSpan<'a>> Layer<S> for TimingLayer {
  fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
    let mut visitor = ElapsedVisitor {
      elapsed_ms: None,
      constraints: None,
      message: None,
    };
    event.record(&mut visitor);

    if let (Some(ms), Some(msg)) = (visitor.elapsed_ms, visitor.message)
      && let Ok(mut map) = self.data.lock()
    {
      *map.entry(msg).or_insert(0) += ms;
    }
    if let Some(c) = visitor.constraints
      && let Ok(mut lock) = self.constraints.lock()
    {
      *lock = Some(c);
    }
  }
}

/// Extract timing values for the given phases from collected data.
/// Returns a map keyed by short_name for easier access.
/// Special entries:
/// - `"__computed_total__"`: computed as the sum of all real span phases
/// - `"__prep__"`, `"__prove__"`: wall-clock times, looked up in the map but NOT summed into total
pub fn snapshot_timings(
  data: &TimingData,
  phases: &[(&str, &'static str)],
) -> HashMap<&'static str, u64> {
  let map = data.lock().unwrap();
  let mut result = HashMap::new();
  let mut sum: u64 = 0;
  for (phase, short_name) in phases {
    if *phase == "__computed_total__" {
      result.insert(*short_name, sum);
    } else if phase.starts_with("__") {
      // Wall-clock entries (e.g. __prep__, __prove__) — lookup but don't sum
      let v = map.get(*phase).copied().unwrap_or(0);
      result.insert(*short_name, v);
    } else {
      let v = map.get(*phase).copied().unwrap_or(0);
      sum += v;
      result.insert(*short_name, v);
    }
  }
  result
}

/// Clear all collected timing data.
pub fn clear_timings(data: &TimingData) {
  data.lock().unwrap().clear();
}

/// Normalize parallel span timings by dividing by the parallelism factor.
/// This converts summed CPU time to approximate wall-clock time.
/// `parallel_spans` is a list of short_names to normalize.
/// `divisor` should typically be `min(num_parallel_tasks, num_cores)`.
pub fn normalize_parallel_timings(
  timings: &mut HashMap<&'static str, u64>,
  parallel_spans: &[&'static str],
  divisor: usize,
) {
  if divisor == 0 {
    return;
  }
  for span in parallel_spans {
    if let Some(value) = timings.get_mut(span) {
      *value /= divisor as u64;
    }
  }
}

/// Compute element-wise minimum across multiple timing snapshots.
pub fn collect_timings(
  timings_all: &[HashMap<&'static str, u64>],
  phases: &[(&str, &'static str)],
) -> HashMap<&'static str, u64> {
  phases
    .iter()
    .map(|(_, short_name)| {
      let min_val = timings_all
        .iter()
        .map(|t| t.get(short_name).copied().unwrap_or(0))
        .min()
        .unwrap_or(0);
      (*short_name, min_val)
    })
    .collect()
}

// ============================================================================
// Spartan prove-phase constants
// ============================================================================

/// Spartan prove phases: (tracing_name, short_display_name).
/// `"__computed_total__"` is a computed sum of all preceding phases (not a span).
/// `"__prep__"` and `"__prove__"` are wall-clock times injected by the benchmark.
pub const SPARTAN_PHASES: &[(&str, &str)] = &[
  ("precommitted_witness_synthesize", "synth_pre"),
  ("commit_witness_precommitted", "commit_pre"),
  ("r1cs_instance_and_witness", "r1cs_rest"),
  ("commit_witness_rest", "commit_rest"),
  ("matrix_vector_multiply", "mat_vec"),
  ("outer_sumcheck", "outer_sc"),
  ("compute_eval_rx", "eval_rx"),
  ("compute_eval_table_sparse", "eval_sparse"),
  ("prepare_poly_ABC", "poly_ABC"),
  ("prepare_poly_z", "poly_z"),
  ("inner_sumcheck", "inner_sc"),
  ("pcs_prove", "pcs"),
  ("__prep__", "prep"),
  ("__prove__", "prove"),
  ("__computed_total__", "total"),
];

// ============================================================================
// NeutronNova NIFS prove-phase constants
// ============================================================================

/// NeutronNova NIFS prove phases: (tracing_name, short_display_name).
pub const NEUTRONNOVA_PHASES: &[(&str, &str)] = &[
  ("generate_shared_witness", "shared_syn"),
  ("precommitted_witness_synthesize", "precom_syn"),
  ("commit_witness_precommitted", "commit_pre"),
  ("matrix_vector_multiply_instances", "mat_vec"),
  ("nifs_folding_rounds", "nifs_fold_sc"),
  ("fold_witnesses", "fold_W"),
  ("fold_instances", "fold_U"),
  ("nifs_prove", "nifs_prove"),
  ("end_to_end_total", "end_to_end"),
];

// ============================================================================
// Spartan ZK prove-phase constants
// ============================================================================

/// Spartan ZK prove phases: (tracing_name, short_display_name).
pub const SPARTAN_ZK_PHASES: &[(&str, &str)] = &[
  // Prep phases
  ("shared_witness_synthesize", "shared_syn"),
  ("commit_witness_shared", "commit_shared"),
  ("precommitted_witness_synthesize", "precom_syn"),
  ("commit_witness_precommitted", "commit_pre"),
  // Prove phases
  ("rerandomize_prep_state", "rerand"),
  ("r1cs_instance_and_witness", "r1cs_inst"),
  ("circuit_synthesize_rest", "synth_rest"),
  ("commit_witness_rest", "commit_rest"),
  ("sample_taus", "taus"),
  ("matrix_vector_multiply", "mat_vec"),
  ("prepare_multilinear_polys", "prep_mle"),
  ("outer_sumcheck", "outer_sc"),
  ("compute_eval_rx", "eval_rx"),
  ("compute_eval_table_sparse", "eval_sparse"),
  ("prepare_poly_z", "poly_z"),
  ("inner_sumcheck", "inner_sc"),
  ("finalize_and_nifs", "nifs"),
  ("pcs_prove", "pcs"),
  ("spartan_zk_prove", "prove_total"),
];

/// NeutronNova full ZkSNARK prove phases: (tracing_name, short_display_name).
pub const NEUTRONNOVA_ZK_PROVE_PHASES: &[(&str, &str)] = &[
  // Prep phase breakdown (witness synthesis and commitment)
  ("neutronnova_prep_prove", "prep"),
  ("generate_shared_witness", "shared_syn"),
  ("precommitted_witness_synthesize", "precom_syn"),
  ("commit_witness_precommitted", "commit_pre"),
  // Rerandomize for ZK
  ("rerandomize_prep_state", "rerand"),
  // Instance generation
  ("generate_instances_witnesses", "gen_inst"),
  // NIFS breakdown
  ("nifs_folding_rounds", "nifs_sc"),
  ("fold_witnesses", "fold_W"),
  ("fold_instances", "fold_U"),
  ("nifs_prove", "nifs"),
  // Post-NIFS
  ("outer_sumcheck_batched", "outer_sc"),
  ("inner_sumcheck_batched", "inner_sc"),
  ("pcs_prove", "pcs"),
  ("neutronnova_prove", "zk_prove"),
  // Total end-to-end (prep + prove)
  ("end_to_end_total", "end_to_end"),
];

/// Print a comparison table of timing data.
pub fn print_table(
  header: &str,
  phases: &[(&str, &str)],
  small: &HashMap<&str, u64>,
  large: &HashMap<&str, u64>,
) {
  let col_w = 12;

  eprintln!("\n{}", header);

  eprint!("{:<14}", "");
  for (_, short_name) in phases {
    eprint!("{:>width$}", short_name, width = col_w);
  }
  eprintln!();

  eprint!("{:<14}", "small");
  for (_, short_name) in phases {
    let v = small.get(short_name).copied().unwrap_or(0);
    eprint!("{:>width$}", v, width = col_w);
  }
  eprintln!();

  eprint!("{:<14}", "large");
  for (_, short_name) in phases {
    let v = large.get(short_name).copied().unwrap_or(0);
    eprint!("{:>width$}", v, width = col_w);
  }
  eprintln!();

  eprint!("{:<14}", "speedup");
  for (_, short_name) in phases {
    let s = small.get(short_name).copied().unwrap_or(0);
    let l = large.get(short_name).copied().unwrap_or(0);
    if s == 0 {
      eprint!("{:>width$}", "-", width = col_w);
    } else {
      eprint!("{:>width$.2}x", l as f64 / s as f64, width = col_w - 1);
    }
  }
  eprintln!();
}
