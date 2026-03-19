// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! This module defines R1CS related types
use crate::{
  Blind, Commitment, CommitmentKey, DEFAULT_COMMITMENT_WIDTH, MULTIROUND_COMMITMENT_WIDTH, PCS,
  VerifierKey,
  digest::SimpleDigestible,
  errors::SpartanError,
  start_span,
  traits::{
    Engine,
    pcs::{FoldingEngineTrait, PCSEngineTrait},
    transcript::{TranscriptEngineTrait, TranscriptReprTrait},
  },
};
use crate::small_constraint_system::SmallCoeff;
use crate::small_field::montgomery::MontgomeryLimbs;
use core::cmp::max;
use ff::Field;
use once_cell::sync::OnceCell;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::info;

mod folds;
mod sparse;
pub(crate) use sparse::SparseMatrix;

/// Parallel chunked buffer accumulation.
///
/// Allocates one `Vec<F>` of length `num_cols` per Rayon thread, fills it by
/// calling `row_fn(buffer, row_idx)` for every row assigned to that thread,
/// then reduces all thread buffers into a single output via parallel element-wise
/// addition.  Thread count is capped so total buffer memory stays ≤ 512 MB.
fn par_chunked_reduce<F, RowFn>(num_rows: usize, num_cols: usize, row_fn: RowFn) -> Vec<F>
where
  F: ff::PrimeField + Send + Sync,
  RowFn: Fn(&mut Vec<F>, usize) + Send + Sync,
{
  let buffer_bytes = num_cols * std::mem::size_of::<F>();
  let max_threads = std::cmp::max(2, 512_000_000 / buffer_bytes);
  let num_threads = std::cmp::min(rayon::current_num_threads(), max_threads);
  let chunk_size = (num_rows + num_threads - 1) / num_threads;

  let mut thread_buffers: Vec<Vec<F>> = (0..num_threads)
    .into_par_iter()
    .map(|thread_idx| {
      let start = thread_idx * chunk_size;
      let end = ((thread_idx + 1) * chunk_size).min(num_rows);
      let mut buffer = vec![F::ZERO; num_cols];
      for row_idx in start..end {
        row_fn(&mut buffer, row_idx);
      }
      buffer
    })
    .collect();

  let mut result = thread_buffers.swap_remove(0);
  for buffer in thread_buffers {
    result
      .par_iter_mut()
      .zip(buffer.par_iter())
      .for_each(|(a, b)| *a += *b);
  }
  result
}

/// Fast-path field multiplication: avoids full mul for common ±1 coefficients.
#[inline(always)]
fn mul_field_fast<F: ff::PrimeField>(x: F, v: &F) -> F {
  if *v == F::ONE {
    x
  } else if *v == -F::ONE {
    -x
  } else {
    x * v
  }
}

fn eq01<F: Field>(bit: u8, r: &F) -> F {
  if bit == 0 { F::ONE - *r } else { *r }
}

#[inline]
pub(crate) fn weights_from_r<F: Field>(r_bs: &[F], n: usize) -> Vec<F> {
  let ell = r_bs.len();
  (0..n)
    .map(|i| {
      let mut wi = F::ONE;
      let mut k = i;
      for r_bs_t in r_bs.iter().take(ell) {
        wi *= eq01((k & 1) as u8, r_bs_t);
        k >>= 1;
      }
      wi
    })
    .collect()
}

/// A type that holds the shape of the R1CS matrices
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct R1CSShape<E: Engine, V = <E as Engine>::Scalar> {
  pub(crate) num_cons: usize,
  pub(crate) num_vars: usize,
  pub(crate) num_io: usize, // input/output
  pub(crate) A: SparseMatrix<V>,
  pub(crate) B: SparseMatrix<V>,
  pub(crate) C: SparseMatrix<V>,
  #[serde(skip, default = "OnceCell::new")]
  pub(crate) digest: OnceCell<E::Scalar>,
}

impl<E: Engine, V: Serialize> SimpleDigestible for R1CSShape<E, V> {}

/// A type that holds a witness for a given R1CS instance
///
/// The type parameter `V` controls the element type of the witness vector.
/// Defaults to `E::Scalar` (field elements). Use `i64` for the small-value path.
/// When `V = i64`, the witness is in pure integer form and must be converted
/// to field elements before folding or PCS operations.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "V: Serialize + for<'a> Deserialize<'a>")]
pub struct R1CSWitness<E: Engine, V = <E as Engine>::Scalar> {
  /// Whether the witness elements fit in machine words.
  /// Only meaningful when V = E::Scalar.
  pub is_small: bool,
  /// The witness vector.
  pub W: Vec<V>,
  /// Blinding factor for the witness commitment.
  pub r_W: Blind<E>,
}

/// A type that holds an R1CS instance
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "V: Serialize + for<'a> Deserialize<'a>")]
pub struct R1CSInstance<E: Engine, V = <E as Engine>::Scalar> {
  /// Commitment to the witness.
  pub comm_W: Commitment<E>,
  /// Public input/output vector.
  pub X: Vec<V>,
}

/// A type that holds a witness for a given Relaxed R1CS instance
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct RelaxedR1CSWitness<E: Engine> {
  pub(crate) W: Vec<E::Scalar>,
  pub(crate) r_W: Blind<E>,
  pub(crate) E: Vec<E::Scalar>,
  pub(crate) r_E: Blind<E>,
}

/// A type that holds a Relaxed R1CS instance
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct RelaxedR1CSInstance<E: Engine> {
  pub(crate) comm_W: Commitment<E>,
  pub(crate) comm_E: Commitment<E>,
  pub(crate) X: Vec<E::Scalar>,
  pub(crate) u: E::Scalar,
}

impl<E: Engine> RelaxedR1CSWitness<E> {
  /// Commits to the witness using the supplied generators
  pub fn commit(
    &self,
    ck: &CommitmentKey<E>,
  ) -> Result<(Commitment<E>, Commitment<E>), SpartanError> {
    Ok((
      PCS::<E>::commit(ck, &self.W, &self.r_W, false)?,
      PCS::<E>::commit(ck, &self.E, &self.r_E, false)?,
    ))
  }
}

#[cfg(test)]
mod tests_relaxed_sample {
  use super::*;
  use crate::{provider::P256HyraxEngine, traits::Engine};
  use ff::Field;

  fn tiny_r1cs<E: Engine>(num_vars: usize) -> R1CSShape<E> {
    let one = <E::Scalar as Field>::ONE;
    let (num_cons, num_vars, num_io, a_entries, b_entries, c_entries) = {
      let num_cons = 4;
      let num_io = 2;

      let mut A: Vec<(usize, usize, E::Scalar)> = Vec::new();
      let mut B: Vec<(usize, usize, E::Scalar)> = Vec::new();
      let mut C: Vec<(usize, usize, E::Scalar)> = Vec::new();

      // constraint 0: I0 * I0 - Z0 = 0
      A.push((0, num_vars + 1, one));
      B.push((0, num_vars + 1, one));
      C.push((0, 0, one));

      // constraint 1: Z0 * I0 - Z1 = 0
      A.push((1, 0, one));
      B.push((1, num_vars + 1, one));
      C.push((1, 1, one));

      // constraint 2: (Z1 + I0) * 1 - Z2 = 0
      A.push((2, 1, one));
      A.push((2, num_vars + 1, one));
      B.push((2, num_vars, one));
      C.push((2, 2, one));

      // constraint 3: (Z2 + 5) * 1 - I1 = 0
      A.push((3, 2, one));
      A.push((3, num_vars, one + one + one + one + one));
      B.push((3, num_vars, one));
      C.push((3, num_vars + 2, one));

      (num_cons, num_vars, num_io, A, B, C)
    };

    let rows = num_cons;
    let cols = num_vars + num_io + 1;

    R1CSShape::new(
      num_cons,
      num_vars,
      num_io,
      SparseMatrix::new(&a_entries, rows, cols),
      SparseMatrix::new(&b_entries, rows, cols),
      SparseMatrix::new(&c_entries, rows, cols),
    )
    .unwrap()
  }

  fn test_random_sample_with<E: Engine>() {
    let s = tiny_r1cs::<E>(4);
    let (ck, _) = s.commitment_key();
    let (inst, wit) = s.sample_random_instance_witness(&ck).unwrap();
    assert!(s.is_sat_relaxed(&ck, &inst, &wit).is_ok());
  }

  #[test]
  fn test_random_sample() {
    test_random_sample_with::<P256HyraxEngine>();
  }
}

/// Round `n` up to the next multiple of width.
/// (If `n` is already a multiple and higher than zero, it is returned unchanged.)
#[inline]
pub fn pad_to_width(width: usize, n: usize) -> usize {
  if n == 0 {
    return 0;
  }

  // width == 1024 == 1 << 10, so the mask is width-1 == 0b111_1111_1111 (10 bits set).
  n.saturating_add(width - 1) & !(width - 1)
}

fn is_sparse_matrix_valid<V: Copy>(
  num_rows: usize,
  num_cols: usize,
  M: &SparseMatrix<V>,
) -> Result<(), SpartanError> {
  // Check if the indices and indptr are valid for the given number of rows and columns
  M.iter().try_for_each(|(row, col, _val)| {
    if row >= num_rows || col >= num_cols {
      Err(SpartanError::InvalidIndex)
    } else {
      Ok(())
    }
  })
}

impl<E: Engine> R1CSShape<E> {
  /// Create an object of type `R1CSShape` from the explicitly specified R1CS matrices
  pub fn new(
    num_cons: usize,
    num_vars: usize,
    num_io: usize,
    A: SparseMatrix<E::Scalar>,
    B: SparseMatrix<E::Scalar>,
    C: SparseMatrix<E::Scalar>,
  ) -> Result<R1CSShape<E>, SpartanError> {
    let num_rows = num_cons;
    let num_cols = num_vars + 1 + num_io; // +1 for the constant term

    is_sparse_matrix_valid(num_rows, num_cols, &A)?;
    is_sparse_matrix_valid(num_rows, num_cols, &B)?;
    is_sparse_matrix_valid(num_rows, num_cols, &C)?;

    Ok(R1CSShape {
      num_cons,
      num_vars,
      num_io,
      A,
      B,
      C,
      digest: OnceCell::new(),
    })
  }

  /// Pads the `R1CSShape` so that the shape passes `is_regular_shape`
  /// Renumbers variables to accommodate padded variables
  pub fn pad(&self) -> Self {
    // check if the provided R1CSShape is already as required
    if self.is_regular_shape() {
      return self.clone();
    }

    // equalize the number of variables and public IO
    let m = self.num_vars.max(self.num_io).next_power_of_two();

    // check if the number of variables are as expected, then
    // we simply set the number of constraints to the next power of two
    if self.num_vars == m {
      return R1CSShape {
        num_cons: self.num_cons.next_power_of_two(),
        num_vars: m,
        num_io: self.num_io,
        A: self.A.clone(),
        B: self.B.clone(),
        C: self.C.clone(),
        digest: OnceCell::new(),
      };
    }

    // otherwise, we need to pad the number of variables and renumber variable accesses
    let num_vars_padded = m;
    let num_cons_padded = self.num_cons.next_power_of_two();

    let apply_pad = |mut M: SparseMatrix<E::Scalar>| -> SparseMatrix<E::Scalar> {
      M.indices.par_iter_mut().for_each(|c| {
        if *c >= self.num_vars {
          *c += num_vars_padded - self.num_vars
        }
      });

      M.cols += num_vars_padded - self.num_vars;

      let ex = {
        let nnz = M.indptr.last().unwrap();
        vec![*nnz; num_cons_padded - self.num_cons]
      };
      M.indptr.extend(ex);
      M
    };

    let A_padded = apply_pad(self.A.clone());
    let B_padded = apply_pad(self.B.clone());
    let C_padded = apply_pad(self.C.clone());

    R1CSShape {
      num_cons: num_cons_padded,
      num_vars: num_vars_padded,
      num_io: self.num_io,
      A: A_padded,
      B: B_padded,
      C: C_padded,
      digest: OnceCell::new(),
    }
  }

  // Checks regularity conditions on the R1CSShape, required in Spartan-class SNARKs
  // Returns false if num_cons or num_vars are not powers of two, or if num_io > num_vars
  #[inline]
  pub(crate) fn is_regular_shape(&self) -> bool {
    let cons_valid = self.num_cons.next_power_of_two() == self.num_cons;
    let vars_valid = self.num_vars.next_power_of_two() == self.num_vars;
    let io_lt_vars = self.num_io < self.num_vars;
    cons_valid && vars_valid && io_lt_vars
  }

  /// Checks if the R1CS instance is satisfiable given a witness and its shape
  pub fn is_sat(
    &self,
    ck: &CommitmentKey<E>,
    U: &R1CSInstance<E>,
    W: &R1CSWitness<E>,
  ) -> Result<(), SpartanError> {
    assert_eq!(W.W.len(), self.num_vars);
    assert_eq!(U.X.len(), self.num_io);

    // verify if Az * Bz = u*Cz
    let res_eq = {
      let z = [W.W.clone(), vec![E::Scalar::ONE], U.X.clone()].concat();
      let (Az, Bz, Cz) = self.multiply_vec(&z)?;
      assert_eq!(Az.len(), self.num_cons);
      assert_eq!(Bz.len(), self.num_cons);
      assert_eq!(Cz.len(), self.num_cons);

      (0..self.num_cons).all(|i| Az[i] * Bz[i] == Cz[i])
    };

    // verify if comm_W is a commitment to W
    let res_comm = U.comm_W == PCS::<E>::commit(ck, &W.W, &W.r_W, W.is_small)?;

    if !res_eq {
      return Err(SpartanError::UnSat {
        reason: "R1CS is unsatisfiable".to_string(),
      });
    }

    if !res_comm {
      return Err(SpartanError::UnSat {
        reason: "Invalid commitment".to_string(),
      });
    }

    Ok(())
  }

  /// Generates public parameters for a Rank-1 Constraint System (R1CS).
  ///
  /// This function takes into consideration the shape of the R1CS matrices
  ///
  /// # Arguments
  ///
  /// * `S`: The shape of the R1CS matrices.
  ///
  pub fn commitment_key(&self) -> (CommitmentKey<E>, VerifierKey<E>) {
    E::PCS::setup(b"ck", self.num_vars, DEFAULT_COMMITMENT_WIDTH)
  }

  pub fn multiply_vec(
    &self,
    z: &[E::Scalar],
  ) -> Result<(Vec<E::Scalar>, Vec<E::Scalar>, Vec<E::Scalar>), SpartanError> {
    if z.len() != self.num_io + 1 + self.num_vars {
      return Err(SpartanError::InvalidWitnessLength);
    }

    let (Az, (Bz, Cz)) = rayon::join(
      || self.A.multiply_vec(z),
      || rayon::join(|| self.B.multiply_vec(z), || self.C.multiply_vec(z)),
    );

    Ok((Az?, Bz?, Cz?))
  }
  /// Checks if the Relaxed R1CS instance is satisfiable given a witness and its shape
  pub fn is_sat_relaxed(
    &self,
    ck: &CommitmentKey<E>,
    U: &RelaxedR1CSInstance<E>,
    W: &RelaxedR1CSWitness<E>,
  ) -> Result<(), SpartanError> {
    assert_eq!(W.W.len(), self.num_vars);
    assert_eq!(W.E.len(), self.num_cons);
    assert_eq!(U.X.len(), self.num_io);

    // verify if Az * Bz = u*Cz + E
    let res_eq = {
      let z = [W.W.clone(), vec![U.u], U.X.clone()].concat();
      let (az, bz, cz) = self.multiply_vec(&z)?;
      (0..self.num_cons).all(|i| az[i] * bz[i] == U.u * cz[i] + W.E[i])
    };

    // verify if comm_E and comm_W are commitments to E and W
    let res_comm = {
      let comm_W = PCS::<E>::commit(ck, &W.W, &W.r_W, false)?;
      let comm_E = PCS::<E>::commit(ck, &W.E, &W.r_E, false)?;
      U.comm_W == comm_W && U.comm_E == comm_E
    };

    if !res_eq {
      return Err(SpartanError::UnSat {
        reason: "Relaxed R1CS is unsatisfiable".to_string(),
      });
    }

    if !res_comm {
      return Err(SpartanError::UnSat {
        reason: "Invalid commitments".to_string(),
      });
    }

    Ok(())
  }

  /// Samples a new random `RelaxedR1CSInstance`/`RelaxedR1CSWitness` pair
  pub fn sample_random_instance_witness(
    &self,
    ck: &CommitmentKey<E>,
  ) -> Result<(RelaxedR1CSInstance<E>, RelaxedR1CSWitness<E>), SpartanError> {
    // sample Z = (W, u, X)
    let Z = (0..self.num_vars + self.num_io + 1)
      .into_par_iter()
      .map(|_| E::Scalar::random(&mut rand_core::OsRng))
      .collect::<Vec<E::Scalar>>();

    let r_W = PCS::<E>::blind(ck, self.num_vars);
    let r_E = PCS::<E>::blind(ck, self.num_cons);

    let u = Z[self.num_vars];

    // compute E <- AZ o BZ - u * CZ
    let (az, bz, cz) = self.multiply_vec(&Z)?;
    let E_vec = az
      .par_iter()
      .zip(bz.par_iter())
      .zip(cz.par_iter())
      .map(|((az_i, bz_i), cz_i)| *az_i * *bz_i - u * *cz_i)
      .collect::<Vec<E::Scalar>>();

    // compute commitments to W,E in parallel
    let (comm_W_res, comm_E_res) = rayon::join(
      || PCS::<E>::commit(ck, &Z[..self.num_vars], &r_W, false),
      || PCS::<E>::commit(ck, &E_vec, &r_E, false),
    );

    Ok((
      RelaxedR1CSInstance {
        comm_W: comm_W_res?,
        comm_E: comm_E_res?,
        u,
        X: Z[self.num_vars + 1..].to_vec(),
      },
      RelaxedR1CSWitness {
        W: Z[..self.num_vars].to_vec(),
        r_W,
        E: E_vec,
        r_E,
      },
    ))
  }
}

impl<E: Engine> R1CSWitness<E> {
  /// A method to create a witness object using a vector of scalars
  pub fn new(
    ck: &CommitmentKey<E>,
    S: &R1CSShape<E>,
    W: &mut Vec<E::Scalar>,
    is_small: bool,
  ) -> Result<(R1CSWitness<E>, Commitment<E>), SpartanError> {
    let r_W = PCS::<E>::blind(ck, W.len());

    // pad with zeros
    let (_pad_span, pad_t) = start_span!("pad_witness");
    if W.len() < S.num_vars {
      W.resize(S.num_vars, E::Scalar::ZERO);
    }
    info!(elapsed_ms = %pad_t.elapsed().as_millis(), "pad_witness");

    let (_commit_span, commit_t) = start_span!("commit_witness");
    let comm_W = PCS::<E>::commit(ck, W, &r_W, is_small)?;
    info!(elapsed_ms = %commit_t.elapsed().as_millis(), "commit_witness");

    let W = R1CSWitness {
      W: W.to_vec(),
      r_W,
      is_small,
    };

    Ok((W, comm_W))
  }

  /// A method to create a witness object using a vector of scalars
  pub fn new_unchecked(
    W: Vec<E::Scalar>,
    r_W: Blind<E>,
    is_small: bool,
  ) -> Result<R1CSWitness<E>, SpartanError> {
    Ok(Self { W, r_W, is_small })
  }

  /// Fold multiple witnesses with a sequence of r_b values
  pub fn fold_multiple(
    r_bs: &[E::Scalar],
    Ws: &[R1CSWitness<E>],
  ) -> Result<R1CSWitness<E>, SpartanError>
  where
    E::PCS: FoldingEngineTrait<E>,
  {
    let n = Ws.len();
    if n == 0 {
      return Err(SpartanError::InvalidInputLength {
        reason: "fold_multiple: empty witness list".into(),
      });
    }

    let w = weights_from_r::<E::Scalar>(r_bs, n);

    if w.len() != n {
      return Err(SpartanError::InvalidInputLength {
        reason: "fold_multiple: weights length mismatch".into(),
      });
    }

    let dim = Ws[0].W.len();

    if !Ws.iter().all(|z| z.W.len() == dim) {
      return Err(SpartanError::InvalidInputLength {
        reason: "fold_multiple: all W vectors must have the same length".into(),
      });
    }

    let mut acc_W = vec![E::Scalar::ZERO; dim];
    let tile = 4096; // process 4096 elements at a time
    acc_W
      .par_chunks_mut(tile)
      .enumerate()
      .for_each(|(block_idx, acc_blk)| {
        let start = block_idx * tile;
        let end = start + acc_blk.len(); // last block may be < tile

        // Stream over the small number of rows for this block.
        // This keeps both `acc_blk` and the row-slice hot in cache.
        for (i, &wi) in w.iter().enumerate() {
          let row_slice = &Ws[i].W[start..end];
          // Accumulate: acc_blk += wi * row_slice
          for (a, &x) in acc_blk.iter_mut().zip(row_slice.iter()) {
            *a += wi * x;
          }
        }
      });

    let acc_r = <E::PCS as FoldingEngineTrait<E>>::fold_blinds(
      &Ws.iter().map(|wz| wz.r_W.clone()).collect::<Vec<_>>(),
      &w,
    )?;

    Ok(R1CSWitness::<E> {
      W: acc_W,
      r_W: acc_r,
      is_small: false,
    })
  }
}

impl<E: Engine> R1CSInstance<E> {
  /// A method to create an instance object using constituent elements
  pub fn new(
    S: &R1CSShape<E>,
    comm_W: &Commitment<E>,
    X: &[E::Scalar],
  ) -> Result<R1CSInstance<E>, SpartanError> {
    if S.num_io != X.len() {
      Err(SpartanError::InvalidInputLength {
        reason: format!(
          "R1CS instance: Expected {} elements in X, got {}",
          S.num_io,
          X.len()
        ),
      })
    } else {
      Ok(R1CSInstance {
        comm_W: comm_W.clone(),
        X: X.to_owned(),
      })
    }
  }

  /// A method to create an instance object using constituent elements
  pub fn new_unchecked(
    comm_W: Commitment<E>,
    X: Vec<E::Scalar>,
  ) -> Result<R1CSInstance<E>, SpartanError> {
    Ok(R1CSInstance { comm_W, X })
  }

  /// Fold multiple instances with a sequence of r_b values
  pub fn fold_multiple(
    r_bs: &[E::Scalar],
    Us: &[R1CSInstance<E>],
  ) -> Result<R1CSInstance<E>, SpartanError>
  where
    E::PCS: FoldingEngineTrait<E>,
  {
    let n = Us.len();
    let w = weights_from_r::<E::Scalar>(r_bs, n);
    let d = Us[0].X.len();

    // X
    let mut X_acc = vec![E::Scalar::ZERO; d];
    for (i, Ui) in Us.iter().enumerate() {
      let wi = w[i];
      for (j, Uij) in Ui.X.iter().enumerate() {
        X_acc[j] += wi * Uij;
      }
    }

    // commitment (group lin. comb)
    let comm_acc = <E::PCS as FoldingEngineTrait<E>>::fold_commitments(
      &Us.iter().map(|U| U.comm_W.clone()).collect::<Vec<_>>(),
      &w,
    )?;

    Ok(R1CSInstance::<E> {
      X: X_acc,
      comm_W: comm_acc,
    })
  }
}

impl<E: Engine> TranscriptReprTrait<E::GE> for R1CSInstance<E> {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    [
      self.comm_W.to_transcript_bytes(),
      self.X.as_slice().to_transcript_bytes(),
    ]
    .concat()
  }
}

///
////////////////// Split R1CS Types //////////////////
///
/// A type that holds a split R1CS shape
///
/// The type parameter `V` controls the coefficient type stored in the matrices.
/// Defaults to `E::Scalar` (field elements) for the standard path.
/// Use `i32` for the pure-integer small-value path.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "V: Serialize + for<'a> Deserialize<'a>")]
pub struct SplitR1CSShape<E: Engine, V = <E as Engine>::Scalar> {
  /// Number of constraints (padded).
  pub num_cons: usize,

  /// Number of constraints before padding.
  pub num_cons_unpadded: usize,
  /// Shared variables before padding.
  pub num_shared_unpadded: usize,
  /// Precommitted variables before padding.
  pub num_precommitted_unpadded: usize,
  /// Rest of variables before padding.
  pub num_rest_unpadded: usize,

  /// Number of shared variables.
  pub num_shared: usize,
  /// Number of precommitted variables.
  pub num_precommitted: usize,
  /// Number of rest variables.
  pub num_rest: usize,
  /// Number of public variables.
  pub num_public: usize,
  /// Number of public challenges.
  pub num_challenges: usize,
  /// A matrix.
  pub A: SparseMatrix<V>,
  /// B matrix.
  pub B: SparseMatrix<V>,
  /// C matrix.
  pub C: SparseMatrix<V>,
  #[serde(skip, default = "OnceCell::new")]
  pub(crate) digest: OnceCell<E::Scalar>,
  /// Column remap: compact dense index → original column index.
  /// Only populated for i32 shapes (the integer proving path).
  #[serde(skip, default)]
  pub(crate) dense_to_col: Vec<u32>,
  /// Column remap: original column index → compact dense index.
  /// Untouched columns map to u32::MAX. Only populated for i32 shapes.
  #[serde(skip, default)]
  pub(crate) col_to_dense: Vec<u32>,
  /// Per-row split: A.data[indptr[row]..A_unit_end[row]] are ±1 entries.
  /// A.data[A_unit_end[row]..indptr[row+1]] are non-±1 entries.
  #[serde(skip, default)]
  pub(crate) A_unit_end: Vec<usize>,
  #[serde(skip, default)]
  pub(crate) B_unit_end: Vec<usize>,
  #[serde(skip, default)]
  pub(crate) C_unit_end: Vec<usize>,
  /// Pre-remapped column indices for A: A_dense_col[i] = col_to_dense[A.indices[i]].
  /// Eliminates random access into col_to_dense during the hot loop.
  #[serde(skip, default)]
  pub(crate) A_dense_col: Vec<u32>,
  #[serde(skip, default)]
  pub(crate) B_dense_col: Vec<u32>,
  #[serde(skip, default)]
  pub(crate) C_dense_col: Vec<u32>,
  /// CSC (column-sorted) entries for cache-efficient scatter in bind_row_vars_combined_small.
  /// For each matrix: col_ptr[c]..col_ptr[c+1] gives the range of entries for dense column c.
  /// row_indices[j] gives the row, values[j] gives the coefficient.
  /// col_unit_end[c] splits ±1 entries (col_ptr[c]..col_unit_end[c]) from non-±1.
  #[serde(skip, default)]
  pub(crate) A_csc_col_ptr: Vec<usize>,
  #[serde(skip, default)]
  pub(crate) A_csc_row: Vec<u32>,
  #[serde(skip, default)]
  pub(crate) A_csc_data: Vec<V>,
  #[serde(skip, default)]
  pub(crate) A_csc_unit_end: Vec<usize>,
  #[serde(skip, default)]
  pub(crate) B_csc_col_ptr: Vec<usize>,
  #[serde(skip, default)]
  pub(crate) B_csc_row: Vec<u32>,
  #[serde(skip, default)]
  pub(crate) B_csc_data: Vec<V>,
  #[serde(skip, default)]
  pub(crate) B_csc_unit_end: Vec<usize>,
  #[serde(skip, default)]
  pub(crate) C_csc_col_ptr: Vec<usize>,
  #[serde(skip, default)]
  pub(crate) C_csc_row: Vec<u32>,
  #[serde(skip, default)]
  pub(crate) C_csc_data: Vec<V>,
  #[serde(skip, default)]
  pub(crate) C_csc_unit_end: Vec<usize>,
  /// Per-column presence bitmask: bit 0 = A non-empty, bit 1 = B non-empty, bit 2 = C non-empty.
  /// Used to skip Montgomery multiplications for empty matrix columns in the hot path.
  #[serde(skip, default)]
  pub(crate) col_presence: Vec<u8>,
}

impl<E: Engine, V: Serialize + for<'a> Deserialize<'a>> SimpleDigestible for SplitR1CSShape<E, V> {}


impl<E: Engine, V> SplitR1CSShape<E, V> {
  /// Returns sizes associated with the SplitR1CSShape.
  pub fn sizes(&self) -> [usize; 10] {
    [
      self.num_cons_unpadded,
      self.num_shared_unpadded,
      self.num_precommitted_unpadded,
      self.num_rest_unpadded,
      self.num_cons,
      self.num_shared,
      self.num_precommitted,
      self.num_rest,
      self.num_public,
      self.num_challenges,
    ]
  }
}

/// A type that holds a split R1CS instance
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "V: Serialize + for<'a> Deserialize<'a>")]
pub struct SplitR1CSInstance<E: Engine, V = <E as Engine>::Scalar> {
  pub(crate) comm_W_shared: Option<Commitment<E>>,
  pub(crate) comm_W_precommitted: Option<Commitment<E>>,
  pub(crate) comm_W_rest: Commitment<E>,

  pub(crate) public_values: Vec<V>,
  pub(crate) challenges: Vec<E::Scalar>, // always field — from transcript
}

impl<E: Engine> SplitR1CSShape<E> {
  /// Create an object of type `R1CSShape` from the explicitly specified R1CS matrices
  pub fn new(
    num_cons: usize,
    num_shared: usize,
    num_precommitted: usize,
    num_rest: usize,
    num_public: usize,
    num_challenges: usize,
    A: SparseMatrix<E::Scalar>,
    B: SparseMatrix<E::Scalar>,
    C: SparseMatrix<E::Scalar>,
  ) -> Result<SplitR1CSShape<E>, SpartanError> {
    let width = DEFAULT_COMMITMENT_WIDTH;

    let num_rows = num_cons;
    let num_cols = num_shared + num_precommitted + num_rest + 1 + num_public + num_challenges; // +1 for the constant term

    is_sparse_matrix_valid(num_rows, num_cols, &A)?;
    is_sparse_matrix_valid(num_rows, num_cols, &B)?;
    is_sparse_matrix_valid(num_rows, num_cols, &C)?;

    // We need to pad num_shared, num_precommitted, and num_rest. We need each of them to be a multiple of num_cols.
    let num_shared_padded = pad_to_width(width, num_shared);
    let num_precommitted_padded = pad_to_width(width, num_precommitted);
    let mut num_rest_padded = pad_to_width(width, num_rest);

    // We need to make sure num_vars_padded >= num_public + num_challenges + 1 (for the constant term).
    let num_vars_padded = num_shared_padded + num_precommitted_padded + num_rest_padded;
    if num_vars_padded < num_public + num_challenges + 1 {
      // If not, we need to pad the rest to make it at least num_public + num_challenges + 1.
      num_rest_padded = max(num_public + num_challenges + 1, num_vars_padded)
        - (num_shared_padded + num_precommitted_padded);
    }

    // We need to make sure num_shared_padded + num_precommitted_padded + num_rest_padded is a power of two.
    let num_vars_padded = num_shared_padded + num_precommitted_padded + num_rest_padded;
    if num_vars_padded.next_power_of_two() != num_vars_padded {
      // If not, we need to pad the rest to the next power of two.
      num_rest_padded =
        num_vars_padded.next_power_of_two() - (num_shared_padded + num_precommitted_padded);
    }

    let num_vars = num_shared + num_precommitted + num_rest;
    let num_vars_padded = num_shared_padded + num_precommitted_padded + num_rest_padded;
    let num_cons_padded = num_cons.next_power_of_two();

    let apply_pad = |mut M: SparseMatrix<E::Scalar>| -> SparseMatrix<E::Scalar> {
      M.indices.par_iter_mut().for_each(|c| {
        if *c >= num_shared && *c < num_shared + num_precommitted {
          // precommitted variables
          *c += num_shared_padded - num_shared;
        } else if *c >= num_shared + num_precommitted && *c < num_vars {
          // rest of the variables
          *c += num_shared_padded + num_precommitted_padded - num_shared - num_precommitted;
        } else if *c >= num_vars {
          // public and challenge variables
          *c += num_vars_padded - num_vars;
        }
      });

      M.cols += num_vars_padded - num_vars;

      let ex = {
        let nnz = if M.indptr.is_empty() {
          0
        } else {
          M.indptr[M.indptr.len() - 1]
        };
        vec![nnz; num_cons_padded - num_cons]
      };
      M.indptr.extend(ex);
      M
    };

    let A_padded = apply_pad(A);
    let B_padded = apply_pad(B);
    let C_padded = apply_pad(C);

    Ok(Self {
      num_cons: num_cons_padded,
      num_shared: num_shared_padded,
      num_precommitted: num_precommitted_padded,
      num_rest: num_rest_padded,

      num_cons_unpadded: num_cons,
      num_shared_unpadded: num_shared,
      num_precommitted_unpadded: num_precommitted,
      num_rest_unpadded: num_rest,

      num_public,
      num_challenges,
      A: A_padded,
      B: B_padded,
      C: C_padded,
      digest: OnceCell::new(),
      dense_to_col: Vec::new(),
      col_to_dense: Vec::new(),
      A_unit_end: Vec::new(),
      B_unit_end: Vec::new(),
      C_unit_end: Vec::new(),
      A_dense_col: Vec::new(),
      B_dense_col: Vec::new(),
      C_dense_col: Vec::new(),
      A_csc_col_ptr: Vec::new(),
      A_csc_row: Vec::new(),
      A_csc_data: Vec::new(),
      A_csc_unit_end: Vec::new(),
      B_csc_col_ptr: Vec::new(),
      B_csc_row: Vec::new(),
      B_csc_data: Vec::new(),
      B_csc_unit_end: Vec::new(),
      C_csc_col_ptr: Vec::new(),
      C_csc_row: Vec::new(),
      C_csc_data: Vec::new(),
      C_csc_unit_end: Vec::new(),
      col_presence: Vec::new(),
    })
  }

  pub fn equalize(S_A: &mut Self, S_B: &mut Self) {
    let orig_cons_a = S_A.num_cons;
    let orig_cons_b = S_B.num_cons;

    let num_cons_padded = max(S_A.num_cons, S_B.num_cons);
    let num_vars_padded = max(
      S_A.num_shared + S_A.num_precommitted + S_A.num_rest,
      S_B.num_shared + S_B.num_precommitted + S_B.num_rest,
    );

    S_A.num_cons = num_cons_padded;
    S_B.num_cons = num_cons_padded;

    let move_public_vars = |M: &mut SparseMatrix<E::Scalar>, num_cons: usize, num_vars: usize| {
      M.indices.par_iter_mut().for_each(|c| {
        if *c >= num_vars {
          // public and challenge variables
          *c += num_vars_padded - num_vars;
        }
      });

      M.cols += num_vars_padded - num_vars;

      let ex = {
        let nnz = if M.indptr.is_empty() {
          0
        } else {
          M.indptr[M.indptr.len() - 1]
        };
        vec![nnz; num_cons_padded - num_cons]
      };
      M.indptr.extend(ex);
    };

    // Grow variables (if needed) and pad rows using original constraint counts
    if S_A.num_shared + S_A.num_precommitted + S_A.num_rest != num_vars_padded {
      let num_vars = S_A.num_shared + S_A.num_precommitted + S_A.num_rest;
      S_A.num_rest = num_vars_padded - (S_A.num_shared + S_A.num_precommitted);
      move_public_vars(&mut S_A.A, orig_cons_a, num_vars);
      move_public_vars(&mut S_A.B, orig_cons_a, num_vars);
      move_public_vars(&mut S_A.C, orig_cons_a, num_vars);
    } else {
      // No var growth; still ensure row padding happens
      let num_vars = S_A.num_shared + S_A.num_precommitted + S_A.num_rest;
      move_public_vars(&mut S_A.A, orig_cons_a, num_vars);
      move_public_vars(&mut S_A.B, orig_cons_a, num_vars);
      move_public_vars(&mut S_A.C, orig_cons_a, num_vars);
    }

    if S_B.num_shared + S_B.num_precommitted + S_B.num_rest != num_vars_padded {
      let num_vars = S_B.num_shared + S_B.num_precommitted + S_B.num_rest;
      S_B.num_rest = num_vars_padded - (S_B.num_shared + S_B.num_precommitted);
      move_public_vars(&mut S_B.A, orig_cons_b, num_vars);
      move_public_vars(&mut S_B.B, orig_cons_b, num_vars);
      move_public_vars(&mut S_B.C, orig_cons_b, num_vars);
    } else {
      let num_vars = S_B.num_shared + S_B.num_precommitted + S_B.num_rest;
      move_public_vars(&mut S_B.A, orig_cons_b, num_vars);
      move_public_vars(&mut S_B.B, orig_cons_b, num_vars);
      move_public_vars(&mut S_B.C, orig_cons_b, num_vars);
    }
  }

  pub fn to_regular_shape(&self) -> R1CSShape<E> {
    R1CSShape {
      num_cons: self.num_cons,
      num_vars: self.num_shared + self.num_precommitted + self.num_rest,
      num_io: self.num_public + self.num_challenges,
      A: self.A.clone(),
      B: self.B.clone(),
      C: self.C.clone(),
      digest: OnceCell::new(),
    }
  }

  /// Generates public parameters for a Rank-1 Constraint System (R1CS).
  ///
  /// This function takes into consideration the shape of the R1CS matrices
  ///
  /// # Arguments
  ///
  /// * `S`: The shape of the R1CS matrices.
  ///
  pub fn commitment_key(
    shapes: &[&SplitR1CSShape<E>],
  ) -> Result<(CommitmentKey<E>, VerifierKey<E>), SpartanError> {
    let max = shapes
      .iter()
      .map(|s| s.num_shared + s.num_precommitted + s.num_rest)
      .max()
      .ok_or(SpartanError::InvalidInputLength {
        reason: "commitment_key: unable to find max number of variables".to_string(),
      })?;

    Ok(E::PCS::setup(b"ck", max, DEFAULT_COMMITMENT_WIDTH))
  }

  pub fn multiply_vec(
    &self,
    z: &[E::Scalar],
  ) -> Result<(Vec<E::Scalar>, Vec<E::Scalar>, Vec<E::Scalar>), SpartanError> {
    if z.len()
      != self.num_public
        + self.num_challenges
        + 1
        + self.num_shared
        + self.num_precommitted
        + self.num_rest
    {
      return Err(SpartanError::InvalidWitnessLength);
    }

    let (Az, (Bz, Cz)) = rayon::join(
      || self.A.multiply_vec(z),
      || rayon::join(|| self.B.multiply_vec(z), || self.C.multiply_vec(z)),
    );

    Ok((Az?, Bz?, Cz?))
  }

  /// Evaluates the MLE of R1CS matrices at the provided point
  pub fn evaluate_with_tables(
    &self,
    T_x: &[E::Scalar],
    T_y: &[E::Scalar],
  ) -> (E::Scalar, E::Scalar, E::Scalar) {
    let multi_eval = |M: &SparseMatrix<E::Scalar>| -> E::Scalar {
      M.indptr
        .par_windows(2)
        .enumerate()
        .map(|(row_idx, ptrs)| {
          M.get_row_unchecked(ptrs.try_into().unwrap())
            .map(|(val, col_idx)| {
              let prod = T_x[row_idx] * T_y[*col_idx];
              if *val == E::Scalar::ONE {
                prod
              } else if *val == -E::Scalar::ONE {
                -prod
              } else {
                prod * val
              }
            })
            .sum::<E::Scalar>()
        })
        .sum()
    };
    (
      multi_eval(&self.A),
      multi_eval(&self.B),
      multi_eval(&self.C),
    )
  }

  /// Computes poly_ABC = A·rx + r·(B·rx) + r²·(C·rx) with full row parallelism.
  ///
  /// Uses thread-local buffers (one per thread) to enable parallel processing
  /// of all rows while fusing the A + r·B + r²·C combination in a single pass.
  pub(crate) fn bind_row_vars_combined(&self, rx: &[E::Scalar], r: E::Scalar) -> Vec<E::Scalar> {
    assert_eq!(rx.len(), self.num_cons);

    let num_vars = self.num_shared + self.num_precommitted + self.num_rest;
    let num_cols = 2 * num_vars;
    let r_sq = r * r;

    par_chunked_reduce(self.num_cons, num_cols, |buffer, row_idx| {
      let rx_row = rx[row_idx];
      let rx_r = rx_row * r;
      let rx_r_sq = rx_row * r_sq;

      let a_ptrs = [self.A.indptr[row_idx], self.A.indptr[row_idx + 1]];
      let b_ptrs = [self.B.indptr[row_idx], self.B.indptr[row_idx + 1]];
      let c_ptrs = [self.C.indptr[row_idx], self.C.indptr[row_idx + 1]];

      for (val, col) in self.A.get_row_unchecked(&a_ptrs) {
        buffer[*col] += mul_field_fast(rx_row, val);
      }
      for (val, col) in self.B.get_row_unchecked(&b_ptrs) {
        buffer[*col] += mul_field_fast(rx_r, val);
      }
      for (val, col) in self.C.get_row_unchecked(&c_ptrs) {
        buffer[*col] += mul_field_fast(rx_r_sq, val);
      }
    })
  }
}

// ---- Pure integer methods for SmallCoeff shapes ----

#[allow(dead_code)]
impl<E: Engine, C: SmallCoeff> SplitR1CSShape<E, C> {
  /// Create a small-coeff shape from explicitly specified R1CS matrices.
  pub fn new_int(
    num_cons: usize,
    num_shared: usize,
    num_precommitted: usize,
    num_rest: usize,
    num_public: usize,
    num_challenges: usize,
    A: SparseMatrix<C>,
    B: SparseMatrix<C>,
    C: SparseMatrix<C>,
  ) -> Result<SplitR1CSShape<E, C>, SpartanError> {
    let width = DEFAULT_COMMITMENT_WIDTH;

    let num_rows = num_cons;
    let num_cols = num_shared + num_precommitted + num_rest + 1 + num_public + num_challenges;

    is_sparse_matrix_valid(num_rows, num_cols, &A)?;
    is_sparse_matrix_valid(num_rows, num_cols, &B)?;
    is_sparse_matrix_valid(num_rows, num_cols, &C)?;

    let num_shared_padded = pad_to_width(width, num_shared);
    let num_precommitted_padded = pad_to_width(width, num_precommitted);
    let mut num_rest_padded = pad_to_width(width, num_rest);

    let num_vars_padded = num_shared_padded + num_precommitted_padded + num_rest_padded;
    if num_vars_padded < num_public + num_challenges + 1 {
      num_rest_padded = max(num_public + num_challenges + 1, num_vars_padded)
        - (num_shared_padded + num_precommitted_padded);
    }

    let num_vars_padded = num_shared_padded + num_precommitted_padded + num_rest_padded;
    if num_vars_padded.next_power_of_two() != num_vars_padded {
      num_rest_padded =
        num_vars_padded.next_power_of_two() - (num_shared_padded + num_precommitted_padded);
    }

    let num_vars = num_shared + num_precommitted + num_rest;
    let num_vars_padded = num_shared_padded + num_precommitted_padded + num_rest_padded;
    let num_cons_padded = num_cons.next_power_of_two();

    let apply_pad = |mut M: SparseMatrix<C>| -> SparseMatrix<C> {
      M.indices.par_iter_mut().for_each(|c| {
        if *c >= num_shared && *c < num_shared + num_precommitted {
          *c += num_shared_padded - num_shared;
        } else if *c >= num_shared + num_precommitted && *c < num_vars {
          *c += num_shared_padded + num_precommitted_padded - num_shared - num_precommitted;
        } else if *c >= num_vars {
          *c += num_vars_padded - num_vars;
        }
      });

      M.cols += num_vars_padded - num_vars;

      let ex = {
        let nnz = if M.indptr.is_empty() {
          0
        } else {
          M.indptr[M.indptr.len() - 1]
        };
        vec![nnz; num_cons_padded - num_cons]
      };
      M.indptr.extend(ex);
      M
    };

    let mut A_padded = apply_pad(A);
    let mut B_padded = apply_pad(B);
    let mut C_padded = apply_pad(C);

    // Partition entries within each row: ±1 entries first, non-±1 after.
    let A_unit_end = A_padded.partition_unit_entries();
    let B_unit_end = B_padded.partition_unit_entries();
    let C_unit_end = C_padded.partition_unit_entries();

    // Build column remap: only ~20% of columns are touched after padding.
    // Compact indexing shrinks thread-local buffers in bind_row_vars_combined_int
    // from 64MB to ~6MB (fits in L2 cache).
    let num_buf_cols = 2 * (num_shared_padded + num_precommitted_padded + num_rest_padded);
    let mut touched = vec![false; num_buf_cols];
    for &col in A_padded
      .indices
      .iter()
      .chain(B_padded.indices.iter())
      .chain(C_padded.indices.iter())
    {
      if col < num_buf_cols {
        touched[col] = true;
      }
    }
    let mut dense_to_col = Vec::new();
    let mut col_to_dense = vec![u32::MAX; num_buf_cols];
    for (col, &is_touched) in touched.iter().enumerate() {
      if is_touched {
        col_to_dense[col] = dense_to_col.len() as u32;
        dense_to_col.push(col as u32);
      }
    }

    // Pre-remap column indices: store col_to_dense[indices[i]] in parallel Vec<u32>
    // so the hot loop reads sequential memory instead of random-accessing col_to_dense.
    let A_dense_col: Vec<u32> = A_padded.indices.iter().map(|&c| col_to_dense[c]).collect();
    let B_dense_col: Vec<u32> = B_padded.indices.iter().map(|&c| col_to_dense[c]).collect();
    let C_dense_col: Vec<u32> = C_padded.indices.iter().map(|&c| col_to_dense[c]).collect();

    // Build CSC (column-sorted) representation for cache-efficient scatter.
    // Entries sorted by dense column → sequential buffer writes in bind_row_vars_combined_small.
    let num_dense = dense_to_col.len();
    let build_csc = |matrix: &SparseMatrix<C>,
                     dense_col: &[u32],
                     num_rows: usize,
                     num_dense_cols: usize|
     -> (Vec<usize>, Vec<u32>, Vec<C>, Vec<usize>) {
      // Count entries per column
      let mut col_count = vec![0usize; num_dense_cols];
      for row in 0..num_rows {
        for i in matrix.indptr[row]..matrix.indptr[row + 1] {
          col_count[dense_col[i] as usize] += 1;
        }
      }
      // Build col_ptr from counts
      let mut col_ptr = Vec::with_capacity(num_dense_cols + 1);
      col_ptr.push(0);
      for &count in &col_count {
        col_ptr.push(col_ptr.last().unwrap() + count);
      }
      let total = *col_ptr.last().unwrap();
      let mut row_indices = vec![0u32; total];
      let mut values = vec![C::default(); total];
      // Fill in entries: ±1 entries first within each column, then non-±1
      // Two-pass: first ±1, then non-±1
      let mut write_pos = col_ptr[..num_dense_cols].to_vec();
      // Pass 1: ±1 entries
      for row in 0..num_rows {
        for i in matrix.indptr[row]..matrix.indptr[row + 1] {
          let c = dense_col[i] as usize;
          let v = matrix.data[i];
          if v.is_unit() {
            let pos = write_pos[c];
            row_indices[pos] = row as u32;
            values[pos] = v;
            write_pos[c] = pos + 1;
          }
        }
      }
      let unit_ends: Vec<usize> = write_pos.clone();
      // Pass 2: non-±1 entries
      for row in 0..num_rows {
        for i in matrix.indptr[row]..matrix.indptr[row + 1] {
          let c = dense_col[i] as usize;
          let v = matrix.data[i];
          if !v.is_unit() {
            let pos = write_pos[c];
            row_indices[pos] = row as u32;
            values[pos] = v;
            write_pos[c] = pos + 1;
          }
        }
      }
      (col_ptr, row_indices, values, unit_ends)
    };

    let (A_csc_col_ptr, A_csc_row, A_csc_data, A_csc_unit_end) =
      build_csc(&A_padded, &A_dense_col, num_cons, num_dense);
    let (B_csc_col_ptr, B_csc_row, B_csc_data, B_csc_unit_end) =
      build_csc(&B_padded, &B_dense_col, num_cons, num_dense);
    let (C_csc_col_ptr, C_csc_row, C_csc_data, C_csc_unit_end) =
      build_csc(&C_padded, &C_dense_col, num_cons, num_dense);

    let col_presence: Vec<u8> = (0..num_dense)
      .map(|c| {
        let a = (A_csc_col_ptr[c + 1] > A_csc_col_ptr[c]) as u8;
        let b = (B_csc_col_ptr[c + 1] > B_csc_col_ptr[c]) as u8 * 2;
        let cc = (C_csc_col_ptr[c + 1] > C_csc_col_ptr[c]) as u8 * 4;
        a | b | cc
      })
      .collect();

    Ok(SplitR1CSShape {
      num_cons: num_cons_padded,
      num_shared: num_shared_padded,
      num_precommitted: num_precommitted_padded,
      num_rest: num_rest_padded,

      num_cons_unpadded: num_cons,
      num_shared_unpadded: num_shared,
      num_precommitted_unpadded: num_precommitted,
      num_rest_unpadded: num_rest,

      num_public,
      num_challenges,
      A: A_padded,
      B: B_padded,
      C: C_padded,
      digest: OnceCell::new(),
      dense_to_col,
      col_to_dense,
      A_unit_end,
      B_unit_end,
      C_unit_end,
      A_dense_col,
      B_dense_col,
      C_dense_col,
      A_csc_col_ptr,
      A_csc_row,
      A_csc_data,
      A_csc_unit_end,
      B_csc_col_ptr,
      B_csc_row,
      B_csc_data,
      B_csc_unit_end,
      C_csc_col_ptr,
      C_csc_row,
      C_csc_data,
      C_csc_unit_end,
      col_presence,
    })
  }

  /// Evaluates the MLE of R1CS matrices at the provided point (small-coeff matrix entries).
  pub fn evaluate_with_tables_int(
    &self,
    T_x: &[E::Scalar],
    T_y: &[E::Scalar],
  ) -> (E::Scalar, E::Scalar, E::Scalar)
  where
    E::Scalar: crate::small_field::montgomery::MontgomeryLimbs,
  {
    let multi_eval = |M: &SparseMatrix<C>| -> E::Scalar {
      M.indptr
        .par_windows(2)
        .enumerate()
        .map(|(row_idx, ptrs)| {
          M.get_row_unchecked(ptrs.try_into().unwrap())
            .map(|(val, col_idx)| {
              SmallCoeff::mul_field(*val, &(T_x[row_idx] * T_y[*col_idx]))
            })
            .sum::<E::Scalar>()
        })
        .sum()
    };
    (
      multi_eval(&self.A),
      multi_eval(&self.B),
      multi_eval(&self.C),
    )
  }
}

// ---- Generic SmallCoeff methods ----

impl<E: Engine, Coeff: SmallCoeff> SplitR1CSShape<E, Coeff> {
  /// Generic matrix-vector multiply: Az, Bz, Cz with `Coeff` coefficients and `W` witnesses.
  pub fn multiply_vec_witness<W>(
    &self,
    z: &[W],
  ) -> Result<(Vec<Coeff>, Vec<Coeff>, Vec<Coeff>), SpartanError>
  where
    W: Copy + Default + PartialEq + Send + Sync,
  {
    let expected_len = self.num_public
      + self.num_challenges
      + 1
      + self.num_shared
      + self.num_precommitted
      + self.num_rest;
    if z.len() != expected_len {
      return Err(SpartanError::InvalidWitnessLength);
    }

    let (Az, (Bz, Cz)) = rayon::join(
      || self.A.multiply_vec_witness(z),
      || {
        rayon::join(
          || self.B.multiply_vec_witness(z),
          || self.C.multiply_vec_witness(z),
        )
      },
    );

    Ok((Az?, Bz?, Cz?))
  }

  /// Computes poly_ABC = A·rx + r·(B·rx) + r²·(C·rx).
  ///
  /// Uses column remapping to shrink thread-local buffers from `2 * num_vars` to
  /// `num_dense_cols` (the number of columns actually touched by nonzero entries).
  /// For SHA-256 after padding, ~80% of columns are empty → buffer shrinks from
  /// ~64MB to ~6MB, fitting in L2 cache.
  pub(crate) fn bind_row_vars_combined_small(
    &self,
    rx: &[E::Scalar],
    r: E::Scalar,
  ) -> Vec<E::Scalar>
  where
    E::Scalar: MontgomeryLimbs,
  {
    assert_eq!(rx.len(), self.num_cons);

    let num_vars = self.num_shared + self.num_precommitted + self.num_rest;
    let num_cols = 2 * num_vars;
    let r_sq = r * r;

    let num_dense = self.dense_to_col.len();

    // Fallback: if remap tables aren't built (e.g., after deserialization), use full buffer
    if num_dense == 0 || self.A_dense_col.is_empty() {
      return self.bind_row_vars_combined_small_no_remap(rx, r);
    }

    // Use CSC (column-sorted) iteration if available. Column-major access makes
    // buffer writes sequential (L1 hits), with random rx reads hitting L2.
    // Separate A/B/C buffers: only read from rx (5.4MB, fits in L2) instead of
    // rx + rx_r + rx_r_sq (16.2MB, spills L2). Combine with r, r² at the end.
    if !self.A_csc_col_ptr.is_empty() {
      /// Accumulate CSC entries for one matrix into a register accumulator.
      #[inline(always)]
      fn accumulate_column<F: ff::PrimeField + MontgomeryLimbs, CC: SmallCoeff>(
        rx_vals: &[F],
        col_ptr: &[usize],
        row_indices: &[u32],
        values: &[CC],
        unit_ends: &[usize],
        c: usize,
      ) -> F {
        let start = col_ptr[c];
        let end = col_ptr[c + 1];

        // Depth-1: single entry — skip loop overhead entirely
        if start + 1 == end {
          return if values[start].is_positive() {
            rx_vals[row_indices[start] as usize]
          } else {
            -rx_vals[row_indices[start] as usize]
          };
        }

        let unit_end = unit_ends[c];
        let mut acc = F::ZERO;
        // ±1 entries: add/sub only
        for j in start..unit_end {
          let row = row_indices[j] as usize;
          if values[j].is_positive() {
            acc += rx_vals[row];
          } else {
            acc -= rx_vals[row];
          }
        }
        // Non-±1 entries
        for j in unit_end..end {
          let row = row_indices[j] as usize;
          acc += SmallCoeff::mul_field(values[j], &rx_vals[row]);
        }
        acc
      }

      // Cache-blocked column partitioning. Process all 3 matrices per column inline:
      // result[c] = A_sum + r * B_sum + r² * C_sum. No separate buffers or combine pass.
      // Dispatches on col_presence to skip Montgomery muls for empty matrix columns.
      let col_chunk = std::cmp::min(7000, num_dense);
      let mut compact = vec![E::Scalar::ZERO; num_dense];

      let use_presence = !self.col_presence.is_empty();

      compact.par_chunks_mut(col_chunk).enumerate().for_each(|(tid, chunk)| {
        let col_start = tid * col_chunk;
        for (i, slot) in chunk.iter_mut().enumerate() {
          let c = col_start + i;

          macro_rules! acc_a {
            () => { accumulate_column::<E::Scalar, Coeff>(rx, &self.A_csc_col_ptr, &self.A_csc_row, &self.A_csc_data, &self.A_csc_unit_end, c) };
          }
          macro_rules! acc_b {
            () => { accumulate_column::<E::Scalar, Coeff>(rx, &self.B_csc_col_ptr, &self.B_csc_row, &self.B_csc_data, &self.B_csc_unit_end, c) };
          }
          macro_rules! acc_c {
            () => { accumulate_column::<E::Scalar, Coeff>(rx, &self.C_csc_col_ptr, &self.C_csc_row, &self.C_csc_data, &self.C_csc_unit_end, c) };
          }

          *slot = if use_presence {
            match self.col_presence[c] {
              0 => E::Scalar::ZERO,
              1 => acc_a!(),
              2 => r * acc_b!(),
              3 => acc_a!() + r * acc_b!(),
              4 => r_sq * acc_c!(),
              5 => acc_a!() + r_sq * acc_c!(),
              6 => r * acc_b!() + r_sq * acc_c!(),
              _ => acc_a!() + r * acc_b!() + r_sq * acc_c!(),
            }
          } else {
            acc_a!() + r * acc_b!() + r_sq * acc_c!()
          };
        }
      });

      // Expand compact buffer → full-size output
      let mut result = vec![E::Scalar::ZERO; num_cols];
      for (dense_idx, &orig_col) in self.dense_to_col.iter().enumerate() {
        result[orig_col as usize] = compact[dense_idx];
      }
      return result;
    }

    // Fallback: CSR-based scatter (used when CSC tables aren't built)
    let num_rows = self.num_cons_unpadded;
    let has_unit_partition = !self.A_unit_end.is_empty();

    let buffer_bytes = num_dense * std::mem::size_of::<E::Scalar>();
    let max_threads = std::cmp::max(2, 512_000_000 / buffer_bytes);
    let num_threads = std::cmp::min(rayon::current_num_threads(), max_threads);
    let chunk_size = (num_rows + num_threads - 1) / num_threads;

    #[inline(always)]
    fn process_matrix_row<F: ff::PrimeField + MontgomeryLimbs, C: SmallCoeff>(
      buffer: &mut [F],
      rx_scaled: F,
      data: &[C],
      dense_col: &[u32],
      start: usize,
      unit_end: usize,
      end: usize,
      has_unit: bool,
    ) {
      if has_unit {
        for i in start..unit_end {
          let idx = dense_col[i] as usize;
          if data[i].is_positive() {
            buffer[idx] += rx_scaled;
          } else {
            buffer[idx] -= rx_scaled;
          }
        }
        for i in unit_end..end {
          let idx = dense_col[i] as usize;
          buffer[idx] += data[i].mul_field(&rx_scaled);
        }
      } else {
        for i in start..end {
          let idx = dense_col[i] as usize;
          buffer[idx] += data[i].mul_field(&rx_scaled);
        }
      }
    }

    let mut thread_buffers: Vec<Vec<E::Scalar>> = (0..num_threads)
      .into_par_iter()
      .map(|thread_idx| {
        let start_row = thread_idx * chunk_size;
        let end_row = ((thread_idx + 1) * chunk_size).min(num_rows);
        let mut buffer = vec![E::Scalar::ZERO; num_dense];

        for row_idx in start_row..end_row {
          let rx_row = rx[row_idx];
          let rx_r = rx_row * r;
          let rx_r_sq = rx_row * r_sq;

          let a_unit = if has_unit_partition { self.A_unit_end[row_idx] } else { self.A.indptr[row_idx] };
          let b_unit = if has_unit_partition { self.B_unit_end[row_idx] } else { self.B.indptr[row_idx] };
          let c_unit = if has_unit_partition { self.C_unit_end[row_idx] } else { self.C.indptr[row_idx] };

          process_matrix_row(
            &mut buffer, rx_row, &self.A.data, &self.A_dense_col,
            self.A.indptr[row_idx], a_unit, self.A.indptr[row_idx + 1], has_unit_partition,
          );
          process_matrix_row(
            &mut buffer, rx_r, &self.B.data, &self.B_dense_col,
            self.B.indptr[row_idx], b_unit, self.B.indptr[row_idx + 1], has_unit_partition,
          );
          process_matrix_row(
            &mut buffer, rx_r_sq, &self.C.data, &self.C_dense_col,
            self.C.indptr[row_idx], c_unit, self.C.indptr[row_idx + 1], has_unit_partition,
          );
        }
        buffer
      })
      .collect();

    let mut compact = thread_buffers.swap_remove(0);
    for buffer in thread_buffers {
      compact
        .par_iter_mut()
        .zip(buffer.par_iter())
        .for_each(|(a, b)| *a += *b);
    }

    let mut result = vec![E::Scalar::ZERO; num_cols];
    for (dense_idx, &orig_col) in self.dense_to_col.iter().enumerate() {
      result[orig_col as usize] = compact[dense_idx];
    }

    result
  }

  /// Fallback without column remapping (used when remap tables are not populated).
  fn bind_row_vars_combined_small_no_remap(
    &self,
    rx: &[E::Scalar],
    r: E::Scalar,
  ) -> Vec<E::Scalar>
  where
    E::Scalar: MontgomeryLimbs,
  {
    let num_vars = self.num_shared + self.num_precommitted + self.num_rest;
    let num_cols = 2 * num_vars;
    let r_sq = r * r;

    par_chunked_reduce(self.num_cons, num_cols, |buffer, row_idx| {
      let rx_row = rx[row_idx];
      let rx_r = rx_row * r;
      let rx_r_sq = rx_row * r_sq;

      let a_ptrs = [self.A.indptr[row_idx], self.A.indptr[row_idx + 1]];
      let b_ptrs = [self.B.indptr[row_idx], self.B.indptr[row_idx + 1]];
      let c_ptrs = [self.C.indptr[row_idx], self.C.indptr[row_idx + 1]];

      for (val, col) in self.A.get_row_unchecked(&a_ptrs) {
        buffer[*col] += val.mul_field(&rx_row);
      }
      for (val, col) in self.B.get_row_unchecked(&b_ptrs) {
        buffer[*col] += val.mul_field(&rx_r);
      }
      for (val, col) in self.C.get_row_unchecked(&c_ptrs) {
        buffer[*col] += val.mul_field(&rx_r_sq);
      }
    })
  }
}

/// A type that holds a multi-round split R1CS shape
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SplitMultiRoundR1CSShape<E: Engine, V = <E as Engine>::Scalar> {
  pub(crate) num_cons: usize,
  pub(crate) num_cons_unpadded: usize, // number of constraints before padding

  pub(crate) num_rounds: usize,
  pub(crate) num_vars_per_round_unpadded: Vec<usize>, // variables per round before padding
  pub(crate) num_vars_per_round: Vec<usize>,          // variables per round after padding
  pub(crate) num_challenges_per_round: Vec<usize>,    // challenges per round
  pub(crate) num_public: usize,                       // number of public variables

  pub(crate) A: SparseMatrix<V>,
  pub(crate) B: SparseMatrix<V>,
  pub(crate) C: SparseMatrix<V>,
  #[serde(skip, default = "OnceCell::new")]
  pub(crate) digest: OnceCell<E::Scalar>,
}

impl<E: Engine, V: Serialize> SimpleDigestible for SplitMultiRoundR1CSShape<E, V> {}

/// A type that holds a multi-round split R1CS instance
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SplitMultiRoundR1CSInstance<E: Engine> {
  pub(crate) comm_w_per_round: Vec<Commitment<E>>,
  pub(crate) public_values: Vec<E::Scalar>,
  pub(crate) challenges_per_round: Vec<Vec<E::Scalar>>,
}

impl<E: Engine> SplitR1CSInstance<E> {
  /// A method to create a split R1CS instance object using constituent elements
  pub fn new(
    S: &SplitR1CSShape<E>,
    comm_W_shared: Option<Commitment<E>>,
    comm_W_precommitted: Option<Commitment<E>>,
    comm_W_rest: Commitment<E>,
    public_values: Vec<E::Scalar>,
    challenges: Vec<E::Scalar>,
  ) -> Result<SplitR1CSInstance<E>, SpartanError> {
    if public_values.len() != S.num_public {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "SplitR1CS instance: Expected {} public values, got {}",
          S.num_public,
          public_values.len()
        ),
      });
    }
    if challenges.len() != S.num_challenges {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "SplitR1CS instance: Expected {} challenges, got {}",
          S.num_challenges,
          challenges.len()
        ),
      });
    }

    // check if the commitments commit to the right number of variables
    if S.num_shared > 0 && comm_W_shared.is_none() {
      return Err(SpartanError::InvalidCommitmentLength {
        reason: "comm_W_shared is missing".to_string(),
      });
    }
    if S.num_precommitted > 0 && comm_W_precommitted.is_none() {
      return Err(SpartanError::InvalidCommitmentLength {
        reason: "comm_W_precommitted is missing".to_string(),
      });
    }

    if let Some(ref comm) = comm_W_shared {
      E::PCS::check_commitment(comm, S.num_shared, DEFAULT_COMMITMENT_WIDTH)?;
    }
    if let Some(ref comm) = comm_W_precommitted {
      E::PCS::check_commitment(comm, S.num_precommitted, DEFAULT_COMMITMENT_WIDTH)?;
    }
    E::PCS::check_commitment(&comm_W_rest, S.num_rest, DEFAULT_COMMITMENT_WIDTH)?;

    Ok(SplitR1CSInstance {
      comm_W_shared,
      comm_W_precommitted,
      comm_W_rest,
      public_values,
      challenges,
    })
  }

  pub fn validate(
    &self,
    S: &SplitR1CSShape<E>,
    transcript: &mut E::TE,
  ) -> Result<(), SpartanError> {
    if S.num_shared > 0 {
      if let Some(comm) = &self.comm_W_shared {
        E::PCS::check_commitment(comm, S.num_shared, DEFAULT_COMMITMENT_WIDTH)?;
        transcript.absorb(b"comm_W_shared", comm);
      } else {
        return Err(SpartanError::ProofVerifyError {
          reason: "comm_W_shared is missing".to_string(),
        });
      }
    }

    if S.num_precommitted > 0 {
      if let Some(comm) = &self.comm_W_precommitted {
        E::PCS::check_commitment(comm, S.num_precommitted, DEFAULT_COMMITMENT_WIDTH)?;
        transcript.absorb(b"comm_W_precommitted", comm);
      } else {
        return Err(SpartanError::ProofVerifyError {
          reason: "comm_W_precommitted is missing".to_string(),
        });
      }
    }

    // obtain challenges from the transcript
    let challenges = (0..S.num_challenges)
      .map(|_| transcript.squeeze(b"challenge"))
      .collect::<Result<Vec<E::Scalar>, SpartanError>>()?;

    // check that the challenges of the circuit matches the expected values
    if challenges != self.challenges {
      return Err(SpartanError::ProofVerifyError {
        reason: "Challenges do not match".to_string(),
      });
    }

    E::PCS::check_commitment(&self.comm_W_rest, S.num_rest, DEFAULT_COMMITMENT_WIDTH)?;
    transcript.absorb(b"comm_W_rest", &self.comm_W_rest);

    Ok(())
  }

  pub fn to_regular_instance(&self) -> Result<R1CSInstance<E>, SpartanError> {
    let partial_comms = [
      self.comm_W_shared.clone(),
      self.comm_W_precommitted.clone(),
      Some(self.comm_W_rest.clone()),
    ]
    .iter()
    .filter_map(|comm| comm.clone())
    .collect::<Vec<Commitment<E>>>();
    let comm_W = PCS::<E>::combine_commitments(&partial_comms)?;

    Ok(R1CSInstance {
      comm_W,
      X: [self.public_values.clone(), self.challenges.clone()].concat(),
    })
  }
}

impl<E: Engine> SplitMultiRoundR1CSShape<E> {
  /// Create an object of type `SplitMultiRoundR1CSShape` from the explicitly specified R1CS matrices
  pub fn new(
    width: usize,
    num_cons: usize,
    num_vars_per_round: Vec<usize>,
    num_challenges_per_round: Vec<usize>,
    num_public: usize,
    A: SparseMatrix<E::Scalar>,
    B: SparseMatrix<E::Scalar>,
    C: SparseMatrix<E::Scalar>,
  ) -> Result<SplitMultiRoundR1CSShape<E>, SpartanError> {
    let num_rounds = num_vars_per_round.len();
    if num_challenges_per_round.len() != num_rounds {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "SplitMultiRoundR1CSShape: Expected {} challenges per round, got {}",
          num_rounds,
          num_challenges_per_round.len()
        ),
      });
    }

    let total_vars: usize = num_vars_per_round.iter().sum();
    let total_challenges: usize = num_challenges_per_round.iter().sum();
    let num_rows = num_cons;
    let num_cols = total_vars + 1 + num_public + total_challenges; // +1 for the constant term

    is_sparse_matrix_valid(num_rows, num_cols, &A)?;
    is_sparse_matrix_valid(num_rows, num_cols, &B)?;
    is_sparse_matrix_valid(num_rows, num_cols, &C)?;

    // Pad each round's variables to be a multiple of width
    let num_vars_per_round_padded: Vec<usize> = num_vars_per_round
      .iter()
      .map(|&n| pad_to_width(width, n))
      .collect();

    let total_vars_padded: usize = num_vars_per_round_padded.iter().sum();
    let num_cons_padded = num_cons.next_power_of_two();

    // Apply padding transformation to matrices
    let apply_pad = |mut m: SparseMatrix<E::Scalar>| -> SparseMatrix<E::Scalar> {
      m.indices.par_iter_mut().for_each(|c| {
        // Find which round this variable belongs to and apply appropriate offset
        let mut current_offset = 0;
        let mut current_padded_offset = 0;

        for round in 0..num_rounds {
          if *c >= current_offset && *c < current_offset + num_vars_per_round[round] {
            // Variable belongs to this round, apply the padded offset
            *c = current_padded_offset + (*c - current_offset);
            return;
          }
          current_offset += num_vars_per_round[round];
          current_padded_offset += num_vars_per_round_padded[round];
        }

        // If we get here, it's a public/challenge variable, apply total padding offset
        if *c >= total_vars {
          *c += total_vars_padded - total_vars;
        }
      });

      m.cols += total_vars_padded - total_vars;

      let ex = {
        let nnz = if m.indptr.is_empty() {
          0
        } else {
          m.indptr[m.indptr.len() - 1]
        };
        vec![nnz; num_cons_padded - num_cons]
      };
      m.indptr.extend(ex);
      m
    };

    let A_padded = apply_pad(A);
    let B_padded = apply_pad(B);
    let C_padded = apply_pad(C);

    Ok(Self {
      num_cons: num_cons_padded,
      num_cons_unpadded: num_cons,
      num_rounds,
      num_vars_per_round_unpadded: num_vars_per_round,
      num_vars_per_round: num_vars_per_round_padded,
      num_challenges_per_round,
      num_public,
      A: A_padded,
      B: B_padded,
      C: C_padded,
      digest: OnceCell::new(),
    })
  }

  pub fn to_regular_shape(&self) -> R1CSShape<E> {
    let total_vars: usize = self.num_vars_per_round.iter().sum();
    let total_challenges: usize = self.num_challenges_per_round.iter().sum();

    R1CSShape {
      num_cons: self.num_cons,
      num_vars: total_vars,
      num_io: total_challenges + self.num_public,
      A: self.A.clone(),
      B: self.B.clone(),
      C: self.C.clone(),
      digest: OnceCell::new(),
    }
  }

  /// Returns statistics about the shape of the multi-round R1CS matrices
  pub fn sizes(&self) -> (usize, Vec<usize>, Vec<usize>, Vec<usize>, usize) {
    (
      self.num_cons_unpadded,
      self.num_vars_per_round_unpadded.clone(),
      self.num_vars_per_round.clone(),
      self.num_challenges_per_round.clone(),
      self.num_public,
    )
  }

  /// Generates public parameters for a multi-round R1CS
  pub fn commitment_key(&self) -> (CommitmentKey<E>, VerifierKey<E>) {
    let total_vars: usize = self.num_vars_per_round.iter().sum();
    // Use a narrower commitment width for multi-round witnesses to reduce padding overhead.
    E::PCS::setup(b"ck", total_vars, MULTIROUND_COMMITMENT_WIDTH)
  }

  pub fn multiply_vec(
    &self,
    z: &[E::Scalar],
  ) -> Result<(Vec<E::Scalar>, Vec<E::Scalar>, Vec<E::Scalar>), SpartanError> {
    let total_vars: usize = self.num_vars_per_round.iter().sum();
    let total_challenges: usize = self.num_challenges_per_round.iter().sum();

    if z.len() != self.num_public + total_challenges + 1 + total_vars {
      return Err(SpartanError::InvalidWitnessLength);
    }

    let (az, (bz, cz)) = rayon::join(
      || self.A.multiply_vec(z),
      || rayon::join(|| self.B.multiply_vec(z), || self.C.multiply_vec(z)),
    );

    Ok((az?, bz?, cz?))
  }
}

impl<E: Engine> SplitMultiRoundR1CSInstance<E> {
  /// A method to create a multi-round split R1CS instance object using constituent elements
  pub fn new(
    s: &SplitMultiRoundR1CSShape<E>,
    comm_w_per_round: Vec<Commitment<E>>,
    public_values: Vec<E::Scalar>,
    challenges_per_round: Vec<Vec<E::Scalar>>,
  ) -> Result<SplitMultiRoundR1CSInstance<E>, SpartanError> {
    if public_values.len() != s.num_public {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "SplitMultiRoundR1CS instance: Expected {} public values, got {}",
          s.num_public,
          public_values.len()
        ),
      });
    }
    if challenges_per_round.len() != s.num_rounds {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "SplitMultiRoundR1CS instance: Expected {} rounds, got {}",
          s.num_rounds,
          challenges_per_round.len()
        ),
      });
    }
    if comm_w_per_round.len() != s.num_rounds {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "SplitMultiRoundR1CS instance: Expected {} rounds, got {}",
          s.num_rounds,
          comm_w_per_round.len()
        ),
      });
    }

    // Validate challenges per round
    for (round, challenges) in challenges_per_round.iter().enumerate() {
      if challenges.len() != s.num_challenges_per_round[round] {
        return Err(SpartanError::InvalidInputLength {
          reason: format!(
            "SplitMultiRoundR1CS instance: Expected {} challenges in round {}, got {}",
            s.num_challenges_per_round[round],
            round,
            challenges.len()
          ),
        });
      }
    }

    // Validate commitments per round
    for (round, comm) in comm_w_per_round.iter().enumerate() {
      E::PCS::check_commitment(
        comm,
        s.num_vars_per_round[round],
        MULTIROUND_COMMITMENT_WIDTH,
      )?;
    }

    Ok(SplitMultiRoundR1CSInstance {
      comm_w_per_round,
      public_values,
      challenges_per_round,
    })
  }

  pub fn validate(
    &self,
    s: &SplitMultiRoundR1CSShape<E>,
    transcript: &mut E::TE,
  ) -> Result<(), SpartanError> {
    // Process each round, absorbing the previous round's commitment before deriving this round's challenges
    for round in 0..s.num_rounds {
      E::PCS::check_commitment(
        &self.comm_w_per_round[round],
        s.num_vars_per_round[round],
        MULTIROUND_COMMITMENT_WIDTH,
      )?;
      transcript.absorb(b"comm_w_round", &self.comm_w_per_round[round]);

      let derived_challenges = (0..s.num_challenges_per_round[round])
        .map(|_| transcript.squeeze(b"challenge"))
        .collect::<Result<Vec<E::Scalar>, SpartanError>>()?;

      if self.challenges_per_round[round] != derived_challenges {
        return Err(SpartanError::ProofVerifyError {
          reason: format!("MultiRoundR1CSInstance:: Challenges do not match for round {round}"),
        });
      }
    }

    Ok(())
  }

  pub fn to_regular_instance(&self) -> Result<R1CSInstance<E>, SpartanError> {
    let partial_comms = self.comm_w_per_round.clone();
    let comm_w = PCS::<E>::combine_commitments(&partial_comms)?;

    let challenges: Vec<E::Scalar> = self
      .challenges_per_round
      .iter()
      .flatten()
      .cloned()
      .collect();

    Ok(R1CSInstance {
      comm_W: comm_w,
      // Multi-round circuits inputize challenges before public values during synthesis.
      // The regular instance must reflect the same ordering for satisfiability checks.
      X: [challenges, self.public_values.clone()].concat(),
    })
  }
}
