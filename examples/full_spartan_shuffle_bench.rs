//! Bellpepper benchmark for the March 25 permutation relation.
//!
//! Included:
//! - simple VRF key correctness and VRF output binding
//! - Poseidon-derived RS trace checks
//! - RS stable-partition checks for each level
//! - transcript-derived `x`, `tau`, and tuple-product challenges
//! - hidden graph relation `b_i = x^{pi(i)}`
//! - Horner linker `L = sum_i b_i * tau^i`
//! - in-circuit Pedersen-style check `C_link = [L]G + [r]H`
//!
//! Excluded:
//! - Groth16 / arkworks gadgets
//! - in-circuit opening proof for `C_power`
//!
//! Run with:
//!   cargo run --release --example full_spartan_shuffle_bench

use bellpepper::util_cs::metric_cs::MetricCS;
use bellpepper_core::{
  ConstraintSystem, SynthesisError,
  boolean::{AllocatedBit, Boolean},
  num::AllocatedNum,
  test_cs::TestConstraintSystem,
};
use clap::Parser;
use ff::{Field, PrimeField, PrimeFieldBits};
use halo2curves::{CurveAffine, group::prime::PrimeCurveAffine};
use num_bigint::BigUint;
use num_traits::{One as _, Zero as _};
use once_cell::sync::Lazy;
use rayon::join;
use std::time::{Duration, Instant};
use std::{collections::BTreeMap, fmt};

use spartan2::{
  bellpepper::{
    solver::SatisfyingAssignment,
    test_r1cs::{TestSpartanShape, TestSpartanWitness},
    test_shape_cs::TestShapeCS,
  },
  cli::SpartanBenchChoice,
  gadgets::{
    ecc::{AllocatedPoint, AllocatedPointNonInfinity},
    utils::{
      alloc_constant, alloc_num_equals, alloc_zero, conditionally_select, select_num_or_one,
      select_num_or_zero,
    },
  },
  provider::traits::DlogGroup,
  provider::traits::DlogGroupExt,
  provider::{PallasHyraxEngine, VestaHyraxEngine, pasta::pallas},
  rs_shuffle_bp::{
    data_structures::{
      ElGamalCiphertext as NativeCiphertext, PermutationWitnessTrace, PermutationWitnessTraceVar,
      SortedRow, SortedRowVar, UnsortedRowVar,
    },
    encryption::precompute_native_powers,
    native::build_level,
    permutation::{IndexPositionPair, check_grand_product},
  },
  spartan::SpartanSNARK,
  spartan_pp::PpSpartanSNARK,
  traits::{Engine, Group, circuit::SpartanCircuit, snark::R1CSSNARKTrait},
};

const N: usize = 52;
const LEVELS: usize = 6;

const POSEIDON_RATE: usize = 2;
const POSEIDON_CAPACITY: usize = 1;
const POSEIDON_WIDTH: usize = POSEIDON_RATE + POSEIDON_CAPACITY;
const POSEIDON_FULL_ROUNDS: usize = 8;
const POSEIDON_PARTIAL_ROUNDS: usize = 57;
const POSEIDON_ALPHA: u64 = 5;

type Scalar = pallas::Scalar;
type CurveEngine = VestaHyraxEngine;
type LinkerScalar = <CurveEngine as Engine>::Scalar;
type ShuffleSnark = SpartanSNARK<PallasHyraxEngine>;
type PpShuffleSnark = PpSpartanSNARK<PallasHyraxEngine>;
type CurvePoint = <CurveEngine as Engine>::GE;
type CurveAffinePoint = <<CurveEngine as Engine>::GE as DlogGroup>::AffineGroupElement;

const EMULATED_LIMB_BITS: usize = 32;
const EMULATED_LIMBS: usize = 8;
const EMULATED_TOTAL_BITS: usize = EMULATED_LIMBS * EMULATED_LIMB_BITS;
const EMULATED_LIMB_BASE_U64: u64 = 1u64 << EMULATED_LIMB_BITS;
const EMULATED_ADD_CARRY_BITS: usize = 2;
const EMULATED_ADD_CARRY_BIAS: i64 = 1;
const EMULATED_MUL_CARRY_BITS: usize = 36;
const EMULATED_MUL_CARRY_BIAS: i64 = 1i64 << (EMULATED_MUL_CARRY_BITS - 1);
const LINK_LINEAR_QUOTIENT_BITS: usize = EMULATED_TOTAL_BITS + 6;
const LINK_LINEAR_QUOTIENT_LIMBS: usize = LINK_LINEAR_QUOTIENT_BITS.div_ceil(EMULATED_LIMB_BITS);
const LINK_LINEAR_CARRY_BITS: usize = 48;
const LINK_LINEAR_CARRY_BIAS: i64 = 1i64 << (LINK_LINEAR_CARRY_BITS - 1);

#[derive(Parser, Debug)]
#[command(name = "full_spartan_shuffle_bench")]
struct Cli {
  /// Which Spartan implementation to benchmark
  #[arg(long, value_enum, default_value_t = SpartanBenchChoice::Both)]
  snark: SpartanBenchChoice,
}

#[derive(Clone, Copy)]
enum SpongeMode {
  Absorbing { next_absorb_index: usize },
  Squeezing { next_squeeze_index: usize },
}

#[derive(Clone)]
struct PoseidonConfigLocal {
  ark: Vec<[Scalar; POSEIDON_WIDTH]>,
  mds: [[Scalar; POSEIDON_WIDTH]; POSEIDON_WIDTH],
}

#[derive(Clone)]
struct PoseidonSpongeNative {
  state: [Scalar; POSEIDON_WIDTH],
  mode: SpongeMode,
}

struct PoseidonSpongeCircuit {
  state: [AllocatedNum<Scalar>; POSEIDON_WIDTH],
  mode: SpongeMode,
  prefix: &'static str,
  tag_counter: usize,
}

#[derive(Clone)]
struct CommitmentBases {
  generator: (Scalar, Scalar),
  generator_affine: CurveAffinePoint,
  generator_powers: Vec<(Scalar, Scalar)>,
  perm_bases_affine: Vec<CurveAffinePoint>,
  perm_blind_base_affine: CurveAffinePoint,
  power_bases_affine: Vec<CurveAffinePoint>,
  power_blind_base_affine: CurveAffinePoint,
  link_base: (Scalar, Scalar),
  link_base_affine: CurveAffinePoint,
  link_blind_base: (Scalar, Scalar),
  link_blind_base_affine: CurveAffinePoint,
  link_base_powers: Vec<(Scalar, Scalar)>,
  link_blind_powers: Vec<(Scalar, Scalar)>,
}

#[derive(Clone)]
struct LinkerChallenges {
  power_challenge: Scalar,
  power_challenge_scalar: LinkerScalar,
  tau_base: Scalar,
  tau_scalar: LinkerScalar,
  tuple_expected_product: Scalar,
}

#[derive(Clone)]
struct ShuffleStatement<const N_LOCAL: usize> {
  pk: (Scalar, Scalar),
  nonce: Scalar,
  seed_digest: Scalar,
  power_challenge: Scalar,
  tau_base: Scalar,
  permutation_commitment: (Scalar, Scalar),
  power_commitment: (Scalar, Scalar),
  link_commitment: (Scalar, Scalar),
  tuple_expected_product: Scalar,
  input_ciphertexts: [NativeCiphertext<CurveEngine>; N_LOCAL],
  output_ciphertexts: [NativeCiphertext<CurveEngine>; N_LOCAL],
}

#[derive(Clone)]
struct ShuffleWitness<const N_LOCAL: usize, const LEVELS_LOCAL: usize> {
  sk: LinkerScalar,
  witness_trace: PermutationWitnessTrace<N_LOCAL, LEVELS_LOCAL>,
  power_perm_vec: [LinkerScalar; N_LOCAL],
  power_challenge_scalar: LinkerScalar,
  tau_scalar: LinkerScalar,
  tau_powers: [LinkerScalar; N_LOCAL],
  perm_blinding: LinkerScalar,
  power_blinding: LinkerScalar,
  link_value: LinkerScalar,
  link_blinding: LinkerScalar,
  rerandomization_scalars: [LinkerScalar; N_LOCAL],
}

struct ShuffleProof<const N_LOCAL: usize> {
  spartan_proof: ShuffleSnark,
  sigma_proof: NativeReencryptionProof<N_LOCAL>,
}

#[derive(Debug)]
enum ShuffleVerifyError {
  InvalidPoint { label: String },
  InvalidCiphertext { label: String },
  InconsistentStatement { reason: String },
  SpartanVerify(spartan2::errors::SpartanError),
  SpartanPublicValuesMismatch,
  SigmaVerification { reason: String },
}

impl fmt::Display for ShuffleVerifyError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::InvalidPoint { label } => write!(f, "invalid curve point: {label}"),
      Self::InvalidCiphertext { label } => write!(f, "invalid ciphertext: {label}"),
      Self::InconsistentStatement { reason } => write!(f, "inconsistent statement: {reason}"),
      Self::SpartanVerify(err) => write!(f, "Spartan verification failed: {err}"),
      Self::SpartanPublicValuesMismatch => {
        write!(
          f,
          "Spartan proof returned public values for a different statement"
        )
      }
      Self::SigmaVerification { reason } => write!(f, "Sigma verification failed: {reason}"),
    }
  }
}

impl std::error::Error for ShuffleVerifyError {}

impl From<spartan2::errors::SpartanError> for ShuffleVerifyError {
  fn from(value: spartan2::errors::SpartanError) -> Self {
    Self::SpartanVerify(value)
  }
}

#[derive(Clone)]
struct FullRSShuffleCircuit {
  statement: ShuffleStatement<N>,
  witness: ShuffleWitness<N, LEVELS>,
  bases: CommitmentBases,
}

struct PoseidonGrainLfsr {
  prime_num_bits: usize,
  state: [bool; 80],
  head: usize,
}

#[derive(Clone)]
struct NativeReencryptionProof<const N_LOCAL: usize> {
  blinding_factor_commitment: (Scalar, Scalar),
  blinding_rerandomization_commitment: NativeCiphertext<CurveEngine>,
  sigma_response_power_permutation_vector: [LinkerScalar; N_LOCAL],
  sigma_response_blinding: LinkerScalar,
  sigma_response_rerand: LinkerScalar,
}

static POSEIDON_CONFIG: Lazy<PoseidonConfigLocal> = Lazy::new(build_poseidon_config);
static SCALAR_MODULUS: Lazy<BigUint> = Lazy::new(scalar_modulus);
static LINKER_SCALAR_MODULUS: Lazy<BigUint> = Lazy::new(linker_scalar_modulus);
static LINKER_SCALAR_MODULUS_BITS: Lazy<Vec<bool>> = Lazy::new(linker_scalar_modulus_bits);
static LINKER_SCALAR_MODULUS_LIMBS: Lazy<[u64; EMULATED_LIMBS]> =
  Lazy::new(linker_scalar_modulus_limbs);
static COMMITMENT_BASES: Lazy<CommitmentBases> = Lazy::new(build_commitment_bases);

struct BaselineBenchSummary {
  constraints: usize,
  setup_time: Duration,
  witness_generation_time: Duration,
  equivalence_check_time: Duration,
  prep_time: Duration,
  prove_time: Duration,
  sigma_prove_time: Duration,
  combined_prove_time: Duration,
  verify_time: Duration,
  sigma_verify_time: Duration,
  combined_verify_time: Duration,
  proof_size: usize,
}

struct PpBenchSummary {
  constraints: usize,
  setup_time: Duration,
  prep_time: Duration,
  prove_time: Duration,
  verify_time: Duration,
  proof_size: usize,
}

fn scalar_modulus() -> BigUint {
  let mut value = BigUint::zero();
  let mut power = BigUint::one();
  for bit in Scalar::char_le_bits() {
    if bit {
      value += &power;
    }
    power <<= 1usize;
  }
  value
}

fn linker_scalar_modulus() -> BigUint {
  let mut value = BigUint::zero();
  let mut power = BigUint::one();
  for bit in LinkerScalar::char_le_bits() {
    if bit {
      value += &power;
    }
    power <<= 1usize;
  }
  value
}

fn linker_scalar_modulus_bits() -> Vec<bool> {
  let mut bits: Vec<bool> = LinkerScalar::char_le_bits().into_iter().collect();
  bits.resize(EMULATED_TOTAL_BITS, false);
  bits
}

fn linker_scalar_modulus_limbs() -> [u64; EMULATED_LIMBS] {
  bits_to_u32_limbs(&LINKER_SCALAR_MODULUS_BITS)
}

fn scalar_from_biguint_checked(value: &BigUint) -> Scalar {
  let bytes = value.to_bytes_le();
  let mut repr = <Scalar as PrimeField>::Repr::default();
  let repr_bytes = repr.as_mut();
  assert!(
    bytes.len() <= repr_bytes.len(),
    "value does not fit in field representation"
  );
  repr_bytes[..bytes.len()].copy_from_slice(&bytes);
  Option::<Scalar>::from(Scalar::from_repr(repr)).expect("canonical field element")
}

fn scalar_from_le_bytes_mod_order(bytes: &[u8]) -> Scalar {
  let value = BigUint::from_bytes_le(bytes) % &*SCALAR_MODULUS;
  scalar_from_biguint_checked(&value)
}

fn scalar_from_bits_le_checked(bits_le: &[bool]) -> Option<Scalar> {
  let num_bytes = bits_le.len().div_ceil(8);
  let mut bytes = vec![0u8; num_bytes];
  for (i, bit) in bits_le.iter().enumerate() {
    if *bit {
      bytes[i / 8] |= 1u8 << (i % 8);
    }
  }

  let mut repr = <Scalar as PrimeField>::Repr::default();
  let repr_bytes = repr.as_mut();
  if bytes.len() > repr_bytes.len() {
    return None;
  }
  repr_bytes[..bytes.len()].copy_from_slice(&bytes);
  Option::<Scalar>::from(Scalar::from_repr(repr))
}

fn scalar_to_bits_le(value: Scalar) -> Vec<bool> {
  value.to_le_bits().into_iter().collect()
}

fn linker_scalar_to_bits_le(value: LinkerScalar) -> Vec<bool> {
  let mut bits: Vec<bool> = value.to_le_bits().into_iter().collect();
  bits.resize(EMULATED_TOTAL_BITS, false);
  bits
}

fn bits_to_le_bytes(bits: &[bool]) -> Vec<u8> {
  let num_bytes = bits.len().div_ceil(8);
  let mut bytes = vec![0u8; num_bytes];
  for (i, bit) in bits.iter().enumerate() {
    if *bit {
      bytes[i / 8] |= 1u8 << (i % 8);
    }
  }
  bytes
}

fn scalar_to_le_bytes(value: Scalar) -> Vec<u8> {
  bits_to_le_bytes(&scalar_to_bits_le(value))
}

fn linker_scalar_to_le_bytes(value: LinkerScalar) -> Vec<u8> {
  bits_to_le_bytes(&linker_scalar_to_bits_le(value))
}

fn linker_scalar_from_biguint_checked(value: &BigUint) -> LinkerScalar {
  let bytes = value.to_bytes_le();
  let mut repr = <LinkerScalar as PrimeField>::Repr::default();
  let repr_bytes = repr.as_mut();
  assert!(
    bytes.len() <= repr_bytes.len(),
    "value does not fit in linker field representation"
  );
  repr_bytes[..bytes.len()].copy_from_slice(&bytes);
  Option::<LinkerScalar>::from(LinkerScalar::from_repr(repr))
    .expect("canonical linker field element")
}

fn linker_scalar_from_le_bytes_mod_order(bytes: &[u8]) -> LinkerScalar {
  let value = BigUint::from_bytes_le(bytes) % &*LINKER_SCALAR_MODULUS;
  linker_scalar_from_biguint_checked(&value)
}

fn bits_to_u32_limbs(bits: &[bool]) -> [u64; EMULATED_LIMBS] {
  std::array::from_fn(|limb_idx| {
    let start = limb_idx * EMULATED_LIMB_BITS;
    let end = core::cmp::min(start + EMULATED_LIMB_BITS, bits.len());
    let mut limb = 0u64;
    for (offset, bit) in bits[start..end].iter().enumerate() {
      if *bit {
        limb |= 1u64 << offset;
      }
    }
    limb
  })
}

fn biguint_to_u32_limbs<const LIMBS: usize>(value: &BigUint) -> [u64; LIMBS] {
  let bytes = value.to_bytes_le();
  std::array::from_fn(|limb_idx| {
    let start = limb_idx * (EMULATED_LIMB_BITS / 8);
    let end = core::cmp::min(start + (EMULATED_LIMB_BITS / 8), bytes.len());
    let mut limb = 0u64;
    for (offset, byte) in bytes[start..end].iter().enumerate() {
      limb |= (*byte as u64) << (8 * offset);
    }
    limb
  })
}

fn linker_scalar_to_limbs(value: LinkerScalar) -> [u64; EMULATED_LIMBS] {
  bits_to_u32_limbs(&linker_scalar_to_bits_le(value))
}

fn poseidon_usable_bytes() -> usize {
  ((Scalar::NUM_BITS - 1) as usize) / 8
}

fn pack_bytes_to_fields(bytes: &[u8]) -> Vec<Scalar> {
  let mut prefixed = Vec::with_capacity(8 + bytes.len());
  prefixed.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
  prefixed.extend_from_slice(bytes);

  let chunk_len = poseidon_usable_bytes();
  prefixed
    .chunks(chunk_len)
    .map(scalar_from_le_bytes_mod_order)
    .collect()
}

impl PoseidonGrainLfsr {
  fn new(
    is_sbox_an_inverse: bool,
    prime_num_bits: usize,
    state_len: usize,
    num_full_rounds: usize,
    num_partial_rounds: usize,
  ) -> Self {
    let mut state = [false; 80];

    state[1] = true;
    state[5] = is_sbox_an_inverse;

    let mut put_bits = |range: std::ops::RangeInclusive<usize>, mut value: usize| {
      for idx in range.rev() {
        state[idx] = value & 1 == 1;
        value >>= 1;
      }
    };

    put_bits(6..=17, prime_num_bits);
    put_bits(18..=29, state_len);
    put_bits(30..=39, num_full_rounds);
    put_bits(40..=49, num_partial_rounds);

    for bit in state.iter_mut().skip(50) {
      *bit = true;
    }

    let mut lfsr = Self {
      prime_num_bits,
      state,
      head: 0,
    };
    lfsr.init();
    lfsr
  }

  fn init(&mut self) {
    for _ in 0..160 {
      let _ = self.update();
    }
  }

  fn update(&mut self) -> bool {
    let new_bit = self.state[(self.head + 62) % 80]
      ^ self.state[(self.head + 51) % 80]
      ^ self.state[(self.head + 38) % 80]
      ^ self.state[(self.head + 23) % 80]
      ^ self.state[(self.head + 13) % 80]
      ^ self.state[self.head];
    self.state[self.head] = new_bit;
    self.head = (self.head + 1) % 80;
    new_bit
  }

  fn get_bits(&mut self, num_bits: usize) -> Vec<bool> {
    let mut bits = Vec::with_capacity(num_bits);
    for _ in 0..num_bits {
      let mut new_bit = self.update();
      while !new_bit {
        let _ = self.update();
        new_bit = self.update();
      }
      bits.push(self.update());
    }
    bits
  }

  fn get_field_elements_rejection_sampling(&mut self, num_elems: usize) -> Vec<Scalar> {
    let mut out = Vec::with_capacity(num_elems);
    for _ in 0..num_elems {
      loop {
        let mut bits = self.get_bits(self.prime_num_bits);
        bits.reverse();
        if let Some(value) = scalar_from_bits_le_checked(&bits) {
          out.push(value);
          break;
        }
      }
    }
    out
  }

  fn get_field_elements_mod_p(&mut self, num_elems: usize) -> Vec<Scalar> {
    let mut out = Vec::with_capacity(num_elems);
    for _ in 0..num_elems {
      let mut bits = self.get_bits(self.prime_num_bits);
      bits.reverse();
      let num_bytes = bits.len().div_ceil(8);
      let mut bytes = vec![0u8; num_bytes];
      for (i, bit) in bits.iter().enumerate() {
        if *bit {
          bytes[i / 8] |= 1u8 << (i % 8);
        }
      }
      out.push(scalar_from_le_bytes_mod_order(&bytes));
    }
    out
  }
}

fn build_poseidon_config() -> PoseidonConfigLocal {
  let mut lfsr = PoseidonGrainLfsr::new(
    false,
    Scalar::NUM_BITS as usize,
    POSEIDON_WIDTH,
    POSEIDON_FULL_ROUNDS,
    POSEIDON_PARTIAL_ROUNDS,
  );

  let ark = (0..(POSEIDON_FULL_ROUNDS + POSEIDON_PARTIAL_ROUNDS))
    .map(|_| {
      let elems = lfsr.get_field_elements_rejection_sampling(POSEIDON_WIDTH);
      elems.try_into().expect("poseidon ark row")
    })
    .collect();

  let xs = lfsr.get_field_elements_mod_p(POSEIDON_WIDTH);
  let ys = lfsr.get_field_elements_mod_p(POSEIDON_WIDTH);
  let mut mds = [[Scalar::ZERO; POSEIDON_WIDTH]; POSEIDON_WIDTH];
  for i in 0..POSEIDON_WIDTH {
    for j in 0..POSEIDON_WIDTH {
      mds[i][j] = (xs[i] + ys[j]).invert().expect("mds denominator");
    }
  }

  PoseidonConfigLocal { ark, mds }
}

impl PoseidonSpongeNative {
  fn new() -> Self {
    Self {
      state: [Scalar::ZERO; POSEIDON_WIDTH],
      mode: SpongeMode::Absorbing {
        next_absorb_index: 0,
      },
    }
  }

  fn permute(&mut self) {
    let cfg = &*POSEIDON_CONFIG;
    let full_rounds_over_2 = POSEIDON_FULL_ROUNDS / 2;

    for round in 0..full_rounds_over_2 {
      self.apply_ark(round);
      self.apply_sbox(true);
      self.apply_mds(cfg);
    }

    for round in full_rounds_over_2..(full_rounds_over_2 + POSEIDON_PARTIAL_ROUNDS) {
      self.apply_ark(round);
      self.apply_sbox(false);
      self.apply_mds(cfg);
    }

    for round in (full_rounds_over_2 + POSEIDON_PARTIAL_ROUNDS)
      ..(POSEIDON_FULL_ROUNDS + POSEIDON_PARTIAL_ROUNDS)
    {
      self.apply_ark(round);
      self.apply_sbox(true);
      self.apply_mds(cfg);
    }
  }

  fn apply_ark(&mut self, round: usize) {
    for i in 0..POSEIDON_WIDTH {
      self.state[i] += POSEIDON_CONFIG.ark[round][i];
    }
  }

  fn apply_sbox(&mut self, full_round: bool) {
    if full_round {
      for elem in &mut self.state {
        *elem = elem.pow_vartime([POSEIDON_ALPHA]);
      }
    } else {
      self.state[0] = self.state[0].pow_vartime([POSEIDON_ALPHA]);
    }
  }

  fn apply_mds(&mut self, cfg: &PoseidonConfigLocal) {
    let old = self.state;
    for row in 0..POSEIDON_WIDTH {
      let mut acc = Scalar::ZERO;
      for (col, value) in old.iter().enumerate() {
        acc += cfg.mds[row][col] * value;
      }
      self.state[row] = acc;
    }
  }

  fn absorb_internal(&mut self, mut rate_start_index: usize, elements: &[Scalar]) {
    let mut remaining = elements;

    loop {
      if rate_start_index + remaining.len() <= POSEIDON_RATE {
        for (i, element) in remaining.iter().enumerate() {
          self.state[POSEIDON_CAPACITY + rate_start_index + i] += element;
        }
        self.mode = SpongeMode::Absorbing {
          next_absorb_index: rate_start_index + remaining.len(),
        };
        return;
      }

      let num_absorbed = POSEIDON_RATE - rate_start_index;
      for (i, element) in remaining.iter().take(num_absorbed).enumerate() {
        self.state[POSEIDON_CAPACITY + rate_start_index + i] += element;
      }
      self.permute();
      remaining = &remaining[num_absorbed..];
      rate_start_index = 0;
    }
  }

  fn absorb_fields(&mut self, elements: &[Scalar]) {
    if elements.is_empty() {
      return;
    }

    match self.mode {
      SpongeMode::Absorbing { next_absorb_index } => {
        let mut absorb_index = next_absorb_index;
        if absorb_index == POSEIDON_RATE {
          self.permute();
          absorb_index = 0;
        }
        self.absorb_internal(absorb_index, elements);
      }
      SpongeMode::Squeezing { .. } => self.absorb_internal(0, elements),
    }
  }

  fn absorb_field(&mut self, value: Scalar) {
    self.absorb_fields(&[value]);
  }

  fn absorb_bytes(&mut self, bytes: &[u8]) {
    let packed = pack_bytes_to_fields(bytes);
    self.absorb_fields(&packed);
  }

  fn absorb_point(&mut self, point: (Scalar, Scalar)) {
    self.absorb_fields(&[point.0, point.1, Scalar::ZERO]);
  }

  fn squeeze_field_elements(&mut self, num_elements: usize) -> Vec<Scalar> {
    let mut out = vec![Scalar::ZERO; num_elements];

    match self.mode {
      SpongeMode::Absorbing { .. } => {
        self.permute();
        self.squeeze_internal(0, &mut out);
      }
      SpongeMode::Squeezing { next_squeeze_index } => {
        let mut squeeze_index = next_squeeze_index;
        if squeeze_index == POSEIDON_RATE {
          self.permute();
          squeeze_index = 0;
        }
        self.squeeze_internal(squeeze_index, &mut out);
      }
    }

    out
  }

  fn squeeze_internal(&mut self, mut rate_start_index: usize, output: &mut [Scalar]) {
    let mut output_remaining = output;
    loop {
      if rate_start_index + output_remaining.len() <= POSEIDON_RATE {
        output_remaining.clone_from_slice(
          &self.state[POSEIDON_CAPACITY + rate_start_index
            ..POSEIDON_CAPACITY + rate_start_index + output_remaining.len()],
        );
        self.mode = SpongeMode::Squeezing {
          next_squeeze_index: rate_start_index + output_remaining.len(),
        };
        return;
      }

      let num_squeezed = POSEIDON_RATE - rate_start_index;
      output_remaining[..num_squeezed].clone_from_slice(
        &self.state[POSEIDON_CAPACITY + rate_start_index
          ..POSEIDON_CAPACITY + rate_start_index + num_squeezed],
      );
      output_remaining = &mut output_remaining[num_squeezed..];
      if !output_remaining.is_empty() {
        self.permute();
      }
      rate_start_index = 0;
    }
  }
}

fn add_nums<F, CS>(
  mut cs: CS,
  a: &AllocatedNum<F>,
  b: &AllocatedNum<F>,
) -> Result<AllocatedNum<F>, SynthesisError>
where
  F: PrimeField,
  CS: ConstraintSystem<F>,
{
  let c = AllocatedNum::alloc(cs.namespace(|| "sum"), || {
    let a_val = a.get_value().ok_or(SynthesisError::AssignmentMissing)?;
    let b_val = b.get_value().ok_or(SynthesisError::AssignmentMissing)?;
    Ok(a_val + b_val)
  })?;
  cs.enforce(
    || "a + b = c",
    |lc| lc + a.get_variable() + b.get_variable(),
    |lc| lc + CS::one(),
    |lc| lc + c.get_variable(),
  );
  Ok(c)
}

fn add_constant_num<F, CS>(
  mut cs: CS,
  a: &AllocatedNum<F>,
  constant: F,
) -> Result<AllocatedNum<F>, SynthesisError>
where
  F: PrimeField,
  CS: ConstraintSystem<F>,
{
  let c = AllocatedNum::alloc(cs.namespace(|| "sum_const"), || {
    let a_val = a.get_value().ok_or(SynthesisError::AssignmentMissing)?;
    Ok(a_val + constant)
  })?;
  cs.enforce(
    || "a + const = c",
    |lc| lc + a.get_variable() + (constant, CS::one()),
    |lc| lc + CS::one(),
    |lc| lc + c.get_variable(),
  );
  Ok(c)
}

fn sub_nums<F, CS>(
  mut cs: CS,
  a: &AllocatedNum<F>,
  b: &AllocatedNum<F>,
) -> Result<AllocatedNum<F>, SynthesisError>
where
  F: PrimeField,
  CS: ConstraintSystem<F>,
{
  let c = AllocatedNum::alloc(cs.namespace(|| "difference"), || {
    let a_val = a.get_value().ok_or(SynthesisError::AssignmentMissing)?;
    let b_val = b.get_value().ok_or(SynthesisError::AssignmentMissing)?;
    Ok(a_val - b_val)
  })?;
  cs.enforce(
    || "a - b = c",
    |lc| lc + a.get_variable() - b.get_variable(),
    |lc| lc + CS::one(),
    |lc| lc + c.get_variable(),
  );
  Ok(c)
}

fn linear_combination_num<F, CS>(
  mut cs: CS,
  terms: &[(F, &AllocatedNum<F>)],
  constant: F,
) -> Result<AllocatedNum<F>, SynthesisError>
where
  F: PrimeField,
  CS: ConstraintSystem<F>,
{
  let out = AllocatedNum::alloc(cs.namespace(|| "linear_combination_alloc"), || {
    let mut acc = constant;
    for (coeff, value) in terms {
      let value = value.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      acc += *coeff * value;
    }
    Ok(acc)
  })?;
  cs.enforce(
    || "linear_combination",
    |lc| {
      let mut lc = lc + (constant, CS::one());
      for (coeff, value) in terms {
        lc = lc + (*coeff, value.get_variable());
      }
      lc
    },
    |lc| lc + CS::one(),
    |lc| lc + out.get_variable(),
  );
  Ok(out)
}

fn enforce_equal_num<F, CS>(
  mut cs: CS,
  a: &AllocatedNum<F>,
  b: &AllocatedNum<F>,
) -> Result<(), SynthesisError>
where
  F: PrimeField,
  CS: ConstraintSystem<F>,
{
  cs.enforce(
    || "a == b",
    |lc| lc + a.get_variable() - b.get_variable(),
    |lc| lc + CS::one(),
    |lc| lc,
  );
  Ok(())
}

fn alloc_point_constant<E: Engine, CS: ConstraintSystem<E::Base>>(
  mut cs: CS,
  coords: (E::Base, E::Base),
) -> Result<AllocatedPointNonInfinity<E>, SynthesisError> {
  Ok(AllocatedPointNonInfinity {
    x: alloc_constant(cs.namespace(|| "x"), &coords.0)?,
    y: alloc_constant(cs.namespace(|| "y"), &coords.1)?,
  })
}

fn alloc_point_public_input<E: Engine, CS: ConstraintSystem<E::Base>>(
  mut cs: CS,
  coords: (E::Base, E::Base),
) -> Result<AllocatedPointNonInfinity<E>, SynthesisError> {
  Ok(AllocatedPointNonInfinity {
    x: AllocatedNum::alloc_input(cs.namespace(|| "x"), || Ok(coords.0))?,
    y: AllocatedNum::alloc_input(cs.namespace(|| "y"), || Ok(coords.1))?,
  })
}

fn byte_nums_from_bits<CS: ConstraintSystem<Scalar>>(
  mut cs: CS,
  bits: &[AllocatedBit],
) -> Result<Vec<AllocatedNum<Scalar>>, SynthesisError> {
  let num_bytes = bits.len().div_ceil(8);
  let mut out = Vec::with_capacity(num_bytes);
  for byte_idx in 0..num_bytes {
    let start = byte_idx * 8;
    let end = core::cmp::min(start + 8, bits.len());
    let chunk = &bits[start..end];
    let byte_num = AllocatedNum::alloc(cs.namespace(|| format!("byte_{byte_idx}")), || {
      let mut acc = Scalar::ZERO;
      let mut coeff = Scalar::ONE;
      for bit in chunk {
        if bit.get_value().ok_or(SynthesisError::AssignmentMissing)? {
          acc += coeff;
        }
        coeff = coeff.double();
      }
      Ok(acc)
    })?;
    cs.enforce(
      || format!("byte_linear_combo_{byte_idx}"),
      |mut lc| {
        let mut coeff = Scalar::ONE;
        for bit in chunk {
          lc = lc + (coeff, bit.get_variable());
          coeff = coeff.double();
        }
        lc
      },
      |lc| lc + CS::one(),
      |lc| lc + byte_num.get_variable(),
    );
    out.push(byte_num);
  }
  Ok(out)
}

fn pack_boolean_bits_to_num<CS: ConstraintSystem<Scalar>>(
  mut cs: CS,
  bits: &[Boolean],
) -> Result<AllocatedNum<Scalar>, SynthesisError> {
  let out = AllocatedNum::alloc(cs.namespace(|| "packed_num"), || {
    let mut acc = Scalar::ZERO;
    let mut coeff = Scalar::ONE;
    for bit in bits {
      if bit.get_value().ok_or(SynthesisError::AssignmentMissing)? {
        acc += coeff;
      }
      coeff = coeff.double();
    }
    Ok(acc)
  })?;
  cs.enforce(
    || "pack_boolean_bits",
    |lc| {
      let mut lc = lc;
      let mut coeff = Scalar::ONE;
      for bit in bits {
        lc = lc + &bit.lc(CS::one(), coeff);
        coeff = coeff.double();
      }
      lc
    },
    |lc| lc + CS::one(),
    |lc| lc + out.get_variable(),
  );
  Ok(out)
}

fn alloc_bounded_num<CS: ConstraintSystem<Scalar>>(
  mut cs: CS,
  value: Option<u64>,
  num_bits: usize,
) -> Result<(AllocatedNum<Scalar>, Vec<AllocatedBit>), SynthesisError> {
  let bit_values: Vec<Option<bool>> = if let Some(value) = value {
    (0..num_bits)
      .map(|bit| Some(((value >> bit) & 1) == 1))
      .collect()
  } else {
    vec![None; num_bits]
  };

  let mut bits = Vec::with_capacity(num_bits);
  for (bit_idx, bit) in bit_values.iter().enumerate() {
    bits.push(AllocatedBit::alloc(
      cs.namespace(|| format!("bit_{bit_idx}")),
      *bit,
    )?);
  }

  let packed = pack_boolean_bits_to_num(
    cs.namespace(|| "pack_bounded_num"),
    &bits.iter().cloned().map(Boolean::from).collect::<Vec<_>>(),
  )?;
  Ok((packed, bits))
}

fn enforce_less_than_constant_bits<CS: ConstraintSystem<Scalar>>(
  mut cs: CS,
  bits: &[AllocatedBit],
  constant_bits: &[bool],
) -> Result<(), SynthesisError> {
  let mut less = Boolean::constant(false);
  let mut equal = Boolean::constant(true);

  for bit_idx in (0..constant_bits.len()).rev() {
    let bit = Boolean::from(bits[bit_idx].clone());
    if constant_bits[bit_idx] {
      let eq_and_not_bit = Boolean::and(
        cs.namespace(|| format!("eq_and_not_bit_{bit_idx}")),
        &equal,
        &bit.not(),
      )?;
      less = Boolean::or(
        cs.namespace(|| format!("less_update_{bit_idx}")),
        &less,
        &eq_and_not_bit,
      )?;
      equal = Boolean::and(
        cs.namespace(|| format!("equal_update_{bit_idx}")),
        &equal,
        &bit,
      )?;
    } else {
      equal = Boolean::and(
        cs.namespace(|| format!("equal_zero_update_{bit_idx}")),
        &equal,
        &bit.not(),
      )?;
    }
  }

  Boolean::enforce_equal(
    cs.namespace(|| "lt_modulus"),
    &less,
    &Boolean::constant(true),
  )
}

fn linker_scalar_to_biguint(value: LinkerScalar) -> BigUint {
  BigUint::from_bytes_le(&linker_scalar_to_le_bytes(value))
}

fn scalar_to_biguint(value: Scalar) -> BigUint {
  BigUint::from_bytes_le(&scalar_to_le_bytes(value))
}

#[derive(Clone)]
struct EmulatedLinkerVar {
  limbs: Vec<AllocatedNum<Scalar>>,
  bits: Vec<AllocatedBit>,
  value: Option<LinkerScalar>,
}

impl EmulatedLinkerVar {
  fn alloc_witness_with_mode<CS: ConstraintSystem<Scalar>>(
    mut cs: CS,
    value: Option<LinkerScalar>,
    enforce_canonical: bool,
  ) -> Result<Self, SynthesisError> {
    let bit_values: Vec<Option<bool>> = if let Some(value) = value {
      linker_scalar_to_bits_le(value)
        .into_iter()
        .map(Some)
        .collect()
    } else {
      vec![None; EMULATED_TOTAL_BITS]
    };

    let mut bits = Vec::with_capacity(EMULATED_TOTAL_BITS);
    for (bit_idx, bit) in bit_values.iter().enumerate() {
      bits.push(AllocatedBit::alloc(
        cs.namespace(|| format!("bit_{bit_idx}")),
        *bit,
      )?);
    }

    let limbs = bits
      .chunks(EMULATED_LIMB_BITS)
      .enumerate()
      .map(|(limb_idx, chunk)| {
        pack_boolean_bits_to_num(
          cs.namespace(|| format!("pack_limb_{limb_idx}")),
          &chunk.iter().cloned().map(Boolean::from).collect::<Vec<_>>(),
        )
      })
      .collect::<Result<Vec<_>, _>>()?;

    if enforce_canonical {
      enforce_less_than_constant_bits(
        cs.namespace(|| "lt_linker_modulus"),
        &bits,
        &LINKER_SCALAR_MODULUS_BITS,
      )?;
    }

    Ok(Self { limbs, bits, value })
  }

  fn alloc_witness<CS: ConstraintSystem<Scalar>>(
    cs: CS,
    value: Option<LinkerScalar>,
  ) -> Result<Self, SynthesisError> {
    Self::alloc_witness_with_mode(cs, value, true)
  }

  fn alloc_noncanonical_witness<CS: ConstraintSystem<Scalar>>(
    cs: CS,
    value: Option<LinkerScalar>,
  ) -> Result<Self, SynthesisError> {
    Self::alloc_witness_with_mode(cs, value, false)
  }

  fn zero<CS: ConstraintSystem<Scalar>>(cs: CS) -> Result<Self, SynthesisError> {
    Self::alloc_witness(cs, Some(LinkerScalar::ZERO))
  }

  fn enforce_equal<CS: ConstraintSystem<Scalar>>(
    &self,
    mut cs: CS,
    other: &Self,
  ) -> Result<(), SynthesisError> {
    for (limb_idx, (lhs, rhs)) in self.limbs.iter().zip(other.limbs.iter()).enumerate() {
      enforce_equal_num(cs.namespace(|| format!("limb_{limb_idx}")), lhs, rhs)?;
    }
    Ok(())
  }

  fn byte_fields<CS: ConstraintSystem<Scalar>>(
    &self,
    cs: CS,
  ) -> Result<Vec<AllocatedNum<Scalar>>, SynthesisError> {
    byte_nums_from_bits(cs, &self.bits)
  }

  fn add_mod<CS: ConstraintSystem<Scalar>>(
    &self,
    mut cs: CS,
    other: &Self,
  ) -> Result<Self, SynthesisError> {
    let lhs_value = self.value.ok_or(SynthesisError::AssignmentMissing)?;
    let rhs_value = other.value.ok_or(SynthesisError::AssignmentMissing)?;
    let lhs_big = linker_scalar_to_biguint(lhs_value);
    let rhs_big = linker_scalar_to_biguint(rhs_value);
    let sum_big = &lhs_big + &rhs_big;
    let reduce = sum_big >= *LINKER_SCALAR_MODULUS;
    let result_value = lhs_value + rhs_value;
    let result = Self::alloc_witness(cs.namespace(|| "result"), Some(result_value))?;
    let lhs_limbs = linker_scalar_to_limbs(lhs_value);
    let rhs_limbs = linker_scalar_to_limbs(rhs_value);
    let result_limbs = linker_scalar_to_limbs(result_value);

    let mut carry_values = [0i64; EMULATED_LIMBS - 1];
    let mut carry = 0i64;
    for limb_idx in 0..EMULATED_LIMBS {
      let lhs = lhs_limbs[limb_idx] as i64 + rhs_limbs[limb_idx] as i64 + carry;
      let rhs = result_limbs[limb_idx] as i64
        + if reduce {
          LINKER_SCALAR_MODULUS_LIMBS[limb_idx] as i64
        } else {
          0
        };
      let diff = lhs - rhs;
      debug_assert_eq!(diff % EMULATED_LIMB_BASE_U64 as i64, 0);
      carry = diff / EMULATED_LIMB_BASE_U64 as i64;
      if limb_idx + 1 < EMULATED_LIMBS {
        carry_values[limb_idx] = carry;
      } else {
        debug_assert_eq!(carry, 0, "final addition carry must vanish");
      }
    }

    let reduce_lc = |coeff: u64, bit: &AllocatedBit| {
      Boolean::from(bit.clone()).lc(CS::one(), Scalar::from(coeff))
    };
    let signed_offset =
      Scalar::from(EMULATED_ADD_CARRY_BIAS as u64) * Scalar::from(EMULATED_LIMB_BASE_U64 - 1);
    let reduce_bit = AllocatedBit::alloc(cs.namespace(|| "reduce_bit"), Some(reduce))?;
    let mut carry_in = alloc_constant(
      cs.namespace(|| "carry_init"),
      &Scalar::from(EMULATED_ADD_CARRY_BIAS as u64),
    )?;

    for limb_idx in 0..EMULATED_LIMBS {
      let carry_out = if limb_idx + 1 < EMULATED_LIMBS {
        alloc_bounded_num(
          cs.namespace(|| format!("carry_{limb_idx}")),
          Some((carry_values[limb_idx] + EMULATED_ADD_CARRY_BIAS) as u64),
          EMULATED_ADD_CARRY_BITS,
        )?
        .0
      } else {
        alloc_constant(
          cs.namespace(|| "carry_final"),
          &Scalar::from(EMULATED_ADD_CARRY_BIAS as u64),
        )?
      };

      cs.enforce(
        || format!("add_mod_limb_{limb_idx}"),
        |lc| {
          lc + (signed_offset, CS::one())
            + carry_in.get_variable()
            + self.limbs[limb_idx].get_variable()
            + other.limbs[limb_idx].get_variable()
        },
        |lc| lc + CS::one(),
        |lc| {
          lc + result.limbs[limb_idx].get_variable()
            + &reduce_lc(LINKER_SCALAR_MODULUS_LIMBS[limb_idx], &reduce_bit)
            + (
              Scalar::from(EMULATED_LIMB_BASE_U64),
              carry_out.get_variable(),
            )
        },
      );

      carry_in = carry_out;
    }

    Ok(result)
  }

  fn mul_mod<CS: ConstraintSystem<Scalar>>(
    &self,
    mut cs: CS,
    other: &Self,
  ) -> Result<Self, SynthesisError> {
    let lhs_value = self.value.ok_or(SynthesisError::AssignmentMissing)?;
    let rhs_value = other.value.ok_or(SynthesisError::AssignmentMissing)?;
    let result_value = lhs_value * rhs_value;
    let result = Self::alloc_witness(cs.namespace(|| "result"), Some(result_value))?;

    let lhs_big = linker_scalar_to_biguint(lhs_value);
    let rhs_big = linker_scalar_to_biguint(rhs_value);
    let result_big = linker_scalar_to_biguint(result_value);
    let quotient_big = (&lhs_big * &rhs_big - &result_big) / &*LINKER_SCALAR_MODULUS;
    let quotient_value = linker_scalar_from_biguint_checked(&quotient_big);
    let quotient = Self::alloc_witness(cs.namespace(|| "quotient"), Some(quotient_value))?;

    let lhs_limbs = linker_scalar_to_limbs(lhs_value);
    let rhs_limbs = linker_scalar_to_limbs(rhs_value);
    let quotient_limbs = linker_scalar_to_limbs(quotient_value);
    let result_limbs = linker_scalar_to_limbs(result_value);

    let products: Vec<Vec<AllocatedNum<Scalar>>> = self
      .limbs
      .iter()
      .enumerate()
      .map(|(lhs_idx, lhs_limb)| {
        other
          .limbs
          .iter()
          .enumerate()
          .map(|(rhs_idx, rhs_limb)| {
            lhs_limb.mul(
              cs.namespace(|| format!("prod_{lhs_idx}_{rhs_idx}")),
              rhs_limb,
            )
          })
          .collect::<Result<Vec<_>, _>>()
      })
      .collect::<Result<Vec<_>, _>>()?;

    let mut carry_values = [0i64; 2 * EMULATED_LIMBS - 1];
    let mut carry = 0i128;
    for limb_idx in 0..(2 * EMULATED_LIMBS) {
      let lhs_sum = carry
        + (0..=limb_idx)
          .filter(|idx| *idx < EMULATED_LIMBS && limb_idx - *idx < EMULATED_LIMBS)
          .map(|idx| (lhs_limbs[idx] as i128) * (rhs_limbs[limb_idx - idx] as i128))
          .sum::<i128>();
      let rhs_sum = (if limb_idx < EMULATED_LIMBS {
        result_limbs[limb_idx] as i128
      } else {
        0i128
      }) + (0..=limb_idx)
        .filter(|idx| *idx < EMULATED_LIMBS && limb_idx - *idx < EMULATED_LIMBS)
        .map(|idx| {
          (quotient_limbs[idx] as i128) * (LINKER_SCALAR_MODULUS_LIMBS[limb_idx - idx] as i128)
        })
        .sum::<i128>();
      let diff = lhs_sum - rhs_sum;
      debug_assert_eq!(diff % EMULATED_LIMB_BASE_U64 as i128, 0);
      carry = diff / EMULATED_LIMB_BASE_U64 as i128;
      if limb_idx + 1 < 2 * EMULATED_LIMBS {
        carry_values[limb_idx] = carry as i64;
      } else {
        debug_assert_eq!(carry, 0, "top carry must vanish");
      }
    }

    let signed_offset =
      Scalar::from(EMULATED_MUL_CARRY_BIAS as u64) * Scalar::from(EMULATED_LIMB_BASE_U64 - 1);
    let mut carry_in = alloc_constant(
      cs.namespace(|| "carry_init"),
      &Scalar::from(EMULATED_MUL_CARRY_BIAS as u64),
    )?;
    for limb_idx in 0..(2 * EMULATED_LIMBS) {
      let carry_out = if limb_idx + 1 < 2 * EMULATED_LIMBS {
        alloc_bounded_num(
          cs.namespace(|| format!("carry_{limb_idx}")),
          Some((carry_values[limb_idx] + EMULATED_MUL_CARRY_BIAS) as u64),
          EMULATED_MUL_CARRY_BITS,
        )?
        .0
      } else {
        alloc_constant(
          cs.namespace(|| "carry_final"),
          &Scalar::from(EMULATED_MUL_CARRY_BIAS as u64),
        )?
      };

      cs.enforce(
        || format!("mul_mod_limb_{limb_idx}"),
        |lc| {
          let mut lc = lc + (signed_offset, CS::one()) + carry_in.get_variable();
          for lhs_idx in 0..=limb_idx {
            if lhs_idx < EMULATED_LIMBS && limb_idx - lhs_idx < EMULATED_LIMBS {
              lc = lc + products[lhs_idx][limb_idx - lhs_idx].get_variable();
            }
          }
          lc
        },
        |lc| lc + CS::one(),
        |lc| {
          let mut lc = lc;
          if limb_idx < EMULATED_LIMBS {
            lc = lc + result.limbs[limb_idx].get_variable();
          }
          for lhs_idx in 0..=limb_idx {
            if lhs_idx < EMULATED_LIMBS && limb_idx - lhs_idx < EMULATED_LIMBS {
              lc = lc
                + (
                  Scalar::from(LINKER_SCALAR_MODULUS_LIMBS[limb_idx - lhs_idx]),
                  quotient.limbs[lhs_idx].get_variable(),
                );
            }
          }
          lc + (
            Scalar::from(EMULATED_LIMB_BASE_U64),
            carry_out.get_variable(),
          )
        },
      );

      carry_in = carry_out;
    }

    Ok(result)
  }
}

fn enforce_base_to_emulated<CS: ConstraintSystem<Scalar>>(
  mut cs: CS,
  base_value: &AllocatedNum<Scalar>,
  base_native: Scalar,
  emulated: &EmulatedLinkerVar,
) -> Result<(), SynthesisError> {
  let mut base_bits = base_value.to_bits_le_strict(cs.namespace(|| "base_bits"))?;
  base_bits.resize(EMULATED_TOTAL_BITS, Boolean::constant(false));
  let base_limbs = base_bits
    .chunks(EMULATED_LIMB_BITS)
    .enumerate()
    .map(|(limb_idx, chunk)| {
      pack_boolean_bits_to_num(cs.namespace(|| format!("base_limb_{limb_idx}")), chunk)
    })
    .collect::<Result<Vec<_>, _>>()?;

  let base_limbs_native = bits_to_u32_limbs(
    &base_native
      .to_le_bits()
      .into_iter()
      .chain(core::iter::repeat(false))
      .take(EMULATED_TOTAL_BITS)
      .collect::<Vec<_>>(),
  );
  let target_value = emulated.value.ok_or(SynthesisError::AssignmentMissing)?;
  let target_limbs = linker_scalar_to_limbs(target_value);
  let reduce = scalar_to_biguint(base_native) >= *LINKER_SCALAR_MODULUS;
  let reduce_bit = AllocatedBit::alloc(cs.namespace(|| "reduce_bit"), Some(reduce))?;
  let mut carry_values = [0i64; EMULATED_LIMBS - 1];
  let mut carry = 0i64;
  for limb_idx in 0..EMULATED_LIMBS {
    let lhs = target_limbs[limb_idx] as i64
      + if reduce {
        LINKER_SCALAR_MODULUS_LIMBS[limb_idx] as i64
      } else {
        0
      }
      + carry;
    let rhs = base_limbs_native[limb_idx] as i64;
    let diff = lhs - rhs;
    debug_assert_eq!(diff % EMULATED_LIMB_BASE_U64 as i64, 0);
    carry = diff / EMULATED_LIMB_BASE_U64 as i64;
    if limb_idx + 1 < EMULATED_LIMBS {
      carry_values[limb_idx] = carry;
    } else {
      debug_assert_eq!(carry, 0, "final base-to-scalar carry must vanish");
    }
  }

  let signed_offset =
    Scalar::from(EMULATED_ADD_CARRY_BIAS as u64) * Scalar::from(EMULATED_LIMB_BASE_U64 - 1);
  let mut carry_in = alloc_constant(
    cs.namespace(|| "carry_init"),
    &Scalar::from(EMULATED_ADD_CARRY_BIAS as u64),
  )?;

  for limb_idx in 0..EMULATED_LIMBS {
    let carry_out = if limb_idx + 1 < EMULATED_LIMBS {
      alloc_bounded_num(
        cs.namespace(|| format!("carry_{limb_idx}")),
        Some((carry_values[limb_idx] + EMULATED_ADD_CARRY_BIAS) as u64),
        EMULATED_ADD_CARRY_BITS,
      )?
      .0
    } else {
      alloc_constant(
        cs.namespace(|| "carry_final"),
        &Scalar::from(EMULATED_ADD_CARRY_BIAS as u64),
      )?
    };

    cs.enforce(
      || format!("base_to_emulated_limb_{limb_idx}"),
      |lc| {
        lc + (signed_offset, CS::one())
          + carry_in.get_variable()
          + emulated.limbs[limb_idx].get_variable()
          + &Boolean::from(reduce_bit.clone()).lc(
            CS::one(),
            Scalar::from(LINKER_SCALAR_MODULUS_LIMBS[limb_idx]),
          )
      },
      |lc| lc + CS::one(),
      |lc| {
        lc + base_limbs[limb_idx].get_variable()
          + (
            Scalar::from(EMULATED_LIMB_BASE_U64),
            carry_out.get_variable(),
          )
      },
    );

    carry_in = carry_out;
  }

  Ok(())
}

fn enforce_constant_link_sum<const N_LOCAL: usize, CS: ConstraintSystem<Scalar>>(
  mut cs: CS,
  values: &[EmulatedLinkerVar],
  coeffs: &[LinkerScalar; N_LOCAL],
  result: &EmulatedLinkerVar,
) -> Result<(), SynthesisError> {
  let value_limbs = values
    .iter()
    .map(|value| {
      value
        .value
        .map(linker_scalar_to_limbs)
        .ok_or(SynthesisError::AssignmentMissing)
    })
    .collect::<Result<Vec<_>, _>>()?;
  let coeff_limbs: Vec<[u64; EMULATED_LIMBS]> =
    coeffs.iter().copied().map(linker_scalar_to_limbs).collect();
  let result_value = result.value.ok_or(SynthesisError::AssignmentMissing)?;
  let result_limbs = linker_scalar_to_limbs(result_value);

  let lhs_big =
    values
      .iter()
      .zip(coeffs.iter())
      .try_fold(BigUint::zero(), |acc, (value, coeff)| {
        let value_big =
          linker_scalar_to_biguint(value.value.ok_or(SynthesisError::AssignmentMissing)?);
        Ok::<_, SynthesisError>(acc + value_big * linker_scalar_to_biguint(*coeff))
      })?;
  let result_big = linker_scalar_to_biguint(result_value);
  let quotient_big = (&lhs_big - &result_big) / &*LINKER_SCALAR_MODULUS;
  debug_assert_eq!(
    (&lhs_big - &result_big) % &*LINKER_SCALAR_MODULUS,
    BigUint::zero()
  );
  let quotient_limbs = biguint_to_u32_limbs::<LINK_LINEAR_QUOTIENT_LIMBS>(&quotient_big);

  let quotient_vars = quotient_limbs
    .iter()
    .enumerate()
    .map(|(limb_idx, limb)| {
      alloc_bounded_num(
        cs.namespace(|| format!("quotient_limb_{limb_idx}")),
        Some(*limb),
        EMULATED_LIMB_BITS,
      )
      .map(|(num, _)| num)
    })
    .collect::<Result<Vec<_>, _>>()?;

  let linear_limbs = core::cmp::max(
    2 * EMULATED_LIMBS - 1,
    LINK_LINEAR_QUOTIENT_LIMBS + EMULATED_LIMBS - 1,
  );
  let mut carry_values = vec![0i64; linear_limbs - 1];
  let mut carry = 0i128;
  for limb_idx in 0..linear_limbs {
    let lhs_sum = carry
      + value_limbs
        .iter()
        .zip(coeff_limbs.iter())
        .map(|(value_limbs, coeff_limbs)| {
          (0..=limb_idx)
            .filter(|idx| *idx < EMULATED_LIMBS && limb_idx - *idx < EMULATED_LIMBS)
            .map(|idx| (value_limbs[idx] as i128) * (coeff_limbs[limb_idx - idx] as i128))
            .sum::<i128>()
        })
        .sum::<i128>();
    let rhs_sum = (if limb_idx < EMULATED_LIMBS {
      result_limbs[limb_idx] as i128
    } else {
      0i128
    }) + (0..=limb_idx)
      .filter(|idx| *idx < LINK_LINEAR_QUOTIENT_LIMBS && limb_idx - *idx < EMULATED_LIMBS)
      .map(|idx| {
        (quotient_limbs[idx] as i128) * (LINKER_SCALAR_MODULUS_LIMBS[limb_idx - idx] as i128)
      })
      .sum::<i128>();
    let diff = lhs_sum - rhs_sum;
    debug_assert_eq!(diff % EMULATED_LIMB_BASE_U64 as i128, 0);
    carry = diff / EMULATED_LIMB_BASE_U64 as i128;
    if limb_idx + 1 < linear_limbs {
      debug_assert!(carry.abs() < (1i128 << (LINK_LINEAR_CARRY_BITS - 1)));
      carry_values[limb_idx] = carry as i64;
    } else {
      debug_assert_eq!(carry, 0, "final link-sum carry must vanish");
    }
  }

  let signed_offset =
    Scalar::from(LINK_LINEAR_CARRY_BIAS as u64) * Scalar::from(EMULATED_LIMB_BASE_U64 - 1);
  let mut carry_in = alloc_constant(
    cs.namespace(|| "carry_init"),
    &Scalar::from(LINK_LINEAR_CARRY_BIAS as u64),
  )?;

  for limb_idx in 0..linear_limbs {
    let carry_out = if limb_idx + 1 < linear_limbs {
      alloc_bounded_num(
        cs.namespace(|| format!("carry_{limb_idx}")),
        Some((carry_values[limb_idx] + LINK_LINEAR_CARRY_BIAS) as u64),
        LINK_LINEAR_CARRY_BITS,
      )?
      .0
    } else {
      alloc_constant(
        cs.namespace(|| "carry_final"),
        &Scalar::from(LINK_LINEAR_CARRY_BIAS as u64),
      )?
    };

    cs.enforce(
      || format!("link_sum_limb_{limb_idx}"),
      |lc| {
        let mut lc = lc + (signed_offset, CS::one()) + carry_in.get_variable();
        for (value, coeff_limbs) in values.iter().zip(coeff_limbs.iter()) {
          for value_limb_idx in 0..=limb_idx {
            if value_limb_idx < EMULATED_LIMBS && limb_idx - value_limb_idx < EMULATED_LIMBS {
              lc = lc
                + (
                  Scalar::from(coeff_limbs[limb_idx - value_limb_idx]),
                  value.limbs[value_limb_idx].get_variable(),
                );
            }
          }
        }
        lc
      },
      |lc| lc + CS::one(),
      |lc| {
        let mut lc = lc;
        if limb_idx < EMULATED_LIMBS {
          lc = lc + result.limbs[limb_idx].get_variable();
        }
        for quotient_limb_idx in 0..=limb_idx {
          if quotient_limb_idx < LINK_LINEAR_QUOTIENT_LIMBS
            && limb_idx - quotient_limb_idx < EMULATED_LIMBS
          {
            lc = lc
              + (
                Scalar::from(LINKER_SCALAR_MODULUS_LIMBS[limb_idx - quotient_limb_idx]),
                quotient_vars[quotient_limb_idx].get_variable(),
              );
          }
        }
        lc + (
          Scalar::from(EMULATED_LIMB_BASE_U64),
          carry_out.get_variable(),
        )
      },
    );

    carry_in = carry_out;
  }

  Ok(())
}

fn replace_zero_with_one<CS: ConstraintSystem<Scalar>>(
  mut cs: CS,
  value: &AllocatedNum<Scalar>,
) -> Result<AllocatedNum<Scalar>, SynthesisError> {
  let zero = alloc_constant(cs.namespace(|| "zero"), &Scalar::ZERO)?;
  let one = alloc_constant(cs.namespace(|| "one"), &Scalar::ONE)?;
  let is_zero = alloc_num_equals(cs.namespace(|| "is_zero"), value, &zero)?;
  conditionally_select(
    cs.namespace(|| "zero_or_one"),
    &one,
    value,
    &Boolean::from(is_zero),
  )
}

impl PoseidonSpongeCircuit {
  fn new<CS: ConstraintSystem<Scalar>>(cs: &mut CS, prefix: &'static str) -> Self {
    Self {
      state: [
        alloc_zero(cs.namespace(|| format!("{prefix}_state_0"))),
        alloc_zero(cs.namespace(|| format!("{prefix}_state_1"))),
        alloc_zero(cs.namespace(|| format!("{prefix}_state_2"))),
      ],
      mode: SpongeMode::Absorbing {
        next_absorb_index: 0,
      },
      prefix,
      tag_counter: 0,
    }
  }

  fn fresh_tag(&mut self) -> usize {
    let tag = self.tag_counter;
    self.tag_counter += 1;
    tag
  }

  fn apply_mds_row<CS: ConstraintSystem<Scalar>>(
    cs: &mut CS,
    row: &[Scalar; POSEIDON_WIDTH],
    state: &[AllocatedNum<Scalar>; POSEIDON_WIDTH],
    name: &str,
  ) -> Result<AllocatedNum<Scalar>, SynthesisError> {
    linear_combination_num(
      cs.namespace(|| name),
      &[
        (row[0], &state[0]),
        (row[1], &state[1]),
        (row[2], &state[2]),
      ],
      Scalar::ZERO,
    )
  }

  fn pow5<CS: ConstraintSystem<Scalar>>(
    cs: &mut CS,
    value: &AllocatedNum<Scalar>,
    name: &str,
  ) -> Result<AllocatedNum<Scalar>, SynthesisError> {
    let x2 = value.mul(cs.namespace(|| format!("{name}_x2")), value)?;
    let x4 = x2.mul(cs.namespace(|| format!("{name}_x4")), &x2)?;
    x4.mul(cs.namespace(|| format!("{name}_x5")), value)
  }

  fn permute<CS: ConstraintSystem<Scalar>>(&mut self, cs: &mut CS) -> Result<(), SynthesisError> {
    let cfg = &*POSEIDON_CONFIG;
    let full_rounds_over_2 = POSEIDON_FULL_ROUNDS / 2;
    let permute_id = self.fresh_tag();

    for round in 0..full_rounds_over_2 {
      self.apply_round(cs, cfg, permute_id, round, true)?;
    }
    for round in full_rounds_over_2..(full_rounds_over_2 + POSEIDON_PARTIAL_ROUNDS) {
      self.apply_round(cs, cfg, permute_id, round, false)?;
    }
    for round in (full_rounds_over_2 + POSEIDON_PARTIAL_ROUNDS)
      ..(POSEIDON_FULL_ROUNDS + POSEIDON_PARTIAL_ROUNDS)
    {
      self.apply_round(cs, cfg, permute_id, round, true)?;
    }

    Ok(())
  }

  fn apply_round<CS: ConstraintSystem<Scalar>>(
    &mut self,
    cs: &mut CS,
    cfg: &PoseidonConfigLocal,
    permute_id: usize,
    round: usize,
    full_round: bool,
  ) -> Result<(), SynthesisError> {
    for i in 0..POSEIDON_WIDTH {
      self.state[i] = add_constant_num(
        cs.namespace(|| format!("{}_ark_p{permute_id}_r{round}_{i}", self.prefix)),
        &self.state[i],
        cfg.ark[round][i],
      )?;
    }

    if full_round {
      for i in 0..POSEIDON_WIDTH {
        self.state[i] = Self::pow5(
          cs,
          &self.state[i],
          &format!("{}_sbox_p{permute_id}_r{round}_{i}", self.prefix),
        )?;
      }
    } else {
      self.state[0] = Self::pow5(
        cs,
        &self.state[0],
        &format!("{}_sbox_p{permute_id}_r{round}_0", self.prefix),
      )?;
    }

    let old_state = self.state.clone();
    self.state = [
      Self::apply_mds_row(
        cs,
        &cfg.mds[0],
        &old_state,
        &format!("{}_mds_p{permute_id}_r{round}_0", self.prefix),
      )?,
      Self::apply_mds_row(
        cs,
        &cfg.mds[1],
        &old_state,
        &format!("{}_mds_p{permute_id}_r{round}_1", self.prefix),
      )?,
      Self::apply_mds_row(
        cs,
        &cfg.mds[2],
        &old_state,
        &format!("{}_mds_p{permute_id}_r{round}_2", self.prefix),
      )?,
    ];
    Ok(())
  }

  fn absorb_allocated_fields<CS: ConstraintSystem<Scalar>>(
    &mut self,
    cs: &mut CS,
    elements: &[AllocatedNum<Scalar>],
  ) -> Result<(), SynthesisError> {
    if elements.is_empty() {
      return Ok(());
    }

    match self.mode {
      SpongeMode::Absorbing { next_absorb_index } => {
        let mut absorb_index = next_absorb_index;
        if absorb_index == POSEIDON_RATE {
          self.permute(cs)?;
          absorb_index = 0;
        }
        self.absorb_internal(cs, absorb_index, elements)?;
      }
      SpongeMode::Squeezing { .. } => {
        self.absorb_internal(cs, 0, elements)?;
      }
    }

    Ok(())
  }

  fn absorb_internal<CS: ConstraintSystem<Scalar>>(
    &mut self,
    cs: &mut CS,
    mut rate_start_index: usize,
    elements: &[AllocatedNum<Scalar>],
  ) -> Result<(), SynthesisError> {
    let absorb_id = self.fresh_tag();
    let mut offset = 0usize;
    while offset < elements.len() {
      let absorb_now = core::cmp::min(POSEIDON_RATE - rate_start_index, elements.len() - offset);
      for i in 0..absorb_now {
        let state_index = POSEIDON_CAPACITY + rate_start_index + i;
        self.state[state_index] = add_nums(
          cs.namespace(|| {
            format!(
              "{}_absorb_{absorb_id}_{state_index}_{}",
              self.prefix,
              offset + i
            )
          }),
          &self.state[state_index],
          &elements[offset + i],
        )?;
      }
      offset += absorb_now;
      rate_start_index += absorb_now;

      if offset < elements.len() {
        self.permute(cs)?;
        rate_start_index = 0;
      } else {
        self.mode = SpongeMode::Absorbing {
          next_absorb_index: rate_start_index,
        };
      }
    }

    Ok(())
  }

  fn absorb_allocated_field<CS: ConstraintSystem<Scalar>>(
    &mut self,
    cs: &mut CS,
    value: &AllocatedNum<Scalar>,
  ) -> Result<(), SynthesisError> {
    self.absorb_allocated_fields(cs, &[value.clone()])
  }

  fn absorb_constant_fields<CS: ConstraintSystem<Scalar>>(
    &mut self,
    cs: &mut CS,
    values: &[Scalar],
  ) -> Result<(), SynthesisError> {
    let absorb_id = self.fresh_tag();
    let vars = values
      .iter()
      .enumerate()
      .map(|(i, value)| {
        alloc_constant(
          cs.namespace(|| format!("{}_const_absorb_{absorb_id}_{i}", self.prefix)),
          value,
        )
      })
      .collect::<Result<Vec<_>, _>>()?;
    self.absorb_allocated_fields(cs, &vars)
  }

  fn absorb_bytes_constant<CS: ConstraintSystem<Scalar>>(
    &mut self,
    cs: &mut CS,
    bytes: &[u8],
  ) -> Result<(), SynthesisError> {
    let packed = pack_bytes_to_fields(bytes);
    self.absorb_constant_fields(cs, &packed)
  }

  fn absorb_point<CS: ConstraintSystem<Scalar>>(
    &mut self,
    cs: &mut CS,
    point: &AllocatedPointNonInfinity<CurveEngine>,
  ) -> Result<(), SynthesisError> {
    let point_id = self.fresh_tag();
    let infinity_flag = alloc_constant(
      cs.namespace(|| format!("{}_point_infinity_flag_{point_id}", self.prefix)),
      &Scalar::ZERO,
    )?;
    self.absorb_allocated_fields(cs, &[point.x.clone(), point.y.clone(), infinity_flag])
  }

  fn squeeze_field_elements<CS: ConstraintSystem<Scalar>>(
    &mut self,
    cs: &mut CS,
    num_elements: usize,
  ) -> Result<Vec<AllocatedNum<Scalar>>, SynthesisError> {
    let mut out = Vec::with_capacity(num_elements);
    match self.mode {
      SpongeMode::Absorbing { .. } => {
        self.permute(cs)?;
        self.squeeze_internal(cs, 0, num_elements, &mut out)?;
      }
      SpongeMode::Squeezing { next_squeeze_index } => {
        let mut squeeze_index = next_squeeze_index;
        if squeeze_index == POSEIDON_RATE {
          self.permute(cs)?;
          squeeze_index = 0;
        }
        self.squeeze_internal(cs, squeeze_index, num_elements, &mut out)?;
      }
    }
    Ok(out)
  }

  fn squeeze_internal<CS: ConstraintSystem<Scalar>>(
    &mut self,
    cs: &mut CS,
    mut rate_start_index: usize,
    num_elements: usize,
    out: &mut Vec<AllocatedNum<Scalar>>,
  ) -> Result<(), SynthesisError> {
    while out.len() < num_elements {
      let available = POSEIDON_RATE - rate_start_index;
      let take = core::cmp::min(available, num_elements - out.len());
      for i in 0..take {
        out.push(self.state[POSEIDON_CAPACITY + rate_start_index + i].clone());
      }
      rate_start_index += take;
      if out.len() < num_elements {
        self.permute(cs)?;
        rate_start_index = 0;
      } else {
        self.mode = SpongeMode::Squeezing {
          next_squeeze_index: rate_start_index,
        };
      }
    }
    Ok(())
  }
}

fn jacobian_double<F: PrimeField>(x: F, y: F, z: F, a: F) -> (F, F, F) {
  if y == F::ZERO {
    return (F::ONE, F::ONE, F::ZERO);
  }
  let xx = x.square();
  let yy = y.square();
  let yyyy = yy.square();
  let zz = z.square();

  let s = ((x + yy).square() - xx - yyyy).double();
  let m = if a == F::ZERO {
    xx + xx + xx
  } else {
    xx + xx + xx + a * zz.square()
  };
  let t = m.square() - s - s;

  let x3 = t;
  let y3 = m * (s - t) - yyyy.double().double().double();
  let z3 = (y + z).square() - yy - zz;

  (x3, y3, z3)
}

fn jacobian_add_affine<F: PrimeField>(x1: F, y1: F, z1: F, x2: F, y2: F, curve_a: F) -> (F, F, F) {
  if z1 == F::ZERO {
    return (x2, y2, F::ONE);
  }

  let z1z1 = z1.square();
  let u2 = x2 * z1z1;
  let s2 = y2 * z1 * z1z1;

  let h = u2 - x1;
  let r = s2 - y1;

  if h == F::ZERO {
    if r == F::ZERO {
      return jacobian_double(x1, y1, z1, curve_a);
    }
    return (F::ONE, F::ONE, F::ZERO);
  }

  let hh = h.square();
  let hhh = hh * h;
  let v = x1 * hh;

  let x3 = r.square() - hhh - v - v;
  let y3 = r * (v - x3) - y1 * hhh;
  let z3 = z1 * h;

  (x3, y3, z3)
}

fn affine_add_safe<F: PrimeField>(x1: F, y1: F, x2: F, y2: F, a: F) -> (F, F) {
  let (rx, ry, rz) = jacobian_add_affine(x1, y1, F::ONE, x2, y2, a);
  let z_inv = rz.invert().unwrap();
  let z_inv2 = z_inv.square();
  let z_inv3 = z_inv2 * z_inv;
  (rx * z_inv2, ry * z_inv3)
}

fn native_scalar_mul_maybe<S, F>(scalar: S, point_x: F, point_y: F, curve_a: F) -> Option<(F, F)>
where
  S: PrimeFieldBits + PartialEq,
  F: PrimeField,
{
  if scalar == S::ZERO {
    return None;
  }

  let bits: Vec<bool> = scalar.to_le_bits().into_iter().collect();
  let highest = bits.iter().rposition(|b| *b)?;

  let mut ax = point_x;
  let mut ay = point_y;
  let mut az = F::ONE;

  for i in (0..highest).rev() {
    let (dx, dy, dz) = jacobian_double(ax, ay, az, curve_a);
    ax = dx;
    ay = dy;
    az = dz;

    if bits[i] {
      let (sx, sy, sz) = jacobian_add_affine(ax, ay, az, point_x, point_y, curve_a);
      ax = sx;
      ay = sy;
      az = sz;
    }
  }

  let z_inv = az.invert().unwrap();
  let z_inv2 = z_inv.square();
  let z_inv3 = z_inv2 * z_inv;
  Some((ax * z_inv2, ay * z_inv3))
}

fn affine_from_coords(point: (Scalar, Scalar)) -> Option<CurveAffinePoint> {
  CurveAffinePoint::from_xy(point.0, point.1).into()
}

fn coords_from_affine(affine: &CurveAffinePoint) -> (Scalar, Scalar) {
  let point = CurvePoint::group(affine);
  let (x, y, is_inf) = point.to_coordinates();
  assert!(!is_inf, "expected non-infinity point");
  (x, y)
}

fn coords_from_group(point: &CurvePoint) -> (Scalar, Scalar) {
  let (x, y, is_inf) = point.to_coordinates();
  assert!(!is_inf, "expected non-infinity point");
  (x, y)
}

fn group_from_affine(point: &CurveAffinePoint) -> CurvePoint {
  CurvePoint::group(point)
}

fn group_from_coords(point: (Scalar, Scalar)) -> CurvePoint {
  group_from_affine(&affine_from_coords(point).expect("expected valid affine point"))
}

fn labeled_affine_points(label: &'static [u8], n: usize) -> Vec<CurveAffinePoint> {
  CurvePoint::from_label(label, n)
}

fn labeled_points(label: &'static [u8], n: usize) -> Vec<(Scalar, Scalar)> {
  labeled_affine_points(label, n)
    .iter()
    .map(coords_from_affine)
    .collect()
}

fn scalar_mul_affine_point(scalar: LinkerScalar, point: &CurveAffinePoint) -> CurvePoint {
  group_from_affine(point) * scalar
}

fn native_ciphertext_from_groups(
  c1: &CurvePoint,
  c2: &CurvePoint,
) -> NativeCiphertext<CurveEngine> {
  let c1 = coords_from_group(c1);
  let c2 = coords_from_group(c2);
  NativeCiphertext::new(c1.0, c1.1, c2.0, c2.1)
}

fn ciphertext_affine_bases<const N_LOCAL: usize>(
  ciphertexts: &[NativeCiphertext<CurveEngine>; N_LOCAL],
) -> (Vec<CurveAffinePoint>, Vec<CurveAffinePoint>) {
  let c1_bases = ciphertexts
    .iter()
    .map(|ct| affine_from_coords((ct.c1_x, ct.c1_y)).expect("expected valid c1 point"))
    .collect();
  let c2_bases = ciphertexts
    .iter()
    .map(|ct| affine_from_coords((ct.c2_x, ct.c2_y)).expect("expected valid c2 point"))
    .collect();
  (c1_bases, c2_bases)
}

fn native_ciphertext_msm_groups(
  c1_bases: &[CurveAffinePoint],
  c2_bases: &[CurveAffinePoint],
  scalars: &[LinkerScalar],
) -> (CurvePoint, CurvePoint) {
  let (c1, c2) = join(
    || {
      CurvePoint::vartime_multiscalar_mul(scalars, c1_bases)
        .expect("ciphertext msm c1 must succeed")
    },
    || {
      CurvePoint::vartime_multiscalar_mul(scalars, c2_bases)
        .expect("ciphertext msm c2 must succeed")
    },
  );
  (c1, c2)
}

fn build_commitment_bases() -> CommitmentBases {
  let generator_affine = CurvePoint::generator().affine();
  let generator = coords_from_affine(&generator_affine);
  let num_bits = LinkerScalar::NUM_BITS as usize;

  let perm_bases_affine = labeled_affine_points(b"spartan2-bench-perm", N + 1);
  let power_bases_affine = labeled_affine_points(b"spartan2-bench-power", N + 1);
  let link_bases_affine = labeled_affine_points(b"spartan2-bench-link", 2);

  let link_points: Vec<_> = link_bases_affine.iter().map(coords_from_affine).collect();

  CommitmentBases {
    generator,
    generator_affine,
    generator_powers: precompute_native_powers::<CurveEngine>(generator, num_bits),
    perm_bases_affine: perm_bases_affine[..N].to_vec(),
    perm_blind_base_affine: perm_bases_affine[N],
    power_bases_affine: power_bases_affine[..N].to_vec(),
    power_blind_base_affine: power_bases_affine[N],
    link_base: link_points[0],
    link_base_affine: link_bases_affine[0],
    link_blind_base: link_points[1],
    link_blind_base_affine: link_bases_affine[1],
    link_base_powers: precompute_native_powers::<CurveEngine>(link_points[0], num_bits),
    link_blind_powers: precompute_native_powers::<CurveEngine>(link_points[1], num_bits),
  }
}

fn native_vector_commitment(
  values: &[LinkerScalar],
  bases: &[CurveAffinePoint],
  blinding: LinkerScalar,
  blind_base: &CurveAffinePoint,
) -> (Scalar, Scalar) {
  assert_eq!(values.len(), bases.len(), "commitment arity mismatch");
  let (msm_term, blind_term) = join(
    || {
      CurvePoint::vartime_multiscalar_mul(values, bases)
        .expect("vector commitment msm must succeed")
    },
    || scalar_mul_affine_point(blinding, blind_base),
  );
  coords_from_group(&(msm_term + blind_term))
}

fn native_link_commitment(
  link_value: LinkerScalar,
  link_blinding: LinkerScalar,
  link_base: &CurveAffinePoint,
  blind_base: &CurveAffinePoint,
) -> (Scalar, Scalar) {
  let (l_term, r_term) = join(
    || scalar_mul_affine_point(link_value, link_base),
    || scalar_mul_affine_point(link_blinding, blind_base),
  );
  coords_from_group(&(l_term + r_term))
}

fn nonzero_linker_scalar_from_base(value: Scalar) -> LinkerScalar {
  let scalar = linker_scalar_from_le_bytes_mod_order(&scalar_to_le_bytes(value));
  if scalar == LinkerScalar::ZERO {
    LinkerScalar::ONE
  } else {
    scalar
  }
}

fn derive_sigma_scalars<const N_LOCAL: usize>(
  seed_digest: Scalar,
  label: &[u8],
) -> [LinkerScalar; N_LOCAL] {
  let mut sponge = PoseidonSpongeNative::new();
  sponge.absorb_bytes(b"reencryption-proof");
  sponge.absorb_field(seed_digest);
  sponge.absorb_bytes(label);
  let squeezed = sponge.squeeze_field_elements(N_LOCAL);
  std::array::from_fn(|i| nonzero_linker_scalar_from_base(squeezed[i]))
}

fn native_add_points(lhs: (Scalar, Scalar), rhs: (Scalar, Scalar)) -> (Scalar, Scalar) {
  let (curve_a, _, _, _) = <CurveEngine as Engine>::GE::group_params();
  affine_add_safe(lhs.0, lhs.1, rhs.0, rhs.1, curve_a)
}

fn native_ciphertext_msm<const N_LOCAL: usize>(
  ciphertexts: &[NativeCiphertext<CurveEngine>; N_LOCAL],
  scalars: &[LinkerScalar; N_LOCAL],
) -> NativeCiphertext<CurveEngine> {
  let (c1_bases, c2_bases) = ciphertext_affine_bases(ciphertexts);
  let (c1, c2) = native_ciphertext_msm_groups(&c1_bases, &c2_bases, scalars);
  native_ciphertext_from_groups(&c1, &c2)
}

fn native_encrypt_zero_and_combine<const N_LOCAL: usize>(
  public_key: (Scalar, Scalar),
  randomness: LinkerScalar,
  ciphertexts: &[NativeCiphertext<CurveEngine>; N_LOCAL],
  scalar_factors: &[LinkerScalar; N_LOCAL],
  generator: (Scalar, Scalar),
) -> NativeCiphertext<CurveEngine> {
  let public_key_affine = affine_from_coords(public_key).expect("expected valid public key");
  let generator_affine = affine_from_coords(generator).expect("expected valid generator");
  let (c1_bases, c2_bases) = ciphertext_affine_bases(ciphertexts);
  let (msm_terms, rerand_terms) = join(
    || native_ciphertext_msm_groups(&c1_bases, &c2_bases, scalar_factors),
    || {
      join(
        || scalar_mul_affine_point(randomness, &generator_affine),
        || scalar_mul_affine_point(randomness, &public_key_affine),
      )
    },
  );
  let c1 = msm_terms.0 + rerand_terms.0;
  let c2 = msm_terms.1 + rerand_terms.1;
  native_ciphertext_from_groups(&c1, &c2)
}

fn sigma_absorb_public_inputs(
  transcript: &mut PoseidonSpongeNative,
  c_aggregator: &NativeCiphertext<CurveEngine>,
  c_b: (Scalar, Scalar),
) {
  let c_in_aggregate = native_add_points(
    (c_aggregator.c1_x, c_aggregator.c1_y),
    (c_aggregator.c2_x, c_aggregator.c2_y),
  );
  transcript.absorb_point(c_in_aggregate);
  transcript.absorb_point(c_b);
}

fn rerandomize_ciphertext(
  ciphertext: &NativeCiphertext<CurveEngine>,
  randomness: LinkerScalar,
  public_key: (Scalar, Scalar),
  generator: (Scalar, Scalar),
) -> NativeCiphertext<CurveEngine> {
  let ciphertext_c1 = group_from_coords((ciphertext.c1_x, ciphertext.c1_y));
  let ciphertext_c2 = group_from_coords((ciphertext.c2_x, ciphertext.c2_y));
  let public_key_affine = affine_from_coords(public_key).expect("expected valid public key");
  let generator_affine = affine_from_coords(generator).expect("expected valid generator");
  let (rerand_c1, rerand_c2) = join(
    || scalar_mul_affine_point(randomness, &generator_affine),
    || scalar_mul_affine_point(randomness, &public_key_affine),
  );
  let c1 = ciphertext_c1 + rerand_c1;
  let c2 = ciphertext_c2 + rerand_c2;
  native_ciphertext_from_groups(&c1, &c2)
}

fn powers_sequence<const N_LOCAL: usize>(x: LinkerScalar) -> [LinkerScalar; N_LOCAL] {
  let mut powers = [LinkerScalar::ZERO; N_LOCAL];
  powers[0] = LinkerScalar::ONE;
  for i in 1..N_LOCAL {
    powers[i] = powers[i - 1] * x;
  }
  powers
}

fn encode_spartan_public_values<const N_LOCAL: usize>(
  statement: &ShuffleStatement<N_LOCAL>,
) -> Vec<Scalar> {
  let mut out = Vec::with_capacity(2 + 1 + N_LOCAL + 1 + 1 + 1 + 2 + 2 + 2 + 1);
  out.push(statement.pk.0);
  out.push(statement.pk.1);
  out.push(statement.nonce);
  for i in 0..N_LOCAL {
    out.push(Scalar::from(i as u64));
  }
  out.push(statement.seed_digest);
  out.push(statement.power_challenge);
  out.push(statement.tau_base);
  out.push(statement.permutation_commitment.0);
  out.push(statement.permutation_commitment.1);
  out.push(statement.power_commitment.0);
  out.push(statement.power_commitment.1);
  out.push(statement.link_commitment.0);
  out.push(statement.link_commitment.1);
  out.push(statement.tuple_expected_product);
  out
}

fn checked_affine_from_coords(
  label: impl Into<String>,
  point: (Scalar, Scalar),
) -> Result<CurveAffinePoint, ShuffleVerifyError> {
  let label = label.into();
  let Some(point) = affine_from_coords(point) else {
    return Err(ShuffleVerifyError::InvalidPoint { label });
  };
  if bool::from(point.is_identity()) {
    return Err(ShuffleVerifyError::InvalidPoint { label });
  }
  Ok(point)
}

fn validate_public_point(
  label: impl Into<String>,
  point: (Scalar, Scalar),
) -> Result<(), ShuffleVerifyError> {
  checked_affine_from_coords(label, point)?;
  Ok(())
}

fn validate_public_ciphertext(
  label: &str,
  index: usize,
  ciphertext: &NativeCiphertext<CurveEngine>,
) -> Result<(), ShuffleVerifyError> {
  validate_public_point(
    format!("{label}[{index}].c1"),
    (ciphertext.c1_x, ciphertext.c1_y),
  )
  .map_err(|_| ShuffleVerifyError::InvalidCiphertext {
    label: format!("{label}[{index}]"),
  })?;
  validate_public_point(
    format!("{label}[{index}].c2"),
    (ciphertext.c2_x, ciphertext.c2_y),
  )
  .map_err(|_| ShuffleVerifyError::InvalidCiphertext {
    label: format!("{label}[{index}]"),
  })?;
  Ok(())
}

fn derive_statement_challenges<const N_LOCAL: usize>(
  statement: &ShuffleStatement<N_LOCAL>,
) -> Result<LinkerChallenges, ShuffleVerifyError> {
  let derived = derive_linker_challenges::<N_LOCAL>(
    statement.seed_digest,
    statement.permutation_commitment,
    statement.power_commitment,
  )
  .map_err(|reason| ShuffleVerifyError::InconsistentStatement { reason })?;

  if derived.power_challenge != statement.power_challenge {
    return Err(ShuffleVerifyError::InconsistentStatement {
      reason: "power challenge x does not match transcript".to_string(),
    });
  }
  if derived.tau_base != statement.tau_base {
    return Err(ShuffleVerifyError::InconsistentStatement {
      reason: "tau challenge does not match transcript".to_string(),
    });
  }
  if derived.tuple_expected_product != statement.tuple_expected_product {
    return Err(ShuffleVerifyError::InconsistentStatement {
      reason: "tuple expected product does not match transcript".to_string(),
    });
  }

  Ok(derived)
}

fn prove_native_sigma<const N_LOCAL: usize>(
  statement: &ShuffleStatement<N_LOCAL>,
  witness: &ShuffleWitness<N_LOCAL, LEVELS>,
  bases: &CommitmentBases,
) -> NativeReencryptionProof<N_LOCAL> {
  let powers = powers_sequence::<N_LOCAL>(witness.power_challenge_scalar);
  let input_ciphertext_aggregator = native_ciphertext_msm(&statement.input_ciphertexts, &powers);
  let pk_affine = affine_from_coords(statement.pk).expect("expected valid public key");
  let (output_c1_bases, output_c2_bases) = ciphertext_affine_bases(&statement.output_ciphertexts);

  let blinding_factors =
    derive_sigma_scalars::<N_LOCAL>(statement.seed_digest, b"sigma-blinding-factors");
  let sigma_aux = derive_sigma_scalars::<2>(statement.seed_digest, b"sigma-aux");
  let blinding_factor_for_commitment = sigma_aux[0];
  let ciphertext_masking_rerand = sigma_aux[1];

  let (blinding_factor_commitment, blinding_rerandomization_commitment) = join(
    || {
      native_vector_commitment(
        &blinding_factors,
        &bases.power_bases_affine,
        blinding_factor_for_commitment,
        &bases.power_blind_base_affine,
      )
    },
    || {
      let (msm_terms, rerand_terms) = join(
        || native_ciphertext_msm_groups(&output_c1_bases, &output_c2_bases, &blinding_factors),
        || {
          join(
            || scalar_mul_affine_point(ciphertext_masking_rerand, &bases.generator_affine),
            || scalar_mul_affine_point(ciphertext_masking_rerand, &pk_affine),
          )
        },
      );
      native_ciphertext_from_groups(
        &(msm_terms.0 + rerand_terms.0),
        &(msm_terms.1 + rerand_terms.1),
      )
    },
  );

  let mut transcript = PoseidonSpongeNative::new();
  sigma_absorb_public_inputs(
    &mut transcript,
    &input_ciphertext_aggregator,
    statement.power_commitment,
  );
  transcript.absorb_point(blinding_factor_commitment);
  transcript.absorb_point((
    blinding_rerandomization_commitment.c1_x,
    blinding_rerandomization_commitment.c1_y,
  ));
  transcript.absorb_point((
    blinding_rerandomization_commitment.c2_x,
    blinding_rerandomization_commitment.c2_y,
  ));
  let challenge = nonzero_linker_scalar_from_base(transcript.squeeze_field_elements(1)[0]);

  let (aggregated_rerandomizer, sigma_response_power_permutation_vector) = join(
    || {
      -witness
        .power_perm_vec
        .iter()
        .zip(witness.rerandomization_scalars.iter())
        .map(|(b_i, rho_i)| *b_i * *rho_i)
        .sum::<LinkerScalar>()
    },
    || std::array::from_fn(|i| blinding_factors[i] + challenge * witness.power_perm_vec[i]),
  );
  let sigma_response_blinding = blinding_factor_for_commitment + challenge * witness.power_blinding;
  let sigma_response_rerand = ciphertext_masking_rerand + challenge * aggregated_rerandomizer;

  NativeReencryptionProof {
    blinding_factor_commitment,
    blinding_rerandomization_commitment,
    sigma_response_power_permutation_vector,
    sigma_response_blinding,
    sigma_response_rerand,
  }
}

fn verify_native_sigma_public<const N_LOCAL: usize>(
  statement: &ShuffleStatement<N_LOCAL>,
  proof: &NativeReencryptionProof<N_LOCAL>,
  bases: &CommitmentBases,
) -> Result<(), ShuffleVerifyError> {
  let derived = derive_statement_challenges(statement)?;
  let powers = powers_sequence::<N_LOCAL>(derived.power_challenge_scalar);
  let input_ciphertext_aggregator = native_ciphertext_msm(&statement.input_ciphertexts, &powers);
  let power_commitment_affine = checked_affine_from_coords("C_power", statement.power_commitment)?;
  let proof_blinding_factor_commitment_affine = checked_affine_from_coords(
    "sigma.blinding_factor_commitment",
    proof.blinding_factor_commitment,
  )?;
  let proof_blinding_rerand_c1 = checked_affine_from_coords(
    "sigma.blinding_rerandomization_commitment.c1",
    (
      proof.blinding_rerandomization_commitment.c1_x,
      proof.blinding_rerandomization_commitment.c1_y,
    ),
  )?;
  let proof_blinding_rerand_c2 = checked_affine_from_coords(
    "sigma.blinding_rerandomization_commitment.c2",
    (
      proof.blinding_rerandomization_commitment.c2_x,
      proof.blinding_rerandomization_commitment.c2_y,
    ),
  )?;
  let input_aggregator_c1 = checked_affine_from_coords(
    "sigma.input_aggregator.c1",
    (
      input_ciphertext_aggregator.c1_x,
      input_ciphertext_aggregator.c1_y,
    ),
  )?;
  let input_aggregator_c2 = checked_affine_from_coords(
    "sigma.input_aggregator.c2",
    (
      input_ciphertext_aggregator.c2_x,
      input_ciphertext_aggregator.c2_y,
    ),
  )?;
  let output_pk_affine = checked_affine_from_coords("pk", statement.pk)?;
  let (output_c1_bases, output_c2_bases) = ciphertext_affine_bases(&statement.output_ciphertexts);

  let mut transcript = PoseidonSpongeNative::new();
  sigma_absorb_public_inputs(
    &mut transcript,
    &input_ciphertext_aggregator,
    statement.power_commitment,
  );
  transcript.absorb_point(proof.blinding_factor_commitment);
  transcript.absorb_point((
    proof.blinding_rerandomization_commitment.c1_x,
    proof.blinding_rerandomization_commitment.c1_y,
  ));
  transcript.absorb_point((
    proof.blinding_rerandomization_commitment.c2_x,
    proof.blinding_rerandomization_commitment.c2_y,
  ));
  let challenge = nonzero_linker_scalar_from_base(transcript.squeeze_field_elements(1)[0]);

  let (lhs_com, lhs_grp) = join(
    || {
      native_vector_commitment(
        &proof.sigma_response_power_permutation_vector,
        &bases.power_bases_affine,
        proof.sigma_response_blinding,
        &bases.power_blind_base_affine,
      )
    },
    || {
      let (msm_terms, rerand_terms) = join(
        || {
          native_ciphertext_msm_groups(
            &output_c1_bases,
            &output_c2_bases,
            &proof.sigma_response_power_permutation_vector,
          )
        },
        || {
          join(
            || scalar_mul_affine_point(proof.sigma_response_rerand, &bases.generator_affine),
            || scalar_mul_affine_point(proof.sigma_response_rerand, &output_pk_affine),
          )
        },
      );
      native_ciphertext_from_groups(
        &(msm_terms.0 + rerand_terms.0),
        &(msm_terms.1 + rerand_terms.1),
      )
    },
  );
  let rhs_com = coords_from_group(
    &(group_from_affine(&proof_blinding_factor_commitment_affine)
      + scalar_mul_affine_point(challenge, &power_commitment_affine)),
  );
  if lhs_com != rhs_com {
    return Err(ShuffleVerifyError::SigmaVerification {
      reason: "power commitment opening equation failed".to_string(),
    });
  }

  let (rhs_c1, rhs_c2) = join(
    || {
      coords_from_group(
        &(group_from_affine(&proof_blinding_rerand_c1)
          + scalar_mul_affine_point(challenge, &input_aggregator_c1)),
      )
    },
    || {
      coords_from_group(
        &(group_from_affine(&proof_blinding_rerand_c2)
          + scalar_mul_affine_point(challenge, &input_aggregator_c2)),
      )
    },
  );

  if lhs_grp.c1_x != rhs_c1.0
    || lhs_grp.c1_y != rhs_c1.1
    || lhs_grp.c2_x != rhs_c2.0
    || lhs_grp.c2_y != rhs_c2.1
  {
    return Err(ShuffleVerifyError::SigmaVerification {
      reason: "reencryption aggregate equation failed".to_string(),
    });
  }

  Ok(())
}

fn ciphertext_eq(lhs: &NativeCiphertext<CurveEngine>, rhs: &NativeCiphertext<CurveEngine>) -> bool {
  lhs.c1_x == rhs.c1_x && lhs.c1_y == rhs.c1_y && lhs.c2_x == rhs.c2_x && lhs.c2_y == rhs.c2_y
}

fn validate_shared_shuffle_relations(
  statement: &ShuffleStatement<N>,
  witness: &ShuffleWitness<N, LEVELS>,
  bases: &CommitmentBases,
) {
  let permutation: [usize; N] =
    std::array::from_fn(|i| witness.witness_trace.next_levels[LEVELS - 1][i].idx as usize);
  let final_indices: [LinkerScalar; N] =
    std::array::from_fn(|i| LinkerScalar::from(permutation[i] as u64));
  let expected_perm_commitment = native_vector_commitment(
    &final_indices,
    &bases.perm_bases_affine,
    witness.perm_blinding,
    &bases.perm_blind_base_affine,
  );
  assert_eq!(
    expected_perm_commitment, statement.permutation_commitment,
    "C_perm does not match final RS indices"
  );

  let expected_power_commitment = native_vector_commitment(
    &witness.power_perm_vec,
    &bases.power_bases_affine,
    witness.power_blinding,
    &bases.power_blind_base_affine,
  );
  assert_eq!(
    expected_power_commitment, statement.power_commitment,
    "C_power does not match power permutation vector"
  );

  let expected_link_value =
    compute_link_value_from_slice(&witness.power_perm_vec, witness.tau_scalar);
  assert_eq!(
    expected_link_value, witness.link_value,
    "linker scalar L does not match Horner reduction"
  );
  let expected_link_commitment = native_link_commitment(
    witness.link_value,
    witness.link_blinding,
    &bases.link_base_affine,
    &bases.link_blind_base_affine,
  );
  assert_eq!(
    expected_link_commitment, statement.link_commitment,
    "C_link does not match linker value"
  );

  let derived = derive_linker_challenges::<N>(
    statement.seed_digest,
    statement.permutation_commitment,
    statement.power_commitment,
  )
  .expect("linker challenge derivation must succeed");
  assert_eq!(
    derived.power_challenge, statement.power_challenge,
    "x challenge mismatch"
  );
  assert_eq!(
    derived.power_challenge_scalar, witness.power_challenge_scalar,
    "x scalar mismatch"
  );
  assert_eq!(derived.tau_base, statement.tau_base, "tau base mismatch");
  assert_eq!(
    derived.tau_scalar, witness.tau_scalar,
    "tau scalar mismatch"
  );
  assert_eq!(
    derived.tuple_expected_product, statement.tuple_expected_product,
    "P_graph mismatch"
  );

  for i in 0..N {
    let expected = rerandomize_ciphertext(
      &statement.input_ciphertexts[permutation[i]],
      witness.rerandomization_scalars[i],
      statement.pk,
      bases.generator,
    );
    assert!(
      ciphertext_eq(&expected, &statement.output_ciphertexts[i]),
      "output ciphertext {i} does not match rerandomized permutation image"
    );
  }

  let powers = powers_sequence::<N>(witness.power_challenge_scalar);
  let input_aggregator = native_ciphertext_msm(&statement.input_ciphertexts, &powers);
  let rerand_sum = -witness
    .power_perm_vec
    .iter()
    .zip(witness.rerandomization_scalars.iter())
    .map(|(b_i, rho_i)| *b_i * *rho_i)
    .sum::<LinkerScalar>();
  let sigma_relation_lhs = native_encrypt_zero_and_combine(
    statement.pk,
    rerand_sum,
    &statement.output_ciphertexts,
    &witness.power_perm_vec,
    bases.generator,
  );
  assert!(
    ciphertext_eq(&sigma_relation_lhs, &input_aggregator),
    "native Sigma shuffle relation does not hold on the shared witness"
  );
}

fn verify_shuffle_proof<const N_LOCAL: usize>(
  vk: &<ShuffleSnark as R1CSSNARKTrait<PallasHyraxEngine>>::VerifierKey,
  statement: &ShuffleStatement<N_LOCAL>,
  proof: &ShuffleProof<N_LOCAL>,
  bases: &CommitmentBases,
) -> Result<Vec<Scalar>, ShuffleVerifyError> {
  validate_public_point("pk", statement.pk)?;
  validate_public_point("C_perm", statement.permutation_commitment)?;
  validate_public_point("C_power", statement.power_commitment)?;
  validate_public_point("C_link", statement.link_commitment)?;
  for (i, ciphertext) in statement.input_ciphertexts.iter().enumerate() {
    validate_public_ciphertext("input_ciphertexts", i, ciphertext)?;
  }
  for (i, ciphertext) in statement.output_ciphertexts.iter().enumerate() {
    validate_public_ciphertext("output_ciphertexts", i, ciphertext)?;
  }
  validate_public_point(
    "sigma.blinding_factor_commitment",
    proof.sigma_proof.blinding_factor_commitment,
  )?;
  validate_public_ciphertext(
    "sigma.blinding_rerandomization_commitment",
    0,
    &proof.sigma_proof.blinding_rerandomization_commitment,
  )?;

  let _derived = derive_statement_challenges(statement)?;
  let expected_public_values = encode_spartan_public_values(statement);
  let returned_public_values = proof.spartan_proof.verify(vk)?;
  if returned_public_values != expected_public_values {
    return Err(ShuffleVerifyError::SpartanPublicValuesMismatch);
  }

  verify_native_sigma_public(statement, &proof.sigma_proof, bases)?;
  Ok(returned_public_values)
}

fn verify_spartan_statement_binding<const N_LOCAL: usize>(
  vk: &<ShuffleSnark as R1CSSNARKTrait<PallasHyraxEngine>>::VerifierKey,
  statement: &ShuffleStatement<N_LOCAL>,
  proof: &ShuffleSnark,
) -> Result<Vec<Scalar>, ShuffleVerifyError> {
  verify_spartan_statement_binding_generic::<ShuffleSnark, N_LOCAL>(vk, statement, proof)
}

fn verify_spartan_statement_binding_generic<S, const N_LOCAL: usize>(
  vk: &<S as R1CSSNARKTrait<PallasHyraxEngine>>::VerifierKey,
  statement: &ShuffleStatement<N_LOCAL>,
  proof: &S,
) -> Result<Vec<Scalar>, ShuffleVerifyError>
where
  S: R1CSSNARKTrait<PallasHyraxEngine>,
{
  validate_public_point("pk", statement.pk)?;
  validate_public_point("C_perm", statement.permutation_commitment)?;
  validate_public_point("C_power", statement.power_commitment)?;
  validate_public_point("C_link", statement.link_commitment)?;

  let _derived = derive_statement_challenges(statement)?;
  let expected_public_values = encode_spartan_public_values(statement);
  let returned_public_values = proof.verify(vk)?;
  if returned_public_values != expected_public_values {
    return Err(ShuffleVerifyError::SpartanPublicValuesMismatch);
  }
  Ok(returned_public_values)
}

fn validate_power_distinctness<F: PrimeField + PartialEq, const N_LOCAL: usize>(
  x: F,
) -> Result<(), String> {
  if x == F::ZERO {
    return Err("power challenge x must be non-zero".to_string());
  }
  let mut power = x;
  for k in 1..N_LOCAL {
    if power == F::ONE {
      return Err(format!(
        "power challenge x has multiplicative order {k} < N={N_LOCAL}"
      ));
    }
    power *= x;
  }
  Ok(())
}

fn compute_link_value_from_slice<F: PrimeField>(values: &[F], tau: F) -> F {
  values
    .iter()
    .rev()
    .fold(F::ZERO, |acc, value| acc * tau + value)
}

fn compute_expected_graph_product<const N_LOCAL: usize>(
  power_challenge: LinkerScalar,
  rho: Scalar,
  index_coeff: Scalar,
  limb_coeffs: &[Scalar],
) -> Scalar {
  let mut current = LinkerScalar::ONE;
  let mut product = Scalar::ONE;
  for idx in 0..N_LOCAL {
    let mut encoded = index_coeff * Scalar::from(idx as u64);
    for (coeff, limb) in limb_coeffs.iter().zip(linker_scalar_to_limbs(current)) {
      encoded += *coeff * Scalar::from(limb);
    }
    product *= rho - encoded;
    current *= power_challenge;
  }
  product
}

fn derive_power_challenge_from_commitment(permutation_commitment: (Scalar, Scalar)) -> Scalar {
  let mut transcript = PoseidonSpongeNative::new();
  transcript.absorb_bytes(b"permutation-proof");
  transcript.absorb_point(permutation_commitment);
  transcript.absorb_bytes(b"power-challenge");
  let x = transcript.squeeze_field_elements(1)[0];
  if x == Scalar::ZERO { Scalar::ONE } else { x }
}

fn derive_linker_challenges<const N_LOCAL: usize>(
  seed_digest: Scalar,
  permutation_commitment: (Scalar, Scalar),
  power_commitment: (Scalar, Scalar),
) -> Result<LinkerChallenges, String> {
  let mut transcript = PoseidonSpongeNative::new();
  transcript.absorb_bytes(b"permutation-proof");
  transcript.absorb_point(permutation_commitment);
  transcript.absorb_bytes(b"power-challenge");
  let power_challenge = {
    let value = transcript.squeeze_field_elements(1)[0];
    if value == Scalar::ZERO {
      Scalar::ONE
    } else {
      value
    }
  };
  let power_challenge_scalar =
    linker_scalar_from_le_bytes_mod_order(&scalar_to_le_bytes(power_challenge));
  validate_power_distinctness::<LinkerScalar, N_LOCAL>(power_challenge_scalar)?;

  transcript.absorb_field(seed_digest);
  transcript.absorb_point(power_commitment);
  transcript.absorb_bytes(b"tau-challenge");
  let tau = {
    let value = transcript.squeeze_field_elements(1)[0];
    if value == Scalar::ZERO {
      Scalar::ONE
    } else {
      value
    }
  };
  let tau_scalar = linker_scalar_from_le_bytes_mod_order(&scalar_to_le_bytes(tau));

  transcript.absorb_bytes(b"tuple-challenges");
  let challenges = transcript.squeeze_field_elements(EMULATED_LIMBS + 2);
  let rho = challenges[0];
  let index_coeff = challenges[1];
  let limb_coeffs = challenges[2..].to_vec();
  let expected = compute_expected_graph_product::<N_LOCAL>(
    power_challenge_scalar,
    rho,
    index_coeff,
    &limb_coeffs,
  );

  Ok(LinkerChallenges {
    power_challenge,
    power_challenge_scalar,
    tau_base: tau,
    tau_scalar,
    tuple_expected_product: expected,
  })
}

fn derive_vrf_value(nonce: Scalar, sk: LinkerScalar) -> Scalar {
  let mut sponge = PoseidonSpongeNative::new();
  sponge.absorb_field(nonce);
  for byte in linker_scalar_to_le_bytes(sk) {
    sponge.absorb_field(Scalar::from(byte as u64));
  }
  sponge.squeeze_field_elements(1)[0]
}

fn seed_digest_from_vrf_value(vrf_value: Scalar) -> Scalar {
  let mut sponge = PoseidonSpongeNative::new();
  sponge.absorb_field(vrf_value);
  sponge.squeeze_field_elements(1)[0]
}

fn rs_num_samples() -> usize {
  let usable_bits_per_sample = (Scalar::NUM_BITS as usize).saturating_sub(2);
  (N * LEVELS).div_ceil(usable_bits_per_sample)
}

fn derive_split_bits_poseidon_native(seed: Scalar) -> ([[bool; N]; LEVELS], usize) {
  let num_samples = rs_num_samples();
  let mut sponge = PoseidonSpongeNative::new();
  sponge.absorb_field(seed);
  let random_values = sponge.squeeze_field_elements(num_samples);

  let mut bit_stream = Vec::with_capacity(num_samples * (Scalar::NUM_BITS as usize - 2));
  for value in random_values {
    let mut value_bits = scalar_to_bits_le(value);
    value_bits.truncate(Scalar::NUM_BITS as usize);
    if value_bits.len() > 2 {
      bit_stream.extend_from_slice(&value_bits[1..value_bits.len() - 1]);
    }
  }

  let bit_matrix = std::array::from_fn(|level| {
    std::array::from_fn(|i| {
      let bit_index = level * N + i;
      bit_stream.get(bit_index).copied().unwrap_or(false)
    })
  });

  (bit_matrix, num_samples)
}

fn prepare_rs_witness_trace_poseidon(seed: Scalar) -> (PermutationWitnessTrace<N, LEVELS>, usize) {
  let (bits_mat, num_samples) = derive_split_bits_poseidon_native(seed);

  let prev: [SortedRow; N] =
    std::array::from_fn(|i| SortedRow::new_with_bucket(i as u16, N as u16, 0));

  let level_results: Vec<(
    [spartan2::rs_shuffle_bp::data_structures::UnsortedRow; N],
    [SortedRow; N],
  )> = (0..LEVELS)
    .scan(prev, |prev_state, level| {
      let (unsorted, next_rows) = build_level::<N>(prev_state, &bits_mat[level]);
      *prev_state = next_rows;
      Some((unsorted, next_rows))
    })
    .collect();

  let uns_levels = std::array::from_fn(|i| level_results[i].0);
  let next_levels = std::array::from_fn(|i| level_results[i].1);

  (
    PermutationWitnessTrace {
      bits_mat,
      uns_levels,
      next_levels,
    },
    num_samples,
  )
}

fn build_shuffle_statement_and_witness(
  bases: &CommitmentBases,
) -> (ShuffleStatement<N>, ShuffleWitness<N, LEVELS>) {
  let sk = LinkerScalar::from(42u64);
  let nonce = Scalar::from(123u64);

  let pk = native_scalar_mul_maybe(sk, bases.generator.0, bases.generator.1, Scalar::ZERO)
    .expect("non-zero pk");

  let vrf_value = derive_vrf_value(nonce, sk);
  let seed_digest = seed_digest_from_vrf_value(vrf_value);
  let (witness_trace, _num_samples) = prepare_rs_witness_trace_poseidon(vrf_value);
  let permutation: [usize; N] =
    std::array::from_fn(|i| witness_trace.next_levels[LEVELS - 1][i].idx as usize);

  let final_indices: [LinkerScalar; N] =
    std::array::from_fn(|i| LinkerScalar::from(permutation[i] as u64));

  let mut perm_blinding = LinkerScalar::from(7u64);
  let mut power_blinding = LinkerScalar::from(11u64);
  let link_blinding = LinkerScalar::from(13u64);

  loop {
    let permutation_commitment = native_vector_commitment(
      &final_indices,
      &bases.perm_bases_affine,
      perm_blinding,
      &bases.perm_blind_base_affine,
    );
    let power_challenge = derive_power_challenge_from_commitment(permutation_commitment);
    let power_challenge_scalar =
      linker_scalar_from_le_bytes_mod_order(&scalar_to_le_bytes(power_challenge));
    if validate_power_distinctness::<LinkerScalar, N>(power_challenge_scalar).is_err() {
      perm_blinding += LinkerScalar::ONE;
      continue;
    }

    let mut x_powers = [LinkerScalar::ZERO; N];
    x_powers[0] = LinkerScalar::ONE;
    for i in 1..N {
      x_powers[i] = x_powers[i - 1] * power_challenge_scalar;
    }
    let power_perm_vec = std::array::from_fn(|i| x_powers[permutation[i]]);
    let power_commitment = native_vector_commitment(
      &power_perm_vec,
      &bases.power_bases_affine,
      power_blinding,
      &bases.power_blind_base_affine,
    );

    let challenges =
      match derive_linker_challenges::<N>(seed_digest, permutation_commitment, power_commitment) {
        Ok(challenges) => challenges,
        Err(_) => {
          power_blinding += LinkerScalar::ONE;
          continue;
        }
      };

    let link_value = compute_link_value_from_slice(&power_perm_vec, challenges.tau_scalar);
    if link_value == LinkerScalar::ZERO {
      power_blinding += LinkerScalar::ONE;
      continue;
    }
    let tau_powers = std::array::from_fn(|i| {
      let mut power = LinkerScalar::ONE;
      for _ in 0..i {
        power *= challenges.tau_scalar;
      }
      power
    });

    let link_commitment = native_link_commitment(
      link_value,
      link_blinding,
      &bases.link_base_affine,
      &bases.link_blind_base_affine,
    );

    let (curve_a, _, _, _) = <CurveEngine as Engine>::GE::group_params();
    let message_points = labeled_points(b"spartan2-bench-msg", N);
    let input_randomness = derive_sigma_scalars::<N>(seed_digest, b"input-ciphertexts");
    let input_ciphertexts = std::array::from_fn(|i| {
      let c1 = native_scalar_mul_maybe(
        input_randomness[i],
        bases.generator.0,
        bases.generator.1,
        curve_a,
      )
      .expect("input c1 must be non-infinity");
      let pk_term = native_scalar_mul_maybe(input_randomness[i], pk.0, pk.1, curve_a)
        .expect("input c2 pk term must be non-infinity");
      let c2 = native_add_points(message_points[i], pk_term);
      NativeCiphertext::new(c1.0, c1.1, c2.0, c2.1)
    });
    let rerandomization_scalars = derive_sigma_scalars::<N>(seed_digest, b"output-rerandomization");
    let output_ciphertexts = std::array::from_fn(|i| {
      rerandomize_ciphertext(
        &input_ciphertexts[permutation[i]],
        rerandomization_scalars[i],
        pk,
        bases.generator,
      )
    });

    let statement = ShuffleStatement {
      pk,
      nonce,
      seed_digest,
      power_challenge: challenges.power_challenge,
      tau_base: challenges.tau_base,
      permutation_commitment,
      power_commitment,
      link_commitment,
      tuple_expected_product: challenges.tuple_expected_product,
      input_ciphertexts,
      output_ciphertexts,
    };
    let witness = ShuffleWitness {
      sk,
      witness_trace,
      power_perm_vec,
      power_challenge_scalar: challenges.power_challenge_scalar,
      tau_scalar: challenges.tau_scalar,
      tau_powers,
      perm_blinding,
      power_blinding,
      link_value,
      link_blinding,
      rerandomization_scalars,
    };
    return (statement, witness);
  }
}

fn verify_row_constraints<F, CS, const N_INNER: usize>(
  mut cs: CS,
  unsorted: &[UnsortedRowVar<F>; N_INNER],
) -> Result<Vec<IndexPositionPair<F>>, SynthesisError>
where
  F: PrimeField,
  CS: ConstraintSystem<F>,
{
  let zero = alloc_constant(cs.namespace(|| "zero"), &F::ZERO)?;
  let one = alloc_constant(cs.namespace(|| "one"), &F::ONE)?;
  let mut idx_next_pos_pairs = Vec::with_capacity(N_INNER);

  for i in 0..N_INNER {
    let u = &unsorted[i];
    let u_prev = if i > 0 { Some(&unsorted[i - 1]) } else { None };
    let u_next = if i + 1 < N_INNER {
      Some(&unsorted[i + 1])
    } else {
      None
    };

    let bit_bool = Boolean::from(u.bit.clone());
    let bit_as_num =
      select_num_or_zero(cs.namespace(|| format!("bit_as_num_{i}")), &one, &bit_bool)?;
    let one_minus_bit = select_num_or_one(
      cs.namespace(|| format!("one_minus_bit_{i}")),
      &zero,
      &bit_bool,
    )?;

    let is_first_in_bucket = if let Some(prev) = u_prev {
      Boolean::from(alloc_num_equals(
        cs.namespace(|| format!("is_first_eq_{i}")),
        &prev.bucket_id,
        &u.bucket_id,
      )?)
      .not()
    } else {
      Boolean::constant(true)
    };

    let zero_check = conditionally_select(
      cs.namespace(|| format!("first_zero_check_{i}")),
      &u.num_zeros,
      &zero,
      &is_first_in_bucket,
    )?;
    enforce_equal_num(
      cs.namespace(|| format!("enforce_first_zero_{i}")),
      &zero_check,
      &zero,
    )?;

    let one_check = conditionally_select(
      cs.namespace(|| format!("first_one_check_{i}")),
      &u.num_ones,
      &zero,
      &is_first_in_bucket,
    )?;
    enforce_equal_num(
      cs.namespace(|| format!("enforce_first_one_{i}")),
      &one_check,
      &zero,
    )?;

    if let Some(next) = u_next {
      let same_bucket_bit = alloc_num_equals(
        cs.namespace(|| format!("same_bucket_{i}")),
        &u.bucket_id,
        &next.bucket_id,
      )?;
      let same_bucket = Boolean::from(same_bucket_bit.clone());

      let expected_next_zeros = add_nums(
        cs.namespace(|| format!("expected_next_zeros_{i}")),
        &u.num_zeros,
        &one_minus_bit,
      )?;
      let diff_zeros = sub_nums(
        cs.namespace(|| format!("diff_zeros_{i}")),
        &next.num_zeros,
        &expected_next_zeros,
      )?;
      let conditional_diff_zeros = conditionally_select(
        cs.namespace(|| format!("conditional_diff_zeros_{i}")),
        &diff_zeros,
        &zero,
        &same_bucket,
      )?;
      enforce_equal_num(
        cs.namespace(|| format!("enforce_diff_zeros_{i}")),
        &conditional_diff_zeros,
        &zero,
      )?;

      let expected_next_ones = add_nums(
        cs.namespace(|| format!("expected_next_ones_{i}")),
        &u.num_ones,
        &bit_as_num,
      )?;
      let diff_ones = sub_nums(
        cs.namespace(|| format!("diff_ones_{i}")),
        &next.num_ones,
        &expected_next_ones,
      )?;
      let conditional_diff_ones = conditionally_select(
        cs.namespace(|| format!("conditional_diff_ones_{i}")),
        &diff_ones,
        &zero,
        &same_bucket,
      )?;
      enforce_equal_num(
        cs.namespace(|| format!("enforce_diff_ones_{i}")),
        &conditional_diff_ones,
        &zero,
      )?;

      let total_zeros_diff = sub_nums(
        cs.namespace(|| format!("total_zeros_diff_{i}")),
        &u.total_zeros_in_bucket,
        &next.total_zeros_in_bucket,
      )?;
      let conditional_total_zeros_diff = conditionally_select(
        cs.namespace(|| format!("conditional_total_zeros_diff_{i}")),
        &total_zeros_diff,
        &zero,
        &same_bucket,
      )?;
      enforce_equal_num(
        cs.namespace(|| format!("enforce_total_zeros_diff_{i}")),
        &conditional_total_zeros_diff,
        &zero,
      )?;

      let length_diff = sub_nums(
        cs.namespace(|| format!("length_diff_{i}")),
        &u.bucket_length,
        &next.bucket_length,
      )?;
      let conditional_length_diff = conditionally_select(
        cs.namespace(|| format!("conditional_length_diff_{i}")),
        &length_diff,
        &zero,
        &same_bucket,
      )?;
      enforce_equal_num(
        cs.namespace(|| format!("enforce_length_diff_{i}")),
        &conditional_length_diff,
        &zero,
      )?;
    }

    let is_last_in_bucket = if let Some(next) = u_next {
      Boolean::from(alloc_num_equals(
        cs.namespace(|| format!("is_last_eq_{i}")),
        &u.bucket_id,
        &next.bucket_id,
      )?)
      .not()
    } else {
      Boolean::constant(true)
    };

    let zeros_plus = add_nums(
      cs.namespace(|| format!("zeros_plus_{i}")),
      &u.num_zeros,
      &one_minus_bit,
    )?;
    let zeros_tally_diff = sub_nums(
      cs.namespace(|| format!("zeros_tally_diff_{i}")),
      &zeros_plus,
      &u.total_zeros_in_bucket,
    )?;
    let conditional_zeros_tally = conditionally_select(
      cs.namespace(|| format!("conditional_zeros_tally_{i}")),
      &zeros_tally_diff,
      &zero,
      &is_last_in_bucket,
    )?;
    enforce_equal_num(
      cs.namespace(|| format!("enforce_zeros_tally_{i}")),
      &conditional_zeros_tally,
      &zero,
    )?;

    let expected_total_ones = sub_nums(
      cs.namespace(|| format!("expected_total_ones_{i}")),
      &u.bucket_length,
      &u.total_zeros_in_bucket,
    )?;
    let ones_plus = add_nums(
      cs.namespace(|| format!("ones_plus_{i}")),
      &u.num_ones,
      &bit_as_num,
    )?;
    let ones_tally_diff = sub_nums(
      cs.namespace(|| format!("ones_tally_diff_{i}")),
      &ones_plus,
      &expected_total_ones,
    )?;
    let conditional_ones_tally = conditionally_select(
      cs.namespace(|| format!("conditional_ones_tally_{i}")),
      &ones_tally_diff,
      &zero,
      &is_last_in_bucket,
    )?;
    enforce_equal_num(
      cs.namespace(|| format!("enforce_ones_tally_{i}")),
      &conditional_ones_tally,
      &zero,
    )?;

    let pos = alloc_constant(cs.namespace(|| format!("pos_{i}")), &F::from(i as u64))?;
    let sum_counts = add_nums(
      cs.namespace(|| format!("sum_counts_{i}")),
      &u.num_zeros,
      &u.num_ones,
    )?;
    let base = sub_nums(cs.namespace(|| format!("base_{i}")), &pos, &sum_counts)?;
    let zeros_gap = sub_nums(
      cs.namespace(|| format!("zeros_gap_{i}")),
      &u.total_zeros_in_bucket,
      &u.num_zeros,
    )?;
    let inner = add_nums(
      cs.namespace(|| format!("inner_{i}")),
      &zeros_gap,
      &u.num_ones,
    )?;
    let bit_times_inner =
      bit_as_num.mul(cs.namespace(|| format!("bit_times_inner_{i}")), &inner)?;
    let offset = add_nums(
      cs.namespace(|| format!("offset_{i}")),
      &u.num_zeros,
      &bit_times_inner,
    )?;
    let expected_dest = add_nums(
      cs.namespace(|| format!("expected_dest_{i}")),
      &base,
      &offset,
    )?;
    enforce_equal_num(
      cs.namespace(|| format!("enforce_next_pos_{i}")),
      &u.next_pos,
      &expected_dest,
    )?;

    idx_next_pos_pairs.push(IndexPositionPair::new(u.idx.clone(), u.next_pos.clone()));
  }

  Ok(idx_next_pos_pairs)
}

fn verify_shuffle_level<F, CS, const N_INNER: usize>(
  mut cs: CS,
  unsorted: &[UnsortedRowVar<F>; N_INNER],
  sorted: &[SortedRowVar<F>; N_INNER],
  alpha: &AllocatedNum<F>,
  beta: &AllocatedNum<F>,
) -> Result<(), SynthesisError>
where
  F: PrimeField,
  CS: ConstraintSystem<F>,
{
  let idx_next_pos_pairs = verify_row_constraints(cs.namespace(|| "row_constraints"), unsorted)?;

  let idx_pos_pairs: Vec<IndexPositionPair<F>> = sorted
    .iter()
    .enumerate()
    .map(|(j, sr)| {
      let pos = alloc_constant(
        cs.namespace(|| format!("sorted_pos_{j}")),
        &F::from(j as u64),
      )?;
      Ok(IndexPositionPair::new(sr.idx.clone(), pos))
    })
    .collect::<Result<Vec<_>, SynthesisError>>()?;

  check_grand_product::<F, IndexPositionPair<F>, _, 2>(
    cs.namespace(|| "grand_product"),
    &idx_next_pos_pairs,
    &idx_pos_pairs,
    &[alpha.clone(), beta.clone()],
  )?;

  Ok(())
}

impl FullRSShuffleCircuit {
  fn new(
    statement: ShuffleStatement<N>,
    witness: ShuffleWitness<N, LEVELS>,
    bases: CommitmentBases,
  ) -> Self {
    Self {
      statement,
      witness,
      bases,
    }
  }
}

impl SpartanCircuit<PallasHyraxEngine> for FullRSShuffleCircuit {
  fn public_values(&self) -> Result<Vec<Scalar>, SynthesisError> {
    Ok(encode_spartan_public_values(&self.statement))
  }

  fn shared<CS: ConstraintSystem<Scalar>>(
    &self,
    _cs: &mut CS,
  ) -> Result<Vec<AllocatedNum<Scalar>>, SynthesisError> {
    Ok(vec![])
  }

  fn precommitted<CS: ConstraintSystem<Scalar>>(
    &self,
    _cs: &mut CS,
    _shared: &[AllocatedNum<Scalar>],
  ) -> Result<Vec<AllocatedNum<Scalar>>, SynthesisError> {
    Ok(vec![])
  }

  fn num_challenges(&self) -> usize {
    0
  }

  fn synthesize<CS: ConstraintSystem<Scalar>>(
    &self,
    cs: &mut CS,
    _shared: &[AllocatedNum<Scalar>],
    _precommitted: &[AllocatedNum<Scalar>],
    _challenges: Option<&[Scalar]>,
  ) -> Result<(), SynthesisError> {
    let public_pk =
      alloc_point_public_input::<CurveEngine, _>(cs.namespace(|| "pk"), self.statement.pk)?;
    let nonce_public =
      AllocatedNum::alloc_input(cs.namespace(|| "nonce"), || Ok(self.statement.nonce))?;
    let initial_indices_public: Vec<AllocatedNum<Scalar>> = (0..N)
      .map(|i| {
        AllocatedNum::alloc_input(cs.namespace(|| format!("idx_init_{i}")), || {
          Ok(Scalar::from(i as u64))
        })
      })
      .collect::<Result<Vec<_>, _>>()?;
    let seed_digest_public = AllocatedNum::alloc_input(cs.namespace(|| "seed_digest"), || {
      Ok(self.statement.seed_digest)
    })?;
    let power_challenge_public =
      AllocatedNum::alloc_input(cs.namespace(|| "x"), || Ok(self.statement.power_challenge))?;
    let tau_base_public =
      AllocatedNum::alloc_input(cs.namespace(|| "tau_base"), || Ok(self.statement.tau_base))?;
    let permutation_commitment_public = alloc_point_public_input::<CurveEngine, _>(
      cs.namespace(|| "C_perm"),
      self.statement.permutation_commitment,
    )?;
    let power_commitment_public = alloc_point_public_input::<CurveEngine, _>(
      cs.namespace(|| "C_power"),
      self.statement.power_commitment,
    )?;
    let link_commitment_public = alloc_point_public_input::<CurveEngine, _>(
      cs.namespace(|| "C_link"),
      self.statement.link_commitment,
    )?;
    let tuple_expected_public = AllocatedNum::alloc_input(cs.namespace(|| "P_graph"), || {
      Ok(self.statement.tuple_expected_product)
    })?;

    let witness_var = PermutationWitnessTraceVar::<Scalar, N, LEVELS>::alloc(
      cs.namespace(|| "witness"),
      &self.witness.witness_trace,
    )?;

    for (i, (expected, row)) in initial_indices_public
      .iter()
      .zip(witness_var.uns_levels[0].iter())
      .enumerate()
    {
      enforce_equal_num(
        cs.namespace(|| format!("enforce_initial_idx_{i}")),
        &row.idx,
        expected,
      )?;
    }

    let sk_witness =
      EmulatedLinkerVar::alloc_witness(cs.namespace(|| "sk"), Some(self.witness.sk))?;
    let power_challenge_scalar_wit = EmulatedLinkerVar::alloc_noncanonical_witness(
      cs.namespace(|| "x_scalar"),
      Some(self.witness.power_challenge_scalar),
    )?;
    let tau_scalar_wit = EmulatedLinkerVar::alloc_noncanonical_witness(
      cs.namespace(|| "tau_scalar"),
      Some(self.witness.tau_scalar),
    )?;
    let power_perm_vec_wit: Vec<EmulatedLinkerVar> = self
      .witness
      .power_perm_vec
      .iter()
      .enumerate()
      .map(|(i, value)| {
        EmulatedLinkerVar::alloc_noncanonical_witness(
          cs.namespace(|| format!("b_{i}")),
          Some(*value),
        )
      })
      .collect::<Result<Vec<_>, _>>()?;
    let link_value_wit = EmulatedLinkerVar::alloc_noncanonical_witness(
      cs.namespace(|| "link_value"),
      Some(self.witness.link_value),
    )?;
    let link_blinding_wit = EmulatedLinkerVar::alloc_noncanonical_witness(
      cs.namespace(|| "link_blinding"),
      Some(self.witness.link_blinding),
    )?;

    let generator =
      alloc_point_constant::<CurveEngine, _>(cs.namespace(|| "generator"), self.bases.generator)?;
    let pk_computed = generator.scalar_mul_fixed_base(
      cs.namespace(|| "pk_check"),
      &sk_witness.bits[..LinkerScalar::NUM_BITS as usize],
      &self.bases.generator_powers,
    )?;
    public_pk.enforce_equal(cs.namespace(|| "pk_enforce"), &pk_computed)?;

    let mut vrf_sponge = PoseidonSpongeCircuit::new(cs, "vrf");
    vrf_sponge.absorb_allocated_field(cs, &nonce_public)?;
    let sk_byte_fields = sk_witness.byte_fields(cs.namespace(|| "sk_bytes"))?;
    vrf_sponge.absorb_allocated_fields(cs, &sk_byte_fields)?;
    let vrf_value = vrf_sponge.squeeze_field_elements(cs, 1)?[0].clone();

    let mut seed_sponge = PoseidonSpongeCircuit::new(cs, "seed");
    seed_sponge.absorb_allocated_field(cs, &vrf_value)?;
    let derived_seed_digest = seed_sponge.squeeze_field_elements(cs, 1)?[0].clone();
    enforce_equal_num(
      cs.namespace(|| "seed_digest_match"),
      &derived_seed_digest,
      &seed_digest_public,
    )?;

    let num_samples = rs_num_samples();
    let mut rs_sponge = PoseidonSpongeCircuit::new(cs, "rs_bits");
    rs_sponge.absorb_allocated_field(cs, &vrf_value)?;
    let random_values = rs_sponge.squeeze_field_elements(cs, num_samples)?;
    let mut bit_stream = Vec::new();
    for (sample_idx, value) in random_values.iter().enumerate() {
      let bits = value.to_bits_le_strict(cs.namespace(|| format!("sample_bits_{sample_idx}")))?;
      if bits.len() > 2 {
        bit_stream.extend_from_slice(&bits[1..bits.len() - 1]);
      }
    }

    for level in 0..LEVELS {
      for i in 0..N {
        let derived = bit_stream[level * N + i].clone();
        Boolean::enforce_equal(
          cs.namespace(|| format!("bind_rs_bit_{level}_{i}")),
          &Boolean::from(witness_var.bits_mat[level][i].clone()),
          &derived,
        )?;
      }
    }

    let mut rs_transcript = PoseidonSpongeCircuit::new(cs, "rs_transcript");
    rs_transcript.absorb_point(cs, &public_pk)?;
    rs_transcript.absorb_allocated_field(cs, &nonce_public)?;
    rs_transcript.absorb_allocated_field(cs, &power_challenge_public)?;
    rs_transcript.absorb_point(cs, &permutation_commitment_public)?;
    rs_transcript.absorb_point(cs, &power_commitment_public)?;
    let perm_alpha = rs_transcript.squeeze_field_elements(cs, 1)?[0].clone();
    let perm_beta = perm_alpha.mul(cs.namespace(|| "perm_beta"), &perm_alpha)?;

    for level in 0..LEVELS {
      verify_shuffle_level(
        cs.namespace(|| format!("verify_level_{level}")),
        &witness_var.uns_levels[level],
        &witness_var.sorted_levels[level],
        &perm_alpha,
        &perm_beta,
      )?;
    }

    let final_indices: Vec<AllocatedNum<Scalar>> = witness_var.sorted_levels[LEVELS - 1]
      .iter()
      .map(|row| row.idx.clone())
      .collect();

    check_grand_product::<Scalar, AllocatedNum<Scalar>, _, 1>(
      cs.namespace(|| "final_multiset_check"),
      &initial_indices_public,
      &final_indices,
      std::slice::from_ref(&perm_alpha)
        .try_into()
        .expect("single challenge"),
    )?;

    let mut link_transcript = PoseidonSpongeCircuit::new(cs, "link_transcript");
    link_transcript.absorb_bytes_constant(cs, b"permutation-proof")?;
    link_transcript.absorb_point(cs, &permutation_commitment_public)?;
    link_transcript.absorb_bytes_constant(cs, b"power-challenge")?;
    let x_squeezed = link_transcript.squeeze_field_elements(cs, 1)?;
    let x_from_commit = replace_zero_with_one(cs.namespace(|| "x_zero_fix"), &x_squeezed[0])?;
    enforce_equal_num(
      cs.namespace(|| "x_match"),
      &x_from_commit,
      &power_challenge_public,
    )?;
    enforce_base_to_emulated(
      cs.namespace(|| "x_base_to_scalar"),
      &power_challenge_public,
      self.statement.power_challenge,
      &power_challenge_scalar_wit,
    )?;
    link_transcript.absorb_allocated_field(cs, &seed_digest_public)?;
    link_transcript.absorb_point(cs, &power_commitment_public)?;
    link_transcript.absorb_bytes_constant(cs, b"tau-challenge")?;
    let tau_squeezed = link_transcript.squeeze_field_elements(cs, 1)?;
    let tau_from_commit = replace_zero_with_one(cs.namespace(|| "tau_zero_fix"), &tau_squeezed[0])?;
    enforce_equal_num(
      cs.namespace(|| "tau_base_match"),
      &tau_from_commit,
      &tau_base_public,
    )?;
    enforce_base_to_emulated(
      cs.namespace(|| "tau_base_to_scalar"),
      &tau_base_public,
      self.statement.tau_base,
      &tau_scalar_wit,
    )?;
    link_transcript.absorb_bytes_constant(cs, b"tuple-challenges")?;
    let tuple_challenges = link_transcript.squeeze_field_elements(cs, EMULATED_LIMBS + 2)?;
    let tuple_rho = tuple_challenges[0].clone();
    let tuple_index_coeff = tuple_challenges[1].clone();
    let tuple_limb_coeffs = tuple_challenges[2..].to_vec();

    let mut tuple_product = alloc_constant(cs.namespace(|| "tuple_prod_init"), &Scalar::ONE)?;
    for (i, (pi_i, b_i)) in final_indices
      .iter()
      .zip(power_perm_vec_wit.iter())
      .enumerate()
    {
      let encoded_idx = tuple_index_coeff.mul(cs.namespace(|| format!("enc_idx_{i}")), pi_i)?;
      let mut encoded = encoded_idx;
      for (limb_idx, (coeff, limb)) in tuple_limb_coeffs.iter().zip(b_i.limbs.iter()).enumerate() {
        let scaled_limb = coeff.mul(cs.namespace(|| format!("enc_limb_{i}_{limb_idx}")), limb)?;
        encoded = add_nums(
          cs.namespace(|| format!("encoded_{i}_{limb_idx}")),
          &encoded,
          &scaled_limb,
        )?;
      }
      let factor = sub_nums(cs.namespace(|| format!("factor_{i}")), &tuple_rho, &encoded)?;
      tuple_product = tuple_product.mul(cs.namespace(|| format!("tuple_prod_{i}")), &factor)?;
    }
    enforce_equal_num(
      cs.namespace(|| "tuple_product_match"),
      &tuple_product,
      &tuple_expected_public,
    )?;

    enforce_constant_link_sum(
      cs.namespace(|| "link_sum"),
      &power_perm_vec_wit,
      &self.witness.tau_powers,
      &link_value_wit,
    )?;

    let link_base =
      alloc_point_constant::<CurveEngine, _>(cs.namespace(|| "link_base"), self.bases.link_base)?;
    let blind_base = alloc_point_constant::<CurveEngine, _>(
      cs.namespace(|| "blind_base"),
      self.bases.link_blind_base,
    )?;

    let l_term = link_base.scalar_mul_fixed_base(
      cs.namespace(|| "L_times_G"),
      &link_value_wit.bits[..LinkerScalar::NUM_BITS as usize],
      &self.bases.link_base_powers,
    )?;
    let r_term = blind_base.scalar_mul_fixed_base(
      cs.namespace(|| "r_times_H"),
      &link_blinding_wit.bits[..LinkerScalar::NUM_BITS as usize],
      &self.bases.link_blind_powers,
    )?;

    let zero_inf = alloc_zero(cs.namespace(|| "zero_infinity"));
    let l_term_complete: AllocatedPoint<CurveEngine> = l_term.to_allocated_point(&zero_inf)?;
    let r_term_complete: AllocatedPoint<CurveEngine> = r_term.to_allocated_point(&zero_inf)?;
    let commitment_sum =
      l_term_complete.add(cs.namespace(|| "commitment_sum"), &r_term_complete)?;
    enforce_equal_num(
      cs.namespace(|| "commitment_not_infinity"),
      &commitment_sum.is_infinity,
      &zero_inf,
    )?;
    let commitment_sum = AllocatedPointNonInfinity::from_allocated_point(&commitment_sum);
    commitment_sum.enforce_equal(cs.namespace(|| "commitment_match"), &link_commitment_public)?;

    Ok(())
  }
}

fn metric_group(name: &str) -> String {
  let top = name.split('/').next().unwrap_or(name);
  if top.starts_with("verify_level_") {
    "rs_levels".to_string()
  } else if top == "final_multiset_check" {
    "rs_final_multiset".to_string()
  } else if top == "witness" {
    "rs_witness_alloc".to_string()
  } else if top.starts_with("rs_bits")
    || top.starts_with("sample_bits_")
    || top.starts_with("bind_rs_bit_")
  {
    "rs_bit_binding".to_string()
  } else if top == "rs_transcript" {
    "rs_transcript".to_string()
  } else if top == "pk_check" || top == "pk_enforce" || top == "generator" {
    "vrf_pk".to_string()
  } else if top.starts_with("vrf") || top == "sk_bytes" {
    "vrf_sponge".to_string()
  } else if top.starts_with("seed") {
    "seed_digest".to_string()
  } else if top == "x_base_to_scalar" || top == "tau_base_to_scalar" {
    "base_to_scalar".to_string()
  } else if top == "link_transcript" {
    "link_transcript".to_string()
  } else if top.starts_with("tuple_")
    || top.starts_with("enc_")
    || top.starts_with("factor_")
    || top.starts_with("encoded_")
  {
    "tuple_graph".to_string()
  } else if top.starts_with("horner_") || top.starts_with("link_sum") {
    "linker_sum".to_string()
  } else if top == "link_base"
    || top == "blind_base"
    || top == "L_times_G"
    || top == "r_times_H"
    || top.starts_with("commitment_")
  {
    "link_commitment".to_string()
  } else if top == "sk"
    || top == "x_scalar"
    || top == "tau_scalar"
    || top == "link_value"
    || top == "link_blinding"
    || top.starts_with("b_")
  {
    "emulated_alloc".to_string()
  } else {
    top.to_string()
  }
}

fn log_sorted_counts(title: &str, counts: BTreeMap<String, usize>, limit: usize) {
  let mut entries: Vec<(String, usize)> = counts.into_iter().collect();
  entries.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
  println!("{title}");
  for (name, count) in entries.into_iter().take(limit) {
    println!("  {:>6}  {}", count, name);
  }
}

fn log_subprefix_counts(entries: &[String], prefix: &str, limit: usize) {
  let mut counts = BTreeMap::<String, usize>::new();
  for entry in entries {
    if entry.starts_with(prefix) {
      let bucket = entry.split('/').take(2).collect::<Vec<_>>().join("/");
      *counts.entry(bucket).or_default() += 1;
    }
  }
  log_sorted_counts(&format!("  Sub-breakdown for `{prefix}`:"), counts, limit);
}

fn log_constraint_profile(circuit: &FullRSShuffleCircuit) {
  let mut metric_cs = MetricCS::<Scalar>::new();
  circuit
    .clone()
    .synthesize(&mut metric_cs, &[], &[], None)
    .expect("metric synthesize failed");

  let entries = metric_cs.pretty_print_list();
  let mut grouped = BTreeMap::<String, usize>::new();
  let mut top_level = BTreeMap::<String, usize>::new();
  for entry in &entries {
    if entry.starts_with("INPUT ") || entry.starts_with("AUX ") {
      continue;
    }
    *grouped.entry(metric_group(&entry)).or_default() += 1;
    *top_level
      .entry(entry.split('/').next().unwrap_or(&entry).to_string())
      .or_default() += 1;
  }

  println!("\nConstraint profile:");
  println!("  Total constraints: {}", metric_cs.num_constraints());
  log_sorted_counts("  Grouped buckets:", grouped, 32);
  log_sorted_counts("  Top-level namespaces:", top_level, 64);
  log_subprefix_counts(&entries, "b_0/", 32);
  log_subprefix_counts(&entries, "link_sum/", 32);
}

fn main() {
  let cli = Cli::parse();
  let _ = tracing_subscriber::fmt()
    .with_target(false)
    .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
    .try_init();

  println!("╔══════════════════════════════════════════════════════════════╗");
  println!("║   Bellpepper March-25 Permutation / Linker Relation Bench   ║");
  println!("╠══════════════════════════════════════════════════════════════╣");
  println!(
    "║  N = {} cards, LEVELS = {}                                   ║",
    N, LEVELS
  );
  println!(
    "║  SNARK mode = {:<11}                                      ║",
    match cli.snark {
      SpartanBenchChoice::Both => "both",
      SpartanBenchChoice::Spartan => "spartan",
      SpartanBenchChoice::PpSpartan => "ppspartan",
    }
  );
  println!("╚══════════════════════════════════════════════════════════════╝\n");

  let bases = COMMITMENT_BASES.clone();
  let (sample_statement, sample_witness) = build_shuffle_statement_and_witness(&bases);
  validate_shared_shuffle_relations(&sample_statement, &sample_witness, &bases);
  let circuit = FullRSShuffleCircuit::new(
    sample_statement.clone(),
    sample_witness.clone(),
    bases.clone(),
  );

  println!("Checking satisfiability...");
  let mut test_cs = TestConstraintSystem::<Scalar>::new();
  circuit
    .clone()
    .synthesize(&mut test_cs, &[], &[], None)
    .expect("test synthesize failed");
  assert!(
    test_cs.is_satisfied(),
    "bellpepper unsatisfied at {:?}",
    test_cs.which_is_unsatisfied()
  );
  println!("  Bellpepper test CS is satisfied");
  log_constraint_profile(&circuit);

  let mut shape_cs = TestShapeCS::<PallasHyraxEngine>::new();
  circuit
    .clone()
    .synthesize(&mut shape_cs, &[], &[], None)
    .expect("shape synthesize failed");
  let (shape, ck, _vk) = shape_cs.r1cs_shape().expect("shape extraction failed");
  let mut witness_cs = SatisfyingAssignment::<PallasHyraxEngine>::new();
  circuit
    .clone()
    .synthesize(&mut witness_cs, &[], &[], None)
    .expect("witness synthesize failed");
  let (instance_sat, witness_sat) = witness_cs
    .r1cs_instance_and_witness(&shape, &ck, false)
    .expect("witness extraction failed");
  shape
    .is_sat(&ck, &instance_sat, &witness_sat)
    .expect("native witness does not satisfy circuit");
  println!("  Circuit is satisfied");

  println!("\nPreparing proving instance...");
  let witness_generation_start = Instant::now();
  let (proving_statement, proving_witness) = build_shuffle_statement_and_witness(&bases);
  let proving_circuit = FullRSShuffleCircuit::new(
    proving_statement.clone(),
    proving_witness.clone(),
    bases.clone(),
  );
  let witness_generation_time = witness_generation_start.elapsed();
  println!("  Witness + commitments: {:?}", witness_generation_time);
  let equivalence_check_start = Instant::now();
  validate_shared_shuffle_relations(&proving_statement, &proving_witness, &bases);
  let equivalence_check_time = equivalence_check_start.elapsed();
  println!("  Equivalence checks: {:?}", equivalence_check_time);

  let baseline_summary = if cli.snark.runs_spartan() {
    println!("\nRunning setup...");
    let setup_start = Instant::now();
    let (pk, vk) = SpartanSNARK::<PallasHyraxEngine>::setup(circuit.clone()).expect("setup failed");
    let setup_time = setup_start.elapsed();

    let sizes = pk.sizes();
    println!("  Setup time: {:?}", setup_time);
    println!("  Constraints (unpadded): {}", sizes[0]);
    println!("  Constraints (padded):   {}", sizes[4]);
    println!("  Variables (shared):     {}", sizes[5]);
    println!("  Variables (precommit):  {}", sizes[6]);
    println!("  Variables (rest):       {}", sizes[7]);

    println!("\nRunning prep_prove...");
    let prep_start = Instant::now();
    let prep =
      ShuffleSnark::prep_prove(&pk, proving_circuit.clone(), false).expect("prep_prove failed");
    let prep_time = prep_start.elapsed();
    println!("  Prep time: {:?}", prep_time);

    println!("\nRunning parallel prove (Spartan + native BG/Sigma)...");
    let parallel_prove_start = Instant::now();
    let (proof, sigma_proof, prove_time, sigma_prove_time) = std::thread::scope(|scope| {
      let sigma_handle = scope.spawn(|| {
        let sigma_prove_start = Instant::now();
        let sigma_proof = prove_native_sigma::<N>(&proving_statement, &proving_witness, &bases);
        (sigma_proof, sigma_prove_start.elapsed())
      });

      let prove_start = Instant::now();
      let proof =
        ShuffleSnark::prove(&pk, proving_circuit.clone(), &prep, false).expect("prove failed");
      let prove_time = prove_start.elapsed();

      let (sigma_proof, sigma_prove_time) =
        sigma_handle.join().expect("sigma prove thread panicked");
      (proof, sigma_proof, prove_time, sigma_prove_time)
    });
    let shuffle_proof = ShuffleProof {
      spartan_proof: proof,
      sigma_proof,
    };
    let spartan_proof_size = bincode::serialize(&shuffle_proof.spartan_proof)
      .expect("baseline proof serialization failed")
      .len();
    let parallel_prove_time = parallel_prove_start.elapsed();
    let combined_prove_time = witness_generation_time + prep_time + parallel_prove_time;
    println!("  Spartan prove time: {:?}", prove_time);
    println!("  Sigma prove time: {:?}", sigma_prove_time);
    println!("  Combined parallel prove time: {:?}", parallel_prove_time);
    println!("  Spartan proof size: {} bytes", spartan_proof_size);

    println!("\nRunning verify...");
    let verify_start = Instant::now();
    let result =
      verify_spartan_statement_binding(&vk, &proving_statement, &shuffle_proof.spartan_proof);
    let verify_time = verify_start.elapsed();

    match result {
      Ok(public_outputs) => {
        println!("  Spartan proof verified successfully");
        println!("  Verify time: {:?}", verify_time);
        println!("  Public outputs: {} values", public_outputs.len());
      }
      Err(err) => {
        println!("  Verification failed: {:?}", err);
        std::process::exit(1);
      }
    }

    println!("\nRunning native BG/Sigma verify...");
    let sigma_verify_start = Instant::now();
    let sigma_ok =
      verify_native_sigma_public(&proving_statement, &shuffle_proof.sigma_proof, &bases);
    let sigma_verify_time = sigma_verify_start.elapsed();
    if let Err(err) = sigma_ok {
      println!("  Sigma verification failed: {err}");
      std::process::exit(1);
    }
    println!("  Sigma verification succeeded");
    println!("  Sigma verify time: {:?}", sigma_verify_time);

    let combined_verify_time = verify_time + sigma_verify_time;

    if let Err(err) = verify_shuffle_proof(&vk, &proving_statement, &shuffle_proof, &bases) {
      println!("  Combined verifier failed: {err}");
      std::process::exit(1);
    }

    Some(BaselineBenchSummary {
      constraints: sizes[0],
      setup_time,
      witness_generation_time,
      equivalence_check_time,
      prep_time,
      prove_time,
      sigma_prove_time,
      combined_prove_time,
      verify_time,
      sigma_verify_time,
      combined_verify_time,
      proof_size: spartan_proof_size,
    })
  } else {
    None
  };

  let pp_summary = if cli.snark.runs_ppspartan() {
    println!("\nRunning preprocessing Spartan setup...");
    let pp_setup_start = Instant::now();
    let (pp_pk, pp_vk) = PpShuffleSnark::setup(proving_circuit.clone()).expect("pp setup failed");
    let pp_setup_time = pp_setup_start.elapsed();
    let pp_sizes = PpShuffleSnark::pk_sizes(&pp_pk);
    println!("  Pp setup time: {:?}", pp_setup_time);
    println!("  Pp constraints (unpadded): {}", pp_sizes[0]);
    println!("  Pp constraints (padded):   {}", pp_sizes[4]);
    println!("  Pp variables (shared):     {}", pp_sizes[5]);
    println!("  Pp variables (precommit):  {}", pp_sizes[6]);
    println!("  Pp variables (rest):       {}", pp_sizes[7]);

    println!("\nRunning preprocessing prep_prove...");
    let pp_prep_start = Instant::now();
    let pp_prep =
      PpShuffleSnark::prep_prove(&pp_pk, proving_circuit.clone(), false).expect("pp prep failed");
    let pp_prep_time = pp_prep_start.elapsed();
    println!("  Pp prep time: {:?}", pp_prep_time);

    println!("\nRunning preprocessing prove...");
    let pp_prove_start = Instant::now();
    let pp_proof = PpShuffleSnark::prove(&pp_pk, proving_circuit.clone(), &pp_prep, false)
      .expect("pp prove failed");
    let pp_prove_time = pp_prove_start.elapsed();
    let pp_proof_size = bincode::serialize(&pp_proof)
      .expect("pp proof serialization failed")
      .len();
    println!("  Pp prove time: {:?}", pp_prove_time);
    println!("  Pp proof size: {} bytes", pp_proof_size);

    println!("\nRunning preprocessing verify...");
    let pp_verify_start = Instant::now();
    let pp_result = verify_spartan_statement_binding_generic::<PpShuffleSnark, N>(
      &pp_vk,
      &proving_statement,
      &pp_proof,
    );
    let pp_verify_time = pp_verify_start.elapsed();
    match pp_result {
      Ok(public_outputs) => {
        println!("  Pp Spartan proof verified successfully");
        println!("  Pp verify time: {:?}", pp_verify_time);
        println!("  Pp public outputs: {} values", public_outputs.len());
      }
      Err(err) => {
        println!("  Pp verification failed: {:?}", err);
        std::process::exit(1);
      }
    }

    Some(PpBenchSummary {
      constraints: pp_sizes[0],
      setup_time: pp_setup_time,
      prep_time: pp_prep_time,
      prove_time: pp_prove_time,
      verify_time: pp_verify_time,
      proof_size: pp_proof_size,
    })
  } else {
    None
  };

  println!("\n╔══════════════════════════════════════════════════════════════╗");
  println!("║                        SUMMARY                               ║");
  println!("╠══════════════════════════════════════════════════════════════╣");
  if let Some(summary) = &baseline_summary {
    println!(
      "║  Spartan constraints: {:>8}                            ║",
      summary.constraints
    );
    println!(
      "║  Spartan setup:       {:>8.2?}                          ║",
      summary.setup_time
    );
    println!(
      "║  Spartan prove:       {:>8.2?}                          ║",
      summary.witness_generation_time + summary.prep_time + summary.prove_time
    );
    println!(
      "║  Spartan equiv:       {:>8.2?}                          ║",
      summary.equivalence_check_time
    );
    println!(
      "║  Spartan verify:      {:>8.2?}                          ║",
      summary.verify_time
    );
    println!(
      "║  Spartan sigma prove: {:>8.2?}                          ║",
      summary.sigma_prove_time
    );
    println!(
      "║  Spartan sigma verify:{:>8.2?}                          ║",
      summary.sigma_verify_time
    );
    println!(
      "║  Spartan combined:    {:>8.2?}                          ║",
      summary.combined_prove_time
    );
    println!(
      "║  Spartan total verify:{:>8.2?}                          ║",
      summary.combined_verify_time
    );
    println!(
      "║  Spartan proof size:  {:>8} bytes                     ║",
      summary.proof_size
    );
    if pp_summary.is_some() {
      println!("╠══════════════════════════════════════════════════════════════╣");
    }
  }
  if let Some(summary) = &pp_summary {
    println!(
      "║  Pp constraints:      {:>8}                            ║",
      summary.constraints
    );
    println!(
      "║  Pp setup:            {:>8.2?}                          ║",
      summary.setup_time
    );
    println!(
      "║  Pp prep:             {:>8.2?}                          ║",
      summary.prep_time
    );
    println!(
      "║  Pp prove:            {:>8.2?}                          ║",
      summary.prove_time
    );
    println!(
      "║  Pp verify:           {:>8.2?}                          ║",
      summary.verify_time
    );
    println!(
      "║  Pp proof size:       {:>8} bytes                     ║",
      summary.proof_size
    );
  }
  println!("╚══════════════════════════════════════════════════════════════╝");
}
