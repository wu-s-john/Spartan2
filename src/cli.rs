// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! CLI utilities for examples and benchmarks.

use clap::ValueEnum;

/// Field choice for benchmarks
#[derive(ValueEnum, Clone, Default, Debug)]
pub enum FieldChoice {
  /// Pallas curve scalar field (Fq)
  #[default]
  PallasFq,
  /// Vesta curve scalar field (Fp)
  VestaFp,
  /// BN254 curve scalar field (Fr)
  Bn254Fr,
}

/// Which Spartan implementation to run in examples and benchmarks
#[derive(ValueEnum, Clone, Copy, Default, Debug, PartialEq, Eq)]
pub enum SpartanBenchChoice {
  /// Run both the baseline Spartan and preprocessing Spartan paths
  #[default]
  #[value(name = "both")]
  Both,
  /// Run only the baseline Spartan path
  #[value(name = "spartan")]
  Spartan,
  /// Run only the preprocessing Spartan path
  #[value(name = "ppspartan", alias = "pp-spartan")]
  PpSpartan,
}

impl SpartanBenchChoice {
  /// Whether this choice includes the baseline Spartan path
  pub fn runs_spartan(self) -> bool {
    matches!(self, Self::Both | Self::Spartan)
  }

  /// Whether this choice includes the preprocessing Spartan path
  pub fn runs_ppspartan(self) -> bool {
    matches!(self, Self::Both | Self::PpSpartan)
  }
}
