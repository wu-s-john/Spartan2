# Spartan ZK Verifier Circuit — Design Specification

## Overview

This document specifies the mathematical constraints for a Spartan ZK verifier circuit
split across a cycle of elliptic curves (Pallas/Vesta). The circuit verifies a Spartan ZK
proof **in-circuit**, deferring only the PCS opening proof and the matrix evaluation to the
native verifier.

### Architecture

```
Primary Circuit C₁ (over F_q = Pallas scalar field):
  - Spartan sum-check verification
  - Fiat-Shamir challenge derivation (in-circuit Poseidon)
  - τ̂, eval_X computation
  - NIFS scalar fold verification
  - Relaxed R1CS satisfiability of folded verifier circuit

Secondary Circuit C₂ (over F_p = Vesta scalar field = Pallas base field):
  - NIFS commitment fold verification (Pallas EC ops, native in F_p)

Zero non-native field operations in either circuit.
```

### Curve Cycle

```
Pallas:  base field F_p,  scalar field F_q
Vesta:   base field F_q,  scalar field F_p

Pallas commitments (coords in F_p) → EC arithmetic native in F_p circuits
Vesta commitments (coords in F_q)  → EC arithmetic native in F_q circuits
```

### What Is Deferred

```
1. q = eval_A(r_x, r_y) + ρ·eval_B(r_x, r_y) + ρ²·eval_C(r_x, r_y)
   (matrix evaluation — native verifier checks; will be moved in-circuit later)

2. PCS: w =? W(s₂,...,sₙ) under comm_W
   (deferred to accumulation)
```

---

## Parameters (constants, fixed at setup)

```
m         = log₂(num_constraints)     number of outer sum-check rounds
n         = log₂(num_vars) + 1        number of inner sum-check rounds
ℓ                                      number of public IO values of original circuit
A_vc, B_vc, C_vc                       R1CS matrices of verifier circuit (sparse, constant, small)
N_vc      ≈ 4m + 3n + 6               number of constraints in verifier circuit
V_vc                                   number of variables in verifier circuit
```

---

## Primary Circuit C₁ (over F_q)

### Public Inputs

```
x₁, ..., xₗ   ∈ F_q      original circuit public IO
comm_W         ∈ F_p²      witness commitment (Pallas point, for deferred PCS claim)
q              ∈ F_q       quotient = eval_A + ρ·eval_B + ρ²·eval_C
                            (native verifier recomputes and checks this)
```

### Witness

```
Spartan sum-check trace:
  a₁,...,aₘ      each aᵢ = (aᵢ₀, aᵢ₁, aᵢ₂, aᵢ₃) ∈ F_q⁴     outer round polys (degree 3)
  eA, eB, eC     ∈ F_q                                          claims (claim_Az, claim_Bz, claim_Cz)
  b₁,...,bₙ      each bⱼ = (bⱼ₀, bⱼ₁, bⱼ₂) ∈ F_q³            inner round polys (degree 2)
  w               ∈ F_q                                          eval_W

NIFS data:
  random_U       = (comm_W₁, comm_E₁, u₁, X₁)                  random relaxed instance
  comm_T          ∈ F_p²                                         cross-term commitment

Folded verifier-circuit witness:
  W_f            ∈ F_q^{V_vc}                                    folded witness
  E_f            ∈ F_q^{N_vc}                                    folded error vector
```

### Section 1: Fiat-Shamir Transcript (Poseidon over F_q)

Derive all challenges in-circuit. Every random value used by the Spartan verifier
is reproduced here.

```
T ← Poseidon.init("SpartanZkSNARK")
T.absorb(vk_digest)                          vk_digest is a circuit constant
T.absorb(x₁,...,xₗ)
T.absorb(comm_W)

τ₁,...,τₘ ← T.squeeze(m)                     tau challenges

For each outer round i = 1,...,m:
  T.absorb(aᵢ₀, aᵢ₁, aᵢ₂, aᵢ₃)
  rᵢ ← T.squeeze()                           outer challenge

T.absorb(eA, eB, eC)
ρ ← T.squeeze()                              combining challenge

For each inner round j = 1,...,n:
  T.absorb(bⱼ₀, bⱼ₁, bⱼ₂)
  sⱼ ← T.squeeze()                           inner challenge

T.absorb(w)                                   eval_W
T.absorb(random_U)
T.absorb(comm_T)
r_fold ← T.squeeze()                         NIFS fold challenge
```

**Cost:** ~250 constraints per Poseidon call × (m + n + ~6 calls) ≈ **250(m + n + 6)**

### Section 2: Outer Sum-Check

The outer sum-check proves:
  Σ_{x ∈ {0,1}^m} eq(τ, x) · [Az(x) · Bz(x) − Cz(x)] = 0

```
c₁ = 0

For i = 1,...,m:

  (2.1ᵢ)  2·aᵢ₀ + aᵢ₁ + aᵢ₂ + aᵢ₃ = cᵢ                         [1 linear]

  Horner evaluation: cᵢ₊₁ = aᵢ(rᵢ)
  (2.2ᵢ)  aᵢ₃ · rᵢ       = hᵢ₁ − aᵢ₂                            [1 mul]
  (2.3ᵢ)  hᵢ₁ · rᵢ       = hᵢ₂ − aᵢ₁                            [1 mul]
  (2.4ᵢ)  hᵢ₂ · rᵢ       = cᵢ₊₁ − aᵢ₀                           [1 mul]
```

**Cost:** m linear + 3m mul = **4m constraints**

### Section 3: Tau Evaluation

Compute τ̂ = eq(τ, r_x) = Π_{k=1}^{m} fₖ

```
For k = 1,...,m:
  (3.1ₖ)  τₖ · rₖ = pₖ                                            [1 mul]
  fₖ = 2·pₖ − τₖ − rₖ + 1                                         [linear, free]

F₁ = f₁
For k = 2,...,m:
  (3.2ₖ)  Fₖ₋₁ · fₖ = Fₖ                                         [1 mul]

τ̂ = Fₘ
```

**Cost:** m + (m − 1) = **2m − 1 mul constraints**

### Section 4: Outer Final Check

```
(4.1)  eA · eB = P                                                 [1 mul]
(4.2)  τ̂ · (P − eC) = cₘ₊₁                                       [1 mul]
```

Together these enforce: **cₘ₊₁ = τ̂ · (eA · eB − eC)**

**Cost:** **2 mul constraints**

### Section 5: Inner Sum-Check

The inner sum-check proves:
  Σ_{y ∈ {0,1}^n} [A(r_x,y) + ρ·B(r_x,y) + ρ²·C(r_x,y)] · z(y) = eA + ρ·eB + ρ²·eC

```
Compute initial claim:
(5.0a)  ρ · ρ = ρ²                                                 [1 mul]
(5.0b)  ρ · eB = t₁                                                [1 mul]
(5.0c)  ρ² · eC = d₁ − eA − t₁                                    [1 mul]

This defines d₁ = eA + ρ · eB + ρ² · eC.

For j = 1,...,n:

  (5.1ⱼ)  2·bⱼ₀ + bⱼ₁ + bⱼ₂ = dⱼ                                [1 linear]

  Horner evaluation: dⱼ₊₁ = bⱼ(sⱼ)
  (5.2ⱼ)  bⱼ₂ · sⱼ = gⱼ − bⱼ₁                                    [1 mul]
  (5.3ⱼ)  gⱼ · sⱼ  = dⱼ₊₁ − bⱼ₀                                  [1 mul]
```

**Cost:** 3 + n linear + 2n mul = **3n + 3 constraints**

### Section 6: eval_X Computation

Compute ê_X = MLE(1 ∥ x₁ ∥ ... ∥ xₗ) evaluated at (s₂,...,sₙ).

Since X is sparse (ℓ+1 non-zero entries at positions 0, 1, ..., ℓ), and the bit
patterns of these indices are constants known at compile time:

```
For each k = 0,...,ℓ:

  eq_k = Π_{j=1}^{n-1} σₖⱼ

  where σₖⱼ = sⱼ₊₁   if bit_j(k) = 1
              1−sⱼ₊₁  if bit_j(k) = 0    (known at compile time, no constraint)

  (6.1ₖ)  Build product eq_k via chain of (n−2) multiplications:
           eq_k⁽¹⁾ = σₖ₁
           eq_k⁽ᵗ⁾ = eq_k⁽ᵗ⁻¹⁾ · σₖₜ     for t = 2,...,n−1       [1 mul each]

(6.2ₖ)  xₖ · eq_k = tₖ                     for k = 1,...,ℓ        [1 mul each]
ê_X = eq_0 + t₁ + ... + tₗ                                         [linear, free]
```

**Cost:** ℓ·(n − 2) + ℓ = **ℓ·(n − 1) mul constraints**

### Section 7: Inner Final Check

```
(7.1)  w · (1 − s₁) = tmp_w                                        [1 mul]
(7.2)  ê_X · s₁ = tmp_x                                            [1 mul]
eval_Z = tmp_w + tmp_x                                              [linear]

(7.3)  q · eval_Z = dₙ₊₁                  q is a public input      [1 mul]
```

This enforces: dₙ₊₁ = q · [(1 − s₁) · w + s₁ · ê_X]

where s₁ = r_y[0] selects between the W part and the (1∥X) part of z = (W ∥ 1 ∥ X).

**Cost:** **3 mul constraints**

### Section 8: NIFS Scalar Fold

Verify the scalar parts of the NIFS fold. Given:
- random_U = (..., u₁, X₁)    — random relaxed instance
- U_verifier = (..., X₂)      — verifier circuit instance (u₂ = 1)
- r_fold                       — derived in Section 1
- claimed folded scalars: u_f, X_f

```
(8.1)  u_f = u₁ + r_fold                                           [linear, 0 mul]

For each k = 0,...,|X|−1:
  (8.2ₖ)  r_fold · X₂[k] = δₖ                                     [1 mul]
  X_f[k] = X₁[k] + δₖ                                              [linear]
```

where |X| = m + n + 4 (number of public IO of verifier circuit).

**Cost:** **m + n + 4 mul constraints**

### Section 9: Relaxed R1CS Satisfiability (Folded Verifier Circuit)

Check: A_vc · z_f ∘ B_vc · z_f = u_f · (C_vc · z_f) + E_f

where z_f = (W_f ∥ u_f ∥ X_f), and A_vc, B_vc, C_vc are **constants**
(the verifier circuit's R1CS matrices, known at setup time).

```
For each constraint i = 0,...,N_vc − 1:

  Lᵢ = Σⱼ A_vc[i,j] · z_f[j]            [linear, free — A_vc entries are constants]
  Rᵢ = Σⱼ B_vc[i,j] · z_f[j]            [linear, free]
  Oᵢ = Σⱼ C_vc[i,j] · z_f[j]            [linear, free]

  (9.1ᵢ)  Lᵢ · Rᵢ = Pᵢ                                           [1 mul]
  (9.2ᵢ)  u_f · Oᵢ = Qᵢ                                           [1 mul]
  (9.3ᵢ)  Pᵢ − Qᵢ = E_f[i]                                        [linear, free]
```

**Cost:** **2 · N_vc** mul constraints, where N_vc ≈ 4m + 3n + 6 ≈ 150

---

## Total Constraint Count for C₁

| Section | Description | Mul constraints |
|---------|-------------|-----------------|
| 1 | Poseidon transcript | 250(m + n + 6) |
| 2 | Outer sum-check | 3m |
| 3 | τ̂ computation | 2m − 1 |
| 4 | Outer final | 2 |
| 5 | Inner sum-check | 2n + 3 |
| 6 | eval_X | ℓ(n − 1) |
| 7 | Inner final | 3 |
| 8 | NIFS scalar fold | m + n + 4 |
| 9 | Relaxed R1CS sat | 2(4m + 3n + 6) |
| **Total** | | **250(m+n+6) + 14m + 9n + 23 + ℓ(n−1)** |

### Concrete Example (m = 20, n = 21, ℓ = 10)

```
Poseidon:        250 × 47     =  11,750
Outer SC:        3 × 20       =      60
Tau:             2 × 20 − 1   =      39
Outer final:                   =       2
Inner SC:        2 × 21 + 3   =      45
eval_X:          10 × 20      =     200
Inner final:                   =       3
NIFS scalar:     20 + 21 + 4  =      45
Relaxed R1CS:    2 × 150      =     300
──────────────────────────────────────────
Total C₁:                     ≈  12,444 constraints
```

Poseidon dominates. The algebraic constraints themselves are < 700.

---

## Secondary Circuit C₂ (over F_p)

Verifies the EC (commitment) parts of the NIFS fold. All Pallas EC operations
are native in F_p.

### Public Inputs

```
h₁   ∈ F_p      hash binding to primary circuit state
```

### Witness

```
comm_W₁, comm_E₁   ∈ F_p²     random relaxed instance commitments (Pallas points)
comm_W₂             ∈ F_p²     verifier circuit witness commitment (Pallas point)
comm_T              ∈ F_p²     cross-term commitment (Pallas point)
r_fold              ∈ F_p      fold challenge (as bits for EC scalar mul)
comm_W_f, comm_E_f  ∈ F_p²     claimed folded commitments (Pallas points)
```

### Constraints

```
D1. Derive fold challenge:
    r_fold = Poseidon_p(random_U, U_verifier, comm_T)              [~250]

D2. Verify commitment fold (Pallas EC, native in F_p):
    comm_W_f =? comm_W₁ + r_fold · comm_W₂
      scalar mul: ~254 double-and-add                               [~254]
      point add:                                                    [~6]

D3. comm_E_f =? comm_E₁ + r_fold · comm_T
      scalar mul:                                                   [~254]
      point add:                                                    [~6]
```

### Total C₂ ≈ 770 constraints

---

## Linking C₁ and C₂

Both circuits hash the same logical inputs (random_U, U_verifier, comm_T) to derive
r_fold, but in their respective native fields. The challenge is represented as bits
so both circuits agree on the same value.

C₁ confirms the scalar parts (u_f, X_f) are correct.
C₂ confirms the EC parts (comm_W_f, comm_E_f) are correct.
Together they prove the entire NIFS fold is valid.

---

## Future Work

- **Matrix evaluation in-circuit (Section 7):** Computing q = eval_A + ρ·eval_B + ρ²·eval_C
  in-circuit requires building eq tables of size 2^m + 2^n and a sparse dot product
  of size O(nnz). Cost: 2^m + 2^n + ~4·nnz mul constraints. For large circuits this
  dominates; will be developed separately.

- **PCS accumulation:** Accumulate deferred PCS claims (comm_W, (s₂,...,sₙ), w) across
  multiple proofs. Single PCS check at the end.

- **Poseidon gadget:** In-circuit Poseidon hash over F_q and F_p. ~250 R1CS constraints
  per absorption/squeeze. Must match the native transcript protocol.
