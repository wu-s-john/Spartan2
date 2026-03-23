# Program: Spartan ZK Prove < 100ms

## Goal

Get **Spartan ZK total prove time under 100ms** while maintaining 128-bit cryptographic security.

Baseline: 238ms. Target: <100ms. That's a 2.4× speedup.

## Metric

The single number to optimize is `TOTAL PROVE (prep+prove)` from:
```bash
cargo run --release --example rs_shuffle_bp_full -- --zk 2>&1
```

Run 3 times, take the **median**.

## Correctness Gate

Every experiment must pass before it can be kept:
1. `cargo test --release` — zero failures
2. Benchmark prints `Proof verified successfully!`
3. Verify time not orders of magnitude worse than baseline

## Security Gate

Every kept experiment must be reviewed by the **security verifier** (see `security_verifier.md`). The security verifier is a separate AI agent that reads the diff and checks for cryptographic soundness. It has veto power — if it rejects, the experiment is discarded regardless of speed.

### How to Run the Security Verifier

Use OpenAI Codex CLI in non-interactive mode at `xhigh` effort. Pipe the diff and the security verifier prompt together:

```bash
# 1. Capture the diff (from last known-good commit)
DIFF=$(git diff HEAD~1)

# 2. Build the review prompt
REVIEW_PROMPT=$(cat <<'PROMPT'
$(cat autoresearch/security_verifier.md)

---

Review this diff for cryptographic security:

Description: "DESCRIPTION_OF_CHANGE"

<diff>
DIFF_PLACEHOLDER
</diff>
PROMPT
)

# 3. Substitute the actual diff into the prompt and run Codex
echo "${REVIEW_PROMPT//DIFF_PLACEHOLDER/$DIFF}" | codex exec \
  --model o4-mini \
  --sandbox read-only \
  -
```

**Interpreting the result:**
- Codex will output a `VERDICT: PASS` or `VERDICT: FAIL` with a checklist.
- If `VERDICT: FAIL` → `git reset HEAD~1` and discard the experiment.
- If `VERDICT: PASS` → record as "keep" in results.tsv and advance.

**Notes:**
- `--sandbox read-only` ensures the reviewer cannot modify any files.
- The `-` at the end tells Codex to read the prompt from stdin.
- If Codex is unavailable, you may use any capable LLM (Claude, GPT-4, etc.) with the same prompt from `security_verifier.md`.

## Files You CAN Modify

| File | Role |
|------|------|
| `src/provider/msm.rs` | Multi-scalar multiplication engine |
| `src/provider/pcs/hyrax_pc.rs` | Hyrax polynomial commitment scheme |
| `src/provider/pcs/ipa.rs` | Inner product argument |
| `src/sumcheck.rs` | Sumcheck prover (outer + inner) |
| `src/spartan_zk.rs` | ZK prover orchestration |
| `src/r1cs/sparse.rs` | Sparse matrix-vector multiply |
| `src/small_field/delayed_reduction.rs` | Wide-limb field accumulation |
| `src/small_field/*.rs` | Field arithmetic kernels |
| `src/provider/pasta.rs` | Pasta curve operations |

You may create new files in `src/` if needed (e.g., SIMD kernels, new data structures).

## Files You MUST NOT Modify

- `examples/rs_shuffle_bp_full.rs` — benchmark harness
- `src/rs_shuffle_bp/` — shuffle circuit definition
- `src/timing/` — instrumentation
- `Cargo.toml` — no new dependencies (feature flags OK)
- `autoresearch/` — these files
- Anything outside this repo (especially `../legit-poker`)

## Where to Look for Speedups

Ordered by expected impact:

1. **MSM (commit_rest = 143ms, 60% of prove)**
   252 independent MSMs of 1024 Pallas points. Current: serial Pippenger per row, rayon across rows, XYZZ bucket coordinates.
   Ideas: window size tuning, precomputation tables, batch MSM, endomorphisms, memory layout, bucket accumulation.

2. **Hyrax commitment width**
   Currently width=1024. This is a tunable parameter, not a protocol constant. Fewer rows × larger MSMs, or more rows × smaller MSMs — profile both.

3. **Sumcheck (outer 20ms + inner 18ms)**
   Already uses delayed modular reduction. Ideas: SIMD, parallelism granularity, fused rounds.

4. **NIFS (16ms)**
   ZK blinding overhead. Can rerandomization be cheaper?

5. **PCS prove (14ms)**
   IPA + one MSM. Small but still counts.

6. **Phase pipelining**
   Can any phases overlap? e.g., start mat_vec while commit finishes.

## Security Constraints — NON-NEGOTIABLE

- 128 bits of security throughout
- Scalar field elements are ~256 bits — do NOT truncate
- Do NOT reduce curve point size, group order, or security parameters
- Do NOT skip or weaken zero-knowledge blinding (NIFS)
- Do NOT remove or simplify grand product checks
- Do NOT change the curve (Pallas/Vesta) or field
- Algorithm changes, data structure changes, parallelism, memory layout, arithmetic optimizations are all fair game

## Experiment Protocol

```
LOOP FOREVER:
  1. Read git log and results.tsv for context
  2. Form a hypothesis
  3. Implement the change (allowed files only)
  4. git commit -m "descriptive message"
  5. cargo test --release
     - Fail → try to fix, or git reset and skip
  6. Benchmark 3 times, take median prove time
  7. If improved AND tests pass:
     a. Run security verifier on the diff (see security_verifier.md)
     b. If security passes → KEEP (record in results.tsv, advance branch)
     c. If security fails → DISCARD (git reset)
  8. If not improved → git reset, try something else
```

## results.tsv Format

```
commit	prove_ms	commit_rest_ms	outer_sc_ms	inner_sc_ms	pcs_ms	nifs_ms	verify_ms	status	description
```

Tab-separated. Do NOT commit this file.

## NEVER STOP

The human might be asleep. You are autonomous. If stuck:
- Re-read the hot-path source files
- Combine previous near-misses
- Try radical algorithmic changes
- Add timing spans to find hidden costs
- Try the opposite of what you've been doing
