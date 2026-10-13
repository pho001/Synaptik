# Model 0033 — Decide stable softmax overflow semantics

## Status

Draft

## Goal and class

Class C numerical-contract decision, not an authorized implementation task. Resolve review G1:
CPU's admitted stable softmax avoids intermediate EXP overflow, while Model's literal named-site
wording can require a different output class. Do not weaken either side silently.

## Concrete conflict

For contiguous FLOAT32 `[1000,1000]`, axis 0, the current CPU provider/finite-shift validator
admit execution. `CpuSoftmaxEmitter` subtracts the maximum, so EXP sees `[0,0]` and softmax
returns `[0.5,0.5]`. `SoftmaxKind` and the foundational composite row instead describe EXP
of each unshifted value, a rounded sum and division: EXP overflows to infinity and infinity
divided by infinity is NaN. The alternative-algorithm clause requires special-value preservation;
finite-input qualification alone does not authorize changing that result class.

This was source-confirmed by an independent read-only review. Existing CPU tests use a stable
oracle, not a demonstration that the literal contract authorizes the result. No new numerical
run was performed in that cross-check. CPU 0021's storage/validation fixes and CPU 0022's
activation-site repairs are separate and do not settle G1.

## Decision to request

Recommended direction: explicitly permit/define stable finite softmax and log-softmax without
incidental unshifted-EXP overflow, retaining precise input NaN/infinity, empty, dtype, accumulator
and publication rules. Preserve the current stable CPU algorithm where conforming; do not
replace useful finite probabilities with NaNs solely to reproduce accidental intermediate overflow.

This is a proposal, not a change to authority. The owner must choose exact observable semantics
and whether softmax/log-softmax share the same stable construction. Do not generalize permission
to every composite, attention, log-sum-exp, normalization or fusion by analogy.

## Future implementation packet requirements

After explicit approval, read root scope/Core invariants and owning
`foundational-modules.md` Profile-free numerical semantics / Composite inheritance / Model
low-precision contract; `SoftmaxKind`, current CPU emitter/validator/reference and focused tests.
Update root and the one owning scoped contract, family Javadoc, numerical explanation and a
significant ADR in one reviewed change. Add independent contract examples and backend-conformance
regressions for finite EXP overflow/underflow, cancellation-relevant offsets, both kinds and dtypes,
plus NaN/infinity/empty behavior. Review consumers and compatibility identities explicitly.
Keep test-only tolerances separate from class/guard semantics and retain clean Java code shape.

## Dependencies and integration

- Depends on: explicit numerical-contract owner decision; Model 0032 Complete; review G1.
- Conflicts with: Model numerical semantics, softmax code/oracles and shared numerical docs.
- Parallel group: None; contract-first serial work.
- Common base revision: not selected until the audit repair group closes.
- Integration order: decision/contract → scoped consumers and conformance → independent review.
- Integration validation: to be fixed in the Ready packet; focused/module tests, Javadoc,
  relevant backend/integration checkpoint and independent Class C review.
- Shared-document integration owner: future task coordinator.

No implementation is authorized by this Draft. The project-review report must keep G1 open
until the decision and required consumer/contract validation are complete.
