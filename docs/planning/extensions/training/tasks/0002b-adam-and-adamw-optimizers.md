# Task 0002B: Adam and AdamW optimizer state

## Status

Ready

## Change class

Class B — extends the public optimizer configurations, module-local update behavior, and exact
in-memory state/restore contract. It does not change Engine/NN ownership, dependency direction,
backend capability, or the prepared execution path. The owning scoped lifecycle contract must
describe the added algorithms and payloads in the same change.

## Goal

Support Adam and AdamW as default or per-group optimizers, including mixed SGD/Adam/AdamW groups,
without another execution pipeline, per-element dispatch, or weaker failure atomicity.

## Scope

- Add immutable public `Adam` and `AdamW` configurations, each constructed as
  `(double learningRate, double beta1, double beta2, double epsilon, double weightDecay)`.
  Require all five values finite, learning rate and epsilon positive,
  `beta1` and `beta2` in `[0, 1)`, and weight decay non-negative. Keep exact configuration
  equality, including algorithm kind, for state compatibility; extend sealed `Optimizer`.
- Retain one global optimizer-step counter. On a successful update with one-based step `t`,
  compute `m = beta1 * oldM + (1 - beta1) * g` and
  `v = beta2 * oldV + (1 - beta2) * g * g`; use bias-corrected
  `m / (1 - beta1^t)` and `v / (1 - beta2^t)`, then subtract
  `learningRate * correctedM / (sqrt(correctedV) + epsilon)`.
  Adam first adds coupled `weightDecay * masterParameter` to `g`; AdamW leaves `g` unchanged
  and applies decoupled `masterParameter * (1 - learningRate * weightDecay)` before the adaptive
  subtraction. `ACCUMULATE_AND_STEP` uses the existing pending gradient **sum**, not its mean.
- Preserve the current FLOAT64 arithmetic domain and FLOAT32 working/master domain for FLOAT32,
  BFLOAT16, and FLOAT16 parameters. Narrow configuration once per assigned dtype; reject a
  learning rate or epsilon that becomes zero, a beta that becomes one, or any coefficient that
  becomes non-finite or leaves its permitted range. Other valid narrowed coefficients retain
  the existing SGD-style narrowing policy. Compute and validate bias corrections outside the
  element loop. Check every intermediate, both moments, master candidate, and final
  low-precision narrowing for finiteness.
- Resolve one immutable algorithm/configuration per parameter at `open` from the default and
  ordered groups. All parameters still update on every successful global step. Preserve SGD
  arithmetic and existing group/path identity. Allow mixed algorithms in one session without
  backend identity checks or an algorithm/map lookup per element.
- Retain the existing first-moment/momentum arrays. Allocate an active second-moment array only
  for Adam/AdamW parameters and its staging array only before an update or restore that needs it.
  Stage all optimizer slots and parameter candidates before any parameter write; commit slots and
  counters only after every candidate and complete binding view pass validation.
- Generalize `TrainingState.optimizer()` to return the exact default `Optimizer`, including
  Adam and AdamW; do not add a second default-optimizer accessor. This intentionally changes
  the method's source and binary signature from `Sgd` to `Optimizer`; update all project
  call sites, tests, Javadocs, and examples together. Retain exact group configurations. Preserve
  `ParameterState.momentumBytes()` as the existing first-slot payload (Adam's first moment);
  add a neutral `firstMomentBytes(): ByteBuffer` alias and
  `secondMomentBytes(): Optional<ByteBuffer>` for Adam/AdamW. Returned views stay fresh and
  read-only.
  No second-moment payload exists for SGD. Snapshot remains detached, canonical, and immutable.
- `restore()` first validates default/group algorithm and configuration, ordered paths and
  schemas, counters, exact slot presence/width, finite values, and non-negative Adam second
  moments. Only then stage/install all parameter and slot bytes. Preserve old SGD snapshots and
  the existing `TrainingSession.open(...)` overloads unchanged.
- Update the owning scoped optimizer/training lifecycle contract, Training API/user guide,
  architecture explanation, affected Javadocs, and targeted glossary terms. Include one
  mixed-group example and explicit accessor compatibility notes.

## Non-goals

- AMSGrad, sparse gradients, maximize mode, per-parameter step counters, per-group gradient modes,
  learning-rate schedules, dynamic regrouping, optimizer-state persistence schema, checkpoint
  files, backend optimizer kernels, cross-backend numerical identity, or a rollback-policy switch.
- New Tensor operations, new gradient rules, expanded Metal low-precision graph capability, or
  an optimizer update inside the compiled graph.

## Contracts

- [`ARCHITECTURE.md` scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts),
  [module ownership](../../../../../ARCHITECTURE.md#module-ownership-routing), and
  [dependency rules](../../../../../ARCHITECTURE.md#dependency-rules) keep optimizer policy in
  Training and preserve the NN/Engine edges.
- [Extensions/training — `extensions/nn`](../../../../architecture/contracts/extensions-training.md#extensionsnn)
  owns deterministic recursive parameter paths, not optimizer algorithms.
- [Extensions/training — `extensions/training`](../../../../architecture/contracts/extensions-training.md#extensionstraining)
  permits Adam/AdamW and backend-neutral state; no backend bridge is introduced.
- [Extensions/training — optimizer/training lifecycle](../../../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle)
  governs one global lifecycle, finite state, exact restore, and ordinary-failure atomicity.
  Its SGD-specific text must be extended for the additional moments and algorithms without
  weakening existing SGD guarantees.

If these contracts require a changed ownership or Engine/backend boundary, stop for a separate
architecture decision rather than silently editing the contract.

## Dependencies and integration

- Depends on: Training 0002A (Complete, validated, reviewed, and pushed at `19b7785c`).
- Conflicts with: concurrent Training public optimizer API, parameter-group assignment,
  `TrainingState`/restore, optimizer-slot, or training-lifecycle work; Training 0003 until this
  state contract is integrated.
- Parallel group: None.
- Common base revision: `19b7785c` on `main`.
- Integration order: implementation and focused validation; independent public API/state and
  documentation review; then integration-owner repository checkpoint.
- Integration validation: configured source-matched Metal test library with
  `./gradlew test javadoc`, plus `git diff --check`, once after integration because Training,
  integration tests, and the owning scoped contract change.
- Shared-document integration owner: main coordination context.

Launch verification (2026-10-13): `main` and `origin/main` are at `19b7785c` with only this
planning update pending. Training 0002A has its complete result, validation, documentation, and
independent approval; 0002B is its sole next Training frontier, and no conflicting writer is
active. Implementation starts only after this plan is committed and pushed.

## Files and symbols

- `extensions/training/src/main/java/io/github/pho001/synaptik/training/Adam.java` and
  `AdamW.java` — new immutable public configurations; `Optimizer.java` sealed set.
- `ParameterGroup.java`, `TrainingSession.java`, `TrainingState.java`, and
  `TrainingParameter.java` in the same package — mixed assignment, update, slots, snapshots,
  and restore. Preserve existing `Sgd.java` semantics.
- Focused Training tests under `extensions/training/src/test/java/`; public CPU scenario in
  `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/TrainingSessionCpuIntegrationTest.java`.
- `docs/architecture/contracts/extensions-training.md`, `docs/architecture/training-graph.md`,
  `docs/api/training-api.md`, `docs/user-guide/training.md`, and targeted `docs/glossary.md`.

## Acceptance criteria

- Existing SGD-only construction, update bytes, snapshot payloads, and restore behavior remain
  compatible, except for the intentional `TrainingState.optimizer()` return-type change. Adam and
  AdamW each have an independent multi-step scalar oracle with beta bias
  correction, coupled versus decoupled decay, all three gradient modes, and `zeroGrad` behavior.
- Mixed default/group configurations (including Adam default with an SGD group) share one global
  step and one atomic commit. Group order, path order, algorithm kind, and every coefficient are
  exact restore identity; no per-element algorithm selection or transcendental calculation occurs.
- FLOAT64, FLOAT32, BFLOAT16, and FLOAT16 parameter cases cover coefficient narrowing, one final
  low-value conversion, moment widths, and non-finite/overflow rejection. SGD-only parameters
  acquire no second-moment allocation or snapshot payload.
- Detached Adam states round-trip raw represented bytes, including signed zeros where valid.
  Missing, extra, wrong-width, negative, or non-finite second moments fail before any parameter,
  slot, accumulation, or counter mutation. A later-parameter injected failure leaves all mixed
  groups unchanged.
- Public CPU integration exercises mixed algorithms and restore across a successful next step;
  no Engine, NN, backend, native ABI, or checkpoint-codec change is needed.

## Validation

Worker uses a clean isolated worktree, runs focused new/existing Training tests and the targeted
CPU integration test, then one
final affected-module/Javadoc run after executable code stabilizes:

```bash
./gradlew :extensions:training:test :extensions:training:javadoc :testing:integration-tests:test
git diff --check
```

Integration owner reuses worker evidence, obtains independent targeted Class B review, and runs
the recorded repository checkpoint once. Configure `SYNAPTIK_METAL_TEST_LIBRARY` to a
source-matched local library for native tests; do not substitute skipped device tests for a pass.

## Documentation and review impact

Independent review finalizes the public/Javadoc contract, two-algorithm mathematics and mixed
example, snapshot payload compatibility and intentional accessor signature change, glossary
terms, and scoped lifecycle wording. No ADR
or root dependency change is expected because Adam/AdamW and Training ownership are already
allowed; stop if implementation discovers an architectural conflict.

## Result

Empty until execution.
