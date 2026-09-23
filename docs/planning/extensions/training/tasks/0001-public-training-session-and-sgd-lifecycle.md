# Task 0001: Public Training Session and SGD Lifecycle

## Status

Complete

## Change class

Class C — adds a public cross-module session, an intentional Training-to-Engine dependency,
optimizer mutation and snapshot state, failure atomicity, and close/in-flight concurrency.

## Goal

Deliver one complete public CPU-capable training lifecycle that compiles and prepares a scalar
forward/backward graph once, maps deterministic NN parameter paths to Compiler gradient
occurrences, applies persistent SGD state over repeated runs, and exposes exact lifecycle and
in-memory restore behavior without another execution pipeline.

## Scope

- Public `Optimizer`, `Sgd`, `GradientMode`, `TrainingSession`, `TrainingStep`, and `TrainingState`.
- One borrowed Engine and Module, one owned reusable inference session, stable recursive parameter
  path/wrapper/Tensor/storage capture, and external input metadata/binding.
- FLOAT32/FLOAT64 dense writable shareable-native parameters, SGD momentum/dampening/coupled weight
  decay/compatible Nesterov, reset/accumulate semantics, persistent counters, immutable state
  snapshot, strict restore, non-finite/ordinary-failure atomicity, and deterministic
  close/in-flight behavior.
- Public documentation, Javadocs, architecture decision, architecture dependency enforcement,
  numerical/state/lifecycle tests, and an end-to-end convergence smoke.

## Non-goals

- Parameter groups, Adam/AdamW, schedulers, mixed precision, BFLOAT16 updates, dynamic or
  strided/view layouts, JVM-heap, read-only, thread-confined, or device-only parameter storage,
  asynchronous runs, or data loaders.
- Durable checkpoint bytes/files/codecs or atomic filesystem publication.
- Compiled/fused optimizer graph operations or dependencies on Runtime, Prepare, tuning, CPU,
  Metal, or CUDA.
- Changing Tensor autograd state or replacing the existing one-shot Engine backward convenience.

## Contracts

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md#core-lifecycle) — one Engine execution path and
  the explicit public-Engine dependency.
- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md#dependency-rules) — Training has no direct
  Runtime, Prepare, tuning, or concrete-backend edge.
- [Extensions/training — optimizer/training lifecycle](../../../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle)
  — session, optimizer, mapping, atomicity, state, and close authority.
- [Compiler/autograd — training graph model](../../../../architecture/contracts/compiler-autograd.md#training-graph-model)
  — compiler owns combined graph and ordered gradient roles.
- [Runtime/Prepare/Engine — run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)
  — the session reuses the existing prepared runner and result lifecycle.

If an applicable contract is missing or ambiguous, stop and report it.

## Dependencies and integration

- Depends on: NN stable recursive parameters; Compiler first-order publications; Engine 0017
- Conflicts with: Training public API, lifecycle contract, and shared training documentation
- Parallel group: None
- Common base revision: `2e110c7688ead480faa4215ff67f9814fa81f4ad`
- Integration order: contract/ADR, implementation/tests, documentation, independent Class C review
- Integration validation: full Gradle test and Javadoc lifecycle plus architecture/docs checks
- Shared-document integration owner: this task

The user explicitly authorized this frontier from the recorded common base. Its prerequisite NN,
Compiler, and Engine capabilities are current; no other Training write frontier is active.

## Files and symbols

- `extensions/training` — public API, optimizer/session implementation, focused tests and Javadocs.
- `extensions/nn/.../Parameter.java`, `Module.java` — stable-storage/exclusive-session wording only.
- `testing/architecture-tests` — dependency-edge enforcement.
- `testing/integration-tests` — actual public CPU convergence smoke.
- `ARCHITECTURE.md`, `docs/architecture/contracts/extensions-training.md`, ADR 0018, lifecycle and
  training-graph explanations — Class C contract and rationale.
- Training API/user guide/glossary and Training planning records — public workflow and status.

## Acceptance criteria

- Opening a session snapshots non-empty recursive parameter paths, rejects empty/non-finite
  parameter state before seed allocation or compile, compiles one scalar objective and its first
  gradients once, prepares once, and exposes only non-parameter bindable inputs.
- Repeated real CPU runs converge on a deterministic learnable problem without recompilation or
  re-preparation.
- SGD numerical updates, first/subsequent dampening, momentum state, weight decay, compatible
  Nesterov behavior, FLOAT32 narrowing, update order, execution numbers, optimizer step numbers,
  zeroing, accumulation, snapshots, and restore are exact.
- Multiple parameter paths map to exact target-index gradient occurrences; aliases are not guessed
  or deduplicated.
- Missing, duplicate, foreign, descriptor-incompatible, absent, dead, inaccessible, read-only,
  overlapping, replaced, unsupported, or non-finite parameter/input-derived optimizer state fails
  before optimizer mutation.
- A failed run, including represented overflow or post-execution/materialization/late-validation
  failure, leaves parameter bytes, optimizer state, counters, and pending gradients unchanged.
- Operation admission uses Engine-closed, session-closed, then busy precedence before argument
  validation. Close waits for an admitted operation, attempts all cleanup, and retains failure.
- Detached step results and immutable finite state snapshots remain readable after session/Engine
  close; restore rejects mismatched sessions/configurations and non-finite payloads.
- CPU and a known configured real Metal/mixed route succeed. A separately classified unsupported
  Metal-only route fails before optimizer mutation; Training contains no backend branch/import.
- Training publishes exact NN, Model, and Engine API dependencies, and a consumer declaring only
  Training can compile every leaked public signature type.
- The optimizer element loop reuses primitive staging buffers and creates no per-element object.

## Validation

Worker validation:

```bash
./gradlew :extensions:training:test :testing:architecture-tests:test :testing:integration-tests:test
./gradlew :extensions:training:javadoc :extensions:nn:javadoc
./gradlew test javadoc
```

Also run the repository documentation/architecture checks named by the root build or maintained
scripts, `git diff --check`, and one executable public CPU convergence smoke.

Integration/repository validation: required because the task changes dependencies, public API,
architecture, concurrency, and multiple modules.

## Documentation and review impact

- Update authoritative root/scoped contract, ADR, architecture explanations, API reference, user
  training guide, glossary, task/master/roadmap, package/type Javadocs, and NN cross-links.
- Mandatory independent targeted Class C review after implementation and validation; leave the
  review-ready change uncommitted.

## Result

- Added the public sealed optimizer configuration boundary, SGD with persistent momentum,
  dampening, coupled weight decay and compatible Nesterov, explicit gradient modes, detached
  step/state values, and one reusable `TrainingSession` over exact NN recursive parameter paths
  and Engine gradient publications.
- Added finite-value/overflow precommit validation, reusable primitive staging/rollback buffers,
  strict snapshot/restore, lifecycle-first single-operation admission, wait-on-close ownership,
  retained attempt-all cleanup, private seed cleanup, and the bounded shareable-native
  non-empty `FLOAT32`/`FLOAT64` update domain.
- Published exact NN, Model, and Engine Gradle API edges plus a compile-only consumer that declares
  only Training.
- Added numerical, mapping, non-finite/overflow failure atomicity, storage, accumulation, restore,
  close/busy/admitted-operation, rollback, retained-cleanup-failure, detached-lifetime,
  architecture dependency, CPU convergence, configured real Metal/mixed success, and precisely
  classified unsupported Metal-only coverage.
- The initial implementation validation passed `./gradlew test javadoc` and `git diff --check`.
  Mandatory external Class C review returned `BLOCK` with six P1 and one P2 findings. All findings
  were remediated in the uncommitted worktree. Focused training/architecture/integration/Javadoc
  validation passed; both configured real Metal cases passed with the built dylib; full
  `./gradlew test javadoc` passed with 98 actionable tasks; and `git diff --check` passed.
- Exact external review trail: initial verdict `BLOCK` with six P1 and one P2 findings; external
  Class C re-review verdict `APPROVE`, findings `0`.
