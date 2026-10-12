# Task 0002A: SGD parameter groups

## Status

Complete

## Change class

Class B — adds a public Training capability and extends its module-local optimizer-state and
restore contract. It changes no module ownership, dependency direction, or backend route.

## Goal

Let one `TrainingSession` apply different immutable SGD configurations to disjoint named
parameter paths while retaining one compiled graph, one global run/step lifecycle, and the
existing ungrouped API.

## Scope

- Add immutable public `io.github.pho001.synaptik.training.ParameterGroup` containing a nonempty
  ordered list of stable recursive parameter paths and one `Optimizer` configuration (currently
  only `Sgd` is permitted). Its ordered path list is its identity; no separate group name or index
  namespace is introduced.
- Add `TrainingSession.open(engine, module, objective, Optimizer defaultOptimizer,
  List<ParameterGroup> groups)`. Every discovered parameter belongs to at most one explicit group;
  omitted paths use `defaultOptimizer`. Empty groups list is equivalent to the existing overload.
  Groups do not freeze or exclude parameters from gradient targets.
- Snapshot the ordered group list at open. Reject a null list or entry, null/blank/duplicate
  paths, empty groups, repeated membership across groups, and paths absent from the captured
  Module before seed allocation, compile, or prepare. The list order and each group's path order
  are part of the exact session/state configuration identity, even where reordering would not
  change arithmetic.
- Resolve one `Sgd` per parameter in Module traversal order at open. Validate coefficient
  narrowing for each assigned parameter dtype, then use that indexed configuration for update,
  momentum staging, and restore validation without a per-element map lookup.
- Preserve the old `TrainingSession.open(..., Optimizer)` behavior and the public
  `TrainingState.optimizer()` result as the default `Sgd`. Add immutable
  `parameterGroups()` accessors on session and state. `state()` captures group configuration;
  `restore()` compares the exact default and ordered groups before any state installation.
- Keep one global execution counter, optimizer-step counter, accumulation count, gradient mode,
  and atomic commit across all parameters. Document the API with a two-rate example.

## Non-goals

- Adam/AdamW, mixed optimizer algorithms, per-group counters or gradient modes, parameter
  freezing, dynamic regrouping after open, persistence codec/schema, backend optimizer kernels,
  an optional rollback policy, or changes to the native parameter-storage domain.

## Contracts

- [`ARCHITECTURE.md` scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts),
  [module ownership](../../../../../ARCHITECTURE.md#module-ownership-routing), and
  [dependency rules](../../../../../ARCHITECTURE.md#dependency-rules) keep optimizer policy in
  Training and the NN/Engine edges unchanged.
- [Extensions/training — `extensions/nn`](../../../../architecture/contracts/extensions-training.md#extensionsnn)
  owns Module parameter declarations and deterministic recursive paths.
- [Extensions/training — `extensions/training`](../../../../architecture/contracts/extensions-training.md#extensionstraining)
  allows `ParameterGroup` and backend-neutral optimizer orchestration.
- [Extensions/training — optimizer/training lifecycle](../../../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle)
  requires all parameter targets, one session-owned state set, finite updates, complete precommit
  validation, ordinary-failure atomicity, and exact state/restore compatibility.

If a selected contract conflicts with the scoped design, stop and request a coordinated
architecture decision rather than changing the contract implicitly.

## Dependencies and integration

- Depends on: Training 0001E (Complete and pushed at `cbd14f3b`).
- Conflicts with: concurrent Training public optimizer API, state/restore, parameter mapping, or
  lifecycle changes; Training 0002B until this contract is integrated.
- Parallel group: None.
- Common base revision: `cbd14f3b` on `main`.
- Integration order: implementation, focused Training/integration validation, independent public
  API and state-contract review, then integration-owner repository validation.
- Integration validation: `./gradlew test javadoc` and `git diff --check` once after integration,
  because the public behavior also changes an end-to-end integration test module.
- Shared-document integration owner: main coordination context.

Training 0001E has final validation, documentation, and independent-review results. The pushed
base is clean, no conflicting writer is active, and 0002A is the sole authorized Training write
frontier. No implementation is launched by this planning update.

Launch verification (2026-10-12): `main` and `origin/main` both point to `cbd14f3b` before
publishing this brief. Training 0001E is Complete with its required validation and review;
the master plan and roadmap expose only 0002A as Ready, and 0002B remains Draft. No conflicting
Training writer is active. Implementation starts only after this planning update is published.

## Files and symbols

- `extensions/training/src/main/java/io/github/pho001/synaptik/training/ParameterGroup.java` —
  new immutable path/configuration value.
- `extensions/training/src/main/java/io/github/pho001/synaptik/training/TrainingSession.java` —
  grouped open, path resolution, run dispatch, state/restore, and public accessor.
- `extensions/training/src/main/java/io/github/pho001/synaptik/training/TrainingState.java` —
  detached ordered groups and exact restore identity, retaining ungrouped API behavior.
- `extensions/training/src/main/java/io/github/pho001/synaptik/training/TrainingParameter.java` —
  assigned-SGD update and momentum validation without changing its per-element algorithm.
- Focused tests in `extensions/training/src/test/java/`; public CPU scenario in
  `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/TrainingSessionCpuIntegrationTest.java`.
- `docs/api/training-api.md`, `docs/user-guide/training.md`, affected Javadocs, and targeted
  Training terminology in `docs/glossary.md`.

## Acceptance criteria

- Existing four-argument `open`, ungrouped state snapshots, arithmetic, and restore remain
  source-compatible and behaviorally unchanged. Empty explicit groups behave identically.
- Two disjoint groups can use distinct learning rates/momentum and produce independently expected
  parameter bytes in one atomic run. Ungrouped parameters use the default SGD.
- Invalid or overlapping path membership fails before compile/prepare or parameter mutation;
  group mutation through caller-owned collections cannot change an open session or detached state.
- State round-trips exact default/group configurations and parameter bytes. Restore rejects any
  changed group order, path order, membership, or SGD configuration before writing parameters,
  slots, accumulation, or counters.
- `RESET_AND_STEP`, `ACCUMULATE`, `ACCUMULATE_AND_STEP`, and `zeroGrad` retain global semantics.
  Per-parameter coefficient narrowing and zero-momentum slot rules hold for FLOAT64, FLOAT32,
  BFLOAT16, and FLOAT16; an injected later-parameter failure rolls back all groups.
- Group resolution occurs once at open; update loops have no per-element path/map dispatch,
  boxing, or new allocation. A public CPU integration test exercises grouped training and restore.

## Validation

Worker: focused new and existing Training tests and the targeted CPU integration test, then one
final affected-module run after executable code stabilizes:

```bash
./gradlew :extensions:training:test :extensions:training:javadoc :testing:integration-tests:test
git diff --check
```

Integration owner: reuse successful worker evidence, obtain independent targeted Class B review,
then run the recorded repository checkpoint once because both Training and integration-test
modules change.

## Documentation and review impact

Independent review finalizes public Javadocs, the API and user-guide example, and glossary
terminology. Explain that group paths come from `Module.parametersRecursively()`, that unlisted
parameters use the default SGD, and that group ordering affects restore compatibility. No
architecture or decision record is expected unless implementation uncovers a contract conflict.

## Result

- Implemented in isolated worktree commit `92494077` and integrated from planning base
  `9ba848a9`: immutable `ParameterGroup`, per-parameter SGD selection, exact ordered group
  snapshot/restore identity, preserved ungrouped API, focused tests, and public CPU integration.
- Changed six Training production files (including new `ParameterGroup`), two Training test
  files, one CPU integration test, and the Training API guide, user guide, and glossary.
- Worker validation passed with
  `SYNAPTIK_METAL_TEST_LIBRARY=<source-matched local dylib> ./gradlew
  :extensions:training:test :extensions:training:javadoc :testing:integration-tests:test`:
  57 Training tests and 8 CPU integration tests, zero failures/skips in those suites.
  `git diff --check` passed.
- Independent Class B public API/state-contract and documentation/Javadoc review returned
  `APPROVE`, zero findings. The two-rate examples and ordered restore identity were reviewed.
- Integration-owner checkpoint passed with the same Metal library and `./gradlew test javadoc`
  (`BUILD SUCCESSFUL`; 100 actionable tasks: 2 executed, 6 from cache, 92 up-to-date).
  Integrated `git diff --check` passed. No architecture or ADR update was needed.
- Limitations: only SGD groups are supported; Adam/AdamW remain for a later task. No unresolved
  issue or required follow-up within 0002A.

Status: Complete
