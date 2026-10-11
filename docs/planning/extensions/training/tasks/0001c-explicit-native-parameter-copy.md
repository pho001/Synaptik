# Task 0001C: Explicit Native Parameter Copy Before Training

## Status

Complete

## Change class

Class C — introduces a public pre-session storage transition, caller-owned native resource lifetime, and a Model/NN/Training workflow contract.

## Goal

Let callers explicitly copy initialized NN module parameters into caller-owned shareable native host memory before opening a `TrainingSession`, without changing Parameter/Tensor identity or silently copying inside `TrainingSession.open`.

## Scope

- Add `io.github.pho001.synaptik.training.TrainingParameterStorage` with one public static `copyToSharedNative(Module, Arena)` operation. It returns no owner; the supplied Arena remains caller-owned and must outlive any session using the copied parameters.
- Copy all currently discovered parameters, whether their existing storage is heap or native, into fresh exact-capacity aligned native segments in recursive path order. Preserve wrapper and Tensor identity, descriptor, graph provenance, represented raw bytes, and source storage ownership. Do not replace any association until every parameter and requested destination has passed validation and copy.
- Reject empty/incompletely initialized module discovery, duplicate Tensor identities, overlapping source parameter regions, missing/dead/inaccessible source storage, invalid or unresolved/strided/view geometry, unsupported type, non-finite initial values, unsnapshotable logical or optimizer-width byte counts, and a dead or non-shareable Arena before mutation. Source storage may be read-only because it is only read; destinations are writable.
- Under the required exclusive caller use, install all prepared associations in path order; an unexpected ordinary install failure restores earlier exact associations with retained failure/suppression. Allocation already consumed from the caller Arena is not individually reclaimed on failure.
- Keep `TrainingSession.open` unchanged: it still rejects heap-backed parameters and captures fixed storage after the explicit copy. Never run the copy while a session is open or while another caller uses the Module/storage.
- Add a real `Linear` initialized-on-heap → copy → same-Tensor graph → `TrainingSession.open`/CPU run integration, focused raw-bit and failure-before-rebinding tests, public Javadoc, user/API examples, architecture explanation, and ADR.

## Non-goals

- Implicit conversion in session open, replacement of Parameter or Tensor objects, allocating/closing the caller's Arena, backend residency or transfers, optimizer memory reduction, accepting views/strides, changing NN initializer defaults, or adding a new Engine execution path.

## Contracts

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md#module-ownership-routing) and [dependency rules](../../../../../ARCHITECTURE.md#dependency-rules) — Training owns the explicit orchestration step and keeps existing edges.
- [Foundational modules — `modules/model`](../../../../architecture/contracts/foundational-modules.md#modulesmodel) — Tensor identity/provenance are immutable; borrowed host-storage association may change.
- [Extensions/training — `extensions/nn`](../../../../architecture/contracts/extensions-training.md#extensionsnn), [`extensions/training`](../../../../architecture/contracts/extensions-training.md#extensionstraining), and [optimizer/training lifecycle](../../../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle) — NN owns Module/Parameter, Training owns backend-neutral preparation and optimizer orchestration, session storage stays fixed after open.

The existing contracts allow caller-coordinated Model host-storage rebinding but do not yet name this public pre-session convenience. Update the root routing summary and the one owning scoped contract explicitly before implementing the helper; preserve the current `TrainingSession.open` domain. If an actual conflict appears, stop and report it before code changes.

## Dependencies and integration

- Depends on: Training 0001A and 0001B (Complete and pushed at `7ea2bd3b`); Model borrowed host-storage and NN recursive parameter APIs (current).
- Conflicts with: any concurrent Training session/storage API or architecture-contract change.
- Parallel group: None.
- Common base revision: `7ea2bd3b` on `main`.
- Integration order: architecture contract/ADR, implementation/tests, explanatory docs and independent review.
- Integration validation: affected-module tests/Javadocs, focused Training CPU integration, architecture tests, then one full `./gradlew test javadoc` because the authoritative module boundary changes.
- Shared-document integration owner: main coordination context.

The prerequisites are complete, the worktree is clean at the pushed common base, this is the sole authorized Training write frontier, and no conflicting writer is active. The user approved the explicit caller-owned native-copy direction on 2026-10-11.

## Files and symbols

- `extensions/training/src/main/java/io/github/pho001/synaptik/training/TrainingParameterStorage.java` — new public operation and private validation/staging.
- `TrainingParameter.java` — reuse package-private width/finite/storage checks only if semantics remain clear; do not broaden `TrainingSession.open`.
- `extensions/training/src/test` and `testing/integration-tests/src/test` — focused utility tests and real initialized `Linear`/CPU session workflow.
- `ARCHITECTURE.md`, `docs/architecture/contracts/extensions-training.md`, `docs/architecture/training-graph.md`, new ADR 0028 — authoritative and explanatory ownership decision.
- `docs/api/training-api.md`, `docs/user-guide/training.md`, affected Javadocs/glossary — runnable caller-owned Arena workflow and failure/lifetime guidance.

## Acceptance criteria

- Initialized heap-backed `Linear` weights/bias remain the exact same Tensor and Parameter objects after copy, retain exact represented bytes, become writable native shared exact-capacity storage, and can be trained through the existing Engine-backed session without a second compile/prepare path.
- Two-parameter validation failure leaves every original association and parameter byte unchanged; no Engine/session work occurs. Copying never closes the source or target Arena.
- Duplicate or overlapping source parameter storage is not silently de-aliased. A confined/dead Arena is rejected before replacement; closing the supplied Arena after session close is the caller's responsibility.
- Copy is explicitly invoked; `TrainingSession.open` still rejects ordinary heap-backed parameters if the caller omits it.
- Public docs show construction/initialization, explicit copy, objective/session, lifetime order, and the no-copy-open boundary. Contract, ADR, Javadocs, docs, glossary, and targeted tests agree.

## Validation

- Worker: targeted new tests, `./gradlew :extensions:training:test :extensions:training:javadoc`, focused `:testing:integration-tests:test --tests '*TrainingSessionCpuIntegrationTest'` (or a new focused class), and `git diff --check`.
- Integration owner: targeted `:testing:architecture-tests:test` and one `./gradlew test javadoc` after implementation/review changes; record counts/skips and platform-dependent Metal limitations.

## Documentation and review impact

Mandatory independent Class C code and documentation review. Use the architecture, decision-record, API/Javadoc, and user-guide profiles; check relevant glossary terms. The contract update must not imply heap becomes directly accepted by `TrainingSession.open` or transfer storage ownership to Training.

## Result

Implemented `TrainingParameterStorage.copyToSharedNative(Module, Arena)` with complete recursive
parameter validation, exact raw-byte staging into caller-owned shared native storage, and
identity-preserving association replacement. `TrainingSession.open` remains native-only and does
not copy implicitly. The architecture root and Training contract, ADR 0028, API/user guides,
glossary, and Javadocs now describe the explicit transition and Arena lifetime.

Focused validation passed: 11 new storage tests, the Training module suite and Javadoc, and the
real initialized-`Linear` CPU integration. Independent Class C review found two P2 documentation/
coverage gaps (foreign-thread confined Arena exception and distinct partially overlapping source
slices); both were corrected and the reviewer returned APPROVE with no residual findings. The
integration-owner `./gradlew test javadoc` run passed with the verified packaged local Metal
library configured; architecture tests were included. Without that environment setting, the
first full attempt reported Metal EXP/SIGMOID failures requiring
`SYNAPTIK_METAL_TEST_LIBRARY` and was cancelled after diagnosis. `git diff --check` passed.

No optimizer-state memory reduction or direct native NN initializer was added.
