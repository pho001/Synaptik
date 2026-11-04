# Task 0001C: Explicit [Native Parameter Copy](../../../../glossary.md#native-parameter-copy--trainingparameterstorage) Before Training

> **Editorial reading context — not a status update.** The application explicitly calls
> the storage-copy utility before opening a session. It copies exact represented values into
> the application's shared native Arena and changes each existing Tensor's storage association,
> not its identity or the Parameter wrapper. Old expressions therefore still identify the
> same parameters. The application keeps that Arena alive through session close; the helper
> neither returns a new owner nor performs a device transfer.
> Today's [explicit native-copy contract](../../../../reference/training/parameters.md#explicit-native-parameter-copy)
> explains allocation/rollback and its later generic parameter-source signature; [session admission](../../../../reference/training/sessions.md#session-construction-and-input-mapping)
> explains why open itself does not silently perform this copy.

## Status

Complete

## Change class

Class C — introduces a public pre-session storage transition, caller-owned native [resource lifetime](../../../../glossary.md#ownership-borrowing-and-lease), and a [Model](../../../../glossary.md#term-model)/[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/Training workflow contract.

## Goal

Let callers explicitly copy initialized [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [module](../../../../glossary.md#term-module) [parameters](../../../../glossary.md#term-parameter) into caller-owned shareable native [host](../../../../glossary.md#term-host) memory before opening a [`TrainingSession`](../../../../glossary.md#training-session--trainingsession), without changing Parameter/[Tensor](../../../../glossary.md#tensor) [identity](../../../../glossary.md#term-identity-meanings) or silently copying inside `TrainingSession.open`.

## [Scope](../../../../glossary.md#term-scope)

- Add `io.github.pho001.synaptik.training.TrainingParameterStorage` with one public static `copyToSharedNative(Module, Arena)` [operation](../../../../glossary.md#operation). It returns no [owner](../../../../glossary.md#term-owner); the supplied [Arena](../../../../glossary.md#term-arena) remains caller-owned and must outlive any session using the copied [parameters](../../../../glossary.md#term-parameter).
- Copy all currently discovered parameters, whether their existing storage is [heap](../../../../glossary.md#term-heap) or native, into fresh exact-capacity aligned native segments in recursive path order. Preserve wrapper and [Tensor](../../../../glossary.md#tensor) [identity](../../../../glossary.md#term-identity-meanings), [descriptor](../../../../glossary.md#term-descriptor), [graph](../../../../glossary.md#graph) [provenance](../../../../glossary.md#provenance), represented raw bytes, and source storage [ownership](../../../../glossary.md#term-ownership-meanings). Do not replace any association until every parameter and requested destination has passed [validation](../../../../glossary.md#term-validation) and copy.
- Reject empty/incompletely initialized [module](../../../../glossary.md#term-module) discovery, duplicate Tensor identities, overlapping source parameter regions, missing/dead/inaccessible source storage, invalid or unresolved/strided/[view](../../../../glossary.md#term-view-meanings) geometry, unsupported type, non-[finite](../../../../glossary.md#term-finite) initial values, unsnapshotable logical or [optimizer](../../../../glossary.md#term-optimizer)-width byte counts, and a dead or non-shareable Arena before mutation. Source storage may be read-only because it is only read; destinations are writable.
- Under the required exclusive caller use, install all prepared associations in path order; an unexpected ordinary install failure restores earlier exact associations with retained failure/suppression. Allocation already consumed from the caller Arena is not individually reclaimed on failure.
- Keep `TrainingSession.open` unchanged: it still rejects heap-backed parameters and [captures](../../../../glossary.md#term-capture-meanings) fixed storage after the explicit copy. Never [run](../../../../glossary.md#term-run) the copy while a session is open or while another caller uses the Module/storage.
- Add a real [`Linear`](../../../../glossary.md#term-linear-meanings) initialized-on-heap → copy → same-Tensor graph → `TrainingSession.open`/[CPU](../../../../glossary.md#term-cpu) run integration, focused raw-bit and failure-before-rebinding tests, public [Javadoc](../../../../glossary.md#term-javadoc), user/[API](../../../../glossary.md#term-application-programming-interface-api) examples, architecture explanation, and [ADR](../../../../glossary.md#term-architecture-decision-record-adr).

## Non-goals

- Implicit conversion in session open, replacement of [Parameter](../../../../glossary.md#term-parameter) or [Tensor](../../../../glossary.md#tensor) objects, allocating/closing the caller's [Arena](../../../../glossary.md#term-arena), [backend](../../../../glossary.md#backend) [residency](../../../../glossary.md#residency) or transfers, [optimizer](../../../../glossary.md#term-optimizer) memory reduction, accepting [views](../../../../glossary.md#term-view-meanings)/[strides](../../../../glossary.md#term-stride), changing [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) initializer defaults, or adding a new [Engine](../../../../glossary.md#term-engine) execution path.

## Contracts

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md#module-ownership-routing) and [dependency rules](../../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span> — Training owns the explicit orchestration step and keeps existing edges.
- [Foundational modules — `modules/model`](../../../../architecture/contracts/foundational-modules.md#modulesmodel) — [Tensor](../../../../glossary.md#tensor) [identity](../../../../glossary.md#term-identity-meanings)/[provenance](../../../../glossary.md#provenance) are [immutable](../../../../glossary.md#term-immutable); [borrowed](../../../../glossary.md#ownership-borrowing-and-lease) [host](../../../../glossary.md#term-host)-storage association may change.
- [Extensions/training — `extensions/nn`](../../../../architecture/contracts/extensions-training.md#extensionsnn), [`extensions/training`](../../../../architecture/contracts/extensions-training.md#extensionstraining), and [optimizer/training lifecycle](../../../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span> — [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) owns [Module](../../../../glossary.md#term-module)/[Parameter](../../../../glossary.md#term-parameter), Training owns [backend](../../../../glossary.md#backend)-neutral preparation and [optimizer](../../../../glossary.md#term-optimizer) orchestration, session storage stays fixed after open.

The existing contracts allow caller-coordinated [Model](../../../../glossary.md#term-model) host-storage rebinding but do not yet name this public pre-session convenience. Update the root routing summary and the one owning scoped contract explicitly before implementing the helper; preserve the current `TrainingSession.open` domain. If an actual conflict appears, stop and report it before code changes.

## Dependencies and integration

- Depends on: Training 0001A and 0001B (Complete and pushed at `7ea2bd3b`); [Model](../../../../glossary.md#term-model) [borrowed](../../../../glossary.md#ownership-borrowing-and-lease) [host](../../../../glossary.md#term-host)-storage and [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) recursive [parameter](../../../../glossary.md#term-parameter) APIs (current).
- Conflicts with: any concurrent [Training session](../../../../glossary.md#training-session--trainingsession)/storage [API](../../../../glossary.md#term-application-programming-interface-api) or architecture-contract change.
- Parallel group: None.
- Common base revision: `7ea2bd3b` on `main`.
- Integration order: [architecture contract](../../../../glossary.md#architecture-contract)/[ADR](../../../../glossary.md#term-architecture-decision-record-adr), implementation/tests, explanatory docs and independent review.
- Integration [validation](../../../../glossary.md#term-validation): affected-[module](../../../../glossary.md#term-module) tests/Javadocs, focused Training [CPU](../../../../glossary.md#term-cpu) integration, architecture tests, then one full `./gradlew test javadoc` because the authoritative module boundary changes.
- Shared-document integration [owner](../../../../glossary.md#term-owner): main coordination context.

The prerequisites are complete, the worktree is clean at the pushed common base, this is the sole authorized Training write frontier, and no conflicting writer is active. The user approved the explicit caller-owned native-copy direction on 2026-10-11.

## Files and symbols

- `extensions/training/src/main/java/io/github/pho001/synaptik/training/TrainingParameterStorage.java` — new public [operation](../../../../glossary.md#operation) and private [validation](../../../../glossary.md#term-validation)/staging.
- `TrainingParameter.java` — reuse package-private width/[finite](../../../../glossary.md#term-finite)/storage checks only if semantics remain clear; do not broaden `TrainingSession.open`.
- `extensions/training/src/test` and `testing/integration-tests/src/test` — focused utility tests and real initialized [`Linear`](../../../../glossary.md#term-linear-meanings)/[CPU](../../../../glossary.md#term-cpu) session workflow.
- `ARCHITECTURE.md`, `docs/architecture/contracts/extensions-training.md`, `docs/architecture/training-graph.md`, new [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0028 — authoritative and explanatory [ownership](../../../../glossary.md#term-ownership-meanings) decision.
- `docs/api/training-api.md`, `docs/user-guide/training.md`, affected Javadocs/glossary — runnable caller-owned [Arena](../../../../glossary.md#term-arena) workflow and failure/[lifetime](../../../../glossary.md#term-lifetime) guidance.

## Acceptance criteria

- Initialized [heap](../../../../glossary.md#term-heap)-backed [`Linear`](../../../../glossary.md#term-linear-meanings) [weights](../../../../glossary.md#term-weight)/[bias](../../../../glossary.md#term-bias) remain the exact same [Tensor](../../../../glossary.md#tensor) and [Parameter](../../../../glossary.md#term-parameter) objects after copy, retain exact represented bytes, become writable native shared exact-capacity storage, and can be trained through the existing [Engine](../../../../glossary.md#term-engine)-backed session without a second [compile](../../../../glossary.md#compile)/[prepare](../../../../glossary.md#prepare) path.
- Two-parameter [validation](../../../../glossary.md#term-validation) failure leaves every original association and parameter byte unchanged; no Engine/session work occurs. Copying never closes the source or [target](../../../../glossary.md#term-target) [Arena](../../../../glossary.md#term-arena).
- Duplicate or overlapping source parameter storage is not silently de-aliased. A confined/dead Arena is rejected before replacement; closing the supplied Arena after session close is the caller's responsibility.
- Copy is explicitly invoked; `TrainingSession.open` still rejects ordinary heap-backed parameters if the caller omits it.
- Public docs show construction/initialization, explicit copy, [objective](../../../../glossary.md#term-objective)/session, [lifetime](../../../../glossary.md#term-lifetime) order, and the no-copy-open boundary. Contract, [ADR](../../../../glossary.md#term-architecture-decision-record-adr), Javadocs, docs, glossary, and targeted tests agree.

## [Validation](../../../../glossary.md#term-validation)

- Worker: targeted new tests, `./gradlew :extensions:training:test :extensions:training:javadoc`, focused `:testing:integration-tests:test --tests '*TrainingSessionCpuIntegrationTest'` (or a new focused class), and `git diff --check`.
- Integration [owner](../../../../glossary.md#term-owner): targeted `:testing:architecture-tests:test` and one `./gradlew test javadoc` after implementation/review changes; record counts/skips and platform-dependent [Metal](../../../../glossary.md#term-metal) limitations.

## Documentation and review impact

Mandatory independent Class C code and documentation review. Use the architecture, decision-record, [API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc), and user-guide profiles; check relevant glossary terms. The contract update must not imply [heap](../../../../glossary.md#term-heap) becomes directly accepted by `TrainingSession.open` or transfer storage [ownership](../../../../glossary.md#term-ownership-meanings) to Training.

## Result

Implemented `TrainingParameterStorage.copyToSharedNative(Module, Arena)` with complete recursive
[parameter](../../../../glossary.md#term-parameter) [validation](../../../../glossary.md#term-validation), exact raw-byte staging into caller-owned shared native storage, and
[identity](../../../../glossary.md#term-identity-meanings)-preserving association replacement. `TrainingSession.open` remains native-only and does
not copy implicitly. The architecture root and Training contract, [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0028, [API](../../../../glossary.md#term-application-programming-interface-api)/user guides,
glossary, and Javadocs now describe the explicit transition and [Arena](../../../../glossary.md#term-arena) [lifetime](../../../../glossary.md#term-lifetime).

Focused validation passed: 11 new storage tests, the Training [module](../../../../glossary.md#term-module) suite and [Javadoc](../../../../glossary.md#term-javadoc), and the
real initialized-[`Linear`](../../../../glossary.md#term-linear-meanings) [CPU](../../../../glossary.md#term-cpu) integration. Independent Class C review found two P2 documentation/
coverage gaps (foreign-thread confined Arena exception and distinct partially overlapping source
slices); both were corrected and the reviewer returned APPROVE with no residual findings. The
integration-[owner](../../../../glossary.md#term-owner) `./gradlew test javadoc` [run](../../../../glossary.md#term-run) passed with the verified packaged local [Metal](../../../../glossary.md#term-metal)
library configured; architecture tests were included. Without that environment setting, the
first full attempt reported Metal [EXP](../../../../glossary.md#term-expx)/[SIGMOID](../../../../glossary.md#term-sigmoid) failures requiring
`SYNAPTIK_METAL_TEST_LIBRARY` and was cancelled after diagnosis. `git diff --check` passed.

No [optimizer](../../../../glossary.md#term-optimizer)-state memory reduction or direct native [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) initializer was added.
