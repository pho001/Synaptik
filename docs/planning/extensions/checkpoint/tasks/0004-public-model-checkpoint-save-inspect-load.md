# Task 0004: Public model checkpoint save, inspect, and strict load

## Status

Complete after [CPU 0024](../../../backends/cpu/tasks/0024-default-segment-route-heap-carrier-binding.md)
and direct restored-heap Engine acceptance. Checkpoint 0003 is Complete at `cc88cab1`. The user
selected typed public inspection on 2026-10-14, including its explicit Model dependency and
coordinated architecture update, and selected target-defined alias groups for v1 restore.
Checkpoint implementation, independent Class C review, direct regression, and combined
validation are recorded below. The v1 schema is unchanged; no competing Checkpoint writer is
active.

## Change class

Class C — public persistence API, published project dependencies, durable model-state restore,
and the Checkpoint-to-NN installation boundary.

## Goal

Give a caller with a recreated NN module a bounded, fail-closed way to save, inspect, and load
the model's complete parameter and buffer values. Reuse the version-one artifact envelope and
internal model candidate; do not serialize arbitrary model-defining Java code.

## Selected policy

- Prefer one `ModelCheckpoint` facade in `checkpoint.model`: `save(Path, Module, Engine,
  configDigest, optionalInputDigest, ArtifactLimits)`, `inspect(Path, ArtifactLimits)`, and
  `load(Path, Module, expectedConfigDigest, expectedInputDigest, ArtifactLimits)`. Null optional
  input digest means absent; presence must match exactly. Save/load require caller quiescence.
  `inspect` validates the complete file, including all payloads; a later load rereads and
  revalidates it rather than trusting a prior inspection of a mutable path.
- Expose an immutable, detached typed inspection value with defensive digest copies and
  ordered path, `StateKind`, `DataType`, `Shape`, and byte-length descriptors. This makes the
  public NN and Engine edges Gradle `api` dependencies and adds a direct public Model `api`
  edge. This specific dependency/API choice is user-approved; coordinate the root dependency
  rule, owning extensions contract, architecture test, explanatory docs, and an ADR in the same
  implementation change before exposing it as current architecture.
- The v1 file stores one value per path, not source Tensor identity. Preserve exact
  alias groups declared by a fully bound recreated target: require byte-identical payloads for
  every target alias group, then install one reconstructed Tensor per group. The caller's
  versioned configuration fingerprint must cover intended sharing; Checkpoint cannot infer
  source alias identity from v1 bytes. Refuse an unbound/lazy target in v1 because its alias
  topology cannot be established through the complete state snapshot. Lazy restore or source
  alias verification requires a separately approved metadata/NN contract revision.
- Restore canonical big-endian payload bits into data-type-compatible JVM host carriers in
  native byte order, with no floating conversion or NaN canonicalization. `BOOL` bytes must be
  `0` or `1`. A loaded Tensor is a fresh dense-contiguous host leaf; parameter candidates have
  `requiresGrad=true`, buffers false unless one target alias group includes a parameter. Heap
  carriers retain automatic lifetime; Checkpoint owns no Arena after return.

## Scope

- Public `ModelCheckpoint` and an immutable inspection type in `checkpoint.model`; focused
  private restoration/bit-copy helpers only where they have a distinct responsibility.
- Save via `ModelArtifactCapture.capture` and `ArtifactFile.save`; inspect/load via
  `ArtifactFile.read` and `ModelArtifactCodec.decode`. Validate expected digests and all
  payload semantics before a single `Module.loadStateDictionary` installation attempt.
- Preserve the v1 envelope and metadata wire; no generic serialization, backend access,
  optimizer state, or alternate publication path.
- Update the owning contract, root dependency rule/routing where needed, architecture test,
  public API guide, Javadocs, glossary impact, and significant decision record.

## Non-goals

Training resume; tokenizer parsing; model lambda/class reconstruction; replacing Engine's
materialization path; compiler/prepare/runtime/backend access; persistence of layouts, device
buffers, or optimizer state; automatic cross-owner transaction; non-atomic file fallback; a
streaming format or a broad memory-budget redesign.

## Contracts

- [`ARCHITECTURE.md` — Authority, incorporation, and precedence](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence),
  [Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
  [Module-ownership routing](../../../../../ARCHITECTURE.md#module-ownership-routing),
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules), and
  [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements): preserve owners;
  authorize any public Model edge explicitly before implementation.
- [`extensions-training.md` — `extensions/nn`](../../../../architecture/contracts/extensions-training.md#extensionsnn)
  and [`extensions/checkpoint`](../../../../architecture/contracts/extensions-training.md#extensionscheckpoint):
  NN owns whole-tree validation/installation; Checkpoint owns model artifact semantics and uses
  only public Engine materialization.
- [`foundational-modules.md` — `modules/model`](../../../../architecture/contracts/foundational-modules.md#modulesmodel):
  use public Tensor descriptors and compatible host storage, never backend representation.
- [`runtime-prepare-engine.md` — `modules/engine`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine):
  use the public Engine facade for save-time host materialization only.

## Dependencies and integration

- Depends on: Checkpoint 0003 (Complete) for implementation; CPU 0024 for direct restored-heap
  Engine acceptance and closure, not for the Checkpoint code or v1 format. NN strict whole-tree
  load and Engine bounded compute already exist; typed inspection/API dependency and target-defined
  v1 aliases are approved, with fail-closed unbound targets.
- Conflicts with: Checkpoint 0005–0007, Training 0004 integration to the same restore contract,
  and concurrent Checkpoint/NN public API, schema, or owning-contract edits.
- Parallel group: None.
- Common base revision: N/A (serial successor from `cc88cab1`; current `main` is the work base).
- Integration order: Checkpoint implementation/review (recorded below), then CPU 0024 repair and
  independent review, then direct Checkpoint regression and 0004 closure; only then consider 0005.
- Integration validation: direct ordinary/restored heap Engine tests, affected CPU and Checkpoint
  checks, and one combined `./gradlew test javadoc` after CPU integration. Earlier successful
  Checkpoint validation remains recorded below.
- Shared-document integration owner: coordinator for the combined closure; no concurrent writer.

## Files and symbols

- `extensions/checkpoint/.../checkpoint/model/` — public facade/inspection and private
  validated restoration; existing codec/capture remain internal.
- `extensions/checkpoint/src/test/.../checkpoint/model/` — file, materialization, restore,
  alias, bit-exactness, bounds, failure, and lifecycle tests.
- `extensions/checkpoint/build.gradle.kts` and
  `testing/architecture-tests/.../CheckpointDependencyContractTest.java` — publish NN, Engine,
  and direct Model `api` dependencies and enforce the approved edge set.
- `ARCHITECTURE.md`, `docs/architecture/contracts/extensions-training.md`, focused public
  documentation and ADR — coordinated architectural/API description.

## Acceptance criteria

- Save uses one complete NN snapshot and public bounded Engine compute, then the existing
  validated atomic-publication path. Existing target remains protected by that path's stated
  failure semantics; no stronger crash-durability or concurrent-writer promise is made.
- Inspect returns only after bounded full-file, checksum, model-schema, and payload-correspondence
  validation. Its values are immutable and detached from file/Engine lifetimes.
- Load compares exact expected configuration and optional input digests, validates all entries
  and canonical `BOOL`, reconstructs seven data types and rank-zero/empty shapes with bit-exact
  raw values, and calls NN's whole-tree strict loader only after candidate construction.
- Missing/extra paths, kind/type/Shape/gradient mismatch, malformed files, bad digests,
  over-limit data, alias-value conflict, and reconstruction failure leave all target bindings
  unchanged. Successful restore preserves required target alias groups under the selected policy.
- A round trip through Engine proves raw represented bits (including signed zero, NaN payloads,
  low-precision bits, and integer patterns) and correct carrier type. No Checkpoint code reads
  backend storage, creates a second execution pipeline, or stores live Engine/NN references in
  inspection results.

## Validation

Worker: focused model-checkpoint tests, then
`./gradlew :extensions:checkpoint:check :extensions:checkpoint:javadoc`.
Integration owner: focused Checkpoint architecture test, independent Class C review, link/anchor
and glossary check, `git diff --check`, and one `./gradlew test javadoc` repository gate if the
public dependency contract is changed. Reuse successful worker evidence in review.

## Documentation and review impact

Public API, persistence, module-edge publication, and strict lifecycle require a clean
implementation context and an independent targeted architecture/documentation review. Update
Javadocs and the public guide with a complete save/inspect/load example and explicit bit/alias,
quiescence, failure, and durability limits. Check the targeted glossary terms.

## Result

The Checkpoint-only public save/inspect/load implementation, typed inspection,
target-defined alias validation, full BOOL inspection, raw-bit heap reconstruction, coordinated
Model `api` edge, tests, Javadocs, guide, glossary, contract, architecture test, and ADR are in
the working tree. The v1 envelope and metadata wire are unchanged. Focused Checkpoint and
architecture tests, Checkpoint Javadoc, and the repository `./gradlew test javadoc` gate passed.
The final empty-state/limit focused test addition also passed under the affected-module check.

Before CPU 0024, direct execution of a restored primitive-array heap Tensor through
`Engine.compute(tensor.contiguous())` failed with
`bufferSelections[0] is incompatible with prepared executable`. A temporary focused diagnostic
test reproduced the same failure for an ordinary `TensorFactory.fromFlatArray` `FLOAT64` heap
leaf, independent of Checkpoint load. Engine's compute flow prepares before borrowing input
storage, while CPU preparation's empty carrier pattern selects `MEMORY_SEGMENT`; at bind the
borrowed heap carrier is an array and violates the specialized pattern. The earlier raw-bit test
copied restored public host bytes into native test storage before an Engine oracle. That
proved bits and carrier reconstruction but did not meet direct restored-Tensor executability.
The diagnostic test was removed after recording the failure to keep the Checkpoint test suite
green; the focused failure output was `ModelCheckpointTest.diagnosticOrdinaryHeapLeafCompute`
under `./gradlew :extensions:checkpoint:test --tests
'*ModelCheckpointTest.diagnosticOrdinaryHeapLeafCompute'`.

The user approved separate CPU follow-up work, not expansion of this Checkpoint implementation
scope. CPU 0024 now binds the prepared segment form to the exact heap segment, with its initial
P1 mixed-carrier overlap finding fixed and clean independent re-review approved. The revised
`ModelCheckpointTest.savesInspectsAndLoadsSevenTypesWithExactBitsAndNativeCarriers` now runs
direct save→inspect→load→`Engine.compute(tensor.contiguous(), 1024)` on each restored heap
Tensor: all seven data types, rank-zero, and empty shape pass with exact bits and carrier checks.
`./gradlew :extensions:checkpoint:check` passed; `ModelCheckpointTest` reports 5 tests and zero
failures. The targeted documentation pass updated `docs/api/model-checkpoint-api.md`,
`docs/api/training-api.md`, and ADR 0034, checked links and relevant glossary terms, and reused
successful executable evidence. The prior independent Checkpoint Class C review had no
high-severity findings; the public Model edge, architecture test, Javadocs, and user guide are
complete. This planning closure made no further Checkpoint code or API documentation edits.

The final integrated command with `SYNAPTIK_METAL_TEST_LIBRARY` set to the local source-matched
dylib, `./gradlew test javadoc --max-workers=2 --console=plain`, passed (105 tasks: 1 executed,
104 up-to-date). An earlier unconfigured run failed only 15 Metal tests requiring that variable;
it did not identify a Checkpoint defect. No unresolved 0004 acceptance gate remains; future
Checkpoint 0005–0007 and Training 0004 remain Draft.

Status: Complete
