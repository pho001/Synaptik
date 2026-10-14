# Task 0004: Public model checkpoint save, inspect, and strict load

## Status

Draft. Checkpoint 0003 is Complete at `cc88cab1`. The user selected typed public inspection
on 2026-10-14, including its explicit Model dependency and coordinated architecture update.
The alias/lazy-target restore policy still needs a decision before this task becomes Ready.
No 0004 executor is authorized yet.

## Change class

Class C — public persistence API, published project dependencies, durable model-state restore,
and the Checkpoint-to-NN installation boundary.

## Goal

Give a caller with a recreated NN module a bounded, fail-closed way to save, inspect, and load
the model's complete parameter and buffer values. Reuse the version-one artifact envelope and
internal model candidate; do not serialize arbitrary model-defining Java code.

## Selected policy and pending decision

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
- The v1 file stores one value per path, not source Tensor identity. Prefer to preserve exact
  alias groups declared by a fully bound recreated target: require byte-identical payloads for
  every target alias group, then install one reconstructed Tensor per group. The caller's
  versioned configuration fingerprint must cover intended sharing; Checkpoint cannot infer
  source alias identity from v1 bytes. Whether unbound/lazy targets can be accepted without a
  provable alias topology remains a decision before this brief becomes Ready. The alternative
  is a new metadata version that records source alias groups before public load.
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

- Depends on: Checkpoint 0003 (Complete); NN strict whole-tree load and Engine bounded compute
  already exist. The typed inspection/API dependency choice is approved; the alias/lazy decision
  above remains before Ready.
- Conflicts with: Checkpoint 0005–0007, Training 0004 integration to the same restore contract,
  and concurrent Checkpoint/NN public API, schema, or owning-contract edits.
- Parallel group: None.
- Common base revision: N/A (serial successor from `cc88cab1`).
- Integration order: decide public contract, then implement 0004, then consider 0005.
- Integration validation: affected Checkpoint and architecture tests plus repository-wide
  `./gradlew test javadoc` once if the dependency/API boundary changes.
- Shared-document integration owner: 0004 implementation owner; no concurrent writer.

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

Empty until implementation.
