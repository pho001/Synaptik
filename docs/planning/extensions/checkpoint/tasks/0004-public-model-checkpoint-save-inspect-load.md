# Task 0004: Public [model](../../../../glossary.md#term-model) [checkpoint](../../../../glossary.md#term-checkpoint) save, inspect, and strict load

> **Editorial reading context — not a status update.** The application recreates a compatible
> NN target and supplies the expected configuration identity. Save asks Engine for values;
> inspection reports validated descriptions; load rebuilds values and asks NN to install a
> complete state dictionary. Inspection is not permission to skip a later read or validation.
> An alias here means several target paths holding the same exact Tensor, not equal numbers.
> Loaded heap values need no Checkpoint-owned Arena; source storage remains its original owner's
> responsibility. See today's [values-only core contracts](../../../../reference/checkpoint/values.md#core-contracts)
> and [ownership consequences](../../../../reference/checkpoint/values.md#inputs-outputs-and-ownership).

## Status

Complete after [CPU 0024](../../../backends/cpu/tasks/0024-default-segment-route-heap-carrier-binding.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
and direct restored-[heap](../../../../glossary.md#term-heap) [Engine](../../../../glossary.md#term-engine) acceptance. [Checkpoint](../../../../glossary.md#term-checkpoint) 0003 is Complete at `cc88cab1`. The user
selected typed public inspection on 2026-10-14, including its explicit [Model](../../../../glossary.md#term-model) [dependency](../../../../glossary.md#term-dependency) and
coordinated architecture update, and selected [target](../../../../glossary.md#term-target)-defined [alias](../../../../glossary.md#aliasing-and-arena) groups for v1 restore.
Checkpoint implementation, independent Class C review, direct [regression](../../../../glossary.md#term-regression), and combined
[validation](../../../../glossary.md#term-validation) are recorded below. The v1 [schema](../../../../glossary.md#term-schema) is unchanged; no competing Checkpoint writer is
active.

## Change class

Class C — public [persistence](../../../../glossary.md#term-persistence) [API](../../../../glossary.md#term-application-programming-interface-api), published project dependencies, durable [model](../../../../glossary.md#term-model)-state restore,
and the [Checkpoint](../../../../glossary.md#term-checkpoint)-to-[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) installation boundary.

## Goal

Give a caller with a recreated [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [module](../../../../glossary.md#term-module) a bounded, [fail-closed](../../../../glossary.md#term-fail-closed) way to save, inspect, and load
the [model](../../../../glossary.md#term-model)'s complete [parameter](../../../../glossary.md#term-parameter) and [buffer](../../../../glossary.md#term-buffer) values. Reuse the version-one [artifact envelope](../../../../glossary.md#term-artifact-envelope) and
internal model candidate; do not serialize arbitrary model-defining Java code.

## Selected policy

- Prefer one `ModelCheckpoint` [facade](../../../../glossary.md#term-facade) in `checkpoint.model`: `save(Path, Module, Engine,
  configDigest, optionalInputDigest, ArtifactLimits)`, `inspect(Path, ArtifactLimits)`, and
  `load(Path, Module, expectedConfigDigest, expectedInputDigest, ArtifactLimits)`. Null optional
  input digest means absent; presence must match exactly. Save/load require caller quiescence.<span data-glossary-navigation="true"> ([digest glossary](../../../../glossary.md#term-digest); [quiescence glossary](../../../../glossary.md#term-quiescence))</span>
  `inspect` validates the complete file, including all payloads; a later load rereads and
  revalidates it rather than trusting a prior inspection of a mutable path.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable))</span>
- Expose an immutable, detached typed inspection value with defensive digest copies and
  ordered path, <span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [detached glossary](../../../../glossary.md#detached-model-results))</span>`StateKind`, [`DataType`](../../../../glossary.md#data-type--datatype), [`Shape`](../../../../glossary.md#shape), and byte-length descriptors. This makes the
  public NN and Engine edges Gradle <span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Engine glossary](../../../../glossary.md#term-engine); [Gradle glossary](../../../../glossary.md#term-gradle))</span>`api` dependencies and adds a direct public [Model](../../../../glossary.md#term-model) `api`
  edge. This specific dependency/API choice is user-approved; coordinate the root dependency
  rule, owning extensions contract, architecture test, explanatory docs, and an ADR in the same
  implementation change before exposing it as current architecture.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [architecture test glossary](../../../../glossary.md#term-architecture-test); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- The v1 file stores one value per path, not source Tensor identity. Preserve exact
  alias groups declared by a fully bound recreated target: require byte-identical payloads for
  every target alias group, then install one reconstructed Tensor per group. The caller's
  versioned configuration fingerprint must cover intended sharing; Checkpoint cannot infer
  source alias identity from v1 bytes. Refuse an unbound/lazy target in v1 because its alias
  topology cannot be established through the complete state snapshot. Lazy restore or source
  alias verification requires a separately approved metadata/NN contract revision.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [alias glossary](../../../../glossary.md#aliasing-and-arena); [target glossary](../../../../glossary.md#term-target); [fingerprint glossary](../../../../glossary.md#term-fingerprint); [Checkpoint glossary](../../../../glossary.md#term-checkpoint); [lazy glossary](../../../../glossary.md#term-lazy-evaluation); [topology glossary](../../../../glossary.md#term-topology); [snapshot glossary](../../../../glossary.md#term-snapshot); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Restore canonical big-endian payload bits into data-type-compatible JVM host carriers in
  native byte order, with no floating conversion or NaN canonicalization. <span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [big-endian glossary](../../../../glossary.md#text-encoding-and-file-boundaries); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm); [byte order glossary](../../../../glossary.md#text-encoding-and-file-boundaries); [NaN glossary](../../../../glossary.md#term-nan); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form))</span>[`BOOL`](../../../../glossary.md#data-type--datatype) bytes must be
  `0` or `1`. A loaded [Tensor](../../../../glossary.md#tensor) is a fresh dense-[contiguous](../../../../glossary.md#term-contiguous) [host](../../../../glossary.md#term-host) [leaf](../../../../glossary.md#term-leaf); [parameter](../../../../glossary.md#term-parameter) candidates have
  `requiresGrad=true`, buffers false unless one target alias group includes a parameter. Heap
  carriers retain automatic lifetime; Checkpoint owns no Arena after return.<span data-glossary-navigation="true"> ([buffers glossary](../../../../glossary.md#term-buffer); [Heap glossary](../../../../glossary.md#term-heap); [lifetime glossary](../../../../glossary.md#term-lifetime); [Arena glossary](../../../../glossary.md#term-arena))</span>

## [Scope](../../../../glossary.md#term-scope)

- Public `ModelCheckpoint` and an [immutable](../../../../glossary.md#term-immutable) inspection type in `checkpoint.model`; focused
  private restoration/bit-copy helpers only where they have a distinct responsibility.
- Save via `ModelArtifactCapture.capture` and `ArtifactFile.save`; inspect/load via
  `ArtifactFile.read` and `ModelArtifactCodec.decode`. Validate expected digests and all
  payload semantics before a single <span data-glossary-navigation="true"> ([payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication))</span>`Module.loadStateDictionary` installation attempt.
- Preserve the v1 envelope and metadata wire; no generic serialization, backend access,
  optimizer state, or alternate publication path.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [serialization glossary](../../../../glossary.md#term-serialization); [backend glossary](../../../../glossary.md#backend); [optimizer state glossary](../../../../glossary.md#term-optimizer-state); [publication glossary](../../../../glossary.md#term-publication))</span>
- Update the owning contract, root dependency rule/routing where needed, architecture test,
  public API guide, Javadocs, glossary impact, and significant decision record.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [architecture test glossary](../../../../glossary.md#term-architecture-test); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Non-goals

Training [resume](../../../../glossary.md#schema-serialization-and-checkpoint); [tokenizer](../../../../glossary.md#frozen-whitespace-tokenizer) parsing; [model](../../../../glossary.md#term-model) lambda/class reconstruction; replacing [Engine](../../../../glossary.md#term-engine)'s
[materialization](../../../../glossary.md#materialization) path; [compiler](../../../../glossary.md#term-compiler)/[prepare](../../../../glossary.md#prepare)/[runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend) access; [persistence](../../../../glossary.md#term-persistence) of [layouts](../../../../glossary.md#layout), [device](../../../../glossary.md#term-device)
[buffers](../../../../glossary.md#term-buffer), or [optimizer state](../../../../glossary.md#term-optimizer-state); automatic cross-[owner](../../../../glossary.md#term-owner) [transaction](../../../../glossary.md#term-transaction); non-[atomic](../../../../glossary.md#resource-transaction-and-rollback) file fallback; a
streaming format or a broad memory-budget redesign.

## Contracts

- [`ARCHITECTURE.md` — Authority, incorporation, and precedence](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence),
  [Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
  [Module-ownership routing](../../../../../ARCHITECTURE.md#module-ownership-routing)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>,
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>, and
  [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements): preserve owners;
  authorize any public Model edge explicitly before implementation.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [`extensions-training.md` — `extensions/nn`](../../../../architecture/contracts/extensions-training.md#extensionsnn)
  and [`extensions/checkpoint`](../../../../architecture/contracts/extensions-training.md#extensionscheckpoint):
  NN owns whole-tree validation/installation; Checkpoint owns model artifact semantics and uses
  only public Engine materialization.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [validation glossary](../../../../glossary.md#term-validation); [Checkpoint glossary](../../../../glossary.md#term-checkpoint); [artifact glossary](../../../../glossary.md#term-artifact); [Engine glossary](../../../../glossary.md#term-engine); [materialization glossary](../../../../glossary.md#materialization))</span>
- [`foundational-modules.md` — `modules/model`](../../../../architecture/contracts/foundational-modules.md#modulesmodel):
  use public Tensor descriptors and compatible host storage, never backend representation.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [descriptors glossary](../../../../glossary.md#term-descriptor); [host storage glossary](../../../../glossary.md#host-storage); [backend glossary](../../../../glossary.md#backend))</span>
- [`runtime-prepare-engine.md` — `modules/engine`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine):
  use the public Engine facade for save-time host materialization only.<span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade); [host glossary](../../../../glossary.md#term-host))</span>

## Dependencies and integration

- Depends on: Checkpoint 0003 (Complete) for implementation; CPU 0024 for direct restored-heap
  Engine acceptance and closure, not for the Checkpoint code or v1 format. NN strict whole-tree
  load and Engine bounded compute already exist; typed inspection/API dependency and target-defined
  v1 aliases are approved, with fail-closed unbound targets.<span data-glossary-navigation="true"> ([Checkpoint glossary](../../../../glossary.md#term-checkpoint); [CPU glossary](../../../../glossary.md#term-cpu); [heap glossary](../../../../glossary.md#term-heap); [Engine glossary](../../../../glossary.md#term-engine); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [dependency glossary](../../../../glossary.md#term-dependency); [target glossary](../../../../glossary.md#term-target); [aliases glossary](../../../../glossary.md#aliasing-and-arena); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- Conflicts with: Checkpoint 0005–0007, Training 0004 integration to the same restore contract,
  and concurrent Checkpoint/NN public API, schema, or owning-contract edits.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema))</span>
- Parallel group: None.
- Common base revision: N/A (serial successor from `cc88cab1`; current `main` is the work base).
- Integration order: Checkpoint implementation/review (recorded below), then CPU 0024 repair and
  independent review, then direct Checkpoint regression and 0004 closure; only then consider 0005.<span data-glossary-navigation="true"> ([regression glossary](../../../../glossary.md#term-regression))</span>
- Integration validation: direct ordinary/restored heap Engine tests, affected CPU and Checkpoint
  checks, and one combined <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>`./gradlew test javadoc` after CPU integration. Earlier successful
  Checkpoint validation remains recorded below.
- Shared-document integration [owner](../../../../glossary.md#term-owner): coordinator for the combined closure; no concurrent writer.

## Files and symbols

- `extensions/checkpoint/.../checkpoint/model/` — public facade/inspection and private
  validated restoration; existing codec/capture remain internal.<span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade); [codec glossary](../../../../glossary.md#term-codec); [capture glossary](../../../../glossary.md#term-capture-meanings))</span>
- `extensions/checkpoint/src/test/.../checkpoint/model/` — file, materialization, restore,
  alias, bit-exactness, bounds, failure, and lifecycle tests.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [alias glossary](../../../../glossary.md#aliasing-and-arena); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- `extensions/checkpoint/build.gradle.kts` and
  `testing/architecture-tests/.../CheckpointDependencyContractTest.java` — publish NN, Engine,
  and direct Model <span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Engine glossary](../../../../glossary.md#term-engine); [Model glossary](../../../../glossary.md#term-model))</span>`api` dependencies and enforce the approved edge set.
- `ARCHITECTURE.md`, `docs/architecture/contracts/extensions-training.md`, focused public
  documentation and ADR — coordinated architectural/API description.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Acceptance criteria

- Save uses one complete NN snapshot and public bounded Engine compute, then the existing
  validated atomic-publication path. Existing target remains protected by that path's stated
  failure semantics; no stronger crash-durability or concurrent-writer promise is made.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [snapshot glossary](../../../../glossary.md#term-snapshot); [Engine glossary](../../../../glossary.md#term-engine); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [publication glossary](../../../../glossary.md#term-publication); [target glossary](../../../../glossary.md#term-target); [durability glossary](../../../../glossary.md#term-durability))</span>
- Inspect returns only after bounded full-file, checksum, model-schema, and payload-correspondence
  validation. Its values are immutable and detached from file/Engine lifetimes.<span data-glossary-navigation="true"> ([checksum glossary](../../../../glossary.md#term-checksum); [model glossary](../../../../glossary.md#term-model); [schema glossary](../../../../glossary.md#term-schema); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [validation glossary](../../../../glossary.md#term-validation); [immutable glossary](../../../../glossary.md#term-immutable); [detached glossary](../../../../glossary.md#detached-model-results))</span>
- Load compares exact expected configuration and optional input digests, validates all entries
  and canonical <span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>[`BOOL`](../../../../glossary.md#data-type--datatype), reconstructs seven data types and rank-zero/empty shapes with bit-exact
  raw values, and calls NN's whole-tree strict loader only after candidate construction.<span data-glossary-navigation="true"> ([rank-zero glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [shapes glossary](../../../../glossary.md#shape))</span>
- Missing/extra paths, kind/type/Shape/gradient mismatch, malformed files, bad digests,
  over-limit data, alias-value conflict, and reconstruction failure leave all target bindings
  unchanged. Successful restore preserves required target alias groups under the selected policy.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [alias glossary](../../../../glossary.md#aliasing-and-arena); [bindings glossary](../../../../glossary.md#term-binding))</span>
- A round trip through Engine proves raw represented bits (including signed zero, NaN payloads,
  low-precision bits, and integer patterns) and correct carrier type. No Checkpoint code reads
  backend storage, creates a second execution pipeline, or stores live Engine/NN references in
  inspection results.<span data-glossary-navigation="true"> ([round trip glossary](../../../../glossary.md#term-round-trip); [represented bits glossary](../../../../glossary.md#term-represented-bits); [signed zero glossary](../../../../glossary.md#term-signed-zero); [NaN glossary](../../../../glossary.md#term-nan); [precision glossary](../../../../glossary.md#term-precision); [carrier glossary](../../../../glossary.md#term-storage-carrier); [Checkpoint glossary](../../../../glossary.md#term-checkpoint); [backend glossary](../../../../glossary.md#backend))</span>

## [Validation](../../../../glossary.md#term-validation)

Worker: focused [model](../../../../glossary.md#term-model)-[checkpoint](../../../../glossary.md#term-checkpoint) tests, then
`./gradlew :extensions:checkpoint:check :extensions:checkpoint:javadoc`.
Integration [owner](../../../../glossary.md#term-owner): focused Checkpoint [architecture test](../../../../glossary.md#term-architecture-test), independent Class C review, link/anchor
and glossary check, `git diff --check`, and one `./gradlew test javadoc` repository gate if the
public [dependency](../../../../glossary.md#term-dependency) contract is changed. Reuse successful worker evidence in review.

## Documentation and review impact

Public [API](../../../../glossary.md#term-application-programming-interface-api), [persistence](../../../../glossary.md#term-persistence), [module](../../../../glossary.md#term-module)-edge [publication](../../../../glossary.md#term-publication), and strict [lifecycle](../../../../glossary.md#lifecycle) require a clean
implementation context and an independent targeted architecture/documentation review. Update
Javadocs and the public guide with a complete save/inspect/load example and explicit bit/[alias](../../../../glossary.md#aliasing-and-arena),
[quiescence](../../../../glossary.md#term-quiescence), failure, and [durability](../../../../glossary.md#term-durability) limits. Check the targeted glossary terms.

## Result

The [Checkpoint](../../../../glossary.md#term-checkpoint)-only public save/inspect/load implementation, typed inspection,
[target](../../../../glossary.md#term-target)-defined [alias](../../../../glossary.md#aliasing-and-arena) [validation](../../../../glossary.md#term-validation), full [BOOL](../../../../glossary.md#data-type--datatype) inspection, raw-bit [heap](../../../../glossary.md#term-heap) reconstruction, coordinated
[Model](../../../../glossary.md#term-model) `api` edge, tests, Javadocs, guide, glossary, contract, [architecture test](../../../../glossary.md#term-architecture-test), and [ADR](../../../../glossary.md#term-architecture-decision-record-adr) are in
the working tree. The v1 envelope and [metadata](../../../../glossary.md#term-metadata) wire are unchanged. Focused Checkpoint and
architecture tests, Checkpoint [Javadoc](../../../../glossary.md#term-javadoc), and the repository `./gradlew test javadoc` gate passed.
The final empty-state/limit focused test addition also passed under the affected-[module](../../../../glossary.md#term-module) check.

Before [CPU](../../../../glossary.md#term-cpu) 0024, direct execution of a restored primitive-array heap [Tensor](../../../../glossary.md#tensor) through
`Engine.compute(tensor.contiguous())` failed with
`bufferSelections[0] is incompatible with prepared executable`. A temporary focused diagnostic
test reproduced the same failure for an ordinary `TensorFactory.fromFlatArray` [`FLOAT64`](../../../../glossary.md#term-float64) heap
[leaf](../../../../glossary.md#term-leaf), independent of Checkpoint load. [Engine](../../../../glossary.md#term-engine)'s compute flow prepares before [borrowing](../../../../glossary.md#term-borrowing) input
storage, while CPU preparation's empty [carrier](../../../../glossary.md#term-storage-carrier) pattern selects `MEMORY_SEGMENT`; at bind the
[borrowed](../../../../glossary.md#ownership-borrowing-and-lease) heap carrier is an array and violates the specialized pattern. The earlier raw-bit test
copied restored public [host](../../../../glossary.md#term-host) bytes into native test storage before an Engine [oracle](../../../../glossary.md#term-test-oracle). That
proved bits and carrier reconstruction but did not meet direct restored-Tensor executability.
The diagnostic test was removed after recording the failure to keep the Checkpoint test suite
green; the focused failure output was `ModelCheckpointTest.diagnosticOrdinaryHeapLeafCompute`
under `./gradlew :extensions:checkpoint:test --tests
'*ModelCheckpointTest.diagnosticOrdinaryHeapLeafCompute'`.

The user approved separate CPU follow-up work, not expansion of this Checkpoint implementation
[scope](../../../../glossary.md#term-scope). CPU 0024 now binds the prepared segment form to the exact heap segment, with its initial
P1 mixed-carrier overlap finding fixed and clean independent re-review approved. The revised
`ModelCheckpointTest.savesInspectsAndLoadsSevenTypesWithExactBitsAndNativeCarriers` now runs
direct save→inspect→load→`Engine.compute(tensor.contiguous(), 1024)` on each restored heap
Tensor: all seven data types, [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix), and empty [shape](../../../../glossary.md#shape) pass with exact bits and carrier checks.
`./gradlew :extensions:checkpoint:check` passed; `ModelCheckpointTest` reports 5 tests and zero
failures. The targeted documentation pass updated `docs/api/model-checkpoint-api.md`,
`docs/api/training-api.md`, and ADR 0034, checked links and relevant glossary terms, and reused
successful executable evidence. The prior independent Checkpoint Class C review had no
high-severity findings; the public Model edge, architecture test, Javadocs, and user guide are
complete. This [planning](../../../../glossary.md#term-planning) closure made no further Checkpoint code or [API](../../../../glossary.md#term-application-programming-interface-api) documentation edits.

The final integrated command with `SYNAPTIK_METAL_TEST_LIBRARY` set to the local source-matched
[dylib](../../../../glossary.md#term-dylib), `./gradlew test javadoc --max-workers=2 --console=plain`, passed (105 tasks: 1 executed,
104 up-to-date). An earlier unconfigured [run](../../../../glossary.md#term-run) failed only 15 [Metal](../../../../glossary.md#term-metal) tests requiring that variable;
it did not identify a Checkpoint defect. No unresolved 0004 acceptance gate remains; future
Checkpoint 0005–0007 and Training 0004 remain Draft.

Status: Complete
