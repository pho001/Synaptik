# Task 0005A: [Model](../../../../glossary.md#term-model)-envelope composition

## Status

Complete — implementation, independent Class C [API](../../../../glossary.md#term-application-programming-interface-api)/documentation review, and coordinator gates
are accepted. 0006 remains Draft, outside the build, and unauthorized; no new [Ready](../../../../glossary.md#term-ready) frontier.

## Change class

Class C — supported [model](../../../../glossary.md#term-model) composition, [detached](../../../../glossary.md#detached-model-results) [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) [ownership](../../../../glossary.md#term-ownership-meanings), and cross-[owner](../../../../glossary.md#term-owner) recovery facts.

## Goal

Expose bounded [model](../../../../glossary.md#term-model) [capture](../../../../glossary.md#term-capture-meanings), inspection, [canonical](../../../../glossary.md#term-canonical-form) bytes, and strict load through the existing
ModelCheckpoint and ArtifactEnvelope so a future optional consumer can compare complete model
[parameter](../../../../glossary.md#term-parameter) [descriptors](../../../../glossary.md#term-descriptor)/raw logical bytes with TrainingState without private access or file staging.

## [Scope](../../../../glossary.md#term-scope)

Implemented only these static ModelCheckpoint operations:

```text
ArtifactEnvelope capture(Module module, Engine engine, byte[] configDigest,
                         byte[] inputDigest, ArtifactLimits limits)
    throws ArtifactFormatException
ModelCheckpointInspection inspect(ArtifactEnvelope envelope, ArtifactLimits limits)
    throws ArtifactFormatException
Map<String, ByteBuffer> canonicalStateBytes(ArtifactEnvelope envelope, ArtifactLimits limits)
    throws ArtifactFormatException
void load(ArtifactEnvelope envelope, Module target, byte[] expectedConfigDigest,
          byte[] expectedInputDigest, ArtifactLimits limits)
    throws ArtifactFormatException
```

Use the existing internal ModelArtifactCapture, ModelArtifactCodec.decode/[preflight](../../../../glossary.md#term-preflight), and
ModelStateRestoration. Delegate file save/inspect/load to the same semantic machinery while
preserving file rereads, IOException declarations, existing failure types and [validation](../../../../glossary.md#term-validation) ordering,
and [publication](../../../../glossary.md#term-publication) policy. [Capture](../../../../glossary.md#term-capture-meanings) retains current unchecked preflight/[materialization](../../../../glossary.md#materialization) failures and
declares checked ArtifactFormatException for the final semantic decode. Required references are
non-null; input [digest](../../../../glossary.md#term-digest) alone may be absent; digests retain the exact 32-byte [SHA-256](../../../../glossary.md#term-sha-256) policy.

## Non-goals

No new production types or package moves, public [model](../../../../glossary.md#term-model)-image wrapper, [codec](../../../../glossary.md#term-codec)/[schema](../../../../glossary.md#term-schema)/wire change,
[registry](../../../../glossary.md#term-registry), file-staging composition, Training validator, general [transaction](../../../../glossary.md#term-transaction), [backend](../../../../glossary.md#backend) access,
optional [module](../../../../glossary.md#term-module)/build/[dependency](../../../../glossary.md#term-dependency) edit, or 0006 implementation. No broader caller-state codecs.
Model + TrainingState is the selected future saved-state product, not automatic exact continuation;
cursor/[RNG](../../../../glossary.md#graph-rng-state--graphrngstate)/modes/effective contexts/scheduler/scaler remain application-owned.

## Contracts

- [Root: Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope); [normative glossary](../../../../glossary.md#term-normative))</span>
  routes this concern to one scoped owner.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
- Root [Module-ownership routing](../../../../../ARCHITECTURE.md#module-ownership-routing)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>,
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>,
  [Core invariants](../../../../../ARCHITECTURE.md#core-invariants), and
  [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- [Extensions/training: `extensions/checkpoint`](../../../../architecture/contracts/extensions-training.md#extensionscheckpoint)
  owns the approved envelope API, v1 schema, limits, bound-target aliases, and installation boundary.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [schema glossary](../../../../glossary.md#term-schema); [target glossary](../../../../glossary.md#term-target); [aliases glossary](../../../../glossary.md#aliasing-and-arena))</span>
- [Extensions/training: `extensions/nn`](../../../../architecture/contracts/extensions-training.md#extensionsnn)
  owns state meaning and strict whole-tree installation.
- [Extensions/training: Optimizer/training lifecycle](../../../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle)<span data-glossary-navigation="true"> ([Optimizer glossary](../../../../glossary.md#term-optimizer); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
  preserves owner validation, private-candidate ordering, exclusion, and exact-claim limits.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- [ADR 0036](../../../../design/decisions/0036-model-envelope-composition.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span> records rationale,
  not additional authority. Stop and report any applicable contract ambiguity.

## Dependencies and integration

- Depends on: Checkpoint 0004A and Checkpoint 0005, both Complete; approved envelope contract and
  independent contract/brief review before Ready.<span data-glossary-navigation="true"> ([Checkpoint glossary](../../../../glossary.md#term-checkpoint); [Ready glossary](../../../../glossary.md#term-ready))</span>
- Conflicts with: writers to ModelCheckpoint/model semantics, artifact v1/limits, the owning
  Checkpoint/Training lifecycle contract, and shared Checkpoint/Training planning documents.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [artifact glossary](../../../../glossary.md#term-artifact); [lifecycle glossary](../../../../glossary.md#lifecycle); [planning glossary](../../../../glossary.md#term-planning))</span>
- Parallel group: None; serial implementation followed by independent review.
- Common base revision: `fbd14c13bcba4d82e875d866336cd1f5cc3448a9`; serial implementation
  branch `work/checkpoint-model-envelope`.
- Integration order: contract/brief review → Ready verification → implementation → independent
  API/documentation review → coordinator acceptance; 0006 stays Draft.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Integration validation: checkpoint module check/Javadoc; CheckpointDependencyContractTest;
  PrivateTrainingRestoreIntegrationTest and EngineCpuHeapContiguousIntegrationTest; one coordinator
  repository test/Javadoc gate for the implemented cross-owner value boundary.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [module glossary](../../../../glossary.md#term-module); [Javadoc glossary](../../../../glossary.md#term-javadoc); [owner glossary](../../../../glossary.md#term-owner))</span>
- Shared-document integration owner: Training/Checkpoint coordinator; independent review finalizes
  focused API/Javadoc documentation serially, coordinator synchronizes masters/roadmap.

## Files and symbols

Under `extensions/checkpoint/src/main/java/io/github/pho001/synaptik/checkpoint/`:

- `model/ModelCheckpoint.java` — new methods and shared file/envelope semantics.
- `model/ModelCheckpointInspection.java` — affected [descriptor](../../../../glossary.md#term-descriptor)/[validation](../../../../glossary.md#term-validation) [Javadoc](../../../../glossary.md#term-javadoc), existing type.
- `model/ModelArtifactCapture.java`, `model/ModelArtifactCodec.java`,
  `model/ModelArtifactCandidate.java`, `model/ModelStateRestoration.java` — reuse internal symbols;
  change only if needed for the bounded facade. All remain package-private.<span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade))</span>
- Read `artifact/ArtifactEnvelope.java`, `ArtifactLimits.java`, `ArtifactWire.java`, and
  `ArtifactFile.java`; their public [API](../../../../glossary.md#term-application-programming-interface-api) and [wire format](../../../../glossary.md#term-wire-format) are unchanged, no new validation abstraction.

Focused tests under `extensions/checkpoint/src/test/java/io/github/pho001/synaptik/checkpoint/`:
`model/ModelCheckpointTest.java`, `model/ModelArtifactTest.java`, `artifact/ArtifactWireTest.java`.
Reuse existing raw-[dtype](../../../../glossary.md#data-type--datatype)/[alias](../../../../glossary.md#aliasing-and-arena)/failure cases; add only the small public-use external-package test
`extensions/checkpoint/src/test/java/io/github/pho001/synaptik/checkpoint/api/ModelCheckpointEnvelopeApiTest.java`,
or [compile](../../../../glossary.md#compile) the final API-guide consumer outside `checkpoint.model`; no reflection.
Read-only cross-[module](../../../../glossary.md#term-module) checks:
`testing/architecture-tests/src/test/java/io/github/pho001/synaptik/testing/architecture/CheckpointDependencyContractTest.java`;
`testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/PrivateTrainingRestoreIntegrationTest.java`
and `EngineCpuHeapContiguousIntegrationTest.java`. No cross-module source edit is authorized.

## Acceptance criteria

- Capture returns a fully detached, semantically validated model.v1 envelope, with one NN snapshot,
  current bounded Engine materialization, exact bits and source bindings unchanged. Empty state
  performs no Engine call; source Arena/Engine closure does not invalidate returned values.<span data-glossary-navigation="true"> ([Capture glossary](../../../../glossary.md#term-capture-meanings); [detached glossary](../../../../glossary.md#detached-model-results); [model glossary](../../../../glossary.md#term-model); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [snapshot glossary](../../../../glossary.md#term-snapshot); [Engine glossary](../../../../glossary.md#term-engine); [materialization glossary](../../../../glossary.md#materialization); [bindings glossary](../../../../glossary.md#term-binding); [Arena glossary](../../../../glossary.md#term-arena))</span>
- All envelope operations independently run existing decode/preflight, including every limit:
  complete file-equivalent size, manifest size, entry count, kind/key lengths, individual payloads,
  aggregate schema plus every aliased path. Cover exact ceilings and one-below rejection without
  an ArtifactFile.read prerequisite; limits are not a peak-heap/workspace/copy budget.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [preflight glossary](../../../../glossary.md#term-preflight); [manifest glossary](../../../../glossary.md#term-manifest); [schema glossary](../../../../glossary.md#term-schema); [heap glossary](../../../../glossary.md#term-heap))</span>
- Canonical map includes all parameters and buffers in metadata order under exact Unicode NN paths
  (not sorted names). Its structure rejects mutation. Per-call, per-path buffers are fresh,
  independent, read-only, position zero, exact-length, BIG_ENDIAN; moving a cursor, limit, or order
  cannot affect other calls/views. Do not claim ByteBuffer cursors are deeply immutable.<span data-glossary-navigation="true"> ([Canonical glossary](../../../../glossary.md#term-canonical-form); [parameters glossary](../../../../glossary.md#term-parameter); [buffers glossary](../../../../glossary.md#term-buffer); [metadata glossary](../../../../glossary.md#term-metadata); [Unicode glossary](../../../../glossary.md#term-unicode); [views glossary](../../../../glossary.md#term-view-meanings); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- Exercise FLOAT64, FLOAT32, BFLOAT16, FLOAT16, INT32, INT64, BOOL, scalar and empty payload/state,
  native and matching heap leaves, aliases/mixed roles, signed zero, infinities, and raw NaN
  sign/payload bits. No numeric conversion, NaN normalization, or source-alias inference.<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT16 glossary](../../../../glossary.md#term-float16); [INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [BOOL glossary](../../../../glossary.md#data-type--datatype); [scalar glossary](../../../../glossary.md#term-scalar); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [aliases glossary](../../../../glossary.md#aliasing-and-arena); [signed zero glossary](../../../../glossary.md#term-signed-zero); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [NaN glossary](../../../../glossary.md#term-nan); [normalization glossary](../../../../glossary.md#term-normalization); [inference glossary](../../../../glossary.md#term-inference))</span>
- Reject wrong kind, malformed/unsupported schema, bad BOOL, bounds, missing/extra/wrong ordinal
  payloads, digest presence/byte mismatch, descriptor/path mismatch, alias-byte conflict, and
  unbound/lazy target before any binding mutation. Check a late mismatch preserves earlier bindings.<span data-glossary-navigation="true"> ([digest glossary](../../../../glossary.md#term-digest); [descriptor glossary](../../../../glossary.md#term-descriptor); [lazy glossary](../../../../glossary.md#term-lazy-evaluation); [target glossary](../../../../glossary.md#term-target))</span>
- Envelope load fully validates/reconstructs before one NN strict whole-tree installation attempt;
  target identity defines aliases, with one fresh Tensor per group and existing gradient policy.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [Tensor glossary](../../../../glossary.md#tensor); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Prior inspection is never a trusted credential for bytes/load. Use an inspected valid envelope,
  then a different invalid envelope to prove independent validation; file load rereads changed files.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- File/envelope parity covers inspection/digests/descriptors, canonical raw payloads, restored alias
  identity, and deterministic v1 file bytes. Existing file-facade argument/failure ordering and
  IOException/ArtifactFormatException policy remain unchanged; no silent exception translation.<span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade); [argument glossary](../../../../glossary.md#term-argument))</span>
- A small consumer compiled outside `checkpoint.model` uses only public APIs to demonstrate complete
  parameter descriptors/raw logical bytes by exact path for future TrainingState comparison.
  It does not import Training, use reflection, or implement its validator/adapter; reuse existing
  owner tests for the broad matrix. Inspection and returned bytes carry no resource handles.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [carry glossary](../../../../glossary.md#term-carry))</span>
- All four helpers remain internal; existing [dependency](../../../../glossary.md#term-dependency) rules and absent adapter are preserved.

## [Validation](../../../../glossary.md#term-validation)

Worker focused development checks:

```bash
./gradlew :extensions:checkpoint:test --tests '*ModelCheckpointTest' --tests '*ModelArtifactTest'
```

After executable code stabilizes, [run](../../../../glossary.md#term-run) the affected-[module](../../../../glossary.md#term-module) gate once:

```bash
./gradlew :extensions:checkpoint:check :extensions:checkpoint:javadoc
```

Coordinator integration checks after ordered integration (reuse successful unchanged evidence):

```bash
./gradlew :testing:architecture-tests:test --tests '*CheckpointDependencyContractTest'
./gradlew :testing:integration-tests:test --tests '*PrivateTrainingRestoreIntegrationTest' --tests '*EngineCpuHeapContiguousIntegrationTest'
git diff --check
```

The coordinator gate covers cross-[owner](../../../../glossary.md#term-owner) values and file-[facade](../../../../glossary.md#term-facade) delegation. Use source-matched
`SYNAPTIK_METAL_TEST_LIBRARY` per the
[native build/test setup](../../../../../modules/backend/metal/macos-arm64/README.md#validation), recording
path, [provenance](../../../../glossary.md#provenance), and actual [device](../../../../glossary.md#term-device) outcome; missing setup is not a passing gate. [Dependency](../../../../glossary.md#term-dependency)-test
inputs remain unchanged. Review/closure reuse accepted executable evidence; final reviewer [Javadoc](../../../../glossary.md#term-javadoc)
and changed links/anchors must be validated.

## Documentation and review impact

Independent targeted [API](../../../../glossary.md#term-application-programming-interface-api)/[ownership](../../../../glossary.md#term-ownership-meanings) review is mandatory before implementation acceptance. Finalize
ModelCheckpoint and affected helper/inspection Javadocs, including every [parameter](../../../../glossary.md#term-parameter)/result/failure,
byte-order/cursor semantics and [detached](../../../../glossary.md#detached-model-results) [lifetime](../../../../glossary.md#term-lifetime). Update `docs/api/model-checkpoint-api.md` when
the API exists, with a complete file-free example; review `docs/api/checkpoint-artifact-api.md`,
`docs/architecture/module-boundaries.md`, and glossary [checkpoint](../../../../glossary.md#term-checkpoint)/state terms only where affected.
Apply documentation rules, general/API/example profiles; preserve planned versus runnable labels.
No terminology change is required by this plan; no glossary edit is needed now.

## Result

- Implemented all four envelope operations, reusing the private capture/codec/restoration core;
  file behavior/parity/error ordering unchanged; all seven types retain raw bits and fresh read-only views.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [raw bits glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [views glossary](../../../../glossary.md#term-view-meanings))</span>
- Production: `ModelCheckpoint.java`, `ModelCheckpointInspection.java` only; extended `ModelCheckpointTest`/`ModelArtifactTest`; added external-package `ModelCheckpointEnvelopeApiTest`.
- Independent Class C source/helper/test/ownership and API/documentation review: APPROVE, no issues.
  API guide complete Java example compiled and ran PASS; final checkpoint Javadoc PASS;
  71 documentation links/anchors and new-file whitespace/diff checks PASS (general/API/example profiles).<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Root/scoped contract, module-boundary explanation, and ADR 0036 labels now implemented, not a new decision.
  Artifact guide and glossary unchanged: meaning, ownership, and terminology remain accurate.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Artifact glossary](../../../../glossary.md#term-artifact))</span>
- Worker commands above: focused 17 tests PASS; [checkpoint](../../../../glossary.md#term-checkpoint) check/[Javadoc](../../../../glossary.md#term-javadoc) 33 tests, 0 failures/skips PASS.
- Coordinator: `./gradlew :testing:architecture-tests:test --tests '*CheckpointDependencyContractTest' :testing:integration-tests:test --tests '*PrivateTrainingRestoreIntegrationTest' --tests '*EngineCpuHeapContiguousIntegrationTest' --max-workers=2 --console=plain` — BUILD SUCCESSFUL (1s).
  Architecture 1 PASS FROM-CACHE (unchanged edges/inputs); private restore 2 + heap-contiguous 3 fresh PASS, 0 failures/skips.<span data-glossary-navigation="true"> ([CACHE glossary](../../../../glossary.md#term-cache); [heap glossary](../../../../glossary.md#term-heap); [contiguous glossary](../../../../glossary.md#term-contiguous))</span>
- Final gate: `SYNAPTIK_METAL_TEST_LIBRARY=/private/tmp/synaptik-envelope-native.CjXawZ/native/metal-macos-arm64/build/package-v1/macos-arm64/libsynaptik_metal_foundation.dylib ./gradlew test javadoc --max-workers=2 --console=plain` — BUILD SUCCESSFUL (6s); 105 tasks: 4 executed, 1 cached, 100 up-to-date.
  Integration: 125 tests, 1 skip, 0 failures/errors; all 77 tests in 8 Metal-named classes fresh PASS,
  zero device skips (XML timestamps 2026-10-14T12:36:47–12:36:50); unchanged modules reused cache/up-to-date results.
  Sole skip: <span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [device glossary](../../../../glossary.md#term-device))</span>`EngineModelAutotuningIntegrationTest.eligibleCpuAlternativesCompleteBothPublicTuningPhases(Path)`;
  assumption “host CPU fixture has no eligible local-workload alternatives” (expected host fixture, not device/checkpoint failure).<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [CPU glossary](../../../../glossary.md#term-cpu); [fixture glossary](../../../../glossary.md#term-fixture))</span>
- Native package freshly built from exact common-base source in an isolated worktree: `build.sh`, fixed ad-hoc codesign,
  `package-local.sh`, `verify-package.sh` PASS; [SHA-256](../../../../glossary.md#term-sha-256) `03cba3ebfa0bbb78c1bff369ff3451c591846fd975c0034fa7f67fdd2209985a`.
  Native/Metal/build sources unchanged; no ABI/schema/dependency change; existing main native artifacts untouched.<span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [schema glossary](../../../../glossary.md#term-schema); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Closure: this brief, both masters, roadmap only; [planning](../../../../glossary.md#term-planning) links/anchors/whitespace/diff PASS; no unresolved issues. Future 0006 needs concrete composition/[codec](../../../../glossary.md#term-codec)/build-gate design, remains Draft/not included; no new [Ready](../../../../glossary.md#term-ready) frontier.

Status: Complete
