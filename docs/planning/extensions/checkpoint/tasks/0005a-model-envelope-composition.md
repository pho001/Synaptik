# Task 0005A: Model-envelope composition

## Status

Complete — implementation, independent Class C API/documentation review, and coordinator gates
are accepted. 0006 remains Draft, outside the build, and unauthorized; no new Ready frontier.

## Change class

Class C — supported model composition, detached payload ownership, and cross-owner recovery facts.

## Goal

Expose bounded model capture, inspection, canonical bytes, and strict load through the existing
ModelCheckpoint and ArtifactEnvelope so a future optional consumer can compare complete model
parameter descriptors/raw logical bytes with TrainingState without private access or file staging.

## Scope

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

Use the existing internal ModelArtifactCapture, ModelArtifactCodec.decode/preflight, and
ModelStateRestoration. Delegate file save/inspect/load to the same semantic machinery while
preserving file rereads, IOException declarations, existing failure types and validation ordering,
and publication policy. Capture retains current unchecked preflight/materialization failures and
declares checked ArtifactFormatException for the final semantic decode. Required references are
non-null; input digest alone may be absent; digests retain the exact 32-byte SHA-256 policy.

## Non-goals

No new production types or package moves, public model-image wrapper, codec/schema/wire change,
registry, file-staging composition, Training validator, general transaction, backend access,
optional module/build/dependency edit, or 0006 implementation. No broader caller-state codecs.
Model + TrainingState is the selected future saved-state product, not automatic exact continuation;
cursor/RNG/modes/effective contexts/scheduler/scaler remain application-owned.

## Contracts

- [Root: Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)
  routes this concern to one scoped owner.
- Root [Module-ownership routing](../../../../../ARCHITECTURE.md#module-ownership-routing),
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules),
  [Core invariants](../../../../../ARCHITECTURE.md#core-invariants), and
  [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- [Extensions/training: `extensions/checkpoint`](../../../../architecture/contracts/extensions-training.md#extensionscheckpoint)
  owns the approved envelope API, v1 schema, limits, bound-target aliases, and installation boundary.
- [Extensions/training: `extensions/nn`](../../../../architecture/contracts/extensions-training.md#extensionsnn)
  owns state meaning and strict whole-tree installation.
- [Extensions/training: Optimizer/training lifecycle](../../../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle)
  preserves owner validation, private-candidate ordering, exclusion, and exact-claim limits.
- [ADR 0036](../../../../design/decisions/0036-model-envelope-composition.md) records rationale,
  not additional authority. Stop and report any applicable contract ambiguity.

## Dependencies and integration

- Depends on: Checkpoint 0004A and Checkpoint 0005, both Complete; approved envelope contract and
  independent contract/brief review before Ready.
- Conflicts with: writers to ModelCheckpoint/model semantics, artifact v1/limits, the owning
  Checkpoint/Training lifecycle contract, and shared Checkpoint/Training planning documents.
- Parallel group: None; serial implementation followed by independent review.
- Common base revision: `fbd14c13bcba4d82e875d866336cd1f5cc3448a9`; serial implementation
  branch `work/checkpoint-model-envelope`.
- Integration order: contract/brief review → Ready verification → implementation → independent
  API/documentation review → coordinator acceptance; 0006 stays Draft.
- Integration validation: checkpoint module check/Javadoc; CheckpointDependencyContractTest;
  PrivateTrainingRestoreIntegrationTest and EngineCpuHeapContiguousIntegrationTest; one coordinator
  repository test/Javadoc gate for the implemented cross-owner value boundary.
- Shared-document integration owner: Training/Checkpoint coordinator; independent review finalizes
  focused API/Javadoc documentation serially, coordinator synchronizes masters/roadmap.

## Files and symbols

Under `extensions/checkpoint/src/main/java/io/github/pho001/synaptik/checkpoint/`:

- `model/ModelCheckpoint.java` — new methods and shared file/envelope semantics.
- `model/ModelCheckpointInspection.java` — affected descriptor/validation Javadoc, existing type.
- `model/ModelArtifactCapture.java`, `model/ModelArtifactCodec.java`,
  `model/ModelArtifactCandidate.java`, `model/ModelStateRestoration.java` — reuse internal symbols;
  change only if needed for the bounded facade. All remain package-private.
- Read `artifact/ArtifactEnvelope.java`, `ArtifactLimits.java`, `ArtifactWire.java`, and
  `ArtifactFile.java`; their public API and wire format are unchanged, no new validation abstraction.

Focused tests under `extensions/checkpoint/src/test/java/io/github/pho001/synaptik/checkpoint/`:
`model/ModelCheckpointTest.java`, `model/ModelArtifactTest.java`, `artifact/ArtifactWireTest.java`.
Reuse existing raw-dtype/alias/failure cases; add only the small public-use external-package test
`extensions/checkpoint/src/test/java/io/github/pho001/synaptik/checkpoint/api/ModelCheckpointEnvelopeApiTest.java`,
or compile the final API-guide consumer outside `checkpoint.model`; no reflection.
Read-only cross-module checks:
`testing/architecture-tests/src/test/java/io/github/pho001/synaptik/testing/architecture/CheckpointDependencyContractTest.java`;
`testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/PrivateTrainingRestoreIntegrationTest.java`
and `EngineCpuHeapContiguousIntegrationTest.java`. No cross-module source edit is authorized.

## Acceptance criteria

- Capture returns a fully detached, semantically validated model.v1 envelope, with one NN snapshot,
  current bounded Engine materialization, exact bits and source bindings unchanged. Empty state
  performs no Engine call; source Arena/Engine closure does not invalidate returned values.
- All envelope operations independently run existing decode/preflight, including every limit:
  complete file-equivalent size, manifest size, entry count, kind/key lengths, individual payloads,
  aggregate schema plus every aliased path. Cover exact ceilings and one-below rejection without
  an ArtifactFile.read prerequisite; limits are not a peak-heap/workspace/copy budget.
- Canonical map includes all parameters and buffers in metadata order under exact Unicode NN paths
  (not sorted names). Its structure rejects mutation. Per-call, per-path buffers are fresh,
  independent, read-only, position zero, exact-length, BIG_ENDIAN; moving a cursor, limit, or order
  cannot affect other calls/views. Do not claim ByteBuffer cursors are deeply immutable.
- Exercise FLOAT64, FLOAT32, BFLOAT16, FLOAT16, INT32, INT64, BOOL, scalar and empty payload/state,
  native and matching heap leaves, aliases/mixed roles, signed zero, infinities, and raw NaN
  sign/payload bits. No numeric conversion, NaN normalization, or source-alias inference.
- Reject wrong kind, malformed/unsupported schema, bad BOOL, bounds, missing/extra/wrong ordinal
  payloads, digest presence/byte mismatch, descriptor/path mismatch, alias-byte conflict, and
  unbound/lazy target before any binding mutation. Check a late mismatch preserves earlier bindings.
- Envelope load fully validates/reconstructs before one NN strict whole-tree installation attempt;
  target identity defines aliases, with one fresh Tensor per group and existing gradient policy.
- Prior inspection is never a trusted credential for bytes/load. Use an inspected valid envelope,
  then a different invalid envelope to prove independent validation; file load rereads changed files.
- File/envelope parity covers inspection/digests/descriptors, canonical raw payloads, restored alias
  identity, and deterministic v1 file bytes. Existing file-facade argument/failure ordering and
  IOException/ArtifactFormatException policy remain unchanged; no silent exception translation.
- A small consumer compiled outside `checkpoint.model` uses only public APIs to demonstrate complete
  parameter descriptors/raw logical bytes by exact path for future TrainingState comparison.
  It does not import Training, use reflection, or implement its validator/adapter; reuse existing
  owner tests for the broad matrix. Inspection and returned bytes carry no resource handles.
- All four helpers remain internal; existing dependency rules and absent adapter are preserved.

## Validation

Worker focused development checks:

```bash
./gradlew :extensions:checkpoint:test --tests '*ModelCheckpointTest' --tests '*ModelArtifactTest'
```

After executable code stabilizes, run the affected-module gate once:

```bash
./gradlew :extensions:checkpoint:check :extensions:checkpoint:javadoc
```

Coordinator integration checks after ordered integration (reuse successful unchanged evidence):

```bash
./gradlew :testing:architecture-tests:test --tests '*CheckpointDependencyContractTest'
./gradlew :testing:integration-tests:test --tests '*PrivateTrainingRestoreIntegrationTest' --tests '*EngineCpuHeapContiguousIntegrationTest'
git diff --check
```

The coordinator gate covers cross-owner values and file-facade delegation. Use source-matched
`SYNAPTIK_METAL_TEST_LIBRARY` per the
[native build/test setup](../../../../../native/metal-macos-arm64/README.md#validation), recording
path, provenance, and actual device outcome; missing setup is not a passing gate. Dependency-test
inputs remain unchanged. Review/closure reuse accepted executable evidence; final reviewer Javadoc
and changed links/anchors must be validated.

## Documentation and review impact

Independent targeted API/ownership review is mandatory before implementation acceptance. Finalize
ModelCheckpoint and affected helper/inspection Javadocs, including every parameter/result/failure,
byte-order/cursor semantics and detached lifetime. Update `docs/api/model-checkpoint-api.md` when
the API exists, with a complete file-free example; review `docs/api/checkpoint-artifact-api.md`,
`docs/architecture/module-boundaries.md`, and glossary checkpoint/state terms only where affected.
Apply documentation rules, general/API/example profiles; preserve planned versus runnable labels.
No terminology change is required by this plan; no glossary edit is needed now.

## Result

- Implemented all four envelope operations, reusing the private capture/codec/restoration core;
  file behavior/parity/error ordering unchanged; all seven types retain raw bits and fresh read-only views.
- Production: `ModelCheckpoint.java`, `ModelCheckpointInspection.java` only; extended `ModelCheckpointTest`/`ModelArtifactTest`; added external-package `ModelCheckpointEnvelopeApiTest`.
- Independent Class C source/helper/test/ownership and API/documentation review: APPROVE, no issues.
  API guide complete Java example compiled and ran PASS; final checkpoint Javadoc PASS;
  71 documentation links/anchors and new-file whitespace/diff checks PASS (general/API/example profiles).
- Root/scoped contract, module-boundary explanation, and ADR 0036 labels now implemented, not a new decision.
  Artifact guide and glossary unchanged: meaning, ownership, and terminology remain accurate.
- Worker commands above: focused 17 tests PASS; checkpoint check/Javadoc 33 tests, 0 failures/skips PASS.
- Coordinator: `./gradlew :testing:architecture-tests:test --tests '*CheckpointDependencyContractTest' :testing:integration-tests:test --tests '*PrivateTrainingRestoreIntegrationTest' --tests '*EngineCpuHeapContiguousIntegrationTest' --max-workers=2 --console=plain` — BUILD SUCCESSFUL (1s).
  Architecture 1 PASS FROM-CACHE (unchanged edges/inputs); private restore 2 + heap-contiguous 3 fresh PASS, 0 failures/skips.
- Final gate: `SYNAPTIK_METAL_TEST_LIBRARY=/private/tmp/synaptik-envelope-native.CjXawZ/native/metal-macos-arm64/build/package-v1/macos-arm64/libsynaptik_metal_foundation.dylib ./gradlew test javadoc --max-workers=2 --console=plain` — BUILD SUCCESSFUL (6s); 105 tasks: 4 executed, 1 cached, 100 up-to-date.
  Integration: 125 tests, 1 skip, 0 failures/errors; all 77 tests in 8 Metal-named classes fresh PASS,
  zero device skips (XML timestamps 2026-10-14T12:36:47–12:36:50); unchanged modules reused cache/up-to-date results.
  Sole skip: `EngineModelAutotuningIntegrationTest.eligibleCpuAlternativesCompleteBothPublicTuningPhases(Path)`;
  assumption “host CPU fixture has no eligible local-workload alternatives” (expected host fixture, not device/checkpoint failure).
- Native package freshly built from exact common-base source in an isolated worktree: `build.sh`, fixed ad-hoc codesign,
  `package-local.sh`, `verify-package.sh` PASS; SHA-256 `03cba3ebfa0bbb78c1bff369ff3451c591846fd975c0034fa7f67fdd2209985a`.
  Native/Metal/build sources unchanged; no ABI/schema/dependency change; existing main native artifacts untouched.
- Closure: this brief, both masters, roadmap only; planning links/anchors/whitespace/diff PASS; no unresolved issues. Future 0006 needs concrete composition/codec/build-gate design, remains Draft/not included; no new Ready frontier.

Status: Complete
