# Task 0009: Owned Model activation source and persistence

## Status

Complete — main accepted 2026-10-19 after Archimedes's final Astra/xhigh APPROVE, zero findings,
and source-matched integration/native root gates. Checkpoint/Compiler/NN frontiers: None.
No implementation issues or required follow-up. No commit or push authorized.

## Change class

Class C — atomic Compiler cold-source and Checkpoint wire/admission capability, consumed by owned
NN Models. Clean implementation and independent targeted code/documentation review are mandatory.

## Goal

Allow owned `Model.defineTensor`/`graph` to construct, compute, train, save and restore a bounded
multilayer perceptron (MLP: two Linear layers with an intervening activation) using RELU, TANH or
SIGMOID. Retain ordinary Tensor semantics and the completed NN0028 ownership/execution lifecycle.

## Scope and atomic boundary

Retain ADD/MUL/NEG/MATMUL and rank-two PERMUTE[1,0]; add only UnaryElementwiseKind.RELU/TANH/SIGMOID
with NoOperationAttrs.INSTANCE, for eight source kinds total. Compiler owns complete static FORWARD
admission/inference and reconstruction through `Tensor.relu/tanh/sigmoid`; Checkpoint owns explicit
versioned wires, normalized source associations, bounds and publication. NN is an unchanged consumer
apart from coverage/Javadoc: `OwnedModelSource.acquireDefinition` and `Model.graph` already call
`GraphCheckpoint.copy`; the former five-kind limit blocked construction/compute, not merely save.

Integrate Compiler and Checkpoint atomically: remove the former encoder's PERMUTE fallback,
explicitly admit kind/attrs/arity during dry sizing and make reconstruction's PERMUTE branch explicit.
No Compiler-only release, staging abstraction or parallel task.

## Accepted format and compatibility policy

Select the smallest sufficient closed version without a public selector or wrapper:

- Existing five-kind graphs, including zero-node graphs, continue to write byte-identical v1:
  kind `graph-model.v1`, magic `SYNGRF01`, i32 version 1, opcodes 1..5 unchanged.
- A source containing any added activation writes kind `graph-model.v2`, magic `SYNGRF02`, i32
  version 2. Keep every other v1 field/order, descriptor/role/dtype ID, payload and limit unchanged;
  append u8 6=RELU, 7=TANH, 8=SIGMOID, each one preceding input ID and no attribute bytes.
- Load recognizes only the two exact kind/magic/version tuples. V1 still rejects opcodes 6..8;
  v2 admits only 1..8 (also accepts five-kind-only data; re-save selects v1). Reject mismatched
  tuples, unknown opcodes/versions and malformed fields before import/reconstruction/publication.
  New readers retain all valid v1 files; old graph readers fail closed on v2, not reinterpret it.
- ArtifactFile v1 framing and model.v1/whole-model.v1 readers remain unchanged and separate.
  Graph kinds have equal encoded length, so existing manifest/file sizing constants remain valid;
  verify exact-size/one-below limits rather than assuming this. No format registry or enum ordinals.

## Contracts and required cutover

Start at [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts).
Read these exact headings, not historical task catalogs:

- Root [Core lifecycle](../../../../../ARCHITECTURE.md#core-lifecycle) and
  [Core invariants](../../../../../ARCHITECTURE.md#core-invariants): immutable Tensor/graphs,
  Model-owned semantics, exact stored state and unchanged compile/prepare/run separation.
- Root [Module-ownership routing](../../../../../ARCHITECTURE.md#module-ownership-routing),
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules),
  [Documentation authority and architecture updates](../../../../../ARCHITECTURE.md#documentation-authority-and-architecture-updates)
  and [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- Compiler/autograd [Compile lifecycle](../../../../architecture/contracts/compiler-autograd.md#compile-lifecycle):
  cold raw capture, complete preflight and call-local family-constructor reconstruction.
- Extensions/training [Owned tensor Models](../../../../architecture/contracts/extensions-training.md#owned-tensor-models),
  [Generic graph source and values](../../../../architecture/contracts/extensions-training.md#generic-graph-source-and-values)
  and [Canonical graph-model version 1](../../../../architecture/contracts/extensions-training.md#canonical-graph-model-version-1).

Before dependent Java, update root scope routing and the two owning contracts for eight-kind
admission and selective v2; retain Canonical graph-model version 1 unchanged. Update explanations and
[ADR0039](../../../../design/decisions/0039-portable-graph-model-persistence.md#coordinated-authority-cutovers-and-validation),
preserving its completed first-slice record. Stop on ambiguity, unstaged authority, overlapping
ownership, new public API/type/edge or expansion; the ADR and this plan never supply authority.

## Non-goals

No operation math, derivative rules, kernels, runtime/lifecycle or static-signature changes; no
prepared-reuse changes, private numerical policy, all-operations claim, general codec campaign,
new session/IR/registry/bridge, build edge, tuning, RNG/BPTT, durable optimizer resume, five-fork
benchmark or performance campaign. Existing raw seven-type persistence is not seven-type activation
or Training support. Only existing floating activation guards apply; objectives are not persisted.

## Dependencies and integration

- Depends on: Compiler0008, Checkpoint0008, Training0008 and NN0028 Complete; their Engine0021/
  Training0007 prerequisites Complete. The selected model-execution program stays Complete.
- Conflicts with: cold-source admission/reconstruction, Checkpoint graph wire, NN owned-source
  admission, root/two owning contracts and their shared documentation; serialize all such writes.
- Parallel group: None.
- Common base revision: main `7e880fb1044168043e45b10419a570d64c5283ae`.
- Integration order: accepted authority cutover before dependent Java; one atomic Compiler +
  Checkpoint + consumer tests/docs integration, then independent review and coordinator checkpoint.
- Integration validation: the focused consumer and repository commands below, once on final source.
- Shared-document integration owner: main model-execution coordinator, with exclusive serial delegation.

Kepler planning review: APPROVE, zero findings. Main verified Ready/frontier/dependencies/base/
contracts/conflicts on 2026-10-19 and recorded user launch authorization. One serial clean executor
owns authority before Java; main owns gates. Two unrelated dirty files stay protected.

## Files and symbols

Java prefix: `io.github.pho001.synaptik`; paths use module `src/main/java` and matching tests.
Added/moved production types: **None**.

| Owner/path | Exact symbols and focused witnesses |
|---|---|
| `modules/compiler`, `compiler` | SourceGraphReconstruction.preflight/reconstruct, GraphCompilationPort.captureSource/reconstructSource/SourceCapture Javadoc; GraphCapture.captureSource unchanged traversal. SourceGraphCaptureTest; SourceGraphReconstructionTest (shared wrappers, equivalent compile/autograd, late preflight failure). |
| `modules/checkpoint`, `checkpoint.graph` | GraphArtifactCodec.encode/write/decode/preflight; GraphSource.capture/copy/rebind associations; GraphCheckpoint copy/save/load/rebind Javadoc. GraphArtifactCodecTest fixed golden bytes, malformed wire, topology and exact limits; GraphCheckpointTest raw seven types, alias/input order and rebind. |
| `extensions/nn`, `nn.module` | Model.defineTensor/graph/load/loadGraph Javadoc; OwnedModelSource.acquireDefinition/adopt and ModelExecutionState forward/save/compute/train are unchanged consumers. OwnedModelSourceTest admits activations and retains an unsupported-kind negative. Reuse OwnedModelExecutionTest/TrainingTest/LifecycleTest evidence. |
| `testing/integration-tests`, `testing.integration` | OwnedModelWorkflowIntegrationTest: one parameterized MLP regression; GraphCheckpointEngineIntegrationTest: restored forward/backward boundary. Retain WholeModelCheckpointEngineIntegrationTest old-reader coverage. |

Read-only semantics: Tensor.relu/tanh/sigmoid, TensorUnaryExpressions, UnaryElementwiseKind,
ElementwiseGradientRules and CPU activation routes. Existing architecture/API-consumer checks
cover unchanged edges/signatures; no new whitelist/build/test dependency.

## Acceptance criteria

1. All eight exact kinds round-trip capture/reconstruct and copy/save/load/rebind without operation
   loss or decomposition. Assert exact kinds, attrs, descriptors, normalized topology, sharing,
   distinct equal occurrences, ordered multiple roots, declared versus first-encounter input order,
   unused slots and identity aliases. Cover each added kind across existing four floating dtypes
   at source level, without claiming execution support for every occurrence.
2. Preserve static/dtype/attrs/full-descriptor guards and preflight every stored node, including
   late/dead invalid work before producer allocation. Integral/BOOL activations, unrelated/custom
   kinds, invalid attrs/arity and malformed wire fail closed. Test explicit opcode selection (no
   default-to-PERMUTE), both version tuples, v1 rejection of new wires, unknown IDs, truncation,
   trailing data and all six bounds. No partial result/import or target replacement on failure.
3. Preserve v1 golden bytes and load fixtures independent of the new encoder; add explicit v2
   golden opcode/header fixtures. Retain old ArtifactFile/model.v1/whole-model.v1 reader fixtures.
   Exact source state/parameter bytes, including signed zero/NaN payloads in non-training fixtures,
   aliases and all seven raw types survive copy/load/re-save without source mutation or batch storage.
4. One parameterized FLOAT32 two-Linear MLP regression runs RELU/TANH/SIGMOID with deterministic
   finite weights/batches distinguishing each activation from identity/PERMUTE and each other;
   include a positive nonsaturated path with nonzero gradients (avoid RELU's zero boundary).
   Compare owned/copied and restored behavior to an ordinary Tensor/Engine or direct TrainingSession
   oracle that never calls cold copy/reconstruction/codec. Assert actual nonzero parameter updates,
   forward/objective/gradient or update parity, save-before-compute, two fresh same-signature batches,
   restored training, trained re-save/load and unchanged external weights. Use existing focused-test
   floating tolerances, never bitwise arithmetic as new policy; state bytes at persistence boundaries
   and operation/topology/wire facts remain exact. Model-vs-Model agreement alone is insufficient.
5. Both defineTensor and graph construction admit activations; explicit symbolic forward remains
   cold. Retain prepared reuse, one owned state set, callback disposal, lazy/borrowed Engine and
   close/invalidation behavior without changing lifecycle code. Preserve objective-only operations
   outside the saved source and existing Training alias/nonfinite rejection.

## Validation

Validation commands (repository root; worker evidence below):

```bash
./gradlew :modules:compiler:test --tests '*SourceGraph*Test' --max-workers=2 --console=plain
./gradlew :extensions:checkpoint:test --tests '*Graph*Test' --max-workers=2 --console=plain
./gradlew :extensions:nn:test --tests '*OwnedModel*Test' --max-workers=2 --console=plain
./gradlew :modules:compiler:test :extensions:checkpoint:test :extensions:nn:test :modules:compiler:javadoc :extensions:checkpoint:javadoc :extensions:nn:javadoc --max-workers=2 --console=plain
./gradlew :testing:integration-tests:test --tests '*OwnedModelWorkflowIntegrationTest' --tests '*GraphCheckpointEngineIntegrationTest' --tests '*WholeModelCheckpointEngineIntegrationTest' --max-workers=2 --console=plain
SYNAPTIK_METAL_TEST_LIBRARY=/private/tmp/synaptik-checkpoint-leaf.oCW897/modules/backend/metal/macos-arm64/build/libsynaptik_metal_foundation.dylib ./gradlew test javadoc :testing:architecture-tests:test --rerun :testing:integration-tests:test --rerun --no-build-cache --no-configuration-cache --max-workers=2 --console=plain
git diff --check
```

Worker owns module validation; main owns integration/native root gates. Review reuses executable
evidence. Final Javadoc follows edits; metadata checks cover links, size, whitespace and hashes.

## Documentation and review impact

Reviewed root/two contracts, compile/graph API guides, running-models MLP example, lifecycle/overview,
ADR0039 and affected Javadocs with params/results/failures. Two existing glossary entries reflect
the format; no new terminology. Write-page/repo profiles preserve format. Active indexes are
synchronized; predecessor briefs are unchanged and the selected program remains Complete.

## Result

Implemented Compiler activation admission/explicit reconstruction and Checkpoint selective v2,
exact tuple/opcode/attrs/arity checks; no new production types/APIs/kernels/build edges/lifecycle.
Changed Compiler source/Javadocs/reconstruction tests, Checkpoint codec/facade/graph tests,
NN Model Javadoc/source tests and two integration tests; contracts/docs/glossary impact above.
V1 bytes and older formats retained; unchanged GraphSource/NN lifecycle Javadocs remain accurate.
SourceGraph focused command above passed: 16 tests. Initial combined graph/owned-model run had
one new unresolved-layout test-fixture failure, corrected with contiguous(); NN passed.
Worker stable module/Javadoc command above plus `:testing:integration-tests:compileTestJava`
passed (5s): Compiler 309/42 suites, Checkpoint 40/5, NN 402/53; 751 tests, zero failures/errors/skips.
All three Javadocs passed. Main's selected three-integration-class command above passed (2s): 11/0 skips.
Archimedes final Astra/xhigh APPROVE: zero findings; all 27 hashes matched inventory
`85d2476b65cad80d1df6428a3fbb092a369111a48a3fc81049602943503d9f05`.
Independent docs check: 16 files/637 links/371 fragments, zero errors; rendered Javadoc verified.
Main's exact native root command above passed, exit 0, BUILD SUCCESSFUL (5m10s): 107 tasks,
12 executed/95 up-to-date. All 2280 checked files were byte-unchanged through that gate.
Source-matched reports across 19 projects: 4240 tests, zero failures/errors, 41 skips
(CPU 30, Metal 10, integration 1); includes up-to-date results, not 4240 fresh executions.
Full integration: 140 tests/1 skip; architecture: 13/0 skips; both zero failures/errors.
Configured mixed-training and Metal-only unsupported-operation classification tests executed, not skipped.
Metadata checks: diff whitespace/links passed apart from seven known historical link errors;
protected hashes unchanged. No Git publication, native rebuild or benchmark campaign.
Main accepted Complete on 2026-10-19. Unresolved issues: None. Required follow-up: None.
Status: Complete
