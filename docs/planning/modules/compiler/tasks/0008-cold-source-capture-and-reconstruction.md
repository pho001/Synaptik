# Task 0008: Cold source capture and reconstruction

## Status

Complete — final independent approval and coordinator Class C checkpoint passed (2026-10-17).
Authorized Compiler frontier: None; next planning candidate Checkpoint0008 remains Draft.

## Change class

Class C — supported Compiler integration API and a narrow exception to the reconstruction-map ban.

## Goal and scope

Extend `GraphCompilationPort`: capture a Tensor-expression directed acyclic graph (DAG) and leaves;
reconstruct against supplied Tensors without execution. Slice:
fully static FORWARD ADD/MUL/NEG/MATMUL and PERMUTE with exactly `[1,0]` (ordinary Linear).
Capture -> validated raw graph + leaves -> cold reconstruction -> ordinary compile/autograd later.
The source graph is not optimized `CompileArtifacts.graph()`.

## Non-goals

No all-family codec, new Tensor/core node, generic constructor/registry, optimizer/Planning/backend/
numerical change, storage inspection/copy/allocation, Engine/run/recompile/discovery/native/GPU work,
Checkpoint encoding/edge or Model/withEngine. Preserve `compile()`/autograd, constants and artifacts.
No commit/push or edits to protected files below.

## Contracts and coordinated cutover

Read [root scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts),
[Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
[Module-ownership routing](../../../../../ARCHITECTURE.md#module-ownership-routing), and
[Explicit non-goals](../../../../../ARCHITECTURE.md#explicit-non-goals).
Owning contract: compiler-autograd [modules/compiler](../../../../architecture/contracts/compiler-autograd.md#modulescompiler),
[Compile artifacts](../../../../architecture/contracts/compiler-autograd.md#compile-artifacts),
[Compile lifecycle](../../../../architecture/contracts/compiler-autograd.md#compile-lifecycle), and
[Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation).
Read Model's [modules/model](../../../../architecture/contracts/foundational-modules.md#modulesmodel)
for immutable graph state and canonical producer-owned output wrappers; no Model contract edit.

Root **Explicit non-goals** and compiler-autograd **modules/compiler**, **Compile lifecycle**,
**Compile artifacts** now enact [ADR0039's bounded decision](../../../../design/decisions/0039-portable-graph-model-persistence.md#coordinated-authority-cutovers-and-validation):
only a call-local reconstruction table after complete cold preflight, discarded on return/failure,
and a borrowed source sidecar. No Tensor retention in graph/artifacts, autograd conversion map,
fabricated placeholder, runtime use or second gradient algebra is permitted. No dependency edge
changed; Checkpoint0008 still requires its separate Compiler-edge cutover. Plans/ADRs do not
override current authority or authorize further exceptions.

## Dependencies and integration

- Depends on: approved P2/ADR0039; Complete Compiler0001/0002/0006B3/0006B4 capture/inference/
  integration/identity boundaries. Complete Training0007 is a serial base, not an algorithm dependency.
- Conflicts with: shared root/compiler authority, source identity/reconstruction and shared program
  documents; Checkpoint0008/Training0008/NN0028 writes. None active; sole writer, no parallel launch.
- Parallel group: None.
- Common base revision: `3d4d3bf5a95f6ea6e65ebd4687735adc0f092fe0` (`main` = `origin/main`).
- Integration order: Engine0021(done) -> Training0007(done) -> Compiler0008 -> Checkpoint0008 ->
  Training0008 -> NN0028; later rows remain Draft, not authorized by this packet.
- Integration validation: worker Compiler test/Javadoc once; coordinator one Class C repository
  `test javadoc` checkpoint after review, with focused architecture/consumer checks below.
- Shared-document integration owner: model-execution coordinator; exclusively integrates root,
  owning contract, explanations, ADR, glossary, compiler master, program and roadmap.

Launch verified 2026-10-17: approved plan, base/dependencies satisfied; no conflicts. Protected files:
`THIRD_PARTY_NOTICES.md` and
`modules/backend/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuOrdinaryPointwiseCastCleanJavaOracle.java`
(comments). Never stage/edit/revert them. Compiler0008 was the sole Ready frontier at launch.

## Files and symbols

Compiler paths use `modules/compiler/src/{main,test}/java/io/github/pho001/synaptik/compiler/`.

- Modify production `GraphCompilationPort.java`: add exactly public static
  `captureSource(List<Tensor> forwardOutputs) -> SourceCapture` and
  `reconstructSource(CompiledGraphModel graph, List<Tensor> leaves) -> List<Tensor>`;
  nested public record `SourceCapture(CompiledGraphModel graph, List<Tensor> leaves)` only.
- Modify `GraphCapture.java`: opt-in `captureSource` entry, collecting leaves at both first-leaf
  insertion sites in `captureInternal`; preserve old entries/IDs/roles with no compile-path sidecar.
- Add package-private final `SourceGraphReconstruction.java` in `io.github.pho001.synaptik.compiler`:
  shared source admission and constructor dispatch. No other new production type/package/framework.
- Reuse unchanged `CapturedGraphInference.inferAndValidate`, `ValidatedGraph`, `ElementwiseInference`,
  `StructuredOperationInference`, `LayoutInference`; read `GraphCompiler`/`StaticResultLogicalLayoutClosure`.
- Model read-only owners under `modules/model/src/main/java/io/github/pho001/synaptik/model/`:
  `graph/{CompiledGraphModel,CompiledNode,GraphValue,GraphPhase,NodeId,ValueId}.java`,
  `tensor/{Tensor,TensorDescriptor,TensorProducer,TensorProvenance,TensorFactory}.java`,
  `tensor/Tensor{Binary,Unary,Matmul,Permutation}Expressions.java`, `operation/Operation.java`,
  `operation/layout/PermutationAttrs.java`, and `shape/Shape.java` for checked static geometry.
  Public construction uses only `Tensor.add(Tensor)`, `mul(Tensor)`, `neg()`, `matmul(Tensor)`,
  `permute(1,0)`; Linear's existing Model expression is PERMUTE -> MATMUL -> optional ADD.
- Add tests `SourceGraphCaptureTest.java`, `SourceGraphReconstructionTest.java`; update
  `GraphCaptureTest.java`, `GraphCompilationPortTest.java`, and
  `spi/GraphCompilationPortPublicShapeTest.java` (both port tests currently assert one public method).
  Read existing `CapturedGraphInferenceTest`, `StaticResultLogicalLayoutClosureTest`,
  `GraphCompilerTest`, `FirstOrderAutogradTest`, `GradientRulesTest`, `FunctionalGradientRequestTest`
  and Model `graph/CompiledGraphModelTest` as focused oracles.

## Acceptance criteria

1. `captureSource` uses one existing traversal with empty constant ingress. Collect exact leaves
   when input IDs are assigned, never by recapture, sorting, descriptor equality or storage identity.
   Preserve first-encounter input order, requested output order, repeated operands, fanout and
   identity-distinct equal-looking producers/leaves. Existing nonempty/unique-output rules stand.
2. Raw capture remains structural; the new boundary preflights the static FORWARD slice and
   runs existing inference on that same graph before return.
   Inference verifies but does not rewrite descriptors; no canonicalization, DCE/CSE, folding,
   published-constant or final logical-layout closure. ADD/MUL/NEG/MATMUL results retain unresolved
   layout; PERMUTE retains precisely the existing constructor's resolved-view/unresolved result.
   Reject incompatible final/closed descriptors rather than stripping layout or fabricating Tensors.
3. `SourceCapture` snapshots exact leaves: graph-input count/order, full descriptor equality,
   provenance-free unique identities and source admission hold even for direct construction.
   This borrowed sidecar has no storage ownership, close lifecycle or cache. Reconstruction
   retains no table/sidecar in any output; only ordinary provenance retains the supplied operands.
4. Reconstruction snapshots the supplied list and associates `graph.inputs().get(i)` with exactly
   `leaves.get(i)`, checking count, nulls and full `TensorDescriptor.equals` (dtype, Shape,
   resolved/unresolved layout and requiresGrad). Replacement TensorIds need not match originals.
   Replacements MAY have ordinary provenance: symbolic composition must not detach them, replace
   them with placeholders or demand storage. Their upstream expressions are opaque substitutions,
   not newly admitted source families; subsequent ordinary compile/autograd validates that DAG.
   Repeated supplied references mean explicit caller substitution, not implicit deduplication;
   preserve returned output positions, and leave ordinary compile's duplicate-output rejection intact.
   Same-descriptor swaps are undetectable: list position is the association contract.
5. Before allocating any reconstructed producer, validate every stored node (including dead nodes),
   every value's static Shape/checked `knownElementCount()`, layout geometry, phase FORWARD, typed kind,
   attrs, arity, family type/shape/gradient rules and output descriptor. Reuse immutable model
   validation: `Operation` validates attrs class, `CompiledNode` signature counts, and
   `CompiledGraphModel` topology/producers/boundaries/exact phase coverage (not FORWARD-only).
   No `GraphValidator` type exists; do not add one. Reuse inference, reject deferred obligations,
   retain Model-valid scalar/empty shapes,
   and impose no new positivity/rank policy except the exact rank-two PERMUTE restriction.
6. Admit only `BinaryArithmeticKind.ADD/MUL`, `UnaryElementwiseKind.NEG`, `MatmulKind.MATMUL`
   with `NoOperationAttrs.INSTANCE`, and `AxisTransformKind.PERMUTE` with
   `PermutationAttrs(List.of(1,0))`. Preserve existing promotion/broadcast/vector/batched MATMUL
   semantics, not a new numerical subset. Unsupported/custom kinds and attrs fail closed.
   Compiler checks logical geometry only; host readability/capacity, canonical persistence payloads,
   leaf names/roles and alias storage admission belong to downstream owners. Never read hostStorage.
7. After preflight, rebuild each node once in topological order with its public family constructor
   and canonical output wrapper; never merge occurrences or rebuild per consumer. One call-local
   ValueId-to-Tensor table preserves producer/use identity.
   Return immutable outputs in graph order. Accept existing zero-node pass-through and unused-input
   structure without a fake op or dense-ID/reachability requirement; no Engine executability claim.
8. Null arguments/elements reject with `NullPointerException`; unsupported/invalid source or
   association rejects with contextual `IllegalArgumentException`, wrapping checked geometry
   overflow. Preserve earlier model-constructor failures for unconstructible malformed fixtures.
   No partially reconstructed result is published on failure, no caller mutation or ID rollback.

## Tests and validation

Test same-traversal alignment, immutable snapshots, shared/repeated inputs, equal distinct
producers/leaves, output order, arbitrary IDs, unused inputs and pass-through. Cover all five kinds,
broadcast ADD/MUL, vector/batched MATMUL, unresolved results and resolved/unresolved transpose.
Negative tests cover null/empty/duplicate outputs, count/descriptor/layout/requiresGrad mismatch,
symbolic Shape, checked overflow, BACKWARD phase, unknown kinds, unsupported valid permutation,
bad output descriptor and constructor-rejected attrs/arity/topology (missing/duplicate producers,
unknown/forward/self/cyclic references, duplicate outputs, missing/extra phases). Reuse existing
Model coverage; do not bypass constructors to manufacture impossible objects.
Test direct SourceCapture rejects nonleaves/duplicate leaves but reconstruction accepts compatible
expressions and explicit repeated replacements; shared upstream provenance and later gradients
must survive. Verify storage-free inputs and no host reads/mutation, including unchanged originals.

Round-trip raw capture with original leaves must preserve graph semantics and occurrence structure.
Compare ordinary `compile()` of original/reconstructed outputs in FORWARD_ONLY and
FORWARD_AND_BACKWARD using existing test providers, including repeated-operand scalar expressions
and `x.matmul(w.permute(1,0)).add(b)` with a later scalar objective. Compare graph, descriptors,
constant splats, derivative roles/order and target/leaf association, normalizing only expected fresh
TensorIds. SUM/gradient-generated families in that later objective are not source-codec admission.
Also test fresh leaf replacements, provenance-bearing composition and gradients to upstream targets;
reject optimized layout-closed source input. Existing compile delegation/error behavior stays tested.

Worker validation (completed):

```bash
./gradlew :modules:compiler:test --tests '*SourceGraph*Test' --tests '*GraphCaptureTest' --tests '*GraphCompilationPort*Test'
./gradlew :modules:compiler:test :modules:compiler:javadoc
git diff --check
```

Consumer/architecture/integration evidence and native-library reuse are recorded below.

## Documentation and review impact

Updated root/owning contract, [Lifecycle — Compile lifecycle](../../../../architecture/lifecycle.md#compile-lifecycle),
[Compile API](../../../../api/compile-api.md), ADR0039, Javadoc and glossary; Tensor producer meaning
unchanged. Clean implementation and independent code/docs review complete.

## Result

Implemented production/tests named above and targeted authority/docs/Javadoc/glossary changes.
Worker commands above PASS: 38 focused; 307 Compiler tests (zero skips/failures), final Javadoc.
Einstein final code/docs APPROVE: all 15 frozen hashes verified, no unresolved findings.
Coordinator accepted Complete after this gate (EXIT 0):
```bash
SYNAPTIK_METAL_TEST_LIBRARY=/private/tmp/synaptik-checkpoint-leaf.oCW897/modules/backend/metal/macos-arm64/build/libsynaptik_metal_foundation.dylib ./gradlew test javadoc :testing:architecture-tests:test --rerun :testing:integration-tests:test --rerun --no-build-cache --no-configuration-cache --max-workers=2 --console=plain
```
BUILD SUCCESSFUL in 5m1s; 106 tasks (24 executed, 82 up-to-date). Current 19-project reports:
4166 cases, zero failures/errors, 41 skips; not all reports fresh. Compiler 307 reused worker evidence.
Fresh CPU 1075/30 skips, Metal 356/10, architecture 12/0, integration 131/1; both actual Metal
training cases passed. CPU/Metal skips are opt-ins; integration skipped the host CPU fixture
without eligible local-workload alternatives. No new performance opt-ins or native rebuild.
All 2255 files identical across the gate; protected files unchanged. Final metadata-only
link/whitespace/outside-scope hash checks passed. No unresolved issues or required follow-up.
Checkpoint0008/Training0008/NN0028 remain Draft; high-level Model/withEngine remains TARGET.
Status: Complete
