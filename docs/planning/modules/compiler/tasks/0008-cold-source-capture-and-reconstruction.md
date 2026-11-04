# Task 0008: Cold source [capture](../../../../glossary.md#term-structural-capture) and reconstruction

> **Editorial reading key — not part of the historical record.**
>
> The cold-source caller asks GraphCompilationPort to capture or reconstruct expressions without
> running them. Capture returns a raw graph plus a borrowed list of exact producer-free leaves; no
> stored values are required. Reconstruction associates supplied Tensors by input position and
> preserves their ordinary upstream provenance. It copies list membership, not tensor payloads,
> and its temporary lookup table owns no storage or cleanup lifecycle. Optimized compile artifacts
> are not interchangeable with this raw source representation.
>
> Current reading: [raw source capture and positional reconstruction](../../../../reference/compiler/cold-source.md#current-cold-source-capture-and-reconstruction); [structural capture and exact leaf identities](../../../../reference/compiler/capture.md#current-package-private-structural-capture).

## Status

Complete — final independent approval and coordinator Class C [checkpoint](../../../../glossary.md#term-checkpoint) passed (2026-10-17).
Authorized [Compiler](../../../../glossary.md#term-compiler) frontier: None; next [planning](../../../../glossary.md#term-planning) candidate Checkpoint0008 remains Draft.

## Change class

Class C — supported [Compiler](../../../../glossary.md#term-compiler) integration [API](../../../../glossary.md#term-application-programming-interface-api) and a narrow exception to the reconstruction-map ban.

## Goal and [scope](../../../../glossary.md#term-scope)

Extend `GraphCompilationPort`: [capture](../../../../glossary.md#term-capture-meanings) a [Tensor](../../../../glossary.md#tensor)-expression [directed acyclic graph](../../../../glossary.md#term-directed-acyclic-graph-dag) ([DAG](../../../../glossary.md#intermediate-representation-dag-and-ssa)) and leaves;
reconstruct against supplied Tensors without execution. [Slice](../../../../glossary.md#slice):
fully static FORWARD ADD/MUL/NEG/[MATMUL](../../../../glossary.md#matrix-multiplication--matmul) and PERMUTE with exactly `[1,0]` (ordinary [Linear](../../../../glossary.md#term-linear-meanings)).
Capture -> validated raw [graph](../../../../glossary.md#graph) + leaves -> [cold reconstruction](../../../../glossary.md#term-cold-reconstruction) -> ordinary [compile](../../../../glossary.md#compile)/[autograd](../../../../glossary.md#autograd) later.
The source graph is not optimized `CompileArtifacts.graph()`.

## Non-goals

No all-family [codec](../../../../glossary.md#term-codec), new [Tensor](../../../../glossary.md#tensor)/core [node](../../../../glossary.md#node), generic constructor/[registry](../../../../glossary.md#term-registry), [optimizer](../../../../glossary.md#term-optimizer)/[Planning](../../../../glossary.md#term-planning)/[backend](../../../../glossary.md#backend)/
numerical change, storage inspection/copy/allocation, [Engine](../../../../glossary.md#term-engine)/[run](../../../../glossary.md#term-run)/recompile/discovery/native/[GPU](../../../../glossary.md#term-gpu) work,
[Checkpoint](../../../../glossary.md#term-checkpoint) encoding/edge or [Model](../../../../glossary.md#term-model)/withEngine. Preserve `compile()`/[autograd](../../../../glossary.md#autograd), [constants](../../../../glossary.md#term-constant) and [artifacts](../../../../glossary.md#term-artifact).
No [commit](../../../../glossary.md#term-commit)/push or edits to protected files below.

## Contracts and coordinated cutover

Read [root scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>,
[Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
[Module-ownership routing](../../../../../ARCHITECTURE.md#module-ownership-routing)<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>, and
[Explicit non-goals](../../../../../ARCHITECTURE.md#explicit-non-goals).
Owning contract: [compiler](../../../../glossary.md#term-compiler)-[autograd](../../../../glossary.md#autograd) [modules/compiler](../../../../architecture/contracts/compiler-autograd.md#modulescompiler),
[Compile artifacts](../../../../architecture/contracts/compiler-autograd.md#compile-artifacts),
[Compile lifecycle](../../../../architecture/contracts/compiler-autograd.md#compile-lifecycle), and
[Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)<span data-glossary-navigation="true"> ([automatic differentiation glossary](../../../../glossary.md#autograd))</span>.
Read [Model](../../../../glossary.md#term-model)'s [modules/model](../../../../architecture/contracts/foundational-modules.md#modulesmodel)
for [immutable](../../../../glossary.md#term-immutable) [graph](../../../../glossary.md#graph) state and [canonical](../../../../glossary.md#term-canonical-form) [producer](../../../../glossary.md#term-producer)-owned output wrappers; no Model contract edit.

Root **Explicit non-goals** and compiler-autograd **[modules](../../../../glossary.md#term-module)/compiler**, **[Compile](../../../../glossary.md#compile) [lifecycle](../../../../glossary.md#lifecycle)**,
**[Compile artifacts](../../../../glossary.md#compile-artifacts)** now enact [ADR0039's bounded decision](../../../../design/decisions/0039-portable-graph-model-persistence.md#coordinated-authority-cutovers-and-validation):
only a call-local reconstruction table after complete cold [preflight](../../../../glossary.md#term-preflight), discarded on return/failure,
and a [borrowed](../../../../glossary.md#ownership-borrowing-and-lease) source sidecar. No [Tensor](../../../../glossary.md#tensor) retention in graph/[artifacts](../../../../glossary.md#term-artifact), autograd conversion map,
fabricated placeholder, [runtime](../../../../glossary.md#term-runtime) use or second [gradient](../../../../glossary.md#term-gradient) algebra is permitted. No [dependency edge](../../../../glossary.md#term-dependency-edge)
changed; Checkpoint0008 still requires its separate Compiler-edge cutover. Plans/ADRs do not
override current authority or authorize further exceptions.

## Dependencies and integration

- Depends on: approved P2/ADR0039; Complete Compiler0001/0002/0006B3/0006B4 capture/inference/
  integration/identity boundaries. Complete Training0007 is a serial base, not an algorithm dependency.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [inference glossary](../../../../glossary.md#term-inference); [identity glossary](../../../../glossary.md#term-identity-meanings); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Conflicts with: shared root/compiler authority, source identity/reconstruction and shared program
  documents; Checkpoint0008/Training0008/NN0028 writes. None active; sole writer, no parallel launch.
- Parallel group: None.
- Common base revision: `3d4d3bf5a95f6ea6e65ebd4687735adc0f092fe0` (`main` = `origin/main`).
- Integration order: Engine0021(done) -> Training0007(done) -> Compiler0008 -> Checkpoint0008 ->
  Training0008 -> NN0028; later rows remain Draft, not authorized by this packet.
- Integration [validation](../../../../glossary.md#term-validation): worker [Compiler](../../../../glossary.md#term-compiler) test/[Javadoc](../../../../glossary.md#term-javadoc) once; coordinator one Class C repository
  `test javadoc` [checkpoint](../../../../glossary.md#term-checkpoint) after review, with focused architecture/consumer checks below.
- Shared-document integration owner: model-execution coordinator; exclusively integrates root,
  owning contract, explanations, ADR, glossary, compiler master, program and roadmap.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [model glossary](../../../../glossary.md#term-model); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>

Launch verified 2026-10-17: approved plan, base/dependencies satisfied; no conflicts. Protected files:
`THIRD_PARTY_NOTICES.md` and
`modules/backend/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuOrdinaryPointwiseCastCleanJavaOracle.java`
(comments). Never stage/edit/revert them. Compiler0008 was the sole [Ready](../../../../glossary.md#term-ready) frontier at launch.

## Files and symbols

[Compiler](../../../../glossary.md#term-compiler) paths use `modules/compiler/src/{main,test}/java/io/github/pho001/synaptik/compiler/`.

- Modify production `GraphCompilationPort.java`: add exactly public static
  `captureSource(List<Tensor> forwardOutputs) -> SourceCapture` and
  `reconstructSource(CompiledGraphModel graph, List<Tensor> leaves) -> List<Tensor>`;
  nested public record `SourceCapture(CompiledGraphModel graph, List<Tensor> leaves)` only.
- Modify `GraphCapture.java`: opt-in `captureSource` entry, collecting leaves at both first-leaf
  insertion sites in <span data-glossary-navigation="true"> ([leaf glossary](../../../../glossary.md#term-leaf))</span>`captureInternal`; preserve old entries/IDs/roles with no [compile](../../../../glossary.md#compile)-path sidecar.
- Add package-private final `SourceGraphReconstruction.java` in `io.github.pho001.synaptik.compiler`:
  shared source admission and constructor dispatch. No other new production type/package/framework.<span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Reuse unchanged `CapturedGraphInference.inferAndValidate`, `ValidatedGraph`, `ElementwiseInference`,
  `StructuredOperationInference`, `LayoutInference`; read `GraphCompiler`/`StaticResultLogicalLayoutClosure`.
- [Model](../../../../glossary.md#term-model) read-only [owners](../../../../glossary.md#term-owner) under `modules/model/src/main/java/io/github/pho001/synaptik/model/`:
  `graph/{CompiledGraphModel,CompiledNode,GraphValue,GraphPhase,NodeId,ValueId}.java`,
  `tensor/{Tensor,TensorDescriptor,TensorProducer,TensorProvenance,TensorFactory}.java`,
  `tensor/Tensor{Binary,Unary,Matmul,Permutation}Expressions.java`, `operation/Operation.java`,
  `operation/layout/PermutationAttrs.java`, and `shape/Shape.java` for checked static geometry.
  Public construction uses only `Tensor.add(Tensor)`, `mul(Tensor)`, `neg()`, `matmul(Tensor)`,
  `permute(1,0)`; [Linear](../../../../glossary.md#term-linear-meanings)'s existing Model expression is PERMUTE -> [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) -> optional ADD.
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
   identity-distinct equal-looking producers/leaves. Existing nonempty/unique-output rules stand.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [descriptor glossary](../../../../glossary.md#term-descriptor); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
2. Raw capture remains structural; the new boundary preflights the static FORWARD slice and
   runs existing inference on that same graph before return.
   Inference verifies but does not rewrite descriptors; no canonicalization, DCE/CSE, folding,
   published-constant or final logical-layout closure. ADD/MUL/NEG/MATMUL results retain unresolved
   layout; PERMUTE retains precisely the existing constructor's resolved-view/unresolved result.
   Reject incompatible final/closed descriptors rather than stripping layout or fabricating Tensors.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [slice glossary](../../../../glossary.md#slice); [inference glossary](../../../../glossary.md#term-inference); [graph glossary](../../../../glossary.md#graph); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [DCE glossary](../../../../glossary.md#term-dead-code-elimination-dce); [CSE glossary](../../../../glossary.md#term-common-subexpression-elimination-cse); [layout glossary](../../../../glossary.md#layout); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [view glossary](../../../../glossary.md#term-view-meanings); [Tensors glossary](../../../../glossary.md#tensor))</span>
3. `SourceCapture` snapshots exact leaves: graph-input count/order, full descriptor equality,
   provenance-free unique identities and source admission hold even for direct construction.
   This borrowed sidecar has no storage ownership, close lifecycle or cache. Reconstruction
   retains no table/sidecar in any output; only ordinary provenance retains the supplied operands.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [admission glossary](../../../../glossary.md#term-admission); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [close lifecycle glossary](../../../../glossary.md#ownership-borrowing-and-lease); [cache glossary](../../../../glossary.md#term-cache))</span>
4. Reconstruction snapshots the supplied list and associates `graph.inputs().get(i)` with exactly
   `leaves.get(i)`, checking count, nulls and full `TensorDescriptor.equals` (dtype, Shape,
   resolved/unresolved layout and requiresGrad). Replacement TensorIds need not match originals.
   Replacements MAY have ordinary provenance: symbolic composition must not detach them, replace
   them with placeholders or demand storage. Their upstream expressions are opaque substitutions,
   not newly admitted source families; subsequent ordinary compile/autograd validates that DAG.
   Repeated supplied references mean explicit caller substitution, not implicit deduplication;
   preserve returned output positions, and leave ordinary compile's duplicate-output rejection intact.
   Same-descriptor swaps are undetectable: list position is the association contract.<span data-glossary-navigation="true"> ([dtype glossary](../../../../glossary.md#data-type--datatype); [Shape glossary](../../../../glossary.md#shape); [opaque glossary](../../../../glossary.md#term-opaque); [compile glossary](../../../../glossary.md#compile); [autograd glossary](../../../../glossary.md#autograd); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [mean glossary](../../../../glossary.md#term-mean))</span>
5. Before allocating any reconstructed producer, validate every stored node (including dead nodes),
   every value's static Shape/checked <span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [node glossary](../../../../glossary.md#node))</span>`knownElementCount()`, layout geometry, phase FORWARD, typed kind,
   attrs, arity, family type/shape/gradient rules and output descriptor. Reuse immutable model
   validation: <span data-glossary-navigation="true"> ([arity glossary](../../../../glossary.md#term-arity); [gradient glossary](../../../../glossary.md#term-gradient); [immutable glossary](../../../../glossary.md#term-immutable); [model glossary](../../../../glossary.md#term-model); [validation glossary](../../../../glossary.md#term-validation))</span>[`Operation`](../../../../glossary.md#operation) validates attrs class, `CompiledNode` [signature](../../../../glossary.md#term-signature) counts, and
   [`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel) topology/producers/boundaries/exact phase coverage (not FORWARD-only).
   No <span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology))</span>`GraphValidator` type exists; do not add one. Reuse inference, reject deferred obligations,
   retain Model-valid scalar/empty shapes,
   and impose no new positivity/rank policy except the exact rank-two PERMUTE restriction.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [rank glossary](../../../../glossary.md#term-rank); [rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
6. Admit only `BinaryArithmeticKind.ADD/MUL`, `UnaryElementwiseKind.NEG`, `MatmulKind.MATMUL`
   with `NoOperationAttrs.INSTANCE`, and `AxisTransformKind.PERMUTE` with
   `PermutationAttrs(List.of(1,0))`. Preserve existing promotion/broadcast/vector/batched MATMUL
   semantics, not a new numerical subset. Unsupported/custom kinds and attrs fail closed.
   Compiler checks logical geometry only; host readability/capacity, canonical persistence payloads,
   leaf names/roles and alias storage admission belong to downstream owners. Never read hostStorage.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector); [fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation); [Compiler glossary](../../../../glossary.md#term-compiler); [host glossary](../../../../glossary.md#term-host); [canonical glossary](../../../../glossary.md#term-canonical-form); [persistence glossary](../../../../glossary.md#term-persistence); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
7. After preflight, rebuild each node once in topological order with its public family constructor
   and canonical output wrapper; never merge occurrences or rebuild per consumer. One call-local
   ValueId-to-Tensor table preserves producer/use identity.
   Return immutable outputs in graph order. Accept existing zero-node pass-through and unused-input
   structure without a fake op or dense-ID/reachability requirement; no Engine executability claim.<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [topological order glossary](../../../../glossary.md#term-topological-order); [ValueId glossary](../../../../glossary.md#logical-valueid-versus-physical-memory-slot); [Engine glossary](../../../../glossary.md#term-engine))</span>
8. Null [arguments](../../../../glossary.md#term-argument)/[elements](../../../../glossary.md#term-element) reject with `NullPointerException`; unsupported/invalid source or
   association rejects with contextual `IllegalArgumentException`, wrapping checked geometry
   overflow. Preserve earlier model-constructor failures for unconstructible malformed fixtures.
   No partially reconstructed result is published on failure, no caller mutation or ID rollback.<span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow); [rollback glossary](../../../../glossary.md#term-rollback))</span>

## Tests and [validation](../../../../glossary.md#term-validation)

Test same-traversal [alignment](../../../../glossary.md#term-alignment), [immutable](../../../../glossary.md#term-immutable) snapshots, shared/repeated inputs, equal distinct
[producers](../../../../glossary.md#term-producer)/leaves, output order, arbitrary IDs, unused inputs and pass-through. Cover all five kinds,
broadcast ADD/MUL, [vector](../../../../glossary.md#term-vector)/batched [MATMUL](../../../../glossary.md#matrix-multiplication--matmul), unresolved results and resolved/unresolved [transpose](../../../../glossary.md#term-transpose).
Negative tests cover null/empty/duplicate outputs, count/[descriptor](../../../../glossary.md#term-descriptor)/[layout](../../../../glossary.md#layout)/requiresGrad mismatch,
symbolic [Shape](../../../../glossary.md#shape), checked [overflow](../../../../glossary.md#term-overflow), BACKWARD phase, unknown kinds, unsupported valid [permutation](../../../../glossary.md#permutation),
bad output descriptor and constructor-rejected attrs/[arity](../../../../glossary.md#term-arity)/[topology](../../../../glossary.md#term-topology) (missing/duplicate producers,
unknown/forward/self/cyclic references, duplicate outputs, missing/extra phases). Reuse existing
[Model](../../../../glossary.md#term-model) coverage; do not bypass constructors to manufacture impossible objects.
Test direct SourceCapture rejects nonleaves/duplicate leaves but reconstruction accepts compatible
expressions and explicit repeated replacements; shared upstream [provenance](../../../../glossary.md#provenance) and later [gradients](../../../../glossary.md#term-gradient)
must survive. Verify storage-free inputs and no [host](../../../../glossary.md#term-host) reads/mutation, including unchanged originals.

[Round-trip](../../../../glossary.md#schema-serialization-and-checkpoint) raw [capture](../../../../glossary.md#term-capture-meanings) with original leaves must preserve [graph](../../../../glossary.md#graph) semantics and [occurrence](../../../../glossary.md#term-occurrence) structure.
Compare ordinary `compile()` of original/reconstructed outputs in FORWARD_ONLY and
FORWARD_AND_BACKWARD using existing test providers, including repeated-operand [scalar](../../../../glossary.md#term-scalar) expressions
and `x.matmul(w.permute(1,0)).add(b)` with a later [scalar objective](../../../../glossary.md#loss). Compare graph, descriptors,
[constant](../../../../glossary.md#term-constant) splats, [derivative](../../../../glossary.md#term-derivative) roles/order and [target](../../../../glossary.md#term-target)/[leaf](../../../../glossary.md#term-leaf) association, normalizing only expected fresh
TensorIds. SUM/gradient-generated families in that later [objective](../../../../glossary.md#term-objective) are not source-[codec](../../../../glossary.md#term-codec) [admission](../../../../glossary.md#term-admission).
Also test fresh leaf replacements, provenance-bearing composition and gradients to upstream targets;
reject optimized layout-closed source input. Existing [compile](../../../../glossary.md#compile) delegation/error behavior stays tested.

Worker validation (completed):

```bash
./gradlew :modules:compiler:test --tests '*SourceGraph*Test' --tests '*GraphCaptureTest' --tests '*GraphCompilationPort*Test'
./gradlew :modules:compiler:test :modules:compiler:javadoc
git diff --check
```

Consumer/architecture/integration evidence and native-library reuse are recorded below.

## Documentation and review impact

Updated root/owning contract, [Lifecycle — Compile lifecycle](../../../../architecture/lifecycle.md#compile-lifecycle)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle); [Compile glossary](../../../../glossary.md#compile))</span>,
[Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>, ADR0039, [Javadoc](../../../../glossary.md#term-javadoc) and glossary; [Tensor](../../../../glossary.md#tensor) [producer](../../../../glossary.md#term-producer) meaning
unchanged. Clean implementation and independent code/docs review complete.

## Result

Implemented production/tests named above and targeted authority/docs/[Javadoc](../../../../glossary.md#term-javadoc)/glossary changes.
Worker commands above PASS: 38 focused; 307 [Compiler](../../../../glossary.md#term-compiler) tests (zero skips/failures), final Javadoc.
Einstein final code/docs APPROVE: all 15 frozen [hashes](../../../../glossary.md#term-hash) verified, no unresolved findings.
Coordinator accepted Complete after this gate (EXIT 0):
```bash
SYNAPTIK_METAL_TEST_LIBRARY=/private/tmp/synaptik-checkpoint-leaf.oCW897/modules/backend/metal/macos-arm64/build/libsynaptik_metal_foundation.dylib ./gradlew test javadoc :testing:architecture-tests:test --rerun :testing:integration-tests:test --rerun --no-build-cache --no-configuration-cache --max-workers=2 --console=plain
```
BUILD SUCCESSFUL in 5m1s; 106 tasks (24 executed, 82 up-to-date). Current 19-project reports:
4166 cases, zero failures/errors, 41 skips; not all reports fresh. Compiler 307 reused worker evidence.
Fresh [CPU](../../../../glossary.md#term-cpu) 1075/30 skips, [Metal](../../../../glossary.md#term-metal) 356/10, architecture 12/0, integration 131/1; both actual Metal
training cases passed. CPU/Metal skips are opt-ins; integration skipped the [host](../../../../glossary.md#term-host) CPU [fixture](../../../../glossary.md#term-fixture)
without eligible local-workload alternatives. No new performance opt-ins or native rebuild.
All 2255 files identical across the gate; protected files unchanged. Final [metadata](../../../../glossary.md#term-metadata)-only
link/whitespace/outside-[scope](../../../../glossary.md#term-scope) [hash](../../../../glossary.md#term-hash) checks passed. No unresolved issues or required follow-up.
Checkpoint0008/Training0008/NN0028 remain Draft; high-level [Model](../../../../glossary.md#term-model)/withEngine remains [TARGET](../../../../glossary.md#term-target).
Status: Complete
