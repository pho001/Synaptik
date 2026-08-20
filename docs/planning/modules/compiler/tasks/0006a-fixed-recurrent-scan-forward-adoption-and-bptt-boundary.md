# Task 0006A: Fixed Recurrent-Scan Forward Adoption and Explicit BPTT Boundary

## Status

Complete

## Goal

Adopt the current Model fixed recurrent-scan family in Compiler's forward-only graph pipeline
without introducing graph regions, recurrence execution, or backpropagation through time (BPTT).

For each identity-distinct `RNN_TANH`, `GRU_RESET_AFTER`, or `LSTM` producer constructed through
the low-level `model.tensor.RecurrentScan` namespace, Compiler must preserve one ordinary flat
multi-output `CompiledNode`, independently infer and validate its complete static descriptor
contract, retain its kind and `FORWARD` or `REVERSE` direction, and hand that same ordinary node
to the existing publication and Planning orchestration.

The selected compiled representation is the existing Model-owned graph representation:

```text
one RecurrentScan TensorProducer occurrence
  -> one ordinary CompiledNode
     operation = exact RecurrentScanKind + exact RecurrentDirection
     inputs    = exact ordered 5/6 or 6/7 ValueIds
     outputs   = exact ordered 2 or 3 ValueIds
  -> existing OperationCapabilityQuery when ordinary Planning orchestration is invoked
```

There is no recurrent-specific compiler intermediate representation, body, nested graph, region,
loop node, callable graph, or per-time-step expansion. Graph construction and compiled topology
for one occurrence remain constant in `time`.

Backward-capable compilation remains intentionally unavailable for every forward graph containing
a recurrent occurrence. Compiler must reject that complete forward inventory during allocation-
free autograd preflight, before constructing a seed, derivative constant, formula Tensor, or
partial gradient graph. The closed first-order gradient inventory therefore continues to exclude
the exact three recurrent signatures even though forward inference adopts them.

## Motivation

Model tasks 0025E and 0025F now supply the stable semantic family, exact signatures, canonical
multi-output wrappers, and corrected stateless `RecurrentScan.rnn/gru/lstm` construction seam.
Before this task, generic Compiler capture already saw those expressions as one flat node, but
`CapturedGraphInference` rejected `RecurrentScanKind` before optimization, publication, or
Planning. This task closes only that forward Compiler gap.

The distinction between forward and gradient inventories is essential:

| Inventory or stage | Result after 0006A |
|---|---|
| Model operation signatures | The three recurrent signatures remain current structural Model declarations. |
| Compiler capture | Each exact producer identity becomes one flat node with every ordered output slot. |
| Compiler forward inference/final validation | The three recurrent signatures are supported and independently revalidated. |
| Compiler first-order gradient coverage | The existing 128 supported signatures remain unchanged; the three recurrent signatures remain the exact disjoint deferred set. |
| Backward-capable compile preflight | Any recurrent occurrence in the complete original forward inventory fails before derivative Tensor allocation. |
| Planning/backend capability | The unchanged ordinary query carries the node; no provider gains support in this task. |

This is the smallest coherent forward adoption because it uses the existing flat graph and
lifecycle boundaries while preserving the explicitly deferred BPTT decision.

## Scope

- Add one package-private, final, stateless `RecurrentScanInference` helper in
  `io.github.pho001.synaptik.compiler`.
- Route every `RecurrentScanKind` from `CapturedGraphInference` to that helper.
- Independently derive and validate all three kinds, both directions, and biased or bias-free
  cardinalities from ordered graph descriptors.
- Preserve the existing ordinary `CompiledNode` as the sole compiled representation.
- Preserve exact producer-identity capture behavior:
  - one producer reached through multiple sibling wrappers is captured once;
  - structurally equal but identity-distinct recurrent producers remain distinct nodes;
  - repeated reachability never duplicates a producer;
  - all two or three output slots remain present on a live recurrent node.
- Exclude recurrent occurrences from common-subexpression elimination (CSE), so equivalent
  recurrent expressions cannot erase their Model occurrence identity. Other current CSE
  eligibility and exact merging behavior remain unchanged.
- Preserve whole-node dead-code elimination (DCE): an entirely unreachable recurrent occurrence
  may be removed, but if any output is live, the node and every declared sibling output slot stay
  structurally present.
- Preserve caller order for multi-root graph outputs, including roots from different slots of one
  occurrence and roots from multiple identity-distinct occurrences.
- Admit the family only through `CompileMode.FORWARD_ONLY` in the current Compiler graph pipeline.
- Add a deterministic allocation-free autograd-preflight guard over the complete original forward
  producer inventory for `FORWARD_AND_BACKWARD` and `TRAINING_STEP`.
- Preserve the exact three recurrent signature fingerprints as deferred from
  `FirstOrderGradientCoverage`; do not add a recurrent formula owner or classify valid lengths as
  differentiable.
- Prove that successful forward graph compilation, canonicalization, optional optimization,
  repeated final inference/validation, publication binding, and the existing ordinary Planning
  query preserve recurrent node semantics and ordered descriptors.
- Update the Compile API and fixed-recurrent glossary entry to mark only Compiler forward
  adoption current and keep execution, capability, and BPTT claims planned.
- Complete a separate documentation-focused review in the same overall change.

## Forward inference and validation contract

### Accepted operation and cardinalities

The helper accepts only an `Operation` whose kind is one of:

```text
RNN_TANH
GRU_RESET_AFTER
LSTM
```

Its exact attributes object is the current `RecurrentDirection.FORWARD` or
`RecurrentDirection.REVERSE`. `Operation` and `CompiledNode` already enforce the family-owned
structural signature, but recurrent inference still dispatches by the closed typed family and
must not accept a custom kind or alternate attributes type.

Accepted ordered inputs are exactly:

```text
RNN_TANH / GRU_RESET_AFTER, bias-free:
  [input, validLengths, initialHidden, inputWeight, hiddenWeight]

RNN_TANH / GRU_RESET_AFTER, biased:
  [input, validLengths, initialHidden, inputWeight, hiddenWeight, bias]

LSTM, bias-free:
  [input, validLengths, initialHidden, initialCell, inputWeight, hiddenWeight]

LSTM, biased:
  [input, validLengths, initialHidden, initialCell, inputWeight, hiddenWeight, bias]
```

Accepted and independently derived output order is exactly:

```text
RNN_TANH / GRU_RESET_AFTER:
  [outputs, finalHidden]

LSTM:
  [outputs, finalHidden, finalCell]
```

### Descriptor derivation

Inference revalidates the current Model descriptor contract in semantic role order without
reading Tensor storage or runtime valid-length values:

1. `input` has one floating data type and a fully static rank-three Shape
   `[time, batch, inputSize]`; `inputSize` is positive while `time` and `batch` may be zero.
2. `validLengths` is fully static rank-one `INT64[batch]` and has
   `requiresGrad == false`.
3. `initialHidden` has the exact common floating type and fully static Shape
   `[batch, hiddenSize]`; `hiddenSize` is positive.
4. For LSTM, `initialCell` has the exact common type and Shape
   `[batch, hiddenSize]`.
5. `gateCount * hiddenSize` is computed with checked arithmetic, where gate count is one, three,
   or four for RNN, GRU, or LSTM.
6. `inputWeight` has the common type and Shape
   `[gateCount * hiddenSize, inputSize]`.
7. `hiddenWeight` has the common type and Shape
   `[gateCount * hiddenSize, hiddenSize]`.
8. When present, `bias` has the common type and Shape `[gateCount * hiddenSize]`.
9. `requiresGrad` is the OR of `input`, initial state, weight, and optional bias roles. The
   valid-length role never contributes.
10. Derived outputs have unresolved layout and exact descriptors:
    - dense `outputs`: common type, `[time, batch, hiddenSize]`, derived gradient eligibility;
    - `finalHidden`: common type, `[batch, hiddenSize]`, derived gradient eligibility;
    - LSTM `finalCell`: common type, `[batch, hiddenSize]`, derived gradient eligibility.

Every Shape is fully static, so this family produces no deferred graph constraint. Direction
changes transition traversal meaning but does not change descriptors.

`CapturedGraphInference` retains its existing deterministic outer behavior: it derives the
complete descriptor list, compares each ordered result with the stored `GraphValue` descriptor,
and reports node position, `NodeId`, kind, output position, `ValueId`, expected descriptor, and
stored descriptor on mismatch. The same pass runs after canonicalization and after every changed
optional optimization candidate, providing final validation without a recurrent-only validation
phase.

### Validation ownership and exclusions

Model construction continues to perform its current local descriptor validation before creating
the producer. Compiler independently checks the captured occurrence because graph inference may
also receive manually assembled immutable graph values and every transformed candidate must be
validated from graph state. Compiler adds no check for facts that require payload values or a
later lifecycle owner.

In particular, Compiler does not:

- read, snapshot, bind, or validate the scalar contents of `validLengths`;
- require host storage or infer lengths from input values, padding, labels, or Tensor identity;
- choose traversal loops, padding writes, gates, accumulators, algorithms, saved state, or
  numerical lowering;
- add dynamic or binding-dependent Shape support; or
- duplicate Model expression-construction checks in another public API.

Runtime length bounds `[0, time]`, complete pre-write validation, positive-zero padding, and
skipped recurrent arithmetic remain concrete-backend execution responsibilities.

## Capture, optimization, and downstream handoff

### Identity and topology

`GraphCapture` remains unchanged. Its existing identity maps already provide the required
representation:

- exact `TensorProducer` reference identity controls capture deduplication;
- one producer allocates one `NodeId` and all output `ValueId` slots in declared order;
- exact `TensorProvenance.outputIndex()` maps canonical wrappers to those slots;
- structurally equal distinct producers remain distinct at capture; and
- topology contains no step, gate, mask, padding, state-update, or body node.

The focused task tests must lock this behavior with `RecurrentScan`-constructed producers rather
than a synthetic custom operation.

### CSE and DCE

Current generic CSE would otherwise consider two equal internal recurrent nodes mergeable by
operation value, remapped inputs, descriptors, phase, and derivative order. That conflicts with
the selected identity-distinct recurrent occurrence contract. CSE eligibility must therefore
exclude every `RecurrentScanKind` occurrence before creating an expression key or representative.
This is a narrow family-specific exclusion; no other CSE key, ordering, output-producer exclusion,
phase rule, or derivative-order rule changes.

DCE remains whole-node structural liveness. When any recurrent output slot is a graph root or
feeds live work, the node remains with every two or three output `ValueId` positions. An unused
sibling is not removed from the node and its slots are not renumbered. If no output is live, DCE
may remove the whole occurrence under its current rules.

### Publication and Planning handoff

Compiler introduces no new handoff record. After final validation, existing publication bindings
refer to the final recurrent output `ValueId` values exactly as for every other graph root.
Existing graph-wide Planning orchestration creates one ordinary `OperationCapabilityQuery` per
final node from:

- the exact retained `Operation` reference, including recurrent kind and direction;
- the exact ordered input descriptor list; and
- the exact ordered two- or three-output descriptor list.

Planning remains responsible only for asking providers and selecting ownership. This task changes
no capability provider and advertises no recurrent support. A Compiler unit test may use an
explicit recording provider that accepts the query solely to prove unchanged handoff and
publication/partition structure; that test provider is not a production capability claim. With
the current real providers unchanged, complete lifecycle availability remains false until a
later backend task truthfully advertises and implements the family.

## Autograd and effect order

### Forward-only path

`GraphCompiler` retains its current order for `FORWARD_ONLY`:

```text
top-level request validation
  -> one identity-based capture
  -> recurrent-aware inference and validation
  -> mandatory canonicalization and validation
  -> optional exact rewrite/fold/DCE/CSE/DCE, validating each changed candidate
  -> forward publication roles
  -> existing ordinary Planning query when complete artifacts are requested
```

No stage reads runtime values or allocates physical state.

### Backward-capable path

`AutogradPreflight` must inspect the already allocation-free complete original forward producer
inventory immediately after that inventory is built. For both `FORWARD_AND_BACKWARD` and
`TRAINING_STEP`, the first recurrent occurrence in deterministic producer postorder fails with
occurrence context and an explanation that fixed recurrent scan is forward-only until BPTT is
implemented.

The guard applies even when a recurrent output is a separate forward root and the requested
gradient targets lie on another branch. This keeps the architecture statement exact:
`FORWARD_ONLY` is the sole current compile mode that adopts recurrent scan. Request/null/
canonical-wrapper validation that necessarily precedes construction of the complete inventory
retains its existing order; after the inventory exists, recurrent rejection precedes seed checks,
stage selection, occurrence formula validation, and all derivative Tensor construction.

Failure must consume no new Tensor ID and must construct no default seed, explicit-seed
normalization, typed splat, `ARGSORT`, local formula, gradient accumulator, combined capture, or
partial backward graph.

### Closed inventory boundary

Do not add `RecurrentScanKind` rows to `FirstOrderGradientCoverage.SIGNATURES`, do not add a
recurrent `FamilyOwner`, and do not change the exact 128 supported-row count. The coordinator-
authorized `FirstOrderGradientCoverageTest` boundary remains authoritative and unchanged:

- the exact recurrent RNN, GRU, and LSTM fingerprints are disjoint from the 128 supported
  signatures;
- their 131-row union equals the complete 38-family, 110-kind Model signature inventory; and
- every recurrent output/input boundary classification remains `FC` with exact reason
  `unknown or unclassified operation kind/attributes pairing`, without Tensor-ID allocation.

This direct classifier boundary and the new complete-forward-inventory preflight guard test
different contracts. Forward inference support must not be mistaken for gradient coverage.
Valid lengths remain non-differentiable, but marking only that role `ND` would be insufficient:
all BPTT formulas and saved-state policy remain absent.

## Out of scope

- any change to `RecurrentScan`, `Tensor`, `TensorProducer`, recurrent result records,
  `RecurrentScanKind`, `RecurrentDirection`, Model descriptors, signatures, semantics, tests,
  Javadocs, or Tensor IDs
- any `Tensor` recurrent receiver method, generic `scan(body)`, callback, lambda, cell body,
  nested `CompiledGraphModel`, graph region, block, loop IR, free-variable capture, or region
  ownership rule
- any change to current `RnnSequence`, `GruSequence`, `LstmSequence`, bidirectional containers,
  cells, parameters, static Java `long[]` length snapshots, compact output lists, static unrolling,
  or NN tests/documentation
- BPTT, derivative formulas, recurrent gradient family owner, gradient signature adoption,
  saved gates, saved-state output slots, tape, checkpointing, recomputation, backward operations,
  second-order recurrent gradients, or Training behavior
- value reads, valid-length bounds checks, runtime input binding, dynamic Shapes, arbitrary masks,
  holes, active-row compaction, packed batches, sorting, padded-output materialization, or skipped-
  arithmetic implementation
- changes to Planning, Prepare, Runtime, Engine, backend-contract, any concrete backend, backend
  capability matrices, providers, routes, lowering, kernels, executables, memory declarations,
  schedules, tracing, serialization, or state dictionaries
- capability advertisement or a claim that any current backend can plan, prepare, or execute a
  recurrent node
- changes to dependencies, Gradle, architecture contracts or explanations, ADR 0012, architecture
  tests, backend conformance tests, integration tests, Tensor API, Training API, Model
  capabilities, or the global roadmap
- redirecting or decomposing current static NN graphs through the new operation
- Compiler task 0007 identities or any relaxed numerical rewrite
- a detailed specification for BPTT, a backend implementation, NN runtime-length integration, or
  any later task

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md), especially fixed recurrent scan,
  Compiler ownership, flat graph identity, and lifecycle invariants
- [ADR 0012: Fixed recurrent scan without graph regions](../../../../design/decisions/0012-fixed-recurrent-scan-without-regions.md)
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Training graph](../../../../architecture/training-graph.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Runtime, Prepare, and backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [General documentation style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)
- [Planning documentation style](../../../../developer-guide/documentation/planning-style.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Compiler master plan](../master-plan.md)
- [Compiler 0001: Tensor expression graph capture](0001-tensor-expression-graph-capture.md)
- [Compiler 0002: Captured-graph inference and validation](0002-captured-graph-inference-and-validation.md)
- [Compiler 0003: Canonicalization and forward optimization](0003-canonicalization-and-forward-optimization.md)
- [Compiler 0004: Pre-capture autograd and combined graph compilation](0004-compiler-owned-pre-capture-autograd-and-combined-graph-compilation.md)
- [Compiler 0005: Publication, Planning orchestration, and artifacts](0005-publication-planning-orchestration-and-compile-artifacts.md)
- [Compiler 0005E: First-order coverage closure](0005e-first-order-gradient-coverage-closure-checkpoint.md)
- [Compiler 0006: Explicit functional gradients](0006-explicit-functional-gradient-requests-and-higher-order-differentiation.md)
- [Model 0025: Canonical producer outputs](../../model/tasks/0025-canonical-tensor-producer-outputs.md)
- [Model 0025E: Recurrent semantic family](../../model/tasks/0025e-fixed-recurrent-scan-semantic-family-and-tensor-expressions.md)
- [Model 0025F: Recurrent namespace correction](../../model/tasks/0025f-recurrent-scan-expression-namespace-correction.md)
- [NN 0021A: Fixed recurrent-scan architecture decision](../../../extensions/nn/tasks/0021a-fixed-recurrent-scan-architecture-decision.md)
- [Compile API](../../../../api/compile-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- One fixed recurrent producer remains one ordinary flat compiled node. Compiler must not create
  a recurrent-specific graph representation or unroll by `time`.
- The current flat meanings of `Tensor`, `TensorProducer`, `Operation`, `CompiledNode`, and
  `CompiledGraphModel` remain unchanged.
- Model owns fixed transition semantics and descriptor-visible expression construction. Compiler
  owns independent captured inference, final validation, graph transformations, and fail-closed
  autograd.
- Recurrent direction and exact ordered inputs/outputs remain immutable Model operation facts.
- Identity-distinct recurrent occurrences remain distinct through the final optimized graph.
- Compiler may remove an entirely dead occurrence but must preserve the complete structural output
  signature of every live multi-output occurrence.
- `FORWARD_ONLY` is the only current mode that may contain a recurrent occurrence. Backward-
  capable modes reject it during allocation-free preflight.
- Compiler output remains immutable compile-time state. Compiler allocates no physical buffer,
  prepared recipe, executable, runtime state, or backend-specific loop plan.
- Planning receives only the existing ordinary operation query and does not interpret recurrence.
- No current capability provider or backend support claim changes.
- No dependency or module-boundary change is authorized. Compiler continues to avoid Runtime,
  Prepare, Engine, NN, Training, Data, and concrete-backend dependencies.
- `ARCHITECTURE.md` remains unchanged. Stop if implementation requires an architecture change.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.compiler` — owns package-private graph inference, validation,
  optimization, autograd preflight, compile orchestration, and focused same-package tests.
- `io.github.pho001.synaptik.model.operation.recurrent` — supplies the current immutable kind and
  direction semantics; read but not changed.
- `io.github.pho001.synaptik.model.tensor` — supplies `RecurrentScan`, canonical Tensor producer
  outputs, and descriptors for boundary tests; only the `RecurrentScan` type Javadoc changes,
  while its executable body and public surface remain unchanged.
- `io.github.pho001.synaptik.model.graph` — supplies the existing immutable ordinary compiled graph
  representation; read but not changed.
- `io.github.pho001.synaptik.planning.capability` — receives the existing ordinary operation query
  through current Compiler orchestration; read but not changed.

Packages added or changed:

- No package is added. The existing Compiler root package gains one cohesive family-owned
  inference helper and retains all implementation types package-private.

Type placement:

- `io.github.pho001.synaptik.compiler.RecurrentScanInference` — package-private stateless
  descriptor derivation for the closed recurrent family; it belongs beside the other
  family-specific inference helpers because Compiler owns graph validation.
- `io.github.pho001.synaptik.compiler.RecurrentScanInferenceTest` — same-package focused valid and
  invalid descriptor tests for the helper and outer final-validation context.
- `io.github.pho001.synaptik.compiler.RecurrentScanCompilerTest` — same-package vertical boundary
  tests for capture identity, CSE/DCE, forward compilation, Planning query handoff, and allocation-
  free backward rejection without widening production visibility.

`CapturedGraphInference`, `ForwardCommonSubexpressionElimination`, and `AutogradPreflight` remain
their existing package-private owners. No public Compiler type or package is added.

## Affected files

Expected production source:

- update
  `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/CapturedGraphInference.java`
- add
  `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/RecurrentScanInference.java`
- update
  `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ForwardCommonSubexpressionElimination.java`
- update
  `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/AutogradPreflight.java`

Expected tests:

- add
  `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/RecurrentScanInferenceTest.java`
- add
  `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/RecurrentScanCompilerTest.java`

Expected documentation and planning:

- update [Compile API](../../../../api/compile-api.md)
- update [Glossary](../../../../glossary.md), limited to the existing fixed-recurrent-scan entry
- update the `RecurrentScan` type Javadoc only; its executable body and public surface remain
  unchanged
- update [Tensor API](../../../../api/tensor-api.md), [Training API](../../../../api/training-api.md),
  and [Model capabilities](../../model/capabilities.md), limited to stale Compiler-adoption status
- update [Module boundaries](../../../../architecture/module-boundaries.md) and
  [Training graph](../../../../architecture/training-graph.md), and
  [Lifecycle](../../../../architecture/lifecycle.md), limited to stale implementation-status
  wording; no architecture rule changes
- update and finalize this task specification
- update [Compiler master plan](../master-plan.md)
- update [Model master plan](../../model/master-plan.md), limited to stale Compiler 0006A status

Review without modification unless a conflict requires stopping:

- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/GraphCapture.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/GraphCompiler.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/FirstOrderGradientCoverage.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/FirstOrderGradientCoverageTest.java`
- other Model recurrent source, tests, and Javadocs; NN source/tests; other architecture/ADR
  documents and tests; Planning/Prepare/Runtime/Engine/backend source/tests and plans;
  dependencies; Gradle; conformance/integration tests; and the global roadmap

## Maximum scope

This task may create or modify at most the exact eighteen paths listed under
[Affected files](#affected-files): four Compiler production paths, two Compiler test paths, one
Model Javadoc path, and eleven documentation/planning paths.

The original ten-path boundary omitted directly affected current-status statements. During the
independent documentation pass, the coordinator first authorized six documentation corrections,
then authorized the separately discovered focused lifecycle correction and the final stale Model
master-plan status correction. The resulting eighteen-path boundary remains one cohesive
Compiler capability:
inference dispatch and family logic, identity-preserving optimization, allocation-free backward
gating, focused vertical tests, and the documentation/planning surfaces that state current
Compiler adoption. If implementation requires a nineteenth path, executable Model/NN/Planning/
Prepare/Runtime/Engine/backend work, or another public type, stop and report the concrete gap
instead of expanding scope.

## Acceptance criteria

- `CapturedGraphInference` accepts every current `RecurrentScanKind`, both directions, and every
  bias-free/biased cardinality through one dedicated package-private inference helper.
- The helper independently derives the exact ordered two or three descriptors and produces no
  deferred constraint for valid fully static occurrences.
- Invalid manually assembled occurrences fail in deterministic semantic-role order for floating
  type, static rank/Shape, positive feature/hidden extent, valid-length type/gradient/Shape,
  state agreement, checked packed extent, weight/bias type and Shape, output count, and stored
  output descriptor mismatches.
- Inference never reads Tensor storage or valid-length values and never validates runtime bounds.
- Each `RecurrentScan` producer captures as exactly one ordinary flat `CompiledNode` with the exact
  operation reference, direction, ordered 5/6 or 6/7 input IDs, ordered 2/3 output IDs, exact
  descriptors, and no time-dependent topology.
- Two canonical output wrappers from one producer deduplicate to one node; two structurally equal
  but producer-identity-distinct calls remain two nodes.
- Recurrent nodes are ineligible for CSE in every phase/order path. Other exact CSE cases retain
  their existing behavior.
- DCE removes a wholly dead recurrent occurrence but retains a live occurrence and every declared
  sibling slot when only one output is a root or consumer input.
- Multi-root capture and final compilation preserve caller output order, output-slot identity,
  and distinct producer occurrences.
- `FORWARD_ONLY` graph compilation succeeds through capture, initial inference, canonicalization,
  optional optimization, and final validation for RNN, GRU, and LSTM examples, including zero
  time and zero batch descriptors.
- Complete Compiler orchestration with an explicit recording test provider passes the exact
  recurrent operation and descriptor order through the existing `OperationCapabilityQuery`,
  publication binding, owner map, partition, logical memory, and diagnostics without adding a
  production capability claim.
- Every backward-capable compile whose complete original forward inventory contains a recurrent
  producer fails in deterministic producer postorder during `AutogradPreflight`, including when
  the requested gradient lies on a separate non-recurrent branch.
- Backward rejection leaves the next Tensor ID unchanged and occurs before any seed, derivative
  constant, formula Tensor, matching `ARGSORT`, accumulated gradient, or combined graph exists.
- `FirstOrderGradientCoverage.SIGNATURES`, its 128-row contract, its family-owner enum, and
  `FirstOrderGradientCoverageTest` remain unchanged. The exact three recurrent signatures remain
  deferred and keep their exact unknown/unclassified no-allocation assertions.
- Current static NN recurrent containers, Model public API/semantics, and every existing capability
  provider remain unchanged.
- Compiler and Model Javadocs, Compile/Tensor/Training API, Model capabilities, glossary, and the
  three focused architecture explanations state that forward Compiler adoption is current while
  BPTT, execution, runtime length reads, backend capability, and NN delegation remain planned.
- Production Javadocs explain recurrent inference, CSE identity exclusion, and allocation-free
  preflight behavior with complete `@param`, `@return`, and expected `@throws` tags on new or
  changed contract-relevant methods.
- The documentation pass records reasoned no-change conclusions for other Model recurrent
  Javadocs and tests, the authoritative architecture contract and ADR 0012, other NN/Planning/
  Prepare/Runtime/Engine/backend docs, architecture/conformance/integration tests, Gradle,
  dependencies, and the global roadmap.
- Exactly the eighteen authorized paths change. No file is staged, and concurrent CPU/backend/Data or
  global-roadmap changes are neither touched nor incorporated.
- Task 0006A and its Compiler master-plan row are `Ready` before implementation and become
  `Complete` only after implementation, focused/module validation, and the separate documentation
  pass. Compiler 0007 remains Draft without a detailed specification; no BPTT task specification
  is created.
- A separate documentation-focused agent pass has independently finalized affected Javadocs,
  Compile API, glossary impact, planning status/evidence, links, and terminology in the same
  overall change.

## Tests / validation

During implementation, run the focused Compiler capability tests:

```bash
./gradlew :modules:compiler:test \
  --tests io.github.pho001.synaptik.compiler.RecurrentScanInferenceTest \
  --tests io.github.pho001.synaptik.compiler.RecurrentScanCompilerTest \
  --tests io.github.pho001.synaptik.compiler.FirstOrderGradientCoverageTest
```

The focused command must prove all valid kinds/directions/cardinalities, invalid descriptor
categories and final mismatch context, identity-distinct capture, multi-root order, CSE/DCE,
forward-only optimized compilation, ordinary Planning-query handoff, exact deferred-gradient
partition, and allocation-free rejection for both backward-capable modes.

Run the existing Model boundary tests once to verify that Compiler consumes the committed public
construction seam and signature/result contracts without changing Model:

```bash
./gradlew :modules:model:test \
  --tests io.github.pho001.synaptik.model.operation.recurrent.RecurrentScanSemanticsTest \
  --tests io.github.pho001.synaptik.model.tensor.RecurrentScanExpressionTest
```

After executable Compiler code stabilizes, run one final affected-module suite:

```bash
./gradlew :modules:compiler:test
```

Record Gradle outcomes and JUnit XML suite/test/skip/failure/error counts. Do not repeat the full
Model suite because no Model executable source changes. Repository-wide and architecture-test
validation is deferred to continuous integration: this task changes one module, no dependency,
no architecture rule, no shared build configuration, and no executable backend or end-to-end
path.

The implementation context hands the exact focused, Model-boundary, and final Compiler evidence
to a separate clean documentation context. That pass must not rerun successful Java tests unless
it changes executable behavior or records a concrete stale-evidence risk.

Documentation pass:

```bash
./gradlew :modules:compiler:javadoc
./gradlew :modules:model:javadoc
python3 /tmp/validate_synaptik_markdown.py
git diff --check
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git diff --cached --name-only
git status --short
```

The documentation pass must also verify generated package-private Compiler Javadocs and rendered
`RecurrentScan` Javadoc; complete `@param`, `@return`, and expected `@throws` coverage; local
Markdown links and anchors; balanced fences; heading uniqueness; final newlines; trailing
whitespace; exact eighteen-path scope; unchanged Model API and 128-plus-three gradient boundary;
no recurrent region/body/receiver spelling; no capability/backend claim; no changes outside the
authorized Compiler, Model Javadoc, and explanatory/planning pages;
task/master status synchronization; Compiler 0007 Draft with no specification; no BPTT task
specification; and an empty Git index.

## Dependencies

- Accepted [ADR 0012](../../../../design/decisions/0012-fixed-recurrent-scan-without-regions.md)
  and completed [NN 0021A](../../../extensions/nn/tasks/0021a-fixed-recurrent-scan-architecture-decision.md)
  supply the fixed-family/no-region architecture — Complete.
- [Model 0025](../../model/tasks/0025-canonical-tensor-producer-outputs.md) supplies canonical
  producer output wrappers — Complete.
- [Model 0025E](../../model/tasks/0025e-fixed-recurrent-scan-semantic-family-and-tensor-expressions.md)
  supplies fixed semantics, signatures, descriptors, and results — Complete.
- [Model 0025F](../../model/tasks/0025f-recurrent-scan-expression-namespace-correction.md) supplies
  the corrected public stateless `RecurrentScan` construction seam and exact supported/deferred
  boundary test — Complete and the final Model prerequisite.
- Compiler 0001–0003B supply identity-based capture, inference/final-validation orchestration,
  canonicalization, exact optimization, DCE/CSE, and constant sidecars — Complete.
- Compiler 0004–0005E supply allocation-free autograd preflight, combined capture, complete
  artifacts/Planning orchestration, and the closed first-order coverage checkpoint — Complete.
- [Compiler 0006](0006-explicit-functional-gradient-requests-and-higher-order-differentiation.md)
  supplies the current one/two-stage functional request and backward-capable entry behavior —
  Complete.
- Existing Compiler dependencies on Model, Config, Planning, Backend Contract, and Trace are
  sufficient. No new dependency is required.

Within the Compiler master plan, 0006A is the first unfinished task and the next applicable
module frontier. The repository roadmap still records the separate CPU 0007A1G execution
frontier; this user-requested planning pass does not edit that concurrent roadmap or itself
authorize Compiler implementation ahead of the coordinator's execution order.

## Follow-up tasks

- A later explicit Compiler architecture and implementation task must design BPTT, including
  derivative equations, valid-length gradient routing, saved-state versus recomputation policy,
  any necessary canonical auxiliaries, higher-order boundary, and complete first-order inventory
  adoption. No ID or detailed specification is created here.
- A later concrete-backend task must truthfully advertise exact supported recurrent kind,
  direction, type, and Shape combinations and implement one-time lowering, resources, complete
  runtime length validation, dense padded outputs, final states, and skipped invalid-coordinate
  arithmetic.
- Existing future Engine work owns typed logical input and publication binding; later NN/Data work
  owns runtime-valid-length convenience and compatibility with static sequence containers.
- Compiler 0007 remains the separate Draft owner of exact constant identities and permission-aware
  algebra. It is not advanced or specified by this task.
- Dynamic Shapes, arbitrary masks, active-row compaction, serialization, general graph regions,
  and static-NN migration remain separate later decisions.

## Architecture impact

Expected impact: None.

This task implements the already accepted Compiler portion of ADR 0012 using the existing flat
graph, inference, optimization, publication, and Planning boundaries. It changes no architecture
rule, ownership, dependency direction, public Model surface, or execution contract. If correct
implementation requires a new graph representation, public Compiler API, shared loop/body
contract, another module change, or capability advertisement, stop and report the conflict.

## Implementation prompt

Use this prompt in a separate agentic task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, ADR 0012, docs/planning/planning-guide.md, the Compiler master
plan, and docs/planning/modules/compiler/tasks/0006a-fixed-recurrent-scan-forward-adoption-and-bptt-boundary.md.
Read the task's directly referenced Model recurrent, Compiler capture/inference/optimization/
autograd/artifact, Planning handoff, source, tests, and documentation contracts.

Implement Compiler 0006A exactly within its eighteen authorized paths. Preserve the existing ordinary
CompiledNode representation and the exact 128-supported-plus-three-deferred gradient boundary.
Do not implement BPTT, a region/body/loop IR, executable Model/NN/API changes, runtime value reads, capability
advertisement, backend execution, another module, dependencies, Gradle, architecture, or global-
roadmap changes. Stop on any architecture, inventory, or scope conflict.

After executable implementation and the recorded focused, Model-boundary, and final Compiler
validation, hand the actual diff and exact evidence to a separate documentation-focused agent or
thread with clean context. That pass must follow docs/developer-guide/documentation-rules.md and
finalize affected Javadocs, Compile API, glossary impact, planning status/evidence, and
documentation validation in the same overall change without repeating successful Java tests
unless executable behavior changes or a concrete stale-evidence risk is recorded.

Update this task with local decisions, exact validation evidence, implementation notes,
completion summary, and final status. Do not mark it Complete before every acceptance criterion
and the documentation pass finish.
```

## Local decisions

- The existing `CompiledNode` is the complete recurrent compiled representation. No new IR type
  or operation wrapper is authorized.
- Recurrent inference is a dedicated family helper because its ordered multi-state/parameter
  descriptor contract is cohesive and should not enlarge the already broad structured helper.
- Recurrent occurrences are excluded from CSE to preserve the architecture's identity-distinct
  occurrence contract; DCE retains its existing whole-node liveness rules.
- Backward-capable modes reject any recurrent occurrence in the complete original forward
  inventory, not only a selected gradient route, because the architecture makes `FORWARD_ONLY`
  the sole initial adopting mode.
- The first-order production gradient inventory and its exact boundary test remain unchanged.
  Forward inference inventory and gradient coverage inventory intentionally differ.
- A recording test provider may prove the unchanged ordinary Planning handoff but cannot create a
  production capability or backend claim.

## Known limitations

- No current production capability provider is expected to accept a recurrent query, so this task
  does not make the operation executable through a real backend or public Engine facade.
- Runtime valid-length contents remain unchecked until a concrete backend owns complete pre-write
  validation.
- Every recurrent Shape is fully static; dynamic or binding-dependent Shapes remain unsupported.
- BPTT and every recurrent derivative role remain fail-closed.
- Existing NN sequences continue to construct static unrolled graphs and are not redirected.

## Validation evidence

- Focused Compiler validation passed after one test-fixture correction:

  ```text
  ./gradlew :modules:compiler:test \
    --tests io.github.pho001.synaptik.compiler.RecurrentScanInferenceTest \
    --tests io.github.pho001.synaptik.compiler.RecurrentScanCompilerTest \
    --tests io.github.pho001.synaptik.compiler.FirstOrderGradientCoverageTest
  BUILD SUCCESSFUL; 3 suites, 21 tests, 0 skipped, 0 failures, 0 errors
  ```

  The first invocation exposed one invalid test setup that attempted to represent a gradient-
  eligible `INT64` descriptor, which `TensorDescriptor` itself correctly rejects. The test was
  corrected to preserve that upstream invariant, and the exact focused command then passed.
- The required committed Model boundary passed without changing Model:

  ```text
  ./gradlew :modules:model:test \
    --tests io.github.pho001.synaptik.model.operation.recurrent.RecurrentScanSemanticsTest \
    --tests io.github.pho001.synaptik.model.tensor.RecurrentScanExpressionTest
  BUILD SUCCESSFUL; 2 suites, 15 tests, 0 skipped, 0 failures, 0 errors
  ```

- The one authoritative affected-module suite passed after executable freeze:

  ```text
  ./gradlew :modules:compiler:test
  BUILD SUCCESSFUL; 33 suites, 224 tests, 0 skipped, 0 failures, 0 errors
  ```

- Final documentation and generated-surface validation passed:

  ```text
  ./gradlew :modules:compiler:javadoc
  BUILD SUCCESSFUL
  ./gradlew :modules:model:javadoc
  BUILD SUCCESSFUL
  javap -classpath modules/compiler/build/classes/java/main -p \
    io.github.pho001.synaptik.compiler.RecurrentScanInference
  javac external six-overload RecurrentScan surface probe
  javac/java package-local Compiler inventory/reflection probe
  supported=128, deferred=3, inference=final, inferMethods=1
  ```

  `javap` shows one package-private final type, one private constructor, and one package-private
  static `infer(Operation, List<TensorDescriptor>, int)` entry. Generated package-private
  Javadocs contain the recurrent inference contract, CSE identity exclusion, and allocation-free
  BPTT boundary, and rendered Model Javadocs contain the corrected `RecurrentScan` forward/backward
  status. The external compile accepts all six unchanged public biased or bias-free overloads.
  The reflection/inventory probe confirms the package-private final inference owner, one `infer`
  entry, 128 supported gradient rows, and exactly three recurrent signatures. The production
  coverage source and corrected `FirstOrderGradientCoverageTest` have no diff.
- Initial implementation scope and hygiene checks found the original ten authorized unstaged
  paths, an empty Git index, no forbidden production imports or added dependency/build/
  architecture/backend path, no Compiler 0007 specification, and no BPTT task specification.
  `git diff --check` passed. The independent documentation pass then found six stale current-
  status statements directly invalidated by forward adoption; the coordinator authorized those
  exact documentation-only corrections and expanded scope to sixteen paths. A later complete
  focused-doc scan found the same stale statement in the lifecycle explanation; the coordinator
  authorized that exact seventeenth path. The final exhaustive relevant-surface scan found stale
  current-status wording in the Model master plan; the coordinator authorized that exact
  eighteenth path. No other current claim says 0006A is Draft, forward recurrent inference
  rejects, or Compiler adoption is future. The completed Model 0025F row remains an accurate
  historical summary of what that earlier task preserved at its own completion.
- Final documentation and repository-hygiene gates passed:

  ```text
  python3 /tmp/validate_synaptik_markdown.py
  validated 11 Markdown files: links, anchors, headings, fences, whitespace
  git diff --check
  exactly 18 changed or untracked task paths
  git diff --cached --name-only
  <empty>
  ```

  The exact path probe finds four Compiler production files, two Compiler tests, one Model
  Javadoc file, and eleven documentation/planning files. Forbidden-import, unchanged-contract,
  status, no-0007-specification, and no-separate-BPTT-specification probes also pass.
- Repository-wide, architecture-test, backend-conformance, integration-test, and full Model-suite
  validation remain intentionally deferred to continuous integration under the task's tier rules:
  no architecture boundary, dependency, shared build, backend behavior, or end-to-end executable
  path changed.

## Implementation notes

- `CapturedGraphInference` now dispatches the closed recurrent family to the new stateless
  `RecurrentScanInference`. The helper validates exact kinds, direction attributes,
  cardinalities, common floating type, fully static role Shapes, positive feature/state extents,
  checked packed gate extent, non-gradient `INT64[batch]` valid lengths, weights/biases, and
  ordered output descriptors without reading values or creating constraints.
- Capture continues to emit one ordinary flat `CompiledNode` per Model producer. Recurrent nodes
  are explicitly excluded from CSE, while existing whole-node DCE retains every sibling output
  when any slot is live and removes only fully dead occurrences. Multi-root slot identity and
  caller order remain intact.
- Ordinary publication, owner selection, partitioning, logical-memory planning, and diagnostics
  receive the exact recurrent operation and ordered descriptors through the existing
  `OperationCapabilityQuery`. No production provider advertises recurrence and no execution path
  was added.
- `AutogradPreflight` inventories the complete original forward graph and rejects the first
  recurrent producer in deterministic producer postorder before request-stage, seed, or formula
  validation. Both backward-capable modes preserve the next Tensor ID on rejection, including
  when the requested gradient belongs to a separate non-recurrent branch.
- Focused tests cover all three kinds, both directions, both bias arities, exact capture and query
  order, zero time/batch, invalid descriptors, final-validation context, producer identity, CSE,
  whole-node DCE, multi-root slots, unrelated CSE behavior, both backward modes, deterministic
  diagnostics, and no-allocation rejection.
- Compile API, Tensor API, Training API, Model capabilities, glossary, `RecurrentScan` Javadoc,
  module boundaries, training graph, lifecycle, and Model master-plan wording describe current
  forward adoption and the unchanged backward/execution boundary. The independent documentation
  pass added only the eight coordinator-authorized stale-status corrections beyond the original
  ten paths. Other Model recurrent Javadocs/tests and planning records, the authoritative
  architecture contract and ADR 0012, NN, Planning, Prepare, Runtime, Engine, backend,
  conformance/integration tests, Gradle, dependencies, and the global roadmap need no change
  because their contracts, historical records, or current status were not altered by this task.

## Completion summary

Implemented, executable-tested, independently documented, and validated the fixed recurrent-scan
forward Compiler boundary in exactly eighteen authorized paths: four Compiler production files,
two focused Compiler tests, one Model Javadoc file, and eleven documentation/planning files. The
successful Java evidence remains the frozen focused 21-test, Model-boundary 15-test, and full
Compiler 224-test results; no executable Java changed during the documentation pass, so those
suites were not repeated. Both Javadoc tasks, generated-page inspection, public external compile,
package-local reflection/inventory, Markdown, stale-status, import, forbidden-scope, exact-path,
whitespace, and empty-index gates pass. No blocker or follow-up remains within task 0006A.

Status: Complete
