# Task 0006B6: Final [Convolution](../../../../glossary.md#term-channels-first-convolution-layers) Logical-[Layout](../../../../glossary.md#layout) Closure

## Status

Complete

## Goal

Close the final [Compiler](../../../../glossary.md#term-compiler)-owned logical-[layout](../../../../glossary.md#layout) gap that prevents [Planning](../../../../glossary.md#term-planning) from asking a [backend](../../../../glossary.md#backend)
about otherwise valid public [NCW](../../../../glossary.md#term-ncw) Conv1d, grouped [NCHW](../../../../glossary.md#term-nchw) [Conv2d](../../../../glossary.md#convolution--conv2d), and grouped [NCDHW](../../../../glossary.md#conv3d--ncdhw) [Conv3d](../../../../glossary.md#conv3d--ncdhw) forward
[graphs](../../../../glossary.md#graph). After final [graph optimization](../../../../glossary.md#forward-graph-optimization) and [validation](../../../../glossary.md#term-validation), assign [canonical](../../../../glossary.md#term-canonical-form) [contiguous](../../../../glossary.md#term-contiguous) logical layout
to eligible static [convolution](../../../../glossary.md#term-channels-first-convolution-layers) results and propagate that newly closed layout through only the
direct [singleton](../../../../glossary.md#term-singleton)-height `SQUEEZE` used by the existing public [Conv1d composition](../../../../glossary.md#conv1d-composition).

Keep [Model](../../../../glossary.md#term-model) construction unchanged: Conv2d and Conv3d expressions continue to begin with unresolved
result layouts, and Conv1d remains the visible `EXPAND_DIMS -> CONV2D -> SQUEEZE` composition.
Planning remains a backend-neutral consumer of final [descriptors](../../../../glossary.md#term-descriptor), [CPU](../../../../glossary.md#term-cpu) capability [admission](../../../../glossary.md#term-admission) remains
strict, and [Prepare](../../../../glossary.md#prepare) and [Runtime](../../../../glossary.md#term-runtime) gain no layout [inference](../../../../glossary.md#term-inference).

## Current defect

`TensorConv2dExpressions` and `TensorConv3dExpressions` intentionally construct [convolution](../../../../glossary.md#term-channels-first-convolution-layers) result
[descriptors](../../../../glossary.md#term-descriptor) with unresolved [layout](../../../../glossary.md#layout). `StructuredOperationInference` independently validates their
type and [Shape](../../../../glossary.md#shape) but derives the same unresolved result. `GraphCompiler` therefore gives [Planning](../../../../glossary.md#term-planning) an
[`OperationCapabilityQuery`](../../../../glossary.md#operation-capability-query--operationcapabilityquery) whose [Conv2d](../../../../glossary.md#convolution--conv2d) or [Conv3d](../../../../glossary.md#conv3d--ncdhw) output layout is absent. The [CPU](../../../../glossary.md#term-cpu) capability
provider correctly rejects every [occurrence](../../../../glossary.md#term-occurrence) with unresolved input or output geometry.

A following `contiguous()` cannot repair the occurrence already queried by Planning. Conv1d has an
additional consequence: its Conv2d result is unresolved when ordinary `SQUEEZE` [inference](../../../../glossary.md#term-inference) runs, so
the public [rank](../../../../glossary.md#term-rank)-three result also remains unresolved even when the input and [weight](../../../../glossary.md#term-weight) leaves have
static [contiguous](../../../../glossary.md#term-contiguous) layouts. CPU already implements the numerical families; the missing fact is the
[Compiler](../../../../glossary.md#term-compiler)'s final logical descriptor, not [backend](../../../../glossary.md#backend) behavior.

## Selected mechanism

Add one package-private `ConvolutionLogicalLayoutClosure` final-[graph](../../../../glossary.md#graph) rewrite in
`io.github.pho001.synaptik.compiler`. Invoke it exactly once in both graph-[compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) branches,
after `ForwardGraphOptimization.optimize(...)` and
`PublishedCompileTimeConstantDescriptorClosure.close(...)`, and before final forward/[gradient](../../../../glossary.md#term-gradient)
[bindings](../../../../glossary.md#term-binding), [publication](../../../../glossary.md#term-publication), capability queries, partitioning, or logical-memory [planning](../../../../glossary.md#term-planning).

This is not a controlled [inference](../../../../glossary.md#term-inference) mode. Current inference remains the independent [validation](../../../../glossary.md#term-validation)
[oracle](../../../../glossary.md#term-test-oracle) for [Model](../../../../glossary.md#term-model) construction semantics, including unresolved [convolution](../../../../glossary.md#term-channels-first-convolution-layers) [layouts](../../../../glossary.md#layout), and every
optimization candidate continues to pass that oracle before closure. The new pass changes only
final logical [descriptors](../../../../glossary.md#term-descriptor) after [topology](../../../../glossary.md#term-topology) and semantics are fixed. It follows the existing
post-validation closure seam established by [Compiler](../../../../glossary.md#term-compiler) 0006B5 and avoids a second inference policy
that would conflate expression validation with final Compiler layout choice.

The pass processes final [nodes](../../../../glossary.md#node) in stored graph order and maintains the descriptor selected for
each [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot). It may make exactly two kinds of replacement:

1. an eligible [`CONV2D`](../../../../glossary.md#convolution--conv2d) or [`CONV3D`](../../../../glossary.md#conv3d--ncdhw) output receives [canonical](../../../../glossary.md#term-canonical-form) [contiguous](../../../../glossary.md#term-contiguous) logical layout; and
2. an eligible direct `SQUEEZE(axis=2)` output receives the exact view layout derived from a
   Conv2d output closed by this same pass.<span data-glossary-navigation="true"> ([view glossary](../../../../glossary.md#term-view-meanings))</span>

It adds no node, edge, value, [operation](../../../../glossary.md#operation), attribute, phase, or synthetic Conv1d kind.

## Exact [convolution](../../../../glossary.md#term-channels-first-convolution-layers) closure rule

A final [node](../../../../glossary.md#node) output is eligible for convolution closure if and only if all of these facts hold:

1. the exact final [operation](../../../../glossary.md#operation) kind is `Conv2dKind.CONV2D` or `Conv3dKind.CONV3D`;
2. the node has the already validated single output for that kind;
3. the output [descriptor](../../../../glossary.md#term-descriptor)'s [`Shape`](../../../../glossary.md#shape) is fully static; and
4. the output descriptor's [layout](../../../../glossary.md#layout) is unresolved.

For an eligible output, replace only its `TensorDescriptor` with one retaining the exact
[`DataType`](../../../../glossary.md#data-type--datatype), exact `Shape`, and exact `requiresGrad` value and containing
`LayoutDescriptor.contiguous(shape)`. This is a [backend](../../../../glossary.md#backend)-independent logical choice for the fresh
convolution result, not a physical representation or [CPU](../../../../glossary.md#term-cpu) route.

Static versus dynamic behavior is strict:

- a fully static eligible Shape closes to the Model's canonical row-major layout, including the
  established scalar/zero-extent behavior even though current convolution ranks are four/five;<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [row-major glossary](../../../../glossary.md#term-row-major); [scalar glossary](../../../../glossary.md#term-scalar); [extent glossary](../../../../glossary.md#term-extent))</span>
- a dynamic or partially dynamic convolution result remains unresolved and unchanged;
- an already resolved result remains unchanged; and
- the pass does not infer bindings, specialize symbolic dimensions, or use input storage to make
  a Shape static.<span data-glossary-navigation="true"> ([bindings glossary](../../../../glossary.md#term-binding); [dimensions glossary](../../../../glossary.md#dimension))</span>

Use `LayoutDescriptor.contiguous(shape)` as the sole source of [stride](../../../../glossary.md#term-stride) and referenced-span
arithmetic. If [canonical](../../../../glossary.md#term-canonical-form) stride or span arithmetic overflows, fail [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) with a deterministic
`IllegalArgumentException` identifying the operation kind, [`NodeId`](../../../../glossary.md#nodeid), and output [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot), and
retain the arithmetic failure as the cause. Do not skip the eligible output or leave it unresolved.

## Exact Conv1d propagation rule

Public Conv1d remains four ordinary [producers](../../../../glossary.md#term-producer): input `EXPAND_DIMS(axis=2)`, [weight](../../../../glossary.md#term-weight)
`EXPAND_DIMS(axis=2)`, mapped [`CONV2D`](../../../../glossary.md#convolution--conv2d), and result `SQUEEZE(axis=2)`. The new pass does not identify
that sequence as a fake Conv1d [operation](../../../../glossary.md#operation) and does not inspect ancestors to classify it.

Instead, after closing a Conv2d result, the pass applies the existing exact [singleton](../../../../glossary.md#term-singleton)-[axis](../../../../glossary.md#term-axis) [view](../../../../glossary.md#term-view-meanings)
rule only when a final [node](../../../../glossary.md#node) satisfies all of these facts:

1. the operation kind is `AxisTransformKind.SQUEEZE` with `AxisTransformAttrs.axis() == 2`;
2. it has one input and one output;
3. its exact input [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) is a Conv2d output newly closed by this pass;
4. the input [Shape](../../../../glossary.md#shape) is fully static [rank](../../../../glossary.md#term-rank) four and axis two has [extent](../../../../glossary.md#term-extent) one;
5. the output Shape is the already validated rank-three Shape obtained by removing axis two; and
6. the output [layout](../../../../glossary.md#layout) is unresolved.

Derive the result layout by dropping [stride](../../../../glossary.md#term-stride) index two from the newly closed Conv2d layout,
preserving its [storage offset](../../../../glossary.md#term-storage-offset), and constructing `LayoutDescriptor.of(outputShape, strides,
offset, true)`. Thus the visible Conv1d output becomes a resolved view without changing the two
upstream `EXPAND_DIMS` nodes, the Conv2d [occurrence](../../../../glossary.md#term-occurrence), or the squeeze occurrence. This rule also
remains correct if the Conv2d result has another consumer or [publication](../../../../glossary.md#term-publication).

If checked stride/span arithmetic overflows, fail with deterministic `IllegalArgumentException`
context naming the squeeze [`NodeId`](../../../../glossary.md#nodeid), input `ValueId`, and output `ValueId`, retaining the arithmetic
cause. If any eligibility fact is absent, preserve the squeeze [descriptor](../../../../glossary.md#term-descriptor) unchanged. Do not
propagate recursively through `PERMUTE`, another `SQUEEZE`, `EXPAND_DIMS`, reshape, [contiguous](../../../../glossary.md#term-contiguous),
[slice](../../../../glossary.md#slice), publication, or any unrelated operation. In particular, do not introduce a general
post-[inference](../../../../glossary.md#term-inference) layout propagation policy.

## Preservation invariants

If no [descriptor](../../../../glossary.md#term-descriptor) is eligible, return the exact input `ValidatedGraph`. Otherwise rebuild only
changed `GraphValue` descriptors and preserve:

- value encounter order, every [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot), and exact unchanged `GraphValue` references;
- [node](../../../../glossary.md#node) order, every [`NodeId`](../../../../glossary.md#nodeid), exact `CompiledNode` and [`Operation`](../../../../glossary.md#operation) instances, inputs, outputs,
  attributes, and phases;
- [graph](../../../../glossary.md#graph) input/output order and [publication](../../../../glossary.md#term-publication) mapping;
- [compile](../../../../glossary.md#compile)-time [constant](../../../../glossary.md#term-constant) splats and their [identities](../../../../glossary.md#term-identity-meanings);
- caller-bindable [`TensorId`](../../../../glossary.md#tensorid) to final-`ValueId` associations;
- deferred constraints and their order;
- derivative-order metadata and the exact mapping values, rebuilding only its graph reference as
  required by immutable graph ownership;<span data-glossary-navigation="true"> ([derivative glossary](../../../../glossary.md#term-derivative); [immutable glossary](../../../../glossary.md#term-immutable); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- forward and [gradient](../../../../glossary.md#term-gradient) publication identities, [target](../../../../glossary.md#term-target) [metadata](../../../../glossary.md#term-metadata), [occurrence](../../../../glossary.md#term-occurrence) order, and roles;
- [CSE](../../../../glossary.md#term-common-subexpression-elimination-cse)/[DCE](../../../../glossary.md#term-dead-code-elimination-dce) results, [aliases](../../../../glossary.md#aliasing-and-arena), and final graph [topology](../../../../glossary.md#term-topology); and
- [Planning](../../../../glossary.md#term-planning) query order, [partition](../../../../glossary.md#partition) inputs, and logical-memory [identity](../../../../glossary.md#term-identity-meanings).

The pass must not rerun CSE, DCE, [constant folding](../../../../glossary.md#term-constant-folding), [inference](../../../../glossary.md#term-inference), or graph [validation](../../../../glossary.md#term-validation) after closure.
It must not allocate storage, select a [backend](../../../../glossary.md#backend), create physical geometry, or modify [Model](../../../../glossary.md#term-model),
Planning, [Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime), [Engine](../../../../glossary.md#term-engine), or [CPU](../../../../glossary.md#term-cpu) production behavior.

## [Scope](../../../../glossary.md#term-scope)

- Add the dedicated package-private final [graph](../../../../glossary.md#graph) closure described above.
- Route forward-only and backward-capable graph [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) through it after 0006B5 closure.
- Add focused Compiler coverage for positive, negative, dynamic, arithmetic-failure, metadata,
  topology, and Planning-query cases.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [metadata glossary](../../../../glossary.md#term-metadata); [topology glossary](../../../../glossary.md#term-topology); [Planning glossary](../../../../glossary.md#term-planning))</span>
- Update the existing `Conv3dCompilerTest` Planning-query assertion so it compares the query
  output with the final compiled Conv3d node output descriptor rather than the intentionally
  unresolved Model producer descriptor.<span data-glossary-navigation="true"> ([Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [node glossary](../../../../glossary.md#node); [descriptor glossary](../../../../glossary.md#term-descriptor); [Model glossary](../../../../glossary.md#term-model); [producer glossary](../../../../glossary.md#term-producer))</span>
- Add exact public Engine numerical integration fixtures for NCW Conv1d, grouped NCHW Conv2d,
  and grouped NCDHW Conv3d using only public Tensor and Engine APIs.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [NCW glossary](../../../../glossary.md#term-ncw); [NCHW glossary](../../../../glossary.md#term-nchw); [Conv2d glossary](../../../../glossary.md#convolution--conv2d); [NCDHW glossary](../../../../glossary.md#conv3d--ncdhw); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Update the Compile API and synchronized planning documents through the required separate clean
  documentation-focused pass.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- Changes to convolution construction, Shape semantics, attributes, gradients, operation kinds,
  Model Javadocs, or the unresolved descriptors visible before compilation.<span data-glossary-navigation="true"> ([convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [Shape glossary](../../../../glossary.md#shape); [operation glossary](../../../../glossary.md#operation); [Model glossary](../../../../glossary.md#term-model); [descriptors glossary](../../../../glossary.md#term-descriptor); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names))</span>
- [Conv3d](../../../../glossary.md#conv3d--ncdhw) [gradients](../../../../glossary.md#term-gradient); [Compiler](../../../../glossary.md#term-compiler) 0006C remains Draft and forward-only for Conv3d remains explicit.
- General default layout, layout propagation, dynamic Shape binding, graph specialization, or an
  inference mode.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [binding glossary](../../../../glossary.md#term-binding); [graph glossary](../../../../glossary.md#graph); [specialization glossary](../../../../glossary.md#term-specialization); [inference glossary](../../../../glossary.md#term-inference))</span>
- Planning policy, capability relaxation, backend fallback, Engine behavior, CPU behavior,
  Prepare/Runtime inference, physical layout, storage, allocation, execution, or materialization.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [backend glossary](../../../../glossary.md#backend); [Engine glossary](../../../../glossary.md#term-engine); [CPU glossary](../../../../glossary.md#term-cpu); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [materialization glossary](../../../../glossary.md#materialization))</span>
- Forged [`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel), [`OperationCapabilityQuery`](../../../../glossary.md#operation-capability-query--operationcapabilityquery), prepared artifacts, CPU internals, or
  manually constructed inward descriptors as substitutes for public integration evidence.<span data-glossary-navigation="true"> ([artifacts glossary](../../../../glossary.md#term-artifact))</span>
- Architecture, ADR, dependency, Gradle, conformance, generated-code, performance, or unrelated
  documentation work.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Reordering or implementing Compiler 0006C or 0007.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Runtime/Prepare/backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend))</span>
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span> and [roadmap](../../../roadmap.md)
- [Compiler master plan](../master-plan.md)
- [Compiler](../../../../glossary.md#term-compiler) [0006B](0006b-conv3d-forward-adoption-and-explicit-gradient-boundary.md),
  [0006B3](0006b3-public-constant-free-complete-compile-entry.md),
  [0006B4](0006b4-stable-caller-input-tensor-identity-bindings.md), and
  [0006B5](0006b5-published-compile-time-constant-descriptor-closure.md)
- [Engine master plan](../../engine/master-plan.md)<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span> and
  [Engine 0008](../../engine/tasks/0008-engine-lifecycle-capability-checkpoint.md)

## Architecture constraints

[Compiler](../../../../glossary.md#term-compiler) owns [inference](../../../../glossary.md#term-inference), [validation](../../../../glossary.md#term-validation), final [immutable](../../../../glossary.md#term-immutable) logical [descriptors](../../../../glossary.md#term-descriptor), and construction of the
capability queries consumed by [Planning](../../../../glossary.md#term-planning). Planning remains [backend](../../../../glossary.md#backend)-neutral and selects [ownership](../../../../glossary.md#term-ownership-meanings)
from exactly those descriptors. [CPU](../../../../glossary.md#term-cpu) continues to require static resolved [injective](../../../../glossary.md#term-injective) [convolution](../../../../glossary.md#term-channels-first-convolution-layers)
descriptors. [Prepare](../../../../glossary.md#prepare) consumes planned facts without inferring [layout](../../../../glossary.md#layout), [Runtime](../../../../glossary.md#term-runtime) executes prepared
work without [graph](../../../../glossary.md#graph) interpretation, and [Engine](../../../../glossary.md#term-engine) remains the outer [composition root](../../../../glossary.md#term-composition-root).

No architecture rule changes. Stop if implementation needs a public Compiler contract, another
[module](../../../../glossary.md#term-module)'s production change, a backend-specific predicate, runtime [binding](../../../../glossary.md#term-binding) inference, an [ADR](../../../../glossary.md#term-architecture-decision-record-adr), or an
`ARCHITECTURE.md` edit.

## Package impact

Add one package-private, field-free transformation beside current [Compiler](../../../../glossary.md#term-compiler) [graph](../../../../glossary.md#graph) transformations
in `io.github.pho001.synaptik.compiler`. No public/exported package [shape](../../../../glossary.md#shape) changes. Add one
[integration test](../../../../glossary.md#term-integration-test) in `io.github.pho001.synaptik.testing.integration`; it validates existing public
cross-[module](../../../../glossary.md#term-module) behavior and adds no [Engine](../../../../glossary.md#term-engine) or [CPU](../../../../glossary.md#term-cpu) production surface.

## Affected files

Exact future implementation allowlist (twelve paths):

1. `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ConvolutionLogicalLayoutClosure.java` (new)
2. `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/GraphCompiler.java`
3. `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/ConvolutionLogicalLayoutClosureTest.java` (new)
4. `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/GraphCompilerTest.java`
5. `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/Conv3dCompilerTest.java`
6. `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/EngineConvolutionIntegrationTest.java` (new)
7. `docs/api/compile-api.md`
8. this task
9. `docs/planning/modules/compiler/master-plan.md`
10. `docs/planning/modules/engine/master-plan.md`
11. `docs/planning/modules/engine/tasks/0008-engine-lifecycle-capability-checkpoint.md`
12. `docs/planning/roadmap.md`

Review without editing [Model](../../../../glossary.md#term-model), [Planning](../../../../glossary.md#term-planning), [Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime), [Engine](../../../../glossary.md#term-engine), [CPU](../../../../glossary.md#term-cpu), Config, [Backend](../../../../glossary.md#backend) Contract,
[Gradle](../../../../glossary.md#term-gradle), architecture/ADRs, architecture tests, backend [conformance](../../../../glossary.md#term-conformance), other [API](../../../../glossary.md#term-application-programming-interface-api) guides, or glossary.
If a required thirteenth path appears, stop and replan rather than [widening](../../../../glossary.md#term-widening) [scope](../../../../glossary.md#term-scope) implicitly.

## Maximum [scope](../../../../glossary.md#term-scope)

At most the twelve paths above: two [Compiler](../../../../glossary.md#term-compiler) production paths, three Compiler test paths, one public
integration-test path, [Compile](../../../../glossary.md#compile) [API](../../../../glossary.md#term-application-programming-interface-api), and five synchronized task/master/roadmap paths. No production
[module](../../../../glossary.md#term-module) outside Compiler changes.

## Exact public numerical fixtures

`EngineConvolutionIntegrationTest` must construct ordinary public storage-backed [FLOAT32](../../../../glossary.md#term-float32) [Tensor](../../../../glossary.md#tensor)
[leaves](../../../../glossary.md#term-leaf) with [canonical](../../../../glossary.md#term-canonical-form) [layouts](../../../../glossary.md#layout), build expressions through public `Tensor.conv1d`, [`conv2d`](../../../../glossary.md#convolution--conv2d), and
[`conv3d`](../../../../glossary.md#conv3d--ncdhw), and execute through public `Engine.standard()` only. For each [fixture](../../../../glossary.md#term-fixture), exercise the
ordinary reusable `compile -> prepare -> run` path with input Tensors supplied in non-[Compiler](../../../../glossary.md#term-compiler)
order to prove typed [`TensorId`](../../../../glossary.md#tensorid) [binding](../../../../glossary.md#term-binding), materialize the public result, and also exercise the
one-shot `compute` path. Do not import Compiler, [Planning](../../../../glossary.md#term-planning), [Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime), [Backend](../../../../glossary.md#backend) Contract, or
[CPU](../../../../glossary.md#term-cpu) types.

Use these exact fixtures and [row-major](../../../../glossary.md#term-row-major) expected values:

- [NCW](../../../../glossary.md#term-ncw) Conv1d: input [Shape](../../../../glossary.md#shape) `[1,1,4]` values `[1,2,3,4]`, [weight](../../../../glossary.md#term-weight) Shape `[1,1,2]` values
  `[1,1]`, [bias](../../../../glossary.md#term-bias) Shape `[1]` value `[0.5]`, default attributes, output Shape `[1,1,3]`, expected
  `[3.5,5.5,7.5]`.
- grouped [NCHW](../../../../glossary.md#term-nchw) Conv2d: input Shape `[1,2,2,2]` values `[1,2,3,4,5,6,7,8]`, weight Shape
  `[2,1,1,1]` values `[2,-1]`, bias Shape `[2]` values `[1,10]`, attributes
  `(strideHeight=1, strideWidth=1, paddingHeight=0, paddingWidth=0, dilationHeight=1,
  dilationWidth=1, groups=2)`, output Shape `[1,2,2,2]`, expected
  `[3,5,7,9,5,4,3,2]`.
- grouped [NCDHW](../../../../glossary.md#conv3d--ncdhw) Conv3d: input Shape `[1,2,2,1,2]` values `[1,2,3,4,5,6,7,8]`, weight Shape
  `[2,1,1,1,1]` values `[2,-1]`, bias Shape `[2]` values `[1,10]`, attributes
  `(strideDepth=1, strideHeight=1, strideWidth=1, paddingDepth=0, paddingHeight=0,
  paddingWidth=0, dilationDepth=1, dilationHeight=1, dilationWidth=1, groups=2)`, output Shape
  `[1,2,2,1,2]`, expected `[3,5,7,9,5,4,3,2]`.

Assert output [descriptor](../../../../glossary.md#term-descriptor) Shape and resolved layout as well as numerical values. Conv3d is forward
only; do not request [gradients](../../../../glossary.md#term-gradient) or imply [derivative](../../../../glossary.md#term-derivative) support.

## Acceptance criteria

- [Model](../../../../glossary.md#term-model) [Conv2d](../../../../glossary.md#convolution--conv2d)/[Conv3d](../../../../glossary.md#conv3d--ncdhw) construction and ordinary [inference](../../../../glossary.md#term-inference) still produce unresolved [layouts](../../../../glossary.md#layout).
- The dedicated pass runs after final optimization and 0006B5 closure, and before every final
  binding, publication, capability, partition, and logical-memory derivation in both compile
  branches.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [publication glossary](../../../../glossary.md#term-publication); [partition glossary](../../../../glossary.md#partition); [compile glossary](../../../../glossary.md#compile))</span>
- Only fully static unresolved Conv2d/Conv3d outputs close to [canonical](../../../../glossary.md#term-canonical-form) [contiguous](../../../../glossary.md#term-contiguous) layout.
- Dynamic/partially dynamic and already resolved [descriptors](../../../../glossary.md#term-descriptor) remain unchanged.
- Only direct unresolved `SQUEEZE(axis=2)` outputs consuming a Conv2d result newly closed by the
  pass receive the exact view layout; no fake Conv1d kind or general propagation policy appears.<span data-glossary-navigation="true"> ([view glossary](../../../../glossary.md#term-view-meanings))</span>
- Canonical and squeeze-view arithmetic uses checked Model factories and failures retain operation,
  node, value, and arithmetic-cause context.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [node glossary](../../../../glossary.md#node))</span>
- IDs, topology, operation instances, constants, bindable Tensor IDs, constraints, phases,
  derivative metadata, publication mapping, CSE/DCE results, and graph order are exact.<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [constants glossary](../../../../glossary.md#term-constant); [Tensor glossary](../../../../glossary.md#tensor); [derivative glossary](../../../../glossary.md#term-derivative); [metadata glossary](../../../../glossary.md#term-metadata); [CSE glossary](../../../../glossary.md#term-common-subexpression-elimination-cse); [DCE glossary](../../../../glossary.md#term-dead-code-elimination-dce); [graph glossary](../../../../glossary.md#graph))</span>
- Recording capability-provider tests prove Planning queries receive resolved Conv2d, Conv1d
  squeeze, and Conv3d result descriptors from the final graph; dynamic convolution queries remain
  unresolved and unsupported rather than being silently closed.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers))</span>
- Existing `Conv3dCompilerTest.passesExactPublicationDiagnosticsAndPlanningQueryWithoutProviderClaim`
  compares `query.outputs()` with the final compiled Conv3d node output descriptor, while the
  Model producer descriptor remains unresolved as required by the construction contract.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer))</span>
- The exact three public Engine fixtures pass through typed binding and one-shot compute without
  inward-artifact construction or CPU-internal imports.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact))</span>
- [CPU](../../../../glossary.md#term-cpu) capability remains strict and no [Engine](../../../../glossary.md#term-engine)/CPU production behavior changes.
- Public/package inventories, [dependency](../../../../glossary.md#term-dependency) declarations, and architecture tests remain unchanged.
- The task is `Complete` only after implementation and the independent documentation pass
  complete; Engine 0008 then returns to [`Ready`](../../../../glossary.md#term-ready) with this prerequisite and its positive fixtures
  satisfied.

## Tests and [validation](../../../../glossary.md#term-validation)

Focused [Compiler](../../../../glossary.md#term-compiler) tests must prove:

- static [Conv2d](../../../../glossary.md#convolution--conv2d) and [Conv3d](../../../../glossary.md#conv3d--ncdhw) [canonical](../../../../glossary.md#term-canonical-form) closure, exact [descriptor](../../../../glossary.md#term-descriptor) fields, and [graph](../../../../glossary.md#graph)-query visibility;
- the direct Conv2d-to-`SQUEEZE(axis=2)` [view](../../../../glossary.md#term-view-meanings) rule and visible Conv1d result [layout](../../../../glossary.md#layout);
- partially dynamic Conv2d/Conv3d and dynamic [Conv1d composition](../../../../glossary.md#conv1d-composition) remain unresolved;
- unrelated squeeze axes, squeeze inputs not newly closed by the pass, and other layout operations
  remain unchanged;<span data-glossary-navigation="true"> ([axes glossary](../../../../glossary.md#term-axis))</span>
- already resolved and non-convolution values remain unchanged, including exact input identity
  when nothing qualifies;<span data-glossary-navigation="true"> ([convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- deterministic canonical-layout overflow failure with retained cause, plus checked-factory use
  and contextual failure handling for squeeze-view construction;<span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow))</span>
- exact preservation of identities, topology, operations, constants, bindable IDs, constraints,
  phases, derivatives, publications, CSE/DCE results, and order; and<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [constants glossary](../../../../glossary.md#term-constant); [derivatives glossary](../../../../glossary.md#term-derivative); [CSE glossary](../../../../glossary.md#term-common-subexpression-elimination-cse); [DCE glossary](../../../../glossary.md#term-dead-code-elimination-dce))</span>
- complete [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) builds each [`OperationCapabilityQuery`](../../../../glossary.md#operation-capability-query--operationcapabilityquery) from the final closed descriptors;
  the existing Conv3d query test must assert that final descriptor instead of the unresolved
  Model producer descriptor.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [producer glossary](../../../../glossary.md#term-producer))</span>

Integration tests must execute the three exact numerical fixtures above through the public [Engine](../../../../glossary.md#term-engine)
typed [lifecycle](../../../../glossary.md#lifecycle) and `compute`, including reversed supplied input order for reusable runs, [detached](../../../../glossary.md#detached-model-results)
[materialization](../../../../glossary.md#materialization), expected [Shape](../../../../glossary.md#shape)/layout, and exact [FLOAT32](../../../../glossary.md#term-float32) arrays. No [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance) or [CPU](../../../../glossary.md#term-cpu)
test is added because backend semantics and strict capability are unchanged.

Future implementation task-tier validation:

```bash
./gradlew :modules:compiler:test --tests io.github.pho001.synaptik.compiler.ConvolutionLogicalLayoutClosureTest --tests io.github.pho001.synaptik.compiler.GraphCompilerTest --tests io.github.pho001.synaptik.compiler.Conv3dCompilerTest
./gradlew :testing:integration-tests:test --tests io.github.pho001.synaptik.testing.integration.EngineConvolutionIntegrationTest
./gradlew :modules:compiler:test
./gradlew :testing:integration-tests:test
./gradlew :modules:compiler:javadoc
git diff --check
```

The clean documentation pass validates Markdown links, anchors, heading uniqueness, fences,
whitespace/final newlines, exact twelve-path [scope](../../../../glossary.md#term-scope), package/public inventories, and task/master/
roadmap status and order. Do not rerun successful Java suites in that pass unless it changes Java
or finds a concrete executable discrepancy. Validation tier: affected Compiler [module](../../../../glossary.md#term-module), affected
integration-test suite, Compiler [Javadoc](../../../../glossary.md#term-javadoc), and documentation; no repository-wide or performance
[checkpoint](../../../../glossary.md#term-checkpoint) is justified.

## Risks

- Closing every static unresolved descriptor would silently create a general default-layout
  policy; eligibility must remain tied to final Conv2d/Conv3d producer occurrences.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [layout glossary](../../../../glossary.md#layout); [Conv2d glossary](../../../../glossary.md#convolution--conv2d); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [producer glossary](../../../../glossary.md#term-producer))</span>
- Closing the Conv2d value without its exact direct squeeze would leave the public Conv1d result
  unresolved; propagating beyond that squeeze would broaden the task into unrelated inference.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference))</span>
- Rerunning inference after closure would compare against Model's intentionally unresolved
  convolution descriptor and conflate validation with final descriptor choice.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [validation glossary](../../../../glossary.md#term-validation))</span>
- Rebuilding graph objects carelessly could change node/operation identity, publications,
  bindable Tensor IDs, derivative graph ownership, or final order even when topology looks equal.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [node glossary](../../../../glossary.md#node); [operation glossary](../../../../glossary.md#operation); [identity glossary](../../../../glossary.md#term-identity-meanings); [Tensor glossary](../../../../glossary.md#tensor); [derivative glossary](../../../../glossary.md#term-derivative); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [topology glossary](../../../../glossary.md#term-topology))</span>
- A fixture using forged descriptors or CPU internals could pass without proving the public
  compile/plan/prepare/run path; the integration allowlist and import boundary must be enforced.<span data-glossary-navigation="true"> ([fixture glossary](../../../../glossary.md#term-fixture); [CPU glossary](../../../../glossary.md#term-cpu); [compile glossary](../../../../glossary.md#compile); [prepare glossary](../../../../glossary.md#prepare); [run glossary](../../../../glossary.md#term-run))</span>
- Treating static logical layout as physical layout could leak backend behavior into Compiler or
  weaken CPU admission; the pass must create no representation or capability fact.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [Compiler glossary](../../../../glossary.md#term-compiler); [admission glossary](../../../../glossary.md#term-admission))</span>
- Marking Engine 0008 Ready before 0006B6 and all three public fixtures complete would recreate
  the checkpoint's original acceptance gap.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [Ready glossary](../../../../glossary.md#term-ready); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- The recorded full Compiler failure shows the pre-existing Conv3d Planning-query test still
  expects the unresolved Model producer descriptor even though the selected closure correctly
  supplies the final canonical contiguous descriptor. That exact stale assertion must be updated,
  so its test path is the required twelfth allowlisted path rather than an implicit scope overrun.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [canonical glossary](../../../../glossary.md#term-canonical-form); [contiguous glossary](../../../../glossary.md#term-contiguous); [scope glossary](../../../../glossary.md#term-scope))</span>

## Dependencies

- [Compiler](../../../../glossary.md#term-compiler) 0006B, 0006B3, 0006B4, and 0006B5 — Complete.
- Current [Model](../../../../glossary.md#term-model) [Conv1d composition](../../../../glossary.md#conv1d-composition) and [Conv2d](../../../../glossary.md#convolution--conv2d)/[Conv3d](../../../../glossary.md#conv3d--ncdhw) unresolved construction semantics — stable.
- Current [Planning](../../../../glossary.md#term-planning) capability-query [handoff](../../../../glossary.md#term-handoff) and strict [CPU](../../../../glossary.md#term-cpu) [convolution](../../../../glossary.md#term-channels-first-convolution-layers) capability — stable.
- [Engine](../../../../glossary.md#term-engine) 0001–0007 and CPU dimensional-convolution execution — Complete.
- Engine 0008 — [Ready](../../../../glossary.md#term-ready) after this task and its public positive fixtures completed.

This task is a user-authorized prerequisite-order exception: after selection of blocked Engine
0008, work returns to the Compiler plan for one bounded owning prerequisite. Within Compiler it is
inserted in normal order immediately after 0006B5 and before Draft 0006C; it does not reorder or
implement 0006C.

## Documentation-focused clean-context pass

After Java and integration evidence stabilizes, hand the exact diff, commands/results, [descriptor](../../../../glossary.md#term-descriptor)
rules, preservation evidence, and baseline [scope](../../../../glossary.md#term-scope) to a distinct clean documentation-focused
context. Apply the General, [API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc), and [Planning](../../../../glossary.md#term-planning) profiles. Finalize both changed production
Javadocs and `docs/api/compile-api.md`, synchronize this task, both master plans, [Engine](../../../../glossary.md#term-engine) 0008, and
the roadmap, and review glossary impact. Record reasoned no-change conclusions for [Tensor](../../../../glossary.md#tensor), Public,
[Runtime](../../../../glossary.md#term-runtime), and Training API guides; [Model](../../../../glossary.md#term-model)/Planning/[Prepare](../../../../glossary.md#prepare)/Runtime/Engine/[CPU](../../../../glossary.md#term-cpu) contracts;
architecture/ADRs; architecture and [conformance](../../../../glossary.md#term-conformance) tests; [Gradle](../../../../glossary.md#term-gradle); and other [modules](../../../../glossary.md#term-module).

## Follow-up tasks

- [Engine](../../../../glossary.md#term-engine) 0008 is now [`Ready`](../../../../glossary.md#term-ready); its documentation-only repository checkpoint consumes, rather than
  recreates, this task's exact public fixture evidence.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [fixture glossary](../../../../glossary.md#term-fixture))</span>
- [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [convolution](../../../../glossary.md#term-channels-first-convolution-layers) integration remains ordered after Engine 0008.
- [Compiler](../../../../glossary.md#term-compiler) 0006C remains the separate Draft [Conv3d](../../../../glossary.md#conv3d--ncdhw) [gradient](../../../../glossary.md#term-gradient) expressibility boundary.
- Compiler 0007 remains a separate Draft algebra side branch.

## Architecture impact

Expected impact: None. The task fills a final [Compiler](../../../../glossary.md#term-compiler) logical-[descriptor](../../../../glossary.md#term-descriptor) fact within existing
[ownership](../../../../glossary.md#term-ownership-meanings) and passes it through the existing [Planning](../../../../glossary.md#term-planning) query boundary. No authoritative contract,
[dependency direction](../../../../glossary.md#java-platform-and-build-tools), public [API](../../../../glossary.md#term-application-programming-interface-api), or [ADR](../../../../glossary.md#term-architecture-decision-record-adr) changes. Stop rather than edit architecture if
implementation evidence contradicts this conclusion.

## Implementation prompt

```text
Work in /Users/phujka/IdeaProjects/Synaptik on Compiler task 0006B6. Do not use GSD, commit, or
push. Use a separate clean implementation context. Read AGENTS.md, ARCHITECTURE.md, the current
architecture plan, planning guide, roadmap, Compiler and Engine master plans, this task, Engine
0008, completed Compiler 0006B/0006B3/0006B4/0006B5, documentation rules/profiles, and every
directly relevant Model/Compiler/Planning/CPU/Engine/integration source and test. Preserve the
dirty worktree.

Implement exactly the twelve-path allowlist. Add the package-private final convolution logical-
layout closure after optimization and published-constant closure. Close only fully static
unresolved Conv2d/Conv3d outputs to LayoutDescriptor.contiguous(shape), then propagate only the
direct newly closed Conv2d -> SQUEEZE(axis=2) result through the existing checked view rule. Keep
dynamic Shapes unresolved. Preserve exact IDs, topology, operation instances, values, constants,
bindings, constraints, phases, derivatives, publications, optimization results, and order. Add no
fake Conv1d operation, general inference mode, physical fact, backend predicate, downstream
production change, dependency, architecture edit, or work on 0006C.

Add the focused negative/preservation/overflow/query tests and the three exact public Engine
numerical fixtures specified by the task. Use only public Tensor/Engine typed binding and compute
in integration tests; do not forge inward artifacts or import CPU internals. In the existing
`Conv3dCompilerTest.passesExactPublicationDiagnosticsAndPlanningQueryWithoutProviderClaim`, update
only the stale query-output expectation to the final compiled Conv3d node output descriptor;
retain the unresolved Model producer contract and all other assertions. Run the specified focused/
module/integration/Javadoc/scope gates once. Then hand stable evidence to a distinct clean
documentation-focused context. Mark 0006B6 Complete and return Engine 0008 to Ready only after all
implementation, public fixtures, documentation, and status gates pass.
```

## Local decisions

- Use a dedicated post-validation final rewrite, not controlled inference, because current
  inference must continue to validate unresolved Model construction exactly.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [inference glossary](../../../../glossary.md#term-inference); [Model glossary](../../../../glossary.md#term-model))</span>
- Run after 0006B5 closure so one ordered final-descriptor pipeline feeds every downstream
  artifact; the predicates do not overlap.<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Close a fresh convolution result from its fully static output Shape alone. Input capability
  remains independently checked by Planning/backend queries.<span data-glossary-navigation="true"> ([convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [Shape glossary](../../../../glossary.md#shape); [Planning glossary](../../../../glossary.md#term-planning); [backend glossary](../../../../glossary.md#backend))</span>
- Propagate only direct Conv2d-to-height-squeeze geometry needed by the ordinary public Conv1d
  result. The pass never classifies a sequence as Conv1d.<span data-glossary-navigation="true"> ([Conv2d glossary](../../../../glossary.md#convolution--conv2d))</span>
- Keep [CPU](../../../../glossary.md#term-cpu) rejection of unresolved/dynamic [descriptors](../../../../glossary.md#term-descriptor) observable and unchanged.
- Add public cross-module numerical evidence in integration tests while keeping all production
  ownership in Compiler.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Compiler glossary](../../../../glossary.md#term-compiler))</span>

## Known limitations

Dynamic and partially dynamic [convolution](../../../../glossary.md#term-channels-first-convolution-layers) results remain unresolved and therefore are not
admitted by the current [CPU](../../../../glossary.md#term-cpu) provider. [Layout](../../../../glossary.md#layout) propagation beyond the exact direct [axis](../../../../glossary.md#term-axis)-two squeeze
is not supplied. [Conv3d](../../../../glossary.md#conv3d--ncdhw) remains forward-only. This task does not add mixed-[backend](../../../../glossary.md#backend) execution,
physical-layout selection, or [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) integration.

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation context `01a0b420-4b38-7533-995c-422d455cc66e` completed the Java and test work.
  Replanning context `01a0b42a-6c55-7560-9a76-6ee6d0682362` amended the task in [commit](../../../../glossary.md#term-commit)
  `f5ca109a` after the initial full Compiler run exposed one stale Conv3d Planning-query
  expectation; the replan added that existing test as the twelfth allowlisted path.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [run glossary](../../../../glossary.md#term-run); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [Planning glossary](../../../../glossary.md#term-planning))</span>
- Reused implementation evidence because this documentation pass changed Javadoc and Markdown
  only: the focused Compiler selection passed 42 tests (8 closure, 30 <span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`GraphCompiler`, and 4
  `Conv3dCompiler`); the public Engine convolution integration selection passed 3 tests; the full
  Compiler suite passed 270 tests; and the full integration suite passed 13 tests, all without
  failures, errors, or skips. No executable Java behavior changed after those runs.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers))</span>
- Documentation-focused context `01a0b435-21ee-7eb3-a391-67c0952ea7ee` independently read the architecture
  contract, planning and documentation rules/profiles, both affected module plans, both task
  specifications, the Compile API, relevant glossary entries, every changed Java/test file, and
  the actual diff. It finalized both production Javadocs and the Compile API under the General,
  API/Javadoc, and Planning profiles.<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract); [module glossary](../../../../glossary.md#term-module); [Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- `./gradlew :modules:compiler:javadoc` passed after the final Javadoc edits: 7 actionable tasks,
  2 executed and 5 up-to-date.
- Final documentation validation checked local Markdown targets and anchors, unique headings,
  balanced fences, LF line endings, terminal newlines, trailing whitespace, synchronized task/
  master/roadmap status and ordering, package/public import boundaries, exact twelve-path dirty
  scope, and an empty staging area. <span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` passed.
- Architecture and ADR impact: none. The change uses Compiler's existing ownership of final
  logical descriptors and the existing Planning query handoff; no module boundary, dependency
  direction, or architectural decision changed.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [descriptors glossary](../../../../glossary.md#term-descriptor); [handoff glossary](../../../../glossary.md#term-handoff); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools))</span>
- Glossary impact: none. Existing entries already define Conv1d composition, Conv2d, Conv3d,
  logical <span data-glossary-navigation="true"> ([Conv1d composition glossary](../../../../glossary.md#conv1d-composition); [Conv2d glossary](../../../../glossary.md#convolution--conv2d))</span>`TensorDescriptor` layout state, views, and operation capability queries; the new pass
  changes no reusable term or definition.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [views glossary](../../../../glossary.md#term-view-meanings); [operation glossary](../../../../glossary.md#operation))</span>
- Other documentation/API impact: no Tensor, Public, Runtime, Training, backend, or user-guide
  contract changed. The Compile API is the sole explanatory surface affected because it owns the
  ordering and boundary of final graph descriptors.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [graph glossary](../../../../glossary.md#graph))</span>
- Test/build impact outside the allowlist: none. No backend-conformance or architecture test was
  needed because backend semantics and dependency rules are unchanged; no Gradle file, generated
  code, production module outside Compiler, or other module contract changed.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [architecture test glossary](../../../../glossary.md#term-architecture-test); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle))</span>

## Implementation notes

- Added package-private `ConvolutionLogicalLayoutClosure` and invoked it in both graph-stage
  branches after optimization and published-constant closure.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [constant glossary](../../../../glossary.md#term-constant))</span>
- Fully static unresolved final Conv2d/Conv3d outputs close to canonical contiguous logical
  layouts. Only the direct unresolved <span data-glossary-navigation="true"> ([Conv2d glossary](../../../../glossary.md#convolution--conv2d); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [canonical glossary](../../../../glossary.md#term-canonical-form); [contiguous glossary](../../../../glossary.md#term-contiguous); [layouts glossary](../../../../glossary.md#layout))</span>`SQUEEZE(axis=2)` whose Conv2d input was newly closed in the
  same invocation receives the derived view layout. Dynamic, partially dynamic, already resolved,
  and unrelated descriptors remain unchanged.<span data-glossary-navigation="true"> ([view glossary](../../../../glossary.md#term-view-meanings); [descriptors glossary](../../../../glossary.md#term-descriptor))</span>
- The implementation preserves graph topology and identities, constants, bindable Tensor IDs,
  constraints, phases, derivative metadata, publications, optimization results, and order.
  Planning/publication consume the resulting final descriptors; no physical or backend behavior
  moved into Compiler.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [Tensor glossary](../../../../glossary.md#tensor); [derivative glossary](../../../../glossary.md#term-derivative); [metadata glossary](../../../../glossary.md#term-metadata); [Planning glossary](../../../../glossary.md#term-planning); [publication glossary](../../../../glossary.md#term-publication); [backend glossary](../../../../glossary.md#backend); [Compiler glossary](../../../../glossary.md#term-compiler))</span>
- Added the exact public NCW Conv1d, grouped NCHW Conv2d, and grouped NCDHW Conv3d Engine fixtures
  through reusable typed binding and one-shot compute. Updated the pre-existing Conv3d query test
  only to compare against the final compiled descriptor while retaining unresolved Model output.<span data-glossary-navigation="true"> ([NCW glossary](../../../../glossary.md#term-ncw); [NCHW glossary](../../../../glossary.md#term-nchw); [NCDHW glossary](../../../../glossary.md#conv3d--ncdhw); [Engine glossary](../../../../glossary.md#term-engine); [binding glossary](../../../../glossary.md#term-binding); [Model glossary](../../../../glossary.md#term-model))</span>

## Completion summary

- Completed changes: implemented bounded final convolution logical-layout closure, integrated it
  into both compile branches, added focused preservation/query coverage and three public Engine
  numerical fixtures, finalized Javadocs and the Compile API, and synchronized Compiler/Engine/
  roadmap status.<span data-glossary-navigation="true"> ([convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [layout glossary](../../../../glossary.md#layout); [compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Files changed or created: `ConvolutionLogicalLayoutClosure.java`, `GraphCompiler.java`,
  `ConvolutionLogicalLayoutClosureTest.java`, `GraphCompilerTest.java`, `Conv3dCompilerTest.java`,
  `EngineConvolutionIntegrationTest.java`, `docs/api/compile-api.md`, this task,
  `docs/planning/modules/compiler/master-plan.md`, `docs/planning/modules/engine/master-plan.md`,
  `docs/planning/modules/engine/tasks/0008-engine-lifecycle-capability-checkpoint.md`, and
  `docs/planning/roadmap.md`.
- Tests and validation: reused the stable 42-test focused Compiler, 3-test focused public Engine,
  270-test full Compiler, and 13-test full integration results; final Compiler Javadoc and all
  requested Markdown, scope, status/order, import, staging, and whitespace gates passed.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: clean context `01a0b435-21ee-7eb3-a391-67c0952ea7ee` completed the independent
  General/API-Javadoc/Planning review and changed no executable semantics or tests.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: Compile API now documents exact ordering, eligibility, preservation,
  failure behavior, final-descriptor consumption, and logical-only boundaries.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- [Javadoc](../../../../glossary.md#term-javadoc) review: both affected production contracts were finalized; [Compiler](../../../../glossary.md#term-compiler) Javadoc passed.
- Glossary impact: no change because all affected terms and distinctions already exist.
- Unresolved issues: None.
- Follow-up required: execute the next planned task, [Engine](../../../../glossary.md#term-engine) 0008; Compiler 0006C remains Draft.

Status: Complete
