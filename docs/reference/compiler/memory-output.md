# Logical memory and [compile](../../glossary.md#compile) outputs

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/CompileArtifacts.java)

Logical memory [planning](../../glossary.md#term-planning) records which [graph value](../../glossary.md#graph-value) is produced, consumed or required as an
output. Physical memory planning must later determine storage size and arrangement. A logical
value ID names a computation result, not a [buffer](../../glossary.md#term-buffer) address; a logical preservation requirement
therefore does not allocate storage or decide which physical copy will be published.

## From a logical value to a physical slot

Compiler calls `LogicalMemoryPlanning.plan` with a verified graph and its ordered partitions.
Planning returns one immutable requirement per value: who produces it, who consumes it, and
whether it must survive as an output. A requirement can exist for a storage-free input or an
unresolved shape. Prepare later calls backend analysis, receives concrete byte requirements and
assigns Runtime slots; only backend creators allocate per-run storage. A logical `ValueId` is
therefore not an address or a cleanup handle. Compare [Runtime byte geometry](../runtime/memory.md)
and [Prepare assignment](../runtime/orchestration.md#current-prepare-orchestration-assignment-and-backend-finalization)
to follow that later boundary.

<a id="current-logical-memory-recipes"></a>
## Current logical-memory recipes

`io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement` is current public
[compile](../../glossary.md#compile)-time data with exactly these ordered components:

```java
// A requirement preserves logical use relationships; it does not allocate bytes.
ValueId valueId
TensorDescriptor descriptor
Optional<PlannedPartition> producerPartition
List<PlannedPartition> consumerPartitions
boolean graphOutput
```

`producerPartition` is empty for a [graph](../../glossary.md#graph) input in a generated plan. Each generated consumer list
contains a supplied [partition](../../glossary.md#partition) at most once and follows supplied partition order, regardless of
repeated input positions or multiple consuming [nodes](../../glossary.md#node) inside one partition. `graphOutput` is true
exactly when an equal value [identity](../../glossary.md#term-identity-meanings) occurs in `CompiledGraphModel.outputs()`. The generated
requirement retains the exact [graph value](../../glossary.md#graph-value) identity and [descriptor](../../glossary.md#term-descriptor) references plus exact supplied
partition-[element](../../glossary.md#term-element) references.

`LogicalMemoryPlan` contains one [immutable](../../glossary.md#term-immutable) ordered `List<LogicalMemoryRequirement>`. Its public
constructor permits an empty standalone plan, rejects null elements and duplicate equal
[`ValueId`](../../glossary.md#logical-valueid-versus-physical-memory-slot) values, snapshots list membership, and retains exact requirement references. The public
requirement constructor likewise snapshots consumer membership and rejects null or equal
duplicate partitions. Directly constructed DTOs have no owning graph and therefore do not prove
that caller-supplied [producer](../../glossary.md#term-producer), consumer, or output facts are graph-valid.

Public stateless `LogicalMemoryPlanning.plan(...)` accepts one [`CompiledGraphModel`](../../glossary.md#compiled-graph--compiledgraphmodel) and one
ordered `List<PlannedPartition>`. It completes all partition null, graph-membership, coverage,
graph-order, and adjacent-[owner](../../glossary.md#term-owner) maximality checks before constructing a requirement. It then emits
exactly one requirement for every `graph.values()` entry in that encounter order. Producer facts
come from node outputs; consumer facts come from node inputs. Repeated inputs, [fan-out](../../glossary.md#term-fan-out), merges,
unused graph inputs, unused produced values, multi-output nodes, forward/backward uses, and
zero-node pass-through graphs require no synthetic node or role.

The primitive facts describe overlapping logical roles: graph input, partition input, partition
output, same-owner or cross-owner boundary, graph-output preservation, and partition-internal
value. The records do not store another role enum. They retain `TensorDescriptor` so static,
dynamic, and expression [Shapes](../../glossary.md#shape) remain representable without calculating an element or byte
count.

This is logical [planning](../../glossary.md#term-planning) only. The derivation accepts no `ForwardPublicationBinding`,
`GradientPublicationBinding`, or [`TensorId`](../../glossary.md#tensorid) and creates no `PublicationPlan`. It selects no [alias](../../glossary.md#aliasing-and-arena),
copy, transfer, [lifetime](../../glossary.md#term-lifetime), physical slot,
allocation, [device](../../glossary.md#term-device), route, [kernel](../../glossary.md#kernel), schedule, executable, or [runtime](../../glossary.md#term-runtime) [residency](../../glossary.md#residency). The current
package-private [compiler](../../glossary.md#term-compiler) invokes it after partition generation and retains the exact plan in
`CompileArtifacts`.

<a id="current-expression-input-and-compile-output"></a>
## Current expression input and [compile](../../glossary.md#compile) output

Ordinary [compilation](../../glossary.md#compiler-engine-and-execution-stage-names) receives an ordered [Tensor](../../glossary.md#tensor)-output list through the current [Engine](../../glossary.md#term-engine):

```java
// Compilation reads the output description; stored input bytes are supplied at a later run.
CompiledGraph graph = engine.compile(List.of(output));
```

- `output` identifies a current public `Tensor` expression for the compiler to capture.
  Public Tensor state plus binary arithmetic, binary comparison, boolean logical, conditional
  selection, cast, unary numeric including exact GELU, fixed tanh-approximation GELU, and SiLU,
  floating-classification, scalar, numeric aggregate expression
  construction for sum, mean,
  product, minimum, and maximum, binding-aware exact-target-Shape sum, masked sum and mean
  construction, boolean aggregate expression
  construction for all and any, ordered multi-axis ordinary/log-sum-exp/statistical/norm
  construction, axis-only arg-min/arg-max construction, and shape-preserving cumulative sum/product
  scan, softmax/log-softmax, trailing-Shape layer- and RMS-normalization construction, explicit
  five-input batch-normalization inference and five-output training/statistic-transition
  construction, exact-shape mean-squared-error construction with explicit loss reduction, and
  dense- or index-target categorical-cross-entropy-with-logits construction with exact target-type
  dispatch and one exact typed ignore overload,
  plus
  first-class
  scaled-dot-product-attention construction with optional BOOL mask and scale/causal attributes,
  preserving the original one-output family and adding an explicit same-producer output/weights
  family, plus grouped NCHW Conv2d construction with optional bias and exact geometry/group
  attributes, plus Model-current grouped NCDHW Conv3d construction with current Compiler
  forward inference, validation and shared first-order differentiation, plus NCHW maximum- and
  average-pooling construction with operation-specific attributes and exact floor/ceil spatial
  expressions, plus Model-current NCDHW maximum- and average-pooling metadata with current
  forward-only Compiler inference and validation, plus static-resolved or
  dynamic-unresolved contiguous
  request construction plus conditional-view reshape, expand, permutation, and rank-two transpose
  construction, conditional-view expand-dimensions/squeeze construction, and general/single-axis
  signed-step slice plus one-occurrence flip construction, finite length-defined extraction across
  unresolved selected extents, functional signed slice update, exact target-relative symbolic
  placement, and exact target/prefix-Shape crop construction, plus conditional-view scalar-select
  construction and
  unresolved-layout Gather/Gather Elements and trailing-axis one-hot construction, plus
  unresolved-layout Gather-ND,
  functional Scatter Elements, Gather-compatible Scatter Add, and functional Scatter-ND
  construction, plus
  unresolved constant-pad and complete-pattern tile
  construction, plus ordered concat/stack and immutable repeated-SELECT unstack construction,
  plus public general-axis unfold/fold and direct- or typed-padding NCHW unfold/fold window-
  transform construction, plus
  explicit-state training-dropout construction with public output and next-state results and one
  non-public producer mask slot,
  are implemented;
  additional ordinary compile options,
  saved-statistic construction and gradient construction outside the closed support table above,
  optional
  softmax, layer-normalization, RMS-normalization, attention, or activation decomposition,
  redundant-cast, redundant-contiguous, reshape/expand/permutation/rank-edit, slice chains, slice
  updates, crops, and selects beyond the current exact inference/preflight/gradient matrix,
  axis-gather/one-hot/Gather-ND/axis-scatter/Scatter-ND/pad/tile/composition/window-transform
  canonicalization or decomposition,
  concrete dynamic binding and value-dependent select/index validation,
  legal
  convolution/pooling decomposition, maximum-pooling saved-index
  policy, average-pooling fixed-divisor preservation, layout materialization
  planning remain planned. Package-private identity-based forward-only and phase-aware combined
  capture, closed first-order autograd, binding-free verification inference, mandatory dense
  graph-local canonicalization, the guarded seven-rule exact arithmetic scan, the bounded
  logical-splat fold matrix, and exact one-shot whole-graph DCE/phase-local-CSE/DCE cleanup are
  current internal behavior.
  All other operation-specific rewrites listed above remain planned.
- [`GraphRngState`](../../glossary.md#graph-rng-state--graphrngstate) is an implemented [opaque](../../glossary.md#term-opaque) [model](../../glossary.md#term-model) expression value rather than a public numerical
  `Tensor` output. Current dropout places its private state Tensor at producer input one and wraps
  producer output two as the next state. Package-private structural capture preserves the
  reachable initial-state producer, state edge, and every dropout output slot. Direct state
  boundary selection, serialization, value rewriting, lifetime policy, and public compilation
  remain planned. Current internal canonicalization preserves every state slot, exact CSE merges
  only complete equal occurrences, and DCE retains all slots of a live dropout node.
- `CompileConfig` will aggregate the current [compile mode](../../glossary.md#compile-mode--compilemode), [`BackendIntent`](../../glossary.md#backend-intent--backendintent), graph-optimization,
  and [`PartitionScoringConfig`](../../glossary.md#partition-scoring-configuration--partitionscoringconfig) values with any later justified planning-cost inputs as data. It will not
  contain live backend services. Its exact surface and defaults remain planned.
- `PublicationPlan` is current compiler-owned output-only context around ordered forward and
  gradient publication bindings and the exact final graph. It is separate from the model graph
  and adds no publication delivery policy or runtime behavior.
- `CompileConstantPlan` is current compiler-owned output-only classification of final graph inputs
  as caller-bindable or exact logical splats. Its typed bindable view associates each final
  caller-bindable [`ValueId`](../../glossary.md#logical-valueid-versus-physical-memory-slot) with its originating [immutable](../../glossary.md#term-immutable) [`TensorId`](../../glossary.md#tensorid); `bindableInputs()` remains
  the compatibility `ValueId` projection.
- `CompileDiagnostics` is current compiler-owned output-only projection of successful deferred
  graph constraints without exposing the internal predicate vocabulary.
- `CompileArtifacts` currently combines mode, the exact final [`CompiledGraphModel`](../../glossary.md#compiled-graph--compiledgraphmodel), maximal
  backend-owned partitions, the derived logical memory plan, publication roles, constant/input
  roles, diagnostics, and derivative-order metadata. Its public record and nested output data are
  implemented, and the public constant-free `GraphCompilationPort` delegates to the
  package-private complete construction entry.
- Engine `CompiledGraph` is an [owner](../../glossary.md#term-owner)-bound [facade](../../glossary.md#term-facade) over immutable `CompileArtifacts`, not the same
  object as `CompiledGraphModel`. It exposes only final caller-bindable Tensor identity and
  descriptor metadata.

The [compile artifacts](../../glossary.md#compile-artifacts) deliberately contain no [device](../../glossary.md#term-device) [buffers](../../glossary.md#term-buffer), [backend](../../glossary.md#backend) executable objects, [runtime](../../glossary.md#term-runtime)
[residency](../../glossary.md#residency), prepared schedules, or [mutable](../../glossary.md#term-mutable) [run state](../../glossary.md#run-state--runstate).


---

[Previous topic](planning.md) · [Next topic](lifecycle.md)
