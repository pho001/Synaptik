# Logical memory and compile outputs

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/CompileArtifacts.java)

Logical memory planning records which graph value is produced, consumed or required as an
output. Physical memory planning must later determine storage size and arrangement. A logical
value ID names a computation result, not a buffer address; a logical preservation requirement
therefore does not allocate storage or decide which physical copy will be published.

<a id="current-logical-memory-recipes"></a>
## Current logical-memory recipes

`io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement` is current public
compile-time data with exactly these ordered components:

```java
ValueId valueId
TensorDescriptor descriptor
Optional<PlannedPartition> producerPartition
List<PlannedPartition> consumerPartitions
boolean graphOutput
```

`producerPartition` is empty for a graph input in a generated plan. Each generated consumer list
contains a supplied partition at most once and follows supplied partition order, regardless of
repeated input positions or multiple consuming nodes inside one partition. `graphOutput` is true
exactly when an equal value identity occurs in `CompiledGraphModel.outputs()`. The generated
requirement retains the exact graph value identity and descriptor references plus exact supplied
partition-element references.

`LogicalMemoryPlan` contains one immutable ordered `List<LogicalMemoryRequirement>`. Its public
constructor permits an empty standalone plan, rejects null elements and duplicate equal
`ValueId` values, snapshots list membership, and retains exact requirement references. The public
requirement constructor likewise snapshots consumer membership and rejects null or equal
duplicate partitions. Directly constructed DTOs have no owning graph and therefore do not prove
that caller-supplied producer, consumer, or output facts are graph-valid.

Public stateless `LogicalMemoryPlanning.plan(...)` accepts one `CompiledGraphModel` and one
ordered `List<PlannedPartition>`. It completes all partition null, graph-membership, coverage,
graph-order, and adjacent-owner maximality checks before constructing a requirement. It then emits
exactly one requirement for every `graph.values()` entry in that encounter order. Producer facts
come from node outputs; consumer facts come from node inputs. Repeated inputs, fan-out, merges,
unused graph inputs, unused produced values, multi-output nodes, forward/backward uses, and
zero-node pass-through graphs require no synthetic node or role.

The primitive facts describe overlapping logical roles: graph input, partition input, partition
output, same-owner or cross-owner boundary, graph-output preservation, and partition-internal
value. The records do not store another role enum. They retain `TensorDescriptor` so static,
dynamic, and expression Shapes remain representable without calculating an element or byte
count.

This is logical planning only. The derivation accepts no `ForwardPublicationBinding`,
`GradientPublicationBinding`, or `TensorId` and creates no `PublicationPlan`. It selects no alias,
copy, transfer, lifetime, physical slot,
allocation, device, route, kernel, schedule, executable, or runtime residency. The current
package-private compiler invokes it after partition generation and retains the exact plan in
`CompileArtifacts`.

<a id="current-expression-input-and-compile-output"></a>
## Current expression input and compile output

Ordinary compilation receives an ordered Tensor-output list through the current Engine:

```java
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
- `GraphRngState` is an implemented opaque model expression value rather than a public numerical
  `Tensor` output. Current dropout places its private state Tensor at producer input one and wraps
  producer output two as the next state. Package-private structural capture preserves the
  reachable initial-state producer, state edge, and every dropout output slot. Direct state
  boundary selection, serialization, value rewriting, lifetime policy, and public compilation
  remain planned. Current internal canonicalization preserves every state slot, exact CSE merges
  only complete equal occurrences, and DCE retains all slots of a live dropout node.
- `CompileConfig` will aggregate the current compile mode, `BackendIntent`, graph-optimization,
  and `PartitionScoringConfig` values with any later justified planning-cost inputs as data. It will not
  contain live backend services. Its exact surface and defaults remain planned.
- `PublicationPlan` is current compiler-owned output-only context around ordered forward and
  gradient publication bindings and the exact final graph. It is separate from the model graph
  and adds no publication delivery policy or runtime behavior.
- `CompileConstantPlan` is current compiler-owned output-only classification of final graph inputs
  as caller-bindable or exact logical splats. Its typed bindable view associates each final
  caller-bindable `ValueId` with its originating immutable `TensorId`; `bindableInputs()` remains
  the compatibility `ValueId` projection.
- `CompileDiagnostics` is current compiler-owned output-only projection of successful deferred
  graph constraints without exposing the internal predicate vocabulary.
- `CompileArtifacts` currently combines mode, the exact final `CompiledGraphModel`, maximal
  backend-owned partitions, the derived logical memory plan, publication roles, constant/input
  roles, diagnostics, and derivative-order metadata. Its public record and nested output data are
  implemented, and the public constant-free `GraphCompilationPort` delegates to the
  package-private complete construction entry.
- Engine `CompiledGraph` is an owner-bound facade over immutable `CompileArtifacts`, not the same
  object as `CompiledGraphModel`. It exposes only final caller-bindable Tensor identity and
  descriptor metadata.

The compile artifacts deliberately contain no device buffers, backend executable objects, runtime
residency, prepared schedules, or mutable run state.


---

[Previous topic](planning.md) · [Next topic](lifecycle.md)
