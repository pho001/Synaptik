# Compile API

## Purpose and implementation status

This reference separates the compile-time model values implemented today from the compiler and
engine APIs that remain planned. The repository does not yet provide a runnable graph compiler.

Compilation will answer two questions: what the computation means, and which backend identity owns
each planned region. It will not create physical buffers, choose concrete kernels, or construct
prepared executables.

## Current model contracts

The `io.github.pho001.synaptik.model.graph` package now provides compiler-neutral data that later
compiler work can produce and consume:

| Current contract | Meaning | Deliberate boundary |
|---|---|---|
| `CompiledGraphModel` | Immutable ordered graph values, topological nodes, declared input/output boundaries, and exact node phases | Structural graph state, not compiler passes, partitions, storage, or execution |
| `GraphPhase` | Exactly `FORWARD` or `BACKWARD` compile-time node classification | Not a compile mode, optimizer phase, or runtime schedule |
| `PublicationBinding` | Standalone `TensorId`-to-`ValueId` association | Not an owning publication plan and not a `CompiledGraphModel` component |

`CompiledGraphModel` validates structural closure when constructed. It snapshots its lists and
phase map, requires resolvable references and topological node order, enforces producer and phase
coverage rules, and stores no derived indexes. This validation does not capture an expression,
infer descriptors, transform a graph, perform autograd, plan backend ownership, or make the model
executable.

A `PublicationBinding` carries only two identities. A later compiler-owned `PublicationPlan` will
group bindings with their owning compilation context and publication policy. The binding itself
does not retain a public `Tensor`, gradient role, runtime target, storage, backend, or execution
state.

The public `Tensor` model is also current. Its seven binary arithmetic methods, six binary
comparison methods, three boolean logical methods, fifteen unary elementwise methods, and five
scalar arithmetic and clamp methods, plus one static conditional-selection method and one explicit
cast method, fifteen full/axis numeric aggregate methods, and six full/axis boolean aggregate
methods, two axis-removing masked aggregate methods, three axis-only `argMax` methods, and two
one-axis `cumSum` methods, plus one-axis `softmax`, `logSoftmax`, scalar `select`, four
tensor-index axis-gather methods, one primitive-array take convenience, and two Gather-ND methods,
plus four functional axis-scatter methods and three functional Scatter-ND methods, construct
storage-free expressions with immutable operation-and-input provenance. The
parameterless `contiguous` method
adds the same expression provenance for a canonical-layout request, and the two `reshape`
overloads add normalized target-shape expressions. Two `expand` overloads add directional
right-aligned target-shape expressions.
Arithmetic, unary, scalar, and conditional-selection results remain floating;
comparison and logical results are unresolved-layout `BOOL` descriptors with false gradient
eligibility. Logical AND and OR require exact BOOL inputs and derive a local broadcast shape;
logical NOT requires exact BOOL and retains the exact input shape. Scalar parameters remain exact
binary64 operation attributes rather than Tensor inputs. `Tensor.where` requires an exact BOOL
condition, promotes two floating branches, composes branch-first and condition-second local
broadcasts, propagates gradient eligibility from the branches only, and records exact ordered
condition/true-branch/false-branch provenance. It constructs no selected values or gradient rule.
`Tensor.cast` accepts every current source/target data-type pair, retains the exact input shape,
leaves layout unresolved, and retains a true gradient request only for floating-to-floating casts.
Every call remains a fresh explicit expression, including a same-type request, with typed target
attributes and exact one-input provenance.
`Tensor.sum`, `mean`, `prod`, reduction `min`, and reduction `max` accept floating inputs and
construct full, axis-removing, or retained-axis expressions. Full forms have canonical rank-zero
shape and use the canonical no-attributes singleton; axis forms normalize the caller axis and
store `AxisReductionAttrs`. They preserve exact input type and gradient eligibility, leave layout
unresolved, and record one-input provenance without aggregating or comparing values or defining
empty-domain, NaN, signed-zero, extrema-tie gradient, numerical, or executable behavior. Aggregate
`MIN`/`MAX` remain typed separately from the equally named two-input binary elementwise kinds.
The masked `Tensor.sum(axis, mask)` and `Tensor.mean(axis, mask)` forms require floating input and
an exact BOOL mask. They resolve an ordered injection from mask dimensions to input axes using
equal dimensions or mask-side singleton expansion, prefer mappings that include the reduction
axis, remove that axis from the result, and record `MaskedReductionAttrs` with ordered
`[input, mask]` provenance. Construction does not align storage, inspect values, select elements,
count true positions, compute a result, define gradients, or execute work.
`Tensor.all` and `Tensor.any` require exact BOOL input and construct full, axis-removing, or retained-axis
expressions with exact BOOL result type, false gradient eligibility, unresolved layout, and
one-input provenance. Aggregate `ALL`/`ANY` remain typed separately from elementwise `AND`/`OR`.
Construction does not inspect truth values or define empty-domain identities, compiler behavior,
backend support, or execution.
`Tensor.argMax` accepts floating or integral input and one positive or negative axis. Its
convenience forms explicitly use `FIRST_INDEX`, while the complete form retains an explicit
first- or last-index policy. Axis removal or retention follows the same structural Shape rules as
ordinary reductions, but every result is fixed unresolved-layout INT64 with false gradient
eligibility and one-input provenance. Construction does not compare values, select an index, or
define NaN, equality, empty-axis, gradient, compiler, backend, or execution behavior.
`Tensor.cumSum` accepts floating or integral input and one positive or negative axis. Its short
form explicitly selects inclusive forward traversal; its complete form retains exact exclusive
and reverse flags. Every result retains the exact input Shape, data type, and gradient eligibility,
leaves layout unresolved, and records `CUM_SUM` with exact one-input provenance. Construction does
not read or accumulate values, create a gradient rule, capture a graph, lower a backend operation,
or execute work.
`Tensor.softmax` and `Tensor.logSoftmax` accept floating input and one positive or negative axis.
Every result retains the exact input Shape, data type, and gradient eligibility, leaves layout
unresolved, and records the requested first-class SOFTMAX or LOG_SOFTMAX kind with exact one-input
provenance. Construction does not read values, calculate probabilities or logarithms, select a
numerical algorithm, define a gradient rule, capture or decompose a graph operation, lower a
backend operation, or execute work.
`Tensor.contiguous()` accepts every current data type and preserves the exact Shape, data type, and
gradient eligibility. It creates new canonical dense row-major, zero-offset layout geometry for a
fully static Shape and leaves a dynamic Shape unresolved. Every call is fresh, unlabeled, and
storage-free, records `CONTIGUOUS` with the canonical no-attributes singleton and exact one-input
provenance, and does not inspect input layout, storage, or values. Resolved result geometry does
not allocate or copy storage. Compiler capture, redundant-request canonicalization,
materialization policy, lowering, and execution remain planned.
`Tensor.reshape(long...)` accepts all current data types, normalizes an empty request or one
inferable `-1`, and rejects locally provable invalid counts. `Tensor.reshape(Shape)` retains an
exact normalized target and defers count equality when either Shape is dynamic. Both overloads
retain type and gradient eligibility, record exact RESHAPE/target-shape semantics with one-input
provenance, and stay unlabeled and storage-free. Only resolved contiguous input plus a static
target produces same-offset canonical view metadata; all other result layout remains unresolved.
This is current model expression construction, not compiler capture, graph-wide dynamic constraint
solving, reshape-chain canonicalization, materialization planning, backend alias/copy lowering, or
execution.
`Tensor.expand(long...)` treats every requested extent as a literal non-negative dimension, while
`Tensor.expand(Shape)` retains the exact target reference. Both overloads require target rank at
least input rank and accept aligned dimensions only when they are structurally equal or the input
is a static singleton; new leading target axes are valid. A fully static target and any resolved
input layout produce new same-offset view geometry with preserved aligned strides and zero strides
for leading or expanded-singleton axes. Dynamic target or unresolved input geometry stays
unresolved. Every result retains exact type and gradient eligibility, records exact
EXPAND/target-shape semantics with one-input provenance, and remains fresh, unlabeled, and
storage-free. This is current model construction, not value repetition, storage aliasing, dynamic
constraint solving, gradient behavior, compiler capture or canonicalization, materialization,
lowering, or execution.
`Tensor.permute(int...)` accepts every current data type, requires a complete output-to-input axis
mapping, normalizes each negative axis once, and reorders exact Dimension references. Any resolved
input layout produces a new same-offset view descriptor with exact reordered strides; unresolved
input layout remains unresolved. `Tensor.transpose()` requires rank two and uses the same PERMUTE
construction with normalized axes `[1, 0]`. Every result preserves type and gradient eligibility,
records exact normalized attributes and one-input provenance, and remains fresh, unlabeled, and
storage-free. This is current model expression construction, not graph capture, permutation
canonicalization, physical aliasing, materialization, lowering, or execution.
`Tensor.expandDims(int)` inserts one static singleton at a caller position normalized against the
result rank. `Tensor.squeeze(int)` instead normalizes an existing input axis and removes it only
when its dimension is statically known as one. Both preserve exact type and gradient eligibility,
retain exact unaffected Dimension references, record matching `AxisTransformAttrs` and one-input
provenance, and leave unresolved input geometry unresolved. Resolved input geometry produces one
new same-offset logical view descriptor with a deterministic checked stride inserted or the
selected stride removed. This is current model expression construction, not graph capture,
dynamic singleton constraint solving, inverse-pair canonicalization, physical aliasing,
materialization, gradient behavior, lowering, or execution.
`Tensor.slice(long[], long[], int[], long[])` clones four parallel request arrays, normalizes raw
axes and bounds against selected static dimensions, clamps bounds, derives a same-rank Shape, and
records normalized positive-step half-open `SliceAttrs`. Empty arrays and zero-extent results are
valid. Resolved non-empty dense, offset, strided, or broadcast input geometry produces checked
start-adjusted, step-multiplied logical view geometry; unresolved input and empty results remain
unresolved. `Tensor.sliceAxis(int, long, long)` delegates the same operation with one step-one
entry. Both forms preserve exact type and gradient eligibility, record exact one-input provenance,
and remain fresh, unlabeled, and storage-free. This is current model expression construction, not
graph capture, slice-chain or identity canonicalization, physical aliasing, gradient-scatter
construction, materialization, backend/ONNX lowering, or execution.
`Tensor.select(int, long)` normalizes one source axis and one scalar coordinate, removes the
selected Dimension, preserves every unaffected exact Dimension reference, and records normalized
`SelectAttrs` with exact one-input provenance. A static selected extent supplies immediate
negative-index normalization and bounds checks; a non-negative coordinate on a dynamic selected
extent remains representable with its upper bound deferred, while a negative coordinate is
rejected. Resolved input geometry with a non-empty result produces checked selected-stride removal
and offset advancement in one new logical view descriptor; unresolved input and empty results stay
unresolved. The fresh result preserves exact type and eligibility and has no label or storage.
This is current model expression construction, not value selection, physical aliasing, gradient
construction, graph capture or canonicalization, materialization, backend lowering, or execution.
`Tensor.gather`, `gatherAxis`, tensor-index `take`, and `takeAlongAxis` consume exact ordered
`[data, indices]` inputs with `INT32` or `INT64` indices. They normalize one data axis and apply
their distinct reduced-Shape, inserted-Shape, or same-rank aligned-Shape rules. Primitive
`take(int, int[])` snapshots one non-empty caller array, creates one independent dense rank-one
INT32 index Tensor through the current flat-import factory, and then delegates once to the same
tensor-index take path. Every fresh result retains data type and gradient eligibility, leaves
layout unresolved, and records exact two-input provenance; both take overloads use the same
`GATHER_AXIS` semantic identity. Primitive adaptation allocates the generated index Tensor before
axis validation and performs no rollback after a later failure. Construction interprets no index
values, checks no index-value bounds, and adds no gradient rule, graph capture, canonicalization,
materialization, lowering, or execution behavior.
`Tensor.gatherNd` consumes exact ordered `[data, indices]` inputs with `INT32` or `INT64` indices.
Its short form uses zero shared batch Dimensions; its complete form retains one normalized
non-negative batch count. Construction validates both ranks, structurally equal leading batch
Dimensions, and a statically known positive tuple depth from the final indices Dimension. It
derives the result as the indices prefix without tuple depth followed by the untouched data
suffix, including canonical scalar and exact retained Dimension references. Every result is fresh,
unlabeled, storage-free, and unresolved-layout, preserves data type and gradient eligibility, and
records `GATHER_ND` with exact `[data, indices]` provenance. It reads no index value, checks no
index-value bound, and adds no gradient, graph capture, compiler transformation, materialization,
lowering, or execution behavior.
`Tensor.scatterAdd`, `scatterAxisAdd`, and both `scatterElements` overloads consume exact ordered
`[data, indices, updates]` inputs. They require `INT32` or `INT64` indices and exact matching
data/update types. The two fixed-add forms require floating values; scatter-elements permits
`NONE` for every current type and arithmetic reductions for floating or integral values. Each
method normalizes one raw data axis, validates its reduced-rank, rank-changing, or same-rank Shape
relationship, and returns a fresh result with the exact data Shape/type, data/update gradient-
eligibility OR, unresolved layout, and exact three-input provenance. Construction reads no values,
checks no index bound or duplicate target, mutates no input, and performs no write or reduction.
It adds no gradient rule, graph capture, compiler transformation, materialization, lowering,
backend behavior, or execution behavior.
The three `Tensor.scatterNd` overloads consume exact ordered `[data, indices, updates]` inputs with
`INT32` or `INT64` indices and exact matching data/update types. Their defaults select
`ScatterReduction.NONE` and zero shared batch Dimensions; complete construction retains the exact
supplied reduction and normalized non-negative batch count. Construction validates reduction
eligibility, both ranks, structurally equal leading batch Dimensions, static positive tuple depth
from the final indices Dimension, and the exact indices-prefix-plus-data-suffix updates Shape.
Every result is fresh, unlabeled, storage-free, and unresolved-layout, retains the exact data
Shape/type, combines data/update gradient eligibility, and records `SCATTER_ND` with exact
three-input provenance. It reads no index or update value, checks no index bound or duplicate
target, mutates no input, performs no write or reduction, and adds no gradient, graph capture,
compiler transformation, materialization, lowering, backend behavior, or execution behavior.
Static `Tensor.concat(int, Tensor...)` and `Tensor.stack(int, Tensor...)` snapshot ordered non-empty
inputs, normalize an existing or inserted axis, enforce exact type and operation-specific Shape
rules, and create fresh unresolved-layout results with eligibility OR and exact ordered provenance.
Instance `Tensor.unstack(int)` requires a static `int`-sized selected extent, removes that axis,
and returns an immutable ordered List whose fresh results have individually indexed UNSTACK
provenance over the same input. A zero extent returns no result or ID, and the output indices do
not establish a shared producer occurrence or graph output-slot contract. This is current model
expression construction, not compiler capture, decomposition, grouping, backward construction,
materialization, backend lowering, ONNX mapping, or execution.
`Tensor.unfold`, `Tensor.foldAxis`, `Tensor.unfold2d`, and `Tensor.fold2d` construct current
storage-free window-transform expressions. They normalize and validate axes against the relevant
source or target rank, use checked static Shape arithmetic, preserve the input data type and
gradient-eligibility flag, leave layout unresolved, and record exact one-input provenance. They
describe materialized windows or overlap-summing scatter-add; they do not read values or provide
compiler capture, canonicalization, gradient generation, lowering, or execution.
That origin metadata gives a future compiler an expression to traverse, but no current API
captures it into `CompiledGraphModel`, performs inference or optimization, or produces compile
artifacts.

## Current expression input and planned compiler output

Conceptually, compilation will receive a requested tensor output and declarative `CompileConfig`:

```java
// Conceptual API; not currently runnable.
CompiledGraph graph = CompiledGraph.compile(output, CompileConfig.auto());
```

- `output` will identify a current public `Tensor` expression for the future compiler to capture.
  Public Tensor state plus binary arithmetic, binary comparison, boolean logical, conditional
  selection, cast, unary, scalar, numeric aggregate expression construction for sum, mean,
  product, minimum, and maximum, masked sum and mean construction, boolean aggregate expression
  construction for all and any, axis-only arg-max construction, and shape-preserving cumulative-
  sum and softmax/log-softmax construction, plus static-resolved or dynamic-unresolved contiguous
  request construction plus conditional-view reshape, expand, permutation, and rank-two transpose
  construction, conditional-view expand-dimensions/squeeze construction, and general/single-axis
  positive-step slice construction, plus conditional-view scalar-select construction and
  unresolved-layout axis-gather construction, including copied primitive take input adaptation,
  plus unresolved-layout Gather-ND, functional axis-scatter, and functional Scatter-ND
  construction, plus
  unresolved constant-pad and complete-pattern tile
  construction, plus ordered concat/stack and immutable individually indexed unstack construction,
  plus general-axis and NCHW unfold/fold window-transform construction,
  are implemented;
  the compiler entry point, traversal, capture,
  scan/reduction/normalization inference and canonicalization, optional softmax decomposition,
  redundant-cast, redundant-contiguous, reshape/expand/permutation/rank-edit/slice-chain/select/
  axis-gather/Gather-ND/axis-scatter/Scatter-ND/pad/tile/composition/window-transform
  canonicalization or decomposition,
  grouped
  producer design,
  compiler-generated `FOLD_AXIS` construction, deferred
  dynamic reshape count validation, expand compatibility constraints, and dynamic select upper-
  bound validation placement, layout materialization
  planning, and conversion into graph values and nodes remain planned.
- `CompileConfig` will describe compile mode, backend intent, optimization, scoring, and
  publication policy as data. It will not contain live backend services.
- `PublicationPlan` will be compiler-owned context around publication bindings. It is planned and
  is separate from the current model graph.
- `CompileArtifacts` will combine a `CompiledGraphModel`, planned partitions, a logical memory
  plan, a `PublicationPlan`, and diagnostics. It is planned and will remain non-executable.
- `CompiledGraph` will be an engine facade over immutable `CompileArtifacts`, not the same object
  as the current `CompiledGraphModel`.

The planned artifacts deliberately contain no device buffers, backend executable objects, runtime
residency, prepared schedules, or mutable run state.

## Planned lifecycle and failures

```text
expression -> capture -> inference and validation -> optimization
           -> optional autograd -> backend ownership -> logical plans
           -> CompileArtifacts
```

Compilation is expected to reject invalid shapes, data types, operations, graph structure, or
unsatisfied backend capabilities before preparation. Exact exception types and callable signatures
remain to be specified by compiler and engine tasks; callers must not code against invented
exceptions from this conceptual page.

## Example interpretation

If a future graph contains a matrix multiplication followed by a small elementwise operation,
capability analysis may find both CPU and Metal valid. Backend-neutral scoring may assign both
nodes to Metal to avoid a transfer boundary. The artifact records only `owner = Metal`; it does
not record MPSGraph or a custom Metal kernel. Metal prepare makes that later choice.

This scenario is conceptual. The current graph DTOs can represent and structurally validate node
relationships, but they cannot run this compilation or select ownership.

## Related contracts

- [Current Tensor and graph-model API](tensor-api.md)
- [Lifecycle](../architecture/lifecycle.md)
- [Partition scoring](../architecture/partition-scoring.md)
- [Compiling graphs user guide](../user-guide/compiling-graphs.md)
- [Roadmap](../planning/roadmap.md)
