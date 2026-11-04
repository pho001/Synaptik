# Task 0018O: Indexing Taxonomy and Unstack [Normalization](../../../../glossary.md#term-normalization)

**Editorial reading key (current context, not a status update).**

This record concerns a historical model-only vocabulary migration, not present execution coverage.
Here, the caller invokes Tensor indexing methods; model helpers derive descriptors and
producer/provenance metadata from the receiver, index Tensor descriptors, and axis attributes—not
index values. Gather inserts the complete indices Shape; Gather Elements uses aligned indices and
retains their Shape. Follow [axis-gather
expressions](../../../../reference/tensor/gather.md#axis-gather-expressions) for current
distinctions, including conveniences outside this task. Model semantics defines meaning; execution
providers realize supported routes rather than redefine it. Semantic responsibility does not imply
storage ownership: these storage-free results neither execute nor own result buffers.

Unstack means ordered independent scalar selections: Shape `[2, 3, 4]` on axis `1` gives three
Shape-`[2, 4]` expressions. A producer identifies one operation occurrence; the provenance output
index identifies its result. Each selection therefore has its own producer and output index zero,
unlike the shared producer used in the record’s prospective top-K comparison. Follow [tensor
composition expressions](../../../../reference/tensor/composition.md#tensor-composition-expressions)
for current count-validation and logical-view rules; a resolved view descriptor does not promise
physical aliasing.

## Status

Complete

## Goal

Replace the provisional indexing vocabulary with one stable, interoperability-oriented public
taxonomy and express unstack as the convenience composition it actually is:

```text
scalar coordinate                 -> SELECT
strided range                    -> SLICE
replace one axis by indices      -> GATHER
same-rank aligned indices        -> GATHER_ELEMENTS
coordinate tuples                -> GATHER_ND
same-rank functional updates     -> SCATTER_ELEMENTS
coordinate-tuple updates         -> SCATTER_ND
unstack(axis)                    -> ordered SELECT(axis, 0..extent-1)
```

The task atomically removes the provisional duplicate names and specialized public [adjoint](../../../../glossary.md#term-adjoint) kinds:
`GATHER_AXIS`, `TAKE_ALONG_AXIS`, reduced-[rank](../../../../glossary.md#term-rank) [`GATHER`](../../../../glossary.md#gather), `SCATTER_ADD`,
`SCATTER_AXIS_ADD`, first-class `UNSTACK`, every public `take` spelling, and their dedicated
helpers or attributes. It retains the current [canonical](../../../../glossary.md#term-canonical-form) [gather-ND](../../../../glossary.md#gather-nd), scatter-[elements](../../../../glossary.md#term-element), [scatter-ND](../../../../glossary.md#scatter-nd),
select, and [slice](../../../../glossary.md#slice) behavior and gives the two [axis](../../../../glossary.md#term-axis)-gather [shapes](../../../../glossary.md#shape) their final names.

This task changes only [backend](../../../../glossary.md#backend)-independent [model](../../../../glossary.md#term-model) semantics, public [Tensor](../../../../glossary.md#tensor) expression construction,
tests, Javadocs, and explanatory documentation. It does not add value execution, index-bound
inspection, [gradients](../../../../glossary.md#term-gradient), [graph capture](../../../../glossary.md#capture-graph-ports-and-reconstruction), [compiler](../../../../glossary.md#term-compiler) rules, [lowering](../../../../glossary.md#lowering), [kernels](../../../../glossary.md#kernel), or backend behavior.

## [Mental model](../../../../glossary.md#term-mental-model) and examples

For data [Shape](../../../../glossary.md#shape) `[2, 3, 4]`, indices Shape `[5, 6]`, and [axis](../../../../glossary.md#term-axis) `1`, [canonical](../../../../glossary.md#term-canonical-form) [gather](../../../../glossary.md#gather) replaces the
selected data [dimension](../../../../glossary.md#dimension) with the complete indices Shape:

```text
GATHER result = data[:axis] + indices[:] + data[axis + 1:]
              = [2] + [5, 6] + [4]
              = [2, 5, 6, 4]
```

[Scalar](../../../../glossary.md#term-scalar) indices therefore produce `[2, 4]`. The exact existing `GATHER_AXIS` implementation has
this contract and becomes `GATHER`; this is a rename of that behavior, not adoption of the current
reduced-[rank](../../../../glossary.md#term-rank) `GATHER` behavior.

For data Shape `[2, 3, 4]`, indices Shape `[2, 7, 4]`, and axis `1`, gather-[elements](../../../../glossary.md#term-element) uses one
index at every aligned non-axis position:

```text
rank(indices) == rank(data)
indices[axis != selected] == data[axis != selected]
GATHER_ELEMENTS result == indices Shape == [2, 7, 4]
```

The exact existing `TAKE_ALONG_AXIS` behavior becomes `GATHER_ELEMENTS` and the public spelling is
only `gatherElements`. There is no `takeAlongAxis` [alias](../../../../glossary.md#aliasing-and-arena).

The current reduced-rank `GATHER` accepts indices Shape `remove(data, axis)` and returns that same
rank-minus-one Shape. It can be expressed as [singleton](../../../../glossary.md#term-singleton)-axis insertion, gather-elements, and
squeeze. No current public use case requires it, so both its semantic kind and public method are
removed rather than renamed.

Unstack is an ordered [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) of already selected scalar-[coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) expressions. For input
`[2, 3, 4]` and axis `1`:

```java
input.unstack(1)
```

is semantically equivalent to:

```java
List.of(input.select(1, 0), input.select(1, 1), input.select(1, 2))
```

Each result is an independent `SELECT` [occurrence](../../../../glossary.md#term-occurrence) with its own [producer](../../../../glossary.md#term-producer) and output index zero.
This differs from a genuine multi-output [operation](../../../../glossary.md#operation) such as planned top-K: top-K values and indices
are two [descriptors](../../../../glossary.md#term-descriptor) emitted by one semantically indivisible selection occurrence and therefore
share one producer with distinct output indices. Unstack results need no grouping [identity](../../../../glossary.md#term-identity-meanings) because
each scalar selection is complete and independently reproducible.

## Current problems

- `AxisGatherKind.GATHER` currently means a nonstandard reduced-[rank](../../../../glossary.md#term-rank) [operation](../../../../glossary.md#operation), while
  `GATHER_AXIS` has the [canonical](../../../../glossary.md#term-canonical-form) [Gather](../../../../glossary.md#gather) [Shape](../../../../glossary.md#shape) formula.
- `TAKE_ALONG_AXIS`, `takeAlongAxis`, [tensor](../../../../glossary.md#tensor)-index `take`, and primitive-array `take` assign local
  meanings to vocabulary that differs across NumPy, JAX, and PyTorch.
- `SCATTER_ADD` and `SCATTER_AXIS_ADD` expose fixed-add shapes principally useful as adjoints of
  the provisional gathers. They duplicate future compiler-generated semantics in the public
  baseline.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler))</span>
- `UNSTACK` and `UnstackOutputAttrs` pretend repeated scalar selections are a distinct operation
  family and encode an output index inside operation attributes even though every output already
  has its own independent producer.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [producer glossary](../../../../glossary.md#term-producer))</span>
- The current unstack implementation forces every result layout unresolved instead of reusing
  scalar select's exact resolved-view rule when geometry permits it.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [scalar select glossary](../../../../glossary.md#scalar-select); [view glossary](../../../../glossary.md#term-view-meanings))</span>
- Completed tasks and tests accurately preserve the provisional history, but the live kinds,
  public methods, API reference, glossary, and capability baseline must now converge atomically.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## [Scope](../../../../glossary.md#term-scope)

- Make `AxisGatherKind` contain exactly [`GATHER`](../../../../glossary.md#gather) and `GATHER_ELEMENTS`, in that order.
- Assign current `GATHER_AXIS` semantics and [Shape](../../../../glossary.md#shape) derivation to final `GATHER`.
- Assign current `TAKE_ALONG_AXIS` semantics and Shape derivation to final `GATHER_ELEMENTS`.
- Remove the current reduced-[rank](../../../../glossary.md#term-rank) gather kind, expression path, and public method.
- Expose exactly `gather(Tensor indices, int axis)` and
  `gatherElements(Tensor indices, int axis)` for [axis](../../../../glossary.md#term-axis)-index [Tensor](../../../../glossary.md#tensor) expressions.
- Remove `gatherAxis`, both public `take` overloads, and `takeAlongAxis`; add no aliases or
  deprecated bridges.<span data-glossary-navigation="true"> ([aliases glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Delete the primitive-array take helper and its focused test.
- Keep `GATHER_ND`, both `gatherNd` overloads, their attributes, [validation](../../../../glossary.md#term-validation), and formulas unchanged.
- Make `AxisScatterKind` contain exactly `SCATTER_ELEMENTS`.
- Remove public `scatterAdd` and `scatterAxisAdd`, their fixed-add semantic kinds, their
  `IndexAxisAttrs` pairing, and their helper paths.
- Keep both `scatterElements` overloads, every `ScatterReduction`, `SCATTER_ND`, all three
  `scatterNd` overloads, and their current validation and Shape formulas unchanged.
- Make `TensorCompositionKind` contain exactly `CONCAT` and `STACK`, in that order.
- Delete `UnstackOutputAttrs` and remove the `UNSTACK` [operation signature](../../../../glossary.md#operation-signature) branch.
- Retain public `List<Tensor> unstack(int axis)` as repeated [scalar](../../../../glossary.md#term-scalar) `SELECT` construction.
- Preserve unstack's current upfront input/axis/static-count validation, zero-extent behavior,
  ordering, immutable return list, and identifier-exhaustion side effects.<span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- Give every non-empty unstack output exactly scalar-select semantics, attributes, layout,
  producer, provenance, and output index zero.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>
- Update semantic, expression, signature, provenance-facing, reflection, and public-inventory
  tests to prove both the retained surface and the absence of removed vocabulary.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature))</span>
- Finalize all affected Javadocs, Tensor API, glossary, capability baseline, task evidence, master
  plan, and roadmap in the required independent documentation pass.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- changing [scalar](../../../../glossary.md#term-scalar) `SELECT` or positive-step [`SLICE`](../../../../glossary.md#slice) semantics
- negative-step slicing, flip, unfold, `FOLD_AXIS`, fold2d, split, or chunk; task 0018R owns the
  slice/window cleanup
- changing Gather-ND or Scatter-ND tuple-depth, batch-dimension, reduction, type, Shape, or
  validation contracts<span data-glossary-navigation="true"> ([Gather-ND glossary](../../../../glossary.md#gather-nd); [Scatter-ND glossary](../../../../glossary.md#scatter-nd); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [dimension glossary](../../../../glossary.md#dimension); [Shape glossary](../../../../glossary.md#shape); [validation glossary](../../../../glossary.md#term-validation))</span>
- changing `SCATTER_ELEMENTS` reductions, duplicate-[target](../../../../glossary.md#term-target) policy, or supported data types
- retaining a public reduced-[rank](../../../../glossary.md#term-rank) [gather](../../../../glossary.md#gather)/scatter convenience under a new name
- retaining `take`, `takeAlongAxis`, `gatherAxis`, `scatterAdd`, or `scatterAxisAdd` as an alias,
  deprecated member, compatibility adapter, or hidden duplicate path<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- adding primitive-array, list, host-value, mask, embedding, one-hot, or index-select convenience
  APIs<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [mask glossary](../../../../glossary.md#term-boolean-mask); [embedding glossary](../../../../glossary.md#embedding))</span>
- adding first-class multi-output operations, top-K, sorting, or modifying the shared
  [`TensorProducer`](../../../../glossary.md#producer--tensorproducer)/`TensorProvenance` contract
- implementing compiler-generated gather/scatter adjoints; task 0023 owns any selected
  backend-neutral adjoint semantics<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [adjoint glossary](../../../../glossary.md#term-adjoint))</span>
- index-value validation, bounds checks, duplicate-target inspection, value reads, storage access,
  execution, numerical reduction, mutation, or materialization<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization))</span>
- gradients, autograd traversal, graph capture, compiler validation, prepare, runtime, engine,
  tracing, ONNX import/export, lowering, kernels, conformance, or integration behavior<span data-glossary-navigation="true"> ([gradients glossary](../../../../glossary.md#term-gradient); [autograd glossary](../../../../glossary.md#autograd); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [engine glossary](../../../../glossary.md#term-engine); [ONNX glossary](../../../../glossary.md#term-onnx); [lowering glossary](../../../../glossary.md#lowering); [kernels glossary](../../../../glossary.md#kernel); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- dependencies, [Gradle](../../../../glossary.md#term-gradle), Java version, architecture rules, architecture tests, or another [module](../../../../glossary.md#term-module)
- an 0018P or later detailed task specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>, especially “Indexing taxonomy”
- [Model master plan](../master-plan.md)
- [Task 0017K](0017k-tensor-composition-semantics.md)
- [Task 0017L](0017l-tensor-composition-expressions.md)
- [Task 0018A](0018a-scalar-select-semantics.md)
- [Task 0018B](0018b-scalar-select-tensor-expression.md)
- [Task 0018C](0018c-axis-gather-semantics.md)
- [Task 0018D](0018d-axis-gather-tensor-expressions.md)
- [Task 0018D1](0018d1-primitive-take-convenience.md)
- [Task 0018E](0018e-gather-nd-semantics.md)
- [Task 0018F](0018f-gather-nd-tensor-expressions.md)
- [Task 0018G](0018g-axis-scatter-semantics.md)
- [Task 0018H](0018h-axis-scatter-tensor-expressions.md)
- [Task 0018I](0018i-scatter-nd-semantics.md)
- [Task 0018J](0018j-scatter-nd-tensor-expression.md)
- [Task 0018K](0018k-operation-signature-and-construction-hardening.md)
- [Task 0018L](0018l-shared-multi-output-tensor-provenance.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

External terminology references are explanatory evidence, not architecture authorities:

- [ONNX Gather](https://onnx.ai/onnx/operators/onnx__Gather.html)<span data-glossary-navigation="true"> ([ONNX glossary](../../../../glossary.md#term-onnx); [Gather glossary](../../../../glossary.md#gather))</span>
- [ONNX GatherElements](https://onnx.ai/onnx/operators/onnx__GatherElements.html)
- [ONNX GatherND](https://onnx.ai/onnx/operators/onnx__GatherND.html)
- [ONNX ScatterElements](https://onnx.ai/onnx/operators/onnx__ScatterElements.html)
- [ONNX ScatterND](https://onnx.ai/onnx/operators/onnx__ScatterND.html)
- [NumPy `take`](https://numpy.org/doc/stable/reference/generated/numpy.take.html)
- [NumPy `take_along_axis`](https://numpy.org/doc/stable/reference/generated/numpy.take_along_axis.html)
- [JAX `take`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.take.html)
- [JAX `take_along_axis`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.take_along_axis.html)
- [PyTorch `take`](https://docs.pytorch.org/docs/stable/generated/torch.take.html)
- [PyTorch `gather`](https://docs.pytorch.org/docs/stable/generated/torch.gather.html)

## Architecture constraints

- Operation identity, immutable operation attributes, result descriptors, and public Tensor
  expression construction remain backend-independent <span data-glossary-navigation="true"> ([Operation glossary](../../../../glossary.md#operation); [identity glossary](../../../../glossary.md#term-identity-meanings); [immutable glossary](../../../../glossary.md#term-immutable); [descriptors glossary](../../../../glossary.md#term-descriptor); [backend glossary](../../../../glossary.md#backend))</span>`modules/model` responsibilities.
- [`Tensor`](../../../../glossary.md#tensor) remains mutable public API state, not graph IR. Every retained expression records only
  pre-capture producer/provenance metadata and never a graph-local node or value identifier.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [graph glossary](../../../../glossary.md#graph); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [capture glossary](../../../../glossary.md#term-capture-meanings); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance); [metadata glossary](../../../../glossary.md#term-metadata); [node glossary](../../../../glossary.md#node))</span>
- Final kinds remain family-owned enums. Do not add a registry, string-dispatch table, adapter
  family, generic indexing descriptor, or broad facade.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [dispatch glossary](../../../../glossary.md#term-dispatch); [facade glossary](../../../../glossary.md#term-facade))</span>
- `OperationSignature` continues to validate exact attributes implementation classes and local
  occurrence cardinality only. It does not validate input descriptors, Shapes, values, or bounds.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [cardinality glossary](../../../../glossary.md#term-cardinality); [Shapes glossary](../../../../glossary.md#shape))</span>
- `IndexAxisAttrs` remains the exact normalized-[axis](../../../../glossary.md#term-axis) attributes type for [`GATHER`](../../../../glossary.md#gather) and
  `GATHER_ELEMENTS`; `ScatterElementsAttrs`, `GatherNdAttrs`, and `ScatterNdAttrs` retain their
  existing roles.
- Unstack must call or share the scalar-select construction path rather than reproduce SELECT
  validation, Shape, operation, layout, descriptor, producer, or provenance logic.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [validation glossary](../../../../glossary.md#term-validation); [layout glossary](../../../../glossary.md#layout))</span>
- No unstack grouping identifier, output-count attribute, list-valued producer, synthetic parent
  operation, or multi-output producer may be introduced.
- Current [`TensorProducer`](../../../../glossary.md#producer--tensorproducer), `TensorProvenance`, and `TensorFactory.createDerivedOutputs` contracts
  remain unchanged. If repeated select cannot satisfy the specified result through the existing
  path, stop and report the conflict instead of changing multi-output provenance.
- Fixed-add gather/scatter adjoints disappear from the public model inventory now. A future task
  0023 design must define any compiler-generated equivalent explicitly rather than depending on
  removed public methods.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- If implementation requires an architecture change, module dependency, cross-layer state, or
  compatibility policy beyond this specification, stop and report the conflict.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Package impact

Existing packages used:

- `model.operation.index` — owns the final gather/scatter/select/ND semantic identities and
  immutable attributes.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([gather glossary](../../../../glossary.md#gather); [ND glossary](../../../../glossary.md#term-nd); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- `model.operation.layout` — retains concat/stack and slice semantics but no longer owns an
  unstack kind or output attribute.<span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice))</span>
- `model.tensor` — owns the final fluent indexing surface, local Shape/type validation, repeated
  SELECT convenience, descriptors, and provenance construction.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [validation glossary](../../../../glossary.md#term-validation); [descriptors glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance))</span>

Packages added or moved:

- No package is added or moved.
- `AxisGatherKind`, `AxisScatterKind`, and the field-free helper type names remain unchanged; the
  final enum constants and public methods carry the stable semantic vocabulary. Renaming these
  package-private/family container types would add churn without changing their boundary.<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant); [carry glossary](../../../../glossary.md#term-carry))</span>
- `TensorPrimitiveTakeExpressions` and `UnstackOutputAttrs` are deleted without replacement.

## Current-to-final mapping

| Current contract | Exact current [Shape](../../../../glossary.md#shape)/meaning | Final disposition |
|---|---|---|
| `AxisGatherKind.GATHER` / `Tensor.gather` | indices and result equal `remove(data, axis)` | Remove; no replacement public convenience. |
| `AxisGatherKind.GATHER_AXIS` / `Tensor.gatherAxis` | `data[:a] + indices[:] + data[a+1:]` | Rename to [`GATHER`](../../../../glossary.md#gather) / `gather`. |
| [tensor](../../../../glossary.md#tensor)-index `Tensor.take` | exact [alias](../../../../glossary.md#aliasing-and-arena) of current `gatherAxis` | Remove. |
| primitive-array `Tensor.take` | [eager](../../../../glossary.md#term-eager-evaluation) [INT32](../../../../glossary.md#data-type--datatype) index Tensor plus current tensor-index `take` | Remove helper, overload, and test. |
| `TAKE_ALONG_AXIS` / `takeAlongAxis` | same [rank](../../../../glossary.md#term-rank), non-[axis](../../../../glossary.md#term-axis) equality, result exact indices Shape | Rename to `GATHER_ELEMENTS` / `gatherElements`. |
| `GATHER_ND` / `gatherNd` | `indices[:Q-1] + data[B+K:R]` | Retain unchanged. |
| `SCATTER_ADD` / `scatterAdd` | indices and updates equal `remove(data, axis)`; fixed add | Remove from public [model](../../../../glossary.md#term-model); task 0023 may later define an [adjoint](../../../../glossary.md#term-adjoint). |
| `SCATTER_AXIS_ADD` / `scatterAxisAdd` | updates equal current gather-axis result Shape; fixed add | Remove from public model; task 0023 may later define an adjoint. |
| `SCATTER_ELEMENTS` / `scatterElements` | same-rank aligned indices/updates; explicit reduction | Retain unchanged. |
| `SCATTER_ND` / `scatterNd` | updates equal [Gather-ND](../../../../glossary.md#gather-nd) result Shape; explicit reduction | Retain unchanged. |
| `UNSTACK` plus `UnstackOutputAttrs` | independent one-input outputs forced unresolved | Remove semantic kind/attrs; public `unstack` emits repeated SELECT. |
| `SELECT` / `select` | remove one axis at a [scalar](../../../../glossary.md#term-scalar) [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) | Retain unchanged and reuse for unstack. |
| [`SLICE`](../../../../glossary.md#slice) / `slice` | positive-step strided range | Retain unchanged in this task. |

This table is [normative](../../../../glossary.md#term-normative) for the migration. Completed task specifications remain historical records
and are not rewritten to pretend they originally implemented the final taxonomy.

## Required contracts

### Final [gather](../../../../glossary.md#gather) semantic surface

`AxisGatherKind` declares exactly:

```java
GATHER,
GATHER_ELEMENTS
```

Both [constants](../../../../glossary.md#term-constant) have [signature](../../../../glossary.md#term-signature):

```text
attributes: IndexAxisAttrs
inputs:     exactly 2, ordered [data, indices]
outputs:    exactly 1
```

`IndexAxisAttrs(int axis)` continues to require a normalized non-negative [axis](../../../../glossary.md#term-axis). Its [Javadoc](../../../../glossary.md#term-javadoc) must
describe only the retained gather and gather-[elements](../../../../glossary.md#term-element) consumers; it no longer mentions fixed-add
scatter or provisional names.

Final `GATHER` preserves the current `GATHER_AXIS` construction contract:

- null-check `data`, then `indices`;
- require exact [INT32](../../../../glossary.md#data-type--datatype) or [INT64](../../../../glossary.md#data-type--datatype) indices before axis [normalization](../../../../glossary.md#term-normalization);
- normalize the raw positive or negative axis against data [rank](../../../../glossary.md#term-rank);
- derive `data[:axis] + indices[:] + data[axis + 1:]`, preserving every exact [Dimension](../../../../glossary.md#dimension) reference;
- preserve [data type](../../../../glossary.md#data-type--datatype) and data `requiresGrad`;
- create a fresh, unlabeled, storage-free result with unresolved [layout](../../../../glossary.md#layout);
- retain ordered [producer](../../../../glossary.md#term-producer) inputs `[data, indices]` and [provenance](../../../../glossary.md#provenance) output index zero; and
- never inspect index values or bounds.

Final `GATHER_ELEMENTS` preserves the current `TAKE_ALONG_AXIS` construction contract:

- use the same null/type/axis ordering as final `GATHER`;
- require indices rank equal data rank;
- require structural Dimension equality at every non-selected axis, in increasing-axis order;
- allow the selected indices [extent](../../../../glossary.md#term-extent) to differ;
- retain the exact indices [Shape](../../../../glossary.md#shape) object as the result Shape;
- preserve data type and data `requiresGrad`; and
- create the same fresh unresolved one-output provenance as final `GATHER`.

Failure text must use only final names, including:

```text
gather indices data type must be INT32 or INT64: <type>
gatherElements indices data type must be INT32 or INT64: <type>
gatherElements indices rank must match data rank: expected=<rank>, actual=<rank>
gatherElements indices dimension at axis <axis> must match data: expected=<d>, actual=<d>
```

Axis-range failures continue to come from `Shape.normalizeAxis`; tests assert exception type and
the repository's exact Shape-axis message. There is no reduced-rank gather Shape failure after the
old path is removed.

`TensorAxisGatherExpressions` remains field-free and contains only final gather/gather-elements
entry points plus their cohesive private [validation](../../../../glossary.md#term-validation), Shape, and construction helpers. Remove
`removeAxis`, the reduced-rank compatibility path, the [tensor](../../../../glossary.md#tensor)-index [alias](../../../../glossary.md#aliasing-and-arena) method, and every
provisional-name branch. Rename private helpers and messages from `gatherAxis` or
`takeAlongAxis` to final vocabulary.

### Final public [gather](../../../../glossary.md#gather) and [ND](../../../../glossary.md#term-nd) surface

[`Tensor`](../../../../glossary.md#tensor) exposes exactly these gather-family instance methods:

```java
public Tensor gather(Tensor indices, int axis)
public Tensor gatherElements(Tensor indices, int axis)
public Tensor gatherNd(Tensor indices)
public Tensor gatherNd(Tensor indices, int batchDimensions)
```

The [parameter](../../../../glossary.md#term-parameter) order for both [axis](../../../../glossary.md#term-axis) methods is `(indices, axis)`, matching the existing gather
methods and keeping the index Tensor adjacent to the other gather-family overloads. Delete:

```java
public Tensor gatherAxis(Tensor indices, int axis)
public Tensor take(int axis, Tensor indices)
public Tensor take(int axis, int[] indices)
public Tensor takeAlongAxis(Tensor indices, int axis)
```

Do not leave deprecated methods, forwarding [aliases](../../../../glossary.md#aliasing-and-arena), package-private compatibility entries, [eager](../../../../glossary.md#term-eager-evaluation)
primitive-array allocation, or alternative parameter-order overloads. `TensorPrimitiveTakeExpressions`
and `TensorPrimitiveTakeExpressionTest` are deleted.

[Gather-ND](../../../../glossary.md#gather-nd) implementation, attributes, formula, [validation](../../../../glossary.md#term-validation) order, overload delegation, result
[metadata](../../../../glossary.md#term-metadata), and tests remain unchanged except references needed to assert distinction from the new
two-[constant](../../../../glossary.md#term-constant) axis-gather enum.

### Final scatter semantic and public surface

`AxisScatterKind` declares exactly:

```java
SCATTER_ELEMENTS
```

Its [signature](../../../../glossary.md#term-signature) remains:

```text
attributes: ScatterElementsAttrs
inputs:     exactly 3, ordered [data, indices, updates]
outputs:    exactly 1
```

Delete `SCATTER_ADD` and `SCATTER_AXIS_ADD` and remove `IndexAxisAttrs` from every [axis](../../../../glossary.md#term-axis)-scatter
signature branch. Do not add `SCATTER_ELEMENTS_ADD` or another fixed-reduction kind.

[`Tensor`](../../../../glossary.md#tensor) retains exactly:

```java
public Tensor scatterElements(Tensor indices, Tensor updates, int axis)
public Tensor scatterElements(
        Tensor indices, Tensor updates, int axis, ScatterReduction reduction)
public Tensor scatterNd(Tensor indices, Tensor updates)
public Tensor scatterNd(
        Tensor indices, Tensor updates, ScatterReduction reduction)
public Tensor scatterNd(
        Tensor indices, Tensor updates, ScatterReduction reduction, int batchDimensions)
```

Delete `scatterAdd` and `scatterAxisAdd`. `TensorAxisScatterExpressions` removes their entry
points, floating-only checks, reduced-axis/[gather](../../../../glossary.md#gather)-axis [Shape](../../../../glossary.md#shape) derivation, and fixed-add construction
branches. It keeps only cohesive scatter-[elements](../../../../glossary.md#term-element) [validation](../../../../glossary.md#term-validation) and construction. The short overload
continues to delegate with `ScatterReduction.NONE`.

All retained scatter-elements and [scatter-ND](../../../../glossary.md#scatter-nd) contracts remain exact, including:

- null order `data`, `indices`, `updates`, then explicit reduction where applicable;
- [INT32](../../../../glossary.md#data-type--datatype)/[INT64](../../../../glossary.md#data-type--datatype) indices;
- exact data/update type equality;
- scatter-elements equal indices/updates Shape, same [rank](../../../../glossary.md#term-rank) as data, and non-axis [alignment](../../../../glossary.md#term-alignment);
- current reduction eligibility and duplicate-[target](../../../../glossary.md#term-target) policy;
- result exact data Shape and type;
- `requiresGrad` equal logical OR of data and updates;
- unresolved [layout](../../../../glossary.md#layout) and ordered `[data, indices, updates]` [provenance](../../../../glossary.md#provenance); and
- no value, bounds, duplicate-target, execution, or [gradient](../../../../glossary.md#term-gradient) behavior.

### Unstack as repeated SELECT

`Tensor.unstack(int axis)` remains public and delegates to the field-free composition helper. The
helper performs this exact sequence before creating a result:

1. reject null input with `NullPointerException("input")`;
2. normalize the existing [axis](../../../../glossary.md#term-axis) with the current unstack-specific range message;
3. require a `StaticDimension` at the [normalized axis](../../../../glossary.md#normalized-axis);
4. require its size not exceed `Integer.MAX_VALUE`;
5. if the size is zero, return `List.of()` without creating an [operation](../../../../glossary.md#operation), [producer](../../../../glossary.md#term-producer), [Tensor](../../../../glossary.md#tensor), or ID;
6. otherwise create [coordinates](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) `0` through `size - 1` in order through the existing scalar-select
   construction path; and<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
7. return an [immutable](../../../../glossary.md#term-immutable) list retaining those fresh result references in coordinate order.

For each non-empty result at coordinate `i`, the observable contract is exactly the same as
`input.select(normalizedAxis, i)`:

- kind `SelectKind.SELECT`;
- attributes `new SelectAttrs(normalizedAxis, (long) i)`;
- exact [Shape](../../../../glossary.md#shape) obtained by removing the selected [Dimension](../../../../glossary.md#dimension);
- exact input [data type](../../../../glossary.md#data-type--datatype) and `requiresGrad`;
- one independent [`TensorProducer`](../../../../glossary.md#producer--tensorproducer) with ordered inputs `[input]` and exactly one output [descriptor](../../../../glossary.md#term-descriptor);
- `TensorProvenance(producer, 0)` and legacy projection output index zero;
- a fresh ID, no label, and no storage;
- resolved selected-stride/advanced-offset layout when SELECT can derive a non-empty resolved
  view; and<span data-glossary-navigation="true"> ([stride glossary](../../../../glossary.md#term-stride); [layout glossary](../../../../glossary.md#layout); [view glossary](../../../../glossary.md#term-view-meanings))</span>
- unresolved layout when the input layout is unresolved or the result is empty, exactly as SELECT
  already specifies.

The helper must not call the generic unresolved composition `create` path for unstack, derive a
parallel `unstackShape`, construct operations directly, or call `createDerivedOutputs`. It must
delegate to `TensorSelectExpressions.select(input, normalizedAxis, coordinate)` or a narrower
shared SELECT routine that preserves all current public `select` behavior.

If ID exhaustion occurs after some coordinates were created, the exception propagates and prior
IDs remain consumed; no partial list is returned and no [rollback](../../../../glossary.md#term-rollback) is attempted. This preserves the
current monotonic identifier contract.

Delete `UnstackOutputAttrs`. `TensorCompositionKind` contains only `CONCAT` and `STACK`; both retain
`CompositionAxisAttrs`, [variadic](../../../../glossary.md#term-variadic) one-to-`Integer.MAX_VALUE` inputs, and exactly one output.
`TensorCompositionKind.signatures()` removes only its obsolete UNSTACK branch and retains
the shared concat/stack variadic [signature](../../../../glossary.md#term-signature). The `OperationSignature` value contract gains no
SELECT special case and remains unchanged.

### [Provenance](../../../../glossary.md#provenance) boundary and planned true multi-output operations

[`TensorProducer`](../../../../glossary.md#producer--tensorproducer), `TensorProvenance`, `TensorFactory.createDerived`, and
`TensorFactory.createDerivedOutputs` require no executable changes. Tests must prove repeated
unstack results do not share [producers](../../../../glossary.md#term-producer), every result has [producer](../../../../glossary.md#term-producer) output count one, and every
provenance output index is zero.

The documentation must explicitly distinguish:

```text
unstack: N independent SELECT occurrences -> N independent producers -> every output index 0
top-K:   one future selection occurrence   -> one shared producer      -> output indices 0 and 1
```

This task must not design or implement top-K; the comparison exists only to prevent future code
from weakening the shared multi-output provenance contract to accommodate a convenience list.

### [Validation](../../../../glossary.md#term-validation) order and side effects

Retained [gather](../../../../glossary.md#gather)/scatter paths preserve their current validation order except renamed failure text.
Removed paths leave no allocation or validation side effects because their public entry points no
longer exist. In particular:

- no primitive-array take allocates an [eager](../../../../glossary.md#term-eager-evaluation) index [Tensor](../../../../glossary.md#tensor);
- invalid final gather/gather-[elements](../../../../glossary.md#term-element)/scatter-elements inputs fail before result ID allocation;
- unstack validates the entire count boundary before the first SELECT result allocation;
- zero-size unstack consumes no ID; and
- successful unstack consumes one ID per SELECT result in order.

Tests that observe ID deltas must reserve the expected interval with the repository's established
test seam and avoid depending on unrelated global counter values.

## Exact affected files

The implementation begins with this exact Java [scope](../../../../glossary.md#term-scope):

Production changes or deletions:

1. `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/AxisGatherKind.java`
2. `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/AxisScatterKind.java`
3. `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/IndexAxisAttrs.java`
4. `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/SelectAttrs.java`
5. `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/SelectKind.java`
6. `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/TensorCompositionKind.java`
7. `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/UnstackOutputAttrs.java` (delete)
8. `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
9. `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorAxisGatherExpressions.java`
10. `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorAxisScatterExpressions.java`
11. `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorCompositionExpressions.java`
12. `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorPrimitiveTakeExpressions.java` (delete)

Focused test changes or deletions:

13. `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/OperationSignatureTest.java`
14. `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/index/AxisGatherSemanticsTest.java`
15. `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/index/AxisScatterSemanticsTest.java`
16. `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/index/SelectSemanticsTest.java`
17. `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/layout/TensorCompositionSemanticsTest.java`
18. `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorAxisGatherExpressionTest.java`
19. `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorAxisScatterExpressionTest.java`
20. `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorCompositionExpressionTest.java`
21. `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorPrimitiveTakeExpressionTest.java` (delete)
22. `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`

The documentation-focused pass may change only these additional paths unless it records a concrete
stale contract discovered during review:

23. `docs/api/tensor-api.md`
24. `docs/api/compile-api.md`
25. `docs/glossary.md`
26. `docs/planning/modules/model/capabilities.md`
27. this task specification
28. `docs/planning/modules/model/master-plan.md`
29. `docs/planning/roadmap.md`

The 22-file Java scope is an explicit [atomic](../../../../glossary.md#resource-transaction-and-rollback)-migration exception to the [planning](../../../../glossary.md#term-planning) guide's ordinary
guardrail. The vocabulary is simultaneously encoded by semantic enums, exact [signature](../../../../glossary.md#term-signature) branches,
fluent methods, field-free helpers, [operation](../../../../glossary.md#operation)-distinction tests, reflection inventories, and
unstack [provenance](../../../../glossary.md#provenance) tests. Splitting those mechanical layers would leave uncompilable references
or a contradictory public/semantic [model](../../../../glossary.md#term-model). The work remains one cohesive model capability, changes
no [module](../../../../glossary.md#term-module) boundary, and prohibits unrelated refactoring.

`GatherNdKind`, `GatherNdAttrs`, `ScatterNdKind`, `ScatterNdAttrs`, `ScatterElementsAttrs`,
`ScatterReduction`, `TensorGatherNdExpressions`, `TensorScatterNdExpressions`, [`TensorProducer`](../../../../glossary.md#producer--tensorproducer),
`TensorProvenance`, and `TensorFactory` are required review surfaces but are expected to remain
unchanged. If an executable change to one becomes necessary, stop and amend the task evidence and
scope before editing it.

## Tests to add or update

### Semantic and [signature](../../../../glossary.md#term-signature) tests

- assert exact ordered `AxisGatherKind` [constants](../../../../glossary.md#term-constant) [`GATHER`](../../../../glossary.md#gather), `GATHER_ELEMENTS` and no others;
- assert both accept only exact `IndexAxisAttrs`, exactly two inputs, and exactly one output;
- assert exact ordered `AxisScatterKind` constant `SCATTER_ELEMENTS` and no others;
- assert it accepts only `ScatterElementsAttrs`, exactly three inputs, and exactly one output;
- assert exact ordered `TensorCompositionKind` constants `CONCAT`, `STACK` and no others;
- remove every `UnstackOutputAttrs` [shape](../../../../glossary.md#shape)/value/failure assertion;
- assert concat/stack signatures remain [variadic](../../../../glossary.md#term-variadic) one-to-`Integer.MAX_VALUE`, exactly one output;
- retain [Gather-ND](../../../../glossary.md#gather-nd), [Scatter-ND](../../../../glossary.md#scatter-nd), SELECT, [SLICE](../../../../glossary.md#slice), reduction, and attributes surface assertions;
- update cross-family distinctness tests to the final inventory; and
- assert source/reflection inventories contain none of the removed kinds or attributes.

### [Gather](../../../../glossary.md#gather) expression tests

- migrate current gather-[axis](../../../../glossary.md#term-axis) examples to `gather` and `GATHER`;
- migrate current take-along-axis examples to `gatherElements` and `GATHER_ELEMENTS`;
- cover [scalar](../../../../glossary.md#term-scalar), ordinary, zero-sized, and dynamic indices [Shapes](../../../../glossary.md#shape) for [canonical](../../../../glossary.md#term-canonical-form) gather;
- cover equal-rank alignment, selected-axis extent changes, dynamic structural equality, and exact
  indices Shape identity for gather-elements;<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [alignment glossary](../../../../glossary.md#term-alignment); [extent glossary](../../../../glossary.md#term-extent); [structural equality glossary](../../../../glossary.md#term-structural-equality); [identity glossary](../../../../glossary.md#term-identity-meanings); [elements glossary](../../../../glossary.md#term-element))</span>
- cover negative-axis [normalization](../../../../glossary.md#term-normalization) and invalid-axis behavior;
- cover [INT32](../../../../glossary.md#data-type--datatype)/[INT64](../../../../glossary.md#data-type--datatype) acceptance and every rejected current non-index type;
- cover exact renamed failure text and [validation](../../../../glossary.md#term-validation) order;
- cover Shape/type/[gradient](../../../../glossary.md#term-gradient)/[layout](../../../../glossary.md#layout)/storage/freshness/ordered-[provenance](../../../../glossary.md#provenance) [metadata](../../../../glossary.md#term-metadata); and
- replace the reflection inventory with exactly the two final public axis-gather methods plus the
  unchanged two gather-ND methods.<span data-glossary-navigation="true"> ([gather-ND glossary](../../../../glossary.md#gather-nd))</span>

### Scatter expression tests

- delete fixed-add behavior tests and every public `scatterAdd`/`scatterAxisAdd` reflection entry;
- retain full scatter-elements short/explicit overload, Shape, reduction, type, null, axis,
  metadata, and failure-order coverage;<span data-glossary-navigation="true"> ([elements glossary](../../../../glossary.md#term-element); [Shape glossary](../../../../glossary.md#shape); [axis glossary](../../../../glossary.md#term-axis); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- retain [scatter-ND](../../../../glossary.md#scatter-nd) coverage unchanged;
- assert the public scatter-family inventory is exactly two scatter-elements plus three scatter-ND
  overloads; and
- assert no removed fixed-add name survives in kinds, [Tensor](../../../../glossary.md#tensor) methods, or helper entry points.

### Unstack [normalization](../../../../glossary.md#term-normalization) tests

- preserve input-null, positive/negative axis, scalar-rank failure, dynamic selected extent,
  greater-than-<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [scalar glossary](../../../../glossary.md#term-scalar); [rank glossary](../../../../glossary.md#term-rank))</span>`Integer.MAX_VALUE`, zero, ordinary, and [immutable](../../../../glossary.md#term-immutable)-list coverage;
- assert zero [extent](../../../../glossary.md#term-extent) returns the [canonical](../../../../glossary.md#term-canonical-form) immutable empty list and consumes no ID;
- for every ordinary output, compare against an independently constructed `select(axis, i)` for
  exact kind, attributes, Shape, type, gradient eligibility, and layout value;<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- assert ordered [coordinates](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) and fresh distinct result IDs;
- assert each output has its own [producer](../../../../glossary.md#term-producer), producer input `[input]`, one output descriptor,
  provenance output index zero, no storage, and no label;<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance))</span>
- assert resolved non-empty input [layout](../../../../glossary.md#layout) produces the same offset/[stride](../../../../glossary.md#term-stride) [view](../../../../glossary.md#term-view-meanings) as [scalar select](../../../../glossary.md#scalar-select);
- assert unresolved input layout and empty result geometry remain unresolved exactly as select;
- assert an ID-exhaustion failure does not roll back previously consumed IDs; and
- remove every assertion of `UNSTACK`, `UnstackOutputAttrs`, shared result Shape forced unresolved,
  or operation-attribute output indices.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>

### Public-surface and negative-presence tests

Update the exact [`Tensor`](../../../../glossary.md#tensor) declared-method inventory and add/retain negative checks for:

```text
gatherAxis
take
takeAlongAxis
scatterAdd
scatterAxisAdd
```

Also verify deleted enum [constants](../../../../glossary.md#term-constant) cannot be recovered through `valueOf`, and use a repository
search in [validation](../../../../glossary.md#term-validation) to prove removed names do not remain in live Java/[API](../../../../glossary.md#term-application-programming-interface-api)/glossary text except
historical completed task documents and the current-to-final migration record in this task.

## Documentation impact

Document type classification for the independent pass:

- affected Java comments are [API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc) documentation;
- `docs/api/tensor-api.md` and `docs/api/compile-api.md` use the API-reference profile;
- `docs/glossary.md` uses the glossary/reference style from the general profile;
- this task, [model](../../../../glossary.md#term-model) master plan, capability baseline, and roadmap use the [planning](../../../../glossary.md#term-planning) profile.

Required updates:

- [`Tensor`](../../../../glossary.md#tensor) and helper/kind/attribute Javadocs describe only final names, exact Shape formulas,
  validation, metadata, and unstack-as-SELECT composition.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [validation glossary](../../../../glossary.md#term-validation); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Tensor API replaces the provisional axis-gather/scatter inventory and first-class unstack
  operation with final kinds, methods, formulas, examples, and provenance behavior.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [gather glossary](../../../../glossary.md#gather); [operation glossary](../../../../glossary.md#operation); [provenance glossary](../../../../glossary.md#provenance))</span>
- Compile API retains its current model/compiler boundary but replaces any provisional operation
  inventory and makes clear that removed fixed-add adjoints are not current public model kinds;
  task 0023 may later add selected compiler-generated semantics.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- Glossary replaces “axis gather,” “axis scatter,” and first-class unstack language with precise
  Gather, Gather Elements, Scatter Elements, and repeated-select definitions; retain Gather-ND,
  Scatter-ND, scalar select, producer, and provenance distinctions.<span data-glossary-navigation="true"> ([Gather Elements glossary](../../../../glossary.md#gather-elements); [Scatter Elements glossary](../../../../glossary.md#scatter-elements); [Gather-ND glossary](../../../../glossary.md#gather-nd); [Scatter-ND glossary](../../../../glossary.md#scatter-nd); [scalar select glossary](../../../../glossary.md#scalar-select); [producer glossary](../../../../glossary.md#term-producer))</span>
- Capability baseline changes its remaining “may remain” language to the final no-alias decision
  and records task 0018O as the owner once complete.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena); [owner glossary](../../../../glossary.md#term-owner))</span>
- Master plan and roadmap link this task, update status/evidence when implementation completes,
  and continue to identify 0018P as the next unfinished task. Do not create 0018P's specification.

No architecture document or [ADR](../../../../glossary.md#term-architecture-decision-record-adr) update is expected because the model/[compiler](../../../../glossary.md#term-compiler)/[runtime](../../../../glossary.md#term-runtime) boundaries
and [dependency](../../../../glossary.md#term-dependency) rules do not change. No Training API update is expected because no training or
[gradient](../../../../glossary.md#term-gradient) behavior changes. The documentation agent must record reasoned no-change conclusions for
those surfaces rather than silently omitting them.

## Acceptance criteria

- [x] `AxisGatherKind` contains exactly [`GATHER`](../../../../glossary.md#gather), `GATHER_ELEMENTS`, in order.
- [x] Final `GATHER` has the former [canonical](../../../../glossary.md#term-canonical-form) gather-[axis](../../../../glossary.md#term-axis) [Shape](../../../../glossary.md#shape) formula.
- [x] Final `GATHER_ELEMENTS` has the former aligned-index Shape formula.
- [x] Current reduced-[rank](../../../../glossary.md#term-rank) gather semantics and public construction are absent.
- [x] [`Tensor`](../../../../glossary.md#tensor) exposes exactly `gather(indices, axis)` and `gatherElements(indices, axis)` for
      axis-index gather, plus unchanged gather-ND overloads.<span data-glossary-navigation="true"> ([gather-ND glossary](../../../../glossary.md#gather-nd))</span>
- [x] No `gatherAxis`, `take`, `takeAlongAxis`, primitive-array helper, or compatibility alias
      remains.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- [x] `AxisScatterKind` contains exactly `SCATTER_ELEMENTS`; fixed-add kinds and methods are absent.
- [x] Scatter-elements, Gather-ND, and Scatter-ND retained behavior and public overloads pass their
      focused tests unchanged.<span data-glossary-navigation="true"> ([elements glossary](../../../../glossary.md#term-element); [Scatter-ND glossary](../../../../glossary.md#scatter-nd))</span>
- [x] `TensorCompositionKind` contains exactly `CONCAT`, `STACK`; `UnstackOutputAttrs` is deleted.
- [x] `unstack` returns ordered immutable repeated-SELECT results after current upfront count
      validation.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [validation glossary](../../../../glossary.md#term-validation))</span>
- [x] Empty unstack creates no producer/Tensor/ID; non-empty outputs have independent one-output
      producers and provenance output index zero.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>
- [x] Every unstack output has scalar select's exact descriptor and conditional view-layout
      behavior.<span data-glossary-navigation="true"> ([scalar select glossary](../../../../glossary.md#scalar-select); [descriptor glossary](../../../../glossary.md#term-descriptor); [view glossary](../../../../glossary.md#term-view-meanings); [layout glossary](../../../../glossary.md#layout))</span>
- [x] [`TensorProducer`](../../../../glossary.md#producer--tensorproducer), `TensorProvenance`, `TensorFactory`, graph contracts, and module boundaries
      remain unchanged.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [module glossary](../../../../glossary.md#term-module))</span>
- [x] Exact operation signatures, reflection inventories, negative-presence assertions, and
      failure order are tested.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [signatures glossary](../../../../glossary.md#term-signature))</span>
- [x] Javadocs document every affected public input, return, failure, layout, and provenance
      contract without restating implementation.
- [x] Tensor API, Compile API, glossary, capability baseline, task evidence, master plan, and
      roadmap are synchronized by an independent documentation-focused agent.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile))</span>
- [x] Architecture, focused architecture docs, ADRs/tests, Training API, backend conformance,
      integration tests, dependencies, Gradle, and other modules have explicit no-change
      conclusions unless a concrete contradiction is found.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- [x] No unrelated refactor, later task specification, [commit](../../../../glossary.md#term-commit), or push is included.

## [Validation](../../../../glossary.md#term-validation)

Implementation-agent validation:

```bash
./gradlew :modules:model:test --tests '*AxisGatherSemanticsTest' \
  --tests '*AxisScatterSemanticsTest' \
  --tests '*TensorCompositionSemanticsTest' \
  --tests '*SelectSemanticsTest' \
  --tests '*OperationSignatureTest' \
  --tests '*TensorAxisGatherExpressionTest' \
  --tests '*TensorAxisScatterExpressionTest' \
  --tests '*TensorCompositionExpressionTest' \
  --tests '*TensorTest'
./gradlew :modules:model:test
```

The implementation agent also runs exact surface searches before [handoff](../../../../glossary.md#term-handoff):

```bash
rg -n 'GATHER_AXIS|TAKE_ALONG_AXIS|SCATTER_ADD|SCATTER_AXIS_ADD|UNSTACK|UnstackOutputAttrs|gatherAxis|takeAlongAxis|scatterAdd|scatterAxisAdd' \
  modules/model/src/main/java modules/model/src/test/java
rg -n 'public Tensor take\(' modules/model/src/main/java modules/model/src/test/java
```

Both searches must be empty. The deleted primitive-take files and unstack-attribute file must be
absent. Historical [planning](../../../../glossary.md#term-planning) documents are intentionally excluded from this absence check.

Independent documentation-agent validation, reusing the successful [model](../../../../glossary.md#term-model)-test evidence unless it
changes executable Java:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

The documentation agent additionally verifies:

- exact public methods and enum [constants](../../../../glossary.md#term-constant) with reflection or `javap`;
- current API/glossary text contains no removed vocabulary except an explicitly labeled migration
  note where needed;<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- every local Markdown link and anchor in changed docs resolves;
- code fences are balanced and no trailing whitespace exists;
- the changed-path set is within the 29 listed paths or has a recorded, justified [scope](../../../../glossary.md#term-scope) amendment;
- task 0018O is `Complete` only after all implementation/documentation evidence passes;
- master plan and roadmap point to 0018P as Draft without a detailed specification; and
- no architecture, [Gradle](../../../../glossary.md#term-gradle), [dependency](../../../../glossary.md#term-dependency), spec, [commit](../../../../glossary.md#term-commit), or push change is present.

Repository-wide tests are not required for this single-[module](../../../../glossary.md#term-module) task. The public-surface cleanup
[checkpoint](../../../../glossary.md#term-checkpoint) after task 0018S owns the next recorded full repository validation unless implementation
reveals a cross-module dependency or shared-build effect.

## Dependencies

- completed tasks 0017K–0017L
- completed tasks 0018A–0018J, including 0018D1
- completed task 0018K
- completed task 0018L

## Follow-up tasks

- 0018P: [elementwise](../../../../glossary.md#term-elementwise) semantic cleanup
- 0018R: [slice](../../../../glossary.md#slice) and window public-contract cleanup
- 0019A: [embedding](../../../../glossary.md#embedding) and one-hot conveniences built from the normalized indexing baseline
- 0019C: sorting and genuine multi-output top-K using shared [producer](../../../../glossary.md#term-producer) [provenance](../../../../glossary.md#provenance)
- 0023: selected compiler-generated semantic operations, including any required gather/scatter
  adjoints<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [gather glossary](../../../../glossary.md#gather))</span>

## Implementation prompt

Work only on task 0018O in a fresh implementation context. Read `AGENTS.md`, `ARCHITECTURE.md`,
the current architecture index, documentation rules, [planning](../../../../glossary.md#term-planning) guide, roadmap, [model](../../../../glossary.md#term-model) capabilities,
model master plan, tasks 0017K–0017L and 0018A–0018L, and every production/test file listed in the
exact affected [scope](../../../../glossary.md#term-scope) before editing. Preserve completed task history.

Implement the final indexing taxonomy atomically: final [`GATHER`](../../../../glossary.md#gather) is current `GATHER_AXIS`; final
`GATHER_ELEMENTS` is current `TAKE_ALONG_AXIS`; remove current reduced-[rank](../../../../glossary.md#term-rank) gather, every `take`
spelling/helper, fixed-add scatter kinds/methods, and first-class `UNSTACK` attributes/kind. Keep
[Gather-ND](../../../../glossary.md#gather-nd), [Scatter Elements](../../../../glossary.md#scatter-elements), [Scatter-ND](../../../../glossary.md#scatter-nd), SELECT, and [SLICE](../../../../glossary.md#slice) contracts unchanged. Implement public
unstack as ordered repeated [scalar SELECT](../../../../glossary.md#scalar-select) after the specified upfront count [validation](../../../../glossary.md#term-validation), with one
independent [producer](../../../../glossary.md#term-producer) and [provenance](../../../../glossary.md#provenance) output index zero per result. Do not alter shared multi-output
provenance, [graph](../../../../glossary.md#graph), [compiler](../../../../glossary.md#term-compiler), [runtime](../../../../glossary.md#term-runtime), [backend](../../../../glossary.md#backend), dependencies, [Gradle](../../../../glossary.md#term-gradle), or architecture.

[Run](../../../../glossary.md#term-run) the focused tests and full model suite, then hand the same working tree and evidence to a
separate documentation-focused agent. That agent must independently finalize affected Javadocs,
[Tensor](../../../../glossary.md#tensor)/[Compile](../../../../glossary.md#compile) [API](../../../../glossary.md#term-application-programming-interface-api), glossary, capability baseline, task evidence, master plan, and roadmap, record
reasoned no-change conclusions, run model [Javadoc](../../../../glossary.md#term-javadoc) and documentation/scope checks, and mark this task
`Complete` only when every acceptance criterion passes. Do not [commit](../../../../glossary.md#term-commit) or push.

## Local decisions

- Final Gather uses the former canonical axis-replacement implementation; Gather Elements uses
  the former same-rank aligned implementation. The reduced-rank spelling was removed.<span data-glossary-navigation="true"> ([Gather glossary](../../../../glossary.md#gather); [canonical glossary](../../../../glossary.md#term-canonical-form); [axis glossary](../../../../glossary.md#term-axis); [Gather Elements glossary](../../../../glossary.md#gather-elements); [rank glossary](../../../../glossary.md#term-rank))</span>
- Public unstack delegates each coordinate to scalar SELECT after validating the complete count.
  It deliberately does not use shared multi-output provenance.<span data-glossary-navigation="true"> ([coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [scalar SELECT glossary](../../../../glossary.md#scalar-select); [provenance glossary](../../../../glossary.md#provenance))</span>
- No compatibility aliases were retained. Fixed-add adjoints are not current public model kinds;
  task 0023 remains the possible future owner of selected compiler-generated semantics.<span data-glossary-navigation="true"> ([aliases glossary](../../../../glossary.md#aliasing-and-arena); [model glossary](../../../../glossary.md#term-model); [owner glossary](../../../../glossary.md#term-owner); [compiler glossary](../../../../glossary.md#term-compiler))</span>

## Known limitations

- Index-value bounds, duplicate-target detection, numerical writes/reductions, gradients,
  compiler capture, lowering, and execution remain outside this model-only task.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [gradients glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [lowering glossary](../../../../glossary.md#lowering); [model glossary](../../../../glossary.md#term-model))</span>
- The next repository-wide validation remains the public-surface cleanup checkpoint after task
  0018S or CI, as specified by the planning guide.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [CI glossary](../../../../glossary.md#term-continuous-integration-ci); [planning glossary](../../../../glossary.md#term-planning))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation context `/root/task_0018o_implementation` ran the exact nine-filter focused
  command from this task. Its final rerun passed with `BUILD SUCCESSFUL in 2s`; the preceding
  diagnostic run reported 58 completed focused tests. It then ran<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
  `./gradlew :modules:model:test`, which passed with `BUILD SUCCESSFUL in 1s`; generated XML
  reports contain 725 tests across 88 suites with zero failures, errors, or skips.
- The same implementation context ran both exact live-Java absence searches from this task; both
  were empty. The three deleted helper/attribute/test paths are absent. No executable Java changed
  after those test runs.
- Independent documentation context
  `/root/task_0018o_implementation/task_0018o_docs` reviewed the final Java diff and every affected
  Javadoc, then finalized <span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>[`Tensor`](../../../../glossary.md#tensor)/helper/kind/attribute Javadocs, Tensor API, Compile API,
  glossary, capability baseline, task, master plan, and roadmap.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile))</span>
- `./gradlew :modules:model:javadoc` passed with `BUILD SUCCESSFUL in 1s` after the final Javadocs;
  the final post-evidence rerun also passed in 331 ms with both tasks up to date.
- `javap -classpath modules/model/build/classes/java/main` confirmed exactly [`GATHER`](../../../../glossary.md#gather) and
  `GATHER_ELEMENTS`, exactly `SCATTER_ELEMENTS`, exactly `CONCAT` and `STACK`, the two final axis
  gather methods, two Scatter Elements overloads, unchanged Gather-ND/Scatter-ND overloads, and
  public <span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [Scatter Elements glossary](../../../../glossary.md#scatter-elements); [Gather-ND glossary](../../../../glossary.md#gather-nd); [Scatter-ND glossary](../../../../glossary.md#scatter-nd))</span>`unstack(int)`.
- Repeated live-Java and current Tensor/Compile API/glossary absence searches for the removed
  constants, attributes, helper spellings, and public <span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant))</span>`take` declaration were empty. The current
  API/glossary contains no migration-only vocabulary.
- A local Markdown path-and-GitHub-anchor check resolved 469 links across the seven changed
  documentation files. Fence, final-newline, and trailing-whitespace checks passed for all seven.
- The final changed-path inventory contains exactly the 29 permitted paths. Deleted paths and the
  absence of an 0018P detailed specification were verified. Task 0018O is Complete; the master
  plan and roadmap identify 0018P as Draft.
- `git diff --check` passed. No architecture contract, focused architecture page, ADR,
  architecture test, Training API, backend-conformance/integration test, Gradle/dependency file,
  other module, commit, or push changed.<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [architecture test glossary](../../../../glossary.md#term-architecture-test); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [integration test glossary](../../../../glossary.md#term-integration-test); [Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [module glossary](../../../../glossary.md#term-module); [commit glossary](../../../../glossary.md#term-commit))</span>

## Implementation notes

- [Gather-ND](../../../../glossary.md#gather-nd), [Scatter-ND](../../../../glossary.md#scatter-nd), SELECT, [SLICE](../../../../glossary.md#slice), and shared [`TensorProducer`](../../../../glossary.md#producer--tensorproducer)/`TensorProvenance` contracts
  remain executable-code unchanged. Unstack reuses SELECT layout and provenance behavior.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance))</span>
- Compile API was updated only to describe the current model/compiler boundary and to state that
  fixed-add adjoints are not current public kinds. Training API remains accurate unchanged because
  this task adds no gradient or training behavior.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [model glossary](../../../../glossary.md#term-model); [compiler glossary](../../../../glossary.md#term-compiler); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Architecture documents, ADRs, and architecture tests remain accurate unchanged because module
  ownership and dependencies did not change. Backend conformance/integration tests remain
  unchanged because no backend or end-to-end execution exists for this metadata-only change.
  Gradle, Java 26 configuration, dependencies, and other modules remain unchanged for the same
  scope reason.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [metadata glossary](../../../../glossary.md#term-metadata); [Gradle glossary](../../../../glossary.md#term-gradle); [scope glossary](../../../../glossary.md#term-scope))</span>

## Completion summary

- Completed changes: finalized Gather, Gather Elements, Scatter Elements, and repeated-SELECT
  unstack taxonomy with removed provisional public surface.<span data-glossary-navigation="true"> ([Gather glossary](../../../../glossary.md#gather); [Gather Elements glossary](../../../../glossary.md#gather-elements); [Scatter Elements glossary](../../../../glossary.md#scatter-elements))</span>
- Files changed or created: exactly the 29 task-authorized Java, test, API, glossary, capability,
  task, master-plan, and roadmap paths.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Tests and [validation](../../../../glossary.md#term-validation): reused passing focused and 725-test [model](../../../../glossary.md#term-model) evidence; model [Javadoc](../../../../glossary.md#term-javadoc),
  `javap`, absence, Markdown, [scope](../../../../glossary.md#term-scope), status, and `git diff --check` validation passed.
- Documentation-agent review: completed in
  `/root/task_0018o_implementation/task_0018o_docs` using the API/Javadoc, planning, example, and
  general documentation profiles.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: Tensor API, Compile API, glossary, capability baseline, task, master plan,
  and roadmap finalized; architecture and Training API require no change for the recorded reasons.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile))</span>
- Javadoc review: affected public and contract-relevant Javadocs finalized; generated successfully.
- Glossary impact: final Gather, Gather Elements, Scatter Elements, and repeated-select terms
  replace the provisional taxonomy.
- Unresolved issues: None.
- Follow-up required: None.

Status: Complete
