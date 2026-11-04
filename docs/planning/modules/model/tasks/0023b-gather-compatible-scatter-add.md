# Task 0023B: [Gather](../../../../glossary.md#gather)-Compatible Scatter-Add

## Status

Complete

## Goal

Add one [backend](../../../../glossary.md#backend)-independent public functional scatter-add whose indices and updates have exactly
the [Shape](../../../../glossary.md#shape) relationship produced by current [`GATHER`](../../../../glossary.md#gather). The [operation](../../../../glossary.md#operation) starts from a base `data`
[Tensor](../../../../glossary.md#tensor), maps the complete indices Shape back into one selected data [axis](../../../../glossary.md#term-axis), adds every update to its
addressed result [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix), and preserves the exact data Shape:

```text
data Shape    = data before axis + selected extent + data after axis
indices Shape = arbitrary index Shape
updates Shape = data before axis + complete indices Shape + data after axis
result Shape  = exact data Shape
```

This is a generally useful public functional update for repeated-index accumulation, bucketed
addition, [embedding](../../../../glossary.md#embedding)-table accumulation, and reversing a Gather selection. It closes the precise
dynamic-[extent](../../../../glossary.md#term-extent) gap proven by the
[adjoint expressibility audit](../adjoint-expressibility-audit.md)<span data-glossary-navigation="true"> ([adjoint glossary](../../../../glossary.md#term-adjoint))</span> without introducing a
`GATHER_BACKWARD` kind, a general scatter [dimension](../../../../glossary.md#dimension)-number language, or [compiler](../../../../glossary.md#term-compiler) behavior.

## [Scope](../../../../glossary.md#term-scope)

- Append exactly one `SCATTER_ADD` [constant](../../../../glossary.md#term-constant) to existing `AxisScatterKind`, preserving
  `SCATTER_ELEMENTS` as the first constant and preserving all existing behavior.
- Pair `SCATTER_ADD` with existing `IndexAxisAttrs` and the exact fixed three-input/one-output
  signature.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature))</span>
- Update `IndexAxisAttrs` [Javadoc](../../../../glossary.md#term-javadoc) only so it accurately covers [`GATHER`](../../../../glossary.md#gather), `GATHER_ELEMENTS`, and
  the new `SCATTER_ADD`; do not change its declaration, validation, components, or bytecode
  behavior.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [bytecode glossary](../../../../glossary.md#term-bytecode))</span>
- Add exactly one public [Tensor](../../../../glossary.md#tensor) method:

  ```java
  public Tensor scatterAdd(Tensor indices, Tensor updates, int axis)
  ```

- Extend existing package-private field-free `TensorAxisScatterExpressions`; add no helper type.
- Require exact INT32 or INT64 indices, exact matching data/update type, and current numeric data
  types BFLOAT16, FLOAT32, FLOAT64, INT32, or INT64; reject BOOL.<span data-glossary-navigation="true"> ([INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- Define exact Gather-compatible Shape validation, fixed addition, duplicate accumulation,
  metadata, failure order, provenance, and identifier effects.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance))</span>
- Preserve both existing `scatterElements` overloads, all `ScatterReduction` values, Scatter-ND,
  Gather, Gather Elements, Gather-ND, and embedding behavior unchanged.<span data-glossary-navigation="true"> ([Scatter-ND glossary](../../../../glossary.md#scatter-nd); [Gather Elements glossary](../../../../glossary.md#gather-elements); [Gather-ND glossary](../../../../glossary.md#gather-nd); [embedding glossary](../../../../glossary.md#embedding))</span>
- Update every exact public Tensor method-count inventory from 189 to 190.
- Finalize Javadocs, Tensor/Compile APIs, glossary, capabilities, and planning records through the
  mandatory separate clean-context documentation pass.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [planning glossary](../../../../glossary.md#term-planning))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- restoring historical `SCATTER_AXIS_ADD`, the removed reduced-[rank](../../../../glossary.md#term-rank) scatter meaning, a
  `GATHER_BACKWARD` kind, [aliases](../../../../glossary.md#aliasing-and-arena), migration shims, deprecated methods, or multiple spellings
- configurable replacement, MUL, MIN, or MAX behavior for the new Shape relationship; callers
  continue to use `scatterElements` or `scatterNd` where their respective [Shape](../../../../glossary.md#shape) contracts apply
- changing `SCATTER_ELEMENTS`, `ScatterElementsAttrs`, `ScatterReduction`, `SCATTER_ND`,
  `ScatterNdAttrs`, or their public methods, [validation](../../../../glossary.md#term-validation), numerical boundaries, or Shapes
- changing `AxisGatherKind`, `TensorAxisGatherExpressions`, Gather validation, embedding,
  Gather-ND, One Hot, Shape, Dimension, TensorDescriptor, TensorFactory, TensorProducer, or
  TensorProvenance<span data-glossary-navigation="true"> ([Gather glossary](../../../../glossary.md#gather); [embedding glossary](../../../../glossary.md#embedding); [Gather-ND glossary](../../../../glossary.md#gather-nd); [Dimension glossary](../../../../glossary.md#dimension); [TensorProducer glossary](../../../../glossary.md#producer--tensorproducer))</span>
- accepting BOOL data, promoting data/update types, converting updates, broadcasting updates,
  inferring an axis, adding a default axis, or accepting raw primitive indices<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [broadcasting glossary](../../../../glossary.md#broadcasting); [axis glossary](../../../../glossary.md#term-axis))</span>
- reading index or update values, checking concrete bounds, normalizing index values, detecting
  duplicates, allocating/copying storage, mutating the base Tensor, or executing addition<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- defining a broad JAX/XLA-style scatter dimension-number contract, coordinate grids, ranges,
  index tuples, batching attributes, builders, registries, factories, or another abstraction<span data-glossary-navigation="true"> ([coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- gradients, adjoint construction, compiler capture, graph traversal, planning, prepare, runtime,
  engine, backend, kernels, atomic implementation, execution ordering, or determinism guarantees<span data-glossary-navigation="true"> ([gradients glossary](../../../../glossary.md#term-gradient); [adjoint glossary](../../../../glossary.md#term-adjoint); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [graph glossary](../../../../glossary.md#graph); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [kernels glossary](../../../../glossary.md#kernel); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback))</span>
- dependencies, Gradle, architecture documents/tests, conformance/integration tests, another
  module, unrelated refactors, or later tasks<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance); [module glossary](../../../../glossary.md#term-module))</span>

## Exact semantic and public contract

### Final [axis](../../../../glossary.md#term-axis)-scatter vocabulary

`AxisScatterKind` contains exactly these [constants](../../../../glossary.md#term-constant) in this order:

```java
SCATTER_ELEMENTS,
SCATTER_ADD
```

The existing constant remains first so its enum order and [identity](../../../../glossary.md#term-identity-meanings) stay stable. The new final
`SCATTER_ADD` name is the additive functional counterpart of current [rank](../../../../glossary.md#term-rank)-changing [`GATHER`](../../../../glossary.md#gather):

```text
GATHER            IndexAxisAttrs   [data, indices]          -> gathered values
SCATTER_ADD       IndexAxisAttrs   [data, indices, updates] -> data-shaped accumulated result

GATHER_ELEMENTS   IndexAxisAttrs   [data, indices]          -> aligned gathered values
SCATTER_ELEMENTS  ScatterElementsAttrs
                                  [data, indices, updates]  -> data-shaped configured result
```

This deliberately does not restore historical `SCATTER_AXIS_ADD`. Before task 0018O, that name
was paired with the [operation](../../../../glossary.md#operation) then called `GATHER_AXIS`; current task 0018O renamed that selected
public gather meaning to final `GATHER`. Historical `SCATTER_ADD` instead belonged to a removed
reduced-rank gather meaning. The new contract reuses the concise public name only for the final
Gather-compatible meaning, with no compatibility [alias](../../../../glossary.md#aliasing-and-arena) or claim that the removed semantics
return.

Family-owned [signatures](../../../../glossary.md#term-signature) are exact:

```text
SCATTER_ELEMENTS  ScatterElementsAttrs  3 inputs, 1 output
SCATTER_ADD       IndexAxisAttrs         3 inputs, 1 output
```

`SCATTER_ADD` accepts no `ScatterElementsAttrs`; `SCATTER_ELEMENTS` accepts no `IndexAxisAttrs`.
`OperationSignatureTest` must lock the two variants separately rather than treating the enum as a
single-signature family.

### Public method and helper surface

[`Tensor`](../../../../glossary.md#tensor) adds exactly:

```java
public Tensor scatterAdd(Tensor indices, Tensor updates, int axis) {
    return TensorAxisScatterExpressions.scatterAdd(this, indices, updates, axis);
}
```

It adds no overload, reduction [argument](../../../../glossary.md#term-argument), static form, [target](../../../../glossary.md#term-target) [Shape](../../../../glossary.md#shape), target Tensor, raw indices,
[alias](../../../../glossary.md#aliasing-and-arena), or convenience spelling.

`TensorAxisScatterExpressions` remains one final package-private non-record class with no fields,
no nested types, and one private zero-argument constructor. Its exact methods become:

```java
static Tensor scatterAdd(Tensor data, Tensor indices, Tensor updates, int axis)
static Tensor scatterElements(Tensor data, Tensor indices, Tensor updates, int axis)
static Tensor scatterElements(
        Tensor data,
        Tensor indices,
        Tensor updates,
        int axis,
        ScatterReduction reduction)
private static void validateIndexType(
        String operation, TensorDescriptor indicesDescriptor)
private static void validateMatchingDataType(
        String operation,
        TensorDescriptor dataDescriptor,
        TensorDescriptor updatesDescriptor)
private static void validateAddDataType(TensorDescriptor dataDescriptor)
private static Shape gatherResultShape(
        Shape dataShape, Shape indicesShape, int normalizedAxis)
private static void validateScatterElementsShape(
        Shape dataShape,
        Shape indicesShape,
        Shape updatesShape,
        int normalizedAxis)
private static Tensor create(
        Tensor data,
        Tensor indices,
        Tensor updates,
        TensorDescriptor dataDescriptor,
        TensorDescriptor updatesDescriptor,
        Operation operation)
```

The existing common `create` method changes only its private construction seam from separate
kind/attributes [parameters](../../../../glossary.md#term-parameter) to one already validated exact [`Operation`](../../../../glossary.md#operation), allowing both typed
variants without accepting a broad public attrs input. Both existing `scatterElements` paths
retain their current delegation and behavior.

### [Gather](../../../../glossary.md#gather)-compatible [Shape](../../../../glossary.md#shape) relationship

Let data [rank](../../../../glossary.md#term-rank) be `R`, [normalized axis](../../../../glossary.md#normalized-axis) be `A`, and indices rank be `Q`. The required updates rank is
`R - 1 + Q`, and its [Dimensions](../../../../glossary.md#dimension) are exactly:

```text
data.dimensions[0:A]
    + indices.dimensions[0:Q]
    + data.dimensions[A + 1:R]
```

Every Dimension reference used to construct the expected Shape is the exact corresponding data or
indices Dimension reference. [Validation](../../../../glossary.md#term-validation) compares the complete updates Shape structurally with
this expected Shape; it does not broadcast, bind symbols, compare [element](../../../../glossary.md#term-element) counts, or replace
Dimensions.

Examples:

```text
data     [2, 3, 4]
axis      1
indices  [5, 6]
updates  [2, 5, 6, 4]
result   [2, 3, 4]
```

```text
data     [batch, vocabulary, width]
axis      1
indices  [tokens]
updates  [batch, tokens, width]
result   exact data Shape
```

[Scalar](../../../../glossary.md#term-scalar) indices remove the selected [axis](../../../../glossary.md#term-axis) from the updates Shape. Zero-element and unresolved
indices Shapes are accepted when the exact structural updates Shape matches. An unresolved or
zero selected data [extent](../../../../glossary.md#term-extent) is also representable because construction does not inspect index
values; eventual valid execution still requires every index to be in bounds. No special case
returns `data`, including an updates domain known to be empty.

### [Coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) and addition meaning

For an update coordinate split as:

```text
[dataPrefixCoordinates, indicesCoordinates, dataSuffixCoordinates]
```

the corresponding result coordinate is:

```text
[dataPrefixCoordinates,
 indices[indicesCoordinates],
 dataSuffixCoordinates]
```

The conceptual result begins with every base `data` value. Each update is added to its addressed
coordinate. Updates whose index values select the same result coordinate all accumulate with the
base value. This fixed addition is intrinsic to `SCATTER_ADD`; no reduction value is stored or
accepted.

[INT32](../../../../glossary.md#data-type--datatype) and [INT64](../../../../glossary.md#data-type--datatype) addition retains exact type and uses the selected [fixed-width modular arithmetic](../../../../glossary.md#arithmetic-rearrangement-and-finite-difference-checks)
contract. Floating addition retains exact type, permits [reassociation](../../../../glossary.md#term-reassociation) of a duplicate group, and
does not promise a [NaN payload](../../../../glossary.md#nan-infinity-and-signed-zero), intermediate [precision](../../../../glossary.md#term-precision), traversal order, bitwise result,
cross-[backend](../../../../glossary.md#backend) determinism, or a particular [atomic](../../../../glossary.md#resource-transaction-and-rollback) algorithm. An empty updates domain leaves every
base value mathematically unchanged. [Model](../../../../glossary.md#term-model) construction records this meaning but reads and adds no
values.

Index values are not normalized during model construction. Eventual valid execution requires each
value to be in bounds for the selected data [extent](../../../../glossary.md#term-extent); negative and out-of-range values are invalid
rather than wrapped, clamped, or ignored. Bounds and duplicate grouping are value-aware later
obligations, while duplicate indices are semantically valid because addition combines them.

### [Validation](../../../../glossary.md#term-validation) order and diagnostics

`scatterAdd` performs local validation in exactly this order:

1. `Objects.requireNonNull(data, "data")`;
2. `Objects.requireNonNull(indices, "indices")`;
3. `Objects.requireNonNull(updates, "updates")`;
4. read [descriptors](../../../../glossary.md#term-descriptor) in data, indices, updates order;
5. require indices type [INT32](../../../../glossary.md#data-type--datatype) or [INT64](../../../../glossary.md#data-type--datatype);
6. require exact data/update type equality;
7. reject [BOOL](../../../../glossary.md#data-type--datatype) data;
8. read exact data [Shape](../../../../glossary.md#shape) and normalize the raw [axis](../../../../glossary.md#term-axis) exactly once through `Shape.normalizeAxis`;
9. derive the expected Gather result Shape exactly once from data, indices Shape, and normalized
   axis;<span data-glossary-navigation="true"> ([Gather glossary](../../../../glossary.md#gather); [normalized axis glossary](../../../../glossary.md#normalized-axis))</span>
10. require exact [structural equality](../../../../glossary.md#term-structural-equality) with updates Shape; and
11. construct `IndexAxisAttrs`, [`Operation`](../../../../glossary.md#operation), result descriptor, [producer](../../../../glossary.md#term-producer), [provenance](../../../../glossary.md#provenance), and [Tensor](../../../../glossary.md#tensor).

Exact failures are:

```text
scatterAdd indices data type must be INT32 or INT64: <dataType>
scatterAdd updates data type must match data: expected=<dataType>, actual=<dataType>
scatterAdd data type must be numeric: BOOL
scatterAdd updates shape must match gather result shape: expected=<shape>, actual=<shape>
```

Axis failure remains the exact `Shape.normalizeAxis` `IndexOutOfBoundsException`. Type failures
precede axis failure; axis failure precedes updates-Shape failure. Every local failure occurs
before `TensorFactory.createDerived` and consumes no Tensor identifier.

### [Descriptor](../../../../glossary.md#term-descriptor), [producer](../../../../glossary.md#term-producer), [provenance](../../../../glossary.md#provenance), and identifiers

Success creates in this order:

1. one `IndexAxisAttrs` with the normalized data [axis](../../../../glossary.md#term-axis);
2. one `Operation(AxisScatterKind.SCATTER_ADD, attrs)`;
3. one unresolved `TensorDescriptor` retaining the exact data type and exact data Shape reference,
   with <span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype); [Shape glossary](../../../../glossary.md#shape))</span>`requiresGrad` equal to data/update eligibility OR;
4. one producer with exact ordered inputs `[data, indices, updates]` and one output descriptor; and
5. output-index-zero provenance plus one fresh unlabeled, storage-free [Tensor](../../../../glossary.md#tensor).

Integral descriptors remain non-[differentiable](../../../../glossary.md#derivative-and-chain-rule) under their existing [invariant](../../../../glossary.md#term-invariant). Index eligibility
never contributes. The input Tensors are unchanged. Every successful call creates a fresh producer
and consumes one Tensor identifier, even for empty updates or a zero-valued base. Current
identifier-exhaustion behavior remains unchanged and no partial state is rolled back.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Roadmap](../../../roadmap.md)
- [Capabilities](../capabilities.md)
- [Master plan](../master-plan.md)
- [Adjoint expressibility audit](../adjoint-expressibility-audit.md)<span data-glossary-navigation="true"> ([Adjoint glossary](../../../../glossary.md#term-adjoint))</span>
- [Task 0018C](0018c-axis-gather-semantics.md)
- [Task 0018D](0018d-axis-gather-tensor-expressions.md)
- [Task 0018G](0018g-axis-scatter-semantics.md)
- [Task 0018H](0018h-axis-scatter-tensor-expressions.md)
- [Task 0018O](0018o-indexing-taxonomy-and-unstack-normalization.md)
- [Task 0018U](0018u-integral-elementwise-arithmetic-and-comparisons.md)
- [Task 0023](0023-adjoint-expressibility-audit.md)
- [Task 0023A](0023a-binding-aware-sum-to-shape.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Glossary](../../../../glossary.md)

## Architecture constraints

- Work stays inside model-owned backend-independent indexing semantics, Tensor metadata,
  pre-capture producer/provenance construction, and directly affected docs/planning files.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [backend glossary](../../../../glossary.md#backend); [Tensor glossary](../../../../glossary.md#tensor); [metadata glossary](../../../../glossary.md#term-metadata); [capture glossary](../../../../glossary.md#term-capture-meanings); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance); [planning glossary](../../../../glossary.md#term-planning))</span>
- `SCATTER_ADD` means functional mathematical accumulation only. It stores no Tensor, Shape,
  bound index, duplicate group, algorithm, atomicity, backend support, kernel, execution route, or
  runtime state.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [atomicity glossary](../../../../glossary.md#term-atomicity); [kernel glossary](../../../../glossary.md#kernel))</span>
- Model construction owns only locally provable descriptor/type/Shape validation. Compiler or
  later concrete binding and execution own value bounds and executable realization.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [validation glossary](../../../../glossary.md#term-validation); [Compiler glossary](../../../../glossary.md#term-compiler); [binding glossary](../../../../glossary.md#term-binding))</span>
- Compiler later may use this public semantic primitive while constructing Gather adjoints; this
  task adds no gradient rule, traversal, graph capture, or compiler behavior.<span data-glossary-navigation="true"> ([Gather glossary](../../../../glossary.md#gather); [gradient glossary](../../../../glossary.md#term-gradient); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction))</span>
- [Runtime](../../../../glossary.md#term-runtime) hot paths must not consume [`Operation`](../../../../glossary.md#operation) or `CompiledNode`; backend prepare owns lowering,
  specialization, fusion, and concrete accumulation implementation.<span data-glossary-navigation="true"> ([prepare glossary](../../../../glossary.md#prepare); [lowering glossary](../../../../glossary.md#lowering); [specialization glossary](../../../../glossary.md#term-specialization); [fusion glossary](../../../../glossary.md#term-fusion))</span>
- No architecture, ADR, architecture-test, dependency, Gradle, cross-module, compiler, runtime,
  prepare, backend, training, conformance, or integration change is authorized. Stop if current
  index attributes, signatures, descriptors, producers, or Shapes cannot express this contract.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module); [conformance glossary](../../../../glossary.md#term-conformance); [signatures glossary](../../../../glossary.md#term-signature))</span>

## Package impact

Existing packages changed:

- `io.github.pho001.synaptik.model.operation.index`
- `io.github.pho001.synaptik.model.tensor`

No package or public type is added, moved, or renamed.

Type placement:

- `...operation.index.AxisScatterKind` owns both selected one-[axis](../../../../glossary.md#term-axis) functional scatter [identities](../../../../glossary.md#term-identity-meanings).
- `...operation.index.IndexAxisAttrs` remains the minimal normalized-axis value shared by Gather,
  Gather Elements, and fixed Gather-compatible scatter-add.<span data-glossary-navigation="true"> ([Gather glossary](../../../../glossary.md#gather); [Gather Elements glossary](../../../../glossary.md#gather-elements))</span>
- `...tensor.TensorAxisScatterExpressions` remains the single package-private owner of local axis-
  scatter expression construction.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
- `...tensor.Tensor` remains the public fluent [facade](../../../../glossary.md#term-facade).

## Affected files

Production (4):

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/AxisScatterKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/IndexAxisAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorAxisScatterExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Tests (15):

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/OperationSignatureTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/index/AxisScatterSemanticsTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorAxisScatterExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBatchNormInferenceExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryArithmeticTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorDenseCategoricalCrossEntropyWithLogitsExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorIndexCategoricalCrossEntropyWithLogitsExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorLayerNormExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorLinearExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMatmulExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMeanSquaredErrorExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorRmsNormExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorScaledDotProductAttentionExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorSumToShapeExpressionTest.java`

Documentation/[planning](../../../../glossary.md#term-planning) (7):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review unchanged unless inaccurate: [Axis](../../../../glossary.md#term-axis) [Gather](../../../../glossary.md#gather), [Gather-ND](../../../../glossary.md#gather-nd), [Scatter-ND](../../../../glossary.md#scatter-nd), Scatter reduction,
[descriptor](../../../../glossary.md#term-descriptor)/factory/[producer](../../../../glossary.md#term-producer)/[provenance](../../../../glossary.md#provenance), [Shape](../../../../glossary.md#shape), arithmetic, Training/[Runtime](../../../../glossary.md#term-runtime) APIs,
architecture/ADRs/tests, [conformance](../../../../glossary.md#term-conformance)/integration, [Gradle](../../../../glossary.md#term-gradle), dependencies, other [modules](../../../../glossary.md#term-module), and later
tasks.

## Maximum [scope](../../../../glossary.md#term-scope)

Exactly 26 paths: four production, fifteen tests, and seven documentation/[planning](../../../../glossary.md#term-planning) paths. The
cohesive capability exceeds the usual 18-path guardrail under the user's standing automatic
higher-path authorization because one semantic [constant](../../../../glossary.md#term-constant), one [facade](../../../../glossary.md#term-facade)/helper path, twelve existing
global [Tensor](../../../../glossary.md#tensor)-count locks, two focused family tests, the global [signature](../../../../glossary.md#term-signature) [matrix](../../../../glossary.md#term-matrix-meanings), and mandatory
documentation must change atomically. Stop for path 27, another public type/method, another
helper/test/document, an existing [Gather](../../../../glossary.md#gather)/[Scatter-ND](../../../../glossary.md#scatter-nd) contract edit, later-task work, cross-[module](../../../../glossary.md#term-module)
work, architecture/[Gradle](../../../../glossary.md#term-gradle) change, or unrelated cleanup. If live repository evidence changes an
inventory path, update this [Ready](../../../../glossary.md#term-ready) task before implementation without exceeding 26.

## [Javadoc](../../../../glossary.md#term-javadoc) and documentation requirements

- Finalize complete Javadocs for both `AxisScatterKind` [constants](../../../../glossary.md#term-constant), `IndexAxisAttrs`, the helper and
  all its methods, and public `Tensor.scatterAdd`, including Shape/coordinate meaning, duplicate
  addition, numeric domain, metadata, provenance/IDs, failures, and lifecycle ownership.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance); [lifecycle glossary](../../../../glossary.md#lifecycle); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Apply General, [API](../../../../glossary.md#term-application-programming-interface-api)/Javadoc, [Planning](../../../../glossary.md#term-planning), and Example profiles as relevant, with complete `@param`,
  `@return`, and expected `@throws` text.
- Tensor API moves Gather-compatible scatter-add from planned to current, adds exact signature,
  static/dynamic/scalar/empty examples, coordinate and type rules, distinction from Scatter
  Elements/Scatter-ND, producer metadata, and lifecycle boundary.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Gather glossary](../../../../glossary.md#gather); [signature glossary](../../../../glossary.md#term-signature); [scalar glossary](../../../../glossary.md#term-scalar); [Scatter Elements glossary](../../../../glossary.md#scatter-elements); [Scatter-ND glossary](../../../../glossary.md#scatter-nd); [producer glossary](../../../../glossary.md#term-producer))</span>
- Compile API records the current model semantic and later bounds/adjoint obligation without
  claiming graph capture, autograd use, lowering, or execution is implemented.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [model glossary](../../../../glossary.md#term-model); [adjoint glossary](../../../../glossary.md#term-adjoint); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [autograd glossary](../../../../glossary.md#autograd); [lowering glossary](../../../../glossary.md#lowering))</span>
- Glossary adds [`Scatter Add`](../../../../glossary.md#scatter-add), cross-links Gather and Scatter Elements, records the exact Shape
  formula, and updates `IndexAxisAttrs` and [operation](../../../../glossary.md#operation)-family inventories.
- Capabilities/task/master/roadmap remain synchronized: 0023–0023B become/stay Complete only after
  implementation, 0023C is the next Draft frontier without a detailed specification, and
  0023D–0024 remain Draft.
- Use official [ONNX Gather](https://onnx.ai/onnx/operators/onnx__Gather.html)<span data-glossary-navigation="true"> ([ONNX glossary](../../../../glossary.md#term-onnx))</span>,
  [ONNX Scatter Elements](https://onnx.ai/onnx/operators/onnx__ScatterElements.html),
  [PyTorch `scatter_add`](https://docs.pytorch.org/docs/stable/generated/torch.scatter_add.html),
  and [JAX `lax.scatter_add`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scatter_add.html)
  only as terminology/comparison evidence. Explain that ONNX/PyTorch aligned scatter differs,
  while JAX's general dimension-number API is intentionally broader than this focused contract.<span data-glossary-navigation="true"> ([dimension glossary](../../../../glossary.md#dimension))</span>
- Record reasoned no-change conclusions for related existing Javadocs, Training/Runtime APIs,
  architecture/ADRs/tests, dependencies, Gradle, conformance/integration, other modules, and later
  tasks.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance); [modules glossary](../../../../glossary.md#term-module))</span>

## Acceptance criteria

- `AxisScatterKind` has exactly `SCATTER_ELEMENTS`, then appended `SCATTER_ADD`; no historical
  `SCATTER_AXIS_ADD`, backward kind, [alias](../../../../glossary.md#aliasing-and-arena), field, nested type, or other [constant](../../../../glossary.md#term-constant).
- Existing [Scatter Elements](../../../../glossary.md#scatter-elements) [signature](../../../../glossary.md#term-signature) and behavior remain exact. `SCATTER_ADD` alone accepts exact
  `IndexAxisAttrs` with three inputs and one output; exact cross-family attrs rejection is tested.
- `IndexAxisAttrs` declaration/validation/value behavior remains bytecode-equivalent apart from
  Javadoc and now accurately documents the new pairing.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [bytecode glossary](../../../../glossary.md#term-bytecode); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Exactly one new public `scatterAdd(Tensor, Tensor, int)` method; public Tensor count is 190; no
  overload, reduction parameter, static method, raw index, target Shape, or alias.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [parameter glossary](../../../../glossary.md#term-parameter); [target glossary](../../../../glossary.md#term-target); [Shape glossary](../../../../glossary.md#shape))</span>
- Axis-scatter helper remains field-free with the exact nine-method surface and private
  constructor; no new helper/type or Gather-helper dependency.<span data-glossary-navigation="true"> ([Gather glossary](../../../../glossary.md#gather); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Exact null/index-type/update-type/numeric/axis/Shape validation order, messages, exception types,
  and no-ID local failures.
- Expected updates Shape is exactly data prefix plus complete indices Shape plus data suffix;
  static, zero, scalar, named/expression, and unresolved Dimensions preserve the selected
  structural contract without broadcasting or binding.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [Dimensions glossary](../../../../glossary.md#dimension); [broadcasting glossary](../../../../glossary.md#broadcasting); [binding glossary](../../../../glossary.md#term-binding))</span>
- Fixed addition starts from data and accumulates every addressed update, including duplicates;
  integral modular and floating non-bitwise boundaries are documented; no value work occurs.
- Result retains exact data Shape/type, data/update eligibility OR, unresolved layout, no
  label/storage, exact <span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout))</span>`SCATTER_ADD`/[axis](../../../../glossary.md#term-axis) attrs, ordered `[data, indices, updates]` producer,
  output index zero, one fresh ID, and unchanged inputs.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer))</span>
- Existing Scatter Elements, Gather families, Scatter-ND, count locks, and operation-signature
  matrix remain covered without unrelated behavior changes.<span data-glossary-navigation="true"> ([Scatter-ND glossary](../../../../glossary.md#scatter-nd); [operation glossary](../../../../glossary.md#operation); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- Exact 26-path/package scope; no compiler, execution, architecture, dependency, build, or later-
  task work.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- Separate clean documentation-focused pass and all required validation/evidence complete before
  status Complete.

## Tests / [validation](../../../../glossary.md#term-validation)

Focused implementation command:

```bash
./gradlew :modules:model:test \
  --tests io.github.pho001.synaptik.model.operation.OperationSignatureTest \
  --tests io.github.pho001.synaptik.model.operation.index.AxisScatterSemanticsTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorAxisScatterExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorBatchNormInferenceExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorBinaryArithmeticTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorDenseCategoricalCrossEntropyWithLogitsExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorIndexCategoricalCrossEntropyWithLogitsExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorLayerNormExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorLinearExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorMatmulExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorMeanSquaredErrorExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorRmsNormExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorScaledDotProductAttentionExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorSumToShapeExpressionTest
```

After executable Java stabilizes, [run](../../../../glossary.md#term-run) exactly one final [model](../../../../glossary.md#term-model) suite:

```bash
./gradlew :modules:model:test
```

Focused coverage must verify enum order and exact [signatures](../../../../glossary.md#term-signature), cross-attrs rejection, public/helper
surfaces, method count 190, static/dynamic/expression/[scalar](../../../../glossary.md#term-scalar)/zero indices [Shapes](../../../../glossary.md#shape), exact expected
updates Shape, first deterministic mismatch, all numeric types, [BOOL](../../../../glossary.md#data-type--datatype)/index/update-type failures,
[axis](../../../../glossary.md#term-axis) [normalization](../../../../glossary.md#term-normalization)/failure, null order, no-ID failures, duplicate-add semantics as [metadata](../../../../glossary.md#term-metadata),
[descriptor](../../../../glossary.md#term-descriptor), [operation](../../../../glossary.md#operation), [producer](../../../../glossary.md#term-producer), [provenance](../../../../glossary.md#provenance), storage/label absence, freshness, and unchanged inputs.
Tests inspect metadata only and do not claim value execution.

Documentation pass after final Javadocs:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

Also validate Markdown links/anchors/fences/final newlines/trailing whitespace, official
references, examples, exact 26 paths, package placement, signatures/count 190, synchronized
status, 0023C Draft/no spec, absence of removed `SCATTER_AXIS_ADD`, and absence of [compiler](../../../../glossary.md#term-compiler)/[runtime](../../../../glossary.md#term-runtime)/
[backend](../../../../glossary.md#backend)/build changes. Reuse successful Java evidence unless executable Java changes.

Repository-wide validation is deferred to the capability [checkpoint](../../../../glossary.md#term-checkpoint) after task 0023F and to [CI](../../../../glossary.md#term-continuous-integration-ci);
this single-[module](../../../../glossary.md#term-module) task changes no [dependency](../../../../glossary.md#term-dependency), build, or architecture boundary.

## Dependencies

- 0001–0002: numeric types and [immutable](../../../../glossary.md#term-immutable) static/symbolic [Shape](../../../../glossary.md#shape)/[Dimension](../../../../glossary.md#dimension) contracts.
- 0005–0007 and 0011–0013: typed operations, descriptors, Tensor, identity, factory, and
  provenance.<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- 0018C–0018D: final [Gather](../../../../glossary.md#gather) Shape and public expression contracts.
- 0018G–0018H: [axis](../../../../glossary.md#term-axis)-scatter semantic vocabulary, reduction semantics, and shared helper behavior.
- 0018K: exact family-owned [signatures](../../../../glossary.md#term-signature); 0018L: [producer](../../../../glossary.md#term-producer)/output-index [provenance](../../../../glossary.md#provenance).
- 0018O: final indexing taxonomy and removal of provisional fixed-add meanings.
- 0018U: selected signed-integral arithmetic semantics.
- 0023: completed audit proving the unresolved-Gather-extent gap and selecting a general public
  primitive rather than a backward kind.<span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent))</span>
- 0023A: completed first audit-selected prerequisite and current public [Tensor](../../../../glossary.md#tensor) count baseline.

All dependencies are Complete.

## Follow-up tasks

- 0023C remains the next Draft frontier for signed slice placement plus target-relative dynamic
  crop; do not create its detailed specification during this task.<span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice); [target glossary](../../../../glossary.md#term-target))</span>
- 0023D–0023F remain concise Draft public-capability rows.
- [Compiler](../../../../glossary.md#term-compiler)/[autograd](../../../../glossary.md#autograd) work later may use `SCATTER_ADD` for Gather/embedding data adjoints and owns
  graph construction, bounds proof, accumulation, canonicalization, and optimization.<span data-glossary-navigation="true"> ([Gather glossary](../../../../glossary.md#gather); [embedding glossary](../../../../glossary.md#embedding); [graph glossary](../../../../glossary.md#graph); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form))</span>
- Task 0024 remains Draft and depends through completed 0023F.

## Architecture impact

Expected impact: None. Stop if implementation needs general scatter [dimension](../../../../glossary.md#dimension) numbers, another
attributes/helper type, [Shape](../../../../glossary.md#shape)/[binding](../../../../glossary.md#term-binding) changes, [compiler](../../../../glossary.md#term-compiler)/[prepare](../../../../glossary.md#prepare)/[backend](../../../../glossary.md#backend) contracts, [dependency](../../../../glossary.md#term-dependency),
architecture update, or work outside the exact [model](../../../../glossary.md#term-model) and documentation [scope](../../../../glossary.md#term-scope).

## Implementation prompt

Use this prompt in a separate clean-context task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, documentation/planning rules, model capabilities/master plan,
roadmap, completed tasks 0018C–0018D/0018G–0018H/0018K–0018L/0018O/0018U/0023–0023A, task
0023B, and every affected/review-only source/test named there in full.

Implement task 0023B exactly inside its 26 authorized paths. Add only final Gather-compatible
`SCATTER_ADD` and one public `Tensor.scatterAdd(indices, updates, axis)` expression. Preserve
Scatter Elements, Scatter-ND, Gather families, Shapes, producer/provenance, and every architecture
boundary. Add no historical alias, configurable reduction, general scatter language, compiler
adoption, value execution, or later task. Stop on architecture, dependency, completed-contract,
validation-order, affected-file, or maximum-scope conflict.

Run focused tests while developing and exactly one final model suite after executable Java
stabilizes. Then hand the actual diff and exact Java evidence to a separate clean-context
documentation-focused agent in the same overall change. That agent must independently inspect
final source/tests, finalize Javadocs, Tensor/Compile APIs, glossary, capability/task/master/
roadmap status and documentation validation, and reuse successful Java evidence unless executable
behavior changes or it records a concrete reason.

Do not mark 0023B Complete until both passes and every acceptance criterion succeed. Leave 0023C
and every later task Draft without a detailed specification.
```

## Documentation-agent [handoff](../../../../glossary.md#term-handoff)

The implementation agent must hand over this task, actual implementation/test diff, exact Java
evidence, final naming/history distinction, kind/[signature](../../../../glossary.md#term-signature) change, [Gather](../../../../glossary.md#gather)-compatible [Shape](../../../../glossary.md#shape) and
coordinate rules, numeric/duplicate/bounds boundaries, [validation](../../../../glossary.md#term-validation) and ID order, [metadata](../../../../glossary.md#term-metadata)/
[provenance](../../../../glossary.md#provenance) behavior, architecture constraints, expected [Tensor](../../../../glossary.md#tensor)/[Compile](../../../../glossary.md#compile) [API](../../../../glossary.md#term-application-programming-interface-api) and glossary changes,
official comparison links, existing-[Javadoc](../../../../glossary.md#term-javadoc) review list, and every documentation/[scope](../../../../glossary.md#term-scope)/status
command. The documentation agent must inspect final source and tests rather than rely on the
summary.

## Local decisions

- Appended final `SCATTER_ADD` after `SCATTER_ELEMENTS` and paired it only with unchanged
  `IndexAxisAttrs`; the historical `SCATTER_AXIS_ADD` and removed reduced-rank meaning remain
  absent.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank))</span>
- Added exactly one public `scatterAdd(Tensor indices, Tensor updates, int axis)` method and kept
  all axis-scatter construction in the existing field-free helper. The helper has exactly nine
  methods plus its private constructor.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis))</span>
- Derived the required updates Shape locally from exact Dimension references rather than calling
  the Gather helper, binding symbols, or introducing a general scatter-dimension mapping.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [Dimension glossary](../../../../glossary.md#dimension); [Gather glossary](../../../../glossary.md#gather); [binding glossary](../../../../glossary.md#term-binding))</span>
- Kept fixed addition intrinsic to the kind. There is no reduction attribute or overload, and
  every duplicate target includes the base value plus all addressed updates.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>

## Known limitations

- Model construction is descriptor-only: it does not read indices or updates, enforce eventual
  index bounds, perform addition, choose accumulation order, construct gradients, capture a graph,
  lower to a backend, or execute work.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [descriptor glossary](../../../../glossary.md#term-descriptor); [gradients glossary](../../../../glossary.md#term-gradient); [capture glossary](../../../../glossary.md#term-capture-meanings); [graph glossary](../../../../glossary.md#graph); [backend glossary](../../../../glossary.md#backend))</span>
- Signed-integral addition is fixed-width modular. Floating addition may be reassociated and has no
  bitwise-order guarantee.
- Compiler/autograd adoption for Gather or embedding adjoints remains later work, as do value-aware
  bounds enforcement and all compiler, prepare, runtime, and backend support.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [autograd glossary](../../../../glossary.md#autograd); [Gather glossary](../../../../glossary.md#gather); [embedding glossary](../../../../glossary.md#embedding); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Tasks 0023C–0023F and 0024 remain Draft; no 0023C detailed specification was created.

## [Validation](../../../../glossary.md#term-validation) evidence

- The focused implementation command passed 124 tests across 15 suites with zero failures,
  errors, or skips.
- After executable Java stabilized, the single final `./gradlew :modules:model:test` run passed
  981 tests across 125 suites with zero failures, errors, or skips. The separate documentation
  pass changed Javadocs and Markdown only, so it reused this evidence without rerunning Java tests.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
- `./gradlew :modules:model:javadoc` completed successfully after final Javadocs.
- The complete Scatter Add Java example compiled and ran against the built model classes. It
  produced the documented static, symbolic, scalar-index, and zero-index Shapes, normalized
  operation attributes, and exact ordered provenance checks.<span data-glossary-navigation="true"> ([Scatter Add glossary](../../../../glossary.md#scatter-add); [model glossary](../../../../glossary.md#term-model); [scalar glossary](../../../../glossary.md#term-scalar); [Shapes glossary](../../../../glossary.md#shape); [operation glossary](../../../../glossary.md#operation); [provenance glossary](../../../../glossary.md#provenance))</span>
- Reflection and `javap` confirmed exactly 190 declared public [Tensor](../../../../glossary.md#tensor) methods, the sole exact
  `scatterAdd(Tensor, Tensor, int)` [signature](../../../../glossary.md#term-signature), enum order `[SCATTER_ELEMENTS, SCATTER_ADD]`, exact
  fixed signatures, nine helper methods, zero helper fields, and unchanged one-component
  `IndexAxisAttrs`.
- Markdown local links and anchors, balanced fences, official references, final newlines,
  synchronized status, 0023C Draft/no specification, and `git diff --check` passed. `git status`
  confirmed the exact authorized 26 paths: four production, fifteen tests, and seven
  documentation/planning paths.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Source/package and path inspection confirmed no `SCATTER_AXIS_ADD` production symbol and no
  compiler, runtime, backend, architecture, ADR, dependency, Gradle, conformance, integration,
  other-module, or later-task change. Repository-wide tests remain deferred to the 0023F
  capability checkpoint and CI as specified.<span data-glossary-navigation="true"> ([symbol glossary](../../../../glossary.md#term-symbol); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance); [module glossary](../../../../glossary.md#term-module); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [CI glossary](../../../../glossary.md#term-continuous-integration-ci))</span>

## Implementation notes

- `AxisScatterKind` now has exact [signatures](../../../../glossary.md#term-signature) `SCATTER_ELEMENTS`/`ScatterElementsAttrs` and
  `SCATTER_ADD`/`IndexAxisAttrs`, each fixed at three inputs and one output. Cross-family
  attributes remain invalid.
- `TensorAxisScatterExpressions.scatterAdd` checks nulls, index type, exact update type, numeric
  domain, normalized axis, and exact Gather-compatible Shape in the specified order before final
  identity allocation.<span data-glossary-navigation="true"> ([normalized axis glossary](../../../../glossary.md#normalized-axis); [Gather glossary](../../../../glossary.md#gather); [Shape glossary](../../../../glossary.md#shape); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Valid results retain exact data Shape/type, combine data/update gradient eligibility, leave
  layout unresolved, omit label/storage, and record one fresh output-index-zero producer with
  ordered exact inputs <span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [layout glossary](../../../../glossary.md#layout); [producer glossary](../../../../glossary.md#term-producer))</span>`[data, indices, updates]`.
- The separate documentation-focused pass reviewed and finalized the four affected Javadocs,
  Tensor and Compile API references, glossary, capability baseline, task, master plan, and roadmap.
  Existing Gather, Gather-ND, Scatter-ND, Shape, descriptor/factory/producer/provenance,
  Training/Runtime API, architecture/ADR/test, conformance/integration, Gradle, dependency, other-
  module, and later-task contracts remained accurate and required no changes.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Gather-ND glossary](../../../../glossary.md#gather-nd); [Scatter-ND glossary](../../../../glossary.md#scatter-nd); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [Runtime glossary](../../../../glossary.md#term-runtime); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [module glossary](../../../../glossary.md#term-module))</span>

## Completion summary

- Added final [Gather](../../../../glossary.md#gather)-compatible `SCATTER_ADD` semantics and one public `Tensor.scatterAdd` metadata
  expression with exact validation, Shape, numeric, duplicate-accumulation, result, identity, and
  provenance contracts.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [validation glossary](../../../../glossary.md#term-validation); [Shape glossary](../../../../glossary.md#shape); [identity glossary](../../../../glossary.md#term-identity-meanings); [provenance glossary](../../../../glossary.md#provenance))</span>
- Extended the existing axis-scatter helper and focused semantic/expression/signature tests while
  preserving all existing Scatter Elements and related indexing behavior.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [signature glossary](../../../../glossary.md#term-signature); [Scatter Elements glossary](../../../../glossary.md#scatter-elements))</span>
- Finalized all affected Javadocs, Tensor and Compile API references, glossary terminology,
  capability baseline, task evidence, master plan, and roadmap. No architecture documentation,
  tests outside the model scope, or other API documentation required a change.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [model glossary](../../../../glossary.md#term-model); [scope glossary](../../../../glossary.md#term-scope))</span>
- All task-level implementation and documentation validation passed; no unresolved issue or
  follow-up is required for 0023B. Tasks 0023C–0023F and 0024 remain Draft.

Status: Complete
