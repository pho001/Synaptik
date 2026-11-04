# Task 0023A: [Binding](../../../../glossary.md#term-binding)-Aware [Sum-to-Shape](../../../../glossary.md#sum-to-shape)

## Status

Complete

## Goal

Add one [backend](../../../../glossary.md#backend)-independent public reduction form that sums a numeric [Tensor](../../../../glossary.md#tensor) to an exact [target](../../../../glossary.md#term-target)
[`Shape`](../../../../glossary.md#shape). The target Shape describes the result directly, while future [binding](../../../../glossary.md#term-binding) determines which
right-aligned [axes](../../../../glossary.md#term-axis) were [singleton](../../../../glossary.md#term-singleton)-expanded and therefore require summation.

```text
numeric input + target Shape
  -> AggregateReductionKind.SUM + SumToShapeAttrs(targetShape)
     -> exact target Shape, type, eligibility, and one-input provenance
        -> later proof: every aligned target extent is one or equals the input extent
```

This is a generally useful public transformation, not a [MATMUL](../../../../glossary.md#matrix-multiplication--matmul)-, attention-, [gradient](../../../../glossary.md#term-gradient)-, or
backend-specific unbroadcast [operation](../../../../glossary.md#operation). It closes only the [model](../../../../glossary.md#term-model)-vocabulary gap proven by the
[adjoint expressibility audit](../adjoint-expressibility-audit.md)<span data-glossary-navigation="true"> ([adjoint glossary](../../../../glossary.md#term-adjoint))</span>; [compiler](../../../../glossary.md#term-compiler) and later [autograd](../../../../glossary.md#autograd)
work remain separately owned.

## [Scope](../../../../glossary.md#term-scope)

- Add public [immutable](../../../../glossary.md#term-immutable) `SumToShapeAttrs` with exactly one `Shape targetShape` component.
- Extend only `AggregateReductionKind.SUM` with the exact one-input/one-output
  `SumToShapeAttrs` [signature](../../../../glossary.md#term-signature); add no [operation](../../../../glossary.md#operation) kind or enum [constant](../../../../glossary.md#term-constant).
- Add exactly one public [Tensor](../../../../glossary.md#tensor) method:

  ```java
  public Tensor sumToShape(Shape targetShape)
  ```

- Add package-private field-free `TensorSumToShapeExpressions` for deterministic validation,
  descriptor, operation, producer, and provenance construction.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [descriptor glossary](../../../../glossary.md#term-descriptor); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>
- Accept current numeric types [BFLOAT16](../../../../glossary.md#term-bfloat16), [FLOAT32](../../../../glossary.md#term-float32), [FLOAT64](../../../../glossary.md#term-float64), [INT32](../../../../glossary.md#data-type--datatype), and [INT64](../../../../glossary.md#data-type--datatype); reject [BOOL](../../../../glossary.md#data-type--datatype).
- Define right-aligned target compatibility, leading-axis reduction, binding-dependent aligned
  reduction, exact target-Shape retention, numerical/empty behavior, metadata, failures,
  provenance, identifiers, and lifecycle boundaries.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [axis glossary](../../../../glossary.md#term-axis); [binding glossary](../../../../glossary.md#term-binding); [Shape glossary](../../../../glossary.md#shape); [metadata glossary](../../../../glossary.md#term-metadata); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Add focused semantic/expression tests and change every exact public Tensor method-count inventory
  from 188 to 189.
- Finalize Javadocs, Tensor/Compile APIs, glossary, capabilities, and planning records through the
  required separate clean-context documentation pass.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [planning glossary](../../../../glossary.md#term-planning))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- a new `SUM_TO_SHAPE`, `UNBROADCAST`, `REDUCE_TO_SHAPE`, MATMUL-backward, attention-backward, or
  other operation kind<span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [operation glossary](../../../../glossary.md#operation))</span>
- changing the existing full-, single-axis, multi-axis, or masked SUM behavior, signatures,
  messages, Shapes, numerical policies, producer contracts, or public methods<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [producer glossary](../../../../glossary.md#term-producer))</span>
- adding a [target](../../../../glossary.md#term-target) [Tensor](../../../../glossary.md#tensor), [axes](../../../../glossary.md#term-axis), `keepDimensions`, raw `long...` target sizes, a public attrs
  overload, aliases, convenience names, builders, registries, or constraint objects<span data-glossary-navigation="true"> ([aliases glossary](../../../../glossary.md#aliasing-and-arena))</span>
- changing [`Shape`](../../../../glossary.md#shape), [`Dimension`](../../../../glossary.md#dimension), `DimensionExpression`, `DimensionExpressions`, `ShapeBroadcast`,
  `TensorDescriptor`, `TensorFactory`, [`TensorProducer`](../../../../glossary.md#producer--tensorproducer), or `TensorProvenance`
- adopting this operation in MATMUL, attention, binary expressions, compiler-generated adjoints,
  graph capture, canonicalization, or optimization<span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [compiler glossary](../../../../glossary.md#term-compiler); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form))</span>
- symbolic binding, concrete-axis resolution, runtime Shape values, execution-time validation,
  value aggregation, algorithms, accumulation buffers, kernels, lowering, fusion, or backend
  support<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [runtime glossary](../../../../glossary.md#term-runtime); [validation glossary](../../../../glossary.md#term-validation); [buffers glossary](../../../../glossary.md#term-buffer); [kernels glossary](../../../../glossary.md#kernel); [lowering glossary](../../../../glossary.md#lowering); [fusion glossary](../../../../glossary.md#term-fusion); [backend glossary](../../../../glossary.md#backend))</span>
- BOOL reduction, mean-to-Shape, product-to-Shape, arbitrary reshape, broadcasting the input to a
  larger Shape, or changing data type during the operation<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [mean glossary](../../../../glossary.md#term-mean); [broadcasting glossary](../../../../glossary.md#broadcasting); [data type glossary](../../../../glossary.md#data-type--datatype))</span>
- gradients, autograd traversal, compiler, planning, prepare, runtime, engine, training, another
  module, dependencies, Gradle, architecture documents/tests, conformance, integration, or later
  tasks<span data-glossary-navigation="true"> ([gradients glossary](../../../../glossary.md#term-gradient); [autograd glossary](../../../../glossary.md#autograd); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [engine glossary](../../../../glossary.md#term-engine); [module glossary](../../../../glossary.md#term-module); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Exact semantic and public contract

### Existing SUM kind and new attributes

`AggregateReductionKind` retains its exact existing enum [constants](../../../../glossary.md#term-constant) and order. No new kind is
introduced. Its [signature](../../../../glossary.md#term-signature) variants become:

```text
SUM:
  NoOperationAttrs                         1 input, 1 output
  AxisReductionAttrs                       1 input, 1 output
  MultiAxisReductionAttrs                  1 input, 1 output
  MaskedReductionAttrs                     2 inputs, 1 output
  SumToShapeAttrs                          1 input, 1 output

MEAN:
  NoOperationAttrs                         1 input, 1 output
  AxisReductionAttrs                       1 input, 1 output
  MultiAxisReductionAttrs                  1 input, 1 output
  MaskedReductionAttrs                     2 inputs, 1 output
```

Every other aggregate signature remains unchanged. The new SUM signature is appended after all
existing SUM variants so existing declaration order remains stable. [`MEAN`](../../../../glossary.md#term-mean) and every other kind
must reject `SumToShapeAttrs` through current exact attributes-class matching.

`SumToShapeAttrs` is a public record in the reduction package with exactly:

```java
Shape targetShape
```

Its [canonical](../../../../glossary.md#term-canonical-form) constructor performs exactly:

```java
targetShape = Objects.requireNonNull(targetShape, "targetShape");
```

The exact [immutable](../../../../glossary.md#term-immutable) [Shape](../../../../glossary.md#shape) reference is retained. The record performs no input-aware
compatibility, [axis](../../../../glossary.md#term-axis), [binding](../../../../glossary.md#term-binding), type, [descriptor](../../../../glossary.md#term-descriptor), [provenance](../../../../glossary.md#provenance), [compiler](../../../../glossary.md#term-compiler), or execution [validation](../../../../glossary.md#term-validation).
It deliberately does not reuse [layout](../../../../glossary.md#layout)-family `TargetShapeAttrs`: reshape/expand [target](../../../../glossary.md#term-target) geometry
and reduction-to-Shape semantics remain separate typed [operation](../../../../glossary.md#operation) variants.

### Public method and helper surface

[`Tensor`](../../../../glossary.md#tensor) adds exactly:

```java
public Tensor sumToShape(Shape targetShape)
```

The method delegates exactly once to `TensorSumToShapeExpressions.apply(this, targetShape)` and
does not compose ordinary public `sum`, `reshape`, or another Tensor expression.

The package-private helper is one final non-record class with no fields, no nested types, one
private zero-[argument](../../../../glossary.md#term-argument) constructor, and exactly these methods:

```java
static Tensor apply(Tensor input, Shape targetShape)
private static void validateInput(Tensor input)
private static void validateCompatibility(Shape inputShape, Shape targetShape)
private static Tensor create(Tensor input, Shape targetShape)
```

It adds no reusable public validator, [Shape](../../../../glossary.md#shape) utility, [binding](../../../../glossary.md#term-binding) object, or factory overload.

### [Validation](../../../../glossary.md#term-validation) order and diagnostics

`apply` performs local validation in exactly this order:

1. `Objects.requireNonNull(input, "input")`;
2. `Objects.requireNonNull(targetShape, "targetShape")`;
3. read and validate the input [data type](../../../../glossary.md#data-type--datatype);
4. read the exact input [Shape](../../../../glossary.md#shape);
5. reject [target](../../../../glossary.md#term-target) [rank](../../../../glossary.md#term-rank) greater than input rank;
6. inspect right-aligned target axes in increasing target-axis order and reject the first
   statically incompatible pair;
7. construct attrs, [descriptor](../../../../glossary.md#term-descriptor), [operation](../../../../glossary.md#operation), [producer](../../../../glossary.md#term-producer), [provenance](../../../../glossary.md#provenance), and [Tensor](../../../../glossary.md#tensor).

[BOOL](../../../../glossary.md#data-type--datatype) fails before Shape compatibility with:

```text
input must have a numeric data type for SUM, but was BOOL
```

A target rank greater than input rank fails with:

```text
sumToShape target rank must not exceed input rank: input=<inputRank>, target=<targetRank>
```

For target [axis](../../../../glossary.md#term-axis) `t`, the aligned input axis is:

```text
i = inputRank - targetRank + t
```

If both aligned [Dimensions](../../../../glossary.md#dimension) are static, the pair is locally compatible exactly when their extents
are equal or the target [extent](../../../../glossary.md#term-extent) is one. The first incompatible pair fails with:

```text
sumToShape incompatible dimension at target axis <t> (input axis <i>): input=<inputDimension>, target=<targetDimension>
```

An equal Dimension pair, a static target [singleton](../../../../glossary.md#term-singleton), or any pair involving an unresolved Dimension
is accepted locally. The latter records no separate constraint object; the operation's source and
target Shapes [carry](../../../../glossary.md#term-carry) the later compatibility obligation. All local failures occur before factory
delegation and consume no Tensor identifier.

### Right-aligned reduction and [binding](../../../../glossary.md#term-binding) obligation

At concrete binding, [target](../../../../glossary.md#term-target) [rank](../../../../glossary.md#term-rank) must remain at most input rank. Every leading input [axis](../../../../glossary.md#term-axis) without
an aligned target axis is reduced and removed. For each aligned pair:

- equal bound extents preserve the [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) without reduction;
- target bound extent one accepts any non-negative input extent, sums that input axis, and retains
  one target position; and<span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent))</span>
- every other pair is invalid and must be rejected before execution.

For example:

```text
input Shape  = [2, 3, 4]
target Shape =    [3, 1]

input axis 0 has no target axis -> reduce and remove
input axis 1 equals target 3    -> preserve coordinates
input axis 2 maps to target 1   -> reduce and retain one position
result Shape                     -> [3, 1]
```

For a binding-dependent example:

```text
input Shape  = [8, 4]
target Shape = [X, 4]
```

Construction accepts the expression. Later binding must prove `X == 1` or `X == 8`. `X == 1`
reduces input axis zero; `X == 8` preserves it; any other value is invalid. Two unequal unresolved
aligned [Dimensions](../../../../glossary.md#dimension) are also accepted because the exact target [Shape](../../../../glossary.md#shape) is already known and later
binding can prove target-one-or-equal without choosing a result Dimension during [model](../../../../glossary.md#term-model)
construction.

[Scalar](../../../../glossary.md#term-scalar) target Shape reduces every input axis. Scalar input accepts only scalar target. Equal
input and target Shapes are valid and still create a fresh explicit SUM [occurrence](../../../../glossary.md#term-occurrence) rather than
returning the input or canonicalizing the expression.

### Numerical and empty-domain semantics

For each [target](../../../../glossary.md#term-target) [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix), the result sums exactly the source coordinates selected by the rules
above. The [operation](../../../../glossary.md#operation) inherits ordinary `AggregateReductionKind.SUM` semantics without defining a
new accumulation policy:

- [INT32](../../../../glossary.md#data-type--datatype) and [INT64](../../../../glossary.md#data-type--datatype) retain their input type and use fixed-width modular addition;
- floating values retain input type and the current SUM NaN, infinity, signed-zero, reassociation,
  and rounding contract;<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [reassociation glossary](../../../../glossary.md#term-reassociation); [rounding glossary](../../../../glossary.md#term-rounding))</span>
- an actually reduced empty source domain produces the current SUM empty identity, numeric
  positive zero; and<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- a coordinate with no reduced [axis](../../../../glossary.md#term-axis) is the corresponding input value, not an arithmetic rewrite.

Target [Shape](../../../../glossary.md#shape) may contain zero, named dynamic, or expression [Dimensions](../../../../glossary.md#dimension). A static source zero
aligned to target one is valid and describes an empty sum yielding one zero result at that target
coordinate. A target zero aligned to source zero is equal and describes no output coordinates.
No [element](../../../../glossary.md#term-element) count is calculated during construction.

### [Descriptor](../../../../glossary.md#term-descriptor), [producer](../../../../glossary.md#term-producer), [provenance](../../../../glossary.md#provenance), and identifiers

Success creates in this order:

1. one `SumToShapeAttrs` retaining the exact caller [target](../../../../glossary.md#term-target) [Shape](../../../../glossary.md#shape);
2. one unresolved `TensorDescriptor` with exact input data type, exact target Shape reference, and
   unchanged <span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype))</span>`requiresGrad` [metadata](../../../../glossary.md#term-metadata);
3. one `Operation(AggregateReductionKind.SUM, attrs)`;
4. one producer with exact ordered inputs `[input]` and one output descriptor;
5. output-index-zero provenance and one fresh, unlabeled, storage-free [Tensor](../../../../glossary.md#tensor).

The input Tensor is unchanged. The result has no resolved [layout](../../../../glossary.md#layout) even for equal Shapes, no label,
no storage, and no additional output. Every successful call creates a fresh producer and consumes
one Tensor identifier. Current identifier-exhaustion behavior remains unchanged and no partial
state is rolled back.

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
- [Task 0023](0023-adjoint-expressibility-audit.md)
- [Shape and dimension model](0002-shape-and-dimension-model.md)<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [dimension glossary](../../../../glossary.md#dimension); [model glossary](../../../../glossary.md#term-model))</span>
- [Operation hardening](0018k-operation-signature-and-construction-hardening.md)<span data-glossary-navigation="true"> ([Operation glossary](../../../../glossary.md#operation))</span>
- [Shared producer provenance](0018l-shared-multi-output-tensor-provenance.md)<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>
- [Symbolic extent expressions](0018m-symbolic-extent-expressions.md)<span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent))</span>
- [Integral reductions](0018u1-integral-reductions-and-arg-min-normalization.md)
- [Multi-axis reductions](0018v-multi-axis-and-statistical-reductions.md)<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis))</span>
- [MATMUL](0019-matmul-semantics-and-tensor-expression.md)<span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>
- [Scaled dot-product attention](0019e-scaled-dot-product-attention.md)<span data-glossary-navigation="true"> ([Scaled dot-product attention glossary](../../../../glossary.md#scaled-dot-product-attention))</span>
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Glossary](../../../../glossary.md)

## Architecture constraints

- Work stays inside model-owned backend-independent operation semantics, Tensor metadata,
  pre-capture producer/provenance construction, and directly affected docs/planning files.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [backend glossary](../../../../glossary.md#backend); [Tensor glossary](../../../../glossary.md#tensor); [metadata glossary](../../../../glossary.md#term-metadata); [capture glossary](../../../../glossary.md#term-capture-meanings); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance); [planning glossary](../../../../glossary.md#term-planning))</span>
- `AggregateReductionKind.SUM` remains semantic metadata and exposes no backend support, bound
  axis set, algorithm, kernel, execution route, or runtime state.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [kernel glossary](../../../../glossary.md#kernel))</span>
- Model construction may reject provable static incompatibility and retain unresolved
  target-one-or-equal obligations. Compiler or later concrete binding owns proof before execution.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [Compiler glossary](../../../../glossary.md#term-compiler); [binding glossary](../../../../glossary.md#term-binding))</span>
- Compiler later owns adjoint graph construction and may emit this public semantic variant;
  this task implements neither compiler use nor gradient rules.<span data-glossary-navigation="true"> ([adjoint glossary](../../../../glossary.md#term-adjoint); [graph glossary](../../../../glossary.md#graph); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- [Runtime](../../../../glossary.md#term-runtime) hot paths must not consume [`Operation`](../../../../glossary.md#operation) or `CompiledNode`; backend prepare owns lowering,
  specialization, fusion, and concrete reduction implementation.<span data-glossary-navigation="true"> ([prepare glossary](../../../../glossary.md#prepare); [lowering glossary](../../../../glossary.md#lowering); [specialization glossary](../../../../glossary.md#term-specialization); [fusion glossary](../../../../glossary.md#term-fusion))</span>
- No architecture, ADR, architecture-test, dependency, Gradle, cross-module, compiler, runtime,
  prepare, backend, training, conformance, or integration change is authorized. Stop if the
  existing Shape, signature, producer, or descriptor foundation cannot express this contract.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module); [conformance glossary](../../../../glossary.md#term-conformance); [Shape glossary](../../../../glossary.md#shape); [signature glossary](../../../../glossary.md#term-signature); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>

## Package impact

Existing packages changed:

- `io.github.pho001.synaptik.model.operation.reduction`
- `io.github.pho001.synaptik.model.tensor`

No package is added, moved, or renamed.

Type placement:

- `...operation.reduction.AggregateReductionKind` owns the existing SUM semantic identity and its
  exact structural variants.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- `...operation.reduction.SumToShapeAttrs` owns the intrinsic immutable target Shape for the new
  SUM variant.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [target glossary](../../../../glossary.md#term-target); [Shape glossary](../../../../glossary.md#shape))</span>
- `...tensor.TensorSumToShapeExpressions` owns package-private operand validation and expression
  construction.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- `...tensor.Tensor` remains the public fluent [facade](../../../../glossary.md#term-facade).

## Affected files

Production (4):

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/reduction/AggregateReductionKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/reduction/SumToShapeAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorSumToShapeExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Tests (14):

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/OperationSignatureTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/reduction/ReductionSemanticsTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorSumToShapeExpressionTest.java`
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

The eleven existing [Tensor](../../../../glossary.md#tensor) tests change only the exact global method inventory from 188 to 189
and, for `TensorTest`, add the exact `sumToShape(Shape)` method to the public surface lock. They
receive no unrelated changes.

Documentation/[planning](../../../../glossary.md#term-planning) (7):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review unchanged unless inaccurate: current [Shape](../../../../glossary.md#shape)/[Dimension](../../../../glossary.md#dimension)/ShapeBroadcast, [descriptor](../../../../glossary.md#term-descriptor), [operation](../../../../glossary.md#operation),
factory, [producer](../../../../glossary.md#term-producer)/[provenance](../../../../glossary.md#provenance), ordinary reduction, [MATMUL](../../../../glossary.md#matrix-multiplication--matmul), attention, Training/[Runtime](../../../../glossary.md#term-runtime) APIs,
architecture/ADRs/tests, [conformance](../../../../glossary.md#term-conformance)/integration, [Gradle](../../../../glossary.md#term-gradle), dependencies, and other [modules](../../../../glossary.md#term-module).

## Maximum [scope](../../../../glossary.md#term-scope)

Exactly 25 paths: four production, fourteen tests, and seven documentation/[planning](../../../../glossary.md#term-planning) paths. This
cohesive public capability exceeds the usual 18-path guardrail under the user's standing automatic
higher-path authorization because one semantic variant, one [facade](../../../../glossary.md#term-facade) method/helper, eleven existing
global [Tensor](../../../../glossary.md#tensor)-count locks, the repository-wide [operation](../../../../glossary.md#operation)-[signature](../../../../glossary.md#term-signature) [matrix](../../../../glossary.md#term-matrix-meanings), focused coverage, and
mandatory documentation must change atomically. The signature matrix was added after the first
final [model](../../../../glossary.md#term-model) [run](../../../../glossary.md#term-run) proved that every new family signature must update this completed global lock.
Stop for path 26, another public type/method, another helper/test/document, a foundation edit,
[compiler](../../../../glossary.md#term-compiler) adoption, later-task work, cross-[module](../../../../glossary.md#term-module) work, architecture/[Gradle](../../../../glossary.md#term-gradle) change, or unrelated
cleanup. If live repository evidence changes another inventory path, update this task before
implementation and stop pending renewed authorization above 25.

## [Javadoc](../../../../glossary.md#term-javadoc) and documentation requirements

- Add complete Javadocs for `SumToShapeAttrs`, the SUM variant, helper, helper methods, and public
  Tensor method, including target/source alignment, dynamic obligation, numerical/empty behavior,
  metadata, provenance/IDs, failures, and lifecycle ownership.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [target glossary](../../../../glossary.md#term-target); [alignment glossary](../../../../glossary.md#term-alignment); [metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance); [lifecycle glossary](../../../../glossary.md#lifecycle); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Apply General, [API](../../../../glossary.md#term-application-programming-interface-api)/Javadoc, [Planning](../../../../glossary.md#term-planning), and Example profiles as relevant, with complete `@param`,
  `@return`, and expected `@throws` text.
- Tensor API moves binding-aware sum-to-Shape from planned to current, adds static and dynamic
  examples, exact public signature, Shape/type rules, producer metadata, and lifecycle boundary.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [signature glossary](../../../../glossary.md#term-signature); [Shape glossary](../../../../glossary.md#shape); [producer glossary](../../../../glossary.md#term-producer))</span>
- Compile API records current model metadata and the later binding-proof obligation; it must not
  claim compiler capture, autograd use, lowering, or execution is implemented.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [model glossary](../../../../glossary.md#term-model); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [autograd glossary](../../../../glossary.md#autograd); [lowering glossary](../../../../glossary.md#lowering))</span>
- Glossary adds or updates [`sum-to-Shape`](../../../../glossary.md#sum-to-shape), distinguishing it from arbitrary reshape, ordinary
  fixed-axis SUM, broadcasting, and compiler-generated adjoint use.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [broadcasting glossary](../../../../glossary.md#broadcasting); [adjoint glossary](../../../../glossary.md#term-adjoint))</span>
- Capabilities/task/master/roadmap remain synchronized: 0023 and 0023A become/stay Complete only
  after implementation, 0023B is the next Draft frontier without a detailed specification, and
  0023C–0024 remain Draft.
- Review the official PyTorch
  [`sum_to_size`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.sum_to_size.html)
  page only as terminology/comparison evidence, not as design authority; repository contracts
  remain primary.
- Record reasoned no-change conclusions for related existing Javadocs, Training/Runtime APIs,
  architecture/ADRs/tests, dependencies, Gradle, conformance/integration, other modules, and later
  tasks.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance); [modules glossary](../../../../glossary.md#term-module))</span>

## Acceptance criteria

- No new [operation](../../../../glossary.md#operation) kind or enum [constant](../../../../glossary.md#term-constant); exact existing `AggregateReductionKind` order remains.
- SUM alone appends exact fixed `SumToShapeAttrs` 1/1 signature; MEAN and every other signature
  remain exact and reject the new attrs class.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [MEAN glossary](../../../../glossary.md#term-mean))</span>
- `SumToShapeAttrs` has exactly one non-null `Shape targetShape` component, explicit documented
  accessor, exact reference retention, generated record value semantics, and no extra API/state.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Exactly one new public `sumToShape(Shape)` method; public Tensor count is 189; no alias, raw-size,
  axes, keep-dimension, attrs, or target-Tensor overload.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [alias glossary](../../../../glossary.md#aliasing-and-arena); [axes glossary](../../../../glossary.md#term-axis); [dimension glossary](../../../../glossary.md#dimension); [target glossary](../../../../glossary.md#term-target))</span>
- Helper has the exact field-free four-method surface and private constructor described above.
- Exact null/type/rank/aligned-static validation order, exception types/messages, and no-ID local
  failures.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [validation glossary](../../../../glossary.md#term-validation))</span>
- Exact right-aligned semantics: leading input axes reduce, equal bound aligned axes preserve,
  target-one axes reduce and retain, and every other concrete pair is invalid.
- Equal, singleton, zero, scalar, lower-rank, named/expression, unresolved/static, and
  unresolved/unresolved cases follow the selected contract and retain exact target Shape.<span data-glossary-navigation="true"> ([singleton glossary](../../../../glossary.md#term-singleton); [scalar glossary](../../../../glossary.md#term-scalar); [Shape glossary](../../../../glossary.md#shape))</span>
- Numeric domain and SUM numerical/empty semantics remain exact; BOOL is rejected; no type
  promotion or value/storage inspection occurs.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- Result has exact input type and eligibility, exact target Shape reference, unresolved layout,
  no label/storage, exact SUM/attrs, ordered <span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout))</span>`[input]` producer, output index zero, one fresh ID,
  and unchanged input.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer))</span>
- Exact 25-path/package scope; no changes to completed reduction behavior, ShapeBroadcast,
  MATMUL/attention helpers, compiler, execution layers, architecture, build, or later tasks.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- Separate clean documentation-focused pass and all required validation/evidence complete before
  status Complete.

## Tests / [validation](../../../../glossary.md#term-validation)

Focused implementation command:

```bash
./gradlew :modules:model:test \
  --tests io.github.pho001.synaptik.model.operation.OperationSignatureTest \
  --tests io.github.pho001.synaptik.model.operation.reduction.ReductionSemanticsTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorSumToShapeExpressionTest \
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
  --tests io.github.pho001.synaptik.model.tensor.TensorScaledDotProductAttentionExpressionTest
```

After executable Java stabilizes, [run](../../../../glossary.md#term-run) exactly one final [model](../../../../glossary.md#term-model) suite:

```bash
./gradlew :modules:model:test
```

Focused coverage must verify exact attrs/[signature](../../../../glossary.md#term-signature) [shape](../../../../glossary.md#shape) and rejection, public/helper surfaces,
static leading and aligned reductions, equal-shape freshness, [scalar](../../../../glossary.md#term-scalar) [target](../../../../glossary.md#term-target)/input, zero extents,
target reference retention, all accepted unresolved pair categories, deterministic first static
mismatch, [BOOL](../../../../glossary.md#data-type--datatype)/[rank](../../../../glossary.md#term-rank)/null failures, no-ID local failures, exact [descriptor](../../../../glossary.md#term-descriptor), [operation](../../../../glossary.md#operation), [producer](../../../../glossary.md#term-producer),
[provenance](../../../../glossary.md#provenance), storage/label absence, input immutability, and public count 189. Tests inspect [metadata](../../../../glossary.md#term-metadata)
only and do not claim value execution.

Documentation pass after final Javadocs:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

Also validate Markdown links/anchors/fences/final newlines/trailing whitespace, official reference,
examples, exact 25 paths, package placement, signatures/count 189, synchronized status, 0023B
Draft/no spec, and absence of [compiler](../../../../glossary.md#term-compiler)/[runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend)/build changes. Reuse successful Java
evidence unless executable Java changes.

Repository-wide validation is deferred to the capability [checkpoint](../../../../glossary.md#term-checkpoint) after task 0023F and to [CI](../../../../glossary.md#term-continuous-integration-ci);
this single-[module](../../../../glossary.md#term-module) task changes no [dependency](../../../../glossary.md#term-dependency), build, or architecture boundary.

## Dependencies

- 0001–0002: numeric types and [immutable](../../../../glossary.md#term-immutable) static/symbolic [Shape](../../../../glossary.md#shape)/[Dimension](../../../../glossary.md#dimension) contracts.
- 0005–0007, 0011–0013: typed operations, [descriptors](../../../../glossary.md#term-descriptor), [Tensor](../../../../glossary.md#tensor), [identity](../../../../glossary.md#term-identity-meanings), factory, and [provenance](../../../../glossary.md#provenance).
- 0016A–0016D and 0018V: SUM family, numerical/empty policy, axis and multi-axis reduction
  construction.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [multi-axis reduction glossary](../../../../glossary.md#multi-axis-reduction))</span>
- 0018K: exact family-owned [signatures](../../../../glossary.md#term-signature); 0018L: [producer](../../../../glossary.md#term-producer)/output-index provenance.
- 0018M–0018M1: symbolic [extent](../../../../glossary.md#term-extent) expressions retained without [model](../../../../glossary.md#term-model)-side [binding](../../../../glossary.md#term-binding).
- 0018U–0018U1: selected signed-integral arithmetic/reduction semantics.
- 0019 and 0019E: current deferred [singleton](../../../../glossary.md#term-singleton)-or-equal [MATMUL](../../../../glossary.md#matrix-multiplication--matmul)/attention [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) obligations.
- 0023: completed [adjoint](../../../../glossary.md#term-adjoint) expressibility audit selecting this exact general public gap.

All dependencies are Complete.

## Follow-up tasks

- 0023B remains the next Draft frontier for Gather-compatible axis scatter-add; do not create its
  detailed specification during this task.<span data-glossary-navigation="true"> ([Gather glossary](../../../../glossary.md#gather); [axis glossary](../../../../glossary.md#term-axis))</span>
- 0023C–0023F remain concise Draft public-capability rows.
- Compiler/autograd work later consumes this semantic variant and owns target binding proof,
  gradient construction, canonicalization, and optimization.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [autograd glossary](../../../../glossary.md#autograd); [target glossary](../../../../glossary.md#term-target); [binding glossary](../../../../glossary.md#term-binding); [gradient glossary](../../../../glossary.md#term-gradient); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form))</span>
- Task 0024 remains Draft and depends through completed 0023F.

## Architecture impact

Expected impact: None. Stop if implementation needs a new [operation](../../../../glossary.md#operation) kind, [Shape](../../../../glossary.md#shape) constraint type,
[runtime](../../../../glossary.md#term-runtime) Shape value, [compiler](../../../../glossary.md#term-compiler)/[prepare](../../../../glossary.md#prepare)/[backend](../../../../glossary.md#backend) contract, [dependency](../../../../glossary.md#term-dependency), architecture update, or work
outside the exact [model](../../../../glossary.md#term-model) and documentation [scope](../../../../glossary.md#term-scope).

## Implementation prompt

Use this prompt in a separate clean-context task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/developer-guide/documentation-rules.md,
docs/planning/planning-guide.md, docs/planning/modules/model/capabilities.md, the model master plan,
roadmap, completed tasks 0016A–0016D/0018K–0018M1/0018U–0019E/0023, and task 0023A in full.
Inspect every affected and review-only source/test named by task 0023A.

Implement task 0023A exactly inside its 25 authorized paths. Add binding-aware sum-to-Shape only
as a new exact AggregateReductionKind.SUM attribute variant and one public Tensor expression; add
no kind, alias, Shape/binding foundation, compiler adoption, execution behavior, or later task.
Stop on architecture, dependency, completed-contract, validation-order, affected-file, or maximum-
scope conflict.

Run focused tests while developing and exactly one final model suite after executable Java
stabilizes. Then hand the actual diff and exact Java evidence to a separate clean-context
documentation-focused agent in the same overall change. That agent must independently inspect
final source/tests, finalize Javadocs, Tensor/Compile APIs, glossary, capability/task/master/roadmap
status and documentation validation, and reuse successful Java evidence unless executable behavior
changes or it records a concrete reason.

Do not mark 0023A Complete until both passes and every acceptance criterion succeed. Leave 0023B
and every later task Draft without a detailed specification.
```

## Documentation-agent [handoff](../../../../glossary.md#term-handoff)

The implementation agent must hand over this task, the actual implementation/test diff, exact
Java evidence, SUM [signature](../../../../glossary.md#term-signature) change, [target](../../../../glossary.md#term-target)-[Shape](../../../../glossary.md#shape)/right-aligned/[binding](../../../../glossary.md#term-binding) semantics, [validation](../../../../glossary.md#term-validation) and
ID order, numeric/empty policy, [metadata](../../../../glossary.md#term-metadata)/[provenance](../../../../glossary.md#provenance) behavior, architecture constraints, expected
[Tensor](../../../../glossary.md#tensor)/[Compile](../../../../glossary.md#compile) [API](../../../../glossary.md#term-application-programming-interface-api) and glossary changes, official comparison link, existing-[Javadoc](../../../../glossary.md#term-javadoc) review list,
and every documentation/[scope](../../../../glossary.md#term-scope)/status command. The documentation agent must inspect final source
and tests rather than rely on the summary.

## Local decisions

- Reused `AggregateReductionKind.SUM` and appended `SumToShapeAttrs` after its four existing
  signature variants; no new kind or enum constant was needed.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [constant glossary](../../../../glossary.md#term-constant))</span>
- Kept `SumToShapeAttrs` to exactly one non-null `Shape targetShape` component. Input-aware
  compatibility remains in the field-free package-private Tensor helper.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- Accepted every aligned pair involving an unresolved Dimension. Construction rejects only a
  fully static pair that is neither equal nor target-one; later binding validation owns proof.<span data-glossary-navigation="true"> ([Dimension glossary](../../../../glossary.md#dimension); [target glossary](../../../../glossary.md#term-target); [binding glossary](../../../../glossary.md#term-binding); [validation glossary](../../../../glossary.md#term-validation))</span>
- Added `OperationSignatureTest` to the authorized scope after the first final model run exposed
  its repository-wide family-signature matrix. This recorded correction changed the task from 24
  to exactly 25 paths before implementation continued.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [model glossary](../../../../glossary.md#term-model); [run glossary](../../../../glossary.md#term-run); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- Used the official PyTorch `sum_to_size` page only to compare terminology. Synaptik retains its
  independently specified exact-Shape, metadata, validation, and lifecycle contract.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [metadata glossary](../../../../glossary.md#term-metadata); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>

## Known limitations

- Model construction does not bind dimensions, resolve a concrete reduction-axis set, inspect
  values, aggregate, construct gradients, capture a graph, lower, or execute.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [dimensions glossary](../../../../glossary.md#dimension); [axis glossary](../../../../glossary.md#term-axis); [gradients glossary](../../../../glossary.md#term-gradient); [capture glossary](../../../../glossary.md#term-capture-meanings); [graph glossary](../../../../glossary.md#graph))</span>
- Compiler adoption, target-one-or-equal proof at concrete binding, adjoint construction,
  canonicalization, and optimization remain future work.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [target glossary](../../../../glossary.md#term-target); [binding glossary](../../../../glossary.md#term-binding); [adjoint glossary](../../../../glossary.md#term-adjoint); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form))</span>
- Runtime, prepare, engine, training, backend, conformance, and integration layers do not yet
  consume this semantic variant.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Repository-wide validation remains deferred to the capability checkpoint after task 0023F and
  to CI, as planned for this single-module task.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [CI glossary](../../../../glossary.md#term-continuous-integration-ci); [module glossary](../../../../glossary.md#term-module))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- The initial 13-suite focused command passed 126 tests.
- The first final [model](../../../../glossary.md#term-model) [run](../../../../glossary.md#term-run) failed only because the completed global `OperationSignatureTest`
  matrix did not yet include the new SUM attributes class. After the affected-file specification
  was corrected and that exact lock was updated, the 14-suite focused command passed 131 tests.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-review-matrix))</span>
- The replacement final `./gradlew :modules:model:test` passed 977 tests. Executable Java did not
  change after that run, so the independent documentation pass reused this evidence as required.
- Manual `javap`, reflection, import, and source inspection confirmed the exact helper/record/enum
  surfaces, signature order, one public <span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature))</span>`sumToShape(Shape)` method, single delegation, and public
  Tensor count 189.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- Independent clean-context Javadocs completed without changing executable Java behavior.
  `./gradlew :modules:model:javadoc` passed with `BUILD SUCCESSFUL`; generated public pages contain
  `SumToShapeAttrs`, SUM's [target](../../../../glossary.md#term-target)-[Shape](../../../../glossary.md#shape) semantics, and `Tensor.sumToShape(Shape)` with parameters,
  result, failures, metadata, and lifecycle boundaries. The package-private helper remains
  intentionally absent from public generated Javadoc and was validated in source and bytecode.<span data-glossary-navigation="true"> ([parameters glossary](../../../../glossary.md#term-parameter); [lifecycle glossary](../../../../glossary.md#lifecycle); [Javadoc glossary](../../../../glossary.md#term-javadoc); [bytecode glossary](../../../../glossary.md#term-bytecode))</span>
- The exact Tensor API example compiled and ran under the repository's Java 26 model classes. It
  printed <span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>`Shape[3, 1]`, exact target/input reference checks `true`, output index `0`, dynamic
  `Shape[X, 4]`, and unresolved-[layout](../../../../glossary.md#layout) `true`, matching the documented [metadata](../../../../glossary.md#term-metadata) output.
- The targeted Markdown validator passed all seven changed documentation/planning files, 717 local
  links, required anchors including the new binding-aware section, balanced fences, and final
  newlines. The official PyTorch page was reviewed only for the documented naming comparison.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [binding glossary](../../../../glossary.md#term-binding))</span>
- Exact-scope validation passed exactly 25 task paths plus the two declared untracked task-0023
  baseline artifacts. Status checks confirmed 0023 and 0023A Complete, 0023B–0023F and 0024 Draft,
  and no 0023B detailed specification. No compiler, runtime, training, backend, testing-layer,
  architecture, dependency, or build path changed. <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [artifacts glossary](../../../../glossary.md#term-artifact); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [dependency glossary](../../../../glossary.md#term-dependency))</span>`git diff --check` passed.

## Implementation notes

- Implementation context `/root/implement_0023a` completed production and test work. Independent
  clean documentation context `/root/implement_0023a/docs_0023a` read the architecture,
  documentation profiles, planning contracts, audit, task, final source/tests, and affected APIs
  before finalizing Javadocs and documentation.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Four production paths implement the semantic variant and public construction. Fourteen tests
  cover the exact signature, validation, unresolved/static Shape matrix, identity discipline,
  descriptor/producer/provenance metadata, helper/public surface, and eleven global count locks.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [validation glossary](../../../../glossary.md#term-validation); [Shape glossary](../../../../glossary.md#shape); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [identity glossary](../../../../glossary.md#term-identity-meanings); [descriptor glossary](../../../../glossary.md#term-descriptor); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Seven documentation/planning paths finalize Tensor and Compile APIs, glossary terminology,
  capability state, task evidence, model master plan, and roadmap.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [model glossary](../../../../glossary.md#term-model))</span>
- Related Shape/binding foundations, ordinary reductions, MATMUL/attention construction, Compile
  and Training public contracts, Runtime API, architecture/ADRs/tests, dependencies, Java 26
  Gradle configuration, conformance/integration suites, other modules, and later tasks were
  reviewed and remain unchanged because this task adds only model-owned semantic/expression
  metadata and crosses no dependency or execution boundary.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [Runtime glossary](../../../../glossary.md#term-runtime); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance); [modules glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Completion summary

- Completed changes: added [binding](../../../../glossary.md#term-binding)-aware exact-[target](../../../../glossary.md#term-target)-[Shape](../../../../glossary.md#shape) SUM [metadata](../../../../glossary.md#term-metadata) and one public
  `sumToShape(Shape)` expression with deterministic static validation, deferred unresolved binding
  obligations, exact descriptor/provenance construction, and no new operation kind.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [operation glossary](../../../../glossary.md#operation))</span>
- Files changed or created: exactly 25 authorized task paths (four production, fourteen tests, and
  seven documentation/planning paths), distinct from the two declared task-0023 baseline files.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Tests and validation: reused the passed 131-test focused and 977-test final model evidence;
  final model Javadoc, generated pages, runnable Java 26 metadata example, Markdown links/anchors/
  fences/newlines, package/signature/helper/public-count/status/no-later-spec, exact-scope, and
  whitespace checks passed.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [signature glossary](../../../../glossary.md#term-signature); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: clean context `/root/implement_0023a/docs_0023a`; Complete.
- Documentation impact: all four affected production Javadocs, Tensor API, Compile API, glossary,
  capabilities, this task, model master plan, and roadmap are finalized. Training/Runtime APIs and
  architecture documentation require no change for the model-only reasons recorded above.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- [Javadoc](../../../../glossary.md#term-javadoc) review: `AggregateReductionKind`, `SumToShapeAttrs`,
  `TensorSumToShapeExpressions`, and `Tensor.sumToShape(Shape)` are complete and accurate. Related
  Shape, descriptor, factory, producer/provenance, ordinary reduction, MATMUL, and attention
  Javadocs remain accurate without modification.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>
- Glossary impact: sum-to-Shape is current and distinguished from reshape, fixed-axis SUM,
  broadcasting, and possible future compiler-generated adjoint use.<span data-glossary-navigation="true"> ([sum-to-Shape glossary](../../../../glossary.md#sum-to-shape); [axis glossary](../../../../glossary.md#term-axis); [broadcasting glossary](../../../../glossary.md#broadcasting); [compiler glossary](../../../../glossary.md#term-compiler); [adjoint glossary](../../../../glossary.md#term-adjoint))</span>
- Unresolved issues: None within task scope. Binding proof, gradients, capture, lowering,
  execution, and backend support are intentional future-layer limitations.<span data-glossary-navigation="true"> ([gradients glossary](../../../../glossary.md#term-gradient); [capture glossary](../../../../glossary.md#term-capture-meanings); [lowering glossary](../../../../glossary.md#lowering); [backend glossary](../../../../glossary.md#backend))</span>
- Follow-up required: Draft task 0023B and later planned work only; no follow-up is required to
  complete task 0023A.

Status: Complete
