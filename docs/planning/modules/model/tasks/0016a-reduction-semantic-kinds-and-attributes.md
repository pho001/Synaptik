# Task 0016A: Reduction Semantic Kinds and Attributes

## Status

Complete

## Goal

Define the typed, [backend](../../../../glossary.md#backend)-independent semantic vocabulary and [immutable](../../../../glossary.md#term-immutable) [parameters](../../../../glossary.md#term-parameter) for Synaptik's
aggregate reductions. The family must represent numeric `SUM`, [`MEAN`](../../../../glossary.md#term-mean), `PROD`, `MIN`, and `MAX`,
[boolean](../../../../glossary.md#data-type--datatype) `ALL` and `ANY`, and index-producing `ARG_MAX`, while distinguishing full reduction from
single-[axis](../../../../glossary.md#term-axis) reduction without negative sentinels.

This task creates semantic [descriptors](../../../../glossary.md#term-descriptor) only. It does not add public [Tensor](../../../../glossary.md#tensor) reduction methods,
derive output [shapes](../../../../glossary.md#shape) or data types, inspect values, define empty-input or numerical behavior,
construct [gradients](../../../../glossary.md#term-gradient), [capture](../../../../glossary.md#term-capture-meanings) [graphs](../../../../glossary.md#graph), or report backend support.

## [Scope](../../../../glossary.md#term-scope)

- Add one public `AggregateReductionKind` enum implementing [`OperationKind`](../../../../glossary.md#operationkind).
- Define exactly `SUM`, [`MEAN`](../../../../glossary.md#term-mean), `PROD`, `MIN`, `MAX`, `ALL`, `ANY`, and `ARG_MAX`, in that order.
- Add one public `AxisReductionAttrs` record implementing [`OperationAttrs`](../../../../glossary.md#operationattrs) with exactly
  non-negative normalized `int axis` and `boolean keepDimensions` components.
- Use `NoOperationAttrs.INSTANCE` to represent full reduction across every input [axis](../../../../glossary.md#term-axis) for `SUM`,
  `MEAN`, `PROD`, `MIN`, `MAX`, `ALL`, and `ANY`.
- Add one public `ArgMaxTiePolicy` enum with exactly `FIRST_INDEX` and `LAST_INDEX`, in that order.
- Add one public `ArgMaxAttrs` record implementing `OperationAttrs` with exactly non-negative
  normalized `int axis`, `boolean keepDimensions`, and non-null `ArgMaxTiePolicy tiePolicy`.
- Define the valid kind/attributes pairings through typed documentation without changing generic
  [`Operation`](../../../../glossary.md#operation) [validation](../../../../glossary.md#term-validation).
- Add one focused same-package test covering exact vocabulary, records, validation, value
  semantics, valid composition, typed identity, and exclusions.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Add the cohesive `model.operation.reduction` package to the [model](../../../../glossary.md#term-model) package map.
- Decompose the broad task 0016 queue into small sequential semantic and Tensor-expression tasks,
  while creating no other detailed task specification.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- Finalize affected Javadocs, Tensor API, glossary, task evidence, model master plan, and roadmap
  through the required independent documentation pass during implementation.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- public `Tensor.sum`, [`mean`](../../../../glossary.md#term-mean), `prod`, reduction `min`/`max`, `all`, `any`, `argMax`, or another
  Tensor method, overload, factory, builder, or expression helper<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- `cumSum`, another [scan](../../../../glossary.md#term-cumulative-scan), [`softmax`](../../../../glossary.md#softmax--log-softmax), `logSoftmax`, masked reductions, loss reductions, pooling,
  normalization, or their semantic kinds and attributes<span data-glossary-navigation="true"> ([loss glossary](../../../../glossary.md#loss); [pooling glossary](../../../../glossary.md#term-pooling); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- Tensor inputs, Shape, axis normalization against rank, reduced output shape, rank-zero result,
  retained-dimension construction, result descriptor, label, identity, storage, provenance, or<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [axis glossary](../../../../glossary.md#term-axis); [rank glossary](../../../../glossary.md#term-rank); [rank-zero glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [descriptor glossary](../../../../glossary.md#term-descriptor); [identity glossary](../../../../glossary.md#term-identity-meanings); [provenance glossary](../../../../glossary.md#provenance))</span>
  `TensorFactory.createDerived`
- data-type eligibility, data-type promotion, result [data type](../../../../glossary.md#data-type--datatype), [BOOL](../../../../glossary.md#data-type--datatype) result rules, `ARG_MAX`
  [`INT64`](../../../../glossary.md#data-type--datatype) result rules, [gradient](../../../../glossary.md#term-gradient) eligibility, or local/[graph](../../../../glossary.md#graph)-wide [inference](../../../../glossary.md#term-inference)
- multiple-axis reduction, an axis collection, empty-axis semantics, caller-supplied negative axis,
  full `ARG_MAX`, full-reduction retained [dimensions](../../../../glossary.md#dimension), or a general reduction specification object
- masked `SUM`/`MEAN`, [mask](../../../../glossary.md#term-boolean-mask) [broadcasting](../../../../glossary.md#broadcasting), valid-count denominator, or all-masked behavior
- empty-input identities or errors, NaN policy, floating accumulation precision, compensated
  summation, overflow, product ordering, min/max tie gradients, comparison policy, or index width<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [precision glossary](../../../../glossary.md#term-precision); [compensated summation glossary](../../../../glossary.md#term-compensated-summation); [overflow glossary](../../../../glossary.md#term-overflow))</span>
- eager evaluation, storage reads or writes, allocation, materialization, aliasing, constant
  folding, canonicalization, or common-subexpression elimination<span data-glossary-navigation="true"> ([eager evaluation glossary](../../../../glossary.md#term-eager-evaluation); [materialization glossary](../../../../glossary.md#materialization); [aliasing glossary](../../../../glossary.md#term-aliasing); [constant folding glossary](../../../../glossary.md#term-constant-folding); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [common-subexpression elimination glossary](../../../../glossary.md#term-common-subexpression-elimination-cse))</span>
- gradient values, backward semantic kinds, gradient rules, autograd expansion, optimizer, or
  training behavior<span data-glossary-navigation="true"> ([autograd glossary](../../../../glossary.md#autograd); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- family factories, visitors, registries, parsers, string dispatch, compatibility maps, operation
  traits, arity metadata, result-kind metadata, costs, fusion, routes, kernels, or backend support<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch); [arity glossary](../../../../glossary.md#term-arity); [metadata glossary](../../../../glossary.md#term-metadata); [fusion glossary](../../../../glossary.md#term-fusion); [kernels glossary](../../../../glossary.md#kernel); [backend glossary](../../../../glossary.md#backend))</span>
- changes to [`Operation`](../../../../glossary.md#operation), [`OperationKind`](../../../../glossary.md#operationkind), [`OperationAttrs`](../../../../glossary.md#operationattrs), [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs), DataType, Shape,
  Tensor, graph records, or existing Java tests<span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype))</span>
- compiler, planning, prepare, runtime, backend, engine, tracing, ONNX, dependency, Gradle,
  architecture, or another-module changes<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [engine glossary](../../../../glossary.md#term-engine); [ONNX glossary](../../../../glossary.md#term-onnx); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>
- detailed specifications for tasks 0016B or later

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Architecture overview](../../../../architecture/overview.md)
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Model master plan](../master-plan.md)
- [Task 0002](0002-shape-and-dimension-model.md)
- [Task 0005](0005-operation-semantic-foundation.md)
- [Task 0006](0006-operation-model.md)
- [Task 0013](0013-tensor-provenance-skeleton.md)
- [Task 0015C](0015c-boolean-logical-semantic-kinds.md)
- [Task 0015G](0015g-cast-semantic-kind-and-attributes.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only `legacy/pre-rewrite` branch exposes full and single-[axis](../../../../glossary.md#term-axis) `sum`, [`mean`](../../../../glossary.md#term-mean), `prod`,
reduction `min` and `max`, [boolean](../../../../glossary.md#data-type--datatype) `all` and `any`, plus axis-only `argMax`. Axis forms accept
negative public axes and support removing the axis or retaining it with [extent](../../../../glossary.md#term-extent) one. `argMax`
supports `FIRST_INDEX` and `LAST_INDEX`, defaults to the first index, and produces index values.
Legacy tests cover full and axis forms, retained [dimensions](../../../../glossary.md#dimension), negative-axis [normalization](../../../../glossary.md#term-normalization),
[broadcasting](../../../../glossary.md#broadcasting) of retained results, tie policies, non-[contiguous](../../../../glossary.md#term-contiguous) inputs, [gradients](../../../../glossary.md#term-gradient) where applicable,
[ONNX](../../../../glossary.md#term-onnx) mapping, [CPU](../../../../glossary.md#term-cpu) routes, [Metal](../../../../glossary.md#term-metal) routes, and [compiler](../../../../glossary.md#term-compiler) [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form).

Legacy [operation](../../../../glossary.md#operation) [descriptors](../../../../glossary.md#term-descriptor) store `-1` as an all-axes sentinel and also expose [arity](../../../../glossary.md#term-arity), semantic
family, result kind, cost, [fusion](../../../../glossary.md#term-fusion), expression text, and [backend](../../../../glossary.md#backend)-facing facts. Those mechanisms are
not copied. Full reduction is represented by the absence of axis [parameters](../../../../glossary.md#term-parameter) through
`NoOperationAttrs.INSTANCE`; a present axis is already normalized and non-negative. Later public
[Tensor](../../../../glossary.md#tensor) tasks own negative-axis normalization, [shape](../../../../glossary.md#shape) and data-type derivation, and [provenance](../../../../glossary.md#provenance).
Compiler [autograd](../../../../glossary.md#autograd) owns gradient expansion, while concrete backends and [conformance](../../../../glossary.md#term-conformance) tests own
numeric accumulation, boolean [identities](../../../../glossary.md#term-identity-meanings), tie execution, storage access, and [kernels](../../../../glossary.md#kernel).

## Architecture constraints

- [Operation](../../../../glossary.md#operation) kinds and attributes are [immutable](../../../../glossary.md#term-immutable) [backend](../../../../glossary.md#backend)-independent [model](../../../../glossary.md#term-model) semantics owned by
  `modules/model`.
- `AggregateReductionKind` identifies mathematical reduction meaning only. It stores no axes,
  input, result, graph occurrence, executable behavior, or backend information.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Full `SUM`, [`MEAN`](../../../../glossary.md#term-mean), `PROD`, `MIN`, `MAX`, `ALL`, and `ANY` compose with
  `NoOperationAttrs.INSTANCE`. No negative numeric sentinel represents all [axes](../../../../glossary.md#term-axis).
- A single-axis form composes with `AxisReductionAttrs`. The stored axis is normalized and
  non-negative; later Tensor construction validates and normalizes a caller axis against Shape.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Shape glossary](../../../../glossary.md#shape))</span>
- `keepDimensions` says whether the one reduced axis remains in the result with static extent one.
  It stores no output Shape and has no meaning on the full no-attributes form.<span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent))</span>
- `ARG_MAX` composes only with `ArgMaxAttrs`, because tie policy is intrinsic to its semantics.
  `ARG_MAX` has no full-reduction form in the selected baseline.
- `ArgMaxTiePolicy` describes selection among equal maxima only. It is not reused for reduction
  `MIN`/`MAX` [gradient](../../../../glossary.md#term-gradient) [distribution](../../../../glossary.md#term-distribution), [pooling](../../../../glossary.md#term-pooling), or another family without a future focused task.
- Generic `Operation` continues to validate only non-null kind and attributes values. It must not
  discover families or enforce the documented pairings.
- Kind/attribute representability does not establish input data-type eligibility, output data
  type, output shape, differentiability, empty-input behavior, or backend availability.<span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype))</span>
- Stable enum names and record text are diagnostic, not serialization, ONNX, registry,
  reflection-dispatch, or kernel identifiers.<span data-glossary-navigation="true"> ([serialization glossary](../../../../glossary.md#term-serialization); [ONNX glossary](../../../../glossary.md#term-onnx); [registry glossary](../../../../glossary.md#term-registry); [dispatch glossary](../../../../glossary.md#term-dispatch); [kernel glossary](../../../../glossary.md#kernel))</span>
- Package direction is `model.operation.reduction -> model.operation`. It must not depend on
  Tensor, datatype, shape, layout, storage, graph, compiler, planning, runtime, prepare, backend,
  or training packages.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare))</span>
- Stop if implementation requires Tensor or Shape behavior, another attributes field/type,
  multiple axes, numerical policy, compatibility validation, dependency, or architecture change.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Package impact

Existing package used:

- `io.github.pho001.synaptik.model.operation` — supplies [`OperationKind`](../../../../glossary.md#operationkind), [`OperationAttrs`](../../../../glossary.md#operationattrs),
  [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs), and generic [`Operation`](../../../../glossary.md#operation) composition.

Package added:

```text
io.github.pho001.synaptik.model.operation.reduction
  Typed aggregate-reduction meanings, normalized single-axis parameters, and arg-max tie policy.
```

The package is a direct operation-family child because aggregate reductions change logical [rank](../../../../glossary.md#term-rank) or
[extent](../../../../glossary.md#term-extent) and are not [elementwise](../../../../glossary.md#term-elementwise). Scans and [softmax](../../../../glossary.md#softmax--log-softmax)-like operations remain separate later concepts
even though the broad capability roadmap currently groups them after reductions.

Type placement:

- `io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind` — public aggregate
  semantic family enum.
- `io.github.pho001.synaptik.model.operation.reduction.AxisReductionAttrs` — public immutable
  normalized single-axis and retained-dimension parameters for ordinary aggregate kinds.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [axis glossary](../../../../glossary.md#term-axis); [dimension glossary](../../../../glossary.md#dimension); [parameters glossary](../../../../glossary.md#term-parameter))</span>
- `io.github.pho001.synaptik.model.operation.reduction.ArgMaxTiePolicy` — public typed tie-breaking
  vocabulary for equal maximum values.
- `io.github.pho001.synaptik.model.operation.reduction.ArgMaxAttrs` — public immutable normalized
  axis, retained-dimension choice, and tie policy for <span data-glossary-navigation="true"> ([normalized axis glossary](../../../../glossary.md#normalized-axis))</span>`ARG_MAX`.
- `ReductionSemanticsTest` — same-package focused test for the cohesive family.

## Required contract

### Aggregate kind vocabulary

Create exactly:

```java
public enum AggregateReductionKind implements OperationKind {
    SUM,
    MEAN,
    PROD,
    MIN,
    MAX,
    ALL,
    ANY,
    ARG_MAX
}
```

The enum declares no project field, explicit constructor, method, nested type, per-[constant](../../../../glossary.md#term-constant) class
body, [alias](../../../../glossary.md#aliasing-and-arena), [symbol](../../../../glossary.md#term-symbol), or [metadata](../../../../glossary.md#term-metadata). [Compiler](../../../../glossary.md#term-compiler)-generated enum machinery is not additional project [API](../../../../glossary.md#term-application-programming-interface-api).
Inherited `Enum.name()` satisfies `OperationKind.name()` and returns the exact constant text.

Semantic meanings:

- `SUM` adds values in the selected reduction domain.
- [`MEAN`](../../../../glossary.md#term-mean) computes the arithmetic mean in the selected reduction domain.
- `PROD` multiplies values in the selected reduction domain.
- `MIN` selects the minimum value in the selected reduction domain.
- `MAX` selects the maximum value in the selected reduction domain.
- `ALL` computes [boolean](../../../../glossary.md#data-type--datatype) conjunction in the selected reduction domain.
- `ANY` computes boolean disjunction in the selected reduction domain.
- `ARG_MAX` selects an index of a maximum value along one [axis](../../../../glossary.md#term-axis) according to `ArgMaxTiePolicy`.

These descriptions identify requested mathematics only. They define no eligible [data type](../../../../glossary.md#data-type--datatype),
accumulation order, empty-input [identity](../../../../glossary.md#term-identity-meanings), [NaN](../../../../glossary.md#term-nan) policy, output [descriptor](../../../../glossary.md#term-descriptor), [gradient](../../../../glossary.md#term-gradient), or execution.

### Single-[axis](../../../../glossary.md#term-axis) aggregate attributes

Create exactly:

```java
public record AxisReductionAttrs(int axis, boolean keepDimensions) implements OperationAttrs {
    public AxisReductionAttrs {
        if (axis < 0) {
            throw new IllegalArgumentException("axis must be non-negative: " + axis);
        }
    }

    @Override
    public int axis() {
        return axis;
    }

    @Override
    public boolean keepDimensions() {
        return keepDimensions;
    }
}
```

The record has exactly two components and no additional instance/static state, nested type,
overload, factory, helper, or [normalization](../../../../glossary.md#term-normalization) method. Construction validates only that the supplied
axis is already non-negative. Zero and every positive `int` are retained unchanged without [Shape](../../../../glossary.md#shape)
or [rank](../../../../glossary.md#term-rank) knowledge. A negative value fails with the exact message shown above.

`keepDimensions == false` requests removal of the selected axis. `true` requests retaining that
axis with [extent](../../../../glossary.md#term-extent) one. The attribute does not construct or store the result Shape.

Record-generated equality and hashing use both components. Generated text is diagnostic only and
is not [serialization](../../../../glossary.md#term-serialization), parsing, [ONNX](../../../../glossary.md#term-onnx), [backend](../../../../glossary.md#backend), or [dispatch](../../../../glossary.md#term-dispatch) behavior. Explicit component accessors
exist only to [carry](../../../../glossary.md#term-carry) complete [Javadoc](../../../../glossary.md#term-javadoc) and return the exact stored primitives.

### Full aggregate composition

Full reduction across every input [axis](../../../../glossary.md#term-axis) is represented exactly by:

```java
Operation fullSum = new Operation(
        AggregateReductionKind.SUM,
        NoOperationAttrs.INSTANCE);
```

The same full-reduction pairing is valid for [`MEAN`](../../../../glossary.md#term-mean), `PROD`, `MIN`, `MAX`, `ALL`, and `ANY`.
`ARG_MAX` must not pair with [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs) under this family contract. There is no all-axis
sentinel, empty axis collection, `FullReductionAttrs`, or full-reduction `keepDimensions` value.

### Single-[axis](../../../../glossary.md#term-axis) aggregate composition

A normalized single-axis reduction is represented exactly by:

```java
AxisReductionAttrs attrs = new AxisReductionAttrs(1, true);
Operation axisSum = new Operation(AggregateReductionKind.SUM, attrs);
```

`AxisReductionAttrs` is valid with `SUM`, [`MEAN`](../../../../glossary.md#term-mean), `PROD`, `MIN`, `MAX`, `ALL`, and `ANY`. It is not
the attributes type for `ARG_MAX`. Generic [`Operation`](../../../../glossary.md#operation) does not enforce these pairings.

### Arg-max tie policy and attributes

Create exactly:

```java
public enum ArgMaxTiePolicy {
    FIRST_INDEX,
    LAST_INDEX
}
```

The enum declares no project field, constructor, method, nested type, [alias](../../../../glossary.md#aliasing-and-arena), index, or [metadata](../../../../glossary.md#term-metadata).
`FIRST_INDEX` requests the smallest [logical index](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) among equal maximum values along the selected
[axis](../../../../glossary.md#term-axis). `LAST_INDEX` requests the largest. The enum does not compare values or define [NaN](../../../../glossary.md#term-nan) behavior.

Create exactly:

```java
public record ArgMaxAttrs(
        int axis,
        boolean keepDimensions,
        ArgMaxTiePolicy tiePolicy) implements OperationAttrs {
    public ArgMaxAttrs {
        if (axis < 0) {
            throw new IllegalArgumentException("axis must be non-negative: " + axis);
        }
        tiePolicy = Objects.requireNonNull(tiePolicy, "tiePolicy");
    }

    @Override
    public int axis() {
        return axis;
    }

    @Override
    public boolean keepDimensions() {
        return keepDimensions;
    }

    @Override
    public ArgMaxTiePolicy tiePolicy() {
        return tiePolicy;
    }
}
```

[Validation](../../../../glossary.md#term-validation) follows component order: reject a negative primitive axis first, then require the tie
policy. Null policy fails with exact `NullPointerException("tiePolicy")`. Valid construction retains
the exact enum reference and primitives. The record adds no defaulting constructor; a later public
[Tensor](../../../../glossary.md#tensor) convenience overload explicitly supplies `FIRST_INDEX`.

Valid composition is:

```java
ArgMaxAttrs attrs = new ArgMaxAttrs(1, false, ArgMaxTiePolicy.FIRST_INDEX);
Operation argMax = new Operation(AggregateReductionKind.ARG_MAX, attrs);
```

`ArgMaxAttrs` duplicates neither input/output data types nor [Shape](../../../../glossary.md#shape). It does not promise [`INT64`](../../../../glossary.md#data-type--datatype)
output, accept a full-reduction sentinel, or perform an index selection.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/reduction/AggregateReductionKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/reduction/AxisReductionAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/reduction/ArgMaxTiePolicy.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/reduction/ArgMaxAttrs.java`

Test:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/reduction/ReductionSemanticsTest.java`

Documentation/[planning](../../../../glossary.md#term-planning) during implementation:

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless a task-related inconsistency requires stopping:

- `docs/api/compile-api.md`
- `docs/api/training-api.md`
- `docs/planning/modules/model/capabilities.md`
- Existing [`OperationKind`](../../../../glossary.md#operationkind), [`OperationAttrs`](../../../../glossary.md#operationattrs), [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs), [`Operation`](../../../../glossary.md#operation), Shape, DataType,
  Tensor, graph, and concrete operation-family Javadocs/tests.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [DataType glossary](../../../../glossary.md#data-type--datatype); [Tensor glossary](../../../../glossary.md#tensor); [graph glossary](../../../../glossary.md#graph))</span>
- Focused architecture documents, ADRs, architecture tests, backend-conformance tests,
  integration tests, and Gradle configuration.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle))</span>

## Maximum [scope](../../../../glossary.md#term-scope)

At most four production files, one focused test, and five documentation/[planning](../../../../glossary.md#term-planning) files: ten paths
total.

No existing Java source or test may change. Do not modify [Operation](../../../../glossary.md#operation) foundations, [Shape](../../../../glossary.md#shape), [DataType](../../../../glossary.md#data-type--datatype),
[Tensor](../../../../glossary.md#tensor), capabilities, [Compile](../../../../glossary.md#compile)/Training [API](../../../../glossary.md#term-application-programming-interface-api), [Gradle](../../../../glossary.md#term-gradle), AGENTS, architecture documents/tests, another
[module](../../../../glossary.md#term-module), or unrelated documentation. Stop if implementation requires another production/test
concept, Tensor/Shape behavior, another documentation file, or an eleventh path. Do not create a
task-0016B specification.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Document `AggregateReductionKind` as backend-independent aggregate-reduction semantics and
  distinguish it from axes, inputs, output descriptors, graph occurrences, numerical policies,
  gradients, and executable support.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [descriptors glossary](../../../../glossary.md#term-descriptor); [graph glossary](../../../../glossary.md#graph); [gradients glossary](../../../../glossary.md#term-gradient))</span>
- Document every enum constant with its mathematical request, valid attributes pairing, and
  deferred input/output/empty-domain/numerical/gradient/backend behavior.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant))</span>
- Document `AxisReductionAttrs` with normalized-axis meaning, keep/remove behavior, immutable
  ownership, validation, record value semantics, and diagnostic-only text.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [validation glossary](../../../../glossary.md#term-validation))</span>
- Document its [canonical](../../../../glossary.md#term-canonical-form) constructor with both `@param` entries and exact `@throws`; document both
  explicit accessors with complete `@return` semantics.
- Document `ArgMaxTiePolicy` and both constants in newcomer-readable terms, including what an equal
  maximum and logical index mean without promising execution or NaN behavior.<span data-glossary-navigation="true"> ([logical index glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [mean glossary](../../../../glossary.md#term-mean); [NaN glossary](../../../../glossary.md#term-nan))</span>
- Document `ArgMaxAttrs` with exact component roles, validation order, explicit default-policy
  deferral, immutable ownership, record semantics, and diagnostic-only text.
- Document its constructor and all three explicit accessors with complete `@param`, `@return`, and
  `@throws` contracts.
- Explain every valid kind/attributes pairing, why full reduction uses [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs) instead
  of `-1`, why [axis](../../../../glossary.md#term-axis) values are already normalized, and why generic [`Operation`](../../../../glossary.md#operation) does not enforce
  family compatibility.
- Review related foundational Javadocs and record why they remain accurate, or stop on an
  out-of-scope inconsistency.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>

## Acceptance criteria

- Exactly four public production types and one focused test are added in the planned reduction
  package; no existing Java source/test changes.
- `AggregateReductionKind` implements [`OperationKind`](../../../../glossary.md#operationkind), declares the exact eight constants in the
  required order, and adds no project state, methods, aliases, traits, or metadata.<span data-glossary-navigation="true"> ([aliases glossary](../../../../glossary.md#aliasing-and-arena); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- `AxisReductionAttrs` has exactly `int axis` and `boolean keepDimensions`, rejects every negative
  axis with the exact message, retains every non-negative axis/boolean, and adds no other API.<span data-glossary-navigation="true"> ([boolean glossary](../../../../glossary.md#data-type--datatype); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- `ArgMaxTiePolicy` declares exactly `FIRST_INDEX` and `LAST_INDEX` in order with no project state
  or behavior.
- `ArgMaxAttrs` has exactly `int axis`, `boolean keepDimensions`, and
  `ArgMaxTiePolicy tiePolicy`; validates negative axis before null tie policy with exact messages;
  retains all valid values and adds no other API.
- Record equality, hashing, and diagnostic text reflect exact components without becoming
  serialization or dispatch contracts.<span data-glossary-navigation="true"> ([serialization glossary](../../../../glossary.md#term-serialization); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- All seven ordinary kinds compose with `AxisReductionAttrs` for single-[axis](../../../../glossary.md#term-axis) reduction and with
  `NoOperationAttrs.INSTANCE` for full reduction, retaining exact kind/attributes references.
- `ARG_MAX` composes with `ArgMaxAttrs` for both tie policies and retained-[dimension](../../../../glossary.md#dimension) choices.
- Kinds remain typed-distinct from equal-name [constants](../../../../glossary.md#term-constant) in other enums, especially [elementwise](../../../../glossary.md#term-elementwise)
  `MIN` and `MAX`.
- No negative all-axes sentinel, Tensor, Shape, data-type/result rules, axis normalization,
  inference, provenance, numerical policy, gradient, graph, compiler, planning, runtime, backend,
  dependency, or architecture behavior is added.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Shape glossary](../../../../glossary.md#shape); [normalization glossary](../../../../glossary.md#term-normalization); [inference glossary](../../../../glossary.md#term-inference); [provenance glossary](../../../../glossary.md#provenance); [gradient glossary](../../../../glossary.md#term-gradient); [graph glossary](../../../../glossary.md#graph); [compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Focused and aggregate model tests, model Javadoc, root tests, reflection/javap/import/scope
  checks, documentation links/formatting, and status synchronization pass.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope); [synchronization glossary](../../../../glossary.md#term-synchronization))</span>
- A separate clean-context documentation-focused agent finalizes all new Javadocs, Tensor API,
  glossary, task evidence, model master plan, and roadmap in the same change and records reasoned
  no-change conclusions for Compile API, Training API, capabilities, architecture, and related
  contracts.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- Task 0016A becomes Complete only after both passes. Task 0016B remains Draft without a detailed
  specification.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) before and after the documentation-focused review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.reduction.ReductionSemanticsTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused test must cover:

- exact packages, public/final enum and record shapes, interfaces, component order/types,
  constructors, fields, declared methods, and absence of nested types;
- exact kind and tie-policy constant vocabularies/order, inherited stable names, standard enum
  identity, and typed distinction from private test-local equal-name kinds and existing families;<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- every negative and representative non-negative axis, both keep-dimension values, constructor
  validation order, exact exception types/messages, and explicit accessor values;
- null tie policy, both tie policies, exact reference retention, and no implicit defaulting;
- generated record equality, hashing, and diagnostic text for equal/different components;
- explicit [axis](../../../../glossary.md#term-axis) [`Operation`](../../../../glossary.md#operation) composition for all seven ordinary kinds with exact references;
- explicit full `Operation` composition for all seven ordinary kinds with the canonical no-attrs
  singleton;<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [singleton glossary](../../../../glossary.md#term-singleton))</span>
- explicit `ARG_MAX` composition for both tie policies and both keep-[dimension](../../../../glossary.md#dimension) choices;
- target-only semantic state with no Tensor, Shape, data type, result, axis collection, sentinel,
  numerical, gradient, factory, registry, compatibility map, or backend dependency.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [data type glossary](../../../../glossary.md#data-type--datatype); [dependency glossary](../../../../glossary.md#term-dependency))</span>

Manually inspect `javap -p -c -s` and reflection for exact enum/record shapes, component order,
constructor validation, explicit accessors, and absence of extra project [API](../../../../glossary.md#term-application-programming-interface-api)/state. [Scan](../../../../glossary.md#term-cumulative-scan) production
imports and [Gradle](../../../../glossary.md#term-gradle) dependencies: kinds and `AxisReductionAttrs` may import only operation
foundations; `ArgMaxAttrs` may additionally import `Objects` and its same-package policy. Confirm
no [Tensor](../../../../glossary.md#tensor), datatype, [shape](../../../../glossary.md#shape), [layout](../../../../glossary.md#layout), storage, [provenance](../../../../glossary.md#provenance), [graph](../../../../glossary.md#graph), [compiler](../../../../glossary.md#term-compiler), [planning](../../../../glossary.md#term-planning), [runtime](../../../../glossary.md#term-runtime),
[prepare](../../../../glossary.md#prepare), [backend](../../../../glossary.md#backend), training, [gradient](../../../../glossary.md#term-gradient), cost, [fusion](../../../../glossary.md#term-fusion), route, [registry](../../../../glossary.md#term-registry), map, reflection, or service
type appears. Validate generated [Javadoc](../../../../glossary.md#term-javadoc), Tensor API status, glossary terminology,
links/anchors/fences/whitespace, exact ten-path [scope](../../../../glossary.md#term-scope), synchronized statuses, package-map placement,
and absence of a task-0016B specification.

## Dependencies

- Task 0005 supplies [`OperationKind`](../../../../glossary.md#operationkind), [`OperationAttrs`](../../../../glossary.md#operationattrs), and [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs).
- Task 0006 supplies [immutable](../../../../glossary.md#term-immutable) generic [`Operation`](../../../../glossary.md#operation) composition and exact reference retention.
- Task 0002 supplies the later Shape axis-normalization contract but is reviewed only; this task
  does not import or modify Shape.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [axis glossary](../../../../glossary.md#term-axis); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- Completed typed operation families establish enum/record/Javadoc/test conventions but are not
  Java dependencies of this family.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

## Follow-up tasks

- 0016B remains Draft for public floating `sum`, [`mean`](../../../../glossary.md#term-mean), and `prod` [Tensor](../../../../glossary.md#tensor) expression construction.
- 0016C remains Draft for public floating reduction `min` and `max` Tensor expressions.
- 0016D remains Draft for public [BOOL](../../../../glossary.md#data-type--datatype) `all` and `any` Tensor expressions.
- 0016E remains Draft for public numeric `argMax` Tensor expressions and [`INT64`](../../../../glossary.md#data-type--datatype) results.
- 0016F remains Draft for masked `sum` and `mean` expression construction; its detailed task will
  decide whether the architecture-compliant representation is composition or a dedicated semantic
  form.
- 0016G–0016J remain Draft for cumulative-sum and [softmax](../../../../glossary.md#softmax--log-softmax) semantic/expression pairs.
- Compiler tasks later own graph capture, reduction canonicalization, autograd expansion, min/max
  tie-gradient policy, and optimization legality.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [autograd glossary](../../../../glossary.md#autograd); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Backend, ONNX, and conformance tasks later own numerical accumulation, empty-domain behavior,
  boolean identities, tie execution, mapping, lowering, storage access, and kernels.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [ONNX glossary](../../../../glossary.md#term-onnx); [conformance glossary](../../../../glossary.md#term-conformance); [boolean glossary](../../../../glossary.md#data-type--datatype); [lowering glossary](../../../../glossary.md#lowering); [kernels glossary](../../../../glossary.md#kernel))</span>

Do not create a detailed follow-up specification in this task.

## Architecture impact

Expected impact: None. The architecture already assigns [backend](../../../../glossary.md#backend)-independent [operation](../../../../glossary.md#operation) semantics
and typed attributes to `modules/model`. The new reduction package refines that [ownership](../../../../glossary.md#term-ownership-meanings) without
[Tensor](../../../../glossary.md#tensor) behavior, dependencies, [shape](../../../../glossary.md#shape) [inference](../../../../glossary.md#term-inference), storage, or executable state.

If implementation requires Tensor/Shape behavior, result inference, numerical policy, backend
[metadata](../../../../glossary.md#term-metadata), another [dependency](../../../../glossary.md#term-dependency), or architecture change, stop and report it.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0002/0005/0006/0013/0015C/0015G/0016A, Tensor API,
Compile API, Training API, glossary, current OperationKind/OperationAttrs/NoOperationAttrs/
Operation and concrete kind/attributes contracts/tests, Shape axis contract, and Java 26 Gradle
configuration.

Implement task 0016A exactly. Add only AggregateReductionKind.java, AxisReductionAttrs.java,
ArgMaxTiePolicy.java, ArgMaxAttrs.java, and ReductionSemanticsTest.java for Java code/tests under
io.github.pho001.synaptik.model.operation.reduction.

The kind enum contains exactly SUM, MEAN, PROD, MIN, MAX, ALL, ANY, ARG_MAX in order. Ordinary
single-axis forms pair with AxisReductionAttrs(non-negative normalized axis, keepDimensions), and
full forms pair with NoOperationAttrs.INSTANCE. Arg-max pairs only with ArgMaxAttrs(axis,
keepDimensions,tiePolicy); the policy enum contains exactly FIRST_INDEX and LAST_INDEX. Follow
exact validation order/messages, immutable value semantics, explicit accessors, and typed
composition rules. Add no negative all-axis sentinel or implicit tie default.

Do not add Tensor methods, Shape/result/data-type inference, multiple axes, scans, softmax, masked
reductions, numerical/empty-input policy, provenance, gradients, graph/compiler/runtime/backend
behavior, factories/registries, dependencies, build/architecture changes, existing Java edits, or
later specs. Stop beyond ten paths or on architecture uncertainty.

Run every specified focused/aggregate test, Javadoc, javap/reflection/import/manual,
documentation/link/whitespace/scope/status check. Then hand the actual diff/evidence to a separate
clean-context documentation agent in the same change. It must inspect source/tests/generated
Javadoc, finalize permitted Javadocs/Tensor API/glossary/planning, record related-contract/
capability/Compile API/Training API/architecture no-change conclusions, and rerun validation.

Update task 0016A, model master plan, and roadmap only for planning status/evidence. Do not mark
0016A Complete until both passes succeed. Leave 0016B Draft without a specification. Do not commit
or push.
```

## Local decisions

- The broad former task 0016 is decomposed into semantic/expression pairs and focused aggregate
  subfamilies. This keeps API/result inference separate from semantic vocabulary and prevents one
  oversized reduction-and-scan task.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [inference glossary](../../../../glossary.md#term-inference); [scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- One `AggregateReductionKind` contains all true aggregate meanings. Typed enum identity safely
  distinguishes reduction <span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>`MIN`/`MAX` from equally named elementwise kinds without prefixes or a
  global registry.<span data-glossary-navigation="true"> ([elementwise glossary](../../../../glossary.md#term-elementwise); [registry glossary](../../../../glossary.md#term-registry))</span>
- Full ordinary reductions use `NoOperationAttrs.INSTANCE`; single-[axis](../../../../glossary.md#term-axis) forms use
  `AxisReductionAttrs`. This removes the legacy `-1` all-axes sentinel and makes parameter absence
  explicit without inventing an axis collection.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter))</span>
- Axis attributes store an already normalized non-negative index. Public Tensor methods later
  accept positive or negative caller axes and normalize through the input Shape before operation
  construction.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Shape glossary](../../../../glossary.md#shape); [operation glossary](../../../../glossary.md#operation))</span>
- `keepDimensions` is spelled descriptively in attributes. It means retention of the one selected
  axis with extent one; full reduction has no retained-dimension option in the selected baseline.<span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent); [dimension glossary](../../../../glossary.md#dimension))</span>
- `ARG_MAX` has a distinct attributes record because its tie policy is intrinsic and it has no
  selected full-reduction form. The record repeats axis and keep-dimension primitives directly
  instead of wrapping another attributes object.
- Null tie policy is rejected rather than silently defaulted. Later convenience overloads supply
  `FIRST_INDEX` explicitly, preserving the observable default without weakening the semantic value.
- Scans and softmax-like operations are not aggregate reductions and therefore receive later
  semantic families rather than broadening this enum or attribute record.<span data-glossary-navigation="true"> ([softmax glossary](../../../../glossary.md#softmax--log-softmax))</span>

## Known limitations

- These contracts represent semantics only and cannot construct or evaluate a [Tensor](../../../../glossary.md#tensor) expression.
- Only one normalized axis or the selected full ordinary reduction form is representable; multiple
  axes and full <span data-glossary-navigation="true"> ([normalized axis glossary](../../../../glossary.md#normalized-axis); [axes glossary](../../../../glossary.md#term-axis))</span>`ARG_MAX` are not part of the current capability baseline.
- Input/output data types, reduced Shape, empty-input behavior, numerical accumulation, NaN/tie
  execution, and gradients remain unspecified here.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [NaN glossary](../../../../glossary.md#term-nan); [gradients glossary](../../../../glossary.md#term-gradient))</span>
- Masked reductions, cumulative sum, softmax, compiler capture, ONNX mapping, backend support,
  storage access, and execution remain planned.<span data-glossary-navigation="true"> ([softmax glossary](../../../../glossary.md#softmax--log-softmax); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [ONNX glossary](../../../../glossary.md#term-onnx); [backend glossary](../../../../glossary.md#backend))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) reviewed the [architecture contract](../../../../glossary.md#architecture-contract) and focused [module](../../../../glossary.md#term-module)/[dependency](../../../../glossary.md#term-dependency)/[lifecycle](../../../../glossary.md#lifecycle) explanations;
documentation and planning rules; roadmap; [model](../../../../glossary.md#term-model) capabilities and master plan; tasks 0002, 0005,
0006, 0013, 0015C, and 0015G; current [operation](../../../../glossary.md#operation) foundations and concrete kind/attributes source
and tests; [Shape](../../../../glossary.md#shape) [normalization](../../../../glossary.md#term-normalization); [Tensor](../../../../glossary.md#tensor)/[Compile](../../../../glossary.md#compile)/Training APIs and glossary; and Java 26 [Gradle](../../../../glossary.md#term-gradle)
configuration.

The read-only `legacy/pre-rewrite` branch was inspected directly. It confirms full and [axis](../../../../glossary.md#term-axis)
`sum`/[`mean`](../../../../glossary.md#term-mean)/`prod`/reduction-min/reduction-max/all/any, axis-only `argMax`, negative caller-axis
normalization, retained [dimensions](../../../../glossary.md#dimension), first/last tie policies, type/result behavior, [gradients](../../../../glossary.md#term-gradient),
non-[contiguous](../../../../glossary.md#term-contiguous) inputs, [ONNX](../../../../glossary.md#term-onnx) mapping, and [backend](../../../../glossary.md#backend) execution evidence. Legacy `-1` sentinels,
operation traits, [mutable](../../../../glossary.md#term-mutable) Tensor gradient callbacks, storage access, [lowering](../../../../glossary.md#lowering), and [kernels](../../../../glossary.md#kernel) are
excluded.

Planning selected four public semantic values and one focused test. Existing generic operation
contracts are sufficient; no Tensor, Shape, [DataType](../../../../glossary.md#data-type--datatype), dependency, or architecture change is
required. The broad 0016 queue was decomposed into 0016A–0016J, but only 0016A has a detailed
specification.

Planning validation:

- `git diff --check` passed, and targeted trailing-whitespace inspection found no matches in the
  three changed planning paths.
- The canonical section scan found every required task-specification section, including exact
  enum/record shapes, package impact, bounded scope, validation, implementation handoff,
  decisions, limitations, and completion-evidence sections.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [scan glossary](../../../../glossary.md#term-cumulative-scan); [scope glossary](../../../../glossary.md#term-scope); [handoff glossary](../../../../glossary.md#term-handoff))</span>
- Every local Markdown file linked from this task, the model master plan, and the roadmap resolves.
  Markdown fences are balanced in all three changed paths.
- Status inspection found 0016A [`Ready`](../../../../glossary.md#term-ready) in this specification, its linked model-master row, and
  its linked roadmap row/current-frontier text. Task 0016B remains `Draft` in both queues, and no
  task-0016B specification exists.
- Package inspection found exactly one planned package with direction
  `model.operation.reduction -> model.operation`; no reverse dependency is introduced.
- Dependency review distinguishes queue order from contract prerequisites. In particular, future
  masked-reduction representation remains undecided rather than gaining speculative dependencies.
- Scope inspection found exactly this new task, the model master plan, and the roadmap changed. No
  Java, test, API, glossary, Gradle, architecture, AGENTS, or other-module path changed.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [module path glossary](../../../../glossary.md#term-module-path))</span>

Implementation and independent documentation validation:

- The implementation pass added exactly the four public production types and one same-package
  focused test under `model.operation.reduction`. No existing Java source or test changed.
- Clean documentation context `/root/review_model_0016a_docs` independently reread the architecture
  contract; focused overview, lifecycle, boundary, and dependency explanations; documentation and
  planning rules; roadmap; model capabilities/master plan; tasks 0002, 0005, 0006, 0013, 0015C,
  0015G, and 0016A; Tensor, Compile, and Training API references; glossary; final source/tests;
  operation foundations and concrete families; Shape axis normalization; generated Javadoc and
  reports; Java 26 build configuration; and the complete workspace diff. It applied General and
  API/Javadoc style to the Java/API/glossary review and Planning style to this task, the model
  master plan, and roadmap. Example format was reviewed; the new API composition snippets are
  focused non-executable contract illustrations rather than a new complete runnable example.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- The documentation pass found `AggregateReductionKind`, `AxisReductionAttrs`,
  `ArgMaxTiePolicy`, and `ArgMaxAttrs` Javadocs complete without revision. They document every
  type, enum constant, record component, constructor parameter, accessor result, expected failure,
  valid family pairing, normalized-axis boundary, full-form no-attributes representation,
  validation order, immutable value semantics, diagnostic-text limit, and deferred inference,
  numerical, gradient, compiler, backend, and execution behavior.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [parameter glossary](../../../../glossary.md#term-parameter); [immutable glossary](../../../../glossary.md#term-immutable); [inference glossary](../../../../glossary.md#term-inference); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- `docs/api/tensor-api.md` now documents the exact eight-kind aggregate vocabulary, ordinary
  full-versus-axis composition, non-negative normalized axes, retained dimensions, explicit
  first/last arg-max tie policy, validation, generic `Operation` compatibility boundary, and
  current-versus-planned status. It explicitly states that no public reduction Tensor method,
  result inference, numerical behavior, gradient rule, or execution exists yet.
- `docs/glossary.md` now records current aggregate semantic support and defines aggregate
  reduction, normalized axis, and arg-max tie policy. It also synchronizes <span data-glossary-navigation="true"> ([aggregate reduction glossary](../../../../glossary.md#aggregate-reduction); [normalized axis glossary](../../../../glossary.md#normalized-axis))</span>[`OperationKind`](../../../../glossary.md#operationkind) and
  [`OperationAttrs`](../../../../glossary.md#operationattrs) distinctions without changing architecture authority.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.operation.reduction.ReductionSemanticsTest` — `BUILD SUCCESSFUL`;
  the XML report contains 11 tests with zero failures, errors, or skips.
- The first independent `./gradlew :modules:model:test` attempt failed before Gradle task execution
  because the restricted sandbox could not open the user Gradle-distribution lock file. The
  approved rerun passed, and the final post-documentation rerun also reported <span data-glossary-navigation="true"> ([distribution glossary](../../../../glossary.md#term-distribution))</span>`BUILD SUCCESSFUL`;
  49 XML suites contain 373 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL`. Generated pages contain all four public
  reduction types, every enum constant, both canonical constructors, all explicit accessors,
  parameter/result/failure documentation, and the semantic/executable boundary.
- `./gradlew test` — `BUILD SUCCESSFUL`; the repository lifecycle completed 36 actionable tasks
  with no failing task.
- `javap -p -c -s` confirmed exact enum constants/order, record components/order, private final
  component fields, explicit accessors, generated record methods, non-negative-axis checks, and
  arg-max axis-before-null-policy validation. Focused reflection tests independently confirmed
  public/final enum and record shapes, exact interfaces, constructors, declared methods, absence
  of nested types and extra project API/state, standard enum identity, and typed distinction from
  equally named kinds.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Production import inspection found only `OperationKind`, `OperationAttrs`,
  [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs), and `java.util.Objects` as applicable. No Tensor, datatype, Shape, layout,
  storage, provenance, graph, compiler, planning, runtime, prepare, backend, engine, trace, or
  training dependency was introduced. The package direction is exactly<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance); [graph glossary](../../../../glossary.md#graph); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [engine glossary](../../../../glossary.md#term-engine); [trace glossary](../../../../glossary.md#trace))</span>
  `model.operation.reduction -> model.operation`.
- The local Markdown target-and-heading checker resolved all 225 links in the five changed
  documentation/planning files with zero errors. Markdown fence counts are balanced; terminology
  and status scans distinguish current semantic values from planned Tensor/executable behavior;
  targeted trailing-whitespace scans found no matches; and <span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>`git diff --check` passed.
- Final scope contains exactly the authorized ten paths: four new production files, one new test,
  Tensor API, glossary, this task, model master plan, and roadmap. Task 0016A is synchronized as
  Complete in all three planning locations. Task 0016B remains Draft, and no task-0016B
  specification exists.
- `docs/api/compile-api.md` remains accurate unchanged because 0016A adds no public expression,
  capture entry point, traversal, inference, canonicalization, artifact, or execution behavior.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [artifact glossary](../../../../glossary.md#term-artifact))</span>
  `docs/api/training-api.md` remains accurate unchanged because no gradient eligibility, gradient
  rule, autograd, optimizer, or session behavior changed. <span data-glossary-navigation="true"> ([autograd glossary](../../../../glossary.md#autograd); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>`capabilities.md` already inventories the
  selected reduction capabilities and distinguishes representation from later public/executable
  support, so it required no status edit.
- `ARCHITECTURE.md`, focused architecture explanations, ADRs, architecture tests,
  backend-conformance tests, and integration tests remain accurate unchanged because the new
  values stay inside model-owned operation semantics and change no module boundary, dependency,
  lifecycle, backend behavior, or end-to-end execution. Java 26 Gradle configuration remains
  accurate unchanged because no build, dependency, preview, incubator, or toolchain behavior
  changed.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [toolchain glossary](../../../../glossary.md#term-toolchain))</span>
- `Shape.normalizeAxis` remains accurate unchanged because it still owns caller-facing
  positive/negative normalization, while the new attributes deliberately accept only an already
  normalized non-negative primitive. `OperationKind`, `OperationAttrs`, `NoOperationAttrs`, and
  `Operation` remain accurate because the family implements their open typed contracts without
  changing generic compatibility validation. Existing concrete families and Tensor expression
  Javadocs/tests remain accurate because typed identity keeps equal names distinct and 0016A adds
  no Tensor construction or provenance.

## Implementation notes

- Added `AggregateReductionKind` with exact order `SUM`, [`MEAN`](../../../../glossary.md#term-mean), `PROD`, `MIN`, `MAX`, `ALL`,
  `ANY`, and `ARG_MAX` and no project [metadata](../../../../glossary.md#term-metadata) or behavior.
- Added `AxisReductionAttrs(axis, keepDimensions)` for ordinary normalized single-axis forms;
  ordinary full forms explicitly use <span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis))</span>`NoOperationAttrs.INSTANCE` without a negative sentinel.
- Added `ArgMaxTiePolicy` with exact `FIRST_INDEX`, `LAST_INDEX` order and
  `ArgMaxAttrs(axis, keepDimensions, tiePolicy)` with axis-first validation and no implicit
  default.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Added one focused 11-test suite covering exact API shapes, validation, record semantics, every
  valid composition, typed identity, and exclusions.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [shapes glossary](../../../../glossary.md#shape); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Finalized Tensor API, glossary, evidence, model master plan, and roadmap without modifying the
  already-complete Javadocs or any executable behavior.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [model glossary](../../../../glossary.md#term-model))</span>

## Completion summary

- Completed changes: Implemented and documented backend-independent aggregate-reduction semantic
  kinds, normalized single-axis/full-form attributes, and explicit arg-max tie policy.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [axis glossary](../../../../glossary.md#term-axis))</span>
- Files changed or created: Exactly four production Java files, one focused test, Tensor API,
  glossary, this task, model master plan, and roadmap.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [model glossary](../../../../glossary.md#term-model))</span>
- Tests and validation: Focused reduction semantics 11/11, all 373 model tests across 49 suites,
  model Javadoc, root tests, bytecode/reflection/import/dependency checks, generated-documentation
  review, 225 Markdown link/anchor checks, fence/terminology/whitespace checks, exact scope/status
  checks, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc); [bytecode glossary](../../../../glossary.md#term-bytecode); [dependency glossary](../../../../glossary.md#term-dependency); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` passed. The one restricted-cache lock failure and approved passing
  rerun are recorded above.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache))</span>
- Documentation-agent review: Clean context `/root/review_model_0016a_docs` completed the
  independent pass using General, API/Javadoc, and Planning profiles.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: Tensor API and glossary now describe current reduction semantics,
  full-versus-axis composition, normalized axes, retained dimensions, tie policy, and the explicit
  boundary before public Tensor reduction methods, inference, gradients, and execution. Compile
  API, Training API, capabilities, architecture/ADRs/tests, conformance/integration tests, and
  build configuration remain accurate unchanged for the recorded reasons.<span data-glossary-navigation="true"> ([dimensions glossary](../../../../glossary.md#dimension); [inference glossary](../../../../glossary.md#term-inference); [gradients glossary](../../../../glossary.md#term-gradient); [Compile glossary](../../../../glossary.md#compile); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Javadoc review: All four new production type/member contracts are complete unchanged; related
  Shape, operation-foundation, concrete-family, and Tensor expression Javadocs remain accurate.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [operation glossary](../../../../glossary.md#operation))</span>
- Glossary impact: Added project-specific aggregate reduction, normalized axis, and arg-max tie
  policy terminology and synchronized operation-kind/attribute status.<span data-glossary-navigation="true"> ([aggregate reduction glossary](../../../../glossary.md#aggregate-reduction); [normalized axis glossary](../../../../glossary.md#normalized-axis))</span>
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0016A. Task 0016B remains Draft without a detailed
  specification.

Status: Complete
