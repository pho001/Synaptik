# Task 0019A2: [One-Hot Encoding](../../../../glossary.md#one-hot-encoding)

**Editorial reading key (current context, not a status update).**

This completed record concerns model construction, not execution support. The caller invokes
`indices.oneHot(depth)`; the receiver supplies the sole integer-index input, while positive `depth`
supplies a static trailing class-axis extent. The helper passes a descriptor and
operation/provenance description to derived construction, not computed BOOL values. A producer
records the producing occurrence; provenance identifies its output. The storage-free result neither
executes encoding nor owns a result buffer. Conceptually, scalar index `2` at depth `4` requests
`[false, false, true, false]`, not a lookup of stored feature values.

Model semantics fixes that meaning; execution providers realize supported routes rather than
redefine it. Semantic responsibility is not storage ownership. Negative or out-of-range indices are
invalid, never all-false rows. Follow the current [one-hot
contract](../../../../reference/tensor/gather.md#one-hot-encoding-expressions) for shape and
metadata details, and [execution
boundary](../../../../reference/tensor/gather.md#failures-and-execution-boundary) for current
capture and qualified execution descriptions. Those later descriptions do not enlarge this
historical task’s metadata-only scope; structural acceptance of a large depth does not promise
materializability.

## Status

Complete

## Goal

Add one first-class, one-input [model](../../../../glossary.md#term-model) semantic and exactly one public [`Tensor`](../../../../glossary.md#tensor) expression:

```java
public Tensor oneHot(long depth)
```

The receiver is an [INT32](../../../../glossary.md#data-type--datatype) or [INT64](../../../../glossary.md#data-type--datatype) Tensor of logical indices. A successful call appends one
positive static trailing [axis](../../../../glossary.md#term-axis) of [extent](../../../../glossary.md#term-extent) `depth` and produces a non-[differentiable](../../../../glossary.md#derivative-and-chain-rule) [BOOL](../../../../glossary.md#data-type--datatype) Tensor.
For each input index `i`, position `i` on the corresponding trailing result axis is `true` and
every other position is `false`.

One-hot is a first-class semantic because the current [metadata](../../../../glossary.md#term-metadata)-only expression vocabulary cannot
derive it without inspecting index values. An [eager](../../../../glossary.md#term-eager-evaluation) range/comparison composition would also
allocate a depth-sized [host](../../../../glossary.md#term-host) Tensor through `TensorFactory` and obscure the single [rank](../../../../glossary.md#term-rank)-changing
logical [operation](../../../../glossary.md#operation) that [compiler](../../../../glossary.md#term-compiler) [capture](../../../../glossary.md#term-capture-meanings) must eventually see.

## Rationale and [mental model](../../../../glossary.md#term-mental-model)

One input position becomes one trailing logical row:

```text
indices Shape [d0, ..., dn] + static depth D
    -> BOOL result Shape [d0, ..., dn, D]
```

For conceptual [INT64](../../../../glossary.md#data-type--datatype) indices `[2, 0, 1]` and depth `3`:

```text
input:  [2, 0, 1]
result: [[false, false, true],
         [true,  false, false],
         [false, true,  false]]
```

The example defines exact logical values but [model](../../../../glossary.md#term-model) construction creates only [metadata](../../../../glossary.md#term-metadata). It does
not read the three input values or allocate result storage. A [scalar](../../../../glossary.md#term-scalar) index `2` with depth `4`
produces [Shape](../../../../glossary.md#shape) `[4]` and conceptually `[false, false, true, false]`.

## [Scope](../../../../glossary.md#term-scope)

- Add public `OneHotKind.ONE_HOT` in the existing [operation](../../../../glossary.md#operation)-index package.
- Add public [immutable](../../../../glossary.md#term-immutable) `OneHotAttrs(long depth)` with intrinsic positive-depth [validation](../../../../glossary.md#term-validation).
- Give the kind exactly one `OneHotAttrs`, one-input, one-output [signature](../../../../glossary.md#term-signature).
- Add one field-free package-private `TensorOneHotExpressions` construction helper.
- Add exactly `public Tensor oneHot(long depth)` to [`Tensor`](../../../../glossary.md#tensor); the receiver is the sole ordered
  producer input and supplies indices rather than values to encode.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer))</span>
- Accept exactly [INT32](../../../../glossary.md#data-type--datatype) and [INT64](../../../../glossary.md#data-type--datatype) receiver [metadata](../../../../glossary.md#term-metadata).
- Append `new StaticDimension(depth)` after every exact receiver [Dimension](../../../../glossary.md#dimension) reference.
- Produce [BOOL](../../../../glossary.md#data-type--datatype), `requiresGrad=false`, unresolved [layout](../../../../glossary.md#layout), no label, and no [host storage](../../../../glossary.md#host-storage).
- Record one fresh producer, output index zero provenance, and one fresh Tensor ID with no
  intermediate Tensor.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>
- Define exact false/true logical values and execution-time index validity while preserving the
  construction-time no-value-inspection boundary.
- Add focused semantic/expression tests and update every current global kind/signature/public API
  inventory affected by the new kind or method.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Finalize Javadocs, Tensor API, Compile API, glossary impact, capabilities, and planning records
  through a mandatory separate clean-context documentation pass.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [planning glossary](../../../../glossary.md#term-planning))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- an alias, static Tensor form, overload, primitive-array convenience, initializer, factory
  method, or eager result<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena); [Tensor glossary](../../../../glossary.md#tensor); [eager glossary](../../../../glossary.md#term-eager-evaluation))</span>
- configurable axis, dynamic/Tensor depth, depth zero, on/off values, output data type, ignore
  index, negative wrapping, clamping, default row, all-false invalid row, or sparse representation<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [data type glossary](../../../../glossary.md#data-type--datatype); [ignore index glossary](../../../../glossary.md#ignore-index))</span>
- composition through `TensorFactory`, range, comparison, broadcast, cast, scatter, or any
  materialized input/result values
- labels, storage allocation, resolved [layout](../../../../glossary.md#layout), input mutation, or reuse of the receiver ID
- [gradient](../../../../glossary.md#term-gradient) rules, straight-through estimation, [autograd](../../../../glossary.md#autograd) construction, or training behavior
- compiler capture, constant folding, bounds proof, backend support/lowering/kernels, preparation,
  runtime checks, execution, conformance, or integration implementation<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [constant folding glossary](../../../../glossary.md#term-constant-folding); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [kernels glossary](../../../../glossary.md#kernel); [runtime glossary](../../../../glossary.md#term-runtime); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- changes to [`DataType`](../../../../glossary.md#data-type--datatype), [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue), [`Shape`](../../../../glossary.md#shape), [`Dimension`](../../../../glossary.md#dimension), `StaticDimension`,
  `DimensionExpressions`, `TensorDescriptor`, `TensorFactory/createDerived`, [`TensorProducer`](../../../../glossary.md#producer--tensorproducer),
  `TensorProvenance`, [`Operation`](../../../../glossary.md#operation), [`OperationKind`](../../../../glossary.md#operationkind), [`OperationAttrs`](../../../../glossary.md#operationattrs), `OperationSignature`, existing
  indexing kinds/attributes/helpers, Gradle, another module, <span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>`ARCHITECTURE.md`, or focused
  architecture documentation
- [dropout](../../../../glossary.md#dropout--inverted-dropout)/[RNG](../../../../glossary.md#graph-rng-state--graphrngstate), sorting/top-K, [linear](../../../../glossary.md#term-linear-meanings), attention, or a detailed specification for 0019B or later

## Exact semantic and [API](../../../../glossary.md#term-application-programming-interface-api) contract

### Package, kind, attributes, and [signature](../../../../glossary.md#term-signature)

Add public types with these exact declaration [shapes](../../../../glossary.md#shape) (their bodies supply the [validation](../../../../glossary.md#term-validation) and
signature behavior specified below):

```java
package io.github.pho001.synaptik.model.operation.index;

public enum OneHotKind implements OperationKind {
    ONE_HOT
}

public record OneHotAttrs(long depth) implements OperationAttrs {
    // Compact constructor specified below.
}
```

`OneHotKind.ONE_HOT` owns one stable [immutable](../../../../glossary.md#term-immutable) signature list containing exactly:

```java
OperationSignature.fixed(OneHotAttrs.class, 1, 1)
```

There is no parameterless or alternate signature. The ordered input is exactly `[indices]`; the
sole output is the dense [BOOL](../../../../glossary.md#data-type--datatype) encoding. `OneHotAttrs` retains the supplied primitive `long`
unchanged after validation and has no [Tensor](../../../../glossary.md#tensor), Shape, [axis](../../../../glossary.md#term-axis), on/off value, output type, policy, or
execution state.

The [canonical](../../../../glossary.md#term-canonical-form) `OneHotAttrs` constructor rejects every `depth <= 0` with exactly:

```text
depth must be positive: <depth>
```

Positive values through `Long.MAX_VALUE` are structurally valid. `long` is selected because Shape
axis extents and `StaticDimension` already use `long`; using `int` would introduce an artificial
public range mismatch, while a Tensor depth would make the result [rank](../../../../glossary.md#term-rank) known but its final [extent](../../../../glossary.md#term-extent)
non-static and broaden the [operation](../../../../glossary.md#operation) into dynamic configuration.

Place both semantic values in `io.github.pho001.synaptik.model.operation.index`. One-hot consumes
logical indices and defines their [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix)-to-indicator meaning, so that existing cohesive
package owns it. Do not add a generic encoding package for two types or place public semantics in
the Tensor package.

### Public receiver role

Add exactly:

```java
public Tensor oneHot(long depth)
```

The receiver is named `indices` in the helper, is never mutated, and is the exact sole [producer](../../../../glossary.md#term-producer)
input. Add no [alias](../../../../glossary.md#aliasing-and-arena) such as `onehot`, `toOneHot`, or `encodeOneHot`, no overload, and no static or
factory form. `Tensor.oneHot(depth)` delegates exactly once to
`TensorOneHotExpressions.apply(this, depth)`.

### Value semantics and index validity

For every logical receiver [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) `p` whose eventual integral value is `i`, and every
trailing result coordinate `j` in `[0, depth)`, exact ideal meaning is:

```text
result[p..., j] = (i == j)
```

The off value is exact [BOOL](../../../../glossary.md#data-type--datatype) `false`; the selected on value is exact BOOL `true`. This is logical
equality, not a numerical approximation, probability, count, or floating encoding. There is no
[rounding](../../../../glossary.md#term-rounding), accumulation, [overflow](../../../../glossary.md#term-overflow), [NaN](../../../../glossary.md#term-nan), [infinity](../../../../glossary.md#term-infinity), or [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero) policy because accepted indices
are signed integral and the result is BOOL.

A valid executed [operation](../../../../glossary.md#operation) requires `0 <= i < depth` for every index value. Negative and
out-of-range indices are invalid at execution: they do not wrap, clamp, select a default, or
produce an all-false row. Construction must not inspect [host storage](../../../../glossary.md#host-storage) even when the receiver is an
[eager](../../../../glossary.md#term-eager-evaluation) [constant](../../../../glossary.md#term-constant). A later [compiler](../../../../glossary.md#term-compiler) may reject provably invalid captured constants; after dynamic
inputs are bound, [backend](../../../../glossary.md#backend) preparation or its [prepared executable](../../../../glossary.md#prepared-executable--preparedexecutable) must prove or check validity and
fail safely. The exact future failure status or exception belongs to the execution contract and
is not invented here. [Runtime](../../../../glossary.md#term-runtime) must execute prepared behavior without inspecting the original
`Operation`.

### [Shape](../../../../glossary.md#shape) and empty cases

For receiver Shape `[d0, ..., dn]`, construct a new Shape whose ordered [Dimensions](../../../../glossary.md#dimension) are:

```text
[exact d0 reference, ..., exact dn reference, new StaticDimension(depth)]
```

- Preserve every existing Dimension object by reference, not merely by [structural equality](../../../../glossary.md#term-structural-equality).
- Append exactly one fresh `StaticDimension(depth)`; do not rewrite or bind existing static,
  named dynamic, or expression Dimensions.
- [Scalar](../../../../glossary.md#term-scalar) receiver Shape `[]` produces [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) Shape `[depth]`.
- A receiver with any zero static [extent](../../../../glossary.md#term-extent) is valid and produces zero logical rows, for example
  `[0, N] -> [0, N, depth]`. There are no index values to validate in a zero-[element](../../../../glossary.md#term-element) input.
- Positive `depth` is required even for a zero-element receiver; depth zero is never an empty-row
  spelling.
- `Long.MAX_VALUE` depth is accepted structurally. Construction does not call
  `Shape.knownElementCount()` or reject a potential total-element-count overflow; later storage or
  execution layers own representability of actual materialization.<span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow); [materialization glossary](../../../../glossary.md#materialization))</span>

### Result [metadata](../../../../glossary.md#term-metadata) and [provenance](../../../../glossary.md#provenance)

Construct exactly one result [descriptor](../../../../glossary.md#term-descriptor):

```text
data type:     BOOL
shape:         input Dimensions plus StaticDimension(depth)
layout:        unresolved (Optional.empty())
requiresGrad:  false
```

The result has no label and no [host storage](../../../../glossary.md#host-storage). It does not inherit receiver type, [gradient](../../../../glossary.md#term-gradient) request,
[layout](../../../../glossary.md#layout), label, storage, or existing provenance. The descriptor's `requiresGrad=false` follows the
current [BOOL](../../../../glossary.md#data-type--datatype) non-differentiability contract and does not define a gradient rule.

Create exactly one [operation](../../../../glossary.md#operation) `new Operation(OneHotKind.ONE_HOT, attrs)` and delegate exactly once
to `TensorFactory.createDerived(descriptor, Optional.empty(), operation, List.of(indices))`.
Every successful call creates one fresh single-output [`TensorProducer`](../../../../glossary.md#producer--tensorproducer), one
`TensorProvenance(producer, 0)`, and one newly allocated [Tensor](../../../../glossary.md#tensor) ID. The [producer](../../../../glossary.md#term-producer) retains exact
ordered input `[indices]`, exact operation/attributes, and the sole exact descriptor. There is no
range Tensor, comparison Tensor, cast, broadcast, intermediate producer, grouped output, or
second ID. Repeated valid calls return [identity](../../../../glossary.md#term-identity-meanings)-distinct results and producers while leaving the
receiver unchanged.

### [Validation](../../../../glossary.md#term-validation) order, exact messages, and ID effects

`TensorOneHotExpressions.apply(indices, depth)` performs exactly this order:

1. `Objects.requireNonNull(indices, "indices")`;
2. read the receiver [descriptor](../../../../glossary.md#term-descriptor) and require exact [INT32](../../../../glossary.md#data-type--datatype) or [INT64](../../../../glossary.md#data-type--datatype) [data type](../../../../glossary.md#data-type--datatype);
3. construct `new OneHotAttrs(depth)`, thereby requiring positive depth;
4. build the appended result [Shape](../../../../glossary.md#shape) and [BOOL](../../../../glossary.md#data-type--datatype) descriptor;
5. construct the exact [operation](../../../../glossary.md#operation);
6. delegate once to `TensorFactory.createDerived`.

Use these exact task-owned failures:

```text
oneHot indices data type must be INT32 or INT64: <type>
depth must be positive: <depth>
```

Null helper input fails with exact message `indices`. Type rejection precedes depth validation;
tests must cover a non-index receiver combined with non-positive depth to lock that order. Depth
validation precedes Shape, descriptor, operation, [producer](../../../../glossary.md#term-producer), [provenance](../../../../glossary.md#provenance), and ID construction.

Every null, type, or depth failure consumes no [Tensor](../../../../glossary.md#tensor) ID. All locally controlled validation and
[metadata](../../../../glossary.md#term-metadata) construction completes before the sole factory delegation. Successful construction
consumes exactly one ID. Identifier exhaustion retains the existing exact message
`tensor identifier space exhausted`; there is no [rollback](../../../../glossary.md#term-rollback) or alternative allocation path.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Model capability baseline](../capabilities.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Model master plan](../master-plan.md)

## Architecture constraints

- Work remains wholly inside model-owned operation semantics, public Tensor metadata
  construction, and their documentation/planning records.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- [`Tensor`](../../../../glossary.md#tensor) remains public mutable API state and is not graph IR. The producer remains pre-capture
  occurrence identity and does not add graph-local IDs to Tensor.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [graph glossary](../../../../glossary.md#graph); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [producer glossary](../../../../glossary.md#term-producer); [capture glossary](../../../../glossary.md#term-capture-meanings); [occurrence glossary](../../../../glossary.md#term-occurrence); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- [`Operation`](../../../../glossary.md#operation) expresses backend-independent one-hot meaning and gains no backend support, route,
  kernel, storage, device, runtime, or prepared-execution state.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [kernel glossary](../../../../glossary.md#kernel); [device glossary](../../../../glossary.md#term-device); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- [Model](../../../../glossary.md#term-model) construction may validate local type/depth/[Shape](../../../../glossary.md#shape) [metadata](../../../../glossary.md#term-metadata) but must not read index values.
- Compiler owns capture, graph-wide validation, constant analysis, canonicalization, and autograd.
  Backend prepare owns lowering, bounds strategy, specialization, and kernel choice. Runtime
  executes prepared schedules only and does not consume <span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [validation glossary](../../../../glossary.md#term-validation); [constant glossary](../../../../glossary.md#term-constant); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [autograd glossary](../../../../glossary.md#autograd); [prepare glossary](../../../../glossary.md#prepare); [lowering glossary](../../../../glossary.md#lowering); [specialization glossary](../../../../glossary.md#term-specialization))</span>`Operation` on the [hot path](../../../../glossary.md#term-hot-path).
- Package direction remains `model.tensor -> model.operation.index`, datatype, shape, and [layout](../../../../glossary.md#layout).
- No architecture, [dependency](../../../../glossary.md#term-dependency), [lifecycle](../../../../glossary.md#lifecycle), [module](../../../../glossary.md#term-module)-boundary, build, or cross-module change.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.operation.index`
- `io.github.pho001.synaptik.model.tensor`
- `io.github.pho001.synaptik.model.datatype`
- `io.github.pho001.synaptik.model.shape`

Packages added or changed:

- No package is added. The existing operation-index package gains the one-hot kind and attributes;
  the existing Tensor package gains one field-free helper and one public facade method.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [Tensor glossary](../../../../glossary.md#tensor); [facade glossary](../../../../glossary.md#term-facade))</span>

Type placement:

- `io.github.pho001.synaptik.model.operation.index.OneHotKind` — public backend-independent
  coordinate-to-indicator semantic identity and family-owned signature.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [identity glossary](../../../../glossary.md#term-identity-meanings); [signature glossary](../../../../glossary.md#term-signature))</span>
- `io.github.pho001.synaptik.model.operation.index.OneHotAttrs` — public immutable positive static
  depth parameter for that semantic.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [parameter glossary](../../../../glossary.md#term-parameter))</span>
- `io.github.pho001.synaptik.model.tensor.TensorOneHotExpressions` — package-private owner of local
  receiver/depth validation, exact Shape/descriptor construction, and derived provenance.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Shape glossary](../../../../glossary.md#shape); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance))</span>
- `io.github.pho001.synaptik.model.tensor.Tensor` — existing public fluent [API](../../../../glossary.md#term-application-programming-interface-api) [owner](../../../../glossary.md#term-owner).

Tests mirror production packages because they inspect exact package-private helper and semantic
surfaces.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/OneHotAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/OneHotKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorOneHotExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/index/OneHotSemanticsTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorOneHotExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/OperationSignatureTest.java`
  — add `OneHotKind`/`OneHotAttrs` to the existing exhaustive production-kind [signature](../../../../glossary.md#term-signature) [matrix](../../../../glossary.md#term-matrix-meanings).
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java` — add the exact
  public name/signature and change the public method count from 161 to 162.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryArithmeticTest.java`
  — change only the shared total public Tensor method count from 161 to 162.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMatmulExpressionTest.java`
  — change only the shared total public Tensor method count from 161 to 162.

Documentation/[planning](../../../../glossary.md#term-planning):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task file
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review unchanged unless final implementation makes a current statement inaccurate: Training [API](../../../../glossary.md#term-application-programming-interface-api);
[DataType](../../../../glossary.md#data-type--datatype)/[ScalarValue](../../../../glossary.md#typed-scalar-value--scalarvalue), [Shape](../../../../glossary.md#shape)/[Dimension](../../../../glossary.md#dimension)/StaticDimension/DimensionExpressions, TensorDescriptor,
TensorFactory/createDerived, [producer](../../../../glossary.md#term-producer)/[provenance](../../../../glossary.md#provenance), existing indexing kinds/helpers, public factory
inventories, architecture/ADRs/tests, [conformance](../../../../glossary.md#term-conformance)/integration, [Gradle](../../../../glossary.md#term-gradle), and other [modules](../../../../glossary.md#term-module).

The existing exhaustive inventories discovered before planning are the production-kind signature
matrix in `OperationSignatureTest` and the public [Tensor](../../../../glossary.md#tensor) inventories/counts in `TensorTest`,
`TensorBinaryArithmeticTest`, and `TensorMatmulExpressionTest`. `OperationKindTest` tests only the
interface contract, `OperationAttrsTest` tests only the marker surface, and `TensorFactoryTest`
tests the unchanged factory surface; none requires a change.

## Maximum [scope](../../../../glossary.md#term-scope)

At most four production, six test, and seven documentation/[planning](../../../../glossary.md#term-planning) files: exactly 17 paths.
`Tensor.java` changes only for the one import/reference needed by its class [Javadoc](../../../../glossary.md#term-javadoc) if applicable,
the exact method, and complete Javadoc. `OperationSignatureTest` changes only for the exhaustive
one-hot family entry/imports. The three historical public [API](../../../../glossary.md#term-application-programming-interface-api) inventory tests change only for the
new exact name/[signature](../../../../glossary.md#term-signature)/count as described above.

The cohesive 17-path exception is justified because the semantic kind, intrinsic attributes,
public expression, focused tests, and already-global inventories must land together in one
compilable state; splitting would temporarily make the exhaustive signature or public API
contracts false. Stop before an eighteenth path, another production/test type, an existing
indexing helper edit, a factory edit, cross-[module](../../../../glossary.md#term-module) work, or any architecture/build change.

## [Javadoc](../../../../glossary.md#term-javadoc) and explanatory documentation requirements

- Give `OneHotKind`, `OneHotAttrs`, `TensorOneHotExpressions`, and `Tensor.oneHot` meaningful,
  detailed Javadocs covering receiver role, exact value formula, type/depth/Shape contract,
  Dimension identity, scalar/empty/maximum-depth cases, result metadata, validation order/messages,
  producer/provenance/ID effects, index-value boundary, and unsupported layers/options.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [Dimension glossary](../../../../glossary.md#dimension); [identity glossary](../../../../glossary.md#term-identity-meanings); [scalar glossary](../../../../glossary.md#term-scalar); [metadata glossary](../../../../glossary.md#term-metadata); [validation glossary](../../../../glossary.md#term-validation); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>
- Every constructor/method input receives `@param`, every non-void method receives `@return`, and
  expected null/argument/exhaustion failures receive <span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument))</span>`@throws` with exact relevant conditions.
- Add the `[2, 0, 1]`, depth-three newcomer example and scalar example to Tensor API. Label values
  conceptual and make clear that construction does not inspect storage or execute encoding.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- Update Compile API only to add one-hot to the current model-expression inventory and preserve
  planned capture/constant-validation/execution boundaries; do not claim compiler support.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [model glossary](../../../../glossary.md#term-model); [capture glossary](../../../../glossary.md#term-capture-meanings); [constant glossary](../../../../glossary.md#term-constant); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- Review Training [API](../../../../glossary.md#term-application-programming-interface-api) and record no change: this task adds fixed `requiresGrad=false` BOOL
  metadata but no gradient rule, parameter role, optimizer, training graph, or execution contract.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [gradient glossary](../../../../glossary.md#term-gradient); [parameter glossary](../../../../glossary.md#term-parameter); [optimizer glossary](../../../../glossary.md#term-optimizer); [training graph glossary](../../../../glossary.md#training-graph))</span>
- Add or revise a glossary entry for one-hot encoding because it is a reusable public semantic;
  distinguish its trailing class/depth axis from indexing selection and state invalid index
  behavior without implying current execution.<span data-glossary-navigation="true"> ([one-hot encoding glossary](../../../../glossary.md#one-hot-encoding); [axis glossary](../../../../glossary.md#term-axis))</span>
- Update capabilities to replace the provisional all-false-invalid-row statement with the final
  invalid-at-execution policy and to describe the implemented contract only after completion.
- Keep task, master plan, and roadmap synchronized. Record reasoned no-change conclusions for all
  reviewed unchanged contracts.

## Acceptance criteria

- Exactly one `ONE_HOT` kind and one `OneHotAttrs(long depth)` record exist in the operation-index
  package with exact one-input/one-output signature.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature))</span>
- Attributes accept every positive `long`, including `Long.MAX_VALUE`, reject zero/negative depth
  with the exact message, retain the value, and expose no extra instance state or API.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Exactly one public `oneHot(long)` exists and the public [Tensor](../../../../glossary.md#tensor) method count is 162.
- Only INT32 and INT64 receiver metadata succeeds; every other current DataType fails with the
  exact message and before depth validation or ID allocation.<span data-glossary-navigation="true"> ([INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [DataType glossary](../../../../glossary.md#data-type--datatype); [validation glossary](../../../../glossary.md#term-validation))</span>
- Result [Shape](../../../../glossary.md#shape) preserves every input [Dimension](../../../../glossary.md#dimension) reference in order and appends one new
  `StaticDimension(depth)`, covering [scalar](../../../../glossary.md#term-scalar), static, zero, named dynamic, and expression cases.
- Result [metadata](../../../../glossary.md#term-metadata) is exact [BOOL](../../../../glossary.md#data-type--datatype), `requiresGrad=false`, unresolved layout, no label, and no storage;
  the receiver is unchanged.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout))</span>
- Every success has one exact `ONE_HOT` [operation](../../../../glossary.md#operation), one [producer](../../../../glossary.md#term-producer) with exact input `[indices]`,
  provenance output index zero, and one fresh ID. There are no intermediate tensors or additional
  IDs.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>
- Construction reads no values. Tests establish semantic metadata without allocating storage or
  pretending to execute the conceptual value example.
- Execution requires every index in `[0, depth)`; negative/out-of-range values are documented as
  invalid with no wrap/clamp/default/all-false behavior, while exact future enforcement remains
  outside this task.
- Scalar, zero-[element](../../../../glossary.md#term-element), `Long.MAX_VALUE` depth, repeated calls, input non-mutation, validation
  order/messages, no-ID failures, identifier exhaustion, exact semantic surfaces, and immutable
  signature/attributes behavior are covered.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- No factory initializer/composition, configurable option, sparse form, gradient/compiler/backend/
  runtime implementation, dependency, Gradle, architecture, focused-architecture, or other-module
  work lands.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>
- Every existing exhaustive global signature/public API inventory is updated now; unrelated
  marker/interface/factory inventories remain unchanged for the recorded reasons.
- Focused tests, exactly one final model test after Java stability, final model Javadoc,
  documentation/link/scope/status/formatting checks, and <span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` pass.
- A separate clean-context documentation pass finalizes all authorized Javadocs/documentation,
  reuses the final Java evidence, and records its context and no-change conclusions.
- Tasks 0019, 0019A, and 0019A1 remain Complete; 0019A2 becomes Complete only after all evidence.
  Tasks 0019B–0019E and every later task remain Draft without detailed specifications.

## Tests / [validation](../../../../glossary.md#term-validation)

Required focused command during implementation:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.index.OneHotSemanticsTest --tests io.github.pho001.synaptik.model.tensor.TensorOneHotExpressionTest --tests io.github.pho001.synaptik.model.operation.OperationSignatureTest --tests io.github.pho001.synaptik.model.tensor.TensorTest
```

Focused tests cover exact kind/record/helper/public surfaces; [signature](../../../../glossary.md#term-signature) immutability and global
coverage; all accepted/rejected data types; all depth boundaries; validation order/messages and ID
effects; [scalar](../../../../glossary.md#term-scalar)/static/zero/dynamic/expression [Shapes](../../../../glossary.md#shape) with exact [Dimension](../../../../glossary.md#dimension) [identity](../../../../glossary.md#term-identity-meanings); exact [BOOL](../../../../glossary.md#data-type--datatype)
[descriptor](../../../../glossary.md#term-descriptor)/[provenance](../../../../glossary.md#provenance)/freshness; one-ID success; receiver non-mutation; conceptual value contract
without [eager](../../../../glossary.md#term-eager-evaluation) execution; and absence of options, composition, or storage inspection.

After executable Java stabilizes, [run](../../../../glossary.md#term-run) exactly one final [model](../../../../glossary.md#term-model) suite:

```bash
./gradlew :modules:model:test
```

The mandatory separate clean-context documentation pass receives and reuses that successful final
Java evidence. It does not rerun Java tests unless it changes executable Java behavior, evidence
is stale/missing, or it records a concrete cross-check risk. After final [Javadoc](../../../../glossary.md#term-javadoc) edits it runs:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

It also validates local Markdown links and anchors, balanced fences, terminology, examples,
generated Javadoc, final newlines, trailing whitespace, exact 17-path [scope](../../../../glossary.md#term-scope), package placement,
the 162-public-method inventory, the exhaustive [operation](../../../../glossary.md#operation)-signature [matrix](../../../../glossary.md#term-matrix-meanings), one [Ready](../../../../glossary.md#term-ready)/active model
frontier, coherent dependencies, synchronized task/master/roadmap status, and absence of a
0019B-or-later detailed specification.

Repository-wide validation is deferred to the selected-modern-operations capability [checkpoint](../../../../glossary.md#term-checkpoint)
after task 0022 and to [CI](../../../../glossary.md#term-continuous-integration-ci). This is task-tier, single-[module](../../../../glossary.md#term-module) semantic [metadata](../../../../glossary.md#term-metadata) with no [dependency](../../../../glossary.md#term-dependency),
architecture, shared-build, or cross-module change.

## Dependencies

- Completed tasks 0001 and 0002 supply exact [INT32](../../../../glossary.md#data-type--datatype)/[INT64](../../../../glossary.md#data-type--datatype)/[BOOL](../../../../glossary.md#data-type--datatype) [metadata](../../../../glossary.md#term-metadata), `long` [Shape](../../../../glossary.md#shape) extents,
  `StaticDimension`, [scalar](../../../../glossary.md#term-scalar)/empty Shapes, and exact [Dimension](../../../../glossary.md#dimension) retention.
- Completed tasks 0005–0007 and 0018K supply operation kinds/attributes, exact family-owned
  signatures, occurrence validation, and descriptor eligibility.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [occurrence glossary](../../../../glossary.md#term-occurrence); [validation glossary](../../../../glossary.md#term-validation); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Completed tasks 0011–0013 and 0018L supply public Tensor state, central derived construction,
  fresh identity, one-producer/output-index provenance, and storage-free result contracts.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [identity glossary](../../../../glossary.md#term-identity-meanings); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>
- Completed tasks 0018N and 0018O clarify that this [operation](../../../../glossary.md#operation) neither uses [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) nor
  changes canonical indexing/bounds policy.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Completed tasks 0019, 0019A, and 0019A1 are the preceding table frontiers and remain unchanged.

## Follow-up tasks

- 0019B remains Draft and solely owns explicit [graph](../../../../glossary.md#graph) [RNG](../../../../glossary.md#graph-rng-state--graphrngstate) and [dropout](../../../../glossary.md#dropout--inverted-dropout).
- 0019C remains Draft for sorting and top-K; 0019D remains Draft for [`linear`](../../../../glossary.md#term-linear-meanings); 0019E remains Draft
  for scaled dot-product attention.<span data-glossary-navigation="true"> ([scaled dot-product attention glossary](../../../../glossary.md#scaled-dot-product-attention))</span>
- Task 0023 may later define compiler-generated backward semantics if an explicit one-hot gradient
  need is selected; this task itself defines no gradient.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Compiler, backend conformance, backend/runtime, and integration work later owns capture,
  constant analysis, safe bounds enforcement, lowering, materialization, and execution.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [runtime glossary](../../../../glossary.md#term-runtime); [capture glossary](../../../../glossary.md#term-capture-meanings); [constant glossary](../../../../glossary.md#term-constant); [lowering glossary](../../../../glossary.md#lowering); [materialization glossary](../../../../glossary.md#materialization))</span>

Do not create another detailed task specification during 0019A2 implementation.

## Architecture impact

Expected impact: None.

If implementation requires an architecture, [dependency](../../../../glossary.md#term-dependency), [lifecycle](../../../../glossary.md#lifecycle), focused-architecture,
cross-[module](../../../../glossary.md#term-module), factory, or [scope](../../../../glossary.md#term-scope) change, stop and report the conflict instead of editing around it.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
Work in Synaptik without commit or push. Read AGENTS.md, ARCHITECTURE.md, current architecture,
documentation/planning rules, roadmap, model capabilities/master plan, completed DataType, Shape,
Tensor, factory/createDerived, provenance, signature-hardening, ScalarValue, indexing/Gather,
reset, 0019, 0019A, and 0019A1 tasks, current affected source/tests/API inventories, and task
0019A2.

Implement docs/planning/modules/model/tasks/0019a2-one-hot-encoding.md exactly inside its 17
authorized paths. Preserve current contracts. Stop on architecture uncertainty, scope overflow,
another type/test/document need, any factory/existing-indexing-helper change, or cross-module work.
Do not commit or push.

Run focused validation and exactly one final model suite after Java stabilizes. Hand the actual
diff and exact Java evidence to a separate clean-context documentation agent in the same overall
change. That agent finalizes Javadocs, Tensor/Compile APIs, glossary, capabilities/planning and
documentation checks while reusing successful Java evidence. Synchronize status only after all
criteria pass; keep 0019, 0019A, and 0019A1 Complete and 0019B/later Draft without specs.
```

## Documentation-agent [handoff](../../../../glossary.md#term-handoff)

Give the separate clean-context documentation agent this task, the complete implementation diff,
exact focused/final [model](../../../../glossary.md#term-model) evidence and whether Java changed afterward, the selected value/[Shape](../../../../glossary.md#shape)/
type/depth/[provenance](../../../../glossary.md#provenance)/bounds policies, the seven authorized documentation paths, and required
[Javadoc](../../../../glossary.md#term-javadoc), Markdown, [scope](../../../../glossary.md#term-scope), and status [validation](../../../../glossary.md#term-validation).

The documentation agent independently reads AGENTS, architecture, documentation rules and the
General/[API](../../../../glossary.md#term-application-programming-interface-api)-Javadoc/[Planning](../../../../glossary.md#term-planning)/Example profiles, this task, actual production/tests/generated
Javadoc, [Tensor](../../../../glossary.md#tensor)/[Compile](../../../../glossary.md#compile)/Training APIs, glossary, capabilities/master/roadmap, and directly related
[DataType](../../../../glossary.md#data-type--datatype)/Shape/[operation](../../../../glossary.md#operation)/[signature](../../../../glossary.md#term-signature)/[descriptor](../../../../glossary.md#term-descriptor)/factory/[producer](../../../../glossary.md#term-producer)/provenance/indexing contracts. It
finalizes Javadocs and explanatory documentation, checks conceptual examples, and records reasoned
no-change conclusions for Training API, foundational and existing indexing contracts,
architecture/ADRs/tests, [conformance](../../../../glossary.md#term-conformance)/integration, [Gradle](../../../../glossary.md#term-gradle), other [modules](../../../../glossary.md#term-module), and Draft follow-ups.

It does not repeat successful Java tests unless executable Java changes, evidence is stale or
missing, or a concrete recorded risk requires a rerun. It records its clean-context identifier,
reused evidence, files/topics reviewed, commands/results, glossary impact, limitations, and
unresolved issues.

## Local decisions

- Kept one-hot as one first-class `ONE_HOT` occurrence rather than composing range, comparison,
  broadcast, cast, or factory allocation. This preserves the exact model meaning and creates no
  intermediate Tensor.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [model glossary](../../../../glossary.md#term-model); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Chose strict eventual bounds: every executed index must satisfy `0 <= i < depth`. Negative and
  out-of-range values are invalid and do not wrap, clamp, select a default, or produce an
  all-false row. Model construction remains value-blind.
- Retained `long` depth through `Long.MAX_VALUE` and avoided element-count calculation during
  construction. Storage and execution layers remain responsible for materializability.
- Preserved every input [Dimension](../../../../glossary.md#dimension) object by reference and appended one fresh
  `StaticDimension(depth)`, including for [scalar](../../../../glossary.md#term-scalar), zero-[element](../../../../glossary.md#term-element), dynamic, and expression [Shapes](../../../../glossary.md#shape).
- Fixed [BOOL](../../../../glossary.md#data-type--datatype) and `requiresGrad=false` as result metadata. This records non-differentiable model
  metadata and does not introduce a gradient rule.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [differentiable glossary](../../../../glossary.md#derivative-and-chain-rule); [gradient glossary](../../../../glossary.md#term-gradient))</span>

## Known limitations

- Model construction cannot validate receiver values because it deliberately does not inspect
  storage; safe invalid-index rejection remains an execution-layer obligation.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- A structurally valid positive depth and Shape need not be materializable by a particular host or
  backend. This task records logical metadata only.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [host glossary](../../../../glossary.md#term-host); [backend glossary](../../../../glossary.md#backend); [metadata glossary](../../../../glossary.md#term-metadata))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) context `/root/plan_0019a2` read the required architecture, documentation/planning,
capability, completed-task, [API](../../../../glossary.md#term-application-programming-interface-api), source, and test contracts; inspected the current global
[signature](../../../../glossary.md#term-signature)/public API inventories; and refined the invalid-index policy before implementation.

Implementation context `/root/task_0019a2_implementation` recorded these final Java results before
the documentation pass, and executable Java did not change afterward:

- The required focused [Gradle](../../../../glossary.md#term-gradle) command selecting `OneHotSemanticsTest`,
  `TensorOneHotExpressionTest`, `OperationSignatureTest`, and `TensorTest` passed 35 tests across
  the four suites: 6, 9, 5, and 15 tests respectively, with zero failures, errors, or skips. The
  exact command remains recorded under Tests / validation.
- Exactly one final `./gradlew :modules:model:test` passed 780 tests across 96 suites with zero
  failures, errors, or skips.
- Targeted [bytecode](../../../../glossary.md#term-bytecode) inspection confirmed one `long` field/constructor/accessor for `OneHotAttrs`,
  the exact `OneHotKind` enum and stable signature list, and a field-free
  `TensorOneHotExpressions` with a private constructor and package-private
  `apply(Tensor, long)`.

Documentation context `/root/task_0019a2_implementation/docs_finalize_0019a2` independently read
the [architecture contract](../../../../glossary.md#architecture-contract); documentation and planning rules; General, API/[Javadoc](../../../../glossary.md#term-javadoc), Planning, and
Example profiles; task, master plan, roadmap, capabilities, APIs, glossary, final source/tests,
generated Javadoc, and directly related [model](../../../../glossary.md#term-model) contracts. It finalized the four production
Javadocs, [Tensor](../../../../glossary.md#tensor) API, [Compile](../../../../glossary.md#compile) API, glossary, capabilities, and planning records. It reused the
successful Java evidence because it changed comments and documentation only. Final documentation
validation passed:

- `./gradlew :modules:model:javadoc` generated model Javadoc successfully after the final edits.
- Generated pages for `OneHotAttrs`, `OneHotKind`, and `Tensor.oneHot(long)` were inspected for the
  formula, Shape/type/depth/provenance boundaries, parameters, returns, and failures.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>
- Local Markdown targets and heading anchors, balanced fences, terminology and conceptual
  examples, final newlines, trailing whitespace, and <span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target))</span>`git diff --check` passed.
- The first three invocations of the temporary local Markdown checker did not complete a valid
  documentation check: the first exposed unavailable `Array#filter_map` in the installed Ruby,
  and the next two exposed an inaccurate slash-to-anchor slug rule. The checker was corrected for
  compatibility and GitHub-style slugs; its final invocation passed all seven authorized
  documentation files with no missing target or anchor.
- The final diff contains exactly 17 authorized paths: four production, six test, and seven
  documentation/planning files. Package placement is exact, Tensor retains 162 public methods,
  and the exhaustive signature matrix includes <span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>`OneHotKind`/`OneHotAttrs`.
- Task/master/roadmap statuses and dependencies are synchronized: 0019, 0019A, 0019A1, and 0019A2
  are Complete; 0019B and every later task remain Draft; no 0019B-or-later detailed specification
  exists, and no model task is currently Ready.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>

Training API required no change because this task adds fixed non-[differentiable](../../../../glossary.md#derivative-and-chain-rule) [BOOL](../../../../glossary.md#data-type--datatype) [metadata](../../../../glossary.md#term-metadata) but
no [gradient](../../../../glossary.md#term-gradient) rule, [parameter](../../../../glossary.md#term-parameter) role, [optimizer](../../../../glossary.md#term-optimizer), [training graph](../../../../glossary.md#training-graph), or execution behavior. [DataType](../../../../glossary.md#data-type--datatype),
[ScalarValue](../../../../glossary.md#typed-scalar-value--scalarvalue), [Shape](../../../../glossary.md#shape)/[Dimension](../../../../glossary.md#dimension)/StaticDimension/DimensionExpressions, TensorDescriptor,
TensorFactory/createDerived, [TensorProducer](../../../../glossary.md#producer--tensorproducer)/TensorProvenance, [Operation](../../../../glossary.md#operation)/[OperationKind](../../../../glossary.md#operationkind)/
[OperationAttrs](../../../../glossary.md#operationattrs)/OperationSignature, and existing indexing/[Gather](../../../../glossary.md#gather) contracts required no change
because implementation reused their existing contracts without modifying them. Architecture,
ADRs, architecture tests, [conformance](../../../../glossary.md#term-conformance)/integration tests, Gradle, and other [modules](../../../../glossary.md#term-module) required no
change because the task adds model-owned semantic metadata and public construction only, with no
[dependency](../../../../glossary.md#term-dependency), [lifecycle](../../../../glossary.md#lifecycle), [backend](../../../../glossary.md#backend), [runtime](../../../../glossary.md#term-runtime), build, or cross-module behavior.

## Implementation notes

- Added exact public `OneHotAttrs`, `OneHotKind.ONE_HOT`, field-free construction helper, and
  `Tensor.oneHot(long)` within existing [operation](../../../../glossary.md#operation)-index and [Tensor](../../../../glossary.md#tensor) packages.
- Added focused semantic/expression coverage and updated the exhaustive production-kind signature
  matrix and all three current public Tensor inventories to 162 methods.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- Final documentation clearly distinguishes conceptual `[2, 0, 1]`/depth-three and scalar values
  from current metadata-only construction. Compile API lists the expression without claiming
  compiler support, and the glossary distinguishes one-hot encoding from Gather selection.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [metadata glossary](../../../../glossary.md#term-metadata); [Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [compiler glossary](../../../../glossary.md#term-compiler); [one-hot encoding glossary](../../../../glossary.md#one-hot-encoding); [Gather glossary](../../../../glossary.md#gather))</span>
- No executable Java changed during the documentation pass, so the final model test evidence
  remains current and was not duplicated.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>

## Completion summary

- Completed changes: Added first-class one-hot semantic [metadata](../../../../glossary.md#term-metadata) and exactly
  `Tensor.oneHot(long)`, with exact validation, Shape, BOOL metadata, provenance, identity, and
  future bounds contracts.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Shape glossary](../../../../glossary.md#shape); [BOOL glossary](../../../../glossary.md#data-type--datatype); [provenance glossary](../../../../glossary.md#provenance); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Files changed or created: Exactly four production, six test, and seven documentation/planning
  files listed under Affected files.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Tests and validation: Focused 35-test command and final 780-test model suite passed in the
  implementation context; final model Javadoc, documentation, scope, status, and whitespace
  checks passed in the documentation context.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: Completed in clean context
  `/root/task_0019a2_implementation/docs_finalize_0019a2` using the required profiles and reused
  Java evidence.
- Documentation impact: Tensor API, Compile API inventory, capabilities, task, master plan, and
  roadmap are synchronized with the implemented metadata-only boundary.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile))</span>
- [Javadoc](../../../../glossary.md#term-javadoc) review: Finalized for `OneHotAttrs`, `OneHotKind`, `TensorOneHotExpressions`, and
  `Tensor.oneHot(long)`; generated Javadoc passed inspection.
- Glossary impact: Added reusable [one-hot encoding](../../../../glossary.md#one-hot-encoding) terminology and its distinction from [Gather](../../../../glossary.md#gather).
- Unresolved issues: None.
- Follow-up required: None.

Status: Complete
