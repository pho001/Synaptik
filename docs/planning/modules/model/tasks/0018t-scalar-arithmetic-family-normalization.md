# Task 0018T: [Scalar](../../../../glossary.md#term-scalar) Arithmetic Family [Normalization](../../../../glossary.md#term-normalization)

**Editorial reading key (current context, not a status update).**

This completed record concerns floating-only vocabulary normalization, not execution; its follow-up
statuses and evidence remain historical. The caller supplies a Tensor receiver and exact typed
scalar to Tensor’s facade; the scalar helper checks receiver/type compatibility, while Operation
checks family-owned attributes and input/output counts. ScalarValueAttrs retains the scalar as
immutable metadata, not a second Tensor input. For example, FLOAT32 multiplication needs a FLOAT32
ScalarValue; the double convenience supplies FLOAT64, never inferred narrowing. Fresh storage-free
results describe producers, not calculated values or owned result buffers.

For current eligibility and compiler distinctions, follow [scalar arithmetic and
clamps](../../../../reference/tensor/elementwise.md#scalar-arithmetic-and-clamp-expressions):
selected integral forms and later compiler rules are outside this task’s scope. Pairwise
minimum/maximum differ from aggregate min/max; clampMin records MAX, while range CLAMP remains one
producer with ordered extrema meaning. NaNs propagate and opposite signed zeros retain directional
extrema rules. [Exact typed
scalars](../../../../reference/tensor/data-types.md#exact-typed-scalar-values) explains bit identity
versus numerical selection. Model semantics defines meaning; execution providers realize supported
routes without redefining it or acquiring storage ownership through semantic responsibility.

## Status

Complete

## Goal

Complete one uniform [floating-point](../../../../glossary.md#term-floating-point) [elementwise](../../../../glossary.md#term-elementwise) arithmetic vocabulary across [Tensor](../../../../glossary.md#tensor)-to-Tensor and
Tensor-to-[scalar](../../../../glossary.md#term-scalar) expressions before the public [model](../../../../glossary.md#term-model) [API](../../../../glossary.md#term-application-programming-interface-api) stabilizes.

The final arithmetic families are parallel:

```text
Tensor-to-Tensor                    Tensor-to-scalar
----------------                   ----------------
ADD  + NoOperationAttrs            ADD  + ScalarValueAttrs
SUB  + NoOperationAttrs            SUB  + ScalarValueAttrs
MUL  + NoOperationAttrs            MUL  + ScalarValueAttrs
DIV  + NoOperationAttrs            DIV  + ScalarValueAttrs
MIN  + NoOperationAttrs            MIN  + ScalarValueAttrs
MAX  + NoOperationAttrs            MAX  + ScalarValueAttrs
POW  + NoOperationAttrs            POW  + ScalarValueAttrs
```

Keep `CLAMP` as the one distinct two-bound scalar [operation](../../../../glossary.md#operation). Remove first-class `CLAMP_MIN` and
`CLAMP_MAX`: public `clampMin` is a convenience for scalar `MAX`, and public `clampMax` is a
convenience for scalar `MIN`.

Atomically rename only the public elementwise Tensor-to-Tensor [extrema](../../../../glossary.md#saturation-pole-and-extremum) methods from `min(Tensor)`
and `max(Tensor)` to `minimum(Tensor)` and `maximum(Tensor)`. Use the same `minimum` and `maximum`
names for scalar overloads. Full and [axis](../../../../glossary.md#term-axis) reductions remain `min(...)` and `max(...)`, so Java
call sites cannot confuse an integral scalar literal with an axis number.

This task constructs [metadata](../../../../glossary.md#term-metadata) only. It does not execute arithmetic, inspect values, implement
[gradients](../../../../glossary.md#term-gradient), or select a [backend route](../../../../glossary.md#backend-route).

## [Mental model](../../../../glossary.md#term-mental-model)

One method name identifies one [elementwise](../../../../glossary.md#term-elementwise) relationship regardless of whether the right operand
is another [Tensor](../../../../glossary.md#tensor) or an exact typed [scalar](../../../../glossary.md#term-scalar):

```java
Tensor tensorSum = left.add(right);
Tensor scalarSum = left.add(ScalarValue.float32(2.0f));

Tensor tensorMinimum = left.minimum(right);
Tensor scalarMinimum = left.minimum(ScalarValue.float32(0.0f));

Tensor reducedMinimum = left.min();
```

The first pair creates different typed [operation](../../../../glossary.md#operation) families because their structural inputs differ:

```text
tensorSum
  operation = BinaryArithmeticKind.ADD + NoOperationAttrs
  inputs    = [left, right]

scalarSum
  operation = ScalarElementwiseKind.ADD + ScalarValueAttrs(FLOAT32 2.0)
  inputs    = [left]
```

The scalar [parameter](../../../../glossary.md#term-parameter) is [immutable](../../../../glossary.md#term-immutable) operation [metadata](../../../../glossary.md#term-metadata) rather than an [eager](../../../../glossary.md#term-eager-evaluation) [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) Tensor. This
avoids allocating a second Tensor [identity](../../../../glossary.md#term-identity-meanings) and storage merely to represent a [constant](../../../../glossary.md#term-constant) parameter,
while keeping the [compiler](../../../../glossary.md#term-compiler)-visible operation explicit.

`minimum` and `maximum` distinguish pairwise [element](../../../../glossary.md#term-element) selection from [aggregate reduction](../../../../glossary.md#aggregate-reduction):

```text
input.minimum(other)  -> one value for every broadcast output position
input.minimum(value)  -> one value for every input position
input.min()            -> one aggregate value over the full input
input.min(axis)        -> one aggregate value per remaining coordinate
```

## Selected [API](../../../../glossary.md#term-application-programming-interface-api)

### [Tensor](../../../../glossary.md#tensor)-to-Tensor arithmetic

The final public methods are exactly:

```java
public Tensor add(Tensor right)
public Tensor sub(Tensor right)
public Tensor mul(Tensor right)
public Tensor div(Tensor right)
public Tensor minimum(Tensor right)
public Tensor maximum(Tensor right)
public Tensor pow(Tensor right)
```

Remove `min(Tensor)` and `max(Tensor)` atomically. Add no [alias](../../../../glossary.md#aliasing-and-arena), deprecated bridge, forwarding
method, or alternate spelling. The existing reduction overloads named `min` and `max` remain
unchanged.

### [Tensor](../../../../glossary.md#tensor)-to-[scalar](../../../../glossary.md#term-scalar) arithmetic

Every arithmetic kind has one exact typed overload and one retained exact-[FLOAT64](../../../../glossary.md#term-float64) convenience:

```java
public Tensor add(ScalarValue value)
public Tensor add(double value)
public Tensor sub(ScalarValue value)
public Tensor sub(double value)
public Tensor mul(ScalarValue value)
public Tensor mul(double value)
public Tensor div(ScalarValue value)
public Tensor div(double value)
public Tensor minimum(ScalarValue value)
public Tensor minimum(double value)
public Tensor maximum(ScalarValue value)
public Tensor maximum(double value)
public Tensor pow(ScalarValue value)
public Tensor pow(double value)
```

The `double` overloads construct `ScalarValue.float64(value)` and delegate to their typed
overloads. They never infer the receiver type or narrow the [parameter](../../../../glossary.md#term-parameter). Therefore a [FLOAT32](../../../../glossary.md#term-float32) or
[BFLOAT16](../../../../glossary.md#term-bfloat16) receiver must use an exact matching [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue).

### Clamp conveniences

Retain these existing methods:

```java
public Tensor clamp(ScalarValue minValue, ScalarValue maxValue)
public Tensor clamp(double minValue, double maxValue)
public Tensor clampMin(ScalarValue minValue)
public Tensor clampMin(double minValue)
public Tensor clampMax(ScalarValue maxValue)
public Tensor clampMax(double maxValue)
```

`clamp` remains one first-class `ScalarElementwiseKind.CLAMP` [operation](../../../../glossary.md#operation) with
`ClampRangeAttrs`. The one-bound methods are conveniences:

```java
clampMin(value) -> maximum(value) -> ScalarElementwiseKind.MAX
clampMax(value) -> minimum(value) -> ScalarElementwiseKind.MIN
```

Each valid convenience call still creates exactly one fresh [producer](../../../../glossary.md#term-producer). It must not create an
intermediate expression, [eager](../../../../glossary.md#term-eager-evaluation) [scalar](../../../../glossary.md#term-scalar) [Tensor](../../../../glossary.md#tensor), or hidden composition.

## Final [scalar](../../../../glossary.md#term-scalar) semantic vocabulary

`ScalarElementwiseKind` must contain exactly these [constants](../../../../glossary.md#term-constant) in this declaration order:

```java
ADD,
SUB,
MUL,
DIV,
MIN,
MAX,
POW,
CLAMP
```

The first seven kinds accept exactly `ScalarValueAttrs`, one logical [Tensor](../../../../glossary.md#tensor) input, and one output.
`CLAMP` accepts exactly `ClampRangeAttrs`, one logical Tensor input, and one output. The family
continues to own these variants through `OperationKind.signatures()`; do not add a [registry](../../../../glossary.md#term-registry),
switch in [`Operation`](../../../../glossary.md#operation), reflective discovery, or string [dispatch](../../../../glossary.md#term-dispatch).

`ScalarValueAttrs` documents its one value as the addend, subtrahend, multiplier, denominator,
minimum candidate, maximum candidate, or [exponent](../../../../glossary.md#term-exponent) selected by the owning kind. It retains the
exact supplied [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) reference and does no receiver-aware [validation](../../../../glossary.md#term-validation).

## Numerical semantics selected here

This task records [operation](../../../../glossary.md#operation) meaning but adds no evaluator. Future [compiler](../../../../glossary.md#term-compiler), [prepare](../../../../glossary.md#prepare), [backend](../../../../glossary.md#backend), and
[conformance](../../../../glossary.md#term-conformance) work must preserve these [model](../../../../glossary.md#term-model)-level choices.

### Arithmetic and power

- `ADD`, `SUB`, `MUL`, and `DIV` mean the ordinary ordered IEEE-754 floating operation in the
  result data type. NaN, infinity, signed-zero, overflow, and underflow classifications follow
  that operation. No NaN payload, exact machine instruction, intermediate precision, or bitwise
  reproducibility is promised by this model-only task.<span data-glossary-navigation="true"> ([mean glossary](../../../../glossary.md#term-mean); [IEEE glossary](../../../../glossary.md#term-ieee); [operation glossary](../../../../glossary.md#operation); [data type glossary](../../../../glossary.md#data-type--datatype); [NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [overflow glossary](../../../../glossary.md#term-overflow); [underflow glossary](../../../../glossary.md#term-underflow); [NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [precision glossary](../../../../glossary.md#term-precision); [model glossary](../../../../glossary.md#term-model))</span>
- `POW` retains its existing ordered base/exponent mathematical request. This task does not choose
  an approximation algorithm or extend its numerical contract beyond the current portable
  meaning.<span data-glossary-navigation="true"> ([exponent glossary](../../../../glossary.md#term-exponent))</span>
- Scalar operations preserve the input data type. Tensor-to-Tensor operations preserve the
  current floating promotion hierarchy and broadcast behavior.<span data-glossary-navigation="true"> ([Scalar glossary](../../../../glossary.md#term-scalar); [Tensor glossary](../../../../glossary.md#tensor))</span>

### Minimum, maximum, and clamp

The same [extrema](../../../../glossary.md#saturation-pole-and-extremum) meaning applies to [Binary](../../../../glossary.md#term-binary-descriptor-and-metadata) and [Scalar](../../../../glossary.md#term-scalar) `MIN`/`MAX`:

- if either compared value is [NaN](../../../../glossary.md#term-nan), the result is NaN; a payload or sign is not promised;
- negative infinity orders below every finite value and positive infinity above every finite
  value;<span data-glossary-navigation="true"> ([negative infinity glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [finite glossary](../../../../glossary.md#term-finite); [positive infinity glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- `minimum(-0.0, +0.0)` is [negative zero](../../../../glossary.md#nan-infinity-and-signed-zero);
- `maximum(-0.0, +0.0)` is [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero); and
- otherwise ordinary numerical order selects the lesser or greater value.

`CLAMP(x, lower, upper)` means `minimum(maximum(x, lower), upper)` under those rules, but remains
one first-class semantic [operation](../../../../glossary.md#operation) rather than two stored [producers](../../../../glossary.md#term-producer). A NaN endpoint is still a
structurally valid exact [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue); the selected extrema policy makes the eventual result NaN
where that endpoint participates. `clampMin` and `clampMax` are public conveniences over `MAX` and
`MIN`, respectively.

This task selects no [gradient](../../../../glossary.md#term-gradient) or subgradient convention at equality, a clamp boundary, NaN, or
[infinity](../../../../glossary.md#term-infinity). [Compiler](../../../../glossary.md#term-compiler)-owned [autograd](../../../../glossary.md#autograd) [planning](../../../../glossary.md#term-planning) must make that decision later.

## [Scope](../../../../glossary.md#term-scope)

- Expand `ScalarElementwiseKind` atomically to the exact final eight-kind vocabulary.
- Remove `CLAMP_MIN` and `CLAMP_MAX` without compatibility [aliases](../../../../glossary.md#aliasing-and-arena).
- Update `ScalarValueAttrs` [Javadoc](../../../../glossary.md#term-javadoc) for all seven [scalar](../../../../glossary.md#term-scalar) roles.
- Preserve `ClampRangeAttrs` declaration, [validation](../../../../glossary.md#term-validation), equality, and behavior unchanged.
- Keep one exact family-owned [signature](../../../../glossary.md#term-signature) for scalar-value variants and one for range-clamp.
- Add exact typed and exact-[FLOAT64](../../../../glossary.md#term-float64) scalar overloads for ADD, SUB, DIV, MIN, and MAX.
- Preserve the existing scalar MUL and POW overloads with the same construction behavior.
- Make `clampMin` and `clampMax` delegate to scalar MAX and MIN with one [producer](../../../../glossary.md#term-producer).
- Rename public [Tensor](../../../../glossary.md#tensor)-to-Tensor `min(Tensor)` and `max(Tensor)` to `minimum(Tensor)` and
  `maximum(Tensor)` without changing `BinaryArithmeticKind.MIN` or `.MAX`.
- Preserve full and [axis](../../../../glossary.md#term-axis) reduction `min`/`max` methods unchanged.
- Preserve all existing Binary arithmetic validation, promotion, broadcasting, result metadata,
  producer/provenance, freshness, and failure ordering.<span data-glossary-navigation="true"> ([Binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [broadcasting glossary](../../../../glossary.md#broadcasting); [metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance))</span>
- Preserve all existing Scalar validation, exact typed-value matching, result metadata,
  producer/provenance, freshness, and failure ordering.
- Document the selected [extrema](../../../../glossary.md#saturation-pole-and-extremum) policy in affected semantic Javadocs and current [API](../../../../glossary.md#term-application-programming-interface-api) guides.
- Update exact enum, operation-signature, helper-surface, public-surface, provenance, delegation,
  failure-order, absence, and freshness tests.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- Update the Tensor API, Compile API current-surface summary, glossary, capability baseline, task,
  model master plan, and roadmap in the required independent documentation pass.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [model glossary](../../../../glossary.md#term-model))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- `rsqrt`, `log1p`, `expm1`, `isFinite`, `isNaN`, or `isInf`; Draft task 0018T1 owns them
- integral or BOOL arithmetic, comparison, reduction, clamp, or public scalar eligibility; task
  0018U owns the selected signed-integral expansion
- `argMin`, multiple-[axis](../../../../glossary.md#term-axis) reductions, `logSumExp`, [variance](../../../../glossary.md#term-variance), [standard deviation](../../../../glossary.md#term-standard-deviation), or norms
- general [scalar](../../../../glossary.md#term-scalar) conversion, [numeric promotion](../../../../glossary.md#numeric-promotion) between a [Tensor](../../../../glossary.md#tensor) and [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue), inferred scalar
  type, boxed `Number`, generic value payloads, or caller-selected [rounding](../../../../glossary.md#term-rounding)
- [eager](../../../../glossary.md#term-eager-evaluation) [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) Tensor creation for a scalar [parameter](../../../../glossary.md#term-parameter)
- float-specific, [BFLOAT16](../../../../glossary.md#term-bfloat16)-specific, primitive integral, or [BOOL](../../../../glossary.md#data-type--datatype) public scalar overloads
- changes to `ScalarValue`, `ClampRangeAttrs`, [`DataType`](../../../../glossary.md#data-type--datatype), `DataTypePromotion`, [Shape](../../../../glossary.md#shape),
  `TensorDescriptor`, [`TensorProducer`](../../../../glossary.md#producer--tensorproducer), `TensorProvenance`, or `TensorFactory`
- numerical evaluation, host-storage reads or writes, constant folding, algebraic simplification,
  canonicalization, common-subexpression elimination, or result interning<span data-glossary-navigation="true"> ([evaluation glossary](../../../../glossary.md#term-evaluation); [host glossary](../../../../glossary.md#term-host); [constant folding glossary](../../../../glossary.md#term-constant-folding); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [common-subexpression elimination glossary](../../../../glossary.md#term-common-subexpression-elimination-cse))</span>
- [gradient](../../../../glossary.md#term-gradient) formulas, subgradient policy, [backward graph](../../../../glossary.md#backward-graph) construction, [optimizer](../../../../glossary.md#term-optimizer), or training
- compiler capture, graph validation, prepare lowering, backend support, kernel selection, fusion,
  runtime behavior, execution, tracing, or engine behavior<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [graph glossary](../../../../glossary.md#graph); [validation glossary](../../../../glossary.md#term-validation); [prepare glossary](../../../../glossary.md#prepare); [lowering glossary](../../../../glossary.md#lowering); [backend glossary](../../../../glossary.md#backend); [kernel glossary](../../../../glossary.md#kernel); [fusion glossary](../../../../glossary.md#term-fusion); [runtime glossary](../../../../glossary.md#term-runtime); [engine glossary](../../../../glossary.md#term-engine))</span>
- dependencies, Gradle, Java version, preview/incubator features, architecture changes, focused
  architecture documentation, architecture tests, another module, or a detailed 0018T1-or-later
  task specification<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [features glossary](../../../../glossary.md#term-software-feature); [module glossary](../../../../glossary.md#term-module))</span>

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>, especially `modules/model` ownership of
  public Tensor and backend-independent operation semantics<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [Tensor glossary](../../../../glossary.md#tensor); [backend glossary](../../../../glossary.md#backend); [operation glossary](../../../../glossary.md#operation))</span>
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Model master plan](../master-plan.md)
- [Task 0014A](0014a-binary-arithmetic-semantic-kinds.md)
- [Task 0014B](0014b-binary-arithmetic-tensor-expressions.md)
- [Task 0014E](0014e-scalar-arithmetic-and-clamp-semantics.md)
- [Task 0014F](0014f-scalar-arithmetic-and-clamp-tensor-expressions.md)
- [Task 0018K](0018k-operation-signature-and-construction-hardening.md)
- [Task 0018N](0018n-typed-scalar-value-contract.md)
- [Task 0018P](0018p-elementwise-semantic-cleanup.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `modules/model` owns both arithmetic kind families, exact scalar attributes, public Tensor
  expression validation, descriptor derivation, and producer/provenance construction.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [validation glossary](../../../../glossary.md#term-validation); [descriptor glossary](../../../../glossary.md#term-descriptor); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>
- [`Operation`](../../../../glossary.md#operation) continues to validate exact family-owned attributes and occurrence cardinality. It
  must not inspect input descriptors or scalar data types.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [cardinality glossary](../../../../glossary.md#term-cardinality))</span>
- The Tensor scalar-construction boundary owns exact receiver/value data-type matching because it
  has the input descriptor.
- Tensor-to-Tensor and Tensor-to-scalar kinds remain distinct typed semantic values even when their
  enum constant names match.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant))</span>
- [`Tensor`](../../../../glossary.md#tensor) remains mutable public API state rather than graph IR. Every valid expression creates a
  fresh storage-free result with one immutable producer occurrence and provenance output index
  zero.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [graph glossary](../../../../glossary.md#graph); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- The public facade may delegate to package-private cohesive expression boundaries; it must not
  absorb inference, execution, or backend decisions.<span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade); [inference glossary](../../../../glossary.md#term-inference); [backend glossary](../../../../glossary.md#backend))</span>
- Compiler owns gradient rules and backward graph construction. Backend prepare owns lowering,
  specialization, fusion, and kernel selection.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [gradient glossary](../../../../glossary.md#term-gradient); [backward graph glossary](../../../../glossary.md#backward-graph); [prepare glossary](../../../../glossary.md#prepare); [lowering glossary](../../../../glossary.md#lowering); [specialization glossary](../../../../glossary.md#term-specialization); [fusion glossary](../../../../glossary.md#term-fusion); [kernel glossary](../../../../glossary.md#kernel))</span>
- [Runtime](../../../../glossary.md#term-runtime) hot paths never consume these operations directly.
- No module/package dependency, registry, service locator, reflective discovery, or architecture
  rule is added.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [registry glossary](../../../../glossary.md#term-registry); [service locator glossary](../../../../glossary.md#term-service-locator))</span>
- If implementation requires an attribute variant beyond the two existing records, a new public
  production type, cross-type scalar conversion, or another module, stop and report the conflict.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.operation.elementwise.binary` — retains the seven
  Tensor-to-Tensor kinds and receives only terminology/numerical-policy Javadoc updates.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- `io.github.pho001.synaptik.model.operation.elementwise.scalar` — owns the normalized eight-kind
  Tensor-to-scalar family and its two existing attribute records.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- `io.github.pho001.synaptik.model.tensor` — owns the public fluent methods and the two existing
  cohesive construction helpers.
- `io.github.pho001.synaptik.model.operation` — owns unchanged signature and operation validation
  consumed by both families.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [operation glossary](../../../../glossary.md#operation); [validation glossary](../../../../glossary.md#term-validation))</span>

Packages added, moved, or removed:

- None.

Type placement:

- `ScalarElementwiseKind` and `ScalarValueAttrs` remain in the scalar semantic package because the
  scalar parameter is operation metadata rather than a second Tensor input.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- `TensorScalarExpressions` remains the one package-private field-free scalar construction
  boundary.
- `TensorBinaryExpressions` remains the one package-private field-free binary construction
  boundary and requires no executable change.<span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata))</span>
- [`Tensor`](../../../../glossary.md#tensor) remains the public fluent [facade](../../../../glossary.md#term-facade); the added overloads must remain thin delegations.

## Required implementation contracts

### [Scalar](../../../../glossary.md#term-scalar) kind and [signature](../../../../glossary.md#term-signature) migration

After implementation:

- `ScalarElementwiseKind.values()` is exactly
  `[ADD, SUB, MUL, DIV, MIN, MAX, POW, CLAMP]`;
- `valueOf("CLAMP_MIN")` and `valueOf("CLAMP_MAX")` fail with ordinary enum [lookup](../../../../glossary.md#term-lookup) failure;
- ADD, SUB, MUL, DIV, MIN, MAX, and POW each return the exact [immutable](../../../../glossary.md#term-immutable)
  `ScalarValueAttrs`/one-input/one-output signature list;
- CLAMP returns the exact immutable `ClampRangeAttrs`/one-input/one-output signature list;
- incompatible kind/attributes combinations fail through the existing [`Operation`](../../../../glossary.md#operation) contract; and
- no new field, nested type, [registry](../../../../glossary.md#term-registry), string mapping, or [mutable](../../../../glossary.md#term-mutable) signature state appears.

### [Scalar](../../../../glossary.md#term-scalar) helper contract

Keep `TensorScalarExpressions` field-free, package-private, final, and non-instantiable. Its method
surface remains exactly:

```java
static Tensor applyScalar(Tensor input, ScalarElementwiseKind kind, ScalarValue value)
static Tensor applyClamp(Tensor input, ScalarValue minValue, ScalarValue maxValue)
private static Tensor create(Tensor input, DataType dataType, Operation operation)
```

`applyScalar` uses this exact [validation](../../../../glossary.md#term-validation) and construction order:

1. null-check `input`, `kind`, then `value`, with the [parameter](../../../../glossary.md#term-parameter) name as the message;
2. reject only `CLAMP`, with exact message `CLAMP requires ClampRangeAttrs`;
3. require a floating input, with existing exact message
   `input must be a floating data type, but was <type>`;
4. require exact scalar/input type equality, with existing exact message
   `scalar data type <scalar> must match input data type <input>`;
5. construct one `ScalarValueAttrs` retaining the exact value reference;
6. construct one [`Operation`](../../../../glossary.md#operation) retaining the exact kind and attributes;
7. construct the current unresolved descriptor with exact input type, exact Shape reference, and
   unchanged <span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [Shape glossary](../../../../glossary.md#shape))</span>`requiresGrad`;
8. call `TensorFactory.createDerived` exactly once with no label and ordered `[input]`.

`applyClamp` preserves its current validation, construction order, failure messages, exact bound
references, and first-class CLAMP [producer](../../../../glossary.md#term-producer) unchanged.

### Public delegation

- Every [Tensor](../../../../glossary.md#tensor)-to-Tensor method delegates exactly once to `TensorBinaryExpressions.apply` with the
  exact kind. Only the two public extrema names change.
- Every typed [scalar](../../../../glossary.md#term-scalar) arithmetic method delegates exactly once to
  `TensorScalarExpressions.applyScalar` with the exact kind.
- Every scalar `double` overload constructs exactly one `ScalarValue.float64(value)` and delegates
  to its corresponding typed overload.
- Typed `clampMin` delegates exactly once to typed `maximum`; typed `clampMax` delegates exactly
  once to typed `minimum`.
- Double `clampMin` delegates through the selected [FLOAT64](../../../../glossary.md#term-float64) maximum path; double `clampMax`
  delegates through the selected FLOAT64 minimum path.
- `clamp` continues to delegate to `applyClamp`, not to two [extrema](../../../../glossary.md#saturation-pole-and-extremum) calls.
- No successful method returns the receiver, reuses an existing producer, creates storage, or
  simplifies identity-like values.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>

### Public naming and absence

After implementation:

- reflection finds `minimum(Tensor)`, `maximum(Tensor)`, `minimum(ScalarValue)`,
  `maximum(ScalarValue)`, `minimum(double)`, and `maximum(double)`;
- reflection does not find `min(Tensor)`, `max(Tensor)`, `min(ScalarValue)`, `max(ScalarValue)`,
  `min(double)`, or `max(double)`;
- reflection still finds reduction `min()`, `min(int)`, `min(int, boolean)`, `max()`, `max(int)`,
  and `max(int, boolean)` unchanged;
- the public method inventory and count match the exact new surface; and
- no compatibility [alias](../../../../glossary.md#aliasing-and-arena), deprecated bridge, or alternate [scalar](../../../../glossary.md#term-scalar) factory entry remains.

## Affected files

Expected production Java:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/binary/BinaryArithmeticKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/scalar/ScalarElementwiseKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/scalar/ScalarValueAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorScalarExpressions.java`

Expected tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/OperationSignatureTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/elementwise/scalar/ScalarElementwiseSemanticsTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryArithmeticTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorScalarElementwiseTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorNumericReductionTest.java`

Expected documentation and [planning](../../../../glossary.md#term-planning):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- `docs/planning/modules/model/tasks/0018t-scalar-arithmetic-family-normalization.md`
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review-only unless an out-of-[scope](../../../../glossary.md#term-scope) discrepancy requires stopping:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/scalar/ClampRangeAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorBinaryExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/datatype/ScalarValue.java`
- `docs/api/training-api.md`
- focused architecture documentation
- architecture tests

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most the exact 18 affected paths listed above: five production
Java files, six test files, and seven documentation/[planning](../../../../glossary.md#term-planning) files including this task.

The original 17-path maximum was explicitly expanded by the user only for
`TensorNumericReductionTest.java` after the first focused [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) exposed its two stale
pairwise `min(Tensor)`/`max(Tensor)` calls. No other production, test, or documentation path is
authorized. If implementation needs a nineteenth path, a new type, helper-surface change, or any
review-only file, stop and report why before editing it.

## Acceptance criteria

- The [scalar](../../../../glossary.md#term-scalar) enum has exactly the selected eight [constants](../../../../glossary.md#term-constant) and exact [signature](../../../../glossary.md#term-signature) variants.
- `CLAMP_MIN` and `CLAMP_MAX` are absent from live production Java and current API/glossary
  documentation; focused tests may name them only to prove enum lookup failure.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup))</span>
- All seven arithmetic relationships have both Tensor-to-Tensor and exact typed scalar public
  forms with matching semantic-kind names.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- All seven scalar relationships also have exact-[FLOAT64](../../../../glossary.md#term-float64) `double` conveniences.
- Public pairwise [extrema](../../../../glossary.md#saturation-pole-and-extremum) are named only `minimum` and `maximum`; reductions remain only `min` and
  `max`.
- `clampMin` produces exactly one scalar MAX [producer](../../../../glossary.md#term-producer) and `clampMax` exactly one scalar MIN
  producer; `clamp` remains exactly one CLAMP producer.
- Exact scalar references, data-type equality, validation order, descriptor metadata, one-input
  producer/provenance, output index zero, freshness, and no-ID-on-preconstruction-failure are
  covered by focused tests.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [descriptor glossary](../../../../glossary.md#term-descriptor); [metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance))</span>
- Binary promotion, broadcasting, ordered two-input provenance, freshness, and error behavior are
  unchanged apart from public extrema method names.<span data-glossary-navigation="true"> ([Binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [broadcasting glossary](../../../../glossary.md#broadcasting))</span>
- The selected NaN, infinity, and signed-zero extrema policy is documented consistently without
  claiming model-level evaluation or backend implementation.<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [model glossary](../../../../glossary.md#term-model); [evaluation glossary](../../../../glossary.md#term-evaluation); [backend glossary](../../../../glossary.md#backend))</span>
- Existing reduction, comparison, unary, typed scalar, clamp-range, Tensor producer/provenance,
  storage, factory, random, indexing, shape, and graph tests remain green.<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary); [shape glossary](../../../../glossary.md#shape); [graph glossary](../../../../glossary.md#graph))</span>
- Public and helper reflection tests prove the exact final surface and removed-name absence.
- Every affected Java declaration and public method has complete accurate Javadoc, including all
  parameters, returns, failures, metadata behavior, and numerical-semantic boundaries.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [parameters glossary](../../../../glossary.md#term-parameter))</span>
- Tensor API, Compile API, glossary, capability baseline, task, master plan, and roadmap describe
  the current final surface without rewriting completed task history.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- Training [API](../../../../glossary.md#term-application-programming-interface-api), architecture docs/tests, `ARCHITECTURE.md`, Gradle, other modules, and dependency
  direction are unchanged, with reasoned no-change conclusions recorded.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools))</span>
- One final `:modules:model:test` [run](../../../../glossary.md#term-run) passes after executable Java stabilizes.
- A separate clean-context documentation-focused pass finalizes Javadocs and documentation in the
  same overall change without repeating successful Java tests unless executable behavior changes
  or it records a concrete reason.
- Model Javadoc, Markdown links/anchors/fences/final-newlines, removed-vocabulary scans, exact-scope
  review, status synchronization, and <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [synchronization glossary](../../../../glossary.md#term-synchronization))</span>`git diff --check` pass.
- Task 0018T is Complete only after both passes and all evidence succeed. Task 0018T1 remains Draft
  without a detailed specification.

## Tests / [validation](../../../../glossary.md#term-validation)

Implementation-focused tests while stabilizing:

```bash
./gradlew :modules:model:test --tests \
  io.github.pho001.synaptik.model.operation.OperationSignatureTest \
  --tests io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseSemanticsTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorBinaryArithmeticTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorScalarElementwiseTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorNumericReductionTest
```

Final Java [checkpoint](../../../../glossary.md#term-checkpoint) after executable code stabilizes:

```bash
./gradlew :modules:model:test
```

The implementation pass records the exact result and hands it to the documentation pass. Do not
repeat that successful suite unless executable Java changes afterward or a concrete risk is
recorded.

Documentation-focused pass after final [Javadoc](../../../../glossary.md#term-javadoc)/documentation edits:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

The documentation pass must also:

- validate local Markdown links and anchors in changed documents;
- validate Java examples affected by the naming/[API](../../../../glossary.md#term-application-programming-interface-api) change against the compiled [model](../../../../glossary.md#term-model) classes;
- scan live production Java and current API/glossary docs for removed `CLAMP_MIN`, `CLAMP_MAX`,
  `min(Tensor)`, and `max(Tensor)` vocabulary while allowing historical completed task records;
- verify the exact 18-path cap and no Java/Gradle/module/architecture/dependency changes outside
  scope;<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [scope glossary](../../../../glossary.md#term-scope))</span>
- verify task 0018T/master/roadmap status agreement and that 0018T1 has no detailed spec; and
- inspect the final diff for stale numerical claims, duplicate terminology, trailing whitespace,
  malformed fences, missing final newlines, and accidental authority changes.

Repository-wide validation is deferred to the capability-reset checkpoint after task 0018V and
[CI](../../../../glossary.md#term-continuous-integration-ci). This single-[module](../../../../glossary.md#term-module) task does not change dependencies, architecture, or build configuration.

## Dependencies

- [Task 0014A](0014a-binary-arithmetic-semantic-kinds.md) — current seven [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) arithmetic kinds.
- [Task 0014B](0014b-binary-arithmetic-tensor-expressions.md) — current promotion, broadcasting,
  and public binary construction.<span data-glossary-navigation="true"> ([broadcasting glossary](../../../../glossary.md#broadcasting))</span>
- [Task 0014E](0014e-scalar-arithmetic-and-clamp-semantics.md) — provisional scalar kinds and
  attribute roles.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- [Task 0014F](0014f-scalar-arithmetic-and-clamp-tensor-expressions.md) — current scalar helper and
  public construction.
- [Task 0018K](0018k-operation-signature-and-construction-hardening.md) — family-owned signature
  validation.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [validation glossary](../../../../glossary.md#term-validation))</span>
- [Task 0018N](0018n-typed-scalar-value-contract.md) — exact typed scalar parameters and
  receiver-aware matching.<span data-glossary-navigation="true"> ([parameters glossary](../../../../glossary.md#term-parameter))</span>
- [Task 0018P](0018p-elementwise-semantic-cleanup.md) — final reciprocal vocabulary and removal of
  backend-route-like fast kinds.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>

All dependencies are Complete.

## Follow-up tasks

- **0018T1 [Unary](../../../../glossary.md#term-unary) numeric gaps and floating diagnostics** — add `RSQRT`, `LOG1P`, `EXPM1`,
  `IS_FINITE`, `IS_NAN`, and `IS_INF` semantics and public construction in a separate cohesive
  task.
- **0018U Integral arithmetic and comparison domains** — deliberately broaden selected arithmetic,
  comparisons, arg-min, and reductions only after overflow and accumulation policies are fixed.<span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow))</span>
- **0018V Multi-[axis](../../../../glossary.md#term-axis) and statistical reductions** — add ordered axes and statistical/norm
  semantics after the numeric foundations are stable.<span data-glossary-navigation="true"> ([norm glossary](../../../../glossary.md#term-norm))</span>
- Later compiler and backend tasks own graph capture validation, gradients, execution, numerical
  conformance, lowering, and kernels.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [validation glossary](../../../../glossary.md#term-validation); [gradients glossary](../../../../glossary.md#term-gradient); [conformance glossary](../../../../glossary.md#term-conformance); [lowering glossary](../../../../glossary.md#lowering); [kernels glossary](../../../../glossary.md#kernel))</span>

## Architecture impact

Expected impact: None.

This is a [backend](../../../../glossary.md#backend)-independent [model](../../../../glossary.md#term-model) vocabulary and public-surface [normalization](../../../../glossary.md#term-normalization) inside the
existing `modules/model` [ownership](../../../../glossary.md#term-ownership-meanings) boundary. It changes no [module](../../../../glossary.md#term-module) [dependency](../../../../glossary.md#term-dependency), [lifecycle](../../../../glossary.md#lifecycle),
architecture rule, or backend contract.

If implementation requires an architecture change, stop and report it before editing
`ARCHITECTURE.md` or focused architecture documentation.

## Implementation prompt

Use this prompt in a separate agentic task/thread with clean context:

```text
You are working in the Synaptik repository.

Read AGENTS.md, ARCHITECTURE.md, documentation/planning rules, roadmap, model capabilities/master
plan, tasks 0014A/0014B/0014E/0014F/0018K/0018N/0018P/0018T, Tensor API, Compile API, Training
API, glossary, and every affected or review-only source/test named by task 0018T in full.

Implement task 0018T exactly. Atomically normalize the complete seven-operation floating
Tensor-to-Tensor/Tensor-to-scalar arithmetic family, rename pairwise public min/max to
minimum/maximum while preserving reduction names, remove CLAMP_MIN/CLAMP_MAX, retain first-class
CLAMP, and make one-bound clamp methods conveniences over scalar MAX/MIN. Preserve exact typed
scalar matching, validation order, descriptor and producer/provenance behavior, and every
architecture boundary. Stay within the exact 18 paths, stop on scope or architecture conflict,
and do not commit or push.

Run focused tests as needed and one final model test after executable Java stabilizes. Then hand
the actual diff and exact evidence to a separate clean-context documentation-focused agent in the
same overall change. That agent must inspect final source/tests, finalize permitted Javadocs,
Tensor/Compile API, glossary, capability/task/master/roadmap documentation, run model Javadoc and
documentation/scope checks, and must not repeat successful Java tests unless executable behavior
changes or it records a concrete reason.

Mark 0018T Complete only after both passes succeed. Leave 0018T1 and every later task Draft
without a detailed specification.
```

## Local decisions

- The [scalar](../../../../glossary.md#term-scalar) family mirrors all seven existing [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) arithmetic kind names and retains one
  `ScalarValueAttrs` variant for those one-[parameter](../../../../glossary.md#term-parameter) requests.
- Pairwise public [extrema](../../../../glossary.md#saturation-pole-and-extremum) use `minimum`/`maximum` so integral scalar literals cannot be confused
  with reduction axes; aggregate <span data-glossary-navigation="true"> ([axes glossary](../../../../glossary.md#term-axis))</span>`min`/`max` remain unchanged.
- `CLAMP` remains one first-class range request. One-bound clamp methods delegate directly to
  scalar MAX/MIN, so each successful convenience call creates one producer and no intermediate.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer))</span>
- The explicitly authorized eighteenth path corrects only stale pairwise method calls in
  `TensorNumericReductionTest`; no further [scope](../../../../glossary.md#term-scope) expansion is permitted.

## Known limitations

- This task constructs model metadata only. It does not calculate values, define gradients,
  capture a graph, lower operations, select kernels, or make a backend executable.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [metadata glossary](../../../../glossary.md#term-metadata); [gradients glossary](../../../../glossary.md#term-gradient); [capture glossary](../../../../glossary.md#term-capture-meanings); [graph glossary](../../../../glossary.md#graph); [kernels glossary](../../../../glossary.md#kernel); [backend glossary](../../../../glossary.md#backend))</span>
- [Scalar](../../../../glossary.md#term-scalar) arithmetic remains floating-only with exact receiver/value data-type equality. Retained
  `double` overloads mean exact [FLOAT64](../../../../glossary.md#term-float64) and do not infer or narrow a scalar type.
- No NaN payload, exact instruction, intermediate precision, bitwise reproducibility, or extrema
  equality/subgradient convention is promised.<span data-glossary-navigation="true"> ([NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [precision glossary](../../../../glossary.md#term-precision); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation context `/root/task_0018t_implementation` ran the focused command containing
  `OperationSignatureTest`, `ScalarElementwiseSemanticsTest`, `TensorBinaryArithmeticTest`,
  `TensorScalarElementwiseTest`, `TensorTest`, and `TensorNumericReductionTest`; it passed after an
  initial test-compilation failure exposed the two stale pairwise calls in the latter test. The
  user explicitly expanded the task from 17 to 18 paths only for that correction.<span data-glossary-navigation="true"> ([compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names))</span>
- The same implementation context ran `./gradlew :modules:model:test` after executable Java
  stabilized. It passed with 715 tests across 88 suites and zero failures, errors, or skips.
  Executable Java did not change afterward; documentation context
  `/root/task_0018t_implementation/task_0018t_docs` reused this evidence and did not rerun Java
  tests.
- The clean documentation context applied the General, API/Javadoc, Planning, and Example
  profiles to the actual final diff, five production contracts, six affected tests, Tensor and
  Compile API references, glossary, capability baseline, task, master plan, and roadmap.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [Planning glossary](../../../../glossary.md#term-planning))</span>
- Its first `./gradlew :modules:model:javadoc` run found two stale reduction-Javadoc links to the
  removed pairwise signatures. After correcting them to <span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [signatures glossary](../../../../glossary.md#term-signature))</span>`minimum(Tensor)` and `maximum(Tensor)`,
  the final run passed with `BUILD SUCCESSFUL`; two actionable tasks executed.
- A Java 26 `ScalarArithmeticExample` compiled and ran against the final model classes. Its six
  checks confirmed binary MIN, scalar MAX, <span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata))</span>`clampMin` as MAX, `clampMax` as MIN, first-class CLAMP,
  and aggregate `min()` retaining [scalar](../../../../glossary.md#term-scalar) result [Shape](../../../../glossary.md#shape).
- The targeted Markdown checker resolved 492 local links, including 138 heading anchors, across
  the seven changed documentation/planning files with zero errors. Generated Javadoc contains the
  new pairwise methods and scalar MIN/MAX/CLAMP links and contains none of the removed vocabulary.
- Live production Java plus current [Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api), [Compile](../../../../glossary.md#compile) API, and glossary scans contain no
  `CLAMP_MIN`, `CLAMP_MAX`, `min(Tensor)`, or `max(Tensor)` contract. Historical completed task
  records retain their original vocabulary as implementation history.
- Final scope inspection found exactly 18 authorized paths: five production Java files, six tests,
  and seven documentation/planning files. No Gradle, dependency, architecture, other-module,
  backend-conformance, or integration path changed. Task 0018T is Complete in the task, master
  plan, and roadmap; 0018T1 and all later rows remain Draft, and no 0018T1 task specification
  exists.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [module glossary](../../../../glossary.md#term-module); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Formatting checks found balanced fences, final newlines, and no trailing whitespace in changed
  documentation. `git diff --check` passed on the final combined change.
- Training API remains accurate unchanged because this task adds no gradient object, autograd,
  optimizer, or training behavior. <span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [autograd glossary](../../../../glossary.md#autograd); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>`ClampRangeAttrs`, [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue), and
  `TensorBinaryExpressions` remain accurate unchanged because their validation, representation,
  and construction behavior did not change. Architecture documents/tests, backend conformance,
  integration tests, Gradle/dependencies, and other modules require no change because ownership,
  module boundaries, build structure, and executable behavior remain unchanged.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>

## Implementation notes

- Expanded `ScalarElementwiseKind` to exact `ADD`, `SUB`, `MUL`, `DIV`, `MIN`, `MAX`, `POW`, and
  `CLAMP`, with exact family-owned [signature](../../../../glossary.md#term-signature) variants and no one-bound clamp kinds.
- Added exact typed and exact-FLOAT64 scalar overloads for the five previously missing arithmetic
  relationships, preserved MUL/POW, and renamed only pairwise Tensor extrema to<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [scalar glossary](../../../../glossary.md#term-scalar); [Tensor glossary](../../../../glossary.md#tensor); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum))</span>
  `minimum`/`maximum`.
- Retained first-class range CLAMP and made one-bound conveniences delegate to scalar MAX/MIN.
  Validation order, exact scalar references, descriptors, producer/provenance output zero,
  freshness, and preconstruction failure effects remain unchanged and covered by focused tests.<span data-glossary-navigation="true"> ([Validation glossary](../../../../glossary.md#term-validation); [descriptors glossary](../../../../glossary.md#term-descriptor); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>
- Finalized all affected Javadocs plus Tensor/Compile APIs, glossary, capability baseline, and
  planning records for the numerical semantics and model-only lifecycle boundary.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [planning glossary](../../../../glossary.md#term-planning); [model glossary](../../../../glossary.md#term-model); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>

## Completion summary

- Completed changes: normalized the full seven-operation binary/scalar arithmetic family,
  pairwise-extrema naming, first-class range clamp, and one-bound clamp conveniences.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [scalar glossary](../../../../glossary.md#term-scalar); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum))</span>
- Files changed or created: exactly five production Java files, six tests, and seven
  documentation/planning files in the authorized 18-path scope.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [scope glossary](../../../../glossary.md#term-scope))</span>
- Tests and validation: reused the passing focused set and final 715-test/88-suite model run;
  final model Javadoc, compiled example, 492-link/138-anchor check, generated-page and removed-
  vocabulary scans, exact scope/status/no-later-spec checks, formatting, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [model glossary](../../../../glossary.md#term-model); [run glossary](../../../../glossary.md#term-run); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`git diff --check`
  passed.
- Documentation-agent review: clean context
  `/root/task_0018t_implementation/task_0018t_docs` independently finalized the change using the
  General, API/Javadoc, Planning, and Example profiles.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Documentation impact: Tensor and Compile APIs, glossary, capability baseline, task, master plan,
  and roadmap describe the final current surface without claiming compiler, gradient, backend, or
  execution support.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [compiler glossary](../../../../glossary.md#term-compiler); [gradient glossary](../../../../glossary.md#term-gradient); [backend glossary](../../../../glossary.md#backend))</span>
- Javadoc review: all five affected production files were reviewed and finalized; two stale
  reduction links found by generated Javadoc were corrected. Adjacent unchanged contracts remain
  accurate for the reasons recorded above.
- Glossary impact: existing arithmetic, operation-kind, provenance, Tensor, reduction, and status
  distinctions now use the final vocabulary; no new reusable domain term was required.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0018T. Task 0018T1 and later work remain Draft.

Status: Complete
