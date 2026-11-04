# Task 0018T1: [Unary](../../../../glossary.md#term-unary) Numeric Gaps and Floating Diagnostics

**Editorial reading key (current context, not a status update).**

This completed record concerns metadata construction, not execution. A caller invoking
`input.log1p()` or `input.isNaN()` passes the receiver Tensor to the corresponding construction
helper: descriptors and input references cross that boundary, not evaluated elements. Unary means
one input; elementwise means positions are treated independently. A descriptor specifies type,
shape, layout and gradient eligibility; provenance identifies the producing occurrence and output.
Transforms preserve floating type and eligibility; classifications describe BOOL results with
eligibility false, not Java booleans or trace diagnostics. These storage-free expressions neither
execute nor own result buffers; Model’s semantic responsibility is not storage ownership.

For example, `isNaN()` describes true at NaN positions without inspecting them; `rsqrt()` maps
negative zero to negative infinity when evaluated. Execution providers realize supported routes, not
redefine meaning. Follow the [current unary
contract](../../../../reference/tensor/elementwise.md#unary-numeric-transforms-and-floating-classifications)
for today’s vocabulary and numerical boundaries: this task’s three-type inventory and
future-conformance language are historical scope. The [metadata
example](../../../../reference/tensor/elementwise.md#complete-transform-and-classification-example)
illustrates descriptions rather than calculated values.

## Status

Complete

## Goal

Complete the selected floating [unary](../../../../glossary.md#term-unary) foundation before integral-domain and statistical-reduction
work begins.

Add three [shape](../../../../glossary.md#shape)-preserving floating transforms:

```text
RSQRT   -> reciprocal square root
LOG1P   -> natural logarithm of one plus the input
EXPM1   -> natural exponential of the input minus one
```

Add three shape-preserving floating classifications with [BOOL](../../../../glossary.md#data-type--datatype) results:

```text
IS_FINITE -> finite value classification
IS_NAN    -> not-a-number classification
IS_INF    -> positive-or-negative infinity classification
```

The two groups deliberately use separate semantic families and construction helpers. The numeric
transforms retain the floating input type and its [gradient](../../../../glossary.md#term-gradient)-eligibility request. Classifications
return fixed BOOL [descriptors](../../../../glossary.md#term-descriptor) with `requiresGrad=false`. Do not hide that result-contract
difference behind a result-kind flag, a generic [registry](../../../../glossary.md#term-registry), or a switch in one catch-all helper.

This task constructs semantic [metadata](../../../../glossary.md#term-metadata) only. It does not evaluate values, inspect storage, define
gradient formulas, [capture](../../../../glossary.md#term-capture-meanings) a [graph](../../../../glossary.md#graph), select an implementation route, or execute [kernels](../../../../glossary.md#kernel).

## [Mental model](../../../../glossary.md#term-mental-model)

### Floating transform

```java
Tensor adjusted = input.log1p();
```

Conceptually records:

```text
TensorProducer
  operation = UnaryElementwiseKind.LOG1P + NoOperationAttrs.INSTANCE
  inputs = [input]
  outputDescriptors = [same floating type, exact input Shape, unresolved layout,
                       unchanged requiresGrad]

adjusted.provenance = (that producer, output index 0)
```

### [Floating classification](../../../../glossary.md#floating-classification)

```java
Tensor invalid = input.isNaN();
```

Conceptually records:

```text
TensorProducer
  operation = FloatingClassificationKind.IS_NAN + NoOperationAttrs.INSTANCE
  inputs = [input]
  outputDescriptors = [BOOL, exact input Shape, unresolved layout, requiresGrad=false]

invalid.provenance = (that producer, output index 0)
```

The classification is a real [operation](../../../../glossary.md#operation) [occurrence](../../../../glossary.md#term-occurrence) rather than an already known Java [boolean](../../../../glossary.md#data-type--datatype). A
public [Tensor](../../../../glossary.md#tensor) expression normally describes values that a future [prepared execution](../../../../glossary.md#prepared-execution--preparedexecution) will produce;
[model](../../../../glossary.md#term-model) construction does not read those values from optional [host storage](../../../../glossary.md#host-storage).

## Why two families

`RSQRT`, `LOG1P`, and `EXPM1` have the same structural and result-[metadata](../../../../glossary.md#term-metadata) contract as the existing
floating [unary](../../../../glossary.md#term-unary) operations. They therefore extend `UnaryElementwiseKind` and reuse
`TensorUnaryExpressions`.

`IS_FINITE`, `IS_NAN`, and `IS_INF` differ in two stable ways:

- their output [data type](../../../../glossary.md#data-type--datatype) is always [BOOL](../../../../glossary.md#data-type--datatype) rather than the input floating type; and
- their result is never [gradient](../../../../glossary.md#term-gradient)-eligible.

Those are [model](../../../../glossary.md#term-model) semantics, not [backend](../../../../glossary.md#backend) details. A separate `FloatingClassificationKind` and
`TensorFloatingClassifications` boundary make the difference explicit without adding attributes,
an interface hierarchy, a broad [operation](../../../../glossary.md#operation) [registry](../../../../glossary.md#term-registry), or result metadata to [`OperationKind`](../../../../glossary.md#operationkind).

The package name uses “classification” rather than “diagnostic” because these are [graph](../../../../glossary.md#graph)-visible
value-producing operations. They are not [trace](../../../../glossary.md#trace) events, logging, [validation](../../../../glossary.md#term-validation) warnings, or debugging
services.

## Selected public [API](../../../../glossary.md#term-application-programming-interface-api)

Add exactly these zero-[argument](../../../../glossary.md#term-argument) public methods to [`Tensor`](../../../../glossary.md#tensor):

```java
public Tensor rsqrt()
public Tensor log1p()
public Tensor expm1()
public Tensor isFinite()
public Tensor isNaN()
public Tensor isInf()
```

All six accept only [`BFLOAT16`](../../../../glossary.md#term-bfloat16), [`FLOAT32`](../../../../glossary.md#term-float32), or [`FLOAT64`](../../../../glossary.md#term-float64) input. [`INT32`](../../../../glossary.md#data-type--datatype), [`INT64`](../../../../glossary.md#data-type--datatype), and [BOOL](../../../../glossary.md#data-type--datatype) are
rejected rather than creating trivially [constant](../../../../glossary.md#term-constant) classifications. Task 0018U owns integral-domain
selection and may reconsider a concrete integral classification use case later; this task adds no
implicit cast or constant-folding rule.

No [aliases](../../../../glossary.md#aliasing-and-arena) such as `inverseSqrt`, `logOnePlus`, `expMinusOne`, `isInfinite`, or `isNotFinite` are
added. No overload accepts a [tolerance](../../../../glossary.md#term-tolerance), output type, algorithm, accuracy mode, or [backend](../../../../glossary.md#backend) hint.

## Final semantic vocabularies

### `UnaryElementwiseKind`

After implementation, the existing enum contains exactly these sixteen [constants](../../../../glossary.md#term-constant) in declaration
order:

```java
ABS,
NEG,
RECIPROCAL,
LOG,
LOG1P,
EXP,
EXPM1,
ERF,
SQRT,
RSQRT,
FLOOR,
CEIL,
SIGN,
RELU,
SIGMOID,
TANH
```

All sixteen variants accept exactly [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs), one logical [Tensor](../../../../glossary.md#tensor) input, and one output.
The three additions are placed beside their mathematical relatives; enum ordinal remains an
implementation detail and is not [serialization](../../../../glossary.md#term-serialization) or [dispatch](../../../../glossary.md#term-dispatch) state.

### `FloatingClassificationKind`

Create one public enum containing exactly:

```java
IS_FINITE,
IS_NAN,
IS_INF
```

Every variant accepts exactly [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs), one logical [Tensor](../../../../glossary.md#tensor) input, and one output. The
enum adds no fields, per-[constant](../../../../glossary.md#term-constant) class bodies, result-type flags, [gradient](../../../../glossary.md#term-gradient) flags, [aliases](../../../../glossary.md#aliasing-and-arena),
symbols, categories, nested types, or other project methods beyond the family-owned [immutable](../../../../glossary.md#term-immutable)
[signature](../../../../glossary.md#term-signature) implementation required by [`OperationKind`](../../../../glossary.md#operationkind).

## Numerical semantics selected here

The [model](../../../../glossary.md#term-model) records the portable mathematical [target](../../../../glossary.md#term-target) and exact special-value classification. It
does not select an algorithm, promise a particular machine instruction, or require bitwise-equal
results across [backends](../../../../glossary.md#backend). A backend claiming these capabilities must later pass the applicable
[conformance](../../../../glossary.md#term-conformance) [tolerance](../../../../glossary.md#term-tolerance) while preserving the exact special-value behavior below.

### `RSQRT`

`rsqrt(x)` means `1 / sqrt(x)` as one first-class semantic request, not two stored operations.

- positive [finite](../../../../glossary.md#term-finite) input produces its positive reciprocal [square root](../../../../glossary.md#norms-elementary-functions-and-probability-distributions);
- [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero) produces [positive infinity](../../../../glossary.md#nan-infinity-and-signed-zero);
- negative zero produces negative infinity, following reciprocal of the sign-preserving principal
  square root;<span data-glossary-navigation="true"> ([negative zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- positive infinity produces positive zero;
- a negative finite value or [negative infinity](../../../../glossary.md#nan-infinity-and-signed-zero) produces [NaN](../../../../glossary.md#term-nan); and
- NaN produces NaN.

No NaN sign or [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication), exact [rounding](../../../../glossary.md#term-rounding), [ULP](../../../../glossary.md#term-unit-in-the-last-place-ulp) bound, intermediate [precision](../../../../glossary.md#term-precision), or bitwise result is
promised by this [model](../../../../glossary.md#term-model) task.

### `LOG1P`

`log1p(x)` means the [natural logarithm](../../../../glossary.md#norms-elementary-functions-and-probability-distributions) of `1 + x` as one first-class request. It exists separately
from `add(1).log()` so a future [compiler](../../../../glossary.md#term-compiler)/[backend](../../../../glossary.md#backend) can preserve useful accuracy near zero without an
[eager](../../../../glossary.md#term-eager-evaluation) [scalar](../../../../glossary.md#term-scalar) [Tensor](../../../../glossary.md#tensor) or forced [decomposition](../../../../glossary.md#term-decomposition).

- input greater than `-1` uses the ordinary real-valued function;
- input exactly `-1` produces [negative infinity](../../../../glossary.md#nan-infinity-and-signed-zero);
- input below `-1`, including negative infinity, produces [NaN](../../../../glossary.md#term-nan);
- [positive infinity](../../../../glossary.md#nan-infinity-and-signed-zero) produces positive infinity;
- positive and [negative zero](../../../../glossary.md#nan-infinity-and-signed-zero) are preserved; and
- NaN produces NaN.

No [NaN payload](../../../../glossary.md#nan-infinity-and-signed-zero), exact [rounding](../../../../glossary.md#term-rounding), [ULP](../../../../glossary.md#term-unit-in-the-last-place-ulp) bound, or implementation algorithm is promised here.

### `EXPM1`

`expm1(x)` means `exp(x) - 1` as one first-class request. It exists separately from
`exp().sub(1)` so a future [compiler](../../../../glossary.md#term-compiler)/[backend](../../../../glossary.md#backend) can preserve useful accuracy near zero.

- finite input uses the ordinary real-valued function rounded to the result format by the eventual
  execution implementation;
- [positive infinity](../../../../glossary.md#nan-infinity-and-signed-zero) produces positive infinity;
- [negative infinity](../../../../glossary.md#nan-infinity-and-signed-zero) produces exactly negative one;
- positive and [negative zero](../../../../glossary.md#nan-infinity-and-signed-zero) are preserved; and
- [NaN](../../../../glossary.md#term-nan) produces NaN.

[Finite](../../../../glossary.md#term-finite) [overflow](../../../../glossary.md#term-overflow) produces positive infinity in the result format; sufficiently negative finite
input may round to negative one. No exact overflow threshold, [NaN payload](../../../../glossary.md#nan-infinity-and-signed-zero), bitwise result, [ULP](../../../../glossary.md#term-unit-in-the-last-place-ulp)
bound, or algorithm is promised by this [model](../../../../glossary.md#term-model) task.

### Floating classifications

Classification uses the logical value represented by the floating input:

- `IS_FINITE` is true for every finite normal, subnormal, positive-zero, or negative-zero value and
  false for both infinities and every NaN;<span data-glossary-navigation="true"> ([subnormal glossary](../../../../glossary.md#term-subnormal); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- `IS_NAN` is true only for NaN, independent of sign, quiet/signaling encoding, or payload, and
  false for finite values and infinities; and<span data-glossary-navigation="true"> ([payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication))</span>
- `IS_INF` is true only for positive or [negative infinity](../../../../glossary.md#nan-infinity-and-signed-zero) and false for [finite](../../../../glossary.md#term-finite) values and [NaN](../../../../glossary.md#term-nan).

For every represented floating value, exactly one of finite, NaN, or [infinity](../../../../glossary.md#term-infinity) classification is
true. The output [carrier](../../../../glossary.md#term-storage-carrier) representation remains the existing [BOOL](../../../../glossary.md#data-type--datatype) contract; this task does not
read or populate it.

## [Gradient](../../../../glossary.md#term-gradient) and accuracy boundary

`requiresGrad` remains [model](../../../../glossary.md#term-model) eligibility [metadata](../../../../glossary.md#term-metadata), not a [derivative](../../../../glossary.md#term-derivative) guarantee:

- `rsqrt`, `log1p`, and `expm1` preserve the exact input flag, including at values where an
  eventual result or derivative is non-finite;<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite))</span>
- `isFinite`, `isNaN`, and `isInf` always set it to false; and
- compiler-owned autograd later decides formulas and discontinuity/domain behavior for the numeric
  transforms and treats classification results as non-differentiable.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [autograd glossary](../../../../glossary.md#autograd); [differentiable glossary](../../../../glossary.md#derivative-and-chain-rule))</span>

The [operation](../../../../glossary.md#operation) names denote the portable mathematical functions. This model task deliberately does
not promise [correct rounding](../../../../glossary.md#term-correct-rounding) or a fixed [ULP](../../../../glossary.md#term-unit-in-the-last-place-ulp) bound. [Backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance) must choose and publish
per-data-type tolerances before execution support is claimed; it may not replace these kinds with
the already rejected public “fast” variants or change their special-value meaning.

## [Scope](../../../../glossary.md#term-scope)

- Extend `UnaryElementwiseKind` with exactly `LOG1P`, `EXPM1`, and `RSQRT` in the selected order.
- Preserve every existing [unary](../../../../glossary.md#term-unary) kind, [signature](../../../../glossary.md#term-signature), typed [identity](../../../../glossary.md#term-identity-meanings), and portable meaning.
- Add public `FloatingClassificationKind` with exactly three parameterless variants.
- Give both families exact [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs), one-input, one-output signatures.
- Add `Tensor.rsqrt`, `Tensor.log1p`, and `Tensor.expm1` through the unchanged
  `TensorUnaryExpressions.apply` construction path.
- Add field-free package-private `TensorFloatingClassifications` with one shared `apply` method for
  `isFinite`, `isNaN`, and `isInf`.
- Add the six exact zero-[argument](../../../../glossary.md#term-argument) methods to [`Tensor`](../../../../glossary.md#tensor) without [aliases](../../../../glossary.md#aliasing-and-arena) or overloads.
- Accept exactly the three current floating data types for both groups.
- Preserve exact input [data type](../../../../glossary.md#data-type--datatype), [Shape](../../../../glossary.md#shape) reference, unresolved [layout](../../../../glossary.md#layout), and `requiresGrad` for the
  numeric transforms.
- Produce exact [BOOL](../../../../glossary.md#data-type--datatype), the input's exact Shape reference, unresolved layout, and
  `requiresGrad=false` for classifications.
- Preserve one exact input reference, one identity-distinct producer, output index zero, no label,
  no storage, and fresh identity for every valid call.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer))</span>
- Preserve [validation](../../../../glossary.md#term-validation)-before-ID-allocation and exact null/type failure order.
- Add exact enum, signature, helper/public-surface, metadata, provenance, freshness, classification,
  invalid-type, domain-noninspection, and no-alias tests.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance))</span>
- Update the [model](../../../../glossary.md#term-model) package map with the cohesive classification package.
- Update Tensor API, Compile API, glossary, capability baseline, task, model master plan, and
  roadmap through the mandatory independent documentation-focused pass.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- numerical evaluation, host-storage reads or writes, output allocation, eager classification, or
  constant folding<span data-glossary-navigation="true"> ([evaluation glossary](../../../../glossary.md#term-evaluation); [host glossary](../../../../glossary.md#term-host); [eager glossary](../../../../glossary.md#term-eager-evaluation); [constant folding glossary](../../../../glossary.md#term-constant-folding))</span>
- `square`, `cbrt`, trigonometric functions, `log2`, `log10`, `exp2`, `isNormal`, `isSubnormal`,
  `isZero`, `isPositive`, `isNegative`, `isNotFinite`, or separate positive/negative-infinity
  operations<span data-glossary-navigation="true"> ([infinity glossary](../../../../glossary.md#term-infinity))</span>
- integral or BOOL classification, implicit cast insertion, data-type promotion, caller-selected
  result type, or another public overload<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- changes to existing `reciprocal`, [`log`](../../../../glossary.md#term-log), [`exp`](../../../../glossary.md#term-expx), `sqrt`, or any other completed [unary](../../../../glossary.md#term-unary) behavior
- decomposition into existing operations, eager scalar constants, canonicalization, algebraic
  simplification, common-subexpression elimination, fusion, or result interning<span data-glossary-navigation="true"> ([decomposition glossary](../../../../glossary.md#term-decomposition); [scalar glossary](../../../../glossary.md#term-scalar); [constants glossary](../../../../glossary.md#term-constant); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [common-subexpression elimination glossary](../../../../glossary.md#term-common-subexpression-elimination-cse); [fusion glossary](../../../../glossary.md#term-fusion))</span>
- attributes, factories, registries, service locators, reflective discovery, generic result-kind
  metadata, category interfaces, or operation-class hierarchies<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata))</span>
- exact rounding, fixed ULP/relative-error promises, backend-specific accuracy flags, algorithm
  choice, approximation variants, or restored <span data-glossary-navigation="true"> ([rounding glossary](../../../../glossary.md#term-rounding); [ULP glossary](../../../../glossary.md#term-unit-in-the-last-place-ulp); [backend glossary](../../../../glossary.md#backend))</span>`FAST_EXP`/`FAST_TANH`
- gradient formulas, subgradient conventions, backward graph construction, optimizer, or training
  behavior<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [backward graph glossary](../../../../glossary.md#backward-graph); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- compiler capture, graph-wide descriptor validation, planning, prepare, backend support, lowering,
  kernel selection, runtime, execution, tracing, or engine behavior<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [graph glossary](../../../../glossary.md#graph); [descriptor glossary](../../../../glossary.md#term-descriptor); [validation glossary](../../../../glossary.md#term-validation); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [lowering glossary](../../../../glossary.md#lowering); [kernel glossary](../../../../glossary.md#kernel); [runtime glossary](../../../../glossary.md#term-runtime); [engine glossary](../../../../glossary.md#term-engine))</span>
- task 0018U integral arithmetic/comparison/reduction work, task 0018V statistical reductions, or
  any later operation family
- changes to [`Operation`](../../../../glossary.md#operation), [`OperationKind`](../../../../glossary.md#operationkind), `OperationSignature`, [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs), DataType,
  Shape, layout, descriptor, Tensor producer/provenance, storage, factory, randoms, or graph model<span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype); [Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [Tensor glossary](../../../../glossary.md#tensor); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance); [model glossary](../../../../glossary.md#term-model))</span>
- dependencies, Gradle, Java version, preview/incubator features, architecture rules, focused
  architecture documents/tests, another module, or a detailed 0018U-or-later task specification<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [features glossary](../../../../glossary.md#term-software-feature); [module glossary](../../../../glossary.md#term-module))</span>

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>, especially model ownership of public
  Tensor and backend-independent operation semantics<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Tensor glossary](../../../../glossary.md#tensor); [backend glossary](../../../../glossary.md#backend); [operation glossary](../../../../glossary.md#operation))</span>
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)
- [Model master plan](../master-plan.md)
- [Task 0014C](0014c-unary-elementwise-semantic-kinds.md)
- [Task 0014D](0014d-unary-elementwise-tensor-expressions.md)
- [Task 0018K](0018k-operation-signature-and-construction-hardening.md)
- [Task 0018P](0018p-elementwise-semantic-cleanup.md)
- [Task 0018T](0018t-scalar-arithmetic-family-normalization.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `modules/model` owns both semantic families, public Tensor methods, local descriptor derivation,
  and producer/provenance construction.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>
- [`OperationKind`](../../../../glossary.md#operationkind) remains a compact family-owned structural contract. It does not receive a result
  data type, differentiability flag, operation category, backend support, or executable behavior.<span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype); [backend glossary](../../../../glossary.md#backend))</span>
- Numeric transforms and floating classifications remain distinct typed kind families even though
  both are parameterless and consume one floating input.
- [`Operation`](../../../../glossary.md#operation) continues to validate only exact attributes type and occurrence cardinality. It has
  no operand descriptor and therefore does not validate floating eligibility or result metadata.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [cardinality glossary](../../../../glossary.md#term-cardinality); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Public [Tensor](../../../../glossary.md#tensor) construction owns local input-type [validation](../../../../glossary.md#term-validation) and result-[descriptor](../../../../glossary.md#term-descriptor) derivation.
- `Tensor` remains mutable public API state rather than graph IR. Each result has one immutable
  identity-bearing producer occurrence and provenance output index zero.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [graph glossary](../../../../glossary.md#graph); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [immutable glossary](../../../../glossary.md#term-immutable); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Compiler owns graph capture, descriptor revalidation, gradient rules, and backward graph
  construction. Backend prepare owns lowering, numerical implementation, specialization, fusion,
  and kernel selection.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [gradient glossary](../../../../glossary.md#term-gradient); [backward graph glossary](../../../../glossary.md#backward-graph); [prepare glossary](../../../../glossary.md#prepare); [lowering glossary](../../../../glossary.md#lowering); [specialization glossary](../../../../glossary.md#term-specialization); [fusion glossary](../../../../glossary.md#term-fusion); [kernel glossary](../../../../glossary.md#kernel))</span>
- [Runtime](../../../../glossary.md#term-runtime) hot paths do not consume these `Operation` values directly.
- Classification terminology must not imply or import `modules/trace`; no trace event or diagnostic
  service is added.<span data-glossary-navigation="true"> ([trace glossary](../../../../glossary.md#trace))</span>
- Package direction remains `model.tensor -> model.operation.elementwise.* -> model.operation`.
  No dependency, registry, service locator, reflection, or classpath discovery is added.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [registry glossary](../../../../glossary.md#term-registry); [service locator glossary](../../../../glossary.md#term-service-locator); [classpath glossary](../../../../glossary.md#term-classpath))</span>
- If implementation needs attributes, a common result-kind abstraction, a change to the existing
  unary helper surface, integral classification, another production type, another module, or a
  nineteenth path, stop and report the conflict.<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary); [module glossary](../../../../glossary.md#term-module))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.operation.elementwise.unary` — owns parameterless unary numeric
  transforms and activations whose public construction retains the floating input type.<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary); [activations glossary](../../../../glossary.md#term-activation))</span>
- `io.github.pho001.synaptik.model.tensor` — owns public methods and the two cohesive package-private
  construction boundaries.
- `io.github.pho001.synaptik.model.operation` — owns unchanged [operation](../../../../glossary.md#operation)/[signature](../../../../glossary.md#term-signature) foundations.

Package added:

```text
io.github.pho001.synaptik.model.operation.elementwise.classification
  Parameterless floating value-classification semantics with fixed BOOL results.
```

Type placement:

- `UnaryElementwiseKind` remains in the unary package because the three new numerical functions
  share its exact input/result construction contract.
- `FloatingClassificationKind` belongs in the new classification package because it names
  graph-visible floating value classifications, not tracing diagnostics.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>
- `TensorFloatingClassifications` remains package-private beside [`Tensor`](../../../../glossary.md#tensor) because it owns local
  input validation, descriptor derivation, and derived construction for the public facade.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [descriptor glossary](../../../../glossary.md#term-descriptor); [facade glossary](../../../../glossary.md#term-facade))</span>
- Focused tests mirror production packages; Tensor construction tests remain in `model.tensor` for
  package-private helper access.

## Required implementation contracts

### [Unary](../../../../glossary.md#term-unary)-family extension

- `UnaryElementwiseKind.values()` equals the exact sixteen-value sequence in this specification.
- Every value returns the existing [immutable](../../../../glossary.md#term-immutable) [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs)/one-input/one-output [signature](../../../../glossary.md#term-signature).
- `TensorUnaryExpressions` remains final, package-private, field-free, non-instantiable, and has
  exactly its existing one package-private `apply(Tensor, UnaryElementwiseKind)` method.
- Its null/type validation, descriptor construction, operation construction, factory delegation,
  exception messages, and side-effect order remain unchanged.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [descriptor glossary](../../../../glossary.md#term-descriptor); [operation glossary](../../../../glossary.md#operation))</span>
- The three new [Tensor](../../../../glossary.md#tensor) methods delegate exactly once with the exact new kind.

### Classification-family contract

Create `FloatingClassificationKind` as one public enum with the exact three [constants](../../../../glossary.md#term-constant) and one
private static final [immutable](../../../../glossary.md#term-immutable) [signature](../../../../glossary.md#term-signature) list. Its only project method is:

```java
@Override
public List<OperationSignature> signatures()
```

The returned list contains exactly
`OperationSignature.fixed(NoOperationAttrs.class, 1, 1)`. Standard [compiler](../../../../glossary.md#term-compiler)-generated enum state
does not count as project state.

Create `TensorFloatingClassifications` as one package-private final non-record class with:

- no fields, nested types, interfaces, or instance state;
- exactly one private zero-[argument](../../../../glossary.md#term-argument) constructor; and
- exactly one package-private static method:

```java
static Tensor apply(Tensor input, FloatingClassificationKind kind)
```

`apply` performs this exact order:

1. `Objects.requireNonNull(input, "input")`;
2. `Objects.requireNonNull(kind, "kind")`;
3. read `input.descriptor().dataType()`;
4. require `dataType.isFloating()`, otherwise throw `IllegalArgumentException` with exact message
   `input must be a floating data type, but was <type>`;
5. construct one `TensorDescriptor` with `DataType.BOOL`, the exact input [Shape](../../../../glossary.md#shape) reference,
   `Optional.empty()`, and `false`;
6. construct one `Operation(kind, NoOperationAttrs.INSTANCE)`;
7. call `TensorFactory.createDerived` exactly once with the descriptor, no label, operation, and
   ordered <span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [operation glossary](../../../../glossary.md#operation))</span>`List.of(input)`.

A pre-factory failure allocates no [Tensor](../../../../glossary.md#tensor) [identity](../../../../glossary.md#term-identity-meanings). A valid call returns the factory's exact fresh,
unlabeled, storage-free result and does not mutate or retain input storage as output storage.

### Public [facade](../../../../glossary.md#term-facade) and exact surface

- `rsqrt`, `log1p`, and `expm1` each delegate exactly once to `TensorUnaryExpressions.apply`.
- `isFinite`, `isNaN`, and `isInf` each delegate exactly once to
  `TensorFloatingClassifications.apply`.
- All six methods are public, non-static, non-synchronized, zero-[argument](../../../../glossary.md#term-argument), and return [`Tensor`](../../../../glossary.md#tensor).
- The final declared public Tensor method count is exactly 127.
- Reflection finds all six new names and no alternate spelling listed in Out of [scope](../../../../glossary.md#term-scope).
- The existing 121-method surface, including `reciprocal`, ordinary [`log`](../../../../glossary.md#term-log)/[`exp`](../../../../glossary.md#term-expx)/`sqrt`, pairwise
  `minimum`/`maximum`, reduction `min`/`max`, scalar operations, and every non-numeric method,
  remains unchanged.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>

## Affected files

Expected production Java:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/unary/UnaryElementwiseKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/classification/FloatingClassificationKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorUnaryExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorFloatingClassifications.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Expected tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/OperationSignatureTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/elementwise/unary/UnaryElementwiseKindTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/elementwise/classification/FloatingClassificationKindTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorUnaryElementwiseTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFloatingClassificationTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`

Expected documentation and [planning](../../../../glossary.md#term-planning):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- `docs/planning/modules/model/tasks/0018t1-unary-numeric-gaps-and-floating-diagnostics.md`
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review-only unless an out-of-[scope](../../../../glossary.md#term-scope) discrepancy requires stopping:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/Operation.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/OperationKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/OperationSignature.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/NoOperationAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/datatype/DataType.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorDescriptor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorProducer.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorProvenance.java`
- `docs/api/training-api.md`
- focused architecture documentation and architecture tests

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify exactly the 18 affected paths listed above: five production Java
files, six test files, and seven documentation/[planning](../../../../glossary.md#term-planning) files including this specification.

If implementation needs a nineteenth path, an attributes type, another helper method, integral
classification, a shared result-kind abstraction, or a review-only file, stop and report the
reason before editing it. Do not trade away a required focused test or documentation path merely
to stay under the cap.

## Acceptance criteria

- The unary enum contains exactly sixteen selected kinds with three additions in the specified
  positions and unchanged family signature.<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary); [signature glossary](../../../../glossary.md#term-signature))</span>
- The classification enum contains exactly three selected kinds and the exact parameterless
  one-input/one-output signature.
- All six public [Tensor](../../../../glossary.md#tensor) methods exist with the exact names and no [aliases](../../../../glossary.md#aliasing-and-arena) or overloads.
- Numeric transforms accept every current floating type, retain the exact input type/[Shape](../../../../glossary.md#shape)/
  `requiresGrad`, and leave [layout](../../../../glossary.md#layout) unresolved.
- Classifications accept every current floating type, return BOOL with the exact input Shape,
  false gradient eligibility, and unresolved layout.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- INT32, INT64, and BOOL fail both construction paths before result identity allocation with the
  exact existing floating-input message.<span data-glossary-navigation="true"> ([INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Every successful call is fresh, unlabeled, storage-free, one-output, and records exact operation,
  canonical no-attributes, exact sole input, and provenance output index zero.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [canonical glossary](../../../../glossary.md#term-canonical-form); [provenance glossary](../../../../glossary.md#provenance))</span>
- Scalar, zero-sized, static, dynamic, resolved-layout, storage-backed, and already-derived inputs
  preserve the specified metadata and ownership boundaries.<span data-glossary-navigation="true"> ([Scalar glossary](../../../../glossary.md#term-scalar); [metadata glossary](../../../../glossary.md#term-metadata); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Tests prove no value or domain inspection by constructing metadata from representative negative,
  zero, infinity, and NaN host-backed leaves without evaluating them.<span data-glossary-navigation="true"> ([infinity glossary](../../../../glossary.md#term-infinity); [NaN glossary](../../../../glossary.md#term-nan); [host glossary](../../../../glossary.md#term-host))</span>
- Javadocs and current API documentation explain the selected special-value semantics, result-type
  distinction, accuracy boundary, and newcomer-readable examples without claiming execution.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Operation signature coverage includes both families and rejects incompatible attributes through
  the existing foundation.<span data-glossary-navigation="true"> ([Operation signature glossary](../../../../glossary.md#operation-signature))</span>
- The exact public Tensor method count is 127 and helper/enum reflection shapes match the plan.
- Tensor API, Compile API, glossary, capability baseline, task, master plan, and roadmap are
  synchronized without rewriting completed task history.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- Training API, architecture contract/docs/tests, dependencies, Gradle, other modules, backend
  conformance, and integration tests remain unchanged with reasoned no-change conclusions.<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- One final `:modules:model:test` [run](../../../../glossary.md#term-run) passes after executable Java stabilizes.
- A separate clean-context documentation-focused pass finalizes all affected Javadocs and
  explanatory/planning documentation in the same overall change and reuses successful Java-test
  evidence unless executable Java changes afterward or a concrete risk is recorded.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Model Javadoc, a runnable Java 26 metadata example, Markdown links/anchors/fences/final-newlines,
  exact 18-path scope, status synchronization, terminology, and <span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope); [synchronization glossary](../../../../glossary.md#term-synchronization))</span>`git diff --check` pass.
- Task 0018T1 is Complete only after both passes succeed. Task 0018U and every later task remain
  Draft without a detailed specification.

## Tests / [validation](../../../../glossary.md#term-validation)

Focused tests while executable Java stabilizes:

```bash
./gradlew :modules:model:test --tests \
  io.github.pho001.synaptik.model.operation.OperationSignatureTest \
  --tests io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKindTest \
  --tests io.github.pho001.synaptik.model.operation.elementwise.classification.FloatingClassificationKindTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorUnaryElementwiseTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorFloatingClassificationTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorTest
```

Final Java [checkpoint](../../../../glossary.md#term-checkpoint) after executable code stabilizes:

```bash
./gradlew :modules:model:test
```

The implementation pass records exact results and hands them to the documentation pass. Do not
repeat the successful final [model](../../../../glossary.md#term-model) suite unless executable Java changes afterward or a concrete
risk is recorded.

Documentation-focused pass:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

The documentation pass must also:

- compile and run one Java 26 example that observes one numeric transform and one classification
  result descriptor, exact operation kinds, one-input producers, and output index zero without
  claiming calculated values;<span data-glossary-navigation="true"> ([producers glossary](../../../../glossary.md#term-producer))</span><span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [run glossary](../../../../glossary.md#term-run); [descriptor glossary](../../../../glossary.md#term-descriptor); [operation glossary](../../../../glossary.md#operation))</span>
- validate local Markdown links and anchors in changed documents;
- inspect generated [Javadoc](../../../../glossary.md#term-javadoc) for all affected/new public contracts and [Tensor](../../../../glossary.md#tensor) methods;
- verify exact enum values, six new public methods, final public count 127, helper shape, and
  absence of aliases or restored fast variants;<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [aliases glossary](../../../../glossary.md#aliasing-and-arena))</span>
- verify the exact 18-path inventory and no [dependency](../../../../glossary.md#term-dependency)/[Gradle](../../../../glossary.md#term-gradle)/architecture/other-[module](../../../../glossary.md#term-module) changes;
- verify synchronized 0018T1 Complete status and no detailed 0018U-or-later specification; and
- review terminology, examples, fences, final newlines, trailing whitespace, authority boundaries,
  and the final diff.

Repository-wide validation is deferred to the capability-reset checkpoint after task 0018V and
[CI](../../../../glossary.md#term-continuous-integration-ci). This task changes one module without changing dependencies, architecture, or shared build
configuration.

## Dependencies

- [Task 0014C](0014c-unary-elementwise-semantic-kinds.md) — current parameterless [unary](../../../../glossary.md#term-unary) family.
- [Task 0014D](0014d-unary-elementwise-tensor-expressions.md) — current floating-preserving unary
  construction path.
- [Task 0018K](0018k-operation-signature-and-construction-hardening.md) — family-owned exact
  signature validation.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [validation glossary](../../../../glossary.md#term-validation))</span>
- [Task 0018P](0018p-elementwise-semantic-cleanup.md) — final portable unary vocabulary and no-fast-
  variant boundary.
- [Task 0018T](0018t-scalar-arithmetic-family-normalization.md) — completed floating arithmetic
  family and numerical-policy conventions.

All dependencies are Complete.

## Follow-up tasks

- **0018U Integral arithmetic and comparison domains** — add only selected signed-integral
  arithmetic, comparisons, arg-min, and reductions after overflow and accumulation policies are
  explicit.<span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow))</span>
- **0018V Multi-[axis](../../../../glossary.md#term-axis) and statistical reductions** — add ordered axes, log-sum-exp, variance,
  standard deviation, and norms after floating/integral foundations are stable.<span data-glossary-navigation="true"> ([log-sum-exp glossary](../../../../glossary.md#log-sum-exp); [variance glossary](../../../../glossary.md#term-variance); [standard deviation glossary](../../../../glossary.md#term-standard-deviation))</span>
- Later compiler and backend tasks own gradients, numerical conformance, lowering, execution, and
  kernels.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [gradients glossary](../../../../glossary.md#term-gradient); [conformance glossary](../../../../glossary.md#term-conformance); [lowering glossary](../../../../glossary.md#lowering); [kernels glossary](../../../../glossary.md#kernel))</span>

## Architecture impact

Expected impact: None.

This adds [backend](../../../../glossary.md#backend)-independent [model](../../../../glossary.md#term-model) semantics, public [metadata](../../../../glossary.md#term-metadata) construction, and one cohesive
[operation](../../../../glossary.md#operation) subpackage within existing [module](../../../../glossary.md#term-module) [ownership](../../../../glossary.md#term-ownership-meanings). It changes no module [dependency](../../../../glossary.md#term-dependency), [lifecycle](../../../../glossary.md#lifecycle),
architecture rule, backend contract, [runtime](../../../../glossary.md#term-runtime) path, or [graph](../../../../glossary.md#graph) representation.

If implementation requires an architecture change, stop and report it before editing
`ARCHITECTURE.md`, focused architecture documentation, ADRs, or architecture tests.

## Implementation prompt

Use this prompt in a separate agentic task/thread with clean context:

```text
You are working in the Synaptik repository.

Read AGENTS.md, ARCHITECTURE.md, documentation/planning rules, roadmap, model capabilities/master
plan, tasks 0014C/0014D/0018K/0018P/0018T/0018T1, Tensor API, Compile API, Training API,
glossary, and every affected or review-only source/test named by task 0018T1 in full.

Implement task 0018T1 exactly. Extend the existing floating-preserving unary family with RSQRT,
LOG1P, and EXPM1, and add the separate parameterless FloatingClassificationKind family plus
public isFinite/isNaN/isInf construction with fixed non-differentiable BOOL results. Preserve
validation order, descriptor and producer/provenance behavior, existing unary contracts, and every
architecture boundary. Stay within the exact 18 paths, stop on scope or architecture conflict,
and do not commit or push.

Run focused tests as needed and one final model test after executable Java stabilizes. Then hand
the actual diff and exact evidence to a separate clean-context documentation-focused agent in the
same overall change. That agent must inspect final source/tests, finalize permitted Javadocs,
Tensor/Compile API, glossary, capability/task/master/roadmap documentation, run model Javadoc and
documentation/scope checks, and must not repeat successful Java tests unless executable behavior
changes or it records a concrete reason.

Mark 0018T1 Complete only after both passes succeed. Leave 0018U and every later task Draft
without a detailed specification.
```

## Local decisions

- Kept `RSQRT`, `LOG1P`, and `EXPM1` in the existing unary family because their public result
  metadata preserves floating input type and gradient eligibility.<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary); [metadata glossary](../../../../glossary.md#term-metadata); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Kept floating classifications in a separate kind family and package-private construction helper
  because their public result metadata is fixed non-differentiable BOOL. No result-kind flag,
  shared category abstraction, registry, or switch was added.<span data-glossary-navigation="true"> ([differentiable glossary](../../../../glossary.md#derivative-and-chain-rule); [BOOL glossary](../../../../glossary.md#data-type--datatype); [registry glossary](../../../../glossary.md#term-registry))</span>
- Preserved each transform as one first-class operation occurrence. No eager constant,
  decomposition, alias, fast variant, accuracy mode, or backend hint was introduced.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [occurrence glossary](../../../../glossary.md#term-occurrence); [eager glossary](../../../../glossary.md#term-eager-evaluation); [constant glossary](../../../../glossary.md#term-constant); [decomposition glossary](../../../../glossary.md#term-decomposition); [alias glossary](../../../../glossary.md#aliasing-and-arena); [backend glossary](../../../../glossary.md#backend))</span>

## Known limitations

- This task constructs model descriptors and provenance only. It does not evaluate numeric or
  classification values, capture a graph, define gradients, lower operations, select kernels, or
  provide execution support.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [descriptors glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [capture glossary](../../../../glossary.md#term-capture-meanings); [graph glossary](../../../../glossary.md#graph); [gradients glossary](../../../../glossary.md#term-gradient); [kernels glossary](../../../../glossary.md#kernel))</span>
- The numeric transforms select portable special-value semantics but no exact rounding, NaN
  payload, bitwise result, fixed ULP/relative-error bound, algorithm, or backend tolerance.<span data-glossary-navigation="true"> ([rounding glossary](../../../../glossary.md#term-rounding); [NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [ULP glossary](../../../../glossary.md#term-unit-in-the-last-place-ulp); [backend glossary](../../../../glossary.md#backend); [tolerance glossary](../../../../glossary.md#term-tolerance))</span>
- Both families remain floating-input-only. Integral arithmetic and comparison domains remain
  deferred to Draft task 0018U.

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation context `/root/task_0018t1_implementation` ran the exact focused command listed
  above. Its initial run exposed only the expected stale <span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>[`Tensor`](../../../../glossary.md#tensor) public-method inventory; after
  correcting that inventory, the exact command passed.
- The same implementation context ran `./gradlew :modules:model:test` after executable Java
  stabilized. It passed with `BUILD SUCCESSFUL in 1s`; three tasks were actionable, one executed,
  two were up-to-date, and the configuration cache was reused. Executable Java did not change
  afterward. Documentation context<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache))</span>
  `/root/task_0018t1_implementation/task_0018t1_docs` reused this evidence and did not rerun the
  successful Java suite.
- The clean documentation context applied the General, API/Javadoc, Planning, and Example
  profiles to the final source/tests and the seven authorized documentation/planning files. It
  independently finalized all five affected production Javadocs plus Tensor API, Compile API,
  glossary, capabilities, task, master plan, and roadmap.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Planning glossary](../../../../glossary.md#term-planning); [Compile glossary](../../../../glossary.md#compile))</span>
- `./gradlew :modules:model:javadoc` passed after final [Javadoc](../../../../glossary.md#term-javadoc) edits with `BUILD SUCCESSFUL in
  1s`; both actionable tasks executed and the configuration cache was reused. Generated pages
  contain all three new unary constants, all three classification constants, all six Tensor
  methods, the exact special-value/result-type boundaries, and valid cross-links.<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary); [constants glossary](../../../../glossary.md#term-constant))</span>
- `javac -cp modules/model/build/classes/java/main -d
  /tmp/synaptik-floating-metadata-example /tmp/FloatingMetadataExample.java && java -cp
  modules/model/build/classes/java/main:/tmp/synaptik-floating-metadata-example
  FloatingMetadataExample` compiled and ran the documented Java 26 example. It printed
  `transformType=FLOAT32`, `transformGrad=true`, `transformKind=LOG1P`,
  `classificationType=BOOL`, `classificationGrad=false`, `classificationKind=IS_NAN`, exact
  Shape and input retention, canonical parameterless attributes, one input for each producer, and
  output indexes <span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [canonical glossary](../../../../glossary.md#term-canonical-form); [producer glossary](../../../../glossary.md#term-producer))</span>`0,0`. It made no evaluated-value claim.
- A compiled Java 26 `FloatingSurfaceCheck` passed. It verified the exact sixteen unary values,
  exact three classification values, all six zero-argument Tensor methods, exactly 127 declared
  public Tensor methods, both field-free/final/package-private one-method helper shapes, and the
  absence of aliases, old <span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument); [aliases glossary](../../../../glossary.md#aliasing-and-arena))</span>`inv`, and fast variants.
- The targeted local Markdown checker resolved 493 local links, including 139 heading anchors,
  across the seven changed documentation/planning files with zero failures. Formatting checks
  found balanced fences, final newlines, and no trailing whitespace.
- Final inventory inspection found exactly 18 authorized paths: five production Java files, six
  tests, and seven documentation/planning files. The task, model master plan, and roadmap all mark
  0018T1 Complete; 0018U and every later task remain Draft, and no detailed 0018U-or-later task
  specification exists. Final <span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>`git diff --check` passed with no output.
- Training API remains accurate unchanged because this task adds no gradient object, autograd,
  optimizer, session, or training behavior. The Compile API was updated only to distinguish
  current model metadata construction from planned capture, descriptor revalidation, autograd,
  lowering, and execution.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [autograd glossary](../../../../glossary.md#autograd); [optimizer glossary](../../../../glossary.md#term-optimizer); [metadata glossary](../../../../glossary.md#term-metadata); [capture glossary](../../../../glossary.md#term-capture-meanings); [descriptor glossary](../../../../glossary.md#term-descriptor); [lowering glossary](../../../../glossary.md#lowering))</span>
- [`Operation`](../../../../glossary.md#operation), [`OperationKind`](../../../../glossary.md#operationkind), `OperationSignature`, [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs), [`DataType`](../../../../glossary.md#data-type--datatype),
  `TensorDescriptor`, [`TensorProducer`](../../../../glossary.md#producer--tensorproducer), and `TensorProvenance` remain accurate unchanged: the new
  kinds use the existing exact no-attributes/one-input/one-output structure, public construction
  owns floating eligibility and result derivation, and provenance remains output index zero over
  one exact input.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>
- `ARCHITECTURE.md`, focused architecture documents/tests, dependencies, Gradle and Java version,
  backend-conformance/integration tests, other modules, and review-only foundations remain
  unchanged because this is one model-owned semantic/metadata capability with no dependency,
  lifecycle, graph representation, compiler, backend, runtime, build, or executable-behavior
  change. Repository-wide validation remains deferred to the recorded capability-reset checkpoint
  after 0018V and CI.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [modules glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [lifecycle glossary](../../../../glossary.md#lifecycle); [graph glossary](../../../../glossary.md#graph); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [CI glossary](../../../../glossary.md#term-continuous-integration-ci))</span>

## Implementation notes

- Extended `UnaryElementwiseKind` and the unchanged [unary](../../../../glossary.md#term-unary) helper path with exact `LOG1P`, `EXPM1`,
  and `RSQRT` placement and public delegations.
- Added exact `FloatingClassificationKind` semantics and one cohesive package-private helper for
  fixed BOOL, unresolved-layout, false-gradient result construction.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [layout glossary](../../../../glossary.md#layout); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Added focused signature, vocabulary, metadata, provenance, special-value non-inspection,
  freshness, validation-order, helper-shape, and exact public-surface coverage.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance); [validation glossary](../../../../glossary.md#term-validation); [shape glossary](../../../../glossary.md#shape))</span>
- Finalized current API, glossary terminology, capability status, and planning evidence without
  changing completed predecessor history or creating the next detailed task.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>

## Completion summary

- Completed changes: Added and documented first-class `RSQRT`, `LOG1P`, and `EXPM1` numeric
  transforms plus separately typed `IS_FINITE`, `IS_NAN`, and `IS_INF` floating classifications
  and their six public Tensor construction methods.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- Files changed or created: Exactly five production Java files, six tests, and seven
  documentation/planning files in the authorized 18-path scope.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [scope glossary](../../../../glossary.md#term-scope))</span>
- Tests and validation: Reused the passing exact focused command and final model suite; final model
  Javadoc, compiled metadata example, generated-page inspection, exact enum/method/helper/alias
  surface check, 493-link/139-anchor check, formatting, exact scope/status/no-later-spec checks,
  and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc); [metadata glossary](../../../../glossary.md#term-metadata); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>`git diff --check` passed.
- Documentation-agent review: Clean context
  `/root/task_0018t1_implementation/task_0018t1_docs` completed the independent pass using the
  General, API/Javadoc, Planning, and Example profiles.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Documentation impact: Tensor and Compile APIs, glossary, capability baseline, task, master plan,
  and roadmap now describe the implemented surface and preserve compiler/backend/execution
  boundaries.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend))</span>
- Javadoc review: All five affected production paths were independently reviewed and finalized;
  review-only operation, descriptor, data-type, producer, and provenance contracts remain accurate
  unchanged for the reasons recorded above.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [descriptor glossary](../../../../glossary.md#term-descriptor); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>
- Glossary impact: Added reusable floating-classification terminology and updated current unary,
  operation-family, provenance, and Tensor distinctions.<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary))</span>
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0018T1. Task 0018U and later work remain Draft without detailed
  specifications.

Status: Complete
