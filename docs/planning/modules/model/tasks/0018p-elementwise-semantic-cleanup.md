# Task 0018P: [Elementwise](../../../../glossary.md#term-elementwise) Semantic Cleanup

## Status

Complete

## Goal

Finalize the parameterless [unary](../../../../glossary.md#term-unary) [elementwise](../../../../glossary.md#term-elementwise) vocabulary before the public [API](../../../../glossary.md#term-application-programming-interface-api) stabilizes:

```text
one portable mathematical request -> one semantic kind -> one public Tensor method
backend prepare                  -> implementation-route selection
```

Atomically rename the reciprocal request from `INV`/`inv()` to
`RECIPROCAL`/`reciprocal()`, and remove `FAST_EXP`/`fastExp()` and
`FAST_TANH`/`fastTanh()` completely. No old and new spelling may coexist in the live [model](../../../../glossary.md#term-model) API.

Retain [`EXP`](../../../../glossary.md#term-expx)/`exp()` and [`TANH`](../../../../glossary.md#term-tanh)/`tanh()` as portable mathematical requests. They identify the
[natural exponential](../../../../glossary.md#norms-elementary-functions-and-probability-distributions) and [hyperbolic tangent](../../../../glossary.md#bias-and-activation), respectively, without selecting an algorithm,
promising a bitwise result or approximation bound, or naming a [backend route](../../../../glossary.md#backend-route). [Concrete backend](../../../../glossary.md#concrete-backend)
[prepare](../../../../glossary.md#prepare) later owns implementation-route selection within the eventual numerical contract; this
task adds no prepare or [backend](../../../../glossary.md#backend) behavior.

The cleanup must preserve the completed unary construction contract and the completed typed
[scalar](../../../../glossary.md#term-scalar)/clamp contract. It changes vocabulary, public surface, tests, Javadocs, and explanatory
documentation only; it does not calculate values or add the numeric gaps assigned to task 0018T.

## [Mental model](../../../../glossary.md#term-mental-model)

The final [unary](../../../../glossary.md#term-unary) family contains one kind for each public request:

```text
Tensor.reciprocal() -> UnaryElementwiseKind.RECIPROCAL
Tensor.exp()        -> UnaryElementwiseKind.EXP
Tensor.tanh()       -> UnaryElementwiseKind.TANH
```

Every successful call records the chosen kind with `NoOperationAttrs.INSTANCE`, one exact input,
one exact output [descriptor](../../../../glossary.md#term-descriptor), and [provenance](../../../../glossary.md#provenance) output index zero. The [model](../../../../glossary.md#term-model) records meaning. It does
not record whether a [backend](../../../../glossary.md#backend) later uses a library routine, hardware instruction, polynomial,
table, vectorized [kernel](../../../../glossary.md#kernel), or another implementation route.

The [scalar](../../../../glossary.md#term-scalar) family remains a separate parameterized family:

```text
ScalarElementwiseKind + ScalarValueAttrs/ClampRangeAttrs
```

This task neither adds scalar add/subtract/divide/minimum/maximum nor moves reciprocal into the
scalar family. Those additions and the separate `rsqrt`, `log1p`, `expm1`, and floating diagnostic
requests remain task 0018T work.

## Current problems

- `INV` and `inv()` abbreviate a mathematical reciprocal and conflict with the selected explicit
  public name.
- `FAST_EXP` and `FAST_TANH` expose implementation intent without a portable accuracy or
  special-value contract.
- Existing Javadocs call [`EXP`](../../../../glossary.md#term-expx) and [`TANH`](../../../../glossary.md#term-tanh) “strict” requests only to distinguish them from the fast
  variants. After those variants are removed, that wording can be misread as a bitwise or
  algorithm guarantee.
- The current [unary](../../../../glossary.md#term-unary) [API](../../../../glossary.md#term-application-programming-interface-api) reference uses `fastExp()` as its complete example and therefore teaches
  a request that is leaving the selected baseline.
- Reflection inventories, exact enum-order tests, operation-composition tests, and public method
  counts encode the provisional names and must change in the same migration.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- The completed scalar family is adjacent to this cleanup, but adding its missing operations here
  would silently broaden the task and overlap task 0018T.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>

## Before / after [API](../../../../glossary.md#term-application-programming-interface-api)

| Current live contract | Final contract | Disposition |
|---|---|---|
| `UnaryElementwiseKind.INV` | `UnaryElementwiseKind.RECIPROCAL` | [Atomic](../../../../glossary.md#resource-transaction-and-rollback) rename; no [alias](../../../../glossary.md#aliasing-and-arena). |
| `Tensor.inv()` | `Tensor.reciprocal()` | Atomic rename; no deprecated bridge. |
| `UnaryElementwiseKind.FAST_EXP` | none | Delete completely. |
| `Tensor.fastExp()` | none | Delete completely. |
| `UnaryElementwiseKind.FAST_TANH` | none | Delete completely. |
| `Tensor.fastTanh()` | none | Delete completely. |
| `UnaryElementwiseKind.EXP` / `Tensor.exp()` | unchanged names | Retain as the portable natural-exponential request. |
| `UnaryElementwiseKind.TANH` / `Tensor.tanh()` | unchanged names | Retain as the portable hyperbolic-[tangent](../../../../glossary.md#term-tangent) request. |
| five `ScalarElementwiseKind` values and existing public [scalar](../../../../glossary.md#term-scalar) methods | unchanged | Preserve exactly; task 0018T owns missing scalar operations. |

Completed tasks 0014C and 0014D remain historical records of the provisional surface. They are
not rewritten to pretend they originally implemented the final vocabulary.

## [Scope](../../../../glossary.md#term-scope)

- Replace `UnaryElementwiseKind.INV` with `RECIPROCAL` in the same declaration position.
- Remove `FAST_EXP` and `FAST_TANH` from `UnaryElementwiseKind` without replacement.
- Make the final enum contain exactly thirteen [constants](../../../../glossary.md#term-constant) in the order specified below.
- Replace public `Tensor.inv()` with `Tensor.reciprocal()` and remove `fastExp()` and `fastTanh()`.
- Make the final public unary method inventory contain exactly thirteen zero-argument methods in
  one-to-one correspondence with the final enum.<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary); [argument glossary](../../../../glossary.md#term-argument))</span>
- Preserve the existing parameterless one-input, one-output [operation signature](../../../../glossary.md#operation-signature).
- Preserve floating-only unary eligibility, exact result metadata, construction order,
  one-input producer/provenance, freshness, and absence of value inspection.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>
- Remove “strict versus fast” terminology from current Javadocs and [API](../../../../glossary.md#term-application-programming-interface-api) documentation.
- Document [`EXP`](../../../../glossary.md#term-expx) and [`TANH`](../../../../glossary.md#term-tanh) as portable mathematical semantic requests with no algorithm,
  bitwise-result, approximation-bound, or backend-route promise.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- Update exact enum, method, reflection, public-API, absence, operation-signature, provenance, and
  chaining tests.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [signature glossary](../../../../glossary.md#term-signature))</span>
- Replace the [Tensor](../../../../glossary.md#tensor) API's current complete `fastExp()` example with a complete
  `reciprocal()`/`RECIPROCAL` [metadata](../../../../glossary.md#term-metadata) example.
- Update the Compile API's current unary method count and terminology without adding compiler
  behavior.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- Update glossary, capability baseline, task evidence, model master plan, and roadmap through the
  required independent documentation-focused pass.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>
- Review and explicitly preserve the completed `ScalarElementwiseKind`, `ScalarValueAttrs`, and
  `ClampRangeAttrs` contracts from task 0018N.

## Out of [scope](../../../../glossary.md#term-scope)

- aliases, deprecated methods, forwarding bridges, duplicate enum constants, compatibility
  adapters, parsers, migration shims, or serialized-name translation for removed spellings<span data-glossary-navigation="true"> ([aliases glossary](../../../../glossary.md#aliasing-and-arena); [constants glossary](../../../../glossary.md#term-constant))</span>
- `rsqrt`, `log1p`, `expm1`, `isFinite`, `isNaN`, or `isInf`; task 0018T owns them
- scalar add, scalar subtract, scalar divide, scalar minimum, or scalar maximum; task 0018T owns
  their semantics and public methods<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- changes to `ScalarElementwiseKind`, `ScalarValueAttrs`, `ClampRangeAttrs`, [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue),
  `TensorScalarExpressions`, or their public [Tensor](../../../../glossary.md#tensor) overloads
- integral or BOOL unary eligibility, implicit conversion, cast insertion, result-type changes, or
  additional overloads<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [unary glossary](../../../../glossary.md#term-unary))</span>
- numerical evaluation, domain checks, zero handling, overflow, underflow, NaN, infinity,
  signed-zero policy, rounding, accuracy, determinism, or special-value promises<span data-glossary-navigation="true"> ([evaluation glossary](../../../../glossary.md#term-evaluation); [overflow glossary](../../../../glossary.md#term-overflow); [underflow glossary](../../../../glossary.md#term-underflow); [NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [rounding glossary](../../../../glossary.md#term-rounding))</span>
- gradient formulas, subgradient conventions, autograd traversal, backward graph construction,
  optimizer, or training behavior<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [autograd glossary](../../../../glossary.md#autograd); [backward graph glossary](../../../../glossary.md#backward-graph); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- compiler capture, canonicalization, common-subexpression elimination, constant folding, or
  graph-wide validation<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [common-subexpression elimination glossary](../../../../glossary.md#term-common-subexpression-elimination-cse); [constant folding glossary](../../../../glossary.md#term-constant-folding); [graph glossary](../../../../glossary.md#graph); [validation glossary](../../../../glossary.md#term-validation))</span>
- planning ownership, capability-provider implementation, prepare lowering, backend routing,
  kernels, fusion, runtime, execution, tracing, or engine behavior<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [prepare glossary](../../../../glossary.md#prepare); [lowering glossary](../../../../glossary.md#lowering); [backend glossary](../../../../glossary.md#backend); [kernels glossary](../../../../glossary.md#kernel); [fusion glossary](../../../../glossary.md#term-fusion); [runtime glossary](../../../../glossary.md#term-runtime); [engine glossary](../../../../glossary.md#term-engine))</span>
- changes to `OperationSignature`, [`OperationKind`](../../../../glossary.md#operationkind), [`Operation`](../../../../glossary.md#operation), [`TensorProducer`](../../../../glossary.md#producer--tensorproducer),
  `TensorProvenance`, `TensorFactory`, [descriptors](../../../../glossary.md#term-descriptor), [Shapes](../../../../glossary.md#shape), [layouts](../../../../glossary.md#layout), storage, or identifiers
- dependencies, Gradle, Java version, preview/incubator configuration, architecture rules,
  focused architecture documents, architecture tests, another module, or a detailed 0018Q-or-
  later task specification<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>, especially model ownership of public
  Tensor and operation semantics, backend-prepare ownership of lowering/kernel selection, and the
  prohibition on backend support in <span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Tensor glossary](../../../../glossary.md#tensor); [backend glossary](../../../../glossary.md#backend); [prepare glossary](../../../../glossary.md#prepare); [lowering glossary](../../../../glossary.md#lowering); [kernel glossary](../../../../glossary.md#kernel))</span>[`Operation`](../../../../glossary.md#operation)
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md), especially the existing capability decisions
  for `fastExp`, `fastTanh`, and `inv`
- [Model master plan](../master-plan.md)
- [Task 0014C](0014c-unary-elementwise-semantic-kinds.md)
- [Task 0014D](0014d-unary-elementwise-tensor-expressions.md)
- [Task 0014E](0014e-scalar-arithmetic-and-clamp-semantics.md)
- [Task 0014F](0014f-scalar-arithmetic-and-clamp-tensor-expressions.md)
- [Task 0018K](0018k-operation-signature-and-construction-hardening.md)
- [Task 0018N](0018n-typed-scalar-value-contract.md)
- [Task 0018O](0018o-indexing-taxonomy-and-unstack-normalization.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `modules/model` owns the final backend-independent unary kind names and public Tensor expression
  construction.<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary))</span>
- [`Operation`](../../../../glossary.md#operation) records semantic meaning, never [backend](../../../../glossary.md#backend) support or an implementation route.
- [`EXP`](../../../../glossary.md#term-expx) and [`TANH`](../../../../glossary.md#term-tanh) remain mathematical meanings. Backend prepare may later choose a concrete route
  only within the numerical contract established by an owning execution/conformance task.<span data-glossary-navigation="true"> ([prepare glossary](../../../../glossary.md#prepare); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- No model field, enum flag, attribute, tag, cost, accuracy class, “fast” bit, backend hint, or
  route selector may replace the removed fast kinds.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>
- [`Tensor`](../../../../glossary.md#tensor) remains [mutable](../../../../glossary.md#term-mutable) public [API](../../../../glossary.md#term-application-programming-interface-api) state rather than [graph](../../../../glossary.md#graph) [intermediate representation](../../../../glossary.md#term-intermediate-representation-ir).
- Unary construction remains metadata-only and must not inspect host storage, values, provenance
  ancestors, or backend capabilities.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [host storage glossary](../../../../glossary.md#host-storage); [provenance glossary](../../../../glossary.md#provenance))</span>
- The parameterless [signature](../../../../glossary.md#term-signature) remains exact [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs), one input, and one output. Kind
  vocabulary cleanup must not change signature mechanics.
- Every result remains one output of one [identity](../../../../glossary.md#term-identity-meanings)-distinct [`TensorProducer`](../../../../glossary.md#producer--tensorproducer); its
  `TensorProvenance.outputIndex()` is zero.
- Package direction remains `model.tensor -> model.operation.elementwise.unary ->
  model.operation`. No package or [module](../../../../glossary.md#term-module) [dependency](../../../../glossary.md#term-dependency) changes.
- Completed scalar semantics remain exact typed attributes. They are not folded into the unary
  family and do not expand in this task.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- Stop if implementation reveals another semantic ambiguity, requires numerical policy, or needs
  a production concept or path outside the bounded scope. Do not silently broaden the cleanup.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.operation.elementwise.unary` — owns the final parameterless
  unary semantic enum.<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary))</span>
- `io.github.pho001.synaptik.model.tensor` — owns the public fluent unary methods, package-private
  construction helper, producer/provenance attachment, and exact public API inventory tests.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- `io.github.pho001.synaptik.model.operation.elementwise.scalar` — review-only adjacent family;
  its vocabulary and attributes remain unchanged.
- `io.github.pho001.synaptik.model.operation` — review-only signature and parameterless-attributes
  foundation.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature))</span>

Packages added or moved:

- None.

Type placement:

- `io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind` remains the
  public unary semantic family; only its constants and Javadocs change.<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant))</span>
- `io.github.pho001.synaptik.model.tensor.Tensor` remains the public fluent surface; only the
  specified unary declarations, type-level wording, and Javadocs change.
- `io.github.pho001.synaptik.model.tensor.TensorUnaryExpressions` remains the package-private,
  field-free common construction boundary; executable behavior remains unchanged.

No package-map or visibility change is permitted.

## Required contracts

### [Atomic](../../../../glossary.md#resource-transaction-and-rollback) migration and compatibility policy

The migration is deliberately breaking and atomic. After implementation:

- `INV`, `FAST_EXP`, and `FAST_TANH` do not exist as enum [constants](../../../../glossary.md#term-constant);
- `inv`, `fastExp`, and `fastTanh` do not exist as declared [Tensor](../../../../glossary.md#tensor) methods;
- `UnaryElementwiseKind.valueOf("INV")`, `valueOf("FAST_EXP")`, and
  `valueOf("FAST_TANH")` fail with standard enum [lookup](../../../../glossary.md#term-lookup) failure;
- reflection cannot find the three removed Tensor methods;
- no deprecated bridge, forwarding alias, package-private duplicate, string mapping, or alternate
  semantic kind retains the old names; and<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- completed historical task documents may retain old names as history, while live Java and
  current API/glossary documentation must not.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

### Final [unary](../../../../glossary.md#term-unary) semantic vocabulary

`UnaryElementwiseKind` declares exactly:

```java
ABS,
NEG,
RECIPROCAL,
LOG,
EXP,
ERF,
SQRT,
FLOOR,
CEIL,
SIGN,
RELU,
SIGMOID,
TANH
```

The declaration order is part of the exact tested public surface. The meanings are:

| Kind | Portable [elementwise](../../../../glossary.md#term-elementwise) meaning |
|---|---|
| `ABS` | Absolute magnitude. |
| `NEG` | Additive inverse. |
| `RECIPROCAL` | Multiplicative reciprocal. |
| [`LOG`](../../../../glossary.md#term-log) | [Natural logarithm](../../../../glossary.md#norms-elementary-functions-and-probability-distributions). |
| [`EXP`](../../../../glossary.md#term-expx) | [Natural exponential](../../../../glossary.md#norms-elementary-functions-and-probability-distributions). |
| [`ERF`](../../../../glossary.md#term-erf) | [Gaussian error function](../../../../glossary.md#norms-elementary-functions-and-probability-distributions). |
| `SQRT` | Principal [square root](../../../../glossary.md#norms-elementary-functions-and-probability-distributions). |
| `FLOOR` | Greatest integer-valued result not greater than the input. |
| `CEIL` | Least integer-valued result not less than the input. |
| `SIGN` | Negative, zero, or positive classification represented numerically. |
| `RELU` | [Rectified linear unit](../../../../glossary.md#bias-and-activation). |
| [`SIGMOID`](../../../../glossary.md#term-sigmoid) | Logistic sigmoid. |
| [`TANH`](../../../../glossary.md#term-tanh) | [Hyperbolic tangent](../../../../glossary.md#bias-and-activation). |

The enum retains one stable [immutable](../../../../glossary.md#term-immutable) [signature](../../../../glossary.md#term-signature) list containing exactly
`OperationSignature.fixed(NoOperationAttrs.class, 1, 1)`. It adds no field or method other than
the existing family-owned signature implementation and [compiler](../../../../glossary.md#term-compiler)-generated enum machinery.

`EXP` and `TANH` do not mean “strict algorithm,” [correctly rounded](../../../../glossary.md#nan-infinity-and-signed-zero) result, bitwise [reproducibility](../../../../glossary.md#artifact-manifest-and-source-provenance),
or a mandatory library/[kernel](../../../../glossary.md#kernel) route. They are portable mathematical requests whose later
executable implementations must satisfy numerical policy defined outside this task. Removing the
fast variants does not strengthen their current numerical promise.

### Final public [unary](../../../../glossary.md#term-unary) [Tensor](../../../../glossary.md#tensor) surface

`Tensor` declares exactly these zero-[argument](../../../../glossary.md#term-argument) unary [elementwise](../../../../glossary.md#term-elementwise) methods:

```java
public Tensor abs()
public Tensor neg()
public Tensor reciprocal()
public Tensor log()
public Tensor exp()
public Tensor erf()
public Tensor sqrt()
public Tensor floor()
public Tensor ceil()
public Tensor sign()
public Tensor relu()
public Tensor sigmoid()
public Tensor tanh()
```

The exact mapping is one-to-one and order-aligned with the enum:

| Tensor method | Kind |
|---|---|
| `abs()` | `ABS` |
| `neg()` | `NEG` |
| `reciprocal()` | `RECIPROCAL` |
| `log()` | [`LOG`](../../../../glossary.md#term-log) |
| `exp()` | [`EXP`](../../../../glossary.md#term-expx) |
| `erf()` | [`ERF`](../../../../glossary.md#term-erf) |
| `sqrt()` | `SQRT` |
| `floor()` | `FLOOR` |
| `ceil()` | `CEIL` |
| `sign()` | `SIGN` |
| `relu()` | `RELU` |
| `sigmoid()` | [`SIGMOID`](../../../../glossary.md#term-sigmoid) |
| `tanh()` | [`TANH`](../../../../glossary.md#term-tanh) |

Each method delegates exactly once to
`TensorUnaryExpressions.apply(this, <matching kind>)` and returns that exact result. It performs
no duplicate [validation](../../../../glossary.md#term-validation), result construction, value inspection, [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form), or exception
translation.

The complete declared public `Tensor` method count becomes 110: the current 112 declarations,
minus `fastExp()` and `fastTanh()`, with `inv()` replaced one-for-one by `reciprocal()`.

### Preserved [unary](../../../../glossary.md#term-unary) [validation](../../../../glossary.md#term-validation) and result construction

`TensorUnaryExpressions` remains a final package-private non-record class with no fields or nested
types, one private zero-[argument](../../../../glossary.md#term-argument) constructor, and exactly one package-private static method:

```java
static Tensor apply(Tensor input, UnaryElementwiseKind kind)
```

`apply` retains this exact order and behavior:

1. null-check `input`, then `kind`, with messages `input` and `kind`;
2. read the exact input [`DataType`](../../../../glossary.md#data-type--datatype) and reject non-floating input with
   `IllegalArgumentException("input must be a floating data type, but was " + dataType)`;
3. create one new `TensorDescriptor` with the exact input [data type](../../../../glossary.md#data-type--datatype), exact input [`Shape`](../../../../glossary.md#shape)
   reference, `Optional.empty()` [layout](../../../../glossary.md#layout), and exact input `requiresGrad` value;
4. create one `Operation(kind, NoOperationAttrs.INSTANCE)`;
5. create one identity-distinct one-output producer through the existing factory path with exact
   ordered inputs <span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer))</span>`List.of(input)`; and
6. return the exact fresh [Tensor](../../../../glossary.md#tensor) at output index zero.

Validation failures precede Tensor [identity](../../../../glossary.md#term-identity-meanings) allocation. Identity exhaustion is allowed to fail
only through the existing factory path after local [immutable](../../../../glossary.md#term-immutable) [model](../../../../glossary.md#term-model) values have been constructed.
Do not add an identifier inspection [API](../../../../glossary.md#term-application-programming-interface-api).

Every successful method therefore preserves:

- exact [`BFLOAT16`](../../../../glossary.md#term-bfloat16), [`FLOAT32`](../../../../glossary.md#term-float32), or [`FLOAT64`](../../../../glossary.md#term-float64) input eligibility;
- exact input data type, exact immutable Shape reference, and exact `requiresGrad` value;
- unresolved result layout for [scalar](../../../../glossary.md#term-scalar), zero-sized, static, dynamic, and resolved-layout inputs;
- a fresh Tensor identity, empty label, and absent [host storage](../../../../glossary.md#host-storage);
- exact matching kind with `NoOperationAttrs.INSTANCE`;
- one identity-distinct [`TensorProducer`](../../../../glossary.md#producer--tensorproducer) with exactly the receiver as its sole ordered input and
  exactly one output descriptor;
- `TensorProvenance.outputIndex() == 0` and the exact result [descriptor](../../../../glossary.md#term-descriptor) reference; and
- input descriptor, [provenance](../../../../glossary.md#provenance), label, storage association, and storage contents unchanged.

Repeated calls and chains remain fresh. `reciprocal().reciprocal()` is not replaced by the
original input, and no other unary chain is simplified. `reciprocal()` does not inspect zero;
`log()` and `sqrt()` do not inspect mathematical domains; no unary method reads values.

### Retained [scalar](../../../../glossary.md#term-scalar) [elementwise](../../../../glossary.md#term-elementwise) contract

No scalar production or test file changes. The final adjacent scalar vocabulary remains exactly:

```java
MUL,
POW,
CLAMP,
CLAMP_MIN,
CLAMP_MAX
```

The pairings remain:

- [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) remains one final [immutable](../../../../glossary.md#term-immutable) [`DataType`](../../../../glossary.md#data-type--datatype) plus exact primitive-bit payload with the
  existing named FLOAT64, FLOAT32, converted/raw BFLOAT16, INT32, INT64, and BOOL factories,
  strict type-specific inspectors, exact type-and-bit equality/hashing, and no general conversion
  API;<span data-glossary-navigation="true"> ([payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [BOOL glossary](../../../../glossary.md#data-type--datatype); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- `MUL`, `POW`, `CLAMP_MIN`, and `CLAMP_MAX` accept exactly `ScalarValueAttrs`, one input, and one
  output;
- `CLAMP` accepts exactly `ClampRangeAttrs`, one input, and one output;
- `ScalarValueAttrs` retains one non-null exact `ScalarValue` reference without conversion;
- `ClampRangeAttrs` null-checks `minValue` then `maxValue`, requires the same data type, rejects
  BOOL, and rejects only strict inversion in the represented primitive type;<span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype))</span>
- `ClampRangeAttrs` continues to accept equal bounds, either signed-zero ordering, ordered
  infinities, and floating NaN endpoints while retaining exact typed bits; and<span data-glossary-navigation="true"> ([signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [NaN glossary](../../../../glossary.md#term-nan))</span>
- public typed scalar/clamp overloads, exact [FLOAT64](../../../../glossary.md#term-float64) `double` adapters, floating receiver/value
  equality, validation order, metadata, freshness, and provenance remain unchanged.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance))</span>

This task does not add missing scalar operations and does not change `ScalarValue`. Focused scalar
tests [run](../../../../glossary.md#term-run) as [regression](../../../../glossary.md#term-regression) coverage and should require no source edit.

### Boundary with task 0018T

Task 0018P removes ambiguous or non-portable surface and finalizes the existing [unary](../../../../glossary.md#term-unary) vocabulary.
Task 0018T later adds capabilities:

- [scalar](../../../../glossary.md#term-scalar) add, subtract, divide, minimum, and maximum using the exact typed scalar foundation;
- reciprocal-adjacent `rsqrt`, `log1p`, and `expm1`; and
- floating diagnostics `isFinite`, `isNaN`, and `isInf`.

Task 0018T may consume `RECIPROCAL`; it must not reintroduce `INV`, fast variants, or [aliases](../../../../glossary.md#aliasing-and-arena). No
detailed task 0018T specification is created here.

### Test migration

Update only these executable tests:

- `UnaryElementwiseKindTest` — assert the exact thirteen constants/order/names, canonical
  no-attributes composition, <span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>`RECIPROCAL` [identity](../../../../glossary.md#term-identity-meanings), absence of removed enum names through
  `valueOf`, and the unchanged [signature](../../../../glossary.md#term-signature) enum [shape](../../../../glossary.md#shape). Remove strict-versus-fast assertions.
- `TensorUnaryElementwiseTest` — use the exact thirteen method-to-kind mappings; replace every
  inverse chain/domain assertion with reciprocal terminology; remove fast-variant construction;
  prove fresh reciprocal chains, no zero inspection, exact producer/output-index-zero behavior,
  and absence of <span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer))</span>`inv`, `fastExp`, and `fastTanh` through reflection.
- `TensorTest` — update the exact declared public method count to 110, replace `inv` with
  `reciprocal` in the name inventory, remove `fastExp` and `fastTanh`, and assert the exact
  thirteen zero-argument unary declarations.<span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument); [unary glossary](../../../../glossary.md#term-unary))</span>

`OperationSignatureTest` remains unchanged because it already applies the same exact one-input,
one-output no-attributes signature to `UnaryElementwiseKind.values()`. `TensorCastExpressionTest`
remains unchanged because it uses retained `ABS` only. [Scalar](../../../../glossary.md#term-scalar) semantic/expression tests remain
unchanged and supply [regression](../../../../glossary.md#term-regression) evidence.

### Documentation migration

- `UnaryElementwiseKind`, [`Tensor`](../../../../glossary.md#tensor), and `TensorUnaryExpressions` Javadocs use “thirteen,”
  `RECIPROCAL`, and `reciprocal()`, remove the fast variants and strict-versus-fast wording, and
  preserve all current ownership/failure/result boundaries.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Tensor [API](../../../../glossary.md#term-application-programming-interface-api) lists exactly thirteen kinds and methods, uses `RECIPROCAL`/`reciprocal`, removes all
  current `INV`/`inv`/fast entries, and explains [`EXP`](../../../../glossary.md#term-expx)/[`TANH`](../../../../glossary.md#term-tanh) portability without algorithm,
  bitwise-result, bound, or route promises.
- Replace the complete `fastExp()` example with a runnable [metadata](../../../../glossary.md#term-metadata)-only reciprocal example using
  `input.reciprocal()` and expected `kind=RECIPROCAL`. Its interpretation must state that it proves
  metadata/provenance only, not division, zero/special-value behavior, numerical accuracy,
  gradients, capture, backend support, or execution.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [gradients glossary](../../../../glossary.md#term-gradient); [capture glossary](../../../../glossary.md#term-capture-meanings); [backend glossary](../../../../glossary.md#backend))</span>
- Compile API changes the current unary method count from fifteen to thirteen and aligns the
  retained semantic wording. It must not claim compiler capture, inference, optimization,
  lowering, or execution.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [unary glossary](../../../../glossary.md#term-unary); [compiler glossary](../../../../glossary.md#term-compiler); [inference glossary](../../../../glossary.md#term-inference); [lowering glossary](../../../../glossary.md#lowering))</span>
- Glossary changes the exact unary inventory and current expression count, removes fast-request
  terminology, and uses reciprocal terminology in provenance/status descriptions.
- Capability baseline records task 0018P as the ready cleanup owner during planning and as the
  completed owner only after implementation evidence is final. It keeps task 0018T's additions
  separate.<span data-glossary-navigation="true"> ([ready glossary](../../../../glossary.md#term-ready); [owner glossary](../../../../glossary.md#term-owner); [planning glossary](../../../../glossary.md#term-planning))</span>
- Model master plan and roadmap link this specification and synchronize status/evidence without
  rewriting completed 0014C/0014D history.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- Training API receives a reasoned no-change conclusion because no gradient, autograd, optimizer,
  or training behavior changes.<span data-glossary-navigation="true"> ([autograd glossary](../../../../glossary.md#autograd); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>

## Affected files

Production:

1. `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/unary/UnaryElementwiseKind.java`
2. `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorUnaryExpressions.java`
3. `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Tests:

4. `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/elementwise/unary/UnaryElementwiseKindTest.java`
5. `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorUnaryElementwiseTest.java`
6. `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`

Documentation and [planning](../../../../glossary.md#term-planning):

7. `docs/api/tensor-api.md`
8. `docs/api/compile-api.md`
9. `docs/glossary.md`
10. `docs/planning/modules/model/capabilities.md`
11. this task specification
12. `docs/planning/modules/model/master-plan.md`
13. `docs/planning/roadmap.md`

Required review without modification unless a contradiction requires stopping:

- `ScalarElementwiseKind`, `ScalarValueAttrs`, `ClampRangeAttrs`, [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue),
  `TensorScalarExpressions`, `ScalarElementwiseSemanticsTest`, and
  `TensorScalarElementwiseTest`
- `OperationSignature`, [`OperationKind`](../../../../glossary.md#operationkind), [`Operation`](../../../../glossary.md#operation), `OperationSignatureTest`,
  [`TensorProducer`](../../../../glossary.md#producer--tensorproducer), `TensorProvenance`, and `TensorFactory`
- `TensorCastExpressionTest`, whose retained `ABS` [provenance](../../../../glossary.md#provenance) setup is unaffected
- Training [API](../../../../glossary.md#term-application-programming-interface-api), `ARCHITECTURE.md`, focused architecture docs, ADRs/tests, backend-conformance,
  integration tests, Gradle configuration, dependencies, and other modules<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module))</span>

## Maximum [scope](../../../../glossary.md#term-scope)

This task may modify exactly the thirteen listed paths and may not create or delete another path.

This is a bounded one-path exception to the [planning](../../../../glossary.md#term-planning) guide's approximate 3–12-file preference.
The thirteenth path is not a second capability: the three production surfaces, their three exact
inventory/behavior tests, and seven required [API](../../../../glossary.md#term-application-programming-interface-api)/glossary/capability/task/master/roadmap records
must migrate together so old and new names cannot coexist. Splitting documentation or public-
inventory updates would leave a contradictory stable API; splitting semantic and public names
would make the build uncompilable. The task remains within one [module](../../../../glossary.md#term-module) and one cohesive cleanup.

Do not use this allowance for [scalar](../../../../glossary.md#term-scalar) changes, [operation](../../../../glossary.md#operation)-[signature](../../../../glossary.md#term-signature) changes, unrelated formatting,
another test, another document, a later detailed specification, or a compatibility bridge. If a
fourteenth path or another semantic ambiguity is discovered, stop and report the exact need before
editing it.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- `UnaryElementwiseKind` type Javadoc must define the final thirteen-kind family, exact shared
  signature, semantic-only role, and diagnostic-name boundary.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature))</span>
- Every retained constant must have meaningful mathematical Javadoc and defer type/result,
  special-value, numerical, gradient, execution, and backend rules as applicable.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [gradient glossary](../../../../glossary.md#term-gradient); [backend glossary](../../../../glossary.md#backend))</span>
- `RECIPROCAL` must say multiplicative reciprocal and must not use “inverse” as the API name or
  promise zero handling.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [`EXP`](../../../../glossary.md#term-expx) and [`TANH`](../../../../glossary.md#term-tanh) must state portable mathematical meaning without “strict” contrast and without
  promising an algorithm, correctly rounded/bitwise result, approximation bound, or route.<span data-glossary-navigation="true"> ([correctly rounded glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Every retained public Tensor method must document floating eligibility, exact metadata
  preservation, unresolved layout, fresh identity, no label/storage, exact kind/no-attributes,
  one-input producer/provenance output zero, deferred value/numerical/gradient/backend behavior,<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [metadata glossary](../../../../glossary.md#term-metadata); [layout glossary](../../../../glossary.md#layout); [identity glossary](../../../../glossary.md#term-identity-meanings); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>
  `@return`, and expected failures.
- `Tensor.reciprocal()` must document zero and special-value policy as deferred.
- `TensorUnaryExpressions` must say thirteen and retain its exact validation, ownership, ID-side-
  effect, and failure documentation.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Review the scalar family Javadocs and record a reasoned no-change conclusion rather than editing
  them.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- A separate clean-context documentation-focused agent must inspect final source, tests,
  generated Javadoc, Tensor/Compile APIs, glossary, capability baseline, and the complete diff
  before completion.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>

## Acceptance criteria

- `UnaryElementwiseKind` contains exactly the thirteen specified [constants](../../../../glossary.md#term-constant) in exact order.
- [`Tensor`](../../../../glossary.md#tensor) contains exactly the thirteen specified public zero-argument unary elementwise methods
  and maps each once to the matching kind.<span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument); [elementwise glossary](../../../../glossary.md#term-elementwise))</span>
- `INV`, `FAST_EXP`, `FAST_TANH`, `inv`, `fastExp`, and `fastTanh` are absent from live production
  and test Java, including reflection and `valueOf` checks.
- No alias, deprecated bridge, string translation, alternate kind, or route flag preserves a
  removed spelling.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- [`EXP`](../../../../glossary.md#term-expx) and [`TANH`](../../../../glossary.md#term-tanh) remain representable portable mathematical requests with no strengthened
  numerical or backend promise.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- Every final [unary](../../../../glossary.md#term-unary) kind retains exact [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs), one-input, one-output signature
  behavior.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature))</span>
- Null/type [validation](../../../../glossary.md#term-validation) order and exact messages remain unchanged.
- Every successful unary call accepts only floating input, preserves exact [Shape](../../../../glossary.md#shape)/type/
  `requiresGrad`, leaves layout unresolved, is fresh/unlabeled/storage-free, and records one exact
  input in an identity-distinct one-output producer with provenance output index zero.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [identity glossary](../../../../glossary.md#term-identity-meanings); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>
- Reciprocal and other unary chains are not canonicalized; values, storage, and mathematical
  domains are not inspected.
- `ScalarElementwiseKind` remains exactly `MUL`, `POW`, `CLAMP`, `CLAMP_MIN`, `CLAMP_MAX`; exact
  `ScalarValueAttrs`/`ClampRangeAttrs` contracts and [scalar](../../../../glossary.md#term-scalar) public methods remain unchanged.
- No task-0018T scalar or unary gap is implemented here.
- Tensor's exact public declared-method count and name inventory are updated to the final surface.
- Tensor API's former fast-exp example is replaced with the specified reciprocal metadata
  example, and current API/glossary text contains no removed vocabulary.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata))</span>
- [Compile](../../../../glossary.md#compile) [API](../../../../glossary.md#term-application-programming-interface-api) reports thirteen unary methods without claiming [compiler](../../../../glossary.md#term-compiler) behavior.
- Focused tests, final [model](../../../../glossary.md#term-model) tests, model [Javadoc](../../../../glossary.md#term-javadoc), documentation checks, exact [scope](../../../../glossary.md#term-scope) checks, and
  `git diff --check` pass.
- The independent documentation pass records reasoned no-change conclusions for scalar contracts,
  operation/signature/provenance foundations, Training API, architecture/ADRs/tests,
  conformance/integration, Gradle/dependencies, and other modules.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module))</span>
- Task, model master-plan row, capability baseline, and roadmap status are synchronized only after
  evidence is final.
- 0018Q and later tasks remain Draft without detailed specifications.

## Tests / [validation](../../../../glossary.md#term-validation)

During implementation, [run](../../../../glossary.md#term-run) the focused contract set:

```bash
./gradlew :modules:model:test \
  --tests io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKindTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorUnaryElementwiseTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorTest \
  --tests io.github.pho001.synaptik.model.operation.OperationSignatureTest \
  --tests io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseSemanticsTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorScalarElementwiseTest
```

After executable Java stabilizes, record one final [module](../../../../glossary.md#term-module) run:

```bash
./gradlew :modules:model:test
```

The implementation pass also runs exact live-Java absence checks:

```bash
rg -n '\bINV\b|FAST_EXP|FAST_TANH|\binv\(|fastExp\(|fastTanh\(' \
  modules/model/src/main/java modules/model/src/test/java
```

The result must be empty. Automated reflection and enum tests must prove the same public absence,
the exact thirteen-item inventories, method count 110, [signature](../../../../glossary.md#term-signature), [producer](../../../../glossary.md#term-producer), [descriptor](../../../../glossary.md#term-descriptor), and
output-index contracts. Do not use repeated manual [bytecode](../../../../glossary.md#term-bytecode) checks for facts covered by stable
tests.

The separate documentation-focused pass reuses the successful [model](../../../../glossary.md#term-model)-test evidence unless it
changes executable Java. After final [Javadoc](../../../../glossary.md#term-javadoc) edits it runs:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

It also:

- compiles and runs the replacement reciprocal example against the final Java 26 model classes;
- confirms exact enum and Tensor surfaces with the automated evidence and a targeted generated-
  Javadoc inspection;
- checks that current [Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api), [Compile](../../../../glossary.md#compile) API, and glossary contain none of the removed names;
- checks local Markdown file links and GitHub-style anchors in all seven changed documentation
  and planning files;<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- checks balanced fences, final newlines, trailing whitespace, terminology, and authority
  boundaries;
- verifies exactly the thirteen authorized paths;
- verifies 0018P is `Complete` only after both passes, 0018Q is the next Draft row, and no 0018Q-
  or-later detailed task exists; and
- confirms no architecture, [Gradle](../../../../glossary.md#term-gradle), [dependency](../../../../glossary.md#term-dependency), other-module, [commit](../../../../glossary.md#term-commit), or push change.

Repository-wide `./gradlew test` is deferred to the recorded public-surface cleanup [checkpoint](../../../../glossary.md#term-checkpoint)
after task 0018S. This task changes one module and no dependency, build, or architecture boundary.

## Dependencies

- Tasks 0014C and 0014D — completed provisional [unary](../../../../glossary.md#term-unary) kinds and public expression construction.
- Tasks 0014E and 0014F — completed [scalar](../../../../glossary.md#term-scalar) vocabulary and expression boundary retained unchanged.
- Task 0018K — completed exact kind/attributes [signatures](../../../../glossary.md#term-signature) and [occurrence](../../../../glossary.md#term-occurrence) [cardinality](../../../../glossary.md#term-cardinality).
- Task 0018N — completed exact typed scalar values and attributes retained unchanged.
- Task 0018O — completed the immediately preceding public-taxonomy cleanup.

All dependencies are Complete. No unresolved architecture or package [dependency](../../../../glossary.md#term-dependency) blocks this task.

## Follow-up tasks

- 0018Q — [masked reduction](../../../../glossary.md#masked-reduction) redesign; independent of this [unary](../../../../glossary.md#term-unary) cleanup.
- 0018T — add [scalar](../../../../glossary.md#term-scalar) add/subtract/divide/minimum/maximum plus `rsqrt`, `log1p`, `expm1`, and
  floating diagnostic semantics on the cleaned vocabulary.
- 0019A — add selected modern [activation](../../../../glossary.md#term-activation) conveniences after 0018P and 0018T.
- Prepare/backend/conformance tasks — select implementation routes and establish executable
  numerical/special-value contracts for retained mathematical requests.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>

Do not create a detailed follow-up specification in this task.

## Architecture impact

Expected impact: None.

The architecture already assigns [backend](../../../../glossary.md#backend)-independent [operation](../../../../glossary.md#operation) semantics and public [Tensor](../../../../glossary.md#tensor)
construction to `modules/model`, while [concrete backend](../../../../glossary.md#concrete-backend) [prepare](../../../../glossary.md#prepare) owns [lowering](../../../../glossary.md#lowering) and [kernel](../../../../glossary.md#kernel) route
selection. Removing route-oriented semantic names and adopting explicit reciprocal terminology
clarifies that boundary without changing it.

If implementation requires a [module](../../../../glossary.md#term-module)/[dependency](../../../../glossary.md#term-dependency)/[lifecycle](../../../../glossary.md#lifecycle) rule change, numerical contract, or
backend behavior, stop and report the exact conflict before editing architecture files.

## Implementation prompt

Use this prompt in a separate agentic task/thread with clean context:

```text
You are working in the Synaptik repository.

Read AGENTS.md, ARCHITECTURE.md, docs/developer-guide/documentation-rules.md,
docs/planning/planning-guide.md, docs/planning/roadmap.md, model capabilities/master plan, tasks
0014C/0014D/0014E/0014F/0018K/0018N/0018O/0018P, and every affected/review-only source and test
listed by task 0018P in full.

Implement task 0018P exactly. Perform the atomic INV/inv to RECIPROCAL/reciprocal migration and
complete FAST_EXP/FAST_TANH/fastExp/fastTanh removal. Preserve unary construction, scalar
contracts, signatures, producer/provenance, metadata, and failure order. Do not add aliases,
numeric policy, task-0018T capabilities, compiler/prepare/backend behavior, dependencies, build,
architecture changes, or paths outside the exact thirteen-path scope. Stop on any new semantic
ambiguity or scope/architecture conflict. Do not commit or push.

Run the focused command, final model suite, and live-Java absence check after executable code
stabilizes. Then hand the actual diff and exact test evidence to a separate clean-context
documentation-focused agent in the same overall change. That agent must independently finalize
the affected Javadocs, replace the Tensor API fast-exp example with the reciprocal example,
finalize Tensor/Compile APIs, glossary, capabilities, task/master/roadmap evidence, and run model
Javadoc plus all specified documentation/scope checks without repeating successful Java tests
unless executable behavior changes or a concrete risk is recorded.

Mark 0018P Complete only after both passes succeed. Leave 0018Q and every later task Draft without
a detailed specification.
```

## Local decisions

- The final kind and method name is `RECIPROCAL`/`reciprocal`, not `INV`, `INVERSE`, or a scalar
  division convenience, because it names the elementwise mathematical request directly.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [elementwise glossary](../../../../glossary.md#term-elementwise))</span>
- No compatibility aliases are retained. The API is still pre-stabilization, and coexistence
  would preserve duplicate semantic identities and enlarge every downstream inventory.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([aliases glossary](../../../../glossary.md#aliasing-and-arena); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [`EXP`](../../../../glossary.md#term-expx) and [`TANH`](../../../../glossary.md#term-tanh) remain one portable semantic request each. Backend prepare selects the route;
  model does not expose a speed/accuracy route preference without a portable contract.<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [prepare glossary](../../../../glossary.md#prepare); [model glossary](../../../../glossary.md#term-model))</span>
- The current reciprocal declaration position is retained, so the final enum and public method
  ordering changes only by the rename and removal of the two trailing fast variants.
- The scalar family remains unchanged. Its missing operations and reciprocal-adjacent additions
  are cohesive task 0018T work, not cleanup fallout.
- The complete API example becomes reciprocal-focused because it demonstrates the renamed public
  contract and metadata boundary without implying that <span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata))</span>`EXP` now has a stronger numerical
  guarantee.

## Known limitations

- The final [unary](../../../../glossary.md#term-unary) expressions still have no computed values or [host storage](../../../../glossary.md#host-storage).
- Numerical domain, special-value, accuracy, determinism, and gradient policies remain undefined
  until owning compiler/backend/conformance tasks specify them.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- No [backend](../../../../glossary.md#backend) is made capable of executing the retained requests by this cleanup.
- Existing completed task documents retain the provisional history and therefore still contain
  the removed names outside current API/glossary and live Java.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Task 0018T remains Draft and its additions are not available after this task alone.

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) context `/root/plan_0018p` read the [architecture contract](../../../../glossary.md#architecture-contract) and current architecture index;
documentation rules and General, [API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc), Planning, and Example profiles; planning guide and
roadmap; [model](../../../../glossary.md#term-model) capabilities and master plan; completed tasks 0014C, 0014D, 0014E, 0014F, 0018K,
0018N, and 0018O; relevant [Tensor](../../../../glossary.md#tensor)/[Compile](../../../../glossary.md#compile) API and glossary sections; Java 26 [Gradle](../../../../glossary.md#term-gradle) configuration;
and the current [unary](../../../../glossary.md#term-unary)/[scalar](../../../../glossary.md#term-scalar)/[signature](../../../../glossary.md#term-signature)/[producer](../../../../glossary.md#term-producer)/[provenance](../../../../glossary.md#provenance) production and focused test surfaces.

Planning found no architecture conflict or additional semantic ambiguity. The current helper,
signature, [descriptor](../../../../glossary.md#term-descriptor), producer, and provenance contracts already support the final vocabulary
without executable redesign. The exact thirteen-path [scope](../../../../glossary.md#term-scope) is cohesive and the scalar family can
remain source- and behavior-unchanged.

Planning-only validation is recorded before [handoff](../../../../glossary.md#term-handoff):

- a canonical-section scan found every required task-specification section exactly once and no
  missing or duplicate section; the task contains no unresolved design placeholder;<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- the final enum/method inventories, failure order, before/after mapping, scalar no-change
  contract, 0018T boundary, affected paths, and maximum scope are explicit;
- status scans found 0018P Ready exactly once in this task, the master-plan row, and the roadmap
  row; 0018O remains Complete, and 0018Q and every later row remain Draft;<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- a task-file inventory found no 0018Q-or-later detailed specification;
- a local Markdown target-and-GitHub-anchor check resolved all 309 relative links across this
  task, model master plan, capability baseline, and roadmap;<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>
- fence checks found 24 balanced backtick fences in this task and two in the master plan, with no
  unbalanced tilde fence; all four changed files have final newlines and no trailing whitespace;
- final planning-only scope contains exactly four paths: this new task plus the model master plan,
  capability baseline, and roadmap; no Java, test, Gradle, architecture, API, glossary, other-
  module, completed-task, or later-task file changed; and<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>
- `git diff --check` passed with no output; the untracked task file separately passed the
  trailing-whitespace and final-newline checks that a tracked diff does not inspect.

No Gradle test or Javadoc task was [run](../../../../glossary.md#term-run) because this planning-only change modifies no Java,
executable behavior, public Javadoc, or current API documentation.

Implementation context `/root/task_0018p_implementation` completed the executable migration and
reported this final Java evidence, which the documentation pass reused because no executable Java
changed afterward:

- the exact focused Gradle command from this task passed with `BUILD SUCCESSFUL`; XML reports
  contain 50 tests across `UnaryElementwiseKindTest` (6), `TensorUnaryElementwiseTest` (8),
  `TensorTest` (15), `OperationSignatureTest` (5), `ScalarElementwiseSemanticsTest` (10), and
  `TensorScalarElementwiseTest` (6), with zero failures, errors, or skips;
- `./gradlew :modules:model:test` passed with `BUILD SUCCESSFUL in 1s`; XML reports contain 725
  tests across 88 suites with zero failures, errors, or skips;
- the exact live-Java search for `INV`, both fast constants, and the three removed method
  spellings returned no matches; and<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant))</span>
- `git diff --check` passed with no output after implementation.

Independent documentation context `/root/task_0018p_implementation/task_0018p_docs` read the
architecture contract and current architecture index; documentation rules and General,
API/Javadoc, Planning, and Example profiles; planning guide and roadmap; model capabilities,
master plan, and task 0018P; the final affected source/tests and actual combined diff; Tensor,
Compile, and Training APIs; glossary; and the adjacent scalar, signature, [operation](../../../../glossary.md#operation), producer,
provenance, and factory contracts. It found no architecture, executable-contract, or scope defect
and finalized the three affected Javadocs plus all seven authorized documentation/planning paths.

Final documentation evidence:

- `./gradlew :modules:model:javadoc` passed after the final Javadocs with `BUILD SUCCESSFUL in
  1s`; two actionable tasks executed and the configuration cache was reused. Targeted generated
  pages show the thirteen-kind family, <span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache))</span>`RECIPROCAL`, the portable [`EXP`](../../../../glossary.md#term-expx)/[`TANH`](../../../../glossary.md#term-tanh) boundaries, and the
  thirteen-method Tensor summary, with no removed public vocabulary.
- `javac -cp modules/model/build/classes/java/main -d
  /tmp/synaptik-reciprocal-doc-example /tmp/UnaryExpressionExample.java && java -cp
  modules/model/build/classes/java/main:/tmp/synaptik-reciprocal-doc-example
  UnaryExpressionExample` compiled and ran against the final Java 26 model classes. It printed
  [`FLOAT32`](../../../../glossary.md#term-float32), exact Shape retention, unresolved layout, true gradient eligibility, absent label and
  storage, <span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [gradient glossary](../../../../glossary.md#term-gradient))</span>`kind=RECIPROCAL`, [canonical](../../../../glossary.md#term-canonical-form) no-attributes, exact input, `outputIndex=0`, one output,
  and fresh identity exactly as documented.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Current Tensor API, Compile API, and glossary searches contain none of `INV`, `FAST_EXP`,
  `FAST_TANH`, `inv(`, `fastExp`, or `fastTanh`. The matching live production/test Java search is
  also empty.
- The targeted local Markdown checker resolved 441 local links, including 113 heading anchors,
  across the seven changed documentation/planning files with zero failures. Formatting checks
  found balanced fences, final newlines, and no trailing whitespace in all seven files.
- Preliminary local checker runs exposed only checker defects: an unescaped Ruby heading marker,
  collapsed double-hyphen GitHub slugs, and a scope-list expression that retained only the
  untracked-file command output. The corrected checker reruns produced the successful Markdown and
  exact-scope results recorded here; they did not expose repository defects.
- The final inventory contains exactly the thirteen authorized paths: three production Java
  files, three tests, and seven documentation/planning files. Task 0018P, its master-plan row, and
  its roadmap row are Complete; 0018Q and every later task remain Draft, and no detailed
  0018Q-or-later task file exists.
- Final `git diff --check` passed with no output. No architecture contract or focused architecture
  page, ADR, architecture test, Training API, backend-conformance/integration test, Gradle or
  dependency file, other module, commit, or push changed.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [architecture test glossary](../../../../glossary.md#term-architecture-test); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [integration test glossary](../../../../glossary.md#term-integration-test); [dependency glossary](../../../../glossary.md#term-dependency); [commit glossary](../../../../glossary.md#term-commit))</span>

Repository-wide `./gradlew test` remains deferred to the recorded public-surface cleanup
[checkpoint](../../../../glossary.md#term-checkpoint) after task 0018S, as specified. No validation was skipped beyond that planned
checkpoint deferral.

## Implementation notes

- The migration is deliberately atomic and retains no alias or deprecated bridge. Historical
  completed task documents remain unchanged records of the provisional names.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- [`EXP`](../../../../glossary.md#term-expx) and [`TANH`](../../../../glossary.md#term-tanh) remain portable mathematical requests. Their Javadocs and API reference select
  no algorithm, bitwise result, approximation bound, or backend route.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [backend route glossary](../../../../glossary.md#backend-route))</span>
- Scalar-family Javadocs and behavior remain accurate unchanged: the exact five-kind vocabulary,
  typed scalar/range attributes, validation order, metadata, and provenance do not depend on the
  renamed unary kind.<span data-glossary-navigation="true"> ([Scalar glossary](../../../../glossary.md#term-scalar); [validation glossary](../../../../glossary.md#term-validation); [metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance); [unary glossary](../../../../glossary.md#term-unary))</span>
- `OperationSignature`, [`OperationKind`](../../../../glossary.md#operationkind), [`Operation`](../../../../glossary.md#operation), [`TensorProducer`](../../../../glossary.md#producer--tensorproducer), `TensorProvenance`, and
  `TensorFactory` remain accurate unchanged because the task preserves the same no-attributes,
  one-input, one-output construction mechanics.
- Training API remains accurate unchanged because no gradient, autograd, optimizer, or training
  behavior changed. Architecture/ADRs/tests remain unchanged because ownership, dependencies, and
  lifecycle boundaries did not change. Backend-conformance/integration tests remain unchanged
  because no lowering or execution behavior changed. Java 26 Gradle configuration, dependencies,
  and other modules remain unchanged because the cleanup is confined to model vocabulary and
  metadata construction.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [autograd glossary](../../../../glossary.md#autograd); [optimizer glossary](../../../../glossary.md#term-optimizer); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [lifecycle glossary](../../../../glossary.md#lifecycle); [Backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [lowering glossary](../../../../glossary.md#lowering); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module); [model glossary](../../../../glossary.md#term-model))</span>

## Completion summary

- Completed changes: atomically renamed [unary](../../../../glossary.md#term-unary) reciprocal vocabulary to
  `RECIPROCAL`/`reciprocal`, removed both fast variants without aliases, retained thirteen exact
  unary kinds/methods, and preserved unary construction plus typed scalar contracts.<span data-glossary-navigation="true"> ([aliases glossary](../../../../glossary.md#aliasing-and-arena); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Files changed or created: exactly the thirteen authorized production, test, API, glossary,
  capability, task, master-plan, and roadmap paths.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Tests and validation: reused the passing 50-test focused set and 725-test/88-suite final model
  run; final model Javadoc, runnable reciprocal example, generated-page, removed-vocabulary,
  Markdown link/anchor, fence/newline/whitespace, exact-scope/status, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [model glossary](../../../../glossary.md#term-model); [run glossary](../../../../glossary.md#term-run); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check`
  validation passed.
- Documentation-agent review: clean-context review
  `/root/task_0018p_implementation/task_0018p_docs` completed.
- Documentation impact: Tensor and Compile APIs, glossary, capability baseline, task, master plan,
  and roadmap now describe the final current vocabulary and ownership boundaries.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Javadoc review: affected unary kind, Tensor, and helper Javadocs are final; adjacent scalar and
  foundational Javadocs remain accurate unchanged for the reasons recorded above.
- Glossary impact: current inventories and Tensor status use thirteen unary requests and
  reciprocal terminology; removed route-oriented vocabulary is absent.
- Unresolved issues: None.
- Follow-up required: None. Task 0018Q remains the next Draft frontier; task 0018T retains the
  missing scalar/unary numeric additions.

Status: Complete
