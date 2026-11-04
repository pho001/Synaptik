# Task 0018U: Integral [Elementwise](../../../../glossary.md#term-elementwise) Arithmetic and Comparisons

**Editorial reading key (current context, not a status update).**

This completed record concerns expression construction, not execution; its deferred-work statements
describe that historical scope, not current capability. The caller invokes a Tensor method: the
receiver is the left operand, the argument the right. Model helpers validate descriptors and record
ordered provenance—the producing operation and input references—without reading element values. A
scalar instead crosses this boundary as an exact typed attribute, not another Tensor input: INT32
multiplied by an INT32 scalar retains INT32; an INT64 scalar is rejected. Storage-free results
neither execute nor own result buffers; semantic responsibility is not storage ownership.

For current promotion, broadcasting and modular-wrap meaning, follow [binary arithmetic
expressions](../../../../reference/tensor/arithmetic.md#binary-arithmetic-expressions). Promotion
selects an operation domain without inserting a cast. For the distinction between that domain and
the output, follow [binary comparison
expressions](../../../../reference/tensor/arithmetic.md#binary-comparison-expressions): comparing
INT32 with INT64 uses signed INT64 ordering but describes a BOOL result, not computed truth values.
Model semantics defines these meanings; execution providers realize supported routes rather than
redefine them.

## Status

Complete

## Goal

Make current [INT32](../../../../glossary.md#data-type--datatype) and [INT64](../../../../glossary.md#data-type--datatype) [Tensors](../../../../glossary.md#tensor) useful as [backend](../../../../glossary.md#backend)-independent [elementwise](../../../../glossary.md#term-elementwise) numeric expressions
without broadening into division, reductions, or execution.

Extend the existing public operations so the selected integral domain supports:

```text
Tensor-to-Tensor arithmetic: ADD, SUB, MUL, MIN, MAX
Tensor-to-scalar arithmetic: ADD, SUB, MUL, MIN, MAX
Tensor-to-Tensor comparison: GREATER_THAN, GREATER_OR_EQUAL,
                             LESS_THAN, LESS_OR_EQUAL, EQUAL, NOT_EQUAL
```

Preserve every existing floating behavior. Reject [BOOL](../../../../glossary.md#data-type--datatype) and mixed floating/integral pairs. Keep
integral `DIV`, `POW`, range `CLAMP`, reductions, and `argMin` outside this task.

This task constructs typed result [descriptors](../../../../glossary.md#term-descriptor) and [producer](../../../../glossary.md#term-producer)/[provenance](../../../../glossary.md#provenance) [metadata](../../../../glossary.md#term-metadata) only. It does not
read values, execute arithmetic, detect [overflow](../../../../glossary.md#term-overflow) at construction, generate [gradients](../../../../glossary.md#term-gradient), [capture](../../../../glossary.md#term-capture-meanings) a
[graph](../../../../glossary.md#graph), lower operations, or add [kernels](../../../../glossary.md#kernel).

## Why this is a separate task

Today an integral [Tensor](../../../../glossary.md#tensor) can [carry](../../../../glossary.md#term-carry) [model](../../../../glossary.md#term-model) inputs, labels, counters, or indices, but the ordinary
arithmetic and comparison helpers reject it before creating an expression:

```java
Tensor shifted = tokenIds.add(offset);       // currently rejected
Tensor equal = predictions.equalTo(labels);  // currently rejected
```

After this task the same public vocabulary works for selected floating and signed-integral domains:

```java
Tensor shifted = tokenIds.add(ScalarValue.int32(1));
Tensor equal = predictions.equalTo(labels);
```

The later task 0018U1 owns integral reductions and `argMin`. Separating it avoids combining local
[elementwise](../../../../glossary.md#term-elementwise) promotion and [broadcasting](../../../../glossary.md#broadcasting) with accumulation, empty-domain, tie-policy, and public
arg-reduction type [normalization](../../../../glossary.md#term-normalization) in one oversized implementation session.

## [Mental model](../../../../glossary.md#term-mental-model)

### Mixed-width integral [Tensor](../../../../glossary.md#tensor) operands

```java
Tensor result = int32Left.add(int64Right);
```

Conceptually records:

```text
promotion = INT64
shape = broadcast(int32Left.shape, int64Right.shape)

TensorProducer
  operation = BinaryArithmeticKind.ADD + NoOperationAttrs.INSTANCE
  inputs = [int32Left, int64Right]
  outputDescriptors = [INT64, broadcast Shape, unresolved layout, requiresGrad=false]
```

The [model](../../../../glossary.md#term-model) does not insert a visible CAST [producer](../../../../glossary.md#term-producer). Promotion is a semantic operand/result-domain
rule for this [operation](../../../../glossary.md#operation) [occurrence](../../../../glossary.md#term-occurrence), just like the existing floating promotion hierarchy.

### Exact typed [scalar](../../../../glossary.md#term-scalar)

```java
Tensor result = int32Input.mul(ScalarValue.int32(4));
```

Conceptually records one input and one exact attribute:

```text
operation = ScalarElementwiseKind.MUL
attrs = ScalarValueAttrs(ScalarValue.int32(4))
inputs = [int32Input]
result type = INT32
```

Scalar [parameters](../../../../glossary.md#term-parameter) do not promote. `ScalarValue.int64(4)` is rejected for an [INT32](../../../../glossary.md#data-type--datatype) receiver because
the exact typed attribute is not another [Tensor](../../../../glossary.md#tensor) operand and no implicit scalar conversion contract
exists.

### Integral comparison

```java
Tensor mask = int32Left.lessThan(int64Right);
```

The comparison domain promotes to [INT64](../../../../glossary.md#data-type--datatype), but the result is:

```text
data type = BOOL
shape = broadcast(left.shape, right.shape)
requiresGrad = false
operation = BinaryComparisonKind.LESS_THAN
inputs = [left, right]
```

## Selected promotion contract

Extend the existing stateless public `DataTypePromotion` utility with exactly one new method:

```java
public static DataType promoteNumeric(DataType left, DataType right)
```

The method accepts only two operands from the same numeric category.

### Floating pairs

Preserve the existing hierarchy exactly:

```text
BFLOAT16 < FLOAT32 < FLOAT64
```

Every floating pair returns the same result as `promoteFloating(left, right)`.

### Integral pairs

Use this exact symmetric hierarchy:

```text
INT32 < INT64
```

The complete [matrix](../../../../glossary.md#term-matrix-meanings) is:

| Left | Right | Result |
|---|---|---|
| [INT32](../../../../glossary.md#data-type--datatype) | INT32 | INT32 |
| INT32 | [INT64](../../../../glossary.md#data-type--datatype) | INT64 |
| INT64 | INT32 | INT64 |
| INT64 | INT64 | INT64 |

INT32 values conceptually sign-extend into the INT64 [operation](../../../../glossary.md#operation) domain. No explicit CAST [producer](../../../../glossary.md#term-producer) is
stored by public expression construction.

### Rejected pairs

- Floating plus integral is rejected in either order. Callers must make conversion intent visible
  through `Tensor.cast(...)`.
- [BOOL](../../../../glossary.md#data-type--datatype) with any operand is rejected. BOOL remains logical rather than numeric.
- No integral value promotes to floating merely because floating has a wider bit width.

`promoteFloating` remains public with its exact existing behavior, messages, and tests. Do not
replace or weaken it.

## Integral arithmetic domain

### Supported kinds

Integral [Tensor](../../../../glossary.md#tensor) operands support exactly:

```text
ADD
SUB
MUL
MIN
MAX
```

The same five kinds are supported for exact typed [scalar](../../../../glossary.md#term-scalar) attributes.

### Deliberately unsupported kinds

Integral operands reject:

```text
DIV
POW
```

`DIV` requires explicit decisions for [truncation](../../../../glossary.md#term-truncation-toward-zero) versus floor [rounding](../../../../glossary.md#term-rounding), division by zero, and
`MIN_VALUE / -1`. `POW` requires negative-[exponent](../../../../glossary.md#term-exponent), [overflow](../../../../glossary.md#term-overflow), and result-domain rules. Neither is
needed for the selected minimal integral baseline, so this task does not invent those policies.

First-class two-bound `CLAMP` remains floating-only. The existing one-bound `clampMin` and
`clampMax` conveniences delegate to [scalar](../../../../glossary.md#term-scalar) MAX and MIN, so they inherit exact typed integral
eligibility without adding an integral CLAMP [operation](../../../../glossary.md#operation):

```java
int32Tensor.clampMin(ScalarValue.int32(0)); // scalar MAX, valid
int32Tensor.clampMax(ScalarValue.int32(9)); // scalar MIN, valid
int32Tensor.clamp(int32Zero, int32Nine);    // first-class CLAMP, still rejected
```

## [Overflow](../../../../glossary.md#term-overflow) and ordering policy

The [model](../../../../glossary.md#term-model) selects ordinary fixed-width [two's-complement modular arithmetic](../../../../glossary.md#arithmetic-rearrangement-and-finite-difference-checks) for integral ADD, SUB,
and MUL:

```text
INT32_MAX + 1 -> INT32_MIN
INT32_MIN - 1 -> INT32_MAX
INT64_MAX * 2 -> wrap modulo 2^64 and reinterpret as signed INT64
```

This means:

- no construction-time or execution-time overflow exception is part of the [operation](../../../../glossary.md#operation) semantics;
- no [saturation](../../../../glossary.md#term-saturation) or [widening](../../../../glossary.md#term-widening) beyond the promoted result type occurs;
- [INT32](../../../../glossary.md#data-type--datatype) arithmetic wraps modulo `2^32` and [INT64](../../../../glossary.md#data-type--datatype) modulo `2^64`; and
- later [backends](../../../../glossary.md#backend) must reproduce the corresponding low-width [two's-complement](../../../../glossary.md#arithmetic-rearrangement-and-finite-difference-checks) result.

Integral MIN/MAX and all six comparisons use ordinary signed numerical ordering after promotion.
There is no [NaN](../../../../glossary.md#term-nan), [infinity](../../../../glossary.md#term-infinity), [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero), [tolerance](../../../../glossary.md#term-tolerance), or approximate-equality policy in the integral
domain. EQUAL and NOT_EQUAL compare exact promoted signed values.

This task records those semantics in model Javadocs and [API](../../../../glossary.md#term-application-programming-interface-api) documentation but does not calculate a
result.

## Result [metadata](../../../../glossary.md#term-metadata)

### [Tensor](../../../../glossary.md#tensor)-to-Tensor arithmetic

- Result [data type](../../../../glossary.md#data-type--datatype) is `promoteNumeric(leftType, rightType)`.
- Floating pairs retain the current floating behavior.
- Integral pairs use the [matrix](../../../../glossary.md#term-matrix-meanings) above.
- Result [Shape](../../../../glossary.md#shape) is the existing right-aligned `ShapeBroadcast` result.
- [Layout](../../../../glossary.md#layout) remains unresolved.
- `requiresGrad` remains the logical OR for floating pairs and is necessarily false for integral
  pairs because integral descriptors cannot request gradients.<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [gradients glossary](../../../../glossary.md#term-gradient))</span>
- Ordered [provenance](../../../../glossary.md#provenance) remains `[left, right]`.

### [Tensor](../../../../glossary.md#tensor)-to-[scalar](../../../../glossary.md#term-scalar) arithmetic

- Receiver must be floating or integral.
- The exact `ScalarValue.dataType()` must equal the receiver [data type](../../../../glossary.md#data-type--datatype).
- Floating receivers retain all seven existing scalar arithmetic kinds.
- Integral receivers accept only ADD, SUB, MUL, MIN, and MAX.
- Result retains the exact receiver DataType and Shape reference, unresolved layout, and exact
  existing <span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype); [Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout))</span>`requiresGrad` value, which is false for integral input.
- [Provenance](../../../../glossary.md#provenance) contains exactly the receiver; the scalar remains exact [operation](../../../../glossary.md#operation) attributes.

### Comparisons

- Both operands must form a valid same-category [numeric promotion](../../../../glossary.md#numeric-promotion) pair.
- Result [data type](../../../../glossary.md#data-type--datatype) is always [BOOL](../../../../glossary.md#data-type--datatype).
- Result [Shape](../../../../glossary.md#shape) is the existing right-aligned broadcast result.
- [Layout](../../../../glossary.md#layout) is unresolved and `requiresGrad=false`.
- Ordered [provenance](../../../../glossary.md#provenance) remains `[left, right]`.

## [Scope](../../../../glossary.md#term-scope)

- Add exact `DataTypePromotion.promoteNumeric` with floating, integral, BOOL, mixed-category, null,
  symmetry, and idempotence contracts.
- Preserve `promoteFloating` unchanged.
- Make `TensorBinaryExpressions` use [numeric promotion](../../../../glossary.md#numeric-promotion) and accept the five selected integral kinds.
- Keep floating support for all seven [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) arithmetic kinds unchanged.
- Reject integral DIV and POW before [Shape](../../../../glossary.md#shape) broadcast and result [identity](../../../../glossary.md#term-identity-meanings) allocation.
- Make `TensorComparisonExpressions` validate through numeric promotion and accept all integral
  pairs for all six existing comparison kinds.
- Preserve comparison [BOOL](../../../../glossary.md#data-type--datatype) result [metadata](../../../../glossary.md#term-metadata) and all floating behavior.
- Make `TensorScalarExpressions.applyScalar` accept floating or integral input, with exact
  scalar/input type equality and kind-aware integral restrictions.
- Preserve first-class `applyClamp` as floating-only.
- Let public [scalar](../../../../glossary.md#term-scalar) `clampMin`/`clampMax` inherit integral MAX/MIN convenience behavior.
- Update every affected [`Tensor`](../../../../glossary.md#tensor) Javadoc and its type-level summary without changing public method
  signatures or method count.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [signatures glossary](../../../../glossary.md#term-signature))</span>
- Add focused promotion, binary arithmetic, scalar arithmetic, comparison, validation-order,
  overflow-semantic metadata, broadcasting, provenance, and no-ID-on-failure tests.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [overflow glossary](../../../../glossary.md#term-overflow); [broadcasting glossary](../../../../glossary.md#broadcasting); [provenance glossary](../../../../glossary.md#provenance))</span>
- Update Tensor API, Compile API, glossary, capability baseline, this task, master plan, and roadmap
  through the mandatory independent documentation-focused pass.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- integral DIV, POW, remainder, modulo, floor division, truncating division, shifts, bitwise
  operations, absolute value, negation, range CLAMP, or another arithmetic kind<span data-glossary-navigation="true"> ([absolute value glossary](../../../../glossary.md#norms-elementary-functions-and-probability-distributions))</span>
- integral `double` convenience conversion; existing double overloads remain exact FLOAT64 and
  therefore mismatch integral receivers<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64))</span>
- generic `Number`, boxed values, caller-selected conversion, scalar promotion, eager scalar Tensor
  creation, or visible cast insertion<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [eager glossary](../../../../glossary.md#term-eager-evaluation); [Tensor glossary](../../../../glossary.md#tensor))</span>
- floating/integral mixed-category promotion, automatic integral-to-floating conversion, BOOL
  arithmetic/comparison, unsigned types, smaller integer widths, big integers, quantization, or
  saturation<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [saturation glossary](../../../../glossary.md#term-saturation))</span>
- numeric or statistical reductions, [`mean`](../../../../glossary.md#term-mean), accumulation types, empty-domain behavior, `argMin`,
  or changes to `argMax`; task 0018U1 owns selected integral reductions and arg-min [normalization](../../../../glossary.md#term-normalization)
- changing operation-kind enums, signatures, attributes, public method names, or Tensor method
  count<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [signatures glossary](../../../../glossary.md#term-signature))</span>
- numerical evaluation, storage reads/writes, overflow checks, constant folding, simplification,
  canonicalization, common-subexpression elimination, or result interning<span data-glossary-navigation="true"> ([evaluation glossary](../../../../glossary.md#term-evaluation); [overflow glossary](../../../../glossary.md#term-overflow); [constant folding glossary](../../../../glossary.md#term-constant-folding); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [common-subexpression elimination glossary](../../../../glossary.md#term-common-subexpression-elimination-cse))</span>
- [gradient](../../../../glossary.md#term-gradient) formulas, [autograd](../../../../glossary.md#autograd), [backward graph](../../../../glossary.md#backward-graph) construction, [optimizer](../../../../glossary.md#term-optimizer), or training behavior
- compiler capture, graph-wide validation, planning, prepare, backend support, lowering, kernel
  selection, runtime, execution, tracing, or engine behavior<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [graph glossary](../../../../glossary.md#graph); [validation glossary](../../../../glossary.md#term-validation); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [kernel glossary](../../../../glossary.md#kernel); [runtime glossary](../../../../glossary.md#term-runtime); [engine glossary](../../../../glossary.md#term-engine))</span>
- changes to DataType constants/categories, ScalarValue representation, Shape/Broadcast,
  TensorDescriptor, producer/provenance, factory, storage, or graph model<span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype); [constants glossary](../../../../glossary.md#term-constant); [ScalarValue glossary](../../../../glossary.md#typed-scalar-value--scalarvalue); [Shape glossary](../../../../glossary.md#shape); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance); [model glossary](../../../../glossary.md#term-model))</span>
- dependencies, Gradle, Java version, preview/incubator features, architecture rules, focused
  architecture documents/tests, another module, or a detailed 0018U1-or-later task specification<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [features glossary](../../../../glossary.md#term-software-feature); [module glossary](../../../../glossary.md#term-module))</span>

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>, especially model ownership of public
  Tensor, DataType, and backend-independent operation semantics<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Tensor glossary](../../../../glossary.md#tensor); [DataType glossary](../../../../glossary.md#data-type--datatype); [backend glossary](../../../../glossary.md#backend); [operation glossary](../../../../glossary.md#operation))</span>
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)
- [Model master plan](../master-plan.md)
- [Task 0001](0001-data-type-model.md)
- [Task 0014A](0014a-binary-arithmetic-semantic-kinds.md)
- [Task 0014B](0014b-binary-arithmetic-tensor-expressions.md)
- [Task 0014E](0014e-scalar-arithmetic-and-clamp-semantics.md)
- [Task 0014F](0014f-scalar-arithmetic-and-clamp-tensor-expressions.md)
- [Task 0015A](0015a-binary-comparison-semantic-kinds.md)
- [Task 0015B](0015b-binary-comparison-tensor-expressions.md)
- [Task 0018K](0018k-operation-signature-and-construction-hardening.md)
- [Task 0018N](0018n-typed-scalar-value-contract.md)
- [Task 0018T](0018t-scalar-arithmetic-family-normalization.md)
- [Task 0018T1](0018t1-unary-numeric-gaps-and-floating-diagnostics.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `modules/model` owns data-type promotion, public expression validation, backend-independent
  arithmetic/comparison semantics, descriptor derivation, and producer/provenance construction.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [backend glossary](../../../../glossary.md#backend); [descriptor glossary](../../../../glossary.md#term-descriptor); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>
- Promotion is a semantic data-type rule, not an inserted Tensor, graph node, compiler pass, or
  backend route.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [node glossary](../../../../glossary.md#node); [compiler glossary](../../../../glossary.md#term-compiler); [backend route glossary](../../../../glossary.md#backend-route))</span>
- [`Operation`](../../../../glossary.md#operation), kind enums, and signatures remain operand-independent. They receive no domain flag,
  overflow mode, promotion table, backend capability, or result descriptor.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [overflow glossary](../../../../glossary.md#term-overflow); [backend capability glossary](../../../../glossary.md#backend-capability))</span>
- Public/package-private Tensor construction owns operand-aware domain validation because it has
  descriptors and ordered Tensor inputs.
- Scalar attributes remain exact typed model values rather than Tensor inputs. No implicit scalar
  conversion is added.<span data-glossary-navigation="true"> ([Scalar glossary](../../../../glossary.md#term-scalar); [model glossary](../../../../glossary.md#term-model))</span>
- [`Tensor`](../../../../glossary.md#tensor) remains mutable public API state rather than IR. Every valid call creates one fresh
  storage-free result with one immutable producer and provenance output index zero.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- Compiler owns graph capture, operand revalidation, canonicalization, gradients, and backward graph
  construction. Backend prepare owns lowering, route selection, specialization, and fusion.<span data-glossary-navigation="true"> ([graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [gradients glossary](../../../../glossary.md#term-gradient); [backward graph glossary](../../../../glossary.md#backward-graph); [prepare glossary](../../../../glossary.md#prepare); [lowering glossary](../../../../glossary.md#lowering); [specialization glossary](../../../../glossary.md#term-specialization); [fusion glossary](../../../../glossary.md#term-fusion))</span>
- [Runtime](../../../../glossary.md#term-runtime) hot paths do not consume Operation values.
- No registry, service locator, reflection, classpath discovery, module/package dependency, or
  architecture rule is added.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [service locator glossary](../../../../glossary.md#term-service-locator); [classpath glossary](../../../../glossary.md#term-classpath); [module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- If implementation needs a new operation kind, public method, promotion object, cast producer,
  attributes type, reduction change, another module, or a seventeenth path, stop and report the
  conflict.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.datatype` — owns public [numeric promotion](../../../../glossary.md#numeric-promotion) rules.
- `io.github.pho001.synaptik.model.tensor` — owns public methods and three existing cohesive
  package-private expression-construction boundaries.
- Existing operation-family packages remain review-only because their semantic identities and
  structural signatures do not change.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [signatures glossary](../../../../glossary.md#term-signature))</span>

Packages added, moved, or removed:

- None.

Type placement:

- `DataTypePromotion` remains the single stateless promotion contract rather than creating a second
  integral-specific utility or a promotion-result object.
- `TensorBinaryExpressions`, `TensorScalarExpressions`, and `TensorComparisonExpressions` retain
  their existing packages and method surfaces because only their accepted data-type domains change.
- [`Tensor`](../../../../glossary.md#tensor) remains the public fluent [facade](../../../../glossary.md#term-facade); no new method or overload is required.

## Required implementation contracts

### `DataTypePromotion.promoteNumeric`

The utility remains public, final, field-free, non-instantiable, and gains exactly one public static
method. No other public or package-private method is added.

[Validation](../../../../glossary.md#term-validation) occurs in this exact order:

1. null-check `left` with message `left`;
2. null-check `right` with message `right`;
3. require left to be floating or integral; otherwise throw `IllegalArgumentException` with exact
   message `left must be a numeric data type, but was BOOL`;
4. require right to be floating or integral; otherwise throw the analogous `right` message;
5. require both operands to share the same category; otherwise throw `IllegalArgumentException`
   with exact message
   `numeric data types must share a category, but were <left> and <right>`;
6. return the existing widest floating result or selected widest integral result.

The method is symmetric and [idempotent](../../../../glossary.md#term-idempotent) for every valid pair. `promoteFloating` retains its current
implementation contract, validation messages, and accepted [matrix](../../../../glossary.md#term-matrix-meanings).

### [Binary](../../../../glossary.md#term-binary-descriptor-and-metadata) arithmetic helper

Keep `TensorBinaryExpressions` field-free with its exact existing constructor and one `apply`
method. Its order becomes:

1. null-check left, right, then kind exactly as today;
2. call `DataTypePromotion.promoteNumeric` exactly once;
3. if the promoted type is integral, accept ADD, SUB, MUL, MIN, or MAX and reject DIV or POW with
   exact message `<KIND> does not support integral data types`;
4. broadcast [Shapes](../../../../glossary.md#shape) exactly once;
5. construct the current result [descriptor](../../../../glossary.md#term-descriptor), exact [operation](../../../../glossary.md#operation), and ordered [provenance](../../../../glossary.md#provenance);
6. delegate exactly once to `TensorFactory.createDerived`.

An unsupported integral kind fails before Shape broadcast and result-ID allocation. Floating
[validation](../../../../glossary.md#term-validation), promotion, [broadcasting](../../../../glossary.md#broadcasting), result [metadata](../../../../glossary.md#term-metadata), and errors remain unchanged except that [BOOL](../../../../glossary.md#data-type--datatype)
and cross-category failures now come from the new numeric contract.

### [Scalar](../../../../glossary.md#term-scalar) arithmetic helper

Keep `TensorScalarExpressions` at exactly its existing three-method surface. `applyClamp` and
private `create` retain their current behavior.

`applyScalar` order becomes:

1. null-check input, kind, then value exactly as today;
2. reject CLAMP with exact existing message `CLAMP requires ClampRangeAttrs`;
3. read the input [data type](../../../../glossary.md#data-type--datatype) and require floating or integral, otherwise throw exact message
   `input must be a numeric data type, but was BOOL`;
4. for integral input accept ADD, SUB, MUL, MIN, or MAX; reject DIV or POW with exact message
   `<KIND> does not support integral data types`;
5. require exact scalar/input type equality with the existing message
   `scalar data type <scalar> must match input data type <input>`;
6. construct exact attributes, [operation](../../../../glossary.md#operation), [descriptor](../../../../glossary.md#term-descriptor), and one-input [producer](../../../../glossary.md#term-producer) as today.

Integral `clampMin` and `clampMax` work only because the existing public conveniences delegate to
MAX and MIN. `applyClamp` continues to reject integral input with its existing floating-input
message before range construction.

### Comparison helper

Keep `TensorComparisonExpressions` field-free with its exact existing one-method surface.

After null checks it calls `promoteNumeric` exactly once to validate a same-category numeric pair,
ignores the promoted type because the result is [BOOL](../../../../glossary.md#data-type--datatype), then preserves the existing broadcast,
[descriptor](../../../../glossary.md#term-descriptor), [operation](../../../../glossary.md#operation), ordered [provenance](../../../../glossary.md#provenance), and factory order.

### Public [Tensor](../../../../glossary.md#tensor) contract

No [signature](../../../../glossary.md#term-signature) or delegation statement changes. Complete Javadocs must explain:

- ADD/SUB/MUL/MIN/MAX accept floating or integral Tensor pairs;
- DIV/POW remain floating-only;
- all six comparisons accept floating or integral pairs;
- scalar ADD/SUB/MUL/MIN/MAX and one-bound clamp conveniences accept exact matching floating or
  integral [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue);
- [scalar](../../../../glossary.md#term-scalar) DIV/POW and first-class CLAMP remain floating-only;
- mixed numeric categories require explicit cast; and
- integral results are non-differentiable, unresolved-layout metadata with modular arithmetic or
  signed-order meaning as applicable.<span data-glossary-navigation="true"> ([differentiable glossary](../../../../glossary.md#derivative-and-chain-rule); [layout glossary](../../../../glossary.md#layout); [metadata glossary](../../../../glossary.md#term-metadata); [modular arithmetic glossary](../../../../glossary.md#term-modular-arithmetic))</span>

The public Tensor method count remains exactly 127.

## Affected files

Expected production Java:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/datatype/DataTypePromotion.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorBinaryExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorComparisonExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorScalarExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Expected tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/datatype/DataTypePromotionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryArithmeticTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryComparisonTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorScalarElementwiseTest.java`

Expected documentation and [planning](../../../../glossary.md#term-planning):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- `docs/planning/modules/model/tasks/0018u-integral-elementwise-arithmetic-and-comparisons.md`
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review-only unless an out-of-[scope](../../../../glossary.md#term-scope) discrepancy requires stopping:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/datatype/DataType.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/datatype/DataTypeCategory.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/datatype/ScalarValue.java`
- [operation](../../../../glossary.md#operation) kind/attributes/[signature](../../../../glossary.md#term-signature) contracts and their tests
- reduction and arg-max contracts/helpers/tests
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `docs/api/training-api.md`
- focused architecture documentation and architecture tests

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify exactly the 16 affected paths listed above: five production Java
files, four test files, and seven documentation/[planning](../../../../glossary.md#term-planning) files including this specification.

If implementation needs a seventeenth path, a new production/test type, another helper method, an
[operation](../../../../glossary.md#operation)-kind edit, a reduction change, or a review-only file, stop and report why before editing
it. Do not omit a required focused test or documentation path merely to stay under the cap.

## Acceptance criteria

- `promoteNumeric` implements the exact floating and integral matrices, null/type/category errors,
  symmetry, and idempotence while <span data-glossary-navigation="true"> ([matrices glossary](../../../../glossary.md#term-matrix-meanings))</span>`promoteFloating` remains unchanged.
- Binary ADD/SUB/MUL/MIN/MAX accept every INT32/INT64 ordered pair and derive the exact promoted
  result type and broadcast Shape.<span data-glossary-navigation="true"> ([Binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [Shape glossary](../../../../glossary.md#shape))</span>
- Binary integral DIV/POW fail with exact kind-specific messages before Shape broadcast and ID
  allocation; their floating behavior remains green.
- [Scalar](../../../../glossary.md#term-scalar) ADD/SUB/MUL/MIN/MAX accept exact matching [INT32](../../../../glossary.md#data-type--datatype)/[INT64](../../../../glossary.md#data-type--datatype) [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue); scalar DIV/POW reject
  integral receivers before scalar mismatch, and BOOL is rejected as non-numeric.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- Integral one-bound clamp conveniences create scalar MAX/MIN producers; first-class range CLAMP
  remains floating-only.<span data-glossary-navigation="true"> ([producers glossary](../../../../glossary.md#term-producer))</span>
- All six comparisons accept every INT32/INT64 pair, produce BOOL/false-gradient unresolved
  descriptors, and preserve broadcasting and ordered provenance.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [descriptors glossary](../../../../glossary.md#term-descriptor); [broadcasting glossary](../../../../glossary.md#broadcasting); [provenance glossary](../../../../glossary.md#provenance))</span>
- Floating behavior for all affected operations remains unchanged, and mixed floating/integral or
  BOOL pairs fail before Shape work and ID allocation.
- Tests and Javadocs record modular two's-complement overflow semantics and signed integral
  extrema/comparison ordering without evaluating values in model.<span data-glossary-navigation="true"> ([two's-complement glossary](../../../../glossary.md#arithmetic-rearrangement-and-finite-difference-checks); [overflow glossary](../../../../glossary.md#term-overflow); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [model glossary](../../../../glossary.md#term-model))</span>
- Every successful expression is fresh, unlabeled, storage-free, one-output, and has exact kind,
  attributes, ordered inputs, and provenance output index zero.
- Public Tensor method count and signatures remain unchanged; no alias, overload, visible cast, or
  new operation kind appears.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [signatures glossary](../../../../glossary.md#term-signature); [alias glossary](../../../../glossary.md#aliasing-and-arena); [operation glossary](../../../../glossary.md#operation))</span>
- Tensor API, Compile API, glossary, capability baseline, task, master plan, and roadmap are
  synchronized without rewriting completed task history.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile))</span>
- Training API, architecture contract/docs/tests, Gradle, dependencies, other modules, backend
  conformance, and integration tests remain unchanged with reasoned conclusions.<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- One final `:modules:model:test` [run](../../../../glossary.md#term-run) passes after executable Java stabilizes.
- A separate clean-context documentation-focused pass finalizes Javadocs and documentation in the
  same overall change without repeating successful Java tests unless executable behavior changes
  or a concrete risk is recorded.
- Model Javadoc, a runnable Java 26 integral-metadata example, Markdown links/anchors/fences/final
  newlines, exact 16-path scope, status synchronization, terminology, and <span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [metadata glossary](../../../../glossary.md#term-metadata); [scope glossary](../../../../glossary.md#term-scope); [synchronization glossary](../../../../glossary.md#term-synchronization))</span>`git diff --check` pass.
- Task 0018U is Complete only after both passes succeed. Task 0018U1 and every later task remain
  Draft without a detailed specification.

## Tests / [validation](../../../../glossary.md#term-validation)

Focused tests while executable Java stabilizes:

```bash
./gradlew :modules:model:test --tests \
  io.github.pho001.synaptik.model.datatype.DataTypePromotionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorBinaryArithmeticTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorBinaryComparisonTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorScalarElementwiseTest
```

Final Java [checkpoint](../../../../glossary.md#term-checkpoint) after executable code stabilizes:

```bash
./gradlew :modules:model:test
```

The implementation pass records the exact evidence and hands it to the documentation pass. Do not
repeat a successful final [model](../../../../glossary.md#term-model) suite unless executable Java changes afterward or a concrete risk
is recorded.

Documentation-focused pass:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

The documentation pass must also:

- compile and run one Java 26 metadata example covering INT32/INT64 promotion, exact scalar
  attributes, BOOL comparison result, ordered provenance, and output index zero without claiming
  evaluated values;<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [run glossary](../../../../glossary.md#term-run); [metadata glossary](../../../../glossary.md#term-metadata); [INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [scalar glossary](../../../../glossary.md#term-scalar); [BOOL glossary](../../../../glossary.md#data-type--datatype); [provenance glossary](../../../../glossary.md#provenance))</span>
- validate local Markdown links and anchors in changed documents;
- inspect generated [Javadoc](../../../../glossary.md#term-javadoc) for `DataTypePromotion`, all affected helpers, and [Tensor](../../../../glossary.md#tensor) methods;
- verify the exact promotion matrices, helper method surfaces, unchanged 127-method public Tensor
  surface, unsupported-kind failures, and absence of aliases/new kinds;<span data-glossary-navigation="true"> ([matrices glossary](../../../../glossary.md#term-matrix-meanings); [aliases glossary](../../../../glossary.md#aliasing-and-arena))</span>
- verify exact 16-path [scope](../../../../glossary.md#term-scope) and no [dependency](../../../../glossary.md#term-dependency)/[Gradle](../../../../glossary.md#term-gradle)/architecture/other-[module](../../../../glossary.md#term-module) changes;
- verify synchronized 0018U Complete status and no detailed 0018U1-or-later specification; and
- review terminology, examples, fences, final newlines, trailing whitespace, authority boundaries,
  and the final diff.

Repository-wide validation is deferred to the capability-reset checkpoint after task 0018V and
[CI](../../../../glossary.md#term-continuous-integration-ci). This task changes one module without changing dependencies, architecture, or shared build
configuration.

## Dependencies

- [Task 0001](0001-data-type-model.md) — current [DataType](../../../../glossary.md#data-type--datatype) categories and floating promotion.
- [Task 0014A](0014a-binary-arithmetic-semantic-kinds.md) and
  [0014B](0014b-binary-arithmetic-tensor-expressions.md) — [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) kinds and construction.
- [Task 0014E](0014e-scalar-arithmetic-and-clamp-semantics.md) and
  [0014F](0014f-scalar-arithmetic-and-clamp-tensor-expressions.md) — scalar kinds, exact attributes,
  and construction.
- [Task 0015A](0015a-binary-comparison-semantic-kinds.md) and
  [0015B](0015b-binary-comparison-tensor-expressions.md) — comparison kinds and construction.
- [Task 0018K](0018k-operation-signature-and-construction-hardening.md) — exact structural family
  validation.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- [Task 0018N](0018n-typed-scalar-value-contract.md) — exact [INT32](../../../../glossary.md#data-type--datatype)/[INT64](../../../../glossary.md#data-type--datatype) [scalar](../../../../glossary.md#term-scalar) representation.
- [Task 0018T](0018t-scalar-arithmetic-family-normalization.md) — final parallel arithmetic
  vocabulary and extrema naming.<span data-glossary-navigation="true"> ([extrema glossary](../../../../glossary.md#saturation-pole-and-extremum))</span>
- [Task 0018T1](0018t1-unary-numeric-gaps-and-floating-diagnostics.md) — completed floating unary
  foundation preceding this frontier.<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary))</span>

All dependencies are Complete.

## Follow-up tasks

- **0018U1 Integral reductions and arg-min [normalization](../../../../glossary.md#term-normalization)** — add selected integral SUM/PROD/MIN/
  MAX domains plus `argMin` and normalize the shared arg-[extrema](../../../../glossary.md#saturation-pole-and-extremum) tie-policy contract.
- **0018V Multi-[axis](../../../../glossary.md#term-axis) and statistical reductions** — add ordered axes and floating statistical/norm
  semantics after integral reduction policy is stable.<span data-glossary-navigation="true"> ([norm glossary](../../../../glossary.md#term-norm))</span>
- Later compiler and backend tasks own graph revalidation, gradients, lowering, modular execution,
  numerical conformance, and kernels.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [graph glossary](../../../../glossary.md#graph); [gradients glossary](../../../../glossary.md#term-gradient); [lowering glossary](../../../../glossary.md#lowering); [conformance glossary](../../../../glossary.md#term-conformance); [kernels glossary](../../../../glossary.md#kernel))</span>

## Architecture impact

Expected impact: None.

This extends [backend](../../../../glossary.md#backend)-independent [model](../../../../glossary.md#term-model) promotion and public expression-construction domains within
existing [module](../../../../glossary.md#term-module)/package [ownership](../../../../glossary.md#term-ownership-meanings). It changes no [operation](../../../../glossary.md#operation) structure, module [dependency](../../../../glossary.md#term-dependency), [lifecycle](../../../../glossary.md#lifecycle),
architecture rule, backend contract, [runtime](../../../../glossary.md#term-runtime) path, or [graph](../../../../glossary.md#graph) representation.

If implementation requires an architecture change, stop and report it before editing
`ARCHITECTURE.md`, focused architecture documentation, ADRs, or architecture tests.

## Implementation prompt

Use this prompt in a separate agentic task/thread with clean context:

```text
You are working in the Synaptik repository.

Read AGENTS.md, ARCHITECTURE.md, documentation/planning rules, roadmap, model capabilities/master
plan, tasks 0001/0014A/0014B/0014E/0014F/0015A/0015B/0018K/0018N/0018T/0018T1/0018U,
Tensor API, Compile API, Training API, glossary, and every affected or review-only source/test
named by task 0018U in full.

Implement task 0018U exactly. Add same-category numeric promotion, selected INT32/INT64
ADD/SUB/MUL/MIN/MAX Tensor and exact-scalar domains, and all six integral comparisons. Preserve all
floating behavior, reject mixed categories/BOOL and integral DIV/POW/range CLAMP, retain exact
validation/descriptor/producer/provenance order, and add no public method or operation kind. Stay
within the exact 16 paths, stop on scope or architecture conflict, and do not commit or push.

Run focused tests as needed and one final model test after executable Java stabilizes. Then hand
the actual diff and exact evidence to a separate clean-context documentation-focused agent in the
same overall change. That agent must inspect final source/tests, finalize permitted Javadocs,
Tensor/Compile API, glossary, capability/task/master/roadmap documentation, run model Javadoc and
documentation/scope checks, and must not repeat successful Java tests unless executable behavior
changes or it records a concrete reason.

Mark 0018U Complete only after both passes succeed. Leave 0018U1 and every later task Draft
without a detailed specification.
```

## Local decisions

- `DataTypePromotion.promoteNumeric` delegates valid floating pairs to the unchanged
  `promoteFloating` contract and selects INT64 for either mixed-width integral ordering. No cast
  producer or promotion object is stored.<span data-glossary-navigation="true"> ([INT64 glossary](../../../../glossary.md#data-type--datatype); [producer glossary](../../../../glossary.md#term-producer))</span>
- Binary and scalar helpers reject integral DIV/POW after numeric-domain validation but before
  broadcasting or result construction. Exact scalar type equality remains after kind eligibility,
  preserving the specified validation order.<span data-glossary-navigation="true"> ([Binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [scalar glossary](../../../../glossary.md#term-scalar); [validation glossary](../../../../glossary.md#term-validation); [broadcasting glossary](../../../../glossary.md#broadcasting))</span>
- Existing `clampMin` and `clampMax` delegation supplies integral one-bound behavior through scalar
  MAX/MIN. First-class two-bound CLAMP remains on its unchanged floating-only helper path.
- No method, operation kind, attributes type, package, dependency, or architecture boundary was
  added.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Known limitations

- Integral DIV, POW, range CLAMP, remainder/modulo, reductions, and arg-min remain unsupported.
  Task 0018U1 owns the selected integral reduction and arg-min policy frontier.
- Model construction records fixed-width modular arithmetic and signed ordering semantics without
  reading or evaluating values. Compiler capture/revalidation, gradients, lowering, backend
  support, kernels, and execution remain unimplemented or separately owned.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [fixed-width modular arithmetic glossary](../../../../glossary.md#arithmetic-rearrangement-and-finite-difference-checks); [Compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [gradients glossary](../../../../glossary.md#term-gradient); [lowering glossary](../../../../glossary.md#lowering); [backend glossary](../../../../glossary.md#backend); [kernels glossary](../../../../glossary.md#kernel))</span>
- Result layout remains unresolved and every successful expression remains a fresh, unlabeled,
  storage-free producer result with provenance output index zero.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation context `/root/task_0018u_implementation` ran the focused command from this task
  after final executable edits. It passed with `DataTypePromotionTest` 6 tests,
  `TensorBinaryArithmeticTest` 12, `TensorBinaryComparisonTest` 10, and
  `TensorScalarElementwiseTest` 8; all 36 tests had zero failures, errors, or skips.
- The same implementation context then ran `./gradlew :modules:model:test`. It passed with 90 XML
  suites and 734 tests, zero failures, errors, or skips. The documentation context changed only
  Javadocs and documentation afterward and therefore did not repeat this successful Java suite.
- Documentation context `/root/task_0018u_implementation/task_0018u_docs` applied the General,
  API/Javadoc, Planning, and Example profiles. It independently reviewed the final production and
  test diff, finalized all permitted Javadocs and seven documentation/planning paths, and reviewed
  the glossary impact.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Planning glossary](../../../../glossary.md#term-planning))</span>
- The documentation context ran `./gradlew :modules:model:javadoc` after final [Javadoc](../../../../glossary.md#term-javadoc) edits and
  `git diff --check`; both passed. Generated pages for public `DataTypePromotion` and the affected
  [`Tensor`](../../../../glossary.md#tensor) methods were inspected. The default public Javadoc output omits the three package-private
  helpers, so their complete source Javadocs were inspected directly instead.
- A Java 26 metadata example compiled and ran against the final model classes. It verified
  INT32/INT64 promotion to INT64, exact retained INT32 scalar attributes, a BOOL comparison result,
  ordered provenance, output index zero, unresolved layout, and absent storage without evaluating
  arithmetic or comparison values.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [model glossary](../../../../glossary.md#term-model); [INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [scalar glossary](../../../../glossary.md#term-scalar); [BOOL glossary](../../../../glossary.md#data-type--datatype); [provenance glossary](../../../../glossary.md#provenance); [layout glossary](../../../../glossary.md#layout))</span>
- Targeted structural checks confirmed both promotion matrices, exact helper method surfaces,
  exactly 127 public declared <span data-glossary-navigation="true"> ([matrices glossary](../../../../glossary.md#term-matrix-meanings))</span>`Tensor` methods, unchanged seven binary kinds/eight scalar kinds/six
  comparison kinds, integral DIV/POW validation messages and pre-broadcast failure order, no
  aliases or new kinds, valid changed-document links/anchors/fences/final newlines, and the exact
  16-path scope.<span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [aliases glossary](../../../../glossary.md#aliasing-and-arena); [scope glossary](../../../../glossary.md#term-scope))</span>
- Training API required no change because task 0018U adds no training, gradient, optimizer, or
  publication contract. Architecture and focused architecture documents/tests required no change
  because ownership and dependency rules are unchanged. Operation, reduction, and scan contracts
  required no change because no kind, attributes, signature, or reduction domain changed. Gradle,
  dependencies, other modules, backend conformance, and integration tests required no change
  because this is model-only metadata construction with no backend or end-to-end execution.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [optimizer glossary](../../../../glossary.md#term-optimizer); [publication glossary](../../../../glossary.md#term-publication); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [dependency glossary](../../../../glossary.md#term-dependency); [Operation glossary](../../../../glossary.md#operation); [scan glossary](../../../../glossary.md#term-cumulative-scan); [signature glossary](../../../../glossary.md#term-signature); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Repository-wide validation remains deferred to the recorded capability-reset checkpoint after
  task 0018V and CI, as planned for this single-module change.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [CI glossary](../../../../glossary.md#term-continuous-integration-ci))</span>

## Implementation notes

- `promoteNumeric` performs exact left/right null, numeric, and same-category validation, preserves
  floating promotion, and implements the complete INT32/INT64 matrix symmetrically and
  idempotently.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- Binary ADD/SUB/MUL/MIN/MAX and all six comparisons now accept every ordered INT32/INT64 pair.
  Binary and scalar integral DIV/POW retain exact failure messages and allocate no result identity.<span data-glossary-navigation="true"> ([Binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [scalar glossary](../../../../glossary.md#term-scalar); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Exact scalar INT32/INT64 ADD/SUB/MUL/MIN/MAX and delegated one-bound clamp conveniences preserve
  exact attributes and one-input provenance; range CLAMP remains floating-only.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>
- Focused tests cover promotion laws and failures, selected integral domains, broadcasting,
  modular semantic requests, signed comparison domains, exact scalar attributes, provenance,
  freshness, storage absence, validation messages/order, and no-ID-on-failure behavior.<span data-glossary-navigation="true"> ([broadcasting glossary](../../../../glossary.md#broadcasting))</span>

## Completion summary

- Completed changes: added same-category numeric promotion and selected signed-integral binary,
  comparison, and exact-scalar expression domains while preserving floating behavior and all
  producer/provenance invariants.<span data-glossary-navigation="true"> ([numeric promotion glossary](../../../../glossary.md#numeric-promotion); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [scalar glossary](../../../../glossary.md#term-scalar); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>
- Files changed or created: exactly the five production Java, four test, and seven
  documentation/planning paths listed under Affected files.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Tests and validation: focused 36-test command and final 734-test model suite passed in the
  implementation context; model Javadoc, Java 26 metadata example, structural/API checks,
  Markdown validation, exact-scope validation, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [model glossary](../../../../glossary.md#term-model); [metadata glossary](../../../../glossary.md#term-metadata); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>`git diff --check` passed in the documentation
  context.
- Documentation-agent review: clean context
  `/root/task_0018u_implementation/task_0018u_docs` completed the independent targeted pass.
- Documentation impact: Tensor API, Compile API, capability baseline, task, master plan, and
  roadmap now distinguish current integral elementwise metadata from planned compiler/backend and
  reduction work.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [elementwise glossary](../../../../glossary.md#term-elementwise); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend))</span>
- [Javadoc](../../../../glossary.md#term-javadoc) review: finalized `DataTypePromotion`, the three helpers, the [`Tensor`](../../../../glossary.md#tensor) type summary, and
  every affected public Tensor method without changing signatures or method count.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature))</span>
- Glossary impact: documented numeric promotion, modular integral arithmetic, signed ordering,
  exact scalar matching, and current-vs-planned boundaries.
- Unresolved issues: None within task [scope](../../../../glossary.md#term-scope).
- Follow-up required: None. Task 0018U1 remains the next Draft frontier without a detailed spec.

Status: Complete
