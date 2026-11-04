# Arithmetic, comparison, [Boolean](../../glossary.md#data-type--datatype), selection and cast kinds

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

## Read these as reusable meanings, not calls on data

A Tensor expression helper, or code constructing a compiled graph, calls `new Operation` with
a kind and immutable attributes. That constructor retains those exact objects; no left/right
Tensor or array crosses this boundary. A producer or compiled node later connects the operation
to operands. For example, the ADD description can be reused for different pairs of Tensors
without those pairs becoming the same occurrence.
Use [binary Tensor calls](arithmetic.md#binary-arithmetic-expressions) when starting from operands,
and [signature checks](operation-foundation.md) when starting from semantic records.

**[Reader model](../../glossary.md#term-reader-model).** These **kinds** are typed names for calculations, not functions that evaluate
data. **[Binary](../../glossary.md#term-binary-descriptor-and-metadata)** arithmetic consumes two logical inputs; **[elementwise](../../glossary.md#term-elementwise)** means corresponding
positions are combined. A comparison returns truth values, Boolean logic combines truth values,
`WHERE` chooses between branch values, and a cast requests type conversion.
**Attributes** hold fixed configuration such as a [target](../../glossary.md#term-target) [data type](../../glossary.md#data-type--datatype); input [shapes](../../glossary.md#shape) live in separate
[descriptors](../../glossary.md#term-descriptor). Read [operation signatures](operation-foundation.md)<span data-glossary-navigation="true"> ([operation glossary](../../glossary.md#operation))</span> for that separation and
[arithmetic expressions](arithmetic.md) or [selection and casts](selection-casts.md) for the
input-aware [Tensor](../../glossary.md#tensor) calls. Equal printed names in different families do not make kinds equal.

<a id="binary-arithmetic-semantic-kinds"></a>
### [Binary](../../glossary.md#term-binary-descriptor-and-metadata) arithmetic semantic kinds

The public enum
`io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind` implements
[`OperationKind`](../../glossary.md#operationkind) with exactly these [constants](../../glossary.md#term-constant) in declaration order:

| Kind | Ordered [elementwise](../../glossary.md#term-elementwise) meaning |
|---|---|
| `ADD` | Left value plus right value. |
| `SUB` | Left value minus right value. |
| `MUL` | Left value multiplied by right value. |
| `DIV` | Left value divided by right value. |
| `MIN` | Mathematical minimum of the left and right values. |
| `MAX` | Mathematical maximum of the left and right values. |
| `POW` | Left value as the base raised to the right value as the [exponent](../../glossary.md#term-exponent). |

The enum values are semantic [identities](../../glossary.md#term-identity-meanings) only. All seven have no intrinsic [parameters](../../glossary.md#term-parameter) and compose
explicitly with the existing generic [descriptor](../../glossary.md#term-descriptor):

```java
// Describe addition without operands; a producer or node will supply the two inputs.
Operation addition = new Operation(
        BinaryArithmeticKind.ADD,
        NoOperationAttrs.INSTANCE);
```

The result is an [immutable](../../glossary.md#term-immutable) [operation](../../glossary.md#operation) description whose kind is exactly
`BinaryArithmeticKind.ADD` and whose attributes value is exactly the [canonical](../../glossary.md#term-canonical-form) [singleton](../../glossary.md#term-singleton). It is
not a [Tensor](../../glossary.md#tensor) result, [provenance](../../glossary.md#provenance) value, [graph](../../glossary.md#graph) [occurrence](../../glossary.md#term-occurrence), executable operation, or [backend](../../glossary.md#backend)-support
claim. No family factory or implicit attributes assignment is provided.

Broadcast geometry is derived from operand [shapes](../../glossary.md#shape) rather than stored on the enum or in operation
attributes. The current public Tensor methods own local floating eligibility, promotion,
[broadcasting](../../glossary.md#broadcasting), result-descriptor construction, and ordered provenance. ADD, SUB, MUL, and DIV are
ordinary ordered [IEEE](../../glossary.md#term-ieee)-754 requests in the result type. MIN and MAX propagate [NaN](../../glossary.md#term-nan), order [infinities](../../glossary.md#nan-infinity-and-signed-zero)
normally, and choose negative or [positive zero](../../glossary.md#nan-infinity-and-signed-zero), respectively, from opposite signed zeros,
independent of operand order. Unequal non-NaN values use ordinary numeric order, while equal
nonzero candidates produce that numeric value without identifying a selected operand. [FLOAT64](../../glossary.md#term-float64) and
[FLOAT32](../../glossary.md#term-float32) use their represented IEEE-754 values; [BFLOAT16](../../glossary.md#term-bfloat16) uses the value represented by its exact
16-bit storage after existing promotion. Integral [extrema](../../glossary.md#saturation-pole-and-extremum) retain ordinary signed order after
promotion. No [NaN payload](../../glossary.md#nan-infinity-and-signed-zero), intermediate [precision](../../glossary.md#term-precision), exact instruction, bitwise result, [gradient](../../glossary.md#term-gradient)
rule, [compiler](../../glossary.md#term-compiler) [capture](../../glossary.md#term-capture-meanings), execution, or backend availability is promised here. The inherited enum
names and text are stable diagnostics, not [serialization](../../glossary.md#term-serialization)
[tokens](../../glossary.md#term-token), [registry](../../glossary.md#term-registry) keys, or string-[dispatch](../../glossary.md#term-dispatch) contracts. Equality remains typed: an `ADD` value
declared by another kind family is not equal to `BinaryArithmeticKind.ADD` even though both
diagnostics may contain `ADD`.

<a id="binary-comparison-semantic-kinds"></a>
### [Binary](../../glossary.md#term-binary-descriptor-and-metadata) comparison semantic kinds

The public enum
`io.github.pho001.synaptik.model.operation.elementwise.comparison.BinaryComparisonKind` implements
[`OperationKind`](../../glossary.md#operationkind) with exactly these [constants](../../glossary.md#term-constant) in declaration order:

| Kind | Ordered [elementwise](../../glossary.md#term-elementwise) meaning |
|---|---|
| `GREATER_THAN` | The left value is strictly greater than the right value; false for [NaN](../../glossary.md#term-nan) or either [signed-zero](../../glossary.md#nan-infinity-and-signed-zero) order. |
| `GREATER_OR_EQUAL` | The left value is greater than or equal to the right value; false for NaN and true for either signed-zero order. |
| `LESS_THAN` | The left value is strictly less than the right value; false for NaN or either signed-zero order. |
| `LESS_OR_EQUAL` | The left value is less than or equal to the right value; false for NaN and true for either signed-zero order. |
| `EQUAL` | Exact represented numeric equality; NaN is unequal to itself and opposite signed zeros are equal. |
| `NOT_EQUAL` | The logical complement of `EQUAL`; true for NaN and false for opposite signed zeros. |

All six kinds are parameterless semantic [identities](../../glossary.md#term-identity-meanings). They compose explicitly with the generic
[operation](../../glossary.md#operation) [descriptor](../../glossary.md#term-descriptor):

```java
// Comparison's result will be Boolean; this descriptor alone has no values to compare.
Operation comparison = new Operation(
        BinaryComparisonKind.GREATER_THAN,
        NoOperationAttrs.INSTANCE);
```

The resulting operation retains the exact `GREATER_THAN` kind and [canonical](../../glossary.md#term-canonical-form) no-attributes value.
It describes ordered comparison meaning only; it stores no operands or broadcast plan and is not
a [Tensor](../../glossary.md#tensor) result, [provenance](../../glossary.md#provenance) value, [graph](../../glossary.md#graph) [occurrence](../../glossary.md#term-occurrence), or executable operation. Construction
enforces the exact [`NoOperationAttrs`](../../glossary.md#nooperationattrs) pairing; the family [signature](../../glossary.md#term-signature) declares two inputs and one
output.

The current public comparison methods separately own floating-pair [validation](../../glossary.md#term-validation), right-aligned local
[broadcasting](../../glossary.md#broadcasting), fixed [`BOOL`](../../glossary.md#data-type--datatype) result derivation with false `requiresGrad`, and ordered provenance
construction. [FLOAT64](../../glossary.md#term-float64) and [FLOAT32](../../glossary.md#term-float32) compare their represented [IEEE](../../glossary.md#term-ieee)-754 values after existing
promotion. [BFLOAT16](../../glossary.md#term-bfloat16) compares the value represented by its exact 16-bit storage; [widening](../../glossary.md#term-widening) for the
existing comparison domain adds no conversion or [rounding](../../glossary.md#term-rounding) contract. The four ordered relations
are false when either operand is NaN. [Negative zero](../../glossary.md#nan-infinity-and-signed-zero) and [positive zero](../../glossary.md#nan-infinity-and-signed-zero) compare equal, so neither
strict relation holds and both inclusive relations hold in either order. `EQUAL` is exact numeric
equality, not bit or [tolerance](../../glossary.md#term-tolerance) equality; NaN is unequal to every value, including itself.
`NOT_EQUAL` is its exact logical complement. [Finite](../../glossary.md#term-finite) values and [infinities](../../glossary.md#nan-infinity-and-signed-zero) otherwise use ordinary
numeric order. Integral comparison remains exact signed comparison after promotion. [Gradient](../../glossary.md#term-gradient)
policy beyond the descriptor flag, [compiler](../../glossary.md#term-compiler) [capture](../../glossary.md#term-capture-meanings), execution, and [backend](../../glossary.md#backend) availability remain
planned. Inherited enum names and text are diagnostic rather than [serialization](../../glossary.md#term-serialization) or [dispatch](../../glossary.md#term-dispatch) keys.
Typed [identity](../../glossary.md#term-identity-meanings) keeps an equally named kind from another family unequal to
`BinaryComparisonKind`.

<a id="boolean-logical-semantic-kinds"></a>
### [Boolean](../../glossary.md#data-type--datatype) logical semantic kinds

The public enum
`io.github.pho001.synaptik.model.operation.elementwise.logical.BooleanLogicalKind` implements
[`OperationKind`](../../glossary.md#operationkind) with exactly these [constants](../../glossary.md#term-constant) in declaration order:

| Kind | [Elementwise](../../glossary.md#term-elementwise) boolean meaning | Logical input roles |
|---|---|---|
| `AND` | True exactly when both input values are true. | Ordered left and right inputs. |
| `OR` | True exactly when at least one input value is true. | Ordered left and right inputs. |
| `NOT` | True exactly when the input value is false. | One input. |

All three kinds are parameterless semantic [identities](../../glossary.md#term-identity-meanings). Their explicit generic composition is:

```java
// Elementwise AND has two ports, unlike a one-input all() reduction.
Operation conjunction = new Operation(
        BooleanLogicalKind.AND,
        NoOperationAttrs.INSTANCE);
```

The resulting [operation](../../glossary.md#operation) retains the exact `AND` kind and [canonical](../../glossary.md#term-canonical-form) no-attributes value. Family
[signatures](../../glossary.md#term-signature) declare two inputs for `AND` and `OR`, one input for `NOT`, and one output for each.
`CompiledNode` validates those local [occurrence](../../glossary.md#term-occurrence) counts. The enum stores no input references and
creates no [Tensor](../../glossary.md#tensor), [provenance](../../glossary.md#provenance), or result [descriptor](../../glossary.md#term-descriptor).

This vocabulary defines conjunction, disjunction, and negation truth meaning only. The current
`Tensor.logicalAnd`, `logicalOr`, and `logicalNot` methods separately own exact [BOOL](../../glossary.md#data-type--datatype) eligibility,
[binary](../../glossary.md#term-binary-descriptor-and-metadata) [broadcasting](../../glossary.md#broadcasting), [unary](../../glossary.md#term-unary) [shape](../../glossary.md#shape) preservation, fixed BOOL result construction, and provenance.
BOOL storage encoding, numeric truthiness, [gradients](../../glossary.md#term-gradient), [compiler](../../glossary.md#term-compiler) [capture](../../glossary.md#term-capture-meanings), execution, and [backend](../../glossary.md#backend)
availability remain outside both semantic [identity](../../glossary.md#term-identity-meanings) and current expression construction. Inherited
enum names and text are diagnostic rather than [serialization](../../glossary.md#term-serialization) or [dispatch](../../glossary.md#term-dispatch) keys, and an equally
named kind from another family remains a different typed value.

<a id="where-selection-semantic-kind"></a>
### Where selection semantic kind

The public enum
`io.github.pho001.synaptik.model.operation.elementwise.selection.WhereSelectionKind` implements
[`OperationKind`](../../glossary.md#operationkind) with exactly one [constant](../../glossary.md#term-constant):

| Kind | [Elementwise](../../glossary.md#term-elementwise) meaning | Logical input roles |
|---|---|---|
| `WHERE` | Choose the true-branch value when the corresponding condition is true; otherwise choose the false-branch value. | Ordered condition, true branch, and false branch. |

`WHERE` has no intrinsic [parameters](../../glossary.md#term-parameter). Its explicit generic composition is:

```java
// WHERE records three roles; it is not a Java branch evaluating just one expression.
Operation selection = new Operation(
        WhereSelectionKind.WHERE,
        NoOperationAttrs.INSTANCE);
```

The resulting [operation](../../glossary.md#operation) retains the exact `WHERE` kind and [canonical](../../glossary.md#term-canonical-form) no-attributes value. Its
family [signature](../../glossary.md#term-signature) declares three ordered inputs and one output and enforces the exact attributes
pairing. The enum itself stores no input objects.

This semantic [identity](../../glossary.md#term-identity-meanings) describes elementwise conditional choice, not [scalar](../../glossary.md#term-scalar)-index `select`,
[gather](../../glossary.md#gather), take, or scatter. The later indexing family owns those distinct capabilities. The enum
does not itself define condition and branch eligibility, branch promotion, three-way [broadcasting](../../glossary.md#broadcasting),
result [descriptors](../../glossary.md#term-descriptor), [provenance](../../glossary.md#provenance), [evaluation](../../glossary.md#term-evaluation) order, [gradients](../../glossary.md#term-gradient), [compiler](../../glossary.md#term-compiler) [capture](../../glossary.md#term-capture-meanings), [ONNX](../../glossary.md#term-onnx) mapping,
execution, or [backend](../../glossary.md#backend) availability. The current static `Tensor.where` expression method separately
owns local [BOOL](../../glossary.md#data-type--datatype)/floating [validation](../../glossary.md#term-validation), branch promotion, ordered pairwise broadcasting, descriptor
construction, and exact three-input provenance. Its inherited name and text are diagnostic rather
than [serialization](../../glossary.md#term-serialization) or [dispatch](../../glossary.md#term-dispatch) keys, and an equally named kind from another family remains a
different typed value.

<a id="cast-semantic-kind-and-attributes"></a>
### Cast semantic kind and attributes

The public enum
`io.github.pho001.synaptik.model.operation.elementwise.cast.CastKind` implements [`OperationKind`](../../glossary.md#operationkind)
with exactly one [constant](../../glossary.md#term-constant):

| Kind | [Elementwise](../../glossary.md#term-elementwise) meaning | Required attributes |
|---|---|---|
| `CAST` | Convert each value from one logical input to a requested [target](../../glossary.md#term-target) [data type](../../glossary.md#data-type--datatype). | `CastAttrs` |

`CastAttrs` has exactly one component, `targetDataType`. It accepts and retains every current
[`DataType`](../../glossary.md#data-type--datatype): [`FLOAT64`](../../glossary.md#term-float64), [`FLOAT32`](../../glossary.md#term-float32), [`BFLOAT16`](../../glossary.md#term-bfloat16), [`INT32`](../../glossary.md#data-type--datatype), [`INT64`](../../glossary.md#data-type--datatype), [`BOOL`](../../glossary.md#data-type--datatype), and appended [`FLOAT16`](../../glossary.md#term-float16).
The source data type is not duplicated in the attributes because it belongs to the later input [Tensor](../../glossary.md#tensor) or [graph](../../glossary.md#graph)-value
[descriptor](../../glossary.md#term-descriptor). The explicit generic composition is:

```java
// Store only the destination type: the eventual input descriptor supplies the source type.
Operation cast = new Operation(
        CastKind.CAST,
        new CastAttrs(DataType.FLOAT64));
```

The resulting descriptor retains the exact kind and attributes values. [`Operation`](../../glossary.md#operation) construction
enforces the exact `CAST`/`CastAttrs` pairing, and the [signature](../../glossary.md#term-signature) declares one input and one output.
A null target fails during `CastAttrs` construction with
`NullPointerException` and message `targetDataType`.

The current `Tensor.cast(DataType)` method separately owns source-descriptor inspection,
exact [shape](../../glossary.md#shape) retention, unresolved result [layout](../../glossary.md#layout), floating-only [gradient](../../glossary.md#term-gradient) eligibility, fresh
same-type [identity](../../glossary.md#term-identity-meanings), and one-input [provenance](../../glossary.md#provenance). `CastValueConversions` defines all 49 ordered pairs'
[scalar](../../glossary.md#term-scalar) value meaning: same-type raw identity, direct target-format [round-to-nearest](../../glossary.md#floating-point-precision-and-rounding) [ties-to-even](../../glossary.md#term-ties-to-even),
deterministic [narrowing and widening](../../glossary.md#floating-point-precision-and-rounding) NaNs, [saturating](../../glossary.md#saturation-pole-and-extremum) floating-to-integral conversion, signed
extension or low-bit integral width conversion, and [canonical](../../glossary.md#term-canonical-form) [Boolean](../../glossary.md#data-type--datatype) truthiness. Expression
construction records but does not evaluate that meaning. [Compiler](../../glossary.md#term-compiler) owns the existing floating-only
reverse rule, and [backend](../../glossary.md#backend) availability remains separate; accepting every pair in [Model](../../glossary.md#term-model) is not a
promise that every backend executes it. Enum and record text remain diagnostic rather than
[serialization](../../glossary.md#term-serialization) or [dispatch](../../glossary.md#term-dispatch) contracts.


---

[Previous topic](operation-foundation.md) · [Next topic](operation-reductions.md)
