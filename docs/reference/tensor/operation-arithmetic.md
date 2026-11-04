# Arithmetic, comparison, Boolean, selection and cast kinds

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

> **Scope:** Model construction, Compiler admission, backend execution, and checkpoint persistence are separate gates. Execution-status sentences retained in this detailed material record earlier qualification slices; they are not an exhaustive current support matrix. The [current backend matrix](../../backend-guide/production-support.md) and its incorporated contracts supersede narrower historical CPU/Metal lists. Package-private implementation sketches are not supported application API.

<a id="binary-arithmetic-semantic-kinds"></a>
### Binary arithmetic semantic kinds

The public enum
`io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind` implements
`OperationKind` with exactly these constants in declaration order:

| Kind | Ordered elementwise meaning |
|---|---|
| `ADD` | Left value plus right value. |
| `SUB` | Left value minus right value. |
| `MUL` | Left value multiplied by right value. |
| `DIV` | Left value divided by right value. |
| `MIN` | Mathematical minimum of the left and right values. |
| `MAX` | Mathematical maximum of the left and right values. |
| `POW` | Left value as the base raised to the right value as the exponent. |

The enum values are semantic identities only. All seven have no intrinsic parameters and compose
explicitly with the existing generic descriptor:

```java
Operation addition = new Operation(
        BinaryArithmeticKind.ADD,
        NoOperationAttrs.INSTANCE);
```

The result is an immutable operation description whose kind is exactly
`BinaryArithmeticKind.ADD` and whose attributes value is exactly the canonical singleton. It is
not a Tensor result, provenance value, graph occurrence, executable operation, or backend-support
claim. No family factory or implicit attributes assignment is provided.

Broadcast geometry is derived from operand shapes rather than stored on the enum or in operation
attributes. The current public Tensor methods own local floating eligibility, promotion,
broadcasting, result-descriptor construction, and ordered provenance. ADD, SUB, MUL, and DIV are
ordinary ordered IEEE-754 requests in the result type. MIN and MAX propagate NaN, order infinities
normally, and choose negative or positive zero, respectively, from opposite signed zeros,
independent of operand order. Unequal non-NaN values use ordinary numeric order, while equal
nonzero candidates produce that numeric value without identifying a selected operand. FLOAT64 and
FLOAT32 use their represented IEEE-754 values; BFLOAT16 uses the value represented by its exact
16-bit storage after existing promotion. Integral extrema retain ordinary signed order after
promotion. No NaN payload, intermediate precision, exact instruction, bitwise result, gradient
rule, compiler capture, execution, or backend availability is promised here. The inherited enum
names and text are stable diagnostics, not serialization
tokens, registry keys, or string-dispatch contracts. Equality remains typed: an `ADD` value
declared by another kind family is not equal to `BinaryArithmeticKind.ADD` even though both
diagnostics may contain `ADD`.

<a id="binary-comparison-semantic-kinds"></a>
### Binary comparison semantic kinds

The public enum
`io.github.pho001.synaptik.model.operation.elementwise.comparison.BinaryComparisonKind` implements
`OperationKind` with exactly these constants in declaration order:

| Kind | Ordered elementwise meaning |
|---|---|
| `GREATER_THAN` | The left value is strictly greater than the right value; false for NaN or either signed-zero order. |
| `GREATER_OR_EQUAL` | The left value is greater than or equal to the right value; false for NaN and true for either signed-zero order. |
| `LESS_THAN` | The left value is strictly less than the right value; false for NaN or either signed-zero order. |
| `LESS_OR_EQUAL` | The left value is less than or equal to the right value; false for NaN and true for either signed-zero order. |
| `EQUAL` | Exact represented numeric equality; NaN is unequal to itself and opposite signed zeros are equal. |
| `NOT_EQUAL` | The logical complement of `EQUAL`; true for NaN and false for opposite signed zeros. |

All six kinds are parameterless semantic identities. They compose explicitly with the generic
operation descriptor:

```java
Operation comparison = new Operation(
        BinaryComparisonKind.GREATER_THAN,
        NoOperationAttrs.INSTANCE);
```

The resulting operation retains the exact `GREATER_THAN` kind and canonical no-attributes value.
It describes ordered comparison meaning only; it stores no operands or broadcast plan and is not
a Tensor result, provenance value, graph occurrence, or executable operation. Construction
enforces the exact `NoOperationAttrs` pairing; the family signature declares two inputs and one
output.

The current public comparison methods separately own floating-pair validation, right-aligned local
broadcasting, fixed `BOOL` result derivation with false `requiresGrad`, and ordered provenance
construction. FLOAT64 and FLOAT32 compare their represented IEEE-754 values after existing
promotion. BFLOAT16 compares the value represented by its exact 16-bit storage; widening for the
existing comparison domain adds no conversion or rounding contract. The four ordered relations
are false when either operand is NaN. Negative zero and positive zero compare equal, so neither
strict relation holds and both inclusive relations hold in either order. `EQUAL` is exact numeric
equality, not bit or tolerance equality; NaN is unequal to every value, including itself.
`NOT_EQUAL` is its exact logical complement. Finite values and infinities otherwise use ordinary
numeric order. Integral comparison remains exact signed comparison after promotion. Gradient
policy beyond the descriptor flag, compiler capture, execution, and backend availability remain
planned. Inherited enum names and text are diagnostic rather than serialization or dispatch keys.
Typed identity keeps an equally named kind from another family unequal to
`BinaryComparisonKind`.

<a id="boolean-logical-semantic-kinds"></a>
### Boolean logical semantic kinds

The public enum
`io.github.pho001.synaptik.model.operation.elementwise.logical.BooleanLogicalKind` implements
`OperationKind` with exactly these constants in declaration order:

| Kind | Elementwise boolean meaning | Logical input roles |
|---|---|---|
| `AND` | True exactly when both input values are true. | Ordered left and right inputs. |
| `OR` | True exactly when at least one input value is true. | Ordered left and right inputs. |
| `NOT` | True exactly when the input value is false. | One input. |

All three kinds are parameterless semantic identities. Their explicit generic composition is:

```java
Operation conjunction = new Operation(
        BooleanLogicalKind.AND,
        NoOperationAttrs.INSTANCE);
```

The resulting operation retains the exact `AND` kind and canonical no-attributes value. Family
signatures declare two inputs for `AND` and `OR`, one input for `NOT`, and one output for each.
`CompiledNode` validates those local occurrence counts. The enum stores no input references and
creates no Tensor, provenance, or result descriptor.

This vocabulary defines conjunction, disjunction, and negation truth meaning only. The current
`Tensor.logicalAnd`, `logicalOr`, and `logicalNot` methods separately own exact BOOL eligibility,
binary broadcasting, unary shape preservation, fixed BOOL result construction, and provenance.
BOOL storage encoding, numeric truthiness, gradients, compiler capture, execution, and backend
availability remain outside both semantic identity and current expression construction. Inherited
enum names and text are diagnostic rather than serialization or dispatch keys, and an equally
named kind from another family remains a different typed value.

<a id="where-selection-semantic-kind"></a>
### Where selection semantic kind

The public enum
`io.github.pho001.synaptik.model.operation.elementwise.selection.WhereSelectionKind` implements
`OperationKind` with exactly one constant:

| Kind | Elementwise meaning | Logical input roles |
|---|---|---|
| `WHERE` | Choose the true-branch value when the corresponding condition is true; otherwise choose the false-branch value. | Ordered condition, true branch, and false branch. |

`WHERE` has no intrinsic parameters. Its explicit generic composition is:

```java
Operation selection = new Operation(
        WhereSelectionKind.WHERE,
        NoOperationAttrs.INSTANCE);
```

The resulting operation retains the exact `WHERE` kind and canonical no-attributes value. Its
family signature declares three ordered inputs and one output and enforces the exact attributes
pairing. The enum itself stores no input objects.

This semantic identity describes elementwise conditional choice, not scalar-index `select`,
gather, take, or scatter. The later indexing family owns those distinct capabilities. The enum
does not itself define condition and branch eligibility, branch promotion, three-way broadcasting,
result descriptors, provenance, evaluation order, gradients, compiler capture, ONNX mapping,
execution, or backend availability. The current static `Tensor.where` expression method separately
owns local BOOL/floating validation, branch promotion, ordered pairwise broadcasting, descriptor
construction, and exact three-input provenance. Its inherited name and text are diagnostic rather
than serialization or dispatch keys, and an equally named kind from another family remains a
different typed value.

<a id="cast-semantic-kind-and-attributes"></a>
### Cast semantic kind and attributes

The public enum
`io.github.pho001.synaptik.model.operation.elementwise.cast.CastKind` implements `OperationKind`
with exactly one constant:

| Kind | Elementwise meaning | Required attributes |
|---|---|---|
| `CAST` | Convert each value from one logical input to a requested target data type. | `CastAttrs` |

`CastAttrs` has exactly one component, `targetDataType`. It accepts and retains every current
`DataType`: `FLOAT64`, `FLOAT32`, `BFLOAT16`, `INT32`, `INT64`, `BOOL`, and appended `FLOAT16`.
The source data type is not duplicated in the attributes because it belongs to the later input Tensor or graph-value
descriptor. The explicit generic composition is:

```java
Operation cast = new Operation(
        CastKind.CAST,
        new CastAttrs(DataType.FLOAT64));
```

The resulting descriptor retains the exact kind and attributes values. `Operation` construction
enforces the exact `CAST`/`CastAttrs` pairing, and the signature declares one input and one output.
A null target fails during `CastAttrs` construction with
`NullPointerException` and message `targetDataType`.

The current `Tensor.cast(DataType)` method separately owns source-descriptor inspection,
exact shape retention, unresolved result layout, floating-only gradient eligibility, fresh
same-type identity, and one-input provenance. `CastValueConversions` defines all 49 ordered pairs'
scalar value meaning: same-type raw identity, direct target-format round-to-nearest ties-to-even,
deterministic narrowing and widening NaNs, saturating floating-to-integral conversion, signed
extension or low-bit integral width conversion, and canonical Boolean truthiness. Expression
construction records but does not evaluate that meaning. Compiler owns the existing floating-only
reverse rule, and backend availability remains separate; accepting every pair in Model is not a
promise that every backend executes it. Enum and record text remain diagnostic rather than
serialization or dispatch contracts.


---

[Previous topic](operation-foundation.md) · [Next topic](operation-reductions.md)
