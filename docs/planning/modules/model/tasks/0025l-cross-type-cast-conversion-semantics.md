# Task 0025L: Cross-type CAST conversion semantics

## Status

Complete

## Goal

Define one [backend](../../../../glossary.md#backend)-independent, raw-bit-testable observable meaning for every ordered `CAST`
pair among the six current data types: [`FLOAT64`](../../../../glossary.md#term-float64), [`FLOAT32`](../../../../glossary.md#term-float32), [`BFLOAT16`](../../../../glossary.md#term-bfloat16), [`INT64`](../../../../glossary.md#data-type--datatype), [`INT32`](../../../../glossary.md#data-type--datatype), and
[`BOOL`](../../../../glossary.md#data-type--datatype). Preserve the existing exact-bit same-type [operation](../../../../glossary.md#operation) and floating-only differentiation
boundary, and add one [Model](../../../../glossary.md#term-model)-owned [scalar](../../../../glossary.md#term-scalar) reference conversion so Model tests and later backend
[conformance](../../../../glossary.md#term-conformance) work share an executable [oracle](../../../../glossary.md#term-test-oracle) without placing that oracle in backend hot paths.

This task closes the numerical-policy gap left deliberately open by Model 0015G–0015H. Public
`Tensor.cast(DataType)` already constructs all 36 ordered pairs; this task gives those existing
expressions exact meaning. It adds no [Tensor](../../../../glossary.md#tensor) overload, operation kind, attributes field, [data
type](../../../../glossary.md#data-type--datatype), backend support, or execution route.

## [Scope](../../../../glossary.md#term-scope)

- Admit all 36 ordered source/[target](../../../../glossary.md#term-target) pairs: six same-type pairs and all 30 cross-type pairs.
- Preserve every same-type raw represented bit, including signed zero and every NaN sign,
  quiet/signaling state, and payload.<span data-glossary-navigation="true"> ([signed zero glossary](../../../../glossary.md#term-signed-zero); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication))</span>
- Define exact floating-to-floating rounding, overflow, underflow, infinity, signed-zero, and NaN
  behavior, including direct <span data-glossary-navigation="true"> ([rounding glossary](../../../../glossary.md#term-rounding); [overflow glossary](../../../../glossary.md#term-overflow); [underflow glossary](../../../../glossary.md#term-underflow); [infinity glossary](../../../../glossary.md#term-infinity); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>`FLOAT64 -> BFLOAT16` conversion without a [`FLOAT32`](../../../../glossary.md#term-float32) intermediate.
- Define exact signed integral [widening](../../../../glossary.md#term-widening)/[narrowing](../../../../glossary.md#term-narrowing) and all floating/integral conversions.
- Define exact [`BOOL`](../../../../glossary.md#data-type--datatype) conversion and numeric truthiness.
- Preserve the existing [descriptor](../../../../glossary.md#term-descriptor) and differentiation contract: only floating-to-floating
  `CAST` can retain `requiresGrad`; [Compiler](../../../../glossary.md#term-compiler) reverses it with an ordinary `CAST` to the source
  floating type. A role with integral or `BOOL` source or target has no [cotangent](../../../../glossary.md#cotangent).
- Add one public final stateless `CastValueConversions` reference utility in the existing cast
  semantic package. Its single conversion entry accepts one exact [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) and a target
  [`DataType`](../../../../glossary.md#data-type--datatype), and returns one exact target-typed `ScalarValue` according to this specification.
  This is the concrete shared oracle needed by Model tests and the immediately following CPU
  conformance work. It is semantic reference code, not a Tensor evaluator or backend route.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [Model glossary](../../../../glossary.md#term-model); [CPU glossary](../../../../glossary.md#term-cpu); [conformance glossary](../../../../glossary.md#term-conformance); [backend route glossary](../../../../glossary.md#backend-route))</span>
- Update affected Javadocs and the [Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api) reference with the exact observable policy.
- Keep `BFloat16Bits.fromFloat(float)` and `BFloat16Bits.toFloat(short)` behavior unchanged,
  including canonical <span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>`FLOAT32 -> BFLOAT16` [NaN](../../../../glossary.md#term-nan) and exact raw `BFLOAT16 -> FLOAT32` expansion.

### [Scalar](../../../../glossary.md#term-scalar) reference contract

The exact new public surface is:

```java
public final class CastValueConversions {
    public static ScalarValue convert(ScalarValue source, DataType targetDataType);
}
```

The class has one private zero-[argument](../../../../glossary.md#term-argument) constructor, no [mutable](../../../../glossary.md#term-mutable) state, and no other public or
protected member declared by the project. `convert` checks `source` and then `targetDataType` with
`Objects.requireNonNull`, using exact messages `source` and `targetDataType`. A same-type call
returns the exact `source` reference, preserving every represented bit without a primitive
[round-trip](../../../../glossary.md#schema-serialization-and-checkpoint). A cross-type call returns a non-null [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) of exactly `targetDataType` and
applies the corresponding [matrix](../../../../glossary.md#term-matrix-meanings) cell below. Inexact, [overflow](../../../../glossary.md#term-overflow), [underflow](../../../../glossary.md#term-underflow), [NaN](../../../../glossary.md#term-nan), [infinity](../../../../glossary.md#term-infinity),
[saturation](../../../../glossary.md#term-saturation), and modulo outcomes are valid results rather than exceptions.

This utility is the executable scalar reference only. `Tensor.cast` still constructs a fresh
expression for a same-type request and does not call `convert` or inspect storage.

### Exhaustive ordered-pair [matrix](../../../../glossary.md#term-matrix-meanings)

The row is the source and the column is the [target](../../../../glossary.md#term-target). Every cell is admitted. `BITS` means exact raw
bit preservation; `RNE` means direct [round-to-nearest, ties-to-even](../../../../glossary.md#floating-point-precision-and-rounding); `WIDEN` means exact [finite](../../../../glossary.md#term-finite)
[widening](../../../../glossary.md#term-widening) plus the [NaN](../../../../glossary.md#term-nan) mapping below; `TRUNC-SAT` means truncate toward zero and saturate; `SIGNEXT`
means signed extension; `LOW32` means retain the low 32 [two's-complement](../../../../glossary.md#arithmetic-rearrangement-and-finite-difference-checks) bits; `01` means [Boolean](../../../../glossary.md#data-type--datatype)
false/true becomes numeric zero/one; and `TRUTH` means false only for integer zero or either
floating [signed zero](../../../../glossary.md#term-signed-zero).

| Source \\ Target | [`FLOAT64`](../../../../glossary.md#term-float64) | [`FLOAT32`](../../../../glossary.md#term-float32) | [`BFLOAT16`](../../../../glossary.md#term-bfloat16) | [`INT64`](../../../../glossary.md#data-type--datatype) | [`INT32`](../../../../glossary.md#data-type--datatype) | [`BOOL`](../../../../glossary.md#data-type--datatype) |
|---|---:|---:|---:|---:|---:|---:|
| `FLOAT64` | `BITS` | `RNE` | direct `RNE` | `TRUNC-SAT` | `TRUNC-SAT` | `TRUTH` |
| `FLOAT32` | `WIDEN` | `BITS` | `RNE` | `TRUNC-SAT` | `TRUNC-SAT` | `TRUTH` |
| `BFLOAT16` | `WIDEN` | exact `WIDEN` | `BITS` | `TRUNC-SAT` | `TRUNC-SAT` | `TRUTH` |
| `INT64` | `RNE` | `RNE` | direct `RNE` | `BITS` | `LOW32` | `TRUTH` |
| `INT32` | `RNE` | `RNE` | direct `RNE` | `SIGNEXT` | `BITS` | `TRUTH` |
| `BOOL` | `01` | `01` | `01` | `01` | `01` | `BITS` |

### Floating formats and [finite](../../../../glossary.md#term-finite) [rounding](../../../../glossary.md#term-rounding)

[`FLOAT64`](../../../../glossary.md#term-float64) is [IEEE](../../../../glossary.md#term-ieee)-754 [binary64](../../../../glossary.md#floating-point-precision-and-rounding) (`p = 53`, [exponent](../../../../glossary.md#term-exponent) range `-1022..1023`), [`FLOAT32`](../../../../glossary.md#term-float32) is [binary32](../../../../glossary.md#floating-point-precision-and-rounding)
(`p = 24`, exponent range `-126..127`), and [`BFLOAT16`](../../../../glossary.md#term-bfloat16) has binary32's eight-bit exponent with
[precision](../../../../glossary.md#term-precision) `p = 8` and exponent range `-126..127`.

For every finite floating-to-floating or integer-to-floating conversion, choose the [target](../../../../glossary.md#term-target) finite
number nearest to the exact mathematical source value. An exact midpoint chooses the target whose
least-significant [significand](../../../../glossary.md#term-significand) bit is zero. Preserve source sign when the rounded result is zero.
Rounding is directly to the requested target format; it is never defined as a chain through an
intermediate format.

If the rounded magnitude exceeds the target's finite range, the result is signed [infinity](../../../../glossary.md#term-infinity). Tiny
values round gradually through target [subnormals](../../../../glossary.md#term-subnormal) and then to [signed zero](../../../../glossary.md#term-signed-zero). The boundary cases are:

| Target | Largest finite bits | [Positive infinity](../../../../glossary.md#nan-infinity-and-signed-zero) bits | Smallest positive subnormal | Zero/subnormal midpoint |
|---|---:|---:|---:|---:|
| `FLOAT64` | `0x7FEFFFFFFFFFFFFF` | `0x7FF0000000000000` | `2^-1074` / `0x0000000000000001` | `2^-1075` (conceptual exact midpoint) |
| `FLOAT32` | `0x7F7FFFFF` | `0x7F800000` | `2^-149` / `0x00000001` | `2^-150` |
| `BFLOAT16` | `0x7F7F` | `0x7F80` | `2^-133` / `0x0001` | `2^-134` |

At a zero/subnormal midpoint, [ties-to-even](../../../../glossary.md#term-ties-to-even) selects signed zero. Immediately above it selects the
signed smallest subnormal. For `FLOAT32` and `BFLOAT16`, the positive finite/infinity RNE [overflow](../../../../glossary.md#term-overflow)
midpoints are respectively `(2 - 2^-24) * 2^127` and `(2 - 2^-8) * 2^127`; the midpoint and larger
magnitudes produce infinity, while smaller magnitudes round to a finite result. Negative behavior
is sign-symmetric.

`FLOAT64 -> BFLOAT16` must operate from the exact binary64 represented value. These required
[vectors](../../../../glossary.md#term-vector) distinguish direct conversion from forbidden double rounding through `FLOAT32`:

| Exact binary64 input | Direct BFLOAT16 | Intermediate FLOAT32 | BFLOAT16 after the forbidden intermediate |
|---:|---:|---:|---:|
| `0x3FF0100000400000` (`1 + 2^-8 + 2^-30`) | `0x3F81` | `0x3F808000` | `0x3F80` |
| `0x3FF02FFFFFC00000` (`1 + 3 * 2^-8 - 2^-30`) | `0x3F81` | `0x3F818000` | `0x3F82` |

Tests must cover both vectors, their negative counterparts, exact halfway values on both even/odd
target neighbors, [carry](../../../../glossary.md#term-carry) into a new exponent, finite/infinity boundaries, and zero/subnormal
boundaries.

### Edge-case audit tables

The following tables are [normative](../../../../glossary.md#term-normative) partitions of the same 36-pair [matrix](../../../../glossary.md#term-matrix-meanings). They make each edge
class independently auditable; they do not add another conversion mode.

#### Zero

| Source zero | Floating [target](../../../../glossary.md#term-target) | Integral target | [`BOOL`](../../../../glossary.md#data-type--datatype) target |
|---|---|---|---|
| floating `+0` | exact `+0` | `0` | `false` |
| floating `-0` | exact `-0` | `0` | `false` |
| integer `0` | `+0` | exact zero under `BITS`, `SIGNEXT`, or `LOW32` | `false` |
| `BOOL false` | `+0` | `0` | `false` |

#### [Subnormal](../../../../glossary.md#term-subnormal) and [underflow](../../../../glossary.md#term-underflow)

| Conversion class | Required behavior |
|---|---|
| same floating type | preserve the exact subnormal [raw bits](../../../../glossary.md#storage-mapping-contiguity-and-injectivity) |
| lossless floating [widening](../../../../glossary.md#term-widening) | preserve the exact [finite](../../../../glossary.md#term-finite) value; a source subnormal may become a normal [target](../../../../glossary.md#term-target) value |
| lossy floating [narrowing](../../../../glossary.md#term-narrowing) | direct target RNE, including gradual target subnormals; exact half of the least target subnormal rounds to [signed zero](../../../../glossary.md#term-signed-zero) and the next representable source value above it rounds to the signed least target subnormal |
| floating to integral | every source subnormal truncates to integer zero |
| floating to [`BOOL`](../../../../glossary.md#data-type--datatype) | every positive or negative non-zero subnormal maps to `true` |

#### [Finite](../../../../glossary.md#term-finite) [overflow](../../../../glossary.md#term-overflow)

| Conversion class | Positive result | Negative result |
|---|---|---|
| lossy floating [narrowing](../../../../glossary.md#term-narrowing) at or beyond the [target](../../../../glossary.md#term-target) RNE overflow midpoint | `+infinity` | `-infinity` |
| floating to integral above/below the target interval | target maximum | target minimum |
| current integer to floating | direct finite RNE result; current [`INT32`](../../../../glossary.md#data-type--datatype)/[`INT64`](../../../../glossary.md#data-type--datatype) magnitudes do not overflow any current floating [exponent](../../../../glossary.md#term-exponent) range | sign-symmetric finite RNE result |

#### [Infinity](../../../../glossary.md#term-infinity)

| Source | Floating [target](../../../../glossary.md#term-target) | Integral target | [`BOOL`](../../../../glossary.md#data-type--datatype) target |
|---|---|---|---|
| `+infinity` | `+infinity` | target maximum | `true` |
| `-infinity` | `-infinity` | target minimum | `true` |

### Floating [NaN](../../../../glossary.md#term-nan) policy

Same-type casts preserve the complete raw pattern. Cross-floating [narrowing](../../../../glossary.md#term-narrowing) is lossy and maps
every source NaN, regardless of sign, signaling/quiet bit, or [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication), to the positive [canonical](../../../../glossary.md#term-canonical-form)
quiet [target](../../../../glossary.md#term-target) pattern:

| Target | Canonical NaN bits |
|---|---:|
| [`FLOAT64`](../../../../glossary.md#term-float64) | `0x7FF8000000000000` |
| [`FLOAT32`](../../../../glossary.md#term-float32) | `0x7FC00000` |
| [`BFLOAT16`](../../../../glossary.md#term-bfloat16) | `0x7FC0` |

No current lossy conversion targets `FLOAT64`, because all current sources are no wider than
[binary64](../../../../glossary.md#floating-point-precision-and-rounding). Its canonical pattern is nevertheless fixed so the family contract names all three
current floating targets; none of the current 36 pairs emits it through lossy conversion.

The only lossless cross-floating directions are `BFLOAT16 -> FLOAT32`, `BFLOAT16 -> FLOAT64`, and
`FLOAT32 -> FLOAT64`. For a NaN in those directions, preserve the sign bit and [exponent](../../../../glossary.md#term-exponent) of all
ones, then place the complete source fraction in the most-significant bits of the target fraction
and fill newly available low fraction bits with zero:

```text
BFLOAT16 -> FLOAT32: targetFraction = sourceFraction << 16
BFLOAT16 -> FLOAT64: targetFraction = sourceFraction << 45
FLOAT32  -> FLOAT64: targetFraction = sourceFraction << 29
```

This preserves all source payload bits and the source quiet/signaling bit deterministically. The
first mapping is exactly the existing `BFloat16Bits.toFloat(bits)` raw expansion. Representative
required mappings include `BF16 0x7F81 -> F32 0x7F810000`, `BF16 0xFFC1 -> F64
0xFFF8200000000000`, and `F32 0x7FA12345 -> F64 0x7FF42468A0000000`. Tests must independently
recalculate these patterns rather than relying on Java's unspecified [NaN payload](../../../../glossary.md#nan-infinity-and-signed-zero) propagation.

The NaN edge cases are therefore:

| Source/target class | Required result |
|---|---|
| same floating type | exact source [raw bits](../../../../glossary.md#storage-mapping-contiguity-and-injectivity) |
| lossy cross-floating narrowing | exact positive canonical quiet target NaN from the table above |
| lossless cross-floating [widening](../../../../glossary.md#term-widening) | exact sign/exponent/left-aligned-fraction mapping above |
| floating NaN to integral | zero |
| floating NaN to [`BOOL`](../../../../glossary.md#data-type--datatype) | `true` |

### Integral policy

- `INT32 -> INT64` sign-extends the exact [two's-complement](../../../../glossary.md#arithmetic-rearrangement-and-finite-difference-checks) value.
- `INT64 -> INT32` retains bits 31 through 0 and interprets them as a signed two's-complement
  [`INT32`](../../../../glossary.md#data-type--datatype), equivalently reduction modulo `2^32`. Examples: `0x0000000100000001L -> 1`,
  `0x00000000FFFFFFFFL -> -1`, `Long.MIN_VALUE -> 0`, and `Long.MAX_VALUE -> -1`.
- Same-type integral casts preserve all bits.
- Integer-to-floating uses the direct target RNE rule. Inexact results are valid. Required boundary
  examples include `Integer.MAX_VALUE -> FLOAT32 0x4F000000` (`2^31`),
  `Long.MAX_VALUE -> FLOAT64 0x43E0000000000000` (`2^63`), and `Long.MIN_VALUE -> FLOAT64
  0xC3E0000000000000` (exact `-2^63`). Integer-to-BFLOAT16 is direct, not FLOAT32-mediated. For
  example, `INT32 1077936129` (`2^30 + 2^22 + 1`) converts directly to [BFLOAT16](../../../../glossary.md#term-bfloat16) `0x4E81`;
  conversion through FLOAT32 first produces midpoint `0x4E808000` and then forbidden BFLOAT16
  `0x4E80`. `INT64 2155872257` similarly converts directly to `0x4F01`, not the double-rounded
  `0x4F00` obtained through [FLOAT32](../../../../glossary.md#term-float32) midpoint `0x4F008000`.
- Floating-to-integer first interprets the source's represented value. NaN becomes zero. Positive
  and negative infinity become the target maximum and minimum. For finite <span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [negative infinity glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [finite glossary](../../../../glossary.md#term-finite))</span>`x`, compute the exact
  mathematical truncation toward zero, then clamp that integer to the inclusive target interval:<span data-glossary-navigation="true"> ([truncation glossary](../../../../glossary.md#term-truncation-toward-zero))</span>
  `[-2^63, 2^63 - 1]` or `[-2^31, 2^31 - 1]`.

Floating-to-integer boundary tests must include positive/negative fractional values, both signed
zeros, smallest/largest source [subnormals](../../../../glossary.md#term-subnormal), exact integral values, and these [target](../../../../glossary.md#term-target) edges:

| Target | Required source/result examples |
|---|---|
| `INT32` | [binary64](../../../../glossary.md#floating-point-precision-and-rounding) `2147483646.9 -> 2147483646`; `2147483647.9 -> 2147483647`; `-2147483648.9 -> -2147483648`; [binary32](../../../../glossary.md#floating-point-precision-and-rounding) bits `0x4EFFFFFF -> 2147483520`; `0x4F000000 -> 2147483647`; `0xCF000000 -> -2147483648` |
| [`INT64`](../../../../glossary.md#data-type--datatype) | binary64 bits `0x43DFFFFFFFFFFFFF -> 9223372036854774784`; `0x43E0000000000000 -> Long.MAX_VALUE`; `0xC3E0000000000000 -> Long.MIN_VALUE` |

The integral-width edge cases are:

| Source | Target | Required result |
|---|---|---|
| `Integer.MIN_VALUE` / `Integer.MAX_VALUE` | `INT64` | exact sign-extended values |
| `Long.MIN_VALUE` | `INT32` | `0` |
| `Long.MAX_VALUE` | `INT32` | `-1` |
| any same-type integral value | same integral type | exact source bits |
| current integral limit | any floating type | one direct target RNE, with no intermediate floating format |

### [Boolean](../../../../glossary.md#data-type--datatype) policy

| Source | [Target](../../../../glossary.md#term-target)/result |
|---|---|
| `BOOL false` | numeric [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero), integral zero, or `BOOL false` |
| `BOOL true` | numeric positive one, integral one, or `BOOL true` |
| numeric `+0` or floating `-0` | `BOOL false` |
| any non-zero [finite](../../../../glossary.md#term-finite) numeric value or [subnormal](../../../../glossary.md#term-subnormal) | `BOOL true` |
| either [infinity](../../../../glossary.md#term-infinity) or any [NaN](../../../../glossary.md#term-nan) raw pattern | `BOOL true` |

[`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) accepts only the [canonical](../../../../glossary.md#term-canonical-form) Boolean domain, so this task does not define truthiness
for arbitrary storage bytes.

### Public expression and differentiation behavior

`Tensor.cast` continues to create one fresh explicit storage-free expression for every valid call,
including same-type casts. [Shape](../../../../glossary.md#shape) [identity](../../../../glossary.md#term-identity-meanings), unresolved result [layout](../../../../glossary.md#layout), [provenance](../../../../glossary.md#provenance), allocation order,
and null failures remain unchanged. This task changes the documented observable value meaning, not
construction mechanics.

The exact existing differentiation [matrix](../../../../glossary.md#term-matrix-meanings) remains:

| Source category | [Target](../../../../glossary.md#term-target) category | Result retains requested `requiresGrad` | Input [cotangent](../../../../glossary.md#cotangent) |
|---|---|---:|---|
| floating | floating | yes | incoming cotangent unchanged for same type; otherwise one ordinary `CAST` to the source type |
| floating | integral or [`BOOL`](../../../../glossary.md#data-type--datatype) | no | none |
| integral or `BOOL` | any | no | none |

The cast-back uses the same forward numerical contract in this task; there is no [gradient](../../../../glossary.md#term-gradient)-specific
[rounding](../../../../glossary.md#term-rounding), [NaN](../../../../glossary.md#term-nan), [saturation](../../../../glossary.md#term-saturation), straight-through estimator, or cotangent for a non-floating role.

### Current and planned behavior

Current after implementation: `Tensor.cast` constructs all 36 expressions, [Model](../../../../glossary.md#term-model) owns their exact
value meaning through `CastValueConversions`, and [Compiler](../../../../glossary.md#term-compiler) retains its existing floating-only
reverse boundary. [CPU](../../../../glossary.md#term-cpu) remains [fail-closed](../../../../glossary.md#term-fail-closed) for cross-type CAST until CPU 0008K implements and
truthfully advertises execution.

CPU 0008K is now unblocked by this completed semantic prerequisite but remains Draft; it owns
[scalar](../../../../glossary.md#term-scalar)/parallel-scalar execution, not this task. Model 0026 remains the separate future Draft [owner](../../../../glossary.md#term-owner)
of [`FLOAT16`](../../../../glossary.md#term-float16) and mixed-[precision](../../../../glossary.md#term-precision) semantics.

## Out of [scope](../../../../glossary.md#term-scope)

- Java implementation by this [planning](../../../../glossary.md#term-planning)/documentation pass
- CPU or another backend capability, lowering, loops, code generation, vectorization, execution,
  storage conversion, hot-path dispatch, or performance evidence<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [backend capability glossary](../../../../glossary.md#backend-capability); [lowering glossary](../../../../glossary.md#lowering); [code generation glossary](../../../../glossary.md#term-code-generation); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- constant folding, redundant-cast elimination, cast-chain rewriting, promotion changes, or
  implicit casts<span data-glossary-navigation="true"> ([constant folding glossary](../../../../glossary.md#term-constant-folding))</span>
- a new [`Tensor`](../../../../glossary.md#tensor) method/overload, [operation](../../../../glossary.md#operation) kind, `CastAttrs` field, configurable conversion mode,
  rounding mode, saturation mode, error flag, or exception-on-loss policy<span data-glossary-navigation="true"> ([rounding glossary](../../../../glossary.md#term-rounding); [saturation glossary](../../../../glossary.md#term-saturation); [loss glossary](../../../../glossary.md#loss))</span>
- changing [Compiler](../../../../glossary.md#term-compiler) [inference](../../../../glossary.md#term-inference), [derivative](../../../../glossary.md#term-derivative) inventory, or [gradient](../../../../glossary.md#term-gradient) formulas
- adding [`FLOAT16`](../../../../glossary.md#term-float16); quantized, unsigned, complex, string, or arbitrary-[precision](../../../../glossary.md#term-precision) types
- mixed-precision accumulation/intermediate policies for non-cast operations; Model 0026 retains
  all true binary16 and mixed-precision scope<span data-glossary-navigation="true"> ([binary16 glossary](../../../../glossary.md#floating-point-precision-and-rounding))</span>
- treating non-[canonical](../../../../glossary.md#term-canonical-form) [backend](../../../../glossary.md#backend) [BOOL](../../../../glossary.md#data-type--datatype) storage bytes as [Model](../../../../glossary.md#term-model) [Boolean](../../../../glossary.md#data-type--datatype) values

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially Core invariants,
  `modules/model`, [Compiler](../../../../glossary.md#term-compiler)-owned [automatic differentiation](../../../../glossary.md#autograd), `modules/compiler`, and concrete
  backend ownership<span data-glossary-navigation="true"> ([concrete backend glossary](../../../../glossary.md#concrete-backend); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Model master plan](../master-plan.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [CPU master plan](../../../backends/cpu/master-plan.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- [Model 0001 data type model](0001-data-type-model.md)<span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype))</span>
- [Model 0003A data type package migration](0003a-data-type-package-migration.md)
- [Model 0015G cast kind and attributes](0015g-cast-semantic-kind-and-attributes.md)
- [Model 0015H cast Tensor expression](0015h-cast-tensor-expression.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- [Model 0018N typed scalar value contract](0018n-typed-scalar-value-contract.md)<span data-glossary-navigation="true"> ([typed scalar value glossary](../../../../glossary.md#typed-scalar-value--scalarvalue))</span>
- [Model 0018U integral arithmetic and comparisons](0018u-integral-elementwise-arithmetic-and-comparisons.md)
- [Model 0025K 3D window transforms](0025k-public-ncdhw-unfold3d-and-fold3d-window-transforms.md)
- [Compiler 0005A derivative policy](../../../modules/compiler/tasks/0005a-derivative-policy-and-elementwise-activation-gradient-completion.md)<span data-glossary-navigation="true"> ([derivative glossary](../../../../glossary.md#term-derivative))</span>
- [Compiler 0005E gradient coverage audit](../../../modules/compiler/tasks/0005e-first-order-gradient-coverage-closure-checkpoint.md)<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient))</span>
- [CPU 0008J BFLOAT16 scalar pointwise closure](../../../backends/cpu/tasks/0008j-bfloat16-scalar-pointwise-closure.md)<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [scalar glossary](../../../../glossary.md#term-scalar); [pointwise glossary](../../../../glossary.md#term-pointwise))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md), with General,
  Planning, Example, and API/Javadoc profiles<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

## Architecture constraints

- Model owns the conversion meaning and reference scalar semantics but no backend capability,
  physical carrier selection, loop, kernel, or execution route.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [scalar glossary](../../../../glossary.md#term-scalar); [backend capability glossary](../../../../glossary.md#backend-capability); [carrier glossary](../../../../glossary.md#term-storage-carrier); [kernel glossary](../../../../glossary.md#kernel))</span>
- [`Operation`](../../../../glossary.md#operation) remains [backend](../../../../glossary.md#backend)-neutral and exposes no support query.
- [`Tensor`](../../../../glossary.md#tensor) remains immutable identity/descriptor/provenance model state, not IR, storage execution,
  or gradient lifecycle state.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [identity glossary](../../../../glossary.md#term-identity-meanings); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [gradient glossary](../../../../glossary.md#term-gradient); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Compiler continues to own derivative rules and constructs them only through public Tensor
  expressions. This task changes no Compiler source.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [derivative glossary](../../../../glossary.md#term-derivative))</span>
- CPU 0008K may consume the completed contract but must implement specialized conversion directly;
  generated or direct backend hot loops must not call <span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>`CastValueConversions`, allocate
  [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) per [element](../../../../glossary.md#term-element), box, reflect, or [dispatch](../../../../glossary.md#term-dispatch) through maps/strings.
- The scalar reference utility may be used by Model tests, backend-conformance tests, and cold
  verification. It is not a fallback executable and does not advertise backend support.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance))</span>
- No [dependency direction](../../../../glossary.md#java-platform-and-build-tools), [module](../../../../glossary.md#term-module) boundary, build structure, or architecture rule changes.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.datatype` — owns [`DataType`](../../../../glossary.md#data-type--datatype), exact typed [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue), and the
  existing BFLOAT16 bit contract.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16))</span>
- `io.github.pho001.synaptik.model.operation.elementwise.cast` — owns CAST meaning and attributes.
- `io.github.pho001.synaptik.model.tensor` — owns public expression construction and [Javadoc](../../../../glossary.md#term-javadoc).

Packages added or changed:

- No package is added.
- The existing cast package gains the conversion reference because it owns the complete CAST
  semantic family rather than general promotion or implicit data-type conversion.

Type placement:

- `io.github.pho001.synaptik.model.operation.elementwise.cast.CastValueConversions` — public final
  stateless scalar reference for the exact operation semantics; public visibility is justified by
  the concrete cross-module conformance consumer, while its Javadoc excludes hot-path use and
  backend-support meaning.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [operation glossary](../../../../glossary.md#operation); [module glossary](../../../../glossary.md#term-module); [conformance glossary](../../../../glossary.md#term-conformance); [backend glossary](../../../../glossary.md#backend))</span>
- Existing `CastKind`, `CastAttrs`, `TensorCastExpressions`, and `Tensor.cast` remain in place and
  gain exact numerical-contract Javadoc only.

## Affected files

Expected implementation and documentation files:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/cast/CastValueConversions.java` (new)
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/cast/CastKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/cast/CastAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorCastExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/elementwise/cast/CastValueConversionsTest.java` (new)
- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/elementwise/cast/CastSemanticsTest.java`, only if needed to lock the new public type/family relationship
- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/tasks/0025l-cross-type-cast-conversion-semantics.md`
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

Reviewed but normally unchanged:

- `BFloat16Bits.java` and `BFloat16BitsTest.java`; their existing `FLOAT32 <-> BFLOAT16` behavior is
  preserved and the widening/narrowing oracle must agree with it.<span data-glossary-navigation="true"> ([widening glossary](../../../../glossary.md#term-widening); [narrowing glossary](../../../../glossary.md#term-narrowing); [oracle glossary](../../../../glossary.md#term-test-oracle))</span>
- `DataType.java`, `DataTypePromotion.java`, and their tests; explicit CAST does not alter type
  vocabulary or promotion.
- `ScalarValue.java` and `ScalarValueTest.java`; the existing typed raw-value [carrier](../../../../glossary.md#term-storage-carrier) is sufficient.
- `TensorCastExpressionTest.java`; it already exhaustively locks 36-pair construction and the
  gradient-eligibility matrix.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- Compiler cast source/tests; the existing floating cast-back and non-differentiable boundary are
  preserved.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [differentiable glossary](../../../../glossary.md#derivative-and-chain-rule))</span>

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most 14 files total, including this task and synchronized
[planning](../../../../glossary.md#term-planning) files. At most two new Java files are allowed: the reference type and its focused test.
No [Gradle](../../../../glossary.md#term-gradle), architecture, [Compiler](../../../../glossary.md#term-compiler), [CPU](../../../../glossary.md#term-cpu)/[backend](../../../../glossary.md#backend), [conformance](../../../../glossary.md#term-conformance), or integration-test file may change.

If a different public method/type, another package, more files, or backend/compiler work is needed,
stop and propose a follow-up task.

## Acceptance criteria

- All 36 ordered pairs are admitted and tested or covered by exhaustive [matrix](../../../../glossary.md#term-matrix-meanings) iteration.
- Same-type conversion returns a [target](../../../../glossary.md#term-target) [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) with identical type and raw [represented bits](../../../../glossary.md#term-represented-bits).
- Every cross-floating finite, zero, infinity, overflow, underflow, and NaN rule above is covered at
  raw-bit level, including direct binary64-to-BFLOAT16 double-rounding counterexamples.<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite); [infinity glossary](../../../../glossary.md#term-infinity); [overflow glossary](../../../../glossary.md#term-overflow); [underflow glossary](../../../../glossary.md#term-underflow); [NaN glossary](../../../../glossary.md#term-nan); [binary64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [rounding glossary](../../../../glossary.md#term-rounding))</span>
- Lossy NaNs produce exactly the three canonical patterns; widening NaNs implement exactly the
  three left-aligned mappings and preserve sign, quiet/signaling state, and payload.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [widening glossary](../../../../glossary.md#term-widening); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication))</span>
- Integral sign extension, low-32-bit narrowing, integer-to-floating RNE, floating truncation/
  saturation, NaN-to-zero, and infinity saturation match the exact boundaries and examples above.<span data-glossary-navigation="true"> ([sign extension glossary](../../../../glossary.md#term-sign-extension); [narrowing glossary](../../../../glossary.md#term-narrowing); [truncation glossary](../../../../glossary.md#term-truncation-toward-zero); [saturation glossary](../../../../glossary.md#term-saturation))</span>
- Boolean conversions match positive 0/1 and numeric truthiness for signed zero, subnormal,
  finite non-zero, infinity, and every selected NaN sign/payload.<span data-glossary-navigation="true"> ([Boolean glossary](../../../../glossary.md#data-type--datatype); [signed zero glossary](../../../../glossary.md#term-signed-zero); [subnormal glossary](../../../../glossary.md#term-subnormal))</span>
- `CastValueConversions` is null-safe with deterministic [parameter](../../../../glossary.md#term-parameter)-order failures, accepts every
  `ScalarValue`/target pair, has no mutable state, storage access, Tensor dependency, backend
  dependency, registry, generic map, reflection, or per-call policy object.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [Tensor glossary](../../../../glossary.md#tensor); [dependency glossary](../../../../glossary.md#term-dependency); [backend glossary](../../../../glossary.md#backend); [registry glossary](../../../../glossary.md#term-registry))</span>
- Its exact declared [API](../../../../glossary.md#term-application-programming-interface-api) is one private zero-[argument](../../../../glossary.md#term-argument) constructor and one public static
  `convert(ScalarValue, DataType)` method; same-type conversion returns the exact source reference,
  and null checks use the specified order/messages without conversion work.
- Existing `BFloat16Bits.fromFloat` [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form) and `toFloat` exact expansion remain passing
  and unchanged in meaning.
- `Tensor.cast` public shape, fresh occurrence, descriptor/provenance, all-pair construction, and
  exact differentiation matrix remain unchanged.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [occurrence glossary](../../../../glossary.md#term-occurrence); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance))</span>
- Javadocs define value behavior without claiming evaluation at Tensor construction or universal
  backend support. Every public parameter/result/failure is documented.<span data-glossary-navigation="true"> ([evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- The Tensor API explains all conversion categories, canonical/mapped NaNs, direct RNE, integer
  boundaries, Boolean truthiness, and differentiation, with at least the direct-BFLOAT16,
  modulo-narrowing, saturation, signed-zero, and NaN examples above.<span data-glossary-navigation="true"> ([signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- The Compile API remains aligned with the already implemented floating-only reverse rule and now
  points to the Model-owned forward CAST value contract. The glossary updates the existing Cast
  expression entry from numerical conversion being deferred to Model-defined semantics whose
  executable support remains backend-specific.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [Cast expression glossary](../../../../glossary.md#cast-expression))</span>
- Existing Compiler tests prove floating-to-floating reverse casts the cotangent to the source
  type, same-type floating reverse may reuse the incoming cotangent, and any cast involving an
  integral or [`BOOL`](../../../../glossary.md#data-type--datatype) role has no [cotangent](../../../../glossary.md#cotangent). [Compiler](../../../../glossary.md#term-compiler) production source remains unchanged.
- The separate documentation-focused pass finalizes affected Javadocs, API text, examples, links,
  and glossary impact in the same overall change.
- [Model](../../../../glossary.md#term-model) 0025L becomes `Complete` only after implementation, Model tests, Javadoc, documentation
  validation, and completion evidence pass. CPU 0008K remains <span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [validation glossary](../../../../glossary.md#term-validation); [CPU glossary](../../../../glossary.md#term-cpu))</span>`Draft` and becomes dependent on
  completed Model 0025L; CPU 0008K–0008P retain their exact relative order and scope. Model 0026
  remains separate future <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>[`FLOAT16`](../../../../glossary.md#term-float16)/mixed-[precision](../../../../glossary.md#term-precision) work without a detailed specification.

## Tests / [validation](../../../../glossary.md#term-validation)

Implementation pass:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.elementwise.cast.CastValueConversionsTest --tests io.github.pho001.synaptik.model.operation.elementwise.cast.CastSemanticsTest --tests io.github.pho001.synaptik.model.datatype.BFloat16BitsTest --tests io.github.pho001.synaptik.model.tensor.TensorCastExpressionTest
./gradlew :modules:model:test
./gradlew :modules:compiler:test --tests io.github.pho001.synaptik.compiler.GradientRulesTest --tests io.github.pho001.synaptik.compiler.AutogradPreflightTest --tests io.github.pho001.synaptik.compiler.FirstOrderGradientCoverageTest
```

The focused tests must use independent expected raw [constants](../../../../glossary.md#term-constant) and exact arithmetic/reasoning for
boundary generation; they must not compute expected results by calling the production [oracle](../../../../glossary.md#term-test-oracle) or
by chaining Java conversions in cases where direct [target](../../../../glossary.md#term-target) [rounding](../../../../glossary.md#term-rounding) is required.

Documentation pass:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

Also perform targeted checks for:

- task/master/roadmap status and `0025K -> 0025L -> 0026` ordering;
- [CPU](../../../../glossary.md#term-cpu) `0008J -> 0008K -> 0008L -> 0008M -> 0008N -> 0008O -> 0008P` preservation and CPU 0008K's
  exact dependency on completed Model 0025L;
- all local Markdown targets and heading anchors changed by this task;
- balanced Markdown fences and final newlines;
- exact changed-path membership, the 14-path ceiling, and absence of Java changes outside [Model](../../../../glossary.md#term-model);
- exact 36 [matrix](../../../../glossary.md#term-matrix-meanings) cells, three [canonical](../../../../glossary.md#term-canonical-form) NaNs, three [widening](../../../../glossary.md#term-widening) mappings, and required edge examples;
- no change to architecture/build/[dependency](../../../../glossary.md#term-dependency) rules, `DataTypePromotion`, Compiler behavior, CPU
  execution, or Model 0026 scope.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>

Repository-wide Java validation is deferred to CPU 0009/[CI](../../../../glossary.md#term-continuous-integration-ci) because this task changes one Model
semantic family and no dependency, architecture, build, [Compiler](../../../../glossary.md#term-compiler), or [backend](../../../../glossary.md#backend) source. The focused
Compiler command is the concrete preservation check for the already implemented reverse boundary;
it does not authorize a Compiler edit. No [JMH](../../../../glossary.md#term-java-microbenchmark-harness-jmh),
performance, generated-Class-File, backend-[conformance](../../../../glossary.md#term-conformance), integration, or CPU test is required here.

## Dependencies

- Project-owner policy in this task — approved and complete; it resolves all conversion-policy
  questions that previously blocked Model meaning and CPU execution.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- [Model](../../../../glossary.md#term-model) 0001 and 0003A — complete; supply the six data types and [BFLOAT16](../../../../glossary.md#term-bfloat16) contracts/package.
- Model 0015G and 0015H — complete; supply `CAST`, attributes, all-pair expression construction,
  and current gradient eligibility.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient))</span>
- Model 0018N — complete; supplies exact [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) representation for all six types.
- Model 0018U — complete; supplies signed integral/[two's-complement](../../../../glossary.md#arithmetic-rearrangement-and-finite-difference-checks) conventions.
- Model 0025K — complete; immediately precedes this authorized interleave.
- [Compiler](../../../../glossary.md#term-compiler) 0005A/0005E — complete; supply and audit the preserved cast differentiation boundary.
- CPU 0008J — complete; supplies exact current BFLOAT16 scalar operation boundaries before CPU
  cast execution.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [operation glossary](../../../../glossary.md#operation))</span>

## Follow-up tasks

- CPU 0008K remains the next Draft CPU implementation frontier. Only after this task is Complete
  may its detailed specification be created; it owns scalar/parallel-scalar backend execution and
  must implement direct unboxed conversion logic rather than call the Model oracle per element.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [oracle glossary](../../../../glossary.md#term-test-oracle); [element glossary](../../../../glossary.md#term-element))</span>
- [CPU](../../../../glossary.md#term-cpu) 0008L–0008P retain their existing order and [scope](../../../../glossary.md#term-scope) after 0008K.
- [Model](../../../../glossary.md#term-model) 0026 remains the separate future [owner](../../../../glossary.md#term-owner) of true [IEEE](../../../../glossary.md#term-ieee)-754 [binary16](../../../../glossary.md#floating-point-precision-and-rounding) [`FLOAT16`](../../../../glossary.md#term-float16) and operation-
  family mixed-precision contracts. It must not reinterpret this task as adding binary16.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [precision glossary](../../../../glossary.md#term-precision))</span>
- [Backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance) CAST [vectors](../../../../glossary.md#term-vector) may consume `CastValueConversions` in test/cold code when the
  first executable backend task establishes that suite; this task does not create that suite.

## Architecture impact

Expected impact: None.

This task fills in [Model](../../../../glossary.md#term-model)-owned [operation](../../../../glossary.md#operation) semantics within existing packages and [dependency](../../../../glossary.md#term-dependency) rules.
It adds no architecture [owner](../../../../glossary.md#term-owner), [lifecycle](../../../../glossary.md#lifecycle) phase, [module](../../../../glossary.md#term-module) dependency, execution contract, or [backend](../../../../glossary.md#backend)
support. If implementation requires any architecture or dependency change, stop and report it.

Required no-change conclusions:

- `ARCHITECTURE.md`, focused architecture pages, ADRs, and architecture tests remain unchanged
  because ownership and dependency direction do not change.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools))</span>
- [`DataType`](../../../../glossary.md#data-type--datatype), `DataTypePromotion`, and Model capabilities remain unchanged because all six types
  and all-pair CAST construction already exist; this task defines values rather than adding a type,
  promotion, expression, or executable capability.
- `BFloat16Bits` remains unchanged because its exact BFLOAT16-to-FLOAT32 expansion and canonical
  FLOAT32-to-BFLOAT16 narrowing are incorporated, not replaced; direct FLOAT64-to-BFLOAT16 belongs
  to the new CAST reference.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT32 glossary](../../../../glossary.md#term-float32); [canonical glossary](../../../../glossary.md#term-canonical-form); [narrowing glossary](../../../../glossary.md#term-narrowing); [FLOAT64 glossary](../../../../glossary.md#term-float64))</span>
- Compiler production and Training API remain unchanged because the current floating-only
  cotangent policy is preserved rather than redesigned.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [cotangent glossary](../../../../glossary.md#cotangent))</span>
- CPU/backend source, backend-conformance tests, and integration tests remain unchanged because no
  backend truthfully executes this new contract until CPU 0008K or a later backend task.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Gradle and Java toolchain configuration remain unchanged because the task adds no dependency,
  source set, module, or language-level requirement.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [toolchain glossary](../../../../glossary.md#term-toolchain))</span>

The implementation and documentation passes must record these conclusions again against the final
diff. If the final implementation invalidates one, stop instead of silently [widening](../../../../glossary.md#term-widening) the task.

## Implementation prompt

Use this prompt in a separate agentic task/thread:

```text
You are the clean-context implementation agent for Synaptik Model task 0025L.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, and
docs/planning/modules/model/tasks/0025l-cross-type-cast-conversion-semantics.md in full. Read the
directly referenced current source/tests and implement exactly that task. Do not broaden scope or
implement CPU/backend execution. Stop and report any architecture, semantic, path-budget, or
direct-rounding conflict instead of inventing policy. Do not commit, push, stage, reset, or alter
unrelated work.

After implementation and the specified Model tests pass, hand the exact diff and test evidence to
a separate clean-context documentation-focused agent. That agent must follow
docs/developer-guide/documentation-rules.md, finalize the affected Javadocs, Tensor API, examples,
links, glossary impact, planning synchronization, and documentation validation without rerunning
successful Java tests unless executable Java changes afterward or a concrete risk requires it.

Record local decisions, limitations, all validation evidence, implementation notes, completion
summary, and the required task status in the task. Do not mark Complete until every acceptance
criterion and the documentation pass are complete.
```

## Local decisions

- The owner-approved policy fixes positive canonical quiet NaNs for lossy narrowing and
  left-aligned fraction mapping for lossless widening; no implementation discretion remains over
  NaN sign/payload behavior.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [canonical glossary](../../../../glossary.md#term-canonical-form); [narrowing glossary](../../../../glossary.md#term-narrowing); [widening glossary](../../../../glossary.md#term-widening); [NaN glossary](../../../../glossary.md#term-nan); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication))</span>
- A single public `CastValueConversions` scalar reference is justified by the current 30-pair
  semantic test matrix and immediate CPU conformance dependency. A test-only duplicate would not
  provide a shared oracle, while backend use in an element loop would violate the performance and
  ownership boundary.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-review-matrix))</span><span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [CPU glossary](../../../../glossary.md#term-cpu); [conformance glossary](../../../../glossary.md#term-conformance); [dependency glossary](../../../../glossary.md#term-dependency); [oracle glossary](../../../../glossary.md#term-test-oracle); [backend glossary](../../../../glossary.md#backend); [element glossary](../../../../glossary.md#term-element); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- The reference consumes and returns [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue), reusing the complete current typed raw-value
  carrier rather than introducing a union, carrier map, or overload per source/target pair.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [target glossary](../../../../glossary.md#term-target))</span>
- Conversion is explicit CAST meaning and does not alter `DataTypePromotion` or authorize implicit
  cross-category arithmetic.

## Known limitations

- No backend executes the newly defined cross-type semantics until an implementation task such as
  CPU 0008K truthfully advertises and lowers them.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- The scalar oracle is not a bulk conversion API, Tensor evaluator, storage adapter, or performance
  implementation.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [oracle glossary](../../../../glossary.md#term-test-oracle); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Tensor glossary](../../../../glossary.md#tensor))</span>
- [`FLOAT16`](../../../../glossary.md#term-float16) and non-cast mixed-[precision](../../../../glossary.md#term-precision) semantics remain deferred exclusively to [Model](../../../../glossary.md#term-model) 0026.

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation context ran the focused Model command from this task after executable code and
  tests stabilized: 32 tests passed. During development, one mistyped FLOAT32 finite-overflow test
  input constant was corrected to the intended boundary value; this was test-fixture correction,
  not a product failure.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [FLOAT32 glossary](../../../../glossary.md#term-float32); [finite glossary](../../../../glossary.md#term-finite); [overflow glossary](../../../../glossary.md#term-overflow); [constant glossary](../../../../glossary.md#term-constant); [fixture glossary](../../../../glossary.md#term-fixture))</span>
- Implementation context ran `./gradlew :modules:model:test`: 1,114 tests passed.
- Implementation context ran the focused Compiler command from this task: 42 tests passed,
  preserving floating-to-floating cast-back, same-type cotangent reuse, and the no-cotangent
  boundary for every cast involving an integral or BOOL role.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [cotangent glossary](../../../../glossary.md#cotangent); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- Implementation context ran `git diff --check`: passed.
- One attempted non-Gradle shell-wrapper invocation failed because Java was absent from that
  wrapper's <span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle))</span>`PATH`. The Gradle validation commands above completed successfully in the repository
  environment, so this environmental diagnostic is not a product or test failure.
- Clean-context documentation finalization agent for task 0025L independently reviewed the final
  implementation diff, `CastValueConversions`, all affected CAST Javadocs, the focused Model test,
  Tensor/Compile APIs, glossary, Compiler CAST-gradient contracts, <span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [gradient glossary](../../../../glossary.md#term-gradient))</span>`BFloat16Bits`, [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue),
  planning status/dependencies, and required no-change areas. No executable Java behavior or test
  was changed, so the successful implementation test evidence remained current and was not rerun.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation-agent validation: `./gradlew :modules:model:javadoc` passed; targeted local
  Markdown target/anchor, balanced-fence, trailing-whitespace/final-newline, exact matrix/example,
  status/order/dependency, 13-path membership/ceiling, and Java-scope checks passed; final<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [dependency glossary](../../../../glossary.md#term-dependency); [scope glossary](../../../../glossary.md#term-scope))</span>
  `git diff --check` passed; final `git status --short` showed exactly the 13 task paths.
- The first ad hoc Ruby link-check invocation used `Array.filter_map`, which is unavailable in the
  repository shell's Ruby version. The compatible `map.compact` form then checked the same targets
  and anchors successfully; this was checker portability, not a documentation or product failure.
- Package placement remains exactly as planned: the new public scalar reference and its test are
  in <span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>`io.github.pho001.synaptik.model.operation.elementwise.cast`; no package was added.
- Repository-wide Java validation remains deferred to CPU 0009/CI as specified. No JMH,
  generated-Class-File, backend-conformance, integration, or CPU execution validation applies to
  this Model semantic task.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [CI glossary](../../../../glossary.md#term-continuous-integration-ci); [JMH glossary](../../../../glossary.md#term-java-microbenchmark-harness-jmh); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Implementation notes

- Added the public final stateless `CastValueConversions` scalar oracle with the exact one-method
  API and deterministic source-then-target null checks. Same-type calls return the exact source;
  all cross-type calls return an exact target-typed <span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [oracle glossary](../../../../glossary.md#term-test-oracle); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [target glossary](../../../../glossary.md#term-target))</span>[`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue).
- Implemented all 36 ordered conversions, including direct binary64/integer-to-BFLOAT16 rounding,
  exact same-type bits, canonical lossy NaNs, left-aligned widening NaNs, signed zero, gradual
  underflow, signed infinity, floating-to-integral truncation/saturation, integral sign extension
  and modulo narrowing, and canonical BOOL conversion/truthiness.<span data-glossary-navigation="true"> ([binary64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [rounding glossary](../../../../glossary.md#term-rounding); [canonical glossary](../../../../glossary.md#term-canonical-form); [widening glossary](../../../../glossary.md#term-widening); [signed zero glossary](../../../../glossary.md#term-signed-zero); [gradual underflow glossary](../../../../glossary.md#term-gradual-underflow); [infinity glossary](../../../../glossary.md#term-infinity); [truncation glossary](../../../../glossary.md#term-truncation-toward-zero); [saturation glossary](../../../../glossary.md#term-saturation); [sign extension glossary](../../../../glossary.md#term-sign-extension); [narrowing glossary](../../../../glossary.md#term-narrowing); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- Finalized the affected CAST Javadocs and Tensor/Compile/glossary documentation while preserving
  the boundary that Tensor construction records but does not evaluate values and backend support
  remains separately advertised.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [backend glossary](../../../../glossary.md#backend))</span>
- No-change conclusions against the final diff: architecture contracts/pages, ADRs, and
  architecture tests remain unchanged because ownership and dependencies did not change;<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
  [`DataType`](../../../../glossary.md#data-type--datatype), `DataTypePromotion`, and Model capabilities remain unchanged because no type,
  promotion, expression family, or backend capability was added; <span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [backend capability glossary](../../../../glossary.md#backend-capability))</span>`BFloat16Bits` and `ScalarValue`
  remain unchanged because their exact carriers already support the policy; Compiler production
  and Training API remain unchanged because the existing floating-only reverse rule was only
  documented; backend source, conformance tests, and integration tests remain unchanged because
  execution is deferred to CPU 0008K; Gradle remains unchanged because no build or toolchain
  requirement changed; and no other module required modification.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [conformance glossary](../../../../glossary.md#term-conformance); [CPU glossary](../../../../glossary.md#term-cpu); [Gradle glossary](../../../../glossary.md#term-gradle); [toolchain glossary](../../../../glossary.md#term-toolchain); [module glossary](../../../../glossary.md#term-module))</span>

## Completion summary

- Completed changes: implemented the exact backend-independent 36-pair CAST value contract and
  scalar oracle; finalized affected Javadocs, API/glossary documentation, planning status, CPU
  dependency wording, validation evidence, and no-change conclusions.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [scalar glossary](../../../../glossary.md#term-scalar); [oracle glossary](../../../../glossary.md#term-test-oracle); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning); [CPU glossary](../../../../glossary.md#term-cpu); [dependency glossary](../../../../glossary.md#term-dependency); [validation glossary](../../../../glossary.md#term-validation))</span>
- Files changed or created: `CastValueConversions.java`, `CastKind.java`, `CastAttrs.java`,
  `TensorCastExpressions.java`, `Tensor.java`, `CastValueConversionsTest.java`, this task,
  `docs/api/tensor-api.md`, `docs/api/compile-api.md`, `docs/glossary.md`, the Model master plan,
  the CPU master plan, and the roadmap (13 paths total).<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- Tests and validation: reused the implementation agent's passing 32-test focused Model,
  1,114-test full Model, 42-test focused Compiler, and <span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>`git diff --check` evidence; the independent
  documentation pass then passed Model Javadoc and all requested documentation, status, order,
  dependency, scope, path-count, whitespace, and final-diff checks without changing executable
  Java or tests.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: complete in the mandatory separate clean context.
- Documentation impact: Tensor and Compile API references now state the exact conversion and
  differentiation boundaries without claiming Tensor evaluation or backend execution.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- Javadoc review: all new and affected CAST contracts meaningfully document purpose, semantics,
  inputs, results, failures, mutation/identity behavior, and ownership boundaries.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Glossary impact: the existing Cast expression entry now names the Model-owned value policy and
  keeps backend support and execution separate; no invented term was added.<span data-glossary-navigation="true"> ([Cast expression glossary](../../../../glossary.md#cast-expression))</span>
- Unresolved issues: None.
- Follow-up required: CPU 0008K remains the next Draft CPU implementation frontier; Model 0026
  remains separate future Draft FLOAT16/mixed-precision work.<span data-glossary-navigation="true"> ([FLOAT16 glossary](../../../../glossary.md#term-float16); [precision glossary](../../../../glossary.md#term-precision))</span>

Status: Complete
