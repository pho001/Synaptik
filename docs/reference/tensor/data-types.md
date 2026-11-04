# Data types and exact scalars

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

**Reader model.** An element's numeric format determines which values fit and how they are encoded.
Signed integer formats store whole numbers; floating-point formats store a sign, an exponent
(scale), and a significand (precision). Fewer bits generally mean fewer representable values.
FP8 means eight-bit floating point; E4M3 and E5M2 name different allocations of exponent and
fraction bits, not interchangeable encodings. BOOL represents false or true.

**Rounding** chooses a representable value when the exact result does not fit.
**Ties-to-even** chooses the candidate whose retained least-significant bit is zero when two
candidates are equally near. **Narrowing** converts to a less expressive format;
**promotion** chooses the common type an operation uses, without itself converting stored data.
An **accumulator** holds intermediate totals, which may use a wider format than the final output.
NaN means “not a number”; infinity and signed zeros are separate floating-point representations.
The detailed format and backend restrictions below still apply.

**Reading numerical edge cases.** Floating overflow exceeds a format's finite range;
underflow reaches its very small-magnitude range, potentially rounding to zero.
A **subnormal** represents values closer to zero than the smallest normal value, with less
relative precision. Denormals-are-zero (**DAZ**) treats such an input as signed zero at a permitted
arithmetic site; flush-to-zero (**FTZ**) does so for a permitted result. These are restricted
permissions below, not universal behavior or permission to change stored comparisons.
A NaN **payload** is extra information in its bit pattern; a **quiet** NaN and a **signaling** NaN
have different encodings. Preserving a numeric class is weaker than preserving every bit.
**Two's complement** is the signed integer encoding in which fixed-width arithmetic can wrap
modulo a power of two; that numeric wrapping differs from checked shape arithmetic throwing.

**Fused multiply-add** combines a product and an addition with one final rounding instead of
rounding the product separately. **Reassociation** changes the grouping of a sequence of
operations. Mathematical equality alone does not authorize either transformation: the applicable
operation's exact numerical rules and route restrictions still decide what is permitted.

<a id="data-types"></a>
## Data types

The public data type contracts live in `io.github.pho001.synaptik.model.datatype`. The current
`DataType` model contains these backend-independent element types in append-only ordinal order:

| Data type | Category | Bit width | Byte width | Differentiable |
|---|---|---:|---:|---|
| `FLOAT64` | `FLOATING` | 64 | 8 | yes |
| `FLOAT32` | `FLOATING` | 32 | 4 | yes |
| `BFLOAT16` | `FLOATING` | 16 | 2 | yes |
| `INT32` | `INTEGRAL` | 32 | 4 | no |
| `INT64` | `INTEGRAL` | 64 | 8 | no |
| `BOOL` | `BOOLEAN` | 8 | 1 | no |
| `FLOAT16` | `FLOATING` | 16 | 2 | yes |
| `FLOAT8_E4M3` | `FLOATING` | 8 | 1 | yes |
| `FLOAT8_E5M2` | `FLOATING` | 8 | 1 | yes |

`FLOAT32` is the default floating data type. Data type metadata describes logical model semantics and does not claim that a particular backend supports the type or uses the same physical allocation alignment.

Homogeneous low arithmetic uses FLOAT32 working values and accumulators with one final
ties-to-even narrowing at each declared ordinary operation, saved or public result boundary.
Exact raw movement, casts, sign operations, stored comparisons, selection, masks, indices and
state remain exact. An E4M3 infinity-class result, including a pole or empty floating extremum,
narrows to NaN; nonempty extrema select represented input winners without an infinity sentinel
poisoning the result. CPU reuses existing operation/shared Compiler AD/Scan/recurrent owners
under their family, geometry and actual derivative guards. Native availability does not define
CPU support: oneDNN is an occurrence-specific typed-input realization with strict FLOAT32 low
working arithmetic, not converted-FLOAT32 input advertised as native FP8. FLOAT64 has no native
oneDNN route. Metal retains its independent seven-carrier/no-FP8 scope.

<a id="numeric-promotion"></a>
### Numeric promotion

`DataTypePromotion` exposes two related contracts. `promoteFloating(left, right)` treats
`BFLOAT16`, `FLOAT16`, `FLOAT8_E4M3` and `FLOAT8_E5M2` as distinct low formats: a homogeneous
low pair retains its type, unlike low pairs promote to FLOAT32, FLOAT32 wins over every low
type, and FLOAT64 wins over every other floating type.

`promoteNumeric(left, right)` accepts a pair only when both operands belong to that floating
category or both belong to the signed-integral hierarchy:

```text
INT32 < INT64
```

For integral pairs, INT64 wins when either operand is INT64; an INT32 operand then participates in
the signed INT64 domain by conceptual sign extension. Same-width pairs retain their type. BOOL,
null, and mixed floating/integral pairs are rejected. The implemented `Tensor.cast` method can
make a category conversion explicit, but it constructs expression metadata rather than converting
values; promotion itself never inserts a cast or evaluates a value.

For example:

```java
DataType result = DataTypePromotion.promoteFloating(
        DataType.BFLOAT16, DataType.FLOAT64);
```

The first input has 16 logical bits and the second has 64, so the result is `FLOAT64`. A
`promoteFloating(FLOAT16, BFLOAT16)` call returns `FLOAT32`. Similarly,
`promoteNumeric(INT32, INT64)` returns `INT64`. These are model-level type decisions; they do not
prove evaluated values, compiler behavior, or backend support.

<a id="sixteen-bit-floating-representations"></a>
### Sixteen-bit floating representations

`BFloat16Bits` converts scalar values between Java `float` and raw BFLOAT16 bits held in a `short`;
`Float16Bits` does the same for IEEE-754 binary16. Conversion uses round-to-nearest with ties to
even, preserves signed zero and infinities, and canonicalizes NaN to the target format's positive
quiet pattern. Each utility describes value format only; neither allocates storage, exposes device
formats, or reports backend capabilities.

`Float8E4M3Bits` and `Float8E5M2Bits` expose `toFloat(byte)`, `fromFloat(float)` and
`fromDouble(double)` for unscaled OFP8. Encoding returns raw bytes with non-saturating
round-to-nearest, ties-to-even; `fromDouble` rounds directly, without a FLOAT32 intermediate.
E4M3 has bias 7, maximum 448 and no infinity: magnitude 464 ties to 448, larger magnitudes
and infinity input become canonical NaN `0x7f`. E5M2 has bias 15, maximum 57344 and signed
infinities `0x7c`/`0xfc`; magnitude at or above 61440 rounds to signed infinity, and numeric
NaN becomes `0x7e`. Both formats have signed zeros and gradual subnormals. Raw decoding
accepts all 256 patterns and preserves represented NaN sign/fraction bits. There is no scale,
clipping, calibration or automatic loss scaling.

<a id="exact-typed-scalar-values"></a>
### Exact typed scalar values

`ScalarValue` is an immutable operation parameter consisting of one exact `DataType` and that
type's primitive bits. It is not a scalar Tensor: a rank-zero Tensor has a descriptor, identity,
and potentially storage, while `ScalarValue` owns none of those. It is also not a conversion
request or executable constant.

Named factories make the representation explicit:

```java
ScalarValue floatValue = ScalarValue.float32(0.5f);
ScalarValue rawBfloatNaN = ScalarValue.bfloat16Bits((short) 0x7FC1);
ScalarValue rawFloat16NaN = ScalarValue.float16Bits((short) 0x7E01);
ScalarValue largeInteger = ScalarValue.int64(9_007_199_254_740_993L);
ScalarValue falseValue = ScalarValue.bool(false);
```

`float64` and `float32` retain raw IEEE-754 bits, including the sign of zero and distinct NaN
payloads. `bfloat16Bits` and `float16Bits` retain every supplied 16-bit pattern without
canonicalizing NaN; `bfloat16(float)` and `float16(float)` are the separately named binary32-to-
16-bit conversions. `int32` and `int64` retain exact two's-complement values, so the INT64 example
remains exact even though it is above `2^53`. `bool` stores only canonical false or true.
Equality and hashing use the data type and exact bits.
`float8E4m3(float)` and `float8E5m2(float)` perform numeric encoding;
`float8E4m3Bits(byte)` / `float8E5m2Bits(byte)` and the corresponding instance accessors
preserve every raw byte, including NaNs. A mismatched accessor throws rather than converting.

Each inspector is strict. For example, calling `float32Value()` on an INT64 value throws
`IllegalStateException` rather than narrowing or interpreting the bits. There is no general
conversion method. Scalar/clamp Tensor expressions require a matching floating value, while
constant padding accepts every Model data type when the value exactly matches the receiver:

```java
Tensor scaled = float32Input.mul(ScalarValue.float32(0.5f));
Tensor padded = boolInput.pad(before, after, ScalarValue.bool(false));
```

The primitive `TensorFactory.scalar`, `scalarBFloat16`, `scalarFloat16`,
`scalarFloat8E4m3`, `scalarFloat8E5m2`, `full`, `fullBFloat16`, `fullFloat16`,
`fullFloat8E4m3`, and `fullFloat8E5m2` methods create eager scalar or dense Tensor storage through
their exact primitive signatures; this API does not add a `ScalarValue` factory overload or
materialize a `ScalarValue` as a Tensor.


---

[Next topic](shapes.md)
