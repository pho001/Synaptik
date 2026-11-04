# Data types and exact [scalars](../../glossary.md#term-scalar)

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

**[Reader model](../../glossary.md#term-reader-model).** An [element](../../glossary.md#term-element)'s numeric format determines which values fit and how they are encoded.
Signed integer formats store whole numbers; [floating-point](../../glossary.md#term-floating-point) formats store a sign, an [exponent](../../glossary.md#term-exponent)
(scale), and a [significand](../../glossary.md#term-significand) ([precision](../../glossary.md#term-precision)). Fewer bits generally [mean](../../glossary.md#term-mean) fewer representable values.
[FP8](../../glossary.md#term-fp8) means [eight-bit floating point](../../glossary.md#floating-point-precision-and-rounding); [E4M3](../../glossary.md#term-e4m3) and [E5M2](../../glossary.md#term-e5m2) name different allocations of exponent and
fraction bits, not interchangeable encodings. [BOOL](../../glossary.md#data-type--datatype) represents false or true.

**[Rounding](../../glossary.md#term-rounding)** chooses a representable value when the exact result does not fit.
**[Ties-to-even](../../glossary.md#term-ties-to-even)** chooses the candidate whose retained least-significant bit is zero when two
candidates are equally near. **[Narrowing](../../glossary.md#term-narrowing)** converts to a less expressive format;
**promotion** chooses the common type an [operation](../../glossary.md#operation) uses, without itself converting stored data.
An **[accumulator](../../glossary.md#accumulator-type)** holds intermediate totals, which may use a wider format than the final output.
[NaN](../../glossary.md#term-nan) means “[not a number](../../glossary.md#nan-infinity-and-signed-zero)”; [infinity](../../glossary.md#term-infinity) and signed zeros are separate floating-point representations.
The detailed format and [backend](../../glossary.md#backend) restrictions below still apply.

**Reading numerical edge cases.** Floating [overflow](../../glossary.md#term-overflow) exceeds a format's [finite](../../glossary.md#term-finite) range;
[underflow](../../glossary.md#term-underflow) reaches its very small-magnitude range, potentially rounding to zero.
A **[subnormal](../../glossary.md#term-subnormal)** represents values closer to zero than the smallest normal value, with less
relative precision. [Denormals-are-zero](../../glossary.md#subnormal-daz-ftz-overflow-and-underflow) (**[DAZ](../../glossary.md#term-daz)**) treats such an input as [signed zero](../../glossary.md#term-signed-zero) at a permitted
arithmetic site; [flush-to-zero](../../glossary.md#subnormal-daz-ftz-overflow-and-underflow) (**[FTZ](../../glossary.md#term-ftz)**) does so for a permitted result. These are restricted
permissions below, not universal behavior or permission to change stored comparisons.
A NaN **[payload](../../glossary.md#artifact-envelope-and-atomic-publication)** is extra information in its bit pattern; a **quiet** NaN and a **signaling** NaN
have different encodings. Preserving a numeric class is weaker than preserving every bit.
**Two's complement** is the signed integer encoding in which fixed-width arithmetic can wrap
modulo a power of two; that numeric wrapping differs from checked [shape](../../glossary.md#shape) arithmetic throwing.

**[Fused multiply-add](../../glossary.md#term-fused-multiply-add-fma)** combines a product and an addition with one final rounding instead of
rounding the product separately. **[Reassociation](../../glossary.md#term-reassociation)** changes the grouping of a sequence of
operations. Mathematical equality alone does not authorize either transformation: the applicable
operation's exact numerical rules and route restrictions still decide what is permitted.

<a id="data-types"></a>
## Data types

The public [data type](../../glossary.md#data-type--datatype) contracts live in `io.github.pho001.synaptik.model.datatype`. The current
[`DataType`](../../glossary.md#data-type--datatype) [model](../../glossary.md#term-model) contains these [backend](../../glossary.md#backend)-independent [element](../../glossary.md#term-element) types in append-only ordinal order:

| Data type | Category | Bit width | Byte width | [Differentiable](../../glossary.md#derivative-and-chain-rule) |
|---|---|---:|---:|---|
| [`FLOAT64`](../../glossary.md#term-float64) | `FLOATING` | 64 | 8 | yes |
| [`FLOAT32`](../../glossary.md#term-float32) | `FLOATING` | 32 | 4 | yes |
| [`BFLOAT16`](../../glossary.md#term-bfloat16) | `FLOATING` | 16 | 2 | yes |
| [`INT32`](../../glossary.md#data-type--datatype) | `INTEGRAL` | 32 | 4 | no |
| [`INT64`](../../glossary.md#data-type--datatype) | `INTEGRAL` | 64 | 8 | no |
| [`BOOL`](../../glossary.md#data-type--datatype) | [`BOOLEAN`](../../glossary.md#data-type--datatype) | 8 | 1 | no |
| [`FLOAT16`](../../glossary.md#term-float16) | `FLOATING` | 16 | 2 | yes |
| `FLOAT8_E4M3` | `FLOATING` | 8 | 1 | yes |
| `FLOAT8_E5M2` | `FLOATING` | 8 | 1 | yes |

`FLOAT32` is the default floating data type. Data type [metadata](../../glossary.md#term-metadata) describes logical model semantics and does not claim that a particular backend supports the type or uses the same physical allocation [alignment](../../glossary.md#term-alignment).

[Homogeneous](../../glossary.md#term-homogeneous) low arithmetic uses FLOAT32 working values and accumulators with one final
[ties-to-even](../../glossary.md#term-ties-to-even) [narrowing](../../glossary.md#term-narrowing) at each declared ordinary [operation](../../glossary.md#operation), saved or public result boundary.
Exact raw movement, casts, sign operations, stored comparisons, selection, [masks](../../glossary.md#term-boolean-mask), indices and
state remain exact. An [E4M3](../../glossary.md#term-e4m3) [infinity](../../glossary.md#term-infinity)-class result, including a [pole](../../glossary.md#term-pole) or empty floating [extremum](../../glossary.md#term-extremum),
narrows to [NaN](../../glossary.md#term-nan); nonempty [extrema](../../glossary.md#saturation-pole-and-extremum) select represented input winners without an infinity sentinel
poisoning the result. [CPU](../../glossary.md#term-cpu) reuses existing operation/shared [Compiler](../../glossary.md#term-compiler) [AD](../../glossary.md#autograd)/[Scan](../../glossary.md#term-cumulative-scan)/recurrent [owners](../../glossary.md#term-owner)
under their family, geometry and actual [derivative](../../glossary.md#term-derivative) guards. Native availability does not define
CPU support: [oneDNN](../../glossary.md#term-onednn) is an [occurrence](../../glossary.md#term-occurrence)-specific typed-input realization with strict FLOAT32 low
[working arithmetic](../../glossary.md#accumulator-type), not converted-FLOAT32 input advertised as native [FP8](../../glossary.md#term-fp8). FLOAT64 has no native
oneDNN route. [Metal](../../glossary.md#term-metal) retains its independent seven-[carrier](../../glossary.md#term-storage-carrier)/no-FP8 [scope](../../glossary.md#term-scope).

<a id="numeric-promotion"></a>
### [Numeric promotion](../../glossary.md#numeric-promotion)

`DataTypePromotion` exposes two related contracts. `promoteFloating(left, right)` treats
[`BFLOAT16`](../../glossary.md#term-bfloat16), [`FLOAT16`](../../glossary.md#term-float16), `FLOAT8_E4M3` and `FLOAT8_E5M2` as distinct low formats: a [homogeneous](../../glossary.md#term-homogeneous)
low pair retains its type, unlike low pairs promote to [FLOAT32](../../glossary.md#term-float32), FLOAT32 wins over every low
type, and [FLOAT64](../../glossary.md#term-float64) wins over every other floating type.

`promoteNumeric(left, right)` accepts a pair only when both operands belong to that floating
category or both belong to the signed-integral hierarchy:

```text
INT32 < INT64
```

For integral pairs, [INT64](../../glossary.md#data-type--datatype) wins when either operand is INT64; an [INT32](../../glossary.md#data-type--datatype) operand then participates in
the signed INT64 domain by conceptual [sign extension](../../glossary.md#term-sign-extension). Same-width pairs retain their type. [BOOL](../../glossary.md#data-type--datatype),
null, and mixed floating/integral pairs are rejected. The implemented `Tensor.cast` method can
make a category conversion explicit, but it constructs expression [metadata](../../glossary.md#term-metadata) rather than converting
values; promotion itself never inserts a cast or evaluates a value.

For example:

```java
DataType result = DataTypePromotion.promoteFloating(
        DataType.BFLOAT16, DataType.FLOAT64);
```

The first input has 16 logical bits and the second has 64, so the result is `FLOAT64`. A
`promoteFloating(FLOAT16, BFLOAT16)` call returns `FLOAT32`. Similarly,
`promoteNumeric(INT32, INT64)` returns `INT64`. These are [model](../../glossary.md#term-model)-level type decisions; they do not
prove evaluated values, [compiler](../../glossary.md#term-compiler) behavior, or [backend](../../glossary.md#backend) support.

<a id="sixteen-bit-floating-representations"></a>
### Sixteen-bit floating representations

`BFloat16Bits` converts [scalar](../../glossary.md#term-scalar) values between Java `float` and raw [BFLOAT16](../../glossary.md#term-bfloat16) bits held in a `short`;
`Float16Bits` does the same for [IEEE](../../glossary.md#term-ieee)-754 [binary16](../../glossary.md#floating-point-precision-and-rounding). Conversion uses [round-to-nearest](../../glossary.md#floating-point-precision-and-rounding) with ties to
even, preserves [signed zero](../../glossary.md#term-signed-zero) and [infinities](../../glossary.md#nan-infinity-and-signed-zero), and canonicalizes [NaN](../../glossary.md#term-nan) to the [target](../../glossary.md#term-target) format's positive
quiet pattern. Each utility describes value format only; neither allocates storage, exposes [device](../../glossary.md#term-device)
formats, or reports [backend](../../glossary.md#backend) capabilities.

`Float8E4M3Bits` and `Float8E5M2Bits` expose `toFloat(byte)`, `fromFloat(float)` and
`fromDouble(double)` for unscaled [OFP8](../../glossary.md#term-ofp8). Encoding returns raw bytes with non-[saturating](../../glossary.md#saturation-pole-and-extremum)
[round-to-nearest, ties-to-even](../../glossary.md#floating-point-precision-and-rounding); `fromDouble` rounds directly, without a [FLOAT32](../../glossary.md#term-float32) intermediate.
[E4M3](../../glossary.md#term-e4m3) has [bias](../../glossary.md#term-bias) 7, maximum 448 and no [infinity](../../glossary.md#term-infinity): magnitude 464 ties to 448, larger magnitudes
and infinity input become [canonical NaN](../../glossary.md#term-canonical-nan) `0x7f`. [E5M2](../../glossary.md#term-e5m2) has bias 15, maximum 57344 and signed
infinities `0x7c`/`0xfc`; magnitude at or above 61440 rounds to signed infinity, and numeric
NaN becomes `0x7e`. Both formats have signed zeros and gradual [subnormals](../../glossary.md#term-subnormal). Raw decoding
accepts all 256 patterns and preserves represented NaN sign/fraction bits. There is no scale,
clipping, calibration or automatic [loss](../../glossary.md#loss) scaling.

<a id="exact-typed-scalar-values"></a>
### Exact typed [scalar](../../glossary.md#term-scalar) values

[`ScalarValue`](../../glossary.md#typed-scalar-value--scalarvalue) is an [immutable](../../glossary.md#term-immutable) [operation](../../glossary.md#operation) [parameter](../../glossary.md#term-parameter) consisting of one exact [`DataType`](../../glossary.md#data-type--datatype) and that
type's primitive bits. It is not a scalar [Tensor](../../glossary.md#tensor): a [rank-zero](../../glossary.md#axis-rank-scalar-vector-and-matrix) Tensor has a [descriptor](../../glossary.md#term-descriptor), [identity](../../glossary.md#term-identity-meanings),
and potentially storage, while `ScalarValue` owns none of those. It is also not a conversion
request or executable [constant](../../glossary.md#term-constant).

Named factories make the representation explicit:

```java
ScalarValue floatValue = ScalarValue.float32(0.5f);
ScalarValue rawBfloatNaN = ScalarValue.bfloat16Bits((short) 0x7FC1);
ScalarValue rawFloat16NaN = ScalarValue.float16Bits((short) 0x7E01);
ScalarValue largeInteger = ScalarValue.int64(9_007_199_254_740_993L);
ScalarValue falseValue = ScalarValue.bool(false);
```

`float64` and `float32` retain raw [IEEE](../../glossary.md#term-ieee)-754 bits, including the sign of zero and distinct [NaN](../../glossary.md#term-nan)
payloads. `bfloat16Bits` and `float16Bits` retain every supplied 16-bit pattern without
canonicalizing NaN; `bfloat16(float)` and `float16(float)` are the separately named [binary32](../../glossary.md#floating-point-precision-and-rounding)-to-
16-bit conversions. `int32` and `int64` retain exact [two's-complement](../../glossary.md#arithmetic-rearrangement-and-finite-difference-checks) values, so the [INT64](../../glossary.md#data-type--datatype) example
remains exact even though it is above `2^53`. `bool` stores only [canonical](../../glossary.md#term-canonical-form) false or true.
Equality and hashing use the [data type](../../glossary.md#data-type--datatype) and exact bits.
`float8E4m3(float)` and `float8E5m2(float)` perform numeric encoding;
`float8E4m3Bits(byte)` / `float8E5m2Bits(byte)` and the corresponding instance accessors
preserve every raw byte, including NaNs. A mismatched accessor throws rather than converting.

Each inspector is strict. For example, calling `float32Value()` on an INT64 value throws
`IllegalStateException` rather than [narrowing](../../glossary.md#term-narrowing) or interpreting the bits. There is no general
conversion method. Scalar/clamp Tensor expressions require a matching floating value, while
constant [padding](../../glossary.md#padding) accepts every [Model](../../glossary.md#term-model) data type when the value exactly matches the receiver:

```java
Tensor scaled = float32Input.mul(ScalarValue.float32(0.5f));
Tensor padded = boolInput.pad(before, after, ScalarValue.bool(false));
```

The primitive `TensorFactory.scalar`, `scalarBFloat16`, `scalarFloat16`,
`scalarFloat8E4m3`, `scalarFloat8E5m2`, `full`, `fullBFloat16`, `fullFloat16`,
`fullFloat8E4m3`, and `fullFloat8E5m2` methods create [eager](../../glossary.md#term-eager-evaluation) scalar or dense Tensor storage through
their exact primitive [signatures](../../glossary.md#term-signature); this [API](../../glossary.md#term-application-programming-interface-api) does not add a `ScalarValue` factory overload or
materialize a `ScalarValue` as a Tensor.


---

[Next topic](shapes.md)
