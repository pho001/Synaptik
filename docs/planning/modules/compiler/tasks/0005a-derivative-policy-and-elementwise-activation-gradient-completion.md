# Task 0005A: [Derivative](../../../../glossary.md#term-derivative) Policy and [Elementwise](../../../../glossary.md#term-elementwise)/[Activation](../../../../glossary.md#term-activation) [Gradient](../../../../glossary.md#term-gradient) Completion

> **Editorial reading key — not part of the historical record.**
>
> The new material in this task is a set of compiler-owned first-order derivative conventions, not
> new forward numerical semantics. Preflight selects exact operation/input roles; the elementwise
> owner builds ordinary Tensor expressions and typed constant facts. A formula-shaped result or a
> passing construction test does not demonstrate backend behavior at ties, infinities or
> singularities. Those generated operations must independently qualify when prepared and executed.
>
> Current reading: [pre-capture differentiation and functional requests](../../../../reference/compiler/autograd.md#current-package-private-pre-capture-autograd); [capability queries and hard-eligible owner selection](../../../../reference/compiler/planning.md#current-operation-capability-contracts).

## Status

Complete

## Goal

Complete first-order [automatic differentiation](../../../../glossary.md#autograd) for the current [elementwise](../../../../glossary.md#term-elementwise) and [activation](../../../../glossary.md#term-activation)
inventory inside the established [compiler](../../../../glossary.md#term-compiler)-owned, pre-[capture](../../../../glossary.md#term-capture-meanings) [Tensor](../../../../glossary.md#tensor)-expression pipeline.

This task preserves every implemented Compiler 0004–0004B rule and adds the missing floating
[binary](../../../../glossary.md#term-binary-descriptor-and-metadata)/[scalar](../../../../glossary.md#term-scalar) arithmetic, [extrema](../../../../glossary.md#saturation-pole-and-extremum), clamp, [unary](../../../../glossary.md#term-unary), and activation rules. It also closes the
differentiability classification for conditional selection, casts, comparisons, [Boolean](../../../../glossary.md#data-type--datatype) logic,
and floating classifications:

```text
selected forward occurrence + accumulated output cotangent
  -> exact local Tensor-expression contribution for each differentiable input
  -> reverse broadcasting and floating-type normalization
  -> deterministic request-local accumulation
  -> one immutable combined forward/backward capture
```

[Model](../../../../glossary.md#term-model) 0025A fixes forward represented-value comparison, extrema, and clamp meaning. This task
selects compiler-owned first-order [derivative](../../../../glossary.md#term-derivative) conventions; it does not change those forward
contracts or add model-owned derivative behavior.

## [Scope](../../../../glossary.md#term-scope)

### Exact [operation](../../../../glossary.md#operation) and role inventory

The implementation must derive and lock this inventory from the production enums and [signatures](../../../../glossary.md#term-signature),
not from this table alone. It covers exactly 48 current kind [constants](../../../../glossary.md#term-constant):

| Family | Exact kinds | [Differentiable](../../../../glossary.md#derivative-and-chain-rule) input roles in 0005A |
|---|---|---|
| `BinaryArithmeticKind` | `ADD`, `SUB`, `MUL`, `DIV`, `MIN`, `MAX`, `POW` | Floating left and right [Tensor](../../../../glossary.md#tensor) roles. Existing ADD/SUB/MUL/DIV rules remain; MIN/MAX/POW are added. |
| `ScalarElementwiseKind` | `ADD`, `SUB`, `MUL`, `DIV`, `MIN`, `MAX`, `POW`, `CLAMP` | The floating Tensor receiver only. [Scalar](../../../../glossary.md#term-scalar) attributes and clamp bounds are [immutable](../../../../glossary.md#term-immutable) non-differentiable configuration. Existing ADD/SUB/MUL/DIV rules remain; MIN/MAX/POW/CLAMP are added. |
| `WhereSelectionKind` | `WHERE` | Floating true and false branches. The [BOOL](../../../../glossary.md#data-type--datatype) condition is non-differentiable. Preserve the existing rule. |
| `CastKind` | `CAST` | Floating source only when the [target](../../../../glossary.md#term-target) is also floating. Preserve the existing same-type [identity](../../../../glossary.md#term-identity-meanings) and cross-floating [cotangent](../../../../glossary.md#cotangent) cast-back policy. All roles involving a BOOL or integral source or target are non-differentiable. |
| `UnaryElementwiseKind` | `ABS`, `NEG`, `RECIPROCAL`, [`LOG`](../../../../glossary.md#term-log), `LOG1P`, [`EXP`](../../../../glossary.md#term-expx), `EXPM1`, [`ERF`](../../../../glossary.md#term-erf), `SQRT`, `RSQRT`, `FLOOR`, `CEIL`, `SIGN`, `RELU`, [`SIGMOID`](../../../../glossary.md#term-sigmoid), [`TANH`](../../../../glossary.md#term-tanh), [`GELU`](../../../../glossary.md#gaussian-error-linear-unit--gelu), `GELU_TANH_APPROXIMATION`, `SILU` | The one floating input. Preserve NEG/EXP/EXPM1/ERF/FLOOR/CEIL/SIGN/SIGMOID/TANH and add the other ten rules. |
| `BinaryComparisonKind` | `GREATER_THAN`, `GREATER_OR_EQUAL`, `LESS_THAN`, `LESS_OR_EQUAL`, `EQUAL`, `NOT_EQUAL` | None. BOOL outputs and both numeric input roles are non-differentiable through the comparison [occurrence](../../../../glossary.md#term-occurrence). |
| `BooleanLogicalKind` | `AND`, `OR`, `NOT` | None. BOOL inputs and outputs are non-differentiable. |
| `FloatingClassificationKind` | `IS_FINITE`, `IS_NAN`, `IS_INF` | None. BOOL outputs and the classified floating input role are non-differentiable through the classification occurrence. |

Every assigned occurrence has exactly one [canonical](../../../../glossary.md#term-canonical-form) output at slot zero. [Binary](../../../../glossary.md#term-binary-descriptor-and-metadata)/scalar arithmetic,
WHERE, floating-to-floating CAST, and [unary](../../../../glossary.md#term-unary) occurrences have one floating output on a selected
route. Comparisons, [Boolean](../../../../glossary.md#data-type--datatype) logic, floating classifications, and casts to BOOL or integral types
have one non-differentiable output. No 0005A kind has a hidden or auxiliary output slot.

Comparison and classification expressions may appear as BOOL conditions inside generated
piecewise formulas. That use does not make their numeric inputs differentiable through those BOOL
operations.

### Notation and [cotangent normalization](../../../../glossary.md#cotangent-normalization)

For one output `y`, incoming [cotangent](../../../../glossary.md#cotangent) `g`, and selected input `x`:

- `Z_x`, `O_x`, `N_x`, and `H_x` are exact typed positive-zero, positive-one, negative-one, and
  one-half splats expanded to `x.shape`.
- `S_x(v)` is an exact typed [scalar](../../../../glossary.md#term-scalar)-value splat for [immutable](../../../../glossary.md#term-immutable) `ScalarValue v`, expanded to
  `x.shape`.
- `B_x(t)` restores `x.shape` by applying ordinary `sumToShape(x.shape)` when `t.shape` differs,
  then restores `x.type` by applying ordinary floating `cast(x.type)` when its type differs.
- A direct local result whose [Shape](../../../../glossary.md#shape) and type already equal `x` requires no redundant
  normalization expression.

Every [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) and broadcast branch contribution must end in `B_x`. Every returned input
contribution must have exactly the selected input [Tensor](../../../../glossary.md#tensor)'s Shape and floating [data type](../../../../glossary.md#data-type--datatype) before
ordinary deterministic accumulation. [Normalization](../../../../glossary.md#term-normalization) uses the existing public Tensor algebra and
the current [Compiler](../../../../glossary.md#term-compiler) 0002 [inference](../../../../glossary.md#term-inference)/[validation](../../../../glossary.md#term-validation) boundary; it is not a [gradient](../../../../glossary.md#term-gradient)-only shape, cast,
or arithmetic system.

### Exact local formulas

The formulas below are structural [Tensor](../../../../glossary.md#tensor)-expression contracts. `where` means public
`Tensor.where`, comparisons and classifications [mean](../../../../glossary.md#term-mean) the current public represented-value
operations, and [scalar](../../../../glossary.md#term-scalar) coefficients mean exact typed [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) [operation](../../../../glossary.md#operation) attributes unless a
splat is explicitly named.

#### [Binary](../../../../glossary.md#term-binary-descriptor-and-metadata) arithmetic

Preserve the existing rules:

| Kind | Left contribution | Right contribution |
|---|---|---|
| `ADD` | `B_left(g)` | `B_right(g)` |
| `SUB` | `B_left(g)` | `B_right(-g)` |
| `MUL` | `B_left(g * right)` | `B_right(g * left)` |
| `DIV` | `B_left(g / right)` | `B_right(-(g * left) / (right * right))` |

Add:

| Kind | Left contribution | Right contribution |
|---|---|---|
| `MIN` | `B_left(where(left < right, g, where(left == right, g * 0.5, Z_y)))` | `B_right(where(right < left, g, where(right == left, g * 0.5, Z_y)))` |
| `MAX` | `B_left(where(left > right, g, where(left == right, g * 0.5, Z_y)))` | `B_right(where(right > left, g, where(right == left, g * 0.5, Z_y)))` |
| `POW` | `B_left(g * right * pow(left, right - O_right))` | `B_right(g * y * log(left))` |

The [Tensor](../../../../glossary.md#tensor)/Tensor `POW` formula subtracts an ordinary exact typed one from the [exponent](../../../../glossary.md#term-exponent) Tensor.
It does not replace `pow(left, right - 1)` with `y / left`, because that changes the selected
zero-base boundary behavior.

#### [Scalar](../../../../glossary.md#term-scalar) arithmetic and first-class clamp

Preserve scalar `ADD -> g`, `SUB -> g`, `MUL -> g * value`, and `DIV -> g / value`.

For scalar MIN/MAX, let `b = S_x(attrs.value)`:

| Kind | Receiver contribution |
|---|---|
| `MIN` | `where(x < b, g, where(x == b, g * 0.5, Z_x))` |
| `MAX` | `where(x > b, g, where(x == b, g * 0.5, Z_x))` |
| `POW` | `g * exponent * pow(x, exponentMinusOne)` |

`exponentMinusOne` is one exact same-typed [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) produced at [compile](../../../../glossary.md#compile) time by subtracting
positive one in the represented type:

- BFLOAT16 expands the exact bits to binary32, subtracts binary32 one, and converts back with the
  existing round-to-nearest-ties-to-even <span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [round-to-nearest glossary](../../../../glossary.md#floating-point-precision-and-rounding); [ties-to-even glossary](../../../../glossary.md#term-ties-to-even))</span>`BFloat16Bits.fromFloat`;
- [FLOAT32](../../../../glossary.md#term-float32) performs one [binary32](../../../../glossary.md#floating-point-precision-and-rounding) subtraction; and
- [FLOAT64](../../../../glossary.md#term-float64) performs one [binary64](../../../../glossary.md#floating-point-precision-and-rounding) subtraction.

No [host](../../../../glossary.md#term-host) transcendental function is used, and the original [exponent](../../../../glossary.md#term-exponent) remains the exact multiplier
attribute.

For first-class `CLAMP`, preserve its ordered forward meaning instead of inventing an unrelated
endpoint rule:

```text
lowered = x.maximum(minValue)
upperRouted =
    where(lowered < S_x(maxValue), g,
          where(lowered == S_x(maxValue), g * 0.5, Z_x))
inputContribution =
    where(x > S_x(minValue), upperRouted,
          where(x == S_x(minValue), upperRouted * 0.5, Z_x))
```

This is the [chain rule](../../../../glossary.md#term-chain-rule) for the selected scalar-[extrema](../../../../glossary.md#saturation-pole-and-extremum) tie convention applied to the exact ordered
`MIN(MAX(x, minValue), maxValue)` value contract. It is expressed with ordinary public [Tensor](../../../../glossary.md#tensor)
operations; it does not add stored forward intermediates or decompose the original forward
[occurrence](../../../../glossary.md#term-occurrence).

#### [Unary](../../../../glossary.md#term-unary) arithmetic and [activations](../../../../glossary.md#term-activation)

Preserve:

| Kind | Input contribution |
|---|---|
| `NEG` | `-g` |
| [`EXP`](../../../../glossary.md#term-expx) | `g * y` |
| `EXPM1` | `g * (y + 1)` |
| [`ERF`](../../../../glossary.md#term-erf) | `g * exp(-(x * x)) * (2 / sqrt(pi))` with the existing fixed typed coefficient |
| `FLOOR`, `CEIL`, `SIGN` | `Z_x` |
| [`SIGMOID`](../../../../glossary.md#term-sigmoid) | `g * y * (1 - y)` |
| [`TANH`](../../../../glossary.md#term-tanh) | `g * (1 - y * y)` |

Add:

| Kind | Input contribution |
|---|---|
| `ABS` | `where(x > Z_x, g, where(x < Z_x, -g, Z_x))` |
| `RECIPROCAL` | `-g / (x * x)` |
| [`LOG`](../../../../glossary.md#term-log) | `g / x` |
| `LOG1P` | `g / (x + 1)` |
| `SQRT` | `g / (2 * y)` |
| `RSQRT` | `-0.5 * g * y * y * y` |
| `RELU` | `where(x > Z_x, g, Z_x)` |

For exact [GELU](../../../../glossary.md#gaussian-error-linear-unit--gelu), define:

```text
phiCdf = 0.5 * (1 + erf(x * invSqrt2))
phiPdfTerm = x * exp(-0.5 * x * x) * invSqrt2Pi
regular = g * (phiCdf + phiPdfTerm)
result = where(isInf(x), where(x > Z_x, g, Z_x), regular)
```

For fixed tanh-approximation GELU, define:

```text
u = sqrt2OverPi * (x + 0.044715 * x * x * x)
t = tanh(u)
uPrime = sqrt2OverPi * (1 + 0.134145 * x * x)
regular = g * (0.5 * (1 + t) + 0.5 * x * (1 - t * t) * uPrime)
result = where(isInf(x), where(x > Z_x, g, Z_x), regular)
```

For [SiLU](../../../../glossary.md#sigmoid-linear-unit--silu), define:

```text
s = sigmoid(x)
regular = g * s * (1 + x * (1 - s))
result = where(isInf(x), where(x > Z_x, g, Z_x), regular)
```

The [infinity](../../../../glossary.md#term-infinity) selections give exact continuous-extension contributions: `g` at [positive infinity](../../../../glossary.md#nan-infinity-and-signed-zero)
and exact [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero) at [negative infinity](../../../../glossary.md#nan-infinity-and-signed-zero). They avoid an otherwise spurious `infinity * zero`
[NaN](../../../../glossary.md#term-nan). [Finite](../../../../glossary.md#term-finite) and NaN inputs use the regular formulas.

### Fixed typed coefficient representations

[Compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) must not calculate transcendental coefficients from [host](../../../../glossary.md#term-host) `Math` functions. Use these
fixed [correctly rounded](../../../../glossary.md#nan-infinity-and-signed-zero) represented values:

| Coefficient | [BFLOAT16](../../../../glossary.md#term-bfloat16) bits | [FLOAT32](../../../../glossary.md#term-float32) bits | [FLOAT64](../../../../glossary.md#term-float64) bits |
|---|---:|---:|---:|
| `0.5` | `0x3F00` | `0x3F000000` | `0x3FE0000000000000` |
| `-0.5` | `0xBF00` | `0xBF000000` | `0xBFE0000000000000` |
| `2` | `0x4000` | `0x40000000` | `0x4000000000000000` |
| `invSqrt2` | `0x3F35` | `0x3F3504F3` | `0x3FE6A09E667F3BCD` |
| `invSqrt2Pi` | `0x3ECC` | `0x3ECC422A` | `0x3FD9884533D43651` |
| `sqrt2OverPi` | `0x3F4C` | `0x3F4C422A` | `0x3FE9884533D43651` |
| `0.044715` | `0x3D37` | `0x3D372713` | `0x3FA6E4E26D4801F7` |
| `0.134145` | `0x3E09` | `0x3E095D4F` | `0x3FC12BA9D1F60179` |

[Positive zero](../../../../glossary.md#nan-infinity-and-signed-zero), positive one, negative one, and the existing [ERF](../../../../glossary.md#term-erf) coefficient retain their current
exact typed representations. [Scalar](../../../../glossary.md#term-scalar) coefficients used only by arithmetic remain [operation](../../../../glossary.md#operation)
[metadata](../../../../glossary.md#term-metadata). Values needed as [Tensor](../../../../glossary.md#tensor) comparison operands are explicit [logical splat](../../../../glossary.md#logical-splat) leaves.

### Tie, endpoint, discontinuity, domain, [NaN](../../../../glossary.md#term-nan), and [infinity](../../../../glossary.md#term-infinity) policy

The complete 0005A policy is:

- Tensor/Tensor MIN/MAX split an exact represented-numeric tie equally between the two Tensor
  roles. Equal finite values, equal same-sign infinities, and either ordering of opposite signed
  zeros are ties because current numeric equality treats the zero signs as equal.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Scalar MIN/MAX assign one half to the Tensor receiver at an exact numeric tie. The scalar
  attribute remains non-differentiable configuration.<span data-glossary-navigation="true"> ([Scalar glossary](../../../../glossary.md#term-scalar); [differentiable glossary](../../../../glossary.md#derivative-and-chain-rule))</span>
- First-class CLAMP follows the ordered composition above. With distinct ordered finite bounds,
  the receiver contribution is `g / 2` at either endpoint, `g` strictly inside, and exact zero
  outside. When both extrema stages tie, including equal same-representation bounds and the
  applicable signed-zero bound cases, the contribution is <span data-glossary-navigation="true"> ([extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>`g / 4`.
- Any NaN operand makes ordered and equality comparisons false. Piecewise MIN/MAX, scalar
  extrema, CLAMP, ABS, and RELU therefore return an exact positive-zero contribution at unordered
  NaN positions, independent of the incoming cotangent. No NaN payload is selected.<span data-glossary-navigation="true"> ([cotangent glossary](../../../../glossary.md#cotangent); [NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- ABS returns `g` for positive [finite](../../../../glossary.md#term-finite) values and [positive infinity](../../../../glossary.md#nan-infinity-and-signed-zero), `-g` for negative finite
  values and negative infinity, and exact positive zero for either signed zero and NaN.<span data-glossary-navigation="true"> ([negative infinity glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [signed zero glossary](../../../../glossary.md#term-signed-zero))</span>
- RELU returns `g` only for values strictly greater than positive zero. It returns exact positive
  zero for negative values, both signed zeros, negative infinity, and NaN; positive infinity
  returns `g`.
- FLOOR, CEIL, and SIGN preserve Compiler 0004B's direct exact-positive-zero contribution for
  every accepted represented value, including integers, discontinuities, both signed zeros,
  infinities, and NaN.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>
- GELU, tanh-approximation GELU, and SiLU use their regular formulas for finite and NaN inputs,
  exact positive-zero contribution at negative infinity, and <span data-glossary-navigation="true"> ([GELU glossary](../../../../glossary.md#gaussian-error-linear-unit--gelu); [tanh glossary](../../../../glossary.md#term-tanh); [SiLU glossary](../../../../glossary.md#sigmoid-linear-unit--silu))</span>`g` at positive infinity.
- ADD, SUB, MUL, DIV, scalar ADD/SUB/MUL/DIV, floating CAST, NEG, RECIPROCAL, LOG, LOG1P, EXP,
  EXPM1, ERF, SQRT, RSQRT, SIGMOID, TANH, and POW use the exact formulas above without
  compiler-inserted finite/domain masks or continuous-extension repair. Zero denominators,
  negative logarithm domains, zero/negative power bases, overflow products, signed zeros,
  infinities, and NaNs flow through the ordinary shared Tensor operations in formula order.
  This raw-formula rule is the selected derivative convention even where the real derivative is
  undefined or a different limiting value exists.<span data-glossary-navigation="true"> ([LOG glossary](../../../../glossary.md#term-log); [EXP glossary](../../../../glossary.md#term-expx); [ERF glossary](../../../../glossary.md#term-erf); [SIGMOID glossary](../../../../glossary.md#term-sigmoid); [masks glossary](../../../../glossary.md#term-boolean-mask); [overflow glossary](../../../../glossary.md#term-overflow); [derivative glossary](../../../../glossary.md#term-derivative))</span>
- WHERE routes the incoming cotangent with selection, not multiplication by a zero mask, so an
  unselected branch receives exact positive zero even if its value or the incoming cotangent is
  exceptional.
- Floating CAST returns the incoming cotangent unchanged for equal source/target types and uses
  one ordinary cast to the source floating type otherwise. No gradient-specific rounding or NaN
  payload policy is added.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [gradient glossary](../../../../glossary.md#term-gradient); [rounding glossary](../../../../glossary.md#term-rounding))</span>
- Comparison, Boolean logical, and floating-classification occurrences stop differentiation.
  Their BOOL outputs never receive cotangents and their numeric inputs receive no contribution
  through those occurrences.<span data-glossary-navigation="true"> ([Boolean glossary](../../../../glossary.md#data-type--datatype); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>

No [backend](../../../../glossary.md#backend) value evaluator exists in this task. Focused tests lock formula structure, [constants](../../../../glossary.md#term-constant),
role routing, generated [provenance](../../../../glossary.md#provenance), and [fail-closed](../../../../glossary.md#term-fail-closed) selection; future backend [conformance](../../../../glossary.md#term-conformance) owns
numeric execution evidence.

### Request-local [constants](../../../../glossary.md#term-constant)

Extend the existing request-[local derivative](../../../../glossary.md#derivative-and-chain-rule)-constant [owner](../../../../glossary.md#term-owner) into one exact typed-splat [cache](../../../../glossary.md#term-cache):

- cache by [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) exact [data type](../../../../glossary.md#data-type--datatype) and bits;
- create at most one provenance-free, storage-free, unlabeled, non-gradient scalar Tensor leaf for
  each exact value used by one expansion;
- register every created [leaf](../../../../glossary.md#term-leaf) explicitly as one `CompileTimeConstantGraph.Splat`;
- return [Shape](../../../../glossary.md#shape)-specific values only through ordinary public `expand`;
- preserve deterministic first-use [binding](../../../../glossary.md#term-binding) order;
- make existing zero/one helpers delegate to the same cache; and
- include only bindings reachable from returned [gradient](../../../../glossary.md#term-gradient) roots in combined-[capture](../../../../glossary.md#term-capture-meanings) ingress.

This cache may hold zero, one, negative one, one half, or an exact [scalar](../../../../glossary.md#term-scalar) MIN/MAX/CLAMP bound. It
must not infer constants from [Tensor](../../../../glossary.md#tensor) storage, labels, [descriptors](../../../../glossary.md#term-descriptor), [provenance](../../../../glossary.md#provenance) absence, factory
history, or [operation](../../../../glossary.md#operation) attributes without explicit registration. It is ephemeral [compiler](../../../../glossary.md#term-compiler)
bookkeeping, not a public constant [API](../../../../glossary.md#term-application-programming-interface-api) or a second [graph](../../../../glossary.md#graph).

### [Preflight](../../../../glossary.md#term-preflight), [validation](../../../../glossary.md#term-validation), and failure order

Preserve the existing top-level order:

1. `GraphCompiler` validates [compile](../../../../glossary.md#compile) [arguments](../../../../glossary.md#term-argument) and mode/request agreement.
2. `AutogradPreflight` validates the scalar floating gradient-eligible objective and exact forward
   output membership.<span data-glossary-navigation="true"> ([objective glossary](../../../../glossary.md#term-objective))</span>
3. It inventories all forward [producers](../../../../glossary.md#term-producer) and validates explicit ingress membership.
4. It inventories objective ancestry and validates ordered targets, identity uniqueness,
   membership, floating type, gradient eligibility, and selected reachability.<span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
5. It visits selected producer occurrences in deterministic producer postorder and validates the
   complete selected route before seed, derivative constant, bound splat, coefficient, or formula
   Tensor construction.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer))</span>

For each selected 0005A [occurrence](../../../../glossary.md#term-occurrence), preserve this order:

1. exact one-output and selected [canonical](../../../../glossary.md#term-canonical-form) output position;
2. floating output type and selected-output [gradient](../../../../glossary.md#term-gradient) eligibility;
3. selected-input gradient eligibility in input-position order;
4. exact typed kind/attributes pairing;
5. exact [signature](../../../../glossary.md#term-signature) [arity](../../../../glossary.md#term-arity) and role differentiability in input-position order;
6. floating input promotion or same-type requirements;
7. exact output [Shape](../../../../glossary.md#shape)/type consistency and availability of `sumToShape`/cast [normalization](../../../../glossary.md#term-normalization); and
8. the fixed local policy/formula row selected by kind and attributes.

Do not inspect [Tensor](../../../../glossary.md#tensor) values or reject [NaN](../../../../glossary.md#term-nan), [infinity](../../../../glossary.md#term-infinity), [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero), zero-denominator, power-domain,
or clamp-bound cases accepted by the forward [model](../../../../glossary.md#term-model). The formulas above cover those represented
values. A selected comparison condition, logical role, classification role, non-floating cast
role, unknown [operation](../../../../glossary.md#operation), wrong attributes variant, unsupported output slot, or deferred family
must [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation) before [derivative](../../../../glossary.md#term-derivative) Tensor allocation.

A known preflight failure consumes no [seed](../../../../glossary.md#term-seed), derivative [constant](../../../../glossary.md#term-constant), bound splat, coefficient, formula
Tensor, or derivative [`TensorId`](../../../../glossary.md#tensorid). After successful preflight, public Tensor construction, exact
[scalar](../../../../glossary.md#term-scalar) derivation, [capture](../../../../glossary.md#term-capture-meanings), [inference](../../../../glossary.md#term-inference), validation, optimization, [publication](../../../../glossary.md#term-publication), or [planning](../../../../glossary.md#term-planning) may
fail after consuming [opaque](../../../../glossary.md#term-opaque) Tensor IDs. IDs are never rolled back or reused.

### Accumulation, [producers](../../../../glossary.md#term-producer), [provenance](../../../../glossary.md#provenance), and combined [graph](../../../../glossary.md#graph)

- Preserve exact [Tensor](../../../../glossary.md#tensor)-object and [`TensorProducer`](../../../../glossary.md#producer--tensorproducer) identity for request-local ancestry, selected
  occurrence, contribution, and accumulation maps.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Append contributions in reverse producer-postorder and input-position order. Repeated operand
  positions remain repeated contributions.
- Normalize every contribution to its input before accumulation, then use left-associated
  ordinary `Tensor.add` in deterministic append order.
- Scalar attributes, bounds, and generated logical-splat leaves are not differentiation targets
  or original forward producers.<span data-glossary-navigation="true"> ([leaves glossary](../../../../glossary.md#term-leaf))</span><span data-glossary-navigation="true"> ([Scalar glossary](../../../../glossary.md#term-scalar); [targets glossary](../../../../glossary.md#term-target))</span>
- Generated formula producers and exact-splat expansion producers are BACKWARD work. Exact
  original producer identities remain FORWARD work.
- Preserve [canonical](../../../../glossary.md#term-canonical-form) exact [producer](../../../../glossary.md#term-producer) output wrappers. Do not reconstruct a Tensor from a captured
  [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) or manufacture a sibling output.
- [Capture](../../../../glossary.md#term-capture-meanings) ordered forward outputs and [gradient](../../../../glossary.md#term-gradient) roots together exactly once. Assign each [`NodeId`](../../../../glossary.md#nodeid)
  and `ValueId` once, retain per-[node](../../../../glossary.md#node) `GraphPhase`, and keep target roles independent when two
  targets share one gradient value.
- Run the existing shared inference, validation, canonicalization, exact arithmetic rewriting,
  constant folding, whole-graph liveness, phase-local common-subexpression elimination, and final
  validation pipeline. Add no gradient-specific rewrite, fold, simplification, or pass.<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run); [inference glossary](../../../../glossary.md#term-inference); [validation glossary](../../../../glossary.md#term-validation); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [constant folding glossary](../../../../glossary.md#term-constant-folding); [liveness glossary](../../../../glossary.md#term-liveness); [common-subexpression elimination glossary](../../../../glossary.md#term-common-subexpression-elimination-cse))</span>
- Return the existing [immutable](../../../../glossary.md#term-immutable) `GraphCompilation` and `CompileArtifacts` boundaries unchanged.

## Out of [scope](../../../../glossary.md#term-scope)

- reductions, products, reduction extrema, scans, softmax/log-softmax, statistics, norms,
  layer/RMS/batch normalization, or any Compiler 0005B work<span data-glossary-navigation="true"> ([extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [softmax glossary](../../../../glossary.md#softmax--log-softmax); [log-softmax glossary](../../../../glossary.md#softmax--log-softmax); [batch normalization glossary](../../../../glossary.md#batch-normalization--batch-normalization-inference-and-training))</span>
- layout, reshape/expand/permute, slice/update/crop, composition, windows, Gather/scatter,
  ordering/top-K, dropout, RNG state, or any Compiler 0005C work<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [slice glossary](../../../../glossary.md#slice); [Gather glossary](../../../../glossary.md#gather); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [RNG state glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
- attention, [convolution](../../../../glossary.md#term-channels-first-convolution-layers), [pooling](../../../../glossary.md#term-pooling), [losses](../../../../glossary.md#loss), structured [ML](../../../../glossary.md#term-machine-learning-ml) formulas, or any [Compiler](../../../../glossary.md#term-compiler) 0005D work
- the source-backed all-operation/output/role closure audit, transitive formula-operation closure,
  capability checkpoint, or any Compiler 0005E work<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- public objectives, targets, seeds, disconnected-result behavior, create-graph, derivative order,
  higher-order execution, order-aware phases, or Compiler 0006<span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target); [seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [graph glossary](../../../../glossary.md#graph); [derivative glossary](../../../../glossary.md#term-derivative))</span>
- a new model operation kind, backward-only kind, Tensor method, attributes type, comparison
  semantics, extrema/clamp forward policy, cast conversion contract, or Tensor gradient state<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [Tensor glossary](../../../../glossary.md#tensor); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- a public compiler gradient registry, facade, derivative-policy object, selectable subgradient
  mode, generic algebra builder, direct graph-node formula construction, placeholder Tensor,
  runtime tape, physical saved value, or backend-owned global autograd<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [facade glossary](../../../../glossary.md#term-facade); [node glossary](../../../../glossary.md#node); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [autograd glossary](../../../../glossary.md#autograd))</span>
- a gradient-specific arithmetic, cast, numerical, validation, constant-folding, simplification,
  optimization, or exceptional-value pipeline<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [constant glossary](../../../../glossary.md#term-constant))</span>
- executable Tensor evaluation, backend lowering, kernel selection, preparation, runtime,
  storage, physical buffers, schedules, optimizer updates, publication delivery, or training APIs<span data-glossary-navigation="true"> ([evaluation glossary](../../../../glossary.md#term-evaluation); [lowering glossary](../../../../glossary.md#lowering); [kernel glossary](../../../../glossary.md#kernel); [buffers glossary](../../../../glossary.md#term-buffer); [optimizer glossary](../../../../glossary.md#term-optimizer); [publication glossary](../../../../glossary.md#term-publication))</span>
- architecture, ADR, module-boundary, dependency, Gradle, config, planning, trace, model,
  runtime, prepare, engine, extension, backend, architecture-test, backend-conformance, or
  integration-test changes<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [planning glossary](../../../../glossary.md#term-planning); [trace glossary](../../../../glossary.md#trace); [prepare glossary](../../../../glossary.md#prepare); [engine glossary](../../../../glossary.md#term-engine); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- a detailed Compiler 0005B, 0005C, 0005D, 0005E, or 0006 specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>
- [ADR 0009](../../../../design/decisions/0009-compiler-owned-pre-capture-tensor-expression-autograd.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [General documentation style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- [Planning documentation style](../../../../developer-guide/documentation/planning-style.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Compiler master plan](../master-plan.md)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>
- [Compiler 0004](0004-compiler-owned-pre-capture-autograd-and-combined-graph-compilation.md)
- [Compiler 0004A](0004a-exact-composition-gradient-rule-extensions.md)
- [Compiler 0004B](0004b-shared-algebra-cotangent-normalization-and-local-derivative-rules.md)
- [Compiler 0005](0005-publication-planning-orchestration-and-compile-artifacts.md)
- [Model adjoint expressibility audit](../../model/adjoint-expressibility-audit.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [adjoint glossary](../../../../glossary.md#term-adjoint))</span>
- [Model 0025A](../../model/tasks/0025a-portable-floating-comparison-extrema-and-clamp-semantics.md)
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- [Training API](../../../../api/training-api.md)
- [Autograd strategy](../../../../design/notes/autograd-strategy.md)<span data-glossary-navigation="true"> ([Autograd glossary](../../../../glossary.md#autograd))</span>
- [Autograd user guide](../../../../user-guide/autograd.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- Compiler owns fail-closed preflight, reverse traversal, local derivative policy, formula
  construction, contribution accumulation, and combined graph construction.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [fail-closed glossary](../../../../glossary.md#term-fail-closed); [preflight glossary](../../../../glossary.md#term-preflight); [local derivative glossary](../../../../glossary.md#derivative-and-chain-rule); [graph glossary](../../../../glossary.md#graph))</span>
- Model continues to own immutable forward Tensor and operation semantics. Model 0025A's forward
  comparison/extrema/clamp contract is consumed unchanged.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [immutable glossary](../../../../glossary.md#term-immutable); [Tensor glossary](../../../../glossary.md#tensor); [operation glossary](../../../../glossary.md#operation); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum))</span>
- Every formula uses existing public Tensor operations. No derivative rule moves into model and no
  second low-level algebra or direct generated graph-node representation is introduced.<span data-glossary-navigation="true"> ([derivative glossary](../../../../glossary.md#term-derivative); [node glossary](../../../../glossary.md#node))</span>
- Tensor retains immutable identity, descriptor, and provenance plus its existing borrowed storage
  association; it gains no gradient field, backward method, tape, or derivative lifecycle.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [gradient glossary](../../../../glossary.md#term-gradient); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Exact Tensor identity maps and generated scalar caches are request-local ephemeral compiler
  bookkeeping and do not escape capture.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-reference-identity))</span><span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [capture glossary](../../../../glossary.md#term-capture-meanings))</span>
- The combined graph remains one immutable compile-time graph with per-node phase. Compiler output
  contains no physical, prepared, backend-specific, runtime, or executable state.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Compiler dependencies remain unchanged and continue to exclude runtime, prepare, engine, and
  concrete backends.<span data-glossary-navigation="true"> ([prepare glossary](../../../../glossary.md#prepare); [engine glossary](../../../../glossary.md#term-engine))</span>
- `ARCHITECTURE.md` remains unchanged. If implementation requires a model API, changed forward
  semantic, dependency, public gradient surface, or lifecycle change, stop and report the
  conflict.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Package impact

Existing package used:

- `io.github.pho001.synaptik.compiler` — remains the single cohesive package-private compiler
  front-end/autograd boundary for preflight, formula construction, request-local constants,
  reverse accumulation, and focused same-package tests.<span data-glossary-navigation="true"> ([autograd glossary](../../../../glossary.md#autograd); [preflight glossary](../../../../glossary.md#term-preflight); [constants glossary](../../../../glossary.md#term-constant))</span>

Packages added or changed:

- None.

Type placement:

- `io.github.pho001.synaptik.compiler.AutogradPreflight` — extends its closed selected-occurrence
  matrix and exact attrs/role/normalization validation.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [normalization glossary](../../../../glossary.md#term-normalization); [validation glossary](../../../../glossary.md#term-validation))</span>
- `io.github.pho001.synaptik.compiler.ElementwiseGradientRules` — owns all 0005A local formulas,
  fixed typed coefficients, scalar-bound comparison construction, and ordinary contribution
  normalization.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- `io.github.pho001.synaptik.compiler.FirstOrderAutograd` — retains reverse accumulation and
  generalizes only its nested request-local derivative-constant owner to exact typed splats.<span data-glossary-navigation="true"> ([local derivative glossary](../../../../glossary.md#derivative-and-chain-rule); [owner glossary](../../../../glossary.md#term-owner))</span>

No public type, package, [facade](../../../../glossary.md#term-facade), [registry](../../../../glossary.md#term-registry), or policy object is added. Tests remain in the mirrored
[compiler](../../../../glossary.md#term-compiler) package.

## Affected files

Expected [compiler](../../../../glossary.md#term-compiler) production:

- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/AutogradPreflight.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ElementwiseGradientRules.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/FirstOrderAutograd.java`

Expected compiler tests:

- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/AutogradPreflightTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/GradientRulesTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/FirstOrderAutogradTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/GraphCompilerTest.java`

Expected documentation and [planning](../../../../glossary.md#term-planning):

- `docs/api/compile-api.md`
- `docs/design/notes/autograd-strategy.md`
- `docs/user-guide/autograd.md`
- `docs/glossary.md`
- this task specification
- `docs/planning/modules/compiler/master-plan.md`
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless a conflict requires stopping: [Tensor](../../../../glossary.md#tensor)/Training/Public APIs,
[model](../../../../glossary.md#term-model) capabilities/audits, model source/tests/Javadocs, compiler [inference](../../../../glossary.md#term-inference) and
optimization source/tests, architecture documents/ADRs/tests, [Gradle](../../../../glossary.md#term-gradle), config, planning, [trace](../../../../glossary.md#trace),
[runtime](../../../../glossary.md#term-runtime), [prepare](../../../../glossary.md#prepare), [engine](../../../../glossary.md#term-engine), extensions, [backends](../../../../glossary.md#backend), backend [conformance](../../../../glossary.md#term-conformance), and integration tests.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most the exact fifteen paths listed under
[Affected files](#affected-files):

- 3 [compiler](../../../../glossary.md#term-compiler) production files;
- 4 compiler test files; and
- 8 documentation/[planning](../../../../glossary.md#term-planning) files.

No other path is authorized. If a formula, exact policy, test, or documentation contract requires
a fifteenth path, a [model](../../../../glossary.md#term-model) or public [API](../../../../glossary.md#term-application-programming-interface-api) change, or another [module](../../../../glossary.md#term-module), stop and report the concrete
blocker rather than silently [narrowing](../../../../glossary.md#term-narrowing) an assigned [operation](../../../../glossary.md#operation) or expanding scope.

## Acceptance criteria

- Source-backed inventory tests lock all 7 binary, 8 scalar, 19 unary, 1 WHERE, 1 CAST, 6
  comparison, 3 Boolean logical, and 3 floating-classification kinds with no unclassified 0005A
  role.<span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [scalar glossary](../../../../glossary.md#term-scalar); [unary glossary](../../../../glossary.md#term-unary); [Boolean glossary](../../../../glossary.md#data-type--datatype))</span>
- Existing ADD/SUB/MUL/DIV, scalar ADD/SUB/MUL/DIV, WHERE branch, floating CAST,
  NEG/EXP/EXPM1/ERF/FLOOR/CEIL/SIGN/SIGMOID/TANH formulas and their Shape/type behavior remain
  unchanged.<span data-glossary-navigation="true"> ([EXP glossary](../../../../glossary.md#term-expx); [ERF glossary](../../../../glossary.md#term-erf); [SIGMOID glossary](../../../../glossary.md#term-sigmoid); [TANH glossary](../../../../glossary.md#term-tanh); [Shape glossary](../../../../glossary.md#shape))</span>
- MIN/MAX, POW, scalar MIN/MAX/POW/CLAMP, ABS/RECIPROCAL/LOG/LOG1P/SQRT/RSQRT/RELU/GELU/
  GELU_TANH_APPROXIMATION/SILU implement exactly the formulas and policies above.<span data-glossary-navigation="true"> ([LOG glossary](../../../../glossary.md#term-log); [GELU glossary](../../../../glossary.md#gaussian-error-linear-unit--gelu))</span>
- Focused formula-structure tests cover both roles of broadcast and mixed-floating binary
  operations, exact normalization order, repeated operands, scalar exact-bit attributes, every
  new unary kind, and first-class CLAMP without decomposing the forward occurrence.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Tests lock equal finite, opposite-signed-zero, equal-infinity, unordered-NaN, clamp endpoint,
  equal-bound, signed-zero-bound, discontinuity, raw-domain, and activation-infinity formula
  structure without adding a model evaluator.<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [infinity glossary](../../../../glossary.md#term-infinity); [NaN glossary](../../../../glossary.md#term-nan); [activation glossary](../../../../glossary.md#term-activation); [model glossary](../../../../glossary.md#term-model))</span>
- Every input contribution has the selected input's exact Shape and floating type before
  accumulation. Mixed-floating POW and extrema cover both selected roles independently.<span data-glossary-navigation="true"> ([extrema glossary](../../../../glossary.md#saturation-pole-and-extremum))</span>
- The exact typed coefficient bit table and represented-type scalar exponent-minus-one conversion
  are tested for BFLOAT16, FLOAT32, and FLOAT64.<span data-glossary-navigation="true"> ([exponent glossary](../../../../glossary.md#term-exponent); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64))</span>
- Request-local exact splats are identity-cached by exact typed bits, explicitly registered,
  deterministically ordered, expanded through public Tensor operations, pruned when unreachable,
  and never inferred from storage or metadata.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [Tensor glossary](../../../../glossary.md#tensor); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Comparison, logical, classification, WHERE-condition, scalar-attribute, clamp-bound, and
  non-floating cast roles remain non-differentiable and fail closed when selected as a route.<span data-glossary-navigation="true"> ([differentiable glossary](../../../../glossary.md#derivative-and-chain-rule); [fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation))</span>
- Unsupported/deferred operations, wrong attrs/signatures, unsupported output roles, and new
  unknown kinds fail before derivative Tensor allocation, with deterministic occurrence context.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [derivative glossary](../../../../glossary.md#term-derivative))</span>
- Repeated operand positions remain repeated deterministic contributions. Multiple target roles
  may share one captured gradient value without identity nodes.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [gradient glossary](../../../../glossary.md#term-gradient); [nodes glossary](../../../../glossary.md#node))</span>
- One combined capture preserves exact original producer identity as FORWARD, generated producers
  as BACKWARD, canonical wrappers, unique graph-local IDs, immutable result roles, and current
  publication/artifact behavior.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [producer glossary](../../../../glossary.md#term-producer); [canonical glossary](../../../../glossary.md#term-canonical-form); [graph glossary](../../../../glossary.md#graph); [immutable glossary](../../../../glossary.md#term-immutable); [publication glossary](../../../../glossary.md#term-publication); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Generated formulas pass existing compiler inference/validation and only the shared exact
  optimization pipeline. No gradient-specific simplifier, constant folder, or exceptional-value
  pass is added.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [validation glossary](../../../../glossary.md#term-validation); [constant glossary](../../../../glossary.md#term-constant))</span>
- No model, public Tensor, request/result, registry/facade, dependency, build, architecture,
  backend, prepare, runtime, training, or higher-order contract changes.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [facade glossary](../../../../glossary.md#term-facade); [dependency glossary](../../../../glossary.md#term-dependency); [backend glossary](../../../../glossary.md#backend); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Compile API, autograd strategy, autograd user guide, glossary, task, compiler and model master
  plans, and roadmap consistently describe the implemented policy and current-versus-planned
  boundary after implementation.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [autograd glossary](../../../../glossary.md#autograd))</span>
- Tensor API needs no change because model forward semantics and public Tensor declarations remain
  unchanged; Training and Public APIs need no change because no public request, publication, or
  training workflow changes. The documentation-focused pass must record these no-change reasons.
- Exactly the fifteen authorized paths change. [Compiler](../../../../glossary.md#term-compiler) 0005A becomes `Complete` only after all
  implementation, module, Javadoc, documentation, scope, and independent documentation-review
  evidence is final. Compiler 0005B–0005E and 0006 remain Draft without detailed specifications.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>
- A separate documentation-focused agent pass has independently finalized affected Javadocs,
  explanatory documentation, glossary impact, planning status/evidence, terminology, links, and
  rendered output in the same overall change.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

During implementation, [run](../../../../glossary.md#term-run) focused suites as needed:

```bash
./gradlew :modules:compiler:test --tests io.github.pho001.synaptik.compiler.AutogradPreflightTest
./gradlew :modules:compiler:test --tests io.github.pho001.synaptik.compiler.GradientRulesTest
./gradlew :modules:compiler:test --tests io.github.pho001.synaptik.compiler.FirstOrderAutogradTest
./gradlew :modules:compiler:test --tests io.github.pho001.synaptik.compiler.GraphCompilerTest
```

After executable Java stabilizes, run one final [compiler](../../../../glossary.md#term-compiler) [module](../../../../glossary.md#term-module) suite:

```bash
./gradlew :modules:compiler:test
```

Record [JUnit](../../../../glossary.md#term-junit) XML suite/test counts and zero skipped/failure/error outcomes. The implementation
context hands that exact evidence and the final executable diff to the documentation-focused
agent. That pass must not repeat successful Java tests unless executable behavior changes after
the evidence or it records a concrete stale-evidence risk.

Documentation pass:

```bash
./gradlew :modules:compiler:javadoc
python3 /tmp/validate_synaptik_markdown.py
git diff --check
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git status --short
```

The documentation pass must additionally verify generated package-private Javadocs, complete
`@param`/`@return`/expected-`@throws` coverage, local links and anchors, balanced fences, final
newlines, terminology, exact fifteen-path [scope](../../../../glossary.md#term-scope), source-backed 48-kind inventory, exact
coefficient bits, no [model](../../../../glossary.md#term-model)/test/[Gradle](../../../../glossary.md#term-gradle)/[dependency](../../../../glossary.md#term-dependency)/architecture change, synchronized task/master/
roadmap status, exactly one [Ready](../../../../glossary.md#term-ready)-or-current detailed compiler task, and absence of detailed
0005B–0005E/0006 specifications.

Repository-wide validation and the compiler first-order capability [checkpoint](../../../../glossary.md#term-checkpoint) are deferred to
Compiler 0005E or [continuous integration](../../../../glossary.md#term-continuous-integration-ci). This task changes one module's package-private [autograd](../../../../glossary.md#autograd)
behavior and no dependency, shared build, architecture boundary, or public [API](../../../../glossary.md#term-application-programming-interface-api).

## Dependencies

- [Model](../../../../glossary.md#term-model) 0025A portable floating comparison, [extrema](../../../../glossary.md#saturation-pole-and-extremum), and clamp semantics — Complete.
- [Compiler](../../../../glossary.md#term-compiler) 0005 [publication](../../../../glossary.md#term-publication), [planning](../../../../glossary.md#term-planning) orchestration, and [compile artifacts](../../../../glossary.md#compile-artifacts) — Complete.
- Compiler 0004–0004B pre-capture autograd, exact-composition extensions, and shared-algebra
  normalization/local rules — Complete.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [autograd glossary](../../../../glossary.md#autograd); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- Model 0025 [canonical](../../../../glossary.md#term-canonical-form) [producer](../../../../glossary.md#term-producer) outputs and the completed [adjoint](../../../../glossary.md#term-adjoint)-expressibility audit — Complete.
- Existing public Tensor binary/scalar/unary/comparison/selection/classification/cast operations,
  typed <span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [scalar glossary](../../../../glossary.md#term-scalar); [unary glossary](../../../../glossary.md#term-unary))</span>[`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue), explicit logical-splat ingress, compiler inference/validation, and exact
  combined optimization — implemented and sufficient.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [validation glossary](../../../../glossary.md#term-validation))</span>

No Config, [Trace](../../../../glossary.md#trace), [Runtime](../../../../glossary.md#term-runtime), [Prepare](../../../../glossary.md#prepare), [Engine](../../../../glossary.md#term-engine), Training, [backend](../../../../glossary.md#backend), or new model task is a [dependency](../../../../glossary.md#term-dependency).

## Follow-up tasks

- Compiler 0005B remains Draft and follows 0005A. It owns reductions, scans, softmax,
  statistics, norms, and normalization policies/formulas.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [softmax glossary](../../../../glossary.md#softmax--log-softmax); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- Compiler 0005C remains Draft and follows 0005B. It owns remaining layout/window/indexing/
  scatter/ordering/stochastic roles.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout))</span>
- Compiler 0005D remains Draft and follows 0005B/0005C. It owns structured attention,
  convolution, pooling, and loss roles.<span data-glossary-navigation="true"> ([convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [pooling glossary](../../../../glossary.md#term-pooling); [loss glossary](../../../../glossary.md#loss))</span>
- Compiler 0005E remains Draft and follows 0005A–0005D. It owns the complete source-backed
  first-order role/output closure audit, transitive formula-operation closure, and capability
  checkpoint.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Compiler 0006 remains Draft and follows 0005E plus the stable public compile/artifact boundary.
  It owns explicit functional requests, seeds, disconnected results, derivative order, and
  higher-order differentiation.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [artifact glossary](../../../../glossary.md#term-artifact); [seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [derivative glossary](../../../../glossary.md#term-derivative))</span>

Do not create any of those detailed specifications during this task.

## Architecture impact

Expected impact: None.

This task fills the existing [compiler](../../../../glossary.md#term-compiler)-owned local differentiation extension point through the
current public [Tensor](../../../../glossary.md#tensor) algebra. It changes no authority, [module](../../../../glossary.md#term-module) [ownership](../../../../glossary.md#term-ownership-meanings), [dependency direction](../../../../glossary.md#java-platform-and-build-tools),
public [lifecycle](../../../../glossary.md#lifecycle), forward semantic, or [artifact](../../../../glossary.md#term-artifact) [shape](../../../../glossary.md#shape).

If implementation requires a new [model](../../../../glossary.md#term-model) [operation](../../../../glossary.md#operation)/[API](../../../../glossary.md#term-application-programming-interface-api), changed forward meaning, public [gradient](../../../../glossary.md#term-gradient)
surface, direct [graph](../../../../glossary.md#graph)-[node](../../../../glossary.md#node) formula construction, [dependency](../../../../glossary.md#term-dependency), [runtime](../../../../glossary.md#term-runtime) tape, physical saved value,
or architecture change, stop and report the exact conflict.

## Implementation prompt

Use this prompt in a separate agentic task/thread with clean context:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, the compiler master plan, and
docs/planning/modules/compiler/tasks/0005a-derivative-policy-and-elementwise-activation-gradient-completion.md.
Read every directly referenced architecture, completed compiler/model task and audit, final
compiler/model source and tests, Compile/Tensor/Training APIs, autograd strategy/user guide, and
glossary needed to verify the exact current operation inventory and shared Tensor algebra.

Implement Compiler 0005A exactly within its fifteen authorized paths. Preserve the existing
request, fail-closed preflight, identity accumulation, one combined capture, phase, validation,
optimization, publication, and artifact contracts. Do not implement 0005B–0005E or 0006, change
model forward semantics or APIs, add a second algebra/registry/facade/policy mode, add
gradient-specific optimization, or touch Gradle, architecture, another module, backend, prepare,
runtime, or training behavior. Stop on any scope, formula-expressibility, inventory, or
architecture conflict rather than inventing a policy or silently omitting an assigned role.

After executable Java stabilizes, run the focused tests as needed and one final compiler module
suite. Hand the actual diff and exact test evidence to a separate documentation-focused agent or
thread with clean context. That targeted pass must follow
docs/developer-guide/documentation-rules.md, independently finalize affected Javadocs, Compile
API, autograd strategy/user guide, glossary impact, planning status/evidence, terminology, links,
and rendered Javadoc in the same overall change, and must not repeat successful Java tests unless
executable behavior changes or a concrete stale-evidence risk is recorded.

Run the final Javadoc/documentation/scope/status checks. Update this task with local decisions,
exact evidence, implementation notes, completion summary, and final status. Do not mark it
Complete before every acceptance criterion and the documentation-focused pass finish.
```

## Local decisions

- The task-specified derivative policies were implemented without additional policy choices:
  symmetric Tensor extrema ties, one-half scalar ties, ordered-composition CLAMP, direct
  ABS/RELU discontinuities, raw analytic domains, and explicit modern-activation infinity
  selection.<span data-glossary-navigation="true"> ([derivative glossary](../../../../glossary.md#term-derivative); [Tensor glossary](../../../../glossary.md#tensor); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [scalar glossary](../../../../glossary.md#term-scalar); [activation glossary](../../../../glossary.md#term-activation); [infinity glossary](../../../../glossary.md#term-infinity))</span>
- `ElementwiseGradientRules` remains the sole local formula owner. The implementation adds no
  registry, facade, selectable policy object, second algebra, backward-only operation, or
  gradient-specific optimization.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [registry glossary](../../../../glossary.md#term-registry); [facade glossary](../../../../glossary.md#term-facade); [operation glossary](../../../../glossary.md#operation); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Fixed arithmetic coefficients remain exact scalar operation metadata. Values needed as Tensor
  comparison operands use the existing request-local derivative-constant owner generalized to
  exact <span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [local derivative glossary](../../../../glossary.md#derivative-and-chain-rule); [constant glossary](../../../../glossary.md#term-constant))</span>[`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) data-type/bit [identity](../../../../glossary.md#term-identity-meanings).
- Scalar POW derives exponent-minus-one once in the represented type: BFLOAT16 through
  binary32 subtraction and the existing round-to-nearest-ties-to-even conversion, FLOAT32 through
  one binary32 subtraction, and FLOAT64 through one binary64 subtraction.<span data-glossary-navigation="true"> ([exponent glossary](../../../../glossary.md#term-exponent); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [binary32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [round-to-nearest glossary](../../../../glossary.md#floating-point-precision-and-rounding); [ties-to-even glossary](../../../../glossary.md#term-ties-to-even); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [binary64 glossary](../../../../glossary.md#floating-point-precision-and-rounding))</span>
- Generated splat bindings remain deterministic by first use and are pruned by exact
  gradient-expression reachability before the caller ingress and derivative ingress are combined.<span data-glossary-navigation="true"> ([bindings glossary](../../../../glossary.md#term-binding))</span>

## Known limitations

- This task completes only its exact 48-kind elementwise/activation classification. Complete
  first-order current-model coverage waits for Compiler 0005B–0005E.<span data-glossary-navigation="true"> ([elementwise glossary](../../../../glossary.md#term-elementwise); [activation glossary](../../../../glossary.md#term-activation); [model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler))</span>
- The internal request remains one implicit-unit-seed scalar-objective first-order request.
  Public requests, seeds, disconnected gradients, create-graph, and derivative order wait for
  Compiler 0006.<span data-glossary-navigation="true"> ([seed glossary](../../../../glossary.md#term-seed); [scalar glossary](../../../../glossary.md#term-scalar); [objective glossary](../../../../glossary.md#term-objective); [seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [gradients glossary](../../../../glossary.md#term-gradient); [graph glossary](../../../../glossary.md#graph); [derivative glossary](../../../../glossary.md#term-derivative))</span>
- Formula-structure tests cannot prove backend numerical execution. Future backend implementations
  require conformance coverage for the selected tie, endpoint, NaN, infinity, singularity, and
  coefficient behavior.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity))</span>
- Raw analytic formulas intentionally retain ordinary shared-operation exceptional behavior at
  mathematically undefined points instead of promising finite gradients or continuous
  extensions, except for the explicit modern-activation infinity selections.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [finite glossary](../../../../glossary.md#term-finite))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

Implementation evidence reused by the documentation pass:

- Implementation context `/root/implement_compiler_0005a` ran the focused combined
  `AutogradPreflightTest`, `GradientRulesTest`, `FirstOrderAutogradTest`, and
  `GraphCompilerTest` command successfully during development.
- After the final test-only represented-boundary and source-inventory strengthening, that context
  ran `./gradlew :modules:compiler:test`; [Gradle](../../../../glossary.md#term-gradle) reported `BUILD SUCCESSFUL`, and the JUnit XML
  audit found 22 suites and 159 tests with 0 skipped, 0 failures, and 0 errors.<span data-glossary-navigation="true"> ([JUnit glossary](../../../../glossary.md#term-junit))</span>
- Production executable Java did not change after that module evidence. Clean documentation
  context <span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>`/root/implement_compiler_0005a/docs_0005a` therefore reused it and did not duplicate
  the successful Java suite.

Documentation and final-gate evidence:

- The clean documentation context independently reviewed the authoritative architecture and ADR,
  General/API-Javadoc/User Guide/Planning/Example profiles, planning guide, completed Compiler
  0004–0005 and Model 0025A contracts, model audits, final three-production/four-test diff,
  source enums/signatures, public APIs, autograd documentation, glossary, and planning state.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc); [Planning glossary](../../../../glossary.md#term-planning); [signatures glossary](../../../../glossary.md#term-signature); [autograd glossary](../../../../glossary.md#autograd))</span>
- `./gradlew :modules:compiler:javadoc` passed with `BUILD SUCCESSFUL in 1s`, 7 actionable tasks
  (`1 executed`, `6 up-to-date`), and no warning. Generated package-private pages for
  `AutogradPreflight`, `ElementwiseGradientRules`, `FirstOrderAutograd`, and its nested request,
  plan, role, expansion, and derivative-constant contracts were inspected. Parameter, result,
  expected-failure, identity, ownership, ordering, and lifecycle descriptions are present.<span data-glossary-navigation="true"> ([derivative glossary](../../../../glossary.md#term-derivative); [constant glossary](../../../../glossary.md#term-constant); [Parameter glossary](../../../../glossary.md#term-parameter); [identity glossary](../../../../glossary.md#term-identity-meanings); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- `python3 /tmp/validate_synaptik_markdown.py` reported `validated 12 Markdown files and 688 local
  links`, including file targets, heading anchors, balanced fences, final newlines, and trailing
  whitespace.<span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target))</span>
- `git diff --check` passed with no output.
- The sorted union of modified and untracked paths contains exactly the authorized fifteen paths:
  three production Java files, four compiler tests, and eight documentation/planning files.
- Manual source-backed inventory review confirmed exactly 7 binary, 8 scalar, 19 unary, 1 WHERE,
  1 CAST, 6 comparison, 3 Boolean logical, and 3 floating-classification kinds, totaling 48.
  The production enum-array test locks every constant and order rather than only counts.<span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [scalar glossary](../../../../glossary.md#term-scalar); [unary glossary](../../../../glossary.md#term-unary); [Boolean glossary](../../../../glossary.md#data-type--datatype))</span>
- Manual formula and policy review confirmed symmetric extrema ties, ordered-composition CLAMP,
  represented-type scalar exponent-minus-one, direct ABS/RELU and FLOOR/CEIL/SIGN conventions,
  raw analytic domains, activation infinity selection, mixed-floating normalization, exact
  request-local splat identity/order/registration/pruning, deterministic accumulation, original
  versus generated producer phase, canonical wrappers, one combined capture, and shared
  inference/validation/optimization.<span data-glossary-navigation="true"> ([extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [exponent glossary](../../../../glossary.md#term-exponent); [activation glossary](../../../../glossary.md#term-activation); [infinity glossary](../../../../glossary.md#term-infinity); [normalization glossary](../../../../glossary.md#term-normalization); [producer glossary](../../../../glossary.md#term-producer); [canonical glossary](../../../../glossary.md#term-canonical-form); [capture glossary](../../../../glossary.md#term-capture-meanings); [inference glossary](../../../../glossary.md#term-inference))</span>
- The fixed coefficient table in source, focused tests, task, and Compile API matches exactly for
  BFLOAT16, FLOAT32, and FLOAT64, including the unchanged ERF coefficient. No host
  transcendental coefficient derivation was introduced.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [ERF glossary](../../../../glossary.md#term-erf); [host glossary](../../../../glossary.md#term-host))</span>
- Task, [compiler](../../../../glossary.md#term-compiler) master plan, [model](../../../../glossary.md#term-model) master plan, and roadmap are synchronized to 0005A
  `Complete`. Compiler 0005B–0005E and 0006 remain concise `Draft` rows, and their detailed task
  files are absent.
- Tensor API needs no change because 0005A changes no public Tensor declaration, forward model
  semantic, descriptor, provenance, or storage contract. Training and Public APIs need no change
  because no public gradient request, publication shape, optimizer, session, prepare, run, or
  execution workflow changes.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [gradient glossary](../../../../glossary.md#term-gradient); [publication glossary](../../../../glossary.md#term-publication); [shape glossary](../../../../glossary.md#shape); [optimizer glossary](../../../../glossary.md#term-optimizer); [prepare glossary](../../../../glossary.md#prepare); [run glossary](../../../../glossary.md#term-run))</span>
- Model capabilities, adjoint/closure audits, model source/tests/Javadocs, and Model 0025A need no
  change because the compiler consumes their completed operation inventory and represented-value
  forward semantics unchanged.<span data-glossary-navigation="true"> ([adjoint glossary](../../../../glossary.md#term-adjoint); [operation glossary](../../../../glossary.md#operation))</span>
- Compiler inference/validation and optimization documentation/source/tests need no change because
  generated formulas use the ordinary existing Tensor algebra and the unchanged shared pipeline;
  no gradient-specific inference, arithmetic, fold, rewrite, simplifier, or pass was added.
- `ARCHITECTURE.md`, focused architecture pages, ADR 0009, and architecture tests need no change
  because compiler ownership, dependency direction, one-capture lifecycle, phase model, and
  artifact boundaries are unchanged. Gradle needs no change because no module, dependency, build,
  or toolchain contract changed.<span data-glossary-navigation="true"> ([dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [artifact glossary](../../../../glossary.md#term-artifact); [dependency glossary](../../../../glossary.md#term-dependency); [toolchain glossary](../../../../glossary.md#term-toolchain))</span>
- Config, Planning, Trace, Runtime, Prepare, Engine, extensions, concrete backends, backend
  conformance, and integration tests need no change because the task adds no cross-module,
  backend, runtime, training, physical-value, end-to-end, or executable behavior.<span data-glossary-navigation="true"> ([Trace glossary](../../../../glossary.md#trace); [Runtime glossary](../../../../glossary.md#term-runtime); [Engine glossary](../../../../glossary.md#term-engine); [backends glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Repository-wide tests remain deferred to Compiler 0005E or continuous integration under the
  task's validation tier.<span data-glossary-navigation="true"> ([continuous integration glossary](../../../../glossary.md#term-continuous-integration-ci))</span>

## Implementation notes

- Extended `AutogradPreflight` to admit the exact 0005A floating arithmetic/unary signatures and
  to reject comparison, Boolean logical, and floating-classification routes explicitly as
  non-differentiable before derivative Tensor allocation.<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary); [signatures glossary](../../../../glossary.md#term-signature); [Boolean glossary](../../../../glossary.md#data-type--datatype); [differentiable glossary](../../../../glossary.md#derivative-and-chain-rule); [derivative glossary](../../../../glossary.md#term-derivative); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Extended `ElementwiseGradientRules` with the exact extrema, POW, CLAMP, unary, GELU,
  tanh-approximation GELU, and SiLU formulas plus the fixed coefficient table.<span data-glossary-navigation="true"> ([extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [GELU glossary](../../../../glossary.md#gaussian-error-linear-unit--gelu); [tanh glossary](../../../../glossary.md#term-tanh); [SiLU glossary](../../../../glossary.md#sigmoid-linear-unit--silu))</span>
- Generalized `FirstOrderAutograd.DerivativeConstants` from zero/one-only helpers to one
  request-local exact typed-splat cache while preserving identity, order, explicit registration,
  reachability pruning, reverse accumulation, and one combined capture.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [identity glossary](../../../../glossary.md#term-identity-meanings); [capture glossary](../../../../glossary.md#term-capture-meanings))</span>
- Added focused tests for the source-backed 48-kind inventory, formula structure, mixed-floating
  role normalization, fixed bits, represented exponent subtraction, exceptional-value policy
  structure, exact splat identity/order/pruning, fail-closed behavior, and shared-pipeline
  capture/optimization.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [exponent glossary](../../../../glossary.md#term-exponent); [fail-closed behavior glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation))</span>
- The independent documentation context finalized the three affected implementation Javadocs,
  Compile API, autograd strategy, user guide, glossary, task, compiler/model master plans, and
  roadmap without changing executable Java behavior.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [autograd glossary](../../../../glossary.md#autograd); [compiler glossary](../../../../glossary.md#term-compiler); [model glossary](../../../../glossary.md#term-model))</span>

## Completion summary

- Completed changes: the exact 48-kind Compiler 0005A elementwise/activation classification,
  formulas, derivative policies, exact coefficients, request-local typed splats, and focused
  regression coverage.<span data-glossary-navigation="true"> ([elementwise glossary](../../../../glossary.md#term-elementwise); [activation glossary](../../../../glossary.md#term-activation); [derivative glossary](../../../../glossary.md#term-derivative); [regression glossary](../../../../glossary.md#term-regression))</span>
- Files changed or created: exactly three compiler production files, four compiler test files,
  and eight documentation/planning files.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Tests and validation: the compiler module passed 22 suites/159 tests with no skips, failures,
  or errors; compiler Javadoc, Markdown links/anchors/fences, exact scope, coefficient,
  inventory, status, final-newline, and whitespace checks passed.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [module glossary](../../../../glossary.md#term-module); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: clean context
  `/root/implement_compiler_0005a/docs_0005a` independently finalized affected Javadocs,
  explanatory documentation, glossary impact, planning state, and no-change conclusions.
- Documentation impact: Compile API, strategy, and user guide now distinguish the implemented
  0005A policy from the Draft 0005B–0005E and 0006 frontiers.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Javadoc review: all three affected package-private production owners were reviewed and
  finalized without executable changes.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span>
- Glossary impact: synchronized GELU, pre-capture autograd, and logical-splat entries; no new
  reusable term was required.<span data-glossary-navigation="true"> ([GELU glossary](../../../../glossary.md#gaussian-error-linear-unit--gelu); [capture glossary](../../../../glossary.md#term-capture-meanings); [autograd glossary](../../../../glossary.md#autograd))</span>
- Unresolved issues: None.
- Follow-up required: None. [Compiler](../../../../glossary.md#term-compiler) 0005B–0005E and 0006 remain their ordered Draft work.

Status: Complete
