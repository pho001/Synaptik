# Task 0005H: Portable [Unary](../../../../glossary.md#term-unary), Transcendental, and [Activation](../../../../glossary.md#term-activation) Closure

## Reader context

Unary means that each output position is derived from one corresponding input position, without
a reduction across neighbors. CPU’s family switches mapped the nineteen Model kinds to generated
scalar instructions and, where qualified, vector instructions. `CpuScalarReferenceKernel`
supplied a cold checking oracle over lowered IR; it was not an additional backend or a Runtime
interpreter.

“Exact GELU” named the semantic function, while its finite portable realization used the
explicitly bounded approximation recorded below. That distinction, exceptional-value handling,
and the order of FLOAT32 widening/narrowing mattered independently of which route was selected.
The original tolerances and helper calls remain historical; the current unary reference and
generator explanation describe the present semantic/construction and realization boundaries.

Current reading alongside this historical record:

- [Unary transforms and exceptional values](../../../../reference/tensor/elementwise.md#unary-numeric-transforms-and-floating-classifications)
  — provides current Model meanings independently of historical CPU realization choices.
- [CPU generated-code ownership](../../../../explanation/packages/cpu.md#iogithubpho001synaptikbackendcpuinternalcodegenemit)
  — separates generation-time emitters, emitted typed work, and JVM compilation.

## Status

Complete

## Goal

Close the current [Model](../../../../glossary.md#term-model) [unary](../../../../glossary.md#term-unary) [pointwise](../../../../glossary.md#term-pointwise) inventory for exact/default portable [CPU](../../../../glossary.md#term-cpu) execution without
adding a second evaluator, [operation](../../../../glossary.md#operation)-specific class hierarchy, public fast-math surface, or [graph](../../../../glossary.md#graph)
rewrite:

```text
nineteen UnaryElementwiseKind values
  -> one closed CPU unary opcode row per semantic kind
  -> existing scalar or parallel-scalar generated execution for FLOAT32/FLOAT64
  -> selected honest FLOAT64 Vector API rows when Java 26 supplies the required lane operations
  -> one compatible generated artifact and one partition executable
```

CPU 0005E–0005G already implement `NEG` for [FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64) and exact-semantic [`GELU`](../../../../glossary.md#gaussian-error-linear-unit--gelu) for
FLOAT64. This task preserves those rows, corrects the existing GELU negative-[infinity](../../../../glossary.md#term-infinity) result to
the Model's continuous extension, adds FLOAT32 GELU, and adds the other seventeen unary kinds for
FLOAT32/FLOAT64. The already-complete `IS_FINITE`, `IS_NAN`, and `IS_INF` classification rows are
audited and [regression](../../../../glossary.md#term-regression)-tested but not reimplemented.

The portable route remains [bytecode](../../../../glossary.md#term-bytecode)-first. `CpuScalarReferenceKernel` is the cold [conformance](../../../../glossary.md#term-conformance) and
fallback [oracle](../../../../glossary.md#term-test-oracle) over already-lowered CPU [IR](../../../../glossary.md#term-intermediate-representation-ir), not another [backend](../../../../glossary.md#backend) or a [Runtime](../../../../glossary.md#term-runtime) interpreter.

## [Scope](../../../../glossary.md#term-scope)

### Exact [operation](../../../../glossary.md#operation) and type [matrix](../../../../glossary.md#term-matrix-meanings)

Every admitted [occurrence](../../../../glossary.md#term-occurrence) has exactly one input and one output, exact
`NoOperationAttrs.INSTANCE`, equal input/output [Shape](../../../../glossary.md#shape) and [data type](../../../../glossary.md#data-type--datatype), fully static Shapes, and
resolved [layouts](../../../../glossary.md#layout). The only admitted data types are [FLOAT32](../../../../glossary.md#term-float32) and [FLOAT64](../../../../glossary.md#term-float64).

The live and task-0005H matrix is:

| [Model](../../../../glossary.md#term-model) kind | Before 0005H | Added or changed by 0005H | Final [scalar](../../../../glossary.md#term-scalar) / parallel-scalar | Final FLOAT64 [vector](../../../../glossary.md#term-vector) / parallel-vector |
|---|---|---|---|---|
| `ABS` | unsupported | add FLOAT32/FLOAT64 | yes | yes, `ABS` |
| `NEG` | FLOAT32/FLOAT64 | preserve | yes | yes, existing `NEG` |
| `RECIPROCAL` | unsupported | add FLOAT32/FLOAT64 | yes | yes, exact `1.0 / x` |
| [`LOG`](../../../../glossary.md#term-log) | unsupported | add FLOAT32/FLOAT64 | yes | yes, `LOG` |
| `LOG1P` | unsupported | add FLOAT32/FLOAT64 | yes | yes, `LOG1P` |
| [`EXP`](../../../../glossary.md#term-expx) | unsupported | add FLOAT32/FLOAT64 | yes | yes, `EXP` |
| `EXPM1` | unsupported | add FLOAT32/FLOAT64 | yes | yes, `EXPM1` |
| [`ERF`](../../../../glossary.md#term-erf) | unsupported | add FLOAT32/FLOAT64 using the existing shared approximation | yes | yes, existing vector polynomial/`EXP` helper |
| `SQRT` | unsupported | add FLOAT32/FLOAT64 | yes | yes, `SQRT` |
| `RSQRT` | unsupported | add FLOAT32/FLOAT64 as `1.0 / sqrt(x)` | yes | yes, `SQRT` then exact division |
| `FLOOR` | unsupported | add FLOAT32/FLOAT64 | yes | no Java 26 Vector operator; scalar compute |
| `CEIL` | unsupported | add FLOAT32/FLOAT64 | yes | no Java 26 Vector operator; scalar compute |
| `SIGN` | unsupported | add FLOAT32/FLOAT64 | yes | no direct Java 26 Vector operator selected; scalar compute |
| `RELU` | unsupported | add FLOAT32/FLOAT64 as `max(x, +0)` | yes | no new [extrema](../../../../glossary.md#saturation-pole-and-extremum) proof in this task; scalar compute |
| [`SIGMOID`](../../../../glossary.md#term-sigmoid) | unsupported | add stable FLOAT32/FLOAT64 realization | yes | scalar compute; no per-[lane](../../../../glossary.md#term-lane) fallback loop |
| [`TANH`](../../../../glossary.md#term-tanh) | unsupported | add FLOAT32/FLOAT64 | yes | yes, `TANH` |
| [`GELU`](../../../../glossary.md#gaussian-error-linear-unit--gelu) | FLOAT64 | add FLOAT32 and fix `-infinity -> -0` in both routes | yes | yes, existing vector helper with corrected edge handling |
| `GELU_TANH_APPROXIMATION` | unsupported | add fixed Model formula for FLOAT32/FLOAT64 | yes | scalar compute; no general fast-math meaning |
| `SILU` | unsupported | add stable FLOAT32/FLOAT64 realization | yes | scalar compute; no per-lane fallback loop |

The Java 26 incubating [Vector API](../../../../glossary.md#term-vector-api) exposes FLOAT64-compatible `ABS`, `NEG`, `EXP`, `LOG`,
`SQRT`, `TANH`, `EXPM1`, and `LOG1P` lane operators. Its math-library operators follow the
accuracy and monotonicity specification of the equivalent Java method, but the [JVM](../../../../glossary.md#term-java-virtual-machine-jvm) may realize
them with scalar lane calls rather than a hardware transcendental instruction. The task may use
those operators and the existing vector helper; it must make no intrinsic, [throughput](../../../../glossary.md#term-throughput), or
speedup claim. It must not generate an explicit lane-extraction/call/reinsertion loop. A [partition](../../../../glossary.md#partition)
containing any vector-ineligible [opcode](../../../../glossary.md#term-opcode) selects scalar or parallel-scalar compute rather than
failing.

FLOAT32 remains scalar or parallel-scalar under the existing FLOAT64-only vector architecture.
[BFLOAT16](../../../../glossary.md#term-bfloat16) remains [fail-closed](../../../../glossary.md#term-fail-closed) even though Model [unary](../../../../glossary.md#term-unary) construction accepts floating inputs and [CPU](../../../../glossary.md#term-cpu)
storage has a two-byte [carrier](../../../../glossary.md#term-storage-carrier). [FLOAT16](../../../../glossary.md#term-float16) remains absent pending Model 0026. Integral and [BOOL](../../../../glossary.md#data-type--datatype) unary
rows, mismatched types, malformed [arity](../../../../glossary.md#term-arity)/attributes/results, dynamic Shapes, and unresolved layouts
remain unsupported.

### Numerical realization [matrix](../../../../glossary.md#term-matrix-meanings)

Generated [scalar](../../../../glossary.md#term-scalar) [bytecode](../../../../glossary.md#term-bytecode) and the reference [kernel](../../../../glossary.md#kernel) must share the following already-lowered
realizations. [FLOAT64](../../../../glossary.md#term-float64) calls operate directly. Unless a native [FLOAT32](../../../../glossary.md#term-float32) primitive method exists,
FLOAT32 exactly widens the represented [binary32](../../../../glossary.md#floating-point-precision-and-rounding) input to [binary64](../../../../glossary.md#floating-point-precision-and-rounding), evaluates the listed [operation](../../../../glossary.md#operation),
and narrows once to binary32. This is [backend](../../../../glossary.md#backend)-private [evaluation](../../../../glossary.md#term-evaluation) of a same-typed FLOAT32 semantic
request, not a [Model](../../../../glossary.md#term-model) cast or promotion.

| Group | Scalar realization | [Finite](../../../../glossary.md#term-finite) [conformance](../../../../glossary.md#term-conformance) |
|---|---|---|
| `ABS`, `NEG` | primitive/`Math.abs` | exact represented result except no [NaN payload](../../../../glossary.md#nan-infinity-and-signed-zero) promise |
| `RECIPROCAL` | exact typed `+1.0 / x` | ordinary [correctly rounded](../../../../glossary.md#nan-infinity-and-signed-zero) primitive division |
| [`LOG`](../../../../glossary.md#term-log), `LOG1P`, [`EXP`](../../../../glossary.md#term-expx), `EXPM1` | matching `StrictMath` method | Java 26 method contract: within 1 ulp and semi-monotonic |
| `SQRT` | `StrictMath.sqrt` | correctly rounded closest result |
| `RSQRT` | exact typed `+1.0 / StrictMath.sqrt(x)` | two specified operations; FLOAT32 widens before both operations and narrows only their combined result; differential [oracle](../../../../glossary.md#term-test-oracle) uses the same order |
| `FLOOR`, `CEIL` | matching `StrictMath` method | exact integral-valued floating result |
| `SIGN` | matching `Math.signum` overload | exact `-1`, [signed zero](../../../../glossary.md#term-signed-zero), `+1`, [infinity](../../../../glossary.md#term-infinity) sign, or [NaN](../../../../glossary.md#term-nan) classification |
| `RELU` | `Math.max(x, +0)` | exact Model [activation](../../../../glossary.md#term-activation) [target](../../../../glossary.md#term-target) with NaN propagation and [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero) at either zero sign |
| [`SIGMOID`](../../../../glossary.md#term-sigmoid) | sign-stable two-branch formula | target [tolerance](../../../../glossary.md#term-tolerance) below; no avoidable [overflow](../../../../glossary.md#term-overflow) |
| [`TANH`](../../../../glossary.md#term-tanh) | `StrictMath.tanh` | Java 26 method contract: within 2.5 ulps |
| [`ERF`](../../../../glossary.md#term-erf) | existing shared piecewise [rational approximation](../../../../glossary.md#range-reduction-and-approximation) | activation tolerance below |
| [`GELU`](../../../../glossary.md#gaussian-error-linear-unit--gelu) | existing `0.5*x*(1+erf(x/sqrt(2)))` finite order plus explicit special cases | activation tolerance below |
| `GELU_TANH_APPROXIMATION` | exact fixed Model formula and coefficient `0.044715` | activation tolerance below, measured against that formula rather than exact GELU |
| `SILU` | sign-stable `x*sigmoid(x)` form plus explicit negative-infinity handling | activation tolerance below |

`SIGMOID` evaluates `1 / (1 + exp(-x))` for nonnegative input and
`exp(x) / (1 + exp(x))` for negative input. `SILU` uses the corresponding stable branch without
computing an overflowing exponential; [negative infinity](../../../../glossary.md#nan-infinity-and-signed-zero) is handled explicitly as [negative zero](../../../../glossary.md#nan-infinity-and-signed-zero).
`GELU` and `GELU_TANH_APPROXIMATION` also handle negative infinity explicitly as negative zero so
their continuous extensions do not become `infinity * zero -> NaN`. No helper is named or exposed
as `fastExp`, `fastTanh`, or a general fast-math policy.

### Exceptional-value contract

Tests and Javadocs must lock these observable classifications and zero signs. [NaN](../../../../glossary.md#term-nan) results are
checked by classification; no [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication), quieting, or sign promise is added.

| Kind/group | `-infinity` | `-0` | `+0` | `+infinity` | domain/NaN rule |
|---|---|---|---|---|---|
| `ABS` | `+infinity` | `+0` | `+0` | `+infinity` | NaN -> NaN |
| `NEG` | `+infinity` | `+0` | `-0` | `-infinity` | NaN -> NaN |
| `RECIPROCAL` | `-0` | `-infinity` | `+infinity` | `+0` | NaN -> NaN |
| [`LOG`](../../../../glossary.md#term-log) | NaN | `-infinity` | `-infinity` | `+infinity` | negative [finite](../../../../glossary.md#term-finite) -> NaN; NaN -> NaN |
| `LOG1P` | NaN | `-0` | `+0` | `+infinity` | `-1 -> -infinity`; below `-1` -> NaN |
| [`EXP`](../../../../glossary.md#term-expx) | `+0` | `1` | `1` | `+infinity` | NaN -> NaN |
| `EXPM1` | `-1` | `-0` | `+0` | `+infinity` | NaN -> NaN |
| [`ERF`](../../../../glossary.md#term-erf) | `-1` | `-0` | `+0` | `+1` | NaN -> NaN |
| `SQRT` | NaN | `-0` | `+0` | `+infinity` | negative finite -> NaN; NaN -> NaN |
| `RSQRT` | NaN | `-infinity` | `+infinity` | `+0` | negative finite -> NaN; NaN -> NaN |
| `FLOOR`, `CEIL` | `-infinity` | preserve `-0` | preserve `+0` | `+infinity` | NaN -> NaN |
| `SIGN` | `-1` | preserve `-0` | preserve `+0` | `+1` | NaN -> NaN |
| `RELU` | `+0` | `+0` | `+0` | `+infinity` | NaN -> NaN |
| [`SIGMOID`](../../../../glossary.md#term-sigmoid) | `+0` | `0.5` | `0.5` | `+1` | NaN -> NaN |
| [`TANH`](../../../../glossary.md#term-tanh) | `-1` | `-0` | `+0` | `+1` | NaN -> NaN |
| both [GELU](../../../../glossary.md#gaussian-error-linear-unit--gelu) kinds | `-0` | `-0` | `+0` | `+infinity` | NaN -> NaN |
| `SILU` | `-0` | `-0` | `+0` | `+infinity` | NaN -> NaN |

Finite [overflow](../../../../glossary.md#term-overflow), [underflow](../../../../glossary.md#term-underflow), and [subnormal](../../../../glossary.md#term-subnormal) transitions follow the listed operations. The task adds
no [flush-to-zero](../../../../glossary.md#subnormal-daz-ftz-overflow-and-underflow), finite-only assumption, [reassociation](../../../../glossary.md#term-reassociation), contraction, reciprocal estimate, or
alternate exceptional-value mode.

### [Conformance](../../../../glossary.md#term-conformance) tolerances

Use an independent high-quality mathematical [oracle](../../../../glossary.md#term-test-oracle) where practical and the exact fixed [Model](../../../../glossary.md#term-model)
formula for `GELU_TANH_APPROXIMATION`. The task-level [CPU](../../../../glossary.md#term-cpu) tests must apply these explicit bounds:

| Result family | [FLOAT64](../../../../glossary.md#term-float64) bound | [FLOAT32](../../../../glossary.md#term-float32) bound |
|---|---:|---:|
| exact/primitive rows and fixed special cases | exact bits when the table fixes a non-[NaN](../../../../glossary.md#term-nan) result | exact bits when the table fixes a non-NaN result |
| [`LOG`](../../../../glossary.md#term-log), `LOG1P`, [`EXP`](../../../../glossary.md#term-expx), `EXPM1` [scalar](../../../../glossary.md#term-scalar) | Java method contract, at most 1 ulp from exact | exact [widening](../../../../glossary.md#term-widening), method result, and one [narrowing](../../../../glossary.md#term-narrowing); at most 1 [binary32](../../../../glossary.md#floating-point-precision-and-rounding) ulp from [correctly rounded](../../../../glossary.md#nan-infinity-and-signed-zero) oracle unless [overflow](../../../../glossary.md#term-overflow)/[underflow](../../../../glossary.md#term-underflow) fixes the result |
| `SQRT` scalar | correctly rounded | exact widening plus one final binary32 narrowing |
| [`TANH`](../../../../glossary.md#term-tanh) scalar | at most 2.5 ulps | at most 2.5 binary32 ulps after final narrowing |
| [`ERF`](../../../../glossary.md#term-erf), [`SIGMOID`](../../../../glossary.md#term-sigmoid), both [GELU](../../../../glossary.md#gaussian-error-linear-unit--gelu) kinds, `SILU` | `max(2e-7, 2e-7 * abs(expected))` | `max(2e-5, 2e-5 * abs(expected))` |
| FLOAT64 [vector](../../../../glossary.md#term-vector) `EXP`/`LOG`/`LOG1P`/`EXPM1` versus scalar oracle | at most 2 ulps | not applicable |
| FLOAT64 vector `TANH` versus scalar oracle | at most 5 ulps | not applicable |
| vector `ERF`/`GELU` versus scalar oracle | `max(2e-7, 2e-7 * abs(expected))` | not applicable |

`RSQRT`, stable [activation](../../../../glossary.md#term-activation) helpers, and GELU helpers are additionally differential-tested against
the scalar reference in the exact selected [operation](../../../../glossary.md#operation) order. The tolerances are correctness
bounds, not permission for another algorithm family or relaxed mode. Random agreement alone is
insufficient: boundary-directed and special-value cases are mandatory.

### Capability, [lowering](../../../../glossary.md#lowering), [IR](../../../../glossary.md#term-intermediate-representation-ir), and generated code

- Extend `CpuCapabilityProvider` to advertise exactly the final [FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64) [matrix](../../../../glossary.md#term-matrix-meanings) above.
- Extend `CpuPointwiseOpcode` from thirty-one to exactly forty-eight values by adding one opcode
  for each of the seventeen previously unsupported <span data-glossary-navigation="true"> ([opcode glossary](../../../../glossary.md#term-opcode))</span>`UnaryElementwiseKind` values. Retain `NEG`
  and `GELU_EXACT`; do not add duplicates for already-implemented semantics.
- Keep all [unary](../../../../glossary.md#term-unary) opcodes in `Family.UNARY`, with one input, input-typed result, no immediate,
  power realization, or clamp fact. IR validation admits FLOAT32/FLOAT64 for every unary opcode.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Extend the existing kind-to-opcode switch, scalar emitter, vector helper, and scalar reference
  switch. Add no operation-specific lowerer, emitter, kernel, route, executable, or planner class.<span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector); [operation glossary](../../../../glossary.md#operation); [kernel glossary](../../../../glossary.md#kernel))</span>
- Generated [scalar](../../../../glossary.md#term-scalar) code may invoke narrow static methods on `CpuScalarReferenceKernel` for the
  shared ERF/activation algorithms. Those methods remain pure primitive helpers used by generated
  bytecode; Runtime never passes them Model operations or IR.<span data-glossary-navigation="true"> ([ERF glossary](../../../../glossary.md#term-erf); [activation glossary](../../../../glossary.md#term-activation); [bytecode glossary](../../../../glossary.md#term-bytecode); [Runtime glossary](../../../../glossary.md#term-runtime); [Model glossary](../../../../glossary.md#term-model))</span>
- Preserve one-through-eight connected straight-line pointwise lowering, stored instruction order,
  virtual intermediates, and one final store. Every unary occurrence is one IR instruction.<span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Correct existing GELU special-value handling in scalar/reference and vector helpers as part of
  closure; do not retain two competing GELU algorithms.<span data-glossary-navigation="true"> ([GELU glossary](../../../../glossary.md#gaussian-error-linear-unit--gelu))</span>

### Routes, access, [materialization](../../../../glossary.md#materialization), and [lifecycle](../../../../glossary.md#lifecycle)

- Scalar and parallel-scalar cover every final unary row over contiguous, strided, broadcast,
  scalar/all-zero, last-axis, block/outer, and general-odometer access as already normalized by
  the five <span data-glossary-navigation="true"> ([Scalar glossary](../../../../glossary.md#term-scalar); [unary glossary](../../../../glossary.md#term-unary); [contiguous glossary](../../../../glossary.md#term-contiguous); [axis glossary](../../../../glossary.md#term-axis))</span>`CpuAccessPlan` regimes. Unary output Shape equals input Shape, but input layouts may
  still be strided or zero-stride and outputs must remain injective.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [layouts glossary](../../../../glossary.md#layout); [stride glossary](../../../../glossary.md#term-stride); [injective glossary](../../../../glossary.md#term-injective))</span>
- Vector and parallel-vector are eligible only for the FLOAT64 rows in the matrix, and only when
  every existing direct contiguous/scalar-broadcast access, carrier, topology, and strategy gate
  passes. General odometers remain scalar. Scalar tails remain unchanged.<span data-glossary-navigation="true"> ([Vector glossary](../../../../glossary.md#term-vector); [FLOAT64 glossary](../../../../glossary.md#term-float64); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [carrier glossary](../../../../glossary.md#term-storage-carrier); [topology glossary](../../../../glossary.md#term-topology))</span>
- Preserve arbitrary `start`/`end` ranges, [heap](../../../../glossary.md#term-heap) arrays, native-order [`MemorySegment`](../../../../glossary.md#term-memorysegment) carriers,
  mixed carrier patterns, cold binding, and concurrent-run isolation.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [run glossary](../../../../glossary.md#term-run))</span>
- Preserve direct execution plus at most three eligible one-input materialization candidates, at
  most one selected contiguous FLOAT64 input copy, four complete candidates, one realized
  artifact, zero fixed-shape variants, and zero unrolled variants.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [unrolled glossary](../../../../glossary.md#backpropagation-through-time-and-recurrent-state))</span>
- Reference code remains conformance/fallback support. It does not become a competing route,
  another prepared executable, or hot Runtime interpretation.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [prepared executable glossary](../../../../glossary.md#prepared-executable--preparedexecutable); [Runtime glossary](../../../../glossary.md#term-runtime))</span>

### [Schema](../../../../glossary.md#term-schema) and [cache](../../../../glossary.md#term-cache) compatibility

- Canonical IR identity includes each exact unary opcode. Route strategy, vector species,
  access/carrier/materialization facts, numerical mode, and lowering fingerprint remain existing
  specialization facts.<span data-glossary-navigation="true"> ([Canonical glossary](../../../../glossary.md#term-canonical-form); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [identity glossary](../../../../glossary.md#term-identity-meanings); [unary glossary](../../../../glossary.md#term-unary); [opcode glossary](../../../../glossary.md#term-opcode); [vector species glossary](../../../../glossary.md#term-vector-species); [carrier glossary](../../../../glossary.md#term-storage-carrier); [materialization glossary](../../../../glossary.md#materialization); [lowering glossary](../../../../glossary.md#lowering); [fingerprint glossary](../../../../glossary.md#term-fingerprint); [specialization glossary](../../../../glossary.md#term-specialization))</span>
- Bump `CpuGeneratorSchema.CURRENT_VERSION` from 7 to 8 because the closed opcode vocabulary,
  scalar helper calls, vector eligibility, and corrected GELU bytecode change.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [vector glossary](../../../../glossary.md#term-vector); [GELU glossary](../../../../glossary.md#gaussian-error-linear-unit--gelu); [bytecode glossary](../../../../glossary.md#term-bytecode))</span>
- Schema 8 accepts only current compatible envelopes. Every schema 7 or older artifact is an
  incompatible miss; add no migration reader or converter.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact))</span>
- Preserve deterministic bytes, structural verification, process-local compatible interning,
  assignment-before-artifact ordering, strong prepared ownership, optional bounded persistence,
  and the CPU 0005D <span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [persistence glossary](../../../../glossary.md#term-persistence); [CPU glossary](../../../../glossary.md#term-cpu))</span>`KEEP_DISABLED` default.
- Do not confuse generated-class compatibility with a future workload-tuning cache. No tuning,
  benchmark, cache-policy, or Runtime cache behavior changes.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime))</span>

### CAST and classification audit

- Keep same-type CAST exactly as implemented for [FLOAT64](../../../../glossary.md#term-float64), [FLOAT32](../../../../glossary.md#term-float32), [INT32](../../../../glossary.md#data-type--datatype), [INT64](../../../../glossary.md#data-type--datatype), and [BOOL](../../../../glossary.md#data-type--datatype).
- Keep every cross-type CAST [fail-closed](../../../../glossary.md#term-fail-closed). Live [Model](../../../../glossary.md#term-model) `CastKind`/`CastAttrs` permit expression
  construction but still do not define numerical conversion, rounding, saturation, NaN, infinity,
  or BOOL conversion semantics. Java casts and Vector conversions are therefore not authority to
  advertise a backend route.<span data-glossary-navigation="true"> ([rounding glossary](../../../../glossary.md#term-rounding); [saturation glossary](../../../../glossary.md#term-saturation); [NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [Vector glossary](../../../../glossary.md#term-vector); [backend route glossary](../../../../glossary.md#backend-route))</span>
- Preserve `IS_FINITE`, `IS_NAN`, and `IS_INF` for FLOAT32/FLOAT64 to canonical BOOL. They are
  already complete classification-family opcodes and are not renamed, duplicated, or counted
  among the seventeen additions.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- public `fastExp`, `fastTanh`, `swish`, configurable GELU coefficients, selectable activation
  approximation, general fast math, relaxed numerical permission, or a new Config policy<span data-glossary-navigation="true"> ([GELU glossary](../../../../glossary.md#gaussian-error-linear-unit--gelu); [activation glossary](../../../../glossary.md#term-activation))</span>
- BFLOAT16/FLOAT16 unary execution, mixed precision, cross-type CAST, CPU-side promotion,
  conversion helpers, integral/BOOL unary arithmetic, or another data type<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT16 glossary](../../../../glossary.md#term-float16); [unary glossary](../../../../glossary.md#term-unary); [mixed precision glossary](../../../../glossary.md#term-mixed-precision); [BOOL glossary](../../../../glossary.md#data-type--datatype); [data type glossary](../../../../glossary.md#data-type--datatype))</span>
- explicit scalar-lane extraction loops inside vector kernels, a claim of hardware transcendental
  intrinsics, FLOAT32 Vector API support, new species, gather, or masked tails<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [lane glossary](../../../../glossary.md#term-lane); [vector glossary](../../../../glossary.md#term-vector); [kernels glossary](../../../../glossary.md#kernel); [FLOAT32 glossary](../../../../glossary.md#term-float32); [Vector API glossary](../../../../glossary.md#term-vector-api); [gather glossary](../../../../glossary.md#gather))</span>
- fixed-shape or unrolled specialization, directed acyclic graph (DAG) fusion, epilogues,
  operation decomposition/recomposition, compiler constant discovery, graph rewrite, or
  publication remapping<span data-glossary-navigation="true"> ([DAG glossary](../../../../glossary.md#intermediate-representation-dag-and-ssa))</span><span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [unrolled glossary](../../../../glossary.md#backpropagation-through-time-and-recurrent-state); [specialization glossary](../../../../glossary.md#term-specialization); [directed acyclic graph glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [fusion glossary](../../../../glossary.md#term-fusion); [operation glossary](../../../../glossary.md#operation); [decomposition glossary](../../../../glossary.md#term-decomposition); [compiler glossary](../../../../glossary.md#term-compiler); [constant glossary](../../../../glossary.md#term-constant); [graph glossary](../../../../glossary.md#graph); [publication glossary](../../../../glossary.md#term-publication))</span>
- reductions, scans, softmax, normalization, layout/indexing/random families, native libraries,
  OpenBLAS/vendor routes, autotuning, benchmarks, Engine composition, or later CPU tasks<span data-glossary-navigation="true"> ([softmax glossary](../../../../glossary.md#softmax--log-softmax); [normalization glossary](../../../../glossary.md#term-normalization); [layout glossary](../../../../glossary.md#layout); [OpenBLAS glossary](../../../../glossary.md#term-openblas); [autotuning glossary](../../../../glossary.md#autotuning--model-autotuning); [Engine glossary](../../../../glossary.md#term-engine))</span>
- per-operation production classes, another IR, another compiler/interpreter, a public CPU facade,
  another backend identity, dependency/Gradle/build changes, or architecture changes<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [facade glossary](../../../../glossary.md#term-facade); [backend identity glossary](../../../../glossary.md#backend-identity--backendid); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- Model, Compiler, Config, Planning, shared Prepare, Runtime, Backend Contract, Trace, Engine,
  OpenBLAS provider, other backend, backend-conformance, or integration source changes<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Backend glossary](../../../../glossary.md#backend); [Trace glossary](../../../../glossary.md#trace); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- a detailed task specification for [CPU](../../../../glossary.md#term-cpu) 0005H1, 0006, or any later task

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Runtime / Prepare / Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Backend glossary](../../../../glossary.md#backend))</span>
- [Backend-owned lowering ADR](../../../../design/decisions/0002-backend-owned-lowering.md)<span data-glossary-navigation="true"> ([Backend-owned lowering glossary](../../../../glossary.md#backend-owned-lowering); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- [Staged backend preparation ADR](../../../../design/decisions/0010-staged-backend-preparation.md)
- [Per-run Runtime ownership ADR](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [General documentation style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- [Backend-guide style](../../../../developer-guide/documentation/backend-guide-style.md)
- [Planning style](../../../../developer-guide/documentation/planning-style.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [CPU master plan](../master-plan.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- [CPU backend guide](../../../../backend-guide/cpu-backend.md)
- [Task 0005F](0005f-floating-division-and-exact-scalar-power-realization.md)
- [Task 0005G](0005g-extrema-clamp-tensor-power-and-logical-coverage.md)
- [Model task 0018P](../../../modules/model/tasks/0018p-elementwise-semantic-cleanup.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Model task 0018T1](../../../modules/model/tasks/0018t1-unary-numeric-gaps-and-floating-diagnostics.md)
- [Model task 0019A](../../../modules/model/tasks/0019a-modern-activation-semantics-and-tensor-expressions.md)

## Architecture constraints

- Model owns the unary semantic identities and fixed activation formulas. CPU may select a
  conforming internal algorithm but may not add a Model kind, Tensor method, parameter, or route
  hint.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [unary glossary](../../../../glossary.md#term-unary); [activation glossary](../../../../glossary.md#term-activation); [CPU glossary](../../../../glossary.md#term-cpu); [Tensor glossary](../../../../glossary.md#tensor); [parameter glossary](../../../../glossary.md#term-parameter))</span>
- Planning selects CPU ownership. CPU analysis owns capability truth, lowering, exact/default
  numerical eligibility, compute strategy, specialization, and declarations; CPU finalization
  owns generated artifact realization after shared slot assignment.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [lowering glossary](../../../../glossary.md#lowering); [specialization glossary](../../../../glossary.md#term-specialization); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Generated code and [Runtime](../../../../glossary.md#term-runtime) never inspect [`Operation`](../../../../glossary.md#operation), `CompiledNode`, Model attributes, or
  strings. Common lowering translates each occurrence once into typed CPU-private IR.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- Compiler owns graph rewrites and gradients. CPU retains every occurrence and applies the same
  forward policy to forward and compiler-generated operations.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [graph glossary](../../../../glossary.md#graph); [gradients glossary](../../../../glossary.md#term-gradient))</span>
- Shared Prepare continues to see one opaque CPU plan plus exact declarations. Runtime performs
  cold checked binding and direct invocation only.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [opaque glossary](../../../../glossary.md#term-opaque); [binding glossary](../../../../glossary.md#term-binding))</span>
- Bytecode-first portable execution remains the semantic baseline. Reference code is a
  conformance/fail-closed helper, not another backend route.<span data-glossary-navigation="true"> ([Bytecode glossary](../../../../glossary.md#term-bytecode); [conformance glossary](../../../../glossary.md#term-conformance); [fail-closed glossary](../../../../glossary.md#term-fail-closed); [backend route glossary](../../../../glossary.md#backend-route))</span>
- No module dependency, package responsibility, lifecycle, public API, architecture rule, or
  Gradle configuration changes. Stop and report if implementation requires one.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [lifecycle glossary](../../../../glossary.md#lifecycle); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Gradle glossary](../../../../glossary.md#term-gradle))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu` — truthful [occurrence](../../../../glossary.md#term-occurrence)-local capability reporting.
- `io.github.pho001.synaptik.backend.cpu.internal.ir` — the closed family opcode vocabulary and
  typed canonical instruction validation.<span data-glossary-navigation="true"> ([opcode glossary](../../../../glossary.md#term-opcode); [canonical glossary](../../../../glossary.md#term-canonical-form); [validation glossary](../../../../glossary.md#term-validation))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.lowering` — common [Model](../../../../glossary.md#term-model)-kind-to-[IR](../../../../glossary.md#term-intermediate-representation-ir) [lowering](../../../../glossary.md#lowering).
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — scalar bytecode emission,
  FLOAT64 vector helper emission, and generated class validation.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [bytecode emission glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [FLOAT64 glossary](../../../../glossary.md#term-float64); [vector glossary](../../../../glossary.md#term-vector))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.reference` — pure scalar conformance helpers over
  already-lowered primitive semantics.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — current generator [schema](../../../../glossary.md#term-schema) compatibility.
- Existing prepare, route, executable, memory, and cache packages — review-only unless current
  validated opcode/vector facts already flow through one of their existing contracts.<span data-glossary-navigation="true"> ([prepare glossary](../../../../glossary.md#prepare); [cache glossary](../../../../glossary.md#term-cache))</span>

Packages added, moved, or removed:

- None.

Type placement:

- The seventeen new opcodes remain values of `CpuPointwiseOpcode`.
- Shared [ERF](../../../../glossary.md#term-erf) and [activation](../../../../glossary.md#term-activation) algorithms remain static primitive methods on the existing
  `CpuScalarReferenceKernel`, callable by generated [bytecode](../../../../glossary.md#term-bytecode) and reference [evaluation](../../../../glossary.md#term-evaluation).
- No new production type is planned. If a correct implementation needs one, stop and revise this
  specification before implementation continues.

## Affected files

Expected [CPU](../../../../glossary.md#term-cpu) production/package paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProvider.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGenerator.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuScalarEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuVectorEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuKernelIr.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuPointwiseOpcode.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionLowering.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuScalarReferenceKernel.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/package-info.java`

Expected CPU test paths:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProviderTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStoreTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuFusedGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuPointwiseGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuKernelIrTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuPointwiseOpcodeTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPointwisePartitionLoweringTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuReferenceDifferentialTest.java`

Expected implementation documentation and [planning](../../../../glossary.md#term-planning) paths:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- this task specification
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

Review-only paths include current [Model](../../../../glossary.md#term-model) [unary](../../../../glossary.md#term-unary)/classification kinds, [Tensor](../../../../glossary.md#tensor) methods and tests,
[DataType](../../../../glossary.md#data-type--datatype)/[ScalarValue](../../../../glossary.md#typed-scalar-value--scalarvalue) contracts; Tensor/[Compile](../../../../glossary.md#compile)/[Backend](../../../../glossary.md#backend)/Public APIs; Config, Planning, [Prepare](../../../../glossary.md#prepare), and
[Runtime](../../../../glossary.md#term-runtime) contracts; generated-[artifact](../../../../glossary.md#term-artifact)/[specialization](../../../../glossary.md#term-specialization)/route/executable sources and tests not listed
above; [Gradle](../../../../glossary.md#term-gradle); architecture/[ADR](../../../../glossary.md#term-architecture-decision-record-adr)/tests; backend [conformance](../../../../glossary.md#term-conformance); and integration tests.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most 30 paths:

| Category | Maximum | Path accounting |
|---|---:|---|
| [CPU](../../../../glossary.md#term-cpu) production/package | 16 | The exact sixteen listed production/package paths |
| CPU tests | 9 | The exact nine listed test paths |
| Explanatory documentation | 2 | CPU [backend](../../../../glossary.md#backend) guide and glossary |
| [Planning](../../../../glossary.md#term-planning)/status | 3 | This task, CPU master plan, and roadmap |
| **Total** | **30** | **16 + 9 + 2 + 3** |

The path count exceeds the usual guardrail for one documented technical reason: capability,
closed [opcode](../../../../glossary.md#term-opcode)/[IR](../../../../glossary.md#term-intermediate-representation-ir) [validation](../../../../glossary.md#term-validation), [lowering](../../../../glossary.md#lowering), [scalar](../../../../glossary.md#term-scalar) and [vector](../../../../glossary.md#term-vector) generated emission, reference behavior,
[schema](../../../../glossary.md#term-schema) compatibility, and their existing focused tests must evolve atomically before any [unary](../../../../glossary.md#term-unary)
row is advertised. The implementation adds no production class and remains one cohesive semantic
family closure. Splitting a Draft 0005H1 would create an artificial partial unary [matrix](../../../../glossary.md#term-matrix-meanings) and
duplicate the same family-switch/schema work, so no follow-up split is planned.

The path map is exact. Unused authorized paths may remain unchanged, but no substitute or extra
path is implied. Stop and revise planning if another path or a new production type is required.

## Acceptance criteria

- Capability reports exactly all nineteen `UnaryElementwiseKind` values for same-typed,
  shape-preserving, fully static resolved FLOAT32/FLOAT64 occurrences and rejects every adjacent
  unsupported or malformed row before declarations or artifact work.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- `NEG` remains supported for both precisions; [`GELU`](../../../../glossary.md#gaussian-error-linear-unit--gelu) expands from FLOAT64 to both precisions and
  returns negative zero at negative infinity; the other seventeen kinds are added without
  duplicate opcodes.<span data-glossary-navigation="true"> ([negative zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [negative infinity glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- `IS_FINITE`, `IS_NAN`, and `IS_INF` remain the existing three classification opcodes for
  FLOAT32/FLOAT64 to canonical BOOL and are not reimplemented as unary numeric operations.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [BOOL glossary](../../../../glossary.md#data-type--datatype); [unary glossary](../../../../glossary.md#term-unary))</span>
- Scalar/reference algorithms, special values, signed zeros, domains, overflow/underflow, and
  tolerances match the matrices in this task. NaNs are classified without a payload promise.<span data-glossary-navigation="true"> ([Scalar glossary](../../../../glossary.md#term-scalar); [overflow glossary](../../../../glossary.md#term-overflow); [underflow glossary](../../../../glossary.md#term-underflow); [matrices glossary](../../../../glossary.md#term-matrix-meanings); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication))</span>
- Stable sigmoid and SiLU avoid avoidable exponential overflow; both GELU variants and SiLU
  implement their continuous negative-infinity extension explicitly.<span data-glossary-navigation="true"> ([sigmoid glossary](../../../../glossary.md#term-sigmoid); [SiLU glossary](../../../../glossary.md#sigmoid-linear-unit--silu); [infinity glossary](../../../../glossary.md#term-infinity))</span>
- The closed opcode inventory becomes exactly forty-eight values and schema 8 rejects every
  older envelope without migration.<span data-glossary-navigation="true"> ([opcode glossary](../../../../glossary.md#term-opcode); [schema glossary](../../../../glossary.md#term-schema))</span>
- Scalar and parallel-scalar execute every final unary row for both types across all five access
  regimes, zero-element ranges, arbitrary start/end bounds, and heap/segment/mixed carriers.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [heap glossary](../../../../glossary.md#term-heap))</span>
- FLOAT64 vector and parallel-vector eligibility matches the exact table. No vector-ineligible
  opcode fails support; it selects scalar compute. No explicit per-lane scalar fallback loop or
  hardware-intrinsic claim is added.<span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector); [lane glossary](../../../../glossary.md#term-lane))</span>
- Every unary occurrence remains one Model node and one IR instruction. One-through-eight fusion,
  virtual intermediates, one final store, access/injectivity/alias rules, one-copy/four-candidate/
  one-artifact budgets, and zero fixed-shape/unroll variants remain unchanged.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [Model glossary](../../../../glossary.md#term-model); [node glossary](../../../../glossary.md#node); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [fusion glossary](../../../../glossary.md#term-fusion); [injectivity glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Cross-type CAST and BFLOAT16 execution remain fail-closed. Same-type CAST and all 31 pre-0005H
  opcodes retain their completed behavior.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- No public fast-math API, relaxed policy, graph rewrite, operation-specific class, native route,
  tuning work, dependency/build/shared-module/architecture change, or later detailed task appears.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [graph glossary](../../../../glossary.md#graph); [operation glossary](../../../../glossary.md#operation); [dependency glossary](../../../../glossary.md#term-dependency); [module glossary](../../../../glossary.md#term-module))</span>
- A separate clean documentation-focused pass finalizes affected Javadocs/package summaries, CPU
  guide, glossary, task evidence, master plan, and roadmap; records reasoned no-change conclusions;
  and reuses stabilized Java evidence unless it changes executable behavior.
- [CPU](../../../../glossary.md#term-cpu) 0005A–0005G remain `Complete`; CPU 0005H remains [`Ready`](../../../../glossary.md#term-ready) until implementation, validation,
  documentation review, and final synchronization pass; CPU 0006–0017 remain <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [synchronization glossary](../../../../glossary.md#term-synchronization))</span>`Draft` without
  detailed specifications.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) focused implementation tests while executable Java changes:

```bash
./gradlew :backends:cpu:test \
  --tests io.github.pho001.synaptik.backend.cpu.CpuCapabilityProviderTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcodeTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIrTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPointwisePartitionLoweringTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuPointwiseGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuFusedGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.reference.CpuReferenceDifferentialTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratedKernelArtifactStoreTest
```

The focused [matrix](../../../../glossary.md#term-matrix-meanings) must include every [unary](../../../../glossary.md#term-unary) kind and both precisions; exact special values and
[signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero) bits; domain boundaries and adjacent `nextUp`/`nextDown` values; [subnormal](../../../../glossary.md#term-subnormal), normal,
[overflow](../../../../glossary.md#term-overflow), and [underflow](../../../../glossary.md#term-underflow) cases; [scalar](../../../../glossary.md#term-scalar) and parallel-scalar; every eligible [FLOAT64](../../../../glossary.md#term-float64) [vector](../../../../glossary.md#term-vector) and
parallel-vector row; vector-ineligible fallback selection; all five access regimes; representative
broadcast/strided input with [injective](../../../../glossary.md#term-injective) output; zero [elements](../../../../glossary.md#term-element); arbitrary subranges; [heap](../../../../glossary.md#term-heap), segment,
and mixed carriers; [fused](../../../../glossary.md#specialization-and-fusion) chains; [schema](../../../../glossary.md#term-schema) incompatibility; and fail-before-[artifact](../../../../glossary.md#term-artifact) rejections.

After executable Java stabilizes, run exactly one final affected-[module](../../../../glossary.md#term-module) suite:

```bash
./gradlew :backends:cpu:test
```

The ordinary suite keeps the opt-in [CPU](../../../../glossary.md#term-cpu) 0005D [persistence](../../../../glossary.md#term-persistence) timing method skipped. Do not rerun the
timing evidence because this task changes neither the persistence verdict nor performance policy.

The separate documentation-focused pass, after final Javadocs and Markdown, runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

It also records checks for local Markdown links and heading anchors, balanced fences, final
newlines, trailing whitespace, exact authorized paths and the 30-path ceiling, nineteen-kind/
three-classification/forty-eight-[opcode](../../../../glossary.md#term-opcode) inventories, type/route/special-value/[tolerance](../../../../glossary.md#term-tolerance) matrices,
schema 8, unchanged [fusion](../../../../glossary.md#term-fusion)/access/[materialization](../../../../glossary.md#materialization) budgets, [fail-closed](../../../../glossary.md#term-fail-closed) CAST/[BFLOAT16](../../../../glossary.md#term-bfloat16) rows, and
synchronized status. It confirms no Java/test path outside `backends/cpu`, [Gradle](../../../../glossary.md#term-gradle),
`ARCHITECTURE.md`, focused architecture/[ADR](../../../../glossary.md#term-architecture-decision-record-adr)/test paths, [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), or integration path
changed.

Repository-wide validation is deferred to CPU 0009, the portable generated-coverage [checkpoint](../../../../glossary.md#term-checkpoint),
or [continuous integration](../../../../glossary.md#term-continuous-integration-ci). Backend-conformance integration remains deferred to that checkpoint
because no composed [Engine](../../../../glossary.md#term-engine)/backend harness exists at this frontier; the current task must provide
the equivalent CPU-internal generated/reference/independent-[oracle](../../../../glossary.md#term-test-oracle) conformance matrix now.

## Dependencies

- Complete CPU 0005A–0005G partition lowering, access/carrier, strategy, materialization,
  pointwise-family, exact power, extrema/clamp, logical, and schema-7 foundations.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [partition glossary](../../../../glossary.md#partition); [lowering glossary](../../../../glossary.md#lowering); [carrier glossary](../../../../glossary.md#term-storage-carrier); [materialization glossary](../../../../glossary.md#materialization); [pointwise glossary](../../../../glossary.md#term-pointwise); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [schema glossary](../../../../glossary.md#term-schema))</span>
- Complete Model 0018P final unary vocabulary, 0018T1 numeric gaps/classifications, and 0019A
  activation formulas and special-value extensions.<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary); [activation glossary](../../../../glossary.md#term-activation))</span>
- Current [Model](../../../../glossary.md#term-model) `UnaryElementwiseKind`, `FloatingClassificationKind`, Tensor unary/classification
  construction, exact DataType, Shape, and resolved-layout contracts.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [DataType glossary](../../../../glossary.md#data-type--datatype); [Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout))</span>
- Current Java 26 `StrictMath`/`Math`, Class-File API, and incubating Vector API contracts and the
  CPU module's existing <span data-glossary-navigation="true"> ([Class-File API glossary](../../../../glossary.md#term-class-file-api); [Vector API glossary](../../../../glossary.md#term-vector-api); [module glossary](../../../../glossary.md#term-module))</span>`jdk.incubator.vector` configuration.
- Current staged [Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime) cold-[binding](../../../../glossary.md#term-binding)/execution, and [backend](../../../../glossary.md#backend)-contract [ownership](../../../../glossary.md#term-ownership-meanings).

## Follow-up tasks

- CPU 0006 remains Draft without a detailed specification and follows this completed unary
  closure for layout, indexing, ordering, and random family coverage.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [unary glossary](../../../../glossary.md#term-unary); [layout glossary](../../../../glossary.md#layout))</span>
- CPU 0007–0008C remain Draft for reductions/normalization, heavy portable families, and later DAG
  decomposition/recognition/profitability. CPU 0009 remains the portable closure checkpoint.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [decomposition glossary](../../../../glossary.md#term-decomposition); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Config 0006 and CPU 0017 remain Draft for future explicit relaxed numerical permission. Nothing
  in this task preauthorizes those candidates.
- Cross-type CAST requires a future completed Model conversion contract before any CPU task may
  advertise it. No CAST follow-up specification is created here.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- No CPU 0005H1 task is needed: the exact 30-path atomic family evolution is documented and adds
  no production class.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback))</span>

## Architecture impact

Expected impact: None.

The task implements existing [backend-owned lowering](../../../../glossary.md#backend-owned-lowering), numerical realization, strategy selection,
[artifact](../../../../glossary.md#term-artifact) compatibility, and executable construction. It changes no [owner](../../../../glossary.md#term-owner), [dependency](../../../../glossary.md#term-dependency), [lifecycle](../../../../glossary.md#lifecycle),
[backend identity](../../../../glossary.md#backend-identity--backendid), public surface, or shared [IR](../../../../glossary.md#term-intermediate-representation-ir). If implementation requires any such change, stop
and report the exact conflict.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, the CPU master plan, completed
CPU tasks 0005A–0005G, and
docs/planning/backends/cpu/tasks/0005h-portable-unary-transcendental-and-activation-closure.md in
full. Read every affected and directly relevant Model/Compiler/Config/Prepare/Runtime contract
named by the task before editing.

Implement CPU 0005H exactly inside its exact 30-path map. Close all nineteen Model unary kinds for
same-typed fully static resolved FLOAT32/FLOAT64 occurrences through the existing capability,
opcode/IR, lowering, scalar/vector emitter, reference, prepare, and schema pipeline. Preserve the
already-complete classification rows and all earlier pointwise behavior. Use only the specified
algorithms, special-value rules, tolerances, and honest Java 26 FLOAT64 vector eligibility. Keep
cross-type CAST and BFLOAT16 fail-closed. Do not add fast-math APIs or policy, per-operation
classes, explicit vector lane scalarization, graph rewrites, fixed-shape/unrolled/DAG fusion,
native/tuning work, public/shared/build/architecture changes, later task specs, commits, or pushes.
Stop on a semantic, architecture, dependency, affected-file, or maximum-scope conflict.

After executable Java stabilizes and the focused plus exactly one final CPU suite pass, hand the
actual diff and exact Java evidence to a separate clean documentation-focused agent/thread. That
pass must follow docs/developer-guide/documentation-rules.md; independently finalize affected
Javadocs/package summaries, CPU guide, glossary, planning evidence, links, scope/status checks,
and reasoned no-change conclusions; and not repeat successful Java tests unless executable
behavior changes or a concrete stale-evidence risk is recorded.

Update this task's decisions, limitations, evidence, implementation notes, completion summary,
and synchronized final status. Do not mark Complete until every criterion and the documentation
pass succeed. Leave CPU 0006 and every later CPU task Draft without a detailed specification.
```

## Local decisions

- Close the whole remaining unary family in 0005H. The work is one atomic evolution of existing
  family switches and schema, uses no new production class, and is safer than a partial 0005H1
  split that would duplicate compatibility work.<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [schema glossary](../../../../glossary.md#term-schema))</span>
- [Scalar](../../../../glossary.md#term-scalar) [FLOAT64](../../../../glossary.md#term-float64) uses Java 26 `StrictMath` plus the existing ERF approximation and stable
  activation helpers. FLOAT32 uses exact represented widening and one final narrowing where the
  JDK lacks a float method.<span data-glossary-navigation="true"> ([ERF glossary](../../../../glossary.md#term-erf); [activation glossary](../../../../glossary.md#term-activation); [FLOAT32 glossary](../../../../glossary.md#term-float32); [widening glossary](../../../../glossary.md#term-widening); [narrowing glossary](../../../../glossary.md#term-narrowing); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk))</span>
- GELU keeps one semantic exact-target opcode even though its portable finite realization is the
  existing bounded approximation. <span data-glossary-navigation="true"> ([GELU glossary](../../../../glossary.md#gaussian-error-linear-unit--gelu); [target glossary](../../../../glossary.md#term-target); [opcode glossary](../../../../glossary.md#term-opcode); [finite glossary](../../../../glossary.md#term-finite))</span>`GELU_TANH_APPROXIMATION` is a separate fixed semantic formula,
  not permission for general approximate math.
- Only direct Java 26 FLOAT64 lane operators, exact division/sqrt composition, and the existing
  ERF/GELU vector helper are vector-eligible. Missing operators and unproved activation
  compositions select scalar compute; no explicit per-lane fallback loop is allowed.<span data-glossary-navigation="true"> ([lane glossary](../../../../glossary.md#term-lane); [vector glossary](../../../../glossary.md#term-vector))</span>
- Existing GELU negative-infinity NaN behavior is incompatible with the Model continuous
  extension and is corrected in the same closure task rather than preserved as historical output.<span data-glossary-navigation="true"> ([infinity glossary](../../../../glossary.md#term-infinity); [NaN glossary](../../../../glossary.md#term-nan); [Model glossary](../../../../glossary.md#term-model))</span>
- Cross-type CAST remains unsupported because live Model contracts still do not define numerical
  conversion. Java primitive/vector conversion availability cannot supply Model semantics.
- Schema 8 is required by the opcode and generated-bytecode changes. No migration reader is
  justified.<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode))</span>

## Known limitations

- Execution remains limited to fully static, resolved-layout occurrences inside the current
  one-through-eight connected straight-line pointwise topology.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [pointwise glossary](../../../../glossary.md#term-pointwise); [topology glossary](../../../../glossary.md#term-topology))</span>
- BFLOAT16 remains storage-only for this family, FLOAT16 is absent, and cross-type CAST remains
  unsupported.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT16 glossary](../../../../glossary.md#term-float16))</span>
- FLOAT32 and vector-ineligible FLOAT64 chains use scalar compute with optional parallel
  orchestration. Vector math-library operators carry no promise of a hardware intrinsic or
  performance improvement.<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [vector glossary](../../../../glossary.md#term-vector); [FLOAT64 glossary](../../../../glossary.md#term-float64); [scalar glossary](../../../../glossary.md#term-scalar); [carry glossary](../../../../glossary.md#term-carry))</span>
- ERF and activation-family finite results use the explicit tolerances in this task; no NaN
  payload or cross-platform bitwise promise is made.<span data-glossary-navigation="true"> ([ERF glossary](../../../../glossary.md#term-erf); [activation glossary](../../../../glossary.md#term-activation); [finite glossary](../../../../glossary.md#term-finite); [NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Optional generated-class persistence remains disabled by default. Native/vendor routes, tuning,
  Engine composition, shared backend conformance, and broader family coverage remain deferred.<span data-glossary-navigation="true"> ([persistence glossary](../../../../glossary.md#term-persistence); [Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) evidence before promotion to [`Ready`](../../../../glossary.md#term-ready):

- Read the architecture contract and focused backend/module/dependency/prepare/runtime documents,
  ADRs 0002/0008/0010/0011, documentation rules and General/Planning profiles, planning guide,
  roadmap, CPU master plan, completed CPU task records through 0005G, relevant Model tasks and
  live unary/classification/Tensor/DataType/ScalarValue contracts, CPU production/tests/Javadocs,
  public API/glossary text, and Java 26 Gradle configuration.<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract); [backend glossary](../../../../glossary.md#backend); [dependency glossary](../../../../glossary.md#term-dependency); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [Model glossary](../../../../glossary.md#term-model); [unary glossary](../../../../glossary.md#term-unary); [Tensor glossary](../../../../glossary.md#tensor); [DataType glossary](../../../../glossary.md#data-type--datatype); [ScalarValue glossary](../../../../glossary.md#typed-scalar-value--scalarvalue); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Audited the live Model inventory: nineteen unary kinds and three floating classifications.
  Audited the live CPU inventory: thirty-one opcodes; `NEG` for [FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64); [`GELU`](../../../../glossary.md#gaussian-error-linear-unit--gelu) for
  FLOAT64; all three classifications for FLOAT32/FLOAT64; same-type CAST only.
- Audited OpenJDK 26.0.1's installed `VectorOperators` surface and source documentation. It exposes
  the named floating lane operators and states equivalent Java accuracy/monotonicity, while making
  no hardware-intrinsic promise. It exposes no selected FLOOR, CEIL, or SIGN lane operator.<span data-glossary-navigation="true"> ([lane glossary](../../../../glossary.md#term-lane))</span>
- Re-audited Model CAST contracts and found no conversion/rounding/saturation/BOOL policy; the
  fail-closed decision remains required.<span data-glossary-navigation="true"> ([rounding glossary](../../../../glossary.md#term-rounding); [saturation glossary](../../../../glossary.md#term-saturation); [BOOL glossary](../../../../glossary.md#data-type--datatype); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- The exact expected implementation map is 30 paths with no new production class. A Draft 0005H1
  would be a mechanical split rather than a coherent capability, so it was not created.
- `ruby /tmp/validate_0005h_markdown.rb` passed for this task, the CPU master plan, and the
  roadmap, checking repository-local links and anchors, balanced fences, final newlines, and
  trailing whitespace.
- `git diff --check` passed for tracked edits. `git diff --no-index --check /dev/null <task-file>`
  produced no whitespace diagnostic; its status was the expected `1` because the task is new.
- Final status/[scope](../../../../glossary.md#term-scope) inspection found exactly the three permitted planning paths, [CPU](../../../../glossary.md#term-cpu) 0005H
  `Ready` in the task/master/roadmap, CPU 0006–0017 still Draft without detailed specifications,
  no 0005H1 specification, and no Java, test, Gradle, architecture, ADR, conformance, integration,
  API, guide, or glossary change in this planning diff.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [conformance glossary](../../../../glossary.md#term-conformance))</span>

Implementation evidence:

- Clean implementation context `/root/cpu_0005h_impl` changed only the authorized CPU Java/test
  and planning paths. `./gradlew :backends:cpu:compileJava` and
  `./gradlew :backends:cpu:compileTestJava` passed.
- After the post-completion FLOAT32 RSQRT operation-order correction, the required focused
  nine-class command passed 43 tests with zero failures, errors, or skips. Exactly one new final<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
  `./gradlew :backends:cpu:test` passed 25 suites and 108 tests with zero failures or errors
  and one expected opt-in `CpuGeneratedKernelPersistenceEvidenceTest` skip. The persistence timing
  evidence remained disabled. This evidence supersedes the earlier 42-test/107-test results;<span data-glossary-navigation="true"> ([persistence glossary](../../../../glossary.md#term-persistence))</span>
  `git diff --check` passed after implementation restabilization.
- Clean documentation context `/root/cpu_0005h_docs` inspected the final source/tests and the 25
  XML suite artifacts under <span data-glossary-navigation="true"> ([artifacts glossary](../../../../glossary.md#term-artifact))</span>`backends/cpu/build/test-results/test`. The artifacts independently
  confirm 108 tests, zero failures/errors, and the one expected skip; the nine named focused
  classes contain 43 recorded test cases. The reopened documentation pass inspected the corrected
  emitter and raw-bit regression, reused the new stabilized test evidence, and did not repeat Java
  test suites; the required Javadoc task retained its normal Gradle compilation dependency.<span data-glossary-navigation="true"> ([regression glossary](../../../../glossary.md#term-regression); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names))</span>

Documentation-focused evidence:

- Context `/root/cpu_0005h_docs` applied the General, API/Javadoc, Backend Guide, and Planning
  profiles. It independently reviewed the architecture contract, documentation and planning
  rules, CPU plan/task history and final diff, all affected production/test contracts, the CPU
  guide, glossary, public Model unary/classification/CAST boundaries, and generated test artifacts.
- Reviewed the affected provider, schema, and opcode Javadocs and retained their accurate
  implementation drafts. Finalized the generator/emitter, IR, lowering, reference, and package
  contracts. Package-private vector helpers now document their inputs/results, the
  orphaned reciprocal comment was removed from the polynomial helper, and the existing implicit
  public generator/lowering constructors were declared explicitly only to document their unchanged
  stateless construction contracts. Those documentation-only edits changed no executable semantics.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [opcode glossary](../../../../glossary.md#term-opcode); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [lowering glossary](../../../../glossary.md#lowering); [vector glossary](../../../../glossary.md#term-vector))</span>
- Reopened the pass after the executable correction and documented that FLOAT32 RSQRT widens its
  represented input, performs both `StrictMath.sqrt` and the reciprocal in FLOAT64, and narrows
  only the final result. The raw-bit regression at input `0x000026f6` distinguishes the obsolete
  intermediate-root result `0x616801ad` from the required final result `0x616801ae`.
- Finalized `docs/backend-guide/cpu-backend.md` and `docs/glossary.md` for all nineteen unary kinds,
  three separate classifications, the exact FLOAT32/FLOAT64 scalar and FLOAT64 vector matrices,
  special values and tolerances, schema 8, exactly 48 opcodes, scalar fallback, and fail-closed
  cross-type CAST/BFLOAT16. Finalized this task, the CPU master plan, and the roadmap at <span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [matrices glossary](../../../../glossary.md#term-matrix-meanings); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16))</span>`Complete`.
- `./gradlew :backends:cpu:javadoc` passed after final Javadocs. Its only two warnings report the
  expected incubating `jdk.incubator.vector` [module](../../../../glossary.md#term-module); there are no missing-[Javadoc](../../../../glossary.md#term-javadoc) warnings.
- `ruby /tmp/validate_0005h_markdown.rb` passed for the CPU guide, glossary, this task, master plan,
  and roadmap, checking repository-local links and heading anchors, balanced fences, final
  newlines, and trailing whitespace. `git diff --check` passed on the final combined change.
- Final semantic/scope checks confirmed nineteen Model unary kinds, nineteen unary opcodes, three
  classification kinds/opcodes, exactly 48 total opcodes, schema 8, the documented scalar/vector
  eligibility and exceptional-value/tolerance matrices, unchanged one-through-eight fusion/five-
  access-regime/one-copy/four-candidate/one-artifact/zero-fixed-shape/zero-unroll budgets, and
  fail-closed cross-type CAST/BFLOAT16.<span data-glossary-navigation="true"> ([tolerance glossary](../../../../glossary.md#term-tolerance); [fusion glossary](../../../../glossary.md#term-fusion); [shape glossary](../../../../glossary.md#shape))</span>
- Final path/status checks found exactly 26 changed paths inside the exact 30-path map: 15 CPU
  production/package paths, 6 CPU test paths, 2 explanatory documents, and 3 planning/status
  paths. No Java/test path outside `backends/cpu`, [Gradle](../../../../glossary.md#term-gradle)/build, `ARCHITECTURE.md`, architecture or
  ADR/test, backend-conformance, integration, public/shared API, or later-task specification changed.
  CPU 0005A–0005H are `Complete`; CPU 0006–0017 remain `Draft` without detailed specifications;
  no 0005H1 or later detailed CPU task exists.

## Implementation notes

- Closed all nineteen Model unary kinds through the existing provider, opcode/IR, lowering,
  generated scalar/vector, reference, preparation, and schema pipeline without adding a type,
  package, route, public API, shared-module change, or operation decomposition.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [opcode glossary](../../../../glossary.md#term-opcode); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [lowering glossary](../../../../glossary.md#lowering); [scalar glossary](../../../../glossary.md#term-scalar); [vector glossary](../../../../glossary.md#term-vector); [schema glossary](../../../../glossary.md#term-schema); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [module glossary](../../../../glossary.md#term-module); [operation glossary](../../../../glossary.md#operation); [decomposition glossary](../../../../glossary.md#term-decomposition))</span>
- Added seventeen distinct [unary](../../../../glossary.md#term-unary) opcodes while retaining `NEG`, `GELU_EXACT`, and the three
  classification opcodes. [`GELU`](../../../../glossary.md#gaussian-error-linear-unit--gelu) now covers FLOAT32 and maps negative infinity to negative zero in
  scalar/reference and vector realization; schema 8 invalidates every older envelope.<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [negative infinity glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [negative zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Stable sigmoid, fixed tanh-approximation GELU, and SiLU reuse pure primitive reference helpers
  from generated scalar bytecode. FLOAT32 widening/narrowing and FLOAT64 vector eligibility follow
  the fixed task matrix; FLOAT32 RSQRT performs its square root and reciprocal before the single
  final narrowing, and vector-ineligible admitted chains select scalar compute.<span data-glossary-navigation="true"> ([sigmoid glossary](../../../../glossary.md#term-sigmoid); [tanh glossary](../../../../glossary.md#term-tanh); [SiLU glossary](../../../../glossary.md#sigmoid-linear-unit--silu); [bytecode glossary](../../../../glossary.md#term-bytecode); [widening glossary](../../../../glossary.md#term-widening); [narrowing glossary](../../../../glossary.md#term-narrowing); [FLOAT64 glossary](../../../../glossary.md#term-float64); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [square root glossary](../../../../glossary.md#norms-elementary-functions-and-probability-distributions))</span>
- Preserved the complete pre-0005H opcode behavior, one-through-eight topology, virtual
  intermediates, access/carrier/materialization/artifact budgets, same-type CAST, and fail-closed
  cross-type CAST/BFLOAT16.<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [carrier glossary](../../../../glossary.md#term-storage-carrier); [materialization glossary](../../../../glossary.md#materialization); [artifact glossary](../../../../glossary.md#term-artifact); [fail-closed glossary](../../../../glossary.md#term-fail-closed); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16))</span>

## Completion summary

- Completed changes: closed the full FLOAT32/FLOAT64 unary family in the existing CPU portable
  pipeline, advanced generated compatibility to schema 8, corrected GELU negative infinity and
  FLOAT32 RSQRT operation order, and synchronized Javadocs, package summaries, guide, glossary,
  task, master plan, and roadmap.<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [unary glossary](../../../../glossary.md#term-unary); [schema glossary](../../../../glossary.md#term-schema); [GELU glossary](../../../../glossary.md#gaussian-error-linear-unit--gelu); [negative infinity glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [operation glossary](../../../../glossary.md#operation))</span>
- Files changed or created by the documentation pass: `CpuClassFileKernelGenerator.java`,
  `CpuScalarEmitter.java`, `CpuVectorEmitter.java`,
  `CpuKernelIr.java`, `CpuPartitionLowering.java`, `CpuScalarReferenceKernel.java`, the six affected
  CPU `package-info.java` files, `docs/backend-guide/cpu-backend.md`, `docs/glossary.md`, this task,
  `docs/planning/backends/cpu/master-plan.md`, and `docs/planning/roadmap.md`.
- Correction-specific documentation changes: the `CpuScalarEmitter.java` class [Javadoc](../../../../glossary.md#term-javadoc), emit
  `package-info.java`, CPU backend guide, this task, CPU master plan, and roadmap. The glossary and
  all other affected Javadocs/package summaries were reviewed and remain accurate without another
  correction-specific edit.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- Combined changed paths: the fifteen production/package paths
  `CpuCapabilityProvider.java`, `CpuGeneratorSchema.java`, `CpuClassFileKernelGenerator.java`,
  `CpuScalarEmitter.java`, `CpuVectorEmitter.java`, `CpuKernelIr.java`, `CpuPointwiseOpcode.java`,
  `CpuPartitionLowering.java`, `CpuScalarReferenceKernel.java`, and root/emit/IR/lowering/reference/
  prepare <span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [lowering glossary](../../../../glossary.md#lowering); [prepare glossary](../../../../glossary.md#prepare))</span>`package-info.java`; six tests `CpuCapabilityProviderTest.java`,
  `CpuGeneratedKernelArtifactStoreTest.java`, `CpuPointwiseGeneratedKernelTest.java`,
  `CpuPointwiseOpcodeTest.java`, `CpuPointwisePartitionLoweringTest.java`, and
  `CpuReferenceDifferentialTest.java`; the CPU guide and glossary; and this task, CPU master plan,
  and roadmap. Total: 26 authorized paths.
- Tests and [validation](../../../../glossary.md#term-validation): reused implementation context `/root/cpu_0005h_impl` compile/focused
  43-test/final 108-test evidence after inspecting the final XML artifacts; documentation context<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [artifacts glossary](../../../../glossary.md#term-artifact))</span>
  `/root/cpu_0005h_docs` passed CPU Javadoc, five-file Markdown/link/anchor/formatting validation,
  exact semantic/scope/status checks, and <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check`.
- Documentation-agent review: `/root/cpu_0005h_docs`; reopened after the executable correction,
  complete, and no Java test suite was repeated by the documentation pass.
- Documentation impact: [CPU](../../../../glossary.md#term-cpu) guide, glossary, task, master plan, and roadmap finalized.
- Javadoc review: affected production and package contracts finalized; public Model, Compiler,
  Config, Planning, Prepare, Runtime, Backend/Public API Javadocs remain accurate because this task
  changes only CPU-internal realization and truthful CPU capability coverage.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning); [Runtime glossary](../../../../glossary.md#term-runtime); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Glossary impact: current CPU portable route, specialization, artifact, IR, execution strategy,
  and preparation-boundary descriptions reflect the unary closure and schema 8. No additional
  correction-specific glossary text is needed because the glossary defines Model semantic identity
  and CPU vector eligibility, while the backend guide owns the private FLOAT32 rounding sequence.<span data-glossary-navigation="true"> ([CPU portable route glossary](../../../../glossary.md#cpu-portable-route); [specialization glossary](../../../../glossary.md#term-specialization); [identity glossary](../../../../glossary.md#term-identity-meanings); [vector glossary](../../../../glossary.md#term-vector); [rounding glossary](../../../../glossary.md#term-rounding))</span>
- No-change conclusions: architecture/ADRs/tests and dependency rules remain unchanged because
  ownership and module boundaries did not move; Gradle/build remains unchanged because Java 26 and
  Vector configuration already existed; backend-conformance and integration remain deferred to
  CPU 0009 because no composed Engine/backend harness exists; all unused authorized package,
  prepare, IR, fused-kernel, and test paths remain accurate because existing typed facts and
  regression suites already cover their unchanged contracts. Model/Compiler/Config/Planning/
  shared Prepare/Runtime/public APIs need no change because Model already owns the semantic kinds
  and CPU 0005H only realizes them behind existing backend boundaries.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [module glossary](../../../../glossary.md#term-module); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine); [fused glossary](../../../../glossary.md#specialization-and-fusion); [kernel glossary](../../../../glossary.md#kernel); [regression glossary](../../../../glossary.md#term-regression))</span>
- Unresolved issues: None.
- Follow-up required: None. CPU 0006 remains the next Draft row and has no detailed specification.

Status: Complete
