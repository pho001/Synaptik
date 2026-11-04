# Task 0018N: [Typed Scalar Value](../../../../glossary.md#typed-scalar-value--scalarvalue) Contract

**Editorial reading key (current context, not a status update).**

This record covers the six-type scalar migration, not execution; its exclusions and follow-up
statuses are historical. The current [exact-scalar
reference](../../../../reference/tensor/data-types.md#exact-typed-scalar-values) explains the
expanded format surface and raw-bit versus conversion factories. `ScalarValue` means an immutable
type-and-bits parameter, not a Tensor or executable constant. For example, FLOAT32 multiplication
requires `ScalarValue.float32(0.5f)`; the `double` convenience supplies FLOAT64, not an inferred
conversion. Raw factories preserve signed zeros and NaN payloads; named numeric conversions have
separate rounding and NaN rules.

The caller supplies that value to a receiver-aware Tensor helper, which checks exact type agreement
and retains it in operation attributes. `OperationSignature` checks attribute class and cardinality,
not operand compatibility; compiler occurrence validation must enforce the latter for arbitrary
operations. These are semantic responsibilities, not storage ownership: derived expressions describe
work without executing or owning result buffers. Model semantics defines meaning; providers realize
supported routes rather than redefine it. The current [data-type
contract](../../../../reference/tensor/data-types.md#data-types) distinguishes logical formats from
backend support.

## Status

Complete

## Goal

Replace the [model](../../../../glossary.md#term-model)'s raw [binary64](../../../../glossary.md#floating-point-precision-and-rounding) [scalar](../../../../glossary.md#term-scalar)-[operation](../../../../glossary.md#operation) and [padding](../../../../glossary.md#padding) [parameters](../../../../glossary.md#term-parameter) with one minimal,
[immutable](../../../../glossary.md#term-immutable), [backend](../../../../glossary.md#backend)-independent value that preserves the exact representation of any current
[`DataType`](../../../../glossary.md#data-type--datatype):

```text
DataType + exact primitive bits = ScalarValue
ScalarValue + matching Tensor DataType = valid scalar expression parameter
```

The migration must preserve [FLOAT64](../../../../glossary.md#term-float64) and [FLOAT32](../../../../glossary.md#term-float32) [raw bits](../../../../glossary.md#storage-mapping-contiguity-and-injectivity), raw [BFLOAT16](../../../../glossary.md#term-bfloat16) bits, every signed [INT32](../../../../glossary.md#data-type--datatype)
and [INT64](../../../../glossary.md#data-type--datatype) value, and [canonical](../../../../glossary.md#term-canonical-form) [BOOL](../../../../glossary.md#data-type--datatype). It must migrate `ScalarValueAttrs`, `ClampRangeAttrs`, and
`PadAttrs` atomically so no supported operation family retains a second raw-`double` scalar
representation.

This task is model representation and public expression construction only. It does not convert
or execute values, add a [registry](../../../../glossary.md#term-registry) or scalar hierarchy, define [gradients](../../../../glossary.md#term-gradient), or add [compiler](../../../../glossary.md#term-compiler),
[prepare](../../../../glossary.md#prepare), [runtime](../../../../glossary.md#term-runtime), or backend behavior.

## [Mental model](../../../../glossary.md#term-mental-model) and examples

[`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) records the type and exact bits before an [operation](../../../../glossary.md#operation) reaches a [compiler](../../../../glossary.md#term-compiler) or [backend](../../../../glossary.md#backend):

```text
ScalarValue.float64(-0.0d)               -> FLOAT64 / 0x8000000000000000
ScalarValue.float32(-0.0f)               -> FLOAT32 / 0x80000000
ScalarValue.bfloat16Bits((short) 0xFFC1) -> BFLOAT16 / 0xFFC1
ScalarValue.int64(9_007_199_254_740_993L) -> INT64 / exact value above 2^53
ScalarValue.bool(true)                    -> BOOL / canonical true
```

The value is not a [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [`Tensor`](../../../../glossary.md#tensor), boxed `Number`, conversion request, [storage carrier](../../../../glossary.md#term-storage-carrier), or
executable [constant](../../../../glossary.md#term-constant). A [scalar](../../../../glossary.md#term-scalar) Tensor contains one logical tensor [element](../../../../glossary.md#term-element) and storage; a
`ScalarValue` is a small [immutable](../../../../glossary.md#term-immutable) semantic [parameter](../../../../glossary.md#term-parameter) retained by operation attributes.

Planned public usage is explicit and type matched:

```java
Tensor float32Result = float32Input.mul(ScalarValue.float32(0.5f));
Tensor int64Padded = int64Input.pad(before, after,
        ScalarValue.int64(9_007_199_254_740_993L));
Tensor boolPadded = boolInput.pad(before, after, ScalarValue.bool(false));
```

The existing `double` overloads remain source-compatible exact-[FLOAT64](../../../../glossary.md#term-float64) conveniences:

```java
float64Input.mul(0.5d);
float64Input.pad(before, after, -1.0d);
```

They do not infer, narrow, or convert to [FLOAT32](../../../../glossary.md#term-float32), [BFLOAT16](../../../../glossary.md#term-bfloat16), [INT32](../../../../glossary.md#data-type--datatype), [INT64](../../../../glossary.md#data-type--datatype), or [BOOL](../../../../glossary.md#data-type--datatype). A call such as
`float32Input.mul(0.5d)` therefore fails the same exact data-type match required by the typed
overload; callers use `ScalarValue.float32(0.5f)` instead.

## Current problems

- `ScalarValueAttrs(double)` cannot say whether the parameter is FLOAT64, FLOAT32, or BFLOAT16
  and cannot represent exact INT64 values above binary64's consecutive-integer range.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [INT64 glossary](../../../../glossary.md#data-type--datatype); [binary64 glossary](../../../../glossary.md#floating-point-precision-and-rounding))</span>
- `ClampRangeAttrs(double, double)` validates only binary64 bounds and cannot retain bounds in the
  input's exact floating representation.
- `PadAttrs(..., double)` permits every tensor data type but leaves BOOL and integral padding
  constants ambiguous and unsafe.<span data-glossary-navigation="true"> ([tensor glossary](../../../../glossary.md#tensor); [data type glossary](../../../../glossary.md#data-type--datatype); [BOOL glossary](../../../../glossary.md#data-type--datatype); [padding glossary](../../../../glossary.md#padding); [constants glossary](../../../../glossary.md#term-constant))</span>
- Current FLOAT32 and BFLOAT16 scalar expressions retain an unconverted binary64 parameter, so
  semantic state does not identify the value belonging to the input data type.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- Adding only a new wrapper would leave raw `double` and typed scalar operation attributes alive
  simultaneously. The migration must therefore be one cohesive task.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>

## [Scope](../../../../glossary.md#term-scope)

- Add one public final [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) class in `model.datatype` for exactly the six current data
  types.
- Store one exact [`DataType`](../../../../glossary.md#data-type--datatype) plus one primitive `long` bit payload; add no boxed payload, subtype,
  collection, map, registry, or service.<span data-glossary-navigation="true"> ([payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [registry glossary](../../../../glossary.md#term-registry))</span>
- Provide named factories for exact FLOAT64, FLOAT32, raw BFLOAT16, INT32, INT64, and BOOL values,
  plus one explicitly named binary32-to-BFLOAT16 conversion factory using <span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [BOOL glossary](../../../../glossary.md#data-type--datatype); [binary32 glossary](../../../../glossary.md#floating-point-precision-and-rounding))</span>`BFloat16Bits`.
- Provide one `dataType()` query and six strict type-specific inspection methods.
- Define equality and hashing by exact [data type](../../../../glossary.md#data-type--datatype) and exact stored bits.
- Migrate `ScalarValueAttrs`, `ClampRangeAttrs`, and `PadAttrs` record components from `double` to
  non-null `ScalarValue`.
- Preserve existing [operation](../../../../glossary.md#operation)-[signature](../../../../glossary.md#term-signature) attribute classes and one-input/one-output cardinalities.
- Add typed `ScalarValue` overloads for the five current public scalar/clamp Tensor methods and
  constant padding.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [Tensor glossary](../../../../glossary.md#tensor); [constant glossary](../../../../glossary.md#term-constant); [padding glossary](../../../../glossary.md#padding))</span>
- Retain all six existing `double` overloads as exact-FLOAT64 convenience methods delegating to
  the typed overloads.
- Require exact `ScalarValue.dataType()` equality with the receiver Tensor data type at public
  expression construction boundaries.
- Keep scalar arithmetic/clamp limited to floating input Tensors; keep pad available for all six
  data types.
- Preserve current result Shape, data type, gradient eligibility, unresolved layout, freshness,
  producer/provenance, storage absence, validation side effects, and identity-allocation rules.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [gradient glossary](../../../../glossary.md#term-gradient); [layout glossary](../../../../glossary.md#layout); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance); [validation glossary](../../../../glossary.md#term-validation); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Add focused tests for representation, attributes, expression compatibility, operation
  signatures, failure order, and unchanged metadata.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata))</span>
- Finalize Javadocs, Tensor API, Compile API, glossary, capability status, task evidence, master
  plan, and roadmap through the mandatory independent documentation pass.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- FLOAT16, unsigned, quantized, sparse, complex, string, arbitrary user-defined, or future data
  types<span data-glossary-navigation="true"> ([FLOAT16 glossary](../../../../glossary.md#term-float16))</span>
- boxed `Number`, `Object`, generic payloads, a sealed subtype per data type, visitor, parser,
  serializer, string dispatch, registry, service locator, or reflective discovery<span data-glossary-navigation="true"> ([sealed glossary](../../../../glossary.md#term-sealed); [data type glossary](../../../../glossary.md#data-type--datatype); [dispatch glossary](../../../../glossary.md#term-dispatch); [registry glossary](../../../../glossary.md#term-registry); [service locator glossary](../../../../glossary.md#term-service-locator))</span>
- an implicit or general cross-data-type conversion [API](../../../../glossary.md#term-application-programming-interface-api)
- FLOAT64-to-FLOAT32 narrowing, FLOAT32-to-integer conversion, integer-to-floating conversion,
  truthiness conversion, saturation, truncation, or rounding policy<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [narrowing glossary](../../../../glossary.md#term-narrowing); [saturation glossary](../../../../glossary.md#term-saturation); [truncation glossary](../../../../glossary.md#term-truncation-toward-zero); [rounding glossary](../../../../glossary.md#term-rounding))</span>
- changing `BFloat16Bits.fromFloat` [round-to-nearest](../../../../glossary.md#floating-point-precision-and-rounding)/[ties-to-even](../../../../glossary.md#term-ties-to-even) and [canonical](../../../../glossary.md#term-canonical-form)-[NaN](../../../../glossary.md#term-nan) behavior
- reading typed [scalar](../../../../glossary.md#term-scalar) values from a [Tensor](../../../../glossary.md#tensor) or materializing a [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) as storage
- adding `ScalarValue` overloads to `TensorFactory.scalar` or `TensorFactory.full`; task 0018S
  owns factory-surface cleanup after this foundation
- removing or changing existing primitive `TensorFactory.scalar`, `scalarBFloat16`, `full`, or
  `fullBFloat16` [signatures](../../../../glossary.md#term-signature) or [eager](../../../../glossary.md#term-eager-evaluation)-storage behavior
- extending current scalar arithmetic or clamp expression eligibility to INT32, INT64, or BOOL;
  tasks 0018T and 0018U own selected numeric-domain expansion<span data-glossary-navigation="true"> ([INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- changing scalar [operation](../../../../glossary.md#operation) vocabulary, removing fast [unary](../../../../glossary.md#term-unary) kinds, or renaming `inv`; task 0018P
  owns that cleanup
- numerical execution, NaN propagation, signed-zero result selection, overflow, underflow,
  backend scalar conversion, kernels, or conformance behavior<span data-glossary-navigation="true"> ([signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [overflow glossary](../../../../glossary.md#term-overflow); [underflow glossary](../../../../glossary.md#term-underflow); [backend glossary](../../../../glossary.md#backend); [kernels glossary](../../../../glossary.md#kernel); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- canonicalization, constant folding, graph capture, gradients, autograd, compiler, planning,
  prepare, runtime, backend, engine, trace, or ONNX behavior<span data-glossary-navigation="true"> ([canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [constant folding glossary](../../../../glossary.md#term-constant-folding); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [gradients glossary](../../../../glossary.md#term-gradient); [autograd glossary](../../../../glossary.md#autograd); [compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [engine glossary](../../../../glossary.md#term-engine); [trace glossary](../../../../glossary.md#trace); [ONNX glossary](../../../../glossary.md#term-onnx))</span>
- dependencies, Gradle, architecture, architecture tests, another module, or a task-0018O
  specification<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>, especially “Typed [scalar](../../../../glossary.md#term-scalar) values”
- [Model master plan](../master-plan.md)
- [Task 0001](0001-data-type-model.md)
- [Task 0014E](0014e-scalar-arithmetic-and-clamp-semantics.md)
- [Task 0014F](0014f-scalar-arithmetic-and-clamp-tensor-expressions.md)
- [Task 0017I](0017i-pad-and-tile-semantics.md)
- [Task 0017J](0017j-pad-and-tile-tensor-expressions.md)
- [Task 0018K](0018k-operation-signature-and-construction-hardening.md)
- [Task 0018M1](0018m1-dynamic-extent-adoption.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- Typed scalar values, operation attributes, public Tensor expression metadata, and exact
  data-type validation remain backend-independent <span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [metadata glossary](../../../../glossary.md#term-metadata); [validation glossary](../../../../glossary.md#term-validation); [backend glossary](../../../../glossary.md#backend))</span>`modules/model` responsibilities.
- [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) is a value-format contract. It contains no Tensor, Shape, layout, storage,
  producer, graph, compiler, backend, device, runtime, or execution state.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [producer glossary](../../../../glossary.md#term-producer); [graph glossary](../../../../glossary.md#graph); [compiler glossary](../../../../glossary.md#term-compiler); [device glossary](../../../../glossary.md#term-device); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Package direction remains acyclic: operation attributes and Tensor construction may depend on
  the foundational datatype package; datatype must not depend on operation or Tensor.
- `OperationSignature` continues to validate the exact attributes implementation class and local
  occurrence cardinality. It must not import <span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [cardinality glossary](../../../../glossary.md#term-cardinality))</span>`ScalarValue`, inspect its data type, or become an
  operand-aware schema.<span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype); [schema glossary](../../../../glossary.md#term-schema))</span>
- [`Operation`](../../../../glossary.md#operation) has no input descriptor and therefore cannot validate scalar/input data-type
  equality. The public Tensor construction boundary owns that local check. Later compiler graph
  validation must enforce the same invariant for operations not created by public Tensor methods.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [invariant glossary](../../../../glossary.md#term-invariant))</span>
- No global registry, factory service, scalar subtype hierarchy, classpath scan, or boxing-heavy
  abstraction is permitted.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [classpath glossary](../../../../glossary.md#term-classpath); [scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- Public [`Tensor`](../../../../glossary.md#tensor) remains mutable API state rather than graph IR. Typed overloads create the same
  storage-free derived metadata as the existing methods.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- If implementation requires a new module dependency, architecture rule, operation-signature
  concept, or conversion policy beyond this specification, stop and report the conflict.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [signature glossary](../../../../glossary.md#term-signature))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.datatype` — owns data-type metadata, BFLOAT16 bit conversion,
  and now the exact typed scalar value.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [typed scalar value glossary](../../../../glossary.md#typed-scalar-value--scalarvalue))</span>
- `io.github.pho001.synaptik.model.operation.elementwise.scalar` — owns typed scalar/clamp
  operation attributes and semantic kinds.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- `io.github.pho001.synaptik.model.operation.layout` — owns typed constant-padding attributes and
  kind.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [padding glossary](../../../../glossary.md#padding))</span>
- `io.github.pho001.synaptik.model.tensor` — owns public expression overloads, exact input/value
  type matching, descriptors, and producer/provenance construction.<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>

Packages added or changed:

- No package is added or moved.
- The [model](../../../../glossary.md#term-model) master-plan package map is clarified so `model.datatype`, not the root operation
  package, owns [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue); [operation](../../../../glossary.md#operation) attributes consume that foundational value.

Type placement:

- `io.github.pho001.synaptik.model.datatype.ScalarValue` — reusable exact scalar representation
  colocated with [`DataType`](../../../../glossary.md#data-type--datatype) and `BFloat16Bits`, independent of any operation family.
- `ScalarValueTest` — same-package focused representation and public-surface test.
- Existing attribute records retain their current packages because they assign operation-specific
  roles to the foundational value.
- Existing Tensor helpers retain their current package because they own receiver-aware
  compatibility and derived construction.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>

## Required contracts

### [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) representation and [API](../../../../glossary.md#term-application-programming-interface-api)

Create one `public final` non-record class with exactly two private final instance fields:

```java
private final DataType dataType;
private final long bits;
```

The private constructor accepts only [canonical](../../../../glossary.md#term-canonical-form) internal representations produced by the named
factories. Add exactly these public construction methods:

```java
public static ScalarValue float64(double value)
public static ScalarValue float32(float value)
public static ScalarValue bfloat16(float value)
public static ScalarValue bfloat16Bits(short bits)
public static ScalarValue int32(int value)
public static ScalarValue int64(long value)
public static ScalarValue bool(boolean value)
```

Exact encodings are:

| Factory | [Data type](../../../../glossary.md#data-type--datatype) | Stored `bits` |
|---|---|---|
| `float64(value)` | [`FLOAT64`](../../../../glossary.md#term-float64) | `Double.doubleToRawLongBits(value)` |
| `float32(value)` | [`FLOAT32`](../../../../glossary.md#term-float32) | unsigned low 32 bits of `Float.floatToRawIntBits(value)` |
| `bfloat16(value)` | [`BFLOAT16`](../../../../glossary.md#term-bfloat16) | unsigned low 16 bits of `BFloat16Bits.fromFloat(value)` |
| `bfloat16Bits(bits)` | `BFLOAT16` | unsigned low 16 bits of the supplied raw pattern |
| `int32(value)` | [`INT32`](../../../../glossary.md#data-type--datatype) | unsigned low 32 bits of the exact [two's-complement](../../../../glossary.md#arithmetic-rearrangement-and-finite-difference-checks) `int` |
| `int64(value)` | [`INT64`](../../../../glossary.md#data-type--datatype) | exact two's-complement `long` bits |
| `bool(value)` | [`BOOL`](../../../../glossary.md#data-type--datatype) | exactly `0` for false or `1` for true |

`bfloat16(float)` is the only conversion factory. Its name, [parameter](../../../../glossary.md#term-parameter) type, and [Javadoc](../../../../glossary.md#term-javadoc) make the
[binary32](../../../../glossary.md#floating-point-precision-and-rounding)-to-BFLOAT16 conversion explicit, and it delegates to the existing conversion utility.
`bfloat16Bits(short)` performs no conversion or [NaN](../../../../glossary.md#term-nan) [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form) and retains all 16 bits.

Add exactly these public inspectors:

```java
public DataType dataType()
public double float64Value()
public float float32Value()
public short bfloat16Bits()
public int int32Value()
public long int64Value()
public boolean booleanValue()
```

Every typed inspector except `dataType()` succeeds only for its exact type. A mismatch throws
`IllegalStateException` with exact message:

```text
scalar value has data type <actual>, not <expected>
```

The class adds no general `value()`, `rawBits()`, `Number`, boxed, conversion, visitor, or [mutable](../../../../glossary.md#term-mutable)
API. Package-private implementation helpers are allowed only to avoid duplicated type checks and
must not widen the public surface.

Override `equals`, [`hashCode`](../../../../glossary.md#hash-codes-and-content-digests), and `toString`:

- equality is exact [`DataType`](../../../../glossary.md#data-type--datatype) [identity](../../../../glossary.md#term-identity-meanings) plus exact canonical stored `bits`;
- [hash code](../../../../glossary.md#term-hash-code) uses the same two facts;
- signed zeros are unequal for FLOAT32 and FLOAT64;
- distinct FLOAT32/FLOAT64 NaN signs or payloads are unequal and retained;
- distinct raw BFLOAT16 NaN patterns are unequal and retained;
- identical numeric values of different data types are unequal; and
- diagnostic text is exactly
  `ScalarValue[dataType=<type>, bits=0x<uppercase fixed-width hexadecimal>]`, using 16 digits for
  FLOAT64/INT64, 8 for FLOAT32/INT32, 4 for BFLOAT16, and 2 for BOOL; it is diagnostic rather
  than serialization.<span data-glossary-navigation="true"> ([serialization glossary](../../../../glossary.md#term-serialization))</span>

No factory or method returns `null`.

### Conversion policy and factory interaction

There is no implicit [scalar](../../../../glossary.md#term-scalar) conversion in [model](../../../../glossary.md#term-model) expression construction and no general
`ScalarValue.convertTo(DataType)` in this task. Callers construct the exact required type.

The only current explicit scalar-format conversion remains [binary32](../../../../glossary.md#floating-point-precision-and-rounding) to [BFLOAT16](../../../../glossary.md#term-bfloat16) through
`BFloat16Bits.fromFloat`, exposed by `ScalarValue.bfloat16(float)`. Raw BFLOAT16 bits use
`bfloat16Bits(short)`. Any future cross-type conversion must be separately specified; it must not
be inferred from Java numeric [widening](../../../../glossary.md#term-widening) or [narrowing](../../../../glossary.md#term-narrowing).

Existing `TensorFactory` primitive scalar/full methods remain source- and behavior-compatible:

- primitive `scalar` and matching `full` overloads already select an exact current [data type](../../../../glossary.md#data-type--datatype);
- `scalarBFloat16` and `fullBFloat16` remain explicit rounded binary32-to-BFLOAT16 paths; and
- no generic typed factory overload is added here. Task 0018S may decide whether the cleaned
  factory surface should later consume [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) directly.

Focused tests cross-check `ScalarValue` encodings against these existing factory and BFLOAT16
contracts without modifying factory implementation.

### Attribute migration

Replace the current components exactly:

```java
public record ScalarValueAttrs(ScalarValue value) implements OperationAttrs

public record ClampRangeAttrs(
        ScalarValue minValue,
        ScalarValue maxValue) implements OperationAttrs

public record PadAttrs(
        List<Long> before,
        List<Long> after,
        ScalarValue constantValue) implements OperationAttrs
```

`ScalarValueAttrs` null-checks `value` with message `value` and retains the exact [immutable](../../../../glossary.md#term-immutable)
reference. It performs no kind or input compatibility [validation](../../../../glossary.md#term-validation).

`ClampRangeAttrs` validates in this order:

1. null-check `minValue`, then `maxValue`, with those exact messages;
2. require identical `dataType()` values, otherwise throw `IllegalArgumentException` with exact
   message `minValue and maxValue must have the same data type: <min> != <max>`;
3. reject [BOOL](../../../../glossary.md#data-type--datatype) with exact message `clamp bounds must be numeric, but were BOOL`; and
4. reject a strict inversion using the exact represented type's primitive `>` operation, with
   the existing message <span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>`minValue must be less than or equal to maxValue`.

[FLOAT64](../../../../glossary.md#term-float64) compares as `double`, [FLOAT32](../../../../glossary.md#term-float32) as `float`, [BFLOAT16](../../../../glossary.md#term-bfloat16) after exact `BFloat16Bits.toFloat`,
[INT32](../../../../glossary.md#data-type--datatype) as `int`, and [INT64](../../../../glossary.md#data-type--datatype) as `long`. The INT64 comparison never passes through [binary64](../../../../glossary.md#floating-point-precision-and-rounding). Equal
bounds, either [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero) ordering, ordered [infinities](../../../../glossary.md#nan-infinity-and-signed-zero), and any range with a floating [NaN](../../../../glossary.md#term-nan) endpoint
remain accepted. NaN payloads remain retained in the exact values.

`PadAttrs` keeps its existing list validation, order, messages, and immutable snapshots. It adds
a null check for `constantValue` after the `before` and `after` container null checks but before
list-size and [element](../../../../glossary.md#term-element) validation, with message `constantValue`. It retains the exact immutable
value reference and performs no [rank](../../../../glossary.md#term-rank) or [Tensor](../../../../glossary.md#tensor) data-type check.

Record-generated attribute equality and hashing now compose [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue)'s exact typed-bit
semantics. Generated diagnostic text remains non-serializing.

### [Operation](../../../../glossary.md#operation) [signatures](../../../../glossary.md#term-signature) and family compatibility

`ScalarElementwiseKind` continues to accept exactly `ScalarValueAttrs` for `MUL`, `POW`,
`CLAMP_MIN`, and `CLAMP_MAX`, and exactly `ClampRangeAttrs` for `CLAMP`. `PadKind.PAD` continues to
accept exactly `PadAttrs`. All variants remain exactly one input and one output.

Do not change `OperationSignature`, [`OperationKind`](../../../../glossary.md#operationkind), or `Operation`. Their exact attribute-class
[validation](../../../../glossary.md#term-validation) remains sufficient because the migrated record classes are unchanged. A raw-double
attribute class or alternate typed attribute class must not coexist with the migrated records.

An `Operation` has no operand [descriptor](../../../../glossary.md#term-descriptor), so direct construction establishes only correct
kind/attribute class pairing and intrinsic attribute invariants. It does not establish that the
attribute value matches an eventual input. Public [Tensor](../../../../glossary.md#tensor) helpers own local equality; [compiler](../../../../glossary.md#term-compiler)
[occurrence](../../../../glossary.md#term-occurrence) validation must repeat it when [graph](../../../../glossary.md#graph) construction later consumes arbitrary operations.

### Public [Tensor](../../../../glossary.md#tensor) surface and compatibility

Add these exact overloads:

```java
public Tensor mul(ScalarValue scalar)
public Tensor pow(ScalarValue exponent)
public Tensor clamp(ScalarValue minValue, ScalarValue maxValue)
public Tensor clampMin(ScalarValue minValue)
public Tensor clampMax(ScalarValue maxValue)
public Tensor pad(long[] before, long[] after, ScalarValue constantValue)
```

Keep the existing methods and define them only as exact-[FLOAT64](../../../../glossary.md#term-float64) adapters:

```java
public Tensor mul(double scalar) {
    return mul(ScalarValue.float64(scalar));
}

// The other four scalar/clamp double methods follow the same rule.

public Tensor pad(long[] before, long[] after, double constantValue) {
    return pad(before, after, ScalarValue.float64(constantValue));
}
```

Declarations remain source-compatible. FLOAT64 behavior remains compatible and exact. Previously
accepted non-FLOAT64 calls through a `double` overload now fail rather than retain an ambiguous
[binary64](../../../../glossary.md#floating-point-precision-and-rounding) [parameter](../../../../glossary.md#term-parameter); this is deliberate hardening. No `float`, primitive integral, [boolean](../../../../glossary.md#data-type--datatype),
raw-short, generic, static, reverse, or caller-[`DataType`](../../../../glossary.md#data-type--datatype) overload is added.

Typed overloads delegate once to the existing package-private helpers. Public methods perform no
duplicated [validation](../../../../glossary.md#term-validation), [descriptor](../../../../glossary.md#term-descriptor) construction, or [identity](../../../../glossary.md#term-identity-meanings) allocation.

### [Scalar](../../../../glossary.md#term-scalar)/clamp helper [validation](../../../../glossary.md#term-validation) and construction

Change `TensorScalarExpressions.applyScalar` to accept [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue). Preserve its three-method,
field-free [shape](../../../../glossary.md#shape). Validation order is:

1. null-check `input`, `kind`, and `value`, with those exact messages;
2. reject `CLAMP` with the existing `CLAMP requires ClampRangeAttrs` message;
3. require the input [data type](../../../../glossary.md#data-type--datatype) to be floating with the existing message;
4. require `value.dataType()` to equal the input data type, otherwise throw
   `IllegalArgumentException` with exact message
   `scalar data type <scalar> must match input data type <input>`;
5. construct one `ScalarValueAttrs`, one valid [`Operation`](../../../../glossary.md#operation), and the existing result.

Change `applyClamp` to accept two `ScalarValue` values. Validation order is:

1. null-check `input`, `minValue`, and `maxValue`, with those exact messages;
2. require the input data type to be floating with the existing message;
3. construct one `ClampRangeAttrs`, applying its null/type/numeric/order contract;
4. require the common bounds data type to equal the input type, otherwise throw
   `IllegalArgumentException` with exact message
   `clamp data type <bounds> must match input data type <input>`;
5. construct one `CLAMP` operation and the existing result.

All validation precedes `TensorFactory.createDerived`, so failures consume no [Tensor](../../../../glossary.md#tensor) ID. Valid
requests retain exact [immutable](../../../../glossary.md#term-immutable) `ScalarValue` references inside the attributes.

### [Padding](../../../../glossary.md#padding) helper [validation](../../../../glossary.md#term-validation) and construction

Change the existing helper `pad` [parameter](../../../../glossary.md#term-parameter) from `double` to [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue); retain the helper's
five-method, field-free [shape](../../../../glossary.md#shape) and all Shape arithmetic.

Validation order is:

1. null-check `input`, `before`, `after`, then `constantValue`, with parameter-name messages;
2. preserve existing before-[rank](../../../../glossary.md#term-rank) then after-rank checks;
3. clone both arrays and construct one `PadAttrs`, preserving width validation and snapshots;
4. require `constantValue.dataType()` to equal the input [descriptor](../../../../glossary.md#term-descriptor) [data type](../../../../glossary.md#data-type--datatype), otherwise throw
   `IllegalArgumentException` with exact message
   `padding constant data type <constant> must match input data type <input>`;
5. derive the same symbolic Shape, construct [`PAD`](../../../../glossary.md#padding), and create the existing result.

The data-type mismatch check follows width validation and precedes Shape arithmetic and [identity](../../../../glossary.md#term-identity-meanings)
allocation. All six exact matching data types are valid. The result remains unresolved, fresh,
unlabeled, storage-free, and one-input.

## Affected files

Expected production:

- new `modules/model/src/main/java/io/github/pho001/synaptik/model/datatype/ScalarValue.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/scalar/ScalarValueAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/scalar/ClampRangeAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/PadAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorScalarExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorPadTileExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Expected tests:

- new `modules/model/src/test/java/io/github/pho001/synaptik/model/datatype/ScalarValueTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/elementwise/scalar/ScalarElementwiseSemanticsTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/layout/PadTileSemanticsTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorScalarElementwiseTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorPadTileExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`

Expected documentation and [planning](../../../../glossary.md#term-planning):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task specification
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Expected no-change review:

- `TensorFactory`, `TensorConstants`, `BFloat16Bits`, and their focused tests
- `OperationSignature`, [`OperationKind`](../../../../glossary.md#operationkind), [`Operation`](../../../../glossary.md#operation), scalar/pad kind declarations, and production
  signature-matrix tests<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [signature glossary](../../../../glossary.md#term-signature); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- Training API, architecture documents/ADRs/tests, backend-conformance tests, integration tests,
  Gradle configuration, dependencies, and other modules<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module))</span>

## Maximum [scope](../../../../glossary.md#term-scope)

At most seven production Java files, six test files, and seven documentation/[planning](../../../../glossary.md#term-planning) files may
change: twenty paths total.

This is a documented [atomic](../../../../glossary.md#resource-transaction-and-rollback)-migration exception to the ordinary task-size preference. Splitting
the value type from attribute and [Tensor](../../../../glossary.md#tensor)-boundary migration would either leave two live [scalar](../../../../glossary.md#term-scalar)
representations or make current [operation](../../../../glossary.md#operation) families temporarily unconstructable. The task remains
inside one [module](../../../../glossary.md#term-module) and one cohesive foundation contract; most affected files receive one component
type or overload adaptation.

Do not use the allowance for factory redesign, unrelated refactoring, kind cleanup, formatting,
another test concept, another documentation file, or a later task specification. If a twenty-first
path or a different production concept is needed, stop and propose the smallest follow-up.

## [Javadoc](../../../../glossary.md#term-javadoc) and documentation requirements

- Document [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) purpose, representation, exact factories/inspectors, nullability,
  mismatch failures, immutability, equality/hash semantics, NaN payload and signed-zero policy,
  canonical BOOL, INT64 exactness, diagnostic text, and excluded conversion/execution behavior.<span data-glossary-navigation="true"> ([nullability glossary](../../../../glossary.md#term-nullability); [hash glossary](../../../../glossary.md#term-hash); [NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [canonical glossary](../../../../glossary.md#term-canonical-form); [BOOL glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype))</span>
- Update all three attribute records and accessors for exact typed values, reference ownership,
  validation order, comparison domain, and failure conditions.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [validation glossary](../../../../glossary.md#term-validation))</span>
- Update Tensor and helper Javadocs for typed overloads, retained exact-FLOAT64 conveniences,
  strict data-type matching, failure order, result metadata, and lifecycle boundaries.<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [metadata glossary](../../../../glossary.md#term-metadata); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api) must distinguish `ScalarValue` from a scalar Tensor, give examples for FLOAT32, raw
  BFLOAT16, INT64 above <span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16))</span>`2^53`, BOOL padding, signed zero, and NaN payload retention, and state
  current factory interoperability without promising a typed factory overload.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [signed zero glossary](../../../../glossary.md#term-signed-zero))</span>
- Compile API must replace the stale claim that scalar parameters are always binary64 and state
  that compiler graph validation must preserve exact scalar/input data-type equality.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [parameters glossary](../../../../glossary.md#term-parameter); [binary64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [compiler glossary](../../../../glossary.md#term-compiler); [graph glossary](../../../../glossary.md#graph))</span>
- Update the glossary with one reusable “typed scalar value” definition and align DataType,
  OperationAttrs, padding, and Tensor-expression status wording.<span data-glossary-navigation="true"> ([typed scalar value glossary](../../../../glossary.md#typed-scalar-value--scalarvalue); [DataType glossary](../../../../glossary.md#data-type--datatype); [OperationAttrs glossary](../../../../glossary.md#operationattrs))</span>
- Update capabilities only from selected direction to completed behavior after implementation.
- A separate clean-context documentation-focused agent must inspect final source, tests, generated
  Javadoc, APIs, glossary, and diff, then finalize these seven documentation paths.
- Record reasoned no-change conclusions for factory constants/BFLOAT16 conversion, operation
  signatures/kinds, Training API, architecture/ADRs/tests, conformance/integration, Gradle,
  dependencies, and other modules.<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant); [operation glossary](../../../../glossary.md#operation); [signatures glossary](../../../../glossary.md#term-signature); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module))</span>

## Acceptance criteria

- Exactly one public final [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) class with two primitive/reference instance fields and
  the specified named public factories/inspectors exists in `model.datatype`.
- All six current [`DataType`](../../../../glossary.md#data-type--datatype) values are covered and no future type or generic [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) is added.
- FLOAT64/FLOAT32 signed zero and raw NaN sign/payload are preserved with exact-bit equality; raw
  BFLOAT16 preserves all 16 bits; INT32/INT64 are exact; BOOL is canonical false or true.<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [signed zero glossary](../../../../glossary.md#term-signed-zero); [NaN glossary](../../../../glossary.md#term-nan); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [INT32 glossary](../../../../glossary.md#data-type--datatype); [BOOL glossary](../../../../glossary.md#data-type--datatype); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- [INT64](../../../../glossary.md#data-type--datatype) values above `2^53`, including `9_007_199_254_740_993L`, round-trip exactly without a
  floating intermediate.<span data-glossary-navigation="true"> ([round-trip glossary](../../../../glossary.md#schema-serialization-and-checkpoint))</span>
- Strict inspectors reject every wrong [data type](../../../../glossary.md#data-type--datatype) with the exact failure message.
- `ScalarValueAttrs`, `ClampRangeAttrs`, and `PadAttrs` have exactly the migrated component types;
  no raw-double semantic attribute remains.
- Clamp validates nulls, same type, non-BOOL numeric domain, and strict inversion in order,
  comparing INT64 exactly and retaining floating NaN payloads.
- Scalar/pad operation signatures keep exact attribute classes and cardinalities; wrong
  cross-family pairings still fail during <span data-glossary-navigation="true"> ([Scalar glossary](../../../../glossary.md#term-scalar); [signatures glossary](../../../../glossary.md#term-signature))</span>[`Operation`](../../../../glossary.md#operation) construction.
- [Tensor](../../../../glossary.md#tensor) exposes exactly six new typed overloads while retaining six existing `double` signatures
  as exact-FLOAT64 adapters; no other overload is added.
- Scalar/clamp accepts only matching floating values. Pad accepts matching values for all six
  data types. Every mismatch has the specified message and consumes no Tensor ID.
- Existing result Shape, symbolic padding arithmetic, type, eligibility, unresolved layout,
  producer/provenance, freshness, label/storage absence, input non-mutation, and ID effects remain
  unchanged.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [padding glossary](../../../../glossary.md#padding); [layout glossary](../../../../glossary.md#layout); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>
- Existing primitive TensorFactory scalar/full methods and explicit BFLOAT16 conversion behavior
  remain unchanged and focused interoperability checks pass.
- No conversion registry, service, subtype hierarchy, boxing-heavy path, executable behavior,
  dependency, build, or architecture change appears.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Focused tests, final module tests, model Javadoc, documentation checks, and the foundation-
  contract checkpoint pass.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Task, master-plan row, roadmap row, current-frontier wording, and capability status are
  synchronized. Task 0018O remains Draft without a detailed specification.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) focused tests as needed. After executable Java stabilizes, record one final [module](../../../../glossary.md#term-module) run:

```bash
./gradlew :modules:model:test
```

Focused automated coverage must include exact [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) [API](../../../../glossary.md#term-application-programming-interface-api) [shape](../../../../glossary.md#shape); every representation and
edge case; migrated records and failure order; unchanged [operation](../../../../glossary.md#operation) [signatures](../../../../glossary.md#term-signature); exact typed and
double [Tensor](../../../../glossary.md#tensor) overloads; helper compatibility, failure precedence, and no-ID effects; unchanged
[descriptors](../../../../glossary.md#term-descriptor), Shapes, [provenance](../../../../glossary.md#provenance), input state, freshness, and [identity](../../../../glossary.md#term-identity-meanings) allocation; and factory/
[BFLOAT16](../../../../glossary.md#term-bfloat16) interoperability without a factory API change.

The separate documentation-focused pass reuses [model](../../../../glossary.md#term-model)-test evidence unless it changes executable
Java, then runs:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

It also validates changed Markdown links/anchors, fences, examples, terminology, whitespace,
final newlines, exact twenty-path [scope](../../../../glossary.md#term-scope), package placement, and synchronized statuses.

Task 0018N closes the recorded foundation-contract [checkpoint](../../../../glossary.md#term-checkpoint) after implementation and
documentation. Run once on the final combined change:

```bash
./gradlew test
```

The root suite includes the architecture-test project. Confirm that result explicitly with final
[Javadoc](../../../../glossary.md#term-javadoc)/documentation checks and deferred cross-task invariants from 0018K–0018N. This repository-
wide run is required by the model master plan for this checkpoint, not for ordinary model tasks.

## Dependencies

- Task 0001 — six data types and `BFloat16Bits` — Complete.
- Task 0014E — raw-double [scalar](../../../../glossary.md#term-scalar)/clamp semantic attributes — Complete and migrated here.
- Task 0014F — public scalar/clamp [Tensor](../../../../glossary.md#tensor) expressions — Complete and migrated here.
- Task 0017I — raw-double pad attributes — Complete and migrated here.
- Task 0017J — public pad Tensor expression — Complete and migrated here.
- Task 0018K — exact [operation](../../../../glossary.md#operation)-[signature](../../../../glossary.md#term-signature) pairing and [cardinality](../../../../glossary.md#term-cardinality) — Complete and preserved.
- Task 0018M1 — [canonical](../../../../glossary.md#term-canonical-form) symbolic pad [Shape](../../../../glossary.md#shape) derivation — Complete and preserved.

## Follow-up tasks

- 0018O remains Draft for indexing taxonomy and unstack [normalization](../../../../glossary.md#term-normalization).
- 0018P consumes typed [scalar](../../../../glossary.md#term-scalar) attributes during [elementwise](../../../../glossary.md#term-elementwise) vocabulary cleanup.
- 0018S may decide whether `TensorFactory` should expose [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) overloads.
- 0018T consumes the typed value contract for new scalar numeric conveniences.
- 0018U owns selected integral arithmetic/comparison domains and [overflow](../../../../glossary.md#term-overflow) policy.
- Later [operation](../../../../glossary.md#operation) families may reuse exact typed [constants](../../../../glossary.md#term-constant) in their focused semantics.
- Compiler and backend tasks must later validate occurrence-level data-type compatibility and
  implement exact executable behavior.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>

Do not create a detailed follow-up specification in this task.

## Architecture impact

Expected impact: None.

The architecture already assigns [DataType](../../../../glossary.md#data-type--datatype) values, [backend](../../../../glossary.md#backend)-independent [operation](../../../../glossary.md#operation) attributes, and
public [Tensor](../../../../glossary.md#tensor) expression construction to `modules/model`. This task refines those contracts
without changing [module](../../../../glossary.md#term-module) [ownership](../../../../glossary.md#term-ownership-meanings), [dependency direction](../../../../glossary.md#java-platform-and-build-tools), [lifecycle](../../../../glossary.md#lifecycle), or [runtime](../../../../glossary.md#term-runtime) visibility.

If implementation requires architecture changes, another module, a [dependency](../../../../glossary.md#term-dependency), or executable
conversion behavior, stop and report the exact conflict.

## Implementation prompt

Use this prompt in a separate agentic task/thread with clean context:

```text
You are working in the Synaptik repository.

Read AGENTS.md, ARCHITECTURE.md, docs/developer-guide/documentation-rules.md,
docs/planning/planning-guide.md, docs/planning/roadmap.md, model capabilities/master plan, tasks
0001/0014E/0014F/0017I/0017J/0018K/0018M1/0018N, and affected datatype, attribute, Tensor,
factory, and focused-test source in full.

Implement task 0018N exactly. Stay inside modules/model and explicitly allowed documentation/
planning files. Perform the atomic typed-scalar migration; do not leave a raw-double attribute
alternative, add a general conversion API, change factory constants, or implement later cleanup/
execution behavior. Stop on scope or architecture conflict. Do not commit or push.

Run the final model test after executable code stabilizes. Then hand the actual diff and evidence
to a separate clean-context documentation-focused agent in the same change. It must inspect final
source/tests and generated Javadoc, finalize allowed API/glossary/capability/planning documents,
and avoid repeating successful Java tests unless executable behavior changes or a concrete risk
is recorded.

After both passes, run the foundation-contract repository checkpoint. Update task 0018N, the model
master plan, roadmap, and capability status only when evidence is final. Leave 0018O and every
later task Draft without a detailed specification.
```

## Local decisions

- [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) lives in `model.datatype` because it is a reusable exact value-format contract
  beside [`DataType`](../../../../glossary.md#data-type--datatype) and `BFloat16Bits`; [operation](../../../../glossary.md#operation) attributes consume it without a package cycle.
- One final class stores `DataType` plus primitive `long` bits. A subtype per data type, boxed
  payload, or generic union adds allocation and dispatch without a current need.<span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Named factories make the chosen type explicit. Raw BFLOAT16 construction is distinct from
  explicit binary32 conversion.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [binary32 glossary](../../../../glossary.md#floating-point-precision-and-rounding))</span>
- Floating equality is exact by raw bits, preserving signed zero and NaN payload policy rather
  than inheriting Java record canonical-NaN equality from raw <span data-glossary-navigation="true"> ([raw bits glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [signed zero glossary](../../../../glossary.md#term-signed-zero); [NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [canonical glossary](../../../../glossary.md#term-canonical-form); [NaN glossary](../../../../glossary.md#term-nan))</span>`double` components.
- No general conversion method is added. Exact construction is public policy; BFLOAT16's existing
  named conversion is the only current exception.
- Existing double Tensor methods remain source-compatible but mean exact FLOAT64. They no longer
  attach ambiguous binary64 parameters to non-FLOAT64 inputs.<span data-glossary-navigation="true"> ([mean glossary](../../../../glossary.md#term-mean); [FLOAT64 glossary](../../../../glossary.md#term-float64); [binary64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [parameters glossary](../../../../glossary.md#term-parameter))</span>
- TensorFactory remains unchanged because primitive overloads already choose exact types and eager
  storage. A generic typed factory entry is public-surface design owned by 0018S.<span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation))</span>
- Data-type matching belongs to receiver-aware [Tensor](../../../../glossary.md#tensor) helpers, not `OperationSignature`, because
  generic operations contain no operand descriptors.<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor))</span>
- The migration is atomic across scalar, clamp, and padding attributes to prevent two semantic
  scalar representations from coexisting.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [scalar glossary](../../../../glossary.md#term-scalar); [padding glossary](../../../../glossary.md#padding))</span>

## Known limitations

- [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) does not convert between data types, read [Tensor](../../../../glossary.md#tensor) storage, or materialize a Tensor.
- Current [scalar](../../../../glossary.md#term-scalar) arithmetic/clamp expressions remain floating-only.
- Direct [`Operation`](../../../../glossary.md#operation) construction cannot validate a future operand descriptor's data type;
  compiler occurrence validation must preserve this invariant later.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [data type glossary](../../../../glossary.md#data-type--datatype); [compiler glossary](../../../../glossary.md#term-compiler); [occurrence glossary](../../../../glossary.md#term-occurrence); [validation glossary](../../../../glossary.md#term-validation); [invariant glossary](../../../../glossary.md#term-invariant))</span>
- No [backend](../../../../glossary.md#backend) numerical behavior, [gradient](../../../../glossary.md#term-gradient) rule, [constant folding](../../../../glossary.md#term-constant-folding), or execution support is implied.
- Typed TensorFactory scalar/full overloads remain undecided until task 0018S.

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) context `/root/plan_0018n` reviewed the required architecture, documentation and planning
rules; roadmap, capability baseline, and [model](../../../../glossary.md#term-model) master plan; completed tasks 0001, 0014E, 0014F,
0017I, 0017J, 0018K, and 0018M1; current [DataType](../../../../glossary.md#data-type--datatype)/[BFLOAT16](../../../../glossary.md#term-bfloat16), [operation signature](../../../../glossary.md#operation-signature), [scalar](../../../../glossary.md#term-scalar)/clamp/pad
attributes, [Tensor](../../../../glossary.md#tensor) helpers and public methods, [eager](../../../../glossary.md#term-eager-evaluation) scalar/full factory contracts, focused tests,
Tensor [API](../../../../glossary.md#term-application-programming-interface-api), [Compile](../../../../glossary.md#compile) API, glossary sections, and Java 26 [Gradle](../../../../glossary.md#term-gradle) configuration.

Planning found no architecture conflict. Package direction is acyclic, current [operation](../../../../glossary.md#operation)-
[signature](../../../../glossary.md#term-signature) classes remain sufficient unchanged, and the twenty-path [atomic](../../../../glossary.md#resource-transaction-and-rollback) migration is inside one
model foundation contract. Implementation and [checkpoint](../../../../glossary.md#term-checkpoint) evidence remain empty until execution.

Planning-only validation:

- Git status contains exactly this new task, the model master plan, and the roadmap. No Java,
  test, Gradle, architecture, API, glossary, capability, other-module, or later-task file changed.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>
- A local Markdown target-and-anchor check resolved all 300 local links across the three changed
  files.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>
- The canonical section scan found every required task-specification section, and task 0018N is
  synchronized as Ready in this specification, the model task table, the roadmap task table, and
  current-frontier wording.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [scan glossary](../../../../glossary.md#term-cumulative-scan); [Ready glossary](../../../../glossary.md#term-ready))</span>
- Task ordering remains unchanged: tasks through 0018M1 are Complete, 0018N is Ready, and 0018O
  and every later task remain Draft. No task-0018O-or-later detailed specification exists.
- Package, dependency, and maximum-scope review confirms one new datatype value, migrations in
  existing operation/Tensor packages, thirteen Java paths, seven documentation paths, and the
  recorded atomic-migration rationale without an architecture conflict.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>
- Code-fence counts are balanced, all changed files end with a newline, targeted trailing-
  whitespace scans found no matches, and `git diff --check` passed.
- No Gradle test or Javadoc task was run because this planning-only change modifies no Java,
  executable behavior, or public Javadoc.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>

Implementation context `/root/task_0018n_implementation` completed the executable migration and
recorded this evidence before the documentation pass:

- Its focused selection passed 57 tests after the final test adaptation.
- `./gradlew :modules:model:test` passed with `BUILD SUCCESSFUL in 2s`; the Gradle XML reports
  total 770 tests and no failure/error files.
- Executable Java did not change after that final module run. The documentation context therefore
  reused the evidence as required by the documentation workflow instead of repeating the suite.

Documentation context `/root/task_0018n_implementation/task_0018n_docs` independently read the
[architecture contract](../../../../glossary.md#architecture-contract), documentation and planning rules, selected General/API-[Javadoc](../../../../glossary.md#term-javadoc)/Planning/
Example profiles, task 0018N, final affected production and test source, generated Javadoc,
Tensor/Compile/Training APIs, glossary, capabilities, model master plan, roadmap, and actual diff.
It found no executable-contract or architecture defect and finalized the documentation-only part
of all seven affected production Java paths plus all seven allowed documentation/planning paths.

Final documentation and checkpoint evidence:

- `./gradlew :modules:model:javadoc` passed after the final Javadoc edit with `BUILD SUCCESSFUL in
  1s`; two actionable tasks executed and the configuration [cache](../../../../glossary.md#term-cache) was reused. Generated pages for
  [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue), `ScalarValueAttrs`, `ClampRangeAttrs`, `PadAttrs`, and `Tensor` were inspected for
  typed-bit semantics, exact matching, retained FLOAT64 conveniences, failures, and boundaries.<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64))</span>
- A local Ruby target-and-anchor check resolved 376 local links across the seven changed
  documentation files.
- A local Ruby formatting check found balanced fences, no trailing whitespace, and final newlines
  in all seven changed documentation files.
- A local Ruby [scope](../../../../glossary.md#term-scope) check compared `git diff --name-only` plus untracked files with the allowed
  list and passed the exact twenty-path scope: seven production files, six tests, and seven
  documentation/planning files.
- Package scans confirmed `ScalarValue` and `ScalarValueTest` in
  `io.github.pho001.synaptik.model.datatype`. Status scans confirmed 0018N Complete and 0018O and
  every later task Draft; no task-0018O specification exists.
- `git diff --check` passed on the final combined change.
- Coordinator context `/root/task_0018n_implementation` ran the required foundation checkpoint
  `./gradlew test` after the documentation pass. It passed with `BUILD SUCCESSFUL in 1s`; 36
  actionable tasks were reported, two executed and 34 up-to-date, with the configuration cache
  reused. Output explicitly included `:modules:model:test` executed and
  `:testing:architecture-tests:test`, `:testing:backend-conformance:test`, and
  `:testing:integration-tests:test` as `NO-SOURCE`.

Reasoned no-change conclusions:

- `TensorFactory`, `TensorConstants`, `BFloat16Bits`, and their focused tests remain correct:
  existing primitive scalar/full methods already select exact types, BFLOAT16 conversion remains
  explicit, and this task adds no typed factory entry point or constant redesign.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant))</span>
- `OperationSignature`, [`OperationKind`](../../../../glossary.md#operationkind), `Operation`, scalar/pad kind declarations, and the
  production signature matrix remain correct because signatures validate only exact attribute
  class and occurrence cardinality; receiver-aware type equality belongs to Tensor construction
  and future compiler occurrence validation.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings); [occurrence glossary](../../../../glossary.md#term-occurrence); [cardinality glossary](../../../../glossary.md#term-cardinality); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- Training API remains accurate unchanged because typed scalar representation adds no optimizer,
  gradient, autograd, training-session, or backend-specific training behavior.<span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer); [gradient glossary](../../../../glossary.md#term-gradient); [autograd glossary](../../../../glossary.md#autograd); [backend glossary](../../../../glossary.md#backend))</span>
- Architecture documents, ADRs, and architecture tests require no change because module
  ownership, dependencies, and lifecycle boundaries did not change. Backend-conformance and
  integration tests remain <span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [lifecycle glossary](../../../../glossary.md#lifecycle); [conformance glossary](../../../../glossary.md#term-conformance))</span>`NO-SOURCE` because no numerical backend or end-to-end execution
  behavior was added.
- Gradle configuration, dependencies, other modules, and later task specifications remain
  unchanged because the implementation is confined to existing `modules/model` packages and
  model documentation.

## Implementation notes

- Added final `model.datatype.ScalarValue` with one exact [`DataType`](../../../../glossary.md#data-type--datatype) and primitive `long` payload,
  named construction for all six current types, strict inspectors, exact typed-bit equality and
  hashing, and fixed-width diagnostics.<span data-glossary-navigation="true"> ([payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication))</span>
- Atomically migrated [scalar](../../../../glossary.md#term-scalar), clamp, and [padding](../../../../glossary.md#padding) attributes from raw `double` to non-null
  [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue), retaining exact references and deterministic [validation](../../../../glossary.md#term-validation).
- Added six typed Tensor overloads with receiver-aware exact data-type equality and retained the
  six existing <span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>`double` overloads as exact-[FLOAT64](../../../../glossary.md#term-float64) adapters.
- Preserved operation-signature classes and cardinalities, expression metadata, Shape arithmetic,
  provenance, storage absence, input state, freshness, and identifier side effects.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [signature glossary](../../../../glossary.md#term-signature); [metadata glossary](../../../../glossary.md#term-metadata); [Shape glossary](../../../../glossary.md#shape); [provenance glossary](../../../../glossary.md#provenance))</span>
- Finalized Javadocs and explanatory documentation for the distinction between a typed scalar
  value and scalar Tensor, exact FLOAT32/raw-BFLOAT16/large-INT64/BOOL cases, signed-zero and NaN
  payload retention, unchanged factory interaction, and future compiler validation responsibility.<span data-glossary-navigation="true"> ([typed scalar value glossary](../../../../glossary.md#typed-scalar-value--scalarvalue); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [INT64 glossary](../../../../glossary.md#data-type--datatype); [BOOL glossary](../../../../glossary.md#data-type--datatype); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [compiler glossary](../../../../glossary.md#term-compiler))</span>

## Completion summary

- Completed changes: exact typed scalar representation, atomic scalar/clamp/pad attribute
  migration, strict Tensor-boundary compatibility, retained FLOAT64 conveniences, focused tests,
  Javadocs, API references, glossary, capability baseline, and synchronized planning status.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [Tensor glossary](../../../../glossary.md#tensor); [FLOAT64 glossary](../../../../glossary.md#term-float64); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>
- Files changed or created: the seven production Java files, six focused test files, and seven
  documentation/planning files listed under Affected files; no other path changed.
- Tests and validation: 57 focused tests passed; final 770-test model suite passed; final model
  Javadoc, generated-page inspection, 376-link/anchor check, formatting/fence/final-newline check,
  exact twenty-path scope, package/status/no-later-spec scans, <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check`, and the root
  foundation checkpoint all passed.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Documentation-agent review: clean context
  `/root/task_0018n_implementation/task_0018n_docs` independently finalized the affected contracts
  using the General, API/Javadoc, Planning, and Example profiles.
- Documentation impact: Tensor API, Compile API, glossary, capability baseline, this task, model
  master plan, and roadmap now describe the completed exact typed scalar contract and its current
  versus planned boundaries.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- Javadoc review: all seven affected production Java files were reviewed and finalized; generated
  public pages were inspected after the last edit.
- Glossary impact: added the reusable typed-scalar-value definition and aligned DataType,
  OperationAttrs, padding, scalar-family, and Tensor wording.<span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype); [OperationAttrs glossary](../../../../glossary.md#operationattrs); [padding glossary](../../../../glossary.md#padding))</span>
- Unresolved issues: None.
- Follow-up required: None for task 0018N. Task 0018O and later work remain Draft.

Status: Complete
