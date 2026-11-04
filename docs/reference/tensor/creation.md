# Allocation, import and [constant](../../glossary.md#term-constant) contracts

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

## Choose the boundary by who already has the data

- If the application has only type/shape facts, it calls `TensorFactory.create`.
- If it needs a new region for already resolved geometry, it calls `allocate`.
- If it has primitive-array values, it calls an import method, which copies them.
- If it already manages a memory region, the complete `create` overload can retain a compatible
  storage wrapper without copying it. That application still keeps the region alive.

“Caller” here is the application or framework component invoking the factory, not the eventual
backend executor. Retaining an immutable descriptor is different from retaining mutable memory.
For runnable examples use [flat/nested import](import.md); for the no-copy association's
cleanup responsibility use [storage lifetimes](host-storage.md).

**[Reader model](../../glossary.md#term-reader-model).** Choose among three different actions: `create` names a described [Tensor](../../glossary.md#tensor),
`allocate` reserves storage for resolved geometry, and import copies supplied values into new
storage. A **[carrier](../../glossary.md#term-storage-carrier)** is the Java primitive array holding values or raw encoded bits.
**[Row-major](../../glossary.md#term-row-major)** order visits the last [axis](../../glossary.md#term-axis) fastest. **Dense-[contiguous](../../glossary.md#term-contiguous)** geometry stores that
sequence without gaps, starting at offset zero. A **[leaf](../../glossary.md#term-leaf)** is a Tensor without expression
[provenance](../../glossary.md#provenance); having stored values does not automatically make it a [compiler](../../glossary.md#term-compiler) constant.
For [coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix), span, and offset terminology, start with [layouts](layouts.md).

Construction remains package-private so `Tensor` keeps one [validation](../../glossary.md#term-validation) path. Code outside
`model.tensor` creates tensors through the public static `TensorFactory`: `create(descriptor)`
creates an unlabeled storage-free tensor, while the complete overload also accepts optional label
text and optional existing [borrowed](../../glossary.md#ownership-borrowing-and-lease) [host storage](../../glossary.md#host-storage). Both overloads retain the exact [descriptor](../../glossary.md#term-descriptor), and
the complete overload retains the exact compatible storage object. The factory does not construct
descriptors or choose or resolve [layouts](../../glossary.md#layout); these two creation overloads do not import values.

For a descriptor whose layout is resolved, `allocate(descriptor)` and
`allocate(descriptor, label)` allocate exactly `layout.referencedElementSpan()` [elements](../../glossary.md#term-element). The
factory maps [`FLOAT64`](../../glossary.md#term-float64) to `double[]`, [`FLOAT32`](../../glossary.md#term-float32) to `float[]`, both [`BFLOAT16`](../../glossary.md#term-bfloat16) and [`FLOAT16`](../../glossary.md#term-float16) to raw
`short[]`, [`INT32`](../../glossary.md#data-type--datatype) to `int[]`, [`INT64`](../../glossary.md#data-type--datatype) to `long[]`, and [BOOL](../../glossary.md#data-type--datatype)/both [FP8](../../glossary.md#term-fp8) types to `byte[]`. The new array
begins with the [JVM](../../glossary.md#term-java-virtual-machine-jvm)'s default all-zero raw representation. A span above `Integer.MAX_VALUE` is
rejected before array allocation, and an unresolved layout is rejected rather than inferred. These overloads do not
provide typed access, copy or conversion, [boolean](../../glossary.md#data-type--datatype) [normalization](../../glossary.md#term-normalization), value import, fill operations,
native allocation, or descriptor synthesis. The separate constant methods deliberately synthesize
only [canonical](../../glossary.md#term-canonical-form) dense descriptors and reuse this allocation path for zeros.

Flat typed import is the current value-population boundary. `fromFlatArray(descriptor, label,
source)` has exactly six overloads and maps carriers without conversion:

| Source carrier | Required [data type](../../glossary.md#data-type--datatype) | Imported meaning |
|---|---|---|
| `double[]` | `FLOAT64` | [IEEE 754](../../glossary.md#term-ieee-754) [binary64](../../glossary.md#floating-point-precision-and-rounding) values copied bit-for-bit |
| `float[]` | `FLOAT32` | IEEE 754 [binary32](../../glossary.md#floating-point-precision-and-rounding) values copied bit-for-bit |
| `short[]` | `BFLOAT16` | Raw BFLOAT16 bits copied unchanged |
| `int[]` | `INT32` | Signed 32-bit integer values |
| `long[]` | `INT64` | Signed 64-bit integer values |
| `byte[]` | `BOOL` | Zero becomes `0`; every non-zero value becomes canonical `1` |
| `byte[]` | `FLOAT8_E4M3` or `FLOAT8_E5M2` | Exact raw bytes copied unchanged, without BOOL normalization |

FLOAT16 raw shorts use the separate `fromFlatFloat16Array(descriptor, label, short[])`
method, with the same dense-layout, exact-count and independent-copy requirements.

The descriptor must have the exact mapped data type and a resolved `DENSE_CONTIGUOUS` layout.
Its known logical element count must equal the Java array length. The factory treats source order
as logical row-major order, allocates a new matching [heap](../../glossary.md#term-heap) array through the existing allocation
path, and completely copies or normalizes the values before returning the tensor. The source
array is neither retained nor mutated, so changing it after return cannot change the tensor.
[Rank](../../glossary.md#term-rank)-0 [scalars](../../glossary.md#term-scalar) require one source element, and empty dense tensors require an empty source array.

Data-type, unresolved-layout, non-dense-layout, and length failures happen before destination or
identifier allocation. A blank label reaches the existing allocation and Tensor-construction
path, so destination storage and an identifier are allocated before the label fails; no copy is
performed. Identifier exhaustion is also observed after destination allocation and before copy.
Unexpected copy failures occur after identifier allocation and are not rolled back. Offset-dense,
strided, broadcast, and unresolved descriptors remain unsupported because importing independent
row-major values into those geometries requires a separate scatter or [view](../../glossary.md#term-view-meanings) policy.

Nested typed import is the descriptor-[inference](../../glossary.md#term-inference-meanings) boundary. The single
`fromNestedArray(source, label, requiresGrad)` method accepts `Object` because Java gives each
primitive-array rank a distinct runtime class and no finite overload family can express arbitrary
rank. The runtime class must prove rank two or greater and an ultimate carrier of `double`,
`float`, `short`, `int`, `long`, or `byte`; those carriers infer `FLOAT64`, `FLOAT32`, `BFLOAT16`,
`INT32`, `INT64`, and `BOOL`, respectively. Boxed arrays, generic object arrays, [rank-one](../../glossary.md#axis-rank-scalar-vector-and-matrix) arrays,
and every other primitive carrier are rejected rather than converted or defaulted.

The complete reachable structure is validated in depth-first row-major order before the
intermediate flat carrier, destination, or tensor ID is allocated. Every corresponding axis must
have the same length, and every required subarray must be non-null. Paths in diagnostics use
zero-based bracket notation: `[]` names the root and `[1][2]` names its second child's third child.
An empty final primitive axis is observable and valid, so `new int[2][0]` infers [shape](../../glossary.md#shape) `[2, 0]`.
An empty earlier axis is rejected because the remaining extents cannot be recovered from the
nested-array structure.

After validation, nested import allocates a fresh matching primitive array and copies leaf arrays
into it in logical row-major order. Numeric values and raw BFLOAT16 shorts are unchanged. BOOL
bytes remain raw in that intermediate array and are normalized once by flat import while copying
to destination storage. The inferred `TensorDescriptor` has the exact data type, fully static
shape, canonical dense-contiguous layout, and requested `requiresGrad` flag; descriptor validation
remains authoritative for [gradient](../../glossary.md#term-gradient) eligibility. No source level is retained or mutated, and later
caller mutation cannot change the tensor. Callers must not mutate the source concurrently with
import because inspection and flattening do not provide an [atomic](../../glossary.md#resource-transaction-and-rollback) deep [snapshot](../../glossary.md#term-snapshot).

Structural, carrier, checked-count, Java-array-limit, and descriptor-eligibility failures consume
no ID and allocate no destination. Descriptor eligibility is checked after the intermediate array
is allocated. A blank label reaches flat import, so destination storage is allocated and one ID is
consumed before Tensor validation fails. Identifier exhaustion is likewise observed after
destination allocation and before the final flat-to-storage copy. Neither failure exposes or
retains the intermediate carrier.

Constant creation is the factory's other descriptor-synthesis boundary. Its scalar methods
infer an exact data type from the declared primitive input and create canonical rank-0 dense
tensors:

| Method input | Result data type | Stored representation |
|---|---|---|
| `double` | `FLOAT64` | Exact binary64 bits |
| `float` | `FLOAT32` | Exact binary32 bits |
| `scalarBFloat16(float)` | `BFLOAT16` | `BFloat16Bits.fromFloat(value)` |
| `scalarFloat16(float)` | `FLOAT16` | One numeric [binary16](../../glossary.md#floating-point-precision-and-rounding) [rounding](../../glossary.md#term-rounding) |
| `scalarFloat8E4m3(float)` | `FLOAT8_E4M3` | One numeric [E4M3](../../glossary.md#term-e4m3) rounding |
| `scalarFloat8E5m2(float)` | `FLOAT8_E5M2` | One numeric [E5M2](../../glossary.md#term-e5m2) rounding |
| `int` | `INT32` | Exact signed 32-bit value |
| `long` | `INT64` | Exact signed 64-bit value |
| `boolean` | `BOOL` | Canonical byte `0` or `1` |

The BFLOAT16 method is named explicitly because it rounds a semantic binary32 input to BFLOAT16
using [round-to-nearest](../../glossary.md#floating-point-precision-and-rounding) with ties to even; it is not the FLOAT32 overload and does not accept raw
`short` bits. Every scalar descriptor uses `Shape.scalar()`, a resolved canonical
dense-contiguous layout, exactly one logical element, and the caller's explicit `requiresGrad`
request.

`zeros(shape, dataType, label, requiresGrad)` and
`ones(shape, dataType, label, requiresGrad)` accept every current data type. The shape must be
fully static, but it may be rank zero or contain a zero-sized [dimension](../../glossary.md#dimension). Both methods synthesize a
new canonical dense descriptor. Zeros allocate the destination once and rely on JVM primitive
array initialization, so no source array or element fill is performed. Ones fill one exact typed
source carrier with `1.0d`, `1.0f`, BFLOAT16 bits `0x3F80`, `1`, `1L`, or BOOL byte `1`, then copy
through the matching flat-import overload. An empty shape therefore creates empty storage rather
than one physical element.

`zerosLike(template, label, requiresGrad)` and
`onesLike(template, label, requiresGrad)` read only the template descriptor's shape and data type.
They do not inspect or preserve the template layout, label, storage, storage [liveness](../../glossary.md#term-liveness), [identity](../../glossary.md#term-identity-meanings),
gradient request, or other state. Static unresolved, offset, strided, broadcast, and dense
templates are all accepted; the result always has a newly synthesized canonical dense layout and
uses the explicit result label and gradient request. Each successful call creates a new Tensor,
descriptor, layout, storage wrapper, backing array, and ID. The template object and any temporary
source carrier are not retained.

Public null checks happen before descriptor or storage work. Constant descriptor validation then
requires a fully static shape, calculates its checked logical element count, applies the Java
array limit, constructs dense layout geometry, and validates gradient eligibility in that order.
Those failures allocate no destination and consume no ID. Source-carrier allocation for scalars
and ones occurs only after descriptor validation. A blank label fails later, after destination and
ID allocation, and consumes that ID; scalar and one creation have also allocated their source
carrier. Identifier exhaustion is observed after destination allocation, with the same source
carrier distinction. A JVM allocation failure that occurs before ID allocation consumes no ID.

Full-value creation extends the same dense constant path without adding a generic conversion or
[mutable](../../glossary.md#term-mutable) fill [operation](../../glossary.md#operation). The type-safe entries are:

| Factory value | Result data type | Stored value |
|---|---|---|
| `double` | `FLOAT64` | Exact binary64 value, including [signed-zero](../../glossary.md#nan-infinity-and-signed-zero) and [NaN payload](../../glossary.md#nan-infinity-and-signed-zero) bits |
| `float` | `FLOAT32` | Exact binary32 value, including signed-zero and NaN payload bits |
| `fullBFloat16(..., float, ...)` | `BFLOAT16` | One `BFloat16Bits.fromFloat(value)` conversion repeated as [raw bits](../../glossary.md#storage-mapping-contiguity-and-injectivity) |
| `fullFloat16(..., float, ...)` | `FLOAT16` | Once-rounded binary16 bits repeated |
| `fullFloat8E4m3(..., float, ...)` | `FLOAT8_E4M3` | Once-rounded E4M3 byte repeated |
| `fullFloat8E5m2(..., float, ...)` | `FLOAT8_E5M2` | Once-rounded E5M2 byte repeated |
| `int` | `INT32` | Exact signed 32-bit value |
| `long` | `INT64` | Exact signed 64-bit value |
| `boolean` | `BOOL` | Canonical false byte `0` or true byte `1` |

Every `full` method requires a fully static caller shape, but [rank-zero](../../glossary.md#axis-rank-scalar-vector-and-matrix) scalar and zero-element
shapes are valid. It constructs one canonical dense descriptor, fills one exact primitive source
carrier, and delegates once to matching flat import. The source is copied and not retained. The
result has independent descriptor, layout, destination storage, Tensor identity, and empty
provenance. The [API](../../glossary.md#term-application-programming-interface-api) deliberately has no boxed value, caller-selected conversion, raw BFLOAT16
scalar, `fullLike`, default-label overload, or post-construction mutation.

`identityMatrix(rows, columns, dataType, label, requiresGrad)` creates a canonical dense [rank-two](../../glossary.md#axis-rank-scalar-vector-and-matrix)
[matrix](../../glossary.md#term-matrix-meanings) for any current [Model](../../glossary.md#term-model) data type. Both dimensions are non-negative `long` values, so
square, wide, tall, zero-row, and zero-column matrices are valid when their checked logical count
fits a Java array. The method writes typed one only at coordinates `(i, i)` for
`0 <= i < min(rows, columns)` and leaves every other position at typed zero. Its exact values are
`1.0d`/`0.0d`, `1.0f`/`0.0f`, converted BFLOAT16 one bits/raw zero bits, `1`/`0`, `1L`/`0L`, or
canonical BOOL `1`/`0`. `eye(...)` is only an unchanged-[argument](../../glossary.md#term-argument) call to `identityMatrix(...)`.
Equal arguments therefore produce equal descriptors and values, while separate successful calls
still return different Tensor, ID, descriptor, layout, storage, and backing-array objects.

Full-value methods check shape then label nullity before static-shape, checked-count, Java array
limit, dense-layout, and gradient validation. Identity checks data type then label nullity,
negative rows then negative columns, and then uses the same checked descriptor path. These early
failures allocate no carrier and consume no ID. Each successful path creates one source carrier,
then flat import creates one destination and allocates the ID. A blank label therefore fails after
both arrays and the ID exist and consumes that ID. Exhaustion occurs after both arrays exist, and
an unexpected flat-copy failure occurs after ID allocation; identifiers are never rolled back.


---

[Previous topic](identity.md) · [Next topic](constants.md)
