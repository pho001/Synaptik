# Task 0013A: Full-Value and [Identity](../../../../glossary.md#term-identity-meanings)-[Matrix](../../../../glossary.md#term-matrix-meanings) [Tensor](../../../../glossary.md#tensor) Creation

## Status

Complete

## Goal

Complete the current [eager](../../../../glossary.md#term-eager-evaluation) factory foundation with type-safe full-value [tensors](../../../../glossary.md#tensor) and dense
rectangular [identity](../../../../glossary.md#term-identity-meanings) [matrices](../../../../glossary.md#term-matrix-meanings). `full` fills every logical [element](../../../../glossary.md#term-element) from one exact primitive [scalar](../../../../glossary.md#term-scalar);
`identityMatrix` creates a [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) matrix with typed one on the main diagonal and typed zero
elsewhere; `eye` is an exact convenience [alias](../../../../glossary.md#aliasing-and-arena) for the [canonical](../../../../glossary.md#term-canonical-form) identity-matrix method.

Every result is a new canonical dense-[contiguous](../../../../glossary.md#term-contiguous) [leaf](../../../../glossary.md#term-leaf) tensor with independent [JVM](../../../../glossary.md#term-java-virtual-machine-jvm)-managed [heap](../../../../glossary.md#term-heap)
storage, factory-assigned identity, and empty [provenance](../../../../glossary.md#provenance). This task adds no expression [Operation](../../../../glossary.md#operation),
[graph](../../../../glossary.md#graph) behavior, generic conversion policy, or typed Tensor mutation [API](../../../../glossary.md#term-application-programming-interface-api).

## [Scope](../../../../glossary.md#term-scope)

- Add six exact primitive-carrier full-value factory methods: FLOAT64, FLOAT32, explicitly converted
  BFLOAT16, INT32, INT64, and semantic BOOL.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- Require a fully static caller-supplied Shape for every full-value result and preserve scalar and
  zero-element shape semantics.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [scalar glossary](../../../../glossary.md#term-scalar); [element glossary](../../../../glossary.md#term-element))</span>
- Add one [canonical](../../../../glossary.md#term-canonical-form) rectangular `identityMatrix` factory method for all six current data types.
- Add one `eye` method with the exact same parameters and behavior, implemented only by delegating
  to <span data-glossary-navigation="true"> ([parameters glossary](../../../../glossary.md#term-parameter))</span>`identityMatrix`.
- Accept non-negative `long` row and column counts, including rectangular and zero-element
  matrices, and synthesize rank-two canonical dense geometry.<span data-glossary-navigation="true"> ([matrices glossary](../../../../glossary.md#term-matrix-meanings); [rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- Reuse the existing package-private `TensorConstants` helper, descriptor validation, exact typed
  flat import, ID allocation, storage ownership, label normalization, and gradient eligibility.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [validation glossary](../../../../glossary.md#term-validation); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [normalization glossary](../../../../glossary.md#term-normalization); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Extend only exact factory/helper surface tests and add one focused full/[identity](../../../../glossary.md#term-identity-meanings) behavior suite.
- Finalize affected Javadocs, Tensor API, glossary, task evidence, master plan, and roadmap through
  an independent documentation pass during implementation.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- boxed/generic `Number` or `Object` values, `DataType + double` conversion, caller-selected
  conversion, implicit widening/narrowing/rounding, numeric truthiness, strings, or null values<span data-glossary-navigation="true"> ([widening glossary](../../../../glossary.md#term-widening); [narrowing glossary](../../../../glossary.md#term-narrowing); [rounding glossary](../../../../glossary.md#term-rounding))</span>
- raw-bit BFLOAT16 full input; BFLOAT16 semantic input remains an explicitly named binary32
  conversion, while raw short arrays remain available through flat import<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [binary32 glossary](../../../../glossary.md#floating-point-precision-and-rounding))</span>
- `fullLike`, default shape/type/value/label/gradient overloads, mutable fill, in-place fill,
  post-construction writes, or typed Tensor access/export<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [gradient glossary](../../../../glossary.md#term-gradient); [mutable glossary](../../../../glossary.md#term-mutable); [Tensor glossary](../../../../glossary.md#tensor))</span>
- square-only convenience overloads, size-only `eye`, Shape-based identity input, diagonal offset,
  batched identity, sparse identity, band matrices, triangular matrices, or view-based identity<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [matrices glossary](../../../../glossary.md#term-matrix-meanings); [view glossary](../../../../glossary.md#term-view-meanings))</span>
- preserving another Tensor's shape/layout/storage/label/gradient/provenance or adding an identity-
  like method<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance))</span>
- treating `eye` as a separate implementation, allocating before canonical delegation, changing
  validation/messages, or returning a shared/cached Tensor<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [validation glossary](../../../../glossary.md#term-validation))</span>
- operations, provenance creation, graph capture/traversal, compiler constants, constant folding,
  autograd, runtime/backend generation, device/native storage, or prepared execution<span data-glossary-navigation="true"> ([graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [compiler glossary](../../../../glossary.md#term-compiler); [constants glossary](../../../../glossary.md#term-constant); [constant folding glossary](../../../../glossary.md#term-constant-folding); [autograd glossary](../../../../glossary.md#autograd); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [device glossary](../../../../glossary.md#term-device); [prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution))</span>
- new production types/packages, storage contracts, dependencies, preview/incubator features,
  Gradle, architecture changes, another module, or task-0014 specification<span data-glossary-navigation="true"> ([features glossary](../../../../glossary.md#term-software-feature); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Architecture overview](../../../../architecture/overview.md)
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Model master plan](../master-plan.md)
- [Task 0001](0001-data-type-model.md)
- [Task 0002](0002-shape-and-dimension-model.md)
- [Task 0003](0003-layout-descriptor-model.md)
- [Task 0007](0007-tensor-descriptor-model.md)
- [Task 0011](0011-public-tensor-skeleton.md)
- [Task 0012](0012-tensor-factory.md)
- [Task 0012A](0012a-host-storage-allocation.md)
- [Task 0012B](0012b-flat-typed-tensor-import.md)
- [Task 0012D](0012d-constant-tensor-creation.md)
- [Task 0013](0013-tensor-provenance-skeleton.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Glossary](../../../../glossary.md)

## Capability origin

Read-only inspection of the legacy [Tensor factory](../../../../glossary.md#tensor-factory) and tests found [scalar](../../../../glossary.md#term-scalar), zeros, ones, like-shaped
[constants](../../../../glossary.md#term-constant), ranges, and [fixture](../../../../glossary.md#term-fixture)-population helpers, but no stable public `full`, `identityMatrix`, or
`eye` contract to copy. These capabilities are explicit additions selected in the current [model](../../../../glossary.md#term-model)
capability baseline. Legacy code remains evidence for exact [carrier](../../../../glossary.md#term-storage-carrier) handling and independent [leaf](../../../../glossary.md#term-leaf)
storage only; it is not source or architecture for this task.

## Architecture constraints

- Work remains entirely in `modules/model` and the existing
  `io.github.pho001.synaptik.model.tensor` package.
- Public creation methods belong to `TensorFactory`; typed population mechanics remain in the
  existing stateless package-private `TensorConstants` helper.
- Every result is [eager](../../../../glossary.md#term-eager-evaluation) copied [leaf](../../../../glossary.md#term-leaf) data with `Optional.empty()` provenance through existing flat
  import. Do not call <span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>`createDerived` or construct an [Operation](../../../../glossary.md#operation).
- Primitive overloads infer exact [DataType](../../../../glossary.md#data-type--datatype) and perform no implicit conversion. Only
  `fullBFloat16(float, ...)` explicitly invokes the completed [BFLOAT16](../../../../glossary.md#term-bfloat16) conversion.
- Full-value and identity results synthesize only canonical dense-contiguous descriptors and new
  independent heap storage. They accept no layout or storage input.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [contiguous glossary](../../../../glossary.md#term-contiguous); [descriptors glossary](../../../../glossary.md#term-descriptor); [heap glossary](../../../../glossary.md#term-heap); [layout glossary](../../../../glossary.md#layout))</span>
- Existing `TensorDescriptor` remains authoritative for requires-grad eligibility.
- Existing flat import remains authoritative for destination allocation, copying, BOOL
  normalization, label validation, identity allocation, and failure side effects.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [normalization glossary](../../../../glossary.md#term-normalization); [validation glossary](../../../../glossary.md#term-validation))</span>
- `eye` delegates exactly once to the public [canonical](../../../../glossary.md#term-canonical-form) `identityMatrix` method and owns no
  validation, descriptor, carrier, population, allocation, or ID logic.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- No new package is justified: the behavior is cohesive constant creation and requires existing
  package-private factory/helper collaboration.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant))</span>
- Stop if implementation requires a generic scalar abstraction, conversion policy, new storage or
  Tensor mutation API, Operation/provenance, dependency, module, or architecture decision.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [dependency glossary](../../../../glossary.md#term-dependency); [module glossary](../../../../glossary.md#term-module))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.tensor` — owns TensorFactory, TensorConstants, descriptors,
  flat-import collaboration, and focused tests.<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor))</span>
- `io.github.pho001.synaptik.model.datatype` — supplies exact data types and [BFLOAT16](../../../../glossary.md#term-bfloat16) conversion.
- `io.github.pho001.synaptik.model.shape` — supplies canonical static rank-two, scalar, and
  zero-element shapes.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [scalar glossary](../../../../glossary.md#term-scalar); [element glossary](../../../../glossary.md#term-element); [shapes glossary](../../../../glossary.md#shape))</span>
- `io.github.pho001.synaptik.model.layout` — supplies canonical dense-contiguous geometry through
  existing TensorConstants descriptor construction.<span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous))</span>

Packages added or changed:

- No package is added. Only the existing `model.tensor` package changes.

Type placement:

- `TensorFactory` receives exactly eight new public methods: six full-value overloads,
  `identityMatrix`, and `eye`.
- Existing package-private `TensorConstants` receives exactly seven new package-private entries:
  six full-value overloads and one identity-matrix entry. It receives no <span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>`eye` entry.
- Existing `TensorFactoryTest` and `TensorFactoryConstantTest` change only their exact public/helper
  surface expectations.
- New `TensorFactoryFullIdentityTest` owns focused value, geometry, validation, ownership, alias,
  and failure-side-effect tests.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>

## Required contract

### Public full-value methods

Add exactly:

```java
public static Tensor full(
        Shape shape, double value, Optional<String> label, boolean requiresGrad)

public static Tensor full(
        Shape shape, float value, Optional<String> label, boolean requiresGrad)

public static Tensor fullBFloat16(
        Shape shape, float value, Optional<String> label, boolean requiresGrad)

public static Tensor full(
        Shape shape, int value, Optional<String> label, boolean requiresGrad)

public static Tensor full(
        Shape shape, long value, Optional<String> label, boolean requiresGrad)

public static Tensor full(
        Shape shape, boolean value, Optional<String> label, boolean requiresGrad)
```

Mapping is exact:

| Method value | [Data type](../../../../glossary.md#data-type--datatype) | Stored value |
|---|---|---|
| `double` | [`FLOAT64`](../../../../glossary.md#term-float64) | exact [binary64](../../../../glossary.md#floating-point-precision-and-rounding) value and raw [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero)/[NaN payload](../../../../glossary.md#nan-infinity-and-signed-zero) |
| `float` | [`FLOAT32`](../../../../glossary.md#term-float32) | exact [binary32](../../../../glossary.md#floating-point-precision-and-rounding) value and raw signed-zero/NaN payload |
| `fullBFloat16(float)` | [`BFLOAT16`](../../../../glossary.md#term-bfloat16) | `BFloat16Bits.fromFloat(value)` in every [element](../../../../glossary.md#term-element) |
| `int` | [`INT32`](../../../../glossary.md#data-type--datatype) | exact signed 32-bit value |
| `long` | [`INT64`](../../../../glossary.md#data-type--datatype) | exact signed 64-bit value |
| [`boolean`](../../../../glossary.md#data-type--datatype) | [`BOOL`](../../../../glossary.md#data-type--datatype) | [canonical](../../../../glossary.md#term-canonical-form) false byte `0` or true byte `1` |

Every public full method validates non-null [`shape`](../../../../glossary.md#shape) and `label` in that order before helper
delegation. No default label, unlabeled overload, [DataType](../../../../glossary.md#data-type--datatype) [argument](../../../../glossary.md#term-argument), raw short/byte overload, boxed
value, full-like variant, or conversion overload is added.

### Full-value [validation](../../../../glossary.md#term-validation) and population

Reuse the existing TensorConstants [descriptor](../../../../glossary.md#term-descriptor) path unchanged:

1. reject dynamic [shape](../../../../glossary.md#shape) with `IllegalArgumentException` and exact message
   `constant tensor creation requires a fully static shape: <shape>`;
2. obtain checked `shape.knownElementCount()`; non-zero multiplication [overflow](../../../../glossary.md#term-overflow) remains
   `ArithmeticException`;
3. reject count above `Integer.MAX_VALUE` with exact message
   `constant tensor element count exceeds Java array limit: required=<required>, maximum=2147483647`;
4. construct [canonical](../../../../glossary.md#term-canonical-form) [contiguous](../../../../glossary.md#term-contiguous) [layout](../../../../glossary.md#layout); checked geometry overflow remains `ArithmeticException`;
5. construct TensorDescriptor with the inferred exact type and explicit gradient request; existing
   gradient eligibility and message remain authoritative;<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient))</span>
6. allocate exactly one matching source [carrier](../../../../glossary.md#term-storage-carrier) of logical [element](../../../../glossary.md#term-element) count;
7. fill the carrier with `Arrays.fill` and the exact mapped value;
8. delegate exactly once to the matching `TensorFactory.fromFlatArray` overload.

Do not special-case zero, one, false, true, empty output, or [scalar](../../../../glossary.md#term-scalar) output through another public
factory method. This preserves exact negative-zero/[NaN](../../../../glossary.md#term-nan) bits and one uniform source/carrier/copy/ID
path. Scalar Shape has one element. A static shape with a zero [dimension](../../../../glossary.md#dimension) creates an empty source
and destination while retaining the requested [rank](../../../../glossary.md#term-rank)/shape.

### Public [identity](../../../../glossary.md#term-identity-meanings)-[matrix](../../../../glossary.md#term-matrix-meanings) methods

Add exactly:

```java
public static Tensor identityMatrix(
        long rows,
        long columns,
        DataType dataType,
        Optional<String> label,
        boolean requiresGrad)

public static Tensor eye(
        long rows,
        long columns,
        DataType dataType,
        Optional<String> label,
        boolean requiresGrad)
```

`identityMatrix` is [canonical](../../../../glossary.md#term-canonical-form). It supports all six current data types. It creates [Shape](../../../../glossary.md#shape)
`[rows, columns]`, stores typed one at [coordinates](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) `(i, i)` for
`0 <= i < min(rows, columns)`, and leaves all other positions at typed zero. This is valid for
square, wide, tall, and zero-[element](../../../../glossary.md#term-element) matrices.

Representations are:

| [Data type](../../../../glossary.md#data-type--datatype) | Diagonal | Off diagonal |
|---|---|---|
| [`FLOAT64`](../../../../glossary.md#term-float64) | `1.0d` | [JVM](../../../../glossary.md#term-java-virtual-machine-jvm) default `0.0d` |
| [`FLOAT32`](../../../../glossary.md#term-float32) | `1.0f` | JVM default `0.0f` |
| [`BFLOAT16`](../../../../glossary.md#term-bfloat16) | `BFloat16Bits.fromFloat(1.0f)` | [raw bits](../../../../glossary.md#storage-mapping-contiguity-and-injectivity) `0` |
| [`INT32`](../../../../glossary.md#data-type--datatype) | `1` | `0` |
| [`INT64`](../../../../glossary.md#data-type--datatype) | `1L` | `0L` |
| [`BOOL`](../../../../glossary.md#data-type--datatype) | canonical byte `1` | canonical byte `0` |

The public canonical method null-checks `dataType` and `label` in that order before helper
delegation. The helper validates and constructs in this order:

1. negative rows: `IllegalArgumentException`, message
   `identity matrix rows must be non-negative: <rows>`;
2. negative columns: `IllegalArgumentException`, message
   `identity matrix columns must be non-negative: <columns>`;
3. create `Shape.of(rows, columns)`;
4. reuse the existing descriptor path for checked logical count, Java-array limit, canonical dense
   layout, and gradient eligibility;
5. allocate exactly one matching default-zero primitive [carrier](../../../../glossary.md#term-storage-carrier) of logical element count;
6. write typed one only at [row-major](../../../../glossary.md#term-row-major) indices `i * columns + i` for the diagonal length;
7. delegate exactly once to the matching flat-import overload.

Because the result is [rank](../../../../glossary.md#term-rank) two, non-negative rows and columns are the complete [dimension](../../../../glossary.md#dimension) input.
Zero by any non-negative column count and any non-negative row count by zero are valid empty
matrices. Positive count [overflow](../../../../glossary.md#term-overflow) remains `ArithmeticException`; non-overflowing count above the
array limit uses the existing [constant](../../../../glossary.md#term-constant)-count error message.

`eye` contains exactly one return delegation to `identityMatrix` with unchanged [arguments](../../../../glossary.md#term-argument). It
performs no own null/dimension/[gradient](../../../../glossary.md#term-gradient)/label check and no allocation. Each successful call still
returns a fresh [Tensor](../../../../glossary.md#tensor) because the canonical method allocates normally. [Alias](../../../../glossary.md#aliasing-and-arena) equality means equal
[descriptor](../../../../glossary.md#term-descriptor)/value behavior for equal arguments, not the same Tensor, ID, storage, descriptor, or
[layout](../../../../glossary.md#layout) object across separate calls.

### Package-private helper surface

Extend TensorConstants with exactly:

```java
static Tensor full(
        Shape shape, double value, Optional<String> label, boolean requiresGrad)
static Tensor full(
        Shape shape, float value, Optional<String> label, boolean requiresGrad)
static Tensor fullBFloat16(
        Shape shape, float value, Optional<String> label, boolean requiresGrad)
static Tensor full(
        Shape shape, int value, Optional<String> label, boolean requiresGrad)
static Tensor full(
        Shape shape, long value, Optional<String> label, boolean requiresGrad)
static Tensor full(
        Shape shape, boolean value, Optional<String> label, boolean requiresGrad)
static Tensor identityMatrix(
        long rows,
        long columns,
        DataType dataType,
        Optional<String> label,
        boolean requiresGrad)
```

The helper remains final, package-private, stateless, field-free, and privately constructed. Its
package-private entry surface contains the existing eight [constant](../../../../glossary.md#term-constant) entries plus these seven, for
exactly fifteen entries. It has no `eye`, generic fill, [identity](../../../../glossary.md#term-identity-meanings) strategy, [cache](../../../../glossary.md#term-cache), [registry](../../../../glossary.md#term-registry),
[singleton](../../../../glossary.md#term-singleton), public/protected member, nested type, or production test hook.

Private helper methods may share exact [carrier](../../../../glossary.md#term-storage-carrier) fill/import and diagonal population only when the
specified public/helper surface, [validation](../../../../glossary.md#term-validation) order, messages, carrier counts, and single flat-import
behavior remain observable and unchanged.

## [Ownership](../../../../glossary.md#term-ownership-meanings) and failure effects

- Every success creates a new descriptor, layout, source carrier, destination carrier, storage,
  Tensor, and ID; no result or intermediate is cached or shared.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [layout glossary](../../../../glossary.md#layout); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Source carriers are not exposed or retained. Flat import copies them into independent storage.
- Results have empty [provenance](../../../../glossary.md#provenance) and no [operation](../../../../glossary.md#operation), [graph](../../../../glossary.md#graph), [runtime](../../../../glossary.md#term-runtime), or [backend](../../../../glossary.md#backend) state.
- Public null, shape/dimension, count, layout, and gradient failures precede source/destination
  allocation and ID consumption.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [dimension glossary](../../../../glossary.md#dimension); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Source-[carrier](../../../../glossary.md#term-storage-carrier) [OOME](../../../../glossary.md#term-oome) propagates before destination or ID allocation.
- A blank label is rejected by existing Tensor validation after source creation, destination
  allocation, and ID allocation, so it consumes one ID without rollback.<span data-glossary-navigation="true"> ([rollback glossary](../../../../glossary.md#term-rollback))</span>
- Identifier exhaustion occurs after source and destination allocation and before publication;
  no allocation or ID is rolled back.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication))</span>
- An unexpected flat-copy failure occurs after ID allocation and consumes that ID.
- `eye` has exactly the same [validation](../../../../glossary.md#term-validation) and side effects as `identityMatrix` because it performs
  only canonical delegation.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorFactory.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorConstants.java`

Existing tests, exact surface only:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryConstantTest.java`

New focused test:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryFullIdentityTest.java`

Documentation/[planning](../../../../glossary.md#term-planning) during implementation:

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless a task-related inconsistency requires stopping:

- `docs/planning/modules/model/capabilities.md` — already contains the selected full/identity
  capability and must not be expanded during implementation.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Existing DataType, BFloat16Bits, Shape, LayoutDescriptor, TensorDescriptor, Tensor,
  TensorProvenance, storage, allocation, and flat-import Javadocs.<span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype); [Shape glossary](../../../../glossary.md#shape); [Tensor glossary](../../../../glossary.md#tensor))</span>
- [Compile](../../../../glossary.md#compile) [API](../../../../glossary.md#term-application-programming-interface-api), Training API, focused architecture documentation, and architecture tests.

## Maximum [scope](../../../../glossary.md#term-scope)

At most two production files, two existing exact-surface tests, one new focused test, and five
documentation/[planning](../../../../glossary.md#term-planning) files: ten paths total. TensorFactory and TensorConstants plus their exact-
surface tests must change atomically so the [module](../../../../glossary.md#term-module) remains buildable.

Do not modify existing behavior assertions in TensorFactoryConstantTest, another factory helper or
test, capabilities, existing completed tasks, [Tensor](../../../../glossary.md#tensor)/[descriptor](../../../../glossary.md#term-descriptor)/storage/[provenance](../../../../glossary.md#provenance)/[operation](../../../../glossary.md#operation)/[graph](../../../../glossary.md#graph)
code, [Gradle](../../../../glossary.md#term-gradle), AGENTS, architecture docs/tests, another module, or unrelated documentation. Do not
create task 0014. Stop beyond ten paths or if implementation needs a new production concept,
package, conversion policy, Tensor mutation, Operation, or architecture decision.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Update TensorFactory type Javadoc to include exact full-value and identity-matrix leaf creation
  without implying general fill mutation or Operation semantics.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings); [leaf glossary](../../../../glossary.md#term-leaf); [Operation glossary](../../../../glossary.md#operation))</span>
- Fully document all eight public methods: exact primitive/type mapping, BFLOAT16 conversion,
  BOOL semantics, static/scalar/empty shapes, rectangular identity, diagonal definition, dense
  layout, explicit gradient/label behavior, ownership, provenance absence, validation order,
  allocation/ID/failure effects, parameters, results, and exceptions.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [BOOL glossary](../../../../glossary.md#data-type--datatype); [scalar glossary](../../../../glossary.md#term-scalar); [shapes glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [gradient glossary](../../../../glossary.md#term-gradient); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [provenance glossary](../../../../glossary.md#provenance); [validation glossary](../../../../glossary.md#term-validation); [parameters glossary](../../../../glossary.md#term-parameter))</span>
- Explicitly document `eye` as pure delegation and define [alias](../../../../glossary.md#aliasing-and-arena) equality versus object [identity](../../../../glossary.md#term-reference-identity).
- Update TensorConstants type Javadoc and document every new package-private entry and private
  population helper with exact carrier, descriptor, fill/diagonal, import, and failure behavior.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Review related component Javadocs and record reasoned no-change conclusions or stop when a
  required correction falls outside the authorized files.
- The documentation pass must add a newcomer-readable Tensor API example showing a nontrivial
  full tensor and one rectangular identity matrix with concrete row-major values and line-by-line
  commentary under the Example-format profile.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [row-major glossary](../../../../glossary.md#term-row-major))</span>

## Acceptance criteria

- TensorFactory exposes exactly six new type-safe full-value methods, one canonical
  identityMatrix, and one exact eye alias, with no other public change.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- TensorConstants exposes exactly seven matching package-private entries and no eye entry; it
  remains stateless with exactly fifteen package-private constant entries after the task.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant))</span>
- Full-value methods support all six data types, preserve exact floating raw values, explicitly
  convert BFLOAT16, use canonical BOOL, support scalar/empty/static shapes, and reject dynamic or
  over-limit shapes before allocation/ID.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [BOOL glossary](../../../../glossary.md#data-type--datatype); [scalar glossary](../../../../glossary.md#term-scalar); [shapes glossary](../../../../glossary.md#shape))</span>
- Identity supports all six data types; square, wide, tall, and zero-element rank-two shapes;
  exact main-diagonal typed ones; default typed zeros elsewhere; canonical dense geometry; and
  explicit gradient eligibility.<span data-glossary-navigation="true"> ([Identity glossary](../../../../glossary.md#term-identity-meanings); [element glossary](../../../../glossary.md#term-element); [rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Negative rows/columns, positive count overflow/limit, invalid gradients, null references, blank
  labels, OOME ordering, and identifier exhaustion follow the exact contracts and messages.<span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow); [OOME glossary](../../../../glossary.md#term-oome))</span>
- Eye bytecode/body is one unchanged-argument call to identityMatrix and has no helper entry or
  independent behavior.<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode); [argument glossary](../../../../glossary.md#term-argument))</span>
- Every result is a fresh provenance-free leaf with independent descriptor/layout/storage/backing
  array/ID and no source retention, Tensor mutation, Operation, or cross-layer state.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [leaf glossary](../../../../glossary.md#term-leaf); [descriptor glossary](../../../../glossary.md#term-descriptor); [layout glossary](../../../../glossary.md#layout); [Tensor glossary](../../../../glossary.md#tensor); [Operation glossary](../../../../glossary.md#operation))</span>
- Existing scalar/zero/one/like behavior and tests remain unchanged except exact helper/API surface
  assertions.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Focused and aggregate tests, Javadoc, root tests, javap/import/bytecode/manual checks,
  documentation links/examples/formatting, exact ten-path scope, and status synchronization pass.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope); [synchronization glossary](../../../../glossary.md#term-synchronization))</span>
- A separate clean-context documentation agent finalizes Javadocs, Tensor API, glossary, planning
  evidence/status, component no-change reviews, and documentation validation in the same change.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [validation glossary](../../../../glossary.md#term-validation))</span>
- Task 0013A becomes Complete only after both passes. The post-foundation checkpoint becomes the
  next planning action; task 0014 remains Draft without a detailed specification.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) before and after the documentation-focused review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryConstantTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryFullIdentityTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

Focused tests cover exact factory/helper surfaces; all carriers; raw [signed zero](../../../../glossary.md#term-signed-zero)/[NaN](../../../../glossary.md#term-nan); [BFLOAT16](../../../../glossary.md#term-bfloat16)
[rounding](../../../../glossary.md#term-rounding); [BOOL](../../../../glossary.md#data-type--datatype) [canonical](../../../../glossary.md#term-canonical-form) bytes; [scalar](../../../../glossary.md#term-scalar)/empty/static/dynamic/over-limit full [shapes](../../../../glossary.md#shape); square/wide/
tall/zero [identity](../../../../glossary.md#term-identity-meanings) [matrices](../../../../glossary.md#term-matrix-meanings); [row-major](../../../../glossary.md#term-row-major) diagonal positions; [alias](../../../../glossary.md#aliasing-and-arena) equivalence/fresh identity; dense
[metadata](../../../../glossary.md#term-metadata); empty [provenance](../../../../glossary.md#provenance); independence; null/negative/count/[gradient](../../../../glossary.md#term-gradient)/label validation and exact
messages; allocation/ID effects; exhaustion-state restoration; and preservation of completed
[constant](../../../../glossary.md#term-constant) behavior. Do not force [OOME](../../../../glossary.md#term-oome).

Manually inspect `javap -p -c -s` and source for exact overload [descriptors](../../../../glossary.md#term-descriptor), helper visibility,
fifteen package-private entries, exact [carrier](../../../../glossary.md#term-storage-carrier) arrays and `Arrays.fill`, default-zero identity
carriers, diagonal index arithmetic, one flat import, `eye -> identityMatrix` delegation, unchanged
allocator/import/constant [bytecode](../../../../glossary.md#term-bytecode), no provenance/[Operation](../../../../glossary.md#operation)/forbidden imports or state, and no new
package. Validate generated Javadocs, executable/concrete [API](../../../../glossary.md#term-application-programming-interface-api) example, links, anchors, fences,
whitespace, exact ten paths, synchronized status, and absence of a task-0014 specification.

## Dependencies

- Task 0012B supplies exact typed copied flat import, destination allocation, BOOL normalization,
  label validation, and ID behavior.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [normalization glossary](../../../../glossary.md#term-normalization); [validation glossary](../../../../glossary.md#term-validation))</span>
- Task 0012D supplies TensorConstants, canonical constant descriptor validation, exact scalar/one
  carrier practice, zeros allocation, and all-data-type one representation.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [constant glossary](../../../../glossary.md#term-constant); [descriptor glossary](../../../../glossary.md#term-descriptor); [scalar glossary](../../../../glossary.md#term-scalar); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Task 0013 confirms that [eager](../../../../glossary.md#term-eager-evaluation) public factory results remain [provenance](../../../../glossary.md#provenance)-free [leaves](../../../../glossary.md#term-leaf).

## Follow-up tasks

- After task 0013A, perform the model foundation checkpoint recorded in the roadmap. Decide whether
  task 0014 remains next or whether a named cross-module vertical slice should be planned.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [module glossary](../../../../glossary.md#term-module); [slice glossary](../../../../glossary.md#slice))</span>
- Task 0014 remains the Draft [elementwise](../../../../glossary.md#term-elementwise)-[operation](../../../../glossary.md#operation) frontier if sequential [model](../../../../glossary.md#term-model) work continues.
- Typed access/export, general mutable fill, identity Operations, batched/offset/sparse identities,
  compiler constants, and backend/runtime generation remain separate future work.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [identity glossary](../../../../glossary.md#term-identity-meanings); [compiler glossary](../../../../glossary.md#term-compiler); [constants glossary](../../../../glossary.md#term-constant); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime))</span>

Do not create detailed follow-up specifications in this task.

## Architecture impact

Expected impact: None. This is [model](../../../../glossary.md#term-model)-owned [eager](../../../../glossary.md#term-eager-evaluation) [leaf](../../../../glossary.md#term-leaf) construction through existing [descriptors](../../../../glossary.md#term-descriptor),
[heap](../../../../glossary.md#term-heap) storage, flat import, and [identity](../../../../glossary.md#term-identity-meanings) allocation. It changes no architecture rule, package/[module](../../../../glossary.md#term-module)
direction, [lifecycle](../../../../glossary.md#lifecycle) stage, [Operation](../../../../glossary.md#operation)/[provenance](../../../../glossary.md#provenance) semantics, or [runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend) boundary.

If implementation requires [Tensor](../../../../glossary.md#tensor) mutation, expression provenance, [compiler](../../../../glossary.md#term-compiler) behavior, another
module/package, new [dependency](../../../../glossary.md#term-dependency), or changed storage [ownership](../../../../glossary.md#term-ownership-meanings), stop and report the issue.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0001/0002/0003/0007/0011/0012/0012A/0012B/0012D/0013/0013A,
Tensor API, glossary, current TensorFactory/TensorConstants/descriptor/shape/layout/storage/
provenance source and all current constant/factory tests, and Java 26 Gradle configuration.

Implement task 0013A exactly. Modify only TensorFactory.java and TensorConstants.java for
production. Update TensorFactoryTest and TensorFactoryConstantTest only for exact public/helper
surface, and add TensorFactoryFullIdentityTest. Add exactly six public/package-private type-safe
full entries, one public/package-private identityMatrix entry, and one public-only eye alias.
Preserve every existing factory/helper method and test behavior.

Full methods require static Java-array-sized Shape, infer exact type from primitive carrier,
explicitly convert only fullBFloat16, fill one exact source carrier, and delegate once to matching
flat import. identityMatrix supports all six types and non-negative rectangular dimensions, writes
typed one on the main diagonal of one default-zero carrier, and delegates once. eye delegates only
to identityMatrix. Follow exact validation/messages, dense descriptor, gradient, ownership,
provenance-free leaf, allocation, label, and ID effects.

Do not add generic conversion/fullLike/default/square overloads, diagonal options, Tensor mutation,
access/export, new type/package/storage, provenance/Operation/graph/compiler/runtime/backend work,
dependencies/build/architecture changes, or later specs. Stop beyond ten paths or on architecture
uncertainty.

Run every specified focused/aggregate test, Javadoc, bytecode/import/manual, documentation example/
link/whitespace/scope/status check. Then hand the actual diff/evidence to a separate clean-context
documentation agent in the same change. It must inspect source/tests/generated Javadoc, finalize
permitted Javadocs/Tensor API/glossary/planning, record component/architecture/capability no-change
conclusions, and rerun validation.

Update task 0013A, model master plan, and roadmap only for planning status/evidence. Do not mark
0013A Complete until both passes succeed. Do not create task 0014. Do not commit or push.
```

## Local decisions

- Six primitive overloads keep fill semantics type-exact and avoid a generic scalar/conversion
  abstraction. BFLOAT16 conversion is named explicitly, matching existing scalar creation.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16))</span>
- `identityMatrix(rows, columns, ...)` is canonical and rectangular; a size-only overload would
  duplicate surface without adding capability. <span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>`eye` exists only as the requested exact [alias](../../../../glossary.md#aliasing-and-arena).
- Identity supports all six data types. BOOL uses true on the diagonal and false elsewhere; integer
  and BOOL gradient eligibility remains rejected by TensorDescriptor when requested.<span data-glossary-navigation="true"> ([Identity glossary](../../../../glossary.md#term-identity-meanings); [BOOL glossary](../../../../glossary.md#data-type--datatype); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Full always uses a filled source plus flat import, even for zero/one, so negative-zero/NaN bits
  and one deterministic side-effect path are preserved.<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan))</span>
- Identity uses one default-zero source and writes only diagonal ones; no general typed mutation or
  access API is introduced.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- No new helper/package is needed because TensorConstants already owns constant descriptor and
  carrier population mechanics.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [descriptor glossary](../../../../glossary.md#term-descriptor); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>

## Known limitations

- Full values are limited to exact primitive carriers and explicit semantic BFLOAT16 conversion;
  no arbitrary conversion or raw BFLOAT16 scalar overload exists.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Identity is eager, dense, rank two, main-diagonal only, and Java-array-sized. It has no batch,
  offset, sparse representation, or Operation form.<span data-glossary-navigation="true"> ([Identity glossary](../../../../glossary.md#term-identity-meanings); [eager glossary](../../../../glossary.md#term-eager-evaluation); [rank glossary](../../../../glossary.md#term-rank); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [Operation glossary](../../../../glossary.md#operation))</span>
- Full and identity allocate both a source carrier and copied destination carrier; zero-specific
  allocation optimization remains available only through <span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier))</span>`zeros`.
- Empty identity matrices may retain a very large extent on the other zero-product axis because
  Shape supports non-negative long dimensions; no physical elements are allocated.<span data-glossary-navigation="true"> ([matrices glossary](../../../../glossary.md#term-matrix-meanings); [extent glossary](../../../../glossary.md#term-extent); [axis glossary](../../../../glossary.md#term-axis); [Shape glossary](../../../../glossary.md#shape); [dimensions glossary](../../../../glossary.md#dimension); [elements glossary](../../../../glossary.md#term-element))</span>
- No typed read/export [API](../../../../glossary.md#term-application-programming-interface-api) is added; focused tests inspect existing [host](../../../../glossary.md#term-host) segments internally.

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) reviewed architecture/planning/documentation rules, current capability/master/roadmap
frontier, tasks 0012B/0012D/0013, TensorFactory/TensorConstants exact surfaces, [shape](../../../../glossary.md#shape)/count/[layout](../../../../glossary.md#layout)/
[descriptor](../../../../glossary.md#term-descriptor)/storage/import contracts, current [constant](../../../../glossary.md#term-constant) tests, [Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api)/glossary, and read-only
legacy TensorDataFactory evidence. Legacy contained no stable public full/[identity](../../../../glossary.md#term-identity-meanings)/eye contract;
the capability is an explicit current addition. The design reuses existing packages and fits ten
implementation/documentation paths with no architecture or [dependency](../../../../glossary.md#term-dependency) change.

- Implementation context `/root/implement_model_0013a` added exactly six public type-safe full
  methods, one public `identityMatrix`, one public-only `eye`, the seven matching package-private
  `TensorConstants` entries, and the focused test coverage. Clean documentation context
  `/root/implement_model_0013a/review_model_0013a_docs` independently inspected the actual final
  source, tests, diff, generated Javadocs, bytecode, API reference, glossary, and planning state.
  It applied General plus API/Javadoc style to Java/API/glossary work, Planning style to this task,
  the model master plan, and roadmap, and Example format to the new executable API example.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>
- The first combined focused-test invocation failed during test compilation because the updated
  helper-surface test reassigned a local set later captured by assertion lambdas. The test was
  corrected to initialize one mutable set without reassignment; all required focused commands
  below then passed independently in both implementation and final documentation validation.<span data-glossary-navigation="true"> ([compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [mutable glossary](../../../../glossary.md#term-mutable))</span>
- `./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryTest`
  — passed; XML reports 8 tests, zero
  failures, errors, or skips.
- `./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryConstantTest`
  — passed; XML reports 9 tests,
  zero failures, errors, or skips.
- `./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryFullIdentityTest`
  — passed; XML reports 8
  tests, zero failures, errors, or skips.
- `./gradlew :modules:model:test` — passed; 34 XML suites report 259 tests, zero failures, errors,
  or skips.
- `./gradlew :modules:model:javadoc` — passed without [Javadoc](../../../../glossary.md#term-javadoc) errors. Generated
  `TensorFactory.html` contains the six full overloads, `fullBFloat16`, `identityMatrix`, and
  `eye`, with exact type/value, static/scalar/empty/rectangular, dense-layout, label/gradient,
  ownership/provenance, failure-order, and alias contracts. Source review covers package-private<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [gradient glossary](../../../../glossary.md#term-gradient); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [provenance glossary](../../../../glossary.md#provenance); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
  `TensorConstants`, its seven entries, and the private count/import/typed-diagonal helpers that
  public generated Javadoc intentionally omits.
- `./gradlew test` — passed for the repository with 36 actionable tasks and no failing task.
- `javap -classpath modules/model/build/classes/java/main -p -c -s` inspection of
  `TensorFactory` and `TensorConstants` confirmed the exact overload descriptors and visibility,
  no additional field or nested type, exactly fifteen package-private constant entries, one exact
  typed source allocation plus `Arrays.fill` and one matching flat import for each full method,
  one default-zero exact identity carrier, row-major <span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [row-major glossary](../../../../glossary.md#term-row-major))</span>`i * columns + i` diagonal writes, and one
  matching flat import. `eye` [bytecode](../../../../glossary.md#term-bytecode) contains only unchanged [argument](../../../../glossary.md#term-argument) loads, one
  `identityMatrix` call, and `areturn`.
- Source, diff, import, and bytecode review confirmed existing allocator, allocation, flat-import,
  scalar/zero/one/like, provenance, label, and storage behavior remained unchanged. No new
  Operation, graph, compiler, planning, runtime, prepare, backend, device, service, cache,
  registry, or stateful dependency was introduced. Public initializer results remain leaves with
  empty provenance and independent metadata/storage/identity.<span data-glossary-navigation="true"> ([Operation glossary](../../../../glossary.md#operation); [graph glossary](../../../../glossary.md#graph); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend); [device glossary](../../../../glossary.md#term-device); [cache glossary](../../../../glossary.md#term-cache); [registry glossary](../../../../glossary.md#term-registry); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- The complete API example was compiled with `javac -cp
  modules/model/build/classes/java/main -d /private/tmp/synaptik-full-identity-example
  /private/tmp/FullIdentityExample.java` and run with
  the model classes. It printed <span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>`[-7, -7, -7, -7, -7, -7]`, the rectangular row-major identity
  carrier `[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]`, `Shape[2, 4]`,
  `DENSE_CONTIGUOUS`, and `true`, exactly as documented.
- A targeted local Markdown validator resolved 177 file targets and heading anchors across the
  five changed documentation/planning files with zero errors. Its first temporary invocation
  stopped before checking content because of Ruby interpolation syntax; the corrected compatible
  rerun is the passing result. Fence counts are balanced, changed files have no trailing
  whitespace, and <span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target))</span>`git diff --check` passed.
- Glossary review updated only the existing implementation-status, Tensor, and Tensor factory
  wording. No new reusable domain term was introduced: <span data-glossary-navigation="true"> ([Tensor factory glossary](../../../../glossary.md#tensor-factory))</span>`full`, `identityMatrix`, and `eye` are
  concrete factory operations whose behavior belongs in the Tensor API, not new architecture or
  lifecycle concepts requiring separate glossary entries.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Existing [`DataType`](../../../../glossary.md#data-type--datatype) and `BFloat16Bits` Javadocs remain accurate because exact type metadata and
  scalar BFLOAT16 conversion did not change. <span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16))</span>`Shape`, `LayoutDescriptor`, and `TensorDescriptor`
  remain accurate because the new paths consume their existing static-count, dense-geometry, and
  gradient-eligibility contracts without changing them. `Tensor` and `TensorProvenance` remain
  accurate because successful results are ordinary independent leaves with empty provenance and
  unchanged sole storage mutation. `HostTensorStorage` and `MemorySegmentStorage` remain accurate
  because storage sizing, borrowing, liveness, raw access, and ownership did not change. Existing
  allocation, flat-import, and constant Javadocs remain accurate; only the affected<span data-glossary-navigation="true"> ([borrowing glossary](../../../../glossary.md#term-borrowing); [liveness glossary](../../../../glossary.md#term-liveness))</span>
  `TensorFactory` and `TensorConstants` contracts required finalization.
- `ARCHITECTURE.md`, focused architecture explanations, ADRs, and architecture tests require no
  update because the change remains model-owned eager leaf creation and changes no architecture
  rule, package/module boundary, dependency direction, lifecycle stage, or backend/runtime
  responsibility. <span data-glossary-navigation="true"> ([leaf glossary](../../../../glossary.md#term-leaf))</span><span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [module glossary](../../../../glossary.md#term-module); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools))</span>`capabilities.md` already selected full and identity creation and required no
  implementation-status expansion. Compile API and Training API remain unchanged because no
  compiler capture/constant, Operation, autograd, gradient, trainable, optimizer, or session
  behavior was added. Backend-conformance and integration tests remain unchanged because no
  backend behavior or end-to-end execution path changed.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [capture glossary](../../../../glossary.md#term-capture-meanings); [autograd glossary](../../../../glossary.md#autograd); [optimizer glossary](../../../../glossary.md#term-optimizer); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Root and model Gradle review confirmed the existing Java 26 toolchain/release and no model
  override, preview/incubator feature, dependency, or build change. Final scope review found
  exactly the ten authorized repository paths: two production/Javadoc files, two existing exact-
  surface tests, one new focused test, Tensor API, glossary, this task, model master plan, and
  roadmap. No task-0014 specification exists.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [toolchain glossary](../../../../glossary.md#term-toolchain); [release glossary](../../../../glossary.md#term-release); [feature glossary](../../../../glossary.md#term-feature); [scope glossary](../../../../glossary.md#term-scope))</span>
- Final status synchronization marks 0013A Complete here, in the model master plan, and in the
  roadmap. The post-foundation checkpoint is the next planning action; task 0014 remains Draft and
  no implementation frontier is selected until that checkpoint records a decision.<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>

## Implementation notes

- Added exactly six type-safe public/package-private full-value entries, including explicit
  binary32-to-BFLOAT16 conversion, and kept one exact source-carrier fill plus one matching flat
  import for every data type.<span data-glossary-navigation="true"> ([binary32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [carrier glossary](../../../../glossary.md#term-storage-carrier); [data type glossary](../../../../glossary.md#data-type--datatype))</span>
- Added [canonical](../../../../glossary.md#term-canonical-form) all-data-type rectangular `identityMatrix` using one default-zero carrier and
  typed main-diagonal writes, plus public-only `eye` as one unchanged-[argument](../../../../glossary.md#term-argument) delegation.
- Added the focused 8-test full/identity suite and changed the two existing focused suites only for
  their exact public/helper surface expectations.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Finalized affected public, package-private, and private-helper Javadocs; documented current API
  behavior and a compiled concrete example; updated existing glossary terms; and synchronized the
  task, model master plan, and roadmap after validation.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [model glossary](../../../../glossary.md#term-model); [validation glossary](../../../../glossary.md#term-validation))</span>

## Completion summary

- Completed changes: Implemented and documented type-safe full-value tensors and dense rectangular
  identity matrices for all six current data types, with <span data-glossary-navigation="true"> ([tensors glossary](../../../../glossary.md#tensor); [identity glossary](../../../../glossary.md#term-identity-meanings); [matrices glossary](../../../../glossary.md#term-matrix-meanings))</span>`eye` exactly [aliasing](../../../../glossary.md#term-aliasing) [canonical](../../../../glossary.md#term-canonical-form)
  `identityMatrix` behavior.
- Files changed or created: The exact ten paths authorized under Affected files and Maximum scope;
  no other repository path changed.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>
- Tests and validation: Focused suites passed 8/8, 9/9, and 8/8; all 259 model tests across 34
  suites, model Javadoc, full repository tests, compiled example, bytecode/API/helper/import/state
  checks, generated-documentation review, 177 local link/anchor checks, fence/whitespace checks,
  exact-scope/status checks, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [model glossary](../../../../glossary.md#term-model); [bytecode glossary](../../../../glossary.md#term-bytecode); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>`git diff --check` passed.
- Documentation-agent review: Clean context
  `/root/implement_model_0013a/review_model_0013a_docs` completed the independent pass using
  General, API/Javadoc, Planning, and Example-format profiles.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: Tensor API and existing glossary/status language now describe current
  full-value and identity creation. Compile API, Training API, capability baseline, architecture
  documentation/tests, backend-conformance, integration documentation/tests, and build structure
  require no change for the reasons recorded above.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- [Javadoc](../../../../glossary.md#term-javadoc) review: `TensorFactory`, `TensorConstants`, all seven new helper entries, and all new
  private population helpers are final. Adjacent data type, BFLOAT16, shape, layout, descriptor,
  Tensor, provenance, storage, allocation, flat-import, and existing constant contracts remain
  accurate without edits.<span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [constant glossary](../../../../glossary.md#term-constant))</span>
- Glossary impact: Existing implementation-status, Tensor, and Tensor factory entries were
  updated; no new reusable project term was introduced.<span data-glossary-navigation="true"> ([Tensor factory glossary](../../../../glossary.md#tensor-factory))</span>
- Unresolved issues: None.
- Follow-up required: None for task 0013A. The model foundation checkpoint is the next planning
  action, while task 0014 remains Draft without a detailed specification.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>

Status: Complete
