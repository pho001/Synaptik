# Task 0012B: Flat Typed [Tensor](../../../../glossary.md#tensor) Import

**Editorial reading key (later context, not a status update).**

Here the application passes actual primitive-array values, not an expression description. fromFlatArray validates the supplied descriptor, allocates independent dense storage and copies the values; later changes to the source array do not change that imported storage. This copy boundary is specific to import: it must not be generalized to create or borrowed storage association. The result is a producerless leaf with populated heap memory, and its automatic lifetime does not require closing Tensor. BOOL normalization and BFLOAT16 raw-bit import have different meanings.

Focused current reading: [Current allocation/import contract](../../../../reference/tensor/creation.md); [Current flat-import example](../../../../reference/tensor/import.md#complete-flat-import-example). The original record below retains its own scope, decisions and evidence.


## Status

Complete

## Goal

Add a bounded public factory surface for creating dense-[contiguous](../../../../glossary.md#term-contiguous) [tensors](../../../../glossary.md#tensor) from copied flat Java
primitive arrays. Each overload accepts exactly the [carrier](../../../../glossary.md#term-storage-carrier) belonging to one [`DataType`](../../../../glossary.md#data-type--datatype), validates
the source length against the [descriptor](../../../../glossary.md#term-descriptor)'s logical [element](../../../../glossary.md#term-element) count, allocates storage through the
completed [JVM](../../../../glossary.md#term-java-virtual-machine-jvm)-managed [heap](../../../../glossary.md#term-heap) allocation path, and populates the new storage without retaining the
caller's array.

This task imports [leaf](../../../../glossary.md#term-leaf) data only. It does not infer [shapes](../../../../glossary.md#shape), synthesize [layouts](../../../../glossary.md#layout), import nested
arrays, expose backing arrays, add tensor element access, or define [view](../../../../glossary.md#term-view-meanings)/scatter population.

## [Scope](../../../../glossary.md#term-scope)

- Modify the existing `TensorFactory` and add six overloaded `fromFlatArray(...)` methods for
  `double[]`, `float[]`, raw [BFLOAT16](../../../../glossary.md#term-bfloat16) `short[]`, `int[]`, `long[]`, and [BOOL](../../../../glossary.md#data-type--datatype) `byte[]`.
- Require a completed [descriptor](../../../../glossary.md#term-descriptor) with resolved `DENSE_CONTIGUOUS` [layout](../../../../glossary.md#layout).
- Require the source length to equal `descriptor.shape().knownElementCount()` exactly.
- Require the source [carrier](../../../../glossary.md#term-storage-carrier)'s [`DataType`](../../../../glossary.md#data-type--datatype) to equal `descriptor.dataType()`.
- Allocate the destination through `TensorFactory.allocate(descriptor, label)` so task 0012A
  remains the single heap-allocation path and task 0012 remains the single ID-allocation path.<span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap))</span>
- Copy numeric and BFLOAT16 raw data into the destination segment without retaining the source.
- Normalize BOOL bytes during import: zero remains `0`; every non-zero value becomes [canonical](../../../../glossary.md#term-canonical-form)
  `1`.
- Preserve source encounter order as logical [row-major](../../../../glossary.md#term-row-major) [element](../../../../glossary.md#term-element) order.
- Update the existing exact-[shape](../../../../glossary.md#shape) `TensorFactoryTest` and add one focused
  `TensorFactoryFlatImportTest`.
- During implementation, update the Tensor API, glossary, task evidence, model master plan, and
  roadmap through the required separate clean-context documentation pass.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [model glossary](../../../../glossary.md#term-model))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- nested arrays, reflection-based shape inference, rectangularity checks, ragged arrays, or task
  0012C behavior<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [inference glossary](../../../../glossary.md#term-inference))</span>
- [scalar](../../../../glossary.md#term-scalar), zeros, ones, zeros-like, ones-like, fill, range, prefix, or random convenience methods
- offset-dense, strided, broadcast, unresolved, or dynamic-layout import; scatter semantics;
  alias-conflict rules; view creation; or materialization<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [alias glossary](../../../../glossary.md#aliasing-and-arena); [view glossary](../../../../glossary.md#term-view-meanings); [materialization glossary](../../../../glossary.md#materialization))</span>
- descriptor construction, shape inference, layout synthesis, default data types, or mutation of
  descriptor metadata<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- implicit numeric conversion, [widening](../../../../glossary.md#term-widening), [narrowing](../../../../glossary.md#term-narrowing), floating promotion, [BFLOAT16](../../../../glossary.md#term-bfloat16) conversion from
  `float`, BOOL-to-numeric conversion, or numeric-to-BOOL conversion other than normalization of
  the matching BOOL <span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [normalization glossary](../../../../glossary.md#term-normalization))</span>`byte[]` [carrier](../../../../glossary.md#term-storage-carrier)
- retaining a source array, exposing a backing array, zero-copy import, ownership transfer, or
  mutation/version tracking<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- public typed reads/writes, bulk export, copies from tensors, conversion methods, or a general
  storage-access API<span data-glossary-navigation="true"> ([tensors glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- a new storage type, native/[off-heap](../../../../glossary.md#term-off-heap-memory)/mapped allocation, [`Arena`](../../../../glossary.md#term-arena), close behavior, pooling, or
  backend/runtime storage<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- provenance, operations, graph IDs, gradients, trainable/publication state, compiler, planning,
  prepare, runtime, engine, backend, device, residency, or execution behavior<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [graph glossary](../../../../glossary.md#graph); [gradients glossary](../../../../glossary.md#term-gradient); [publication glossary](../../../../glossary.md#term-publication); [compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [engine glossary](../../../../glossary.md#term-engine); [device glossary](../../../../glossary.md#term-device); [residency glossary](../../../../glossary.md#residency))</span>
- dependencies, preview/incubator [features](../../../../glossary.md#term-software-feature), [Gradle](../../../../glossary.md#term-gradle) changes, architecture changes, or another [module](../../../../glossary.md#term-module)
- a detailed specification for task 0012C, task 0013, or any later task

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>, especially model ownership of public
  Tensor construction and host storage and the exclusion of runtime/backend storage<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend))</span>
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Architecture overview](../../../../architecture/overview.md)
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md), especially typed flat import and copy semantics
- [Model master plan](../master-plan.md)
- [Task 0001](0001-data-type-model.md), which defines the six data types
- [Task 0003](0003-layout-descriptor-model.md), which defines [layout](../../../../glossary.md#layout) kind and physical geometry
- [Task 0007](0007-tensor-descriptor-model.md), which defines logical [descriptor](../../../../glossary.md#term-descriptor) invariants
- [Task 0010](0010-host-storage-abstraction.md), which defines raw [host storage](../../../../glossary.md#host-storage)
- [Task 0011](0011-public-tensor-skeleton.md), which defines [tensor](../../../../glossary.md#tensor)/storage association
- [Task 0012](0012-tensor-factory.md), which defines public construction and [identity](../../../../glossary.md#term-identity-meanings) allocation
- [Task 0012A](0012a-host-storage-allocation.md), which defines typed [heap](../../../../glossary.md#term-heap) allocation
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span> and [glossary](../../../../glossary.md)

## Legacy evidence and rejected coupling

Read-only legacy inspection covered `TensorDataFactory`, `TensorStorageAccess`, typed [heap](../../../../glossary.md#term-heap) storage
classes, [Tensor](../../../../glossary.md#tensor) constructors, and flat/shaped data tests. Useful capability evidence is limited to
[carrier](../../../../glossary.md#term-storage-carrier)-specific input, [defensive copying](../../../../glossary.md#memory-management-atomic-counters-and-complexity), logical-count [validation](../../../../glossary.md#term-validation), [BFLOAT16](../../../../glossary.md#term-bfloat16) raw-bit storage, and
[canonical](../../../../glossary.md#term-canonical-form) [BOOL](../../../../glossary.md#data-type--datatype) storage.

The new design rejects the legacy constructor explosion, [mutable](../../../../glossary.md#term-mutable) [metadata](../../../../glossary.md#term-metadata), per-data-type storage
classes, nullable labels, reflection-based flattening, implicit floating conversions, raw mutable
array exposure, version counters in import code, [runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend) coupling, and legacy [scalar](../../../../glossary.md#term-scalar) or
empty-[shape](../../../../glossary.md#shape) conventions. Nested-array behavior remains task 0012C and is not copied here.

## Architecture constraints

- Production remains in `io.github.pho001.synaptik.model.tensor`; only `TensorFactory` changes.
- Package direction remains `model.tensor -> model.storage` plus existing foundational model
  packages. No reverse or cross-module dependency is introduced.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Import creates a new tensor and new heap storage. The caller's source array is never retained,
  used as the destination backing array, or exposed through the result.<span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap))</span>
- A flat source describes independent logical values in row-major order. Therefore the descriptor
  must have resolved <span data-glossary-navigation="true"> ([row-major glossary](../../../../glossary.md#term-row-major))</span>`LayoutKind.DENSE_CONTIGUOUS` geometry. Offset, sparse strided, and broadcast
  layouts require distinct scatter or alias semantics and are rejected.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Resolved dense-contiguous layout implies a fully static shape, zero offset, canonical strides,
  and referenced span equal to known logical element count, including scalar and empty shapes.<span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous); [shape glossary](../../../../glossary.md#shape); [strides glossary](../../../../glossary.md#term-stride); [referenced span glossary](../../../../glossary.md#referenced-element-span); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Source length is compared with the checked `long` logical element count before allocation. Java
  arrays have `int` length, so a [descriptor](../../../../glossary.md#term-descriptor) requiring more [elements](../../../../glossary.md#term-element) cannot pass this check.
- Numeric carriers and [BFLOAT16](../../../../glossary.md#term-bfloat16) `short[]` are copied bit-for-bit through temporary source
  [`MemorySegment`](../../../../glossary.md#term-memorysegment) values. BFLOAT16 shorts are raw format bits; this task performs no conversion.
- [BOOL](../../../../glossary.md#data-type--datatype) `byte[]` is logical input rather than raw unrestricted storage: `0` becomes canonical
  false byte `0`, and every non-zero byte becomes [canonical](../../../../glossary.md#term-canonical-form) true byte `1`.
- `TensorFactory.allocate(...)` remains the allocation path, and its delegation to `create(...)`
  remains the ID and Tensor-validation path. Flat import must not call <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>`nextTensorId()`, construct
  [`Tensor`](../../../../glossary.md#tensor), or construct `MemorySegmentStorage` directly.
- Population occurs before the new tensor is returned to the caller. The result is never publicly
  observable in a partially populated state.
- Input copying is not synchronized with concurrent caller mutation of the source. Callers must
  not mutate an array concurrently with import; no atomic snapshot guarantee is added.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- No existing storage, Tensor, descriptor, [layout](../../../../glossary.md#layout), or factory-allocation contract changes.
- If implementation requires general view scattering, conversion policy, another production
  type, a storage-contract change, native allocation, or an architecture/dependency change, stop
  and report the conflict.<span data-glossary-navigation="true"> ([view glossary](../../../../glossary.md#term-view-meanings))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.tensor` — owns `TensorFactory`, [`Tensor`](../../../../glossary.md#tensor), and [descriptors](../../../../glossary.md#term-descriptor).
- `io.github.pho001.synaptik.model.storage` — supplies the destination raw segment through the
  tensor's attached `HostTensorStorage`.
- `io.github.pho001.synaptik.model.datatype` — supplies [carrier](../../../../glossary.md#term-storage-carrier)-to-[`DataType`](../../../../glossary.md#data-type--datatype) matching.
- `io.github.pho001.synaptik.model.layout` — supplies `LayoutKind.DENSE_CONTIGUOUS`.

Packages added or changed:

- No package is added. Only the existing `model.tensor` package changes.

Type placement:

- `TensorFactory` gains the six import overloads and one private shared import method.
- `TensorFactoryFlatImportTest` is a focused same-package test.

## Required contract

### Public methods

Add exactly these six public static overloads:

```java
public static Tensor fromFlatArray(
        TensorDescriptor descriptor, Optional<String> label, double[] source)

public static Tensor fromFlatArray(
        TensorDescriptor descriptor, Optional<String> label, float[] source)

public static Tensor fromFlatArray(
        TensorDescriptor descriptor, Optional<String> label, short[] source)

public static Tensor fromFlatArray(
        TensorDescriptor descriptor, Optional<String> label, int[] source)

public static Tensor fromFlatArray(
        TensorDescriptor descriptor, Optional<String> label, long[] source)

public static Tensor fromFlatArray(
        TensorDescriptor descriptor, Optional<String> label, byte[] source)
```

The overload mapping is exact:

| Source [carrier](../../../../glossary.md#term-storage-carrier) | Required [`DataType`](../../../../glossary.md#data-type--datatype) | Meaning |
|---|---|---|
| `double[]` | [`FLOAT64`](../../../../glossary.md#term-float64) | [IEEE 754](../../../../glossary.md#term-ieee-754) [binary64](../../../../glossary.md#floating-point-precision-and-rounding) values |
| `float[]` | [`FLOAT32`](../../../../glossary.md#term-float32) | IEEE 754 [binary32](../../../../glossary.md#floating-point-precision-and-rounding) values |
| `short[]` | [`BFLOAT16`](../../../../glossary.md#term-bfloat16) | raw BFLOAT16 bit patterns |
| `int[]` | [`INT32`](../../../../glossary.md#data-type--datatype) | signed 32-bit integer values |
| `long[]` | [`INT64`](../../../../glossary.md#data-type--datatype) | signed 64-bit integer values |
| `byte[]` | [`BOOL`](../../../../glossary.md#data-type--datatype) | zero is false; non-zero is normalized to true byte `1` |

Do not add unlabeled convenience overloads, generic `Object`/array methods, conversion overloads,
varargs, nested arrays, or source-storage overloads.

### Private shared method

Add exactly one private static shared method:

```java
private static Tensor importFlat(
        TensorDescriptor descriptor,
        Optional<String> label,
        DataType sourceDataType,
        int sourceLength,
        MemorySegment sourceSegment)
```

Each public overload performs its three null checks, creates a temporary source segment using the
matching `MemorySegment.ofArray(...)` overload, and invokes `importFlat` with the exact [carrier](../../../../glossary.md#term-storage-carrier)
[`DataType`](../../../../glossary.md#data-type--datatype). The helper retains no [argument](../../../../glossary.md#term-argument) and adds no public [API](../../../../glossary.md#term-application-programming-interface-api).

### [Validation](../../../../glossary.md#term-validation) order and exact failures

Every public overload validates in this order:

1. Reject null [`descriptor`](../../../../glossary.md#term-descriptor) with `NullPointerException` and exact message `descriptor`.
2. Reject null `label` with `NullPointerException` and exact message `label`.
3. Reject null `source` with `NullPointerException` and exact message `source`.
4. Create the temporary source [`MemorySegment`](../../../../glossary.md#term-memorysegment) and delegate to `importFlat`.

The private method then validates in this order, before destination allocation or ID allocation:

1. If `descriptor.dataType()` differs from `sourceDataType`, throw `IllegalArgumentException`
   with exact message
   `flat source data type must match descriptor: expected=<sourceDataType>, actual=<descriptorDataType>`.
2. If `descriptor.layout()` is empty, throw `IllegalArgumentException` with exact message
   `flat tensor import requires a resolved layout`.
3. If the resolved kind is not `DENSE_CONTIGUOUS`, throw `IllegalArgumentException` with exact
   message `flat tensor import requires dense-contiguous layout: actual=<layoutKind>`.
4. Read `descriptor.shape().knownElementCount()`. The completed descriptor/layout invariant makes
   it present. If absent because an impossible inconsistent contract is encountered, throw<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [invariant glossary](../../../../glossary.md#term-invariant))</span>
   `IllegalStateException` with exact message
   `resolved tensor layout requires a fully static shape`.
5. Compare the required logical count with `sourceLength`. On mismatch throw
   `IllegalArgumentException` with exact message
   `flat source length must equal logical element count: required=<required>, actual=<actual>`.

These validation failures consume no [tensor](../../../../glossary.md#tensor) ID and allocate no destination storage. The temporary
source segment is only a non-owning [view](../../../../glossary.md#term-view-meanings) of the caller array and adds no copy or [lifetime](../../../../glossary.md#term-lifetime) change.

### Allocation and population

After [validation](../../../../glossary.md#term-validation):

1. Call `Tensor tensor = allocate(descriptor, label)` exactly once.
2. Obtain the exact attached destination segment through
   `tensor.hostStorage().orElseThrow().segment()`.
3. For [`FLOAT64`](../../../../glossary.md#term-float64), [`FLOAT32`](../../../../glossary.md#term-float32), [`BFLOAT16`](../../../../glossary.md#term-bfloat16), [`INT32`](../../../../glossary.md#data-type--datatype), and [`INT64`](../../../../glossary.md#data-type--datatype), copy exactly
   `sourceSegment.byteSize()` bytes into the destination at offset zero with [JDK](../../../../glossary.md#term-java-development-kit-jdk)
   `MemorySegment.copy(...)` or an equivalent segment-to-segment bulk copy.
4. For [`BOOL`](../../../../glossary.md#data-type--datatype), inspect each source byte and write [canonical](../../../../glossary.md#term-canonical-form) `0` or `1` into the corresponding
   destination byte. Use stable JDK `ValueLayout.JAVA_BYTE`; add no public typed-access [API](../../../../glossary.md#term-application-programming-interface-api).
5. Return the exact populated [tensor](../../../../glossary.md#tensor).

Dense-[contiguous](../../../../glossary.md#term-contiguous) geometry guarantees destination byte size equals source segment byte size. The
source segment and source array are not stored. Mutating the source after return cannot change the
tensor. Source order, floating bit patterns, integer values, and BFLOAT16 [raw bits](../../../../glossary.md#storage-mapping-contiguity-and-injectivity) are preserved.

### Failure side effects

- Null, data-type, layout, and source-length failures occur before destination allocation and
  before ID allocation; they consume no ID.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout))</span>
- A blank label passes import [validation](../../../../glossary.md#term-validation), then `allocate(...)` allocates storage and delegates to
  `create(...)`; Tensor rejects the label after ID allocation. The ID is consumed and no data copy
  occurs.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- Identifier exhaustion is observed inside `allocate(...)` after destination storage allocation
  and before import copying. The existing exhaustion exception propagates.
- [`OutOfMemoryError`](../../../../glossary.md#memory-management-atomic-counters-and-complexity) from destination allocation occurs before ID allocation and propagates.
- Bulk copying and BOOL normalization occur after successful tensor/ID creation. Their inputs and
  bounds are already validated; unexpected JDK errors propagate and the allocated ID remains
  consumed. No rollback is attempted.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [normalization glossary](../../../../glossary.md#term-normalization); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [rollback glossary](../../../../glossary.md#term-rollback))</span>

## Valid and invalid scenarios

| Scenario | Result |
|---|---|
| [Scalar](../../../../glossary.md#term-scalar) [`FLOAT64`](../../../../glossary.md#term-float64), one `double` | Valid copied scalar |
| Empty dense [shape](../../../../glossary.md#shape), empty matching array | Valid zero-capacity [tensor](../../../../glossary.md#tensor) |
| Dense `[2,3]`, six matching [carrier](../../../../glossary.md#term-storage-carrier) [elements](../../../../glossary.md#term-element) | Valid [row-major](../../../../glossary.md#term-row-major) copy |
| [BOOL](../../../../glossary.md#data-type--datatype) bytes `{0, -2, 3}` | Destination bytes `{0, 1, 1}` |
| [BFLOAT16](../../../../glossary.md#term-bfloat16) `short[]` | [Raw bits](../../../../glossary.md#storage-mapping-contiguity-and-injectivity) copied unchanged |
| Source mutated after return | Tensor data unchanged |
| Matching length but wrong carrier for [descriptor](../../../../glossary.md#term-descriptor) | Data-type mismatch |
| Unresolved [layout](../../../../glossary.md#layout) | Rejected before allocation |
| Offset-dense, strided, or broadcast layout | Rejected before allocation |
| Too short or too long source | Logical-count mismatch |
| Blank label with otherwise valid data | Delegated failure consumes one ID |

## Affected files

Expected production update:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorFactory.java`

Expected test updates:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryFlatImportTest.java`

Expected documentation and [planning](../../../../glossary.md#term-planning) updates during implementation:

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task specification
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most one production file, one existing test, one new focused
test, and the five documentation/[planning](../../../../glossary.md#term-planning) files above: eight paths total.

Do not modify another production/test file, [Gradle](../../../../glossary.md#term-gradle), `AGENTS.md`, `ARCHITECTURE.md`, focused
architecture documentation, architecture tests, `capabilities.md`, another [module](../../../../glossary.md#term-module), or unrelated
documentation. Do not create a task-0012C or task-0013 specification. Stop if another file, public
type, conversion policy, [layout](../../../../glossary.md#layout)-scatter rule, or architecture clarification is required.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Update `TensorFactory` type Javadoc to describe copied flat import while preserving factory ID,
  allocation, ownership, and deferred-capability boundaries.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Document every overload fully: exact [carrier](../../../../glossary.md#term-storage-carrier), required [`DataType`](../../../../glossary.md#data-type--datatype), dense-contiguous/layout and
  logical-count preconditions, label behavior, copying, BOOL normalization or BFLOAT16 raw bits,
  return value, side effects, and every failure.<span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous); [layout glossary](../../../../glossary.md#layout); [BOOL glossary](../../../../glossary.md#data-type--datatype); [normalization glossary](../../../../glossary.md#term-normalization); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [raw bits glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity))</span>
- Document the private shared method, including why it validates logical row-major import and why
  only BOOL receives normalization.<span data-glossary-navigation="true"> ([row-major glossary](../../../../glossary.md#term-row-major))</span>
- Do not describe numeric conversion, zero-copy ownership, general view population, nested import,
  typed tensor access, or factory parity as implemented.<span data-glossary-navigation="true"> ([view glossary](../../../../glossary.md#term-view-meanings); [tensor glossary](../../../../glossary.md#tensor))</span>
- Record-generated or serialization claims are inapplicable; Tensor and factory identity behavior
  remains unchanged.<span data-glossary-navigation="true"> ([serialization glossary](../../../../glossary.md#term-serialization); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>

## Acceptance criteria

- `TensorFactory` adds exactly the six public overloads and one private shared method specified.
- Existing create/allocate methods, fields, ID allocator, and behavior remain unchanged.
- All six carriers require the exact matching [descriptor](../../../../glossary.md#term-descriptor) [data type](../../../../glossary.md#data-type--datatype).
- Import requires resolved `DENSE_CONTIGUOUS` [layout](../../../../glossary.md#layout) and exact logical source length.
- Numeric and [BFLOAT16](../../../../glossary.md#term-bfloat16) raw values are copied exactly; [BOOL](../../../../glossary.md#data-type--datatype) values are canonicalized to `0`/`1`.
- Source arrays are not retained and later source mutation cannot affect [tensor](../../../../glossary.md#tensor) storage.
- Scalar and empty imports work; offset, strided, broadcast, dynamic/unresolved, and length-mismatch
  inputs fail with exact messages before destination/ID allocation.<span data-glossary-navigation="true"> ([Scalar glossary](../../../../glossary.md#term-scalar))</span>
- Import calls `allocate(...)`, never `nextTensorId()`, `new Tensor`, or `new MemorySegmentStorage`.
- Existing allocator/[concurrency](../../../../glossary.md#term-concurrency)/exhaustion and allocation tests are not weakened.
- Complete Javadoc, Tensor API, glossary, task/master/roadmap status, and separate documentation
  review are finished in the same change.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Task 0012B is `Complete` only after all [validation](../../../../glossary.md#term-validation); task 0012C remains `Draft` without a spec.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) before and after the documentation pass:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryFlatImportTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

Manually verify reflection and `javap -p -c -s` show the exact six overloads, one private helper,
unchanged allocator, validation order, allocation delegation, segment copy/[BOOL](../../../../glossary.md#data-type--datatype) [normalization](../../../../glossary.md#term-normalization), and
no direct [Tensor](../../../../glossary.md#tensor)/storage construction. Verify production imports contain only current [model](../../../../glossary.md#term-model)/[JDK](../../../../glossary.md#term-java-development-kit-jdk)
types and no reflection, arrays utility retention, [graph](../../../../glossary.md#graph)/[runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend) types, [Arena](../../../../glossary.md#term-arena), or native
access. Verify exact eight-path [scope](../../../../glossary.md#term-scope), links/anchors, fences, whitespace, status [synchronization](../../../../glossary.md#term-synchronization),
and absence of task-0012C/task-0013 specs.

## Dependencies

- Task 0012A provides dense [heap](../../../../glossary.md#term-heap) allocation and is complete.
- Task 0012 provides [identity](../../../../glossary.md#term-identity-meanings) allocation and public construction and is complete.
- Tasks 0001, 0003, 0007, 0010, and 0011 provide data type, layout, descriptor, storage, and Tensor
  contracts.<span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype); [layout glossary](../../../../glossary.md#layout); [descriptor glossary](../../../../glossary.md#term-descriptor); [Tensor glossary](../../../../glossary.md#tensor))</span>

## Follow-up tasks

- Task 0012C will define nested typed import and rectangular [shape](../../../../glossary.md#shape) [inference](../../../../glossary.md#term-inference).
- Task 0012D will define [scalar](../../../../glossary.md#term-scalar) and [constant](../../../../glossary.md#term-constant) conveniences.
- Task 0012E will define ranges and prefix population.
- Task 0012F will define random creation and reproducibility policy.
- Later work owns typed export/access, mutation versions, [views](../../../../glossary.md#term-view-meanings)/scatter, [provenance](../../../../glossary.md#provenance), and operations.

Do not create a detailed task-0012C specification as part of task 0012B.

## Architecture impact

Expected impact: None.

The architecture already assigns TensorFactory, [host storage](../../../../glossary.md#host-storage), and public [tensor](../../../../glossary.md#tensor) state to
`modules/model`. This task adds copied dense [leaf](../../../../glossary.md#term-leaf) data only and changes no [module](../../../../glossary.md#term-module) boundary,
[dependency direction](../../../../glossary.md#java-platform-and-build-tools), [lifecycle](../../../../glossary.md#lifecycle) stage, storage [ownership](../../../../glossary.md#term-ownership-meanings), or [runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend) rule. If
implementation reveals otherwise, stop before editing architecture files.

## Implementation prompt

Use this prompt in a separate agentic task/thread with a clean context:

```text
You are a clean-context implementation agent working in the Synaptik repository.

Read AGENTS.md, ARCHITECTURE.md, docs/developer-guide/documentation-rules.md,
docs/planning/planning-guide.md, docs/planning/modules/model/capabilities.md, the model master plan,
roadmap, tasks 0001/0003/0007/0010/0011/0012/0012A/0012B, Tensor API, glossary, and the current
TensorFactory/Tensor/descriptor/layout/storage source and tests in full.

Implement task 0012B exactly as specified. Modify TensorFactory.java, update TensorFactoryTest only
for exact API shape, and add TensorFactoryFlatImportTest. Add exactly six fromFlatArray overloads
and the one private importFlat method from the task. Preserve every existing factory method,
allocator, allocation path, and test.

Require exact carrier/DataType matching, resolved DENSE_CONTIGUOUS layout, and source length equal
to known logical element count. Delegate destination allocation to allocate(descriptor, label).
Copy numeric and raw BFLOAT16 data without retaining source arrays; normalize BOOL bytes so zero
remains zero and every non-zero value becomes one. Follow exact validation order and messages.

Do not add conversion, nested import, descriptor/layout synthesis, general view/scatter population,
typed tensor access/export, storage types, Arena/native allocation, provenance, operations,
gradients, graph/compiler/runtime/backend behavior, dependencies, build changes, or follow-up specs.
Stop if work exceeds the eight permitted paths or requires an architecture decision.

Run every focused/aggregate test, Javadoc, bytecode/import/manual, documentation, link, whitespace,
scope, and status check in the task.

After initial implementation validation, spawn a separate clean-context documentation-focused
agent in the same change. Hand it the task, actual diff, carrier/layout/count/copy/normalization and
ID-side-effect behavior, architecture constraints, Tensor API/glossary/Javadoc impact, and all
validation commands. It must inspect source/tests/evidence, finalize permitted documentation and
Javadocs, record no-change conclusions for architecture and existing component contracts, and
synchronize task/master/roadmap status.

Do not mark 0012B Complete until both passes and all validation succeed. Then leave 0012C Draft
without a detailed specification. Do not commit or push.
```

## Local decisions

- Flat import is dense-contiguous only. Supporting offset/strided/broadcast descriptors would
  require sparse scatter and alias-conflict semantics, which are separate from copying a logical
  row-major leaf array.<span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous); [descriptors glossary](../../../../glossary.md#term-descriptor); [alias glossary](../../../../glossary.md#aliasing-and-arena); [row-major glossary](../../../../glossary.md#term-row-major); [leaf glossary](../../../../glossary.md#term-leaf))</span>
- Six overloads preserve [compile](../../../../glossary.md#compile)-time [carrier](../../../../glossary.md#term-storage-carrier) typing without an `Object` API or per-dtype storage
  hierarchy. Optional label remains explicit; convenience overload proliferation is deferred.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [dtype glossary](../../../../glossary.md#data-type--datatype))</span>
- [BFLOAT16](../../../../glossary.md#term-bfloat16) input is raw `short` bits. Converting `float[]` to BFLOAT16 is a conversion capability,
  not typed carrier matching.
- BOOL follows legacy capability evidence but stores a canonical representation: zero is false and
  any non-zero byte imports as one.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Allocation occurs through task 0012A, so storage [lifetime](../../../../glossary.md#term-lifetime) and ID side effects stay centralized.
- Copying after successful allocation means unexpected copy failure consumes the already allocated
  ID; validated bounds make such failure outside ordinary input behavior.

## Known limitations

- Only resolved dense-[contiguous](../../../../glossary.md#term-contiguous) [descriptors](../../../../glossary.md#term-descriptor) are accepted.
- No unlabeled convenience overloads are provided; callers pass `Optional.empty()`.
- Import is not [atomic](../../../../glossary.md#resource-transaction-and-rollback) with concurrent mutation of the source array.
- There is no numeric conversion, nested shape inference, typed export, public element access, or
  mutation/version tracking.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [inference glossary](../../../../glossary.md#term-inference); [element glossary](../../../../glossary.md#term-element))</span>
- Very large logical counts cannot be represented by a Java source array and fail length
  validation before allocation.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) reviewed the authoritative architecture, planning/documentation rules, capability
baseline, completed factory/storage/[descriptor](../../../../glossary.md#term-descriptor)/[layout](../../../../glossary.md#layout) contracts, current tests, and legacy factory
evidence. The resulting design changes only [model](../../../../glossary.md#term-model) planning documents and introduces no architecture
decision.

- The implementation changed only `TensorFactory.java`, updated the existing exact-[shape](../../../../glossary.md#shape)
  `TensorFactoryTest`, and added `TensorFactoryFlatImportTest`. Clean documentation-focused context
  `/root/implement_model_0012b/review_model_0012b_docs` independently read the architecture,
  documentation workflow, General style, API and Javadoc style, Planning style, Example format,
  planning chain, completed prerequisite tasks, final source/tests, generated Javadoc, complete
  diff, Tensor API, and glossary before finalizing documentation.
- The documentation pass applied General plus [API](../../../../glossary.md#term-application-programming-interface-api) and [Javadoc](../../../../glossary.md#term-javadoc) style to `TensorFactory` and the
  Tensor API, Planning style to this task/master/roadmap, and Example format to the new complete
  flat-import example. It changed only Javadoc in `TensorFactory.java` among Java sources; method
  signatures, implementation behavior, and tests were not altered.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature))</span>
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorFactoryTest` passed after documentation edits. XML
  reports 7 tests, 0 failures, 0 errors, and 0 skipped.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorFactoryFlatImportTest` passed after documentation
  edits. XML reports 10 tests, 0 failures, 0 errors, and 0 skipped, including the 2-by-3 INT32
  copy and identifier-exhaustion-before-population cases.<span data-glossary-navigation="true"> ([INT32 glossary](../../../../glossary.md#data-type--datatype))</span>
- `./gradlew :modules:model:test` passed. Aggregating 25 XML suites reports 177 tests, 0 failures,
  0 errors, and 0 skipped.
- `./gradlew :modules:model:javadoc` passed without warnings. Generated `TensorFactory.html`
  contains exactly six rendered flat-import overloads and documents exact carriers, dense layout
  and count requirements, source copying, raw BFLOAT16 bits, BOOL normalization, label/ID/allocation
  side effects, results, failures, and deferred boundaries. Source review confirms the private
  helper contract, which public generated Javadoc intentionally omits.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- `./gradlew test` passed for the complete repository with 36 actionable tasks and no failing task.
- `javap -classpath modules/model/build/classes/java/main -p -c -s
  io.github.pho001.synaptik.model.tensor.TensorFactory` confirmed the six exact public overloads,
  one private `importFlat` method, unchanged allocator, public null-check order, helper validation
  order/messages, exactly one `allocate(...)` call, attached-segment retrieval, numeric and raw
  BFLOAT16 `MemorySegment.copy`, [BOOL](../../../../glossary.md#data-type--datatype) `ValueLayout.JAVA_BYTE` normalization, and no lambda helper,
  direct Tensor construction, storage construction, or ID allocation inside `importFlat`.
- Production imports are exactly [`DataType`](../../../../glossary.md#data-type--datatype), `LayoutKind`, existing storage contracts,
  [`MemorySegment`](../../../../glossary.md#term-memorysegment), `ValueLayout`, `Objects`, `Optional`, and the two existing atomics. Source and
  bytecode inspection found no reflection, array retention, conversion, nested import, typed
  Tensor access, scatter/view policy, native allocation, graph, compiler, runtime, backend, or
  service state.<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode); [view glossary](../../../../glossary.md#term-view-meanings); [graph glossary](../../../../glossary.md#graph); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend))</span>
- The complete flat-import example in the [Tensor](../../../../glossary.md#tensor) API compiled with `javac -cp
  modules/model/build/classes/java/main` and ran with the model classes. It printed [`mask`](../../../../glossary.md#term-boolean-mask) and
  `[0, 1, 1]`, confirming the documented normalization and post-return source independence. Two
  preliminary JShell runs produced the same values but failed while flushing host macOS
  preferences; the successful compile-and-run removed that environmental limitation from example
  validation.<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [compile glossary](../../../../glossary.md#compile); [run glossary](../../../../glossary.md#term-run))</span>
- A targeted Ruby validator resolved 144 local Markdown links, including 58 heading anchors,
  across the five changed documentation/planning files. Two preliminary validator attempts failed
  before checking content because of Ruby interpolation and the host Ruby's missing
  `Array#filter_map`; the compatible rerun passed with zero errors. Fence counts are balanced,
  trailing-whitespace checks found no matches, and `git diff --check` passed.
- Existing `Tensor`, `TensorDescriptor`, `HostTensorStorage`, `MemorySegmentStorage`, `DataType`,
  `Shape`, and `LayoutDescriptor` Javadocs remain accurate without edits. They already define the
  stable tensor association, immutable logical descriptor, raw borrowed storage, exact segment
  wrapper, carrier-independent data type, logical shape/count, and resolved geometry/span
  contracts consumed by import; task 0012B changes none of those ownership or behavior contracts.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [carrier glossary](../../../../glossary.md#term-storage-carrier); [data type glossary](../../../../glossary.md#data-type--datatype); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- `ARCHITECTURE.md`, focused architecture documentation, ADRs, architecture tests,
  backend-conformance tests, integration tests, and <span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance))</span>`capabilities.md` remain unchanged. Flat import
  is an already authorized model-owned TensorFactory/host-storage capability, changes no module or
  package direction, and introduces no backend or end-to-end execution behavior.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>
- The Tensor API now treats only copied flat typed import as current while preserving nested,
  constant, range/prefix, random, typed access/export, conversion, scatter/view, native, runtime,
  and backend behavior as planned. The glossary updates its existing Tensor and TensorFactory
  definitions; no new reusable domain term was introduced, so a separate flat-import entry would
  duplicate the API contract.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant))</span>
- Final [scope](../../../../glossary.md#term-scope) review found exactly the eight permitted paths: `TensorFactory.java`,
  `TensorFactoryTest.java`, `TensorFactoryFlatImportTest.java`, Tensor API, glossary, this task,
  model master plan, and roadmap. No task-0012C or task-0013 specification exists. Task 0012B is
  synchronized as `Complete`; task 0012C remains the next `Draft` frontier. No commit or push was
  performed.<span data-glossary-navigation="true"> ([commit glossary](../../../../glossary.md#term-commit))</span>

## Implementation notes

- Added the six exact [carrier](../../../../glossary.md#term-storage-carrier)-specific `fromFlatArray(...)` overloads and one private shared
  `importFlat(...)` path to `TensorFactory` while preserving existing creation, allocation, and ID
  behavior.
- Added exact preallocation validation for carrier data type, resolved dense-contiguous layout,
  and logical source count. Numeric values and raw BFLOAT16 bits use segment bulk copy; BOOL bytes
  are normalized to canonical zero or one after allocation.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [data type glossary](../../../../glossary.md#data-type--datatype); [contiguous glossary](../../../../glossary.md#term-contiguous); [layout glossary](../../../../glossary.md#layout); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [BOOL glossary](../../../../glossary.md#data-type--datatype); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Added focused coverage for all carriers, source isolation, BOOL normalization, scalar/empty
  values, exact failures and ID side effects, exhaustion before population, and unchanged gradient
  eligibility. The existing factory suite changed only its exact API-shape expectation.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [scalar glossary](../../../../glossary.md#term-scalar); [gradient glossary](../../../../glossary.md#term-gradient); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [shape glossary](../../../../glossary.md#shape))</span>
- Finalized TensorFactory Javadoc, the Tensor API and compiled example, glossary status, and
  task/master/roadmap evidence through the required independent documentation pass.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [Tensor glossary](../../../../glossary.md#tensor))</span>

## Completion summary

- Completed changes: Implemented and documented copied flat typed tensor import for all six model
  data types with exact carrier, dense-layout, logical-count, ownership, normalization, and failure
  side-effect contracts.<span data-glossary-navigation="true"> ([tensor glossary](../../../../glossary.md#tensor); [model glossary](../../../../glossary.md#term-model); [carrier glossary](../../../../glossary.md#term-storage-carrier); [layout glossary](../../../../glossary.md#layout); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- Files changed or created: `TensorFactory.java`, `TensorFactoryTest.java`,
  `TensorFactoryFlatImportTest.java`, `docs/api/tensor-api.md`, `docs/glossary.md`, this task, the
  model master plan, and the roadmap.
- Tests and validation: Focused suites passed 7/7 and 10/10; all 177 model tests across 25 suites,
  model Javadoc, and full repository tests passed. Bytecode/API shape, imports and forbidden
  references, generated Javadoc, compiled example, links/anchors, fences, whitespace, exact scope,
  status synchronization, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Bytecode glossary](../../../../glossary.md#term-bytecode); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [shape glossary](../../../../glossary.md#shape); [scope glossary](../../../../glossary.md#term-scope); [synchronization glossary](../../../../glossary.md#term-synchronization))</span>`git diff --check` passed.
- Documentation-agent review: Complete in [canonical](../../../../glossary.md#term-canonical-form) clean context
  `/root/implement_model_0012b/review_model_0012b_docs` using General, API/Javadoc, Planning, and
  Example-format profiles.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: Copied flat typed import is current; nested/constants/ranges/prefixes/
  random/access/export/conversion/view-scatter/native/runtime/backend behavior remains planned.<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant); [view glossary](../../../../glossary.md#term-view-meanings); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend))</span>
- [Javadoc](../../../../glossary.md#term-javadoc) review: `TensorFactory` type, six overloads, and private helper are final. Existing
  Tensor, descriptor, storage, data type, shape, and layout Javadocs remain accurate for the
  reasons recorded above.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [data type glossary](../../../../glossary.md#data-type--datatype))</span>
- Glossary impact: Existing Tensor and TensorFactory status/definitions now include flat import;
  no new term was needed because the behavior is a factory operation, not a new domain concept.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- Architecture impact: None. No architecture document, ADR, architecture test, dependency rule,
  module boundary, package direction, or capability-baseline decision changed.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [architecture test glossary](../../../../glossary.md#term-architecture-test); [dependency glossary](../../../../glossary.md#term-dependency); [module glossary](../../../../glossary.md#term-module))</span>
- Unresolved issues: None.
- Follow-up required: None for task 0012B. Task 0012C remains the next separate `Draft` frontier
  without a detailed specification.

Status: Complete
