# Task 0012C: Nested Typed [Tensor](../../../../glossary.md#tensor) Import

## Status

Complete

## Goal

Add one bounded public factory path that accepts a rectangular multidimensional Java primitive
array, infers its fully static [shape](../../../../glossary.md#shape) and exact [model](../../../../glossary.md#term-model) [data type](../../../../glossary.md#data-type--datatype), flattens its values in logical
[row-major](../../../../glossary.md#term-row-major) order, and delegates final [tensor](../../../../glossary.md#tensor) creation to the completed flat typed import path.

This task provides readable nested input without weakening [carrier](../../../../glossary.md#term-storage-carrier) typing. It accepts only true
primitive arrays with [rank](../../../../glossary.md#term-rank) at least two, preserves raw [BFLOAT16](../../../../glossary.md#term-bfloat16) bits, canonicalizes [BOOL](../../../../glossary.md#data-type--datatype) through
task 0012B, and never retains or mutates the caller's nested array [graph](../../../../glossary.md#graph).

## [Scope](../../../../glossary.md#term-scope)

- Add exactly one public `TensorFactory.fromNestedArray(...)` method accepting a source object,
  optional label, and `requiresGrad` flag.
- Support multidimensional arrays whose ultimate primitive component is exactly `double`, `float`,
  `short`, `int`, `long`, or `byte`.
- Map those carriers exactly to [`FLOAT64`](../../../../glossary.md#term-float64), [`FLOAT32`](../../../../glossary.md#term-float32), [`BFLOAT16`](../../../../glossary.md#term-bfloat16), [`INT32`](../../../../glossary.md#data-type--datatype), [`INT64`](../../../../glossary.md#data-type--datatype), and [`BOOL`](../../../../glossary.md#data-type--datatype).
- Require runtime array rank of at least two. Rank-one primitive arrays remain the task-0012B flat
  import surface.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [rank glossary](../../../../glossary.md#term-rank); [Rank-one glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- Infer one static [axis](../../../../glossary.md#term-axis) size per observable nested level and require a rectangular structure.
- Reject null subarrays, ragged lengths, unsupported leaf carriers, generic/object arrays, and
  empty non-leaf axes whose trailing extents cannot be observed.
- Accept an empty final [leaf](../../../../glossary.md#term-leaf) axis when all preceding axes are observable, such as `new int[2][0]`.
- Flatten values into a new matching primitive array in [row-major](../../../../glossary.md#term-row-major) encounter order.
- Construct a resolved dense-[contiguous](../../../../glossary.md#term-contiguous) `TensorDescriptor` from the inferred type, [shape](../../../../glossary.md#shape), and
  `requiresGrad` value.
- Delegate the matching flat primitive array to the existing `fromFlatArray(...)` overload so
  allocation, ID assignment, copying, BOOL normalization, and label behavior remain centralized.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization))</span>
- Add one package-private same-package helper for structural [validation](../../../../glossary.md#term-validation), [inference](../../../../glossary.md#term-inference), and flattening.
- Update the exact `TensorFactory` [API](../../../../glossary.md#term-application-programming-interface-api)-shape test and add one focused nested-import test suite.
- During implementation, update Tensor API, glossary, task evidence, model master plan, and roadmap
  through the required separate clean-context documentation pass.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [model glossary](../../../../glossary.md#term-model))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- rank-one arrays, scalar values, boxed arrays, collections, streams, buffers, memory segments, or
  generic iterables<span data-glossary-navigation="true"> ([rank-one glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [buffers glossary](../../../../glossary.md#term-buffer))</span>
- [`boolean`](../../../../glossary.md#data-type--datatype)/`boolean[]` carriers; [model](../../../../glossary.md#term-model) [BOOL](../../../../glossary.md#data-type--datatype) [host](../../../../glossary.md#term-host) interchange remains normalized `byte`
- [heterogeneous](../../../../glossary.md#term-heterogeneous) `Object[]`, arrays whose ultimate component is `Object` or a wrapper class, mixed
  leaf carriers, implicit conversion, numeric promotion, or default data-type selection<span data-glossary-navigation="true"> ([numeric promotion glossary](../../../../glossary.md#numeric-promotion))</span>
- caller-supplied shape, descriptor, layout, or data type; dynamic shape inference; symbolic
  dimensions; reshape; broadcasting; offset, strided, or view layout population<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [descriptor glossary](../../../../glossary.md#term-descriptor); [layout glossary](../../../../glossary.md#layout); [data type glossary](../../../../glossary.md#data-type--datatype); [inference glossary](../../../../glossary.md#term-inference); [dimensions glossary](../../../../glossary.md#dimension); [broadcasting glossary](../../../../glossary.md#broadcasting); [view glossary](../../../../glossary.md#term-view-meanings))</span>
- inferring unobservable trailing extents after an empty non-[leaf](../../../../glossary.md#term-leaf) [axis](../../../../glossary.md#term-axis)
- accepting or repairing ragged structures, padding rows, truncating rows, cyclic fill, prefix
  fill, or scatter into arbitrary geometry<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding))</span>
- zero-copy import, retaining source arrays, exposing the intermediate flat array, typed Tensor
  access/export, mutation/version tracking, or concurrent-source snapshot guarantees<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- task 0012D [constants](../../../../glossary.md#term-constant), [scalars](../../../../glossary.md#term-scalar), zeros, ones, zeros-like, or ones-like behavior
- task 0012E ranges or strict/cyclic prefix population
- task 0012F random generation or [reproducibility](../../../../glossary.md#artifact-manifest-and-source-provenance) policy
- native/[off-heap](../../../../glossary.md#term-off-heap-memory)/mapped allocation, [`Arena`](../../../../glossary.md#term-arena), [ownership](../../../../glossary.md#term-ownership-meanings)/close behavior, [pooling](../../../../glossary.md#term-pooling), or new storage types
- provenance, expression operations, graph IDs, gradients beyond descriptor eligibility,
  publication state, compiler, planning, prepare, runtime, engine, backend, device, residency, or
  execution behavior<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [graph glossary](../../../../glossary.md#graph); [gradients glossary](../../../../glossary.md#term-gradient); [publication glossary](../../../../glossary.md#term-publication); [compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [device glossary](../../../../glossary.md#term-device); [residency glossary](../../../../glossary.md#residency))</span>
- dependencies, preview/incubator features, Gradle changes, architecture changes, another module,
  or a detailed specification for task 0012D or any later task<span data-glossary-navigation="true"> ([features glossary](../../../../glossary.md#term-software-feature); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>, especially model ownership of public
  Tensor construction and host storage and the exclusion of runtime/backend storage<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Tensor glossary](../../../../glossary.md#tensor); [host storage glossary](../../../../glossary.md#host-storage); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend))</span>
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Architecture overview](../../../../architecture/overview.md)
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md), especially supported nested Java arrays and the
  six host carrier mappings<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host))</span>
- [Model master plan](../master-plan.md)
- [Task 0001](0001-data-type-model.md), which defines the six data types and differentiability
- [Task 0002](0002-shape-and-dimension-model.md), which defines static [shapes](../../../../glossary.md#shape) and checked counts
- [Task 0003](0003-layout-descriptor-model.md), which defines dense-[contiguous](../../../../glossary.md#term-contiguous) geometry
- [Task 0007](0007-tensor-descriptor-model.md), which defines [descriptor](../../../../glossary.md#term-descriptor) [validation](../../../../glossary.md#term-validation)
- [Task 0012](0012-tensor-factory.md), which defines public construction and [identity](../../../../glossary.md#term-identity-meanings) allocation
- [Task 0012A](0012a-host-storage-allocation.md), which defines [JVM](../../../../glossary.md#term-java-virtual-machine-jvm) [heap](../../../../glossary.md#term-heap) allocation
- [Task 0012B](0012b-flat-typed-tensor-import.md), which defines [carrier](../../../../glossary.md#term-storage-carrier)-specific flat import
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span> and [glossary](../../../../glossary.md)

## Legacy evidence and rejected coupling

Read-only legacy inspection covered [`Tensor`](../../../../glossary.md#tensor), `TensorArrayData`, `TensorDataFactory`,
`TensorStorageAccess`, `TensorConstructorDataTypeTest`, and the legacy Tensor [API](../../../../glossary.md#term-application-programming-interface-api)/README. The useful
capability is ergonomic construction from an arbitrary-[rank](../../../../glossary.md#term-rank) nested Java primitive array with
inferred [shape](../../../../glossary.md#shape) and [row-major](../../../../glossary.md#term-row-major) flattening.

The legacy implementation accepted `Object`, inspected only the first branch for shape, required
non-empty levels accidentally, flattened only `double` leaves, permitted explicit conversion to
another floating [data type](../../../../glossary.md#data-type--datatype), and did not validate rectangularity before copying. Empty prefixes
could fail with incidental indexing exceptions, ragged data could [overflow](../../../../glossary.md#term-overflow) or underfill the flat
array, and Tensor constructors mixed import with [graph](../../../../glossary.md#graph)/[autograd](../../../../glossary.md#autograd) state.

The new task keeps arbitrary-rank primitive-array convenience but rejects those couplings and
ambiguities. [Runtime](../../../../glossary.md#term-runtime) primitive [carrier](../../../../glossary.md#term-storage-carrier) determines [`DataType`](../../../../glossary.md#data-type--datatype) exactly; every observable branch is
validated; empty-shape policy is explicit; construction remains in `TensorFactory`; and graph,
autograd, [backend](../../../../glossary.md#backend), and runtime state remain absent.

## Architecture constraints

- Production remains in `io.github.pho001.synaptik.model.tensor`. The public entry belongs to
  `TensorFactory`; structural array handling belongs to one package-private helper in the same
  package.
- Package direction remains `model.tensor` toward existing `datatype`, [`shape`](../../../../glossary.md#shape), [`layout`](../../../../glossary.md#layout), and
  `storage` foundations. No reverse or cross-[module](../../../../glossary.md#term-module) [dependency](../../../../glossary.md#term-dependency) is introduced.
- `Object` is used only because Java has a distinct runtime class for every primitive-array rank
  and no finite overload family can express arbitrary rank. The method must reject non-arrays,
  rank-one arrays, generic object arrays, wrapper arrays, and unsupported primitive leaves.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [rank glossary](../../../../glossary.md#term-rank); [finite glossary](../../../../glossary.md#term-finite); [rank-one glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- The source runtime class supplies declared rank and ultimate primitive carrier. The implementation
  must not infer type from values, inspect boxed numbers, or select a default type.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Shape [inference](../../../../glossary.md#term-inference) validates every reachable subarray. It must not trust only the first branch.
- A zero length at the final primitive-leaf axis is observable and valid. A zero length at an
  earlier axis hides all following extents and is rejected rather than inventing sizes.<span data-glossary-navigation="true"> ([leaf glossary](../../../../glossary.md#term-leaf))</span><span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis))</span>
- Inferred sizes are non-negative `long` dimensions. Checked logical count must fit a Java array
  because flattening creates one primitive array; no native or chunked fallback is permitted.<span data-glossary-navigation="true"> ([dimensions glossary](../../../../glossary.md#dimension))</span>
- Flattening creates a fresh primitive array and uses row-major depth-first encounter order. The
  caller's arrays remain owned and mutable by the caller and are neither retained nor modified.<span data-glossary-navigation="true"> ([row-major glossary](../../../../glossary.md#term-row-major); [mutable glossary](../../../../glossary.md#term-mutable))</span>
- The helper constructs only fully static shape and contiguous layout metadata. It performs no
  symbolic inference, graph-wide inference, layout policy, or materialization decision.<span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous); [metadata glossary](../../../../glossary.md#term-metadata); [graph glossary](../../../../glossary.md#graph); [materialization glossary](../../../../glossary.md#materialization))</span>
- [Descriptor](../../../../glossary.md#term-descriptor) construction remains the authority for `requiresGrad` eligibility. Do not duplicate
  differentiability rules in the nested-array helper.
- Final creation dispatches to exactly one matching task-0012B `fromFlatArray(...)` overload. The
  nested path must not call `allocate`, `create`, `nextTensorId`, `new Tensor`, or
  `new MemorySegmentStorage` directly.
- Input inspection and copying are not synchronized with caller mutation. Callers must not mutate
  any source level concurrently; no atomic deep-snapshot guarantee is added.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- If implementation needs a public shape-inference type, general reflection utility, conversion
  policy, another storage contract, native allocation, architecture change, or another module,
  stop and report the conflict.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.tensor` — owns Tensor construction, descriptors, and the new
  nested-import boundary.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [descriptors glossary](../../../../glossary.md#term-descriptor))</span>
- `io.github.pho001.synaptik.model.datatype` — supplies the exact primitive-[carrier](../../../../glossary.md#term-storage-carrier) mapping.
- `io.github.pho001.synaptik.model.shape` — supplies inferred fully static [shapes](../../../../glossary.md#shape).
- `io.github.pho001.synaptik.model.layout` — supplies resolved dense-[contiguous](../../../../glossary.md#term-contiguous) geometry.

Packages added or changed:

- No package is added. Only the existing `model.tensor` package changes.

Type placement:

- `io.github.pho001.synaptik.model.tensor.TensorFactory` — receives the single public nested-array
  entry and delegates its bounded implementation.
- `io.github.pho001.synaptik.model.tensor.NestedTensorArray` — package-private final helper for
  runtime carrier classification, rectangular shape validation, checked flattening, descriptor
  construction, and dispatch to the matching flat-import overload.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [validation glossary](../../../../glossary.md#term-validation); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- `io.github.pho001.synaptik.model.tensor.TensorFactoryNestedImportTest` — focused same-package test
  for the public behavior and package-private boundary.

## Required contract

### Public method

Add exactly this public static method to `TensorFactory`:

```java
public static Tensor fromNestedArray(
        Object source,
        Optional<String> label,
        boolean requiresGrad)
```

The method validates `source` and `label` for null in that order and delegates all remaining work
to `NestedTensorArray`. Do not add unlabeled overloads, explicit-[`DataType`](../../../../glossary.md#data-type--datatype) overloads, [rank](../../../../glossary.md#term-rank)-specific
overloads, generic methods, varargs, boxed-array overloads, or nested-array constructors on
[`Tensor`](../../../../glossary.md#tensor).

The exact [runtime](../../../../glossary.md#term-runtime) [carrier](../../../../glossary.md#term-storage-carrier) mapping is:

| Ultimate primitive component | Inferred `DataType` | Flat carrier |
|---|---|---|
| `double` | [`FLOAT64`](../../../../glossary.md#term-float64) | `double[]` |
| `float` | [`FLOAT32`](../../../../glossary.md#term-float32) | `float[]` |
| `short` | [`BFLOAT16`](../../../../glossary.md#term-bfloat16) [raw bits](../../../../glossary.md#storage-mapping-contiguity-and-injectivity) | `short[]` |
| `int` | [`INT32`](../../../../glossary.md#data-type--datatype) | `int[]` |
| `long` | [`INT64`](../../../../glossary.md#data-type--datatype) | `long[]` |
| `byte` | [`BOOL`](../../../../glossary.md#data-type--datatype) logical bytes | `byte[]` |

### Package-private helper

Add one package-private final `NestedTensorArray` class in `model.tensor`.

It must:

- have no public or protected member and no [mutable](../../../../glossary.md#term-mutable) static state;
- have a private zero-[argument](../../../../glossary.md#term-argument) constructor;
- expose exactly one package-private static entry:

  ```java
  static Tensor importArray(
          Object source,
          Optional<String> label,
          boolean requiresGrad)
  ```

- use private implementation methods or private [immutable](../../../../glossary.md#term-immutable) analysis values only when needed;
- classify source [rank](../../../../glossary.md#term-rank) and ultimate primitive component from its [runtime](../../../../glossary.md#term-runtime) array class;
- validate the entire reachable nested structure before allocating the flat [carrier](../../../../glossary.md#term-storage-carrier);
- infer a [`Shape`](../../../../glossary.md#shape), validate checked count and Java-array capacity, flatten into one matching
  primitive array, construct a contiguous <span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous))</span>`TensorDescriptor`, and [dispatch](../../../../glossary.md#term-dispatch) to task 0012B; and
- retain no source reference after the public call returns.

Do not add a reusable public reflection utility, public nested-array [descriptor](../../../../glossary.md#term-descriptor), public flattening
result, [cache](../../../../glossary.md#term-cache), [registry](../../../../glossary.md#term-registry), service, thread-local state, or source-retaining object.

### Structural [model](../../../../glossary.md#term-model)

`source.getClass()` must represent an array of declared [rank](../../../../glossary.md#term-rank) at least two. Repeated
`Class.getComponentType()` calls determine rank and ultimate component type without examining a
[sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) value. This preserves primitive [carrier](../../../../glossary.md#term-storage-carrier) [identity](../../../../glossary.md#term-identity-meanings) even for an empty [leaf](../../../../glossary.md#term-leaf).

For each array encountered at [axis](../../../../glossary.md#term-axis) `a`:

1. Its length establishes the expected size for axis `a` when no earlier branch established it.
2. Otherwise its length must equal the established size.
3. If `a` is not the final leaf axis, every element must be a non-null subarray of the declared
   next-level array type.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element))</span>
4. If a non-final axis length is zero, reject because later extents cannot be observed.
5. At the final axis, the primitive array length may be zero.

Paths in diagnostics use zero-based bracket notation: root is `[]`, its second child is `[1]`, and
the third child of that array is `[1][2]`.

Examples:

| Source | Result |
|---|---|
| `new double[][] {{1, 2}, {3, 4}}` | [`FLOAT64`](../../../../glossary.md#term-float64), [shape](../../../../glossary.md#shape) `[2, 2]`, flat `[1, 2, 3, 4]` |
| `new int[2][0]` | [`INT32`](../../../../glossary.md#data-type--datatype), shape `[2, 0]`, empty flat carrier |
| `new float[0][3]` | rejected; axis 0 is empty and axis 1 is unobservable |
| `new long[2][0][4]` | rejected; axis 1 is empty and axis 2 is unobservable |
| `new byte[][] {{0, -2}, {3, 0}}` | [`BOOL`](../../../../glossary.md#data-type--datatype), shape `[2, 2]`, stored `[0, 1, 1, 0]` |
| `new double[][] {{1}, {2, 3}}` | rejected as ragged at axis 1 |
| `new double[][] {null, {1}}` | rejected for null subarray at path `[0]` |

### [Validation](../../../../glossary.md#term-validation) order and exact failures

The public method validates before helper delegation:

1. Null `source` throws `NullPointerException` with exact message `source`.
2. Null `label` throws `NullPointerException` with exact message `label`.

The helper then validates in this order before flat-[carrier](../../../../glossary.md#term-storage-carrier), destination, or ID allocation:

1. A non-array source throws `IllegalArgumentException` with exact message
   `nested tensor source must be an array: actual=<runtimeClassName>`.
2. Declared array [rank](../../../../glossary.md#term-rank) below two throws `IllegalArgumentException` with exact message
   `nested tensor source must have rank at least 2: actual=<rank>`.
3. An ultimate component other than the six supported primitives throws
   `IllegalArgumentException` with exact message
   `nested tensor source leaf carrier is unsupported: <componentTypeName>`.
4. Traverse arrays in [row-major](../../../../glossary.md#term-row-major) depth-first order. A null required subarray throws
   `IllegalArgumentException` with exact message
   `nested tensor source contains null subarray at path <path>`.
5. A subarray with a different length from the first established length at its [axis](../../../../glossary.md#term-axis) throws
   `IllegalArgumentException` with exact message
   `nested tensor source is ragged at axis <axis>, path <path>: expected=<expected>, actual=<actual>`.
6. A zero-length non-final axis throws `IllegalArgumentException` with exact message
   `nested tensor source cannot infer dimensions after empty axis <axis> at path <path>`.
7. Checked logical [element](../../../../glossary.md#term-element)-count [overflow](../../../../glossary.md#term-overflow) remains `ArithmeticException`. A non-overflowing count
   above `Integer.MAX_VALUE` throws `IllegalArgumentException` with exact message
   `nested tensor element count exceeds Java array limit: required=<required>, maximum=2147483647`.

All failures above consume no [tensor](../../../../glossary.md#tensor) identifier and allocate no destination storage. The helper may
allocate bounded traversal [metadata](../../../../glossary.md#term-metadata), but it must not allocate the flat carrier before complete
structural validation succeeds.

After structural validation:

1. Allocate exactly one flat primitive array of the inferred logical count.
2. Copy primitive leaves into it in row-major order without boxing individual numeric values.
3. Construct `Shape.of(inferredSizes)` and `LayoutDescriptor.contiguous(shape)`.
4. Construct `TensorDescriptor(inferredDataType, shape, Optional.of(layout), requiresGrad)`.
5. [Dispatch](../../../../glossary.md#term-dispatch) the flat carrier to exactly one matching `TensorFactory.fromFlatArray(...)` overload.

For non-[differentiable](../../../../glossary.md#derivative-and-chain-rule) data types, `requiresGrad=true` is rejected by `TensorDescriptor` before
destination or ID allocation. It may occur after the intermediate flat carrier has been allocated;
the helper must not duplicate [descriptor](../../../../glossary.md#term-descriptor) validation solely to avoid that allocation.

A blank label reaches task 0012B after [inference](../../../../glossary.md#term-inference) and flattening. It therefore allocates destination
storage and consumes one identifier through existing Tensor validation before failing. Identifier
exhaustion occurs after destination allocation and before final flat copy, exactly as in task 0012B.
Neither failure exposes or retains the intermediate flattened carrier.

### Flattening and [ownership](../../../../glossary.md#term-ownership-meanings)

- Numeric carriers and [BFLOAT16](../../../../glossary.md#term-bfloat16) raw shorts are copied unchanged into the intermediate flat array.
- BOOL source bytes are copied raw into the intermediate array; task 0012B performs the one
  canonical zero/non-zero normalization while copying into tensor storage.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [canonical glossary](../../../../glossary.md#term-canonical-form); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- Primitive [leaf](../../../../glossary.md#term-leaf) copying should use typed `System.arraycopy` or equivalent bulk copy. Do not box
  each numeric element, convert through <span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element))</span>`double`, or use a generic numeric list.
- The nested source, every subarray, and the intermediate flat array remain implementation inputs;
  none is returned, stored in Tensor metadata, or retained by destination storage.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata))</span>
- Later caller mutation of any source level cannot change the returned [tensor](../../../../glossary.md#tensor).
- Concurrent caller mutation during inspection or flattening has no snapshot guarantee and is not
  supported.<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot))</span>

## Affected files

Expected implementation and tests:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorFactory.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/NestedTensorArray.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryNestedImportTest.java`

Expected documentation and [planning](../../../../glossary.md#term-planning) updates in the same overall change:

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/tasks/0012c-nested-typed-tensor-import.md`
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most the nine paths listed under Affected files.

No existing production Java contract other than `TensorFactory` may change. `TensorFactoryTest`
may change only for the deliberate exact [API](../../../../glossary.md#term-application-programming-interface-api)-[shape](../../../../glossary.md#shape) expectation. If another production type, test,
documentation file, architecture file, build file, [module](../../../../glossary.md#term-module), or [dependency](../../../../glossary.md#term-dependency) is needed, stop and
propose a separate follow-up or architecture decision.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Update `TensorFactory` type Javadoc to distinguish completed flat and nested import.
- Fully document `fromNestedArray(...)`, including why `Object` is necessary for arbitrary rank,
  supported runtime carriers, rank and rectangularity, inferred shape/type/layout, empty-axis
  policy, raw BFLOAT16 and BOOL behavior, source ownership, concurrent mutation limitation,<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [runtime glossary](../../../../glossary.md#term-runtime); [layout glossary](../../../../glossary.md#layout); [axis glossary](../../../../glossary.md#term-axis); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [BOOL glossary](../../../../glossary.md#data-type--datatype); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
  `requiresGrad`, label/ID/allocation side effects, result, and every failure category.
- Document every parameter with nullability, ownership, and semantic constraints; document the
  non-null result and all expected exceptions.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [nullability glossary](../../../../glossary.md#term-nullability))</span>
- Document package-private `NestedTensorArray` and its package-private entry with purpose,
  invariants, traversal/flattening behavior, ownership, parameters, result, and failures.
- Review existing [`Tensor`](../../../../glossary.md#tensor), `TensorDescriptor`, [`Shape`](../../../../glossary.md#shape), `LayoutDescriptor`, [`DataType`](../../../../glossary.md#data-type--datatype),
  `HostTensorStorage`, `MemorySegmentStorage`, and task-0012B Javadocs. Change none unless an
  in-scope stale statement can be corrected without changing behavior; otherwise stop on an
  out-of-scope discrepancy and record reasoned no-change conclusions.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>

## Acceptance criteria

- `TensorFactory` adds exactly the one public method specified and no other public surface.
- One package-private helper implements only nested primitive-array [inference](../../../../glossary.md#term-inference) and flattening.
- All six supported ultimate primitive carriers infer the exact matching [`DataType`](../../../../glossary.md#data-type--datatype).
- [Runtime](../../../../glossary.md#term-runtime) [rank](../../../../glossary.md#term-rank) must be at least two; unsupported, boxed, generic, and [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) sources fail exactly.
- Every reachable branch is validated; ragged and null-subarray failures include exact axis/path
  diagnostics and occur before flat/destination/ID allocation.
- Empty final [leaf](../../../../glossary.md#term-leaf) [axes](../../../../glossary.md#term-axis) work; empty non-final axes fail because trailing [dimensions](../../../../glossary.md#dimension) are unknown.
- Inferred [shape](../../../../glossary.md#shape) is fully static, [layout](../../../../glossary.md#layout) is resolved dense-[contiguous](../../../../glossary.md#term-contiguous), and `requiresGrad` is
  preserved subject to existing descriptor eligibility.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Flattening preserves row-major order, raw numeric/BFLOAT16 bits, source independence, and task-
  0012B BOOL normalization.<span data-glossary-navigation="true"> ([row-major glossary](../../../../glossary.md#term-row-major); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [BOOL glossary](../../../../glossary.md#data-type--datatype); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- Final creation invokes one matching flat-import overload and never directly allocates storage or
  identity and never constructs <span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>[`Tensor`](../../../../glossary.md#tensor).
- Scalar/flat import, constants, range/prefix, random, typed access/export, storage, Tensor,
  descriptor, layout, allocator, and graph behavior remain unchanged.<span data-glossary-navigation="true"> ([Scalar glossary](../../../../glossary.md#term-scalar); [constants glossary](../../../../glossary.md#term-constant); [graph glossary](../../../../glossary.md#graph))</span>
- Complete Javadoc, Tensor API, glossary, task/master/roadmap status, and separate documentation
  review are finished in the same change.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Task 0012C is `Complete` only after all [validation](../../../../glossary.md#term-validation); task 0012D remains `Draft` without a spec.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) before and after the documentation pass:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryNestedImportTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

Focused tests must cover:

- exact factory [API](../../../../glossary.md#term-application-programming-interface-api) and helper visibility/state [shape](../../../../glossary.md#shape);
- [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) and [rank](../../../../glossary.md#term-rank)-three [row-major](../../../../glossary.md#term-row-major) imports for all six carriers;
- raw floating/[BFLOAT16](../../../../glossary.md#term-bfloat16) preservation, [BOOL](../../../../glossary.md#data-type--datatype) [normalization](../../../../glossary.md#term-normalization), and post-return source mutation;
- inferred [`DataType`](../../../../glossary.md#data-type--datatype), `Shape`, dense [layout](../../../../glossary.md#layout), label, and `requiresGrad`;
- accepted empty final [axis](../../../../glossary.md#term-axis) and exact rejection of ambiguous empty non-final axes;
- null argument order, non-array, rank-one, unsupported primitive, boxed/generic array, null
  subarray, and ragged array failures with exact messages;<span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument); [rank-one glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- no ID consumption for structural and [descriptor](../../../../glossary.md#term-descriptor)-eligibility failures;
- delegated blank-label consumption and identifier-exhaustion behavior; and
- preservation of all existing task-0012B tests.

Manually verify reflection and `javap -p -c -s` show the exact new public method, package-private
helper, existing methods/allocator unchanged, public null order, helper [dispatch](../../../../glossary.md#term-dispatch) into flat import,
and no direct [Tensor](../../../../glossary.md#tensor)/storage/ID construction. Verify production imports contain only current
[model](../../../../glossary.md#term-model)/[JDK](../../../../glossary.md#term-java-development-kit-jdk) types and no collections used as value [buffers](../../../../glossary.md#term-buffer), [graph](../../../../glossary.md#graph)/[runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend) types, [Arena](../../../../glossary.md#term-arena),
native access, conversion utilities, or service state. Verify exact nine-path [scope](../../../../glossary.md#term-scope), generated
[Javadoc](../../../../glossary.md#term-javadoc), documentation examples, links/anchors, fences, whitespace, status [synchronization](../../../../glossary.md#term-synchronization), and
absence of task-0012D/task-0013 specs.

## Dependencies

- Task 0012B provides the exact typed flat-import and [BOOL](../../../../glossary.md#data-type--datatype)-[normalization](../../../../glossary.md#term-normalization) boundary and is complete.
- Task 0012A provides dense [heap](../../../../glossary.md#term-heap) allocation and is complete.
- Tasks 0001, 0002, 0003, 0007, and 0012 provide data type, shape, layout, descriptor, and factory
  identity contracts.<span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype); [shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [descriptor glossary](../../../../glossary.md#term-descriptor); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>

## Follow-up tasks

- Task 0012D will define [scalar](../../../../glossary.md#term-scalar) and [constant](../../../../glossary.md#term-constant) creation, zeros, ones, zeros-like, and ones-like.
- Task 0012E will define integer ranges and strict/cyclic prefix population.
- Task 0012F will define random creation and [reproducibility](../../../../glossary.md#artifact-manifest-and-source-provenance) policy.
- Later work owns typed export/access, mutation versions, [views](../../../../glossary.md#term-view-meanings)/scatter, [provenance](../../../../glossary.md#provenance), and operations.

Do not create a detailed task-0012D or later specification as part of task 0012C.

## Architecture impact

Expected impact: None.

The architecture already assigns TensorFactory, [shape](../../../../glossary.md#shape)/data-type/[layout](../../../../glossary.md#layout) [model](../../../../glossary.md#term-model), and [host storage](../../../../glossary.md#host-storage) to
`modules/model`. This task adds copied nested [leaf](../../../../glossary.md#term-leaf) data only and changes no [module](../../../../glossary.md#term-module) boundary,
[dependency direction](../../../../glossary.md#java-platform-and-build-tools), [lifecycle](../../../../glossary.md#lifecycle) stage, storage [ownership](../../../../glossary.md#term-ownership-meanings), [runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend) rule, or [compiler](../../../../glossary.md#term-compiler)
responsibility. If implementation reveals otherwise, stop before editing architecture files.

## Implementation prompt

Use this prompt in a separate agentic task/thread with a clean context:

```text
You are a clean-context implementation agent working in the Synaptik repository.

Read first and in full:
- AGENTS.md
- ARCHITECTURE.md
- docs/architecture/current-architecture-plan.md
- docs/architecture/overview.md
- docs/architecture/lifecycle.md
- docs/architecture/module-boundaries.md
- docs/architecture/dependency-rules.md
- docs/developer-guide/documentation-rules.md
- docs/planning/planning-guide.md
- docs/planning/roadmap.md
- docs/planning/modules/model/capabilities.md
- docs/planning/modules/model/master-plan.md
- docs/planning/modules/model/tasks/0001-data-type-model.md
- docs/planning/modules/model/tasks/0002-shape-and-dimension-model.md
- docs/planning/modules/model/tasks/0003-layout-descriptor-model.md
- docs/planning/modules/model/tasks/0007-tensor-descriptor-model.md
- docs/planning/modules/model/tasks/0012-tensor-factory.md
- docs/planning/modules/model/tasks/0012a-host-storage-allocation.md
- docs/planning/modules/model/tasks/0012b-flat-typed-tensor-import.md
- docs/planning/modules/model/tasks/0012c-nested-typed-tensor-import.md
- docs/api/tensor-api.md
- docs/glossary.md
- current TensorFactory/Tensor/descriptor/shape/layout/storage production and tests
- root/model Gradle configuration only to confirm Java 26

Implement task 0012C exactly as specified. Modify TensorFactory.java, add the one package-private
NestedTensorArray.java helper, update TensorFactoryTest only for exact API shape, and add
TensorFactoryNestedImportTest.java. Add exactly one public fromNestedArray(Object, Optional<String>,
boolean) method and no other public API. Preserve every existing factory method, allocator,
allocation/import path, and test.

Accept only true rank-two-or-greater arrays whose ultimate primitive carrier is double, float,
short, int, long, or byte. Infer exact DataType, validate every branch for rectangularity and null
subarrays, reject ambiguous empty non-final axes, accept an observable empty final axis, infer a
static Shape and dense-contiguous descriptor, flatten into a fresh matching primitive array in
row-major order, and delegate exactly once to the matching task-0012B fromFlatArray overload.
Preserve raw numeric/BFLOAT16 values and let flat import normalize BOOL. Follow exact validation
order, messages, ownership, and ID side effects from the task.

Do not add conversion, rank-specific overloads, boxed/generic inputs, collections, scalar/constant,
range/prefix, random, typed access/export, view/scatter behavior, storage types, Arena/native
allocation, provenance, operations, graph/compiler/runtime/backend behavior, dependencies, build
changes, or follow-up specs. Stop if work exceeds the nine permitted paths or requires an
architecture decision.

Run every focused/aggregate test, Javadoc, bytecode/import/manual, documentation, link, whitespace,
scope, and status check in the task.

After initial implementation validation, hand the actual diff to a separate documentation-focused
agent or thread with clean context in the same overall change. Keep task 0012C incomplete until
that pass finishes. The handoff must include this task, implementation/test diff, Object/runtime-
carrier rationale, rectangularity and empty-axis rules, shape/type/layout inference, flattening and
ownership, flat-import delegation and ID side effects, architecture constraints, expected Tensor
API/glossary/Javadoc impact, existing-Javadoc review list, and every validation command.

That documentation agent must independently read AGENTS.md, ARCHITECTURE.md,
docs/developer-guide/documentation-rules.md, the documentation profile index, General style,
API/Javadoc style, Planning style, Example format when an example changes, this task, final
source/tests/generated Javadoc, Tensor API, glossary, model master plan, roadmap, and existing
Tensor/TensorFactory/TensorDescriptor/DataType/Shape/LayoutDescriptor/storage contracts. It must
inspect actual implementation and evidence rather than rely on the handoff. It must finalize all
new/affected Javadocs, move only nested primitive-array import into current API/glossary language,
preserve tasks 0012D–0012F as planned, review links/anchors/fences/whitespace and terminology,
record reasoned existing-Javadoc and architecture/capability no-change conclusions, and
synchronize only the allowed planning files.

At the end, update this task, model master plan, and roadmap for status/evidence. Record local
decisions, known limitations, exact validation evidence including documentation-agent identity and
results, implementation notes, and the canonical completion summary. Do not mark task 0012C
Complete until implementation, tests, Javadoc, independent documentation pass, scope review, and
status synchronization all pass. Task 0012D then remains the next Draft frontier without a detailed
specification. Do not commit or push.
```

## Local decisions

- One `Object` entry is deliberate and narrow. Java preserves primitive leaf type and declared rank
  in the runtime array class, but arbitrary rank cannot be represented by a finite overload family.
  Runtime validation therefore recovers strong carrier semantics without accepting arbitrary
  objects or boxed values.<span data-glossary-navigation="true"> ([leaf glossary](../../../../glossary.md#term-leaf); [rank glossary](../../../../glossary.md#term-rank); [runtime glossary](../../../../glossary.md#term-runtime); [finite glossary](../../../../glossary.md#term-finite); [validation glossary](../../../../glossary.md#term-validation))</span>
- [Carrier](../../../../glossary.md#term-storage-carrier) determines [`DataType`](../../../../glossary.md#data-type--datatype); there is no explicit type argument and no conversion. This is
  stricter than the legacy double-to-floating conversion path and consistent with flat import.<span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument))</span>
- Rank one remains flat import. Nested import begins at rank two so the two APIs have non-overlapping
  responsibilities.
- Rectangularity validates the complete reachable source rather than trusting the first branch.
- Empty final leaves are valid because their zero extent is observable. Earlier empty axes are
  rejected because Java runtime objects do not retain requested trailing extents when no child
  array exists.<span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent); [axes glossary](../../../../glossary.md#term-axis))</span>
- Flattening uses a fresh typed carrier and then flat import performs the destination copy. The
  extra bounded copy centralizes allocation, BOOL normalization, label behavior, and ID semantics.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- Package-private `NestedTensorArray` prevents `TensorFactory` from becoming a traversal god class
  while avoiding a new public concept or generic utility package.

## Known limitations

- Only multidimensional primitive arrays with one of the six exact [leaf](../../../../glossary.md#term-leaf) carriers are accepted.
- Empty non-final axes are rejected even when source code used a multidimensional allocation whose
  trailing dimensions were syntactically specified; those extents are not recoverable from the
  empty runtime object graph.<span data-glossary-navigation="true"> ([axes glossary](../../../../glossary.md#term-axis); [dimensions glossary](../../../../glossary.md#dimension); [runtime glossary](../../../../glossary.md#term-runtime); [graph glossary](../../../../glossary.md#graph))</span>
- Nested import performs one intermediate flattening allocation and one destination copy.
- Import is not [atomic](../../../../glossary.md#resource-transaction-and-rollback) with concurrent caller mutation of any source level.
- No conversion, boxed values, collections, dynamic shape, custom layout, typed export, or general
  view/scatter population is provided.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [view glossary](../../../../glossary.md#term-view-meanings))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

Implementation and the independent documentation pass reviewed the authoritative architecture,
[planning](../../../../glossary.md#term-planning)/documentation rules, capability baseline, completed factory/[descriptor](../../../../glossary.md#term-descriptor)/[shape](../../../../glossary.md#shape)/[layout](../../../../glossary.md#layout) and
storage contracts, final source and tests, generated [Javadoc](../../../../glossary.md#term-javadoc), and the task-0012B flat-import
boundary. The separate documentation-focused context was
`/root/implement_model_0012c/review_model_0012c_docs`. It applied General style, [API](../../../../glossary.md#term-application-programming-interface-api) and Javadoc,
Planning, and Example format profiles and independently finalized Javadoc, [Tensor](../../../../glossary.md#tensor) API, glossary,
planning status, and evidence against the actual implementation.

- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorFactoryTest` — passed; JUnit XML reports 7 tests,
  0 failures, 0 errors, and 0 skipped.<span data-glossary-navigation="true"> ([JUnit glossary](../../../../glossary.md#term-junit))</span>
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorFactoryNestedImportTest` — passed; JUnit XML reports
  10 tests, 0 failures, 0 errors, and 0 skipped.
- `./gradlew :modules:model:test` — passed; 26 JUnit XML suites report 187 tests, 0 failures,
  0 errors, and 0 skipped.
- `./gradlew :modules:model:javadoc` — passed with generated public Javadoc inspected for the
  exact `fromNestedArray(Object, Optional<String>, boolean)` signature, rendered ownership,
  diagnostics, and side effects. Package-private <span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>`NestedTensorArray` is correctly absent from the
  public generated index; its complete source Javadoc was reviewed independently.
- `./gradlew test` — passed; the final standalone run reported 36 actionable lifecycle tasks, all
  up-to-date after the preceding focused validation.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- The complete nested-import API example was compiled and [run](../../../../glossary.md#term-run) directly from standard input with
  `java --source 26 --class-path modules/model/build/classes/java/main /dev/stdin`; it exited 0 and
  printed [`FLOAT32`](../../../../glossary.md#term-float32), `Shape[2, 2]`, `DENSE_CONTIGUOUS`, `true`, [`weights`](../../../../glossary.md#term-weight), and
  `[1.0, 2.0, 3.0, 4.0]` as documented.
- `javap -classpath modules/model/build/classes/java/main -p -c -s` confirmed that
  `TensorFactory` has exactly the one new public nested method, checks `source` before `label`, and
  delegates directly to the helper. Existing public methods, private flat importer, two private
  final allocator fields, private constructor, and private ID allocator remain unchanged.
- The same `javap` command confirmed that package-private final `NestedTensorArray` has no fields,
  nested classes, public members, or protected members; its only non-private entry is
  `importArray(Object, Optional, boolean)`. Bytecode validates the structure before the primitive
  flat allocation, uses <span data-glossary-navigation="true"> ([Bytecode glossary](../../../../glossary.md#term-bytecode))</span>`System.arraycopy` for leaves, constructs `Shape`, contiguous layout, and
  descriptor in order, and dispatches to exactly one matching flat overload. It contains no direct<span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous))</span>
  `allocate`, `create`, `nextTensorId`, `new Tensor`, or `new MemorySegmentStorage` reference.
- Production imports are limited to current model foundations and JDK reflection/utility types.
  Review found no collection value buffer, graph/compiler/planning/runtime/backend type, arena,
  native access, conversion utility, service state, or added dependency.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [buffer glossary](../../../../glossary.md#term-buffer); [graph glossary](../../../../glossary.md#graph); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [arena glossary](../../../../glossary.md#term-arena); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- A targeted local Markdown checker validated all five changed documentation/planning files: 147
  local links, including 58 anchors, with 0 errors. Fence checks found balanced fences; trailing-
  whitespace review found 0 findings; `git diff --check` passed.
- Scope review found exactly the nine permitted repository paths, including the pre-existing
  planning changes. There is no architecture, focused architecture-documentation, architecture-
  test, build, dependency, capability-baseline, other-module, backend-conformance, or integration-
  test change. The capability baseline already selects nested Java arrays, so no baseline update
  was needed. This model-only construction path changes no backend or end-to-end execution
  behavior, so backend-conformance and integration tests require no focused additions.<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope); [module glossary](../../../../glossary.md#term-module); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Existing `Tensor`, `TensorDescriptor`, [`DataType`](../../../../glossary.md#data-type--datatype), `Shape`, `LayoutDescriptor`,
  `HostTensorStorage`, and `MemorySegmentStorage` Javadocs remain accurate: nested import composes
  their existing identity, gradient-eligibility, static-shape, dense-layout, carrier-width, and
  borrowed-storage contracts without changing them. Existing descriptor creation, allocation,
  flat-import, and private allocator Javadocs also remain accurate because the new path delegates
  to them without changing their direct contracts. No out-of-scope Javadoc correction was needed.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [gradient glossary](../../../../glossary.md#term-gradient); [carrier glossary](../../../../glossary.md#term-storage-carrier); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease))</span>
- Tensor API and glossary now classify only rectangular nested primitive-array import as current;
  constants, ranges/prefixes, random generation, and later capabilities remain planned. Task,
  model master plan, and roadmap agree that 0012C is <span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant))</span>`Complete`, 0012D is the next `Draft`
  frontier, and no task-0012D detailed specification exists.

## Implementation notes

- Added the one public `TensorFactory.fromNestedArray(Object, Optional<String>, boolean)` entry and
  kept its null checks at the public boundary.
- Added package-private final `NestedTensorArray` for runtime rank/carrier classification, complete
  rectangular/null and empty-axis validation, checked shape/capacity inference, typed row-major
  flattening, exact descriptor construction, and one matching flat-import dispatch.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [rank glossary](../../../../glossary.md#term-rank); [carrier glossary](../../../../glossary.md#term-storage-carrier); [axis glossary](../../../../glossary.md#term-axis); [validation glossary](../../../../glossary.md#term-validation); [inference glossary](../../../../glossary.md#term-inference); [row-major glossary](../../../../glossary.md#term-row-major); [descriptor glossary](../../../../glossary.md#term-descriptor); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Added [API](../../../../glossary.md#term-application-programming-interface-api)-[shape](../../../../glossary.md#shape) coverage to `TensorFactoryTest` and the focused ten-test nested-import suite,
  including all six carriers, rank two and three, raw floating/BFLOAT16 values, BOOL normalization,
  source independence, empty-axis policy, exact diagnostics, gradient eligibility, and ID effects.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [BOOL glossary](../../../../glossary.md#data-type--datatype); [normalization glossary](../../../../glossary.md#term-normalization); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Finalized `TensorFactory` and helper Javadocs. In particular, the factory type contract now
  identifies nested import as the bounded descriptor-synthesis exception instead of incorrectly
  claiming that no factory path constructs descriptors.
- Updated Tensor API and glossary current-vs-planned language and added one complete verified
  nested-import example. Architecture explanations and tests need no update because module
  ownership and dependency rules did not change.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [module glossary](../../../../glossary.md#term-module); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Completion summary

- Completed changes: implemented exact-carrier rectangular nested primitive-array import with
  static dense descriptor inference, row-major copied ownership, flat-import delegation, and the
  specified validation and ID side effects.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [descriptor glossary](../../../../glossary.md#term-descriptor); [inference glossary](../../../../glossary.md#term-inference); [row-major glossary](../../../../glossary.md#term-row-major); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [validation glossary](../../../../glossary.md#term-validation))</span>
- Files changed or created: `TensorFactory.java`, `NestedTensorArray.java`, `TensorFactoryTest.java`,
  `TensorFactoryNestedImportTest.java`, `docs/api/tensor-api.md`, `docs/glossary.md`, this task,
  model <span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>`master-plan.md`, and `docs/planning/roadmap.md`.
- Tests and validation: focused tests 7/7 and 10/10; all model tests 187/187; model Javadoc and full
  repository tests passed; bytecode, imports, generated docs, example, links/anchors, fences,
  whitespace, exact scope, and status synchronization passed.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [bytecode glossary](../../../../glossary.md#term-bytecode); [scope glossary](../../../../glossary.md#term-scope); [synchronization glossary](../../../../glossary.md#term-synchronization))</span>
- Documentation-agent review: completed by
  `/root/implement_model_0012c/review_model_0012c_docs` using General, API/Javadoc, Planning, and
  Example format profiles.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: Tensor API and glossary now document nested import as implemented; no
  architecture or capability-baseline document changed.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- Javadoc review: new and affected factory/helper contracts were finalized; adjacent tensor,
  descriptor, data-type, shape, layout, storage, flat-import, and allocator contracts were reviewed
  and require no change.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout))</span>
- Glossary impact: current implementation status, Tensor, and Tensor factory entries were updated;
  no separate reusable term was needed.<span data-glossary-navigation="true"> ([Tensor factory glossary](../../../../glossary.md#tensor-factory))</span>
- Unresolved issues: None.
- Follow-up required: None. Task 0012D remains the next Draft planning frontier without a detailed
  specification.

Status: Complete
