# Task 0012A: JVM-Managed Heap Host Storage Allocation

## Status

Complete

## Goal

Add the smallest public host-storage allocation surface to the existing `TensorFactory`. Allocate
one Java primitive array whose carrier matches the descriptor data type and whose length equals the
resolved layout's referenced element span, wrap the array's Java 26 heap `MemorySegment` in the
existing borrowed `MemorySegmentStorage`, and delegate final tensor construction and identifier
allocation to the existing `TensorFactory.create(...)` path.

This task adds JVM-managed heap allocation, not a new owning storage abstraction. Java 26 heap
segments use an automatic scope that keeps the backing array reachable and is accessible from any
thread. Retaining the segment through `MemorySegmentStorage` therefore gives the allocated array a
safe garbage-collected lifetime without an `Arena`, close operation, external owner, or change to
the completed storage contracts.

## Scope

- Modify only the existing public final `TensorFactory` production type.
- Add exactly two public static allocation overloads: one descriptor-only convenience and one
  descriptor-plus-optional-label method.
- Require the supplied descriptor to contain a resolved `LayoutDescriptor`.
- Allocate exactly `layout.referencedElementSpan()` physical elements, including offset and
  strided geometry rather than logical element count.
- Reject a required span greater than `Integer.MAX_VALUE` before attempting Java-array allocation.
- Allocate one correctly typed, JVM-zero-initialized primitive array for each of the six current
  `DataType` values.
- Create a heap segment with the matching `MemorySegment.ofArray(...)` overload and wrap it in the
  existing `MemorySegmentStorage` with the descriptor data type and exact span.
- Delegate to the existing complete `TensorFactory.create(...)` overload with the supplied
  descriptor and label plus the newly created storage.
- Preserve the existing factory identifier allocator, constructor delegation, label validation,
  tensor validation, and failure-consumption semantics.
- Update the existing exact-shape `TensorFactoryTest` and add one focused
  `TensorFactoryAllocationTest`.
- During implementation, update the Tensor API, glossary, task evidence, model master plan, and
  roadmap through a separate clean-context documentation-focused pass.

## Out of scope

- a new owning storage wrapper, another `HostTensorStorage` implementation, a change to the sealed
  permits list, or any modification to `HostTensorStorage` or `MemorySegmentStorage`
- an `Arena` field, native or off-heap allocation, mapped memory, external allocation handles,
  `AutoCloseable`, close methods, cleanup callbacks, `Cleaner`, finalizers, leases, reference
  counting, pooling, or runtime resources
- typed element reads or writes, bulk access, public backing-array access, copying, conversion,
  BFLOAT16 conversion, boolean normalization, fill loops, or mutation/version tracking
- flat typed import, nested import, logical-count/data validation, or any task-0012B/0012C behavior
- scalar, zeros, ones, zeros-like, ones-like, or other public constant-population conveniences
  assigned to task 0012D
- integer ranges, strict/cyclic prefixes, or random population assigned to tasks 0012E/0012F
- accepting caller-supplied storage in either new allocation method; the existing `create(...)`
  overload remains the only factory method in this task that accepts storage
- descriptor construction, replacement, or mutation; default data types; shape inference; layout
  synthesis; resolving an absent layout; contiguous-layout inference; symbolic binding; or
  materialization policy
- provenance, operations, graph IDs, gradients, trainable state, publication, compiler, planning,
  prepare, runtime, engine, backend, device, residency, or execution behavior
- a helper production type, new package, new dependency, preview/incubator feature, Gradle change,
  architecture change, architecture test, backend-conformance test, or integration test
- a detailed specification for task 0012B, task 0013, or any later task

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md), especially model ownership of
  `TensorFactory`, `HostTensorStorage`, and `MemorySegmentStorage` and the exclusion of runtime and
  backend storage
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Architecture overview](../../../../architecture/overview.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md), especially data type, layout, host storage, and
  factory capabilities
- [Model master plan](../master-plan.md), especially `model.tensor`, `model.storage`, and package
  direction
- [Task 0001](0001-data-type-model.md), which defines the six data types and byte widths
- [Task 0003](0003-layout-descriptor-model.md), which defines resolved span geometry
- [Task 0007](0007-tensor-descriptor-model.md), which defines explicit resolved/unresolved layout
  state
- [Task 0010](0010-host-storage-abstraction.md), which defines the exact-size borrowed segment
  wrapper
- [Task 0011](0011-public-tensor-skeleton.md), which defines tensor/storage compatibility
- [Task 0012](0012-tensor-factory.md), which defines the public factory, ID allocation, and
  canonical construction path
- [Tensor API](../../../../api/tensor-api.md) and [glossary](../../../../glossary.md), which
  currently describe allocation as planned

## Legacy evidence and rejected coupling

Read-only inspection of `legacy/pre-rewrite` covered `TensorDataFactory`, the six heap storage
classes, `TensorStorage`, factory tests, storage data-type tests, native storage allocation tests,
and representative tensor-construction call sites.

Useful capability evidence is limited to these facts:

- every selected data type needs a matching zero-initialized heap carrier;
- allocation and later population are separable operations;
- BFLOAT16 uses raw `short` storage and BOOL uses `byte` storage;
- scalar and zero-sized allocation need explicit behavior; and
- native allocation has a distinct close/lease lifecycle that must not leak into model heap
  allocation.

The new implementation must not copy the legacy per-data-type storage classes, mutable-array
accessors, version counters, data-type access switch, broad population factory, native allocation
handles, runtime `ExecutionResource` coupling, pooling, close behavior, or legacy shape/scalar
conventions. Legacy code is capability evidence only.

## Architecture constraints

- Production remains in `io.github.pho001.synaptik.model.tensor`. This task modifies only
  `TensorFactory`; it adds no production package or type.
- Package direction remains `model.tensor -> model.storage` plus the existing foundational
  descriptor packages. No reverse or cross-module dependency is added.
- `TensorFactory` may use `MemorySegment` and the existing `MemorySegmentStorage` implementation
  to realize model-owned host allocation. It must not import `Arena`, runtime resources, concrete
  backend storage, or another module.
- `HostTensorStorage` remains a sealed raw boundary permitting exactly `MemorySegmentStorage`.
  `MemorySegmentStorage` remains a final borrowed, non-owning, non-closing identity wrapper with
  its existing constructor and six-method API.
- Each allocated segment is created by `MemorySegment.ofArray(...)`. Under Java 26, the returned
  heap segment has an automatic scope that keeps the supplied primitive array reachable and is
  always accessible from any thread. `MemorySegmentStorage` retains that segment, so no separate
  array field, external owner, arena, lifetime token, or close path is needed.
- The factory-created array is managed by Java garbage collection. The returned tensor borrows the
  `MemorySegmentStorage` object under the existing Tensor contract, while the segment's automatic
  scope supplies the heap-base reachability. This does not convert `MemorySegmentStorage` into an
  owning resource or add deterministic release semantics.
- Allocation requires a present resolved layout. `Optional.empty()` is rejected for both fully
  static and dynamic shapes. The factory must not infer row-major layout, call
  `LayoutDescriptor.contiguous(...)`, or build a replacement descriptor.
- The required capacity is exactly `LayoutDescriptor.referencedElementSpan()`, not
  `Shape.knownElementCount()`. The span is already checked, non-negative, and includes offset,
  strided, and broadcast geometry.
- A scalar resolved layout allocates one element. A resolved layout for a shape with a zero-sized
  dimension allocates a zero-length array even when its storage offset is non-zero. Dense, offset,
  strided, and broadcast layouts all allocate their exact referenced span, which may be greater
  than, equal to, or less than logical element count.
- Java primitive-array length is limited to `int`. A required span greater than
  `Integer.MAX_VALUE` is rejected deterministically; there is no native/off-heap fallback.
- New primitive arrays contain the JVM default all-zero raw representation. This task promises
  only that allocation fact. It does not introduce or claim public zeros, ones, scalar,
  population, normalization, or conversion APIs.
- The existing `TensorFactory.create(...)` method remains the sole ID allocation and Tensor
  construction path. The allocation methods must not call `nextTensorId()` directly, inspect
  allocator state, or duplicate label/storage validation.
- Stable Java 26 APIs are sufficient. No preview/incubator option, native-access flag, dependency,
  or build change is allowed.
- If implementation requires a new owner, new storage type, storage-contract change, descriptor
  synthesis, native allocation, deterministic close, another public method, or architecture or
  dependency change, stop and report the conflict.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.tensor` — owns `TensorFactory`, `Tensor`, and
  `TensorDescriptor`.
- `io.github.pho001.synaptik.model.storage` — owns `HostTensorStorage` and
  `MemorySegmentStorage`.
- `io.github.pho001.synaptik.model.datatype` — reached through the descriptor's `DataType` value
  and the exhaustive allocation switch.
- `io.github.pho001.synaptik.model.layout` — reached through the descriptor's resolved
  `LayoutDescriptor` value.

Packages added or changed:

- No package is added. Only the existing `model.tensor` package changes.

Type placement:

- `io.github.pho001.synaptik.model.tensor.TensorFactory` — the existing public creation boundary
  gains the two allocation overloads and performs the complete allocation algorithm.
- No new production type is needed.

Test placement:

- `io.github.pho001.synaptik.model.tensor.TensorFactoryTest` — retains all task-0012 allocator and
  creation tests and updates only its exact API-shape expectation for the two deliberate methods.
- `io.github.pho001.synaptik.model.tensor.TensorFactoryAllocationTest` — focused allocation,
  carrier, span, zeroing, lifetime, validation, and side-effect tests without weakening or
  duplicating the original allocator suite.

## Required contract

### Public methods

Add exactly these two public methods to `TensorFactory`:

```java
public static Tensor allocate(TensorDescriptor descriptor)

public static Tensor allocate(
        TensorDescriptor descriptor,
        Optional<String> label)
```

`allocate(descriptor)` is the only allocation convenience and is exactly equivalent to:

```java
allocate(descriptor, Optional.empty())
```

The complete overload accepts no storage argument. On success it must return the result of:

```java
create(descriptor, label, Optional.of(storage))
```

where `storage` is the exact `MemorySegmentStorage` created by the allocation algorithm below.
Do not add data-type, shape, layout, raw-array, scalar, fill, label-only, storage-policy, native,
arena, or owner overloads.

### Validation, allocation, and delegation order

The complete overload performs work in this exact order:

1. Reject null `descriptor` with `NullPointerException` and exact message `descriptor`.
2. Reject null `label` with `NullPointerException` and exact message `label`.
3. Read `descriptor.layout()`. If it is empty, throw `IllegalArgumentException` with exact message
   `tensor allocation requires a resolved layout`.
4. Read `layout.referencedElementSpan()` into `requiredSpan`. If it is greater than
   `Integer.MAX_VALUE`, throw `IllegalArgumentException` with exact message
   `tensor allocation span exceeds Java array limit: required=<requiredSpan>, maximum=2147483647`.
5. Convert the validated span to `int` and allocate exactly one matching primitive array as
   specified by the carrier table.
6. Pass that array to the matching `MemorySegment.ofArray(...)` overload.
7. Construct `new MemorySegmentStorage(descriptor.dataType(), requiredSpan, segment)`.
8. Delegate to `create(descriptor, label, Optional.of(storage))` and return its exact result.

The convenience overload delegates to the complete allocation overload so it has the same null,
layout, span, allocation, storage, and identifier behavior.

Do not call `Shape.knownElementCount()`, `Shape.toLongArray()`,
`LayoutDescriptor.contiguous(...)`, `nextTensorId()`, `Arena.of*`, an allocation fallback, or a
typed memory access API. Do not catch and translate `OutOfMemoryError` or another virtual-machine
array-allocation failure.

### Carrier mapping

The mapping is exhaustive and exact:

| `DataType` | New primitive array | Heap-segment factory | Raw initial contents |
|---|---|---|---|
| `FLOAT64` | `new double[length]` | `MemorySegment.ofArray(double[])` | `+0.0d` bit pattern |
| `FLOAT32` | `new float[length]` | `MemorySegment.ofArray(float[])` | `+0.0f` bit pattern |
| `BFLOAT16` | `new short[length]` | `MemorySegment.ofArray(short[])` | raw bits `0x0000` |
| `INT32` | `new int[length]` | `MemorySegment.ofArray(int[])` | `0` |
| `INT64` | `new long[length]` | `MemorySegment.ofArray(long[])` | `0L` |
| `BOOL` | `new byte[length]` | `MemorySegment.ofArray(byte[])` | raw byte `0` |

The array is the segment's heap base but is not exposed by a new Synaptik API. Tests may inspect
`MemorySegment.heapBase()` to prove carrier selection and default zeroing. Production must not
store the array separately or expose mutable backing-array access.

### Capacity and geometry scenarios

| Resolved geometry | Required allocation |
|---|---:|
| scalar at offset zero | span and array length `1` |
| any zero-sized shape, including non-zero stored offset | span and array length `0` |
| contiguous shape `[2, 3]` | span and array length `6` |
| canonical `[2, 3]` with offset `5` | span and array length `11` |
| strided shape `[2, 2]`, strides `[5, 1]` | span and array length `7` |
| broadcast shape `[2, 3]`, strides `[0, 1]` | span and array length `3` |

The offset and strided examples prove that capacity can exceed logical element count. The
broadcast example proves that repeated logical positions do not force allocation of the logical
count. The zero-sized example proves that offset does not create a reference when no logical
element exists.

### Side effects and failure consumption

- Null descriptor and null label failures occur before layout inspection, array allocation, or ID
  allocation. They consume no ID.
- Unresolved-layout and over-array-limit failures occur before array allocation and before
  delegation to `create(...)`. They consume no ID.
- Primitive-array allocation happens before `create(...)`. If the JVM throws `OutOfMemoryError`
  or another allocation error, no tensor ID has been allocated or consumed because the existing
  ID allocator has not been reached. The error propagates unchanged.
- `MemorySegment.ofArray(...)` and `MemorySegmentStorage` construction also happen before
  `create(...)`; any unexpected failure from those completed JDK/storage contracts propagates and
  consumes no ID.
- Blank-label validation is deliberately not duplicated. The array and storage are allocated
  first, then `create(...)` allocates an ID, and the existing `Tensor` constructor rejects the
  blank label. The failed construction therefore consumes one ID; the unreachable heap allocation
  becomes eligible for garbage collection.
- On success, `create(...)` allocates the ID and reuses existing Tensor label and storage
  validation. The new storage is matching, exact-capacity, live, and writable by construction.
- Exhausted identifier space is observed only after successful heap allocation and storage
  wrapping. The existing exact exhaustion exception wins over delegated Tensor label validation.
  No ID is reused, and the unreachable heap allocation becomes eligible for garbage collection.

No public contract promises that failed allocation leaves JVM memory usage unchanged, that garbage
collection occurs immediately, or that an ID can be rolled back.

### Ownership and lifetime

- The factory allocates a Java primitive array and immediately creates its heap segment.
- The segment's Java 26 automatic scope keeps the array reachable and is always accessible from
  any thread. The segment cannot be closed by the tensor or caller.
- `MemorySegmentStorage` retains the exact heap segment and reports it alive. It remains the same
  borrowed/non-owning, non-`AutoCloseable` wrapper defined by task 0010.
- `Tensor` retains the exact storage object through its existing association and owns no close or
  release behavior.
- When the tensor, storage, and segment become unreachable, normal garbage collection may reclaim
  the segment and backing array. There is no deterministic lifetime endpoint or native resource.

The resulting object chain is:

```text
Tensor -> MemorySegmentStorage -> heap MemorySegment -> primitive array
                         automatic JDK scope keeps the heap base reachable
```

This is JVM-managed heap lifetime, not hidden arena ownership.

## Affected files

Expected production update:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorFactory.java`

Expected test updates:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryAllocationTest.java`

Expected documentation and planning updates during implementation:

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task specification
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless an inconsistency requires stopping:

- `Tensor`, `TensorDescriptor`, `HostTensorStorage`, `MemorySegmentStorage`, `DataType`, `Shape`,
  and `LayoutDescriptor` source and Javadocs
- `ARCHITECTURE.md`, focused architecture documentation, architecture tests, and
  `capabilities.md`

## Maximum scope

This task may create or modify at most:

- one existing production Java file;
- one existing focused test file;
- one new focused test file; and
- the five documentation and planning files listed as expected updates above.

The final implementation change therefore contains at most eight paths. Do not modify another
production/test file, Gradle, `AGENTS.md`, `ARCHITECTURE.md`, focused architecture documentation,
architecture tests, `capabilities.md`, another module, or unrelated documentation. Do not create a
task-0012B or task-0013 specification. If another file, type, method, dependency, owner, or
architecture clarification is required, stop and report it instead of expanding this task.

## Javadoc requirements

- Update `TensorFactory` type Javadoc so it no longer claims that the factory never allocates host
  memory. It must distinguish the new JVM-managed heap allocation overloads from existing
  caller-supplied borrowed storage creation and from native/runtime/backend allocation.
- Preserve complete Javadocs for all existing fields, constructor, create methods, and private ID
  allocator. Do not weaken task-0012 identity, concurrency, failure-consumption, or exhaustion
  documentation.
- Both new methods must document every parameter, nullability, resolved-layout precondition,
  exact span capacity, Java array limit, optional-label semantics, and exact-reference retention.
- Both methods must document a non-null fresh Tensor result, matching writable heap storage,
  factory-assigned opaque identity, and the fact that raw contents start at the JVM default zero
  representation without claiming a public zeros convenience.
- Both methods must document `NullPointerException` for null public arguments,
  `IllegalArgumentException` for unresolved layout, over-limit span, and delegated blank label,
  `IllegalStateException` for existing ID exhaustion, and `OutOfMemoryError` for JVM array
  allocation failure.
- Javadoc must explain which failures happen before ID allocation and which delegated failures
  consume an ID. It must not promise numeric order, gaplessness, deterministic reclamation, or
  rollback.
- Javadoc must explain the exhaustive carrier mapping, automatic heap-segment scope, reachability,
  cross-thread accessibility, no arena/close/external owner, and ordinary garbage-collected
  lifetime.
- Javadoc must not claim typed access, fill/population, BFLOAT16 conversion, boolean normalization,
  backing-array exposure, native allocation, descriptor/layout synthesis, runtime residency,
  backend storage, or execution support.
- The documentation-focused pass must review existing `Tensor`, `TensorDescriptor`,
  `HostTensorStorage`, `MemorySegmentStorage`, `DataType`, `Shape`, and `LayoutDescriptor`
  Javadocs. It must record why each remains accurate without edits or stop if a required correction
  falls outside scope.

## Acceptance criteria

- `TensorFactory` is the only modified production type; no production type, field, helper, nested
  type, package, or dependency is added.
- The factory declares exactly the two existing `create(...)` overloads, the two specified new
  `allocate(...)` overloads, and the existing private `nextTensorId()` method. No other public API
  or helper method appears.
- Existing private constructor, allocator fields, compare-and-set algorithm, ID scope,
  concurrency, final-candidate handling, permanent exhaustion, and `create(...)` behavior remain
  unchanged.
- `TensorFactoryTest` keeps every task-0012 behavioral test. Its exact-shape assertion is updated
  only to include the two new public methods and still rejects extra fields, constructors,
  interfaces, public methods, and allocator changes.
- `TensorFactoryAllocationTest` separately covers the allocation contract; no original allocator
  test is deleted, moved, relaxed, or rewritten to assume allocation behavior.
- Null descriptor and null label fail in exact order/messages before allocation and do not consume
  IDs.
- Static-unresolved and dynamic-unresolved descriptors fail with the exact unresolved-layout
  message, without descriptor replacement, array allocation, or ID consumption.
- A span of `Integer.MAX_VALUE + 1L` fails with the exact array-limit message before allocation or
  ID consumption. No native fallback or narrowed/wrapped length is attempted.
- Every one of the six `DataType` values produces exactly the specified primitive heap carrier,
  exact element capacity, checked byte size, writable/alive storage, and JVM-zero raw contents.
- `MemorySegment.heapBase()` in focused tests contains the expected primitive-array type and
  length. Production exposes no new backing-array API.
- Scalar, zero-sized-with-offset, contiguous, offset-dense, general strided, and broadcast layouts
  allocate exactly their referenced span. Tests explicitly cover capacities greater than logical
  element count.
- The allocated segment is a heap segment, not native or mapped; its automatic scope is alive and
  accessible from the allocating thread and a representative second thread. A tensor returned by
  a helper retains the segment and heap base without an external array owner or close handle.
- Blank label is rejected by existing Tensor validation after allocation and consumes one ID.
  Unresolved and over-limit failures do not consume IDs. Tests observe this without adding a
  production counter or reset hook.
- Source and bytecode order prove that JVM array allocation and storage wrapping precede
  `create(...)`; therefore an `OutOfMemoryError` before delegation consumes no ID. Tests must not
  attempt an unsafe or nondeterministic forced-OOM scenario.
- The full overload delegates to the existing complete `create(...)` method and does not invoke
  `nextTensorId()` or `new Tensor(...)` directly.
- Production contains exactly one exhaustive data-type switch and no `Arena`, native/mapped
  allocation, `AutoCloseable`, close, Cleaner/finalizer, array copy/fill, typed memory get/set,
  conversion, descriptor construction, layout inference, storage registry, runtime resource, or
  backend state.
- Reflection, `javap -p -c`, import checks, and source inspection confirm exact API shape,
  validation order, carrier allocation, `MemorySegment.ofArray(...)`, existing storage wrapping,
  create delegation, and absence of hidden ownership or population behavior.
- Complete `TensorFactory` Javadocs satisfy every requirement and generated model Javadoc renders
  both allocation methods and their ownership/failure contracts.
- A separate clean-context documentation-focused agent independently inspects the final source,
  tests, bytecode, generated Javadoc, test evidence, and diff; it finalizes TensorFactory Javadoc,
  Tensor API, glossary, task evidence/status, master-plan status, and roadmap status in the same
  overall change.
- The Tensor API moves only JVM-managed heap allocation from planned to current and does not claim
  flat/nested import, public constants, typed access, ranges/prefixes, random values, native
  allocation, runtime residency, or backend storage.
- The glossary updates the existing Tensor factory/host-storage wording rather than adding an
  unnecessary new project term. It distinguishes JVM-managed heap allocation from arena-owned
  native resources and keeps later factory families planned.
- Existing component Javadocs, focused architecture documents, architecture tests, and
  `capabilities.md` receive reasoned no-change conclusions because ownership, module boundaries,
  and the broader capability baseline are unchanged.
- Task 0012A is `Complete` in this task, the master plan, and roadmap only after implementation,
  tests, Javadoc, documentation review, scope review, and status synchronization pass. Task 0012B
  then becomes the next `Draft` frontier; tasks 0012B–0012F and 0013 remain Draft and no detailed
  task-0012B/task-0013 specification is created.
- No commit or push occurs.

## Tests / validation

Run after implementation and again after the separate documentation-focused pass:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryAllocationTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

Manually verify:

- the final diff contains only the eight permitted paths;
- reflection reports the unchanged factory fields/private constructor, two existing create
  overloads, two exact allocation overloads, unchanged private allocator, and no extra API;
- `javap -p -c` shows null checks, resolved-layout check, `Integer.MAX_VALUE` check, exhaustive
  primitive-array allocation, matching `MemorySegment.ofArray(...)`,
  `MemorySegmentStorage` construction, and delegation to complete `create(...)` in that order;
- focused allocation tests cover all carriers and zero values, scalar, zero-sized with offset,
  dense, offset, strided, broadcast, over-logical-count capacity, unresolved static/dynamic,
  array limit, heap lifetime/accessibility, label normalization, and ID side effects;
- original factory tests still cover descriptor/storage creation, null non-consumption, delegated
  failure consumption, ordinary concurrency, final-value concurrency, permanent exhaustion, and
  precedence without weakening;
- production imports contain no `Arena`, runtime/backend/resource, reflection, random, or external
  dependency type;
- source contains no native/mapped allocation, close path, owner field, typed read/write,
  population, conversion, copy/fill, descriptor construction, layout synthesis, or direct ID
  allocation from the new methods;
- generated Javadoc documents heap ownership/lifetime, exact span and carrier mapping, zeroed raw
  allocation, side-effect order, array limit, and all public failures;
- existing Tensor/storage/descriptor/layout/data-type/shape Javadocs are reviewed with reasoned
  no-change results unless an out-of-scope inconsistency requires stopping;
- the documentation-focused context follows
  `docs/developer-guide/documentation-rules.md`, applies General plus API/Javadoc style to Java/API
  work and Planning style to planning updates, inspects actual source/tests/diff, and records its
  identity, commands, outcomes, limitations, Javadoc review, glossary impact, and architecture/
  capability no-change rationale;
- local links and anchors in the five changed documentation/planning files resolve, fences are
  balanced, terminology agrees with the glossary, and all changed files lack trailing whitespace;
- task 0012A status matches the master plan and roadmap; 0012B is the next Draft frontier; later
  rows remain Draft; no task-0012B or task-0013 specification exists; and
- no commit or push occurs.

## Dependencies

- Task 0001 is complete and provides the exact six-value `DataType` set and byte widths.
- Task 0003 is complete and provides checked resolved referenced-element spans.
- Task 0007 is complete and provides explicit resolved/unresolved layout state.
- Task 0010 is complete and provides exact-size borrowed `MemorySegmentStorage` over heap segments.
- Task 0011 is complete and provides canonical Tensor storage compatibility and label validation.
- Task 0012 is complete and provides public construction, JVM-wide ID allocation, and the complete
  `create(...)` delegation path.
- The Java toolchain and release are 26. Java 26 `MemorySegment.ofArray(...)` uses an automatic
  scope that keeps its primitive array reachable and remains accessible from any thread without
  preview or incubator features.

## Follow-up tasks

- Task 0012B remains the next Draft task after this work. It owns flat typed import, carrier/type
  matching, logical-count checks, copying, and population on top of the allocation contract.
- Task 0012C remains Draft and owns rectangular nested-array inference and flattening after flat
  import.
- Task 0012D remains Draft and owns public scalar, zeros, ones, zeros-like, and ones-like
  conveniences after typed population exists.
- Task 0012E remains Draft and owns integer ranges and strict/cyclic prefix population.
- Task 0012F remains Draft and owns random-source/reproducibility policy and normal population.
- Task 0013 remains Draft after the factory families and owns minimal provenance.
- Runtime and concrete backend tasks continue to own runtime buffers, device/native allocation,
  residency, pooling, transfers, and deterministic resource lifetimes.

Do not create a detailed specification for any follow-up as part of task 0012A.

## Architecture impact

Expected impact: None.

The architecture already assigns `TensorFactory`, public Tensor state, and host-visible storage to
`modules/model`. Java heap arrays wrapped by the existing `MemorySegmentStorage` remain host memory,
not runtime device residency or backend-specific storage. The JDK automatic scope supplies safe
garbage-collected reachability without changing the wrapper's borrowed/non-closing contract.
Therefore no module boundary, dependency direction, lifecycle stage, storage ownership contract,
ADR, architecture test, or architecture document changes. If implementation reveals otherwise,
stop and report the conflicting rule and required decision before editing architecture files.

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
- docs/planning/modules/model/tasks/0003-layout-descriptor-model.md
- docs/planning/modules/model/tasks/0007-tensor-descriptor-model.md
- docs/planning/modules/model/tasks/0010-host-storage-abstraction.md
- docs/planning/modules/model/tasks/0011-public-tensor-skeleton.md
- docs/planning/modules/model/tasks/0012-tensor-factory.md
- docs/planning/modules/model/tasks/0012a-host-storage-allocation.md
- docs/api/tensor-api.md
- docs/glossary.md
- current production/tests for TensorFactory, Tensor, TensorDescriptor, LayoutDescriptor,
  HostTensorStorage, MemorySegmentStorage, DataType, and Shape
- root/model Gradle configuration only to confirm Java 26

Implement task 0012A exactly as specified. Modify only TensorFactory.java for production. Add
exactly allocate(TensorDescriptor) and allocate(TensorDescriptor, Optional<String>), with no helper
method or other public surface. Update TensorFactoryTest only for the deliberate exact-API shape and
add TensorFactoryAllocationTest for the allocation contract. Do not weaken or remove any existing
factory allocator/concurrency/exhaustion test.

Require a resolved layout and allocate exactly referencedElementSpan(), not logical element count.
Reject span above Integer.MAX_VALUE with the exact task message and do not fall back to native
memory. Map FLOAT64 to double[], FLOAT32 to float[], BFLOAT16 to short[], INT32 to int[], INT64 to
long[], and BOOL to byte[]. Wrap each array with its matching MemorySegment.ofArray overload and
the existing MemorySegmentStorage. Preserve JVM default zeroed raw contents. Delegate final
construction to existing create(descriptor, label, Optional.of(storage)); do not call nextTensorId
or new Tensor from either allocation method.

Follow the exact null, unresolved-layout, array-limit, allocation, segment, storage, and create
order. Do not duplicate blank-label or Tensor storage validation. Blank labels fail after heap
allocation and consume an ID through create. Preallocation validation and a JVM allocation failure
consume no ID. Do not catch OutOfMemoryError.

Use only JVM-managed primitive arrays. Do not add or modify a storage type, permits list, Arena,
native/off-heap/mapped allocation, external owner, AutoCloseable, close method, Cleaner/finalizer,
lease, reference counting, pooling, or runtime resource. Do not add typed reads/writes, backing-
array API, copies, conversions, BFLOAT16 conversion, BOOL normalization, fill loops, public zeros/
ones/scalar methods, flat or nested data imports, ranges, prefixes, random values,
descriptor/layout synthesis, provenance, operations, graph/compiler/runtime/backend state,
dependency, build option, or unrelated refactor.

Stop and report if implementation requires another production type/file, a storage or ownership
contract change, descriptor synthesis, native fallback, deterministic cleanup, architecture change,
or more than the eight permitted paths.

Add and finalize every Javadoc contract required by the task. Run both focused factory test suites,
all model tests, model Javadoc, full repository tests, git diff checks, and every reflection,
javap, import, carrier, zeroing, geometry, lifetime, ID-side-effect, documentation, scope, and
status check in the task. Do not force an unsafe OutOfMemoryError test; prove that side-effect order
through source and bytecode inspection.

After implementation and initial validation, hand the actual diff to a separate documentation-
focused agent or thread with a clean context in the same overall change. Keep task 0012A incomplete
until that pass finishes. The handoff must include this task, source/test/bytecode diff, Java 26
automatic-scope lifetime, carrier mapping, span and array limit, side-effect order, deferred
factory families, architecture constraints, expected Tensor API/glossary changes, component-
Javadoc no-change review list, and every validation command.

That documentation agent must independently read AGENTS.md, ARCHITECTURE.md,
docs/developer-guide/documentation-rules.md, the documentation profile index, General style,
API/Javadoc style, Planning style, Example format if an example changes, this task, final
source/tests/generated Javadoc, Tensor API, glossary, model master plan, roadmap, and the existing
Tensor/TensorDescriptor/HostTensorStorage/MemorySegmentStorage/DataType/Shape/LayoutDescriptor
contracts. It must inspect actual implementation and evidence rather than rely on the handoff. It
must finalize TensorFactory Javadoc, move only JVM-managed heap allocation into current API/glossary
language, preserve tasks 0012B-0012F as planned, review links/anchors/fences/whitespace and
terminology, record reasoned component-Javadoc and architecture/capability no-change conclusions,
and synchronize only the allowed planning files.

At the end, update this task, the model master plan, and roadmap for status/evidence. Record local
decisions, known limitations, exact validation evidence including documentation-agent identity and
results, implementation notes, and the canonical completion summary. Do not mark task 0012A
Complete until implementation, tests, Javadoc, independent documentation pass, scope review, and
status synchronization all pass. Task 0012B then remains the next Draft frontier without a detailed
specification. Do not commit or push.
```

## Local decisions

- The task is named **JVM-managed heap host storage allocation** rather than owning host storage.
  The factory creates the array, but no Synaptik object owns a closeable resource. Java 26's
  automatic heap-segment scope supplies reachability and cross-thread accessibility.
- No new storage implementation is needed. The one `MemorySegmentStorage` implementation already
  accepts heap segments, retains exact segment identity, validates exact capacity bytes, and owns
  no arena or close operation.
- Allocation requires a resolved layout. Treating a static unresolved descriptor as contiguous
  would silently perform layout inference and treating a dynamic shape numerically is impossible
  under the completed contracts.
- Referenced span is the physical-capacity rule because it is the completed descriptor of every
  referenced storage index. Logical count would under-allocate offset/strided layouts and
  over-allocate broadcast layouts.
- Primitive arrays impose an explicit `Integer.MAX_VALUE` element limit. This factory does not
  choose a native fallback or weaken the model's `long` shape/layout contracts.
- The carrier switch lives directly in the complete allocation method so the factory adds exactly
  two methods and no helper or per-data-type storage hierarchy.
- Allocation and population remain separate. JVM zeroing is an unavoidable primitive-array fact,
  not a public task-0012D zeros API.
- Delegating to `create(...)` preserves one ID allocator and one Tensor validation path. A blank
  label therefore allocates storage and consumes an ID; preallocation validation and allocation
  errors do not.

## Known limitations

- Only resolved layouts whose referenced span fits Java primitive-array length can be allocated.
- Practical JVM array-size and heap limits may be lower than `Integer.MAX_VALUE`; the JVM may throw
  `OutOfMemoryError`, which propagates unchanged before ID allocation.
- Allocation is heap-only and garbage-collected. There is no deterministic release, native/mapped
  allocation, pooling, or backend/runtime storage policy.
- The returned raw storage is writable and zero-initialized, but this task adds no typed access,
  import, conversion, fill, or public constant convenience.
- The backing array remains reachable through the heap segment but is not exposed through a new
  Synaptik API. JDK `MemorySegment.heapBase()` remains observable to callers of the existing raw
  `segment()` API.
- Mutation/version semantics remain undefined because the completed raw storage contract exposes
  writable memory without tracking writes.

## Validation evidence

- Implementation context `/root/implement_model_0012a` modified only `TensorFactory.java`, updated
  the existing factory API-shape test, and added `TensorFactoryAllocationTest.java`. It then handed
  the actual shared-tree source, tests, bytecode evidence, and task contract to canonical clean
  documentation context `/root/implement_model_0012a/review_model_0012a_docs`.
- The documentation context read the complete agent instructions, architecture contract, current
  focused architecture documents, documentation workflow and profile index, General style,
  API/Javadoc style, Planning style, Example format, planning guide, roadmap, capability baseline,
  model master plan, tasks 0012 and 0012A, Tensor API, glossary, final source/tests, generated
  Javadoc, and all requested component contracts. General plus API/Javadoc style governed Java and
  API work; Planning style governed task/master/roadmap synchronization. No example changed, so the
  Example format required no example rewrite or compilation. Aggregate reads that reached tool
  output limits were repeated as complete per-file or bounded-range reads before editing.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorFactoryTest` passed after documentation edits. Its
  XML report records 7 tests, 0 failures, 0 errors, and 0 skipped, including the exact utility/API
  shape and preserved task-0012 identity, concurrency, exhaustion, and failure-consumption checks.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorFactoryAllocationTest` passed after documentation
  edits. Its XML report records 6 tests, 0 failures, 0 errors, and 0 skipped, covering all carriers
  and raw zeros, scalar/empty/dense/offset/strided/broadcast spans, unresolved and array-limit
  failures, label/ID side effects, automatic-scope reachability, and cross-thread access.
- `./gradlew :modules:model:test` passed after documentation edits. Aggregation of 24 XML suites
  records 167 tests, 0 failures, 0 errors, and 0 skipped.
- `./gradlew :modules:model:javadoc` passed after documentation edits without warnings. Generated
  `TensorFactory.html` was inspected for both allocation overloads, exact span and six carriers,
  raw zero initialization, `Integer.MAX_VALUE`, automatic-scope reachability and cross-thread
  access, garbage-collected lifetime, no arena/owner/close, failure and ID ordering,
  `OutOfMemoryError`, and deferred typed/native/runtime/backend boundaries.
- The first generated-Javadoc search used a nonexistent module-qualified output path and failed
  before inspection. `rg --files modules/model/build/docs/javadoc` located the actual Gradle output
  path, and the corrected generated-page inspection above passed.
- `./gradlew test` passed after documentation edits with 36 actionable tasks, 1 executed and 35
  up-to-date. No architecture, backend-conformance, or integration test was added because no
  dependency rule, backend behavior, or end-to-end execution behavior changed.
- `javap -classpath modules/model/build/classes/java/main -p -c -s
  io.github.pho001.synaptik.model.tensor.TensorFactory` confirmed descriptor then label null checks,
  resolved-layout then `Integer.MAX_VALUE` checks, the six exact primitive arrays and matching
  `MemorySegment.ofArray(...)` calls, `MemorySegmentStorage` construction, and final delegation to
  `create(...)`. Neither allocation overload invokes `nextTensorId()` or constructs `Tensor`
  directly, proving predelegation failures consume no ID and heap work precedes blank-label or
  exhaustion failure. No unsafe forced-`OutOfMemoryError` test was attempted.
- Source, reflection-test, and import review confirmed exactly two allocator fields, one private
  constructor, two existing `create(...)` overloads, two exact `allocate(...)` overloads, and one
  private allocator. Production has one exhaustive data-type switch and no helper method, arena,
  native/mapped fallback, owner/close path, backing-array API, typed read/write, copy/conversion,
  fill/population, descriptor synthesis, layout inference, runtime resource, or backend state.
- Existing component Javadocs remain accurate without edits: `Tensor` still owns only metadata and
  a borrowed storage association; `TensorDescriptor` remains logical and allocation-free;
  `HostTensorStorage` remains the raw non-allocating boundary; `MemorySegmentStorage` already
  accepts exact heap segments without ownership or close behavior; `DataType` remains carrier-
  independent metadata; `Shape` remains logical geometry independent of array limits; and
  `LayoutDescriptor` remains storage-free resolved geometry whose referenced span is consumed by
  the factory. No out-of-scope correction was required.
- `docs/api/tensor-api.md` now treats only exact-span JVM-managed heap allocation as current and
  preserves flat/nested import, scalar/constants, ranges/prefixes, random population, typed access,
  copy/conversion/fill, and native/runtime/backend allocation as planned. `docs/glossary.md`
  revises the existing Tensor factory, Tensor, and host-storage language without adding a term and
  distinguishes automatic-scope factory heap segments from caller-supplied arena-backed storage
  and deterministic native resources.
- `ARCHITECTURE.md`, focused architecture documents, architecture tests, and `capabilities.md`
  remain unchanged. The architecture already authorizes model-owned `TensorFactory` and host
  storage; this task adds only a JDK heap allocation policy within the existing direction from
  `model.tensor` to `model.storage` and changes no ownership, lifecycle, dependency, backend,
  runtime, or capability-baseline rule.
- A targeted Ruby validator resolved 140 local links, including 58 heading anchors, across the five
  changed documentation/planning files with zero errors. Fence inspection found balanced backtick
  markers: 24 in Tensor API, 12 in this task, two in the master plan, two in the roadmap, and none
  in the glossary; no tilde fence is present. `rg -n '[[:blank:]]+$'` found no trailing whitespace
  in the eight allowed paths.
- Two initial Ruby link-check scripts failed before producing documentation results: the first used
  an interpolation-sensitive heading regular expression, and the corrected parser then exposed
  that the host Ruby lacks `Array#filter_map`. The final compatible `map`/`compact` validator is the
  successful 140-link/58-anchor result recorded above.
- `git diff --check` passed, and the untracked task/test files have no whitespace diagnostic.
  `git status --short --untracked-files=all` confirms exactly the eight permitted paths: factory,
  existing factory test, new allocation test, Tensor API, glossary, this task, model master plan,
  and roadmap. No commit or push was performed.
- Final status review marks task 0012A `Complete` in this task, master plan, and roadmap. Task 0012B
  is the next `Draft` frontier; tasks 0012B–0012F and 0013 remain `Draft`; no task-0012B or
  task-0013 specification exists.

## Implementation notes

- Added the two exact public allocation overloads directly to `TensorFactory` without a helper or
  another production type. The complete overload validates preallocation inputs, creates one
  data-type-matched primitive array at exact referenced span, wraps its automatic-scope segment in
  `MemorySegmentStorage`, and delegates to the existing complete `create(...)` path.
- Preserved the existing constructor delegation and full-range concurrent identifier allocator.
  Preallocation and JVM/storage-construction failures occur before ID allocation, while blank label
  and exhaustion remain delegated after heap allocation.
- Added focused allocation coverage while changing the task-0012 factory suite only for the two
  deliberate method signatures. Finalized factory Javadoc, Tensor API, glossary, and planning
  status/evidence through the required independent documentation pass.

## Completion summary

- Completed changes: added exact-span JVM-managed primitive-array heap allocation for all six data
  types through `TensorFactory`, focused tests, complete allocation Javadocs, API/glossary updates,
  and synchronized planning status/evidence.
- Files changed or created: `TensorFactory.java`, `TensorFactoryTest.java`,
  `TensorFactoryAllocationTest.java`, `docs/api/tensor-api.md`, `docs/glossary.md`, this task, the
  model master plan, and the roadmap.
- Tests and validation: focused suites passed 7/7 and 6/6; model tests passed 167/167 across 24
  suites; model Javadoc and full repository tests passed; bytecode, imports, API shape, carriers,
  zeros, geometry, lifetime/thread access, ID ordering, generated Javadoc, links/anchors, fences,
  whitespace, exact scope, and synchronized status checks passed.
- Documentation-agent review: complete in canonical context
  `/root/implement_model_0012a/review_model_0012a_docs` using General, API/Javadoc, and Planning
  profiles; no example changed.
- Documentation impact: only JVM-managed heap allocation is current; import/population, typed
  access, copy/conversion/fill, and native/runtime/backend allocation remain planned.
- Javadoc review: `TensorFactory` is final; the seven requested component Javadocs remain accurate
  without edits for the reasons recorded above.
- Glossary impact: existing Tensor factory, Tensor, and host-storage definitions now distinguish
  automatic-scope factory heap storage from caller arena lifetime and deterministic native
  resources; no new term was added.
- Architecture impact: None. No architecture document, ADR, architecture test, dependency,
  ownership, or lifecycle rule changed; the broader capability baseline remains unchanged.
- Unresolved issues: None.
- Follow-up required: None for task 0012A. Task 0012B remains the next separate `Draft` planning
  frontier without a detailed specification.

Status: Complete
