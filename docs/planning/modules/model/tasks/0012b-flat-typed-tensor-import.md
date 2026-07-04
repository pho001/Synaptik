# Task 0012B: Flat Typed Tensor Import

## Status

Complete

## Goal

Add a bounded public factory surface for creating dense-contiguous tensors from copied flat Java
primitive arrays. Each overload accepts exactly the carrier belonging to one `DataType`, validates
the source length against the descriptor's logical element count, allocates storage through the
completed JVM-managed heap allocation path, and populates the new storage without retaining the
caller's array.

This task imports leaf data only. It does not infer shapes, synthesize layouts, import nested
arrays, expose backing arrays, add tensor element access, or define view/scatter population.

## Scope

- Modify the existing `TensorFactory` and add six overloaded `fromFlatArray(...)` methods for
  `double[]`, `float[]`, raw BFLOAT16 `short[]`, `int[]`, `long[]`, and BOOL `byte[]`.
- Require a completed descriptor with resolved `DENSE_CONTIGUOUS` layout.
- Require the source length to equal `descriptor.shape().knownElementCount()` exactly.
- Require the source carrier's `DataType` to equal `descriptor.dataType()`.
- Allocate the destination through `TensorFactory.allocate(descriptor, label)` so task 0012A
  remains the single heap-allocation path and task 0012 remains the single ID-allocation path.
- Copy numeric and BFLOAT16 raw data into the destination segment without retaining the source.
- Normalize BOOL bytes during import: zero remains `0`; every non-zero value becomes canonical
  `1`.
- Preserve source encounter order as logical row-major element order.
- Update the existing exact-shape `TensorFactoryTest` and add one focused
  `TensorFactoryFlatImportTest`.
- During implementation, update the Tensor API, glossary, task evidence, model master plan, and
  roadmap through the required separate clean-context documentation pass.

## Out of scope

- nested arrays, reflection-based shape inference, rectangularity checks, ragged arrays, or task
  0012C behavior
- scalar, zeros, ones, zeros-like, ones-like, fill, range, prefix, or random convenience methods
- offset-dense, strided, broadcast, unresolved, or dynamic-layout import; scatter semantics;
  alias-conflict rules; view creation; or materialization
- descriptor construction, shape inference, layout synthesis, default data types, or mutation of
  descriptor metadata
- implicit numeric conversion, widening, narrowing, floating promotion, BFLOAT16 conversion from
  `float`, BOOL-to-numeric conversion, or numeric-to-BOOL conversion other than normalization of
  the matching BOOL `byte[]` carrier
- retaining a source array, exposing a backing array, zero-copy import, ownership transfer, or
  mutation/version tracking
- public typed reads/writes, bulk export, copies from tensors, conversion methods, or a general
  storage-access API
- a new storage type, native/off-heap/mapped allocation, `Arena`, close behavior, pooling, or
  backend/runtime storage
- provenance, operations, graph IDs, gradients, trainable/publication state, compiler, planning,
  prepare, runtime, engine, backend, device, residency, or execution behavior
- dependencies, preview/incubator features, Gradle changes, architecture changes, or another module
- a detailed specification for task 0012C, task 0013, or any later task

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md), especially model ownership of public
  Tensor construction and host storage and the exclusion of runtime/backend storage
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Architecture overview](../../../../architecture/overview.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md), especially typed flat import and copy semantics
- [Model master plan](../master-plan.md)
- [Task 0001](0001-data-type-model.md), which defines the six data types
- [Task 0003](0003-layout-descriptor-model.md), which defines layout kind and physical geometry
- [Task 0007](0007-tensor-descriptor-model.md), which defines logical descriptor invariants
- [Task 0010](0010-host-storage-abstraction.md), which defines raw host storage
- [Task 0011](0011-public-tensor-skeleton.md), which defines tensor/storage association
- [Task 0012](0012-tensor-factory.md), which defines public construction and identity allocation
- [Task 0012A](0012a-host-storage-allocation.md), which defines typed heap allocation
- [Tensor API](../../../../api/tensor-api.md) and [glossary](../../../../glossary.md)

## Legacy evidence and rejected coupling

Read-only legacy inspection covered `TensorDataFactory`, `TensorStorageAccess`, typed heap storage
classes, Tensor constructors, and flat/shaped data tests. Useful capability evidence is limited to
carrier-specific input, defensive copying, logical-count validation, BFLOAT16 raw-bit storage, and
canonical BOOL storage.

The new design rejects the legacy constructor explosion, mutable metadata, per-data-type storage
classes, nullable labels, reflection-based flattening, implicit floating conversions, raw mutable
array exposure, version counters in import code, runtime/backend coupling, and legacy scalar or
empty-shape conventions. Nested-array behavior remains task 0012C and is not copied here.

## Architecture constraints

- Production remains in `io.github.pho001.synaptik.model.tensor`; only `TensorFactory` changes.
- Package direction remains `model.tensor -> model.storage` plus existing foundational model
  packages. No reverse or cross-module dependency is introduced.
- Import creates a new tensor and new heap storage. The caller's source array is never retained,
  used as the destination backing array, or exposed through the result.
- A flat source describes independent logical values in row-major order. Therefore the descriptor
  must have resolved `LayoutKind.DENSE_CONTIGUOUS` geometry. Offset, sparse strided, and broadcast
  layouts require distinct scatter or alias semantics and are rejected.
- Resolved dense-contiguous layout implies a fully static shape, zero offset, canonical strides,
  and referenced span equal to known logical element count, including scalar and empty shapes.
- Source length is compared with the checked `long` logical element count before allocation. Java
  arrays have `int` length, so a descriptor requiring more elements cannot pass this check.
- Numeric carriers and BFLOAT16 `short[]` are copied bit-for-bit through temporary source
  `MemorySegment` values. BFLOAT16 shorts are raw format bits; this task performs no conversion.
- BOOL `byte[]` is logical input rather than raw unrestricted storage: `0` becomes canonical
  false byte `0`, and every non-zero byte becomes canonical true byte `1`.
- `TensorFactory.allocate(...)` remains the allocation path, and its delegation to `create(...)`
  remains the ID and Tensor-validation path. Flat import must not call `nextTensorId()`, construct
  `Tensor`, or construct `MemorySegmentStorage` directly.
- Population occurs before the new tensor is returned to the caller. The result is never publicly
  observable in a partially populated state.
- Input copying is not synchronized with concurrent caller mutation of the source. Callers must
  not mutate an array concurrently with import; no atomic snapshot guarantee is added.
- No existing storage, Tensor, descriptor, layout, or factory-allocation contract changes.
- If implementation requires general view scattering, conversion policy, another production
  type, a storage-contract change, native allocation, or an architecture/dependency change, stop
  and report the conflict.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.tensor` — owns `TensorFactory`, `Tensor`, and descriptors.
- `io.github.pho001.synaptik.model.storage` — supplies the destination raw segment through the
  tensor's attached `HostTensorStorage`.
- `io.github.pho001.synaptik.model.datatype` — supplies carrier-to-`DataType` matching.
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

| Source carrier | Required `DataType` | Meaning |
|---|---|---|
| `double[]` | `FLOAT64` | IEEE 754 binary64 values |
| `float[]` | `FLOAT32` | IEEE 754 binary32 values |
| `short[]` | `BFLOAT16` | raw BFLOAT16 bit patterns |
| `int[]` | `INT32` | signed 32-bit integer values |
| `long[]` | `INT64` | signed 64-bit integer values |
| `byte[]` | `BOOL` | zero is false; non-zero is normalized to true byte `1` |

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
matching `MemorySegment.ofArray(...)` overload, and invokes `importFlat` with the exact carrier
`DataType`. The helper retains no argument and adds no public API.

### Validation order and exact failures

Every public overload validates in this order:

1. Reject null `descriptor` with `NullPointerException` and exact message `descriptor`.
2. Reject null `label` with `NullPointerException` and exact message `label`.
3. Reject null `source` with `NullPointerException` and exact message `source`.
4. Create the temporary source `MemorySegment` and delegate to `importFlat`.

The private method then validates in this order, before destination allocation or ID allocation:

1. If `descriptor.dataType()` differs from `sourceDataType`, throw `IllegalArgumentException`
   with exact message
   `flat source data type must match descriptor: expected=<sourceDataType>, actual=<descriptorDataType>`.
2. If `descriptor.layout()` is empty, throw `IllegalArgumentException` with exact message
   `flat tensor import requires a resolved layout`.
3. If the resolved kind is not `DENSE_CONTIGUOUS`, throw `IllegalArgumentException` with exact
   message `flat tensor import requires dense-contiguous layout: actual=<layoutKind>`.
4. Read `descriptor.shape().knownElementCount()`. The completed descriptor/layout invariant makes
   it present. If absent because an impossible inconsistent contract is encountered, throw
   `IllegalStateException` with exact message
   `resolved tensor layout requires a fully static shape`.
5. Compare the required logical count with `sourceLength`. On mismatch throw
   `IllegalArgumentException` with exact message
   `flat source length must equal logical element count: required=<required>, actual=<actual>`.

These validation failures consume no tensor ID and allocate no destination storage. The temporary
source segment is only a non-owning view of the caller array and adds no copy or lifetime change.

### Allocation and population

After validation:

1. Call `Tensor tensor = allocate(descriptor, label)` exactly once.
2. Obtain the exact attached destination segment through
   `tensor.hostStorage().orElseThrow().segment()`.
3. For `FLOAT64`, `FLOAT32`, `BFLOAT16`, `INT32`, and `INT64`, copy exactly
   `sourceSegment.byteSize()` bytes into the destination at offset zero with JDK
   `MemorySegment.copy(...)` or an equivalent segment-to-segment bulk copy.
4. For `BOOL`, inspect each source byte and write canonical `0` or `1` into the corresponding
   destination byte. Use stable JDK `ValueLayout.JAVA_BYTE`; add no public typed-access API.
5. Return the exact populated tensor.

Dense-contiguous geometry guarantees destination byte size equals source segment byte size. The
source segment and source array are not stored. Mutating the source after return cannot change the
tensor. Source order, floating bit patterns, integer values, and BFLOAT16 raw bits are preserved.

### Failure side effects

- Null, data-type, layout, and source-length failures occur before destination allocation and
  before ID allocation; they consume no ID.
- A blank label passes import validation, then `allocate(...)` allocates storage and delegates to
  `create(...)`; Tensor rejects the label after ID allocation. The ID is consumed and no data copy
  occurs.
- Identifier exhaustion is observed inside `allocate(...)` after destination storage allocation
  and before import copying. The existing exhaustion exception propagates.
- `OutOfMemoryError` from destination allocation occurs before ID allocation and propagates.
- Bulk copying and BOOL normalization occur after successful tensor/ID creation. Their inputs and
  bounds are already validated; unexpected JDK errors propagate and the allocated ID remains
  consumed. No rollback is attempted.

## Valid and invalid scenarios

| Scenario | Result |
|---|---|
| Scalar `FLOAT64`, one `double` | Valid copied scalar |
| Empty dense shape, empty matching array | Valid zero-capacity tensor |
| Dense `[2,3]`, six matching carrier elements | Valid row-major copy |
| BOOL bytes `{0, -2, 3}` | Destination bytes `{0, 1, 1}` |
| BFLOAT16 `short[]` | Raw bits copied unchanged |
| Source mutated after return | Tensor data unchanged |
| Matching length but wrong carrier for descriptor | Data-type mismatch |
| Unresolved layout | Rejected before allocation |
| Offset-dense, strided, or broadcast layout | Rejected before allocation |
| Too short or too long source | Logical-count mismatch |
| Blank label with otherwise valid data | Delegated failure consumes one ID |

## Affected files

Expected production update:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorFactory.java`

Expected test updates:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryFlatImportTest.java`

Expected documentation and planning updates during implementation:

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task specification
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum scope

This task may create or modify at most one production file, one existing test, one new focused
test, and the five documentation/planning files above: eight paths total.

Do not modify another production/test file, Gradle, `AGENTS.md`, `ARCHITECTURE.md`, focused
architecture documentation, architecture tests, `capabilities.md`, another module, or unrelated
documentation. Do not create a task-0012C or task-0013 specification. Stop if another file, public
type, conversion policy, layout-scatter rule, or architecture clarification is required.

## Javadoc requirements

- Update `TensorFactory` type Javadoc to describe copied flat import while preserving factory ID,
  allocation, ownership, and deferred-capability boundaries.
- Document every overload fully: exact carrier, required `DataType`, dense-contiguous/layout and
  logical-count preconditions, label behavior, copying, BOOL normalization or BFLOAT16 raw bits,
  return value, side effects, and every failure.
- Document the private shared method, including why it validates logical row-major import and why
  only BOOL receives normalization.
- Do not describe numeric conversion, zero-copy ownership, general view population, nested import,
  typed tensor access, or factory parity as implemented.
- Record-generated or serialization claims are inapplicable; Tensor and factory identity behavior
  remains unchanged.

## Acceptance criteria

- `TensorFactory` adds exactly the six public overloads and one private shared method specified.
- Existing create/allocate methods, fields, ID allocator, and behavior remain unchanged.
- All six carriers require the exact matching descriptor data type.
- Import requires resolved `DENSE_CONTIGUOUS` layout and exact logical source length.
- Numeric and BFLOAT16 raw values are copied exactly; BOOL values are canonicalized to `0`/`1`.
- Source arrays are not retained and later source mutation cannot affect tensor storage.
- Scalar and empty imports work; offset, strided, broadcast, dynamic/unresolved, and length-mismatch
  inputs fail with exact messages before destination/ID allocation.
- Import calls `allocate(...)`, never `nextTensorId()`, `new Tensor`, or `new MemorySegmentStorage`.
- Existing allocator/concurrency/exhaustion and allocation tests are not weakened.
- Complete Javadoc, Tensor API, glossary, task/master/roadmap status, and separate documentation
  review are finished in the same change.
- Task 0012B is `Complete` only after all validation; task 0012C remains `Draft` without a spec.

## Tests / validation

Run before and after the documentation pass:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryFlatImportTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

Manually verify reflection and `javap -p -c -s` show the exact six overloads, one private helper,
unchanged allocator, validation order, allocation delegation, segment copy/BOOL normalization, and
no direct Tensor/storage construction. Verify production imports contain only current model/JDK
types and no reflection, arrays utility retention, graph/runtime/backend types, Arena, or native
access. Verify exact eight-path scope, links/anchors, fences, whitespace, status synchronization,
and absence of task-0012C/task-0013 specs.

## Dependencies

- Task 0012A provides dense heap allocation and is complete.
- Task 0012 provides identity allocation and public construction and is complete.
- Tasks 0001, 0003, 0007, 0010, and 0011 provide data type, layout, descriptor, storage, and Tensor
  contracts.

## Follow-up tasks

- Task 0012C will define nested typed import and rectangular shape inference.
- Task 0012D will define scalar and constant conveniences.
- Task 0012E will define ranges and prefix population.
- Task 0012F will define random creation and reproducibility policy.
- Later work owns typed export/access, mutation versions, views/scatter, provenance, and operations.

Do not create a detailed task-0012C specification as part of task 0012B.

## Architecture impact

Expected impact: None.

The architecture already assigns TensorFactory, host storage, and public tensor state to
`modules/model`. This task adds copied dense leaf data only and changes no module boundary,
dependency direction, lifecycle stage, storage ownership, or runtime/backend rule. If
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
  row-major leaf array.
- Six overloads preserve compile-time carrier typing without an `Object` API or per-dtype storage
  hierarchy. Optional label remains explicit; convenience overload proliferation is deferred.
- BFLOAT16 input is raw `short` bits. Converting `float[]` to BFLOAT16 is a conversion capability,
  not typed carrier matching.
- BOOL follows legacy capability evidence but stores a canonical representation: zero is false and
  any non-zero byte imports as one.
- Allocation occurs through task 0012A, so storage lifetime and ID side effects stay centralized.
- Copying after successful allocation means unexpected copy failure consumes the already allocated
  ID; validated bounds make such failure outside ordinary input behavior.

## Known limitations

- Only resolved dense-contiguous descriptors are accepted.
- No unlabeled convenience overloads are provided; callers pass `Optional.empty()`.
- Import is not atomic with concurrent mutation of the source array.
- There is no numeric conversion, nested shape inference, typed export, public element access, or
  mutation/version tracking.
- Very large logical counts cannot be represented by a Java source array and fail length
  validation before allocation.

## Validation evidence

Planning reviewed the authoritative architecture, planning/documentation rules, capability
baseline, completed factory/storage/descriptor/layout contracts, current tests, and legacy factory
evidence. The resulting design changes only model planning documents and introduces no architecture
decision.

- The implementation changed only `TensorFactory.java`, updated the existing exact-shape
  `TensorFactoryTest`, and added `TensorFactoryFlatImportTest`. Clean documentation-focused context
  `/root/implement_model_0012b/review_model_0012b_docs` independently read the architecture,
  documentation workflow, General style, API and Javadoc style, Planning style, Example format,
  planning chain, completed prerequisite tasks, final source/tests, generated Javadoc, complete
  diff, Tensor API, and glossary before finalizing documentation.
- The documentation pass applied General plus API and Javadoc style to `TensorFactory` and the
  Tensor API, Planning style to this task/master/roadmap, and Example format to the new complete
  flat-import example. It changed only Javadoc in `TensorFactory.java` among Java sources; method
  signatures, implementation behavior, and tests were not altered.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorFactoryTest` passed after documentation edits. XML
  reports 7 tests, 0 failures, 0 errors, and 0 skipped.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorFactoryFlatImportTest` passed after documentation
  edits. XML reports 10 tests, 0 failures, 0 errors, and 0 skipped, including the 2-by-3 INT32
  copy and identifier-exhaustion-before-population cases.
- `./gradlew :modules:model:test` passed. Aggregating 25 XML suites reports 177 tests, 0 failures,
  0 errors, and 0 skipped.
- `./gradlew :modules:model:javadoc` passed without warnings. Generated `TensorFactory.html`
  contains exactly six rendered flat-import overloads and documents exact carriers, dense layout
  and count requirements, source copying, raw BFLOAT16 bits, BOOL normalization, label/ID/allocation
  side effects, results, failures, and deferred boundaries. Source review confirms the private
  helper contract, which public generated Javadoc intentionally omits.
- `./gradlew test` passed for the complete repository with 36 actionable tasks and no failing task.
- `javap -classpath modules/model/build/classes/java/main -p -c -s
  io.github.pho001.synaptik.model.tensor.TensorFactory` confirmed the six exact public overloads,
  one private `importFlat` method, unchanged allocator, public null-check order, helper validation
  order/messages, exactly one `allocate(...)` call, attached-segment retrieval, numeric and raw
  BFLOAT16 `MemorySegment.copy`, BOOL `ValueLayout.JAVA_BYTE` normalization, and no lambda helper,
  direct Tensor construction, storage construction, or ID allocation inside `importFlat`.
- Production imports are exactly `DataType`, `LayoutKind`, existing storage contracts,
  `MemorySegment`, `ValueLayout`, `Objects`, `Optional`, and the two existing atomics. Source and
  bytecode inspection found no reflection, array retention, conversion, nested import, typed
  Tensor access, scatter/view policy, native allocation, graph, compiler, runtime, backend, or
  service state.
- The complete flat-import example in the Tensor API compiled with `javac -cp
  modules/model/build/classes/java/main` and ran with the model classes. It printed `mask` and
  `[0, 1, 1]`, confirming the documented normalization and post-return source independence. Two
  preliminary JShell runs produced the same values but failed while flushing host macOS
  preferences; the successful compile-and-run removed that environmental limitation from example
  validation.
- A targeted Ruby validator resolved 144 local Markdown links, including 58 heading anchors,
  across the five changed documentation/planning files. Two preliminary validator attempts failed
  before checking content because of Ruby interpolation and the host Ruby's missing
  `Array#filter_map`; the compatible rerun passed with zero errors. Fence counts are balanced,
  trailing-whitespace checks found no matches, and `git diff --check` passed.
- Existing `Tensor`, `TensorDescriptor`, `HostTensorStorage`, `MemorySegmentStorage`, `DataType`,
  `Shape`, and `LayoutDescriptor` Javadocs remain accurate without edits. They already define the
  stable tensor association, immutable logical descriptor, raw borrowed storage, exact segment
  wrapper, carrier-independent data type, logical shape/count, and resolved geometry/span
  contracts consumed by import; task 0012B changes none of those ownership or behavior contracts.
- `ARCHITECTURE.md`, focused architecture documentation, ADRs, architecture tests,
  backend-conformance tests, integration tests, and `capabilities.md` remain unchanged. Flat import
  is an already authorized model-owned TensorFactory/host-storage capability, changes no module or
  package direction, and introduces no backend or end-to-end execution behavior.
- The Tensor API now treats only copied flat typed import as current while preserving nested,
  constant, range/prefix, random, typed access/export, conversion, scatter/view, native, runtime,
  and backend behavior as planned. The glossary updates its existing Tensor and TensorFactory
  definitions; no new reusable domain term was introduced, so a separate flat-import entry would
  duplicate the API contract.
- Final scope review found exactly the eight permitted paths: `TensorFactory.java`,
  `TensorFactoryTest.java`, `TensorFactoryFlatImportTest.java`, Tensor API, glossary, this task,
  model master plan, and roadmap. No task-0012C or task-0013 specification exists. Task 0012B is
  synchronized as `Complete`; task 0012C remains the next `Draft` frontier. No commit or push was
  performed.

## Implementation notes

- Added the six exact carrier-specific `fromFlatArray(...)` overloads and one private shared
  `importFlat(...)` path to `TensorFactory` while preserving existing creation, allocation, and ID
  behavior.
- Added exact preallocation validation for carrier data type, resolved dense-contiguous layout,
  and logical source count. Numeric values and raw BFLOAT16 bits use segment bulk copy; BOOL bytes
  are normalized to canonical zero or one after allocation.
- Added focused coverage for all carriers, source isolation, BOOL normalization, scalar/empty
  values, exact failures and ID side effects, exhaustion before population, and unchanged gradient
  eligibility. The existing factory suite changed only its exact API-shape expectation.
- Finalized TensorFactory Javadoc, the Tensor API and compiled example, glossary status, and
  task/master/roadmap evidence through the required independent documentation pass.

## Completion summary

- Completed changes: Implemented and documented copied flat typed tensor import for all six model
  data types with exact carrier, dense-layout, logical-count, ownership, normalization, and failure
  side-effect contracts.
- Files changed or created: `TensorFactory.java`, `TensorFactoryTest.java`,
  `TensorFactoryFlatImportTest.java`, `docs/api/tensor-api.md`, `docs/glossary.md`, this task, the
  model master plan, and the roadmap.
- Tests and validation: Focused suites passed 7/7 and 10/10; all 177 model tests across 25 suites,
  model Javadoc, and full repository tests passed. Bytecode/API shape, imports and forbidden
  references, generated Javadoc, compiled example, links/anchors, fences, whitespace, exact scope,
  status synchronization, and `git diff --check` passed.
- Documentation-agent review: Complete in canonical clean context
  `/root/implement_model_0012b/review_model_0012b_docs` using General, API/Javadoc, Planning, and
  Example-format profiles.
- Documentation impact: Copied flat typed import is current; nested/constants/ranges/prefixes/
  random/access/export/conversion/view-scatter/native/runtime/backend behavior remains planned.
- Javadoc review: `TensorFactory` type, six overloads, and private helper are final. Existing
  Tensor, descriptor, storage, data type, shape, and layout Javadocs remain accurate for the
  reasons recorded above.
- Glossary impact: Existing Tensor and TensorFactory status/definitions now include flat import;
  no new term was needed because the behavior is a factory operation, not a new domain concept.
- Architecture impact: None. No architecture document, ADR, architecture test, dependency rule,
  module boundary, package direction, or capability-baseline decision changed.
- Unresolved issues: None.
- Follow-up required: None for task 0012B. Task 0012C remains the next separate `Draft` frontier
  without a detailed specification.

Status: Complete
