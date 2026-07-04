# Task 0012C: Nested Typed Tensor Import

## Status

Complete

## Goal

Add one bounded public factory path that accepts a rectangular multidimensional Java primitive
array, infers its fully static shape and exact model data type, flattens its values in logical
row-major order, and delegates final tensor creation to the completed flat typed import path.

This task provides readable nested input without weakening carrier typing. It accepts only true
primitive arrays with rank at least two, preserves raw BFLOAT16 bits, canonicalizes BOOL through
task 0012B, and never retains or mutates the caller's nested array graph.

## Scope

- Add exactly one public `TensorFactory.fromNestedArray(...)` method accepting a source object,
  optional label, and `requiresGrad` flag.
- Support multidimensional arrays whose ultimate primitive component is exactly `double`, `float`,
  `short`, `int`, `long`, or `byte`.
- Map those carriers exactly to `FLOAT64`, `FLOAT32`, `BFLOAT16`, `INT32`, `INT64`, and `BOOL`.
- Require runtime array rank of at least two. Rank-one primitive arrays remain the task-0012B flat
  import surface.
- Infer one static axis size per observable nested level and require a rectangular structure.
- Reject null subarrays, ragged lengths, unsupported leaf carriers, generic/object arrays, and
  empty non-leaf axes whose trailing extents cannot be observed.
- Accept an empty final leaf axis when all preceding axes are observable, such as `new int[2][0]`.
- Flatten values into a new matching primitive array in row-major encounter order.
- Construct a resolved dense-contiguous `TensorDescriptor` from the inferred type, shape, and
  `requiresGrad` value.
- Delegate the matching flat primitive array to the existing `fromFlatArray(...)` overload so
  allocation, ID assignment, copying, BOOL normalization, and label behavior remain centralized.
- Add one package-private same-package helper for structural validation, inference, and flattening.
- Update the exact `TensorFactory` API-shape test and add one focused nested-import test suite.
- During implementation, update Tensor API, glossary, task evidence, model master plan, and roadmap
  through the required separate clean-context documentation pass.

## Out of scope

- rank-one arrays, scalar values, boxed arrays, collections, streams, buffers, memory segments, or
  generic iterables
- `boolean`/`boolean[]` carriers; model BOOL host interchange remains normalized `byte`
- heterogeneous `Object[]`, arrays whose ultimate component is `Object` or a wrapper class, mixed
  leaf carriers, implicit conversion, numeric promotion, or default data-type selection
- caller-supplied shape, descriptor, layout, or data type; dynamic shape inference; symbolic
  dimensions; reshape; broadcasting; offset, strided, or view layout population
- inferring unobservable trailing extents after an empty non-leaf axis
- accepting or repairing ragged structures, padding rows, truncating rows, cyclic fill, prefix
  fill, or scatter into arbitrary geometry
- zero-copy import, retaining source arrays, exposing the intermediate flat array, typed Tensor
  access/export, mutation/version tracking, or concurrent-source snapshot guarantees
- task 0012D constants, scalars, zeros, ones, zeros-like, or ones-like behavior
- task 0012E ranges or strict/cyclic prefix population
- task 0012F random generation or reproducibility policy
- native/off-heap/mapped allocation, `Arena`, ownership/close behavior, pooling, or new storage types
- provenance, expression operations, graph IDs, gradients beyond descriptor eligibility,
  publication state, compiler, planning, prepare, runtime, engine, backend, device, residency, or
  execution behavior
- dependencies, preview/incubator features, Gradle changes, architecture changes, another module,
  or a detailed specification for task 0012D or any later task

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
- [Model capability baseline](../capabilities.md), especially supported nested Java arrays and the
  six host carrier mappings
- [Model master plan](../master-plan.md)
- [Task 0001](0001-data-type-model.md), which defines the six data types and differentiability
- [Task 0002](0002-shape-and-dimension-model.md), which defines static shapes and checked counts
- [Task 0003](0003-layout-descriptor-model.md), which defines dense-contiguous geometry
- [Task 0007](0007-tensor-descriptor-model.md), which defines descriptor validation
- [Task 0012](0012-tensor-factory.md), which defines public construction and identity allocation
- [Task 0012A](0012a-host-storage-allocation.md), which defines JVM heap allocation
- [Task 0012B](0012b-flat-typed-tensor-import.md), which defines carrier-specific flat import
- [Tensor API](../../../../api/tensor-api.md) and [glossary](../../../../glossary.md)

## Legacy evidence and rejected coupling

Read-only legacy inspection covered `Tensor`, `TensorArrayData`, `TensorDataFactory`,
`TensorStorageAccess`, `TensorConstructorDataTypeTest`, and the legacy Tensor API/README. The useful
capability is ergonomic construction from an arbitrary-rank nested Java primitive array with
inferred shape and row-major flattening.

The legacy implementation accepted `Object`, inspected only the first branch for shape, required
non-empty levels accidentally, flattened only `double` leaves, permitted explicit conversion to
another floating data type, and did not validate rectangularity before copying. Empty prefixes
could fail with incidental indexing exceptions, ragged data could overflow or underfill the flat
array, and Tensor constructors mixed import with graph/autograd state.

The new task keeps arbitrary-rank primitive-array convenience but rejects those couplings and
ambiguities. Runtime primitive carrier determines `DataType` exactly; every observable branch is
validated; empty-shape policy is explicit; construction remains in `TensorFactory`; and graph,
autograd, backend, and runtime state remain absent.

## Architecture constraints

- Production remains in `io.github.pho001.synaptik.model.tensor`. The public entry belongs to
  `TensorFactory`; structural array handling belongs to one package-private helper in the same
  package.
- Package direction remains `model.tensor` toward existing `datatype`, `shape`, `layout`, and
  `storage` foundations. No reverse or cross-module dependency is introduced.
- `Object` is used only because Java has a distinct runtime class for every primitive-array rank
  and no finite overload family can express arbitrary rank. The method must reject non-arrays,
  rank-one arrays, generic object arrays, wrapper arrays, and unsupported primitive leaves.
- The source runtime class supplies declared rank and ultimate primitive carrier. The implementation
  must not infer type from values, inspect boxed numbers, or select a default type.
- Shape inference validates every reachable subarray. It must not trust only the first branch.
- A zero length at the final primitive-leaf axis is observable and valid. A zero length at an
  earlier axis hides all following extents and is rejected rather than inventing sizes.
- Inferred sizes are non-negative `long` dimensions. Checked logical count must fit a Java array
  because flattening creates one primitive array; no native or chunked fallback is permitted.
- Flattening creates a fresh primitive array and uses row-major depth-first encounter order. The
  caller's arrays remain owned and mutable by the caller and are neither retained nor modified.
- The helper constructs only fully static shape and contiguous layout metadata. It performs no
  symbolic inference, graph-wide inference, layout policy, or materialization decision.
- Descriptor construction remains the authority for `requiresGrad` eligibility. Do not duplicate
  differentiability rules in the nested-array helper.
- Final creation dispatches to exactly one matching task-0012B `fromFlatArray(...)` overload. The
  nested path must not call `allocate`, `create`, `nextTensorId`, `new Tensor`, or
  `new MemorySegmentStorage` directly.
- Input inspection and copying are not synchronized with caller mutation. Callers must not mutate
  any source level concurrently; no atomic deep-snapshot guarantee is added.
- If implementation needs a public shape-inference type, general reflection utility, conversion
  policy, another storage contract, native allocation, architecture change, or another module,
  stop and report the conflict.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.tensor` — owns Tensor construction, descriptors, and the new
  nested-import boundary.
- `io.github.pho001.synaptik.model.datatype` — supplies the exact primitive-carrier mapping.
- `io.github.pho001.synaptik.model.shape` — supplies inferred fully static shapes.
- `io.github.pho001.synaptik.model.layout` — supplies resolved dense-contiguous geometry.

Packages added or changed:

- No package is added. Only the existing `model.tensor` package changes.

Type placement:

- `io.github.pho001.synaptik.model.tensor.TensorFactory` — receives the single public nested-array
  entry and delegates its bounded implementation.
- `io.github.pho001.synaptik.model.tensor.NestedTensorArray` — package-private final helper for
  runtime carrier classification, rectangular shape validation, checked flattening, descriptor
  construction, and dispatch to the matching flat-import overload.
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
to `NestedTensorArray`. Do not add unlabeled overloads, explicit-`DataType` overloads, rank-specific
overloads, generic methods, varargs, boxed-array overloads, or nested-array constructors on
`Tensor`.

The exact runtime carrier mapping is:

| Ultimate primitive component | Inferred `DataType` | Flat carrier |
|---|---|---|
| `double` | `FLOAT64` | `double[]` |
| `float` | `FLOAT32` | `float[]` |
| `short` | `BFLOAT16` raw bits | `short[]` |
| `int` | `INT32` | `int[]` |
| `long` | `INT64` | `long[]` |
| `byte` | `BOOL` logical bytes | `byte[]` |

### Package-private helper

Add one package-private final `NestedTensorArray` class in `model.tensor`.

It must:

- have no public or protected member and no mutable static state;
- have a private zero-argument constructor;
- expose exactly one package-private static entry:

  ```java
  static Tensor importArray(
          Object source,
          Optional<String> label,
          boolean requiresGrad)
  ```

- use private implementation methods or private immutable analysis values only when needed;
- classify source rank and ultimate primitive component from its runtime array class;
- validate the entire reachable nested structure before allocating the flat carrier;
- infer a `Shape`, validate checked count and Java-array capacity, flatten into one matching
  primitive array, construct a contiguous `TensorDescriptor`, and dispatch to task 0012B; and
- retain no source reference after the public call returns.

Do not add a reusable public reflection utility, public nested-array descriptor, public flattening
result, cache, registry, service, thread-local state, or source-retaining object.

### Structural model

`source.getClass()` must represent an array of declared rank at least two. Repeated
`Class.getComponentType()` calls determine rank and ultimate component type without examining a
sample value. This preserves primitive carrier identity even for an empty leaf.

For each array encountered at axis `a`:

1. Its length establishes the expected size for axis `a` when no earlier branch established it.
2. Otherwise its length must equal the established size.
3. If `a` is not the final leaf axis, every element must be a non-null subarray of the declared
   next-level array type.
4. If a non-final axis length is zero, reject because later extents cannot be observed.
5. At the final axis, the primitive array length may be zero.

Paths in diagnostics use zero-based bracket notation: root is `[]`, its second child is `[1]`, and
the third child of that array is `[1][2]`.

Examples:

| Source | Result |
|---|---|
| `new double[][] {{1, 2}, {3, 4}}` | `FLOAT64`, shape `[2, 2]`, flat `[1, 2, 3, 4]` |
| `new int[2][0]` | `INT32`, shape `[2, 0]`, empty flat carrier |
| `new float[0][3]` | rejected; axis 0 is empty and axis 1 is unobservable |
| `new long[2][0][4]` | rejected; axis 1 is empty and axis 2 is unobservable |
| `new byte[][] {{0, -2}, {3, 0}}` | `BOOL`, shape `[2, 2]`, stored `[0, 1, 1, 0]` |
| `new double[][] {{1}, {2, 3}}` | rejected as ragged at axis 1 |
| `new double[][] {null, {1}}` | rejected for null subarray at path `[0]` |

### Validation order and exact failures

The public method validates before helper delegation:

1. Null `source` throws `NullPointerException` with exact message `source`.
2. Null `label` throws `NullPointerException` with exact message `label`.

The helper then validates in this order before flat-carrier, destination, or ID allocation:

1. A non-array source throws `IllegalArgumentException` with exact message
   `nested tensor source must be an array: actual=<runtimeClassName>`.
2. Declared array rank below two throws `IllegalArgumentException` with exact message
   `nested tensor source must have rank at least 2: actual=<rank>`.
3. An ultimate component other than the six supported primitives throws
   `IllegalArgumentException` with exact message
   `nested tensor source leaf carrier is unsupported: <componentTypeName>`.
4. Traverse arrays in row-major depth-first order. A null required subarray throws
   `IllegalArgumentException` with exact message
   `nested tensor source contains null subarray at path <path>`.
5. A subarray with a different length from the first established length at its axis throws
   `IllegalArgumentException` with exact message
   `nested tensor source is ragged at axis <axis>, path <path>: expected=<expected>, actual=<actual>`.
6. A zero-length non-final axis throws `IllegalArgumentException` with exact message
   `nested tensor source cannot infer dimensions after empty axis <axis> at path <path>`.
7. Checked logical element-count overflow remains `ArithmeticException`. A non-overflowing count
   above `Integer.MAX_VALUE` throws `IllegalArgumentException` with exact message
   `nested tensor element count exceeds Java array limit: required=<required>, maximum=2147483647`.

All failures above consume no tensor identifier and allocate no destination storage. The helper may
allocate bounded traversal metadata, but it must not allocate the flat carrier before complete
structural validation succeeds.

After structural validation:

1. Allocate exactly one flat primitive array of the inferred logical count.
2. Copy primitive leaves into it in row-major order without boxing individual numeric values.
3. Construct `Shape.of(inferredSizes)` and `LayoutDescriptor.contiguous(shape)`.
4. Construct `TensorDescriptor(inferredDataType, shape, Optional.of(layout), requiresGrad)`.
5. Dispatch the flat carrier to exactly one matching `TensorFactory.fromFlatArray(...)` overload.

For non-differentiable data types, `requiresGrad=true` is rejected by `TensorDescriptor` before
destination or ID allocation. It may occur after the intermediate flat carrier has been allocated;
the helper must not duplicate descriptor validation solely to avoid that allocation.

A blank label reaches task 0012B after inference and flattening. It therefore allocates destination
storage and consumes one identifier through existing Tensor validation before failing. Identifier
exhaustion occurs after destination allocation and before final flat copy, exactly as in task 0012B.
Neither failure exposes or retains the intermediate flattened carrier.

### Flattening and ownership

- Numeric carriers and BFLOAT16 raw shorts are copied unchanged into the intermediate flat array.
- BOOL source bytes are copied raw into the intermediate array; task 0012B performs the one
  canonical zero/non-zero normalization while copying into tensor storage.
- Primitive leaf copying should use typed `System.arraycopy` or equivalent bulk copy. Do not box
  each numeric element, convert through `double`, or use a generic numeric list.
- The nested source, every subarray, and the intermediate flat array remain implementation inputs;
  none is returned, stored in Tensor metadata, or retained by destination storage.
- Later caller mutation of any source level cannot change the returned tensor.
- Concurrent caller mutation during inspection or flattening has no snapshot guarantee and is not
  supported.

## Affected files

Expected implementation and tests:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorFactory.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/NestedTensorArray.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryNestedImportTest.java`

Expected documentation and planning updates in the same overall change:

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/tasks/0012c-nested-typed-tensor-import.md`
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum scope

This task may create or modify at most the nine paths listed under Affected files.

No existing production Java contract other than `TensorFactory` may change. `TensorFactoryTest`
may change only for the deliberate exact API-shape expectation. If another production type, test,
documentation file, architecture file, build file, module, or dependency is needed, stop and
propose a separate follow-up or architecture decision.

## Javadoc requirements

- Update `TensorFactory` type Javadoc to distinguish completed flat and nested import.
- Fully document `fromNestedArray(...)`, including why `Object` is necessary for arbitrary rank,
  supported runtime carriers, rank and rectangularity, inferred shape/type/layout, empty-axis
  policy, raw BFLOAT16 and BOOL behavior, source ownership, concurrent mutation limitation,
  `requiresGrad`, label/ID/allocation side effects, result, and every failure category.
- Document every parameter with nullability, ownership, and semantic constraints; document the
  non-null result and all expected exceptions.
- Document package-private `NestedTensorArray` and its package-private entry with purpose,
  invariants, traversal/flattening behavior, ownership, parameters, result, and failures.
- Review existing `Tensor`, `TensorDescriptor`, `Shape`, `LayoutDescriptor`, `DataType`,
  `HostTensorStorage`, `MemorySegmentStorage`, and task-0012B Javadocs. Change none unless an
  in-scope stale statement can be corrected without changing behavior; otherwise stop on an
  out-of-scope discrepancy and record reasoned no-change conclusions.

## Acceptance criteria

- `TensorFactory` adds exactly the one public method specified and no other public surface.
- One package-private helper implements only nested primitive-array inference and flattening.
- All six supported ultimate primitive carriers infer the exact matching `DataType`.
- Runtime rank must be at least two; unsupported, boxed, generic, and rank-one sources fail exactly.
- Every reachable branch is validated; ragged and null-subarray failures include exact axis/path
  diagnostics and occur before flat/destination/ID allocation.
- Empty final leaf axes work; empty non-final axes fail because trailing dimensions are unknown.
- Inferred shape is fully static, layout is resolved dense-contiguous, and `requiresGrad` is
  preserved subject to existing descriptor eligibility.
- Flattening preserves row-major order, raw numeric/BFLOAT16 bits, source independence, and task-
  0012B BOOL normalization.
- Final creation invokes one matching flat-import overload and never directly allocates storage or
  identity and never constructs `Tensor`.
- Scalar/flat import, constants, range/prefix, random, typed access/export, storage, Tensor,
  descriptor, layout, allocator, and graph behavior remain unchanged.
- Complete Javadoc, Tensor API, glossary, task/master/roadmap status, and separate documentation
  review are finished in the same change.
- Task 0012C is `Complete` only after all validation; task 0012D remains `Draft` without a spec.

## Tests / validation

Run before and after the documentation pass:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryNestedImportTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

Focused tests must cover:

- exact factory API and helper visibility/state shape;
- rank-two and rank-three row-major imports for all six carriers;
- raw floating/BFLOAT16 preservation, BOOL normalization, and post-return source mutation;
- inferred `DataType`, `Shape`, dense layout, label, and `requiresGrad`;
- accepted empty final axis and exact rejection of ambiguous empty non-final axes;
- null argument order, non-array, rank-one, unsupported primitive, boxed/generic array, null
  subarray, and ragged array failures with exact messages;
- no ID consumption for structural and descriptor-eligibility failures;
- delegated blank-label consumption and identifier-exhaustion behavior; and
- preservation of all existing task-0012B tests.

Manually verify reflection and `javap -p -c -s` show the exact new public method, package-private
helper, existing methods/allocator unchanged, public null order, helper dispatch into flat import,
and no direct Tensor/storage/ID construction. Verify production imports contain only current
model/JDK types and no collections used as value buffers, graph/runtime/backend types, Arena,
native access, conversion utilities, or service state. Verify exact nine-path scope, generated
Javadoc, documentation examples, links/anchors, fences, whitespace, status synchronization, and
absence of task-0012D/task-0013 specs.

## Dependencies

- Task 0012B provides the exact typed flat-import and BOOL-normalization boundary and is complete.
- Task 0012A provides dense heap allocation and is complete.
- Tasks 0001, 0002, 0003, 0007, and 0012 provide data type, shape, layout, descriptor, and factory
  identity contracts.

## Follow-up tasks

- Task 0012D will define scalar and constant creation, zeros, ones, zeros-like, and ones-like.
- Task 0012E will define integer ranges and strict/cyclic prefix population.
- Task 0012F will define random creation and reproducibility policy.
- Later work owns typed export/access, mutation versions, views/scatter, provenance, and operations.

Do not create a detailed task-0012D or later specification as part of task 0012C.

## Architecture impact

Expected impact: None.

The architecture already assigns TensorFactory, shape/data-type/layout model, and host storage to
`modules/model`. This task adds copied nested leaf data only and changes no module boundary,
dependency direction, lifecycle stage, storage ownership, runtime/backend rule, or compiler
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
  objects or boxed values.
- Carrier determines `DataType`; there is no explicit type argument and no conversion. This is
  stricter than the legacy double-to-floating conversion path and consistent with flat import.
- Rank one remains flat import. Nested import begins at rank two so the two APIs have non-overlapping
  responsibilities.
- Rectangularity validates the complete reachable source rather than trusting the first branch.
- Empty final leaves are valid because their zero extent is observable. Earlier empty axes are
  rejected because Java runtime objects do not retain requested trailing extents when no child
  array exists.
- Flattening uses a fresh typed carrier and then flat import performs the destination copy. The
  extra bounded copy centralizes allocation, BOOL normalization, label behavior, and ID semantics.
- Package-private `NestedTensorArray` prevents `TensorFactory` from becoming a traversal god class
  while avoiding a new public concept or generic utility package.

## Known limitations

- Only multidimensional primitive arrays with one of the six exact leaf carriers are accepted.
- Empty non-final axes are rejected even when source code used a multidimensional allocation whose
  trailing dimensions were syntactically specified; those extents are not recoverable from the
  empty runtime object graph.
- Nested import performs one intermediate flattening allocation and one destination copy.
- Import is not atomic with concurrent caller mutation of any source level.
- No conversion, boxed values, collections, dynamic shape, custom layout, typed export, or general
  view/scatter population is provided.

## Validation evidence

Implementation and the independent documentation pass reviewed the authoritative architecture,
planning/documentation rules, capability baseline, completed factory/descriptor/shape/layout and
storage contracts, final source and tests, generated Javadoc, and the task-0012B flat-import
boundary. The separate documentation-focused context was
`/root/implement_model_0012c/review_model_0012c_docs`. It applied General style, API and Javadoc,
Planning, and Example format profiles and independently finalized Javadoc, Tensor API, glossary,
planning status, and evidence against the actual implementation.

- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorFactoryTest` — passed; JUnit XML reports 7 tests,
  0 failures, 0 errors, and 0 skipped.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorFactoryNestedImportTest` — passed; JUnit XML reports
  10 tests, 0 failures, 0 errors, and 0 skipped.
- `./gradlew :modules:model:test` — passed; 26 JUnit XML suites report 187 tests, 0 failures,
  0 errors, and 0 skipped.
- `./gradlew :modules:model:javadoc` — passed with generated public Javadoc inspected for the
  exact `fromNestedArray(Object, Optional<String>, boolean)` signature, rendered ownership,
  diagnostics, and side effects. Package-private `NestedTensorArray` is correctly absent from the
  public generated index; its complete source Javadoc was reviewed independently.
- `./gradlew test` — passed; the final standalone run reported 36 actionable lifecycle tasks, all
  up-to-date after the preceding focused validation.
- The complete nested-import API example was compiled and run directly from standard input with
  `java --source 26 --class-path modules/model/build/classes/java/main /dev/stdin`; it exited 0 and
  printed `FLOAT32`, `Shape[2, 2]`, `DENSE_CONTIGUOUS`, `true`, `weights`, and
  `[1.0, 2.0, 3.0, 4.0]` as documented.
- `javap -classpath modules/model/build/classes/java/main -p -c -s` confirmed that
  `TensorFactory` has exactly the one new public nested method, checks `source` before `label`, and
  delegates directly to the helper. Existing public methods, private flat importer, two private
  final allocator fields, private constructor, and private ID allocator remain unchanged.
- The same `javap` command confirmed that package-private final `NestedTensorArray` has no fields,
  nested classes, public members, or protected members; its only non-private entry is
  `importArray(Object, Optional, boolean)`. Bytecode validates the structure before the primitive
  flat allocation, uses `System.arraycopy` for leaves, constructs `Shape`, contiguous layout, and
  descriptor in order, and dispatches to exactly one matching flat overload. It contains no direct
  `allocate`, `create`, `nextTensorId`, `new Tensor`, or `new MemorySegmentStorage` reference.
- Production imports are limited to current model foundations and JDK reflection/utility types.
  Review found no collection value buffer, graph/compiler/planning/runtime/backend type, arena,
  native access, conversion utility, service state, or added dependency.
- A targeted local Markdown checker validated all five changed documentation/planning files: 147
  local links, including 58 anchors, with 0 errors. Fence checks found balanced fences; trailing-
  whitespace review found 0 findings; `git diff --check` passed.
- Scope review found exactly the nine permitted repository paths, including the pre-existing
  planning changes. There is no architecture, focused architecture-documentation, architecture-
  test, build, dependency, capability-baseline, other-module, backend-conformance, or integration-
  test change. The capability baseline already selects nested Java arrays, so no baseline update
  was needed. This model-only construction path changes no backend or end-to-end execution
  behavior, so backend-conformance and integration tests require no focused additions.
- Existing `Tensor`, `TensorDescriptor`, `DataType`, `Shape`, `LayoutDescriptor`,
  `HostTensorStorage`, and `MemorySegmentStorage` Javadocs remain accurate: nested import composes
  their existing identity, gradient-eligibility, static-shape, dense-layout, carrier-width, and
  borrowed-storage contracts without changing them. Existing descriptor creation, allocation,
  flat-import, and private allocator Javadocs also remain accurate because the new path delegates
  to them without changing their direct contracts. No out-of-scope Javadoc correction was needed.
- Tensor API and glossary now classify only rectangular nested primitive-array import as current;
  constants, ranges/prefixes, random generation, and later capabilities remain planned. Task,
  model master plan, and roadmap agree that 0012C is `Complete`, 0012D is the next `Draft`
  frontier, and no task-0012D detailed specification exists.

## Implementation notes

- Added the one public `TensorFactory.fromNestedArray(Object, Optional<String>, boolean)` entry and
  kept its null checks at the public boundary.
- Added package-private final `NestedTensorArray` for runtime rank/carrier classification, complete
  rectangular/null and empty-axis validation, checked shape/capacity inference, typed row-major
  flattening, exact descriptor construction, and one matching flat-import dispatch.
- Added API-shape coverage to `TensorFactoryTest` and the focused ten-test nested-import suite,
  including all six carriers, rank two and three, raw floating/BFLOAT16 values, BOOL normalization,
  source independence, empty-axis policy, exact diagnostics, gradient eligibility, and ID effects.
- Finalized `TensorFactory` and helper Javadocs. In particular, the factory type contract now
  identifies nested import as the bounded descriptor-synthesis exception instead of incorrectly
  claiming that no factory path constructs descriptors.
- Updated Tensor API and glossary current-vs-planned language and added one complete verified
  nested-import example. Architecture explanations and tests need no update because module
  ownership and dependency rules did not change.

## Completion summary

- Completed changes: implemented exact-carrier rectangular nested primitive-array import with
  static dense descriptor inference, row-major copied ownership, flat-import delegation, and the
  specified validation and ID side effects.
- Files changed or created: `TensorFactory.java`, `NestedTensorArray.java`, `TensorFactoryTest.java`,
  `TensorFactoryNestedImportTest.java`, `docs/api/tensor-api.md`, `docs/glossary.md`, this task,
  model `master-plan.md`, and `docs/planning/roadmap.md`.
- Tests and validation: focused tests 7/7 and 10/10; all model tests 187/187; model Javadoc and full
  repository tests passed; bytecode, imports, generated docs, example, links/anchors, fences,
  whitespace, exact scope, and status synchronization passed.
- Documentation-agent review: completed by
  `/root/implement_model_0012c/review_model_0012c_docs` using General, API/Javadoc, Planning, and
  Example format profiles.
- Documentation impact: Tensor API and glossary now document nested import as implemented; no
  architecture or capability-baseline document changed.
- Javadoc review: new and affected factory/helper contracts were finalized; adjacent tensor,
  descriptor, data-type, shape, layout, storage, flat-import, and allocator contracts were reviewed
  and require no change.
- Glossary impact: current implementation status, Tensor, and Tensor factory entries were updated;
  no separate reusable term was needed.
- Unresolved issues: None.
- Follow-up required: None. Task 0012D remains the next Draft planning frontier without a detailed
  specification.

Status: Complete
