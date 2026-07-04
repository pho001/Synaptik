# Task 0012E: Range and Prefix Population

## Status

Complete

## Goal

Add deterministic factory population for exclusive-end integer ranges and for tensors populated
from a strict or cyclic prefix of a typed flat Java primitive array. Every result is a new
dense-contiguous leaf tensor whose descriptor, storage, and identity are produced through the
completed model factory paths.

This task preserves the useful legacy range and fixture-data capabilities while replacing legacy
implicit `double[]` conversion with exact carrier-to-`DataType` matching. It does not add general
fill, conversion, typed tensor access, or an executable Range operation.

## Scope

- Add `INT32` and `INT64` exclusive-end integer range creation with non-zero positive or negative
  steps.
- Reject empty ranges and step directions that cannot advance from the inclusive start toward the
  exclusive end.
- Calculate range length without signed overflow and reject results above the Java array limit.
- Add strict flat-prefix creation for all six primitive carriers. A strict source may contain
  extra values, but it must contain at least the requested logical element count.
- Add cyclic flat-prefix creation for all six primitive carriers. A short non-empty source repeats
  from its beginning until the requested logical element count is reached.
- Accept an empty source for an empty cyclic result because no value must be repeated.
- Require a fully static caller-supplied shape for prefix creation and synthesize only canonical
  dense-contiguous layout.
- Preserve numeric values and raw BFLOAT16 bits; delegate BOOL normalization to the existing flat
  import path.
- Preserve explicit optional label and gradient intent for prefix creation. Integer ranges are
  always non-differentiable and therefore expose no `requiresGrad` argument.
- Modify `TensorFactory`, add one package-private `TensorPopulations` helper, update the exact API
  shape test, and add one focused population test.
- During implementation, update the Tensor API, glossary, task evidence, model master plan, and
  roadmap through the required separate clean-context documentation pass.

## Out of scope

- floating-point, BFLOAT16, BOOL, date/time, inclusive-end, open-ended, lazy, or symbolic ranges
- an operation kind for range, compiler-generated ranges, ONNX Range mapping, graph capture,
  provenance, or runtime generation
- empty-range construction, automatic step-direction correction, default steps, default labels,
  default data types, caller-selected integer `DataType`, or `requiresGrad` for integer ranges
- implicit numeric conversion, widening, narrowing, floating promotion, `double[]` conversion to
  another data type, boxed values, generic public arrays, collections, iterators, streams, or
  varargs
- prefix shape inference, nested-prefix import, descriptor or layout input, offset/strided/
  broadcast destination population, view scattering, alias-conflict policy, or source retention
- a general `fill`, arbitrary constant, linear spacing, logarithmic spacing, mask, repeat, tile, or
  sequence operation
- random creation or reproducibility policy from task 0012F
- typed Tensor reads/writes, bulk export, source Tensor population, mutation/version tracking,
  backing-array exposure, or zero-copy ownership transfer
- new storage types, native/off-heap/mapped allocation, `Arena`, deterministic close behavior,
  pooling, runtime residency, or backend storage
- gradient state, trainable/publication behavior, provenance, operations, compiler, planning,
  prepare, runtime, engine, backend, device, or execution behavior
- dependencies, preview/incubator features, Gradle changes, architecture changes, or another module
- a detailed specification for task 0012F, task 0013, or any later task

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md), especially model ownership of public
  Tensor construction and host storage and the exclusion of runtime/backend state
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Architecture overview](../../../../architecture/overview.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md), especially integer ranges and strict/cyclic
  prefix filling
- [Model master plan](../master-plan.md)
- [Task 0001](0001-data-type-model.md), which defines exact carrier data types and gradient
  eligibility
- [Task 0002](0002-shape-and-dimension-model.md), which defines static shapes and checked counts
- [Task 0003](0003-layout-descriptor-model.md), which defines canonical dense layout
- [Task 0007](0007-tensor-descriptor-model.md), which defines descriptor and gradient invariants
- [Task 0011](0011-public-tensor-skeleton.md), which defines public Tensor state
- [Task 0012](0012-tensor-factory.md), which defines construction and identity allocation
- [Task 0012A](0012a-host-storage-allocation.md), which defines heap allocation
- [Task 0012B](0012b-flat-typed-tensor-import.md), which defines exact typed copied population
- [Task 0012D](0012d-constant-tensor-creation.md), which defines bounded synthesized descriptors
- [Tensor API](../../../../api/tensor-api.md) and [glossary](../../../../glossary.md)

## Legacy evidence and rejected coupling

Read-only inspection of `TensorDataFactory`, `TensorDataFactoryTest`, the legacy Tensor range
convenience, and numerics fixtures confirms three useful capabilities: an inclusive-start,
exclusive-end integer arithmetic range with positive or negative non-zero step; strict copying of
the requested prefix; and cyclic repetition when fixture input is shorter than the requested
tensor.

The new design rejects the legacy nullable label, positive-dimension-only shape, rank-one scalar
convention, automatic `arange` label, caller-selected floating output for an integer range,
`double[]`-to-arbitrary-dtype conversion, mutable metadata, direct Tensor construction, and raw
array/storage exposure. It also rejects legacy implementation structure, runtime coupling, and
unrelated factory conveniences. Only the selected observable capabilities are retained.

## Architecture constraints

- Production remains in `io.github.pho001.synaptik.model.tensor`. Public methods belong to
  `TensorFactory`; deterministic population mechanics belong to one package-private helper in the
  same package.
- Package direction remains `model.tensor` toward existing `datatype`, `shape`, `layout`, and
  `storage` foundations. No reverse or cross-module dependency is introduced.
- Range creation is eager model-owned leaf-data construction, not an `Operation`, graph node,
  compiler pass, prepared executable, runtime generator, or backend kernel.
- The primitive range overload determines the exact result type: `int` produces `INT32`, and
  `long` produces `INT64`. No data-type argument or conversion policy is added.
- Range length is calculated exactly before a Java array is allocated. Signed subtraction,
  absolute-value, or addition overflow must not change the result or validation outcome.
- Prefix overloads determine data type from the exact primitive source carrier. They do not accept
  a separate data type or convert between carriers.
- Prefix creation synthesizes a descriptor only from the explicit shape, inferred exact data type,
  canonical dense-contiguous layout, and explicit gradient intent. It accepts no source layout or
  descriptor.
- Range and prefix methods create one exact temporary destination carrier and delegate exactly
  once to the matching task-0012B `fromFlatArray(...)` overload. They do not allocate Tensor IDs,
  construct Tensor/storage directly, or write through a public typed-access API.
- Strict and cyclic population copy values. Neither the complete source nor any source subrange is
  retained. Later caller mutation cannot change the tensor.
- BOOL prefix input is semantic: downstream flat import canonicalizes zero to `0` and every
  non-zero byte to `1`. BFLOAT16 `short[]` values remain raw bits.
- A fully static zero-element shape is valid. Strict population needs zero source values. Cyclic
  population also accepts an empty source because no modulo/repetition is performed for an empty
  result.
- If implementation needs conversion, scatter/view semantics, a public population-policy type,
  another production helper, a storage-contract change, architecture/dependency changes, or
  another module, stop and report the conflict.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.tensor` — owns TensorFactory, descriptors, and deterministic
  leaf-data construction.
- `io.github.pho001.synaptik.model.datatype` — supplies exact data types.
- `io.github.pho001.synaptik.model.shape` — supplies fully static result shapes.
- `io.github.pho001.synaptik.model.layout` — supplies canonical dense-contiguous geometry.

Packages added or changed:

- No package is added. Only the existing `model.tensor` package changes.

Type placement:

- `io.github.pho001.synaptik.model.tensor.TensorFactory` — receives the exact public range and
  prefix methods and remains the public construction boundary.
- `io.github.pho001.synaptik.model.tensor.TensorPopulations` — package-private final helper for
  exact range sizing/filling and strict/cyclic prefix copying.
- `io.github.pho001.synaptik.model.tensor.TensorFactoryPopulationTest` — same-package focused tests
  for all deterministic population behavior and internal boundary checks.

## Required contract

### Public integer range methods

Add exactly these two public static methods:

```java
public static Tensor range(
        int startInclusive,
        int endExclusive,
        int step,
        Optional<String> label)

public static Tensor range(
        long startInclusive,
        long endExclusive,
        long step,
        Optional<String> label)
```

The `int` overload creates `INT32`; the `long` overload creates `INT64`. Both results have shape
`Shape.of(elementCount)`, canonical dense-contiguous layout, and `requiresGrad == false`.

Values are exactly:

```text
startInclusive + i * step, for i in [0, elementCount)
```

The exclusive bound is never emitted. Positive steps require `startInclusive < endExclusive`.
Negative steps require `startInclusive > endExclusive`. Equality is an empty range and is rejected.
The final emitted value may be less than one step from the exclusive bound.

Do not add overloads without labels, a default step, caller-selected data type, floating values,
`requiresGrad`, inclusive bounds, `arange` aliases, or automatic labels.

### Range validation and sizing

Each public range method validates non-null `label` first with `Objects.requireNonNull(label,
"label")`, before range validation, carrier allocation, destination allocation, or ID allocation.
The helper then validates in this order:

1. If `step == 0`, throw `IllegalArgumentException` with exact message
   `range step must not be zero`.
2. If `startInclusive == endExclusive`, throw `IllegalArgumentException` with exact message
   `range must contain at least one element`.
3. If the step sign does not advance toward the end, throw `IllegalArgumentException` with exact
   message `range step direction does not advance toward end`.
4. Calculate the mathematical positive distance and ceiling-divided element count without
   overflowing the primitive carrier. `java.math.BigInteger` may be used inside the helper for
   exact sizing; no arbitrary-precision value is exposed.
5. If the count exceeds `Integer.MAX_VALUE`, throw `IllegalArgumentException` with exact message
   `range element count exceeds Java array limit: required=<required>, maximum=2147483647`.
6. Create the rank-one static shape, canonical dense layout, and exact non-differentiable
   descriptor.
7. Allocate exactly one matching primitive array and populate it in encounter order. Do not add a
   value after the final element merely to advance a loop; ordinary valid boundary ranges must not
   fail from an unused post-final overflow.
8. Delegate exactly once to the matching flat-import overload with the supplied label.

All range argument, direction, count, shape, layout, and descriptor failures occur before carrier,
destination, or ID allocation. A blank label is rejected through existing Tensor validation after
the range carrier and destination storage are allocated and an ID is consumed, but before flat
copying. Identifier exhaustion occurs after both carriers exist and before flat copying.

### Public strict flat-prefix methods

Add exactly six public static `fromStrictFlatPrefix(...)` overloads, one for each source carrier:

```java
public static Tensor fromStrictFlatPrefix(
        Shape shape, Optional<String> label, boolean requiresGrad, double[] source)
public static Tensor fromStrictFlatPrefix(
        Shape shape, Optional<String> label, boolean requiresGrad, float[] source)
public static Tensor fromStrictFlatPrefix(
        Shape shape, Optional<String> label, boolean requiresGrad, short[] source)
public static Tensor fromStrictFlatPrefix(
        Shape shape, Optional<String> label, boolean requiresGrad, int[] source)
public static Tensor fromStrictFlatPrefix(
        Shape shape, Optional<String> label, boolean requiresGrad, long[] source)
public static Tensor fromStrictFlatPrefix(
        Shape shape, Optional<String> label, boolean requiresGrad, byte[] source)
```

The carrier mapping is the existing exact mapping:

| Source | Data type | Semantics |
|---|---|---|
| `double[]` | `FLOAT64` | binary64 values |
| `float[]` | `FLOAT32` | binary32 values |
| `short[]` | `BFLOAT16` | raw BFLOAT16 bits |
| `int[]` | `INT32` | signed 32-bit values |
| `long[]` | `INT64` | signed 64-bit values |
| `byte[]` | `BOOL` | zero false; non-zero normalized downstream to one |

Strict prefix requires `source.length >= logicalElementCount`. It copies exactly the first
`logicalElementCount` values into a fresh exact-length carrier and ignores any remaining source
tail. Equal-size input is still copied and is not retained. A zero-element shape accepts any
source length and creates an empty carrier.

### Public cyclic flat-prefix methods

Add exactly six public static `fromCyclicFlatPrefix(...)` overloads with the same parameters and
carrier mapping:

```java
public static Tensor fromCyclicFlatPrefix(
        Shape shape, Optional<String> label, boolean requiresGrad, double[] source)
public static Tensor fromCyclicFlatPrefix(
        Shape shape, Optional<String> label, boolean requiresGrad, float[] source)
public static Tensor fromCyclicFlatPrefix(
        Shape shape, Optional<String> label, boolean requiresGrad, short[] source)
public static Tensor fromCyclicFlatPrefix(
        Shape shape, Optional<String> label, boolean requiresGrad, int[] source)
public static Tensor fromCyclicFlatPrefix(
        Shape shape, Optional<String> label, boolean requiresGrad, long[] source)
public static Tensor fromCyclicFlatPrefix(
        Shape shape, Optional<String> label, boolean requiresGrad, byte[] source)
```

For output index `i`, cyclic population copies `source[i % source.length]`. When the source is at
least as long as the output, this is exactly its requested prefix. A non-empty output requires a
non-empty source. A zero-element output accepts an empty source and performs no modulo operation.
The source is never retained.

### Public prefix validation order

Every strict and cyclic public overload validates in this order before helper delegation:

1. Reject null `shape` with `NullPointerException` and exact message `shape`.
2. Reject null `label` with `NullPointerException` and exact message `label`.
3. Reject null `source` with `NullPointerException` and exact message `source`.

The helper then validates before creating the destination carrier, allocating Tensor storage, or
allocating an ID:

1. If shape is not fully static, throw `IllegalArgumentException` with exact message
   `prefix tensor creation requires a fully static shape: <shape>`.
2. Read `shape.knownElementCount()`. Checked multiplication overflow remains
   `ArithmeticException`.
3. If count exceeds `Integer.MAX_VALUE`, throw `IllegalArgumentException` with exact message
   `prefix tensor element count exceeds Java array limit: required=<required>, maximum=2147483647`.
4. For strict mode, if source is shorter than count, throw `IllegalArgumentException` with exact
   message `strict flat prefix source is too short: required=<required>, actual=<actual>`.
5. For cyclic mode, if count is positive and source length is zero, throw
   `IllegalArgumentException` with exact message
   `cyclic flat prefix source must not be empty for non-empty output`.
6. Create `LayoutDescriptor.contiguous(shape)` and then
   `TensorDescriptor(dataType, shape, Optional.of(layout), requiresGrad)`. Existing descriptor
   gradient validation remains authoritative and occurs before carrier allocation.
7. Create exactly one fresh matching output carrier of the logical count, populate it, and
   delegate exactly once to the matching flat-import overload.

Null, shape, count, source-availability, layout, and descriptor/gradient failures consume no ID and
allocate no output carrier or destination storage. A blank label passes prevalidation, then fails
through existing Tensor validation after the output carrier, destination, and ID are allocated;
the ID is consumed and no flat copy occurs. Unexpected population or flat-copy failures propagate
without rollback.

### Package-private helper

Add one package-private final `TensorPopulations` class in `model.tensor`.

It must:

- have no fields, public/protected members, cache, registry, singleton, or mutable state;
- have one private zero-argument constructor;
- expose package-private typed entry methods only for the two range overloads and the twelve
  strict/cyclic primitive-array prefix overloads;
- use private shared implementation methods for exact range sizing, descriptor construction,
  prefix count validation, and carrier population;
- use no reflection, generic public API, backing-array retention, storage construction, Tensor
  construction, or ID allocation;
- allocate exactly one output carrier for every successful result; and
- delegate exactly once to the existing matching `TensorFactory.fromFlatArray(...)` overload.

`BigInteger` is permitted only as a private exact range-count calculation detail. Do not add a
public range descriptor, prefix mode, fill policy, sequence abstraction, scalar wrapper, stateful
factory, public helper, or production test hook.

## Valid and invalid scenarios

| Scenario | Result |
|---|---|
| `range(1, 8, 3, label)` | `INT32 [1, 4, 7]` |
| `range(5L, -2L, -2L, label)` | `INT64 [5, 3, 1, -1]` |
| Range start equals end | Rejected as empty |
| Positive step with descending bounds | Rejected as wrong direction |
| Range count above Java array limit | Rejected before carrier/ID allocation |
| Strict shape `[2,2]`, source `[1,2,3,4,5]` | Copies `[1,2,3,4]`; ignores `5` |
| Strict shape `[2,2]`, source `[1,2,3]` | Rejected before allocation |
| Cyclic shape `[2,3]`, source `[1,2,3,4]` | Copies `[1,2,3,4,1,2]` |
| Empty shape `[0,3]`, empty source | Valid empty strict or cyclic tensor |
| BOOL source `[0,-4,3]` | Destination bytes `[0,1,1]` |
| BFLOAT16 source | Raw prefix bits copied unchanged |
| Dynamic prefix shape | Rejected before allocation |
| Integral or BOOL prefix with `requiresGrad=true` | Existing descriptor rejection |
| Source mutated after return | Tensor contents remain unchanged |

## Affected files

Expected production changes:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorFactory.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorPopulations.java`

Expected test changes:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryPopulationTest.java`

Expected documentation and planning updates during implementation:

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task specification
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum scope

This task may create or modify at most two production files, one existing test, one new focused
test, and the five documentation/planning files above: nine paths total.

Do not modify another production/test file, Gradle, `AGENTS.md`, `ARCHITECTURE.md`, focused
architecture documentation, architecture tests, `capabilities.md`, another module, or unrelated
documentation. Do not create a task-0012F or task-0013 specification. Stop if another file, public
type, conversion policy, population mode, view/scatter rule, or architecture clarification is
required.

## Javadoc requirements

- Update `TensorFactory` type Javadoc to describe deterministic ranges and prefix population while
  preserving identity, allocation, ownership, and deferred-capability boundaries.
- Document both range overloads fully: carrier/result data type, inclusive/exclusive bounds,
  positive/negative step, non-empty requirement, exact sizing, label, copying/delegation, result,
  side effects, and every failure.
- Document all strict/cyclic overloads fully: carrier mapping, static shape and count, exact prefix
  or repetition semantics, source copying, BFLOAT16/BOOL behavior, explicit label/gradient intent,
  result, side effects, and every failure.
- Document the package-private helper and each package-private entry. Private shared methods must
  explain validation and why exact range sizing or temporary generic implementation detail does
  not create a public conversion contract.
- Do not describe random creation, typed Tensor access/export, conversion, general fill, view
  population, native/runtime/backend behavior, or complete factory parity as implemented.
- Review existing `DataType`, `Shape`, `LayoutDescriptor`, `TensorDescriptor`, `Tensor`,
  `TensorFactory`, and storage Javadocs and record why unchanged contracts remain accurate or stop
  if an out-of-scope correction is required.

## Acceptance criteria

- `TensorFactory` adds exactly two typed integer range methods, six strict-prefix overloads, and
  six cyclic-prefix overloads with the specified signatures and no other public API.
- Existing create, allocate, flat/nested import, constants, fields, allocator, and behavior remain
  unchanged; no current test is weakened or removed.
- `int` range produces only `INT32`; `long` range produces only `INT64`; both are rank one,
  dense-contiguous, non-differentiable, inclusive-start, and exclusive-end.
- Range validates label, non-zero step, non-empty bounds, direction, and exact Java-array-limited
  count in the specified order without primitive overflow.
- Every strict prefix copies exactly the requested first values, ignores a source tail, rejects a
  short source, and never retains the source.
- Every cyclic prefix repeats in encounter order, rejects an empty source only for non-empty
  output, and never retains the source.
- All prefix methods require fully static shape, synthesize canonical dense layout, infer exact
  data type from carrier, preserve raw numeric/BFLOAT16 values, and reuse flat BOOL normalization.
- All successful paths create one output carrier and delegate exactly once to matching flat import;
  they never call `nextTensorId()`, construct Tensor/storage, or write through a public access API.
- Validation messages, allocation order, ID side effects, and blank-label/exhaustion behavior
  match the required contract.
- Exact API/helper shape, imports, bytecode delegation, no reflection, no forbidden architecture
  references, and the nine-path maximum scope are verified.
- Complete Javadoc, Tensor API, glossary, task/master/roadmap status, and separate documentation
  review are finished in the same change.
- Task 0012E is `Complete` only after all validation; task 0012F remains `Draft` without a spec.

## Tests / validation

Run before and after the documentation pass:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryPopulationTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused population suite must cover both carriers; positive and negative steps; uneven final
steps; primitive minimum/maximum boundary cases; zero step; equality; wrong direction; count above
the array limit; exact labels/descriptors/storage values; all six strict and cyclic carriers;
source tails; short strict source; repetition; empty shapes/sources; dynamic shapes; count overflow
and limit; gradient eligibility; raw BFLOAT16; BOOL normalization; source isolation; validation
order; ID non-consumption and consumption; and permanent ID exhaustion. It must restore reflected
allocator state in `finally` and must not introduce a production hook.

Manually verify reflection and `javap -p -c -s` show the exact fourteen new public methods, exact
package-private helper shape, unchanged factory allocator and prior methods, descriptor synthesis,
one matching flat-import delegation per successful path, and no direct Tensor/storage/ID
construction. Verify production imports contain only current model/JDK types and no reflection,
streams, graph/runtime/backend types, Arena, or native access. Verify exact nine-path scope,
links/anchors, fences, whitespace, status synchronization, and absence of task-0012F/task-0013
specifications.

## Dependencies

- Task 0012B provides exact typed copied flat import and is complete.
- Task 0012 provides identity allocation and public construction and is complete.
- Tasks 0001, 0002, 0003, 0007, 0010, and 0011 provide data type, shape, layout, descriptor,
  storage, and Tensor contracts.
- Task 0012D provides the current synthesized-descriptor helper precedent and is complete; task
  0012E does not depend on its constant methods for behavior.

## Follow-up tasks

- Task 0012F will decide random-source ownership and reproducibility policy, then add normally
  distributed floating tensor creation.
- Task 0013 will define minimal Tensor provenance for future graph capture.
- Later work owns typed access/export, mutation/version tracking, views/scatter, range operations,
  ONNX Range mapping, compiler/runtime generation, and backend execution.

Do not create a detailed task-0012F or task-0013 specification as part of task 0012E.

## Architecture impact

Expected impact: None.

The architecture already assigns TensorFactory, public Tensor state, shape/data-type/layout model,
and host storage to `modules/model`. This task adds eager copied leaf data only and changes no
module boundary, dependency direction, lifecycle stage, storage ownership, graph semantics, or
runtime/backend rule. If implementation reveals otherwise, stop before editing architecture files.

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
- docs/planning/modules/model/tasks/0010-host-storage-abstraction.md
- docs/planning/modules/model/tasks/0011-public-tensor-skeleton.md
- docs/planning/modules/model/tasks/0012-tensor-factory.md
- docs/planning/modules/model/tasks/0012a-host-storage-allocation.md
- docs/planning/modules/model/tasks/0012b-flat-typed-tensor-import.md
- docs/planning/modules/model/tasks/0012d-constant-tensor-creation.md
- docs/planning/modules/model/tasks/0012e-range-and-prefix-population.md
- docs/api/tensor-api.md
- docs/glossary.md
- current TensorFactory/Tensor/descriptor/shape/layout/storage source and all current factory tests
- root/model Gradle configuration only to confirm Java 26

Implement task 0012E exactly as specified. Modify TensorFactory.java, add the one package-private
TensorPopulations.java helper, update TensorFactoryTest only for exact API shape, and add
TensorFactoryPopulationTest.java. Add exactly two typed integer range methods, six strict flat-
prefix overloads, and six cyclic flat-prefix overloads. Preserve every existing factory method,
allocator, allocation/import/constant helper, and test.

Ranges must be non-empty, inclusive-start/exclusive-end, accept positive or negative non-zero
steps, calculate count without primitive overflow, infer INT32 or INT64 from the overload, and
delegate through matching flat import. Prefix methods must infer exact data type from carrier,
require fully static shape, synthesize dense-contiguous descriptors, copy strict prefixes or
cyclically repeat them into one fresh exact carrier, and delegate through matching flat import.
Preserve raw numeric/BFLOAT16 values and let flat import normalize BOOL. Follow exact validation
order, messages, ownership, and ID side effects from the task.

Do not add floating ranges, empty ranges, implicit conversion, generic public arrays, descriptor/
layout prefix inputs, general fill/repeat/tile, random creation, typed access/export, views/scatter,
storage types, Arena/native allocation, provenance, operations, graph/compiler/runtime/backend
behavior, dependencies, build changes, or follow-up specs. Stop if work exceeds the nine permitted
paths or requires an architecture decision.

Run every focused/aggregate test, Javadoc, bytecode/import/manual, documentation, link, whitespace,
scope, and status check in the task.

After initial implementation validation, hand the actual diff to a separate documentation-focused
agent or thread with clean context in the same overall change. Keep task 0012E incomplete until
that pass finishes. The handoff must include this task, implementation/test/bytecode diff, exact
range sizing and boundary semantics, strict/cyclic carrier/count/copy behavior, descriptor and
gradient validation, BFLOAT16/BOOL rules, ownership, delegation and ID side effects, architecture
constraints, expected Tensor API/glossary/Javadoc impact, existing-Javadoc review list, and every
validation command.

That documentation agent must independently read AGENTS.md, ARCHITECTURE.md,
docs/developer-guide/documentation-rules.md, the documentation profile index, General style,
API/Javadoc style, Planning style, Example format when an example changes, this task, final
source/tests/generated Javadoc, Tensor API, glossary, model master plan, roadmap, and existing
Tensor/TensorFactory/TensorDescriptor/DataType/Shape/LayoutDescriptor/storage contracts. It must
inspect actual implementation and evidence rather than rely on the handoff. It must finalize all
new/affected Javadocs, move only deterministic integer range and strict/cyclic prefix creation into
current API/glossary language, preserve task 0012F as planned, review links/anchors/fences/
whitespace and terminology, record reasoned existing-Javadoc and architecture/capability no-change
conclusions, and synchronize only the allowed planning files.

At the end, update this task, model master plan, and roadmap for status/evidence. Record local
decisions, known limitations, exact validation evidence including documentation-agent identity and
results, implementation notes, and the canonical completion summary. Do not mark task 0012E
Complete until implementation, tests, Javadoc, independent documentation pass, scope review, and
status synchronization all pass. Task 0012F then remains the next Draft frontier without a detailed
specification. Do not commit or push.
```

## Local decisions

- Range is a typed eager factory capability, not an operation. The `int`/`long` overloads avoid a
  caller-selected data type and the conversion ambiguity present in legacy code.
- Empty ranges remain rejected to preserve the selected legacy failure contract. Empty prefix
  outputs remain valid because the new shape/storage baseline explicitly supports zero-sized
  dimensions.
- Range labels are explicit and never synthesized. Integer range results are non-differentiable by
  construction, so no unusable `requiresGrad` parameter is exposed.
- Strict and cyclic prefix APIs are carrier-specific. This increases overload count but preserves
  compile-time exact typing and avoids a public `Object`, policy enum, boolean mode, or conversion
  contract.
- Prefix creation takes a shape rather than a descriptor because it deliberately produces a new
  canonical dense leaf. Populating arbitrary layout geometry would require separate scatter and
  alias semantics.
- A cyclic empty source is valid only for a zero-element result. This makes zero-sized shapes
  composable without inventing a repeated value or performing modulo by zero.
- The combined task is retained as one bounded deterministic-population concept because range and
  prefix share the same helper, exact carrier creation, flat-import delegation, affected files,
  and validation evidence. It changes four Java files and remains within one isolated session.

## Known limitations

- Range supports only eager `INT32` and `INT64` results and cannot produce an empty tensor.
- Prefix creation supports only fully static canonical dense results and exact primitive carriers.
- Prefix methods do not infer shape, convert values, retain source layout, or populate views.
- Java arrays limit every result to `Integer.MAX_VALUE` elements; practical memory limits may be
  lower and `OutOfMemoryError` propagates.
- Population is not atomic with concurrent mutation of the source array. Callers must not mutate a
  prefix source concurrently with construction.
- No typed Tensor access/export exists yet; focused tests inspect the existing raw host segment
  only inside the model test package.

## Validation evidence

Planning read the architecture contract, planning/documentation rules, model capability baseline,
completed shape/layout/descriptor/storage/Tensor/factory tasks, current factory source/tests, and
read-only legacy `TensorDataFactory`, its tests, the Tensor range convenience, and numerics fixture
uses. The resulting specification introduced no architecture, dependency, package, or build
decision.

Implementation and documentation validation on 2026-07-04:

- `./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryTest`
  — passed, `BUILD SUCCESSFUL`; exact factory API-shape test remained green.
- `./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryPopulationTest`
  — passed, `BUILD SUCCESSFUL`; all 9 focused population tests passed.
- `./gradlew :modules:model:test` — passed, `BUILD SUCCESSFUL`; XML aggregation reported 205 tests,
  0 failures, 0 errors, and 0 skipped.
- `./gradlew :modules:model:javadoc` — passed, `BUILD SUCCESSFUL`; generated TensorFactory Javadoc
  was inspected for both ranges and all twelve prefix overloads, including parameters, results,
  failures, ownership, validation/allocation ordering, and ID side effects.
- `javadoc -quiet -private -d /private/tmp/synaptik-0012e-private-javadoc -classpath modules/model/build/classes/java/main modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorPopulations.java`
  — passed with no diagnostics, independently validating the package-private helper documentation.
- `./gradlew test` — passed, `BUILD SUCCESSFUL`; the final repeat reported 36 actionable tasks all
  up-to-date (the preceding run reported 1 executed and 35 up-to-date).
- The documentation example was compiled with
  `javac -cp modules/model/build/classes/java/main -d /private/tmp /private/tmp/DeterministicPopulationExample.java`
  and run with
  `java -cp modules/model/build/classes/java/main:/private/tmp DeterministicPopulationExample`.
  It printed the documented `INT32`, `[1, 4, 7]`, `Shape[2, 2]`, strict-prefix values, and canonical
  cyclic BOOL values.
- `javap -p -s` confirmed the prior TensorFactory API plus exactly 14 new public methods and the
  unchanged private `importFlat` and `nextTensorId`; TensorPopulations has exactly 14 package-private
  typed entries, four private shared methods, one private constructor, and no fields.
- `javap -p -c` confirmed each helper entry delegates exactly once to the matching typed
  `TensorFactory.fromFlatArray(...)` overload. Source/import inspection found no reflection,
  streams, `Arena`, native allocation, direct Tensor/storage/ID construction, or architecture-layer
  reference in TensorPopulations. The existing direct Tensor/storage/ID construction paths in
  TensorFactory were unchanged.
- Manual test and source review confirmed exact range count/direction messages, no unused
  post-final primitive addition, prefix null and semantic validation order, canonical descriptor
  synthesis, source copying, BFLOAT16 preservation, downstream BOOL normalization, blank-label ID
  consumption, permanent exhaustion, and allocator restoration.
- Documentation-focused context
  `/root/implement_model_0012e/review_model_0012e_docs` independently applied General style,
  API/Javadoc style, Planning style, and Example format. It reviewed the final Java diff, tests,
  generated Javadoc, Tensor API, glossary, task/master/roadmap status, links, anchors, fences,
  terminology, and the existing Tensor, TensorDescriptor, DataType, Shape, LayoutDescriptor,
  HostTensorStorage, MemorySegmentStorage, allocation, flat/nested import, and constant contracts.
- `git diff --check` and targeted trailing-whitespace checks passed. The final local Markdown
  checker resolved 155 links and anchors with 0 failures; fence counts were even in all five
  touched Markdown files. Two initial checker invocations failed only from Ruby-script syntax/API
  compatibility and were corrected before the successful final check. Exact nine-path scope,
  status synchronization, and absence of task-0012F/task-0013 specifications also passed.

Existing Tensor, TensorDescriptor, DataType, Shape, LayoutDescriptor, HostTensorStorage, and
MemorySegmentStorage Javadocs required no changes: deterministic population creates ordinary
Tensor state through the existing descriptor, dense layout, copied flat-import, heap-storage, and
factory-ID contracts without changing any of those types' ownership, validity, lifecycle, or
failure semantics. Existing TensorFactory create/allocate/flat/nested/constant Javadocs also remain
accurate because their implementations and contracts did not change.

Architecture and cross-module review concluded that `ARCHITECTURE.md`, focused architecture
documents, `capabilities.md`, ADRs, architecture tests, backend conformance, integration tests,
build configuration, and dependencies require no changes. The implementation stays inside
`modules/model`, adds eager copied leaf-data conveniences only, preserves package and dependency
direction, and changes no architecture, backend, runtime, end-to-end execution, or build rule.

## Implementation notes

- `TensorFactory` exposes exactly the two typed range overloads and twelve carrier-specific prefix
  overloads. Public methods own null-order checks and delegate deterministic mechanics to the one
  package-private stateless helper.
- `TensorPopulations` confines exact `BigInteger` range sizing to one private method, synthesizes
  only canonical dense descriptors, creates one complete matching carrier, and delegates once to
  the existing flat import. Range loops skip step addition after the final emitted element.
- Strict prefixes use `Arrays.copyOf` for the validated logical count. Cyclic prefixes fill by
  `source[index % source.length]`; zero output performs no modulo operation. Both paths leave BOOL
  normalization in flat import and preserve raw BFLOAT16 bits.
- Focused tests cover helper/API shape, typed and primitive-boundary ranges, exact messages and ID
  effects, all six strict and cyclic carriers, source isolation, empty results, validation order,
  gradient rules, permanent exhaustion, and reflected allocator restoration.
- The Tensor API and glossary now describe deterministic range/prefix population as current. A
  compiled and run complete API example demonstrates range typing, strict-tail handling, cyclic
  BOOL normalization, and copied source ownership. Random creation remains the next Draft frontier.

## Completion summary

- Completed changes: added exact typed integer ranges and strict/cyclic flat-prefix population with
  overflow-safe sizing, canonical dense descriptors, copied ownership, and existing factory ID and
  flat-import behavior.
- Files changed or created: exactly the two production files, two test files, Tensor API, glossary,
  this task, model master plan, and roadmap listed in the nine-path maximum scope.
- Tests and validation: both focused suites, aggregate model tests, generated model Javadoc, full
  repository tests, bytecode/API/import/manual checks, compiled documentation example, XML counts,
  documentation checks, and `git diff --check` passed.
- Documentation-agent review: completed independently by
  `/root/implement_model_0012e/review_model_0012e_docs` using the required profiles and final
  implementation evidence.
- Documentation impact: Tensor API, glossary, task evidence, model master plan, and roadmap were
  synchronized; architecture, capability, ADR, backend, integration, build, and dependency
  documents required no change for the recorded reasons.
- Javadoc review: TensorFactory and TensorPopulations were finalized; existing related public and
  storage contracts remain accurate without edits for the recorded reasons.
- Glossary impact: deterministic range and strict/cyclic flat-prefix population moved from planned
  to current language; no new standalone domain term was needed beyond the TensorFactory entry.
- Unresolved issues: None.
- Follow-up required: None. Task 0012F remains the next Draft task without a detailed specification.

Status: Complete
