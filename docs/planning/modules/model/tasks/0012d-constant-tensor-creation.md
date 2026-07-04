# Task 0012D: Constant Tensor Creation

## Status

Complete

## Goal

Add a bounded, type-explicit factory surface for rank-zero scalar tensors, zero-filled tensors,
one-filled tensors, and zero/one tensors shaped like an existing Tensor. Every result is an
independent dense-contiguous leaf tensor created through the completed descriptor, allocation, and
flat-import paths.

This task defines constant creation without a generic boxed value, implicit numeric conversion,
default data type, inherited storage, or hidden gradient behavior.

## Scope

- Add exact scalar methods for `double`, `float`, BFLOAT16 converted explicitly from `float`,
  `int`, `long`, and semantic `boolean` values.
- Create scalar tensors with canonical rank-zero `Shape.scalar()` metadata.
- Add `zeros(...)` and `ones(...)` methods taking explicit `Shape`, `DataType`, optional label, and
  `requiresGrad`.
- Add `zerosLike(...)` and `onesLike(...)` methods taking a Tensor template, optional label, and
  explicit `requiresGrad`.
- Define `*Like` as copying only the template's immutable logical shape and data type. Create a new
  dense-contiguous descriptor, storage, identity, and label state.
- Support all six current data types.
- Use JVM default-zero heap allocation for zeros.
- Populate ones with exact carrier values: `1.0d`, `1.0f`, BFLOAT16 representation of `1.0f`,
  `1`, `1L`, and canonical BOOL byte `1`.
- Use the completed BFLOAT16 conversion contract for semantic BFLOAT16 scalar input and one-fill.
- Require fully static shapes and preserve rank-zero and zero-sized-shape behavior.
- Add one package-private helper for constant descriptor construction and population.
- Update the exact `TensorFactory` API-shape test and add one focused constant-factory test suite.
- During implementation, update Tensor API, glossary, task evidence, model master plan, and roadmap
  through the required separate clean-context documentation pass.

## Out of scope

- a generic `Number`, `Object`, boxed-value, string-parsed, or caller-supplied `DataType + double`
  scalar API
- implicit widening, narrowing, integer rounding, floating promotion, truthiness, or conversion
  between scalar carriers
- raw-bit BFLOAT16 scalar input; raw BFLOAT16 arrays remain available through flat import
- arbitrary `full` or fill-value tensors, NaN/infinity policies beyond exact carrier semantics,
  or scalar broadcasting operations
- copying the template label, `requiresGrad`, layout, storage, storage liveness, provenance,
  publication state, operation/input references, graph IDs, or runtime/backend state in `*Like`
- preserving offset, strided, broadcast, view, or unresolved template layout; every result is new
  dense-contiguous leaf storage
- dynamic-shape allocation or symbolic binding
- task 0012E integer ranges, strict-prefix filling, or cyclic-prefix filling
- task 0012F random values, random sources, seeds, or reproducibility
- typed Tensor reads/writes or export, mutation/version tracking, views/scatter, or zero-copy state
- native/off-heap/mapped allocation, `Arena`, ownership/close behavior, pooling, or new storage types
- provenance, expression operations, graph/compiler/planning/prepare/runtime/engine/backend/device
  behavior, dependencies, build changes, architecture changes, another module, or later task specs

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md), especially model ownership of public
  Tensor construction and host storage and exclusion of runtime/backend storage
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Architecture overview](../../../../architecture/overview.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md), especially scalar, zeros, ones, zeros-like, and
  ones-like factory capabilities
- [Model master plan](../master-plan.md)
- [Task 0001](0001-data-type-model.md), which defines all data types and BFLOAT16 conversion
- [Task 0002](0002-shape-and-dimension-model.md), which defines rank-zero, static, and empty shapes
- [Task 0003](0003-layout-descriptor-model.md), which defines dense-contiguous geometry
- [Task 0007](0007-tensor-descriptor-model.md), which defines descriptor gradient eligibility
- [Task 0011](0011-public-tensor-skeleton.md), which defines stable Tensor metadata and storage
- [Task 0012](0012-tensor-factory.md), which defines public construction and identity allocation
- [Task 0012A](0012a-host-storage-allocation.md), which defines exact-span JVM heap allocation
- [Task 0012B](0012b-flat-typed-tensor-import.md), which defines exact-carrier population
- [Task 0012C](0012c-nested-typed-tensor-import.md), which remains unchanged
- [Tensor API](../../../../api/tensor-api.md) and [glossary](../../../../glossary.md)

## Legacy evidence and rejected coupling

Read-only legacy inspection covered `TensorDataFactory`, Tensor convenience methods,
`TensorStorageAccess`, `TensorConstructorDataTypeTest`, `NdTensorSequencePrimitivesTest`, and uses of
`scalar`, `zeros`, `ones`, `zerosLike`, and `onesLike` throughout tests and graph code.

Useful legacy capabilities are semantic scalar creation, all-data-type zeros and ones, independent
like-shaped constants, and BOOL/INT64 handling. The old design represented scalars as shape `[1]`,
accepted one `double` plus a requested data type, rounded integral values with incomplete range
validation, converted floating carriers implicitly, attached default labels, rejected empty
dimensions, and recreated Tensor/autograd state directly.

The new contract keeps the capabilities while rejecting those couplings. Scalars use the model's
canonical rank-zero shape. Java scalar carrier determines data type exactly; BFLOAT16 conversion is
named explicitly. Shape, label, and gradient intent are explicit. Like methods copy no mutable or
layout state. Final allocation and identity remain in the already completed factory paths.

## Architecture constraints

- Production remains in `io.github.pho001.synaptik.model.tensor`. Public methods belong to
  `TensorFactory`; constant construction internals belong to one package-private helper in the same
  package.
- Package direction remains `model.tensor` toward existing `datatype`, `shape`, `layout`, and
  `storage` foundations. No reverse or cross-module dependency is introduced.
- Scalar methods infer data type only from their declared primitive signature. They do not accept
  a separate `DataType`, boxed value, or generic conversion request.
- `scalarBFloat16(float, ...)` is a deliberately named semantic conversion using
  `BFloat16Bits.fromFloat`. Other scalar methods perform no conversion.
- Every scalar uses `Shape.scalar()`, canonical contiguous scalar layout, exactly one logical
  element, and new independent storage.
- Zero/one methods create only canonical dense-contiguous layout from a caller-supplied fully
  static shape. They never accept or preserve caller-supplied layout geometry.
- Like methods read only `template.descriptor().shape()` and `dataType()`. They do not observe or
  retain template label, storage, liveness, layout, ID, or future provenance.
- Like methods take explicit `requiresGrad`; no gradient flag is inherited. Descriptor validation
  remains the authority for differentiability.
- Zero-filled storage must use the existing exact-span heap allocation path and JVM default-zero
  representation. Do not run a per-element zero loop.
- One-filled storage must use a fresh exact typed primitive carrier and delegate to exactly one
  matching task-0012B flat-import overload. Do not write through a new public typed-access API.
- Constants must not call `nextTensorId`, construct `Tensor`, construct `MemorySegmentStorage`, or
  access backend/runtime state directly.
- If implementation needs a new public scalar-value abstraction, conversion policy, general fill
  API, storage-contract change, native allocation, architecture change, or another module, stop
  and report the conflict.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.tensor` — owns TensorFactory, Tensor, descriptors, and the
  constant-creation helper.
- `io.github.pho001.synaptik.model.datatype` — supplies exact data types and BFLOAT16 conversion.
- `io.github.pho001.synaptik.model.shape` — supplies static, scalar, and zero-sized shapes.
- `io.github.pho001.synaptik.model.layout` — supplies canonical dense-contiguous geometry.

Packages added or changed:

- No package is added. Only the existing `model.tensor` package changes.

Type placement:

- `io.github.pho001.synaptik.model.tensor.TensorFactory` — receives the ten exact public constant
  methods and remains the sole public construction boundary.
- `io.github.pho001.synaptik.model.tensor.TensorConstants` — package-private final helper for
  constant descriptor validation, carrier construction, zero allocation, one population, and flat
  import dispatch.
- `io.github.pho001.synaptik.model.tensor.TensorFactoryConstantTest` — focused same-package tests
  for public constant behavior and the package-private helper boundary.

## Required contract

### Public scalar methods

Add exactly these six public static methods:

```java
public static Tensor scalar(
        double value, Optional<String> label, boolean requiresGrad)

public static Tensor scalar(
        float value, Optional<String> label, boolean requiresGrad)

public static Tensor scalarBFloat16(
        float value, Optional<String> label, boolean requiresGrad)

public static Tensor scalar(
        int value, Optional<String> label, boolean requiresGrad)

public static Tensor scalar(
        long value, Optional<String> label, boolean requiresGrad)

public static Tensor scalar(
        boolean value, Optional<String> label, boolean requiresGrad)
```

The mapping is exact:

| Method value | Result data type | Stored representation |
|---|---|---|
| `double` | `FLOAT64` | exact binary64 bits |
| `float` | `FLOAT32` | exact binary32 bits |
| `scalarBFloat16(float)` | `BFLOAT16` | `BFloat16Bits.fromFloat(value)` |
| `int` | `INT32` | exact signed 32-bit value |
| `long` | `INT64` | exact signed 64-bit value |
| `boolean` | `BOOL` | false byte `0` or true byte `1` |

Every method null-checks `label` before descriptor, carrier, destination, or ID allocation. It
constructs a rank-zero dense descriptor with the inferred exact data type and supplied
`requiresGrad`, then delegates one-element population through the matching flat-import overload.

Do not add default labels, unlabeled overloads, a default floating type, a `short` raw-BFLOAT16
scalar, a `byte` BOOL scalar, `DataType` arguments, boxed/generic scalar values, or conversion
overloads.

### Public zero/one methods

Add exactly these four public static methods:

```java
public static Tensor zeros(
        Shape shape,
        DataType dataType,
        Optional<String> label,
        boolean requiresGrad)

public static Tensor ones(
        Shape shape,
        DataType dataType,
        Optional<String> label,
        boolean requiresGrad)

public static Tensor zerosLike(
        Tensor template,
        Optional<String> label,
        boolean requiresGrad)

public static Tensor onesLike(
        Tensor template,
        Optional<String> label,
        boolean requiresGrad)
```

`zeros` and `ones` validate non-null `shape`, `dataType`, and `label` in that order before helper
delegation. Like methods validate non-null `template` and `label` in that order, then delegate the
template's exact immutable shape and data type to the corresponding zero/one helper.

`zerosLike` and `onesLike` do not preserve the template's layout, label, storage, ID, gradient flag,
or any other state. They accept a static unresolved, dense, offset, strided, or broadcast template
descriptor but synthesize new canonical dense-contiguous geometry from its shape. A dynamic shape
is rejected because physical constant storage cannot be allocated without binding it.

### Package-private helper

Add one package-private final `TensorConstants` class in `model.tensor`.

Its package-private static entry surface is exactly:

```java
static Tensor scalar(
        double value, Optional<String> label, boolean requiresGrad)
static Tensor scalar(
        float value, Optional<String> label, boolean requiresGrad)
static Tensor scalarBFloat16(
        float value, Optional<String> label, boolean requiresGrad)
static Tensor scalar(
        int value, Optional<String> label, boolean requiresGrad)
static Tensor scalar(
        long value, Optional<String> label, boolean requiresGrad)
static Tensor scalar(
        boolean value, Optional<String> label, boolean requiresGrad)
static Tensor zeros(
        Shape shape, DataType dataType, Optional<String> label, boolean requiresGrad)
static Tensor ones(
        Shape shape, DataType dataType, Optional<String> label, boolean requiresGrad)
```

It must:

- have no fields, public/protected members, cache, registry, singleton, or mutable state;
- have one private zero-argument constructor;
- expose only the eight package-private static entries above;
- use private implementation methods only for descriptor/count validation, exact carrier creation,
  typed fill, and flat-import dispatch;
- construct only rank-zero or caller-shape dense-contiguous descriptors;
- call existing `allocate(descriptor, label)` for zero-filled tensors;
- call one matching existing `fromFlatArray(descriptor, label, carrier)` for scalars and ones; and
- retain no template or intermediate carrier after return.

Do not add a public constant descriptor, fill enum, scalar wrapper, generic utility, stateful
factory, public backing-array surface, or production test hook.

### Shape, count, and descriptor validation

For `zeros`, `ones`, `zerosLike`, and `onesLike`, the helper validates in this order before carrier,
destination, or ID allocation:

1. If shape is not fully static, throw `IllegalArgumentException` with exact message
   `constant tensor creation requires a fully static shape: <shape>`.
2. Read `shape.knownElementCount()`. Checked multiplication overflow remains
   `ArithmeticException`.
3. If the non-overflowing count exceeds `Integer.MAX_VALUE`, throw `IllegalArgumentException` with
   exact message
   `constant tensor element count exceeds Java array limit: required=<required>, maximum=2147483647`.
4. Create `LayoutDescriptor.contiguous(shape)`. Checked stride/span overflow remains
   `ArithmeticException`.
5. Create `TensorDescriptor(dataType, shape, Optional.of(layout), requiresGrad)`. Existing gradient
   eligibility validation remains authoritative.

Scalar methods use the same descriptor path with `Shape.scalar()` and a known count of one.

All failures above consume no tensor ID and allocate no destination storage. Descriptor eligibility
must be checked before a scalar/one carrier is allocated. Public null checks also consume no ID.

### Zero and one population

Zeros call `TensorFactory.allocate(descriptor, label)` exactly once. JVM primitive arrays already
contain the correct raw representation for numeric positive zero and BOOL false. Do not allocate a
second source array or perform a fill/copy loop.

Ones allocate exactly one primitive carrier of logical element count and fill it using the matching
typed `Arrays.fill` operation:

| Data type | Carrier | Fill value |
|---|---|---|
| `FLOAT64` | `double[]` | `1.0d` |
| `FLOAT32` | `float[]` | `1.0f` |
| `BFLOAT16` | `short[]` | `BFloat16Bits.fromFloat(1.0f)` (`0x3F80`) |
| `INT32` | `int[]` | `1` |
| `INT64` | `long[]` | `1L` |
| `BOOL` | `byte[]` | `1` |

Then dispatch exactly once to the matching flat-import overload. Empty shapes allocate an empty
carrier and produce empty storage. Rank-zero shape has one value.

### Ownership and failure side effects

- Every method creates a new Tensor, descriptor, layout, storage object, backing array, and ID.
- No result aliases template storage or another result.
- No method retains its template or an intermediate carrier.
- A blank label reaches existing Tensor validation after destination and ID allocation. It consumes
  one ID without rollback. For ones/scalars, their source carrier was also allocated first.
- Identifier exhaustion occurs after destination allocation and before zero result publication or
  flat copy. For ones/scalars, the filled source carrier already exists. No ID is rolled back.
- JVM allocation failure before ID allocation propagates unchanged and consumes no ID.

## Affected files

Expected implementation and tests:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorFactory.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorConstants.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryConstantTest.java`

Expected documentation and planning updates in the same overall change:

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/tasks/0012d-constant-tensor-creation.md`
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum scope

This task may create or modify at most the nine paths listed under Affected files.

No existing production Java contract other than `TensorFactory` may change. `TensorFactoryTest`
may change only for the deliberate exact API-shape expectation. Do not modify `NestedTensorArray`
or its tests. If another production type, test, documentation file, architecture file, build file,
module, or dependency is needed, stop and propose a separate follow-up or architecture decision.

## Javadoc requirements

- Update `TensorFactory` type Javadoc to distinguish descriptor creation, allocation, import, and
  constant creation.
- Fully document all ten new public methods, including exact carrier/type meaning, BFLOAT16
  rounding, rank-zero scalar semantics, dense descriptor synthesis, static/empty shape behavior,
  like-method copied and non-copied facts, `requiresGrad`, ownership, label/ID/allocation side
  effects, result, nullability, and every failure category.
- Document package-private `TensorConstants` and every package-private entry with purpose,
  validation, population, ownership, parameters, results, and failures.
- Review existing `BFloat16Bits`, `Tensor`, `TensorDescriptor`, `Shape`, `LayoutDescriptor`,
  `DataType`, storage contracts, allocation, flat-import, and nested-import Javadocs. Change none
  unless an in-scope stale `TensorFactory` statement can be corrected without behavioral change;
  otherwise stop on an out-of-scope discrepancy and record reasoned no-change conclusions.

## Acceptance criteria

- `TensorFactory` adds exactly the ten public methods specified and no other public surface.
- One package-private stateless helper implements only constant descriptor/population behavior.
- All scalar methods create rank-zero dense tensors with exact data type and value semantics.
- BFLOAT16 semantic scalar and one-fill use the completed round-to-nearest-even conversion.
- Zeros and ones support all six data types, rank-zero and zero-sized static shapes, and explicit
  labels and gradient eligibility.
- Like methods copy only shape and data type, synthesize dense layout, honor explicit label and
  `requiresGrad`, and do not inspect or alias template storage.
- Dynamic shapes, over-limit counts, non-differentiable gradient requests, and null arguments fail
  in the specified order before destination/ID allocation.
- Blank-label and exhaustion behavior remain delegated and consume IDs exactly as specified.
- Zeros use direct default-zero allocation; scalars/ones use one matching flat-import overload.
- No new conversion, full/fill, range/prefix, random, storage, access, provenance, graph, runtime,
  backend, dependency, or build behavior is introduced.
- Complete Javadoc, Tensor API, glossary, task/master/roadmap status, and separate documentation
  review are finished in the same change.
- Task 0012D is `Complete` only after all validation; task 0012E remains `Draft` without a spec.

## Tests / validation

Run before and after the documentation pass:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryConstantTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

Focused tests must cover:

- exact factory API and helper visibility/state shape;
- all six scalar methods, rank-zero metadata, exact values/bits, BFLOAT16 rounding, BOOL bytes,
  labels, and differentiable `requiresGrad`;
- zeros and ones for all six data types, a normal shape, scalar shape, and zero-sized shape;
- like methods from static unresolved, dense, and non-dense/view descriptors, proving only
  shape/type reuse, dense new layout, explicit label/gradient, and independent storage/identity;
- null validation order for scalar label, shape/dataType/label, and template/label;
- dynamic shape, over-limit count, and non-differentiable gradient rejection without ID use;
- blank-label ID consumption, exhaustion timing, and source/carrier non-retention; and
- preservation of all completed allocation, flat-import, and nested-import tests.

Manually verify reflection and `javap -p -c -s` show exactly the ten new public methods, the
package-private stateless helper, existing factory surface/allocator unchanged, exact null order,
rank-zero and dense descriptor construction, zero delegation to `allocate`, scalar/one delegation
to one flat overload, and no direct Tensor/storage/ID construction. Verify imports contain only
current model/JDK types and no boxed value buffer, reflection, graph/runtime/backend type, Arena,
native access, service, or random state. Verify exact nine-path scope, generated Javadoc,
documentation examples, links/anchors, fences, whitespace, status synchronization, and absence of
task-0012E/task-0013 specs.

## Dependencies

- Task 0012B provides exact-carrier flat population and is complete.
- Task 0012A provides exact-span zero-initialized heap allocation and is complete.
- Task 0012 provides public construction and ID allocation and is complete.
- Tasks 0001, 0002, 0003, 0007, and 0011 provide data type/BFLOAT16, shape, layout, descriptor, and
  Tensor contracts.
- Task 0012C is complete but its nested-array helper is not a dependency and remains unchanged.

## Follow-up tasks

- Task 0012E will define integer ranges and strict/cyclic prefix population.
- Task 0012F will define random creation and reproducibility policy.
- Later work owns typed export/access, mutation versions, views/scatter, provenance, and operations.

Do not create a detailed task-0012E or later specification as part of task 0012D.

## Architecture impact

Expected impact: None.

The architecture already assigns TensorFactory, data type/shape/layout, and host storage to
`modules/model`. This task adds independent constant leaf creation only and changes no module
boundary, dependency direction, lifecycle stage, storage ownership, runtime/backend rule, or
compiler responsibility. If implementation reveals otherwise, stop before editing architecture
files.

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
- docs/planning/modules/model/tasks/0011-public-tensor-skeleton.md
- docs/planning/modules/model/tasks/0012-tensor-factory.md
- docs/planning/modules/model/tasks/0012a-host-storage-allocation.md
- docs/planning/modules/model/tasks/0012b-flat-typed-tensor-import.md
- docs/planning/modules/model/tasks/0012c-nested-typed-tensor-import.md
- docs/planning/modules/model/tasks/0012d-constant-tensor-creation.md
- docs/api/tensor-api.md
- docs/glossary.md
- current TensorFactory/Tensor/descriptor/shape/layout/storage source and all current factory tests
- root/model Gradle configuration only to confirm Java 26

Implement task 0012D exactly as specified. Modify TensorFactory.java, add the one package-private
TensorConstants.java helper, update TensorFactoryTest only for exact API shape, and add
TensorFactoryConstantTest.java. Add exactly the six scalar and four zero/one public methods from
the task and no other public API. Preserve every existing factory method, allocator, allocation,
flat-import, nested-import, helper, and test.

Use exact primitive scalar semantics, explicit scalarBFloat16(float) conversion, canonical rank-
zero dense descriptors, and explicit labels/requiresGrad. Require fully static shapes. Zeros must
delegate to allocate without a source array or fill loop. Ones must create one exact typed filled
carrier and delegate to the matching flat import. Like methods copy only template shape/data type,
synthesize dense layout, and never inspect or retain template storage, label, ID, gradient flag, or
layout. Follow exact validation order, messages, ownership, and ID side effects from the task.

Do not add generic/boxed scalars, implicit conversion, default labels/types, arbitrary full/fill,
range/prefix, random, typed access/export, view preservation, storage types, Arena/native
allocation, provenance, operations, graph/compiler/runtime/backend behavior, dependencies, build
changes, or follow-up specs. Do not modify NestedTensorArray or its tests. Stop if work exceeds the
nine permitted paths or requires an architecture decision.

Run every focused/aggregate test, Javadoc, bytecode/import/manual, documentation, link, whitespace,
scope, and status check in the task.

After initial implementation validation, hand the actual diff to a separate documentation-focused
agent or thread with clean context in the same overall change. Keep task 0012D incomplete until
that pass finishes. The handoff must include this task, implementation/test diff, scalar carrier
semantics, BFLOAT16 conversion, shape/count/gradient validation, zero/one/like behavior, ownership,
delegation and ID side effects, architecture constraints, expected Tensor API/glossary/Javadoc
impact, existing-Javadoc review list, and every validation command.

That documentation agent must independently read AGENTS.md, ARCHITECTURE.md,
docs/developer-guide/documentation-rules.md, the documentation profile index, General style,
API/Javadoc style, Planning style, Example format when an example changes, this task, final
source/tests/generated Javadoc, Tensor API, glossary, model master plan, roadmap, and existing
Tensor/TensorFactory/TensorDescriptor/BFloat16Bits/DataType/Shape/LayoutDescriptor/storage
contracts. It must inspect actual implementation and evidence rather than rely on the handoff. It
must finalize all new/affected Javadocs, move only scalar/zero/one/like creation into current
API/glossary language, preserve tasks 0012E–0012F as planned, review links/anchors/fences/whitespace
and terminology, record reasoned existing-Javadoc and architecture/capability no-change
conclusions, and synchronize only the allowed planning files.

At the end, update this task, model master plan, and roadmap for status/evidence. Record local
decisions, known limitations, exact validation evidence including documentation-agent identity and
results, implementation notes, and the canonical completion summary. Do not mark task 0012D
Complete until implementation, tests, Javadoc, independent documentation pass, scope review, and
status synchronization all pass. Task 0012E then remains the next Draft frontier without a detailed
specification. Do not commit or push.
```

## Local decisions

- Primitive overloads preserve exact type semantics without a boxed scalar abstraction. Java
  literal type therefore matters: `1` is INT32, `1L` is INT64, `1.0f` is FLOAT32, and `1.0d` is
  FLOAT64.
- BFLOAT16 uses the explicit name `scalarBFloat16(float)` because a Java `short` would expose raw
  bits rather than a semantic scalar and a `float` overload already means FLOAT32.
- BOOL uses `boolean` because scalar creation is semantic; its destination carrier remains the
  canonical byte representation established by the model.
- Scalars are rank zero, deliberately correcting the legacy `[1]` convention to match `Shape`.
- Like means same shape and data type only. Explicit label and gradient intent prevent silent
  inheritance, while new dense layout and storage prevent view/alias propagation.
- Zeros use default-zero allocation without redundant population. Ones and scalars reuse typed
  flat import so carrier copy, BOOL normalization, label behavior, and ID semantics stay
  centralized.
- Package-private `TensorConstants` keeps constant mechanics out of the already broad public
  `TensorFactory` while avoiding a public concept or generic utility package.

## Known limitations

- There is no unlabeled/default-type scalar or constant convenience.
- BFLOAT16 scalar input is rounded from binary32; raw bit-pattern scalar input uses flat import.
- Like methods require a fully static template shape and do not preserve layout or metadata beyond
  shape and data type.
- One creation allocates an intermediate typed carrier before destination allocation; zero creation
  does not.
- No arbitrary fill value, conversion, range/prefix, random generation, typed access/export, or
  dynamic-shape allocation is provided.

## Validation evidence

Planning reviewed the authoritative architecture, planning/documentation rules, capability
baseline, completed data type/BFLOAT16, shape, layout, descriptor, Tensor, allocation, flat import,
nested import, current tests/Javadocs, and read-only legacy factory/Tensor/tests and usage evidence.

- The selected capability is exact semantic scalar plus zero/one and like-shaped independent dense
  constants for all six data types.
- The design changes only this task, model master plan, and roadmap during planning.
- Package placement, nine-path implementation limit, exact ten-method public API, validation order,
  edge cases, tests, documentation pass, and stop conditions are specified.
- Local Markdown links, anchors, fences, trailing whitespace, and `git diff --check` must pass.
- No code, tests, build, architecture, API, glossary, capability, or other-module file changes are
  part of planning.
- Planning-stage evidence preceded implementation; the implementation and independent
  documentation evidence below supersedes that earlier frontier state.
- Implementation context `/root/implement_model_0012d` added the exact ten public constant methods,
  package-private `TensorConstants`, and focused tests, then handed the actual shared-tree diff and
  validation requirements to the independent documentation context
  `/root/implement_model_0012d/review_model_0012d_docs`.
- The documentation context read the complete agent instructions, architecture contract, current
  architecture index, documentation workflow, General/API-Javadoc/Planning profiles, Example
  format, planning guide, roadmap, capability baseline, model master plan, this task, Tensor API,
  glossary, implementation, tests, generated Javadoc, and adjacent Tensor, descriptor, data type,
  shape, layout, BFLOAT16, storage, allocation, flat-import, and nested-import contracts.
- The documentation pass applied General style and API/Javadoc style to `TensorFactory`,
  `TensorConstants`, Tensor API, and glossary language; Planning style to this task, master plan,
  and roadmap; and Example format to the new complete constant-creation API example.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorFactoryTest` — passed; the focused suite contains
  7 tests with 0 failures, errors, or skips.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorFactoryConstantTest` — passed; the focused suite
  contains 9 tests with 0 failures, errors, or skips.
- `./gradlew :modules:model:test` — passed; 27 JUnit XML suites report 196 tests, 0 failures,
  0 errors, and 0 skipped.
- `./gradlew :modules:model:javadoc` — passed. Generated `TensorFactory` documentation was inspected
  for all six scalar signatures, four zero/one signatures, rendered BFLOAT16 conversion, dense and
  rank-zero semantics, nullability, ownership, validation, and allocation/ID side effects.
  Package-private `TensorConstants` is correctly outside the public generated index; its complete
  source Javadoc was reviewed.
- `./gradlew test` — passed; the standalone root run reported 36 actionable tasks with no failure.
- The documented constant-creation example was compiled and run from standard input with
  `java --source 26 --class-path modules/model/build/classes/java/main /dev/stdin`; it exited 0 and
  printed `3F80`, `[1, 1, 1, 1]`, `Shape[2, 2]`, `DENSE_CONTIGUOUS`, and `true` as documented.
- `javap -classpath modules/model/build/classes/java/main -p -c -s` confirmed the exact existing
  `TensorFactory` surface plus only the ten requested public methods. It confirmed public null-check
  order, scalar rank-zero construction, dense descriptor construction, zero delegation to one
  `allocate` call, scalar/one delegation to one matching flat import, and no direct constant-helper
  construction of Tensor, storage wrappers, or IDs.
- The same bytecode and reflection-backed test inspection confirmed package-private final
  `TensorConstants` has no fields, one private zero-argument constructor, exactly eight
  package-private static Tensor entries, and only private descriptor/import helpers. Descriptor
  validation follows static shape, checked count, Java array limit, contiguous layout, and gradient
  eligibility order; typed one carriers use `Arrays.fill`, BFLOAT16 conversion, and BOOL byte one.
- Production import review found only current model foundations and JDK memory/utility/atomic
  types. Targeted forbidden-vocabulary/state review found no reflection, graph/compiler/planning/
  runtime/backend type, Arena, random/service state, boxed scalar buffer, generic fill surface, or
  added dependency.
- A targeted Ruby local-link checker initially failed because its first expression interpolated a
  heading regex and its corrected version used unavailable `Array#filter_map`; a compatible rerun
  then found 152 local links, including 58 heading anchors, with 0 errors. All five changed
  documentation/planning files have balanced fences; trailing-whitespace review found 0 findings.
- `git diff --check` passed. Final status/scope review found exactly the nine permitted repository
  paths and no task-0012E/task-0013 specification. Task 0012D, the master plan, and roadmap agree
  that 0012D is `Complete` and 0012E is the next `Draft` frontier; 0012F and 0013 remain `Draft`.
- Existing `Tensor`, `TensorDescriptor`, `BFloat16Bits`, `DataType`, `Shape`, `LayoutDescriptor`,
  `HostTensorStorage`, and `MemorySegmentStorage` Javadocs remain accurate because this task composes
  their existing identity, gradient eligibility, conversion, static-shape, dense-layout, and
  borrowed-storage contracts without changing them. Existing allocation, flat-import, and nested-
  import Javadocs also remain accurate; the only stale in-scope wording was finalized in
  `TensorFactory` itself.
- Tensor API now describes only exact scalar/zero/one/like creation as current and keeps range,
  prefix, random, typed access/export, and execution planned. The glossary needed no new term;
  existing Tensor and Tensor factory entries and the implementation-status convention were updated.
- Architecture, focused architecture documents, ADRs, architecture tests, backend conformance,
  integration tests, and the capability baseline require no change: the capability baseline already
  selected constant creation, and implementation stays inside the existing model.tensor-to-
  datatype/shape/layout/storage direction without changing module, dependency, lifecycle, backend,
  compiler, runtime, or build behavior.

## Implementation notes

- Added six exact primitive scalar overloads to `TensorFactory`, including the explicitly named
  binary32-to-BFLOAT16 conversion, and added zeros, ones, zeros-like, and ones-like with explicit
  label and gradient intent.
- Added stateless package-private `TensorConstants` to centralize static/count/layout/gradient
  validation, typed scalar/one carrier construction, default-zero allocation, and matching flat-
  import dispatch without exposing another public abstraction.
- Kept like methods limited to template shape and data type. Every result uses a new canonical
  dense descriptor, independent storage and backing array, and factory-assigned identity.
- Updated only the exact TensorFactory API-shape expectation and added the focused constant suite;
  allocation, flat import, nested import, Tensor, storage, and identifier implementation remain
  unchanged.
- Finalized affected public and package-private Javadocs, Tensor API current behavior and example,
  glossary status language, and synchronized planning status in the independent documentation pass.

## Completion summary

- Completed changes: exact typed rank-zero scalar creation and independent dense zero, one,
  zero-like, and one-like creation for all six data types, with required validation, ownership,
  population, and ID side effects.
- Files changed or created: the four implementation/test paths and five documentation/planning
  paths listed under Affected files; no other repository path changed.
- Tests and validation: both focused suites, full model tests, model Javadoc, root tests, API example,
  generated Javadoc, bytecode/surface/import/state checks, links/anchors, fences, whitespace, status,
  scope, and `git diff --check` passed with the counts recorded above.
- Documentation-agent review: completed by
  `/root/implement_model_0012d/review_model_0012d_docs` using the API/Javadoc and Planning profiles,
  General style, and Example format.
- Documentation impact: `docs/api/tensor-api.md` and planning status/evidence were updated; no
  architecture or capability document change is required.
- Javadoc review: `TensorFactory` and `TensorConstants` were finalized; adjacent Tensor, descriptor,
  BFLOAT16, data type, shape, layout, storage, allocation, flat-import, and nested-import Javadocs
  remain accurate without changes.
- Glossary impact: existing implementation-status, Tensor, and Tensor factory language was updated;
  no reusable domain term was introduced.
- Unresolved issues: None.
- Follow-up required: None. Task 0012E remains the next Draft planning frontier without a detailed
  specification.

Status: Complete
