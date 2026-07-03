# Task 0012: Tensor Factory Foundation

## Status

Complete

## Goal

Make the completed package-private `Tensor` constructible through the smallest public creation
surface. Add one non-instantiable static `TensorFactory` that accepts a completed
`TensorDescriptor`, optional diagnostic label, and optional caller-supplied borrowed
`HostTensorStorage`, while assigning collision-free tensor identity across factory calls in one
Java virtual machine (JVM). Do not add storage allocation, data import, population conveniences,
provenance, operations, or runtime behavior.

## Scope

- Add exactly one public final `TensorFactory` class in the existing `model.tensor` package.
- Make the factory a non-instantiable static utility with one private constructor.
- Expose one storage-free convenience method and one complete creation method.
- Allocate non-negative `TensorId` values from one hidden JVM-wide monotonic sequence shared by
  every factory call.
- Make allocation safe under concurrent calls and fail permanently instead of wrapping after
  `Long.MAX_VALUE` is claimed.
- Reject null public arguments before allocating an identifier.
- Delegate label normalization and all host-storage compatibility/liveness validation to the
  existing package-private `Tensor` constructor.
- Attach only caller-supplied borrowed storage; create no arena, segment, array, or owned memory.
- Add one focused `TensorFactoryTest` covering API shape, successful creation, validation order,
  identifier consumption, concurrency, exhaustion, and exclusions.
- During implementation, update the Tensor API, glossary, task evidence, model master plan, and
  roadmap through the required separate clean-context documentation pass.

## Out of scope

- scalar, zeros, ones, zeros-like, ones-like, random, range, flat-array, nested-array, strict-prefix,
  cyclic-prefix, fill, copy, conversion, or typed-access methods
- storage allocation, `Arena` creation, segment allocation, implicit `MemorySegmentStorage`
  construction, owned storage, lifetime extension, closing, retaining, pooling, or `AutoCloseable`
- a public constructor, instance factory, builder, allocator interface, allocator object, registry,
  service locator, dependency injection, reset hook, test hook, seed, exposed counter, raw next ID,
  or caller-supplied `TensorId`
- UUIDs, random identifiers, persistence, serialization, cross-process/distributed identity, or an
  externally stable ID ordering/format
- changes to `Tensor` behavior, fields, constructor visibility/signature, storage synchronization,
  equality, hashing, diagnostics, label semantics, or compatibility validation
- changes to `TensorId`, `TensorDescriptor`, `HostTensorStorage`, `MemorySegmentStorage`,
  `DataType`, `Shape`, or `LayoutDescriptor` behavior or signatures
- descriptor construction from `DataType`, `Shape`, or `LayoutDescriptor`; default data types;
  layout inference/resolution; contiguous-layout synthesis; or materialization policy
- provenance, operation/input relationships, graph-local IDs, graph capture, gradients, trainable
  state, publication policy, typed scalar/bulk access, mutation versions, or expression methods
- compiler, planning, prepare, runtime, engine, backend, device, residency, execution, or tracing
  state and behavior
- dependencies, preview/incubator features, Gradle changes, architecture changes, architecture
  tests, backend-conformance tests, integration tests, or another module
- creating a detailed task specification for task 0012A, task 0013, or any later task

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md), especially public mutable `Tensor`,
  model ownership of `TensorFactory`, and forbidden runtime/backend state
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Architecture overview](../../../../architecture/overview.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md), especially identifier, public Tensor, host
  storage, and factory capabilities
- [Model master plan](../master-plan.md), especially `model.tensor`, `model.storage`, and package
  dependency direction
- [Task 0004](0004-typed-identifiers.md), which defines non-negative `TensorId` values without an
  allocation policy
- [Task 0007](0007-tensor-descriptor-model.md), which defines completed immutable descriptor
  validation and explicit resolved/unresolved layout state
- [Task 0010](0010-host-storage-abstraction.md), which defines exact-size borrowed host storage
  without allocation or lifetime ownership
- [Task 0011](0011-public-tensor-skeleton.md), which defines the package-private constructor and
  owns label/storage validation
- [Tensor API](../../../../api/tensor-api.md) and [glossary](../../../../glossary.md), which describe
  the current construction boundary and identity/storage terminology

## Legacy evidence and rejected coupling

Read-only inspection of `legacy/pre-rewrite` covered the old `tensor.Tensor`, mutable
`TensorMetadata`, `tensor.factory.TensorDataFactory`, `TensorArrayData`, constructor/data-factory
tests, typed-storage tests, shape-validation tests, and representative uses of scalar, zeros, ones,
range, typed flat arrays, nested arrays, random data, and prefix filling.

Useful capability evidence is that users need a public way to create tensor leaves, attach labels
and data, and eventually use common population patterns. The legacy implementation had no distinct
`TensorId`; object identity and compiler-assigned raw graph IDs served unrelated roles. That is
evidence for separating public tensor identity from graph identity, not for copying an identity
scheme.

The following legacy design is deliberately rejected:

- dozens of public constructors and duplicated static factory methods on `Tensor`;
- mutable metadata with defaulted null data type, mutable label/gradient/trainable state, and
  storage-indexing behavior;
- scalar shape `[1]`, rejection of zero-sized shapes, and `int`-limited shape/storage calculations;
- reflection-based nested-array flattening with accidental empty/ragged-array failures;
- per-data-type storage hierarchies and constructors coupled directly to typed backing arrays;
- factory allocation tied to storage and tensor construction without an explicit lifetime model;
- graph predecessors, operations, gradients, compiler/runtime resources, backend intent, and
  execution behavior on the public tensor; and
- accidental labels, defaults, validation messages, and runtime coupling.

Task 0012 selects only the public construction and identity-allocation foundation. The broad
factory capability baseline is split into later focused Draft rows rather than being claimed as
parity here.

## Architecture constraints

- Production packages remain below `io.github.pho001.synaptik.*`.
- `TensorFactory` lives in `io.github.pho001.synaptik.model.tensor` beside the package-private
  `Tensor` constructor it invokes. It may compose `TensorDescriptor`, `TensorId`,
  `HostTensorStorage`, and JDK values only.
- Package direction remains `model.tensor -> model.storage` plus the descriptor's existing
  foundational dependencies. `model.storage` must not depend on `model.tensor`, and the factory
  must not import graph, operation, compiler, planning, runtime, prepare, engine, backend, config,
  trace, or training types.
- The factory is a static utility, not a live runtime service, service locator, registry, or
  composition root. Its hidden state allocates model identity only and retains no tensor,
  descriptor, storage, arena, backend, graph, or execution object.
- The factory is the public construction path. The existing `Tensor` constructor remains the one
  package-private constructor and must not change visibility, signature, validation, or behavior.
- `TensorId` remains an opaque non-negative `long` value. The factory uses no negative sentinel and
  exposes no counter, reset, reservation, parsing, or caller-selected value.
- Every ID successfully allocated by `TensorFactory` is unique among all IDs allocated by this
  factory during the current JVM lifetime, across threads and all call sites. This JVM scope is
  sufficient to prevent separately factory-created tensors from colliding when later associated
  with publication or graph-capture state in the same process.
- Process restart, multiple JVMs, persisted artifacts, and manually constructed `TensorId` values
  are outside the uniqueness guarantee. The value is not a distributed or persistent identity.
- Allocation begins at zero in a fresh JVM. Candidates increase monotonically through
  `Long.MAX_VALUE` and are never reused by the public allocator. Numeric order, adjacency,
  gaplessness, allocation timing, and correspondence with method-completion order are not public
  caller contracts; callers treat IDs as opaque equality values.
- Concurrent allocation is linearized by JDK atomic compare-and-set operations. Completion order
  may differ from numeric allocation order.
- `Long.MAX_VALUE` is a valid final candidate. Exactly one caller may claim it. After it is claimed,
  every later allocation attempt fails with `IllegalStateException` and exact message
  `tensor identifier space exhausted`; the allocator never wraps to a negative or reused value.
- Factory-level null checks occur before ID allocation. Semantic label/storage validation occurs
  after allocation inside `Tensor`. A semantic construction failure therefore may consume an ID.
  Gaps are accepted because avoiding them would duplicate validation, widen `Tensor`, or require a
  rollback that is unsafe under concurrency.
- A consumed ID is never returned to the allocator, including when blank-label, storage
  compatibility, or attachment-liveness validation fails after allocation.
- The factory does not reimplement `TensorDescriptor`, label, or storage validation. A descriptor
  is already a completed validated value. Label stripping/blank rejection and storage data-type,
  resolved-span, and liveness checks remain in the `Tensor` constructor as one validation path.
- Public absence is represented with `Optional.empty()`, never null. The complete overload rejects
  null optional containers before allocation and passes non-null containers unchanged to `Tensor`.
- A present storage object is borrowed exactly as defined by `Tensor` and `HostTensorStorage`. The
  factory claims no ownership, accepts read-only storage through the existing constructor, and
  creates no lifetime promise.
- The storage-free overload creates a tensor with no host storage. It does not allocate zeroed
  memory, infer storage from shape, or resolve a layout.
- The factory accepts a completed `TensorDescriptor` exactly. It does not build or alter data type,
  shape, layout, or `requiresGrad`; it does not invent contiguous geometry for an unresolved
  descriptor.
- Label normalization and blank validation remain delegated to `Tensor`. No mutable label API or
  factory default label is introduced.
- Factory-created tensors retain ordinary object equality/hashing and existing stable diagnostics.
  Uniqueness of factory-assigned `TensorId` values does not turn `Tensor` into a structural value.
- Stable Java 26 and JDK concurrency primitives are sufficient. No dependency, preview feature,
  incubator API, UUID provider, or external service is allowed.
- If implementation requires storage ownership/allocation, another production type, a helper
  file, an allocator abstraction, a public test hook, a `Tensor` behavior/signature change, or an
  architecture/dependency change, stop and report the issue.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.tensor` — owns `Tensor`, `TensorId`, `TensorDescriptor`, and the
  public factory introduced here.
- `io.github.pho001.synaptik.model.storage` — owns caller-supplied borrowed host storage and remains
  independent of `model.tensor`.

Packages added or changed:

- No package is added. The existing `model.tensor` package gains one public final class.

Type placement:

- `io.github.pho001.synaptik.model.tensor.TensorFactory` — public construction boundary and
  JVM-wide allocator for public tensor identity.

Test placement:

- `io.github.pho001.synaptik.model.tensor.TensorFactoryTest` — mirrors the production package and
  verifies the public surface plus private static allocator state through focused reflection. Test
  helpers remain private inside this one test class.

## Required contract

### Type, fields, and constructor

Implement exactly one public final non-record class:

```java
public final class TensorFactory {
    private static final AtomicLong NEXT_TENSOR_ID = new AtomicLong();
    private static final AtomicBoolean MAXIMUM_TENSOR_ID_CLAIMED = new AtomicBoolean();

    private TensorFactory() { }
}
```

The class has exactly the two private static final allocator fields shown above and no instance
fields. `NEXT_TENSOR_ID` holds the next ordinary candidate until it reaches `Long.MAX_VALUE`.
`MAXIMUM_TENSOR_ID_CLAIMED` distinguishes the unclaimed final value from permanent exhaustion.
This boolean is allocator state, not a negative identifier sentinel.

The private constructor prevents supported instantiation and has no side effects. Do not add a
singleton instance, nested type, static initializer, tensor cache, storage cache, lock object,
random source, clock, service handle, or other field.

### Public methods

Expose exactly these two declared public methods:

```java
public static Tensor create(TensorDescriptor descriptor)

public static Tensor create(
        TensorDescriptor descriptor,
        Optional<String> label,
        Optional<HostTensorStorage> hostStorage)
```

`create(descriptor)` is the only convenience overload. It is exactly equivalent to:

```java
create(descriptor, Optional.empty(), Optional.empty())
```

It returns a new storage-free, unlabeled tensor that retains the exact descriptor reference. It
does not create a default layout or storage.

The complete overload retains no arguments in factory state. On success it returns the exact new
`Tensor` created by:

```java
new Tensor(nextTensorId(), descriptor, label, hostStorage)
```

after the required preallocation null checks. The returned tensor retains the exact descriptor and,
when present, exact borrowed storage reference; label value normalization remains the constructor's
contract.

Do not add overloads for labels alone, raw storage, data type, shape, layout, `requiresGrad`, raw
arrays, scalar values, fills, random values, IDs, or allocation policy.

### Validation order and exact failures

The complete overload performs only these factory checks, in this exact order:

1. Reject null `descriptor` with `NullPointerException` and exact message `descriptor`.
2. Reject null `label` optional with `NullPointerException` and exact message `label`.
3. Reject null `hostStorage` optional with `NullPointerException` and exact message `hostStorage`.
4. Allocate the next `TensorId`; if the full non-negative range is exhausted, throw
   `IllegalStateException` with exact message `tensor identifier space exhausted`.
5. Invoke the existing package-private `Tensor` constructor with the allocated ID and the exact
   validated references. Its existing deterministic label and storage validation types, order,
   messages, normalization, and atomic attachment semantics remain observable unchanged.

The convenience overload rejects a null descriptor with the same type/message before allocation
through the complete overload. `Optional` cannot contain a null element through its public
construction API; the factory defines no null element behavior or sentinel.

The factory must not inspect label text, storage data type/capacity/liveness, descriptor layout, or
descriptor shape. In particular, it must not call `strip`, `dataType`, `elementCapacity`,
`isAlive`, `layout`, `knownElementCount`, or `LayoutDescriptor.contiguous`.

If exhaustion and a semantic Tensor failure are both possible, exhaustion wins because allocation
precedes constructor semantic validation. Null factory arguments still win over exhaustion because
they are checked before allocation.

### Identifier allocation algorithm

Implement one private static method with this exact signature:

```java
private static TensorId nextTensorId()
```

Its algorithm is:

1. Read `NEXT_TENSOR_ID` into `candidate` inside a retry loop.
2. If `candidate` is less than `Long.MAX_VALUE`, compare-and-set `NEXT_TENSOR_ID` from `candidate`
   to `candidate + 1`. On success return `new TensorId(candidate)`; on failure retry.
3. If `candidate` equals `Long.MAX_VALUE`, compare-and-set
   `MAXIMUM_TENSOR_ID_CLAIMED` from false to true. On success return
   `new TensorId(Long.MAX_VALUE)`; on failure throw the exact exhaustion exception.

Do not call `getAndIncrement` because signed overflow would wrap. Do not synchronize on a lock,
reserve a negative value, decrement or roll back after failure, spin after permanent exhaustion,
or allocate an ID through random/UUID/time/hash/object identity.

The normal-state invariant is that `MAXIMUM_TENSOR_ID_CLAIMED` remains false while
`NEXT_TENSOR_ID` is below `Long.MAX_VALUE`. Reflection-based tests may temporarily place the two
atomics at the exhaustion boundary, restore their captured contents in `finally`, and must not
expose that mutation through production API.

### Construction failure and ID consumption

These cases do not consume an ID because the factory rejects them before allocation:

- null descriptor;
- null label optional; and
- null host-storage optional.

These cases consume the allocated ID because `Tensor` rejects them after allocation:

- present label that is blank after stripping;
- storage data type different from the descriptor;
- resolved layout span larger than storage capacity; and
- storage not alive at attachment time.

Tests must observe consumption through surrounding successful factory calls or private atomic
state inspection, not through a production counter/reset API. Consumed values create permitted
gaps and are never reused.

### Storage, descriptor, label, and identity behavior

- `create(descriptor)` returns a tensor whose `descriptor()` is the exact argument,
  `label()` is empty, and `hostStorage()` is empty.
- The complete overload returns a tensor with normalized label value and the exact compatible
  borrowed storage reference when present.
- Resolved and unresolved descriptors pass through unchanged. An unresolved descriptor with no
  storage remains unresolved and storage-free.
- Read-only and aliased storage behavior is exactly the existing `Tensor` behavior. The factory
  neither accesses memory nor changes ownership.
- Distinct successful factory calls return distinct tensor objects with distinct `TensorId` values,
  including under concurrency. Tensor equality remains object identity.
- No public behavior promises the first observed ID is zero, that sequential calls differ by one,
  or that a smaller ID means a method returned earlier. Zero is nevertheless a valid possible ID
  in a fresh JVM.
- Factory diagnostics add no new formatting or serialization contract. Returned tensors retain the
  existing metadata-only `toString()` behavior.

## Valid, invalid, concurrent, and exhaustion scenarios

| Scenario | Result |
|---|---|
| Fresh JVM, first valid factory call | May receive `TensorId(0)`; zero is valid |
| Valid descriptor through convenience overload | New unlabeled storage-free tensor retaining the exact descriptor |
| Valid descriptor, normalized label, matching live storage | New tensor retaining exact descriptor/storage and stripped label value |
| Fully static unresolved descriptor, no storage | Valid; no layout or storage is synthesized |
| Dynamic unresolved descriptor, no storage | Valid; no shape binding or layout is synthesized |
| Null descriptor/optional container | Exact factory null failure; no ID consumed |
| Present blank label | Existing Tensor blank-label failure; allocated ID consumed |
| Wrong-type, undersized, or dead storage | Existing Tensor failure in its existing order; allocated ID consumed |
| Failed semantic construction followed by success | Success receives a later candidate; the failed candidate is not reused |
| Many concurrent valid calls | Every returned tensor and ID is distinct; no collision |
| Concurrent calls when final candidate is unclaimed | Exactly one call can obtain `Long.MAX_VALUE`; all other allocation attempts fail |
| Any call after final candidate was claimed | Permanent exact exhaustion failure; no wrap or reuse |
| Exhausted allocator plus null descriptor | Null descriptor failure wins because precheck precedes allocation |
| Exhausted allocator plus blank label | Exhaustion wins because allocation precedes Tensor label validation |

The table describes observable outcomes and required implementation behavior. It does not make
numeric order or gaplessness a public caller contract.

## Affected files

Expected new production file:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorFactory.java`

Expected new test file:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryTest.java`

Expected Javadoc-only production update during the documentation pass:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java` — replace the
  now-stale statement that the factory is merely planned; do not change behavior, fields,
  signatures, constructor visibility, or any other contract

Expected documentation and planning updates during implementation:

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task specification
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless an inconsistency requires stopping:

- `TensorId`, `TensorDescriptor`, `HostTensorStorage`, `MemorySegmentStorage`, `DataType`, `Shape`,
  and `LayoutDescriptor` source Javadocs
- focused architecture documentation and the capability baseline

`TensorId` remains a general value type that can be manually constructed, so its existing warning
that the value type itself does not guarantee process-global uniqueness remains accurate. The new
factory documents the narrower guarantee for IDs it allocates.

## Maximum scope

This task may create or modify at most:

- one new production Java file;
- one existing production Java file for the stated Javadoc-only temporal correction;
- one new focused test Java file; and
- the five documentation and planning files listed as expected updates above.

Do not modify `Tensor` implementation or signatures, another existing Java/test file, Gradle,
`AGENTS.md`, `ARCHITECTURE.md`, focused architecture documents, architecture tests,
`capabilities.md`, another module, or unrelated documentation. Do not create another production
type/helper or a later task specification. If another file, public method, allocator concept,
storage ownership decision, dependency, or architecture clarification is required, stop and report
the issue instead of expanding this task.

## Javadoc requirements

- Every declared type, field, constructor, and method, including the private allocator, must have
  meaningful detailed Javadoc. Field comments must explain the two-atomic invariant without
  describing either field as public policy.
- Type Javadoc must define the factory as the public construction boundary, explain the static
  utility choice, JVM-wide uniqueness scope, thread safety, opacity/gap behavior, exhaustion, and
  the distinction from a runtime service locator or backend/graph registry.
- Type Javadoc must explain that the factory holds no tensor/storage/service state and does not
  allocate host memory, own lifetime, build descriptors, resolve layout, or create provenance.
- The private constructor Javadoc must explain non-instantiability and absence of instance state.
- Both public methods must document every parameter with constraints, nullability, optional
  absence semantics, exact-reference retention, borrowing, and validation ownership as applicable.
- Both public methods must document a non-null fresh Tensor result, factory-assigned identity, and
  exact descriptor/storage retention without promising numeric order or gaplessness.
- Both public methods must document every observable `NullPointerException`,
  `IllegalArgumentException`, and `IllegalStateException` condition they can expose, including
  exact factory messages and delegated Tensor conditions. The complete overload must explain which
  failures consume an ID.
- The allocator Javadoc must document candidate range, compare-and-set linearization, uniqueness,
  final-value claim, permanent exhaustion, absence of wrap/reuse/rollback, and its non-null return.
- Javadoc must not promise cross-process uniqueness, persistence, a reset, a caller-selected ID,
  numeric creation order, storage ownership, future liveness, layout inference, allocation/import,
  typed access, graph membership, publication behavior, compiler behavior, runtime residency, or
  backend support.
- The documentation-focused pass must review and finalize the temporal factory wording in
  `Tensor` Javadoc. It must independently review `TensorId`, `TensorDescriptor`,
  `HostTensorStorage`, `MemorySegmentStorage`, `DataType`, `Shape`, and `LayoutDescriptor` Javadocs
  and record why each remains accurate without edits or stop if a required correction is outside
  scope.

## Acceptance criteria

- Exactly one new public final non-record `TensorFactory` class and one focused
  `TensorFactoryTest` are added; no second production concept, nested type, helper file, allocator
  interface, builder, registry, or service locator appears.
- The factory has exactly two private static final fields of the required atomic types/names, no
  instance fields, exactly one private zero-argument constructor, exactly two declared public
  static methods with the required signatures, and exactly one private static allocator method.
- No public/protected constructor, singleton instance, reset/test hook, seed, counter accessor,
  ID reservation, raw next ID, or caller-supplied ID API exists.
- The storage-free overload creates a new unlabeled, storage-free tensor and retains the exact
  completed descriptor without synthesizing layout or storage.
- The complete overload accepts explicit optional label and optional existing borrowed storage,
  returns a new Tensor, and retains the exact compatible descriptor/storage references while
  existing Tensor label normalization remains unchanged.
- Factory null checks occur in the specified order/messages before allocation and therefore do not
  consume an ID.
- Blank label and type/span/liveness storage failures use the existing Tensor order, exception
  types, and exact messages after allocation; each consumes an ID, does not create a tensor, and
  never causes reuse.
- The factory performs no semantic label, descriptor, shape, layout, capacity, or liveness
  validation itself. Bytecode/source inspection shows the complete overload only null-checks,
  allocates, and invokes the package-private Tensor constructor.
- Sequential and concurrent successful factory calls never return equal IDs. A concurrency test
  starts callers together with stable JDK primitives and verifies the expected count of distinct
  tensors/IDs without assuming completion order.
- The allocator starts from zero in normal fresh static state, advances candidates monotonically by
  successful compare-and-set, permits `Long.MAX_VALUE` exactly once, then fails permanently with
  the exact exhaustion exception without negative values, wrap, reuse, rollback, or spinning.
- A focused reflective boundary test captures the two atomic contents, temporarily establishes the
  final-candidate state, proves exactly one final success under concurrency and exact permanent
  failure afterward, verifies null-versus-exhaustion precedence, and restores captured contents in
  `finally`. It introduces no production reset hook and does not run concurrently with other
  factory tests.
- Tensor objects retain ordinary object equality/hashing and existing metadata-only diagnostics;
  the factory does not override or replace those contracts.
- Existing `Tensor` fields, constructor, methods, validation, synchronization, and bytecode remain
  behaviorally unchanged. Only the permitted stale factory wording in its Javadoc changes.
- Production imports are exactly the focused model-storage/JDK utility/concurrency imports needed
  by the specified implementation. There is no `Arena`, `MemorySegment`, concrete
  `MemorySegmentStorage`, graph, operation, compiler, planning, runtime, prepare, engine, backend,
  config, trace, training, UUID, reflection, random, or external dependency import.
- Production source contains no allocation, array creation for tensor data, typed access, fill,
  copy, conversion, descriptor construction, layout resolution, ownership, close, provenance,
  operation, publication, graph, runtime, backend, or execution behavior.
- Reflection and `javap -p -c` confirm the exact API/field shapes, private construction, static
  modifiers, null-check order, compare-and-set algorithm, final candidate handling, permanent
  exhaustion, direct package-private Tensor construction, and absence of hidden state/behavior.
- Complete Javadocs satisfy every type/member requirement. Generated model Javadoc contains the
  public factory and overload contracts; source review covers private fields/constructor/allocator.
- A separate documentation-focused agent or thread with clean context independently inspects the
  final source, tests, generated Javadoc, test evidence, bytecode, imports, static state, and diff;
  finalizes factory and temporal Tensor Javadoc, Tensor API, glossary, task evidence/status,
  master-plan status, and roadmap status in the same overall change.
- The Tensor API moves only factory foundation/public creation and JVM-scoped ID allocation from
  planned to current. It continues to mark allocation, import, population, typed access,
  provenance, operations, gradients/publication, compiler, runtime, and backend behavior planned.
- The glossary marks `TensorFactory` and factory-assigned identity policy implemented, updates the
  Tensor construction wording, and keeps general `TensorId`, graph identity, publication,
  storage/lifetime, and runtime residency scopes distinct.
- Existing component Javadocs receive a reasoned no-change result except the explicit temporal
  `Tensor` wording. Focused architecture documents and `capabilities.md` remain unchanged because
  this is an already authorized subset and does not claim factory parity.
- Task, master-plan row/current status/decisions/notes, and roadmap frontier/table have matching
  final status. Task 0012A becomes the next `Draft` frontier only after 0012 completes; no detailed
  follow-up specification is created.
- No existing behavior/test, build file, architecture file, capability baseline, other module, or
  unrelated documentation changes, and no commit or push occurs.

## Tests / validation

Run after implementation and again after the separate documentation-focused pass:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

Manually verify:

- the final diff contains only the new factory, new focused test, Tensor Javadoc-only correction,
  Tensor API, glossary, this task, model master plan, and roadmap;
- reflection reports a public final non-record utility, two exact private static final atomic
  fields, no instance state, one private constructor, two exact public static overloads, one
  private static allocator, and no extra public API/interface;
- `javap -p -c` shows descriptor/label/storage null checks before allocation, no semantic
  validation in the factory, the exact retry/CAS algorithm, a single final-value claim, exact
  permanent exhaustion, and direct invocation of the package-private Tensor constructor;
- focused tests cover convenience/full creation, exact-reference retention, label normalization,
  storage-free/resolved/unresolved/borrowed storage cases, null order/messages/non-consumption,
  delegated failure order/messages/consumption, object/ID uniqueness, ordinary concurrency, final
  candidate concurrency, exhaustion permanence, and validation precedence;
- exhaustion tests use only test reflection to mutate existing atomic contents, restore captured
  contents in `finally`, and expose no production hook or field value;
- production imports and bytecode contain no forbidden package, storage allocation, memory access,
  random/UUID/time identity, synchronized lock, counter rollback, negative sentinel, or service
  lookup;
- no production `Arena`, `MemorySegment`, `MemorySegmentStorage`, array data, `DataType`, `Shape`,
  `LayoutDescriptor`, provenance, operation, graph, compiler, planning, runtime, prepare, backend,
  gradient, publication, or device reference appears in `TensorFactory`;
- generated Javadoc documents public construction, JVM uniqueness, concurrency, opacity, failure
  consumption, exhaustion, storage borrowing, and exclusions, while source Javadoc covers every
  private member;
- the documentation-focused context follows
  `docs/developer-guide/documentation-rules.md`, applies General plus API/Javadoc style to Java/API
  work and Planning style to planning updates, inspects actual source/tests/diff, and records its
  identity, selected profiles, exact commands/outcomes, limitations, component-Javadoc review,
  glossary impact, and architecture/capability no-change rationale;
- Tensor API and glossary current/planned language does not claim allocation/import/population,
  factory parity, typed access, provenance, operation, training, compiler, runtime, or backend
  behavior;
- all local Markdown links and anchors in the five changed documentation/planning files resolve,
  fences are balanced, terminology agrees with the glossary, and changed files have no trailing
  whitespace;
- task 0012 status matches master plan and roadmap, all 0012A–0012F rows remain `Draft`, 0013
  remains `Draft` after them, and no task-0012A/task-0013/follow-up specification exists; and
- no commit or push occurs.

## Dependencies

- Task 0004 is complete and provides validated non-negative `TensorId` values without allocation.
- Task 0007 is complete and provides the exact immutable `TensorDescriptor` contract.
- Task 0010 is complete and provides borrowed `HostTensorStorage` plus liveness/read-only facts.
- Task 0011 is complete and provides the one package-private Tensor constructor and canonical
  descriptor/label/storage validation path.
- The repository Java toolchain and release are 26; this task uses only stable JDK atomics and
  adds no build option or dependency.

## Follow-up tasks

- Task 0012A: owning host storage and allocation decision. Define the model-owned lifetime needed
  before factories may allocate memory; it depends on tasks 0010 and 0012.
- Task 0012B: flat typed tensor import. Define carrier/type matching, logical-count checks, copying,
  and storage population after an owning allocation contract exists; it depends on task 0012A.
- Task 0012C: nested typed tensor import. Define supported rectangular Java array forms, empty/ragged
  behavior, shape inference, and flattening on top of flat import; it depends on task 0012B.
- Task 0012D: constant tensor creation. Add scalar, zeros, ones, zeros-like, and ones-like in a
  focused family after typed allocation/population exists; it depends on task 0012B.
- Task 0012E: range and prefix population. Specify integer ranges plus strict/cyclic prefix data
  preparation without accidental legacy defaults; it depends on task 0012B.
- Task 0012F: random tensor creation. Decide random-source and reproducibility policy without live
  services before adding normally distributed population; it depends on task 0012B.
- Task 0013 remains after these factory capability rows and owns minimal provenance without
  changing tensor identity scope.

These are concise master-plan rows only. Do not create their detailed specifications during task
0012. Later planning may split a Draft family further before it becomes `Ready` if its file or
concept count would exceed the planning guide.

## Architecture impact

Expected impact: None.

The architecture already names `TensorFactory`, `Tensor`, `TensorId`, `TensorDescriptor`, and host
storage as `modules/model` responsibilities. This task exposes the completed package-private
constructor and adds only model-identity allocation state with no live service, storage ownership,
graph membership, runtime/backend state, module dependency, or lifecycle-stage change. Therefore
`ARCHITECTURE.md`, focused architecture documents, ADRs, and architecture tests require no update.
If implementation reveals otherwise, stop and report the conflicting rule and required decision
before editing architecture files.

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
- docs/developer-guide/documentation/README.md
- docs/developer-guide/documentation/general-style.md
- docs/developer-guide/documentation/api-and-javadoc-style.md
- docs/developer-guide/documentation/planning-style.md
- docs/developer-guide/documentation/example-format.md when an API example changes
- docs/planning/planning-guide.md
- docs/planning/roadmap.md
- docs/planning/modules/model/capabilities.md
- docs/planning/modules/model/master-plan.md
- docs/planning/modules/model/tasks/0004-typed-identifiers.md
- docs/planning/modules/model/tasks/0007-tensor-descriptor-model.md
- docs/planning/modules/model/tasks/0010-host-storage-abstraction.md
- docs/planning/modules/model/tasks/0011-public-tensor-skeleton.md
- docs/planning/modules/model/tasks/0012-tensor-factory.md
- docs/api/tensor-api.md
- docs/glossary.md
- current production/tests for Tensor, TensorId, TensorDescriptor, HostTensorStorage,
  MemorySegmentStorage, DataType, Shape, and LayoutDescriptor
- root/model Gradle configuration only to confirm Java 26

Implement task 0012 exactly as specified. Create only TensorFactory.java and
TensorFactoryTest.java for implementation and tests. TensorFactory must be one public final
non-record static utility with exactly two private static final fields:
AtomicLong NEXT_TENSOR_ID and AtomicBoolean MAXIMUM_TENSOR_ID_CLAIMED. Give it one private
zero-argument constructor, no instance state, exactly create(TensorDescriptor), exactly
create(TensorDescriptor, Optional<String>, Optional<HostTensorStorage>), and one private static
nextTensorId() method. Add no other public method, constructor, nested type, helper file, allocator
abstraction, singleton, registry, service, reset/test hook, seed, counter accessor, raw next ID, or
caller-supplied TensorId.

The complete overload must null-check descriptor, label, and hostStorage in that exact order before
allocating. It must not inspect or validate descriptor contents, label text, storage type/capacity/
liveness, shape, or layout. Allocate and then invoke the existing package-private Tensor
constructor so Tensor remains the sole semantic label/storage validation path. Null failures do
not consume IDs; Tensor semantic failures do consume IDs and are never rolled back or reused.

Implement the exact retry/CAS allocator in the task. Allocate zero through Long.MAX_VALUE
monotonically within one JVM. AtomicLong advances ordinary candidates; AtomicBoolean lets exactly
one caller claim Long.MAX_VALUE. Every later attempt must throw IllegalStateException with exact
message "tensor identifier space exhausted". Never wrap, reserve a negative sentinel, synchronize
on a lock, spin after exhaustion, roll back, use random/UUID/time/hash/object identity, or expose
allocator state. The hidden allocator is model identity state only, not a runtime service locator.

Attach only caller-supplied borrowed HostTensorStorage. Do not allocate Arena/native/heap memory,
construct MemorySegmentStorage, own/close/retain storage, implement AutoCloseable, synthesize
layout, build descriptors from component values, or add scalar/zeros/ones/random/range/flat-array/
nested-array/prefix/fill/typed-access methods. Do not add provenance, operation/input state,
graph IDs, gradients/trainable/publication state, copies/conversions, compiler/planning/prepare/
runtime/engine/backend/device/execution behavior, dependencies, preview/incubator features, or
Gradle changes.

Do not change Tensor behavior, fields, signatures, constructor visibility, validation,
synchronization, equality, hashing, or diagnostics. During the later documentation pass, permit
only the task's Javadoc-only correction to Tensor's stale statement that the factory is planned.
Do not modify other existing Java/tests, AGENTS.md, ARCHITECTURE.md, focused architecture docs,
architecture tests, capabilities.md, another module, or unrelated docs. Do not create a task-0012A,
task-0013, or other follow-up specification. Stop and report if any requirement exceeds the
affected-file/maximum-scope list or architecture uncertainty appears.

Add all Javadocs required by the task. The one focused test must verify exact API/field shape,
valid storage-free and full construction, exact reference/label/storage behavior, null validation
and non-consumption, delegated failures and consumption, ordinary concurrent uniqueness, and the
Long.MAX_VALUE boundary/permanent exhaustion. Boundary tests may reflectively mutate the contents
of the two private atomics only inside the test, must capture and restore both contents in finally,
must not run concurrently with other factory tests, and must not introduce a production hook.

Run the focused factory test, all model tests, model Javadoc, full repository tests, git diff
checks, and every reflection, javap, import, static-state, concurrency, exhaustion, documentation,
scope, and status check in the task.

After implementation and initial validation, hand the actual diff to a separate
documentation-focused agent or thread with a clean context in the same overall change. Keep task
0012 incomplete until that pass finishes. The handoff must include this task, source/test diff,
JVM identity scope and exhaustion, failure-consumption behavior, storage borrowing/no allocation,
package-private Tensor delegation, architecture constraints, expected Tensor API/glossary and
Tensor-Javadoc temporal updates, existing-Javadoc review list, and every validation command.

That documentation agent must independently read AGENTS.md, ARCHITECTURE.md,
docs/developer-guide/documentation-rules.md, the documentation profile index, General style,
API/Javadoc style, Planning style, Example format if an example changes, this task, final
source/tests/generated Javadoc, Tensor API, glossary, model master plan, roadmap, and the existing
Tensor/TensorId/TensorDescriptor/HostTensorStorage/MemorySegmentStorage/DataType/Shape/
LayoutDescriptor contracts. It must inspect the actual diff, bytecode, imports, and test evidence
rather than rely on the handoff summary. It must finalize TensorFactory Javadoc, only the permitted
temporal Tensor Javadoc wording, move only factory foundation into current Tensor API/glossary
language, preserve all deferred factory families, review links/anchors/fences/whitespace and
terminology, record reasoned existing-Javadoc and architecture/capability no-change conclusions,
and synchronize only the allowed planning files.

At the end, update this task file, the model master plan, and the roadmap for status/evidence.
Record local decisions, known limitations, exact validation evidence including the documentation
agent identity/results, implementation notes, and canonical completion summary. Do not mark task
0012 Complete until implementation, tests, Javadoc, the independent documentation pass, scope
review, and status synchronization all pass. Then task 0012A is the next Draft frontier. Do not
create its specification and do not commit or push.
```

## Local decisions

- A static utility is selected instead of an instance factory. JVM-wide identity must be shared
  across unrelated call sites, and user-created factory instances would either collide or require
  a shared allocator anyway. Static immutable API plus JDK atomics is thread-safe, leaves no
  injectable service/backend state, and future focused overloads can use the same allocator.
  Testability comes from public concurrent behavior plus narrowly scoped test-only reflection at
  the exhaustion boundary; it does not justify a production reset or allocator-injection hook.
- Two atomic fields are used instead of a negative/exhausted sentinel. `AtomicLong` handles the
  ordinary non-negative range, while `AtomicBoolean` lets the valid final long be claimed once and
  records permanent exhaustion without wrap.
- Identifier uniqueness is guaranteed only for values allocated by this factory in one JVM.
  `TensorId` remains freely constructible as a value type, and no distributed/persistent policy is
  implied.
- Factory null checks precede allocation for deterministic boundary failures without avoidable
  gaps. Label/storage semantics remain after allocation in `Tensor`; those failures consume IDs so
  validation stays canonical and concurrency needs no rollback.
- The only convenience is storage-free creation from a completed descriptor. This immediately
  makes Tensor publicly constructible without selecting defaults or introducing another
  capability family.
- The complete overload uses `Optional` for label and storage absence because the existing Tensor
  contract does. No null-as-absence or label-only overload is added.
- Caller-supplied storage is attached but never allocated or owned. The completed storage wrapper
  is borrowed and hides no arena, so allocation/import waits for a focused owning-lifetime
  decision.
- Completed descriptors pass through exactly. The factory does not duplicate descriptor
  validation or infer a resolved layout for unresolved geometry.
- The broad legacy factory baseline is preserved as explicit Draft follow-up families before
  provenance; task 0012 does not claim scalar/data/population parity.

## Known limitations

- The factory creates only metadata plus an optional borrowed storage association. It cannot
  allocate usable storage or populate values.
- A factory-created storage-free tensor may have resolved or unresolved layout but no memory. A
  later owning API must attach or allocate compatible storage before typed access/execution.
- IDs are unique only among this factory's allocations in one JVM. Restarted/separate JVMs and
  manually constructed `TensorId` values may use equal numeric values.
- IDs may contain gaps after semantic construction failures. Numeric order and adjacency are not
  caller contracts.
- Exhaustion is permanent for the JVM and intentionally has no reset hook. Reflection-based tests
  temporarily alter private atomics outside the supported API and restore their contents.
- The factory supplies no provenance, operations, graph membership, gradient/publication state,
  typed access, compiler integration, runtime residency, or backend support.

## Validation evidence

- Clean planning context read the complete agent instructions, architecture contract and focused
  architecture documents, documentation workflow and General/Planning/API-Javadoc profiles,
  planning guide, roadmap, capability baseline, model master plan, tasks 0004/0007/0010/0011,
  Tensor API, glossary, Java 26 Gradle configuration, and all requested current production/test
  contracts before defining this task.
- Read-only legacy inspection used `git ls-tree`, `git grep`, and
  `git show legacy/pre-rewrite:<path>` for `Tensor`, `TensorMetadata`, `TensorDataFactory`,
  `TensorArrayData`, constructor/data-factory/sequence/shape/storage tests, and representative
  factory call sites. The branch was not checked out or modified, and no legacy source, constructor
  surface, mutable metadata, storage hierarchy, runtime/compiler coupling, or accidental defaults
  were copied.
- Root and model Gradle configuration review confirmed Java toolchain and release 26 with common
  root configuration, no model override, and no preview/incubator setting. No build file changed.
- `git status --short --untracked-files=all`, `git diff --name-only`, and exact path review confirmed
  exactly three planning paths changed: this new task, the model master plan, and the roadmap. No
  Java, test, Gradle, agent instruction, architecture, API, glossary, capability-baseline,
  other-module, or unrelated documentation file changed.
- A targeted Ruby local-link check resolved all 64 Markdown links in the three changed planning
  files with no missing target or heading anchor. None of the changed links uses a heading anchor.
- Fence inspection reported balanced backtick fences: 14 markers in this task and two each in the
  model master plan and roadmap. `rg -n '[[:blank:]]+$'` found no trailing whitespace in any
  changed file.
- `git diff --check` passed for tracked changes. The first combined new-file whitespace command
  reached the check successfully but its shell-result wrapper used zsh's read-only `status` name;
  the corrected `git diff --no-index --check` rerun used `rc`, produced no whitespace diagnostic,
  and returned the expected difference status for this untracked task file.
- Planning-stage status and file-presence review confirmed task 0012 was linked and `Ready` in the
  task, master-plan row/current status/decisions/notes, and roadmap frontier/table before
  implementation. Tasks 0012A–0012F and 0013 remained concise `Draft` rows in execution order, and
  no task-0012A, task-0013, or other follow-up specification existed.
- Gradle tests and Javadoc were not run for the preceding planning-only change because no Java,
  test, build, API, or glossary file had changed. The implementation validation below supersedes
  that planning-stage limitation.
- Implementation context `/root/implement_model_0012` added only `TensorFactory.java` and
  `TensorFactoryTest.java`, then handed the actual shared-tree source, tests, and evidence to the
  independent documentation context
  `/root/implement_model_0012/review_model_0012_docs`. The documentation pass applied General
  style plus API/Javadoc style to Java and API reference work and Planning style to planning
  synchronization. Example format was reviewed but not applied because no example changed.
- The documentation context reread the complete agent instructions, architecture contract,
  current architecture index, documentation workflow and selected profiles, planning guide,
  roadmap, capability baseline, model master plan, this task, Tensor API, glossary, root/model
  Gradle configuration, final factory source/test, and the complete `Tensor`, `TensorId`,
  `TensorDescriptor`, `HostTensorStorage`, `MemorySegmentStorage`, `DataType`, `Shape`, and
  `LayoutDescriptor` source contracts before editing.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorFactoryTest` passed after the documentation edits:
  7 tests, 0 failures, 0 errors, and 0 skipped. The suite's
  `hasExactlyTheRequiredStaticUtilityShape` reflection test confirmed the public final non-record
  class, no interfaces, the two exact private static final atomic fields, no instance state, one
  private zero-argument constructor, two exact public static overloads, and one private static
  allocator. The same-thread suite also proved ordinary concurrent uniqueness, final-candidate
  concurrency, permanent exhaustion, precedence, and nested-finally restoration of both atomics.
- `./gradlew :modules:model:test` passed after the documentation edits. Aggregation of all 23
  module XML suites reported 161 tests, 0 failures, 0 errors, and 0 skipped; the focused XML suite
  independently reported its 7 passing tests.
- `./gradlew :modules:model:javadoc` passed after the documentation edits with no warning. The
  generated `TensorFactory.html` and `Tensor.html` pages were inspected for public construction,
  exact parameter/return/failure contracts, JVM scope, concurrency, opaque/gapped identity,
  failure consumption, `Long.MAX_VALUE` and permanent exhaustion, exact borrowed-storage
  retention, no allocation/ownership, delegated Tensor validation, and compiler/runtime/backend
  exclusions. Source review confirmed meaningful Javadocs on both private fields, the private
  constructor, and the private allocator, which generated public Javadoc intentionally omits.
- `./gradlew test` passed after the documentation edits: an initial run reported 36 actionable
  tasks, 1 executed and 35 up-to-date; the final rerun after evidence synchronization reported all
  36 up-to-date, with no failing task. No architecture, backend-conformance, or integration test
  was added because the change affects no dependency rule, backend behavior, or end-to-end
  execution.
- `javap -classpath modules/model/build/classes/java/main -p -c -s
  io.github.pho001.synaptik.model.tensor.TensorFactory` confirmed the exact two fields and three
  declared methods, descriptor/label/storage null checks before allocation, direct package-private
  Tensor construction, ordinary `AtomicLong.compareAndSet`, one final
  `AtomicBoolean.compareAndSet`, exact exhaustion text, retry only after ordinary contention, and
  no rollback, negative sentinel, wrap, semantic validation, lock, or hidden service state.
- A supplemental `jshell --class-path modules/model/build/classes/java/main --feedback concise`
  reflection probe printed the same exact class, field, constructor, and method shape. The host
  then made JShell exit non-zero while flushing macOS preference history; rerunning with
  `-J-Djava.util.prefs.userRoot=/private/tmp/synaptik-jshell-prefs` produced the same correct output
  and the same host-only flush limitation. This did not limit acceptance because the focused JUnit
  reflection test completed successfully.
- Import review found exactly `HostTensorStorage`, `Objects`, `Optional`, `AtomicBoolean`, and
  `AtomicLong`. Targeted source and bytecode searches found no arena/segment construction,
  `MemorySegmentStorage`, arrays, descriptor/layout construction, semantic label/storage
  inspection, random/UUID/time identity, synchronized lock, registry/service lookup, provenance,
  graph, gradient/publication, compiler, planning, prepare, runtime, backend, or device behavior.
- Existing component Javadocs required no edits: `TensorId` remains a freely constructible value
  type and correctly disclaims global uniqueness; `TensorDescriptor` still owns completed logical
  validation and no allocation; `HostTensorStorage` remains the raw borrowed-storage boundary;
  `MemorySegmentStorage` still wraps only caller-supplied exact memory without allocation or
  ownership; `DataType` remains backend-independent element metadata; `Shape` remains logical
  geometry without storage; and `LayoutDescriptor` remains resolved logical geometry without
  storage, materialization, runtime, or backend policy. Only Tensor's stale temporal construction
  paragraph required the permitted Javadoc-only correction; its fields, signatures, constructor,
  validation, synchronization, and executable behavior remain unchanged.
- `docs/api/tensor-api.md` now treats only descriptor-based public construction and JVM-scoped
  factory identity as current. `docs/glossary.md` now defines the implemented TensorFactory and
  its narrower factory-assigned identity policy while preserving manual `TensorId`, graph-local
  identity, standalone publication binding, borrowed host-storage lifetime, and runtime residency
  as distinct scopes. Owning allocation, import/population, typed access, provenance, operations,
  gradients/publication, compiler, runtime, and backend families remain planned; no broad factory
  parity is claimed.
- `ARCHITECTURE.md` and the focused architecture index required no update because the contract
  already authorizes `TensorFactory`, `Tensor`, model identity, and borrowed host storage in
  `modules/model`; the implementation changes no module dependency, lifecycle stage, storage
  ownership, graph/publication boundary, runtime state, or backend state. `capabilities.md` also
  remains unchanged because it is a broader multi-task factory baseline, not evidence that
  allocation/import/population parity is complete.
- The targeted Ruby Markdown validator resolved 135 local links, including 58 heading anchors,
  across the five changed documentation/planning files. Fence validation reported balanced
  backtick fences: 24 markers in the Tensor API, 14 in this task, two in the master plan, two in
  the roadmap, and none in the glossary. `rg -n '[[:blank:]]+$'` reported no trailing whitespace
  in all eight allowed paths.
- `git diff --check` passed. `git diff --no-index --check /dev/null <path>` produced no whitespace
  diagnostic for each of the three untracked files, with only the expected difference exit status.
  Exact status review found only the eight allowed paths: the new factory and test, Tensor's
  Javadoc-only correction, Tensor API, glossary, this task, model master plan, and roadmap.
- Final synchronization marks task 0012 `Complete` in this task, the model master plan, and the
  roadmap. Task 0012A is the next `Draft` planning frontier; tasks 0012A–0012F and 0013 all remain
  `Draft` in order, and no detailed task-0012A, task-0013, or other follow-up specification exists.
  No commit or push was performed.

## Implementation notes

- Added the non-instantiable static `TensorFactory` with two exact public creation overloads and a
  hidden two-atomic allocator covering the full non-negative `long` range without wrap or reuse.
- Kept public-argument null rejection before allocation and delegated label/storage semantics to
  the existing package-private Tensor constructor after allocation, preserving one validation
  path and intentional ID consumption on semantic failure.
- Added the focused same-thread factory suite covering exact API shape, descriptor/storage
  retention, label normalization, null ordering/non-consumption, delegated failures/consumption,
  ordinary concurrency, final-value concurrency, permanent exhaustion, and restoration.
- Finalized all factory member Javadocs, the permitted temporal Tensor Javadoc correction, the
  Tensor API and glossary current/planned boundary, and synchronized planning status/evidence.

## Completion summary

- Completed changes: implemented the minimal descriptor-based public TensorFactory, JVM-wide
  concurrent non-negative ID allocation with exact permanent exhaustion, optional borrowed-storage
  attachment, focused tests, complete Javadocs, API/glossary updates, and planning synchronization.
- Files changed or created: `TensorFactory.java`, `TensorFactoryTest.java`, Tensor Javadoc,
  `docs/api/tensor-api.md`, `docs/glossary.md`, this task, the model master plan, and the roadmap.
- Tests and validation: focused factory tests 7/7; model tests 161/161 across 23 suites; model
  Javadoc and full repository tests passed; bytecode, reflection, generated Javadoc, imports,
  forbidden state/behavior, links/anchors, fences, whitespace, exact scope, and status checks
  passed. The supplemental JShell history-flush limitation did not affect the passing JUnit
  reflection validation.
- Documentation-agent review: completed in canonical clean context
  `/root/implement_model_0012/review_model_0012_docs` using General, API/Javadoc, and Planning
  profiles; no example changed.
- Documentation impact: public factory foundation and JVM-scoped factory identity are current;
  allocation/import/population, typed access, provenance, operations, gradients/publication,
  compiler, runtime, and backend behavior remain planned.
- Javadoc review: TensorFactory and the permitted Tensor temporal wording are final; all seven
  reviewed component contracts remain accurate for the reasons recorded above.
- Glossary impact: TensorFactory, construction, and factory-assigned identity are current while
  manual IDs, graph identity, publication, borrowed lifetime, and runtime residency remain
  distinct.
- Architecture impact: None; no architecture document, ADR, architecture test, module boundary,
  dependency, or lifecycle contract changed. The capability baseline remains unchanged and does
  not claim task-level completion.
- Unresolved issues: None.
- Follow-up required: None for task 0012. Task 0012A remains the next separate `Draft` planning
  frontier.

Status: Complete
