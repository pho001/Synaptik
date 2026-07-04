# Task 0013A: Full-Value and Identity-Matrix Tensor Creation

## Status

Complete

## Goal

Complete the current eager factory foundation with type-safe full-value tensors and dense
rectangular identity matrices. `full` fills every logical element from one exact primitive scalar;
`identityMatrix` creates a rank-two matrix with typed one on the main diagonal and typed zero
elsewhere; `eye` is an exact convenience alias for the canonical identity-matrix method.

Every result is a new canonical dense-contiguous leaf tensor with independent JVM-managed heap
storage, factory-assigned identity, and empty provenance. This task adds no expression Operation,
graph behavior, generic conversion policy, or typed Tensor mutation API.

## Scope

- Add six exact primitive-carrier full-value factory methods: FLOAT64, FLOAT32, explicitly converted
  BFLOAT16, INT32, INT64, and semantic BOOL.
- Require a fully static caller-supplied Shape for every full-value result and preserve scalar and
  zero-element shape semantics.
- Add one canonical rectangular `identityMatrix` factory method for all six current data types.
- Add one `eye` method with the exact same parameters and behavior, implemented only by delegating
  to `identityMatrix`.
- Accept non-negative `long` row and column counts, including rectangular and zero-element
  matrices, and synthesize rank-two canonical dense geometry.
- Reuse the existing package-private `TensorConstants` helper, descriptor validation, exact typed
  flat import, ID allocation, storage ownership, label normalization, and gradient eligibility.
- Extend only exact factory/helper surface tests and add one focused full/identity behavior suite.
- Finalize affected Javadocs, Tensor API, glossary, task evidence, master plan, and roadmap through
  an independent documentation pass during implementation.

## Out of scope

- boxed/generic `Number` or `Object` values, `DataType + double` conversion, caller-selected
  conversion, implicit widening/narrowing/rounding, numeric truthiness, strings, or null values
- raw-bit BFLOAT16 full input; BFLOAT16 semantic input remains an explicitly named binary32
  conversion, while raw short arrays remain available through flat import
- `fullLike`, default shape/type/value/label/gradient overloads, mutable fill, in-place fill,
  post-construction writes, or typed Tensor access/export
- square-only convenience overloads, size-only `eye`, Shape-based identity input, diagonal offset,
  batched identity, sparse identity, band matrices, triangular matrices, or view-based identity
- preserving another Tensor's shape/layout/storage/label/gradient/provenance or adding an identity-
  like method
- treating `eye` as a separate implementation, allocating before canonical delegation, changing
  validation/messages, or returning a shared/cached Tensor
- operations, provenance creation, graph capture/traversal, compiler constants, constant folding,
  autograd, runtime/backend generation, device/native storage, or prepared execution
- new production types/packages, storage contracts, dependencies, preview/incubator features,
  Gradle, architecture changes, another module, or task-0014 specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Architecture overview](../../../../architecture/overview.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)
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
- [Tensor API](../../../../api/tensor-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

Read-only inspection of the legacy Tensor factory and tests found scalar, zeros, ones, like-shaped
constants, ranges, and fixture-population helpers, but no stable public `full`, `identityMatrix`, or
`eye` contract to copy. These capabilities are explicit additions selected in the current model
capability baseline. Legacy code remains evidence for exact carrier handling and independent leaf
storage only; it is not source or architecture for this task.

## Architecture constraints

- Work remains entirely in `modules/model` and the existing
  `io.github.pho001.synaptik.model.tensor` package.
- Public creation methods belong to `TensorFactory`; typed population mechanics remain in the
  existing stateless package-private `TensorConstants` helper.
- Every result is eager copied leaf data with `Optional.empty()` provenance through existing flat
  import. Do not call `createDerived` or construct an Operation.
- Primitive overloads infer exact DataType and perform no implicit conversion. Only
  `fullBFloat16(float, ...)` explicitly invokes the completed BFLOAT16 conversion.
- Full-value and identity results synthesize only canonical dense-contiguous descriptors and new
  independent heap storage. They accept no layout or storage input.
- Existing `TensorDescriptor` remains authoritative for requires-grad eligibility.
- Existing flat import remains authoritative for destination allocation, copying, BOOL
  normalization, label validation, identity allocation, and failure side effects.
- `eye` delegates exactly once to the public canonical `identityMatrix` method and owns no
  validation, descriptor, carrier, population, allocation, or ID logic.
- No new package is justified: the behavior is cohesive constant creation and requires existing
  package-private factory/helper collaboration.
- Stop if implementation requires a generic scalar abstraction, conversion policy, new storage or
  Tensor mutation API, Operation/provenance, dependency, module, or architecture decision.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.tensor` — owns TensorFactory, TensorConstants, descriptors,
  flat-import collaboration, and focused tests.
- `io.github.pho001.synaptik.model.datatype` — supplies exact data types and BFLOAT16 conversion.
- `io.github.pho001.synaptik.model.shape` — supplies canonical static rank-two, scalar, and
  zero-element shapes.
- `io.github.pho001.synaptik.model.layout` — supplies canonical dense-contiguous geometry through
  existing TensorConstants descriptor construction.

Packages added or changed:

- No package is added. Only the existing `model.tensor` package changes.

Type placement:

- `TensorFactory` receives exactly eight new public methods: six full-value overloads,
  `identityMatrix`, and `eye`.
- Existing package-private `TensorConstants` receives exactly seven new package-private entries:
  six full-value overloads and one identity-matrix entry. It receives no `eye` entry.
- Existing `TensorFactoryTest` and `TensorFactoryConstantTest` change only their exact public/helper
  surface expectations.
- New `TensorFactoryFullIdentityTest` owns focused value, geometry, validation, ownership, alias,
  and failure-side-effect tests.

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

| Method value | Data type | Stored value |
|---|---|---|
| `double` | `FLOAT64` | exact binary64 value and raw signed-zero/NaN payload |
| `float` | `FLOAT32` | exact binary32 value and raw signed-zero/NaN payload |
| `fullBFloat16(float)` | `BFLOAT16` | `BFloat16Bits.fromFloat(value)` in every element |
| `int` | `INT32` | exact signed 32-bit value |
| `long` | `INT64` | exact signed 64-bit value |
| `boolean` | `BOOL` | canonical false byte `0` or true byte `1` |

Every public full method validates non-null `shape` and `label` in that order before helper
delegation. No default label, unlabeled overload, DataType argument, raw short/byte overload, boxed
value, full-like variant, or conversion overload is added.

### Full-value validation and population

Reuse the existing TensorConstants descriptor path unchanged:

1. reject dynamic shape with `IllegalArgumentException` and exact message
   `constant tensor creation requires a fully static shape: <shape>`;
2. obtain checked `shape.knownElementCount()`; non-zero multiplication overflow remains
   `ArithmeticException`;
3. reject count above `Integer.MAX_VALUE` with exact message
   `constant tensor element count exceeds Java array limit: required=<required>, maximum=2147483647`;
4. construct canonical contiguous layout; checked geometry overflow remains `ArithmeticException`;
5. construct TensorDescriptor with the inferred exact type and explicit gradient request; existing
   gradient eligibility and message remain authoritative;
6. allocate exactly one matching source carrier of logical element count;
7. fill the carrier with `Arrays.fill` and the exact mapped value;
8. delegate exactly once to the matching `TensorFactory.fromFlatArray` overload.

Do not special-case zero, one, false, true, empty output, or scalar output through another public
factory method. This preserves exact negative-zero/NaN bits and one uniform source/carrier/copy/ID
path. Scalar Shape has one element. A static shape with a zero dimension creates an empty source
and destination while retaining the requested rank/shape.

### Public identity-matrix methods

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

`identityMatrix` is canonical. It supports all six current data types. It creates Shape
`[rows, columns]`, stores typed one at coordinates `(i, i)` for
`0 <= i < min(rows, columns)`, and leaves all other positions at typed zero. This is valid for
square, wide, tall, and zero-element matrices.

Representations are:

| Data type | Diagonal | Off diagonal |
|---|---|---|
| `FLOAT64` | `1.0d` | JVM default `0.0d` |
| `FLOAT32` | `1.0f` | JVM default `0.0f` |
| `BFLOAT16` | `BFloat16Bits.fromFloat(1.0f)` | raw bits `0` |
| `INT32` | `1` | `0` |
| `INT64` | `1L` | `0L` |
| `BOOL` | canonical byte `1` | canonical byte `0` |

The public canonical method null-checks `dataType` and `label` in that order before helper
delegation. The helper validates and constructs in this order:

1. negative rows: `IllegalArgumentException`, message
   `identity matrix rows must be non-negative: <rows>`;
2. negative columns: `IllegalArgumentException`, message
   `identity matrix columns must be non-negative: <columns>`;
3. create `Shape.of(rows, columns)`;
4. reuse the existing descriptor path for checked logical count, Java-array limit, canonical dense
   layout, and gradient eligibility;
5. allocate exactly one matching default-zero primitive carrier of logical element count;
6. write typed one only at row-major indices `i * columns + i` for the diagonal length;
7. delegate exactly once to the matching flat-import overload.

Because the result is rank two, non-negative rows and columns are the complete dimension input.
Zero by any non-negative column count and any non-negative row count by zero are valid empty
matrices. Positive count overflow remains `ArithmeticException`; non-overflowing count above the
array limit uses the existing constant-count error message.

`eye` contains exactly one return delegation to `identityMatrix` with unchanged arguments. It
performs no own null/dimension/gradient/label check and no allocation. Each successful call still
returns a fresh Tensor because the canonical method allocates normally. Alias equality means equal
descriptor/value behavior for equal arguments, not the same Tensor, ID, storage, descriptor, or
layout object across separate calls.

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
package-private entry surface contains the existing eight constant entries plus these seven, for
exactly fifteen entries. It has no `eye`, generic fill, identity strategy, cache, registry,
singleton, public/protected member, nested type, or production test hook.

Private helper methods may share exact carrier fill/import and diagonal population only when the
specified public/helper surface, validation order, messages, carrier counts, and single flat-import
behavior remain observable and unchanged.

## Ownership and failure effects

- Every success creates a new descriptor, layout, source carrier, destination carrier, storage,
  Tensor, and ID; no result or intermediate is cached or shared.
- Source carriers are not exposed or retained. Flat import copies them into independent storage.
- Results have empty provenance and no operation, graph, runtime, or backend state.
- Public null, shape/dimension, count, layout, and gradient failures precede source/destination
  allocation and ID consumption.
- Source-carrier OOME propagates before destination or ID allocation.
- A blank label is rejected by existing Tensor validation after source creation, destination
  allocation, and ID allocation, so it consumes one ID without rollback.
- Identifier exhaustion occurs after source and destination allocation and before publication;
  no allocation or ID is rolled back.
- An unexpected flat-copy failure occurs after ID allocation and consumes that ID.
- `eye` has exactly the same validation and side effects as `identityMatrix` because it performs
  only canonical delegation.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorFactory.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorConstants.java`

Existing tests, exact surface only:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryConstantTest.java`

New focused test:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryFullIdentityTest.java`

Documentation/planning during implementation:

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless a task-related inconsistency requires stopping:

- `docs/planning/modules/model/capabilities.md` — already contains the selected full/identity
  capability and must not be expanded during implementation.
- Existing DataType, BFloat16Bits, Shape, LayoutDescriptor, TensorDescriptor, Tensor,
  TensorProvenance, storage, allocation, and flat-import Javadocs.
- Compile API, Training API, focused architecture documentation, and architecture tests.

## Maximum scope

At most two production files, two existing exact-surface tests, one new focused test, and five
documentation/planning files: ten paths total. TensorFactory and TensorConstants plus their exact-
surface tests must change atomically so the module remains buildable.

Do not modify existing behavior assertions in TensorFactoryConstantTest, another factory helper or
test, capabilities, existing completed tasks, Tensor/descriptor/storage/provenance/operation/graph
code, Gradle, AGENTS, architecture docs/tests, another module, or unrelated documentation. Do not
create task 0014. Stop beyond ten paths or if implementation needs a new production concept,
package, conversion policy, Tensor mutation, Operation, or architecture decision.

## Javadoc requirements

- Update TensorFactory type Javadoc to include exact full-value and identity-matrix leaf creation
  without implying general fill mutation or Operation semantics.
- Fully document all eight public methods: exact primitive/type mapping, BFLOAT16 conversion,
  BOOL semantics, static/scalar/empty shapes, rectangular identity, diagonal definition, dense
  layout, explicit gradient/label behavior, ownership, provenance absence, validation order,
  allocation/ID/failure effects, parameters, results, and exceptions.
- Explicitly document `eye` as pure delegation and define alias equality versus object identity.
- Update TensorConstants type Javadoc and document every new package-private entry and private
  population helper with exact carrier, descriptor, fill/diagonal, import, and failure behavior.
- Review related component Javadocs and record reasoned no-change conclusions or stop when a
  required correction falls outside the authorized files.
- The documentation pass must add a newcomer-readable Tensor API example showing a nontrivial
  full tensor and one rectangular identity matrix with concrete row-major values and line-by-line
  commentary under the Example-format profile.

## Acceptance criteria

- TensorFactory exposes exactly six new type-safe full-value methods, one canonical
  identityMatrix, and one exact eye alias, with no other public change.
- TensorConstants exposes exactly seven matching package-private entries and no eye entry; it
  remains stateless with exactly fifteen package-private constant entries after the task.
- Full-value methods support all six data types, preserve exact floating raw values, explicitly
  convert BFLOAT16, use canonical BOOL, support scalar/empty/static shapes, and reject dynamic or
  over-limit shapes before allocation/ID.
- Identity supports all six data types; square, wide, tall, and zero-element rank-two shapes;
  exact main-diagonal typed ones; default typed zeros elsewhere; canonical dense geometry; and
  explicit gradient eligibility.
- Negative rows/columns, positive count overflow/limit, invalid gradients, null references, blank
  labels, OOME ordering, and identifier exhaustion follow the exact contracts and messages.
- Eye bytecode/body is one unchanged-argument call to identityMatrix and has no helper entry or
  independent behavior.
- Every result is a fresh provenance-free leaf with independent descriptor/layout/storage/backing
  array/ID and no source retention, Tensor mutation, Operation, or cross-layer state.
- Existing scalar/zero/one/like behavior and tests remain unchanged except exact helper/API surface
  assertions.
- Focused and aggregate tests, Javadoc, root tests, javap/import/bytecode/manual checks,
  documentation links/examples/formatting, exact ten-path scope, and status synchronization pass.
- A separate clean-context documentation agent finalizes Javadocs, Tensor API, glossary, planning
  evidence/status, component no-change reviews, and documentation validation in the same change.
- Task 0013A becomes Complete only after both passes. The post-foundation checkpoint becomes the
  next planning action; task 0014 remains Draft without a detailed specification.

## Tests / validation

Run before and after the documentation-focused review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryConstantTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryFullIdentityTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

Focused tests cover exact factory/helper surfaces; all carriers; raw signed zero/NaN; BFLOAT16
rounding; BOOL canonical bytes; scalar/empty/static/dynamic/over-limit full shapes; square/wide/
tall/zero identity matrices; row-major diagonal positions; alias equivalence/fresh identity; dense
metadata; empty provenance; independence; null/negative/count/gradient/label validation and exact
messages; allocation/ID effects; exhaustion-state restoration; and preservation of completed
constant behavior. Do not force OOME.

Manually inspect `javap -p -c -s` and source for exact overload descriptors, helper visibility,
fifteen package-private entries, exact carrier arrays and `Arrays.fill`, default-zero identity
carriers, diagonal index arithmetic, one flat import, `eye -> identityMatrix` delegation, unchanged
allocator/import/constant bytecode, no provenance/Operation/forbidden imports or state, and no new
package. Validate generated Javadocs, executable/concrete API example, links, anchors, fences,
whitespace, exact ten paths, synchronized status, and absence of a task-0014 specification.

## Dependencies

- Task 0012B supplies exact typed copied flat import, destination allocation, BOOL normalization,
  label validation, and ID behavior.
- Task 0012D supplies TensorConstants, canonical constant descriptor validation, exact scalar/one
  carrier practice, zeros allocation, and all-data-type one representation.
- Task 0013 confirms that eager public factory results remain provenance-free leaves.

## Follow-up tasks

- After task 0013A, perform the model foundation checkpoint recorded in the roadmap. Decide whether
  task 0014 remains next or whether a named cross-module vertical slice should be planned.
- Task 0014 remains the Draft elementwise-operation frontier if sequential model work continues.
- Typed access/export, general mutable fill, identity Operations, batched/offset/sparse identities,
  compiler constants, and backend/runtime generation remain separate future work.

Do not create detailed follow-up specifications in this task.

## Architecture impact

Expected impact: None. This is model-owned eager leaf construction through existing descriptors,
heap storage, flat import, and identity allocation. It changes no architecture rule, package/module
direction, lifecycle stage, Operation/provenance semantics, or runtime/backend boundary.

If implementation requires Tensor mutation, expression provenance, compiler behavior, another
module/package, new dependency, or changed storage ownership, stop and report the issue.

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
  abstraction. BFLOAT16 conversion is named explicitly, matching existing scalar creation.
- `identityMatrix(rows, columns, ...)` is canonical and rectangular; a size-only overload would
  duplicate surface without adding capability. `eye` exists only as the requested exact alias.
- Identity supports all six data types. BOOL uses true on the diagonal and false elsewhere; integer
  and BOOL gradient eligibility remains rejected by TensorDescriptor when requested.
- Full always uses a filled source plus flat import, even for zero/one, so negative-zero/NaN bits
  and one deterministic side-effect path are preserved.
- Identity uses one default-zero source and writes only diagonal ones; no general typed mutation or
  access API is introduced.
- No new helper/package is needed because TensorConstants already owns constant descriptor and
  carrier population mechanics.

## Known limitations

- Full values are limited to exact primitive carriers and explicit semantic BFLOAT16 conversion;
  no arbitrary conversion or raw BFLOAT16 scalar overload exists.
- Identity is eager, dense, rank two, main-diagonal only, and Java-array-sized. It has no batch,
  offset, sparse representation, or Operation form.
- Full and identity allocate both a source carrier and copied destination carrier; zero-specific
  allocation optimization remains available only through `zeros`.
- Empty identity matrices may retain a very large extent on the other zero-product axis because
  Shape supports non-negative long dimensions; no physical elements are allocated.
- No typed read/export API is added; focused tests inspect existing host segments internally.

## Validation evidence

Planning reviewed architecture/planning/documentation rules, current capability/master/roadmap
frontier, tasks 0012B/0012D/0013, TensorFactory/TensorConstants exact surfaces, shape/count/layout/
descriptor/storage/import contracts, current constant tests, Tensor API/glossary, and read-only
legacy TensorDataFactory evidence. Legacy contained no stable public full/identity/eye contract;
the capability is an explicit current addition. The design reuses existing packages and fits ten
implementation/documentation paths with no architecture or dependency change.

- Implementation context `/root/implement_model_0013a` added exactly six public type-safe full
  methods, one public `identityMatrix`, one public-only `eye`, the seven matching package-private
  `TensorConstants` entries, and the focused test coverage. Clean documentation context
  `/root/implement_model_0013a/review_model_0013a_docs` independently inspected the actual final
  source, tests, diff, generated Javadocs, bytecode, API reference, glossary, and planning state.
  It applied General plus API/Javadoc style to Java/API/glossary work, Planning style to this task,
  the model master plan, and roadmap, and Example format to the new executable API example.
- The first combined focused-test invocation failed during test compilation because the updated
  helper-surface test reassigned a local set later captured by assertion lambdas. The test was
  corrected to initialize one mutable set without reassignment; all required focused commands
  below then passed independently in both implementation and final documentation validation.
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
- `./gradlew :modules:model:javadoc` — passed without Javadoc errors. Generated
  `TensorFactory.html` contains the six full overloads, `fullBFloat16`, `identityMatrix`, and
  `eye`, with exact type/value, static/scalar/empty/rectangular, dense-layout, label/gradient,
  ownership/provenance, failure-order, and alias contracts. Source review covers package-private
  `TensorConstants`, its seven entries, and the private count/import/typed-diagonal helpers that
  public generated Javadoc intentionally omits.
- `./gradlew test` — passed for the repository with 36 actionable tasks and no failing task.
- `javap -classpath modules/model/build/classes/java/main -p -c -s` inspection of
  `TensorFactory` and `TensorConstants` confirmed the exact overload descriptors and visibility,
  no additional field or nested type, exactly fifteen package-private constant entries, one exact
  typed source allocation plus `Arrays.fill` and one matching flat import for each full method,
  one default-zero exact identity carrier, row-major `i * columns + i` diagonal writes, and one
  matching flat import. `eye` bytecode contains only unchanged argument loads, one
  `identityMatrix` call, and `areturn`.
- Source, diff, import, and bytecode review confirmed existing allocator, allocation, flat-import,
  scalar/zero/one/like, provenance, label, and storage behavior remained unchanged. No new
  Operation, graph, compiler, planning, runtime, prepare, backend, device, service, cache,
  registry, or stateful dependency was introduced. Public initializer results remain leaves with
  empty provenance and independent metadata/storage/identity.
- The complete API example was compiled with `javac -cp
  modules/model/build/classes/java/main -d /private/tmp/synaptik-full-identity-example
  /private/tmp/FullIdentityExample.java` and run with
  the model classes. It printed `[-7, -7, -7, -7, -7, -7]`, the rectangular row-major identity
  carrier `[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]`, `Shape[2, 4]`,
  `DENSE_CONTIGUOUS`, and `true`, exactly as documented.
- A targeted local Markdown validator resolved 177 file targets and heading anchors across the
  five changed documentation/planning files with zero errors. Its first temporary invocation
  stopped before checking content because of Ruby interpolation syntax; the corrected compatible
  rerun is the passing result. Fence counts are balanced, changed files have no trailing
  whitespace, and `git diff --check` passed.
- Glossary review updated only the existing implementation-status, Tensor, and Tensor factory
  wording. No new reusable domain term was introduced: `full`, `identityMatrix`, and `eye` are
  concrete factory operations whose behavior belongs in the Tensor API, not new architecture or
  lifecycle concepts requiring separate glossary entries.
- Existing `DataType` and `BFloat16Bits` Javadocs remain accurate because exact type metadata and
  scalar BFLOAT16 conversion did not change. `Shape`, `LayoutDescriptor`, and `TensorDescriptor`
  remain accurate because the new paths consume their existing static-count, dense-geometry, and
  gradient-eligibility contracts without changing them. `Tensor` and `TensorProvenance` remain
  accurate because successful results are ordinary independent leaves with empty provenance and
  unchanged sole storage mutation. `HostTensorStorage` and `MemorySegmentStorage` remain accurate
  because storage sizing, borrowing, liveness, raw access, and ownership did not change. Existing
  allocation, flat-import, and constant Javadocs remain accurate; only the affected
  `TensorFactory` and `TensorConstants` contracts required finalization.
- `ARCHITECTURE.md`, focused architecture explanations, ADRs, and architecture tests require no
  update because the change remains model-owned eager leaf creation and changes no architecture
  rule, package/module boundary, dependency direction, lifecycle stage, or backend/runtime
  responsibility. `capabilities.md` already selected full and identity creation and required no
  implementation-status expansion. Compile API and Training API remain unchanged because no
  compiler capture/constant, Operation, autograd, gradient, trainable, optimizer, or session
  behavior was added. Backend-conformance and integration tests remain unchanged because no
  backend behavior or end-to-end execution path changed.
- Root and model Gradle review confirmed the existing Java 26 toolchain/release and no model
  override, preview/incubator feature, dependency, or build change. Final scope review found
  exactly the ten authorized repository paths: two production/Javadoc files, two existing exact-
  surface tests, one new focused test, Tensor API, glossary, this task, model master plan, and
  roadmap. No task-0014 specification exists.
- Final status synchronization marks 0013A Complete here, in the model master plan, and in the
  roadmap. The post-foundation checkpoint is the next planning action; task 0014 remains Draft and
  no implementation frontier is selected until that checkpoint records a decision.

## Implementation notes

- Added exactly six type-safe public/package-private full-value entries, including explicit
  binary32-to-BFLOAT16 conversion, and kept one exact source-carrier fill plus one matching flat
  import for every data type.
- Added canonical all-data-type rectangular `identityMatrix` using one default-zero carrier and
  typed main-diagonal writes, plus public-only `eye` as one unchanged-argument delegation.
- Added the focused 8-test full/identity suite and changed the two existing focused suites only for
  their exact public/helper surface expectations.
- Finalized affected public, package-private, and private-helper Javadocs; documented current API
  behavior and a compiled concrete example; updated existing glossary terms; and synchronized the
  task, model master plan, and roadmap after validation.

## Completion summary

- Completed changes: Implemented and documented type-safe full-value tensors and dense rectangular
  identity matrices for all six current data types, with `eye` exactly aliasing canonical
  `identityMatrix` behavior.
- Files changed or created: The exact ten paths authorized under Affected files and Maximum scope;
  no other repository path changed.
- Tests and validation: Focused suites passed 8/8, 9/9, and 8/8; all 259 model tests across 34
  suites, model Javadoc, full repository tests, compiled example, bytecode/API/helper/import/state
  checks, generated-documentation review, 177 local link/anchor checks, fence/whitespace checks,
  exact-scope/status checks, and `git diff --check` passed.
- Documentation-agent review: Clean context
  `/root/implement_model_0013a/review_model_0013a_docs` completed the independent pass using
  General, API/Javadoc, Planning, and Example-format profiles.
- Documentation impact: Tensor API and existing glossary/status language now describe current
  full-value and identity creation. Compile API, Training API, capability baseline, architecture
  documentation/tests, backend-conformance, integration documentation/tests, and build structure
  require no change for the reasons recorded above.
- Javadoc review: `TensorFactory`, `TensorConstants`, all seven new helper entries, and all new
  private population helpers are final. Adjacent data type, BFLOAT16, shape, layout, descriptor,
  Tensor, provenance, storage, allocation, flat-import, and existing constant contracts remain
  accurate without edits.
- Glossary impact: Existing implementation-status, Tensor, and Tensor factory entries were
  updated; no new reusable project term was introduced.
- Unresolved issues: None.
- Follow-up required: None for task 0013A. The model foundation checkpoint is the next planning
  action, while task 0014 remains Draft without a detailed specification.

Status: Complete
