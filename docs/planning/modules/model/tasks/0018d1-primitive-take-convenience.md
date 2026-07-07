# Task 0018D1: Primitive Take Convenience

## Status

Complete

## Goal

Add the legacy-compatible `take(int axis, int[] indices)` convenience without duplicating
axis-gather semantics.

The method snapshots one non-empty caller array, creates one independent dense rank-one `INT32`
index Tensor through the existing flat-import factory, and delegates to the completed tensor-index
`take` path. The returned expression therefore has exact `GATHER_AXIS` semantics and ordered
provenance `[data, generatedIndices]`.

## Scope

- Add exactly one public instance overload to `Tensor`: `Tensor take(int axis, int[] indices)`.
- Add one field-free package-private final `TensorPrimitiveTakeExpressions` helper.
- Give the helper exactly one package-private entry and one private index-Tensor creation method.
- Null-check data and the primitive array, reject an empty array, and clone the array exactly once.
- Create a fully static rank-one Shape whose extent is the copied array length.
- Create one resolved dense-contiguous `INT32`, non-differentiable descriptor.
- Import the copied values through existing `TensorFactory.fromFlatArray` with no label.
- Delegate exactly once to `TensorAxisGatherExpressions.take(data, axis, generatedIndices)`.
- Preserve every primitive value unchanged, including negative and extreme integers; do not
  inspect index bounds.
- Document the deliberate allocation/identifier ordering and lack of rollback.
- Update `TensorTest` only for the one-method public API expansion and add one focused test.
- Finalize Javadocs, Tensor API, Compile API, glossary, task evidence, master plan, and roadmap
  through the mandatory independent documentation pass.

## Out of scope

- `take(int, long[])`, generic arrays, collections, scalar convenience, nullable/default indices,
  or another overload
- accepting an empty primitive array; the legacy convenience requires at least one index
- changing tensor-index `take`, gatherAxis, the nine-method axis-gather helper, semantic kinds,
  attributes, Shape rules, provenance order, or validation messages from task 0018D
- interpreting, normalizing, clamping, validating, sorting, deduplicating, or converting primitive
  index values
- retaining the caller array, exposing the generated index Tensor directly, adding a label, or
  creating an INT64 index Tensor
- native/off-heap allocation, Arena ownership, pooling, lifecycle APIs, storage types, or direct
  writes outside existing TensorFactory import
- gradients, scatter backward, graph capture, compiler, planning, prepare, runtime, backend,
  engine, trace, ONNX implementation, training, or execution behavior
- another production helper/type, dependency, Gradle/build option, architecture change, another
  module, or task-0018E specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Architecture overview](../../../../architecture/overview.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Runtime / Prepare / Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Documentation profile index](../../../../developer-guide/documentation/README.md)
- [General documentation style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)
- [Planning style](../../../../developer-guide/documentation/planning-style.md)
- [Example format](../../../../developer-guide/documentation/example-format.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)
- [Model master plan](../master-plan.md)
- [Task 0001](0001-data-type-model.md)
- [Task 0002](0002-shape-and-dimension-model.md)
- [Task 0003](0003-layout-descriptor-model.md)
- [Task 0007](0007-tensor-descriptor-model.md)
- [Task 0010](0010-host-storage-abstraction.md)
- [Task 0012](0012-tensor-factory.md)
- [Task 0012A](0012a-host-storage-allocation.md)
- [Task 0012B](0012b-flat-typed-tensor-import.md)
- [Task 0018C](0018c-axis-gather-semantics.md)
- [Task 0018D](0018d-axis-gather-tensor-expressions.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only legacy Tensor API includes:

```java
Tensor take(int axis, int[] indices)
```

Legacy rejects null or empty arrays, clones the caller values, constructs a rank-one INT32 index
Tensor, and delegates to tensor-index take. The new implementation preserves that observable
surface and uses the completed current factory/storage contracts rather than a legacy constructor.

Task 0018D already implements `take(int, Tensor)` as an exact alias for `GATHER_AXIS`. This task
therefore owns only primitive request adaptation and eager copied index-Tensor creation. It adds no
new operation identity or Shape rule.

## Architecture constraints

- The convenience remains in `modules/model` because it adapts public Tensor construction and
  expression metadata without runtime/backend state.
- Primitive source ownership and copying use the existing eager TensorFactory path; no new storage
  abstraction or lifecycle policy is introduced.
- The generated index Tensor is a normal eager model Tensor leaf with INT32 type, rank-one dense
  descriptor, absent label/provenance, and existing JVM-managed heap storage.
- The final result is built only by the completed tensor-index take helper, which owns axis
  normalization, result Shape, semantic attributes, and ordered provenance.
- Invalid index values cannot be detected without execution/value access and remain untouched.
- No dependency, package ownership, or module boundary changes are authorized.

## Package impact

No package is added or moved.

- `io.github.pho001.synaptik.model.tensor.Tensor` — receives the primitive-array overload and
  current wording for both take overloads.
- `io.github.pho001.synaptik.model.tensor.TensorPrimitiveTakeExpressions` — owns source validation,
  snapshot, eager index-Tensor import, and delegation.
- `TensorPrimitiveTakeExpressionTest` — mirrors `model.tensor` for focused validation.
- `TensorTest` — changes only its exact public API inventory/reflection assertions.

The completed `TensorAxisGatherExpressions` helper remains unchanged.

## Required contract

### Public Tensor overload

Add exactly:

```java
public Tensor take(int axis, int[] indices) {
    return TensorPrimitiveTakeExpressions.take(this, axis, indices);
}
```

The method contains one return statement and one helper call. It is public, non-static,
non-synchronized, performs no direct validation or cloning, and adds no other overload.

Update the existing tensor-index take Javadoc in the same `Tensor.java` file so it describes the
primitive overload as current rather than deferred. Do not change the existing method declaration
or behavior.

### Helper shape

Create one package-private final, field-free class with one private zero-argument constructor and
exactly these two static methods:

```java
static Tensor take(Tensor data, int axis, int[] indices)
private static Tensor createIndices(int[] snapshot)
```

Add no field, nested type, alternate constructor, overload, cache, mutable state, or extra method.

### Validation, snapshot, allocation, and delegation order

`take` performs exactly:

1. `Objects.requireNonNull(data, "data")`;
2. `Objects.requireNonNull(indices, "indices")`;
3. if `indices.length == 0`, throw `IllegalArgumentException` with exact message
   `take indices must not be empty`;
4. clone `indices` exactly once into `snapshot`;
5. call `createIndices(snapshot)` exactly once;
6. call `TensorAxisGatherExpressions.take(data, axis, generatedIndices)` exactly once and return
   its result.

The helper neither normalizes nor prevalidates `axis`. Consequently null/empty failures occur
before allocation and consume no ID. Index-Tensor creation occurs before tensor-index take checks
the axis. An invalid axis therefore leaves one generated eager index Tensor allocated and consumes
its ID; no result ID is consumed. If final result ID allocation fails, the generated index Tensor
already exists. No ID, storage, or allocation is rolled back.

### Generated index Tensor

`createIndices` constructs exactly:

```java
Shape shape = Shape.of(snapshot.length);
TensorDescriptor descriptor = new TensorDescriptor(
        DataType.INT32,
        shape,
        Optional.of(LayoutDescriptor.contiguous(shape)),
        false);
return TensorFactory.fromFlatArray(descriptor, Optional.empty(), snapshot);
```

The generated Tensor:

- has exact `INT32`, false gradient eligibility, Shape `[indices.length]`, and resolved canonical
  dense layout;
- has absent label and provenance;
- owns independent JVM-managed heap contents created by the existing factory;
- retains neither the caller array nor the helper snapshot;
- contains exact copied primitive values without normalization or conversion.

The result Tensor's provenance input zero is the exact receiver/data Tensor. Provenance input one
is the exact generated index Tensor. Result kind, normalized attributes, Shape, unresolved layout,
data type, eligibility, label/storage absence, and fresh identity remain exactly those of completed
tensor-index take.

### Source ownership example

For data Shape `[2, 3, 4]`, `take(1, new int[] {2, 0})` creates an internal dense INT32 index Tensor
of Shape `[2]` containing `[2, 0]`, then produces GATHER_AXIS result Shape `[2, 2, 4]`. Mutating the
caller array afterward cannot affect the internal index Tensor or result provenance.

Negative and extreme integer entries are copied unchanged. This task does not claim they are valid
for a particular data extent during execution.

## Affected files

Expected implementation change:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorPrimitiveTakeExpressions.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorPrimitiveTakeExpressionTest.java`
- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- this task file
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum scope

This task may create or modify at most the ten paths listed above.

If implementation requires changing TensorFactory, TensorAxisGatherExpressions, semantic or
foundational contracts, another test/helper/type, capability baseline, architecture, dependency,
build, another module, or more than ten paths, stop and propose a follow-up task.

## Javadoc requirements

- Document the new overload, helper class, constructor, and both helper methods completely.
- Explain non-empty validation, snapshot ownership, exact INT32 descriptor, factory copying, and
  delegation.
- Explain index-Tensor/result ID and allocation ordering for success, null/empty failure, invalid
  axis, and final identifier exhaustion.
- Explain exact generated index Tensor metadata and final provenance order.
- Include the `[2, 3, 4]`, axis `1`, indices `[2, 0]`, result `[2, 2, 4]` example.
- State that values are copied unchanged but never bounds-checked.
- Correct stale deferred wording for primitive take only within the affected `Tensor.java` file.
- Do not promise gradients, compiler capture, backend support, execution, or cleanup beyond current
  factory/storage contracts.

## Acceptance criteria

- Tensor adds exactly one `take(int, int[]): Tensor` overload and no long/generic overload.
- The public method delegates once and contains no validation, cloning, or construction.
- The helper is package-private, final, field-free, has one private constructor, and exactly two
  specified methods.
- Input-null, indices-null, and empty-array failures follow exact order/types/messages before ID or
  storage allocation.
- The caller array is cloned once; later mutation cannot affect generated index contents.
- Generated indices have exact INT32/rank-one/dense/non-differentiable/unlabeled/provenance-free
  metadata and copied values.
- Delegation uses the unchanged tensor-index take path and produces exact GATHER_AXIS semantics,
  unresolved result layout, retained data metadata, and `[data, generatedIndices]` provenance.
- Invalid axis and exhaustion side effects follow the documented order without rollback.
- No index values are validated or converted; no existing helper/factory/semantic behavior changes.
- Tensor API, Compile API, glossary, task evidence, master plan, and roadmap are independently
  reviewed and synchronized; Training API, capabilities, architecture, and related contracts
  receive reasoned no-change conclusions.
- All validation passes and the final diff contains exactly the ten permitted paths.

## Tests / validation

Run:

```bash
./gradlew :modules:model:test --tests \
  io.github.pho001.synaptik.model.tensor.TensorPrimitiveTakeExpressionTest
./gradlew :modules:model:test --tests \
  io.github.pho001.synaptik.model.tensor.TensorTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

Focused tests must verify exact public/helper API shape, validation order/messages, source
snapshot, exact generated index descriptor/storage/contents/absence metadata, result Shape/kind/
attrs/provenance/layout/type/eligibility, negative/extreme value preservation, caller mutation,
fresh identities, invalid-axis ID side effects where safely observable, and lack of cross-layer
imports or behavior.

Manual validation must inspect `javap -p -c -s`, one-call delegation, reflection, imports, source,
generated Javadoc, executable documentation example, Markdown links/anchors/fences/whitespace,
exact ten-path scope, synchronized status, and absence of a task-0018E specification.

## Dependencies

- Task 0012B provides exact copied dense INT32 flat import and its allocation/ID semantics.
- Task 0018D provides tensor-index take, GATHER_AXIS Shape construction, validation, and provenance.

## Follow-up tasks

- 0018E: gather-ND semantic identity and batch-dimension attributes.

Do not create its specification during this task.

## Architecture impact

Expected impact: None.

This is a model-owned public convenience over existing model factory and expression contracts. If
implementation requires a new storage/resource policy or cross-module dependency, stop and report.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0001/0002/0003/0007/0010/0012/0012A/0012B/0018C/0018D/
0018D1, Tensor API, Compile API, Training API, glossary, current DataType/Shape/LayoutDescriptor/
TensorDescriptor/Tensor/TensorFactory/TensorAxisGatherExpressions and focused factory/gather tests,
and Java 26 Gradle configuration.

Implement task 0018D1 exactly. Modify Tensor.java and add package-private final
TensorPrimitiveTakeExpressions.java. Update TensorTest only for the exact one-overload API expansion
and add TensorPrimitiveTakeExpressionTest. Add exactly take(int axis, int[] indices).

The field-free helper has exactly two methods. Null-check data/indices, reject empty, clone once,
create one dense rank-one INT32 non-differentiable index Tensor through existing fromFlatArray,
then delegate exactly once to TensorAxisGatherExpressions.take. Preserve every int unchanged and
document exact source/allocation/ID side effects. The final result uses existing GATHER_AXIS
semantics and [data, generatedIndices] provenance.

Do not add long[]/generic overloads, modify TensorFactory or axis-gather helper/semantics, inspect
bounds, add gradients/compiler/runtime/backend behavior, dependencies, build/architecture changes,
or later specs. Stop beyond ten paths or on uncertainty.

Run all specified validation, then hand actual diff/evidence to a separate clean-context docs
agent in the same change. It must inspect source/tests/generated Javadoc, finalize permitted
Javadocs/Tensor API/Compile API/glossary/planning, record Training API/capability/architecture and
related-contract no-change conclusions, and rerun validation.

Update task 0018D1, model master plan, and roadmap only for planning status/evidence. Do not mark
Complete until both passes succeed. Leave 0018E Draft without a specification. Do not commit/push.
```

## Local decisions

- The public overload remains a six-bytecode one-call delegation. One package-private final helper
  owns primitive validation, snapshot, eager index-Tensor construction, and delegation without
  adding state or another public surface.
- The caller array is cloned once before descriptor or Tensor creation. Existing flat import then
  copies that private snapshot into independent storage; neither source is retained.
- The generated index Tensor is always exact INT32, rank one, canonical dense-contiguous,
  non-differentiable, unlabeled, and provenance-free. Primitive values are copied bit-for-bit as
  signed integers and are not interpreted as coordinates during expression construction.
- Axis normalization remains exclusively in the completed tensor-index take path. This preserves
  one GATHER_AXIS construction path and deliberately places generated index allocation and ID
  consumption before invalid-axis failure.
- Existing factory and derived-result ID allocation are not transactional. Invalid axis and final
  result-ID exhaustion retain the generated index Tensor, storage, and identifier without
  rollback.

## Known limitations

- The overload accepts only a non-empty `int[]`; it has no `long[]`, collection, scalar, generic,
  or empty-sequence form.
- Negative and extreme values are preserved but not proved valid for the selected data extent.
- The result defines no numerical gather, gradient or repeated-index rule, compiler capture or
  canonicalization, materialization, backend/ONNX lowering, or execution behavior.
- ID or storage side effects after late failure are permanent within the existing JVM-scoped
  factory contract; no cleanup or rollback API is introduced.

## Validation evidence

- Clean implementation context `/root/implement_model_0018d1` added the exact public overload,
  helper, focused tests, and initial planning updates. Independent documentation context
  `/root/implement_model_0018d1/review_model_0018d1_docs` inspected the actual shared-tree diff,
  source, tests, generated Javadoc, bytecode, APIs, glossary, planning state, and build
  configuration before finalizing documentation in the same overall change.
- The documentation pass applied General plus API/Javadoc style to production Javadocs, Tensor
  API, Compile API, and glossary; Planning style to this task, the model master plan, and roadmap;
  and Example format to the executable primitive-take example.
- Reviewed architecture and process material included `AGENTS.md`, `ARCHITECTURE.md`, current
  architecture documentation, overview, lifecycle, module boundaries, dependency rules, and the
  runtime/prepare/backend boundary; documentation rules and General/API-Javadoc/Planning/Example
  profiles; planning guide and roadmap; model capabilities/master plan; tasks 0001, 0002, 0003,
  0007, 0010, 0012, 0012A, 0012B, 0018C, 0018D, and 0018D1; Tensor, Compile, and Training API
  references; glossary; Java 26 root/model Gradle configuration; final implementation/tests; and
  generated model Javadoc.
- Related source and contract review covered DataType, Shape, LayoutDescriptor, TensorDescriptor,
  Tensor, TensorFactory, TensorAxisGatherExpressions, the new helper, focused Tensor/factory/
  gather tests, and generated `Tensor.html`. The submitted Tensor and helper Javadocs already
  accurately documented source ownership, exact descriptor and factory copy, provenance,
  allocation/ID ordering, failure side effects, no rollback, and cross-layer exclusions, so the
  independent pass retained their content unchanged.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorPrimitiveTakeExpressionTest` — `BUILD SUCCESSFUL`;
  XML reports 8 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorTest` — `BUILD SUCCESSFUL`; XML reports 14 tests
  with zero failures, errors, or skips.
- `./gradlew :modules:model:test` — `BUILD SUCCESSFUL`; XML aggregation reports 675 tests across
  79 suites with zero failures, errors, or skips.
- `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL`. Generated `Tensor.html` contains the
  exact primitive overload, independent dense INT32 index-Tensor contract, `[2, 3, 4]`/axis
  `1`/`[2, 0]`/`[2, 2, 4]` example, ordered provenance, failure messages, allocation/ID timing,
  no-rollback rule, and explicit no-gradient/compiler/backend/execution boundary. The
  package-private helper Javadocs were reviewed in source.
- `./gradlew test` — `BUILD SUCCESSFUL`; all 36 root lifecycle tasks completed or were up-to-date
  with no failing task.
- `javap -classpath modules/model/build/classes/java/main -p -c -s` plus focused reflection tests
  confirmed exactly 102 declared public Tensor methods, the one new public six-bytecode one-call
  delegation, and one final package-private field-free helper with one private constructor and
  exactly two methods. Bytecode confirms data then indices null checks, exact empty rejection, one
  primitive-array clone, one `createIndices` call, one axis-gather take call, exact rank-one Shape/
  INT32/contiguous/false-gradient descriptor construction, and one flat-import call.
- Source, imports, tests, and bytecode confirm no cross-layer imports or behavior; preservation of
  negative and extreme values; exact generated descriptor/storage/absence metadata; independent
  repeated storage and identities; unresolved storage-free final results; exact kind/attrs/
  `[data, generatedIndices]` provenance; invalid-axis consumption of only the generated ID; and
  final-ID exhaustion after the generated Tensor claims the last identity.
- The documented `PrimitiveTakeExample` compiled with
  `javac -cp modules/model/build/classes/java/main -d
  /tmp/synaptik-primitive-take-doc-example /tmp/PrimitiveTakeExample.java` and ran with model
  classes. It printed `INT32`, Shape `[2]`, `DENSE_CONTIGUOUS`, stored values `[2, 0]`, result Shape
  `[2, 2, 4]`, `GATHER_AXIS`, and four expected `true` ownership/provenance/metadata facts after
  the caller array was mutated.
- The targeted Markdown validator resolved 380 local links, including 91 heading anchors, across
  the six changed documentation/planning files. All Markdown fences balance, all ten paths end
  with a newline, targeted trailing-whitespace scans found no matches, and `git diff --check`
  passes.
- Final changed-path inventory contains exactly the ten authorized paths: Tensor, the new helper,
  TensorTest, the new focused test, Tensor API, Compile API, glossary, this task, model master
  plan, and roadmap. Task/master/roadmap status is synchronized as Complete. Task 0018E remains
  Draft and no task-0018E specification exists.
- Training API remains accurate unchanged because the task adds no gradient object or rule,
  autograd, parameter, optimizer, publication, session, or training execution behavior. The model
  capability baseline remains accurate unchanged because it already lists the legacy take
  capability, axis-gather family, exact integral index representation, and separate support
  layers; this task realizes the already-recorded primitive convenience rather than changing the
  baseline.
- DataType, Shape, LayoutDescriptor, TensorDescriptor, TensorFactory, and
  TensorAxisGatherExpressions remain accurate unchanged because the task composes their exact
  existing INT32, rank-one dense layout, copied flat import, derived identity, axis normalization,
  Shape construction, and provenance contracts without modifying them. Related scalar-select,
  axis-gather semantic, and other operation-family contracts also remain accurate unchanged.
- `ARCHITECTURE.md`, focused architecture explanations, ADRs, architecture tests,
  backend-conformance tests, integration tests, Gradle/dependencies, and other modules remain
  accurate unchanged because the convenience stays within model ownership and changes no module
  boundary, dependency rule, backend behavior, executable end-to-end behavior, or build
  requirement.

## Implementation notes

- Added exactly `Tensor.take(int, int[])` and the exact field-free two-method primitive adapter.
- Added focused eight-test coverage and expanded the exact Tensor public-method inventory by one.
- Finalized the Tensor/helper Javadocs, Tensor and Compile API references, glossary Axis gather
  entry, executable primitive example, and synchronized planning status without changing any
  completed factory, tensor-index take, semantic, architecture, or cross-layer contract.

## Completion summary

- Completed changes: Implemented and documented primitive-array take adaptation through one
  copied dense INT32 index Tensor and the existing GATHER_AXIS tensor-index path, including exact
  failure order and permanent late-failure side effects.
- Files changed or created: Exactly the ten authorized production, test, API, glossary, task,
  master-plan, and roadmap paths.
- Tests and validation: Focused 8-test and 14-test suites, all 675 model tests across 79 suites,
  model Javadoc, root tests, javap/reflection/import/source/generated-page review, executable Java
  26 example, 380-link/91-anchor checks, fence/whitespace/newline checks, exact scope/status and
  no-0018E-spec checks, and `git diff --check` passed.
- Documentation-agent review: Clean context
  `/root/implement_model_0018d1/review_model_0018d1_docs` completed the required independent pass
  with General, API/Javadoc, Planning, and Example profiles.
- Documentation impact: Tensor API, Compile API, and glossary now describe the current primitive
  adapter, generated index Tensor, ownership, allocation/ID ordering, and existing GATHER_AXIS
  boundary without adding compiler or execution claims.
- Javadoc review: Tensor's affected take Javadocs and the helper type, constructor, and two methods
  are final. Related foundational and axis-gather contracts remain accurate for the reasons
  recorded above.
- Glossary impact: The existing Axis gather entry now distinguishes tensor-index aliasing from
  primitive input adaptation and records copying, generated metadata, value-bounds, and late-
  failure ownership boundaries; no gratuitous new term was added.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0018D1. Task 0018E remains Draft without a detailed
  specification.

Status: Complete
