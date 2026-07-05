# Task 0017B: Contiguous Tensor Expression

## Status

Complete

## Goal

Add the public `Tensor.contiguous()` expression that records the completed
`ContiguousKind.CONTIGUOUS` semantic request without reading or copying values.

The expression preserves the input's exact logical shape, data type, and gradient eligibility.
For a fully static shape it publishes newly resolved canonical dense row-major zero-offset layout
geometry. For a shape containing a dynamic dimension it leaves layout unresolved because numeric
strides and referenced span cannot yet be calculated. Every valid call creates a fresh,
storage-free derived Tensor with exact one-input provenance.

## Scope

- Add exactly one public no-argument `Tensor.contiguous()` instance method.
- Add one package-private final stateless `TensorContiguousExpressions` construction boundary.
- Give the helper exactly one package-private static `apply(Tensor)` method and one private
  zero-argument constructor.
- Null-check the helper input before reading metadata.
- Retain the exact input `Shape`, `DataType`, and `requiresGrad` value.
- For a fully static shape, create exactly one new `LayoutDescriptor.contiguous(shape)` and store
  it as the resolved result layout.
- For a dynamic shape, store `Optional.empty()` and do not invent numeric strides, offset, or span.
- Construct exactly `Operation(ContiguousKind.CONTIGUOUS, NoOperationAttrs.INSTANCE)`.
- Record exact one-input provenance `[input]` and delegate exactly once to
  `TensorFactory.createDerived` with no label or storage.
- Return a fresh expression even when the input already has canonical dense-contiguous geometry or
  is itself a contiguous expression.
- Accept all six current data types, scalar shapes, zero extents, ordinary static shapes, and
  dynamic shapes.
- Add one focused same-package expression test and update `TensorTest` only for the deliberate
  one-method public API expansion.
- Finalize affected Javadocs, Tensor API, Compile API, glossary, task evidence, model master plan,
  and roadmap through the required independent documentation pass during implementation.

## Out of scope

- eager value access, memory copy, materialization, allocation, storage attachment, alias creation,
  mutation, host-storage replacement, or ownership/lifetime changes
- inspecting the input descriptor's layout optional, layout kind, strides, offset, span, view flag,
  contiguity, label, provenance, storage association, storage liveness, or values
- returning the input, reusing its `LayoutDescriptor`, preserving its view flag or offset,
  canonicalizing repeated requests, eliminating an already-contiguous request, or constant folding
- public overloads, static facades, optional copy flags, layout-order arguments, memory-format
  options, destination storage, device arguments, factories, builders, or another public type
- reshape, expand, permute, transpose, expand-dimensions, squeeze, slice, pad, tile, concat, stack,
  unstack, unfold, fold, select, gather, or scatter behavior
- data-type conversion or promotion, Shape mutation, symbolic binding, graph-wide inference,
  materialization scoring, backend ownership, kernel or route selection, or executable behavior
- gradient formulas, backward operation kinds, saved values, autograd expansion, optimizer, or
  training execution
- compiler capture, graph traversal, canonicalization implementation, planning output, prepare,
  runtime residency, backend storage, engine composition, tracing, ONNX mapping, or conformance
- changes to `ContiguousKind`, `NoOperationAttrs`, `Operation`, `TensorDescriptor`,
  `LayoutDescriptor`, `LayoutKind`, `TensorFactory`, `TensorProvenance`, Shape, DataType, storage,
  graph contracts, existing expression helpers, or their tests
- dependencies, Gradle, Java version, AGENTS, ARCHITECTURE, focused architecture documentation,
  architecture tests, capabilities, another module, or unrelated documentation
- a detailed task-0017C specification or implementation of a later 0017 task

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Architecture overview](../../../../architecture/overview.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Runtime / Prepare / Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)
- [Model master plan](../master-plan.md)
- [Task 0002](0002-shape-and-dimension-model.md)
- [Task 0003](0003-layout-descriptor-model.md)
- [Task 0007](0007-tensor-descriptor-model.md)
- [Task 0011](0011-public-tensor-skeleton.md)
- [Task 0012](0012-tensor-factory.md)
- [Task 0013](0013-tensor-provenance-skeleton.md)
- [Task 0017A](0017a-contiguous-semantic-kind.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only `legacy/pre-rewrite` branch exposes:

```java
Tensor contiguous()
TensorOps.contiguous(Tensor input)
```

Legacy public behavior preserves logical values, shape, and data type while producing canonical
row-major output. Execution evidence covers every selected data type, including BOOL; already
dense inputs; permuted, expanded, sliced, offset, and other non-dense inputs; chained expressions;
and CPU, Metal, and CUDA paths. Legacy implementations may materialize a new buffer or reuse an
already suitable representation.

This task preserves only the public model expression and deterministic descriptor facts. It does
not copy the legacy immediate storage/view construction, gradient callback, graph builder,
operation traits, compiler layout flags, materialization planners, physical view types, kernels,
lowering, fallback, or runtime propagation. Those mechanisms either violate the new boundaries or
belong to later compiler, planning, prepare, runtime, training, and backend tasks.

## Architecture constraints

- `Tensor` remains public mutable API state and is not an intermediate-representation node.
- The result is a fresh public Tensor carrying immutable expression provenance, not a compiled
  node, runtime tensor, physical buffer, prepared unit, or executable copy.
- `ContiguousKind.CONTIGUOUS` is the exact operation identity and
  `NoOperationAttrs.INSTANCE` is its complete attributes value.
- The result retains the exact immutable input `Shape` reference, exact `DataType`, and unchanged
  `requiresGrad` value. This task defines no gradient formula or guarantee that one exists.
- A fully static Shape supplies enough information to resolve canonical dense row-major strides,
  zero logical storage offset, non-view metadata, and referenced span through
  `LayoutDescriptor.contiguous(shape)`.
- A dynamic Shape cannot carry numeric `LayoutDescriptor` geometry. The result must use
  `Optional.empty()` without binding symbols or guessing concrete dimensions.
- Resolved result layout expresses the requested logical geometry. It does not allocate storage,
  prove that values have been copied, require a distinct physical buffer, or select an executable
  materialization route.
- Input layout and storage are deliberately irrelevant to expression construction. Compiler may
  later eliminate a redundant request; planning may derive a logical materialization requirement;
  backend prepare may choose alias or copy lowering; runtime executes the prepared schedule.
- Every valid request remains an explicit fresh expression at the model boundary, including an
  already-dense input and nested contiguous requests.
- The helper remains in `model.tensor` because it coordinates the public Tensor, descriptor,
  operation, provenance, and factory contracts. The semantic kind remains in `model.operation.layout`.
- No project dependency or module boundary changes. Stop if implementation requires storage
  access, mutation, graph state, another type, cross-module behavior, or architecture changes.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.tensor` — owns `Tensor`, its package-private expression helper,
  `TensorDescriptor`, `TensorProvenance`, and `TensorFactory.createDerived`.
- `io.github.pho001.synaptik.model.operation.layout` — supplies the completed
  `ContiguousKind.CONTIGUOUS` identity.
- `io.github.pho001.synaptik.model.operation` — supplies `Operation` and
  `NoOperationAttrs.INSTANCE`.
- `io.github.pho001.synaptik.model.layout` — supplies canonical resolved layout construction for
  fully static shapes.
- `io.github.pho001.synaptik.model.shape` — supplies the immutable Shape and its
  `isFullyStatic()` decision.

No package is added or moved.

Type placement:

- `io.github.pho001.synaptik.model.tensor.Tensor` — gains the one public fluent expression entry.
- `io.github.pho001.synaptik.model.tensor.TensorContiguousExpressions` — package-private stateless
  coordinator colocated with Tensor and its existing derived-construction seam.
- `TensorContiguousExpressionTest` — same-package focused test that can inspect the package-private
  helper without widening production visibility.
- `TensorTest` — changes only its exact public-method inventory from 76 to 77 and includes the
  no-argument `contiguous` signature.

## Required contract

### Public Tensor method

Add exactly:

```java
public Tensor contiguous() {
    return TensorContiguousExpressions.apply(this);
}
```

The method is public, final by virtue of `Tensor` being final, non-static, non-synchronized,
parameterless, and returns `Tensor`. Its body contains exactly one delegation to the helper and no
inline validation, descriptor construction, storage access, or optimization.

### Helper shape

Create one package-private final class with exactly:

```java
final class TensorContiguousExpressions {
    private TensorContiguousExpressions() {
    }

    static Tensor apply(Tensor input) {
        // Exact construction contract below.
    }
}
```

The helper has no fields, nested types, public or protected members, overloads, factory instance,
cache, registry, service, layout policy, or additional methods. The constructor is private and
zero-argument. `apply` is package-private static and is the sole declared method.

### Validation and construction order

`apply` performs exactly this observable order:

1. Null-check `input` with `Objects.requireNonNull(input, "input")`.
2. Read and retain the exact `TensorDescriptor` reference once.
3. Read and retain that descriptor's exact `Shape` reference.
4. Call `shape.isFullyStatic()` once.
5. For a static shape, call `LayoutDescriptor.contiguous(shape)` exactly once and use
   `Optional.of` on that exact new descriptor. For a dynamic shape, use `Optional.empty()` and do
   not call layout construction.
6. Create one `TensorDescriptor` from the input descriptor's exact DataType, exact Shape reference,
   resolved-or-unresolved result layout, and unchanged `requiresGrad` value.
7. Create one `Operation` from exact `ContiguousKind.CONTIGUOUS` and
   `NoOperationAttrs.INSTANCE`.
8. Create one `TensorProvenance` with that exact operation and exact ordered input `[input]`.
9. Call `TensorFactory.createDerived(descriptor, Optional.empty(), provenance)` exactly once and
   return its exact result.

No input layout, label, provenance, storage, liveness, or tensor value is read. The helper does not
derive geometry itself; on the static branch, `LayoutDescriptor.contiguous(shape)` necessarily
reads the immutable Shape dimensions and performs its existing checked stride/span calculations.
A static layout arithmetic failure propagates before Tensor-ID allocation. A dynamic result does
not inspect symbolic sizes or attempt partial layout resolution.

### Result contract

Every successful call returns a fresh Tensor with:

| Result fact | Required value |
|---|---|
| ID | fresh factory-allocated `TensorId` |
| DataType | exact input DataType |
| Shape | exact input Shape reference |
| `requiresGrad` | unchanged input value |
| static layout | new canonical `DENSE_CONTIGUOUS`, offset-zero, non-view descriptor |
| dynamic layout | unresolved `Optional.empty()` |
| label | absent |
| host storage | absent |
| operation | exact `CONTIGUOUS` plus `NoOperationAttrs.INSTANCE` |
| provenance inputs | exact immutable ordered `[input]` |

The resolved descriptor records logical result geometry, not evidence of eager storage
materialization. Repeated calls and a call on another contiguous expression remain distinct fresh
expressions. The model layer performs no redundant-request elimination.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorContiguousExpressions.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorContiguousExpressionTest.java`

Documentation/planning:

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless a task-related inconsistency requires stopping:

- `docs/api/training-api.md`
- `docs/planning/modules/model/capabilities.md`
- `ContiguousKind`, Operation foundations, DataType, Shape, LayoutDescriptor, LayoutKind,
  TensorDescriptor, TensorFactory, TensorProvenance, storage contracts, graph contracts, existing
  expression helpers, and their Javadocs/tests
- focused architecture documents, ADRs, architecture tests, backend-conformance material,
  integration tests, Gradle configuration, and every other module

## Maximum scope

At most two production files, two test files, three API/glossary files, and three planning files:
ten paths total.

Do not modify `ContiguousKind`, any other existing Java source/test, capabilities, Training API,
Gradle, AGENTS, architecture documents/tests, another module, completed task, or unrelated
documentation. Stop if another production type, public method, helper method, dependency,
cross-layer behavior, or eleventh path is needed. Do not create task 0017C.

## Javadoc requirements

- Update the `Tensor` type overview so it no longer claims every expression result has unresolved
  layout. Explain the contiguous expression's static-resolved/dynamic-unresolved exception.
- Document `Tensor.contiguous()` as an expression-construction method, not an immediate copy.
- Document unchanged logical values, exact Shape/DataType/gradient-eligibility retention, static
  canonical dense row-major zero-offset layout, and dynamic unresolved layout.
- Document the fresh ID, absent label/storage, exact `CONTIGUOUS` operation,
  `NoOperationAttrs.INSTANCE`, and one-input provenance.
- Document `ArithmeticException` for static canonical layout stride/span overflow and
  `IllegalStateException` for exhausted Tensor identity space.
- Explain that input layout/storage are not inspected, and that already-dense and repeated
  requests remain fresh until compiler canonicalization.
- Document the package-private helper and its private constructor, input nullability, exact
  construction order, static/dynamic branch, return value, side effects, ownership, and failures.
- State that resolved result geometry does not prove eager allocation, copy, distinct storage,
  runtime residency, or a backend route.
- Review `ContiguousKind`, Shape, LayoutDescriptor, LayoutKind, TensorDescriptor, TensorFactory,
  TensorProvenance, and Operation Javadocs and record why they remain accurate, or stop on an
  inconsistency.

## Acceptance criteria

- `Tensor` exposes exactly one new public method, parameterless `contiguous()`, bringing its exact
  declared-public-method count from 76 to 77.
- The public method delegates exactly once to `TensorContiguousExpressions.apply(this)` and adds no
  inline behavior.
- The helper is package-private final and stateless, with exactly one private zero-argument
  constructor and exactly one package-private static `apply(Tensor)` method.
- The helper rejects null input with exact `NullPointerException` message `input` before reading
  metadata or allocating an ID.
- All six current DataTypes are accepted. Shape, DataType, and `requiresGrad` are retained exactly.
- Every fully static Shape, including scalar and zero-extent shapes, receives a newly constructed
  canonical `DENSE_CONTIGUOUS`, offset-zero, non-view layout with correct strides and span.
- Every dynamic Shape retains the exact Shape reference and receives unresolved layout without
  numeric stride/span calculation.
- Static layout overflow propagates as `ArithmeticException` before identity allocation; no
  unresolved fallback is used.
- Result operation and provenance retain exact `ContiguousKind.CONTIGUOUS`,
  `NoOperationAttrs.INSTANCE`, and input `[input]` references.
- Results are fresh, unlabeled, and storage-free even for already-contiguous input, an input with
  attached/dead storage, repeated calls, or nested contiguous requests.
- Input descriptor, label, provenance, storage association, liveness, storage bytes, and layout
  object remain unchanged and are not used to select a code path.
- No eager copy, allocation, storage alias, data access, canonicalization, gradient rule, compiler,
  planning, prepare, runtime, backend, graph, ONNX, dependency, or build behavior is added.
- Focused/aggregate tests, model Javadoc, root tests, reflection/javap/bytecode/import/source/scope
  checks, documentation links/examples/formatting, and synchronized statuses pass.
- A separate clean-context documentation agent finalizes all affected Javadocs, Tensor API,
  Compile API, glossary, task evidence, master plan, and roadmap and records related no-change
  conclusions.
- Task 0017B becomes Complete only after both passes. Task 0017C remains Draft without a detailed
  specification.

## Tests / validation

Run before and after the documentation-focused review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorContiguousExpressionTest --tests io.github.pho001.synaptik.model.tensor.TensorTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused expression test must verify:

- exact public method and exact one-method helper reflection surface;
- all six DataTypes and valid gradient eligibility combinations;
- exact Shape reference, static canonical layout kind/strides/offset/view/span, and dynamic
  unresolved layout;
- scalar and zero-extent canonical geometry;
- exact Operation, no-attributes singleton, one-input provenance, absent label/storage, and fresh
  identity;
- input layouts covering unresolved static, dense, offset-dense, strided, and broadcast geometry
  without changing result semantics;
- input label, provenance, live or subsequently dead storage, and raw contents remain untouched;
- repeated, already-contiguous, and nested requests are never model-canonicalized;
- null-helper input and static layout overflow consume no Tensor identity; and
- identifier exhaustion propagates only after valid local metadata construction.

Manually inspect `javap -p -c -s` and reflection. Confirm the public method has one exact helper
call; the helper has one method, no state, one static/dynamic branch, one canonical-layout call on
the static branch, and one descriptor/operation/provenance/createDerived path. Inspect imports and
source for absence of host-storage access, values, copies, allocation, input-layout reads,
graph/compiler/planning/prepare/runtime/backend types, gradient rules, caches, services, and hidden
canonicalization. Validate generated Javadoc, Tensor/Compile API status, glossary terms, examples,
links, anchors, fences, whitespace, exact ten-path scope, synchronized statuses, and absence of a
task-0017C specification.

## Dependencies

- Task 0002 supplies `Shape.isFullyStatic()` and dynamic/static shape representation.
- Task 0003 supplies `LayoutDescriptor.contiguous(shape)` and canonical resolved geometry.
- Task 0007 supplies explicit resolved-or-unresolved `TensorDescriptor` layout state.
- Tasks 0011–0013 supply public Tensor metadata, centralized derived identity allocation, and
  immutable one-input provenance.
- Task 0017A supplies the exact parameterless `ContiguousKind.CONTIGUOUS` semantic identity.

## Follow-up tasks

- 0017C remains Draft for immutable reshape and expand semantic kinds/attributes.
- 0017D remains Draft for public reshape and expand expression construction.
- Compiler later owns redundant contiguous-request elimination and graph-wide layout reasoning.
- Planning later derives logical materialization requirements without selecting concrete routes.
- Backend prepare later chooses whether an owned contiguous request is an alias, copy, fused
  route, or backend-native transform; runtime executes the prepared schedule.
- Training and compiler-generated semantic tasks later own differentiation and backward forms.

Do not create a detailed follow-up specification in this task.

## Architecture impact

Expected impact: None. The architecture already assigns public Tensor expression construction,
operation semantics, Shape, and logical layout values to `modules/model`; compiler optimization to
compiler; logical materialization requirements to planning; concrete lowering and storage to
backend prepare; and prepared execution to runtime.

If implementation requires eager storage behavior, physical materialization, graph capture,
cross-module state, a dependency, or an architecture change, stop and report it.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0002/0003/0007/0011/0012/0013/0017A/0017B, Tensor API,
Compile API, Training API, glossary, current Shape/LayoutDescriptor/LayoutKind/TensorDescriptor/
Tensor/TensorFactory/TensorProvenance/Operation/NoOperationAttrs/ContiguousKind contracts and
focused tests, and Java 26 Gradle configuration.

Implement task 0017B exactly. Modify Tensor.java and add package-private final
TensorContiguousExpressions.java for production. Update TensorTest only for the exact one-method
public API expansion and add TensorContiguousExpressionTest. Add exactly public parameterless
contiguous(), delegating once to the helper.

The helper has exactly one package-private static apply(Tensor) method, no fields, and one private
constructor. Null-check input, retain its exact descriptor/Shape/DataType/requiresGrad, branch once
on Shape.isFullyStatic(), create exactly one new LayoutDescriptor.contiguous(shape) for static
Shape or unresolved layout for dynamic Shape, construct exact CONTIGUOUS/NoOperationAttrs
semantics with [input] provenance, and call createDerived once with no label/storage. Every valid
call is fresh, including already-contiguous and nested requests.

Do not inspect input layout/label/provenance/storage/liveness/values, copy or allocate storage,
return input, reuse a layout, canonicalize, add overloads/helpers/types, change existing contracts,
define gradients, capture graphs, or add compiler/planning/prepare/runtime/backend behavior,
dependencies, build/architecture changes, or later specs. Stop beyond ten paths or on
architecture uncertainty.

Run every specified focused/aggregate test, Javadoc, javap/reflection/bytecode/import/manual,
documentation/example/link/whitespace/scope/status check. Then hand the actual diff and evidence
to a separate clean-context documentation agent in the same change. It must inspect source/tests/
generated Javadoc, finalize permitted Javadocs/Tensor API/Compile API/glossary/planning, record
related-contract/capability/Training API/architecture no-change conclusions, and rerun validation.

Update task 0017B, model master plan, and roadmap only for planning status/evidence. Do not mark
0017B Complete until both passes succeed. Leave 0017C Draft without a specification. Do not commit
or push.
```

## Local decisions

- The public API is exactly `Tensor.contiguous()` with no arguments. A caller requests canonical
  row-major geometry; implementation policy is not exposed as flags or storage arguments.
- The helper is `TensorContiguousExpressions` with one method because validation and construction
  form one small cohesive path. Additional factories or layout-policy abstractions have no current
  use.
- Fully static output resolves layout immediately because Shape determines canonical strides,
  offset, view flag, and referenced span without backend or storage facts.
- Dynamic output remains unresolved rather than storing symbolic strides or deferring a partially
  numeric descriptor. This preserves the existing LayoutDescriptor contract.
- Resolved result layout is newly constructed and non-view. It describes requested logical result
  geometry; later physical buffer reuse is a compiler/planning/backend concern and does not turn
  this model descriptor into an alias view.
- Every call is a fresh explicit expression. Returning the input or eliminating nested requests
  would hide semantic provenance and perform compiler canonicalization inside the public model API.
- All DataTypes are accepted because contiguity changes layout representation, not logical element
  type. Gradient eligibility is copied unchanged but no gradient rule is introduced.
- Layout arithmetic overflow is reported rather than silently converting a fully static result to
  unresolved layout.

## Known limitations

- The expression does not copy values or make host/device data contiguous. No execution exists yet.
- Dynamic results do not expose numeric strides, offset, layout kind, or referenced span until
  later symbolic binding and preparation.
- The model does not eliminate an already-contiguous or repeated request.
- No gradient rule, compiler capture, logical materialization plan, backend lowering, runtime
  residency, storage alias/copy, ONNX mapping, or conformance behavior is implemented.

## Validation evidence

Planning read the architecture contract and focused lifecycle/module/dependency/runtime-boundary
explanations; documentation and planning rules; roadmap; model capabilities and master plan;
tasks 0002, 0003, 0007, 0011, 0012, 0013, 0017A; current Shape, layout, descriptor, Tensor,
factory, provenance, Operation, and contiguous semantic source/tests; Tensor/Compile/Training APIs
and glossary; and Java 26 Gradle configuration.

The read-only `legacy/pre-rewrite` branch was inspected directly. It confirms the public
no-argument `Tensor.contiguous()` capability, parameterless semantic identity, logical
shape/type/value preservation, support for every selected data type, and materialization evidence
for permuted, expanded, sliced, offset, CPU, Metal, and CUDA paths. Legacy immediate storage/view
construction, graph builders, operation traits, gradient callbacks, layout planners, physical
views, kernels, lowering, and runtime handling are excluded or assigned to later owners.

Planning selected one public method, one single-method package-private helper, and one focused test.
The current Shape and LayoutDescriptor contracts already support the static/dynamic distinction;
TensorDescriptor, provenance, and `createDerived` already support exact storage-free result
construction. No new public value type, dependency, foundational change, or architecture decision
is required.

Planning validation after synchronizing this task, the model master plan, and roadmap:

- `git diff --check` passed.
- The targeted trailing-whitespace scan returned no matches across the three changed planning
  files.
- All 186 local Markdown file links across the three planning files resolve.
- Markdown code-fence counts are balanced: ten in this task, two in the master plan, and zero in
  the roadmap.
- All 20 canonical task-specification headings are present, together with focused Capability
  origin, Required contract, and Javadoc requirements sections.
- Task, model master plan, and roadmap consistently identify 0017B as Ready. Task 0017C remains
  Draft, and no task-0017C specification exists.
- Package review confirms no new package: the public entry and package-private helper remain in
  `model.tensor`, while the completed semantic kind remains in `model.operation.layout`.
- Scope review confirms exactly ten permitted implementation paths and exactly three planning
  paths in the current diff. No Java, API, glossary, architecture, Gradle, AGENTS, completed-task,
  or other-module file changed during planning.
- Dependency review confirms Shape, layout, descriptor, Tensor/provenance/factory, and 0017A
  semantics are real prerequisites. Task 0017B does not depend on later layout/view rows and does
  not create a false sequential dependency chain.
- Static/dynamic review confirms that current public contracts support the planned branch:
  `Shape.isFullyStatic()` gates `LayoutDescriptor.contiguous(shape)`, while `TensorDescriptor`
  already requires dynamic shapes to use unresolved layout.

Implementation and independent documentation validation:

- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorContiguousExpressionTest --tests
  io.github.pho001.synaptik.model.tensor.TensorTest` passed. The focused suites contain 8 and 14
  tests respectively, with zero failures, errors, or skips.
- `./gradlew :modules:model:test --rerun-tasks` passed from fresh execution: 60 suites and 462
  tests, with zero failures, errors, or skips.
- `./gradlew :modules:model:javadoc --rerun-tasks` passed. Generated `Tensor.html` contains the
  updated type-level static/dynamic layout exception and the complete public `contiguous()`
  contract. The package-private helper is intentionally absent from the public generated API;
  its source Javadoc was independently reviewed and finalized.
- `./gradlew test --rerun-tasks` passed with all 36 actionable tasks executed. No architecture,
  backend-conformance, or integration behavior needed task-specific changes because this is a
  model-only metadata construction boundary.
- `javap -p -c -s` confirms the public method contains only one helper invocation and return. The
  final package-private helper has one private constructor, no fields, and one static `apply`
  method. Its bytecode performs one null check, one descriptor/Shape read, one
  `isFullyStatic()` branch, one static-branch `LayoutDescriptor.contiguous` call, and one shared
  descriptor/operation/provenance/`createDerived` path.
- Reflection tests confirm exactly 77 declared public Tensor methods and the exact zero-argument
  public, non-static, non-synchronized `Tensor contiguous()` signature. They also confirm the
  helper is final and package-private, stateless, and has exactly the required constructor and
  method surface.
- Source and import inspection found no input-layout read, host-storage/value access, copy,
  storage allocation, graph/compiler/planning/prepare/runtime/backend dependency, gradient rule,
  cache, service, overload, or hidden canonicalization. The implementation imports only existing
  model and Java collection/null-check contracts.
- The complete Tensor API Java example compiled with Java 26 against the built model classes and
  printed the documented static strides, offset, view flag, span, dynamic unresolved state,
  operation kind, exact input provenance, and absent storage.
- Local Markdown links across the six affected documentation/planning files resolve. Added
  `contiguous-expressions` and existing `contiguous-semantic-kind` anchors resolve, code fences are
  balanced, and targeted trailing-whitespace checks report no matches.
- Scope inspection reports exactly the ten authorized paths: two production files, two tests,
  three API/glossary files, and three planning files. `git diff --check`, untracked-file diff
  checks, newline, status, and no-task-0017C checks pass.
- Task 0017B, the model master plan, and roadmap consistently identify 0017B as Complete. Task
  0017C remains Draft without a detailed specification.
- Independent review found the Training API and model capabilities accurate without change: this
  task adds no gradient/training promise or capability-baseline category. `ContiguousKind`,
  Operation foundations, DataType, Shape, LayoutDescriptor/LayoutKind, TensorDescriptor,
  TensorFactory, TensorProvenance, storage and graph contracts, existing expression helpers, and
  their tests remain accurate because their behavior and public contracts did not change.
- ARCHITECTURE, focused architecture documents, ADRs, architecture tests, backend-conformance and
  integration material, Gradle/Java 26 configuration, dependencies, AGENTS, completed tasks,
  other modules, and examples outside the affected Tensor API remain accurate without change.

## Implementation notes

- Added the exact one-line `Tensor.contiguous()` delegation and one package-private final,
  stateless helper with the required static/dynamic Shape branch and shared derived-construction
  path.
- Added focused coverage for every data type, valid gradient choice, scalar/zero/dynamic Shape,
  every representative input layout state, exact provenance, storage/value non-interference,
  fresh repeated/nested requests, validation order, overflow, and identifier exhaustion.
- Updated the exact public Tensor method inventory from 76 to 77 without widening any other API.
- Finalized Tensor and helper Javadocs, Tensor API, Compile API, glossary, task evidence, model
  master plan, and roadmap in the mandatory independent documentation context. The documentation
  distinguishes current metadata construction from planned compiler canonicalization and
  materialization policy.
- No architecture decision, dependency, module boundary, storage behavior, gradient rule,
  compiler behavior, backend behavior, execution behavior, or later-task specification was added.

## Completion summary

Completed changes:

- implemented public storage-free contiguous expression construction with exact static-resolved
  and dynamic-unresolved descriptor rules;
- added exact operation/provenance construction and focused behavioral/structural tests; and
- finalized all affected Javadocs, API/glossary explanations, planning status, and validation
  evidence.

Files changed or created:

- `Tensor.java` and new `TensorContiguousExpressions.java`;
- `TensorTest.java` and new `TensorContiguousExpressionTest.java`;
- Tensor API, Compile API, glossary, this task, model master plan, and roadmap.

Tests and validation performed:

- focused Tensor suites, all model tests, generated model Javadoc, and all root tests passed;
- bytecode, reflection, source/import, generated-Javadoc, example, documentation-integrity,
  whitespace, scope, diff, newline, and synchronized-status checks passed.

Unresolved issues: None.

Required follow-up: None for task 0017B. Task 0017C remains a separate Draft planning frontier
without a detailed specification.

Status: Complete
