# Task 0017C: Reshape and Expand Semantics

## Status

Complete

## Goal

Define typed, backend-independent semantic identities and immutable target-shape attributes for
`RESHAPE` and `EXPAND`.

Both operations have one logical input and one exact target `Shape`, but they mean different
things. `RESHAPE` preserves the logical element sequence while changing its coordinate shape.
`EXPAND` preserves existing values while logically repeating compatible singleton dimensions or
adding leading dimensions. This task defines those meanings only; public Tensor requests,
compatibility validation, layout derivation, provenance, gradients, materialization, and execution
remain outside its scope.

## Scope

- Add one public `ShapeTransformKind` enum implementing `OperationKind`.
- Define exactly `RESHAPE` and `EXPAND`, in that order.
- Add one public `TargetShapeAttrs` record implementing `OperationAttrs` with exactly one
  `Shape targetShape` component.
- Reject null target Shape with exact parameter-name failure and retain every non-null immutable
  Shape reference unchanged.
- Accept scalar, zero-extent, ordinary static, and explicit dynamic target Shapes structurally.
- Document that stored target Shape is normalized model semantics and never a raw public
  reshape-request array containing a numeric `-1` inference sentinel.
- Pair both kinds explicitly with `TargetShapeAttrs` without changing generic `Operation`
  validation.
- Document one logical input, result-shape meaning, logical element-order behavior, and the
  semantic difference between shape reinterpretation and broadcast repetition.
- Add one focused same-package semantic-contract test for both cohesive types.
- Keep both types in the existing `model.operation.layout` package.
- Finalize Javadocs, Tensor API semantic reference, glossary, task evidence, model master plan, and
  roadmap through the required independent documentation pass during implementation.

## Out of scope

- public `Tensor.reshape`, `Tensor.expand`, another Tensor method, overload, static facade, factory,
  builder, expression helper, or task-0017D implementation
- raw requested-shape arrays, varargs, a public numeric `-1` sentinel, inferred-dimension
  calculation, element-count comparison, divisibility, overflow policy, or request normalization
- input Tensor, input Shape, input/output DataType, `requiresGrad`, descriptor construction, label,
  identity, provenance, host storage, or `TensorFactory.createDerived`
- reshape view eligibility, input contiguity, effective strides, storage offset, view flag,
  broadcast zero strides, result `LayoutDescriptor`, aliasing, copying, or materialization policy
- right-aligned expand compatibility validation, target-rank validation, singleton-axis checks,
  symbolic constraint solving, dynamic-dimension binding, or graph-wide inference
- contiguous, permute, transpose, expand-dimensions, squeeze, slice, pad, tile, concat, stack,
  unstack, unfold, fold, select, gather, or scatter semantics
- gradients, reshape backward, broadcast reduction, autograd, compiler-generated operations,
  optimizer, or training behavior
- operation factories, registries, parsers, visitors, aliases, string dispatch, reflection
  discovery, maps, services, arity fields, result-kind fields, cost, fusion, capability, backend
  support, lowering, route, kernel, executable, or physical-buffer metadata
- compiler capture/canonicalization, planning ownership/materialization, prepare, runtime residency,
  backend storage/execution, engine, trace, ONNX mapping, or conformance
- changes to `ContiguousKind`, Operation foundations, Shape/Dimension, LayoutDescriptor/LayoutKind,
  Tensor contracts, graph records, existing Java tests, dependencies, Gradle, architecture, or
  another module
- a detailed task-0017D specification or any later task implementation

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
- [Task 0005](0005-operation-semantic-foundation.md)
- [Task 0006](0006-operation-model.md)
- [Task 0013](0013-tensor-provenance-skeleton.md)
- [Task 0017A](0017a-contiguous-semantic-kind.md)
- [Task 0017B](0017b-contiguous-tensor-expression.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only `legacy/pre-rewrite` branch exposes:

```java
Tensor reshape(int... requestedShape)
Tensor expand(int... requestedShape)
```

Legacy reshape accepts one `-1` request dimension, infers it from the input element count, rejects
multiple sentinels or incompatible counts, preserves row-major logical element order, and chooses
between a view and a materialized result according to input layout. Legacy expand accepts a target
rank at least as large as the source rank, aligns source axes from the right, retains equal axes,
repeats singleton axes, permits new leading axes, and represents repetition through zero strides.

The new model deliberately supports rank-zero scalar Shapes, zero extents, long static dimensions,
and explicit dynamic dimensions. Therefore legacy positive-int restrictions and empty-shape
rejection are not copied into the immutable semantic attributes. Task 0017D will specify the new
public request forms, local inference/compatibility rules, result descriptor, and provenance.

Legacy mutable arrays, immediate layout/storage views, graph builders, gradient callbacks,
operation traits, cost/fusion/result tags, materialization selection, kernels, lowering, fallback,
and runtime state are not copied. They are either implementation details or responsibilities of
later compiler, planning, prepare, runtime, training, and backend tasks.

## Architecture constraints

- Production remains in `modules/model`, which owns backend-independent operation semantics and
  immutable Shape values.
- `ShapeTransformKind` identifies semantic meaning only. It is not a Tensor, graph occurrence,
  result descriptor, layout, materialization requirement, executable operation, or backend route.
- `RESHAPE` means a one-input coordinate-shape change that preserves the ordered logical element
  sequence and therefore requires compatible element counts after local or graph-wide binding.
- `EXPAND` means a one-input logical broadcast to a target Shape through compatible singleton or
  leading dimensions; repeated logical positions may later use zero-stride view geometry.
- `TargetShapeAttrs` stores the exact semantic result Shape shared by both kinds. It does not store
  the input Shape, raw request syntax, inferred-axis index, input/output strides, layout, or
  compatibility proof.
- A stored `Shape` has already passed its own model invariants. Static dimensions are non-negative,
  dynamic dimensions use explicit symbols, scalar Shape has rank zero, and zero extents are valid.
- Numeric `-1` is public reshape request syntax only. It cannot appear inside `Shape` and is never
  stored by these attributes. Task 0017D owns request normalization before Operation construction.
- The attributes constructor validates only non-nullity. It cannot compare element counts, decide
  reshape view eligibility, validate expand compatibility, bind symbols, or derive layout.
- Generic `Operation` remains an open kind/attributes pair and does not enforce family
  compatibility, arity, target-shape rules, descriptor inference, gradients, or backend support.
- Package direction is `model.operation.layout -> model.operation + model.shape`. It must not
  depend on tensor, layout-value, storage, graph, compiler, planning, prepare, runtime, backend,
  engine, trace, or training packages.
- Stop if implementation needs input state, arrays, another attributes type, result inference,
  layout/materialization behavior, dependency, or architecture change.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.operation.layout` — owns cohesive layout/view operation
  semantics and receives both new public contracts.
- `io.github.pho001.synaptik.model.operation` — supplies `OperationKind`, `OperationAttrs`, and
  generic immutable `Operation` composition.
- `io.github.pho001.synaptik.model.shape` — supplies the immutable `Shape` target value.

No package is added or moved.

Type placement:

- `io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind` — public enum for the two
  target-shape-based semantic identities.
- `io.github.pho001.synaptik.model.operation.layout.TargetShapeAttrs` — public immutable exact
  target-Shape parameter shared by both kinds.
- `ShapeTransformSemanticsTest` — same-package focused test for vocabulary, record invariants,
  typed composition, and dependency boundaries.

## Required contract

### Semantic kind vocabulary

Create exactly:

```java
public enum ShapeTransformKind implements OperationKind {
    RESHAPE,
    EXPAND
}
```

The enum declares no project field, explicit constructor, method, nested type, per-constant class
body, alias, symbol, or metadata. Compiler-generated enum machinery is not additional project API.
Inherited `Enum.name()` satisfies `OperationKind.name()` and returns the exact constant text.

| Kind | One-input semantic meaning | Not decided here |
|---|---|---|
| `RESHAPE` | preserve the ordered logical element sequence while interpreting it through `targetShape` coordinates | count compatibility, view/copy choice, layout, gradients, execution |
| `EXPAND` | logically repeat compatible singleton or new leading dimensions to `targetShape` | broadcast validation, zero strides, aliasing, gradients, execution |

Neither kind stores arity, target Shape, input facts, output facts, layout, or backend metadata.

### Target-shape attributes

Create exactly:

```java
public record TargetShapeAttrs(Shape targetShape) implements OperationAttrs
```

The record has exactly one component, one public canonical constructor, one explicit documented
`targetShape()` accessor, and record-generated `equals`, `hashCode`, and `toString`. Add no
overload, factory, builder, array accessor, inferred-axis field, input Shape, layout, cache, nested
type, or helper API.

The canonical constructor performs exactly:

```java
targetShape = Objects.requireNonNull(targetShape, "targetShape");
```

It retains the exact immutable Shape reference. Null fails with `NullPointerException` and exact
message `targetShape`. Scalar, zero-extent, ordinary static, mixed static/dynamic, and fully dynamic
Shapes are accepted structurally without element-count or broadcast validation.

Record equality and hashing use structural Shape equality. Generated text is diagnostic only and
is not serialization, public request syntax, parser input, ONNX mapping, backend dispatch, or a
layout plan.

### Typed composition

Document these exact valid pairings:

```java
new Operation(ShapeTransformKind.RESHAPE, attrs)
new Operation(ShapeTransformKind.EXPAND, attrs)
```

Both operations retain the exact `TargetShapeAttrs` reference. Do not use
`NoOperationAttrs.INSTANCE`, add a family factory, expose an enum `operation()` method, or change
generic `Operation` to validate pairings. A target Shape is intrinsic semantic data for both
operations even when it equals the eventual input Shape.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/ShapeTransformKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/TargetShapeAttrs.java`

Test:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/layout/ShapeTransformSemanticsTest.java`

Documentation/planning:

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless a task-related inconsistency requires stopping:

- `docs/api/compile-api.md`
- `docs/api/training-api.md`
- `docs/planning/modules/model/capabilities.md`
- `ContiguousKind`, Operation foundations, Shape/Dimension, LayoutDescriptor/LayoutKind, Tensor
  contracts and expressions, graph contracts, and their Javadocs/tests
- focused architecture documents, ADRs, architecture tests, backend-conformance material,
  integration tests, Gradle configuration, and every other module

## Maximum scope

At most two production files, one focused test, and five documentation/planning files: eight paths
total.

Do not modify existing Java source/tests, capabilities, Compile/Training API, Gradle, AGENTS,
architecture documents/tests, another module, completed tasks, or unrelated documentation. Stop if
another production concept, input validation, Tensor expression, layout rule, dependency, or ninth
path is needed. Do not create task 0017D.

## Javadoc requirements

- Document `ShapeTransformKind` as typed one-input target-shape transformation semantics rather
  than Tensor behavior, resolved layout, graph occurrence, or executable support.
- Document `RESHAPE` as ordered logical element-sequence preservation under new coordinates and
  `EXPAND` as logical singleton/leading-axis repetition.
- Explain that arity and target-shape compatibility are family context, not enum metadata.
- Document `TargetShapeAttrs` construction, exact reference retention, nullability, accepted Shape
  categories, record value semantics, diagnostic text, and explicit accessor.
- Explain why stored `Shape` is normalized model semantics and cannot contain a raw numeric `-1`
  inference sentinel.
- Document exact Operation pairings and absence of generic compatibility validation.
- Defer input/target validation, count inference, broadcasting, layout/view/copy decisions,
  provenance, gradients, compiler behavior, materialization, backend support, and execution.
- Review ContiguousKind, Shape, Dimension, Operation/OperationAttrs/OperationKind, LayoutDescriptor,
  LayoutKind, TensorDescriptor, and Tensor Javadocs and record why they remain accurate, or stop on
  an inconsistency.

## Acceptance criteria

- Exactly one public `ShapeTransformKind` enum and one public `TargetShapeAttrs` record are added in
  the existing operation-layout package.
- The enum implements `OperationKind` and declares exactly `RESHAPE`, then `EXPAND`.
- The enum adds no project field, method, constructor, nested type, constant body, alias, arity,
  target, layout, cost, fusion, backend, inference, materialization, or execution metadata.
- The record implements `OperationAttrs` and has exactly one `Shape targetShape` component, one
  canonical constructor, one explicit accessor, and no additional state or API.
- Null target Shape fails with exact `NullPointerException` message `targetShape`; every non-null
  Shape is retained by exact reference.
- Scalar, zero-extent, static, and dynamic target Shapes are accepted without input-dependent
  validation.
- Record equality/hash/text remain ordinary value/diagnostic behavior; the accessor returns the
  exact stored Shape reference.
- Both kinds compose with exact `TargetShapeAttrs`; neither composes with no-attributes in the
  documented family contract, and no compatibility validator is added.
- Production imports only Operation contracts, Shape, and `Objects`. No Tensor, layout-value,
  storage, graph, compiler, planning, prepare, runtime, backend, engine, trace, training,
  dependency, or executable behavior is added.
- No public Tensor method, raw request array, `-1` inference, element-count/broadcast validation,
  result descriptor, layout geometry, provenance, gradient, or execution is implemented.
- Focused/aggregate tests, model Javadoc, root tests, reflection/javap/import/source/scope checks,
  documentation links/formatting, and synchronized statuses pass.
- A separate clean-context documentation agent finalizes both Javadocs, Tensor API, glossary,
  task evidence, master plan, and roadmap and records related no-change conclusions.
- Task 0017C becomes Complete only after both passes. Task 0017D remains Draft without a detailed
  specification.

## Tests / validation

Run before and after the documentation-focused review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.layout.ShapeTransformSemanticsTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused test must cover exact package/visibility/interfaces; enum constant names and order;
absence of project enum state, behavior, nested types, aliases, and constant bodies; record status,
component order/type, constructor/accessor surface, absence of extra fields/methods/nested types;
null failure; exact reference retention for scalar/static/zero/dynamic Shapes; structural record
equality; typed distinction from a private test-local equal-name kind; and explicit Operation
composition retaining exact kind and attributes references.

Manually inspect `javap -p -c -s`, reflection, imports, and Gradle dependencies. Confirm no Tensor,
layout-value, storage, provenance, graph/compiler/planning/prepare/runtime/backend type, raw shape
array, sentinel, count/broadcast inference, gradient, cost, fusion, route, registry, map, or service
appears. Validate generated Javadoc, Tensor API semantic status, glossary terms, links, anchors,
fences, whitespace, exact eight-path scope, synchronized statuses, and absence of a task-0017D
specification.

## Dependencies

- Task 0002 supplies immutable static/dynamic/scalar/zero-extent `Shape` values.
- Task 0005 supplies `OperationKind` and `OperationAttrs`.
- Task 0006 supplies immutable generic `Operation` composition and reference retention.
- Completed tasks 0017A–0017B establish the operation-layout package and semantic/expression split;
  they are implementation-order context rather than Java dependencies of these two types.

## Follow-up tasks

- 0017D remains Draft for exact public reshape/expand request forms, `-1` inference, local
  element-count and right-aligned expansion validation, result descriptor/layout construction,
  provenance, and storage-free derived Tensor creation.
- 0017E–0017F remain Draft for permutation, transpose, expand-dimensions, and squeeze semantics and
  expressions.
- Compiler later owns graph-wide symbolic constraints and canonicalization.
- Planning later derives logical materialization requirements without selecting a concrete route.
- Backend prepare later chooses view, alias, copy, or backend-native lowering; runtime executes the
  prepared schedule.
- Training and compiler-generated semantic tasks later own differentiation and backward forms.

Do not create a detailed follow-up specification in this task.

## Architecture impact

Expected impact: None. The architecture already assigns backend-independent operation semantics,
Shape, and logical layout facts to `modules/model`; compiler inference to compiler; materialization
requirements to planning; concrete lowering/storage to backend prepare; and prepared execution to
runtime.

If implementation requires Tensor behavior, result inference, layout/materialization policy,
backend metadata, another dependency, or architecture change, stop and report it.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0002/0003/0005/0006/0013/0017A/0017B/0017C, Tensor API,
Compile API, Training API, glossary, current Shape/Dimension/OperationKind/OperationAttrs/
Operation/ContiguousKind contracts and focused tests, and Java 26 Gradle configuration.

Implement task 0017C exactly. Add only ShapeTransformKind.java, TargetShapeAttrs.java, and
ShapeTransformSemanticsTest.java under io.github.pho001.synaptik.model.operation.layout for Java
code and tests.

The public enum contains exactly RESHAPE and EXPAND in that order and adds no project state,
methods, aliases, arity, target, layout, or metadata. The public record implements OperationAttrs
with exactly non-null Shape targetShape, exact NullPointerException message targetShape, explicit
documented accessor, exact reference retention, and no other API/state. Accept every current
Shape category. Compose each kind explicitly with TargetShapeAttrs.

Document RESHAPE as preserving ordered logical elements under target coordinates and EXPAND as
logical singleton/leading-axis repetition. The stored Shape is normalized semantics, never raw
public request syntax or a numeric -1 sentinel. Do not implement Tensor methods, raw arrays,
inference, element-count/broadcast validation, descriptors, layouts, provenance, gradients,
compiler/planning/prepare/runtime/backend behavior, dependencies, build/architecture changes,
existing Java edits, or later specs. Stop beyond eight paths or on architecture uncertainty.

Run every specified focused/aggregate test, Javadoc, javap/reflection/import/manual,
documentation/link/whitespace/scope/status check. Then hand the actual diff and evidence to a
separate clean-context documentation agent in the same change. It must inspect source/tests/
generated Javadoc, finalize permitted Javadocs/Tensor API/glossary/planning, record related-
contract/capability/Compile API/Training API/architecture no-change conclusions, and rerun
validation.

Update task 0017C, model master plan, and roadmap only for planning status/evidence. Do not mark
0017C Complete until both passes succeed. Leave 0017D Draft without a specification. Do not commit
or push.
```

## Local decisions

- `ShapeTransformKind` groups exactly the two transformations whose intrinsic parameter is a
  complete target Shape. Contiguous has no parameter and remains in `ContiguousKind`; later axis,
  slice, pad/tile, composition, and window families need different typed attributes.
- `TargetShapeAttrs` is shared because both operations store the same semantic fact. Separate
  reshape/expand records would duplicate an identical immutable contract without adding meaning.
- The record stores `Shape`, not `long[]`, `int[]`, or `List<Long>`, so semantic attributes reuse
  canonical static/dynamic/scalar/zero-extent invariants and require no defensive array copy.
- Every non-null Shape is accepted structurally. Input-dependent count and broadcast validation
  belongs to task 0017D, which has both input and target context.
- Numeric `-1` belongs only to the future public reshape request boundary. It is normalized before
  `TargetShapeAttrs` construction and never becomes model Shape state.
- RESHAPE and EXPAND remain distinct kinds despite sharing attributes because one reinterprets an
  ordered element sequence while the other introduces logical repetition.
- No alias/view/materialization flag is stored. Logical geometry and executable realization are
  derived later by their owning layers.

## Known limitations

- No public reshape or expand expression exists until task 0017D.
- The attributes cannot prove element-count compatibility, singleton expansion compatibility,
  target rank, symbolic constraints, view eligibility, or layout geometry.
- Raw public `-1` inference syntax and exact failure messages remain unspecified until task 0017D.
- No provenance, gradient rule, compiler capture, materialization plan, backend lowering, runtime
  execution, ONNX mapping, or conformance behavior is implied.

## Validation evidence

Planning read the architecture contract and focused lifecycle/module/dependency/runtime-boundary
explanations; documentation and planning rules; roadmap; model capabilities and master plan;
tasks 0002, 0003, 0005, 0006, 0013, 0017A, and 0017B; current Shape/Dimension, Operation, layout
semantic/value, Tensor, and descriptor contracts/tests; Tensor/Compile/Training APIs and glossary;
and Java 26 Gradle configuration.

The read-only `legacy/pre-rewrite` branch was inspected directly. It confirms public reshape and
expand capabilities, a single raw reshape `-1` request sentinel, element-count inference,
right-aligned singleton/leading-axis expansion, immutable operation target-shape meaning, view or
materialization decisions, gradient construction, and CPU/accelerator execution evidence. Legacy
arrays, positive-int restrictions, scalar/zero rejection, graph builders, storage views, operation
traits, planners, kernels, lowering, and runtime code are excluded or assigned to later owners.

Planning selected one two-value enum, one shared exact-Shape attributes record, and one focused
test. Existing Shape invariants provide normalized semantic target values, while generic Operation
already supplies typed composition. No Tensor behavior, result inference, layout contract,
dependency, foundational change, or architecture decision is required.

Planning validation after synchronizing this task, the model master plan, and roadmap:

- `git diff --check` passed.
- The targeted trailing-whitespace scan returned no matches across the three changed planning
  files.
- All 190 local Markdown file links across the three planning files resolve.
- Markdown code-fence counts are balanced: fourteen in this task, two in the master plan, and zero
  in the roadmap.
- All 20 canonical task-specification headings are present, together with focused Capability
  origin, Required contract, and Javadoc requirements sections.
- Task, model master plan, and roadmap consistently identify 0017C as Ready. Task 0017D remains
  Draft, and no task-0017D specification exists.
- Package review confirms both planned public types remain in the existing
  `model.operation.layout` package and introduce only the allowed dependency on foundational
  `model.shape` plus Operation contracts.
- Scope review confirms exactly eight permitted implementation paths and exactly three planning
  paths in the current diff. No Java, API, glossary, architecture, Gradle, AGENTS, completed-task,
  or other-module file changed during planning.
- Contract review confirms `Shape` already excludes negative static dimensions, represents scalar
  and zero extents explicitly, and models dynamic dimensions through symbols. Therefore the stored
  target Shape cannot contain a raw numeric `-1` sentinel and needs no duplicate validation.
- Dependency review confirms 0002 and 0005–0006 are the real Java prerequisites. Tasks 0017A–0017B
  provide package and sequencing context without creating false source dependencies.

Implementation and independent documentation validation:

- Implementation context `/root/implement_model_0017c` added exactly
  `ShapeTransformKind.java`, `TargetShapeAttrs.java`, and `ShapeTransformSemanticsTest.java`.
  Clean documentation context
  `/root/implement_model_0017c/review_model_0017c_docs` independently read the required
  architecture, documentation and planning profiles, plans and historical tasks, APIs, glossary,
  Java 26 build configuration, final source/test, generated Javadoc, XML reports, and complete
  workspace diff. It applied General and API/Javadoc style to Java, Tensor API, and glossary;
  Planning style to this task, the model master plan, and roadmap; and Example format to the
  conceptual reshape/expand comparison.
- Independent source and generated-Javadoc review found both submitted production Javadocs
  complete unchanged. They document one-input family context, exact RESHAPE and EXPAND meaning,
  explicit `TargetShapeAttrs` pairings, accepted Shape categories, exact reference retention,
  null failure, normalized semantic Shape versus raw `-1` request syntax, record value semantics,
  diagnostic text, and the required inference/layout/provenance/gradient/compiler/planning/
  backend/execution boundaries.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.operation.layout.ShapeTransformSemanticsTest` —
  `BUILD SUCCESSFUL`; XML records 8 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:test` — `BUILD SUCCESSFUL`; 61 XML suites record 470 tests with zero
  failures, errors, or skips.
- `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL`. Generated pages contain the enum type,
  both constants, record type/component, canonical constructor, explicit accessor, exception,
  ownership/value semantics, exact pairings, normalized-Shape boundary, and cross-layer
  exclusions.
- `./gradlew test` — `BUILD SUCCESSFUL`; all repository test tasks completed without failure.
- `javap -classpath modules/model/build/classes/java/main -p -c -s` confirms the enum has exactly
  `RESHAPE`, then `EXPAND`, plus compiler-generated enum machinery and no project field or method.
  It confirms the record has exactly one private final `Shape targetShape` field, a canonical
  constructor whose only semantic validation is
  `Objects.requireNonNull(targetShape, "targetShape")`, a direct accessor, and record-generated
  equality, hashing, and diagnostic text. The eight focused reflection tests independently cover
  the same surface and exact Operation composition.
- Production imports are exactly `OperationKind` for the enum and `OperationAttrs`, `Shape`, and
  `Objects` for the record. Source and bytecode inspection found no Tensor, raw array, inferred
  axis, count/broadcast validation, descriptor, layout-value, provenance, gradient, graph,
  compiler, planning, prepare, runtime, backend, engine, trace, training, registry, map, service,
  or executable state or behavior.
- Tensor API now presents `ShapeTransformKind` and `TargetShapeAttrs` as current semantic
  contracts, explains normalized target-Shape ownership and the RESHAPE-versus-EXPAND distinction,
  and keeps public Tensor construction and cross-layer behavior planned. The glossary synchronizes
  implementation status, the OperationKind/OperationAttrs distinction, and reusable reshape/
  expand terminology.
- A targeted Markdown check validated 285 local links and heading anchors across the five changed
  documentation/planning files. Backtick fences are balanced, no tilde fence is present, every
  changed file ends with a newline, targeted trailing-whitespace scans return no matches, and
  `git diff --check` passes.
- Exact scope is the eight authorized paths: two production files, one focused test, Tensor API,
  glossary, this task, model master plan, and roadmap. Task 0017C, the master-plan row/current
  status/notes, and roadmap frontier/table are synchronized as Complete. Task 0017D remains Draft,
  and no task-0017D specification exists. No commit or push was performed.
- `Operation`, `OperationAttrs`, and `OperationKind` remain accurate unchanged because the new
  values conform to their existing open typed-pairing contracts without changing compatibility
  validation. `Shape` and `Dimension` remain accurate because the record only retains an already
  valid immutable Shape. `ContiguousKind` remains accurate because its parameterless canonical-
  geometry request is distinct from target-shape transformations. `LayoutDescriptor`,
  `LayoutKind`, `TensorDescriptor`, and `Tensor` remain accurate because no resolved geometry,
  result descriptor, public method, provenance, or storage behavior was added; their focused tests
  therefore require no change.
- `capabilities.md` already inventories reshape/expand capability and correctly separates model
  semantics, public Tensor construction, compiler, planning, backend prepare, runtime, and tests.
  Compile API remains accurate because no public expression, capture, inference, canonicalization,
  artifact, or materialization behavior was added. Training API remains accurate because no
  gradient, autograd, optimizer, parameter, publication, or session behavior changed.
- `ARCHITECTURE.md`, focused architecture explanations, ADRs, architecture tests, backend-
  conformance material, and integration tests remain accurate unchanged because module ownership,
  dependency direction, lifecycle, backend behavior, and end-to-end behavior did not change.
  Root/model Java 26 Gradle configuration, dependencies, other modules, and unrelated tests also
  remain unchanged because the task adds only model-owned semantic vocabulary.

## Implementation notes

- Added the exact two-value `ShapeTransformKind` enum and one-component `TargetShapeAttrs` record in
  the existing operation-layout package, with no additional project API or state.
- Added the focused eight-test suite for vocabulary/order, exact enum and record reflection shape,
  every current Shape category, null failure, reference retention, record value semantics, typed
  identity, exact Operation composition, and cross-layer state exclusion.
- Finalized Tensor API, glossary, task evidence, model master plan, and roadmap in the mandatory
  independent documentation context. The submitted production Javadocs required no correction.
- Added no Tensor expression, request array, inference, compatibility validation, descriptor,
  layout, provenance, gradient, compiler, planning, backend, execution, dependency, build, or
  architecture behavior.

## Completion summary

- Completed changes: Implemented and documented exact RESHAPE and EXPAND semantic identities with
  one shared immutable normalized target-Shape attributes value.
- Files changed or created: Exactly two production Java files, one focused test, Tensor API,
  glossary, this task, model master plan, and roadmap.
- Tests and validation: Focused tests passed 8/8; all 470 model tests across 61 suites, generated
  model Javadoc, root tests, javap/reflection/source/import/generated-documentation checks, 285
  Markdown link/anchor checks, fence/terminology/whitespace checks, exact scope/status checks, and
  `git diff --check` passed.
- Documentation-agent review: Clean context
  `/root/implement_model_0017c/review_model_0017c_docs` completed the independent pass using
  General, API/Javadoc, Planning, and Example profiles.
- Documentation impact: Tensor API and glossary now present target-shape semantics as current while
  keeping public request normalization, compatibility validation, expressions, descriptors,
  provenance, and cross-layer behavior planned.
- Javadoc review: Both new production Javadocs are complete unchanged; related operation, Shape,
  contiguous, resolved-layout, Tensor descriptor/expression, and cross-layer contracts remain
  accurate for the reasons recorded above.
- Glossary impact: Implementation status, target-shape attributes, and the reusable reshape versus
  expand distinction now reflect the implemented contract.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0017C. Task 0017D remains Draft without a detailed
  specification.

Status: Complete
