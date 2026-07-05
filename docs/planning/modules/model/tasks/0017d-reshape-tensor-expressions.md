# Task 0017D: Reshape Tensor Expressions

## Status

Complete

## Goal

Add public `Tensor.reshape` expressions that preserve the ordered logical element sequence while
changing its coordinate Shape.

The API supports a legacy-compatible `long...` request with at most one numeric `-1` inference
sentinel and an exact `Shape` overload for normalized static or dynamic model Shapes. Locally
provable element counts must match. A resolved contiguous input may produce resolved alias-view
geometry for a fully static target; otherwise layout remains unresolved for later compiler,
planning, and backend work.

## Scope

- Add exactly `Tensor.reshape(long... requestedShape)` and `Tensor.reshape(Shape targetShape)`.
- Add one package-private final stateless `TensorReshapeExpressions` helper.
- Normalize a defensive copy of the raw request without retaining or mutating caller arrays.
- Accept non-negative requested dimensions and at most one exact `-1` sentinel; reject every other
  negative value.
- Treat an empty raw request as canonical rank-zero scalar Shape.
- Infer `-1` only from a fully known input element count and a non-zero checked product of the
  other requested dimensions.
- Accept zero extents, including inferred zero when the other requested product is non-zero; reject
  ambiguous inference when the known requested product is zero.
- Reject unequal element counts when both input and target counts are known; defer the equality
  constraint when either Shape is dynamic.
- Accept an exact non-null target Shape reference without copying or normalization.
- Retain exact input DataType, exact normalized target Shape, and unchanged `requiresGrad`.
- Resolve a result alias-view layout only when the input has resolved contiguous geometry and the
  target Shape is fully static.
- Preserve the input layout's element offset, use target canonical row-major strides, and mark the
  resolved result layout as a view.
- Leave result layout unresolved for absent, strided, broadcast, or dynamic geometry instead of
  choosing materialization.
- Construct exact `ShapeTransformKind.RESHAPE` plus `TargetShapeAttrs(targetShape)` semantics,
  exact one-input provenance, no label, and no storage.
- Return a fresh expression for same-shape, repeated, and nested reshape requests.
- Accept all six current DataTypes, valid gradient eligibility, scalar/zero/static/dynamic Shapes,
  and resolved or unresolved input layouts.
- Add one focused same-package expression test and update `TensorTest` only for the two deliberate
  public overloads.
- Finalize affected Javadocs, Tensor API, Compile API, glossary, task evidence, model master plan,
  and roadmap through the required independent documentation pass during implementation.

## Out of scope

- public `Tensor.expand`, expand broadcasting, zero-stride derivation, or task-0017D1 implementation
- `int...` overload, generic collection/array overload, multiple inferred dimensions, named
  dimensions, flatten convenience, memory-order option, copy/view flag, destination storage, or
  another public type
- storing raw request arrays, numeric `-1` in Shape or `TargetShapeAttrs`, an inferred-axis field,
  an element-count proof object, or a symbolic equation object
- binding dynamic symbols, solving graph-wide element-count constraints, accepting `-1` for a
  dynamic input, or silently inventing a dynamic symbol for the sentinel
- reading or copying tensor values, allocating or attaching host storage, mutating input metadata,
  reusing input layout objects, or claiming eager materialization
- treating a strided or broadcast input as a reshape view, forcing a contiguous result, inserting
  `contiguous()`, or selecting alias/copy policy when view geometry cannot be proven locally
- returning the input, eliminating same-shape or nested reshapes, merging provenance, constant
  folding, or compiler canonicalization
- data-type conversion or promotion, new gradient rules, backward operations, saved values,
  autograd expansion, optimizer, or training execution
- graph capture/traversal, compiler pass implementation, planning ownership/materialization,
  prepare/runtime/backend behavior, engine, tracing, ONNX mapping, or conformance
- changes to Shape/Dimension, LayoutDescriptor/LayoutKind, TensorDescriptor, ShapeTransformKind,
  TargetShapeAttrs, TensorFactory, TensorProvenance, Operation, storage or graph contracts,
  existing expression helpers, or their tests
- dependencies, Gradle, Java version, AGENTS, ARCHITECTURE, focused architecture documentation,
  architecture tests, capabilities, another module, completed tasks, or unrelated documentation
- a detailed task-0017D1 or task-0017E specification

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
- [Task 0017B](0017b-contiguous-tensor-expression.md)
- [Task 0017C](0017c-reshape-and-expand-semantics.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only `legacy/pre-rewrite` branch exposes:

```java
Tensor reshape(int... requestedShape)
```

Legacy behavior accepts one `-1`, infers it from a known flat size, rejects multiple sentinels or
incompatible counts, preserves logical element order and DataType, creates a view for contiguous
input, and otherwise selects a materialized result. Legacy gradient construction reshapes the
output gradient back to the source Shape.

The new API widens raw dimensions to `long`, permits empty varargs as scalar Shape, supports zero
extents, and adds an exact `Shape` overload so explicit dynamic model Shapes are representable.
Those differences follow the completed Shape contract rather than legacy int-array, positive-only,
rank-at-least-one limitations. Locally unprovable dynamic count equality is deferred; a raw `-1`
cannot be inferred from an unknown input count.

Legacy immediate storage views, automatic materialized output, mutable arrays, graph builders,
gradient callbacks, operation traits, layout planners, physical buffers, kernels, lowering,
fallback, and runtime state are not copied. This task creates only model expression metadata and
locally provable view geometry.

## Architecture constraints

- `Tensor` remains public mutable API state and is not an intermediate-representation node.
- The result is a fresh storage-free expression with immutable provenance, not an eagerly reshaped
  buffer, compiled node, runtime value, prepared unit, or executable copy.
- `ShapeTransformKind.RESHAPE` plus `TargetShapeAttrs` are the exact semantic pair. Raw `-1` request
  syntax is normalized before attributes construction and is never stored in model Shape state.
- Reshape preserves the ordered logical element sequence. When both element counts are known they
  must be equal; unknown dynamic equality becomes a later compiler constraint.
- A resolved reshape view is locally provable only when input geometry is contiguous and target
  geometry is fully static. It uses canonical target strides, the exact input element offset, and
  explicit view metadata.
- A resolved input layout is compatible only with a fully static input Shape by existing
  `TensorDescriptor` invariants. No additional storage or backend fact is needed to describe the
  logical view.
- Absent/non-contiguous input layout or dynamic target Shape yields unresolved result layout. The
  model must not force materialization, insert another operation, or guess physical geometry.
- Resolved alias-view metadata does not attach host storage, allocate a buffer, promise zero-copy
  execution, or require one backend route. Planning and backend prepare retain those decisions.
- Every valid request remains explicit and fresh. Compiler later owns redundant reshape
  elimination, reshape-chain canonicalization, and graph-wide symbolic reasoning.
- All current DataTypes are accepted because reshape changes coordinates, not logical values.
  Gradient eligibility is retained unchanged but no gradient rule is added.
- The helper remains in `model.tensor`; semantics remain in `model.operation.layout`; Shape and
  LayoutDescriptor remain foundational immutable values. No project dependency changes.
- Stop if implementation requires new public value types, dynamic binding, storage access,
  materialization policy, cross-module state, or architecture changes.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.tensor` — owns public Tensor entries, the package-private helper,
  descriptor/provenance construction, and the existing derived factory seam.
- `io.github.pho001.synaptik.model.operation.layout` — supplies `ShapeTransformKind.RESHAPE` and
  `TargetShapeAttrs`.
- `io.github.pho001.synaptik.model.operation` — supplies immutable `Operation`.
- `io.github.pho001.synaptik.model.shape` — supplies normalized static/dynamic/scalar Shape and
  checked element counts.
- `io.github.pho001.synaptik.model.layout` — supplies canonical target strides and explicit
  resolved view geometry.

No package is added or moved.

Type placement:

- `io.github.pho001.synaptik.model.tensor.Tensor` — gains the two public reshape overloads.
- `io.github.pho001.synaptik.model.tensor.TensorReshapeExpressions` — package-private stateless
  coordinator colocated with Tensor and `TensorFactory.createDerived`.
- `TensorReshapeExpressionTest` — same-package focused validation of public overloads, private
  normalization rules, layout/provenance construction, and side-effect boundaries.
- `TensorTest` — changes only its exact public-method inventory from 77 to 79 and adds both reshape
  signatures.

## Required contract

### Public Tensor methods

Add exactly:

```java
public Tensor reshape(long... requestedShape) {
    return TensorReshapeExpressions.apply(this, requestedShape);
}

public Tensor reshape(Shape targetShape) {
    return TensorReshapeExpressions.apply(this, targetShape);
}
```

Both methods are public, final by virtue of Tensor, non-static, and non-synchronized. Each body
contains exactly one delegation and no inline validation, inference, layout logic, storage access,
or optimization. Java widens individual int literals to the long varargs element type; an existing
`int[]` is not accepted as a hidden compatibility overload.

### Helper shape

Create one package-private final class with no fields or nested types, one private zero-argument
constructor, and exactly these six methods:

```java
static Tensor apply(Tensor input, long[] requestedShape)
static Tensor apply(Tensor input, Shape targetShape)
private static Shape normalizeRequestedShape(Shape inputShape, long[] requestedShape)
private static void validateTargetShape(Shape inputShape, Shape targetShape)
private static Optional<LayoutDescriptor> resolveViewLayout(
        TensorDescriptor inputDescriptor, Shape targetShape)
private static Tensor create(
        Tensor input,
        TensorDescriptor inputDescriptor,
        Shape targetShape,
        Optional<LayoutDescriptor> resultLayout)
```

Add no overload, product helper, validator type, builder, cache, registry, service, policy object,
or other production file. Product and sentinel handling remain local to
`normalizeRequestedShape`.

### Raw request validation and inference

The raw-array entry follows this order:

1. Null-check `input` with message `input`.
2. Null-check `requestedShape` with message `requestedShape`.
3. Read the exact input descriptor once and its exact Shape once.
4. Pass the Shape and caller array to `normalizeRequestedShape`.
5. Resolve view layout and perform common construction.

`normalizeRequestedShape` never mutates or retains the caller array. It first scans every element
in ascending index order:

- non-negative values are ordinary static dimensions, including zero;
- exact `-1` marks the sole inferred dimension;
- any value below `-1` fails with
  `requestedShape[<index>] must be non-negative or -1: <value>`; and
- a second `-1` fails with `requestedShape must contain at most one -1`.

With no sentinel, create `Shape.of(requestedShape)` and apply the common target validation. Empty
varargs therefore produce `Shape.scalar()`.

With one sentinel:

- require `inputShape.knownElementCount()` to be present, otherwise fail with
  `cannot infer -1 from dynamic input shape <inputShape>`;
- compute the checked product of all non-sentinel dimensions after the full value/sentinel scan;
  if any such dimension is zero, the known product is zero without multiplying other values;
- reject zero known product with
  `cannot infer -1 when known requested dimensions have product zero`;
- require exact divisibility, otherwise fail with
  `cannot infer reshape dimension: input element count <count> is not divisible by known requested product <product>`;
- replace the sentinel in a fresh cloned array with `count / product`, create `Shape.of(copy)`, and
  run common target validation.

Checked multiplication overflow propagates as `ArithmeticException`. Zero-aware product handling
does not report irrelevant overflow when the mathematical product is already zero.

### Exact Shape validation

The Shape overload follows this order:

1. Null-check `input` with message `input`.
2. Null-check `targetShape` with message `targetShape`.
3. Read the exact input descriptor once and its exact Shape once.
4. Call `validateTargetShape(inputShape, targetShape)`.
5. Resolve view layout and perform common construction.

`validateTargetShape` calls `knownElementCount()` once for each Shape. If both are present and
unequal, fail with:

```text
reshape element count mismatch: input=<inputCount>, target=<targetCount>
```

If either count is unknown, accept the target and defer the equality constraint. Element-count
overflow propagates. The exact target Shape reference supplied to the overload is retained in
attributes and result descriptor.

### View-layout derivation

`resolveViewLayout` reads the input descriptor's layout optional exactly once.

Return `Optional.empty()` when:

- target Shape is dynamic;
- input layout is unresolved; or
- resolved input layout is not contiguous, including strided and broadcast geometry.

Otherwise:

1. Create one temporary `LayoutDescriptor.contiguous(targetShape)` to obtain canonical target
   strides.
2. Create the exact stored result with `LayoutDescriptor.of(targetShape, canonical.strides(),
   inputLayout.storageOffset(), true)`.
3. Return that exact new view descriptor in `Optional.of`.

The result is `DENSE_CONTIGUOUS` when offset is zero or `DENSE_WITH_OFFSET` when it is non-zero,
and `isView()` is true in both cases. Scalar and zero-extent targets follow existing layout
geometry rules. No input layout object is reused.

### Common result construction

`create` constructs exactly:

1. one `TensorDescriptor` with input DataType, exact target Shape, supplied layout optional, and
   unchanged input `requiresGrad`;
2. one `TargetShapeAttrs` retaining the exact target Shape;
3. one `Operation(ShapeTransformKind.RESHAPE, attrs)`;
4. one `TensorProvenance(operation, List.of(input))`; and
5. one `TensorFactory.createDerived(descriptor, Optional.empty(), provenance)` call.

Every result is fresh, unlabeled, and storage-free. Input label, provenance, storage, liveness,
values, and bytes are not read or changed. The result's resolved view metadata is a logical fact,
not an attached host-storage alias or an executable guarantee.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorReshapeExpressions.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorReshapeExpressionTest.java`

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
- Shape/Dimension, LayoutDescriptor/LayoutKind, TensorDescriptor, ShapeTransformKind,
  TargetShapeAttrs, TensorFactory, TensorProvenance, Operation, storage/graph contracts, existing
  expression helpers, and their Javadocs/tests
- focused architecture documents, ADRs, architecture tests, backend-conformance material,
  integration tests, Gradle configuration, and every other module

## Maximum scope

At most two production files, two test files, three API/glossary files, and three planning files:
ten paths total.

Do not modify ShapeTransformKind, TargetShapeAttrs, another existing Java source/test,
capabilities, Training API, Gradle, AGENTS, architecture documents/tests, another module, completed
task, or unrelated documentation. Stop if another production type/file, public overload, helper
method, dependency, cross-layer behavior, or eleventh path is needed. Do not create task 0017D1 or
0017E.

## Javadoc requirements

- Update the Tensor type overview to include reshape and distinguish resolved alias-view geometry
  from contiguous expression geometry and unresolved ordinary expression layouts.
- Document both public overloads completely, including array ownership, `-1`, scalar empty request,
  zero extents, dynamic deferral, element-count failures, exact Shape retention, and overflow.
- Document result DataType/Shape/eligibility/layout/provenance/label/storage/freshness for every
  branch.
- Explain that resolved view layout preserves canonical target strides and input offset but does
  not attach storage or promise zero-copy execution.
- Explain why absent/non-contiguous/dynamic geometry remains unresolved and why the model does not
  insert contiguous materialization.
- Document helper class, private constructor, all six methods, parameters, ownership, mutation,
  exact return semantics, validation order, and every failure.
- Document `NullPointerException`, `IllegalArgumentException`, `ArithmeticException`, and
  `IllegalStateException` on the applicable public/helper methods.
- Review ShapeTransformKind, TargetShapeAttrs, Shape, LayoutDescriptor, LayoutKind,
  TensorDescriptor, TensorFactory, TensorProvenance, Operation, and Contiguous Javadocs and record
  why they remain accurate, or stop on inconsistency.

## Acceptance criteria

- Tensor adds exactly two reshape overloads and its declared-public-method count changes from 77
  to 79 without another public API change.
- Each public overload delegates exactly once to the matching helper overload.
- The helper is package-private final/stateless with one private constructor and exactly the six
  specified methods, visibilities, parameters, and return types.
- Null checks, raw-request scan, `-1` inference, count validation, view resolution, and construction
  follow the exact documented order and messages.
- Raw arrays are neither retained nor mutated; exact Shape overload references are retained.
- Empty varargs produce scalar Shape; zero extents are accepted; one inferable `-1` works; invalid
  negatives, duplicate sentinels, dynamic inference, zero-product ambiguity, non-divisibility,
  and known count mismatch fail exactly.
- When either exact Shape count is dynamic, construction succeeds without inventing a binding or
  suppressing later compiler validation.
- All six DataTypes and valid gradient flags are retained unchanged.
- Resolved contiguous input plus static target produces a new canonical-stride, same-offset,
  view-marked resolved layout. Offset zero and non-zero cases classify correctly.
- Unresolved, strided, broadcast, or dynamic cases produce unresolved result layout without
  materialization or layout reuse.
- Exact RESHAPE kind, exact TargetShapeAttrs/target Shape, one-input provenance, fresh ID, absent
  label, and absent storage are recorded.
- Same-shape, repeated, and nested reshapes remain fresh explicit expressions.
- Early validation/layout arithmetic failures consume no Tensor ID; exhaustion propagates only at
  final factory delegation after valid model construction.
- Input descriptor/layout/label/provenance/storage/liveness/values remain unchanged; no eager copy,
  allocation, gradient, compiler, planning, prepare, runtime, backend, graph, ONNX, dependency, or
  build behavior is added.
- Focused/aggregate tests, model Javadoc, root tests, reflection/javap/bytecode/import/source/scope
  checks, documentation examples/links/formatting, and synchronized statuses pass.
- A separate clean-context documentation agent finalizes affected Javadocs, Tensor API, Compile
  API, glossary, task evidence, master plan, and roadmap and records related no-change conclusions.
- Task 0017D becomes Complete only after both passes. Task 0017D1 remains Draft without a detailed
  specification.

## Tests / validation

Run before and after the documentation-focused review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorReshapeExpressionTest --tests io.github.pho001.synaptik.model.tensor.TensorTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused test must cover exact public/helper API; all DataTypes/valid eligibility; scalar empty
request; zero extents; exact static/dynamic Shape overloads; valid `-1` positions; defensive array
behavior; every exact validation message and precedence; checked arithmetic; resolved dense and
offset view layouts; unresolved/strided/broadcast/dynamic layouts; exact operation/attributes/
provenance; absent label/storage; dead-storage non-interference; freshness; no canonicalization;
early failure ID non-consumption; and final identifier exhaustion.

Manually inspect `javap -p -c -s`, reflection, imports, and source. Confirm two one-call public
delegations; exact six-method stateless helper; no hidden overload; one request scan; one count
validation path; one view-layout decision; one descriptor/attributes/Operation/provenance/factory
path; and no value copy, host allocation, storage access, graph/compiler/planning/prepare/runtime/
backend type, gradient rule, registry, cache, service, or hidden materialization. Validate generated
Javadoc, executable documentation examples, Tensor/Compile API status, glossary, links, anchors,
fences, whitespace, exact ten-path scope, synchronized statuses, and absence of a task-0017D1
specification.

## Dependencies

- Task 0002 supplies Shape construction, static/dynamic dimensions, and checked known element
  counts.
- Task 0003 supplies canonical and explicit resolved layout geometry.
- Task 0007 supplies resolved-or-unresolved TensorDescriptor state.
- Tasks 0011–0013 supply public Tensor metadata, centralized derived identity allocation, and
  immutable provenance.
- Task 0017C supplies exact RESHAPE semantics and normalized TargetShapeAttrs.
- Tasks 0017A–0017B establish the adjacent contiguous semantic/materialization boundary used when
  documenting why reshape does not insert contiguous automatically.

## Follow-up tasks

- 0017D1 remains Draft for public expand overloads, right-aligned equal/singleton/leading-axis
  validation, zero-stride alias-view geometry, target Shape retention, and provenance.
- 0017E–0017F remain Draft for permutation, transpose, expand-dimensions, and squeeze.
- Compiler later owns deferred dynamic element-count constraints and reshape-chain canonicalization.
- Planning later derives materialization requirements for unresolved/non-view-compatible reshape.
- Backend prepare later chooses alias/copy/native lowering; runtime executes prepared work.
- Training and compiler-generated semantic tasks later own reshape backward behavior.

Do not create a detailed follow-up specification in this task.

## Architecture impact

Expected impact: None. Model already owns public Tensor expression construction, operation
semantics, Shape, and logical layout descriptors. Compiler owns graph-wide inference and
canonicalization; planning owns logical materialization requirements; backend prepare owns
concrete alias/copy lowering and storage; runtime executes prepared schedules.

If implementation requires dynamic binding, eager storage, materialization policy, cross-module
state, another dependency, or architecture change, stop and report it.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0002/0003/0007/0011/0012/0013/0017A/0017B/0017C/0017D,
Tensor API, Compile API, Training API, glossary, current Shape/Dimension/LayoutDescriptor/
LayoutKind/TensorDescriptor/Tensor/TensorFactory/TensorProvenance/Operation/ShapeTransformKind/
TargetShapeAttrs contracts and focused tests, and Java 26 Gradle configuration.

Implement task 0017D exactly. Modify Tensor.java and add package-private final
TensorReshapeExpressions.java. Update TensorTest only for the exact two-overload API expansion and
add TensorReshapeExpressionTest. Add exactly reshape(long...) and reshape(Shape), each delegating
once to the matching helper overload.

The stateless helper has exactly the six specified methods. Normalize a copied raw request, accept
non-negative dimensions and one -1, infer only from known input count and non-zero checked known
product, validate known counts, and defer unknown dynamic equality. Resolve a new same-offset,
canonical-stride view layout only for resolved contiguous input plus static target; otherwise use
unresolved layout. Retain exact type/target Shape/eligibility, create exact RESHAPE +
TargetShapeAttrs and [input] provenance, and call createDerived once with no label/storage. Every
valid request is fresh.

Do not implement expand, bind symbols, inspect/copy values or storage, force contiguous or
materialization, return input, canonicalize, add APIs/helpers/types, change existing contracts,
define gradients, capture graphs, or add compiler/planning/prepare/runtime/backend behavior,
dependencies, build/architecture changes, or later specs. Stop beyond ten paths or on architecture
uncertainty.

Run every specified focused/aggregate test, Javadoc, javap/reflection/bytecode/import/manual,
documentation/example/link/whitespace/scope/status check. Then hand the actual diff/evidence to a
separate clean-context documentation agent in the same change. It must inspect source/tests/
generated Javadoc, finalize permitted Javadocs/Tensor API/Compile API/glossary/planning, record
related-contract/capability/Training API/architecture no-change conclusions, and rerun validation.

Update task 0017D, model master plan, and roadmap only for planning status/evidence. Do not mark
0017D Complete until both passes succeed. Leave 0017D1 Draft without a specification. Do not
commit or push.
```

## Local decisions

- The former combined reshape/expand expression row is split into 0017D and 0017D1. Each operation
  has independent validation and layout algebra; combining them would exceed one isolated task's
  conceptual and test scope.
- Public raw dimensions use `long...`, matching the model's long static sizes while preserving
  ordinary calls such as `reshape(2, 3)` through primitive widening.
- The exact `Shape` overload exposes normalized dynamic Shapes without inventing a numeric sentinel
  or a second symbolic-shape representation.
- Empty varargs means scalar Shape, consistent with `Shape.of()` and the model's rank-zero scalar.
- One `-1` is request syntax only. Dynamic input cannot infer it because no unique numeric extent
  exists; callers can use the Shape overload for explicit symbolic targets.
- Zero-product inference is rejected as ambiguous, while zero may be inferred when the other
  requested product is non-zero and input count is zero.
- Unknown dynamic count equality is deferred rather than rejected or assumed. Compiler later owns
  the constraint.
- Resolved reshape view geometry is published only when contiguity and static target geometry prove
  it locally. It retains input offset, uses canonical target strides, and marks the new descriptor
  as a view.
- Non-contiguous/unresolved/dynamic cases remain unresolved; reshape does not implicitly request or
  insert materialization.
- Every call is fresh. Same-shape and nested elimination belong to compiler canonicalization.

## Known limitations

- Expand remains unavailable until task 0017D1.
- Dynamic count compatibility is recorded but not solved; invalid bindings must fail in compiler
  inference later.
- Non-contiguous reshape result geometry is unresolved and no executable materialization exists.
- No host-storage alias is attached even when resolved layout describes a logical view.
- No gradient rule, compiler capture, planning requirement, backend lowering, runtime execution,
  ONNX mapping, or conformance behavior is implemented.

## Validation evidence

Planning read the architecture contract and focused lifecycle/module/dependency/runtime-boundary
explanations; documentation and planning rules; roadmap; model capabilities/master plan; tasks
0002, 0003, 0007, 0011, 0012, 0013, 0017A, 0017B, and 0017C; current Shape/Dimension,
LayoutDescriptor/LayoutKind, TensorDescriptor/Tensor/factory/provenance, Operation, and
ShapeTransform semantic source/tests; Tensor/Compile/Training APIs and glossary; and Java 26
Gradle configuration.

The read-only `legacy/pre-rewrite` branch was inspected directly. It confirms public varargs
reshape, one `-1`, count inference and validation, logical order/type preservation, contiguous
view behavior, non-contiguous materialization, gradient reconstruction, and backend execution.
Legacy int limits, positive-only/rank-at-least-one restrictions, immediate storage, graph builders,
operation traits, planners, kernels, lowering, and runtime behavior are excluded or deliberately
adapted to completed new-model contracts.

Planning found reshape and expand expression construction too broad for one task and split the
former row into 0017D reshape and 0017D1 expand. Existing public contracts support both planned
reshape overloads, local count validation, conditional view layout, immutable provenance, and
central identity allocation without a new type, dependency, foundational edit, or architecture
decision.

Planning validation after synchronizing this task, the model master plan, and roadmap:

- `git diff --check` passed.
- The targeted trailing-whitespace scan returned no matches across the three changed planning
  files.
- All 196 local Markdown file links across the three planning files resolve.
- Markdown code-fence counts are balanced: twelve in this task, two in the master plan, and zero
  in the roadmap.
- All 20 canonical task-specification headings are present, together with focused Capability
  origin, Required contract, and Javadoc requirements sections.
- At the planning stage, task, model master plan, and roadmap consistently identified 0017D as
  Ready. Task 0017D1 remained Draft, and no task-0017D1 specification existed.
- Package review confirms no new package. Public entries and the helper remain in `model.tensor`,
  and completed Shape/layout/operation values are consumed through their existing packages.
- Scope review confirms exactly ten permitted implementation paths and exactly three planning
  paths in the current diff. No Java, API, glossary, architecture, Gradle, AGENTS, completed-task,
  or other-module file changed during planning.
- Granularity review split the former combined reshape/expand expression row into 0017D and
  0017D1. Reshape inference/count/view rules and expand broadcast/zero-stride rules now have
  independent implementation and validation sessions.
- Contract review confirms the planned raw `long...` and exact Shape overloads reuse the completed
  long-dimension, scalar, zero-extent, dynamic, known-count, resolved-layout, descriptor,
  provenance, and identity contracts without a new abstraction.
- The model-task sequence was renumbered after inserting 0017D1 and contains no duplicate order
  number within that table.

Implementation and independent documentation validation:

- Implementation context `/root/implement_model_0017d` added the two public overloads, the exact
  six-method package-private helper, and focused tests. Canonical clean documentation context
  `/root/implement_model_0017d/review_model_0017d_docs` independently inspected the actual diff,
  source, tests, generated Javadoc, bytecode, imports, build configuration, and documentation. It
  applied General and API/Javadoc style to Java and API work, Planning style to status/evidence,
  and Example format to the new reshape example.
- Independent source and test inspection found no implementation defect. Raw requests are scanned
  before inference, caller arrays are never retained or mutated, empty varargs produce scalar
  Shape, one `-1` uses a known input count and non-zero checked product, inferred zero is valid,
  zero-product ambiguity is rejected, known mismatches fail, and dynamic equality is deferred.
  The exact Shape overload retains its target reference. All six data types and valid gradient
  choices retain exact metadata.
- Resolved contiguous input plus static target creates one new view-marked layout with canonical
  target strides and the same input element offset. Offset zero and non-zero classify as
  `DENSE_CONTIGUOUS` and `DENSE_WITH_OFFSET`; unresolved, strided, broadcast, or dynamic geometry
  remains unresolved. Construction records exact `RESHAPE`, `TargetShapeAttrs(targetShape)`, and
  `[input]` provenance and attaches no label or storage.
- The documentation pass finalized Tensor API reshape behavior, the Compile API current-versus-
  planned boundary, glossary status and terminology, and synchronized planning. It corrected the
  Tensor type overview to link `ShapeTransformKind`; the submitted public overload and helper
  Javadocs were otherwise complete and required no behavioral-source change.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorReshapeExpressionTest --tests
  io.github.pho001.synaptik.model.tensor.TensorTest` — `BUILD SUCCESSFUL`; 28 tests across two
  suites, specifically 14 reshape tests and 14 Tensor tests, with zero failures, errors, or skips.
- `./gradlew :modules:model:test` — `BUILD SUCCESSFUL`; 62 suites and 484 tests, with zero
  failures, errors, or skips.
- `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL`. Generated `Tensor.html` contains both
  public overloads, their parameters/results/four failure categories, raw ownership and inference,
  dynamic deferral, conditional view geometry, storage/materialization boundaries, and the
  `ShapeTransformKind` type-overview link. Package-private helper Javadocs were reviewed in source.
- `./gradlew test` — `BUILD SUCCESSFUL`; all 36 repository test tasks completed without failure.
  No architecture, backend-conformance, or integration test changed because the task adds only
  model expression metadata and no dependency, backend, or end-to-end behavior.
- `javap -p -c -s` confirms each public overload contains one matching helper call. The final
  stateless helper has one private constructor and exactly six methods; bytecode shows one raw
  validation/inference path, one exact-Shape validation path, one layout decision, and one shared
  descriptor/attributes/Operation/provenance/`createDerived` construction path.
- Import and source scans found only the permitted model/JDK contracts. There is no storage/value
  access, allocation, copy, gradient rule, graph/compiler/planning/prepare/runtime/backend type,
  cache, registry, service, implicit contiguous operation, or hidden canonicalization.
- The Tensor API reshape example compiled with Java 26 against built model classes and printed the
  documented normalized Shape `[3, 2]`, canonical strides `[2, 1]`, offset five, view flag,
  `RESHAPE` kind, input identity, exact dynamic-Shape retention, unresolved dynamic layout, and
  absent storage.
- The targeted Markdown validator resolved 298 local file links and heading anchors across the six
  changed documentation/planning files. Backtick fences are balanced (132 Tensor API, four Compile
  API, twelve task, two master-plan markers; zero in glossary and roadmap), no tilde fence was
  introduced, trailing-whitespace scans found no matches, every changed file ends with a newline,
  and `git diff --check` plus untracked-file whitespace checks passed.
- Exact scope is the authorized ten paths: `Tensor.java`, `TensorReshapeExpressions.java`,
  `TensorTest.java`, `TensorReshapeExpressionTest.java`, Tensor API, Compile API, glossary, this
  task, model master plan, and roadmap. Task 0017D1 remains Draft, and no detailed 0017D1 task
  specification exists. No commit or push occurred.
- Shape and Dimension Javadocs remain accurate because they already define scalar/zero/dynamic
  Shapes, checked known counts, defensive ownership, and exclusion of numeric dynamic sentinels.
  LayoutDescriptor/LayoutKind remain accurate because they already define canonical and offset
  contiguous geometry, independent view metadata, and no storage/materialization policy.
  TensorDescriptor, TensorFactory, TensorProvenance, Operation, ShapeTransformKind,
  TargetShapeAttrs, and Contiguous contracts remain accurate because reshape composes their
  existing immutable descriptor, centralized identity, origin, semantic-pairing, normalized-Shape,
  and materialization boundaries without changing them.
- Training API remains unchanged because no gradient, autograd, parameter, optimizer, publication,
  or session behavior was added. `capabilities.md` already records reshape and the layer split.
  `ARCHITECTURE.md`, focused architecture explanations, ADRs, architecture tests,
  backend-conformance and integration material, Java 26 Gradle configuration, dependencies, other
  modules, and unrelated tests remain accurate because ownership, dependency direction, lifecycle,
  build structure, backend behavior, and executable behavior did not change.

## Implementation notes

- Added exactly `Tensor.reshape(long...)` and `Tensor.reshape(Shape)` as one-call delegations to a
  package-private final stateless helper with the six required methods.
- Added focused coverage for API shape, all data types and valid gradient choices, scalar/zero/
  static/dynamic Shapes, every inference and validation edge, view and unresolved geometry,
  provenance/freshness, storage non-interference, identity consumption, and exhaustion.
- Finalized the Tensor overview and reshape Javadocs, Tensor/Compile APIs, glossary, task evidence,
  model master plan, and roadmap in the mandatory independent documentation context.
- Added no expand behavior, value movement, storage attachment, gradient rule, compiler/planning/
  prepare/runtime/backend behavior, dependency, build change, or architecture change.

## Completion summary

- Completed changes: Implemented and documented raw-inferred and exact-Shape public reshape
  expressions with local count validation, conditional same-offset canonical view geometry, and
  exact one-input provenance.
- Files changed or created: Exactly two production files, two focused test files, Tensor API,
  Compile API, glossary, this task, model master plan, and roadmap.
- Tests and validation: Focused tests passed 28/28; all 484 model tests across 62 suites, model
  Javadoc, root tests, Java 26 example, bytecode/reflection/import/source/generated-Javadoc,
  link/anchor/fence/terminology/whitespace, exact-scope/status, and diff checks passed.
- Documentation-agent review: Canonical clean context
  `/root/implement_model_0017d/review_model_0017d_docs` completed the independent pass using
  General, API/Javadoc, Planning, and Example profiles.
- Documentation impact: Tensor API and glossary now present reshape construction as current;
  Compile API preserves the boundary between current expression metadata and planned capture,
  dynamic constraints, canonicalization, materialization, lowering, and execution.
- Javadoc review: Tensor and helper reshape contracts are complete; the Tensor overview now links
  `ShapeTransformKind`. All reviewed foundational, layout, descriptor, factory, provenance,
  operation, shape-transform, and contiguous contracts remain accurate unchanged.
- Glossary impact: Reshape semantics and Tensor status now cover raw inference, exact Shape
  retention, conditional view metadata, unresolved cases, freshness, and cross-layer boundaries.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0017D. Task 0017D1 remains Draft without a detailed
  specification.

Status: Complete
