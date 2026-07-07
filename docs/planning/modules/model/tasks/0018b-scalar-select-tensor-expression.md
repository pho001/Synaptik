# Task 0018B: Scalar Select Tensor Expression

## Status

Complete

## Goal

Add the public model-level expression that selects one scalar coordinate from one Tensor axis.

The expression normalizes a positive or negative axis, normalizes and checks the scalar index when
the selected extent is statically known, removes the selected axis from the result Shape, derives
view geometry only when it is locally provable, and records exact `SELECT` semantics with
one-input provenance. It constructs immutable expression metadata only: it neither reads values
nor promises that a physical storage alias will be used during execution.

For input Shape `[2, 3, 4]`, `select(1, 2)` produces Shape `[2, 4]`. The same selection can be
written `select(-2, -1)` because the selected extent is statically known as `3`.

## Scope

- Add exactly one public instance method to `Tensor`:
  `Tensor select(int axis, long index)`.
- Add one field-free package-private final `TensorSelectExpressions` helper in `model.tensor`.
- Give the helper exactly one package-private entry and four private methods specified below.
- Null-check the helper input, read its immutable descriptor and Shape, and normalize the raw axis
  exactly once through `Shape.normalizeAxis`.
- For a static selected extent, normalize one negative index by adding the extent once and require
  the normalized coordinate to be in bounds.
- For a dynamic selected extent, accept a non-negative index unchanged with deferred bounds
  validation and reject a negative index because local normalization is impossible.
- Remove the selected Dimension while preserving every unaffected Dimension reference exactly.
- For resolved non-empty input geometry, remove the selected stride and advance the storage offset
  by `normalizedIndex * selectedStride` using checked arithmetic, then construct one new view
  `LayoutDescriptor`.
- Leave result layout unresolved when input layout is unresolved or result element count is zero.
- Preserve exact input data type and gradient eligibility for every current data type.
- Construct exact `SelectKind.SELECT`, `SelectAttrs`, and ordered provenance `[input]` and delegate
  once to `TensorFactory.createDerived` with no label or storage.
- Keep every valid request explicit and fresh, including repeated, nested, same-coordinate, and
  rank-one-to-scalar selection.
- Update `TensorTest` only for the deliberate public API expansion and add one focused expression
  test.
- Finalize Javadocs, Tensor API, Compile API, glossary, task evidence, master plan, and roadmap
  through the mandatory independent documentation pass.

## Out of scope

- another public overload, static facade, `int`-specific overload, default axis/index, index Tensor,
  collection of indices, mask, range, slice, gather, take, or scatter behavior
- accepting a negative index for a dynamic selected extent, inventing a symbolic extent, adding a
  runtime-bound index expression, or silently retaining an unnormalized negative sentinel
- eagerly rejecting every non-negative index on a dynamic extent; its upper-bound validation is
  intentionally deferred until that extent is bound
- changing `SelectKind`, `SelectAttrs`, `Shape`, `Dimension`, `LayoutDescriptor`,
  `TensorDescriptor`, `TensorFactory`, `TensorProvenance`, `Operation`, or any completed Java
  contract/test other than the exact Tensor API inventory
- reading, copying, selecting, or materializing values; attaching or observing host storage;
  asserting a physical alias, zero-copy execution, or backend view support
- returning the input, rewriting select as `SLICE`, `UNSTACK`, or gather, folding nested selects,
  canonicalizing coordinates, or eliminating expressions
- defining select backward/scatter-add, gradient rules, autograd, graph capture, compiler passes,
  planning requirements, prepare, runtime, backend lowering/kernels, engine, trace, ONNX, training,
  or execution behavior
- another production helper/type, dependency, Gradle/build option, architecture change, another
  module, or task-0018C specification

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
- [Task 0002](0002-shape-and-dimension-model.md)
- [Task 0003](0003-layout-descriptor-model.md)
- [Task 0007](0007-tensor-descriptor-model.md)
- [Task 0011](0011-public-tensor-skeleton.md)
- [Task 0012](0012-tensor-factory.md)
- [Task 0013](0013-tensor-provenance-skeleton.md)
- [Task 0015E](0015e-where-selection-semantic-kind.md)
- [Task 0017G](0017g-slice-semantics.md)
- [Task 0017H](0017h-slice-tensor-expressions.md)
- [Task 0017K](0017k-tensor-composition-semantics.md)
- [Task 0017L](0017l-tensor-composition-expressions.md)
- [Task 0018A](0018a-scalar-select-semantics.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The selected baseline requires public scalar-axis selection. The read-only legacy branch exposes
an input, one raw axis, and one raw scalar index. It normalizes both against a concrete static
shape, removes the selected axis, removes its stride, advances the storage offset by index times
that stride, records `SELECT`, and defines backward through scatter-add.

The new model preserves the public forward-expression capability while using current contracts:
long-valued Shape coordinates, explicit static/dynamic Dimension values, immutable optional layout,
typed `SELECT` semantics, and immutable Tensor provenance. It deliberately broadens the locally
representable dynamic case: a non-negative scalar coordinate can be retained when the selected
extent is symbolic because axis removal does not require its size. A negative index still needs a
known extent and is therefore rejected for a dynamic selected Dimension.

Legacy storage aliases, eager values, gradient callbacks, graph builders, operation traits,
compiler/lowering code, kernels, and runtime/backend behavior are capability evidence only and are
not copied.

## Architecture constraints

- `Tensor` remains public mutable API state, not compiled IR. The method constructs a fresh
  storage-free expression through the existing package-private derived-factory seam.
- Semantic identity and normalized attributes come only from completed task 0018A.
- The helper may inspect immutable descriptor, Shape, Dimension, and optional layout metadata, but
  never values, host storage, runtime residency, device state, or backend capability.
- Axis and static-index normalization are local request validation. A dynamic non-negative index
  is normalized already but has deferred upper-bound validation; no hidden symbolic constraint is
  added to Shape.
- Result Shape removal is locally exact even when the selected Dimension is dynamic. Every
  unaffected Dimension reference is retained exactly.
- Resolved layout is logical view metadata only. It neither attaches storage nor promises physical
  aliasing or zero-copy execution.
- A result with zero known elements has unresolved layout because it references no storage element
  and needs no arbitrary offset geometry.
- Compiler owns capture/canonicalization and later dynamic-bound validation placement; compiler-
  generated/training work owns select backward; planning/prepare/backend own materialization,
  lowering, and execution.
- No dependency, package ownership, or module boundary changes are authorized.

## Package impact

No package is added or moved.

- `io.github.pho001.synaptik.model.tensor.Tensor` — receives the one public expression surface.
- `io.github.pho001.synaptik.model.tensor.TensorSelectExpressions` — owns local normalization,
  Shape/layout derivation, and semantic construction.
- `TensorSelectExpressionTest` — mirrors `model.tensor` for focused helper/API validation.
- `TensorTest` — changes only its exact public API inventory and reflection assertions.

`SelectKind` and `SelectAttrs` remain in `model.operation.index`; no semantic contract moves into
the Tensor package.

## Required contract

### Public Tensor method

Add exactly:

```java
public Tensor select(int axis, long index) {
    return TensorSelectExpressions.apply(this, axis, index);
}
```

The method contains one return statement and exactly one helper call. It is non-static,
non-synchronized, performs no direct validation or field access, and adds no overload.

### Helper shape

Create one package-private final, field-free class with one private zero-argument constructor and
exactly these five static methods:

```java
static Tensor apply(Tensor input, int axis, long index)
private static long normalizeIndex(Dimension selected, int normalizedAxis, long rawIndex)
private static Shape removeAxis(Shape inputShape, int normalizedAxis)
private static Optional<LayoutDescriptor> resolveViewLayout(
        TensorDescriptor inputDescriptor,
        Shape resultShape,
        int normalizedAxis,
        long normalizedIndex)
private static Tensor create(
        Tensor input,
        TensorDescriptor inputDescriptor,
        Shape resultShape,
        Optional<LayoutDescriptor> resultLayout,
        SelectAttrs attrs)
```

Add no field, nested type, alternate constructor, overload, cache, mutable state, or extra method.

### Entry validation and construction order

`apply` performs exactly this order:

1. null-check `input` with `Objects.requireNonNull(input, "input")`;
2. read the exact input `TensorDescriptor` once;
3. read its exact `Shape` once;
4. normalize `axis` exactly once through `inputShape.normalizeAxis(axis)`;
5. read the exact selected `Dimension` once;
6. call `normalizeIndex` once;
7. construct one `SelectAttrs(normalizedAxis, normalizedIndex)`;
8. call `removeAxis` once;
9. call `resolveViewLayout` once;
10. call `create` once and return its result.

All local validation, Shape construction, layout arithmetic, descriptor construction, Operation,
and provenance creation occur before identifier allocation. A local failure consumes no Tensor
identifier. Identifier exhaustion occurs only at the final `createDerived` call.

### Axis and index normalization

Axis behavior is exactly the current `Shape.normalizeAxis(int)` contract:

- non-negative axes address the input from the front;
- negative axes add rank once;
- invalid axes, including every axis for a scalar input, throw `IndexOutOfBoundsException` with
  the existing Shape message `Axis <raw> is outside shape rank <rank>`.

`normalizeIndex` distinguishes the selected Dimension category.

For `StaticDimension(size)`:

1. copy `rawIndex` into a local `long normalized`;
2. if negative, add `size` once;
3. if the result is negative or at least `size`, throw `IndexOutOfBoundsException` with exact
   message `select index <rawIndex> is outside axis <normalizedAxis> extent <size>`;
4. return the normalized non-negative coordinate.

This naturally rejects every index for a zero-sized selected extent. Adding a non-negative static
extent once to a negative long cannot overflow.

For a dynamic selected Dimension:

- reject a negative raw index with `IllegalArgumentException` and exact message
  `select index <rawIndex> cannot be normalized against dynamic axis <normalizedAxis>`;
- return a non-negative raw index unchanged without claiming that it is in bounds.

Do not use `-1` or another sentinel in `SelectAttrs`.

### Result Shape

`removeAxis` allocates one `Dimension[]` of length `inputRank - 1`, copies every unaffected exact
Dimension reference in original order, and calls `Shape.ofDimensions` once. Selecting the only
axis of a rank-one Tensor returns canonical scalar Shape. A selected dynamic Dimension disappears;
unselected dynamic Dimensions remain exact references.

Examples:

| Input Shape | Request | Normalized attrs | Result Shape |
|---|---|---|---|
| `[2, 3, 4]` | `select(1, 2)` | axis `1`, index `2` | `[2, 4]` |
| `[2, 3, 4]` | `select(-2, -1)` | axis `1`, index `2` | `[2, 4]` |
| `[N, 4]` | `select(0, 7)` | axis `0`, index `7`, bounds deferred | `[4]` |
| `[5]` | `select(0, 0)` | axis `0`, index `0` | scalar `[]` |

### Result layout

`resolveViewLayout` reads the input layout Optional once.

- If input layout is unresolved, return `Optional.empty()`.
- If the fully static result Shape has known element count zero, return `Optional.empty()`.
- Otherwise copy the input strides once, remove the selected stride while preserving order, and
  calculate the result offset exactly as:

```java
long resultOffset = Math.addExact(
        inputLayout.storageOffset(),
        Math.multiplyExact(normalizedIndex, inputStrides[normalizedAxis]));
```

- Construct exactly one
  `LayoutDescriptor.of(resultShape, resultStrides, resultOffset, true)` and return it present.

Every resolved input layout is accepted, including dense, offset, strided, and zero-stride view
geometry. The descriptor reclassifies the reduced geometry. A dynamic input cannot have resolved
layout under current `TensorDescriptor`, so a dynamic selected extent always produces unresolved
result geometry even if removing that axis makes the result Shape fully static.

For contiguous `[2, 3, 4]` with strides `[12, 4, 1]`, selecting axis `1`, index `2` produces
Shape `[2, 4]`, strides `[12, 1]`, and storage offset `8`, marked as a view. For input `[2, 3, 0]`,
selecting axis `1`, index `1` produces empty Shape `[2, 0]` with unresolved layout.

### Descriptor, semantics, provenance, and identity

`create` constructs exactly:

```java
TensorDescriptor descriptor = new TensorDescriptor(
        inputDescriptor.dataType(),
        resultShape,
        resultLayout,
        inputDescriptor.requiresGrad());
Operation operation = new Operation(SelectKind.SELECT, attrs);
TensorProvenance provenance = new TensorProvenance(operation, List.of(input));
return TensorFactory.createDerived(descriptor, Optional.empty(), provenance);
```

Every current data type is accepted. Exact type and `requiresGrad` are retained. Every valid call
returns a fresh Tensor with no label, no host storage, exact one-input provenance, and the exact
normalized `SelectAttrs` reference. Fresh identity does not imply eager values or execution.

## Affected files

Expected implementation change:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorSelectExpressions.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorSelectExpressionTest.java`
- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- this task file
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum scope

This task may create or modify at most the ten paths listed above.

If implementation requires another production type, semantic-contract change, another test,
capability-baseline edit, architecture document, dependency, build change, another module, or more
than ten paths, stop and propose a follow-up task. Do not trade away validation or documentation
to fit the limit.

## Javadoc requirements

- Document the new public Tensor method and every helper method, including the private constructor.
- Explain axis and index normalization separately, including the dynamic selected-extent policy.
- Include the `[2, 3, 4]` positive and negative request examples with concrete normalized/result
  values.
- Explain exact Shape reference retention and rank-one-to-scalar behavior.
- Explain conditional view geometry with the concrete strides `[12, 4, 1]` to `[12, 1]` and
  offset `8` calculation.
- Explain empty-result and unresolved-layout behavior without promising a physical alias.
- Explain exact type/eligibility retention, one-input provenance, fresh identity, and absence of
  label/storage/value execution.
- Distinguish scalar select from `WHERE`, `UNSTACK`, `SLICE`, and tensor-index gather.
- Do not promise gradient rules, compiler capture, backend support, materialization, or execution.

## Acceptance criteria

- `Tensor` has exactly one new public non-static, non-synchronized
  `select(int, long): Tensor` method and no overload.
- The public method delegates once to the exact helper and performs no other work.
- The helper is package-private, final, field-free, has one private constructor, and exactly the
  five specified static methods.
- Input null, axis normalization, static index normalization/bounds, and dynamic negative-index
  behavior follow the exact order, exception types, and messages.
- Dynamic non-negative selected indices remain representable with unresolved layout and deferred
  upper-bound validation.
- Result Shape removes exactly one axis and preserves every unaffected Dimension reference.
- Resolved non-empty layout removes the selected stride, advances offset with checked arithmetic,
  and is marked as a new view; unresolved and empty cases remain unresolved.
- Exact input data type and gradient eligibility are retained for all six current types.
- Every valid call creates exact `SELECT`/`SelectAttrs`, ordered `[input]` provenance, no label or
  storage, and fresh identity.
- No value access, storage access, gradient rule, graph capture, compiler/runtime/backend behavior,
  semantic-contract modification, dependency, build change, or architecture change is introduced.
- Tensor API, Compile API, glossary, task evidence, master plan, and roadmap are independently
  reviewed and synchronized; Training API and capability/architecture docs record reasoned
  no-change conclusions.
- All validation passes and the final diff contains exactly the ten permitted paths.

## Tests / validation

Run:

```bash
./gradlew :modules:model:test --tests \
  io.github.pho001.synaptik.model.tensor.TensorSelectExpressionTest
./gradlew :modules:model:test --tests \
  io.github.pho001.synaptik.model.tensor.TensorTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused tests must verify:

- exact public Tensor signature, modifiers, single helper delegation, and public API count;
- exact helper package, visibility, final/field-free shape, private constructor, and five methods;
- input-null failure through direct package-private helper invocation;
- positive/negative axis and positive/negative static-index normalization;
- exact static out-of-bounds, zero-extent, dynamic-negative, and scalar-axis failures;
- accepted dynamic non-negative index, selected dynamic-axis removal, unaffected reference
  retention, and unresolved result layout;
- rank-one-to-scalar and all six data types/eligibility values;
- contiguous, offset, strided, and zero-stride resolved input geometry; selected-stride removal,
  checked offset, new view classification, and exact result layout values;
- unresolved input and empty-result unresolved layout;
- exact Operation kind/attributes, normalized values, provenance input reference/order, absent
  label/storage, and fresh identity across repeated/nested calls;
- no eager value/storage access and no compiler/runtime/backend imports.

Manual validation must inspect `javap -p -c -s`, reflection/source/imports, generated Javadoc,
the executable documentation example when changed, Markdown links/anchors/fences/whitespace,
exact ten-path scope, synchronized task/master/roadmap status, and absence of a task-0018C
specification.

## Dependencies

- Task 0002 provides Shape, Dimension, static/dynamic, and axis-normalization contracts.
- Task 0003 provides resolved logical layout geometry and checked descriptor construction.
- Task 0007 provides the resolved/unresolved TensorDescriptor boundary.
- Tasks 0011–0013 provide Tensor, derived identity allocation, and immutable provenance.
- Task 0018A provides exact SELECT semantics and normalized attributes.

## Follow-up tasks

- 0018C: axis gather semantic identities and normalized attributes.
- 0018D: public axis gather/take Tensor expressions.

Do not create either follow-up specification during this task.

## Architecture impact

Expected impact: None.

This task fills the existing model-owned public expression surface. If implementation requires a
new architecture rule, dynamic-shape contract change, runtime service, or cross-module dependency,
stop and report the issue.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0002/0003/0007/0011/0012/0013/0015E/0017G/0017H/0017K/
0017L/0018A/0018B, Tensor API, Compile API, Training API, glossary, current Shape/Dimension/
LayoutDescriptor/TensorDescriptor/Tensor/TensorFactory/TensorProvenance/Operation/SelectKind/
SelectAttrs contracts and focused expression tests, and Java 26 Gradle configuration.

Implement task 0018B exactly. Modify Tensor.java and add package-private final
TensorSelectExpressions.java. Update TensorTest only for the exact one-method API expansion and add
TensorSelectExpressionTest. Add exactly select(int axis, long index), delegating once to the
helper.

The field-free helper has exactly five specified methods. Null-check input, normalize the axis
once, normalize/check a static selected extent, accept a non-negative dynamic index with deferred
bounds, and reject a negative dynamic index with the exact message. Remove the selected axis while
preserving unaffected Dimension references. For resolved non-empty geometry, remove the selected
stride and checked-advance the offset in one new view; unresolved or empty results stay unresolved.
Preserve exact type/eligibility, create exact SELECT/SelectAttrs/[input], and call createDerived
once with no label/storage. Every request is fresh.

Do not modify semantic/foundational contracts, add overloads/types/helpers, inspect/copy values or
storage, attach physical aliases, define gradients, capture/canonicalize graphs, or add compiler/
planning/prepare/runtime/backend behavior, dependencies, build/architecture changes, or later
specs. Stop beyond ten paths or on architecture uncertainty.

Run all specified validation, then hand the actual diff/evidence to a separate clean-context docs
agent in the same change. It must inspect source/tests/generated Javadoc, finalize permitted
Javadocs/Tensor API/Compile API/glossary/planning, record Training API/capability/architecture and
related-contract no-change conclusions, and rerun validation.

Update task 0018B, model master plan, and roadmap only for planning status/evidence. Do not mark
Complete until both passes succeed. Leave 0018C Draft without a specification. Do not commit/push.
```

## Local decisions

- A negative public index is normalized only when the selected Dimension is static. A dynamic
  selected Dimension accepts every non-negative `long` unchanged and defers only its upper-bound
  check; it rejects a negative index rather than storing an unnormalized sentinel.
- Result Shape construction removes exactly one Dimension and retains every unaffected exact
  reference. Rank-one selection uses the canonical scalar Shape returned by
  `Shape.ofDimensions()`.
- Resolved input layout produces resolved result geometry only when the result is non-empty. The
  selected stride is removed and the offset is checked-advanced; an unresolved input or empty
  result stays unresolved.
- Resolved layout is logical view metadata only. The expression remains storage-free and makes no
  physical-alias or zero-copy promise.
- Repeated, nested, and same-coordinate requests remain fresh explicit expressions. Compiler-owned
  capture and canonicalization are not performed by the model helper.

## Known limitations

- A negative index on a dynamic selected extent is intentionally unsupported because no numeric
  extent exists for local normalization. The upper bound of an accepted non-negative dynamic
  index remains deferred.
- The task defines no selected values, physical storage alias, gradient or backward rule, graph
  capture, canonicalization, materialization, backend lowering, ONNX mapping, or execution.

## Validation evidence

- Clean implementation context `/root/implement_model_0018b` added the exact public method,
  field-free five-method helper, Tensor API inventory expansion, and focused ten-test suite before
  handing the actual shared-tree diff to independent documentation context
  `/root/implement_model_0018b/review_model_0018b_docs`.
- The documentation context applied General plus API/Javadoc style to production Javadocs, Tensor
  API, Compile API, and glossary; Planning style to this task, the model master plan, and roadmap;
  and Example format to the executable Tensor API example. It inspected the final source, tests,
  generated Javadoc, and actual diff rather than relying on the implementation summary.
- Reviewed architecture and process material: `AGENTS.md`, `ARCHITECTURE.md`, current focused
  architecture navigation/boundaries, documentation rules and General/API-Javadoc/Planning/
  Example profiles, planning guide and roadmap, model capabilities/master plan, and tasks 0002,
  0003, 0007, 0011, 0012, 0013, 0015E, 0017G, 0017H, 0017K, 0017L, 0018A, and 0018B.
- Reviewed API and implementation material: Tensor, Compile, and Training API references; glossary;
  final `Tensor`, `TensorSelectExpressions`, `TensorTest`, and `TensorSelectExpressionTest`;
  Shape/Dimension, LayoutDescriptor, TensorDescriptor, TensorFactory, TensorProvenance, Operation,
  SelectKind, and SelectAttrs contracts; related WHERE, SLICE, and UNSTACK contracts/tests; model
  generated Javadoc; and Java 26 root/model Gradle configuration.
- The Javadoc review corrected the distinction between resolved input geometry and a non-empty
  result and completed result-element-count overflow wording. The public method and helper type,
  private constructor, and all five methods document parameters, returns, failures, axis/index
  normalization, exact Shape references, scalar output, conditional logical view geometry,
  provenance, freshness, and cross-layer exclusions.
- Tensor API now documents scalar SELECT versus WHERE/UNSTACK/SLICE/tensor-index gather; raw and
  normalized axes/indexes; static versus dynamic selected extents; exact unaffected Dimension
  references; rank-one scalar output; `[2, 3, 4]` selection; `[12, 4, 1]` to `[12, 1]` strides and
  offset `8`; empty/unresolved layout; logical-view versus physical-alias status; exact result
  metadata/provenance/freshness; and value/gradient/compiler/backend/execution boundaries.
- Compile API now lists scalar select as current model expression construction while graph capture,
  canonicalization, dynamic upper-bound validation placement, materialization, gradients,
  lowering, and execution remain planned. Glossary wording now makes the same reusable semantic,
  expression, and ownership distinctions.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorSelectExpressionTest` — `BUILD SUCCESSFUL`; the XML
  report contains 10 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorTest` — `BUILD SUCCESSFUL`; the XML report contains
  14 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:test` — `BUILD SUCCESSFUL`; 76 XML suites contain 648 tests with zero
  failures, errors, or skips.
- `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL`. Generated `Tensor.html` contains the
  exact `select(int, long)` signature, positive/negative normalization example, dynamic selected-
  extent policy, Shape/view result contract, complete parameter/return/failure documentation,
  exact SELECT semantics, and physical-alias/compiler/backend/execution boundaries. The package-
  private helper Javadocs were reviewed in source because standard public Javadoc omits it.
- `./gradlew test` — `BUILD SUCCESSFUL`; all 36 actionable tasks completed without a failing task
  in the final repository lifecycle run.
- `javap -classpath modules/model/build/classes/java/main -p -c -s` for `Tensor` and
  `TensorSelectExpressions`, focused reflection tests, and source inspection confirm exactly 97
  declared public Tensor methods, the one exact public one-call delegation, a private zero-argument
  helper constructor, exactly five static helper methods and no fields/nested types, one Shape axis
  normalization, static/dynamic index branches, exact Shape/layout construction, checked offset,
  exact descriptor/SELECT/attrs/`[input]` construction, and one final `createDerived` call.
- Production import inspection found only model-owned semantic/layout/shape/tensor contracts and
  JDK collections/null checking. Source, bytecode, and focused tests show no compiler, planning,
  prepare, runtime, backend, engine, trace, training, value-read, or storage-read behavior.
- The documented Java 26 `ScalarSelectExpressionExample` compiled with
  `javac -cp modules/model/build/classes/java/main -d /tmp/synaptik-select-doc-example
  /tmp/ScalarSelectExpressionExample.java` and ran with the model classes. It printed exact Shape
  `[2, 4]`, `SelectAttrs[axis=1, index=2]`, strides `[12, 1]`, offset `8`, view/provenance/
  eligibility/storage facts exactly as documented.
- The targeted Markdown path-and-heading validator resolved all 388 local links, including 110
  heading anchors, across the six changed documentation/planning files. Code fences are balanced,
  trailing-whitespace scans found no matches, and all ten authorized paths have final newlines.
- Final changed-path inventory contains exactly the ten authorized paths: Tensor and its select
  helper, TensorTest and the focused select test, Tensor API, Compile API, glossary, this task,
  model master plan, and roadmap. Task/master-plan/roadmap status is synchronized as Complete;
  task 0018C remains Draft and no task-0018C specification exists. `git diff --check` passes.
- Training API remains accurate unchanged because the task adds no gradient object/rule, autograd,
  parameter, optimizer, session, publication, or training execution behavior. Capabilities remain
  accurate unchanged because they already list scalar-index select and distinguish model/public-
  API support from compiler, planning, backend-prepare, runtime, and end-to-end support.
- Shape/Dimension, LayoutDescriptor, TensorDescriptor, TensorFactory, TensorProvenance, Operation,
  SelectKind, SelectAttrs, and WHERE/SLICE/UNSTACK contracts remain accurate unchanged because
  this task composes their existing normalization, geometry, identity, semantic, and provenance
  rules without changing them. Completed task specifications remain historical evidence.
- `ARCHITECTURE.md`, focused architecture explanations, ADRs, architecture tests, backend-
  conformance tests, integration tests, Gradle/dependencies, and other modules remain unchanged
  because the task stays inside existing model ownership and adds no dependency rule, backend
  behavior, executable end-to-end behavior, or build requirement.

## Implementation notes

- Added exactly `Tensor.select(int axis, long index)` as one helper delegation.
- Added package-private final, field-free `TensorSelectExpressions` with the exact five methods for
  deterministic validation, normalization, Shape/layout derivation, and fresh storage-free SELECT
  provenance.
- Expanded the exact Tensor reflection inventory and added ten focused tests covering API/helper
  shape, validation/messages, static/dynamic normalization, Shape references, all data types and
  valid eligibility choices, resolved/unresolved/empty layout, provenance/freshness, storage
  noninterference, and identity timing.
- Finalized production/helper Javadocs, Tensor and Compile API references, glossary terminology,
  executable example, and synchronized planning status/evidence without architecture changes.

## Completion summary

- Completed changes: Implemented and documented public scalar-index selection with static/dynamic
  normalization, exact axis removal, conditional checked logical-view geometry, and fresh exact
  SELECT provenance.
- Files changed or created: Exactly the ten authorized production, test, API, glossary, task,
  master-plan, and roadmap paths.
- Tests and validation: Focused 10-test and 14-test suites, all 648 model tests across 76 suites,
  model Javadoc, root tests, javap/reflection/import/source/generated-page review, executable Java
  26 example, 388 link/110-anchor checks, fence/whitespace/newline checks, exact scope/status and
  no-0018C-spec checks, and `git diff --check` passed.
- Documentation-agent review: Clean context
  `/root/implement_model_0018b/review_model_0018b_docs` completed the required independent pass
  with General, API/Javadoc, Planning, and Example profiles.
- Documentation impact: Tensor API, Compile API, and glossary now describe current scalar-select
  expression construction while gradients, compiler behavior, physical materialization, backend
  behavior, and execution remain planned or separately owned.
- Javadoc review: Public Tensor method and helper type/constructor/five methods are final; related
  semantic, Shape/layout, descriptor/factory/provenance/operation, and adjacent expression
  contracts remain accurate unchanged for the recorded reasons.
- Glossary impact: Scalar-select terminology now distinguishes normalized semantic attributes,
  public input-aware expression construction, logical view metadata, and physical/executable
  behavior.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0018B. Task 0018C remains Draft without a detailed
  specification.

Status: Complete
