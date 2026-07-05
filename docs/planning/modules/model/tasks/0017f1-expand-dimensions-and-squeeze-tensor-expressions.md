# Task 0017F1: Expand-Dimensions and Squeeze Tensor Expressions

## Status

Complete

## Goal

Add public, storage-free Tensor expressions that insert one singleton axis or remove one statically
known singleton axis using the completed `EXPAND_DIMS` and `SQUEEZE` semantics.

Both transformations preserve logical values and the exact order of unaffected dimensions.
Resolved input geometry produces a new same-offset view layout with one stride inserted or removed;
unresolved geometry remains unresolved. This task constructs model metadata only and does not
attach storage, define gradients, capture a graph, choose materialization, lower, or execute work.

## Scope

- Add exactly `Tensor.expandDims(int axis)` and `Tensor.squeeze(int axis)`.
- Make each public method delegate exactly once to its matching package-private helper entry.
- Add one package-private final field-free `TensorRankEditingExpressions` helper with exactly the
  nine methods specified under Required contract and one private constructor.
- Normalize expand-dimensions insertion axes over the output positions `[0, inputRank]`, accepting
  negative positions relative to `inputRank + 1`.
- Normalize squeeze axes through the existing input-Shape axis contract.
- Require the selected squeeze dimension to be statically known with extent exactly one.
- Reject a dynamic selected squeeze dimension because singleton compatibility is not locally
  provable.
- Build inserted result Shape with one new `StaticDimension(1)` and exact unaffected input
  Dimension references.
- Build squeezed result Shape by removing one Dimension and preserving exact unaffected references.
- For resolved input layout, insert one deterministic stride for expand-dimensions while preserving
  existing strides and offset.
- For resolved input layout, remove the selected stride for squeeze while preserving remaining
  strides and offset.
- Support every resolved input layout kind and leave layout unresolved when input layout is absent.
- Preserve exact input data type and gradient eligibility.
- Construct exact `AxisTransformKind.EXPAND_DIMS` or `SQUEEZE`,
  `AxisTransformAttrs(normalizedAxis)`, and ordered provenance `[input]`.
- Call `TensorFactory.createDerived` once with no label or host storage.
- Return a fresh Tensor for every valid request, including inverse-like, repeated, and nested calls.
- Add one focused same-package test and update Tensor's exact public method inventory.
- Finalize Javadocs, Tensor API, Compile API, glossary, task evidence, master plan, and roadmap
  through the mandatory independent documentation pass.

## Out of scope

- permute/transpose changes, permutation attributes, another operation kind/attributes type, or
  modification of completed task 0017F
- automatic squeeze of all singleton axes, multi-axis insertion/removal, axis arrays/lists,
  `unsqueeze` alias, variadic methods, or another public Tensor overload
- squeezing a dynamic dimension by assumption, binding a symbol to one, graph-wide constraint
  solving, or runtime-dependent result rank
- changing Shape, Dimension, StaticDimension, DynamicDimension, LayoutDescriptor, LayoutKind,
  TensorDescriptor, TensorFactory, TensorProvenance, AxisTransformKind, AxisTransformAttrs,
  Operation, or their completed tests
- reading/copying values, storage lookup/attachment, physical aliasing, allocation, materialization,
  mutation, capacity/lifetime checks, or zero-copy execution guarantees
- returning input, cancelling expand/squeeze pairs, canonicalizing nested edits, caching, interning,
  or CSE
- gradients, inverse rank-edit operations, autograd, training, graph capture, compiler passes,
  planning, prepare, backend lowering, runtime, execution, ONNX mapping, or conformance
- another module, dependency, Gradle/build change, preview feature, architecture change, or task-
  0017G implementation/specification

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
- [Task 0017E](0017e-axis-transform-semantics.md)
- [Task 0017F](0017f-permute-and-transpose-tensor-expressions.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only `legacy/pre-rewrite` branch exposes `Tensor.expandDims(int)` and
`Tensor.squeeze(int)`. Insertion accepts positions through rank inclusive, normalizes negative axes
against rank plus one, inserts static extent one, and derives a view stride from the following input
axis or uses one at the end. Squeeze normalizes an existing axis, requires extent one, removes its
Shape/stride entries, and preserves storage offset. Legacy gradients use the inverse rank edit.

This task preserves model-level request normalization, Shape/layout metadata, exact operation
attributes, data-type/eligibility retention, and provenance. It excludes immediate storage aliasing,
mutable graph builders, gradient callbacks, operation traits, compiler/lowering, kernels, runtime,
and backend behavior.

## Architecture constraints

- Tensor remains public mutable API state and is not an IR node.
- Operation stores backend-neutral meaning and no support, cost, route, or executable state.
- Shape and LayoutDescriptor are logical immutable metadata; resolved view geometry neither
  attaches physical storage nor guarantees zero-copy execution.
- Model validates only locally provable insertion/removal facts.
- Dynamic singleton binding and graph-wide constraints belong to compiler shape inference.
- Compiler owns graph capture/canonicalization; planning/backend prepare own materialization and
  concrete lowering; runtime executes prepared schedules.
- No service locator, backend lookup, new dependency, or architecture change is authorized.

## Package impact

No package is added or moved.

- Public methods remain in `io.github.pho001.synaptik.model.tensor.Tensor`.
- Package-private `TensorRankEditingExpressions` lives in `model.tensor` beside other expression
  helpers.
- Focused `TensorRankEditingExpressionTest` mirrors that package.
- Existing shape, layout, operation-layout, descriptor, provenance, and factory packages are only
  consumed and remain unchanged.

## Required contract

### Public Tensor surface

Add exactly:

```java
public Tensor expandDims(int axis) {
    return TensorRankEditingExpressions.expandDims(this, axis);
}

public Tensor squeeze(int axis) {
    return TensorRankEditingExpressions.squeeze(this, axis);
}
```

Do not add aliases, overloads, arrays, collections, or automatic all-axis squeeze.

### Helper structure

`TensorRankEditingExpressions` is package-private, final, field-free, and non-instantiable. Apart
from one private zero-argument constructor, it declares exactly:

```java
static Tensor expandDims(Tensor input, int axis)
static Tensor squeeze(Tensor input, int axis)
private static int normalizeInsertionAxis(int axis, int rank)
private static void validateSqueezableDimension(
        Shape inputShape, int normalizedAxis)
private static Shape insertSingleton(Shape inputShape, int normalizedAxis)
private static Shape removeSingleton(Shape inputShape, int normalizedAxis)
private static Optional<LayoutDescriptor> resolveInsertedLayout(
        TensorDescriptor inputDescriptor,
        Shape inputShape,
        Shape resultShape,
        int normalizedAxis)
private static Optional<LayoutDescriptor> resolveSqueezedLayout(
        TensorDescriptor inputDescriptor, Shape resultShape, int normalizedAxis)
private static Tensor create(
        Tensor input,
        TensorDescriptor inputDescriptor,
        Shape resultShape,
        Optional<LayoutDescriptor> resultLayout,
        AxisTransformKind kind,
        int normalizedAxis)
```

Do not add fields, nested types, another constructor, helper file, or hidden overload.

### Expand-dimensions validation and order

`expandDims` performs exactly:

1. null-check input with exact message `input`;
2. read exact input descriptor once and exact Shape once;
3. normalize insertion axis;
4. derive result Shape;
5. resolve or defer inserted layout;
6. construct the result once with `EXPAND_DIMS`.

Normalize in `long`. If raw axis is negative, add `rank + 1` once; otherwise retain it. Valid
normalized positions are inclusive `[0, rank]`. Invalid values throw `IndexOutOfBoundsException`
with exact message:

```text
Axis <axis> is outside insertion range for shape rank <rank>
```

For rank two, valid raw axes are `-3` through `2`; `-3` and `0` select the start, while `-1` and
`2` select the end. For scalar rank zero, both `-1` and `0` select its only insertion position.

### Squeeze validation and order

`squeeze` performs exactly:

1. null-check input with exact message `input`;
2. read exact input descriptor once and exact Shape once;
3. normalize through `inputShape.normalizeAxis(axis)` exactly once;
4. validate the selected Dimension as a static singleton;
5. derive result Shape;
6. resolve or defer squeezed layout;
7. construct the result once with `SQUEEZE`.

Invalid existing axes retain Shape's `IndexOutOfBoundsException` and exact message
`Axis <axis> is outside shape rank <rank>`. A selected dimension other than
`StaticDimension(1)`, including zero, another static extent, or DynamicDimension, throws
`IllegalArgumentException` with exact message:

```text
cannot squeeze axis <normalizedAxis> of <inputShape>: dimension must be statically known as 1
```

Scalar squeeze always fails axis normalization. Do not assume a dynamic symbol equals one.

### Result Shapes

Insertion allocates one `Dimension[]` of length `inputRank + 1`, inserts exactly one new
`StaticDimension(1)` at normalizedAxis, and preserves exact input Dimension references before and
after it. Removal allocates one array of length `inputRank - 1`, skips the selected dimension, and
preserves every unaffected exact reference. Construct both through `Shape.ofDimensions`.

Rank-one squeeze may produce canonical scalar Shape. Do not mutate or retain arrays.

### Inserted layout

Read `inputDescriptor.layout()` exactly once. If absent, return `Optional.empty()`.

For a present layout, allocate one result-rank stride array. Copy exact input strides around the
inserted position. The inserted stride is:

```text
1                                                     when normalizedAxis == inputRank
Math.multiplyExact(inputLayout.stride(normalizedAxis),
                   inputShape dimension size)         otherwise
```

The input Shape is fully static when layout is present. The multiplication uses checked long
arithmetic and may fail before Tensor identity allocation. This deterministic singleton stride
preserves canonical dense classification when the input is canonical while remaining address-
equivalent for any resolved strided or broadcast view.

Create one `LayoutDescriptor.of(resultShape, resultStrides, inputOffset, true)`. Preserve offset,
all existing raw strides, and explicit view metadata. Let LayoutDescriptor derive kind/span.

### Squeezed layout

Read `inputDescriptor.layout()` exactly once. If absent, return `Optional.empty()`.

For a present layout, allocate one result-rank stride array, omit the selected stride, and preserve
all other exact strides in order. Create one
`LayoutDescriptor.of(resultShape, resultStrides, inputOffset, true)`. Preserve exact offset and let
LayoutDescriptor derive kind/span. Never reuse input layout.

### Result construction

Create exactly:

```java
AxisTransformAttrs attrs = new AxisTransformAttrs(normalizedAxis);
TensorDescriptor descriptor = new TensorDescriptor(
        inputDescriptor.dataType(),
        resultShape,
        resultLayout,
        inputDescriptor.requiresGrad());
Operation operation = new Operation(kind, attrs);
TensorProvenance provenance = new TensorProvenance(operation, List.of(input));
return TensorFactory.createDerived(descriptor, Optional.empty(), provenance);
```

`kind` must be exactly EXPAND_DIMS or SQUEEZE according to the public entry. Every successful call
consumes one fresh Tensor ID and has absent label/storage. Failures before `createDerived` consume no
ID; identifier exhaustion occurs only at final delegation.

## Affected files

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorRankEditingExpressions.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorRankEditingExpressionTest.java`
- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/tasks/0017f1-expand-dimensions-and-squeeze-tensor-expressions.md`
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum scope

At most these ten paths may change. Production is limited to Tensor and one package-private helper;
tests to Tensor inventory and one focused suite; documentation to public API/status and planning.

If another type, helper, overload, existing contract edit, dependency, build/architecture change,
or eleventh path is required, stop and report. Do not create task 0017G.

## Javadoc requirements

- Fully document both public methods and the helper type, constructor, and all nine methods.
- Explain insertion-axis versus existing-axis normalization, including negative boundaries and
  scalar cases.
- Explain static singleton proof and why dynamic selected dimensions are rejected.
- Document exact Dimension-reference preservation and rank changes.
- Explain inserted/removed stride geometry, checked multiplication, offset/view retention, all
  resolved layout kinds, and unresolved cases.
- Document result type, Shape, layout, eligibility, label, storage, operation, attributes,
  provenance, freshness, every failure, and identifier exhaustion.
- Explain that view metadata is not attached storage or execution proof.
- Independently review AxisTransformKind/Attrs, Shape/Dimension, layout, descriptor, factory,
  provenance, and neighboring view-expression Javadocs; stop on an out-of-scope discrepancy.

## Acceptance criteria

- Tensor exposes exactly `expandDims(int)` and `squeeze(int)`; public method inventory rises from
  83 to 85.
- Each method delegates once; helper has exact final/package-private/field-free/nine-method shape.
- Insertion and squeeze validation order, normalization, exception types/messages, scalar and
  negative boundaries match the specification.
- Squeeze accepts only exact static singleton and rejects zero/non-one/dynamic dimensions.
- Result Shapes preserve exact unaffected Dimension references and insert/remove exactly one axis.
- Every resolved layout kind derives exact inserted/removed strides, offset, view flag, kind, and
  span; unresolved input remains unresolved.
- Checked inserted-stride overflow fails before ID consumption.
- All six data types and valid eligibility states retain exact metadata.
- Results have exact kind/AxisTransformAttrs/[input], absent label/storage, and fresh identity.
- Repeated, nested, and inverse-like calls remain explicit and uncanonicalized.
- No values/storage/cross-layer behavior, dependency, or architecture change.
- Independent documentation review and all validation/status synchronization complete before
  marking Complete.

## Tests / validation

Run before and after documentation review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorRankEditingExpressionTest --tests io.github.pho001.synaptik.model.tensor.TensorTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

Focused tests cover exact public/helper API; all data types/eligibility; insertion positive/
negative/boundary/scalar axes; squeeze positive/negative/rank-one/scalar cases; static singleton,
zero/non-one/dynamic rejection; exact Dimension identity; dense/offset/strided/broadcast layouts;
inserted/removed strides, offsets, kinds, view flags, spans; unresolved geometry; checked overflow;
operation/attributes/provenance; absent label/storage; dead-storage non-interference; freshness; ID
side effects; and exhaustion.

Inspect `javap -p -c -s`, reflection, imports, and source. Confirm two one-call public methods,
exact nine-method helper, one insertion normalization, one squeeze normalization/proof, two Shape
paths, two layout paths, one shared construction, and no permute/transpose/value/storage/cross-layer
behavior. Validate generated Javadoc, executable examples, Tensor/Compile API, glossary, links/
anchors/fences/whitespace, exact ten paths, synchronized status, and no task-0017G spec.

## Dependencies

- 0002 supplies immutable Dimension/Shape, axis normalization, scalar/zero/dynamic forms.
- 0003 supplies resolved stride/offset/view geometry and classification.
- 0007 supplies resolved-or-unresolved TensorDescriptor.
- 0011–0013 supply Tensor, centralized derived identity, and provenance.
- 0017E supplies EXPAND_DIMS/SQUEEZE and AxisTransformAttrs.
- 0017F provides adjacent axis-transform expression patterns but no hard production dependency.

## Follow-up tasks

- 0017G remains Draft for slice semantics.
- Compiler later owns inverse-pair/nested rank-edit canonicalization and dynamic constraints.
- Planning/backend prepare later own materialization and view/copy lowering.
- Training/compiler-generated semantics later own inverse rank-edit gradients.

Do not create a detailed follow-up specification here.

## Architecture impact

Expected impact: None. Model owns every composed contract; compiler, planning, prepare, backend,
runtime, and training ownership remains unchanged.

Stop if implementation needs symbolic binding, storage aliasing, gradient rules, cross-layer
behavior, another dependency, or architecture change.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0002/0003/0007/0011/0012/0013/0017E/0017F/0017F1, Tensor
API, Compile API, Training API, glossary, current Dimension/Shape/LayoutDescriptor/LayoutKind/
TensorDescriptor/Tensor/TensorFactory/TensorProvenance/Operation/AxisTransformKind/
AxisTransformAttrs and focused expression contracts/tests, and Java 26 Gradle configuration.

Implement task 0017F1 exactly. Modify Tensor.java and add package-private final
TensorRankEditingExpressions.java. Update TensorTest only for exact two-method API expansion and add
TensorRankEditingExpressionTest. Add exactly expandDims(int) and squeeze(int).

The field-free helper has exactly nine specified methods. Normalize insertion axes against rank+1;
normalize squeeze through Shape and accept only a statically known singleton. Preserve exact
unaffected Dimension references. For any resolved layout, insert the checked following-axis
stride (or one at the end) or remove the selected stride, preserving offset in a new view;
unresolved input stays unresolved. Preserve type/eligibility, create exact kind/
AxisTransformAttrs/[input], and call createDerived once with no label/storage. Every call is fresh.

Do not modify semantic/foundational/completed permutation contracts, assume dynamic singleton,
inspect/copy values or storage, attach aliases, canonicalize, add APIs/helpers/types, define
gradients, capture graphs, or add cross-layer behavior, dependencies, build/architecture changes,
or later specs. Stop beyond ten paths or on architecture uncertainty.

Run all specified validation, then hand actual diff/evidence to a separate clean-context docs agent
in the same change. It must inspect source/tests/generated Javadoc, finalize permitted Javadocs/
Tensor API/Compile API/glossary/planning, record no-change conclusions, and rerun validation.

Update task 0017F1, model master plan, and roadmap only for planning status/evidence. Do not mark
Complete until both passes succeed. Leave 0017G Draft without a specification. Do not commit/push.
```

## Local decisions

- The former combined expression frontier remains split because singleton rank editing has
  different normalization and layout algebra from permutation.
- Expand-dimensions negative axes normalize against rank plus one, so `-1` inserts at the end.
- Squeeze uses Shape's established existing-axis normalization and requires local static proof of
  extent one; dynamic dimensions are not guessed or constrained here.
- Insertion stride follows legacy-compatible deterministic view geometry: following stride times
  following extent, or one at the end, with checked arithmetic.
- Squeeze removes only the selected stride. Both paths preserve offset and mark resolved results as
  logical views without storage attachment or execution guarantees.
- Every valid call is fresh; compiler owns inverse-pair and nested canonicalization.

## Known limitations

- Dynamic dimensions cannot be squeezed even if a future binding could equal one.
- No all-axis squeeze, multi-axis edit, host-storage alias, gradient inverse, compiler capture,
  materialization, backend lowering, runtime execution, ONNX mapping, or conformance behavior.

## Validation evidence

Planning reviewed architecture/focused boundary docs; documentation/planning rules; roadmap; model
capabilities/master plan; prerequisite tasks; current Dimension/Shape, layout, descriptor,
Tensor/factory/provenance, Operation, AxisTransform semantics, completed permutation and neighboring
view-expression source/tests; Tensor/Compile/Training APIs, glossary, and Java 26 Gradle.

The read-only legacy branch confirms insertion-axis normalization against rank plus one, selected-
axis squeeze, singleton proof, Shape/stride insertion/removal, offset-preserving views, and inverse
gradient capability. Coupled storage, graph, gradient, compiler, kernel, runtime, and backend design
is excluded.

Existing contracts support the exact methods/helper without another public type, package,
dependency, foundational edit, or architecture decision. Initial planning validation is recorded
below; implementation and independent documentation evidence follow it.

Planning validation after synchronization:

- `git diff --check` passed, and targeted trailing-whitespace scans found no matches.
- Exact planning scope is three paths: this task, model master plan, and roadmap. No Java, Gradle,
  architecture, API, glossary, completed-task, or other-module file changed during planning.
- All 20 canonical task sections are present.
- Markdown backtick fences are balanced: sixteen in this task, two in the master plan, and zero in
  the roadmap.
- All 211 local Markdown file links across the three changed planning files resolve.
- Every changed file ends with a newline.
- At initial planning time, task, master plan, and roadmap consistently queued 0017F1 for execution
  while leaving 0017G Draft; no detailed 0017G specification existed or has since been created.
- The model task sequence contains 74 ordered rows with no duplicate order number.
- Package review confirms no new package: public methods and helper stay in `model.tensor`, and the
  focused test mirrors that package.
- Scope review confirms exactly ten permitted implementation paths and no need for another public
  type, dependency, foundational edit, or architecture decision.
- Granularity review confirms singleton insertion/removal is one cohesive rank-editing task and
  remains separate from completed permutation and future slice semantics.

## Implementation notes

- `Tensor` now exposes exactly `expandDims(int)` and `squeeze(int)`, each as one direct delegation
  to the matching package-private helper entry.
- New final, field-free `TensorRankEditingExpressions` has one private constructor and exactly the
  specified nine methods. It performs one insertion normalization or one existing-axis
  normalization/static-singleton proof, builds one rank-edited Shape, conditionally builds one
  same-offset logical view layout, and delegates once to shared derived construction.
- Inserted Shapes retain every unaffected Dimension reference and add one new
  `StaticDimension(1)`. Squeezed Shapes retain every unaffected reference and reject zero,
  non-one, and dynamic selected dimensions. Resolved scalar, empty, dense, offset, strided, and
  broadcast geometry follows the required checked stride rules; unresolved geometry remains
  unresolved.
- Both paths retain exact input data type and gradient eligibility and record exact
  `AxisTransformKind`, `AxisTransformAttrs(normalizedAxis)`, and ordered `[input]` provenance.
  Every valid result is fresh, unlabeled, and storage-free.
- `TensorTest` now fixes the public method inventory at 85. The focused rank-editing suite covers
  both normalization domains, scalar and empty geometry, all data types/eligibility, exact
  Dimension identity, every resolved layout category, unresolved layout, overflow and validation
  side effects, provenance, freshness, storage non-interference, and identifier exhaustion.
- The independent documentation context was
  `/root/implement_model_0017f1/review_model_0017f1_docs`. It applied the General,
  API/Javadoc, Planning, and Example profiles; independently reviewed final source, tests,
  generated Javadoc, neighboring contracts, and the exact diff; and retained the complete
  production Javadocs unchanged. It finalized the Tensor API, Compile API, glossary, task, master
  plan, and roadmap, including one executable Java 26 rank-editing example.
- No Training API or capability-baseline update was needed: rank editing changes neither training
  state nor the already listed model capability. No semantic/foundational/completed permutation
  contract changed because this task composes them through separate rank-editing algebra. No
  architecture document, ADR, architecture test, backend-conformance test, integration test,
  Gradle file, dependency, or other module changed because no boundary, dependency, executable,
  backend, or end-to-end behavior changed.

Final validation evidence:

- The focused Gradle command passed 11 `TensorRankEditingExpressionTest` and 14 `TensorTest` cases
  with zero failures, errors, or skips. Full model tests passed 529 cases with zero failures,
  errors, or skips.
- `./gradlew :modules:model:javadoc` and `./gradlew test` passed; the root run reported 36
  actionable tasks. Generated Javadoc contains both public methods and their complete result,
  failure, axis, layout, storage, provenance, and freshness contracts.
- The documented Java 26 `RankEditingExpressionExample` compiled and ran. It printed input Shape
  `[2, 1, 3]`, expanded Shape `[2, 1, 3, 1]`, normalized insertion axis `3`, squeezed Shape
  `[2, 3]`, normalized removal axis `1`, exact kinds, unresolved layouts, and absent storage.
- `javap -p -c -s`, reflection tests, import scans, and source inspection confirm two one-call
  public delegates, exactly 85 public Tensor methods, the final/package-private/field-free helper
  with one private constructor and exactly nine methods, one normalization/proof path per
  operation, two Shape paths, two layout paths, and one shared construction path.
- Local Markdown links and anchors, balanced fences, trailing whitespace, final newlines,
  generated Javadoc text, and `git diff --check` passed. Exactly the ten permitted paths changed.
  Task/master/roadmap status is synchronized at Complete, 0017G remains Draft, and no 0017G task
  specification exists.

## Completion summary

- Completed changes: Added exact singleton-axis insertion/removal Tensor expressions, their bounded
  helper, focused tests, the 85-method public inventory, and synchronized status/evidence.
- Files changed/created: Exactly the ten paths listed under Affected files.
- Tests/validation: Focused 25-test selection, 529-test model suite, model Javadoc, root test
  lifecycle, Java 26 example, bytecode/reflection/import/source checks, 264 local links/anchors,
  fences/whitespace/newlines, exact scope/status, and `git diff --check` passed.
- Documentation-agent review: Clean context
  `/root/implement_model_0017f1/review_model_0017f1_docs` independently reviewed final source,
  tests, generated Javadoc, related contracts, and the actual diff under the General,
  API/Javadoc, Planning, and Example profiles.
- Documentation impact: Tensor API gained the complete current rank-editing contract and runnable
  example; Compile API gained current expression inventory without compiler claims; task, master
  plan, and roadmap now record completion. Training API, capabilities, architecture/ADRs/tests,
  conformance/integration, Gradle/dependencies, and other modules remain accurate unchanged.
- Javadoc review: New Tensor and helper Javadocs are complete and accurate; related
  AxisTransformKind/Attrs, Shape/Dimension, layout, descriptor, factory, provenance, and neighboring
  view-expression Javadocs remain accurate unchanged.
- Glossary impact: Existing implementation-status, axis-transform, normalized-axis, provenance,
  Tensor, and common-distinction text now records current expand-dimensions/squeeze construction;
  no new glossary term was needed.
- Unresolved issues: None within task scope; the documented Known limitations are intentional
  layer boundaries.
- Follow-up required: None for task 0017F1. Task 0017G remains Draft without a specification.

Status: Complete
