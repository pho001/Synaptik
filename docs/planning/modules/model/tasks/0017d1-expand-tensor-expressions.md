# Task 0017D1: Expand Tensor Expressions

## Status

Complete

## Goal

Add public, storage-free Tensor expressions for the completed `EXPAND` semantic meaning.

Expansion right-aligns the input Shape with an exact target Shape. Existing input dimensions must
equal their aligned target dimensions or be statically known singletons. Additional leading target
axes are permitted because missing input axes behave as logical singletons. When input layout and
target geometry are resolved locally, construction derives zero-stride alias-view geometry.

This task creates immutable model metadata only. It does not repeat values, attach storage, define
gradients, capture a graph, choose materialization, lower the operation, or execute it.

## Scope

- Add exactly `Tensor.expand(long...)` and `Tensor.expand(Shape)`.
- Make each public method delegate exactly once to its matching helper overload.
- Add one package-private final stateless `TensorExpandExpressions` helper in `model.tensor`.
- Give the helper exactly the six methods specified under Required contract and one private
  zero-argument constructor.
- Normalize raw dimensions through `Shape.of` without retaining or mutating the caller array.
- Treat empty raw dimensions as the canonical scalar target Shape.
- Accept non-negative static extents, including zero; numeric `-1` has no expand meaning.
- Retain the exact scalar, zero-extent, static, mixed dynamic, or fully dynamic target `Shape`
  supplied to the Shape overload.
- Require target rank to be at least input rank and validate aligned axes from right to left by
  structural equality or an input-side static singleton.
- Accept additional leading target axes of every valid Dimension category.
- Reject locally unprovable dynamic compatibility instead of binding symbols or creating hidden
  constraints.
- For a fully static target and any resolved input layout, derive one new view layout by preserving
  input offset and unchanged aligned strides while assigning zero strides to additional leading
  axes and expanded singleton axes.
- Derive geometry from dense, offset-dense, strided, and already-broadcast input layouts.
- Leave result layout unresolved for a dynamic target or unresolved input layout.
- Preserve exact input data type and gradient eligibility.
- Construct exact `ShapeTransformKind.EXPAND`, `TargetShapeAttrs(targetShape)`, and provenance
  `[input]`.
- Call `TensorFactory.createDerived` exactly once with no label or host storage.
- Return a fresh Tensor for every valid call, including identity-like, scalar, repeated, and nested
  requests.
- Add focused tests and update Tensor's exact public method inventory.
- Complete the mandatory independent documentation pass before marking the task Complete.

## Out of scope

- reshape changes, reshape inference, numeric `-1`, element-count comparison, or modification of
  `TensorReshapeExpressions`
- `expandDims`, squeeze, permute, transpose, tile, repeat, broadcast-to aliases, or another public
  Tensor method or overload
- changes to Shape, Dimension, ShapeBroadcast, LayoutDescriptor, LayoutKind, TensorDescriptor,
  TensorFactory, TensorProvenance, Operation, ShapeTransformKind, or TargetShapeAttrs
- target rank below input rank, shrinking non-singleton dimensions, binding dynamic dimensions,
  resolving unequal symbols, or recording deferred symbolic constraints
- value reads or repetition, storage lookup or alias attachment, allocation, copy, capacity or
  lifetime validation, mutation, or eager materialization
- forcing contiguous input, choosing alias versus copy, or promising zero-copy execution
- returning the input, expansion-chain canonicalization, caching, interning, or deduplication
- gradient formulas, broadcast-gradient reduction, autograd, or training behavior
- graph capture, compiler passes, planning requirements, prepare lowering, backend support, runtime
  behavior, execution, or ONNX mapping
- another module, dependency, Gradle/build change, architecture change, or task-0017E specification

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md) — authoritative model ownership and layer
  boundaries.
- [Architecture overview](../../../../architecture/overview.md) — compile/prepare/run separation.
- [Lifecycle](../../../../architecture/lifecycle.md) — Tensor expressions precede compilation.
- [Module boundaries](../../../../architecture/module-boundaries.md) — model contains no runtime or
  concrete backend behavior.
- [Dependency rules](../../../../architecture/dependency-rules.md) — model remains independent of
  compiler, runtime, prepare, engine, and backends.
- [Runtime / Prepare / Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
  — later layers own materialization, lowering, storage, and execution.
- [Planning guide](../../../planning-guide.md) — task scope, evidence, and clean-context rules.
- [Model capabilities](../capabilities.md) — selected `expand` and broadcast-view capability.

## Capability origin

The selected capability baseline requires `expand` and expanded/broadcast views. The read-only
`legacy/pre-rewrite` implementation exposes `Tensor.expand(int...)`, right-aligns source and
target, permits leading-axis and singleton expansion, preserves source strides for equal axes, and
uses zero strides for repeated axes. Tests cover leading-rank and singleton expansion, BOOL and
floating data, view behavior, later contiguous materialization, and backward reduction.

This task preserves only model-level capability: target validation, logical semantics, descriptor
geometry, metadata retention, and provenance. It adapts legacy positive-only int Shapes to the
completed long-sized scalar/zero/dynamic model and excludes immediate storage aliasing, mutable
graph nodes, gradient callbacks, compiler/lowering code, kernels, runtime, and backends.

## Architecture constraints

- Tensor remains public mutable API state and is not an IR node.
- Operation records backend-neutral semantics and no backend support or route.
- LayoutDescriptor records logical geometry, not physical storage or executable aliasing.
- Model may validate only compatibility provable from current immutable Shape metadata.
- Model must not bind symbols or create graph-wide constraints.
- Compiler owns graph capture, constraints, and canonicalization.
- Planning owns logical materialization requirements; backend prepare owns concrete lowering.
- Runtime owns prepared storage/residency and execution.
- No service locator, backend lookup, new dependency, or architecture change is authorized.

## Package impact

No new package is introduced.

- Public overloads remain in `io.github.pho001.synaptik.model.tensor.Tensor`.
- Package-private `TensorExpandExpressions` lives beside Tensor and existing expression helpers.
- Focused tests mirror `model.tensor` for package-private structural inspection.
- Existing shape, layout, operation-layout, descriptor, provenance, and factory packages remain
  unchanged.

## Required contract

### Public Tensor surface

Add exactly:

```java
public Tensor expand(long... requestedShape) {
    return TensorExpandExpressions.apply(this, requestedShape);
}

public Tensor expand(Shape targetShape) {
    return TensorExpandExpressions.apply(this, targetShape);
}
```

Do not add a static form, alias, descriptor overload, factory entry, or default target.

### Helper structure

`TensorExpandExpressions` is package-private, final, field-free, and non-instantiable. Apart from
one private constructor, it declares exactly:

```java
static Tensor apply(Tensor input, long[] requestedShape)
static Tensor apply(Tensor input, Shape targetShape)
private static void validateExpansion(Shape inputShape, Shape targetShape)
private static Optional<LayoutDescriptor> resolveViewLayout(
        TensorDescriptor inputDescriptor, Shape targetShape)
private static long[] deriveExpandedStrides(
        Shape inputShape, LayoutDescriptor inputLayout, Shape targetShape)
private static Tensor create(
        Tensor input,
        TensorDescriptor inputDescriptor,
        Shape targetShape,
        Optional<LayoutDescriptor> resultLayout)
```

Do not add fields, nested types, another constructor, helper file, or hidden overload.

### Validation and normalization order

For `apply(Tensor, long[])`:

1. null-check input with exact message `input`;
2. null-check requestedShape with exact message `requestedShape`;
3. read the exact input descriptor once and its Shape once;
4. construct target through `Shape.of(requestedShape)`;
5. validate expansion;
6. resolve or defer layout;
7. construct the result once.

`Shape.of` supplies existing static-dimension validation. The first negative extent, including
`-1`, fails with `IllegalArgumentException` message
`Static dimension size must be non-negative: <value>`. The array is never retained or mutated.
Empty varargs produce `Shape.scalar()`.

For `apply(Tensor, Shape)`:

1. null-check input with exact message `input`;
2. null-check targetShape with exact message `targetShape`;
3. read the exact input descriptor once and its Shape once;
4. validate expansion;
5. resolve or defer layout;
6. construct the result once.

The exact supplied target Shape reference becomes both descriptor Shape and attribute target.

### Directional right-aligned compatibility

If target rank is below input rank, throw `IllegalArgumentException` with exact message:

```text
expand target rank <targetRank> must be at least input rank <inputRank>
```

Otherwise compute `rankOffset = targetRank - inputRank` and inspect aligned input axes in ascending
order. An aligned pair is valid only when its Dimensions are structurally equal or the input
Dimension is `StaticDimension(1)`. For the first invalid pair throw:

```text
cannot expand input shape <inputShape> to target shape <targetShape> at target axis <targetAxis>
```

Axes before `rankOffset` are new leading axes and are valid. Consequences include:

- scalar input may expand to any target Shape;
- `[3]` may expand to `[2, 3]`;
- `[1, 3]` may expand to `[2, 3]`;
- `[N, 1]` may expand to `[N, 4]`;
- `[1]` may expand to `[N]`;
- `[N]` may expand to `[N]`, but not locally to `[1]`, `[2]`, or `[M]`;
- `[2, 3]` may not expand to `[2, 4]` or a lower-rank target.

Target singleton is not permission to shrink an input. Do not use symmetric broadcasting as a
constraint solver or replace the caller's exact target Shape.

### Resolved and unresolved layout

Read `inputDescriptor.layout()` exactly once. Return `Optional.empty()` when target is dynamic or
input layout is absent. Do not guess strides or synthesize a materialization request.

For a static target and present input layout, allocate one target-rank `long[]` and derive:

- zero for every new leading axis;
- zero for an aligned source extent one whose target extent differs from one;
- otherwise the exact `inputLayout.stride(sourceAxis)`, including existing zero/non-canonical
  strides.

Create exactly:

```java
LayoutDescriptor.of(
        targetShape,
        derivedStrides,
        inputLayout.storageOffset(),
        true)
```

The result is always marked as a view. Repetition of target extents greater than one through zero
stride classifies as `BROADCAST_ZERO_STRIDE`; identity-like, singleton, or empty geometry may
classify as dense, offset-dense, or strided. Preserve the exact offset and never reuse the input
layout object. This is logical metadata only; no storage is attached and zero-copy execution is
not promised.

### Result construction

Common construction creates exactly:

```java
TensorDescriptor descriptor = new TensorDescriptor(
        inputDescriptor.dataType(),
        targetShape,
        resultLayout,
        inputDescriptor.requiresGrad());
TargetShapeAttrs attrs = new TargetShapeAttrs(targetShape);
Operation operation = new Operation(ShapeTransformKind.EXPAND, attrs);
TensorProvenance provenance = new TensorProvenance(operation, List.of(input));
return TensorFactory.createDerived(descriptor, Optional.empty(), provenance);
```

Every success consumes one fresh Tensor ID and has absent label/storage. Failures before
`createDerived` consume no ID; exhaustion remains the final factory failure.

## Affected files

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorExpandExpressions.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorExpandExpressionTest.java`
- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/tasks/0017d1-expand-tensor-expressions.md`
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum scope

At most the ten paths above may change. Production is limited to Tensor plus one new
package-private helper; tests to inventory plus one focused suite; documentation to the public
model API/status and planning evidence.

If another production type, helper, overload, foundational edit, dependency, build/architecture
change, or eleventh path is needed, stop and report. Do not create task 0017E.

## Javadoc requirements

- Fully document both public overloads and every helper contract.
- Explain logical repetition without claiming eager values or storage.
- Document raw-array ownership, scalar empty request, non-negative dimensions, no `-1`, exact Shape
  retention, right alignment, leading axes, singleton expansion, and dynamic compatibility.
- Document result type, Shape, layout, eligibility, label, storage, provenance, and freshness.
- Document every parameter, result, null failure, compatibility/static-dimension/arithmetic
  failure, and identifier exhaustion.
- Explain zero-stride derivation, offset retention, all resolved input layout kinds, unresolved
  dynamic geometry, and why view metadata is not attached storage or execution proof.
- Independently review related shape/layout/descriptor/factory/provenance/operation/reshape/
  contiguous Javadocs and stop on any out-of-scope contradiction.

## Acceptance criteria

- Exactly two expand overloads exist; Tensor's declared public method inventory rises 79 to 81.
- Each overload delegates once; helper is exact: final, package-private, field-free, one private
  constructor, six methods.
- Null order/messages, raw ownership, negative handling, scalar empty request, and exact Shape
  retention match this specification.
- Rank, equality, input-singleton, leading-axis, zero-extent, scalar, and dynamic rules match this
  specification.
- All six data types and valid eligibility states retain exact metadata.
- Dense, offset, strided, and broadcast resolved inputs derive correct strides, offset, view flag,
  kind, and span; dynamic/unresolved cases stay unresolved.
- Result records exact EXPAND/target/[input], absent label/storage, and fresh identity.
- Same-Shape, repeated, and nested calls remain explicit.
- Early failures consume no ID; final factory exhaustion is preserved.
- No values/storage or cross-layer behavior is introduced.
- Independent documentation review and status/evidence synchronization complete before status
  becomes Complete.

## Tests / validation

Run before and after the documentation pass:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorExpandExpressionTest --tests io.github.pho001.synaptik.model.tensor.TensorTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

Focused tests cover API/helper shape; all data types and eligibility; array ownership; scalar,
zero, static, and dynamic targets; leading/equal/singleton cases; exact failures; dense, offset,
strided, and broadcast layouts; preserved/zero strides, offset/kind/view/span; unresolved geometry;
operation/attributes/provenance; absent label/storage; dead-storage non-interference; freshness;
identity side effects; and exhaustion.

Inspect `javap -p -c -s`, reflection, imports, and source. Confirm two one-call delegations, exact
six-method helper, one compatibility path, one layout decision, one stride derivation, one shared
construction path, and no value/storage/cross-layer behavior. Validate generated Javadoc,
executable examples, API/glossary status, links, anchors, fences, whitespace, exact ten paths,
synchronized statuses, and no task-0017E specification.

## Dependencies

- 0002: Shape, Dimension, scalar/zero/dynamic forms, structural equality, defensive construction.
- 0003: resolved strides, offsets, views, layout classification, and referenced spans.
- 0007: resolved-or-unresolved TensorDescriptor.
- 0011–0013: Tensor, centralized derived identity, and immutable provenance.
- 0017C: exact EXPAND semantics and TargetShapeAttrs.
- 0017D: adjacent target-Shape expression pattern and reshape/expand boundary; contextual, not a
  production dependency.

## Follow-up tasks

- 0017E remains Draft for axis-transform semantics.
- 0017F remains Draft for permute, transpose, expand-dimensions, and squeeze expressions.
- Compiler later owns dynamic constraints and expansion canonicalization.
- Planning/backend prepare later own materialization and alias/copy lowering.
- Runtime later executes prepared work; training/compiler-generated tasks own backward reduction.

Do not create a detailed follow-up specification here.

## Architecture impact

Expected impact: None. Model already owns all composed contracts. Compiler, planning, prepare,
backend, runtime, and training responsibilities remain unchanged. Stop if symbolic binding,
storage aliasing, materialization policy, gradient rules, another dependency, or architecture
change becomes necessary.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0002/0003/0007/0011/0012/0013/0017C/0017D/0017D1, Tensor
API, Compile API, Training API, glossary, current Shape/Dimension/LayoutDescriptor/LayoutKind/
TensorDescriptor/Tensor/TensorFactory/TensorProvenance/Operation/ShapeTransformKind/
TargetShapeAttrs/reshape contracts and tests, and Java 26 Gradle configuration.

Implement task 0017D1 exactly. Modify Tensor.java and add package-private final
TensorExpandExpressions.java. Update TensorTest only for the exact two-overload API expansion and
add TensorExpandExpressionTest. Add exactly expand(long...) and expand(Shape), each delegating once
to the matching helper overload.

The stateless helper has exactly six specified methods. Raw requests use Shape.of, accept
non-negative dimensions including zero, treat empty as scalar, and give -1 no special meaning.
Retain exact Shape-overload references. Require target rank at least input rank; accept aligned
pairs only when equal or input is a static singleton; accept new leading axes; reject unprovable
dynamic combinations.

For static target plus any resolved input layout, derive a new view layout with exact input offset,
preserved aligned strides, and zero strides for leading or expanded singleton axes. Otherwise leave
layout unresolved. Retain exact type/target/eligibility, create exact EXPAND + TargetShapeAttrs and
[input] provenance, and call createDerived once with no label/storage. Every request is fresh.

Do not modify reshape/foundational contracts, bind symbols, inspect/repeat/copy values or storage,
attach aliases, force materialization, return input, canonicalize, add APIs/helpers/types, define
gradients, capture graphs, or add cross-layer behavior, dependencies, build/architecture changes,
or later specs. Stop beyond ten paths or on architecture uncertainty.

Run all specified validation, then hand actual diff/evidence to a separate clean-context docs agent
in the same change. It must inspect source/tests/generated Javadoc, finalize permitted Javadocs/
Tensor API/Compile API/glossary/planning, record no-change conclusions, and rerun validation.

Update 0017D1, model master plan, and roadmap only for planning status/evidence. Do not mark
Complete until both passes succeed. Leave 0017E Draft without a specification. Do not commit/push.
```

## Local decisions

- Expand mirrors reshape with raw `long...` and exact `Shape`, but has no inference sentinel.
- Empty raw dimensions mean scalar, consistent with Shape.of.
- Validation is directional: target singleton cannot shrink an input.
- Equal dynamic symbols and input static singleton to dynamic target are locally provable; other
  dynamic combinations are rejected rather than hidden as constraints.
- Leading axes are expansion from implicit singleton axes.
- Every resolved input layout supports stride derivation; contiguity is unnecessary because expand
  preserves aligned source strides and inserts zero strides.
- Resolved results are view metadata without storage attachment or zero-copy guarantee.
- Every call is fresh; compiler later owns identity/nested canonicalization.

## Known limitations

- Unprovable dynamic expansion is rejected and no graph-wide constraint is recorded.
- Dynamic target or unresolved input layout stays unresolved.
- No host-storage alias, gradient, compiler capture, planning requirement, backend lowering,
  runtime execution, ONNX mapping, or conformance behavior exists yet.

## Validation evidence

Planning reviewed the architecture and focused lifecycle/module/dependency/runtime-boundary docs;
documentation/planning rules; roadmap; model capabilities/master plan; prerequisite tasks; current
Shape/Dimension/ShapeBroadcast, layout, descriptor, Tensor/factory/provenance, operation, and
shape-transform source/tests; Tensor/Compile/Training APIs, glossary, and Java 26 Gradle baseline.

The read-only legacy branch confirms varargs expand, right alignment, singleton/leading expansion,
preserved and zero strides, view semantics, all selected data categories, later materialization,
and backward reduction. Coupled legacy storage, graph, gradient, compiler, kernel, and runtime
design is excluded.

Current contracts support this task without a new public type, package, dependency, foundational
edit, or architecture decision.

Planning validation after synchronizing this task, the model master plan, and roadmap:

- `git diff --check` passed, and targeted trailing-whitespace scans found no matches.
- Exact planning scope is three paths: this task, model master plan, and roadmap. No Java, Gradle,
  architecture, API, glossary, completed-task, or other-module file changed during planning.
- All 20 canonical task sections are present.
- Markdown backtick fences are balanced: sixteen in this task, two in the master plan, and zero in
  the roadmap.
- All 183 local Markdown file links across the three changed planning files resolve.
- Every changed file ends with a newline.
- Task, master plan, and roadmap consistently identify 0017D1 as Ready and 0017E as Draft; no
  detailed 0017E specification exists.
- The model task sequence contains 73 ordered rows with no duplicate order number.
- Package review confirms no new package: public methods and the package-private helper stay in
  `model.tensor`, and tests mirror that package.
- Scope and dependency review confirms the implementation can stay within ten authorized paths
  and compose only completed model contracts.
- Granularity review confirms expansion compatibility/stride algebra is one isolated concept and
  remains separate from task 0017E axis-transform semantics.

Implementation and independent documentation validation:

- Implementation context `/root/implement_model_0017d1` added the two public overloads, exact
  six-method helper, and focused tests. Clean documentation context
  `/root/implement_model_0017d1/docs_review_0017d1` independently inspected the final diff,
  source, tests, generated Javadoc, bytecode, imports, Java 26 build configuration, API pages,
  glossary, and planning state. It applied General plus API/Javadoc style to Java and API work,
  Planning style to evidence/status, and Example format to the executable expand example.
- Source and test review confirmed exact raw `Shape.of` ownership, literal non-negative dimensions
  with no `-1` inference, scalar empty request, exact Shape retention, directional right alignment,
  input-singleton and leading-axis expansion, rejection of unprovable dynamics, all six data
  types, unchanged valid gradient eligibility, and fresh unlabeled storage-free results.
- Resolved static geometry preserves input offset and unchanged aligned strides and inserts zero
  strides for leading and expanded-singleton axes across dense, offset, strided, and already-
  broadcast layouts. Dynamic target or unresolved input geometry remains unresolved. Construction
  records exact `EXPAND`, `TargetShapeAttrs(targetShape)`, and `[input]` provenance and calls
  `createDerived` once without inspecting or attaching storage.
- The implementation context's first focused command, `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorExpandExpressionTest --tests
  io.github.pho001.synaptik.model.tensor.TensorTest`, compiled and ran 27 tests but failed one
  test-only assertion in `earlyValidationAndArithmeticFailuresConsumeNoTensorIdentity`. The
  assertion incorrectly expected valid expansion geometry to produce a new `ArithmeticException`;
  valid source layouts combined with zero-stride expansion cannot create that claimed case. The
  invalid assertion was removed and the test renamed to describe the remaining early-validation
  contract. Production source did not change, and every subsequent focused, model, and root run
  passed as recorded below.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorExpandExpressionTest --tests
  io.github.pho001.synaptik.model.tensor.TensorTest` — `BUILD SUCCESSFUL`; XML records 27 tests
  across the two suites (13 expand and 14 Tensor), with zero failures, errors, or skips.
- `./gradlew :modules:model:test` — `BUILD SUCCESSFUL`; 63 XML suites record 497 tests with zero
  failures, errors, or skips.
- `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL`. Generated `Tensor.html` contains both
  overloads, complete parameters/results/failures, raw ownership, directional compatibility,
  dynamic rejection, resolved/unresolved view geometry, storage/materialization boundaries, and
  identifier exhaustion. Package-private helper Javadocs were reviewed in source.
- `./gradlew test` — `BUILD SUCCESSFUL`; all 36 repository test tasks completed without failure.
  No architecture, backend-conformance, or integration test changed because the task adds model
  expression metadata without dependency, backend, or end-to-end behavior.
- `javap -p -c -s` and reflection-backed focused tests confirmed each public overload is one
  matching helper delegation; Tensor exposes exactly 81 declared public methods; and the helper is
  final, package-private, field-free, has one private constructor and exact six methods. Bytecode
  confirms one compatibility path, one layout decision, one stride derivation, one common
  descriptor/attributes/Operation/provenance path, and exactly one `createDerived` call.
- Production helper imports are limited to model and JDK contracts. Source, bytecode, and tests
  found no value repetition, host-storage access or alias attachment, allocation, gradient rule,
  graph/compiler/planning/prepare/runtime/backend dependency, cache, registry, service, implicit
  materialization, or hidden canonicalization.
- The Tensor API example was compiled and executed by Java 26 JShell against the built model
  classes and printed the documented Shape `[2, 4, 3]`, strides `[0, 0, 1]`, offset five,
  `BROADCAST_ZERO_STRIDE`, span eight, view flag, `EXPAND` kind, exact input identity, exact
  dynamic-Shape retention, unresolved dynamic layout, and absent storage. JShell reported a
  macOS preferences-history flush exception only after all statements and output completed; this
  host-only exit limitation did not affect compilation or execution evidence.
- The targeted Markdown validator resolved 232 local file targets and 85 heading anchors across
  the six changed documentation/planning files. Backtick fences are balanced (136 Tensor API,
  four Compile API, sixteen task, two master-plan markers; zero in glossary and roadmap), no
  trailing whitespace was found, every changed file ends with a newline, and `git diff --check`
  passed.
- Exact scope is the authorized ten paths: `Tensor.java`, `TensorExpandExpressions.java`,
  `TensorTest.java`, `TensorExpandExpressionTest.java`, Tensor API, Compile API, glossary, this
  task, model master plan, and roadmap. Task 0017D1, the master-plan row/current status/notes, and
  roadmap frontier/table are synchronized as Complete. Task 0017E remains Draft, and no detailed
  0017E specification exists. No commit or push occurred.
- Training API remains accurate unchanged because no gradient, autograd, parameter, optimizer,
  publication, or session behavior was added. `capabilities.md` already records expand and the
  layer split. Shape/Dimension, LayoutDescriptor/LayoutKind, TensorDescriptor, TensorFactory,
  TensorProvenance, Operation, ShapeTransformKind, TargetShapeAttrs, reshape, and contiguous
  contracts remain accurate because expand composes their existing immutable semantics without
  changing them.
- `ARCHITECTURE.md`, focused architecture explanations, ADRs, architecture tests, backend-
  conformance and integration material, Java 26 Gradle configuration, dependencies, other modules,
  and unrelated tests remain accurate unchanged because ownership, dependency direction,
  lifecycle, build structure, backend behavior, and executable behavior did not change.

## Implementation notes

- Added exactly `Tensor.expand(long...)` and `Tensor.expand(Shape)` as one-call delegations to a
  package-private final stateless helper with the six required methods.
- Added focused coverage for API shape, all data types and valid gradient choices, scalar/zero/
  static/dynamic Shapes, exact compatibility failures, every resolved input layout family,
  unresolved geometry, provenance/freshness, storage non-interference, identity consumption, and
  exhaustion.
- Finalized the Tensor/helper Javadocs, Tensor/Compile APIs, glossary, task evidence, model master
  plan, and roadmap in the mandatory independent documentation context.
- Added no value work, storage attachment, gradient rule, compiler/planning/prepare/runtime/backend
  behavior, dependency, build change, or architecture change.

## Completion summary

- Completed changes: Implemented and documented raw and exact-Shape public expand expressions with
  directional compatibility, conditional same-offset zero-stride view geometry, and exact one-
  input provenance.
- Files changed or created: Exactly two production files, two focused test files, Tensor API,
  Compile API, glossary, this task, model master plan, and roadmap.
- Tests and validation: Focused tests passed 27/27; all 497 model tests across 63 suites, model
  Javadoc, root tests, Java 26 example execution, bytecode/reflection/import/source/generated-
  Javadoc, link/anchor/fence/terminology/whitespace, exact-scope/status, and diff checks passed.
- Documentation-agent review: Clean context
  `/root/implement_model_0017d1/docs_review_0017d1` completed the independent pass using General,
  API/Javadoc, Planning, and Example profiles.
- Documentation impact: Tensor API and glossary now present expand construction as current;
  Compile API preserves the boundary between current expression metadata and planned capture,
  dynamic constraints, canonicalization, materialization, lowering, and execution.
- Javadoc review: Tensor and helper expand contracts are complete. All reviewed foundational,
  layout, descriptor, factory, provenance, operation, reshape, and contiguous contracts remain
  accurate unchanged.
- Glossary impact: Reshape/expand semantics and Tensor status now cover raw ownership, exact Shape
  retention, directional compatibility, conditional zero-stride view metadata, freshness, and
  cross-layer boundaries.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0017D1. Task 0017E remains Draft without a detailed
  specification.

Status: Complete
