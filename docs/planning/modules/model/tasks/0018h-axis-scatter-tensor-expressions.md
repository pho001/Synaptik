# Task 0018H: Axis Scatter Tensor Expressions

## Status

Complete

## Goal

Add public, locally validated Tensor-expression construction for the three completed functional
axis-scatter meanings.

Every method consumes ordered logical inputs `[data, indices, updates]`, returns a fresh
storage-free expression with the exact data Shape and data type, leaves layout unresolved, and
never mutates the data input. The methods validate only metadata available without reading index
or update values.

## Scope

- Add exactly four public instance methods to `Tensor`:
  - `Tensor scatterAdd(Tensor indices, Tensor updates, int axis)`
  - `Tensor scatterAxisAdd(Tensor indices, Tensor updates, int axis)`
  - `Tensor scatterElements(Tensor indices, Tensor updates, int axis)`
  - `Tensor scatterElements(Tensor indices, Tensor updates, int axis,
    ScatterReduction reduction)`
- Add one field-free package-private final `TensorAxisScatterExpressions` helper with exactly the
  eleven methods specified below.
- Make the no-reduction scatter-elements entry delegate exactly to the explicit entry with
  `ScatterReduction.NONE`.
- Require exact `INT32` or `INT64` indices and exact matching data/update types.
- Require floating data/updates for fixed-add `scatterAdd` and `scatterAxisAdd`.
- Permit all current data types for `scatterElements(..., NONE)`; permit floating and integral
  types, but not BOOL, for `ADD`, `MUL`, `MAX`, and `MIN`.
- Normalize the raw public axis exactly once through the data Shape.
- Validate the reduced-rank, rank-changing, and same-rank Shape relationships exactly.
- Preserve exact data Shape/type and data/update gradient-eligibility OR in an unresolved result.
- Construct exact semantics and ordered provenance, then call `createDerived` once without label
  or storage. Every valid request is fresh.
- Update `TensorTest` only for the four-method API expansion and add one focused expression test.
- Permit Javadoc-only current-status corrections to `AxisScatterKind`, `IndexAxisAttrs`, and
  `ScatterElementsAttrs`; their declarations and behavior must not change.
- Finalize Tensor API, Compile API, glossary, task evidence, master plan, and roadmap through the
  mandatory independent documentation pass.

## Out of scope

- scatter-ND, gather/select/take backward, fold, masks, slices, or another operation family
- primitive-array or collection conveniences, factories, static methods, default indices, another
  reduction overload, or other public API
- floating/BOOL indices, implicit conversion, promotion, mixed data/update types, or output-type
  selection
- reading, normalizing, clamping, bounds-checking, or otherwise inspecting index values
- detecting duplicate targets for `NONE`; that requires index values
- applying writes/reductions or defining accumulation order, reproducibility, overflow, NaN,
  signed-zero, empty-domain, atomicity, or backend numerical policy
- resolved layout, aliasing, in-place mutation, storage access, materialization, device state, or
  backend routes
- rejecting floating gradient eligibility for reductions without a current backward rule;
  eligibility metadata is not a backward-support guarantee
- modifying foundational or completed semantic behavior, dependencies, Gradle, architecture,
  another module, or creating a task-0018I specification
- graph capture, gradients, compiler, planning, prepare, runtime, backend, engine, trace, ONNX,
  training, or execution behavior

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
- [Task 0007](0007-tensor-descriptor-model.md)
- [Task 0011](0011-public-tensor-skeleton.md)
- [Task 0012](0012-tensor-factory.md)
- [Task 0013](0013-tensor-provenance-skeleton.md)
- [Task 0018C](0018c-axis-gather-semantics.md)
- [Task 0018D](0018d-axis-gather-tensor-expressions.md)
- [Task 0018G](0018g-axis-scatter-semantics.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only legacy branch exposes all four selected public methods. Fixed-add operations accept
matching floating data and updates. Scatter-elements accepts replacement for all current types,
permits arithmetic reductions for numeric types, and rejects BOOL arithmetic reduction. The new
baseline consistently requires exact `INT32` or `INT64` indices.

The selected Shape meanings are:

- `scatterAdd`: indices and updates equal data Shape with the selected axis removed;
- `scatterAxisAdd`: updates equal `gatherAxis(data, indices, axis)` result Shape; and
- `scatterElements`: equal same-rank indices/update Shapes matching data off-axis.

The new model uses canonical rank-zero Shape, immutable Dimensions, explicit non-null reduction,
and unresolved output layout. Legacy value access, graph builders, gradient callbacks, mutable
Shapes, traits, lowering, kernels, and runtime/backend behavior are not copied.

## Architecture constraints

- `Tensor` remains public mutable API state, not IR. Methods create fresh storage-free expressions
  through the existing derived-factory seam.
- Semantics come only from completed task 0018G.
- The helper may inspect immutable descriptors and Shapes, never values or storage.
- Index type, matching data/update type, raw-axis normalization, and structural Shape rules are
  locally decidable. Bounds and duplicate validity are not.
- Results preserve exact data Shape/type, are new semantic values, and always have unresolved
  layout.
- Gradient eligibility is data/update metadata OR; indices never contribute it.
- Compiler/training owns backward construction. Later layers own value-aware validation,
  lowering, materialization, and execution.
- No dependency, package ownership, or module boundary change is authorized.

## Package impact

No package is added or moved.

- `io.github.pho001.synaptik.model.tensor.Tensor` — four public axis-scatter methods.
- `io.github.pho001.synaptik.model.tensor.TensorAxisScatterExpressions` — validation, Shape
  checks, semantics, and provenance.
- `TensorAxisScatterExpressionTest` — same-package focused coverage.
- `TensorTest` — exact public API inventory/reflection changes only.
- `AxisScatterKind`, `IndexAxisAttrs`, and `ScatterElementsAttrs` — Javadoc-only temporal updates.

## Required contract

### Public Tensor methods

Add exactly:

```java
public Tensor scatterAdd(Tensor indices, Tensor updates, int axis) {
    return TensorAxisScatterExpressions.scatterAdd(this, indices, updates, axis);
}

public Tensor scatterAxisAdd(Tensor indices, Tensor updates, int axis) {
    return TensorAxisScatterExpressions.scatterAxisAdd(this, indices, updates, axis);
}

public Tensor scatterElements(Tensor indices, Tensor updates, int axis) {
    return TensorAxisScatterExpressions.scatterElements(this, indices, updates, axis);
}

public Tensor scatterElements(
        Tensor indices, Tensor updates, int axis, ScatterReduction reduction) {
    return TensorAxisScatterExpressions.scatterElements(
            this, indices, updates, axis, reduction);
}
```

Each method is one matching return/delegate, non-static, non-synchronized, and performs no direct
validation or field access.

### Helper shape

Create one package-private final field-free class, one private zero-argument constructor, and
exactly these eleven static methods:

```java
static Tensor scatterAdd(Tensor data, Tensor indices, Tensor updates, int axis)
static Tensor scatterAxisAdd(Tensor data, Tensor indices, Tensor updates, int axis)
static Tensor scatterElements(Tensor data, Tensor indices, Tensor updates, int axis)
static Tensor scatterElements(
        Tensor data, Tensor indices, Tensor updates, int axis, ScatterReduction reduction)
private static void validateIndexType(String operation, TensorDescriptor indicesDescriptor)
private static void validateMatchingDataType(
        String operation, TensorDescriptor dataDescriptor, TensorDescriptor updatesDescriptor)
private static void validateFloating(String operation, TensorDescriptor dataDescriptor)
private static Shape removeAxis(Shape dataShape, int normalizedAxis)
private static Shape gatherAxisShape(
        Shape dataShape, Shape indicesShape, int normalizedAxis)
private static void validateScatterElementsShape(
        Shape dataShape, Shape indicesShape, Shape updatesShape, int normalizedAxis)
private static Tensor create(
        Tensor data,
        Tensor indices,
        Tensor updates,
        TensorDescriptor dataDescriptor,
        TensorDescriptor updatesDescriptor,
        AxisScatterKind kind,
        OperationAttrs attrs)
```

Add no field, nested type, alternate constructor, overload, cache, mutable state, or extra method.

The short scatter-elements entry contains exactly:

```java
return scatterElements(data, indices, updates, axis, ScatterReduction.NONE);
```

### Shared validation helpers

Index type accepts only `INT32`/`INT64`; failure is:

```text
<operation> indices data type must be INT32 or INT64: <actual>
```

Data/update exact-type mismatch is:

```text
<operation> updates data type must match data: expected=<dataType>, actual=<updatesType>
```

Fixed-add non-floating failure is:

```text
<operation> data and updates must use floating data type: <actual>
```

### Fixed-add scatter

`scatterAdd` performs exactly:

1. null-check data, indices, updates in order with parameter-name messages;
2. read their descriptors once each in that order;
3. validate index type, matching update type, then floating data with operation `scatterAdd`;
4. read exact data Shape and normalize axis once;
5. derive reduced Shape once, preserving unaffected Dimension references;
6. require indices equality or throw
   `scatterAdd indices shape must equal data shape without scattered axis: expected=<expected>, actual=<actual>`;
7. require updates equality or throw
   `scatterAdd updates shape must equal data shape without scattered axis: expected=<expected>, actual=<actual>`;
8. create one `IndexAxisAttrs` and call common create once with `SCATTER_ADD`.

Rank-one data produces canonical scalar expected Shape.

### Rank-changing fixed-add axis scatter

`scatterAxisAdd` performs the same null/descriptor/index/matching/floating/axis order with operation
name `scatterAxisAdd`, derives `gatherAxisShape` once, then requires exact updates equality or
throws:

```text
scatterAxisAdd updates shape must match gatherAxis result shape: expected=<expected>, actual=<actual>
```

It creates one `IndexAxisAttrs` and calls common create once with `SCATTER_AXIS_ADD`. The Shape
helper replaces the selected data axis with every exact indices Dimension; scalar indices remove
the axis.

### Configurable scatter-elements

The explicit path performs exactly:

1. null-check data, indices, updates, reduction in order;
2. read their descriptors once each in order;
3. validate index type and matching data/update type with operation `scatterElements`;
4. reject BOOL with a non-`NONE` reduction using exact message
   `scatterElements BOOL data supports only NONE reduction: <reduction>`;
5. read exact data Shape and normalize axis once;
6. call `validateScatterElementsShape` once;
7. create one `ScatterElementsAttrs` and call common create once with `SCATTER_ELEMENTS`.

The Shape validator checks exactly in order:

1. indices rank equals data rank, else
   `scatterElements indices rank must match data rank: expected=<dataRank>, actual=<indicesRank>`;
2. updates rank equals indices rank, else
   `scatterElements updates rank must match indices rank: expected=<indicesRank>, actual=<updatesRank>`;
3. increasing-axis update Dimensions equal indices Dimensions, else
   `scatterElements updates dimension at axis <axis> must match indices: expected=<expected>, actual=<actual>`;
4. increasing non-selected-axis indices Dimensions equal data Dimensions, else
   `scatterElements indices dimension at axis <axis> must match data: expected=<expected>, actual=<actual>`.

Selected indices/update extent may differ from data. No index values are read.

### Common result construction

Create exactly one descriptor with exact data type, exact data Shape reference, empty layout, and
data/update `requiresGrad` OR; one Operation with exact kind/attributes; one provenance with exact
ordered references `[data, indices, updates]`; and one final
`TensorFactory.createDerived(descriptor, Optional.empty(), provenance)` call. Add no label or
storage. All local failures precede identity allocation and every success is fresh.

## Affected files

Expected implementation change:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorAxisScatterExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/AxisScatterKind.java`
  — Javadoc only
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/IndexAxisAttrs.java`
  — Javadoc only
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/ScatterElementsAttrs.java`
  — Javadoc only
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorAxisScatterExpressionTest.java`
- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- this task file
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum scope

At most the thirteen paths above. The bound is deliberate: four public expressions share one
implementation/test concept, three semantic Javadocs require temporal correction, and public API
plus planning documentation must remain synchronized.

Stop if another production type/test, semantic behavior change, capability edit, dependency,
build/architecture change, another module, or a fourteenth path is required.

## Javadoc requirements

- Document all public/helper methods with parameters, result, nullability, ownership, constraints,
  side effects, identity allocation, and failures.
- Define data, indices, updates, target, reduction, and duplicate target; explain exact provenance.
- Include the three task-0018G Shape examples and raw versus normalized axis.
- Explain index/data-type rules, fixed floating add, and configurable scatter-elements rules.
- Explain exact data Shape/type retention, unresolved layout, and eligibility metadata boundary.
- State that values/storage, bounds, duplicates, writes, and reductions are not inspected/executed.
- Correct stale semantic Javadocs only; declarations/behavior remain bytecode-equivalent.
- Promise no gradients, compiler capture, numeric order, backend support, or execution.

## Acceptance criteria

- Exactly four requested public Tensor methods and no other public API are added.
- The helper has the exact class/constructor/field/method surface and short delegation.
- Every explicit path follows exact validation order/messages.
- Fixed-add accepts three floating types only. Scatter-elements accepts all six types for `NONE`,
  floating/integral for arithmetic reductions, and rejects BOOL arithmetic reductions.
- Index type, exact data/update type, raw axis, and all three Shape rules are validated.
- Valid results retain exact data Shape/type, eligibility OR, empty layout, no label/storage, exact
  semantic values, ordered provenance, and fresh identity.
- No values/storage/bounds/duplicates are inspected and no operation is executed.
- Three semantic contracts remain behaviorally bytecode-equivalent after Javadoc-only changes.
- Tensor/Compile API, glossary, task/master/roadmap are independently finalized. Training API,
  capabilities, architecture, and unrelated contracts receive reasoned no-change conclusions.
- All validation passes with exactly thirteen changed paths.

## Tests / validation

Run:

```bash
./gradlew :modules:model:test --tests \
  io.github.pho001.synaptik.model.tensor.TensorAxisScatterExpressionTest
./gradlew :modules:model:test --tests \
  io.github.pho001.synaptik.model.tensor.TensorTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

Focused tests cover exact API/helper shape/delegation; null/type/reduction/axis/Shape order and
messages; all allowed/rejected types; positive/negative axes; scalar, static, and dynamic Shapes;
exact descriptor/provenance/identity behavior; and absence of value/storage/bounds/duplicate
inspection.

Manually inspect `javap -p -c -s`, reflection/source/imports, delegate bytecode, helper method
count, generated Javadoc, Markdown links/anchors/examples/fences/whitespace, exact scope, status,
semantic bytecode equivalence, and absence of a task-0018I specification.

## Dependencies

- 0001 and 0002: DataType, Shape, Dimension, axis normalization.
- 0007, 0011, 0012, 0013: descriptor, Tensor, derived identity, provenance.
- 0018D: completed gather Shape terminology.
- 0018G: axis-scatter kinds, attributes, reductions, and fixed-add pairing.

## Follow-up tasks

- 0018I: scatter-ND semantics.
- 0018J: public scatter-ND expression.
- 0023: compiler-generated backward semantics.

Do not create a follow-up specification here.

## Architecture impact

Expected impact: None. Stop if a new rule, dependency, or cross-layer service is required.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0001/0002/0007/0011/0012/0013/0018C/0018D/0018G/0018H,
Tensor API, Compile API, Training API, glossary, current Tensor/Shape/Dimension/descriptor/factory/
provenance, axis-scatter semantic contracts/tests, related gather helpers/tests, and Java 26 Gradle.

Implement task 0018H exactly. Modify Tensor.java and add package-private final
TensorAxisScatterExpressions.java. Update TensorTest only for exact four-method API expansion and
add TensorAxisScatterExpressionTest. Add exactly scatterAdd, scatterAxisAdd, and two
scatterElements overloads.

The field-free helper has exactly eleven methods. Follow exact null/descriptor/index-type/
data-update-type/reduction/axis/Shape/construction validation order and messages. Fixed-add paths
accept matching floating data/updates. Scatter-elements NONE accepts all current types; arithmetic
reductions accept floating/integral and reject BOOL. Preserve exact data Shape/type, data/update
eligibility OR, unresolved layout, exact semantics, ordered [data, indices, updates] provenance,
and one createDerived call. Every request is fresh. Default scatter-elements delegates with NONE.
Never inspect index or update values.

Permit Javadoc-only current-status corrections in AxisScatterKind, IndexAxisAttrs, and
ScatterElementsAttrs; do not change declarations/behavior. Do not add scatter-ND, gradients,
compiler/runtime/backend behavior, dependencies, build/architecture changes, or later specs. Stop
beyond thirteen paths or on architecture uncertainty.

Run all validation, then hand actual diff/evidence to a separate clean-context docs agent. It must
inspect source/tests/Javadoc, finalize permitted Javadocs/Tensor API/Compile API/glossary/planning,
record no-change conclusions, and rerun validation.

Update 0018H, model master plan, and roadmap only for status/evidence. Do not mark Complete before
both passes. Leave 0018I Draft without a specification. Do not commit/push.
```

## Local decisions

- Four public methods preserve capability; the short scatter-elements overload is exact `NONE`.
- Fixed-add stays floating-only; scatter-elements keeps replacement for all types and arithmetic
  reduction for numeric types.
- Data/update types match exactly; no cast or promotion is inserted.
- `requiresGrad` is eligibility OR, not a backward-support claim.
- Every result layout is unresolved because functional scatter is a new materialized value.
- Bounds and `NONE` duplicate validity remain value-aware responsibilities.

## Known limitations

- Index values, negative-index policy, bounds, and duplicates are not inspected.
- Numeric order, overflow, NaN/signed-zero, atomics, backend support, and execution are undefined.
- No gradient rule, graph capture, compiler/ONNX/planning/prepare/runtime/backend work is added.
- Scatter-ND remains tasks 0018I–0018J.

## Validation evidence

- Clean implementation context `/root/implement_model_0018h` added the exact public methods,
  field-free helper, focused tests, and initial planning updates. Independent documentation context
  `/root/implement_model_0018h/review_model_0018h_docs` inspected the actual shared-tree diff,
  source, tests, generated Javadoc, bytecode, public references, glossary, planning state, and Java
  26 build configuration before finalizing documentation in the same overall change.
- The documentation pass applied General plus API/Javadoc style to production Javadocs, Tensor API,
  Compile API, and glossary; Planning style to this task, the model master plan, and roadmap; and
  Example format to the runnable axis-scatter example. It finalized only the thirteen authorized
  paths and made no declaration or executable-behavior change during the documentation pass.
- Reviewed architecture and process material included `AGENTS.md`, `ARCHITECTURE.md`, the current
  architecture index, module-boundary and dependency explanations; documentation rules and
  General/API-Javadoc/Planning/Example profiles; planning guide and roadmap; model capabilities
  and master plan; tasks 0001, 0002, 0007, 0011, 0012, 0013, 0018C, 0018D, 0018G, and 0018H;
  Tensor, Compile, and Training API references; glossary; final implementation/tests; related
  gather/semantic/foundation contracts; generated model Javadoc; and root/model Java 26 Gradle
  configuration.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorAxisScatterExpressionTest` — `BUILD SUCCESSFUL`;
  XML reports 10 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorTest` — `BUILD SUCCESSFUL`; XML reports 14 tests
  with zero failures, errors, or skips.
- `./gradlew :modules:model:test` — `BUILD SUCCESSFUL`; XML aggregation reports 716 tests across
  83 suites with zero failures, errors, or skips.
- `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL` after final Javadoc edits. Generated
  `Tensor.html`, `AxisScatterKind.html`, `IndexAxisAttrs.html`, and
  `ScatterElementsAttrs.html` contain the exact signatures, ordered roles, type domains, all three
  Shape relationships, raw/normalized-axis boundary, result metadata, provenance, duplicate
  boundary, and explicit no-value/no-gradient/no-compiler/no-backend/no-execution limits. The
  package-private helper Javadocs were reviewed in source.
- `./gradlew test` — `BUILD SUCCESSFUL`; all 36 root lifecycle tasks completed or were up-to-date
  with no failing task.
- `javap -classpath modules/model/build/classes/java/main -p -c -s` plus focused reflection and
  source inspection confirmed exactly four new public non-static/non-synchronized Tensor methods,
  each with one matching helper call; one final package-private field-free helper with one private
  constructor and exactly eleven static methods; the exact default-`NONE` delegation; validation
  order/messages; one normalization per explicit path; exact Shape/type/eligibility/provenance
  construction; and one final `createDerived` call.
- `javap -p -c -s` output for `AxisScatterKind`, `IndexAxisAttrs`, and `ScatterElementsAttrs`
  diffed with no output against `/tmp/0018h-axis-scatter-kind.before`,
  `/tmp/0018h-index-axis-attrs.before`, and `/tmp/0018h-scatter-elements-attrs.before`. Their
  declarations and executable bytecode therefore remain equivalent after Javadoc-only temporal
  corrections.
- Source, imports, tests, and bytecode confirm exact INT32/INT64 index validation; exact
  data/update type equality; floating fixed-add eligibility; BOOL arithmetic-reduction rejection;
  reduced-rank, rank-changing, and same-rank Shape rules; exact data Shape/type retention;
  data/update eligibility OR; unresolved layout; exact kind/attributes and ordered
  `[data, indices, updates]` provenance; no label/storage; and fresh repeated/nested identities.
  No data, index, or update value, bound, duplicate, write, reduction, mutation, gradient,
  compiler, planning, prepare, runtime, backend, engine, trace, training, or ONNX behavior was
  added.
- The documented `AxisScatterExpressionExample` compiled with
  `javac -cp modules/model/build/classes/java/main -d
  /tmp/synaptik-axis-scatter-doc-example /tmp/AxisScatterExpressionExample.java` and ran with the
  model classes. It printed four exact `Shape[2, 3, 4]` results, `SCATTER_ADD`, normalized
  `IndexAxisAttrs[axis=1]`, default `ScatterElementsAttrs[axis=1, reduction=NONE]`, and three
  expected true metadata/provenance facts.
- The targeted Markdown validator resolved 425 local links, including 127 heading anchors, across
  the six changed documentation/planning files. All Markdown fences balance, all thirteen paths
  end with a newline, targeted trailing-whitespace scans found no matches, and `git diff --check`
  passes.
- Final changed-path inventory contains exactly the thirteen authorized paths: Tensor, the new
  helper, three Javadoc-only semantic contracts, TensorTest, the new focused test, Tensor API,
  Compile API, glossary, this task, model master plan, and roadmap. Task/master/roadmap status is
  synchronized as Complete. Task 0018I remains Draft and no task-0018I specification exists.
- Training API remains accurate unchanged because the task adds no gradient object/rule, autograd,
  parameter, optimizer, publication, session, or training execution behavior. The capability
  baseline remains accurate unchanged because it already inventories the three axis-scatter
  operations, five reductions, exact integral index types, and distinct support layers.
- DataType, Dimension/Shape, TensorDescriptor, TensorFactory, TensorProvenance, Operation,
  ScatterReduction, axis-gather helpers/tests, and other operation-family contracts remain
  accurate unchanged because task 0018H composes their existing metadata, structural equality,
  derived identity, reduction vocabulary, and provenance rules without modifying them. Only the
  three authorized semantic Javadocs required current-versus-planned wording corrections.
- `ARCHITECTURE.md`, focused architecture explanations, ADRs, architecture tests,
  backend-conformance tests, integration tests, Gradle/dependencies, other modules, and later task
  specifications remain accurate unchanged because the task stays within existing model
  ownership and changes no dependency rule, backend behavior, executable end-to-end behavior, or
  build requirement.

## Implementation notes

- Added the exact four public Tensor methods and the exact field-free eleven-method helper.
- Added focused ten-test coverage and expanded the exact Tensor public-method inventory by four.
- Finalized Tensor/helper and three semantic Javadocs, Tensor and Compile API references, glossary
  terminology/status, a runnable example, and synchronized planning evidence without changing
  semantic declarations or behavior.

## Completion summary

- Completed changes: Implemented and documented public functional axis-scatter expression
  construction with exact type, reduction, axis, Shape, result, provenance, and validation-order
  contracts.
- Files changed or created: Exactly the thirteen authorized production, test, API, glossary, task,
  master-plan, and roadmap paths, including the three Javadoc-only semantic corrections.
- Tests and validation: Focused 10-test and 14-test suites, all 716 model tests across 83 suites,
  model Javadoc, root tests, bytecode/reflection/import/source/generated-page review, executable
  Java 26 example, 425-link/127-anchor checks, fence/whitespace/newline checks, exact scope/status
  and no-0018I-spec checks, semantic bytecode equivalence, and `git diff --check` passed.
- Documentation-agent review: Clean context
  `/root/implement_model_0018h/review_model_0018h_docs` completed the mandatory independent pass
  with General, API/Javadoc, Planning, and Example profiles.
- Documentation impact: Tensor API, Compile API, and glossary now describe current axis-scatter
  expression construction while values, bounds, duplicate detection, writes/reductions,
  gradients, compiler behavior, lowering, backend behavior, and execution remain separately owned.
- Javadoc review: Tensor's four methods, the helper type/constructor/eleven methods, and the three
  authorized semantic temporal corrections are final. Related foundational and adjacent contracts
  remain accurate for the reasons recorded above.
- Glossary impact: Axis-scatter terminology now distinguishes data, indices, updates, target,
  duplicate target, reduction, current input-aware expression construction, and deferred
  value/executable behavior.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0018H. Task 0018I remains Draft without a detailed
  specification.

Status: Complete
