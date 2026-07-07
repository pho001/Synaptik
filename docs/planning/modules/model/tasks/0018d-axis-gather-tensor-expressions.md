# Task 0018D: Axis Gather Tensor Expressions

## Status

Complete

## Goal

Add public model-level Tensor expressions for the three completed axis-gather meanings:
shape-reducing `gather`, ONNX-style `gatherAxis` with its `take` alias, and shape-aligned
`takeAlongAxis`.

Every expression consumes ordered logical inputs `[data, indices]`, requires an `INT32` or `INT64`
indices Tensor, normalizes one data axis, derives the operation-specific result Shape, preserves
the data Tensor's type and gradient eligibility, leaves layout unresolved, and records exact
two-input provenance. This task constructs metadata only; it never reads index values or data.

## Scope

- Add exactly these public instance methods to `Tensor`:
  - `Tensor gather(Tensor indices, int axis)`
  - `Tensor gatherAxis(Tensor indices, int axis)`
  - `Tensor take(int axis, Tensor indices)`
  - `Tensor takeAlongAxis(Tensor indices, int axis)`
- Add one field-free package-private final `TensorAxisGatherExpressions` helper in `model.tensor`.
- Give the helper exactly four package-private entries and five private methods specified below.
- Null-check data and indices in deterministic order and accept only exact index data types
  `INT32` and `INT64`.
- Normalize each raw positive or negative axis exactly once through the data Shape.
- For `GATHER`, require indices Shape equal to data Shape with the selected axis removed and use
  that derived reduced Shape as the result.
- For `GATHER_AXIS`, insert every exact indices Dimension at the selected data-axis position while
  removing that selected data Dimension.
- Make tensor-index `take` delegate exactly to the helper's `gatherAxis` entry and therefore create
  exact `GATHER_AXIS` semantics rather than a separate operation.
- For `TAKE_ALONG_AXIS`, require equal ranks and equal non-axis Dimensions, then retain the exact
  indices Shape as the result Shape.
- Preserve exact data type and `requiresGrad`; always use unresolved result layout.
- Create exact `AxisGatherKind`, `IndexAxisAttrs`, ordered `[data, indices]` provenance, no label or
  storage, and one final `TensorFactory.createDerived` call per valid request.
- Keep every valid request fresh, including repeated and nested expressions.
- Update `TensorTest` only for the deliberate four-method public API expansion and add one focused
  expression test.
- Finalize Javadocs, Tensor API, Compile API, glossary, task evidence, master plan, and roadmap
  through the mandatory independent documentation pass.
- Apply the explicitly authorized Javadoc-only timing and bounds-ownership correction to
  `AxisGatherKind` and `IndexAxisAttrs`; declarations, validation, semantics, and tests remain
  unchanged.

## Out of scope

- `take(int, int[])`, `take(int, long[])`, another primitive-array or collection convenience,
  eager index-Tensor construction, or task-0018D1 implementation
- gather-ND, scalar select, scatter, masks, slices, ranges, index conversion, or another operation
  family
- accepting floating or BOOL index tensors, implicitly casting indices, inspecting whether index
  values are integral, or converting `INT64` to `INT32`
- reading, normalizing, clamping, or bounds-checking actual index values; negative index-value
  policy remains an execution/input-value concern outside metadata construction
- broadcasting gather inputs, accepting mismatched dynamic symbols as equal, inventing symbolic
  constraints, or binding dynamic extents
- deriving or preserving resolved layout, view/alias geometry, storage offsets, materialization,
  host storage, device state, or backend routes
- modifying `AxisGatherKind` or `IndexAxisAttrs` declarations or behavior, or modifying Shape,
  Dimension, DataType, TensorDescriptor, TensorFactory, TensorProvenance, Operation, or any
  completed Java contract/test other than the exact Tensor API inventory; the two semantic files
  have one explicit Javadoc-only exception for stale task timing and index-bounds ownership
- gradient rules, repeated-index accumulation, scatter backward, graph capture, canonicalization,
  compiler passes, planning, prepare, runtime, backend, engine, trace, ONNX implementation,
  training, or execution behavior
- another production helper/type, dependency, Gradle/build option, architecture change, another
  module, or task-0018D1 specification

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
- [Task 0007](0007-tensor-descriptor-model.md)
- [Task 0011](0011-public-tensor-skeleton.md)
- [Task 0012](0012-tensor-factory.md)
- [Task 0013](0013-tensor-provenance-skeleton.md)
- [Task 0018A](0018a-scalar-select-semantics.md)
- [Task 0018B](0018b-scalar-select-tensor-expression.md)
- [Task 0018C](0018c-axis-gather-semantics.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only legacy branch exposes:

```java
Tensor gather(Tensor indices, int dimension)
Tensor gatherAxis(Tensor indices, int axis)
Tensor take(int axis, Tensor indices)
Tensor take(int axis, int[] indices)
Tensor takeAlongAxis(Tensor indices, int dimension)
```

Legacy `gather` removes the selected data axis and requires one index for every coordinate of that
reduced Shape. Legacy `gatherAxis` uses ONNX-style Shape insertion, and tensor-index `take`
delegates to it. Legacy `takeAlongAxis` requires indices rank equal to data rank and identical
non-axis extents, then returns the indices Shape.

This task preserves the four public methods that consume an existing index Tensor. The primitive
`int[]` convenience additionally owns eager copied index-Tensor creation and therefore remains the
separate task 0018D1. The selected new baseline tightens legacy index typing to exact `INT32` or
`INT64`; inconsistent acceptance of other numeric data types is not preserved.

Legacy graph builders, direct data access, gradient callbacks, scatter implementations, traits,
compiler/lowering code, kernels, and runtime/backend behavior are capability evidence only and are
not copied.

## Architecture constraints

- `Tensor` remains public mutable API state, not IR. Each method creates a fresh storage-free
  expression through the existing package-private derived-factory seam.
- Semantic identities and normalized axis attributes come only from completed task 0018C.
- The helper may inspect immutable data/index descriptors and Shapes, but never values, host
  storage, runtime residency, device state, or backend capability.
- Index type validation is a model API contract because the index representation is explicit
  metadata. Index value bounds cannot be checked without reading values and remain deferred.
- Shape compatibility is structural and conservative: equal static sizes or equal dynamic symbols
  pass through existing Dimension/Shape equality; no new symbolic constraint is invented.
- Gather results are value-reordered/materialized semantics rather than logical storage views, so
  every result layout is unresolved even when all input geometry is resolved.
- Data type and gradient eligibility come only from `data`; indices never contribute gradient
  eligibility. Current TensorDescriptor already prevents gradients for integral/BOOL data.
- Compiler owns graph capture and canonicalization; compiler-generated/training tasks own gather
  backward/scatter; planning/prepare/backend own materialization, lowering, and execution.
- No dependency, package ownership, or module boundary changes are authorized.

## Package impact

No package is added or moved.

- `io.github.pho001.synaptik.model.tensor.Tensor` — receives four Tensor-index expression methods.
- `io.github.pho001.synaptik.model.tensor.TensorAxisGatherExpressions` — owns local type/axis/Shape
  validation and semantic construction.
- `TensorAxisGatherExpressionTest` — mirrors `model.tensor` for focused helper/API validation.
- `TensorTest` — changes only its exact public API inventory and reflection assertions.

`AxisGatherKind` and `IndexAxisAttrs` remain in `model.operation.index`; no semantic contract moves
into the Tensor package.

## Required contract

### Public Tensor methods

Add exactly:

```java
public Tensor gather(Tensor indices, int axis) {
    return TensorAxisGatherExpressions.gather(this, indices, axis);
}

public Tensor gatherAxis(Tensor indices, int axis) {
    return TensorAxisGatherExpressions.gatherAxis(this, indices, axis);
}

public Tensor take(int axis, Tensor indices) {
    return TensorAxisGatherExpressions.take(this, axis, indices);
}

public Tensor takeAlongAxis(Tensor indices, int axis) {
    return TensorAxisGatherExpressions.takeAlongAxis(this, indices, axis);
}
```

Each public method contains one return statement and exactly one matching helper call. The methods
are non-static and non-synchronized, perform no direct validation or field access, and add no
overloads in this task.

### Helper shape

Create one package-private final, field-free class with one private zero-argument constructor and
exactly these nine static methods:

```java
static Tensor gather(Tensor data, Tensor indices, int axis)
static Tensor gatherAxis(Tensor data, Tensor indices, int axis)
static Tensor take(Tensor data, int axis, Tensor indices)
static Tensor takeAlongAxis(Tensor data, Tensor indices, int axis)
private static void validateIndexType(String operation, TensorDescriptor indicesDescriptor)
private static Shape removeAxis(Shape dataShape, int normalizedAxis)
private static Shape gatherAxisShape(
        Shape dataShape, Shape indicesShape, int normalizedAxis)
private static void validateTakeAlongAxis(
        Shape dataShape, Shape indicesShape, int normalizedAxis)
private static Tensor create(
        Tensor data,
        Tensor indices,
        TensorDescriptor dataDescriptor,
        Shape resultShape,
        AxisGatherKind kind,
        IndexAxisAttrs attrs)
```

Add no field, nested type, alternate constructor, overload, cache, mutable state, or extra method.

### Shared validation order

The `gather`, `gatherAxis`, and `takeAlongAxis` entries each perform this common prefix in exact
order:

1. null-check `data` with message `data`;
2. null-check `indices` with message `indices`;
3. read the exact data descriptor once;
4. read the exact indices descriptor once;
5. call `validateIndexType` once with operation name `gather`, `gatherAxis`, or
   `takeAlongAxis`;
6. read the exact data Shape once;
7. normalize the raw axis exactly once through `dataShape.normalizeAxis(axis)`;
8. construct one `IndexAxisAttrs(normalizedAxis)` after operation-specific Shape validation;
9. call `create` once.

Null and index-type failures occur before axis normalization. All type/axis/Shape validation and
metadata construction occur before identifier allocation, so local failures consume no Tensor ID.

`validateIndexType` accepts exactly `DataType.INT32` and `DataType.INT64`. Every other type throws
`IllegalArgumentException` with exact message:

```text
<operation> indices data type must be INT32 or INT64: <actual>
```

No method inspects index values or storage.

### GATHER Shape rule

After the shared prefix, `gather` calls `removeAxis(dataShape, normalizedAxis)` once. The helper
creates one rank-minus-one Shape by preserving every unaffected exact data Dimension reference.
Rank-one data produces canonical scalar Shape.

Require `indicesDescriptor.shape().equals(resultShape)`. Otherwise throw
`IllegalArgumentException` with exact message:

```text
gather indices shape must equal data shape without gathered axis: expected=<expected>, actual=<actual>
```

Use the derived reduced data Shape as the result, not the indices Shape reference. Then construct
`AxisGatherKind.GATHER`.

Examples:

- data `[2, 3, 4]`, axis `1`, indices `[2, 4]` produces `[2, 4]`;
- data `[N, 3]`, axis `1`, indices `[N]` passes only when the same symbolic Dimension is used;
- rank-one data `[5]`, axis `0`, scalar indices produces scalar result.

### GATHER_AXIS and take Shape rule

After the shared prefix, `gatherAxis` calls `gatherAxisShape` once. The result Dimension order is:

1. exact data Dimensions before the selected axis;
2. every exact indices Dimension in order;
3. exact data Dimensions after the selected axis.

The selected data Dimension is omitted. No other compatibility check is required. Examples:

- data `[2, 3, 4]`, axis `1`, indices `[5, 6]` produces `[2, 5, 6, 4]`;
- scalar indices replace the selected axis with no Dimensions, so data `[2, 3, 4]`, axis `1`
  produces `[2, 4]`;
- data `[2, N]`, axis `0`, indices `[K]` produces `[K, N]` with exact `K` and `N` references.

Construct `AxisGatherKind.GATHER_AXIS`.

The helper's `take` method contains exactly:

```java
return gatherAxis(data, indices, axis);
```

It performs no independent validation or construction. Therefore tensor-index take shares exact
errors, Shape, kind, attributes, provenance, layout, and identifier side effects with gatherAxis.

### TAKE_ALONG_AXIS Shape rule

After the shared prefix, read the exact indices Shape once and call `validateTakeAlongAxis` once.

- If ranks differ, throw `IllegalArgumentException` with exact message
  `takeAlongAxis indices rank must match data rank: expected=<dataRank>, actual=<indicesRank>`.
- Compare every non-selected axis in increasing order with `Dimension.equals`.
- On the first mismatch, throw `IllegalArgumentException` with exact message
  `takeAlongAxis indices dimension at axis <axis> must match data: expected=<expected>, actual=<actual>`.
- Do not compare the selected-axis extents.

Use the exact indices Shape reference as result Shape and construct
`AxisGatherKind.TAKE_ALONG_AXIS`.

Examples:

- data `[2, 3, 4]`, indices `[2, 7, 4]`, axis `1` produces exact indices Shape `[2, 7, 4]`;
- data `[N, 3]`, indices `[N, K]`, axis `1` passes with the same `N` reference/meaning;
- data `[N, 3]`, indices `[M, 2]`, axis `1` fails when `N` and `M` differ.

### Result descriptor, semantics, provenance, and identity

`create` constructs exactly:

```java
TensorDescriptor descriptor = new TensorDescriptor(
        dataDescriptor.dataType(),
        resultShape,
        Optional.empty(),
        dataDescriptor.requiresGrad());
Operation operation = new Operation(kind, attrs);
TensorProvenance provenance = new TensorProvenance(operation, List.of(data, indices));
return TensorFactory.createDerived(descriptor, Optional.empty(), provenance);
```

Every current data type is accepted for data. Exact data type and `requiresGrad` are retained;
indices never affect either. Every result layout is unresolved. Every valid call returns a fresh
Tensor with no label or storage, exact ordered `[data, indices]` provenance, exact kind, and exact
normalized attributes.

## Affected files

Expected implementation change:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorAxisGatherExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/AxisGatherKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/IndexAxisAttrs.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorAxisGatherExpressionTest.java`
- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- this task file
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum scope

This task may create or modify at most the twelve paths listed above. The two semantic-contract
paths are an explicitly user-authorized documentation-review exception and may change only
Javadoc wording; their declarations and executable behavior must remain unchanged.

If implementation requires primitive take convenience, another production type/helper, semantic
or foundational declaration/behavior change, another test, capability-baseline edit, architecture
document, dependency, build change, another module, or more than twelve paths, stop and propose a
follow-up task.

## Javadoc requirements

- Document all four public Tensor methods and every helper method, including the private
  constructor.
- Define ordered `[data, indices]` roles and exact `INT32`/`INT64` index requirement.
- Explain each Shape rule with the concrete examples above, including dynamic Dimension behavior.
- Explain `take` as an exact alias for tensor-index `gatherAxis` and task 0018D1 as the deferred
  primitive convenience.
- Explain exact type/eligibility retention, unresolved layout, provenance, fresh identity, and
  absence of label/storage/value execution.
- State that no index values or bounds are inspected and no gradient rule is defined.
- Distinguish scalar `SELECT`, gather-ND, and scatter.
- Do not promise compiler capture, backend support, materialization, lowering, or execution.

## Acceptance criteria

- Tensor adds exactly the four specified public, non-static, non-synchronized methods and no
  primitive overload.
- Every public method delegates once to its exact helper entry and performs no other work.
- The helper is package-private, final, field-free, has one private constructor, and exactly nine
  specified static methods.
- Null, index-type, axis, and family-specific Shape validation follows the exact order, exception
  types, and messages.
- `GATHER` derives and retains the reduced data Shape and requires structurally equal indices Shape.
- `GATHER_AXIS` inserts exact indices Dimensions; `take` delegates to it and records the same kind.
- `TAKE_ALONG_AXIS` requires equal rank/non-axis Dimensions and retains the exact indices Shape.
- Dynamic Dimensions are accepted only through existing structural equality; no constraints are
  invented.
- All results preserve exact data type/eligibility, use unresolved layout, exact normalized attrs,
  ordered `[data, indices]` provenance, no label/storage, and fresh identity.
- No values/storage, primitive index creation, gradients, graph/compiler/runtime/backend behavior,
  dependencies, build, semantic declarations/behavior, or architecture are changed; the two
  authorized semantic Javadocs only correct current-versus-planned and bounds-ownership wording.
- Tensor API, Compile API, glossary, task evidence, master plan, and roadmap are independently
  reviewed and synchronized; Training API, capabilities, architecture, and related contracts
  receive reasoned no-change conclusions.
- All validation passes and the final diff contains exactly the twelve permitted paths.

## Tests / validation

Run:

```bash
./gradlew :modules:model:test --tests \
  io.github.pho001.synaptik.model.tensor.TensorAxisGatherExpressionTest
./gradlew :modules:model:test --tests \
  io.github.pho001.synaptik.model.tensor.TensorTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

Focused tests must verify exact API/helper shape, null/type/axis validation order and messages,
all six data types for data, both index types, every static/dynamic Shape rule and failure,
rank-one/scalar indices, take alias behavior, exact kinds/attrs/provenance, unresolved layout,
absent label/storage, fresh identities, and lack of value/storage/cross-layer access.

Manual validation must inspect `javap -p -c -s`, one-call delegation bytecode, reflection, source,
imports, generated Javadoc, the executable documentation example when changed, Markdown links,
anchors, fences, whitespace, exact twelve-path scope, synchronized status, and absence of a
task-0018D1 specification.

## Dependencies

- Task 0001 provides exact DataType categories and `INT32`/`INT64` identities.
- Task 0002 provides Shape, Dimension, dynamic-symbol equality, and axis normalization.
- Tasks 0011–0013 provide Tensor, derived identity allocation, and immutable provenance.
- Task 0018C provides exact axis-gather kinds and normalized attributes.

## Follow-up tasks

- 0018D1: legacy primitive `take(int, int[])` convenience with copied eager INT32 indices.
- 0018E: gather-ND semantics and batch-dimension attributes.

Do not create either follow-up specification during this task.

## Architecture impact

Expected impact: None.

This task fills the existing model-owned public expression surface. If implementation requires a
new architecture rule, symbolic constraint system, value access, or cross-module dependency, stop
and report the issue.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0001/0002/0007/0011/0012/0013/0018A/0018B/0018C/0018D,
Tensor API, Compile API, Training API, glossary, current DataType/Dimension/Shape/TensorDescriptor/
Tensor/TensorFactory/TensorProvenance/Operation/AxisGatherKind/IndexAxisAttrs contracts and focused
expression tests, and Java 26 Gradle configuration.

Implement task 0018D exactly. Modify Tensor.java and add package-private final
TensorAxisGatherExpressions.java. Update TensorTest only for the exact four-method API expansion
and add TensorAxisGatherExpressionTest. Add exactly gather(Tensor,int), gatherAxis(Tensor,int),
take(int,Tensor), and takeAlongAxis(Tensor,int).

The field-free helper has exactly nine specified methods. Follow exact null/index-type/axis/Shape
validation order and messages. GATHER requires indices Shape equal to data Shape without the axis.
GATHER_AXIS inserts the full indices Shape; take delegates exactly to it. TAKE_ALONG_AXIS requires
equal ranks/non-axis Dimensions and returns the exact indices Shape. Accept only INT32/INT64
indices. Preserve exact data type/eligibility, leave layout unresolved, create exact kind/attrs and
ordered [data, indices] provenance, and call createDerived once. Every request is fresh.

Do not add primitive take, gather-ND/scatter, value/bounds/storage access, resolved layout,
gradients, graph/compiler/planning/runtime/backend behavior, semantic/foundational declaration or
behavior changes, dependencies, build/architecture changes, or later specs. The documentation pass
may apply only the explicitly authorized stale timing/bounds Javadoc corrections to
`AxisGatherKind` and `IndexAxisAttrs`. Stop beyond twelve paths or on uncertainty.

Run all specified validation, then hand actual diff/evidence to a separate clean-context docs
agent in the same change. It must inspect source/tests/generated Javadoc, finalize permitted
Javadocs/Tensor API/Compile API/glossary/planning, record Training API/capability/architecture and
related-contract no-change conclusions, and rerun validation.

Update task 0018D, model master plan, and roadmap only for planning status/evidence. Do not mark
Complete until both passes succeed. Leave 0018D1 Draft without a specification. Do not commit/push.
```

## Local decisions

- One field-free helper owns all local validation and construction. The four public methods remain
  one-call delegations, and `take` delegates inside the helper to the exact `gatherAxis` entry so
  there is one GATHER_AXIS validation and construction path.
- `GATHER` always derives a new reduced data Shape and compares the indices Shape structurally;
  even when equal, the result retains the derived Shape rather than the indices Shape reference.
- `GATHER_AXIS` retains exact unaffected data Dimensions and exact inserted indices Dimensions.
  Scalar indices insert no Dimensions and therefore remove the selected data axis.
- `TAKE_ALONG_AXIS` validates rank before increasing-order non-axis Dimension equality and retains
  the exact indices Shape reference after validation. The selected extents are deliberately not
  compared.
- Index type is input metadata and is validated locally as exact INT32 or INT64. Index values and
  their bounds require value access and remain outside expression construction.
- An explicit user authorization expanded the task from ten to twelve paths solely to correct
  stale timing and bounds-ownership Javadoc in `AxisGatherKind` and `IndexAxisAttrs`. Their
  declarations, bytecode behavior, and focused semantic tests remain unchanged.

## Known limitations

- No method reads index values, normalizes negative index values, or checks index-value bounds.
- Dynamic compatibility is structural only: equal symbols pass and different symbols fail; no
  symbolic equality constraint or runtime binding is created.
- Every result layout is unresolved. The expressions create no values, physical aliases, storage,
  gradient or repeated-index rule, graph/compiler behavior, materialization, backend lowering,
  ONNX mapping, or execution.
- Primitive `take(int, int[])` remains the Draft task 0018D1 without a detailed specification.

## Validation evidence

- Clean implementation context `/root/implement_model_0018d` added the exact public methods,
  helper, focused tests, and initial planning updates. Independent documentation context
  `/root/implement_model_0018d/review_model_0018d_docs` inspected the actual shared-tree diff,
  source, tests, generated Javadoc, bytecode, APIs, glossary, planning state, and build
  configuration before finalizing documentation in the same overall change.
- The documentation pass applied General plus API/Javadoc style to the four affected production
  Javadocs, Tensor API, Compile API, and glossary; Planning style to this task, the model master
  plan, and roadmap; and Example format to the executable axis-gather example.
- Reviewed architecture and process material included `AGENTS.md`, `ARCHITECTURE.md`, the current
  architecture index, overview, lifecycle, module boundaries, and dependency rules; documentation
  rules and General/API-Javadoc/Planning/Example profiles; planning guide and roadmap; model
  capabilities/master plan; tasks 0001, 0002, 0007, 0011, 0012, 0013, and 0018A–0018D; Tensor,
  Compile, and Training API references; glossary; Java 26 root/model Gradle configuration; final
  implementation/tests; and generated model Javadoc.
- Related-contract review covered DataType, Dimension, Shape, TensorDescriptor, Tensor,
  TensorFactory, TensorProvenance, Operation, AxisGatherKind, IndexAxisAttrs, SelectKind,
  SelectAttrs, and adjacent select/gather contracts/tests. `AxisGatherKind` and `IndexAxisAttrs`
  contained stale “later task” and index-bounds ownership wording. Explicit user authorization
  expanded the task from ten to twelve paths solely for Javadoc corrections; declarations,
  constructor validation, record/enum structure, generated methods, and semantics remain
  unchanged.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorAxisGatherExpressionTest` — `BUILD SUCCESSFUL`; XML
  reports 10 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorTest` — `BUILD SUCCESSFUL`; XML reports 14 tests
  with zero failures, errors, or skips.
- `./gradlew :modules:model:test` — `BUILD SUCCESSFUL`; XML aggregation reports 667 tests across
  78 suites with zero failures, errors, or skips.
- `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL` after the final Javadoc edits. Generated
  `Tensor.html`, `AxisGatherKind.html`, and `IndexAxisAttrs.html` contain the exact signatures,
  `INT32`/`INT64` requirement, three Shape rules, alias, current input-aware boundary, normalized
  axis, ordered provenance, and explicit no-value/no-bounds/no-gradient/no-compiler/no-backend/
  no-execution boundaries. The package-private helper Javadocs were reviewed in source.
- `./gradlew test` — `BUILD SUCCESSFUL`; all 36 root lifecycle tasks completed or were up-to-date
  with no failing task.
- `javap -classpath modules/model/build/classes/java/main -p -c -s` plus focused reflection tests
  confirmed exactly 101 declared public Tensor methods, four new non-static/non-synchronized
  one-call delegations, one final package-private field-free helper with one private constructor
  and exactly nine static methods, exact validation order, one-axis normalization per canonical
  entry, one-call `take` delegation, exact Shape construction, and one final `createDerived` call.
- Source, imports, tests, and bytecode confirm exact INT32/INT64 validation messages; GATHER's new
  reduced Shape and equality check; GATHER_AXIS Dimension insertion; TAKE_ALONG_AXIS rank-first and
  increasing non-axis checks with exact indices Shape retention; exact data type/eligibility;
  unresolved layout; exact kind/attrs/`[data, indices]` provenance; no label/storage; and fresh
  repeated/nested identities. No value, index-bound, storage, gradient, compiler, planning,
  prepare, runtime, backend, engine, trace, training, or ONNX behavior was added.
- The documented `AxisGatherExpressionExample` compiled with
  `javac -cp modules/model/build/classes/java/main -d
  /tmp/synaptik-axis-gather-doc-example /tmp/AxisGatherExpressionExample.java` and ran with model
  classes. It printed exact Shapes `[2, 4]`, `[2, 5, 6, 4]`, `[2, 5, 6, 4]`, and `[2, 7, 4]`,
  `GATHER_AXIS`, `IndexAxisAttrs[axis=1]`, and four expected `true` metadata/provenance facts.
- The targeted Markdown validator resolved 399 local links, including 115 heading anchors, across
  the six changed documentation/planning files. All Markdown fences balance, all twelve paths end
  with a newline, targeted trailing-whitespace scans found no matches, and `git diff --check`
  passes.
- Final changed-path inventory contains exactly the twelve authorized paths: Tensor, the new
  helper, two Javadoc-only semantic contracts, TensorTest, the new focused test, Tensor API,
  Compile API, glossary, this task, model master plan, and roadmap. Task/master/roadmap status is
  synchronized as Complete. Task 0018D1 remains Draft and no task-0018D1 specification exists.
- Training API remains accurate unchanged because the task adds no gradient object/rule, autograd,
  parameter, optimizer, publication, session, or training execution behavior. The capability
  baseline remains accurate unchanged because it already lists gather/gather-axis/take-along-axis,
  exact integral index types, and the separate support layers.
- DataType, Dimension/Shape, TensorDescriptor, TensorFactory, TensorProvenance, Operation,
  SelectKind/SelectAttrs, scalar-select construction, and other operation-family contracts remain
  accurate unchanged because this task composes their existing metadata, structural equality,
  derived identity, and provenance rules without modifying them. Only the two authorized
  axis-gather semantic Javadocs required timing/bounds wording corrections.
- `ARCHITECTURE.md`, focused architecture explanations, ADRs, architecture tests,
  backend-conformance tests, integration tests, Gradle/dependencies, and other modules remain
  accurate unchanged because the task stays within model ownership and changes no dependency
  rule, backend behavior, executable end-to-end behavior, or build requirement.

## Implementation notes

- Added the exact four public Tensor-index methods and the exact field-free nine-method helper.
- Added focused ten-test coverage and expanded the exact Tensor public-method inventory by four.
- Finalized public/helper/semantic Javadocs, Tensor and Compile API references, the glossary,
  executable example, and synchronized planning status without declaration or behavior changes to
  completed semantic contracts.

## Completion summary

- Completed changes: Implemented and documented public axis gather, gather-axis/take, and
  take-along-axis expression construction with exact type/axis/Shape validation, unresolved result
  layout, retained data metadata, and fresh ordered provenance.
- Files changed or created: Exactly the twelve authorized production, test, API, glossary, task,
  master-plan, and roadmap paths, including the two explicitly authorized Javadoc-only semantic
  corrections.
- Tests and validation: Focused 10-test and 14-test suites, all 667 model tests across 78 suites,
  model Javadoc, root tests, javap/reflection/import/source/generated-page review, executable Java
  26 example, 399-link/115-anchor checks, fence/whitespace/newline checks, exact scope/status and
  no-0018D1-spec checks, and `git diff --check` passed.
- Documentation-agent review: Clean context
  `/root/implement_model_0018d/review_model_0018d_docs` completed the required independent pass
  with General, API/Javadoc, Planning, and Example profiles.
- Documentation impact: Tensor API, Compile API, and glossary now describe current axis-gather
  expression construction while index-value bounds, gradients, compiler behavior, lowering, and
  execution remain planned or separately owned.
- Javadoc review: Tensor's four methods, the helper type/constructor/nine methods, and the two
  authorized semantic timing/bounds corrections are final. Related foundational and adjacent
  contracts remain accurate for the reasons recorded above.
- Glossary impact: Axis-gather terminology now distinguishes semantic values, current input-aware
  expression construction, structural dynamic compatibility, exact take aliasing, and deferred
  value/executable behavior.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0018D. Task 0018D1 remains Draft without a detailed
  specification.

Status: Complete
