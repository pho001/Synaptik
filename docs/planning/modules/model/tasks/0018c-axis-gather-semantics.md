# Task 0018C: Axis Gather Semantics

## Status

Complete

## Goal

Define typed, backend-independent semantic identities and one immutable normalized-axis parameter
for the three axis-based tensor-index operations in the selected capability baseline.

The three operations all consume ordered logical inputs `[data, indices]`, but they are not
interchangeable:

- `GATHER` uses one index value for every coordinate of the data Shape with the selected axis
  removed, and its result has that reduced Shape;
- `GATHER_AXIS` uses ONNX-style axis gather, replacing the selected data axis with the complete
  indices Shape; public `take` will be an alias for this exact semantic identity; and
- `TAKE_ALONG_AXIS` aligns indices with data coordinates of the same rank and returns the exact
  indices Shape.

This task defines meanings and one already-normalized axis only. Public Tensor methods, index
data-type validation, Shape compatibility, result descriptors, provenance, gradients, compiler
behavior, lowering, and execution remain later responsibilities.

## Scope

- Add one public `AxisGatherKind` enum implementing `OperationKind` with exactly `GATHER`,
  `GATHER_AXIS`, and `TAKE_ALONG_AXIS`, in that order.
- Add one public `IndexAxisAttrs` record implementing `OperationAttrs` with exactly one
  non-negative normalized `int axis`.
- Pair all three kinds explicitly with `IndexAxisAttrs` without adding a generic compatibility
  validator.
- Document exact ordered `[data, indices]` input roles and the distinct result-Shape relationships.
- Document `take` as a future public alias of `GATHER_AXIS`, not another semantic kind.
- Keep index data type, rank, Shape, bounds, output descriptor, provenance, and execution outside
  the semantic contracts.
- Add one focused same-package structural, validation, value-semantics, distinction, and
  composition test.
- Keep production in the existing `io.github.pho001.synaptik.model.operation.index` package.
- Finalize Javadocs, Tensor API semantic reference, glossary, task evidence, master plan, and
  roadmap through the mandatory independent documentation pass during implementation.

## Out of scope

- public `Tensor.gather`, `gatherAxis`, `take`, `takeAlongAxis`, overloads, primitive-array
  conveniences, factories, helpers, or task-0018D implementation
- gather-ND, scalar select, scatter, gradient-scatter, masks, slices, or their parameter types
- storing data, indices, input count, rank, Shape, selected extent, index data type, result data
  type, descriptor, layout, requiresGrad, provenance, label, storage, or backend facts
- accepting/rejecting `INT32`, `INT64`, floating, or BOOL index tensors; task 0018D owns the exact
  current requirement that public index tensors use `INT32` or `INT64`
- axis normalization from raw negative syntax, rank validation, index bounds, dynamic-dimension
  policy, Shape compatibility, broadcasting, result Shape construction, or overflow checks
- adding a separate `TAKE` kind; public take aliases ONNX-style `GATHER_AXIS`
- adding shape-rule metadata, operation arity, result-kind, costs, fusion, backend support, routes,
  kernels, traits, registries, factories, visitors, parsers, maps, or reflective discovery
- gradients, repeated-index accumulation, graph capture, canonicalization, compiler, planning,
  prepare, runtime, backend, engine, trace, ONNX implementation, training, or execution behavior
- changing existing Java/tests, dependencies, Gradle, architecture, another module, or creating a
  task-0018D specification

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
- [Task 0005](0005-operation-semantic-foundation.md)
- [Task 0006](0006-operation-model.md)
- [Task 0017K](0017k-tensor-composition-semantics.md)
- [Task 0018A](0018a-scalar-select-semantics.md)
- [Task 0018B](0018b-scalar-select-tensor-expression.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only legacy branch exposes three distinct axis-indexing contracts:

```java
Tensor gather(Tensor indices, int axis)
Tensor gatherAxis(Tensor indices, int axis)
Tensor take(int axis, Tensor indices)
Tensor takeAlongAxis(Tensor indices, int axis)
```

Legacy `gather` requires the indices Shape to equal the data Shape with the selected axis removed;
the result has that same reduced Shape. At each result coordinate, the corresponding scalar index
chooses the data coordinate along the removed axis.

Legacy `gatherAxis` follows ONNX Gather Shape semantics. For data Shape
`[D0, D1, ..., Dn]`, selected axis `a`, and indices Shape `[I0, ..., Im]`, the conceptual result is
`[D0, ..., D(a-1), I0, ..., Im, D(a+1), ..., Dn]`. Legacy `take` delegates to this operation and
therefore is a public convenience alias, not a distinct semantic identity.

Legacy `takeAlongAxis` requires data and indices to have the same rank and compatible non-selected
dimensions. Its result Shape is exactly the indices Shape. Every result coordinate reads an index
from the corresponding indices coordinate and uses it on the selected data axis.

The new model preserves all three capability meanings while using one immutable normalized-axis
record. Legacy accepted more numeric index data types in some paths; the selected new baseline
requires exactly `INT32` or `INT64`, which task 0018D will enforce at the input-aware expression
boundary. Legacy graph builders, mutable Shapes, gradient callbacks, scatter implementations,
traits, lowering, kernels, and runtime/backend behavior are not copied.

## Architecture constraints

- Production remains in `modules/model`, which owns backend-neutral operation meanings.
- `AxisGatherKind` identifies mathematical tensor-index semantics only, not an occurrence, Tensor,
  graph node, result descriptor, executable, kernel, or backend route.
- `IndexAxisAttrs.axis` is already normalized and non-negative. The value stores no rank and cannot
  prove that the axis exists for a particular data input.
- Ordered input roles are semantically `[data, indices]`, but the attributes store neither input.
- The distinct Shape relationships are explanatory contracts used later by task 0018D; these types
  perform no Shape calculation or validation.
- `GATHER_AXIS` is the semantic identity used by both future `gatherAxis` and `take` public methods.
  Alias naming belongs to the Tensor API rather than operation identity.
- Generic `Operation` remains an open kind/attributes pair and does not validate family pairing,
  arity, rank, data type, Shape, bounds, gradients, or backend support.
- Package direction is `model.operation.index -> model.operation + java.base` only.
- Stop if implementation requires Tensor, Shape, DataType, provenance, another production type,
  another test, dependency, architecture change, or cross-layer behavior.

## Package impact

No package is added or moved.

- `io.github.pho001.synaptik.model.operation.index.AxisGatherKind` — exact three-way semantic
  vocabulary for axis tensor-index operations.
- `io.github.pho001.synaptik.model.operation.index.IndexAxisAttrs` — shared normalized axis value.
- `AxisGatherSemanticsTest` — same-package focused structural and semantic-composition test.

The existing `model.operation.index` package owns scalar select and now gains cohesive
tensor-index axis semantics. It remains independent of public Tensor and graph packages.

## Required contract

### Semantic kind vocabulary

Create exactly:

```java
public enum AxisGatherKind implements OperationKind {
    GATHER,
    GATHER_AXIS,
    TAKE_ALONG_AXIS
}
```

The enum declares no project field, explicit constructor, method, nested type, per-constant body,
alias, arity, axis, Shape, data type, result, cost, fusion, route, or backend metadata. Inherited
`Enum.name()` satisfies `OperationKind.name()`.

Document exact meanings:

| Kind | Ordered logical inputs | Conceptual Shape relationship | Attributes |
|---|---|---|---|
| `GATHER` | `[data, indices]` | `indices.shape == remove(data.shape, axis)` and result has that reduced Shape | `IndexAxisAttrs` |
| `GATHER_AXIS` | `[data, indices]` | replace the selected data axis with the complete indices Shape | `IndexAxisAttrs` |
| `TAKE_ALONG_AXIS` | `[data, indices]` | data and indices have the same rank, indices align with data off-axis, result Shape equals indices Shape | `IndexAxisAttrs` |

The Javadocs must state that the table explains meaning rather than performing validation. They
must distinguish scalar `SELECT` and gather-ND. `GATHER_AXIS` documentation must explicitly name
future public `take` as an alias and forbid a separate `TAKE` enum constant.

### Normalized axis attributes

Create exactly:

```java
public record IndexAxisAttrs(int axis) implements OperationAttrs
```

The record has exactly one component, one canonical constructor, one explicit documented accessor,
and record-generated `equals`, `hashCode`, and `toString`. Add no rank, dimension extent, raw axis,
normalization flag, input role, index type, Shape, result, factory, overload, builder, nested type,
or extra state/API.

Constructor behavior is exact:

- if `axis < 0`, throw `IllegalArgumentException` with exact message
  `axis must be non-negative: <axis>`;
- otherwise retain the primitive unchanged.

Zero and `Integer.MAX_VALUE` are structurally valid because no input rank is present.

### Typed composition

Each valid semantic composition uses the same axis value:

```java
IndexAxisAttrs attrs = new IndexAxisAttrs(1);
Operation gather = new Operation(AxisGatherKind.GATHER, attrs);
Operation gatherAxis = new Operation(AxisGatherKind.GATHER_AXIS, attrs);
Operation takeAlongAxis = new Operation(AxisGatherKind.TAKE_ALONG_AXIS, attrs);
```

The exact attributes reference is retained by each independently constructed Operation. Generic
Operation does not enforce the pairing. Add no public operation factory or compatibility matrix.

### Concrete Shape examples

The documentation and focused test terminology must use these non-executable examples:

- Data `[2, 3, 4]`, axis `1`, indices `[2, 4]` with `GATHER` conceptually produces `[2, 4]`.
- Data `[2, 3, 4]`, axis `1`, indices `[5, 6]` with `GATHER_AXIS` conceptually produces
  `[2, 5, 6, 4]`; `take` names the same operation.
- Data `[2, 3, 4]`, axis `1`, indices `[2, 7, 4]` with `TAKE_ALONG_AXIS` conceptually produces
  `[2, 7, 4]`, subject to later non-axis compatibility checks.

No production code stores or calculates these Shapes.

## Affected files

Expected implementation change:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/AxisGatherKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/IndexAxisAttrs.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/index/AxisGatherSemanticsTest.java`
- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task file
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum scope

This task may create or modify at most the eight paths listed above.

If implementation requires another production type, another test, existing Java edits, public
Tensor behavior, Compile API change, capability-baseline edit, dependency, build change,
architecture document, another module, or more than eight paths, stop and propose a follow-up
task.

## Javadoc requirements

- Document every public type, enum constant, record component/accessor, and canonical constructor.
- Explain `[data, indices]` input order before using it in each kind description.
- Explain all three distinct result-Shape relationships with the concrete examples above.
- Explain normalized axis meaning and why rank validation is deferred.
- State that only task 0018D validates index tensors as `INT32` or `INT64`.
- Document `take` as a future public alias for `GATHER_AXIS`, not an enum alias or new kind.
- Distinguish axis gather from scalar `SELECT`, gather-ND, and functional scatter.
- Do not promise gradients, compiler capture, backend support, numerical execution, bounds checks,
  or materialization.

## Acceptance criteria

- `AxisGatherKind` is a public enum implementing `OperationKind` with exactly `GATHER`,
  `GATHER_AXIS`, and `TAKE_ALONG_AXIS`, in that order.
- The enum adds no project-declared state, methods, constructors, nested types, or metadata.
- `IndexAxisAttrs` is a public record implementing `OperationAttrs` with exactly one `int axis`.
- Negative-axis failure uses the exact exception type and message; valid values are unchanged.
- Record-generated value semantics remain the object contract and the explicit accessor is fully
  documented.
- All three exact kind/attributes compositions work through unchanged `Operation`.
- Javadocs preserve the three distinct meanings, input order, examples, and take alias boundary.
- No Tensor, DataType, Shape, descriptor, provenance, graph, gradient, compiler, runtime, backend,
  or execution behavior is introduced.
- Tensor API, glossary, task evidence, master plan, and roadmap are independently reviewed and
  synchronized. Compile API, Training API, capabilities, architecture, and related contracts
  receive reasoned no-change conclusions.
- All validation passes and the final diff contains exactly the eight permitted paths.

## Tests / validation

Run:

```bash
./gradlew :modules:model:test --tests \
  io.github.pho001.synaptik.model.operation.index.AxisGatherSemanticsTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused test must verify:

- exact enum constant count, names, order, interface, and absence of project API/state;
- exact record status, component name/type/order, interface, accessor, and absence of extra state;
- zero, ordinary, and `Integer.MAX_VALUE` axis retention;
- exact negative-axis validation and message;
- generated equality, hashing, and diagnostic text;
- exact composition of every kind with one `IndexAxisAttrs` reference;
- kind identity distinctions from each other and `SelectKind.SELECT`;
- absence of `TAKE`, gather-ND, scatter, gradient, or alias enum constants;
- absence of Tensor, DataType, Shape, layout, graph, compiler, runtime, and backend dependencies.

Manually inspect `javap -p -c -s`, reflection/source/imports, generated Javadoc, Markdown links,
anchors, fences, whitespace, exact eight-path scope, synchronized task/master/roadmap status, and
absence of a task-0018D specification.

## Dependencies

- Task 0005 defines the minimal operation-kind and typed-attributes contracts.
- Task 0006 defines the open immutable Operation pair.
- Tasks 0018A and 0018B establish the index package and distinguish scalar select.
- Task 0018D depends on this task for public Tensor validation, Shape derivation, and provenance.

## Follow-up tasks

- 0018D: public gather, gatherAxis/take, and takeAlongAxis Tensor expressions.
- 0018E: gather-ND semantic identity and batch-dimension attributes.

Do not create either follow-up specification during this task.

## Architecture impact

Expected impact: None.

The task fills the existing model-owned operation vocabulary. If implementation requires a new
architecture rule or cross-module dependency, stop and report the issue.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0002/0005/0006/0017K/0018A/0018B/0018C, Tensor API,
Compile API, Training API, glossary, current OperationKind/OperationAttrs/Operation/SelectKind/
SelectAttrs and related operation-family contracts/tests, and Java 26 Gradle configuration.

Implement task 0018C exactly. Add only AxisGatherKind.java, IndexAxisAttrs.java, and
AxisGatherSemanticsTest.java under io.github.pho001.synaptik.model.operation.index.

AxisGatherKind contains exactly GATHER, GATHER_AXIS, and TAKE_ALONG_AXIS in order, with no project
state/methods/nested types/metadata. IndexAxisAttrs contains exactly one normalized non-negative
int axis, exact validation/message, and an explicit documented accessor. Document ordered
[data, indices] roles, the three distinct Shape relationships and examples, exact typed pairings,
and public take as a future GATHER_AXIS alias rather than another kind.

Do not add Tensor methods, DataType/Shape/result/provenance validation, gather-ND/scatter/gradient
types, factories, graph/compiler/planning/runtime/backend behavior, dependencies, build or
architecture changes, existing Java edits, or later specs. Stop beyond eight paths or on
architecture uncertainty.

Run all specified validation, then hand the actual diff/evidence to a separate clean-context docs
agent in the same change. It must inspect source/tests/generated Javadoc, finalize permitted
Javadocs/Tensor API/glossary/planning, record Compile API/Training API/capability/architecture and
related-contract no-change conclusions, and rerun validation.

Update task 0018C, model master plan, and roadmap only for planning status/evidence. Do not mark
Complete until both passes succeed. Leave 0018D Draft without a specification. Do not commit/push.
```

## Local decisions

- One enum preserves the common axis-index family while three constants preserve the distinct
  index-alignment and result-Shape meanings. A kind carries no input or shape-rule metadata.
- One `IndexAxisAttrs` record is shared because every kind needs the same already-normalized
  non-negative data-axis position. The eventual data rank remains input context rather than
  duplicated attribute state.
- Ordered `[data, indices]` roles are semantic documentation and test terminology, not stored
  arity or operand state. Generic `Operation` continues to retain any non-null kind/attributes
  pair without family validation.
- Future public `take` names exact `GATHER_AXIS` semantics. No alias constant, `TAKE` kind,
  registry, or factory was added.
- Index data-type eligibility remains at the future public input-aware boundary: task 0018D will
  require `INT32` or `INT64` rather than encoding input types in semantic attributes.

## Known limitations

- The semantic values do not normalize a caller's negative axis or validate data rank, indices
  rank/type/Shape, selected extent, bounds, dynamic-dimension compatibility, result Shape, or
  overflow. Task 0018D owns those input-aware rules and public expression construction.
- No Tensor method, result descriptor, layout, provenance, storage, value access, gradient or
  repeated-index behavior, graph capture, compiler transformation, ONNX mapping, planning,
  materialization, backend lowering, runtime behavior, or execution is implemented.
- Gather-ND and all functional-scatter semantic families remain later Draft tasks.

## Validation evidence

- Clean implementation context `/root/implement_model_0018c` added exactly the two production
  contracts and one focused test, then handed the shared-tree diff to independent documentation
  context `/root/implement_model_0018c/review_model_0018c_docs`.
- The documentation context applied General plus API/Javadoc style to both production Javadocs,
  Tensor API, and glossary; Planning style to this task, the model master plan, and roadmap; and
  Example format to the three-way conceptual Shape example. It independently inspected final
  source, tests, generated Javadoc, and the actual diff.
- Reviewed architecture and process material: `AGENTS.md`, `ARCHITECTURE.md`, focused current
  architecture, overview, lifecycle, module-boundary, dependency, and runtime/prepare/backend
  explanations; documentation rules and General/API-Javadoc/Planning/Example profiles; planning
  guide and roadmap; model capabilities/master plan; and tasks 0002, 0005, 0006, 0017K, 0018A,
  0018B, and 0018C.
- Reviewed API and implementation material: Tensor, Compile, and Training API references;
  glossary; final `AxisGatherKind`, `IndexAxisAttrs`, and `AxisGatherSemanticsTest`; current
  `OperationKind`, `OperationAttrs`, `Operation`, `SelectKind`, `SelectAttrs`, composition and
  related family contracts/tests; generated model Javadoc; and Java 26 root/model Gradle
  configuration.
- The Javadoc review finalized both production contracts without behavior changes. The rendered
  pages document ordered `[data, indices]` roles, all three exact Shape meanings and examples,
  normalized-axis/rank boundaries, exact pairings, the future `take` alias with no `TAKE` kind,
  task-0018D `INT32`/`INT64` ownership, constructor failure, accessor result, and distinctions from
  scalar select, gather-ND, and functional scatter.
- Tensor API now lists `AxisGatherKind` and `IndexAxisAttrs` as current, explains their
  relationship through a three-way conceptual example, records exact structural failure behavior,
  and keeps public Tensor input validation/result construction and every cross-layer behavior
  planned. Glossary status, `OperationKind`/`OperationAttrs` inventories, and the reusable axis-
  gather definition carry the same current-versus-planned boundary.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.operation.index.AxisGatherSemanticsTest` — `BUILD SUCCESSFUL`;
  the XML report contains 9 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:test` — `BUILD SUCCESSFUL`; 77 XML suites contain 657 tests with zero
  failures, errors, or skips.
- `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL`; generated public pages contain both new
  contracts and the complete reviewed semantic, validation, accessor, and boundary documentation.
- `./gradlew test` — `BUILD SUCCESSFUL`; the root lifecycle reported 36 actionable tasks with no
  failing task.
- `javap -classpath modules/model/build/classes/java/main -p -c -s` confirmed only three enum
  constants plus compiler-generated enum machinery. The record has exactly one private final
  `int axis`, one public canonical constructor with a single negative check and direct assignment,
  one explicit direct accessor, and generated `equals`, `hashCode`, and `toString`.
- Focused reflection, source, and import inspection confirmed exact enum order/interfaces, no
  project-declared enum API/state/nested types, exact record structure, exact validation message,
  generated value semantics, exact kind/attributes compositions, and only the two permitted local
  `OperationKind`/`OperationAttrs` production imports. No Tensor or cross-layer dependency exists.
- Targeted Markdown validation resolved 363 local links, including 91 heading anchors, across the
  five changed documentation/planning files. Code fences are balanced, trailing-whitespace scans
  found no matches, all eight files have final newlines, and `git diff --check` passes.
- Final changed-path inventory contains exactly the eight authorized paths: the two production
  contracts, one focused test, Tensor API, glossary, this task, model master plan, and roadmap.
  Task/master-plan/roadmap status is synchronized as Complete; task 0018D remains Draft and no
  task-0018D specification exists.
- Compile API remains accurate unchanged because this task adds semantic vocabulary only: it adds
  no Tensor expression input, graph capture, inference, validation, optimization, artifact, or
  engine behavior. Training API remains accurate unchanged because no gradient object/rule,
  autograd, parameter, optimizer, session, publication, or training execution behavior is added.
- The capability baseline remains accurate unchanged because it already inventories gather,
  gather-axis/take, take-along-axis, exact integral index types, and the separate support layers;
  this task implements only its model-semantic portion.
- `ARCHITECTURE.md`, focused architecture explanations, ADRs, architecture tests, backend-
  conformance tests, integration tests, Gradle/dependencies, and other modules remain accurate
  unchanged because the task stays inside existing model ownership and changes no boundary,
  dependency rule, backend behavior, end-to-end behavior, or build requirement.
- Operation foundations, `Operation`, scalar select, tensor composition, Shape, DataType, Tensor,
  provenance, and other operation-family contracts/tests remain accurate unchanged because the
  new types compose the open semantic contracts without altering normalization, descriptors,
  public expressions, provenance, or existing family behavior.

## Implementation notes

- Added `AxisGatherKind` with exact ordered `GATHER`, `GATHER_AXIS`, and `TAKE_ALONG_AXIS`
  constants and no project-declared behavior or metadata.
- Added `IndexAxisAttrs(int axis)` with one exact non-negative check, direct primitive retention,
  explicit documented accessor, and record-generated object methods.
- Added nine focused tests covering exact structure, retained extremes, exact failures, value
  semantics, typed composition, semantic distinctions, and absence of alias/cross-layer state.
- Finalized the two production Javadocs, Tensor API semantic reference and current/planned
  inventory, glossary terminology/inventories, and synchronized planning evidence/status.

## Completion summary

- Completed changes: Implemented and documented the three distinct axis-gather semantic identities
  and their shared normalized non-negative axis attributes.
- Files changed or created: Exactly the eight authorized production, test, API, glossary, task,
  master-plan, and roadmap paths.
- Tests and validation: Focused 9-test and all 657-model-test/77-suite runs, model Javadoc, root
  tests, bytecode/reflection/source/import/generated-page review, 363-link/91-anchor checks,
  fence/whitespace/newline checks, exact scope/status and no-0018D-spec checks, and
  `git diff --check` passed.
- Documentation-agent review: Clean context
  `/root/implement_model_0018c/review_model_0018c_docs` completed the required independent pass
  with General, API/Javadoc, Planning, and Example profiles.
- Documentation impact: Tensor API and glossary now document current axis-gather semantics and
  keep public input-aware expressions, gradients, compiler behavior, lowering, and execution
  planned.
- Javadoc review: Both new public types, all constants, the canonical constructor, and the explicit
  accessor are final and document semantics, inputs, constraints, results, failures, and ownership
  boundaries.
- Glossary impact: Added reusable axis-gather terminology and synchronized `OperationKind` and
  `OperationAttrs` current-family inventories.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0018C. Task 0018D remains the next Draft frontier without a
  detailed specification.

Status: Complete
