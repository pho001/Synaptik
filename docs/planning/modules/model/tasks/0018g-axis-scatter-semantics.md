# Task 0018G: Axis Scatter Semantics

## Status

Complete

## Goal

Define typed, backend-independent semantic identities for the three selected functional
axis-scatter operations and one reusable immutable reduction vocabulary.

All three operations consume ordered logical inputs `[data, indices, updates]` and produce a new
result with the exact `data` Shape. They never mutate `data` in place:

- `SCATTER_ADD` adds reduced-rank updates selected like the inverse of `GATHER`;
- `SCATTER_AXIS_ADD` adds rank-changing updates selected like the inverse of `GATHER_AXIS`; and
- `SCATTER_ELEMENTS` writes or reduces same-rank updates along one selected axis.

This task defines meanings and immutable parameters only. Public Tensor methods, input data-type
and Shape validation, descriptors, provenance, index-value bounds and duplicate checks, gradients,
compiler behavior, lowering, and execution remain later responsibilities.

## Scope

- Add one public `AxisScatterKind` enum implementing `OperationKind` with exactly
  `SCATTER_ADD`, `SCATTER_AXIS_ADD`, and `SCATTER_ELEMENTS`, in that order.
- Add one public `ScatterReduction` enum with exactly `NONE`, `ADD`, `MUL`, `MAX`, and `MIN`, in
  that order.
- Add one public `ScatterElementsAttrs` record implementing `OperationAttrs` with exactly a
  non-negative normalized `int axis` and a non-null `ScatterReduction reduction`.
- Pair `SCATTER_ADD` and `SCATTER_AXIS_ADD` explicitly with the existing `IndexAxisAttrs`; their
  kind names define fixed addition and they do not carry a configurable reduction.
- Pair `SCATTER_ELEMENTS` explicitly with `ScatterElementsAttrs`.
- Document exact ordered `[data, indices, updates]` roles, functional result semantics, the three
  distinct Shape relationships, and reduction meanings.
- Document `NONE` as replacement semantics whose duplicate target coordinates are invalid rather
  than resolved by an unspecified update order.
- Keep value-aware index bounds, duplicate detection, numerical accumulation order, and execution
  outside the semantic contracts.
- Add one focused same-package structural, validation, value-semantics, distinction, and
  composition test.
- Permit Javadoc-only wording changes to `IndexAxisAttrs` so its completed normalized-axis value
  accurately documents both gather and scatter pairings without changing declaration or behavior.
- Keep production in the existing `io.github.pho001.synaptik.model.operation.index` package.
- Finalize Javadocs, Tensor API semantic reference, glossary, task evidence, master plan, and
  roadmap through the mandatory independent documentation pass during implementation.

## Out of scope

- public `Tensor.scatterAdd`, `scatterAxisAdd`, or `scatterElements` methods, overloads, factories,
  helpers, or task-0018H implementation
- scatter-ND, gather-ND backward, scalar-select backward, gather backward, take-along-axis
  backward, fold, masks, slices, or another operation family
- storing data, indices, updates, input count, rank, Shape, selected extent, data type, result data
  type, descriptor, layout, requiresGrad, provenance, label, storage, or backend facts
- validating `INT32` or `INT64` index tensors, matching data/update types, floating eligibility,
  BOOL restrictions, ranks, Shapes, broadcasting, bounds, duplicate indices, or result metadata
- axis normalization from raw negative syntax or validating an axis against a concrete rank
- applying a configurable reduction to `SCATTER_ADD` or `SCATTER_AXIS_ADD`; both are intrinsically
  additive semantic identities
- treating null reduction as `NONE`, adding an implicit default, or using null as semantic state
- defining iteration order, floating-point accumulation order, NaN behavior, signed-zero behavior,
  overflow behavior, empty-domain behavior, atomicity, determinism across backends, or kernels
- adding operation arity, result-kind, costs, fusion, backend support, routes, traits, registries,
  factories, visitors, parsers, maps, aliases, or reflective discovery
- gradients, graph capture, canonicalization, compiler, planning, prepare, runtime, backend,
  engine, trace, ONNX implementation, training, or execution behavior
- changing existing Java behavior/tests, dependencies, Gradle, architecture, another module, or
  creating a task-0018H specification

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
- [Task 0005](0005-operation-semantic-foundation.md)
- [Task 0006](0006-operation-model.md)
- [Task 0018C](0018c-axis-gather-semantics.md)
- [Task 0018D](0018d-axis-gather-tensor-expressions.md)
- [Task 0018E](0018e-gather-nd-semantics.md)
- [Task 0018F](0018f-gather-nd-tensor-expressions.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only legacy branch exposes three distinct public axis-scatter capabilities:

```java
Tensor scatterAdd(Tensor indices, Tensor updates, int axis)
Tensor scatterAxisAdd(Tensor indices, Tensor updates, int axis)
Tensor scatterElements(Tensor indices, Tensor updates, int axis, ScatterReduction reduction)
```

Legacy `scatterAdd` is the fixed-add counterpart of reduced-rank `gather`. Its indices and updates
have the data Shape with the selected axis removed. One index value at each reduced coordinate
chooses the destination coordinate on that axis, and repeated destination coordinates accumulate.

Legacy `scatterAxisAdd` is the fixed-add counterpart of ONNX-style `gatherAxis`. Its updates have
the Shape that `gatherAxis(data, indices, axis)` would produce: the complete indices Shape replaces
the selected data axis. Repeated destinations accumulate into the result.

Legacy `scatterElements` uses same-rank indices and updates with equal Shapes. Non-axis Dimensions
match `data`; the selected indices Dimension may differ from the selected data Dimension. It
supports replacement plus `ADD`, `MUL`, `MAX`, and `MIN` reductions. The selected baseline keeps
legacy rejection of duplicate targets for replacement semantics so result meaning does not depend
on an unspecified update order.

The new model preserves these public mathematical meanings while replacing nullable defaults and
legacy operation classes with explicit immutable typed values. Legacy mutable Shapes, graph
builders, gradient callbacks, traits, lowering, kernels, and runtime/backend behavior are not
copied.

## Architecture constraints

- Production remains in `modules/model`, which owns backend-neutral operation meanings.
- `AxisScatterKind` identifies mathematical functional-scatter meaning only, not an occurrence,
  Tensor, graph node, descriptor, executable, kernel, or backend route.
- Every kind has ordered logical inputs `[data, indices, updates]`; semantic attributes store none
  of those operands.
- Every conceptual result starts from `data`, applies writes or reductions, retains the exact data
  Shape, and is a new value. These contracts do not mutate data or execute that transformation.
- `IndexAxisAttrs.axis` and `ScatterElementsAttrs.axis` are already normalized and non-negative.
  Neither stores rank or proves that its axis exists for a particular input.
- `SCATTER_ADD` and `SCATTER_AXIS_ADD` use existing `IndexAxisAttrs`; addition is intrinsic to the
  semantic kind and must not be duplicated as configurable attribute state.
- `SCATTER_ELEMENTS` uses `ScatterElementsAttrs` because its reduction is caller-selected semantic
  state in addition to its normalized axis.
- `ScatterReduction` is reusable semantic vocabulary. Task 0018I may use it for scatter-ND without
  introducing another reduction enum.
- Shape relationships and reduction descriptions are explanatory contracts used by task 0018H;
  these types perform no input-aware Shape or data-type validation.
- Generic `Operation` remains an open kind/attributes pair and does not validate family pairing,
  arity, ranks, Shapes, data types, bounds, duplicates, gradients, or backend support.
- Package direction is `model.operation.index -> model.operation + java.base` only.
- Stop if implementation requires Tensor, Shape, DataType, provenance, another production type,
  another test, dependency, architecture change, or cross-layer behavior.

## Package impact

No package is added or moved.

- `io.github.pho001.synaptik.model.operation.index.AxisScatterKind` — exact three-way semantic
  vocabulary for functional axis scatter.
- `io.github.pho001.synaptik.model.operation.index.ScatterReduction` — reusable immutable
  replacement/reduction vocabulary for functional scatter families.
- `io.github.pho001.synaptik.model.operation.index.ScatterElementsAttrs` — normalized axis and
  explicit reduction for `SCATTER_ELEMENTS` only.
- `io.github.pho001.synaptik.model.operation.index.IndexAxisAttrs` — existing normalized axis
  contract reused unchanged by the two fixed-add kinds; only Javadoc wording may change.
- `AxisScatterSemanticsTest` — same-package focused structural and semantic-composition test.

The existing index-operation package remains cohesive and independent of public Tensor and graph
packages.

## Required contract

### Semantic kind vocabulary

Create exactly:

```java
public enum AxisScatterKind implements OperationKind {
    SCATTER_ADD,
    SCATTER_AXIS_ADD,
    SCATTER_ELEMENTS
}
```

The enum declares no project field, explicit constructor, method, nested type, per-constant body,
alias, arity, axis, reduction, Shape, data type, result, cost, fusion, route, or backend metadata.
Inherited `Enum.name()` satisfies `OperationKind.name()`.

Document exact meanings:

| Kind | Ordered logical inputs | Conceptual Shape relationship | Attributes |
|---|---|---|---|
| `SCATTER_ADD` | `[data, indices, updates]` | `indices.shape == updates.shape == remove(data.shape, axis)`; fixed addition produces `data.shape` | `IndexAxisAttrs` |
| `SCATTER_AXIS_ADD` | `[data, indices, updates]` | `updates.shape == gatherAxisResultShape(data.shape, indices.shape, axis)`; fixed addition produces `data.shape` | `IndexAxisAttrs` |
| `SCATTER_ELEMENTS` | `[data, indices, updates]` | indices and updates have equal rank/Shape, match data off-axis, and reduce into `data.shape` | `ScatterElementsAttrs` |

The Javadocs must state that the table explains meaning rather than performing validation. They
must distinguish the three kinds from axis gather, Gather-ND, scatter-ND, and in-place mutation.

### Reduction vocabulary

Create exactly:

```java
public enum ScatterReduction {
    NONE,
    ADD,
    MUL,
    MAX,
    MIN
}
```

The enum declares no project field, explicit constructor, method, nested type, per-constant body,
alias, identity value, operator function, numeric type, or backend metadata.

Document the conceptual role of each value:

- `NONE` replaces the base value at each addressed target and requires target coordinates within
  one operation to be unique; duplicate targets are invalid and are not resolved by update order.
- `ADD` combines the base value and all updates addressed to a target by addition.
- `MUL` combines the base value and all updates addressed to a target by multiplication.
- `MAX` combines the base value and all updates addressed to a target by maximum.
- `MIN` combines the base value and all updates addressed to a target by minimum.

These descriptions define semantic operators, not execution algorithms, accumulation order,
numeric edge behavior, supported DataTypes, or backend capabilities.

### Scatter-elements attributes

Create exactly:

```java
public record ScatterElementsAttrs(int axis, ScatterReduction reduction)
        implements OperationAttrs
```

The record has exactly two components in that order, one canonical constructor, two explicit
documented accessors, and record-generated `equals`, `hashCode`, and `toString`. Add no rank,
selected extent, raw axis, normalization flag, input, Shape, result, default constructor, factory,
builder, nested type, or extra state/API.

Constructor validation order and behavior are exact:

1. if `axis < 0`, throw `IllegalArgumentException` with exact message
   `axis must be non-negative: <axis>`;
2. require non-null `reduction` with `Objects.requireNonNull(reduction, "reduction")`;
3. otherwise retain both values unchanged.

Zero and `Integer.MAX_VALUE` axes are structurally valid because no input rank is present. Every
enum reduction is structurally valid. Null never means `NONE`.

### Fixed-add attribute reuse

Compose both fixed-add kinds with the existing normalized-axis record:

```java
IndexAxisAttrs axis = new IndexAxisAttrs(1);
Operation scatterAdd = new Operation(AxisScatterKind.SCATTER_ADD, axis);
Operation scatterAxisAdd = new Operation(AxisScatterKind.SCATTER_AXIS_ADD, axis);
```

Do not add reduction state to either composition. Update `IndexAxisAttrs` Javadoc only so it names
the completed axis-gather and fixed-add axis-scatter pairings. Its record declaration, constructor,
validation, accessor bytecode, value semantics, and behavior must remain unchanged.

### Configurable composition

Compose scatter-elements explicitly:

```java
ScatterElementsAttrs attrs =
        new ScatterElementsAttrs(1, ScatterReduction.ADD);
Operation scatterElements =
        new Operation(AxisScatterKind.SCATTER_ELEMENTS, attrs);
```

The exact attributes reference is retained. Generic `Operation` does not enforce family pairing.
Add no operation factory, default reduction, compatibility validator, or matrix.

### Concrete Shape examples

Javadocs and focused test terminology must use these non-executable examples:

- `SCATTER_ADD`: data `[2, 3, 4]`, axis `1`, indices `[2, 4]`, and updates `[2, 4]` produce result
  `[2, 3, 4]`. At reduced coordinate `[0, 2]`, index `1` adds update `[0, 2]` to data coordinate
  `[0, 1, 2]`.
- `SCATTER_AXIS_ADD`: data `[2, 3, 4]`, axis `1`, indices `[5, 6]`, and updates `[2, 5, 6, 4]`
  produce result `[2, 3, 4]`. Update `[0, i, j, 2]` targets data
  `[0, indices[i, j], 2]`.
- `SCATTER_ELEMENTS`: data `[2, 3, 4]`, axis `1`, and equal indices/updates Shapes `[2, 5, 4]`
  produce result `[2, 3, 4]`. At update coordinate `[0, 4, 2]`, the corresponding index chooses
  the middle coordinate of data target `[0, index, 2]`; the selected reduction determines how the
  update combines with the base and any repeated target.

No production code in this task stores operands, computes Shapes, reads indices, or executes these
examples.

## Affected files

Expected implementation change:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/AxisScatterKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/ScatterReduction.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/ScatterElementsAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/IndexAxisAttrs.java`
  — Javadoc only
- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/index/AxisScatterSemanticsTest.java`
- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task file
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum scope

This task may create or modify at most the ten paths listed above.

If implementation requires another production type, another test, behavioral changes to existing
Java, public Tensor behavior, Compile API change, capability-baseline edit, dependency, build
change, architecture document, another module, or more than ten paths, stop and report the issue.

## Javadoc requirements

- Document every public type, enum constant, record component/accessor, and canonical constructor.
- Define functional scatter, base data, indices, updates, target coordinate, duplicate target, and
  reduction before relying on those terms.
- Explain `[data, indices, updates]` order and exact result-Shape preservation.
- Explain all three distinct Shape relationships with the concrete examples above.
- Explain normalized axis meaning and why rank validation is deferred.
- Explain fixed-add kinds versus configurable scatter-elements reduction.
- Explain every reduction without promising an execution algorithm or backend behavior.
- State that `NONE` duplicate targets are invalid but value-aware detection occurs after model
  metadata construction.
- State that only task 0018H will validate index types, input/update DataTypes, Shapes, and public
  caller axes.
- Distinguish axis scatter from axis gather, Gather-ND, scatter-ND, fold, and in-place mutation.
- Do not promise gradients, compiler capture, backend support, numerical execution, index bounds,
  materialization, or a particular accumulation order.

## Acceptance criteria

- `AxisScatterKind` is a public enum implementing `OperationKind` with exactly `SCATTER_ADD`,
  `SCATTER_AXIS_ADD`, and `SCATTER_ELEMENTS`, in that order.
- `ScatterReduction` is a public enum with exactly `NONE`, `ADD`, `MUL`, `MAX`, and `MIN`, in that
  order.
- Neither enum adds project-declared state, methods, constructors, nested types, aliases, or
  metadata.
- `ScatterElementsAttrs` is a public record implementing `OperationAttrs` with exactly `int axis`
  and `ScatterReduction reduction`, in that order.
- Axis and null-reduction failures use the exact validation order, exception types, and messages;
  valid values are unchanged.
- Record-generated value semantics remain the object contract and both accessors are documented.
- Both fixed-add kinds compose with unchanged `IndexAxisAttrs`; scatter-elements composes with
  `ScatterElementsAttrs` for every exact reduction.
- `IndexAxisAttrs` declaration and behavior remain bytecode-equivalent; only its Javadoc may
  change.
- Javadocs preserve ordered inputs, functional result meaning, Shape distinctions, examples,
  reduction meanings, duplicate boundary, and deferred validation.
- No Tensor, DataType, Shape, descriptor, provenance, graph, gradient, compiler, runtime, backend,
  or execution behavior is introduced.
- Tensor API, glossary, task evidence, master plan, and roadmap are independently reviewed and
  synchronized. Compile API, Training API, capabilities, architecture, and related contracts
  receive reasoned no-change conclusions.
- All validation passes and the final diff contains exactly the ten permitted paths.

## Tests / validation

Run:

```bash
./gradlew :modules:model:test --tests \
  io.github.pho001.synaptik.model.operation.index.AxisScatterSemanticsTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused test must verify:

- exact kind-enum constant count, names, order, interface, and absence of project API/state;
- exact reduction-enum constant count, names, order, and absence of project API/state;
- exact record status, component names/types/order, interface, accessors, and absence of extra
  state/API;
- zero, ordinary, and `Integer.MAX_VALUE` axis retention with every reduction;
- exact negative-axis-first validation and message, followed by exact null-reduction failure and
  message;
- record-generated equality, hashing, and diagnostic text;
- exact composition of fixed-add kinds with one `IndexAxisAttrs` reference;
- exact composition of `SCATTER_ELEMENTS` with one `ScatterElementsAttrs` reference for every
  reduction;
- kind identity distinctions from each other, axis-gather kinds, and Gather-ND;
- absence of scatter-ND, gradient-only, gather-gradient, alias, or default enum constants;
- absence of Tensor, DataType, Shape, layout, graph, compiler, runtime, and backend dependencies.

Manually inspect `javap -p -c -s`, reflection/source/imports, generated Javadoc, Markdown links,
anchors, fences, whitespace, exact ten-path scope, synchronized task/master/roadmap status,
bytecode-equivalent `IndexAxisAttrs`, and absence of a task-0018H specification.

## Dependencies

- Task 0005 defines the minimal operation-kind and typed-attributes contracts.
- Task 0006 defines the open immutable `Operation` pair.
- Task 0018C defines `IndexAxisAttrs` and the three gather meanings that the fixed-add scatter
  kinds invert at the semantic level.
- Task 0018D supplies the completed public gather Shape terminology used to explain the inverse
  relationships without adding those public methods here.

## Follow-up tasks

- 0018H: public scatter-add, scatter-axis-add, and scatter-elements Tensor expressions.
- 0018I: scatter-ND semantics reusing `ScatterReduction` with batch-dimension parameters.

Do not create either follow-up specification during this task.

## Architecture impact

Expected impact: None.

The task fills the existing model-owned operation vocabulary. If implementation requires a new
architecture rule or cross-module dependency, stop and report the issue.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0005/0006/0018C/0018D/0018E/0018F/0018G, Tensor API,
Compile API, Training API, glossary, current OperationKind/OperationAttrs/Operation/IndexAxisAttrs/
AxisGatherKind/GatherNdKind and related index-family contracts/tests, and Java 26 Gradle.

Implement task 0018G exactly. Add only AxisScatterKind.java, ScatterReduction.java,
ScatterElementsAttrs.java, and AxisScatterSemanticsTest.java for new Java files under
io.github.pho001.synaptik.model.operation.index. Permit Javadoc-only changes to IndexAxisAttrs;
its declaration and behavior must remain bytecode-equivalent.

AxisScatterKind contains exactly SCATTER_ADD, SCATTER_AXIS_ADD, and SCATTER_ELEMENTS in order.
ScatterReduction contains exactly NONE, ADD, MUL, MAX, and MIN in order. ScatterElementsAttrs
contains exactly normalized non-negative int axis then non-null ScatterReduction reduction, with
the exact validation order/messages and explicit documented accessors. Fixed-add kinds pair with
IndexAxisAttrs; configurable scatter-elements pairs with ScatterElementsAttrs. Document ordered
[data, indices, updates], functional data-shaped results, three distinct Shape relationships,
fixed-add versus configurable reduction, all reduction meanings, and invalid NONE duplicates.

Do not add Tensor methods, Shape/DataType/result/provenance validation, scatter-ND/gradient types,
value execution, numeric policy, factories, graph/compiler/planning/runtime/backend behavior,
dependencies, build or architecture changes, behavioral existing-Java edits, or later specs. Stop
beyond ten paths or on architecture uncertainty.

Run all specified validation, then hand the actual diff/evidence to a separate clean-context docs
agent in the same change. It must inspect source/tests/generated Javadoc, finalize permitted
Javadocs/Tensor API/glossary/planning, record Compile API/Training API/capability/architecture and
related-contract no-change conclusions, and rerun validation.

Update task 0018G, model master plan, and roadmap only for planning status/evidence. Do not mark
Complete until both passes succeed. Leave 0018H Draft without a specification. Do not commit/push.
```

## Local decisions

- Three enum constants preserve distinct update-alignment contracts rather than treating every
  axis scatter as one operation plus a mode flag.
- `SCATTER_ADD` and `SCATTER_AXIS_ADD` reuse `IndexAxisAttrs` because both need only an already-
  normalized target axis; their fixed addition is intrinsic to their kind names.
- `SCATTER_ELEMENTS` receives `ScatterElementsAttrs` because its reduction is caller-selected
  semantic state. Null is rejected rather than silently becoming `NONE`.
- `ScatterReduction` is a separate reusable enum because both scatter-elements and later
  scatter-ND expose the same selected replacement/reduction vocabulary.
- `NONE` means unambiguous replacement. Duplicate target coordinates are invalid rather than
  inheriting traversal-order-dependent last-write behavior.
- Ordered `[data, indices, updates]` roles and Shape relationships are semantic documentation, not
  stored arity, operand, or inference state. Generic `Operation` remains open.

## Known limitations

- The semantic values do not normalize raw caller axes or validate data/indices/update ranks,
  types, Shapes, bounds, duplicate target coordinates, dynamic Dimensions, result descriptors, or
  gradient eligibility. Task 0018H owns the locally decidable public expression rules.
- The reduction vocabulary defines mathematical combination choices but not numerical evaluation
  order, floating-point reproducibility, overflow, NaN/signed-zero handling, atomic execution, or
  backend support.
- No Tensor method, provenance, storage, graph capture, compiler transformation, gradient rule,
  ONNX mapping, planning, materialization, lowering, runtime behavior, or execution is implemented.
- Scatter-ND remains the separate later semantic/expression pair 0018I–0018J.

## Validation evidence

- Clean implementation context `/root/implement_model_0018g` added exactly `AxisScatterKind`,
  `ScatterReduction`, `ScatterElementsAttrs`, and `AxisScatterSemanticsTest`, plus the authorized
  Javadoc-only `IndexAxisAttrs` update. Independent documentation context
  `/root/implement_model_0018g/review_model_0018g_docs` inspected the actual shared-tree diff,
  final source/test, generated Javadoc, related contracts, APIs, glossary, planning state, and Java
  26 build configuration before finalizing documentation in the same overall change.
- The documentation pass applied General plus API/Javadoc style to production Javadocs, Tensor API,
  and glossary; Planning style to this task, the model master plan, and roadmap; and Example format
  to the conceptual Shape/coordinate examples. It retained the three new production Javadocs as
  complete and refined only `IndexAxisAttrs` wording to distinguish the current gather boundary
  from task-0018H public scatter validation. No declaration, test, or executable behavior changed.
- Reviewed architecture and process material included `AGENTS.md`, `ARCHITECTURE.md`, the focused
  current architecture, overview, lifecycle, module-boundary, dependency, and
  runtime/prepare/backend explanations; documentation rules and General/API-Javadoc/Planning/
  Example profiles; planning guide and roadmap; model capabilities/master plan; tasks 0005, 0006,
  and 0018C–0018G; Tensor, Compile, and Training API references; glossary; Java 26 root/model Gradle
  configuration; final implementation/tests; and generated model Javadoc.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.operation.index.AxisScatterSemanticsTest` — `BUILD SUCCESSFUL`;
  XML reports 12 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:test` — `BUILD SUCCESSFUL`; XML aggregation reports 706 tests across
  82 suites with zero failures, errors, or skips.
- `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL`; generated public pages contain all three
  new types, every enum constant, the record components, canonical constructor, explicit accessors,
  exact failures, ordered roles, functional result meaning, Shape examples, reduction meanings,
  duplicate-target boundary, task-0018H ownership, family distinctions, and exclusions.
- `./gradlew test` — `BUILD SUCCESSFUL`; the repository lifecycle reports 36 actionable tasks with
  no failing task.
- `javap -classpath modules/model/build/classes/java/main -p -c -s` confirmed exact enum order and
  generated-only enum machinery. `ScatterElementsAttrs` has exactly the two specified fields,
  negative-axis-first then non-null-reduction validation, direct accessors, and generated record
  methods. Diffing current `IndexAxisAttrs` output against `/tmp/index-axis-before.javap` produced
  no output, proving declaration and executable-bytecode equivalence after its Javadoc-only edit.
- Focused reflection, source, and import inspection confirmed exact packages, modifiers,
  interfaces, constants, record components/methods, validation messages, value semantics,
  kind/attributes compositions, and only the permitted local operation/JDK imports. No Tensor,
  DataType, Shape, layout, provenance, graph, compiler, planning, prepare, runtime, backend,
  training, or execution dependency or behavior was introduced.
- Generated-page inspection and a targeted Markdown validator passed for all changed documentation:
  392 local links, including 100 heading anchors, resolved with zero errors. Fences are balanced,
  trailing-whitespace scans found no matches, all ten paths have final newlines, and
  `git diff --check` passes.
- Final changed-path inventory contains exactly the ten authorized paths: three new production
  contracts, Javadoc-only `IndexAxisAttrs`, one focused test, Tensor API, glossary, this task,
  model master plan, and roadmap. Task/master-plan/roadmap status is synchronized as Complete.
  Task 0018H remains Draft, and no task-0018H specification exists.
- Compile API remains accurate unchanged because this task adds semantic vocabulary only: it adds
  no Tensor expression, graph capture, inference, validation, optimization, artifact, or engine
  behavior. Training API remains accurate unchanged because no gradient, autograd, parameter,
  optimizer, publication, session, or training execution behavior changed. The capability
  baseline already inventories all three axis-scatter operations, the five reductions, exact
  integral index types, and separate support layers, so it required no edit.
- Existing Operation foundations, aggregate/scan reductions, fold, scalar select, axis gather,
  Gather-ND, Tensor, provenance, graph, and other operation-family contracts remain accurate
  unchanged because the new values compose the open semantic contracts without altering their
  declarations or behavior. Scatter-ND and later tasks remain deferred.
- `ARCHITECTURE.md`, focused architecture explanations, ADRs, architecture tests,
  backend-conformance tests, integration tests, Gradle/dependencies, other modules, and later task
  specifications remain accurate unchanged because this task stays within existing model
  ownership and changes no dependency rule, backend behavior, executable end-to-end behavior, or
  build requirement.

## Implementation notes

- Added `AxisScatterKind` with exact ordered `SCATTER_ADD`, `SCATTER_AXIS_ADD`, and
  `SCATTER_ELEMENTS` constants and no project-declared behavior or metadata.
- Added reusable `ScatterReduction` with exact ordered `NONE`, `ADD`, `MUL`, `MAX`, and `MIN`
  values, including unambiguous replacement semantics for `NONE`.
- Added `ScatterElementsAttrs(int axis, ScatterReduction reduction)` with exact validation order,
  direct accessors, and generated record value semantics. Both fixed-add kinds reuse unchanged
  `IndexAxisAttrs`.
- Added 12 focused tests covering exact structure, retained values, exact failures, value
  semantics, typed composition, family distinctions, and dependency boundaries.
- Finalized production Javadocs, Tensor API semantic reference, glossary terminology/inventories,
  and synchronized planning evidence/status through the independent documentation pass.

## Completion summary

- Completed changes: Implemented and documented three functional axis-scatter semantic identities,
  reusable five-way reduction meaning, and explicit scatter-elements attributes without public
  Tensor or cross-layer behavior.
- Files changed or created: Exactly the ten authorized production, test, API, glossary, task,
  master-plan, and roadmap paths.
- Tests and validation: Focused 12-test and all 706-model-test/82-suite runs, model Javadoc, root
  tests, javap/reflection/import/source/generated-page review, 392-link/100-anchor checks,
  fence/whitespace/newline checks, exact scope/status and no-0018H-spec checks,
  `IndexAxisAttrs` bytecode equivalence, and `git diff --check` passed.
- Documentation-agent review: Clean context
  `/root/implement_model_0018g/review_model_0018g_docs` completed the mandatory independent pass
  with General, API/Javadoc, Planning, and Example profiles.
- Documentation impact: Tensor API and glossary now define current functional axis-scatter
  vocabulary while public Tensor construction, type/Shape/axis validation, index bounds,
  duplicate detection, gradients, compiler behavior, lowering, backend behavior, and execution
  remain separately owned.
- Javadoc review: All three new public types, every constant, the attributes record components,
  canonical constructor, and accessors are complete. `IndexAxisAttrs` received only an ownership-
  boundary clarification and remains bytecode-equivalent.
- Glossary impact: Added reusable axis-scatter, base/indices/updates, target, duplicate-target, and
  reduction terminology and synchronized current `OperationKind`/`OperationAttrs` inventories.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0018G. Task 0018H remains Draft without a detailed
  specification.

Status: Complete
