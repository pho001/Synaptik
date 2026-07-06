# Task 0017K: Tensor Composition Semantics

## Status

Complete

## Goal

Define typed, backend-independent semantic identities and immutable normalized axis attributes for
concatenation, stacking, and individually identified unstack outputs.

Concat joins ordered input tensors along an existing axis. Stack joins ordered same-shaped inputs
along one newly inserted axis. A public unstack request returns multiple tensors; with the current
TensorProvenance contract, each result is represented as one indexed UNSTACK output operation.
This task defines meaning and intrinsic parameters only. Public Tensor methods, input validation,
Shape derivation, result collection construction, provenance, graph capture, gradients,
materialization, and execution remain later responsibilities.

## Scope

- Add one public `TensorCompositionKind` enum implementing OperationKind with exactly CONCAT,
  STACK, and UNSTACK in that order.
- Add one public `CompositionAxisAttrs` record implementing OperationAttrs with exactly one
  non-negative normalized int axis.
- Use CompositionAxisAttrs for CONCAT existing-axis meaning and STACK inserted-output-axis meaning.
- Add one public `UnstackOutputAttrs` record implementing OperationAttrs with exactly normalized
  int axis and non-negative int outputIndex, in that order.
- Define UNSTACK as one indexed output from a logical multi-result unstack request, so separate
  public result Tensors remain distinguishable in current one-provenance-per-Tensor metadata.
- Document ordered input roles and exact kind/attribute pairings without adding generic validators.
- Keep input lists, input count, output count, Shapes, descriptors, and grouping identity out of
  semantic attributes.
- Add one focused same-package test for all three production contracts.
- Keep production in existing `model.operation.layout` package.
- Finalize Javadocs, Tensor API semantic reference, glossary, task evidence, master plan, and
  roadmap through the mandatory independent documentation pass.

## Out of scope

- public Tensor.concat, Tensor.stack, Tensor.unstack, another Tensor method, helper, factory,
  result collection, or task-0017L implementation
- deciding whether unstack returns array, List, stream, iterator, or another collection surface
- adding output index/group ID to TensorProvenance, changing TensorFactory, adding a multi-output
  public factory, or changing current graph records
- requiring one CompiledNode with multiple outputs, grouping separately created outputs into one
  node, allocating NodeId/ValueId, or compiler graph capture
- storing ordered input Tensors, input count, output count, rank, Shape, DataType, layout,
  descriptor, requiresGrad, label, provenance, storage, or backend facts in attributes
- concat compatibility, stack identical-shape checks, axis normalization, insertion-axis rules,
  unstack static-axis/output-count validation, output Shape derivation, or overflow
- implementing stack as expandDims+concat or unstack as select; decomposition/canonicalization is a
  later compiler choice, not the semantic contract
- gradients, concat split, stack split, unstack scatter, autograd, compiler-generated operations,
  optimizer, or training behavior
- graph/compiler/planning/prepare/runtime/backend/engine/trace/ONNX/conformance behavior
- factories, registries, parsers, maps, string dispatch, reflection discovery, arity/result/cost/
  fusion/backend-support metadata, dependencies, Gradle, architecture changes, existing Java edits,
  another module, or task-0017L specification

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
- [Task 0005](0005-operation-semantic-foundation.md)
- [Task 0006](0006-operation-model.md)
- [Task 0008](0008-graph-value-and-node-model.md)
- [Task 0013](0013-tensor-provenance-skeleton.md)
- [Task 0017E](0017e-axis-transform-semantics.md)
- [Task 0017I](0017i-pad-and-tile-semantics.md)
- [Task 0017J](0017j-pad-and-tile-tensor-expressions.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The baseline requires concat, stack, and unstack. Legacy exposes:

```java
static Tensor concat(int axis, Tensor... inputs)
static Tensor stack(int axis, Tensor... inputs)
Tensor[] unstack(int axis)
```

Legacy CONCAT is a first-class n-ary operation with one normalized existing axis. Legacy stack
normalizes an insertion position, applies expandDims to every input, and delegates to concat.
Legacy unstack normalizes an existing axis and creates one select result for every coordinate.
Those decompositions prove capability but do not require the new semantic model to erase STACK or
UNSTACK names.

Current CompiledNode can represent multiple output values, but current public TensorProvenance has
only Operation plus ordered inputs: it has neither producer-group identity nor output slot. Adding
those concepts would exceed this task and change a completed public contract. The initial model
therefore gives every unstack result exact UNSTACK semantics with its own outputIndex. A future
compiler may keep those operations separate or introduce an explicit grouped multi-output capture
design through a separately approved task.

Legacy graph builders, mutable arrays, storage/materialization, gradient callbacks, traits,
compiler/lowering code, kernels, and runtime/backend behavior are not copied.

## Architecture constraints

- Production remains in modules/model, which owns backend-neutral operation semantics.
- TensorCompositionKind describes meaning only, not an occurrence, Tensor, graph node, executable,
  backend route, or physical data movement.
- CompositionAxisAttrs stores one already normalized non-negative axis but no rank. CONCAT and
  STACK interpret that position differently through their kind.
- UnstackOutputAttrs stores enough intrinsic information to distinguish one output Tensor using
  current provenance: normalized source axis plus coordinate/output index on that axis.
- UNSTACK outputIndex is not a Tensor array index object, graph ValueId, NodeId, output count,
  producer-group ID, storage offset, or runtime slot.
- Input order for concat/stack is semantic but belongs to later TensorProvenance, not attributes.
- Generic Operation remains an open kind/attributes pair and does not validate pairings, arity,
  rank, Shapes, output count, or backend support.
- CompiledNode multi-output support remains unchanged. This task neither requires nor forbids a
  later grouped compiler representation.
- Package direction is model.operation.layout -> model.operation + java.base only.
- Stop if implementation needs Tensor/provenance/graph changes, another type, dependency, or
  architecture decision.

## Package impact

No package is added or moved.

- `io.github.pho001.synaptik.model.operation.layout.TensorCompositionKind` — three composition
  semantic identities.
- `io.github.pho001.synaptik.model.operation.layout.CompositionAxisAttrs` — normalized concat or
  stack axis position.
- `io.github.pho001.synaptik.model.operation.layout.UnstackOutputAttrs` — normalized source axis
  and individually identified output coordinate.
- `TensorCompositionSemanticsTest` — same-package focused structural/composition test.

Composition operations transform logical axes and remain cohesive with existing layout-operation
semantics. No provenance or graph package change is introduced.

## Required contract

### Semantic kind vocabulary

Create exactly:

```java
public enum TensorCompositionKind implements OperationKind {
    CONCAT,
    STACK,
    UNSTACK
}
```

The enum adds no project field, explicit constructor, method, nested type, per-constant body,
alias, arity, input/output count, axis, Shape, result, layout, or backend metadata.

Document meanings:

| Kind | Semantic inputs and result | Attributes |
|---|---|---|
| CONCAT | ordered non-empty inputs joined along one existing normalized axis; rank is preserved | CompositionAxisAttrs |
| STACK | ordered non-empty inputs joined along one newly inserted normalized result axis; rank increases by one | CompositionAxisAttrs |
| UNSTACK | one indexed result obtained by fixing one normalized source axis to outputIndex and removing that axis | UnstackOutputAttrs |

No kind executes, validates input count, or derives Shape.

### Shared concat/stack axis attributes

Create exactly:

```java
public record CompositionAxisAttrs(int axis) implements OperationAttrs
```

It has one component, one canonical constructor, one explicit documented accessor, and generated
object methods. Reject negative axis with IllegalArgumentException and exact message:

```text
axis must be non-negative: <axis>
```

Retain every non-negative int unchanged, including Integer.MAX_VALUE. The record has no rank and
cannot decide whether axis is an existing CONCAT axis or STACK insertion position.

### Indexed unstack-output attributes

Create exactly:

```java
public record UnstackOutputAttrs(int axis, int outputIndex) implements OperationAttrs
```

It has exactly two components in that order, one canonical constructor, two explicit documented
accessors, and generated object methods. Validation order is exact:

1. reject negative axis with exact message `axis must be non-negative: <axis>`;
2. reject negative outputIndex with exact message
   `outputIndex must be non-negative: <outputIndex>`.

Retain all non-negative values unchanged. The record knows no input rank or selected-axis extent,
so task 0017L must verify both bounds before constructing it. Integer.MAX_VALUE is structurally
valid for either component.

outputIndex identifies the fixed logical coordinate on the removed source axis and therefore the
specific output from one public unstack request. It does not identify a graph output slot or
promise grouping with other UNSTACK values.

### Typed composition

Document exact valid pairings:

```java
Operation concatenated = new Operation(TensorCompositionKind.CONCAT, axisAttrs);
Operation stacked = new Operation(TensorCompositionKind.STACK, axisAttrs);
Operation unstackedOutput =
        new Operation(TensorCompositionKind.UNSTACK, unstackOutputAttrs);
```

Operation retains exact references. Do not use NoOperationAttrs, pair UNSTACK with
CompositionAxisAttrs, add a compatibility validator/factory, or modify Operation.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/TensorCompositionKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/CompositionAxisAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/UnstackOutputAttrs.java`

Test:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/layout/TensorCompositionSemanticsTest.java`

Documentation/planning:

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless inconsistency requires stopping: Compile API, Training API,
capabilities, Operation/TensorProvenance/graph/Shape/layout/Tensor contracts, existing semantic
families/tests, architecture/ADRs/tests, conformance/integration tests, Gradle, and other modules.

## Maximum scope

At most three production files, one focused test, and five documentation/planning files: nine
paths. If another Java type/test, existing Java edit, provenance/graph change, dependency,
build/architecture change, or tenth path is required, stop and report. Do not create task 0017L.

## Javadoc requirements

- Document enum/type/constants, both records, constructors, components, and explicit accessors.
- Explain ordered concat with a concrete Shape example and the difference between an existing axis
  and stack insertion position.
- Explain stack as first-class semantics even though it can later decompose to expand+concat.
- Explain unstack with Shape [2,3,4], axis 1, and output indices 0..2, each yielding conceptual
  Shape [2,4].
- Explain why outputIndex is required by current one-provenance-per-Tensor construction and why it
  is not a graph output slot/group ID.
- Explain validation, normalized values, deferred rank/output-count checks, and extreme ints.
- Explain no Tensor, input list, Shape, descriptor, layout, provenance, graph grouping, gradient,
  compiler, backend, ONNX, or execution behavior.
- Review Operation, TensorProvenance, CompiledNode, axis-transform, and Shape terminology Javadocs;
  record unchanged reasons or stop on discrepancy.

## Acceptance criteria

- Enum is exact public three-constant ordered vocabulary with no extra project API/state.
- CompositionAxisAttrs and UnstackOutputAttrs have exact record shapes/accessors and validation.
- Zero and positive values are accepted; negative values fail with exact type/message/precedence.
- Extreme non-negative ints and record value semantics work unchanged.
- Exact kind/attributes pairings retain references; no generic validator/factory is added.
- UNSTACK output semantics are distinguishable by outputIndex without provenance/graph changes.
- No Tensor/input/Shape/layout/provenance/gradient/cross-layer behavior or existing Java edit.
- Complete Javadocs, Tensor API, glossary, independent docs review, evidence, and status
  synchronization finish before marking Complete.

## Tests / validation

Run before and after documentation review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.layout.TensorCompositionSemanticsTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

Focused tests cover exact enum order/API; exact record components/order/types/API; zero, ordinary,
and extreme values; every validation failure/precedence; record value semantics; explicit typed
Operation composition/reference retention; concat/stack shared attrs; distinct indexed UNSTACK
outputs; no grouping/graph fields; and no extra members or forbidden dependencies.

Inspect javap, reflection, imports, bytecode, and source. Confirm three production types, exact
enum/records, no stored derived/helper state, and no Tensor/Shape/layout/provenance/graph/compiler/
planning/prepare/runtime/backend/ONNX/training types. Validate generated Javadoc, API/glossary,
links/anchors/fences/whitespace, exact nine paths, synchronized statuses, and no task-0017L spec.

## Dependencies

- 0005 supplies OperationKind and OperationAttrs.
- 0006 supplies immutable generic Operation composition.
- 0002 supplies axis/Shape terminology but is not a production dependency.
- 0008 confirms graph nodes may have multiple outputs without requiring public provenance grouping.
- 0013 confirms current TensorProvenance intentionally has no output index/group identity.

## Follow-up tasks

- 0017L remains Draft for public concat/stack/unstack APIs, ordered input validation, Shape/type/
  eligibility rules, output collection, descriptors, and provenance.
- Any grouped public multi-output provenance/capture design requires a separate explicit task and
  must not be hidden inside 0017L.
- Compiler later owns stack decomposition, concat/unstack capture, canonicalization, and backward
  graph construction.
- Planning/backend prepare own materialization, lowering, and kernels; ONNX extension owns mapping.

Do not create a detailed follow-up specification here.

## Architecture impact

Expected impact: None. The design uses current Operation and one-provenance-per-Tensor contracts
without modifying module boundaries or graph records.

If implementation needs producer grouping, graph output slots, TensorProvenance changes, another
dependency, or architecture changes, stop and report.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0002/0005/0006/0008/0013/0017K, Tensor API, Compile API,
Training API, glossary, current OperationKind/OperationAttrs/Operation/TensorProvenance/
CompiledNode and layout-operation semantic contracts/tests, Shape terminology, and Java 26 Gradle.

Implement task 0017K exactly. Add only TensorCompositionKind.java, CompositionAxisAttrs.java,
UnstackOutputAttrs.java, and TensorCompositionSemanticsTest.java under
io.github.pho001.synaptik.model.operation.layout.

The enum contains exactly CONCAT, STACK, UNSTACK. CompositionAxisAttrs contains exactly one
non-negative normalized int axis and pairs with CONCAT/STACK. UnstackOutputAttrs contains exactly
axis then outputIndex, validates both non-negative in order, and pairs only with UNSTACK. Document
UNSTACK as one individually indexed output from a public multi-result request, distinguishable by
current Tensor provenance without producer grouping or graph output-slot state.

Do not add Tensor methods, input lists/counts, Shape/result/layout/provenance/graph behavior,
multi-output grouping, decomposition, gradients, compiler/planning/runtime/backend/ONNX behavior,
factories, dependencies, build/architecture changes, existing Java edits, or later specs. Stop
beyond nine paths or on architecture uncertainty.

Run all specified validation, then hand actual diff/evidence to a separate clean-context docs agent
in the same change. It must inspect source/tests/generated Javadoc, finalize permitted Javadocs/
Tensor API/glossary/planning, record no-change conclusions, and rerun validation.

Update 0017K, model master plan, and roadmap only for status/evidence. Do not mark Complete until
both passes succeed. Leave 0017L Draft without a specification. Do not commit/push.
```

## Local decisions

- CONCAT, STACK, and UNSTACK are distinct first-class meanings even when compiler decomposition is
  possible.
- CONCAT and STACK share one normalized axis record; their kind supplies existing-axis versus
  inserted-axis interpretation.
- Each UNSTACK result stores outputIndex because current TensorProvenance cannot distinguish
  outputs of one grouped producer.
- The initial design does not group results into one multi-output CompiledNode. This preserves
  capability without changing completed provenance/graph contracts; future grouping needs an
  explicit task.
- Attributes store no input/output count or collection state; task 0017L owns local validation.

## Known limitations

- No public composition APIs, input validation, result collection, Shape/descriptor/provenance,
  grouped multi-output capture, gradients, compiler behavior, materialization, lowering,
  execution, or ONNX mapping exists.

## Validation evidence

- Architecture/focused boundary docs, documentation/planning rules, roadmap, model capabilities and
  master plan, Operation/TensorProvenance/CompiledNode/Shape contracts and tests, neighboring
  semantic tasks, Tensor/Compile/Training APIs, glossary, Java 26 build, and read-only legacy
  concat/stack/unstack builders/tests were reviewed.
- Legacy evidence confirms first-class n-ary concat plus stack decomposition through expand/concat
  and unstack decomposition through indexed select outputs. Current provenance has no output slot
  or producer grouping, while CompiledNode independently permits multiple outputs.
- Planning resolved that boundary without contract changes: every initial UNSTACK output carries
  its own normalized axis and outputIndex; grouped multi-output capture remains explicitly
  deferred rather than silently invented.
- `git diff --check` passed with no whitespace errors.
- Changed-path inventory contains exactly three planning paths: this task, model master plan, and
  roadmap. No Java, API, glossary, Gradle, AGENTS, ARCHITECTURE, focused architecture, completed
  task, or other-module file changed during planning.
- Markdown structure check passed: 23 level-two sections, 16 balanced code-fence markers, no
  trailing whitespace, and final newline present.
- Every local Markdown link in all three changed files resolves.
- Roadmap contains all 74 ordered task rows. Task 0017K is linked and Ready at row 64; task 0017L
  remains Draft at row 65; no task-0017L specification exists.
- Task, master plan, and roadmap consistently identify 0017K as the next Ready frontier.
- Package/scope review confirms three semantic contracts remain in `model.operation.layout`, one
  focused test mirrors that package, and implementation is bounded to nine authorized paths
  without provenance, graph, dependency, or architecture changes.
- No Gradle test was run because this planning-only change modifies no production or test code.
- Implementation context `/root/implement_model_0017k` added exactly
  `TensorCompositionKind`, `CompositionAxisAttrs`, `UnstackOutputAttrs`, and
  `TensorCompositionSemanticsTest`. Clean documentation context
  `/root/implement_model_0017k/review_model_0017k_docs` independently inspected the actual diff,
  source, focused test, generated Javadoc, XML results, bytecode, imports, build configuration,
  APIs, glossary, planning state, and directly related model contracts before finalizing the
  documentation. It applied General plus API/Javadoc style and Example format to Javadoc, Tensor
  API, and glossary review, and Planning style to this task, the model master plan, and roadmap.
- The documentation review retained all three submitted production Javadocs unchanged. They
  already document enum/type/constant meaning, constructor/accessor contracts, ordered concat,
  existing-axis versus stack-insertion semantics, first-class STACK, the conceptual `[2, 3, 4]`
  unstack example, the current one-provenance-per-Tensor reason for `outputIndex`, exact validation
  and extreme values, deferred rank/output-count checks, record semantics, and all required
  Tensor/Shape/layout/provenance/graph/gradient/compiler/backend/ONNX/execution exclusions.
- `docs/api/tensor-api.md` now presents the implemented composition vocabulary and exact kind/
  attributes pairings, explains ordered concat and stack insertion, and gives an individually
  indexed UNSTACK example. `docs/glossary.md` now defines tensor composition and aligns operation-
  kind/attribute inventories and normalized-axis terminology. Both documents state that public
  composition methods, result construction, provenance attachment, grouping, and cross-layer
  behavior remain planned.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.operation.layout.TensorCompositionSemanticsTest` — `BUILD
  SUCCESSFUL`; 11 tests, zero failures, errors, or skips.
- `./gradlew :modules:model:test` — `BUILD SUCCESSFUL`; XML aggregation reports 588 tests with zero
  failures, errors, or skips.
- `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL` without Javadoc errors. Generated pages
  contain all three new contracts, constants, canonical constructors, explicit accessors, exact
  failure messages, examples, current-provenance distinction, and cross-layer exclusions.
- `./gradlew test` — `BUILD SUCCESSFUL`; the repository run reported 36 actionable tasks with no
  failing task.
- `javap -classpath modules/model/build/classes/java/main -p -c -s` confirmed the exact three-value
  enum, one-field and two-field records, constructor signatures, explicit accessors, direct field
  returns, axis-first validation order, exact assignments, and generated record methods. Reflection
  coverage in the focused test independently verifies exact members, visibility, field/component
  order, interfaces, and absence of nested or extra project state.
- Production import inspection found only `OperationKind` or `OperationAttrs`; record component
  state is primitive `int` only. Source, bytecode, and test review found no Tensor, Shape, layout,
  provenance, graph, compiler, planning, prepare, runtime, backend, ONNX, or training dependency or
  behavior.
- Targeted local Markdown target and new-anchor checks passed for all five changed documentation/
  planning files. Code fences are balanced, trailing-whitespace scans found no matches, generated
  Javadoc contains the required rendered contracts, and `git diff --check` passed.
- Final scope contains exactly the authorized nine paths: three new production files, one focused
  test, Tensor API, glossary, this task, model master plan, and roadmap. Task 0017L remains Draft,
  and no task-0017L specification exists.
- `docs/api/compile-api.md` remains accurate unchanged because task 0017K adds no public Tensor
  expression, compiler entry point, capture, inference, grouped graph output, transformation, or
  compile artifact. `docs/api/training-api.md` remains unchanged because no gradient, autograd,
  optimizer, parameter, or training-session behavior changed. The model capability baseline
  already inventories concat, stack, and unstack and correctly separates semantic representation,
  public Tensor construction, compiler, and executable support, so it required no status edit.
- `OperationKind`, `OperationAttrs`, and `Operation` remain accurate because the new types implement
  their open semantic roles without changing generic family compatibility. `TensorProvenance`
  remains accurate because it still stores exactly one Operation and ordered inputs without an
  output slot or producer group. `CompiledNode` remains accurate because its existing ordered
  multi-output capability is neither required nor changed. `Shape`, resolved-layout, Tensor,
  descriptor, axis-transform, slice, pad/tile, and other operation-family contracts remain
  accurate because this task derives no result, layout, Tensor, or provenance and changes no
  neighboring behavior.
- `ARCHITECTURE.md`, focused architecture explanations, ADRs, and architecture tests remain
  unchanged because the work stays inside model-owned backend-neutral semantics and changes no
  module boundary, dependency rule, or lifecycle. Backend-conformance and integration tests remain
  unchanged because there is no backend or end-to-end behavior. Java 26 Gradle configuration and
  other modules remain unchanged because no dependency, build, preview/incubator, or cross-module
  requirement changed.

## Implementation notes

- Added exactly the three typed composition semantic contracts and one focused structural/
  validation/composition test in `model.operation.layout`.
- CONCAT and STACK share one normalized axis attributes value while retaining distinct existing-
  axis and inserted-result-axis meaning. Each UNSTACK output stores its normalized source axis and
  logical output coordinate, so separate future Tensor results remain semantically distinguishable
  without changing provenance or graph contracts.
- Finalized Tensor API, glossary, and synchronized planning status through the independent
  documentation pass. No submitted production Javadoc required correction.

## Completion summary

- Completed changes: Implemented and documented first-class CONCAT, STACK, and individually
  indexed UNSTACK-output semantic values without public Tensor or cross-layer behavior.
- Files changed or created: Three production Java files, one focused test, Tensor API, glossary,
  this task specification, model master plan, and implementation roadmap.
- Tests and validation: Focused 11-test suite, all 588 model tests, generated model Javadoc, full
  repository tests, javap/reflection/import/source checks, generated-page review, Markdown link/
  anchor/fence/whitespace checks, exact-scope/status checks, and `git diff --check` passed.
- Documentation-agent review: Clean context
  `/root/implement_model_0017k/review_model_0017k_docs` completed the required independent pass in
  the same overall change.
- Documentation impact: Tensor API and glossary now explain ordered concat, stack insertion,
  indexed unstack outputs, normalized attributes, provenance constraints, and current-versus-
  planned boundaries. Compile API, Training API, capabilities, architecture/ADRs/tests,
  conformance/integration material, Gradle, and other modules remain accurate unchanged for the
  reasons recorded above.
- Javadoc review: Complete for all three new production types, enum constants, constructors,
  components, accessors, failures, examples, and exclusions; no correction was required.
- Glossary impact: Added implemented tensor-composition terminology and aligned normalized-axis,
  operation-kind, and operation-attributes inventories.
- Unresolved issues: None.
- Follow-up required: None for task 0017K. Task 0017L remains Draft without a detailed
  specification.

Status: Complete
