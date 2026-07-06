# Task 0017I: Pad and Tile Semantics

## Status

Complete

## Goal

Define typed, backend-independent semantic identities and immutable normalized attributes for
constant padding and per-axis tiling.

Padding adds constant-filled logical positions before and after each input axis. Tiling repeats the
complete input pattern a positive number of times along every axis. This task defines operation
meaning and parameter invariants only. Public Tensor methods, Shape/data-type validation, result
descriptors, provenance, gradients, materialization, compiler behavior, and execution remain in
later tasks and layers.

## Scope

- Add one public `PadKind` enum implementing `OperationKind` with exactly `PAD`.
- Add one public `TileKind` enum implementing `OperationKind` with exactly `TILE`.
- Add one public `PadAttrs` record implementing `OperationAttrs` with exactly, in order:
  - `List<Long> before`
  - `List<Long> after`
  - `double constantValue`
- Require equal before/after sizes, non-null elements, and non-negative padding widths.
- Preserve every double constant unchanged without conversion, finiteness checks, or
  normalization.
- Add one public `TileAttrs` record implementing `OperationAttrs` with exactly one
  `List<Long> repeats` component.
- Require non-null repeat elements and strictly positive repeat counts.
- Accept empty padding/repeat lists as rank-zero scalar identity parameters.
- Preserve list order and store immutable snapshots after validation.
- Document exact kind/attribute pairings without adding generic compatibility validation.
- Add one focused same-package semantic test for all four cohesive production contracts.
- Keep production in the existing `model.operation.layout` package.
- Finalize Javadocs, Tensor API semantic reference, glossary, task evidence, model master plan, and
  roadmap through the mandatory independent documentation pass during implementation.

## Out of scope

- public `Tensor.pad`, `Tensor.tile`, another Tensor method, overload, expression helper, factory,
  builder, or task-0017J implementation
- input Tensor, input/output Shape or DataType, rank validation, result dimensions, overflow
  checks, TensorDescriptor, identity, label, provenance, storage, or gradient eligibility
- converting the double padding constant into FLOAT64, FLOAT32, BFLOAT16, integral, or BOOL
  values; range checks, rounding, saturation, BOOL normalization, NaN policy, or infinity policy
- negative padding as cropping, reflect/edge/wrap padding, per-axis constants, tensor-valued
  padding, or another padding mode
- zero/negative tile repeats, implicit leading axes, NumPy-style repeat-rank promotion, partial
  repeat lists, or a separate scalar repeat overload
- layout/view/alias derivation, zero strides, allocation, copying, materialization, storage access,
  value population, or execution
- gradients, pad backward slicing, tile backward reduction, autograd, compiler-generated
  operations, optimizer, or training behavior
- graph capture/canonicalization, planning, prepare, runtime, backend lowering/kernels, engine,
  trace, ONNX mapping, or conformance behavior
- factories, registries, parsers, visitors, maps, string dispatch, reflective discovery, arity,
  result, cost, fusion, backend-support, route, or kernel metadata
- changing existing Java/tests, dependencies, Gradle, architecture, another module, or task-0017J
  specification

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
- [Task 0001](0001-data-type-model.md)
- [Task 0002](0002-shape-and-dimension-model.md)
- [Task 0005](0005-operation-semantic-foundation.md)
- [Task 0006](0006-operation-model.md)
- [Task 0017G](0017g-slice-semantics.md)
- [Task 0017H](0017h-slice-tensor-expressions.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The selected baseline requires constant `pad` and per-axis `tile`. The read-only
`legacy/pre-rewrite` branch exposes:

```java
Tensor pad(int[] before, int[] after, double constantValue)
Tensor tile(int... repeats)
```

Legacy padding requires one non-negative before/after width per input axis, preserves rank, adds
the widths to each result dimension, and fills added positions from one double constant. Legacy
tiling requires one positive repeat per input axis and multiplies each result dimension by its
repeat. Both preserve input data type; their builders also couple graph construction, Shape,
gradients, storage/materialization, and execution concerns.

The new model preserves the semantic capability while using immutable `List<Long>` parameters
compatible with long Shape dimensions. This task stores only normalized intrinsic values. Task
0017J will own public primitive-array/varargs syntax, exact rank, checked Shape arithmetic,
data-type compatibility for the constant, result descriptors, and provenance.

Legacy mutable arrays, immediate graph/storage objects, gradient callbacks, traits,
materialization choices, lowering, kernels, and runtime/backend behavior are not copied.

## Architecture constraints

- Production remains in `modules/model`, which owns backend-neutral operation semantics.
- `PadKind` and `TileKind` identify logical meaning only; they are not Tensors, graph occurrences,
  descriptors, layouts, executables, or backend routes.
- Attributes store normalized intrinsic parameters only, never input rank, Shape, DataType,
  result Shape, layout, storage, provenance, or runtime facts.
- Pad list index `i` couples a before and after width for the same normalized axis position.
- Padding widths are non-negative. The double constant is retained as supplied and receives no
  data-type interpretation in this task.
- Tile list index `i` gives the positive number of complete input-pattern repetitions along axis
  position `i`.
- Empty lists are structurally valid scalar identity parameters. Eventual input-rank matching is
  a task-0017J responsibility.
- Generic `Operation` remains an open kind/attributes pair and does not validate family pairing,
  arity, rank, Shape, data type, gradients, or backend support.
- Package direction is `model.operation.layout -> model.operation + java.base` only.
- Stop if implementation needs DataType/Shape state, another semantic type, existing Java edits,
  a dependency, or an architecture decision.

## Package impact

No package is added or moved.

- `io.github.pho001.synaptik.model.operation.layout.PadKind` — constant-padding identity.
- `io.github.pho001.synaptik.model.operation.layout.PadAttrs` — immutable normalized per-axis
  widths and raw semantic constant.
- `io.github.pho001.synaptik.model.operation.layout.TileKind` — per-axis tiling identity.
- `io.github.pho001.synaptik.model.operation.layout.TileAttrs` — immutable positive repeat counts.
- `PadTileSemanticsTest` — same-package structural, validation, ownership, and composition test.

Both operations transform logical layout/extent and therefore remain cohesive with existing
contiguous, shape-transform, axis-transform, and slice semantics in `model.operation.layout`.

## Required contract

### Semantic kind vocabularies

Create exactly:

```java
public enum PadKind implements OperationKind {
    PAD
}

public enum TileKind implements OperationKind {
    TILE
}
```

Each enum declares no project field, explicit constructor, method, nested type, per-constant body,
alias, arity, parameter, Shape, DataType, result, layout, or backend metadata. Inherited
`Enum.name()` satisfies `OperationKind.name()`.

`PAD` means one logical input with constant-filled positions inserted before/after every axis
without changing rank. `TILE` means one logical input whose complete pattern is repeated along
each axis without changing rank. Neither kind calculates output extents or values.

### Padding attributes

Create exactly:

```java
public record PadAttrs(
        List<Long> before,
        List<Long> after,
        double constantValue) implements OperationAttrs
```

The record has exactly three components in that order, one public canonical constructor, three
explicit documented accessors, and record-generated object methods. Add no arrays, factory,
builder, rank, size method, padding mode, DataType, converted scalar, Shape, cache, nested type, or
extra API/state.

Constructor validation order is exact:

1. null-check `before`, then `after`, with exact component-name messages;
2. reject unequal sizes with `IllegalArgumentException` and exact message
   `before and after must have matching sizes`;
3. inspect entries in ascending index order;
4. at each index null-check before then after with exact messages `before[<index>]` and
   `after[<index>]`;
5. reject negative before with exact message
   `before[<index>] must be non-negative: <value>`;
6. reject negative after with exact message
   `after[<index>] must be non-negative: <value>`;
7. after every entry passes, assign `before = List.copyOf(before)` then
   `after = List.copyOf(after)`, exactly once each.

Perform no validation or normalization of `constantValue`. Retain the supplied primitive exactly,
including finite values, signed zero, infinities, and NaN. The accessor must expose the retained
primitive; data-type conversion is deferred. Record-generated equality/hash semantics remain
ordinary Java record/double semantics and are not redefined as a bitwise serialization contract.

The two lists preserve caller order but not identity. Caller mutation after construction cannot
affect stored values and accessor mutation fails. Empty lists are valid. `Long.MAX_VALUE` padding
is structurally valid because output Shape arithmetic is absent here.

### Tiling attributes

Create exactly:

```java
public record TileAttrs(List<Long> repeats) implements OperationAttrs
```

The record has one component, one public canonical constructor, one explicit documented accessor,
and generated object methods. Add no primitive-array overload, factory, builder, rank, Shape,
cache, nested type, or extra API/state.

Constructor validation order is exact:

1. null-check `repeats` with exact message `repeats`;
2. inspect entries in ascending index order;
3. null-check each element with exact message `repeats[<index>]`;
4. reject zero or negative values with exact message
   `repeats[<index>] must be positive: <value>`;
5. after every entry passes, assign `repeats = List.copyOf(repeats)` exactly once.

Preserve order and values but not caller list identity. Empty list is valid scalar identity;
`Long.MAX_VALUE` is structurally valid. Result Shape multiplication and overflow are deferred.

### Typed composition

Document exactly these valid pairings:

```java
Operation padded = new Operation(PadKind.PAD, padAttrs);
Operation tiled = new Operation(TileKind.TILE, tileAttrs);
```

Operation retains exact kind and attribute references. Do not use `NoOperationAttrs`, pair a kind
with the other family attributes, add compatibility validators/factories, or change Operation.
The valid pairs are documented rather than generically enforced.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/PadKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/PadAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/TileKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/TileAttrs.java`

Test:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/layout/PadTileSemanticsTest.java`

Documentation/planning:

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless inconsistency requires stopping: Compile API, Training API,
capabilities, DataType/Shape/layout/Tensor contracts, Operation foundations, existing semantic
families/tests, architecture/ADRs/tests, conformance/integration tests, Gradle, and other modules.

## Maximum scope

At most four production files, one focused test, and five documentation/planning files: ten paths.
If another Java type/test, existing Java edit, dependency, build/architecture change, or eleventh
path is required, stop and report. Do not create task 0017J.

## Javadoc requirements

- Document both enums, every constant, both records, canonical constructors, components, and all
  explicit accessors.
- Explain constant padding with a concrete one-dimensional example: input `[10, 20]`, before `1`,
  after `2`, constant `-1` semantically requests `[-1, 10, 20, -1, -1]`, without claiming
  execution exists.
- Explain two-dimensional tiling with input `[[1, 2], [3, 4]]` and repeats `[2, 3]`, distinguishing
  complete-pattern repetition from scalar element repeat, without claiming values are computed.
- Explain list ownership, ordering, empty scalar identity, extreme accepted values, and deferred
  rank/Shape arithmetic.
- Explain raw double constant retention and deferred DataType conversion, including signed zero,
  NaN, and infinity.
- Document exact validation order, parameters, results, null failures, and value failures.
- Explain that no Tensor, Shape/result inference, layout, storage, materialization, provenance,
  gradient, compiler, backend, ONNX, or execution behavior is defined.
- Review related Operation, DataType, Shape, SliceAttrs, and layout-semantic Javadocs; record why
  unchanged contracts remain accurate or stop on discrepancy.

## Acceptance criteria

- Both public enums have exactly their specified one constant and no extra project API/state.
- PadAttrs is exactly the specified three-component record; TileAttrs is exactly the specified
  one-component record; neither adds extra API/state.
- Empty and ordinary lists are accepted; caller order/values are retained.
- Null containers/elements, mismatched padding sizes, negative widths, and non-positive repeats
  fail with exact type/message/precedence.
- Pad lists receive exactly two post-validation snapshots; repeats receives exactly one.
- Caller/accessor mutation cannot alter stored values.
- Every double constant and extreme valid long is retained without Shape/DataType interpretation.
- Exact typed Operation compositions retain references; no validator/factory is added.
- No Tensor/Shape/layout/provenance/gradient/cross-layer behavior or existing Java edit appears.
- Complete Javadocs, Tensor API, glossary, independent docs review, evidence, and statuses finish
  before marking Complete.

## Tests / validation

Run before and after documentation review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.layout.PadTileSemanticsTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

Focused tests cover exact enum order/API; exact record components/order/types/API; empty/scalar and
ordinary examples; list ownership/immutability; value semantics; all constant categories including
raw signed-zero/NaN/infinity retention; extreme long values; every validation failure/precedence;
exact Operation pairings/reference retention; absence of cross-family pairing in examples; and no
extra members or forbidden dependencies.

Inspect `javap -p -c -s`, reflection, imports, and source. Confirm four production types, exact
records/enums, two plus one post-validation `List.copyOf` calls, no retained helper state, and no
DataType/Shape/Tensor/layout/storage/graph/compiler/planning/prepare/runtime/backend/ONNX/training
types. Validate generated Javadoc, API/glossary terminology, links/anchors/fences/whitespace,
exact ten paths, synchronized statuses, and no task-0017J specification.

## Dependencies

- 0005 supplies OperationKind and OperationAttrs.
- 0006 supplies immutable generic Operation composition.
- 0001/0002 supply deferred DataType/Shape terminology but are not production dependencies.
- Completed layout/view semantics establish package ownership and semantic/expression separation.

## Follow-up tasks

- 0017J remains Draft for public pad/tile requests, rank validation, array ownership, checked Shape
  arithmetic, constant/data-type policy, result descriptors, provenance, and local layout rules.
- Compiler later owns identity/canonicalization and backward graph construction.
- Planning/backend prepare later own allocation, materialization, lowering, and kernels.
- ONNX extension later owns Pad/Tile mappings without entering model or runtime hot path.

Do not create a detailed follow-up specification here.

## Architecture impact

Expected impact: None. These model-owned semantic values remain in the existing package and
dependency direction. Stop if implementation needs Tensor/Shape/DataType state, dependencies,
generic Operation changes, or architecture changes.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0001/0002/0005/0006/0017G/0017H/0017I, Tensor API, Compile
API, Training API, glossary, current OperationKind/OperationAttrs/Operation and layout-operation
semantic contracts/tests, DataType/Shape terminology, and Java 26 Gradle configuration.

Implement task 0017I exactly. Add only PadKind.java, PadAttrs.java, TileKind.java, TileAttrs.java,
and PadTileSemanticsTest.java under io.github.pho001.synaptik.model.operation.layout.

PadKind contains exactly PAD; TileKind contains exactly TILE. PadAttrs contains exactly
List<Long> before, List<Long> after, and double constantValue; validate equal sizes, indexed
non-null elements, and non-negative widths with exact order/messages, then snapshot both lists.
Retain every double unchanged. TileAttrs contains exactly List<Long> repeats; validate indexed
non-null strictly positive values and snapshot once. Empty lists and extreme valid longs are
accepted. Document exact typed pairings, scalar identity, constant-padding meaning, and complete-
pattern tiling meaning.

Do not add Tensor methods, result Shape/DataType/layout/provenance behavior, constant conversion,
other padding modes, zero/negative repeats, gradients, graph/compiler/planning/runtime/backend/
ONNX behavior, factories, dependencies, build/architecture changes, existing Java edits, or later
specs. Stop beyond ten paths or on architecture uncertainty.

Run all specified validation, then hand actual diff/evidence to a separate clean-context docs agent
in the same change. It must inspect source/tests/generated Javadoc, finalize permitted Javadocs/
Tensor API/glossary/planning, record no-change conclusions, and rerun validation.

Update 0017I, model master plan, and roadmap only for planning status/evidence. Do not mark
Complete until both passes succeed. Leave 0017J Draft without a specification. Do not commit/push.
```

## Local decisions

- Pad and tile use separate one-constant enums because they have unrelated attribute shapes and
  typed pairings; neither is an alias or mode of the other.
- Parameters use immutable long lists to align with current Shape geometry while avoiding mutable
  legacy arrays.
- Padding constant remains `double` to preserve the selected legacy capability. Semantic storage
  does not imply a conversion rule for any DataType.
- All double values are structurally accepted and retained; public/task-0017J policy owns any
  eventual data-type compatibility decision.
- Tile repeats remain strictly positive for parity; zero-repeat empty-result semantics are not
  introduced silently.
- Empty lists are valid rank-zero scalar identity parameters; input-rank matching is deferred.

## Known limitations

- No public Tensor methods, rank/Shape arithmetic, constant conversion, result descriptor,
  provenance, gradient, materialization, lowering, execution, or ONNX mapping exists.
- Only constant padding and positive per-axis complete-pattern tiling are represented.

## Validation evidence

- Architecture/focused boundary docs, documentation/planning rules, roadmap, model capabilities and
  master plan, Operation foundations, current layout semantic contracts/tests, DataType/Shape
  terminology, Tensor/Compile/Training APIs, glossary, Java 26 build, and read-only legacy
  pad/tile source/tests were reviewed before specifying behavior.
- Legacy evidence confirms constant double padding with equal-rank non-negative before/after
  arrays and positive per-axis complete-pattern tiling. Coupled legacy Shape, graph, storage,
  gradient, compiler, kernel, ONNX, and runtime design is excluded.
- `git diff --check` passed with no whitespace errors.
- Changed-path inventory contains exactly three planning paths: this task, model master plan, and
  roadmap. No Java, API, glossary, Gradle, AGENTS, ARCHITECTURE, focused architecture, completed
  task, or other-module file changed during planning.
- Markdown structure check passed: 23 level-two sections, 14 balanced code-fence markers, no
  trailing whitespace, and final newline present.
- Every local Markdown link in all three changed files resolves.
- Roadmap contains all 74 ordered task rows. Task 0017I is linked and Ready at row 62; task 0017J
  remains Draft at row 63; no task-0017J specification exists.
- Task, master plan, and roadmap consistently identify 0017I as the next Ready frontier.
- Package/scope review confirms four cohesive public semantic contracts remain in
  `model.operation.layout`, one focused test mirrors that package, and implementation is bounded
  to ten authorized paths without dependency or architecture changes.
- No Gradle test was run because this planning-only change modifies no production or test code.
- Clean implementation context `/root/implement_model_0017i` added exactly the four production
  contracts and focused test specified by this task before handing the actual shared-tree diff to
  the independent documentation context.
- Clean documentation context
  `/root/implement_model_0017i/review_model_0017i_docs` applied the General, API/Javadoc,
  Planning, and Example profiles. It independently reviewed the architecture and focused boundary
  documents, documentation/planning rules, capability baseline, prerequisite and neighboring
  tasks, Tensor/Compile/Training APIs, glossary, final source/tests, generated Javadoc, Java 26
  build configuration, and actual diff.
- The documentation pass retained the complete `PadKind` and `TileKind` Javadocs unchanged,
  tightened `PadAttrs` and `TileAttrs` constructor Javadocs with exact indexed failure messages,
  and finalized Tensor API and glossary coverage. The result documents exact typed pairings,
  scalar identity, immutable ordered snapshots, constant padding `[10, 20] ->
  [-1, 10, 20, -1, -1]`, two-dimensional complete-pattern tiling, structurally valid
  `Long.MAX_VALUE`, raw signed-zero/NaN/infinity retention, deferred Shape/DataType policy, and
  every excluded layer.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.operation.layout.PadTileSemanticsTest` — `BUILD SUCCESSFUL`; the
  focused XML report contains 12 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:test` — `BUILD SUCCESSFUL`; 69 XML reports contain 566 tests with zero
  failures, errors, or skips.
- `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL`; generated pages contain both enums and
  constants, both record components and canonical constructors, explicit accessors, parameters,
  results, exact failure conditions, ownership, both examples, scalar/extreme/raw-double cases,
  and cross-layer exclusions.
- `./gradlew test` — `BUILD SUCCESSFUL`; the root lifecycle reported 36 actionable tasks without a
  failing task.
- `javap -classpath modules/model/build/classes/java/main -p -c -s` for all four production types,
  focused reflection tests, source inspection, and import scans confirm the exact one-constant
  enums; exact three- and one-component records; validation order; direct accessors; generated
  record methods; exactly two plus one post-validation `List.copyOf` calls; no retained helper
  state or extra API; and production imports only `model.operation` plus `java.base`.
- A targeted local Markdown file-and-GitHub-heading checker resolved all 334 links and anchors in
  the five changed documentation/planning files with zero errors. Fence counts are balanced
  (`162`, `0`, `14`, `2`, `0`), trailing-whitespace scans found no matches, and every changed file
  ends with a newline.
- Final path inventory contains exactly the permitted ten paths: four production sources, one
  focused test, Tensor API, glossary, this task, model master plan, and roadmap. Task, master plan,
  and roadmap identify 0017I as Complete; 0017J remains Draft and no task-0017J specification
  exists.
- Compile API remains accurate unchanged because there is no public pad/tile Tensor expression or
  compiler capture. Training API remains accurate unchanged because no gradient, autograd,
  optimizer, parameter, session, or training behavior is defined. Capabilities remains accurate
  unchanged because it already selects constant pad and tile and distinguishes model semantics
  from later public and executable layers.
- `OperationKind`, `OperationAttrs`, `Operation`, DataType, Shape/Dimension, SliceAttrs/SliceKind,
  adjacent layout-operation semantics, Tensor/layout/graph contracts, generated related Javadocs,
  and completed task specifications remain accurate unchanged because this family composes their
  existing ownership and value contracts without changing signatures or behavior.
- No architecture document, ADR, architecture test, backend-conformance test, integration test,
  Gradle file, dependency, existing unrelated Java/test, other module, or later task specification
  changed because this task adds only backend-independent model semantic values and no boundary,
  executable, backend, or end-to-end behavior.
- `git diff --check` passed with no whitespace errors after final documentation and planning
  synchronization.

## Implementation notes

- Added separate `PadKind.PAD` and `TileKind.TILE` identities plus exact immutable `PadAttrs` and
  `TileAttrs` records in the existing layout-operation package.
- `PadAttrs` validates containers, matching sizes, indexed elements, and non-negative widths in the
  specified precedence, then snapshots both lists. It retains every `double` bit pattern without
  DataType interpretation. `TileAttrs` validates indexed strictly positive repeats and snapshots
  once. Empty lists and extreme valid longs remain structurally valid.
- Added one focused same-package test covering exact API shape, validation messages and precedence,
  immutable ownership, value semantics, raw-double retention, scalar/extreme cases, typed
  Operation composition, and forbidden-state absence.
- Finalized public Javadocs, Tensor API, glossary, task evidence, model master plan, and roadmap
  without creating the 0017J specification or changing architecture or another contract.

## Completion summary

- Completed changes: Added and documented exact backend-neutral constant-padding and
  complete-pattern tiling semantic identities and immutable normalized attributes, with focused
  semantic tests and synchronized planning evidence.
- Files changed or created: Exactly the ten paths listed under Affected files.
- Tests and validation: Focused 12-test suite, all 566 model tests, model Javadoc, root tests,
  bytecode/reflection/import/source checks, generated-Javadoc review, 334 local link/anchor checks,
  fence/whitespace/final-newline checks, exact scope/status review, no-0017J-spec check, and
  `git diff --check` passed.
- Documentation-agent review: Clean context
  `/root/implement_model_0017i/review_model_0017i_docs` independently finalized the change using
  the General, API/Javadoc, Planning, and Example profiles.
- Documentation impact: Tensor API and glossary now define current pad/tile semantics, exact
  examples, ownership, validation, and deferred boundaries while public Tensor construction
  remains Draft task 0017J work.
- Javadoc review: PadKind/TileKind type and constant Javadocs were complete unchanged; PadAttrs and
  TileAttrs constructor failure documentation was tightened, and all type/component/accessor/
  ownership/example/exclusion contracts are complete. Related Javadocs remain accurate unchanged.
- Glossary impact: Added reusable Padding and Tiling terms and updated implementation status,
  OperationKind, OperationAttrs, and the kind/attributes/operation distinction.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0017I. Task 0017J remains Draft without a specification.

Status: Complete
