# Task 0018M1: Dynamic Extent Adoption in Pad, Tile, and Concat

## Status

Complete

## Goal

Adopt the canonical symbolic extent arithmetic completed by task 0018M in the existing
`Tensor.pad`, `Tensor.tile`, and `Tensor.concat` result-Shape derivation.

The task replaces three conservative dynamic-Shape rejections with exact model-owned formulas:

```text
pad:     N, before=2, after=3  -> N + 5
tile:    N, repeat=4           -> 4 * N
concat:  [N, 8] + [M, 8]       -> [N + M, 8]
```

Public Tensor signatures, typed operations, ordered provenance, result layout, data type,
gradient eligibility, and lifecycle ownership remain unchanged. This is a bounded adoption of an
existing shape-value foundation, not a new expression system or execution feature.

## Current limitation

Before this task, `DimensionExpressions` could represent exact linear combinations, but the older
Tensor helpers still implemented the pre-0018M fallback rules: pad accepted only zero widths on a
dynamic axis, tile accepted only repeat one, and concat accepted at most one dynamic selected
extent with only static-zero companions. Those restrictions rejected exactly representable
results and are the limitation this completed task removes.

## Scope

- Modify only the existing package-private pad/tile and composition expression helpers.
- Use `DimensionExpressions.addConstant` for every padded result extent.
- Use `DimensionExpressions.multiply` for every tiled result extent.
- Fold every selected concat extent through `DimensionExpressions.add` in encounter order.
- Accept all current Dimension categories uniformly through the canonical public construction
  boundary.
- Preserve public signatures, validation outside the obsolete dynamic rejections, input ownership,
  typed attributes, operation kinds, provenance, descriptor metadata, unresolved layouts,
  freshness, and Tensor-ID side effects.
- Preserve checked `long` arithmetic and fail before Tensor identity allocation on overflow.
- Replace focused dynamic-rejection tests with canonical-expression success tests while retaining
  static, validation, ownership, metadata, and overflow coverage.
- Finalize affected Javadocs, Tensor API, glossary, capability status, and planning evidence through
  the mandatory independent documentation pass.

## Out of scope

- any new public Tensor API, helper type, expression form, dimension category, binding API,
  evaluator, solver, or constraint system
- changing `Dimension`, `DimensionExpression`, `ExpressionDimension`, `DimensionExpressions`,
  `Shape`, `Tensor`, descriptors, factories, producer/provenance contracts, operation kinds, or
  operation attributes
- symbolic adoption in stack, unstack, reshape, expand, slice, window operations, convolution,
  pooling, or another Tensor expression
- changing raw-array validation, concat input validation, or unrelated exception messages and
  validation precedence
- physical layout derivation, aliases, values/storage, padding-constant conversion, output buffers,
  graph-wide inference, autograd, compiler, prepare, runtime, backend, ONNX, or execution behavior
- dependencies, Gradle/build, architecture, architecture tests, another module, unrelated
  refactors, or a detailed task-0018N specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)
- [Model master plan](../master-plan.md)
- [Task 0002](0002-shape-and-dimension-model.md)
- [Task 0017J](0017j-pad-and-tile-tensor-expressions.md)
- [Task 0017L](0017l-tensor-composition-expressions.md)
- [Task 0018M](0018m-symbolic-extent-expressions.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- All executable changes remain inside `modules/model` and use only the JDK and local model types.
- Tensor remains public API state, not graph IR. Existing helpers continue to create immutable
  metadata and delegate final identity creation through `TensorFactory`.
- `DimensionExpressions` is the sole construction and canonicalization boundary. Helpers must not
  inspect expression internals or reproduce their algebra.
- Shape derivation must not bind symbols, solve graph-wide equalities, or introduce compiler,
  prepared, runtime, storage, or backend state.
- Operation and ordered producer/provenance semantics remain exactly PAD/TILE with `[input]` and
  CONCAT with the existing ordered input snapshot.
- Every result layout remains unresolved; precise logical extents do not imply physical geometry.
- Stop if implementation requires a dependency or architecture change.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.tensor`
- `io.github.pho001.synaptik.model.shape`

No package is added or reorganized.

Type placement:

- `TensorPadTileExpressions` remains the owner of local pad/tile validation and Shape derivation.
- `TensorCompositionExpressions` remains the owner of concat validation and Shape derivation.
- `DimensionExpressions` remains unchanged and supplies canonical arithmetic; no algebra is copied
  into the tensor package.
- Focused tests remain in the mirrored tensor package.

## Required behavior

### Public and helper surfaces

Do not change any public Tensor signature. Retain exactly:

```java
public Tensor pad(long[] before, long[] after, double constantValue)
public Tensor tile(long... repeats)
public static Tensor concat(int axis, Tensor... inputs)
```

Both helpers remain final, package-private, field-free classes with their existing constructors,
methods, parameters, returns, and visibility. Add no wrapper method.

### Dynamic padding

Preserve the current validation and construction order through creation of one validated
`PadAttrs`. Change only `paddedShape`. For each axis derive:

```text
result = addConstant(addConstant(inputDimension, before), after)
```

Both calls use `DimensionExpressions.addConstant`, in before-then-after order. Therefore static
`5` plus `2` and `3` remains static `10`; named `N` becomes `N + 5`; existing `N + 4` becomes
`N + 9`; and zero/zero returns the exact input Dimension reference for every category. Checked
overflow propagates before Tensor identity allocation.

Remove the former `cannot pad dynamic axis ...` path. Null, rank, negative-width, and attribute
failures keep their existing types, exact messages, and precedence.

### Dynamic tiling

Preserve validation through creation of one validated `TileAttrs`. Change only `tiledShape`. For
each axis call `DimensionExpressions.multiply(inputDimension, repeat)` exactly once. Static `5`
repeated `3` remains `15`; `N` becomes `3 * N`; `N + 2` becomes `3 * N + 6`; and repeat one
returns the exact input Dimension reference. Static value, coefficient, or offset overflow fails
before Tensor identity allocation.

Remove the former `cannot tile dynamic axis ...` path. Null, rank, and non-positive-repeat failures
keep their existing types, exact messages, and precedence.

### Dynamic concat

Keep ordered snapshot, null/empty, axis, exact DataType, rank, non-concat-dimension, and eligibility
validation unchanged. Change only selected-axis calculation in `concatShape`.

Create one static-zero accumulator and add every selected Dimension in encounter order with
`DimensionExpressions.add`. Store the final canonical result at the selected axis; preserve exact
first-input references on every non-selected axis.

This supports `N + M`, `N + 3`, repeated `N` as `2 * N`, and flattened existing linear
expressions. Division and constrained-unknown Dimensions remain valid atomic terms rather than
being solved or reassociated. Static-zero companions are neutral, all-static results remain
static, and all checked overflow occurs before Tensor identity allocation.

Remove the former `cannot represent concat axis ... with dynamic extents` paths. Non-selected
dimensions must still compare structurally equal; no broadcasting or equality inference is added.

### Result metadata and canonical references

All successful results retain exact current type and eligibility propagation, unresolved layout,
absent label/storage, fresh identity, typed operations/attributes, existing producer/provenance,
and ordered inputs. Identity pad/tile and one-input concat remain explicit fresh Tensor operations.

Use canonical factory results rather than reconstructing equal Dimensions:

- zero/zero pad and repeat-one tile retain the exact input Dimension reference;
- static-zero concat companions preserve the opposing reference when the canonical addition rule
  permits it;
- independently built equal formulas compare structurally equal; and
- tests must not require a new equal Dimension where the canonical factory returns an existing
  reference.

Static numeric Shapes and non-identity overflow behavior remain unchanged. Reference preservation
is the intentional adoption difference.

## Affected files

Expected implementation and tests:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorPadTileExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorCompositionExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java` — Javadoc-only
  correction authorized after independent review found the public methods still stated the
  removed dynamic rejection rules
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorPadTileExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorCompositionExpressionTest.java`

Expected documentation and planning:

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task file
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum scope

At most the eleven paths above may change. The original ten-path cap was explicitly expanded
during independent documentation review solely to correct stale Javadocs on the three existing
public Tensor methods. No Java file, public API, declaration, method body, or executable behavior
is added or changed. Stop and propose a follow-up if another path, contract, module, dependency,
build file, or architecture document is required.

## Javadoc and documentation requirements

- Update helper type descriptions, affected Shape-derivation Javadocs, and the three existing
  public Tensor method Javadocs to describe exact symbolic results instead of removed rejection
  rules.
- Document canonical references, checked coefficient/offset overflow, unresolved layout, and the
  boundary between formula construction and later binding/evaluation.
- Preserve complete documentation for every unchanged parameter, result, ownership rule,
  side effect, and failure; remove only obsolete dynamic-rejection claims.
- Add newcomer-readable `N + constant`, `repeat * N`, and `N + M` examples to the Tensor API.
- Correct glossary wording that currently limits pad/tile and concat dynamic results. Reuse the
  existing symbolic-extent term rather than adding a duplicate.
- Move capability wording to completed behavior only after implementation and documentation pass.
- Record reasoned no-change conclusions for public Tensor Javadoc/signatures, Compile API,
  Training API, architecture docs/tests, semantic attributes, shape foundation, conformance,
  integration, Gradle, dependencies, and other modules.

## Acceptance criteria

- Only the two existing production helpers and two focused suites change as executable paths;
  `Tensor.java` changes Javadoc only under the explicitly authorized scope expansion.
- Pad and tile preserve request validation/ownership and accept every Dimension category only via
  `DimensionExpressions` public methods.
- Padding derives canonical `N + before + after`; tiling derives canonical `repeat * N`.
- Concat accepts ordered sums of static, named, exact-expression, and constrained-unknown selected
  extents and preserves non-selected references.
- Tests cover dynamic plus constant, `N + M`, repeated terms, pre-existing expressions, neutral
  identities, structural equality, and exact canonical references.
- Existing scalar/static/zero, data type, eligibility, semantics, provenance, layout, storage,
  freshness, ownership, validation, and overflow coverage remains passing except for the specified
  reference identity adjustment.
- Obsolete dynamic errors are no longer asserted; unrelated exact failures and no-ID effects remain
  covered. Arithmetic overflow always precedes Tensor allocation.
- No binding/evaluation, expression-form, graph inference, value/storage, cross-layer, dependency,
  build, or architecture change appears.
- Javadocs, independent documentation review, and task/master/roadmap/capability status are final
  before Complete.

## Tests / validation

Run focused suites as needed during development. After executable Java stabilizes, record one final
module run:

```bash
./gradlew :modules:model:test
```

The documentation pass reuses that evidence unless it changes executable Java, then runs:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

It also validates changed Markdown links/anchors, fences, whitespace, final newlines, examples,
eleven-path scope, package placement, and synchronized statuses. Source/diff review confirms
public Tensor signatures and bodies are untouched, arithmetic comes only from
`DimensionExpressions`, and no value/storage/cross-layer import appears.

Repository-wide tests are deferred to the foundation-contract checkpoint after task 0018N and CI.
Do not duplicate a successful model suite in the documentation context without executable changes
or a recorded concrete reason.

## Dependencies

- [Task 0002](0002-shape-and-dimension-model.md) — Dimension and Shape foundation.
- [Task 0017J](0017j-pad-and-tile-tensor-expressions.md) — existing pad/tile behavior.
- [Task 0017L](0017l-tensor-composition-expressions.md) — existing concat behavior.
- [Task 0018M](0018m-symbolic-extent-expressions.md) — canonical arithmetic consumed here.

## Follow-up tasks

- Task 0018N, Typed scalar value contract, remains the next Draft frontier and owns the
  foundation-contract checkpoint. Do not create its specification here.
- Task 0018R owns later slice/window cleanup and any remaining supported window-formula adoption.
- Binding and evaluation require later compiler/prepare/runtime tasks.

## Architecture impact

Expected impact: None. Model already owns Tensor/Shape semantics, while compiler owns graph-wide
inference. Stop if implementation needs binding, evaluation, another module, or lifecycle changes.

## Implementation prompt

Use this prompt in a separate clean-context agentic task/thread:

```text
Read AGENTS.md, ARCHITECTURE.md, documentation/planning rules, model capabilities/master plan,
roadmap, tasks 0002, 0017J, 0017L, 0018M, and 0018M1, current symbolic-shape and affected Tensor
helper source/tests, Tensor API, glossary, and Java 26 Gradle configuration.

Implement task 0018M1 exactly. Stay inside the two existing helpers, their focused tests, and the
explicitly allowed documentation/planning files. Adopt only canonical DimensionExpressions
arithmetic in pad, tile, and concat Shape derivation. Preserve public signatures, unrelated
validation, semantics, producer/provenance, unresolved layout, metadata, ID effects, and
architecture boundaries. Stop beyond eleven paths or on architecture uncertainty. The eleventh
path is `Tensor.java` and is authorized only for Javadoc correction; do not change its declarations
or executable behavior.

Run one final model test after executable code stabilizes. Then hand the diff and test evidence to
a separate clean-context documentation agent in the same change. It must inspect final source/tests,
finalize affected Javadocs, Tensor API, glossary, capability/task/master/roadmap status, and
documentation validation without repeating successful Java tests unless executable behavior
changes or it records a concrete reason.

Do not mark 0018M1 Complete until both passes succeed. Leave 0018N Draft without a detailed
specification. Do not commit or push.
```

## Local decisions

- Reuse only the existing six-method `DimensionExpressions` boundary.
- Apply pad widths in before-then-after order to preserve checked sequencing.
- Fold concat from static zero in input order for deterministic failure timing.
- Canonical neutral identities intentionally preserve references even where the older static-only
  helper reconstructed an equal static value.
- Keep all layouts unresolved because logical extent precision does not imply physical geometry.
- Independent documentation review found stale public Tensor Javadocs that contradicted the
  removed dynamic rejection paths. Explicit user authorization expanded the maximum from ten to
  eleven paths solely for Javadoc correction; declarations, signatures, bodies, and executable
  behavior remain unchanged.

## Known limitations

- Symbolic results cannot yet be bound or evaluated.
- Division nodes and unknowns are atomic terms; non-selected concat dimensions still require
  structural equality.
- Pad constants and other provisional public contracts remain for their dedicated cleanup tasks.
- Window, convolution, pooling, and other helpers do not adopt symbolic arithmetic here.

## Validation evidence

Implementation context `/root/task_0018m1_implementation` inspected the final source and focused
tests, changed only the two existing helpers and two focused suites as executable paths, and
recorded these successful runs after executable Java stabilized:

- `./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorPadTileExpressionTest --tests io.github.pho001.synaptik.model.tensor.TensorCompositionExpressionTest`
  passed all 25 focused tests: 11 pad/tile plus 14 composition, with zero failures, errors, or
  skips.
- `./gradlew :modules:model:test` passed with `BUILD SUCCESSFUL in 1s`; the reports contain 766
  tests across 88 suites, zero failures, zero errors, and zero skipped tests.

No executable Java changed after those runs. Documentation context
`/root/task_0018m1_implementation/task_0018m1_documentation` reused that evidence as required and
applied General, API/Javadoc, Planning, and Example profiles. It independently reviewed the
architecture contract, documentation/planning rules, capability baseline, master plan, roadmap,
tasks 0002/0017J/0017L/0018M/0018M1, final helpers and focused tests, public Tensor contracts,
shape-expression foundation, Tensor API, glossary, Compile API, Training API, and Java 26 Gradle
configuration.

- `./gradlew :modules:model:javadoc` passed with `BUILD SUCCESSFUL in 2s`; generated public Tensor
  documentation contains the `N + 5`, `4 * N`, and `N + M` contracts and no removed dynamic-error
  claims.
- A targeted Ruby check validated 452 local Markdown links, including 139 anchors, across the six
  changed Markdown files.
- Targeted Ruby checks confirmed balanced code fences, final newlines, and no trailing whitespace
  across those six files. The `N + 5`, `4 * N`, and `N + M` examples were recalculated against the
  canonical factory rules and focused tests.
- Source/diff checks confirmed the exact eleven authorized paths, correct existing package
  placement, unchanged public Tensor signatures and bodies, Javadoc-only `Tensor.java` changes,
  before-then-after pad calls, one tile multiply call per axis, static-zero encounter-order concat
  folding, and no helper arithmetic outside `DimensionExpressions`.
- Source/import review found no new value, storage, graph-inference, compiler, prepare, runtime,
  backend, dependency, build, or cross-module state. The two helper imports are limited to JDK and
  existing model types.
- Task, master-plan row, roadmap row, current frontier, and capability wording are synchronized at
  Complete. Task 0018N remains Draft and no detailed 0018N specification exists.
- `git diff --check` passed on the final combined change.

The original ten-path cap was insufficient because independent review found that the public
`Tensor.pad`, `tile`, and `concat` Javadocs still promised the removed dynamic rejection behavior.
Explicit user authorization expanded the maximum to eleven paths solely for those Javadoc
corrections; declarations, signatures, bodies, and executable behavior remain unchanged.

Reasoned no-change conclusions:

- Compile API remains accurate because it identifies current pad/tile/composition metadata and
  keeps capture, inference, canonicalization, decomposition, and execution planned without
  specifying the replaced local fallback rules.
- Training API remains accurate because it contains only planned training ownership and no
  pad/tile/concat Shape contract.
- `Dimension`, `DimensionExpression`, `ExpressionDimension`, `DimensionExpressions`, and `Shape`
  remain accurate unchanged because this task consumes their completed canonical arithmetic and
  does not add an expression form, binding, evaluation, or category.
- Pad/tile/composition semantic kinds and attributes remain accurate unchanged because they do not
  own input Shapes or result-extent derivation.
- Architecture documents/tests, backend conformance tests, and integration tests remain unchanged
  because no dependency, boundary, backend, or end-to-end executable behavior changes.
- Java 26 Gradle configuration, dependencies, other modules, values/storage, layout derivation,
  gradients, compiler, prepare, runtime, backend, and execution remain unchanged because the task
  is confined to model metadata derivation and its documentation. Result layout is still
  deliberately unresolved.

## Implementation notes

- `paddedShape` applies `DimensionExpressions.addConstant` twice per axis in before-then-after
  order.
- `tiledShape` applies `DimensionExpressions.multiply` exactly once per axis.
- `concatShape` starts with `StaticDimension(0)` and encounter-order folds every selected
  Dimension through `DimensionExpressions.add`.
- Focused tests replace obsolete dynamic-error assertions with canonical named, repeated,
  pre-existing-linear, division, constrained-unknown, neutral-reference, structural-equality, and
  checked-overflow coverage while retaining metadata, provenance, ownership, validation, and
  no-ID effects.
- The documentation pass changed no executable Java. It finalized the two helper contracts, the
  explicitly authorized public Tensor Javadocs, Tensor API examples, glossary definitions,
  capability status, and synchronized planning evidence.

## Completion summary

- Completed changes: Adopted canonical symbolic extent arithmetic for pad, tile, and concat while
  preserving validation, metadata, unresolved layout, provenance, freshness, and ID behavior.
- Files changed or created: Eleven authorized paths: two helper sources, two focused tests,
  Javadoc-only `Tensor.java`, Tensor API, glossary, capability baseline, this task, model master
  plan, and roadmap.
- Tests and validation: Reused the final 25-test focused and 766-test/88-suite model results;
  model Javadoc, generated-page inspection, 452-link/139-anchor Markdown validation,
  fence/newline/whitespace checks, example review, exact eleven-path/package/signature/import/
  status/no-0018N-spec checks, and final diff checks passed.
- Documentation-agent review: Clean context
  `/root/task_0018m1_implementation/task_0018m1_documentation` completed the independent targeted
  review using General, API/Javadoc, Planning, and Example profiles and changed no executable Java.
- Documentation impact: Tensor API, glossary, capability baseline, task evidence, master plan, and
  roadmap now describe current canonical symbolic pad/tile/concat Shapes and deferred
  binding/evaluation.
- Javadoc review: Helper and public Tensor contracts now document exact formulas, neutral
  references, checked static/coefficient/offset overflow, unresolved layout, ownership, failures,
  and lifecycle boundaries. Public declarations and bodies are unchanged.
- Glossary impact: Corrected the existing Dimension, padding, tiling, composition, and Tensor
  wording without adding a duplicate symbolic-extent term.
- Unresolved issues: None.
- Follow-up required: None. Task 0018N remains Draft without a detailed specification.

Status: Complete
