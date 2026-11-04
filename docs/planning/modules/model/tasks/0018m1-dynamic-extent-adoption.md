# Task 0018M1: Dynamic [Extent](../../../../glossary.md#term-extent) Adoption in Pad, Tile, and Concat

## Status

Complete

## Goal

Adopt the [canonical](../../../../glossary.md#term-canonical-form) symbolic [extent](../../../../glossary.md#term-extent) arithmetic completed by task 0018M in the existing
`Tensor.pad`, `Tensor.tile`, and `Tensor.concat` result-[Shape](../../../../glossary.md#shape) derivation.

The task replaces three conservative dynamic-Shape rejections with exact [model](../../../../glossary.md#term-model)-owned formulas:

```text
pad:     N, before=2, after=3  -> N + 5
tile:    N, repeat=4           -> 4 * N
concat:  [N, 8] + [M, 8]       -> [N + M, 8]
```

Public [Tensor](../../../../glossary.md#tensor) [signatures](../../../../glossary.md#term-signature), typed operations, ordered [provenance](../../../../glossary.md#provenance), result [layout](../../../../glossary.md#layout), [data type](../../../../glossary.md#data-type--datatype),
[gradient](../../../../glossary.md#term-gradient) eligibility, and [lifecycle](../../../../glossary.md#lifecycle) [ownership](../../../../glossary.md#term-ownership-meanings) remain unchanged. This is a bounded adoption of an
existing shape-value foundation, not a new expression system or execution [feature](../../../../glossary.md#term-feature).

## Current limitation

Before this task, `DimensionExpressions` could represent exact [linear](../../../../glossary.md#term-linear-meanings) combinations, but the older
[Tensor](../../../../glossary.md#tensor) helpers still implemented the pre-0018M fallback rules: pad accepted only zero widths on a
dynamic [axis](../../../../glossary.md#term-axis), tile accepted only repeat one, and concat accepted at most one dynamic selected
[extent](../../../../glossary.md#term-extent) with only static-zero companions. Those restrictions rejected exactly representable
results and are the limitation this completed task removes.

## [Scope](../../../../glossary.md#term-scope)

- Modify only the existing package-private pad/tile and composition expression helpers.
- Use `DimensionExpressions.addConstant` for every padded result [extent](../../../../glossary.md#term-extent).
- Use `DimensionExpressions.multiply` for every tiled result extent.
- Fold every selected concat extent through `DimensionExpressions.add` in encounter order.
- Accept all current Dimension categories uniformly through the canonical public construction
  boundary.<span data-glossary-navigation="true"> ([Dimension glossary](../../../../glossary.md#dimension); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Preserve public signatures, validation outside the obsolete dynamic rejections, input ownership,
  typed attributes, operation kinds, provenance, descriptor metadata, unresolved layouts,
  freshness, and Tensor-ID side effects.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [validation glossary](../../../../glossary.md#term-validation); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [operation glossary](../../../../glossary.md#operation); [provenance glossary](../../../../glossary.md#provenance); [descriptor glossary](../../../../glossary.md#term-descriptor); [metadata glossary](../../../../glossary.md#term-metadata); [layouts glossary](../../../../glossary.md#layout))</span>
- Preserve checked `long` arithmetic and fail before [Tensor](../../../../glossary.md#tensor) [identity](../../../../glossary.md#term-identity-meanings) allocation on [overflow](../../../../glossary.md#term-overflow).
- Replace focused dynamic-rejection tests with canonical-expression success tests while retaining
  static, validation, ownership, metadata, and overflow coverage.
- Finalize affected Javadocs, Tensor API, glossary, capability status, and planning evidence through
  the mandatory independent documentation pass.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- any new public Tensor API, helper type, expression form, dimension category, binding API,
  evaluator, solver, or constraint system<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [binding glossary](../../../../glossary.md#term-binding))</span>
- changing [`Dimension`](../../../../glossary.md#dimension), `DimensionExpression`, `ExpressionDimension`, `DimensionExpressions`,
  [`Shape`](../../../../glossary.md#shape), [`Tensor`](../../../../glossary.md#tensor), descriptors, factories, producer/provenance contracts, operation kinds, or
  operation attributes<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance); [operation glossary](../../../../glossary.md#operation))</span>
- symbolic adoption in stack, unstack, reshape, expand, slice, window operations, convolution,
  pooling, or another Tensor expression<span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [pooling glossary](../../../../glossary.md#term-pooling))</span>
- changing raw-array validation, concat input validation, or unrelated exception messages and
  validation precedence<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- physical layout derivation, aliases, values/storage, padding-constant conversion, output buffers,
  graph-wide inference, autograd, compiler, prepare, runtime, backend, ONNX, or execution behavior<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [aliases glossary](../../../../glossary.md#aliasing-and-arena); [padding glossary](../../../../glossary.md#padding); [constant glossary](../../../../glossary.md#term-constant); [buffers glossary](../../../../glossary.md#term-buffer); [graph glossary](../../../../glossary.md#graph); [inference glossary](../../../../glossary.md#term-inference); [autograd glossary](../../../../glossary.md#autograd); [compiler glossary](../../../../glossary.md#term-compiler); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [ONNX glossary](../../../../glossary.md#term-onnx))</span>
- dependencies, Gradle/build, architecture, architecture tests, another module, unrelated
  refactors, or a detailed task-0018N specification<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Model master plan](../master-plan.md)
- [Task 0002](0002-shape-and-dimension-model.md)
- [Task 0017J](0017j-pad-and-tile-tensor-expressions.md)
- [Task 0017L](0017l-tensor-composition-expressions.md)
- [Task 0018M](0018m-symbolic-extent-expressions.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Glossary](../../../../glossary.md)

## Architecture constraints

- All executable changes remain inside `modules/model` and use only the [JDK](../../../../glossary.md#term-java-development-kit-jdk) and local [model](../../../../glossary.md#term-model) types.
- Tensor remains public API state, not graph IR. Existing helpers continue to create immutable
  metadata and delegate final identity creation through <span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [graph glossary](../../../../glossary.md#graph); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [immutable glossary](../../../../glossary.md#term-immutable); [metadata glossary](../../../../glossary.md#term-metadata); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>`TensorFactory`.
- `DimensionExpressions` is the sole construction and canonicalization boundary. Helpers must not
  inspect expression internals or reproduce their algebra.<span data-glossary-navigation="true"> ([canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form))</span>
- Shape derivation must not bind symbols, solve graph-wide equalities, or introduce compiler,
  prepared, runtime, storage, or backend state.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend))</span>
- [Operation](../../../../glossary.md#operation) and ordered [producer](../../../../glossary.md#term-producer)/[provenance](../../../../glossary.md#provenance) semantics remain exactly [PAD](../../../../glossary.md#padding)/TILE with `[input]` and
  CONCAT with the existing ordered input snapshot.<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- Every result [layout](../../../../glossary.md#layout) remains unresolved; precise logical extents do not imply physical geometry.
- Stop if implementation requires a [dependency](../../../../glossary.md#term-dependency) or architecture change.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.tensor`
- `io.github.pho001.synaptik.model.shape`

No package is added or reorganized.

Type placement:

- `TensorPadTileExpressions` remains the [owner](../../../../glossary.md#term-owner) of local pad/tile [validation](../../../../glossary.md#term-validation) and [Shape](../../../../glossary.md#shape) derivation.
- `TensorCompositionExpressions` remains the owner of concat validation and Shape derivation.
- `DimensionExpressions` remains unchanged and supplies canonical arithmetic; no algebra is copied
  into the tensor package.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Focused tests remain in the mirrored [tensor](../../../../glossary.md#tensor) package.

## Required behavior

### Public and helper surfaces

Do not change any public [Tensor](../../../../glossary.md#tensor) [signature](../../../../glossary.md#term-signature). Retain exactly:

```java
public Tensor pad(long[] before, long[] after, double constantValue)
public Tensor tile(long... repeats)
public static Tensor concat(int axis, Tensor... inputs)
```

Both helpers remain final, package-private, field-free classes with their existing constructors,
methods, [parameters](../../../../glossary.md#term-parameter), returns, and visibility. Add no wrapper method.

### Dynamic [padding](../../../../glossary.md#padding)

Preserve the current [validation](../../../../glossary.md#term-validation) and construction order through creation of one validated
`PadAttrs`. Change only `paddedShape`. For each [axis](../../../../glossary.md#term-axis) derive:

```text
result = addConstant(addConstant(inputDimension, before), after)
```

Both calls use `DimensionExpressions.addConstant`, in before-then-after order. Therefore static
`5` plus `2` and `3` remains static `10`; named `N` becomes `N + 5`; existing `N + 4` becomes
`N + 9`; and zero/zero returns the exact input [Dimension](../../../../glossary.md#dimension) reference for every category. Checked
[overflow](../../../../glossary.md#term-overflow) propagates before [Tensor](../../../../glossary.md#tensor) [identity](../../../../glossary.md#term-identity-meanings) allocation.

Remove the former `cannot pad dynamic axis ...` path. Null, [rank](../../../../glossary.md#term-rank), negative-width, and attribute
failures keep their existing types, exact messages, and precedence.

### Dynamic [tiling](../../../../glossary.md#tiling)

Preserve [validation](../../../../glossary.md#term-validation) through creation of one validated `TileAttrs`. Change only `tiledShape`. For
each [axis](../../../../glossary.md#term-axis) call `DimensionExpressions.multiply(inputDimension, repeat)` exactly once. Static `5`
repeated `3` remains `15`; `N` becomes `3 * N`; `N + 2` becomes `3 * N + 6`; and repeat one
returns the exact input [Dimension](../../../../glossary.md#dimension) reference. Static value, coefficient, or offset [overflow](../../../../glossary.md#term-overflow) fails
before [Tensor](../../../../glossary.md#tensor) [identity](../../../../glossary.md#term-identity-meanings) allocation.

Remove the former `cannot tile dynamic axis ...` path. Null, [rank](../../../../glossary.md#term-rank), and non-positive-repeat failures
keep their existing types, exact messages, and precedence.

### Dynamic concat

Keep ordered [snapshot](../../../../glossary.md#term-snapshot), null/empty, [axis](../../../../glossary.md#term-axis), exact [DataType](../../../../glossary.md#data-type--datatype), [rank](../../../../glossary.md#term-rank), non-concat-[dimension](../../../../glossary.md#dimension), and eligibility
[validation](../../../../glossary.md#term-validation) unchanged. Change only selected-axis calculation in `concatShape`.

Create one static-zero [accumulator](../../../../glossary.md#accumulator-type) and add every selected Dimension in encounter order with
`DimensionExpressions.add`. Store the final [canonical](../../../../glossary.md#term-canonical-form) result at the selected axis; preserve exact
first-input references on every non-selected axis.

This supports `N + M`, `N + 3`, repeated `N` as `2 * N`, and flattened existing [linear](../../../../glossary.md#term-linear-meanings)
expressions. Division and constrained-unknown Dimensions remain valid [atomic](../../../../glossary.md#resource-transaction-and-rollback) terms rather than
being solved or reassociated. Static-zero companions are neutral, all-static results remain
static, and all checked [overflow](../../../../glossary.md#term-overflow) occurs before [Tensor](../../../../glossary.md#tensor) [identity](../../../../glossary.md#term-identity-meanings) allocation.

Remove the former `cannot represent concat axis ... with dynamic extents` paths. Non-selected
dimensions must still compare structurally equal; no [broadcasting](../../../../glossary.md#broadcasting) or equality [inference](../../../../glossary.md#term-inference) is added.

### Result [metadata](../../../../glossary.md#term-metadata) and [canonical](../../../../glossary.md#term-canonical-form) references

All successful results retain exact current type and eligibility propagation, unresolved [layout](../../../../glossary.md#layout),
absent label/storage, fresh [identity](../../../../glossary.md#term-identity-meanings), typed operations/attributes, existing [producer](../../../../glossary.md#term-producer)/[provenance](../../../../glossary.md#provenance),
and ordered inputs. Identity pad/tile and one-input concat remain explicit fresh [Tensor](../../../../glossary.md#tensor) operations.

Use canonical factory results rather than reconstructing equal [Dimensions](../../../../glossary.md#dimension):

- zero/zero pad and repeat-one tile retain the exact input Dimension reference;
- static-zero concat companions preserve the opposing reference when the canonical addition rule
  permits it;
- independently built equal formulas compare structurally equal; and
- tests must not require a new equal Dimension where the canonical factory returns an existing
  reference.

Static numeric [Shapes](../../../../glossary.md#shape) and non-identity [overflow](../../../../glossary.md#term-overflow) behavior remain unchanged. Reference preservation
is the intentional adoption difference.

## Affected files

Expected implementation and tests:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorPadTileExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorCompositionExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java` — Javadoc-only
  correction authorized after independent review found the public methods still stated the
  removed dynamic rejection rules<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorPadTileExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorCompositionExpressionTest.java`

Expected documentation and [planning](../../../../glossary.md#term-planning):

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task file
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum [scope](../../../../glossary.md#term-scope)

At most the eleven paths above may change. The original ten-path cap was explicitly expanded
during independent documentation review solely to correct stale Javadocs on the three existing
public [Tensor](../../../../glossary.md#tensor) methods. No Java file, public [API](../../../../glossary.md#term-application-programming-interface-api), declaration, method body, or executable behavior
is added or changed. Stop and propose a follow-up if another path, contract, [module](../../../../glossary.md#term-module), [dependency](../../../../glossary.md#term-dependency),
build file, or architecture document is required.

## [Javadoc](../../../../glossary.md#term-javadoc) and documentation requirements

- Update helper type descriptions, affected Shape-derivation Javadocs, and the three existing
  public Tensor method Javadocs to describe exact symbolic results instead of removed rejection
  rules.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape))</span>
- Document canonical references, checked coefficient/offset overflow, unresolved layout, and the
  boundary between formula construction and later binding/evaluation.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [overflow glossary](../../../../glossary.md#term-overflow); [layout glossary](../../../../glossary.md#layout); [binding glossary](../../../../glossary.md#term-binding); [evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- Preserve complete documentation for every unchanged parameter, result, ownership rule,
  side effect, and failure; remove only obsolete dynamic-rejection claims.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [side effect glossary](../../../../glossary.md#term-side-effect))</span>
- Add newcomer-readable `N + constant`, `repeat * N`, and `N + M` examples to the [Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api).
- Correct glossary wording that currently limits pad/tile and concat dynamic results. Reuse the
  existing symbolic-extent term rather than adding a duplicate.<span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent))</span>
- Move capability wording to completed behavior only after implementation and documentation pass.
- Record reasoned no-change conclusions for public Tensor Javadoc/signatures, Compile API,
  Training API, architecture docs/tests, semantic attributes, shape foundation, conformance,
  integration, Gradle, dependencies, and other modules.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [Compile glossary](../../../../glossary.md#compile); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module))</span>

## Acceptance criteria

- Only the two existing production helpers and two focused suites change as executable paths;
  `Tensor.java` changes [Javadoc](../../../../glossary.md#term-javadoc) only under the explicitly authorized [scope](../../../../glossary.md#term-scope) expansion.
- Pad and tile preserve request [validation](../../../../glossary.md#term-validation)/[ownership](../../../../glossary.md#term-ownership-meanings) and accept every [Dimension](../../../../glossary.md#dimension) category only via
  `DimensionExpressions` public methods.
- [Padding](../../../../glossary.md#padding) derives [canonical](../../../../glossary.md#term-canonical-form) `N + before + after`; [tiling](../../../../glossary.md#tiling) derives canonical `repeat * N`.
- Concat accepts ordered sums of static, named, exact-expression, and constrained-unknown selected
  extents and preserves non-selected references.
- Tests cover dynamic plus [constant](../../../../glossary.md#term-constant), `N + M`, repeated terms, pre-existing expressions, neutral
  identities, structural equality, and exact canonical references.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([structural equality glossary](../../../../glossary.md#term-structural-equality))</span>
- Existing scalar/static/zero, data type, eligibility, semantics, provenance, layout, storage,
  freshness, ownership, validation, and overflow coverage remains passing except for the specified
  reference identity adjustment.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [data type glossary](../../../../glossary.md#data-type--datatype); [provenance glossary](../../../../glossary.md#provenance); [layout glossary](../../../../glossary.md#layout); [overflow glossary](../../../../glossary.md#term-overflow); [reference identity glossary](../../../../glossary.md#term-reference-identity))</span>
- Obsolete dynamic errors are no longer asserted; unrelated exact failures and no-ID effects remain
  covered. Arithmetic overflow always precedes Tensor allocation.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- No binding/evaluation, expression-form, graph inference, value/storage, cross-layer, dependency,
  build, or architecture change appears.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [evaluation glossary](../../../../glossary.md#term-evaluation); [graph glossary](../../../../glossary.md#graph); [inference glossary](../../../../glossary.md#term-inference); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Javadocs, independent documentation review, and task/master/roadmap/capability status are final
  before Complete.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) focused suites as needed during development. After executable Java stabilizes, record one final
[module](../../../../glossary.md#term-module) run:

```bash
./gradlew :modules:model:test
```

The documentation pass reuses that evidence unless it changes executable Java, then runs:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

It also validates changed Markdown links/anchors, fences, whitespace, final newlines, examples,
eleven-path [scope](../../../../glossary.md#term-scope), package placement, and synchronized statuses. Source/diff review confirms
public [Tensor](../../../../glossary.md#tensor) [signatures](../../../../glossary.md#term-signature) and bodies are untouched, arithmetic comes only from
`DimensionExpressions`, and no value/storage/cross-layer import appears.

Repository-wide tests are deferred to the foundation-contract [checkpoint](../../../../glossary.md#term-checkpoint) after task 0018N and [CI](../../../../glossary.md#term-continuous-integration-ci).
Do not duplicate a successful [model](../../../../glossary.md#term-model) suite in the documentation context without executable changes
or a recorded concrete reason.

## Dependencies

- [Task 0002](0002-shape-and-dimension-model.md) — [Dimension](../../../../glossary.md#dimension) and [Shape](../../../../glossary.md#shape) foundation.
- [Task 0017J](0017j-pad-and-tile-tensor-expressions.md) — existing pad/tile behavior.
- [Task 0017L](0017l-tensor-composition-expressions.md) — existing concat behavior.
- [Task 0018M](0018m-symbolic-extent-expressions.md) — [canonical](../../../../glossary.md#term-canonical-form) arithmetic consumed here.

## Follow-up tasks

- Task 0018N, Typed scalar value contract, remains the next Draft frontier and owns the
  foundation-contract checkpoint. Do not create its specification here.<span data-glossary-navigation="true"> ([Typed scalar value glossary](../../../../glossary.md#typed-scalar-value--scalarvalue); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Task 0018R owns later [slice](../../../../glossary.md#slice)/window cleanup and any remaining supported window-formula adoption.
- [Binding](../../../../glossary.md#term-binding) and [evaluation](../../../../glossary.md#term-evaluation) require later [compiler](../../../../glossary.md#term-compiler)/[prepare](../../../../glossary.md#prepare)/[runtime](../../../../glossary.md#term-runtime) tasks.

## Architecture impact

Expected impact: None. [Model](../../../../glossary.md#term-model) already owns [Tensor](../../../../glossary.md#tensor)/[Shape](../../../../glossary.md#shape) semantics, while [compiler](../../../../glossary.md#term-compiler) owns [graph](../../../../glossary.md#graph)-wide
[inference](../../../../glossary.md#term-inference). Stop if implementation needs [binding](../../../../glossary.md#term-binding), [evaluation](../../../../glossary.md#term-evaluation), another [module](../../../../glossary.md#term-module), or [lifecycle](../../../../glossary.md#lifecycle) changes.

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
  helper reconstructed an equal static value.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([Canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Keep all [layouts](../../../../glossary.md#layout) unresolved because logical [extent](../../../../glossary.md#term-extent) [precision](../../../../glossary.md#term-precision) does not imply physical geometry.
- Independent documentation review found stale public Tensor Javadocs that contradicted the
  removed dynamic rejection paths. Explicit user authorization expanded the maximum from ten to
  eleven paths solely for Javadoc correction; declarations, signatures, bodies, and executable
  behavior remain unchanged.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Javadoc glossary](../../../../glossary.md#term-javadoc); [signatures glossary](../../../../glossary.md#term-signature))</span>

## Known limitations

- Symbolic results cannot yet be bound or evaluated.
- Division nodes and unknowns are atomic terms; non-selected concat dimensions still require
  structural equality.<span data-glossary-navigation="true"> ([nodes glossary](../../../../glossary.md#node); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [dimensions glossary](../../../../glossary.md#dimension); [structural equality glossary](../../../../glossary.md#term-structural-equality))</span>
- Pad [constants](../../../../glossary.md#term-constant) and other provisional public contracts remain for their dedicated cleanup tasks.
- Window, [convolution](../../../../glossary.md#term-channels-first-convolution-layers), [pooling](../../../../glossary.md#term-pooling), and other helpers do not adopt symbolic arithmetic here.

## [Validation](../../../../glossary.md#term-validation) evidence

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
applied General, [API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc), [Planning](../../../../glossary.md#term-planning), and Example profiles. It independently reviewed the
[architecture contract](../../../../glossary.md#architecture-contract), documentation/planning rules, capability baseline, master plan, roadmap,
tasks 0002/0017J/0017L/0018M/0018M1, final helpers and focused tests, public [Tensor](../../../../glossary.md#tensor) contracts,
[shape](../../../../glossary.md#shape)-expression foundation, Tensor API, glossary, [Compile](../../../../glossary.md#compile) API, Training API, and Java 26 [Gradle](../../../../glossary.md#term-gradle)
configuration.

- `./gradlew :modules:model:javadoc` passed with `BUILD SUCCESSFUL in 2s`; generated public Tensor
  documentation contains the `N + 5`, `4 * N`, and `N + M` contracts and no removed dynamic-error
  claims.
- A targeted Ruby check validated 452 local Markdown links, including 139 anchors, across the six
  changed Markdown files.
- Targeted Ruby checks confirmed balanced code fences, final newlines, and no trailing whitespace
  across those six files. The `N + 5`, `4 * N`, and `N + M` examples were recalculated against the
  canonical factory rules and focused tests.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Source/diff checks confirmed the exact eleven authorized paths, correct existing package
  placement, unchanged public Tensor signatures and bodies, Javadoc-only `Tensor.java` changes,
  before-then-after pad calls, one tile multiply call per axis, static-zero encounter-order concat
  folding, and no helper arithmetic outside <span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis))</span>`DimensionExpressions`.
- Source/import review found no new value, storage, graph-inference, compiler, prepare, runtime,
  backend, dependency, build, or cross-module state. The two helper imports are limited to JDK and
  existing model types.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [inference glossary](../../../../glossary.md#term-inference); [compiler glossary](../../../../glossary.md#term-compiler); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [dependency glossary](../../../../glossary.md#term-dependency); [module glossary](../../../../glossary.md#term-module); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [model glossary](../../../../glossary.md#term-model))</span>
- Task, master-plan row, roadmap row, current frontier, and capability wording are synchronized at
  Complete. Task 0018N remains Draft and no detailed 0018N specification exists.
- `git diff --check` passed on the final combined change.

The original ten-path cap was insufficient because independent review found that the public
`Tensor.pad`, `tile`, and `concat` Javadocs still promised the removed dynamic rejection behavior.
Explicit user authorization expanded the maximum to eleven paths solely for those Javadoc
corrections; declarations, [signatures](../../../../glossary.md#term-signature), bodies, and executable behavior remain unchanged.

Reasoned no-change conclusions:

- Compile API remains accurate because it identifies current pad/tile/composition metadata and
  keeps capture, inference, canonicalization, decomposition, and execution planned without
  specifying the replaced local fallback rules.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [capture glossary](../../../../glossary.md#term-capture-meanings); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [decomposition glossary](../../../../glossary.md#term-decomposition))</span>
- Training API remains accurate because it contains only planned training ownership and no
  pad/tile/concat Shape contract.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- [`Dimension`](../../../../glossary.md#dimension), `DimensionExpression`, `ExpressionDimension`, `DimensionExpressions`, and `Shape`
  remain accurate unchanged because this task consumes their completed canonical arithmetic and
  does not add an expression form, binding, evaluation, or category.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- Pad/tile/composition semantic kinds and attributes remain accurate unchanged because they do not
  own input Shapes or result-extent derivation.<span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent))</span>
- Architecture documents/tests, backend conformance tests, and integration tests remain unchanged
  because no dependency, boundary, backend, or end-to-end executable behavior changes.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance))</span>
- Java 26 Gradle configuration, dependencies, other modules, values/storage, layout derivation,
  gradients, compiler, prepare, runtime, backend, and execution remain unchanged because the task
  is confined to model metadata derivation and its documentation. Result layout is still
  deliberately unresolved.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [gradients glossary](../../../../glossary.md#term-gradient))</span>

## Implementation notes

- `paddedShape` applies `DimensionExpressions.addConstant` twice per axis in before-then-after
  order.
- `tiledShape` applies `DimensionExpressions.multiply` exactly once per [axis](../../../../glossary.md#term-axis).
- `concatShape` starts with `StaticDimension(0)` and encounter-order folds every selected
  Dimension through <span data-glossary-navigation="true"> ([Dimension glossary](../../../../glossary.md#dimension))</span>`DimensionExpressions.add`.
- Focused tests replace obsolete dynamic-error assertions with canonical named, repeated,
  pre-existing-linear, division, constrained-unknown, neutral-reference, structural-equality, and
  checked-overflow coverage while retaining metadata, provenance, ownership, validation, and
  no-ID effects.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [linear glossary](../../../../glossary.md#term-linear-meanings); [overflow glossary](../../../../glossary.md#term-overflow); [metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [validation glossary](../../../../glossary.md#term-validation))</span>
- The documentation pass changed no executable Java. It finalized the two helper contracts, the
  explicitly authorized public Tensor Javadocs, Tensor API examples, glossary definitions,
  capability status, and synchronized planning evidence.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>

## Completion summary

- Completed changes: Adopted canonical symbolic extent arithmetic for pad, tile, and concat while
  preserving validation, metadata, unresolved layout, provenance, freshness, and ID behavior.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [extent glossary](../../../../glossary.md#term-extent); [validation glossary](../../../../glossary.md#term-validation); [metadata glossary](../../../../glossary.md#term-metadata); [layout glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance))</span>
- Files changed or created: Eleven authorized paths: two helper sources, two focused tests,
  Javadoc-only <span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`Tensor.java`, Tensor API, glossary, capability baseline, this task, model master
  plan, and roadmap.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [model glossary](../../../../glossary.md#term-model))</span>
- Tests and validation: Reused the final 25-test focused and 766-test/88-suite model results;
  model Javadoc, generated-page inspection, 452-link/139-anchor Markdown validation,
  fence/newline/whitespace checks, example review, exact eleven-path/package/signature/import/
  status/no-0018N-spec checks, and final diff checks passed.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature))</span>
- Documentation-agent review: Clean context
  `/root/task_0018m1_implementation/task_0018m1_documentation` completed the independent targeted
  review using General, API/Javadoc, Planning, and Example profiles and changed no executable Java.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: Tensor API, glossary, capability baseline, task evidence, master plan, and
  roadmap now describe current canonical symbolic pad/tile/concat Shapes and deferred
  binding/evaluation.<span data-glossary-navigation="true"> ([Shapes glossary](../../../../glossary.md#shape); [binding glossary](../../../../glossary.md#term-binding); [evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- Javadoc review: Helper and public Tensor contracts now document exact formulas, neutral
  references, checked static/coefficient/offset overflow, unresolved layout, ownership, failures,
  and lifecycle boundaries. Public declarations and bodies are unchanged.<span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Glossary impact: Corrected the existing Dimension, padding, tiling, composition, and Tensor
  wording without adding a duplicate symbolic-extent term.<span data-glossary-navigation="true"> ([Dimension glossary](../../../../glossary.md#dimension); [padding glossary](../../../../glossary.md#padding); [tiling glossary](../../../../glossary.md#tiling))</span>
- Unresolved issues: None.
- Follow-up required: None. Task 0018N remains Draft without a detailed specification.

Status: Complete
