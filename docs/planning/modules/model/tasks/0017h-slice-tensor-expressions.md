# Task 0017H: Slice Tensor Expressions

## Status

Complete

## Goal

Add public general and single-axis positive-step slice expression construction to `Tensor`.

Both forms must normalize caller coordinates against locally known static dimensions, preserve
rank and unaffected Dimension references, derive resolved strided-view geometry when it is safe
and meaningful, record exact `SLICE` semantics with one-input provenance, and return fresh
storage-free Tensor expressions. This task constructs model metadata only; it does not read values,
attach storage aliases, execute slicing, define gradients, or choose materialization.

## Scope

- Add exactly these public instance methods to `Tensor`:
  - `slice(long[] starts, long[] ends, int[] axes, long[] steps)`
  - `sliceAxis(int axis, long fromInclusive, long toExclusive)`
- Add one field-free package-private final `TensorSliceExpressions` helper in `model.tensor`.
- Give the helper exactly one general entry, one single-axis entry, and six private methods defined
  by this task.
- Require non-null general-request arrays with equal lengths and defensively clone all four before
  inspecting their elements.
- Normalize positive or negative raw axes exactly once against input rank and reject duplicates
  after normalization.
- Require strictly positive steps; do not support reverse slicing.
- Require every selected input dimension to be statically known. Preserve unselected static or
  dynamic Dimension references exactly.
- Normalize negative starts/ends by adding selected dimension size once, then clamp every bound
  into inclusive range `[0, dimensionSize]`.
- Use half-open `[start, end)` meaning and checked overflow-safe positive-step extent calculation.
- Permit empty results when normalized start is at least normalized end or the selected/input
  shape already contains a zero extent.
- Construct normalized `SliceAttrs` in original request-entry order.
- Derive one new view-marked LayoutDescriptor for non-empty results when input layout is resolved;
  leave layout unresolved when input layout is unresolved or result element count is zero.
- Preserve input data type and gradient eligibility; create exact `SLICE`, `SliceAttrs`, and
  ordered provenance `[input]`; attach no label or storage.
- Keep every valid request explicit and fresh, including identity, empty-entry, unit-step, repeated,
  and nested slices.
- Update only Tensor API inventory plus one focused expression test.
- Finalize Javadocs, Tensor/Compile API, glossary, task evidence, master plan, and roadmap through
  the mandatory independent documentation pass.

## Out of scope

- another public overload, List-based API, nullable/default axes or steps, omitted bounds,
  ellipsis, open-ended slice object, scalar indexing, `select`, or task-0018 indexing work
- `int[]` bound/step overloads retained solely for legacy signature compatibility
- zero or negative steps, reverse slicing, axis repetition, axis sorting, slice-chain
  canonicalization, identity elimination, or bounds reordering
- slicing a selected dynamic dimension, symbolic bound expressions, constraint generation, graph-
  wide Shape inference, or runtime dimension binding
- rejecting empty slices merely because legacy rejected them; zero-extent Shape is a supported
  current model value
- reading, copying, scattering, or eagerly materializing values; attaching, retaining, replacing,
  or observing host storage; asserting a physical alias or zero-copy execution
- modifying SliceKind, SliceAttrs, Shape, Dimension, LayoutDescriptor, TensorDescriptor,
  TensorFactory, TensorProvenance, Operation, or any completed Java contract/test
- slice backward, gradients, autograd, graph capture, compiler passes, planning requirements,
  prepare, runtime, backend lowering/execution, engine, trace, ONNX, or training behavior
- another production helper/type, dependency, Gradle/build option, architecture change, another
  module, or task-0017I specification

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
- [Task 0017G](0017g-slice-semantics.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The selected baseline includes general strided `slice` and one-axis `sliceAxis`. The read-only
legacy branch exposes four parallel `int[]` parameters, accepts negative axes/bounds, clamps bounds
to a concrete dimension, supports positive steps, rejects duplicate axes and empty dimensions,
and derives offset/stride view geometry. Its one-axis convenience delegates to one general entry
with step one.

The new API retains the capability but uses `long[]` bounds/steps to match current long Shape and
layout geometry. It requires all four arrays explicitly rather than preserving nullable legacy
defaults. It permits zero-extent results because current Shape, LayoutDescriptor, TensorDescriptor,
storage allocation, and factory contracts deliberately support empty tensors. It rejects selected
dynamic dimensions because local normalization and clamping cannot prove their result extent.

Legacy storage aliases, eager value access, graph builders, gradient callbacks, traits, compiler,
lowering, kernels, and runtime/backend behavior are not copied.

## Architecture constraints

- `Tensor` remains public mutable API state, not IR. The new methods build fresh expression
  metadata through the existing package-private derived factory seam.
- Slice semantic identity/attributes come only from completed task 0017G.
- The helper may inspect immutable descriptor, Shape, and optional layout metadata, but never
  values, host storage, runtime residency, device state, or backend capability.
- Raw caller arrays are request syntax only. Normalized/clamped values alone enter SliceAttrs.
- A selected dynamic Dimension is rejected locally instead of becoming a hidden symbolic
  constraint. Unselected dynamic Dimensions retain exact references.
- Result rank equals input rank. Selected dimensions become new StaticDimension extents;
  unselected dimensions retain exact references.
- Resolved layout is logical view metadata only. It does not attach storage or promise physical
  aliasing or zero-copy execution.
- Empty results use unresolved layout because no storage element is referenced and one-past-end
  offsets/strides would be arbitrary or could overflow despite having no observable geometry.
- Compiler owns canonicalization and capture; planning/backend prepare own materialization and
  lowering; compiler-generated/training work owns slice backward.
- No dependency or module boundary changes are authorized.

## Package impact

No package is added or moved.

- `io.github.pho001.synaptik.model.tensor.Tensor` — receives the two public convenience surfaces.
- `io.github.pho001.synaptik.model.tensor.TensorSliceExpressions` — owns local validation,
  normalization, Shape/layout derivation, and semantic construction.
- `TensorSliceExpressionTest` — mirrors the package for focused helper/API validation.
- `TensorTest` — changes only its exact public API inventory and reflection assertions.

SliceKind and SliceAttrs remain in `model.operation.layout`; no semantic contract moves into the
Tensor package.

## Required contract

### Public Tensor methods

Add exactly:

```java
public Tensor slice(long[] starts, long[] ends, int[] axes, long[] steps) {
    return TensorSliceExpressions.apply(this, starts, ends, axes, steps);
}

public Tensor sliceAxis(int axis, long fromInclusive, long toExclusive) {
    return TensorSliceExpressions.applyAxis(this, axis, fromInclusive, toExclusive);
}
```

Each public method contains one return statement and exactly one matching helper call. Neither
method performs validation, allocates another request representation, reads fields directly, or
calls the other public method. Both are non-static and non-synchronized.

`slice` accepts arrays of equal length in caller entry order. Empty arrays are valid and produce a
fresh explicit identity slice. `sliceAxis` is equivalent to one entry with the supplied axis and
bounds plus step `1L`; it has no distinct operation kind.

### Helper shape

Create one package-private final, field-free class with one private zero-argument constructor and
exactly these eight static methods:

```java
static Tensor apply(Tensor input, long[] starts, long[] ends, int[] axes, long[] steps)
static Tensor applyAxis(Tensor input, int axis, long fromInclusive, long toExclusive)
private static SliceAttrs normalize(
        Shape inputShape, long[] starts, long[] ends, int[] axes, long[] steps)
private static long normalizeBound(long rawBound, long dimensionSize)
private static long sliceExtent(long start, long end, long step)
private static Shape deriveShape(Shape inputShape, SliceAttrs attrs)
private static Optional<LayoutDescriptor> resolveViewLayout(
        TensorDescriptor inputDescriptor, Shape resultShape, SliceAttrs attrs)
private static Tensor create(
        Tensor input,
        TensorDescriptor inputDescriptor,
        Shape resultShape,
        Optional<LayoutDescriptor> resultLayout,
        SliceAttrs attrs)
```

Add no field, nested type, alternate constructor, overload, cache, mutable state, or extra method.

### General validation and normalization

`apply` performs this exact order:

1. null-check `input`, `starts`, `ends`, `axes`, and `steps`, in order, using those exact messages;
2. require equal array lengths, otherwise throw `IllegalArgumentException` with exact message
   `starts, ends, axes, and steps must have matching lengths`;
3. clone starts, ends, axes, and steps once each, in that order;
4. read the exact input descriptor and Shape once;
5. call `normalize` once with only the private copies;
6. call `deriveShape` once;
7. call `resolveViewLayout` once;
8. call `create` once and return its result.

Null/length failures occur before cloning, descriptor inspection, allocation of semantic lists, or
Tensor identity consumption. Caller arrays are never retained or mutated, and concurrent caller
mutation after the clones cannot change the expression.

`normalize` processes entries in ascending index order. It creates one rank-sized `boolean[]` for
normalized-axis duplicate detection and four mutable constructor-local lists. At each index:

1. normalize raw `int axis` once using `long`; a negative value adds rank once;
2. reject an out-of-range result with exact message
   `slice axis <raw> at index <index> is outside rank <rank>`;
3. reject the first repeated normalized axis with exact message
   `slice contains duplicate normalized axis <axis> at index <index>`;
4. reject a non-positive step with exact message `steps[<index>] must be positive: <step>`;
5. require `inputShape.dimensions().get(axis)` to be StaticDimension, otherwise throw
   `IllegalArgumentException` with exact message
   `slice axis <axis> at index <index> must have a statically known dimension`;
6. normalize and clamp the copied start/end against that dimension;
7. append normalized start, end, axis, and unchanged positive step to the four lists.

After all entries pass, construct exactly one `SliceAttrs`, which performs the final immutable
snapshots. Do not sort entries or normalize their order.

`normalizeBound` adds `dimensionSize` once when the raw bound is negative, using long arithmetic,
then clamps below zero to zero and above the dimension to the dimension. Every raw long is valid.

`sliceExtent` returns zero when start is at least end. Otherwise it calculates exactly:

```java
1L + (end - 1L - start) / step
```

This form is safe for normalized `0 <= start < end <= dimensionSize <= Long.MAX_VALUE` and avoids
overflow from `end - start + step - 1`.

### Single-axis delegation

`applyAxis` creates exactly four private one-element arrays and calls `apply` once:

```java
return apply(
        input,
        new long[] {fromInclusive},
        new long[] {toExclusive},
        new int[] {axis},
        new long[] {1L});
```

It performs no independent normalization or semantic construction. Therefore both public forms
share identical errors, clamping, geometry, provenance, and ID side effects.

### Result Shape

`deriveShape` copies exact input Dimension references into a rank-sized array. For each SliceAttrs
entry in order, replace the selected axis with one new `StaticDimension(sliceExtent(...))`. Return
`Shape.ofDimensions` once. Scalar plus empty entries remains the canonical scalar Shape.

Unselected dynamic, static, and zero dimensions preserve exact references. Selected dimensions
are always static because normalize rejected dynamic selection. Result rank never changes.

Examples for input Shape `[3, 6]`:

| Request | Normalized attrs | Result Shape |
|---|---|---|
| starts `[0,1]`, ends `[3,6]`, axes `[0,1]`, steps `[1,2]` | unchanged | `[3,3]` |
| starts `[-2]`, ends `[-1]`, axes `[-1]`, steps `[1]` | starts `[4]`, ends `[5]`, axes `[1]` | `[3,1]` |
| starts `[5]`, ends `[2]`, axes `[1]`, steps `[1]` | unchanged bounds | `[3,0]` |
| all arrays empty | empty attrs | `[3,6]` with exact Dimension references |

### Result layout

`resolveViewLayout` reads the input layout Optional once.

- If input layout is unresolved, return `Optional.empty()`.
- If `resultShape.knownElementCount()` is zero, return `Optional.empty()` even when input layout is
  resolved.
- Otherwise copy input strides once and start from exact input storage offset.
- For each attrs entry in order, update offset with checked
  `offset + start * originalInputStride(axis)` and replace that axis stride with checked
  `originalInputStride(axis) * step`.
- Use the original input stride for both calculations so request-entry processing cannot compound
  a stride accidentally.
- Create exactly one `LayoutDescriptor.of(resultShape, resultStrides, resultOffset, true)` and
  return it in Optional.

Every resolved input layout kind is accepted, including dense, offset, strided, and broadcast.
LayoutDescriptor reclassifies result kind/span. Arithmetic overflow propagates as
`ArithmeticException`. View metadata does not attach storage or guarantee executable aliasing.

### Result construction

`create` constructs exactly:

```java
TensorDescriptor descriptor = new TensorDescriptor(
        inputDescriptor.dataType(),
        resultShape,
        resultLayout,
        inputDescriptor.requiresGrad());
Operation operation = new Operation(SliceKind.SLICE, attrs);
TensorProvenance provenance = new TensorProvenance(operation, List.of(input));
return TensorFactory.createDerived(descriptor, Optional.empty(), provenance);
```

Every success consumes one fresh Tensor ID, retains exact input data type and eligibility, and has
absent label/storage. Failures before `createDerived` consume no ID. Factory exhaustion remains
the final failure. No storage, previous provenance, or input label is inspected or retained except
the exact input reference in new provenance.

## Affected files

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorSliceExpressions.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorSliceExpressionTest.java`
- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum scope

At most the ten paths above may change. Production is limited to Tensor plus one package-private
helper; tests to Tensor inventory plus one focused suite; documentation to public API/status and
planning evidence.

If another helper/type/test, overload, semantic/foundational edit, dependency, build/architecture
change, or eleventh path is needed, stop and report. Do not create task 0017I.

## Javadoc requirements

- Fully document both public methods and helper type/constructor/all eight methods.
- Explain four-array parallel ownership, half-open bounds, positive steps, negative axis/bound
  normalization, clamping, entry order, duplicate rules, and static selected dimensions.
- Explain empty arrays, empty results, zero extents, and why empty layout remains unresolved.
- Document Shape Dimension-reference retention and selected static extent calculation with
  concrete numeric examples.
- Document resolved stride/offset calculations, checked arithmetic, every accepted input layout
  kind, view flag, unresolved cases, and lack of physical alias guarantee.
- Document result type, Shape, layout, eligibility, label, storage, Operation, SliceAttrs,
  provenance, freshness, failure order, and identifier side effects.
- Explain single-axis step-one delegation and why no distinct operation kind exists.
- Independently review SliceKind/SliceAttrs, Shape/Dimension, LayoutDescriptor, TensorDescriptor,
  TensorFactory, TensorProvenance, and adjacent view-expression Javadocs; record why unchanged
  contracts remain accurate or stop on discrepancy.

## Acceptance criteria

- Tensor exposes exactly the two specified methods; declared public method count rises 85 to 87.
- Each public method delegates once; helper is final/package-private/field-free with exact private
  constructor and eight methods.
- Exact null/length/order messages, defensive clones, axis/step/static-dimension validation,
  negative normalization, clamping, duplicate handling, and extent arithmetic match this task.
- General empty arrays and scalar identity are valid and fresh.
- Empty slices produce zero-extent Shape and unresolved layout rather than legacy rejection.
- Shape preserves exact unaffected Dimension references and rank.
- Resolved non-empty dense/offset/strided/broadcast layouts derive exact checked offset/strides,
  view flag, kind, and span; unresolved/empty results stay unresolved.
- All six data types and valid eligibility states retain exact metadata.
- Result records exact `SLICE`, normalized SliceAttrs, and `[input]`, with absent label/storage and
  fresh identity.
- Single-axis form uses exact one-entry step-one general path.
- Early validation/layout failures consume no ID; final exhaustion behavior is preserved.
- No values/storage/gradient/compiler/planning/runtime/backend behavior or architecture change.
- Independent documentation review and all status/evidence synchronization complete before
  marking Complete.

## Tests / validation

Run before and after documentation review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorSliceExpressionTest --tests io.github.pho001.synaptik.model.tensor.TensorTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

Focused tests cover exact public/helper shape; all data types/eligibility; null/length validation
precedence; caller-array ownership; positive/negative/extreme axes and bounds; clamping; positive
steps; normalized duplicate axes; selected dynamic rejection; unselected dynamic retention;
scalar/empty/zero/static/mixed-dynamic Shapes; exact Dimension identity; extent examples and
overflow-safe large values; dense/offset/strided/broadcast/unresolved/empty layout outcomes;
checked offset/stride overflow; exact operation/attrs/provenance; absent label/storage; dead-storage
non-interference; freshness; nested/identity calls; ID side effects; and exhaustion.

Inspect `javap -p -c -s`, reflection, imports, and source. Confirm two one-call public methods,
exact eight-method helper, four defensive clones, one normalization pass, one Shape derivation,
one layout path, one common construction, and no forbidden behavior. Validate generated Javadoc,
executable examples, Tensor/Compile API/glossary, links/anchors/fences/whitespace, exact ten paths,
synchronized statuses, and no task-0017I specification.

## Dependencies

- 0002 supplies immutable static/dynamic Dimensions, Shape, scalar/zero forms, and axis vocabulary.
- 0003 supplies resolved strides, offsets, view metadata, classification, and spans.
- 0007 supplies resolved-or-unresolved TensorDescriptor.
- 0011–0013 supply Tensor, centralized derived identity, and immutable provenance.
- 0017G supplies SLICE and immutable normalized SliceAttrs.
- Completed neighboring view-expression tasks provide patterns but no extra production dependency.

## Follow-up tasks

- 0017I remains Draft for pad/tile semantic identities and immutable parameters.
- Compiler later owns slice capture and identity/nested-slice canonicalization.
- Compiler-generated/training tasks later own slice-backward scatter semantics.
- Planning/backend prepare later own materialization and concrete view/copy lowering.
- ONNX extension later owns Slice mapping without entering the model or runtime hot path.

Do not create a detailed follow-up specification here.

## Architecture impact

Expected impact: None. Model already owns Tensor expression construction, Shape/layout metadata,
operation semantics, descriptors, and provenance. Lifecycle ownership remains unchanged.

Stop if implementation needs symbolic binding, physical storage aliasing, gradients,
compiler/planning/backend behavior, another dependency, or architecture changes.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0002/0003/0007/0011/0012/0013/0017G/0017H, Tensor API,
Compile API, Training API, glossary, current Dimension/Shape/LayoutDescriptor/LayoutKind/
TensorDescriptor/Tensor/TensorFactory/TensorProvenance/Operation/SliceKind/SliceAttrs contracts and
focused view-expression tests, and Java 26 Gradle configuration.

Implement task 0017H exactly. Modify Tensor.java and add package-private final
TensorSliceExpressions.java. Update TensorTest only for exact two-method API expansion and add
TensorSliceExpressionTest. Add exactly slice(long[],long[],int[],long[]) and
sliceAxis(int,long,long).

The field-free helper has exactly eight specified methods. General slicing null-checks and clones
all four equal-length arrays, normalizes negative axes/bounds once, clamps bounds, rejects duplicate
axes/non-positive steps/selected dynamic dimensions, permits empty results, and constructs
normalized SliceAttrs. Preserve rank and unaffected Dimension references. For non-empty resolved
input geometry, derive checked start-adjusted offset and step-multiplied strides in a new view;
unresolved or empty results stay unresolved. Preserve type/eligibility, create exact SLICE/attrs/
[input], and call createDerived once with no label/storage. Single-axis delegates through one
step-one general entry. Every request is fresh.

Do not modify semantic/foundational contracts, add overloads/types/helpers, support reverse steps
or selected dynamic axes, inspect/copy values/storage, attach physical aliases, define gradients,
capture/canonicalize graphs, or add compiler/planning/prepare/runtime/backend/ONNX behavior,
dependencies, build/architecture changes, or later specs. Stop beyond ten paths or on architecture
uncertainty.

Run all specified validation, then hand actual diff/evidence to a separate clean-context docs agent
in the same change. It must inspect source/tests/generated Javadoc, finalize permitted Javadocs/
Tensor API/Compile API/glossary/planning, record no-change conclusions, and rerun validation.

Update 0017H, model master plan, and roadmap only for planning status/evidence. Do not mark
Complete until both passes succeed. Leave 0017I Draft without a specification. Do not commit/push.
```

## Local decisions

- Public bounds/steps use `long[]` to match current Shape/LayoutDescriptor geometry; axes use
  `int[]` and retain familiar axis addressing.
- All four arrays are explicit and non-null. Legacy nullable/default arrays are not retained
  because the full capability and one-axis convenience do not require ambiguous defaults.
- Selected dynamic dimensions are rejected; local code does not invent symbolic constraints.
- Raw negative bounds add dimension size once and clamp, preserving useful legacy behavior.
- Empty slices are accepted because zero extents are first-class in the current model; the legacy
  rejection is not copied.
- Empty results keep layout unresolved to avoid arbitrary/no-element offset and stride facts.
- Resolved non-empty results are view metadata only, with no attached storage or execution promise.
- Identity, repeated, and nested requests remain explicit; compiler owns canonicalization.

## Known limitations

- Reverse/negative-step and selected-dynamic slicing are unsupported.
- No open-ended bounds/default arrays, physical storage alias, eager value access, gradients,
  compiler capture/canonicalization, materialization, lowering, execution, or ONNX mapping exists.

## Validation evidence

- Architecture, focused boundary documentation, planning/documentation rules, roadmap, model
  capabilities/master plan, completed semantic task 0017G, neighboring view-expression tasks,
  current Shape/Dimension/LayoutDescriptor/Tensor contracts/tests, Tensor/Compile/Training APIs,
  glossary, Java 26 Gradle configuration, and read-only legacy slice implementation/tests were
  reviewed before specifying behavior.
- Legacy evidence confirms general parallel-parameter slicing, negative normalization/clamping,
  positive steps, duplicate-axis rejection, strided offset views, and one-axis step-one delegation.
  Coupled legacy storage, graph, gradient, compiler, kernel, and runtime design is excluded.
- `git diff --check` passed with no whitespace errors.
- Changed-path inventory contains exactly three planning paths: this task, the model master plan,
  and roadmap. No Java, API, glossary, Gradle, AGENTS, ARCHITECTURE, focused architecture, completed
  task, or other-module file changed during planning.
- Markdown structure check passed: 23 level-two sections, 14 balanced code-fence markers, no
  trailing whitespace, and final newline present.
- Every local Markdown link in all three changed files resolves.
- Roadmap contains all 74 ordered unique task rows; task 0017H is linked and Ready at row 61, task
  0017I remains Draft at row 62, and no task-0017I specification exists.
- Task, master plan, and roadmap consistently identify 0017H as the next Ready frontier.
- Package/scope review confirms no package addition: two public methods and the package-private
  helper remain in `model.tensor`, the focused test mirrors that package, and implementation is
  bounded to the ten authorized paths without a dependency or architecture change.
- No Gradle test was run because this planning-only change modifies no production or test code.
- Clean implementation context `/root/implement_model_0017h` added the two public delegating
  methods, exact field-free eight-method helper, Tensor API inventory updates, and focused
  expression test before handing the actual shared-tree diff to the independent documentation
  context.
- Clean documentation context
  `/root/implement_model_0017h/review_model_0017h_docs` applied the General, API/Javadoc,
  Planning, and Example profiles. It read the architecture and focused boundary documents,
  documentation/planning rules, model capabilities/master plan, prerequisite tasks, Tensor/
  Compile/Training APIs, glossary, final production/tests, generated Javadoc, Java 26 Gradle
  configuration, and actual diff rather than relying on the implementation handoff.
- The documentation pass corrected the public Javadoc distinction between resolved input geometry
  and a non-empty result, completed element-count/layout arithmetic failure wording, and confirmed
  the helper type, constructor, and all eight methods document parameters, results, ownership,
  normalization, Shape/layout derivation, provenance, freshness, and cross-layer exclusions.
- Tensor API now documents four-array ownership/order, half-open positive-step normalization,
  static selected dimensions, empty requests/results, exact unaffected Dimension retention,
  checked offset/stride derivation for every resolved input layout kind, logical view status,
  single-axis delegation, result metadata, failure/ID behavior, and current limitations. Compile
  API now lists slice expressions as current model inputs while capture, canonicalization,
  gradient scatter, materialization, backend/ONNX lowering, and execution remain planned. The
  glossary aligns the reusable slice and provenance terminology with those boundaries.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorSliceExpressionTest --tests
  io.github.pho001.synaptik.model.tensor.TensorTest` — `BUILD SUCCESSFUL`; XML reports contain 11
  slice-expression tests and 14 Tensor tests, with zero failures, errors, or skips.
- `./gradlew :modules:model:test` — `BUILD SUCCESSFUL`; 68 XML suites contain 554 tests with zero
  failures, errors, or skips.
- `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL`. Generated `Tensor.html` contains both
  public methods, non-empty-result/unresolved-layout rules, complete parameter/return/failure
  documentation, exact SLICE semantics, and physical-alias/compiler/backend/execution boundaries.
  Package-private helper Javadocs were reviewed in source because standard public Javadoc omits
  the helper.
- `./gradlew test` — `BUILD SUCCESSFUL`; all 36 actionable tasks completed without a failing task
  in the final repository lifecycle run.
- `javap -classpath modules/model/build/classes/java/main -p -c -s` for `Tensor` and
  `TensorSliceExpressions`, focused reflection tests, import scans, and source inspection confirmed
  exactly 87 declared public Tensor methods; the two exact one-call slice methods; one private
  zero-argument helper constructor; exactly eight static helper methods and no fields; four clones
  in order; one normalization, Shape, layout, and creation path; checked arithmetic; exact
  descriptor/SLICE/attrs/`[input]` construction; and no forbidden layer import or behavior.
- The documented Java 26 `SliceExpressionExample` compiled and ran against model classes. It
  printed Shape `[3, 3]`, normalized attributes, strides `[6, 2]`, offset `1`, `STRIDED`, span
  `18`, view/provenance/storage facts, and normalized one-axis Shape `[3, 1]` exactly as shown.
- A targeted Markdown path-and-heading validator resolved all 332 local links and anchors across
  the six changed documentation/planning files. Fence counts are balanced, trailing-whitespace
  checks found no matches, and every changed file has a final newline.
- Final scope review contains exactly the ten authorized paths: Tensor and its slice helper, the
  two Tensor tests, Tensor API, Compile API, glossary, this task, model master plan, and roadmap.
  Task/master-plan/roadmap status is synchronized as `Complete`; task 0017I and later rows remain
  Draft, and no task-0017I specification exists. `git diff --check` passes.
- `SliceKind`/`SliceAttrs`, Dimension/Shape, LayoutKind/LayoutDescriptor, TensorDescriptor,
  TensorFactory, TensorProvenance, Operation, and adjacent view-expression contracts remain
  accurate unchanged: this task composes their normalized semantics, immutable geometry,
  eligibility, central identity, and provenance contracts without changing them. Completed task
  specifications likewise remain historical contracts and required no edits.
- Training API remains unchanged because no gradient object/rule, autograd, parameter, optimizer,
  session, or training execution behavior was added. Capabilities remain unchanged because they
  already select general/single-axis slicing and logical sliced views without serving as task-
  completion evidence.
- `ARCHITECTURE.md`, focused architecture/ADRs/tests, backend-conformance and integration tests,
  Gradle/dependencies, and other modules remain unchanged because this task stays within existing
  model ownership and adds no dependency rule, compiler/planning/prepare/runtime/backend behavior,
  executable alias, end-to-end behavior, or build requirement.

## Implementation notes

- Added exactly `Tensor.slice(long[], long[], int[], long[])` and
  `Tensor.sliceAxis(int, long, long)` as one-call delegations.
- Added the package-private field-free `TensorSliceExpressions` helper with the exact eight methods
  for deterministic request ownership, normalization/clamping, same-rank Shape derivation,
  conditional checked view geometry, and fresh storage-free SLICE provenance.
- Expanded Tensor reflection inventory and added the focused 11-test slice suite covering all
  required API, validation, Shape, layout, metadata, freshness, storage-noninterference, and ID
  cases.
- Finalized public/helper Javadocs, Tensor and Compile API references, glossary terminology, and
  planning status/evidence without changing architecture or another contract.

## Completion summary

- Completed changes: Implemented and documented general and single-axis positive-step slice Tensor
  expression construction with defensive request ownership, selected-static normalization,
  zero-extent support, conditional checked view geometry, and exact fresh provenance.
- Files changed or created: Exactly the ten authorized production, test, API, glossary, task,
  master-plan, and roadmap paths.
- Tests and validation: Focused 25 tests, all 554 model tests, model Javadoc, root tests, compiled
  Java 26 example, bytecode/reflection/import/source checks, generated-Javadoc review, 332 local
  link/anchor checks, fence/whitespace/final-newline checks, exact scope/status review, no-0017I-
  spec check, and `git diff --check` passed.
- Documentation-agent review: Clean context
  `/root/implement_model_0017h/review_model_0017h_docs` completed the independent pass using
  General, API/Javadoc, Planning, and Example profiles.
- Documentation impact: Slice expression construction is current in Tensor API, Compile API, and
  glossary; compiler/training/planning/prepare/runtime/backend/ONNX execution boundaries remain
  explicitly planned or separately owned.
- Javadoc review: Both public methods and helper type/constructor/eight methods are final; related
  semantic, Shape/layout, descriptor/factory/provenance/operation, and adjacent view contracts
  remain accurate unchanged for the reasons recorded above.
- Glossary impact: Slice and provenance entries now distinguish semantic attributes, public
  expression construction, logical view metadata, and physical/executable behavior.
- Architecture impact: None; no architecture contract, ADR, architecture test, dependency, or
  module boundary changed.
- Unresolved issues: None.
- Follow-up required: None for task 0017H. Task 0017I remains Draft without a detailed
  specification.

Status: Complete
