# Task 0017J: Pad and Tile Tensor Expressions

## Status

Complete

## Goal

Add public constant-padding and per-axis tiling expression construction to `Tensor`.

Both methods must validate complete rank-aligned requests, derive every locally provable result
Shape, retain exact data type and gradient eligibility, record typed pad/tile semantics with
one-input provenance, and return fresh storage-free expressions with unresolved layout. This task
does not allocate or populate output values, convert padding constants, define gradients, choose
materialization, lower operations, or execute work.

## Scope

- Add exactly these public instance methods to Tensor:
  - `pad(long[] before, long[] after, double constantValue)`
  - `tile(long... repeats)`
- Add one field-free package-private final `TensorPadTileExpressions` helper in `model.tensor`.
- Give the helper exactly two package-private entry methods and three private shared/Shape methods.
- Require non-null request arrays whose length equals input rank; defensively clone before
  constructing semantic attributes.
- Use PadAttrs and TileAttrs as the sole width/repeat validation and immutable snapshot boundary.
- Preserve every raw double padding constant without eager conversion or validation.
- Derive padding extents with checked addition and tile extents with checked multiplication.
- Preserve an exact dynamic Dimension reference only when padding is zero on both sides or repeat
  count is one; reject transformations of dynamic dimensions that cannot be represented locally.
- Accept scalar empty requests and zero-extent static dimensions.
- Preserve input DataType and requiresGrad for all six current data types.
- Keep result layout unresolved for every pad/tile expression, including identity requests.
- Create exact PAD/PadAttrs or TILE/TileAttrs operation, ordered provenance `[input]`, absent label,
  absent storage, and fresh identity.
- Update TensorTest only for exact public API inventory and add one focused expression suite.
- Finalize Javadocs, Tensor/Compile API, glossary, task evidence, master plan, and roadmap through
  the mandatory independent documentation pass.

## Out of scope

- another public overload, int-array compatibility overload, List-based API, per-axis padding
  constants, default padding value, default repeats, or partial-rank requests
- reflect, edge, wrap, symmetric, negative/cropping, tensor-valued, or another padding mode
- zero/negative repeats, implicit leading axes, rank promotion, element-wise repeat, interleave,
  broadcast expand, or a scalar-repeat overload
- eagerly converting constantValue to FLOAT64/FLOAT32/BFLOAT16/INT32/INT64/BOOL, rejecting NaN or
  infinity, applying range checks, saturation, truncation, BFLOAT16 rounding, or BOOL truthiness
- changing PadKind, PadAttrs, TileKind, TileAttrs, DataType, Shape, Dimension, LayoutDescriptor,
  TensorDescriptor, TensorFactory, TensorProvenance, Operation, or completed tests/contracts
- deriving resolved output layout, claiming a view/alias, attaching or inspecting host storage,
  allocating output, copying/repeating/filling values, or choosing physical materialization
- gradients, pad-backward slice, tile-backward reduction, autograd, graph capture, compiler passes,
  planning, prepare, runtime, backend lowering/kernels, engine, trace, ONNX, or training behavior
- another helper/type/test, dependency, Gradle/build option, architecture change, another module,
  or task-0017K specification

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
- [Task 0003](0003-layout-descriptor-model.md)
- [Task 0007](0007-tensor-descriptor-model.md)
- [Task 0011](0011-public-tensor-skeleton.md)
- [Task 0012](0012-tensor-factory.md)
- [Task 0013](0013-tensor-provenance-skeleton.md)
- [Task 0017I](0017i-pad-and-tile-semantics.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The selected baseline exposes constant `pad(int[] before, int[] after, double constantValue)` and
`tile(int... repeats)`. Legacy code requires one width/repeat per axis, non-negative padding,
positive repeats, checked int Shape arithmetic, and preserves input DataType. Its CPU route later
interprets the stored double as exact FLOAT64, Java-narrowed FLOAT32/integral, binary32-to-BFLOAT16,
or zero/non-zero BOOL. It also couples model construction to eager storage, gradients, compiler,
kernels, and runtime behavior.

The new public API widens dimensions to long. It retains raw double semantics and all current data
types without doing conversion in model expression construction. Concrete lowering/execution must
eventually implement a documented cross-backend conversion policy, but this task neither chooses a
kernel nor materializes a converted scalar. Shape arithmetic remains local and checked.

Current Dimension supports static sizes or a symbol, not affine expressions such as `N + 2` or
`N * 3`. Therefore zero-width pad and repeat-one tile preserve a dynamic symbol, while non-identity
transformations of a dynamic dimension are rejected rather than hidden as an unsound constraint.

## Architecture constraints

- Tensor remains public mutable API state, not graph IR. New methods create expression metadata
  only through the existing derived factory seam.
- Semantic identity and immutable parameters come only from completed task 0017I.
- Input descriptor/Shape may be inspected; values, storage, residency, backend capability, and
  previous provenance must not be inspected.
- Raw arrays are caller syntax only. PadAttrs/TileAttrs hold immutable copied lists.
- Result rank equals input rank. Static result extents are checked long arithmetic; locally
  unchanged dynamic dimensions retain exact references.
- All result layouts are Optional.empty. Pad introduces new positions and tile repeats values, so
  physical geometry/materialization cannot be represented as an input alias here. Even semantic
  identity requests remain explicit for later compiler canonicalization.
- Raw constantValue remains inside PadAttrs. Result descriptor preserves input DataType; no model-
  layer value conversion or validation is performed.
- Compiler owns capture/canonicalization/backward graph; planning/backend prepare own
  materialization/lowering; runtime executes only prepared work.
- No dependency or module boundary changes are authorized.

## Package impact

No package is added or moved.

- `io.github.pho001.synaptik.model.tensor.Tensor` receives exactly two public methods.
- `io.github.pho001.synaptik.model.tensor.TensorPadTileExpressions` owns request validation, Shape
  derivation, typed operation/provenance construction, and unresolved result descriptors.
- `TensorPadTileExpressionTest` mirrors the package for focused helper/API validation.
- `TensorTest` changes only its exact public API inventory/reflection assertions.

PadKind/PadAttrs/TileKind/TileAttrs remain in `model.operation.layout` unchanged.

## Required contract

### Public Tensor methods

Add exactly:

```java
public Tensor pad(long[] before, long[] after, double constantValue) {
    return TensorPadTileExpressions.pad(this, before, after, constantValue);
}

public Tensor tile(long... repeats) {
    return TensorPadTileExpressions.tile(this, repeats);
}
```

Each method contains exactly one return and one matching helper invocation. Both are non-static,
non-synchronized, and perform no direct validation, allocation, field access, or delegation to
another public Tensor method.

### Helper shape

Create one package-private final, field-free class with one private zero-argument constructor and
exactly these five static methods:

```java
static Tensor pad(Tensor input, long[] before, long[] after, double constantValue)
static Tensor tile(Tensor input, long[] repeats)
private static Shape paddedShape(Shape inputShape, PadAttrs attrs)
private static Shape tiledShape(Shape inputShape, TileAttrs attrs)
private static Tensor create(
        Tensor input,
        TensorDescriptor inputDescriptor,
        Shape resultShape,
        Operation operation)
```

Add no fields, nested type, overload, alternate constructor, cache, mutable state, or extra method.

### Padding construction

`pad` performs this exact order:

1. null-check input, before, and after, in order, with exact messages `input`, `before`, `after`;
2. read exact input descriptor and Shape once;
3. require before length equal rank, otherwise throw IllegalArgumentException with exact message
   `padding before length <length> must equal input rank <rank>`;
4. require after length equal rank, otherwise throw with exact message
   `padding after length <length> must equal input rank <rank>`;
5. clone before then after exactly once;
6. box each private clone in order and construct exactly one PadAttrs with unchanged constantValue;
7. call paddedShape once;
8. construct exactly one Operation(PadKind.PAD, attrs);
9. call create once and return.

PadAttrs supplies indexed null-free/non-negative validation and immutable list snapshots. Primitive
arrays cannot contain null. No constantValue check occurs. Caller arrays are never retained or
mutated; post-clone caller mutation cannot affect attributes or Shape.

### Tiling construction

`tile` performs this exact order:

1. null-check input then repeats, with exact messages `input` and `repeats`;
2. read exact descriptor and Shape once;
3. require repeat length equal rank, otherwise throw IllegalArgumentException with exact message
   `tile repeats length <length> must equal input rank <rank>`;
4. clone repeats exactly once;
5. box the private clone in order and construct exactly one TileAttrs;
6. call tiledShape once;
7. construct exactly one Operation(TileKind.TILE, attrs);
8. call create once and return.

TileAttrs supplies positive-repeat validation and immutable snapshot. Empty repeats are valid only
for scalar rank zero because exact rank equality is required.

### Padded Shape

`paddedShape` visits axes in increasing order and creates one rank-sized Dimension array.

- StaticDimension size `s` produces new StaticDimension exactly
  `Math.addExact(Math.addExact(s, before), after)`.
- DynamicDimension is retained by exact reference only when before and after are both zero.
- A dynamic dimension with non-zero padding throws IllegalArgumentException with exact message
  `cannot pad dynamic axis <axis> with before=<before> and after=<after>`.

Return Shape.ofDimensions once. Scalar empty lists return canonical scalar Shape. Static zero
extents are accepted; zero/zero pad preserves logical size zero through a new StaticDimension.
Checked overflow propagates as ArithmeticException before Tensor ID allocation.

### Tiled Shape

`tiledShape` also creates one rank-sized Dimension array in axis order.

- StaticDimension size `s` produces new StaticDimension `Math.multiplyExact(s, repeat)`.
- DynamicDimension is retained by exact reference only when repeat equals one.
- A dynamic dimension with another repeat throws IllegalArgumentException with exact message
  `cannot tile dynamic axis <axis> with repeat=<repeat>`.

Return Shape.ofDimensions once. Scalar empty repeats return canonical scalar Shape. Static zero
extents remain zero for every positive repeat. Overflow propagates before ID allocation.

### Constant and data-type boundary

Every current DataType is accepted. Result DataType exactly equals input DataType and requiresGrad
is unchanged. PadAttrs retains constantValue as the exact supplied double primitive; model code
does not narrow, round, saturate, normalize, or reject it.

This task documents only that later execution must interpret the constant according to the
prepared output DataType consistently across backends. Detailed finite-precision conversion and
kernel conformance belong to backend contracts/tests, not this expression constructor. No eager
converted scalar, carrier, storage, or secondary attribute is created.

### Common result construction

`create` constructs exactly:

```java
TensorDescriptor descriptor = new TensorDescriptor(
        inputDescriptor.dataType(),
        resultShape,
        Optional.empty(),
        inputDescriptor.requiresGrad());
TensorProvenance provenance = new TensorProvenance(operation, List.of(input));
return TensorFactory.createDerived(descriptor, Optional.empty(), provenance);
```

The supplied Operation is exact PAD/PadAttrs or TILE/TileAttrs. Every success consumes one fresh ID
and has absent label/storage. Validation, attribute, Shape, arithmetic, and operation failures
consume no ID. Identifier exhaustion is final. No input label, old provenance, layout, values, or
storage is retained except the exact input reference inside new provenance.

## Affected files

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorPadTileExpressions.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorPadTileExpressionTest.java`
- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum scope

At most these ten paths may change. Production is limited to Tensor plus one package-private
helper; tests to Tensor inventory plus one focused suite; documentation to public API/status and
planning evidence.

If another helper/type/test, overload, semantic/foundational edit, dependency, build/architecture
change, or eleventh path is needed, stop and report. Do not create task 0017K.

## Javadoc requirements

- Fully document both public methods and helper type/constructor/all five methods.
- Explain exact rank requests, primitive-array ownership, validation order, scalar empty requests,
  static zero extents, and dynamic identity-only rules.
- Explain checked Shape arithmetic with numeric examples for pad and complete-pattern tile.
- Document raw double retention, acceptance of all DataTypes, unchanged eligibility, and explicit
  deferral of concrete conversion/materialization.
- Explain why every result layout is unresolved and why this differs from view expressions.
- Document result Shape/type/layout/label/storage/operation/attrs/provenance/freshness and every
  null/rank/width/repeat/dynamic/arithmetic/ID failure.
- Explain no eager values, physical alias, gradient, compiler, backend, ONNX, or execution work.
- Independently review Pad/Tile semantics, DataType, Shape/Dimension, TensorDescriptor,
  TensorFactory, TensorProvenance, and adjacent view-expression Javadocs; record unchanged reasons
  or stop on discrepancy.

## Acceptance criteria

- Tensor exposes exactly pad(long[],long[],double) and tile(long...); public method count rises 87
  to 89.
- Each public method delegates once; helper is exact final/package-private/field-free with private
  constructor and five methods.
- Null/rank/order, defensive clone, PadAttrs/TileAttrs validation, messages, and ownership match.
- Scalar, static, zero-extent, and locally unchanged dynamic Shapes behave exactly as specified;
  non-identity dynamic transforms and checked overflow fail before ID allocation.
- All six DataTypes and valid eligibility states are retained exactly; every raw double is accepted
  without conversion.
- Every result layout is unresolved, including identity and fully static requests.
- Result records exact typed operation/attributes/[input], absent label/storage, fresh identity.
- Repeated, nested, zero-width, repeat-one, and identity requests stay explicit.
- No values/storage/gradient/compiler/planning/runtime/backend behavior or architecture change.
- Independent documentation review and all status/evidence synchronization complete before
  marking Complete.

## Tests / validation

Run before and after documentation review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorPadTileExpressionTest --tests io.github.pho001.synaptik.model.tensor.TensorTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

Focused tests cover exact public/helper API; all data types/eligibility; raw constant categories;
null/rank/negative/repeat validation and precedence; caller-array ownership; scalar/static/zero/
dynamic Shape rules; exact unaffected dynamic identity; checked overflow; unresolved layout for
every input layout state; exact operation/attrs/provenance; absent label/storage; dead-storage
non-interference; freshness/nesting/identity; ID side effects; and exhaustion.

Inspect javap, reflection, imports, bytecode, and source. Confirm two one-call public delegates,
exact five-method helper, one clone per array, one attribute/Shape/Operation path per family, one
common createDerived, and no value/storage/cross-layer behavior. Validate generated Javadoc,
executable examples, Tensor/Compile API/glossary, links/anchors/fences/whitespace, exact ten paths,
synchronized statuses, and no task-0017K specification.

## Dependencies

- 0001/0002 supply DataType and static/dynamic Shape contracts.
- 0003/0007 supply unresolved-layout TensorDescriptor construction.
- 0011–0013 supply Tensor, derived identity, and provenance.
- 0017I supplies PAD/TILE and immutable attributes.
- Neighboring expressions provide patterns but no extra production dependency.

## Follow-up tasks

- 0017K remains Draft for concat/stack/unstack semantic identities and parameters.
- Compiler later owns pad/tile capture, identity canonicalization, and backward graph construction.
- Planning/backend prepare own allocation, materialization, lowering, constant conversion, and
  kernels; backend conformance must enforce common observable semantics.
- ONNX extension later owns Pad/Tile mappings.

Do not create a detailed follow-up specification here.

## Architecture impact

Expected impact: None. Model already owns Tensor expression construction, Shape metadata,
operation semantics, descriptors, and provenance. Lifecycle ownership remains unchanged.

Stop if implementation needs physical output storage, converted constants, gradients,
compiler/planning/backend behavior, another dependency, or architecture changes.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0001/0002/0003/0007/0011/0012/0013/0017I/0017J, Tensor API,
Compile API, Training API, glossary, current DataType/Dimension/Shape/TensorDescriptor/Tensor/
TensorFactory/TensorProvenance/Operation/PadKind/PadAttrs/TileKind/TileAttrs contracts and focused
expression tests, and Java 26 Gradle configuration.

Implement task 0017J exactly. Modify Tensor.java and add package-private final
TensorPadTileExpressions.java. Update TensorTest only for exact two-method API expansion and add
TensorPadTileExpressionTest. Add exactly pad(long[],long[],double) and tile(long...).

The field-free helper has exactly five specified methods. Both paths null/rank validate and clone
complete arrays, construct exact immutable attrs, derive checked static Shapes, preserve only
identity-transformed dynamic Dimension references, retain exact type/eligibility, and create exact
PAD or TILE one-input provenance through one common createDerived call. Every result layout is
unresolved and every result is fresh. Preserve raw double without conversion and accept all data
types.

Do not modify semantics/foundations, add overloads/types/helpers, convert constants, support other
padding/repeat modes, inspect/copy values/storage, derive physical layout, define gradients,
capture/canonicalize graphs, or add compiler/planning/prepare/runtime/backend/ONNX behavior,
dependencies, build/architecture changes, or later specs. Stop beyond ten paths or on architecture
uncertainty.

Run all specified validation, then hand actual diff/evidence to a separate clean-context docs agent
in the same change. It must inspect source/tests/generated Javadoc, finalize permitted Javadocs/
Tensor API/Compile API/glossary/planning, record no-change conclusions, and rerun validation.

Update 0017J, model master plan, and roadmap only for status/evidence. Do not mark Complete until
both passes succeed. Leave 0017K Draft without a specification. Do not commit/push.
```

## Local decisions

- Public width/repeat arrays use long geometry; tile remains varargs for the one-value-per-axis
  ergonomic form.
- Every current DataType is accepted and raw double is retained without model-layer conversion.
- Dynamic dimensions support only transformations whose extent is provably unchanged: zero/zero
  pad or repeat one.
- All results keep layout unresolved because pad/tile require output materialization and cannot be
  represented as ordinary input view geometry.
- Scalar empty requests, static zero extents, zero-width pad, and repeat-one tile are valid fresh
  explicit operations; compiler owns identity elimination.

## Known limitations

- No non-constant padding, negative crop, zero repeats, dynamic affine Shape, converted constant,
  resolved output layout, values, storage, gradients, compiler capture, materialization, lowering,
  execution, or ONNX mapping exists.

## Validation evidence

- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorPadTileExpressionTest --tests
  io.github.pho001.synaptik.model.tensor.TensorTest --rerun-tasks` passed: 11 focused pad/tile tests
  plus 14 exact Tensor API tests, with zero failures, errors, or skips.
- `./gradlew :modules:model:test --rerun-tasks` passed: 70 suites and 577 tests, with zero failures,
  errors, or skips. `./gradlew test --rerun-tasks` also passed with 36 executed actionable tasks.
- `./gradlew :modules:model:javadoc --rerun-tasks` passed. Generated `Tensor` Javadoc contains the
  complete `pad(long[], long[], double)` and `tile(long...)` contracts, parameters, results, and
  expected failures.
- `javap -p -s -c`, reflection assertions, import/source inspection, and focused tests confirm 89
  public Tensor methods; two one-call public delegates; one final package-private, field-free
  helper with a private constructor and exactly five methods; one clone per request array; one
  attributes/Shape/Operation path per family; and one shared `TensorFactory.createDerived` call.
- All six data types and valid eligibility states, raw constant categories, scalar/static/zero/
  dynamic Shapes, overflow and validation precedence, array ownership, every input layout state,
  provenance, freshness, nesting, absent label/storage, ID side effects, and exhaustion passed the
  focused test matrix.
- The documented `PadTileExpressionExample` compiled against current model classes and printed
  Shapes `[5, 7]` and `[4, 12]`, immutable normalized attributes, `PAD`/`TILE`, exact one-input
  provenance, unresolved layouts, and absent storage exactly as documented.
- Tensor API, Compile API, and glossary local-link targets and new anchors resolve; code fences are
  balanced; changed Markdown has no trailing whitespace and ends with newlines; `git diff --check`
  passes.
- The changed-path inventory is exactly the authorized ten paths: two production sources, two
  tests, Tensor API, Compile API, glossary, this task, model master plan, and roadmap. Task 0017K
  remains Draft at row 64 and no task-0017K specification exists.
- Training API remains accurate unchanged because no gradient or training contract was added.
  Capabilities remain accurate because they already select constant pad and complete-pattern tile.
  Pad/Tile semantics and DataType, Dimension, Shape, TensorDescriptor, TensorFactory,
  TensorProvenance, Operation, and adjacent expression contracts remain accurate because this task
  composes their existing APIs without changing them.
- Architecture, ADRs, architecture tests, backend conformance, integration tests, Gradle/Java 26,
  dependencies, and other modules remain accurate unchanged because this is model-only expression
  metadata without a boundary, backend, execution, or end-to-end behavior change.

## Implementation notes

- `Tensor.pad` and `Tensor.tile` delegate to one focused package-private helper. The helper validates
  references and exact rank before cloning complete arrays, then relies on `PadAttrs` or `TileAttrs`
  for ordered element validation and immutable snapshots.
- Static pad and tile extents use `Math.addExact` and `Math.multiplyExact`. Dynamic Dimensions are
  retained by exact reference only for zero/zero padding or repeat one; unrepresentable symbolic
  transforms fail before identity allocation.
- The common construction path preserves exact data type and gradient eligibility, deliberately
  supplies unresolved layout and absent label, records exact typed operation plus provenance
  `[input]`, and delegates final identity creation to `TensorFactory.createDerived`.
- The independent documentation pass finalized public and helper Javadocs, Tensor API, Compile API,
  glossary, task evidence, model master plan, and roadmap. It found no source/test inconsistency
  requiring an implementation-owned Java edit.

## Completion summary

- Completed changes: added public constant-pad and complete-pattern tile Tensor expression
  construction with checked Shape validation, exact metadata retention, unresolved result layout,
  and fresh one-input provenance; added focused and exact API tests; finalized all required
  documentation and planning evidence.
- Files changed or created: exactly the ten authorized Tensor/helper/test/API/glossary/task/master-
  plan/roadmap paths.
- Validation: focused 25 tests, full model 577 tests, model Javadoc, root test, bytecode/reflection/
  source/import inspection, executable documentation example, links/anchors/fences/whitespace,
  exact-path/status/no-follow-up-spec checks, and `git diff --check` all passed.
- Unresolved issues: None for task 0017J.
- Required follow-up: None. Task 0017K remains Draft without a detailed specification.

Status: Complete
