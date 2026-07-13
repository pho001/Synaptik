# Task 0023C: Slice Update and Target-Relative Crop

## Status

Complete

## Goal

Add one cohesive backend-independent slice-family capability with two public transformations:

```java
Tensor updated = base.sliceUpdate(update, starts, axes, steps);
Tensor cropped = input.cropToShape(targetShape, prefixShape);
```

`sliceUpdate` functionally replaces the coordinates of a normalized signed, strided, multi-axis
slice in `base` with `update`, while retaining every base value outside that region. `cropToShape`
extracts an exact target Shape after skipping a non-negative per-axis prefix Shape. Prefix and
target Dimensions may remain symbolic, so padding and concatenation regions can be cropped without
binding their extents in the model.

Together these operations close the exact slice/select/pad/concat gaps proven by the
[adjoint expressibility audit](../adjoint-expressibility-audit.md). They remain generally useful
public tensor transformations rather than `SLICE_BACKWARD`, `SELECT_BACKWARD`, `PAD_BACKWARD`, or
`CONCAT_BACKWARD` kinds.

## Scope

- Append exactly `SLICE_UPDATE` after existing `SliceKind.SLICE`.
- Preserve existing `SLICE` plus `SliceAttrs` behavior and add these exact family-owned signature
  variants:

  ```text
  SLICE         SliceAttrs          1 input, 1 output
  SLICE         CropToShapeAttrs    1 input, 1 output
  SLICE_UPDATE  SliceAttrs          2 inputs, 1 output
  ```

- Add exactly one public record `CropToShapeAttrs(Shape targetShape, Shape prefixShape)` under the
  existing layout-operation package.
- Update `SliceAttrs` Javadoc only for its new exact `SLICE_UPDATE` pairing; preserve its record
  declaration, components, validation, snapshots, equality, hashing, and bytecode behavior.
- Add one package-private field-free `TensorSlicePlacementExpressions` helper.
- Add exactly two public Tensor methods:

  ```java
  public Tensor sliceUpdate(
          Tensor update, long[] starts, int[] axes, long[] steps)
  public Tensor cropToShape(Shape targetShape, Shape prefixShape)
  ```

- Normalize slice-update axes and starts locally, derive selected lengths from static update
  Dimensions, require exact same-rank Shape compatibility, and permit deferred upper-bound proof
  only for unresolved base Dimensions.
- Treat `prefixShape` Dimensions as non-negative per-axis prefix extents. Validate every fully
  static crop bound locally and retain unresolved `prefix + target <= input` obligations for later
  binding.
- Preserve all current data types, descriptor invariants, producer/provenance, identifier behavior,
  existing signed Slice methods, Pad, Concat, Select, and architecture boundaries.
- Update every exact public Tensor method-count inventory from 190 to 192.
- Finalize Javadocs, Tensor/Compile APIs, glossary, capabilities, and planning records through the
  mandatory separate clean-context documentation pass.

## Out of scope

- operation-specific backward kinds, gradient rules, adjoint construction, graph traversal,
  compiler adoption, saved-value policy, cotangent accumulation, or autograd execution
- changing existing `SLICE`, `SliceAttrs`, `TensorSliceExpressions`, `slice`, either `sliceAxis`,
  `flip`, `SELECT`, `PAD`, `CONCAT`, Scatter families, Shape, Dimension, DimensionExpressions,
  descriptors, factory, producer, or provenance contracts
- an additive slice update, overlap reduction, duplicate-coordinate policy, mask update, tensor
  indices, coordinate tensors, arbitrary index tuples, or a general scatter dimension language
- clamping, wrapping, silently shifting, truncating, or ignoring an invalid update or crop region
- dynamic selected update lengths in `sliceUpdate`; every selected update Dimension must be
  statically known so existing normalized `SliceAttrs` remains exact
- negative unresolved update starts; a negative start requires a static base extent for local
  normalization
- raw exclusive ends for `sliceUpdate`; selected lengths come exactly from the update Shape
- signed crop prefixes, crop steps, reversed crop, broadcasted crop, inferred target Shape,
  inferred prefix, or a second crop overload
- resolved result layout, storage aliasing, storage allocation/copying, mutation, value access,
  eager execution, materialization, or view guarantees
- dependencies, Gradle, architecture documents/tests, backend conformance, integration tests,
  another module, unrelated refactors, or later tasks

## Exact semantic and public contract

### Slice-family identities and signatures

`SliceKind` contains exactly these constants in order:

```java
SLICE,
SLICE_UPDATE
```

`SLICE` remains extraction. Existing normalized `SliceAttrs` continues to select one finite
coordinate sequence from one input. New `CropToShapeAttrs` is another exact `SLICE` representation
whose lengths and prefix extents may be symbolic.

`SLICE_UPDATE` is functional replacement. Its ordered inputs are `[base, update]`; its result has
the exact base Shape. Existing `SliceAttrs` identifies the unique target coordinates occupied by
`update`. The kind performs no addition and accepts no reduction attribute.

The exact signatures are:

```text
SLICE         SliceAttrs          fixed 1 input, fixed 1 output
SLICE         CropToShapeAttrs    fixed 1 input, fixed 1 output
SLICE_UPDATE  SliceAttrs          fixed 2 inputs, fixed 1 output
```

`OperationSignatureTest` locks both `SLICE` variants and the `SLICE_UPDATE` variant separately.
Cross-pairings remain invalid: `SLICE_UPDATE` does not accept `CropToShapeAttrs`, and no other kind
accepts either slice attribute type.

### CropToShapeAttrs

Add exactly:

```java
public record CropToShapeAttrs(
        Shape targetShape,
        Shape prefixShape) implements OperationAttrs
```

Construction null-checks `targetShape` and then `prefixShape` with exact messages
`targetShape` and `prefixShape`. It retains both exact immutable references through ordinary record
assignment. It performs no rank, input, bounds, data-type, layout, binding, or execution check.

`targetShape` is the exact result Shape. `prefixShape` has the same eventual rank and describes how
many logical positions precede the crop region on each axis. It is a Shape of prefix extents, not a
Tensor, an index vector, a bound value, storage geometry, or runtime binding. For input Shape
`[N + 3]`, target Shape `[N]`, and prefix Shape `[1]`, the crop begins after one position and has
symbolic length `N`.

The record exposes only explicit documented component accessors plus record-generated
`equals`/`hashCode`/`toString`. It contains no helper, factory, normalization, sentinel, optional,
or extra state.

### Public methods and helper surface

`Tensor` adds exactly:

```java
public Tensor sliceUpdate(
        Tensor update, long[] starts, int[] axes, long[] steps) {
    return TensorSlicePlacementExpressions.update(
            this, update, starts, axes, steps);
}

public Tensor cropToShape(Shape targetShape, Shape prefixShape) {
    return TensorSlicePlacementExpressions.cropToShape(
            this, targetShape, prefixShape);
}
```

No overload, default axis, end array, scalar-index shortcut, additive mode, alias, in-place form,
or raw `Dimension[]` surface is added.

`TensorSlicePlacementExpressions` is one final package-private non-record class with no fields,
no nested types, and one private zero-argument constructor. Its exact method names are:

```text
update
cropToShape
normalizeUpdate
normalizeUpdateStart
expectedUpdateShape
validateStaticCropBounds
createUpdate
createCrop
```

The two package-private entries have the exact public-delegation parameter order. The six private
methods own only local normalization, Shape validation, and final construction. The helper does
not call or modify `TensorSliceExpressions`, Pad, Concat, Select, Scatter, compiler, runtime, or
backend code.

### Functional signed slice update

For base rank `R`, update rank must also be `R`. Arrays `starts`, `axes`, and `steps` are parallel,
equal-length arrays. Each normalized axis is unique. For entry `i`:

```text
target coordinate(k) = normalizedStart[i] + k * steps[i]
0 <= k < updateShape[axis]
```

The selected update Dimension must be static; its non-negative size becomes the corresponding
`SliceAttrs.lengths` entry. Unselected update Dimensions must be structurally equal to the exact
base Dimensions. The expected update Shape is therefore the base Shape with every selected axis
replaced by that update's exact static Dimension.

Examples:

```text
base Shape    [3, 6]
update Shape  [3, 2]
starts        [1]
axes          [1]
steps         [2]
target axes   coordinates 1 and 3
result Shape  exact [3, 6]
```

```text
base Shape    [5]
update Shape  [3]
starts        [4]
axes          [0]
steps         [-2]
target axes   coordinates 4, 2, 0
```

For a non-empty selected length, a negative raw start adds one static base extent exactly once.
Static target axes require the normalized first and last coordinates to lie in `[0, extent)`.
Dynamic target axes accept only a non-negative raw start; the normalized sequence must remain
non-negative, while its upper bound is a deferred binding/execution obligation. There is no clamp,
wrap, suffix adjustment, or partial update.

A selected zero length uses canonical start zero and maps no coordinates. Empty update values
leave the base mathematically unchanged but still create one fresh explicit occurrence. Empty
arrays mean unrestricted full-shape replacement: update Shape must equal base Shape, every result
value comes from update, and one fresh `SLICE_UPDATE` occurrence is still recorded.

Every mapped update coordinate replaces exactly one base coordinate. Unique axes plus non-zero
steps make the multi-axis Cartesian mapping injective, so no duplicate reduction or ordering
policy exists. Values outside the mapped region retain base values. All six current data types are
accepted with exact base/update type equality; there is no promotion or scalar conversion.

### Target-relative crop

`cropToShape(targetShape, prefixShape)` requires input, target, and prefix ranks to be equal. On
each axis the logical region is:

```text
[prefixExtent, prefixExtent + targetExtent)
```

The result Shape is the exact supplied `targetShape` reference. The prefix and target Shapes retain
their exact component references in `CropToShapeAttrs`.

When input, prefix, and target Dimensions on an axis are all static, checked arithmetic requires:

```text
prefixExtent + targetExtent <= inputExtent
```

If any of those Dimensions is unresolved, model construction retains the same inequality as a
later binding obligation. It does not prove, bind, clamp, or evaluate the expression. This permits:

```text
PAD adjoint-like crop
input Shape   [N + 3]
prefix Shape  [1]
target Shape  [N]

CONCAT role crop
input Shape   [batch, P + Q + R]
prefix Shape  [0, P]
target Shape  [batch, Q]
```

Scalar input uses scalar target and prefix Shapes. Static zero target extents are valid. A static
out-of-bounds region fails locally; a deferred region that later binds out of bounds is invalid at
the binding/execution boundary rather than shifted, truncated, or padded.

### Validation order and diagnostics

`sliceUpdate` validates in this exact order:

1. null-check `base`, `update`, `starts`, `axes`, and `steps`;
2. require equal array lengths;
3. clone all three arrays in declaration order;
4. read base and update descriptors in that order;
5. require exact data-type equality;
6. require equal ranks;
7. process entries in caller order: normalize axis, reject a duplicate, reject zero step, require a
   static selected update Dimension, normalize/canonicalize start, and validate the coordinate
   sequence;
8. construct exact `SliceAttrs` once;
9. require complete expected update Shape equality; and
10. construct descriptor, operation, producer, provenance, and Tensor.

Exact task-owned failures are:

```text
starts, axes, and steps must have matching lengths
slice update data types must match: base=<dataType>, update=<dataType>
slice update rank must match base rank: base=<rank>, update=<rank>
slice update axis <rawAxis> at index <index> is outside rank <rank>
slice update contains duplicate normalized axis <axis> at index <index>
steps[<index>] must be non-zero: 0
slice update axis <axis> at index <index> must have a statically known update dimension
slice update start <start> at index <index> cannot be negative for dynamic base axis <axis>
slice update coordinates at index <index> do not fit base extent <extent>: start=<start>, length=<length>, step=<step>
slice update shape must match base Shape with selected axes replaced: expected=<shape>, actual=<shape>
```

Checked start/last-coordinate arithmetic may throw `ArithmeticException` before allocation.

`cropToShape` validates in this exact order:

1. null-check `input`, `targetShape`, and `prefixShape`;
2. read the input descriptor and Shape;
3. require target rank equal input rank;
4. require prefix rank equal input rank;
5. inspect axes in ascending order and validate every fully static bound with checked addition;
6. construct exact `CropToShapeAttrs`; and
7. construct descriptor, operation, producer, provenance, and Tensor.

Exact task-owned failures are:

```text
crop target rank must match input rank: input=<rank>, target=<rank>
crop prefix rank must match input rank: input=<rank>, prefix=<rank>
crop region exceeds input extent at axis <axis>: input=<extent>, prefix=<extent>, target=<extent>
```

All local failures occur before `TensorFactory.createDerived` and consume no Tensor identifier.

### Descriptor, producer, provenance, and identity

`sliceUpdate` creates one unresolved descriptor with exact base Shape and data type. Its
`requiresGrad` is base/update eligibility OR, subject to existing data-type invariants. It creates
one `Operation(SliceKind.SLICE_UPDATE, attrs)`, exact ordered inputs `[base, update]`, one output
descriptor, output-index-zero provenance, and one fresh unlabeled, storage-free Tensor.

`cropToShape` creates one unresolved descriptor with exact target Shape, exact input data type, and
unchanged input eligibility. It creates one `Operation(SliceKind.SLICE, attrs)`, exact ordered input
`[input]`, one output descriptor, output-index-zero provenance, and one fresh unlabeled,
storage-free Tensor.

Neither path inspects labels, provenance, layouts, storage, liveness, or values. Neither mutates an
input. Every success consumes exactly one Tensor identifier; exhaustion remains the final failure.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Training graph](../../../../architecture/training-graph.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Capabilities](../capabilities.md)
- [Master plan](../master-plan.md)
- [Adjoint expressibility audit](../adjoint-expressibility-audit.md)
- [Task 0017G](0017g-slice-semantics.md)
- [Task 0017H](0017h-slice-tensor-expressions.md)
- [Task 0017I](0017i-pad-and-tile-semantics.md)
- [Task 0017J](0017j-pad-and-tile-tensor-expressions.md)
- [Task 0017K](0017k-tensor-composition-semantics.md)
- [Task 0017L](0017l-tensor-composition-expressions.md)
- [Task 0018M](0018m-symbolic-extent-expressions.md)
- [Task 0018M1](0018m1-dynamic-extent-adoption.md)
- [Task 0018R](0018r-slice-and-window-public-contract-cleanup.md)
- [Task 0023](0023-adjoint-expressibility-audit.md)
- [Task 0023A](0023a-binding-aware-sum-to-shape.md)
- [Task 0023B](0023b-gather-compatible-scatter-add.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- Work stays inside model-owned backend-independent layout-operation semantics, Tensor metadata,
  pre-capture producer/provenance construction, and directly affected documentation/planning.
- Shape and prefix expressions are immutable model values. This task does not bind them, add
  constraints to Shape, or decide where binding occurs.
- `SLICE_UPDATE` means functional replacement, never mutation, addition, storage aliasing, or a
  physical write. `SLICE` crop means logical extraction, never a promised view or copy.
- Model construction validates only locally provable type/rank/Shape/static-bound facts. Compiler
  or later binding/execution owns unresolved inequalities and concrete bounds.
- Compiler later may compose these public primitives when constructing adjoints. This task adds no
  compiler rule, graph traversal, capture, accumulation, or execution behavior.
- Runtime hot paths must not consume `Operation` or `CompiledNode`; backend prepare owns lowering,
  materialization, kernel selection, and concrete memory access.
- Stop if the contract requires a Shape/Dimension change, coordinate Tensor, another public
  abstraction, compiler behavior, architecture update, dependency, Gradle change, or work outside
  the exact model/documentation scope.

## Package impact

Existing packages changed:

- `io.github.pho001.synaptik.model.operation.layout`
- `io.github.pho001.synaptik.model.tensor`

No package is added, moved, or renamed.

Type placement:

- `...operation.layout.CropToShapeAttrs` owns the minimal immutable target/prefix Shape parameters
  for the target-relative `SLICE` variant.
- `...operation.layout.SliceKind` owns extraction and functional slice-update identities.
- `...operation.layout.SliceAttrs` remains the normalized signed finite coordinate-sequence value
  shared by extraction and update.
- `...tensor.TensorSlicePlacementExpressions` owns local update/crop normalization, validation,
  descriptor construction, and provenance.
- `...tensor.Tensor` remains the public fluent facade.

## Affected files

Production (5):

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/SliceKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/SliceAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/CropToShapeAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorSlicePlacementExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Tests (15):

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/OperationSignatureTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/layout/SliceSemanticsTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorSlicePlacementExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBatchNormInferenceExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryArithmeticTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorDenseCategoricalCrossEntropyWithLogitsExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorIndexCategoricalCrossEntropyWithLogitsExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorLayerNormExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorLinearExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMatmulExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMeanSquaredErrorExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorRmsNormExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorScaledDotProductAttentionExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorSumToShapeExpressionTest.java`

Documentation/planning (7):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review unchanged unless inaccurate: current Slice extraction helper/tests, Pad, Concat, Select,
Shape/Dimension expressions, descriptor/factory/producer/provenance, Scatter families,
Training/Runtime APIs, architecture/ADRs/tests, dependencies, Gradle, conformance/integration,
other modules, and later tasks.

## Maximum scope

Exactly 27 paths: five production, fifteen tests, and seven documentation/planning paths. This
cohesive capability exceeds the usual 18-path guardrail under the user's standing automatic
higher-path authorization because two inseparable slice-family variants, twelve existing global
Tensor-count locks, the global signature matrix, focused semantic/expression coverage, and
mandatory documentation must change atomically. Stop for path 28, another public method/type,
another helper/test/document, an existing behavior change, Shape/Dimension work, later-task work,
cross-module work, architecture/Gradle change, or unrelated cleanup. If live repository evidence
changes an inventory path, update this Ready task before implementation without exceeding 27.

## Javadoc and documentation requirements

- Finalize complete Javadocs for both `SliceKind` constants and all signature variants,
  `SliceAttrs` pairing, `CropToShapeAttrs`, every helper method, and both public Tensor methods.
- Explain functional replacement versus extraction, normalized coordinate mapping, static versus
  deferred bounds, prefix Shape meaning, empty/full replacement, exact metadata/provenance, ID
  effects, failures, lifecycle ownership, and no view/mutation promise.
- Apply General, API/Javadoc, Planning, and Example profiles as relevant, with complete `@param`,
  `@return`, and expected `@throws` text.
- Tensor API moves slice update and target-relative crop from planned to current, includes
  static/signed/dynamic examples, distinguishes current raw-bound Slice, slice update, crop, Pad,
  Select, and index Scatter, and records metadata/lifecycle boundaries.
- Compile API records the current model primitives plus later binding/bounds/adjoint obligations
  without claiming compiler capture, gradient construction, lowering, or execution exists.
- Glossary adds `Slice Update` and `Target-relative crop`, updates Slice and prefix terminology,
  and cross-links Shape expressions.
- Capabilities/task/master/roadmap synchronize 0023C only after both implementation and
  documentation passes. Then 0023D becomes the next Draft frontier without a detailed
  specification; 0023E–0024 remain Draft.
- Use official [PyTorch `slice_scatter`](https://docs.pytorch.org/docs/stable/generated/torch.slice_scatter.html),
  [JAX `dynamic_update_slice`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.dynamic_update_slice.html),
  and [ONNX Slice](https://onnx.ai/onnx/operators/onnx__Slice.html) only as terminology/comparison
  evidence. Explain that Synaptik's update is functional, multi-axis, and metadata-only; JAX start
  values and ONNX dynamic bound inputs are execution values rather than this task's prefix-Shape
  metadata.
- Record reasoned no-change conclusions for existing Slice extraction Javadocs, Pad/Concat/Select,
  Shape/Dimension, Training/Runtime APIs, architecture/ADRs/tests, dependencies, Gradle,
  conformance/integration, other modules, and later tasks.

## Acceptance criteria

- `SliceKind` has exactly `SLICE`, then `SLICE_UPDATE`, with exact signature variants and no
  backward, crop, flip, additive, or alias kind.
- `CropToShapeAttrs` is exactly the two-component public record with ordered null checks, exact
  reference retention, explicit accessors, value semantics, and no extra state/API.
- `SliceAttrs` declaration and behavior remain bytecode-equivalent apart from Javadoc and its new
  exact pairing.
- Exactly two public methods are added; public Tensor count is 192; no overload or alias exists.
- New helper remains field-free with the exact eight-method surface and private constructor; no
  existing helper is modified.
- Slice update clones arrays, derives lengths from static selected update Dimensions, normalizes
  signed starts/axes once, rejects duplicates/zero steps/out-of-bounds static sequences, defers
  only unresolved upper bounds, and validates the complete same-rank update Shape.
- Slice update is exact functional replacement for all current types, retains base values outside
  the injective region, uses no reduction, and does no value/storage work.
- Crop requires equal input/target/prefix ranks, validates every fully static prefix-plus-target
  bound with checked arithmetic, retains unresolved inequalities, and returns the exact target
  Shape without binding or clamping.
- Exact null/type/rank/entry/Shape/bound validation order, messages, exception types, and no-ID
  local failures are tested.
- Update result retains exact base Shape/type, base/update eligibility OR, unresolved layout,
  exact `SLICE_UPDATE`/`SliceAttrs`, ordered `[base, update]` producer, output index zero, no
  label/storage, one fresh ID, and unchanged inputs.
- Crop result retains exact target Shape/input type/eligibility, unresolved layout, exact
  `SLICE`/`CropToShapeAttrs`, ordered `[input]` producer, output index zero, no label/storage, one
  fresh ID, and unchanged input.
- Existing Slice, SliceAttrs behavior, Pad, Concat, Select, Scatter, count locks, and signature
  matrix remain covered without unrelated changes.
- Exact 27-path/package scope; no Shape/Dimension, compiler, execution, architecture, dependency,
  build, or later-task work.
- Separate clean documentation-focused pass and all required validation/evidence complete before
  status Complete.

## Tests / validation

Focused implementation command:

```bash
./gradlew :modules:model:test \
  --tests io.github.pho001.synaptik.model.operation.OperationSignatureTest \
  --tests io.github.pho001.synaptik.model.operation.layout.SliceSemanticsTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorSlicePlacementExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorBatchNormInferenceExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorBinaryArithmeticTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorDenseCategoricalCrossEntropyWithLogitsExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorIndexCategoricalCrossEntropyWithLogitsExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorLayerNormExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorLinearExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorMatmulExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorMeanSquaredErrorExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorRmsNormExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorScaledDotProductAttentionExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorSumToShapeExpressionTest
```

After executable Java stabilizes, run exactly one final model suite:

```bash
./gradlew :modules:model:test
```

Focused coverage must verify exact enum/record/helper/public surfaces, signature pairings and
cross-attrs rejection, count 192, all data types and valid gradient choices, positive/negative/
multi-axis/identity/empty/static/dynamic update regions, exact update Shape, prefix/target static
and symbolic crops, scalar/zero extents, caller-array ownership, exact validation order and
messages, overflow, no-ID failures, descriptor/operation/producer/provenance/storage/label
metadata, freshness, and unchanged inputs. Tests inspect metadata only and claim no value execution.

Documentation pass after final Javadocs:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

Also validate the runnable Java 26 example, Markdown links/anchors/fences/final newlines/trailing
whitespace, official references, exact 27 paths, package placement, signatures/count 192,
synchronized status, 0023D Draft/no spec, absence of backward/alias/additive kinds, and absence of
Shape/compiler/runtime/backend/build changes. Reuse successful Java evidence unless executable
Java changes.

Repository-wide validation is deferred to the capability checkpoint after task 0023F and to CI;
this single-module task changes no dependency, build, or architecture boundary.

## Dependencies

- 0002 and 0018M–0018M1: immutable static/named/expression Dimensions, exact Shapes, and canonical
  prefix/target formulas.
- 0005–0007 and 0011–0013: typed operations/signatures, descriptors, Tensor, identity, factory,
  producer, and provenance.
- 0017G–0017H and 0018R: normalized signed finite Slice semantics and public extraction/flip.
- 0017I–0017J: symbolic constant-padding Shape formulas.
- 0017K–0017L: variadic Concat with symbolic summed extent.
- 0018A–0018B: Select's static and deferred dynamic-axis behavior.
- 0023: completed audit proving the general placement/crop gap and rejecting backward-only kinds.
- 0023A–0023B: completed preceding audit prerequisites and current 190-method baseline.

All dependencies are Complete.

## Follow-up tasks

- 0023D becomes the next Draft frontier for public `foldAxis` plus redesigned dynamic/configurable
  2D windows; do not create its detailed specification during this task.
- 0023E–0023F and 0024 remain concise Draft rows without detailed specifications.
- Compiler/autograd work later may compose zero-expanded bases with `sliceUpdate` and use
  `cropToShape` for Pad/Concat roles. It owns traversal, binding proof, accumulation,
  canonicalization, saved values, and optimization.

## Architecture impact

Expected impact: None. Stop if implementation needs Shape/Dimension changes, coordinate Tensors,
another semantic/public/helper type, compiler/prepare/backend contracts, dependencies, Gradle,
architecture updates, or work outside the exact model/documentation scope.

## Implementation prompt

Use this prompt in a separate clean-context task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, documentation/planning rules, model capabilities/master plan,
roadmap, completed tasks 0017G–0017J/0017K–0017L/0018A–0018B/0018M–0018M1/0018R/0023–0023B,
task 0023C, and every affected/review-only source/test named there in full.

Implement task 0023C exactly inside its 27 authorized paths. Add only functional signed multi-axis
`SLICE_UPDATE`, target-relative symbolic `SLICE` crop, and exactly two public Tensor methods.
Preserve existing Slice/Pad/Concat/Select/Shape/producer/provenance contracts and every architecture
boundary. Add no backward kind, additive mode, coordinate Tensor, Shape change, compiler adoption,
value execution, or later task. Stop on architecture, dependency, completed-contract,
validation-order, affected-file, or maximum-scope conflict.

Run focused tests while developing and exactly one final model suite after executable Java
stabilizes. Then hand the actual diff and exact Java evidence to a separate clean-context
documentation-focused agent in the same overall change. That agent must independently inspect
final source/tests, finalize Javadocs, Tensor/Compile APIs, glossary, capability/task/master/
roadmap status and documentation validation, and reuse successful Java evidence unless executable
behavior changes or it records a concrete reason.

Do not mark 0023C Complete until both passes and every acceptance criterion succeed. Leave 0023D
and every later task Draft without a detailed specification.
```

## Documentation-agent handoff

The implementation agent must hand over this task, actual source/test diff, exact Java evidence,
kind/signature/attribute changes, update and crop Shape formulas, static/deferred bound rules,
validation and ID order, metadata/provenance behavior, architecture constraints, expected
Tensor/Compile API and glossary changes, official comparison links, existing-Javadoc review list,
and every documentation/scope/status command. The documentation agent must inspect final source
and tests rather than rely on the summary.

## Local decisions

- Reused the existing normalized signed `SliceAttrs` value for `SLICE_UPDATE`; functional update
  therefore shares extraction's exact finite coordinate-sequence representation rather than
  adding another bounds language. `CropToShapeAttrs` remains the exact target/prefix `Shape` pair
  because those extents may be unresolved.
- Selected update-axis lengths come from static update Dimensions. A negative raw start requires a
  static base extent; only the unresolved base upper bound is deferred. Crop construction likewise
  defers only an inequality containing an unresolved input, target, or prefix Dimension.
- Update is functional replacement, never addition or mutation. Both transformations retain exact
  ordered producer inputs and output-index-zero provenance, and deliberately leave result layout
  unresolved.
- The documentation pass corrected `Tensor`'s stale class overview, which had described all Slice
  expressions as positive-step-only even though signed extraction already existed before this
  task.

## Known limitations

- These expressions describe metadata only. They do not read or write values, allocate storage,
  promise a view or copy, mutate an input, or resolve layout.
- Deferred update upper bounds and crop inequalities are retained as later binding/execution
  obligations; the current `Shape` model does not store or solve them.
- No gradient rule, graph traversal, compiler capture, lowering, runtime, backend, or execution
  behavior was added. Tasks 0023D–0023F and 0024 remain Draft.

## Validation evidence

- The implementation context's `./gradlew :modules:model:compileJava` run passed. During focused
  development, the exact 15-suite command first ran 139 tests with six test-expectation and
  surface-lock failures, then ran 139 tests with one remaining diagnostic-expectation failure,
  and on its third run passed all 139 tests. These were focused development reruns before
  executable stabilization, not additional final suites. The one final
  `./gradlew :modules:model:test` run then passed 996 tests across 126 suites with zero failures,
  errors, or skips. The documentation context `/root/implement_0023c/docs_0023c` reused that
  evidence because it changed no executable Java.
- `./gradlew :modules:model:javadoc` completed successfully after final Javadocs.
- The Tensor API's complete Java 26 example compiled against model classes with
  `javac --release 26` and ran successfully. Its exact output confirmed result `Shape[5, 6]`,
  normalized `SliceAttrs[starts=[4, 1], lengths=[3, 2], axes=[0, 1], steps=[-2, 2]]`, and both
  static and symbolic crop assertions.
- An independent Java 26 reflection check confirmed 192 public Tensor methods; exact enum order
  `[SLICE, SLICE_UPDATE]`; the two `SLICE` signatures and one `SLICE_UPDATE` signature; exact record
  components `targetShape,prefixShape`; a field-free helper; and exact helper method names
  `createCrop`, `createUpdate`, `cropToShape`, `expectedUpdateShape`, `normalizeUpdate`,
  `normalizeUpdateStart`, `update`, and `validateStaticCropBounds`.
- Repository Markdown validation covered 201 files, 3,436 local links, and 205 local anchors, and
  passed fence balance, final-newline, and trailing-whitespace checks. The three official
  comparison references returned HTTP 200.
- The combined tracked/untracked inventory contains exactly the authorized 27 paths: five
  production, fifteen tests, and seven documentation/planning paths. Package/status scans confirm
  0023C Complete, 0023D Draft with no detailed task specification, no cross-layer/build change,
  and no extra operation kind or public alias. `git diff --check` passes.
- Review concluded that existing Slice extraction behavior, Pad/Concat/Select, Shape/Dimension,
  Training/Runtime APIs, architecture/ADRs/tests, dependencies, Gradle, conformance/integration,
  other modules, and later tasks remain accurate and require no change.

## Implementation notes

- Added `SLICE_UPDATE` and the exact operation-signature matrix while preserving `SLICE` and
  `SliceAttrs` behavior. Added the exact two-component `CropToShapeAttrs` record.
- Added the field-free package-private placement helper and exactly two Tensor methods. Update
  construction clones caller arrays, normalizes signed axes/starts, derives selected lengths from
  static update Dimensions, validates complete same-rank Shape compatibility, and creates ordered
  `[base, update]` provenance. Crop validates static bounds and creates ordered `[input]`
  provenance with the exact supplied target Shape.
- Added focused semantic and expression tests plus the existing public-method count-lock updates
  from 190 to 192. Tests cover surface shape, signatures, validation/ID order, static and deferred
  bounds, all data types, eligibility, metadata/provenance, array ownership, freshness, and
  unchanged inputs without making value-execution claims.
- Finalized Javadocs, Tensor and Compile API explanations, glossary vocabulary, capabilities,
  master plan, task status, and roadmap. The API guide includes a compiled signed multi-axis
  update plus static and symbolic crop example and distinguishes the new primitives from existing
  extraction, Select, Pad, and Scatter behavior.

## Completion summary

Completed changes: implemented backend-independent functional signed slice update and
target-relative crop metadata, exact public/semantic surfaces, validation, producer/provenance,
tests, Javadocs, API documentation, glossary entries, and synchronized planning records.

Files changed or created: exactly the five production, fifteen test, and seven
documentation/planning paths listed under Affected files.

Tests or validation performed: focused 15-suite model run (139 tests), final model suite (996 tests
across 126 suites), model Javadoc, runnable Java 26 example, Java 26 reflection/surface check,
Markdown links/anchors/structure, official-reference reachability, exact-scope/status/package
checks, and `git diff --check` all passed.

Documentation impact: finalized in the mandatory independent documentation-focused context.
Affected API and implementation Javadocs, Tensor/Compile API guides, glossary, capabilities, task,
master plan, and roadmap are current. Review found no required change to Training/Runtime APIs,
architecture documents/tests, or unrelated contracts.

Unresolved issues: None.

Required follow-up: None for task 0023C. Draft task 0023D is the next planning frontier.

Status: Complete
