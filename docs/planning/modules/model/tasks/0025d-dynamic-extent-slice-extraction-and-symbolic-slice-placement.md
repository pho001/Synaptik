# Task 0025D: Dynamic-Extent Slice Extraction and Symbolic Slice Placement

## Status

Complete

## Goal

Add exactly two generally useful public Tensor transformations that close the remaining
dynamic-extent slice-construction gap before Compiler task 0005C:

```java
Tensor selected = input.sliceByLength(starts, lengths, axes, steps);
Tensor placed = base.sliceUpdate(update, prefixShape);
```

`sliceByLength` consumes already normalized non-negative starts and finite non-negative lengths.
It can therefore describe a finite signed, strided selection even when a selected input extent is
unresolved: the model proves every lower bound and every statically decidable upper bound, then
defers only an upper-bound proof that depends on the unresolved selected extent.

The new `sliceUpdate` overload functionally places the complete `update` region after a
non-negative per-axis prefix Shape. It retains the exact update Shape as
`CropToShapeAttrs.targetShape`, retains the exact caller prefix Shape, and returns the exact base
Shape. This is the target-relative functional-update counterpart of current target-relative crop.

The relationship is:

```text
normalized finite coordinates + possibly unresolved selected input extent
  -> SLICE / SliceAttrs
  -> exact static selected lengths and deferred unresolved upper bounds

update Shape + prefix Shape + base Shape
  -> SLICE_UPDATE / CropToShapeAttrs
  -> exact target-relative functional replacement
```

This task changes only model-owned semantic signatures and Tensor expression metadata. Compiler
0005C later owns explicit inference-constraint adoption, fail-closed preflight, and gradient rules
for these new public construction forms.

## Scope

- Add exactly these two public instance methods to `Tensor`:

  ```java
  public Tensor sliceByLength(
          long[] starts, long[] lengths, int[] axes, long[] steps)

  public Tensor sliceUpdate(Tensor update, Shape prefixShape)
  ```

- Preserve the current `SliceKind` constants and append no operation identity.
- Extend only the existing `SLICE_UPDATE` signature matrix:

  ```text
  SLICE         SliceAttrs          1 input, 1 output
  SLICE         CropToShapeAttrs    1 input, 1 output
  SLICE_UPDATE  SliceAttrs          2 inputs, 1 output
  SLICE_UPDATE  CropToShapeAttrs    2 inputs, 1 output
  ```

- Keep `SliceAttrs` and `CropToShapeAttrs` declarations, components, construction validation,
  snapshots/reference ownership, equality, hashing, and generated behavior unchanged.
- Generalize `CropToShapeAttrs` Javadoc from extraction-only wording to its exact extraction and
  functional-placement pairings.
- Extend the existing field-free `TensorSliceExpressions` helper with one public-delegation entry
  and one private normalization method for length-based requests.
- Extend the existing field-free `TensorSlicePlacementExpressions` helper with one exact
  Shape-based `update` overload and one typed `createUpdate` overload.
- Clone every caller-owned primitive array before entry inspection.
- Define exact validation, checked coordinate arithmetic, zero-length canonicalization, Shape,
  layout, metadata, producer, provenance, freshness, and identifier behavior.
- Preserve existing raw-bound `slice`, both `sliceAxis` overloads, `flip`, array-based
  `sliceUpdate`, and `cropToShape` public signatures and behavior exactly.
- Update every live public Tensor method-count lock from 200 to 202.
- Finalize affected Javadocs, Tensor API, Compile API, glossary, capability baseline, and planning
  records through the mandatory independent documentation-focused pass.

## Out of scope

- another public method, overload, alias, builder, slice object, bounds object, entry type,
  coordinate Tensor, raw `Dimension[]` surface, or default/nullable parameter convention
- a new `OperationKind`, `OperationAttrs`, `Shape`, `Dimension`, expression form, package,
  helper class, result carrier, backward-only operation, or additive update mode
- changing `SliceAttrs` components or validation; adding a raw-end, sentinel, optional bound,
  compatibility accessor, or secondary constructor
- changing `CropToShapeAttrs` components, validation, exact-reference retention, or record API
- negative normalized starts in `sliceByLength`, negative lengths, infinite/open lengths,
  clamping, wrapping, truncation, bound shifting, or inferred axes/steps
- treating a zero-length request as an out-of-bounds coordinate or retaining a non-zero empty
  start in `SliceAttrs`
- resolving negative-stride layout, adding symbolic layout, attaching storage, promising an alias,
  selecting copy/view/materialization behavior, reading values, mutation, or execution
- compiler source/tests, graph capture changes, inference predicates, preflight adoption, gradient
  formulas, canonicalization, lowering, backend behavior, runtime, prepare, engine, training,
  ONNX, or trace work
- changing existing compiler behavior merely because the new extraction reuses `SliceAttrs`
- architecture, ADR, architecture-test, dependency, Gradle, Java-version, backend-conformance, or
  integration changes
- editing completed task specifications or the completed adjoint/closure audit artifacts
- creating Compiler 0005C, 0005D, 0005E, or 0006 task specifications or promoting their rows
- unrelated documentation or source cleanup

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md), especially model-owned Tensor,
  operation, Shape, layout, descriptor, and producer semantics plus compiler-owned inference and
  gradient construction
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Training graph](../../../../architecture/training-graph.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [General documentation style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)
- [Planning style](../../../../developer-guide/documentation/planning-style.md)
- [Example format](../../../../developer-guide/documentation/example-format.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Model capabilities](../capabilities.md)
- [Model master plan](../master-plan.md)
- [Compiler master plan](../../compiler/master-plan.md)
- [Adjoint expressibility audit](../adjoint-expressibility-audit.md)
- [Model capability and contract closure audit](../model-capability-contract-closure-audit.md)
- [Task 0002](0002-shape-and-dimension-model.md)
- [Task 0017G](0017g-slice-semantics.md)
- [Task 0017H](0017h-slice-tensor-expressions.md)
- [Task 0018M](0018m-symbolic-extent-expressions.md)
- [Task 0018M1](0018m1-dynamic-extent-adoption.md)
- [Task 0018R](0018r-slice-and-window-public-contract-cleanup.md)
- [Task 0023](0023-adjoint-expressibility-audit.md)
- [Task 0023C](0023c-slice-update-and-target-relative-crop.md)
- [Task 0025](0025-canonical-tensor-producer-outputs.md)
- [Task 0025B](0025b-binding-aware-expansion.md)
- [Task 0025C](0025c-portable-functional-scatter-reduction-semantics.md)
- [Completed Compiler 0004A](../../compiler/tasks/0004a-exact-composition-gradient-rule-extensions.md)
- [Completed Compiler 0005B](../../compiler/tasks/0005b-reduction-scan-softmax-statistics-and-normalization-gradient-completion.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- Work remains inside `modules/model` plus directly affected explanatory/planning documentation.
- `Tensor` remains public model state, not graph intermediate representation.
- `SLICE` and `SLICE_UPDATE` remain backend-independent logical semantics. They gain no backend
  support, route, materialization, cost, kernel, storage, runtime, or execution metadata.
- `SliceAttrs` continues to identify an exact finite coordinate sequence. `sliceByLength` is a new
  Shape-aware public construction path for that existing normalized value, not a second bounds
  language.
- `CropToShapeAttrs` continues to retain only exact target and prefix Shapes. Under
  `SLICE_UPDATE`, its target is the exact update-region Shape, not the operation result Shape.
- Model validates locally provable rank, type, coordinate, and static-fit facts. It does not bind,
  solve, or store a deferred constraint.
- A selected unresolved extent does not prevent finite extraction when the requested sequence has
  a provably non-negative first and last coordinate. Only its upper bound remains deferred.
- Resolved layout may be derived only through the existing non-negative-stride
  `LayoutDescriptor` contract.
- Every derived expression retains the canonical exact wrapper returned by its identity-distinct
  producer. No graph-local ID or compiler state enters Tensor.
- Compiler 0005C must explicitly adopt the new forms. Existing compiler inference, preflight, and
  gradient code is review-only evidence and receives no change or implied guarantee here.
- No dependency, package boundary, module ownership, or lifecycle rule changes.
- Stop if implementation requires another type/member beyond the two approved public methods,
  compiler work, Shape/Dimension changes, a negative-stride layout, or architecture changes.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.operation.layout`
- `io.github.pho001.synaptik.model.tensor`
- `io.github.pho001.synaptik.model.shape`

Packages added, moved, renamed, or removed:

- None.

Type placement:

- `io.github.pho001.synaptik.model.operation.layout.SliceKind` — continues to own the complete
  typed signature variants for extraction and functional replacement.
- `io.github.pho001.synaptik.model.operation.layout.CropToShapeAttrs` — continues to own the exact
  target/prefix Shape pair used by target-relative extraction and placement.
- `io.github.pho001.synaptik.model.tensor.TensorSliceExpressions` — continues to own local
  slice-request validation, Shape/layout derivation, and exact `SLICE` construction.
- `io.github.pho001.synaptik.model.tensor.TensorSlicePlacementExpressions` — continues to own local
  functional-placement validation and exact `SLICE_UPDATE` construction.
- `io.github.pho001.synaptik.model.tensor.Tensor` — remains the public fluent facade.

No generic utility, new family package, second slice helper, or compiler-facing adapter is added.

## Exact API and signature matrix

### Public Tensor methods

Add exactly:

```java
public Tensor sliceByLength(
        long[] starts, long[] lengths, int[] axes, long[] steps) {
    return TensorSliceExpressions.applyByLength(
            this, starts, lengths, axes, steps);
}

public Tensor sliceUpdate(Tensor update, Shape prefixShape) {
    return TensorSlicePlacementExpressions.update(
            this, update, prefixShape);
}
```

Each method has one return statement and one matching helper call. Neither method reads fields,
performs validation, calls another public Tensor method, or constructs attributes directly. Both
are non-static, non-synchronized, non-varargs methods.

There is exactly one public method named `sliceByLength` and exactly two public overloads named
`sliceUpdate`. No alias such as `dynamicSlice`, `sliceLengths`, `updateSlice`, `place`, or
`sliceScatter` is added.

The declared public Tensor method inventory changes exactly from 200 to 202.

### Operation signatures

`SliceKind` retains exactly:

```java
SLICE,
SLICE_UPDATE
```

Its exact ordered signature lists become:

```text
SLICE:
  SliceAttrs       fixed 1 input, fixed 1 output
  CropToShapeAttrs fixed 1 input, fixed 1 output

SLICE_UPDATE:
  SliceAttrs       fixed 2 inputs, fixed 1 output
  CropToShapeAttrs fixed 2 inputs, fixed 1 output
```

Do not reorder an existing variant. Do not add a permissive `OperationAttrs` signature.
`SLICE_UPDATE/CropToShapeAttrs` ordered inputs are always `[base, update]`.

### Existing semantic records

`SliceAttrs` remains bytecode and behavior unchanged:

```java
public record SliceAttrs(
        List<Long> starts,
        List<Long> lengths,
        List<Integer> axes,
        List<Long> steps) implements OperationAttrs
```

`CropToShapeAttrs` remains bytecode and behavior unchanged:

```java
public record CropToShapeAttrs(
        Shape targetShape,
        Shape prefixShape) implements OperationAttrs
```

Only `CropToShapeAttrs` Javadoc changes. It must explain:

- for `SLICE`, `targetShape` is the exact extraction result Shape;
- for `SLICE_UPDATE`, `targetShape` is the exact update-region Shape and the operation result keeps
  the base Shape;
- `prefixShape` identifies the exact non-negative logical extent preceding the region on each
  axis in both variants; and
- the record itself performs no rank, input, type, bounds, layout, binding, compiler, or execution
  validation.

## `sliceByLength` contract

### Meaning

For input rank `R`, the four arrays are parallel and equal length. Each raw axis is normalized
once against `R`, and normalized axes must be distinct. Entry `i` requests:

```text
coordinate(k) = starts[i] + k * steps[i]
0 <= k < lengths[i]
```

The public `starts` values are already normalized logical coordinates and must be non-negative.
Lengths are finite, non-negative selected-coordinate counts. Steps are signed and non-zero.
No bound is clamped, shifted, wrapped, or inferred.

The exact normalized arrays are represented by one `SliceAttrs`, except that every zero-length
entry stores canonical start zero as required by the completed attributes contract.

### Helper surface

`TensorSliceExpressions` remains final, package-private, field-free, and non-instantiable. It
retains its exact nine existing methods and adds exactly:

```java
static Tensor applyByLength(
        Tensor input, long[] starts, long[] lengths, int[] axes, long[] steps)

private static SliceAttrs normalizeByLength(
        Shape inputShape, long[] starts, long[] lengths, int[] axes, long[] steps)
```

The helper therefore has exactly eleven declared non-synthetic methods and one private
zero-argument constructor. Existing methods, signatures, and behavior remain unchanged.

`applyByLength` uses the existing `deriveShape`, `resolveViewLayout`, and `create` methods exactly
once each after normalization. Add no duplicate Shape, layout, or factory path.

### Container validation and ownership

`applyByLength` validates in this exact order:

1. null-check `input`, `starts`, `lengths`, `axes`, and `steps`, in order, with those exact
   parameter-name messages;
2. require all four array lengths to match, otherwise throw `IllegalArgumentException` with:

   ```text
   starts, lengths, axes, and steps must have matching lengths
   ```

3. clone `starts`, `lengths`, `axes`, and `steps` exactly once each, in that order;
4. read the exact input descriptor and Shape once;
5. call `normalizeByLength` once with only the private copies;
6. call existing Shape derivation once;
7. call existing layout resolution once; and
8. call existing creation once.

Null and length failures occur before cloning, descriptor inspection, semantic-list allocation, or
Tensor identity consumption. Caller arrays are never retained or mutated. Concurrent caller
mutation after the clones cannot affect the expression.

### Entry validation order and diagnostics

`normalizeByLength` processes entries in ascending caller index. It creates one rank-sized
duplicate-detection array and four constructor-local lists. At each index it validates exactly:

1. normalize the raw `int` axis once through a `long` intermediate; a negative axis adds rank once;
2. reject an out-of-range normalized axis with:

   ```text
   slice by length axis <rawAxis> at index <index> is outside rank <rank>
   ```

3. reject the first repeated normalized axis with:

   ```text
   slice by length contains duplicate normalized axis <axis> at index <index>
   ```

4. reject a negative start with:

   ```text
   starts[<index>] must be non-negative: <start>
   ```

5. reject a negative length with:

   ```text
   lengths[<index>] must be non-negative: <length>
   ```

6. reject a zero step with:

   ```text
   steps[<index>] must be non-zero: 0
   ```

7. validate or defer the finite coordinate sequence as specified below; and
8. append canonical start, unchanged length, normalized axis, and unchanged step.

Axis and duplicate failures therefore precede start/length/step failures at the same entry.
Earlier entries are fully checked before later entries. Exactly one `SliceAttrs` is constructed
after all entries pass.

### Checked coordinates and bounds

When `length > 0`, calculate the final selected coordinate exactly:

```java
last = Math.addExact(
        start,
        Math.multiplyExact(Math.subtractExact(length, 1L), step));
```

The first coordinate is `start`; the final coordinate is `last`. Because the sequence is monotonic,
both are in bounds exactly when their minimum is non-negative and, for a static input extent,
their maximum is less than that extent.

Reject a provable lower-bound failure or any fully static upper-bound failure with:

```text
slice by length coordinates at index <index> do not fit input extent <extent>: start=<start>, length=<length>, step=<step>
```

Rules:

- `start` is already non-negative by the preceding validation.
- A negative `last` is always rejected, including when the selected input extent is unresolved.
- If the selected input Dimension is static, reject when `start >= extent` or `last >= extent`.
- If the selected input Dimension is unresolved and `start` and `last` are non-negative, defer
  only the proof that both are below the eventual extent.
- Checked subtraction, multiplication, or addition overflow propagates as
  `ArithmeticException`.
- There is no clamp, wrap, truncation, suffix adjustment, or partial selection.

This is the only deferred `sliceByLength` validation: an upper-bound comparison involving the
unresolved selected input extent.

### Zero-length behavior

When `length == 0`:

- the entry selects no coordinate;
- the caller start still must be non-negative and the step still must be non-zero under the exact
  validation order above;
- no first/final-coordinate or input-bound proof occurs;
- the stored `SliceAttrs` start is canonical zero, regardless of the caller's non-negative start;
- the selected result Dimension is a new static zero;
- static-zero, other static, named, expression, and constrained-unknown selected input extents are
  all accepted; and
- the complete result layout is unresolved because it is known empty.

An empty four-array request remains a fresh explicit identity `SLICE` and preserves every exact
input Dimension reference.

### Result Shape

Use the existing `deriveShape` contract:

- rank is unchanged;
- each selected axis is replaced by one new `StaticDimension(length)`;
- every unselected Dimension reference is retained exactly;
- a scalar accepts only empty arrays and preserves the canonical scalar Shape;
- empty attributes preserve all exact Dimension references; and
- selected unresolved Dimensions are allowed because the requested finite result extent is static.

Examples:

| Input Shape | Starts | Lengths | Axes | Steps | Result Shape | Proof |
|---|---:|---:|---:|---:|---|---|
| `[3, 8]` | `[1]` | `[3]` | `[1]` | `[2]` | `[3, 3]` | coordinates `[1,3,5]` fit statically |
| `[N, 8]` | `[2]` | `[3]` | `[0]` | `[2]` | `[3, 8]` | lower bound proven; upper bound deferred against `N` |
| `[N]` | `[4]` | `[3]` | `[0]` | `[-2]` | `[3]` | coordinates `[4,2,0]`; upper bound deferred against `N` |
| `[N]` | `[1]` | `[2]` | `[0]` | `[-2]` | invalid | final coordinate `-1` is provably below zero |
| `[0, M]` | `[99]` | `[0]` | `[1]` | `[-1]` | `[0, 0]` | start canonicalizes to zero; no coordinate exists |
| scalar `[]` | `[]` | `[]` | `[]` | `[]` | scalar `[]` | explicit identity |

These are metadata examples. They do not claim compiler adoption, value evaluation, or backend
execution.

### Result layout

Reuse the current resolved-view proof exactly:

1. unresolved input layout -> unresolved result layout;
2. known-empty result -> unresolved result layout;
3. any negative step -> unresolved result layout;
4. otherwise derive one checked positive-step view from the exact resolved input layout.

For step four, use each original selected input stride to calculate checked
`offset + start * stride` and checked `stride * step`, then create one
`LayoutDescriptor.of(resultShape, resultStrides, resultOffset, true)`.

Every current resolved input layout kind is accepted. Selected unresolved extents necessarily
have unresolved input geometry and therefore remain layout-unresolved. Logical view metadata
attaches no storage and promises no executable alias or materialization route.

### Result metadata and identity

Every valid call:

- retains the exact input data type and `requiresGrad`;
- uses the derived same-rank Shape and resolved-or-unresolved layout above;
- has no label and no host storage;
- records `Operation(SliceKind.SLICE, attrs)`;
- records exact ordered inputs `[input]`;
- has one identity-distinct producer, one output descriptor, and output index zero;
- returns the canonical exact fresh Tensor wrapper for that producer slot; and
- consumes exactly one Tensor ID at final creation.

Every validation, Shape, or layout-arithmetic failure precedes derived creation and consumes no
Tensor ID. Identifier exhaustion remains the final failure.

## `sliceUpdate(update, prefixShape)` contract

### Meaning

For equal rank Shapes, axis `i` places the update region into the base interval:

```text
[prefixShape[i], prefixShape[i] + update.shape[i])
```

`CropToShapeAttrs.targetShape` is the exact `update.descriptor().shape()` reference.
`CropToShapeAttrs.prefixShape` is the exact caller-supplied `prefixShape` reference.

Every logical update coordinate replaces exactly one corresponding base coordinate. Values outside
the region retain base values. The operation performs no addition, overlap reduction, mutation,
clamping, padding, broadcasting, or partial placement.

All current data types are accepted with exact base/update type equality. Scalar base, update, and
prefix Shapes form one valid scalar replacement. Zero update extents are valid when every fully
static region fits.

### Helper surface

`TensorSlicePlacementExpressions` remains final, package-private, field-free, and
non-instantiable. It retains every current method and adds exactly these overloads:

```java
static Tensor update(Tensor base, Tensor update, Shape prefixShape)

private static Tensor createUpdate(
        Tensor base,
        Tensor update,
        TensorDescriptor baseDescriptor,
        TensorDescriptor updateDescriptor,
        CropToShapeAttrs attrs)
```

The helper therefore has exactly ten declared non-synthetic methods:

- two `update` overloads;
- `cropToShape`;
- `normalizeUpdate`;
- `normalizeUpdateStart`;
- `expectedUpdateShape`;
- `validateStaticCropBounds`;
- two typed `createUpdate` overloads; and
- `createCrop`.

The existing array-based `update` and `createUpdate(..., SliceAttrs)` methods remain unchanged in
signature and behavior. No broad `OperationAttrs` construction path is added.

### Validation order and diagnostics

The new helper overload validates exactly:

1. null-check `base`, `update`, and `prefixShape`, in order, with those exact messages;
2. read base and update descriptors, in that order;
3. require exact data-type equality, reusing:

   ```text
   slice update data types must match: base=<dataType>, update=<dataType>
   ```

4. read the exact base and update Shapes;
5. require update rank equal base rank, reusing:

   ```text
   slice update rank must match base rank: base=<rank>, update=<rank>
   ```

6. require prefix rank equal base rank, otherwise throw:

   ```text
   slice update prefix rank must match base rank: base=<rank>, prefix=<rank>
   ```

7. inspect axes in ascending order and validate every fully static
   `prefix + update <= base` fit with checked addition;
8. construct exactly one `CropToShapeAttrs(updateShape, prefixShape)`; and
9. call the typed target-relative `createUpdate` overload once.

If base, prefix, and update Dimensions on an axis are all static, calculate
`prefix + update` with `Math.addExact`. Reject the first out-of-bounds axis with:

```text
slice update region exceeds base extent at axis <axis>: base=<baseExtent>, prefix=<prefixExtent>, update=<updateExtent>
```

If any one of the three Dimensions on an axis is unresolved, defer that axis's fit without
partial arithmetic, expression evaluation, or a hidden constraint object. This is the only
deferred validation in this overload.

Every local failure and checked overflow occurs before attribute construction, producer creation,
or Tensor ID allocation.

### Result metadata, producer, and identity

The target-relative `createUpdate` overload constructs:

- exact base data type;
- exact base Shape reference as the result Shape;
- unresolved layout;
- `requiresGrad = base.requiresGrad || update.requiresGrad`;
- `Operation(SliceKind.SLICE_UPDATE, attrs)`;
- exact ordered inputs `[base, update]`;
- one output descriptor and output-index-zero provenance;
- absent label and storage; and
- one fresh canonical Tensor wrapper and one final Tensor ID.

Neither input is mutated. Labels, prior provenance, layouts, storage associations, liveness, and
values are not inspected or retained except for the exact input references in the new producer.

## Existing-behavior preservation matrix

| Existing contract | Required result |
|---|---|
| `slice(long[], long[], int[], long[])` | Exact signature, directional raw-bound normalization, clamping, validation, Shape/layout, producer, ID behavior, and Javadoc remain unchanged. |
| Both `sliceAxis` overloads | Exact signatures and one-general-path behavior remain unchanged. |
| `flip(int...)` | Exact one-`SLICE` behavior remains unchanged. |
| `sliceUpdate(Tensor,long[],int[],long[])` | Exact signature, cloning, validation order/messages, `SliceAttrs`, metadata, producer, and ID behavior remain unchanged. |
| `cropToShape(Shape,Shape)` | Exact signature, validation, `SLICE/CropToShapeAttrs`, exact target Shape, and metadata remain unchanged. |
| `SliceAttrs` | Declaration, components, validation, snapshots, accessors, equality, hashing, text, and bytecode behavior remain unchanged. |
| `CropToShapeAttrs` | Declaration and behavior remain unchanged; Javadoc alone gains the exact update pairing. |
| `SliceKind` | Enum constants/order unchanged; only the appended `SLICE_UPDATE` signature variant and Javadoc change. |
| Existing compiler code | No source, test, or documented support guarantee changes in this task. |

Focused regression tests must exercise representative positive, negative, empty, dynamic, failure,
metadata, and ID cases for the existing public slice/update paths in addition to the new forms.

## Affected files

Production Java (5):

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/SliceKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/CropToShapeAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorSliceExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorSlicePlacementExpressions.java`

Tests (15):

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/OperationSignatureTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorSliceExpressionTest.java`
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

Documentation and planning (8):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task specification
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/modules/compiler/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless an inconsistency requires stopping:

- `SliceAttrs`, `Shape`, every Dimension implementation, `LayoutDescriptor`, `TensorDescriptor`,
  `TensorFactory`, `TensorProducer`, `TensorProvenance`, and current extraction/update tests not
  listed above
- current Compiler `LayoutInference`, `AutogradPreflight`, `LayoutGradientRules`, graph capture,
  constraint types, tests, and completed Compiler 0004A/0005B task evidence
- completed Model task specifications and both completed Model audit artifacts
- Training/Runtime/Public APIs, architecture/ADRs/tests, backend-conformance/integration,
  dependencies, Gradle, Java 26 configuration, other modules, and later task directories

## Maximum scope

At most the exact 28 paths above may change: five production Java paths, fifteen test paths, and
eight documentation/planning paths.

This cohesive capability is above the normal task guardrail because thirteen existing tests lock
the global public Tensor method count and must move atomically from 200 to 202. The remaining two
test paths own the exact signature matrix and the two focused slice helper/public contracts.

Do not use the allowance for unrelated changes. Stop if implementation needs path 29, another
production/test/documentation file, another public method/type/helper, a `SliceAttrs` executable
change, compiler source/test work, dependency/build/architecture work, or another module.

## Javadoc and documentation requirements

- Apply General, API/Javadoc, Planning, and Example profiles in the independent documentation
  pass.
- Fully document both new public methods, both added helper entries, the private length
  normalizer, the typed target-relative creation overload, and every changed helper/type contract.
- Document every parameter, result, nullability/ownership rule, mutation boundary, validation
  order, exact failure class, checked overflow, metadata result, and ID side effect.
- Explain normalized finite-sequence input versus the existing raw half-open slice input.
- Explain zero-length canonical start, unresolved selected extents, the exact lower-versus-upper
  proof boundary, and why no deferred constraint object is stored in Model.
- Explain resolved positive-step view geometry and every unresolved-layout case without promising
  storage aliasing, materialization, lowering, or execution.
- Explain `CropToShapeAttrs.targetShape` as extraction result Shape under `SLICE` versus update
  region Shape under `SLICE_UPDATE`.
- Tensor API must include complete static, dynamic, negative-step, zero-length, and symbolic
  placement examples with inputs, attributes, result Shapes, and interpretation.
- Compile API must call both new model expressions current while explicitly assigning their
  inference-constraint/preflight/gradient adoption to Draft Compiler 0005C. It must not claim
  current compiler or execution support.
- Glossary must update the existing Slice, Slice Update, Target-relative crop, normalized axis,
  and provenance entries. Add a new term only if a reusable distinction cannot be explained under
  those existing entries.
- Capabilities must add the two current public transformations without rewriting the historical
  200-method task-0024 closure verdict.
- Task/master plans/roadmap must keep Model 0025D as the sole Ready detailed task until
  implementation and documentation validation complete.
- Record reasoned no-change conclusions for `SliceAttrs`; existing raw slice/update/crop APIs;
  Shape/Dimension/layout/descriptor/factory/producer contracts; completed audits/tasks;
  Training/Runtime/Public APIs; architecture/ADRs/tests; backend conformance/integration;
  dependencies/Gradle/Java 26; other modules; and later compiler task files.

## Acceptance criteria

- Exactly the two approved public methods are added and no alias/overload appears.
- Public Tensor method count is exactly 202 in all thirteen live locks.
- `SliceKind` constant order is unchanged and the signature matrix has exactly the four required
  variants in the specified order.
- `SliceAttrs` declaration and executable behavior are unchanged.
- `CropToShapeAttrs` declaration/executable behavior are unchanged and its Javadoc accurately
  covers extraction and placement.
- `TensorSliceExpressions` has exactly eleven methods, with one new package-private entry and one
  private normalizer; all existing method signatures/behavior remain exact.
- `sliceByLength` applies exact null/length/axis/duplicate/start/length/step/coordinate validation
  order and messages, clones arrays, and constructs one attributes value.
- Every non-empty request performs checked final-coordinate arithmetic and rejects all provable
  lower/static-upper failures before ID allocation.
- Only an unresolved selected extent's upper-bound proof is deferred.
- Zero length canonicalizes start to zero, checks neither bound, yields a new static zero selected
  Dimension, and leaves layout unresolved.
- Result Shape uses static requested lengths and exact unaffected Dimension references.
- Resolved layout exists only for a non-empty result with resolved input geometry and all-positive
  steps; all other cases are unresolved.
- `sliceByLength` preserves exact type/eligibility and creates one fresh unlabeled storage-free
  `SLICE/SliceAttrs/[input]` result with output index zero.
- `TensorSlicePlacementExpressions` has exactly ten methods with the two typed `update` and
  `createUpdate` overloads; existing array-update signatures/behavior remain exact.
- The new `sliceUpdate` validates null/type/update-rank/prefix-rank/static-fit in exact order,
  defers fit whenever any involved Dimension is unresolved, and retains exact update/prefix Shape
  references.
- Its result has exact base Shape/type, eligibility OR, unresolved layout, exact
  `SLICE_UPDATE/CropToShapeAttrs/[base, update]`, output index zero, no label/storage, one producer,
  one ID, and fresh canonical wrapper.
- Every local validation/overflow failure consumes no ID; exhaustion remains final.
- Existing raw slice, both axis forms, flip, array-based slice update, and crop behavior/signatures
  remain unchanged and covered.
- No value/storage mutation, compiler behavior, gradient rule, backend behavior, dependency,
  build, architecture, or unrelated capability is added.
- Exact 28-path/package scope, separate documentation pass, tests, Javadoc, examples,
  Markdown/scope/status checks, and `git diff --check` all pass before Complete.
- Compiler 0005C–0005E and 0006 remain Draft without detailed task specifications.

## Tests / validation

Focused implementation command:

```bash
./gradlew :modules:model:test \
  --tests io.github.pho001.synaptik.model.operation.OperationSignatureTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorSliceExpressionTest \
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

After executable Java stabilizes, run one final module suite:

```bash
./gradlew :modules:model:test
```

Focused coverage must verify:

- exact enum/signature/public/helper inventories and absence of aliases/new kinds/types;
- all data types and valid gradient-eligibility combinations;
- null and unequal-array precedence;
- caller-array ownership and concurrent post-clone mutation isolation;
- positive/negative/extreme axes, starts, lengths, and steps;
- duplicate precedence and exact diagnostics;
- static/unresolved/scalar/zero-extent selected and unselected Dimensions;
- checked final-coordinate lower/upper/overflow cases;
- zero-length canonicalization and empty-array identity;
- exact unaffected Dimension identity and selected static lengths;
- dense/offset/strided/broadcast/unresolved/empty layout outcomes;
- checked offset/stride/span failures;
- exact target/prefix reference retention and static/deferred placement fits;
- descriptor/operation/producer/provenance/label/storage/freshness/ID behavior;
- unchanged raw slice, axis slice, flip, array update, and crop behavior; and
- exact method count 202 in every live inventory lock.

The separate documentation-focused pass receives the final diff and successful model-test
evidence. It must not repeat successful Java tests unless executable Java changes afterward or a
concrete stale-evidence risk is recorded. After final Javadocs it runs:

```bash
./gradlew :modules:model:javadoc
python3 /tmp/validate_synaptik_markdown.py
git diff --check
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git status --short
```

If the temporary Markdown validator is absent, create an equivalent validator outside the
repository. It must check repository-local targets and heading anchors, balanced fences, final
newlines, and trailing whitespace.

The documentation pass must also:

- compile and run the final Java 26 metadata example;
- inspect generated public Javadocs and package-private helper Javadocs in source;
- use reflection or `javap` to verify exact signatures, helper inventories, and 202 public methods;
- verify unchanged `SliceAttrs` declaration/behavior and existing public signatures;
- verify exact 28-path scope and package placement;
- verify task/master/roadmap status synchronization;
- verify Compiler 0005C–0005E and 0006 remain Draft and have no task files;
- verify no compiler/model-shape/architecture/Gradle/dependency/other-module change; and
- record every command/result, reused evidence, examples, limitations, and no-change conclusion.

Repository-wide validation is deferred to Compiler 0005E's first-order capability checkpoint and
continuous integration. This task changes one module, no dependency, build configuration, or
architecture boundary.

## Dependencies

- Model 0002 supplies immutable non-negative static and unresolved Dimensions, exact Shapes, and
  rank/axis vocabulary.
- Model 0003 supplies the existing non-negative-stride resolved layout contract.
- Model 0005–0007 and 0011–0013 supply typed operations/signatures, descriptors, Tensor, identity,
  factory, producer, and provenance.
- Model 0017G–0017H and 0018R supply normalized signed finite `SliceAttrs`, raw public slicing,
  Shape/layout derivation, flip, and exact validation precedents.
- Model 0018M–0018M1 supply unresolved and expression Dimension values without runtime binding.
- Model 0023/0023C prove and implement generally useful functional slice update and target-relative
  crop rather than backward-only operations.
- Model 0025 supplies canonical exact producer output wrappers.
- Model 0025B and Compiler 0005B establish the current binding-aware Model/Compiler handoff.
- Model 0025C is the immediately preceding Complete model prerequisite for Compiler 0005C.

All dependencies are Complete.

## Follow-up tasks

- Compiler 0005C remains Draft. After Model 0025D is Complete, it must explicitly:
  - adopt `sliceByLength`-origin `SLICE/SliceAttrs` constraints and fail-closed preflight for
    unresolved selected extents;
  - distinguish `SLICE` and `SLICE_UPDATE` for both exact attributes variants before casting;
  - infer `SLICE_UPDATE/CropToShapeAttrs` as exact base Shape/type with update/prefix fit
    constraints;
  - construct correct base/update cotangents for target-relative placement;
  - preserve all existing normalized array-slice gradients and constraints; and
  - keep index/configuration roles non-differentiable and all new obligations fail closed.
- Compiler 0005D–0005E and 0006 remain ordered Draft rows after 0005C.
- Future backend implementations require conformance coverage for finite signed extraction and
  target-relative functional replacement once execution exists.

Do not create any follow-up specification during this task.

## Architecture impact

Expected impact: None.

The architecture already assigns these semantic values, public Tensor expressions, local
Shape/layout derivation, and immutable producer metadata to Model. Compiler already owns graph-wide
constraints and gradients, and backend prepare already owns materialization and lowering.

If implementation requires a Shape/Dimension contract change, compiler implementation, another
public abstraction, dependency, module boundary, or lifecycle change, stop and report the exact
conflict before editing architecture files.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules and
profiles, roadmap, model/compiler master plans, model capabilities and completed audits, completed
Model tasks 0002/0017G/0017H/0018M/0018M1/0018R/0023/0023C/0025/0025B/0025C, completed Compiler
0004A/0005B, this task, and every affected/review-only source/test/API/glossary path named here.

Implement Model 0025D exactly within its 28 authorized paths. Add only
sliceByLength(long[],long[],int[],long[]) and sliceUpdate(Tensor,Shape), the exact
SLICE_UPDATE/CropToShapeAttrs signature, and the specified existing-helper extensions. Preserve
SliceAttrs, existing raw slice/axis/flip, array-based sliceUpdate, cropToShape, Shape/Dimension,
producer/provenance, and all architecture boundaries. Add no kind, attrs/Shape type, package,
alias, backward-only operation, compiler behavior, value execution, dependency, build, or
architecture change. Stop on validation-order/message, scope, completed-contract, or architecture
conflict.

Run focused tests while developing and one final model suite after executable Java stabilizes.
Then hand the actual diff and exact Java evidence to a separate clean-context documentation-focused
agent in the same overall change. That pass must independently inspect final source/tests,
finalize affected Javadocs, Tensor/Compile APIs, glossary, capabilities, task/master/roadmap
status, runnable example, and documentation/scope/no-change checks. It must reuse successful Java
evidence unless executable behavior changes or it records a concrete stale-evidence risk.

Do not mark 0025D Complete until both passes and every acceptance criterion succeed. Keep Compiler
0005C-0005E and 0006 Draft without task files.
```

## Documentation-agent handoff

The implementation agent must hand over:

- this exact task and the final source/test diff;
- exact model-test commands, counts, and result;
- both public signatures and helper surface changes;
- signature-matrix, coordinate, zero-length, Shape/layout, target/prefix, validation, metadata,
  provenance, freshness, and ID contracts;
- unchanged-behavior evidence for existing slice/update/crop paths;
- the no-compiler-change boundary and future Compiler 0005C obligations;
- expected Tensor/Compile API, glossary, capability, and planning changes;
- exact 28-path scope and 202-method inventory; and
- all Javadoc/Markdown/example/scope/status commands.

The documentation agent must inspect final source and tests rather than rely only on that summary.

## Local decisions

- `sliceByLength` is the sole new normalized finite-extraction spelling. Existing `slice` remains
  the sole raw directional half-open spelling.
- Public axes retain current positive/negative input syntax and normalize once. Starts and lengths
  are already normalized non-negative values and are never clamped.
- A zero-length entry accepts any non-negative caller start but stores canonical zero because no
  logical coordinate exists and `SliceAttrs` requires canonical empty state.
- A non-empty unresolved selected extent defers only its upper bound; lower bounds and checked
  coordinate representation remain locally decidable.
- Existing Shape/layout/creation methods are reused so length-based and raw-bound extraction share
  one result contract after normalized attributes exist.
- `CropToShapeAttrs` is reused for target-relative update because its exact update Shape and prefix
  Shape are sufficient; another attributes type would duplicate one region language.
- Typed overloads preserve the existing `SliceAttrs` update path and avoid a broad generic
  `OperationAttrs` creation helper.
- Target-relative placement always leaves layout unresolved and returns base Shape; the exact
  target Shape belongs to the update region, not the result descriptor.
- Compiler 0005C owns explicit adoption. Shared current attributes do not silently promote the new
  public forms to completed compiler support.

## Known limitations

- Deferred upper-bound and fits-within obligations are not stored or solved by Model.
- Negative-step results have unresolved layout under the current non-negative-stride descriptor.
- No value evaluation, storage alias/copy decision, mutation, compiler adoption, gradient rule,
  lowering, backend behavior, runtime, or execution exists in this task.
- Existing raw slice still requires static selected extents because raw negative-bound
  normalization and directional clamping need the concrete extent.
- Compiler 0005C–0005E and 0006 remain incomplete Draft work.

## Validation evidence

Planning context: `/root/plan_model_0025d_dynamic_slice`.

- Read `AGENTS.md`, the complete architecture contract and focused model/compiler/lifecycle
  boundaries, documentation rules and General/API-Javadoc/Planning/Example profiles, planning
  guide, roadmap, Model and Compiler master plans, Model capabilities and both completed audits,
  completed Shape/dynamic/slice/crop/producer tasks, completed Compiler work through 0005B, current
  Tensor/Compile APIs and glossary, and the exact source/test/compiler-inference evidence named by
  this task.
- Reviewed the shared uncommitted 0025C implementation/documentation diff first. This planning
  change preserves every existing 0025C path and does not modify its completed task file.
- No architecture conflict was found. Model already owns both requested semantic construction
  forms; compiler inference/constraint/preflight/gradient adoption remains cleanly separable.
- Source inventory confirms exactly 200 current public Tensor methods, thirteen live
  method-count locks, current nine-method slice helper, current eight-method placement helper, and
  the exact three-variant slice signature matrix before this task.
- Current compiler evidence confirms that its inference/preflight/gradient paths distinguish only
  their already completed slice cases. This task changes no compiler source or tests and assigns
  deliberate adoption of both new forms to Draft Compiler 0005C.
- Planning scope contains exactly four paths: this new task, Model master plan, Compiler master
  plan, and roadmap. Implementation scope is exactly the 28 paths listed above.
- `rg` found `0025D`, `0025d`, `sliceByLength`, or the selected title in exactly those four
  planning paths and nowhere under other documentation or modules.
- The implementation-scope inventory contains exactly five production, fifteen test, and eight
  documentation/planning paths; every named existing path is present and this task is the eighth
  documentation/planning path.
- The existing repository Markdown validator passed 12 changed Markdown files and 719 local
  links, including targets and heading anchors.
- The task has 36 balanced fence delimiters; all four planning files have final newlines; no
  trailing-whitespace error was reported.
- At planning time exactly one detailed task had status `Ready`: this task. Compiler 0005C–0005E
  and 0006 remained Draft, and no matching later compiler task file existed.
- `git diff --check` passed. Shared worktree status still contains the pre-existing completed
  0025C implementation/documentation paths plus these four planning paths; no shared 0025C path
  was reverted or rewritten as 0025D implementation.
- No Gradle test or Javadoc task is required for this planning-only change because it modifies no
  Java, executable behavior, current API guide, or Javadoc.

## Implementation notes

- Added exactly the two approved public `Tensor` methods:
  `sliceByLength(long[], long[], int[], long[])` and `sliceUpdate(Tensor, Shape)`.
- Extended the existing slice helpers without introducing another facade, semantic operation,
  attribute record, or alias. The final public surface has 202 methods; the extraction helper has
  11 methods; the placement helper has 10 methods.
- `sliceByLength` normalizes axes once, validates starts, lengths, steps, duplicates, lower bounds,
  coordinate representation, and every statically decidable upper bound, and canonicalizes
  zero-length selections to zero start. It retains `SLICE/SliceAttrs` metadata, the exact selected
  Shape, and current layout/provenance behavior.
- `sliceUpdate(Tensor, Shape)` retains the exact update Shape and prefix Shape in
  `SLICE_UPDATE/CropToShapeAttrs`, returns the base Shape, and deliberately leaves layout
  unresolved.
- Updated all thirteen live public-method count locks and added focused construction, validation,
  metadata, zero-length, unresolved-extent, overflow, and overload-boundary coverage in the two
  existing slice test classes.
- Finalized affected API and helper Javadocs, Tensor API examples and comparison tables, Compile
  API deferral language, glossary entries, capability inventory, and dependent planning status.
- Reviewed and intentionally left unchanged `SliceAttrs`; the existing raw slice, axis-slice,
  flip, array-bound update, and crop contracts; Shape, Dimension, layout, descriptor, factory, and
  producer contracts; completed audits and historical task records; Training and Runtime APIs;
  architecture, ADRs, and architecture tests; backend conformance and integration tests; Gradle,
  dependencies, and Java 26 configuration; other modules; and later compiler task files. None of
  those contracts changes: this task adds Model expression construction only, while Compiler
  0005C retains ownership of constraint adoption, preflight, and gradients.

## Completion summary

Completed changes: implemented finite length-defined slice extraction across unresolved selected
extents and exact-Shape target-relative slice placement, with the required tests, Javadocs, API
documentation, glossary, capability inventory, and planning synchronization.

Files changed or created: exactly the five production, fifteen test, and eight
documentation/planning paths authorized by this task, in addition to the separately completed
0025C paths already present in the shared worktree.

Tests or validation performed:

- Focused exact 15-suite validation passed 159 tests after correcting one stale test-only Tensor
  method-count lock found by the first run.
- Final `./gradlew :modules:model:test` passed 127 suites and 1,031 tests with zero failures,
  errors, or skips.
- `./gradlew :modules:model:javadoc` completed successfully.
- Compiled and ran the documented `DynamicSliceMetadataExample` with Java and `javac` 26.0.1; its
  six output lines matched the documented output exactly.
- Reflection and `javap` checks confirmed 202 public Tensor methods, no prohibited aliases, helper
  counts of 11 and 10, `SliceKind` order `[SLICE, SLICE_UPDATE]`, the exact four-signature matrix,
  unchanged `SliceAttrs` components, and exact `CropToShapeAttrs(Shape, Shape)` components.
- The repository Markdown validator passed all 12 changed Markdown files and 718 local links;
  `git diff --check` passed.

Unresolved issues: none. Compiler 0005C–0005E and 0006 intentionally remain Draft without detailed
task files.

Required follow-up: prepare Compiler 0005C when selected by the roadmap; it owns explicit adoption
of the retained slice constraints, fail-closed preflight, and gradient rules.

Implementation context: `/root/implement_model_0025d_dynamic_slice`.

Documentation context:
`/root/implement_model_0025d_dynamic_slice/docs_0025d_dynamic_slice`.

Status: Complete
