# Task 0018R: Slice and Window Public-Contract Cleanup

## Status

Complete

## Goal

Finalize slicing and the public window-transform surface before the model API stabilizes:

```text
raw half-open slice request
  -> one normalized coordinate sequence per selected static axis
  -> one SLICE occurrence

flip over explicit axes
  -> the same normalized negative-step SLICE occurrence

public window API
  -> unfold, unfold2d, fold2d

compiler-only window semantic
  -> FOLD_AXIS, retained for task 0023
```

General slice requests must support positive and negative non-zero steps with exact directional
half-open meaning, checked result-length arithmetic, deterministic normalization and failures,
and no undocumented sentinel. `flip(int... axes)` must be a convenience over one `SLICE`, not a
new operation kind or a chain. Positive-step slices keep the existing resolved logical view when
it is provable. Negative-step slices remain layout-unresolved because the current
`LayoutDescriptor` forbids negative strides.

At the same time, remove the provisional public `Tensor.foldAxis` surface. Retain
`WindowTransformKind.FOLD_AXIS` and `FoldAxisAttrs` as backend-neutral compiler-facing model
semantics; task 0023 owns their first compiler-generated construction and later use. Public
`unfold`, `unfold2d`, and `fold2d` remain unchanged.

## Mental model

For a selected axis of static extent `D`, public bounds are raw signed coordinates and the end is
exclusive in the direction of travel:

```text
step > 0: start, start + step, ... while coordinate < end
step < 0: start, start + step, ... while coordinate > end
```

For conceptual values `[0, 1, 2, 3, 4]`:

```text
sliceAxis(0, 1, 5, 2)    -> coordinates [1, 3]
sliceAxis(0, 4, -6, -1)  -> coordinates [4, 3, 2, 1, 0]
flip(0)                   -> the same coordinate sequence [4, 3, 2, 1, 0]
```

The public request remains half-open. The normalized attributes instead store the exact finite
coordinate sequence as `start`, `length`, and signed `step`:

```text
starts=[4], lengths=[5], axes=[0], steps=[-1]
coordinate(k) = start + k * step, for 0 <= k < length
```

This is equivalent to the normalized half-open request but needs neither an ambiguous `-1` end
sentinel nor an optional-bound hierarchy.

## Current problems

- `SliceAttrs` rejects negative steps, so the primitive cannot represent a complete strided slice
  family and `flip` cannot be expressed as a convenience.
- A normalized negative-step exclusive end may be the conceptual boundary before coordinate zero.
  Storing that boundary as raw `-1` would conflict with the current “all normalized coordinates
  are non-negative” contract and would be an undocumented magic sentinel.
- The current helper adds a static extent to a raw negative bound and supports only the positive
  clamp range. It has no directional clamp, negative-step length formula, or explicit zero-step
  contract.
- `LayoutDescriptor` accepts only non-negative element strides. Publishing a negative-stride
  reverse view would violate a completed foundational contract.
- `sliceAxis(int, long, long)` cannot directly express a signed step even though the general
  primitive can.
- There is no final `flip` spelling, duplicate-axis policy, empty-axis policy, or producer/ID
  contract.
- Public `Tensor.foldAxis` exposes a specialized scatter-add adjoint even though the selected
  baseline now classifies `FOLD_AXIS` as compiler-only.
- Existing Tensor, helper, API, glossary, and planning inventories still describe the provisional
  positive-step and public-fold contracts.

## Decision and rationale

### Normalized slice attributes

Replace the second `SliceAttrs` component from exclusive ends to selected lengths:

```java
public record SliceAttrs(
        List<Long> starts,
        List<Long> lengths,
        List<Integer> axes,
        List<Long> steps) implements OperationAttrs
```

Entry `i` means `lengths[i]` coordinates beginning at `starts[i]` and advancing by the signed
non-zero `steps[i]` on normalized axis `axes[i]`. Empty entries use canonical start zero and length
zero. A non-empty entry must have a representable non-negative final coordinate.

This keeps the existing four-parallel-list representation and order-sensitive value semantics.
It adds no public bound-state type, sealed hierarchy, nullable bound, optional primitive, builder,
or sentinel. Storing a selected length is the smallest complete normalized contract because the
input-aware public boundary already computes that length for the result Shape.

Rejected alternatives:

| Alternative | Decision | Reason |
|---|---|---|
| Store `-1` as a normalized negative-step end | Reject | It is a direction-dependent magic sentinel and contradicts non-negative normalized coordinate terminology. |
| Add a `SliceBound` hierarchy or bound-state enum/value pair | Reject | It enlarges the public semantic model when normalized start, length, and step already identify the exact finite coordinate sequence. |
| Preserve raw starts/ends in attributes | Reject | Raw negative values and clamping are request syntax tied to an input extent, not normalized intrinsic semantics. |
| Split positive and negative slice kinds or attributes | Reject | Direction is completely represented by the signed step; two identities would duplicate one operation. |

### Public slice surface

Keep the four primitive arrays as the canonical general public request:

```java
Tensor slice(long[] starts, long[] ends, int[] axes, long[] steps)
```

Keep the step-one convenience and add one explicit-step overload:

```java
Tensor sliceAxis(int axis, long fromInclusive, long toExclusive)
Tensor sliceAxis(int axis, long fromInclusive, long toExclusive, long step)
```

The three-argument form delegates through the shared helper with step `1L`. The four-argument form
is needed so callers can express one-axis reverse and strided requests without allocating four
arrays. Both produce exactly one `SLICE` occurrence.

Do not add list overloads, nullable/default arrays, omitted bounds, a Python-style slice object,
ellipsis, implicit axes, or open-ended bounds. ONNX Slice is useful terminology evidence for
signed steps, adding a static extent once to negative bounds, and direction-dependent clamping,
but Synaptik does not adopt ONNX's omitted-input or `INT_MIN`/`INT_MAX` open-bound convention.
See the official [ONNX Slice contract](https://onnx.ai/onnx/operators/onnx__Slice.html).

### Flip convenience

Add exactly:

```java
public Tensor flip(int... axes)
```

The axes array is explicit, non-null, defensively cloned, and processed in caller order. Negative
axes add rank once. The first invalid raw axis or repeated normalized axis fails. Duplicate axes
are rejected rather than canceled or silently deduplicated.

An empty axes array is an explicit identity flip, matching the existing empty `SliceAttrs`
identity rather than meaning “all axes.” It is valid for scalar, static, zero-extent, and dynamic
Shapes and still creates one fresh `SLICE` occurrence. A non-empty request on a scalar fails axis
validation. A selected zero-extent static axis contributes canonical start zero, length zero, and
step `-1`. A selected non-zero static axis contributes start `D - 1`, length `D`, and step `-1`.
Any selected named, expression, or unknown dynamic Dimension is rejected; unselected dynamic
Dimensions retain exact references.

All requested axes are encoded in one `SliceAttrs` value and one `TensorProducer`. Flip does not
call public `slice`, create one operation per axis, reverse the axes order, or add a `FLIP`
`OperationKind`. A flip is exactly a normalized negative-step slice, so another kind would add no
semantic information. The official [JAX flip reference](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.flip.html)
confirms established “reverse along an axis or sequence of axes” terminology; Synaptik deliberately
requires an explicit axis list and gives empty varargs identity meaning.

### Static and dynamic selected axes

General `slice`, both `sliceAxis` overloads, and non-empty `flip` keep the current rule that every
selected Dimension must be `StaticDimension`. This applies to both step directions.

The signed raw-bound normalization and clamp depend on the concrete selected extent. The current
symbolic extent foundation can store formulas but cannot bind a runtime size or represent a raw
bound clamped against an unknown extent. Accepting only some positive requests on dynamic axes
would create a second deferred-validation contract and inconsistent negative behavior. Compiler
shape constraints and runtime binding remain separately owned.

### Layout policy

Positive-step slices preserve current behavior:

- unresolved input or a known-empty result -> unresolved result layout;
- resolved non-empty input -> checked start-adjusted offset and original-stride-times-step in one
  new view-marked `LayoutDescriptor`.

If any selected step is negative, the complete result layout is unresolved, including a mixed-
direction multi-axis slice. Do not calculate an offset or publish negative strides. The logical
Shape, operation, producer, and provenance remain exact; later planning/backend prepare may choose
a view, copy, or kernel route without changing this model contract.

### FOLD_AXIS disposition

Remove public `Tensor.foldAxis(int, long, long)` and remove every package-private helper method,
test path, API inventory entry, example, or alias that constructs it from public Tensor state.
Do not deprecate or retain a forwarding bridge.

Retain both:

```text
WindowTransformKind.FOLD_AXIS
FoldAxisAttrs(axis, outputSize, step)
```

They remain public Java semantic values because `modules/model` owns the backend-neutral operation
vocabulary that compiler-generated graphs may contain. They are not a public Tensor expression.
Their exact one-input/one-output `OperationSignature` remains unchanged. Task 0023 owns the first
production compiler-generated construction, operand compatibility, and backward-graph use.

Deleting and later recreating the semantic types is rejected: it would create needless source and
documentation churn, weaken the already selected compiler-only capability, and leave the planned
adjoint without a stable model contract between tasks 0018R and 0023. Keeping public `foldAxis` is
also rejected because it preserves a public use case that the capability reset explicitly did not
select.

### Unchanged public window operations

`Tensor.unfold`, `Tensor.unfold2d`, and `Tensor.fold2d` retain their exact signatures, validation
order/messages, static/dynamic Shape policies, checked arithmetic, data-type eligibility,
unresolved layouts, producer/provenance behavior, and ID side effects. `WindowTransformKind`
retains all four constants in the current order. `UnfoldAxisAttrs`, `Window2dAttrs`, and
`Fold2dAttrs` retain declarations and behavior.

No symbolic window-formula adoption is included. The current public window compatibility checks
require static transformed channel/spatial or selected dimensions, and this cleanup has no reason
to change those semantics.

## Before / after API and semantic tables

### Slice surface

| Current contract | Final contract | Disposition |
|---|---|---|
| `SliceAttrs(starts, ends, axes, steps)` with positive steps | `SliceAttrs(starts, lengths, axes, steps)` with signed non-zero steps | Atomic normalized representation change; no compatibility constructor/accessor. |
| `Tensor.slice(long[], long[], int[], long[])` | same signature | Keep; add negative-step normalization. |
| `Tensor.sliceAxis(int, long, long)` | same signature | Keep as exact step-one convenience. |
| no step-aware one-axis convenience | `Tensor.sliceAxis(int, long, long, long)` | Add; one `SLICE`. |
| no flip convenience | `Tensor.flip(int... axes)` | Add; one `SLICE`, no `FLIP` kind. |

### Window surface

| Current contract | Final contract | Disposition |
|---|---|---|
| public `Tensor.foldAxis(int, long, long)` | absent | Remove without alias. |
| helper `TensorWindowExpressions.foldAxis` and `foldAxisShape` | absent | Remove every public-construction path. |
| `WindowTransformKind.FOLD_AXIS` + `FoldAxisAttrs` | same declarations and signature | Retain as compiler-only model semantics owned for use by task 0023. |
| `Tensor.unfold`, `unfold2d`, `fold2d` | unchanged | Preserve exact public semantics. |
| `UNFOLD_AXIS`, `UNFOLD2D`, `FOLD2D` | unchanged | Preserve exact semantic identities. |

### Directional raw-bound normalization

Let `D` be the selected static extent. A negative raw bound first adds `D` exactly once. The sum
cannot overflow because the raw value is negative and `D` is non-negative. Then clamp:

| Step | Start clamp | Exclusive-end clamp | Non-empty condition |
|---|---|---|---|
| `step > 0` | `[0, D]` | `[0, D]` | `start < end` |
| `step < 0`, `D > 0` | `[0, D - 1]` | `[-1, D - 1]` | `start > end` |
| `step < 0`, `D == 0` | canonical empty start `0` | canonical empty | never |

An explicit raw `-1` is still a coordinate relative to the end: it becomes `D - 1`. To select
through coordinate zero with a negative step, the raw exclusive end must normalize below zero;
for example `-D - 1` when representable. Callers do not need to manufacture such a bound for
reversal because `flip` constructs normalized sequence attributes directly.

### Result and occurrence behavior

| Case | Result |
|---|---|
| Positive non-empty step | Same-rank Shape; resolved checked logical view when input layout is resolved. |
| Positive empty step | Same-rank zero-extent Shape; unresolved layout. |
| Negative non-empty or empty step | Same-rank Shape; unresolved layout. |
| Empty general slice arrays | Fresh explicit identity `SLICE`; exact input Shape references. |
| Empty flip axes | Fresh explicit identity `SLICE`; exact input Shape references. |
| Selected dynamic axis | Reject before Tensor ID allocation. |
| Unselected dynamic axis | Preserve the exact Dimension reference. |
| Every success | Preserve exact data type and `requiresGrad`; fresh unlabeled storage-free Tensor; one producer, one output descriptor, provenance output index zero, exact ordered input `[receiver]`. |

## Scope

- Redesign `SliceAttrs` to exact normalized starts, lengths, distinct axes, and signed non-zero
  steps.
- Update `SliceKind` documentation and retain its exact one-input/one-output signature.
- Add directional half-open normalization, canonical empty entries, checked length formulas, and
  signed-step Shape derivation to `TensorSliceExpressions`.
- Preserve positive resolved-view layout and make every negative-step result layout-unresolved.
- Keep the general slice method, keep step-one `sliceAxis`, add explicit-step `sliceAxis`, and add
  varargs `flip`.
- Define exact array ownership, validation order/messages, duplicate handling, freshness,
  producer/provenance, and identifier side effects.
- Remove `Tensor.foldAxis` plus its helper and focused public-expression tests.
- Retain `FOLD_AXIS`/`FoldAxisAttrs` as compiler-only semantic contracts and correct their
  current-versus-planned Javadocs.
- Preserve all other public window behavior and semantic declarations.
- Update exact Tensor and helper reflection inventories and focused semantic/expression tests.
- Finalize Tensor API, Compile API, glossary, capability baseline, task evidence, model master
  plan, and roadmap through the mandatory independent documentation-focused pass.

## Out of scope

- omitted/open bounds, ellipsis, nullable/default arrays, implicit axes, a slice object, or Python
  `None` conventions
- selected dynamic-axis slicing, runtime Shape binding, symbolic clamp/min/max expressions,
  graph-wide constraints, or compiler shape inference
- negative-stride `LayoutDescriptor`, layout-kind redesign, symbolic layout, physical storage
  aliasing, eager copying, materialization choice, or value execution
- a `FLIP` kind, flip composition, all-axes default, duplicate cancellation, or axis sorting
- slice value bounds beyond static logical extents, slice backward, gradient rules, autograd
  traversal, or compiler-generated backward implementation
- removal or implementation use of compiler-only `FOLD_AXIS`; task 0023 owns compiler construction
- changes to public `unfold`, `unfold2d`, `fold2d`, their current semantics, or two-dimensional
  geometry
- split, chunk, diagonal, roll, rotate, select, gather, scatter, pad, tile, composition, or another
  operation family
- compiler capture/canonicalization, planning, prepare, runtime, backend/ONNX lowering, kernels,
  conformance execution, engine, tracing, or training behavior
- dependencies, Gradle, Java version, preview/incubator configuration, architecture contracts,
  focused architecture documents, architecture tests, another module, commit, or push
- a detailed task-0018S-or-later specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md), especially model ownership of Tensor,
  operation semantics, Shape/layout values, and immutable graph model
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)
- [Model master plan](../master-plan.md)
- [Task 0002](0002-shape-and-dimension-model.md)
- [Task 0003](0003-layout-descriptor-model.md)
- [Task 0017G](0017g-slice-semantics.md)
- [Task 0017H](0017h-slice-tensor-expressions.md)
- [Task 0017M](0017m-unfold-and-fold-semantics.md)
- [Task 0017N](0017n-unfold-and-fold-tensor-expressions.md)
- [Task 0018K](0018k-operation-signature-and-construction-hardening.md)
- [Task 0018M](0018m-symbolic-extent-expressions.md)
- [Task 0018P](0018p-elementwise-semantic-cleanup.md)
- [Task 0018Q](0018q-masked-reduction-redesign.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- Work remains entirely in `modules/model` plus explanatory/planning documentation.
- Tensor remains public mutable API state, not graph IR.
- `SLICE` and retained `FOLD_AXIS` describe backend-independent semantics only. Neither kind gains
  backend support, a route, cost, fusion, kernel, storage, runtime, or compiler-service field.
- Normalized `SliceAttrs` contains intrinsic finite coordinate-sequence state, never an input
  Shape, raw bounds, Tensor, layout, producer, policy flag, or hidden sentinel.
- Every expression result uses the existing identity-distinct `TensorProducer` and output-index
  provenance model; no graph-local ID enters Tensor.
- Resolved layout may be published only through the completed `LayoutDescriptor` contract.
  Negative strides are forbidden, so a reverse slice cannot be described as resolved geometry.
- Compiler owns graph capture, canonicalization, constraints, gradients, and backward graph
  construction. Task 0023 owns compiler-generated `FOLD_AXIS` construction.
- Planning/backend prepare own materialization and concrete lowering; model does not select view
  versus copy execution.
- No module dependency or package boundary changes are authorized.
- Stop if implementation needs a negative-stride layout, dynamic bound solver, another operation
  kind, compiler code, or architecture change.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.operation.layout` — owns `SLICE`, normalized slice attributes,
  and retained window-transform semantic values.
- `io.github.pho001.synaptik.model.tensor` — owns the public fluent slice/flip/window surface,
  local Shape/layout validation, and derived Tensor construction.
- `io.github.pho001.synaptik.model.shape` — supplies immutable static/dynamic Dimensions and Shape.
- `io.github.pho001.synaptik.model.layout` — supplies non-negative resolved logical geometry.

Packages added or moved: None.

Type placement remains unchanged:

- `SliceAttrs` stays beside layout/view semantic values because it is intrinsic `SLICE` state.
- `TensorSliceExpressions` stays package-private beside Tensor and the derived factory seam.
- `TensorWindowExpressions` remains the shared public unfold/unfold2d/fold2d construction helper.
- `FoldAxisAttrs` remains beside `WindowTransformKind` as compiler-facing model semantics.

## Required contracts

### `SliceAttrs`

The record has exactly four components in this order:

```java
List<Long> starts
List<Long> lengths
List<Integer> axes
List<Long> steps
```

It has one canonical constructor, four explicit documented accessors, and generated object
methods. Add no compatibility `ends()` accessor, secondary constructor, bound type, entry type,
factory, builder, output Shape, or cache.

Constructor validation order is exact:

1. null-check `starts`, `lengths`, `axes`, and `steps`, in order, using component-name messages;
2. reject unequal sizes with
   `starts, lengths, axes, and steps must have matching sizes`;
3. create one constructor-local set for seen axes;
4. inspect entries in ascending index order;
5. null-check start, length, axis, and step, in order, with messages `starts[i]`, `lengths[i]`,
   `axes[i]`, and `steps[i]`;
6. reject negative start with `starts[i] must be non-negative: <value>`;
7. reject negative length with `lengths[i] must be non-negative: <value>`;
8. reject negative axis with `axes[i] must be non-negative: <value>`;
9. reject the first repeated axis with `axes contains duplicate axis <axis> at index <i>`;
10. reject step zero with `steps[i] must be non-zero: 0`;
11. if length is zero, require start zero, otherwise reject with
    `starts[i] must be zero when lengths[i] is zero: <start>`;
12. if length is positive, calculate the last selected coordinate with checked
    `start + (length - 1) * step`; propagate `ArithmeticException` on overflow and reject a
    negative result with `last slice coordinate at index <i> must be non-negative: <last>`;
13. only after all entries pass, store one `List.copyOf` snapshot for each component in order.

Empty lists remain valid identity semantics. Every non-zero signed step, including
`Long.MIN_VALUE`, is structurally accepted when the declared sequence has a representable
non-negative final coordinate. Coordinates, lengths, and steps use `long`; axes use `int`.

### General and one-axis normalization

`TensorSliceExpressions` remains final, package-private, field-free, and non-instantiable. It has
exactly these nine static methods after cleanup:

```java
static Tensor apply(Tensor input, long[] starts, long[] ends, int[] axes, long[] steps)
static Tensor applyAxis(
        Tensor input, int axis, long fromInclusive, long toExclusive, long step)
static Tensor flip(Tensor input, int[] axes)
private static SliceAttrs normalize(
        Shape inputShape, long[] starts, long[] ends, int[] axes, long[] steps)
private static long normalizeBound(
        long rawBound, long dimensionSize, long step, boolean startBound)
private static long sliceLength(long start, long end, long step)
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

Add no field, nested type, overload, cache, or synthetic/lambda-generated method. For a
negative-step zero-extent selected dimension, normalization bypasses bound arithmetic and emits
the canonical empty entry directly. Otherwise `normalizeBound` applies the direction and
start-versus-end clamp table exactly.

`TensorSliceExpressions.apply` retains the current reference and equal-length validation order,
then clones all four arrays before element inspection. Entry processing order is:

1. normalize the raw axis once with a `long` intermediate;
2. reject an invalid axis with
   `slice axis <raw> at index <i> is outside rank <rank>`;
3. reject the first duplicate normalized axis with
   `slice contains duplicate normalized axis <axis> at index <i>`;
4. reject zero step with `steps[i] must be non-zero: 0`;
5. require a `StaticDimension`, otherwise fail with
   `slice axis <axis> at index <i> must have a statically known dimension`;
6. normalize/clamp the copied start and end according to the table above;
7. calculate the selected length with checked arithmetic;
8. store canonical start zero for an empty sequence, otherwise the normalized start; and
9. append length, normalized axis, and the unchanged signed step.

Length formulas are exact and avoid negating `Long.MIN_VALUE`:

```text
step > 0 and start < end:
  length = 1 + (end - 1 - start) / step

step < 0 and start > end:
  length = 1 - ((start - 1 - end) / step)

otherwise:
  length = 0
```

Use `Math.addExact`/`subtractExact` for the additions and subtractions. Normalized ranges make the
result fit in non-negative `long`, including a full axis of extent `Long.MAX_VALUE`.

The step-one public overload calls the shared one-axis helper with `1L`. The explicit-step overload
passes its exact step. Neither method allocates a different semantic representation or calls
another public Tensor method.

### Flip validation and construction

`TensorSliceExpressions.flip(input, axes)` performs this order:

1. null-check `input`, then `axes`, with exact parameter-name messages;
2. clone the caller array once;
3. read the exact descriptor and Shape once;
4. process copied axes in caller order with normalized duplicate detection;
5. use `flip axis <raw> at index <i> is outside rank <rank>` for invalid axes;
6. use `flip contains duplicate normalized axis <axis> at index <i>` for duplicates;
7. use `flip axis <axis> at index <i> must have a statically known dimension` for a selected
   dynamic Dimension;
8. build one normalized `SliceAttrs` as specified above; and
9. use the common Shape, layout, and creation path exactly once.

Empty copied axes create empty attributes. Selected axes retain request order. Add no implicit
all-axes expansion. Every success consumes one Tensor identity at final derived creation; every
validation or checked-arithmetic failure occurs first and consumes none.

### Shape and layout derivation

Result Shape copies every exact input Dimension reference, then replaces each selected axis with
one new `StaticDimension(length)`. Rank is unchanged. Scalar plus empty entries remains the
canonical scalar Shape. Unselected named, expression, and unknown Dimensions retain exact
references.

Layout resolution checks, in order:

1. unresolved input -> unresolved result;
2. known-empty result -> unresolved result;
3. any negative step -> unresolved result;
4. otherwise derive positive-step view geometry from the resolved input.

For step four, copy input strides once, begin with the exact input offset, and for every entry use
the original selected input stride to calculate checked `offset + start * stride` and checked
`stride * step`. Construct one `LayoutDescriptor.of(resultShape, strides, offset, true)`. Every
resolved input layout kind remains accepted. Overflow propagates before identity allocation.

### Result metadata, producer, provenance, and IDs

The common creation path preserves exact input DataType and `requiresGrad`, uses the derived
Shape and resolved-or-unresolved layout, creates `Operation(SliceKind.SLICE, attrs)`, and calls the
existing derived factory once with ordered inputs `[input]`.

Every valid general slice, one-axis slice, flip, identity request, repeated call, and nested call
returns a fresh unlabeled storage-free Tensor. It has one identity-distinct producer, one exact
output descriptor, and `TensorProvenance.outputIndex() == 0`. Input descriptor, label,
provenance, storage association, liveness, and values remain unchanged.

### Public FOLD_AXIS removal and retained semantics

Delete the public Tensor declaration and Javadoc for `foldAxis`. Delete
`TensorWindowExpressions.foldAxis`, `foldAxisShape`, and the now-unused numeric-type validator.
The helper becomes field-free with exactly eleven methods: the three public-operation entries for
unfold/unfold2d/fold2d plus the eight remaining private methods. Remove foldAxis public-expression
tests and reflection expectations without weakening unfold/unfold2d/fold2d coverage.

Keep all `FOLD_AXIS` semantic tests. Update `WindowTransformKind` and `FoldAxisAttrs` Javadocs to
state that no public Tensor expression constructs them and task 0023 owns compiler generation.
Do not change enum order, record components, validation, signature, equality, or composition.

## Operation signatures

- `SliceKind.SLICE` continues to accept exactly `SliceAttrs`, one input, and one output.
- `WindowTransformKind.FOLD_AXIS` continues to accept exactly `FoldAxisAttrs`, one input, and one
  output even though public Tensor no longer constructs it.
- All other `WindowTransformKind` signatures remain unchanged.
- No `FLIP` kind/signature and no generic signature-mechanics change is permitted.

## Affected files

Production Java:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/SliceKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/SliceAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/WindowTransformKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/FoldAxisAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorSliceExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorWindowExpressions.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/layout/SliceSemanticsTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorSliceExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorWindowExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`

Documentation and planning:

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task specification
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless an inconsistency requires stopping: `UnfoldAxisAttrs`,
`Window2dAttrs`, `Fold2dAttrs`, `WindowTransformSemanticsTest`, Shape/Dimension,
`LayoutDescriptor`, operation-signature mechanics/tests, Tensor descriptor/factory/producer/
provenance contracts, Training API, completed task history, architecture/ADRs/tests,
backend-conformance/integration tests, Gradle/dependencies, other modules, and later tasks.

## Maximum scope

At most the exact 18 paths above may change: seven production files, four tests, and seven
documentation/planning files. This is the smallest justified atomic exception at the upper edge of
the normal task guardrail. Splitting slice semantics/public methods from FOLD_AXIS demotion would
temporarily leave the public capability inventory inconsistent and require two migrations through
the same Tensor, helper, tests, API, glossary, capability, and planning files.

Do not use the allowance for unrelated refactoring or formatting. Stop if a nineteenth path,
another production/test type, semantic declaration change outside the listed types, dependency,
build/architecture change, another module, or detailed later task is required.

## Javadoc and documentation requirements

- Apply General, API/Javadoc, Planning, and Example profiles in the independent documentation pass.
- Fully document the new `SliceAttrs` components, constructor, accessors, signed sequence formula,
  ownership, value semantics, failures, and absence of sentinels/input state.
- Fully document all four final slice/flip Tensor methods and the helper type, constructor, and
  exact methods after implementation.
- Explain raw directional half-open normalization, the explicit `-1` coordinate distinction,
  checked length formulas, static selected-axis rule, canonical empty state, duplicate order, and
  `Long.MIN_VALUE` step handling.
- Include complete positive, negative, empty, zero-extent, mixed-axis, and flip examples with exact
  normalized attributes and result Shapes.
- Explain positive resolved-view geometry and why negative steps are unresolved under the current
  non-negative-stride layout contract without implying a copy or execution.
- Explain one producer/output-index-zero provenance and exact ID side effects for every convenience.
- Remove public foldAxis from Tensor/Compile API inventories and examples while retaining a clear
  compiler-only `FOLD_AXIS` semantic entry linked to task 0023.
- Review all unchanged public window methods and semantic Javadocs and record reasoned no-change
  conclusions.
- Update the glossary's Slice, normalized axis, OperationAttrs, OperationKind, provenance, and
  window-transform entries consistently.
- Use only the official ONNX and JAX links above for external terminology evidence. Synaptik's
  selected differences must be stated locally.
- Record reasoned no-change conclusions for Training API, Shape/Dimension, LayoutDescriptor,
  operation signatures, Tensor producer/provenance/factory, focused architecture material,
  architecture tests, conformance/integration, Gradle/dependencies, other modules, and completed
  task history.

## Acceptance criteria

- `SliceAttrs` has exactly the four required components and signed non-zero sequence semantics,
  with no end sentinel, compatibility constructor/accessor, or added abstraction.
- Exact constructor validation, messages, checked last-coordinate arithmetic, snapshots, and
  value semantics match this task.
- General slice supports both directions, zero rejects exactly, raw axes/bounds normalize and
  clamp directionally, and checked length formulas cover `Long.MIN_VALUE`/`Long.MAX_VALUE` cases.
- The public Tensor surface contains exactly general slice, step-one sliceAxis, step-aware
  sliceAxis, and varargs flip; no `foldAxis` or flip alias/kind remains. The final declared public
  Tensor method count is 111.
- Flip clones axes, rejects duplicates after normalization, preserves request order, accepts empty
  identity, handles scalar/zero/static/dynamic cases exactly, and creates one `SLICE` occurrence.
- Result Shapes retain rank and exact unaffected Dimensions. All selected extents are exact static
  lengths.
- Positive non-empty slices retain exact checked resolved logical view behavior. Empty,
  unresolved-input, and any-negative-step slices remain layout-unresolved.
- Every valid request preserves exact type/eligibility, is fresh/unlabeled/storage-free, and has
  one exact `[input]` producer with output index zero. Early failures consume no ID.
- `Tensor.foldAxis` and every public/helper construction path or alias are absent.
- `FOLD_AXIS` and `FoldAxisAttrs` remain exact compiler-only semantic values with unchanged
  declarations, signature, validation, and focused semantic coverage; task 0023 ownership is
  explicit and no ownerless transition remains.
- Public unfold, unfold2d, and fold2d declarations, behavior, tests, and Javadocs remain accurate.
- No negative-stride layout, dynamic-bound solving, value execution, gradients, compiler,
  backend, dependency, build, architecture, or unrelated operation capability is added.
- Focused tests, final model tests, model Javadoc, documentation checks, exact 18-path scope,
  synchronized Ready/Complete status, and `git diff --check` pass.
- A separate clean-context documentation pass finalizes Javadocs, Tensor/Compile APIs, glossary,
  capability/task/master/roadmap evidence, official links, and no-change conclusions in the same
  overall change.
- 0018S and every later task remain Draft without detailed specifications.

## Tests / validation

During implementation, run the focused contract set:

```bash
./gradlew :modules:model:test \
  --tests io.github.pho001.synaptik.model.operation.layout.SliceSemanticsTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorSliceExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorWindowExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorTest \
  --tests io.github.pho001.synaptik.model.operation.layout.WindowTransformSemanticsTest \
  --tests io.github.pho001.synaptik.model.operation.OperationSignatureTest \
  --tests io.github.pho001.synaptik.model.layout.LayoutDescriptorTest
```

After executable Java stabilizes, record one final module run:

```bash
./gradlew :modules:model:test
```

Automated tests must cover exact record/public/helper inventories, signed sequence validation,
all normalization/clamping directions, zero and extreme steps, full reversal, empty results,
static/dynamic/zero/scalar Shapes, mixed signed multi-axis slices, positive layout geometry,
negative unresolved layout, one-operation flip, duplicate order, producer/provenance/descriptor
identity, storage non-interference, ID failure/success effects, absence of public foldAxis, retained
FOLD_AXIS composition, and unchanged unfold/unfold2d/fold2d behavior.

The separate documentation-focused pass receives the final diff and model-test evidence. It does
not repeat successful Java tests unless executable Java changes afterward or a concrete risk is
recorded. After final Javadoc edits it runs:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

It also:

- compiles and runs the final Java 26 slice/flip metadata example;
- inspects generated Javadoc for all changed public contracts and retained compiler-only wording;
- validates every changed local Markdown link and GitHub-style anchor;
- checks the two official external URLs used above;
- checks balanced fences, final newlines, trailing whitespace, terminology, and authority;
- verifies exact live Java/public API absence of `Tensor.foldAxis` and `FLIP`;
- verifies exact 18-path scope and no executable Java change after reused test evidence;
- confirms task/master/roadmap/capability synchronization, 0018S Draft status, and absence of any
  detailed 0018S-or-later task; and
- confirms no architecture, Gradle, dependency, other-module, commit, or push change.

Repository-wide `./gradlew test` is deferred to the recorded public-surface cleanup checkpoint
after task 0018S. This task changes one module and no dependency, build, or architecture boundary.

## Dependencies

- Tasks 0002 and 0003 — current Shape/Dimension and non-negative-stride layout contracts —
  Complete.
- Tasks 0017G and 0017H — provisional positive-step slice semantics and public expressions —
  Complete historical prerequisites replaced by this cleanup.
- Tasks 0017M and 0017N — window semantic values and provisional four-method public window
  surface — Complete historical prerequisites.
- Task 0018K — exact kind/attributes signatures and occurrence cardinality — Complete.
- Tasks 0018L and 0018M — producer/output-index provenance and symbolic extent foundation —
  Complete.
- Tasks 0018P and 0018Q — immediately preceding public semantic cleanups — Complete.

All dependencies are Complete. No architecture or package decision blocks this task.

## Follow-up tasks

- 0018S remains Draft for TensorFactory surface cleanup.
- 0023 retains ownership of compiler-generated `FOLD_AXIS`, slice/window backward semantics, and
  backward-graph construction decisions. It reuses the semantics retained here.
- Compiler tasks later own capture, slice-chain/flip canonicalization, dynamic constraints, and
  gradient expansion.
- Planning/backend/conformance tasks later own materialization, reverse-copy or view routes,
  numerical window accumulation, and execution parity.

Do not create a detailed follow-up specification in this task.

## Architecture impact

Expected impact: None.

The architecture already assigns backend-independent Tensor/operation/Shape/layout semantics to
model and compiler-generated backward graphs to compiler. This task tightens the public model
surface and leaves the retained adjoint semantic with its selected future owner. It changes no
module ownership, dependency direction, lifecycle boundary, or architecture rule.

If implementation requires negative-stride layout, runtime binding, compiler code, a new
operation family, or another architecture decision, stop and report the exact conflict before
editing architecture files.

## Implementation prompt

Use this prompt in a separate agentic task/thread with clean context:

```text
You are working in the Synaptik repository.

Read AGENTS.md, ARCHITECTURE.md, documentation/planning rules, roadmap, model capabilities/master
plan, tasks 0002/0003/0017G/0017H/0017M/0017N/0018K/0018M/0018P/0018Q/0018R, relevant
Tensor/Compile/Training API and glossary sections, and every affected or review-only source/test
named by task 0018R in full.

Implement task 0018R exactly. Atomically redesign SliceAttrs to normalized starts/lengths/axes/
signed-nonzero-steps, implement exact directional raw half-open normalization and checked length
arithmetic, preserve only positive-step resolved views, add step-aware sliceAxis and one-SLICE
varargs flip, and remove public/helper foldAxis construction while retaining FOLD_AXIS and
FoldAxisAttrs as compiler-only model semantics for task 0023. Preserve unfold, unfold2d, fold2d,
producer/provenance, ID, and architecture contracts. Stay within the exact 18 paths, stop on scope
or architecture conflict, and do not commit or push.

Run the focused command and final model suite after executable Java stabilizes. Then hand the
actual diff and exact evidence to a separate clean-context documentation-focused agent in the
same overall change. That agent must independently finalize Javadocs, Tensor/Compile APIs,
glossary, capability/task/master/roadmap status, official links, the runnable example, and every
specified documentation/scope/no-change check without repeating successful Java tests unless
executable behavior changes or a concrete risk is recorded.

Mark 0018R Complete only after both passes succeed. Leave 0018S and every later task Draft without
a detailed specification.
```

## Local decisions

- Normalized attributes use start plus length plus signed step; no sentinel or additional bound
  abstraction is needed.
- General slice arrays remain the public primitive. The explicit-step one-axis overload is added
  because it avoids four-array ceremony without adding semantics.
- Flip uses explicit varargs axes, rejects duplicates, gives empty axes identity meaning, and
  creates exactly one SLICE occurrence.
- Selected dimensions remain static-only for every slice direction; unselected dynamic
  Dimensions retain exact references.
- Positive resolved logical views remain; any negative step makes the result layout unresolved.
- FOLD_AXIS/FoldAxisAttrs remain compiler-only model semantics now, while every public Tensor
  construction path is removed and task 0023 owns compiler generation.
- No completed historical task is rewritten; current API, capability, and planning documents
  record the cleanup.

## Known limitations

- Selected dynamic extents cannot be sliced or flipped until an owning compiler/lifecycle task
  defines clamp and runtime-binding semantics.
- Negative-step results have no resolved layout even when a backend might later implement a
  reverse view. The current descriptor cannot represent negative strides.
- No omitted bounds or implicit all-axes flip exists.
- Slice and window values, gradients, compiler capture, lowering, and execution remain future
  work.
- FOLD_AXIS is representable but has no current public Tensor constructor or compiler emission
  until task 0023.

## Validation evidence

Planning context: clean task `/root/plan_0018r`.

- Read the repository instructions and architecture contract; current architecture index;
  documentation rules and General/API-Javadoc/Planning/Example profiles; planning guide and
  roadmap; model capability baseline and master plan; completed tasks 0002, 0003, 0017G, 0017H,
  0017M, 0017N, 0018K, 0018M, 0018P, and 0018Q; relevant Tensor/Compile/Training API and glossary
  sections; and current slice/window/Shape/Dimension/layout/signature/producer/provenance source,
  tests, and public inventories.
- Consulted only official primary external documentation: ONNX Slice for signed-step and
  direction-dependent clamp terminology, and JAX flip for axis-sequence reversal terminology.
  Synaptik's explicit arrays, static selected-axis rule, normalized length attributes, empty-axis
  identity, and no sentinel remain locally selected differences.
- Planning found no architecture conflict. The current model boundary can implement the cleanup
  without dependency, build, module, or lifecycle changes.
- Initial `git status --short` was empty, so planning started from a clean worktree and introduced
  only the four intended planning paths.
- The canonical-section scan found every required task-specification section, plus the decision,
  before/after, operation-signature, documentation, mental-model, and current-problem sections.
  No unresolved design placeholder remains in the Ready contract.
- Task/master-plan/roadmap scans identify 0018R as Ready, 0018Q as Complete, and 0018S plus every
  later row as Draft. The master-plan dependency cell is `0017G–0017N, 0018K–0018M`.
- The task-file inventory found no detailed 0018S-or-later specification.
- A targeted Markdown target checker resolved all 324 local links across this task, capabilities,
  master plan, and roadmap; none of those links has a fragment anchor. All 32 task backtick-fence
  markers and both master-plan fence markers are balanced; no tilde fence is present.
- Direct final URL checks returned HTTP 200 for the official ONNX Slice and JAX flip references.
- All four planning files are non-empty, end with a newline, and contain no trailing whitespace.
- Final planning scope is exactly four paths: this new task plus the capability baseline, model
  master plan, and roadmap. No Java, test, API, glossary, architecture, Gradle, dependency,
  completed-task, other-module, or later-task path changed during planning.
- `git diff --check` passed with no output. The untracked task separately passed the trailing-
  whitespace and final-newline checks that a tracked diff does not inspect.
- No Gradle test or Javadoc task ran because this planning-only change modifies no Java,
  executable behavior, public Javadoc, or current API reference.

Implementation validation: clean context `/root/task_0018r_implementation`.

- The focused command specified by this task passed with `BUILD SUCCESSFUL`: 78 tests across
  `SliceSemanticsTest`, `TensorSliceExpressionTest`, `TensorWindowExpressionTest`, `TensorTest`,
  `WindowTransformSemanticsTest`, `OperationSignatureTest`, and `LayoutDescriptorTest`; zero
  failures.
- `./gradlew :modules:model:test` passed with `BUILD SUCCESSFUL`: 715 tests across 88 suites; zero
  failures, errors, or skips.
- A migration-intermediate `compileTestJava` failure exposed stale test references while the
  contract was being replaced. The implementation context corrected those references before the
  final focused and module runs; it is not final failing evidence.
- No executable Java changed after those successful runs. The documentation context changed only
  Javadocs/comments in the seven permitted production paths and the seven permitted documentation/
  planning paths; it did not modify tests or executable statements and did not rerun Java tests.

Documentation-agent validation: clean context
`/root/task_0018r_implementation/documentation_pass`, using the General, API/Javadoc, Planning,
and Example profiles.

- Independently read the repository instructions and architecture contract, current architecture
  index, documentation rules and selected profiles, planning guide and roadmap, model capability
  baseline and master plan, this task, the actual final diff, all seven affected production files,
  all four affected tests, Tensor and Compile API sections, glossary entries, and the unchanged
  Training API.
- Finalized Javadocs in all seven production paths. `SliceAttrs`, `SliceKind`, Tensor slice/flip,
  and the helper now document finite start/length/signed-step sequences, directional raw half-open
  normalization, the explicit raw `-1` coordinate distinction, checked length and final-coordinate
  arithmetic, canonical empty state, static selected axes, `Long.MIN_VALUE`, layout policy,
  ownership, failures, one producer, output index zero, and identifier effects. Window wording now
  states that public unfold/unfold2d/fold2d remain current and `FOLD_AXIS`/`FoldAxisAttrs` are
  compiler-only values for task 0023.
- Finalized `docs/api/tensor-api.md`, `docs/api/compile-api.md`, and `docs/glossary.md`, including
  complete positive, negative, empty, zero-extent, mixed-axis, and flip metadata examples and the
  current-versus-planned window boundary. The compiled Java 26 example printed the exact documented
  Shapes, attributes, layout-presence flags, producer count, output index, and input identity.
- `./gradlew :modules:model:javadoc` passed with `BUILD SUCCESSFUL`; two tasks executed. Generated
  pages for `SliceAttrs`, `SliceKind`, `Tensor`, `WindowTransformKind`, and `FoldAxisAttrs` contain
  the final signed-step and compiler-only wording.
- `javac --release 26 -cp modules/model/build/classes/java/main -d /tmp
  /tmp/SliceExpressionExample.java && java -cp modules/model/build/classes/java/main:/tmp
  SliceExpressionExample` passed and matched the documented output exactly.
- Targeted Markdown validation resolved all 480 local links and 137 GitHub-style fragments in all
  seven changed documentation/planning files; code fences were balanced. The only newly used external
  terminology sources are the official ONNX Slice and JAX flip pages selected by this task, and
  both returned HTTP 200.
- Generated-Javadoc, live public-surface, source, and terminology checks confirmed exactly four
  public slice/flip methods, 111 declared public Tensor methods, one `SLICE` kind, no `FLIP` kind,
  no public/helper `foldAxis`, retained four-constant window kind order, retained compiler-only
  `FoldAxisAttrs`, and unchanged one-input/one-output operation signatures.
- Exact-scope validation found exactly the authorized eighteen paths: seven production Java, four
  tests, and seven documentation/planning files. No architecture, ADR, architecture-test,
  conformance, integration, Gradle, dependency, build-configuration, other-module, completed-task,
  or later-task path changed. No commit or push occurred.
- Status validation found task 0018R `Complete` in this task, the model master plan, and roadmap;
  task 0018S and every later row remain `Draft`, and no detailed 0018S-or-later task exists.
- All eighteen paths are non-empty and end with a newline; no trailing whitespace or unbalanced
  backtick/tilde fence remains. `git diff --check` passed with no output; the untracked task file
  separately passed newline and whitespace checks.
- Reviewed unchanged public `Tensor.unfold`, `unfold2d`, and `fold2d` Javadocs and focused tests;
  no edit was needed because their signatures, validation, Shape/layout, metadata, provenance,
  and identifier behavior did not change. `UnfoldAxisAttrs`, `Window2dAttrs`, `Fold2dAttrs`, and
  retained semantic tests likewise remain accurate unchanged.
- Training API needs no change because this task adds no gradients, trainable state, optimizer, or
  training workflow. Shape/Dimension, LayoutDescriptor, operation signatures, Tensor producer/
  provenance/factory, focused architecture material, architecture tests, conformance/integration,
  Java 26 Gradle configuration, dependencies, other modules, and completed task history need no
  change because the implementation stays inside the existing model contract and changes none of
  those declarations or ownership boundaries.

## Implementation notes

Implementation replaced exclusive-end positive-only `SliceAttrs` with exact finite
start/length/signed-step sequences and updated every current construction path atomically. General
and one-axis slicing now normalize bounds by direction with checked lengths; negative-step results
remain layout-unresolved, while positive non-empty resolved inputs retain checked view geometry.
The public surface adds explicit-step `sliceAxis` and one-producer `flip(int... axes)`.

Public `Tensor.foldAxis` and its package-private helper construction were removed without an alias.
`WindowTransformKind.FOLD_AXIS`, `FoldAxisAttrs`, their exact signature, and semantic tests remain
for task 0023. Public unfold, unfold2d, and fold2d implementation and behavior were not changed.

## Completion summary

- Completed changes: finalized signed non-zero slice semantics, directional public normalization,
  step-aware one-axis slicing, one-SLICE flip, negative-step unresolved layout, and public
  foldAxis removal while retaining compiler-only FOLD_AXIS semantics.
- Files changed or created: exactly the seven production Java, four test, and seven documentation/
  planning paths listed under Affected files.
- Tests and validation: implementation-focused 78 tests and final 715-test/88-suite model run
  passed; documentation reused that stable evidence, then passed model Javadoc, the Java 26
  example, generated-page/public-surface, Markdown/link/anchor/fence/newline, official-URL,
  exact-scope/status/terminology, and `git diff --check` validation.
- Documentation-agent review: completed in clean context
  `/root/task_0018r_implementation/documentation_pass` with the General, API/Javadoc, Planning,
  and Example profiles.
- Documentation impact: Tensor API, Compile API, glossary, capability baseline, this task, model
  master plan, and roadmap now describe the final current contract and future ownership accurately.
- Javadoc review: all seven permitted production paths finalized; unchanged public
  unfold/unfold2d/fold2d method Javadocs remain accurate.
- Glossary impact: Slice, normalized axis, OperationAttrs, OperationKind, provenance, and window
  transform entries now match the final semantics and public/compiler-only boundary.
- Unresolved issues: None.
- Follow-up required: None. Task 0018S remains Draft; task 0023 retains compiler-generated
  FOLD_AXIS and backward-graph ownership.

Status: Complete
