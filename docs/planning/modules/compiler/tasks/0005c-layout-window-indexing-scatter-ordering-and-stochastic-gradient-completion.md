# Task 0005C: [Layout](../../../../glossary.md#layout), Window, Indexing, Scatter, Ordering, and Stochastic [Gradient](../../../../glossary.md#term-gradient) Completion

> **Editorial reading key — not part of the historical record.**
>
> Index, mask and saved-output roles are data dependencies even when they are not differentiation
> targets. Compiler routes cotangents using the exact original ordering, top-K indices or dropout
> mask associated with an occurrence; it cannot substitute an equal-looking producer. Preflight
> checks formula constructibility from descriptions, not actual index bounds or scatter duplicates
> in stored values. Constructing the inverse expression neither allocates a runtime tape nor
> grants backend support.
>
> Current reading: [pre-capture differentiation and functional requests](../../../../reference/compiler/autograd.md#current-package-private-pre-capture-autograd); [structural capture and exact leaf identities](../../../../reference/compiler/capture.md#current-package-private-structural-capture).

## Status

Complete

## Goal

Complete the first-order [compiler](../../../../glossary.md#term-compiler) [gradient](../../../../glossary.md#term-gradient) [matrix](../../../../glossary.md#term-matrix-meanings) for the remaining [layout](../../../../glossary.md#layout), [slice](../../../../glossary.md#slice), composition,
window, indexing, scatter, ordering, and explicit-state [dropout](../../../../glossary.md#dropout--inverted-dropout) [operation](../../../../glossary.md#operation) families.

The task extends the existing compiler-owned [pre-capture Tensor-expression autograd](../../../../glossary.md#pre-capture-tensor-expression-autograd) pipeline. It
uses ordinary public [`Tensor`](../../../../glossary.md#tensor) operations, exact original [producer](../../../../glossary.md#term-producer) occurrences, [canonical](../../../../glossary.md#term-canonical-form) auxiliary
outputs, and request-local logical splats. It does not add a second algebra, a backward-only
operation kind, [mutable](../../../../glossary.md#term-mutable) Tensor gradient state, or a public compiler gradient surface.

The central relationship is:

```text
exact original occurrence and selected floating role
  -> allocation-free fail-closed preflight
  -> ordinary public Tensor formula
  -> one combined phase-aware capture
```

Models 0025C and 0025D are Complete. They fix the forward represented-value meaning of
configurable scatter reductions and supply the two dynamic-slice construction forms that were
previously missing. No remaining [Model](../../../../glossary.md#term-model) or architecture prerequisite blocks this task.

## [Scope](../../../../glossary.md#term-scope)

### Source-backed [operation](../../../../glossary.md#operation) and role inventory

The inventory below is exhaustive over the assigned current source families and exact [signatures](../../../../glossary.md#term-signature).
`D` means a floating data role through which a [cotangent](../../../../glossary.md#cotangent) may propagate. [`ND`](../../../../glossary.md#term-nd) means intentionally
non-[differentiable](../../../../glossary.md#derivative-and-chain-rule).

| Exact kind and attributes | Ordered outputs | Ordered input roles | Task result |
|---|---|---|---|
| `CONTIGUOUS + NoOperationAttrs` | value slot 0 | input `D` | Preserve the implemented [identity](../../../../glossary.md#term-identity-meanings) cotangent. |
| `RESHAPE`, `EXPAND + TargetShapeAttrs` | value slot 0 | input `D` | Preserve the implemented reshape and [binding](../../../../glossary.md#term-binding)-aware [sum-to-Shape](../../../../glossary.md#sum-to-shape) cotangents. |
| `PERMUTE + PermutationAttrs` | value slot 0 | input `D` | Preserve the implemented inverse [permutation](../../../../glossary.md#permutation). |
| `EXPAND_DIMS`, `SQUEEZE + AxisTransformAttrs` | value slot 0 | input `D` | Preserve the implemented inverse [rank](../../../../glossary.md#term-rank) edit. |
| `SLICE + SliceAttrs` | value slot 0 | source `D` | Support every normalized [finite](../../../../glossary.md#term-finite) signed region, including regions created by `sliceByLength`. |
| `SLICE + CropToShapeAttrs` | value slot 0 | source `D` | Add [target](../../../../glossary.md#term-target)-relative dynamic placement through `sliceUpdate(Tensor, Shape)`. |
| `SLICE_UPDATE + SliceAttrs` | value slot 0 | base `D`, update `D` | Preserve the base formula and generalize update extraction through `sliceByLength`. |
| `SLICE_UPDATE + CropToShapeAttrs` | value slot 0 | base `D`, update `D` | Add exact target-relative base masking and update extraction. |
| `SELECT + SelectAttrs` | value slot 0 | source `D` | Preserve exact [singleton](../../../../glossary.md#term-singleton) placement, including unresolved selected extents. |
| `PAD + PadAttrs` | value slot 0 | source `D`; [constant](../../../../glossary.md#term-constant) [metadata](../../../../glossary.md#term-metadata) `ND` | Preserve [target-relative crop](../../../../glossary.md#target-relative-crop). |
| `TILE + TileAttrs` | value slot 0 | source `D`; repeats metadata `ND` | Preserve interleaved reshape/reduction. |
| `CONCAT`, `STACK + CompositionAxisAttrs` | value slot 0 | every ordered input position `D`; [axis](../../../../glossary.md#term-axis) metadata `ND` | Preserve position-specific crop/select contributions. |
| `UNFOLD_AXIS + UnfoldAxisAttrs` | value slot 0 | source `D` | Add `foldAxis` overlap-add [adjoint](../../../../glossary.md#term-adjoint). |
| `FOLD_AXIS + FoldAxisAttrs` | value slot 0 | window [tensor](../../../../glossary.md#tensor) `D` | Add matching `unfold` adjoint. |
| `UNFOLD2D + Window2dAttrs` | value slot 0 | [NCHW](../../../../glossary.md#term-nchw) source `D` | Add exact `fold2d` adjoint. |
| `UNFOLD2D + Unfold2dAttrs` | value slot 0 | NCHW source `D`; [padding](../../../../glossary.md#padding) [scalar](../../../../glossary.md#term-scalar) metadata `ND` | Add the same `fold2d` adjoint; padding is not a Tensor role. |
| `FOLD2D + Fold2dAttrs` | value slot 0 | [canonical](../../../../glossary.md#term-canonical-form) columns `D` | Add zero-padding `unfold2d` adjoint. |
| [`GATHER`](../../../../glossary.md#gather), `GATHER_ELEMENTS + IndexAxisAttrs` | value slot 0 | data `D`, indices `ND` | Add Gather-compatible or same-rank additive scatter. |
| `GATHER_ND + GatherNdAttrs` | value slot 0 | data `D`, indices `ND` | Add additive [Scatter-ND](../../../../glossary.md#scatter-nd). |
| `SCATTER_ADD + IndexAxisAttrs` | value slot 0 | data `D`, indices `ND`, updates `D` | Data identity plus matching Gather update route. |
| `SCATTER_ELEMENTS + ScatterElementsAttrs` for `NONE`, `ADD`, `MUL`, `MIN`, `MAX` | value slot 0 | data `D`, indices `ND`, updates `D` | Add the complete reduction-specific [matrix](../../../../glossary.md#term-matrix-meanings) below. |
| `SCATTER_ND + ScatterNdAttrs` for `NONE`, `ADD`, `MUL`, `MIN`, `MAX` | value slot 0 | data `D`, indices `ND`, updates `D` | Add the complete reduction-specific matrix below. |
| `ONE_HOT + OneHotAttrs` | [BOOL](../../../../glossary.md#data-type--datatype) slot 0 | integral indices `ND`; depth metadata `ND` | No differentiable role. |
| `SORT + SortAttrs` | values slot 0 | input `D` only for floating values | Add one exact matching stable `ARGSORT` routing formula. |
| `ARGSORT + SortAttrs` | [INT64](../../../../glossary.md#data-type--datatype) slot 0 | input `ND` | No differentiable role. |
| `TOP_K + TopKAttrs` | values slot 0, INT64 indices slot 1 | input `D` only from values slot 0 | Route through canonical slot-1 indices; indices output is `ND`. |
| `INITIAL_STATE + GraphRngStateAttrs` | INT64 state slot 0 | no inputs; key/counter metadata `ND` | No differentiable role. |
| `DROPOUT + DropoutAttrs` | value slot 0, BOOL [mask](../../../../glossary.md#term-boolean-mask) slot 1, INT64 next-state slot 2 | floating input `D` only from value slot 0; [RNG state](../../../../glossary.md#graph-rng-state--graphrngstate) `ND` | Reuse the canonical mask; mask and state outputs remain `ND`. |

Only [`FLOAT64`](../../../../glossary.md#term-float64), [`FLOAT32`](../../../../glossary.md#term-float32), and [`BFLOAT16`](../../../../glossary.md#term-bfloat16) data roles are differentiable. Integral [layout](../../../../glossary.md#layout),
gather/scatter, sort, or top-K values remain valid forward operations but have no first-order
cotangent route. BOOL values, [coordinates](../../../../glossary.md#axis-rank-scalar-vector-and-matrix), indices, one-hot values, keep masks, [graph RNG state](../../../../glossary.md#graph-rng-state--graphrngstate),
padding values, probabilities, axes, [Shapes](../../../../glossary.md#shape), lengths, steps, repeats, window geometry, selection
counts, direction flags, and reduction attributes are non-differentiable.

Conveniences add no rows: [`transpose`](../../../../glossary.md#term-transpose) is `PERMUTE`, `flip` is [`SLICE`](../../../../glossary.md#slice), [`embedding`](../../../../glossary.md#embedding) is
`GATHER(axis=0)`, and `unstack` is repeated independent `SELECT`.

### Notation and shared construction rules

For one selected [occurrence](../../../../glossary.md#term-occurrence):

- `g` is the accumulated [cotangent](../../../../glossary.md#cotangent) for the selected output slot;
- `x`, `data`, `indices`, `updates`, and `state` are exact original input wrappers;
- `y` is the exact [canonical](../../../../glossary.md#term-canonical-form) forward value output;
- `Z_t` and `O_t` are request-local exact typed positive-zero and positive-one splats expanded to
  `t.shape()`;
- `G_axis(v)` means the exact matching [`gather`](../../../../glossary.md#gather), `gatherElements`, or `gatherNd` operation for the
  original indexing geometry; and
- `S_axis(base, v, reduction)` means the exact matching `scatterAdd`, `scatterElements`, or
  `scatterNd` [operation](../../../../glossary.md#operation) with the original indices, [normalized axis](../../../../glossary.md#normalized-axis) or [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) count, and reduction.

Every formula must call existing public [`Tensor`](../../../../glossary.md#tensor) methods and [immutable](../../../../glossary.md#term-immutable) public [model](../../../../glossary.md#term-model) values. Rule
[owners](../../../../glossary.md#term-owner) must not construct `Operation`, [`TensorProducer`](../../../../glossary.md#producer--tensorproducer), `TensorProvenance`, [graph](../../../../glossary.md#graph) [nodes](../../../../glossary.md#node), graph
values, [descriptors](../../../../glossary.md#term-descriptor), or graph IDs directly.

All selected input and output data types are one exact floating type. No family in this task
requires mixed-floating cotangent conversion. Every formula must return the exact selected input
[Shape](../../../../glossary.md#shape) and type. Request-local [constants](../../../../glossary.md#term-constant) retain the existing exact [represented-bit](../../../../glossary.md#storage-mapping-contiguity-and-injectivity) [cache](../../../../glossary.md#term-cache) and
explicit `CompileTimeConstantGraph` ingress; Shape-specific constants are ordinary public
`expand` expressions.

### [Layout](../../../../glossary.md#layout), [slice](../../../../glossary.md#slice), selection, [padding](../../../../glossary.md#padding), tile, and composition formulas

Preserve the completed formulas for [`CONTIGUOUS`](../../../../glossary.md#term-contiguous), `RESHAPE`, [binding](../../../../glossary.md#term-binding)-aware `EXPAND`,
`EXPAND_DIMS`, `SQUEEZE`, `PERMUTE`, `SELECT`, [`PAD`](../../../../glossary.md#padding), `TILE`, `CONCAT`, and `STACK`.

The complete slice formulas become:

| Exact [occurrence](../../../../glossary.md#term-occurrence) and selected role | Formula |
|---|---|
| `SLICE + SliceAttrs(starts,lengths,axes,steps)`, source | `Z_x.sliceUpdate(g, starts, axes, steps)` |
| `SLICE + CropToShapeAttrs(target,prefix)`, source | `Z_x.sliceUpdate(g, prefix)` |
| `SLICE_UPDATE + SliceAttrs`, base | `g.sliceUpdate(Z_updates, starts, axes, steps)` |
| `SLICE_UPDATE + SliceAttrs`, update | `g.sliceByLength(starts, lengths, axes, steps)` |
| `SLICE_UPDATE + CropToShapeAttrs(target,prefix)`, base | `g.sliceUpdate(Z_updates, prefix)` |
| `SLICE_UPDATE + CropToShapeAttrs(target,prefix)`, update | `g.cropToShape(target, prefix)` |

`sliceByLength` removes the old static-selected-base restriction from the update-role formula.
The exact [finite](../../../../glossary.md#term-finite) lengths are already stored in `SliceAttrs`; no raw end is reconstructed.
`sliceUpdate(Tensor, Shape)` makes [target](../../../../glossary.md#term-target)-relative placement valid when the base, prefix, or update
[extent](../../../../glossary.md#term-extent) is unresolved. Empty regions remain exact empty placements/extractions. Signed non-zero
steps remain [injective](../../../../glossary.md#term-injective), so these formulas replace rather than add.

`SELECT` continues to use:

```text
Z_x.sliceUpdate(g.expandDims(axis), Shape prefix with prefix[axis] = index)
```

The implementation may retain its existing one-[axis](../../../../glossary.md#term-axis) `SliceAttrs` spelling only when that spelling
is valid for unresolved selected extents. The preferred unified formula uses target-relative
placement and must preserve the exact input [Shape](../../../../glossary.md#shape).

`PAD` remains `g.cropToShape(x.shape(), Shape.of(beforeWidths))`.

For `TILE`, reshape `g` to
`[repeat0,x0,repeat1,x1,...]`, sum the even repeat axes with [dimensions](../../../../glossary.md#dimension) retained, then reshape to
the exact `x.shape()`. [Scalar](../../../../glossary.md#term-scalar) tile returns `g`.

For `CONCAT` input position `i`, use
`g.cropToShape(input_i.shape(), prefix_i)`, where every non-concat prefix extent is zero and the
concat-axis prefix is the exact ordered symbolic sum of earlier input extents. For `STACK` input
position `i`, use `g.select(axis, i)`. Repeated exact [Tensor](../../../../glossary.md#tensor) inputs remain separate positional
contributions.

### Window formulas and dynamic geometry

Use the exact recorded attributes and [Shapes](../../../../glossary.md#shape):

| Kind | Formula |
|---|---|
| `UNFOLD_AXIS + UnfoldAxisAttrs(axis,size,step)` | `g.foldAxis(axis, static x.shape[axis], step)` |
| `FOLD_AXIS + FoldAxisAttrs(axis,outputSize,step)` | `g.unfold(axis, static input.shape[last], step)` |
| either `UNFOLD2D` attributes variant | `g.fold2d(x.shape(), window)` |
| `FOLD2D + Fold2dAttrs(outputShape,window)` | `g.unfold2d(window)` |

The `UNFOLD2D + Unfold2dAttrs` [padding](../../../../glossary.md#padding) value is non-[differentiable](../../../../glossary.md#derivative-and-chain-rule) [scalar](../../../../glossary.md#term-scalar) [metadata](../../../../glossary.md#term-metadata). Its source
[adjoint](../../../../glossary.md#term-adjoint) is still `fold2d`, because out-of-domain padding [samples](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) do not refer to source positions.
The `FOLD2D` adjoint uses the existing conceptual-positive-zero `unfold2d` form so column entries
that map outside the output receive [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero).

General-[axis](../../../../glossary.md#term-axis) unfold/fold retain their current statically known transformed extents. Unaffected
[Dimensions](../../../../glossary.md#dimension) may remain unresolved.

For both 2D directions, preserve exact symbolic [channel](../../../../glossary.md#term-channel) and [spatial](../../../../glossary.md#term-spatial-axes) formulas. [Inference](../../../../glossary.md#term-inference) must emit
an [occurrence](../../../../glossary.md#term-occurrence)-owned constraint for each height/width domain that cannot yet prove:

```text
inputOrTargetSpatial + 2 * padding >= dilation * (kernel - 1) + 1
```

The existing symbolic floor/ceiling and product expressions derive the exact output or expected
column Shape. A disproven domain fails inference; an undecidable domain remains a
`DeferredGraphConstraint`. No [binding](../../../../glossary.md#term-binding), unknown placeholder, or new predicate vocabulary is
needed.

### [Gather](../../../../glossary.md#gather) formulas

Indices are always non-[differentiable](../../../../glossary.md#derivative-and-chain-rule):

| Kind | Data [cotangent](../../../../glossary.md#cotangent) |
|---|---|
| `GATHER(data,indices,axis)` | `Z_data.scatterAdd(indices, g, axis)` |
| `GATHER_ELEMENTS(data,indices,axis)` | `Z_data.scatterElements(indices, g, axis, ADD)` |
| `GATHER_ND(data,indices,batchDimensions)` | `Z_data.scatterNd(indices, g, ADD, batchDimensions)` |

Repeated indices accumulate through the explicit `ADD` reduction. Dynamic gathered extents remain
representable because `SCATTER_ADD` has the exact Gather-compatible [Shape](../../../../glossary.md#shape) contract.

### Scatter `NONE` and `ADD` formulas

For `SCATTER_ADD`, the base [cotangent](../../../../glossary.md#cotangent) is `g` and the updates cotangent is matching `GATHER(g)`.

For `SCATTER_ELEMENTS` and `SCATTER_ND`:

| Reduction | Base cotangent | Updates cotangent |
|---|---|---|
| `NONE` | `S_axis(g, Z_updates, NONE)` | `G_axis(g)` |
| `ADD` | `g` | `G_axis(g)` |

Valid `NONE` occurrences require unique [targets](../../../../glossary.md#term-target) under the forward contract. [Preflight](../../../../glossary.md#term-preflight) does not
inspect index values or invent an encounter-order rule. Bounds and duplicate validity remain
forward execution obligations.

### Scatter `MUL` formula and numerical policy

For one configurable scatter [occurrence](../../../../glossary.md#term-occurrence), build update-group facts with the exact matching
scatter/[gather](../../../../glossary.md#gather) geometry:

```text
isZero             = (updates == Z_updates)
zeroIndicator      = where(isZero, O_updates, Z_updates)
safeUpdates        = where(isZero, O_updates, updates)

updateZeroCount    = S_axis(Z_data, zeroIndicator, ADD)
updateSafeProduct  = S_axis(O_data, safeUpdates, MUL)
allUpdateProduct   = S_axis(O_data, updates, MUL)

countAtUpdate      = G_axis(updateZeroCount)
safeProductAtUpdate= G_axis(updateSafeProduct)
gAtUpdate          = G_axis(g)
dataAtUpdate       = G_axis(data)
safeDenominator    = where(isZero, O_updates, updates)

regular            = gAtUpdate * dataAtUpdate
                     * (safeProductAtUpdate / safeDenominator)
soleZero           = gAtUpdate * dataAtUpdate * safeProductAtUpdate

dData              = g * allUpdateProduct
dUpdates           = where(
                         countAtUpdate == Z_updates,
                         regular,
                         where(
                             (countAtUpdate == O_updates) AND isZero,
                             soleZero,
                             Z_updates))
```

The base participates exactly once in every update [derivative](../../../../glossary.md#term-derivative) through `dataAtUpdate`. Duplicate
updates remain distinct and each receives its own gathered result. An unaddressed base [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix)
uses the exact multiplicative [identity](../../../../glossary.md#term-identity-meanings) from `O_data`, so its base [cotangent](../../../../glossary.md#cotangent) is `g`.

Both signed zeros compare equal to [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero) for zero counting. The safe denominator is never
zero. With one update zero, only that update receives the product of the base and all non-zero
updates. With more than one update zero, every update contribution is selected as exact positive
zero. The base cotangent always uses the [Model](../../../../glossary.md#term-model) 0025C product of all updates, including its [NaN](../../../../glossary.md#term-nan),
zero-times-[infinity](../../../../glossary.md#term-infinity), sign, [overflow](../../../../glossary.md#term-overflow), [underflow](../../../../glossary.md#term-underflow), and [rounding](../../../../glossary.md#term-rounding) behavior.

NaN updates are not zero and make the safe product NaN. A zero-free NaN group therefore routes NaN
through ordinary formula arithmetic. A sole zero beside a NaN or infinity routes the Model
non-zero product to that zero update. The selected multiple-zero branch is positive zero even if
another update is NaN or infinite. This is the explicit [compiler](../../../../glossary.md#term-compiler) first-order convention at that
non-regular boundary; it does not change the forward product.

### Scatter `MIN` and `MAX` formula and numerical policy

Use the exact [canonical](../../../../glossary.md#term-canonical-form) forward output `y`; do not recompute the [extremum](../../../../glossary.md#term-extremum):

```text
baseMatches       = (data == y)
outputAtUpdate    = G_axis(y)
updateMatches     = (updates == outputAtUpdate)
baseIndicator     = where(baseMatches, O_data, Z_data)
updateIndicator   = where(updateMatches, O_updates, Z_updates)
tieCount          = S_axis(baseIndicator, updateIndicator, ADD)
hasWinner         = tieCount > Z_data
safeTieCount      = where(hasWinner, tieCount, O_data)
shared            = where(hasWinner, g / safeTieCount, Z_data)

dData             = where(baseMatches, shared, Z_data)
dUpdates          = where(updateMatches, G_axis(shared), Z_updates)
```

The base and every duplicate update are distinct candidates. All exact represented-numeric
winners share the [cotangent](../../../../glossary.md#cotangent) equally. Numeric equality makes opposite signed zeros ties even though
[Model](../../../../glossary.md#term-model) 0025C selects [negative zero](../../../../glossary.md#nan-infinity-and-signed-zero) for `MIN` and [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero) for `MAX`. Equal [finite](../../../../glossary.md#term-finite) values and
same-sign [infinities](../../../../glossary.md#nan-infinity-and-signed-zero) also share. Non-winners receive positive zero.

If the forward result is [NaN](../../../../glossary.md#term-nan), numeric equality is false for every candidate. `hasWinner` is false
and every candidate receives exact positive zero, matching the completed reduction-[extrema](../../../../glossary.md#saturation-pole-and-extremum)
[compiler](../../../../glossary.md#term-compiler) convention. The safe denominator prevents division by zero in the selected formula.

### Ordering formulas and boundary policy

`ARGSORT` has no [differentiable](../../../../glossary.md#derivative-and-chain-rule) result.

For each selected one-output floating `SORT(x, SortAttrs(axis,descending))` [occurrence](../../../../glossary.md#term-occurrence), construct
exactly one ordinary public:

```text
indices = x.argsort(axis, descending)
dX = Z_x.scatterElements(indices, g, axis, NONE)
```

Before any [Tensor](../../../../glossary.md#tensor) allocation, [preflight](../../../../glossary.md#term-preflight) must validate:

- exact `OrderingKind.SORT`;
- exact `SortAttrs`;
- one input and one output;
- exact equal floating input/output type and [Shape](../../../../glossary.md#shape);
- [normalized axis](../../../../glossary.md#normalized-axis) for the exact input [rank](../../../../glossary.md#term-rank);
- the exact direction flag;
- stable, [NaN](../../../../glossary.md#term-nan)-last ordering from the current [Model](../../../../glossary.md#term-model) contract; and
- constructibility of one matching `ARGSORT + SortAttrs`, one input, one [INT64](../../../../glossary.md#data-type--datatype) output occurrence.

This is the sole permitted matching recomputation in this task. It is one separate, stable,
single-output `ARGSORT` Tensor occurrence per selected `SORT` occurrence. It is not a hidden SORT
output, a public `SortResult`, a [producer](../../../../glossary.md#term-producer) sibling [lookup](../../../../glossary.md#term-lookup), a producer reconstruction, or a Model [API](../../../../glossary.md#term-application-programming-interface-api)
change.

The [derivative](../../../../glossary.md#term-derivative) freezes the exact current stable [permutation](../../../../glossary.md#permutation). Equal-value and same-NaN classes use
their stable increasing logical input-index order; opposite signed zeros use the Model total order.
No [cotangent](../../../../glossary.md#cotangent) averaging occurs across equal keys. At crossings, NaN transitions, and other ordering
discontinuities, the selected first-order convention remains this exact routed permutation.

For `TOP_K`, only values slot 0 is differentiable:

```text
indices = producer.output(1)
dX = Z_x.scatterElements(indices, g, axis, NONE)
```

Preflight requires the exact [canonical](../../../../glossary.md#term-canonical-form) wrapper at output slot 1, exact producer [identity](../../../../glossary.md#term-identity-meanings), exact
INT64 type, exact values Shape, non-[gradient](../../../../glossary.md#term-gradient) eligibility, and matching `TopKAttrs`. Selected
indices are unique. `k == 0` returns `Z_x`. At equal cutoffs, only the stable selected members
receive cotangents; excluded equal values receive zero. NaN membership and `sorted` output order
are routed through the same canonical slot-1 indices. No selected-set averaging or recomputation
is allowed.

### Explicit-state [dropout](../../../../glossary.md#dropout--inverted-dropout) formula and probability policy

Only dropout value slot 0 may propagate to floating input position 0:

```text
mask = producer.output(1)
dInput = where(mask, g.div(1.0d - probability), Z_input)
```

[Preflight](../../../../glossary.md#term-preflight) requires exact `DropoutKind.DROPOUT`, `DropoutAttrs`, inputs `[input,state]`, outputs
`[value,mask,nextState]`, exact [canonical](../../../../glossary.md#term-canonical-form) slot-1 wrapper, matching [Shape](../../../../glossary.md#shape), [BOOL](../../../../glossary.md#data-type--datatype) [mask](../../../../glossary.md#term-boolean-mask) type,
non-[gradient](../../../../glossary.md#term-gradient) mask eligibility, and valid [INT64](../../../../glossary.md#data-type--datatype) `Shape[2]` state [descriptors](../../../../glossary.md#term-descriptor) at input 1 and output
2. The probability must remain [finite](../../../../glossary.md#term-finite) and numerically in `[0,1)`.

The formula reuses the exact forward keep mask. It must not infer a mask from the output, resample,
construct another dropout [occurrence](../../../../glossary.md#term-occurrence), advance or replace [RNG state](../../../../glossary.md#graph-rng-state--graphrngstate), or differentiate either state
edge. `where`, rather than mask multiplication, preserves exact [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero) for dropped [NaN](../../../../glossary.md#term-nan) or
[infinity](../../../../glossary.md#term-infinity) inputs. At either signed probability zero, the denominator is positive one and the
canonical all-kept mask routes `g`; the occurrence and state transition remain present. There is
no probability-one case because the [Model](../../../../glossary.md#term-model) rejects it.

`INITIAL_STATE`, dropout state input/output, and dropout mask output remain non-[differentiable](../../../../glossary.md#derivative-and-chain-rule).

### [Inference](../../../../glossary.md#term-inference) [normalization](../../../../glossary.md#term-normalization) and retained constraints

Inference must independently derive the current exact output [descriptors](../../../../glossary.md#term-descriptor) and preserve the existing
[Shape](../../../../glossary.md#shape)/type normalization for every assigned family.

The [slice](../../../../glossary.md#slice)-family rules are exact:

- `SLICE + SliceAttrs` derives selected static lengths and emits no constraint for a zero-length
  entry; a non-empty signed entry retains the upper-bound obligation
  `max(start,last)+1 <= sourceExtent`;
- `SLICE_UPDATE + SliceAttrs` additionally requires update Shape equal to the derived region and
  returns the exact base Shape with base/update gradient-eligibility OR;<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient))</span>
- `SLICE + CropToShapeAttrs` returns exact `targetShape` and retains
  `prefix + target <= source` for every [axis](../../../../glossary.md#term-axis);
- `SLICE_UPDATE + CropToShapeAttrs` requires update Shape equal to exact `targetShape`, returns
  exact base Shape and type with base/update eligibility OR, and retains
  `prefix + target <= base` for every axis.

This corrects the current `LayoutInference` branch that treats both `CropToShapeAttrs` kinds as
one-input extraction. It must not change [Model](../../../../glossary.md#term-model) construction.

Use checked arithmetic for [finite](../../../../glossary.md#term-finite) final [coordinates](../../../../glossary.md#axis-rank-scalar-vector-and-matrix), `maxCoordinate + 1`, effective [kernels](../../../../glossary.md#kernel),
[padding](../../../../glossary.md#padding) offsets, static counts, and interleaved Shapes. A statically contradicted constraint fails.
An unresolved but validly representable relation remains ordered [occurrence](../../../../glossary.md#term-occurrence)-local [compiler](../../../../glossary.md#term-compiler) state.
Do not replace a dynamic [Dimension](../../../../glossary.md#dimension) with a static guess or unnamed unknown.

Index-value bounds and `NONE` duplicate uniqueness cannot be proven from descriptors and are not
invented as Shape constraints. They remain validity obligations of eventual execution.

### [Fail-closed](../../../../glossary.md#term-fail-closed) [preflight](../../../../glossary.md#term-preflight) and failure order

Preserve the top-level [Compiler](../../../../glossary.md#term-compiler) 0004–0005B [validation](../../../../glossary.md#term-validation) order. Before constructing a [seed](../../../../glossary.md#term-seed), [logical
splat](../../../../glossary.md#logical-splat), matching ARGSORT, formula [Tensor](../../../../glossary.md#tensor), or any [derivative](../../../../glossary.md#term-derivative) [`TensorId`](../../../../glossary.md#tensorid), visit all selected
occurrences in deterministic [producer](../../../../glossary.md#term-producer) postorder and validate:

1. exact kind and exact attributes class/value;
2. exact input and output [cardinality](../../../../glossary.md#term-cardinality);
3. selected [canonical](../../../../glossary.md#term-canonical-form) output slot and output-specific differentiability;
4. input roles in ascending position order, including every [`ND`](../../../../glossary.md#term-nd) index/state role;
5. exact floating type and [gradient](../../../../glossary.md#term-gradient) eligibility of each selected data role;
6. independently re-derived [descriptors](../../../../glossary.md#term-descriptor) and every non-disproven [occurrence](../../../../glossary.md#term-occurrence) constraint;
7. normalized axis, batch count, tuple depth, Shape, slice region, window geometry, reduction, K,
   probability, and stable-order facts;<span data-glossary-navigation="true"> ([normalized axis glossary](../../../../glossary.md#normalized-axis); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [tuple depth glossary](../../../../glossary.md#term-tuple-depth); [Shape glossary](../../../../glossary.md#shape); [slice glossary](../../../../glossary.md#slice))</span>
8. exact canonical auxiliary output [identity](../../../../glossary.md#term-identity-meanings) for TOP_K and [DROPOUT](../../../../glossary.md#dropout--inverted-dropout);
9. constructibility of the sole matching ARGSORT occurrence for SORT; and
10. the exact numerical/boundary policy row above.

Preflight must not inspect Tensor payloads, index values, [masks](../../../../glossary.md#term-boolean-mask), storage, labels, [layouts](../../../../glossary.md#layout) for
value [inference](../../../../glossary.md#term-inference), or [backend](../../../../glossary.md#backend) behavior. Unknown kinds, wrong attributes, wrong cardinality, wrong
output slot, missing canonical outputs, descriptor contradictions, unsupported roles, or
disproved constraints fail deterministically.

No known failure may consume a seed, [constant](../../../../glossary.md#term-constant), formula Tensor, matching ARGSORT Tensor, or
derivative ID. After successful complete preflight, ordinary public Tensor construction, [capture](../../../../glossary.md#term-capture-meanings),
inference, validation, optimization, [publication](../../../../glossary.md#term-publication), or [planning](../../../../glossary.md#term-planning) failure may consume [opaque](../../../../glossary.md#term-opaque)
non-reusable [model](../../../../glossary.md#term-model) IDs under the existing [architecture contract](../../../../glossary.md#architecture-contract).

### Determinism, [producers](../../../../glossary.md#term-producer), outputs, [provenance](../../../../glossary.md#provenance), and [constants](../../../../glossary.md#term-constant)

- Preserve exact [Tensor](../../../../glossary.md#tensor) and [`TensorProducer`](../../../../glossary.md#producer--tensorproducer) identity in inventory, route, output-slot,
  contribution, and accumulation bookkeeping.
- Reverse accumulation order remains reverse producer postorder, ascending selected output slot,
  then ascending input position. Contributions are left-associated through ordinary `Tensor.add`.
- Repeated input positions and duplicate scatter updates remain logically distinct.
- Use `producer.output(1)` for TOP_K indices and [DROPOUT](../../../../glossary.md#dropout--inverted-dropout) [mask](../../../../glossary.md#term-boolean-mask). Do not reconstruct a wrapper.
- Original [producer](../../../../glossary.md#term-producer) occurrences and every original output slot remain `GraphPhase.FORWARD`.
- The permitted matching ARGSORT and every other generated formula producer are
  `GraphPhase.BACKWARD`.
- [Capture](../../../../glossary.md#term-capture-meanings) ordered forward outputs and [gradient](../../../../glossary.md#term-gradient) roots together exactly once. Assign every [`NodeId`](../../../../glossary.md#nodeid)
  and [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) once. Per-[node](../../../../glossary.md#node) phase remains authoritative.
- Distinct [target](../../../../glossary.md#term-target) roles may share one captured gradient value; add no [identity](../../../../glossary.md#term-identity-meanings) node.
- Reuse only reachable request-local exact typed logical splats. Storage, labels, descriptor
  Shape, factory history, or absent provenance never imply constant status.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [Shape glossary](../../../../glossary.md#shape))</span>
- Run the existing combined inference, validation, canonicalization, exact rewriting, constant
  folding, whole-graph dead-code elimination, phase-local common-subexpression elimination,
  cleanup, final validation, publication, and planning pipeline unchanged.<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run); [inference glossary](../../../../glossary.md#term-inference); [validation glossary](../../../../glossary.md#term-validation); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [constant folding glossary](../../../../glossary.md#term-constant-folding); [graph glossary](../../../../glossary.md#graph); [dead-code elimination glossary](../../../../glossary.md#term-dead-code-elimination-dce); [common-subexpression elimination glossary](../../../../glossary.md#term-common-subexpression-elimination-cse); [publication glossary](../../../../glossary.md#term-publication); [planning glossary](../../../../glossary.md#term-planning))</span>

### Transitive formula-[operation](../../../../glossary.md#operation) closure

The formulas use only current public operations already covered by [Compiler](../../../../glossary.md#term-compiler) 0005A–0005B or
covered in this task:

- `ADD`, `MUL`, `DIV`, comparisons, [Boolean](../../../../glossary.md#data-type--datatype) `AND`, and `WHERE`;
- exact typed logical-splat expansion;
- `RESHAPE`, `EXPAND_DIMS`, `SQUEEZE`, [`SLICE`](../../../../glossary.md#slice), `SLICE_UPDATE`, `SELECT`, and [target-relative crop](../../../../glossary.md#target-relative-crop);
- reductions used by the retained TILE formula;
- [Gather](../../../../glossary.md#gather), [Gather Elements](../../../../glossary.md#gather-elements), [Gather-ND](../../../../glossary.md#gather-nd), [Scatter Add](../../../../glossary.md#scatter-add), [Scatter Elements](../../../../glossary.md#scatter-elements), and [Scatter-ND](../../../../glossary.md#scatter-nd);
- UNFOLD/FOLD [axis](../../../../glossary.md#term-axis) and 2D window transformations; and
- the explicitly approved matching ARGSORT.

No formula requires a new operation kind, public method, cast policy, [mutable](../../../../glossary.md#term-mutable) [gradient](../../../../glossary.md#term-gradient), or captured
[IR](../../../../glossary.md#term-intermediate-representation-ir) algebra. Task 0005E still owns the final source-backed transitive-closure [checkpoint](../../../../glossary.md#term-checkpoint) before
higher-order requests become [Ready](../../../../glossary.md#term-ready).

## Out of [scope](../../../../glossary.md#term-scope)

- any Model source, test, operation, attributes, Tensor method, result carrier, producer-output,
  descriptor, Shape, DataType, or public API change<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [operation glossary](../../../../glossary.md#operation); [carrier glossary](../../../../glossary.md#term-storage-carrier); [producer glossary](../../../../glossary.md#term-producer); [descriptor glossary](../../../../glossary.md#term-descriptor); [Shape glossary](../../../../glossary.md#shape); [DataType glossary](../../../../glossary.md#data-type--datatype); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- a hidden SORT indices output, multi-output SORT, public sort result, or any recomputation other
  than the exact matching stable ARGSORT formula specified above
- recomputing TOP_K indices, [dropout](../../../../glossary.md#dropout--inverted-dropout) [masks](../../../../glossary.md#term-boolean-mask), scatter [extrema](../../../../glossary.md#saturation-pole-and-extremum), or another [canonical](../../../../glossary.md#term-canonical-form) auxiliary/output
- [derivative](../../../../glossary.md#term-derivative) rows owned by [Compiler](../../../../glossary.md#term-compiler) 0005D: attention, [convolution](../../../../glossary.md#term-channels-first-convolution-layers), [pooling](../../../../glossary.md#term-pooling), and [losses](../../../../glossary.md#loss)
- the Compiler 0005E source-backed closure [checkpoint](../../../../glossary.md#term-checkpoint)
- explicit objectives, seeds, Jacobians, create-graph, derivative order, disconnected-target
  policy, or other Compiler 0006 work<span data-glossary-navigation="true"> ([seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [graph glossary](../../../../glossary.md#graph); [target glossary](../../../../glossary.md#term-target))</span>
- [mutable](../../../../glossary.md#term-mutable) [Tensor](../../../../glossary.md#tensor) [gradients](../../../../glossary.md#term-gradient), `Tensor.backward`, a tape, backward-only kind, captured-value-to-Tensor
  conversion, second algebra, public registry/facade/policy, reflection, service loading, string
  dispatch, or global cache<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [facade glossary](../../../../glossary.md#term-facade); [dispatch glossary](../../../../glossary.md#term-dispatch); [cache glossary](../../../../glossary.md#term-cache))</span>
- new [graph optimization](../../../../glossary.md#forward-graph-optimization), algebraic rewrite, folding rule, [CSE](../../../../glossary.md#term-common-subexpression-elimination-cse) policy, phase rule, or ID [lifecycle](../../../../glossary.md#lifecycle)
- physical saved buffers, materialization, backend/runtime state, sampling, lowering, execution,
  kernel, algorithm, route, conformance, or integration behavior<span data-glossary-navigation="true"> ([buffers glossary](../../../../glossary.md#term-buffer); [materialization glossary](../../../../glossary.md#materialization); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime); [sampling glossary](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters); [lowering glossary](../../../../glossary.md#lowering); [kernel glossary](../../../../glossary.md#kernel); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Config, Planning, Runtime, Prepare, Engine, backend, training, trace, ONNX, architecture, ADR,
  dependency, Gradle, Java-version, or build changes<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [trace glossary](../../../../glossary.md#trace); [ONNX glossary](../../../../glossary.md#term-onnx); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- detailed task specifications or promotion for Compiler 0005D, 0005E, or 0006
- unrelated refactoring or documentation cleanup

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>, especially model ownership,
  compiler-owned pre-capture autograd, canonical producer outputs, one combined capture, phase,
  constant, and forbidden-gradient-surface rules<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [capture glossary](../../../../glossary.md#term-capture-meanings); [autograd glossary](../../../../glossary.md#autograd); [canonical glossary](../../../../glossary.md#term-canonical-form); [producer glossary](../../../../glossary.md#term-producer); [constant glossary](../../../../glossary.md#term-constant); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [General documentation style](../../../../developer-guide/documentation/general-style.md)
- [Planning documentation style](../../../../developer-guide/documentation/planning-style.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Compiler master plan](../master-plan.md)
- [Model master plan](../../model/master-plan.md)
- [Model capabilities](../../model/capabilities.md)
- [Adjoint expressibility audit](../../model/adjoint-expressibility-audit.md)<span data-glossary-navigation="true"> ([Adjoint glossary](../../../../glossary.md#term-adjoint))</span>
- [Model 0025 canonical outputs](../../model/tasks/0025-canonical-tensor-producer-outputs.md)
- [Model 0025C scatter semantics](../../model/tasks/0025c-portable-functional-scatter-reduction-semantics.md)
- [Model 0025D dynamic slice forms](../../model/tasks/0025d-dynamic-extent-slice-extraction-and-symbolic-slice-placement.md)<span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice))</span>
- completed [Compiler](../../../../glossary.md#term-compiler) tasks [0004](0004-compiler-owned-pre-capture-autograd-and-combined-graph-compilation.md),
  [0004A](0004a-exact-composition-gradient-rule-extensions.md),
  [0004B](0004b-shared-algebra-cotangent-normalization-and-local-derivative-rules.md),
  [0005A](0005a-derivative-policy-and-elementwise-activation-gradient-completion.md), and
  [0005B](0005b-reduction-scan-softmax-statistics-and-normalization-gradient-completion.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `modules/compiler` owns fail-closed preflight, named derivative-rule dispatch, reverse
  accumulation, inference constraints, combined capture, and compile artifacts.<span data-glossary-navigation="true"> ([fail-closed glossary](../../../../glossary.md#term-fail-closed); [dispatch glossary](../../../../glossary.md#term-dispatch); [inference glossary](../../../../glossary.md#term-inference); [compile artifacts glossary](../../../../glossary.md#compile-artifacts))</span>
- `modules/model` remains [derivative](../../../../glossary.md#term-derivative)-agnostic and unchanged.
- Formulas are ordinary public Tensor expressions. They are not a second graph, low-level IR, or
  model-owned derivative language.<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- Exact original [producer](../../../../glossary.md#term-producer) and [canonical](../../../../glossary.md#term-canonical-form) output wrappers are the only saved-value [identity](../../../../glossary.md#term-identity-meanings) source.
- [Compiler](../../../../glossary.md#term-compiler) identity maps remain request-local and ephemeral.
- One complete [preflight](../../../../glossary.md#term-preflight) precedes any generated [Tensor](../../../../glossary.md#tensor) allocation.
- One combined phase-aware [capture](../../../../glossary.md#term-capture-meanings) assigns [graph](../../../../glossary.md#graph)-local IDs once.
- Compiler creates no physical buffer, saved-tensor store, runtime tape, prepared state, backend
  executable, kernel, or route.<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [kernel glossary](../../../../glossary.md#kernel))</span>
- No [dependency](../../../../glossary.md#term-dependency) or [module](../../../../glossary.md#term-module) boundary changes are authorized.

If implementation requires a public [mask](../../../../glossary.md#term-boolean-mask)/index surface, another Tensor [operation](../../../../glossary.md#operation), a [Model](../../../../glossary.md#term-model) change,
an architecture change, or a formula outside the closed [matrix](../../../../glossary.md#term-matrix-meanings) above, stop and report the
conflict.

## Package impact

Existing package used:

- `io.github.pho001.synaptik.compiler` — the cohesive package-private compiler front-end,
  inference, preflight, named rule owners, reverse accumulation, and tests.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [inference glossary](../../../../glossary.md#term-inference); [preflight glossary](../../../../glossary.md#term-preflight))</span>

Packages added, moved, renamed, or made public: None.

Type placement:

- `LayoutInference` continues to own [layout](../../../../glossary.md#layout)/[slice](../../../../glossary.md#slice)/window [descriptor](../../../../glossary.md#term-descriptor) derivation and constraints.
- `LayoutGradientRules` continues to own layout, slice, selection, padding, tile, composition, and
  now window formulas.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding))</span>
- new package-private `IndexingGradientRules` owns Gather and scatter formulas because those roles
  form one exact indexing geometry boundary.<span data-glossary-navigation="true"> ([Gather glossary](../../../../glossary.md#gather))</span>
- new package-private `OrderingGradientRules` owns SORT and TOP_K routing because it alone needs
  stable permutation/selection policy.<span data-glossary-navigation="true"> ([permutation glossary](../../../../glossary.md#permutation))</span>
- new package-private `StochasticGradientRules` owns DROPOUT mask routing because RNG-state roles
  and probability scaling are distinct from deterministic indexing.<span data-glossary-navigation="true"> ([DROPOUT glossary](../../../../glossary.md#dropout--inverted-dropout); [mask glossary](../../../../glossary.md#term-boolean-mask); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
- `AutogradPreflight` remains the sole complete support selector.
- `FirstOrderAutograd` remains the sole reverse traversal, contribution accumulation, constant
  ownership, and named-rule dispatch owner.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [dispatch glossary](../../../../glossary.md#term-dispatch); [owner glossary](../../../../glossary.md#term-owner))</span>

The three new rule owners are field-free, package-private, final, and stateless. Do not add a
generic `GradientManager`, `GradientUtil`, [registry](../../../../glossary.md#term-registry), [facade](../../../../glossary.md#term-facade), service, policy object, or subpackage.

## Affected files

Expected production Java:

- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/LayoutInference.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/LayoutGradientRules.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/IndexingGradientRules.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/OrderingGradientRules.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/StochasticGradientRules.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/AutogradPreflight.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/FirstOrderAutograd.java`

Expected [compiler](../../../../glossary.md#term-compiler) tests:

- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/LayoutInferenceTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/AutogradPreflightTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/GraphCompilerTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/LayoutWindowGradientRulesTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/IndexingScatterGradientRulesTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/OrderingStochasticGradientRulesTest.java`

Expected documentation and [planning](../../../../glossary.md#term-planning):

- `docs/api/compile-api.md`
- `docs/api/tensor-api.md`, authorized after implementation solely to correct four stale
  Compiler-0005C current-status statements found by the independent documentation pass
- `docs/glossary.md`, only if the documentation pass finds an existing term whose compiler meaning
  must be clarified
- this task specification
- `docs/planning/modules/compiler/master-plan.md`
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless a contradiction requires stopping:

- [Model](../../../../glossary.md#term-model) source/tests and completed Model task evidence
- `docs/api/training-api.md`
- model capabilities and both [adjoint](../../../../glossary.md#term-adjoint)/closure audits
- compile capture, graph-ID, phase, constants, optimization, publication, and planning code/tests
  outside the listed owners<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [capture glossary](../../../../glossary.md#term-capture-meanings); [graph glossary](../../../../glossary.md#graph); [constants glossary](../../../../glossary.md#term-constant); [publication glossary](../../../../glossary.md#term-publication))</span>
- architecture documents/tests, Gradle/build files, dependencies, other modules, backend
  conformance, and integration tests<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most 23 paths:

- at most nine [compiler](../../../../glossary.md#term-compiler) production paths;
- at most seven compiler test paths; and
- at most seven documentation/[planning](../../../../glossary.md#term-planning) paths, including this task.

The final expected set is seven production paths, six test paths, and seven
documentation/planning paths. The original 22-path ceiling was explicitly expanded by one path
for the four necessary status-only corrections in `docs/api/tensor-api.md`. Any [Model](../../../../glossary.md#term-model) source/test
path, another [module](../../../../glossary.md#term-module), build/[dependency](../../../../glossary.md#term-dependency) path, architecture/[ADR](../../../../glossary.md#term-architecture-decision-record-adr) path, [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance) path,
integration-test path, or twenty-fourth touched path requires stopping and proposing a separately
reviewed follow-up.

## Acceptance criteria

### Inventory and [preflight](../../../../glossary.md#term-preflight)

- The exact assigned source inventory above is encoded in role-aware [fail-closed](../../../../glossary.md#term-fail-closed) preflight.
- Every selected occurrence validates exact kind, attributes, signature, output slot, input role,
  type, Shape, constraints, auxiliary identity, and policy before generated Tensor allocation.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [signature glossary](../../../../glossary.md#term-signature); [Shape glossary](../../../../glossary.md#shape); [identity glossary](../../../../glossary.md#term-identity-meanings); [Tensor glossary](../../../../glossary.md#tensor))</span>
- BOOL, integral index/coordinate/one-hot, RNG-state, mask, and configuration roles remain
  non-differentiable.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [mask glossary](../../../../glossary.md#term-boolean-mask); [differentiable glossary](../../../../glossary.md#derivative-and-chain-rule))</span>
- Unknown/custom kinds and every unsupported exact variant still fail before partial backward
  construction.
- `SLICE_UPDATE + CropToShapeAttrs` inference returns base Shape and validates the update against
  the exact target Shape.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [target glossary](../../../../glossary.md#term-target))</span>
- Dynamic slice/crop and 2D-window obligations are retained or disproved exactly; no static guess
  or binding is introduced.<span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice); [binding glossary](../../../../glossary.md#term-binding))</span>

### Formula behavior

- All [layout](../../../../glossary.md#layout)/[slice](../../../../glossary.md#slice)/selection/pad/tile/composition baseline rows remain supported.
- `sliceByLength` handles signed, strided, empty, and unresolved-base update extraction without raw
  end reconstruction.
- [target](../../../../glossary.md#term-target)-relative slice extraction/update formulas use exact [Shapes](../../../../glossary.md#shape) and prefixes.
- all four window kinds use the exact public inverse/adjoint operation and preserve dynamic
  unaffected/2D Dimensions.<span data-glossary-navigation="true"> ([adjoint glossary](../../../../glossary.md#term-adjoint); [operation glossary](../../../../glossary.md#operation); [Dimensions glossary](../../../../glossary.md#dimension))</span>
- Gather, Gather Elements, and Gather-ND accumulate duplicate-index contributions through their
  exact additive scatter.<span data-glossary-navigation="true"> ([Gather glossary](../../../../glossary.md#gather); [Gather Elements glossary](../../../../glossary.md#gather-elements); [Gather-ND glossary](../../../../glossary.md#gather-nd))</span>
- scatter `NONE`, `ADD`, `MUL`, `MIN`, and `MAX` implement both floating data roles and the fixed
  duplicate/zero/tie/NaN/signed-zero policies above.<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- SORT constructs exactly one matching stable ARGSORT [occurrence](../../../../glossary.md#term-occurrence) and routes through it.
- TOP_K uses exact [canonical](../../../../glossary.md#term-canonical-form) indices output slot 1 and never recomputes selection.
- [DROPOUT](../../../../glossary.md#dropout--inverted-dropout) uses exact canonical [mask](../../../../glossary.md#term-boolean-mask) output slot 1 and never resamples or advances state.
- Every returned [cotangent](../../../../glossary.md#cotangent) has the exact selected input floating type and Shape.

### Shared pipeline and boundaries

- Reverse accumulation order, Tensor identity handling, constant ingress, one combined capture,
  per-node phase, graph-ID assignment, optimization, publication, and planning remain unchanged.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [identity glossary](../../../../glossary.md#term-identity-meanings); [constant glossary](../../../../glossary.md#term-constant); [capture glossary](../../../../glossary.md#term-capture-meanings); [node glossary](../../../../glossary.md#node); [graph glossary](../../../../glossary.md#graph); [publication glossary](../../../../glossary.md#term-publication); [planning glossary](../../../../glossary.md#term-planning))</span>
- Original auxiliary-output producers remain FORWARD; all generated formula producers, including
  matching ARGSORT, are BACKWARD.<span data-glossary-navigation="true"> ([producers glossary](../../../../glossary.md#term-producer))</span>
- No public type/member, Model behavior, compiler facade/registry/policy, gradient-only operation,
  second algebra, mutable Tensor gradient, tape, physical saved buffer, or backend/runtime behavior
  is added.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [facade glossary](../../../../glossary.md#term-facade); [registry glossary](../../../../glossary.md#term-registry); [gradient glossary](../../../../glossary.md#term-gradient); [operation glossary](../../../../glossary.md#operation); [mutable glossary](../../../../glossary.md#term-mutable); [buffer glossary](../../../../glossary.md#term-buffer); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- [Compiler](../../../../glossary.md#term-compiler) 0005D, 0005E, and 0006 remain Draft without detailed specifications.
- A separate documentation-focused clean-context pass finalizes affected Javadocs, Compile API,
  glossary impact, planning status, and documentation validation in the same overall change.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [validation glossary](../../../../glossary.md#term-validation))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

During implementation, [run](../../../../glossary.md#term-run) focused tests for each changed [owner](../../../../glossary.md#term-owner):

```bash
./gradlew :modules:compiler:test \
  --tests io.github.pho001.synaptik.compiler.LayoutInferenceTest \
  --tests io.github.pho001.synaptik.compiler.AutogradPreflightTest \
  --tests io.github.pho001.synaptik.compiler.FirstOrderAutogradTest \
  --tests io.github.pho001.synaptik.compiler.LayoutWindowGradientRulesTest \
  --tests io.github.pho001.synaptik.compiler.IndexingScatterGradientRulesTest \
  --tests io.github.pho001.synaptik.compiler.OrderingStochasticGradientRulesTest
```

The focused [matrix](../../../../glossary.md#term-matrix-meanings) must cover:

- every exact inventory kind/attributes/output/input classification;
- wrong attributes, input/output counts, output slot, auxiliary wrapper, type, Shape, axis,
  reduction, batch, K, probability, and state descriptors;<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [descriptors glossary](../../../../glossary.md#term-descriptor))</span>
- no-ID [preflight](../../../../glossary.md#term-preflight) failure, including a later invalid [occurrence](../../../../glossary.md#term-occurrence) after an earlier valid SORT;
- signed/strided/empty/dynamic `SliceAttrs` and both `CropToShapeAttrs` variants;
- static and dynamic 2D window constraints plus general-[axis](../../../../glossary.md#term-axis) overlap formulas;
- repeated [Gather](../../../../glossary.md#gather) indices and every scatter reduction for both [Elements](../../../../glossary.md#term-element) and [ND](../../../../glossary.md#term-nd) geometry;
- MUL zero-free, one-zero, multiple-zero, duplicate, NaN, infinity, and signed-zero formula
  structure;<span data-glossary-navigation="true"> ([infinity glossary](../../../../glossary.md#term-infinity); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- MIN/MAX base/update/duplicate ties, opposite signed zeros, infinities, and NaN-output zero
  routing;<span data-glossary-navigation="true"> ([infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- ascending/descending SORT, ties, [NaN](../../../../glossary.md#term-nan)-last, [signed zero](../../../../glossary.md#term-signed-zero), and exactly one matching ARGSORT;
- TOP_K values versus indices slots, `k == 0`, stable cutoff, and [canonical](../../../../glossary.md#term-canonical-form) indices [identity](../../../../glossary.md#term-identity-meanings);
- [DROPOUT](../../../../glossary.md#dropout--inverted-dropout) probability zero/non-zero, canonical [mask](../../../../glossary.md#term-boolean-mask) identity, and all state/mask ND routes;
- repeated exact inputs, contribution ordering, GraphPhase, canonical output capture, constants,
  and shared gradient <span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [constants glossary](../../../../glossary.md#term-constant); [gradient glossary](../../../../glossary.md#term-gradient))</span>[`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) behavior; and
- unknown kinds and unassigned [Compiler](../../../../glossary.md#term-compiler) 0005D families still failing closed.

After executable code stabilizes, run one final [module](../../../../glossary.md#term-module) suite:

```bash
./gradlew :modules:compiler:test
```

The separate documentation-focused pass then runs:

```bash
./gradlew :modules:compiler:javadoc
git diff --check
```

It must also check local Markdown links and heading anchors, balanced fences, trailing whitespace,
exact task/master/roadmap status [synchronization](../../../../glossary.md#term-synchronization), the authorized 23-path ceiling, no later
compiler task specifications, and the absence of [Model](../../../../glossary.md#term-model)/[API](../../../../glossary.md#term-application-programming-interface-api)/build/architecture/[backend](../../../../glossary.md#backend)/[runtime](../../../../glossary.md#term-runtime)
[scope](../../../../glossary.md#term-scope) drift.

Repository-wide tests, architecture tests, backend [conformance](../../../../glossary.md#term-conformance), and integration tests are deferred
to the Compiler 0005E first-order capability [checkpoint](../../../../glossary.md#term-checkpoint) or [CI](../../../../glossary.md#term-continuous-integration-ci). This task changes one module,
introduces no [dependency](../../../../glossary.md#term-dependency)/build/boundary change, and must not repeat a successful final compiler
suite in the documentation pass unless executable Java changed afterward.

## Dependencies

- [Model](../../../../glossary.md#term-model) 0025, [canonical](../../../../glossary.md#term-canonical-form) [TensorProducer](../../../../glossary.md#producer--tensorproducer) outputs — Complete.
- Model 0025C, portable functional-scatter reduction semantics — Complete.
- Model 0025D, dynamic-[extent](../../../../glossary.md#term-extent) [slice](../../../../glossary.md#slice) extraction and symbolic slice placement — Complete.
- [Compiler](../../../../glossary.md#term-compiler) 0001–0005B — Complete.

No [dependency](../../../../glossary.md#term-dependency) remains unresolved.

## Follow-up tasks

- [Compiler](../../../../glossary.md#term-compiler) 0005D remains Draft and owns attention, [convolution](../../../../glossary.md#term-channels-first-convolution-layers), [pooling](../../../../glossary.md#term-pooling), and [loss](../../../../glossary.md#loss) [gradients](../../../../glossary.md#term-gradient).
- Compiler 0005E remains Draft and owns the full source-backed first-order and transitive formula
  closure checkpoint.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Compiler 0006 remains Draft and owns explicit gradient requests and higher-order
  differentiation only after 0005E.

Do not create detailed specifications for those tasks in this change.

## Architecture impact

Expected impact: None.

This task implements [compiler](../../../../glossary.md#term-compiler)-owned [inference](../../../../glossary.md#term-inference), [preflight](../../../../glossary.md#term-preflight), and ordinary [Tensor](../../../../glossary.md#tensor)-expression
[derivative](../../../../glossary.md#term-derivative) rules inside the existing [architecture contract](../../../../glossary.md#architecture-contract). If implementation requires a [Model](../../../../glossary.md#term-model),
public [API](../../../../glossary.md#term-application-programming-interface-api), [dependency](../../../../glossary.md#term-dependency), [module](../../../../glossary.md#term-module)-boundary, phase, [capture](../../../../glossary.md#term-capture-meanings), ID, [runtime](../../../../glossary.md#term-runtime), [backend](../../../../glossary.md#backend), or architecture
change, stop and report the conflict.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are working in the Synaptik repository.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md,
docs/planning/modules/compiler/master-plan.md, and
docs/planning/modules/compiler/tasks/0005c-layout-window-indexing-scatter-ordering-and-stochastic-gradient-completion.md.
Read the directly referenced completed Model 0025C/0025D and Compiler 0004A/0005A/0005B contracts,
then inspect the affected source/tests.

Implement Compiler task 0005C exactly as specified. Preserve the existing uncommitted Model
0025C/0025D change. Do not implement out-of-scope work, do not create later task specs, and do not
commit or push.

Stop before edits if a concrete architecture, Model/API, representability, or maximum-scope
conflict remains. Otherwise complete the implementation and focused/final compiler validation.

Then hand the exact diff and reusable Java-test evidence to a separate documentation-focused
clean-context agent/thread. That pass must follow docs/developer-guide/documentation-rules.md,
review affected Javadocs, Compile/Tensor/Training API impact and glossary terminology, finalize
the same task/master/roadmap status, run final Javadoc and documentation checks, and record exact
evidence. It must not rerun successful Java tests unless executable Java changes.

Update this task's local decisions, evidence, implementation notes, completion summary, and final
status. Mark Complete only when implementation, tests, documentation review, status
synchronization, scope checks, and all required validation pass.
```

Documentation-agent [handoff](../../../../glossary.md#term-handoff) requirements:

- identify this task and the exact implementation diff;
- include final [compiler](../../../../glossary.md#term-compiler) test commands/results and whether Java changed afterward;
- identify affected public behavior, package-private Javadocs, canonical auxiliary-output use,
  numerical policies, and architecture boundaries;<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- require review of `docs/api/compile-api.md`, no-change conclusions for Tensor/Training APIs and
  glossary when appropriate, and exact planning synchronization; and<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [planning glossary](../../../../glossary.md#term-planning); [synchronization glossary](../../../../glossary.md#term-synchronization))</span>
- require link/anchor/fence/whitespace/[scope](../../../../glossary.md#term-scope)/status [validation](../../../../glossary.md#term-validation) and a completion summary.

## Local decisions

- One matching stable ARGSORT [occurrence](../../../../glossary.md#term-occurrence) is the sole approved forward-semantic recomputation.
- Scatter MUL uses an explicit update-zero-count and safe-product formula. Multiple update zeros
  route positive-zero update cotangents even at other exceptional update values; the base
  cotangent retains Model 0025C product semantics.<span data-glossary-navigation="true"> ([cotangent glossary](../../../../glossary.md#cotangent); [Model glossary](../../../../glossary.md#term-model))</span>
- Scatter MIN/MAX split among numeric-equality winners, including opposite signed zeros, and
  return positive zero for every candidate when the forward result is NaN.<span data-glossary-navigation="true"> ([positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [NaN glossary](../../../../glossary.md#term-nan))</span>
- SORT freezes the exact stable permutation; TOP_K freezes the exact canonical selected indices.
  Neither averages across equal or cutoff values.<span data-glossary-navigation="true"> ([permutation glossary](../../../../glossary.md#permutation))</span>
- [DROPOUT](../../../../glossary.md#dropout--inverted-dropout) reuses the same-occurrence [canonical](../../../../glossary.md#term-canonical-form) [mask](../../../../glossary.md#term-boolean-mask) and inverted-dropout scale.
- Existing compiler predicates are sufficient for dynamic slice/crop/window obligations; no new
  public or internal constraint vocabulary is planned.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [slice glossary](../../../../glossary.md#slice))</span>
- Separate narrow indexing, ordering, and stochastic rule [owners](../../../../glossary.md#term-owner) keep `LayoutGradientRules` and
  `FirstOrderAutograd` from becoming catch-all formula classes.

## Known limitations

- Index bounds and `NONE` duplicate uniqueness depend on eventual values and remain execution-time
  validity obligations; compiler preflight does not inspect payloads.<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight))</span>
- General-[axis](../../../../glossary.md#term-axis) unfold/fold retain their current static transformed-axis contract.
- First-order policies at non-regular scatter/order boundaries are the selected conventions above;
  they do not claim a unique mathematical derivative.
- Higher [derivatives](../../../../glossary.md#term-derivative) and the final transitive formula audit remain deferred to tasks 0005E–0006.
- No [backend](../../../../glossary.md#backend) currently gains execution support from this [compiler](../../../../glossary.md#term-compiler)-only task.

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning)-context evidence:

- Architecture, documentation rules/profiles, planning guide, roadmap, compiler/model plans,
  completed prerequisite tasks, adjoint audits, APIs, glossary, assigned Model kinds/attributes,
  compiler inference/preflight/autograd/rule sources, and focused tests were reviewed.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [model glossary](../../../../glossary.md#term-model); [adjoint glossary](../../../../glossary.md#term-adjoint); [inference glossary](../../../../glossary.md#term-inference); [preflight glossary](../../../../glossary.md#term-preflight); [autograd glossary](../../../../glossary.md#autograd))</span>
- Representability audit: passed. Every formula is expressible through the current public Tensor
  surface and canonical producer outputs; no Model or architecture blocker remains.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [canonical glossary](../../../../glossary.md#term-canonical-form); [producer glossary](../../../../glossary.md#term-producer))</span>
- [Scope](../../../../glossary.md#term-scope) audit: passed. The implementation stays in `modules/compiler` plus directly affected
  documentation/planning. The authorized ceiling is 23 paths after adding only
  `docs/api/tensor-api.md` for four stale current-status corrections.

Implementation-context Java evidence:

- `./gradlew :modules:compiler:compileJava :modules:compiler:compileTestJava` passed.
- The early four-suite focus over `LayoutInferenceTest`, `AutogradPreflightTest`,
  `FirstOrderAutogradTest`, and `GradientRulesTest` initially ran 44 tests with three failures.
  All three were stale pre-0005C crop/dynamic-slice expectations; after those tests were updated
  to the implemented 0005C contract, the same four-suite focus passed.<span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice))</span>
- The first `./gradlew :modules:compiler:test` attempt ran 177 tests with two failures. Both were
  stale `GraphCompilerTest` unsupported-operation expectations, not production defects. After
  those expectations were updated, the final module rerun passed as recorded below.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [module glossary](../../../../glossary.md#term-module))</span>
- The exact six-suite focused command from this task passed.
- Final `./gradlew :modules:compiler:test` passed 25 suites and 177 tests with zero failures,
  errors, or skips.
- After that module run, the three new rule tests were strengthened to compile representative
  formulas through the combined <span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [compile glossary](../../../../glossary.md#compile))</span>`GraphCompiler` pipeline and assert gradient roles/phases. The
  focused command for <span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient))</span>`LayoutWindowGradientRulesTest`, `IndexingScatterGradientRulesTest`, and
  `OrderingStochasticGradientRulesTest` passed. No production/executable Java changed after the
  passing final module suite, so the documentation pass reused both results and did not rerun Java
  tests.

Documentation-focused context
`/root/implement_compiler_0005c/docs_compiler_0005c`:

- Independently reviewed the architecture contract, documentation rules and General/API-Javadoc/
  Planning/User-guide/Example profiles, planning guide, roadmap, compiler/model plans, this task,
  Compile/Tensor/Training APIs, glossary, affected source/tests/Javadocs, and the relevant
  autograd/training architecture.<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Finalized all seven affected production Javadocs, the Compile API, glossary compiler-status
  references, this task, both master plans, and the roadmap. The authorized Tensor API expansion
  changed exactly four stale Compiler-0005C status/adoption statements and no Tensor contract.
- Training API: accurate unchanged because 0005C adds package-private compiler formula coverage,
  not a public objective/seed, publication, optimizer, session, prepare, or execution contract.<span data-glossary-navigation="true"> ([objective glossary](../../../../glossary.md#term-objective); [seed glossary](../../../../glossary.md#term-seed); [publication glossary](../../../../glossary.md#term-publication); [optimizer glossary](../../../../glossary.md#term-optimizer); [prepare glossary](../../../../glossary.md#prepare))</span>
- Model capabilities and Model source/tests: accurate unchanged outside the preserved 0025C/0025D
  baseline because the task adds no Model semantic, Tensor member, producer-output, or derivative
  ownership.<span data-glossary-navigation="true"> ([derivative glossary](../../../../glossary.md#term-derivative); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Architecture/ADRs/tests, Gradle/build/dependencies, other modules, backend conformance, and
  integration: accurate unchanged because no boundary, dependency, public lifecycle, backend, or
  executable cross-module behavior changed.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [dependency glossary](../../../../glossary.md#term-dependency); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- `./gradlew :modules:compiler:javadoc` passed after final [Javadoc](../../../../glossary.md#term-javadoc) edits (`BUILD SUCCESSFUL`, seven
  actionable tasks, two executed and five up-to-date).
- `python3 /tmp/validate_synaptik_markdown.py` passed 12 affected Markdown files and 722 local
  links, including heading anchors, balanced fences, final newlines, and trailing whitespace.
- `git diff --check` passed with no output. A separate affected-path trailing-whitespace scan
  returned no matches.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- Final Compiler-0005C scope is exactly 20 paths: seven production, six test, and seven
  documentation/planning paths, below the authorized 23-path ceiling. Preserved Model 0025C/0025D
  source, tests, APIs, task files, capabilities, and planning synchronization were audited
  separately and are not attributed to Compiler 0005C.<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization))</span>
- Status synchronization passed: task/master/roadmap record 0005C Complete; 0005D–0005E and 0006
  remain Draft; no detailed specification exists for any later compiler task.

## Implementation notes

- Extended `LayoutInference` for exact `CropToShapeAttrs` extraction versus placement, finite
  signed-length slice bounds, and deferred two-dimensional window-domain constraints.<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite); [slice glossary](../../../../glossary.md#slice))</span>
- Extended layout formulas through length-defined/target-relative slice inverses and exact
  axis/two-dimensional window adjoints.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [target glossary](../../../../glossary.md#term-target); [axis glossary](../../../../glossary.md#term-axis))</span>
- Added narrow indexing, ordering, and stochastic rule owners; preflight remains the sole
  allocation-free selector and <span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight))</span>`FirstOrderAutograd` remains the sole reverse dispatcher.
- Gather/scatter formulas preserve duplicate routing and the fixed replacement/addition,
  zero-count product, and numeric-equality extrema policies.<span data-glossary-navigation="true"> ([Gather glossary](../../../../glossary.md#gather); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum))</span>
- SORT creates one exact matching stable ARGSORT; TOP_K and DROPOUT consume canonical
  same-occurrence indices/mask outputs. Index, mask, one-hot, and RNG-state roles remain
  non-differentiable.<span data-glossary-navigation="true"> ([DROPOUT glossary](../../../../glossary.md#dropout--inverted-dropout); [canonical glossary](../../../../glossary.md#term-canonical-form); [occurrence glossary](../../../../glossary.md#term-occurrence); [mask glossary](../../../../glossary.md#term-boolean-mask); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [differentiable glossary](../../../../glossary.md#derivative-and-chain-rule))</span>
- No public API, Model behavior, architecture, dependency, build, backend/runtime behavior, or
  executable Java changed during the documentation pass.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Model glossary](../../../../glossary.md#term-model); [dependency glossary](../../../../glossary.md#term-dependency); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime))</span>

## Completion summary

- Completed changes: Completed the exact assigned layout/window/indexing/scatter/ordering/dropout
  inference, preflight, and first-order formula matrix through the existing one-capture pipeline.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [inference glossary](../../../../glossary.md#term-inference); [preflight glossary](../../../../glossary.md#term-preflight); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [capture glossary](../../../../glossary.md#term-capture-meanings))</span>
- Files changed or created: Seven compiler production paths, six compiler test paths, and seven
  documentation/planning paths; 20 exact task paths total.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Tests and validation: Reused the passing six-suite focus and 25-suite/177-test final module
  evidence, plus the later passing three-suite combined-pipeline test strengthening; final
  Javadoc, Markdown, scope, status, whitespace, and diff checks passed.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [module glossary](../../../../glossary.md#term-module); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: Complete in clean context
  `/root/implement_compiler_0005c/docs_compiler_0005c`.
- Documentation impact: Compile and Tensor API status, glossary status references, task, compiler
  and model master plans, and roadmap finalized.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [model glossary](../../../../glossary.md#term-model))</span>
- [Javadoc](../../../../glossary.md#term-javadoc) review: All seven affected package-private production contracts finalized.
- Glossary impact: Existing slice/placement entries updated for current compiler adoption; no new
  reusable term was introduced.<span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice))</span>
- Unresolved issues: None.
- Follow-up required: None. [Compiler](../../../../glossary.md#term-compiler) 0005D–0005E and 0006 remain Draft without specifications.

Status: Complete
