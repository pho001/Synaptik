# Task 0005C: Layout, Window, Indexing, Scatter, Ordering, and Stochastic Gradient Completion

## Status

Complete

## Goal

Complete the first-order compiler gradient matrix for the remaining layout, slice, composition,
window, indexing, scatter, ordering, and explicit-state dropout operation families.

The task extends the existing compiler-owned pre-capture Tensor-expression autograd pipeline. It
uses ordinary public `Tensor` operations, exact original producer occurrences, canonical auxiliary
outputs, and request-local logical splats. It does not add a second algebra, a backward-only
operation kind, mutable Tensor gradient state, or a public compiler gradient surface.

The central relationship is:

```text
exact original occurrence and selected floating role
  -> allocation-free fail-closed preflight
  -> ordinary public Tensor formula
  -> one combined phase-aware capture
```

Models 0025C and 0025D are Complete. They fix the forward represented-value meaning of
configurable scatter reductions and supply the two dynamic-slice construction forms that were
previously missing. No remaining Model or architecture prerequisite blocks this task.

## Scope

### Source-backed operation and role inventory

The inventory below is exhaustive over the assigned current source families and exact signatures.
`D` means a floating data role through which a cotangent may propagate. `ND` means intentionally
non-differentiable.

| Exact kind and attributes | Ordered outputs | Ordered input roles | Task result |
|---|---|---|---|
| `CONTIGUOUS + NoOperationAttrs` | value slot 0 | input `D` | Preserve the implemented identity cotangent. |
| `RESHAPE`, `EXPAND + TargetShapeAttrs` | value slot 0 | input `D` | Preserve the implemented reshape and binding-aware sum-to-Shape cotangents. |
| `PERMUTE + PermutationAttrs` | value slot 0 | input `D` | Preserve the implemented inverse permutation. |
| `EXPAND_DIMS`, `SQUEEZE + AxisTransformAttrs` | value slot 0 | input `D` | Preserve the implemented inverse rank edit. |
| `SLICE + SliceAttrs` | value slot 0 | source `D` | Support every normalized finite signed region, including regions created by `sliceByLength`. |
| `SLICE + CropToShapeAttrs` | value slot 0 | source `D` | Add target-relative dynamic placement through `sliceUpdate(Tensor, Shape)`. |
| `SLICE_UPDATE + SliceAttrs` | value slot 0 | base `D`, update `D` | Preserve the base formula and generalize update extraction through `sliceByLength`. |
| `SLICE_UPDATE + CropToShapeAttrs` | value slot 0 | base `D`, update `D` | Add exact target-relative base masking and update extraction. |
| `SELECT + SelectAttrs` | value slot 0 | source `D` | Preserve exact singleton placement, including unresolved selected extents. |
| `PAD + PadAttrs` | value slot 0 | source `D`; constant metadata `ND` | Preserve target-relative crop. |
| `TILE + TileAttrs` | value slot 0 | source `D`; repeats metadata `ND` | Preserve interleaved reshape/reduction. |
| `CONCAT`, `STACK + CompositionAxisAttrs` | value slot 0 | every ordered input position `D`; axis metadata `ND` | Preserve position-specific crop/select contributions. |
| `UNFOLD_AXIS + UnfoldAxisAttrs` | value slot 0 | source `D` | Add `foldAxis` overlap-add adjoint. |
| `FOLD_AXIS + FoldAxisAttrs` | value slot 0 | window tensor `D` | Add matching `unfold` adjoint. |
| `UNFOLD2D + Window2dAttrs` | value slot 0 | NCHW source `D` | Add exact `fold2d` adjoint. |
| `UNFOLD2D + Unfold2dAttrs` | value slot 0 | NCHW source `D`; padding scalar metadata `ND` | Add the same `fold2d` adjoint; padding is not a Tensor role. |
| `FOLD2D + Fold2dAttrs` | value slot 0 | canonical columns `D` | Add zero-padding `unfold2d` adjoint. |
| `GATHER`, `GATHER_ELEMENTS + IndexAxisAttrs` | value slot 0 | data `D`, indices `ND` | Add Gather-compatible or same-rank additive scatter. |
| `GATHER_ND + GatherNdAttrs` | value slot 0 | data `D`, indices `ND` | Add additive Scatter-ND. |
| `SCATTER_ADD + IndexAxisAttrs` | value slot 0 | data `D`, indices `ND`, updates `D` | Data identity plus matching Gather update route. |
| `SCATTER_ELEMENTS + ScatterElementsAttrs` for `NONE`, `ADD`, `MUL`, `MIN`, `MAX` | value slot 0 | data `D`, indices `ND`, updates `D` | Add the complete reduction-specific matrix below. |
| `SCATTER_ND + ScatterNdAttrs` for `NONE`, `ADD`, `MUL`, `MIN`, `MAX` | value slot 0 | data `D`, indices `ND`, updates `D` | Add the complete reduction-specific matrix below. |
| `ONE_HOT + OneHotAttrs` | BOOL slot 0 | integral indices `ND`; depth metadata `ND` | No differentiable role. |
| `SORT + SortAttrs` | values slot 0 | input `D` only for floating values | Add one exact matching stable `ARGSORT` routing formula. |
| `ARGSORT + SortAttrs` | INT64 slot 0 | input `ND` | No differentiable role. |
| `TOP_K + TopKAttrs` | values slot 0, INT64 indices slot 1 | input `D` only from values slot 0 | Route through canonical slot-1 indices; indices output is `ND`. |
| `INITIAL_STATE + GraphRngStateAttrs` | INT64 state slot 0 | no inputs; key/counter metadata `ND` | No differentiable role. |
| `DROPOUT + DropoutAttrs` | value slot 0, BOOL mask slot 1, INT64 next-state slot 2 | floating input `D` only from value slot 0; RNG state `ND` | Reuse the canonical mask; mask and state outputs remain `ND`. |

Only `FLOAT64`, `FLOAT32`, and `BFLOAT16` data roles are differentiable. Integral layout,
gather/scatter, sort, or top-K values remain valid forward operations but have no first-order
cotangent route. BOOL values, coordinates, indices, one-hot values, keep masks, graph RNG state,
padding values, probabilities, axes, Shapes, lengths, steps, repeats, window geometry, selection
counts, direction flags, and reduction attributes are non-differentiable.

Conveniences add no rows: `transpose` is `PERMUTE`, `flip` is `SLICE`, `embedding` is
`GATHER(axis=0)`, and `unstack` is repeated independent `SELECT`.

### Notation and shared construction rules

For one selected occurrence:

- `g` is the accumulated cotangent for the selected output slot;
- `x`, `data`, `indices`, `updates`, and `state` are exact original input wrappers;
- `y` is the exact canonical forward value output;
- `Z_t` and `O_t` are request-local exact typed positive-zero and positive-one splats expanded to
  `t.shape()`;
- `G_axis(v)` means the exact matching `gather`, `gatherElements`, or `gatherNd` operation for the
  original indexing geometry; and
- `S_axis(base, v, reduction)` means the exact matching `scatterAdd`, `scatterElements`, or
  `scatterNd` operation with the original indices, normalized axis or batch count, and reduction.

Every formula must call existing public `Tensor` methods and immutable public model values. Rule
owners must not construct `Operation`, `TensorProducer`, `TensorProvenance`, graph nodes, graph
values, descriptors, or graph IDs directly.

All selected input and output data types are one exact floating type. No family in this task
requires mixed-floating cotangent conversion. Every formula must return the exact selected input
Shape and type. Request-local constants retain the existing exact represented-bit cache and
explicit `CompileTimeConstantGraph` ingress; Shape-specific constants are ordinary public
`expand` expressions.

### Layout, slice, selection, padding, tile, and composition formulas

Preserve the completed formulas for `CONTIGUOUS`, `RESHAPE`, binding-aware `EXPAND`,
`EXPAND_DIMS`, `SQUEEZE`, `PERMUTE`, `SELECT`, `PAD`, `TILE`, `CONCAT`, and `STACK`.

The complete slice formulas become:

| Exact occurrence and selected role | Formula |
|---|---|
| `SLICE + SliceAttrs(starts,lengths,axes,steps)`, source | `Z_x.sliceUpdate(g, starts, axes, steps)` |
| `SLICE + CropToShapeAttrs(target,prefix)`, source | `Z_x.sliceUpdate(g, prefix)` |
| `SLICE_UPDATE + SliceAttrs`, base | `g.sliceUpdate(Z_updates, starts, axes, steps)` |
| `SLICE_UPDATE + SliceAttrs`, update | `g.sliceByLength(starts, lengths, axes, steps)` |
| `SLICE_UPDATE + CropToShapeAttrs(target,prefix)`, base | `g.sliceUpdate(Z_updates, prefix)` |
| `SLICE_UPDATE + CropToShapeAttrs(target,prefix)`, update | `g.cropToShape(target, prefix)` |

`sliceByLength` removes the old static-selected-base restriction from the update-role formula.
The exact finite lengths are already stored in `SliceAttrs`; no raw end is reconstructed.
`sliceUpdate(Tensor, Shape)` makes target-relative placement valid when the base, prefix, or update
extent is unresolved. Empty regions remain exact empty placements/extractions. Signed non-zero
steps remain injective, so these formulas replace rather than add.

`SELECT` continues to use:

```text
Z_x.sliceUpdate(g.expandDims(axis), Shape prefix with prefix[axis] = index)
```

The implementation may retain its existing one-axis `SliceAttrs` spelling only when that spelling
is valid for unresolved selected extents. The preferred unified formula uses target-relative
placement and must preserve the exact input Shape.

`PAD` remains `g.cropToShape(x.shape(), Shape.of(beforeWidths))`.

For `TILE`, reshape `g` to
`[repeat0,x0,repeat1,x1,...]`, sum the even repeat axes with dimensions retained, then reshape to
the exact `x.shape()`. Scalar tile returns `g`.

For `CONCAT` input position `i`, use
`g.cropToShape(input_i.shape(), prefix_i)`, where every non-concat prefix extent is zero and the
concat-axis prefix is the exact ordered symbolic sum of earlier input extents. For `STACK` input
position `i`, use `g.select(axis, i)`. Repeated exact Tensor inputs remain separate positional
contributions.

### Window formulas and dynamic geometry

Use the exact recorded attributes and Shapes:

| Kind | Formula |
|---|---|
| `UNFOLD_AXIS + UnfoldAxisAttrs(axis,size,step)` | `g.foldAxis(axis, static x.shape[axis], step)` |
| `FOLD_AXIS + FoldAxisAttrs(axis,outputSize,step)` | `g.unfold(axis, static input.shape[last], step)` |
| either `UNFOLD2D` attributes variant | `g.fold2d(x.shape(), window)` |
| `FOLD2D + Fold2dAttrs(outputShape,window)` | `g.unfold2d(window)` |

The `UNFOLD2D + Unfold2dAttrs` padding value is non-differentiable scalar metadata. Its source
adjoint is still `fold2d`, because out-of-domain padding samples do not refer to source positions.
The `FOLD2D` adjoint uses the existing conceptual-positive-zero `unfold2d` form so column entries
that map outside the output receive positive zero.

General-axis unfold/fold retain their current statically known transformed extents. Unaffected
Dimensions may remain unresolved.

For both 2D directions, preserve exact symbolic channel and spatial formulas. Inference must emit
an occurrence-owned constraint for each height/width domain that cannot yet prove:

```text
inputOrTargetSpatial + 2 * padding >= dilation * (kernel - 1) + 1
```

The existing symbolic floor/ceiling and product expressions derive the exact output or expected
column Shape. A disproven domain fails inference; an undecidable domain remains a
`DeferredGraphConstraint`. No binding, unknown placeholder, or new predicate vocabulary is
needed.

### Gather formulas

Indices are always non-differentiable:

| Kind | Data cotangent |
|---|---|
| `GATHER(data,indices,axis)` | `Z_data.scatterAdd(indices, g, axis)` |
| `GATHER_ELEMENTS(data,indices,axis)` | `Z_data.scatterElements(indices, g, axis, ADD)` |
| `GATHER_ND(data,indices,batchDimensions)` | `Z_data.scatterNd(indices, g, ADD, batchDimensions)` |

Repeated indices accumulate through the explicit `ADD` reduction. Dynamic gathered extents remain
representable because `SCATTER_ADD` has the exact Gather-compatible Shape contract.

### Scatter `NONE` and `ADD` formulas

For `SCATTER_ADD`, the base cotangent is `g` and the updates cotangent is matching `GATHER(g)`.

For `SCATTER_ELEMENTS` and `SCATTER_ND`:

| Reduction | Base cotangent | Updates cotangent |
|---|---|---|
| `NONE` | `S_axis(g, Z_updates, NONE)` | `G_axis(g)` |
| `ADD` | `g` | `G_axis(g)` |

Valid `NONE` occurrences require unique targets under the forward contract. Preflight does not
inspect index values or invent an encounter-order rule. Bounds and duplicate validity remain
forward execution obligations.

### Scatter `MUL` formula and numerical policy

For one configurable scatter occurrence, build update-group facts with the exact matching
scatter/gather geometry:

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

The base participates exactly once in every update derivative through `dataAtUpdate`. Duplicate
updates remain distinct and each receives its own gathered result. An unaddressed base coordinate
uses the exact multiplicative identity from `O_data`, so its base cotangent is `g`.

Both signed zeros compare equal to positive zero for zero counting. The safe denominator is never
zero. With one update zero, only that update receives the product of the base and all non-zero
updates. With more than one update zero, every update contribution is selected as exact positive
zero. The base cotangent always uses the Model 0025C product of all updates, including its NaN,
zero-times-infinity, sign, overflow, underflow, and rounding behavior.

NaN updates are not zero and make the safe product NaN. A zero-free NaN group therefore routes NaN
through ordinary formula arithmetic. A sole zero beside a NaN or infinity routes the Model
non-zero product to that zero update. The selected multiple-zero branch is positive zero even if
another update is NaN or infinite. This is the explicit compiler first-order convention at that
non-regular boundary; it does not change the forward product.

### Scatter `MIN` and `MAX` formula and numerical policy

Use the exact canonical forward output `y`; do not recompute the extremum:

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
winners share the cotangent equally. Numeric equality makes opposite signed zeros ties even though
Model 0025C selects negative zero for `MIN` and positive zero for `MAX`. Equal finite values and
same-sign infinities also share. Non-winners receive positive zero.

If the forward result is NaN, numeric equality is false for every candidate. `hasWinner` is false
and every candidate receives exact positive zero, matching the completed reduction-extrema
compiler convention. The safe denominator prevents division by zero in the selected formula.

### Ordering formulas and boundary policy

`ARGSORT` has no differentiable result.

For each selected one-output floating `SORT(x, SortAttrs(axis,descending))` occurrence, construct
exactly one ordinary public:

```text
indices = x.argsort(axis, descending)
dX = Z_x.scatterElements(indices, g, axis, NONE)
```

Before any Tensor allocation, preflight must validate:

- exact `OrderingKind.SORT`;
- exact `SortAttrs`;
- one input and one output;
- exact equal floating input/output type and Shape;
- normalized axis for the exact input rank;
- the exact direction flag;
- stable, NaN-last ordering from the current Model contract; and
- constructibility of one matching `ARGSORT + SortAttrs`, one input, one INT64 output occurrence.

This is the sole permitted matching recomputation in this task. It is one separate, stable,
single-output `ARGSORT` Tensor occurrence per selected `SORT` occurrence. It is not a hidden SORT
output, a public `SortResult`, a producer sibling lookup, a producer reconstruction, or a Model API
change.

The derivative freezes the exact current stable permutation. Equal-value and same-NaN classes use
their stable increasing logical input-index order; opposite signed zeros use the Model total order.
No cotangent averaging occurs across equal keys. At crossings, NaN transitions, and other ordering
discontinuities, the selected first-order convention remains this exact routed permutation.

For `TOP_K`, only values slot 0 is differentiable:

```text
indices = producer.output(1)
dX = Z_x.scatterElements(indices, g, axis, NONE)
```

Preflight requires the exact canonical wrapper at output slot 1, exact producer identity, exact
INT64 type, exact values Shape, non-gradient eligibility, and matching `TopKAttrs`. Selected
indices are unique. `k == 0` returns `Z_x`. At equal cutoffs, only the stable selected members
receive cotangents; excluded equal values receive zero. NaN membership and `sorted` output order
are routed through the same canonical slot-1 indices. No selected-set averaging or recomputation
is allowed.

### Explicit-state dropout formula and probability policy

Only dropout value slot 0 may propagate to floating input position 0:

```text
mask = producer.output(1)
dInput = where(mask, g.div(1.0d - probability), Z_input)
```

Preflight requires exact `DropoutKind.DROPOUT`, `DropoutAttrs`, inputs `[input,state]`, outputs
`[value,mask,nextState]`, exact canonical slot-1 wrapper, matching Shape, BOOL mask type,
non-gradient mask eligibility, and valid INT64 `Shape[2]` state descriptors at input 1 and output
2. The probability must remain finite and numerically in `[0,1)`.

The formula reuses the exact forward keep mask. It must not infer a mask from the output, resample,
construct another dropout occurrence, advance or replace RNG state, or differentiate either state
edge. `where`, rather than mask multiplication, preserves exact positive zero for dropped NaN or
infinity inputs. At either signed probability zero, the denominator is positive one and the
canonical all-kept mask routes `g`; the occurrence and state transition remain present. There is
no probability-one case because the Model rejects it.

`INITIAL_STATE`, dropout state input/output, and dropout mask output remain non-differentiable.

### Inference normalization and retained constraints

Inference must independently derive the current exact output descriptors and preserve the existing
Shape/type normalization for every assigned family.

The slice-family rules are exact:

- `SLICE + SliceAttrs` derives selected static lengths and emits no constraint for a zero-length
  entry; a non-empty signed entry retains the upper-bound obligation
  `max(start,last)+1 <= sourceExtent`;
- `SLICE_UPDATE + SliceAttrs` additionally requires update Shape equal to the derived region and
  returns the exact base Shape with base/update gradient-eligibility OR;
- `SLICE + CropToShapeAttrs` returns exact `targetShape` and retains
  `prefix + target <= source` for every axis;
- `SLICE_UPDATE + CropToShapeAttrs` requires update Shape equal to exact `targetShape`, returns
  exact base Shape and type with base/update eligibility OR, and retains
  `prefix + target <= base` for every axis.

This corrects the current `LayoutInference` branch that treats both `CropToShapeAttrs` kinds as
one-input extraction. It must not change Model construction.

Use checked arithmetic for finite final coordinates, `maxCoordinate + 1`, effective kernels,
padding offsets, static counts, and interleaved Shapes. A statically contradicted constraint fails.
An unresolved but validly representable relation remains ordered occurrence-local compiler state.
Do not replace a dynamic Dimension with a static guess or unnamed unknown.

Index-value bounds and `NONE` duplicate uniqueness cannot be proven from descriptors and are not
invented as Shape constraints. They remain validity obligations of eventual execution.

### Fail-closed preflight and failure order

Preserve the top-level Compiler 0004–0005B validation order. Before constructing a seed, logical
splat, matching ARGSORT, formula Tensor, or any derivative `TensorId`, visit all selected
occurrences in deterministic producer postorder and validate:

1. exact kind and exact attributes class/value;
2. exact input and output cardinality;
3. selected canonical output slot and output-specific differentiability;
4. input roles in ascending position order, including every `ND` index/state role;
5. exact floating type and gradient eligibility of each selected data role;
6. independently re-derived descriptors and every non-disproven occurrence constraint;
7. normalized axis, batch count, tuple depth, Shape, slice region, window geometry, reduction, K,
   probability, and stable-order facts;
8. exact canonical auxiliary output identity for TOP_K and DROPOUT;
9. constructibility of the sole matching ARGSORT occurrence for SORT; and
10. the exact numerical/boundary policy row above.

Preflight must not inspect Tensor payloads, index values, masks, storage, labels, layouts for
value inference, or backend behavior. Unknown kinds, wrong attributes, wrong cardinality, wrong
output slot, missing canonical outputs, descriptor contradictions, unsupported roles, or
disproved constraints fail deterministically.

No known failure may consume a seed, constant, formula Tensor, matching ARGSORT Tensor, or
derivative ID. After successful complete preflight, ordinary public Tensor construction, capture,
inference, validation, optimization, publication, or planning failure may consume opaque
non-reusable model IDs under the existing architecture contract.

### Determinism, producers, outputs, provenance, and constants

- Preserve exact Tensor and `TensorProducer` identity in inventory, route, output-slot,
  contribution, and accumulation bookkeeping.
- Reverse accumulation order remains reverse producer postorder, ascending selected output slot,
  then ascending input position. Contributions are left-associated through ordinary `Tensor.add`.
- Repeated input positions and duplicate scatter updates remain logically distinct.
- Use `producer.output(1)` for TOP_K indices and DROPOUT mask. Do not reconstruct a wrapper.
- Original producer occurrences and every original output slot remain `GraphPhase.FORWARD`.
- The permitted matching ARGSORT and every other generated formula producer are
  `GraphPhase.BACKWARD`.
- Capture ordered forward outputs and gradient roots together exactly once. Assign every `NodeId`
  and `ValueId` once. Per-node phase remains authoritative.
- Distinct target roles may share one captured gradient value; add no identity node.
- Reuse only reachable request-local exact typed logical splats. Storage, labels, descriptor
  Shape, factory history, or absent provenance never imply constant status.
- Run the existing combined inference, validation, canonicalization, exact rewriting, constant
  folding, whole-graph dead-code elimination, phase-local common-subexpression elimination,
  cleanup, final validation, publication, and planning pipeline unchanged.

### Transitive formula-operation closure

The formulas use only current public operations already covered by Compiler 0005A–0005B or
covered in this task:

- `ADD`, `MUL`, `DIV`, comparisons, Boolean `AND`, and `WHERE`;
- exact typed logical-splat expansion;
- `RESHAPE`, `EXPAND_DIMS`, `SQUEEZE`, `SLICE`, `SLICE_UPDATE`, `SELECT`, and target-relative crop;
- reductions used by the retained TILE formula;
- Gather, Gather Elements, Gather-ND, Scatter Add, Scatter Elements, and Scatter-ND;
- UNFOLD/FOLD axis and 2D window transformations; and
- the explicitly approved matching ARGSORT.

No formula requires a new operation kind, public method, cast policy, mutable gradient, or captured
IR algebra. Task 0005E still owns the final source-backed transitive-closure checkpoint before
higher-order requests become Ready.

## Out of scope

- any Model source, test, operation, attributes, Tensor method, result carrier, producer-output,
  descriptor, Shape, DataType, or public API change
- a hidden SORT indices output, multi-output SORT, public sort result, or any recomputation other
  than the exact matching stable ARGSORT formula specified above
- recomputing TOP_K indices, dropout masks, scatter extrema, or another canonical auxiliary/output
- derivative rows owned by Compiler 0005D: attention, convolution, pooling, and losses
- the Compiler 0005E source-backed closure checkpoint
- explicit objectives, seeds, Jacobians, create-graph, derivative order, disconnected-target
  policy, or other Compiler 0006 work
- mutable Tensor gradients, `Tensor.backward`, a tape, backward-only kind, captured-value-to-Tensor
  conversion, second algebra, public registry/facade/policy, reflection, service loading, string
  dispatch, or global cache
- new graph optimization, algebraic rewrite, folding rule, CSE policy, phase rule, or ID lifecycle
- physical saved buffers, materialization, backend/runtime state, sampling, lowering, execution,
  kernel, algorithm, route, conformance, or integration behavior
- Config, Planning, Runtime, Prepare, Engine, backend, training, trace, ONNX, architecture, ADR,
  dependency, Gradle, Java-version, or build changes
- detailed task specifications or promotion for Compiler 0005D, 0005E, or 0006
- unrelated refactoring or documentation cleanup

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md), especially model ownership,
  compiler-owned pre-capture autograd, canonical producer outputs, one combined capture, phase,
  constant, and forbidden-gradient-surface rules
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Training graph](../../../../architecture/training-graph.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [General documentation style](../../../../developer-guide/documentation/general-style.md)
- [Planning documentation style](../../../../developer-guide/documentation/planning-style.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Compiler master plan](../master-plan.md)
- [Model master plan](../../model/master-plan.md)
- [Model capabilities](../../model/capabilities.md)
- [Adjoint expressibility audit](../../model/adjoint-expressibility-audit.md)
- [Model 0025 canonical outputs](../../model/tasks/0025-canonical-tensor-producer-outputs.md)
- [Model 0025C scatter semantics](../../model/tasks/0025c-portable-functional-scatter-reduction-semantics.md)
- [Model 0025D dynamic slice forms](../../model/tasks/0025d-dynamic-extent-slice-extraction-and-symbolic-slice-placement.md)
- completed Compiler tasks [0004](0004-compiler-owned-pre-capture-autograd-and-combined-graph-compilation.md),
  [0004A](0004a-exact-composition-gradient-rule-extensions.md),
  [0004B](0004b-shared-algebra-cotangent-normalization-and-local-derivative-rules.md),
  [0005A](0005a-derivative-policy-and-elementwise-activation-gradient-completion.md), and
  [0005B](0005b-reduction-scan-softmax-statistics-and-normalization-gradient-completion.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `modules/compiler` owns fail-closed preflight, named derivative-rule dispatch, reverse
  accumulation, inference constraints, combined capture, and compile artifacts.
- `modules/model` remains derivative-agnostic and unchanged.
- Formulas are ordinary public Tensor expressions. They are not a second graph, low-level IR, or
  model-owned derivative language.
- Exact original producer and canonical output wrappers are the only saved-value identity source.
- Compiler identity maps remain request-local and ephemeral.
- One complete preflight precedes any generated Tensor allocation.
- One combined phase-aware capture assigns graph-local IDs once.
- Compiler creates no physical buffer, saved-tensor store, runtime tape, prepared state, backend
  executable, kernel, or route.
- No dependency or module boundary changes are authorized.

If implementation requires a public mask/index surface, another Tensor operation, a Model change,
an architecture change, or a formula outside the closed matrix above, stop and report the
conflict.

## Package impact

Existing package used:

- `io.github.pho001.synaptik.compiler` — the cohesive package-private compiler front-end,
  inference, preflight, named rule owners, reverse accumulation, and tests.

Packages added, moved, renamed, or made public: None.

Type placement:

- `LayoutInference` continues to own layout/slice/window descriptor derivation and constraints.
- `LayoutGradientRules` continues to own layout, slice, selection, padding, tile, composition, and
  now window formulas.
- new package-private `IndexingGradientRules` owns Gather and scatter formulas because those roles
  form one exact indexing geometry boundary.
- new package-private `OrderingGradientRules` owns SORT and TOP_K routing because it alone needs
  stable permutation/selection policy.
- new package-private `StochasticGradientRules` owns DROPOUT mask routing because RNG-state roles
  and probability scaling are distinct from deterministic indexing.
- `AutogradPreflight` remains the sole complete support selector.
- `FirstOrderAutograd` remains the sole reverse traversal, contribution accumulation, constant
  ownership, and named-rule dispatch owner.

The three new rule owners are field-free, package-private, final, and stateless. Do not add a
generic `GradientManager`, `GradientUtil`, registry, facade, service, policy object, or subpackage.

## Affected files

Expected production Java:

- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/LayoutInference.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/LayoutGradientRules.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/IndexingGradientRules.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/OrderingGradientRules.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/StochasticGradientRules.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/AutogradPreflight.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/FirstOrderAutograd.java`

Expected compiler tests:

- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/LayoutInferenceTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/AutogradPreflightTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/GraphCompilerTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/LayoutWindowGradientRulesTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/IndexingScatterGradientRulesTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/OrderingStochasticGradientRulesTest.java`

Expected documentation and planning:

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

- Model source/tests and completed Model task evidence
- `docs/api/training-api.md`
- model capabilities and both adjoint/closure audits
- compile capture, graph-ID, phase, constants, optimization, publication, and planning code/tests
  outside the listed owners
- architecture documents/tests, Gradle/build files, dependencies, other modules, backend
  conformance, and integration tests

## Maximum scope

This task may create or modify at most 23 paths:

- at most nine compiler production paths;
- at most seven compiler test paths; and
- at most seven documentation/planning paths, including this task.

The final expected set is seven production paths, six test paths, and seven
documentation/planning paths. The original 22-path ceiling was explicitly expanded by one path
for the four necessary status-only corrections in `docs/api/tensor-api.md`. Any Model source/test
path, another module, build/dependency path, architecture/ADR path, backend-conformance path,
integration-test path, or twenty-fourth touched path requires stopping and proposing a separately
reviewed follow-up.

## Acceptance criteria

### Inventory and preflight

- The exact assigned source inventory above is encoded in role-aware fail-closed preflight.
- Every selected occurrence validates exact kind, attributes, signature, output slot, input role,
  type, Shape, constraints, auxiliary identity, and policy before generated Tensor allocation.
- BOOL, integral index/coordinate/one-hot, RNG-state, mask, and configuration roles remain
  non-differentiable.
- Unknown/custom kinds and every unsupported exact variant still fail before partial backward
  construction.
- `SLICE_UPDATE + CropToShapeAttrs` inference returns base Shape and validates the update against
  the exact target Shape.
- Dynamic slice/crop and 2D-window obligations are retained or disproved exactly; no static guess
  or binding is introduced.

### Formula behavior

- All layout/slice/selection/pad/tile/composition baseline rows remain supported.
- `sliceByLength` handles signed, strided, empty, and unresolved-base update extraction without raw
  end reconstruction.
- target-relative slice extraction/update formulas use exact Shapes and prefixes.
- all four window kinds use the exact public inverse/adjoint operation and preserve dynamic
  unaffected/2D Dimensions.
- Gather, Gather Elements, and Gather-ND accumulate duplicate-index contributions through their
  exact additive scatter.
- scatter `NONE`, `ADD`, `MUL`, `MIN`, and `MAX` implement both floating data roles and the fixed
  duplicate/zero/tie/NaN/signed-zero policies above.
- SORT constructs exactly one matching stable ARGSORT occurrence and routes through it.
- TOP_K uses exact canonical indices output slot 1 and never recomputes selection.
- DROPOUT uses exact canonical mask output slot 1 and never resamples or advances state.
- Every returned cotangent has the exact selected input floating type and Shape.

### Shared pipeline and boundaries

- Reverse accumulation order, Tensor identity handling, constant ingress, one combined capture,
  per-node phase, graph-ID assignment, optimization, publication, and planning remain unchanged.
- Original auxiliary-output producers remain FORWARD; all generated formula producers, including
  matching ARGSORT, are BACKWARD.
- No public type/member, Model behavior, compiler facade/registry/policy, gradient-only operation,
  second algebra, mutable Tensor gradient, tape, physical saved buffer, or backend/runtime behavior
  is added.
- Compiler 0005D, 0005E, and 0006 remain Draft without detailed specifications.
- A separate documentation-focused clean-context pass finalizes affected Javadocs, Compile API,
  glossary impact, planning status, and documentation validation in the same overall change.

## Tests / validation

During implementation, run focused tests for each changed owner:

```bash
./gradlew :modules:compiler:test \
  --tests io.github.pho001.synaptik.compiler.LayoutInferenceTest \
  --tests io.github.pho001.synaptik.compiler.AutogradPreflightTest \
  --tests io.github.pho001.synaptik.compiler.FirstOrderAutogradTest \
  --tests io.github.pho001.synaptik.compiler.LayoutWindowGradientRulesTest \
  --tests io.github.pho001.synaptik.compiler.IndexingScatterGradientRulesTest \
  --tests io.github.pho001.synaptik.compiler.OrderingStochasticGradientRulesTest
```

The focused matrix must cover:

- every exact inventory kind/attributes/output/input classification;
- wrong attributes, input/output counts, output slot, auxiliary wrapper, type, Shape, axis,
  reduction, batch, K, probability, and state descriptors;
- no-ID preflight failure, including a later invalid occurrence after an earlier valid SORT;
- signed/strided/empty/dynamic `SliceAttrs` and both `CropToShapeAttrs` variants;
- static and dynamic 2D window constraints plus general-axis overlap formulas;
- repeated Gather indices and every scatter reduction for both Elements and ND geometry;
- MUL zero-free, one-zero, multiple-zero, duplicate, NaN, infinity, and signed-zero formula
  structure;
- MIN/MAX base/update/duplicate ties, opposite signed zeros, infinities, and NaN-output zero
  routing;
- ascending/descending SORT, ties, NaN-last, signed zero, and exactly one matching ARGSORT;
- TOP_K values versus indices slots, `k == 0`, stable cutoff, and canonical indices identity;
- DROPOUT probability zero/non-zero, canonical mask identity, and all state/mask ND routes;
- repeated exact inputs, contribution ordering, GraphPhase, canonical output capture, constants,
  and shared gradient `ValueId` behavior; and
- unknown kinds and unassigned Compiler 0005D families still failing closed.

After executable code stabilizes, run one final module suite:

```bash
./gradlew :modules:compiler:test
```

The separate documentation-focused pass then runs:

```bash
./gradlew :modules:compiler:javadoc
git diff --check
```

It must also check local Markdown links and heading anchors, balanced fences, trailing whitespace,
exact task/master/roadmap status synchronization, the authorized 23-path ceiling, no later
compiler task specifications, and the absence of Model/API/build/architecture/backend/runtime
scope drift.

Repository-wide tests, architecture tests, backend conformance, and integration tests are deferred
to the Compiler 0005E first-order capability checkpoint or CI. This task changes one module,
introduces no dependency/build/boundary change, and must not repeat a successful final compiler
suite in the documentation pass unless executable Java changed afterward.

## Dependencies

- Model 0025, canonical TensorProducer outputs — Complete.
- Model 0025C, portable functional-scatter reduction semantics — Complete.
- Model 0025D, dynamic-extent slice extraction and symbolic slice placement — Complete.
- Compiler 0001–0005B — Complete.

No dependency remains unresolved.

## Follow-up tasks

- Compiler 0005D remains Draft and owns attention, convolution, pooling, and loss gradients.
- Compiler 0005E remains Draft and owns the full source-backed first-order and transitive formula
  closure checkpoint.
- Compiler 0006 remains Draft and owns explicit gradient requests and higher-order
  differentiation only after 0005E.

Do not create detailed specifications for those tasks in this change.

## Architecture impact

Expected impact: None.

This task implements compiler-owned inference, preflight, and ordinary Tensor-expression
derivative rules inside the existing architecture contract. If implementation requires a Model,
public API, dependency, module-boundary, phase, capture, ID, runtime, backend, or architecture
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

Documentation-agent handoff requirements:

- identify this task and the exact implementation diff;
- include final compiler test commands/results and whether Java changed afterward;
- identify affected public behavior, package-private Javadocs, canonical auxiliary-output use,
  numerical policies, and architecture boundaries;
- require review of `docs/api/compile-api.md`, no-change conclusions for Tensor/Training APIs and
  glossary when appropriate, and exact planning synchronization; and
- require link/anchor/fence/whitespace/scope/status validation and a completion summary.

## Local decisions

- One matching stable ARGSORT occurrence is the sole approved forward-semantic recomputation.
- Scatter MUL uses an explicit update-zero-count and safe-product formula. Multiple update zeros
  route positive-zero update cotangents even at other exceptional update values; the base
  cotangent retains Model 0025C product semantics.
- Scatter MIN/MAX split among numeric-equality winners, including opposite signed zeros, and
  return positive zero for every candidate when the forward result is NaN.
- SORT freezes the exact stable permutation; TOP_K freezes the exact canonical selected indices.
  Neither averages across equal or cutoff values.
- DROPOUT reuses the same-occurrence canonical mask and inverted-dropout scale.
- Existing compiler predicates are sufficient for dynamic slice/crop/window obligations; no new
  public or internal constraint vocabulary is planned.
- Separate narrow indexing, ordering, and stochastic rule owners keep `LayoutGradientRules` and
  `FirstOrderAutograd` from becoming catch-all formula classes.

## Known limitations

- Index bounds and `NONE` duplicate uniqueness depend on eventual values and remain execution-time
  validity obligations; compiler preflight does not inspect payloads.
- General-axis unfold/fold retain their current static transformed-axis contract.
- First-order policies at non-regular scatter/order boundaries are the selected conventions above;
  they do not claim a unique mathematical derivative.
- Higher derivatives and the final transitive formula audit remain deferred to tasks 0005E–0006.
- No backend currently gains execution support from this compiler-only task.

## Validation evidence

Planning-context evidence:

- Architecture, documentation rules/profiles, planning guide, roadmap, compiler/model plans,
  completed prerequisite tasks, adjoint audits, APIs, glossary, assigned Model kinds/attributes,
  compiler inference/preflight/autograd/rule sources, and focused tests were reviewed.
- Representability audit: passed. Every formula is expressible through the current public Tensor
  surface and canonical producer outputs; no Model or architecture blocker remains.
- Scope audit: passed. The implementation stays in `modules/compiler` plus directly affected
  documentation/planning. The authorized ceiling is 23 paths after adding only
  `docs/api/tensor-api.md` for four stale current-status corrections.

Implementation-context Java evidence:

- `./gradlew :modules:compiler:compileJava :modules:compiler:compileTestJava` passed.
- The early four-suite focus over `LayoutInferenceTest`, `AutogradPreflightTest`,
  `FirstOrderAutogradTest`, and `GradientRulesTest` initially ran 44 tests with three failures.
  All three were stale pre-0005C crop/dynamic-slice expectations; after those tests were updated
  to the implemented 0005C contract, the same four-suite focus passed.
- The first `./gradlew :modules:compiler:test` attempt ran 177 tests with two failures. Both were
  stale `GraphCompilerTest` unsupported-operation expectations, not production defects. After
  those expectations were updated, the final module rerun passed as recorded below.
- The exact six-suite focused command from this task passed.
- Final `./gradlew :modules:compiler:test` passed 25 suites and 177 tests with zero failures,
  errors, or skips.
- After that module run, the three new rule tests were strengthened to compile representative
  formulas through the combined `GraphCompiler` pipeline and assert gradient roles/phases. The
  focused command for `LayoutWindowGradientRulesTest`, `IndexingScatterGradientRulesTest`, and
  `OrderingStochasticGradientRulesTest` passed. No production/executable Java changed after the
  passing final module suite, so the documentation pass reused both results and did not rerun Java
  tests.

Documentation-focused context
`/root/implement_compiler_0005c/docs_compiler_0005c`:

- Independently reviewed the architecture contract, documentation rules and General/API-Javadoc/
  Planning/User-guide/Example profiles, planning guide, roadmap, compiler/model plans, this task,
  Compile/Tensor/Training APIs, glossary, affected source/tests/Javadocs, and the relevant
  autograd/training architecture.
- Finalized all seven affected production Javadocs, the Compile API, glossary compiler-status
  references, this task, both master plans, and the roadmap. The authorized Tensor API expansion
  changed exactly four stale Compiler-0005C status/adoption statements and no Tensor contract.
- Training API: accurate unchanged because 0005C adds package-private compiler formula coverage,
  not a public objective/seed, publication, optimizer, session, prepare, or execution contract.
- Model capabilities and Model source/tests: accurate unchanged outside the preserved 0025C/0025D
  baseline because the task adds no Model semantic, Tensor member, producer-output, or derivative
  ownership.
- Architecture/ADRs/tests, Gradle/build/dependencies, other modules, backend conformance, and
  integration: accurate unchanged because no boundary, dependency, public lifecycle, backend, or
  executable cross-module behavior changed.
- `./gradlew :modules:compiler:javadoc` passed after final Javadoc edits (`BUILD SUCCESSFUL`, seven
  actionable tasks, two executed and five up-to-date).
- `python3 /tmp/validate_synaptik_markdown.py` passed 12 affected Markdown files and 722 local
  links, including heading anchors, balanced fences, final newlines, and trailing whitespace.
- `git diff --check` passed with no output. A separate affected-path trailing-whitespace scan
  returned no matches.
- Final Compiler-0005C scope is exactly 20 paths: seven production, six test, and seven
  documentation/planning paths, below the authorized 23-path ceiling. Preserved Model 0025C/0025D
  source, tests, APIs, task files, capabilities, and planning synchronization were audited
  separately and are not attributed to Compiler 0005C.
- Status synchronization passed: task/master/roadmap record 0005C Complete; 0005D–0005E and 0006
  remain Draft; no detailed specification exists for any later compiler task.

## Implementation notes

- Extended `LayoutInference` for exact `CropToShapeAttrs` extraction versus placement, finite
  signed-length slice bounds, and deferred two-dimensional window-domain constraints.
- Extended layout formulas through length-defined/target-relative slice inverses and exact
  axis/two-dimensional window adjoints.
- Added narrow indexing, ordering, and stochastic rule owners; preflight remains the sole
  allocation-free selector and `FirstOrderAutograd` remains the sole reverse dispatcher.
- Gather/scatter formulas preserve duplicate routing and the fixed replacement/addition,
  zero-count product, and numeric-equality extrema policies.
- SORT creates one exact matching stable ARGSORT; TOP_K and DROPOUT consume canonical
  same-occurrence indices/mask outputs. Index, mask, one-hot, and RNG-state roles remain
  non-differentiable.
- No public API, Model behavior, architecture, dependency, build, backend/runtime behavior, or
  executable Java changed during the documentation pass.

## Completion summary

- Completed changes: Completed the exact assigned layout/window/indexing/scatter/ordering/dropout
  inference, preflight, and first-order formula matrix through the existing one-capture pipeline.
- Files changed or created: Seven compiler production paths, six compiler test paths, and seven
  documentation/planning paths; 20 exact task paths total.
- Tests and validation: Reused the passing six-suite focus and 25-suite/177-test final module
  evidence, plus the later passing three-suite combined-pipeline test strengthening; final
  Javadoc, Markdown, scope, status, whitespace, and diff checks passed.
- Documentation-agent review: Complete in clean context
  `/root/implement_compiler_0005c/docs_compiler_0005c`.
- Documentation impact: Compile and Tensor API status, glossary status references, task, compiler
  and model master plans, and roadmap finalized.
- Javadoc review: All seven affected package-private production contracts finalized.
- Glossary impact: Existing slice/placement entries updated for current compiler adoption; no new
  reusable term was introduced.
- Unresolved issues: None.
- Follow-up required: None. Compiler 0005D–0005E and 0006 remain Draft without specifications.

Status: Complete
