# Recurrent and geometry qualification notes

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/CompileArtifacts.java)

> **Scope:** Model construction, Compiler admission, backend execution, and checkpoint persistence are separate gates. Execution-status sentences retained in this detailed material record earlier qualification slices; they are not an exhaustive current support matrix. The [current backend matrix](../../backend-guide/production-support.md) and its incorporated contracts supersede narrower historical CPU/Metal lists. Package-private implementation sketches are not supported application API.

Those model expressions are compiler-visible inputs to current package-private structural capture,
verification, and the closed floating first-order autograd matrix, including the current product
adjoint built from public prefix and suffix scans. The current fail-closed CPU path advertises,
lowers, prepares, and executes only fully static, resolved-layout CUM_SUM/CUM_PROD occurrences over
FLOAT64, FLOAT32, INT32, or INT64 with a non-scalar Shape, exact input/output
type and Shape, normalized axis, inclusive/exclusive and forward/reverse mode, and injective output.
Gradient support remains limited to the closed matrix above. Dynamic binding, saved-value policy,
and other-backend execution remain planned.
The current Model also exposes fixed `RNN_TANH`, `GRU_RESET_AFTER`, and `LSTM` recurrent-scan
expressions through the advanced low-level static `RecurrentScan.rnn`, `gru`, and `lstm`
namespace. Each occurrence has one `FORWARD` or `REVERSE` attribute, fully static time-major
descriptors, one ordinary non-gradient `INT64[batch]` valid-length input, and two or three outputs
from one exact flat producer. Generic capture preserves that producer as one `CompiledNode`, with
every declared output position and ordered input edge.
Current package-private Compiler forward adoption independently revalidates the complete static
descriptor contract for all three kinds, both directions, and biased or bias-free cardinalities.
It preserves one identity-distinct ordinary node per Model producer, excludes recurrent
occurrences from common-subexpression elimination, and retains every declared sibling output
slot when any slot remains live. Existing publication and Planning orchestration carries the
exact operation and ordered descriptors through one ordinary `OperationCapabilityQuery` without
interpreting recurrence.

Backward-capable modes instead reject the first recurrent occurrence in deterministic complete
forward-producer postorder immediately after allocation-free inventory construction. This happens
before seed validation or any derivative Tensor is created. The first-order coverage boundary
still classifies each of the exact three deferred recurrent signatures with the
unknown/unclassified reason; their disjoint union with the unchanged 128 supported signatures is
the 131-signature pre-Conv3d inventory. No current capability provider, public compiler API,
prepared executable, backend route, runtime length read, or backpropagation-through-time (BPTT)
implementation supports the family.
`Tensor.softmax` and `Tensor.logSoftmax` accept floating input and one positive or negative axis.
Every result retains the exact input Shape, data type, and gradient eligibility, leaves layout
unresolved, and records the requested first-class SOFTMAX or LOG_SOFTMAX kind with exact one-input
provenance. Model construction does not read values, calculate probabilities or logarithms,
select a numerical algorithm, or own a gradient rule or backend lowering. Package-private
capture, verification, and the closed floating first-order autograd matrix are current. The
current fail-closed CPU path advertises, lowers, prepares, and executes only fully static,
resolved-layout FLOAT32/FLOAT64 SOFTMAX/LOG_SOFTMAX occurrences with positive rank
and selected-axis extent, exact Shape/type/gradient-eligibility preservation, non-negative layouts,
and an injective output; CPU execution rejects non-finite represented inputs before output
mutation. Dynamic binding, decomposition, and other-backend execution remain planned.
`Tensor.layerNorm` is current model metadata construction for exact trailing-Shape population
normalization. The no-affine form records `LayerNormAttrs` and ordered input `[input]`; the affine
form records `AffineLayerNormAttrs` and ordered inputs `[input, scale, bias]`. Both retain exact
normalized Shape and typed epsilon parameters, exact input result Shape, one output at index zero,
and no saved-statistic output. Local construction rejects known static mismatches and defers an
unresolved trailing-dimension equality when output Shape is still exact. Package-private capture,
verification, and the closed floating first-order autograd matrix are current. The current fail-
closed CPU path advertises, lowers, prepares, and executes only fully static, resolved-layout
FLOAT32/FLOAT64 input-only or exact `[input, scale, bias]` LayerNorm occurrences
with a positive-rank static trailing Shape, ordered promotion, compatible epsilon metadata, and
injective output. Gradient support remains limited to the closed matrix above. Dynamic binding, saved-
statistic lifetime, decomposition, and other-backend execution remain planned.
`Tensor.rmsNorm` is current model metadata construction for exact trailing-Shape uncentered
root-mean-square normalization. One `RmsNormAttrs(normalizedShape, epsilon)` value supports the
safe ordered inputs `[input]` and `[input, scale]`, each with exactly one output at index zero. The
result retains the exact input Shape; the scaled form requires scale Shape exactly equal to the
normalized Shape. Local construction rejects known static trailing mismatches and defers
unresolved equality. Package-private capture, verification, and the closed floating first-order
autograd matrix are current. The current fail-closed CPU path advertises, lowers, prepares, and
executes only fully static, resolved-layout FLOAT32/FLOAT64 input-only or exact
`[input, scale]` RMSNorm occurrences with a positive-rank static trailing Shape, ordered
promotion, compatible epsilon metadata, and injective output. Gradient support remains limited to
the closed matrix above. Dynamic binding, saved-value policy, decomposition, tolerance policy,
and other-backend execution remain planned.
`Tensor.batchNormInference` is current stateless model metadata construction with exact ordered
inputs `[input, scale, bias, runningMean, runningVariance]` and exactly one output at index zero.
It retains `BatchNormInferenceAttrs(normalizedChannelAxis, epsilon)`, preserves the exact input
Shape, requires rank-one `[C]` affine/statistic vectors, defers unresolved channel-extent equality,
and uses ordered floating promotion with exact-result-typed positive epsilon. The channel axis is
layout-neutral, and running variance is interpreted directly by the formula with epsilon inside
the denominator square root. Package-private capture and verification are current, as is the
closed floating first-order autograd matrix for all five input roles. The current fail-closed CPU
path advertises, lowers, prepares, and executes only fully static, resolved-layout
FLOAT32/FLOAT64 five-input BatchNorm-inference occurrences with rank at least two,
exact channel-axis and `[C]` operand relationships, ordered promotion, compatible epsilon metadata,
and an injective output. Gradient support remains limited to the closed matrix above. Dynamic binding,
saved-value policy, decomposition, tolerance policy, and other-backend execution remain planned.
`Tensor.batchNormTraining` is current five-output model metadata with exact ordered inputs
`[input, scale, bias, runningMean, runningVariance]`. Its producer describes normalized output,
next running mean, next running variance, saved batch mean, and saved inverse standard deviation
in that order. `BatchNormTrainingResult` exposes positions zero through two; positions three and
four remain real producer descriptors that package-private capture preserves. The model records a normalized
channel axis, exact typed new-batch-weight momentum and epsilon, the all-non-channel reduction
domain, biased forward variance, correction-one running variance, output Shapes, types, gradient
eligibility, and indexed provenance. Structural capture of every position and package-private
proof or retention of the `C == 0 || N >= 2` constraint are current; the closed first-order
autograd matrix covers the supported public output roles. The current fail-closed CPU path
advertises, lowers, prepares, and executes only fully static, resolved-layout
FLOAT32/FLOAT64 five-input/five-output BatchNorm-training occurrences with exact
channel, promotion, compatible momentum and epsilon metadata, reduction-domain, output-role, and
injective-output relationships. Gradient roots and stages outside that current matrix remain
unsupported. Dynamic binding, saved-value/publication policy, cross-step statistic ownership, the
Training extension, and other-backend execution remain planned.
`Tensor.meanSquaredError(target, reduction)` is current one-output model metadata with exact
ordered inputs `[prediction, target]`. It records `LossKind.MEAN_SQUARED_ERROR` and one
`MeanSquaredErrorAttrs` carrying explicit `NONE`, `SUM`, or `MEAN` reduction. Local construction
accepts only floating inputs, promotes them in input order, rejects unequal known static Shapes
without broadcasting, and defers unequal pairs involving an unresolved Dimension. `NONE` retains
the exact prediction Shape; `SUM` and `MEAN` use the scalar Shape. Mean divides by the complete
logical element count, so scalar count is one, empty sum is positive zero, and empty mean is NaN.
This compiler-visible requested meaning is structurally capturable, and package-private
verification now revalidates it and proves or retains deferred equality. Current package-private
first-order autograd supports both prediction and target roles, restores `NONE`, `SUM`, and
`MEAN` cotangents through logical Tensor counts, and uses exact typed scalar-operation
coefficients `2` and `-2`. The current fail-closed CPU path advertises, lowers, prepares, and
executes only fully static, resolved-layout FLOAT32/FLOAT64 MSE occurrences with
exact equal prediction/target Shape, ordered promotion, reduction/result Shape, gradient eligibility,
and injective output. Metal additionally owns the admitted homogeneous BFLOAT16/FLOAT16/FLOAT32
canonical positive-rank forward MSE domain for all three reductions; the result gradient flag is
the exact input OR. Generated backward ownership requires each derived occurrence to qualify independently. Dynamic binding,
training coordination, other carriers, mixed carriers, and all other Metal loss or normalization families remain
fail-closed or planned.
`Tensor.categoricalCrossEntropyWithLogits(target, classAxis, reduction)` is current one-output
model metadata with ordered inputs `[logits, target]`. Exact floating target type dispatches to the
unchanged dense target-weighted stable-log-softmax meaning, including floating promotion,
combined gradient eligibility, exact logits/target Shape compatibility, and sample-count mean.
Exact INT32 or INT64 target type dispatches to selected-class negative log-softmax: the target
Shape is constrained to logits Shape without the normalized class axis, result type and gradient
eligibility come only from logits, `NONE` retains exact target Shape, and `MEAN` divides by target
count. The four-argument overload adds one exact target-typed INT32/INT64 `ScalarValue` ignore
index. Matching ignore precedes bounds and logits evaluation, contributes positive zero, and is
excluded from the mean denominator; it is attributes metadata, so provenance remains exactly
`[logits, target]`.

Construction reads no values. Dense target normalization, unresolved mapped Shape equality,
non-ignored index bounds, and class-extent alternatives therefore remain obligations. Empty or
all-ignored index means are NaN and sums are positive zero; with ignore, a zero class extent may
remain valid when every target is ignored. This compiler-visible requested meaning is structurally
capturable. Current package-private first-order autograd supports both dense floating logits and
target roles. It supports only the logits role for index targets, requires a positive static class
depth, clamps ignored targets before one-hot construction, and excludes ignored rows through a
final `where`; dynamic or zero class depth fails closed. The current fail-closed CPU path
advertises, lowers, prepares, and executes only fully static, resolved-layout
FLOAT32/FLOAT64 categorical-loss occurrences with exact normalized class geometry,
dense-floating or INT32/INT64 index target roles, optional exact-typed ignore index,
reduction/result Shape, gradient eligibility, and injective output. Gradient support remains
limited to the exact dense- and index-target roles above. Dynamic binding, publication/training
coordination, and other-backend execution remain planned.
`Tensor.contiguous()` accepts every current data type and preserves the exact Shape, data type, and
gradient eligibility. It creates new canonical dense row-major, zero-offset layout geometry for a
fully static Shape and leaves a dynamic Shape unresolved. Every call is fresh, unlabeled, and
storage-free, records `CONTIGUOUS` with the canonical no-attributes singleton and exact one-input
provenance, and does not inspect input layout, storage, or values. Resolved result geometry does
not allocate or copy storage. Package-private structural capture is current. The current fail-
closed CPU path advertises, lowers, prepares, and executes only fully static, resolved-layout
`CONTIGUOUS` occurrences with exact type/Shape preservation and canonical dense output, within a
bounded one-through-eight-node affine unit. Metal also executes strict static positive-rank
`FLOAT32` `CONTIGUOUS` when its input is canonical or a locally produced affine view. Dynamic
binding outside the ordinary named-B FLOAT32 rank-two closure and redundant-request
canonicalization remain planned. The qualified closure materializes actual logical counts.
`Tensor.reshape(long...)` accepts all current data types, normalizes an empty request or one
inferable `-1`, and rejects locally provable invalid counts. `Tensor.reshape(Shape)` retains an
exact normalized target and defers count equality when either Shape is dynamic. Both overloads
retain type and gradient eligibility, record exact RESHAPE/target-shape semantics with one-input
provenance, and stay unlabeled and storage-free. Only resolved contiguous input plus a static
target produces same-offset canonical view metadata; all other result layout remains unresolved.
Package-private structural capture can preserve this current model expression. The current fail-
closed CPU path advertises, lowers, prepares, and executes only fully static, resolved-layout
`RESHAPE` occurrences with contiguous input, equal checked element count, exact same-offset
canonical view metadata, and membership in a bounded one-through-eight-node affine unit. Metal
also executes this strict static `FLOAT32` relationship for a canonical or contiguous locally
produced input inside one maximal composed partition. Dynamic constraint solving, reshape-chain
canonicalization, and non-view materialization remain planned.
`Tensor.expand(long...)` treats every requested extent as a literal non-negative dimension, while
`Tensor.expand(Shape)` retains the exact target reference. Both overloads require target rank at
least input rank and allow new leading target axes. Structural equality and a static source
singleton are immediately compatible. A fully static unequal aligned pair is rejected when the
source is not one; zero is an ordinary static extent. If either aligned dimension is unresolved,
Model construction now retains the exact target and the later condition
`source == 1 || source == target`, leaves layout unresolved, and creates no
`DeferredGraphConstraint`. A fully static target and resolved input layout otherwise produce new
same-offset view geometry with preserved aligned strides and zero strides for leading or
expanded-singleton axes. Every result retains exact type and gradient eligibility, records exact
EXPAND/target-shape semantics with one-input provenance, and remains fresh, unlabeled, and
storage-free.

Package-private compiler inference now adopts that broadened occurrence. For each non-equal,
non-static-source-singleton unresolved aligned pair, it emits the occurrence-owned predicate
`AnyOf(DimensionEqual(source, 1), DimensionEqual(source, target))` in aligned-axis order and
leaves layout unresolved. Floating EXPAND autograd constructs
`gradient.sumToShape(input.descriptor().shape())`; preflight admits the route only when that
inverse uses the same forward predicate relationship. Binding-dependent `SUM_TO_SHAPE` similarly
inverts through `gradient.expand(input.descriptor().shape())` under the matching
target-one-or-target-equal-source obligation. Existing Model reduction semantics are unchanged.
This compiler contract binds no dimension and claims no value repetition, storage aliasing, or
materialization. The current fail-closed CPU path advertises, lowers, prepares, and executes only
fully static, resolved-layout `EXPAND` occurrences with the exact singleton-expansion Shape and
zero-stride view relationship, within a bounded one-through-eight-node affine unit. Metal also
executes that exact static `FLOAT32` relationship after a canonical or local affine producer.
Ordinary named-B FLOAT32 EXPAND is also current within the qualified rank-two closure; general
dynamic binding, wider dynamic types and broader materialization remain planned.
`Tensor.permute(int...)` accepts every current data type, requires a complete output-to-input axis
mapping, normalizes each negative axis once, and reorders exact Dimension references. Any resolved
input layout produces a new same-offset view descriptor with exact reordered strides; unresolved
input layout remains unresolved. `Tensor.transpose()` requires rank two and uses the same PERMUTE
construction with normalized axes `[1, 0]`. Every result preserves type and gradient eligibility,
records exact normalized attributes and one-input provenance, and remains fresh, unlabeled, and
storage-free. Package-private structural capture can preserve the occurrence. The current fail-
closed CPU path advertises, lowers, prepares, and executes only fully static, resolved-layout
`PERMUTE`/rank-two transpose occurrences with the exact same-offset reordered Shape/stride
relationship, within a bounded one-through-eight-node affine unit. Metal also executes that strict
static `FLOAT32` relationship after a canonical or local affine producer. Permutation
canonicalization, dynamic binding outside the qualified ordinary named-B FLOAT32 rank-two
PERMUTE closure, and broader materialization remain planned.
`Tensor.expandDims(int)` inserts one static singleton at a caller position normalized against the
result rank. `Tensor.squeeze(int)` instead normalizes an existing input axis and removes it only
when its dimension is statically known as one. Both preserve exact type and gradient eligibility,
retain exact unaffected Dimension references, record matching `AxisTransformAttrs` and one-input
provenance, and leave unresolved input geometry unresolved. Resolved input geometry produces one
new same-offset logical view descriptor with a deterministic checked stride inserted or the
selected stride removed. Package-private structural capture can preserve the occurrence. The
current fail-closed CPU path advertises, lowers, prepares, and executes only fully static,
resolved-layout `EXPAND_DIMS`/`SQUEEZE` occurrences with the exact inserted/removed singleton
Shape and same-offset stride relationship, within a bounded one-through-eight-node affine unit.
Metal also executes those exact static `FLOAT32` relationships after a canonical or local affine
producer. Gradient support remains the exact current Compiler matrix above. Dynamic singleton
solving, inverse-pair canonicalization, and broader materialization remain planned. The bounded
ordinary named-B FLOAT32 closure independently supports EXPAND_DIMS with a fixed inserted one;
this does not grant symbolic SQUEEZE or dynamic singleton solving.
`Tensor.slice(long[], long[], int[], long[])` clones four parallel request arrays, normalizes raw
axes and bounds against selected static dimensions, clamps bounds by step direction, derives a
same-rank Shape, and records normalized start/length/axis/signed-step sequences in `SliceAttrs`.
Empty arrays and zero-extent results are valid. Resolved non-empty all-positive geometry produces
checked start-adjusted, step-multiplied logical view metadata; unresolved input, empty results, and
any negative step remain layout-unresolved. `Tensor.sliceAxis(int, long, long)` supplies one
step-one entry, its four-argument overload supplies an explicit signed step, and `Tensor.flip`
creates one negative-step SLICE occurrence for explicit axes. Empty flip axes mean identity.
Every success preserves exact type and gradient eligibility, records one identity-distinct
producer with exact `[input]`, one output descriptor, and provenance index zero, and remains fresh,
unlabeled, and storage-free. Package-private structural capture can preserve the occurrence.
Closed first-order autograd currently constructs the normalized floating SLICE cotangent described
above; unsupported types and non-normalized metadata fail during preflight. The current fail-
closed CPU path advertises, lowers, prepares, and executes only fully static, resolved-layout
positive-step `SLICE` occurrences with exact Shape, offset/stride, and finite-coordinate
relationships, within a bounded one-through-eight-node affine unit. Negative-step extraction,
dynamic binding, slice/flip canonicalization, materialization, and other-backend execution remain
planned.
`Tensor.sliceByLength(starts, lengths, axes, steps)` is current Model expression construction. It
records the same exact `SLICE`/`SliceAttrs` occurrence after proving non-negative first/final
coordinates and every statically decidable selected-extent upper bound. Selected result
Dimensions are exact static lengths; unaffected Dimension references remain exact. An unresolved
selected extent defers only its upper-bound comparison, and Model stores no constraint object.
Empty or negative-step results remain layout-unresolved; resolved input geometry produces a view
only for a non-empty all-positive request. Current compiler inference derives the exact selected
lengths and proves or retains each non-empty signed region's upper-bound obligation. Fail-closed
preflight validates the exact occurrence before current floating autograd uses that same
length-defined form to invert SLICE_UPDATE without requiring a static selected base extent.
The current fail-closed CPU path also advertises, lowers, prepares, and executes only the fully
static, resolved-layout positive-step length-defined `SLICE` subset with exact finite bounds and
affine view geometry. Dynamic selected-extent binding and other-backend execution remain planned.
`Tensor.sliceUpdate(update, starts, axes, steps)` is now current model construction for functional
signed multi-axis replacement. It derives normalized finite `SliceAttrs` lengths from selected
static update Dimensions, requires exact base/update type and same-rank Shape compatibility,
retains the exact base Shape, combines eligibility, leaves layout unresolved, and records one
`SLICE_UPDATE` occurrence with ordered `[base, update]` provenance. Static first/final coordinate
bounds fail locally. A non-negative start on an unresolved base axis retains its upper-bound
obligation for later binding or execution; no start is clamped or shifted to fit.
`Tensor.cropToShape(targetShape, prefixShape)` is also current model construction. It records one
`SLICE` occurrence with exact `CropToShapeAttrs`, retains the supplied target Shape as the result,
and interprets the prefix Shape as per-axis logical extents preceding the region. Fully static
`prefix + target <= input` bounds are checked locally; inequalities involving any unresolved
Dimension remain deferred. Both primitives are structurally capturable and neither mutates values.
Closed first-order autograd currently constructs guarded floating cotangents for normalized SLICE
and for both normalized SLICE_UPDATE data roles. The update role uses exact recorded lengths, so
unresolved selected base extents are supported by the Compiler when their bounds can be proved or
retained. Compiler inference also distinguishes target-relative extraction from placement,
preserves exact Shapes, and retains unresolved bounds. The current fail-closed CPU path
advertises, lowers, prepares, and executes only fully static, resolved-layout target-relative
`SLICE` and one-node signed-step `SLICE_UPDATE` occurrences with exact finite placement, matching
types and ranks, base-shaped distinct injective output, and no input mutation. Dynamic binding,
saved-value policy, canonicalization, and other-backend execution remain planned.
`Tensor.sliceUpdate(update, prefixShape)` is also current Model expression construction. It
records exact `SLICE_UPDATE`/`CropToShapeAttrs` with ordered inputs `[base, update]`, retains the
exact update Shape as the target region and the exact caller prefix Shape, and returns the exact
base Shape/type with eligibility OR and unresolved layout. A fully static axis proves checked
`prefix + update <= base`; if any base, prefix, or update extent is unresolved, Model defers that
whole axis fit without partial arithmetic or a constraint object. Structural capture can preserve
the occurrence. Current compiler inference distinguishes both slice kinds and both exact
attributes variants, retains `prefix + target <= base` obligations, and validates update Shape
against the exact target. Floating autograd uses target-relative placement for the source/base
cotangent and exact target-relative crop for the update cotangent. The current fail-closed CPU
path advertises, lowers, prepares, and executes only the fully static, resolved-layout one-node
target-relative `SLICE_UPDATE` subset with exact prefix/target fit, matching types and ranks, and
a base-shaped distinct injective output. Dynamic binding, materialization, and other-backend
execution remain planned.
`Tensor.select(int, long)` normalizes one source axis and one scalar coordinate, removes the
selected Dimension, preserves every unaffected exact Dimension reference, and records normalized
`SelectAttrs` with exact one-input provenance. A static selected extent supplies immediate
negative-index normalization and bounds checks; a non-negative coordinate on a dynamic selected
extent remains representable with its upper bound deferred, while a negative coordinate is
rejected. Resolved input geometry with a non-empty result produces checked selected-stride removal
and offset advancement in one new logical view descriptor; unresolved input and empty results stay
unresolved. The fresh result preserves exact type and eligibility and has no label or storage.
Package-private structural capture can preserve this expression. Closed first-order autograd
currently constructs the guarded floating SELECT cotangent described above. The current fail-
closed CPU path advertises, lowers, prepares, and executes only fully static, resolved-layout
`SELECT` occurrences with an exact in-bounds coordinate, selected-axis removal, same-offset
stride relationship, and membership in a bounded one-through-eight-node affine unit. Dynamic-
coordinate binding, materialization, and other-backend execution remain planned.
`Tensor.gather` and `gatherElements` consume exact ordered `[data, indices]` inputs with `INT32` or
`INT64` indices. They normalize one data axis and apply canonical axis-replacement or same-rank
aligned-Shape rules. Every fresh result retains data type and gradient eligibility, leaves layout
unresolved, and records exact two-input provenance. Package-private structural capture can
preserve the occurrence. Construction interprets no index values and adds no Model-owned gradient
rule. Current compiler autograd routes a floating data cotangent through matching `scatterAdd` or
additive `scatterElements`; indices remain non-differentiable. The current fail-closed CPU path
advertises, lowers, prepares, and executes fully static, resolved-layout GATHER/GATHER_ELEMENTS
occurrences with INT32/INT64 indices, exact family Shape alignment, preserved data type, and
injective output. The current Metal path additionally admits exact canonical positive-rank
FLOAT32+INT32 GATHER, with no GATHER_ELEMENTS or INT64 support, and validates
every index before selector dispatch or target writes. Dynamic binding remains planned.
`Tensor.gatherNd` consumes exact ordered `[data, indices]` inputs with `INT32` or `INT64` indices.
Its short form uses zero shared batch Dimensions; its complete form retains one normalized
non-negative batch count. Construction validates both ranks, structurally equal leading batch
Dimensions, and a statically known positive tuple depth from the final indices Dimension. It
derives the result as the indices prefix without tuple depth followed by the untouched data
suffix, including canonical scalar and exact retained Dimension references. Every result is fresh,
unlabeled, storage-free, and unresolved-layout, preserves data type and gradient eligibility, and
records `GATHER_ND` with exact `[data, indices]` provenance. Package-private structural capture can
preserve the occurrence. Construction reads no index value and adds no Model-owned gradient.
Current compiler autograd routes a floating data cotangent through matching additive Scatter-ND;
coordinate tuples remain non-differentiable. The current fail-closed CPU path advertises, lowers,
prepares, and executes only fully static, resolved-layout GATHER_ND occurrences with INT32/INT64
indices, static positive tuple depth, exact batch/prefix/suffix Shape geometry, preserved data
type, and injective output. Dynamic binding and other-backend execution remain planned; run-bound
tuple checks remain at the CPU execution boundary.
Both `Tensor.scatterElements` overloads consume exact ordered `[data, indices, updates]` inputs.
They require `INT32` or `INT64` indices and exact matching data/update types. `NONE` is permitted
for every current type and arithmetic reductions for floating or integral values. Each method
normalizes one raw data axis, validates the same-rank Shape relationship, and returns a fresh
result with the exact data Shape/type, data/update gradient-
eligibility OR, unresolved layout, and exact three-input provenance. Construction reads no values,
checks no index bound or duplicate target, mutates no input, and performs no write or reduction.
Package-private structural capture can preserve the occurrence. Current compiler autograd
supports both floating data roles for every current reduction: NONE masks addressed base
positions, ADD preserves the base, MUL uses the zero-count/safe-product policy, and MIN/MAX share
among exact numeric-equality winners. The current fail-closed CPU path advertises, lowers,
prepares, and executes only fully static, resolved-layout SCATTER_ELEMENTS occurrences with
INT32/INT64 indices, exact data/update type and same-rank Shape relationships, permitted
represented reduction, data-shaped distinct injective output, and execution-time index/duplicate
validation. Gradient support remains limited to the exact floating roles above. The current Metal
path additionally admits only canonical positive-rank FLOAT32 data/updates/results with INT32
indices and `NONE` under the one Model contract. It validates the complete bounds domain before
complete-coordinate uniqueness and before dispatch or writes, preserves exact addressed-update and
unaddressed-base bits, and leaves every input unchanged. This does not add Compiler production or a
complete Metal backward route: the existing update cotangent needs unsupported GATHER_ELEMENTS.
Dynamic binding and every other Metal scatter type/reduction remain planned.
For `MUL`, `MIN`, and `MAX`, the current Model contract combines the base exactly once with every
addressed update exactly once, counts duplicate targets as distinct contributions, and preserves
the exact representation of an unaddressed base coordinate. Its floating and integral result is
independent of encounter, layout, stride, atomic, tree, and backend order. Compiler work must
preserve that represented-value contract; it may not infer a fixed accumulation sequence or add a
derivative or subgradient policy to the Model.
`Tensor.scatterAdd(indices, updates, axis)` is now the current Gather-compatible fixed-add model
primitive. It records `SCATTER_ADD` with normalized `IndexAxisAttrs` and ordered
`[data, indices, updates]` inputs. Updates must have the exact Shape that `data.gather(indices,
axis)` would produce, while the functional result retains the exact data Shape and accumulates
duplicate targets. Construction validates metadata only and leaves layout unresolved.
Package-private structural capture can preserve the occurrence. Current compiler autograd uses
this primitive for floating GATHER data cotangents and preserves its Shape, duplicate
accumulation, and eventual index-bounds obligations. The current fail-closed CPU path advertises,
lowers, prepares, and executes only fully static, resolved-layout SCATTER_ADD occurrences with
INT32/INT64 indices, exact Gather-compatible update Shape, fixed represented addition,
data-shaped distinct injective output, and execution-time index validation. Dynamic binding and
other-backend execution remain planned.
The three `Tensor.scatterNd` overloads consume exact ordered `[data, indices, updates]` inputs with
`INT32` or `INT64` indices and exact matching data/update types. Their defaults select
`ScatterReduction.NONE` and zero shared batch Dimensions; complete construction retains the exact
supplied reduction and normalized non-negative batch count. Construction validates reduction
eligibility, both ranks, structurally equal leading batch Dimensions, static positive tuple depth
from the final indices Dimension, and the exact indices-prefix-plus-data-suffix updates Shape.
Every result is fresh, unlabeled, storage-free, and unresolved-layout, retains the exact data
Shape/type, combines data/update gradient eligibility, and records `SCATTER_ND` with exact
three-input provenance. Package-private structural capture can preserve the occurrence. Model
construction reads no index or update value, mutates no input, and adds no Model-owned gradient.
Current compiler autograd supports both floating data roles for NONE, ADD, MUL, MIN, and MAX
through exact matching Gather-ND/Scatter-ND geometry; tuple indices remain non-differentiable. The
current fail-closed CPU path advertises, lowers, prepares, and executes only fully static,
resolved-layout SCATTER_ND occurrences with INT32/INT64 indices, static positive tuple depth,
exact batch/prefix/suffix Shape and data/update type relationships, permitted represented
reduction, data-shaped distinct injective output, and execution-time index/duplicate validation.
Gradient support remains limited to the exact floating roles above. Dynamic binding and other-
backend execution remain planned.
For arithmetic reduction, every tuple contributes its complete update suffix slice scalar by
scalar. Current Model `MUL`, `MIN`, and `MAX` semantics include each target's base and every
addressed scalar exactly once, count duplicate tuples as distinct contributions, preserve the
exact representation of an unaddressed coordinate, and are independent of encounter, layout,
stride, atomic, tree, and backend order. `NONE`, configurable `ADD`, and fixed `SCATTER_ADD`
remain unchanged. Compiler work must preserve this forward contract without assigning a Model
derivative or subgradient policy.
Static `Tensor.concat(int, Tensor...)` and `Tensor.stack(int, Tensor...)` snapshot ordered non-empty
inputs, normalize an existing or inserted axis, enforce exact type and operation-specific Shape
rules, and create fresh unresolved-layout results with eligibility OR and exact ordered provenance.
Instance `Tensor.unstack(int)` requires a static `int`-sized selected extent and returns an
immutable ordered List of scalar `SELECT` expressions after upfront count validation. Each fresh
result removes that axis, has an independent one-output producer, and uses provenance index zero
over the same input; its `SelectAttrs` stores the coordinate. A zero extent returns no result or
ID. Package-private structural capture preserves each independent occurrence. Closed first-order
autograd currently constructs guarded floating CONCAT and STACK cotangents, including STACK
occurrences produced by repeated SELECT operations during `unstack`. The current fail-closed CPU
path advertises, lowers, prepares, and executes only one fully static, resolved-layout CONCAT or
STACK occurrence with one through sixteen same-typed semantic input occurrences, exact axis and
Shape relationships, preserved semantic order, and a distinct injective output. Dynamic binding,
grouping/decomposition, ONNX mapping, arbitrary larger composition, and other-backend execution
remain planned.
`Tensor.unfold`, `Tensor.foldAxis`, both `Tensor.unfold2d` forms, and `Tensor.fold2d` construct the
current public storage-free window-transform expressions. General-axis fold restores an explicit
target extent under overlap summation. The 2D forms preserve canonical rank-three im2col/col2im,
retain exact static or symbolic channel/spatial formulas, and distinguish direct conceptual-zero
padding from an exact typed padding scalar. `fold2d` accepts only complete structural formula
matches rather than recording equality between unrelated unresolved symbols. Every result
preserves input data type and gradient eligibility, leaves layout unresolved, and records exact
one-input provenance. Package-private structural capture can preserve these occurrences. Current
compiler inference retains unresolved two-dimensional height/width domain constraints, and
current floating autograd uses each exact public inverse or overlap-add transformation. The
current fail-closed CPU path advertises, lowers, prepares, and executes fully static,
resolved-layout UNFOLD_AXIS occurrences over all nine represented types and
UNFOLD2D/UNFOLD3D over all six floating types; FOLD_AXIS over floating and integral types,
and FOLD2D/FOLD3D over floating types, with exact window, direct-zero or typed-padding,
overlap-add, Shape and distinct injective-output relationships. Low folds accumulate in FLOAT32
and narrow once at each final output; raw unfold preserves represented bits.
Metal independently admits canonical all-carrier UNFOLD_AXIS within its bounded rank/window
domain and static canonical four-floating-type UNFOLD2D/UNFOLD3D raw movement.
FOLD_AXIS and FLOAT64/BFLOAT16/FLOAT16 image folds require non-overlap; FLOAT32
FOLD2D/FOLD3D additionally accumulate overlapping contributors deterministically with one
output writer, no atomics, scratch or host fallback/repair. Raw routes preserve represented
bits; folds write represented positive zero to uncovered output cells. Gradient support
requires the entire generated topology; dynamic and other unlisted Metal window routes remain
unsupported. See the [backend-qualified production matrix](../../backend-guide/production-support.md).
That origin metadata is now traversed by the package-private structural capture described above.
The internal step observes that two result Tensors belong to the same expression occurrence only
when their provenance carries the same exact `TensorProducer` reference. Different output indices
then select different output positions of one captured node. The producer does not prescribe
graph-local identity: each capture assigns fresh `NodeId` and `ValueId` sequences from its own
deterministic traversal.

Dropout is the first current production occurrence that uses this shared-output seam. Its public
output selects producer slot zero and its opaque next-state wrapper retains slot two. The producer
also describes slot one as a same-Shape BOOL auxiliary keep mask even though `DropoutResult` exposes
no mask Tensor. Current internal capture creates graph values for all three slots by reading the
reachable producer's ordered descriptors and also preserves the zero-input initial-state producer
and its edge into dropout. It needs no public sibling-output lookup. Current floating values-slot
autograd retrieves the exact canonical mask wrapper at producer slot one and constructs
`where(mask, g / (1 - probability), zero)`. It never resamples, infers a mask from values, or
advances/differentiates graph RNG state. The current Metal Task-0065 path gives all three forward
outputs ordinary run-local storage and keeps the mask live through that exact generated WHERE;
other backend-specific saved-value policies remain separately owned.

Explicit attention output-and-weights construction uses the same foundation without an auxiliary
or hidden output. `ScaledDotProductAttentionResult.output()` and `weights()` expose the exact
wrappers for producer slots zero and one. If either wrapper is requested, current internal capture
emits their shared producer once and creates both ordered graph values. Attention operand
revalidation and adjoints are current Compiler responsibilities; backward preservation and
saved-value lifetime remain planned. The current fail-closed CPU path separately advertises,
lowers, prepares, and executes only the fully static, resolved-layout forward attention subset
described above. It does not alter Compiler capture, inference, gradient formulas, or this API's
backend-neutral contract, and it does not imply dynamic or other-backend execution.


---

[Previous topic](operation-qualification.md) · [Next topic](configuration.md)
