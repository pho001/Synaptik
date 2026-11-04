# Task 0004A: Exact-composition [Gradient](../../../../glossary.md#term-gradient)-rule Extensions

> **Editorial reading key — not part of the historical record.**
>
> Exact composition means Compiler expresses each admitted derivative using existing ordinary
> Tensor operations, not that compilation evaluates the derivative numerically. Preflight checks
> the selected input role and whether the whole formula can be constructed before derivative
> identities are allocated. Repeated operand positions contribute separately even when they refer
> to the same Tensor. Unsupported symbolic inverses remain rejected rather than guessed from
> hypothetical runtime values.
>
> Current reading: [pre-capture differentiation and functional requests](../../../../reference/compiler/autograd.md#current-package-private-pre-capture-autograd); [descriptor verification and deferred constraints](../../../../reference/compiler/verification-scan.md#current-package-private-verification-inference).

## Status

Complete

## Goal

Extend [Compiler](../../../../glossary.md#term-compiler) 0004's proved compiler-owned pre-[capture](../../../../glossary.md#term-capture-meanings) automatic-differentiation pipeline with
one bounded set of first-order [derivative](../../../../glossary.md#term-derivative) rules whose formulas are exact compositions of current
public [`Tensor`](../../../../glossary.md#tensor) operations.

This task adds no derivative convention. Every selected row must be valid without choosing a
tie, endpoint, discontinuity, singularity, empty-domain, [NaN](../../../../glossary.md#term-nan)/[infinity](../../../../glossary.md#term-infinity), stochastic, saved-value,
mixed-floating [cotangent](../../../../glossary.md#cotangent), or numerical-[kernel](../../../../glossary.md#kernel) policy. Unsupported variants continue to fail in
`AutogradPreflight` before any derivative Tensor [identity](../../../../glossary.md#term-identity-meanings) is allocated.

The task reuses the existing [objective](../../../../glossary.md#term-objective), [target](../../../../glossary.md#term-target), implicit [seed](../../../../glossary.md#term-seed), object-identity accumulation,
logical-splat, one combined capture, [graph phase](../../../../glossary.md#graph-phase), [validation](../../../../glossary.md#term-validation), optimization, and
`GraphCompilation` contracts unchanged.

## [Scope](../../../../glossary.md#term-scope)

### Pipeline reuse

Do not redesign [Compiler](../../../../glossary.md#term-compiler) 0004. Extend only its closed [preflight](../../../../glossary.md#term-preflight) and named rule [dispatch](../../../../glossary.md#term-dispatch):

```text
original forward Tensor DAG
  -> complete fail-closed preflight
  -> ordinary public-Tensor derivative formulas
  -> existing deterministic identity-keyed accumulation
  -> existing one phase-aware combined capture
  -> existing validation and exact combined optimization
```

`GraphCompiler`, `GraphCapture`, `GraphCompilation`, the request [shape](../../../../glossary.md#shape), the implicit [scalar](../../../../glossary.md#term-scalar) unit
[seed](../../../../glossary.md#term-seed), result roles, [graph](../../../../glossary.md#graph)-output ordering, and optimization pass order remain behaviorally
unchanged. A successful 0004A formula is an ordinary [differentiable](../../../../glossary.md#derivative-and-chain-rule) [Tensor](../../../../glossary.md#tensor) expression; this task
does not claim that Compiler 0006's future higher-order request contract is complete.

### Closed `SUPPORTED_0004A` [matrix](../../../../glossary.md#term-matrix-meanings)

`SUPPORTED_0004A` is additive to `SUPPORTED_0004`. No existing row is weakened. In the formulas
below:

- `g` is the selected output [cotangent](../../../../glossary.md#cotangent);
- `x`, `left`, and `right` are exact original inputs;
- `Z_x` is the existing explicit typed positive-zero [logical splat](../../../../glossary.md#logical-splat) expanded to `x.shape()`;
- `B_x(v)` is `v.sumToShape(x.shape())`;
- `swapLastTwo(v)` is `v.permute(...)` with only its final two [axes](../../../../glossary.md#term-axis) exchanged; and
- all operation kinds, attribute objects, input positions, output positions, descriptors, and
  Shapes are the exact facts recorded by preflight.

| Family and exact variant | Additional [preflight](../../../../glossary.md#term-preflight) [guard](../../../../glossary.md#term-guard) | Formula for selected [differentiable](../../../../glossary.md#derivative-and-chain-rule) roles |
|---|---|---|
| `UnaryElementwiseKind.ERF` with `NoOperationAttrs.INSTANCE` | Input and output have one exact equal floating type and [Shape](../../../../glossary.md#shape). | `g.mul(x.mul(x).neg().exp()).mul(C_erf)`, where `C_erf` is the exact typed representation of `2 / sqrt(pi)` specified below. |
| `AggregateReductionKind.SUM` with `MaskedReductionAttrs(axis)` | Data input and output have one exact equal floating type; [mask](../../../../glossary.md#term-boolean-mask) input is exact [BOOL](../../../../glossary.md#data-type--datatype); the current mask-broadcast and normalized-axis [descriptor](../../../../glossary.md#term-descriptor) contract holds. | Restore the removed axis in `g`, expand to `x.shape()`, then `Tensor.where(mask, restored, Z_x)`. The mask role is non-differentiable. |
| `AggregateReductionKind.SUM` with `SumToShapeAttrs(targetShape)` | Input and output have one exact equal floating type. For every right-aligned pair, the [target](../../../../glossary.md#term-target) [Dimension](../../../../glossary.md#dimension) is structurally equal to the input Dimension or is static one. Leading input axes are permitted. [Binding](../../../../glossary.md#term-binding)-dependent unresolved [singleton](../../../../glossary.md#term-singleton)-or-equal pairs are rejected. | `g.expand(x.shape())`. |
| `MatmulKind.MATMUL` with `NoOperationAttrs.INSTANCE` | Both inputs and output are floating; the selected target input has the output's exact floating type; all current [rank](../../../../glossary.md#term-rank), [contraction](../../../../glossary.md#contraction-dimension), promotion, and [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-broadcast descriptor facts hold. | The complete rank-case formulas below, followed by `B_left` or `B_right` where batch [broadcasting](../../../../glossary.md#broadcasting) may have occurred. |
| `SliceKind.SLICE` with `SliceAttrs` | Input and output have one exact equal floating type; current normalized [finite](../../../../glossary.md#term-finite) [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) and Shape facts hold. | `Z_x.sliceUpdate(g, starts, axes, steps)`. |
| `SliceKind.SLICE_UPDATE` with `SliceAttrs`, base role | Base, update, and output have one exact equal floating type and the current normalized replacement contract holds. | `g.sliceUpdate(Z_update, starts, axes, steps)`. |
| `SliceKind.SLICE_UPDATE` with `SliceAttrs`, update role | Same guard as the base role, and every selected base Dimension is static so the normalized finite coordinate sequences can be expressed by public [`slice`](../../../../glossary.md#slice). | Extract the exact normalized coordinate sequences from `g` with public `slice`, using the deterministic raw-end reconstruction below. |
| `SelectKind.SELECT` with `SelectAttrs(axis, index)` | Input and output have one exact equal floating type and the current normalized select facts hold. | `Z_x.sliceUpdate(g.expandDims(axis), [index], [axis], [1])`. |
| `PadKind.PAD` with `PadAttrs` | Input and output have one exact equal floating type; the typed [padding](../../../../glossary.md#padding) value is non-differentiable [metadata](../../../../glossary.md#term-metadata). | `g.cropToShape(x.shape(), Shape.of(beforeWidths))`. |
| `TileKind.TILE` with `TileAttrs` | Input and output have one exact equal floating type and every repeat is positive. | Reshape `g` to the interleaved Shape `[repeat0, x0, repeat1, x1, ...]`, sum the even repeat axes, then reshape to the exact `x.shape()`. [Scalar](../../../../glossary.md#term-scalar) tile returns `g`. |
| `TensorCompositionKind.CONCAT` with `CompositionAxisAttrs(axis)` | Every input and output has one exact equal floating type and the current same-rank/non-axis-Shape contract holds. | For input `i`, `g.cropToShape(input_i.shape(), prefix_i)`, where `prefix_i` is zero on other axes and the exact ordered symbolic sum of prior input extents on `axis`. |
| `TensorCompositionKind.STACK` with `CompositionAxisAttrs(axis)` | Every input and output has one exact equal floating type and the current exact input-Shape contract holds. | For input `i`, `g.select(axis, i)`. |

The [`MATMUL`](../../../../glossary.md#matrix-multiplication--matmul) rank cases are exact:

| Left rank | Right rank | Left contribution | Right contribution |
|---|---|---|---|
| 1 | 1 | `g.mul(right)` | `g.mul(left)` |
| 1 | at least 2 | `B_left(g.expandDims(g.rank() - 1).matmul(swapLastTwo(right)).squeeze(g.rank() - 1))` | `B_right(left.expandDims(1).matmul(g.expandDims(g.rank() - 1)))` |
| at least 2 | 1 | `B_left(g.expandDims(g.rank()).matmul(right.expandDims(0)))` | `B_right(swapLastTwo(left).matmul(g))` |
| at least 2 | at least 2 | `B_left(g.matmul(swapLastTwo(right)))` | `B_right(swapLastTwo(left).matmul(g))` |

Preflight is role-aware. A `MATMUL` [occurrence](../../../../glossary.md#term-occurrence) may therefore propagate to a selected same-type
input even when the unselected floating operand was promoted. It rejects a selected operand whose
cotangent would require a cross-floating conversion. Integral `MATMUL` remains
non-differentiable.

For `SLICE_UPDATE`'s update role, reconstruct each public `slice` entry from normalized
`SliceAttrs` without changing its coordinate sequence:

1. zero length uses raw start zero and raw end zero;
2. positive step uses the first coordinate after the finite sequence when representable and no
   greater than the static base extent, otherwise the base extent;<span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent))</span>
3. negative step uses the first coordinate after the finite sequence when non-negative, otherwise
   raw `-baseExtent - 1`, which normalizes to the clamped conceptual end `-1`; and
4. checked arithmetic is mandatory and any contradiction is a preflight failure, not an
   alternative formula.

This conversion is internal formula construction from already normalized [immutable](../../../../glossary.md#term-immutable) attributes.
It does not add a [model](../../../../glossary.md#term-model) [operation](../../../../glossary.md#operation) or bypass the public `Tensor.slice` contract.

### Typed [`ERF`](../../../../glossary.md#term-erf) coefficient

`C_erf` is a [scalar](../../../../glossary.md#term-scalar) [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) attribute used by public scalar `mul`; it is not a new [Tensor](../../../../glossary.md#tensor)
[leaf](../../../../glossary.md#term-leaf), [constant](../../../../glossary.md#term-constant)-sidecar [binding](../../../../glossary.md#term-binding), [eager](../../../../glossary.md#term-eager-evaluation) value, or numerical implementation choice. Use these fixed
[correctly rounded](../../../../glossary.md#nan-infinity-and-signed-zero) representations of `2 / sqrt(pi)`:

| [Data type](../../../../glossary.md#data-type--datatype) | Exact representation |
|---|---|
| [`BFLOAT16`](../../../../glossary.md#term-bfloat16) | [raw bits](../../../../glossary.md#storage-mapping-contiguity-and-injectivity) `0x3F90` |
| [`FLOAT32`](../../../../glossary.md#term-float32) | raw bits `0x3F906EBB` |
| [`FLOAT64`](../../../../glossary.md#term-float64) | raw bits `0x3FF20DD750429B6D` |

Do not compute the coefficient from [host](../../../../glossary.md#term-host) transcendental functions during [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names). Preserve
the existing generated zero/one logical-splat rules unchanged; no additional splat source is
needed for this scalar [operation](../../../../glossary.md#operation) attribute.

### Rule [ownership](../../../../glossary.md#term-ownership-meanings)

Keep the package-private named [owners](../../../../glossary.md#term-owner) narrow:

- `ElementwiseGradientRules` gains only [`ERF`](../../../../glossary.md#term-erf);
- `ReductionGradientRules` gains masked `SUM` and the guarded `SUM_TO_SHAPE` variant;
- `LayoutGradientRules` gains [`SLICE`](../../../../glossary.md#slice), role-aware `SLICE_UPDATE`, `SELECT`, [`PAD`](../../../../glossary.md#padding), `TILE`,
  `CONCAT`, and `STACK`; and
- new `LinearAlgebraGradientRules` owns only [`MATMUL`](../../../../glossary.md#matrix-multiplication--matmul) rank dispatch and last-two-axis
  permutation.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [axis glossary](../../../../glossary.md#term-axis); [permutation glossary](../../../../glossary.md#permutation))</span>

`FirstOrderAutograd` remains the single reverse-traversal and contribution-accumulation [owner](../../../../glossary.md#term-owner). It
may [dispatch](../../../../glossary.md#term-dispatch) to the new [linear](../../../../glossary.md#term-linear-meanings)-algebra owner but must not absorb formula implementations.
`AutogradPreflight` remains the single complete [fail-closed](../../../../glossary.md#term-fail-closed) selector.

Rules call only current public [`Tensor`](../../../../glossary.md#tensor) methods and public [immutable](../../../../glossary.md#term-immutable) [model](../../../../glossary.md#term-model) value types. They do
not construct [`Operation`](../../../../glossary.md#operation), [`TensorProducer`](../../../../glossary.md#producer--tensorproducer), captured [nodes](../../../../glossary.md#node), [graph](../../../../glossary.md#graph) IDs, or [descriptors](../../../../glossary.md#term-descriptor) directly.
Do not add a [registry](../../../../glossary.md#term-registry), map-based kind dispatch, service loader, reflection, string dispatch,
generic symbolic algebra, second graph, or public [facade](../../../../glossary.md#term-facade).

### Complete current-[operation](../../../../glossary.md#operation) audit

The following audit is exhaustive over the current [model](../../../../glossary.md#term-model) [`OperationKind`](../../../../glossary.md#operationkind) inventory. “0004”
means already supported and unchanged. “0004A” means exactly the guarded rows above. “0004B”
means a [derivative](../../../../glossary.md#term-derivative) convention must be selected first. “Later cohesive task” means policy-free
formula work that is too large or structurally distinct for this bounded change; progressive
[planning](../../../../glossary.md#term-planning) deliberately assigns no detailed task ID here. “[ND](../../../../glossary.md#term-nd)” means no [cotangent](../../../../glossary.md#cotangent) propagates
through that role.

| Current operation inventory | Classification and reason |
|---|---|
| [Binary](../../../../glossary.md#term-binary-descriptor-and-metadata) `ADD`, `SUB`, `MUL`; [scalar](../../../../glossary.md#term-scalar) `ADD`, `SUB`, `MUL` | 0004 for its exact same-floating variants. Mixed-floating cotangent conversion remains 0004B. |
| Binary/scalar `DIV`, `POW`; `RECIPROCAL`, [`LOG`](../../../../glossary.md#term-log), `LOG1P`, `SQRT`, `RSQRT` | 0004B: zero, invalid-domain, singularity, [infinity](../../../../glossary.md#term-infinity), and [NaN](../../../../glossary.md#term-nan) derivative behavior. |
| Binary/scalar `MIN`, `MAX`; scalar `CLAMP`; `ABS`, `FLOOR`, `CEIL`, `SIGN`, `RELU` | 0004B: ties, endpoints, discontinuities, [signed zero](../../../../glossary.md#term-signed-zero), NaN, or subgradient selection. |
| `NEG`, [`EXP`](../../../../glossary.md#term-expx), `EXPM1`, [`SIGMOID`](../../../../glossary.md#term-sigmoid), [`TANH`](../../../../glossary.md#term-tanh) | 0004. |
| [`ERF`](../../../../glossary.md#term-erf) | 0004A with fixed typed coefficient bits. |
| [`GELU`](../../../../glossary.md#gaussian-error-linear-unit--gelu), `GELU_TANH_APPROXIMATION`, `SILU` | 0004B: otherwise regular formulas contain special-value products such as infinity times zero and require an explicit exceptional-value contract. |
| `WHERE` | 0004 for same-floating branches; condition is ND. Mixed-floating branch cotangents remain 0004B. |
| `CAST` | 0004 only for same-type floating [identity](../../../../glossary.md#term-identity-meanings). Cross-floating conversion remains 0004B; integral and [BOOL](../../../../glossary.md#data-type--datatype) roles are ND. |
| Comparisons `GREATER_THAN`, `GREATER_OR_EQUAL`, `LESS_THAN`, `LESS_OR_EQUAL`, `EQUAL`, `NOT_EQUAL` | ND BOOL results. |
| BOOL `AND`, `OR`, `NOT`; `IS_FINITE`, `IS_NAN`, `IS_INF` | ND BOOL results. |
| Ordinary `SUM` | 0004. |
| Masked `SUM` | 0004A; [mask](../../../../glossary.md#term-boolean-mask) is ND. |
| `SUM_TO_SHAPE` | 0004A only for the locally provable expansion subset. [Binding](../../../../glossary.md#term-binding)-dependent unresolved [singleton](../../../../glossary.md#term-singleton)-or-equal inversion remains a later cohesive [shape](../../../../glossary.md#shape)-binding task. |
| Ordinary or masked [`MEAN`](../../../../glossary.md#term-mean); `PROD`; reduction `MIN`, `MAX`; `LOG_SUM_EXP`, [`VARIANCE`](../../../../glossary.md#term-variance), `STANDARD_DEVIATION`, `L1_NORM`, `L2_NORM` | 0004B: empty count, zero products/norms, [extrema](../../../../glossary.md#saturation-pole-and-extremum) ties, singularities, or exceptional values. |
| `ALL`, `ANY`, `ARG_MAX`, `ARG_MIN` | ND outputs; extrema index/tie policy is not a floating cotangent route. |
| `CUM_SUM` | 0004. |
| `CUM_PROD` | 0004B: zero and exceptional-value product behavior. |
| [`SOFTMAX`](../../../../glossary.md#softmax--log-softmax), `LOG_SOFTMAX` | 0004B: the current contract deliberately leaves empty slices, NaNs, and [infinities](../../../../glossary.md#nan-infinity-and-signed-zero) open. |
| [`CONTIGUOUS`](../../../../glossary.md#term-contiguous), `RESHAPE`, `EXPAND`, `EXPAND_DIMS`, `SQUEEZE`, `PERMUTE` | 0004. |
| [`SLICE`](../../../../glossary.md#slice) with `SliceAttrs`; `SLICE_UPDATE`; `SELECT`; [`PAD`](../../../../glossary.md#padding); `TILE`; `CONCAT`; `STACK` | 0004A exactly as guarded above. `PAD`'s [constant](../../../../glossary.md#term-constant) and `SELECT`'s index are ND [metadata](../../../../glossary.md#term-metadata). |
| `SLICE` with `CropToShapeAttrs` | Later cohesive shape-binding task: dynamic [target](../../../../glossary.md#term-target)-relative prefixes cannot in general be inverted through a current public update/pad operation. |
| [`GATHER`](../../../../glossary.md#gather), `GATHER_ELEMENTS`, `GATHER_ND`, `SCATTER_ELEMENTS`, `SCATTER_ADD`, `SCATTER_ND` | Later cohesive indexing task using current scatter-add/gather primitives, with exact reduction-variant and duplicate-index validity guards. Index inputs are ND. |
| `ONE_HOT` | ND index/depth roles and output for current [gradient](../../../../glossary.md#term-gradient) purposes. |
| `UNFOLD_AXIS`, `FOLD_AXIS`, `UNFOLD2D`, `FOLD2D` | Later cohesive window-[adjoint](../../../../glossary.md#term-adjoint) task. Current public inverse/overlap-sum primitives make policy-free formulas plausible, but their complete dynamic geometry [matrix](../../../../glossary.md#term-matrix-meanings) is deliberately not added here. |
| [`MATMUL`](../../../../glossary.md#matrix-multiplication--matmul) | 0004A for all current floating [rank](../../../../glossary.md#term-rank) cases, role-aware exact-type targets, and current [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) [broadcasting](../../../../glossary.md#broadcasting). Integral routes are ND. |
| [`CONV2D`](../../../../glossary.md#convolution--conv2d) | Later cohesive structured-[linear](../../../../glossary.md#term-linear-meanings) task; grouped/batched window and [parameter](../../../../glossary.md#term-parameter)-role formulas require their own bounded matrix. |
| `MAX_POOL2D` | 0004B: maxima ties, NaNs, and saved-selection policy. |
| `AVERAGE_POOL2D` | Later cohesive [pooling](../../../../glossary.md#term-pooling)/window task after its divisor and [padding](../../../../glossary.md#padding)-edge matrix is specified; no policy is selected here. |
| `MEAN_SQUARED_ERROR`, `DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS`, `INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS` | Later cohesive [loss](../../../../glossary.md#loss) task for regular variants; mean-empty and logits exceptional cases remain 0004B. Index labels are ND. |
| `SCALED_DOT_PRODUCT_ATTENTION` | Later cohesive multi-output structured task after 0004B selects all-masked-row, positive-infinity tie, and NaN behavior. Mask and state roles are ND. |
| `LAYER_NORM`, `RMS_NORM`, `BATCH_NORM_INFERENCE`, `BATCH_NORM_TRAINING` | Later cohesive [normalization](../../../../glossary.md#term-normalization)/multi-output task after 0004B selects zero-variance and exceptional-value behavior. Saved statistics require explicit logical-role handling, not a physical [lifetime](../../../../glossary.md#term-lifetime) policy in this task. |
| `SORT`, `ARGSORT`, `TOP_K` | 0004B for selected-value ties, NaNs, discontinuities, and index routing; index outputs are ND. |
| [`DROPOUT`](../../../../glossary.md#dropout--inverted-dropout) | Later stochastic/multi-output task after explicit probability-edge and saved-mask rules; [RNG state](../../../../glossary.md#graph-rng-state--graphrngstate) and mask outputs are ND. |
| [graph](../../../../glossary.md#graph) [RNG](../../../../glossary.md#graph-rng-state--graphrngstate) `INITIAL_STATE` | ND [opaque](../../../../glossary.md#term-opaque) state. |
| Unknown/custom kind, wrong attributes class, wrong [cardinality](../../../../glossary.md#term-cardinality), missing [canonical](../../../../glossary.md#term-canonical-form) output, or [descriptor](../../../../glossary.md#term-descriptor) contradiction | Deterministic [preflight](../../../../glossary.md#term-preflight) rejection. |

`linear` has no independent operation kind: its current expression is
`PERMUTE -> MATMUL -> ADD`. It becomes [differentiable](../../../../glossary.md#derivative-and-chain-rule) through the selected `MATMUL` row plus
existing rules and receives no special [compiler](../../../../glossary.md#term-compiler) rule.

## Out of [scope](../../../../glossary.md#term-scope)

- changing Compiler 0004's request, seed, identity, accumulation, capture, phase, result,
  validation, or optimization contracts<span data-glossary-navigation="true"> ([seed glossary](../../../../glossary.md#term-seed); [identity glossary](../../../../glossary.md#term-identity-meanings); [capture glossary](../../../../glossary.md#term-capture-meanings); [validation glossary](../../../../glossary.md#term-validation))</span>
- any [derivative](../../../../glossary.md#term-derivative) row or attributes variant outside `SUPPORTED_0004` plus `SUPPORTED_0004A`
- selecting any 0004B derivative convention
- [binding](../../../../glossary.md#term-binding)-dependent `SUM_TO_SHAPE` or [target-relative crop](../../../../glossary.md#target-relative-crop) inversion
- gather/scatter, windows, convolution, pooling, loss, attention, normalization, ordering,
  top-k, dropout, or RNG derivative implementation<span data-glossary-navigation="true"> ([gather glossary](../../../../glossary.md#gather); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [pooling glossary](../../../../glossary.md#term-pooling); [loss glossary](../../../../glossary.md#loss); [normalization glossary](../../../../glossary.md#term-normalization); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
- public [compiler](../../../../glossary.md#term-compiler), [gradient](../../../../glossary.md#term-gradient), [registry](../../../../glossary.md#term-registry), [compile](../../../../glossary.md#compile)-request, result, [artifact](../../../../glossary.md#term-artifact), or [publication](../../../../glossary.md#term-publication) APIs
- explicit seeds, non-scalar objectives, disconnected-target zeros, Jacobians, create-graph, or
  higher derivative order<span data-glossary-navigation="true"> ([seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [scalar glossary](../../../../glossary.md#term-scalar); [target glossary](../../../../glossary.md#term-target))</span>
- changing model operations, Tensor public APIs, descriptors, producer output identity, or result
  carriers<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [Tensor glossary](../../../../glossary.md#tensor); [descriptors glossary](../../../../glossary.md#term-descriptor); [producer glossary](../../../../glossary.md#term-producer))</span>
- a second gradient algebra, captured-value-to-Tensor conversion, direct IR construction, mutable
  gradient state, tape, global cache, or backend-owned autograd<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [mutable glossary](../../../../glossary.md#term-mutable); [cache glossary](../../../../glossary.md#term-cache); [backend glossary](../../../../glossary.md#backend); [autograd glossary](../../../../glossary.md#autograd))</span>
- runtime, prepare, engine, planning orchestration, lowering, backend, execution, trace, physical
  saved-value lifetime, or numerical kernel work<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [engine glossary](../../../../glossary.md#term-engine); [planning glossary](../../../../glossary.md#term-planning); [lowering glossary](../../../../glossary.md#lowering); [trace glossary](../../../../glossary.md#trace); [lifetime glossary](../../../../glossary.md#term-lifetime); [kernel glossary](../../../../glossary.md#kernel))</span>
- new [graph](../../../../glossary.md#graph) rewrites, [constant](../../../../glossary.md#term-constant)-fold rows, cross-phase [CSE](../../../../glossary.md#term-common-subexpression-elimination-cse), pass iteration, or approximate algebra
- architecture, ADR, module dependency, Gradle/build, Java-version, architecture-test,
  backend-conformance, or integration-test changes<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- creating a detailed specification for 0004B or any later cohesive rule task

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>
- [ADR 0009](../../../../design/decisions/0009-compiler-owned-pre-capture-tensor-expression-autograd.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- [Autograd strategy](../../../../design/notes/autograd-strategy.md)<span data-glossary-navigation="true"> ([Autograd glossary](../../../../glossary.md#autograd))</span>
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Compiler master plan](../master-plan.md)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>
- [Compiler 0004](0004-compiler-owned-pre-capture-autograd-and-combined-graph-compilation.md)
- [Model master plan](../../model/master-plan.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Model capabilities](../../model/capabilities.md)
- [Adjoint expressibility audit](../../model/adjoint-expressibility-audit.md)<span data-glossary-navigation="true"> ([Adjoint glossary](../../../../glossary.md#term-adjoint))</span>
- [Model contract-closure audit](../../model/model-capability-contract-closure-audit.md)
- [Model 0025](../../model/tasks/0025-canonical-tensor-producer-outputs.md)
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- [Training API](../../../../api/training-api.md)
- [Autograd user guide](../../../../user-guide/autograd.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `ARCHITECTURE.md` remains authoritative; this task implements the already accepted
  compiler-owned pre-capture design and changes no architecture decision.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings))</span>
- Compiler owns preflight, rule selection, reverse accumulation, formula construction, combined
  capture, phases, and validation.<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [validation glossary](../../../../glossary.md#term-validation))</span>
- [Model](../../../../glossary.md#term-model) owns only the ordinary public [Tensor](../../../../glossary.md#tensor) vocabulary and [immutable](../../../../glossary.md#term-immutable) [operation](../../../../glossary.md#operation) [metadata](../../../../glossary.md#term-metadata).
- Formulas exist as Tensor expressions before graph capture; graph-local IDs are assigned exactly
  once by the existing combined capture.<span data-glossary-navigation="true"> ([graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [graph glossary](../../../../glossary.md#graph))</span>
- Exact Tensor object identity remains request-local bookkeeping, never graph identity or
  persistent state.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- No [dependency direction](../../../../glossary.md#java-platform-and-build-tools) or [module](../../../../glossary.md#term-module) boundary changes. Architecture tests do not change.
- If a selected formula needs a new model API, derivative policy, public compiler contract,
  second algebra, or path outside maximum scope, stop and report the conflict.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [derivative glossary](../../../../glossary.md#term-derivative); [scope glossary](../../../../glossary.md#term-scope))</span>

## Package impact

Existing package:

- `io.github.pho001.synaptik.compiler` remains the single cohesive package-private compiler
  front-end boundary.

Types:

- `AutogradPreflight` extends its closed typed, role-aware [occurrence](../../../../glossary.md#term-occurrence) selection.
- `ElementwiseGradientRules`, `ReductionGradientRules`, and `LayoutGradientRules` gain only their
  assigned rows.
- `LinearAlgebraGradientRules` is one new package-private stateless [owner](../../../../glossary.md#term-owner) for [`MATMUL`](../../../../glossary.md#matrix-multiplication--matmul).
- `FirstOrderAutograd` changes only enough to [dispatch](../../../../glossary.md#term-dispatch) the additional selected plan entries.

No package or public type is added. Tests remain in the mirrored [compiler](../../../../glossary.md#term-compiler) package.

## Affected files

Expected production files:

- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/AutogradPreflight.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ElementwiseGradientRules.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ReductionGradientRules.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/LayoutGradientRules.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/LinearAlgebraGradientRules.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/FirstOrderAutograd.java`

Expected tests:

- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/AutogradPreflightTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/GradientRulesTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/FirstOrderAutogradTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/GraphCompilerTest.java`

Expected documentation and [planning](../../../../glossary.md#term-planning) files:

- `docs/api/compile-api.md`
- `docs/design/notes/autograd-strategy.md`
- `docs/user-guide/autograd.md`
- `docs/glossary.md`
- `docs/planning/modules/compiler/tasks/0004a-exact-composition-gradient-rule-extensions.md`
- `docs/planning/modules/compiler/master-plan.md`
- `docs/planning/roadmap.md`

No [Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api), Training API, architecture, [ADR](../../../../glossary.md#term-architecture-decision-record-adr), [model](../../../../glossary.md#term-model), [Gradle](../../../../glossary.md#term-gradle), or test-infrastructure file is
expected to change. Review those surfaces and record the reasoned no-change conclusion.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most:

- 6 [compiler](../../../../glossary.md#term-compiler) production files,
- 4 compiler test files, and
- 7 documentation/[planning](../../../../glossary.md#term-planning) files,

for a strict ceiling of 17 touched paths.

If implementation needs another path, a [model](../../../../glossary.md#term-model) [API](../../../../glossary.md#term-application-programming-interface-api), or a broader rule [matrix](../../../../glossary.md#term-matrix-meanings), stop and propose a
follow-up instead of [widening](../../../../glossary.md#term-widening) this task.

## Acceptance criteria

### [Preflight](../../../../glossary.md#term-preflight) and [matrix](../../../../glossary.md#term-matrix-meanings)

- Existing `SUPPORTED_0004` success and rejection behavior remains unchanged.
- Preflight accepts every `SUPPORTED_0004A` row only with its exact kind, attributes class,
  output/input role, descriptor, type, Shape, and local-constructibility guard.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [Shape glossary](../../../../glossary.md#shape); [guard glossary](../../../../glossary.md#term-guard))</span>
- Preflight remains role-aware for promoted [`MATMUL`](../../../../glossary.md#matrix-multiplication--matmul) and for the two `SLICE_UPDATE` inputs.
- Every unsupported occurrence on a selected route fails before a new Tensor ID is consumed, with
  the existing deterministic occurrence/output/input/kind/attributes/reason context.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [Tensor glossary](../../../../glossary.md#tensor))</span>
- The complete current-operation classification above is reflected in code tests: no generic
  fallback or accidental kind admission exists.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [admission glossary](../../../../glossary.md#term-admission))</span>

### Formula behavior

- [`ERF`](../../../../glossary.md#term-erf) uses only the fixed raw coefficient bits and public [Tensor](../../../../glossary.md#tensor) operations.
- Masked `SUM` restores the selected axis, broadcasts to the data Shape, masks with exact zero,
  and never propagates a cotangent to the BOOL mask.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [Shape glossary](../../../../glossary.md#shape); [masks glossary](../../../../glossary.md#term-boolean-mask); [cotangent glossary](../../../../glossary.md#cotangent); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- `SUM_TO_SHAPE` accepts exact-equal/static-one aligned Dimensions, including leading reductions,
  and rejects binding-dependent aligned pairs.<span data-glossary-navigation="true"> ([Dimensions glossary](../../../../glossary.md#dimension); [binding glossary](../../../../glossary.md#term-binding))</span>
- Every floating [`MATMUL`](../../../../glossary.md#matrix-multiplication--matmul) rank pairing has exact left/right formulas, batch unbroadcasting, vector
  insertion/removal, and selected-role type guards.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [vector glossary](../../../../glossary.md#term-vector))</span>
- Slice extraction, signed slice replacement, select, pad, tile, concat, and stack formulas
  preserve exact normalized geometry, input order, symbolic Shape references where specified,
  zero extents, and repeated-input contribution multiplicity.<span data-glossary-navigation="true"> ([Slice glossary](../../../../glossary.md#slice))</span>
- Every formula is built through public Tensor operations; no operation, producer, captured node,
  graph ID, or descriptor is manufactured directly.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [producer glossary](../../../../glossary.md#term-producer); [node glossary](../../../../glossary.md#node); [graph glossary](../../../../glossary.md#graph); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>

### Existing pipeline

- Contribution accumulation remains left-associated and deterministic by original occurrence and
  input-position order.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Repeated exact operands and repeated concat/stack inputs receive every positional contribution.
- Generated formulas join the existing combined [Tensor](../../../../glossary.md#tensor) [DAG](../../../../glossary.md#term-directed-acyclic-graph-dag) and are captured once with correct
  `FORWARD`/`BACKWARD` phases.
- Forward outputs, gradient result roles, output de-duplication, constant ingress, validation, and
  exact optimization behavior remain unchanged.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [constant glossary](../../../../glossary.md#term-constant); [validation glossary](../../../../glossary.md#term-validation))</span>
- `FORWARD_ONLY`, `FORWARD_AND_BACKWARD`, and `TRAINING_STEP` retain the Compiler 0004 mode
  contract.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>

### Documentation and completion

- Every changed Java type/method has meaningful complete Javadoc, including parameters, results,
  failures, nullability, and ownership where applicable.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [parameters glossary](../../../../glossary.md#term-parameter); [nullability glossary](../../../../glossary.md#term-nullability); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Compile API, autograd strategy, user guide, glossary impact, compiler master plan, roadmap, and
  this task agree on the exact supported/deferred matrix.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [autograd glossary](../../../../glossary.md#autograd); [compiler glossary](../../../../glossary.md#term-compiler); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- Tensor API and Training API are reviewed and left unchanged because no public Tensor or
  training lifecycle contract changes; the completion summary records this.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- A separate clean-context documentation-focused agent finalizes affected Javadoc and prose after
  implementation, reusing successful Java test evidence unless it changes executable behavior.
- Final status changes to `Complete` only after implementation, documentation, all validation,
  exact-scope inspection, and the completion summary are complete.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [scope glossary](../../../../glossary.md#term-scope))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

Implementation tests must cover at least:

- positive [`ERF`](../../../../glossary.md#term-erf) formula structure and exact coefficient bits for [BFLOAT16](../../../../glossary.md#term-bfloat16)/[FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64);
- masked `SUM` full Shape restoration, broadcast mask, masked false branch, zero extent, mask ND,
  and wrong-type/attrs rejection;<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [mask glossary](../../../../glossary.md#term-boolean-mask); [ND glossary](../../../../glossary.md#term-nd))</span>
- `SUM_TO_SHAPE` leading axes, exact aligned dimensions, static-one expansion, scalar target,
  zero extent, and binding-dependent rejection;<span data-glossary-navigation="true"> ([axes glossary](../../../../glossary.md#term-axis); [dimensions glossary](../../../../glossary.md#dimension); [target glossary](../../../../glossary.md#term-target); [binding glossary](../../../../glossary.md#term-binding))</span>
- all four [`MATMUL`](../../../../glossary.md#matrix-multiplication--matmul) rank pairings, batched broadcast on either operand, vector promotion,
  repeated operand, selected promoted operand success/failure, zero contraction, and integral
  rejection;<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [vector glossary](../../../../glossary.md#term-vector); [contraction glossary](../../../../glossary.md#contraction-dimension))</span>
- positive, negative, strided, empty, and [identity](../../../../glossary.md#term-identity-meanings) [`SLICE`](../../../../glossary.md#slice);
- `SLICE_UPDATE` base/update roles, signed steps, empty replacement, repeated exact base/update,
  dynamic selected-base rejection only for the update role, and no partial allocation;
- [scalar](../../../../glossary.md#term-scalar)-result and dynamic-[extent](../../../../glossary.md#term-extent) `SELECT`, scalar [`PAD`](../../../../glossary.md#padding), symbolic/zero-extent padding,
  scalar/dynamic <span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding))</span>`TILE`, multi-input symbolic `CONCAT`, `STACK`, and repeated exact composition
  inputs;
- unsupported 0004B and later-cohesive categories still failing [preflight](../../../../glossary.md#term-preflight) before allocation;
- end-to-end combined capture, phase facts, gradient roles, logical constants, optional
  optimization, and mode behavior for representative new rows; and<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [gradient glossary](../../../../glossary.md#term-gradient); [constants glossary](../../../../glossary.md#term-constant))</span>
- existing [Compiler](../../../../glossary.md#term-compiler) 0004 [regression](../../../../glossary.md#term-regression) coverage.

[Run](../../../../glossary.md#term-run) once after executable code stabilizes:

```bash
./gradlew :modules:compiler:test
./gradlew :modules:compiler:javadoc
```

The documentation-focused pass validates links, anchors, fences, terminology, public-surface
claims, and exact support-[matrix](../../../../glossary.md#term-review-matrix) consistency without rerunning successful Java tests unless it
changes executable Java behavior.

Final repository checks:

```bash
git diff --check
git status --short
git diff --name-only
git ls-files --others --exclude-standard
```

The union of modified and untracked paths must equal only the affected-file list and remain at or
below 17 paths. No architecture-test, [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), integration-test, [Gradle](../../../../glossary.md#term-gradle), [model](../../../../glossary.md#term-model),
[Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api), or Training API change is expected.

## Dependencies

- [Compiler](../../../../glossary.md#term-compiler) 0004 — Complete
- [Model](../../../../glossary.md#term-model) 0025 and Compiler 0001–0003B — Complete through Compiler 0004
- current public [Tensor](../../../../glossary.md#tensor) operations and [operation](../../../../glossary.md#operation) [metadata](../../../../glossary.md#term-metadata) — Complete
- accepted compiler-owned pre-[capture](../../../../glossary.md#term-capture-meanings) architecture and [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0009 — Complete

No 0004B policy decision is a [dependency](../../../../glossary.md#term-dependency) because every selected row is policy-free.

## Follow-up tasks

- Compiler 0004B selects explicit tie, endpoint, discontinuity, singularity, empty-domain,
  exceptional-value, and cross-floating cotangent policies before adding dependent formulas.<span data-glossary-navigation="true"> ([cotangent glossary](../../../../glossary.md#cotangent))</span>
- Later cohesive tasks may address binding-dependent crop/shape inversion, indexing/scatter,
  windows, structured linear operations, pooling, losses, normalization, multi-output attention,
  ordering, and stochastic operations. Progressive planning assigns detailed specifications only
  when each becomes the current frontier.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [shape glossary](../../../../glossary.md#shape); [linear glossary](../../../../glossary.md#term-linear-meanings); [pooling glossary](../../../../glossary.md#term-pooling); [losses glossary](../../../../glossary.md#loss); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- [Compiler](../../../../glossary.md#term-compiler) 0005 owns [publication](../../../../glossary.md#term-publication), [planning](../../../../glossary.md#term-planning) orchestration, and [immutable](../../../../glossary.md#term-immutable) [compile artifacts](../../../../glossary.md#compile-artifacts).
- Compiler 0006 owns public functional gradient requests, explicit seeds, derivative order, and
  the higher-order formula-coverage contract.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [derivative glossary](../../../../glossary.md#term-derivative))</span>

## Architecture impact

None. This task fills the already accepted [Compiler](../../../../glossary.md#term-compiler) 0004A extension point inside the compiler
package. It changes no [module](../../../../glossary.md#term-module) boundary, [dependency direction](../../../../glossary.md#java-platform-and-build-tools), [lifecycle](../../../../glossary.md#lifecycle), [graph phase](../../../../glossary.md#graph-phase), [capture](../../../../glossary.md#term-capture-meanings),
public [API](../../../../glossary.md#term-application-programming-interface-api), or authoritative architecture rule. No [ADR](../../../../glossary.md#term-architecture-decision-record-adr) or [architecture test](../../../../glossary.md#term-architecture-test) is required.

## Implementation prompt

Implement [Compiler](../../../../glossary.md#term-compiler) task 0004A exactly as specified in this file.

Work in a separate clean implementation context. First read `AGENTS.md`, `ARCHITECTURE.md`, the
focused architecture documents and [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0009, documentation rules and relevant profiles, [planning](../../../../glossary.md#term-planning)
guide and roadmap, compiler master plan, Compiler 0001–0004, [Model](../../../../glossary.md#term-model) 0025, both model audits, [Compile](../../../../glossary.md#compile)
[API](../../../../glossary.md#term-application-programming-interface-api), [Tensor](../../../../glossary.md#tensor) API, Training API, [autograd](../../../../glossary.md#autograd) strategy/user guide, glossary, and the current compiler
and model [operation](../../../../glossary.md#operation) sources/tests needed to verify every selected attributes variant.

Extend the existing package-private [preflight](../../../../glossary.md#term-preflight) and named rules only. Implement the exact additive
[matrix](../../../../glossary.md#term-matrix-meanings), fixed [`ERF`](../../../../glossary.md#term-erf) coefficient bits, role-aware [`MATMUL`](../../../../glossary.md#matrix-multiplication--matmul) and `SLICE_UPDATE` guards, deterministic
formula construction, and tests. Preserve all Compiler 0004 request, [identity](../../../../glossary.md#term-identity-meanings), accumulation,
[constant](../../../../glossary.md#term-constant), [capture](../../../../glossary.md#term-capture-meanings), phase, result, [validation](../../../../glossary.md#term-validation), and optimization contracts. Do not add a public
surface, [registry](../../../../glossary.md#term-registry), second algebra, direct [IR](../../../../glossary.md#term-intermediate-representation-ir) construction, [derivative](../../../../glossary.md#term-derivative) policy, model API, or
unlisted operation.

After Java behavior stabilizes, [run](../../../../glossary.md#term-run) the specified compiler tests and [Javadoc](../../../../glossary.md#term-javadoc) once. Do not repeat
the repository-wide capability [checkpoint](../../../../glossary.md#term-checkpoint) reserved by the master plan for 0004B. Then use a
distinct clean documentation-focused context to independently finalize Javadoc, Compile API,
autograd strategy, user guide, glossary impact, task, master plan, and roadmap. Reuse successful
test evidence unless documentation changes executable Java behavior. Run the
documentation/[scope](../../../../glossary.md#term-scope)/whitespace checks and record exact evidence.

If any selected row cannot be expressed through current public Tensor operations, if a derivative
policy is required, or if more than 17 paths are needed, stop and report the exact conflict rather
than [widening](../../../../glossary.md#term-widening) scope. Do not [commit](../../../../glossary.md#term-commit) or push unless explicitly requested.

## Local decisions

- The extension is deliberately vertical across four existing formula families, but remains
  bounded to formulas with no derivative convention.<span data-glossary-navigation="true"> ([derivative glossary](../../../../glossary.md#term-derivative))</span>
- [`ERF`](../../../../glossary.md#term-erf) coefficient representation is fixed at planning time so compilation never depends on host
  transcendental evaluation.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [host glossary](../../../../glossary.md#term-host); [evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- [`MATMUL`](../../../../glossary.md#matrix-multiplication--matmul) is role-aware under floating promotion; same-type selected cotangents are supported
  without forcing the unrelated operand to have that type.
- `SUM_TO_SHAPE` admits only locally constructible inverse expansion. Deferred binding
  obligations are not guessed.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>
- `SLICE_UPDATE` base and update roles have different constructibility guards.
- Data-movement formulas preserve repeated positional contributions and symbolic Shapes rather
  than collapsing equal Tensor occurrences.<span data-glossary-navigation="true"> ([Shapes glossary](../../../../glossary.md#shape); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Plausible policy-free indexing, window, and structured formulas remain later cohesive work
  because including their complete variant matrices would exceed this task's risk and path budget.<span data-glossary-navigation="true"> ([matrices glossary](../../../../glossary.md#term-matrix-meanings))</span>

## Known limitations

- The request remains one [scalar objective](../../../../glossary.md#loss), implicit unit [seed](../../../../glossary.md#term-seed), and ordered connected [targets](../../../../glossary.md#term-target).
- Mixed-floating cotangents and all 0004B conventions remain unsupported.
- [Binding](../../../../glossary.md#term-binding)-dependent `SUM_TO_SHAPE` and crop inversion remain unsupported.
- The [operation](../../../../glossary.md#operation) audit classifies later work but implements only `SUPPORTED_0004A`.
- Higher-order request semantics and complete [derivative](../../../../glossary.md#term-derivative) closure remain [Compiler](../../../../glossary.md#term-compiler) 0006 work.

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) and implementation evidence:

- architecture, focused design, planning, documentation, API, completed Compiler 0001–0004,
  Model 0025, model audits, compiler source/tests, and complete current operation-kind inventory
  reviewed<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compiler glossary](../../../../glossary.md#term-compiler); [Model glossary](../../../../glossary.md#term-model); [operation glossary](../../../../glossary.md#operation))</span>
- no architecture or [dependency](../../../../glossary.md#term-dependency) conflict found
- selected rules verified as compositions of current public [Tensor](../../../../glossary.md#tensor) operations
- selected/deferred/non-[differentiable](../../../../glossary.md#derivative-and-chain-rule) classification recorded exhaustively
- task path budget fixed at 17
- implementation context `/root/implement_compiler_0004a_gradient_extensions` completed the six
  production-file and four test-file implementation without changing architecture, dependencies,
  public APIs, build configuration, or test infrastructure
- `./gradlew :modules:compiler:compileJava` and
  `./gradlew :modules:compiler:compileTestJava` passed
- the focused `AutogradPreflightTest`, `GradientRulesTest`, `FirstOrderAutogradTest`, and
  `GraphCompilerTest` run first exposed one assertion-message expectation, then passed all 28
  tests after that test-only correction<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
- the final `./gradlew :modules:compiler:test` run passed 131 tests with zero failures, errors, or
  skips; `git diff --check` also passed in the implementation context
- documentation context `/root/document_compiler_0004a_gradient_extensions` changed no executable
  Java tokens and deliberately did not repeat the successful Java suite; comment-stripped SHA-256
  hashes for all six authorized production paths matched the hashes recorded before the Javadoc
  pass<span data-glossary-navigation="true"> ([hashes glossary](../../../../glossary.md#term-hash))</span><span data-glossary-navigation="true"> ([tokens glossary](../../../../glossary.md#term-token); [SHA-256 glossary](../../../../glossary.md#term-sha-256); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- `./gradlew :modules:compiler:javadoc` passed; its one warning is the pre-existing undocumented
  default constructor of `FirstOrderAutograd.DerivativeConstants`, and generated pages for all
  six affected owners were inspected for the finalized contracts<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span>
- `node /tmp/validate_synaptik_markdown.js` passed 240 Markdown files, 4,402 local links, 271
  anchors, and 2,950 fence markers
- final whitespace, status, terminology, and exact 17-path [scope](../../../../glossary.md#term-scope) checks passed
- repository-wide validation remains deferred to the Compiler 0004B capability checkpoint and CI
  under the planning guide's validation tiers<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [CI glossary](../../../../glossary.md#term-continuous-integration-ci))</span>

## Completion summary

- Completed the additive [fail-closed](../../../../glossary.md#term-fail-closed) `SUPPORTED_0004A` preflight matrix and rule dispatch for
  typed ERF, masked SUM, locally invertible SUM_TO_SHAPE, every floating MATMUL rank pairing, and
  the guarded exact layout/selection/composition families.<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [dispatch glossary](../../../../glossary.md#term-dispatch); [ERF glossary](../../../../glossary.md#term-erf); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [rank glossary](../../../../glossary.md#term-rank); [layout glossary](../../../../glossary.md#layout))</span>
- Added and finalized the narrow `LinearAlgebraGradientRules` owner while preserving the existing
  request, seed, identity accumulation, deterministic repeated-position handling, one-capture,
  phase, result, validation, and optimization contracts.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [seed glossary](../../../../glossary.md#term-seed); [identity glossary](../../../../glossary.md#term-identity-meanings); [capture glossary](../../../../glossary.md#term-capture-meanings); [validation glossary](../../../../glossary.md#term-validation))</span>
- Added focused preflight, formula-structure, repeated-role, and end-to-end graph-compiler
  regression coverage across the four authorized test files.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [compiler glossary](../../../../glossary.md#term-compiler); [regression glossary](../../../../glossary.md#term-regression))</span>
- Finalized Javadocs, Compile API, autograd strategy, user guide, glossary, task, compiler master
  plan, and roadmap within the exact 17-path ceiling.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [autograd glossary](../../../../glossary.md#autograd))</span>
- Reviewed Tensor API and Training API and left them unchanged because the implementation changes
  no public Tensor surface, Tensor result/gradient lifecycle, training lifecycle, or executable
  contract.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [gradient glossary](../../../../glossary.md#term-gradient); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Reviewed `ARCHITECTURE.md`, focused architecture documents, ADR 0009, related operation
  contracts, architecture tests, backend conformance, integration tests, Gradle configuration,
  and other modules. No architecture decision, dependency boundary, backend behavior, end-to-end
  behavior, build contract, or cross-module public contract changed, so no update was required.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [operation glossary](../../../../glossary.md#operation); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- No unresolved issue or task-local follow-up remains. Compiler 0004B retains derivative-policy
  selection and remains Draft without a detailed specification.<span data-glossary-navigation="true"> ([derivative glossary](../../../../glossary.md#term-derivative))</span>

Status: Complete
