# Task 0004B: Shared-algebra [Cotangent Normalization](../../../../glossary.md#cotangent-normalization) and [Local Derivative](../../../../glossary.md#derivative-and-chain-rule) Rules

## Status

Complete

## Goal

Extend the completed [Compiler](../../../../glossary.md#term-compiler) 0004/0004A first-order pre-[capture](../../../../glossary.md#term-capture-meanings) automatic-differentiation
pipeline with one closed [matrix](../../../../glossary.md#term-matrix-meanings) for:

```text
mixed-floating cotangent Shape/DataType normalization
  + binary and scalar division differentiation rules
  + direct-zero FLOOR/CEIL/SIGN local derivative conventions
  + ordinary and masked mean differentiation rules
  -> ordinary public Tensor expressions
  -> the existing shared validation and exact optimization pipeline
```

The combined forward/[backward graph](../../../../glossary.md#backward-graph) has one shared algebra and one shared [inference](../../../../glossary.md#term-inference), [validation](../../../../glossary.md#term-validation),
numerical-semantics, and optimization contract. A generated [gradient](../../../../glossary.md#term-gradient) expression is an ordinary
[Tensor](../../../../glossary.md#tensor) expression. There is no gradient-only arithmetic, comparison, cast, [NaN](../../../../glossary.md#term-nan)/[infinity](../../../../glossary.md#term-infinity),
rewriting, folding, common-subexpression-elimination, or dead-code-elimination policy or pass.

[Autograd](../../../../glossary.md#autograd) adds only the local differentiation transformation: for one selected [producer](../../../../glossary.md#term-producer) [occurrence](../../../../glossary.md#term-occurrence)
and incoming [cotangent](../../../../glossary.md#cotangent), it constructs ordinary Tensor expressions, applies any genuinely required
[local derivative](../../../../glossary.md#derivative-and-chain-rule) or subgradient convention, and normalizes each contribution to its selected
input's [Shape](../../../../glossary.md#shape) and [DataType](../../../../glossary.md#data-type--datatype) before deterministic accumulation.

## [Scope](../../../../glossary.md#term-scope)

### Existing pipeline remains the [owner](../../../../glossary.md#term-owner)

Extend only the existing package-private [preflight](../../../../glossary.md#term-preflight), reverse-[dispatch](../../../../glossary.md#term-dispatch) and [derivative](../../../../glossary.md#term-derivative)-[constant](../../../../glossary.md#term-constant)
helper, and named [gradient](../../../../glossary.md#term-gradient) rule owners:

```text
complete original Tensor inventory
  -> complete fail-closed differentiation-rule preflight
  -> local formulas through ordinary public Tensor operations
  -> cotangent Shape/DataType normalization
  -> identity-keyed ordered contribution accumulation
  -> one phase-aware combined capture
  -> Compiler 0002 inference and validation
  -> existing exact shared optimization
```

Preserve the `GraphCompiler.compile(...)` [parameter](../../../../glossary.md#term-parameter) list, `AutogradPreflight.FirstOrderRequest`
record shape, implicit [scalar](../../../../glossary.md#term-scalar) unit [seed](../../../../glossary.md#term-seed), `GraphCompilation` result, result-role ordering,
one-[capture](../../../../glossary.md#term-capture-meanings) semantics, [graph](../../../../glossary.md#graph) phases, constant ingress, [validation](../../../../glossary.md#term-validation) boundaries, and optimization
pass order.

`FirstOrderAutograd.apply()` currently routes only `AggregateReductionKind.SUM` and
`CumulativeScanKind.CUM_SUM` to `ReductionGradientRules`. Extend that existing closed dispatch
only enough to route preflight-approved ordinary and masked `AggregateReductionKind.MEAN`
occurrences to the same named reduction-rule owner. This is required plumbing for the selected
[MEAN](../../../../glossary.md#term-mean) rows, not a new formula owner, rule family, policy, public [API](../../../../glossary.md#term-application-programming-interface-api), or algebra.

### One shared algebra

Forward expressions and [compiler](../../../../glossary.md#term-compiler)-generated [gradient](../../../../glossary.md#term-gradient) expressions use the same [model](../../../../glossary.md#term-model)-owned
[operation](../../../../glossary.md#operation) vocabulary and contracts:

- the model's public Tensor methods construct every arithmetic, cast, comparison, selection,
  reduction, and Shape expression;<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Shape glossary](../../../../glossary.md#shape))</span>
- Compiler 0002 derives and validates descriptors and deferred constraints without a
  forward/backward semantic split;<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor))</span>
- each generated node inherits the operation kind's current mathematical and numerical meaning,
  including its ordinary NaN, infinity, signed-zero, overflow, underflow, rounding, and
  empty-domain contract where one exists;<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [overflow glossary](../../../../glossary.md#term-overflow); [underflow glossary](../../../../glossary.md#term-underflow); [rounding glossary](../../../../glossary.md#term-rounding))</span>
- the exact Compiler 0003A rewrite matrix and Compiler 0003B fold matrix apply under the same
  existing guards in either graph phase;<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings); [graph phase glossary](../../../../glossary.md#graph-phase))</span>
- dead-code elimination remains whole-graph and common-subexpression elimination remains
  phase-local; and<span data-glossary-navigation="true"> ([dead-code elimination glossary](../../../../glossary.md#term-dead-code-elimination-dce); [graph glossary](../../../../glossary.md#graph); [common-subexpression elimination glossary](../../../../glossary.md#term-common-subexpression-elimination-cse))</span>
- no formula in this task authorizes reassociation, reciprocal/division substitution, floating
  constant evaluation, approximate algebra, a new rewrite, or a new pass.<span data-glossary-navigation="true"> ([reassociation glossary](../../../../glossary.md#term-reassociation); [constant glossary](../../../../glossary.md#term-constant); [evaluation glossary](../../../../glossary.md#term-evaluation))</span>

`GraphPhase.BACKWARD` classifies generated work. It does not select a different algebra,
[validation](../../../../glossary.md#term-validation) rule, or optimization pipeline.

### Shared notation and [cotangent normalization](../../../../glossary.md#cotangent-normalization)

For each selected output:

- `g` is its accumulated [cotangent](../../../../glossary.md#cotangent);
- `x`, `left`, and `right` are exact original input [Tensors](../../../../glossary.md#tensor);
- `y` is the [producer](../../../../glossary.md#term-producer)'s [canonical](../../../../glossary.md#term-canonical-form) exact selected output wrapper;
- `Z_t` and `O_t` are the existing explicit exact typed positive-zero and positive-one logical
  splats expanded through public `Tensor.expand` to Tensor `t`'s [Shape](../../../../glossary.md#shape);
- `R_x(v)` restores any reduction [axes](../../../../glossary.md#term-axis) removed from `v` and expands through public Tensor
  operations to `x.shape()`; and
- `C_x(v)` first applies ordinary `v.sumToShape(x.shape())` when broadcast reversal is required,
  then applies ordinary `cast(x.dataType())` exactly when the result type differs from `x`'s
  floating type.

`C_x` is cotangent Shape/[DataType](../../../../glossary.md#data-type--datatype) [normalization](../../../../glossary.md#term-normalization), not an alternative arithmetic or conversion
policy. `sumToShape` and `cast` retain their current [model](../../../../glossary.md#term-model) semantics and enter the same combined
[capture](../../../../glossary.md#term-capture-meanings), [inference](../../../../glossary.md#term-inference), [validation](../../../../glossary.md#term-validation), and optimization pipeline as any forward [occurrence](../../../../glossary.md#term-occurrence). Same-type
rows add no redundant cast.

For role-aware [`MATMUL`](../../../../glossary.md#matrix-multiplication--matmul), construct the existing [rank](../../../../glossary.md#term-rank)-specific formula, reverse [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) [broadcasting](../../../../glossary.md#broadcasting)
with `sumToShape`, and then cast once to the selected operand type if required. Every contribution
must have its selected input's exact Shape and floating DataType before it is appended for
accumulation.

### Exact typed generated [constants](../../../../glossary.md#term-constant)

Retain the current request-local exact positive-zero and positive-one logical-splat behavior.
This task adds no half, bound, count, [NaN](../../../../glossary.md#term-nan), [infinity](../../../../glossary.md#term-infinity), or other [scalar](../../../../glossary.md#term-scalar) [leaf](../../../../glossary.md#term-leaf). Forward-ingress [bindings](../../../../glossary.md#term-binding)
remain first, generated [derivative](../../../../glossary.md#term-derivative) bindings retain deterministic first-use order, and no value is
inferred from storage, labels, [descriptors](../../../../glossary.md#term-descriptor), factory history, [Shape](../../../../glossary.md#shape), or missing [provenance](../../../../glossary.md#provenance).

Logical-one expansion is valid for static, dynamic, and expression Shapes. A [logical splat](../../../../glossary.md#logical-splat) has one
typed scalar fact for every eventual logical [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) without storing, enumerating, or binding
those coordinates.

### Mixed-floating request and accumulation

Relax only the current exact [objective](../../../../glossary.md#term-objective)-type restriction on requested [targets](../../../../glossary.md#term-target) and selected
intermediate inputs:

- the objective remains scalar, floating, gradient-eligible, and implicitly seeded in its exact
  type;<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- every target remains floating, gradient-eligible, identity-unique, in objective ancestry, and
  connected through a selected differentiable route, but may have a different floating type;<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- every selected output and selected [differentiable](../../../../glossary.md#derivative-and-chain-rule) input is floating;
- [operation](../../../../glossary.md#operation)-specific promotion and [Shape](../../../../glossary.md#shape) facts must match the current [model](../../../../glossary.md#term-model) contract; and
- every contribution passes through `C_x`, or the corresponding rank-specific normalization,
  before entering the selected input's contribution list.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [normalization glossary](../../../../glossary.md#term-normalization))</span>

The explicit `CAST` occurrences are ordinary model conversion requests. This task adds no
[gradient](../../../../glossary.md#term-gradient)-specific [rounding](../../../../glossary.md#term-rounding), [NaN](../../../../glossary.md#term-nan)-[payload](../../../../glossary.md#artifact-envelope-and-atomic-publication), conversion, or [backend](../../../../glossary.md#backend)-instruction promise.

### Closed `SUPPORTED_0004B` [matrix](../../../../glossary.md#term-matrix-meanings)

The rows below are additive to `SUPPORTED_0004` and `SUPPORTED_0004A`. Existing same-type
construction remains unchanged except where [dispatch](../../../../glossary.md#term-dispatch) is naturally shared with the normalized
path.

#### Mixed-floating [normalization](../../../../glossary.md#term-normalization) and division rules

| Family and exact variant | [Preflight](../../../../glossary.md#term-preflight) [guard](../../../../glossary.md#term-guard) | Local formula and normalization |
|---|---|---|
| [Binary](../../../../glossary.md#term-binary-descriptor-and-metadata) `ADD` | Exact [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs), two floating inputs, current promotion and broadcast facts. | Left `C_left(g)`; right `C_right(g)`. |
| Binary `SUB` | Same as binary `ADD`. | Left `C_left(g)`; right `C_right(g.neg())`. |
| Binary `MUL` | Same as binary `ADD`. | Left `C_left(g.mul(right))`; right `C_right(g.mul(left))`. |
| Binary `DIV` | Exact `NoOperationAttrs`, two floating inputs, current promotion and broadcast facts. | Left `C_left(g.div(right))`; right `C_right(g.mul(left).neg().div(right.mul(right)))`. |
| [Scalar](../../../../glossary.md#term-scalar) `DIV` | Exact `ScalarValueAttrs`; input, output, and scalar have one exact floating type. | `g.div(attrs.value())`. |
| `WHERE` branch roles | Exact `NoOperationAttrs`; [BOOL](../../../../glossary.md#data-type--datatype) condition; current three-way broadcast; selected branch and output are floating. | Existing zero-routed true/false branch formula, followed by `C_branch`; condition is non-[differentiable](../../../../glossary.md#derivative-and-chain-rule). |
| Floating-to-floating `CAST` | Exact `CastAttrs`; source and [target](../../../../glossary.md#term-target) are floating; target attribute equals the output [DataType](../../../../glossary.md#data-type--datatype). | Same type returns `g`; otherwise `g.cast(sourceType)`. |
| Floating [`MATMUL`](../../../../glossary.md#matrix-multiplication--matmul) | Existing complete [rank](../../../../glossary.md#term-rank), [contraction](../../../../glossary.md#contraction-dimension), promotion, and [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-broadcast facts. | Existing four rank-case formulas; restore selected operand [Shape](../../../../glossary.md#shape), then cast once to that operand DataType when needed. |

Binary `DIV` is an ordinary [local derivative](../../../../glossary.md#derivative-and-chain-rule) formula expressed in the public [Tensor](../../../../glossary.md#tensor) vocabulary.
Construct the right contribution in the exact visible order shown above; do not replace it with
an unproved [reassociation](../../../../glossary.md#term-reassociation). Every generated `MUL`, `NEG`, and `DIV` [node](../../../../glossary.md#node) inherits the current [model](../../../../glossary.md#term-model)
semantics, and the existing global exact optimization rules apply under their normal guards.
There is no separate [gradient](../../../../glossary.md#term-gradient) singularity, [NaN](../../../../glossary.md#term-nan), [infinity](../../../../glossary.md#term-infinity), [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero), [overflow](../../../../glossary.md#term-overflow), [underflow](../../../../glossary.md#term-underflow), or
[rounding](../../../../glossary.md#term-rounding) policy and no exemption from shared optimization.

#### Direct-zero [local derivative](../../../../glossary.md#derivative-and-chain-rule) conventions

| Family and exact variant | Local derivative convention |
|---|---|
| `FLOOR`, `CEIL`, `SIGN` | Return `Z_x` directly for the represented input, independent of `g`; do not construct `g.mul(0)` and do not construct a floating comparison. |

These are genuine first-order local derivative conventions at discontinuous or otherwise
nondifferentiable inputs. They choose a direct zero [cotangent](../../../../glossary.md#cotangent) for every represented input,
including signed zeros and non-[finite](../../../../glossary.md#term-finite) inputs, without defining or changing the forward [operation](../../../../glossary.md#operation)'s
[model](../../../../glossary.md#term-model) semantics.

#### Ordinary [`MEAN`](../../../../glossary.md#term-mean)

Support current floating full, single-[axis](../../../../glossary.md#term-axis), and multi-axis `MEAN` variants for static, dynamic,
and expression selected extents. An empty multi-axis list remains a point domain. Do not compute
the domain count in [host](../../../../glossary.md#term-host) code, inspect `Shape.knownElementCount()`, compare the count with
[DataType](../../../../glossary.md#data-type--datatype)-specific exact-integer thresholds, or introduce a [scalar](../../../../glossary.md#term-scalar) count attribute.

[Preflight](../../../../glossary.md#term-preflight) requires one exact floating input/output type and the current family-owned [descriptor](../../../../glossary.md#term-descriptor)
contract:

- full form uses `NoOperationAttrs.INSTANCE` and scalar output;
- single-axis form uses exact `AxisReductionAttrs` with its normalized axis and recorded
  keep-dimensions result; and<span data-glossary-navigation="true"> ([normalized axis glossary](../../../../glossary.md#normalized-axis))</span>
- multi-axis form uses exact `MultiAxisReductionAttrs` with ordered distinct normalized axes and
  the recorded keep-dimensions result.

Construct the denominator through ordinary [Tensor](../../../../glossary.md#tensor) operations:

```text
ones  = O_x
count = ones.sum()                  // full reduction
      | ones.sum(axis, true)        // single axis
      | ones.sum(axes, true)        // ordered multi-axis set, including []
dx    = R_x(g).div(count.expand(x.shape()))
```

The expanded logical-one expression and the reduction retain dynamic and expression [Dimensions](../../../../glossary.md#dimension),
so the existing expression vocabulary represents a count whose concrete [extent](../../../../glossary.md#term-extent) is bound later.
The `SUM`, `DIV`, and `EXPAND` occurrences have exactly their ordinary [model](../../../../glossary.md#term-model) numerical and [Shape](../../../../glossary.md#shape)
semantics. A static or later-bound zero reduced domain produces an empty input [cotangent](../../../../glossary.md#cotangent) Shape;
the generated denominator remains an ordinary zero-domain `SUM` result and receives no special
[gradient](../../../../glossary.md#term-gradient) execution rule.

#### Masked [`MEAN`](../../../../glossary.md#term-mean)

Support the current masked, [axis](../../../../glossary.md#term-axis)-removing floating `MEAN` variant with exact
`MaskedReductionAttrs`, ordered inputs `[input, mask]`, exact [BOOL](../../../../glossary.md#data-type--datatype) [mask](../../../../glossary.md#term-boolean-mask), current proof that the mask
broadcasts exactly to the input [Shape](../../../../glossary.md#shape), and the current output [descriptor](../../../../glossary.md#term-descriptor).

Use only ordinary public expressions:

```text
selected = Tensor.where(mask, O_x, Z_x)
count    = selected.sum(axis, true)
restored = R_x(g)
dx       = Tensor.where(
             mask,
             restored.div(count.expand(x.shape())),
             Z_x)
```

The mask role is non-[differentiable](../../../../glossary.md#derivative-and-chain-rule). For a [slice](../../../../glossary.md#slice) with zero selected values, the forward masked
mean is [NaN](../../../../glossary.md#term-nan) under the current [model](../../../../glossary.md#term-model) contract; this task's explicit [local derivative](../../../../glossary.md#derivative-and-chain-rule) convention
returns zero for every input [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) in that slice. The final `WHERE` expresses that choice
through ordinary conditional-selection semantics. It does not prescribe [eager](../../../../glossary.md#term-eager-evaluation) or [lazy](../../../../glossary.md#term-lazy-evaluation) branch
[evaluation](../../../../glossary.md#term-evaluation), suppress an intermediate calculation, or add special [compiler](../../../../glossary.md#term-compiler)/[backend](../../../../glossary.md#backend) execution
behavior.

### Complete deferred and non-[differentiable](../../../../glossary.md#derivative-and-chain-rule) classification

Every current family not in `SUPPORTED_0004`, `SUPPORTED_0004A`, or the exact rows above remains
rejected on a selected route.

| Current family | Classification after 0004B |
|---|---|
| [Binary](../../../../glossary.md#term-binary-descriptor-and-metadata)/[scalar](../../../../glossary.md#term-scalar) `POW` | **Blocked by general [model](../../../../glossary.md#term-model) semantics.** The current model leaves real-domain and numerical-edge behavior open. [Autograd](../../../../glossary.md#autograd) cannot repair that missing shared algebra contract. |
| `RECIPROCAL`, [`LOG`](../../../../glossary.md#term-log), `SQRT` | **Blocked by incomplete general model semantics.** Zero/domain/[signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero)/non-[finite](../../../../glossary.md#term-finite) meaning is not yet complete enough for a shared forward/[backward graph](../../../../glossary.md#backward-graph) contract. |
| `LOG1P`, `RSQRT` | **Later cohesive [unary](../../../../glossary.md#term-unary) differentiation task.** Their current model semantics are sufficient for ordinary formulas, but this bounded task does not add that independent rule family. |
| [`GELU`](../../../../glossary.md#gaussian-error-linear-unit--gelu), `GELU_TANH_APPROXIMATION`, `SILU` | **Later cohesive [activation](../../../../glossary.md#term-activation) differentiation task.** Generated operations would retain ordinary shared semantics; no special [gradient](../../../../glossary.md#term-gradient) exceptional-value algebra is implied. |
| Binary/scalar `MIN`/`MAX`, `CLAMP`, `ABS`, `RELU`, and reduction `MIN`/`MAX` | **Blocked by missing general floating-comparison semantics.** The public comparison contract still leaves [NaN](../../../../glossary.md#term-nan), signed-zero, and equality/[tolerance](../../../../glossary.md#term-tolerance) behavior open, so the [compiler](../../../../glossary.md#term-compiler) cannot construct a model-owned predicate indirectly. |
| `PROD`, `CUM_PROD` | **Later cohesive product differentiation task.** Zero-safe prefix/suffix geometry and complete full/[axis](../../../../glossary.md#term-axis)/multi-axis coverage are independently substantial; generated arithmetic would use the shared algebra. |
| `LOG_SUM_EXP` | **Later cohesive reduction differentiation task.** Its current forward semantics are not a new gradient algebra; the complete local formula and selected family coverage remain outside this bounded [matrix](../../../../glossary.md#term-matrix-meanings). |
| [`VARIANCE`](../../../../glossary.md#term-variance), `STANDARD_DEVIATION`, `L1_NORM`, `L2_NORM` | **Later cohesive statistical/[norm](../../../../glossary.md#term-norm) differentiation task.** Correction and [local derivative](../../../../glossary.md#derivative-and-chain-rule)/subgradient conventions must be selected with the complete family. |
| [`SOFTMAX`](../../../../glossary.md#softmax--log-softmax), `LOG_SOFTMAX` | **Blocked by incomplete general model numerical-edge semantics.** Empty slices, NaNs, [infinities](../../../../glossary.md#nan-infinity-and-signed-zero), and finite-[precision](../../../../glossary.md#term-precision) behavior remain outside the current shared model contract. |
| `LAYER_NORM`, `RMS_NORM` | **Later cohesive [normalization](../../../../glossary.md#term-normalization) task**, after every required shared model semantic [dependency](../../../../glossary.md#term-dependency) is complete. |
| `BATCH_NORM_INFERENCE`, `BATCH_NORM_TRAINING` | **Later cohesive normalization/multi-output task.** Saved output slots are available, but role coverage and complete local formulas require their own matrix. |
| `MAX_POOL2D` | **Later cohesive [pooling](../../../../glossary.md#term-pooling) task.** The current first-[sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) selection semantics are model-owned; complete window routing remains outside 0004B. |
| `AVERAGE_POOL2D` | **Later cohesive pooling/window task.** Fixed divisor semantics are known, but complete dynamic window/fold geometry is separate. |
| [Mean](../../../../glossary.md#term-mean)-squared-error and categorical-cross-entropy [losses](../../../../glossary.md#loss) | **Later cohesive loss task.** Their complete [prediction](../../../../glossary.md#term-prediction)/[target](../../../../glossary.md#term-target)/reduction/ignored-role matrices are not added here. Index targets remain non-differentiable. |
| [Scaled dot-product attention](../../../../glossary.md#scaled-dot-product-attention) | **Later cohesive multi-output structured task.** [Weights](../../../../glossary.md#term-weight) and [mask](../../../../glossary.md#term-boolean-mask) roles, causal behavior, and complete local formulas must be selected together. |
| `SORT`, values output of `TOP_K` | **Later cohesive ordering task.** Stable order is fixed, but equal-key/cutoff and selected-value routing need a complete local convention. `ARGSORT` and top-K indices are non-differentiable. |
| [`GATHER`](../../../../glossary.md#gather), `GATHER_ELEMENTS`, `GATHER_ND`, replacement/ADD scatter | **Later cohesive indexing/scatter task.** Variant, duplicate-target, and dynamic-[Shape](../../../../glossary.md#shape) coverage is too large for 0004B. Index roles are non-differentiable. |
| MUL/MIN/MAX scatter reductions | **Later cohesive scatter-reduction task**, dependent on complete shared reduction and, for [extrema](../../../../glossary.md#saturation-pole-and-extremum), floating-comparison semantics. |
| `UNFOLD_AXIS`, `FOLD_AXIS`, `UNFOLD2D`, `FOLD2D`, [`CONV2D`](../../../../glossary.md#convolution--conv2d) | **Later cohesive window/structured-[linear](../../../../glossary.md#term-linear-meanings) tasks.** Their geometry matrices are independent of this task. |
| [`DROPOUT`](../../../../glossary.md#dropout--inverted-dropout) | **Later cohesive stochastic/multi-output task.** Probability endpoints, state roles, scaling formula, and [canonical](../../../../glossary.md#term-canonical-form) saved-mask use must be specified together. State and mask outputs are non-differentiable. |
| Comparisons, [BOOL](../../../../glossary.md#data-type--datatype) logic/classification, `ALL`, `ANY`, `ARG_MIN`, `ARG_MAX`, one-hot indices, [graph RNG state](../../../../glossary.md#graph-rng-state--graphrngstate) | **Non-differentiable output/roles.** They may be fixed formula conditions but receive no [cotangent](../../../../glossary.md#cotangent). |
| Existing 0004A [binding](../../../../glossary.md#term-binding)-dependent `SUM_TO_SHAPE`, [target-relative crop](../../../../glossary.md#target-relative-crop), and unselected [layout](../../../../glossary.md#layout) variants | **Unchanged later Shape/layout work.** |
| Unknown/custom kind, wrong attributes class/[cardinality](../../../../glossary.md#term-cardinality), missing canonical output, or [descriptor](../../../../glossary.md#term-descriptor) contradiction | **Deterministic [preflight](../../../../glossary.md#term-preflight) failure.** |

This classification does not add all remaining gradient families and does not create detailed
follow-up task specifications.

### [Preflight](../../../../glossary.md#term-preflight) and failure ordering

Preserve the existing order:

1. validate `GraphCompiler` [arguments](../../../../glossary.md#term-argument) and mode/request agreement without creating a [Tensor](../../../../glossary.md#tensor);
2. inventory the complete original forward [DAG](../../../../glossary.md#term-directed-acyclic-graph-dag);
3. validate [objective](../../../../glossary.md#term-objective)/[target](../../../../glossary.md#term-target) membership, floating eligibility, and selected reachability;
4. validate every selected occurrence in deterministic producer/input order, including exact
   kind, attributes, output role, input role, descriptors, promotion, Shape, cotangent
   normalization, and any explicit local derivative convention;<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [descriptors glossary](../../../../glossary.md#term-descriptor); [Shape glossary](../../../../glossary.md#shape); [cotangent normalization glossary](../../../../glossary.md#cotangent-normalization); [local derivative glossary](../../../../glossary.md#derivative-and-chain-rule))</span>
5. reject the first unsupported fact with existing occurrence/output/input/kind/attributes/reason
   context; and
6. only after the complete selected slice succeeds, create the seed, generated zero/one splats, or
   a formula Tensor.<span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice); [seed glossary](../../../../glossary.md#term-seed))</span>

Known preflight rejection must consume no new [`TensorId`](../../../../glossary.md#tensorid), including an unsupported [operation](../../../../glossary.md#operation) after
an otherwise valid mixed-floating, division, direct-zero, or [mean](../../../../glossary.md#term-mean) [occurrence](../../../../glossary.md#term-occurrence). Later construction,
[capture](../../../../glossary.md#term-capture-meanings), [inference](../../../../glossary.md#term-inference), [validation](../../../../glossary.md#term-validation), or optimization failures retain the existing monotonic-ID
behavior.

### Determinism, [capture](../../../../glossary.md#term-capture-meanings), [validation](../../../../glossary.md#term-validation), and shared optimization

- Contribution insertion remains producer- and input-position-ordered, including repeated exact
  operands.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer))</span>
- [Shape](../../../../glossary.md#shape)/[DataType](../../../../glossary.md#data-type--datatype) [normalization](../../../../glossary.md#term-normalization) finishes before contribution insertion.
- [Scalar](../../../../glossary.md#term-scalar)-splat generation and ingress [binding](../../../../glossary.md#term-binding) follow deterministic first use.
- Forward and generated expressions are captured together once with the existing original-
  producer identity phase classification.
- Compiler 0002 inference and validation remain authoritative after capture and after every
  changed optimization candidate.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [inference glossary](../../../../glossary.md#term-inference))</span>
- The existing order remains mandatory canonicalization/validation, then exact rewrite, exact
  fold, whole-graph DCE, phase-local CSE, and whole-graph cleanup DCE once each when optional
  optimization is enabled.<span data-glossary-navigation="true"> ([canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [graph glossary](../../../../glossary.md#graph); [DCE glossary](../../../../glossary.md#term-dead-code-elimination-dce))</span>
- Existing rewrite/fold [matrices](../../../../glossary.md#term-matrix-meanings) and guards do not widen. Generated casts, divisions, `WHERE`
  nodes, and mean denominator expressions are ordinary candidates; they receive neither special
  protection nor special transformation.<span data-glossary-navigation="true"> ([nodes glossary](../../../../glossary.md#node); [mean glossary](../../../../glossary.md#term-mean))</span>
- No cross-phase [CSE](../../../../glossary.md#term-common-subexpression-elimination-cse), fixed point, new algebraic [identity](../../../../glossary.md#term-identity-meanings), [reassociation](../../../../glossary.md#term-reassociation), or pass is introduced.

### First-order and higher-order implications

This task implements first-order rules only. Generated formulas remain ordinary [differentiable](../../../../glossary.md#derivative-and-chain-rule)
[Tensor](../../../../glossary.md#tensor) expressions. [Compiler](../../../../glossary.md#term-compiler) 0006 must still define create-[graph](../../../../glossary.md#graph)/[derivative](../../../../glossary.md#term-derivative) order, disconnected
higher-order zero behavior, phase-plus-order representation, and complete formula-[operation](../../../../glossary.md#operation)
coverage.

0004B does not claim second derivatives, differentiate a local convention, or retrofit Tensor
[gradient](../../../../glossary.md#term-gradient) state. No selected row needs a hidden auxiliary output, but [preflight](../../../../glossary.md#term-preflight) and [capture](../../../../glossary.md#term-capture-meanings)
continue inventorying every [canonical](../../../../glossary.md#term-canonical-form) [producer](../../../../glossary.md#term-producer) output slot exactly as Compiler 0004 requires.

## Out of [scope](../../../../glossary.md#term-scope)

- any row outside the exact `SUPPORTED_0004B` [matrix](../../../../glossary.md#term-matrix-meanings)
- changing model operation semantics, public Tensor methods, operation kinds/attributes, producer
  outputs, descriptors, cast conversion semantics, comparison semantics, or Shape contracts<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [operation glossary](../../../../glossary.md#operation); [Tensor glossary](../../../../glossary.md#tensor); [producer glossary](../../../../glossary.md#term-producer); [descriptors glossary](../../../../glossary.md#term-descriptor); [Shape glossary](../../../../glossary.md#shape))</span>
- a gradient-specific arithmetic, comparison, cast, reduction, NaN/infinity, folding, rewriting,
  validation, or execution contract<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [validation glossary](../../../../glossary.md#term-validation))</span>
- softmax, statistical/norm, product, activation, normalization, pooling, loss, attention,
  ordering, indexing/scatter, window, convolution, dropout, or batch-normalization rules<span data-glossary-navigation="true"> ([softmax glossary](../../../../glossary.md#softmax--log-softmax); [norm glossary](../../../../glossary.md#term-norm); [activation glossary](../../../../glossary.md#term-activation); [normalization glossary](../../../../glossary.md#term-normalization); [pooling glossary](../../../../glossary.md#term-pooling); [loss glossary](../../../../glossary.md#loss); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
- a public [derivative](../../../../glossary.md#term-derivative)-policy enum, [registry](../../../../glossary.md#term-registry), [facade](../../../../glossary.md#term-facade), request, result, [compile](../../../../glossary.md#compile) [API](../../../../glossary.md#term-application-programming-interface-api), or [artifact](../../../../glossary.md#term-artifact)
- explicit seeds, non-scalar objectives, disconnected-target public policy, Jacobians, vector-
  Jacobian products, create-graph, or higher derivative order<span data-glossary-navigation="true"> ([seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [scalar glossary](../../../../glossary.md#term-scalar); [target glossary](../../../../glossary.md#term-target); [vector glossary](../../../../glossary.md#term-vector); [Jacobian glossary](../../../../glossary.md#term-jacobian); [graph glossary](../../../../glossary.md#graph))</span>
- model-owned derivative dispatch, Tensor gradient/backward state, a second gradient algebra,
  captured-value-to-Tensor conversion, direct node construction, tape, or global cache<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch); [node glossary](../../../../glossary.md#node); [cache glossary](../../../../glossary.md#term-cache))</span>
- new graph rewrite/fold rows, cross-phase CSE, approximate algebra, reassociation, or pass
  iteration<span data-glossary-navigation="true"> ([CSE glossary](../../../../glossary.md#term-common-subexpression-elimination-cse); [reassociation glossary](../../../../glossary.md#term-reassociation))</span>
- runtime, prepare, engine, planning orchestration, trace emission, backend lowering, kernels,
  execution, physical saved-value lifetime, or numerical conformance<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [engine glossary](../../../../glossary.md#term-engine); [planning glossary](../../../../glossary.md#term-planning); [trace glossary](../../../../glossary.md#trace); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [kernels glossary](../../../../glossary.md#kernel); [lifetime glossary](../../../../glossary.md#term-lifetime); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- architecture, ADR, dependency, Gradle/build, Java-version, architecture-test source,
  backend-conformance source, or integration-test source changes<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- creating a detailed [Compiler](../../../../glossary.md#term-compiler) 0005/0006 or later [gradient](../../../../glossary.md#term-gradient)-family task specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>
- [ADR 0009](../../../../design/decisions/0009-compiler-owned-pre-capture-tensor-expression-autograd.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- [Autograd strategy](../../../../design/notes/autograd-strategy.md)<span data-glossary-navigation="true"> ([Autograd glossary](../../../../glossary.md#autograd))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Compiler master plan](../master-plan.md)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>
- [Compiler 0004](0004-compiler-owned-pre-capture-autograd-and-combined-graph-compilation.md)
- [Compiler 0004A](0004a-exact-composition-gradient-rule-extensions.md)
- [Adjoint expressibility audit](../../model/adjoint-expressibility-audit.md)<span data-glossary-navigation="true"> ([Adjoint glossary](../../../../glossary.md#term-adjoint))</span>
- [Model capability closure audit](../../model/model-capability-contract-closure-audit.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Model 0025](../../model/tasks/0025-canonical-tensor-producer-outputs.md)
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- [Training API](../../../../api/training-api.md)
- [Autograd user guide](../../../../user-guide/autograd.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `modules/model` owns the one shared operation algebra, inference inputs, public Tensor
  vocabulary, and numerical semantics for forward and generated expressions.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [inference glossary](../../../../glossary.md#term-inference))</span>
- `modules/compiler` owns local differentiation-rule selection, preflight, reverse accumulation,
  cotangent normalization, formula construction, combined capture, and validation orchestration.<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [cotangent normalization glossary](../../../../glossary.md#cotangent-normalization); [validation glossary](../../../../glossary.md#term-validation))</span>
- Formula construction uses only current public [Tensor](../../../../glossary.md#tensor) operations and [immutable](../../../../glossary.md#term-immutable) [model](../../../../glossary.md#term-model) values.
- Compiler identity maps and generated constant caches are request-local ephemeral bookkeeping,
  not graph IR, public state, or registries.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [identity glossary](../../../../glossary.md#term-identity-meanings); [constant glossary](../../../../glossary.md#term-constant); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- No [derivative](../../../../glossary.md#term-derivative) [dispatch](../../../../glossary.md#term-dispatch) moves into model and no [gradient](../../../../glossary.md#term-gradient) [lifecycle](../../../../glossary.md#lifecycle) state moves into Tensor.
- [Graph](../../../../glossary.md#graph)-local IDs are assigned exactly once during the existing combined [capture](../../../../glossary.md#term-capture-meanings).
- No [dependency](../../../../glossary.md#term-dependency) or [module](../../../../glossary.md#term-module) boundary changes.
- If implementation requires a model/public API, a new algebra or optimization rule, an
  architecture change, or a seventeenth path, stop and report the conflict.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Package impact

Existing package:

- `io.github.pho001.synaptik.compiler` remains the single package-private compiler front-end
  boundary.

Type placement:

- `AutogradPreflight` extends the closed selected-role, mixed-floating normalization, division,
  direct-zero convention, and MEAN guards.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization))</span>
- `ElementwiseGradientRules` owns mixed arithmetic/cast/WHERE normalization, division formulas,
  and the three direct-zero conventions.
- `ReductionGradientRules` owns ordinary and masked [MEAN](../../../../glossary.md#term-mean) local formulas.
- `LinearAlgebraGradientRules` extends only selected-role cotangent normalization after its
  existing rank/Shape restoration.<span data-glossary-navigation="true"> ([cotangent normalization glossary](../../../../glossary.md#cotangent-normalization); [rank glossary](../../../../glossary.md#term-rank); [Shape glossary](../../../../glossary.md#shape))</span>
- `FirstOrderAutograd` extends only its existing closed typed dispatch so preflight-approved
  ordinary and masked MEAN occurrences reach <span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch); [preflight glossary](../../../../glossary.md#term-preflight))</span>`ReductionGradientRules`.

No package or public type is added. Tests remain in the mirrored [compiler](../../../../glossary.md#term-compiler) package.

## Affected files

Expected production files:

- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/AutogradPreflight.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ElementwiseGradientRules.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ReductionGradientRules.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/LinearAlgebraGradientRules.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/FirstOrderAutograd.java`

Expected tests:

- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/AutogradPreflightTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/GradientRulesTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/FirstOrderAutogradTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/GraphCompilerTest.java`

Expected documentation and [planning](../../../../glossary.md#term-planning) files during implementation:

- `docs/api/compile-api.md`
- `docs/design/notes/autograd-strategy.md`
- `docs/user-guide/autograd.md`
- `docs/glossary.md`
- `docs/planning/modules/compiler/tasks/0004b-shared-algebra-cotangent-normalization-and-local-derivative-rules.md`
- `docs/planning/modules/compiler/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification and record reasoned no-change conclusions for the [Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api), Training
API, `ARCHITECTURE.md`, focused architecture documents, ADRs, architecture tests, [model](../../../../glossary.md#term-model)/config/
planning/[trace](../../../../glossary.md#trace)/[runtime](../../../../glossary.md#term-runtime)/[prepare](../../../../glossary.md#prepare)/[engine](../../../../glossary.md#term-engine)/training/[backend](../../../../glossary.md#backend) sources and tests, backend [conformance](../../../../glossary.md#term-conformance),
integration tests, and [Gradle](../../../../glossary.md#term-gradle)/build files.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most:

- 5 [compiler](../../../../glossary.md#term-compiler) production files,
- 4 compiler test files, and
- 7 documentation/[planning](../../../../glossary.md#term-planning) files,

for a strict ceiling of 16 touched paths.

The fifth production path is mechanically necessary because the current
`FirstOrderAutograd.apply()` [dispatch](../../../../glossary.md#term-dispatch) excludes `AggregateReductionKind.MEAN`; implementing the
selected ordinary/masked [MEAN](../../../../glossary.md#term-mean) formulas only in `ReductionGradientRules` would leave them
unreachable. The user's standing automatic approval for necessary higher path counts authorizes
this exact one-path increase. It does not authorize another [operation](../../../../glossary.md#operation) family, formula, test,
documentation surface, or architectural change.

If another path, [model](../../../../glossary.md#term-model) [API](../../../../glossary.md#term-application-programming-interface-api), public compiler type, test-infrastructure change, new optimization
rule, or broader differentiation [matrix](../../../../glossary.md#term-matrix-meanings) is required, stop and propose a follow-up or task
revision.

## Acceptance criteria

### [Preflight](../../../../glossary.md#term-preflight) and request contract

- Existing 0004/0004A success, failure, and zero-ID behavior remains unchanged.
- A target may differ from the objective's floating type only when every selected role has a
  complete ordinary <span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [objective glossary](../../../../glossary.md#term-objective))</span>`sumToShape`/`cast` [normalization](../../../../glossary.md#term-normalization) path under this task.
- Every selected contribution has the exact target input Shape and floating DataType before
  accumulation.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [DataType glossary](../../../../glossary.md#data-type--datatype))</span>
- Preflight admits only the exact new kind/attributes/output/input/descriptor/Shape/local-
  convention rows.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Every unsupported row in the classification fails before any [derivative](../../../../glossary.md#term-derivative) [Tensor](../../../../glossary.md#tensor) ID is consumed.
- Diagnostics retain deterministic occurrence, output-role, input-role, kind, attributes, and
  reason context.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence))</span>

### Formula and local-convention behavior

- Mixed `ADD`/`SUB`/`MUL`, `WHERE`, and every selected [`MATMUL`](../../../../glossary.md#matrix-multiplication--matmul) rank/role unbroadcast in the
  promoted type and cast once afterward; same-type formulas add no cast.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank))</span>
- Floating cross-type `CAST` returns exactly one ordinary reverse cast to the source type.
- [Binary](../../../../glossary.md#term-binary-descriptor-and-metadata)/[scalar](../../../../glossary.md#term-scalar) `DIV` constructs the exact local formulas specified above across BFLOAT16,
  FLOAT32, and FLOAT64; generated nodes retain ordinary model semantics and shared optimization
  eligibility without a singularity-specific policy.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [nodes glossary](../../../../glossary.md#node); [model glossary](../../../../glossary.md#term-model))</span>
- `FLOOR`, `CEIL`, and `SIGN` return direct exact zeros without an incoming-cotangent
  multiplication or floating comparison.<span data-glossary-navigation="true"> ([cotangent glossary](../../../../glossary.md#cotangent))</span>
- Ordinary full/single-[axis](../../../../glossary.md#term-axis)/multi-axis [`MEAN`](../../../../glossary.md#term-mean) constructs its denominator by reducing `O_x` for
  static, dynamic, expression, zero, and empty-axis-list geometry; no host count or exact-integer
  threshold exists.<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host))</span>
- Masked `MEAN` constructs its true-count through `WHERE` plus `SUM`, routes its quotient through
  a final ordinary `WHERE`, and applies the explicit all-false-slice zero-cotangent convention
  without claiming execution suppression.<span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice))</span>
- `FirstOrderAutograd` routes only the newly [preflight](../../../../glossary.md#term-preflight)-approved ordinary and masked `MEAN`
  occurrences to `ReductionGradientRules`; existing dispatch and formula ownership remain
  otherwise unchanged.<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Repeated operands retain repeated positional contributions in deterministic order.

### Shared algebra, [capture](../../../../glossary.md#term-capture-meanings), [validation](../../../../glossary.md#term-validation), and optimization

- Forward and generated expressions obey the same model algebra, inference, validation, and
  numerical contracts.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [inference glossary](../../../../glossary.md#term-inference))</span>
- Existing generated zero/one constants remain storage-free, non-gradient, explicitly bound
  logical splats with unchanged exact bits and deterministic order.<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Forward ingress and all generated derivative bindings retain deterministic order and fixed
  sources remain absent from bindable inputs.<span data-glossary-navigation="true"> ([derivative glossary](../../../../glossary.md#term-derivative); [bindings glossary](../../../../glossary.md#term-binding))</span>
- One combined capture preserves every original/generated phase, canonical output identity,
  output de-duplication, and target role.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [identity glossary](../../../../glossary.md#term-identity-meanings); [target glossary](../../../../glossary.md#term-target))</span>
- [Compiler](../../../../glossary.md#term-compiler) 0002 validation runs at the existing boundaries.
- Existing exact rewrite/fold matrices, phase-local CSE, whole-graph DCE, and pass order remain
  unchanged and apply under the same guards to both phases.<span data-glossary-navigation="true"> ([matrices glossary](../../../../glossary.md#term-matrix-meanings); [CSE glossary](../../../../glossary.md#term-common-subexpression-elimination-cse); [graph glossary](../../../../glossary.md#graph); [DCE glossary](../../../../glossary.md#term-dead-code-elimination-dce))</span>
- No public compiler declaration, registry, second algebra, direct IR formula, Tensor gradient
  state, gradient-only pass, or new dependency is introduced.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [Tensor glossary](../../../../glossary.md#tensor); [dependency glossary](../../../../glossary.md#term-dependency))</span>

### Documentation and completion

- Every changed Java declaration has meaningful complete Javadoc for local differentiation,
  cotangent normalization, constants, ordering, types, Shape, failures, and ownership.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [cotangent normalization glossary](../../../../glossary.md#cotangent-normalization); [constants glossary](../../../../glossary.md#term-constant); [Shape glossary](../../../../glossary.md#shape); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Compile API, autograd strategy, user guide, glossary, task, master plan, and roadmap agree that
  forward/backward share one algebra and on the exact selected/deferred matrix.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- Tensor API and Training API are reviewed unchanged because no public Tensor or training
  lifecycle contract changes.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- The independent clean-context documentation pass reuses successful Java evidence unless it
  changes executable behavior or records a concrete stale-evidence risk.
- Exact 16-path scope, statuses, links, anchors, fences, final newlines, terminology, and
  whitespace pass.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>
- The [compiler](../../../../glossary.md#term-compiler) transformation/[autograd](../../../../glossary.md#autograd) capability [checkpoint](../../../../glossary.md#term-checkpoint) passes before this task is marked
  `Complete`.

## Tests / [validation](../../../../glossary.md#term-validation)

Implementation-focused tests must cover:

- all three unordered mixed-floating type pairs, with the narrower selected input in every
  applicable operand or branch position;
- broadcast-before-cast ordering for [binary](../../../../glossary.md#term-binary-descriptor-and-metadata)/WHERE and every [`MATMUL`](../../../../glossary.md#matrix-multiplication--matmul) [rank](../../../../glossary.md#term-rank) pairing;
- cross-floating cast reversal, shared/repeated mixed inputs, and mixed contribution accumulation;
- exact binary/scalar division formula structure with optional optimization disabled, plus proof
  that enabled optimization uses only the existing shared pass matrix and order;<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- FLOOR/CEIL/SIGN direct-zero structure with no comparison or `g * 0` [occurrence](../../../../glossary.md#term-occurrence);
- full, single-axis, multi-axis, point-domain, static-empty, dynamic, expression-extent, and
  masked-mean variants, including all-false masks and absence of exact-count thresholds;<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [extent glossary](../../../../glossary.md#term-extent); [mean glossary](../../../../glossary.md#term-mean); [masks glossary](../../../../glossary.md#term-boolean-mask))</span>
- existing zero/one exact bits, explicit ingress, deterministic reuse/order, and no storage
  inference;<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference))</span>
- zero-ID [preflight](../../../../glossary.md#term-preflight) rejection for every deferred/blocked classification; and
- end-to-end modes, one capture, phases, result roles, constants, shared optional optimization,
  and 0004/0004A regressions.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [constants glossary](../../../../glossary.md#term-constant))</span>

After executable code stabilizes, [run](../../../../glossary.md#term-run) once:

```bash
./gradlew :modules:compiler:test
git diff --check
```

The implementation context must hand the exact test evidence and unchanged executable state to a
separate clean-context documentation-focused agent. That pass runs:

```bash
./gradlew :modules:compiler:javadoc
node /tmp/validate_synaptik_markdown.js
git diff --check
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git status --short
```

If the established [Node](../../../../glossary.md#node) validator is unavailable, use or recreate the equivalent repository-wide
check for local Markdown links and anchors, balanced fences, final newlines, and trailing
whitespace, and record the exact command and counts.

After implementation and documentation validation are stable, run the [compiler](../../../../glossary.md#term-compiler)
transformation/[autograd](../../../../glossary.md#autograd) capability [checkpoint](../../../../glossary.md#term-checkpoint):

```bash
./gradlew test :testing:architecture-tests:test
```

Record XML test counts and [Gradle](../../../../glossary.md#term-gradle) outcomes. Do not rerun successful [module](../../../../glossary.md#term-module) tests in the
documentation pass without executable changes. Repository-wide validation after this checkpoint
is deferred to [CI](../../../../glossary.md#term-continuous-integration-ci) unless executable code changes or a concrete repository-wide risk appears.

[Planning](../../../../glossary.md#term-planning) this task does not run Java tests or the checkpoint.

## Dependencies

- [Compiler](../../../../glossary.md#term-compiler) 0004A — Complete
- Compiler 0004, 0001–0003B — Complete
- [Model](../../../../glossary.md#term-model) 0025 — Complete
- Current model operation semantics and public Tensor operations used by the selected formulas —
  Complete<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Config 0002 and [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0009 — Complete

No architecture or model change blocks the selected [matrix](../../../../glossary.md#term-matrix-meanings). Current dynamic/expression [Shape](../../../../glossary.md#shape),
logical-splat expansion, ordinary reduction, [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) arithmetic, `WHERE`, `sumToShape`, and cast
contracts can represent every selected formula.

## Follow-up tasks

- Later cohesive differentiation tasks may address the explicitly classified unary, activation,
  product, statistical/norm, softmax, normalization, pooling, loss, attention, ordering,
  indexing/scatter, window/structured-linear, stochastic, and batch-normalization families when
  each reaches the progressive-planning frontier. Do not create those detailed specifications
  now.<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary); [activation glossary](../../../../glossary.md#term-activation); [norm glossary](../../../../glossary.md#term-norm); [softmax glossary](../../../../glossary.md#softmax--log-softmax); [normalization glossary](../../../../glossary.md#term-normalization); [pooling glossary](../../../../glossary.md#term-pooling); [loss glossary](../../../../glossary.md#loss); [linear glossary](../../../../glossary.md#term-linear-meanings); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [planning glossary](../../../../glossary.md#term-planning))</span>
- General model semantic gaps, including floating comparisons and incomplete softmax/power/unary
  edge contracts, must be resolved by their model owner before dependent differentiation rules.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [owner glossary](../../../../glossary.md#term-owner))</span>
- Compiler 0005 remains the next ordered compiler row after successful 0004B and the capability
  checkpoint. It owns publication, planning orchestration, diagnostics, and immutable compile
  artifacts.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [publication glossary](../../../../glossary.md#term-publication); [immutable glossary](../../../../glossary.md#term-immutable); [compile artifacts glossary](../../../../glossary.md#compile-artifacts))</span>
- Compiler 0006 remains Draft and owns public functional requests, explicit seeds, create-graph/
  derivative order, and higher-order representation.<span data-glossary-navigation="true"> ([seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [graph glossary](../../../../glossary.md#graph); [derivative glossary](../../../../glossary.md#term-derivative))</span>

## Architecture impact

Expected impact: None.

This task fills the accepted [compiler](../../../../glossary.md#term-compiler)-owned local differentiation extension point while preserving
one [model](../../../../glossary.md#term-model) algebra for the combined [graph](../../../../glossary.md#graph). It changes no architecture rule, [dependency direction](../../../../glossary.md#java-platform-and-build-tools),
public [lifecycle](../../../../glossary.md#lifecycle), [graph phase](../../../../glossary.md#graph-phase) vocabulary, model semantics, or [backend](../../../../glossary.md#backend)/[runtime](../../../../glossary.md#term-runtime) responsibility. If
implementation requires any such change, stop and report the exact conflict.

## Implementation prompt

Use this prompt in a separate clean implementation context:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, the focused compiler/autograd/training/lifecycle/dependency
architecture documents, ADR 0009, documentation rules and applicable profiles, the planning guide
and roadmap, the compiler master plan, completed Compiler 0001–0004A, Model 0025 and the model
audits, current Compile/Tensor/Training APIs, autograd strategy/user guide/glossary, this task
specification, and the complete current compiler/model source and tests needed to verify the
selected and deferred operation matrix.

Implement Compiler 0004B exactly as specified. Preserve the one shared forward/backward Tensor
algebra and the existing package-private request, one-capture, identity, accumulation, phase,
validation, and exact optimization contracts. Add only the closed mixed-floating cotangent
normalization, division, direct-zero local-convention, and ordinary/masked mean matrix. Do not add
a public surface, registry, second algebra, gradient-only numerical or optimization policy, model
API, floating-comparison-dependent rule, later operation family, runtime/backend behavior, build
change, or seventeenth path. Extend `FirstOrderAutograd` only enough to route the selected ordinary
and masked MEAN occurrences to `ReductionGradientRules`. Stop on an architecture, shared-semantic,
formula-expressibility, or maximum-scope conflict.

After executable code stabilizes, run the compiler module tests once and hand the exact diff and
evidence to a distinct clean-context documentation-focused agent. That pass must independently
finalize affected Javadocs, Compile API, autograd strategy, user guide, glossary impact, planning
status/evidence, links, and no-change conclusions in the same overall change. It must reuse the
successful Java evidence unless executable behavior changes or a concrete rerun risk is recorded.
Run final Javadoc/documentation/scope checks, then run the recorded compiler
transformation/autograd capability checkpoint before marking the task Complete.

Update this task with local decisions, limitations, exact validation evidence, implementation
notes, completion summary, and final status. Do not create a later task specification, commit, or
push.
```

## Local decisions

- The combined graph has one shared model algebra; Compiler 0004B adds local differentiation
  formulas, explicit local conventions, and cotangent normalization only.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [cotangent normalization glossary](../../../../glossary.md#cotangent-normalization))</span>
- Mixed cotangents use ordinary `sumToShape` followed by ordinary `cast`; the operations inherit
  the same semantics and optimization guards as forward occurrences.
- Binary/scalar DIV uses its ordinary public expression formula and receives no separate gradient
  singularity or exceptional-value policy.<span data-glossary-navigation="true"> ([Binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [scalar glossary](../../../../glossary.md#term-scalar); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- FLOOR, CEIL, and SIGN use the genuine local first-order convention of a direct exact-zero
  cotangent.<span data-glossary-navigation="true"> ([cotangent glossary](../../../../glossary.md#cotangent))</span>
- Ordinary MEAN derives its denominator by reducing generated logical ones. Dynamic and expression
  selected extents are representable, so no static-count gate or DataType-specific exact-count
  threshold remains.<span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype))</span>
- Masked [MEAN](../../../../glossary.md#term-mean) uses an ordinary final `WHERE` plus an explicit all-false-slice zero-cotangent local
  convention and makes no branch-evaluation claim.<span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice); [evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- `FirstOrderAutograd.java` is the necessary fifth production path because its current closed
  dispatch sends only SUM and CUM_SUM to <span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch))</span>`ReductionGradientRules`; the authorized change adds only
  ordinary/masked MEAN routing and leaves formula ownership in <span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>`ReductionGradientRules`.
- A clean implementation attempt stopped before edits after verifying that the original
  four-production-file scope could not make the selected MEAN rules reachable. Under the user's
  standing automatic approval for necessary higher path counts, the exact production count is
  therefore five and the exact total ceiling is sixteen.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>
- Floating-comparison-dependent formulas remain blocked by the general model comparison contract;
  independently cohesive families remain deferred rather than widening 0004B.<span data-glossary-navigation="true"> ([widening glossary](../../../../glossary.md#term-widening))</span>

## Known limitations

- The request remains one [scalar objective](../../../../glossary.md#loss) with one implicit unit [seed](../../../../glossary.md#term-seed) and connected [targets](../../../../glossary.md#term-target).
- Cotangent conversion is expressed by current public CAST and therefore promises no conversion
  behavior beyond the shared model contract.<span data-glossary-navigation="true"> ([Cotangent glossary](../../../../glossary.md#cotangent); [model glossary](../../../../glossary.md#term-model))</span>
- `SUPPORTED_0004B` is not a complete operation inventory for autograd; every classified later or
  blocked family remains fail-closed.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [autograd glossary](../../../../glossary.md#autograd); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- The task implements only first-order formulas. Compiler 0006 still owns derivative order and
  higher-order lifecycle.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [derivative glossary](../../../../glossary.md#term-derivative); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) evidence:

- required architecture, ADR, documentation profiles, planning contracts, completed compiler
  tasks, model closure/audit artifacts, APIs/guides/glossary, compiler production/tests, and model
  operation/expression semantics were audited;<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [compiler glossary](../../../../glossary.md#term-compiler); [artifacts glossary](../../../../glossary.md#term-artifact); [operation glossary](../../../../glossary.md#operation))</span>
- the current logical-splat, `expand`, reduction, and symbolic Shape contracts can express
  ordinary and masked MEAN denominators for static, dynamic, and expression extents;<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [MEAN glossary](../../../../glossary.md#term-mean))</span>
- no architecture or [dependency](../../../../glossary.md#term-dependency) conflict was found;
- every selected formula uses the shared public Tensor algebra and every selected contribution
  has an ordinary Shape/DataType normalization path;<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [DataType glossary](../../../../glossary.md#data-type--datatype); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- general [model](../../../../glossary.md#term-model)-semantics gaps and later cohesive families are classified explicitly;
- a clean implementation attempt made no edits and stopped after verifying from
  `FirstOrderAutograd.apply()` that ordinary/masked MEAN would otherwise fall through the closed
  dispatch instead of reaching <span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch))</span>`ReductionGradientRules`;
- the user's standing automatic approval for necessary higher path counts adds only
  `FirstOrderAutograd.java`, fixing the implementation ceiling at 16 paths: five compiler
  production files, four compiler test files, and seven documentation/planning files; and
- Java tests, Javadoc, and the capability checkpoint were not run during the planning-only pass;
  their final implementation/documentation outcomes are recorded below.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>

Implementation evidence:

- implementation context
  `/root/implement_compiler_0004b_shared_algebra_retry` changed exactly the five authorized
  compiler production paths and four authorized compiler test paths;
- its final `./gradlew :modules:compiler:test` [run](../../../../glossary.md#term-run) passed with `BUILD SUCCESSFUL in 1s`,
  13 actionable tasks (`1 executed`, `12 up-to-date`), and 18 XML suites containing 136 tests,
  0 skipped, 0 failures, and 0 errors;
- its focused development run of `AutogradPreflightTest`, `GradientRulesTest`,
  `FirstOrderAutogradTest`, and `GraphCompilerTest` also passed after an earlier failure exposed
  and the implementation fixed unreachable generated-seed ingress for a direct-zero target
  gradient; and<span data-glossary-navigation="true"> ([seed glossary](../../../../glossary.md#term-seed); [target glossary](../../../../glossary.md#term-target); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- no executable Java or test changed after that final module evidence. The documentation pass
  therefore reused it rather than repeating the successful module suite.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>

Documentation evidence:

- clean documentation-focused context
  `/root/implement_compiler_0004b_shared_algebra_retry/compiler_0004b_docs` independently read
  the architecture and ADR contracts, General/API-Javadoc/Planning/User-guide/Example profiles,
  planning guide, task/master/roadmap, actual five-production/four-test diff, Compile/Tensor/
  Training APIs, autograd strategy/user guide/glossary, and directly relevant model/compiler
  contracts;<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile); [autograd glossary](../../../../glossary.md#autograd))</span>
- it finalized affected Javadocs plus exactly `docs/api/compile-api.md`,
  `docs/design/notes/autograd-strategy.md`, `docs/user-guide/autograd.md`,
  `docs/glossary.md`, this task, the compiler master plan, and the roadmap without changing
  executable Java or tests;
- `./gradlew :modules:compiler:javadoc` passed after final [Javadoc](../../../../glossary.md#term-javadoc) edits with
  `BUILD SUCCESSFUL in 1s`, 7 actionable tasks (`2 executed`, `5 up-to-date`), and no warning;
- `node /tmp/validate_synaptik_markdown.js` passed repository-wide: 241 Markdown files,
  4,429 local links, 273 anchors, and 2,966 fence markers;
- `git diff --check` passed;
- the sorted union of tracked modifications and untracked files contains exactly 16 paths:
  five compiler production files, four compiler tests, and seven documentation/planning files;
- manual review confirmed balanced fences, valid local links/anchors, final newlines, no trailing
  whitespace, shared-algebra wording, exact selected/deferred classification, and no unexpected
  public, architecture, dependency, build, backend, runtime, or execution claim;<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Tensor API remained unchanged because 0004B adds no public Tensor method or model semantic;
  Training API remained unchanged because it adds no public gradient/training request,
  publication, optimizer, session, prepare, run, or lifecycle behavior;<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [optimizer glossary](../../../../glossary.md#term-optimizer); [prepare glossary](../../../../glossary.md#prepare); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- `ARCHITECTURE.md`, focused architecture pages, and ADR 0009 remained unchanged because the
  implementation fills the accepted compiler-owned pre-capture extension point without changing
  ownership, dependency direction, phase/capture structure, or the one-algebra contract;<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools))</span>
- architecture tests, model/config/planning/trace/runtime/prepare/engine/training/backend sources
  and tests, backend conformance, integration tests, and Gradle/build files remained unchanged
  because there is no boundary, public model, cross-module, execution, or build behavior change;
  and<span data-glossary-navigation="true"> ([trace glossary](../../../../glossary.md#trace); [engine glossary](../../../../glossary.md#term-engine); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- after the finalized documentation [handoff](../../../../glossary.md#term-handoff), implementation context
  `/root/implement_compiler_0004b_shared_algebra_retry` ran
  `./gradlew test :testing:architecture-tests:test`; it passed with `BUILD SUCCESSFUL in 2s`,
  48 actionable tasks (`6 executed`, `42 up-to-date`), and a stored configuration [cache](../../../../glossary.md#term-cache);
- the checkpoint XML audit found 167 suites and 1,275 tests with 0 skipped, 0 failures, and
  0 errors: backend-contract 4 suites/22 tests, config 4/17, model 127/1,018, planning 8/63,
  compiler 18/136, trace 3/16, and architecture-tests 3/3; and
- no executable Java changed before or after that checkpoint. Task, master-plan, and roadmap
  status are synchronized to Complete; Compiler 0005 and 0006 remain Draft, and neither has a
  detailed task specification.

## Implementation notes

- `AutogradPreflight` now accepts floating targets independently of objective type and validates
  only the exact selected 0004B binary, scalar DIV, WHERE, floating CAST, direct-zero unary,
  ordinary/masked MEAN, and mixed-floating MATMUL rows before derivative allocation.<span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target); [objective glossary](../../../../glossary.md#term-objective); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [scalar glossary](../../../../glossary.md#term-scalar); [unary glossary](../../../../glossary.md#term-unary); [MEAN glossary](../../../../glossary.md#term-mean); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [derivative glossary](../../../../glossary.md#term-derivative))</span>
- `ElementwiseGradientRules` normalizes selected mixed contributions with ordinary
  `sumToShape` followed by ordinary `cast` when required, constructs the exact binary/scalar DIV
  formulas, and returns direct exact zeros for FLOOR/CEIL/SIGN.
- `LinearAlgebraGradientRules` retains all four MATMUL rank formulas, reverses batch broadcasting,
  and casts once to the selected operand type when promotion requires it.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [broadcasting glossary](../../../../glossary.md#broadcasting))</span>
- `ReductionGradientRules` derives ordinary MEAN counts by reducing logical ones and masked MEAN
  true counts with ordinary WHERE/SUM, then uses the final ordinary WHERE for the selected
  all-false-slice zero convention.<span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice))</span>
- `FirstOrderAutograd` routes preflight-approved ordinary and masked MEAN occurrences to the
  reduction owner. It also retains only generated constant bindings reachable from returned
  gradient expressions, so direct-zero target gradients do not expose an unreachable seed input.<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [owner glossary](../../../../glossary.md#term-owner); [constant glossary](../../../../glossary.md#term-constant); [bindings glossary](../../../../glossary.md#term-binding); [gradient glossary](../../../../glossary.md#term-gradient); [seed glossary](../../../../glossary.md#term-seed))</span>
- Focused tests extend preflight, formula structure, mixed-floating normalization, direct-zero,
  static/dynamic/expression/empty MEAN, dispatch, and both optimization-mode coverage while
  preserving existing 0004/0004A behavior.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- The independent documentation pass finalized all five affected implementation contracts,
  synchronized the four explanatory/public-status surfaces, added the reusable cotangent-
  normalization glossary distinction, and recorded reasoned no-change conclusions for every
  reviewed out-of-scope surface.<span data-glossary-navigation="true"> ([cotangent glossary](../../../../glossary.md#cotangent); [scope glossary](../../../../glossary.md#term-scope))</span>

## Completion summary

- Completed changes: added the closed 0004B mixed-floating cotangent-normalization, binary/scalar
  DIV, direct-zero FLOOR/CEIL/SIGN, ordinary/masked MEAN, and closed MEAN-dispatch behavior while
  preserving one shared Tensor algebra, fail-closed preflight, deterministic accumulation, one
  combined capture, and the existing validation/optimization contracts.<span data-glossary-navigation="true"> ([cotangent glossary](../../../../glossary.md#cotangent); [normalization glossary](../../../../glossary.md#term-normalization); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [scalar glossary](../../../../glossary.md#term-scalar); [MEAN glossary](../../../../glossary.md#term-mean); [dispatch glossary](../../../../glossary.md#term-dispatch); [Tensor glossary](../../../../glossary.md#tensor); [fail-closed glossary](../../../../glossary.md#term-fail-closed); [preflight glossary](../../../../glossary.md#term-preflight); [capture glossary](../../../../glossary.md#term-capture-meanings); [validation glossary](../../../../glossary.md#term-validation))</span>
- Files changed or created: exactly five compiler production files, four compiler test files, and
  seven documentation/planning files.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning))</span>
- Tests and validation: compiler module tests passed with 18 suites/136 tests; the required
  repository/architecture checkpoint passed with 167 suites/1,275 tests and no skipped tests,
  failures, or errors; compiler Javadoc, repository Markdown links/anchors/fences, exact scope,
  statuses, final newlines, whitespace, and <span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` passed.
- Documentation-agent review: clean context
  `/root/implement_compiler_0004b_shared_algebra_retry/compiler_0004b_docs` independently
  finalized all affected implementation Javadocs, Compile API, autograd strategy, user guide,
  glossary, task evidence, compiler master plan, and roadmap.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [autograd glossary](../../../../glossary.md#autograd))</span>
- Documentation impact: current status and the exact selected/deferred matrix now consistently
  describe one shared forward/backward algebra and the bounded 0004B local-rule extension.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- Javadoc review: all five affected implementation contracts were reviewed; four required final
  documentation edits, while `ElementwiseGradientRules` already contained the complete 0004B
  normalization/formula contract drafted with implementation.
- Glossary impact: added the reusable cotangent-normalization distinction and synchronized
  autograd, cast, logical-splat, masked-reduction, MATMUL, and implementation-status entries.<span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>
- Unresolved issues: None.
- Follow-up required: None. Compiler 0005 and 0006 remain their ordered Draft rows without
  detailed specifications.

Status: Complete
