# [Autograd](../../glossary.md#autograd) strategy


## Editorial reading context

This note records the bounded first-order strategy and the implementation frontier
named in its original text. Its “current” and “planned” statements are relative to
that frontier, not a replacement for today's supported API. Use the
[current training-graph explanation](../../architecture/training-graph.md) for the
present public path and [ADR 0009](../decisions/0009-compiler-owned-pre-capture-tensor-expression-autograd.md)
for the capture-ownership decision.

### Who constructs the derivative?

The application describes a forward objective using Tensor expressions. Compiler
is then the caller of ordinary Tensor methods that build derivative expressions;
Training does not supply a separate derivative language. A target can be a leaf,
meaning it has no producing expression, or a produced intermediate. A storage-free
seed is still a leaf: absence of a producer does not establish stored data,
constant status or a close obligation.

### Read the formulas as sensitivities, not update instructions

For a dimensionless scalar differentiable real objective $L$ and a nonempty finite softmax slice
$x_i$, with $1\le i\le n$, write $y_i$ for the saved forward probabilities and
$g_i=\partial L/\partial y_i$ for incoming sensitivities. The reverse formula is

$$
\frac{\partial L}{\partial x_i}
=y_i\left(g_i-\sum_{j=1}^{n}g_jy_j\right).
$$

For $y=(1/2,1/2)$ and $g=(1,0)$, the shared sum is $1/2$ and the input
sensitivities are $(1/4,-1/4)$. These dimensionless illustrative values explain
how both outputs affect each input. The implementation consumes exact saved
forward Tensor outputs and evaluates ordinary rounded operations; real-number
calculus does not authorize recomputation, special-value masking or a different
floating reduction order. The original rule inventory and exceptional-value
conventions below remain unchanged historical text.

> [`ARCHITECTURE.md`](../../../ARCHITECTURE.md) is authoritative; this note explains its
> compiler-owned pre-capture automatic-differentiation design.<span data-glossary-navigation="true"> ([compiler glossary](../../glossary.md#term-compiler); [capture glossary](../../glossary.md#term-capture-meanings))</span>

## Purpose and status

[Automatic differentiation](../../glossary.md#autograd) ([autograd](../../glossary.md#autograd)) derives [gradient](../../glossary.md#term-gradient) computations from a forward expression.
The design is accepted and its first bounded package-private implementation is current. [Model](../../glossary.md#term-model) task
0025 lets every [producer](../../glossary.md#term-producer) return its [canonical](../../glossary.md#term-canonical-form) exact [Tensor](../../glossary.md#tensor) wrapper for each output position.
[Compiler task 0004](../../planning/modules/compiler/tasks/0004-compiler-owned-pre-capture-autograd-and-combined-graph-compilation.md)
is Complete with the [scalar](../../glossary.md#term-scalar)-[objective](../../glossary.md#term-objective), implicit-unit-[seed](../../glossary.md#term-seed) first-order [graph](../../glossary.md#graph)-stage contract.
[Compiler task 0004A](../../planning/modules/compiler/tasks/0004a-exact-composition-gradient-rule-extensions.md)
is Complete with the first policy-free exact-composition rule extension.
[Compiler task 0004B](../../planning/modules/compiler/tasks/0004b-shared-algebra-cotangent-normalization-and-local-derivative-rules.md)
is Complete with mixed-floating [cotangent normalization](../../glossary.md#cotangent-normalization), division, direct-zero local conventions,
and ordinary or masked [mean](../../glossary.md#term-mean).
[Compiler task 0005A](../../planning/modules/compiler/tasks/0005a-derivative-policy-and-elementwise-activation-gradient-completion.md)
is Complete with the exact 48-kind [elementwise](../../glossary.md#term-elementwise)/[activation](../../glossary.md#term-activation) classification and formulas.
[Compiler task 0005B](../../planning/modules/compiler/tasks/0005b-reduction-scan-softmax-statistics-and-normalization-gradient-completion.md)
is Complete with [binding](../../glossary.md#term-binding)-aware expansion plus reduction, [scan](../../glossary.md#term-cumulative-scan), [softmax](../../glossary.md#softmax--log-softmax), statistics, [norm](../../glossary.md#term-norm), and
Layer/RMS/[batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters)-[normalization](../../glossary.md#term-normalization) rules. [Compiler](../../glossary.md#term-compiler) 0005 already provides current [immutable](../../glossary.md#term-immutable) [compile
artifacts](../../glossary.md#compile-artifacts), while public gradient requests, higher [derivatives](../../glossary.md#term-derivative), [optimizer](../../glossary.md#term-optimizer) updates, preparation,
and execution remain planned.

## [Mental model](../../glossary.md#term-mental-model)

```text
original forward Tensor expression DAG
  -> fail-closed compiler preflight
  -> reverse traversal using named compiler gradient rules
  -> ordinary Tensor expressions for contributions and accumulation
  -> combined forward + gradient Tensor expression DAG
  -> one phase-aware capture
  -> immutable combined compiler graph
  -> inference, validation, exact combined optimization, final validation
  -> publication and planning
```

[Tensor](../../glossary.md#tensor) expression objects are the single construction language before [capture](../../glossary.md#term-capture-meanings).
[`CompiledGraphModel`](../../glossary.md#compiled-graph--compiledgraphmodel) is the [immutable](../../glossary.md#term-immutable) [graph](../../glossary.md#graph) state after capture. The [compiler](../../glossary.md#term-compiler) does not convert
captured [`ValueId`](../../glossary.md#logical-valueid-versus-physical-memory-slot) values back into placeholder Tensors and does not maintain a second algebra.

Compiler task 0004 implements the general package-private entry [owner](../../glossary.md#term-owner) `GraphCompiler` and its
mode-neutral graph-stage result `GraphCompilation`, without adding a request aggregate.
`FORWARD_ONLY` produces no BACKWARD [nodes](../../glossary.md#node) and empty [gradient](../../glossary.md#term-gradient) results. Backward-capable modes may
produce the combined forward/[backward graph](../../glossary.md#backward-graph) described below. `GraphCompilation` is distinct from
the later `CompileArtifacts` aggregate.

## Current first-order request

One backward-capable internal request contains:

- one exact objective Tensor that is also a requested forward output, has scalar Shape, one
  floating data type, and gradient eligibility;<span data-glossary-navigation="true"> ([Tensor glossary](../../glossary.md#tensor); [scalar glossary](../../glossary.md#term-scalar); [Shape glossary](../../glossary.md#shape); [data type glossary](../../glossary.md#data-type--datatype))</span>
- one non-empty ordered list of exact-object-identity-unique target Tensors in the objective
  ancestry, each with a floating type, gradient eligibility, and a selected differentiable route;
  a target may have a different floating type from the objective; and<span data-glossary-navigation="true"> ([identity glossary](../../glossary.md#term-identity-meanings); [differentiable glossary](../../glossary.md#derivative-and-chain-rule))</span>
- one implicit [rank-zero](../../glossary.md#axis-rank-scalar-vector-and-matrix) positive-one [seed](../../glossary.md#term-seed) with the [objective](../../glossary.md#term-objective)'s exact type.

A [target](../../glossary.md#term-target) may be a [leaf](../../glossary.md#term-leaf), an intermediate, or the objective itself. Target order becomes
[gradient](../../glossary.md#term-gradient)-result-role order; it does not stop traversal needed by another upstream target. There
is no current caller-supplied seed, non-[scalar objective](../../glossary.md#loss), disconnected-target zero policy,
[vector-Jacobian product](../../glossary.md#term-vector-jacobian-product-vjp), public target/[publication](../../glossary.md#term-publication) request, or higher [derivative](../../glossary.md#term-derivative).

## [Ownership](../../glossary.md#term-ownership-meanings) and reverse accumulation

- `modules/model` owns public Tensor operations, exact producer occurrence identity, canonical
  output wrappers, and immutable indexed provenance. It owns no derivative rule.<span data-glossary-navigation="true"> ([occurrence glossary](../../glossary.md#term-occurrence); [canonical glossary](../../glossary.md#term-canonical-form); [immutable glossary](../../glossary.md#term-immutable); [provenance glossary](../../glossary.md#provenance); [derivative glossary](../../glossary.md#term-derivative))</span>
- `modules/compiler` owns preflight, output seeds, differentiation targets, rule dispatch,
  deterministic reverse traversal, contribution accumulation, phase-aware capture, validation,
  and combined optimization.<span data-glossary-navigation="true"> ([preflight glossary](../../glossary.md#term-preflight); [seeds glossary](../../glossary.md#commit-release-distribution-log-and-seed-meanings); [targets glossary](../../glossary.md#term-target); [capture glossary](../../glossary.md#term-capture-meanings); [validation glossary](../../glossary.md#term-validation))</span>
- Planning assigns backend ownership after expansion. Concrete backends prepare assigned regions.
  Runtime executes prepared work and never derives gradients.<span data-glossary-navigation="true"> ([Planning glossary](../../glossary.md#term-planning); [backend ownership glossary](../../glossary.md#backend-ownership); [backends glossary](../../glossary.md#backend); [prepare glossary](../../glossary.md#prepare); [Runtime glossary](../../glossary.md#term-runtime))</span>

Named [compiler](../../glossary.md#term-compiler) components such as `ElementwiseGradientRules` call only existing public methods
such as `mul`, `add`, `sumToShape`, and [`transpose`](../../glossary.md#term-transpose). During one [compile](../../glossary.md#compile) request, [identity](../../glossary.md#term-identity-meanings)-based
maps associate exact [Tensor](../../glossary.md#tensor) objects with ordered contributions and accumulated [gradients](../../glossary.md#term-gradient). The
compiler processes selected occurrences in reverse [producer](../../glossary.md#term-producer) postorder, ascending selected output
slot for one producer, and ascending input position, then combines contributions with ordinary
left-associated `Tensor.add`. These maps are temporary bookkeeping, not [graph](../../glossary.md#graph) [IR](../../glossary.md#term-intermediate-representation-ir), public Tensor
state, a tape, or a [registry](../../glossary.md#term-registry).

## [Preflight](../../glossary.md#term-preflight) and construction failures

Before constructing a backward expression, the [compiler](../../glossary.md#term-compiler) inventories every backward-reachable
[producer](../../glossary.md#term-producer) [occurrence](../../glossary.md#term-occurrence), output role, exact attributes, and required [derivative](../../glossary.md#term-derivative) policy. Unsupported
or ambiguous work fails closed. This prevents a known incomplete rule [matrix](../../glossary.md#term-matrix-meanings) from creating a
partial backward expression.

The closed implemented matrix through Compiler 0005B is:

| Family | Current exact variants |
|---|---|
| [Elementwise](../../glossary.md#term-elementwise) | All seven promoted floating [binary](../../glossary.md#term-binary-descriptor-and-metadata) arithmetic kinds; all eight exact-type floating [scalar](../../glossary.md#term-scalar-immediate) kinds, including first-class `CLAMP`; promoted branch-only `WHERE`; floating-to-floating `CAST`; and all nineteen floating [unary](../../glossary.md#term-unary) kinds |
| Reduction/[scan](../../glossary.md#term-cumulative-scan)/[softmax](../../glossary.md#softmax--log-softmax) | Floating ordinary full, single-[axis](../../glossary.md#term-axis), and ordered multi-axis `SUM`/[`MEAN`](../../glossary.md#term-mean)/`PROD`/`MIN`/`MAX`; masked floating `SUM`/`MEAN`; [binding](../../glossary.md#term-binding)-aware floating `SUM_TO_SHAPE`; floating `LOG_SUM_EXP`, [`VARIANCE`](../../glossary.md#term-variance), `STANDARD_DEVIATION`, `L1_NORM`, `L2_NORM`, `CUM_SUM`, `CUM_PROD`, `SOFTMAX`, and `LOG_SOFTMAX` |
| [Normalization](../../glossary.md#term-normalization) | No-affine and affine floating [Layer normalization](../../glossary.md#layer-normalization); no-scale and scaled floating [RMS normalization](../../glossary.md#rms-normalization); all five floating [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters)-[inference](../../glossary.md#term-inference) inputs; and batch-training public output slots zero through two with their exact input-role matrices |
| [Linear](../../glossary.md#term-linear-meanings) algebra | Every current floating [`MATMUL`](../../glossary.md#matrix-multiplication--matmul) [vector](../../glossary.md#term-vector)/matrix [rank](../../glossary.md#term-rank) pairing, with role-aware mixed-floating normalization and batch unbroadcasting |
| Logical [layout](../../glossary.md#layout)/selection | Floating [`CONTIGUOUS`](../../glossary.md#term-contiguous), `RESHAPE`, binding-aware `EXPAND`, `EXPAND_DIMS`, `SQUEEZE`, `PERMUTE`, normalized [`SLICE`](../../glossary.md#slice), both normalized `SLICE_UPDATE` data roles, `SELECT`, [`PAD`](../../glossary.md#padding), `TILE`, `CONCAT`, and `STACK` |

Forward and generated expressions use one [model](../../glossary.md#term-model)-owned [Tensor](../../glossary.md#tensor) algebra, inference/[validation](../../glossary.md#term-validation)
contract, numerical-semantics contract, and exact optimization pipeline. For a selected mixed-
floating input, [cotangent normalization](../../glossary.md#cotangent-normalization) first reverses ordinary [broadcasting](../../glossary.md#broadcasting) with public
`sumToShape` when needed and then calls ordinary `cast` exactly when the contribution type
differs. The contribution has the selected input's exact [Shape](../../glossary.md#shape) and floating [data type](../../glossary.md#data-type--datatype) before it
enters deterministic accumulation. `WHERE` applies this rule independently to its branches;
MATMUL applies it after the rank-specific formula and batch unbroadcasting. Same-type rows add no
cast.

Binary DIV constructs `g / right` for the left input and the exact ordered expression
`-(g * left) / (right * right)` for the right input. Scalar DIV constructs `g / scalar`. These
ordinary operations retain the shared arithmetic, exceptional-value, validation, rewrite, fold,
[DCE](../../glossary.md#term-dead-code-elimination-dce), and phase-local [CSE](../../glossary.md#term-common-subexpression-elimination-cse) contracts. There is no [gradient](../../glossary.md#term-gradient)-only singularity, [NaN](../../glossary.md#term-nan), [infinity](../../glossary.md#term-infinity),
[signed-zero](../../glossary.md#nan-infinity-and-signed-zero), [overflow](../../glossary.md#term-overflow), [underflow](../../glossary.md#term-underflow), [rounding](../../glossary.md#term-rounding), or optimization policy.

`FLOOR`, `CEIL`, and `SIGN` use a direct exact positive-zero first-order [cotangent](../../glossary.md#cotangent) at every
represented input, independent of `g`; the rule constructs neither `g * 0` nor a floating
comparison. Ordinary MEAN derives its denominator by reducing an input-shaped exact logical-one
expression over the selected full, single-axis, or ordered multi-axis domain, then divides the
restored cotangent by the expanded count. This formula represents static, dynamic, expression,
zero-sized, and empty-axis-list extents without a [host](../../glossary.md#term-host) count or exact-integer threshold.

Masked SUM restores the reduced axis, expands the cotangent, and routes it with the original [mask](../../glossary.md#term-boolean-mask)
while placing an exact typed zero elsewhere. Masked MEAN constructs its true count with ordinary
`WHERE` and `SUM`, divides the restored cotangent, and uses a final ordinary `WHERE`. The selected
all-false-slice convention returns zero at every input [coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix); it neither suppresses the
intermediate quotient nor defines branch [evaluation](../../glossary.md#term-evaluation). [BOOL](../../glossary.md#data-type--datatype) conditions and masks receive no
cotangent. `SUM_TO_SHAPE` is invertible only when every aligned input/[target](../../glossary.md#term-target) [Dimension](../../glossary.md#dimension) is either
exactly equal, the static target [extent](../../glossary.md#term-extent) is one, or the binding-dependent inverse uses the exact
forward target-one-or-target-equal-source predicate. Binding-dependent `EXPAND` similarly emits
`source == 1 || source == target` for each unresolved aligned pair. These obligations remain
deferred [compile](../../glossary.md#compile) constraints; preflight proves the inverse relationship without binding a [symbol](../../glossary.md#dimension).
CUM_SUM retains exclusivity and reverses scan direction; PERMUTE uses the inverse [permutation](../../glossary.md#permutation).

Product reduction is division-free: sequential keep-dimension reductions are reversed with
exclusive prefix and suffix products. Cumulative product replaces zeros with one for its safe
product, tracks zero-prefix counts, and accumulates in the opposite direction. Reduction [extrema](../../glossary.md#saturation-pole-and-extremum)
compare with the exact saved output and divide among all matching positions. A NaN output matches
nothing and therefore selects exact zero.

Softmax and [log-softmax](../../glossary.md#softmax--log-softmax) consume their exact forward output rather than recomputing it:

```text
softmax:     y * (g - sum(g * y, axis, true))
logSoftmax:  g - exp(y) * sum(g, axis, true)
```

[Log-sum-exp](../../glossary.md#log-sum-exp) uses its restored saved output. Variance and [standard deviation](../../glossary.md#term-standard-deviation) derive count and
correction through exact logical-one Tensor expressions. Standard deviation and [L2 norm](../../glossary.md#norms-elementary-functions-and-probability-distributions) select
zero when their saved result is not strictly positive; [L1 norm](../../glossary.md#norms-elementary-functions-and-probability-distributions) selects zero at [signed zero](../../glossary.md#term-signed-zero) and
NaN. Other exceptional values flow through ordinary Tensor operations in formula order.

Layer and RMS normalization reconstruct their population statistics over exact trailing axes.
Affine operands are visibly aligned, mixed-floating calculation uses the selected forward-output
type, and each completed contribution is cast back to its selected input type. Batch inference
supports all five floating Tensor inputs. Batch training selects input/scale/[bias](../../glossary.md#term-bias) from output slot
zero, input/running mean from slot one, and input/running variance from slot two. Its formulas
consume the [canonical](../../glossary.md#term-canonical-form) saved mean and inverse-standard-deviation wrappers at slots three and four;
those slots cannot be independent cotangent roots.

[ERF](../../glossary.md#term-erf) constructs `g * exp(-(x * x)) * (2 / sqrt(pi))`. Its coefficient is exact scalar-[operation](../../glossary.md#operation)
[metadata](../../glossary.md#term-metadata) with fixed [BFLOAT16](../../glossary.md#term-bfloat16)/[FLOAT16](../../glossary.md#term-float16)/[FLOAT32](../../glossary.md#term-float32)/[FLOAT64](../../glossary.md#term-float64) bits `0x3F90`, `0x3C83`, `0x3F906EBB`,
and `0x3FF20DD750429B6D`; the rule does not evaluate host floating arithmetic.

MIN and MAX split an exact represented-numeric tie equally between Tensor inputs; scalar extrema
give the Tensor receiver one half at a tie. First-class CLAMP applies that rule to the exact
ordered composition `MIN(MAX(x, min), max)`, producing one half at an ordinary endpoint and one
quarter when both stages tie. Opposite signed zeros compare equal. Unordered NaN comparisons
make piecewise extrema, CLAMP, ABS, and RELU return exact [positive zero](../../glossary.md#nan-infinity-and-signed-zero).

Binary POW constructs `g * right * pow(left, right - 1)` and `g * output * log(left)`. Scalar POW
derives [exponent](../../glossary.md#term-exponent)-minus-one with exactly one subtraction in the exponent's represented floating
type. RECIPROCAL, [LOG](../../glossary.md#term-log), LOG1P, SQRT, and RSQRT use their direct analytic formulas. ABS returns
zero at signed zero and NaN; RELU returns `g` only when its input is strictly greater than
positive zero.

Exact [GELU](../../glossary.md#gaussian-error-linear-unit--gelu), fixed [tanh](../../glossary.md#term-tanh)-approximation GELU, and [SiLU](../../glossary.md#sigmoid-linear-unit--silu) use fixed-coefficient analytic formulas for
[finite](../../glossary.md#term-finite) and NaN inputs. They return `g` at [positive infinity](../../glossary.md#nan-infinity-and-signed-zero) and exact positive zero at [negative
infinity](../../glossary.md#nan-infinity-and-signed-zero). Other analytic rows add no domain, finite, singularity, or continuous-extension mask;
ordinary Tensor operations determine their exceptional behavior in formula order. Compiler
0005A uses fixed BFLOAT16/FLOAT16/FLOAT32/FLOAT64 coefficient bits for one half, negative one half,
two, the inverse-square-root terms, and `0.044715`/`0.134145`; the FLOAT16 `0.044715` coefficient is
the raw half word `0x29B9`. [Compilation](../../glossary.md#compiler-engine-and-execution-stage-names) evaluates no host transcendental coefficient.

MATMUL handles vector-vector, vector-matrix, matrix-vector, and matrix-matrix rank promotion.
Each selected result reverses batch broadcasting and then casts once if the operand type differs
from the promoted type. Integral MATMUL is rejected.
SLICE scatters into an input-shaped typed zero. SLICE_UPDATE masks the base cotangent or extracts
the update cotangent; only the update role requires all selected base extents to be static.
SELECT scatters at one restored axis coordinate. PAD crops its before-width prefix. TILE reshapes
to interleaved repeat/input axes and sums the repeat axes. CONCAT crops by ordered symbolic input
prefixes. STACK selects the corresponding inserted-axis coordinate. Repeated input positions
remain repeated contributions and therefore accumulate deterministically.

Everything outside this table fails closed on a selected route. Compiler 0005C retains remaining
layout/indexing/ordering/stochastic families; Compiler 0005D retains attention, [convolution](../../glossary.md#term-channels-first-convolution-layers),
[pooling](../../glossary.md#term-pooling), and [losses](../../glossary.md#loss); Compiler 0005E owns the complete first-order closure audit. Comparisons, BOOL
logic/classification, the `WHERE` condition, scalar attributes and bounds, non-floating casts,
`ALL`, `ANY`, arg-extrema outputs, batch saved auxiliary roots, one-hot and other index roles,
masks, and [graph RNG state](../../glossary.md#graph-rng-state--graphrngstate) remain non-[differentiable](../../glossary.md#derivative-and-chain-rule).

Preflight is not full [graph](../../glossary.md#graph) inference. The compiler performs authoritative inference and
validation after the one combined [capture](../../glossary.md#term-capture-meanings). A later Tensor construction, capture, inference,
validation, or optimization failure can therefore consume temporary [`TensorId`](../../glossary.md#tensorid) values. This is
compatible with the existing [opaque](../../glossary.md#term-opaque), monotonic, non-reusable ID contract.

## [Constants](../../glossary.md#term-constant) and hidden outputs

The implicit unit [seed](../../glossary.md#term-seed), routing zeros, [MEAN](../../glossary.md#term-mean) logical ones, and [scalar](../../glossary.md#term-scalar) values needed as [Tensor](../../glossary.md#tensor)
comparison operands are storage-free Tensor [leaves](../../glossary.md#term-leaf) or expressions. One request-local [cache](../../glossary.md#term-cache) keys
each base by exact [`ScalarValue`](../../glossary.md#typed-scalar-value--scalarvalue) [data type](../../glossary.md#data-type--datatype) and [represented bits](../../glossary.md#term-represented-bits), registers it explicitly as one
[logical splat](../../glossary.md#logical-splat), and preserves deterministic first-use order. [BFLOAT16](../../glossary.md#term-bfloat16) zero/one use exact bits
`0x0000`/`0x3F80`; [FLOAT32](../../glossary.md#term-float32) and [FLOAT64](../../glossary.md#term-float64) use exact [positive zero](../../glossary.md#nan-infinity-and-signed-zero)/one. [Shape](../../glossary.md#shape)-specific values are
ordinary `expand` expressions. Arithmetic-only coefficients remain exact scalar-[operation](../../glossary.md#operation)
[metadata](../../glossary.md#term-metadata) rather than splat leaves. [Host storage](../../glossary.md#host-storage), labels, factory history, Shape, [layout](../../glossary.md#layout), and
[provenance](../../glossary.md#provenance) absence never imply constant status. Only generated bases reachable from returned
[gradient](../../glossary.md#term-gradient) expressions remain in combined-[capture](../../glossary.md#term-capture-meanings) ingress.

Some formulas need [producer](../../glossary.md#term-producer) outputs omitted from a public ergonomic result. [Dropout](../../glossary.md#dropout--inverted-dropout), for example,
returns the public result and next [RNG state](../../glossary.md#graph-rng-state--graphrngstate) while its same-[occurrence](../../glossary.md#term-occurrence) keep [mask](../../glossary.md#term-boolean-mask) is hidden.
[Batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters)-[normalization](../../glossary.md#term-normalization) training similarly hides saved batch statistics. [Model](../../glossary.md#term-model) task 0025 makes each
producer retain the [canonical](../../glossary.md#term-canonical-form) Tensor wrapper for every slot and exposes the smallest indexed
retrieval contract needed by [compiler](../../glossary.md#term-compiler). Compiler 0005B now consumes the exact batch saved mean and
inverse-standard-deviation wrappers from slots three and four without reconstructing a wrapper or
recomputing the statistics.

This creates an intentional reference cycle:

```text
Tensor -> TensorProvenance -> TensorProducer -> canonical outputs -> Tensor
```

The cycle is [immutable](../../glossary.md#term-immutable) expression metadata. Factory construction finishes all final fields before
publishing any output, and ordinary [garbage collection](../../glossary.md#term-garbage-collection-gc) can reclaim the whole unreachable
occurrence. There is no global [registry](../../glossary.md#term-registry), weak-reference protocol, [graph](../../glossary.md#graph) membership, or [runtime](../../glossary.md#term-runtime)
resource [ownership](../../glossary.md#term-ownership).

## One phase-aware [capture](../../glossary.md#term-capture-meanings)

Capture receives:

- ordered forward outputs;
- ordered [gradient](../../glossary.md#term-gradient) roots with [target](../../glossary.md#term-target)-specific roles;
- the [identity](../../glossary.md#term-identity-meanings) set of original forward [producers](../../glossary.md#term-producer); and
- explicit [constant](../../glossary.md#term-constant)-splat facts.

It traverses the combined expression once, assigns [`NodeId`](../../glossary.md#nodeid) and [`ValueId`](../../glossary.md#logical-valueid-versus-physical-memory-slot) once, and gives every
[producer](../../glossary.md#term-producer) [occurrence](../../glossary.md#term-occurrence) a per-[node](../../glossary.md#node) `FORWARD` or `BACKWARD` phase. A single positional
`backwardStartIndex` cannot replace this phase map.

Multiple targets may share the exact same accumulated-gradient [Tensor](../../glossary.md#tensor) and therefore one captured
gradient `ValueId`. Result roles still map those targets independently. The [graph](../../glossary.md#graph) output boundary
lists each distinct gradient value once; no manufactured identity node is needed.

## Combined optimization

The [immutable](../../glossary.md#term-immutable) combined [graph](../../glossary.md#graph), not a forward-only prefix, enters optimization. [Compiler](../../glossary.md#term-compiler) task 0004
adapts the completed task-0003, 0003A, and 0003B orchestration:

- [canonicalization](../../glossary.md#occurrence-identity-and-canonical-form) remains mandatory;
- the already selected exact rewrites and constant folds may apply in either phase only when
  their existing guards remain valid;
- [dead-code elimination](../../glossary.md#term-dead-code-elimination-dce) sees the whole graph;
- [common-subexpression elimination](../../glossary.md#term-common-subexpression-elimination-cse) remains phase-local initially; and
- every changed candidate returns through Compiler 0002 [validation](../../glossary.md#term-validation).

The sequence runs once: canonicalize/validate, exact rewrite, exact fold, whole-graph dead-code
elimination, phase-local common-subexpression elimination, then whole-graph cleanup dead-code
elimination. This migration authorizes no new rewrite, fixed point, relaxed arithmetic, floating
[evaluation](../../glossary.md#term-evaluation), or physical [constant](../../glossary.md#term-constant) [materialization](../../glossary.md#materialization).

## [Compile](../../glossary.md#compile) modes and future [derivatives](../../glossary.md#term-derivative)

`FORWARD_ONLY` skips [autograd](../../glossary.md#autograd). `FORWARD_AND_BACKWARD` and the initial `TRAINING_STEP` build the
combined expression before [capture](../../glossary.md#term-capture-meanings). `TRAINING_STEP` does not add [optimizer](../../glossary.md#term-optimizer) updates yet.

Generated [gradients](../../glossary.md#term-gradient) are ordinary [differentiable](../../glossary.md#derivative-and-chain-rule) [Tensor](../../glossary.md#tensor) expressions, preserving a route to higher
derivatives. Higher derivatives are not part of the current first-order implementation. A later
task must define an
explicit create-[graph](../../glossary.md#graph) or derivative-order [lifecycle](../../glossary.md#lifecycle) contract, provide rules for every [operation](../../glossary.md#operation)
used in gradient formulas, and represent derivative order in addition to [graph phase](../../glossary.md#graph-phase).

The design adds no `Tensor.gradient`, `Tensor.backward`, [mutable](../../glossary.md#term-mutable) gradient field, ThreadLocal
[compilation](../../glossary.md#compiler-engine-and-execution-stage-names) [scope](../../glossary.md#term-scope), [model](../../glossary.md#term-model)-owned derivative rule, public [compiler](../../glossary.md#term-compiler) [registry](../../glossary.md#term-registry), physical saved [buffer](../../glossary.md#term-buffer),
or [backend](../../glossary.md#backend)-owned global autograd.

See [Training graph](../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../glossary.md#training-graph))</span>,
[ADR 0009](../decisions/0009-compiler-owned-pre-capture-tensor-expression-autograd.md)<span data-glossary-navigation="true"> ([ADR glossary](../../glossary.md#term-architecture-decision-record-adr))</span>, the
[model master plan](../../planning/modules/model/master-plan.md), and the
[compiler master plan](../../planning/modules/compiler/master-plan.md).
