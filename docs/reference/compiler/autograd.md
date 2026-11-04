# Pre-[capture](../../glossary.md#term-capture-meanings) [autograd](../../glossary.md#autograd) and functional [gradients](../../glossary.md#term-gradient)

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/CompileArtifacts.java)

[Automatic differentiation](../../glossary.md#autograd) (**autograd**, or [AD](../../glossary.md#autograd)) builds expressions for [derivatives](../../glossary.md#term-derivative) by following
how an output depends on selected inputs. A **[cotangent seed](../../glossary.md#primal-adjoint-and-differentiation-seed)** specifies the output [sensitivity](../../glossary.md#term-sensitivity)
being propagated backward; for a [scalar objective](../../glossary.md#loss), a [seed](../../glossary.md#term-seed) of one asks for its ordinary gradient.
Here those expressions are built before **capture**, the conversion to a [graph](../../glossary.md#graph) with local [node](../../glossary.md#node)
and value identifiers. Constructing derivatives does not execute or update [parameters](../../glossary.md#term-parameter).

## What a seed and a gradient mean

The application selects outputs and differentiation targets through Engine; Engine calls
Compiler, which builds ordinary Tensor derivative expressions. Compiler is not an optimizer and
does not write updated parameter bytes. For a differentiable real function
$f:\mathbb{R}^n\to\mathbb{R}^m$, let $x_j$ be input coordinate $j$ for $1\le j\le n$ and
$y_i=f_i(x)$ output coordinate $i$ for $1\le i\le m$. A seed $s_i$ says how much sensitivity to
propagate from output $i$. The vector-Jacobian product at input coordinate $j$ is

$$
g_j=\sum_{i=1}^{m}s_i\frac{\partial f_i}{\partial x_j}.
$$

For the scalar case $f(x)=x^2$ with seed $s=1$, this gives $g=2x$: at $x=3$, the gradient is $6$.
A second admitted stage differentiates that expression and yields $2$. A vector needs an explicit
matching seed because different $s$ ask different questions. These real derivatives explain the
sensitivity; the implementation constructs the exact ordered Tensor formulas described below,
including finite-path restrictions, local non-differentiable conventions and represented
NaN/infinity/signed-zero behavior. It does not replace rounded computation by real algebra.
See [Engine compile entry points](entry-points.md) for which seed policies ordinary calls select
and [Training](../training/sessions.md#who-calls-whom-during-one-step) for later parameter mutation.

<a id="current-package-private-pre-capture-autograd"></a>
### Current package-private pre-[capture](../../glossary.md#term-capture-meanings) [autograd](../../glossary.md#autograd)

[Model](../../glossary.md#term-model) task 0025 is Complete, so a [producer](../../glossary.md#term-producer) now returns the [canonical](../../glossary.md#term-canonical-form) exact [Tensor](../../glossary.md#tensor) wrapper for
every output slot, including hidden [dropout](../../glossary.md#dropout--inverted-dropout) [masks](../../glossary.md#term-boolean-mask) and [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters)-[normalization](../../glossary.md#term-normalization) saved statistics,
without changing public Tensor methods or ergonomic result carriers. [Compiler](../../glossary.md#term-compiler) task 0004 is
Complete and supplies the first internal autograd consumer of that [identity](../../glossary.md#term-identity-meanings) contract. Compiler
tasks 0004A and 0004B are also Complete with the exact-composition and shared-algebra local-rule
extensions described below. Compiler task 0005A is Complete and closes the exact 48-kind current
[elementwise](../../glossary.md#term-elementwise)/[activation](../../glossary.md#term-activation) inventory with explicit [derivative](../../glossary.md#term-derivative) and non-[differentiable](../../glossary.md#derivative-and-chain-rule)-role policies.
Compiler task 0005B is Complete and adds [binding](../../glossary.md#term-binding)-aware expansion plus the current reduction,
[scan](../../glossary.md#term-cumulative-scan), [softmax](../../glossary.md#softmax--log-softmax), statistics, [norm](../../glossary.md#term-norm), and Layer/RMS/batch-normalization first-order [matrix](../../glossary.md#term-matrix-meanings).
Compiler task 0005C is Complete and adds dynamic [slice](../../glossary.md#slice)/window constraints plus the current
[layout](../../glossary.md#layout), [Gather](../../glossary.md#gather)/scatter, ordering/top-K, and explicit-state dropout first-order matrix.
Compiler task 0005D is Complete and adds the current two-output attention, grouped [convolution](../../glossary.md#term-channels-first-convolution-layers),
[pooling](../../glossary.md#term-pooling), and [loss](../../glossary.md#loss) first-order matrix without adding a public [gradient](../../glossary.md#term-gradient) [API](../../glossary.md#term-application-programming-interface-api) or backward-only
[operation](../../glossary.md#operation) vocabulary.
Model task 0025G adds [NCW](../../glossary.md#term-ncw) Conv1d only as the ordinary visible
`EXPAND_DIMS(2) -> EXPAND_DIMS(2) -> CONV2D -> SQUEEZE(2)` [Tensor composition](../../glossary.md#tensor-composition). Generic capture
therefore sees two [rank](../../glossary.md#term-rank)-edit [nodes](../../glossary.md#node), one existing grouped [Conv2d](../../glossary.md#convolution--conv2d) node, and one rank-edit node; it
does not see `CONV1D`, `ConvNd`, or a Conv1d attributes value. Existing rank-edit and Conv2d
[inference](../../glossary.md#term-inference-meanings) and first-order gradient rules apply to those ordinary nodes without a new Compiler
inventory entry or formula. This structural reuse is a current compiler fact, not a [fusion](../../glossary.md#term-fusion),
[lowering](../../glossary.md#lowering), [backend](../../glossary.md#backend)-capability, execution, or performance claim.
Model [`CONV3D`](../../glossary.md#conv3d--ncdhw) records one first-class grouped [NCDHW](../../glossary.md#conv3d--ncdhw) [occurrence](../../glossary.md#term-occurrence) with two or three ordered inputs,
one output, and `Conv3dAttrs`. Compiler forward verification and first-order differentiation
adopt this [signature](../../glossary.md#term-signature). Independent inference derives the promoted type, rank-five result and
depth/height/width geometry, then final [validation](../../glossary.md#term-validation) compares that result with the captured
[descriptor](../../glossary.md#term-descriptor). It proves or retains ordered obligations for input-[channel](../../glossary.md#term-channel) divisibility,
output-channel divisibility, [weight](../../glossary.md#term-weight) channels per group, optional [bias](../../glossary.md#term-bias) length, and non-negative
depth, height, then width numerators. Conv3d uses the shared Conv2d/Conv3d gradient [owner](../../glossary.md#term-owner),
not a global backward rejection or a new backend-owned derivative.
Model task 0025J adds first-class `MAX_POOL3D` and `AVERAGE_POOL3D` Model [metadata](../../glossary.md#term-metadata) with exact
one-input/one-output signatures and rank-five NCDHW descriptors. Complete Compiler tasks 0006B1
and 0006B2 adopt both signatures, together with the three Model-current
`UNFOLD3D`/`FOLD3D` signatures, in ordinary forward verification and first-order differentiation.
[CPU](../../glossary.md#term-cpu) 0008G1 originally implemented the fully static resolved-layout non-gradient [Pool3d](../../glossary.md#pool3d--ncdhw-pooling) subset.
Current matching CPU [FLOAT32](../../glossary.md#term-float32)/[FLOAT64](../../glossary.md#term-float64) gradient metadata and qualified lowering reuse ordinary
PoolingGradientRules and UNFOLD3D/FOLD3D for Pool3d training; qualified [Metal](../../glossary.md#term-metal) FLOAT32 supports
qualifying overlap through the same derivative expressions. Backend [admission](../../glossary.md#term-admission) still requires
every generated occurrence. CPU low-[precision](../../glossary.md#term-precision) execution and actual first-order [AD](../../glossary.md#autograd) use the same
ordinary owners under their current family/geometry guards, not an obsolete [F32](../../glossary.md#floating-point-precision-and-rounding)/[F64](../../glossary.md#floating-point-precision-and-rounding) ceiling.
Earlier task receipts counted the inventory at individual [checkpoints](../../glossary.md#term-checkpoint). Those counts and their
three-signature fixed-scan exclusion are not the current support contract. Fixed [RNN](../../glossary.md#term-recurrent-neural-network-rnn), [GRU](../../glossary.md#term-gated-recurrent-unit-gru) and
[LSTM](../../glossary.md#term-long-short-term-memory-lstm) first-order [BPTT](../../glossary.md#term-backpropagation-through-time-bptt) is implemented by
[RecurrentGradientRules](../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/RecurrentGradientRules.java),
which constructs a closed transition and differentiates it through the shared Scan owner.
Actual selected roles, source validity, canonical output availability and generated-[topology](../../glossary.md#term-topology)
backend [qualification](../../glossary.md#term-qualification) still determine admission.

Current Compiler forward inference independently derives the `UNFOLD3D` and `FOLD3D` descriptors
as canonical columns
`[N, C * kD * kH * kW, DOut * HOut * WOut]`, literal floor/ceil depth-height-width grids, exact
direct-positive-zero or typed-[padding](../../glossary.md#padding) unfold attributes, and an exact NCDHW fold [target](../../glossary.md#term-target) [Shape](../../glossary.md#shape).
It proves, rejects, or retains non-negative [spatial](../../glossary.md#term-spatial-axes)-domain obligations and final-validates the
captured descriptor without binding [dimensions](../../glossary.md#dimension) or materializing windows. In backward-capable
modes, either `UNFOLD3D` form folds its [cotangent](../../glossary.md#cotangent) into the exact captured input Shape, while
`FOLD3D` unfolds its cotangent with the direct positive-zero-padding overload. Typed unfold
padding is configuration and has no cotangent. These [linear](../../glossary.md#term-linear-meanings) adjoints alternate under the supported
[second derivative](../../glossary.md#term-second-derivative) stage.

Package-private `GraphCompiler.compile` takes [`CompileMode`](../../glossary.md#compile-mode--compilemode), ordered forward outputs, an optional
public `FunctionalGradientRequest`, explicit forward [constant](../../glossary.md#term-constant) ingress, and
[`GraphOptimizationConfig`](../../glossary.md#graph-optimization-configuration--graphoptimizationconfig) directly; there is no request aggregate. It returns package-private
mode-neutral `GraphCompilation`. `FORWARD_ONLY` requires no functional request, produces no
`BACKWARD` nodes, and has empty gradient results. `FORWARD_AND_BACKWARD` and the current internal
`TRAINING_STEP` path require the same request and may carry one combined [graph](../../glossary.md#graph).
`GraphCompilation` retains the final `ValidatedGraph`, ordered forward output values, and ordered
public `GradientPublicationBinding` values. Its package-private component names remain
`forwardOutputs` and `gradientResults`; the public [publication plan](../../glossary.md#publication-plan) instead exposes
`forwardBindings()` and `gradientBindings()`. It is graph-stage state, not the current public
cross-package `CompileArtifacts`.

The current functional request contains exactly one or two [reverse-mode](../../glossary.md#primal-adjoint-and-differentiation-seed) stages. Stage-one outputs
identify exact requested forward Tensors; stage-two outputs identify generated first-stage
gradients by target-list index. A one-stage request has `createGraph=false`; a two-stage request
has `true` for stage one and `false` for stage two. Each stage has a non-empty,
exact-identity-unique target list drawn from the complete original forward inventory and an
output-aligned list of optional cotangent [seeds](../../glossary.md#commit-release-distribution-log-and-seed-meanings). An absent [seed](../../glossary.md#term-seed) is valid only for a [scalar](../../glossary.md#term-scalar),
floating, gradient-eligible output and creates an exact typed positive one. A present seed must
have the output's exact Shape and floating [data type](../../glossary.md#data-type--datatype) and must not request gradients. Multiple
selected outputs contribute a deterministic sum of [vector](../../glossary.md#term-vector)-[Jacobian](../../glossary.md#term-jacobian) products.

`DisconnectedPolicy.ERROR` rejects a target without a differentiable route.
`DisconnectedPolicy.ZERO` returns an ordinary exact typed zero expression; several result roles
may share that value. Results are ordered by derivative order and then stage-local target index.
`DerivativeGraphMetadata` classifies final nodes as derivative order zero, one, or two while
unchanged Model `GraphPhase` continues to distinguish only forward and backward nodes.

The sidecar also retains `scanActivityByNode()`, an [immutable](../../glossary.md#term-immutable) map keyed by containing Scan node
IDs. Each `ScanActivityPlan` names the exact independently scoped body graph, not a structurally
equal graph or a body-local ID in the containing inventory. The three-[argument](../../glossary.md#term-argument)
`DerivativeGraphMetadata(graph, derivativeOrderByNode, scanActivityByNode)` constructor validates
these associations. The two-argument form intentionally creates no activity annotations; graph
rewrites and copies carrying annotated scans must preserve or explicitly remap them. This stays
inside the existing eighth `CompileArtifacts` component, not an additional aggregate component.

<a id="functional-gradient-examples"></a>
### Functional-[gradient](../../glossary.md#term-gradient) examples

These are [compiler](../../glossary.md#term-compiler)-package integration sketches because `GraphCompiler` remains package-private;
they describe current semantics without claiming a public [compile](../../glossary.md#compile) [facade](../../glossary.md#term-facade).

For [scalar](../../glossary.md#term-scalar) `loss = x * x`, request stage one from [`loss`](../../glossary.md#loss) to `x` with its default [seed](../../glossary.md#term-seed) and
`createGraph=true`, then request stage two from
`FirstStageGradientReference(0)` to `x` with its default seed and `createGraph=false`. One combined
compile returns an order-one role for $2x$ and an order-two role for $2$; both are ordinary
values in the same [graph](../../glossary.md#graph), and no [Tensor](../../glossary.md#tensor) gradient field is written.

For a [vector](../../glossary.md#term-vector) output, supply an exact-[Shape](../../glossary.md#shape) explicit seed to select its [vector-Jacobian product](../../glossary.md#term-vector-jacobian-product-vjp).
A multi-output stage supplies one aligned seed per output and sums those products in output order.
The second stage may then seed a selected first-stage gradient with an exact-Shape vector to form
a [Hessian-vector product](../../glossary.md#term-hessian-vector-product-hvp). Omitting a seed for any non-scalar output is rejected before [derivative](../../glossary.md#term-derivative)
formulas are allocated.

Before constructing the seed or any formula Tensor, `AutogradPreflight` iteratively inventories
the complete original forward request. The package-private first-order coverage checker assigns
each exact selected output/input role one current disposition: conditionally [differentiable](../../glossary.md#derivative-and-chain-rule)
(`D`), intentionally non-differentiable ([`ND`](../../glossary.md#term-nd)), or [fail-closed](../../glossary.md#term-fail-closed) (`FC`). Every `ND` or `FC`
decision has a deterministic reason. Every `D` decision names exactly one existing formula-family
[owner](../../glossary.md#term-owner), but remains conditional until [preflight](../../glossary.md#term-preflight) validates the [occurrence](../../glossary.md#term-occurrence)'s exact attributes,
[cardinality](../../glossary.md#term-cardinality), data-type and Shape relationships, [canonical](../../glossary.md#term-canonical-form) auxiliaries, [normalization](../../glossary.md#term-normalization) path, and
derivative policy. Unknown kinds, unclassified kind/attributes pairings, malformed
cardinalities, illegal slots, and unsupported prerequisites [fail closed](../../glossary.md#admission-preflight-and-fail-closed-validation) before derivative Tensor
[identity](../../glossary.md#term-identity-meanings) allocation.

Preflight follows required [cotangent](../../glossary.md#cotangent) paths from the selected stage outputs, stopping at `ND`
roles such as comparisons, [masks](../../glossary.md#term-boolean-mask), and indices. A required `FC` path to a requested [target](../../glossary.md#term-target) rejects
the stage even when a differentiable sibling path also reaches that target. Neither disconnected
policy permits omitting such a contribution. An unselected output or a saved value used only
behind an intentional non-differentiable boundary does not create a required derivative path.

The same family-owner decision is retained in the successful preflight plan and drives
`FirstOrderAutograd` [dispatch](../../glossary.md#term-dispatch), so role selection and formula routing do not maintain independent
kind branches. Named `ElementwiseGradientRules`, `ReductionGradientRules`,
`NormalizationGradientRules`, `LinearAlgebraGradientRules`, `LayoutGradientRules`,
`IndexingGradientRules`, `OrderingGradientRules`, and `StochasticGradientRules` then build
formulas only with existing public Tensor operations. `AttentionGradientRules`,
`ConvolutionGradientRules`, `PoolingGradientRules`, and `LossGradientRules` own the structured
Compiler 0005D formulas. Exact Tensor [identity](../../glossary.md#term-reference-identity) keys ordered contributions during one compile
request. Reverse accumulation visits [producer](../../glossary.md#term-producer) postorder in reverse, selected output slots in
ascending order for that producer, and input positions in ascending order; ordinary
left-associated `Tensor.add` accumulates contributions. This ephemeral state is neither Tensor
state nor graph [intermediate representation](../../glossary.md#term-intermediate-representation-ir) ([IR](../../glossary.md#intermediate-representation-dag-and-ssa)).

The current [matrix](../../glossary.md#term-matrix-meanings) is the union of the Compiler 0004–0004B rows and the Compiler 0005A–0005D
family completions:

| Family | Current exact supported variants |
|---|---|
| [Elementwise](../../glossary.md#term-elementwise) and [activation](../../glossary.md#term-activation) | All seven promoted floating [binary](../../glossary.md#term-binary-descriptor-and-metadata) arithmetic kinds: `ADD`, `SUB`, `MUL`, `DIV`, `MIN`, `MAX`, and `POW`; all eight exact-type floating scalar kinds, including first-class `CLAMP`; promoted floating branch-only `WHERE`; floating-to-floating `CAST`; and all nineteen floating [unary](../../glossary.md#term-unary) kinds from `ABS` through `SILU` |
| Reductions, scans, and [softmax](../../glossary.md#softmax--log-softmax) | Ordinary full, single-[axis](../../glossary.md#term-axis), and ordered multi-axis floating `SUM`, [`MEAN`](../../glossary.md#term-mean), `PROD`, `MIN`, and `MAX`; masked floating `SUM` and `MEAN`; [binding](../../glossary.md#term-binding)-aware floating `SUM_TO_SHAPE`; floating `LOG_SUM_EXP`, [`VARIANCE`](../../glossary.md#term-variance), `STANDARD_DEVIATION`, `L1_NORM`, and `L2_NORM`; floating `CUM_SUM` and `CUM_PROD` for every exclusive/reverse combination; floating `SOFTMAX` and `LOG_SOFTMAX` |
| Normalization | No-affine and affine floating `LAYER_NORM`; no-scale and scaled floating `RMS_NORM`; all five floating inputs of [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters)-normalization [inference](../../glossary.md#term-inference); and the exact output-slot-specific batch-normalization-training roles for public slots zero through two |
| [Linear](../../glossary.md#term-linear-meanings) algebra | Every current floating [`MATMUL`](../../glossary.md#matrix-multiplication--matmul) vector/matrix [rank](../../glossary.md#term-rank) pairing, with role-aware mixed-floating normalization and batch unbroadcasting |
| Logical [layout](../../glossary.md#layout) and selection | Floating [`CONTIGUOUS`](../../glossary.md#term-contiguous), `RESHAPE`, binding-aware `EXPAND`, `EXPAND_DIMS`, `SQUEEZE`, `PERMUTE`, both [`SLICE`](../../glossary.md#slice)/`SLICE_UPDATE` attribute forms, `SELECT`, [`PAD`](../../glossary.md#padding), `TILE`, `CONCAT`, and `STACK` |
| Window transforms | Floating `UNFOLD_AXIS`, `FOLD_AXIS`, both `UNFOLD2D` attribute forms, and `FOLD2D` through their exact public inverse or overlap-add [operation](../../glossary.md#operation) |
| Indexing and scatter | Floating data for [`GATHER`](../../glossary.md#gather), `GATHER_ELEMENTS`, and `GATHER_ND`; floating base and update roles for `SCATTER_ADD`, `SCATTER_ELEMENTS`, and `SCATTER_ND` with `NONE`, `ADD`, `MUL`, `MIN`, or `MAX` as applicable |
| Ordering | One-output floating `SORT` through one exact matching stable `ARGSORT`; floating `TOP_K` values slot zero through the canonical indices at slot one |
| Stochastic | Floating [`DROPOUT`](../../glossary.md#dropout--inverted-dropout) values slot zero through the exact same-occurrence [BOOL](../../glossary.md#data-type--datatype) keep mask; mask and graph-state roles remain non-differentiable |
| Attention | Both public outputs of exact two-output floating `SCALED_DOT_PRODUCT_ATTENTION`: query, key, and value from values slot zero; query and key from canonical [weights](../../glossary.md#term-weight) slot one; an optional BOOL mask remains non-differentiable |
| [Convolution](../../glossary.md#term-channels-first-convolution-layers) and [pooling](../../glossary.md#term-pooling) | Every floating input, weight, and optional [bias](../../glossary.md#term-bias) role of grouped [NCHW](../../glossary.md#term-nchw) [`CONV2D`](../../glossary.md#convolution--conv2d); the sole floating input of fixed-count `AVERAGE_POOL2D` and exact first-logical-winner `MAX_POOL2D` |
| Losses | [Prediction](../../glossary.md#term-prediction) and target of floating `MEAN_SQUARED_ERROR`; logits and dense floating targets of dense categorical cross-entropy with logits; logits only for index-target categorical cross-entropy with a positive static class depth |

Forward and generated expressions use one [model](../../glossary.md#term-model)-owned Tensor algebra. For a selected mixed-
floating input, the compiler first reverses ordinary [broadcasting](../../glossary.md#broadcasting) with `sumToShape` when needed
and then uses ordinary `cast` exactly when the contribution type differs from the input type.
The resulting contribution has that input's exact Shape and floating [data type](../../glossary.md#data-type--datatype) before
deterministic accumulation. `WHERE` applies this rule independently to its two branch roles, and
MATMUL applies it after its rank-specific formula and batch-unbroadcast step. The BOOL condition,
mask, and other non-floating control roles receive no cotangent.

For a forward `CAST`, the existing rule is similarly limited to floating source and target types.
A same-type occurrence reuses the incoming cotangent; a cross-floating occurrence creates one
ordinary `CAST` back to the source type. That generated expression uses the same Model-owned
forward conversion contract documented in the [Tensor API](../tensor/selection-casts.md#cast-expressions),
including direct target-format [rounding](../../glossary.md#term-rounding) and deterministic [NaN](../../glossary.md#term-nan) behavior. Compiler adds no
gradient-specific rounding, [saturation](../../glossary.md#term-saturation), straight-through estimator, or value evaluator. Any cast
with an integral or BOOL source or target remains non-differentiable, and this documentation change
adds no Compiler production behavior.

#### Elementwise local contributions

In the formulas in this subsection, $g$ is the incoming cotangent at one output coordinate,
$x$ is a unary input, $y$ its saved forward output, $\ell,r$ are binary left/right inputs, and
$a$ is a fixed scalar attribute. Products and quotients follow the displayed construction order.
For classical real derivatives, division needs a nonzero denominator; powers and logarithms
use a positive base for the stated real formulas. The represented program still constructs the
listed formulas at exceptional inputs, with the precise NaN/infinity/signed-zero behavior below;
these mathematical domains do not introduce a new runtime guard or repaired value.

Binary DIV constructs the left contribution as $g/r$ and the right contribution in the
exact order $-(g\ell)/(rr)$. Scalar DIV constructs $g/a$. These are
ordinary Tensor expressions: they inherit the same division, multiplication, negation, NaN,
[infinity](../../glossary.md#term-infinity), [signed-zero](../../glossary.md#nan-infinity-and-signed-zero), [overflow](../../glossary.md#term-overflow), [underflow](../../glossary.md#term-underflow), rounding, [validation](../../glossary.md#term-validation), and exact-optimization
contracts as forward expressions. No gradient-specific singularity or exceptional-value rule is
added.

Binary `MIN` and `MAX` route `g` to the strictly selected operand and split an exact numeric tie
equally. Each selected contribution then uses ordinary `sumToShape`-then-`cast` normalization.
Binary `POW(left, right)` constructs $g\,r\,\ell^{r-1}$ for `left` and
$g\,y\,\log(\ell)$ for `right`; it deliberately does not replace the first formula with
$y/\ell$.

Scalar `MIN` and `MAX` give the Tensor receiver `g` on its strict side, $g\cdot0.5$ at an exact
numeric tie, and exact [positive zero](../../glossary.md#nan-infinity-and-signed-zero) otherwise. Scalar `POW` constructs
$g\,a\,x^{a-1}$. The derived scalar uses exactly one subtraction in
the represented type: [BFLOAT16](../../glossary.md#term-bfloat16) expands to [binary32](../../glossary.md#floating-point-precision-and-rounding), subtracts binary32 one, and rounds back;
[FLOAT32](../../glossary.md#term-float32) and [FLOAT64](../../glossary.md#term-float64) subtract one in their own formats. First-class `CLAMP` differentiates the
exact ordered forward composition `MIN(MAX(input, minValue), maxValue)`. With distinct [finite](../../glossary.md#term-finite)
bounds the contribution is zero outside, `g` strictly inside, and $g/2$ at either endpoint;
when both [extrema](../../glossary.md#saturation-pole-and-extremum) stages tie it is $g/4$.

For the next table, $\operatorname{where}(c,u,v)$ selects $u$ where condition $c$ is true
and $v$ otherwise; it is not a promise of lazy branch evaluation. For real calculus, LOG needs
$x>0$, LOG1P needs $x>-1$, SQRT and RSQRT formulas need $x>0$, and RECIPROCAL needs $x\ne0$.
ABS and RELU use the listed local convention at their nondifferentiable zero. For example,
RECIPROCAL at $x=2$, $g=1$ gives $-1/(2\cdot2)=-0.25$; the sign means increasing the input
decreases the output. Here $\Phi$ is the standard normal cumulative distribution and
$\sigma(x)=1/(1+e^{-x})$ is sigmoid; these identify the GELU/SiLU functions below.

The added unary formulas are:

| Kind | Input contribution |
|---|---|
| `ABS` | $\operatorname{where}(x>0,g,\operatorname{where}(x<0,-g,0))$ |
| `RECIPROCAL` | $-g/(xx)$ |
| [`LOG`](../../glossary.md#term-log) | $g/x$ |
| `LOG1P` | $g/(x+1)$ |
| `SQRT` | $g/(2y)$ |
| `RSQRT` | $-0.5\,g\,y\,y\,y$ |
| `RELU` | $\operatorname{where}(x>0,g,0)$ |

Exact [GELU](../../glossary.md#gaussian-error-linear-unit--gelu) uses the derivative of $x\Phi(x)$. Fixed [tanh](../../glossary.md#term-tanh)-approximation GELU differentiates its
specified `0.044715` cubic approximation, and [SiLU](../../glossary.md#sigmoid-linear-unit--silu) differentiates $x\sigma(x)$. Each of these
three activations selects `g` at [positive infinity](../../glossary.md#nan-infinity-and-signed-zero) and exact positive zero at [negative infinity](../../glossary.md#nan-infinity-and-signed-zero)
to avoid a spurious infinity-times-zero NaN; finite and NaN inputs use the ordinary analytic
formula. Every other raw formula retains ordinary shared Tensor exceptional behavior without a
compiler-inserted finite, domain, singularity, or continuous-extension mask.

Piecewise extrema, `CLAMP`, `ABS`, and `RELU` use represented-value comparisons. Equal finite
values, equal same-sign [infinities](../../glossary.md#nan-infinity-and-signed-zero), and opposite signed zeros are ties. An unordered NaN makes
the ordered and equality tests false, so these piecewise rules return exact positive zero.
Comparisons, [Boolean](../../glossary.md#data-type--datatype) logic, floating classifications, the `WHERE` condition, scalar attributes
and clamp bounds, and non-floating cast roles remain non-differentiable.

`FLOOR`, `CEIL`, and `SIGN` use a direct exact positive-zero cotangent as their selected
first-order local convention. They do not construct `g * 0` or a floating comparison. This
convention does not change the forward operations' model semantics.

#### Reductions, counts and normalization

Reduction domains are the selected logical coordinates, not the Tensor's physical storage span.
A nonempty mean divides by its selected count; empty/all-false cases retain the explicit
represented conventions below rather than an assumed real-number division by zero.

Ordinary MEAN restores and expands `g`, creates an input-shaped exact logical-one expression,
reduces those ones across the same full, single-axis, or ordered multi-axis domain, expands the
count, and divides. The same expression works for static, dynamic, expression, zero-sized, and
empty-axis-list domains without a [host](../../glossary.md#term-host)-computed count or type-specific threshold. Masked MEAN
constructs its per-slice true count with ordinary `WHERE` and `SUM`, divides the restored
cotangent by that count, then applies a final ordinary `WHERE`. The selected local convention
returns exact zero at every input [coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix) of an all-false slice; it does not prescribe branch
[evaluation](../../glossary.md#term-evaluation) or suppress the intermediate quotient.

For input `x` and output cotangent `g`, [ERF](../../glossary.md#term-erf) constructs
$g\exp(-(xx))\,(2/\sqrt{\pi})$. The coefficient is exact scalar-operation [metadata](../../glossary.md#term-metadata) with
fixed BFLOAT16 bits `0x3F90`, [FLOAT16](../../glossary.md#term-float16) bits `0x3C83`, FLOAT32 bits `0x3F906EBB`, or
FLOAT64 bits `0x3FF20DD750429B6D`; [compilation](../../glossary.md#compiler-engine-and-execution-stage-names) does not evaluate a host transcendental function or add a
coefficient Tensor [leaf](../../glossary.md#term-leaf).

Masked SUM restores the removed axis, expands `g` to the data Shape, and selects that value where
the exact BOOL mask is true or an explicit typed zero where it is false. The mask receives no
cotangent. A forward `sumToShape` is inverted only when every right-aligned target [Dimension](../../glossary.md#dimension) is
the exact input Dimension, is static one, or retains the exact target-one-or-target-equal-source
predicate needed by the inverse `expand`. Leading input axes are permitted. Conversely, a
binding-dependent forward `EXPAND` emits, in aligned-axis order,
`source == 1 || source == target`; its inverse `SUM_TO_SHAPE` is admitted only through that same
predicate. Deferred predicates remain attached to the occurrence for later binding validation.
The compiler does not bind a Dimension, choose a [stride](../../glossary.md#term-stride), or treat a deferred result as execution
permission.

`PROD` uses sequential keep-dimension product stages and reverses each stage with exclusive
prefix and suffix `CUM_PROD`; it never divides by the selected input. `CUM_PROD` replaces
represented zeros with one for safe products, counts zeros with `CUM_SUM`, and accumulates in the
opposite [scan](../../glossary.md#term-cumulative-scan) direction. These formulas handle zero-free, one-zero, multiple-zero, exclusive,
reverse, and zero-length structure without a gradient-specific infinity or NaN repair.

Reduction `MIN` and `MAX` compare each input with the exact restored forward output and divide the
restored cotangent equally among all matching coordinates. Opposite signed zeros are ties. A NaN
forward output matches no coordinate, so the selected contribution is exact positive zero.
`LOG_SUM_EXP` uses $\operatorname{restored}(g)\exp(x-\operatorname{restored}(y))$.

`VARIANCE` and `STANDARD_DEVIATION` derive selected-domain count by reducing exact logical ones
and construct correction as a typed logical-one reduction rather than a host floating
conversion. [Standard deviation](../../glossary.md#term-standard-deviation) returns exact zero when the saved result is not strictly
positive. `L1_NORM` selects sign with exact zero at [signed zero](../../glossary.md#term-signed-zero) or NaN; `L2_NORM` returns exact
zero when its saved [norm](../../glossary.md#term-norm) is not strictly positive. Other exceptional values follow the ordinary
Tensor formulas in their recorded order.

For the softmax equations, $y$ is the exact saved forward output, $g$ its same-shape
cotangent, and $k$ the selected axis. $\operatorname{sum}_k$ sums all coordinates along that
axis while keeping a singleton dimension, so multiplication broadcasts back to the original
shape. For a two-coordinate finite real slice $y=(0.25,0.75)$ and $g=(1,0)$, the inner sum is
$0.25$ and the SOFTMAX input contribution is $(0.1875,-0.1875)$. This is an interpretation of
the rule, not permission to recompute saved probabilities or reorder rounded arithmetic.
$\operatorname{restored}$ below means reinsert reduced singleton axes for broadcasting.

`SOFTMAX` consumes exact forward output `y` and constructs
$y\bigl(g-\operatorname{sum}_{k}(gy)\bigr)$. `LOG_SOFTMAX` constructs
$g-\exp(y)\operatorname{sum}_{k}(g)$. Neither rule recomputes a softmax occurrence or selects a
forward numerical algorithm.

Layer and root-mean-square (RMS) normalization rebuild their population statistics with ordinary
Tensor operations over the exact trailing normalized axes. Mixed-floating occurrences compute in
the selected forward-output type, except FLOAT16 and FP8 use FLOAT32 working expressions, align affine operands by visible reshape/expand expressions,
and cast each completed Shape-correct contribution back to its selected input type. [Layer
normalization](../../glossary.md#layer-normalization) supports input, scale, and bias roles; [RMS normalization](../../glossary.md#rms-normalization) supports input and
optional scale.

Batch-normalization inference supports input, scale, bias, running mean, and running variance.
Batch-normalization training is output-slot-aware: normalized output slot zero contributes to
input/scale/bias, next-running-mean slot one contributes to input/running mean, and
next-running-variance slot two contributes to input/running variance. Its formulas retrieve the
exact same-occurrence saved mean and inverse-standard-deviation wrappers from slots three and
four. Those saved slots are formula inputs, not independent cotangent roots, publications,
physical [buffers](../../glossary.md#term-buffer), or a [runtime](../../glossary.md#term-runtime) tape.
For example, differentiating `sum(batchNormTraining(...).output())` with respect to scale is
supported, but differentiating that scale gradient with respect to the original input is rejected:
it would require derivatives of the saved statistics. The ordinary input path in the same formula
does not make that [second derivative](../../glossary.md#term-second-derivative) complete. This restriction does not change first-order
BatchNorm or prohibit using a [saved statistic](../../glossary.md#saved-statistic) solely to construct a comparison mask.

#### Matrix, movement and indexing contributions

These rules route cotangents back to the exact selected input shape. Repeated operand/index
positions can contribute more than once; the ordered accumulation is part of the constructed
program, not a set of unique numerical values.

MATMUL handles vector/vector, vector/matrix, matrix/vector, and matrix/matrix formulas with public
rank edits, last-two-axis [permutation](../../glossary.md#permutation), multiplication or MATMUL, followed by `sumToShape` for
each selected operand where batch broadcasting may have occurred and then one ordinary cast when
the promoted contribution type differs from that operand. Integral MATMUL remains
non-differentiable.
For matrix/vector input `A[batch..., rows, columns] @ b[columns]`, the vector cotangent restores
the seed's final [singleton](../../glossary.md#term-singleton) column: $A^{\mathsf T}\operatorname{expandDims}(g,-1)$, then squeezes that column and sums
the batch axes. Here `T` swaps the final two axes and `g` has Shape `[batch..., rows]`; these are
formula sketches, with explicit contiguous [materialization](../../glossary.md#materialization) at rank edits in the implementation.
For `A[2,2,2]` containing `1..8` and unit `g[2,2]`, the two batch contributions are `[4,6]` and
`[12,14]`, yielding vector gradient `[16,20]`. Treating `g` as a matrix would conflate its batch
axis with a matrix axis, even though the coincident dimensions happen to fit.

SLICE writes `g` into an input-shaped typed zero. `SliceAttrs` placement uses the exact stored
starts, axes, and steps; `CropToShapeAttrs` placement uses the exact prefix Shape. For
SLICE_UPDATE, the base role replaces the selected region of `g` with an update-shaped zero. The
update role uses `sliceByLength` with the exact recorded finite lengths for `SliceAttrs`, including
empty, signed-step, and unresolved selected-base regions, or `cropToShape` with the exact target
and prefix Shapes for `CropToShapeAttrs`. No raw end is reconstructed. SELECT writes an
axis-restored `g` at the selected coordinate. PAD crops away its before-width prefix. TILE
reshapes to interleaved repeat/input axes, sums the repeat axes, then restores the input Shape.
CONCAT crops `g` by each ordered symbolic input prefix, and STACK selects the corresponding
inserted-axis coordinate. Repeated input positions remain repeated contributions.

UNFOLD_AXIS uses `foldAxis` with the original selected [extent](../../glossary.md#term-extent) and step; FOLD_AXIS uses `unfold`
with the original window extent and step. Both UNFOLD2D forms use `fold2d` with the exact original
input Shape and window, while FOLD2D uses the conceptual-positive-zero `unfold2d` form. Typed
unfold [padding](../../glossary.md#padding) is scalar metadata and receives no cotangent. Dynamic two-dimensional [spatial](../../glossary.md#term-spatial-axes)
domains retain separate height and width constraints requiring the padded input or target extent
to contain the effective dilated [kernel](../../glossary.md#kernel). An undecidable relation remains an occurrence-owned
[deferred graph constraint](../../glossary.md#deferred-graph-constraint); a contradicted relation fails inference. This compiler stage neither
binds Dimensions nor chooses materialization.

GATHER, GATHER_ELEMENTS, and GATHER_ND route floating data cotangents through their matching
additive scatter, so repeated indices accumulate. SCATTER_ADD and configurable [scatter ADD](../../glossary.md#scatter-add)
preserve the base cotangent and Gather the update cotangent. Scatter NONE replacement zeros the
addressed base positions and Gathers the update cotangent; its forward uniqueness rule remains a
value-dependent execution obligation.

Scatter MUL counts represented zeros and constructs safe per-target products without dividing by
zero. A zero-free update receives the base times every other update; a sole zero receives the base
times all non-zero updates; every update in a multiple-zero group receives exact positive zero.
The base contribution multiplies `g` by the canonical Model product of all updates. NaN, infinity,
signed-zero, overflow, underflow, and rounding therefore follow the ordinary recorded Tensor
formula and Model forward product, except for the explicit positive-zero multiple-zero branch.
Scatter MIN/MAX compare the base and every duplicate update with the canonical forward result and
split `g` equally among numeric-equality winners. Opposite signed zeros tie. A [canonical NaN](../../glossary.md#term-canonical-nan)
result matches no candidate, so every candidate receives exact positive zero.

SORT constructs exactly one stable ARGSORT occurrence with the exact original input, [normalized
axis](../../glossary.md#normalized-axis), and direction, then replacement-scatters `g` through that permutation. This is the sole
matching recomputation in the current matrix. TOP_K instead uses the exact canonical slot-one
indices from its original producer; `k == 0` routes an input-shaped zero. Equal keys, NaN-last
membership, signed-zero order, and equal-cutoff membership freeze the forward route without
averaging. DROPOUT uses the exact canonical same-occurrence BOOL mask and
$\operatorname{where}(M,g/(1-q),0)$. It neither infers nor resamples a mask and does not
advance or differentiate [graph RNG state](../../glossary.md#graph-rng-state--graphrngstate). Here $M$ is the saved BOOL keep mask and $q$ the configured probability with
$0\le q<1$; the selected zero has the input type. At either signed probability zero, the denominator is
positive one and the all-kept mask routes `g`.

The original rules remain unchanged: broadcasted elementwise contributions use `sumToShape`;
ordinary SUM restores removed axes before expansion; CUM_SUM reverses scan direction while
retaining exclusivity; and PERMUTE uses the inverse axis order. WHERE routes no cotangent through
its BOOL condition.

#### Exact typed constants and accumulation

Generated BFLOAT16, FLOAT16, FLOAT32, and FLOAT64 exact scalar bases are [provenance](../../glossary.md#provenance)-free,
storage-free, non-gradient scalar leaves. BFLOAT16 uses exact bits `0x0000` and `0x3F80`;
FLOAT16 uses `0x0000` and `0x3C00`; FLOAT32 and FLOAT64 use exact positive `0.0` and `1.0`.
Each base is registered explicitly as one [logical splat](../../glossary.md#logical-splat), and Shape-specific values are ordinary
`expand` expressions.
One request-local [cache](../../glossary.md#term-cache) keys zero, one, extrema/clamp bounds, and other Tensor comparison
operands by exact data type and [represented bits](../../glossary.md#term-represented-bits), preserves deterministic first-use order, and
creates at most one base for each exact value.
No storage, factory history, label, [descriptor](../../glossary.md#term-descriptor), layout, Shape, or provenance absence implies a
[constant](../../glossary.md#term-constant). Only bases reachable from returned gradient expressions remain in combined-[capture](../../glossary.md#term-capture-meanings)
ingress; direct-zero local formulas therefore do not retain an unreachable unit seed.

Arithmetic-only coefficients remain scalar operation metadata. Compiler 0005A, the additive
Compiler 0005D loss rule, and the FLOAT16 cutover fix these exact bit patterns:

| Coefficient | BFLOAT16 | FLOAT16 | FLOAT32 | FLOAT64 |
|---|---:|---:|---:|---:|
| `0.5` | `0x3F00` | `0x3800` | `0x3F000000` | `0x3FE0000000000000` |
| `-0.5` | `0xBF00` | `0xB800` | `0xBF000000` | `0xBFE0000000000000` |
| $2$ | `0x4000` | `0x4000` | `0x40000000` | `0x4000000000000000` |
| `-2` | `0xC000` | `0xC000` | `0xC0000000` | `0xC000000000000000` |
| `invSqrt2` | `0x3F35` | `0x39A8` | `0x3F3504F3` | `0x3FE6A09E667F3BCD` |
| `invSqrt2Pi` | `0x3ECC` | `0x3662` | `0x3ECC422A` | `0x3FD9884533D43651` |
| `sqrt2OverPi` | `0x3F4C` | `0x3A62` | `0x3F4C422A` | `0x3FE9884533D43651` |
| `0.044715` | `0x3D37` | `0x29B9` | `0x3D372713` | `0x3FA6E4E26D4801F7` |
| `0.134145` | `0x3E09` | `0x304B` | `0x3E095D4F` | `0x3FC12BA9D1F60179` |

The ERF coefficient is `0x3F90`, `0x3C83`, `0x3F906EBB`, and
`0x3FF20DD750429B6D` in BFLOAT16, FLOAT16, FLOAT32, and FLOAT64 order. Compilation derives none of
these coefficients with a host transcendental function.

#### Combined capture and remaining boundaries

The compiler captures forward outputs and generated gradient roots together once. The request
includes target-specific roles, the original forward-producer identity set, and merged explicit
splats. Original producer occurrences become `FORWARD`; generated producers become `BACKWARD`.
The graph boundary keeps requested forward values first, then stable first-occurrence-distinct
gradient values. Every target role remains ordered even when multiple targets share one final
gradient [`ValueId`](../../glossary.md#logical-valueid-versus-physical-memory-slot).

Authoritative inference and validation follow the combined capture. [Canonicalization](../../glossary.md#occurrence-identity-and-canonical-form), the exact
task-0003A rewrites, task-0003B folding, [dead-code elimination](../../glossary.md#term-dead-code-elimination-dce), and initially phase-local
[common-subexpression elimination](../../glossary.md#term-common-subexpression-elimination-cse) [run](../../glossary.md#term-run) once over the complete [immutable](../../glossary.md#term-immutable) selected graph under their
existing guards. Every changed candidate is revalidated through the current verification pass.
A later construction, capture, validation, or optimization failure may consume temporary Tensor
IDs; IDs remain [opaque](../../glossary.md#term-opaque) and are not rolled back.

Everything outside the table fails closed when it lies on a selected route. The remaining matrix
is intentionally explicit:

| Classification | Current deferred or rejected families |
|---|---|
| Structured fail-closed boundaries | One-output attention lacks canonical same-occurrence weights and is rejected; index-target categorical cross-entropy rejects a dynamic or zero class depth; batch-training saved auxiliary cotangent paths are rejected, including beside connected sibling paths |
| Current source-backed closure | The shared ordinary formula owners cover admitted first-order roles, including fixed [RNN](../../glossary.md#term-recurrent-neural-network-rnn)/[GRU](../../glossary.md#term-gated-recurrent-unit-gru)/[LSTM](../../glossary.md#term-long-short-term-memory-lstm) through RecurrentGradientRules and length-aware Scan. [Conv3d](../../glossary.md#conv3d--ncdhw), both [Pool3d](../../glossary.md#pool3d--ncdhw-pooling) [signatures](../../glossary.md#term-signature), UNFOLD3D and FOLD3D use public Tensor expressions. Selected roles retain their differentiable, non-differentiable or fail-closed disposition; a Model-kind count alone is not an [admission](../../glossary.md#term-admission) test. Every actual generated occurrence must independently qualify on the selected [backend](../../glossary.md#backend). |
| Non-differentiable roles and outputs | Comparisons, BOOL logic/classification, `ALL`, `ANY`, arg-extrema results, one-hot and other index roles, ordering indices, dropout masks, padding constants, select coordinates, and graph RNG state |

Unknown/custom kinds, wrong attribute classes or cardinalities, missing canonical outputs, and
descriptor contradictions also fail deterministically. Later formulas must continue using the
shared Tensor algebra; this table does not reserve a gradient-specific numerical or optimization
policy for them.

This strategy adds no `Tensor.gradient`, `Tensor.backward`, [mutable](../../glossary.md#term-mutable) gradient field, ThreadLocal
[scope](../../glossary.md#term-scope), model derivative rule, placeholder/`ValueId` conversion map, direct graph-[node](../../glossary.md#node) formula
language, public gradient [registry](../../glossary.md#term-registry), runtime tape, physical saved buffer, backend [autograd](../../glossary.md#autograd), or
ordinary-user compile facade inside this formula owner. [Publication](../../glossary.md#term-publication) delivery, [optimizer](../../glossary.md#term-optimizer)
updates, [TrainingSession](../../glossary.md#training-session--trainingsession), [Planning](../../glossary.md#term-planning), backend [lowering](../../glossary.md#lowering), preparation and execution are implemented
by their existing separate owners; formula construction neither invokes nor duplicates them. The
functional request supports exactly one optional second [reverse-mode](../../glossary.md#primal-adjoint-and-differentiation-seed) stage; further orders remain
future work.


---

[Previous topic](optimization.md) · [Next topic](artifacts.md)
