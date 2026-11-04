# Pre-capture autograd and functional gradients

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/CompileArtifacts.java)

Automatic differentiation (**autograd**, or AD) builds expressions for derivatives by following
how an output depends on selected inputs. A **cotangent seed** specifies the output sensitivity
being propagated backward; for a scalar objective, a seed of one asks for its ordinary gradient.
Here those expressions are built before **capture**, the conversion to a graph with local node
and value identifiers. Constructing derivatives does not execute or update parameters.

<a id="current-package-private-pre-capture-autograd"></a>
### Current package-private pre-capture autograd

Model task 0025 is Complete, so a producer now returns the canonical exact Tensor wrapper for
every output slot, including hidden dropout masks and batch-normalization saved statistics,
without changing public Tensor methods or ergonomic result carriers. Compiler task 0004 is
Complete and supplies the first internal autograd consumer of that identity contract. Compiler
tasks 0004A and 0004B are also Complete with the exact-composition and shared-algebra local-rule
extensions described below. Compiler task 0005A is Complete and closes the exact 48-kind current
elementwise/activation inventory with explicit derivative and non-differentiable-role policies.
Compiler task 0005B is Complete and adds binding-aware expansion plus the current reduction,
scan, softmax, statistics, norm, and Layer/RMS/batch-normalization first-order matrix.
Compiler task 0005C is Complete and adds dynamic slice/window constraints plus the current
layout, Gather/scatter, ordering/top-K, and explicit-state dropout first-order matrix.
Compiler task 0005D is Complete and adds the current two-output attention, grouped convolution,
pooling, and loss first-order matrix without adding a public gradient API or backward-only
operation vocabulary.
Model task 0025G adds NCW Conv1d only as the ordinary visible
`EXPAND_DIMS(2) -> EXPAND_DIMS(2) -> CONV2D -> SQUEEZE(2)` Tensor composition. Generic capture
therefore sees two rank-edit nodes, one existing grouped Conv2d node, and one rank-edit node; it
does not see `CONV1D`, `ConvNd`, or a Conv1d attributes value. Existing rank-edit and Conv2d
inference and first-order gradient rules apply to those ordinary nodes without a new Compiler
inventory entry or formula. This structural reuse is a current compiler fact, not a fusion,
lowering, backend-capability, execution, or performance claim.
Model `CONV3D` records one first-class grouped NCDHW occurrence with two or three ordered inputs,
one output, and `Conv3dAttrs`. Compiler forward verification and first-order differentiation
adopt this signature. Independent inference derives the promoted type, rank-five result and
depth/height/width geometry, then final validation compares that result with the captured
descriptor. It proves or retains ordered obligations for input-channel divisibility,
output-channel divisibility, weight channels per group, optional bias length, and non-negative
depth, height, then width numerators. Conv3d uses the shared Conv2d/Conv3d gradient owner,
not a global backward rejection or a new backend-owned derivative.
Model task 0025J adds first-class `MAX_POOL3D` and `AVERAGE_POOL3D` Model metadata with exact
one-input/one-output signatures and rank-five NCDHW descriptors. Complete Compiler tasks 0006B1
and 0006B2 adopt both signatures, together with the three Model-current
`UNFOLD3D`/`FOLD3D` signatures, in ordinary forward verification and first-order differentiation.
CPU 0008G1 originally implemented the fully static resolved-layout non-gradient Pool3d subset.
Current matching CPU FLOAT32/FLOAT64 gradient metadata and qualified lowering reuse ordinary
PoolingGradientRules and UNFOLD3D/FOLD3D for Pool3d training; qualified Metal FLOAT32 supports
qualifying overlap through the same derivative expressions. Backend admission still requires
every generated occurrence. CPU low-precision execution and actual first-order AD use the same
ordinary owners under their current family/geometry guards, not an obsolete F32/F64 ceiling.
Earlier task receipts counted the inventory at individual checkpoints. Those counts and their
three-signature fixed-scan exclusion are not the current support contract. Fixed RNN, GRU and
LSTM first-order BPTT is implemented by
[RecurrentGradientRules](../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/RecurrentGradientRules.java),
which constructs a closed transition and differentiates it through the shared Scan owner.
Actual selected roles, source validity, canonical output availability and generated-topology
backend qualification still determine admission.

Current Compiler forward inference independently derives the `UNFOLD3D` and `FOLD3D` descriptors
as canonical columns
`[N, C * kD * kH * kW, DOut * HOut * WOut]`, literal floor/ceil depth-height-width grids, exact
direct-positive-zero or typed-padding unfold attributes, and an exact NCDHW fold target Shape.
It proves, rejects, or retains non-negative spatial-domain obligations and final-validates the
captured descriptor without binding dimensions or materializing windows. In backward-capable
modes, either `UNFOLD3D` form folds its cotangent into the exact captured input Shape, while
`FOLD3D` unfolds its cotangent with the direct positive-zero-padding overload. Typed unfold
padding is configuration and has no cotangent. These linear adjoints alternate under the supported
second derivative stage.

Package-private `GraphCompiler.compile` takes `CompileMode`, ordered forward outputs, an optional
public `FunctionalGradientRequest`, explicit forward constant ingress, and
`GraphOptimizationConfig` directly; there is no request aggregate. It returns package-private
mode-neutral `GraphCompilation`. `FORWARD_ONLY` requires no functional request, produces no
`BACKWARD` nodes, and has empty gradient results. `FORWARD_AND_BACKWARD` and the current internal
`TRAINING_STEP` path require the same request and may carry one combined graph.
`GraphCompilation` retains the final `ValidatedGraph`, ordered forward output values, and ordered
public `GradientPublicationBinding` values. Its package-private component names remain
`forwardOutputs` and `gradientResults`; the public publication plan instead exposes
`forwardBindings()` and `gradientBindings()`. It is graph-stage state, not the current public
cross-package `CompileArtifacts`.

The current functional request contains exactly one or two reverse-mode stages. Stage-one outputs
identify exact requested forward Tensors; stage-two outputs identify generated first-stage
gradients by target-list index. A one-stage request has `createGraph=false`; a two-stage request
has `true` for stage one and `false` for stage two. Each stage has a non-empty,
exact-identity-unique target list drawn from the complete original forward inventory and an
output-aligned list of optional cotangent seeds. An absent seed is valid only for a scalar,
floating, gradient-eligible output and creates an exact typed positive one. A present seed must
have the output's exact Shape and floating data type and must not request gradients. Multiple
selected outputs contribute a deterministic sum of vector-Jacobian products.

`DisconnectedPolicy.ERROR` rejects a target without a differentiable route.
`DisconnectedPolicy.ZERO` returns an ordinary exact typed zero expression; several result roles
may share that value. Results are ordered by derivative order and then stage-local target index.
`DerivativeGraphMetadata` classifies final nodes as derivative order zero, one, or two while
unchanged Model `GraphPhase` continues to distinguish only forward and backward nodes.

The sidecar also retains `scanActivityByNode()`, an immutable map keyed by containing Scan node
IDs. Each `ScanActivityPlan` names the exact independently scoped body graph, not a structurally
equal graph or a body-local ID in the containing inventory. The three-argument
`DerivativeGraphMetadata(graph, derivativeOrderByNode, scanActivityByNode)` constructor validates
these associations. The two-argument form intentionally creates no activity annotations; graph
rewrites and copies carrying annotated scans must preserve or explicitly remap them. This stays
inside the existing eighth `CompileArtifacts` component, not an additional aggregate component.

<a id="functional-gradient-examples"></a>
### Functional-gradient examples

These are compiler-package integration sketches because `GraphCompiler` remains package-private;
they describe current semantics without claiming a public compile facade.

For scalar `loss = x * x`, request stage one from `loss` to `x` with its default seed and
`createGraph=true`, then request stage two from
`FirstStageGradientReference(0)` to `x` with its default seed and `createGraph=false`. One combined
compile returns an order-one role for `2 * x` and an order-two role for `2`; both are ordinary
values in the same graph, and no Tensor gradient field is written.

For a vector output, supply an exact-Shape explicit seed to select its vector-Jacobian product.
A multi-output stage supplies one aligned seed per output and sums those products in output order.
The second stage may then seed a selected first-stage gradient with an exact-Shape vector to form
a Hessian-vector product. Omitting a seed for any non-scalar output is rejected before derivative
formulas are allocated.

Before constructing the seed or any formula Tensor, `AutogradPreflight` iteratively inventories
the complete original forward request. The package-private first-order coverage checker assigns
each exact selected output/input role one current disposition: conditionally differentiable
(`D`), intentionally non-differentiable (`ND`), or fail-closed (`FC`). Every `ND` or `FC`
decision has a deterministic reason. Every `D` decision names exactly one existing formula-family
owner, but remains conditional until preflight validates the occurrence's exact attributes,
cardinality, data-type and Shape relationships, canonical auxiliaries, normalization path, and
derivative policy. Unknown kinds, unclassified kind/attributes pairings, malformed
cardinalities, illegal slots, and unsupported prerequisites fail closed before derivative Tensor
identity allocation.

Preflight follows required cotangent paths from the selected stage outputs, stopping at `ND`
roles such as comparisons, masks, and indices. A required `FC` path to a requested target rejects
the stage even when a differentiable sibling path also reaches that target. Neither disconnected
policy permits omitting such a contribution. An unselected output or a saved value used only
behind an intentional non-differentiable boundary does not create a required derivative path.

The same family-owner decision is retained in the successful preflight plan and drives
`FirstOrderAutograd` dispatch, so role selection and formula routing do not maintain independent
kind branches. Named `ElementwiseGradientRules`, `ReductionGradientRules`,
`NormalizationGradientRules`, `LinearAlgebraGradientRules`, `LayoutGradientRules`,
`IndexingGradientRules`, `OrderingGradientRules`, and `StochasticGradientRules` then build
formulas only with existing public Tensor operations. `AttentionGradientRules`,
`ConvolutionGradientRules`, `PoolingGradientRules`, and `LossGradientRules` own the structured
Compiler 0005D formulas. Exact Tensor identity keys ordered contributions during one compile
request. Reverse accumulation visits producer postorder in reverse, selected output slots in
ascending order for that producer, and input positions in ascending order; ordinary
left-associated `Tensor.add` accumulates contributions. This ephemeral state is neither Tensor
state nor graph intermediate representation (IR).

The current matrix is the union of the Compiler 0004–0004B rows and the Compiler 0005A–0005D
family completions:

| Family | Current exact supported variants |
|---|---|
| Elementwise and activation | All seven promoted floating binary arithmetic kinds: `ADD`, `SUB`, `MUL`, `DIV`, `MIN`, `MAX`, and `POW`; all eight exact-type floating scalar kinds, including first-class `CLAMP`; promoted floating branch-only `WHERE`; floating-to-floating `CAST`; and all nineteen floating unary kinds from `ABS` through `SILU` |
| Reductions, scans, and softmax | Ordinary full, single-axis, and ordered multi-axis floating `SUM`, `MEAN`, `PROD`, `MIN`, and `MAX`; masked floating `SUM` and `MEAN`; binding-aware floating `SUM_TO_SHAPE`; floating `LOG_SUM_EXP`, `VARIANCE`, `STANDARD_DEVIATION`, `L1_NORM`, and `L2_NORM`; floating `CUM_SUM` and `CUM_PROD` for every exclusive/reverse combination; floating `SOFTMAX` and `LOG_SOFTMAX` |
| Normalization | No-affine and affine floating `LAYER_NORM`; no-scale and scaled floating `RMS_NORM`; all five floating inputs of batch-normalization inference; and the exact output-slot-specific batch-normalization-training roles for public slots zero through two |
| Linear algebra | Every current floating `MATMUL` vector/matrix rank pairing, with role-aware mixed-floating normalization and batch unbroadcasting |
| Logical layout and selection | Floating `CONTIGUOUS`, `RESHAPE`, binding-aware `EXPAND`, `EXPAND_DIMS`, `SQUEEZE`, `PERMUTE`, both `SLICE`/`SLICE_UPDATE` attribute forms, `SELECT`, `PAD`, `TILE`, `CONCAT`, and `STACK` |
| Window transforms | Floating `UNFOLD_AXIS`, `FOLD_AXIS`, both `UNFOLD2D` attribute forms, and `FOLD2D` through their exact public inverse or overlap-add operation |
| Indexing and scatter | Floating data for `GATHER`, `GATHER_ELEMENTS`, and `GATHER_ND`; floating base and update roles for `SCATTER_ADD`, `SCATTER_ELEMENTS`, and `SCATTER_ND` with `NONE`, `ADD`, `MUL`, `MIN`, or `MAX` as applicable |
| Ordering | One-output floating `SORT` through one exact matching stable `ARGSORT`; floating `TOP_K` values slot zero through the canonical indices at slot one |
| Stochastic | Floating `DROPOUT` values slot zero through the exact same-occurrence BOOL keep mask; mask and graph-state roles remain non-differentiable |
| Attention | Both public outputs of exact two-output floating `SCALED_DOT_PRODUCT_ATTENTION`: query, key, and value from values slot zero; query and key from canonical weights slot one; an optional BOOL mask remains non-differentiable |
| Convolution and pooling | Every floating input, weight, and optional bias role of grouped NCHW `CONV2D`; the sole floating input of fixed-count `AVERAGE_POOL2D` and exact first-logical-winner `MAX_POOL2D` |
| Losses | Prediction and target of floating `MEAN_SQUARED_ERROR`; logits and dense floating targets of dense categorical cross-entropy with logits; logits only for index-target categorical cross-entropy with a positive static class depth |

Forward and generated expressions use one model-owned Tensor algebra. For a selected mixed-
floating input, the compiler first reverses ordinary broadcasting with `sumToShape` when needed
and then uses ordinary `cast` exactly when the contribution type differs from the input type.
The resulting contribution has that input's exact Shape and floating data type before
deterministic accumulation. `WHERE` applies this rule independently to its two branch roles, and
MATMUL applies it after its rank-specific formula and batch-unbroadcast step. The BOOL condition,
mask, and other non-floating control roles receive no cotangent.

For a forward `CAST`, the existing rule is similarly limited to floating source and target types.
A same-type occurrence reuses the incoming cotangent; a cross-floating occurrence creates one
ordinary `CAST` back to the source type. That generated expression uses the same Model-owned
forward conversion contract documented in the [Tensor API](../tensor/selection-casts.md#cast-expressions),
including direct target-format rounding and deterministic NaN behavior. Compiler adds no
gradient-specific rounding, saturation, straight-through estimator, or value evaluator. Any cast
with an integral or BOOL source or target remains non-differentiable, and this documentation change
adds no Compiler production behavior.

Binary DIV constructs the left contribution as `g / right` and the right contribution in the
exact order `-(g * left) / (right * right)`. Scalar DIV constructs `g / scalar`. These are
ordinary Tensor expressions: they inherit the same division, multiplication, negation, NaN,
infinity, signed-zero, overflow, underflow, rounding, validation, and exact-optimization
contracts as forward expressions. No gradient-specific singularity or exceptional-value rule is
added.

Binary `MIN` and `MAX` route `g` to the strictly selected operand and split an exact numeric tie
equally. Each selected contribution then uses ordinary `sumToShape`-then-`cast` normalization.
Binary `POW(left, right)` constructs `g * right * pow(left, right - 1)` for `left` and
`g * output * log(left)` for `right`; it deliberately does not replace the first formula with
`output / left`.

Scalar `MIN` and `MAX` give the Tensor receiver `g` on its strict side, `g * 0.5` at an exact
numeric tie, and exact positive zero otherwise. Scalar `POW` constructs
`g * exponent * pow(input, exponentMinusOne)`. The derived scalar uses exactly one subtraction in
the represented type: BFLOAT16 expands to binary32, subtracts binary32 one, and rounds back;
FLOAT32 and FLOAT64 subtract one in their own formats. First-class `CLAMP` differentiates the
exact ordered forward composition `MIN(MAX(input, minValue), maxValue)`. With distinct finite
bounds the contribution is zero outside, `g` strictly inside, and `g / 2` at either endpoint;
when both extrema stages tie it is `g / 4`.

The added unary formulas are:

| Kind | Input contribution |
|---|---|
| `ABS` | `where(x > 0, g, where(x < 0, -g, 0))` |
| `RECIPROCAL` | `-g / (x * x)` |
| `LOG` | `g / x` |
| `LOG1P` | `g / (x + 1)` |
| `SQRT` | `g / (2 * output)` |
| `RSQRT` | `-0.5 * g * output * output * output` |
| `RELU` | `where(x > 0, g, 0)` |

Exact GELU uses the derivative of `x * Phi(x)`. Fixed tanh-approximation GELU differentiates its
specified `0.044715` cubic approximation, and SiLU differentiates `x * sigmoid(x)`. Each of these
three activations selects `g` at positive infinity and exact positive zero at negative infinity
to avoid a spurious infinity-times-zero NaN; finite and NaN inputs use the ordinary analytic
formula. Every other raw formula retains ordinary shared Tensor exceptional behavior without a
compiler-inserted finite, domain, singularity, or continuous-extension mask.

Piecewise extrema, `CLAMP`, `ABS`, and `RELU` use represented-value comparisons. Equal finite
values, equal same-sign infinities, and opposite signed zeros are ties. An unordered NaN makes
the ordered and equality tests false, so these piecewise rules return exact positive zero.
Comparisons, Boolean logic, floating classifications, the `WHERE` condition, scalar attributes
and clamp bounds, and non-floating cast roles remain non-differentiable.

`FLOOR`, `CEIL`, and `SIGN` use a direct exact positive-zero cotangent as their selected
first-order local convention. They do not construct `g * 0` or a floating comparison. This
convention does not change the forward operations' model semantics.

Ordinary MEAN restores and expands `g`, creates an input-shaped exact logical-one expression,
reduces those ones across the same full, single-axis, or ordered multi-axis domain, expands the
count, and divides. The same expression works for static, dynamic, expression, zero-sized, and
empty-axis-list domains without a host-computed count or type-specific threshold. Masked MEAN
constructs its per-slice true count with ordinary `WHERE` and `SUM`, divides the restored
cotangent by that count, then applies a final ordinary `WHERE`. The selected local convention
returns exact zero at every input coordinate of an all-false slice; it does not prescribe branch
evaluation or suppress the intermediate quotient.

For input `x` and output cotangent `g`, ERF constructs
`g * exp(-(x * x)) * (2 / sqrt(pi))`. The coefficient is exact scalar-operation metadata with
fixed BFLOAT16 bits `0x3F90`, FLOAT16 bits `0x3C83`, FLOAT32 bits `0x3F906EBB`, or
FLOAT64 bits `0x3FF20DD750429B6D`; compilation does not evaluate a host transcendental function or add a
coefficient Tensor leaf.

Masked SUM restores the removed axis, expands `g` to the data Shape, and selects that value where
the exact BOOL mask is true or an explicit typed zero where it is false. The mask receives no
cotangent. A forward `sumToShape` is inverted only when every right-aligned target Dimension is
the exact input Dimension, is static one, or retains the exact target-one-or-target-equal-source
predicate needed by the inverse `expand`. Leading input axes are permitted. Conversely, a
binding-dependent forward `EXPAND` emits, in aligned-axis order,
`source == 1 || source == target`; its inverse `SUM_TO_SHAPE` is admitted only through that same
predicate. Deferred predicates remain attached to the occurrence for later binding validation.
The compiler does not bind a Dimension, choose a stride, or treat a deferred result as execution
permission.

`PROD` uses sequential keep-dimension product stages and reverses each stage with exclusive
prefix and suffix `CUM_PROD`; it never divides by the selected input. `CUM_PROD` replaces
represented zeros with one for safe products, counts zeros with `CUM_SUM`, and accumulates in the
opposite scan direction. These formulas handle zero-free, one-zero, multiple-zero, exclusive,
reverse, and zero-length structure without a gradient-specific infinity or NaN repair.

Reduction `MIN` and `MAX` compare each input with the exact restored forward output and divide the
restored cotangent equally among all matching coordinates. Opposite signed zeros are ties. A NaN
forward output matches no coordinate, so the selected contribution is exact positive zero.
`LOG_SUM_EXP` uses `restored(g) * exp(input - restored(output))`.

`VARIANCE` and `STANDARD_DEVIATION` derive selected-domain count by reducing exact logical ones
and construct correction as a typed logical-one reduction rather than a host floating
conversion. Standard deviation returns exact zero when the saved result is not strictly
positive. `L1_NORM` selects sign with exact zero at signed zero or NaN; `L2_NORM` returns exact
zero when its saved norm is not strictly positive. Other exceptional values follow the ordinary
Tensor formulas in their recorded order.

`SOFTMAX` consumes exact forward output `y` and constructs
`y * (g - sum(g * y, axis, true))`. `LOG_SOFTMAX` constructs
`g - exp(y) * sum(g, axis, true)`. Neither rule recomputes a softmax occurrence or selects a
forward numerical algorithm.

Layer and root-mean-square (RMS) normalization rebuild their population statistics with ordinary
Tensor operations over the exact trailing normalized axes. Mixed-floating occurrences compute in
the selected forward-output type, align affine operands by visible reshape/expand expressions,
and cast each completed Shape-correct contribution back to its selected input type. Layer
normalization supports input, scale, and bias roles; RMS normalization supports input and
optional scale.

Batch-normalization inference supports input, scale, bias, running mean, and running variance.
Batch-normalization training is output-slot-aware: normalized output slot zero contributes to
input/scale/bias, next-running-mean slot one contributes to input/running mean, and
next-running-variance slot two contributes to input/running variance. Its formulas retrieve the
exact same-occurrence saved mean and inverse-standard-deviation wrappers from slots three and
four. Those saved slots are formula inputs, not independent cotangent roots, publications,
physical buffers, or a runtime tape.
For example, differentiating `sum(batchNormTraining(...).output())` with respect to scale is
supported, but differentiating that scale gradient with respect to the original input is rejected:
it would require derivatives of the saved statistics. The ordinary input path in the same formula
does not make that second derivative complete. This restriction does not change first-order
BatchNorm or prohibit using a saved statistic solely to construct a comparison mask.

MATMUL handles vector/vector, vector/matrix, matrix/vector, and matrix/matrix formulas with public
rank edits, last-two-axis permutation, multiplication or MATMUL, followed by `sumToShape` for
each selected operand where batch broadcasting may have occurred and then one ordinary cast when
the promoted contribution type differs from that operand. Integral MATMUL remains
non-differentiable.
For matrix/vector input `A[batch..., rows, columns] @ b[columns]`, the vector cotangent restores
the seed's final singleton column: `T(A) @ expandDims(g, -1)`, then squeezes that column and sums
the batch axes. Here `T` swaps the final two axes and `g` has Shape `[batch..., rows]`; these are
formula sketches, with explicit contiguous materialization at rank edits in the implementation.
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

UNFOLD_AXIS uses `foldAxis` with the original selected extent and step; FOLD_AXIS uses `unfold`
with the original window extent and step. Both UNFOLD2D forms use `fold2d` with the exact original
input Shape and window, while FOLD2D uses the conceptual-positive-zero `unfold2d` form. Typed
unfold padding is scalar metadata and receives no cotangent. Dynamic two-dimensional spatial
domains retain separate height and width constraints requiring the padded input or target extent
to contain the effective dilated kernel. An undecidable relation remains an occurrence-owned
deferred graph constraint; a contradicted relation fails inference. This compiler stage neither
binds Dimensions nor chooses materialization.

GATHER, GATHER_ELEMENTS, and GATHER_ND route floating data cotangents through their matching
additive scatter, so repeated indices accumulate. SCATTER_ADD and configurable scatter ADD
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
split `g` equally among numeric-equality winners. Opposite signed zeros tie. A canonical NaN
result matches no candidate, so every candidate receives exact positive zero.

SORT constructs exactly one stable ARGSORT occurrence with the exact original input, normalized
axis, and direction, then replacement-scatters `g` through that permutation. This is the sole
matching recomputation in the current matrix. TOP_K instead uses the exact canonical slot-one
indices from its original producer; `k == 0` routes an input-shaped zero. Equal keys, NaN-last
membership, signed-zero order, and equal-cutoff membership freeze the forward route without
averaging. DROPOUT uses the exact canonical same-occurrence BOOL mask and
`where(mask, g / (1 - probability), zero)`. It neither infers nor resamples a mask and does not
advance or differentiate graph RNG state. At either signed probability zero, the denominator is
positive one and the all-kept mask routes `g`.

The original rules remain unchanged: broadcasted elementwise contributions use `sumToShape`;
ordinary SUM restores removed axes before expansion; CUM_SUM reverses scan direction while
retaining exclusivity; and PERMUTE uses the inverse axis order. WHERE routes no cotangent through
its BOOL condition.

Generated BFLOAT16, FLOAT16, FLOAT32, and FLOAT64 exact scalar bases are provenance-free,
storage-free, non-gradient scalar leaves. BFLOAT16 uses exact bits `0x0000` and `0x3F80`;
FLOAT16 uses `0x0000` and `0x3C00`; FLOAT32 and FLOAT64 use exact positive `0.0` and `1.0`.
Each base is registered explicitly as one logical splat, and Shape-specific values are ordinary
`expand` expressions.
One request-local cache keys zero, one, extrema/clamp bounds, and other Tensor comparison
operands by exact data type and represented bits, preserves deterministic first-use order, and
creates at most one base for each exact value.
No storage, factory history, label, descriptor, layout, Shape, or provenance absence implies a
constant. Only bases reachable from returned gradient expressions remain in combined-capture
ingress; direct-zero local formulas therefore do not retain an unreachable unit seed.

Arithmetic-only coefficients remain scalar operation metadata. Compiler 0005A, the additive
Compiler 0005D loss rule, and the FLOAT16 cutover fix these exact bit patterns:

| Coefficient | BFLOAT16 | FLOAT16 | FLOAT32 | FLOAT64 |
|---|---:|---:|---:|---:|
| `0.5` | `0x3F00` | `0x3800` | `0x3F000000` | `0x3FE0000000000000` |
| `-0.5` | `0xBF00` | `0xB800` | `0xBF000000` | `0xBFE0000000000000` |
| `2` | `0x4000` | `0x4000` | `0x40000000` | `0x4000000000000000` |
| `-2` | `0xC000` | `0xC000` | `0xC0000000` | `0xC000000000000000` |
| `invSqrt2` | `0x3F35` | `0x39A8` | `0x3F3504F3` | `0x3FE6A09E667F3BCD` |
| `invSqrt2Pi` | `0x3ECC` | `0x3662` | `0x3ECC422A` | `0x3FD9884533D43651` |
| `sqrt2OverPi` | `0x3F4C` | `0x3A62` | `0x3F4C422A` | `0x3FE9884533D43651` |
| `0.044715` | `0x3D37` | `0x29B9` | `0x3D372713` | `0x3FA6E4E26D4801F7` |
| `0.134145` | `0x3E09` | `0x304B` | `0x3E095D4F` | `0x3FC12BA9D1F60179` |

The ERF coefficient is `0x3F90`, `0x3C83`, `0x3F906EBB`, and
`0x3FF20DD750429B6D` in BFLOAT16, FLOAT16, FLOAT32, and FLOAT64 order. Compilation derives none of
these coefficients with a host transcendental function.

The compiler captures forward outputs and generated gradient roots together once. The request
includes target-specific roles, the original forward-producer identity set, and merged explicit
splats. Original producer occurrences become `FORWARD`; generated producers become `BACKWARD`.
The graph boundary keeps requested forward values first, then stable first-occurrence-distinct
gradient values. Every target role remains ordered even when multiple targets share one final
gradient `ValueId`.

Authoritative inference and validation follow the combined capture. Canonicalization, the exact
task-0003A rewrites, task-0003B folding, dead-code elimination, and initially phase-local
common-subexpression elimination run once over the complete immutable selected graph under their
existing guards. Every changed candidate is revalidated through the current verification pass.
A later construction, capture, validation, or optimization failure may consume temporary Tensor
IDs; IDs remain opaque and are not rolled back.

Everything outside the table fails closed when it lies on a selected route. The remaining matrix
is intentionally explicit:

| Classification | Current deferred or rejected families |
|---|---|
| Structured fail-closed boundaries | One-output attention lacks canonical same-occurrence weights and is rejected; index-target categorical cross-entropy rejects a dynamic or zero class depth; batch-training saved auxiliary cotangent paths are rejected, including beside connected sibling paths |
| Current source-backed closure | The shared ordinary formula owners cover admitted first-order roles, including fixed RNN/GRU/LSTM through RecurrentGradientRules and length-aware Scan. Conv3d, both Pool3d signatures, UNFOLD3D and FOLD3D use public Tensor expressions. Selected roles retain their differentiable, non-differentiable or fail-closed disposition; a Model-kind count alone is not an admission test. Every actual generated occurrence must independently qualify on the selected backend. |
| Non-differentiable roles and outputs | Comparisons, BOOL logic/classification, `ALL`, `ANY`, arg-extrema results, one-hot and other index roles, ordering indices, dropout masks, padding constants, select coordinates, and graph RNG state |

Unknown/custom kinds, wrong attribute classes or cardinalities, missing canonical outputs, and
descriptor contradictions also fail deterministically. Later formulas must continue using the
shared Tensor algebra; this table does not reserve a gradient-specific numerical or optimization
policy for them.

This strategy adds no `Tensor.gradient`, `Tensor.backward`, mutable gradient field, ThreadLocal
scope, model derivative rule, placeholder/`ValueId` conversion map, direct graph-node formula
language, public gradient registry, runtime tape, physical saved buffer, backend autograd, or
ordinary-user compile facade inside this formula owner. Publication delivery, optimizer
updates, TrainingSession, Planning, backend lowering, preparation and execution are implemented
by their existing separate owners; formula construction neither invokes nor duplicates them. The
functional request supports exactly one optional second reverse-mode stage; further orders remain
future work.


---

[Previous topic](optimization.md) · [Next topic](artifacts.md)
