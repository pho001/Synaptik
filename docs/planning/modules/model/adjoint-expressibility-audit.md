# [Adjoint](../../../glossary.md#term-adjoint) Expressibility Audit

## Purpose and authority

This [planning](../../../glossary.md#term-planning) [artifact](../../../glossary.md#term-artifact) records whether each selected public [Tensor](../../../glossary.md#tensor) [operation](../../../glossary.md#operation) has an exact
[backend](../../../glossary.md#backend)-neutral [adjoint](../../../glossary.md#term-adjoint) composition in the current [model](../../../glossary.md#term-model) vocabulary. It is authoritative only for
subsequent planning. It is not an [architecture contract](../../../glossary.md#architecture-contract), public [API](../../../glossary.md#term-application-programming-interface-api) reference, implemented
[gradient](../../../glossary.md#term-gradient) rule, [compiler](../../../glossary.md#term-compiler) design, or backend promise.

The organizing question is:

```text
forward occurrence + incoming output cotangent
  -> restore each input's exact Shape and type
     -> current composition, existing producer output, general primitive gap,
        non-differentiable role, or deferred derivative policy
```

The audit uses current repository semantics as primary evidence. It does not infer a [derivative](../../../glossary.md#term-derivative)
policy from `requiresGrad`, formula popularity, implementation cost, or likely [fusion](../../../glossary.md#term-fusion).

## Notation and shared obligations

For an [operation](../../../glossary.md#operation) `y = f(x...)`:

- `g_y` is the incoming [cotangent](../../../glossary.md#cotangent) with `y.shape` and `y.type`.
- `a_x` is the accumulated [adjoint](../../../glossary.md#term-adjoint) for input `x`, with exactly `x.shape` and `x.type`.
- `S_A(t, axes)` is `sum(axes, keepDimensions=true)` followed, when necessary, by
  `reshape(t, A.shape)`; selected [axes](../../../glossary.md#term-axis) are fixed by the forward attributes.
- `B_A(t)` is the current broadcast reversal for ordinary elementwise broadcasting. It sums
  target-only leading axes and axes where <span data-glossary-navigation="true"> ([elementwise glossary](../../../glossary.md#term-elementwise); [broadcasting glossary](../../../glossary.md#broadcasting); [target glossary](../../../glossary.md#term-target))</span>`A` has a statically known [singleton](../../../glossary.md#term-singleton), then reshapes to
  `A.shape`. `ShapeBroadcast` and public `expand` reject every other unequal unresolved pair, so
  these axes are known from the forward Shapes without later binding.
- `R_A(t)` is a planned binding-aware sum-to-Shape transformation. It removes leading broadcast
  axes and sums axes whose bound <span data-glossary-navigation="true"> ([sum-to-Shape glossary](../../../glossary.md#sum-to-shape))</span>`A` extent is one. It is needed only by current operations such
  as MATMUL and attention that explicitly retain an unresolved singleton-or-equal batch
  obligation; current fixed-axis reductions cannot express that decision before binding.<span data-glossary-navigation="true"> ([extent glossary](../../../glossary.md#term-extent))</span>
- `Z_A` and `O_A` are graph expressions made today from an exact typed rank-zero zero or one leaf
  and <span data-glossary-navigation="true"> ([graph glossary](../../../glossary.md#graph); [rank-zero glossary](../../../glossary.md#axis-rank-scalar-vector-and-matrix); [leaf glossary](../../../glossary.md#term-leaf))</span>`expand(A.shape)`. They do not inspect `A`, so [NaN](../../../glossary.md#term-nan) or [infinity](../../../glossary.md#term-infinity) in `A` cannot contaminate the
  constant. <span data-glossary-navigation="true"> ([constant glossary](../../../glossary.md#term-constant))</span>`TensorFactory.zerosLike` and `onesLike` are eager fully-static alternatives, not the
  general graph construction.<span data-glossary-navigation="true"> ([eager glossary](../../../glossary.md#term-eager-evaluation))</span>
- `E_A(t)` inserts every axis removed by a forward reduction, then expands to `A.shape`.
- `P^-1(t)` applies the inverse of a recorded [permutation](../../../glossary.md#permutation).
- `Y`, `M`, and similar capitals denote a saved public output or recomputed forward value.
- `aux[n]` is exact output slot `n` of the same [`TensorProducer`](../../../glossary.md#producer--tensorproducer); it is not a newly exposed public
  Tensor.<span data-glossary-navigation="true"> ([Tensor glossary](../../../glossary.md#tensor))</span>
- [`ND`](../../../glossary.md#term-nd) marks a non-[differentiable](../../../glossary.md#derivative-and-chain-rule) [Boolean](../../../glossary.md#data-type--datatype), integral, index, or [RNG](../../../glossary.md#graph-rng-state--graphrngstate)-state role.

Ordinary [binary](../../../glossary.md#term-binary-descriptor-and-metadata), `where`, and explicit-expand reversal ends in `B_A`; [MATMUL](../../../glossary.md#matrix-multiplication--matmul) and attention [batch](../../../glossary.md#term-batch-machine-learning-and-hyperparameters)
reversal ends in `R_A` when the forward [occurrence](../../../glossary.md#term-occurrence) retained a [binding](../../../glossary.md#term-binding)-dependent singleton. [Compiler](../../../glossary.md#term-compiler)
[gradient accumulation](../../../glossary.md#term-gradient-accumulation) occurs only after each contribution has the input's exact [Shape](../../../glossary.md#shape) and type,
using ordinary `ADD`. Floating promotion may require an explicit floating `CAST` back to the
input's type; the cross-floating cast [derivative](../../../glossary.md#term-derivative) policy itself remains deferred below.

The regular-domain formulas below [mean](../../../glossary.md#term-mean) [finite](../../../glossary.md#term-finite) inputs at points where the mathematical derivative
exists and every forward [validation](../../../glossary.md#term-validation) obligation is satisfied. A separate boundary row records
unselected behavior at discontinuities, ties, [infinities](../../../glossary.md#nan-infinity-and-signed-zero), NaNs, zero denominators, or other
non-differentiable points. That split prevents a regular formula from silently selecting policy
for all accepted floating values.

## Fixed classifications

- `EXACT_CURRENT_COMPOSITION` — current public operations express the complete regular-domain
  adjoint for every Shape accepted by that row.<span data-glossary-navigation="true"> ([adjoint glossary](../../../glossary.md#term-adjoint); [Shape glossary](../../../glossary.md#shape))</span>
- `EXACT_WITH_EXISTING_AUXILIARY_OUTPUT` — the composition additionally uses a producer-described
  output that already exists.<span data-glossary-navigation="true"> ([producer glossary](../../../glossary.md#term-producer))</span>
- `MISSING_GENERAL_PUBLIC_PRIMITIVE` — a stable tensor transformation useful outside autograd is
  required but absent.<span data-glossary-navigation="true"> ([tensor glossary](../../../glossary.md#tensor); [autograd glossary](../../../glossary.md#autograd))</span>
- `GENUINELY_NON_EXPRESSIBLE_SEMANTIC_GAP` — no exact composition or appropriate general public
  primitive can represent the stable meaning. The completed matrix selects no row in this class.<span data-glossary-navigation="true"> ([matrix glossary](../../../glossary.md#term-matrix-meanings))</span>
- `NON_DIFFERENTIABLE` — the role is outside the selected [differentiable](../../../glossary.md#derivative-and-chain-rule) domain.
- `POLICY_DEFERRED` — current semantics do not choose the derivative behavior at an observable
  boundary.<span data-glossary-navigation="true"> ([derivative glossary](../../../glossary.md#term-derivative))</span>

## Repository evidence inventory

The inventory covered all 188 current public [Tensor](../../../glossary.md#tensor) methods and every concrete [operation](../../../glossary.md#operation) family.
The principal evidence is:

- [public Tensor construction](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)
  and the [Tensor API reference](../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([API glossary](../../../glossary.md#term-application-programming-interface-api))</span>;
- [operation families](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/OperationKind.java),
  family attributes and signatures, and their matching expression helpers and tests under<span data-glossary-navigation="true"> ([signatures glossary](../../../glossary.md#term-signature))</span>
  `modules/model/src/main` and `modules/model/src/test`;
- [shared producer identity](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorProducer.java)<span data-glossary-navigation="true"> ([identity glossary](../../../glossary.md#term-identity-meanings))</span>
  and [indexed provenance](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorProvenance.java)<span data-glossary-navigation="true"> ([provenance glossary](../../../glossary.md#provenance))</span>;
- completed semantic and expression tasks [0014A through 0022B](tasks/0014a-binary-arithmetic-semantic-kinds.md),
  including [operation hardening](tasks/0018k-operation-signature-and-construction-hardening.md),
  [shared outputs](tasks/0018l-shared-multi-output-tensor-provenance.md),
  [symbolic extents](tasks/0018m-symbolic-extent-expressions.md),
  [indexing cleanup](tasks/0018o-indexing-taxonomy-and-unstack-normalization.md), and
  [slice/window cleanup](tasks/0018r-slice-and-window-public-contract-cleanup.md).

Conveniences are audited through their actual chains: [`linear`](../../../glossary.md#term-linear-meanings) is `PERMUTE -> MATMUL -> optional
ADD`; [`embedding`](../../../glossary.md#embedding) is `GATHER(axis=0)`; [`transpose`](../../../glossary.md#term-transpose) is [rank-two](../../../glossary.md#axis-rank-scalar-vector-and-matrix) `PERMUTE[1,0]`; `flip` is one signed
[`SLICE`](../../../glossary.md#slice); `clampMin` and `clampMax` are [scalar](../../../glossary.md#term-scalar) `MAX` and `MIN`; and `unstack` is repeated `SELECT`,
not a multi-output operation.

Existing [producer](../../../glossary.md#term-producer) output positions are exact:

| [Occurrence](../../../glossary.md#term-occurrence) | Output positions | Audit use |
|---|---|---|
| [`DROPOUT`](../../../glossary.md#dropout--inverted-dropout) | `0=output`, `1=BOOL keep mask`, `2=next INT64 state` | `aux[1]` is required; slots 0 and 2 are public through `DropoutResult`. |
| `BATCH_NORM_TRAINING` | `0=output`, `1=next running mean`, `2=next running variance`, `3=saved batch mean`, `4=saved inverse standard deviation` | `aux[3]` and `aux[4]` are required by the regular input [adjoint](../../../glossary.md#term-adjoint). |
| `TOP_K` | `0=values`, `1=INT64 indices` | Slot 1 routes the value [cotangent](../../../glossary.md#cotangent); it is public through `TopKResult`. |

Attention and maximum [pooling](../../../glossary.md#term-pooling) each have only output slot zero. The [matrix](../../../glossary.md#term-matrix-meanings) proves that attention
needs its [weights](../../../glossary.md#term-weight) from the same forward occurrence, while maximum-pool selection can be recomputed
from current first-index arg-maximum semantics once the general window gap is closed.

## Expressibility [matrix](../../../glossary.md#term-matrix-meanings)

Each row states the forward kind/attributes or [producer](../../../glossary.md#term-producer) chain, audited role, [Shape](../../../glossary.md#shape) assumptions,
required values, exact regular formula, obligations, classification, minimum gap when applicable,
later [owner](../../../glossary.md#term-owner), and repository evidence. [Scalar](../../../glossary.md#term-scalar) attributes are configuration, not [differentiable](../../../glossary.md#derivative-and-chain-rule)
[Tensor](../../../glossary.md#tensor) roles.

### [Elementwise](../../../glossary.md#term-elementwise) arithmetic, selection, casts, and [activations](../../../glossary.md#term-activation)

| ID | Kind and audited role | [Shape](../../../glossary.md#shape), values, and exact formula | Obligations and decision | Classification; gap; [owner](../../../glossary.md#term-owner); evidence |
|---|---|---|---|---|
| E1 | `BinaryArithmeticKind.ADD`; left/right | `y,g_y` use broadcast Shape. `a_left=B_left(g_y)`; `a_right=B_right(g_y)`. | The current broadcast contract exposes only leading or static-[singleton](../../../glossary.md#term-singleton) expansion [axes](../../../glossary.md#term-axis). | `EXACT_CURRENT_COMPOSITION`; [compiler](../../../glossary.md#term-compiler); [binary](../../../glossary.md#term-binary-descriptor-and-metadata) helper/tests and tasks 0014A–0014B, 0018T. |
| E2 | `SUB`; left/right | `a_left=B_left(g_y)`; `a_right=B_right(-g_y)`. | Same statically determined broadcast reversal as E1. | `EXACT_CURRENT_COMPOSITION`; compiler. |
| E3 | `MUL`; left/right | Save or recompute inputs. `a_left=B_left(g_y*right)`; `a_right=B_right(g_y*left)`. | Ordinary [IEEE](../../../glossary.md#term-ieee) multiplication is retained; accepted expansion axes are statically identifiable. | `EXACT_CURRENT_COMPOSITION` on the regular domain; compiler. |
| E4 | `DIV`; left/right, regular denominator | `a_left=B_left(g_y/right)`; `a_right=B_right(-(g_y*left)/(right*right))`. | Zero, [infinity](../../../glossary.md#term-infinity), [NaN](../../../glossary.md#term-nan), and [overflow](../../../glossary.md#term-overflow) [derivative](../../../glossary.md#term-derivative) behavior is not selected. | `EXACT_CURRENT_COMPOSITION` on the regular domain; compiler. Boundary is E8. |
| E5 | [tensor](../../../glossary.md#tensor) `POW`; base/[exponent](../../../glossary.md#term-exponent), regular real domain | With `Y=pow(left,right)`: `a_left=B_left(g_y*right*pow(left,right-1))`; `a_right=B_right(g_y*Y*log(left))`. | Negative bases, zero, [infinities](../../../glossary.md#nan-infinity-and-signed-zero), NaNs, and domain transitions need policy. | `EXACT_CURRENT_COMPOSITION` on the regular domain; compiler. Boundary is E8. |
| E6 | [scalar](../../../glossary.md#term-scalar) `ADD`,`SUB`,`MUL`; receiver | `g_y`, `g_y`, and `g_y*scalar`, respectively; Shape/type already match. Scalar attrs are not Tensor roles. | Exact typed scalar bits are retained. | `EXACT_CURRENT_COMPOSITION`; compiler; scalar family/tasks 0018N and 0018T. |
| E7 | scalar `DIV`,`POW`; receiver, regular domain | `g_y/scalar`; and `g_y*scalar*pow(x,scalar-1)`. Scalar attrs are non-[differentiable](../../../glossary.md#derivative-and-chain-rule) configuration. | Zero/domain/special boundaries remain unselected. | `EXACT_CURRENT_COMPOSITION` on the regular domain; compiler. Boundary is E8. |
| E8 | binary/scalar `DIV`/`POW` boundary cases | Regular formulas are E4, E5, E7. | Required policy is the [adjoint](../../../glossary.md#term-adjoint) at zero denominators, non-real power points, infinities, NaNs, and overflow-generated exceptional values. | `POLICY_DEFERRED`; compiler derivative-policy task, not a [model](../../../glossary.md#term-model) kind. |
| E9 | pairwise/scalar `MIN`,`MAX`, `CLAMP`; Tensor roles | Away from equal operands/endpoints, route `g_y` to the uniquely selected input and zero to the other, then apply `B` for tensor [broadcasting](../../../glossary.md#broadcasting). `clampMin`/`clampMax` are scalar MAX/MIN. | Current forward semantics fix NaN and [signed-zero](../../../glossary.md#nan-infinity-and-signed-zero) selection but no tie, endpoint, or NaN derivative convention. | `POLICY_DEFERRED`; compiler policy. No backward kind or primitive is selected. |
| E10 | `WHERE`; true/false branches; condition [ND](../../../glossary.md#term-nd) | `a_true=B_true(where(condition,g_y,Z_y))`; `a_false=B_false(where(condition,Z_y,g_y))`. Condition is `NON_DIFFERENTIABLE`. | Scalar-expanded exact zero avoids `0*NaN`; branch expansion axes are leading or statically singleton. | Branches: `EXACT_CURRENT_COMPOSITION`. Condition: `NON_DIFFERENTIABLE`; compiler; tasks 0015E–0015F. |
| E11 | floating `CAST`; source | Same-type cast: `a_x=g_y`. Cross-floating cast would require a compiler-selected [cotangent](../../../glossary.md#cotangent) conversion. Integral/[BOOL](../../../glossary.md#data-type--datatype) sources or [targets](../../../glossary.md#term-target) are ND. | Current cast semantics deliberately do not define numerical conversion or [gradients](../../../glossary.md#term-gradient). | Same type: `EXACT_CURRENT_COMPOSITION`. Cross-floating: `POLICY_DEFERRED`. Non-floating role: `NON_DIFFERENTIABLE`; compiler policy. |
| E12 | `NEG`,[`EXP`](../../../glossary.md#term-expx),`EXPM1`,[`SIGMOID`](../../../glossary.md#term-sigmoid),[`TANH`](../../../glossary.md#term-tanh); receiver, regular domain | Respectively `-g_y`, `g_y*Y`, `g_y*(Y+1)`, `g_y*Y*(1-Y)`, `g_y*(1-Y*Y)`. | Shape/type preserve exactly; outputs may be saved or recomputed. | `EXACT_CURRENT_COMPOSITION`; compiler; [unary](../../../glossary.md#term-unary) helper and tasks 0014C–0014D, 0018P–0018T1. |
| E13 | `RECIPROCAL`,[`LOG`](../../../glossary.md#term-log),`LOG1P`,`SQRT`,`RSQRT`; regular domain | `-g_y/(x*x)`, `g_y/x`, `g_y/(1+x)`, `g_y/(2*Y)`, and `-0.5*g_y*Y*Y*Y`. | Zero, negative-domain, infinities, and NaNs require an explicit derivative boundary policy. | `EXACT_CURRENT_COMPOSITION` on regular points; compiler. Boundary: `POLICY_DEFERRED` under E17. |
| E14 | [`ERF`](../../../glossary.md#term-erf); receiver, regular domain | `a_x=g_y*(2/sqrt(pi))*exp(-(x*x))`; the coefficient is an exact typed scalar [leaf](../../../glossary.md#term-leaf) selected by compiler policy. | [Finite](../../../glossary.md#term-finite) [rounding](../../../glossary.md#term-rounding) [tolerance](../../../glossary.md#term-tolerance) is later [backend](../../../glossary.md#backend) [conformance](../../../glossary.md#term-conformance), not semantic evidence. | `EXACT_CURRENT_COMPOSITION`; compiler. |
| E15 | [`GELU`](../../../glossary.md#gaussian-error-linear-unit--gelu),`GELU_TANH_APPROXIMATION`,`SILU`; receiver, regular domain | Exact GELU uses its `erf` formula and derivative; approximation differentiates the fixed tanh polynomial; [SiLU](../../../glossary.md#sigmoid-linear-unit--silu) uses `sigmoid(x)*(1+x*(1-sigmoid(x)))`. Multiply each by `g_y`. | [Constants](../../../glossary.md#term-constant) use typed scalar leaves; saved `x` and recomputed intermediates suffice. | `EXACT_CURRENT_COMPOSITION`; compiler; modern activation task 0019A. |
| E16 | `ABS`,`RELU`,`FLOOR`,`CEIL`,`SIGN`; receiver | Away from boundaries: ABS routes `sign(x)*g_y`; [ReLU](../../../glossary.md#term-relu) uses `where(x>0,g_y,Z_y)`; floor, ceil, and sign have zero derivative on open constant regions. | Behavior at zero, integers/discontinuities, NaN, infinities, and [signed zero](../../../glossary.md#term-signed-zero) is not selected. | `POLICY_DEFERRED`; compiler derivative policy. |
| E17 | exceptional unary points | Regular formulas are E12–E16. | No current contract selects derivatives at invalid domains, infinities, NaNs, or zero singularities. | `POLICY_DEFERRED`; compiler policy, with no model-semantic gap. |
| E18 | comparisons, [Boolean](../../../glossary.md#data-type--datatype) logic, `IS_FINITE`/`IS_NAN`/`IS_INF` | Inputs may be floating, but results are BOOL and no output cotangent participates. | Classification is independent of `requiresGrad` [metadata](../../../glossary.md#term-metadata). | `NON_DIFFERENTIABLE`; compiler stops propagation. |

### Reductions, scans, and [normalization](../../../glossary.md#term-normalization)

| ID | Kind and audited role | [Shape](../../../glossary.md#shape), values, and exact formula | Obligations and decision | Classification; gap; [owner](../../../glossary.md#term-owner); evidence |
|---|---|---|---|---|
| R1 | full/single-/multi-[axis](../../../glossary.md#term-axis) `SUM`; input | `a_x=E_x(g_y)`. Keep-[dimension](../../../glossary.md#dimension) forms only expand; removed-axis forms first insert recorded axes. | Axes are intrinsic and fixed; dynamic extents do not affect axis [identity](../../../glossary.md#term-identity-meanings). | `EXACT_CURRENT_COMPOSITION`; [compiler](../../../glossary.md#term-compiler); reduction attrs/helpers and tasks 0016A–0016B, 0018V. |
| R2 | [`MEAN`](../../../glossary.md#term-mean); input, non-empty domain | Let `N=sum(E_x(O_y), reducedAxes, keep=true)` or equivalently sum `O_x`; `a_x=E_x(g_y)/E_x(N)`. | Typed [scalar](../../../glossary.md#term-scalar) one plus expand is [graph](../../../glossary.md#graph)-expressible for unresolved Shape. Empty means [NaN](../../../glossary.md#term-nan) and has no selected [derivative](../../../glossary.md#term-derivative). | `EXACT_CURRENT_COMPOSITION` on `N>0`; compiler. Empty boundary: `POLICY_DEFERRED`. |
| R3 | `PROD`; input, regular [finite](../../../glossary.md#term-finite) domain including zeros | `prefix=cumProd(x,axis,exclusive=true,reverse=false)` and `suffix=cumProd(x,axis,exclusive=true,reverse=true)`; `a_x=E_x(g_y)*prefix*suffix`. [Multi-axis reduction](../../../glossary.md#multi-axis-reduction) first permutes and reshapes selected axes into one [scan](../../../glossary.md#term-cumulative-scan) axis, then restores the input Shape. | Division by `x` is not exact at zeros. Current [model](../../../glossary.md#term-model) has only cumulative sum, so exact zero-safe products are absent. | `MISSING_GENERAL_PUBLIC_PRIMITIVE`; general cumulative-product scan; model then compiler. Special-value policy remains deferred. |
| R4 | reduction `MIN`,`MAX`; input | Route expanded `g_y` among [extrema](../../../glossary.md#saturation-pole-and-extremum) only after a tie/subgradient policy exists. | Forward value semantics fix NaN and [signed-zero](../../../glossary.md#nan-infinity-and-signed-zero) extrema but not [gradient](../../../glossary.md#term-gradient) sharing or first/last routing. | `POLICY_DEFERRED`; compiler policy. Arg-extrema are not a substitute because their first/last policy belongs to distinct index operations. |
| R5 | masked `SUM`; input/[mask](../../../glossary.md#term-boolean-mask) | `a_x=where(expand(mask),E_x(g_y),Z_x)`. Mask is [ND](../../../glossary.md#term-nd). | False positions must not multiply NaN or [infinity](../../../glossary.md#term-infinity); `where` provides selection. Mask broadcast to input was validated forward. | Input: `EXACT_CURRENT_COMPOSITION`; mask: `NON_DIFFERENTIABLE`; compiler; task 0018Q. |
| R6 | masked `MEAN`; input/mask, positive selected count | `selected=where(expand(mask),O_x,Z_x)`; `count=sum(selected,axis,keep=true)`; `a_x=where(mask,E_x(g_y)/count,Z_x)`. Mask is ND. | All-false count is zero and forward result is NaN; its derivative is unselected. | Input on positive count: `EXACT_CURRENT_COMPOSITION`; mask `NON_DIFFERENTIABLE`; all-false boundary `POLICY_DEFERRED`. |
| R7 | `LOG_SUM_EXP`; input, finite non-empty regular domain | `a_x=E_x(g_y)*exp(x-E_x(Y))`. | Empty/all-negative-infinity, positive-infinity ties, and NaNs do not have a selected derivative [distribution](../../../glossary.md#term-distribution). | `EXACT_CURRENT_COMPOSITION` on regular domain; boundary `POLICY_DEFERRED`; compiler. |
| R8 | [`VARIANCE`](../../../glossary.md#term-variance); input, valid finite domain | With `mu=mean(x,axes,keep=true)`, `N=sum(O_x,axes,keep=true)`, `a_x=E_x(g_y)*2*(x-mu)/(N-correction)`. | Forward requires positive denominator. Infinity/NaN derivative behavior is not selected. | `EXACT_CURRENT_COMPOSITION` on regular domain; boundary `POLICY_DEFERRED`; compiler. |
| R9 | `STANDARD_DEVIATION`; input, positive finite result | `a_x=E_x(g_y)*(x-mu)/((N-correction)*E_x(Y))`. | Zero [standard deviation](../../../glossary.md#term-standard-deviation) and exceptional values need policy. | `EXACT_CURRENT_COMPOSITION` where `Y>0`; boundary `POLICY_DEFERRED`; compiler. |
| R10 | `L1_NORM`,`L2_NORM`; input | Regular formulas are `E_x(g_y)*sign(x)` and `E_x(g_y)*x/E_x(Y)`. | [L1](../../../glossary.md#term-l1) at zero and [L2](../../../glossary.md#term-l2) at zero [norm](../../../glossary.md#term-norm), plus NaN/infinity cases, are unselected. | `POLICY_DEFERRED` for the complete accepted boundary; regular composition is current. |
| R11 | `ALL`,`ANY`,`ARG_MIN`,`ARG_MAX`; input | [BOOL](../../../glossary.md#data-type--datatype) reductions and [INT64](../../../glossary.md#data-type--datatype) index results do not [carry](../../../glossary.md#term-carry) [differentiable](../../../glossary.md#derivative-and-chain-rule) output roles. | [Arg-extrema tie policy](../../../glossary.md#arg-extrema-tie-policy) is value-selection policy, not a gradient policy. | `NON_DIFFERENTIABLE`; compiler stops propagation. |
| R12 | `CUM_SUM`; input | `a_x=cumSum(g_y,axis,exclusive,reverse=!reverse)`. Output positions remain in input order; exclusivity is unchanged. | Empty and dynamic axes preserve Shape and require no new indices. | `EXACT_CURRENT_COMPOSITION`; compiler; tasks 0016G–0016H. |
| R13 | [`SOFTMAX`](../../../glossary.md#softmax--log-softmax); input, finite non-empty [slice](../../../glossary.md#slice) | With saved/recomputed `Y`, `a_x=Y*(g_y-sum(g_y*Y,axis,keep=true))`. | `SoftmaxKind` explicitly leaves numerical edge-case and gradient policy open. | `EXACT_CURRENT_COMPOSITION` on regular slices; empty/NaN/infinity boundaries `POLICY_DEFERRED`; compiler. |
| R14 | `LOG_SOFTMAX`; input, finite non-empty slice | `a_x=g_y-exp(Y)*sum(g_y,axis,keep=true)`. | Same unselected numerical boundary as R13. | `EXACT_CURRENT_COMPOSITION` on regular slices; boundary `POLICY_DEFERRED`; compiler. |
| R15 | `LAYER_NORM`; input, no affine, finite slice | `mu=mean(x,A,keep=true)`, `r=rsqrt(mean((x-mu)^2,A,keep=true)+eps)`, `h=g_y`; `a_x=(r/N)*(N*h-sum(h)- (x-mu)*r*r*sum(h*(x-mu)))`. | `N` comes from expanded one and reduction. Forward rejects empty [normalized Shape](../../../glossary.md#normalized-shape); exceptional values remain policy boundaries. | `EXACT_CURRENT_COMPOSITION` regular; boundary `POLICY_DEFERRED`; compiler; task 0021. |
| R16 | affine `LAYER_NORM`; input/scale/[bias](../../../glossary.md#term-bias) | Use R15 with `h=g_y*scale`; `a_scale=S_scale(g_y*((x-mu)*r),prefixAxes)`; `a_bias=S_bias(g_y,prefixAxes)`. | Affine Shapes equal the normalized trailing Shape, so axes are fixed, not [binding](../../../glossary.md#term-binding)-dependent broadcast axes. | `EXACT_CURRENT_COMPOSITION` regular; boundary `POLICY_DEFERRED`; compiler. |
| R17 | `RMS_NORM`; input, optional scale | `r=rsqrt(mean(x*x,A,keep=true)+eps)`, `h=g_y` or `g_y*scale`; `a_x=r*h-x*(r^3/N)*sum(h*x,A,keep=true)`; `a_scale=S_scale(g_y*x*r,prefixAxes)`. | Empty normalized Shape is invalid forward. Zero/infinity/NaN boundaries remain unselected. | `EXACT_CURRENT_COMPOSITION` regular; boundary `POLICY_DEFERRED`; compiler; task 0021A. |
| R18 | [batch](../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-normalization [inference](../../../glossary.md#term-inference); input, scale, bias, running mean, running variance | `r=rsqrt(var+eps)`, [channel](../../../glossary.md#term-channel)-expanded. `a_x=g*scale*r`; `a_scale=sum_nonchannel(g*(x-mean)*r)`; `a_bias=sum_nonchannel(g)`; `a_mean=sum_nonchannel(-g*scale*r)`; `a_var=sum_nonchannel(-0.5*g*scale*(x-mean)*r^3)`. | Channel axis and all reduction axes are fixed; no `R` is needed. Negative radicand, zero denominator, infinity, and NaN require policy. | `EXACT_CURRENT_COMPOSITION` regular; boundary `POLICY_DEFERRED`; compiler; task 0021B. |
| R19 | batch-normalization training output slot 0; input/scale/bias | Use `mu=aux[3]`, `r=aux[4]`, channel count `N`; apply the R15 batch-axis formula with `h=g_0*scale`; scale/bias sums use fixed non-channel axes. | Saved values are same-[occurrence](../../../glossary.md#term-occurrence) slots, not recomputed statistics. | `EXACT_WITH_EXISTING_AUXILIARY_OUTPUT` regular; compiler [captures](../../../glossary.md#term-capture-meanings) slots 3/4; task 0021C. |
| R20 | batch-training next-[statistic](../../../glossary.md#term-statistic) slots 1/2; input/running mean/running variance | Slot 1 gives `a_runningMean=(1-momentum)*g_1` and adds `expand(momentum*g_1/N)` to `a_input`. Slot 2 gives `a_runningVariance=(1-momentum)*g_2` and adds the correction-one variance [VJP](../../../glossary.md#term-vector-jacobian-product-vjp) `expand(momentum*g_2)*2*(x-mu)/(N-1)` to `a_input`. Add both input contributions to R19 when slot 0 is also seeded. | Public outputs 1/2 may both [seed](../../../glossary.md#term-seed) cotangents. Count must be at least two for non-empty channels. Scale and bias receive contributions only from slot 0. | `EXACT_WITH_EXISTING_AUXILIARY_OUTPUT` regular; compiler; exact [producer](../../../glossary.md#term-producer) slots from `BatchNormTrainingResult` and helper. |
| R21 | batch-training saved slots 3/4 | They are floating compiler-capturable intermediates, not independent public [cotangent](../../../glossary.md#cotangent) roots. If a later compiler transformation consumes them, ordinary [chain rule](../../../glossary.md#term-chain-rule) applies to their defining mean/variance expressions. | Public [API](../../../glossary.md#term-application-programming-interface-api) intentionally omits them; producer [descriptors](../../../glossary.md#term-descriptor) retain them for R19. | No separate input-role classification: they are auxiliary forward values, not non-differentiable outputs or a new model [operation](../../../glossary.md#operation). |

### [Linear](../../../glossary.md#term-linear-meanings) algebra, attention, [convolution](../../../glossary.md#term-channels-first-convolution-layers), [pooling](../../../glossary.md#term-pooling), and [losses](../../../glossary.md#loss)

| ID | Kind and audited role | [Shape](../../../glossary.md#shape), values, and exact formula | Obligations and decision | Classification; gap; [owner](../../../glossary.md#term-owner); evidence |
|---|---|---|---|---|
| M1 | [`MATMUL`](../../../glossary.md#matrix-multiplication--matmul); left/right | Swap each operand's final two [axes](../../../glossary.md#term-axis): `a_left=R_left(matmul(g_y,transposeLast2(right)))`; `a_right=R_right(matmul(transposeLast2(left),g_y))`. [Rank-one](../../../glossary.md#axis-rank-scalar-vector-and-matrix) cases first insert/remove the semantic [vector](../../../glossary.md#term-vector) axes. | [Batch](../../../glossary.md#term-batch-machine-learning-and-hyperparameters) [Dimensions](../../../glossary.md#dimension) may bind as [singleton](../../../glossary.md#term-singleton) or result [extent](../../../glossary.md#term-extent), so static reduction axes are insufficient. | `MISSING_GENERAL_PUBLIC_PRIMITIVE`; [binding](../../../glossary.md#term-binding)-aware [sum-to-Shape](../../../glossary.md#sum-to-shape); [model](../../../glossary.md#term-model) then [compiler](../../../glossary.md#term-compiler); task 0019. |
| M2 | `linear` chain; input/[weight](../../../glossary.md#term-weight)/[bias](../../../glossary.md#term-bias) | Through exact `PERMUTE -> MATMUL`, `a_input=matmul(g_y,weight)`; reduce the statically known leading input axes from the inverse-permuted weight contribution; `a_bias` sums those same leading axes. | Weight has [rank](../../../glossary.md#term-rank) two, bias is rank one and exactly matches out-[features](../../../glossary.md#term-software-feature), so this chain has no broadcast batch axis with a deferred singleton-or-equal obligation. No `LINEAR` kind exists. | `EXACT_CURRENT_COMPOSITION`; compiler; task 0019D. |
| M3 | scaled-dot-product attention; value | With same-[occurrence](../../../glossary.md#term-occurrence) weights `W`, `a_value=R_value(matmul(transposeLast2(W),g_y))`. [Mask](../../../glossary.md#term-boolean-mask) is [ND](../../../glossary.md#term-nd). | Current [producer](../../../glossary.md#term-producer) has no weights output; recomputation cannot reproduce first-class all-masked and positive-[infinity](../../../glossary.md#term-infinity) semantics with current [softmax](../../../glossary.md#softmax--log-softmax)/mask vocabulary. | `MISSING_GENERAL_PUBLIC_PRIMITIVE`; attention result/producer with weights as a public generally useful output; model then compiler; task 0019E. |
| M4 | attention; query/key | `dW=matmul(g_y,transposeLast2(value))`; `dS=W*(dW-sum(dW*W,lastAxis,keep=true))`; exclusions already have zero `W`; `a_query=R_query(scale*matmul(dS,key))`; `a_key=R_key(scale*matmul(transposeLast2(dS),query))`. | Default scale depends on bound [embedding](../../../glossary.md#embedding) extent; producer attrs retain that obligation. Batch unbroadcast needs `R`. Causal/explicit masks are ND. | `MISSING_GENERAL_PUBLIC_PRIMITIVE`; weights output plus sum-to-Shape. Model then compiler. |
| M5 | attention special rows | Regular formulas are M3–M4. | [Derivative](../../../glossary.md#term-derivative) behavior for eligible [NaN](../../../glossary.md#term-nan) scores, positive-infinity score ties, all-masked rows, and exceptional values is not selected by the forward contract. | `POLICY_DEFERRED`; compiler policy. This does not weaken the weights-output proof for regular rows. |
| M6 | grouped [NCHW](../../../glossary.md#term-nchw) [`CONV2D`](../../../glossary.md#convolution--conv2d); input | `cols_g=matmul(transpose(weight_g),reshape(g_y_g))`; `a_input=fold2d(cols, input.shape, matching window)`, group-concatenated. Exact geometry includes [stride](../../../glossary.md#term-stride), symmetric [padding](../../../glossary.md#padding), [dilation](../../../glossary.md#term-dilation), [kernel](../../../glossary.md#kernel), and crop to input Shape. | Current `unfold2d`/`fold2d` require static [channel](../../../glossary.md#term-channel)/[spatial](../../../glossary.md#term-spatial-axes) extents while conv accepts unresolved extents. Their flattened rank-three column Shape also needs `outputHeight*outputWidth`, which current Dimension expressions cannot represent when both factors are unresolved. | `MISSING_GENERAL_PUBLIC_PRIMITIVE`; redesigned dynamic [target](../../../glossary.md#term-target)-Shape [window transform](../../../glossary.md#window-transform) family; model then compiler; task 0020. |
| M7 | `CONV2D`; weight | `X=unfold2d(input,matching window)`; group/batch matmuls compute `sum_N(g_y_g * X_g^T)`, then reshape to exact weight Shape. | Groups are static attrs; unresolved input channels/spatial extents currently block `unfold2d`, and current Shape algebra cannot always flatten two unresolved output spatial extents. | `MISSING_GENERAL_PUBLIC_PRIMITIVE`; same redesigned dynamic window family; model then compiler. |
| M8 | biased `CONV2D`; bias | `a_bias=sum(g_y,axes=[N,Hout,Wout])`, reshaped to bias Shape. | Fixed axes restore exact rank-one channel Shape. | `EXACT_CURRENT_COMPOSITION`; compiler. Exceptional arithmetic policy follows the forward/boundary task. |
| M9 | `AVERAGE_POOL2D`; input | Divide `g_y` by fixed `kernelHeight*kernelWidth`, expand each value to every logical kernel position, then `fold2d(...,input.shape,attrs)`; padded targets are cropped by fold geometry. | Count-padding and ceil-mode terminal windows are retained. Dynamic accepted inputs hit current window staticity. | `MISSING_GENERAL_PUBLIC_PRIMITIVE`; dynamic window family; model then compiler; task 0020A1. |
| M10 | `MAX_POOL2D`; input, regular [finite](../../../glossary.md#term-finite) selection | Materialize each logical kernel domain with typed [negative infinity](../../../glossary.md#nan-infinity-and-signed-zero) for excluded padding, reshape, recompute `argMax(FIRST_INDEX)`, make numeric one-hot by `where(oneHot,O,Z)`, multiply by expanded `g_y`, and fold/crop to input Shape. | Existing arg-max ordering matches NaN preference, [signed-zero](../../../glossary.md#nan-infinity-and-signed-zero) ordering, and first logical tie. Current `unfold2d` has positive-zero padding and static spatial requirements, so it cannot represent excluded/all-padding ceil windows generally. | `MISSING_GENERAL_PUBLIC_PRIMITIVE`; dynamic/configurable window family. No maximum-pool-index output is required; model then compiler; task 0020A. |
| M11 | max-pool NaN and undefined boundary | M10 exactly recomputes forward selection, but derivative of a selected NaN or other non-[differentiable](../../../glossary.md#derivative-and-chain-rule) exceptional value is not chosen. | Tie routing is already fixed by first logical kernel [sample](../../../glossary.md#term-sample-machine-learning-and-hyperparameters); it is not deferred. | `POLICY_DEFERRED` only for exceptional derivative points; compiler policy. |
| M12 | [mean-squared error](../../../glossary.md#mean-squared-error--mse); [prediction](../../../glossary.md#term-prediction)/target, reduction NONE/SUM | `d=prediction-target`; scale incoming [scalar](../../../glossary.md#term-scalar) or [elementwise](../../../glossary.md#term-elementwise) [cotangent](../../../glossary.md#cotangent) by reduction; `a_prediction=2*d*g`; `a_target=-a_prediction`. | Shapes are exact, with no [broadcasting](../../../glossary.md#broadcasting). | `EXACT_CURRENT_COMPOSITION` on regular values; compiler; task 0022. |
| M13 | [MSE](../../../glossary.md#mean-squared-error--mse) [MEAN](../../../glossary.md#term-mean) | M12 divided by exact logical [element](../../../glossary.md#term-element) count made by summing `O_prediction`; empty mean is NaN. | Empty and exceptional floating derivatives are unselected. | `EXACT_CURRENT_COMPOSITION` for non-empty regular domain; boundary `POLICY_DEFERRED`. |
| M14 | dense-target [categorical cross entropy](../../../glossary.md#categorical-cross-entropy); logits | For each sample, `p=softmax(logits,classAxis)`, `T=sum(target,classAxis,keep=true)`, and `a_logits=(p*T-target)*expandedReductionCotangent`. | `NONE`, `SUM`, and sample-count `MEAN` use exact Shape restoration/count. Positive-infinity, all-negative-infinity, NaN, and zero target-weight special cases need derivative policy. | `EXACT_CURRENT_COMPOSITION` regular; boundary `POLICY_DEFERRED`; compiler; task 0022A. |
| M15 | dense cross entropy; target | `a_target=-logSoftmax(logits,classAxis)*expandedReductionCotangent`. | The forward's exact-zero weighted contribution at target zero with infinite [log](../../../glossary.md#term-log) probability is not differentiable without policy. | `EXACT_CURRENT_COMPOSITION` regular; boundary `POLICY_DEFERRED`; compiler. |
| M16 | index-target cross entropy; logits/target | For non-ignored samples, `a_logits=(softmax(logits)-numericOneHot(target))*scale`; ignored samples select `Z_logits` with `where`. `MEAN` divides by a non-ignored count built from numeric `where`. Target is ND. | One-hot depth is the static class extent required by current forward construction; bounds and ignore filtering follow forward order. | Logits: `EXACT_CURRENT_COMPOSITION` regular; target: `NON_DIFFERENTIABLE`; all-ignored/empty/special boundary `POLICY_DEFERRED`; compiler; task 0022B. |

### [Layout](../../../glossary.md#layout), slicing, windows, indexing, scatter, and ordering

| ID | Kind and audited role | [Shape](../../../glossary.md#shape), values, and exact formula | Obligations and decision | Classification; gap; [owner](../../../glossary.md#term-owner); evidence |
|---|---|---|---|---|
| [L1](../../../glossary.md#term-l1) | [`CONTIGUOUS`](../../../glossary.md#term-contiguous); input | `a_x=g_y`; logical values and Shape are unchanged. | [Materialization](../../../glossary.md#materialization) is irrelevant to semantic adjoints. | `EXACT_CURRENT_COMPOSITION`; [compiler](../../../glossary.md#term-compiler); tasks 0017A–0017B. |
| [L2](../../../glossary.md#term-l2) | `RESHAPE`; input | `a_x=reshape(g_y,x.shape)`. | Exact [element](../../../glossary.md#term-element)-count obligation is already represented by the forward [target](../../../glossary.md#term-target) Shape. | `EXACT_CURRENT_COMPOSITION`; compiler. |
| L3 | `EXPAND`; input | `a_x=B_x(g_y)`. | Public expand accepts unequal aligned [Dimensions](../../../glossary.md#dimension) only when the input [extent](../../../glossary.md#term-extent) is a static [singleton](../../../glossary.md#term-singleton); equal unresolved Dimensions are not expanded. | `EXACT_CURRENT_COMPOSITION`; compiler. |
| L4 | `EXPAND_DIMS`,`SQUEEZE`; input | Apply the inverse [rank](../../../glossary.md#term-rank) edit on the recorded [axis](../../../glossary.md#term-axis). | Squeeze already proves singleton; dynamic unresolved singleton is rejected forward. | `EXACT_CURRENT_COMPOSITION`; compiler. |
| L5 | `PERMUTE` and [`transpose`](../../../glossary.md#term-transpose); input | `a_x=P^-1(g_y)`. | Transpose is the exact [rank-two](../../../glossary.md#axis-rank-scalar-vector-and-matrix) [permutation](../../../glossary.md#permutation) chain. | `EXACT_CURRENT_COMPOSITION`; compiler. |
| L6 | [`SLICE`](../../../glossary.md#slice); input, [identity](../../../glossary.md#term-identity-meanings) or unit positive-step static slice | Identity returns `g_y`. A step-one crop places `g_y` with zero [constant](../../../glossary.md#term-constant) [padding](../../../glossary.md#padding) before/after each selected axis. | Current selected slice axes are static. Multi-axis padding remains exact when every step is one. | `EXACT_CURRENT_COMPOSITION`; compiler. |
| L7 | signed or strided `SLICE`/`flip`; input | Place each `g_y` [coordinate](../../../glossary.md#axis-rank-scalar-vector-and-matrix) at recorded `start+k*step` in `Z_x`. | Non-zero slice steps make selected coordinates [injective](../../../glossary.md#term-injective), so this [adjoint](../../../glossary.md#term-adjoint) needs placement, not overlap addition. Current scatter shapes do not encode multi-axis signed affine coordinates; padding cannot create gaps or reverse. | `MISSING_GENERAL_PUBLIC_PRIMITIVE`; general target-Shape slice placement/update transformation, useful for sparse region edits; [model](../../../glossary.md#term-model) then compiler. |
| L8 | `SELECT`; input | For a static selected extent, `expandDims(g_y,axis)`, then zero-pad `index` before and `extent-index-1` after. For an unresolved selected extent, place the singleton result at the recorded non-negative coordinate in the exact input Shape. | Forward permits a non-negative index on an unresolved selected extent with deferred bounds, but current long-valued padding cannot express the [binding](../../../glossary.md#term-binding)-dependent suffix. | Static: `EXACT_CURRENT_COMPOSITION`. Unresolved selected extent: `MISSING_GENERAL_PUBLIC_PRIMITIVE`; target-Shape slice placement; model then compiler. |
| L9 | [`PAD`](../../../glossary.md#padding); input | Crop `g_y` by known `before` and original input Shape. Static selected extents use current `SLICE`. Padding constant is [scalar](../../../glossary.md#term-scalar) configuration, not a [Tensor](../../../glossary.md#tensor) role. | `PAD` accepts unresolved padded extents, but current `SLICE` rejects a selected unresolved Dimension. | `MISSING_GENERAL_PUBLIC_PRIMITIVE`; target-relative dynamic crop in the same general slice family; model then compiler. |
| L10 | `TILE`; input | Reshape `g_y` to interleave each repeat count and original Dimension, sum repeat axes, then reshape to `x.shape`. | Repeat counts are static attrs; original Dimensions may remain symbolic in target Shape. | `EXACT_CURRENT_COMPOSITION`; compiler. |
| L11 | `CONCAT`; one input role | Slice the output [cotangent](../../../glossary.md#cotangent) along the concat axis at the encounter-order prefix and input extent. | Static extents use current slice. Concat accepts unresolved selected extents, whose prefix/length cannot be stored by current long-valued `SliceAttrs`. | `MISSING_GENERAL_PUBLIC_PRIMITIVE`; target-relative dynamic crop; model then compiler. |
| L12 | `STACK`; one input role | `select(g_y,stackAxis,inputPosition)`. | Input position and new static stack extent are known. | `EXACT_CURRENT_COMPOSITION`; compiler. |
| L13 | `unstack` convenience | Each output is an independent `SELECT`; pad each incoming cotangent as L8 and add all contributions for the original input. | No shared unstack [producer](../../../glossary.md#term-producer) or kind exists. | `EXACT_CURRENT_COMPOSITION`; compiler; tasks 0017K–0017L and 0018O. |
| L14 | `UNFOLD_AXIS`; input | The exact adjoint is overlap-add `FOLD_AXIS(g_y,axis,inputExtent,step)`. | No public Tensor method constructs `FOLD_AXIS`. Current unfold already requires its selected extent static, so the retained `long outputSize` is sufficient; unaffected unresolved Dimensions flow from `g_y`. | `MISSING_GENERAL_PUBLIC_PRIMITIVE`; restore the existing general-axis fold as a public Tensor transformation; model then compiler; tasks 0017M–0017N, 0018R. |
| L15 | `UNFOLD2D`; input | `a_x=fold2d(g_y,x.shape,window)`. | Current forward already requires static channels/[spatial](../../../glossary.md#term-spatial-axes) extents; [batch](../../../glossary.md#term-batch-machine-learning-and-hyperparameters) may be unresolved and exact reference is retained. | `EXACT_CURRENT_COMPOSITION`; compiler. |
| L16 | `FOLD2D`; column input | `a_columns=unfold2d(g_y,window)`. | Forward fold validates static output [channel](../../../glossary.md#term-channel)/spatial geometry and exact column compatibility. | `EXACT_CURRENT_COMPOSITION`; compiler. |
| I1 | [`GATHER`](../../../glossary.md#gather); data/indices | Data adjoint is rank-changing axis scatter-add of `g_y` through the complete indices Shape into `Z_data`. For a positive static gathered extent, current `oneHot(indices,extent)`, `where`, axis edits, and reduction express it; a statically zero valid domain returns `Z_data`. Indices are [ND](../../../glossary.md#term-nd). | `SCATTER_ELEMENTS` requires same-rank indices/updates. `SCATTER_ND` would require unavailable coordinate grids, while one-hot requires positive static depth; Gather accepts an unresolved gathered extent. | Static gathered extent: `EXACT_CURRENT_COMPOSITION`. General dynamic contract: `MISSING_GENERAL_PUBLIC_PRIMITIVE`; Gather-compatible axis scatter-add. Indices: `NON_DIFFERENTIABLE`; model then compiler; tasks 0018C–0018D. |
| I2 | [`embedding`](../../../glossary.md#embedding) chain | Exactly I1 with `GATHER(axis=0)`; indices are ND. | No embedding kind or separate rule exists. | Data: the same static `EXACT_CURRENT_COMPOSITION`/dynamic `MISSING_GENERAL_PUBLIC_PRIMITIVE` split as I1; indices `NON_DIFFERENTIABLE`; compiler. |
| I3 | `GATHER_ELEMENTS`; data/indices | `a_data=scatterElements(Z_data,indices,g_y,axis,ADD)`. Indices are ND. | ADD is required because repeated indices accumulate. Shapes, bounds, and duplicate targets match the forward [occurrence](../../../glossary.md#term-occurrence). | Data: `EXACT_CURRENT_COMPOSITION`; indices `NON_DIFFERENTIABLE`; compiler; tasks 0018O and current scatter contracts. |
| I4 | `GATHER_ND`; data/indices | `a_data=scatterNd(Z_data,indices,g_y,batchDimensions,ADD)`. Indices are ND. | [Tuple depth](../../../glossary.md#term-tuple-depth), batch prefix, suffix Shape, bounds, and repeated-index accumulation match exactly. | Data: `EXACT_CURRENT_COMPOSITION`; indices `NON_DIFFERENTIABLE`; compiler. |
| I5 | `SCATTER_ELEMENTS`/`SCATTER_ND`, reduction `NONE`; data | With unique valid targets, replace addressed positions of `g_y` by matching zero updates using `NONE`; untouched positions retain `g_y`. | Duplicate targets are invalid forward, not a [derivative](../../../glossary.md#term-derivative) policy. | `EXACT_CURRENT_COMPOSITION`; compiler. |
| I6 | scatter `NONE`; updates/indices | Gather `g_y` with the corresponding [Gather Elements](../../../glossary.md#gather-elements) or [Gather-ND](../../../glossary.md#gather-nd) geometry. Indices are ND. | Bounds and unique-target [validation](../../../glossary.md#term-validation) are inherited from the forward occurrence. | Updates: `EXACT_CURRENT_COMPOSITION`; indices `NON_DIFFERENTIABLE`; compiler. |
| I7 | scatter reduction `ADD`; data/updates | `a_data=g_y`; updates gather `g_y` as I6. Indices are ND. | Every duplicate update receives the same target cotangent; accumulation occurred only in forward. | Data/updates: `EXACT_CURRENT_COMPOSITION`; indices `NON_DIFFERENTIABLE`; compiler. |
| I8 | scatter reduction `MUL`; data/updates | On non-zero [finite](../../../glossary.md#term-finite) groups, data receives `g_y*product(updates)` and each update receives gathered `g_y*data*product(other updates)`. | Current `ScatterReduction` explicitly leaves numerical edge behavior unspecified; zeros, [infinities](../../../glossary.md#nan-infinity-and-signed-zero), NaNs, and duplicate-group partial products lack an exact selected contract. | `POLICY_DEFERRED`; first select forward numerical and derivative boundary policy. No backward kind is preselected. |
| I9 | scatter reduction `MIN`,`MAX`; data/updates | Route cotangents only after selecting treatment of ties among base and duplicate updates. | Current scatter reduction has no tie, [signed-zero](../../../glossary.md#nan-infinity-and-signed-zero), [NaN](../../../glossary.md#term-nan), or derivative-sharing policy. | `POLICY_DEFERRED`; compiler/model semantic-policy task only if the forward contract is first completed. |
| I10 | `ONE_HOT`; indices | [BOOL](../../../glossary.md#data-type--datatype) output and integral input are outside the [differentiable](../../../glossary.md#derivative-and-chain-rule) domain. | It remains usable as a condition in numeric `where`. | `NON_DIFFERENTIABLE`; compiler. |
| O1 | stable `SORT`; input | Recompute exact stable `argsort` with matching axis/direction; `a_x=scatterElements(Z_x,indices,g_y,axis,NONE)`. | Indices form a permutation, so targets are unique. Exact representations, signed-zero order, and NaN-last stability match. | `EXACT_CURRENT_COMPOSITION` on regions with stable ordering; equal-value crossing/NaN derivative boundary `POLICY_DEFERRED`; compiler; task 0019C. |
| O2 | `ARGSORT`; input | [INT64](../../../glossary.md#data-type--datatype) result has no differentiable output role. | Floating input does not make an index result differentiable. | `NON_DIFFERENTIABLE`. |
| O3 | `TOP_K`; input from values slot 0 | Use existing slot-1 indices: `a_x=scatterElements(Z_x,indices,g_values,axis,NONE)`. | Selected indices are unique. `k=0` yields unchanged zero base. Index output is ND. | Values: `EXACT_WITH_EXISTING_AUXILIARY_OUTPUT`; indices `NON_DIFFERENTIABLE`; compiler; task 0019C1. |
| O4 | sort/top-K selection boundaries | Regular routing is O1/O3. | Membership changes at equal cutoffs and NaN/discontinuity points need a derivative convention even though forward stability fixes output order. | `POLICY_DEFERRED`; compiler policy. |
| O5 | `INITIAL_STATE`, [dropout](../../../glossary.md#dropout--inverted-dropout) state input/output | INT64 [graph RNG state](../../../glossary.md#graph-rng-state--graphrngstate) is explicit but [opaque](../../../glossary.md#term-opaque) and non-differentiable. | [State threading](../../../glossary.md#state-threading) must not be sampled, inferred, or differentiated. | `NON_DIFFERENTIABLE`; compiler; task 0019B. |
| O6 | `DROPOUT`; floating input | With `mask=aux[1]`, `a_x=where(mask,g_y/(1-probability),Z_x)`. Do not infer from output, resample, or advance state. | Selection preserves dropped zero without `0*NaN`; probability is scalar configuration; state roles are ND. | `EXACT_WITH_EXISTING_AUXILIARY_OUTPUT`; compiler [captures](../../../glossary.md#term-capture-meanings) slot 1; task 0019B1. |

## Required decision probes

### Broadcast reversal and dynamic [Shapes](../../../glossary.md#shape)

[Binding](../../../glossary.md#term-binding)-dependent unbroadcasting is a proven general gap for [MATMUL](../../../glossary.md#matrix-multiplication--matmul) and attention, whose helpers
explicitly accept an unresolved [batch Dimension](../../../glossary.md#batch-dimension) against a static non-[singleton](../../../glossary.md#term-singleton) and defer whether
it binds to one or to that result [extent](../../../glossary.md#term-extent). Ordinary [binary](../../../glossary.md#term-binary-descriptor-and-metadata) arithmetic, `where`, and `EXPAND` do not
have that gap: their current Shape contracts accept unequal aligned [Dimensions](../../../glossary.md#dimension) only when the
expanding input is a statically known singleton. [Linear](../../../glossary.md#term-linear-meanings) uses a [rank-two](../../../glossary.md#axis-rank-scalar-vector-and-matrix) [weight](../../../glossary.md#term-weight) and exact [rank-one](../../../glossary.md#axis-rank-scalar-vector-and-matrix)
[bias](../../../glossary.md#term-bias), so its reduction [axes](../../../glossary.md#term-axis) are also fixed. The audit therefore selects one binding-aware
[sum-to-Shape](../../../glossary.md#sum-to-shape) primitive and no family-specific unbroadcast [operation](../../../glossary.md#operation).

Typed [scalar](../../../glossary.md#term-scalar) zero/one leaves plus `expand(targetShape)` solve general [graph](../../../glossary.md#graph) [constants](../../../glossary.md#term-constant) for unresolved
Shapes. This closes scatter bases, numeric [masks](../../../glossary.md#term-boolean-mask), counts, empty updates, and [padding](../../../glossary.md#padding) [identities](../../../glossary.md#term-identity-meanings)
without reading a template value. [Eager](../../../glossary.md#term-eager-evaluation) `zerosLike`/`onesLike` remain fully-static conveniences.

### [Gather](../../../glossary.md#gather) and scatter

[Gather Elements](../../../glossary.md#gather-elements) maps exactly to [Scatter Elements](../../../glossary.md#scatter-elements) ADD, and [Gather-ND](../../../glossary.md#gather-nd) maps exactly to [Scatter-ND](../../../glossary.md#scatter-nd)
ADD, including duplicate accumulation and [batch](../../../glossary.md#term-batch-machine-learning-and-hyperparameters)/suffix geometry. [Rank](../../../glossary.md#term-rank)-changing Gather with a
positive static gathered [extent](../../../glossary.md#term-extent) can also compose through current one-hot selection and reduction;
a statically zero valid domain returns the zero base. The full Gather contract still does not:
one-hot requires positive static depth, neither current scatter accepts its updates/indices [Shape](../../../glossary.md#shape)
relation, and constructing Scatter-ND [coordinate](../../../glossary.md#axis-rank-scalar-vector-and-matrix) tuples would require absent dynamic ranges and
grids. One general Gather-compatible [axis](../../../glossary.md#term-axis) scatter-add primitive is therefore selected.
Replacement and ADD scatter adjoints compose
today; MUL and [extrema](../../../glossary.md#saturation-pole-and-extremum) reductions remain policy-deferred because the current forward reduction
contract omits the necessary numerical/tie semantics.

### [Slice](../../../glossary.md#slice), select, and windows

Select uses [axis](../../../glossary.md#term-axis) insertion plus zero [padding](../../../glossary.md#padding). Unit-step static slices use padding. Signed/strided
slices require a general [target](../../../glossary.md#term-target)-[Shape](../../../glossary.md#shape) slice placement/update transformation. Non-zero slice steps
are [injective](../../../glossary.md#term-injective), so the [matrix](../../../glossary.md#term-matrix-meanings) does not require an additive overlap mode. [PAD](../../../glossary.md#padding) and CONCAT accept
unresolved selected extents that current long-valued `SliceAttrs` cannot crop, so the same slice
follow-up must include target-relative dynamic crop semantics.

Public general-axis `foldAxis` must be restored. Its retained `long outputSize` is sufficient for
the current unfold [adjoint](../../../glossary.md#term-adjoint) because unfold requires that selected [extent](../../../glossary.md#term-extent) static; unrelated dynamic
[Dimensions](../../../glossary.md#dimension) flow through the input. Existing 2D window methods are mutually adjoint for their
static accepted contract, but [convolution](../../../glossary.md#term-channels-first-convolution-layers) and [pooling](../../../glossary.md#term-pooling) accept unresolved [spatial](../../../glossary.md#term-spatial-axes) extents that
current `unfold2d`/`fold2d` reject. Maximum pooling additionally needs generally useful window
[materialization](../../../glossary.md#materialization) with an exact excluded-padding fill rather than conceptual positive-zero padding,
including literal ceil-grid terminal windows. The current [canonical](../../../glossary.md#term-canonical-form) [rank](../../../glossary.md#term-rank)-three column Shape
flattens `outputHeight*outputWidth`; current Dimension expressions cannot encode that product when
both spatial factors are unresolved. The follow-up must therefore choose either sufficient Shape
algebra or a non-flattened dynamic window result/target contract instead of merely relaxing the
current helper's staticity check. One cohesive public foldAxis and dynamic/configurable 2D
window-transform follow-up covers these proven gaps. No convolution-[transpose](../../../glossary.md#term-transpose),
convolution-backward, pooling-backward, or maximum-pool-index kind is selected.

### [Convolution](../../../glossary.md#term-channels-first-convolution-layers) and [pooling](../../../glossary.md#term-pooling)

Grouped convolution input and [weight](../../../glossary.md#term-weight) formulas are [im2col](../../../glossary.md#term-im2col)/matmul/[col2im](../../../glossary.md#term-col2im) compositions. Groups only
[partition](../../../glossary.md#partition) known [axes](../../../glossary.md#term-axis); they do not justify another semantic kind. The sole general representation
gap is dynamic window construction and [target](../../../glossary.md#term-target)-[Shape](../../../glossary.md#shape) fold/crop. Average pool uses fixed divisor
expansion plus fold. Max pool recomputes the exact first logical selection with existing arg-max
ordering over negative-[infinity](../../../glossary.md#term-infinity)-padded windows. Performance and [fusion](../../../glossary.md#term-fusion) are irrelevant to these
classifications.

### [Extrema](../../../glossary.md#saturation-pole-and-extremum), special values, [normalization](../../../glossary.md#term-normalization), attention, and [losses](../../../glossary.md#loss)

Extrema ties, clamp endpoints, [absolute value](../../../glossary.md#norms-elementary-functions-and-probability-distributions)/[ReLU](../../../glossary.md#term-relu) at zero, discontinuous [unary](../../../glossary.md#term-unary) operations,
scatter extrema, ordering cutoffs, and undefined exceptional points remain `POLICY_DEFERRED`.
Regular formulas do not choose those policies.

[Softmax](../../../glossary.md#softmax--log-softmax), [log-softmax](../../../glossary.md#softmax--log-softmax), layer/RMS/[batch normalization](../../../glossary.md#batch-normalization--batch-normalization-inference-and-training), attention, and losses have exact regular-domain
formulas. Their documented [NaN](../../../glossary.md#term-nan), [infinity](../../../glossary.md#term-infinity), empty/all-masked, correction, [epsilon](../../../glossary.md#epsilon), ignore-index, and
denominator boundaries remain visible in the [matrix](../../../glossary.md#term-matrix-meanings). [Batch](../../../glossary.md#term-batch-machine-learning-and-hyperparameters) training reuses exact saved slots 3/4;
[dropout](../../../glossary.md#dropout--inverted-dropout) reuses [mask](../../../glossary.md#term-boolean-mask) slot 1; top-K reuses index slot 1. Attention is different: its one-output
[producer](../../../glossary.md#term-producer) exposes neither [weights](../../../glossary.md#term-weight) nor a current composition that reproduces its first-class masked
special semantics. A generally useful public values-and-weights attention result is selected.

### Randomness and shared outputs

[Dropout](../../../glossary.md#dropout--inverted-dropout) never resamples, infers a [mask](../../../glossary.md#term-boolean-mask) from zero output, or differentiates state. Its exact mask is
the same [producer](../../../glossary.md#term-producer)'s slot 1. [Batch](../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-training saved statistics and top-K indices likewise retain
producer [identity](../../../glossary.md#term-identity-meanings). [Compiler](../../../glossary.md#term-compiler) later owns [capture](../../../glossary.md#term-capture-meanings) and [lifetime](../../../glossary.md#term-lifetime); the [model](../../../glossary.md#term-model) producer contract needs no
redesign.

## Minimum follow-up queue

The completed [matrix](../../../glossary.md#term-matrix-meanings) selects six Draft [model](../../../glossary.md#term-model) follow-ups before task 0024:

1. [binding](../../../glossary.md#term-binding)-aware [sum-to-Shape](../../../glossary.md#sum-to-shape);
2. [Gather](../../../glossary.md#gather)-compatible [axis](../../../glossary.md#term-axis) scatter-add;
3. signed [slice](../../../glossary.md#slice) placement plus [target](../../../glossary.md#term-target)-relative dynamic crop;
4. public general-axis fold plus dynamic/configurable 2D window transforms;
5. cumulative-product [scan](../../../glossary.md#term-cumulative-scan); and
6. attention output plus same-[occurrence](../../../glossary.md#term-occurrence) [weights](../../../glossary.md#term-weight).

These are general public [tensor](../../../glossary.md#tensor) capabilities useful outside [autograd](../../../glossary.md#autograd). Each rejects a narrower
[operation](../../../glossary.md#operation)-specific backward spelling. No `GENUINELY_NON_EXPRESSIBLE_SEMANTIC_GAP` survived the
general-primitive test, so no [compiler](../../../glossary.md#term-compiler)-only semantic row is added. Compiler traversal, saved-value
[lifetime](../../../glossary.md#term-lifetime), accumulation, optimization, and [publication](../../../glossary.md#term-publication) remain later compiler [planning](../../../glossary.md#term-planning).

## Boundaries and no-change conclusions

- Tensor, Compile, Runtime, and Training API references do not change: this artifact plans future
  work and implements no API, compiler, runtime, or training behavior.<span data-glossary-navigation="true"> ([Tensor glossary](../../../glossary.md#tensor); [Compile glossary](../../../glossary.md#compile); [Runtime glossary](../../../glossary.md#term-runtime); [API glossary](../../../glossary.md#term-application-programming-interface-api); [artifact glossary](../../../glossary.md#term-artifact); [compiler glossary](../../../glossary.md#term-compiler))</span>
- The glossary does not change: every reusable term already exists or is defined locally as audit
  notation; proposed follow-up names are planning labels, not stabilized vocabulary.<span data-glossary-navigation="true"> ([planning glossary](../../../glossary.md#term-planning))</span>
- [Javadoc](../../../glossary.md#term-javadoc) and Java source/tests do not change because no executable or public Java contract changes.
- Architecture documents, ADRs, and architecture tests do not change because module ownership and
  dependencies remain exactly as contracted.<span data-glossary-navigation="true"> ([module glossary](../../../glossary.md#term-module); [ownership glossary](../../../glossary.md#term-ownership-meanings))</span>
- Gradle, other modules, backend conformance, and integration tests do not change because this is
  planning-only analysis with no build, backend, or end-to-end behavior.<span data-glossary-navigation="true"> ([Gradle glossary](../../../glossary.md#term-gradle); [backend glossary](../../../glossary.md#backend); [conformance glossary](../../../glossary.md#term-conformance))</span>

## Related [planning](../../../glossary.md#term-planning)

- [Task 0023 execution record](tasks/0023-adjoint-expressibility-audit.md)
- [Model capability baseline](capabilities.md)<span data-glossary-navigation="true"> ([Model glossary](../../../glossary.md#term-model))</span>
- [Model master plan](master-plan.md)
- [Implementation roadmap](../../roadmap.md)
- [Planning guide](../../planning-guide.md)
