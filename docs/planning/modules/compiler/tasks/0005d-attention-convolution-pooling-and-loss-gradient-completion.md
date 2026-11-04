# Task 0005D: Attention, [Convolution](../../../../glossary.md#term-channels-first-convolution-layers), [Pooling](../../../../glossary.md#term-pooling), and [Loss](../../../../glossary.md#loss) [Gradient](../../../../glossary.md#term-gradient) Completion

> **Editorial reading key — not part of the historical record.**
>
> These rule owners consume original operand roles and canonical auxiliary outputs to construct
> attention, convolution, pooling and loss derivatives. In particular, saved attention weights
> belong to the same producer occurrence; recomputing plausible weights is not the recorded
> contract. Shapes and generated operations cross Compiler verification and capability boundaries
> as descriptions. The historical supported/fail-closed matrix below is its checkpoint inventory,
> not a statement of every current backend or derivative capability.
>
> Current reading: [pre-capture differentiation and functional requests](../../../../reference/compiler/autograd.md#current-package-private-pre-capture-autograd); [capability queries and hard-eligible owner selection](../../../../reference/compiler/planning.md#current-operation-capability-contracts).

## Status

Complete.

## Goal

Complete the remaining structured machine-learning part of the current first-order [gradient](../../../../glossary.md#term-gradient)
inventory inside `modules/compiler`. Preserve and verify the implemented [`MATMUL`](../../../../glossary.md#matrix-multiplication--matmul) rules and public
`linear(...)` composition, then add [fail-closed](../../../../glossary.md#term-fail-closed) [compiler](../../../../glossary.md#term-compiler)-owned reverse formulas for [scaled
dot-product attention](../../../../glossary.md#scaled-dot-product-attention), grouped [NCHW](../../../../glossary.md#term-nchw) [convolution](../../../../glossary.md#term-channels-first-convolution-layers), NCHW max/average [pooling](../../../../glossary.md#term-pooling), [mean-squared error](../../../../glossary.md#mean-squared-error--mse), and
dense- and index-[target](../../../../glossary.md#term-target) categorical cross-entropy with logits.

Every [derivative](../../../../glossary.md#term-derivative) remains an ordinary public [`Tensor`](../../../../glossary.md#tensor) expression. Use [canonical](../../../../glossary.md#term-canonical-form) same-[occurrence](../../../../glossary.md#term-occurrence)
attention [weights](../../../../glossary.md#term-weight) where they exist, current public window transforms for convolution/pooling,
retained forward obligations for symbolic formulas, and exact [Shape](../../../../glossary.md#shape)/data-type [normalization](../../../../glossary.md#term-normalization) for
every selected floating input. Unsupported [signatures](../../../../glossary.md#term-signature) fail during [preflight](../../../../glossary.md#term-preflight) before any generated
Tensor or derivative [constant](../../../../glossary.md#term-constant) is allocated.

This task adds no [model](../../../../glossary.md#term-model) operations, hidden outputs, second gradient algebra, physical saved values,
[runtime](../../../../glossary.md#term-runtime) tape, Tensor gradient state, [backend](../../../../glossary.md#backend) behavior, execution, or higher-order requests.

## [Scope](../../../../glossary.md#term-scope)

### Source-backed [operation](../../../../glossary.md#operation) and role inventory

Inventory current source at implementation time and keep this [matrix](../../../../glossary.md#term-matrix-meanings) exact. `D` is a floating
input role with an implemented contribution, [`ND`](../../../../glossary.md#term-nd) is intentionally non-[differentiable](../../../../glossary.md#derivative-and-chain-rule), and `FC`
is structurally [gradient](../../../../glossary.md#term-gradient)-eligible but deliberately [fail-closed](../../../../glossary.md#term-fail-closed) because the exact rule is not
expressible from the [signature](../../../../glossary.md#term-signature)'s [canonical](../../../../glossary.md#term-canonical-form) outputs and current public [Tensor](../../../../glossary.md#tensor) algebra.

| [Occurrence](../../../../glossary.md#term-occurrence) | Output slot | Ordered roles | Disposition |
|---|---:|---|---|
| [`MATMUL`](../../../../glossary.md#matrix-multiplication--matmul) | 0 | left, right | floating `D`, `D`; signed-integral `ND`, `ND` |
| public `linear(weight[, bias])` | visible composition | input, [weight](../../../../glossary.md#term-weight), optional [bias](../../../../glossary.md#term-bias) | verify `PERMUTE -> MATMUL -> optional ADD`; no [`LINEAR`](../../../../glossary.md#term-linear-meanings) kind/rule |
| one-output attention | 0 output | query, key, value, optional [BOOL](../../../../glossary.md#data-type--datatype) [mask](../../../../glossary.md#term-boolean-mask) | query/key/value `FC`; mask `ND` |
| two-output attention | 0 output | query, key, value, optional BOOL mask | query/key/value `D`; mask `ND` |
| two-output attention | 1 weights | query, key, value, optional BOOL mask | query/key `D`; value/mask `ND` |
| [`CONV2D`](../../../../glossary.md#convolution--conv2d) without/with bias | 0 | input, weight, optional bias | every floating data role `D` |
| `MAX_POOL2D` | 0 | input | input `D`; attrs are configuration |
| `AVERAGE_POOL2D` | 0 | input | input `D`; attrs are configuration |
| `MEAN_SQUARED_ERROR` | 0 | [prediction](../../../../glossary.md#term-prediction), [target](../../../../glossary.md#term-target) | both `D` for `NONE`, `SUM`, [`MEAN`](../../../../glossary.md#term-mean) |
| dense categorical cross-entropy with logits | 0 | logits, [dense target](../../../../glossary.md#dense-target) | both `D` for `NONE`, `SUM`, `MEAN` |
| index categorical cross-entropy with logits | 0 | logits, [INT32](../../../../glossary.md#data-type--datatype)/[INT64](../../../../glossary.md#data-type--datatype) target | logits `D`, target `ND` for all reductions, subject to static class depth |

One-output attention [producers](../../../../glossary.md#term-producer) literally have no canonical weights slot. Reconstructing a sibling,
re-running attention, or rebuilding masked [softmax](../../../../glossary.md#softmax--log-softmax) would violate canonical-output and exceptional-
value contracts. Any selected path through one-output attention therefore fails closed. The
explicit `scaledDotProductAttentionWithWeights(...)` forms are the differentiable signatures.

Index [loss](../../../../glossary.md#loss) is representable only when [preflight](../../../../glossary.md#term-preflight) observes a statically known positive class [extent](../../../../glossary.md#term-extent).
`Tensor.oneHot(long)` requires that depth. Dynamic class extent, and static zero class extent
accepted by forward empty/all-ignored semantics, [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation) before construction. Do not add a
range Tensor, materialized index [constant](../../../../glossary.md#term-constant), or integral selection operation.

### Notation and shared construction

For input `x`, [occurrence](../../../../glossary.md#term-occurrence) output `y`, and incoming [cotangent](../../../../glossary.md#cotangent) `g`:

- `R_x(t)` is existing exact [cotangent normalization](../../../../glossary.md#cotangent-normalization): `sumToShape(x.shape())` where required,
  followed by cast to `x.dataType()` where required.
- `Z_x`/`O_x` are request-local exact typed positive-zero/positive-one logical splats.
- Floating operands are cast to the occurrence output type before formula arithmetic.
- `T(t)` swaps the final two [axes](../../../../glossary.md#term-axis); `Σ_A(t, keep)` is ordinary sum over axes `A`.
- `P(a,b) = where(a == +0, +0, a*b)` is the selection-safe product used where exact-zero `a`
  semantically excludes `b` before arithmetic.
- A logical count is formed by expanding one exact typed one over the exact domain Shape and
  summing it, never by host <span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [host glossary](../../../../glossary.md#term-host))</span>`long` multiplication/conversion.
- Reshapes/permutations use ordinary public operations and exact symbolic [`Dimension`](../../../../glossary.md#dimension) expressions
  selected or inferred from the forward occurrence.

Keep helpers family-owned and package-private. Add no [registry](../../../../glossary.md#term-registry), [derivative](../../../../glossary.md#term-derivative) [DSL](../../../../glossary.md#term-dsl), broad builder,
generic manager/service/processor, utility package, or public [compiler](../../../../glossary.md#term-compiler) surface.

### Existing [`MATMUL`](../../../../glossary.md#matrix-multiplication--matmul) and public `linear(...)`

Preserve the exact task-0004A [matrix](../../../../glossary.md#term-matrix-meanings):

| Left [rank](../../../../glossary.md#term-rank) | Right rank | Left contribution before [normalization](../../../../glossary.md#term-normalization) | Right contribution before normalization |
|---|---|---|---|
| 1 | 1 | `g * right` | `g * left` |
| 1 | at least 2 | squeeze temporary row of `expandDims(g) @ T(right)` | `expandDims(left,1) @ expandDims(g,-2)` |
| at least 2 | 1 | `expandDims(g,-1) @ expandDims(right,0)` | `T(left) @ g` |
| at least 2 | at least 2 | `g @ T(right)` | `T(left) @ g` |

Verify [vector](../../../../glossary.md#term-vector)/vector, vector/matrix, matrix/vector, batched/broadcast forms, mixed floating types,
exact restoration, and signed-integral rejection.

Verify `input.linear(weight, bias)` through its actual [weight](../../../../glossary.md#term-weight) [transpose](../../../../glossary.md#term-transpose), `MATMUL`, and `ADD`, into
input, weight, and [bias](../../../../glossary.md#term-bias). Do not add a [`LINEAR`](../../../../glossary.md#term-linear-meanings) row, kind, [facade](../../../../glossary.md#term-facade), or special rule.

### [Scaled dot-product attention](../../../../glossary.md#scaled-dot-product-attention)

For a two-output [occurrence](../../../../glossary.md#term-occurrence):

```text
Q=query; K=key; V=value
W=producer.output(1)                 [...batch,L,S]
O=producer.output(0)                 [...batch,L,Ev]
```

For absent scale, expand one typed logical one over the exact query [embedding](../../../../glossary.md#embedding) `E`, sum to its
[scalar](../../../../glossary.md#term-scalar) logical count, and apply `rsqrt`, yielding `1/sqrt(E)` without reading a [binding](../../../../glossary.md#term-binding). A present
scale retains its exact typed [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue).

Define:

```text
softmaxPullback(W,U)
  = P(W, U - sum(P(W,U), axis=-1, keepDimensions=true))
```

For output slot zero with [cotangent](../../../../glossary.md#cotangent) `gO`:

```text
U  = gO @ T(V)
dS = softmaxPullback(W,U)
dV = R_V(selectionSafeContractOverL(W,gO))
dQ = R_Q(scale * selectionSafeContractOverS(dS,K))
dK = R_K(scale * selectionSafeContractOverL(dS,Q))
```

For [weights](../../../../glossary.md#term-weight) slot one with cotangent `gW`:

```text
dS = softmaxPullback(W,gW)
dQ = R_Q(scale * selectionSafeContractOverS(dS,K))
dK = R_K(scale * selectionSafeContractOverL(dS,Q))
```

Selection-safe contractions are ordinary expand/`P`/sum/reshape compositions equivalent to the
named [matrix](../../../../glossary.md#term-matrix-meanings) contractions. Do not use raw `0*NaN` products there. `U` may be ordinary [`MATMUL`](../../../../glossary.md#matrix-multiplication--matmul);
excluded entries are removed by the subsequent saved-weight pullback.

If both outputs contribute, traverse [canonical](../../../../glossary.md#term-canonical-form) wrappers in stable output-slot order and use
existing [identity](../../../../glossary.md#term-identity-meanings) accumulation. Slot zero alone contributes to value. Policy:

- explicit/causal [masks](../../../../glossary.md#term-boolean-mask) are [`ND`](../../../../glossary.md#term-nd); exact saved `W` alone records eligibility;
- excluded, all-masked, and all-negative-[infinity](../../../../glossary.md#term-infinity) rows contribute [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero);
- positive-infinity ties use exact saved split weights and the saved-output [Jacobian](../../../../glossary.md#term-jacobian);
- eligible [NaN](../../../../glossary.md#term-nan) weights follow raw saved-output arithmetic and propagate NaN;
- exact-zero saved weight selects zero even when its score was eligible [negative infinity](../../../../glossary.md#nan-infinity-and-signed-zero);
- [finite](../../../../glossary.md#term-finite) [rounding](../../../../glossary.md#term-rounding)/[reassociation](../../../../glossary.md#term-reassociation) remains that of emitted public operations.

### Grouped [NCHW](../../../../glossary.md#term-nchw) [convolution](../../../../glossary.md#term-channels-first-convolution-layers)

For input `X:[N,Cin,H,W]`, [weight](../../../../glossary.md#term-weight) `F:[Cout,CinPerGroup,kH,kW]`, [cotangent](../../../../glossary.md#cotangent)
`Gout:[N,Cout,Hout,Wout]`, positive static `groups`, and exact convolution-matching
`Window2dAttrs` with `ceilMode=false`:

```text
Xcol = unfold2d(X,window)                       [N,Cin*kH*kW,L]
Og   = Cout/groups
Kg   = CinPerGroup*kH*kW
Xg   = reshape/permute(Xcol)                    [groups,N,Kg,L]
Fg   = reshape(F)                               [groups,Og,Kg]
Gg   = reshape/permute(Gout)                    [groups,N,Og,L]

dXg  = expandDims(T(Fg),batch=N) @ Gg           [groups,N,Kg,L]
dX   = R_X(fold2d(inverseReshapePermute(dXg),X.shape(),window))

dFg  = sum(Gg @ T(Xg),axis=N)                   [groups,Og,Kg]
dF   = R_F(reshape(dFg,F.shape()))

dBias = R_bias(sum(Gout,axes=[N,Hout,Wout],keepDimensions=false))
```

Equivalent public reshape/permute/expand/[`MATMUL`](../../../../glossary.md#matrix-multiplication--matmul) ordering is allowed, but group isolation is
mandatory. Do not create a dense cross-group [matrix](../../../../glossary.md#term-matrix-meanings), backward kind, [element](../../../../glossary.md#term-element) enumeration, or
physical columns.

[Inference](../../../../glossary.md#term-inference)/[preflight](../../../../glossary.md#term-preflight) retain/prove positive static [kernels](../../../../glossary.md#kernel), input/output [channel](../../../../glossary.md#term-channel) divisibility,
`Cin == groups*CinPerGroup`, [bias](../../../../glossary.md#term-bias)/output-channel equality, NCHW [rank](../../../../glossary.md#term-rank)/type, effective-[kernel](../../../../glossary.md#kernel) fit,
and exact output geometry. Dynamic/expression channel/[spatial](../../../../glossary.md#term-spatial-axes) [dimensions](../../../../glossary.md#dimension) remain supported when
typed constraints prove these relations; do not impose new static requirements.

Empty domains and exceptional values follow ordinary unfold/`MATMUL`/sum/fold semantics. Select no
extra [NaN](../../../../glossary.md#term-nan)/[infinity](../../../../glossary.md#term-infinity)/[signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero) repair.

### Average [pooling](../../../../glossary.md#term-pooling)

With matching `Window2dAttrs`, flatten `g` to `[N,C,L]` and form divisor `D` by expanding logical
ones over `[kH,kW]` and summing:

```text
perPosition = restored(g)/D
columns     = expand/reshape(perPosition)        [N,C*kH*kW,L]
dInput      = R_input(fold2d(columns,input.shape(),window))
```

This is fixed count-[padding](../../../../glossary.md#padding): every logical position receives `g/D`, while fold contributes only
in-bounds positions. [Dilation](../../../../glossary.md#term-dilation) and literal ceil mode use the exact forward grid. All-padding
windows contribute to no input. Never materialize [host](../../../../glossary.md#term-host) `kH*kW`, use valid-[sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) counting, or add
divisor configuration. Exceptional/empty/overlap behavior is ordinary division/fold behavior.

### Max [pooling](../../../../glossary.md#term-pooling)

`MAX_POOL2D` has no indices output, but its exact first logical winner is reconstructable without
re-running max pooling:

1. Unfold input through matching window with exact typed negative-[infinity](../../../../glossary.md#term-infinity) [padding](../../../../glossary.md#padding); reshape to
   `[N,C,L,K]`, `K=kH*kW`.
2. Unfold `O_input` through the same window with positive-zero padding. Nonzero entries mark
   in-bounds samples, distinguishing real negative infinity from padding.<span data-glossary-navigation="true"> ([samples glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
3. Align the exact same-[occurrence](../../../../glossary.md#term-occurrence) output to `[N,C,L,1]`.
4. Candidate means in-bounds and either both values are NaN or numerically equal with signed-zero
   agreement. Since equality equates signed zeros, compare <span data-glossary-navigation="true"> ([signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>`O/candidate` with `O/output` in the
   both-zero case; the reciprocals are signed infinities.<span data-glossary-navigation="true"> ([infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
5. Convert candidates to zero/one, use `argMax(-1,false,FIRST_INDEX)`, one-hot its static positive
   `K`, and intersect with candidates. The intersection makes all-padding select nothing.
6. Route aligned `g` with `where(selected,g,+0)`, reverse the column [layout](../../../../glossary.md#layout), and fold to input.

Policy:

- equal finite values, infinities, signed zeros, and multiple NaNs route only to the first
  increasing kernel-height/kernel-width in-bounds sample;<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite); [kernel glossary](../../../../glossary.md#kernel))</span>
- [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero) wins over [negative zero](../../../../glossary.md#nan-infinity-and-signed-zero);
- selected [NaN](../../../../glossary.md#term-nan) routes the incoming [cotangent](../../../../glossary.md#cotangent) to that exact first NaN [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix);
- padding never wins, including a tie with real [negative infinity](../../../../glossary.md#nan-infinity-and-signed-zero);
- all-padding and zero [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)/[channel](../../../../glossary.md#term-channel) domains contribute zero;
- overlapping winners add through `fold2d`;
- no hidden index, unpool kind, storage, or physical saved state is created.

### [Loss](../../../../glossary.md#loss) [cotangent](../../../../glossary.md#cotangent) restoration

For `NONE` result domain [Shape](../../../../glossary.md#shape) `S`:

```text
restore(NONE,g,S) = g
restore(SUM,g,S)  = expand(g,S)
restore(MEAN,g,S) = expand(g,S)/logicalCount(S)
```

Categorical losses insert a [singleton](../../../../glossary.md#term-singleton) at the normalized [class axis](../../../../glossary.md#class-axis) afterward. [MSE](../../../../glossary.md#mean-squared-error--mse) uses the full
[prediction](../../../../glossary.md#term-prediction) Shape. Empty domain yields an empty restored [Tensor](../../../../glossary.md#tensor) even though [scalar](../../../../glossary.md#term-scalar) forward [mean](../../../../glossary.md#term-mean) is
[NaN](../../../../glossary.md#term-nan). Never use a static [host](../../../../glossary.md#term-host) [element](../../../../glossary.md#term-element) count.

### [Mean-squared error](../../../../glossary.md#mean-squared-error--mse)

After output-type casts:

```text
D  = prediction-target
dP = R_prediction( 2*D*restore(reduction,g,prediction.shape()))
dT = R_target(     -2*D*restore(reduction,g,prediction.shape()))
```

Two and negative two are exact in result type. Preserve this unexpanded difference formula.
[NaN](../../../../glossary.md#term-nan)/[infinity](../../../../glossary.md#term-infinity)/[signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero)/[overflow](../../../../glossary.md#term-overflow)/[underflow](../../../../glossary.md#term-underflow) and empty behavior follows it for all reductions,
mixed floating types, and dynamic [Shapes](../../../../glossary.md#shape).

### Dense-[target](../../../../glossary.md#term-target) categorical cross-entropy

Let [class axis](../../../../glossary.md#class-axis) `c`, `P=softmax(logits,c)`, `LP=logSoftmax(logits,c)`, target `T` cast to result
type, restored [cotangent](../../../../glossary.md#cotangent) `R` with [singleton](../../../../glossary.md#term-singleton) class [dimension](../../../../glossary.md#dimension), and `Tsum=sum(T,c,true)`:

```text
dLogits = R_logits((P*Tsum-T)*R)

rawTarget = (-LP)*R
dTarget = R_target(where(
    (T == +0) AND NOT isFinite(LP),
    +0,
    rawTarget))
```

Retain `Tsum`; do not silently normalize targets. Existing [finite](../../../../glossary.md#term-finite)/non-negative/normalized target
requirements remain caller/execution obligations, and violating inputs gain no [gradient](../../../../glossary.md#term-gradient)
guarantee. The target rule selects zero only at the forward zero-[weight](../../../../glossary.md#term-weight)/non-finite-[log](../../../../glossary.md#term-log)-probability
discontinuity; zero target with finite `LP` retains ordinary `-LP`. Other exceptional logits
follow current [softmax](../../../../glossary.md#softmax--log-softmax)/[log-softmax](../../../../glossary.md#softmax--log-softmax) and raw arithmetic.

### Index-[target](../../../../glossary.md#term-target) categorical cross-entropy

[INT32](../../../../glossary.md#data-type--datatype)/[INT64](../../../../glossary.md#data-type--datatype) target is always [`ND`](../../../../glossary.md#term-nd). For static positive depth `C`, let `upper` be `C-1`
when representable in the target type and that type's maximum value otherwise:

```text
safeTarget = minimum(
    maximum(target,exactIntegral(0)),
    exactIntegral(upper))
hotBool    = oneHot(safeTarget,C)
hot        = where(hotBool,O_logits,Z_logits)
```

No-ignore:

```text
dLogits = R_logits((softmax(logits,c)-hot)*restoredScale)
```

Present exact typed `ignoreIndex`:

```text
delta       = target-ignoreIndex
ignored     = equalTo(delta,delta-delta)
activeFloat = where(ignored,Z_sample,O_sample)
activeCount = sum(activeFloat)

restoredScale(NONE/SUM) = ordinary restored cotangent
restoredScale(MEAN)     = ordinary restored cotangent/activeCount

dLogits = R_logits(where(
    expandDims(ignored,c),
    Z_logits,
    (softmax(logits,c)-hot)*restoredScale))
```

Integral subtraction is exact modulo target width, so `delta == delta-delta` exactly detects the
configured ignore value without an integral [Tensor](../../../../glossary.md#tensor) [constant](../../../../glossary.md#term-constant). [Scalar](../../../../glossary.md#term-scalar) `maximum`/`minimum` before
one-hot makes any ignored value safe while staying in the exact integral category; it is [identity](../../../../glossary.md#term-identity-meanings)
for obligation-satisfying non-ignored targets. Out-of-range non-ignored targets already have no
forward execution meaning.

Final `where` excludes ignored logits before arithmetic, so ignored rows remain zero with
[NaN](../../../../glossary.md#term-nan)/infinite logits and all-ignored [`MEAN`](../../../../glossary.md#term-mean). `MEAN` divides by non-ignored count; `NONE`/`SUM` do
not. Dynamic/zero class depth fails closed.

### [Inference](../../../../glossary.md#term-inference) and [preflight](../../../../glossary.md#term-preflight)

Review `StructuredOperationInference`; extend only for a missing typed constraint. It must retain:

- attention embedding positivity/equality, key/value sequence equality, three-way batch and mask
  broadcast, exact output count/descriptors, and scale type;<span data-glossary-navigation="true"> ([embedding glossary](../../../../glossary.md#embedding); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [mask glossary](../../../../glossary.md#term-boolean-mask))</span>
- [convolution](../../../../glossary.md#term-channels-first-convolution-layers) [channel](../../../../glossary.md#term-channel) divisibility/equality, [bias](../../../../glossary.md#term-bias) length, effective-[kernel](../../../../glossary.md#kernel) fit, and [NCHW](../../../../glossary.md#term-nchw) geometry;
- [pooling](../../../../glossary.md#term-pooling) effective-kernel/grid obligations for floor and literal ceil modes;
- [MSE](../../../../glossary.md#mean-squared-error--mse) positional [Shape](../../../../glossary.md#shape) equality;
- dense/index [loss](../../../../glossary.md#loss) [target](../../../../glossary.md#term-target) Shape and positive-class-or-empty/all-ignored forward conditions.

Preflight inspects [canonical](../../../../glossary.md#term-canonical-form) [producers](../../../../glossary.md#term-producer)/attrs, normalizes [axes](../../../../glossary.md#term-axis) once, selects requested roles, and
proves formula prerequisites before construction:

- exactly two canonical attention wrappers with exact [descriptors](../../../../glossary.md#term-descriptor);
- convolution group reshapes preserve exact symbolic [element](../../../../glossary.md#term-element) counts;
- pool window Shapes match attrs and output descriptor;
- reduction restoration matches exact `NONE` domain;
- index class [extent](../../../../glossary.md#term-extent) is static positive before `oneHot`.

One-output attention, dynamic/zero-depth index loss, malformed manual operations, unexpected
attrs/counts/slots, unknown structured kinds, and unproved obligations fail with deterministic
[occurrence](../../../../glossary.md#term-occurrence)/role detail.

### Failure order, determinism, [producers](../../../../glossary.md#term-producer), and [constants](../../../../glossary.md#term-constant)

Preserve request-wide order:

1. validate request/[objective](../../../../glossary.md#term-objective)/[targets](../../../../glossary.md#term-target) and complete [preflight](../../../../glossary.md#term-preflight);
2. reject every unsupported [signature](../../../../glossary.md#term-signature)/slot/role/constraint;
3. only then allocate [seed](../../../../glossary.md#term-seed), splats, and [derivative](../../../../glossary.md#term-derivative) expressions;
4. reverse-traverse exact original producers/[canonical](../../../../glossary.md#term-canonical-form) wrappers;
5. accumulate by [Tensor](../../../../glossary.md#tensor) [identity](../../../../glossary.md#term-identity-meanings) in deterministic encounter order;
6. [capture](../../../../glossary.md#term-capture-meanings) combined forward/[backward graph](../../../../glossary.md#backward-graph) once with exact phase [provenance](../../../../glossary.md#provenance).

Failure consumes no Tensor IDs and leaves no partial state. Generated floating constants use the
request-local exact-bit [cache](../../../../glossary.md#term-cache) in first-use order: [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero), positive one, and other exact
typed splats as reached. Exact two and negative two remain [scalar](../../../../glossary.md#term-scalar)-[operation](../../../../glossary.md#operation) coefficients; [pooling](../../../../glossary.md#term-pooling)
[padding](../../../../glossary.md#padding) uses exact positive-zero or negative-[infinity](../../../../glossary.md#term-infinity) scalar attributes where required. Integral
zero/`upper` are same-type [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) attrs, not Tensor [leaves](../../../../glossary.md#term-leaf). Reuse exact type/bit matches.

Attention [weights](../../../../glossary.md#term-weight) and max-pool outputs are exact original Tensor identities. Max winner-selection
expressions are new backward-phase public occurrences, not physical saved state.

### Transitive emitted-[operation](../../../../glossary.md#operation) boundary

Formulas may emit only current public operations already covered by [Compiler](../../../../glossary.md#term-compiler) 0004–0005C or this
task: cast; [scalar](../../../../glossary.md#term-scalar)/[binary](../../../../glossary.md#term-binary-descriptor-and-metadata) arithmetic; comparison/logical/classification/`where`; `rsqrt`;
[softmax](../../../../glossary.md#softmax--log-softmax)/[log-softmax](../../../../glossary.md#softmax--log-softmax); sum/arg-max; [Shape](../../../../glossary.md#shape) [normalization](../../../../glossary.md#term-normalization); expand/reshape/permute/squeeze/unsqueeze;
[`MATMUL`](../../../../glossary.md#matrix-multiplication--matmul); `oneHot`; `unfold2d`; and `fold2d`.

Lock the exact emitted-kind inventory in tests. Task 0005E owns the source-wide transitive audit
before higher-order requests.

## Out of [scope](../../../../glossary.md#term-scope)

- changes under `modules/model` or any public [Tensor](../../../../glossary.md#tensor)/[operation](../../../../glossary.md#operation)/result-[carrier](../../../../glossary.md#term-storage-carrier) [signature](../../../../glossary.md#term-signature)
- new hidden outputs, range/integral-selection operations, or backward kinds
- [differentiable](../../../../glossary.md#derivative-and-chain-rule) one-output attention or dynamic/zero-depth index [loss](../../../../glossary.md#loss)
- derivatives for masks, integral targets/indices, axes, attrs, reductions, scales, groups, or
  window/configuration values<span data-glossary-navigation="true"> ([derivatives glossary](../../../../glossary.md#term-derivative); [masks glossary](../../../../glossary.md#term-boolean-mask); [axes glossary](../../../../glossary.md#term-axis))</span>
- attention dropout, transposed convolution, broader bias rules, pool indices/unpool, configurable
  average divisor, extra losses, label smoothing, class weights, or sparse conversions<span data-glossary-navigation="true"> ([dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [bias glossary](../../../../glossary.md#term-bias); [weights glossary](../../../../glossary.md#term-weight))</span>
- execution, numerical evaluation, backend/conformance, kernels, runtime, prepare, engine,
  training, optimizers, storage, buffers, or tape state<span data-glossary-navigation="true"> ([evaluation glossary](../../../../glossary.md#term-evaluation); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [kernels glossary](../../../../glossary.md#kernel); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [engine glossary](../../../../glossary.md#term-engine); [optimizers glossary](../../../../glossary.md#term-optimizer); [buffers glossary](../../../../glossary.md#term-buffer))</span>
- public [compile](../../../../glossary.md#compile) expansion, explicit [seeds](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings)/objectives/[targets](../../../../glossary.md#term-target), disconnected policy, `createGraph`,
  higher order, or task 0005E's full closure checkpoint<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- architecture/[dependency](../../../../glossary.md#term-dependency)/[module](../../../../glossary.md#term-module)/[Gradle](../../../../glossary.md#term-gradle) changes or unrelated refactors

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Overview](../../../../architecture/overview.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>
- [ADR 0009](../../../../design/decisions/0009-compiler-owned-pre-capture-tensor-expression-autograd.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Roadmap](../../../roadmap.md)
- [Compiler master plan](../master-plan.md)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>
- [Model master plan](../../model/master-plan.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Model capabilities](../../model/capabilities.md)
- [Model 0023 adjoint audit](../../model/tasks/0023-adjoint-expressibility-audit.md)<span data-glossary-navigation="true"> ([adjoint glossary](../../../../glossary.md#term-adjoint))</span>
- [Model 0023D window transforms](../../model/tasks/0023d-public-fold-axis-and-dynamic-window-transforms.md)
- [Model 0023F attention weights](../../model/tasks/0023f-scaled-dot-product-attention-weights-output.md)<span data-glossary-navigation="true"> ([weights glossary](../../../../glossary.md#term-weight))</span>
- [Model 0025 canonical outputs](../../model/tasks/0025-canonical-tensor-producer-outputs.md)<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- [Compiler 0004A](0004a-exact-composition-gradient-rule-extensions.md)
- [Compiler 0004B](0004b-shared-algebra-cotangent-normalization-and-local-derivative-rules.md)
- [Compiler 0005B](0005b-reduction-scan-softmax-statistics-and-normalization-gradient-completion.md)
- [Compiler 0005C](0005c-layout-window-indexing-scatter-ordering-and-stochastic-gradient-completion.md)

## Architecture constraints

- `ARCHITECTURE.md` is authoritative.
- [Preflight](../../../../glossary.md#term-preflight), formulas, accumulation, phase assignment, and combined [capture](../../../../glossary.md#term-capture-meanings) remain [compiler](../../../../glossary.md#term-compiler)-owned.
- Forward/backward expressions share one Tensor algebra, inference, validation, numerical, and
  optimization contract.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [validation glossary](../../../../glossary.md#term-validation))</span>
- Only [canonical](../../../../glossary.md#term-canonical-form) original-[producer](../../../../glossary.md#term-producer) [Tensors](../../../../glossary.md#tensor) are same-[occurrence](../../../../glossary.md#term-occurrence) auxiliaries.
- [Identity](../../../../glossary.md#term-identity-meanings) maps are request-local implementation state, not [graph](../../../../glossary.md#graph)/[artifact](../../../../glossary.md#term-artifact) representation.
- No runtime/prepare/engine/training/backend dependency, Tensor gradient state, mutable gradient
  storage, second algebra, tape, physical buffer, schedule, or execution unit is added.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [dependency glossary](../../../../glossary.md#term-dependency); [gradient glossary](../../../../glossary.md#term-gradient); [mutable glossary](../../../../glossary.md#term-mutable); [buffer glossary](../../../../glossary.md#term-buffer))</span>
- Architectural uncertainty stops implementation for clarification.

## Package impact

The cohesive `io.github.pho001.synaptik.compiler` package gains four package-private final family
[owners](../../../../glossary.md#term-owner):

- `AttentionGradientRules`
- `ConvolutionGradientRules`
- `PoolingGradientRules`
- `LossGradientRules`

`AutogradPreflight` owns role/prerequisite selection; `FirstOrderAutograd` owns [dispatch](../../../../glossary.md#term-dispatch) and
request-local [constants](../../../../glossary.md#term-constant); `StructuredOperationInference` changes only if source review finds a
missing forward constraint. Add no subpackage, public type, [registry](../../../../glossary.md#term-registry), or generic [facade](../../../../glossary.md#term-facade).

## Affected files

Production:

- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/AutogradPreflight.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/FirstOrderAutograd.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/StructuredOperationInference.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/AttentionGradientRules.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ConvolutionGradientRules.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/PoolingGradientRules.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/LossGradientRules.java`

Tests:

- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/StructuredOperationInferenceTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/AutogradPreflightTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/FirstOrderAutogradTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/GraphCompilerTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/AttentionGradientRulesTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/ConvolutionAndPoolingGradientRulesTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/LossGradientRulesTest.java`

Documentation/[planning](../../../../glossary.md#term-planning):

- `docs/api/compile-api.md`
- `docs/api/tensor-api.md`
- `docs/glossary.md`
- `docs/planning/modules/compiler/tasks/0005d-attention-convolution-pooling-and-loss-gradient-completion.md`
- `docs/planning/modules/compiler/master-plan.md`
- `docs/planning/roadmap.md`

Review-only unless contradiction requires stopping: architecture/[ADR](../../../../glossary.md#term-architecture-decision-record-adr) files; Training [API](../../../../glossary.md#term-application-programming-interface-api);
[model](../../../../glossary.md#term-model) capabilities/master/tasks and current structured/window/[canonical](../../../../glossary.md#term-canonical-form)-output source/tests;
`LinearAlgebraGradientRules`; completed [Compiler](../../../../glossary.md#term-compiler) 0004A/0004B/0005B/0005C; architecture,
[backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), integration, [Gradle](../../../../glossary.md#term-gradle), and other [modules](../../../../glossary.md#term-module).

## Maximum [scope](../../../../glossary.md#term-scope)

At most 20 paths: seven production, seven tests, and six documentation/[planning](../../../../glossary.md#term-planning) paths above.
A separate documentation-focused clean context finalizes Javadocs/docs/status in those paths
without repeating successful Java suites unless it changes Java behavior.

If a [model](../../../../glossary.md#term-model)/[API](../../../../glossary.md#term-application-programming-interface-api) [operation](../../../../glossary.md#operation), twenty-first path, another [module](../../../../glossary.md#term-module), architecture/[dependency](../../../../glossary.md#term-dependency)/build change, or
different [derivative](../../../../glossary.md#term-derivative) policy is needed, stop for clarification.

## Acceptance criteria

### Inventory and [preflight](../../../../glossary.md#term-preflight)

- Every assigned [signature](../../../../glossary.md#term-signature), output slot, reduction, and role appears exactly once.
- Existing floating [`MATMUL`](../../../../glossary.md#matrix-multiplication--matmul) and visible linear composition are revalidated; integral roles remain
  fail-closed.<span data-glossary-navigation="true"> ([linear glossary](../../../../glossary.md#term-linear-meanings); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- One-output attention and dynamic/zero-depth index [loss](../../../../glossary.md#loss) fail before allocation.
- Two-output attention uses exact [canonical](../../../../glossary.md#term-canonical-form) [weights](../../../../glossary.md#term-weight); [masks](../../../../glossary.md#term-boolean-mask)/indices/configuration remain [`ND`](../../../../glossary.md#term-nd).
- Malformed/unknown/unproved occurrences fail with deterministic [occurrence](../../../../glossary.md#term-occurrence)/slot/role context.

### Formulas

- Attention covers both slots, scale forms, mixed types, broadcasting, masks, all-masked/all
  negative-infinity, positive-infinity ties, NaNs, and stable accumulation as specified.<span data-glossary-navigation="true"> ([broadcasting glossary](../../../../glossary.md#broadcasting); [masks glossary](../../../../glossary.md#term-boolean-mask); [infinity glossary](../../../../glossary.md#term-infinity))</span>
- Grouped/ungrouped [convolution](../../../../glossary.md#term-channels-first-convolution-layers) restores exact symbolic group/window geometry and input types.
- Average pool uses count-padding fold; max pool reconstructs first in-bounds winner including
  NaN/infinity/signed-zero/real-negative-infinity/padding/overlap/all-padding cases.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [NaN glossary](../../../../glossary.md#term-nan); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- MSE and dense/index loss cover every reduction, mixed type, dynamic/empty domain,
  ignore/all-ignored, and stated exceptional policy.<span data-glossary-navigation="true"> ([MSE glossary](../../../../glossary.md#mean-squared-error--mse); [loss glossary](../../../../glossary.md#loss))</span>
- Tests inspect exact kinds, attrs, ordered provenance, canonical identities/slots, constants, and
  absence of storage/hidden backward kinds—not only names/counts.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [canonical glossary](../../../../glossary.md#term-canonical-form); [constants glossary](../../../../glossary.md#term-constant))</span>

### Pipeline and boundaries

- One deterministic combined [capture](../../../../glossary.md#term-capture-meanings), exact phases/[publication](../../../../glossary.md#term-publication), and logical-splat ingress remain.
- Failed [preflight](../../../../glossary.md#term-preflight) consumes no ID and creates no partial formula.
- Emitted kinds stay inside the stated shared-algebra boundary.
- No model/runtime/prepare/engine/training/backend/architecture/dependency/Gradle/public compile
  behavior changes.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [compile glossary](../../../../glossary.md#compile))</span>
- Java contracts/Javadocs and affected docs are finalized in an independent documentation context;
  reviewed no-change conclusions are recorded.

## Tests / [validation](../../../../glossary.md#term-validation)

Focused:

```text
./gradlew :modules:compiler:test \
  --tests io.github.pho001.synaptik.compiler.StructuredOperationInferenceTest \
  --tests io.github.pho001.synaptik.compiler.AutogradPreflightTest \
  --tests io.github.pho001.synaptik.compiler.FirstOrderAutogradTest \
  --tests io.github.pho001.synaptik.compiler.GraphCompilerTest \
  --tests io.github.pho001.synaptik.compiler.AttentionGradientRulesTest \
  --tests io.github.pho001.synaptik.compiler.ConvolutionAndPoolingGradientRulesTest \
  --tests io.github.pho001.synaptik.compiler.LossGradientRulesTest
```

Required once:

```text
./gradlew :modules:compiler:test
./gradlew :modules:compiler:javadoc
git diff --check
```

Also record source-[signature](../../../../glossary.md#term-signature)/role inventory; emitted kind/attrs/[provenance](../../../../glossary.md#provenance) and [canonical](../../../../glossary.md#term-canonical-form)-output
inspection; ID-delta failures; combined phase/[publication](../../../../glossary.md#term-publication)/[constant](../../../../glossary.md#term-constant)/determinism checks; scans proving
no [`LINEAR`](../../../../glossary.md#term-linear-meanings), hidden/backward/tape/[registry](../../../../glossary.md#term-registry)/public [API](../../../../glossary.md#term-application-programming-interface-api); exact 20-path ceiling inventory; Markdown
links/headings/fences/whitespace/status; and General/API-[Javadoc](../../../../glossary.md#term-javadoc)/[Planning](../../../../glossary.md#term-planning)/Example documentation
review.

Repository-wide validation is deferred to [Compiler](../../../../glossary.md#term-compiler) 0005E's first-order [checkpoint](../../../../glossary.md#term-checkpoint) and [CI](../../../../glossary.md#term-continuous-integration-ci). No
architecture/[backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance)/[integration test](../../../../glossary.md#term-integration-test) is required because no such behavior changes.

## Dependencies

- Models 0023, 0023D, 0023F, and 0025 — Complete.
- Compilers 0004A, 0004B, 0005B, and 0005C — Complete.
- Stable current structured operations, window transforms, canonical outputs, inference,
  preflight, one-capture, and publication contracts.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [inference glossary](../../../../glossary.md#term-inference); [preflight glossary](../../../../glossary.md#term-preflight); [capture glossary](../../../../glossary.md#term-capture-meanings); [publication glossary](../../../../glossary.md#term-publication))</span>

## Follow-up tasks

- Compiler 0005E performs source-wide first-order/fail-closed and transitive-formula audit, records
  intentional limitations, and runs the capability checkpoint.<span data-glossary-navigation="true"> ([fail-closed glossary](../../../../glossary.md#term-fail-closed); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- [Compiler](../../../../glossary.md#term-compiler) 0006 may add explicit functional [gradient](../../../../glossary.md#term-gradient) requests only after 0005E.
- A future focused model/compiler task may reconsider one-output attention or dynamic/zero-depth
  index loss only if a generally useful public capability is independently justified.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [loss glossary](../../../../glossary.md#loss))</span>

## Architecture impact

None intended. This implements [compiler](../../../../glossary.md#term-compiler)-owned reverse formulas through the existing [Tensor](../../../../glossary.md#tensor) algebra
and one combined [capture](../../../../glossary.md#term-capture-meanings) under [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0009. Architecture documents/tests, dependencies, and build
remain unchanged.

## Implementation prompt

Work in a clean implementation context. Read root `AGENTS.md`, authoritative/focused architecture
and [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0009, documentation rules/profiles, [planning](../../../../glossary.md#term-planning) guide/roadmap, [compiler](../../../../glossary.md#term-compiler)/[model](../../../../glossary.md#term-model) master plans and
capabilities, this task, Model 0023/0023D/0023F/0025, Compiler 0004A/0004B/0005B/0005C, and every
affected/review-only source/test/document above.

Implement only this exact [matrix](../../../../glossary.md#term-matrix-meanings). Preserve [`MATMUL`](../../../../glossary.md#matrix-multiplication--matmul); verify visible [linear](../../../../glossary.md#term-linear-meanings); add four narrow
package-private rule [owners](../../../../glossary.md#term-owner); extend [inference](../../../../glossary.md#term-inference)/[preflight](../../../../glossary.md#term-preflight) only as required; keep [derivatives](../../../../glossary.md#term-derivative) public
[Tensor](../../../../glossary.md#tensor) expressions. Use [canonical](../../../../glossary.md#term-canonical-form) two-output attention [weights](../../../../glossary.md#term-weight), grouped unfold/`MATMUL`/fold,
count-[padding](../../../../glossary.md#padding) average fold, exact max winner reconstruction, and exact [loss](../../../../glossary.md#loss) policies.

Fail before allocation for one-output attention, dynamic/zero-depth index loss, malformed
occurrences, unexpected slots/attrs/counts, and unproved constraints. Never invent a hidden output,
recompute attention, add backward kinds, read storage/[bindings](../../../../glossary.md#term-binding), materialize window/index data, add
Tensor [gradient](../../../../glossary.md#term-gradient) state, or touch another [module](../../../../glossary.md#term-module).

[Run](../../../../glossary.md#term-run) focused tests, the full compiler tests once, [Javadoc](../../../../glossary.md#term-javadoc), [scope](../../../../glossary.md#term-scope)/status/link/manual checks, and
`git diff --check`. Then use a separate clean documentation context to finalize Javadocs, [Compile](../../../../glossary.md#compile)
[API](../../../../glossary.md#term-application-programming-interface-api), glossary, task evidence/summary, master plan, and roadmap without repeating Java tests unless
behavior changes. Return both context IDs, exact paths/evidence/no-change conclusions/issues and:

```text
Status: Complete
```

only when every criterion passes; otherwise use the repository's exact incomplete format.

## Local decisions

- Exact same-[occurrence](../../../../glossary.md#term-occurrence) [weights](../../../../glossary.md#term-weight) take precedence over one-output attention coverage.
- Attention uses saved-output Jacobian, zero all-masked/all-negative-infinity policy, exact saved
  positive-infinity split, and selection-safe contractions.<span data-glossary-navigation="true"> ([Jacobian glossary](../../../../glossary.md#term-jacobian); [infinity glossary](../../../../glossary.md#term-infinity))</span>
- [Convolution](../../../../glossary.md#term-channels-first-convolution-layers) uses symbolic grouped window composition; average pool uses count-[padding](../../../../glossary.md#padding).
- Max pool routes first logical in-bounds winner, distinguishes signed zeros, and routes selected
  NaN.<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan))</span>
- [MSE](../../../../glossary.md#mean-squared-error--mse) remains unexpanded and applies exact `2` or `-2` before the restored cotangent so target
  signed-zero/NaN behavior is not changed by a final negation; dense target is zero only at
  zero-weight/non-finite-log-probability; index loss sanitizes before one-hot and excludes ignored
  rows after scale.<span data-glossary-navigation="true"> ([cotangent glossary](../../../../glossary.md#cotangent); [target glossary](../../../../glossary.md#term-target); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [dense target glossary](../../../../glossary.md#dense-target); [finite glossary](../../../../glossary.md#term-finite); [log glossary](../../../../glossary.md#term-log); [loss glossary](../../../../glossary.md#loss))</span>
- No [Model](../../../../glossary.md#term-model) prerequisite is needed for the representable [matrix](../../../../glossary.md#term-matrix-meanings).

## Known limitations

- One-output attention intentionally fails because its producer has no canonical weights. Gradient
  callers must construct <span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [canonical glossary](../../../../glossary.md#term-canonical-form); [Gradient glossary](../../../../glossary.md#term-gradient))</span>`WithWeights`, even if [weights](../../../../glossary.md#term-weight) are not a final published output.
- Index loss requires statically positive class depth; dynamic and zero-class empty/all-ignored
  forward cases fail closed.<span data-glossary-navigation="true"> ([loss glossary](../../../../glossary.md#loss); [fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation))</span>
- Current scalar-objective, implicit-unit-seed, requested-target first-order contract remains;
  task 0006 owns broader requests.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [objective glossary](../../../../glossary.md#term-objective); [seed glossary](../../../../glossary.md#term-seed); [target glossary](../../../../glossary.md#term-target))</span>
- Formulas are symbolic; execution/[conformance](../../../../glossary.md#term-conformance) remain later [lifecycle](../../../../glossary.md#lifecycle) work.

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) evidence:

- Read repository instructions, architecture/focused docs/ADR, documentation profiles, planning
  guide/roadmap, compiler/model master/capabilities, completed tasks through Compiler 0005C, model
  adjoint/prerequisite tasks, and current structured source/tests.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [compiler glossary](../../../../glossary.md#term-compiler); [model glossary](../../../../glossary.md#term-model); [adjoint glossary](../../../../glossary.md#term-adjoint))</span>
- Confirmed one- and two-output attention, grouped convolution, separate pools, every loss/reduction,
  public dynamic window/fold operations, and canonical producer wrappers.<span data-glossary-navigation="true"> ([convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [loss glossary](../../../../glossary.md#loss); [canonical glossary](../../../../glossary.md#term-canonical-form); [producer glossary](../../../../glossary.md#term-producer))</span>
- Confirmed static-depth ignored index loss is expressible by clamp-before-one-hot and exact
  integral masking. Only one-output attention and dynamic/zero class depth remain fail-closed.<span data-glossary-navigation="true"> ([fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- Confirmed a clean repository and no prior detailed 0005D file before planning.

Implementation validation:

- Implementation context: `/root/implement_compiler_0005d`.
- The required focused seven-suite command passed after the final executable correction.
- The replacement `./gradlew :modules:compiler:test` run passed after correcting the MSE target
  coefficient order: 28 XML suites, 189 tests, zero skipped tests, zero failures, and zero errors.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [MSE glossary](../../../../glossary.md#mean-squared-error--mse); [target glossary](../../../../glossary.md#term-target))</span>
- The exact BFLOAT16/FLOAT32/FLOAT64 negative-two bit patterns and MSE expression order are covered
  directly. No executable Java or test changed after the replacement full-module evidence.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [module glossary](../../../../glossary.md#term-module))</span>
- Final source/test inspection confirmed the exact role/output matrix, canonical attention/max-
  pool identities, grouped convolution geometry, pooling policies, loss reductions, stable
  accumulation, preflight-before-allocation failures, combined phase/publication behavior, and
  emitted ordinary-operation boundary.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings); [pooling glossary](../../../../glossary.md#term-pooling); [preflight glossary](../../../../glossary.md#term-preflight); [publication glossary](../../../../glossary.md#term-publication); [operation glossary](../../../../glossary.md#operation))</span>

Documentation validation:

- Documentation context: `/root/implement_compiler_0005d/compiler_0005d_docs`.
- Applied the General, API/Javadoc, Planning, and Example profiles; no new standalone example was
  necessary because the existing Tensor API examples remain accurate and the formula tables are
  the task-appropriate evidence.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Finalized Javadocs for both changed owners and all four new family owners; synchronized Compile
  API, Tensor API, glossary, task, compiler master plan, and roadmap.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- Compiler Javadoc generation, Markdown links/headings/anchors/fences/final-newline/trailing-
  whitespace checks, exact 20-path-ceiling inventory, lifecycle-status checks, forbidden-surface
  scans, and <span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>`git diff --check` passed after the final documentation edits.

## Implementation notes

- Added package-private `AttentionGradientRules`, `ConvolutionGradientRules`,
  `PoolingGradientRules`, and `LossGradientRules`; extended `AutogradPreflight` and
  `FirstOrderAutograd` without changing public [compiler](../../../../glossary.md#term-compiler) APIs.
- Added three new suites covering the four family [owners](../../../../glossary.md#term-owner) and extended
  `StructuredOperationInferenceTest`, `AutogradPreflightTest`, `FirstOrderAutogradTest`, and
  `GraphCompilerTest`.
- `StructuredOperationInference` was reviewed and remained unchanged because its existing typed
  constraints already prove every required formula prerequisite. `LinearAlgebraGradientRules`
  was reviewed unchanged; its MATMUL matrix and the public PERMUTE/MATMUL/ADD linear composition
  remain sufficient.<span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [linear glossary](../../../../glossary.md#term-linear-meanings))</span>
- Tensor API required a narrowly authorized sixth documentation path because its attention,
  convolution, pooling, and loss sections explicitly described the new compiler gradients as
  future work. The task ceiling therefore increased from 19 to 20 paths; the final change uses
  19.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [pooling glossary](../../../../glossary.md#term-pooling); [loss glossary](../../../../glossary.md#loss); [gradients glossary](../../../../glossary.md#term-gradient))</span>
- Training API remained accurate because 0005D adds no public gradient publication, optimizer,
  prepared execution, or training-session contract. Model capabilities/master/tasks,
  architecture/ADR/tests, backend conformance, integration, Gradle, and other modules required no
  change.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [optimizer glossary](../../../../glossary.md#term-optimizer); [prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution); [Model glossary](../../../../glossary.md#term-model); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module))</span>

## Completion summary

- Completed the exact representable structured-neural first-order matrix and preserved MATMUL/
  linear behavior, one combined capture, compiler ownership, and exact cotangent normalization.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [linear glossary](../../../../glossary.md#term-linear-meanings); [capture glossary](../../../../glossary.md#term-capture-meanings); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [cotangent normalization glossary](../../../../glossary.md#cotangent-normalization))</span>
- Added or changed 19 paths: six production files, seven test files, and six documentation/
  planning files, within the authorized 20-path ceiling.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Focused and full compiler validation passed with 189 final module tests; Javadoc, Markdown,
  scope, status, surface, and whitespace validation also passed.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [module glossary](../../../../glossary.md#term-module); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>
- Unresolved implementation issues: none. Intentional limitations remain one-output attention and
  index loss with dynamic or zero class depth; Compiler 0005E owns the closure checkpoint.<span data-glossary-navigation="true"> ([loss glossary](../../../../glossary.md#loss); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Required follow-up: [Compiler](../../../../glossary.md#term-compiler) 0005E, which remains Draft without a detailed specification.

Status: Complete
