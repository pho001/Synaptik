# Task 0005B: Reduction, [Scan](../../../../glossary.md#term-cumulative-scan), [Softmax](../../../../glossary.md#softmax--log-softmax), Statistics, and [Normalization](../../../../glossary.md#term-normalization) [Gradient](../../../../glossary.md#term-gradient) Completion

## Status

Complete

## Goal

Complete first-order [automatic differentiation](../../../../glossary.md#autograd) for the current reduction, cumulative-[scan](../../../../glossary.md#term-cumulative-scan),
[softmax](../../../../glossary.md#softmax--log-softmax), statistical-reduction, [norm](../../../../glossary.md#term-norm), and [normalization](../../../../glossary.md#term-normalization) inventory inside the established
[compiler](../../../../glossary.md#term-compiler)-owned pre-[capture](../../../../glossary.md#term-capture-meanings) [Tensor](../../../../glossary.md#tensor)-expression pipeline.

This task also adopts the [binding](../../../../glossary.md#term-binding)-aware `EXPAND` contract completed by [Model](../../../../glossary.md#term-model) 0025B:

```text
binding-aware forward occurrence
  -> compiler inference emits the exact unresolved Shape obligation
  -> preflight proves that the inverse formula has the same or a stronger obligation
  -> ordinary public Tensor operations construct the contribution
  -> one combined capture retains every unresolved predicate fail-closed
```

The task preserves every completed Compiler 0004–0005A rule. It extends the closed rule [matrix](../../../../glossary.md#term-matrix-meanings); it
does not replace the current request, Tensor algebra, capture, [inference](../../../../glossary.md#term-inference), optimization,
[publication](../../../../glossary.md#term-publication), or [artifact](../../../../glossary.md#term-artifact) [lifecycle](../../../../glossary.md#lifecycle).

## [Scope](../../../../glossary.md#term-scope)

### Source-backed [operation](../../../../glossary.md#operation) and role inventory

Implementation must derive and lock the inventory from the production kinds, [signatures](../../../../glossary.md#term-signature),
attributes, [producer](../../../../glossary.md#term-producer) output [descriptors](../../../../glossary.md#term-descriptor), [Tensor](../../../../glossary.md#tensor) helpers, and tests rather than trusting this table
alone.

| Family | Exact current variants | [Differentiable](../../../../glossary.md#derivative-and-chain-rule) roles in 0005B |
|---|---|---|
| `AggregateReductionKind` | `SUM`, [`MEAN`](../../../../glossary.md#term-mean), `PROD`, `MIN`, `MAX`, `ALL`, `ANY`, `ARG_MIN`, `ARG_MAX`, `LOG_SUM_EXP`, [`VARIANCE`](../../../../glossary.md#term-variance), `STANDARD_DEVIATION`, `L1_NORM`, `L2_NORM` | Floating inputs of ordinary full, single-[axis](../../../../glossary.md#term-axis), and multi-axis `SUM`, `MEAN`, `PROD`, `MIN`, and `MAX`; floating data for masked `SUM`/`MEAN`; floating input for [binding](../../../../glossary.md#term-binding)-aware `SUM_TO_SHAPE`; and the floating input of all five advanced reductions. `ALL`, `ANY`, and both arg-[extrema](../../../../glossary.md#saturation-pole-and-extremum) operations are non-differentiable. The [BOOL](../../../../glossary.md#data-type--datatype) [mask](../../../../glossary.md#term-boolean-mask) is non-differentiable. |
| `CumulativeScanKind` | `CUM_SUM`, `CUM_PROD` | The one floating input for every exclusive/reverse combination. Integral scans are non-differentiable. |
| `SoftmaxKind` | [`SOFTMAX`](../../../../glossary.md#softmax--log-softmax), `LOG_SOFTMAX` | The one floating input. |
| `LayerNormKind` | no-affine `[input]` and affine `[input, scale, bias]` `LAYER_NORM` | Input in both forms; scale and [bias](../../../../glossary.md#term-bias) in the affine form. |
| `RmsNormKind` | `[input]` and `[input, scale]` `RMS_NORM` | Input in both forms; scale in the two-input form. |
| `BatchNormKind.BATCH_NORM_INFERENCE` | inputs `[input, scale, bias, runningMean, runningVariance]`, output slot `0` | All five floating Tensor inputs. [Channel](../../../../glossary.md#term-channel) axis and [epsilon](../../../../glossary.md#epsilon) are non-differentiable configuration. |
| `BatchNormKind.BATCH_NORM_TRAINING` | the same five inputs; outputs `0=output`, `1=nextRunningMean`, `2=nextRunningVariance`, `3=savedBatchMean`, `4=savedInverseStandardDeviation` | Slot 0: input, scale, and bias. Slot 1: input and running mean. Slot 2: input and running variance. Slots 3 and 4 are exact same-[occurrence](../../../../glossary.md#term-occurrence) auxiliaries used by formulas, not independent requested [cotangent](../../../../glossary.md#cotangent) roots. [Momentum](../../../../glossary.md#term-momentum), epsilon, and channel axis are non-differentiable configuration. |
| `ShapeTransformKind.EXPAND` | existing one-input `TargetShapeAttrs` occurrence | Preserve the existing floating input rule while admitting its [Model](../../../../glossary.md#term-model)-0025B binding-aware [Shape](../../../../glossary.md#shape) cases. |

The exact aggregate-kind count is fourteen, the [scan](../../../../glossary.md#term-cumulative-scan)-kind count is two, the softmax-kind count is
two, and Layer, RMS, and [batch normalization](../../../../glossary.md#batch-normalization--batch-normalization-inference-and-training) contribute one, one, and two kind [constants](../../../../glossary.md#term-constant). Tests
must lock enum contents, signature variants, input roles, output slots, and classification so a
new kind, attributes variant, or output position fails closed until deliberately planned.

### Notation and shared construction rules

For one selected output `y`, incoming [cotangent](../../../../glossary.md#cotangent) `g`, and selected input `x`:

- `Z_x` and `O_x` are request-local exact typed positive-zero and positive-one logical splats
  expanded to `x.shape`.
- `K_t(v)` is the existing exact same-typed [scalar](../../../../glossary.md#term-scalar)-[operation](../../../../glossary.md#operation) attribute `v`.
- `T_y(t)` casts an operand to the selected forward output's exact floating data type when it
  differs. Mixed-floating normalization formulas first apply <span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization))</span>`T_y` to every forward-domain Tensor
  operand so intermediate Tensor operations use the occurrence result type in the original input
  promotion order; only the completed cotangent is cast back to its selected input type.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- `E_x(t, A, keep)` restores [axes](../../../../glossary.md#term-axis) `A` removed by the forward occurrence in ascending axis order
  when `keep` is false, then expands through ordinary `Tensor.expand(x.shape)`.
- `C_x(t)` casts one already [Shape](../../../../glossary.md#shape)-correct floating contribution to `x.type` exactly when its type
  differs.
- `N_A(t)` is `O_t.sum(A, true)`, with full reduction using all natural axes. It represents the
  selected logical count for static, zero, named-dynamic, and expression extents without a host
  element-count calculation.<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [element glossary](../../../../glossary.md#term-element))</span>
- `Q_t(c)` is the exact typed logical value of non-negative integer correction `c`, constructed as
  a typed scalar one expanded to static Shape `[c]` and reduced with ordinary full `SUM`.
  Correction zero is therefore an empty-domain positive zero. Do not convert <span data-glossary-navigation="true"> ([positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>`long correction`
  through an inexact host floating cast.
- `V(t, target, mappedAxes)` reshapes a lower-rank affine/statistic operand by inserting static
  singleton axes outside <span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [statistic glossary](../../../../glossary.md#term-statistic); [singleton glossary](../../../../glossary.md#term-singleton))</span>`mappedAxes`, then ordinarily expands it to [`target`](../../../../glossary.md#term-target). Layer/RMS affine
  axes are the trailing normalized axes; a batch-normalization vector maps to the channel axis.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [vector glossary](../../../../glossary.md#term-vector); [channel glossary](../../../../glossary.md#term-channel))</span>
- `R_x(t, reducedAxes)` reduces the complement or prefix axes selected by the formula, reshapes to
  the exact operand Shape when unresolved but forward-constrained Dimensions differ
  structurally, and applies <span data-glossary-navigation="true"> ([Dimensions glossary](../../../../glossary.md#dimension))</span>`C_x`.

Every returned contribution must have exactly the selected input's Shape and floating [data type](../../../../glossary.md#data-type--datatype)
before deterministic accumulation. A Shape/type-correct direct result adds no redundant reshape,
[sum-to-Shape](../../../../glossary.md#sum-to-shape), expand, or cast. All construction uses current public [Tensor](../../../../glossary.md#tensor) operations and the
existing request-local logical-splat [owner](../../../../glossary.md#term-owner).

### [Binding](../../../../glossary.md#term-binding)-aware `EXPAND` adoption

`LayoutInference` must preserve current [target](../../../../glossary.md#term-target)-[rank](../../../../glossary.md#term-rank) [validation](../../../../glossary.md#term-validation), right [alignment](../../../../glossary.md#term-alignment), fully static
rejection, [descriptor](../../../../glossary.md#term-descriptor) type/eligibility, target [Shape](../../../../glossary.md#shape), and resolved zero-[stride](../../../../glossary.md#term-stride) [view](../../../../glossary.md#term-view-meanings) [layout](../../../../glossary.md#layout). For
each aligned pair in increasing input-[axis](../../../../glossary.md#term-axis) order:

1. [structural equality](../../../../glossary.md#term-structural-equality) is proved;
2. a static source [singleton](../../../../glossary.md#term-singleton) is proved;
3. unequal fully static extents fail with the current incompatible-expand behavior; and
4. every remaining pair emits the existing predicate:

   ```text
   AnyOf(
       DimensionEqual(source, StaticDimension(1)),
       DimensionEqual(source, target))
   ```

   with deterministic subject text naming the aligned target axis.

`CapturedGraphInference` continues to classify each candidate as `PROVEN`, `DISPROVEN`, or
`DEFERRED`. A deferred result remains an [occurrence](../../../../glossary.md#term-occurrence)-owned `DeferredGraphConstraint`; it is never
treated as proof, binding, or execution permission. New leading target axes have implicit source
one and need no aligned predicate.

Resolved layout remains byte-for-byte equivalent to current static behavior. A binding-dependent
pair has unresolved layout. No predicate changes the exact target Shape or selects a future
stride, [materialization](../../../../glossary.md#materialization), or [backend route](../../../../glossary.md#backend-route).

[Preflight](../../../../glossary.md#term-preflight) must admit a binding-dependent forward `EXPAND` only when each aligned inverse
`SUM_TO_SHAPE` obligation is structurally the same as, or implied by, the forward
source-one-or-source-equal obligation. Conversely, a binding-dependent `SUM_TO_SHAPE` inverse is
ordinary `gradient.expand(input.shape())`; preflight admits it only when every generated aligned
expand predicate is exactly the forward target-one-or-target-equal-source obligation after the
source/target roles are mapped. A static contradiction, rank mismatch, differently aligned
predicate, or missing forward obligation fails before [derivative](../../../../glossary.md#term-derivative) [Tensor](../../../../glossary.md#tensor) allocation.

This is a [fail-closed](../../../../glossary.md#term-fail-closed) proof-equivalence check, not a concrete binding solver. The combined [graph](../../../../glossary.md#graph)
retains the predicate for later binding validation.

### Reduction [axes](../../../../glossary.md#term-axis) and [cotangent](../../../../glossary.md#cotangent) restoration

For full reductions, `A` is every input axis in natural order. For `AxisReductionAttrs`, `A`
contains its one [normalized axis](../../../../glossary.md#normalized-axis). For `MultiAxisReductionAttrs` and
`StatisticalReductionAttrs`, `A` is the exact ordered distinct axis list; axis order remains
attributes [identity](../../../../glossary.md#term-identity-meanings) even when a formula needs a sorted copy for axis insertion. An empty
multi-axis list selects a point domain.

Existing `SUM`, [`MEAN`](../../../../glossary.md#term-mean), masked `SUM`, masked `MEAN`, and statically provable
`SUM_TO_SHAPE` formulas remain unchanged except for [binding](../../../../glossary.md#term-binding)-aware inverse [admission](../../../../glossary.md#term-admission):

| Variant | Input contribution |
|---|---|
| ordinary `SUM` | `E_x(g, A, keepDimensions)` |
| ordinary `MEAN` | `E_x(g, A, keepDimensions) / N_A(x).expand(x.shape)` |
| masked `SUM` | `where(mask, E_x(g, [axis], false), Z_x)` |
| masked `MEAN` | `where(mask, E_x(g, [axis], false) / selectedCount, Z_x)` |
| `SUM_TO_SHAPE` | `g.expand(x.shape)` under the exact proof-equivalence rule above |

The [mask](../../../../glossary.md#term-boolean-mask) uses its original forward [broadcasting](../../../../glossary.md#broadcasting) relation and receives no cotangent. Masked
all-false slices retain [Compiler](../../../../glossary.md#term-compiler) 0004B's direct-zero contribution policy. An ordinary empty
reduction domain has no input [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) at which its zero denominator is evaluated; the
resulting contribution retains the corresponding zero-[extent](../../../../glossary.md#term-extent) [Shape](../../../../glossary.md#shape). Empty axis lists have count
one and behave as point identities.

### Zero-safe product reduction

Do not use `forwardProduct / x`. For [axes](../../../../glossary.md#term-axis) `A`, construct a deterministic sequence of ordinary
single-axis keep-[dimension](../../../../glossary.md#dimension) products in recorded axis order:

```text
stage[0] = x
stage[k + 1] = stage[k].prod(A[k], true)
```

Restore `g` to the final keep-dimension [Shape](../../../../glossary.md#shape). Traverse stages in reverse. For stage input `v` and
axis `a`:

```text
prefix = v.cumProd(a, true, false)
suffix = v.cumProd(a, true, true)
h = h.expand(v.shape) * prefix * suffix
```

The final `h` has `x.shape`. Full reduction uses natural axis order; an empty axis list or [scalar](../../../../glossary.md#term-scalar)
full product returns the restored `g` directly. This formula excludes the current [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix)
without division and is exact at any number of represented signed zeros. It also handles zero
extents structurally. [NaN](../../../../glossary.md#term-nan), [infinity](../../../../glossary.md#term-infinity), zero-times-infinity, [overflow](../../../../glossary.md#term-overflow), [underflow](../../../../glossary.md#term-underflow), sign, [rounding](../../../../glossary.md#term-rounding), and
[reassociation](../../../../glossary.md#term-reassociation) follow the ordinary `CUM_PROD` and multiplication contracts in formula order; no
[gradient](../../../../glossary.md#term-gradient)-only repair is added at those non-regular boundaries.

### Reduction [extrema](../../../../glossary.md#saturation-pole-and-extremum)

For floating reduction `MIN` or `MAX`, restore and expand both `g` and exact forward output `y`:

```text
matches = (x == E_x(y, A, keepDimensions))
tieCount = where(matches, O_x, Z_x).sum(A, true).expand(x.shape)
dx = where(matches, E_x(g, A, keepDimensions) / tieCount, Z_x)
```

All exact represented-numeric ties share the [cotangent](../../../../glossary.md#cotangent) equally. Numeric equality makes opposite
signed zeros ties; equal [finite](../../../../glossary.md#term-finite) values and equal same-sign [infinities](../../../../glossary.md#nan-infinity-and-signed-zero) also tie. If the forward
output is [NaN](../../../../glossary.md#term-nan), numeric equality is false at every [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) and the selected contribution is
exact [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero). Empty domains have no input positions. Integral extrema are
non-[differentiable](../../../../glossary.md#derivative-and-chain-rule).

### Cumulative scans

`CUM_SUM` preserves the current formula:

```text
dx = g.cumSum(axis, exclusive, !reverse)
```

For `CUM_PROD`, define:

```text
isZero = (x == Z_x)
safeX = where(isZero, O_x, x)
zeroPrefix = where(isZero, O_x, Z_x).cumSum(axis, exclusive, reverse)
safeProduct = safeX.cumProd(axis, exclusive, reverse)

q0 = where(zeroPrefix == Z_x, g * safeProduct, Z_x)
q1 = where(zeroPrefix == O_x, g * safeProduct, Z_x)

s0 = q0.cumSum(axis, exclusive, !reverse)
s1 = q1.cumSum(axis, exclusive, !reverse)

dx = where(isZero, s1, s0 / safeX)
```

The same `exclusive` flag and opposite traversal direction identify exactly the outputs whose
[scan](../../../../glossary.md#term-cumulative-scan) domain contains each input position. Replacing [signed zero](../../../../glossary.md#term-signed-zero) by positive one makes division
zero-safe and gives the exact [derivative](../../../../glossary.md#term-derivative) for zero-free prefixes, the unique-zero product for a
prefix with one zero, and zero for prefixes with more than one zero. Zero-length [axes](../../../../glossary.md#term-axis) remain
zero-length. [NaN](../../../../glossary.md#term-nan), [infinity](../../../../glossary.md#term-infinity), exceptional products, and floating [rounding](../../../../glossary.md#term-rounding) follow the ordinary
formula; division by infinity is not given a [gradient](../../../../glossary.md#term-gradient)-specific limiting repair.

### [Softmax](../../../../glossary.md#softmax--log-softmax) and [log-softmax](../../../../glossary.md#softmax--log-softmax)

Use the exact forward output `y`, never a separately recomputed softmax [occurrence](../../../../glossary.md#term-occurrence):

| Kind | Input contribution |
|---|---|
| `SOFTMAX` | `y * (g - (g * y).sum(axis, true))` |
| `LOG_SOFTMAX` | `g - exp(y) * g.sum(axis, true)` |

The formulas preserve [Shape](../../../../glossary.md#shape) and type. A zero-length [normalization](../../../../glossary.md#term-normalization) [axis](../../../../glossary.md#term-axis) has no [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) and
therefore produces a zero-length contribution. [Finite](../../../../glossary.md#term-finite) slices use the exact analytic
[vector](../../../../glossary.md#term-vector)-[Jacobian](../../../../glossary.md#term-jacobian) products. [NaN](../../../../glossary.md#term-nan), [infinity](../../../../glossary.md#term-infinity), [signed zero](../../../../glossary.md#term-signed-zero), [overflow](../../../../glossary.md#term-overflow), [underflow](../../../../glossary.md#term-underflow), and any forward
finite-[precision](../../../../glossary.md#term-precision) choice flow through the exact saved output and ordinary operations in formula
order. The [compiler](../../../../glossary.md#term-compiler) adds no forward softmax algorithm or exceptional-value rewrite.

### Advanced reductions and boundary policies

For every formula below, restore `g` and `y` through `E_x` as applicable.

| Kind | Exact input contribution and policy |
|---|---|
| `LOG_SUM_EXP` | `E_x(g) * exp(x - E_x(y))`. Non-empty [finite](../../../../glossary.md#term-finite) slices use the analytic formula. Empty slices have no input [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix). All-negative-[infinity](../../../../glossary.md#term-infinity), positive-infinity, [NaN](../../../../glossary.md#term-nan), [overflow](../../../../glossary.md#term-overflow), and [underflow](../../../../glossary.md#term-underflow) use the raw saved-output formula without a [compiler](../../../../glossary.md#term-compiler) [normalization](../../../../glossary.md#term-normalization) repair. |
| [`VARIANCE`](../../../../glossary.md#term-variance) | Let `mu=x.mean(A,true)`, `N=N_A(x)`, `D=N-Q_t(correction)`. Return `E_x(g) * 2 * (x-mu) / D.expand(x.shape)`. The forward `D>0` constraint remains authoritative. Valid [constant](../../../../glossary.md#term-constant) finite slices return exact zero. NaN and infinity use the raw formula. |
| `STANDARD_DEVIATION` | With the same `mu` and `D`, let `regular=E_x(g)*(x-mu)/(D.expand(x.shape)*E_x(y))`; return `where(E_x(y) > Z_x, regular, Z_x)`. A valid zero [standard deviation](../../../../glossary.md#term-standard-deviation) therefore has exact zero [cotangent](../../../../glossary.md#cotangent) at every coordinate. NaN output also selects zero; [positive infinity](../../../../glossary.md#nan-infinity-and-signed-zero) enters the raw branch. |
| `L1_NORM` | `where(x>Z_x, E_x(g), where(x<Z_x, -E_x(g), Z_x))`. Both signed zeros and NaN select exact zero; [infinities](../../../../glossary.md#nan-infinity-and-signed-zero) use their sign. |
| `L2_NORM` | `regular=E_x(g)*x/E_x(y)`; return `where(E_x(y)>Z_x, regular, Z_x)`. A zero [norm](../../../../glossary.md#term-norm) and NaN norm select exact zero for the complete [slice](../../../../glossary.md#slice); positive infinity uses the raw quotient. |

`Q_t(correction)` must use logical ones as defined above. Do not introduce an exact-integer
threshold, [host](../../../../glossary.md#term-host)-size calculation, physical constant, or correction-specific [model](../../../../glossary.md#term-model) kind.

### [Layer normalization](../../../../glossary.md#layer-normalization)

Let `A` be the natural trailing [axes](../../../../glossary.md#term-axis) named by `normalizedShape`, `P` the natural leading prefix
axes, `xT=T_y(x)`, and:

```text
mu = xT.mean(A, true)
centered = xT - mu
r = (centered * centered).mean(A, true).add(epsilon).rsqrt()
h = g                                      // no affine form
h = g * V(T_y(scale), x.shape, A)         // affine form

dxPromoted =
    r * (h
         - h.mean(A, true)
         - centered * r * r * (h * centered).mean(A, true))
dx = C_x(dxPromoted)
```

For the affine form:

```text
normalized = centered * r
dScale = C_scale((g * normalized).sum(P, false).reshape(scale.shape))
dBias  = C_bias(g.sum(P, false).reshape(bias.shape))
```

When `P` is empty, the point-domain reductions retain [Shape](../../../../glossary.md#shape) and are still ordinary expressions;
an implementation may omit only a provably redundant reshape. Scale and [bias](../../../../glossary.md#term-bias) Shapes equal the
exact [normalized Shape](../../../../glossary.md#normalized-shape); they are not arbitrary broadcast [parameters](../../../../glossary.md#term-parameter).

Positive [epsilon](../../../../glossary.md#epsilon) makes [finite](../../../../glossary.md#term-finite) zero-[variance](../../../../glossary.md#term-variance) slices regular. Empty outputs have no normalized
[coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix). [NaN](../../../../glossary.md#term-nan), [infinity](../../../../glossary.md#term-infinity), [overflow](../../../../glossary.md#term-overflow), [underflow](../../../../glossary.md#term-underflow), and [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero) behavior follows the recomputed
ordinary population-[mean](../../../../glossary.md#term-mean) formula; the [compiler](../../../../glossary.md#term-compiler) adds no [gradient](../../../../glossary.md#term-gradient)-only finite [mask](../../../../glossary.md#term-boolean-mask).

### [RMS normalization](../../../../glossary.md#rms-normalization)

With the same trailing `A` and leading `P` convention:

```text
xT = T_y(x)
r = (xT * xT).mean(A, true).add(epsilon).rsqrt()
h = g                                      // no-scale form
h = g * V(T_y(scale), x.shape, A)         // scaled form

dxPromoted = r * h - xT * r * r * r * (h * xT).mean(A, true)
dx = C_x(dxPromoted)
```

For the scaled form:

```text
dScale = C_scale((g * xT * r).sum(P, false).reshape(scale.shape))
```

Zero input with positive [epsilon](../../../../glossary.md#epsilon) is regular. Empty and exceptional-value policy is the same raw
ordinary-[Tensor](../../../../glossary.md#tensor) policy as [layer normalization](../../../../glossary.md#layer-normalization).

### [Batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-[normalization](../../../../glossary.md#term-normalization) [inference](../../../../glossary.md#term-inference)

Let `B` be every natural non-[channel](../../../../glossary.md#term-channel) [axis](../../../../glossary.md#term-axis), and align every [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) channel [vector](../../../../glossary.md#term-vector) with `V`:

```text
xT = T_y(x)
s = V(T_y(scale), x.shape, [channelAxis])
m = V(T_y(runningMean), x.shape, [channelAxis])
v = V(T_y(runningVariance), x.shape, [channelAxis])
r = (v + epsilon).rsqrt()
centered = xT - m
```

The selected contributions are:

```text
dx = C_x(g * s * r)
dScale = R_scale(g * centered * r, B)
dBias = R_bias(g, B)
dRunningMean = R_runningMean(-(g * s * r), B)
dRunningVariance =
    R_runningVariance(-0.5 * g * s * centered * r * r * r, B)
```

Negative radicands, zero denominators, [NaN](../../../../glossary.md#term-nan), [infinity](../../../../glossary.md#term-infinity), [overflow](../../../../glossary.md#term-overflow), and [rounding](../../../../glossary.md#term-rounding) use ordinary
`RSQRT` and arithmetic behavior. [Epsilon](../../../../glossary.md#epsilon) and channel axis receive no [cotangent](../../../../glossary.md#cotangent).

### [Batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-[normalization](../../../../glossary.md#term-normalization) training

[Preflight](../../../../glossary.md#term-preflight) must require exactly five outputs and retrieve:

```text
savedMean = producer.output(3)
savedInvStd = producer.output(4)
```

Those exact [canonical](../../../../glossary.md#term-canonical-form) wrappers are forward values of the selected [occurrence](../../../../glossary.md#term-occurrence). Do not reconstruct
a wrapper, recompute batch statistics, decompose the original forward occurrence, or create a
physical saved [buffer](../../../../glossary.md#term-buffer).

Let `B` be every natural non-[channel](../../../../glossary.md#term-channel) [axis](../../../../glossary.md#term-axis) and `xT=T_y(x)` for every selected public output slot.
For selected output slot 0, align `scale`, `savedMean`, and `savedInvStd` to the input [Shape](../../../../glossary.md#shape):

```text
centered = xT - V(savedMean)
r = V(savedInvStd)
h = g0 * V(T_y(scale))

dx = C_x(r * (h - h.mean(B,true)
                - centered * r * r * (h * centered).mean(B,true)))
dScale = R_scale(g0 * centered * r, B)
dBias = R_bias(g0, B)
```

Running [mean](../../../../glossary.md#term-mean) and running [variance](../../../../glossary.md#term-variance) receive no contribution from slot 0.

For selected next-running-mean slot 1:

```text
oneMinusMomentum = O_g1 - momentum
dRunningMean = C_runningMean(g1 * oneMinusMomentum)
N = N_B(xT)
dx = C_x(V(g1 * momentum) / N.expand(x.shape))
```

For selected next-running-variance slot 2:

```text
oneMinusMomentum = O_g2 - momentum
dRunningVariance = C_runningVariance(g2 * oneMinusMomentum)
N = N_B(xT)
dx = C_x(V(g2 * momentum) * 2
         * (xT - V(savedMean))
         / (N - O_N).expand(x.shape))
```

The existing forward predicate `channelExtent == 0 OR N >= 2` makes the correction-one
denominator valid whenever an input [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) exists. Slot 1/2 input contributions accumulate
with slot 0 when the [objective](../../../../glossary.md#term-objective) reaches multiple outputs. Scale and [bias](../../../../glossary.md#term-bias) receive contributions only
from slot 0. Running-[statistic](../../../../glossary.md#term-statistic) inputs receive contributions only from their matching transition
slot. Slots 3 and 4 cannot be independent first-order request roots in this task; they are
auxiliary forward values for slot-0/slot-2 formulas.

### Output-slot-aware [preflight](../../../../glossary.md#term-preflight) and accumulation

The current one-output behavior must remain unchanged. [Batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) training additionally requires
output-slot-aware route selection:

- inventory every exact output wrapper through [canonical](../../../../glossary.md#term-canonical-form) [producer](../../../../glossary.md#term-producer) [identity](../../../../glossary.md#term-identity-meanings);
- for one producer reached through multiple outputs, retain every objective-ancestry output slot
  rather than only the first encountered slot;<span data-glossary-navigation="true"> ([objective glossary](../../../../glossary.md#term-objective))</span>
- classify differentiable input roles by exact output slot before propagating target
  reachability;<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>
- intentionally non-[differentiable](../../../../glossary.md#derivative-and-chain-rule) roles do not create a false path through another output;
- an unknown kind, attributes variant, output slot, or ambiguous role fails closed;
- a target whose only structural paths use non-differentiable roles fails before derivative
  Tensor allocation with deterministic occurrence/output/input context; and<span data-glossary-navigation="true"> ([derivative glossary](../../../../glossary.md#term-derivative); [Tensor glossary](../../../../glossary.md#tensor); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- hidden slots 3/4 may be retrieved by an approved formula but are rejected as independent
  selected output roots.

Reverse accumulation order is:

1. reverse producer postorder;
2. selected output slots in ascending numeric order for that producer; and
3. input positions in ascending numeric order for that output.

This preserves every existing one-output contribution order. Multiple output slots append
separate contributions to a shared input and left-associate them through ordinary `Tensor.add`.
Repeated input positions remain repeated contributions.

### [Validation](../../../../glossary.md#term-validation) and failure order

Preserve top-level [argument](../../../../glossary.md#term-argument), [compile](../../../../glossary.md#compile)-mode, [objective](../../../../glossary.md#term-objective), forward-output membership, ingress,
[canonical](../../../../glossary.md#term-canonical-form)-wrapper, [target](../../../../glossary.md#term-target) [identity](../../../../glossary.md#term-identity-meanings), floating eligibility, and ancestry validation from [Compiler](../../../../glossary.md#term-compiler)
0004–0005A.

Before [seed](../../../../glossary.md#term-seed), [constant](../../../../glossary.md#term-constant), formula, or [derivative](../../../../glossary.md#term-derivative) [Tensor](../../../../glossary.md#tensor) construction, visit selected [producers](../../../../glossary.md#term-producer) in
deterministic [producer](../../../../glossary.md#term-producer) postorder and validate:

1. exact kind and attributes pairing;
2. exact input and output counts;
3. selected canonical output slot and its floating [gradient](../../../../glossary.md#term-gradient) eligibility;
4. output-slot-specific [differentiable](../../../../glossary.md#derivative-and-chain-rule) and non-differentiable roles in input-position order;
5. exact input floating eligibility and current ordered promotion;
6. normalized axes, ordered distinct axis lists, keep-dimension result Shape, masks, correction,
   normalized Shape, channel Shape, and result descriptors;<span data-glossary-navigation="true"> ([axes glossary](../../../../glossary.md#term-axis); [dimension glossary](../../../../glossary.md#dimension); [Shape glossary](../../../../glossary.md#shape); [masks glossary](../../../../glossary.md#term-boolean-mask); [normalized Shape glossary](../../../../glossary.md#normalized-shape); [channel glossary](../../../../glossary.md#term-channel); [descriptors glossary](../../../../glossary.md#term-descriptor))</span>
7. directly proved Shape relations or exact forward-predicate implication for every generated
   expand/reshape/sum-to-Shape;<span data-glossary-navigation="true"> ([sum-to-Shape glossary](../../../../glossary.md#sum-to-shape))</span>
8. availability and exact identity of any required same-[occurrence](../../../../glossary.md#term-occurrence) auxiliary output; and
9. the fixed formula and boundary-policy row selected above.

Known failure consumes no seed, [logical splat](../../../../glossary.md#logical-splat), formula Tensor, or derivative [`TensorId`](../../../../glossary.md#tensorid).
Successful [preflight](../../../../glossary.md#term-preflight) may be followed by ordinary public Tensor construction, [capture](../../../../glossary.md#term-capture-meanings), [inference](../../../../glossary.md#term-inference),
validation, optimization, [publication](../../../../glossary.md#term-publication), or [planning](../../../../glossary.md#term-planning) failure that consumes [opaque](../../../../glossary.md#term-opaque) non-reusable IDs.

Do not inspect Tensor payloads. Do not reject a represented value merely because it is zero, [NaN](../../../../glossary.md#term-nan),
infinite, negative, or exceptional under a raw formula. Forward constraints such as positive
statistical denominator and [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-training domain remain ordinary compiler predicates.

### [Producers](../../../../glossary.md#term-producer), [provenance](../../../../glossary.md#provenance), phase, [constants](../../../../glossary.md#term-constant), and shared pipeline

- Preserve exact [Tensor](../../../../glossary.md#tensor) and [`TensorProducer`](../../../../glossary.md#producer--tensorproducer) object identity in ancestry, output-slot,
  contribution, and accumulation bookkeeping.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-reference-identity))</span>
- Use `producer.output(3)` and `producer.output(4)` for [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-training saved values.
- Original [producer](../../../../glossary.md#term-producer) [identity](../../../../glossary.md#term-identity-meanings), including all five batch-training outputs, remains `FORWARD`.
  Every generated formula, alignment, logical-count, scan, reduction, comparison, and constant
  expansion producer is <span data-glossary-navigation="true"> ([alignment glossary](../../../../glossary.md#term-alignment); [scan glossary](../../../../glossary.md#term-cumulative-scan))</span>`BACKWARD`.
- [Capture](../../../../glossary.md#term-capture-meanings) ordered forward outputs and [gradient](../../../../glossary.md#term-gradient) roots together exactly once. Assign each
  [`NodeId`](../../../../glossary.md#nodeid) and [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) once and retain per-[node](../../../../glossary.md#node) `GraphPhase`.
- Multiple [target](../../../../glossary.md#term-target) roles may share one captured gradient `ValueId`; no identity node is added.
- Reuse the request-local exact typed-splat [cache](../../../../glossary.md#term-cache). Reuse the fixed typed `2` and `-0.5`
  coefficient bits established by Compiler 0005A. Scalar epsilon and momentum remain their exact
  operation attributes.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [Scalar glossary](../../../../glossary.md#term-scalar); [epsilon glossary](../../../../glossary.md#epsilon); [momentum glossary](../../../../glossary.md#term-momentum); [operation glossary](../../../../glossary.md#operation))</span>
- Only generated base constants reachable from returned gradient roots remain in combined
  ingress. Storage, labels, provenance absence, Shape, or factory history never imply constant
  status.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape))</span>
- Run the existing inference, validation, canonicalization, exact rewriting, constant folding,
  whole-graph dead-code elimination, phase-local common-subexpression elimination, cleanup, final
  validation, publication, and planning pipeline. Add no gradient-specific optimizer or pass.<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run); [inference glossary](../../../../glossary.md#term-inference); [validation glossary](../../../../glossary.md#term-validation); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [constant folding glossary](../../../../glossary.md#term-constant-folding); [graph glossary](../../../../glossary.md#graph); [dead-code elimination glossary](../../../../glossary.md#term-dead-code-elimination-dce); [common-subexpression elimination glossary](../../../../glossary.md#term-common-subexpression-elimination-cse); [publication glossary](../../../../glossary.md#term-publication); [planning glossary](../../../../glossary.md#term-planning); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- Compiler 0005C layout/window/indexing/scatter/ordering/stochastic completion beyond the narrow
  binding-aware <span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [binding glossary](../../../../glossary.md#term-binding))</span>`EXPAND` adoption required here
- [Compiler](../../../../glossary.md#term-compiler) 0005D attention/[convolution](../../../../glossary.md#term-channels-first-convolution-layers)/[pooling](../../../../glossary.md#term-pooling)/[loss](../../../../glossary.md#loss) work or Compiler 0005E closure/[checkpoint](../../../../glossary.md#term-checkpoint) work
- Compiler 0006 public objectives, targets, seeds, disconnected-result policy, create-graph,
  derivative order, higher derivatives, or order-aware phase representation<span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target); [seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [graph glossary](../../../../glossary.md#graph); [derivative glossary](../../../../glossary.md#term-derivative))</span>
- another model kind, Tensor method, attributes type, result carrier, public saved-statistic
  accessor, backward-only operation, or public compiler request/result surface<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [carrier glossary](../../../../glossary.md#term-storage-carrier); [statistic glossary](../../../../glossary.md#term-statistic); [operation glossary](../../../../glossary.md#operation))</span>
- [mutable](../../../../glossary.md#term-mutable) [Tensor](../../../../glossary.md#tensor) [gradients](../../../../glossary.md#term-gradient), `Tensor.backward`, a tape, physical saved buffers, recomputed batch
  saved values, direct captured-node formula construction, or a second algebra<span data-glossary-navigation="true"> ([buffers glossary](../../../../glossary.md#term-buffer); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [node glossary](../../../../glossary.md#node))</span>
- a public gradient registry, facade, policy object, configurable subgradient mode, generic
  algebra builder, or gradient-specific optimizer/folder/rewriter<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [facade glossary](../../../../glossary.md#term-facade); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- changing forward reduction, scan, softmax, statistics, normalization, epsilon, momentum,
  correction, comparison, signed-zero, NaN, infinity, accumulation, or layout semantics<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [softmax glossary](../../../../glossary.md#softmax--log-softmax); [normalization glossary](../../../../glossary.md#term-normalization); [epsilon glossary](../../../../glossary.md#epsilon); [momentum glossary](../../../../glossary.md#term-momentum); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity))</span>
- model, config, planning, trace, runtime, prepare, engine, extension, backend, storage,
  execution, lowering, kernel, physical memory, schedule, or optimizer-update behavior<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [trace glossary](../../../../glossary.md#trace); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [kernel glossary](../../../../glossary.md#kernel))</span>
- module dependencies, Gradle, Java version, architecture contracts, ADRs, architecture tests,
  backend-conformance tests, or integration tests<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- unrelated refactoring or absorbing any part of 0005C–0005E or 0006

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Runtime / Prepare / Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Backend glossary](../../../../glossary.md#backend))</span>
- [ADR 0009](../../../../design/decisions/0009-compiler-owned-pre-capture-tensor-expression-autograd.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- [Autograd strategy](../../../../design/notes/autograd-strategy.md)<span data-glossary-navigation="true"> ([Autograd glossary](../../../../glossary.md#autograd))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Roadmap](../../../roadmap.md)
- [Compiler master plan](../master-plan.md)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>
- [Model master plan](../../model/master-plan.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Model capabilities](../../model/capabilities.md)
- [Adjoint expressibility audit](../../model/adjoint-expressibility-audit.md)<span data-glossary-navigation="true"> ([Adjoint glossary](../../../../glossary.md#term-adjoint))</span>
- [Model contract-closure audit](../../model/model-capability-contract-closure-audit.md)
- [Model 0023A](../../model/tasks/0023a-binding-aware-sum-to-shape.md)
- [Model 0023E](../../model/tasks/0023e-cumulative-scan-normalization-and-product.md)
- [Model 0025](../../model/tasks/0025-canonical-tensor-producer-outputs.md)
- [Model 0025B](../../model/tasks/0025b-binding-aware-expansion.md)
- [Compiler 0002](0002-captured-graph-inference-and-validation.md)
- [Compiler 0004](0004-compiler-owned-pre-capture-autograd-and-combined-graph-compilation.md)
- [Compiler 0004A](0004a-exact-composition-gradient-rule-extensions.md)
- [Compiler 0004B](0004b-shared-algebra-cotangent-normalization-and-local-derivative-rules.md)
- [Compiler 0005](0005-publication-planning-orchestration-and-compile-artifacts.md)
- [Compiler 0005A](0005a-derivative-policy-and-elementwise-activation-gradient-completion.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- Compiler owns fail-closed preflight, named gradient-rule dispatch, ephemeral identity
  accumulation, phase-aware combined capture, graph validation, exact optimization, and
  compile-artifact orchestration.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [fail-closed glossary](../../../../glossary.md#term-fail-closed); [preflight glossary](../../../../glossary.md#term-preflight); [gradient glossary](../../../../glossary.md#term-gradient); [dispatch glossary](../../../../glossary.md#term-dispatch); [identity glossary](../../../../glossary.md#term-identity-meanings); [capture glossary](../../../../glossary.md#term-capture-meanings); [graph glossary](../../../../glossary.md#graph); [validation glossary](../../../../glossary.md#term-validation); [compile glossary](../../../../glossary.md#compile); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Model owns public Tensor algebra, immutable operation semantics, canonical producer output
  wrappers, and exact provenance. It owns no derivative rule or graph predicate.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Tensor glossary](../../../../glossary.md#tensor); [immutable glossary](../../../../glossary.md#term-immutable); [operation glossary](../../../../glossary.md#operation); [canonical glossary](../../../../glossary.md#term-canonical-form); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance); [derivative glossary](../../../../glossary.md#term-derivative))</span>
- Generated gradients are ordinary Tensor expressions and use exactly the forward
  inference/validation/numerical/optimization contracts.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference))</span>
- Deferred constraints remain logical compile obligations. They are not bindings, physical
  values, runtime checks implemented by this task, or execution permission.<span data-glossary-navigation="true"> ([bindings glossary](../../../../glossary.md#term-binding); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Same-occurrence batch saved statistics are logical forward values. Their use creates no tape,
  recomputation policy, buffer, publication requirement, or runtime ownership.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [buffer glossary](../../../../glossary.md#term-buffer); [publication glossary](../../../../glossary.md#term-publication); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- The compiler remains independent of runtime, prepare, engine, extensions, and concrete
  backends.<span data-glossary-navigation="true"> ([prepare glossary](../../../../glossary.md#prepare); [engine glossary](../../../../glossary.md#term-engine); [backends glossary](../../../../glossary.md#backend))</span>
- If implementation needs a model/public API, new dependency, architecture change, physical
  saved-value contract, or formula outside the current Tensor algebra, stop and report the exact
  blocker.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Package impact

Existing package retained:

```text
io.github.pho001.synaptik.compiler
  package-private inference, preflight, gradient formulas, accumulation, and combined capture
```

One package-private type may be added:

- `io.github.pho001.synaptik.compiler.NormalizationGradientRules` — owns Layer/RMS/batch
  normalization formulas so <span data-glossary-navigation="true"> ([batch normalization glossary](../../../../glossary.md#batch-normalization--batch-normalization-inference-and-training))</span>`ReductionGradientRules` remains focused on aggregate reductions,
  statistics, softmax, and scans.<span data-glossary-navigation="true"> ([softmax glossary](../../../../glossary.md#softmax--log-softmax))</span>

No public type or package is added, moved, or widened.

## Affected files

Production and [Javadoc](../../../../glossary.md#term-javadoc) (5):

- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/AutogradPreflight.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/FirstOrderAutograd.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/LayoutInference.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ReductionGradientRules.java`
- add `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/NormalizationGradientRules.java`

Tests (5):

- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/AutogradPreflightTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/FirstOrderAutogradTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/GradientRulesTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/GraphCompilerTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/LayoutInferenceTest.java`

Documentation and [planning](../../../../glossary.md#term-planning) (8):

- `docs/api/compile-api.md`
- `docs/design/notes/autograd-strategy.md`
- `docs/user-guide/autograd.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/compiler/master-plan.md`
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review unchanged unless evidence contradicts this specification:

- `ARCHITECTURE.md`, focused architecture pages, and [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0009
- [Tensor](../../../../glossary.md#tensor) and Training APIs
- Model kinds, attributes, Tensor helpers, producer/provenance, source/tests/Javadocs, tasks,
  capabilities, and audits<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>
- compiler inference/constraint/optimization/publication/planning sources and tests outside the
  exact paths above<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [inference glossary](../../../../glossary.md#term-inference); [publication glossary](../../../../glossary.md#term-publication))</span>
- Config, Planning, Trace, Runtime, Prepare, Engine, extensions, backends, architecture tests,
  backend conformance, integration tests, Gradle, dependencies, and Java 26 configuration<span data-glossary-navigation="true"> ([Trace glossary](../../../../glossary.md#trace); [Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [backends glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle))</span>

## Maximum [scope](../../../../glossary.md#term-scope)

At most the exact eighteen paths listed above may change: five [compiler](../../../../glossary.md#term-compiler) production/[Javadoc](../../../../glossary.md#term-javadoc) paths,
five compiler test paths, and eight documentation/[planning](../../../../glossary.md#term-planning) paths.

The task may add only `NormalizationGradientRules`. Stop before a nineteenth path, a second type,
a public surface, another [module](../../../../glossary.md#term-module), a [model](../../../../glossary.md#term-model)/test/[Gradle](../../../../glossary.md#term-gradle)/architecture change, or any 0005C–0005E/0006
task file. If an assigned formula cannot be completed inside this boundary, report the exact
missing primitive, forward policy, type, or architecture decision instead of silently omitting
the role.

## Acceptance criteria

- Source-backed inventory tests lock all fourteen aggregate kinds and every attributes/signature
  variant, both scan kinds, both softmax kinds, Layer/RMS/batch kinds, all five batch-training
  output slots, and every differentiable/non-differentiable role assigned above.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [scan glossary](../../../../glossary.md#term-cumulative-scan); [softmax glossary](../../../../glossary.md#softmax--log-softmax); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [differentiable glossary](../../../../glossary.md#derivative-and-chain-rule))</span>
- `LayoutInference` accepts [Model](../../../../glossary.md#term-model)-0025B [binding](../../../../glossary.md#term-binding)-aware `EXPAND`, emits exact ordered
  source-one-or-source-equal predicates, rejects contradictions, preserves static layout, and
  retains unresolved layout and deferred obligations without binding.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout))</span>
- [Preflight](../../../../glossary.md#term-preflight) admits binding-dependent `EXPAND` and `SUM_TO_SHAPE` inverses only through exact
  predicate-equivalence/implication proof and rejects any differently aligned or unsupported
  obligation before derivative allocation.<span data-glossary-navigation="true"> ([derivative glossary](../../../../glossary.md#term-derivative))</span>
- Existing `SUM`, [`MEAN`](../../../../glossary.md#term-mean), [masked reduction](../../../../glossary.md#masked-reduction), `SUM_TO_SHAPE`, and `CUM_SUM` formulas and policies
  remain unchanged for their previously supported cases.
- Product reduction uses sequential exclusive prefix/suffix `CUM_PROD` formulas without division
  and covers full, scalar, single-axis, ordered multi-axis, keep/remove-dimension, empty-axis-list,
  zero-extent, one-zero, and multiple-zero structure.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [axis glossary](../../../../glossary.md#term-axis); [dimension glossary](../../../../glossary.md#dimension); [extent glossary](../../../../glossary.md#term-extent))</span>
- `CUM_PROD` implements the exact zero-safe formula for all four exclusive/reverse modes and
  covers signed zero, one zero, multiple zeros, zero-length axes, and raw NaN/infinity structure.<span data-glossary-navigation="true"> ([signed zero glossary](../../../../glossary.md#term-signed-zero); [NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity))</span>
- Reduction extrema split exact represented ties equally and lock finite, signed-zero, infinity,
  NaN, empty, axis, and keep-dimension policy.<span data-glossary-navigation="true"> ([extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [finite glossary](../../../../glossary.md#term-finite); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Softmax, log-softmax, log-sum-exp, variance, standard deviation, L1 norm, and L2 norm implement
  the exact formulas, correction construction, count semantics, and zero/NaN/infinity policies
  above.<span data-glossary-navigation="true"> ([log-softmax glossary](../../../../glossary.md#softmax--log-softmax); [log-sum-exp glossary](../../../../glossary.md#log-sum-exp); [variance glossary](../../../../glossary.md#term-variance); [standard deviation glossary](../../../../glossary.md#term-standard-deviation); [L1 norm glossary](../../../../glossary.md#norms-elementary-functions-and-probability-distributions); [L2 norm glossary](../../../../glossary.md#norms-elementary-functions-and-probability-distributions))</span>
- Layer and RMS normalization cover no-affine/affine forms, exact trailing axes, prefix
  reductions, epsilon, mixed-floating normalization, zero/empty/exceptional policy, and exact
  scale/bias Shapes.<span data-glossary-navigation="true"> ([RMS normalization glossary](../../../../glossary.md#rms-normalization); [epsilon glossary](../../../../glossary.md#epsilon); [normalization glossary](../../../../glossary.md#term-normalization); [bias glossary](../../../../glossary.md#term-bias); [Shapes glossary](../../../../glossary.md#shape))</span>
- Batch inference covers all five inputs, arbitrary normalized channel axis, vector alignment,
  non-channel reductions, mixed-floating normalization, and raw radicand/special-value policy.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [channel glossary](../../../../glossary.md#term-channel); [vector glossary](../../../../glossary.md#term-vector); [alignment glossary](../../../../glossary.md#term-alignment))</span>
- Batch training covers output slots 0–2, exact canonical saved slots 3/4, slot-specific roles,
  momentum, biased versus correction-one formulas, channel-empty/domain constraints, and
  deterministic accumulation across multiple reached output slots. Saved values are never
  recomputed or materialized by compiler policy.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [momentum glossary](../../../../glossary.md#term-momentum); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- BOOL reductions, arg extrema, integral results/scans, masks, saved auxiliary roots, axes,
  correction, epsilon, momentum, and other configuration roles remain non-differentiable.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [masks glossary](../../../../glossary.md#term-boolean-mask))</span>
- Every contribution has the selected input's exact Shape and floating type before accumulation.
  Unresolved Shape normalization is accepted only when implied by the exact forward predicate.
- Known unsupported kinds, attributes, counts, output slots, roles, Shapes, types, or policies fail
  before seed/constant/formula allocation with deterministic occurrence/output/input context.<span data-glossary-navigation="true"> ([seed glossary](../../../../glossary.md#term-seed); [constant glossary](../../../../glossary.md#term-constant); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Reverse accumulation uses reverse producer postorder, ascending selected output slots, ascending
  input positions, repeated contributions, left-associated <span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer))</span>`Tensor.add`, and exact object
  identity.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-reference-identity))</span>
- One combined capture preserves canonical wrappers, all batch output slots, original/generated
  producer phase, unique graph-local IDs, target roles, explicit reachable constants, and the
  existing immutable graph/artifact boundary.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [graph glossary](../../../../glossary.md#graph); [target glossary](../../../../glossary.md#term-target); [immutable glossary](../../../../glossary.md#term-immutable); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Generated formulas pass the shared inference/validation and exact optimization pipeline. No
  gradient-specific inference system, simplifier, constant folder, rewrite, or optimizer exists.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [gradient glossary](../../../../glossary.md#term-gradient); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- No model/public API, dependency, build, architecture, backend, prepare, runtime, training,
  physical-value, or higher-order contract changes.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [dependency glossary](../../../../glossary.md#term-dependency); [backend glossary](../../../../glossary.md#backend); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Compile API, autograd strategy/user guide, glossary, task, both master plans, and roadmap
  consistently distinguish completed 0005B behavior from planned 0005C–0005E/0006 behavior.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [autograd glossary](../../../../glossary.md#autograd))</span>
- Tensor API needs no change because no public Tensor declaration or Model forward semantic
  changes. Training API needs no change because no optimizer/session/module/public training
  workflow changes. The documentation pass records both reasons.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [module glossary](../../../../glossary.md#term-module))</span>
- Exactly the eighteen authorized paths change. Only 0005B becomes Complete after implementation;
  0005C–0005E and 0006 remain concise Draft rows without detailed specifications.
- A separate clean-context documentation-focused pass independently finalizes affected Javadocs,
  explanatory documentation, glossary impact, planning status/evidence, terminology, links, and
  rendered Javadoc in the same overall change.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

During implementation, [run](../../../../glossary.md#term-run) focused suites as needed:

```bash
./gradlew :modules:compiler:test \
  --tests io.github.pho001.synaptik.compiler.LayoutInferenceTest \
  --tests io.github.pho001.synaptik.compiler.AutogradPreflightTest \
  --tests io.github.pho001.synaptik.compiler.GradientRulesTest \
  --tests io.github.pho001.synaptik.compiler.FirstOrderAutogradTest \
  --tests io.github.pho001.synaptik.compiler.GraphCompilerTest
```

After executable Java stabilizes, run one final [compiler](../../../../glossary.md#term-compiler) [module](../../../../glossary.md#term-module) suite:

```bash
./gradlew :modules:compiler:test
```

Record [JUnit](../../../../glossary.md#term-junit) XML suite/test counts and zero skipped/failure/error outcomes. Hand the exact
executable diff and test evidence to the documentation-focused pass. That pass must not repeat
successful Java tests unless executable behavior changes afterward or it records a concrete
stale-evidence risk.

Documentation pass:

```bash
./gradlew :modules:compiler:javadoc
python3 /tmp/validate_synaptik_markdown.py
git diff --check
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git status --short
```

The documentation pass must additionally inspect generated package-private Javadocs; verify all
[parameters](../../../../glossary.md#term-parameter), results, expected failures, [ownership](../../../../glossary.md#term-ownership-meanings), output-slot, ordering, and [lifecycle](../../../../glossary.md#lifecycle) contracts;
check local links/anchors, balanced fences, final newlines, and terminology; verify exact
eighteen-path [scope](../../../../glossary.md#term-scope); lock the source-backed inventory and formula [constants](../../../../glossary.md#term-constant); confirm no
[model](../../../../glossary.md#term-model)/test/[Gradle](../../../../glossary.md#term-gradle)/[dependency](../../../../glossary.md#term-dependency)/architecture change; synchronize 0005B Complete in task/master/
roadmap; confirm exactly one current detailed compiler task; and verify that no detailed
0005C–0005E/0006 specification exists.

Repository-wide validation and the first-order capability [checkpoint](../../../../glossary.md#term-checkpoint) remain deferred to Compiler
0005E or [continuous integration](../../../../glossary.md#term-continuous-integration-ci). This task changes one module's package-private compiler behavior
and no dependency, build, architecture boundary, or public [API](../../../../glossary.md#term-application-programming-interface-api).

## Dependencies

- [Model](../../../../glossary.md#term-model) 0025B [binding](../../../../glossary.md#term-binding)-aware expansion — Complete.
- Model 0025 [canonical](../../../../glossary.md#term-canonical-form) [producer](../../../../glossary.md#term-producer) outputs — Complete.
- Model 0023A binding-aware [sum-to-Shape](../../../../glossary.md#sum-to-shape) and Model 0023E cumulative product — Complete.
- Model reduction, scan, softmax, statistics, Layer/RMS/batch normalization tasks and the
  adjoint/closure audits — Complete.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [softmax glossary](../../../../glossary.md#softmax--log-softmax); [batch normalization glossary](../../../../glossary.md#batch-normalization--batch-normalization-inference-and-training); [adjoint glossary](../../../../glossary.md#term-adjoint))</span>
- Compiler 0001–0005A, including shared Shape/type normalization, exact splats, fail-closed
  preflight, canonical outputs, one combined capture, inference/constraints, exact optimization,
  publication, planning, and artifacts — Complete.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [Shape glossary](../../../../glossary.md#shape); [normalization glossary](../../../../glossary.md#term-normalization); [fail-closed glossary](../../../../glossary.md#term-fail-closed); [preflight glossary](../../../../glossary.md#term-preflight); [capture glossary](../../../../glossary.md#term-capture-meanings); [inference glossary](../../../../glossary.md#term-inference); [publication glossary](../../../../glossary.md#term-publication); [planning glossary](../../../../glossary.md#term-planning); [artifacts glossary](../../../../glossary.md#term-artifact))</span>
- Existing public Tensor algebra is sufficient. The representability re-audit found no missing
  primitive, unresolved required forward policy, new type, module, dependency, or architecture
  decision.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [module glossary](../../../../glossary.md#term-module))</span>

No Config, [Trace](../../../../glossary.md#trace), [Runtime](../../../../glossary.md#term-runtime), [Prepare](../../../../glossary.md#prepare), [Engine](../../../../glossary.md#term-engine), Training, [backend](../../../../glossary.md#backend), or new Model task is a [dependency](../../../../glossary.md#term-dependency).

## Follow-up tasks

- Compiler 0005C remains Draft and follows 0005B. It owns remaining layout/window/indexing/
  scatter/ordering/stochastic roles.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [layout glossary](../../../../glossary.md#layout))</span>
- Compiler 0005D remains Draft and follows 0005B/0005C. It owns attention, convolution, pooling,
  and loss roles.<span data-glossary-navigation="true"> ([convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [pooling glossary](../../../../glossary.md#term-pooling); [loss glossary](../../../../glossary.md#loss))</span>
- Compiler 0005E remains Draft and follows 0005A–0005D. It owns the complete source-backed
  first-order role/output closure audit, transitive formula-operation closure, and capability
  checkpoint.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Compiler 0006 remains Draft and follows 0005E plus the stable compile/artifact boundary. It owns
  public functional requests, seeds, disconnected results, derivative order, and higher-order
  differentiation.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [artifact glossary](../../../../glossary.md#term-artifact); [seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [derivative glossary](../../../../glossary.md#term-derivative))</span>

Do not create those detailed specifications during this task.

## Architecture impact

Expected impact: None.

This task fills the established [compiler](../../../../glossary.md#term-compiler)-owned first-order extension point, adopts an already
completed [Model](../../../../glossary.md#term-model) semantic contract through the existing predicate vocabulary, and uses [canonical](../../../../glossary.md#term-canonical-form)
[producer](../../../../glossary.md#term-producer) outputs through the existing one-[capture](../../../../glossary.md#term-capture-meanings) [lifecycle](../../../../glossary.md#lifecycle). It changes no authority, [ownership](../../../../glossary.md#term-ownership-meanings),
[dependency direction](../../../../glossary.md#java-platform-and-build-tools), public lifecycle, Model forward meaning, or [artifact](../../../../glossary.md#term-artifact) [shape](../../../../glossary.md#shape).

If implementation requires a new model/public [operation](../../../../glossary.md#operation), a changed forward policy, a physical
saved value, a [runtime](../../../../glossary.md#term-runtime) tape, direct [graph](../../../../glossary.md#graph)-[node](../../../../glossary.md#node) algebra, another [dependency](../../../../glossary.md#term-dependency), or an architecture
change, stop and report the exact conflict.

## Implementation prompt

Use this prompt in a separate agentic task/thread with clean context:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, the compiler/model master plans,
and docs/planning/modules/compiler/tasks/0005b-reduction-scan-softmax-statistics-and-normalization-gradient-completion.md.
Read every directly referenced architecture, completed compiler/model task and audit, final
compiler/model source and tests, Tensor/Compile/Training APIs, autograd strategy/user guide, and
glossary needed to verify the exact current inventory and shared Tensor algebra.

Implement Compiler 0005B exactly within its eighteen authorized paths. Preserve completed
0004–0005A behavior, request, fail-closed preflight, identity accumulation, one combined capture,
phase, validation, exact optimization, publication, and artifact contracts. Adopt only the narrow
binding-aware EXPAND inference/preflight work assigned here and implement every exact reduction,
scan, softmax, statistics, Layer/RMS/batch-normalization role and policy in the task. Use exact
same-occurrence batch saved outputs. Do not implement 0005C–0005E or 0006, add a model/public API,
second algebra/registry/facade/policy mode, recompute saved batch values, add gradient-specific
optimization, or touch Gradle, architecture, another module, backend, prepare, runtime, or
training behavior. Stop on any scope, inventory, formula-expressibility, forward-policy, or
architecture conflict.

After executable Java stabilizes, run focused tests as needed and one final compiler module suite.
Hand the actual diff and exact test evidence to a separate documentation-focused agent or thread
with clean context. That pass must follow docs/developer-guide/documentation-rules.md and
independently finalize affected Javadocs, Compile API, autograd strategy/user guide, glossary,
planning status/evidence, terminology, links, and rendered Javadoc in the same overall change. It
must record reasoned no-change conclusions for Tensor/Training APIs, Model capabilities/contracts,
architecture/tests, conformance/integration, Gradle, dependencies, and other modules, and must not
repeat successful Java tests unless executable behavior changes or a concrete stale-evidence risk
is recorded.

Run the final Javadoc/Markdown/scope/status/whitespace checks. Update this task with local
decisions, exact evidence, implementation notes, completion summary, and final status. Do not mark
it Complete before every acceptance criterion and the documentation-focused pass finish.
```

## Local decisions

- The public Tensor algebra is sufficient after Model 0025B. No model prerequisite, public API,
  new operation kind, or architecture change is required.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Model glossary](../../../../glossary.md#term-model); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [operation glossary](../../../../glossary.md#operation))</span>
- [Binding](../../../../glossary.md#term-binding)-aware `EXPAND` reuses `AnyOf` and `DimensionEqual`; no predicate or binding type is
  added.
- Product reduction uses sequential single-axis keep-dimension products and exclusive
  prefix/suffix scans. This avoids both division by zero and a flattened selected-domain extent
  that could overflow <span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [dimension glossary](../../../../glossary.md#dimension); [extent glossary](../../../../glossary.md#term-extent); [overflow glossary](../../../../glossary.md#term-overflow))</span>`long`.
- Cumulative-product differentiation uses safe products, cumulative zero counts, and
  opposite-direction cumulative sums. It is zero-safe without claiming an infinity-safe limiting
  extension.<span data-glossary-navigation="true"> ([infinity glossary](../../../../glossary.md#term-infinity))</span>
- Extrema use equal tie sharing; standard deviation and L2 norm select zero at zero result; L1
  selects zero at signed zero; raw saved-output formulas govern the explicitly listed exceptional
  cases.<span data-glossary-navigation="true"> ([Extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [standard deviation glossary](../../../../glossary.md#term-standard-deviation); [L2 norm glossary](../../../../glossary.md#norms-elementary-functions-and-probability-distributions); [L1 glossary](../../../../glossary.md#term-l1); [signed zero glossary](../../../../glossary.md#term-signed-zero))</span>
- [Statistical correction](../../../../glossary.md#statistical-correction) is a logical sum of exact typed ones, not a [host](../../../../glossary.md#term-host) floating conversion.
- Layer/RMS formulas use population means directly, avoiding a host or separately represented
  cardinality.<span data-glossary-navigation="true"> ([cardinality glossary](../../../../glossary.md#term-cardinality))</span>
- Batch training is output-slot-aware. Slots 3/4 remain exact auxiliary forward values, and slots
  0–2 contribute through their exact role matrices.<span data-glossary-navigation="true"> ([Batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [matrices glossary](../../../../glossary.md#term-matrix-meanings))</span>
- `NormalizationGradientRules` is the sole new package-private type and prevents one broad
  reduction/normalization rule owner.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [owner glossary](../../../../glossary.md#term-owner))</span>

## Known limitations

- The internal request remains one scalar objective with an implicit unit seed. Public requests,
  caller seeds, disconnected gradients, create-graph, and derivative order wait for Compiler
  0006.<span data-glossary-navigation="true"> ([scalar objective glossary](../../../../glossary.md#loss); [seed glossary](../../../../glossary.md#term-seed); [seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [gradients glossary](../../../../glossary.md#term-gradient); [graph glossary](../../../../glossary.md#graph); [derivative glossary](../../../../glossary.md#term-derivative))</span>
- Formula-structure tests do not prove backend numerical execution. Backend implementations later
  need conformance coverage for the selected zero, tie, correction, exceptional-value, and
  normalization behavior.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- Deferred Shape predicates remain compile obligations. This task does not implement concrete
  dimension binding, preparation, lowering, or execution.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [compile glossary](../../../../glossary.md#compile); [dimension glossary](../../../../glossary.md#dimension); [binding glossary](../../../../glossary.md#term-binding); [lowering glossary](../../../../glossary.md#lowering))</span>
- Raw exceptional-value formulas intentionally can produce NaN or infinity where the selected
  policy says to use ordinary Tensor arithmetic. Only the explicit zero/tie cases above receive a
  local repair.<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Complete current-[model](../../../../glossary.md#term-model) first-order coverage waits for [Compiler](../../../../glossary.md#term-compiler) 0005C–0005E.

## [Validation](../../../../glossary.md#term-validation) evidence

Implementation evidence reused by the replacement documentation pass:

- The implementation pass ran compiler main/test compilation successfully after the final
  executable edits.<span data-glossary-navigation="true"> ([compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names))</span>
- Its focused five-class command covering `LayoutInferenceTest`, `AutogradPreflightTest`,
  `GradientRulesTest`, `FirstOrderAutogradTest`, and `GraphCompilerTest` passed 60 tests.
- Its one final `./gradlew :modules:compiler:test` passed 22 suites and 170 tests with 0 skipped,
  0 failures, and 0 errors.
- Executable Java did not change after that evidence. Replacement clean documentation context
  `/root/docs_compiler_0005b_replacement` changed Javadocs and documentation only, so it reused
  the successful Java evidence and did not duplicate the test suite.

Documentation and final-gate evidence:

- The replacement clean documentation context independently reviewed `AGENTS.md`, the complete
  architecture contract, focused compiler/model/dependency/lifecycle/training/prepare
  architecture pages, General/API-Javadoc/User Guide/Planning/Example profiles, planning guide,
  roadmap, Compiler/Model master plans, Compiler 0005A, Model 0025B, the final five-production/
  five-test diff, generated/current compiler Javadocs, Compile/Tensor/Training APIs, autograd
  strategy/user guide, glossary, and current operation/constraint/provenance contracts.<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract); [dependency glossary](../../../../glossary.md#term-dependency); [lifecycle glossary](../../../../glossary.md#lifecycle); [prepare glossary](../../../../glossary.md#prepare); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc); [Planning glossary](../../../../glossary.md#term-planning); [Compile glossary](../../../../glossary.md#compile); [Tensor glossary](../../../../glossary.md#tensor); [autograd glossary](../../../../glossary.md#autograd); [operation glossary](../../../../glossary.md#operation); [provenance glossary](../../../../glossary.md#provenance))</span>
- `./gradlew :modules:compiler:javadoc` passed after final Javadoc edits in 1 second with seven
  actionable tasks: two executed and five up-to-date. It emitted no warnings. Generated
  package-private pages for `AutogradPreflight`, `FirstOrderAutograd`, `LayoutInference`,
  `ReductionGradientRules`, and `NormalizationGradientRules` were inspected for parameter,
  result, failure, identity, ownership, output-slot, ordering, and lifecycle contracts.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [identity glossary](../../../../glossary.md#term-identity-meanings); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- `python3 /tmp/validate_synaptik_markdown.py` passed across 12 Markdown files and 702 local
  links, including local file targets, heading anchors, balanced fences, final newlines, and
  trailing whitespace.<span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target))</span>
- The user-guide example remains explicitly conceptual because no public objective/target compile
  API exists. A runnable public example was therefore neither possible nor required; the focused
  structural tests and generated-document inspection validate the current package-private
  behavior.<span data-glossary-navigation="true"> ([objective glossary](../../../../glossary.md#term-objective))</span>
- `git diff --check` passed with no output.
- The sorted union of tracked and untracked changed paths contains exactly the authorized
  eighteen paths: five compiler production/Javadoc files, five compiler tests, and eight
  documentation/planning files.
- `javap` and manual source checks confirmed that all five affected compiler production types
  remain package-private; exactly one new package-private `NormalizationGradientRules` type was
  added; the Tensor surface remains at 200 public members with exactly the existing two `expand`
  overloads; `CompileArtifacts`, `BatchNormTrainingResult`, [`TensorProducer`](../../../../glossary.md#producer--tensorproducer), and the empty
  `TrainingModule` placeholder retain their public surfaces; no public compiler, Tensor, or
  Training declaration changed; and the fixed typed coefficient values remain `2` and `-0.5`
  with the recorded BFLOAT16/FLOAT32/FLOAT64 bits.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64))</span>
- Manual source-backed inventory review confirmed exactly fourteen aggregate, two scan, two
  softmax, one Layer, one RMS, and two batch-normalization kinds; batch training retains exactly
  five output slots and the implemented public-slot role matrix. The enum-array and focused tests
  lock the current inventory.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [softmax glossary](../../../../glossary.md#softmax--log-softmax); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [normalization glossary](../../../../glossary.md#term-normalization); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- Final `AutogradPreflight` source/Javadoc review confirmed that ordinary selected occurrences
  retain their one-output guard, batch training retains its exact five-output guard, reduction/
  scan/normalization inference re-derives the complete output count and every descriptor, and
  saved batch slots three and four are rejected both as requested targets and selected cotangent
  roots before derivative construction.<span data-glossary-navigation="true"> ([guard glossary](../../../../glossary.md#term-guard); [inference glossary](../../../../glossary.md#term-inference); [descriptor glossary](../../../../glossary.md#term-descriptor); [cotangent glossary](../../../../glossary.md#cotangent); [derivative glossary](../../../../glossary.md#term-derivative))</span>
- Manual formula review confirmed division-free product reduction, zero-safe cumulative product,
  equal-tie extrema, saved-output softmax/log-softmax, logical-count/correction construction,
  zero-result standard-deviation/L2 and signed-zero/NaN L1 policies, mixed-floating Layer/RMS/
  batch formulas, exact same-occurrence batch saved outputs, ascending output-slot accumulation,
  request-local splats, one combined phase-aware capture, and the shared inference/validation/
  optimization pipeline.<span data-glossary-navigation="true"> ([extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [log-softmax glossary](../../../../glossary.md#softmax--log-softmax); [L2 glossary](../../../../glossary.md#term-l2); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [NaN glossary](../../../../glossary.md#term-nan); [L1 glossary](../../../../glossary.md#term-l1); [occurrence glossary](../../../../glossary.md#term-occurrence); [capture glossary](../../../../glossary.md#term-capture-meanings))</span>
- Task, [Compiler](../../../../glossary.md#term-compiler) master plan, [Model](../../../../glossary.md#term-model) master plan, and roadmap are synchronized to Compiler 0005B
  `Complete`. Compiler 0005C–0005E and 0006 remain concise `Draft` rows without detailed task
  specifications, and no later compiler row is [`Ready`](../../../../glossary.md#term-ready).

Reasoned no-change conclusions:

- Tensor API and Model public/Javadoc contracts remain unchanged because 0005B adds no public
  Tensor declaration, operation kind, attributes, result carrier, forward semantic, descriptor,
  provenance, or storage behavior; the compiler consumes completed Model 0025B and canonical
  producer outputs unchanged.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [canonical glossary](../../../../glossary.md#term-canonical-form); [producer glossary](../../../../glossary.md#term-producer))</span>
- Training API and Public APIs remain unchanged because there is still no public gradient
  request, seed, publication workflow, module/parameter/session contract, optimizer update,
  preparation, execution, or run result.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [seed glossary](../../../../glossary.md#term-seed); [publication glossary](../../../../glossary.md#term-publication); [module glossary](../../../../glossary.md#term-module); [optimizer glossary](../../../../glossary.md#term-optimizer); [run result glossary](../../../../glossary.md#run-result--runresult))</span>
- Model capabilities, adjoint/closure audits, related operation contracts, and Model tests need
  no change because every selected Model capability and forward boundary was already complete;
  0005B changes compiler-owned derivative construction only.<span data-glossary-navigation="true"> ([adjoint glossary](../../../../glossary.md#term-adjoint))</span>
- `ARCHITECTURE.md`, focused architecture pages, ADR 0009, and architecture tests need no change
  because compiler ownership, dependency direction, canonical-output use, deferred-constraint
  meaning, one-capture lifecycle, phase model, and immutable artifact boundary are unchanged.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [immutable glossary](../../../../glossary.md#term-immutable); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Compiler public APIs, publication/planning contracts, and existing optimization documentation
  need no change beyond the Compile API/autograd surfaces listed in scope because generated
  formulas use the ordinary shared Tensor algebra and unchanged validation/optimization pipeline.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>
- Config, Planning, Trace, Runtime, Prepare, Engine, extensions, concrete backends, backend
  conformance, and integration tests need no change because the task adds no cross-module,
  backend, physical saved-value, runtime, training, or end-to-end executable behavior.<span data-glossary-navigation="true"> ([Trace glossary](../../../../glossary.md#trace); [Runtime glossary](../../../../glossary.md#term-runtime); [Engine glossary](../../../../glossary.md#term-engine); [backends glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Gradle, dependencies, and Java 26 configuration need no change because no module, package
  dependency, build logic, source set, toolchain, or external library changed.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [toolchain glossary](../../../../glossary.md#term-toolchain))</span>
- Repository-wide validation and the complete first-order capability checkpoint remain deferred
  to Compiler 0005E or continuous integration under the recorded task validation tier.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [continuous integration glossary](../../../../glossary.md#term-continuous-integration-ci))</span>

## Implementation notes

- Extended `LayoutInference` to retain binding-dependent EXPAND predicates in deterministic aligned
  axis order while preserving static layout and unresolved binding-dependent layout.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [axis glossary](../../../../glossary.md#term-axis); [layout glossary](../../../../glossary.md#layout))</span>
- Extended `AutogradPreflight` with the exact 0005B source-backed family inventory, predicate
  consistency checks, output-slot-aware batch-training selection, canonical saved-output checks,
  and deterministic fail-closed occurrence/output/input diagnostics.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [fail-closed glossary](../../../../glossary.md#term-fail-closed); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Extended `FirstOrderAutograd` to dispatch reduction/softmax and normalization families and to
  accumulate selected slots of one producer in ascending slot order without changing existing
  one-output ordering.<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch); [softmax glossary](../../../../glossary.md#softmax--log-softmax); [normalization glossary](../../../../glossary.md#term-normalization); [producer glossary](../../../../glossary.md#term-producer))</span>
- Extended `ReductionGradientRules` with product/extrema, cumulative-product, softmax/log-softmax,
  advanced-statistics, and norm formulas; added package-private<span data-glossary-navigation="true"> ([extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [log-softmax glossary](../../../../glossary.md#softmax--log-softmax); [norm glossary](../../../../glossary.md#term-norm))</span>
  `NormalizationGradientRules` for Layer/RMS/[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) formulas.
- Added focused source-inventory, Shape-predicate, formula-structure, mixed-floating role,
  exceptional-policy, same-occurrence saved-output, slot-order, fail-closed, and combined-capture
  tests in the exact five authorized test files.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [capture glossary](../../../../glossary.md#term-capture-meanings))</span>
- The replacement documentation context finalized all five affected Javadocs, Compile API,
  autograd strategy/user guide, glossary, this task, both master plans, and roadmap without
  changing executable Java.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [autograd glossary](../../../../glossary.md#autograd))</span>

## Completion summary

- Completed changes: binding-aware EXPAND compiler adoption and the assigned first-order
  reduction, scan, softmax, statistics, norm, Layer/RMS, and batch-normalization matrix, including
  exact output-slot roles and same-occurrence saved-statistic use.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [compiler glossary](../../../../glossary.md#term-compiler); [scan glossary](../../../../glossary.md#term-cumulative-scan); [softmax glossary](../../../../glossary.md#softmax--log-softmax); [norm glossary](../../../../glossary.md#term-norm); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [normalization glossary](../../../../glossary.md#term-normalization); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [occurrence glossary](../../../../glossary.md#term-occurrence); [statistic glossary](../../../../glossary.md#term-statistic))</span>
- Files changed or created: exactly five compiler production/Javadoc files, five compiler test
  files, and eight documentation/planning files.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [planning glossary](../../../../glossary.md#term-planning))</span>
- Tests and validation: compiler compilation passed; the focused five-class suite passed 60
  tests; the final compiler module passed 22 suites/170 tests with no skips, failures, or errors;
  compiler Javadoc, generated-doc inspection, Markdown, bytecode/surface, exact scope, inventory,
  formula, status, no-later-spec, final-newline, and whitespace checks passed.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [module glossary](../../../../glossary.md#term-module); [bytecode glossary](../../../../glossary.md#term-bytecode); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: replacement clean context
  `/root/docs_compiler_0005b_replacement` independently finalized the affected Javadocs,
  explanatory documentation, glossary impact, planning state, and reasoned no-change
  conclusions.
- Documentation impact: Compile API, strategy, user guide, glossary, task, master plans, and
  roadmap now distinguish completed 0005B behavior from Draft 0005C–0005E/0006 work.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Javadoc review: all five affected package-private production owners were finalized without
  executable changes.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span>
- Glossary impact: synchronized autograd, deferred-constraint, cumulative-scan, softmax,
  Layer/RMS/batch-normalization, saved-statistic, and expand entries; no new reusable term was
  required.<span data-glossary-navigation="true"> ([autograd glossary](../../../../glossary.md#autograd))</span>
- Unresolved issues: None.
- Follow-up required: None. Compiler 0005C–0005E and 0006 remain ordered Draft work without
  detailed specifications.

Status: Complete
