# Task 0005B: Reduction, Scan, Softmax, Statistics, and Normalization Gradient Completion

## Status

Complete

## Goal

Complete first-order automatic differentiation for the current reduction, cumulative-scan,
softmax, statistical-reduction, norm, and normalization inventory inside the established
compiler-owned pre-capture Tensor-expression pipeline.

This task also adopts the binding-aware `EXPAND` contract completed by Model 0025B:

```text
binding-aware forward occurrence
  -> compiler inference emits the exact unresolved Shape obligation
  -> preflight proves that the inverse formula has the same or a stronger obligation
  -> ordinary public Tensor operations construct the contribution
  -> one combined capture retains every unresolved predicate fail-closed
```

The task preserves every completed Compiler 0004–0005A rule. It extends the closed rule matrix; it
does not replace the current request, Tensor algebra, capture, inference, optimization,
publication, or artifact lifecycle.

## Scope

### Source-backed operation and role inventory

Implementation must derive and lock the inventory from the production kinds, signatures,
attributes, producer output descriptors, Tensor helpers, and tests rather than trusting this table
alone.

| Family | Exact current variants | Differentiable roles in 0005B |
|---|---|---|
| `AggregateReductionKind` | `SUM`, `MEAN`, `PROD`, `MIN`, `MAX`, `ALL`, `ANY`, `ARG_MIN`, `ARG_MAX`, `LOG_SUM_EXP`, `VARIANCE`, `STANDARD_DEVIATION`, `L1_NORM`, `L2_NORM` | Floating inputs of ordinary full, single-axis, and multi-axis `SUM`, `MEAN`, `PROD`, `MIN`, and `MAX`; floating data for masked `SUM`/`MEAN`; floating input for binding-aware `SUM_TO_SHAPE`; and the floating input of all five advanced reductions. `ALL`, `ANY`, and both arg-extrema operations are non-differentiable. The BOOL mask is non-differentiable. |
| `CumulativeScanKind` | `CUM_SUM`, `CUM_PROD` | The one floating input for every exclusive/reverse combination. Integral scans are non-differentiable. |
| `SoftmaxKind` | `SOFTMAX`, `LOG_SOFTMAX` | The one floating input. |
| `LayerNormKind` | no-affine `[input]` and affine `[input, scale, bias]` `LAYER_NORM` | Input in both forms; scale and bias in the affine form. |
| `RmsNormKind` | `[input]` and `[input, scale]` `RMS_NORM` | Input in both forms; scale in the two-input form. |
| `BatchNormKind.BATCH_NORM_INFERENCE` | inputs `[input, scale, bias, runningMean, runningVariance]`, output slot `0` | All five floating Tensor inputs. Channel axis and epsilon are non-differentiable configuration. |
| `BatchNormKind.BATCH_NORM_TRAINING` | the same five inputs; outputs `0=output`, `1=nextRunningMean`, `2=nextRunningVariance`, `3=savedBatchMean`, `4=savedInverseStandardDeviation` | Slot 0: input, scale, and bias. Slot 1: input and running mean. Slot 2: input and running variance. Slots 3 and 4 are exact same-occurrence auxiliaries used by formulas, not independent requested cotangent roots. Momentum, epsilon, and channel axis are non-differentiable configuration. |
| `ShapeTransformKind.EXPAND` | existing one-input `TargetShapeAttrs` occurrence | Preserve the existing floating input rule while admitting its Model-0025B binding-aware Shape cases. |

The exact aggregate-kind count is fourteen, the scan-kind count is two, the softmax-kind count is
two, and Layer, RMS, and batch normalization contribute one, one, and two kind constants. Tests
must lock enum contents, signature variants, input roles, output slots, and classification so a
new kind, attributes variant, or output position fails closed until deliberately planned.

### Notation and shared construction rules

For one selected output `y`, incoming cotangent `g`, and selected input `x`:

- `Z_x` and `O_x` are request-local exact typed positive-zero and positive-one logical splats
  expanded to `x.shape`.
- `K_t(v)` is the existing exact same-typed scalar-operation attribute `v`.
- `T_y(t)` casts an operand to the selected forward output's exact floating data type when it
  differs. Mixed-floating normalization formulas first apply `T_y` to every forward-domain Tensor
  operand so intermediate Tensor operations use the occurrence result type in the original input
  promotion order; only the completed cotangent is cast back to its selected input type.
- `E_x(t, A, keep)` restores axes `A` removed by the forward occurrence in ascending axis order
  when `keep` is false, then expands through ordinary `Tensor.expand(x.shape)`.
- `C_x(t)` casts one already Shape-correct floating contribution to `x.type` exactly when its type
  differs.
- `N_A(t)` is `O_t.sum(A, true)`, with full reduction using all natural axes. It represents the
  selected logical count for static, zero, named-dynamic, and expression extents without a host
  element-count calculation.
- `Q_t(c)` is the exact typed logical value of non-negative integer correction `c`, constructed as
  a typed scalar one expanded to static Shape `[c]` and reduced with ordinary full `SUM`.
  Correction zero is therefore an empty-domain positive zero. Do not convert `long correction`
  through an inexact host floating cast.
- `V(t, target, mappedAxes)` reshapes a lower-rank affine/statistic operand by inserting static
  singleton axes outside `mappedAxes`, then ordinarily expands it to `target`. Layer/RMS affine
  axes are the trailing normalized axes; a batch-normalization vector maps to the channel axis.
- `R_x(t, reducedAxes)` reduces the complement or prefix axes selected by the formula, reshapes to
  the exact operand Shape when unresolved but forward-constrained Dimensions differ
  structurally, and applies `C_x`.

Every returned contribution must have exactly the selected input's Shape and floating data type
before deterministic accumulation. A Shape/type-correct direct result adds no redundant reshape,
sum-to-Shape, expand, or cast. All construction uses current public Tensor operations and the
existing request-local logical-splat owner.

### Binding-aware `EXPAND` adoption

`LayoutInference` must preserve current target-rank validation, right alignment, fully static
rejection, descriptor type/eligibility, target Shape, and resolved zero-stride view layout. For
each aligned pair in increasing input-axis order:

1. structural equality is proved;
2. a static source singleton is proved;
3. unequal fully static extents fail with the current incompatible-expand behavior; and
4. every remaining pair emits the existing predicate:

   ```text
   AnyOf(
       DimensionEqual(source, StaticDimension(1)),
       DimensionEqual(source, target))
   ```

   with deterministic subject text naming the aligned target axis.

`CapturedGraphInference` continues to classify each candidate as `PROVEN`, `DISPROVEN`, or
`DEFERRED`. A deferred result remains an occurrence-owned `DeferredGraphConstraint`; it is never
treated as proof, binding, or execution permission. New leading target axes have implicit source
one and need no aligned predicate.

Resolved layout remains byte-for-byte equivalent to current static behavior. A binding-dependent
pair has unresolved layout. No predicate changes the exact target Shape or selects a future
stride, materialization, or backend route.

Preflight must admit a binding-dependent forward `EXPAND` only when each aligned inverse
`SUM_TO_SHAPE` obligation is structurally the same as, or implied by, the forward
source-one-or-source-equal obligation. Conversely, a binding-dependent `SUM_TO_SHAPE` inverse is
ordinary `gradient.expand(input.shape())`; preflight admits it only when every generated aligned
expand predicate is exactly the forward target-one-or-target-equal-source obligation after the
source/target roles are mapped. A static contradiction, rank mismatch, differently aligned
predicate, or missing forward obligation fails before derivative Tensor allocation.

This is a fail-closed proof-equivalence check, not a concrete binding solver. The combined graph
retains the predicate for later binding validation.

### Reduction axes and cotangent restoration

For full reductions, `A` is every input axis in natural order. For `AxisReductionAttrs`, `A`
contains its one normalized axis. For `MultiAxisReductionAttrs` and
`StatisticalReductionAttrs`, `A` is the exact ordered distinct axis list; axis order remains
attributes identity even when a formula needs a sorted copy for axis insertion. An empty
multi-axis list selects a point domain.

Existing `SUM`, `MEAN`, masked `SUM`, masked `MEAN`, and statically provable
`SUM_TO_SHAPE` formulas remain unchanged except for binding-aware inverse admission:

| Variant | Input contribution |
|---|---|
| ordinary `SUM` | `E_x(g, A, keepDimensions)` |
| ordinary `MEAN` | `E_x(g, A, keepDimensions) / N_A(x).expand(x.shape)` |
| masked `SUM` | `where(mask, E_x(g, [axis], false), Z_x)` |
| masked `MEAN` | `where(mask, E_x(g, [axis], false) / selectedCount, Z_x)` |
| `SUM_TO_SHAPE` | `g.expand(x.shape)` under the exact proof-equivalence rule above |

The mask uses its original forward broadcasting relation and receives no cotangent. Masked
all-false slices retain Compiler 0004B's direct-zero contribution policy. An ordinary empty
reduction domain has no input coordinate at which its zero denominator is evaluated; the
resulting contribution retains the corresponding zero-extent Shape. Empty axis lists have count
one and behave as point identities.

### Zero-safe product reduction

Do not use `forwardProduct / x`. For axes `A`, construct a deterministic sequence of ordinary
single-axis keep-dimension products in recorded axis order:

```text
stage[0] = x
stage[k + 1] = stage[k].prod(A[k], true)
```

Restore `g` to the final keep-dimension Shape. Traverse stages in reverse. For stage input `v` and
axis `a`:

```text
prefix = v.cumProd(a, true, false)
suffix = v.cumProd(a, true, true)
h = h.expand(v.shape) * prefix * suffix
```

The final `h` has `x.shape`. Full reduction uses natural axis order; an empty axis list or scalar
full product returns the restored `g` directly. This formula excludes the current coordinate
without division and is exact at any number of represented signed zeros. It also handles zero
extents structurally. NaN, infinity, zero-times-infinity, overflow, underflow, sign, rounding, and
reassociation follow the ordinary `CUM_PROD` and multiplication contracts in formula order; no
gradient-only repair is added at those non-regular boundaries.

### Reduction extrema

For floating reduction `MIN` or `MAX`, restore and expand both `g` and exact forward output `y`:

```text
matches = (x == E_x(y, A, keepDimensions))
tieCount = where(matches, O_x, Z_x).sum(A, true).expand(x.shape)
dx = where(matches, E_x(g, A, keepDimensions) / tieCount, Z_x)
```

All exact represented-numeric ties share the cotangent equally. Numeric equality makes opposite
signed zeros ties; equal finite values and equal same-sign infinities also tie. If the forward
output is NaN, numeric equality is false at every coordinate and the selected contribution is
exact positive zero. Empty domains have no input positions. Integral extrema are
non-differentiable.

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
scan domain contains each input position. Replacing signed zero by positive one makes division
zero-safe and gives the exact derivative for zero-free prefixes, the unique-zero product for a
prefix with one zero, and zero for prefixes with more than one zero. Zero-length axes remain
zero-length. NaN, infinity, exceptional products, and floating rounding follow the ordinary
formula; division by infinity is not given a gradient-specific limiting repair.

### Softmax and log-softmax

Use the exact forward output `y`, never a separately recomputed softmax occurrence:

| Kind | Input contribution |
|---|---|
| `SOFTMAX` | `y * (g - (g * y).sum(axis, true))` |
| `LOG_SOFTMAX` | `g - exp(y) * g.sum(axis, true)` |

The formulas preserve Shape and type. A zero-length normalization axis has no coordinate and
therefore produces a zero-length contribution. Finite slices use the exact analytic
vector-Jacobian products. NaN, infinity, signed zero, overflow, underflow, and any forward
finite-precision choice flow through the exact saved output and ordinary operations in formula
order. The compiler adds no forward softmax algorithm or exceptional-value rewrite.

### Advanced reductions and boundary policies

For every formula below, restore `g` and `y` through `E_x` as applicable.

| Kind | Exact input contribution and policy |
|---|---|
| `LOG_SUM_EXP` | `E_x(g) * exp(x - E_x(y))`. Non-empty finite slices use the analytic formula. Empty slices have no input coordinate. All-negative-infinity, positive-infinity, NaN, overflow, and underflow use the raw saved-output formula without a compiler normalization repair. |
| `VARIANCE` | Let `mu=x.mean(A,true)`, `N=N_A(x)`, `D=N-Q_t(correction)`. Return `E_x(g) * 2 * (x-mu) / D.expand(x.shape)`. The forward `D>0` constraint remains authoritative. Valid constant finite slices return exact zero. NaN and infinity use the raw formula. |
| `STANDARD_DEVIATION` | With the same `mu` and `D`, let `regular=E_x(g)*(x-mu)/(D.expand(x.shape)*E_x(y))`; return `where(E_x(y) > Z_x, regular, Z_x)`. A valid zero standard deviation therefore has exact zero cotangent at every coordinate. NaN output also selects zero; positive infinity enters the raw branch. |
| `L1_NORM` | `where(x>Z_x, E_x(g), where(x<Z_x, -E_x(g), Z_x))`. Both signed zeros and NaN select exact zero; infinities use their sign. |
| `L2_NORM` | `regular=E_x(g)*x/E_x(y)`; return `where(E_x(y)>Z_x, regular, Z_x)`. A zero norm and NaN norm select exact zero for the complete slice; positive infinity uses the raw quotient. |

`Q_t(correction)` must use logical ones as defined above. Do not introduce an exact-integer
threshold, host-size calculation, physical constant, or correction-specific model kind.

### Layer normalization

Let `A` be the natural trailing axes named by `normalizedShape`, `P` the natural leading prefix
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

When `P` is empty, the point-domain reductions retain Shape and are still ordinary expressions;
an implementation may omit only a provably redundant reshape. Scale and bias Shapes equal the
exact normalized Shape; they are not arbitrary broadcast parameters.

Positive epsilon makes finite zero-variance slices regular. Empty outputs have no normalized
coordinate. NaN, infinity, overflow, underflow, and signed-zero behavior follows the recomputed
ordinary population-mean formula; the compiler adds no gradient-only finite mask.

### RMS normalization

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

Zero input with positive epsilon is regular. Empty and exceptional-value policy is the same raw
ordinary-Tensor policy as layer normalization.

### Batch-normalization inference

Let `B` be every natural non-channel axis, and align every rank-one channel vector with `V`:

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

Negative radicands, zero denominators, NaN, infinity, overflow, and rounding use ordinary
`RSQRT` and arithmetic behavior. Epsilon and channel axis receive no cotangent.

### Batch-normalization training

Preflight must require exactly five outputs and retrieve:

```text
savedMean = producer.output(3)
savedInvStd = producer.output(4)
```

Those exact canonical wrappers are forward values of the selected occurrence. Do not reconstruct
a wrapper, recompute batch statistics, decompose the original forward occurrence, or create a
physical saved buffer.

Let `B` be every natural non-channel axis and `xT=T_y(x)` for every selected public output slot.
For selected output slot 0, align `scale`, `savedMean`, and `savedInvStd` to the input Shape:

```text
centered = xT - V(savedMean)
r = V(savedInvStd)
h = g0 * V(T_y(scale))

dx = C_x(r * (h - h.mean(B,true)
                - centered * r * r * (h * centered).mean(B,true)))
dScale = R_scale(g0 * centered * r, B)
dBias = R_bias(g0, B)
```

Running mean and running variance receive no contribution from slot 0.

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
denominator valid whenever an input coordinate exists. Slot 1/2 input contributions accumulate
with slot 0 when the objective reaches multiple outputs. Scale and bias receive contributions only
from slot 0. Running-statistic inputs receive contributions only from their matching transition
slot. Slots 3 and 4 cannot be independent first-order request roots in this task; they are
auxiliary forward values for slot-0/slot-2 formulas.

### Output-slot-aware preflight and accumulation

The current one-output behavior must remain unchanged. Batch training additionally requires
output-slot-aware route selection:

- inventory every exact output wrapper through canonical producer identity;
- for one producer reached through multiple outputs, retain every objective-ancestry output slot
  rather than only the first encountered slot;
- classify differentiable input roles by exact output slot before propagating target
  reachability;
- intentionally non-differentiable roles do not create a false path through another output;
- an unknown kind, attributes variant, output slot, or ambiguous role fails closed;
- a target whose only structural paths use non-differentiable roles fails before derivative
  Tensor allocation with deterministic occurrence/output/input context; and
- hidden slots 3/4 may be retrieved by an approved formula but are rejected as independent
  selected output roots.

Reverse accumulation order is:

1. reverse producer postorder;
2. selected output slots in ascending numeric order for that producer; and
3. input positions in ascending numeric order for that output.

This preserves every existing one-output contribution order. Multiple output slots append
separate contributions to a shared input and left-associate them through ordinary `Tensor.add`.
Repeated input positions remain repeated contributions.

### Validation and failure order

Preserve top-level argument, compile-mode, objective, forward-output membership, ingress,
canonical-wrapper, target identity, floating eligibility, and ancestry validation from Compiler
0004–0005A.

Before seed, constant, formula, or derivative Tensor construction, visit selected producers in
deterministic producer postorder and validate:

1. exact kind and attributes pairing;
2. exact input and output counts;
3. selected canonical output slot and its floating gradient eligibility;
4. output-slot-specific differentiable and non-differentiable roles in input-position order;
5. exact input floating eligibility and current ordered promotion;
6. normalized axes, ordered distinct axis lists, keep-dimension result Shape, masks, correction,
   normalized Shape, channel Shape, and result descriptors;
7. directly proved Shape relations or exact forward-predicate implication for every generated
   expand/reshape/sum-to-Shape;
8. availability and exact identity of any required same-occurrence auxiliary output; and
9. the fixed formula and boundary-policy row selected above.

Known failure consumes no seed, logical splat, formula Tensor, or derivative `TensorId`.
Successful preflight may be followed by ordinary public Tensor construction, capture, inference,
validation, optimization, publication, or planning failure that consumes opaque non-reusable IDs.

Do not inspect Tensor payloads. Do not reject a represented value merely because it is zero, NaN,
infinite, negative, or exceptional under a raw formula. Forward constraints such as positive
statistical denominator and batch-training domain remain ordinary compiler predicates.

### Producers, provenance, phase, constants, and shared pipeline

- Preserve exact Tensor and `TensorProducer` object identity in ancestry, output-slot,
  contribution, and accumulation bookkeeping.
- Use `producer.output(3)` and `producer.output(4)` for batch-training saved values.
- Original producer identity, including all five batch-training outputs, remains `FORWARD`.
  Every generated formula, alignment, logical-count, scan, reduction, comparison, and constant
  expansion producer is `BACKWARD`.
- Capture ordered forward outputs and gradient roots together exactly once. Assign each
  `NodeId` and `ValueId` once and retain per-node `GraphPhase`.
- Multiple target roles may share one captured gradient `ValueId`; no identity node is added.
- Reuse the request-local exact typed-splat cache. Reuse the fixed typed `2` and `-0.5`
  coefficient bits established by Compiler 0005A. Scalar epsilon and momentum remain their exact
  operation attributes.
- Only generated base constants reachable from returned gradient roots remain in combined
  ingress. Storage, labels, provenance absence, Shape, or factory history never imply constant
  status.
- Run the existing inference, validation, canonicalization, exact rewriting, constant folding,
  whole-graph dead-code elimination, phase-local common-subexpression elimination, cleanup, final
  validation, publication, and planning pipeline. Add no gradient-specific optimizer or pass.

## Out of scope

- Compiler 0005C layout/window/indexing/scatter/ordering/stochastic completion beyond the narrow
  binding-aware `EXPAND` adoption required here
- Compiler 0005D attention/convolution/pooling/loss work or Compiler 0005E closure/checkpoint work
- Compiler 0006 public objectives, targets, seeds, disconnected-result policy, create-graph,
  derivative order, higher derivatives, or order-aware phase representation
- another model kind, Tensor method, attributes type, result carrier, public saved-statistic
  accessor, backward-only operation, or public compiler request/result surface
- mutable Tensor gradients, `Tensor.backward`, a tape, physical saved buffers, recomputed batch
  saved values, direct captured-node formula construction, or a second algebra
- a public gradient registry, facade, policy object, configurable subgradient mode, generic
  algebra builder, or gradient-specific optimizer/folder/rewriter
- changing forward reduction, scan, softmax, statistics, normalization, epsilon, momentum,
  correction, comparison, signed-zero, NaN, infinity, accumulation, or layout semantics
- model, config, planning, trace, runtime, prepare, engine, extension, backend, storage,
  execution, lowering, kernel, physical memory, schedule, or optimizer-update behavior
- module dependencies, Gradle, Java version, architecture contracts, ADRs, architecture tests,
  backend-conformance tests, or integration tests
- unrelated refactoring or absorbing any part of 0005C–0005E or 0006

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Training graph](../../../../architecture/training-graph.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Runtime / Prepare / Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [ADR 0009](../../../../design/decisions/0009-compiler-owned-pre-capture-tensor-expression-autograd.md)
- [Autograd strategy](../../../../design/notes/autograd-strategy.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Compiler master plan](../master-plan.md)
- [Model master plan](../../model/master-plan.md)
- [Model capabilities](../../model/capabilities.md)
- [Adjoint expressibility audit](../../model/adjoint-expressibility-audit.md)
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
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- Compiler owns fail-closed preflight, named gradient-rule dispatch, ephemeral identity
  accumulation, phase-aware combined capture, graph validation, exact optimization, and
  compile-artifact orchestration.
- Model owns public Tensor algebra, immutable operation semantics, canonical producer output
  wrappers, and exact provenance. It owns no derivative rule or graph predicate.
- Generated gradients are ordinary Tensor expressions and use exactly the forward
  inference/validation/numerical/optimization contracts.
- Deferred constraints remain logical compile obligations. They are not bindings, physical
  values, runtime checks implemented by this task, or execution permission.
- Same-occurrence batch saved statistics are logical forward values. Their use creates no tape,
  recomputation policy, buffer, publication requirement, or runtime ownership.
- The compiler remains independent of runtime, prepare, engine, extensions, and concrete
  backends.
- If implementation needs a model/public API, new dependency, architecture change, physical
  saved-value contract, or formula outside the current Tensor algebra, stop and report the exact
  blocker.

## Package impact

Existing package retained:

```text
io.github.pho001.synaptik.compiler
  package-private inference, preflight, gradient formulas, accumulation, and combined capture
```

One package-private type may be added:

- `io.github.pho001.synaptik.compiler.NormalizationGradientRules` — owns Layer/RMS/batch
  normalization formulas so `ReductionGradientRules` remains focused on aggregate reductions,
  statistics, softmax, and scans.

No public type or package is added, moved, or widened.

## Affected files

Production and Javadoc (5):

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

Documentation and planning (8):

- `docs/api/compile-api.md`
- `docs/design/notes/autograd-strategy.md`
- `docs/user-guide/autograd.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/compiler/master-plan.md`
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review unchanged unless evidence contradicts this specification:

- `ARCHITECTURE.md`, focused architecture pages, and ADR 0009
- Tensor and Training APIs
- Model kinds, attributes, Tensor helpers, producer/provenance, source/tests/Javadocs, tasks,
  capabilities, and audits
- compiler inference/constraint/optimization/publication/planning sources and tests outside the
  exact paths above
- Config, Planning, Trace, Runtime, Prepare, Engine, extensions, backends, architecture tests,
  backend conformance, integration tests, Gradle, dependencies, and Java 26 configuration

## Maximum scope

At most the exact eighteen paths listed above may change: five compiler production/Javadoc paths,
five compiler test paths, and eight documentation/planning paths.

The task may add only `NormalizationGradientRules`. Stop before a nineteenth path, a second type,
a public surface, another module, a model/test/Gradle/architecture change, or any 0005C–0005E/0006
task file. If an assigned formula cannot be completed inside this boundary, report the exact
missing primitive, forward policy, type, or architecture decision instead of silently omitting
the role.

## Acceptance criteria

- Source-backed inventory tests lock all fourteen aggregate kinds and every attributes/signature
  variant, both scan kinds, both softmax kinds, Layer/RMS/batch kinds, all five batch-training
  output slots, and every differentiable/non-differentiable role assigned above.
- `LayoutInference` accepts Model-0025B binding-aware `EXPAND`, emits exact ordered
  source-one-or-source-equal predicates, rejects contradictions, preserves static layout, and
  retains unresolved layout and deferred obligations without binding.
- Preflight admits binding-dependent `EXPAND` and `SUM_TO_SHAPE` inverses only through exact
  predicate-equivalence/implication proof and rejects any differently aligned or unsupported
  obligation before derivative allocation.
- Existing `SUM`, `MEAN`, masked reduction, `SUM_TO_SHAPE`, and `CUM_SUM` formulas and policies
  remain unchanged for their previously supported cases.
- Product reduction uses sequential exclusive prefix/suffix `CUM_PROD` formulas without division
  and covers full, scalar, single-axis, ordered multi-axis, keep/remove-dimension, empty-axis-list,
  zero-extent, one-zero, and multiple-zero structure.
- `CUM_PROD` implements the exact zero-safe formula for all four exclusive/reverse modes and
  covers signed zero, one zero, multiple zeros, zero-length axes, and raw NaN/infinity structure.
- Reduction extrema split exact represented ties equally and lock finite, signed-zero, infinity,
  NaN, empty, axis, and keep-dimension policy.
- Softmax, log-softmax, log-sum-exp, variance, standard deviation, L1 norm, and L2 norm implement
  the exact formulas, correction construction, count semantics, and zero/NaN/infinity policies
  above.
- Layer and RMS normalization cover no-affine/affine forms, exact trailing axes, prefix
  reductions, epsilon, mixed-floating normalization, zero/empty/exceptional policy, and exact
  scale/bias Shapes.
- Batch inference covers all five inputs, arbitrary normalized channel axis, vector alignment,
  non-channel reductions, mixed-floating normalization, and raw radicand/special-value policy.
- Batch training covers output slots 0–2, exact canonical saved slots 3/4, slot-specific roles,
  momentum, biased versus correction-one formulas, channel-empty/domain constraints, and
  deterministic accumulation across multiple reached output slots. Saved values are never
  recomputed or materialized by compiler policy.
- BOOL reductions, arg extrema, integral results/scans, masks, saved auxiliary roots, axes,
  correction, epsilon, momentum, and other configuration roles remain non-differentiable.
- Every contribution has the selected input's exact Shape and floating type before accumulation.
  Unresolved Shape normalization is accepted only when implied by the exact forward predicate.
- Known unsupported kinds, attributes, counts, output slots, roles, Shapes, types, or policies fail
  before seed/constant/formula allocation with deterministic occurrence/output/input context.
- Reverse accumulation uses reverse producer postorder, ascending selected output slots, ascending
  input positions, repeated contributions, left-associated `Tensor.add`, and exact object
  identity.
- One combined capture preserves canonical wrappers, all batch output slots, original/generated
  producer phase, unique graph-local IDs, target roles, explicit reachable constants, and the
  existing immutable graph/artifact boundary.
- Generated formulas pass the shared inference/validation and exact optimization pipeline. No
  gradient-specific inference system, simplifier, constant folder, rewrite, or optimizer exists.
- No model/public API, dependency, build, architecture, backend, prepare, runtime, training,
  physical-value, or higher-order contract changes.
- Compile API, autograd strategy/user guide, glossary, task, both master plans, and roadmap
  consistently distinguish completed 0005B behavior from planned 0005C–0005E/0006 behavior.
- Tensor API needs no change because no public Tensor declaration or Model forward semantic
  changes. Training API needs no change because no optimizer/session/module/public training
  workflow changes. The documentation pass records both reasons.
- Exactly the eighteen authorized paths change. Only 0005B becomes Complete after implementation;
  0005C–0005E and 0006 remain concise Draft rows without detailed specifications.
- A separate clean-context documentation-focused pass independently finalizes affected Javadocs,
  explanatory documentation, glossary impact, planning status/evidence, terminology, links, and
  rendered Javadoc in the same overall change.

## Tests / validation

During implementation, run focused suites as needed:

```bash
./gradlew :modules:compiler:test \
  --tests io.github.pho001.synaptik.compiler.LayoutInferenceTest \
  --tests io.github.pho001.synaptik.compiler.AutogradPreflightTest \
  --tests io.github.pho001.synaptik.compiler.GradientRulesTest \
  --tests io.github.pho001.synaptik.compiler.FirstOrderAutogradTest \
  --tests io.github.pho001.synaptik.compiler.GraphCompilerTest
```

After executable Java stabilizes, run one final compiler module suite:

```bash
./gradlew :modules:compiler:test
```

Record JUnit XML suite/test counts and zero skipped/failure/error outcomes. Hand the exact
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
parameters, results, expected failures, ownership, output-slot, ordering, and lifecycle contracts;
check local links/anchors, balanced fences, final newlines, and terminology; verify exact
eighteen-path scope; lock the source-backed inventory and formula constants; confirm no
model/test/Gradle/dependency/architecture change; synchronize 0005B Complete in task/master/
roadmap; confirm exactly one current detailed compiler task; and verify that no detailed
0005C–0005E/0006 specification exists.

Repository-wide validation and the first-order capability checkpoint remain deferred to Compiler
0005E or continuous integration. This task changes one module's package-private compiler behavior
and no dependency, build, architecture boundary, or public API.

## Dependencies

- Model 0025B binding-aware expansion — Complete.
- Model 0025 canonical producer outputs — Complete.
- Model 0023A binding-aware sum-to-Shape and Model 0023E cumulative product — Complete.
- Model reduction, scan, softmax, statistics, Layer/RMS/batch normalization tasks and the
  adjoint/closure audits — Complete.
- Compiler 0001–0005A, including shared Shape/type normalization, exact splats, fail-closed
  preflight, canonical outputs, one combined capture, inference/constraints, exact optimization,
  publication, planning, and artifacts — Complete.
- Existing public Tensor algebra is sufficient. The representability re-audit found no missing
  primitive, unresolved required forward policy, new type, module, dependency, or architecture
  decision.

No Config, Trace, Runtime, Prepare, Engine, Training, backend, or new Model task is a dependency.

## Follow-up tasks

- Compiler 0005C remains Draft and follows 0005B. It owns remaining layout/window/indexing/
  scatter/ordering/stochastic roles.
- Compiler 0005D remains Draft and follows 0005B/0005C. It owns attention, convolution, pooling,
  and loss roles.
- Compiler 0005E remains Draft and follows 0005A–0005D. It owns the complete source-backed
  first-order role/output closure audit, transitive formula-operation closure, and capability
  checkpoint.
- Compiler 0006 remains Draft and follows 0005E plus the stable compile/artifact boundary. It owns
  public functional requests, seeds, disconnected results, derivative order, and higher-order
  differentiation.

Do not create those detailed specifications during this task.

## Architecture impact

Expected impact: None.

This task fills the established compiler-owned first-order extension point, adopts an already
completed Model semantic contract through the existing predicate vocabulary, and uses canonical
producer outputs through the existing one-capture lifecycle. It changes no authority, ownership,
dependency direction, public lifecycle, Model forward meaning, or artifact shape.

If implementation requires a new model/public operation, a changed forward policy, a physical
saved value, a runtime tape, direct graph-node algebra, another dependency, or an architecture
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
  new operation kind, or architecture change is required.
- Binding-aware `EXPAND` reuses `AnyOf` and `DimensionEqual`; no predicate or binding type is
  added.
- Product reduction uses sequential single-axis keep-dimension products and exclusive
  prefix/suffix scans. This avoids both division by zero and a flattened selected-domain extent
  that could overflow `long`.
- Cumulative-product differentiation uses safe products, cumulative zero counts, and
  opposite-direction cumulative sums. It is zero-safe without claiming an infinity-safe limiting
  extension.
- Extrema use equal tie sharing; standard deviation and L2 norm select zero at zero result; L1
  selects zero at signed zero; raw saved-output formulas govern the explicitly listed exceptional
  cases.
- Statistical correction is a logical sum of exact typed ones, not a host floating conversion.
- Layer/RMS formulas use population means directly, avoiding a host or separately represented
  cardinality.
- Batch training is output-slot-aware. Slots 3/4 remain exact auxiliary forward values, and slots
  0–2 contribute through their exact role matrices.
- `NormalizationGradientRules` is the sole new package-private type and prevents one broad
  reduction/normalization rule owner.

## Known limitations

- The internal request remains one scalar objective with an implicit unit seed. Public requests,
  caller seeds, disconnected gradients, create-graph, and derivative order wait for Compiler
  0006.
- Formula-structure tests do not prove backend numerical execution. Backend implementations later
  need conformance coverage for the selected zero, tie, correction, exceptional-value, and
  normalization behavior.
- Deferred Shape predicates remain compile obligations. This task does not implement concrete
  dimension binding, preparation, lowering, or execution.
- Raw exceptional-value formulas intentionally can produce NaN or infinity where the selected
  policy says to use ordinary Tensor arithmetic. Only the explicit zero/tie cases above receive a
  local repair.
- Complete current-model first-order coverage waits for Compiler 0005C–0005E.

## Validation evidence

Implementation evidence reused by the replacement documentation pass:

- The implementation pass ran compiler main/test compilation successfully after the final
  executable edits.
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
  strategy/user guide, glossary, and current operation/constraint/provenance contracts.
- `./gradlew :modules:compiler:javadoc` passed after final Javadoc edits in 1 second with seven
  actionable tasks: two executed and five up-to-date. It emitted no warnings. Generated
  package-private pages for `AutogradPreflight`, `FirstOrderAutograd`, `LayoutInference`,
  `ReductionGradientRules`, and `NormalizationGradientRules` were inspected for parameter,
  result, failure, identity, ownership, output-slot, ordering, and lifecycle contracts.
- `python3 /tmp/validate_synaptik_markdown.py` passed across 12 Markdown files and 702 local
  links, including local file targets, heading anchors, balanced fences, final newlines, and
  trailing whitespace.
- The user-guide example remains explicitly conceptual because no public objective/target compile
  API exists. A runnable public example was therefore neither possible nor required; the focused
  structural tests and generated-document inspection validate the current package-private
  behavior.
- `git diff --check` passed with no output.
- The sorted union of tracked and untracked changed paths contains exactly the authorized
  eighteen paths: five compiler production/Javadoc files, five compiler tests, and eight
  documentation/planning files.
- `javap` and manual source checks confirmed that all five affected compiler production types
  remain package-private; exactly one new package-private `NormalizationGradientRules` type was
  added; the Tensor surface remains at 200 public members with exactly the existing two `expand`
  overloads; `CompileArtifacts`, `BatchNormTrainingResult`, `TensorProducer`, and the empty
  `TrainingModule` placeholder retain their public surfaces; no public compiler, Tensor, or
  Training declaration changed; and the fixed typed coefficient values remain `2` and `-0.5`
  with the recorded BFLOAT16/FLOAT32/FLOAT64 bits.
- Manual source-backed inventory review confirmed exactly fourteen aggregate, two scan, two
  softmax, one Layer, one RMS, and two batch-normalization kinds; batch training retains exactly
  five output slots and the implemented public-slot role matrix. The enum-array and focused tests
  lock the current inventory.
- Final `AutogradPreflight` source/Javadoc review confirmed that ordinary selected occurrences
  retain their one-output guard, batch training retains its exact five-output guard, reduction/
  scan/normalization inference re-derives the complete output count and every descriptor, and
  saved batch slots three and four are rejected both as requested targets and selected cotangent
  roots before derivative construction.
- Manual formula review confirmed division-free product reduction, zero-safe cumulative product,
  equal-tie extrema, saved-output softmax/log-softmax, logical-count/correction construction,
  zero-result standard-deviation/L2 and signed-zero/NaN L1 policies, mixed-floating Layer/RMS/
  batch formulas, exact same-occurrence batch saved outputs, ascending output-slot accumulation,
  request-local splats, one combined phase-aware capture, and the shared inference/validation/
  optimization pipeline.
- Task, Compiler master plan, Model master plan, and roadmap are synchronized to Compiler 0005B
  `Complete`. Compiler 0005C–0005E and 0006 remain concise `Draft` rows without detailed task
  specifications, and no later compiler row is `Ready`.

Reasoned no-change conclusions:

- Tensor API and Model public/Javadoc contracts remain unchanged because 0005B adds no public
  Tensor declaration, operation kind, attributes, result carrier, forward semantic, descriptor,
  provenance, or storage behavior; the compiler consumes completed Model 0025B and canonical
  producer outputs unchanged.
- Training API and Public APIs remain unchanged because there is still no public gradient
  request, seed, publication workflow, module/parameter/session contract, optimizer update,
  preparation, execution, or run result.
- Model capabilities, adjoint/closure audits, related operation contracts, and Model tests need
  no change because every selected Model capability and forward boundary was already complete;
  0005B changes compiler-owned derivative construction only.
- `ARCHITECTURE.md`, focused architecture pages, ADR 0009, and architecture tests need no change
  because compiler ownership, dependency direction, canonical-output use, deferred-constraint
  meaning, one-capture lifecycle, phase model, and immutable artifact boundary are unchanged.
- Compiler public APIs, publication/planning contracts, and existing optimization documentation
  need no change beyond the Compile API/autograd surfaces listed in scope because generated
  formulas use the ordinary shared Tensor algebra and unchanged validation/optimization pipeline.
- Config, Planning, Trace, Runtime, Prepare, Engine, extensions, concrete backends, backend
  conformance, and integration tests need no change because the task adds no cross-module,
  backend, physical saved-value, runtime, training, or end-to-end executable behavior.
- Gradle, dependencies, and Java 26 configuration need no change because no module, package
  dependency, build logic, source set, toolchain, or external library changed.
- Repository-wide validation and the complete first-order capability checkpoint remain deferred
  to Compiler 0005E or continuous integration under the recorded task validation tier.

## Implementation notes

- Extended `LayoutInference` to retain binding-dependent EXPAND predicates in deterministic aligned
  axis order while preserving static layout and unresolved binding-dependent layout.
- Extended `AutogradPreflight` with the exact 0005B source-backed family inventory, predicate
  consistency checks, output-slot-aware batch-training selection, canonical saved-output checks,
  and deterministic fail-closed occurrence/output/input diagnostics.
- Extended `FirstOrderAutograd` to dispatch reduction/softmax and normalization families and to
  accumulate selected slots of one producer in ascending slot order without changing existing
  one-output ordering.
- Extended `ReductionGradientRules` with product/extrema, cumulative-product, softmax/log-softmax,
  advanced-statistics, and norm formulas; added package-private
  `NormalizationGradientRules` for Layer/RMS/batch formulas.
- Added focused source-inventory, Shape-predicate, formula-structure, mixed-floating role,
  exceptional-policy, same-occurrence saved-output, slot-order, fail-closed, and combined-capture
  tests in the exact five authorized test files.
- The replacement documentation context finalized all five affected Javadocs, Compile API,
  autograd strategy/user guide, glossary, this task, both master plans, and roadmap without
  changing executable Java.

## Completion summary

- Completed changes: binding-aware EXPAND compiler adoption and the assigned first-order
  reduction, scan, softmax, statistics, norm, Layer/RMS, and batch-normalization matrix, including
  exact output-slot roles and same-occurrence saved-statistic use.
- Files changed or created: exactly five compiler production/Javadoc files, five compiler test
  files, and eight documentation/planning files.
- Tests and validation: compiler compilation passed; the focused five-class suite passed 60
  tests; the final compiler module passed 22 suites/170 tests with no skips, failures, or errors;
  compiler Javadoc, generated-doc inspection, Markdown, bytecode/surface, exact scope, inventory,
  formula, status, no-later-spec, final-newline, and whitespace checks passed.
- Documentation-agent review: replacement clean context
  `/root/docs_compiler_0005b_replacement` independently finalized the affected Javadocs,
  explanatory documentation, glossary impact, planning state, and reasoned no-change
  conclusions.
- Documentation impact: Compile API, strategy, user guide, glossary, task, master plans, and
  roadmap now distinguish completed 0005B behavior from Draft 0005C–0005E/0006 work.
- Javadoc review: all five affected package-private production owners were finalized without
  executable changes.
- Glossary impact: synchronized autograd, deferred-constraint, cumulative-scan, softmax,
  Layer/RMS/batch-normalization, saved-statistic, and expand entries; no new reusable term was
  required.
- Unresolved issues: None.
- Follow-up required: None. Compiler 0005C–0005E and 0006 remain ordered Draft work without
  detailed specifications.

Status: Complete
