# Task 0006B2: [Pool3d](../../../../glossary.md#pool3d--ncdhw-pooling) and 3D-Window [Gradient](../../../../glossary.md#term-gradient) Closure

> **Editorial reading key — not part of the historical record.**
>
> Compiler builds these adjoints from public unfold/fold and ordinary Tensor algebra before
> capture. Typed window padding is configuration, not a differentiable input; maximum-pool routing
> depends on the selected same-occurrence result and its first-winner convention. Constructed
> formulas and transitive differentiation closure are semantic claims, not physical window
> materialization or CPU execution. The shrinking deferred inventory below is a historical
> checkpoint, not a current global exclusion list.
>
> Current reading: [pre-capture differentiation and functional requests](../../../../reference/compiler/autograd.md#current-package-private-pre-capture-autograd).

## Status

Complete

## Goal

Close the five first-order [gradient](../../../../glossary.md#term-gradient) [signatures](../../../../glossary.md#term-signature) deliberately deferred by completed [Compiler](../../../../glossary.md#term-compiler) task
0006B1, without changing the complete [Model](../../../../glossary.md#term-model) or Compiler forward inventory:

```text
Pool3dKind.MAX_POOL3D         + MaxPool3dAttrs      1 -> 1
Pool3dKind.AVERAGE_POOL3D     + AveragePool3dAttrs  1 -> 1
WindowTransformKind.UNFOLD3D  + Window3dAttrs       1 -> 1
WindowTransformKind.UNFOLD3D  + Unfold3dAttrs       1 -> 1
WindowTransformKind.FOLD3D    + Fold3dAttrs         1 -> 1
```

Implement their exact adjoints solely as ordinary public [`Tensor`](../../../../glossary.md#tensor) expressions constructed by
Compiler before [graph capture](../../../../glossary.md#capture-graph-ports-and-reconstruction). The formulas must themselves remain valid inputs to the existing
two-stage functional-gradient machinery. No saved maximum index, backward-only [operation](../../../../glossary.md#operation),
direct [graph](../../../../glossary.md#graph)-[node](../../../../glossary.md#node) construction, hidden execution, or [pooling](../../../../glossary.md#term-pooling)-backward kind is permitted.

Forward coverage remains exactly 40 operation-kind enum families, 115 [constants](../../../../glossary.md#term-constant), and 137
signatures. Production first-order support moves from 37 families, 107 constants, and 128
signatures to exactly 38 families, 111 constants, and 133 signatures. The exact disjoint deferred
[partition](../../../../glossary.md#partition) shrinks from nine to four signatures: [`RNN`](../../../../glossary.md#term-recurrent-neural-network-rnn), [`GRU`](../../../../glossary.md#term-gated-recurrent-unit-gru), [`LSTM`](../../../../glossary.md#term-long-short-term-memory-lstm), and [`CONV3D`](../../../../glossary.md#conv3d--ncdhw).

## [Scope](../../../../glossary.md#term-scope)

- Add the two Pool3d and three 3D-window signatures to the closed production first-order coverage
  inventory and route <span data-glossary-navigation="true"> ([Pool3d glossary](../../../../glossary.md#pool3d--ncdhw-pooling); [signatures glossary](../../../../glossary.md#term-signature))</span>`Pool3dKind` through the existing [pooling](../../../../glossary.md#term-pooling) rule [owner](../../../../glossary.md#term-owner).
- Remove only these five signatures from complete-forward-inventory rejection. Preserve
  allocation-free fail-closed rejection for the three recurrent signatures and <span data-glossary-navigation="true"> ([fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>[`CONV3D`](../../../../glossary.md#conv3d--ncdhw).
- Extend `PoolingGradientRules` with exact [NCDHW](../../../../glossary.md#conv3d--ncdhw) maximum- and average-pooling formulas.
- Extend `LayoutGradientRules` with both `UNFOLD3D` adjoints and the `FOLD3D` [adjoint](../../../../glossary.md#term-adjoint).
- Preserve exact BFLOAT16, FLOAT32, and FLOAT64 types, ordinary public-operation accumulation and
  narrowing, canonical captured Shape/Dimension references, and exact same-occurrence provenance.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [operation glossary](../../../../glossary.md#operation); [narrowing glossary](../../../../glossary.md#term-narrowing); [canonical glossary](../../../../glossary.md#term-canonical-form); [Shape glossary](../../../../glossary.md#shape); [Dimension glossary](../../../../glossary.md#dimension); [occurrence glossary](../../../../glossary.md#term-occurrence); [provenance glossary](../../../../glossary.md#provenance))</span>
- Prove formula constructibility, shape compatibility, exact attributes, structural kernel
  volume, and every existing symbolic obligation during preflight before derivative allocation.<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel); [preflight glossary](../../../../glossary.md#term-preflight); [derivative glossary](../../../../glossary.md#term-derivative))</span>
- Add source-backed inventory, preflight, first-order formula, transitive-closure, and explicit
  two-stage semantic coverage for all five signatures and their exceptional-value policies.
- Finalize affected Javadocs, API explanations, glossary impact, and synchronized planning in a
  mandatory independent clean documentation-focused context after a clean implementation context.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- recurrent [BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt), [`CONV3D`](../../../../glossary.md#conv3d--ncdhw) adjoints, or changing their deferred/[fail-closed behavior](../../../../glossary.md#admission-preflight-and-fail-closed-validation)
- Pool1d composition, Pool2d formulas, existing general-axis or 2D-window formulas, or any
  completed task's semantics<span data-glossary-navigation="true"> ([Pool1d composition glossary](../../../../glossary.md#pool1d-composition); [axis glossary](../../../../glossary.md#term-axis))</span>
- a saved-index Pool3d result, a maximum-pooling backward operation, a generic PoolNd/WindowNd
  abstraction, direct <span data-glossary-navigation="true"> ([Pool3d glossary](../../../../glossary.md#pool3d--ncdhw-pooling); [pooling glossary](../../../../glossary.md#term-pooling))</span>[`Operation`](../../../../glossary.md#operation)/[graph](../../../../glossary.md#graph)-[node](../../../../glossary.md#node) construction, or a private formula [IR](../../../../glossary.md#term-intermediate-representation-ir)
- Model kinds, attributes, signatures, Tensor methods, Shape/Dimension contracts, provenance,
  numerical semantics, source, tests, public method count, or API surface<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [signatures glossary](../../../../glossary.md#term-signature); [Tensor glossary](../../../../glossary.md#tensor); [Shape glossary](../../../../glossary.md#shape); [Dimension glossary](../../../../glossary.md#dimension); [provenance glossary](../../../../glossary.md#provenance); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- lowering, Planning capability ownership, Prepare, Runtime, Engine, backend contracts, CPU
  execution, generated code, storage, numerical kernels, or performance claims<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [Planning glossary](../../../../glossary.md#term-planning); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [kernels glossary](../../../../glossary.md#kernel))</span>
- changing forward Pool3d accumulation, narrowing, excluded-padding, NaN, signed-zero, first-winner,
  or fixed-divisor semantics<span data-glossary-navigation="true"> ([narrowing glossary](../../../../glossary.md#term-narrowing); [padding glossary](../../../../glossary.md#padding); [NaN glossary](../../../../glossary.md#term-nan); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- more than two derivative stages, distributional derivatives at discontinuities, or a new
  differentiability policy for comparisons, classification, Boolean expressions, <span data-glossary-navigation="true"> ([derivative glossary](../../../../glossary.md#term-derivative); [Boolean glossary](../../../../glossary.md#data-type--datatype))</span>`ARG_MAX`,
  `ONE_HOT`, or `WHERE` conditions
- architecture, dependency, module, Gradle, architecture-test, conformance-test, or integration-test
  changes<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [module glossary](../../../../glossary.md#term-module); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- a detailed task file for [CPU](../../../../glossary.md#term-cpu) 0008G1, [Compiler](../../../../glossary.md#term-compiler) 0006C, or Compiler 0007

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>, especially Model/Compiler ownership,
  compiler-owned pre-capture autograd, the flat graph model, and dependency direction<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [capture glossary](../../../../glossary.md#term-capture-meanings); [autograd glossary](../../../../glossary.md#autograd); [graph glossary](../../../../glossary.md#graph); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools))</span>
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Roadmap](../../../roadmap.md)
- [Compiler master plan](../master-plan.md)
- [Compiler 0005C](0005c-layout-window-indexing-scatter-ordering-and-stochastic-gradient-completion.md)
- [Compiler 0005D](0005d-attention-convolution-pooling-and-loss-gradient-completion.md)
- [Compiler 0005E](0005e-first-order-gradient-coverage-closure-checkpoint.md)
- [Compiler 0006](0006-explicit-functional-gradient-requests-and-higher-order-differentiation.md)
- [Compiler 0006B1](0006b1-pool3d-and-3d-window-forward-adoption-and-explicit-gradient-boundary.md)
- [Model capabilities](../../model/capabilities.md)
- [Model master plan](../../model/master-plan.md)
- [Model 0025J](../../model/tasks/0025j-first-class-ncdhw-max-average-pool3d-semantics.md)
- [Model 0025K](../../model/tasks/0025k-public-ncdhw-unfold3d-and-fold3d-window-transforms.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [General documentation style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- [Planning documentation style](../../../../developer-guide/documentation/planning-style.md)

## Architecture constraints

- Model remains the sole owner of operation meaning, public Tensor construction, immutable
  attributes, Shapes, scalar values, and numerical policy. Compiler owns pre-capture formula
  generation and closed derivative coverage.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [owner glossary](../../../../glossary.md#term-owner); [operation glossary](../../../../glossary.md#operation); [immutable glossary](../../../../glossary.md#term-immutable); [Shapes glossary](../../../../glossary.md#shape); [scalar glossary](../../../../glossary.md#term-scalar); [Compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [derivative glossary](../../../../glossary.md#term-derivative))</span>
- Every [adjoint](../../../../glossary.md#term-adjoint) is composed through public [`Tensor`](../../../../glossary.md#tensor) algebra. It may refer to exact captured input,
  exact same-occurrence producer output, exact immutable attrs, and incoming cotangent, but never
  instantiate a graph node or operation directly.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [producer glossary](../../../../glossary.md#term-producer); [cotangent glossary](../../../../glossary.md#cotangent); [graph glossary](../../../../glossary.md#graph); [node glossary](../../../../glossary.md#node))</span>
- Generated formulas are ordinary Tensor expression graphs. Capture, inference, validation,
  optimization, publication, and Planning handoff remain unchanged.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [validation glossary](../../../../glossary.md#term-validation); [publication glossary](../../../../glossary.md#term-publication); [Planning glossary](../../../../glossary.md#term-planning); [handoff glossary](../../../../glossary.md#term-handoff))</span>
- Complete-forward preflight must prove every current prerequisite for every requested stage before
  seed normalization, constants, formula Tensors, Tensor identifiers, or combined capture.<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [seed glossary](../../../../glossary.md#term-seed); [normalization glossary](../../../../glossary.md#term-normalization); [constants glossary](../../../../glossary.md#term-constant))</span>
- Comparison, classification, Boolean, arg-extrema, one-hot, and selection-condition expressions
  keep the completed 0005A/0006 non-differentiable-condition policy. Differentiable <span data-glossary-navigation="true"> ([Boolean glossary](../../../../glossary.md#data-type--datatype); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [differentiable glossary](../../../../glossary.md#derivative-and-chain-rule))</span>`WHERE`
  branches remain ordinary expressions.
- Compiler adds no dependency and no public API. If exact semantics require a new Model operation,
  new predicate, direct node construction, saved state, architecture change, or a path outside the
  authorized ceiling, stop and keep the affected signature deferred.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [signature glossary](../../../../glossary.md#term-signature))</span>

## Package impact

Existing package changed:

- `io.github.pho001.synaptik.compiler` — closed first-order inventory, preflight, and existing
  pooling/layout rule owners only.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [pooling glossary](../../../../glossary.md#term-pooling); [layout glossary](../../../../glossary.md#layout))</span>

Existing [Model](../../../../glossary.md#term-model) packages are read-only inputs:

- `io.github.pho001.synaptik.model.operation.pooling`
- `io.github.pho001.synaptik.model.operation.layout`
- `io.github.pho001.synaptik.model`

No package, public type, public method, [operation](../../../../glossary.md#operation) kind, attribute type, [registry](../../../../glossary.md#term-registry), [facade](../../../../glossary.md#term-facade), or [module](../../../../glossary.md#term-module)
is added.

## Exact [adjoint](../../../../glossary.md#term-adjoint) contract

For [NCDHW](../../../../glossary.md#conv3d--ncdhw) input `[N,C,D,H,W]`, [kernel](../../../../glossary.md#kernel) `[kD,kH,kW]`, [stride](../../../../glossary.md#term-stride) `[sD,sH,sW]`, [dilation](../../../../glossary.md#term-dilation)
`[dD,dH,dW]`, symmetric [padding](../../../../glossary.md#padding) `[pD,pH,pW]`, and output grid `[DOut,HOut,WOut]`, [canonical](../../../../glossary.md#term-canonical-form)
unfold columns have [Shape](../../../../glossary.md#shape) `[N,C*kD*kH*kW,DOut*HOut*WOut]`. Their exact [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) order is:

```text
id = od*sD - pD + kd*dD
ih = oh*sH - pH + kh*dH
iw = ow*sW - pW + kw*dW
q  = (((c*kD + kd)*kH + kh)*kW + kw)
r  = ((od*HOut + oh)*WOut + ow)
```

Depth precedes height, height precedes width, and width is fastest within both kernel and output
coordinates. All formulas must use the exact captured original Shape and [immutable](../../../../glossary.md#term-immutable) window attrs;
they must not rederive an equivalent but [identity](../../../../glossary.md#term-identity-meanings)-different symbolic Shape when the public method
accepts the original reference.

### Three-dimensional window adjoints

- For either direct `UNFOLD3D + Window3dAttrs` or typed
  `UNFOLD3D + Unfold3dAttrs`, the input [cotangent](../../../../glossary.md#cotangent) is:

  ```text
  dInput = incoming.fold3d(originalInput.shape(), window)
  ```

  Typed [padding](../../../../glossary.md#padding) is [immutable](../../../../glossary.md#term-immutable) configuration and has no cotangent. `fold3d` excludes geometrically
  padded and ceil-tail samples, so their constants contribute nothing to the input cotangent.<span data-glossary-navigation="true"> ([samples glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [constants glossary](../../../../glossary.md#term-constant))</span>
- For `FOLD3D + Fold3dAttrs`, the columns-input cotangent is:

  ```text
  dColumns = incoming.unfold3d(window)
  ```

  This is the direct positive-zero-padding overload. It is not an arbitrary typed-padding
  reconstruction: columns discarded by forward fold because they address padding or a ceil tail
  must receive exact positive zero.<span data-glossary-navigation="true"> ([positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- [Linear](../../../../glossary.md#term-linear-meanings) closure alternates these public operations. Generated `UNFOLD3D` differentiates through
  `FOLD3D`; generated `FOLD3D` differentiates through direct positive-zero `UNFOLD3D`.

### Average [Pool3d](../../../../glossary.md#pool3d--ncdhw-pooling) [gradient](../../../../glossary.md#term-gradient)

Let `K = kD*kH*kW` be the logical [kernel](../../../../glossary.md#kernel) [cardinality](../../../../glossary.md#term-cardinality) and `L = DOut*HOut*WOut`. Restore incoming
[cotangent](../../../../glossary.md#cotangent) to `[N,C,L]`. Construct the fixed divisor as a [Tensor](../../../../glossary.md#tensor) expression by expanding an exact
typed one over [Shape](../../../../glossary.md#shape) `[kD,kH,kW]` and reducing all three [axes](../../../../glossary.md#term-axis). Do not count in-bounds [samples](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) and
do not use a [host](../../../../glossary.md#term-host)-computed `long K` as the numerical divisor. Then:

```text
perOutput = reshape(incoming, [N,C,L]) / logicalKernelDivisor
columns   = expand/reshape(perOutput, [N,C*kD*kH*kW,L])
dInput    = columns.fold3d(originalInput.shape(), window)
```

Every logical kernel position receives `incoming/K`; padded and ceil-tail positions are discarded
by fold. Overlapping valid positions accumulate through public `fold3d`. The result retains the
input floating type through the existing [cotangent normalization](../../../../glossary.md#cotangent-normalization) boundary. [BFLOAT16](../../../../glossary.md#term-bfloat16), [FLOAT32](../../../../glossary.md#term-float32), and
[FLOAT64](../../../../glossary.md#term-float64) use their existing public-[operation](../../../../glossary.md#operation) promotion, accumulation, and [narrowing](../../../../glossary.md#term-narrowing) contracts; no
gradient-only cast, host arithmetic, alternative [accumulator](../../../../glossary.md#accumulator-type), or changed forward [rounding](../../../../glossary.md#term-rounding) rule is
introduced.

In particular, every emitted [scalar](../../../../glossary.md#term-scalar) and arithmetic expression stays in the original floating
type under the existing public promotion table. `fold3d` accumulates contributions in [canonical](../../../../glossary.md#term-canonical-form)
column order: FLOAT64 performs sequential [binary64](../../../../glossary.md#floating-point-precision-and-rounding) additions, FLOAT32 performs sequential [binary32](../../../../glossary.md#floating-point-precision-and-rounding)
additions, and BFLOAT16 widens the represented accumulator and operand to FLOAT32, adds once, then
narrows to BFLOAT16 after every contribution. The logical fixed divisor is represented through the
ordinary Tensor reduction/division rules. These [derivative](../../../../glossary.md#term-derivative)-expression rules neither reinterpret
nor change forward Average Pool3d's FLOAT32 accumulation/division for BFLOAT16/FLOAT32 inputs,
FLOAT64 accumulation/division for FLOAT64 input, or BFLOAT16 single result narrowing.

### Maximum [Pool3d](../../../../glossary.md#pool3d--ncdhw-pooling) [gradient](../../../../glossary.md#term-gradient)

Winner reconstruction must use the exact same-[occurrence](../../../../glossary.md#term-occurrence) forward output `producer.output(0)` and
must not re-[run](../../../../glossary.md#term-run) Pool3d. With `K` and `L` as above:

1. Unfold input with an exact typed negative-[infinity](../../../../glossary.md#term-infinity) [padding](../../../../glossary.md#padding) [scalar](../../../../glossary.md#term-scalar), reshape to `[N,C,K,L]`, and
   permute to `[N,C,L,K]` candidates.
2. Unfold an exact typed one-like input with direct positive-zero padding and the same window;
   reshape/permute identically and compare with zero to form the in-bounds mask.<span data-glossary-navigation="true"> ([mask glossary](../../../../glossary.md#term-boolean-mask))</span>
3. Reshape the same-occurrence output to `[N,C,L,1]`. An eligible candidate is in bounds and either
   both candidate and output are NaN, or they are numerically equal with exact signed-zero
   agreement. Signed-zero agreement is proved by reciprocal signs (<span data-glossary-navigation="true"> ([signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>`1/candidate == 1/output`) when
   both values compare equal to zero.
4. Convert eligibility to exact typed zero/one, take `argMax(-1, false, FIRST_INDEX)`, construct a
   public one-hot mask of static depth `K`, and intersect it with eligibility. The intersection is
   mandatory so an all-padding output selects nothing even though first-index arg-max returns zero.
5. Route reshaped incoming [cotangent](../../../../glossary.md#cotangent) through `where(selected, incoming, +0)`, inverse-permute and
   reshape to canonical columns, then fold into <span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>`originalInput.shape()`.

This selects the first exact eligible occurrence in depth-height-width order, excludes padding
even when a real input is [negative infinity](../../../../glossary.md#nan-infinity-and-signed-zero), treats [NaN](../../../../glossary.md#term-nan) as dominant through same-output matching,
distinguishes `+0.0` from `-0.0`, chooses the first equal occurrence, produces zero for all-padding
outputs, and accumulates overlapping winners through fold. Exact occurrence matching, not merely
numeric equality, is required.

## Higher-order and [preflight](../../../../glossary.md#term-preflight) contract

- Average Pool3d and the three window transforms are linear in their differentiable input. Their
  generated formulas remain ordinary public Tensor graphs; a second derivative with respect to
  that input is disconnected and follows the request's existing <span data-glossary-navigation="true"> ([Pool3d glossary](../../../../glossary.md#pool3d--ncdhw-pooling); [linear glossary](../../../../glossary.md#term-linear-meanings); [differentiable glossary](../../../../glossary.md#derivative-and-chain-rule); [Tensor glossary](../../../../glossary.md#tensor); [graphs glossary](../../../../glossary.md#graph); [second derivative glossary](../../../../glossary.md#term-second-derivative))</span>`ERROR` or exact typed `ZERO`
  policy.
- Maximum Pool3d's reconstructed selector is a non-differentiable constant under the existing
  comparison/selection policy. Away from winner changes, its derivative with respect to input is
  zero. At equal-value ties, signed-zero boundaries, NaN selection changes, and other
  discontinuities, the same selected zero/almost-everywhere convention applies; no distributional
  derivative is represented. <span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [derivative glossary](../../../../glossary.md#term-derivative); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [NaN glossary](../../../../glossary.md#term-nan))</span>`DisconnectedPolicy.ZERO` returns exact typed zero and `ERROR`
  rejects the disconnected second derivative. A differentiable incoming branch still propagates
  through the selected `WHERE` branch normally.
- Preflight validates exact signatures, descriptors, floating eligibility, attrs, Shapes, output
  alignment, and public-formula availability for every reachable requested stage.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [descriptors glossary](../../../../glossary.md#term-descriptor); [Shapes glossary](../../../../glossary.md#shape); [alignment glossary](../../../../glossary.md#term-alignment))</span>
- Checked host multiplication may prove only the positive, statically representable structural
  kernel volume required by reshape and one-hot depth. It must not become average's numerical
  divisor. An overflow, non-positive/unrepresentable one-hot depth, unprovable Shape equality,
  malformed manual occurrence, or unavailable public expression fails before allocation.<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [kernel glossary](../../../../glossary.md#kernel); [overflow glossary](../../../../glossary.md#term-overflow); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Existing canonical Dimension expressions and forward inference constraints remain authoritative.
  Compiler does not bind symbols, materialize data, or defer gradient-formula legality to Runtime.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [Dimension glossary](../../../../glossary.md#dimension); [inference glossary](../../../../glossary.md#term-inference); [Compiler glossary](../../../../glossary.md#term-compiler); [gradient glossary](../../../../glossary.md#term-gradient); [Runtime glossary](../../../../glossary.md#term-runtime))</span>

## Affected files

Production, exactly these existing files when implementation requires their owned change:

1. `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/AutogradPreflight.java`
2. `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/FirstOrderGradientCoverage.java`
3. `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/PoolingGradientRules.java`
4. `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/LayoutGradientRules.java`

Tests, exactly these existing files:

5. `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/AutogradPreflightTest.java`
6. `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/FirstOrderGradientCoverageTest.java`
7. `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/ConvolutionAndPoolingGradientRulesTest.java`
8. `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/LayoutWindowGradientRulesTest.java`
9. `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/Pool3dAndWindow3dCompilerTest.java`

Documentation and synchronized [planning](../../../../glossary.md#term-planning):

10. `docs/api/compile-api.md`
11. `docs/api/tensor-api.md`
12. `docs/glossary.md`
13. `docs/planning/modules/compiler/tasks/0006b2-pool3d-and-3d-window-gradient-closure.md`
14. `docs/planning/modules/compiler/master-plan.md`
15. `docs/planning/modules/model/capabilities.md`
16. `docs/planning/modules/model/master-plan.md`
17. `docs/planning/roadmap.md`

Review without modification unless a documented conflict forces the task to stop:
`FirstOrderAutograd`, `GraphCompiler`, `CapturedGraphInference`,
`StructuredOperationInference`, `LayoutInference`, public [Tensor](../../../../glossary.md#tensor) [Pool3d](../../../../glossary.md#pool3d--ncdhw-pooling)/window methods and
Javadocs, [Compile](../../../../glossary.md#compile) and Training APIs, architecture documents, [Model](../../../../glossary.md#term-model) tests, architecture tests,
[backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance), integration tests, [Gradle](../../../../glossary.md#term-gradle) files, and other [modules](../../../../glossary.md#term-module).

## Maximum [scope](../../../../glossary.md#term-scope)

The hard ceiling is 17 paths: four existing production files, five existing test files, and eight
documentation/[planning](../../../../glossary.md#term-planning) files. No new Java type or test file is authorized. Fewer paths are
preferred when existing Javadocs or prose remain exact, but every no-change conclusion must be
recorded. Crossing the ceiling, changing [Model](../../../../glossary.md#term-model) or public [API](../../../../glossary.md#term-application-programming-interface-api), or needing a new [operation](../../../../glossary.md#operation) is a stop
condition requiring a new planning decision.

## Acceptance criteria

- The exact five signatures are production-supported and owner-dispatched; forward remains
  40/115/137, support is exactly 38/111/133, and deferred is exactly the four recurrent/Conv3d
  signatures with disjoint union equality against the reflection-derived Model inventory.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [owner glossary](../../../../glossary.md#term-owner); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [Model glossary](../../../../glossary.md#term-model))</span>
- Both backward-capable modes accept the five signatures when all prerequisites hold, including on
  unrelated requested branches, while recurrent and Conv3d still fail before any derivative
  allocation.
- Both unfold variants and fold produce the exact adjoints above, including positive-zero padded
  column cotangents for fold and overlap accumulation for unfold.
- Average [Pool3d](../../../../glossary.md#pool3d--ncdhw-pooling) divides by fixed logical `kD*kH*kW`, never valid-sample count, and proves padding,
  dilation, ceil-tail, overlap, symbolic Shape, all-padding, and all three floating-type cases.<span data-glossary-navigation="true"> ([sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [padding glossary](../../../../glossary.md#padding); [dilation glossary](../../../../glossary.md#term-dilation); [Shape glossary](../../../../glossary.md#shape))</span>
- Maximum Pool3d proves same-occurrence output use, excluded padding, real negative infinity,
  all-padding zero, NaN dominance, signed-zero ordering, first equal depth-height-width occurrence,
  exact occurrence matching, overlap accumulation, and all three floating types.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [negative infinity glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [NaN glossary](../../../../glossary.md#term-nan); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Formula graphs contain only public Tensor operations and captured references. Tests prove there
  is no saved index, backward-only kind, direct graph-node construction, hidden helper dispatch, or
  forward Pool3d reconstruction.<span data-glossary-navigation="true"> ([graphs glossary](../../../../glossary.md#graph); [Tensor glossary](../../../../glossary.md#tensor); [node glossary](../../../../glossary.md#node); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Two-stage tests prove [linear](../../../../glossary.md#term-linear-meanings) second-[derivative](../../../../glossary.md#term-derivative) `ERROR`/`ZERO` behavior and maximum's fixed-mask
  almost-everywhere convention, including ties/discontinuities and propagation through a
  differentiable incoming branch.<span data-glossary-navigation="true"> ([mask glossary](../../../../glossary.md#term-boolean-mask); [differentiable glossary](../../../../glossary.md#derivative-and-chain-rule))</span>
- Every unprovable structural or semantic prerequisite fails in preflight before seed/constants,
  formula Tensor, Tensor-ID, or combined-capture allocation.<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [seed glossary](../../../../glossary.md#term-seed); [constants glossary](../../../../glossary.md#term-constant); [capture glossary](../../../../glossary.md#term-capture-meanings))</span>
- Directly affected Javadocs are meaningful and complete. Compile/Tensor API and glossary remain
  synchronized with current-versus-planned boundaries; unchanged documentation has an explicit
  reasoned no-change conclusion.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- No public API, Model behavior, forward inventory, dependency, architecture, backend capability,
  Gradle configuration, or CPU task file changes.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [backend capability glossary](../../../../glossary.md#backend-capability); [Gradle glossary](../../../../glossary.md#term-gradle); [CPU glossary](../../../../glossary.md#term-cpu))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

Implementation context, focused affected [Compiler](../../../../glossary.md#term-compiler) suites:

```bash
./gradlew :modules:compiler:test \
  --tests io.github.pho001.synaptik.compiler.AutogradPreflightTest \
  --tests io.github.pho001.synaptik.compiler.FirstOrderGradientCoverageTest \
  --tests io.github.pho001.synaptik.compiler.ConvolutionAndPoolingGradientRulesTest \
  --tests io.github.pho001.synaptik.compiler.LayoutWindowGradientRulesTest \
  --tests io.github.pho001.synaptik.compiler.Pool3dAndWindow3dCompilerTest
```

Focused unchanged [Model](../../../../glossary.md#term-model) boundary evidence:

```bash
./gradlew :modules:model:test \
  --tests io.github.pho001.synaptik.model.operation.pooling.Pool3dSemanticsTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorMaxPool3dExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorAveragePool3dExpressionTest \
  --tests io.github.pho001.synaptik.model.operation.layout.WindowTransformSemanticsTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorWindowExpressionTest
```

[Run](../../../../glossary.md#term-run) the final affected-[module](../../../../glossary.md#term-module) suite exactly once after focused evidence:

```bash
./gradlew :modules:compiler:test
```

Documentation context, after reviewing the implementation evidence and changing no executable
Java unless it discovers a concrete defect that returns to implementation context:

```bash
./gradlew :modules:compiler:javadoc :modules:model:javadoc
python3 /tmp/validate_synaptik_markdown.py
git diff --check
```

Inventory/reflection evidence must enumerate all current Model declarations, prove 40/115/137,
prove exact supported 38/111/133 and deferred four, verify [owner](../../../../glossary.md#term-owner) [dispatch](../../../../glossary.md#term-dispatch), and verify the public
[Tensor](../../../../glossary.md#tensor) method surface is unchanged. The source-backed/reflection-driven assertions in
`FirstOrderGradientCoverageTest` and existing Model [signature](../../../../glossary.md#term-signature)/surface tests are authoritative;
record their exact results rather than hand-counting.

Also validate Markdown links and anchors, balanced fences, exact 17-path ceiling, no unexpected
files, synchronized statuses/order, and absence of
`docs/planning/backends/cpu/tasks/0008g1-*.md`. Do not rerun the repository-wide suite: this task
changes one module, no [dependency](../../../../glossary.md#term-dependency) or shared build boundary, and the [planning](../../../../glossary.md#term-planning) guide assigns the
full repository tier to a later recorded [checkpoint](../../../../glossary.md#term-checkpoint) or [CI](../../../../glossary.md#term-continuous-integration-ci).

## Dependencies

- Complete [Model 0025J](../../model/tasks/0025j-first-class-ncdhw-max-average-pool3d-semantics.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- Complete [Model 0025K](../../model/tasks/0025k-public-ncdhw-unfold3d-and-fold3d-window-transforms.md)
- Complete [Compiler 0005D](0005d-attention-convolution-pooling-and-loss-gradient-completion.md)
- Complete [Compiler 0006B1](0006b1-pool3d-and-3d-window-forward-adoption-and-explicit-gradient-boundary.md)
- Existing completed [Compiler](../../../../glossary.md#term-compiler) 0005A/0005C/0005E/0006 policies and public [Tensor](../../../../glossary.md#tensor) algebra

## Follow-up tasks

- Draft [Compiler](../../../../glossary.md#term-compiler) 0006C remains the separate [Conv3d](../../../../glossary.md#conv3d--ncdhw) [adjoint](../../../../glossary.md#term-adjoint)-expressibility/closure decision.
- Draft Compiler 0007 remains later exact algebra work.
- Draft CPU 0008G1 follows this task and owns Pool1d composition validation plus Pool3d execution;
  it remains without a detailed task file.<span data-glossary-navigation="true"> ([Pool1d composition glossary](../../../../glossary.md#pool1d-composition); [validation glossary](../../../../glossary.md#term-validation); [Pool3d glossary](../../../../glossary.md#pool3d--ncdhw-pooling))</span>
- Draft [CPU](../../../../glossary.md#term-cpu) 0008H remains after CPU 0008G1. This task must not reorder or detail either CPU task.

## Architecture impact

No architecture change is expected. This task fills five already assigned [Compiler](../../../../glossary.md#term-compiler)-owned [gradient](../../../../glossary.md#term-gradient)
slots using existing public [Model](../../../../glossary.md#term-model) algebra and the established pre-[capture](../../../../glossary.md#term-capture-meanings) [autograd](../../../../glossary.md#autograd) boundary. It
adds no [module](../../../../glossary.md#term-module) edge, public surface, [graph](../../../../glossary.md#graph) layer, [runtime](../../../../glossary.md#term-runtime) behavior, or [backend](../../../../glossary.md#backend) contract. Any need
to alter `ARCHITECTURE.md`, architecture tests, or an [ADR](../../../../glossary.md#term-architecture-decision-record-adr) is a stop condition, not implied [scope](../../../../glossary.md#term-scope).

## Implementation prompt

Execute this task in a separate clean implementation context. Do not use a GSD workflow, [commit](../../../../glossary.md#term-commit),
or push. Read `AGENTS.md`, `ARCHITECTURE.md`, the current architecture plan, [planning](../../../../glossary.md#term-planning) guide,
roadmap, this task, its dependencies, the directly affected source/tests, and public [Tensor](../../../../glossary.md#tensor)
[Pool3d](../../../../glossary.md#pool3d--ncdhw-pooling)/window contracts in full. Implement exactly the five formulas and inventory/[preflight](../../../../glossary.md#term-preflight)
transition specified here within the 17-path ceiling. Use only public Tensor expressions and exact
captured references. Preserve all unrelated behavior and stop on architectural uncertainty.

[Run](../../../../glossary.md#term-run) the focused [Compiler](../../../../glossary.md#term-compiler) suites, focused unchanged [Model](../../../../glossary.md#term-model) boundary suites, and one final full
Compiler suite. Record exact commands/results and all reasoned no-change conclusions. Then hand
the same uncommitted change to a distinct clean documentation-focused context. That context must
read the documentation rules and General/[API](../../../../glossary.md#term-application-programming-interface-api)-[Javadoc](../../../../glossary.md#term-javadoc)/Planning profiles, independently inspect the
source and evidence, finalize affected Javadocs/API/glossary/planning, run Javadoc/Markdown/[scope](../../../../glossary.md#term-scope)/
hygiene checks without repeating successful Java suites, and leave the overall task Complete only
when implementation, documentation, and [validation](../../../../glossary.md#term-validation) all pass.

## Local decisions

- The task is one cohesive closure because Pool3d gradients require the same public 3D
  unfold/fold algebra whose adjoints are needed for transitive and higher-order differentiation.<span data-glossary-navigation="true"> ([Pool3d glossary](../../../../glossary.md#pool3d--ncdhw-pooling); [gradients glossary](../../../../glossary.md#term-gradient))</span>
- `FOLD3D`'s adjoint uses direct positive-zero unfold; typed unfold padding is configuration, not a
  differentiable input.<span data-glossary-navigation="true"> ([adjoint glossary](../../../../glossary.md#term-adjoint); [padding glossary](../../../../glossary.md#padding); [differentiable glossary](../../../../glossary.md#derivative-and-chain-rule))</span>
- Average uses a [Tensor](../../../../glossary.md#tensor)-computed fixed logical divisor. [Host](../../../../glossary.md#term-host) [kernel](../../../../glossary.md#kernel) volume is structural only.
- Maximum reconstructs the selected occurrence from the exact same-occurrence output and an
  explicit in-bounds mask. Same numeric value alone is insufficient for NaN and signed zero.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [mask glossary](../../../../glossary.md#term-boolean-mask); [NaN glossary](../../../../glossary.md#term-nan); [signed zero glossary](../../../../glossary.md#term-signed-zero))</span>
- The first-winner mask is non-differentiable under the already accepted policy. Higher-order
  behavior is the fixed-mask, almost-everywhere convention with existing disconnected handling.
- The production inventory transition is exactly 37/107/128 -> 38/111/133; forward 40/115/137 is
  unchanged; only recurrent ×3 and Conv3d remain deferred.<span data-glossary-navigation="true"> ([Conv3d glossary](../../../../glossary.md#conv3d--ncdhw))</span>
- No repository-wide [validation](../../../../glossary.md#term-validation) is required because there is no boundary/build/[dependency](../../../../glossary.md#term-dependency) change.

## Known limitations

- Maximum [Pool3d](../../../../glossary.md#pool3d--ncdhw-pooling) [derivatives](../../../../glossary.md#term-derivative) are not distributional at ties, [NaN](../../../../glossary.md#term-nan) boundaries, or winner changes.
- No saved maximum indices are exposed, so exact reconstruction intentionally uses public algebra
  and the same-occurrence output.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Recurrent and [Conv3d](../../../../glossary.md#conv3d--ncdhw) [gradients](../../../../glossary.md#term-gradient) remain unavailable and [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation).
- This task adds no [CPU](../../../../glossary.md#term-cpu) execution. Successful formula [capture](../../../../glossary.md#term-capture-meanings) is not a [backend capability](../../../../glossary.md#backend-capability) claim.

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) preparation established from current source and completed task evidence:

- complete forward inventory: 40 families / 115 [constants](../../../../glossary.md#term-constant) / 137 [signatures](../../../../glossary.md#term-signature);
- current support: 37 / 107 / 128;
- this exact addition: [Pool3d](../../../../glossary.md#pool3d--ncdhw-pooling) `+1/+2/+2`, 3D-window `+0/+2/+3`;
- resulting support: 38 / 111 / 133, with exactly four deferred signatures;
- the five public formulas are expressible through existing unfold3d/fold3d, reshape, permute,
  expand, reduction, comparison/classification, arg-max, one-hot, Boolean, and where operations;<span data-glossary-navigation="true"> ([Boolean glossary](../../../../glossary.md#data-type--datatype))</span>
- no [CPU](../../../../glossary.md#term-cpu) 0008G1 detailed task file exists.

Implementation and documentation evidence follows.

Implementation-context evidence on 2026-08-30:

- `FirstOrderGradientCoverageTest` passed its reflection-derived partition assertion: complete
  Model forward inventory remains exactly 40 families / 115 constants / 137 signatures;
  production first-order coverage is exactly 38 / 111 / 133; the disjoint deferred set is exactly<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [Model glossary](../../../../glossary.md#term-model))</span>
  `RNN_TANH`, `GRU_RESET_AFTER`, [`LSTM`](../../../../glossary.md#term-long-short-term-memory-lstm), and [`CONV3D`](../../../../glossary.md#conv3d--ncdhw).
- The exact focused Compiler command passed 34 tests across the five requested suites with zero
  failures, errors, or skips: <span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>`AutogradPreflightTest` 14,
  `FirstOrderGradientCoverageTest` 6, `ConvolutionAndPoolingGradientRulesTest` 6,
  `LayoutWindowGradientRulesTest` 3, and `Pool3dAndWindow3dCompilerTest` 5.
- The exact focused unchanged Model command was up-to-date and its retained JUnit XML reports 54
  tests with zero failures, errors, or skips: <span data-glossary-navigation="true"> ([JUnit glossary](../../../../glossary.md#term-junit))</span>`Pool3dSemanticsTest` 4,
  `TensorMaxPool3dExpressionTest` 6, `TensorAveragePool3dExpressionTest` 6,
  `WindowTransformSemanticsTest` 14, and `TensorWindowExpressionTest` 24.
- The single final `./gradlew :modules:compiler:test` run passed 241 tests across 35 suites with
  zero failures, errors, or skips.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
- Formula-structure tests prove both unfold attribute variants fold into the exact captured input
  Shape, fold uses direct <span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape))</span>`UNFOLD3D + Window3dAttrs`, average constructs its logical divisor by
  expanding one across `[kD,kH,kW]` and reducing, and maximum uses the same-occurrence output,
  typed negative-infinity candidates, a direct-positive-zero in-bounds probe, NaN and reciprocal
  signed-zero matching, first-index arg-max, static-depth one-hot, eligibility intersection,
  public <span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [infinity glossary](../../../../glossary.md#term-infinity); [NaN glossary](../../../../glossary.md#term-nan); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>`WHERE`, inverse [permutation](../../../../glossary.md#permutation)/reshape, and overlap `FOLD3D`. No generated Pool3d
  reconstruction occurs.
- Two-stage tests prove disconnected [linear](../../../../glossary.md#term-linear-meanings) `UNFOLD3D` uses the existing `ERROR`/typed-`ZERO`
  policy and a maximum fixed mask still permits a differentiable incoming branch to close through
  a connected second stage.<span data-glossary-navigation="true"> ([mask glossary](../../../../glossary.md#term-boolean-mask); [differentiable glossary](../../../../glossary.md#derivative-and-chain-rule))</span>
- `javap -public` still reports exactly 213 public [`Tensor`](../../../../glossary.md#tensor) declarations. Neither rule owner
  imports or constructs <span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>[`Operation`](../../../../glossary.md#operation), [`TensorProducer`](../../../../glossary.md#producer--tensorproducer), or graph capture state. Existing central
  dispatch remains <span data-glossary-navigation="true"> ([graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>`POOLING -> PoolingGradientRules` and `LAYOUT -> LayoutGradientRules`.
- The exact changed-path audit contains 14 paths, all within the authorized 17-path ceiling;
  `git diff --check` passes; no `docs/planning/backends/cpu/tasks/0008g1-*.md` exists.
- Reasoned no-change conclusions: Model source/tests and public Tensor/Shape/Dimension contracts
  are unchanged because all derivatives compose existing public operations; <span data-glossary-navigation="true"> ([Dimension glossary](../../../../glossary.md#dimension); [derivatives glossary](../../../../glossary.md#term-derivative))</span>`FirstOrderAutograd`,
  `GraphCompiler`, inference owners, Compile/Training APIs, architecture documents/tests, backend
  conformance, integration tests, Gradle files, dependencies, and every other module retain their
  existing contracts. Compile/Tensor API prose and glossary need only the mandatory independent
  documentation review/finalization; this implementation context did not revise them.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [Compile glossary](../../../../glossary.md#compile); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Implementation notes

Implementation-context work is complete and remains uncommitted for the mandatory independent
documentation-focused [handoff](../../../../glossary.md#term-handoff). The five exact [signatures](../../../../glossary.md#term-signature) are now in the closed coverage inventory;
[Pool3d](../../../../glossary.md#pool3d--ncdhw-pooling) dispatches to the existing [pooling](../../../../glossary.md#term-pooling) [owner](../../../../glossary.md#term-owner) and all three volumetric window signatures [dispatch](../../../../glossary.md#term-dispatch)
to the existing [layout](../../../../glossary.md#layout) owner. Complete-forward rejection now retains only fixed recurrence and
[Conv3d](../../../../glossary.md#conv3d--ncdhw).

`AutogradPreflight` independently revalidates each selected forward [occurrence](../../../../glossary.md#term-occurrence), checks the
three-factor structural [kernel](../../../../glossary.md#kernel) product, and proves the public 3D unfold/fold counterpart restores
the exact input [descriptor](../../../../glossary.md#term-descriptor) before [derivative](../../../../glossary.md#term-derivative) allocation. `PoolingGradientRules` implements the
[rank](../../../../glossary.md#term-rank)-specific average and maximum formulas without a backward kind, saved index, hidden output,
direct [graph](../../../../glossary.md#graph) construction, or [host](../../../../glossary.md#term-host) numerical divisor. `LayoutGradientRules` implements the two
UNFOLD3D adjoints and one FOLD3D [adjoint](../../../../glossary.md#term-adjoint) with exact captured [Shape](../../../../glossary.md#shape)/window references and ignores
typed [padding](../../../../glossary.md#padding) as configuration.

The mandatory independent documentation-focused review finalized the changed source Javadocs,
[Compile](../../../../glossary.md#compile)/[Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api) explanations, glossary impact, synchronized [planning](../../../../glossary.md#term-planning) status/evidence, and this
completion summary without changing executable Java behavior. It found no concrete executable
defect and therefore did not repeat the successful Java suites.

Documentation-context evidence on 2026-08-30:

- The isolated documentation review independently inspected all four production sources, all five
  tests, relevant Model/Compiler contracts, public Tensor and Compile surfaces, and the retained
  implementation evidence. No executable defect, architecture conflict, or stale-evidence risk
  was found.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler))</span>
- `./gradlew :modules:compiler:javadoc :modules:model:javadoc` completed successfully. Generated
  Compiler Javadocs contain the finalized Pool3d/3D-window contracts, including the corrected
  two- and three-dimensional pooling owner description; Model Javadocs remain unchanged and
  current.
- `/tmp/validate_synaptik_markdown.py` validates the seven changed Markdown files without the
  repository's long-standing repeated-example headings. Its repository-wide run reports only
  pre-existing duplicate-heading diagnostics; <span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>`docs/api/tensor-api.md` changed no heading, and
  its task edits introduce no link-target, fence, or final-newline error. A separate fragment
  check validated every local Markdown anchor across all eight changed documentation/planning
  files.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>
- Generated-Javadoc/source inspection confirms both UNFOLD3D adjoints, direct-positive-zero
  FOLD3D adjoint, logical fixed-divisor Average Pool3d formula, explicit eligible-first-winner
  Maximum Pool3d formula, and fixed-mask higher-order policy are described consistently with the
  implementation.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [mask glossary](../../../../glossary.md#term-boolean-mask))</span>
- `javap -public` reports exactly 213 public `Tensor` declarations. Retained JUnit XML still
  reports the final Compiler 35 suites / 241 tests and focused Model 5 suites / 54 tests, with
  zero failures, errors, or skips. The authoritative reflection-derived test evidence remains
  40/115/137 forward, 38/111/133 supported, and exactly recurrent RNN/GRU/LSTM plus Conv3d
  deferred.<span data-glossary-navigation="true"> ([JUnit glossary](../../../../glossary.md#term-junit); [RNN glossary](../../../../glossary.md#term-recurrent-neural-network-rnn); [GRU glossary](../../../../glossary.md#term-gated-recurrent-unit-gru); [LSTM glossary](../../../../glossary.md#term-long-short-term-memory-lstm))</span>
- The final audit contains exactly 17 authorized paths: four production sources, five tests, and
  eight documentation/planning files. The Git index is empty, `git diff --check` passes, Compiler
  0006C and 0007 remain Draft without detailed specifications, and no CPU 0008G1 detailed task
  file exists. CPU 0008G1 remains the next Draft frontier.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- Reasoned no-change conclusions: public Tensor, Compile, and Training Java APIs remain exact;
  Model source/tests, architecture contracts/tests, backend conformance, integration tests,
  Gradle/build/dependencies, CPU planning files, and unrelated modules require no change. The
  existing glossary terms cover the behavior, so only their current status and explanation were
  synchronized.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module))</span>

## Completion summary

- Completed changes: closed the five Pool3d/3D-window first-order signatures through public Tensor
  algebra, finalized their preflight, formula, exceptional-value, higher-order, and inventory
  documentation, and synchronized current/planned status.<span data-glossary-navigation="true"> ([Pool3d glossary](../../../../glossary.md#pool3d--ncdhw-pooling); [signatures glossary](../../../../glossary.md#term-signature); [Tensor glossary](../../../../glossary.md#tensor); [preflight glossary](../../../../glossary.md#term-preflight))</span>
- Files changed or created: exactly four Compiler production sources, five Compiler tests, and the
  eight authorized documentation/planning paths listed under Package and documentation impact.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning))</span>
- Validation: reused the successful focused Compiler 34-test, focused Model 54-test, and final
  Compiler 241-test evidence; regenerated Compiler/Model Javadocs; inspected generated output and
  public surface; validated Markdown links, anchors, fences, and final newlines; confirmed exact
  inventory, scope, empty index, whitespace, task ordering, and absence of a CPU 0008G1 task file.<span data-glossary-navigation="true"> ([Validation glossary](../../../../glossary.md#term-validation); [Model glossary](../../../../glossary.md#term-model); [scope glossary](../../../../glossary.md#term-scope); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Unresolved issues: none.
- Required follow-up: none for this task. Compiler 0006C, Compiler 0007, CPU 0008G1, and CPU 0008H
  remain separately owned Draft work in the recorded order.

Status: Complete
