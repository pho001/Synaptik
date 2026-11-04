# Task 0036: Extended 3D [Inference](../../../../glossary.md#term-inference)

## Reader context

Only bias-free accelerator CONV3D was the smallest candidate; Pool3d was assessed separately
and window unfold/fold remained other work. NCDHW names batch, channel, depth, height and width
axes; OIDHW names the weight's output-channel, grouped-input-channel and spatial axes. Those
orders describe tensors, not a proof of the arithmetic used for their stored values.

Model owned geometry, padding and numerical meaning; Metal's direct descriptor mapping established
structural compatibility only. A higher-dimensional pooling API with a singleton dimension did
not by itself prove the required 3D pooling behavior. No device experiment ran for this task.
The [current 3D and selected-derivative support ledger](../../../../backend-guide/production-support.md#metal-production-target)
provides the later bounded qualification path while preserving these original independent blockers.

## Status

Blocked

## Change class

Class C if unblocked — the smallest candidate would add [accelerator](../../../../glossary.md#term-accelerator) [`CONV3D`](../../../../glossary.md#conv3d--ncdhw), evolve private
[schema](../../../../glossary.md#term-schema)/route [identities](../../../../glossary.md#term-identity-meanings), and change native [graphs](../../../../glossary.md#graph). [Pool3d](../../../../glossary.md#pool3d--ncdhw-pooling) or custom 3D-window execution would add
separate numerical or [kernel](../../../../glossary.md#kernel) responsibilities. This [planning](../../../../glossary.md#term-planning)-only blocker changes no executable
behavior and ran no [device](../../../../glossary.md#term-device) probe.

## Goal

Assess the smallest first-class [Metal](../../../../glossary.md#term-metal) 3D [inference](../../../../glossary.md#term-inference) [slice](../../../../glossary.md#slice): unbiased, no-grad, [canonical](../../../../glossary.md#term-canonical-form), positive,
fully static [rank](../../../../glossary.md#term-rank)-five [`FLOAT32`](../../../../glossary.md#term-float32) [`CONV3D`](../../../../glossary.md#conv3d--ncdhw) under [`ACCELERATOR`](../../../../glossary.md#term-accelerator). Also close the adjacent
`MAX_POOL3D`/`AVERAGE_POOL3D` selector question without treating a four-[spatial](../../../../glossary.md#term-spatial-axes)-dimensional [API](../../../../glossary.md#term-application-programming-interface-api) or
generic stencil as an undocumented three-dimensional [pooling](../../../../glossary.md#term-pooling) contract.

Direct [MPSGraph](../../../../glossary.md#term-mpsgraph) Conv3d maps the complete structural candidate but inherits Task 0031's [opaque](../../../../glossary.md#term-opaque)
[shape](../../../../glossary.md#shape)-dependent [contraction](../../../../glossary.md#contraction-dimension) blocker. Pooling has no named 3D selector, and Pooling4D/stencil do not
establish exact profile-common [Pool3d](../../../../glossary.md#pool3d--ncdhw-pooling) semantics. [Lean](../../../../glossary.md#proof-assistant-and-model-checking-tools) policy blocks both before a probe. Task 0036
is Blocked and exposes no [Ready](../../../../glossary.md#term-ready) Metal task.

## Smallest candidate if unblocked

- Admit exactly `Conv3dKind.CONV3D + Conv3dAttrs`, ordered `[input, weight] -> [output]`.
- Require [`ACCELERATOR`](../../../../glossary.md#term-accelerator); `STRICT_IEEE` remains unsupported.
- Require [canonical](../../../../glossary.md#term-canonical-form) dense [contiguous](../../../../glossary.md#term-contiguous), zero-offset, non-[view](../../../../glossary.md#term-view-meanings) [`FLOAT32`](../../../../glossary.md#term-float32) rank-five NCDHW input,
  OIDHW weight, and NCDHW output with fully static positive extents and <span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [NCDHW glossary](../../../../glossary.md#conv3d--ncdhw); [weight glossary](../../../../glossary.md#term-weight))</span>`requiresGrad=false`.
- Independently validate grouped channels and derive checked `[N,Cout,Dout,Hout,Wout]` from
  positive stride/dilation/groups, non-negative symmetric padding, and positive weight kernels.<span data-glossary-navigation="true"> ([stride glossary](../../../../glossary.md#term-stride); [dilation glossary](../../../../glossary.md#term-dilation); [padding glossary](../../../../glossary.md#padding); [kernels glossary](../../../../glossary.md#kernel))</span>
- Admit every documented Model-valid geometry fitting checked Java/native conversions and buffers;
  add no limit inferred from execution.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [buffers glossary](../../../../glossary.md#term-buffer))</span>
- Retain the whole-[partition](../../../../glossary.md#partition) [MPSGraph](../../../../glossary.md#term-mpsgraph) [lifecycle](../../../../glossary.md#lifecycle), FLOAT32 transfer, and profile-free [Runtime](../../../../glossary.md#term-runtime)/[Trace](../../../../glossary.md#trace).

## Non-goals

- Biased [`CONV3D`](../../../../glossary.md#conv3d--ncdhw), [fused](../../../../glossary.md#specialization-and-fusion) [epilogue](../../../../glossary.md#term-epilogue), hidden add, [gradients](../../../../glossary.md#term-gradient)/training, saved values, or hidden outputs.
- Claiming [Pool3d](../../../../glossary.md#pool3d--ncdhw-pooling) from a plausible but undocumented Pooling4D/stencil composition.
- `UNFOLD3D`/`FOLD3D`, overlap-add, generic windows, or [convolution](../../../../glossary.md#term-channels-first-convolution-layers) [decomposition](../../../../glossary.md#term-decomposition).
- Dynamic/zero extents, views, another layout/type/profile, transfer widening, fallback, custom
  kernels, performance claims, a device matrix, single-shape narrowing, or relaxed oracles.<span data-glossary-navigation="true"> ([views glossary](../../../../glossary.md#term-view-meanings); [layout glossary](../../../../glossary.md#layout); [widening glossary](../../../../glossary.md#term-widening); [kernels glossary](../../../../glossary.md#kernel); [device glossary](../../../../glossary.md#term-device); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [shape glossary](../../../../glossary.md#shape); [narrowing glossary](../../../../glossary.md#term-narrowing))</span>

## Contracts, dependencies, and integration

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence) keeps
  Model as semantic owner and requires fail-closed backend support.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [fail-closed glossary](../../../../glossary.md#term-fail-closed); [backend glossary](../../../../glossary.md#backend))</span>
- [Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)
  give [`CONV3D`](../../../../glossary.md#conv3d--ncdhw) only family reassociation/corresponding-FMA freedom plus row-scoped FLOAT32 DAZ/FTZ;
  pooling and window movement receive no accelerator relaxation.<span data-glossary-navigation="true"> ([reassociation glossary](../../../../glossary.md#term-reassociation); [FMA glossary](../../../../glossary.md#term-fused-multiply-add-fma); [FLOAT32 glossary](../../../../glossary.md#term-float32); [DAZ glossary](../../../../glossary.md#term-daz); [FTZ glossary](../../../../glossary.md#term-ftz); [pooling glossary](../../../../glossary.md#term-pooling); [accelerator glossary](../../../../glossary.md#term-accelerator))</span>
- [Metal](../../../../architecture/contracts/backend-execution.md#metal-backend)<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal))</span> owns lowering and
  execution, not meaning; <span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering))</span>[Prepare](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare))</span>
  keeps route selection cold.
- [Model](../../../../glossary.md#term-model) [Conv3d](../../../modules/model/tasks/0025h-ncdhw-conv3d-semantics-and-tensor-expressions.md),
  [Pool3d](../../../modules/model/tasks/0025j-first-class-ncdhw-max-average-pool3d-semantics.md)<span data-glossary-navigation="true"> ([Pool3d glossary](../../../../glossary.md#pool3d--ncdhw-pooling))</span>,
  and [3D windows](../../../modules/model/tasks/0025k-public-ncdhw-unfold3d-and-fold3d-window-transforms.md)
  own signatures, geometry, attributes, and numerical meaning.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature))</span>
- [Compiler](../../../../glossary.md#term-compiler) [Conv3d](../../../modules/compiler/tasks/0006b-conv3d-forward-adoption-and-explicit-gradient-boundary.md),
  [Pool3d/windows](../../../modules/compiler/tasks/0006b1-pool3d-and-3d-window-forward-adoption-and-explicit-gradient-boundary.md),
  and [gradient closure](../../../modules/compiler/tasks/0006b2-pool3d-and-3d-window-gradient-closure.md)<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient))</span>
  own inference, canonical publication, and generated gradients.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [canonical glossary](../../../../glossary.md#term-canonical-form); [publication glossary](../../../../glossary.md#term-publication))</span>
- [CPU](../../../../glossary.md#term-cpu) [Conv3d](../../cpu/tasks/0008a-portable-channels-first-dimensional-convolution-closure.md)
  and [Pool3d](../../cpu/tasks/0008g1-portable-pool1d-composition-validation-and-pool3d-generated-execution.md)
  are exact rank-specific private precedents, not MPSGraph authorization.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph))</span>
- Depends on: Metal 0025; Model 0025H/J/K; Compiler 0006B/B1/B2; CPU 0008A/G1; Config 0006;
  Engine 0018. Tasks 0030/0031 provide evidence, not executable dependencies.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span>
- Conflicts with: every concurrent Metal capability/schema/native/candidate/materialization/public
  Engine scope; convolution/pooling/window/custom-kernel work; any 0016–0018 restart.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [materialization glossary](../../../../glossary.md#materialization); [scope glossary](../../../../glossary.md#term-scope); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [kernel glossary](../../../../glossary.md#kernel))</span>
- Parallel group: None.
- Common base: `e3de755abfc0a15fc54c7bebe88a84de5df4511c`.
- Integration order: documentation-only review while Blocked; no implementation or [device](../../../../glossary.md#term-device) gate.

## Exact [Model](../../../../glossary.md#term-model) and [Compiler](../../../../glossary.md#term-compiler) boundary

[`CONV3D`](../../../../glossary.md#conv3d--ncdhw) is `Conv3dAttrs 2..3 -> 1`: [NCDHW](../../../../glossary.md#conv3d--ncdhw) `[N,Cin,D,H,W]`, OIDHW
`[Cout,Cin/groups,Kd,Kh,Kw]`, optional `[Cout]` [bias](../../../../glossary.md#term-bias), and NCDHW output. Its ten ordered attributes
are [stride](../../../../glossary.md#term-stride) D/H/W, [padding](../../../../glossary.md#padding) D/H/W, [dilation](../../../../glossary.md#term-dilation) D/H/W, and groups. Each output [axis](../../../../glossary.md#term-axis) uses:

```text
effectiveKernel = dilation * (kernel - 1) + 1
output          = floor((input + 2 * padding - effectiveKernel) / stride) + 1
```

It is grouped cross-correlation. Conceptual positive-zero padding participates in multiplication,
including with [infinity](../../../../glossary.md#term-infinity). [FLOAT32](../../../../glossary.md#term-float32) accumulates in FLOAT32. [Reassociation](../../../../glossary.md#term-reassociation)/corresponding [FMA](../../../../glossary.md#term-fused-multiply-add-fma) and
[ACCELERATOR](../../../../glossary.md#term-accelerator)-only Conv3d FLOAT32 [DAZ](../../../../glossary.md#term-daz)/[FTZ](../../../../glossary.md#term-ftz) are allowed; [MATMUL](../../../../glossary.md#matrix-multiplication--matmul)'s final-zero choice, generic ADD zero
freedom, reduced [precision](../../../../glossary.md#term-precision), arbitrary transformed arithmetic, and [tolerance](../../../../glossary.md#term-tolerance) results are not.

Compiler accepts Conv3d forward and canonicalizes eligible static results, but rejects original
Conv3d in backward-capable modes before [derivative](../../../../glossary.md#term-derivative) allocation. [Pool3d](../../../../glossary.md#pool3d--ncdhw-pooling) and all three
`UNFOLD3D`/`FOLD3D` [signatures](../../../../glossary.md#term-signature) have exact Compiler [gradients](../../../../glossary.md#term-gradient), but current [Metal](../../../../glossary.md#term-metal) cannot execute the
complete generated [topology](../../../../glossary.md#term-topology). The initial [backend](../../../../glossary.md#backend) candidate remains no-grad.

Both Pool3d kinds are `1 -> 1` NCDHW operations with thirteen fields: [kernel](../../../../glossary.md#kernel), stride, padding, and
dilation D/H/W plus `ceilMode`. Literal ceil retains terminal all-padding windows. Max excludes
padding, returns exact [negative infinity](../../../../glossary.md#nan-infinity-and-signed-zero) when all-padding, makes [NaN](../../../../glossary.md#term-nan) win, orders `+0 > -0`, keeps
the first equal D/H/W candidate, and preserves selected non-NaN bits. Average uses fixed
`Kd*Kh*Kw`, counts positive-zero padding, accumulates/divides FLOAT32 for FLOAT32/[BFLOAT16](../../../../glossary.md#term-bfloat16) results,
narrows BFLOAT16 once, and fixes exceptional and zero classes. [Pooling](../../../../glossary.md#term-pooling) is unrelaxed in both profiles.

## Direct [Conv3d](../../../../glossary.md#conv3d--ncdhw) mapping and inherited blocker

Installed `MPSGraphConvolutionOps.h` declares [rank](../../../../glossary.md#term-rank)-five
`convolution3DWithSourceTensor:weightsTensor:descriptor:name:` on macOS 13.2+:

```text
strideInX/Y/Z       = strideWidth/Height/Depth
dilationRateInX/Y/Z = dilationWidth/Height/Depth
paddingLeft/right   = paddingWidth
paddingTop/bottom   = paddingHeight
paddingFront/back   = paddingDepth
paddingStyle        = Explicit
dataLayout          = NCDHW
weightsLayout       = OIDHW
groups              = groups
```

This maps the full candidate structurally without a probe. The selector has no [bias](../../../../glossary.md#term-bias) input; a
reshape/broadcast add needs separate [convolution](../../../../glossary.md#term-channels-first-convolution-layers)-level proof and is excluded. The macOS 26.6.2 [host](../../../../glossary.md#term-host)
contains the [API](../../../../glossary.md#term-application-programming-interface-api), but the native build sets no minimum [target](../../../../glossary.md#term-target); implementation must establish a
macOS 13.2 floor or explicit [fail-closed](../../../../glossary.md#term-fail-closed) availability.

The header does not specify [contraction](../../../../glossary.md#contraction-dimension) algorithm, [FLOAT32](../../../../glossary.md#term-float32) accumulation/[rounding](../../../../glossary.md#term-rounding), [shape](../../../../glossary.md#shape)-dependent
algorithm selection, or a pinning control. `MPSGraphReducedPrecisionFastMathNone` disables one
reduced-[precision](../../../../glossary.md#term-precision) option; it does not force a declared-product contraction.

This is Task 0031's blocker at larger rank. One execution exercises one geometry and [opaque](../../../../glossary.md#term-opaque)
realization; it cannot authorize other [kernels](../../../../glossary.md#kernel), groups, channels, [strides](../../../../glossary.md#term-stride), dilations, paddings, or
extents. Transformed convolution may use arithmetic outside [reassociation](../../../../glossary.md#term-reassociation)/corresponding [FMA](../../../../glossary.md#term-fused-multiply-add-fma). [Lean](../../../../glossary.md#proof-assistant-and-model-checking-tools)
policy prohibits the required geometry/algorithm [matrix](../../../../glossary.md#term-matrix-meanings), and one passing geometry cannot justify a
capability cut. No Conv3d probe ran.

## [Pool3d](../../../../glossary.md#pool3d--ncdhw-pooling) surfaces and exact-semantic gaps

The [SDK](../../../../glossary.md#term-sdk) has no `MPSGraphPooling3DOpDescriptor` or named Pool3d selector. Pooling4D on macOS 12+
requires four [spatial](../../../../glossary.md#term-spatial-axes) [kernel](../../../../glossary.md#kernel)/[stride](../../../../glossary.md#term-stride)/[dilation](../../../../glossary.md#term-dilation) and eight [padding](../../../../glossary.md#padding) values. The only plausible [NCDHW](../../../../glossary.md#conv3d--ncdhw)
proposal treats `[C,D,H,W]` as four spatial [dimensions](../../../../glossary.md#dimension):

```text
kernelSizes    = [1, Kd, Kh, Kw]
strides        = [1, sD, sH, sW]
dilationRates  = [1, dD, dH, dW]
paddingValues  = [0,0, pD,pD, pH,pH, pW,pW]
```

Its header does not state the source [rank](../../../../glossary.md#term-rank)-to-[axis](../../../../glossary.md#term-axis) mapping, named NCDHW [layout](../../../../glossary.md#layout), [singleton](../../../../glossary.md#term-singleton)-[channel](../../../../glossary.md#term-channel)
preservation, or complete output formula. `ceilMode` says division rounds up, not that [Model](../../../../glossary.md#term-model)'s
terminal all-padding window remains. The proposal is [inference](../../../../glossary.md#term-inference), not authorization.

Stencil explicitly reduces the last four source dimensions with rank-four [weights](../../../../glossary.md#term-weight). NCDHW could use
positive-one `[1,Kd,Kh,Kw]` weights, unit/unpadded channel geometry, and Model D/H/W geometry; max
could propose negative-[infinity](../../../../glossary.md#term-infinity) boundary padding, and average could propose positive-zero padding,
SUM, then fixed-volume division. It remains insufficient:

- stencil has no `ceilMode`;
- max does not document [NaN](../../../../glossary.md#term-nan) dominance, `+0 > -0`, first winner, selected bits, or all-padding;
- SUM/division does not document required [accumulator](../../../../glossary.md#accumulator-type)/division, [narrowing](../../../../glossary.md#term-narrowing), [infinities](../../../../glossary.md#nan-infinity-and-signed-zero), NaN, or zero;
- synthetic weights/composed arithmetic cannot import another [operation](../../../../glossary.md#operation)'s semantic freedoms.

[Pooling](../../../../glossary.md#term-pooling) is unrelaxed. Task 0030's sole 2D max-pool gate returned `-0` for `[+0,-0]` and [finite](../../../../glossary.md#term-finite)
`3.0f` for every first/later qNaN/sNaN window. That does not prove Pooling4D/stencil identical, but
it proves a generic [MPSGraph](../../../../glossary.md#term-mpsgraph) max-pool assumption unsound. Average has independent divisor, padding,
ceil-tail, accumulator, exceptional-value, and zero-sign gaps. Documentation already blocks both;
no redundant Pool3d probe ran.

## Separate 3D-window work

`UNFOLD3D`/`FOLD3D` are first-class operations used by [Compiler](../../../../glossary.md#term-compiler) formulas, not hidden [Pool3d](../../../../glossary.md#pool3d--ncdhw-pooling)/[Conv3d](../../../../glossary.md#conv3d--ncdhw)
lowerings. Installed im-to-column/column-to-image are [rank](../../../../glossary.md#term-rank)-four 2D surfaces. Exact custom 3D
movement or overlap-add needs separate mapping, [padding](../../../../glossary.md#padding), ceil-tail, contribution-order, workspace,
overlap, [binding](../../../../glossary.md#term-binding), and [lifecycle](../../../../glossary.md#lifecycle) proof. It is not a Task 0036 substitute.

## Conditional [schema](../../../../glossary.md#term-schema) only

Task 0036 reserves nothing. If direct [Conv3d](../../../../glossary.md#conv3d--ncdhw) were authorized and landed first, conditionally advance
schema `11 -> 12`, append `CONV3D=20` and `AttributeKind.CONV3D=7`, and advance workload,
exact-policy, candidate, compatibility, route-policy, and [codec](../../../../glossary.md#term-codec) [identities](../../../../glossary.md#term-identity-meanings) `12 -> 13`. A one-output
[MPSGraph](../../../../glossary.md#term-mpsgraph) route can retain wires `1..19`, [ABI](../../../../glossary.md#term-application-binary-interface-abi) v4, 160 bytes, and thirteen exports.

Encode two inputs/one output, `attributeCount=10`, absent [axis](../../../../glossary.md#term-axis), `auxiliary=0`, and ordered
`[strideD,strideH,strideW,paddingD,paddingH,paddingW,dilationD,dilationH,dilationW,groups]`; cells
`10..15` are zero and [weight](../../../../glossary.md#term-weight) [Shape](../../../../glossary.md#shape) supplies [kernels](../../../../glossary.md#kernel). Java/native independently validate profile,
type, no-grad state, [topology](../../../../glossary.md#term-topology), Shapes, geometry, conversions, and sentinels.

Alternatively, one authorized [Pool3d](../../../../glossary.md#pool3d--ncdhw-pooling) kind could conditionally use [operation](../../../../glossary.md#operation) `20`,
`AttributeKind.POOL3D=7`, one input/output, sentinels, `attributeCount=12`, ordered geometry, and
`auxiliary=0/1` for ceil; both kinds could use `20/21`. An [atomic](../../../../glossary.md#resource-transaction-and-rollback) Conv3d/both-pools design could use
operations `20/21/22` and attributes `CONV3D=7`, `POOL3D=8`. These examples are mutually
conditional. Tasks 0030–0035 reserve none of the same next values; the first implementation owns
identities and every other plan rebases. Custom kernels must separately define record/ABI/exports.

## Unblock, [validation](../../../../glossary.md#term-validation), and result

Unblocking requires an exact custom [kernel](../../../../glossary.md#kernel) over the complete admitted domain, authoritative complete
[operation](../../../../glossary.md#operation)-scoped documentation, or preceding [Model](../../../../glossary.md#term-model)/architecture broadening. [Pool3d](../../../../glossary.md#pool3d--ncdhw-pooling) also needs exact
[NCDHW](../../../../glossary.md#conv3d--ncdhw) mapping and max/average proof. Do not retry Task 0030 through Pooling4D, [run](../../../../glossary.md#term-run) one [Conv3d](../../../../glossary.md#conv3d--ncdhw) [shape](../../../../glossary.md#shape),
infer contracts from [API](../../../../glossary.md#term-application-programming-interface-api) names, narrow to passing geometry, or weaken an [oracle](../../../../glossary.md#term-test-oracle).

Validate only Markdown links/anchors/fences/newlines, whitespace, [planning](../../../../glossary.md#term-planning)-only [scope](../../../../glossary.md#term-scope), and final
diff. Run no [device](../../../../glossary.md#term-device) probe, test, formatter, build, [schema](../../../../glossary.md#term-schema) reservation, or production edit.

Planning from clean `e3de755abfc0a15fc54c7bebe88a84de5df4511c` found one structurally exact candidate
but no authorized [MPSGraph](../../../../glossary.md#term-mpsgraph) realization. Conv3d inherits Task 0031's [opaque](../../../../glossary.md#term-opaque) [contraction](../../../../glossary.md#contraction-dimension) blocker;
Pooling4D/stencil do not establish exact Pool3d structure/semantics, and Task 0030 forbids generic
max assumptions. UNFOLD3D/FOLD3D remain separate custom work. Task 0036 is Blocked; no [Metal](../../../../glossary.md#term-metal) task
is [Ready](../../../../glossary.md#term-ready).
