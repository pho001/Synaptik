# Task 0036: Extended 3D Inference

## Status

Blocked

## Change class

Class C if unblocked — the smallest candidate would add accelerator `CONV3D`, evolve private
schema/route identities, and change native graphs. Pool3d or custom 3D-window execution would add
separate numerical or kernel responsibilities. This planning-only blocker changes no executable
behavior and ran no device probe.

## Goal

Assess the smallest first-class Metal 3D inference slice: unbiased, no-grad, canonical, positive,
fully static rank-five `FLOAT32` `CONV3D` under `ACCELERATOR`. Also close the adjacent
`MAX_POOL3D`/`AVERAGE_POOL3D` selector question without treating a four-spatial-dimensional API or
generic stencil as an undocumented three-dimensional pooling contract.

Direct MPSGraph Conv3d maps the complete structural candidate but inherits Task 0031's opaque
shape-dependent contraction blocker. Pooling has no named 3D selector, and Pooling4D/stencil do not
establish exact profile-common Pool3d semantics. Lean policy blocks both before a probe. Task 0036
is Blocked and exposes no Ready Metal task.

## Smallest candidate if unblocked

- Admit exactly `Conv3dKind.CONV3D + Conv3dAttrs`, ordered `[input, weight] -> [output]`.
- Require `ACCELERATOR`; `STRICT_IEEE` remains unsupported.
- Require canonical dense contiguous, zero-offset, non-view `FLOAT32` rank-five NCDHW input,
  OIDHW weight, and NCDHW output with fully static positive extents and `requiresGrad=false`.
- Independently validate grouped channels and derive checked `[N,Cout,Dout,Hout,Wout]` from
  positive stride/dilation/groups, non-negative symmetric padding, and positive weight kernels.
- Admit every documented Model-valid geometry fitting checked Java/native conversions and buffers;
  add no limit inferred from execution.
- Retain the whole-partition MPSGraph lifecycle, FLOAT32 transfer, and profile-free Runtime/Trace.

## Non-goals

- Biased `CONV3D`, fused epilogue, hidden add, gradients/training, saved values, or hidden outputs.
- Claiming Pool3d from a plausible but undocumented Pooling4D/stencil composition.
- `UNFOLD3D`/`FOLD3D`, overlap-add, generic windows, or convolution decomposition.
- Dynamic/zero extents, views, another layout/type/profile, transfer widening, fallback, custom
  kernels, performance claims, a device matrix, single-shape narrowing, or relaxed oracles.

## Contracts, dependencies, and integration

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence) keeps
  Model as semantic owner and requires fail-closed backend support.
- [Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)
  give `CONV3D` only family reassociation/corresponding-FMA freedom plus row-scoped FLOAT32 DAZ/FTZ;
  pooling and window movement receive no accelerator relaxation.
- [Metal](../../../../architecture/contracts/backend-execution.md#metal-backend) owns lowering and
  execution, not meaning; [Prepare](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)
  keeps route selection cold.
- Model [Conv3d](../../../modules/model/tasks/0025h-ncdhw-conv3d-semantics-and-tensor-expressions.md),
  [Pool3d](../../../modules/model/tasks/0025j-first-class-ncdhw-max-average-pool3d-semantics.md),
  and [3D windows](../../../modules/model/tasks/0025k-public-ncdhw-unfold3d-and-fold3d-window-transforms.md)
  own signatures, geometry, attributes, and numerical meaning.
- Compiler [Conv3d](../../../modules/compiler/tasks/0006b-conv3d-forward-adoption-and-explicit-gradient-boundary.md),
  [Pool3d/windows](../../../modules/compiler/tasks/0006b1-pool3d-and-3d-window-forward-adoption-and-explicit-gradient-boundary.md),
  and [gradient closure](../../../modules/compiler/tasks/0006b2-pool3d-and-3d-window-gradient-closure.md)
  own inference, canonical publication, and generated gradients.
- CPU [Conv3d](../../cpu/tasks/0008a-portable-channels-first-dimensional-convolution-closure.md)
  and [Pool3d](../../cpu/tasks/0008g1-portable-pool1d-composition-validation-and-pool3d-generated-execution.md)
  are exact rank-specific private precedents, not MPSGraph authorization.
- Depends on: Metal 0025; Model 0025H/J/K; Compiler 0006B/B1/B2; CPU 0008A/G1; Config 0006;
  Engine 0018. Tasks 0030/0031 provide evidence, not executable dependencies.
- Conflicts with: every concurrent Metal capability/schema/native/candidate/materialization/public
  Engine scope; convolution/pooling/window/custom-kernel work; any 0016–0018 restart.
- Parallel group: None.
- Common base: `e3de755abfc0a15fc54c7bebe88a84de5df4511c`.
- Integration order: documentation-only review while Blocked; no implementation or device gate.

## Exact Model and Compiler boundary

`CONV3D` is `Conv3dAttrs 2..3 -> 1`: NCDHW `[N,Cin,D,H,W]`, OIDHW
`[Cout,Cin/groups,Kd,Kh,Kw]`, optional `[Cout]` bias, and NCDHW output. Its ten ordered attributes
are stride D/H/W, padding D/H/W, dilation D/H/W, and groups. Each output axis uses:

```text
effectiveKernel = dilation * (kernel - 1) + 1
output          = floor((input + 2 * padding - effectiveKernel) / stride) + 1
```

It is grouped cross-correlation. Conceptual positive-zero padding participates in multiplication,
including with infinity. FLOAT32 accumulates in FLOAT32. Reassociation/corresponding FMA and
ACCELERATOR-only Conv3d FLOAT32 DAZ/FTZ are allowed; MATMUL's final-zero choice, generic ADD zero
freedom, reduced precision, arbitrary transformed arithmetic, and tolerance results are not.

Compiler accepts Conv3d forward and canonicalizes eligible static results, but rejects original
Conv3d in backward-capable modes before derivative allocation. Pool3d and all three
`UNFOLD3D`/`FOLD3D` signatures have exact Compiler gradients, but current Metal cannot execute the
complete generated topology. The initial backend candidate remains no-grad.

Both Pool3d kinds are `1 -> 1` NCDHW operations with thirteen fields: kernel, stride, padding, and
dilation D/H/W plus `ceilMode`. Literal ceil retains terminal all-padding windows. Max excludes
padding, returns exact negative infinity when all-padding, makes NaN win, orders `+0 > -0`, keeps
the first equal D/H/W candidate, and preserves selected non-NaN bits. Average uses fixed
`Kd*Kh*Kw`, counts positive-zero padding, accumulates/divides FLOAT32 for FLOAT32/BFLOAT16 results,
narrows BFLOAT16 once, and fixes exceptional and zero classes. Pooling is unrelaxed in both profiles.

## Direct Conv3d mapping and inherited blocker

Installed `MPSGraphConvolutionOps.h` declares rank-five
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

This maps the full candidate structurally without a probe. The selector has no bias input; a
reshape/broadcast add needs separate convolution-level proof and is excluded. The macOS 26.6.2 host
contains the API, but the native build sets no minimum target; implementation must establish a
macOS 13.2 floor or explicit fail-closed availability.

The header does not specify contraction algorithm, FLOAT32 accumulation/rounding, shape-dependent
algorithm selection, or a pinning control. `MPSGraphReducedPrecisionFastMathNone` disables one
reduced-precision option; it does not force a declared-product contraction.

This is Task 0031's blocker at larger rank. One execution exercises one geometry and opaque
realization; it cannot authorize other kernels, groups, channels, strides, dilations, paddings, or
extents. Transformed convolution may use arithmetic outside reassociation/corresponding FMA. Lean
policy prohibits the required geometry/algorithm matrix, and one passing geometry cannot justify a
capability cut. No Conv3d probe ran.

## Pool3d surfaces and exact-semantic gaps

The SDK has no `MPSGraphPooling3DOpDescriptor` or named Pool3d selector. Pooling4D on macOS 12+
requires four spatial kernel/stride/dilation and eight padding values. The only plausible NCDHW
proposal treats `[C,D,H,W]` as four spatial dimensions:

```text
kernelSizes    = [1, Kd, Kh, Kw]
strides        = [1, sD, sH, sW]
dilationRates  = [1, dD, dH, dW]
paddingValues  = [0,0, pD,pD, pH,pH, pW,pW]
```

Its header does not state the source rank-to-axis mapping, named NCDHW layout, singleton-channel
preservation, or complete output formula. `ceilMode` says division rounds up, not that Model's
terminal all-padding window remains. The proposal is inference, not authorization.

Stencil explicitly reduces the last four source dimensions with rank-four weights. NCDHW could use
positive-one `[1,Kd,Kh,Kw]` weights, unit/unpadded channel geometry, and Model D/H/W geometry; max
could propose negative-infinity boundary padding, and average could propose positive-zero padding,
SUM, then fixed-volume division. It remains insufficient:

- stencil has no `ceilMode`;
- max does not document NaN dominance, `+0 > -0`, first winner, selected bits, or all-padding;
- SUM/division does not document required accumulator/division, narrowing, infinities, NaN, or zero;
- synthetic weights/composed arithmetic cannot import another operation's semantic freedoms.

Pooling is unrelaxed. Task 0030's sole 2D max-pool gate returned `-0` for `[+0,-0]` and finite
`3.0f` for every first/later qNaN/sNaN window. That does not prove Pooling4D/stencil identical, but
it proves a generic MPSGraph max-pool assumption unsound. Average has independent divisor, padding,
ceil-tail, accumulator, exceptional-value, and zero-sign gaps. Documentation already blocks both;
no redundant Pool3d probe ran.

## Separate 3D-window work

`UNFOLD3D`/`FOLD3D` are first-class operations used by Compiler formulas, not hidden Pool3d/Conv3d
lowerings. Installed im-to-column/column-to-image are rank-four 2D surfaces. Exact custom 3D
movement or overlap-add needs separate mapping, padding, ceil-tail, contribution-order, workspace,
overlap, binding, and lifecycle proof. It is not a Task 0036 substitute.

## Conditional schema only

Task 0036 reserves nothing. If direct Conv3d were authorized and landed first, conditionally advance
schema `11 -> 12`, append `CONV3D=20` and `AttributeKind.CONV3D=7`, and advance workload,
exact-policy, candidate, compatibility, route-policy, and codec identities `12 -> 13`. A one-output
MPSGraph route can retain wires `1..19`, ABI v4, 160 bytes, and thirteen exports.

Encode two inputs/one output, `attributeCount=10`, absent axis, `auxiliary=0`, and ordered
`[strideD,strideH,strideW,paddingD,paddingH,paddingW,dilationD,dilationH,dilationW,groups]`; cells
`10..15` are zero and weight Shape supplies kernels. Java/native independently validate profile,
type, no-grad state, topology, Shapes, geometry, conversions, and sentinels.

Alternatively, one authorized Pool3d kind could conditionally use operation `20`,
`AttributeKind.POOL3D=7`, one input/output, sentinels, `attributeCount=12`, ordered geometry, and
`auxiliary=0/1` for ceil; both kinds could use `20/21`. An atomic Conv3d/both-pools design could use
operations `20/21/22` and attributes `CONV3D=7`, `POOL3D=8`. These examples are mutually
conditional. Tasks 0030–0035 reserve none of the same next values; the first implementation owns
identities and every other plan rebases. Custom kernels must separately define record/ABI/exports.

## Unblock, validation, and result

Unblocking requires an exact custom kernel over the complete admitted domain, authoritative complete
operation-scoped documentation, or preceding Model/architecture broadening. Pool3d also needs exact
NCDHW mapping and max/average proof. Do not retry Task 0030 through Pooling4D, run one Conv3d shape,
infer contracts from API names, narrow to passing geometry, or weaken an oracle.

Validate only Markdown links/anchors/fences/newlines, whitespace, planning-only scope, and final
diff. Run no device probe, test, formatter, build, schema reservation, or production edit.

Planning from clean `e3de755abfc0a15fc54c7bebe88a84de5df4511c` found one structurally exact candidate
but no authorized MPSGraph realization. Conv3d inherits Task 0031's opaque contraction blocker;
Pooling4D/stencil do not establish exact Pool3d structure/semantics, and Task 0030 forbids generic
max assumptions. UNFOLD3D/FOLD3D remain separate custom work. Task 0036 is Blocked; no Metal task
is Ready.
