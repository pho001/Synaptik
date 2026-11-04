# Task 0031: [ACCELERATOR](../../../../glossary.md#term-accelerator) [Canonical](../../../../glossary.md#term-canonical-form) [FLOAT32](../../../../glossary.md#term-float32) Unbiased [CONV2D](../../../../glossary.md#convolution--conv2d) Forward

## Reader context

“Unbiased” means this candidate had input and weight roles but no bias operand. The direct
MPSGraph descriptor could express the grouped convolution geometry; that structural match did
not establish which arithmetic algorithm every compiled shape would use. No device probe ran.

Model defined operation-specific product, accumulation and padding behavior. Permissions attached
to another operation could not be imported into convolution. Shapes and layouts described the
candidate, but actual buffers still included exceptional values that its whole-domain proof
would need to handle. Conditional wire and schema examples below were neither implemented nor
reserved.
The [current numerical convolution and geometry boundary](../../../../backend-guide/metal-backend.md#ordinary-numerical-and-exact-domains)
explains the later routes; it is not retrospective evidence for this blocked direct mapping.

## Status

Blocked

## Change class

Class C if unblocked — the candidate would add a numerical [operation](../../../../glossary.md#operation) to the [accelerator](../../../../glossary.md#term-accelerator) [Metal](../../../../glossary.md#term-metal)
[matrix](../../../../glossary.md#term-matrix-meanings), evolve the [backend](../../../../glossary.md#backend)-private native [schema](../../../../glossary.md#term-schema) and route [identities](../../../../glossary.md#term-identity-meanings), and change generated native
[graphs](../../../../glossary.md#graph). This [planning](../../../../glossary.md#term-planning)-only blocker record changes no executable behavior and ran no [device](../../../../glossary.md#term-device) probe.

## Goal

Assess the smallest first-class [Metal](../../../../glossary.md#term-metal) [`CONV2D`](../../../../glossary.md#convolution--conv2d) forward [slice](../../../../glossary.md#slice): unbiased, no-grad, [canonical](../../../../glossary.md#term-canonical-form),
positive, fully static [rank](../../../../glossary.md#term-rank)-four [`FLOAT32`](../../../../glossary.md#term-float32) under [`ACCELERATOR`](../../../../glossary.md#term-accelerator). Preserve the complete documented
[Model](../../../../glossary.md#term-model) geometry and exact [operation](../../../../glossary.md#operation)-local allowed-result set without a probe-derived geometry limit.

The direct [MPSGraph](../../../../glossary.md#term-mpsgraph) selector maps the complete structural domain, but its [shape](../../../../glossary.md#shape)-dependent
[contraction](../../../../glossary.md#contraction-dimension) realization is [opaque](../../../../glossary.md#term-opaque) and undocumented. Under the mandatory [lean](../../../../glossary.md#proof-assistant-and-model-checking-tools) policy, one allowed-
set execution cannot authorize the full domain and a geometry/algorithm [matrix](../../../../glossary.md#term-matrix-meanings) is prohibited.
Therefore this task is Blocked before any probe or production edit.

## Candidate domain if unblocked

- Admit exactly `Conv2dKind.CONV2D + Conv2dAttrs` with ordered `[input, weight]` and one output.
- Require `NumericalProfile.ACCELERATOR`; `STRICT_IEEE` remains unsupported.
- Require [canonical](../../../../glossary.md#term-canonical-form) dense [contiguous](../../../../glossary.md#term-contiguous), zero-offset, non-[view](../../../../glossary.md#term-view-meanings) [`FLOAT32`](../../../../glossary.md#term-float32) rank-four input, OIHW weight,
  and NCHW output descriptors with fully static positive extents and <span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [weight glossary](../../../../glossary.md#term-weight); [NCHW glossary](../../../../glossary.md#term-nchw); [descriptors glossary](../../../../glossary.md#term-descriptor))</span>`requiresGrad=false`.
- Require exact grouped-[channel](../../../../glossary.md#term-channel) relations and independently derive checked
  `[N,Cout,Hout,Wout]` from positive stride/dilation/groups, non-negative symmetric padding, and
  positive static weight kernel extents.<span data-glossary-navigation="true"> ([stride glossary](../../../../glossary.md#term-stride); [dilation glossary](../../../../glossary.md#term-dilation); [padding glossary](../../../../glossary.md#padding); [kernel glossary](../../../../glossary.md#kernel))</span>
- Admit every documented Model-valid geometry that fits checked Java/native arithmetic, native
  integer conversions, and buffer extents. Add no kernel, channel, group, stride, dilation, or
  padding limit inferred from one device execution.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [buffer glossary](../../../../glossary.md#term-buffer); [device glossary](../../../../glossary.md#term-device))</span>
- Retain the current whole-partition MPSGraph lifecycle, ABI v4, 160-byte node record, thirteen
  exports, FLOAT32-only transfer, and profile-free Runtime/Trace boundary.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [lifecycle glossary](../../../../glossary.md#lifecycle); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [node glossary](../../../../glossary.md#node); [Runtime glossary](../../../../glossary.md#term-runtime); [Trace glossary](../../../../glossary.md#trace))</span>

## Non-goals

- Three-input biased [`CONV2D`](../../../../glossary.md#convolution--conv2d), a nullable [bias](../../../../glossary.md#term-bias), [fused](../../../../glossary.md#specialization-and-fusion) [epilogue](../../../../glossary.md#term-epilogue), or generic [scalar](../../../../glossary.md#term-scalar) arithmetic.
- `requiresGrad=true`, direct MPSGraph gradient selectors, generated backward ownership, training,
  saved values, or hidden outputs.<span data-glossary-navigation="true"> ([MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [gradient glossary](../../../../glossary.md#term-gradient); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Conv1d/[Conv3d](../../../../glossary.md#conv3d--ncdhw), max/average Pool1d/2d/3d, adaptive/global [pooling](../../../../glossary.md#term-pooling), or [convolution](../../../../glossary.md#term-channels-first-convolution-layers) [decomposition](../../../../glossary.md#term-decomposition).
- Dynamic or zero extents, views/affine inputs, another layout/type/profile, transfer widening,
  fallback, host execution, custom-kernel implementation, or performance claims.<span data-glossary-navigation="true"> ([views glossary](../../../../glossary.md#term-view-meanings); [layout glossary](../../../../glossary.md#layout); [widening glossary](../../../../glossary.md#term-widening); [host glossary](../../../../glossary.md#term-host); [kernel glossary](../../../../glossary.md#kernel))</span>
- A shape, group, kernel, stride, dilation, padding, profile, option, context, optimization, or
  repetition device matrix; a single-geometry capability cut; or a relaxed oracle.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [stride glossary](../../../../glossary.md#term-stride); [dilation glossary](../../../../glossary.md#term-dilation); [padding glossary](../../../../glossary.md#padding); [device glossary](../../../../glossary.md#term-device); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [oracle glossary](../../../../glossary.md#term-test-oracle))</span>

## Contracts

- [`ARCHITECTURE.md` — Authority, incorporation, and precedence](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence)
  requires plans to preserve the root and incorporated scoped contracts.
- [Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)
  gives [`CONV2D`](../../../../glossary.md#convolution--conv2d) only its existing [reassociation](../../../../glossary.md#term-reassociation)/corresponding-[FMA](../../../../glossary.md#term-fused-multiply-add-fma) freedom plus row-scoped [DAZ](../../../../glossary.md#term-daz)/[FTZ](../../../../glossary.md#term-ftz).
- [Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [backend glossary](../../../../glossary.md#backend))</span> owns
  lowering, native integration, storage, and materialization, not Model semantics.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [materialization glossary](../../../../glossary.md#materialization); [Model glossary](../../../../glossary.md#term-model))</span>
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
  keeps route construction cold and Runtime free of policy lookup.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- [Model Conv2d](../../../modules/model/tasks/0020-nchw-conv2d-semantics-and-tensor-expressions.md)
  owns the signature, NCHW/OIHW grouped cross-correlation, Shape, accumulation, padding, and
  special-value contract.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [NCHW glossary](../../../../glossary.md#term-nchw); [Shape glossary](../../../../glossary.md#shape); [padding glossary](../../../../glossary.md#padding))</span>
- [Compiler convolution gradients](../../../modules/compiler/tasks/0005d-attention-convolution-pooling-and-loss-gradient-completion.md)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [gradients glossary](../../../../glossary.md#term-gradient))</span>
  owns backward formulas independently of backend execution.

## Exact [Model](../../../../glossary.md#term-model) and [Compiler](../../../../glossary.md#term-compiler) boundary

[`CONV2D`](../../../../glossary.md#convolution--conv2d) has one [signature](../../../../glossary.md#term-signature), `Conv2dAttrs 2..3 -> 1`. [Bias](../../../../glossary.md#term-bias) presence is represented only by the
[occurrence](../../../../glossary.md#term-occurrence) input count; there is no bias flag or second [operation](../../../../glossary.md#operation) kind. `Conv2dAttrs` contains, in
order, [stride](../../../../glossary.md#term-stride) height/width, [padding](../../../../glossary.md#padding) height/width, [dilation](../../../../glossary.md#term-dilation) height/width, and groups. The public
unbiased `conv2d(weight, attrs)` expression is first-class.

Input/output use [NCHW](../../../../glossary.md#term-nchw) and [weight](../../../../glossary.md#term-weight) uses `[Cout,Cin/groups,Kh,Kw]` OIHW. For each [spatial axis](../../../../glossary.md#spatial-axes-channels-and-window-sampling):

```text
effectiveKernel = dilation * (kernel - 1) + 1
output          = floor((input + 2 * padding - effectiveKernel) / stride) + 1
```

The output uses promoted floating type, [Shape](../../../../glossary.md#shape) `[N,Cout,Hout,Wout]`, unresolved Model [layout](../../../../glossary.md#layout) later
closed by Compiler for eligible fully static results, and `requiresGrad = input || weight || bias`.
The two-input no-grad candidate therefore requires all three [descriptors](../../../../glossary.md#term-descriptor) to be no-grad.

Each output is the sum of every declared group-local input/weight product. Conceptual padding is
[positive zero](../../../../glossary.md#nan-infinity-and-signed-zero) and participates in ordinary multiplication, including with [infinity](../../../../glossary.md#term-infinity). [FLOAT32](../../../../glossary.md#term-float32)
accumulates in FLOAT32. [Reassociation](../../../../glossary.md#term-reassociation) and a corresponding multiply/add [FMA](../../../../glossary.md#term-fused-multiply-add-fma) are permitted;
[`ACCELERATOR`](../../../../glossary.md#term-accelerator) additionally permits only CONV2D-scoped [DAZ](../../../../glossary.md#term-daz)/[FTZ](../../../../glossary.md#term-ftz). It does not inherit [MATMUL](../../../../glossary.md#matrix-multiplication--matmul)'s final
exact-zero [publication](../../../../glossary.md#term-publication) choice or [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) ADD's exact-zero sign freedom. Term [loss](../../../../glossary.md#loss), arbitrary reduced
[precision](../../../../glossary.md#term-precision), algebraic replacement, cross-[node](../../../../glossary.md#node) [contraction](../../../../glossary.md#contraction-dimension), and [tolerance](../../../../glossary.md#term-tolerance) acceptance remain forbidden.

Compiler already reconstructs selected input/weight cotangents through `UNFOLD2D`, `FOLD2D`,
grouped reshape/permute, higher-[rank](../../../../glossary.md#term-rank) `MATMUL`, and reduction. That [graph](../../../../glossary.md#graph) is outside current [Metal](../../../../glossary.md#term-metal)
capability. Task 0031 must not call [MPSGraph](../../../../glossary.md#term-mpsgraph) [convolution](../../../../glossary.md#term-channels-first-convolution-layers)-[gradient](../../../../glossary.md#term-gradient) selectors or imply backward
closure; `requiresGrad=true` remains rejected.

## Documented direct [MPSGraph](../../../../glossary.md#term-mpsgraph) mapping

`convolution2DWithSourceTensor:weightsTensor:descriptor:name:` consumes exactly source and [weights](../../../../glossary.md#term-weight).
`MPSGraphConvolution2DOpDescriptor` documents every candidate attribute:

```text
strideInX       = strideWidth       strideInY       = strideHeight
dilationRateInX = dilationWidth     dilationRateInY = dilationHeight
paddingLeft/right = paddingWidth    paddingTop/bottom = paddingHeight
paddingStyle = Explicit             dataLayout = NCHW
weightsLayout = OIHW                groups = groups
```

The [descriptor](../../../../glossary.md#term-descriptor) therefore structurally represents the full [Model](../../../../glossary.md#term-model) geometry without a geometry probe.
The selector has no [bias](../../../../glossary.md#term-bias) input. A future biased route could structurally reshape `[Cout]` to
`[1,Cout,1,1]` and use documented broadcast addition, but that internal add cannot borrow generic
[Metal](../../../../glossary.md#term-metal) ADD semantics: [accelerator](../../../../glossary.md#term-accelerator) ADD has exact-zero sign freedom that [CONV2D](../../../../glossary.md#convolution--conv2d) explicitly lacks.
Bias needs its own [convolution](../../../../glossary.md#term-channels-first-convolution-layers)-level numerical proof and is excluded here.

## Documentation-first blocker

The [MPSGraph](../../../../glossary.md#term-mpsgraph) [convolution](../../../../glossary.md#term-channels-first-convolution-layers) header specifies [rank](../../../../glossary.md#term-rank), [layouts](../../../../glossary.md#layout), groups, [stride](../../../../glossary.md#term-stride), [dilation](../../../../glossary.md#term-dilation), explicit [padding](../../../../glossary.md#padding),
and the direct selector. It does not specify the [contraction](../../../../glossary.md#contraction-dimension) algorithm, accumulation expression,
[FLOAT32](../../../../glossary.md#term-float32) [rounding](../../../../glossary.md#term-rounding) sequence, algorithm-selection boundary, or a control that pins one realization.
`MPSGraphReducedPrecisionFastMathNone` can prohibit a reduced-[precision](../../../../glossary.md#term-precision) option; it does not document
or force a direct declared-product contraction.

A single process/context/[graph](../../../../glossary.md#graph)/executable/execution can exercise only one compiled geometry and its
selected [opaque](../../../../glossary.md#term-opaque) realization. Even a compact corpus containing group-isolation, all-term,
[reassociation](../../../../glossary.md#term-reassociation)/[FMA](../../../../glossary.md#term-fused-multiply-add-fma), [DAZ](../../../../glossary.md#term-daz)/[FTZ](../../../../glossary.md#term-ftz), [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero), [NaN](../../../../glossary.md#term-nan)/[infinity](../../../../glossary.md#term-infinity), and conceptual-padding witnesses cannot show
that other documented [kernels](../../../../glossary.md#kernel), groups, strides, dilations, paddings, channels, or extents use the
same realization or remain inside the [Model](../../../../glossary.md#term-model) set. [Shape](../../../../glossary.md#shape)-dependent alternatives such as transformed
convolution would require arithmetic not authorized merely as reassociation/corresponding FMA.

The [lean](../../../../glossary.md#proof-assistant-and-model-checking-tools) policy prohibits the geometry/algorithm [matrix](../../../../glossary.md#term-matrix-meanings) needed to characterize those regimes. A
one-geometry capability would be an unsound probe-derived [narrowing](../../../../glossary.md#term-narrowing) and is forbidden. No [device](../../../../glossary.md#term-device)
probe ran, and no [schema](../../../../glossary.md#term-schema), [identity](../../../../glossary.md#term-identity-meanings), capability, [lowering](../../../../glossary.md#lowering), test, or source change exists.

## Why adjacent slices are not smaller sound routes

- `AVERAGE_POOL2D` is a different exact unrelaxed numerical family with divisor, padding, and ceil
  behavior; its direct selector does not prove convolution and needs its own semantic authorization.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers))</span>
- Max Pool1d is visible Pool2d composition and inherits blocked 0030 selection behavior. Average
  Pool1d inherits the unproved Average Pool2d route; rank edits do not supply missing numerical proof.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank))</span>
- Pool3d adds rank, geometry, schema, and selector surface while retaining the corresponding exact
  max/average questions. It is larger, not a fallback for blocked Pool2d.<span data-glossary-navigation="true"> ([Pool3d glossary](../../../../glossary.md#pool3d--ncdhw-pooling); [schema glossary](../../../../glossary.md#term-schema))</span>
- Conv1d is visible `EXPAND_DIMS -> CONV2D -> SQUEEZE` and still requires the same blocked direct
  Conv2d occurrence. Conv3d has a larger opaque contraction and no smaller proof boundary.<span data-glossary-navigation="true"> ([Conv2d glossary](../../../../glossary.md#convolution--conv2d); [occurrence glossary](../../../../glossary.md#term-occurrence); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [opaque glossary](../../../../glossary.md#term-opaque); [contraction glossary](../../../../glossary.md#contraction-dimension))</span>
- Convolution or pooling backward requires unsupported generated window/fold/contraction topology
  or separate gradient selectors, saved selection policy, and wider lifecycle proof. It cannot
  establish a forward route.<span data-glossary-navigation="true"> ([pooling glossary](../../../../glossary.md#term-pooling); [topology glossary](../../../../glossary.md#term-topology); [gradient glossary](../../../../glossary.md#term-gradient); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>

These families remain separately unauthorized; none reopens 0026, 0027, or 0030 or supplies a
sound [Ready](../../../../glossary.md#term-ready) [Metal](../../../../glossary.md#term-metal) task.

## Prospective [schema](../../../../glossary.md#term-schema) if unblocked

Because blocked 0030 implemented none of its planned numbers, the first implementation may advance
[node](../../../../glossary.md#node) schema `11 -> 12`, append `CONV2D=20`, add `AttributeKind.CONV2D=7`, and advance workload,
exact-policy, candidate, compatibility, route-policy, and [codec](../../../../glossary.md#term-codec) [identities](../../../../glossary.md#term-identity-meanings) together `12 -> 13`.
Retain wires `1..19`, [ABI](../../../../glossary.md#term-application-binary-interface-abi) v4, the 160-byte record, and thirteen exports.

Encode two inputs and one output, `attributeCount=7`, absent-[axis](../../../../glossary.md#term-axis) sentinel, `auxiliary=0`, and the
seven `Conv2dAttrs` values in declaration order; require cells `7..15` zero. [Kernel](../../../../glossary.md#kernel) extents derive
from [weight](../../../../glossary.md#term-weight) [Shape](../../../../glossary.md#shape). Fixed [NCHW](../../../../glossary.md#term-nchw)/OIHW/explicit-[padding](../../../../glossary.md#padding) policy and [bias](../../../../glossary.md#term-bias) absence need no [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) flag.
Java validates profile, no-grad [metadata](../../../../glossary.md#term-metadata), [descriptors](../../../../glossary.md#term-descriptor), [topology](../../../../glossary.md#term-topology), groups, geometry, and output Shape;
native independently validates schema, types, Shapes, conversions, and the same geometry before
[graph](../../../../glossary.md#graph) creation. `requiresGrad` does not cross ABI v4 and remains Java-[preflight](../../../../glossary.md#term-preflight)/workload [identity](../../../../glossary.md#term-identity-meanings).

## [Lean](../../../../glossary.md#proof-assistant-and-model-checking-tools) [validation](../../../../glossary.md#term-validation) if unblocked

Use focused capability/preparer positives and rejection boundaries; one exact [schema](../../../../glossary.md#term-schema)/raw-[ABI](../../../../glossary.md#term-application-binary-interface-abi) record
and malformed-field checks; version-thirteen [identity](../../../../glossary.md#term-identity-meanings)/[codec](../../../../glossary.md#term-codec) stale rejection; native build/export
audit; and one [CPU](../../../../glossary.md#term-cpu)-free public [Engine](../../../../glossary.md#term-engine) exact-integer unbiased no-grad [smoke](../../../../glossary.md#term-smoke-test). Do not add a [device](../../../../glossary.md#term-device)
geometry [matrix](../../../../glossary.md#term-matrix-meanings), duplicate the numerical corpus in permanent tests, or claim backward/[bias](../../../../glossary.md#term-bias) support.

## Unblocking and result

Task 0031 can unblock only through one of these separately authorized paths:

1. an exact custom Metal kernel whose declared-product evaluation is proved for the complete
   candidate geometry; or<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel); [evaluation glossary](../../../../glossary.md#term-evaluation))</span>
2. a preceding Model/architecture decision that explicitly broadens the convolution allowed-result
   envelope to cover the intended MPSGraph realization.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers))</span>

Do not retry with one [MPSGraph](../../../../glossary.md#term-mpsgraph) geometry, narrow capability to a passing [shape](../../../../glossary.md#shape), infer behavior from a
[pooling](../../../../glossary.md#term-pooling) or [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) route, or weaken the [oracle](../../../../glossary.md#term-test-oracle). Task 0031 is Blocked without a [device](../../../../glossary.md#term-device) probe or any
production/test change. No [Metal](../../../../glossary.md#term-metal) task is [Ready](../../../../glossary.md#term-ready).
