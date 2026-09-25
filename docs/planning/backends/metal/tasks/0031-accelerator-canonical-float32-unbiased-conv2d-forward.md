# Task 0031: ACCELERATOR Canonical FLOAT32 Unbiased CONV2D Forward

## Status

Blocked

## Change class

Class C if unblocked — the candidate would add a numerical operation to the accelerator Metal
matrix, evolve the backend-private native schema and route identities, and change generated native
graphs. This planning-only blocker record changes no executable behavior and ran no device probe.

## Goal

Assess the smallest first-class Metal `CONV2D` forward slice: unbiased, no-grad, canonical,
positive, fully static rank-four `FLOAT32` under `ACCELERATOR`. Preserve the complete documented
Model geometry and exact operation-local allowed-result set without a probe-derived geometry limit.

The direct MPSGraph selector maps the complete structural domain, but its shape-dependent
contraction realization is opaque and undocumented. Under the mandatory lean policy, one allowed-
set execution cannot authorize the full domain and a geometry/algorithm matrix is prohibited.
Therefore this task is Blocked before any probe or production edit.

## Candidate domain if unblocked

- Admit exactly `Conv2dKind.CONV2D + Conv2dAttrs` with ordered `[input, weight]` and one output.
- Require `NumericalProfile.ACCELERATOR`; `STRICT_IEEE` remains unsupported.
- Require canonical dense contiguous, zero-offset, non-view `FLOAT32` rank-four input, OIHW weight,
  and NCHW output descriptors with fully static positive extents and `requiresGrad=false`.
- Require exact grouped-channel relations and independently derive checked
  `[N,Cout,Hout,Wout]` from positive stride/dilation/groups, non-negative symmetric padding, and
  positive static weight kernel extents.
- Admit every documented Model-valid geometry that fits checked Java/native arithmetic, native
  integer conversions, and buffer extents. Add no kernel, channel, group, stride, dilation, or
  padding limit inferred from one device execution.
- Retain the current whole-partition MPSGraph lifecycle, ABI v4, 160-byte node record, thirteen
  exports, FLOAT32-only transfer, and profile-free Runtime/Trace boundary.

## Non-goals

- Three-input biased `CONV2D`, a nullable bias, fused epilogue, or generic scalar arithmetic.
- `requiresGrad=true`, direct MPSGraph gradient selectors, generated backward ownership, training,
  saved values, or hidden outputs.
- Conv1d/Conv3d, max/average Pool1d/2d/3d, adaptive/global pooling, or convolution decomposition.
- Dynamic or zero extents, views/affine inputs, another layout/type/profile, transfer widening,
  fallback, host execution, custom-kernel implementation, or performance claims.
- A shape, group, kernel, stride, dilation, padding, profile, option, context, optimization, or
  repetition device matrix; a single-geometry capability cut; or a relaxed oracle.

## Contracts

- [`ARCHITECTURE.md` — Authority, incorporation, and precedence](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence)
  requires plans to preserve the root and incorporated scoped contracts.
- [Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)
  gives `CONV2D` only its existing reassociation/corresponding-FMA freedom plus row-scoped DAZ/FTZ.
- [Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend) owns
  lowering, native integration, storage, and materialization, not Model semantics.
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)
  keeps route construction cold and Runtime free of policy lookup.
- [Model Conv2d](../../../modules/model/tasks/0020-nchw-conv2d-semantics-and-tensor-expressions.md)
  owns the signature, NCHW/OIHW grouped cross-correlation, Shape, accumulation, padding, and
  special-value contract.
- [Compiler convolution gradients](../../../modules/compiler/tasks/0005d-attention-convolution-pooling-and-loss-gradient-completion.md)
  owns backward formulas independently of backend execution.

## Exact Model and Compiler boundary

`CONV2D` has one signature, `Conv2dAttrs 2..3 -> 1`. Bias presence is represented only by the
occurrence input count; there is no bias flag or second operation kind. `Conv2dAttrs` contains, in
order, stride height/width, padding height/width, dilation height/width, and groups. The public
unbiased `conv2d(weight, attrs)` expression is first-class.

Input/output use NCHW and weight uses `[Cout,Cin/groups,Kh,Kw]` OIHW. For each spatial axis:

```text
effectiveKernel = dilation * (kernel - 1) + 1
output          = floor((input + 2 * padding - effectiveKernel) / stride) + 1
```

The output uses promoted floating type, Shape `[N,Cout,Hout,Wout]`, unresolved Model layout later
closed by Compiler for eligible fully static results, and `requiresGrad = input || weight || bias`.
The two-input no-grad candidate therefore requires all three descriptors to be no-grad.

Each output is the sum of every declared group-local input/weight product. Conceptual padding is
positive zero and participates in ordinary multiplication, including with infinity. FLOAT32
accumulates in FLOAT32. Reassociation and a corresponding multiply/add FMA are permitted;
`ACCELERATOR` additionally permits only CONV2D-scoped DAZ/FTZ. It does not inherit MATMUL's final
exact-zero publication choice or binary ADD's exact-zero sign freedom. Term loss, arbitrary reduced
precision, algebraic replacement, cross-node contraction, and tolerance acceptance remain forbidden.

Compiler already reconstructs selected input/weight cotangents through `UNFOLD2D`, `FOLD2D`,
grouped reshape/permute, higher-rank `MATMUL`, and reduction. That graph is outside current Metal
capability. Task 0031 must not call MPSGraph convolution-gradient selectors or imply backward
closure; `requiresGrad=true` remains rejected.

## Documented direct MPSGraph mapping

`convolution2DWithSourceTensor:weightsTensor:descriptor:name:` consumes exactly source and weights.
`MPSGraphConvolution2DOpDescriptor` documents every candidate attribute:

```text
strideInX       = strideWidth       strideInY       = strideHeight
dilationRateInX = dilationWidth     dilationRateInY = dilationHeight
paddingLeft/right = paddingWidth    paddingTop/bottom = paddingHeight
paddingStyle = Explicit             dataLayout = NCHW
weightsLayout = OIHW                groups = groups
```

The descriptor therefore structurally represents the full Model geometry without a geometry probe.
The selector has no bias input. A future biased route could structurally reshape `[Cout]` to
`[1,Cout,1,1]` and use documented broadcast addition, but that internal add cannot borrow generic
Metal ADD semantics: accelerator ADD has exact-zero sign freedom that CONV2D explicitly lacks.
Bias needs its own convolution-level numerical proof and is excluded here.

## Documentation-first blocker

The MPSGraph convolution header specifies rank, layouts, groups, stride, dilation, explicit padding,
and the direct selector. It does not specify the contraction algorithm, accumulation expression,
FLOAT32 rounding sequence, algorithm-selection boundary, or a control that pins one realization.
`MPSGraphReducedPrecisionFastMathNone` can prohibit a reduced-precision option; it does not document
or force a direct declared-product contraction.

A single process/context/graph/executable/execution can exercise only one compiled geometry and its
selected opaque realization. Even a compact corpus containing group-isolation, all-term,
reassociation/FMA, DAZ/FTZ, signed-zero, NaN/infinity, and conceptual-padding witnesses cannot show
that other documented kernels, groups, strides, dilations, paddings, channels, or extents use the
same realization or remain inside the Model set. Shape-dependent alternatives such as transformed
convolution would require arithmetic not authorized merely as reassociation/corresponding FMA.

The lean policy prohibits the geometry/algorithm matrix needed to characterize those regimes. A
one-geometry capability would be an unsound probe-derived narrowing and is forbidden. No device
probe ran, and no schema, identity, capability, lowering, test, or source change exists.

## Why adjacent slices are not smaller sound routes

- `AVERAGE_POOL2D` is a different exact unrelaxed numerical family with divisor, padding, and ceil
  behavior; its direct selector does not prove convolution and needs its own semantic authorization.
- Max Pool1d is visible Pool2d composition and inherits blocked 0030 selection behavior. Average
  Pool1d inherits the unproved Average Pool2d route; rank edits do not supply missing numerical proof.
- Pool3d adds rank, geometry, schema, and selector surface while retaining the corresponding exact
  max/average questions. It is larger, not a fallback for blocked Pool2d.
- Conv1d is visible `EXPAND_DIMS -> CONV2D -> SQUEEZE` and still requires the same blocked direct
  Conv2d occurrence. Conv3d has a larger opaque contraction and no smaller proof boundary.
- Convolution or pooling backward requires unsupported generated window/fold/contraction topology
  or separate gradient selectors, saved selection policy, and wider lifecycle proof. It cannot
  establish a forward route.

These families remain separately unauthorized; none reopens 0026, 0027, or 0030 or supplies a
sound Ready Metal task.

## Prospective schema if unblocked

Because blocked 0030 implemented none of its planned numbers, the first implementation may advance
node schema `11 -> 12`, append `CONV2D=20`, add `AttributeKind.CONV2D=7`, and advance workload,
exact-policy, candidate, compatibility, route-policy, and codec identities together `12 -> 13`.
Retain wires `1..19`, ABI v4, the 160-byte record, and thirteen exports.

Encode two inputs and one output, `attributeCount=7`, absent-axis sentinel, `auxiliary=0`, and the
seven `Conv2dAttrs` values in declaration order; require cells `7..15` zero. Kernel extents derive
from weight Shape. Fixed NCHW/OIHW/explicit-padding policy and bias absence need no payload flag.
Java validates profile, no-grad metadata, descriptors, topology, groups, geometry, and output Shape;
native independently validates schema, types, Shapes, conversions, and the same geometry before
graph creation. `requiresGrad` does not cross ABI v4 and remains Java-preflight/workload identity.

## Lean validation if unblocked

Use focused capability/preparer positives and rejection boundaries; one exact schema/raw-ABI record
and malformed-field checks; version-thirteen identity/codec stale rejection; native build/export
audit; and one CPU-free public Engine exact-integer unbiased no-grad smoke. Do not add a device
geometry matrix, duplicate the numerical corpus in permanent tests, or claim backward/bias support.

## Unblocking and result

Task 0031 can unblock only through one of these separately authorized paths:

1. an exact custom Metal kernel whose declared-product evaluation is proved for the complete
   candidate geometry; or
2. a preceding Model/architecture decision that explicitly broadens the convolution allowed-result
   envelope to cover the intended MPSGraph realization.

Do not retry with one MPSGraph geometry, narrow capability to a passing shape, infer behavior from a
pooling or MATMUL route, or weaken the oracle. Task 0031 is Blocked without a device probe or any
production/test change. No Metal task is Ready.
