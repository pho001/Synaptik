# Task 0032: Profile-Common [Canonical](../../../../glossary.md#term-canonical-form) [FLOAT32](../../../../glossary.md#term-float32) No-Grad [MSE](../../../../glossary.md#mean-squared-error--mse) NONE

## Reader context

Mean-squared error (MSE) remained one Model operation even if a backend could spell its formula
as subtraction followed by square. NONE removed the final reduction; it did not transfer the
numerical permissions of separately supported binary operations into this loss.

Prediction and target had distinct semantic roles even when backed by the same graph value.
Equal-shape, type and layout descriptions did not exclude exceptional or subnormal buffer values.
Model owned the atomic loss contract; Metal needed a route satisfying that contract across the
candidate domain. No task-specific probe ran, and the proposed schema values were not reserved.
The [current loss and numerical domains](../../../../backend-guide/metal-backend.md#ordinary-numerical-and-exact-domains)
explain later MSE execution separately from this blocked two-profile assessment.

## Status

Blocked

## Change class

Class C if unblocked — the candidate would add a first-class [loss](../../../../glossary.md#loss) [operation](../../../../glossary.md#operation) to both [Metal](../../../../glossary.md#term-metal) profile
[matrices](../../../../glossary.md#term-matrix-meanings), evolve [backend](../../../../glossary.md#backend)-private [schema](../../../../glossary.md#term-schema) and route [identities](../../../../glossary.md#term-identity-meanings), and change generated native [graphs](../../../../glossary.md#graph).
This [planning](../../../../glossary.md#term-planning)-only blocker record changes no executable behavior and ran no [device](../../../../glossary.md#term-device) probe.

## Goal

Assess the smallest current first-class [Metal](../../../../glossary.md#term-metal) [loss](../../../../glossary.md#loss) [slice](../../../../glossary.md#slice): [canonical](../../../../glossary.md#term-canonical-form), positive, fully static,
same-typed [`FLOAT32`](../../../../glossary.md#term-float32), no-grad `MEAN_SQUARED_ERROR` with `LossReduction.NONE` under both numerical
profiles. `NONE` removes complete-domain loss accumulation, [scalar](../../../../glossary.md#term-scalar) [publication](../../../../glossary.md#term-publication), denominator, and
empty-[mean](../../../../glossary.md#term-mean) behavior while preserving one atomic [Model](../../../../glossary.md#term-model) loss [occurrence](../../../../glossary.md#term-occurrence).

[MPSGraph](../../../../glossary.md#term-mpsgraph) exposes no direct [MSE](../../../../glossary.md#mean-squared-error--mse) selector. Its documented arithmetic surface can express subtraction
followed by square, but that [decomposition](../../../../glossary.md#term-decomposition) does not transfer the [operation](../../../../glossary.md#operation)-scoped [`ACCELERATOR`](../../../../glossary.md#term-accelerator)
[binary](../../../../glossary.md#term-binary-descriptor-and-metadata) relaxations into loss semantics. Loss remains unrelaxed under both profiles. The headers do
not close independent [FTZ](../../../../glossary.md#term-ftz), special-class, and [finite](../../../../glossary.md#term-finite)-[rounding](../../../../glossary.md#term-rounding) questions for the composed result;
one execution cannot close all of them and the required [matrix](../../../../glossary.md#term-matrix-meanings) is prohibited. Task 0032 is
therefore Blocked before any probe or production edit.

## Candidate domain if unblocked

- Admit exactly `LossKind.MEAN_SQUARED_ERROR + MeanSquaredErrorAttrs(LossReduction.NONE)` with
  ordered `[prediction, target]` inputs and one output.
- Report the identical exact [occurrence](../../../../glossary.md#term-occurrence) under `STRICT_IEEE` and [`ACCELERATOR`](../../../../glossary.md#term-accelerator). Accelerator adds no
  loss result freedom and may not remove the strict capability.<span data-glossary-navigation="true"> ([loss glossary](../../../../glossary.md#loss))</span>
- Require all three [descriptors](../../../../glossary.md#term-descriptor) to be [`FLOAT32`](../../../../glossary.md#term-float32), `requiresGrad=false`, canonical dense contiguous,
  zero-offset, non-view, fully static rank <span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [contiguous glossary](../../../../glossary.md#term-contiguous); [view glossary](../../../../glossary.md#term-view-meanings); [rank glossary](../../../../glossary.md#term-rank))</span>`1..16`, and positive in every [extent](../../../../glossary.md#term-extent).
- Require prediction and target to have the exact same Shape and output to retain that Shape.
  Broadcasting, implicit casts, rank-zero feeds, empty extents, and dynamic obligations are absent.<span data-glossary-navigation="true"> ([prediction glossary](../../../../glossary.md#term-prediction); [target glossary](../../../../glossary.md#term-target); [Shape glossary](../../../../glossary.md#shape); [Broadcasting glossary](../../../../glossary.md#broadcasting); [rank-zero glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- Permit repeated input identity, fan-out, direct publication, and checked element/byte geometry
  without changing the two semantic input roles.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [fan-out glossary](../../../../glossary.md#term-fan-out); [publication glossary](../../../../glossary.md#term-publication); [element glossary](../../../../glossary.md#term-element))</span>
- Preserve the current whole-partition prepared lifecycle, FLOAT32-only transfer, profile-free
  Runtime/Trace boundary, and one atomic loss node at capability and lowering boundaries.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [lifecycle glossary](../../../../glossary.md#lifecycle); [Runtime glossary](../../../../glossary.md#term-runtime); [Trace glossary](../../../../glossary.md#trace); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [node glossary](../../../../glossary.md#node); [lowering glossary](../../../../glossary.md#lowering))</span>

## Non-goals

- `LossReduction.SUM` or [`MEAN`](../../../../glossary.md#term-mean); dense-target or index-target categorical cross-entropy; another
  loss kind, default, weighting, mask, smoothing, probability-input form, or generic loss registry.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [mask glossary](../../../../glossary.md#term-boolean-mask); [registry glossary](../../../../glossary.md#term-registry))</span>
- BFLOAT16, FLOAT64, mixed promotion, FLOAT16, integral/BOOL loss data, views, broadcasting,
  zero/dynamic extents, another layout, or a value-dependent capability branch.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT16 glossary](../../../../glossary.md#term-float16); [BOOL glossary](../../../../glossary.md#data-type--datatype); [views glossary](../../../../glossary.md#term-view-meanings); [broadcasting glossary](../../../../glossary.md#broadcasting); [layout glossary](../../../../glossary.md#layout))</span>
- `requiresGrad=true`, Compiler-generated pullbacks, direct MPSGraph gradient selectors, training,
  saved values, hidden outputs, or a Metal-only backward claim.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [gradient glossary](../../../../glossary.md#term-gradient); [Metal glossary](../../../../glossary.md#term-metal))</span>
- Importing [binary](../../../../glossary.md#term-binary-descriptor-and-metadata)/[scalar](../../../../glossary.md#term-scalar)/reduction [`ACCELERATOR`](../../../../glossary.md#term-accelerator) DAZ/FTZ or signed-zero permissions, recognizing a
  public <span data-glossary-navigation="true"> ([DAZ glossary](../../../../glossary.md#term-daz); [FTZ glossary](../../../../glossary.md#term-ftz); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>`SUB -> MUL/SQUARE` [topology](../../../../glossary.md#term-topology) as [loss](../../../../glossary.md#loss), fallback, [host](../../../../glossary.md#term-host) repair, or special-case [dispatch](../../../../glossary.md#term-dispatch).
- A kind, reduction, Shape, option, optimization, context, or special-value probe matrix; narrowing
  capability to a passing value or one tested Shape; tolerance invented by Metal.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [narrowing glossary](../../../../glossary.md#term-narrowing); [tolerance glossary](../../../../glossary.md#term-tolerance))</span>

## Contracts

- [`ARCHITECTURE.md` — Authority, incorporation, and precedence](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence)
  preserves the root and incorporated scoped contracts.
- [Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)
  places loss in the unlisted-family row: both profiles retain the exact operation contract.<span data-glossary-navigation="true"> ([loss glossary](../../../../glossary.md#loss); [operation glossary](../../../../glossary.md#operation))</span>
- [Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [backend glossary](../../../../glossary.md#backend))</span> owns
  truthful capability, lowering, native integration, storage, and materialization, not loss meaning.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [materialization glossary](../../../../glossary.md#materialization))</span>
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
  keeps route and resource selection cold and Runtime free of operation-policy lookup.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- [Model MSE](../../../modules/model/tasks/0022-mean-squared-error-loss.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [MSE glossary](../../../../glossary.md#mean-squared-error--mse))</span> owns signature, exact
  Shape, promotion, reduction, special-value, and metadata semantics.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [Shape glossary](../../../../glossary.md#shape); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- [Compiler loss gradients](../../../modules/compiler/tasks/0005d-attention-convolution-pooling-and-loss-gradient-completion.md)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [gradients glossary](../../../../glossary.md#term-gradient))</span>
  owns pullback construction independently of backend forward execution.
- [CPU loss execution](../../cpu/tasks/0008i-portable-loss-family-execution.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span> and
  [CPU vector MSE NONE](../../cpu/tasks/0008m-vector-mse-none.md)<span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector))</span> provide an independent atomic-loss
  oracle and bounded same-typed precedent, not Metal numerical authorization.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [oracle glossary](../../../../glossary.md#term-test-oracle))</span>

## Exact [Model](../../../../glossary.md#term-model) and [Compiler](../../../../glossary.md#term-compiler) boundary

`MEAN_SQUARED_ERROR` accepts only `MeanSquaredErrorAttrs 2 -> 1`. For logical [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) `i` after
both inputs participate as the result type:

```text
difference[i] = prediction[i] - target[i]
loss[i]       = difference[i] * difference[i]
```

`NONE` retains the exact [prediction](../../../../glossary.md#term-prediction) [Shape](../../../../glossary.md#shape) and promoted type. For this candidate, every operand and
result is [`FLOAT32`](../../../../glossary.md#term-float32); no reduction or denominator exists. The result's [gradient](../../../../glossary.md#term-gradient) [metadata](../../../../glossary.md#term-metadata) is
`prediction.requiresGrad || target.requiresGrad`, so the selected no-grad row requires all three
flags false.

A [NaN](../../../../glossary.md#term-nan) input or equal-sign [infinity](../../../../glossary.md#term-infinity) pair produces NaN. One [finite](../../../../glossary.md#term-finite) and one infinite input, or
opposite-sign [infinities](../../../../glossary.md#nan-infinity-and-signed-zero), produces [positive infinity](../../../../glossary.md#nan-infinity-and-signed-zero). Every exact-zero difference, including every
[signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero) pairing, squares to [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero). Finite subtraction or square may [overflow](../../../../glossary.md#term-overflow) or
[underflow](../../../../glossary.md#term-underflow) in `FLOAT32`. Equal-or-wider intermediates, [fusion](../../../../glossary.md#term-fusion), and finite [reassociation](../../../../glossary.md#term-reassociation) are permitted
only while these special classes and the [operation](../../../../glossary.md#operation)'s result contract remain intact; no [binary](../../../../glossary.md#term-binary-descriptor-and-metadata)
profile row is inherited.

Compiler inference independently proves the exact Shape, promoted type, and OR gradient flag.
Compiler pullbacks restore a reduction [cotangent](../../../../glossary.md#cotangent) and use `2 * (prediction - target)` and its
negative for selected roles. They are deliberately outside this no-grad forward candidate; Task
0032 does not use [MPSGraph](../../../../glossary.md#term-mpsgraph) [loss](../../../../glossary.md#loss)-gradient selectors or claim that generated backward [topology](../../../../glossary.md#term-topology) is
[Metal](../../../../glossary.md#term-metal)-supported.

## Installed [MPSGraph](../../../../glossary.md#term-mpsgraph) surface

The installed `MPSGraphLossOps.h` exposes one forward [loss](../../../../glossary.md#loss) selector:
`softMaxCrossEntropyWithSourceTensor:labelsTensor:axis:reductionType:name:`. It also exposes that
[operation](../../../../glossary.md#operation)'s [gradient](../../../../glossary.md#term-gradient) selector and `None/Axis`, `Sum`, and [`Mean`](../../../../glossary.md#term-mean) reduction values. It exposes no
[MSE](../../../../glossary.md#mean-squared-error--mse), index-[target](../../../../glossary.md#term-target), ignore-index, selected-data, or pure classification-loss selector.

`MPSGraphArithmeticOps.h` documents
`subtractionWithPrimaryTensor:secondaryTensor:name:` and `squareWithTensor:name:`. These can
structurally produce the MSE [Shape](../../../../glossary.md#shape) from two exact-shape [tensors](../../../../glossary.md#tensor), but the header specifies only the
arithmetic expression and [elementwise](../../../../glossary.md#term-elementwise) result. It does not promise no [DAZ](../../../../glossary.md#term-daz)/[FTZ](../../../../glossary.md#term-ftz), special-value
classification, intermediate format, [finite](../../../../glossary.md#term-finite) [rounding](../../../../glossary.md#term-rounding) boundary, or an operation-scoped MSE result.
`MPSGraphReducedPrecisionFastMathNone` disables a reduced-[precision](../../../../glossary.md#term-precision) option; it does not document
those missing properties.

## Documentation-first blocker

[Loss](../../../../glossary.md#loss) is explicitly unrelaxed in the [normative](../../../../glossary.md#term-normative) profile table. Complete [Metal](../../../../glossary.md#term-metal) 0015 authorizes
`ADD/SUB/MUL/DIV` only under their named [accelerator](../../../../glossary.md#term-accelerator) [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) row. That permission cannot be
smuggled through a private subtraction/square [decomposition](../../../../glossary.md#term-decomposition) of [MSE](../../../../glossary.md#mean-squared-error--mse). Historical Metal 0005 also
recorded 12,096 [subnormal](../../../../glossary.md#term-subnormal)-domain [MPSGraph](../../../../glossary.md#term-mpsgraph) arithmetic mismatches and found that basic [MSL](../../../../glossary.md#term-metal-shading-language-msl) float
arithmetic, including safe math mode, flushed subnormals. This evidence blocks any assumption that
a composed arithmetic [graph](../../../../glossary.md#graph) is exact merely because its [Shapes](../../../../glossary.md#shape) and formulas match.

The remaining questions are independent: preservation of a representable final subnormal MSE
value; [NaN](../../../../glossary.md#term-nan), equal/opposite [infinity](../../../../glossary.md#term-infinity), and [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero) result classes across a composed graph; and
[finite](../../../../glossary.md#term-finite) computation/[rounding](../../../../glossary.md#term-rounding) within the [Model](../../../../glossary.md#term-model) envelope. A single execution can establish at most
one concrete undocumented property. Passing one cell cannot establish the other properties or
every admitted [extent](../../../../glossary.md#term-extent), while a corpus crossed with Shapes, optimizations, and special classes is
the prohibited [matrix](../../../../glossary.md#term-matrix-meanings).

A [descriptor](../../../../glossary.md#term-descriptor) or value [narrowing](../../../../glossary.md#term-narrowing) does not repair the proof: capability cannot exclude future
subnormal or exceptional inputs within an admitted [`FLOAT32`](../../../../glossary.md#term-float32) [buffer](../../../../glossary.md#term-buffer). No probe ran. No [schema](../../../../glossary.md#term-schema),
[identity](../../../../glossary.md#term-identity-meanings), capability, [lowering](../../../../glossary.md#lowering), native, test, or source change exists.

## Why adjacent [loss](../../../../glossary.md#loss) slices are larger

- Dense categorical cross-entropy has a direct [MPSGraph](../../../../glossary.md#term-mpsgraph) [softmax](../../../../glossary.md#softmax--log-softmax)-cross-entropy selector, but
  `NONE` still accumulates over the class axis. Its header does not promise Model-required stable
  log-sum-exp, exact zero-weight exclusion of non-finite log probabilities, special classes, or
  accumulation/rounding behavior. Those are more independent questions than MSE <span data-glossary-navigation="true"> ([class axis glossary](../../../../glossary.md#class-axis); [Model glossary](../../../../glossary.md#term-model); [log-sum-exp glossary](../../../../glossary.md#log-sum-exp); [weight glossary](../../../../glossary.md#term-weight); [finite glossary](../../../../glossary.md#term-finite); [log glossary](../../../../glossary.md#term-log); [rounding glossary](../../../../glossary.md#term-rounding))</span>`NONE`.
- Index categorical cross-entropy has no direct selector. It additionally needs INT32/INT64 target
  handling, bounds-before-read validation, optional exact ignore comparison, non-ignored mean
  counting, stable log-sum-exp, and selected-logit subtraction. GATHER and ONE_HOT are separate
  operations and cannot be relabeled as an exact loss fallback.<span data-glossary-navigation="true"> ([INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [target glossary](../../../../glossary.md#term-target); [validation glossary](../../../../glossary.md#term-validation); [logit glossary](../../../../glossary.md#logit); [GATHER glossary](../../../../glossary.md#gather))</span>
- [MSE](../../../../glossary.md#mean-squared-error--mse) `SUM`/[`MEAN`](../../../../glossary.md#term-mean) add complete-domain unrelaxed accumulation; `MEAN` also adds a denominator,
  empty-domain NaN, scalar publication, and rank-zero lifecycle. Existing accelerator reductions
  authorize only their own operation row and cannot widen loss reductions.<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [scalar glossary](../../../../glossary.md#term-scalar); [publication glossary](../../../../glossary.md#term-publication); [rank-zero glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [lifecycle glossary](../../../../glossary.md#lifecycle); [accelerator glossary](../../../../glossary.md#term-accelerator); [operation glossary](../../../../glossary.md#operation))</span>
- Categorical `SUM`/`MEAN` retain their class-axis work and add sample-domain accumulation and
  denominator rules. Direct loss-gradient selectors or Compiler pullbacks add unsupported backward
  topology and do not prove any forward result.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [gradient glossary](../../../../glossary.md#term-gradient); [Compiler glossary](../../../../glossary.md#term-compiler); [topology glossary](../../../../glossary.md#term-topology))</span>

None is a smaller fallback, reopens 0026/0027/0030/0031, or supplies a [Ready](../../../../glossary.md#term-ready) [Metal](../../../../glossary.md#term-metal) task.

## Conditional [schema](../../../../glossary.md#term-schema) only

Task 0032 reserves and changes no schema number. Blocked 0030 and 0031 implemented none of their
conditional numbers, so a separately authorized [MPSGraph](../../../../glossary.md#term-mpsgraph) implementation could then advance [node](../../../../glossary.md#node)
schema `11 -> 12`, append `MEAN_SQUARED_ERROR=20`, add
`AttributeKind.LOSS_REDUCTION=7`, and advance workload, exact-policy, candidate, compatibility,
route-policy, and [codec](../../../../glossary.md#term-codec) [identities](../../../../glossary.md#term-identity-meanings) together `12 -> 13`. Retain wires `1..19`, [ABI](../../../../glossary.md#term-application-binary-interface-abi) v4, the 160-byte
record, and thirteen exports.

A conditional [MSE](../../../../glossary.md#mean-squared-error--mse) record has two ordered inputs, one [canonical](../../../../glossary.md#term-canonical-form) output, `attributeCount=1`, absent-
[axis](../../../../glossary.md#term-axis) sentinel, `auxiliary=0`, `attributeValues[0]=1` for a schema-local `NONE` [identity](../../../../glossary.md#term-identity-meanings), and zero in
cells `1..15`. Java and native [preflight](../../../../glossary.md#term-preflight) accept no other reduction or [loss](../../../../glossary.md#loss) kind. Java validates the
identical both-profile domain, [descriptors](../../../../glossary.md#term-descriptor), no-grad [metadata](../../../../glossary.md#term-metadata), exact [Shape](../../../../glossary.md#shape), [topology](../../../../glossary.md#term-topology), and checked
geometry. Native independently validates schema, [FLOAT32](../../../../glossary.md#term-float32) types, Shapes, sentinels, and conversions.
`requiresGrad` remains Java-preflight/workload identity and does not cross ABI v4.

These values are conditional, not reserved. While Task 0032 is Blocked, schema 11, wires `1..19`,
attribute kinds `0..6`, version-twelve identities, ABI v4, and thirteen exports remain unchanged.
An exact custom-[kernel](../../../../glossary.md#kernel) unblock must define its own typed route instead of pretending to consume the
conditional MPSGraph wire.

## Unblocking and result

Task 0032 can unblock only through one separately authorized path that closes the complete
[operation](../../../../glossary.md#operation)-scoped domain before implementation:

1. an exact custom [Metal](../../../../glossary.md#term-metal) [kernel](../../../../glossary.md#kernel) with documented proof of the full [MSE](../../../../glossary.md#mean-squared-error--mse) `NONE` result set, including
   final subnormals and every required special class; ordinary MSL float arithmetic is insufficient
   under the retained FTZ evidence;<span data-glossary-navigation="true"> ([subnormals glossary](../../../../glossary.md#term-subnormal); [MSL glossary](../../../../glossary.md#term-metal-shading-language-msl); [FTZ glossary](../../../../glossary.md#term-ftz))</span>
2. authoritative MPSGraph documentation or a direct MSE selector contract that closes the
   operation's complete numerical envelope, leaving at most one genuinely isolated property for a
   later lean gate; or<span data-glossary-navigation="true"> ([MPSGraph glossary](../../../../glossary.md#term-mpsgraph))</span>
3. a preceding Model/architecture decision that gives MSE a named, operation-specific accelerator
   envelope covering the intended implementation without importing generic binary fast math.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [accelerator glossary](../../../../glossary.md#term-accelerator))</span>

Do not [run](../../../../glossary.md#term-run) a probe now: one run cannot close the independent current gaps, and the required [matrix](../../../../glossary.md#term-matrix-meanings)
is prohibited. Do not narrow [Shape](../../../../glossary.md#shape) or values, infer from [CPU](../../../../glossary.md#term-cpu) or [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) capability, weaken the
[oracle](../../../../glossary.md#term-test-oracle), or reserve conditional [schema](../../../../glossary.md#term-schema) [identities](../../../../glossary.md#term-identity-meanings). Task 0032 is Blocked without a [device](../../../../glossary.md#term-device) probe or
any production/test change. No Metal task is [Ready](../../../../glossary.md#term-ready).
