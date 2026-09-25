# Task 0032: Profile-Common Canonical FLOAT32 No-Grad MSE NONE

## Status

Blocked

## Change class

Class C if unblocked — the candidate would add a first-class loss operation to both Metal profile
matrices, evolve backend-private schema and route identities, and change generated native graphs.
This planning-only blocker record changes no executable behavior and ran no device probe.

## Goal

Assess the smallest current first-class Metal loss slice: canonical, positive, fully static,
same-typed `FLOAT32`, no-grad `MEAN_SQUARED_ERROR` with `LossReduction.NONE` under both numerical
profiles. `NONE` removes complete-domain loss accumulation, scalar publication, denominator, and
empty-mean behavior while preserving one atomic Model loss occurrence.

MPSGraph exposes no direct MSE selector. Its documented arithmetic surface can express subtraction
followed by square, but that decomposition does not transfer the operation-scoped `ACCELERATOR`
binary relaxations into loss semantics. Loss remains unrelaxed under both profiles. The headers do
not close independent FTZ, special-class, and finite-rounding questions for the composed result;
one execution cannot close all of them and the required matrix is prohibited. Task 0032 is
therefore Blocked before any probe or production edit.

## Candidate domain if unblocked

- Admit exactly `LossKind.MEAN_SQUARED_ERROR + MeanSquaredErrorAttrs(LossReduction.NONE)` with
  ordered `[prediction, target]` inputs and one output.
- Report the identical exact occurrence under `STRICT_IEEE` and `ACCELERATOR`. Accelerator adds no
  loss result freedom and may not remove the strict capability.
- Require all three descriptors to be `FLOAT32`, `requiresGrad=false`, canonical dense contiguous,
  zero-offset, non-view, fully static rank `1..16`, and positive in every extent.
- Require prediction and target to have the exact same Shape and output to retain that Shape.
  Broadcasting, implicit casts, rank-zero feeds, empty extents, and dynamic obligations are absent.
- Permit repeated input identity, fan-out, direct publication, and checked element/byte geometry
  without changing the two semantic input roles.
- Preserve the current whole-partition prepared lifecycle, FLOAT32-only transfer, profile-free
  Runtime/Trace boundary, and one atomic loss node at capability and lowering boundaries.

## Non-goals

- `LossReduction.SUM` or `MEAN`; dense-target or index-target categorical cross-entropy; another
  loss kind, default, weighting, mask, smoothing, probability-input form, or generic loss registry.
- BFLOAT16, FLOAT64, mixed promotion, FLOAT16, integral/BOOL loss data, views, broadcasting,
  zero/dynamic extents, another layout, or a value-dependent capability branch.
- `requiresGrad=true`, Compiler-generated pullbacks, direct MPSGraph gradient selectors, training,
  saved values, hidden outputs, or a Metal-only backward claim.
- Importing binary/scalar/reduction `ACCELERATOR` DAZ/FTZ or signed-zero permissions, recognizing a
  public `SUB -> MUL/SQUARE` topology as loss, fallback, host repair, or special-case dispatch.
- A kind, reduction, Shape, option, optimization, context, or special-value probe matrix; narrowing
  capability to a passing value or one tested Shape; tolerance invented by Metal.

## Contracts

- [`ARCHITECTURE.md` — Authority, incorporation, and precedence](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence)
  preserves the root and incorporated scoped contracts.
- [Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)
  places loss in the unlisted-family row: both profiles retain the exact operation contract.
- [Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend) owns
  truthful capability, lowering, native integration, storage, and materialization, not loss meaning.
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)
  keeps route and resource selection cold and Runtime free of operation-policy lookup.
- [Model MSE](../../../modules/model/tasks/0022-mean-squared-error-loss.md) owns signature, exact
  Shape, promotion, reduction, special-value, and metadata semantics.
- [Compiler loss gradients](../../../modules/compiler/tasks/0005d-attention-convolution-pooling-and-loss-gradient-completion.md)
  owns pullback construction independently of backend forward execution.
- [CPU loss execution](../../cpu/tasks/0008i-portable-loss-family-execution.md) and
  [CPU vector MSE NONE](../../cpu/tasks/0008m-vector-mse-none.md) provide an independent atomic-loss
  oracle and bounded same-typed precedent, not Metal numerical authorization.

## Exact Model and Compiler boundary

`MEAN_SQUARED_ERROR` accepts only `MeanSquaredErrorAttrs 2 -> 1`. For logical coordinate `i` after
both inputs participate as the result type:

```text
difference[i] = prediction[i] - target[i]
loss[i]       = difference[i] * difference[i]
```

`NONE` retains the exact prediction Shape and promoted type. For this candidate, every operand and
result is `FLOAT32`; no reduction or denominator exists. The result's gradient metadata is
`prediction.requiresGrad || target.requiresGrad`, so the selected no-grad row requires all three
flags false.

A NaN input or equal-sign infinity pair produces NaN. One finite and one infinite input, or
opposite-sign infinities, produces positive infinity. Every exact-zero difference, including every
signed-zero pairing, squares to positive zero. Finite subtraction or square may overflow or
underflow in `FLOAT32`. Equal-or-wider intermediates, fusion, and finite reassociation are permitted
only while these special classes and the operation's result contract remain intact; no binary
profile row is inherited.

Compiler inference independently proves the exact Shape, promoted type, and OR gradient flag.
Compiler pullbacks restore a reduction cotangent and use `2 * (prediction - target)` and its
negative for selected roles. They are deliberately outside this no-grad forward candidate; Task
0032 does not use MPSGraph loss-gradient selectors or claim that generated backward topology is
Metal-supported.

## Installed MPSGraph surface

The installed `MPSGraphLossOps.h` exposes one forward loss selector:
`softMaxCrossEntropyWithSourceTensor:labelsTensor:axis:reductionType:name:`. It also exposes that
operation's gradient selector and `None/Axis`, `Sum`, and `Mean` reduction values. It exposes no
MSE, index-target, ignore-index, selected-data, or pure classification-loss selector.

`MPSGraphArithmeticOps.h` documents
`subtractionWithPrimaryTensor:secondaryTensor:name:` and `squareWithTensor:name:`. These can
structurally produce the MSE Shape from two exact-shape tensors, but the header specifies only the
arithmetic expression and elementwise result. It does not promise no DAZ/FTZ, special-value
classification, intermediate format, finite rounding boundary, or an operation-scoped MSE result.
`MPSGraphReducedPrecisionFastMathNone` disables a reduced-precision option; it does not document
those missing properties.

## Documentation-first blocker

Loss is explicitly unrelaxed in the normative profile table. Complete Metal 0015 authorizes
`ADD/SUB/MUL/DIV` only under their named accelerator binary row. That permission cannot be
smuggled through a private subtraction/square decomposition of MSE. Historical Metal 0005 also
recorded 12,096 subnormal-domain MPSGraph arithmetic mismatches and found that basic MSL float
arithmetic, including safe math mode, flushed subnormals. This evidence blocks any assumption that
a composed arithmetic graph is exact merely because its Shapes and formulas match.

The remaining questions are independent: preservation of a representable final subnormal MSE
value; NaN, equal/opposite infinity, and signed-zero result classes across a composed graph; and
finite computation/rounding within the Model envelope. A single execution can establish at most
one concrete undocumented property. Passing one cell cannot establish the other properties or
every admitted extent, while a corpus crossed with Shapes, optimizations, and special classes is
the prohibited matrix.

A descriptor or value narrowing does not repair the proof: capability cannot exclude future
subnormal or exceptional inputs within an admitted `FLOAT32` buffer. No probe ran. No schema,
identity, capability, lowering, native, test, or source change exists.

## Why adjacent loss slices are larger

- Dense categorical cross-entropy has a direct MPSGraph softmax-cross-entropy selector, but
  `NONE` still accumulates over the class axis. Its header does not promise Model-required stable
  log-sum-exp, exact zero-weight exclusion of non-finite log probabilities, special classes, or
  accumulation/rounding behavior. Those are more independent questions than MSE `NONE`.
- Index categorical cross-entropy has no direct selector. It additionally needs INT32/INT64 target
  handling, bounds-before-read validation, optional exact ignore comparison, non-ignored mean
  counting, stable log-sum-exp, and selected-logit subtraction. GATHER and ONE_HOT are separate
  operations and cannot be relabeled as an exact loss fallback.
- MSE `SUM`/`MEAN` add complete-domain unrelaxed accumulation; `MEAN` also adds a denominator,
  empty-domain NaN, scalar publication, and rank-zero lifecycle. Existing accelerator reductions
  authorize only their own operation row and cannot widen loss reductions.
- Categorical `SUM`/`MEAN` retain their class-axis work and add sample-domain accumulation and
  denominator rules. Direct loss-gradient selectors or Compiler pullbacks add unsupported backward
  topology and do not prove any forward result.

None is a smaller fallback, reopens 0026/0027/0030/0031, or supplies a Ready Metal task.

## Conditional schema only

Task 0032 reserves and changes no schema number. Blocked 0030 and 0031 implemented none of their
conditional numbers, so a separately authorized MPSGraph implementation could then advance node
schema `11 -> 12`, append `MEAN_SQUARED_ERROR=20`, add
`AttributeKind.LOSS_REDUCTION=7`, and advance workload, exact-policy, candidate, compatibility,
route-policy, and codec identities together `12 -> 13`. Retain wires `1..19`, ABI v4, the 160-byte
record, and thirteen exports.

A conditional MSE record has two ordered inputs, one canonical output, `attributeCount=1`, absent-
axis sentinel, `auxiliary=0`, `attributeValues[0]=1` for a schema-local `NONE` identity, and zero in
cells `1..15`. Java and native preflight accept no other reduction or loss kind. Java validates the
identical both-profile domain, descriptors, no-grad metadata, exact Shape, topology, and checked
geometry. Native independently validates schema, FLOAT32 types, Shapes, sentinels, and conversions.
`requiresGrad` remains Java-preflight/workload identity and does not cross ABI v4.

These values are conditional, not reserved. While Task 0032 is Blocked, schema 11, wires `1..19`,
attribute kinds `0..6`, version-twelve identities, ABI v4, and thirteen exports remain unchanged.
An exact custom-kernel unblock must define its own typed route instead of pretending to consume the
conditional MPSGraph wire.

## Unblocking and result

Task 0032 can unblock only through one separately authorized path that closes the complete
operation-scoped domain before implementation:

1. an exact custom Metal kernel with documented proof of the full MSE `NONE` result set, including
   final subnormals and every required special class; ordinary MSL float arithmetic is insufficient
   under the retained FTZ evidence;
2. authoritative MPSGraph documentation or a direct MSE selector contract that closes the
   operation's complete numerical envelope, leaving at most one genuinely isolated property for a
   later lean gate; or
3. a preceding Model/architecture decision that gives MSE a named, operation-specific accelerator
   envelope covering the intended implementation without importing generic binary fast math.

Do not run a probe now: one run cannot close the independent current gaps, and the required matrix
is prohibited. Do not narrow Shape or values, infer from CPU or binary capability, weaken the
oracle, or reserve conditional schema identities. Task 0032 is Blocked without a device probe or
any production/test change. No Metal task is Ready.
