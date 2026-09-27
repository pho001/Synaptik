# Task 0064: Convolution, Pooling, and Attention Boundary

## Status

Review needed — the plan-only audit is complete. This brief, the Metal master plan, and the project
roadmap are the only authorized planning edits. No production, native, test, package, probe, or
benchmark edit may begin until an independent Class C plan review returns `APPROVE`. That review
will authorize one serial implementation without another approval checkpoint.

## Change class

Class C — the reviewed implementation will widen Metal capability for six convolution/pooling
kinds, add exact custom-kernel execution, expand the authenticated Conv1d/Pool1d composition
boundary, advance private route identities, change public Engine behavior, and update current
backend documentation. Scaled dot-product attention and convolution transpose remain fail-closed
for explicit numerical, selector, Model, schema, and gradient reasons. An independent cumulative
code, evidence, documentation, and security review is required after implementation.

## Goal

Implement the maximal deterministic domain currently provable from Model and Compiler contracts for
`CONV2D`, `CONV3D`, `MAX_POOL2D`, `AVERAGE_POOL2D`, `MAX_POOL3D`, and `AVERAGE_POOL3D` through one
fixed custom-program route. Preserve exact NCHW/NCDHW layout, grouping, dilation, stride, padding,
ceil-grid, fixed-divisor, special-class, signed-zero, bias, promotion, output-Shape, and gradient-
metadata contracts. Select the route before resource creation and use no opaque-selector numerical
assumption, timing, autotuning, value-dependent choice, retry, fallback, or host repair.

Audit `SCALED_DOT_PRODUCT_ATTENTION`, its optional mask/causal/scale/two-output contract, and the
absence of `CONV_TRANSPOSE` completely, but do not make either production-capable without the
missing proof or Model/schema contract.

## Scope

### Exact audited 115-kind registry boundary

The current registry has exactly seven family rows in scope:

| Wire/kind | Arity and attribute wire | Exact Model role | Pre-cutover Metal state | Task-0064 decision |
|---|---|---|---|---|
| 35 `SCALED_DOT_PRODUCT_ATTENTION` | 3..4 inputs, 1..2 outputs; 39 `ATTENTION` | broadcast query/key/value, optional BOOL mask, optional scale, causal guard, output and optional saved weights | non-executable; `DIRECT / MD_ATTENTION_BASE`; custom `PENDING / CP_CONTRACT` | remain blocked |
| 36 `CONV2D` | 2..3/1; 40 `CONV_2D` | grouped NCHW cross-correlation with optional bias | non-executable; `DIRECT / MD_CONV_BASE`; custom pending | accelerator custom route |
| 37 `CONV3D` | 2..3/1; 41 `CONV_3D` | grouped NCDHW cross-correlation with optional bias | non-executable; `DIRECT / MD_CONV_BASE`; custom pending | accelerator custom route |
| 97 `MAX_POOL2D` | 1/1; 19 `WINDOW_2D` | exact NCHW winner selection, excluded padding | non-executable; `DIRECT / MD_POOL2D`; custom `PENDING / CP_AGGREGATE` | exact both-profile custom route |
| 98 `AVERAGE_POOL2D` | 1/1; 19 `WINDOW_2D` | fixed-count NCHW average including padding | non-executable; `DIRECT / MD_POOL2D`; custom pending | accelerator FLOAT32 custom route |
| 99 `MAX_POOL3D` | 1/1; 22 `WINDOW_3D` | exact NCDHW winner selection, excluded padding | non-executable; `COMPOSED / MC_WINDOW3D`; custom pending | exact both-profile custom route |
| 100 `AVERAGE_POOL3D` | 1/1; 22 `WINDOW_3D` | fixed-count NCDHW average including padding | non-executable; `COMPOSED / MC_WINDOW3D`; custom pending | accelerator FLOAT32 custom route |

The schema-15 attribute grammar already carries every admitted intrinsic field:

- `CONV_2D` has seven words in declaration order: two strides, two symmetric paddings, two
  dilations, and groups. `CONV_3D` has ten words: three strides, three paddings, three dilations,
  and groups. Stride, dilation, and groups are positive; padding is non-negative. Kernel extents,
  optional bias presence, types, and Shapes remain occurrence facts.
- `WINDOW_2D` has nine words: kernel H/W, stride H/W, padding H/W, dilation H/W, and Boolean
  `ceilMode`. `WINDOW_3D` has the analogous thirteen D/H/W words. There is no
  `countIncludePad` wire: average pooling always counts every kernel position.
- `ATTENTION` is `[scalePresent, causal]` when scale is absent, or
  `[1, scalarType, scalarBits, causal]` when present. The scalar carrier is FLOAT64, FLOAT32, or
  BFLOAT16; Model additionally requires exact promoted-type equality and positive finite value.
  The optional mask is the fourth BOOL tensor input, not an attribute.

Type wires remain exactly `1..6`: FLOAT32 `1`, INT32 `2`, BOOL `3`, FLOAT64 `4`, BFLOAT16 `5`, and
INT64 `6`. Only floating carriers are Model-legal for the seven family rows.

No Model operation kind, registry row, attribute, Tensor overload, Compiler inference row, or native
wire named `CONV_TRANSPOSE` exists. `ConvolutionGradientRules` deliberately reconstructs Conv2d
input cotangents with `UNFOLD2D`/`MATMUL`/`FOLD2D` and states that it introduces no transposed-
convolution or backward-only kind. Task 0064 must not reserve a wire, alias `CONV2D`, or invent a
backend-private public semantic. A future transposed-convolution operation requires a Model task,
shape/type semantics, Compiler adoption, a new registry/schema decision, and its own reviewed Metal
plan.

Related existing rows are dependencies or callers, not new family kinds:

- wires 6..11 provide shape/layout compositions; public Conv1d is
  `EXPAND_DIMS(input) + EXPAND_DIMS(weight) + CONV2D + SQUEEZE`, and public max/average Pool1d is
  `EXPAND_DIMS + Pool2d + SQUEEZE`;
- wires 13, 15, 17, 20..25, 39..45, 46..52, 73, 81..84, 109, and shape operations occur in
  Compiler-generated Conv2d/pooling/attention formulas; their current capability domains remain
  independently enforced;
- wire 101 `DROPOUT` is a separate explicit-state two-input/three-output random operation with
  attribute wire 33. Attention has no dropout field or implicit RNG state, so fused attention
  dropout is outside the current Model contract.

### Exact convolution production domain

Admit `CONV2D` and `CONV3D` only under `ACCELERATOR` when all of the following hold:

- input/weight rank is exactly four for NCHW/OIHW or five for NCDHW/OIDHW; optional bias rank is
  one; output rank matches the input; every dimension is fully static and positive;
- input, weight, and optional bias are each FLOAT32 or BFLOAT16, exact ordered promotion produces
  FLOAT32, and the output is FLOAT32. At least one participating role is therefore FLOAT32.
  All-FLOAT32 occurrences may retain Model's exact input-OR/output gradient metadata. If any role is
  BFLOAT16, every input and output is no-grad, matching the current proved mixed-carrier custom
  boundary. FLOAT64 results and all-BFLOAT16/BFLOAT16 results remain false;
- input channels and output channels are divisible by positive groups, weight channel extent one is
  exactly `Cin/groups`, bias length is exactly `Cout`, and groups do not exceed either channel
  count;
- for each spatial axis `a`, effective kernel is
  `Ea = dilation_a * (kernel_a - 1) + 1`, padded input is `Pa = input_a + 2 * padding_a`,
  `Pa >= Ea`, and output is exactly `floor((Pa - Ea) / stride_a) + 1`;
- every output cell uses the exact contiguous channel group and every channel/kernel position once.
  Conceptual positive-zero padding is an ordinary multiplicand, so `+0 * +/-infinity` produces NaN
  rather than being skipped. The optional bias is the initial promoted accumulator; an unbiased
  cell begins at positive zero;
- ordinary Conv2d/Conv3d roles are canonical dense, zero-offset, non-view storage. Conv2d input and
  weight may additionally be the exact direct local `EXPAND_DIMS(axis=2)` singleton-height affine
  views of canonical rank-three NCW/OIW sources used by public Conv1d. Complete-partition analysis,
  Java image validation, and native validation must authenticate the producer, singleton extent,
  offset, positive strides, span, and physical source; no other affine form is admitted. Bias and
  every convolution output remain canonical; and
- each output is fresh and physically disjoint from every input and sibling live value. Repeated
  input identity and arbitrary fan-out are legal, but in-place or overlapping publication is not.

This is a recursive-result-set domain, not a tolerance. The fixed source loop may use deterministic
FLOAT32 rounding, DAZ/FTZ, any compiler-selected binary association, and corresponding multiply/add
fusion because current Model explicitly permits those at `ACCELERATOR FLOAT32` convolution sites.
It may not drop, duplicate, pretruncate, invent, or replace a term; alter group membership; change
placement; or use MATMUL's separate final-zero freedom. Strict FLOAT32, every FLOAT64 result, and
every BFLOAT16 result remain blocked because ordinary Metal floating arithmetic cannot prove their
strict subnormal/rounding contract and no software IEEE implementation exists here.

### Exact pooling production domains

For all four pooling kinds, require rank-four NCHW or rank-five NCDHW, same input/output type, exact
batch/channel identity, fully static positive Shapes, exact derived output Shape, fresh canonical
output, and the same checked layout/alias rules as convolution. Pool2d additionally accepts only the
authenticated direct local singleton-height `EXPAND_DIMS(axis=2)` view used by public Pool1d; Pool3d
accepts canonical input only.

For each spatial axis, define effective kernel and padded input as above. Output is exactly

`(ceilMode ? ceil((Pa - Ea) / stride_a) : floor((Pa - Ea) / stride_a)) + 1`.

Literal ceiling mode retains a terminal all-padding window; no framework-style terminal-window trim
is permitted.

`MAX_POOL2D` and `MAX_POOL3D` admit FLOAT64, FLOAT32, and BFLOAT16 under both profiles, including
legal gradient metadata. Padding is excluded from selection. Logical traversal is H then W or D
then H then W. NaN dominates every non-NaN, positive zero is greater than negative zero, equal
represented values retain the first logical candidate, and the selected original raw carrier is
published. An all-padding window returns exact negative infinity in the input type. These are exact
discrete semantics with no profile relaxation.

`AVERAGE_POOL2D` and `AVERAGE_POOL3D` admit only FLOAT32 under `ACCELERATOR`, with matching input/
output gradient metadata. Every logical kernel position participates, padding contributes positive
zero, and the divisor is always the exact kernel-position product converted to FLOAT32 for the one
final division. A fixed FLOAT32 reduction may reassociate and use per-step rounding plus DAZ/FTZ,
but may not omit a padding or input contributor. NaN propagates, opposing infinities produce NaN,
one infinity retains its sign, all-padding returns positive zero, and an exact-zero result is
negative only when every divisor position is an in-bounds negative zero. Strict FLOAT32, FLOAT64,
and BFLOAT16 average pooling remain false without exact non-flushing/software arithmetic.

### Static positive and unsigned-32-bit boundary

Before capability returns true, checked `long`/`uint64_t` arithmetic must prove all of these facts:

- every dimension, stride, dilation, padding, group count, channel-per-group count, effective kernel,
  padded extent, output extent, kernel volume, convolution contributor count, element count,
  canonical stride, direct-view stride, logical coordinate, linear element index, custom thread
  count, and one-dimensional grid width is representable without narrowing or signed overflow;
- positive dimensions/counts/extents/attributes are in `1..UINT32_MAX`, padding and derived
  zero-based coordinates/indices are in `0..UINT32_MAX`, and every access index is strictly below
  its governing count/extent. Signed window origins may be negative but must fit checked signed
  64-bit arithmetic before an in-bounds decision;
- each input/output element count and the one-thread-per-output grid width are in
  `1..UINT32_MAX`; every loop is half-open and never requires `limit + 1`, so exact
  `UINT32_MAX` limits terminate without increment wrap; and
- byte widths, byte offsets/spans, metadata sizes, products, `NSUInteger` conversions, buffer
  lengths, and allocations separately pass existing native-size bounds. A 64-bit metadata or byte
  carrier never widens the logical unsigned-32-bit domain.

Java capability/preparation and native image validation enforce the same facts before pipeline,
buffer, or executable creation and before any device write. Dynamic/expression, zero extent, scalar,
over-limit, general affine, negative/zero-stride, overlapping, aliased, unresolved, or malformed
occurrences fail closed without MPSGraph or host fallback.

### Fixed custom route and native algorithms

Add one Task-0064 custom Metal source/header and append it unconditionally to the existing fixed
custom-program library. Add native operation constants only for wires 36, 37, and 97..100. Reuse the
Task-0061 raw BFLOAT16-to-FLOAT32 widening convention and Task-0063 raw floating classification/key
convention rather than native BFLOAT16 or FLOAT64 arithmetic.

Every admitted node dispatches one one-dimensional thread per output element:

- Conv2d/Conv3d decode the output's N/C/spatial coordinate, derive the contiguous group, initialize
  from promoted bias or positive zero, then visit input-channel and kernel coordinates in fixed
  lexicographic order. Each in-bounds input is loaded as FLOAT32; BFLOAT16 is widened exactly from
  bits. Each out-of-bounds input is literal positive zero but is still multiplied by the promoted
  weight. The source expresses every multiply/add contributor exactly once; any emitted FMA,
  association, DAZ, or FTZ remains inside the current accelerator recursive result set.
- Maximum pooling visits only in-bounds candidates, classifies and compares raw FLOAT64/FLOAT32/
  BFLOAT16 encodings using integer exponent/fraction/sign keys, replaces on NaN preference or a
  strictly greater numerical value, and never replaces on an exact tie. It copies the winning raw
  word. With no candidate it writes canonical negative infinity in the carrier.
- Average pooling initializes positive-zero FLOAT32, visits every kernel position, selects either
  the input value or literal positive zero, and performs one fixed accumulation plus one division
  by the FLOAT32 kernel volume. It separately tracks whether every position is in-bounds raw
  negative zero and applies the exact final zero-sign rule. No scratch, atomic, reduction pass,
  host calculation, or data-dependent route exists.

Use the schema's authenticated layout strides only for canonical storage and the exact Conv1d/
Pool1d singleton view above. Kernel metadata must not trust Java-derived output dimensions,
groups, channel relations, view geometry, or products: native preflight independently recomputes and
compares all of them. Unique output-thread ownership proves race-free writes. Byte addressing uses
checked wider arithmetic after the logical unsigned-32-bit proof.

Any partition containing one of the six admitted rows selects `CUSTOM_PROGRAM` only. The custom
program may compose these kernels with already approved custom steps and nested MPSGraph steps in
one fixed topological program, but candidate generation never offers MPSGraph for a family row.
There is one deterministic result set per valid partition; route choice never consults timing,
cache winners, input values, runtime failure, or retry.

### Opaque MPSGraph routes are not numerical proof

Keep every existing MPSGraph catalog state and reason unchanged as structural metadata:

- Installed Conv2d/Conv3d selectors document basic NCHW/NCDHW descriptor mapping but not the
  shape-dependent contraction tree, per-step rounding, FMA placement, DAZ/FTZ behavior, padding-
  times-infinity participation, bias placement, or the complete admitted geometry. Historical
  [Task 0031](0031-accelerator-canonical-float32-unbiased-conv2d-forward.md) correctly rejected
  inference from one geometry. Optional bias is also outside its audited direct mapping.
- The direct Pool2d maximum selector failed historical
  [Task 0030](0030-profile-common-exact-canonical-float32-max-pool2d-forward.md): it returned
  negative zero for `[+0,-0]` and finite `3.0` for every NaN pair. That is a Model-visible exact
  winner failure. The average selector exposes include-zero-pad and ceil fields but gives no
  complete reduction/special-class/zero-sign proof. Neither direct row is selectable here.
- No installed true Pool3d selector closes the NCDHW contract. The existing
  `COMPOSED / MC_WINDOW3D` catalog entry records only a structural slice/reshape/concat/reduction
  recipe. Historical [Task 0036](0036-extended-3d-inference.md) remains correct that Pooling4D or
  stencil names do not prove literal ceil-tail, fixed divisor, padding, or exact winner semantics.
- The base attention selector mapped by the catalog takes query/key/value and scale and returns only
  the main output. It does not close Model's BOOL mask, top-left causal guard, saved weights output,
  special classes, default-scale construction, or exact output roles, and its contraction/softmax
  implementation is opaque.

The custom source is instead structurally auditable over every admitted type, Shape, and attribute.
A bounded real-device corpus corroborates the implementation but never authorizes unsampled domain.
No selector probe or benchmark is required before the reviewed source-proved cutover.

### Attention audit and explicit blocker

Current Model attention accepts rank-two-or-higher floating query `[...,L,E]`, key `[...,S,E]`, and
value `[...,S,Ev]`, with exact right-aligned three-way batch broadcasting. Optional BOOL mask must
broadcast to scores `[...,L,S]`; causal eligibility additionally requires key index `j <= i`.
Output slot zero is `[...,L,Ev]`; optional canonical saved normalized weights at slot one are
`[...,L,S]`. Query/key/value promotion selects both floating outputs. Explicit scale must have that
promoted type and positive finite bits; absent scale is `1 / sqrt(E)`.

The exact formula guards exclusion before arithmetic. No eligible score and all eligible negative
infinities produce positive-zero outputs; eligible NaN propagates; positive-infinity ties split unit
weight equally; and excluded exceptional score/value operands never enter arithmetic. Under current
Model, `ACCELERATOR FLOAT32` does have recursive freedoms for score and value contractions, scale,
default-scale conversion/sqrt/division, and guarded exp/sum/division sites. Historical
[Task 0033](0033-profile-common-canonical-float32-no-grad-unmasked-noncausal-default-scale-sdpa-forward.md)
predates part of that current profile wording, so its old broad “unrelaxed” premise is not reused.
Its selector-structure and opaque-complete-domain objections remain valid.

Attention nevertheless remains blocked because:

1. no current custom EXP/SQRT/softmax core has a complete recursive-result-set proof. In particular,
   [Task 0053](0053-certified-accelerator-float32-custom-exp-stable-sigmoid.md) remains blocked on
   the constructive-real exponential bridge; an MSL builtin or opaque MPSGraph softmax cannot
   substitute for that proof;
2. the direct selector cannot produce Model's optional mask/causal semantics and exact saved weights
   role, and schema-valid two-output infrastructure does not manufacture missing numerical output;
3. one-output no-grad, explicit-scale, unmasked, or noncausal subsets still require the same guarded
   exponential/normalization and special-class proof. Capability cannot inspect runtime values to
   admit only a convenient finite corpus; and
4. Compiler gradients require the exact same-occurrence output slot one. Their formulas use saved
   weights, selection-safe products, comparisons/WHERE, reductions, contractions, configured or
   default scale, and normalization back to each role. No partial forward route may claim this
   saved-state or gradient contract.

Wire 35 therefore remains non-executable, production false, and custom pending. Masked, causal,
explicit/default scale, one/two output, all types/profiles, and all gradients fail before resource
creation. Dropout remains the separate wire-101 contract; there is no attention-dropout value to
ignore or default.

### Compiler gradients and public callers

Model and Compiler sources remain unchanged. Audit `StructuredOperationInference`,
`ConvolutionLogicalLayoutClosure`, `AutogradPreflight`, `ConvolutionGradientRules`,
`PoolingGradientRules`, `AttentionGradientRules`, `FirstOrderGradientCoverage`, public Tensor
methods/results, and the NN convolution callers.

Forward capability and complete derivative-partition capability are distinct:

- Conv2d gradients use group-preserving reshape/permute, `UNFOLD2D`, FLOAT32 `MATMUL`, reductions,
  and overlap-accumulating `FOLD2D`; bias uses reduction. Prove CPU-free accelerator checkpoints for
  each separately selected all-FLOAT32 input, weight, and bias cotangent where the other source
  roles are no-grad and any input-gradient fold has non-overlapping windows. These are the maximal
  current primitive-closed cases by source inspection. Joint gradient-bearing source roles,
  overlapping folds, and BFLOAT16 normalization remain explicit negative cases unless a separate
  reviewed primitive-domain plan closes them.
- Conv3d remains forward-only because `AutogradPreflight` rejects every Conv3d occurrence before
  derivative allocation until Compiler task 0006C. Task 0064 must preserve and publicly prove that
  rejection while admitting the forward occurrence.
- Average Pool2d/Pool3d gradients divide the output cotangent by the fixed kernel-position divisor,
  expand columns, and fold. Prove accelerator FLOAT32 backward only when the current fold domain is
  statically non-overlapping. Overlap accumulation remains outside Task 0060's exact-copy fold
  capability.
- Maximum-pool gradients read the gradient-bearing original input, reconstruct in-bounds exact
  winners with padded unfold, classification, comparisons, reciprocal signed-zero tests, Boolean
  logic, `ARG_MAX(FIRST_INDEX)`, one-hot, WHERE, and fold. Current Metal read-only unfold,
  comparison, reciprocal, and overlapping-fold domains do not collectively accept that generated
  graph. Preserve forward gradient metadata but require generated max-pool backward to reject before
  preparation under both profiles and every carrier; do not widen unrelated primitive rows here.
- Attention gradients remain blocked for the saved-weights and numerical reasons above.

Public Engine proof must use only the packaged Metal provider, with no CPU provider or skip:

- accelerator biased/unbiased grouped Conv2d/Conv3d covering stride, dilation, symmetric padding,
  groups, positive-zero-padding times infinity, FLOAT32 and no-grad mixed BFLOAT16/FLOAT32 inputs,
  retained reuse, and independent sessions;
- Conv1d through its exact expand/Conv2d/squeeze composition, including bias and groups;
- maximum Pool1d/2d/3d under both profiles and all three carriers, including signed zeros, multiple
  NaNs/raw payloads, ties, infinities, excluded padding, dilation, ceil terminal all-padding window,
  and exact selected bits;
- average Pool1d/2d/3d accelerator FLOAT32 with fixed divisor, padding inclusion, ceil/floor,
  subnormals, signed zeros, NaN, same/opposing infinities, all-padding, reuse, and sessions;
- the provable Conv2d and average-pool gradient checkpoints above plus explicit early failures for
  Conv3d, max-pool, attention, overlap, mixed-gradient, strict convolution/average, malformed, and
  one-past-limit graphs.

### Catalog, schema, identity, and lifecycle

- Mark only wires 36, 37, and 97..100 structurally executable: `93/22 -> 99/16`. Production
  capability becomes `75/40 -> 81/34`. Wire 35 remains false in both counts.
- Keep the MPSGraph catalog exactly `75 DIRECT / 35 COMPOSED / 5 UNAVAILABLE`. Conv and Pool2d
  retain their direct structural reasons; Pool3d retains its composed reason; attention retains its
  direct-base reason. Catalog structure is not selector authorization.
- Add `CA_0064`, change exactly the six convolution/pooling custom rows from pending to available,
  and produce custom catalog `58 AVAILABLE / 57 PENDING / 0 UNAVAILABLE-WITH-PROOF`. Attention
  remains `PENDING / CP_CONTRACT`.
- Keep schema 15, ABI 5, type wires `1..6`, operation wires `1..115`, attribute wires `0..41`, route
  wires `1..3`, image grammar, generic multi-output records, and thirteen exports. Advance workload-
  signature, exact-default-policy, candidate-schema, compatibility-schema, route-policy, and codec
  identities together `19 -> 20`; identity 19 rejects with no alias, migration reader, or shim.
- Preserve cold deterministic route selection, stable value/feed/target order, transactional
  finalization, synchronous invocation, direct/internal output liveness, physical-span
  authentication, leases, rollback and suppression, reuse, concurrency/session isolation, close
  ordering, trace identity, and the profile-free Runtime/Trace boundary.

## Provable domains versus explicit blockers

The provable cutover is exactly:

- accelerator FLOAT32-result Conv2d/Conv3d over FLOAT32/BFLOAT16 roles, with all-FLOAT32 gradient
  metadata or no-grad mixed roles, exact canonical/singleton-composition layouts, and the complete
  checked unsigned-32-bit domain;
- both-profile exact max Pool2d/Pool3d over FLOAT64/FLOAT32/BFLOAT16 with legal forward gradient
  metadata and the same domain; and
- accelerator FLOAT32 average Pool2d/Pool3d with legal gradient metadata and the same domain.

Remain explicitly blocked:

- all attention forms and gradients; every convolution transpose form; fused attention dropout;
- strict convolution/average FLOAT32, convolution FLOAT64 or BFLOAT16 result, average FLOAT64 or
  BFLOAT16, mixed convolution gradients, Conv3d gradients, maximum-pool gradients, and any
  overlap-accumulating generated fold;
- count-exclude-pad average, asymmetric padding, alternate NHWC/NDHWC or weight layouts, output
  padding, transposed kernels, dynamic/symbolic/empty/scalar/general-view/in-place/aliased geometry,
  and any dimension/count/coordinate/index/grid above the declared unsigned-32-bit boundary;
- direct MPSGraph family execution, host calculation/validation/repair, sampled-value authorization,
  retry, fallback, timing, autotuning, cache winner, performance claim, or a Task-0053 workaround.

If implementation needs another carrier, profile, layout, gradient primitive widening, numerical
freedom, schema field, ABI export, 64-bit/multidimensional dispatch, scratch/atomic route, or runtime
choice, stop for a new independently reviewed plan rather than narrowing or silently changing this
one.

## Non-goals

No Model, Compiler, public Tensor, NN, Config, Planning, Prepare, Runtime, Trace, or Engine API
change; no operation/attribute/type wire; no ABI export; no general affine layout; no general
multi-output widening; no dynamic or empty kernel; no benchmark; no selector rehabilitation; and no
rewrite of historical blocked evidence. Task 0064 does not complete attention, Conv3d backward,
maximum-pool backward, overlap folds, or `CONV_TRANSPOSE`.

## Contracts

- [`ARCHITECTURE.md` — Core lifecycle and Core invariants](../../../../../ARCHITECTURE.md#core-lifecycle)
  — Model owns exact profile-indexed meaning; route choice is cold and Runtime/Trace stay
  profile-free.
- [Foundational modules — Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)
  — contraction/reduction recursive result sets and exact discrete winner/guard semantics.
- [Backend execution — Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)
  and [Numerical-profile backend identity](../../../../architecture/contracts/backend-execution.md#numerical-profile-backend-identity)
  — truthful occurrence capability, fixed routes, native state, and private identities.
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)
  and [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle).
- Model [Conv2d](../../../modules/model/tasks/0020-nchw-conv2d-semantics-and-tensor-expressions.md),
  [Conv1d](../../../modules/model/tasks/0025g-ncw-conv1d-composition.md),
  [Conv3d](../../../modules/model/tasks/0025h-ncdhw-conv3d-semantics-and-tensor-expressions.md),
  [max Pool2d](../../../modules/model/tasks/0020a-nchw-max-pool2d-semantics-and-tensor-expression.md),
  [average Pool2d](../../../modules/model/tasks/0020a1-nchw-average-pool2d-semantics-and-tensor-expression.md),
  [Pool1d](../../../modules/model/tasks/0025i-ncw-max-average-pool1d-composition.md),
  [Pool3d](../../../modules/model/tasks/0025j-first-class-ncdhw-max-average-pool3d-semantics.md), and
  [attention](../../../modules/model/tasks/0019e-scaled-dot-product-attention.md) plus
  [weights output](../../../modules/model/tasks/0023f-scaled-dot-product-attention-weights-output.md)
  own operation semantics.
- Compiler [structured gradients](../../../modules/compiler/tasks/0005d-attention-convolution-pooling-and-loss-gradient-completion.md),
  [Conv3d boundary](../../../modules/compiler/tasks/0006b-conv3d-forward-adoption-and-explicit-gradient-boundary.md),
  and [Pool3d gradients](../../../modules/compiler/tasks/0006b2-pool3d-and-3d-window-gradient-closure.md)
  own formula/rejection topology. CPU implementations are exact oracles/precedents, never Metal
  authorization.

## Dependencies and integration

- Depends on: Task 0063 Complete through `c80d79c0`; current Model/Compiler contracts named above;
  Tasks 0030/0031/0033/0036 retained blocker evidence; Task 0053's current EXP proof blocker; and
  completed six-carrier, BFLOAT16 widening, schema-15, custom-program, non-overlap fold, general
  MATMUL, exact ordering, lifecycle, package, and public Engine foundations.
- Supersedes only the future-route conclusions of historical Tasks 0030, 0031, and 0036 by choosing
  a separately proved custom route. Those direct/composed MPSGraph tasks remain Blocked and their
  evidence remains immutable. Task 0033 and Task 0053 remain Blocked.
- Conflicts with: every concurrent Metal capability/schema/native/custom-source/catalog/candidate/
  route/preparation/package/shared-document scope and resumed Task-0053 production.
- Parallel group: None.
- Common base revision: `bc8fb4a5baa188af86092b06a96bc4134a8a4eed`.
- Integration order: this independent plan review, then one serial proof/kernel/capability/
  preparation/execution/test/documentation cutover; no production edit before plan approval.
- Integration validation: native build/sign/package/verification, complete Metal, conformance,
  source-backed Model/Compiler contract tests, CPU-free public Engine forward/backward/negative
  evidence, Javadoc/architecture/docs, exact count/identity/export audits, and independent
  cumulative Class C review.
- Shared-document integration owner: Task-0064 coordinator.

## Files and symbols

Expected implementation owners:

- `MetalCapabilityProvider`, `MetalMpsGraphProgram`, `MetalNegPartitionPreparer`,
  `MetalNegPreparationPlan`, `MetalNativeApi`, `MetalOperationRouteCatalog`, custom route candidate/
  tuning/codec identity owners, and affected package/type Javadocs;
- native foundation operation/preflight/metadata/pipeline/step/binding code plus a new
  `synaptik_task0064_convolution_pooling_kernels.h`, with no new export or schema field;
- focused capability/catalog/schema/malformed/prepared/native/raw-word/result-set tests, including
  allocation-free exact-`UINT32_MAX` and one-past/product/stride/span rejection parity;
- existing Compiler convolution/pooling/attention contract suites, Metal conformance, and a
  dedicated CPU-free Engine Metal integration for direct 1D/2D/3D forward, bounded generated
  gradients, exclusions, reuse, and sessions; and
- Metal/backend/native guides, public API scope/capability/preparer status, targeted glossary and
  architecture identity claims, this brief, master plan, and roadmap after behavior stabilizes.

The planning-only revision changes exactly this brief, the Metal master plan, and the roadmap.

## Acceptance criteria

1. The seven-row audit and tests retain exact wires, cardinalities, attributes, carriers, profiles,
   layouts, gradient metadata, output roles/Shapes, groups, geometry, special classes, UINT32/native-
   size bounds, and blocker decisions. `CONV_TRANSPOSE` remains absent rather than aliased.
2. Exactly wires 36, 37, and 97..100 become executable/custom-available and occurrence-capable only
   in the domains above. Wire 35 and every excluded occurrence reject before resource creation.
3. Conv kernels visit every group/channel/kernel term once, including padding multiplication;
   output is inside Model's recursive accelerator set. Maximum pools implement exact raw winner
   semantics and average pools implement every-contributor fixed-divisor semantics and exact
   special-class/zero-sign rules.
4. Conv1d and Pool1d authenticate only their exact local singleton-height view composition. General
   affine/view/alias paths remain closed. Outputs have unique writers and no input is mutated.
5. Every admitted partition has only `CUSTOM_PROGRAM`; MPSGraph family selectors, host repair,
   retry, fallback, timing, autotuning, value-based decisions, and cache winners are absent.
6. Counts become `81/34` capability, `99/16` structural, `75/35/5` MPSGraph, and `58/57/0` custom;
   schema 15, ABI 5, and thirteen exports remain; all backend-local identities are 20 and identity
   19 rejects.
7. CPU-free packaged-Metal Engine proves direct and composed forward domains, raw special values,
   grouping/geometry/bias/promotion, both relevant profiles, reuse/sessions, the provable Conv2d and
   average-pool gradients, and every named early-failure boundary without skip or CPU fallback.
8. Attention one/two-output, mask/causal/scale, gradients, dropout claim, Conv3d/max-pool blocked
   gradients, convolution transpose, strict/missing carriers, overlap, malformed, and one-past
   domains stay visibly fail-closed.
9. Native/package, complete Metal, conformance, dedicated Engine, Javadoc, architecture, Markdown/
   link/diff checks pass; independent cumulative Class C review has no unresolved P0/P1/P2.

## Validation after implementation

```bash
./native/metal-macos-arm64/build.sh
codesign --force --sign - --identifier io.github.pho001.synaptik.metal.foundation native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
./native/metal-macos-arm64/package-local.sh native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
./native/metal-macos-arm64/verify-package.sh native/metal-macos-arm64/build/package-v1/macos-arm64
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/package-v1/macos-arm64/libsynaptik_metal_foundation.dylib" ./gradlew :backends:metal:test
./gradlew :modules:compiler:test --tests '*Convolution*' --tests '*Pool*' --tests '*Attention*'
./gradlew :testing:backend-conformance:test --tests '*Metal*'
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/package-v1/macos-arm64/libsynaptik_metal_foundation.dylib" ./gradlew :testing:integration-tests:test --tests '*ConvolutionPoolingMetalIntegrationTest*'
./gradlew :backends:metal:javadoc :testing:architecture-tests:test
git diff --check
```

No timing or benchmark command is authorized. No full repository build is planned absent an
unexpected shared API/dependency/build change; the complete packaged Metal suite, focused Compiler
contracts, conformance, public Engine, and architecture gates are stronger for this backend-local
cutover.

## Documentation and review impact

Implementation must update every current capability/count/identity/convolution/pooling/attention
statement and describe only the exact bounded domains above. Public/backend/native/Javadoc wording
must name profile/type/layout/gradient boundaries, literal ceil and fixed-divisor behavior, raw max
winner rules, pre-resource failure, direct-selector exclusion, attention/ConvTranspose blockers,
and unchanged ABI/schema/exports. Preserve historical tasks as evidence records.

Independent plan review precedes every production edit. Independent final review must inspect the
full plan-to-source proof, term/contributor preservation, padding-times-infinity, raw comparator,
zero-sign logic, coordinate and byte arithmetic, exact-limit termination, singleton-view
provenance, Java/native preflight parity, route exclusivity, identity invalidation, generated-
gradient boundaries, public Engine behavior, documentation, and changed-path scope. Passing worker
evidence may be reused unless remediation changes executable behavior.

## Planning result

The audited maximal current cutover is six fixed custom convolution/pooling rows, not attention.
Attention remains blocked on the unproved exponential/softmax numerical core and incomplete opaque
selector/output contract; convolution transpose remains blocked before Metal because no Model or
registry contract exists. Current production remains unchanged at `75/40` capability, `93/22`
structural execution, catalogs `75/35/5` and `52/63/0`, schema 15, identity 19, ABI 5, and thirteen
exports until independent plan review approves and the implementation/evidence cutover completes.
