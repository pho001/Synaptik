# Task 0027: Profile-Common FLOAT32 Batch-Normalization Inference

## Status

Ready

## Change class

Class C — this task adds a five-input floating numerical operation to both Metal profile matrices,
evolves the backend-private native schema and route identities, and changes the generated native
graph. Use one clean implementation context and an independent lean Class C review.

## Goal

Implement only first-class `BatchNormKind.BATCH_NORM_INFERENCE` for canonical, positive, fully
static `FLOAT32` tensors under both numerical profiles. Preserve Model's arbitrary normalized
channel axis, five ordered inputs, one output, exact formula, and special-value classes.

```text
[input, scale, bias, runningMean, runningVariance]
  -> reshape each [C] vector locally to [1,...,C,...,1]
  -> direct MPSGraph normalization
  -> one fresh canonical output
```

This is forward execution only. It adds no batch-normalization training, multi-output Metal node,
backend gradient operation, or all-Metal backward claim.

## Dependencies and integration

- Depends on: Metal 0025 Complete; Model 0021B; Compiler 0005B/0005C; CPU 0007F1 only as an
  independent numerical-policy precedent; Config 0006; Engine 0018; current completed Metal
  lifecycle and profile contracts. Metal 0026 remains Blocked and is not a functional dependency.
- Conflicts with: every concurrent Metal capability, node-schema, native-preflight,
  candidate/codec, materialization, or public Engine scope; any Metal 0016–0018 restart; another
  normalization, training, multi-output, or shared semantic task.
- Parallel group: None.
- Common base revision: `33df376e3bb6a79b8d7714f63eb151ab512feffd`.
- Integration order: mandatory one-run direct-selector gate; schema and identity; capability and
  Java/native preflight; native lowering; focused execution proof; current docs; independent review.
- Integration validation: focused Metal/conformance/real-Engine proof, native export audit,
  architecture checks, Javadoc, one final build, Markdown and diff validation.
- Shared-document integration owner: Metal 0027 implementer.

The dependencies are Complete and the exact common base contains the finalized Metal 0026 blocker.
Task 0027 is the sole authorized Metal frontier.

## Exact capability and profile contract

Admit one occurrence exactly when:

- kind and attributes are exactly `BATCH_NORM_INFERENCE + BatchNormInferenceAttrs`;
- ordered inputs are `[input, scale, bias, runningMean, runningVariance]` and there is one output;
- every descriptor is `FLOAT32`, fully static, canonical dense contiguous, zero-offset, non-view,
  rank `2..16` for input/output or rank one for each channel vector, with positive extents;
- the normalized `channelAxis` is in `[0, inputRank)` and every vector Shape is exactly `[C]`, where
  `C` is the input extent at that axis;
- output Shape equals input Shape, output type is `FLOAT32`, and output `requiresGrad` is the logical
  OR of all five input flags;
- epsilon is exact finite positive `FLOAT32`; and
- checked element/byte counts, dimension cells, and native integer conversions are representable.

The four vector roles may repeat the same `ValueId`. Deduplicate boundary feeds in first-occurrence
order, preserve repeated role indices in the node, mutate no input, and reject every output/input
storage overlap before execution. Reject training, wrong arity/attrs/type/Shape/layout/gradient
state, scalar/dynamic/zero-extent/rank-above-16/view forms, malformed aliases, and overflow before
native graph creation.

Batch normalization is unlisted in the architecture's accelerator-relaxation table. The same exact
domain and result policy therefore apply under `STRICT_IEEE` and `ACCELERATOR`; adding the row to
both matrices preserves strict-subset-of-accelerator capability monotonicity.

## Documented structural basis

The installed `MPSGraphNormalizationOps.h` documents
`normalizationWithTensor:meanTensor:varianceTensor:gammaTensor:betaTensor:epsilon:name:` with
explicit tensor, mean, variance, gamma, beta, and `float` epsilon parameters. It documents gamma
and beta broadcasting and gives full-rank singleton-expanded channel tensors as the batch-normalization
shape. Existing MPSGraph reshape support documents the local `[C]` to `[1,...,C,...,1]` adapters.

For input rank `R` and channel axis `a`, construct one rank-`R` vector Shape containing `C` at `a`
and one elsewhere. Reshape scale, bias, running mean, and running variance to that Shape, then call
the direct selector with the original input. These graph-local tensors are not semantic nodes,
public values, materializations, workspace, or targets. Java/native structural tests, not a device
matrix, prove first/middle/last axes and ranks `2..16` use this same construction.

The header does not document finite error, subnormal preservation, NaN/infinity behavior, signed
zero, overflow, or whether a supplied statistic tensor is consumed without an undocumented
numerical rewrite. Structural selector validity does not authorize those results.

## Model formula and numerical contract

For each coordinate with channel `c`, preserve exactly:

```text
centered     = input - runningMean[c]
denominator  = sqrt(runningVariance[c] + epsilon)
standardized = centered / denominator
output       = standardized * scale[c] + bias[c]
```

Epsilon is inside the square root; running variance is used directly; affine order is multiply then
add. Computation is `FLOAT32`. Finite evidence may use the CPU precedent's two-result-ULP oracle
ceiling for the stated real formula, but that ceiling is test evidence, not a new allowed-result set
or accelerator relaxation. It cannot admit denormals-are-zero, flush-to-zero, class changes, wrong
infinity/zero signs, reciprocal substitution, or another algebraic formula.

Preserve Model's exact classes: a negative radicand produces NaN; a zero radicand produces a zero
denominator, hence centered zero gives NaN and centered nonzero gives signed infinity; positive
infinite variance gives a positive-infinite denominator, finite centered values give signed zero,
and infinite centered values give NaN. Participating NaN remains NaN even through zero affine
scale. Infinite scale times standardized zero gives NaN. Ordinary signed-zero, overflow, underflow,
and final affine behavior remain observable. NaN payload and sign are not promised.

## Exactly one pre-production numerical gate

Before any production edit, run one disposable Objective-C program exactly once: one process,
Metal device/queue, context, graph, executable, direct selector target, and synchronous execution.
Use production settings: default optimization level, set/read
`MPSGraphReducedPrecisionFastMathNone` where available, and `MPSGraphOptionsNone`. Use canonical
input Shape `[2,8,2]`, channel axis `1`, four `[8]` vectors locally reshaped to `[1,8,1]`, and exact
epsilon `0.25f`.

Each channel has four independent cells because the two non-channel extents contribute
`2 * 2 = 4` coordinates. This makes the corpus internally satisfiable without another Shape:

| `c` | `(mean, variance, scale, bias)` | Four input cells | Contract defended |
|---:|---|---|---|
| 0 | `(1, 3.75, 2, 0.5)` | `1, 2, -1, qNaN` | supplied statistics, epsilon placement, affine order, ordinary finite values, input NaN |
| 1 | `(0, -0.5, +0, 7)` | `0, 1, -1, 2` | negative radicand and no suppression of its NaN by zero scale |
| 2 | `(0, -0.25, 1, +0)` | `+0, 1, -1, -0` | zero denominator: NaN, positive infinity, negative infinity, NaN |
| 3 | `(0, +infinity, 1, -0)` | `1, -1, +infinity, -infinity` | positive-infinite variance: positive zero, negative zero, NaN, NaN |
| 4 | `(qNaN, 0.75, +0, 7)` | `0, 1, -1, +infinity` | participating statistic NaN survives zero scale and finite bias |
| 5 | `(0, 0.75, +infinity, +0)` | `+0, -0, 1, -1` | infinite scale: NaN, NaN, positive infinity, negative infinity |
| 6 | `(0, 0.75, minSubnormal, -0)` | `1, -1, +0, -0` | exact signed minimum subnormals and positive/negative zero; no FTZ |
| 7 | `(0, 0.75, maxFinite, +0)` | `2, -2, +0, qNaN` | finite overflow signs, positive zero, and input-NaN propagation |

All eight rows defend distinct Model obligations; none is a control-only or padded channel. Compare
ordinary finite cells with an independent binary64 real-formula reference narrowed once to
`FLOAT32`, allowing at most two result ULPs. Require exact NaN class, infinity sign, zero raw sign,
and channel-6 minimum-subnormal raw bits; require exact output Shape and unchanged raw bits for all
five feeds. Do not accept zero as one ULP from a minimum subnormal.

Do not run an axis, rank, profile, optimization, context, selector, or repetition matrix. If any
cell fails, remove source/binary, mark 0027 `Blocked`, retain only exact evidence, and make no
production/schema/capability change. Do not narrow the value domain, weaken a class/sign rule, move
normalization to accelerator-only, decompose it, or add fallback. A pass authorizes only this direct
composition and proceeds to ordinary implementation verification.

## Typed schema and identity cutover

- Retain application binary interface (ABI) version 4, all statuses/signatures, the fixed 160-byte
  record, and exactly thirteen exports.
- Evolve node schema `11 -> 12`; retain wires `1..19`; append
  `BATCH_NORM_INFERENCE=20` and typed attribute discriminator `BATCH_NORM_INFERENCE=7`.
- Encode `first_input=input`, `second_input=scale`, `auxiliary=bias`, `axis=channelAxis`,
  `attribute_values[0]=runningMean index`, `[1]=runningVariance index`, and
  `[2]=zero-extended raw FLOAT32 epsilon bits`; set `attributeCount=4` for this closed typed form and
  require every remaining payload cell to be zero.
- Validate all five indices independently for range, availability, canonical state, FLOAT32 type,
  exact role Shape, and repeated-index legality. Reconstruct epsilon from raw bits, not numeric
  integer conversion, and revalidate finite-positive FLOAT32.
- Advance workload, exact-policy, candidate, compatibility, route-policy, and codec identities
  together `12 -> 13`. Identity covers ordered/repeated five-role topology, axis, raw epsilon,
  profile, schema, descriptors/states, feeds/targets, ABI, and splats. Version 12 safely misses or
  rejects; add no migration alias.
- Keep the existing whole-partition MPSGraph candidate, publication, leases, rollback, reuse,
  concurrency, cleanup, and session isolation. Add no export, workspace, custom route, retry, host
  result, or CPU fallback.

## Training and backward boundary

`BATCH_NORM_TRAINING` is an indivisible five-input/five-output operation. Current Metal capability
rejects output count other than one; preparation throws for a multi-output node; Java and native
node records contain one output index. The public result exposing only slots zero through two does
not remove saved mean and inverse-standard-deviation slots three and four. Training therefore
requires a later explicit multi-output schema/preparer design plus separate reduction/statistic
numerics and is outside 0027.

Compiler's existing inference backward emits ordinary align/reshape, tensor arithmetic, scalar
arithmetic, RSQRT, reduction, and NEG nodes for up to five input cotangents. Current Metal does not
close that graph under either profile. A forward occurrence may retain `requiresGrad=true`, but
0027 adds no direct gradient selector, recognition/fusion, saved state, Compiler change, seeded
backward success, or strict/accelerator backward claim.

## Files and symbols

Expected implementation scope is limited to current Metal-private owners:

- `MetalCapabilityProvider`, `MetalMpsGraphProgram`, `MetalNativeApi`, and
  `MetalNegPartitionPreparer`;
- `MetalNegRouteCandidateGenerator`, `MetalNegTuningBatch`, and `MetalNegTuningCodec`;
- Metal `package-info.java` and `native/metal-macos-arm64/src/synaptik_metal_foundation.m`;
- existing focused capability, raw-ABI, prepared-execution, identity/codec, conformance, and public
  Engine integration tests; add one focused native numerical owner only if no current class owns it.

Model, Compiler, Config, Planning, Prepare, Runtime, Backend Contract, Trace, Engine, Training, CPU,
and transfer production remain read-only.

## Acceptance criteria and validation

1. The one gate passes before production edits, or the task blocks without them.
2. Both profiles admit exactly the canonical five-input/one-output domain and preserve arbitrary
   normalized channel-axis semantics, alias-safe role topology, formula, classes, and no-FTZ rule.
3. Schema 12/wire 20/attribute 7 and identity 13 reject stale/malformed records while ABI 4,
   160-byte records, thirteen exports, transfer, publication, and lifecycle remain unchanged.
4. Focused tests prove capability exclusions, first/middle/last structural geometry, repeated vector
   roles, raw schema cells/epsilon, native numerical results, identity separation, maximal-region
   composition, and one CPU-free public Engine forward under both profiles.
5. Training and generated backward remain explicitly unsupported; no shared production changes.
6. Native export/link audit, focused Metal/conformance/Engine tests, architecture checks, Javadoc,
   one final build, synchronized docs, clean diff, and independent lean Class C approval pass.

At implementation stabilization, update current Metal/backend documentation only; Model/API/compiler
semantics and glossary terminology are already current. Review uses successful implementation
evidence and reruns device work only for one newly identified concrete risk.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants)
- [Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)
- [Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)
- [Backend execution — Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)
- [Model batch-normalization inference](../../../modules/model/tasks/0021b-batch-normalization-inference.md)
- [Compiler normalization inference and gradients](../../../modules/compiler/tasks/0005b-reduction-scan-softmax-statistics-and-normalization-gradient-completion.md)
- [CPU batch-normalization inference precedent](../../../backends/cpu/tasks/0007f1-portable-batch-normalization-inference-coverage.md)

## Architecture impact

Expected impact: None. Stop and replan if implementation requires a shared/public semantic, module
edge, resource kind, profile relaxation, multi-output Metal lifecycle, or fallback.
