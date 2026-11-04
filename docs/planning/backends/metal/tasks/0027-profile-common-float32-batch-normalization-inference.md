# Task 0027: Profile-Common [FLOAT32](../../../../glossary.md#term-float32) [Batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-[Normalization](../../../../glossary.md#term-normalization) [Inference](../../../../glossary.md#term-inference)

## Status

Blocked

## Change class

Class C — this task adds a five-input floating numerical [operation](../../../../glossary.md#operation) to both [Metal](../../../../glossary.md#term-metal) profile [matrices](../../../../glossary.md#term-matrix-meanings),
evolves the [backend](../../../../glossary.md#backend)-private native [schema](../../../../glossary.md#term-schema) and route [identities](../../../../glossary.md#term-identity-meanings), and changes the generated native
[graph](../../../../glossary.md#graph). Use one clean implementation context and an independent [lean](../../../../glossary.md#proof-assistant-and-model-checking-tools) Class C review.

## Goal

Implement only first-class `BatchNormKind.BATCH_NORM_INFERENCE` for [canonical](../../../../glossary.md#term-canonical-form), positive, fully
static [`FLOAT32`](../../../../glossary.md#term-float32) [tensors](../../../../glossary.md#tensor) under both numerical profiles. Preserve [Model](../../../../glossary.md#term-model)'s arbitrary normalized
[channel](../../../../glossary.md#term-channel) [axis](../../../../glossary.md#term-axis), five ordered inputs, one output, exact formula, and special-value classes.

```text
[input, scale, bias, runningMean, runningVariance]
  -> reshape each [C] vector locally to [1,...,C,...,1]
  -> direct MPSGraph normalization
  -> one fresh canonical output
```

This is forward execution only. It adds no [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-[normalization](../../../../glossary.md#term-normalization) training, multi-output [Metal](../../../../glossary.md#term-metal) [node](../../../../glossary.md#node),
[backend](../../../../glossary.md#backend) [gradient](../../../../glossary.md#term-gradient) [operation](../../../../glossary.md#operation), or all-Metal backward claim.

## Dependencies and integration

- Depends on: Metal 0025 Complete; Model 0021B; Compiler 0005B/0005C; CPU 0007F1 only as an
  independent numerical-policy precedent; Config 0006; Engine 0018; current completed Metal
  lifecycle and profile contracts. Metal 0026 remains Blocked and is not a functional dependency.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine); [lifecycle glossary](../../../../glossary.md#lifecycle); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Conflicts with: every concurrent Metal capability, node-schema, native-preflight,
  candidate/codec, materialization, or public Engine scope; any Metal 0016–0018 restart; another
  normalization, training, multi-output, or shared semantic task.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [schema glossary](../../../../glossary.md#term-schema); [preflight glossary](../../../../glossary.md#term-preflight); [codec glossary](../../../../glossary.md#term-codec); [materialization glossary](../../../../glossary.md#materialization); [scope glossary](../../../../glossary.md#term-scope); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- Parallel group: None.
- Common base revision: `33df376e3bb6a79b8d7714f63eb151ab512feffd`.
- Integration order: mandatory one-run direct-selector gate; schema and identity; capability and
  Java/native preflight; native lowering; focused execution proof; current docs; independent review.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [identity glossary](../../../../glossary.md#term-identity-meanings); [lowering glossary](../../../../glossary.md#lowering))</span>
- Integration validation: focused Metal/conformance/real-Engine proof, native export audit,
  architecture checks, Javadoc, one final build, Markdown and diff validation.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [conformance glossary](../../../../glossary.md#term-conformance); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): [Metal](../../../../glossary.md#term-metal) 0027 implementer.

The dependencies were Complete and the exact common base contained the finalized Metal 0026
blocker. Task 0027 was the sole authorized Metal frontier until its mandatory gate failed.

## Exact capability and profile contract

Admit one [occurrence](../../../../glossary.md#term-occurrence) exactly when:

- kind and attributes are exactly `BATCH_NORM_INFERENCE + BatchNormInferenceAttrs`;
- ordered inputs are `[input, scale, bias, runningMean, runningVariance]` and there is one output;
- every [descriptor](../../../../glossary.md#term-descriptor) is [`FLOAT32`](../../../../glossary.md#term-float32), fully static, canonical dense contiguous, zero-offset, non-view,
  rank <span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [contiguous glossary](../../../../glossary.md#term-contiguous))</span>`2..16` for input/output or [rank](../../../../glossary.md#term-rank) one for each [channel](../../../../glossary.md#term-channel) [vector](../../../../glossary.md#term-vector), with positive extents;
- the normalized `channelAxis` is in `[0, inputRank)` and every vector [Shape](../../../../glossary.md#shape) is exactly `[C]`, where
  `C` is the input [extent](../../../../glossary.md#term-extent) at that [axis](../../../../glossary.md#term-axis);
- output Shape equals input Shape, output type is `FLOAT32`, and output `requiresGrad` is the logical
  OR of all five input flags;
- [epsilon](../../../../glossary.md#epsilon) is exact [finite](../../../../glossary.md#term-finite) positive `FLOAT32`; and
- checked [element](../../../../glossary.md#term-element)/byte counts, [dimension](../../../../glossary.md#dimension) cells, and native integer conversions are representable.

The four vector roles may repeat the same [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot). Deduplicate boundary feeds in first-occurrence
order, preserve repeated role indices in the [node](../../../../glossary.md#node), mutate no input, and reject every output/input
[storage overlap](../../../../glossary.md#aliasing-and-arena) before execution. Reject training, wrong [arity](../../../../glossary.md#term-arity)/attrs/type/Shape/[layout](../../../../glossary.md#layout)/[gradient](../../../../glossary.md#term-gradient)
state, [scalar](../../../../glossary.md#term-scalar)/dynamic/zero-extent/rank-above-16/[view](../../../../glossary.md#term-view-meanings) forms, malformed [aliases](../../../../glossary.md#aliasing-and-arena), and [overflow](../../../../glossary.md#term-overflow) before
native [graph](../../../../glossary.md#graph) creation.

[Batch normalization](../../../../glossary.md#batch-normalization--batch-normalization-inference-and-training) is unlisted in the architecture's [accelerator](../../../../glossary.md#term-accelerator)-relaxation table. The same exact
domain and result policy therefore apply under `STRICT_IEEE` and `ACCELERATOR`; adding the row to
both [matrices](../../../../glossary.md#term-matrix-meanings) preserves strict-subset-of-accelerator capability monotonicity.

## Documented structural basis

The installed `MPSGraphNormalizationOps.h` documents
`normalizationWithTensor:meanTensor:varianceTensor:gammaTensor:betaTensor:epsilon:name:` with
explicit [tensor](../../../../glossary.md#tensor), [mean](../../../../glossary.md#term-mean), [variance](../../../../glossary.md#term-variance), gamma, beta, and `float` [epsilon](../../../../glossary.md#epsilon) [parameters](../../../../glossary.md#term-parameter). It documents gamma
and beta [broadcasting](../../../../glossary.md#broadcasting) and gives full-[rank](../../../../glossary.md#term-rank) [singleton](../../../../glossary.md#term-singleton)-expanded [channel](../../../../glossary.md#term-channel) tensors as the [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-[normalization](../../../../glossary.md#term-normalization)
[shape](../../../../glossary.md#shape). Existing [MPSGraph](../../../../glossary.md#term-mpsgraph) reshape support documents the local `[C]` to `[1,...,C,...,1]` adapters.

For input rank `R` and channel [axis](../../../../glossary.md#term-axis) `a`, construct one rank-`R` [vector](../../../../glossary.md#term-vector) Shape containing `C` at `a`
and one elsewhere. Reshape scale, [bias](../../../../glossary.md#term-bias), running mean, and running variance to that Shape, then call
the direct selector with the original input. These [graph](../../../../glossary.md#graph)-local tensors are not semantic [nodes](../../../../glossary.md#node),
public values, materializations, workspace, or [targets](../../../../glossary.md#term-target). Java/native structural tests, not a [device](../../../../glossary.md#term-device)
[matrix](../../../../glossary.md#term-matrix-meanings), prove first/middle/last axes and ranks `2..16` use this same construction.

The header does not document [finite](../../../../glossary.md#term-finite) error, [subnormal](../../../../glossary.md#term-subnormal) preservation, [NaN](../../../../glossary.md#term-nan)/[infinity](../../../../glossary.md#term-infinity) behavior, [signed
zero](../../../../glossary.md#term-signed-zero), [overflow](../../../../glossary.md#term-overflow), or whether a supplied [statistic](../../../../glossary.md#term-statistic) tensor is consumed without an undocumented
numerical rewrite. Structural selector validity does not authorize those results.

## [Model](../../../../glossary.md#term-model) formula and numerical contract

For each [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) with [channel](../../../../glossary.md#term-channel) `c`, preserve exactly:

```text
centered     = input - runningMean[c]
denominator  = sqrt(runningVariance[c] + epsilon)
standardized = centered / denominator
output       = standardized * scale[c] + bias[c]
```

[Epsilon](../../../../glossary.md#epsilon) is inside the [square root](../../../../glossary.md#norms-elementary-functions-and-probability-distributions); running [variance](../../../../glossary.md#term-variance) is used directly; affine order is multiply then
add. Computation is [`FLOAT32`](../../../../glossary.md#term-float32). [Finite](../../../../glossary.md#term-finite) evidence may use the [CPU](../../../../glossary.md#term-cpu) precedent's two-result-[ULP](../../../../glossary.md#term-unit-in-the-last-place-ulp) [oracle](../../../../glossary.md#term-test-oracle)
ceiling for the stated real formula, but that ceiling is test evidence, not a new allowed-result set
or [accelerator](../../../../glossary.md#term-accelerator) relaxation. It cannot admit [denormals-are-zero](../../../../glossary.md#subnormal-daz-ftz-overflow-and-underflow), [flush-to-zero](../../../../glossary.md#subnormal-daz-ftz-overflow-and-underflow), class changes, wrong
[infinity](../../../../glossary.md#term-infinity)/zero signs, reciprocal substitution, or another algebraic formula.

Preserve Model's exact classes: a negative radicand produces [NaN](../../../../glossary.md#term-nan); a zero radicand produces a zero
denominator, hence centered zero gives NaN and centered nonzero gives signed infinity; positive
infinite variance gives a positive-infinite denominator, finite centered values give [signed zero](../../../../glossary.md#term-signed-zero),
and infinite centered values give NaN. Participating NaN remains NaN even through zero affine
scale. Infinite scale times standardized zero gives NaN. Ordinary [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero), [overflow](../../../../glossary.md#term-overflow), [underflow](../../../../glossary.md#term-underflow),
and final affine behavior remain observable. [NaN payload](../../../../glossary.md#nan-infinity-and-signed-zero) and sign are not promised.

## Exactly one pre-production numerical gate

Before any production edit, [run](../../../../glossary.md#term-run) one disposable [Objective-C](../../../../glossary.md#native-packaging-and-platform-vocabulary) program exactly once: one process,
[Metal](../../../../glossary.md#term-metal) [device](../../../../glossary.md#term-device)/queue, context, [graph](../../../../glossary.md#graph), executable, direct selector [target](../../../../glossary.md#term-target), and [synchronous](../../../../glossary.md#term-synchronous) execution.
Use production settings: default optimization level, set/read
`MPSGraphReducedPrecisionFastMathNone` where available, and `MPSGraphOptionsNone`. Use [canonical](../../../../glossary.md#term-canonical-form)
input [Shape](../../../../glossary.md#shape) `[2,8,2]`, [channel](../../../../glossary.md#term-channel) [axis](../../../../glossary.md#term-axis) `1`, four `[8]` [vectors](../../../../glossary.md#term-vector) locally reshaped to `[1,8,1]`, and exact
[epsilon](../../../../glossary.md#epsilon) `0.25f`.

Each channel has four independent cells because the two non-channel extents contribute
`2 * 2 = 4` [coordinates](../../../../glossary.md#axis-rank-scalar-vector-and-matrix). This makes the corpus internally satisfiable without another Shape:

| `c` | `(mean, variance, scale, bias)` | Four input cells | Contract defended |
|---:|---|---|---|
| 0 | `(1, 3.75, 2, 0.5)` | `1, 2, -1, qNaN` | supplied statistics, epsilon placement, affine order, ordinary [finite](../../../../glossary.md#term-finite) values, input [NaN](../../../../glossary.md#term-nan) |
| 1 | `(0, -0.5, +0, 7)` | `0, 1, -1, 2` | negative radicand and no suppression of its NaN by zero scale |
| 2 | `(0, -0.25, 1, +0)` | `+0, 1, -1, -0` | zero denominator: NaN, [positive infinity](../../../../glossary.md#nan-infinity-and-signed-zero), [negative infinity](../../../../glossary.md#nan-infinity-and-signed-zero), NaN |
| 3 | `(0, +infinity, 1, -0)` | `1, -1, +infinity, -infinity` | positive-infinite [variance](../../../../glossary.md#term-variance): [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero), [negative zero](../../../../glossary.md#nan-infinity-and-signed-zero), NaN, NaN |
| 4 | `(qNaN, 0.75, +0, 7)` | `0, 1, -1, +infinity` | participating [statistic](../../../../glossary.md#term-statistic) NaN survives zero scale and finite [bias](../../../../glossary.md#term-bias) |
| 5 | `(0, 0.75, +infinity, +0)` | `+0, -0, 1, -1` | infinite scale: NaN, NaN, positive infinity, negative infinity |
| 6 | `(0, 0.75, minSubnormal, -0)` | `1, -1, +0, -0` | exact signed minimum [subnormals](../../../../glossary.md#term-subnormal) and positive/negative zero; no [FTZ](../../../../glossary.md#term-ftz) |
| 7 | `(0, 0.75, maxFinite, +0)` | `2, -2, +0, qNaN` | finite [overflow](../../../../glossary.md#term-overflow) signs, positive zero, and input-NaN propagation |

All eight rows defend distinct [Model](../../../../glossary.md#term-model) obligations; none is a control-only or padded channel. Compare
ordinary finite cells with an independent [binary64](../../../../glossary.md#floating-point-precision-and-rounding) real-formula reference narrowed once to
[`FLOAT32`](../../../../glossary.md#term-float32), allowing at most two result ULPs. Require exact NaN class, [infinity](../../../../glossary.md#term-infinity) sign, zero raw sign,
and channel-6 minimum-subnormal [raw bits](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); require exact output Shape and unchanged raw bits for all
five feeds. Do not accept zero as one [ULP](../../../../glossary.md#term-unit-in-the-last-place-ulp) from a minimum subnormal.

Do not run an axis, [rank](../../../../glossary.md#term-rank), profile, optimization, context, selector, or repetition [matrix](../../../../glossary.md#term-matrix-meanings). If any
cell fails, remove source/[binary](../../../../glossary.md#term-binary-descriptor-and-metadata), mark 0027 `Blocked`, retain only exact evidence, and make no
production/[schema](../../../../glossary.md#term-schema)/capability change. Do not narrow the value domain, weaken a class/sign rule, move
[normalization](../../../../glossary.md#term-normalization) to [accelerator](../../../../glossary.md#term-accelerator)-only, decompose it, or add fallback. A pass authorizes only this direct
composition and proceeds to ordinary implementation verification.

## Blocker evidence

The mandatory gate ran from exact clean [planning](../../../../glossary.md#term-planning) revision
`3150d1871805108f48359b5b8d91124309a71f53`. One disposable [Objective-C](../../../../glossary.md#native-packaging-and-platform-vocabulary) source compiled against
the active [SDK](../../../../glossary.md#term-sdk), then its executable ran exactly once on the M3 Max. It created one [Metal](../../../../glossary.md#term-metal)
[device](../../../../glossary.md#term-device)/queue, one context, one [graph](../../../../glossary.md#graph) with five placeholders and four graph-local `[8]` to
`[1,8,1]` reshapes, one direct
`normalizationWithTensor:meanTensor:varianceTensor:gammaTensor:betaTensor:epsilon:name:` [target](../../../../glossary.md#term-target),
one executable at the production default optimization level with
`MPSGraphReducedPrecisionFastMathNone` set/read and `MPSGraphOptionsNone`, and one [synchronous](../../../../glossary.md#term-synchronous)
execution for `[2,8,2]`, [axis](../../../../glossary.md#term-axis) one, and [epsilon](../../../../glossary.md#epsilon) `0.25f`.

Exactly two of 32 output cells failed, both in [channel](../../../../glossary.md#term-channel) 6:

| Cell | Input bits | Required output | [MPSGraph](../../../../glossary.md#term-mpsgraph) output |
|---:|---:|---:|---:|
| 0 | `0x3f800000` (`+1`) | `0x00000001` (positive minimum [subnormal](../../../../glossary.md#term-subnormal)) | `0x00000000` ([positive zero](../../../../glossary.md#nan-infinity-and-signed-zero)) |
| 1 | `0xbf800000` (`-1`) | `0x80000001` (negative minimum subnormal) | `0x80000000` ([negative zero](../../../../glossary.md#nan-infinity-and-signed-zero)) |

The other 30 formula/class/sign cells passed. Exact output [Shape](../../../../glossary.md#shape) and all five unchanged feed [buffers](../../../../glossary.md#term-buffer)
passed. The failure is prohibited [flush-to-zero](../../../../glossary.md#subnormal-daz-ftz-overflow-and-underflow) under both profiles, not a [ULP](../../../../glossary.md#term-unit-in-the-last-place-ulp)-[tolerance](../../../../glossary.md#term-tolerance) issue:
zero cannot replace the required nonzero subnormal. The disposable source and executable were
removed immediately. No second device [run](../../../../glossary.md#term-run), axis/[rank](../../../../glossary.md#term-rank)/profile/options/context/repetition [matrix](../../../../glossary.md#term-matrix-meanings),
production edit, test edit, [schema](../../../../glossary.md#term-schema)/[identity](../../../../glossary.md#term-identity-meanings)/capability change, backward work, or fallback exists.

Task 0027 is therefore `Blocked`. Unblocking requires a separately planned exact replacement route
or a preceding [Model](../../../../glossary.md#term-model)/architecture numerical decision. Do not narrow inputs or outputs, exclude
subnormal-producing values, weaken the raw-bit/class rule, retry the selector, move BatchNorm to
[accelerator](../../../../glossary.md#term-accelerator)-only, decompose it, or add fallback.

## Typed [schema](../../../../glossary.md#term-schema) and [identity](../../../../glossary.md#term-identity-meanings) cutover

- Retain application binary interface (ABI) version 4, all statuses/signatures, the fixed 160-byte
  record, and exactly thirteen exports.<span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#application-binary-interface-and-native-bridge))</span><span data-glossary-navigation="true"> ([application binary interface glossary](../../../../glossary.md#term-application-binary-interface-abi); [signatures glossary](../../../../glossary.md#term-signature))</span>
- Evolve [node](../../../../glossary.md#node) schema `11 -> 12`; retain wires `1..19`; append
  `BATCH_NORM_INFERENCE=20` and typed attribute discriminator `BATCH_NORM_INFERENCE=7`.
- Encode `first_input=input`, `second_input=scale`, `auxiliary=bias`, `axis=channelAxis`,
  `attribute_values[0]=runningMean index`, `[1]=runningVariance index`, and
  `[2]=zero-extended raw FLOAT32 epsilon bits`; set `attributeCount=4` for this closed typed form and
  require every remaining payload cell to be zero.<span data-glossary-navigation="true"> ([payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication))</span>
- Validate all five indices independently for range, availability, canonical state, FLOAT32 type,
  exact role Shape, and repeated-index legality. Reconstruct epsilon from raw bits, not numeric
  integer conversion, and revalidate finite-positive FLOAT32.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [FLOAT32 glossary](../../../../glossary.md#term-float32); [Shape glossary](../../../../glossary.md#shape); [epsilon glossary](../../../../glossary.md#epsilon); [raw bits glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [finite glossary](../../../../glossary.md#term-finite))</span>
- Advance workload, exact-policy, candidate, compatibility, route-policy, and codec identities
  together <span data-glossary-navigation="true"> ([codec glossary](../../../../glossary.md#term-codec))</span>`12 -> 13`. Identity covers ordered/repeated five-role topology, axis, raw epsilon,
  profile, schema, descriptors/states, feeds/targets, ABI, and splats. Version 12 safely misses or
  rejects; add no migration alias.<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [axis glossary](../../../../glossary.md#term-axis); [descriptors glossary](../../../../glossary.md#term-descriptor); [targets glossary](../../../../glossary.md#term-target); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Keep the existing whole-partition MPSGraph candidate, publication, leases, rollback, reuse,
  concurrency, cleanup, and session isolation. Add no export, workspace, custom route, retry, host
  result, or CPU fallback.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [publication glossary](../../../../glossary.md#term-publication); [leases glossary](../../../../glossary.md#term-lease); [rollback glossary](../../../../glossary.md#term-rollback); [concurrency glossary](../../../../glossary.md#term-concurrency); [host glossary](../../../../glossary.md#term-host); [CPU glossary](../../../../glossary.md#term-cpu))</span>

## Training and backward boundary

`BATCH_NORM_TRAINING` is an indivisible five-input/five-output [operation](../../../../glossary.md#operation). Current [Metal](../../../../glossary.md#term-metal) capability
rejects output count other than one; preparation throws for a multi-output [node](../../../../glossary.md#node); Java and native
node records contain one output index. The public result exposing only slots zero through two does
not remove saved [mean](../../../../glossary.md#term-mean) and inverse-standard-deviation slots three and four. Training therefore
requires a later explicit multi-output [schema](../../../../glossary.md#term-schema)/preparer design plus separate reduction/[statistic](../../../../glossary.md#term-statistic)
numerics and is outside 0027.

[Compiler](../../../../glossary.md#term-compiler)'s existing [inference](../../../../glossary.md#term-inference) backward emits ordinary align/reshape, [tensor](../../../../glossary.md#tensor) arithmetic, [scalar](../../../../glossary.md#term-scalar)
arithmetic, RSQRT, reduction, and NEG nodes for up to five input cotangents. Current Metal does not
close that [graph](../../../../glossary.md#graph) under either profile. A forward [occurrence](../../../../glossary.md#term-occurrence) may retain `requiresGrad=true`, but
0027 adds no direct [gradient](../../../../glossary.md#term-gradient) selector, recognition/[fusion](../../../../glossary.md#term-fusion), saved state, Compiler change, seeded
backward success, or strict/[accelerator](../../../../glossary.md#term-accelerator) backward claim.

## Files and symbols

Expected implementation [scope](../../../../glossary.md#term-scope) is limited to current [Metal](../../../../glossary.md#term-metal)-private [owners](../../../../glossary.md#term-owner):

- `MetalCapabilityProvider`, `MetalMpsGraphProgram`, `MetalNativeApi`, and
  `MetalNegPartitionPreparer`;
- `MetalNegRouteCandidateGenerator`, `MetalNegTuningBatch`, and `MetalNegTuningCodec`;
- Metal `package-info.java` and `native/metal-macos-arm64/src/synaptik_metal_foundation.m`;
- existing focused capability, raw-ABI, prepared-execution, identity/codec, conformance, and public
  Engine integration tests; add one focused native numerical owner only if no current class owns it.<span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [identity glossary](../../../../glossary.md#term-identity-meanings); [codec glossary](../../../../glossary.md#term-codec); [conformance glossary](../../../../glossary.md#term-conformance); [owner glossary](../../../../glossary.md#term-owner))</span>

[Model](../../../../glossary.md#term-model), [Compiler](../../../../glossary.md#term-compiler), Config, [Planning](../../../../glossary.md#term-planning), [Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime), [Backend](../../../../glossary.md#backend) Contract, [Trace](../../../../glossary.md#trace), [Engine](../../../../glossary.md#term-engine), Training, [CPU](../../../../glossary.md#term-cpu),
and transfer production remain read-only.

## Acceptance criteria and [validation](../../../../glossary.md#term-validation)

1. The one gate passes before production edits, or the task blocks without them.
2. Both profiles admit exactly the canonical five-input/one-output domain and preserve arbitrary
   normalized channel-axis semantics, alias-safe role topology, formula, classes, and no-FTZ rule.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [channel glossary](../../../../glossary.md#term-channel); [axis glossary](../../../../glossary.md#term-axis); [alias glossary](../../../../glossary.md#aliasing-and-arena); [topology glossary](../../../../glossary.md#term-topology); [FTZ glossary](../../../../glossary.md#term-ftz))</span>
3. Schema 12/wire 20/attribute 7 and identity 13 reject stale/malformed records while ABI 4,
   160-byte records, thirteen exports, transfer, publication, and lifecycle remain unchanged.<span data-glossary-navigation="true"> ([Schema glossary](../../../../glossary.md#term-schema); [identity glossary](../../../../glossary.md#term-identity-meanings); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [publication glossary](../../../../glossary.md#term-publication); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
4. Focused tests prove capability exclusions, first/middle/last structural geometry, repeated vector
   roles, raw schema cells/epsilon, native numerical results, identity separation, maximal-region
   composition, and one CPU-free public Engine forward under both profiles.<span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector); [epsilon glossary](../../../../glossary.md#epsilon); [CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine))</span>
5. Training and generated backward remain explicitly unsupported; no shared production changes.
6. Native export/link audit, focused Metal/conformance/Engine tests, architecture checks, Javadoc,
   one final build, synchronized docs, clean diff, and independent lean Class C approval pass.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [Javadoc glossary](../../../../glossary.md#term-javadoc); [lean glossary](../../../../glossary.md#proof-assistant-and-model-checking-tools))</span>

At implementation stabilization, update current [Metal](../../../../glossary.md#term-metal)/[backend](../../../../glossary.md#backend) documentation only; [Model](../../../../glossary.md#term-model)/[API](../../../../glossary.md#term-application-programming-interface-api)/[compiler](../../../../glossary.md#term-compiler)
semantics and glossary terminology are already current. Review uses successful implementation
evidence and reruns [device](../../../../glossary.md#term-device) work only for one newly identified concrete risk.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants)
- [Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)
- [Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [automatic differentiation glossary](../../../../glossary.md#autograd))</span>
- [Backend execution — Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [Metal glossary](../../../../glossary.md#term-metal))</span>
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Model batch-normalization inference](../../../modules/model/tasks/0021b-batch-normalization-inference.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [normalization glossary](../../../../glossary.md#term-normalization); [inference glossary](../../../../glossary.md#term-inference))</span>
- [Compiler normalization inference and gradients](../../../modules/compiler/tasks/0005b-reduction-scan-softmax-statistics-and-normalization-gradient-completion.md)<span data-glossary-navigation="true"> ([gradients glossary](../../../../glossary.md#term-gradient))</span>
- [CPU batch-normalization inference precedent](../../../backends/cpu/tasks/0007f1-portable-batch-normalization-inference-coverage.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>

## Architecture impact

Expected impact: None. Stop and replan if implementation requires a shared/public semantic, [module](../../../../glossary.md#term-module)
edge, resource kind, profile relaxation, multi-output [Metal](../../../../glossary.md#term-metal) [lifecycle](../../../../glossary.md#lifecycle), or fallback.

## Result

The only permitted direct-selector gate failed the unchanged no-[FTZ](../../../../glossary.md#term-ftz) contract at [channel](../../../../glossary.md#term-channel) 6 cells
zero and one. Task 0027 is Blocked without production changes or retained probe [artifacts](../../../../glossary.md#term-artifact).
The brief remains one [atomic](../../../../glossary.md#resource-transaction-and-rollback) historical record despite exceeding the 15 KB [planning](../../../../glossary.md#term-planning) [target](../../../../glossary.md#term-target) because
the exact prospective contract and its non-retry blocker evidence must remain auditable together.
