# Task 0023: Exact [INT32](../../../../glossary.md#data-type--datatype) [GATHER](../../../../glossary.md#gather) and ONE_HOT

## Reader context

There are three boundaries here: admitting descriptors, validating actual index values, and
publishing computed results. Shapes, axis and depth described the operation; the INT32 buffer
contents determined whether its indices were valid. Native validation happened before dispatch
or destination writes. Java's exceptional download reconstructed a matching error message,
not a CPU fallback result.

GATHER copied selected FLOAT32 representations; ONE_HOT constructed canonical Boolean bytes.
Neither acquired relaxed numerical semantics from a profile. Local INT32 ingress and BOOL
publication did not themselves authorize cross-owner transfer or the generated scatter needed
for backward execution.
The [current exact indexing and movement domain](../../../../backend-guide/metal-backend.md#capability-whole-partition-analysis-and-route-selection)
connects this original forward slice to later typed routes without expanding its historical proof.

## Status

Complete

Completed at implementation `9a7911c9447eeb5125ceb8a4d78e55349ec18b97` from clean base
`854c2e8e4b1c3b4d81fd3c1f0e99656e9ae40b3e`, with committed evidence checkpoint
`80e6cedda96df2a03f284d551e6e34046eaaa2f3`. The final independent lean Class C review returned
`APPROVE` with zero findings. It reused the committed evidence and ran only one concrete-risk
disposable production-[ABI](../../../../glossary.md#term-application-binary-interface-abi) probe: [GATHER](../../../../glossary.md#gather) preserved signaling-[NaN](../../../../glossary.md#term-nan) encodings `0x7fa12345` and
`0xffa54321` exactly. The reviewer removed every probe [artifact](../../../../glossary.md#term-artifact). [Metal](../../../../glossary.md#term-metal) 0024 is the sole [Ready](../../../../glossary.md#term-ready)
successor.

## Change class

Class C — this adds profile-qualified indexing capability, the first typed [`INT32`](../../../../glossary.md#data-type--datatype) [Metal](../../../../glossary.md#term-metal) ingress and
splat path, local [`BOOL`](../../../../glossary.md#data-type--datatype) [publication](../../../../glossary.md#term-publication), native pre-write value [validation](../../../../glossary.md#term-validation) and exceptional diagnostics,
[node](../../../../glossary.md#node)/[cache](../../../../glossary.md#term-cache) [schema](../../../../glossary.md#term-schema) evolution, and public [Engine](../../../../glossary.md#term-engine) execution across the native boundary. A minimal
real-[device](../../../../glossary.md#term-device) [smoke](../../../../glossary.md#term-smoke-test) and an independent final review are mandatory.

## Goal

Add only [canonical](../../../../glossary.md#term-canonical-form) positive-[rank](../../../../glossary.md#term-rank) [`INT32`](../../../../glossary.md#data-type--datatype)-indexed [`GATHER`](../../../../glossary.md#gather) and trailing-depth `ONE_HOT` under both
`NumericalProfile.STRICT_IEEE` and `NumericalProfile.ACCELERATOR`:

- `GATHER` selects raw [`FLOAT32`](../../../../glossary.md#term-float32) values exactly, preserving every selected bit;
- `ONE_HOT` produces canonical [`BOOL`](../../../../glossary.md#data-type--datatype) bytes with a trailing depth [axis](../../../../glossary.md#term-axis);
- every logical index is validated before any target write or MPSGraph dispatch, and invalid values
  expose the same deterministic <span data-glossary-navigation="true"> ([logical index glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [target glossary](../../../../glossary.md#term-target); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>`IndexOutOfBoundsException` messages as the [CPU](../../../../glossary.md#term-cpu) route;
- Metal-local caller/splat ingress and publication widen only as explicitly declared here, while
  CPU↔Metal transfer remains canonical positive-rank <span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [publication glossary](../../../../glossary.md#term-publication))</span>`FLOAT32` only.

Keep [Compiler](../../../../glossary.md#term-compiler) production unchanged and make no gather-backward or broader training claim.

## Exact capability and [Shape](../../../../glossary.md#shape) domain

### [GATHER](../../../../glossary.md#gather)

- Admit only `AxisGatherKind.GATHER` with `IndexAxisAttrs`; keep `GATHER_ELEMENTS`, `GATHER_ND`,
  scatter, take conveniences as distinct operations, and every other indexing family false.
- Require exactly two inputs and one output. Data and output are [`FLOAT32`](../../../../glossary.md#term-float32); indices are [`INT32`](../../../../glossary.md#data-type--datatype).
  Data/output `requiresGrad` values are equal. The integral indices [descriptor](../../../../glossary.md#term-descriptor) and every [`BOOL`](../../../../glossary.md#data-type--datatype)
  descriptor are necessarily non-gradient by the Model invariant.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [Model glossary](../../../../glossary.md#term-model); [invariant glossary](../../../../glossary.md#term-invariant))</span>
- Require fully static Shapes with strictly positive extents, checked element/byte geometry, and
  resolved canonical dense-contiguous, zero-offset, non-view layouts for both inputs and output.<span data-glossary-navigation="true"> ([Shapes glossary](../../../../glossary.md#shape); [element glossary](../../../../glossary.md#term-element); [canonical glossary](../../../../glossary.md#term-canonical-form); [contiguous glossary](../../../../glossary.md#term-contiguous); [view glossary](../../../../glossary.md#term-view-meanings); [layouts glossary](../../../../glossary.md#layout))</span>
- Data [rank](../../../../glossary.md#term-rank) is `1..16`; indices rank is `1..16`; the [normalized axis](../../../../glossary.md#normalized-axis) is in `[0,dataRank)`. The
  exact output Shape is
  `data[0:axis] + indices.shape + data[axis+1:dataRank]`, and its rank
  `dataRank - 1 + indicesRank` must remain in `1..16`.
- The result is canonical. It may feed or be fed by any currently admitted same-profile operation
  whose exact canonical FLOAT32 state requirements are met. No affine-view data/index input and no
  implicit canonicalization are admitted.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form))</span>

### ONE_HOT

- Admit only `OneHotKind.ONE_HOT` with `OneHotAttrs`; require exactly one [`INT32`](../../../../glossary.md#data-type--datatype) input and one
  [`BOOL`](../../../../glossary.md#data-type--datatype) output.
- Require a fully static input Shape with strictly positive extents and canonical dense-contiguous,
  zero-offset, non-view input/output layouts. Input rank is <span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous); [view glossary](../../../../glossary.md#term-view-meanings); [layouts glossary](../../../../glossary.md#layout))</span>`1..15`; output [rank](../../../../glossary.md#term-rank) is
  `inputRank + 1 <= 16`.
- Require positive `depth`, checked native/[element](../../../../glossary.md#term-element)/byte geometry, and output [Shape](../../../../glossary.md#shape) exactly
  `indices.shape + [depth]`. The new one-hot [axis](../../../../glossary.md#term-axis) is always trailing. Values are fixed to [canonical](../../../../glossary.md#term-canonical-form)
  `true`/`false`; there is no configurable axis, on-value, or off-value.
- The output is canonical and `requiresGrad=false`. Under the current Metal matrix no operation
  consumes BOOL, so ONE_HOT is a direct partition target/publication only.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [operation glossary](../../../../glossary.md#operation); [partition glossary](../../../../glossary.md#partition); [target glossary](../../../../glossary.md#term-target); [publication glossary](../../../../glossary.md#term-publication))</span>

### Boundary forms and [fail-closed](../../../../glossary.md#term-fail-closed) exclusions

- Accept [`INT32`](../../../../glossary.md#data-type--datatype) indices from caller storage or an authenticated logical splat constant. Repeat a
  splat's exact signed 32-bit value without float conversion. Preserve existing FLOAT32 caller and
  splat behavior for GATHER data.<span data-glossary-navigation="true"> ([logical splat glossary](../../../../glossary.md#logical-splat); [constant glossary](../../../../glossary.md#term-constant); [GATHER glossary](../../../../glossary.md#gather))</span>
- Reject before native creation: [`INT64`](../../../../glossary.md#data-type--datatype) indices; non-FLOAT32 GATHER data; scalar or zero-extent
  Shapes; dynamic dimensions; rank overflow; malformed output Shapes; noncanonical layouts;
  views; byte/element overflow; wrong attributes/signatures; unsupported consumers; and every
  unlisted type or operation.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [extent glossary](../../../../glossary.md#term-extent); [Shapes glossary](../../../../glossary.md#shape); [dimensions glossary](../../../../glossary.md#dimension); [rank glossary](../../../../glossary.md#term-rank); [overflow glossary](../../../../glossary.md#term-overflow); [layouts glossary](../../../../glossary.md#layout); [views glossary](../../../../glossary.md#term-view-meanings); [element glossary](../../../../glossary.md#term-element); [signatures glossary](../../../../glossary.md#term-signature); [operation glossary](../../../../glossary.md#operation))</span>
- Keep Metal-local rank-zero reduction publication and every existing capability unchanged.
  Caller ingress gains only the canonical positive-rank <span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [rank-zero glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [publication glossary](../../../../glossary.md#term-publication); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>`INT32` form needed here. Metal-local
  publication gains canonical positive-rank [`BOOL`](../../../../glossary.md#data-type--datatype). CPU↔Metal transfer remains fully static,
  canonical, positive-rank <span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>[`FLOAT32`](../../../../glossary.md#term-float32) only; no `INT32` or `BOOL` transfer is added.

## Controlling semantics and numerical profiles

The [Model](../../../../glossary.md#term-model) sources remain authoritative: `AxisGatherKind`, `IndexAxisAttrs`, `OneHotKind`,
`OneHotAttrs`, `TensorAxisGatherExpressions`, and `TensorOneHotExpressions`. The broader Model
surface permits [INT64](../../../../glossary.md#data-type--datatype), [scalar](../../../../glossary.md#term-scalar)/zero-[extent](../../../../glossary.md#term-extent) forms, every [GATHER](../../../../glossary.md#gather) [data type](../../../../glossary.md#data-type--datatype), and any structurally valid
positive long depth; Task 0023 advertises only the bounded subset above.

Indexing/classification receives no numerical-profile relaxation. Both [Metal](../../../../glossary.md#term-metal) profile [matrices](../../../../glossary.md#term-matrix-meanings) must
therefore implement one exact result set:

- GATHER copies the selected FLOAT32 representation bit-for-bit, including both zeros, every finite
  subnormal, normals, infinities, and signed quiet/signaling NaNs with their payloads;<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [finite glossary](../../../../glossary.md#term-finite); [subnormal glossary](../../../../glossary.md#term-subnormal); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- ONE_HOT writes exactly one byte `1` at each valid selected class and byte `0` everywhere else;
- no DAZ, FTZ, reassociation, FMA, tolerance, reduced precision, conversion, NaN repair, signed-zero
  choice, wrap, clamp, default value, or all-false invalid-index behavior is permitted.<span data-glossary-navigation="true"> ([DAZ glossary](../../../../glossary.md#term-daz); [FTZ glossary](../../../../glossary.md#term-ftz); [reassociation glossary](../../../../glossary.md#term-reassociation); [FMA glossary](../../../../glossary.md#term-fused-multiply-add-fma); [tolerance glossary](../../../../glossary.md#term-tolerance); [precision glossary](../../../../glossary.md#term-precision); [NaN glossary](../../../../glossary.md#term-nan); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>

The requested numerical profile remains part of capability, preparation, workload, compatibility,
and route [identity](../../../../glossary.md#term-identity-meanings) even though the [operation](../../../../glossary.md#operation) result set is identical under both profiles.

## Invalid-index policy and deterministic failure

Resolve the cross-[backend](../../../../glossary.md#backend) diagnostic policy in favor of exact [CPU](../../../../glossary.md#term-cpu)-observable parity.

- Before any [MPSGraph](../../../../glossary.md#term-mpsgraph) execution or [target](../../../../glossary.md#term-target) write, validate every indexing [node](../../../../glossary.md#node)'s actual bound
  [`INT32`](../../../../glossary.md#data-type--datatype) Metal input buffer. Visit indexing nodes in stable encoded program order, then logical
  index elements in canonical row-major ordinal order. The first invalid value of the first invalid
  node is the sole reported error.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [buffer glossary](../../../../glossary.md#term-buffer); [logical index glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [elements glossary](../../../../glossary.md#term-element); [canonical glossary](../../../../glossary.md#term-canonical-form); [row-major glossary](../../../../glossary.md#term-row-major))</span>
- A [GATHER](../../../../glossary.md#gather) index is valid exactly when `0 <= value < data.shape[axis]`. On failure throw:
  `IndexOutOfBoundsException("GATHER index at logical position <ordinal> for data axis <axis> is out of bounds: value=<value>, extent=<extent>")`.
- A ONE_HOT index is valid exactly when `0 <= value < depth`. On failure throw:
  `IndexOutOfBoundsException("ONE_HOT index at logical position <ordinal> is out of bounds: value=<value>, depth=<depth>")`.
- Native `synaptik_metal_mpsgraph_executable_run` scans the actual shared MTLBuffer contents and
  returns existing status `RANGE_OUT_OF_BOUNDS` before dispatch. On that exceptional status only,
  Java downloads/rescans the immutable bound index buffers in the same node/ordinal order to
  reconstruct the exact exception class, message, ordinal, signed value, axis, and bound. If the
  Java rescan cannot reproduce the native result, propagate the original native failure rather
  than inventing a diagnostic.<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch); [immutable glossary](../../../../glossary.md#term-immutable); [axis glossary](../../../../glossary.md#term-axis))</span>
- Invalid execution must not dispatch the MPSGraph executable and must leave every caller-supplied
  target byte and canary unchanged. No output clearing, partial write, retry, CPU result fallback,
  or selector-dependent behavior is allowed.<span data-glossary-navigation="true"> ([canary glossary](../../../../glossary.md#term-canary))</span>
- Validation metadata is immutable prepared executable state. The valid hot path retains one Java
  run downcall and one synchronous MPSGraph submission; error-only download/rescan is not a normal
  route or undeclared workspace.<span data-glossary-navigation="true"> ([Validation glossary](../../../../glossary.md#term-validation); [metadata glossary](../../../../glossary.md#term-metadata); [prepared executable glossary](../../../../glossary.md#prepared-executable--preparedexecutable); [hot path glossary](../../../../glossary.md#term-hot-path); [run glossary](../../../../glossary.md#term-run); [downcall glossary](../../../../glossary.md#term-downcall); [synchronous glossary](../../../../glossary.md#term-synchronous))</span>

## Selector evidence policy and minimal [device](../../../../glossary.md#term-device) [smoke](../../../../glossary.md#term-smoke-test)

Apple's [MPSGraph](../../../../glossary.md#term-mpsgraph) documentation is authoritative for selector availability, accepted type/[Shape](../../../../glossary.md#shape)
forms, and valid-index [GATHER](../../../../glossary.md#gather)/ONE_HOT mapping. Task 0023 must use only the documented selectors:

- `gatherWithUpdatesTensor:indicesTensor:axis:batchDimensions:name:` with
  `batchDimensions=0`;
- `oneHotWithIndicesTensor:depth:dataType:onValue:offValue:name:` with trailing depth,
  `MPSDataTypeBool`, on-value `1.0`, and off-value `0.0`.

Do not create a context/optimization/executable cross-product probe. One small real-device smoke is
sufficient: execute a representative valid GATHER corpus containing both zero signs, [subnormal](../../../../glossary.md#term-subnormal),
ordinary, [infinity](../../../../glossary.md#term-infinity), and [NaN payload](../../../../glossary.md#nan-infinity-and-signed-zero) bits and verify selected [raw bits](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); execute one representative
ONE_HOT input and verify every output byte and Shape. Use guarded caller-supplied direct [targets](../../../../glossary.md#term-target),
check input preservation and canaries, and remove any disposable smoke [artifact](../../../../glossary.md#term-artifact).

Ordinary-GATHER and ONE_HOT out-of-bounds selector behavior is undocumented and is not part of the
implementation contract. [Host](../../../../glossary.md#term-host)/native prevalidation tests must instead prove negative,
exactly-equal-to-bound, later-ordinal, and multi-[node](../../../../glossary.md#node) invalid inputs are rejected before selector
[dispatch](../../../../glossary.md#term-dispatch), preserve target bytes/canaries, and reconstruct the deterministic [CPU](../../../../glossary.md#term-cpu)-parity diagnostic.
No selector observation may weaken that [validation](../../../../glossary.md#term-validation) requirement.

More generally, a numerical device probe is required only when authoritative documentation is
silent and a new floating numerical capability claim depends on device behavior. Such a probe must
use the smallest adversarial corpus that can decide the claim; combinatorial optimization,
context, executable, repetition, or [topology](../../../../glossary.md#term-topology) [matrices](../../../../glossary.md#term-matrix-meanings) are prohibited unless a concrete observed
risk independently justifies them.

## Native [schema](../../../../glossary.md#term-schema), [identity](../../../../glossary.md#term-identity-meanings), [lifecycle](../../../../glossary.md#lifecycle), and [cache](../../../../glossary.md#term-cache)

- Keep ABI v4, all current statuses and signatures, and exactly thirteen exports. Do not add a
  type-table parameter or diagnostic export. Native schema validation infers a closed per-value
  type state from topology: all existing nodes are FLOAT32; GATHER first input/output are FLOAT32
  and second input is INT32; ONE_HOT input is INT32 and output is BOOL. Reject inconsistent reused
  values or unknown feed/target types before graph compilation.<span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [signatures glossary](../../../../glossary.md#term-signature); [parameter glossary](../../../../glossary.md#term-parameter); [validation glossary](../../../../glossary.md#term-validation); [topology glossary](../../../../glossary.md#term-topology); [FLOAT32 glossary](../../../../glossary.md#term-float32); [GATHER glossary](../../../../glossary.md#gather); [INT32 glossary](../../../../glossary.md#data-type--datatype); [BOOL glossary](../../../../glossary.md#data-type--datatype); [graph glossary](../../../../glossary.md#graph); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names))</span>
- Evolve the fixed 160-byte [MPSGraph](../../../../glossary.md#term-mpsgraph) [node](../../../../glossary.md#node) schema `8 -> 9`. Retain wires `1..15`; append only
  `GATHER=16` and `ONE_HOT=17`. Reuse attribute kind `AXIS=3` for the normalized GATHER axis.
  Append only <span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis))</span>`DEPTH=5` for ONE_HOT, with one positive 64-bit depth in `attributeValues[0]` and all
  unused fields/reserved/padding zero. Unknown wires `18+`, unknown attributes `6+`, stale schema,
  malformed input slots, wrong states, invalid Shapes, and nonzero sentinels fail before resource
  creation.<span data-glossary-navigation="true"> ([Shapes glossary](../../../../glossary.md#shape))</span>
- [Compile](../../../../glossary.md#compile) placeholders and direct [target](../../../../glossary.md#term-target) `MPSGraphTensorData` with inferred
  `MPSDataTypeFloat32`, `MPSDataTypeInt32`, or `MPSDataTypeBool`; compute exact per-value/feed/target
  bytes from the inferred width rather than assuming four bytes. Store immutable validation records
  and stable feed/target type/permutation arrays in the native executable box.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [permutation glossary](../../../../glossary.md#permutation))</span>
- Keep the existing whole-[partition](../../../../glossary.md#partition) `MPSGRAPH` candidate. Analysis declares all buffers and existing
  binding workspace exactly; no custom compute kernel, late route, retry, or result fallback is
  added. Finalization remains transactional and creates one shape-specialized executable.<span data-glossary-navigation="true"> ([buffers glossary](../../../../glossary.md#term-buffer); [binding glossary](../../../../glossary.md#term-binding); [kernel glossary](../../../../glossary.md#kernel))</span>
- Preserve direct targets, leases, rollback and suppressed failures, reuse, concurrency,
  close/quiescence, cleanup, context/session isolation, splat initialization, and binding order.
  Exceptional diagnostic download must use the already bound exact Metal buffer and must not retain
  an error-path host allocation after the call.<span data-glossary-navigation="true"> ([leases glossary](../../../../glossary.md#term-lease); [rollback glossary](../../../../glossary.md#term-rollback); [concurrency glossary](../../../../glossary.md#term-concurrency); [quiescence glossary](../../../../glossary.md#term-quiescence); [Metal glossary](../../../../glossary.md#term-metal); [host glossary](../../../../glossary.md#term-host))</span>
- Advance all current identity meanings `9 -> 10`:
  `MetalNegRouteCandidateGenerator.WORKLOAD_SIGNATURE_VERSION`, `EXACT_DEFAULT_POLICY`,
  `MetalNegTuningBatch.CANDIDATE_SCHEMA_VERSION`, `COMPATIBILITY_SCHEMA_VERSION`,
  `ROUTE_POLICY_VERSION`, and `MetalNegTuningCodec.CODEC_VERSION`. Identity covers profile, schema,
  operation/attributes, ordered topology, inferred types/widths, descriptors/states, axis/depth,
  validation policy, feeds/targets, ABI, and typed splats. Version-nine, cross-profile, corrupt,
  malformed, stale-session, changed-type/Shape/topology, and foreign decisions reject or miss
  safely.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [descriptors glossary](../../../../glossary.md#term-descriptor))</span>

## [Compiler](../../../../glossary.md#term-compiler), [publication](../../../../glossary.md#term-publication), and backward boundary

No Compiler production file changes.

- Existing Compiler forward [capture](../../../../glossary.md#term-capture-meanings) and `PublicationPlan` already publish selected GATHER and
  ONE_HOT outputs. Task 0023 changes only Metal-local binding/materialization needed to realize
  those publications.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [binding glossary](../../../../glossary.md#term-binding); [materialization glossary](../../../../glossary.md#materialization))</span>
- `IndexingGradientRules` already defines [GATHER](../../../../glossary.md#gather)'s data [cotangent](../../../../glossary.md#cotangent) as
  `zeroLike(data).scatterAdd(indices, gradient, axis)` and no indices cotangent. Metal does not
  admit `SCATTER_ADD`, and INT32 cross-owner transfer remains false. Do not claim Metal-only or
  mixed-owner gather backward, widen transfer, or add a decomposition/fallback.<span data-glossary-navigation="true"> ([INT32 glossary](../../../../glossary.md#data-type--datatype); [owner glossary](../../../../glossary.md#term-owner); [decomposition glossary](../../../../glossary.md#term-decomposition))</span>
- ONE_HOT produces non-[differentiable](../../../../glossary.md#derivative-and-chain-rule) [BOOL](../../../../glossary.md#data-type--datatype) with no [Tensor](../../../../glossary.md#tensor) cotangent route under
  `FirstOrderGradientCoverage`/`AutogradPreflight`; leave that behavior unchanged.
- Public Engine proof is forward-only. A Metal-only gradient request containing the generated
  scatter path must fail ownership selection/preparation before Metal native creation; tests must
  not turn that negative into a backward capability claim.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [gradient glossary](../../../../glossary.md#term-gradient); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>

## Non-goals

No [INT64](../../../../glossary.md#data-type--datatype); non-[FLOAT32](../../../../glossary.md#term-float32) [GATHER](../../../../glossary.md#gather) data; GATHER_ELEMENTS/GATHER_ND/scatter; [scalar](../../../../glossary.md#term-scalar), dynamic, or zero-[extent](../../../../glossary.md#term-extent)
form; arbitrary [view](../../../../glossary.md#term-view-meanings); [BOOL](../../../../glossary.md#data-type--datatype) consumer; configurable one-hot [axis](../../../../glossary.md#term-axis)/value; [INT32](../../../../glossary.md#data-type--datatype)/BOOL transfer; custom
indexing [kernel](../../../../glossary.md#kernel); [host](../../../../glossary.md#term-host)-computed result; selector-dependent OOB policy; [Compiler](../../../../glossary.md#term-compiler) production change;
gather [gradient](../../../../glossary.md#term-gradient) execution; broader training; packaging/discovery; public Java [signature](../../../../glossary.md#term-signature);
[dependency](../../../../glossary.md#term-dependency); [Runtime](../../../../glossary.md#term-runtime)/[Trace](../../../../glossary.md#trace) profile state; or performance claim.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants)
- [Foundational modules — Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)<span data-glossary-navigation="true"> ([modules glossary](../../../../glossary.md#term-module))</span>
- [Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)<span data-glossary-navigation="true"> ([automatic differentiation glossary](../../../../glossary.md#autograd))</span>
- [Backend execution — Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [Concrete backend glossary](../../../../glossary.md#concrete-backend))</span>
  and [Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal))</span>
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>,
  [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run))</span>, and
  [public explicit composition](../../../../architecture/contracts/runtime-prepare-engine.md#public-explicit-composition)

If implementation needs wider [Model](../../../../glossary.md#term-model) semantics, [INT32](../../../../glossary.md#data-type--datatype)/[BOOL](../../../../glossary.md#data-type--datatype) cross-[owner](../../../../glossary.md#term-owner) transfer, a new native export
or [ABI](../../../../glossary.md#term-application-binary-interface-abi) [signature](../../../../glossary.md#term-signature), a custom result [kernel](../../../../glossary.md#kernel), [host](../../../../glossary.md#term-host) result fallback, [Compiler](../../../../glossary.md#term-compiler) rewriting, [Runtime](../../../../glossary.md#term-runtime)/[Trace](../../../../glossary.md#trace)
profile state, or a different invalid-index diagnostic, stop and report the contract conflict.

## Dependencies and integration

- Depends on: Metal 0022 Complete; Metal 0021 Complete; Model 0018C–0018D and 0019A2; Compiler
  0005C; CPU 0006A2 as the observable indexing oracle; Config 0006; Engine 0018<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [CPU glossary](../../../../glossary.md#term-cpu); [oracle glossary](../../../../glossary.md#term-test-oracle); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Conflicts with: any Metal 0016 restart and every Metal capability/preparation/native schema/
  candidate/codec/ingress/materialization/public Engine scope<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [codec glossary](../../../../glossary.md#term-codec); [materialization glossary](../../../../glossary.md#materialization); [scope glossary](../../../../glossary.md#term-scope))</span>
- Parallel group: None
- Common base revision: `90cd5fd925a149d055d434fac2e5bb2ee38b130f`
- Integration order: Serial after Metal 0022; documented-selector implementation with the minimal
  valid-value device smoke, then schema/type/prevalidation, capability/topology, lifecycle, Engine
  proof, documentation, and independent Class C review<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device); [smoke glossary](../../../../glossary.md#term-smoke-test); [topology glossary](../../../../glossary.md#term-topology); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Integration [validation](../../../../glossary.md#term-validation): exact [INT32](../../../../glossary.md#data-type--datatype) [GATHER](../../../../glossary.md#gather)/ONE_HOT [Metal](../../../../glossary.md#term-metal) Class C checkpoint
- Shared-document integration [owner](../../../../glossary.md#term-owner): task implementer, finalized by independent Class C review

## Files and symbols

- [Model](../../../../glossary.md#term-model) contracts, read-only: `AxisGatherKind`, `IndexAxisAttrs`, `OneHotKind`, `OneHotAttrs`,
  `TensorAxisGatherExpressions`, `TensorOneHotExpressions`.
- [Compiler](../../../../glossary.md#term-compiler) contracts, unchanged: `IndexingGradientRules`, `FirstOrderGradientCoverage`,
  `AutogradPreflight`, `GraphCompiler`, `PublicationPlan`; existing
  `IndexingScatterGradientRulesTest` remains the formula [guard](../../../../glossary.md#term-guard).
- [CPU](../../../../glossary.md#term-cpu) [oracle](../../../../glossary.md#term-test-oracle), unchanged: `CpuCapabilityProvider.supportsIndexing`, `CpuIndexingLowering`,
  `CpuPreparedExecutable.IndexValidation`, `CpuScalarReferenceKernel.execute`, and focused
  indexing/one-hot generated, ordering, differential, and semantic-closure tests.
- `MetalCapabilityProvider`, `MetalNegPartitionPreparer`, `MetalNegPreparationPlan` — exact
  both-profile capability, canonical state, type/Shape/byte/topology validation, typed splats, and
  retained validation records.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [Shape glossary](../../../../glossary.md#shape); [topology glossary](../../../../glossary.md#term-topology); [validation glossary](../../../../glossary.md#term-validation))</span>
- `MetalMpsGraphProgram`, `MetalNativeApi.MpsGraphExecutableAbi`, `MetalDeviceContext`,
  `MetalMpsGraphExecutableResource` — schema nine, wires 16/17, depth attribute five, inferred type
  state, typed data binding, exact diagnostics, and ABI-v4 stability.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [binding glossary](../../../../glossary.md#term-binding); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>
- `MetalNegPartitionFinalizer`, `MetalNegPreparedExecutable`,
  `MetalNegPreparedScheduleAssembler`, `MetalBackendRuntime`, `MetalBackendIntegration` — prepared
  resources, INT32 caller/splat ingress, BOOL local publication, exceptional rescan, lifecycle, and
  unchanged transfer predicate.<span data-glossary-navigation="true"> ([INT32 glossary](../../../../glossary.md#data-type--datatype); [BOOL glossary](../../../../glossary.md#data-type--datatype); [publication glossary](../../../../glossary.md#term-publication); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- `MetalNegRouteCandidateGenerator`, `MetalNegTuningBatch`, `MetalNegTuningDecision`,
  `MetalNegTuningCodec` — version-ten deterministic [identity](../../../../glossary.md#term-identity-meanings) and stale rejection.
- `native/metal-macos-arm64/src/synaptik_metal_foundation.m` — schema-nine validation/lowering,
  type inference, MPSGraph selectors, typed tensor data, stable permutations, and pre-dispatch
  index scan.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [tensor glossary](../../../../glossary.md#tensor); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- `MetalCapabilityProviderTest`, `MetalMpsGraphAffineSchemaTest`,
  `MetalMpsGraphRawAbiNativeTest`, `MetalNegPreparedExecutionTest`,
  `MetalNegRouteCandidateGeneratorTest`, and new `MetalMpsGraphIndexingNativeTest`.
- `testing/backend-conformance/.../MetalNegCapabilityPartitionConformanceTest` and
  `testing/integration-tests/.../EngineExplicitCompositionMetalIntegrationTest`.
- Affected Metal/native Javadocs, package information, backend/native/API/user/capability/preparer
  guides, targeted glossary, authoritative current Metal capability text, and synchronized
  planning after executable behavior passes.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [backend glossary](../../../../glossary.md#backend); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>

## Test and checkpoint requirements

Focused capability tests cover the exact positive rows under both profiles and representative
wrong kind, attr, type, [rank](../../../../glossary.md#term-rank), [axis](../../../../glossary.md#term-axis)/depth, output [Shape](../../../../glossary.md#shape), [layout](../../../../glossary.md#layout)/state, [INT64](../../../../glossary.md#data-type--datatype), [view](../../../../glossary.md#term-view-meanings), and transfer
boundaries. [Conformance](../../../../glossary.md#term-conformance) proves maximal same-[owner](../../../../glossary.md#term-owner) closure for [GATHER](../../../../glossary.md#gather) with existing [canonical](../../../../glossary.md#term-canonical-form)
operations and terminal ONE_HOT [publication](../../../../glossary.md#term-publication) without admitting a [BOOL](../../../../glossary.md#data-type--datatype) consumer.

[Schema](../../../../glossary.md#term-schema)/raw-native tests prove schema-nine encoding, retained wires `1..15`, appended wires `16/17`,
`AXIS=3`, `DEPTH=5`, inferred type consistency and byte widths, stale/unknown wire rejection, and
malformed [topology](../../../../glossary.md#term-topology)/Shape/state failure before resource publication. Prepared tests cover exact
typed caller/splat ingress, GATHER selected [raw bits](../../../../glossary.md#storage-mapping-contiguity-and-injectivity), canonical ONE_HOT bytes, direct
[materialization](../../../../glossary.md#materialization), and [lifecycle](../../../../glossary.md#lifecycle) through existing unchanged generic resource tests.

Focused [host](../../../../glossary.md#term-host)/native prevalidation tests instrument [dispatch](../../../../glossary.md#term-dispatch) and prove negative,
exactly-equal-to-bound, later-ordinal, and multi-[node](../../../../glossary.md#node) invalid inputs produce the deterministic first
failure, zero selector dispatch, unchanged prefilled [targets](../../../../glossary.md#term-target)/canaries, and exact exception text.

One small real [CPU](../../../../glossary.md#term-cpu)-free [Engine](../../../../glossary.md#term-engine) [smoke](../../../../glossary.md#term-smoke-test) runs representative forward GATHER and ONE_HOT, materializes
exact [FLOAT32](../../../../glossary.md#term-float32) bits and BOOL bytes, and proves [Metal](../../../../glossary.md#term-metal)-only [ownership](../../../../glossary.md#term-ownership-meanings). Separate negatives prove
unsupported forms fail before native creation, Metal-only gather [gradient](../../../../glossary.md#term-gradient) is unsupported, and
CPU↔Metal transfer remains FLOAT32-only. Existing generic tests own unchanged reuse, [concurrency](../../../../glossary.md#term-concurrency),
session/context, [rollback](../../../../glossary.md#term-rollback), cleanup, and close behavior and must not be duplicated.

Update architecture/current-capability text, public [API](../../../../glossary.md#term-application-programming-interface-api) status, Metal [backend](../../../../glossary.md#backend)/native guides,
Javadocs, capability/preparer status, targeted glossary, and [planning](../../../../glossary.md#term-planning) only after behavior is proven.
Do not change [Compiler](../../../../glossary.md#term-compiler) production documentation to imply a new gradient route.

## Acceptance criteria

1. The documented MPSGraph selectors and type/Shape/valid-index mapping are used directly. One
   minimal real-device smoke preserves representative selected GATHER raw bits and exact ONE_HOT
   bytes/Shape; no combinatorial executable/run matrix is added.<span data-glossary-navigation="true"> ([MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [Shape glossary](../../../../glossary.md#shape); [device glossary](../../../../glossary.md#term-device); [smoke glossary](../../../../glossary.md#term-smoke-test); [GATHER glossary](../../../../glossary.md#gather); [raw bits glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [run glossary](../../../../glossary.md#term-run); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
2. Both profiles admit exactly the bounded canonical INT32 GATHER/ONE_HOT domain. Every excluded
   kind/type/rank/Shape/layout/state/attribute/transfer/backward form rejects before native
   creation, and all prior capability remains unchanged.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [INT32 glossary](../../../../glossary.md#data-type--datatype); [rank glossary](../../../../glossary.md#term-rank); [layout glossary](../../../../glossary.md#layout))</span>
3. Every valid GATHER result preserves selected FLOAT32 bits exactly and every ONE_HOT target byte
   is canonical. No profile relaxation, conversion, invalid default, host result fallback, or
   selector-dependent semantics exists.<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [target glossary](../../../../glossary.md#term-target); [host glossary](../../../../glossary.md#term-host))</span>
4. Native validates all indexing inputs before target writes/dispatch. Focused instrumented gates
   prove deterministic node/ordinal first error, exact CPU-parity <span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch); [node glossary](../../../../glossary.md#node); [CPU glossary](../../../../glossary.md#term-cpu))</span>`IndexOutOfBoundsException`
   messages, zero selector dispatch, and byte-for-byte unchanged targets/canaries for documented
   OOB classes.
5. ABI v4 retains exactly thirteen exports; schema nine appends only wires 16/17 and attribute
   depth five; Java/native type, Shape, byte, state, and topology validation agree. Version-ten
   identities reject old, cross-profile, changed-type/Shape/topology, corrupt, stale, and foreign
   data.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [schema glossary](../../../../glossary.md#term-schema); [topology glossary](../../../../glossary.md#term-topology); [validation glossary](../../../../glossary.md#term-validation))</span>
6. Focused prepared proof covers typed caller/splat ingress, BOOL publication, direct targets, and
   valid-path one-downcall/one-submission behavior while existing generic lifecycle coverage remains
   unchanged. Cross-owner transfer remains exact FLOAT32-only.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [publication glossary](../../../../glossary.md#term-publication); [downcall glossary](../../../../glossary.md#term-downcall); [lifecycle glossary](../../../../glossary.md#lifecycle); [owner glossary](../../../../glossary.md#term-owner))</span>
7. One real CPU-free Engine forward smoke proves Metal-only execution and exact materialization.
   Existing Compiler gather-scatter formula and ONE_HOT nondifferentiability stay unchanged; no
   gather backward claim exists.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [Metal glossary](../../../../glossary.md#term-metal); [materialization glossary](../../../../glossary.md#materialization); [Compiler glossary](../../../../glossary.md#term-compiler))</span>
8. Native/export, focused Metal/Javadoc, conformance, Compiler formula guard, real Engine,
   architecture, full-build, Markdown/status/frontier, smoke-artifact removal, and diff checks
   pass; independent Class C review has no unresolved finding.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [conformance glossary](../../../../glossary.md#term-conformance); [guard glossary](../../../../glossary.md#term-guard); [artifact glossary](../../../../glossary.md#term-artifact))</span>

## [Validation](../../../../glossary.md#term-validation)

Worker validation:

```bash
./native/metal-macos-arm64/build.sh
nm -gU native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" ./gradlew :backends:metal:test :backends:metal:javadoc
./gradlew :modules:compiler:test --tests '*IndexingScatterGradientRulesTest*'
./gradlew :testing:backend-conformance:test --tests '*Metal*'
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" ./gradlew :testing:integration-tests:test --tests '*EngineExplicitCompositionMetalIntegrationTest*'
./gradlew :testing:architecture-tests:test
./gradlew build
python3 /tmp/validate_synaptik_markdown.py
git diff --check
```

Record the documented selectors, minimal valid-value [device](../../../../glossary.md#term-device) [smoke](../../../../glossary.md#term-smoke-test) and [artifact](../../../../glossary.md#term-artifact) removal,
pre-[dispatch](../../../../glossary.md#term-dispatch) OOB evidence, exact exception messages and unchanged [targets](../../../../glossary.md#term-target); exports/[schema](../../../../glossary.md#term-schema)/wires/
attribute/[identity](../../../../glossary.md#term-identity-meanings) versions; focused test counts/skips; both-profile positives and negatives;
typed ingress/[publication](../../../../glossary.md#term-publication) and unchanged transfer; [Compiler](../../../../glossary.md#term-compiler) no-production-change evidence; changed
paths; and synchronized frontier/status. The serial worker owns one full checkpoint. Independent
review reuses evidence and reruns only for a concrete risk or relevant executable change.

## Documentation and review impact

Independent Class C review inspects documented-selector use, the minimal raw-bit/[canonical](../../../../glossary.md#term-canonical-form)-[BOOL](../../../../glossary.md#data-type--datatype)
[smoke](../../../../glossary.md#term-smoke-test) and [artifact](../../../../glossary.md#term-artifact) removal, OOB non-reliance, native prevalidation and Java exceptional rescan,
deterministic first error, unchanged [targets](../../../../glossary.md#term-target), [schema](../../../../glossary.md#term-schema)/type inference, version-ten [identity](../../../../glossary.md#term-identity-meanings), typed
ingress/local [publication](../../../../glossary.md#term-publication) versus unchanged transfer, focused [lifecycle](../../../../glossary.md#lifecycle) reliance, the real
[Metal](../../../../glossary.md#term-metal)-only [Engine](../../../../glossary.md#term-engine) smoke, forward-only boundary, [Runtime](../../../../glossary.md#term-runtime)/[Trace](../../../../glossary.md#trace) absence, changed [scope](../../../../glossary.md#term-scope), and all
[validation](../../../../glossary.md#term-validation)/documentation evidence.

## Implementation evidence

Implementation `9a7911c9447eeb5125ceb8a4d78e55349ec18b97` began from clean
`854c2e8e4b1c3b4d81fd3c1f0e99656e9ae40b3e`; committed evidence checkpoint
`80e6cedda96df2a03f284d551e6e34046eaaa2f3` records the final implementation evidence.

Before any production edit, one disposable [Objective-C](../../../../glossary.md#native-packaging-and-platform-vocabulary) program at
`/tmp/synaptik_task0023_selector_smoke.m` compiled with `xcrun --sdk macosx clang -arch arm64`
against Foundation, [Metal](../../../../glossary.md#term-metal), and MetalPerformanceShadersGraph and ran once on the M3 Max. It used
exactly `gatherWithUpdatesTensor:indicesTensor:axis:batchDimensions:name:` with [axis](../../../../glossary.md#term-axis) `1` and
`batchDimensions=0`, plus
`oneHotWithIndicesTensor:depth:dataType:onValue:offValue:name:` with depth `4`, [`BOOL`](../../../../glossary.md#data-type--datatype), one, and
zero. The single execution reported:

```text
task0023 selector smoke passed: gather raw bits, one-hot BOOL bytes, shapes [2,3]/[3,4]
```

[GATHER](../../../../glossary.md#gather) selected raw [FLOAT32](../../../../glossary.md#term-float32) [signed zero](../../../../glossary.md#term-signed-zero), [subnormal](../../../../glossary.md#term-subnormal), [infinity](../../../../glossary.md#term-infinity), and [NaN](../../../../glossary.md#term-nan)-payload encodings without
conversion; ONE_HOT produced exact zero/one bytes and the declared [Shape](../../../../glossary.md#shape); all inputs remained
unchanged. No optimization, context, repetition, or executable [matrix](../../../../glossary.md#term-matrix-meanings) ran. The source and
`/tmp/synaptik_task0023_selector_smoke` [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) were removed before production edits.

Production uses the same selectors. Native executable creation retains indexing validations in
stable [node](../../../../glossary.md#node) order. Each [run](../../../../glossary.md#term-run) validates every input/output resource and scans each [canonical](../../../../glossary.md#term-canonical-form) [INT32](../../../../glossary.md#data-type--datatype)
index in [row-major](../../../../glossary.md#term-row-major) ordinal order before constructing `MPSGraphTensorData`, calling the executable,
or writing a [target](../../../../glossary.md#term-target). Equal-bound GATHER and negative later-ordinal ONE_HOT real-[device](../../../../glossary.md#term-device) failures
return range status with every target byte still at its sentinel. Java rescans the exact retained
Metal inputs only after that native range status and publishes:

```text
GATHER index at logical position 1 for data axis 1 is out of bounds: value=4, extent=4
ONE_HOT index at logical position 1 is out of bounds: value=-1, depth=4
```

The [schema](../../../../glossary.md#term-schema) is `9`; retained [operation](../../../../glossary.md#operation) wires are `1..15`, new wires are `GATHER=16` and
`ONE_HOT=17`, and `DEPTH=5`. [ABI](../../../../glossary.md#term-application-binary-interface-abi) `4` and all thirteen exports are unchanged. Workload,
exact-policy, candidate, compatibility, route-policy, and [codec](../../../../glossary.md#term-codec) [identities](../../../../glossary.md#term-identity-meanings) are `10`. Caller and
type-matching splat ingress accept exact canonical FLOAT32 or INT32; local canonical BOOL
[publication](../../../../glossary.md#term-publication) uses one byte per [element](../../../../glossary.md#term-element). [CPU](../../../../glossary.md#term-cpu)-to-Metal and Metal-to-CPU transfer remain positive-[rank](../../../../glossary.md#term-rank)
canonical FLOAT32 only. No [Compiler](../../../../glossary.md#term-compiler) production, [Runtime](../../../../glossary.md#term-runtime), or [Trace](../../../../glossary.md#trace) file changed; the route is
forward-only and adds no [INT64](../../../../glossary.md#data-type--datatype), CPU fallback, transfer [widening](../../../../glossary.md#term-widening), or general BOOL consumer.

[Validation](../../../../glossary.md#term-validation) evidence:

- native `build.sh`: pass; `nm -gU` lists exactly the documented thirteen symbols and no removed
  NEG-only create symbol;<span data-glossary-navigation="true"> ([symbol glossary](../../../../glossary.md#term-symbol))</span>
- focused Metal set: 7 tests, 0 skipped, 0 failures, including one capability test, one
  schema/preflight test, one raw-ABI test, one prepared typed-ingress/diagnostic test, two
  route/codec tests, and one real-device native test;<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight))</span>
- Metal [Javadoc](../../../../glossary.md#term-javadoc): pass;
- [backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance): 1 focused test, 0 skipped, 0 failures;
- existing Compiler `IndexingScatterGradientRulesTest`: 2 tests, 0 skipped, 0 failures;
- real CPU-free [Engine](../../../../glossary.md#term-engine) [smoke](../../../../glossary.md#term-smoke-test): 1 test, 0 skipped, 0 failures;
- architecture checks: pass;
- one full `./gradlew build`: pass, 87 actionable tasks;
- Markdown/status/frontier validation passed for the other ten changed Markdown files. The changed
  `docs/api/tensor-api.md` retains exactly the same 172 pre-existing duplicate-heading reports as
  clean HEAD and adds no validator error; `git diff --check` passes.

Changed paths:

- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalBackendIntegration.java`
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalBackendRuntime.java`
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalCapabilityProvider.java`
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalMpsGraphProgram.java`
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalNativeApi.java`
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalNegPartitionPreparer.java`
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalNegPreparedExecutable.java`
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalNegPreparedScheduleAssembler.java`
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalNegRouteCandidateGenerator.java`
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalNegTuningBatch.java`
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalNegTuningCodec.java`
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/package-info.java`
- `backends/metal/src/test/java/io/github/pho001/synaptik/backend/metal/MetalCapabilityProviderTest.java`
- `backends/metal/src/test/java/io/github/pho001/synaptik/backend/metal/MetalMpsGraphAbsNativeTest.java`
- `backends/metal/src/test/java/io/github/pho001/synaptik/backend/metal/MetalMpsGraphAffineSchemaTest.java`
- `backends/metal/src/test/java/io/github/pho001/synaptik/backend/metal/MetalMpsGraphBinaryNativeTest.java`
- `backends/metal/src/test/java/io/github/pho001/synaptik/backend/metal/MetalMpsGraphIndexingNativeTest.java`
- `backends/metal/src/test/java/io/github/pho001/synaptik/backend/metal/MetalMpsGraphRawAbiNativeTest.java`
- `backends/metal/src/test/java/io/github/pho001/synaptik/backend/metal/MetalMpsGraphReductionNativeTest.java`
- `backends/metal/src/test/java/io/github/pho001/synaptik/backend/metal/MetalNegPreparedExecutionTest.java`
- `backends/metal/src/test/java/io/github/pho001/synaptik/backend/metal/MetalNegRouteCandidateGeneratorTest.java`
- `native/metal-macos-arm64/src/synaptik_metal_foundation.m`
- `native/metal-macos-arm64/README.md`
- `testing/backend-conformance/src/test/java/io/github/pho001/synaptik/testing/conformance/MetalNegCapabilityPartitionConformanceTest.java`
- `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/EngineExplicitCompositionMetalIntegrationTest.java`
- `ARCHITECTURE.md`
- `docs/api/compile-api.md`
- `docs/api/tensor-api.md`
- `docs/architecture/contracts/backend-execution.md`
- `docs/backend-guide/metal-backend.md`
- `docs/backend-guide/partition-preparer.md`
- `docs/glossary.md`
- `docs/planning/backends/metal/master-plan.md`
- `docs/planning/backends/metal/tasks/0023-exact-int32-gather-and-one-hot.md`
- `docs/planning/roadmap.md`

## Final independent review

The independent lean Class C review returned `APPROVE` with zero findings. It reused the committed
implementation/evidence instead of repeating the established selector, [lifecycle](../../../../glossary.md#lifecycle), native,
[conformance](../../../../glossary.md#term-conformance), [Engine](../../../../glossary.md#term-engine), architecture, or build [matrix](../../../../glossary.md#term-review-matrix). The reviewer identified one concrete residual
raw-bit risk and ran one disposable production-[ABI](../../../../glossary.md#term-application-binary-interface-abi) [GATHER](../../../../glossary.md#gather) probe only: signaling-[NaN](../../../../glossary.md#term-nan) encodings
`0x7fa12345` and `0xffa54321` were preserved exactly. All disposable source, [binary](../../../../glossary.md#term-binary-descriptor-and-metadata), and generated
[artifacts](../../../../glossary.md#term-artifact) were removed. Implementation `9a7911c9` plus evidence `80e6cedd` is therefore Complete.
