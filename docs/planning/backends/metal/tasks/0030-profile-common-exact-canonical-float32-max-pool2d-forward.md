# Task 0030: Profile-Common Exact [Canonical](../../../../glossary.md#term-canonical-form) [FLOAT32](../../../../glossary.md#term-float32) MAX_POOL2D Forward

## Reader context

This was a failed authorization gate, not an implemented pooling route. Structural mapping
of window geometry could not prove which represented value won each window. The frozen corpus
tested signed-zero preference and NaN propagation as well as ordinary values, and the recorded
failures stopped production work.

Model owned that exact winner policy; Compiler's later gradient construction needed the same
winner semantics without a saved-index public output. Metal could not substitute the direct
selector's observed policy. Input descriptors described geometry and type, while the raw test
words established value behavior.
The [current pooling rules and numerical domain](../../../../backend-guide/metal-backend.md#ordinary-numerical-and-exact-domains)
state the later NaN-first, positive-zero-preferred implementation boundary without changing this
historical output or its Blocked status.

## Status

Blocked

## Change class

Class C — this task adds a numerical [operation](../../../../glossary.md#operation) to both [Metal](../../../../glossary.md#term-metal) profile [matrices](../../../../glossary.md#term-matrix-meanings), evolves the
[backend](../../../../glossary.md#backend)-private native [schema](../../../../glossary.md#term-schema) and route [identities](../../../../glossary.md#term-identity-meanings), and changes the generated native [graph](../../../../glossary.md#graph). Use one
clean implementation context and an independent [lean](../../../../glossary.md#proof-assistant-and-model-checking-tools) Class C review.

## Goal

Implement only first-class `Pool2dKind.MAX_POOL2D` forward execution for [canonical](../../../../glossary.md#term-canonical-form), positive, fully
static [rank](../../../../glossary.md#term-rank)-four [NCHW](../../../../glossary.md#term-nchw) [`FLOAT32`](../../../../glossary.md#term-float32) [tensors](../../../../glossary.md#tensor) under both numerical profiles. Preserve every [Model](../../../../glossary.md#term-model)-valid
checked geometry and exact maximum-selection semantic while [gradient](../../../../glossary.md#term-gradient)-bearing occurrences [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation).

## [Scope](../../../../glossary.md#term-scope)

- Admit exactly `MAX_POOL2D + MaxPool2dAttrs`, one input and one output, when both descriptors are
  canonical dense contiguous, zero-offset, non-view <span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [canonical glossary](../../../../glossary.md#term-canonical-form); [contiguous glossary](../../../../glossary.md#term-contiguous); [view glossary](../../../../glossary.md#term-view-meanings))</span>[`FLOAT32`](../../../../glossary.md#term-float32) [NCHW](../../../../glossary.md#term-nchw) [rank](../../../../glossary.md#term-rank)-four [tensors](../../../../glossary.md#tensor) with positive static extents and `requiresGrad=false`.
- Independently rederive checked `[N,C,Hout,Wout]` geometry and require exact output Shape, type,
  canonical layout, and no-gradient metadata. Admit the complete documented Model-valid attribute domain that fits checked Java/native geometry; add no probe-derived limit.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [gradient glossary](../../../../glossary.md#term-gradient); [metadata glossary](../../../../glossary.md#term-metadata); [Model glossary](../../../../glossary.md#term-model))</span>
- Lower one native [node](../../../../glossary.md#node) through `maxPooling2DWithSourceTensor:descriptor:name:` and
  `MPSGraphPooling2DOpDescriptor`, with `MPSGraphPoolingReturnIndicesNone` and one result.
- Add [schema](../../../../glossary.md#term-schema) 12, wire `MAX_POOL2D=20`, discriminator `POOL2D=7`, and version-thirteen identities
  while retaining ABI 4, the 160-byte record, and exactly thirteen exports.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>
- Keep current whole-[partition](../../../../glossary.md#partition) [MPSGraph](../../../../glossary.md#term-mpsgraph) preparation, storage, [publication](../../../../glossary.md#term-publication), reuse, [rollback](../../../../glossary.md#term-rollback), [concurrency](../../../../glossary.md#term-concurrency), cleanup, and session isolation unchanged.

## Non-goals

- `AVERAGE_POOL2D`; Conv1d/2d/3d; Pool1d or [Pool3d](../../../../glossary.md#pool3d--ncdhw-pooling); adaptive/global [pooling](../../../../glossary.md#term-pooling); another [layout](../../../../glossary.md#layout) or type
- `requiresGrad=true`, pooling backward, [gradient](../../../../glossary.md#term-gradient) selectors, indices, a second output, unpooling, recognition, [fusion](../../../../glossary.md#term-fusion), or hidden state
- affine/[view](../../../../glossary.md#term-view-meanings) inputs, dynamic or zero extents, noncanonical layouts, transfer [widening](../../../../glossary.md#term-widening), workspace, custom [kernels](../../../../glossary.md#kernel), fallback, retries, performance claims, or shared production changes
- a geometry, [padding](../../../../glossary.md#padding), [dilation](../../../../glossary.md#term-dilation), ceil, [rank](../../../../glossary.md#term-rank), profile, optimization, context, or repetition [device](../../../../glossary.md#term-device) [matrix](../../../../glossary.md#term-matrix-meanings); documented structural mappings are authorized and tested without numerical probing

## Contracts

- [`ARCHITECTURE.md` — Authority, incorporation, and precedence](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence)
  requires plans to preserve the root and incorporated scoped contracts.
- [Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)
  leaves pooling unrelaxed; strict and accelerator therefore use the identical Model allowed set.<span data-glossary-navigation="true"> ([pooling glossary](../../../../glossary.md#term-pooling); [accelerator glossary](../../../../glossary.md#term-accelerator); [Model glossary](../../../../glossary.md#term-model))</span>
- [Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [backend glossary](../../../../glossary.md#backend))</span> owns
  MPSGraph lowering, native integration, storage, and materialization.<span data-glossary-navigation="true"> ([MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [lowering glossary](../../../../glossary.md#lowering); [materialization glossary](../../../../glossary.md#materialization))</span>
- [Numerical-profile backend identity](../../../../architecture/contracts/backend-execution.md#numerical-profile-backend-identity)<span data-glossary-navigation="true"> ([backend identity glossary](../../../../glossary.md#backend-identity--backendid))</span>
  requires profile-qualified capability and compatible route identities.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span>
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare))</span>
  keeps lowering in Metal analysis/finalization and Runtime free of route selection.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime))</span>
- [Model MAX_POOL2D](../../../modules/model/tasks/0020a-nchw-max-pool2d-semantics-and-tensor-expression.md)
  owns NCHW geometry, padding exclusion, all-padding negative infinity, NaN, signed-zero, and
  first-logical-winner meaning.<span data-glossary-navigation="true"> ([NCHW glossary](../../../../glossary.md#term-nchw); [padding glossary](../../../../glossary.md#padding); [negative infinity glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [NaN glossary](../../../../glossary.md#term-nan); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>

If implementation needs a profile relaxation, public/shared semantic, [module](../../../../glossary.md#term-module) edge, new resource
kind, multi-output [lifecycle](../../../../glossary.md#lifecycle), or fallback, stop and replan.

## Dependencies and integration

- Depends on: [Metal](../../../../glossary.md#term-metal) 0025 Complete; [Model](../../../../glossary.md#term-model) 0020A; [Compiler](../../../../glossary.md#term-compiler) 0005D/0006B11; [CPU](../../../../glossary.md#term-cpu) 0008G exact-policy precedent; Config 0006; [Engine](../../../../glossary.md#term-engine) 0018; current Metal [lifecycle](../../../../glossary.md#lifecycle).
- Conflicts with: every concurrent Metal capability/[schema](../../../../glossary.md#term-schema)/native-[preflight](../../../../glossary.md#term-preflight)/candidate/[codec](../../../../glossary.md#term-codec)/[materialization](../../../../glossary.md#materialization)/public Engine [scope](../../../../glossary.md#term-scope); any 0016–0018 restart; [convolution](../../../../glossary.md#term-channels-first-convolution-layers), average/1D/3D [pooling](../../../../glossary.md#term-pooling), backward, multi-output, or shared semantic work.
- Parallel group: None.
- Common base revision: `030d4c561cc1a836d94a8c01c41ffef86a8c807b`; execution starts only from the committed [planning](../../../../glossary.md#term-planning) revision containing this brief.
- Integration order: sole one-execution raw-winner gate; schema/[identity](../../../../glossary.md#term-identity-meanings); capability and Java/native preflight; native [lowering](../../../../glossary.md#lowering); focused proof; current docs; independent review.
- Integration [validation](../../../../glossary.md#term-validation): focused Metal/[conformance](../../../../glossary.md#term-conformance)/one real-Engine [checkpoint](../../../../glossary.md#term-checkpoint), native export audit, architecture checks, [Javadoc](../../../../glossary.md#term-javadoc), one final build, Markdown and diff validation.
- Shared-document integration [owner](../../../../glossary.md#term-owner): Metal 0030 implementer.

All declared predecessors were Complete, blocked 0026/0027 had no active write scope, and 0030 was
the sole authorized Metal frontier until its mandatory gate failed.

## Documented structural authorization

The installed `MPSGraphPoolingOps.h` documents a [rank](../../../../glossary.md#term-rank)-four source, [descriptor](../../../../glossary.md#term-descriptor)-owned [layout](../../../../glossary.md#layout), a
one-result max-[pooling](../../../../glossary.md#term-pooling) selector, [kernel](../../../../glossary.md#kernel), X/Y [stride](../../../../glossary.md#term-stride) and [dilation](../../../../glossary.md#term-dilation), explicit four-side [padding](../../../../glossary.md#padding),
`ceilMode`, and optional indices. `MPSGraphCore.h` defines [`NCHW`](../../../../glossary.md#term-nchw) and explicit padding.

Map [Model](../../../../glossary.md#term-model) width to X, height to Y, and symmetric padding to equal left/right and top/bottom values.
Set explicit padding, NCHW, exact geometry, `ceilMode`, and no indices. These declarations plus
Model's formula authorize all admitted geometry. Java/native [preflight](../../../../glossary.md#term-preflight) independently recomputes
effective [kernels](../../../../glossary.md#kernel), padded numerators, literal floor/ceil output extents, conversions, counts, and
exact output [Shape](../../../../glossary.md#shape). Structural tests cover non-square/padded/dilated/ceil/all-padding geometry without a [device](../../../../glossary.md#term-device) [matrix](../../../../glossary.md#term-matrix-meanings).

The one-output selector matches Model's 1 -> 1 [signature](../../../../glossary.md#term-signature). [Compiler](../../../../glossary.md#term-compiler) reconstructs max-pool winners
through ordinary public operations and exposes no saved index result. Do not call the return-indices
or [gradient](../../../../glossary.md#term-gradient) selectors.

## Exact numerical contract

[Padding](../../../../glossary.md#padding) is excluded; an all-padding window returns exact [negative infinity](../../../../glossary.md#nan-infinity-and-signed-zero). Otherwise [NaN](../../../../glossary.md#term-nan) wins,
then ordinary maximum applies with [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero) above [negative zero](../../../../glossary.md#nan-infinity-and-signed-zero). Equal candidates keep the
first height-then-width [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters). Preserve the selected non-NaN [`FLOAT32`](../../../../glossary.md#term-float32) word exactly.

[Model](../../../../glossary.md#term-model) does not promise [NaN payload](../../../../glossary.md#nan-infinity-and-signed-zero), sign, or signaling preservation. Require NaN class for a
selected quiet or [signaling NaN](../../../../glossary.md#term-signaling-nan) without inventing [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) [identity](../../../../glossary.md#term-identity-meanings). Multiple-NaN first-winner order
remains Model/[Compiler](../../../../glossary.md#term-compiler) [gradient](../../../../glossary.md#term-gradient) policy, is not distinguishable in allowed forward bits, and justifies no extra probe.

[Pooling](../../../../glossary.md#term-pooling) is unlisted in the [accelerator](../../../../glossary.md#term-accelerator)-relaxation table. Both profiles use this exact policy: no
[DAZ](../../../../glossary.md#term-daz), [FTZ](../../../../glossary.md#term-ftz), [tolerance](../../../../glossary.md#term-tolerance), alternate [signed zero](../../../../glossary.md#term-signed-zero), or [finite](../../../../glossary.md#term-finite)-value approximation.

## Sole pre-production numerical gate

Before any production edit, [run](../../../../glossary.md#term-run) one disposable [Objective-C](../../../../glossary.md#native-packaging-and-platform-vocabulary) program exactly once: one process, one
[device](../../../../glossary.md#term-device)/queue context, [graph](../../../../glossary.md#graph), executable, direct max-pool [node](../../../../glossary.md#node), [target](../../../../glossary.md#term-target), and [synchronous](../../../../glossary.md#term-synchronous) execution.
Use production default [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) settings once; vary no profile, optimization, option, context, [binding](../../../../glossary.md#term-binding), or repetition.

Use [canonical](../../../../glossary.md#term-canonical-form) input `[1,1,1,18]`, output `[1,1,1,9]`, and attrs
`(kernelHeight=1, kernelWidth=2, strideHeight=1, strideWidth=2, paddingHeight=0,
paddingWidth=0, dilationHeight=1, dilationWidth=1, ceilMode=false)`. Its nine disjoint windows are:

```text
[1, 2]
[-0, +0]
[+0, -0]
[+0, +minSubnormal]
[-1, -minSubnormal]
[qNaN-A, 3]
[3, qNaN-B]
[sNaN-A, 3]
[3, sNaN-B]
```

Freeze `qNaN-A=0x7fc12345`, `qNaN-B=0xffc23456`, `sNaN-A=0x7fa12345`, and
`sNaN-B=0xffa23456`. Read every output raw word.
Require exact `2`, positive-zero, positive-zero, positive minimum [subnormal](../../../../glossary.md#term-subnormal), and negative minimum
subnormal words for the first five windows; require [NaN](../../../../glossary.md#term-nan) class for each last-four word because [Model](../../../../glossary.md#term-model)
leaves [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication)/sign/signaling unspecified. Require exact output [Shape](../../../../glossary.md#shape), unchanged input words, and no
[guard](../../../../glossary.md#term-guard) corruption. This defends ordinary selection; [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero) ordering in both operand orders;
positive/negative subnormal comparison and selected-word preservation; quiet/[signaling NaN](../../../../glossary.md#term-signaling-nan)
propagation from first and later positions; source immutability; and exact [publication](../../../../glossary.md#term-publication).

Do not add [padding](../../../../glossary.md#padding)/[dilation](../../../../glossary.md#term-dilation)/ceil cases or any [matrix](../../../../glossary.md#term-matrix-meanings)/repeat. On any failure, remove source/[binary](../../../../glossary.md#term-binary-descriptor-and-metadata),
mark 0030 Blocked with exact evidence, and make no production/[schema](../../../../glossary.md#term-schema)/capability change. Do not
narrow geometry or values, weaken the [oracle](../../../../glossary.md#term-test-oracle), move [pooling](../../../../glossary.md#term-pooling) to [accelerator](../../../../glossary.md#term-accelerator)-only, use indices,
decompose, retry, or add fallback. A pass authorizes implementation; independent review reuses it
unless it identifies a new concrete risk and first replans the one-execution policy.

## Blocker evidence

The sole gate ran from exact clean [planning](../../../../glossary.md#term-planning) revision
`f5e819cb27ac863eb04b6ff0f80b5c43182b332c` on the M3 Max. One disposable [Objective-C](../../../../glossary.md#native-packaging-and-platform-vocabulary) source
compiled against the active [SDK](../../../../glossary.md#term-sdk), then its executable ran exactly once. It created one [Metal](../../../../glossary.md#term-metal)
[device](../../../../glossary.md#term-device)/queue context, [graph](../../../../glossary.md#graph), executable, direct max-pool [node](../../../../glossary.md#node), [target](../../../../glossary.md#term-target), and [synchronous](../../../../glossary.md#term-synchronous) execution
for the frozen `[1,1,1,18] -> [1,1,1,9]` corpus. Production default [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names), successful
reduced-[precision](../../../../glossary.md#term-precision)-fast-math-none set/read, and `MPSGraphOptionsNone` were used once.

The nine actual output words, in frozen window order, were:

```text
0x40000000, 0x00000000, 0x80000000, 0x00000001, 0x80000001,
0x40400000, 0x40400000, 0x40400000, 0x40400000
```

Ordinary `2`, `[-0,+0] -> +0`, positive/negative minimum-[subnormal](../../../../glossary.md#term-subnormal) preservation, exact output
[Shape](../../../../glossary.md#shape), unchanged input words, both [guard](../../../../glossary.md#term-guard) words, and synchronous execution passed.
`[+0,-0]` returned [negative zero](../../../../glossary.md#nan-infinity-and-signed-zero) `0x80000000` instead of required [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero) `0x00000000`.
Each first/later qNaN/sNaN window returned [finite](../../../../glossary.md#term-finite) `3.0f` (`0x40400000`) instead of [NaN](../../../../glossary.md#term-nan). Thus the
required [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero) order and all four NaN-class cells failed; this is not a [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication)-policy issue.

The source and [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) were removed immediately. No second execution, retry, geometry/profile/
option/context/repetition [matrix](../../../../glossary.md#term-matrix-meanings), production or test edit, [schema](../../../../glossary.md#term-schema)/[identity](../../../../glossary.md#term-identity-meanings)/capability change,
[narrowing](../../../../glossary.md#term-narrowing), indices route, [decomposition](../../../../glossary.md#term-decomposition), or fallback exists. Task 0030 is `Blocked`. Unblocking
requires a separately planned exact replacement route or a preceding [Model](../../../../glossary.md#term-model)/architecture decision;
do not retry or weaken this direct-selector contract.

## Typed [schema](../../../../glossary.md#term-schema) and [identity](../../../../glossary.md#term-identity-meanings) cutover

- Append `NodeKind.MAX_POOL2D(20, 1, POOL2D, CANONICAL, false)` and `AttributeKind.POOL2D(7)`;
  advance node schema <span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node))</span>`11 -> 12` and retain wires `1..19` unchanged.
- Encode `first_input=input`, absent second-input sentinel, `output=output`, absent [axis](../../../../glossary.md#term-axis) sentinel,
  `auxiliary=ceilMode ? 1 : 0`, `attributeCount=8`, and attribute values in this order:
  kernel height/width, stride height/width, padding height/width, dilation height/width. Require every
  remaining cell zero.<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel); [stride glossary](../../../../glossary.md#term-stride); [padding glossary](../../../../glossary.md#padding); [dilation glossary](../../../../glossary.md#term-dilation))</span>
- Java and native validation independently authenticate kind/attrs/profile, one-input/one-output
  topology, canonical state, FLOAT32, rank four, positive static Shapes, no-grad flags, geometry,
  exact output Shape, sentinels, payload cells, and checked native conversions before graph creation.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [topology glossary](../../../../glossary.md#term-topology); [canonical glossary](../../../../glossary.md#term-canonical-form); [FLOAT32 glossary](../../../../glossary.md#term-float32); [rank glossary](../../../../glossary.md#term-rank); [Shapes glossary](../../../../glossary.md#shape); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [graph glossary](../../../../glossary.md#graph))</span>
- Advance workload signature, exact default policy, candidate schema, compatibility schema,
  route-policy, and codec together <span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [codec glossary](../../../../glossary.md#term-codec))</span>`12 -> 13`. Identity includes profile, schema, operation/attribute
  wires, all geometry plus ceil mode, topology, descriptors/states, feeds/targets, ABI, and splats.
  Version 12 safely misses or rejects; add no alias or migration.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [descriptors glossary](../../../../glossary.md#term-descriptor); [targets glossary](../../../../glossary.md#term-target); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Retain ABI 4, 160 bytes, thirteen exports, the current MPSGraph candidate, and all existing
  resource/lifecycle rules.<span data-glossary-navigation="true"> ([MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>

## Files and symbols

Production remains [Metal](../../../../glossary.md#term-metal)-private: `MetalCapabilityProvider`, `MetalMpsGraphProgram` (`NodeKind`,
`AttributeKind`, [`Node`](../../../../glossary.md#node), `SCHEMA_VERSION`), `MetalNativeApi.MpsGraphExecutableAbi`,
`MetalNegPartitionPreparer`, `MetalNegRouteCandidateGenerator`, `MetalNegTuningBatch`,
`MetalNegTuningCodec`, Metal `package-info.java`, and
`native/metal-macos-arm64/src/synaptik_metal_foundation.m`.

Extend existing capability, raw-[ABI](../../../../glossary.md#term-application-binary-interface-abi), prepared-execution, [identity](../../../../glossary.md#term-identity-meanings)/[codec](../../../../glossary.md#term-codec),
`MetalNegCapabilityPartitionConformanceTest`, and `EngineExplicitCompositionMetalIntegrationTest`
[owners](../../../../glossary.md#term-owner). Add one focused native max-pool numerical [owner](../../../../glossary.md#term-owner) only if no existing native test can own the
single frozen corpus without duplicating [lifecycle](../../../../glossary.md#lifecycle) [matrices](../../../../glossary.md#term-matrix-meanings). [Model](../../../../glossary.md#term-model), [Compiler](../../../../glossary.md#term-compiler), [CPU](../../../../glossary.md#term-cpu), Config, [Planning](../../../../glossary.md#term-planning),
[Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime), [Backend](../../../../glossary.md#backend) Contract, [Trace](../../../../glossary.md#trace), [Engine](../../../../glossary.md#term-engine), Training, and transfer production are read-only.

## Acceptance criteria

1. The sole gate passes before production edits or blocks the task without them.
2. Both profiles admit exactly the no-grad canonical positive static rank-four FLOAT32 domain and
   every documented valid attribute geometry; all exclusions reject before native graph creation.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [rank glossary](../../../../glossary.md#term-rank); [FLOAT32 glossary](../../../../glossary.md#term-float32); [graph glossary](../../../../glossary.md#graph))</span>
3. Direct one-output [lowering](../../../../glossary.md#lowering) preserves the exact [Model](../../../../glossary.md#term-model) result policy without returned indices.
4. Schema 12/wire 20/POOL2D 7 and identities 13 fail closed while ABI 4, 160 bytes, thirteen exports,
   transfer, publication, and lifecycle remain unchanged.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([Schema glossary](../../../../glossary.md#term-schema); [fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [publication glossary](../../../../glossary.md#term-publication); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
5. Focused structural, ABI, capability, conformance, prepared-lifecycle, and one CPU-free public
   Engine proof pass without another winner corpus or device matrix.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine); [device glossary](../../../../glossary.md#term-device); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
6. Current documentation, clean diff, one final build, and independent [lean](../../../../glossary.md#proof-assistant-and-model-checking-tools) Class C review pass.

## [Validation](../../../../glossary.md#term-validation)

Worker validation uses the task-owned one-[run](../../../../glossary.md#term-run) gate, then focused [Metal](../../../../glossary.md#term-metal) capability/raw-[ABI](../../../../glossary.md#term-application-binary-interface-abi)/prepared/
[identity](../../../../glossary.md#term-identity-meanings) suites, `MetalNegCapabilityPartitionConformanceTest`, one ordinary [CPU](../../../../glossary.md#term-cpu)-free
`EngineExplicitCompositionMetalIntegrationTest` forward, native build/export audit, Metal [Javadoc](../../../../glossary.md#term-javadoc),
architecture tests, and `git diff --check`. Run `./gradlew build` exactly once after stabilization.
The independent review reuses successful executable evidence and performs no [device](../../../../glossary.md#term-device) rerun unless a
new concrete risk first changes this plan.

## Documentation and review impact

After executable behavior stabilizes, update current [Metal](../../../../glossary.md#term-metal)/[backend](../../../../glossary.md#backend)/[API](../../../../glossary.md#term-application-programming-interface-api)/glossary status only where
implemented behavior changes it. [Model](../../../../glossary.md#term-model) and [Compiler](../../../../glossary.md#term-compiler) semantics remain unchanged. Class C requires an
independent targeted implementation/documentation review.

## Result

The only permitted direct-selector execution failed one [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero) cell and all four [NaN](../../../../glossary.md#term-nan)-class
cells. Task 0030 is Blocked without production changes or retained probe [artifacts](../../../../glossary.md#term-artifact).
The brief remains one [atomic](../../../../glossary.md#resource-transaction-and-rollback) historical record despite exceeding the 200-line [planning](../../../../glossary.md#term-planning) [target](../../../../glossary.md#term-target)
because its prospective contract and non-retry blocker evidence must remain auditable together.
