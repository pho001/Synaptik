# Task 0080: Dense-[view](../../../../glossary.md#term-view-meanings) [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) composition for production consumers

## Reader context

A dense logical view can still refer to ordinary row-major storage. MATMUL needs the selected buffer plus effective strides, offset and span. Rejecting every `view=true` loses legal composition; accepting any equal-sized ancestor reads the wrong values. A transpose elided into source addressing must apply exactly once, including for other consumers/publication.

Compiler owns generated derivatives; Metal authenticates each original-typed occurrence. MLP/CNN training/save-load receipts below are component evidence, not workaround graphs or joined release approval. The earlier failed aggregate remains failed; the separate RESHAPE packet owns the materialization repair completing those consumers.

- [Current dense-view operands and generated derivatives](../../../../backend-guide/metal-backend.md#general-static-matmul-and-generated-gradients) — Explains operand and actual backward boundaries.
- [Current logical versus physical source checks](../../../../backend-guide/metal-backend.md#physical-authentication-and-fixed-routes) — Connects truthful views to actual buffer selection.

## Status

Complete — exact reviewed dense-[view](../../../../glossary.md#term-view-meanings) prerequisites and the explicit Metal0081 RESHAPE extension
are integrated at identity38. Source-matched [GPU](../../../../glossary.md#term-gpu) consumers, current [conformance](../../../../glossary.md#term-conformance) and independent
final source/evidence/documentation review passed. Whole-[release](../../../../glossary.md#term-release) acceptance remains separate.

## Change class

Class C — [backend capability](../../../../glossary.md#backend-capability), logical/physical [view](../../../../glossary.md#term-view-meanings) [authentication](../../../../glossary.md#term-authentication), native program [validation](../../../../glossary.md#term-validation),
compatibility [identity](../../../../glossary.md#term-identity-meanings) and actual generated-[gradient](../../../../glossary.md#term-gradient) consumer closure.

## Goal and observed failure

Enable ordinary supported dense reshape/flatten -> [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) composition without consumer copies,
erased [alias](../../../../glossary.md#aliasing-and-arena) [metadata](../../../../glossary.md#term-metadata), [CPU](../../../../glossary.md#term-cpu) fallback or a [CNN](../../../../glossary.md#term-convolutional-neural-network-cnn)-specific path. Parent built/signed/verified the current
native source, extracted/reverified its ZIP outside checkout, and ran a [Metal](../../../../glossary.md#term-metal)-only public [linear](../../../../glossary.md#term-linear-meanings)
[Model](../../../../glossary.md#term-model) compute/[SGD](../../../../glossary.md#term-stochastic-gradient-descent-sgd)/save/load journey successfully. A [Conv2d](../../../../glossary.md#convolution--conv2d) -> RELU -> reshape -> linear Model then
failed configureTraining at forward `nodes[4] MATMUL`: no hard-eligible [backend](../../../../glossary.md#backend).
A separate nontrivial biased [ReLU](../../../../glossary.md#term-relu) [MLP](../../../../glossary.md#term-multilayer-perceptron-mlp), `[B=2,F=3] -> H=4 -> O=2`, matches independent [scalar](../../../../glossary.md#term-scalar)
forward values on Metal, then configureTraining rejects generated `nodes[21] MATMUL`. Retain
`mlp-metal-before.log` and unchanged `MlpProductionSmoke.java` beside the CNN evidence. The same
public MLP passes both CPU [dtypes](../../../../glossary.md#data-type--datatype), all four [parameter](../../../../glossary.md#term-parameter) updates and save/close/load/fresh training;
CPU passage does not qualify the Metal backward closure.

[[INFERENCE](../../../../glossary.md#term-inference)] Source review identifies dense zero-offset reshape geometry with `view=true`, rejected
by [canonical](../../../../glossary.md#term-canonical-form)-non-[view](../../../../glossary.md#term-view-meanings)-or-local-[transpose](../../../../glossary.md#term-transpose)-only predicates; the failure [log](../../../../glossary.md#term-log) does not print that [layout](../../../../glossary.md#layout).
Generated ordinary MATMUL [AD](../../../../glossary.md#autograd) exposes dense
views and transposes rooted in those views again. Do not rerun the failing scenario merely to
confirm; retain `/var/folders/z2/kdt1_wq1213732bs4wkhfrh40000gq/T/synaptik-production-run.gloed_6d/metal-cnn-runtime.log`
and the unchanged parent `MetalCnnSmoke.java` for after-fix proof.

## Bounded change and non-goals

Retain fully static positive operand [rank](../../../../glossary.md#term-rank) `1..16`, exact [contraction](../../../../glossary.md#contraction-dimension), [vector](../../../../glossary.md#term-vector) promotion, [MATMUL](../../../../glossary.md#matrix-multiplication--matmul)
[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) [broadcasting](../../../../glossary.md#broadcasting) and checked byte/[dispatch](../../../../glossary.md#term-dispatch) bounds. Admit exact [canonical](../../../../glossary.md#term-canonical-form) [row-major](../../../../glossary.md#term-row-major) [strides](../../../../glossary.md#term-stride),
zero offset and [referenced span](../../../../glossary.md#referenced-element-span) equal to logical [element](../../../../glossary.md#term-element) count, whether logical `isView` is true
or false. Output remains canonical. Reject zero-stride/broadcast operand storage, not legal
MATMUL batch broadcasting. Non-dense [transpose](../../../../glossary.md#term-transpose) still requires an actual earlier local PERMUTE
with [identity](../../../../glossary.md#term-identity-meanings) prefix and the final two [axes](../../../../glossary.md#term-axis) exchanged, rooted in an authenticated dense value;
retain exact source/[shape](../../../../glossary.md#shape)/[carrier](../../../../glossary.md#term-storage-carrier)/[gradient](../../../../glossary.md#term-gradient)/span relationships.

Authenticate actual [descriptors](../../../../glossary.md#term-descriptor) in provider, preparer, independent Java/native [validation](../../../../glossary.md#term-validation) and
route [planning](../../../../glossary.md#term-planning). `MetalMpsGraphProgram.NodeKind.accepts` and `ValueDescriptor` state are part of
this boundary: dense external `MATERIALIZED_LAYOUT` feeds and dense local `AFFINE_VIEW` results
must work without blanket [admission](../../../../glossary.md#term-admission) of those states. Java's canonical feed initialization must
not bypass checks that native declared-state initialization enforces. Do not relabel logical
descriptors, erase [view](../../../../glossary.md#term-view-meanings) [provenance](../../../../glossary.md#provenance) or clear physical-state facts merely to pass dispatch.

Prove the selected [buffer](../../../../glossary.md#term-buffer) and its effective strides/offset/span together. An affine [operation](../../../../glossary.md#operation)
executed into its existing dense result buffer keeps that represented order. An authenticated
transpose whose execution is already elided instead binds the authenticated source buffer and
applies transposed addressing exactly once. Keep source mapping, physical [normalization](../../../../glossary.md#term-normalization), nested
[MPSGraph](../../../../glossary.md#term-mpsgraph) reconstruction and transpose [publication](../../../../glossary.md#term-publication)/non-MATMUL [fan-out](../../../../glossary.md#term-fan-out) consistent. Do not follow
arbitrary logical ancestry just because element counts agree; no new view-elision optimization.

Reuse existing affine execution, MATMUL geometry/type helpers, [kernels](../../../../glossary.md#kernel), gradients and source
mapping. No duplicate mathematics, fallback, public [API](../../../../glossary.md#term-application-programming-interface-api)/type, wrapper/[registry](../../../../glossary.md#term-registry), consumer [CONTIGUOUS](../../../../glossary.md#term-contiguous),
synthetic [layout](../../../../glossary.md#layout), general strided/offset/overlap storage [widening](../../../../glossary.md#term-widening), extra copy or changed [Model](../../../../glossary.md#term-model)/
[Compiler](../../../../glossary.md#term-compiler) view semantics. Independent Java/native checks remain necessary; factor repeated local
geometry through the existing [owner](../../../../glossary.md#term-owner). Keep provenance, physical representation and elision separate.

Independently qualify original typed [FLOAT32](../../../../glossary.md#term-float32), [BFLOAT16](../../../../glossary.md#term-bfloat16) and [FLOAT16](../../../../glossary.md#term-float16) forward/actual-gradient rows.
Do not widen the frozen FLOAT32 proxy to claim new low admission. Retain no-gradient promoted
[INT32](../../../../glossary.md#data-type--datatype)/[INT64](../../../../glossary.md#data-type--datatype) and existing BFLOAT16/FLOAT32 widening orders. Do not add FLOAT16/FLOAT32 widening,
[FLOAT64](../../../../glossary.md#term-float64) MATMUL, mixed BFLOAT16/FLOAT16 or integral/widening gradients. Every low-containing
[partition](../../../../glossary.md#partition) remains exclusively `CUSTOM_PROGRAM`, with custom low-valued steps and no generated
[pointwise](../../../../glossary.md#term-pointwise) or anchor-[epilogue](../../../../glossary.md#term-epilogue) [fusion](../../../../glossary.md#term-fusion). Preserve historical v1 evidence; update current [conformance](../../../../glossary.md#term-conformance).

Advance the coordinated identities from 36 to 37: `MetalNegTuningBatch` candidate/compatibility/
route-policy, `MetalNegRouteCandidateGenerator` workload/exact-policy and `MetalNegTuningCodec`
[codec](../../../../glossary.md#term-codec). Reject stale-36 tuning handoffs/decisions. Retain [schema](../../../../glossary.md#term-schema) 20, its 124-byte header, [ABI](../../../../glossary.md#term-application-binary-interface-abi) 7 and
thirteen exports where bytes/[signatures](../../../../glossary.md#term-signature) are unchanged; no new image/operation/route fields.
Preserve source/[manifest](../../../../glossary.md#term-manifest)/package [authentication](../../../../glossary.md#term-authentication). Retain direct/nested [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) FLOAT32 policy
where applicable, existing authenticated-transpose custom handling and non-direct/low custom
handling. Do not broaden general `canonicalStorage` or canonical-only anchor recognition.

## Contracts

- [Root authority/index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts):
  Core invariants, lifecycle, concrete-backend ownership and numerical/compatibility boundaries.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [backend ownership glossary](../../../../glossary.md#backend-ownership))</span>
- [Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [backend glossary](../../../../glossary.md#backend))</span>,
  Low-precision capability evidence and Profile-free backend capability and identity: exact
  occurrence support, independent dtype qualification, no hidden fallback or numerical selector.<span data-glossary-navigation="true"> ([precision glossary](../../../../glossary.md#term-precision); [backend capability glossary](../../../../glossary.md#backend-capability); [identity glossary](../../../../glossary.md#term-identity-meanings); [occurrence glossary](../../../../glossary.md#term-occurrence); [dtype glossary](../../../../glossary.md#data-type--datatype); [qualification glossary](../../../../glossary.md#term-qualification))</span>
- [Runtime/Prepare/Engine](../../../../architecture/contracts/runtime-prepare-engine.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine))</span>:
  existing logical/physical resources, complete preparation rollback, run leases and publication.<span data-glossary-navigation="true"> ([rollback glossary](../../../../glossary.md#term-rollback); [run glossary](../../../../glossary.md#term-run); [leases glossary](../../../../glossary.md#term-lease); [publication glossary](../../../../glossary.md#term-publication))</span>
- Owning Metal operand-domain wording must explicitly include this dense-view extension before
  implementation; a provider-only guard relaxation is incomplete and unauthorized.<span data-glossary-navigation="true"> ([view glossary](../../../../glossary.md#term-view-meanings); [guard glossary](../../../../glossary.md#term-guard))</span>

## Dependencies and integration

- Depends on: Metal0079, reviewed source-matched native baseline and completed practical RNN;
  independent bounded operand/identity design, coordinated authority and exact Ready verification.
  CPU0078/general-Scan source is not a technical prerequisite; final release integration joins both.<span data-glossary-navigation="true"> ([RNN glossary](../../../../glossary.md#term-recurrent-neural-network-rnn); [identity glossary](../../../../glossary.md#term-identity-meanings); [Ready glossary](../../../../glossary.md#term-ready); [Scan glossary](../../../../glossary.md#term-cumulative-scan); [release glossary](../../../../glossary.md#term-release))</span>
- Conflicts with: other Metal capability/preparer/native/route/identity writers and native package
  publication. CPU0078 backend source is disjoint; shared authorities and ledger integration are MAIN-only.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [publication glossary](../../../../glossary.md#term-publication); [backend glossary](../../../../glossary.md#backend))</span>
- Shared-file exception: LowPrecisionCapabilityLedgerTest, its canonical occurrence/basis registry
  and low-precision-capability-ledger-v2.tsv are not backend-disjoint. CPU0078's current isolated
  migration integrates first. Metal source writers MUST NOT edit these files. After both cutovers,
  MAIN serially integrates new original-typed Metal basis cases and renders the combined current
  CPU+Metal snapshot, preserving v1 and the accepted CPU lock. No independently rendered overwrite.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [occurrence glossary](../../../../glossary.md#term-occurrence); [registry glossary](../../../../glossary.md#term-registry); [precision glossary](../../../../glossary.md#term-precision); [CPU glossary](../../../../glossary.md#term-cpu); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- Parallel group: RELEASE-METAL80. Two isolated source slices on the same frozen accepted base:
  Java admission/state/route/identity plus non-native JVM/integration tests excluding shared ledger
  files; native .m validation/actual-source mapping plus MetalMpsGraphRawAbiNativeTest and
  MetalMpsGraphMatmulNativeTest. Native wire fields remain unchanged.<span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission); [JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm); [validation glossary](../../../../glossary.md#term-validation))</span>
- Common base revision: HEAD `07a281edd3086efde2853211b71b48288db027e1` plus the exact accepted
  dirty source/authority snapshot `metal0080-base.json`, captured by MAIN for both isolated writers.
  Native wire/schema/header/ABI and all Java/native shared field meanings stay unchanged.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>
- Integration order: authority/Ready -> complete disjoint Java/native source -> independent
  code/native review -> MAIN SHA integration -> source-matched rebuild/package and unchanged
  real-device consumers -> affected closure. No child builds/checks while source is in flight.<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device))</span>
- Integration validation: focused capability/Java-native parity/conformance/Engine workflow and
  rebuilt extracted native package; one final production repository checkpoint joins all changes.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): MAIN.

## Files and symbols

- MetalCapabilityProvider.matmulInput and existing dense/[layout](../../../../glossary.md#layout) geometry helpers.
- MetalNegPartitionPreparer affine/[MATMUL](../../../../glossary.md#matrix-multiplication--matmul) state and physical-layout [authentication](../../../../glossary.md#term-authentication).
- MetalMpsGraphProgram.NodeKind.accepts and ValueDescriptor logical/physical state semantics.
- MetalNativeApi independent program [metadata](../../../../glossary.md#term-metadata) and [transpose](../../../../glossary.md#term-transpose)-source [validation](../../../../glossary.md#term-validation).
- MetalPointwiseFusionPlanner route eligibility for authenticated dense/transposed operands.
- macos-[arm64](../../../../glossary.md#native-packaging-and-platform-vocabulary)/src/synaptik_metal_foundation.m matching independent decoder and actual-source mapping.
- Current coordinated [identity](../../../../glossary.md#term-identity-meanings)/[codec](../../../../glossary.md#term-codec)/catalog [owners](../../../../glossary.md#term-owner) and focused tests affected by domain change.
- Actual public [Model](../../../../glossary.md#term-model)/[Engine](../../../../glossary.md#term-engine) integration [regression](../../../../glossary.md#term-regression) for repeated [MLP](../../../../glossary.md#term-multilayer-perceptron-mlp)/[CNN](../../../../glossary.md#term-convolutional-neural-network-cnn) training and save/load.
- MAIN owns root/scoped/[backend](../../../../glossary.md#backend)/[API](../../../../glossary.md#term-application-programming-interface-api)/user/[release](../../../../glossary.md#term-release) docs and recorded native evidence.

## Acceptance criteria

- Non-singleton/non-square dense views on the left, right and both operands; dense boundary feeds,
  local affine values, transpose-of-dense and transpose-back forms. Actual Compiler-generated
  gradients of both ordinary MATMUL operands match independent numerical expectations.<span data-glossary-navigation="true"> ([singleton glossary](../../../../glossary.md#term-singleton); [views glossary](../../../../glossary.md#term-view-meanings); [transpose glossary](../../../../glossary.md#term-transpose); [Compiler glossary](../../../../glossary.md#term-compiler); [gradients glossary](../../../../glossary.md#term-gradient); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>
- Unchanged parent Metal-only Conv2d/RELU/reshape/linear scenario matches independent scalar
  forward, all weight/bias SGD updates, changed inputs, save/close/load and fresh configured train.
  Do not remove the head, reduce gradient targets or add a copy to make this pass.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [Conv2d glossary](../../../../glossary.md#convolution--conv2d); [linear glossary](../../../../glossary.md#term-linear-meanings); [scalar glossary](../../../../glossary.md#term-scalar); [weight glossary](../../../../glossary.md#term-weight); [bias glossary](../../../../glossary.md#term-bias); [SGD glossary](../../../../glossary.md#term-stochastic-gradient-descent-sgd); [targets glossary](../../../../glossary.md#term-target))</span>
- ReLU MLP with nontrivial dimensions/repeated updates and restored weights passes through the
  declared Metal-only owner; mixed execution is qualified separately, not mistaken for GPU-only proof.<span data-glossary-navigation="true"> ([ReLU glossary](../../../../glossary.md#term-relu); [MLP glossary](../../../../glossary.md#term-multilayer-perceptron-mlp); [dimensions glossary](../../../../glossary.md#dimension); [owner glossary](../../../../glossary.md#term-owner); [GPU glossary](../../../../glossary.md#term-gpu))</span>
- Homogeneous FLOAT32, BFLOAT16 and FLOAT16 dense-view forward and actual generated-gradient
  occurrences qualify independently on original typed descriptors. Preserve the other typed
  bounds above and do not expand Java CPU low arithmetic.<span data-glossary-navigation="true"> ([Homogeneous glossary](../../../../glossary.md#term-homogeneous); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT16 glossary](../../../../glossary.md#term-float16); [descriptors glossary](../../../../glossary.md#term-descriptor); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Java-only and raw-native rejection of forged provenance/source/stride/span/offset metadata,
  holes, zero-stride operand broadcasting and overlap precedes target writes.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [stride glossary](../../../../glossary.md#term-stride); [metadata glossary](../../../../glossary.md#term-metadata); [broadcasting glossary](../../../../glossary.md#broadcasting))</span>
- Preserve existing transpose fan-out/publication/custom/MPSGraph regression domains and the
  separate unsupported overlapping convolution input-gradient case. Parameter-only CNN updates
  do not require an unselected input-cotangent fold.<span data-glossary-navigation="true"> ([fan-out glossary](../../../../glossary.md#term-fan-out); [publication glossary](../../../../glossary.md#term-publication); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [regression glossary](../../../../glossary.md#term-regression); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [Parameter glossary](../../../../glossary.md#term-parameter); [CNN glossary](../../../../glossary.md#term-convolutional-neural-network-cnn); [cotangent glossary](../../../../glossary.md#cotangent))</span>
- No extra consumer copy or retained resource; logical view descriptors stay truthful. Reuse
  existing kernels and keep each authentication responsibility readable and narrowly named.<span data-glossary-navigation="true"> ([kernels glossary](../../../../glossary.md#kernel); [authentication glossary](../../../../glossary.md#term-authentication))</span>
- Current [identity](../../../../glossary.md#term-identity-meanings) 37 accepts the extended domain; stale-36 tuning [handoff](../../../../glossary.md#term-handoff)/decision inputs reject.
- Root and scoped authority are updated together before Ready; this Draft does not grant code
  admission or claim current source implements the extended operand domain.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready); [admission glossary](../../../../glossary.md#term-admission))</span>

## [Validation](../../../../glossary.md#term-validation)

No build/lint/tests/formatters mid-flight. MAIN runs targeted Java/native gates and rebuilds,
signs, packages, extracts and verifies the final exact native source once after code review.
Exercise actual [device](../../../../glossary.md#term-device) and unchanged consumer smokes; preserve original failures. Optional
instrumented [observer](../../../../glossary.md#term-observer) tests use a separate harness, not distributable native bytes. No broad
performance campaign or new numerical policy; final root gate belongs to production acceptance.

## Documentation and review impact

Independent review must [trace](../../../../glossary.md#trace) actual physical source selection across all [validation](../../../../glossary.md#term-validation) layers,
[identity](../../../../glossary.md#term-identity-meanings) invalidation, unsafe forged-[metadata](../../../../glossary.md#term-metadata) rejection, mathematical/[kernel](../../../../glossary.md#kernel) reuse, numerical
consumer evidence and absence of workaround copies/fallbacks. Update public support bounds and
source/package evidence in the same integrated change. Hosted [GPU](../../../../glossary.md#term-gpu) [CI](../../../../glossary.md#term-continuous-integration-ci) remains a separate provider gate.

## MAIN [Ready](../../../../glossary.md#term-ready) [admission](../../../../glossary.md#term-admission)

MAIN verified Complete Metal0079/practical [RNN](../../../../glossary.md#term-recurrent-neural-network-rnn) and current source-matched native package/[device](../../../../glossary.md#term-device)
baseline, retained actual [MLP](../../../../glossary.md#term-multilayer-perceptron-mlp)/[CNN](../../../../glossary.md#term-convolutional-neural-network-cnn) failures, independent bounded design and final root/scoped/
[decomposition](../../../../glossary.md#term-decomposition) APPROVE0, no other [Metal](../../../../glossary.md#term-metal) source writer, and explicit nonoverlapping Java/native
file [ownership](../../../../glossary.md#term-ownership-meanings). Shared ledger test/basis/resource remain excluded and integrate serially under MAIN.
The two clean contexts use the same full dirty-source [manifest](../../../../glossary.md#term-manifest), not HEAD-only copies. Source work
may proceed without waiting for CPU0078's disjoint test migration; no native package or capability
completion is claimed. Workers skip all builds/checks; MAIN integrates reviewed bytes then runs
`:backends:metal:test`, Javadocs and affected [conformance](../../../../glossary.md#term-conformance)/integration against the rebuilt extracted
library with `SYNAPTIK_METAL_TEST_LIBRARY`, including both assigned raw/[MATMUL](../../../../glossary.md#matrix-multiplication--matmul) native classes.
Actual unchanged public MLP/CNN and each declared original-typed row remain required proof.

## Result

MAIN integrated the exact reviewed37 prerequisite bytes together with the independently reviewed
Metal0081 final38 extension. The earlier37 aggregate typed-[gradient](../../../../glossary.md#term-gradient) failure remains FAILED;
ordinary materializing RESHAPE was corrected in its explicit owning packet, not hidden with
[CONTIGUOUS](../../../../glossary.md#term-contiguous), smaller [targets](../../../../glossary.md#term-target), handwritten [derivatives](../../../../glossary.md#term-derivative) or a consumer [graph](../../../../glossary.md#graph) change.

The new exact native source was built/signed/packaged, clean ZIP-extracted outside checkout and
verified. Full [Metal](../../../../glossary.md#term-metal)/[Javadoc](../../../../glossary.md#term-javadoc)/jar passed376tests with10opt-in [observer](../../../../glossary.md#term-observer) skips; all16actual typed
integration cases passed, including unchanged all15F32/[BF16](../../../../glossary.md#floating-point-precision-and-rounding)/[F16](../../../../glossary.md#floating-point-precision-and-rounding) five-form forward/both-gradient
graphs. Unchanged [F32](../../../../glossary.md#floating-point-precision-and-rounding) public [MLP](../../../../glossary.md#term-multilayer-perceptron-mlp)/[CNN](../../../../glossary.md#term-convolutional-neural-network-cnn) passed [scalar](../../../../glossary.md#term-scalar)-[oracle](../../../../glossary.md#term-test-oracle) forward/all-[parameter](../../../../glossary.md#term-parameter) [SGD](../../../../glossary.md#term-stochastic-gradient-descent-sgd), save/close/
load and fresh training. Metal-only ordinary raw reshape additionally passed all seven carriers.

MAIN's sole actual provider renderer produced306current-v2rows: all290oldrows remain literally
unchanged; sixteen added rows describe eight distinct original-typed occurrences per [backend](../../../../glossary.md#backend).
Every new Metal F32/BF16/F16 answer is independently true; [CPU](../../../../glossary.md#term-cpu) numeric low stays false and raw
reshape stays true. Historical v1 and complete CPU [owner](../../../../glossary.md#term-owner) inventory are literal unchanged.
The full affected [conformance](../../../../glossary.md#term-conformance) gate passed78tests after deleting an obsolete [identity](../../../../glossary.md#term-identity-meanings)-only
reflection/copy test, not repinning36to38. Its first79-test failed gate is retained.

[Metal0081 Result](0081-materializing-reshape-composition.md#result) and the [run](../../../../glossary.md#term-run)'s exact-source,
current-ledger [authentication](../../../../glossary.md#term-authentication), copied XML and consumer receipts own the evidence and limits.
No full repository/[release](../../../../glossary.md#term-release) readiness or official trusted prebuilt [distribution](../../../../glossary.md#term-distribution) is inferred.

Independent final source/evidence/docs review returned APPROVE0 after closing the current-identity
and qualified [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) state descriptions. This freezes the declared Metal production [matrix](../../../../glossary.md#term-matrix-meanings), not
native-memory plateau, official prebuilt trust or joined repository/release acceptance.
