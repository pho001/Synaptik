# Task 0080: Dense-view MATMUL composition for production consumers

## Status

Complete — exact reviewed dense-view prerequisites and the explicit Metal0081 RESHAPE extension
are integrated at identity38. Source-matched GPU consumers, current conformance and independent
final source/evidence/documentation review passed. Whole-release acceptance remains separate.

## Change class

Class C — backend capability, logical/physical view authentication, native program validation,
compatibility identity and actual generated-gradient consumer closure.

## Goal and observed failure

Enable ordinary supported dense reshape/flatten -> MATMUL composition without consumer copies,
erased alias metadata, CPU fallback or a CNN-specific path. Parent built/signed/verified the current
native source, extracted/reverified its ZIP outside checkout, and ran a Metal-only public linear
Model compute/SGD/save/load journey successfully. A Conv2d -> RELU -> reshape -> linear Model then
failed configureTraining at forward `nodes[4] MATMUL`: no hard-eligible backend.
A separate nontrivial biased ReLU MLP, `[B=2,F=3] -> H=4 -> O=2`, matches independent scalar
forward values on Metal, then configureTraining rejects generated `nodes[21] MATMUL`. Retain
`mlp-metal-before.log` and unchanged `MlpProductionSmoke.java` beside the CNN evidence. The same
public MLP passes both CPU dtypes, all four parameter updates and save/close/load/fresh training;
CPU passage does not qualify the Metal backward closure.

[INFERENCE] Source review identifies dense zero-offset reshape geometry with `view=true`, rejected
by canonical-non-view-or-local-transpose-only predicates; the failure log does not print that layout.
Generated ordinary MATMUL AD exposes dense
views and transposes rooted in those views again. Do not rerun the failing scenario merely to
confirm; retain `/var/folders/z2/kdt1_wq1213732bs4wkhfrh40000gq/T/synaptik-production-run.gloed_6d/metal-cnn-runtime.log`
and the unchanged parent `MetalCnnSmoke.java` for after-fix proof.

## Bounded change and non-goals

Retain fully static positive operand rank `1..16`, exact contraction, vector promotion, MATMUL
batch broadcasting and checked byte/dispatch bounds. Admit exact canonical row-major strides,
zero offset and referenced span equal to logical element count, whether logical `isView` is true
or false. Output remains canonical. Reject zero-stride/broadcast operand storage, not legal
MATMUL batch broadcasting. Non-dense transpose still requires an actual earlier local PERMUTE
with identity prefix and the final two axes exchanged, rooted in an authenticated dense value;
retain exact source/shape/carrier/gradient/span relationships.

Authenticate actual descriptors in provider, preparer, independent Java/native validation and
route planning. `MetalMpsGraphProgram.NodeKind.accepts` and `ValueDescriptor` state are part of
this boundary: dense external `MATERIALIZED_LAYOUT` feeds and dense local `AFFINE_VIEW` results
must work without blanket admission of those states. Java's canonical feed initialization must
not bypass checks that native declared-state initialization enforces. Do not relabel logical
descriptors, erase view provenance or clear physical-state facts merely to pass dispatch.

Prove the selected buffer and its effective strides/offset/span together. An affine operation
executed into its existing dense result buffer keeps that represented order. An authenticated
transpose whose execution is already elided instead binds the authenticated source buffer and
applies transposed addressing exactly once. Keep source mapping, physical normalization, nested
MPSGraph reconstruction and transpose publication/non-MATMUL fan-out consistent. Do not follow
arbitrary logical ancestry just because element counts agree; no new view-elision optimization.

Reuse existing affine execution, MATMUL geometry/type helpers, kernels, gradients and source
mapping. No duplicate mathematics, fallback, public API/type, wrapper/registry, consumer CONTIGUOUS,
synthetic layout, general strided/offset/overlap storage widening, extra copy or changed Model/
Compiler view semantics. Independent Java/native checks remain necessary; factor repeated local
geometry through the existing owner. Keep provenance, physical representation and elision separate.

Independently qualify original typed FLOAT32, BFLOAT16 and FLOAT16 forward/actual-gradient rows.
Do not widen the frozen FLOAT32 proxy to claim new low admission. Retain no-gradient promoted
INT32/INT64 and existing BFLOAT16/FLOAT32 widening orders. Do not add FLOAT16/FLOAT32 widening,
FLOAT64 MATMUL, mixed BFLOAT16/FLOAT16 or integral/widening gradients. Every low-containing
partition remains exclusively `CUSTOM_PROGRAM`, with custom low-valued steps and no generated
pointwise or anchor-epilogue fusion. Preserve historical v1 evidence; update current conformance.

Advance the coordinated identities from 36 to 37: `MetalNegTuningBatch` candidate/compatibility/
route-policy, `MetalNegRouteCandidateGenerator` workload/exact-policy and `MetalNegTuningCodec`
codec. Reject stale-36 tuning handoffs/decisions. Retain schema 20, its 124-byte header, ABI 7 and
thirteen exports where bytes/signatures are unchanged; no new image/operation/route fields.
Preserve source/manifest/package authentication. Retain direct/nested rank-two FLOAT32 policy
where applicable, existing authenticated-transpose custom handling and non-direct/low custom
handling. Do not broaden general `canonicalStorage` or canonical-only anchor recognition.

## Contracts

- [Root authority/index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts):
  Core invariants, lifecycle, concrete-backend ownership and numerical/compatibility boundaries.
- [Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend),
  Low-precision capability evidence and Profile-free backend capability and identity: exact
  occurrence support, independent dtype qualification, no hidden fallback or numerical selector.
- [Runtime/Prepare/Engine](../../../../architecture/contracts/runtime-prepare-engine.md):
  existing logical/physical resources, complete preparation rollback, run leases and publication.
- Owning Metal operand-domain wording must explicitly include this dense-view extension before
  implementation; a provider-only guard relaxation is incomplete and unauthorized.

## Dependencies and integration

- Depends on: Metal0079, reviewed source-matched native baseline and completed practical RNN;
  independent bounded operand/identity design, coordinated authority and exact Ready verification.
  CPU0078/general-Scan source is not a technical prerequisite; final release integration joins both.
- Conflicts with: other Metal capability/preparer/native/route/identity writers and native package
  publication. CPU0078 backend source is disjoint; shared authorities and ledger integration are MAIN-only.
- Shared-file exception: LowPrecisionCapabilityLedgerTest, its canonical occurrence/basis registry
  and low-precision-capability-ledger-v2.tsv are not backend-disjoint. CPU0078's current isolated
  migration integrates first. Metal source writers MUST NOT edit these files. After both cutovers,
  MAIN serially integrates new original-typed Metal basis cases and renders the combined current
  CPU+Metal snapshot, preserving v1 and the accepted CPU lock. No independently rendered overwrite.
- Parallel group: RELEASE-METAL80. Two isolated source slices on the same frozen accepted base:
  Java admission/state/route/identity plus non-native JVM/integration tests excluding shared ledger
  files; native .m validation/actual-source mapping plus MetalMpsGraphRawAbiNativeTest and
  MetalMpsGraphMatmulNativeTest. Native wire fields remain unchanged.
- Common base revision: HEAD `07a281edd3086efde2853211b71b48288db027e1` plus the exact accepted
  dirty source/authority snapshot `metal0080-base.json`, captured by MAIN for both isolated writers.
  Native wire/schema/header/ABI and all Java/native shared field meanings stay unchanged.
- Integration order: authority/Ready -> complete disjoint Java/native source -> independent
  code/native review -> MAIN SHA integration -> source-matched rebuild/package and unchanged
  real-device consumers -> affected closure. No child builds/checks while source is in flight.
- Integration validation: focused capability/Java-native parity/conformance/Engine workflow and
  rebuilt extracted native package; one final production repository checkpoint joins all changes.
- Shared-document integration owner: MAIN.

## Files and symbols

- MetalCapabilityProvider.matmulInput and existing dense/layout geometry helpers.
- MetalNegPartitionPreparer affine/MATMUL state and physical-layout authentication.
- MetalMpsGraphProgram.NodeKind.accepts and ValueDescriptor logical/physical state semantics.
- MetalNativeApi independent program metadata and transpose-source validation.
- MetalPointwiseFusionPlanner route eligibility for authenticated dense/transposed operands.
- macos-arm64/src/synaptik_metal_foundation.m matching independent decoder and actual-source mapping.
- Current coordinated identity/codec/catalog owners and focused tests affected by domain change.
- Actual public Model/Engine integration regression for repeated MLP/CNN training and save/load.
- MAIN owns root/scoped/backend/API/user/release docs and recorded native evidence.

## Acceptance criteria

- Non-singleton/non-square dense views on the left, right and both operands; dense boundary feeds,
  local affine values, transpose-of-dense and transpose-back forms. Actual Compiler-generated
  gradients of both ordinary MATMUL operands match independent numerical expectations.
- Unchanged parent Metal-only Conv2d/RELU/reshape/linear scenario matches independent scalar
  forward, all weight/bias SGD updates, changed inputs, save/close/load and fresh configured train.
  Do not remove the head, reduce gradient targets or add a copy to make this pass.
- ReLU MLP with nontrivial dimensions/repeated updates and restored weights passes through the
  declared Metal-only owner; mixed execution is qualified separately, not mistaken for GPU-only proof.
- Homogeneous FLOAT32, BFLOAT16 and FLOAT16 dense-view forward and actual generated-gradient
  occurrences qualify independently on original typed descriptors. Preserve the other typed
  bounds above and do not expand Java CPU low arithmetic.
- Java-only and raw-native rejection of forged provenance/source/stride/span/offset metadata,
  holes, zero-stride operand broadcasting and overlap precedes target writes.
- Preserve existing transpose fan-out/publication/custom/MPSGraph regression domains and the
  separate unsupported overlapping convolution input-gradient case. Parameter-only CNN updates
  do not require an unselected input-cotangent fold.
- No extra consumer copy or retained resource; logical view descriptors stay truthful. Reuse
  existing kernels and keep each authentication responsibility readable and narrowly named.
- Current identity 37 accepts the extended domain; stale-36 tuning handoff/decision inputs reject.
- Root and scoped authority are updated together before Ready; this Draft does not grant code
  admission or claim current source implements the extended operand domain.

## Validation

No build/lint/tests/formatters mid-flight. MAIN runs targeted Java/native gates and rebuilds,
signs, packages, extracts and verifies the final exact native source once after code review.
Exercise actual device and unchanged consumer smokes; preserve original failures. Optional
instrumented observer tests use a separate harness, not distributable native bytes. No broad
performance campaign or new numerical policy; final root gate belongs to production acceptance.

## Documentation and review impact

Independent review must trace actual physical source selection across all validation layers,
identity invalidation, unsafe forged-metadata rejection, mathematical/kernel reuse, numerical
consumer evidence and absence of workaround copies/fallbacks. Update public support bounds and
source/package evidence in the same integrated change. Hosted GPU CI remains a separate provider gate.

## MAIN Ready admission

MAIN verified Complete Metal0079/practical RNN and current source-matched native package/device
baseline, retained actual MLP/CNN failures, independent bounded design and final root/scoped/
decomposition APPROVE0, no other Metal source writer, and explicit nonoverlapping Java/native
file ownership. Shared ledger test/basis/resource remain excluded and integrate serially under MAIN.
The two clean contexts use the same full dirty-source manifest, not HEAD-only copies. Source work
may proceed without waiting for CPU0078's disjoint test migration; no native package or capability
completion is claimed. Workers skip all builds/checks; MAIN integrates reviewed bytes then runs
`:backends:metal:test`, Javadocs and affected conformance/integration against the rebuilt extracted
library with `SYNAPTIK_METAL_TEST_LIBRARY`, including both assigned raw/MATMUL native classes.
Actual unchanged public MLP/CNN and each declared original-typed row remain required proof.

## Result

MAIN integrated the exact reviewed37 prerequisite bytes together with the independently reviewed
Metal0081 final38 extension. The earlier37 aggregate typed-gradient failure remains FAILED;
ordinary materializing RESHAPE was corrected in its explicit owning packet, not hidden with
CONTIGUOUS, smaller targets, handwritten derivatives or a consumer graph change.

The new exact native source was built/signed/packaged, clean ZIP-extracted outside checkout and
verified. Full Metal/Javadoc/jar passed376tests with10opt-in observer skips; all16actual typed
integration cases passed, including unchanged all15F32/BF16/F16 five-form forward/both-gradient
graphs. Unchanged F32 public MLP/CNN passed scalar-oracle forward/all-parameter SGD, save/close/
load and fresh training. Metal-only ordinary raw reshape additionally passed all seven carriers.

MAIN's sole actual provider renderer produced306current-v2rows: all290oldrows remain literally
unchanged; sixteen added rows describe eight distinct original-typed occurrences per backend.
Every new Metal F32/BF16/F16 answer is independently true; CPU numeric low stays false and raw
reshape stays true. Historical v1 and complete CPU owner inventory are literal unchanged.
The full affected conformance gate passed78tests after deleting an obsolete identity-only
reflection/copy test, not repinning36to38. Its first79-test failed gate is retained.

[Metal0081 Result](0081-materializing-reshape-composition.md#result) and the run's exact-source,
current-ledger authentication, copied XML and consumer receipts own the evidence and limits.
No full repository/release readiness or official trusted prebuilt distribution is inferred.

Independent final source/evidence/docs review returned APPROVE0 after closing the current-identity
and qualified MATMUL state descriptions. This freezes the declared Metal production matrix, not
native-memory plateau, official prebuilt trust or joined repository/release acceptance.
