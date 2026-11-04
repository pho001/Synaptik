# Repository review — findings and repair record, 2026-10-13

## [Scope](../glossary.md#term-scope), method and limits

This is an engineering review report and implementation record, not a numerical [certificate](../glossary.md#formal-proof-certificate-and-trusted-assumptions) or
an [architecture contract](../glossary.md#architecture-contract). The [review brief](project-review-2026-10-13.md) defines [ownership](../glossary.md#term-ownership-meanings) and
acceptance. Findings refer to production baseline `728680eaabc44a485da9f32050914ca4f09b295c`;
source line numbers below are baseline locations and move after repairs unless a later inspected
revision is explicitly identified.

Six independent review areas covered all 22 [Gradle](../glossary.md#term-gradle) leaf projects, native [Metal](../glossary.md#term-metal)/[CUDA](../glossary.md#term-cuda) sources,
build wiring and all 23 area master plans. Review combined source tracing, existing focused-test
inspection, small reproductions, contract comparison and [dependency](../glossary.md#term-dependency)/frontier reconciliation.
Nontrivial repairs use isolated implementation contexts and independent code/documentation review.
The coordinator integrates them and owns the combined [validation](../glossary.md#term-validation) [checkpoint](../glossary.md#term-checkpoint).

This was a risk-focused whole-project review, **not a line-by-line proof of every file**. [Operation](../glossary.md#operation)
kind counts do not establish all [dtype](../glossary.md#data-type--datatype)/[layout](../glossary.md#layout)/strategy/[gradient](../glossary.md#term-gradient) combinations. No exhaustive fuzzing,
large-[model](../glossary.md#term-model) memory test, cross-platform [OpenBLAS](../glossary.md#term-openblas) campaign, general performance comparison or new
historical five-[fork](../glossary.md#term-fork-meanings) benchmark [matrix](../glossary.md#term-review-matrix) was performed. A passing test [run](../glossary.md#term-run) is not a correctness proof.

The initial full `./gradlew test javadoc` checkpoint passed before repairs (100 tasks: 4 executed,
4 from [cache](../glossary.md#term-cache), 92 up-to-date). That baseline did not cover the counterexamples found below.
Actual-[device](../glossary.md#term-device) execution is recorded separately from source review and skipped tests.

## Result at a glance

Confirmed implementation findings: **6 P1 and 10 P2**, no confirmed P0. Separately: **G1, a P2
numerical-contract discrepancy requiring an [owner](../glossary.md#term-owner) decision**, two test defects and several
documentation inconsistencies. P1 means a supported path can return a wrong result, fail
for ordinary valid input, or access native [metadata](../glossary.md#term-metadata) out of bounds. P2 covers narrower numerical,
resource, [API](../glossary.md#term-application-programming-interface-api)-consumer or bounded-memory failures. These priorities are repair urgency, not a
formal security classification.

All sixteen implementation findings and the test/documentation defects were repaired in ten
independently reviewed packages. Their integrated [checkpoint](../glossary.md#term-checkpoint) passed. G1 remains open by the
owner's explicit decision to discuss it after this report; the [vector](../glossary.md#term-vector) [ERF](../glossary.md#term-erf) timing/shape follow-up
also remains Draft. This is review-[scope](../glossary.md#term-scope) closure, not a claim of universal numerical or performance
closure.

| ID | Priority | Finding | Repair |
|---|---|---|---|
| A1 | P1 | Batched [matrix](../glossary.md#term-matrix-meanings)–vector right [gradient](../glossary.md#term-gradient) loses the column [axis](../glossary.md#term-axis). | [Compiler](../glossary.md#term-compiler) 0006D |
| A2 | P1 | A connected sibling hides a required unsupported [derivative](../glossary.md#term-derivative) path. | Compiler 0006D |
| B1 | P2 | Duplicate [prepared resource](../glossary.md#prepared-resource--preparedresource) causes trailing unique resources to leak. | [Prepare](../glossary.md#prepare) 0009 |
| [C1](../glossary.md#term-c1) | P2 | Large [finite](../glossary.md#term-finite) ERF/[GELU](../glossary.md#gaussian-error-linear-unit--gelu) can produce [NaN](../glossary.md#term-nan); vector body disagrees with tail. | [CPU](../glossary.md#term-cpu) 0022 |
| [C2](../glossary.md#term-c2) | P1 | [FLOAT16](../glossary.md#term-float16) [softmax](../glossary.md#softmax--log-softmax) validator reads [FLOAT32](../glossary.md#term-float32) storage. | CPU 0021 |
| C3 | P2 | Value [validation](../glossary.md#term-validation) runs before an internal [producer](../glossary.md#term-producer) executes. | CPU 0021 |
| C4 | P1 | Composed [losses](../glossary.md#loss) compare the wrong access representation. | CPU 0021 |
| C5 | P2 | Low [activation](../glossary.md#term-activation) composites use DOUBLE instead of named FLOAT32 sites. | CPU 0022 |
| D1 | P1 | [Rank](../glossary.md#term-rank)-16 WHERE false-input metadata overflows its native array. | [Metal](../glossary.md#term-metal) 0078 |
| D2 | P2 | Low ScatterAdd changes untouched raw words. | Metal 0078 |
| D3 | P2 | Low inclusive [scan](../glossary.md#term-cumulative-scan) [seed](../glossary.md#term-seed) loses negative-zero prefixes. | Metal 0078 |
| D4 | P2 | Native invocation does not enforce exact creating-context [identity](../glossary.md#term-identity-meanings). | Metal 0078 |
| D5 | P1 | Advertised [scalar](../glossary.md#term-scalar) RELU/MUL are rejected by Java [preflight](../glossary.md#term-preflight) and the native decoder. | Metal 0079 |
| E1 | P2 | [Adam](../glossary.md#adaptive-optimizer--adam-adamw) divides before multiplying and rejects a finite documented update. | Training 0002C |
| E2 | P2 | [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) public [Model](../glossary.md#term-model) types are missing from an NN-only consumer's [compile](../glossary.md#compile) path. | NN 0025B |
| F1 | P2 | Model-plan [cache](../glossary.md#term-cache) checks its bound after aggregate allocation. | Tuning 0005 |
| G1 | P2, contract | Stable finite softmax changes the output class implied by literal [EXP](../glossary.md#term-expx) [overflow](../glossary.md#term-overflow). | Model 0033 Draft; owner decision required |
| B2 | P2, test | Several “mixed-[backend](../glossary.md#backend)” tests now [run](../glossary.md#term-run) entirely on Metal. | [Engine](../glossary.md#term-engine) 0019 |
| B3 | P3, test | Copy/close test assumes executor submission guarantees [admission](../glossary.md#term-admission). | Engine 0019 |
| B4 | P3, docs | Public tuning [Javadoc](../glossary.md#term-javadoc) incorrectly describes CPU-only support. | Engine 0019 |

## Detailed production findings

### A1 — Batched matvec [gradient](../glossary.md#term-gradient)

Location: `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/LinearAlgebraGradientRules.java:86`.

For `A[..., M, K] @ x[K]`, the output [cotangent](../glossary.md#cotangent) is `g[..., M]`. The old right-gradient formula
used `transpose(A) @ g` directly. A batched [vector](../glossary.md#term-vector) cotangent then looks like a [matrix](../glossary.md#term-matrix-meanings) operand.
With `A[2,3,4]`, [compilation](../glossary.md#compiler-engine-and-execution-stage-names) rejects inner [dimensions](../glossary.md#dimension) 3 versus 2. With coincident dimensions,
it can silently compute a different [contraction](../glossary.md#contraction-dimension): `A[2,2,2] = 1..8`, unit cotangent gives
`[36,36]` instead of `[16,20]` when evaluating the emitted algebra.

Repair: explicitly form `g[..., M, 1]`, multiply, remove the last [singleton](../glossary.md#term-singleton) [axis](../glossary.md#term-axis) and reduce
[batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) dimensions to `x.shape()`. No new [derivative](../glossary.md#term-derivative) family is introduced. [Regression](../glossary.md#term-regression) evidence
uses independent [host](../glossary.md#term-host)-loop oracles, unequal and coincident dimensions, multiple batch axes,
nonuniform [seeds](../glossary.md#commit-release-distribution-log-and-seed-meanings), both operands and mixed second derivatives, with optimization on/off.

### A2 — Required unsupported [derivative](../glossary.md#term-derivative) is silently omitted

Location: `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/AutogradPreflight.java:506`.

The old [preflight](../glossary.md#term-preflight) treated a [target](../glossary.md#term-target) with any connected [differentiable](../glossary.md#derivative-and-chain-rule) path as acceptable before
checking another required blocked path. It also conflated intentionally non-differentiable roles
with [fail-closed](../glossary.md#term-fail-closed) roles. In a [second derivative](../glossary.md#term-second-derivative) through BatchNorm training, saved [mean](../glossary.md#term-mean)/inverse
[standard deviation](../glossary.md#term-standard-deviation) can require derivatives that are not implemented. A direct sibling path
does not make those contributions optional. For `x=[-1,1]`, [epsilon](../glossary.md#epsilon) 1, the inspected two-stage
formula can yield `[0.7071,0.7071]` where the complete derivative is zero.

Repair: intersect selected-output [cotangent](../glossary.md#cotangent) demand with target reachability; reject a required
fail-closed role even when a sibling connects. Stop demand at intentional [mask](../glossary.md#term-boolean-mask)/index/comparison
boundaries. Ordinary first-order saved-value use remains valid. This is truthful rejection,
**not** implementation of saved-[statistic](../glossary.md#term-statistic) higher derivatives.

### B1 — Resource [rollback](../glossary.md#term-rollback) misses [owners](../glossary.md#term-owner) after a duplicate

Location: `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/BackendPartitionFinalizationHandoff.java:265`.

A finalizer returning resources `[A,A,B]` transfers both unique owners, but the old loop threw
at the second `A` before registering `B`. The diagnostic closed only `A`; expected cleanup is
`B,A`. The same issue occurs across results, e.g. first `[A]`, then `[A,B]`.

Repair: collect every unique [identity](../glossary.md#term-identity-meanings) from the returned result before rejecting its first
duplicate. Do not validate the rejected executable or call later finalizers. Close each [owner](../glossary.md#term-owner)
once in reverse order, preserving the primary error and suppressing distinct cleanup failures.
Tests cover identity versus equality, trailing owners, multiple duplicates and throwing cleanup.

### [C1](../glossary.md#term-c1) — [Activation](../glossary.md#term-activation) polynomial [overflow](../glossary.md#term-overflow) for [finite](../glossary.md#term-finite) input

Locations: `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuScalarEmitter.java:381`
and `.../internal/codegen/emit/CpuVectorMath.java:156`.

[ERF](../glossary.md#term-erf)'s rational-polynomial tail evaluates overflowing numerator and denominator before multiplying
by an exponentially tiny value. `Infinity/Infinity` becomes [NaN](../glossary.md#term-nan); multiplication by zero cannot
repair it. Actual generated [FLOAT64](../glossary.md#term-float64) execution with `±1e100` returned NaN. A [FLOAT32](../glossary.md#term-float32) [vector](../glossary.md#term-vector)
body plus [scalar tail](../glossary.md#term-scalar-tail) over five identical `1e10` inputs returned `[NaN,NaN,NaN,NaN,1]`.
[GELU](../glossary.md#gaussian-error-linear-unit--gelu) inherits the tail problem. The reference shared the defect, so agreement with it was weak evidence.

Repair [scope](../glossary.md#term-scope): a mathematically justified [saturation](../glossary.md#term-saturation) [guard](../glossary.md#term-guard) before unsafe [evaluation](../glossary.md#term-evaluation), independent
special-value tests, consistent body/tail behavior, generated/direct Java shape inspection and
a generated compatibility [identity](../glossary.md#term-identity-meanings) change. Existing NaN/[infinity](../glossary.md#term-infinity)/zero contracts must remain intact.

### [C2](../glossary.md#term-c2) — [FLOAT16](../glossary.md#term-float16) [softmax](../glossary.md#softmax--log-softmax) is decoded with the wrong [carrier](../glossary.md#term-storage-carrier) width

Location: `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuSoftmaxInputValidator.java:36`.

The validator handled [BFLOAT16](../glossary.md#term-bfloat16) specially but let FLOAT16 fall through to [FLOAT32](../glossary.md#term-float32) access.
A `short[]` then causes `ClassCastException`; a native half [buffer](../glossary.md#term-buffer) can receive an out-of-bounds
four-byte read where only two bytes remain. The public [Engine](../glossary.md#term-engine) path reproduced the failure.

Repair: decode FLOAT16 exactly using its existing representation and validate in the declared
working [precision](../glossary.md#term-precision). Cover softmax and [log-softmax](../glossary.md#softmax--log-softmax), arrays/native segments, offsets and [strides](../glossary.md#term-stride).
Do not claim expanded [infinity](../glossary.md#term-infinity)/[NaN](../glossary.md#term-nan) support merely by fixing the storage reader.

### C3 — Value-dependent [validation](../glossary.md#term-validation) reads unproduced intermediates

Locations: `CpuPreparedExecutable.java:1663` and `CpuPreparedPartitionExecutable.java:194`
under `backends/cpu/src/main/java/.../backend/cpu/internal/`.

The composite [partition](../glossary.md#partition) binds children before executing any of them. A [softmax](../glossary.md#softmax--log-softmax)/[loss](../glossary.md#loss)/[mask](../glossary.md#term-boolean-mask) scan
performed during [binding](../glossary.md#term-binding) therefore reads an intermediate [buffer](../glossary.md#term-buffer) before its [producer](../glossary.md#term-producer) has written
the current [run](../glossary.md#term-run)'s values. `[0,1000].exp().softmax()` returned NaNs while direct `[1,Infinity]`
softmax correctly rejected the [backend](../glossary.md#backend)'s [finite](../glossary.md#term-finite)-input domain. Reused storage can similarly
validate stale values rather than this run's data.

Repair: keep structural, [carrier](../glossary.md#term-storage-carrier), range and [layout](../glossary.md#layout) checks at bind time. Perform value-dependent
validation immediately before the consuming unit writes output, after its producers execute,
on the orchestrating path before parallel workers. Audit the same issue for loss indices and
[Boolean](../glossary.md#data-type--datatype)/attention masks. This does not promise that earlier internal producer writes are rolled
back on rejection; [publication](../glossary.md#term-publication) and consumer-write boundaries remain as specified.

### C4 — Composed [loss](../glossary.md#loss) [binding](../glossary.md#term-binding) rejects valid unit [topology](../glossary.md#term-topology)

Locations: `CpuPartitionPreparationPlan.java:1596` and `CpuLossIr.java:131` in [CPU](../glossary.md#term-cpu) internals.

`unitAccessesAgree` compares a loss's [rank-zero](../glossary.md#axis-rank-scalar-vector-and-matrix) [cache](../glossary.md#term-cache)-key projection instead of its actual
boundary accesses. Direct [MSE](../glossary.md#mean-squared-error--mse) over `[1,2]` and `[0,0]` with [MEAN](../glossary.md#term-mean) yields 2.5; inserting a valid
[prediction](../glossary.md#term-prediction) NEG causes “general CPU unit topology is not stable”. A computed categorical [target](../glossary.md#term-target)
using integer addition also fails.

Repair: validate concrete boundary accesses without changing the intentionally abstract cache
projection. [Regression](../glossary.md#term-regression) [scope](../glossary.md#term-scope) includes composed predictions/targets and strided loss bindings.

### C5 — Low composite [activations](../glossary.md#term-activation) bypass [FLOAT32](../glossary.md#term-float32) working sites

Location: `CpuScalarEmitter.java:277,304` in the [CPU](../glossary.md#term-cpu) generated [scalar](../glossary.md#term-scalar) path.

[BFLOAT16](../glossary.md#term-bfloat16)/[FLOAT16](../glossary.md#term-float16) inputs were decoded and widened to DOUBLE for the entire composite expression,
with one [narrowing](../glossary.md#term-narrowing) at the end. The current [Model](../glossary.md#term-model) contract instead names FLOAT32 intermediate
sites. For [BF16](../glossary.md#floating-point-precision-and-rounding) [GELU](../glossary.md#gaussian-error-linear-unit--gelu)(-6), the old result is raw `0xb1cb` (about `-5.9e-9`); at the required [F32](../glossary.md#floating-point-precision-and-rounding)
[ERF](../glossary.md#term-erf) site the value rounds to -1, giving a zero composite result.

Repair: preserve exact low decode, F32 [constants](../glossary.md#term-constant) and named F32 operations/elementary-return
[rounding](../glossary.md#term-rounding), then final low narrowing. Verify emitted [bytecode](../glossary.md#term-bytecode) and independent expected values,
not merely a reference implementation with the same [widening](../glossary.md#term-widening) error. No numerical-policy reset
or new low-type capability is needed.

### D1 — [Rank](../glossary.md#term-rank)-16 WHERE [metadata](../glossary.md#term-metadata) exceeds the native array

Location: `native/metal-macos-arm64/src/synaptik_metal_foundation.m:3993`, plus the exact and
low WHERE [shader](../glossary.md#term-shader) definitions.

The false-input rank was stored in `inputExtents[0]`, then its [dimensions](../glossary.md#dimension) in positions `1..rank`.
The array has 16 [elements](../glossary.md#term-element), so valid rank 16 writes and later reads position 16 out of bounds.
This is a source-confirmed native safety defect; the audit did not present a [device](../glossary.md#term-device)-result
counterexample as proof of memory corruption.

Repair: use the existing per-role rank/dimension fields for the third operand. Add actual-device
[FLOAT32](../glossary.md#term-float32)/[FLOAT16](../glossary.md#term-float16) rank-16 cases whose selected false input and last-[axis](../glossary.md#term-axis) [broadcasting](../glossary.md#broadcasting) exercise
the former boundary, plus source/[schema](../glossary.md#term-schema) checks and source-matched native packaging.

### D2 — ScatterAdd modifies untouched low words

Location: `native/metal-macos-arm64/src/synaptik_low_precision_kernels.h:91`.

The low [kernel](../glossary.md#kernel) widened and narrowed every destination, including destinations with no update.
That turns an untouched [BF16](../glossary.md#floating-point-precision-and-rounding) negative [NaN](../glossary.md#term-nan) raw word `0xffc1` into `0x7fc0`, and an [F16](../glossary.md#floating-point-precision-and-rounding) `0xfe01`
into `0x7e00`. Example: base `[1, negative-NaN]`, index `[0]`, update `[1]` must not arithmetically
re-encode the second word.

Repair: raw-copy an unaddressed destination; only addressed values use the arithmetic contract.
Cover both low [dtypes](../glossary.md#data-type--datatype), both index widths and untouched NaN/zero/[subnormal](../glossary.md#term-subnormal) representations.

### D3 — Low [scan](../glossary.md#term-cumulative-scan) [seed](../glossary.md#term-seed) changes a negative-zero prefix

Location: `synaptik_low_precision_kernels.h:64`.

Starting inclusive cumulative SUM at [positive zero](../glossary.md#nan-infinity-and-signed-zero) makes input `[-0,-0]` produce positive-zero
prefixes instead of starting from the first contributor. The same distinction matters in the
reverse direction and in nonempty exclusive prefixes.

Repair: seed from the first contributing value while preserving the specified empty-prefix
[identity](../glossary.md#term-identity-meanings). Test inclusive/exclusive and forward/reverse variants independently.

### D4 — Native [buffers](../glossary.md#term-buffer) are [device](../glossary.md#term-device)-scoped instead of exact-context-scoped

Locations: `synaptik_metal_foundation.m:1372,9518,9843,9858,10000`.

Buffer boxes retained the device but not a durable [identity](../glossary.md#term-identity-meanings) for their creating native context.
Three invocation paths checked device and [extent](../glossary.md#term-extent), allowing an executable from context A to
accept a buffer from context B on the same device. The [normative](../glossary.md#term-normative) boundary requires the exact
creating context. Java [binding](../glossary.md#term-binding) already enforced this; the finding is an exposed native-[ABI](../glossary.md#term-application-binary-interface-abi)
[validation](../glossary.md#term-validation) gap, not a demonstrated public [Engine](../glossary.md#term-engine) bypass.

Repair: retain a safe context identity whose equality cannot be confused by address reuse and
check it before submission/mutation on all three paths. Keep [lifetime](../glossary.md#term-lifetime) [ownership](../glossary.md#term-ownership-meanings) explicit and
test cross-context rejection with unchanged destination bytes.

### D5 — Provider-true [scalar](../glossary.md#term-scalar) operations fail during preparation

Locations: `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalCapabilityProvider.java:1480,1640`
and `MetalNativeApi.java:875,904`; the native [schema](../glossary.md#term-schema) decoder has corresponding guards around
`synaptik_metal_foundation.m:8551,9428` after the 0078 integration.

Discovered while restoring [Engine](../glossary.md#term-engine) mixed-[owner](../glossary.md#term-owner) tests, after the initial audit pass. With [Metal](../glossary.md#term-metal)
registered first, [CPU](../glossary.md#term-cpu) second, a native-[host](../glossary.md#term-host) [FLOAT32](../glossary.md#term-float32) scalar 1.25 and `x.exp().relu()`, [compilation](../glossary.md#compiler-engine-and-execution-stage-names)
selects `[cpu,metal]`. Session preparation then throws “Metal [MPSGraph](../glossary.md#term-mpsgraph) RELU input/output [shape](../glossary.md#shape) is
unsupported” before invoking [native code](../glossary.md#native-numerical-library-and-instruction-architecture). The provider and its test explicitly admit scalar RELU,
but Java [preflight](../glossary.md#term-preflight) requires a scalar RELU to be inside an anchor [epilogue](../glossary.md#term-epilogue). The same source-level
mismatch affects scalar MUL. The later native [operation](../glossary.md#operation)-shape checks accept matching scalar
shapes, but earlier native decoder guards also require anchor membership. Existing raw-[ABI](../glossary.md#term-application-binary-interface-abi) tests
confirm UNSUPPORTED_SHAPE; this is not only a Java [guard](../glossary.md#term-guard) defect. After correcting [admission](../glossary.md#term-admission),
actual-[device](../glossary.md#term-device) integration exposes a third part: the FLOAT32 MUL literal has shape `[1]`, so its
MPSGraph result does not match the advertised [rank-zero](../glossary.md#axis-rank-scalar-vector-and-matrix) output and compilation fails. Low MUL,
RELU/CLAMP and the mixed CPU [EXP](../glossary.md#term-expx) → Metal RELU case passed that diagnostic [run](../glossary.md#term-run).

Repair [scope](../glossary.md#term-scope): align Java/native admission with already-advertised occurrences, retain separate
anchor-record [authentication](../glossary.md#term-authentication), rebuild the native package and advance compatibility [identity](../glossary.md#term-identity-meanings) to
36; correct only the rank-zero MUL literal/result shape rather than globally changing shared
scalar helpers. Prove direct/composed scalar execution on the actual device. Check advertised
low [dtypes](../glossary.md#data-type--datatype) separately and scalar CLAMP as an adjacent control. Replacing the Engine [fixture](../glossary.md#term-fixture) with
`exp().reshape(1)` made that transfer test valid but did not close this production finding.

### E1 — [Adam](../glossary.md#adaptive-optimizer--adam-adamw)/[AdamW](../glossary.md#adaptive-optimizer--adam-adamw) intermediate order rejects a [finite](../glossary.md#term-finite) update

Location: `extensions/training/src/main/java/io/github/pho001/synaptik/training/TrainingParameter.java:455,540`.

The implementation divided the corrected first moment by the denominator before multiplying by
the [learning rate](../glossary.md#term-learning-rate). The documented/direct formula multiplies first. In [FLOAT32](../glossary.md#term-float32), [parameter](../glossary.md#term-parameter) 1,
learning rate and [epsilon](../glossary.md#epsilon) `1e-30`, beta1 0.5, beta2 0, zero decay, and [gradients](../glossary.md#term-gradient) `1e19` then 0
produce a finite second update around `3.333e18`. The old unscaled quotient overflows and rejects
it. Existing [transaction](../glossary.md#term-transaction) behavior correctly leaves the completed-step count at one, but the
valid documented update should not have failed. [FLOAT64](../glossary.md#term-float64) uses the same erroneous ordering.

Repair: validate the scaled numerator, then divide and validate the update, in both widths.
Keep staged candidates and [atomic](../glossary.md#resource-transaction-and-rollback) [commit](../glossary.md#term-commit) unchanged. Tests cover Adam/AdamW, both widths,
intermediate [overflow](../glossary.md#term-overflow), state/counter preservation and successful retry.

### E2 — [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) public types are not exported to consumers

Location: `extensions/nn/build.gradle.kts:2`.

NN exposes [Model](../glossary.md#term-model) [`Tensor`](../glossary.md#tensor), [`Shape`](../glossary.md#shape) and [`DataType`](../glossary.md#data-type--datatype) in public [signatures](../glossary.md#term-signature), but declared the Model
[dependency](../glossary.md#term-dependency) as `implementation`. A [Gradle](../glossary.md#term-gradle) consumer depending only on NN fails to [compile](../glossary.md#compile) with
“cannot access Tensor”; adding Model manually masks the [API](../glossary.md#term-application-programming-interface-api) [metadata](../glossary.md#term-metadata) error.

Repair: use `api` for the already permitted NN→Model edge. An isolated NN-only consumer now
compiles as part of architecture [validation](../glossary.md#term-validation); the existing Training-only [fixture](../glossary.md#term-fixture) remains separate.
This exports an existing boundary, not a new [module](../glossary.md#term-module) dependency or an [Engine](../glossary.md#term-engine) [facade](../glossary.md#term-facade) inside NN.

### F1 — [Cache](../glossary.md#term-cache) size bound is enforced after the expensive allocation

Location: `tools/tuning/src/main/java/io/github/pho001/synaptik/tools/tuning/ModelPlanCacheFile.java:505`.

The encoder accumulated all entries in a growing byte stream, copied the [payload](../glossary.md#artifact-envelope-and-atomic-publication) and computed
its [checksum](../glossary.md#term-checksum) before rejecting a total above 16 MiB. Individually valid fields can have an
arbitrarily excessive aggregate within the entry-count limit. The neighboring local-workload
cache already preflights aggregate size.

Repair: count the complete [schema](../glossary.md#term-schema)-1 encoding with checked arithmetic before allocating the
exact-sized output. Include all length prefixes, fixed fields and checksum; retain [canonical](../glossary.md#term-canonical-form)
bytes and [atomic](../glossary.md#resource-transaction-and-rollback) file replacement. Tests cover the inclusive limit, one byte above/below,
large logical totals without enormous allocation, [fixture](../glossary.md#term-fixture) compatibility and unchanged [target](../glossary.md#term-target).

## Test and documentation findings

### G1 — Stable [softmax](../glossary.md#softmax--log-softmax) and literal intermediate [overflow](../glossary.md#term-overflow) disagree

An independent source cross-check at integrated revision `c2461e1e` found a separate conflict
beyond [CPU](../glossary.md#term-cpu) 0021's [validation](../glossary.md#term-validation) repair. The following G1 source locations refer to that revision.
For admitted [contiguous](../glossary.md#term-contiguous) [FLOAT32](../glossary.md#term-float32) `[1000,1000]`, [axis](../glossary.md#term-axis) 0, `CpuSoftmaxEmitter` max-shifts the values
to `[0,0]`, yielding `[0.5,0.5]`. `SoftmaxKind` and the foundational contract's Composite inheritance
row instead specify literal unshifted [EXP](../glossary.md#term-expx), sum and division: the EXP results overflow to [infinity](../glossary.md#term-infinity),
and infinity divided by infinity produces [NaN](../glossary.md#term-nan). This is a result-class discrepancy, not a demand
for cross-[backend](../glossary.md#backend) bit equality.

Relevant sources: `CpuCapabilityProvider.java:529`, `CpuSoftmaxInputValidator.java:101`,
`CpuSoftmaxEmitter.java:146`, `SoftmaxKind.java:19`, and
`docs/architecture/contracts/foundational-modules.md:193,265,266,274`. The alternative-algorithm
clause requires special-value preservation; the source review found no express exception for
avoiding literal intermediate overflow. Existing Stage-A/[normalization](../glossary.md#term-normalization)/[conformance](../glossary.md#term-conformance) tests use a
stable [oracle](../glossary.md#term-test-oracle) and cannot grant that permission. No fresh numerical diagnostic was [run](../glossary.md#term-run) in this
cross-check; the trigger and both evaluations are source-derived.

**Unresolved, not silently repaired:** the recommended direction is to retain useful stable
softmax and clarify its explicit [finite](../glossary.md#term-finite)-overflow semantics, while explicitly preserving/specifying
existing NaN/infinity/empty rules and supported/rejected domains. This changes a numerical
contract and needs an [owner](../glossary.md#term-owner) decision plus
independent Class C review. [Model 0033](modules/model/tasks/0033-stable-softmax-overflow-semantics.md)
records the Draft decision task. No broader composite/[fusion](../glossary.md#term-fusion) permission follows by analogy.
The owner explicitly deferred this decision until after the final review report (2026-10-13).
[Model](../glossary.md#term-model) 0033 remains Draft; neither direction nor implementation is authorized, and G1 remains open.

### Test and explanation repairs

- **B2:** `testing/integration-tests/.../EngineExplicitCompositionMetalIntegrationTest.java:1970,2036`
  uses EXP/SIGMOID as old CPU-only separators. Both are now supported on Metal; several rank-1,
  rank-16, multi-hop and mixed-publication tests assert only <span data-glossary-navigation="true"> ([EXP glossary](../glossary.md#term-expx); [SIGMOID glossary](../glossary.md#term-sigmoid); [CPU glossary](../glossary.md#term-cpu); [rank glossary](../glossary.md#term-rank); [publication glossary](../glossary.md#term-publication))</span>[`metal`](../glossary.md#term-metal) ownership. They no longer
  test the boundary named by the fixture. Retain genuine rank-zero coverage and add operations
  that actually force the intended owner sequence, with transfer directions/counts and fan-out
  assertions. Do not hardcode a false provider capability to make a test pass.<span data-glossary-navigation="true"> ([ownership glossary](../glossary.md#term-ownership-meanings); [fixture glossary](../glossary.md#term-fixture); [rank-zero glossary](../glossary.md#axis-rank-scalar-vector-and-matrix); [owner glossary](../glossary.md#term-owner); [fan-out glossary](../glossary.md#term-fan-out))</span>
- **B3:** `modules/engine/src/test/java/.../EngineTypedLifecycleTest.java:629` submits a second
  copy then closes the Engine. Submission does not prove admission via <span data-glossary-navigation="true"> ([Engine glossary](../glossary.md#term-engine); [admission glossary](../glossary.md#term-admission))</span>`beginOperation`.
  Close may legally win. The race is source-confirmed; this audit did not reproduce an actual
  flaky failure. Use deterministic synchronization or split the legal outcomes without sleeps
  or a production lifecycle change.<span data-glossary-navigation="true"> ([race glossary](../glossary.md#term-race); [synchronization glossary](../glossary.md#term-synchronization); [lifecycle glossary](../glossary.md#lifecycle))</span>
- **B4:** `modules/engine/src/main/java/.../Engine.java:380` describes `prepareTuned` as CPU-only,
  omitting the existing bounded Metal singleton NEG session route, same-owner fallback and
  heterogeneous rejection. The glossary's matching tuning entries repeat that stale claim.
  Correct both explanations without expanding tuning capability.<span data-glossary-navigation="true"> ([singleton glossary](../glossary.md#term-singleton); [heterogeneous glossary](../glossary.md#term-heterogeneous))</span>
- [Trace](../glossary.md#trace) `LowPrecisionTraceMetadata` Javadoc says “three execution facts” for two route/dtype facts.
  This is a local wording repair; the DTO and runtime behavior stay unchanged.<span data-glossary-navigation="true"> ([Javadoc glossary](../glossary.md#term-javadoc); [dtype glossary](../glossary.md#data-type--datatype); [DTO glossary](../glossary.md#data-transfer-object--dto); [runtime glossary](../glossary.md#term-runtime))</span>
- Independent documentation review found a stale glossary opening that still described Model
  0032 as private/uncommitted and a duplicated implementation inventory that called delivered
  Compiler, backend and Training work planned. Replace that inventory with a short current
  lifecycle/status convention; preserve the glossary's term definitions and historical task records.
  The roadmap also no longer implies an ONNX mapping task exists when its table is still empty.<span data-glossary-navigation="true"> ([Model glossary](../glossary.md#term-model); [Compiler glossary](../glossary.md#term-compiler); [backend glossary](../glossary.md#backend); [ONNX glossary](../glossary.md#term-onnx))</span>

## Whole-project coverage and intentional limits

Counts below describe reviewed-area inventory, not per-file proof. Java production/test counts
were captured after the first five scoped repairs; integration tests live in a separate project.

| Area | Production / test Java files | Review outcome and principal [scope](../glossary.md#term-scope) |
|---|---:|---|
| [Model](../glossary.md#term-model) | 199 / 141 | Typed [operation](../glossary.md#operation) contracts, [shape](../glossary.md#shape)/promotion/casts, saved roles; no separate confirmed Model defect. |
| [Compiler](../glossary.md#term-compiler) | 45 / 40 | [Graph](../glossary.md#graph)/[gradient](../glossary.md#term-gradient)/[preflight](../glossary.md#term-preflight)/[layout](../glossary.md#layout) paths; A1/A2. [Conv3d](../glossary.md#conv3d--ncdhw) adjoints and recurrent [BPTT](../glossary.md#term-backpropagation-through-time-bptt) remain intentionally unsupported. |
| [Planning](../glossary.md#term-planning) | 13 / 9 | [Ownership](../glossary.md#term-ownership-meanings)/partitioning and capability use; no confirmed executable defect. |
| [Runtime](../glossary.md#term-runtime) | 26 / 17 | [Binding](../glossary.md#term-binding), [leases](../glossary.md#term-lease), isolation and closure; no confirmed executable defect. |
| [Prepare](../glossary.md#prepare) | 27 / 12 | [Transaction](../glossary.md#term-transaction)/finalization/ownership; B1. |
| [Engine](../glossary.md#term-engine) | 23 / 10 | [Admission](../glossary.md#term-admission), sessions, transfer/[publication](../glossary.md#term-publication) and tuning; B2–B4 in tests/docs. |
| Config | 7 / 5 | Current leaves/[validation](../glossary.md#term-validation) and consumers; no confirmed executable defect. |
| [Trace](../glossary.md#trace) | 22 / 6 | Current [payload](../glossary.md#artifact-envelope-and-atomic-publication) facts/immutability and listeners; wording issue only. |
| [Backend](../glossary.md#backend) Contract | 9 / 4 | Minimal shared capability seam and [dependency direction](../glossary.md#java-platform-and-build-tools); no confirmed defect. |
| [CPU](../glossary.md#term-cpu) | 172 / 217 | Generated/direct families, validator ordering, composed binding and [identity](../glossary.md#term-identity-meanings); [C1](../glossary.md#term-c1)–C5. |
| [OpenBLAS](../glossary.md#term-openblas) provider | 8 / 6 | Discovery, four-[symbol](../glossary.md#term-symbol) boundary, thread/route qualifications; no confirmed defect. No new real-library [run](../glossary.md#term-run). |
| [Metal](../glossary.md#term-metal) | 35 / 47 | Native range/context/layout, low [kernels](../glossary.md#kernel), catalog/provider truth and source/package checks; D1–D5. |
| [CUDA](../glossary.md#term-cuda) | 1 / 0 | Marker and native scaffold, not an executable backend. |
| [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) | 46 / 41 | Public [API](../glossary.md#term-application-programming-interface-api)/dependencies, state/load/init and layer graph construction; E2. |
| Training | 13 / 5 | [Optimizer](../glossary.md#term-optimizer) staging, accumulation, [rollback](../glossary.md#term-rollback)/[snapshot](../glossary.md#term-snapshot) and orchestration; E1. |
| [ONNX](../glossary.md#term-onnx) | 1 / 0 | Marker, not an importer/exporter. |
| Tuning | 15 / 12 | [Cache](../glossary.md#term-cache) bounds/encoding, candidate/decision ownership and compatibility; F1. |
| Benchmarks | 3 / 0 | Report-only harness/configuration, no general performance claim. |
| [CLI](../glossary.md#term-cli) | 1 / 0 | Marker; no completed CLI workflow. |
| Architecture tests | 1 / 7 | [Dependency](../glossary.md#term-dependency) restrictions and isolated public consumers; NN [fixture](../glossary.md#term-fixture) added. |
| Backend [conformance](../glossary.md#term-conformance) | 1 / 9 | Relevant semantic/representation regressions reviewed and extended by repairs. |
| Integration tests | 1 / 20 | Public composed execution and native routes; gradient regressions added, mixed tests repaired. |

Data, Text, Vision and [Checkpoint](../glossary.md#term-checkpoint) have plans but no implementation [modules](../glossary.md#term-module). Their architecture
decisions remain gates. No `.github` workflow was present at baseline; provisioning hosted [GPU](../glossary.md#term-gpu)
[CI](../glossary.md#term-continuous-integration-ci) is a separate unresolved operational choice, not a defect silently repaired in this audit.

The Metal baseline audit reports 88 supported / 27 unsupported representative kinds, structural
execution 103/12, [MPSGraph](../glossary.md#term-mpsgraph) catalog 74/36/5 and custom catalog 73/42/0. These are different questions,
not interchangeable support totals. CPU support similarly must be queried for the concrete typed
[occurrence](../glossary.md#term-occurrence); no claim of universal [vector](../glossary.md#term-vector)/parallel/layout coverage follows from this review.

## Risks and follow-ups, not confirmed correctness findings

1. **Training capacity:** FLOAT32 Adam can retain roughly 36 bytes per parameter element,
   including caller storage, after an update; a retained snapshot adds roughly 20 bytes and
   accumulation staging can add another 4. At 100 million elements, the first two figures are
   about 3.6 GB plus 2 GB before backend buffers. Per-array limits do not impose an aggregate
   memory budget. Measure a representative model before choosing scratch reuse, snapshot
   streaming or backend optimizer state. Do not disable rollback silently.<span data-glossary-navigation="true"> ([FLOAT32 glossary](../glossary.md#term-float32); [Adam glossary](../glossary.md#adaptive-optimizer--adam-adamw); [parameter glossary](../glossary.md#term-parameter); [element glossary](../glossary.md#term-element); [snapshot glossary](../glossary.md#term-snapshot); [backend glossary](../glossary.md#backend); [buffers glossary](../glossary.md#term-buffer); [model glossary](../glossary.md#term-model); [optimizer state glossary](../glossary.md#term-optimizer-state); [rollback glossary](../glossary.md#term-rollback))</span>
2. **Cold [planning](../glossary.md#term-planning) complexity:** `LogicalMemoryPlanning` contains value/node scanning with a
   possible quadratic cost. No benchmark establishes a present bottleneck; measure graph scaling
   before changing the ownership/interval algorithm.<span data-glossary-navigation="true"> ([node glossary](../glossary.md#node); [graph glossary](../glossary.md#graph); [ownership glossary](../glossary.md#term-ownership-meanings))</span>
3. **[CPU](../glossary.md#term-cpu) indexed-update uniqueness:** a path in `CpuPreparedExecutable` compares updates
   quadratically. A valid concern for large index sets, not evidence of a measured regression.<span data-glossary-navigation="true"> ([regression glossary](../glossary.md#term-regression))</span>
4. **Static recurrent expansion:** bidirectional sequence alignment can construct work
   proportional to sequence length times squared batch size. Runtime scan adoption is already
   a separate backend-dependent program; this audit does not silently replace it.<span data-glossary-navigation="true"> ([alignment glossary](../glossary.md#term-alignment); [batch glossary](../glossary.md#term-batch-machine-learning-and-hyperparameters); [Runtime glossary](../glossary.md#term-runtime))</span>
5. **Large implementation [owners](../glossary.md#term-owner):** CPU prepared execution and native Metal remain large,
   multi-branch files. Extract responsibilities only at a demonstrated seam with focused tests;
   avoid a broad cosmetic rewrite during correctness repair.<span data-glossary-navigation="true"> ([prepared execution glossary](../glossary.md#prepared-execution--preparedexecution); [Metal glossary](../glossary.md#term-metal))</span>
6. **Platform confidence:** actual Apple-device tests and source/package authentication do not
   prove other GPU/OS behavior. Optional BF16 OpenBLAS remains blocked on ABI/full-accumulation
   semantics; that does not block current FLOAT32/FLOAT64 routes.<span data-glossary-navigation="true"> ([device glossary](../glossary.md#term-device); [authentication glossary](../glossary.md#term-authentication); [GPU glossary](../glossary.md#term-gpu); [BF16 glossary](../glossary.md#floating-point-precision-and-rounding); [OpenBLAS glossary](../glossary.md#term-openblas); [ABI glossary](../glossary.md#term-application-binary-interface-abi); [FLOAT64 glossary](../glossary.md#term-float64))</span>
7. **[Vector](../glossary.md#term-vector) [ERF](../glossary.md#term-erf) shape/performance:** CPU 0022's bounded diagnostic recorded about 3.28× FLOAT32
   and 3.51× FLOAT64 generated/direct time. Both are vector implementations of the same numerical
   algorithm, with 4,097 elements, one tail, one process, 3,000 warm-up pairs and seven measured
   pairs. Production retains preexisting helper calls and coefficient loops; the direct Java
   helper uses explicit constant Horner steps. The current gap is measured, but its cause and
   regression versus baseline are not established. Performance parity is not proven.<span data-glossary-navigation="true"> ([warm-up glossary](../glossary.md#benchmark-measurement-vocabulary); [constant glossary](../glossary.md#term-constant))</span>
   [CPU 0023](backends/cpu/tasks/0023-vector-erf-direct-java-shape.md) records a bounded Draft
   shape follow-up without restarting historical five-fork matrices.<span data-glossary-navigation="true"> ([fork glossary](../glossary.md#term-fork-meanings); [matrices glossary](../glossary.md#term-matrix-meanings))</span>

## Plan reconciliation and proposed next work

The original plan problem was primarily stale live prose, not task-table status disagreement.
The initial mechanical master-row/brief comparison found no mismatch. Roadmap and [Metal](../glossary.md#term-metal) master
plan had accumulated repeated chronology, obsolete global “no [Ready](../glossary.md#term-ready) task” statements, old
NumericalProfile gates and completed work described as prerequisites.

Changes preserve historical task records while making the current map actionable:

- Compact roadmap: one active repair frontier table, all 23 area links, explicit next decisions
  and non-goals. Compact Metal master: current rules/gates plus the retained historical task table.
  The original 75 historical Metal table lines remain unchanged. Across roadmap and 23 masters,
  active-index prose shrank from 6,128 to 4,390 lines; completed task records were not deleted.
- Config/Trace/CPU/Engine and future extension plans now describe the existing profile-free
  contract rather than resurrecting <span data-glossary-navigation="true"> ([Trace glossary](../glossary.md#trace); [CPU glossary](../glossary.md#term-cpu); [Engine glossary](../glossary.md#term-engine))</span>`STRICT_IEEE`/[`ACCELERATOR`](../glossary.md#term-accelerator) [API](../glossary.md#term-application-programming-interface-api) or [certificate](../glossary.md#formal-proof-certificate-and-trusted-assumptions) gates.
- Model's unbriefed obsolete FLOAT16 0026 is Superseded by delivered consolidated work, not a
  new prerequisite. Compiler exact rewrites no longer wait for obsolete relaxed-profile policy.<span data-glossary-navigation="true"> ([FLOAT16 glossary](../glossary.md#term-float16))</span>
- [Planning](../glossary.md#term-planning)'s old glossary/GraphCompilationPort follow-up is already closed by [Compiler](../glossary.md#term-compiler) 0006B10.
- Checkpoint recognizes existing Engine host materialization; durable encoding/state handoff,
  not inability to read bytes, is the missing capability.<span data-glossary-navigation="true"> ([host glossary](../glossary.md#term-host); [materialization glossary](../glossary.md#materialization); [handoff glossary](../glossary.md#term-handoff))</span>
- Training 0003 no longer waits for an undefined “Data lifecycle” that Data explicitly excludes.
  Draft decision 0003A must define exact-resume participants; caller-owned progress/sampler
  state is a recommendation, not an architecture change already made.<span data-glossary-navigation="true"> ([lifecycle glossary](../glossary.md#lifecycle))</span>
- Training/[Checkpoint](../glossary.md#term-checkpoint) dependencies name tasks and avoid the apparent adapter cycle:
  `Training 0003A → 0003`, `Checkpoint 0001 → 0004`, then
  `Training 0004 → Checkpoint 0005 → Checkpoint 0006`.
- CUDA/ONNX/CLI markers, absent future modules, hosted GPU runner blockage, BF16 native provider
  limits and missing adjoints remain visible. Completed strict-era probe failures are not
  implicit prerequisites for today's qualified routes.<span data-glossary-navigation="true"> ([CUDA glossary](../glossary.md#term-cuda); [ONNX glossary](../glossary.md#term-onnx); [CLI glossary](../glossary.md#term-cli); [modules glossary](../glossary.md#term-module); [GPU glossary](../glossary.md#term-gpu); [BF16 glossary](../glossary.md#floating-point-precision-and-rounding))</span>

The bounded repairs and combined checkpoint are complete. Recommended next order: resolve [Model](../glossary.md#term-model) 0033's
stable-[softmax](../glossary.md#softmax--log-softmax) contract decision before claiming numerical closure; then select Training 0003A's
[resume](../glossary.md#schema-serialization-and-checkpoint)-state decision. If durable saving is the product priority, authorize Checkpoint's
architecture/model-only work. Data/Text/Vision form a separately selected architecture stream.
These successor proposals remain Draft until their actual decisions and [scope](../glossary.md#term-scope) are approved.

## Repair, review and [validation](../glossary.md#term-validation) ledger

| Task | Implementation / main integration | Focused and [module](../glossary.md#term-module) evidence | Independent review | State |
|---|---|---|---|---|
| [Compiler](../glossary.md#term-compiler) 0006D | `e6fb9d28` / `5dead33c` | 292 Compiler tests; 4 numerical [CPU](../glossary.md#term-cpu) integration tests; [Javadoc](../glossary.md#term-javadoc) | APPROVE | Complete |
| [Prepare](../glossary.md#prepare) 0009 | `ec2e7ddf` / `d7de5721` | 18 focused / 65 module tests; Javadoc | APPROVE | Complete |
| Training 0002C | `97e66ee3` / `1f9ea4bd` | 11 focused / 68 module tests; Javadoc | APPROVE | Complete |
| [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0025B | `6eefd08d` / `7905036f` | NN-only and Training-only consumer [compilation](../glossary.md#compiler-engine-and-execution-stage-names); 306 NN tests; [architecture test](../glossary.md#term-architecture-test); Javadoc | APPROVE | Complete |
| Tuning 0005 | `ea9059dc` / `d324ee13` | 20 focused / 73 module tests; exact encoding/bounds; Javadoc | APPROVE | Complete |
| CPU 0021 | `ca21c80c` / `c0559e2b` | 1,026 CPU passed / 28 skipped; 2 [conformance](../glossary.md#term-conformance) + 6 integration passed; Javadoc | APPROVE after [softmax](../glossary.md#softmax--log-softmax) glossary correction | Complete |
| CPU 0022 | `a1e00857` / `349709d6` | 1,033 CPU passed / 29 opt-in skips; 2 conformance + 2 public integration passed; 18 Class-Files; bounded generated/direct diagnostic; Javadoc | APPROVE, including final [oracle](../glossary.md#term-test-oracle)/TSV and shared-version changes | Complete |
| [Metal](../glossary.md#term-metal) 0078 | `50f570d8` + docs `12babf46` / `6e8d6b26`, `1db53756` | 4 new [device](../glossary.md#term-device) regressions, 340 Metal passed / 10 optional [observer](../glossary.md#term-observer) skips, 37 conformance passed; native/package/source-[AIR](../glossary.md#term-air)/Javadoc | APPROVE after version-document correction | Complete |
| [Engine](../glossary.md#term-engine) 0019 | `a73822ca` / `241ce92b` | 122 Engine tests, 3 final cleanup tests; 33 actual-device integration tests without skips; Javadoc | APPROVE including glossary/user-guide corrections | Complete |
| Metal 0079 | `1f9a5833` / `22a2f1be` | 25 focused passed; 346 Metal passed / 10 optional observer skips; 21 conformance + 10 [scalar](../glossary.md#term-scalar) integration passed; native/package/source/Javadoc | APPROVE, including shared-version/glossary corrections | Complete |

Independent reviewers inspected the actual diffs and relevant contracts/tests/Javadocs and reused
successful executable evidence rather than rerunning it merely to duplicate a result. Known
limits: Compiler new numerical cases use static [contiguous](../glossary.md#term-contiguous) [FLOAT32](../glossary.md#term-float32); Prepare cleanup tests use
resource doubles; [cache](../glossary.md#term-cache) allocation safety is structural plus deterministic size tests, not an
intentional out-of-memory experiment; NN isolated source consumption is not a published-[artifact](../glossary.md#term-artifact)
[release](../glossary.md#term-release) test. Training's new order stress tests cover FLOAT32/[FLOAT64](../glossary.md#term-float64), with existing low routing
and [narrowing](../glossary.md#term-narrowing) tests reused.
CPU 0022 additionally exhausts both low 16-bit encoding spaces across five composites (655,360
input/[operation](../glossary.md#operation) cases) against an independent Java oracle. The separate conformance tests execute
128 [carrier](../glossary.md#term-storage-carrier)/[layout](../glossary.md#layout)/type/operation combinations, 96 with native storage; these are bounded cases,
not universal operation coverage. Its initial module runs exposed stale [schema](../glossary.md#term-schema)/seal fixtures and
one old whole-DOUBLE low-[GELU](../glossary.md#gaussian-error-linear-unit--gelu) structural oracle. Current fixtures and the oracle formula were
corrected without weakening comparisons; the final committed module [run](../glossary.md#term-run) passed all 1,033 active
tests. The large generated inventory diff refreshes [identity](../glossary.md#term-identity-meanings)/[hash](../glossary.md#term-hash)/body keys over 41,336 generated
rows, retaining 265 rejected rows and unchanged strategy/capability/disposition facts. Historical
performance measurements are not requalified by that refresh.
CPU's composed invalid-[mask](../glossary.md#term-boolean-mask) [regression](../glossary.md#term-regression) rejects in the NOT [producer](../glossary.md#term-producer), so it does not independently
prove [failure atomicity](../glossary.md#term-failure-atomicity) of both attention outputs. Materialized [BOOL](../glossary.md#data-type--datatype) validation lacks a combined
sentinel regression, although source ordering confirms [materialization](../glossary.md#materialization) executes before consumer
validation. These are bounded coverage limitations, not demonstrated remaining defects.

Independent [planning](../glossary.md#term-planning) reconciliation review: APPROVE after correcting the glossary cutover status,
[ONNX](../glossary.md#term-onnx)'s undefined first task and the explicit architecture gate for [backend](../glossary.md#backend) discovery. All 287
post-introduction glossary heading/explicit anchors were retained. Added D5's narrow repair brief
and final status roster receive the coordinator's final checks separately.

## Final integrated [checkpoint](../glossary.md#term-checkpoint)

After ordered integration through [CPU](../glossary.md#term-cpu) `349709d6` and [Metal](../glossary.md#term-metal) `22a2f1be`, plus the reviewed shared
CPU70/Metal36 documentation and [identity](../glossary.md#term-identity-meanings)-test adjustment:

```sh
SYNAPTIK_METAL_TEST_LIBRARY=/private/tmp/synaptik-review-metal-0079/native/metal-macos-arm64/build/package-v1/macos-arm64/libsynaptik_metal_foundation.dylib ./gradlew test javadoc --max-workers=2 --console=plain
```

Result: **BUILD SUCCESSFUL**, 15 seconds; 101 actionable tasks, 35 executed, 16 from [cache](../glossary.md#term-cache),
50 up-to-date. Resulting XML contains **4,004 test cases: 3,964 passed, 0 failures/errors,
40 skipped**. This includes reused/up-to-date results, not 4,004 freshly executed cases.
CPU, Metal and [Compiler](../glossary.md#term-compiler) test tasks reused matching worker build-cache results; [Engine](../glossary.md#term-engine), [Prepare](../glossary.md#prepare),
Training, Tuning, architecture, [conformance](../glossary.md#term-conformance) and integration tests executed in the combined [run](../glossary.md#term-run).
No second successful full repository run was performed.

Skip inventory: 29 opt-in CPU performance/evidence tests, 10 optional Metal [dispatch](../glossary.md#term-dispatch)-[observer](../glossary.md#term-observer)
tests, and one Engine CPU-[autotuning](../glossary.md#autotuning--model-autotuning) case whose [host](../glossary.md#term-host) [fixture](../glossary.md#term-fixture) offered no eligible local-workload
alternatives. The [activation](../glossary.md#term-activation) diagnostic was separately enabled and passed. No repaired [scalar](../glossary.md#term-scalar)
or mixed-[owner](../glossary.md#term-owner) Metal [regression](../glossary.md#term-regression) skipped. Combined integration: 119 passed / 1 skipped; [backend](../glossary.md#backend)
conformance: 46 passed; architecture: 9 passed, including isolated [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/Training consumer [compilation](../glossary.md#compiler-engine-and-execution-stage-names).
The freshly executed actual-[device](../glossary.md#term-device) integration includes 33 mixed-owner and 10 scalar Metal tests.

Source-matched Apple M3 Max native package ([ABI](../glossary.md#term-application-binary-interface-abi) 7/[schema](../glossary.md#term-schema) 20, Java identity 36):

- Library [SHA-256](../glossary.md#term-sha-256): `03cba3ebfa0bbb78c1bff369ff3451c591846fd975c0034fa7f67fdd2209985a`.
- Integrated foundation-source SHA-256: `0c79c560fb3eaed65fa2fec0b91e2602e54b5fb5146cf0683a446d1290fd9524`.
- Both were checked against the reviewed [0079 record](../../modules/backend/metal/evidence/0079/validation.json).
  Build/sign/package/source checks are recorded there; unchanged shader/AIR evidence was reused.<span data-glossary-navigation="true"> ([shader glossary](../glossary.md#term-shader); [AIR glossary](../glossary.md#term-air))</span>

[Javadoc](../glossary.md#term-javadoc) succeeded. Existing [module](../glossary.md#term-module) warnings, [Vector API](../glossary.md#term-vector-api) incubator warnings and the existing
[OpenBLAS](../glossary.md#term-openblas) foreign-function native-access warning remain; this audit does not claim a warning-free
build. No universal device, performance, numerical, or [dtype](../glossary.md#data-type--datatype)/[layout](../glossary.md#layout) coverage follows from this run.

Final documentation checks: 3,990 local links touching 51 changed Markdown files, no broken
[target](../glossary.md#term-target)/anchor; 488 master-row/brief status comparisons, no mismatch; `git diff --check` passed.
All 75 historical Metal table lines remain unchanged. [Planning](../glossary.md#term-planning) review approved the reconciliation;
the final closure roster is independently checked without repeating executable tests.

## Change size and interpretation

Compared with production baseline `728680ea`, the completed repair set touches:

| File group | Files | Added / removed lines, excluding documentation-only rows |
|---|---:|---:|
| Production/native, including implementation Javadocs/package documentation | 33 | +830 / -510 |
| Tests and isolated consumer source | 59 | +4,074 / -270 |
| Current test-resource [identities](../glossary.md#term-identity-meanings)/seals | 8 | +41,840 / -41,840 |
| Native evidence records/source [binding](../glossary.md#term-binding) | 4 | +244 / -11 |
| Build wiring | 2 | +4 / -1 |
| Documentation, plans and this report | 51 | Status, [identity](../glossary.md#term-identity-meanings) and explanation changes; indexes reduced as above |

The large line count is dominated by [schema](../glossary.md#term-schema)-dependent generated test identities, not tens of
thousands of new implementation lines. The current inventory refresh does not relabel historical
timings or certify every generated form. The original three user-side local [commits](../glossary.md#commit-release-distribution-log-and-seed-meanings) were preserved.
Changes are local commits; no push is part of this review request.

Status: Complete — review and bounded repairs.
Open follow-up: [Model](../glossary.md#term-model) 0033's [owner](../glossary.md#term-owner)-deferred numerical decision, [CPU](../glossary.md#term-cpu) 0023 Draft shape/performance
work and the explicitly listed architecture/product/coverage risks remain outside repaired [scope](../glossary.md#term-scope).
