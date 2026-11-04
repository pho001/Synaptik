# Task 0071: Authenticated [Metal](../../../../glossary.md#term-metal) Anchor-[Epilogue](../../../../glossary.md#term-epilogue) [Fusion](../../../../glossary.md#term-fusion) and Structural Tracing

## Status

Complete.

## Change class

Class C — private [lowering](../../../../glossary.md#lowering), native source, numerical-site evidence, preparation [identity](../../../../glossary.md#term-identity-meanings), and public [trace](../../../../glossary.md#trace) payloads change without [widening](../../../../glossary.md#term-widening) [operation](../../../../glossary.md#operation) capability or the native [ABI](../../../../glossary.md#term-application-binary-interface-abi).

## Goal

Fuse exact [`ACCELERATOR`](../../../../glossary.md#term-accelerator) [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) and [Conv2d](../../../../glossary.md#convolution--conv2d) suffixes into one authenticated custom step while preserving [Model](../../../../glossary.md#term-model) order and numerical-profile semantics. Publish bounded structural [PREPARE](../../../../glossary.md#prepare) and invocation-plan [RUN](../../../../glossary.md#term-run) [trace](../../../../glossary.md#trace) facts after final route selection and immediately before native execution. Retain [ABI](../../../../glossary.md#term-application-binary-interface-abi) 5, all thirteen exports, capability `86/29`, structural coverage `101/14`, and the route catalogs.

## Exact [fusion](../../../../glossary.md#term-fusion) domain

Every admitted anchor and suffix has [FLOAT32](../../../../glossary.md#term-float32) output. Existing all-FLOAT32 [gradient](../../../../glossary.md#term-gradient) [metadata](../../../../glossary.md#term-metadata) and no-gradient mixed [BFLOAT16](../../../../glossary.md#term-bfloat16)/FLOAT32 [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) or [convolution](../../../../glossary.md#term-channels-first-convolution-layers) [carrier](../../../../glossary.md#term-storage-carrier) rules remain unchanged. `STRICT_IEEE` never admits an anchor [epilogue](../../../../glossary.md#term-epilogue).

### [MATMUL](../../../../glossary.md#matrix-multiplication--matmul)

The suffix is source-ordered and contains only:

1. optional literal [FLOAT32](../../../../glossary.md#term-float32) `SCALAR_MUL`;
2. at most one [tensor](../../../../glossary.md#tensor) `ADD`; and
3. optional terminal `RELU` or no-[gradient](../../../../glossary.md#term-gradient) `CLAMP`.

The ADD may name the anchor-derived value on either side, but the other value is a distinct [canonical](../../../../glossary.md#term-canonical-form) FLOAT32 external role. Its [Shape](../../../../glossary.md#shape) is either the exact output Shape or a proved ordinary right-aligned broadcast. Order is never commuted or normalized.

### [Conv2d](../../../../glossary.md#convolution--conv2d)

The [convolution](../../../../glossary.md#term-channels-first-convolution-layers) may retain its optional intrinsic [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) `[C]` [bias](../../../../glossary.md#term-bias) as its third input. Its external suffix contains at most one [tensor](../../../../glossary.md#tensor) ADD followed by optional `RELU` or no-[gradient](../../../../glossary.md#term-gradient) `CLAMP`; [scalar](../../../../glossary.md#term-scalar) multiplication is never admitted.

External ADD uses ordinary right-aligned [broadcasting](../../../../glossary.md#broadcasting). For [NCHW](../../../../glossary.md#term-nchw) output, rank-one `[W]` therefore broadcasts over width, never channels. `[1,C,1,1]` is the ordinary [channel](../../../../glossary.md#term-channel)-broadcast form. Intrinsic rank-one `[C]` remains only the third convolution input and is never reclassified as an external ADD role.

### Privacy and rejection

Every absorbed non-final value is [canonical](../../../../glossary.md#term-canonical-form), private, single-consumer, non-[target](../../../../glossary.md#term-target), and absent from the compact materialized set. [Publication](../../../../glossary.md#term-publication), fanout, unsupported order, duplicate/aliased external roles, type/[layout](../../../../glossary.md#layout)/[gradient](../../../../glossary.md#term-gradient) mismatch, strict profile, or any cap failure preserves the existing composed plan. The recognizer is cold and deterministic; native independently reconstructs its exact record grammar before resource creation.

## Image, [identity](../../../../glossary.md#term-identity-meanings), and source [ownership](../../../../glossary.md#term-ownership-meanings)

- [ABI](../../../../glossary.md#term-application-binary-interface-abi) remains version 5 with exactly thirteen exports.
- Program image [schema](../../../../glossary.md#term-schema) advances cleanly from 17/`SM17` to 18/`SM18`; there is no predecessor reader.
- Generator schema advances from 1 to 2 and adds typed anchor execution instructions.
- [Backend](../../../../glossary.md#backend)-local workload, exact-policy, candidate, compatibility, route-policy, and [codec](../../../../glossary.md#term-codec) identities advance from 26 to 27. Identity 26 and every older value [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation).
- The [canonical](../../../../glossary.md#term-canonical-form) [manifest](../../../../glossary.md#term-manifest) and image [digest](../../../../glossary.md#term-digest) bind anchor family, ordered suffix members, ADD side and external role, raw [scalar](../../../../glossary.md#term-scalar)/clamp words, profile, types, [Shapes](../../../../glossary.md#shape), [layouts](../../../../glossary.md#layout), [gradients](../../../../glossary.md#term-gradient), materialized values, and source-size/cap facts.
- Native authenticates ten fixed source components totaling exactly 84,541 [UTF-8](../../../../glossary.md#term-utf-8) bytes before [compiler](../../../../glossary.md#term-compiler) entry. Java transports typed records and integer byte counts, never source text or source [hashes](../../../../glossary.md#term-hash).
- `AnchorEpilogueMeta` is 6,096 bytes with [alignment](../../../../glossary.md#term-alignment) eight and embeds the existing data geometry plus right-aligned ADD [metadata](../../../../glossary.md#term-metadata) and raw scalar/clamp fields.
- Six [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) type/add [kernel](../../../../glossary.md#kernel) variants and two [Conv2d](../../../../glossary.md#convolution--conv2d) add/no-add kernel variants execute with `MTLMathModeSafe`, one physical [dispatch](../../../../glossary.md#term-dispatch), no suffix-intermediate store, and one final raw-word store. There is no retry, fallback, hot [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names), [host](../../../../glossary.md#term-host) repair, or alternate [publication](../../../../glossary.md#term-publication) path.

The local native package manifest records [node](../../../../glossary.md#node) schema 18 while retaining ABI 5 and the same export surface.

## Structural [trace](../../../../glossary.md#trace) payloads

The trace surface adds [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) records and enums without changing existing public record constructors.

After final route and anchor recognition and before native creation, [PREPARE](../../../../glossary.md#prepare) emits a structural payload containing only:

- prepared-unit correlation and numerical profile;
- final route and [fused](../../../../glossary.md#specialization-and-fusion)-versus-composed family;
- anchor kind, exact [epilogue](../../../../glossary.md#term-epilogue) flags/order/terminal, and member count;
- [schema](../../../../glossary.md#term-schema) and generator versions;
- bounded execution counts; and
- the [canonical](../../../../glossary.md#term-canonical-form) [digest](../../../../glossary.md#term-digest).

Immediately before `resource.run`, [RUN](../../../../glossary.md#term-run) emits an invocation-plan payload containing only:

- invocation correlation and final route;
- bounded route/step counts;
- aggregate input/output/splat/workspace bytes; and
- bounded typed custom-step summaries.

Tracing contains no source text, [tensor](../../../../glossary.md#tensor) data, [scalar](../../../../glossary.md#term-scalar)/clamp bits, pointers, handles, paths, names, secrets, or [graph](../../../../glossary.md#graph)-wide/per-[buffer](../../../../glossary.md#term-buffer) inventory. Disabled tracing performs no payload construction. Structural PREPARE and pre-run [planning](../../../../glossary.md#term-planning) [callback](../../../../glossary.md#term-callback) failures, including [observer](../../../../glossary.md#term-observer) errors, disable tracing and are contained without aborting native finalization or execution. Outcome callback [runtime](../../../../glossary.md#term-runtime) failures disable later tracing without changing [backend](../../../../glossary.md#backend) results; observer errors propagate only from successful outcome reporting and remain suppressed behind backend failure during failure reporting. Native failures retain native outcome authority.

Per-unit [lifecycle](../../../../glossary.md#lifecycle)-close payloads are intentionally deferred. Logical `PreparedUnit` instances can share and reference-count native resources, so a close record at that boundary would imply false physical [lifetime](../../../../glossary.md#term-lifetime) and add high-volume noise. Existing outcome events already close the preparation and invocation diagnostic correlations.

## Evidence

Task-local evidence lives under `backends/metal/evidence/0071`.

- `proof/Task0071AnchorEpilogue.lean` checks literal [epilogue](../../../../glossary.md#term-epilogue) order, one [dispatch](../../../../glossary.md#term-dispatch), no intermediate store, one final store, right-aligned [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) indexing, and exhaustive [DAZ](../../../../glossary.md#term-daz)/[FTZ](../../../../glossary.md#term-ftz) boundary alternatives.
- `source-compiler-site-certificate.json` authenticates the Java planner/image path, native decoder/[preflight](../../../../glossary.md#term-preflight)/dispatch path, fixed source, and test/evidence inputs.
- `audit-anchor-air.py` inspects the exact fixed [runtime](../../../../glossary.md#term-runtime) source compiled by pinned `metal3.2 -fno-fast-math`.
- `compiled-anchor-air-audit.json` records all eight anchor [kernels](../../../../glossary.md#kernel) plus the retained [Conv2d](../../../../glossary.md#convolution--conv2d) helper, seven unflagged [FMA](../../../../glossary.md#term-fused-multiply-add-fma) sites, six unflagged `fmul` sites, four unflagged `fadd` sites, no unsafe floating flags, and exactly one static final byte store per anchor [kernel](../../../../glossary.md#kernel).
- Real-[device](../../../../glossary.md#term-device) [observer](../../../../glossary.md#term-observer) coverage proves one dispatch for [fused](../../../../glossary.md#specialization-and-fusion) [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) and Conv2d, ordinary width broadcast for rank-one Conv2d ADD, exact outputs, [NaN](../../../../glossary.md#term-nan) [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form), [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero) handling, [infinities](../../../../glossary.md#nan-infinity-and-signed-zero), [subnormal](../../../../glossary.md#term-subnormal) FTZ, and clamp endpoints.
- A public [`Engine`](../../../../glossary.md#term-engine) integration [smoke](../../../../glossary.md#term-smoke-test) proves both anchor families and the structural [PREPARE](../../../../glossary.md#prepare)/planned [RUN](../../../../glossary.md#term-run) event order through the normal [Metal](../../../../glossary.md#term-metal) integration surface.

Task 0069 and Task 0070 proof and compiled-source runners remain green against the expanded fixed-source corpus, preserving their reviewed numerical and generator claims.

## [Validation](../../../../glossary.md#term-validation)

The completed change is covered by:

```bash
./native/metal-macos-arm64/build.sh
./backends/metal/evidence/0069/run-proof.sh
./backends/metal/evidence/0070/run-compiled-generated-msl-audit.sh
python3 backends/metal/evidence/0070/verify-source-certificate.py
./backends/metal/evidence/0071/run-proof.sh
./backends/metal/evidence/0070/run-dispatch-observer.sh
```

Focused Java/native coverage includes planner grammar and rejection, [schema](../../../../glossary.md#term-schema)-17/[identity](../../../../glossary.md#term-identity-meanings)-26 predecessor rejection, [manifest](../../../../glossary.md#term-manifest)/[digest](../../../../glossary.md#term-digest) determinism, [fused](../../../../glossary.md#specialization-and-fusion)/composed [trace](../../../../glossary.md#trace) identity, event ordering, [callback](../../../../glossary.md#term-callback) `RuntimeException`/`Error` behavior, native failure behavior, sensitive-field exclusion, one-[dispatch](../../../../glossary.md#term-dispatch) [observer](../../../../glossary.md#term-observer) behavior, and [CPU](../../../../glossary.md#term-cpu)-free public [Engine](../../../../glossary.md#term-engine) execution.

## Non-goals retained

This task adds no [operation](../../../../glossary.md#operation) kind, capability row, native export, public [fusion](../../../../glossary.md#term-fusion) control, benchmark winner, [autotuning](../../../../glossary.md#autotuning--model-autotuning) decision, persistent [cache](../../../../glossary.md#term-cache), general [graph](../../../../glossary.md#graph) rewrite system, broad graph tracing, [asynchronous](../../../../glossary.md#term-asynchronous) execution, [lifecycle](../../../../glossary.md#lifecycle)-close telemetry, native fallback, or new [gradient](../../../../glossary.md#term-gradient) [ownership](../../../../glossary.md#term-ownership-meanings).
