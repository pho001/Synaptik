# Task 0071: Authenticated Metal Anchor-Epilogue Fusion and Structural Tracing

## Status

Complete.

## Change class

Class C — private lowering, native source, numerical-site evidence, preparation identity, and public trace payloads change without widening operation capability or the native ABI.

## Goal

Fuse exact `ACCELERATOR` MATMUL and Conv2d suffixes into one authenticated custom step while preserving Model order and numerical-profile semantics. Publish bounded structural PREPARE and invocation-plan RUN trace facts after final route selection and immediately before native execution. Retain ABI 5, all thirteen exports, capability `86/29`, structural coverage `101/14`, and the route catalogs.

## Exact fusion domain

Every admitted anchor and suffix has FLOAT32 output. Existing all-FLOAT32 gradient metadata and no-gradient mixed BFLOAT16/FLOAT32 MATMUL or convolution carrier rules remain unchanged. `STRICT_IEEE` never admits an anchor epilogue.

### MATMUL

The suffix is source-ordered and contains only:

1. optional literal FLOAT32 `SCALAR_MUL`;
2. at most one tensor `ADD`; and
3. optional terminal `RELU` or no-gradient `CLAMP`.

The ADD may name the anchor-derived value on either side, but the other value is a distinct canonical FLOAT32 external role. Its Shape is either the exact output Shape or a proved ordinary right-aligned broadcast. Order is never commuted or normalized.

### Conv2d

The convolution may retain its optional intrinsic rank-one `[C]` bias as its third input. Its external suffix contains at most one tensor ADD followed by optional `RELU` or no-gradient `CLAMP`; scalar multiplication is never admitted.

External ADD uses ordinary right-aligned broadcasting. For NCHW output, rank-one `[W]` therefore broadcasts over width, never channels. `[1,C,1,1]` is the ordinary channel-broadcast form. Intrinsic rank-one `[C]` remains only the third convolution input and is never reclassified as an external ADD role.

### Privacy and rejection

Every absorbed non-final value is canonical, private, single-consumer, non-target, and absent from the compact materialized set. Publication, fanout, unsupported order, duplicate/aliased external roles, type/layout/gradient mismatch, strict profile, or any cap failure preserves the existing composed plan. The recognizer is cold and deterministic; native independently reconstructs its exact record grammar before resource creation.

## Image, identity, and source ownership

- ABI remains version 5 with exactly thirteen exports.
- Program image schema advances cleanly from 17/`SM17` to 18/`SM18`; there is no predecessor reader.
- Generator schema advances from 1 to 2 and adds typed anchor execution instructions.
- Backend-local workload, exact-policy, candidate, compatibility, route-policy, and codec identities advance from 26 to 27. Identity 26 and every older value fail closed.
- The canonical manifest and image digest bind anchor family, ordered suffix members, ADD side and external role, raw scalar/clamp words, profile, types, Shapes, layouts, gradients, materialized values, and source-size/cap facts.
- Native authenticates ten fixed source components totaling exactly 84,541 UTF-8 bytes before compiler entry. Java transports typed records and integer byte counts, never source text or source hashes.
- `AnchorEpilogueMeta` is 6,096 bytes with alignment eight and embeds the existing data geometry plus right-aligned ADD metadata and raw scalar/clamp fields.
- Six MATMUL type/add kernel variants and two Conv2d add/no-add kernel variants execute with `MTLMathModeSafe`, one physical dispatch, no suffix-intermediate store, and one final raw-word store. There is no retry, fallback, hot compilation, host repair, or alternate publication path.

The local native package manifest records node schema 18 while retaining ABI 5 and the same export surface.

## Structural trace payloads

The trace surface adds payload records and enums without changing existing public record constructors.

After final route and anchor recognition and before native creation, PREPARE emits a structural payload containing only:

- prepared-unit correlation and numerical profile;
- final route and fused-versus-composed family;
- anchor kind, exact epilogue flags/order/terminal, and member count;
- schema and generator versions;
- bounded execution counts; and
- the canonical digest.

Immediately before `resource.run`, RUN emits an invocation-plan payload containing only:

- invocation correlation and final route;
- bounded route/step counts;
- aggregate input/output/splat/workspace bytes; and
- bounded typed custom-step summaries.

Tracing contains no source text, tensor data, scalar/clamp bits, pointers, handles, paths, names, secrets, or graph-wide/per-buffer inventory. Disabled tracing performs no payload construction. Observer runtime failures disable later tracing without changing backend results. Observer errors retain the established success/failure propagation and suppression semantics. Native failures retain native outcome authority.

Per-unit lifecycle-close payloads are intentionally deferred. Logical `PreparedUnit` instances can share and reference-count native resources, so a close record at that boundary would imply false physical lifetime and add high-volume noise. Existing outcome events already close the preparation and invocation diagnostic correlations.

## Evidence

Task-local evidence lives under `backends/metal/evidence/0071`.

- `proof/Task0071AnchorEpilogue.lean` checks literal epilogue order, one dispatch, no intermediate store, one final store, right-aligned rank-one indexing, and exhaustive DAZ/FTZ boundary alternatives.
- `source-compiler-site-certificate.json` authenticates the Java planner/image path, native decoder/preflight/dispatch path, fixed source, and test/evidence inputs.
- `audit-anchor-air.py` inspects the exact fixed runtime source compiled by pinned `metal3.2 -fno-fast-math`.
- `compiled-anchor-air-audit.json` records all eight anchor kernels plus the retained Conv2d helper, seven unflagged FMA sites, six unflagged `fmul` sites, four unflagged `fadd` sites, no unsafe floating flags, and exactly one static final byte store per anchor kernel.
- Real-device observer coverage proves one dispatch for fused MATMUL and Conv2d, ordinary width broadcast for rank-one Conv2d ADD, exact outputs, NaN canonicalization, signed-zero handling, infinities, subnormal FTZ, and clamp endpoints.
- A public `Engine` integration smoke proves both anchor families and the structural PREPARE/planned RUN event order through the normal Metal integration surface.

Task 0069 and Task 0070 proof and compiled-source runners remain green against the expanded fixed-source corpus, preserving their reviewed numerical and generator claims.

## Validation

The completed change is covered by:

```bash
./native/metal-macos-arm64/build.sh
./backends/metal/evidence/0069/run-proof.sh
./backends/metal/evidence/0070/run-compiled-generated-msl-audit.sh
python3 backends/metal/evidence/0070/verify-source-certificate.py
./backends/metal/evidence/0071/run-proof.sh
./backends/metal/evidence/0070/run-dispatch-observer.sh
```

Focused Java/native coverage includes planner grammar and rejection, schema-17/identity-26 predecessor rejection, manifest/digest determinism, fused/composed trace identity, event ordering, callback `RuntimeException`/`Error` behavior, native failure behavior, sensitive-field exclusion, one-dispatch observer behavior, and CPU-free public Engine execution.

## Non-goals retained

This task adds no operation kind, capability row, native export, public fusion control, benchmark winner, autotuning decision, persistent cache, general graph rewrite system, broad graph tracing, asynchronous execution, lifecycle-close telemetry, native fallback, or new gradient ownership.
