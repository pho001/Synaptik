# Task 0023: Exact INT32 GATHER and ONE_HOT

## Status

Draft

Metal 0021 is Complete at implementation `ef2c6a1a`, worker-evidence remediation `be5543f9`,
and final evidence correction `5631d51f`; independent Class C review returned `APPROVE` with zero
findings. This researched successor is intentionally Draft and inactive until Ready Metal 0022
completes the profile-monotonicity correction. Model 0018C–0018D and 0019A2, Compiler 0005C,
CPU 0006A2, Config 0006, and Engine 0018 are Complete. No production edit or Task 0023 runtime
probe has run.

## Change class

Class C — this adds profile-qualified indexing capability, the first typed `INT32` Metal ingress and
splat path, local `BOOL` publication, native pre-write value validation and exceptional diagnostics,
node/cache schema evolution, and public Engine execution across the native boundary. A minimal
real-device smoke and an independent final review are mandatory.

## Goal

Add only canonical positive-rank `INT32`-indexed `GATHER` and trailing-depth `ONE_HOT` under both
`NumericalProfile.STRICT_IEEE` and `NumericalProfile.ACCELERATOR`:

- `GATHER` selects raw `FLOAT32` values exactly, preserving every selected bit;
- `ONE_HOT` produces canonical `BOOL` bytes with a trailing depth axis;
- every logical index is validated before any target write or MPSGraph dispatch, and invalid values
  expose the same deterministic `IndexOutOfBoundsException` messages as the CPU route;
- Metal-local caller/splat ingress and publication widen only as explicitly declared here, while
  CPU↔Metal transfer remains canonical positive-rank `FLOAT32` only.

Keep Compiler production unchanged and make no gather-backward or broader training claim.

## Exact capability and Shape domain

### GATHER

- Admit only `AxisGatherKind.GATHER` with `IndexAxisAttrs`; keep `GATHER_ELEMENTS`, `GATHER_ND`,
  scatter, take conveniences as distinct operations, and every other indexing family false.
- Require exactly two inputs and one output. Data and output are `FLOAT32`; indices are `INT32`.
  Data/output `requiresGrad` values are equal. The integral indices descriptor and every `BOOL`
  descriptor are necessarily non-gradient by the Model invariant.
- Require fully static Shapes with strictly positive extents, checked element/byte geometry, and
  resolved canonical dense-contiguous, zero-offset, non-view layouts for both inputs and output.
- Data rank is `1..16`; indices rank is `1..16`; the normalized axis is in `[0,dataRank)`. The
  exact output Shape is
  `data[0:axis] + indices.shape + data[axis+1:dataRank]`, and its rank
  `dataRank - 1 + indicesRank` must remain in `1..16`.
- The result is canonical. It may feed or be fed by any currently admitted same-profile operation
  whose exact canonical FLOAT32 state requirements are met. No affine-view data/index input and no
  implicit canonicalization are admitted.

### ONE_HOT

- Admit only `OneHotKind.ONE_HOT` with `OneHotAttrs`; require exactly one `INT32` input and one
  `BOOL` output.
- Require a fully static input Shape with strictly positive extents and canonical dense-contiguous,
  zero-offset, non-view input/output layouts. Input rank is `1..15`; output rank is
  `inputRank + 1 <= 16`.
- Require positive `depth`, checked native/element/byte geometry, and output Shape exactly
  `indices.shape + [depth]`. The new one-hot axis is always trailing. Values are fixed to canonical
  `true`/`false`; there is no configurable axis, on-value, or off-value.
- The output is canonical and `requiresGrad=false`. Under the current Metal matrix no operation
  consumes BOOL, so ONE_HOT is a direct partition target/publication only.

### Boundary forms and fail-closed exclusions

- Accept `INT32` indices from caller storage or an authenticated logical splat constant. Repeat a
  splat's exact signed 32-bit value without float conversion. Preserve existing FLOAT32 caller and
  splat behavior for GATHER data.
- Reject before native creation: `INT64` indices; non-FLOAT32 GATHER data; scalar or zero-extent
  Shapes; dynamic dimensions; rank overflow; malformed output Shapes; noncanonical layouts;
  views; byte/element overflow; wrong attributes/signatures; unsupported consumers; and every
  unlisted type or operation.
- Keep Metal-local rank-zero reduction publication and every existing capability unchanged.
  Caller ingress gains only the canonical positive-rank `INT32` form needed here. Metal-local
  publication gains canonical positive-rank `BOOL`. CPU↔Metal transfer remains fully static,
  canonical, positive-rank `FLOAT32` only; no `INT32` or `BOOL` transfer is added.

## Controlling semantics and numerical profiles

The Model sources remain authoritative: `AxisGatherKind`, `IndexAxisAttrs`, `OneHotKind`,
`OneHotAttrs`, `TensorAxisGatherExpressions`, and `TensorOneHotExpressions`. The broader Model
surface permits INT64, scalar/zero-extent forms, every GATHER data type, and any structurally valid
positive long depth; Task 0023 advertises only the bounded subset above.

Indexing/classification receives no numerical-profile relaxation. Both Metal profile matrices must
therefore implement one exact result set:

- GATHER copies the selected FLOAT32 representation bit-for-bit, including both zeros, every finite
  subnormal, normals, infinities, and signed quiet/signaling NaNs with their payloads;
- ONE_HOT writes exactly one byte `1` at each valid selected class and byte `0` everywhere else;
- no DAZ, FTZ, reassociation, FMA, tolerance, reduced precision, conversion, NaN repair, signed-zero
  choice, wrap, clamp, default value, or all-false invalid-index behavior is permitted.

The requested numerical profile remains part of capability, preparation, workload, compatibility,
and route identity even though the operation result set is identical under both profiles.

## Invalid-index policy and deterministic failure

Resolve the cross-backend diagnostic policy in favor of exact CPU-observable parity.

- Before any MPSGraph execution or target write, validate every indexing node's actual bound
  `INT32` Metal input buffer. Visit indexing nodes in stable encoded program order, then logical
  index elements in canonical row-major ordinal order. The first invalid value of the first invalid
  node is the sole reported error.
- A GATHER index is valid exactly when `0 <= value < data.shape[axis]`. On failure throw:
  `IndexOutOfBoundsException("GATHER index at logical position <ordinal> for data axis <axis> is out of bounds: value=<value>, extent=<extent>")`.
- A ONE_HOT index is valid exactly when `0 <= value < depth`. On failure throw:
  `IndexOutOfBoundsException("ONE_HOT index at logical position <ordinal> is out of bounds: value=<value>, depth=<depth>")`.
- Native `synaptik_metal_mpsgraph_executable_run` scans the actual shared MTLBuffer contents and
  returns existing status `RANGE_OUT_OF_BOUNDS` before dispatch. On that exceptional status only,
  Java downloads/rescans the immutable bound index buffers in the same node/ordinal order to
  reconstruct the exact exception class, message, ordinal, signed value, axis, and bound. If the
  Java rescan cannot reproduce the native result, propagate the original native failure rather
  than inventing a diagnostic.
- Invalid execution must not dispatch the MPSGraph executable and must leave every caller-supplied
  target byte and canary unchanged. No output clearing, partial write, retry, CPU result fallback,
  or selector-dependent behavior is allowed.
- Validation metadata is immutable prepared executable state. The valid hot path retains one Java
  run downcall and one synchronous MPSGraph submission; error-only download/rescan is not a normal
  route or undeclared workspace.

## Selector evidence policy and minimal device smoke

Apple's MPSGraph documentation is authoritative for selector availability, accepted type/Shape
forms, and valid-index GATHER/ONE_HOT mapping. Task 0023 must use only the documented selectors:

- `gatherWithUpdatesTensor:indicesTensor:axis:batchDimensions:name:` with
  `batchDimensions=0`;
- `oneHotWithIndicesTensor:depth:dataType:onValue:offValue:name:` with trailing depth,
  `MPSDataTypeBool`, on-value `1.0`, and off-value `0.0`.

Do not create a context/optimization/executable cross-product probe. One small real-device smoke is
sufficient: execute a representative valid GATHER corpus containing both zero signs, subnormal,
ordinary, infinity, and NaN payload bits and verify selected raw bits; execute one representative
ONE_HOT input and verify every output byte and Shape. Use guarded caller-supplied direct targets,
check input preservation and canaries, and remove any disposable smoke artifact.

Ordinary-GATHER and ONE_HOT out-of-bounds selector behavior is undocumented and is not part of the
implementation contract. Host/native prevalidation tests must instead prove negative,
exactly-equal-to-bound, later-ordinal, and multi-node invalid inputs are rejected before selector
dispatch, preserve target bytes/canaries, and reconstruct the deterministic CPU-parity diagnostic.
No selector observation may weaken that validation requirement.

More generally, a numerical device probe is required only when authoritative documentation is
silent and a new floating numerical capability claim depends on device behavior. Such a probe must
use the smallest adversarial corpus that can decide the claim; combinatorial optimization,
context, executable, repetition, or topology matrices are prohibited unless a concrete observed
risk independently justifies them.

## Native schema, identity, lifecycle, and cache

- Keep ABI v4, all current statuses and signatures, and exactly thirteen exports. Do not add a
  type-table parameter or diagnostic export. Native schema validation infers a closed per-value
  type state from topology: all existing nodes are FLOAT32; GATHER first input/output are FLOAT32
  and second input is INT32; ONE_HOT input is INT32 and output is BOOL. Reject inconsistent reused
  values or unknown feed/target types before graph compilation.
- Evolve the fixed 160-byte MPSGraph node schema `8 -> 9`. Retain wires `1..15`; append only
  `GATHER=16` and `ONE_HOT=17`. Reuse attribute kind `AXIS=3` for the normalized GATHER axis.
  Append only `DEPTH=5` for ONE_HOT, with one positive 64-bit depth in `attributeValues[0]` and all
  unused fields/reserved/padding zero. Unknown wires `18+`, unknown attributes `6+`, stale schema,
  malformed input slots, wrong states, invalid Shapes, and nonzero sentinels fail before resource
  creation.
- Compile placeholders and direct target `MPSGraphTensorData` with inferred
  `MPSDataTypeFloat32`, `MPSDataTypeInt32`, or `MPSDataTypeBool`; compute exact per-value/feed/target
  bytes from the inferred width rather than assuming four bytes. Store immutable validation records
  and stable feed/target type/permutation arrays in the native executable box.
- Keep the existing whole-partition `MPSGRAPH` candidate. Analysis declares all buffers and existing
  binding workspace exactly; no custom compute kernel, late route, retry, or result fallback is
  added. Finalization remains transactional and creates one shape-specialized executable.
- Preserve direct targets, leases, rollback and suppressed failures, reuse, concurrency,
  close/quiescence, cleanup, context/session isolation, splat initialization, and binding order.
  Exceptional diagnostic download must use the already bound exact Metal buffer and must not retain
  an error-path host allocation after the call.
- Advance all current identity meanings `9 -> 10`:
  `MetalNegRouteCandidateGenerator.WORKLOAD_SIGNATURE_VERSION`, `EXACT_DEFAULT_POLICY`,
  `MetalNegTuningBatch.CANDIDATE_SCHEMA_VERSION`, `COMPATIBILITY_SCHEMA_VERSION`,
  `ROUTE_POLICY_VERSION`, and `MetalNegTuningCodec.CODEC_VERSION`. Identity covers profile, schema,
  operation/attributes, ordered topology, inferred types/widths, descriptors/states, axis/depth,
  validation policy, feeds/targets, ABI, and typed splats. Version-nine, cross-profile, corrupt,
  malformed, stale-session, changed-type/Shape/topology, and foreign decisions reject or miss
  safely.

## Compiler, publication, and backward boundary

No Compiler production file changes.

- Existing Compiler forward capture and `PublicationPlan` already publish selected GATHER and
  ONE_HOT outputs. Task 0023 changes only Metal-local binding/materialization needed to realize
  those publications.
- `IndexingGradientRules` already defines GATHER's data cotangent as
  `zeroLike(data).scatterAdd(indices, gradient, axis)` and no indices cotangent. Metal does not
  admit `SCATTER_ADD`, and INT32 cross-owner transfer remains false. Do not claim Metal-only or
  mixed-owner gather backward, widen transfer, or add a decomposition/fallback.
- ONE_HOT produces non-differentiable BOOL with no Tensor cotangent route under
  `FirstOrderGradientCoverage`/`AutogradPreflight`; leave that behavior unchanged.
- Public Engine proof is forward-only. A Metal-only gradient request containing the generated
  scatter path must fail ownership selection/preparation before Metal native creation; tests must
  not turn that negative into a backward capability claim.

## Non-goals

No INT64; non-FLOAT32 GATHER data; GATHER_ELEMENTS/GATHER_ND/scatter; scalar, dynamic, or zero-extent
form; arbitrary view; BOOL consumer; configurable one-hot axis/value; INT32/BOOL transfer; custom
indexing kernel; host-computed result; selector-dependent OOB policy; Compiler production change;
gather gradient execution; broader training; packaging/discovery; public Java signature;
dependency; Runtime/Trace profile state; or performance claim.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants)
- [Foundational modules — Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)
- [Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)
- [Backend execution — Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)
  and [Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle),
  [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle), and
  [public explicit composition](../../../../architecture/contracts/runtime-prepare-engine.md#public-explicit-composition)

If implementation needs wider Model semantics, INT32/BOOL cross-owner transfer, a new native export
or ABI signature, a custom result kernel, host result fallback, Compiler rewriting, Runtime/Trace
profile state, or a different invalid-index diagnostic, stop and report the contract conflict.

## Dependencies and integration

- Depends on: Metal 0022 Complete; Metal 0021 Complete; Model 0018C–0018D and 0019A2; Compiler
  0005C; CPU 0006A2 as the observable indexing oracle; Config 0006; Engine 0018
- Conflicts with: any Metal 0016 restart and every Metal capability/preparation/native schema/
  candidate/codec/ingress/materialization/public Engine scope
- Parallel group: None
- Common base revision: N/A until Metal 0022 is Complete and this Draft is reverified for promotion
- Integration order: Serial after Metal 0022; when promoted, documented-selector implementation
  with the minimal valid-value device smoke, then schema/type/prevalidation, capability/topology,
  lifecycle, Engine proof, documentation, and independent Class C review
- Integration validation: exact INT32 GATHER/ONE_HOT Metal Class C checkpoint
- Shared-document integration owner: task implementer, finalized by independent Class C review

## Files and symbols

- Model contracts, read-only: `AxisGatherKind`, `IndexAxisAttrs`, `OneHotKind`, `OneHotAttrs`,
  `TensorAxisGatherExpressions`, `TensorOneHotExpressions`.
- Compiler contracts, unchanged: `IndexingGradientRules`, `FirstOrderGradientCoverage`,
  `AutogradPreflight`, `GraphCompiler`, `PublicationPlan`; existing
  `IndexingScatterGradientRulesTest` remains the formula guard.
- CPU oracle, unchanged: `CpuCapabilityProvider.supportsIndexing`, `CpuIndexingLowering`,
  `CpuPreparedExecutable.IndexValidation`, `CpuScalarReferenceKernel.execute`, and focused
  indexing/one-hot generated, ordering, differential, and semantic-closure tests.
- `MetalCapabilityProvider`, `MetalNegPartitionPreparer`, `MetalNegPreparationPlan` — exact
  both-profile capability, canonical state, type/Shape/byte/topology validation, typed splats, and
  retained validation records.
- `MetalMpsGraphProgram`, `MetalNativeApi.MpsGraphExecutableAbi`, `MetalDeviceContext`,
  `MetalMpsGraphExecutableResource` — schema nine, wires 16/17, depth attribute five, inferred type
  state, typed data binding, exact diagnostics, and ABI-v4 stability.
- `MetalNegPartitionFinalizer`, `MetalNegPreparedExecutable`,
  `MetalNegPreparedScheduleAssembler`, `MetalBackendRuntime`, `MetalBackendIntegration` — prepared
  resources, INT32 caller/splat ingress, BOOL local publication, exceptional rescan, lifecycle, and
  unchanged transfer predicate.
- `MetalNegRouteCandidateGenerator`, `MetalNegTuningBatch`, `MetalNegTuningDecision`,
  `MetalNegTuningCodec` — version-ten deterministic identity and stale rejection.
- `native/metal-macos-arm64/src/synaptik_metal_foundation.m` — schema-nine validation/lowering,
  type inference, MPSGraph selectors, typed tensor data, stable permutations, and pre-dispatch
  index scan.
- `MetalCapabilityProviderTest`, `MetalMpsGraphAffineSchemaTest`,
  `MetalMpsGraphRawAbiNativeTest`, `MetalNegPreparedExecutionTest`,
  `MetalNegRouteCandidateGeneratorTest`, and new `MetalMpsGraphIndexingNativeTest`.
- `testing/backend-conformance/.../MetalNegCapabilityPartitionConformanceTest` and
  `testing/integration-tests/.../EngineExplicitCompositionMetalIntegrationTest`.
- Affected Metal/native Javadocs, package information, backend/native/API/user/capability/preparer
  guides, targeted glossary, authoritative current Metal capability text, and synchronized
  planning after executable behavior passes.

## Test and checkpoint requirements

Focused capability tests cover the exact positive rows under both profiles and representative
wrong kind, attr, type, rank, axis/depth, output Shape, layout/state, INT64, view, and transfer
boundaries. Conformance proves maximal same-owner closure for GATHER with existing canonical
operations and terminal ONE_HOT publication without admitting a BOOL consumer.

Schema/raw-native tests prove schema-nine encoding, retained wires `1..15`, appended wires `16/17`,
`AXIS=3`, `DEPTH=5`, inferred type consistency and byte widths, stale/unknown wire rejection, and
malformed topology/Shape/state failure before resource publication. Prepared tests cover exact
typed caller/splat ingress, GATHER selected raw bits, canonical ONE_HOT bytes, direct
materialization, and lifecycle through existing unchanged generic resource tests.

Focused host/native prevalidation tests instrument dispatch and prove negative,
exactly-equal-to-bound, later-ordinal, and multi-node invalid inputs produce the deterministic first
failure, zero selector dispatch, unchanged prefilled targets/canaries, and exact exception text.

One small real CPU-free Engine smoke runs representative forward GATHER and ONE_HOT, materializes
exact FLOAT32 bits and BOOL bytes, and proves Metal-only ownership. Separate negatives prove
unsupported forms fail before native creation, Metal-only gather gradient is unsupported, and
CPU↔Metal transfer remains FLOAT32-only. Existing generic tests own unchanged reuse, concurrency,
session/context, rollback, cleanup, and close behavior and must not be duplicated.

Update architecture/current-capability text, public API status, Metal backend/native guides,
Javadocs, capability/preparer status, targeted glossary, and planning only after behavior is proven.
Do not change Compiler production documentation to imply a new gradient route.

## Acceptance criteria

1. The documented MPSGraph selectors and type/Shape/valid-index mapping are used directly. One
   minimal real-device smoke preserves representative selected GATHER raw bits and exact ONE_HOT
   bytes/Shape; no combinatorial executable/run matrix is added.
2. Both profiles admit exactly the bounded canonical INT32 GATHER/ONE_HOT domain. Every excluded
   kind/type/rank/Shape/layout/state/attribute/transfer/backward form rejects before native
   creation, and all prior capability remains unchanged.
3. Every valid GATHER result preserves selected FLOAT32 bits exactly and every ONE_HOT target byte
   is canonical. No profile relaxation, conversion, invalid default, host result fallback, or
   selector-dependent semantics exists.
4. Native validates all indexing inputs before target writes/dispatch. Focused instrumented gates
   prove deterministic node/ordinal first error, exact CPU-parity `IndexOutOfBoundsException`
   messages, zero selector dispatch, and byte-for-byte unchanged targets/canaries for documented
   OOB classes.
5. ABI v4 retains exactly thirteen exports; schema nine appends only wires 16/17 and attribute
   depth five; Java/native type, Shape, byte, state, and topology validation agree. Version-ten
   identities reject old, cross-profile, changed-type/Shape/topology, corrupt, stale, and foreign
   data.
6. Focused prepared proof covers typed caller/splat ingress, BOOL publication, direct targets, and
   valid-path one-downcall/one-submission behavior while existing generic lifecycle coverage remains
   unchanged. Cross-owner transfer remains exact FLOAT32-only.
7. One real CPU-free Engine forward smoke proves Metal-only execution and exact materialization.
   Existing Compiler gather-scatter formula and ONE_HOT nondifferentiability stay unchanged; no
   gather backward claim exists.
8. Native/export, focused Metal/Javadoc, conformance, Compiler formula guard, real Engine,
   architecture, full-build, Markdown/status/frontier, smoke-artifact removal, and diff checks
   pass; independent Class C review has no unresolved finding.

## Validation

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

Record the documented selectors, minimal valid-value device smoke and artifact removal,
pre-dispatch OOB evidence, exact exception messages and unchanged targets; exports/schema/wires/
attribute/identity versions; focused test counts/skips; both-profile positives and negatives;
typed ingress/publication and unchanged transfer; Compiler no-production-change evidence; changed
paths; and synchronized frontier/status. The serial worker owns one full checkpoint. Independent
review reuses evidence and reruns only for a concrete risk or relevant executable change.

## Documentation and review impact

Independent Class C review inspects documented-selector use, the minimal raw-bit/canonical-BOOL
smoke and artifact removal, OOB non-reliance, native prevalidation and Java exceptional rescan,
deterministic first error, unchanged targets, schema/type inference, version-ten identity, typed
ingress/local publication versus unchanged transfer, focused lifecycle reliance, the real
Metal-only Engine smoke, forward-only boundary, Runtime/Trace absence, changed scope, and all
validation/documentation evidence.
