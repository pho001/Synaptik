# Task 0005: MPSGraph Mixed NEG/Binary FLOAT32 Whole-Partition Route

## Status

Complete

## Review state

Initial independent Class C review of implementation commit `8f117dcfa764483c997767acd6eb202608763ef3`
returned `BLOCK` with two P1 and two P2 findings:

- P1: MPSGraph create and run lacked local Objective-C autorelease pools around transient
  framework objects.
- P1: public Engine evidence registered CPU beside Metal and therefore did not prove a CPU-free
  Metal-only builder, independent prepared sessions, or closed-session rejection priority.
- P2: task, master-plan, and roadmap state was not synchronized for the blocked review, and the
  master plan did not record the ABI-v4 cutover and exact export inventory.
- P2: the backend guide misstated executable reuse across sessions, the version-two workload
  signature Javadoc still said version one, and result text overstated public splat evidence.

Remediation commit `9f3a264` added local create/run pools inside native exception boundaries,
retained the executable box across the pool, ran the 5,000-invocation real MPSGraph stress
regression, separated the CPU-free caller-input Engine test from mixed CPU/Metal composition,
deepened backend-local typed-splat proof, and corrected planning/Javadoc/guide evidence.

Independent Class C re-review of `9f3a264` returned `APPROVE` with zero residual findings.
Evidence: native build/export audit passed; ABI version `4` retained the exact thirteen exports and
the old NEG-only create symbol was absent; real Metal tests passed 55/55 with zero failures,
errors, or skips including the 5,000-run retained-executable stress; focused conformance passed
2/2; real CPU-free Engine integration passed 2/2 with mixed unary/binary partition, broadcast,
ordered SUB/DIV, fan-out, repeated runs, independent sessions, direct publication, close and
rejection, while mixed-owner composition remained separate; Javadoc, architecture, public-shape,
documentation, allowlist, and `git diff --check` audits passed.

ABI v4 remains exact: typed node schema `1` is the fixed 16-byte four-`uint32_t` record
(`NEG=1`, `ADD=2`, `SUB=3`, `MUL=4`, `DIV=5`, `UINT32_MAX` unary sentinel), with no generic
attributes, compatibility symbol, string dispatch, source payload, or fallback. The exact
thirteen-symbol inventory is recorded in the Result below and in the master plan.

## Change class

Class C — this crosses public capability truth, whole-partition analysis/lowering, a versioned native ABI, prepared-resource lifecycle, and public Engine/native integration. It changes observable backend behavior and a high-risk Java/Objective-C boundary while preserving the existing custom singleton NEG route.

## Goal

Deliver the next bounded Metal vertical slice: a static canonical-dense `FLOAT32` MPSGraph route for one complete maximal Metal-owned partition containing any supported mixture of unary `NEG` and binary arithmetic `ADD`, `SUB`, `MUL`, and `DIV`, with exact right-aligned broadcasting. Keep the existing custom singleton NEG implementation and its route boundary intact. Replace the NEG-only MPSGraph native-create schema with a versioned typed graph ABI that describes operation kinds, ordered operands, output values, descriptors, and feed/target roles without a generic attributes bag. Prove the capability, analysis, native execution, lifecycle rollback, and public `Engine` behavior on real native tests.

## Scope

- Expand Metal capability to exactly unary `NEG` and binary `ADD/SUB/MUL/DIV`, all `NoOperationAttrs.INSTANCE`, static positive rank `1..16`, `FLOAT32`, canonical dense-contiguous non-view zero-offset layouts, and equal `requiresGrad` flags. Binary operands and result must satisfy exact right-aligned broadcasting: aligned extents are equal or one, leading missing axes are implicit ones, result shape is the aligned maximum, and no other broadcast/layout form is admitted.
- Analyze the complete maximal partition in partition order, preserving stable value/feed/target order and rejecting malformed or semantically inconsistent projected DAGs. Lower every node to typed MPSGraph operations; preserve left/right operand order and broadcast shape semantics. Do not split, fuse, rescore, or route part of a partition.
- Retain `CUSTOM_SINGLE_NEG` exactly for the existing one-NEG/one-feed/one-target/index-domain singleton. All other supported partitions, including mixed partitions and multi-node NEG-only partitions, use the generalized MPSGraph route. No custom binary route is introduced.
- Evolve the native dylib to the next exact ABI version (the implementation must choose and document the concrete version number), replacing the NEG-only graph-create symbol/schema with one typed graph-create operation. Preserve foundation, custom-pipeline, and graph release/run ownership semantics; no generic parameter map, string dispatch, source text, or untyped operation payload may cross the ABI. Typed enums/records and bounded fixed-width arrays are required.
- Preserve transactional Prepare finalization, context child leases, reverse/attempt-all cleanup, close/run leases, isolated per-run buffers/workspaces, direct assigned output publication, and one synchronous native invocation per prepared partition. Graph execution may retain one run-owned address workspace; the custom singleton retains none.
- Add real public `Engine` tests using explicit `MetalBackendIntegration` and the freshly built dylib. Cover a CPU-free Metal-only builder with caller inputs, exact broadcast shapes, mixed operations, repeated runs and independently prepared sessions, direct output publication, and close/rejection behavior; keep mixed CPU/Metal composition separate. Prove compile-time `FLOAT32` splats only through the supported backend-local `PrepareContext.constants()` path because public `GraphCompilationPort` has no positive-rank forward-constant ingress. Do not add packaging, discovery, tuning, or performance claims.

## Non-goals

No operations beyond `NEG`, `ADD`, `SUB`, `MUL`, `DIV`; no dtype beyond `FLOAT32`; no dynamic/zero shapes, unresolved/strided/view layouts, non-right-aligned broadcasting, scalar/generic attrs bag, custom binary kernels, CPU fallback, repartitioning, async execution, pooling, cache/serialization, tuning, packaging/distribution, or unrelated Engine/API redesign. Do not change the custom NEG kernel source or its native typed pipeline symbols except required ABI version resolution.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) — Planning selects ownership only; backend analysis/lowering owns route choice; Runtime sees prepared typed work; Engine is composition root.
- [`ARCHITECTURE.md` — Concrete CPU, Metal, and CUDA backends](../../../../../ARCHITECTURE.md#concrete-backend-modules) — Metal owns lowering, native resources, and execution without depending on Engine.
- [`ARCHITECTURE.md` — Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend) — capability truth, MPSGraph lowering, native ABI, and custom route remain Metal-owned.
- [`ARCHITECTURE.md` — Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle) — analysis declares exact resources before assignment; finalization cannot add requirements or reselect.
- [`ARCHITECTURE.md` — Public explicit composition](../../../../architecture/contracts/runtime-prepare-engine.md#public-explicit-composition) — public tests use explicit integration ownership; no discovery or implicit fallback.
- [`ARCHITECTURE.md` — Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle) — prepared recipes and direct cold-bound references only on the hot path.
- [Metal backend guide](../../../../backend-guide/metal-backend.md) and [native guide](../../../../native/metal-macos-arm64/README.md) — update current domain, ABI inventory, and validation commands.

If the generalized schema requires a new shared contract or architecture change, stop and report it before implementation.

## Dependencies and integration

- Depends on: Metal 0001–0004; Engine 0017; Compiler 0006B7; Prepare 0008; Runtime 0016; current public mixed-owner Engine composition.
- Conflicts with: any task changing Metal capability, `MetalNeg*`/MPSGraph preparation, native Metal ABI, or the shared public Engine Metal integration; serialize those scopes.
- Parallel group: None
- Common base revision: `8c83d01e74f9a0bc57fce84d68ba82f26007a75d` (`main`)
- Integration order: one serial implementation; native ABI/schema first, then capability/analysis and Prepare, then Engine/public tests, then documentation and independent review.
- Integration validation: native build/export audit, focused Metal and conformance tests, real public Engine test with `SYNAPTIK_METAL_TEST_LIBRARY`, architecture/dependency checks, and `git diff --check`.
- Shared-document integration owner: Main planner; implementation may edit only its task-owned docs after behavior stabilizes.

## Files and symbols

- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalCapabilityProvider.java` — exact NEG plus four binary capability predicates and right-aligned broadcast validation.
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalBackendRuntime.java` — wire the generalized partition preparer/finalizer/schedule while retaining custom NEG ownership.
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalNativeApi.java` — versioned typed graph ABI symbols, fixed-width carriers, validation/status mapping, and fail-closed resolution order.
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalDeviceContext.java`, `MetalMpsGraphExecutableResource.java`, `MetalNegPartitionFinalizer.java`, `MetalNegPreparedExecutable.java`, `MetalNegPreparedScheduleAssembler.java`, `MetalNegPartitionPreparer.java`, `MetalNegPreparationPlan.java` — generalized graph facts and operation tables; preserve custom singleton pipeline branch and resource/lifecycle invariants. Rename only where required by the final typed ownership model; do not add a generic route framework.
- `native/metal-macos-arm64/src/synaptik_metal_foundation.m`, `native/metal-macos-arm64/README.md` — typed graph-create implementation, exact ABI/version/status/schema inventory, MPSGraph arithmetic lowering, and export/linkage documentation.
- `backends/metal/src/test/java/io/github/pho001/synaptik/backend/metal/MetalFoundationTest.java`, `MetalNegPreparedExecutionTest.java`, `MetalNegRouteCandidateGeneratorTest.java`, plus focused generalized graph tests as needed — fake-native ABI/status/rollback/binding coverage and existing NEG regressions.
- `testing/backend-conformance/src/test/java/io/github/pho001/synaptik/testing/conformance/MetalNegCapabilityPartitionConformanceTest.java` (rename if needed) — capability and maximal-partition closure for NEG and all four binary kinds, including broadcast rejection boundaries.
- `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/EngineExplicitCompositionMetalIntegrationTest.java` — real public Engine mixed-operation whole-partition execution, session reuse, publication, and lifecycle.
- `docs/backend-guide/metal-backend.md`, `docs/api/public-api.md`, `docs/glossary.md` (targeted terms only), and affected Javadocs — current support/ABI/public-test explanation; no performance or packaging promise.

Exact implementation file additions/renames require the implementer to record the final allowlist in the Result; unrelated paths are forbidden.

## Acceptance criteria

- Capability is true iff the exact static canonical-dense FLOAT32 NEG/binary domain holds; all unsupported operations, attrs, dtypes, shapes, layouts, and broadcast alignments fail closed. Planning still chooses only `metal` ownership.
- Analysis accepts every supported maximal partition topology in one route: mixed NEG/binary chains, fan-out, repeated values, multiple feeds/targets, and exact right-aligned broadcasts. Stable feed/target/value ordering is identical across analysis, native create, cold binding, and publication.
- MPSGraph computes ordered `ADD`/`SUB`/`MUL`/`DIV` with left/right semantics and NEG correctly; no generic attrs bag, string operation name, hidden copy, or per-node native invocation exists.
- The generalized native ABI has one documented exact version and typed operation/value schemas, bounded dimensions/counts, deterministic validation/status mapping, exact symbol resolution, and no ABI fallback. Custom pipeline handles remain a distinct typed family and continue to work.
- Finalization cannot change route or declarations. Any native/create/bind/execute/close failure rolls back acquired resources with the original primary and distinct suppressed cleanup failures preserved. Repeated/concurrent sessions have isolated run-owned outputs/workspaces; caller buffers remain borrowed.
- Real public `Engine.builder()` tests execute mixed partitions against the built dylib, verify direct assigned outputs and numerical results for non-broadcast and broadcast cases, reuse one prepared session across runs, and verify close/rejection behavior. Existing custom singleton NEG real tests remain green.
- No production dependency direction, Engine facade signature, packaging/discovery behavior, tuning/cache behavior, or unrelated operation/dtype/layout scope changes.

## Validation

Worker validation:

```bash
./native/metal-macos-arm64/build.sh
file native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
nm -gU native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
otool -L native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
./gradlew :backends:metal:test --tests '*Metal*'
./gradlew :testing:backend-conformance:test --tests '*Metal*'
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" ./gradlew :testing:integration-tests:test --tests '*EngineExplicitCompositionMetalIntegrationTest*'
./gradlew :backends:metal:javadoc
rg -n -i 'MPSGraph|broadcast|binary|ABI|custom.*NEG|Metal' docs/backend-guide/metal-backend.md docs/api/public-api.md docs/glossary.md
./gradlew :testing:architecture-tests:test
./gradlew :modules:engine:test --tests '*Engine*PublicShapeTest'
git diff --check
```

Integration/repository validation: Main runs the affected-module suites once after integration, confirms the exact native export set/version and public Engine real-device result, reviews the final allowlist, and records any expected opt-in skips. No full repository suite is required unless dependency/build files change.

## Documentation and review impact

Update backend guide, native README, public API scope text, targeted glossary entries, package/type Javadocs, Metal master plan, roadmap, and this brief. Class C requires a clean implementation context plus an independent targeted review/documentation context. The reviewer must inspect the exact diff, ABI schema/export audit, capability/conformance matrix, lifecycle rollback, real public Engine evidence, and documentation for accidental generic attrs, fallback, or scope expansion. Review is mandatory before marking Complete.

## Result

Approved after independent Class C re-review of remediation commit `9f3a264` returned `APPROVE`
with zero residual findings. Metal capability now admits exactly parameterless `NEG`,
`ADD`, `SUB`, `MUL`, and `DIV` over positive static rank-`1..16` canonical contiguous `FLOAT32`
descriptors with equal per-occurrence gradient flags and exact right-aligned binary broadcasting.
Analysis lowers the entire maximal partition to immutable version-one typed node records while
preserving operation identity, operand order, stable value/feed/target order, fan-out, and repeated
inputs. The custom singleton-NEG route is unchanged; every other supported partition uses one
typed whole-partition MPSGraph executable. Existing transactional finalization, direct assigned
outputs, isolated run state, rollback, close/run leases, and concurrency ownership remain intact.

Native ABI version `4` exports exactly:

```text
synaptik_metal_foundation_abi_version
synaptik_metal_context_create
synaptik_metal_context_release
synaptik_metal_buffer_create
synaptik_metal_buffer_release
synaptik_metal_buffer_upload
synaptik_metal_buffer_download
synaptik_metal_mpsgraph_executable_create
synaptik_metal_mpsgraph_executable_release
synaptik_metal_mpsgraph_executable_run
synaptik_metal_neg_kernel_pipeline_create
synaptik_metal_neg_kernel_pipeline_release
synaptik_metal_neg_kernel_pipeline_run
```

The old `synaptik_metal_mpsgraph_neg_executable_create` export is absent. Node schema `1` is a
fixed 16-byte record of four `uint32_t` cells: operation (`NEG=1`, `ADD=2`, `SUB=3`, `MUL=4`,
`DIV=5`), first input, second input (`UINT32_MAX` exactly for `NEG`), and output. Java and native
preflight both validate ranks/dimensions, topology, ordered operands, sentinels, exact NEG shape,
exact binary broadcast output, unique feeds/targets/outputs, and checked `FLOAT32` geometry.
There is no compatibility symbol, generic attributes bag, string dispatch, source payload, or ABI
fallback.

Remediation validation completed on Apple arm64:

- `./native/metal-macos-arm64/build.sh` passed.
- `file` reported a Mach-O 64-bit arm64 dylib; `nm -gU` reported exactly the thirteen exports
  above with the old NEG-only create symbol absent; `otool -L` reported Foundation, Metal, and
  MetalPerformanceShadersGraph linkage.
- Real-device `:backends:metal:test --tests '*Metal*' --rerun-tasks` passed 55 tests with zero
  failures, errors, or skips. It includes backend-local typed logical-splat execution and 5,000
  consecutive MPSGraph runs through one retained executable, plus custom/codec, ordering,
  rollback, close/rejection, and concurrency regressions.
- `:testing:backend-conformance:test --tests '*Metal*'` passed both focused conformance tests with
  no skips.
- Real-device `EngineExplicitCompositionMetalIntegrationTest` passed two tests with no skip. The
  CPU-free test registers only Metal and exercises all five operations in one caller-input
  broadcast partition, asymmetric ordered `SUB`/`DIV`, multiple feeds/targets, fan-out, direct
  internal publications, repeated runs, two independently prepared sessions, explicit close, and
  closed-session rejection priority. The mixed-owner test remains separate and covers both
  transfer directions.
- Metal Javadoc, architecture tests, Engine public-shape tests, targeted documentation/state
  audit, final allowlist audit, and `git diff --check` passed.

Final task-owned allowlist:

- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalCapabilityProvider.java`
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalDeviceContext.java`
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalMpsGraphExecutableResource.java`
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalMpsGraphProgram.java`
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalNativeApi.java`
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalNegAnalysisInputs.java`
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalNegPartitionFinalizer.java`
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalNegPartitionPreparer.java`
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalNegPreparationPlan.java`
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalNegPreparedExecutable.java`
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalNegPreparedScheduleAssembler.java`
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalNegRouteCandidateGenerator.java`
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalNegTuningBatch.java`
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalNegTuningCodec.java`
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalNegTuningDecision.java`
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/package-info.java`
- `backends/metal/src/test/java/io/github/pho001/synaptik/backend/metal/MetalCapabilityProviderTest.java`
- `backends/metal/src/test/java/io/github/pho001/synaptik/backend/metal/MetalFoundationTest.java`
- `backends/metal/src/test/java/io/github/pho001/synaptik/backend/metal/MetalNegPreparedExecutionTest.java`
- `backends/metal/src/test/java/io/github/pho001/synaptik/backend/metal/MetalNegRouteCandidateGeneratorTest.java`
- `native/metal-macos-arm64/src/synaptik_metal_foundation.m`
- `native/metal-macos-arm64/README.md`
- `testing/backend-conformance/src/test/java/io/github/pho001/synaptik/testing/conformance/MetalNegCapabilityPartitionConformanceTest.java`
- `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/EngineExplicitCompositionMetalIntegrationTest.java`
- `docs/backend-guide/metal-backend.md`
- `docs/api/public-api.md`
- `docs/glossary.md`
- `docs/planning/backends/metal/master-plan.md`
- `docs/planning/roadmap.md`
- this task brief.

No public Java type or method changed. The package-private `MetalNeg*` class names remain historical
implementation names to avoid a weightless file-renaming cutover; their contracts and Javadocs
describe the generalized typed elementwise route. Master plan and roadmap remain synchronized at
`Complete` after the initial `BLOCK` → remediation → independent `APPROVE` trail above. ABI v4
remains exact and Metal 0006 is now the sole Ready successor.
