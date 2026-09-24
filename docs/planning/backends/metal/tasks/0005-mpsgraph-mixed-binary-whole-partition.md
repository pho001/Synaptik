# Task 0005: MPSGraph Mixed NEG/Binary FLOAT32 Whole-Partition Route

## Status

Ready

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
- Add real public `Engine` tests using explicit `MetalBackendIntegration` and the freshly built dylib. Cover caller inputs, compile-time splat where already supported, exact broadcast shapes, mixed operations, repeated prepared sessions/runs, direct output publication, and closure/rollback. Do not add packaging, discovery, tuning, or performance claims.

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

Empty until execution. On completion record changed files, exact commands/outcomes and skips, native export/version evidence, public Engine real-device evidence, documentation/review findings, limitations, and follow-up.
