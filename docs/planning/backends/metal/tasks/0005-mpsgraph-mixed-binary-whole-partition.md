# Task 0005: [MPSGraph](../../../../glossary.md#term-mpsgraph) Mixed NEG/[Binary](../../../../glossary.md#term-binary-descriptor-and-metadata) [FLOAT32](../../../../glossary.md#term-float32) Whole-[Partition](../../../../glossary.md#partition) Route

## Reader context

Read both parts of this chronology: the route passed its recorded integration and lifecycle
checks, then a later numerical audit withdrew its binary capability. Passing a storage,
binding or cleanup test does not prove the complete floating-point result contract. Complete
here preserves the original delivery history, not an endorsement of the subsequently withdrawn
arithmetic.

The typed program records described operations, operand references and geometry; actual values
remained in buffers. Metal retained the reusable native executable, while local autorelease
pools drained transient Apple framework objects rather than closing borrowed caller buffers.
The [current numerical and exact domains](../../../../backend-guide/metal-backend.md#ordinary-numerical-and-exact-domains)
explain today's guarded implementation boundary separately from this withdrawal and its old
ABI and schema versions.

## Status

Complete

## Post-completion correctness withdrawal

The [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) route documented below is historical delivery evidence, not current [Metal](../../../../glossary.md#term-metal) capability.
An independent Apple M3 Max audit exercised `ADD`, `SUB`, `MUL`, and `DIV` at [MPSGraph](../../../../glossary.md#term-mpsgraph)
optimization levels `0` and `1`: 576 runs and 79,488 arithmetic comparisons produced 12,096
[subnormal](../../../../glossary.md#term-subnormal)-domain failures while [identity](../../../../glossary.md#term-identity-meanings), [permutation](../../../../glossary.md#permutation), input-preservation, and [canary](../../../../glossary.md#term-canary) controls all
passed. Basic [MSL](../../../../glossary.md#term-metal-shading-language-msl) float arithmetic, including safe math mode, also flushed subnormals. Those
results violate the [Model](../../../../glossary.md#term-model)'s ordinary [IEEE](../../../../glossary.md#term-ieee) [`FLOAT32`](../../../../glossary.md#term-float32) [underflow](../../../../glossary.md#term-underflow) contract.

The accepted urgent remediation therefore removes all four operations from Metal capability and
preparation, removes their Java/native [lowering](../../../../glossary.md#lowering), and makes native [node](../../../../glossary.md#node) wires `2..5` [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation).
Metal-only binary [graphs](../../../../glossary.md#graph) reject during [compile](../../../../glossary.md#compile) [ownership](../../../../glossary.md#term-ownership-meanings) selection; explicitly registered [CPU](../../../../glossary.md#term-cpu) may
own them. [ABI](../../../../glossary.md#term-application-binary-interface-abi) version `4`, its exact thirteen exports, custom `NEG`, ordinary MPSGraph `NEG`, and
the five 0008 affine operations remain. The live fixed-width node [schema](../../../../glossary.md#term-schema) is version `3`; accepted
[operation](../../../../glossary.md#operation) wires are `NEG=1` and affine `6..10`. No fallback, value inspection, relaxed numerical
contract, or custom float emulation was added.

The original Goal, [Scope](../../../../glossary.md#term-scope), Result, and review evidence below are retained only to record what task
0005 delivered before withdrawal.


## Review state

Initial independent Class C review of implementation [commit](../../../../glossary.md#term-commit) `8f117dcfa764483c997767acd6eb202608763ef3`
returned `BLOCK` with two P1 and two P2 findings:

- P1: MPSGraph create and run lacked local Objective-C autorelease pools around transient
  framework objects.<span data-glossary-navigation="true"> ([Objective-C glossary](../../../../glossary.md#native-packaging-and-platform-vocabulary))</span>
- P1: public Engine evidence registered CPU beside Metal and therefore did not prove a CPU-free
  Metal-only builder, independent prepared sessions, or closed-session rejection priority.
- P2: task, master-plan, and roadmap state was not synchronized for the blocked review, and the
  master plan did not record the ABI-v4 cutover and exact export inventory.
- P2: the backend guide misstated executable reuse across sessions, the version-two workload
  signature Javadoc still said version one, and result text overstated public splat evidence.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature))</span>

Remediation commit `9f3a264` added local create/[run](../../../../glossary.md#term-run) pools inside native exception boundaries,
retained the executable box across the pool, ran the 5,000-invocation real [MPSGraph](../../../../glossary.md#term-mpsgraph) stress
[regression](../../../../glossary.md#term-regression), separated the [CPU](../../../../glossary.md#term-cpu)-free caller-input [Engine](../../../../glossary.md#term-engine) test from mixed CPU/[Metal](../../../../glossary.md#term-metal) composition,
deepened [backend](../../../../glossary.md#backend)-local typed-splat proof, and corrected [planning](../../../../glossary.md#term-planning)/[Javadoc](../../../../glossary.md#term-javadoc)/guide evidence.

Independent Class C re-review of `9f3a264` returned `APPROVE` with zero residual findings.
Evidence: native build/export audit passed; [ABI](../../../../glossary.md#term-application-binary-interface-abi) version `4` retained the exact thirteen exports and
the old NEG-only create [symbol](../../../../glossary.md#term-symbol) was absent; real Metal tests passed 55/55 with zero failures,
errors, or skips including the 5,000-run retained-executable stress; focused [conformance](../../../../glossary.md#term-conformance) passed
2/2; real CPU-free Engine integration passed 2/2 with mixed [unary](../../../../glossary.md#term-unary)/[binary](../../../../glossary.md#term-binary-descriptor-and-metadata) [partition](../../../../glossary.md#partition), broadcast,
ordered SUB/DIV, [fan-out](../../../../glossary.md#term-fan-out), repeated runs, independent sessions, direct [publication](../../../../glossary.md#term-publication), close and
rejection, while mixed-[owner](../../../../glossary.md#term-owner) composition remained separate; Javadoc, architecture, public-shape,
documentation, allowlist, and `git diff --check` audits passed.

ABI v4 still has the exact thirteen-symbol inventory recorded in the historical Result below.
Current [schema](../../../../glossary.md#term-schema) `3` keeps its 160-byte typed record, accepts `NEG=1` and affine wires `6..10`, and
rejects the withdrawn binary wires `2..5`.

## Change class

Class C — this crosses public capability truth, whole-[partition](../../../../glossary.md#partition) analysis/[lowering](../../../../glossary.md#lowering), a versioned native [ABI](../../../../glossary.md#term-application-binary-interface-abi), prepared-resource [lifecycle](../../../../glossary.md#lifecycle), and public [Engine](../../../../glossary.md#term-engine)/native integration. It changes observable [backend](../../../../glossary.md#backend) behavior and a high-risk Java/[Objective-C](../../../../glossary.md#native-packaging-and-platform-vocabulary) boundary while preserving the existing custom [singleton](../../../../glossary.md#term-singleton) NEG route.

## Goal

Deliver the next bounded [Metal](../../../../glossary.md#term-metal) vertical [slice](../../../../glossary.md#slice): a static [canonical](../../../../glossary.md#term-canonical-form)-dense [`FLOAT32`](../../../../glossary.md#term-float32) [MPSGraph](../../../../glossary.md#term-mpsgraph) route for one complete maximal Metal-owned [partition](../../../../glossary.md#partition) containing any supported mixture of [unary](../../../../glossary.md#term-unary) `NEG` and [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) arithmetic `ADD`, `SUB`, `MUL`, and `DIV`, with exact right-aligned [broadcasting](../../../../glossary.md#broadcasting). Keep the existing custom [singleton](../../../../glossary.md#term-singleton) NEG implementation and its route boundary intact. Replace the NEG-only MPSGraph native-create [schema](../../../../glossary.md#term-schema) with a versioned typed [graph](../../../../glossary.md#graph) [ABI](../../../../glossary.md#term-application-binary-interface-abi) that describes [operation](../../../../glossary.md#operation) kinds, ordered operands, output values, [descriptors](../../../../glossary.md#term-descriptor), and feed/[target](../../../../glossary.md#term-target) roles without a generic attributes bag. Prove the capability, analysis, native execution, [lifecycle](../../../../glossary.md#lifecycle) [rollback](../../../../glossary.md#term-rollback), and public [`Engine`](../../../../glossary.md#term-engine) behavior on real native tests.

## [Scope](../../../../glossary.md#term-scope)

- Expand [Metal](../../../../glossary.md#term-metal) capability to exactly [unary](../../../../glossary.md#term-unary) `NEG` and [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) `ADD/SUB/MUL/DIV`, all `NoOperationAttrs.INSTANCE`, static positive [rank](../../../../glossary.md#term-rank) `1..16`, [`FLOAT32`](../../../../glossary.md#term-float32), [canonical](../../../../glossary.md#term-canonical-form) dense-[contiguous](../../../../glossary.md#term-contiguous) non-[view](../../../../glossary.md#term-view-meanings) zero-offset [layouts](../../../../glossary.md#layout), and equal `requiresGrad` flags. Binary operands and result must satisfy exact right-aligned [broadcasting](../../../../glossary.md#broadcasting): aligned extents are equal or one, leading missing [axes](../../../../glossary.md#term-axis) are implicit ones, result [shape](../../../../glossary.md#shape) is the aligned maximum, and no other broadcast/layout form is admitted.
- Analyze the complete maximal [partition](../../../../glossary.md#partition) in partition order, preserving stable value/feed/[target](../../../../glossary.md#term-target) order and rejecting malformed or semantically inconsistent projected DAGs. Lower every [node](../../../../glossary.md#node) to typed [MPSGraph](../../../../glossary.md#term-mpsgraph) operations; preserve left/right operand order and broadcast shape semantics. Do not split, fuse, rescore, or route part of a partition.
- Retain `CUSTOM_SINGLE_NEG` exactly for the existing one-NEG/one-feed/one-target/index-domain [singleton](../../../../glossary.md#term-singleton). All other supported partitions, including mixed partitions and multi-node NEG-only partitions, use the generalized MPSGraph route. No custom binary route is introduced.
- Evolve the native [dylib](../../../../glossary.md#term-dylib) to the next exact [ABI](../../../../glossary.md#term-application-binary-interface-abi) version (the implementation must choose and document the concrete version number), replacing the NEG-only [graph](../../../../glossary.md#graph)-create [symbol](../../../../glossary.md#term-symbol)/[schema](../../../../glossary.md#term-schema) with one typed graph-create [operation](../../../../glossary.md#operation). Preserve foundation, custom-pipeline, and graph [release](../../../../glossary.md#term-release)/[run](../../../../glossary.md#term-run) [ownership](../../../../glossary.md#term-ownership-meanings) semantics; no generic [parameter](../../../../glossary.md#term-parameter) map, string [dispatch](../../../../glossary.md#term-dispatch), source text, or untyped operation payload may cross the ABI. Typed enums/records and bounded fixed-width arrays are required.
- Preserve transactional [Prepare](../../../../glossary.md#prepare) finalization, context child [leases](../../../../glossary.md#term-lease), reverse/attempt-all cleanup, close/run leases, isolated per-run [buffers](../../../../glossary.md#term-buffer)/workspaces, direct assigned output [publication](../../../../glossary.md#term-publication), and one [synchronous](../../../../glossary.md#term-synchronous) native invocation per [prepared partition](../../../../glossary.md#prepared-partition--preparedpartition). Graph execution may retain one run-owned address workspace; the custom singleton retains none.
- Add real public [`Engine`](../../../../glossary.md#term-engine) tests using explicit `MetalBackendIntegration` and the freshly built dylib. Cover a [CPU](../../../../glossary.md#term-cpu)-free Metal-only builder with caller inputs, exact broadcast shapes, mixed operations, repeated runs and independently prepared sessions, direct output publication, and close/rejection behavior; keep mixed CPU/Metal composition separate. Prove [compile](../../../../glossary.md#compile)-time `FLOAT32` splats only through the supported [backend](../../../../glossary.md#backend)-local `PrepareContext.constants()` path because public `GraphCompilationPort` has no positive-rank forward-[constant](../../../../glossary.md#term-constant) ingress. Do not add packaging, discovery, tuning, or performance claims.

## Non-goals

No operations beyond `NEG`, `ADD`, `SUB`, `MUL`, `DIV`; no [dtype](../../../../glossary.md#data-type--datatype) beyond [`FLOAT32`](../../../../glossary.md#term-float32); no dynamic/zero [shapes](../../../../glossary.md#shape), unresolved/strided/[view](../../../../glossary.md#term-view-meanings) [layouts](../../../../glossary.md#layout), non-right-aligned [broadcasting](../../../../glossary.md#broadcasting), [scalar](../../../../glossary.md#term-scalar)/generic attrs bag, custom [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) [kernels](../../../../glossary.md#kernel), [CPU](../../../../glossary.md#term-cpu) fallback, repartitioning, async execution, pooling, [cache](../../../../glossary.md#term-cache)/[serialization](../../../../glossary.md#term-serialization), tuning, packaging/[distribution](../../../../glossary.md#term-distribution), or unrelated [Engine](../../../../glossary.md#term-engine)/[API](../../../../glossary.md#term-application-programming-interface-api) redesign. Do not change the custom NEG [kernel](../../../../glossary.md#kernel) source or its native typed pipeline symbols except required [ABI](../../../../glossary.md#term-application-binary-interface-abi) version resolution.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) — [Planning](../../../../glossary.md#term-planning) selects [ownership](../../../../glossary.md#term-ownership-meanings) only; [backend](../../../../glossary.md#backend) analysis/[lowering](../../../../glossary.md#lowering) owns route choice; [Runtime](../../../../glossary.md#term-runtime) sees prepared typed work; [Engine](../../../../glossary.md#term-engine) is [composition root](../../../../glossary.md#term-composition-root).
- [Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)<span data-glossary-navigation="true"> ([Concrete backend glossary](../../../../glossary.md#concrete-backend); [modules glossary](../../../../glossary.md#term-module))</span> — [Metal](../../../../glossary.md#term-metal) owns lowering, native resources, and execution without depending on Engine.
- [`ARCHITECTURE.md` — Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend) — capability truth, [MPSGraph](../../../../glossary.md#term-mpsgraph) lowering, native [ABI](../../../../glossary.md#term-application-binary-interface-abi), and custom route remain Metal-owned.
- [`ARCHITECTURE.md` — Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [lifecycle glossary](../../../../glossary.md#lifecycle))</span> — analysis declares exact resources before assignment; finalization cannot add requirements or reselect.
- [`ARCHITECTURE.md` — Public explicit composition](../../../../architecture/contracts/runtime-prepare-engine.md#public-explicit-composition) — public tests use explicit integration ownership; no discovery or implicit fallback.
- [`ARCHITECTURE.md` — Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run))</span> — prepared recipes and direct cold-bound references only on the [hot path](../../../../glossary.md#term-hot-path).
- [Metal backend guide](../../../../backend-guide/metal-backend.md) and [native guide](../../../../../modules/backend/metal/macos-arm64/README.md) — update current domain, ABI inventory, and [validation](../../../../glossary.md#term-validation) commands.

If the generalized [schema](../../../../glossary.md#term-schema) requires a new shared contract or architecture change, stop and report it before implementation.

## Dependencies and integration

- Depends on: [Metal](../../../../glossary.md#term-metal) 0001–0004; [Engine](../../../../glossary.md#term-engine) 0017; [Compiler](../../../../glossary.md#term-compiler) 0006B7; [Prepare](../../../../glossary.md#prepare) 0008; [Runtime](../../../../glossary.md#term-runtime) 0016; current public mixed-[owner](../../../../glossary.md#term-owner) Engine composition.
- Conflicts with: any task changing Metal capability, `MetalNeg*`/[MPSGraph](../../../../glossary.md#term-mpsgraph) preparation, native Metal [ABI](../../../../glossary.md#term-application-binary-interface-abi), or the shared public Engine Metal integration; serialize those scopes.
- Parallel group: None
- Common base revision: `8c83d01e74f9a0bc57fce84d68ba82f26007a75d` (`main`)
- Integration order: one serial implementation; native ABI/[schema](../../../../glossary.md#term-schema) first, then capability/analysis and Prepare, then Engine/public tests, then documentation and independent review.
- Integration [validation](../../../../glossary.md#term-validation): native build/export audit, focused Metal and [conformance](../../../../glossary.md#term-conformance) tests, real public Engine test with `SYNAPTIK_METAL_TEST_LIBRARY`, architecture/[dependency](../../../../glossary.md#term-dependency) checks, and `git diff --check`.
- Shared-document integration owner: Main planner; implementation may edit only its task-owned docs after behavior stabilizes.

## Files and symbols

- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalCapabilityProvider.java` — exact NEG plus four [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) capability predicates and right-aligned broadcast [validation](../../../../glossary.md#term-validation).
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalBackendRuntime.java` — wire the generalized [partition](../../../../glossary.md#partition) preparer/finalizer/schedule while retaining custom NEG [ownership](../../../../glossary.md#term-ownership-meanings).
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalNativeApi.java` — versioned typed [graph](../../../../glossary.md#graph) [ABI](../../../../glossary.md#term-application-binary-interface-abi) symbols, fixed-width carriers, validation/status mapping, and [fail-closed](../../../../glossary.md#term-fail-closed) resolution order.
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalDeviceContext.java`, `MetalMpsGraphExecutableResource.java`, `MetalNegPartitionFinalizer.java`, `MetalNegPreparedExecutable.java`, `MetalNegPreparedScheduleAssembler.java`, `MetalNegPartitionPreparer.java`, `MetalNegPreparationPlan.java` — generalized graph facts and [operation](../../../../glossary.md#operation) tables; preserve custom [singleton](../../../../glossary.md#term-singleton) pipeline branch and resource/[lifecycle](../../../../glossary.md#lifecycle) invariants. Rename only where required by the final typed ownership model; do not add a generic route framework.
- `native/metal-macos-arm64/src/synaptik_metal_foundation.m`, `native/metal-macos-arm64/README.md` — typed graph-create implementation, exact ABI/version/status/[schema](../../../../glossary.md#term-schema) inventory, [MPSGraph](../../../../glossary.md#term-mpsgraph) arithmetic [lowering](../../../../glossary.md#lowering), and export/linkage documentation.
- `backends/metal/src/test/java/io/github/pho001/synaptik/backend/metal/MetalFoundationTest.java`, `MetalNegPreparedExecutionTest.java`, `MetalNegRouteCandidateGeneratorTest.java`, plus focused generalized graph tests as needed — fake-native ABI/status/[rollback](../../../../glossary.md#term-rollback)/[binding](../../../../glossary.md#term-binding) coverage and existing NEG regressions.
- `testing/backend-conformance/src/test/java/io/github/pho001/synaptik/testing/conformance/MetalNegCapabilityPartitionConformanceTest.java` (rename if needed) — capability and maximal-partition closure for NEG and all four binary kinds, including broadcast rejection boundaries.
- `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/EngineExplicitCompositionMetalIntegrationTest.java` — real public [Engine](../../../../glossary.md#term-engine) mixed-operation whole-partition execution, session reuse, [publication](../../../../glossary.md#term-publication), and lifecycle.
- `docs/backend-guide/metal-backend.md`, `docs/api/public-api.md`, `docs/glossary.md` (targeted terms only), and affected Javadocs — current support/ABI/public-test explanation; no performance or packaging promise.

Exact implementation file additions/renames require the implementer to record the final allowlist in the Result; unrelated paths are forbidden.

## Acceptance criteria

- Capability is true iff the exact static [canonical](../../../../glossary.md#term-canonical-form)-dense [FLOAT32](../../../../glossary.md#term-float32) NEG/[binary](../../../../glossary.md#term-binary-descriptor-and-metadata) domain holds; all unsupported operations, attrs, [dtypes](../../../../glossary.md#data-type--datatype), [shapes](../../../../glossary.md#shape), [layouts](../../../../glossary.md#layout), and broadcast [alignments](../../../../glossary.md#term-alignment) [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation). [Planning](../../../../glossary.md#term-planning) still chooses only [`metal`](../../../../glossary.md#term-metal) [ownership](../../../../glossary.md#term-ownership-meanings).
- Analysis accepts every supported maximal [partition](../../../../glossary.md#partition) [topology](../../../../glossary.md#term-topology) in one route: mixed NEG/binary chains, [fan-out](../../../../glossary.md#term-fan-out), repeated values, multiple feeds/[targets](../../../../glossary.md#term-target), and exact right-aligned broadcasts. Stable feed/target/value ordering is identical across analysis, native create, cold [binding](../../../../glossary.md#term-binding), and [publication](../../../../glossary.md#term-publication).
- [MPSGraph](../../../../glossary.md#term-mpsgraph) computes ordered `ADD`/`SUB`/`MUL`/`DIV` with left/right semantics and NEG correctly; no generic attrs bag, string [operation](../../../../glossary.md#operation) name, hidden copy, or per-[node](../../../../glossary.md#node) native invocation exists.
- The generalized native [ABI](../../../../glossary.md#term-application-binary-interface-abi) has one documented exact version and typed operation/value [schemas](../../../../glossary.md#term-schema), bounded [dimensions](../../../../glossary.md#dimension)/counts, deterministic [validation](../../../../glossary.md#term-validation)/status mapping, exact [symbol](../../../../glossary.md#term-symbol) resolution, and no ABI fallback. Custom pipeline handles remain a distinct typed family and continue to work.
- Finalization cannot change route or declarations. Any native/create/bind/execute/close failure rolls back acquired resources with the original primary and distinct suppressed cleanup failures preserved. Repeated/concurrent sessions have isolated [run](../../../../glossary.md#term-run)-owned outputs/workspaces; caller [buffers](../../../../glossary.md#term-buffer) remain [borrowed](../../../../glossary.md#ownership-borrowing-and-lease).
- Real public `Engine.builder()` tests execute mixed partitions against the built [dylib](../../../../glossary.md#term-dylib), verify direct assigned outputs and numerical results for non-broadcast and broadcast cases, reuse one prepared session across runs, and verify close/rejection behavior. Existing custom [singleton](../../../../glossary.md#term-singleton) NEG real tests remain green.
- No production [dependency direction](../../../../glossary.md#java-platform-and-build-tools), [Engine](../../../../glossary.md#term-engine) [facade](../../../../glossary.md#term-facade) [signature](../../../../glossary.md#term-signature), packaging/discovery behavior, tuning/[cache](../../../../glossary.md#term-cache) behavior, or unrelated operation/dtype/layout [scope](../../../../glossary.md#term-scope) changes.

## [Validation](../../../../glossary.md#term-validation)

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

Integration/repository validation: Main runs the affected-[module](../../../../glossary.md#term-module) suites once after integration, confirms the exact native export set/version and public [Engine](../../../../glossary.md#term-engine) real-[device](../../../../glossary.md#term-device) result, reviews the final allowlist, and records any expected opt-in skips. No full repository suite is required unless [dependency](../../../../glossary.md#term-dependency)/build files change.

## Documentation and review impact

Update [backend](../../../../glossary.md#backend) guide, native README, public [API](../../../../glossary.md#term-application-programming-interface-api) [scope](../../../../glossary.md#term-scope) text, targeted glossary entries, package/type Javadocs, [Metal](../../../../glossary.md#term-metal) master plan, roadmap, and this brief. Class C requires a clean implementation context plus an independent targeted review/documentation context. The reviewer must inspect the exact diff, [ABI](../../../../glossary.md#term-application-binary-interface-abi) [schema](../../../../glossary.md#term-schema)/export audit, capability/[conformance](../../../../glossary.md#term-conformance) [matrix](../../../../glossary.md#term-matrix-meanings), [lifecycle](../../../../glossary.md#lifecycle) [rollback](../../../../glossary.md#term-rollback), real public [Engine](../../../../glossary.md#term-engine) evidence, and documentation for accidental generic attrs, fallback, or scope expansion. Review is mandatory before marking Complete.

## Result

Approved after independent Class C re-review of remediation [commit](../../../../glossary.md#term-commit) `9f3a264` returned `APPROVE`
with zero residual findings. [Metal](../../../../glossary.md#term-metal) capability now admits exactly parameterless `NEG`,
`ADD`, `SUB`, `MUL`, and `DIV` over positive static [rank](../../../../glossary.md#term-rank)-`1..16` [canonical](../../../../glossary.md#term-canonical-form) [contiguous](../../../../glossary.md#term-contiguous) [`FLOAT32`](../../../../glossary.md#term-float32)
[descriptors](../../../../glossary.md#term-descriptor) with equal per-[occurrence](../../../../glossary.md#term-occurrence) [gradient](../../../../glossary.md#term-gradient) flags and exact right-aligned [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) [broadcasting](../../../../glossary.md#broadcasting).
Analysis lowers the entire maximal [partition](../../../../glossary.md#partition) to [immutable](../../../../glossary.md#term-immutable) version-one typed [node](../../../../glossary.md#node) records while
preserving [operation](../../../../glossary.md#operation) [identity](../../../../glossary.md#term-identity-meanings), operand order, stable value/feed/[target](../../../../glossary.md#term-target) order, [fan-out](../../../../glossary.md#term-fan-out), and repeated
inputs. The custom [singleton](../../../../glossary.md#term-singleton)-NEG route is unchanged; every other supported partition uses one
typed whole-partition [MPSGraph](../../../../glossary.md#term-mpsgraph) executable. Existing transactional finalization, direct assigned
outputs, isolated [run state](../../../../glossary.md#run-state--runstate), [rollback](../../../../glossary.md#term-rollback), close/[run](../../../../glossary.md#term-run) [leases](../../../../glossary.md#term-lease), and [concurrency](../../../../glossary.md#term-concurrency) [ownership](../../../../glossary.md#term-ownership-meanings) remain intact.

Native [ABI](../../../../glossary.md#term-application-binary-interface-abi) version `4` exports exactly:

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

The old `synaptik_metal_mpsgraph_neg_executable_create` export is absent. Node [schema](../../../../glossary.md#term-schema) `1` is a
fixed 16-byte record of four [`uint32_t`](../../../../glossary.md#index-mappings-and-cost-dominance) cells: operation (`NEG=1`, `ADD=2`, `SUB=3`, `MUL=4`,
`DIV=5`), first input, second input (`UINT32_MAX` exactly for `NEG`), and output. Java and native
[preflight](../../../../glossary.md#term-preflight) both validate ranks/[dimensions](../../../../glossary.md#dimension), [topology](../../../../glossary.md#term-topology), ordered operands, sentinels, exact NEG [shape](../../../../glossary.md#shape),
exact binary broadcast output, unique feeds/targets/outputs, and checked `FLOAT32` geometry.
There is no compatibility [symbol](../../../../glossary.md#term-symbol), generic attributes bag, string [dispatch](../../../../glossary.md#term-dispatch), source payload, or ABI
fallback.

Remediation [validation](../../../../glossary.md#term-validation) completed on Apple [arm64](../../../../glossary.md#native-packaging-and-platform-vocabulary):

- `./native/metal-macos-arm64/build.sh` passed.
- `file` reported a [Mach-O](../../../../glossary.md#term-mach-o) 64-bit arm64 [dylib](../../../../glossary.md#term-dylib); `nm -gU` reported exactly the thirteen exports
  above with the old NEG-only create symbol absent; `otool -L` reported Foundation, Metal, and
  MetalPerformanceShadersGraph linkage.
- Real-[device](../../../../glossary.md#term-device) `:backends:metal:test --tests '*Metal*' --rerun-tasks` passed 55 tests with zero
  failures, errors, or skips. It includes backend-local typed logical-splat execution and 5,000
  consecutive MPSGraph runs through one retained executable, plus custom/codec, ordering,
  rollback, close/rejection, and concurrency regressions.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [codec glossary](../../../../glossary.md#term-codec))</span>
- `:testing:backend-conformance:test --tests '*Metal*'` passed both focused conformance tests with
  no skips.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance))</span>
- Real-device `EngineExplicitCompositionMetalIntegrationTest` passed two tests with no skip. The
  CPU-free test registers only Metal and exercises all five operations in one caller-input
  broadcast partition, asymmetric ordered <span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>`SUB`/`DIV`, multiple feeds/targets, fan-out, direct
  internal publications, repeated runs, two independently prepared sessions, explicit close, and
  closed-session rejection priority. The mixed-owner test remains separate and covers both
  transfer directions.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
- Metal Javadoc, architecture tests, Engine public-shape tests, targeted documentation/state
  audit, final allowlist audit, and <span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [Engine glossary](../../../../glossary.md#term-engine))</span>`git diff --check` passed.

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

No public Java type or method changed. The package-private `MetalNeg*` class names remain
historical implementation names to avoid a weightless file-renaming cutover; their current
contracts and Javadocs describe the safe NEG-and-affine route. Master plan and roadmap record the
delivered-then-withdrawn 0005 state and retain independent affine-[layout](../../../../glossary.md#layout) task 0014 as the sole
[`Ready`](../../../../glossary.md#term-ready) Metal successor. ABI v4 remains exact.
