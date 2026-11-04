# Task 0010F: Supported [CPU](../../../../glossary.md#term-cpu) [Lifecycle](../../../../glossary.md#lifecycle) Integration Adapter

## Reader context — historical integration boundary

The caller of this adapter is Engine's composition layer, not necessarily the application. The
application uses Engine; Engine obtains CPU capability, preparation and borrowing services through
this supported service-provider interface. CPU then supplies immutable recipes that Runtime realizes
with fresh run-owned buffers. `borrow(HostTensorStorage)` creates a non-owning representation: it
does not copy bytes or take responsibility for closing the original array/arena. It can check
intrinsic storage consistency but cannot compare against an expected logical input absent from that
call. This record's direct Compiler dependency and `preparations(CompileArtifacts)` signature are
historical; the maintained integration explanation describes the later shared-Prepare projection
boundary.

Current reading: [Supported lifecycle integration](../../../../backend-guide/cpu-backend.md#supported-lifecycle-integration).

## Status

Complete

## Goal

Publish the smallest [CPU](../../../../glossary.md#term-cpu)-owned, public-for-[module](../../../../glossary.md#term-module)-integration [lifecycle](../../../../glossary.md#lifecycle) adapter that future
[Engine](../../../../glossary.md#term-engine) code can consume without importing `io.github.pho001.synaptik.backend.cpu.internal`.
The adapter composes the existing CPU capability provider, [host](../../../../glossary.md#term-host) availability, [partition](../../../../glossary.md#partition) analysis
and finalization, physical representation recipes, deterministic schedule assembly, portable
execution, and bounded automatic [OpenBLAS](../../../../glossary.md#term-openblas) discovery behind one explicitly owned [lifetime](../../../../glossary.md#term-lifetime).

The type is cross-module [SPI](../../../../glossary.md#term-service-provider-interface-spi), not an ordinary user [facade](../../../../glossary.md#term-facade). Engine later owns both its advanced
explicit composition path and the standard built-in factory through which ordinary users obtain
CPU execution without constructing or naming this adapter.

## [Scope](../../../../glossary.md#term-scope)

- Add one public final `AutoCloseable` integration type,
  `io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration`, in the existing supported CPU
  package. Give it a private constructor and one public static `open()` factory for the fixed
  exact/default built-in composition.
- Expose only these public instance collaborations, using existing public shared types in every
  signature:<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature))</span>
  - `capabilityProvider()` returns the retained `CpuCapabilityProvider`;
  - `availabilitySnapshot()` returns one [immutable](../../../../glossary.md#term-immutable) [CPU](../../../../glossary.md#term-cpu) [snapshot](../../../../glossary.md#term-snapshot) containing the stable [device](../../../../glossary.md#term-device)
    `BackendDeviceId(CpuCapabilityProvider.CPU_BACKEND_ID, "host")` classified as
    `DeviceClass.CPU`;
  - `preparations(CompileArtifacts)` accepts only a compile artifact containing exactly one
    non-empty maximal partition owned by CPU and returns an immutable singleton positional<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [artifact glossary](../../../../glossary.md#term-artifact); [partition glossary](../../../../glossary.md#partition); [singleton glossary](../../../../glossary.md#term-singleton))</span>
    `List<PartitionPreparation<?, ?>>`, with exact CPU-private analysis inputs, one retained
    preparer, and one retained finalizer hidden behind shared Prepare roles;<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare))</span>
  - `scheduleAssembler()` returns the retained `PreparedScheduleAssembler` that builds the CPU
    representation, execution, constant-initialization, and publication recipe described below;<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [publication glossary](../../../../glossary.md#term-publication))</span>
  - `borrow(HostTensorStorage)` returns a `BufferRepresentation` for one intrinsically valid
    representation-level caller storage without exposing `CpuBorrowedBuffer`. Before returning,
    it validates only facts supplied by that storage: non-nullity, current scope liveness and
    current-thread segment accessibility, checked equality among element capacity, data-type byte
    width, reported byte size, and exact segment byte size, and supported data-type/observable
    heap-carrier compatibility. It receives no prepared binding expectation; and<span data-glossary-navigation="true"> ([liveness glossary](../../../../glossary.md#term-liveness); [element glossary](../../../../glossary.md#term-element); [heap glossary](../../../../glossary.md#term-heap); [carrier glossary](../../../../glossary.md#term-storage-carrier); [binding glossary](../../../../glossary.md#term-binding))</span>
  - `close()` owns deterministic composition cleanup.
- Add the direct `implementation(project(":modules:compiler"))` dependency required because the
  supported <span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>`preparations(CompileArtifacts)` signature and implementation directly consume
  Compiler-owned artifacts. Do not obtain Compiler accidentally through Prepare or another
  transitive dependency.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [transitive dependency glossary](../../../../glossary.md#term-transitive-dependency))</span>
- Clarify `docs/architecture/dependency-rules.md` narrowly: concrete CPU now realizes the already
  permitted direct Compiler edge because it consumes public `CompileArtifacts`; the execution-side
  direction, all other dependency rules, root `ARCHITECTURE.md`, and ADRs remain unchanged.
- Add one focused `CpuDependencyContractTest` under `testing/architecture-tests` that locks CPU's
  exact direct dependency set with Compiler permitted and Engine forbidden.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span>
- Keep every existing CPU preparation plan, analysis input, executable, representation,
  OpenBLAS discovery/qualification/coordinator, worker, and schedule-assembly implementation type
  under <span data-glossary-navigation="true"> ([qualification glossary](../../../../glossary.md#term-qualification))</span>`.internal`.
- At `open()`, construct one capacity-one CPU concurrency budget; attempt the existing bounded
  automatic OpenBLAS discovery;
  and, when loading succeeds, transfer the provider owner into one coordinator, qualify that exact
  installed target, and retain it for the adapter lifetime. Discovery, load, qualification, or
  coordinator setup failure closes partial resources and leaves a fully usable portable adapter.<span data-glossary-navigation="true"> ([CPU concurrency budget glossary](../../../../glossary.md#cpu-concurrency-budget--cpuconcurrencybudget); [owner glossary](../../../../glossary.md#term-owner); [target glossary](../../../../glossary.md#term-target); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- Use one CPU-private immutable exact/default untuned preparation profile. Preserve current
  scalar, single-thread portable defaults and direct-only materialization. It has no selected
  tuning decision and admits only the already-supported FLOAT32/FLOAT64 OpenBLAS route with the
  single proved thread-count candidate <span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [materialization glossary](../../../../glossary.md#materialization); [selected tuning decision glossary](../../../../glossary.md#selected-tuning-decision); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64))</span>`1`. Reuse the established native-checkpoint selection
  constants exactly: portable costs <span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>`(fixed=1_000, perOutput=10, perMac=100)`, [OpenBLAS](../../../../glossary.md#term-openblas) costs
  `(fixed=1, perOutput=1, perMac=1)`, zero representation-transition costs, minimum absolute
  benefit `1`, and minimum relative benefit `1` basis point. These are deterministic safe-
  heuristic policy inputs, not benchmark evidence or a performance claim.<span data-glossary-navigation="true"> ([heuristic glossary](../../../../glossary.md#term-heuristic))</span>
- Validate the complete compile partition list before constructing analysis inputs: reject a
  zero-partition/pass-through artifact, more than one partition, an empty partition, or any
  non-CPU owner before backend analysis. For the sole accepted partition, derive carrier,
  native-storage, execution, and route facts from the exact compile artifacts; reject incomplete
  positional coverage, unresolved physical geometry, a closed adapter, or an unsupported
  representation before analysis/finalization can escape the adapter.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [compile artifacts glossary](../../../../glossary.md#compile-artifacts))</span>
- Make the retained assembler independently reject any schedule context that does not contain
  exactly one non-empty CPU-owned compile/prepared partition before constructing a representation
  plan or schedule. For the accepted context, assemble one immutable schedule with this order:
  one representation-creation prefix; the sole partition execution step; then publication steps
  in compile publication order. Create fresh run-owned aligned CPU buffers and workspaces from the
  exact prepared memory entries, create and initialize compiler constant sources once per run,
  mark bindable inputs as caller-input occurrences, and select the exact CPU representation used
  by the finalized executable and publications. Perform no allocation, copy, execution, or
  creator invocation while assembling.<span data-glossary-navigation="true"> ([prepared partition glossary](../../../../glossary.md#prepared-partition--preparedpartition); [run glossary](../../../../glossary.md#term-run); [buffers glossary](../../../../glossary.md#term-buffer))</span>
- Keep representation creators and the assembled schedule immutable and safe for concurrent
  reuse. Each run receives distinct owned buffers/workspaces; borrowed input ownership stays with
  the caller and Runtime closes only run-owned representations through its existing rules.<span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Add focused CPU tests for the public shape, portable lifecycle, schedule recipe, concurrent
  reuse, cleanup/failure rollback, and native-free automatic-discovery fallback. Extend the
  existing CPU/OpenBLAS backend-conformance coverage only where it proves the supported adapter
  crosses the real Prepare and Runtime recipe boundary.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [lifecycle glossary](../../../../glossary.md#lifecycle); [rollback glossary](../../../../glossary.md#term-rollback); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Require a separate clean documentation-focused pass after Java implementation to finalize all
  affected Javadocs, supported-API/backend guidance, glossary impact, and planning evidence.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- Any [Engine](../../../../glossary.md#term-engine) production type or task specification, including Engine 0001 or later.
- An ordinary-user CPU builder, CPU option/configuration object, provider-name/path override,
  enable/disable switch, public cost/thread knobs, or CPU type in ordinary Engine signatures.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [signatures glossary](../../../../glossary.md#term-signature))</span>
- Config aggregates (`CompileConfig`, `PrepareConfig`, or `RunOptions`), Config 0006 relaxed
  permissions, or Config 0006A mapping.
- Tuning measurement, candidate execution, selection, decision/cache lookup, persistence,
  workload extraction, or tools/tuning integration. The adapter supplies no<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [persistence glossary](../../../../glossary.md#term-persistence))</span>
  `CpuOpenBlasTuningDecision`.
- [Runtime](../../../../glossary.md#term-runtime) [backend](../../../../glossary.md#backend) registration, selection, discovery, provider [lookup](../../../../glossary.md#term-lookup), or route fallback.
- Reflection, `ServiceLoader`, classpath/annotation scanning, service location, mutable global
  registry, or process-global singleton ownership.<span data-glossary-navigation="true"> ([classpath glossary](../../../../glossary.md#term-classpath); [mutable glossary](../../../../glossary.md#term-mutable); [registry glossary](../../../../glossary.md#term-registry); [singleton glossary](../../../../glossary.md#term-singleton); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- New OpenBLAS symbols, provider API changes, name/path selection, BFLOAT16 OpenBLAS, optional
  vendor peers, relaxed numerics, or cross-route CPU tuning.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16))</span>
- Typed logical Engine input mapping, arbitrary heap-carrier adaptation, host-result
  materialization, one-shot execution, or user-visible result access; those remain Engine
  0003–0005 work.<span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap); [carrier glossary](../../../../glossary.md#term-storage-carrier); [host glossary](../../../../glossary.md#term-host); [materialization glossary](../../../../glossary.md#materialization))</span>
- Successful preparation or schedule assembly for zero-partition/pass-through artifacts. This is
  the existing Prepare 0003 limitation: requested values have no backend-declared buffer geometry
  or <span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [artifacts glossary](../../../../glossary.md#term-artifact); [Prepare glossary](../../../../glossary.md#prepare); [buffer glossary](../../../../glossary.md#term-buffer))</span>[`PreparedBufferAssignment`](../../../../glossary.md#prepared-buffer-assignment--preparedbufferassignment), so [`GraphPreparation`](../../../../glossary.md#graph-preparation--graphpreparation) intentionally fails closed rather than
  inventing allocation policy.
- Successful CPU-only assembly of a multi-partition artifact or mixed-backend schedule
  composition. Compiler/Planning already makes every non-empty all-CPU graph one maximal CPU
  partition; any valid multi-partition artifact therefore contains another owner. The later
  Engine/Prepare composition boundary must decide how multiple backend assemblers contribute one
  complete schedule.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning); [graph glossary](../../../../glossary.md#graph); [owner glossary](../../../../glossary.md#term-owner))</span>
- Comparing borrowed storage with an expected logical tensor data type, required element/byte
  capacity, layout span, or read/write access role. <span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [tensor glossary](../../../../glossary.md#tensor); [data type glossary](../../../../glossary.md#data-type--datatype); [element glossary](../../../../glossary.md#term-element); [layout glossary](../../../../glossary.md#layout))</span>`borrow(HostTensorStorage)` receives none of
  those target facts. Engine 0003 must perform that typed logical binding validation before
  Runtime execution; CPU 0010F must not guess or manufacture the missing expectation.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [binding glossary](../../../../glossary.md#term-binding); [validation glossary](../../../../glossary.md#term-validation))</span>
- New CPU semantics, lowering, generated kernel forms, generator schema, artifact identity,
  generated bytecode, generated/direct oracle, or performance claims.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [kernel glossary](../../../../glossary.md#kernel); [schema glossary](../../../../glossary.md#term-schema); [identity glossary](../../../../glossary.md#term-identity-meanings); [bytecode glossary](../../../../glossary.md#term-bytecode); [oracle glossary](../../../../glossary.md#term-test-oracle))</span>
- Shared Prepare, Runtime, Backend Contract, Compiler, Config, Planning, Model, Trace, or provider
  Java changes. The only build/dependency change is the exact direct CPU-to-Compiler dependency;
  the only architecture-document/test changes are the focused dependency-rules clarification and
  CPU dependency-contract enforcement specified below.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Trace glossary](../../../../glossary.md#trace); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially lifecycle, Engine, Prepare,
  Runtime, concrete-backend, resource-ownership, and dependency rules.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership))</span><span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [Engine glossary](../../../../glossary.md#term-engine); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- [`docs/architecture/lifecycle.md`](../../../../architecture/lifecycle.md)
- [`docs/architecture/module-boundaries.md`](../../../../architecture/module-boundaries.md)
- [`docs/architecture/dependency-rules.md`](../../../../architecture/dependency-rules.md)
- [`docs/architecture/runtime-prepare-backend-boundary.md`](../../../../architecture/runtime-prepare-backend-boundary.md)
- [`docs/planning/planning-guide.md`](../../../planning-guide.md)
- [`docs/planning/backends/cpu/master-plan.md`](../master-plan.md)
- [`docs/planning/modules/prepare/master-plan.md`](../../../modules/prepare/master-plan.md)
- [`docs/planning/modules/runtime/master-plan.md`](../../../modules/runtime/master-plan.md)
- [`docs/planning/modules/engine/master-plan.md`](../../../modules/engine/master-plan.md)
- [CPU 0010A automatic OpenBLAS discovery](0010a-automatic-openblas-discovery-and-internal-composition-foundation.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [OpenBLAS glossary](../../../../glossary.md#term-openblas))</span>
- [CPU 0010C coordinated OpenBLAS threads](0010c-coordinated-openblas-thread-candidates-and-shared-cpu-thread-budget.md)
- [CPU 0010D qualification](0010d-installed-openblas-qualification-and-target-fingerprinting.md)<span data-glossary-navigation="true"> ([qualification glossary](../../../../glossary.md#term-qualification))</span>
- [CPU 0010E compatible candidate decisions](0010e-float32-float64-openblas-tuning-candidates-and-compatible-decisions.md)

## Architecture constraints

- Engine remains the outer composition root. CPU supplies one concrete backend adapter but never
  depends on Engine; ordinary Engine users neither construct nor name the adapter.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [composition root glossary](../../../../glossary.md#term-composition-root); [concrete backend glossary](../../../../glossary.md#concrete-backend))</span>
- [CPU](../../../../glossary.md#term-cpu) directly consumes public `CompileArtifacts` in `preparations(CompileArtifacts)`, so
  `backends/cpu` must directly depend on `modules/compiler`. This realizes the existing execution-
  side architecture and direct-contract dependency rule; it does not reverse the lifecycle,
  expose Compiler internals, or permit CPU to own compilation.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [lifecycle glossary](../../../../glossary.md#lifecycle); [Compiler glossary](../../../../glossary.md#term-compiler); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names))</span>
- [Planning](../../../../glossary.md#term-planning) continues to see one [backend identity](../../../../glossary.md#backend-identity--backendid), `cpu`, and one host CPU availability fact.
  OpenBLAS remains a CPU-internal peer route rather than another backend or device.<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [OpenBLAS glossary](../../../../glossary.md#term-openblas); [backend glossary](../../../../glossary.md#backend); [device glossary](../../../../glossary.md#term-device))</span>
- Compiler/Planning partitions maximal consecutive same-owner runs. Therefore every non-empty
  graph whose selected owners are all CPU has exactly one non-empty CPU partition; accepting only
  that shape loses no currently executable CPU-only graph. A valid multi-partition artifact is
  necessarily mixed-owner and is outside this CPU-owned assembler's complete-schedule authority.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [graph glossary](../../../../glossary.md#graph); [partition glossary](../../../../glossary.md#partition); [shape glossary](../../../../glossary.md#shape); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Prepare owns graph preparation, resource assignment, staged analysis/finalization, and schedule
  validation. CPU owns concrete analysis inputs, lowering, executable finalization, physical
  representation recipes, and the concrete schedule assembler. Runtime owns reusable recipe
  contracts, per-run creation/binding/execution/publication, validity, rollback, and cleanup.<span data-glossary-navigation="true"> ([graph preparation glossary](../../../../glossary.md#graph-preparation--graphpreparation); [validation glossary](../../../../glossary.md#term-validation); [lowering glossary](../../../../glossary.md#lowering); [run glossary](../../../../glossary.md#term-run); [binding glossary](../../../../glossary.md#term-binding); [publication glossary](../../../../glossary.md#term-publication); [rollback glossary](../../../../glossary.md#term-rollback))</span>
- The adapter may return public [Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime) [SPI](../../../../glossary.md#term-service-provider-interface-spi) types, but no CPU `.internal` type may occur in
  its public or protected signatures, generic bounds, exceptions, or Javadoc links.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Automatic discovery is a bounded cold CPU action. It uses the existing fixed platform table,
  performs no reflection/service lookup, and never makes OpenBLAS mandatory. Portable CPU remains
  available when discovery or qualification fails or when the safe heuristic keeps portable.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup); [qualification glossary](../../../../glossary.md#term-qualification); [heuristic glossary](../../../../glossary.md#term-heuristic))</span>
- A successfully qualified OpenBLAS invocation is borrowed by finalized recipes and therefore
  must remain alive until the adapter closes. <span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease))</span>`close()` first rejects new preparation/borrowing,
  quiesces and closes the coordinator so admitted native calls finish and provider thread state is
  restored. Repeated or concurrent close is idempotent; the first cleanup failure remains primary
  and later distinct cleanup failures are suppressed. No worker group is created by this smallest
  scalar/single-thread adapter.<span data-glossary-navigation="true"> ([borrowing glossary](../../../../glossary.md#term-borrowing); [idempotent glossary](../../../../glossary.md#term-idempotent); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Prepared executions must not outlive their adapter. Closing while any run may still execute is
  a caller/Engine lifecycle error; coordinator quiescence protects admitted OpenBLAS calls but
  does not grant continued use of closed provider resources.<span data-glossary-navigation="true"> ([quiescence glossary](../../../../glossary.md#term-quiescence))</span>
- Adapter construction and reusable recipe access must be thread-safe. Preparation may run
  concurrently only through immutable/shared-safe collaborators; per-run representations remain
  isolated and Runtime retains its existing run-state concurrency rules.<span data-glossary-navigation="true"> ([thread-safe glossary](../../../../glossary.md#term-thread-safe); [immutable glossary](../../../../glossary.md#term-immutable); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>
- All discovery/qualification/setup work is cold. No hot invocation queries or mutates provider
  thread state, chooses a backend/route, reads Config, consults tuning, or performs lookup.
- Exact/default means current strict CPU numerical behavior. No relaxed candidate is admitted and
  no optional Config permission is inferred.
- Compiler constants become run-owned initialized CPU representations; bindable inputs remain
  borrowed representation-level occurrences. Ownership, initialization, failure rollback, and
  publication must use the existing Prepare/Runtime contracts rather than a parallel lifecycle.<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant); [Ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- The CPU borrow port validates only intrinsic storage facts. Expected logical type, required
  capacity/span, and required writeability belong to Engine 0003's future typed binding because
  they require both the caller storage and its target logical binding. Runtime/executable cold
  binding retains its existing representation checks; neither layer may be described as receiving
  target facts through <span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>`borrow(HostTensorStorage)`.
- If implementation requires changing a public Prepare, Runtime, Backend Contract, or Compiler
  seam; adding any dependency/build change other than the exact CPU-to-Compiler line; changing the
  authoritative architecture decision/direction; introducing global coordination; or exposing an
  internal CPU type, stop and report the concrete architecture gap.

## Package impact

Supported production package:

- `io.github.pho001.synaptik.backend.cpu` gains only
  `CpuBackendIntegration`. Existing `CpuCapabilityProvider` remains unchanged unless Javadoc must
  point to the integration SPI.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [SPI glossary](../../../../glossary.md#term-service-provider-interface-spi))</span>

[CPU](../../../../glossary.md#term-cpu)-private implementation packages:

- `io.github.pho001.synaptik.backend.cpu.internal` may gain one narrowly named composition helper
  if keeping factory/lifecycle mechanics out of the public type materially improves cohesion.
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` owns construction of exact positional
  analysis inputs and may gain the concrete schedule assembler.
- `io.github.pho001.synaptik.backend.cpu.internal.memory` owns allocation, borrowing, constant
  initialization, and any concrete CPU transfer/creator recipe required by that assembler.<span data-glossary-navigation="true"> ([borrowing glossary](../../../../glossary.md#term-borrowing); [constant glossary](../../../../glossary.md#term-constant))</span>
- Existing `.internal.executable` and `.internal.route.nativeblas.openblas` types remain the
  implementation collaborators; none becomes supported API.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

Test package:

- `io.github.pho001.synaptik.backend.cpu` proves supported [shape](../../../../glossary.md#shape) and [lifecycle](../../../../glossary.md#lifecycle) without importing
  `.internal` from the public-shape [fixture](../../../../glossary.md#term-fixture).
- Existing CPU internal and backend-conformance packages may inspect internals only for focused
  mechanism evidence.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>

No new [module](../../../../glossary.md#term-module) or exported user package is introduced. The only [dependency](../../../../glossary.md#term-dependency) realization is the
direct `backends/cpu` -> `modules/compiler` edge required by the public [compile](../../../../glossary.md#compile)-[artifact](../../../../glossary.md#term-artifact) use;
`modules/engine` remains forbidden.

## Affected files

Expected production files:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuBackendIntegration.java`
- one internal composition [owner](../../../../glossary.md#term-owner) under `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/`
- one internal [CPU](../../../../glossary.md#term-cpu) schedule assembler under
  `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/`
- at most one internal memory/transfer helper under
  `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/memory/`
- directly affected package-info or [Javadoc](../../../../glossary.md#term-javadoc) files only when needed by the documentation pass

Expected build file:

- `backends/cpu/build.gradle.kts`, adding only
  `implementation(project(":modules:compiler"))`

Expected test files:

- one supported-package public-[shape](../../../../glossary.md#shape)/[lifecycle](../../../../glossary.md#lifecycle) test
- one internal preparation/schedule test
- one focused automatic-[OpenBLAS](../../../../glossary.md#term-openblas) lifecycle/fallback test
- `testing/backend-conformance/src/test/java/io/github/pho001/synaptik/testing/conformance/CpuOpenBlasRouteConformanceTest.java`,
  only for adapter-level conformance<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance))</span>
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuInternalPackageInventoryTest.java`
  if internal inventory changes
- `testing/architecture-tests/src/test/java/io/github/pho001/synaptik/testing/architecture/CpuDependencyContractTest.java`,
  a narrow new test proving the exact CPU direct dependency set includes Compiler, continues to
  exclude Engine, and preserves every other declared CPU dependency boundary<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>

Expected documentation and [planning](../../../../glossary.md#term-planning) files finalized in the same overall change:

- `docs/api/public-api.md`, limited to supported-[SPI](../../../../glossary.md#term-service-provider-interface-spi) classification
- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`, only if the integration-adapter term or current CPU lifecycle status needs it
- `docs/architecture/dependency-rules.md`, limited to clarifying the already-authorized concrete
  CPU-to-Compiler direct use of public compile artifacts; this is not a new architecture direction<span data-glossary-navigation="true"> ([compile artifacts glossary](../../../../glossary.md#compile-artifacts))</span>
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/backends/cpu/tasks/0010f-supported-cpu-lifecycle-integration-adapter.md`
- `docs/planning/roadmap.md`

No Java, test, build, or architecture file outside these bounded areas is expected to change.
The documentation pass must record reasoned no-change conclusions for `ARCHITECTURE.md`, ADRs,
[module](../../../../glossary.md#term-module)-boundary/lifecycle pages, [Compiler](../../../../glossary.md#term-compiler)/[Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime)/[Backend](../../../../glossary.md#backend) Contract public contracts,
[Engine](../../../../glossary.md#term-engine), Config, tuning, Training, provider [API](../../../../glossary.md#term-application-programming-interface-api), other [Gradle](../../../../glossary.md#term-gradle) files, other architecture tests, and
integration tests unless a concrete contradiction is found.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most:

- four [CPU](../../../../glossary.md#term-cpu) production Java files, including exactly one supported public type;
- five CPU/[conformance](../../../../glossary.md#term-conformance) test Java files plus one focused architecture-test Java file;
- four explanatory/glossary/architecture Markdown files;
- three [planning](../../../../glossary.md#term-planning) Markdown files, including this task specification;
- one CPU [Gradle](../../../../glossary.md#term-gradle) build file; and
- eighteen paths total.

Generated source, generator [schema](../../../../glossary.md#term-schema), root/other-[module](../../../../glossary.md#term-module) Gradle, `ARCHITECTURE.md`, ADRs, architecture
changes beyond the focused explanatory [dependency](../../../../glossary.md#term-dependency)-rules update and one focused enforcement test,
shared-module Java, and [Engine](../../../../glossary.md#term-engine) Java do not count as optional headroom: any such required change
triggers the stop condition. Fewer paths are preferred; a nineteenth path requires a separately
planned follow-up.

## Acceptance criteria

- `CpuBackendIntegration` is the sole new supported [CPU](../../../../glossary.md#term-cpu) production type, is public, final, and
  `AutoCloseable`, has no public/protected constructor, and has exactly the seven public operations
  fixed in Scope: <span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope))</span>`open`, `capabilityProvider`, `availabilitySnapshot`, `preparations`,
  `scheduleAssembler`, `borrow`, and the inherited contract implementation `close` (seven method
  declarations including the static factory).
- Every public signature and generic projection uses only supported public shared types; a
  distinct-package source test uses the adapter without importing or reflecting on <span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature))</span>`.internal`.
- `backends/cpu/build.gradle.kts` adds exactly one direct
  `implementation(project(":modules:compiler"))` dependency and no other build change. CPU uses it
  only for public compile-artifact consumption and gains no Engine dependency.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [compile glossary](../../../../glossary.md#compile); [artifact glossary](../../../../glossary.md#term-artifact); [Engine glossary](../../../../glossary.md#term-engine))</span>
- One open adapter reports the stable CPU provider and exactly one `cpu/host` CPU device; the
  snapshot performs no discovery and OpenBLAS availability never changes that backend identity.<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device); [snapshot glossary](../../../../glossary.md#term-snapshot); [OpenBLAS glossary](../../../../glossary.md#term-openblas); [backend identity glossary](../../../../glossary.md#backend-identity--backendid))</span>
- `preparations(artifacts)` accepts exactly one non-empty CPU-owned maximal partition and returns
  one immutable singleton entry that hides exact CPU inputs/plan types behind<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [immutable glossary](../../../../glossary.md#term-immutable); [singleton glossary](../../../../glossary.md#term-singleton))</span>
  `PartitionPreparation<?, ?>` while remaining directly accepted by `GraphPreparation.prepare`.
  It rejects a zero-partition/pass-through artifact, any count greater than one, an empty
  partition, or any non-CPU owner before invoking backend analysis.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
- `scheduleAssembler()` independently rejects zero, multiple, empty, mixed, or non-CPU compile/
  prepared partition contexts before constructing any representation plan or schedule. For the
  sole accepted CPU partition it creates a valid deterministic schedule: representation creation
  first, exactly one execution occurrence retaining the finalized executable, publications in
  publication order as a final suffix, and exact reference identity to the prepared memory plan
  throughout.<span data-glossary-navigation="true"> ([prepared partition glossary](../../../../glossary.md#prepared-partition--preparedpartition); [occurrence glossary](../../../../glossary.md#term-occurrence); [publication glossary](../../../../glossary.md#term-publication); [reference identity glossary](../../../../glossary.md#term-reference-identity); [prepared memory plan glossary](../../../../glossary.md#prepared-memory-plan--preparedmemoryplan))</span>
- Every buffer/workspace creator is cold, immutable, thread-safe, and returns a fresh exact
  run-owned representation. Constant sources are initialized with their exact compiled scalar
  value; bindable inputs consume caller representations; publications select the intended valid
  CPU representation.<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer); [thread-safe glossary](../../../../glossary.md#term-thread-safe); [run glossary](../../../../glossary.md#term-run); [Constant glossary](../../../../glossary.md#term-constant); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- `borrow(storage)` accepts only live, currently accessible CPU-compatible representation-level
  host storage and retains it without ownership transfer. Before returning it rejects null or dead
  storage, a segment inaccessible from the current thread, inconsistent/overflowing
  capacity-to-byte geometry, disagreement between reported and exact segment byte size, an
  unsupported data type, or an incompatible observable heap carrier. Closing the wrapper remains
  a no-op for caller-owned storage.<span data-glossary-navigation="true"> ([host storage glossary](../../../../glossary.md#host-storage); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [data type glossary](../../../../glossary.md#data-type--datatype); [heap glossary](../../../../glossary.md#term-heap); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- `borrow(storage)` does not and cannot reject a value as the wrong logical type, wrong required
  capacity/size, or read-only for an expected write role, because its signature carries no target
  descriptor or access requirement. Engine 0003 owns those comparisons during typed logical input
  binding. This task adds no surrogate expectation, implicit lookup, or hard-coded write rule.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [descriptor glossary](../../../../glossary.md#term-descriptor); [binding glossary](../../../../glossary.md#term-binding))</span>
- Automatic discovery uses only the existing bounded table. Load/qualification/setup failure
  closes partial native ownership and produces a portable-capable adapter without changing the
  CPU availability snapshot or surfacing an optional-provider failure to ordinary preparation.<span data-glossary-navigation="true"> ([qualification glossary](../../../../glossary.md#term-qualification))</span>
- A successful load transfers ownership exactly once to one coordinator, qualification belongs to
  that session/target, prepared OpenBLAS executables borrow that coordinator, and the adapter
  retains it until close.
- The no-tuning exact/default profile supplies complete deterministic route, representation,
  thread, and threshold facts. It supplies no tuning decision; the existing safe heuristic either
  selects one current qualified OpenBLAS candidate or retains portable. No late Runtime fallback
  is introduced.<span data-glossary-navigation="true"> ([heuristic glossary](../../../../glossary.md#term-heuristic))</span>
- Repeated/concurrent close is idempotent, prevents new borrowing/preparation, quiesces native
  calls, restores provider thread state, closes provider ownership, and preserves
  primary/suppressed cleanup failure identity in deterministic order.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([idempotent glossary](../../../../glossary.md#term-idempotent); [borrowing glossary](../../../../glossary.md#term-borrowing))</span>
- The same prepared recipe can create and run isolated states concurrently while the adapter is
  open; per-run resource rollback/cleanup remains Runtime-owned and no physical resource is
  shared between runs except explicitly borrowed caller storage and the adapter-owned provider.<span data-glossary-navigation="true"> ([rollback glossary](../../../../glossary.md#term-rollback); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease))</span>
- Public Javadoc documents SPI status, lifecycle ordering, thread safety, ownership, nullability,
  parameters, returns, and failure modes without recommending direct ordinary-user construction.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [SPI glossary](../../../../glossary.md#term-service-provider-interface-spi); [lifecycle glossary](../../../../glossary.md#lifecycle); [nullability glossary](../../../../glossary.md#term-nullability); [parameters glossary](../../../../glossary.md#term-parameter))</span>
- The focused dependency-rules update describes this edge as a direct realization of the existing
  execution-side direction, and `CpuDependencyContractTest` permits Compiler while still rejecting
  Engine and locking the remaining exact CPU dependency set. Root <span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>`ARCHITECTURE.md` and ADRs remain
  unchanged because no architecture decision changes.
- No Config aggregate, tuning [lookup](../../../../glossary.md#term-lookup)/decision, [Runtime](../../../../glossary.md#term-runtime) [backend](../../../../glossary.md#backend) selection, reflection,
  `ServiceLoader`, global singleton, new provider API, or other architecture/dependency/build
  change appears.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Generated-code non-impact is explicit: no generator, emitter, schema, specialization,
  generated artifact identity, direct-Java oracle, decompilation, or generated-performance
  evidence changes because this task composes existing prepared recipes only.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [specialization glossary](../../../../glossary.md#term-specialization); [oracle glossary](../../../../glossary.md#term-test-oracle); [decompilation glossary](../../../../glossary.md#term-decompilation))</span>
- A separate clean documentation-focused context finalizes Javadocs, explanatory documentation,
  glossary impact, and planning evidence before the task can become Complete.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

Implementation-focused checks:

```bash
./gradlew :backends:cpu:test --tests '*CpuBackendIntegration*' --tests '*CpuPreparedScheduleAssembler*'
./gradlew :backends:cpu:test
./gradlew :testing:backend-conformance:test --tests '*CpuOpenBlasRouteConformanceTest*'
./gradlew :testing:architecture-tests:test --tests '*CpuDependencyContractTest*'
```

The focused tests must cover public [shape](../../../../glossary.md#shape), exact availability, exactly-one-[partition](../../../../glossary.md#partition) all-[CPU](../../../../glossary.md#term-cpu)
preparation, [constant](../../../../glossary.md#term-constant) initialization, caller-input [borrowing](../../../../glossary.md#term-borrowing), every intrinsic borrow rejection
listed in the acceptance criteria, and explicit acceptance of intrinsically valid storage without
an expected logical [binding](../../../../glossary.md#term-binding). They must prove that zero-partition/pass-through [artifacts](../../../../glossary.md#term-artifact) and
mixed/multi-partition artifacts are rejected before [backend](../../../../glossary.md#backend) analysis and that independently
supplied invalid schedule contexts are rejected before representation-plan or schedule assembly.
For the accepted one-partition case they must cover exact creation/execution/[publication](../../../../glossary.md#term-publication) ordering,
publication ordering, prepared-recipe reuse and concurrent isolated runs, portable fallback,
successful native-free [ownership](../../../../glossary.md#term-ownership-meanings) transfer, [qualification](../../../../glossary.md#term-qualification)/setup failure [rollback](../../../../glossary.md#term-rollback), close
ordering/idempotence, and post-close rejection. No empty- or multi-partition schedule is a success
case for this adapter.

Documentation-focused pass:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

Manual validation:

- inspect `javap -public` for `CpuBackendIntegration` and confirm the exact supported surface and
  absence of `.internal` [descriptors](../../../../glossary.md#term-descriptor)/[signatures](../../../../glossary.md#term-signature);
- compile a distinct-package Java fixture against CPU and its declared dependencies without
  reflective access;<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [fixture glossary](../../../../glossary.md#term-fixture))</span>
- inspect changed CPU production imports and Gradle files and confirm the sole new edge is the
  direct <span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle))</span>`implementation(project(":modules:compiler"))` [dependency](../../../../glossary.md#term-dependency) required by
  `CompileArtifacts`, with [Engine](../../../../glossary.md#term-engine) still absent;
- inspect schedule construction for immutable recipe-only behavior and exact
  representation/sole-execution/publication ordering, and inspect rejection instrumentation to
  confirm zero/multiple/mixed/non-CPU inputs reach neither backend analysis nor assembly;<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- inspect `borrow(HostTensorStorage)` and its tests to confirm it uses only intrinsic storage facts,
  accepts both read-only and writable storage when otherwise intrinsically valid, and contains no
  invented expected descriptor, required capacity/span, prepared-position lookup, or write-role
  policy;<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup))</span>
- inspect all failure paths for deterministic partial-resource cleanup and suppressed failures;
- [scan](../../../../glossary.md#term-cumulative-scan) for `ServiceLoader`, reflection, global registries/singletons, Config/tuning decisions, and
  Runtime backend selection;<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime))</span>
- confirm generator [schema](../../../../glossary.md#term-schema) and generated-code paths are unchanged;
- confirm the focused architecture test reads the declared CPU dependency set directly, permits
  Compiler, rejects Engine, and preserves every other expected edge;<span data-glossary-navigation="true"> ([architecture test glossary](../../../../glossary.md#term-architecture-test); [Compiler glossary](../../../../glossary.md#term-compiler))</span>
- confirm the dependency-rules wording is explanatory and consistent with `ARCHITECTURE.md`, so no
  root-contract or ADR change is needed;<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- validate Markdown links, anchors, heading order, fences, trailing whitespace, and newline at [EOF](../../../../glossary.md#term-eof);
- confirm exact changed-path scope is at most eighteen and contains no shared Java, Engine Java,
  root/other-module build file, root architecture contract, ADR, or integration-test file.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [module glossary](../../../../glossary.md#term-module); [architecture contract glossary](../../../../glossary.md#architecture-contract))</span>

Repository-wide validation is deferred to Engine's [lifecycle](../../../../glossary.md#lifecycle) [checkpoint](../../../../glossary.md#term-checkpoint) or [CI](../../../../glossary.md#term-continuous-integration-ci). The task changes
one [concrete backend](../../../../glossary.md#concrete-backend)'s supported composition seam and realizes one already-authorized direct
dependency with focused architecture enforcement; it changes no shared contract, architecture
decision/direction, generator, or cross-backend semantics.

## Dependencies

- CPU 0010E and its CPU 0010A–0010D OpenBLAS discovery, coordination, qualification, route, and
  exact/default safe-heuristic prerequisites — Complete.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas); [qualification glossary](../../../../glossary.md#term-qualification); [heuristic glossary](../../../../glossary.md#term-heuristic))</span>
- Prepare 0003, 0003A, and 0004 public analysis/finalization/orchestration/candidate handoffs —
  Complete; no new Prepare seam is required. Prepare 0003's documented zero-node requested-value
  limitation remains unchanged and is rejected at this adapter boundary.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [node glossary](../../../../glossary.md#node))</span>
- Runtime 0010 plus closure tasks 0012–0014 for representation creation, execution, publication,
  runner, cleanup, and architecture enforcement — Complete; no new Runtime seam is required.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [publication glossary](../../../../glossary.md#term-publication))</span>
- Backend Contract 0001–0004 identities, device classification, availability, and requirements —
  Complete; no new backend contract is required.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [device glossary](../../../../glossary.md#term-device))</span>
- [Compiler](../../../../glossary.md#term-compiler) 0006B3 public complete [compile](../../../../glossary.md#compile) port and current `CompileArtifacts` — Complete.
- Direct [CPU](../../../../glossary.md#term-cpu) use of `CompileArtifacts` requires and authorizes a direct `modules/compiler`
  dependency; relying on Prepare's transitive exposure is explicitly rejected.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>
- Current CPU representation allocation/borrowing, preparer/finalizer, and generated
  executable implementation — Complete internal prerequisites.<span data-glossary-navigation="true"> ([borrowing glossary](../../../../glossary.md#term-borrowing))</span>

## Follow-up tasks

- Engine 0001, now the next Draft frontier: consume this supported adapter in the advanced
  representation-level lifecycle composition without exposing it to ordinary users.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- The later Engine/Prepare composition frontier must decide how multiple backend-owned recipe
  contributions form one mixed-backend schedule; CPU 0010F neither invents that shared contract
  nor creates a premature detailed follow-up task.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Engine 0002: own deterministic standard built-in construction so ordinary users neither name nor
  construct `CpuBackendIntegration`.
- Engine 0003–0004: add typed logical input/result mapping and host materialization, including any
  broader heap-to-CPU representation transfer justified at that boundary.<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [materialization glossary](../../../../glossary.md#materialization); [heap glossary](../../../../glossary.md#term-heap))</span>
- [Engine](../../../../glossary.md#term-engine) 0005–0007 and tools/tuning 0002 remain later ordered convenience and optional tuning work.
- CPU 0010D1, 0011–0017, Config 0004–0007, and optional provider/vendor work remain independent
  blocked or Draft branches and do not gate this exact/default adapter.

## Architecture impact

Architecture-decision impact: None. [Dependency](../../../../glossary.md#term-dependency) realization: one bounded direct edge.

This task realizes the already documented concrete-[backend](../../../../glossary.md#backend)-to-[Engine](../../../../glossary.md#term-engine) integration boundary using
existing execution-side [dependency direction](../../../../glossary.md#java-platform-and-build-tools) and public [Compiler](../../../../glossary.md#term-compiler)/[Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime)/Backend Contract
seams. Because [CPU](../../../../glossary.md#term-cpu) directly names public `CompileArtifacts`, direct CPU-to-Compiler declaration is
the truthful minimal realization; Prepare must not export Compiler transitively to hide it. The
targeted dependency-rules clarification and [architecture test](../../../../glossary.md#term-architecture-test) record and enforce that existing
direction. `ARCHITECTURE.md` and ADRs remain unchanged. The CPU-owned [SPI](../../../../glossary.md#term-service-provider-interface-spi) prevents Engine from
importing `.internal`, changes no [lifecycle](../../../../glossary.md#lifecycle) [ownership](../../../../glossary.md#term-ownership-meanings), and is not an ordinary user [facade](../../../../glossary.md#term-facade). If the
design needs any further shared-contract, dependency, build, or architecture change, stop and
report the gap instead of modifying those contracts.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are working in the Synaptik repository on CPU task 0010F.

Read AGENTS.md, ARCHITECTURE.md, the lifecycle/module-boundary/dependency and
runtime-prepare-backend architecture documents, docs/planning/planning-guide.md, the CPU master
plan, and docs/planning/backends/cpu/tasks/0010f-supported-cpu-lifecycle-integration-adapter.md in
full. Inspect completed CPU 0010A-0010E and current Prepare/Runtime contracts before editing.

Implement task 0010F exactly as specified. Add only the narrow supported CpuBackendIntegration
SPI and CPU-private composition/schedule helpers required to connect current capability,
availability, preparation/finalization, representation recipes, schedule assembly, automatic
OpenBLAS qualification/lifetime, portable fallback, and the untuned safe heuristic. Do not add
Engine code, Config/tuning integration, public CPU configuration, Runtime selection, discovery
mechanisms beyond the existing bounded OpenBLAS table, generator changes, or shared-contract
changes. Add only the direct CPU-to-Compiler Gradle dependency, focused dependency-rules
clarification, and focused CPU architecture dependency test authorized by this task. Stop and
report if another dependency/build/architecture change is needed or the work cannot fit the
eighteen-path ceiling. Do not commit or push.

Accept only artifacts containing exactly one non-empty maximal CPU partition. Reject zero-
partition/pass-through and mixed/multi-partition artifacts before backend analysis, and make the
assembler independently reject invalid partition contexts before representation-plan or schedule
construction. Test the sole successful schedule's exact creation/execution/publication ordering
and reusable/concurrent run isolation; do not implement empty- or multi-partition CPU schedule
success cases.

Run the focused, CPU-module, backend-conformance, public-shape, manual, and scope validation in the
task. Then hand the diff and evidence to a separate clean documentation-focused context. That pass
must follow docs/developer-guide/documentation-rules.md, finalize affected Javadocs, public/backend
guidance, glossary impact, CPU master/task/roadmap evidence, and reuse successful Java evidence
unless it changes executable Java. Mark Complete only after all required validation passes.
```

## Local decisions

- One lifecycle-owning supported type is smaller and safer than separately publishing discovery,
  preparation, representation, and scheduling factories.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- A wildcard `PartitionPreparation<?, ?>` list is the existing callable Prepare seam that hides
  CPU plan/input types without a new shared abstraction.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Maximal same-owner partitioning means a non-empty all-CPU compiled graph has exactly one
  partition. The adapter can therefore require one non-empty CPU partition without losing any
  current CPU-only executable graph, while rejecting mixed ownership before CPU analysis or
  assembly.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [compiled graph glossary](../../../../glossary.md#compiled-graph--compiledgraphmodel); [partition glossary](../../../../glossary.md#partition); [graph glossary](../../../../glossary.md#graph); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Because `preparations(CompileArtifacts)` directly consumes a Compiler-owned public contract,
  CPU must declare <span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>`modules/compiler` directly. Making Prepare expose Compiler transitively would
  obscure the dependency actually used and is rejected.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>
- The fixed `cpu/host` availability fact describes portable CPU presence; optional OpenBLAS affects
  only CPU-private route eligibility.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas))</span>
- Automatic OpenBLAS with portable fallback is the only initial factory. Exact name/path and public
  enable/disable configuration remain excluded until a concrete Engine/Config consumer justifies
  them.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span>
- The first Engine foundation is representation-level. Typed input conversion and host output
  materialization must not be pulled forward merely to make the CPU SPI look user-facing.<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [materialization glossary](../../../../glossary.md#materialization); [SPI glossary](../../../../glossary.md#term-service-provider-interface-spi))</span>
- `HostTensorStorage` supplies intrinsic type, capacity, byte-size, segment, read-only, liveness,
  and accessibility facts, but <span data-glossary-navigation="true"> ([liveness glossary](../../../../glossary.md#term-liveness))</span>`borrow(HostTensorStorage)` supplies no other side of a comparison.
  CPU 0010F validates internal consistency and carrier compatibility only; Engine 0003 later pairs
  storage with an expected logical binding and validates type, required capacity/span, and access.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [binding glossary](../../../../glossary.md#term-binding))</span>
- The fixed untuned profile preserves scalar/single-thread portable defaults and the established
  native-checkpoint costs and one-unit/one-basis-point thresholds. Those constants are CPU
  selection policy and carry no measured-performance claim; a tuning decision is absent.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [constants glossary](../../../../glossary.md#term-constant); [carry glossary](../../../../glossary.md#term-carry))</span>

## Known limitations

- The supported adapter covers exactly one non-empty maximal CPU partition, which is the shape of
  every current non-empty all-CPU compiled graph. It rejects every mixed/multi-partition artifact;
  mixed-backend schedule composition remains a later Engine/Prepare boundary question.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [partition glossary](../../../../glossary.md#partition); [shape glossary](../../../../glossary.md#shape); [compiled graph glossary](../../../../glossary.md#compiled-graph--compiledgraphmodel); [artifact glossary](../../../../glossary.md#term-artifact); [backend glossary](../../../../glossary.md#backend); [Engine glossary](../../../../glossary.md#term-engine); [Prepare glossary](../../../../glossary.md#prepare))</span>
- Zero-node/pass-through compilation remains valid, but requested publication cannot currently be
  prepared because no backend analysis declares its byte geometry and no<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [publication glossary](../../../../glossary.md#term-publication))</span>
  [`PreparedBufferAssignment`](../../../../glossary.md#prepared-buffer-assignment--preparedbufferassignment) exists. CPU rejects that artifact up front; resolving the existing
  Prepare limitation is outside this task.
- Only the existing narrow positive rank-two FLOAT32/FLOAT64 bare-MATMUL subset may select
  OpenBLAS. All other work remains portable.<span data-glossary-navigation="true"> ([rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [OpenBLAS glossary](../../../../glossary.md#term-openblas))</span>
- Bounded automatic discovery has no public diagnostic/configuration [facade](../../../../glossary.md#term-facade) in this task.
- Representation-level caller inputs must already satisfy the documented CPU buffer contract;
  arbitrary logical host conversion and materialized output access remain deferred. Passing<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer); [host glossary](../../../../glossary.md#term-host))</span>
  `borrow(...)` proves no expected logical type, required capacity/span, or writeability match;
  Engine 0003 must establish those facts during typed binding.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>
- Adapter closure is an outer-lifecycle barrier, not permission to continue using previously
  prepared recipes afterward.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- [Planning](../../../../glossary.md#term-planning)-time implementation discovery, not completion evidence: clean implementation context
  `01a09f3d-0012-72b3-ba71-38e2f5f6c279` attempted the mandatory
  `CpuBackendIntegration.preparations(CompileArtifacts)` surface. CPU compilation could not
  resolve the Compiler-owned <span data-glossary-navigation="true"> ([compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names))</span>`CompileArtifacts` import/type because `backends/cpu` had no direct
  `modules/compiler` dependency. The agent obeyed the prior stop condition and removed all Java
  changes, leaving only these planning paths. Coordinator architecture review confirmed that root<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>
  `ARCHITECTURE.md` forbids [CPU](../../../../glossary.md#term-cpu)-to-[Engine](../../../../glossary.md#term-engine) but not CPU-to-[Compiler](../../../../glossary.md#term-compiler), while
  `docs/architecture/dependency-rules.md` requires modules to depend directly on contracts they
  use. This evidence authorizes the single direct dependency, explanatory synchronization, and
  focused architecture enforcement above; it does not mark any implementation criterion complete.<span data-glossary-navigation="true"> ([modules glossary](../../../../glossary.md#term-module); [synchronization glossary](../../../../glossary.md#term-synchronization))</span>
- The same context's resumed implementation and successful partial validation established two
  further planning facts before all Java changes were removed. First, a requested zero-node/
  pass-through artifact has no backend analysis and therefore no <span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [artifact glossary](../../../../glossary.md#term-artifact); [backend glossary](../../../../glossary.md#backend))</span>[`PreparedBufferAssignment`](../../../../glossary.md#prepared-buffer-assignment--preparedbufferassignment);
  current [`GraphPreparation`](../../../../glossary.md#graph-preparation--graphpreparation) fails closed with
  `requested value has no prepared buffer assignment`. Second, Compiler/Planning's maximal
  same-owner contract makes every non-empty all-CPU artifact exactly one partition, whereas a
  valid multi-partition artifact necessarily contains another owner that this adapter must reject.
  The partial validation is planning-time discovery only: no implementation diff remains and no
  CPU 0010F acceptance criterion is complete.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [partition glossary](../../../../glossary.md#partition))</span>
- Implementation-owned validation ran
  `./gradlew :backends:cpu:test --tests '*CpuBackendIntegration*' --tests
  '*CpuPreparedScheduleAssembler*'`; the final strengthened report records 3 tests with zero
  failures, errors, or skips, including exact executable/order/memory-plan assertions and
  coordinated two-thread close. The earlier `./gradlew :backends:cpu:test` run passed 998 tests
  with zero failures in approximately 2m41s before the final test-only strengthening. It was not
  repeated because production code did not change; the later focused run replaced the local full-
  suite XML report, so the full count is reused implementation evidence rather than a claim from
  the remaining focused report.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
- Implementation-owned
  `./gradlew :testing:backend-conformance:test --tests '*CpuOpenBlasRouteConformanceTest*'`
  passed 10 tests with zero failures, errors, or skips. Implementation-owned
  `./gradlew :testing:architecture-tests:test --tests '*CpuDependencyContractTest*'` passed 1 test
  with zero failures, errors, or skips. The documentation pass independently inspected both
  remaining XML reports and the final focused CPU XML report.
- Manual implementation evidence used `javap -public` to confirm exactly seven public method
  declarations and compiled a distinct-package fixture. Documentation context<span data-glossary-navigation="true"> ([fixture glossary](../../../../glossary.md#term-fixture))</span>
  `01a09f80-d7a3-7930-aee3-aad2482c9f5d` repeated those read-only checks successfully after final
  Javadocs.
- Documentation context `01a09f80-d7a3-7930-aee3-aad2482c9f5d` ran
  `./gradlew :backends:cpu:javadoc`; it succeeded after final comments with 94 pre-existing
  warnings confined to unchanged CPU lowering/preparation types. It inspected the generated<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering))</span>
  `CpuBackendIntegration.html` page, ran a targeted six-file Ruby local-link/anchor/fence check,
  compared comment-stripped before/after copies of all four production files, verified the exact
  17-path allowlist and Engine 0001 Draft/no-spec state, checked EOF/trailing whitespace, and ran<span data-glossary-navigation="true"> ([EOF glossary](../../../../glossary.md#term-eof))</span>
  `git diff --check`; every check passed.
- After that documentation pass, coordinator review found one cold fatal-failure rollback gap.
  The implementation context updated the existing <span data-glossary-navigation="true"> ([rollback glossary](../../../../glossary.md#term-rollback))</span>`CpuBackendComposition` path so
  `open(session)` catches an `Error` after partial coordinator ownership transfer, invokes the
  same deterministic rollback helper, and rethrows the identical <span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>`Error`; the existing
  `RuntimeException` path still rolls back and returns portable. It strengthened the existing
  `CpuBackendCompositionTest` path to prove primary `Error` identity, restoration failure
  suppression on that primary, owner-close failure suppression beneath the restoration failure,
  and exactly-once owner cleanup. Its focused selection of <span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>`CpuBackendCompositionTest` and
  `CpuBackendIntegrationAndCpuPreparedScheduleAssemblerPublicTest` passed 6 tests with zero
  failures or errors; the retained XML reports record 3 passing tests in each class. The
  implementation context also reran `git diff --check` successfully. No full CPU rerun was made:
  the correction is confined to this bounded cold failure path, with the earlier 998-test run and
  the new focused regression evidence retained.<span data-glossary-navigation="true"> ([regression glossary](../../../../glossary.md#term-regression))</span>
- Resumed documentation review in the same clean context
  `01a09f80-d7a3-7930-aee3-aad2482c9f5d` inspected the exact correction and both focused reports.
  Existing `CpuBackendComposition` Javadoc already states that fatal setup or cleanup errors
  propagate and that the rollback helper attaches distinct cleanup failures as suppressed
  exceptions, so no Javadoc comment changed and the prior successful CPU Javadoc remains valid.
  Explanatory documents, glossary, architecture contract, ADRs, master plan, and roadmap require
  no semantic change because the fix restores their already documented failure-lifecycle
  behavior without changing a signature, dependency, owner, route, or frontier.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [architecture contract glossary](../../../../glossary.md#architecture-contract); [lifecycle glossary](../../../../glossary.md#lifecycle); [signature glossary](../../../../glossary.md#term-signature))</span>

## Implementation notes

- Added `CpuBackendIntegration` as the sole new supported CPU production type. Its seven public
  declarations expose capability, fixed host availability, exact-one-partition preparation,
  schedule assembly, intrinsic host-storage borrowing, and lifecycle ownership without exposing<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [host glossary](../../../../glossary.md#term-host); [partition glossary](../../../../glossary.md#partition); [borrowing glossary](../../../../glossary.md#term-borrowing); [lifecycle glossary](../../../../glossary.md#lifecycle); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
  `.internal` types.
- Added CPU-private composition, representation-recipe, and schedule-assembly owners. The
  composition uses a capacity-one budget, bounded automatic OpenBLAS discovery and qualification,
  the established exact/default safe-heuristic constants, no selected tuning decision, and
  portable fallback. The assembler emits one creation prefix, one execution occurrence, and the
  forward-then-gradient publication suffix without performing physical work.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas); [heuristic glossary](../../../../glossary.md#term-heuristic); [constants glossary](../../../../glossary.md#term-constant); [selected tuning decision glossary](../../../../glossary.md#selected-tuning-decision); [occurrence glossary](../../../../glossary.md#term-occurrence); [gradient glossary](../../../../glossary.md#term-gradient); [publication glossary](../../../../glossary.md#term-publication))</span>
- Added the direct CPU-to-Compiler Gradle dependency and focused architecture enforcement. CPU
  remains independent of Engine.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [Engine glossary](../../../../glossary.md#term-engine))</span>
- The post-implementation test review strengthened the public test's executable-identity and
  schedule-order assertions and changed close coverage to two coordinated threads. Production
  Java did not change afterward, so the successful 998-test CPU suite was not repeated.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- A later cold-path correction made fatal [qualification](../../../../glossary.md#term-qualification) `Error` rollback match the documented
  ownership contract: partial coordinator/session ownership is cleaned deterministically and the
  identical fatal error is rethrown. Focused regression coverage proves the complete nested
  suppression tree and exactly-once owner cleanup without changing public signatures or hot
  execution.<span data-glossary-navigation="true"> ([rollback glossary](../../../../glossary.md#term-rollback); [regression glossary](../../../../glossary.md#term-regression); [owner glossary](../../../../glossary.md#term-owner); [signatures glossary](../../../../glossary.md#term-signature))</span>
- Documentation-focused context `01a09f80-d7a3-7930-aee3-aad2482c9f5d` independently checked the
  source, tests, current diff, architecture and shared contracts; finalized all four new
  production Javadocs plus dependency, public-API, CPU-guide, and planning text; and changed no
  executable Java token.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning); [token glossary](../../../../glossary.md#term-token))</span>

## Completion summary

- Completed changes: supported CPU lifecycle SPI; CPU-private composition, representation, and
  schedule recipes; direct Compiler dependency enforcement; focused lifecycle/conformance tests;
  and final documentation/Javadoc.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [lifecycle glossary](../../../../glossary.md#lifecycle); [SPI glossary](../../../../glossary.md#term-service-provider-interface-spi); [Compiler glossary](../../../../glossary.md#term-compiler); [dependency glossary](../../../../glossary.md#term-dependency); [conformance glossary](../../../../glossary.md#term-conformance); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Files changed or created: four CPU production Java files, four CPU test files, one
  backend-conformance test, one architecture test, one CPU build file, three explanatory docs,
  and three planning docs (seventeen paths total).<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [architecture test glossary](../../../../glossary.md#term-architecture-test); [planning glossary](../../../../glossary.md#term-planning))</span>
- Tests and validation: the final post-correction focused CPU selection passed 6 tests with zero
  failures or errors, comprising 3 <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>`CpuBackendCompositionTest` tests and 3 public CPU integration
  tests with zero skips; the reused implementation-owned full CPU run passed 998 tests with zero
  failures in approximately 2m41s before the final focused strengthening and cold rollback
  correction; the affected
  backend-conformance report records 10 tests with zero failures/errors/skips; and the focused CPU
  dependency report records 1 test with zero failures/errors/skips. CPU Javadoc completed after
  final comments with only 94 pre-existing unrelated warnings. Targeted Markdown local-link,
  anchor, and fence validation passed for all six changed Markdown files; comment-stripped
  before/after comparison proved the documentation pass changed no Java token; exact 17-path
  scope, public-surface/import/forbidden-mechanism/status checks, generated-page inspection, EOF/
  trailing-whitespace checks, and <span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [rollback glossary](../../../../glossary.md#term-rollback); [token glossary](../../../../glossary.md#term-token); [scope glossary](../../../../glossary.md#term-scope); [EOF glossary](../../../../glossary.md#term-eof))</span>`git diff --check` passed.
- Documentation-agent review: clean context `01a09f80-d7a3-7930-aee3-aad2482c9f5d`; General,
  API/Javadoc, Architecture, Backend-guide, Developer-guide, Example, and Planning profiles were
  applied to the affected document types.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Documentation impact: dependency rules now explain the already-authorized direct CPU-to-
  Compiler realization; public API and CPU backend guidance classify the adapter as Engine-facing
  SPI and document its exact accepted domain, lifecycle, ownership, fallback, and deferred Engine
  boundary.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Javadoc review: every type, constructor, and method in the four new production files documents
  its meaningful contract, inputs, result where applicable, ownership, nullability, lifecycle,
  concurrency, and expected failures. The rollback correction requires no Javadoc edit because<span data-glossary-navigation="true"> ([nullability glossary](../../../../glossary.md#term-nullability); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>
  `CpuBackendComposition.open(session)` already documents fatal setup/cleanup `Error` propagation
  and `closeAfterFailure` already documents deterministic suppression on the primary failure; the
  prior successful Javadoc result therefore remains current.
- Glossary impact: no change. Existing entries already define lifecycle, partition, host storage,
  Prepare/Runtime recipes, CPU route, OpenBLAS discovery, qualification, and coordinator concepts;
  the adapter is a narrow composition of those terms rather than a new reusable domain concept.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [host storage glossary](../../../../glossary.md#host-storage); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [OpenBLAS glossary](../../../../glossary.md#term-openblas); [qualification glossary](../../../../glossary.md#term-qualification))</span>
- Architecture impact: none. `ARCHITECTURE.md`, ADRs, other architecture explanations, and other
  architecture tests remain accurate; only the focused dependency explanation and enforcement
  changed.
- Generated-code impact: none. No generator, emitter, schema, operation semantics, generated
  artifact, direct-Java oracle, bytecode equivalence, or performance evidence changed or was newly
  validated.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [operation glossary](../../../../glossary.md#operation); [artifact glossary](../../../../glossary.md#term-artifact); [oracle glossary](../../../../glossary.md#term-test-oracle); [bytecode glossary](../../../../glossary.md#term-bytecode))</span>
- Other API/test impact: no Compiler, Prepare, Runtime, Backend Contract, Config, Engine, Training,
  provider, or other shared Java contract changed. No build file other than CPU, test outside the
  bounded CPU/conformance/architecture set, or integration test required a change.<span data-glossary-navigation="true"> ([integration test glossary](../../../../glossary.md#term-integration-test))</span>
- Unresolved issues: None.
- Follow-up required: Engine 0001 remains the next Draft frontier; typed logical binding and the
  simpler end-user execution/result surfaces remain later Engine work. No Engine task
  specification was created.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>

Status: Complete
