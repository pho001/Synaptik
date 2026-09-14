# Task 0010F: Supported CPU Lifecycle Integration Adapter

## Status

Complete

## Goal

Publish the smallest CPU-owned, public-for-module-integration lifecycle adapter that future
Engine code can consume without importing `io.github.pho001.synaptik.backend.cpu.internal`.
The adapter composes the existing CPU capability provider, host availability, partition analysis
and finalization, physical representation recipes, deterministic schedule assembly, portable
execution, and bounded automatic OpenBLAS discovery behind one explicitly owned lifetime.

The type is cross-module SPI, not an ordinary user facade. Engine later owns both its advanced
explicit composition path and the standard built-in factory through which ordinary users obtain
CPU execution without constructing or naming this adapter.

## Scope

- Add one public final `AutoCloseable` integration type,
  `io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration`, in the existing supported CPU
  package. Give it a private constructor and one public static `open()` factory for the fixed
  exact/default built-in composition.
- Expose only these public instance collaborations, using existing public shared types in every
  signature:
  - `capabilityProvider()` returns the retained `CpuCapabilityProvider`;
  - `availabilitySnapshot()` returns one immutable CPU snapshot containing the stable device
    `BackendDeviceId(CpuCapabilityProvider.CPU_BACKEND_ID, "host")` classified as
    `DeviceClass.CPU`;
  - `preparations(CompileArtifacts)` accepts only a compile artifact containing exactly one
    non-empty maximal partition owned by CPU and returns an immutable singleton positional
    `List<PartitionPreparation<?, ?>>`, with exact CPU-private analysis inputs, one retained
    preparer, and one retained finalizer hidden behind shared Prepare roles;
  - `scheduleAssembler()` returns the retained `PreparedScheduleAssembler` that builds the CPU
    representation, execution, constant-initialization, and publication recipe described below;
  - `borrow(HostTensorStorage)` returns a `BufferRepresentation` for one intrinsically valid
    representation-level caller storage without exposing `CpuBorrowedBuffer`. Before returning,
    it validates only facts supplied by that storage: non-nullity, current scope liveness and
    current-thread segment accessibility, checked equality among element capacity, data-type byte
    width, reported byte size, and exact segment byte size, and supported data-type/observable
    heap-carrier compatibility. It receives no prepared binding expectation; and
  - `close()` owns deterministic composition cleanup.
- Add the direct `implementation(project(":modules:compiler"))` dependency required because the
  supported `preparations(CompileArtifacts)` signature and implementation directly consume
  Compiler-owned artifacts. Do not obtain Compiler accidentally through Prepare or another
  transitive dependency.
- Clarify `docs/architecture/dependency-rules.md` narrowly: concrete CPU now realizes the already
  permitted direct Compiler edge because it consumes public `CompileArtifacts`; the execution-side
  direction, all other dependency rules, root `ARCHITECTURE.md`, and ADRs remain unchanged.
- Add one focused `CpuDependencyContractTest` under `testing/architecture-tests` that locks CPU's
  exact direct dependency set with Compiler permitted and Engine forbidden.
- Keep every existing CPU preparation plan, analysis input, executable, representation,
  OpenBLAS discovery/qualification/coordinator, worker, and schedule-assembly implementation type
  under `.internal`.
- At `open()`, construct one capacity-one CPU concurrency budget; attempt the existing bounded
  automatic OpenBLAS discovery;
  and, when loading succeeds, transfer the provider owner into one coordinator, qualify that exact
  installed target, and retain it for the adapter lifetime. Discovery, load, qualification, or
  coordinator setup failure closes partial resources and leaves a fully usable portable adapter.
- Use one CPU-private immutable exact/default untuned preparation profile. Preserve current
  scalar, single-thread portable defaults and direct-only materialization. It has no selected
  tuning decision and admits only the already-supported FLOAT32/FLOAT64 OpenBLAS route with the
  single proved thread-count candidate `1`. Reuse the established native-checkpoint selection
  constants exactly: portable costs `(fixed=1_000, perOutput=10, perMac=100)`, OpenBLAS costs
  `(fixed=1, perOutput=1, perMac=1)`, zero representation-transition costs, minimum absolute
  benefit `1`, and minimum relative benefit `1` basis point. These are deterministic safe-
  heuristic policy inputs, not benchmark evidence or a performance claim.
- Validate the complete compile partition list before constructing analysis inputs: reject a
  zero-partition/pass-through artifact, more than one partition, an empty partition, or any
  non-CPU owner before backend analysis. For the sole accepted partition, derive carrier,
  native-storage, execution, and route facts from the exact compile artifacts; reject incomplete
  positional coverage, unresolved physical geometry, a closed adapter, or an unsupported
  representation before analysis/finalization can escape the adapter.
- Make the retained assembler independently reject any schedule context that does not contain
  exactly one non-empty CPU-owned compile/prepared partition before constructing a representation
  plan or schedule. For the accepted context, assemble one immutable schedule with this order:
  one representation-creation prefix; the sole partition execution step; then publication steps
  in compile publication order. Create fresh run-owned aligned CPU buffers and workspaces from the
  exact prepared memory entries, create and initialize compiler constant sources once per run,
  mark bindable inputs as caller-input occurrences, and select the exact CPU representation used
  by the finalized executable and publications. Perform no allocation, copy, execution, or
  creator invocation while assembling.
- Keep representation creators and the assembled schedule immutable and safe for concurrent
  reuse. Each run receives distinct owned buffers/workspaces; borrowed input ownership stays with
  the caller and Runtime closes only run-owned representations through its existing rules.
- Add focused CPU tests for the public shape, portable lifecycle, schedule recipe, concurrent
  reuse, cleanup/failure rollback, and native-free automatic-discovery fallback. Extend the
  existing CPU/OpenBLAS backend-conformance coverage only where it proves the supported adapter
  crosses the real Prepare and Runtime recipe boundary.
- Require a separate clean documentation-focused pass after Java implementation to finalize all
  affected Javadocs, supported-API/backend guidance, glossary impact, and planning evidence.

## Out of scope

- Any Engine production type or task specification, including Engine 0001 or later.
- An ordinary-user CPU builder, CPU option/configuration object, provider-name/path override,
  enable/disable switch, public cost/thread knobs, or CPU type in ordinary Engine signatures.
- Config aggregates (`CompileConfig`, `PrepareConfig`, or `RunOptions`), Config 0006 relaxed
  permissions, or Config 0006A mapping.
- Tuning measurement, candidate execution, selection, decision/cache lookup, persistence,
  workload extraction, or tools/tuning integration. The adapter supplies no
  `CpuOpenBlasTuningDecision`.
- Runtime backend registration, selection, discovery, provider lookup, or route fallback.
- Reflection, `ServiceLoader`, classpath/annotation scanning, service location, mutable global
  registry, or process-global singleton ownership.
- New OpenBLAS symbols, provider API changes, name/path selection, BFLOAT16 OpenBLAS, optional
  vendor peers, relaxed numerics, or cross-route CPU tuning.
- Typed logical Engine input mapping, arbitrary heap-carrier adaptation, host-result
  materialization, one-shot execution, or user-visible result access; those remain Engine
  0003–0005 work.
- Successful preparation or schedule assembly for zero-partition/pass-through artifacts. This is
  the existing Prepare 0003 limitation: requested values have no backend-declared buffer geometry
  or `PreparedBufferAssignment`, so `GraphPreparation` intentionally fails closed rather than
  inventing allocation policy.
- Successful CPU-only assembly of a multi-partition artifact or mixed-backend schedule
  composition. Compiler/Planning already makes every non-empty all-CPU graph one maximal CPU
  partition; any valid multi-partition artifact therefore contains another owner. The later
  Engine/Prepare composition boundary must decide how multiple backend assemblers contribute one
  complete schedule.
- Comparing borrowed storage with an expected logical tensor data type, required element/byte
  capacity, layout span, or read/write access role. `borrow(HostTensorStorage)` receives none of
  those target facts. Engine 0003 must perform that typed logical binding validation before
  Runtime execution; CPU 0010F must not guess or manufacture the missing expectation.
- New CPU semantics, lowering, generated kernel forms, generator schema, artifact identity,
  generated bytecode, generated/direct oracle, or performance claims.
- Shared Prepare, Runtime, Backend Contract, Compiler, Config, Planning, Model, Trace, or provider
  Java changes. The only build/dependency change is the exact direct CPU-to-Compiler dependency;
  the only architecture-document/test changes are the focused dependency-rules clarification and
  CPU dependency-contract enforcement specified below.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially lifecycle, Engine, Prepare,
  Runtime, concrete-backend, resource-ownership, and dependency rules.
- [`docs/architecture/lifecycle.md`](../../../../architecture/lifecycle.md)
- [`docs/architecture/module-boundaries.md`](../../../../architecture/module-boundaries.md)
- [`docs/architecture/dependency-rules.md`](../../../../architecture/dependency-rules.md)
- [`docs/architecture/runtime-prepare-backend-boundary.md`](../../../../architecture/runtime-prepare-backend-boundary.md)
- [`docs/planning/planning-guide.md`](../../../planning-guide.md)
- [`docs/planning/backends/cpu/master-plan.md`](../master-plan.md)
- [`docs/planning/modules/prepare/master-plan.md`](../../../modules/prepare/master-plan.md)
- [`docs/planning/modules/runtime/master-plan.md`](../../../modules/runtime/master-plan.md)
- [`docs/planning/modules/engine/master-plan.md`](../../../modules/engine/master-plan.md)
- [CPU 0010A automatic OpenBLAS discovery](0010a-automatic-openblas-discovery-and-internal-composition-foundation.md)
- [CPU 0010C coordinated OpenBLAS threads](0010c-coordinated-openblas-thread-candidates-and-shared-cpu-thread-budget.md)
- [CPU 0010D qualification](0010d-installed-openblas-qualification-and-target-fingerprinting.md)
- [CPU 0010E compatible candidate decisions](0010e-float32-float64-openblas-tuning-candidates-and-compatible-decisions.md)

## Architecture constraints

- Engine remains the outer composition root. CPU supplies one concrete backend adapter but never
  depends on Engine; ordinary Engine users neither construct nor name the adapter.
- CPU directly consumes public `CompileArtifacts` in `preparations(CompileArtifacts)`, so
  `backends/cpu` must directly depend on `modules/compiler`. This realizes the existing execution-
  side architecture and direct-contract dependency rule; it does not reverse the lifecycle,
  expose Compiler internals, or permit CPU to own compilation.
- Planning continues to see one backend identity, `cpu`, and one host CPU availability fact.
  OpenBLAS remains a CPU-internal peer route rather than another backend or device.
- Compiler/Planning partitions maximal consecutive same-owner runs. Therefore every non-empty
  graph whose selected owners are all CPU has exactly one non-empty CPU partition; accepting only
  that shape loses no currently executable CPU-only graph. A valid multi-partition artifact is
  necessarily mixed-owner and is outside this CPU-owned assembler's complete-schedule authority.
- Prepare owns graph preparation, resource assignment, staged analysis/finalization, and schedule
  validation. CPU owns concrete analysis inputs, lowering, executable finalization, physical
  representation recipes, and the concrete schedule assembler. Runtime owns reusable recipe
  contracts, per-run creation/binding/execution/publication, validity, rollback, and cleanup.
- The adapter may return public Prepare/Runtime SPI types, but no CPU `.internal` type may occur in
  its public or protected signatures, generic bounds, exceptions, or Javadoc links.
- Automatic discovery is a bounded cold CPU action. It uses the existing fixed platform table,
  performs no reflection/service lookup, and never makes OpenBLAS mandatory. Portable CPU remains
  available when discovery or qualification fails or when the safe heuristic keeps portable.
- A successfully qualified OpenBLAS invocation is borrowed by finalized recipes and therefore
  must remain alive until the adapter closes. `close()` first rejects new preparation/borrowing,
  quiesces and closes the coordinator so admitted native calls finish and provider thread state is
  restored. Repeated or concurrent close is idempotent; the first cleanup failure remains primary
  and later distinct cleanup failures are suppressed. No worker group is created by this smallest
  scalar/single-thread adapter.
- Prepared executions must not outlive their adapter. Closing while any run may still execute is
  a caller/Engine lifecycle error; coordinator quiescence protects admitted OpenBLAS calls but
  does not grant continued use of closed provider resources.
- Adapter construction and reusable recipe access must be thread-safe. Preparation may run
  concurrently only through immutable/shared-safe collaborators; per-run representations remain
  isolated and Runtime retains its existing run-state concurrency rules.
- All discovery/qualification/setup work is cold. No hot invocation queries or mutates provider
  thread state, chooses a backend/route, reads Config, consults tuning, or performs lookup.
- Exact/default means current strict CPU numerical behavior. No relaxed candidate is admitted and
  no optional Config permission is inferred.
- Compiler constants become run-owned initialized CPU representations; bindable inputs remain
  borrowed representation-level occurrences. Ownership, initialization, failure rollback, and
  publication must use the existing Prepare/Runtime contracts rather than a parallel lifecycle.
- The CPU borrow port validates only intrinsic storage facts. Expected logical type, required
  capacity/span, and required writeability belong to Engine 0003's future typed binding because
  they require both the caller storage and its target logical binding. Runtime/executable cold
  binding retains its existing representation checks; neither layer may be described as receiving
  target facts through `borrow(HostTensorStorage)`.
- If implementation requires changing a public Prepare, Runtime, Backend Contract, or Compiler
  seam; adding any dependency/build change other than the exact CPU-to-Compiler line; changing the
  authoritative architecture decision/direction; introducing global coordination; or exposing an
  internal CPU type, stop and report the concrete architecture gap.

## Package impact

Supported production package:

- `io.github.pho001.synaptik.backend.cpu` gains only
  `CpuBackendIntegration`. Existing `CpuCapabilityProvider` remains unchanged unless Javadoc must
  point to the integration SPI.

CPU-private implementation packages:

- `io.github.pho001.synaptik.backend.cpu.internal` may gain one narrowly named composition helper
  if keeping factory/lifecycle mechanics out of the public type materially improves cohesion.
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` owns construction of exact positional
  analysis inputs and may gain the concrete schedule assembler.
- `io.github.pho001.synaptik.backend.cpu.internal.memory` owns allocation, borrowing, constant
  initialization, and any concrete CPU transfer/creator recipe required by that assembler.
- Existing `.internal.executable` and `.internal.route.nativeblas.openblas` types remain the
  implementation collaborators; none becomes supported API.

Test package:

- `io.github.pho001.synaptik.backend.cpu` proves supported shape and lifecycle without importing
  `.internal` from the public-shape fixture.
- Existing CPU internal and backend-conformance packages may inspect internals only for focused
  mechanism evidence.

No new module or exported user package is introduced. The only dependency realization is the
direct `backends/cpu` -> `modules/compiler` edge required by the public compile-artifact use;
`modules/engine` remains forbidden.

## Affected files

Expected production files:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuBackendIntegration.java`
- one internal composition owner under `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/`
- one internal CPU schedule assembler under
  `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/`
- at most one internal memory/transfer helper under
  `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/memory/`
- directly affected package-info or Javadoc files only when needed by the documentation pass

Expected build file:

- `backends/cpu/build.gradle.kts`, adding only
  `implementation(project(":modules:compiler"))`

Expected test files:

- one supported-package public-shape/lifecycle test
- one internal preparation/schedule test
- one focused automatic-OpenBLAS lifecycle/fallback test
- `testing/backend-conformance/src/test/java/io/github/pho001/synaptik/testing/conformance/CpuOpenBlasRouteConformanceTest.java`,
  only for adapter-level conformance
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuInternalPackageInventoryTest.java`
  if internal inventory changes
- `testing/architecture-tests/src/test/java/io/github/pho001/synaptik/testing/architecture/CpuDependencyContractTest.java`,
  a narrow new test proving the exact CPU direct dependency set includes Compiler, continues to
  exclude Engine, and preserves every other declared CPU dependency boundary

Expected documentation and planning files finalized in the same overall change:

- `docs/api/public-api.md`, limited to supported-SPI classification
- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`, only if the integration-adapter term or current CPU lifecycle status needs it
- `docs/architecture/dependency-rules.md`, limited to clarifying the already-authorized concrete
  CPU-to-Compiler direct use of public compile artifacts; this is not a new architecture direction
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/backends/cpu/tasks/0010f-supported-cpu-lifecycle-integration-adapter.md`
- `docs/planning/roadmap.md`

No Java, test, build, or architecture file outside these bounded areas is expected to change.
The documentation pass must record reasoned no-change conclusions for `ARCHITECTURE.md`, ADRs,
module-boundary/lifecycle pages, Compiler/Prepare/Runtime/Backend Contract public contracts,
Engine, Config, tuning, Training, provider API, other Gradle files, other architecture tests, and
integration tests unless a concrete contradiction is found.

## Maximum scope

This task may create or modify at most:

- four CPU production Java files, including exactly one supported public type;
- five CPU/conformance test Java files plus one focused architecture-test Java file;
- four explanatory/glossary/architecture Markdown files;
- three planning Markdown files, including this task specification;
- one CPU Gradle build file; and
- eighteen paths total.

Generated source, generator schema, root/other-module Gradle, `ARCHITECTURE.md`, ADRs, architecture
changes beyond the focused explanatory dependency-rules update and one focused enforcement test,
shared-module Java, and Engine Java do not count as optional headroom: any such required change
triggers the stop condition. Fewer paths are preferred; a nineteenth path requires a separately
planned follow-up.

## Acceptance criteria

- `CpuBackendIntegration` is the sole new supported CPU production type, is public, final, and
  `AutoCloseable`, has no public/protected constructor, and has exactly the seven public operations
  fixed in Scope: `open`, `capabilityProvider`, `availabilitySnapshot`, `preparations`,
  `scheduleAssembler`, `borrow`, and the inherited contract implementation `close` (seven method
  declarations including the static factory).
- Every public signature and generic projection uses only supported public shared types; a
  distinct-package source test uses the adapter without importing or reflecting on `.internal`.
- `backends/cpu/build.gradle.kts` adds exactly one direct
  `implementation(project(":modules:compiler"))` dependency and no other build change. CPU uses it
  only for public compile-artifact consumption and gains no Engine dependency.
- One open adapter reports the stable CPU provider and exactly one `cpu/host` CPU device; the
  snapshot performs no discovery and OpenBLAS availability never changes that backend identity.
- `preparations(artifacts)` accepts exactly one non-empty CPU-owned maximal partition and returns
  one immutable singleton entry that hides exact CPU inputs/plan types behind
  `PartitionPreparation<?, ?>` while remaining directly accepted by `GraphPreparation.prepare`.
  It rejects a zero-partition/pass-through artifact, any count greater than one, an empty
  partition, or any non-CPU owner before invoking backend analysis.
- `scheduleAssembler()` independently rejects zero, multiple, empty, mixed, or non-CPU compile/
  prepared partition contexts before constructing any representation plan or schedule. For the
  sole accepted CPU partition it creates a valid deterministic schedule: representation creation
  first, exactly one execution occurrence retaining the finalized executable, publications in
  publication order as a final suffix, and exact reference identity to the prepared memory plan
  throughout.
- Every buffer/workspace creator is cold, immutable, thread-safe, and returns a fresh exact
  run-owned representation. Constant sources are initialized with their exact compiled scalar
  value; bindable inputs consume caller representations; publications select the intended valid
  CPU representation.
- `borrow(storage)` accepts only live, currently accessible CPU-compatible representation-level
  host storage and retains it without ownership transfer. Before returning it rejects null or dead
  storage, a segment inaccessible from the current thread, inconsistent/overflowing
  capacity-to-byte geometry, disagreement between reported and exact segment byte size, an
  unsupported data type, or an incompatible observable heap carrier. Closing the wrapper remains
  a no-op for caller-owned storage.
- `borrow(storage)` does not and cannot reject a value as the wrong logical type, wrong required
  capacity/size, or read-only for an expected write role, because its signature carries no target
  descriptor or access requirement. Engine 0003 owns those comparisons during typed logical input
  binding. This task adds no surrogate expectation, implicit lookup, or hard-coded write rule.
- Automatic discovery uses only the existing bounded table. Load/qualification/setup failure
  closes partial native ownership and produces a portable-capable adapter without changing the
  CPU availability snapshot or surfacing an optional-provider failure to ordinary preparation.
- A successful load transfers ownership exactly once to one coordinator, qualification belongs to
  that session/target, prepared OpenBLAS executables borrow that coordinator, and the adapter
  retains it until close.
- The no-tuning exact/default profile supplies complete deterministic route, representation,
  thread, and threshold facts. It supplies no tuning decision; the existing safe heuristic either
  selects one current qualified OpenBLAS candidate or retains portable. No late Runtime fallback
  is introduced.
- Repeated/concurrent close is idempotent, prevents new borrowing/preparation, quiesces native
  calls, restores provider thread state, closes provider ownership, and preserves
  primary/suppressed cleanup failure identity in deterministic order.
- The same prepared recipe can create and run isolated states concurrently while the adapter is
  open; per-run resource rollback/cleanup remains Runtime-owned and no physical resource is
  shared between runs except explicitly borrowed caller storage and the adapter-owned provider.
- Public Javadoc documents SPI status, lifecycle ordering, thread safety, ownership, nullability,
  parameters, returns, and failure modes without recommending direct ordinary-user construction.
- The focused dependency-rules update describes this edge as a direct realization of the existing
  execution-side direction, and `CpuDependencyContractTest` permits Compiler while still rejecting
  Engine and locking the remaining exact CPU dependency set. Root `ARCHITECTURE.md` and ADRs remain
  unchanged because no architecture decision changes.
- No Config aggregate, tuning lookup/decision, Runtime backend selection, reflection,
  `ServiceLoader`, global singleton, new provider API, or other architecture/dependency/build
  change appears.
- Generated-code non-impact is explicit: no generator, emitter, schema, specialization,
  generated artifact identity, direct-Java oracle, decompilation, or generated-performance
  evidence changes because this task composes existing prepared recipes only.
- A separate clean documentation-focused context finalizes Javadocs, explanatory documentation,
  glossary impact, and planning evidence before the task can become Complete.

## Tests / validation

Implementation-focused checks:

```bash
./gradlew :backends:cpu:test --tests '*CpuBackendIntegration*' --tests '*CpuPreparedScheduleAssembler*'
./gradlew :backends:cpu:test
./gradlew :testing:backend-conformance:test --tests '*CpuOpenBlasRouteConformanceTest*'
./gradlew :testing:architecture-tests:test --tests '*CpuDependencyContractTest*'
```

The focused tests must cover public shape, exact availability, exactly-one-partition all-CPU
preparation, constant initialization, caller-input borrowing, every intrinsic borrow rejection
listed in the acceptance criteria, and explicit acceptance of intrinsically valid storage without
an expected logical binding. They must prove that zero-partition/pass-through artifacts and
mixed/multi-partition artifacts are rejected before backend analysis and that independently
supplied invalid schedule contexts are rejected before representation-plan or schedule assembly.
For the accepted one-partition case they must cover exact creation/execution/publication ordering,
publication ordering, prepared-recipe reuse and concurrent isolated runs, portable fallback,
successful native-free ownership transfer, qualification/setup failure rollback, close
ordering/idempotence, and post-close rejection. No empty- or multi-partition schedule is a success
case for this adapter.

Documentation-focused pass:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

Manual validation:

- inspect `javap -public` for `CpuBackendIntegration` and confirm the exact supported surface and
  absence of `.internal` descriptors/signatures;
- compile a distinct-package Java fixture against CPU and its declared dependencies without
  reflective access;
- inspect changed CPU production imports and Gradle files and confirm the sole new edge is the
  direct `implementation(project(":modules:compiler"))` dependency required by
  `CompileArtifacts`, with Engine still absent;
- inspect schedule construction for immutable recipe-only behavior and exact
  representation/sole-execution/publication ordering, and inspect rejection instrumentation to
  confirm zero/multiple/mixed/non-CPU inputs reach neither backend analysis nor assembly;
- inspect `borrow(HostTensorStorage)` and its tests to confirm it uses only intrinsic storage facts,
  accepts both read-only and writable storage when otherwise intrinsically valid, and contains no
  invented expected descriptor, required capacity/span, prepared-position lookup, or write-role
  policy;
- inspect all failure paths for deterministic partial-resource cleanup and suppressed failures;
- scan for `ServiceLoader`, reflection, global registries/singletons, Config/tuning decisions, and
  Runtime backend selection;
- confirm generator schema and generated-code paths are unchanged;
- confirm the focused architecture test reads the declared CPU dependency set directly, permits
  Compiler, rejects Engine, and preserves every other expected edge;
- confirm the dependency-rules wording is explanatory and consistent with `ARCHITECTURE.md`, so no
  root-contract or ADR change is needed;
- validate Markdown links, anchors, heading order, fences, trailing whitespace, and newline at EOF;
- confirm exact changed-path scope is at most eighteen and contains no shared Java, Engine Java,
  root/other-module build file, root architecture contract, ADR, or integration-test file.

Repository-wide validation is deferred to Engine's lifecycle checkpoint or CI. The task changes
one concrete backend's supported composition seam and realizes one already-authorized direct
dependency with focused architecture enforcement; it changes no shared contract, architecture
decision/direction, generator, or cross-backend semantics.

## Dependencies

- CPU 0010E and its CPU 0010A–0010D OpenBLAS discovery, coordination, qualification, route, and
  exact/default safe-heuristic prerequisites — Complete.
- Prepare 0003, 0003A, and 0004 public analysis/finalization/orchestration/candidate handoffs —
  Complete; no new Prepare seam is required. Prepare 0003's documented zero-node requested-value
  limitation remains unchanged and is rejected at this adapter boundary.
- Runtime 0010 plus closure tasks 0012–0014 for representation creation, execution, publication,
  runner, cleanup, and architecture enforcement — Complete; no new Runtime seam is required.
- Backend Contract 0001–0004 identities, device classification, availability, and requirements —
  Complete; no new backend contract is required.
- Compiler 0006B3 public complete compile port and current `CompileArtifacts` — Complete.
- Direct CPU use of `CompileArtifacts` requires and authorizes a direct `modules/compiler`
  dependency; relying on Prepare's transitive exposure is explicitly rejected.
- Current CPU representation allocation/borrowing, preparer/finalizer, and generated
  executable implementation — Complete internal prerequisites.

## Follow-up tasks

- Engine 0001, now the next Draft frontier: consume this supported adapter in the advanced
  representation-level lifecycle composition without exposing it to ordinary users.
- The later Engine/Prepare composition frontier must decide how multiple backend-owned recipe
  contributions form one mixed-backend schedule; CPU 0010F neither invents that shared contract
  nor creates a premature detailed follow-up task.
- Engine 0002: own deterministic standard built-in construction so ordinary users neither name nor
  construct `CpuBackendIntegration`.
- Engine 0003–0004: add typed logical input/result mapping and host materialization, including any
  broader heap-to-CPU representation transfer justified at that boundary.
- Engine 0005–0007 and tools/tuning 0002 remain later ordered convenience and optional tuning work.
- CPU 0010D1, 0011–0017, Config 0004–0007, and optional provider/vendor work remain independent
  blocked or Draft branches and do not gate this exact/default adapter.

## Architecture impact

Architecture-decision impact: None. Dependency realization: one bounded direct edge.

This task realizes the already documented concrete-backend-to-Engine integration boundary using
existing execution-side dependency direction and public Compiler/Prepare/Runtime/Backend Contract
seams. Because CPU directly names public `CompileArtifacts`, direct CPU-to-Compiler declaration is
the truthful minimal realization; Prepare must not export Compiler transitively to hide it. The
targeted dependency-rules clarification and architecture test record and enforce that existing
direction. `ARCHITECTURE.md` and ADRs remain unchanged. The CPU-owned SPI prevents Engine from
importing `.internal`, changes no lifecycle ownership, and is not an ordinary user facade. If the
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
  preparation, representation, and scheduling factories.
- A wildcard `PartitionPreparation<?, ?>` list is the existing callable Prepare seam that hides
  CPU plan/input types without a new shared abstraction.
- Maximal same-owner partitioning means a non-empty all-CPU compiled graph has exactly one
  partition. The adapter can therefore require one non-empty CPU partition without losing any
  current CPU-only executable graph, while rejecting mixed ownership before CPU analysis or
  assembly.
- Because `preparations(CompileArtifacts)` directly consumes a Compiler-owned public contract,
  CPU must declare `modules/compiler` directly. Making Prepare expose Compiler transitively would
  obscure the dependency actually used and is rejected.
- The fixed `cpu/host` availability fact describes portable CPU presence; optional OpenBLAS affects
  only CPU-private route eligibility.
- Automatic OpenBLAS with portable fallback is the only initial factory. Exact name/path and public
  enable/disable configuration remain excluded until a concrete Engine/Config consumer justifies
  them.
- The first Engine foundation is representation-level. Typed input conversion and host output
  materialization must not be pulled forward merely to make the CPU SPI look user-facing.
- `HostTensorStorage` supplies intrinsic type, capacity, byte-size, segment, read-only, liveness,
  and accessibility facts, but `borrow(HostTensorStorage)` supplies no other side of a comparison.
  CPU 0010F validates internal consistency and carrier compatibility only; Engine 0003 later pairs
  storage with an expected logical binding and validates type, required capacity/span, and access.
- The fixed untuned profile preserves scalar/single-thread portable defaults and the established
  native-checkpoint costs and one-unit/one-basis-point thresholds. Those constants are CPU
  selection policy and carry no measured-performance claim; a tuning decision is absent.

## Known limitations

- The supported adapter covers exactly one non-empty maximal CPU partition, which is the shape of
  every current non-empty all-CPU compiled graph. It rejects every mixed/multi-partition artifact;
  mixed-backend schedule composition remains a later Engine/Prepare boundary question.
- Zero-node/pass-through compilation remains valid, but requested publication cannot currently be
  prepared because no backend analysis declares its byte geometry and no
  `PreparedBufferAssignment` exists. CPU rejects that artifact up front; resolving the existing
  Prepare limitation is outside this task.
- Only the existing narrow positive rank-two FLOAT32/FLOAT64 bare-MATMUL subset may select
  OpenBLAS. All other work remains portable.
- Bounded automatic discovery has no public diagnostic/configuration facade in this task.
- Representation-level caller inputs must already satisfy the documented CPU buffer contract;
  arbitrary logical host conversion and materialized output access remain deferred. Passing
  `borrow(...)` proves no expected logical type, required capacity/span, or writeability match;
  Engine 0003 must establish those facts during typed binding.
- Adapter closure is an outer-lifecycle barrier, not permission to continue using previously
  prepared recipes afterward.

## Validation evidence

- Planning-time implementation discovery, not completion evidence: clean implementation context
  `01a09f3d-0012-72b3-ba71-38e2f5f6c279` attempted the mandatory
  `CpuBackendIntegration.preparations(CompileArtifacts)` surface. CPU compilation could not
  resolve the Compiler-owned `CompileArtifacts` import/type because `backends/cpu` had no direct
  `modules/compiler` dependency. The agent obeyed the prior stop condition and removed all Java
  changes, leaving only these planning paths. Coordinator architecture review confirmed that root
  `ARCHITECTURE.md` forbids CPU-to-Engine but not CPU-to-Compiler, while
  `docs/architecture/dependency-rules.md` requires modules to depend directly on contracts they
  use. This evidence authorizes the single direct dependency, explanatory synchronization, and
  focused architecture enforcement above; it does not mark any implementation criterion complete.
- The same context's resumed implementation and successful partial validation established two
  further planning facts before all Java changes were removed. First, a requested zero-node/
  pass-through artifact has no backend analysis and therefore no `PreparedBufferAssignment`;
  current `GraphPreparation` fails closed with
  `requested value has no prepared buffer assignment`. Second, Compiler/Planning's maximal
  same-owner contract makes every non-empty all-CPU artifact exactly one partition, whereas a
  valid multi-partition artifact necessarily contains another owner that this adapter must reject.
  The partial validation is planning-time discovery only: no implementation diff remains and no
  CPU 0010F acceptance criterion is complete.
- Implementation-owned validation ran
  `./gradlew :backends:cpu:test --tests '*CpuBackendIntegration*' --tests
  '*CpuPreparedScheduleAssembler*'`; the final strengthened report records 3 tests with zero
  failures, errors, or skips, including exact executable/order/memory-plan assertions and
  coordinated two-thread close. The earlier `./gradlew :backends:cpu:test` run passed 998 tests
  with zero failures in approximately 2m41s before the final test-only strengthening. It was not
  repeated because production code did not change; the later focused run replaced the local full-
  suite XML report, so the full count is reused implementation evidence rather than a claim from
  the remaining focused report.
- Implementation-owned
  `./gradlew :testing:backend-conformance:test --tests '*CpuOpenBlasRouteConformanceTest*'`
  passed 10 tests with zero failures, errors, or skips. Implementation-owned
  `./gradlew :testing:architecture-tests:test --tests '*CpuDependencyContractTest*'` passed 1 test
  with zero failures, errors, or skips. The documentation pass independently inspected both
  remaining XML reports and the final focused CPU XML report.
- Manual implementation evidence used `javap -public` to confirm exactly seven public method
  declarations and compiled a distinct-package fixture. Documentation context
  `01a09f80-d7a3-7930-aee3-aad2482c9f5d` repeated those read-only checks successfully after final
  Javadocs.
- Documentation context `01a09f80-d7a3-7930-aee3-aad2482c9f5d` ran
  `./gradlew :backends:cpu:javadoc`; it succeeded after final comments with 94 pre-existing
  warnings confined to unchanged CPU lowering/preparation types. It inspected the generated
  `CpuBackendIntegration.html` page, ran a targeted six-file Ruby local-link/anchor/fence check,
  compared comment-stripped before/after copies of all four production files, verified the exact
  17-path allowlist and Engine 0001 Draft/no-spec state, checked EOF/trailing whitespace, and ran
  `git diff --check`; every check passed.
- After that documentation pass, coordinator review found one cold fatal-failure rollback gap.
  The implementation context updated the existing `CpuBackendComposition` path so
  `open(session)` catches an `Error` after partial coordinator ownership transfer, invokes the
  same deterministic rollback helper, and rethrows the identical `Error`; the existing
  `RuntimeException` path still rolls back and returns portable. It strengthened the existing
  `CpuBackendCompositionTest` path to prove primary `Error` identity, restoration failure
  suppression on that primary, owner-close failure suppression beneath the restoration failure,
  and exactly-once owner cleanup. Its focused selection of `CpuBackendCompositionTest` and
  `CpuBackendIntegrationAndCpuPreparedScheduleAssemblerPublicTest` passed 6 tests with zero
  failures or errors; the retained XML reports record 3 passing tests in each class. The
  implementation context also reran `git diff --check` successfully. No full CPU rerun was made:
  the correction is confined to this bounded cold failure path, with the earlier 998-test run and
  the new focused regression evidence retained.
- Resumed documentation review in the same clean context
  `01a09f80-d7a3-7930-aee3-aad2482c9f5d` inspected the exact correction and both focused reports.
  Existing `CpuBackendComposition` Javadoc already states that fatal setup or cleanup errors
  propagate and that the rollback helper attaches distinct cleanup failures as suppressed
  exceptions, so no Javadoc comment changed and the prior successful CPU Javadoc remains valid.
  Explanatory documents, glossary, architecture contract, ADRs, master plan, and roadmap require
  no semantic change because the fix restores their already documented failure-lifecycle
  behavior without changing a signature, dependency, owner, route, or frontier.

## Implementation notes

- Added `CpuBackendIntegration` as the sole new supported CPU production type. Its seven public
  declarations expose capability, fixed host availability, exact-one-partition preparation,
  schedule assembly, intrinsic host-storage borrowing, and lifecycle ownership without exposing
  `.internal` types.
- Added CPU-private composition, representation-recipe, and schedule-assembly owners. The
  composition uses a capacity-one budget, bounded automatic OpenBLAS discovery and qualification,
  the established exact/default safe-heuristic constants, no selected tuning decision, and
  portable fallback. The assembler emits one creation prefix, one execution occurrence, and the
  forward-then-gradient publication suffix without performing physical work.
- Added the direct CPU-to-Compiler Gradle dependency and focused architecture enforcement. CPU
  remains independent of Engine.
- The post-implementation test review strengthened the public test's executable-identity and
  schedule-order assertions and changed close coverage to two coordinated threads. Production
  Java did not change afterward, so the successful 998-test CPU suite was not repeated.
- A later cold-path correction made fatal qualification `Error` rollback match the documented
  ownership contract: partial coordinator/session ownership is cleaned deterministically and the
  identical fatal error is rethrown. Focused regression coverage proves the complete nested
  suppression tree and exactly-once owner cleanup without changing public signatures or hot
  execution.
- Documentation-focused context `01a09f80-d7a3-7930-aee3-aad2482c9f5d` independently checked the
  source, tests, current diff, architecture and shared contracts; finalized all four new
  production Javadocs plus dependency, public-API, CPU-guide, and planning text; and changed no
  executable Java token.

## Completion summary

- Completed changes: supported CPU lifecycle SPI; CPU-private composition, representation, and
  schedule recipes; direct Compiler dependency enforcement; focused lifecycle/conformance tests;
  and final documentation/Javadoc.
- Files changed or created: four CPU production Java files, four CPU test files, one
  backend-conformance test, one architecture test, one CPU build file, three explanatory docs,
  and three planning docs (seventeen paths total).
- Tests and validation: the final post-correction focused CPU selection passed 6 tests with zero
  failures or errors, comprising 3 `CpuBackendCompositionTest` tests and 3 public CPU integration
  tests with zero skips; the reused implementation-owned full CPU run passed 998 tests with zero
  failures in approximately 2m41s before the final focused strengthening and cold rollback
  correction; the affected
  backend-conformance report records 10 tests with zero failures/errors/skips; and the focused CPU
  dependency report records 1 test with zero failures/errors/skips. CPU Javadoc completed after
  final comments with only 94 pre-existing unrelated warnings. Targeted Markdown local-link,
  anchor, and fence validation passed for all six changed Markdown files; comment-stripped
  before/after comparison proved the documentation pass changed no Java token; exact 17-path
  scope, public-surface/import/forbidden-mechanism/status checks, generated-page inspection, EOF/
  trailing-whitespace checks, and `git diff --check` passed.
- Documentation-agent review: clean context `01a09f80-d7a3-7930-aee3-aad2482c9f5d`; General,
  API/Javadoc, Architecture, Backend-guide, Developer-guide, Example, and Planning profiles were
  applied to the affected document types.
- Documentation impact: dependency rules now explain the already-authorized direct CPU-to-
  Compiler realization; public API and CPU backend guidance classify the adapter as Engine-facing
  SPI and document its exact accepted domain, lifecycle, ownership, fallback, and deferred Engine
  boundary.
- Javadoc review: every type, constructor, and method in the four new production files documents
  its meaningful contract, inputs, result where applicable, ownership, nullability, lifecycle,
  concurrency, and expected failures. The rollback correction requires no Javadoc edit because
  `CpuBackendComposition.open(session)` already documents fatal setup/cleanup `Error` propagation
  and `closeAfterFailure` already documents deterministic suppression on the primary failure; the
  prior successful Javadoc result therefore remains current.
- Glossary impact: no change. Existing entries already define lifecycle, partition, host storage,
  Prepare/Runtime recipes, CPU route, OpenBLAS discovery, qualification, and coordinator concepts;
  the adapter is a narrow composition of those terms rather than a new reusable domain concept.
- Architecture impact: none. `ARCHITECTURE.md`, ADRs, other architecture explanations, and other
  architecture tests remain accurate; only the focused dependency explanation and enforcement
  changed.
- Generated-code impact: none. No generator, emitter, schema, operation semantics, generated
  artifact, direct-Java oracle, bytecode equivalence, or performance evidence changed or was newly
  validated.
- Other API/test impact: no Compiler, Prepare, Runtime, Backend Contract, Config, Engine, Training,
  provider, or other shared Java contract changed. No build file other than CPU, test outside the
  bounded CPU/conformance/architecture set, or integration test required a change.
- Unresolved issues: None.
- Follow-up required: Engine 0001 remains the next Draft frontier; typed logical binding and the
  simpler end-user execution/result surfaces remain later Engine work. No Engine task
  specification was created.

Status: Complete
