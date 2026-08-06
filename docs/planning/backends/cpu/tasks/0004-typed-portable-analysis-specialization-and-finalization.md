# Task 0004: Typed Portable Analysis, Specialization, and Finalization

## Status

Superseded

## Goal

Connect the completed CPU representation, generated-kernel, and durable-artifact foundations to
the existing staged Prepare and Runtime contracts without claiming a real Model operation:

```text
already CPU-owned PrepareContext
  -> CPU-private typed candidate source
  -> exact hard filtering and deterministic portable candidate selection
  -> opaque CpuPortablePreparationPlan + exact shared requirements
  -> shared Prepare assigns slots
  -> CPU BackendPartitionFinalizer
  -> CPU-0003 cold artifact load/generation
  -> immutable CpuPortablePreparedExecutable retaining CpuGeneratedKernel + direct MethodHandle
  -> per-run CpuBorrowedBuffer/CpuNativeBuffer representations
  -> one checked cold bind
  -> family-owned typed BoundInvocation with direct fields
  -> Runtime hot call
```

This task delivers the CPU-private lifecycle foundation used by later portable operation-family
tasks. It proves the complete analysis-to-bound-invocation path only with bounded synthetic
candidates and emitters. `CpuCapabilityProvider.supports(...)` remains unconditionally false;
CPU 0005 and later tasks must add truthful family computation and capability coverage before any
Model operation is advertised.

## Scope

- Add one package-private immutable CPU implementation of `BackendAnalysisInputs` carrying only
  the exact supported Vector species snapshot and prepared parallel configuration supplied by a
  later CPU composition owner.
- Add one package-private immutable prepared parallel configuration with positive worker count,
  positive minimum range size, and explicit deterministic/non-deterministic range intent.
- Add one package-private direct typed candidate-source collaboration. It receives the complete
  `PrepareContext<CpuPortableAnalysisInputs>` and returns candidates in deterministic preference
  order. The task supplies no production operation-family implementation; tests supply bounded
  synthetic sources only.
- Add one immutable complete portable candidate that binds together exactly one
  `CpuKernelSpecialization`, its matching `CpuFamilyKernelEmitter`, its ordered exact buffer and
  workspace declarations, its ordered buffer/workspace uses, and one family-owned typed
  invocation binder.
- Treat each `CpuKernelSpecialization.Argument` as the exact selected typed portable
  representation and access fact for its aligned buffer use. The carrier is selected before
  shared slot assignment and is never rediscovered from storage during execution.
- Analyze only a partition whose `PlannedPartition.owner()` equals
  `CpuCapabilityProvider.CPU_BACKEND_ID`; reject another owner before candidate generation.
- Validate every candidate against the exact `PrepareContext`: all resource values are projected,
  buffer uses align one-for-one with specialization arguments, every use refers by exact
  reference to a declared requirement, every declaration is used, data type/access/carrier facts
  agree, and no declaration or use introduces an unprojected value or undeclared workspace.
- Filter Vector candidates against the exact supported `VectorShape` snapshot and filter parallel
  candidates against the prepared worker configuration. Select the first valid candidate in the
  source's deterministic preference order. Fail closed when the source is empty, contains an
  invalid result, or has no eligible candidate.
- Return `BackendPartitionAnalysis<CpuPortablePreparationPlan>` retaining the exact context
  partition, selected immutable plan, and the selected candidate's exact ordered declarations.
- Add one CPU-private `BackendPartitionFinalizer<CpuPortablePreparationPlan>` with exact `cpu`
  identity. Its constructor receives an explicit artifact root and an already-owned
  `CpuWorkerGroup`; it invents no default root, Config API, environment lookup, home-directory
  convention, worker owner, registry, or service locator.
- During finalization, consume only the existing validated
  `BackendPartitionFinalization`/`PreparationResourceAssignment` handoff. Resolve every buffer and
  workspace use to its assigned dense plan position before any artifact-store operation.
- Consult `CpuGeneratedKernelArtifactStore` only after all assignments, worker configuration, and
  selected-plan facts validate. A compatible hit or verified miss follows the completed CPU 0003
  contract unchanged.
- Construct an immutable `CpuPortablePreparedExecutable` that strongly retains the exact
  `CpuGeneratedKernel`, the exact direct `MethodHandle` returned by that artifact, the exact
  specialization, prepared parallel configuration, borrowed worker-group reference, selected
  resource recipes, and family-owned typed invocation binder.
- Extend the existing `CpuPreparedExecutable` cold boundary. Validate the actual
  `CpuBufferArgument` variant against the selected specialization carrier, validate dynamic array
  offsets and exact segment form without copying, and prove every segment used by a parallel mode
  is accessible to every retained worker before invoking the binder.
- Require the family-owned binder to copy the exact handle, direct array/segment carrier fields,
  primitive offsets/extents/range configuration, and direct workspace fields needed by its
  signature into a typed `BoundInvocation`. It must not retain the fresh cold-binding arrays.
- For the bounded synthetic proof, use signature-specific test-only `BoundInvocation` subclasses
  whose `executeBound()` calls the exact handle without `invokeWithArguments`, spreaders,
  collectors, reflection, `Object[]`, or argument-kind switches.
- Exercise all four existing `CpuPortableExecutionMode` values, heap, exact-segment, and mixed
  signatures, single-thread and worker-dispatched parallel paths, shared assignment, artifact
  reuse, repeated/concurrent binds, failure order, and direct hot calls with synthetic work only.
- Preserve `CpuBorrowedBuffer` as the completed CPU 0001 non-owning
  `HostTensorStorage` representation/lifetime boundary. Production portable cold binding in this
  task consumes it through Runtime's `BufferRepresentation`/`RunState` path; no executable or
  binder accepts `HostTensorStorage` directly.
- Finalize affected Javadocs, CPU guidance, glossary impact, and planning evidence through a
  separate clean documentation-focused handoff after Java stabilizes.

## Out of scope

- any truthful Model operation-family computation, numerical algorithm, Tensor result,
  operation-kind lowering, fusion, capability advertisement, backend-conformance claim, or
  integration claim;
- changing `CpuCapabilityProvider`, including returning `true` for a synthetic, metadata-only,
  zero-element, view-like, or otherwise bounded test occurrence;
- a public CPU backend, public facade, public preparer/finalizer, public route/candidate/config
  API, registry, service locator, generic manager, reflective discovery, or default artifact-root
  Config API without an existing composition owner;
- replacing `CpuBorrowedBuffer` with direct `HostTensorStorage`-to-`CpuBufferArgument` binding,
  bypassing `BufferRepresentation`, changing borrowed lifetime, or reopening CPU 0001 storage;
- changing the CPU 0001 buffer/workspace allocation, representation, ownership, close, or worker
  lifecycle contracts;
- changing the CPU 0002 schema, specialization equality/fingerprint, four-mode vocabulary,
  generated entry shape, emitter ownership, Class-File API mechanism, or Vector build flags;
- changing the CPU 0003 artifact format, key, root trust boundary, publication, validation,
  single-flight, weak interning, corruption, or lifetime policy;
- a tuning cache, compatible tuning-decision input, measurements, benchmarks, autotuning,
  broad cost model, transition-cost model, performance threshold, vendor priority, or mutation of
  any persistent evidence;
- OpenBLAS, oneMKL, oneDNN, Accelerate, AOCL, ZenDNN, native vendor, Metal, or CUDA routes;
- operation-family kernels assigned to CPU 0005–0008, the CPU 0009 capability checkpoint, or any
  CPU 0010–0016 behavior or detailed specification;
- FLOAT16 or mixed-precision semantics/capability; future Model 0026 remains the prerequisite for
  any FLOAT16 claim;
- a new schedule assembler, public composition owner, PreparedExecution owner/close lifecycle,
  prepared-resource serialization, worker pooling, allocation pooling, materialization policy,
  transfer policy, publication behavior, or output access;
- changes to Model, Config, Planning, Prepare, Runtime, Backend Contract, Trace, Compiler, Engine,
  OpenBLAS provider, other backends, extensions, tools, Gradle, dependencies, architecture,
  focused architecture documentation, ADRs, architecture tests, backend conformance, or
  integration tests; and
- commits or pushes.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
  - Core invariants
  - `modules/runtime`
  - `modules/prepare`
  - Concrete backend modules
  - CPU backend routes
  - Prepare lifecycle
  - Run lifecycle
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Runtime, Prepare, and Backend Boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [ADR 0010: Staged backend preparation](../../../../design/decisions/0010-staged-backend-preparation.md)
- [ADR 0011: Per-run Runtime resource ownership and cold binding](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)
- [CPU backend guide](../../../../backend-guide/cpu-backend.md)
- [Task 0001 CPU capability, representation, binding, and parallel foundation](0001-cpu-capability-representation-binding-and-parallel-foundation.md)
- [Task 0002 Portable Class-File API generator foundation](0002-portable-class-file-api-generator-foundation.md)
- [Task 0003 Durable generated-kernel artifact store and cold loading](0003-bounded-generated-artifact-cache-and-cold-finalization.md)

## Architecture constraints

- Planning has already selected CPU ownership. CPU analysis may reject that partition or select
  its private portable route; it must not reconsider backend ownership.
- `PrepareContext` is the complete concrete-backend-facing semantic/planning projection. CPU must
  not accept or retain `CompileArtifacts`, Compiler aggregates, a Tensor, or a second graph model.
- Analysis is deterministic from the exact context, explicit CPU inputs, and the direct candidate
  source. It performs no filesystem access, artifact lookup, class generation/definition,
  allocation, measurement, cache mutation, executable construction, or Runtime work.
- The selected `CpuPortablePreparationPlan` is immutable opaque backend state. It retains the
  selected lowering/emitter, specialization, resource-use mapping, binder, and parallel recipe;
  it is not a generated artifact, executable, physical resource, slot assignment, or per-run
  object.
- Exact buffer/workspace requirements are returned before shared assignment. Finalization must
  use the same declarations and cannot change the selected mode, carrier, specialization,
  parallel configuration, route, or resource set.
- CPU 0003 artifact lookup/generation/definition occurs only in backend finalization after the
  complete assignment mapping succeeds. Runtime performs none of that work.
- Finalization constructs immutable Java recipe state only. It borrows the externally owned
  worker group and acquires no closeable resource. The generated artifact has no close lifecycle
  and is retained strongly by the prepared executable.
- `PreparedExecutable.bind(RunState)` remains the only heterogeneous compatibility boundary.
  `CpuBorrowedBuffer` and `CpuNativeBuffer` remain Runtime-visible nominal representations until
  the CPU cold boundary classifies them to exact `CpuBufferArgument` values.
- A family-owned typed binder is required because the generated entry `MethodType` varies with
  exact carriers, dynamic offsets/extents, and execution mode. One generic `Object[]` or adapted
  handle invoker would violate the direct hot-path contract.
- The bound invocation retains the exact `RunState` through `BoundInvocation`, performs only its
  final state-open guard, then invokes direct typed fields. Runtime execution performs no route,
  cache, disk, hash, checksum, class-file, lookup, reflection, storage discovery, heap-base
  discovery, argument classification, graph inspection, or slot lookup.
- No public/shared contract, dependency direction, module boundary, Gradle rule, or architecture
  decision changes. Stop if implementation needs one.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu` — unchanged fail-closed public capability identity.
- `io.github.pho001.synaptik.backend.cpu.execution` — extended with the package-private typed
  analysis/finalization/recipe foundation beside the completed representation, worker,
  specialization, generator, and artifact machinery.

Packages added or changed:

- No Java package is added. All production types remain package-private in `execution`; this lets
  them collaborate with `CpuBorrowedBuffer`, `CpuBufferArgument`, `CpuWorkerGroup`,
  `CpuKernelSpecialization`, and `CpuGeneratedKernelArtifactStore` without widening completed
  internals or creating a facade.

Type placement:

- `CpuPortableAnalysisInputs` — exact immutable CPU target/parallel inputs implementing
  `BackendAnalysisInputs`.
- `CpuPreparedParallelConfiguration` — selected primitive parallel recipe, not a worker owner.
- `CpuPortableCandidateSource` — one direct typed candidate-generation collaboration; no
  registry or discovery.
- `CpuPortableKernelCandidate` — complete typed candidate, declarations, use mapping, emitter,
  specialization, and binder.
- `CpuPortableInvocationBinder` — cold family-owned construction seam for a signature-specific
  direct `BoundInvocation`.
- `CpuPortablePreparationPlan` — immutable selected opaque plan implementing
  `BackendPreparationPlan`.
- `CpuPortablePartitionPreparer` — small analysis/filter/selection owner implementing the current
  `BackendPartitionPreparer` contract.
- `CpuPortablePartitionFinalizer` — small assignment/artifact/executable owner implementing the
  current `BackendPartitionFinalizer` contract.
- `CpuPortablePreparedExecutable` — immutable CPU Runtime recipe and cold specialization check.

## Exact package-private surface

The task must implement the following package-private source-level surface. Explicit compact
constructors/accessors may be added only to validate, snapshot, or document these exact members.
A signature or package change requires updating this task before implementation continues.

```java
record CpuPreparedParallelConfiguration(
        int workerCount,
        long minimumRangeSize,
        boolean deterministic) {}

record CpuPortableAnalysisInputs(
        List<CpuKernelSpecialization.VectorShape> supportedVectorShapes,
        CpuPreparedParallelConfiguration parallelConfiguration)
        implements BackendAnalysisInputs {}

@FunctionalInterface
interface CpuPortableCandidateSource {
    List<CpuPortableKernelCandidate> candidates(
            PrepareContext<CpuPortableAnalysisInputs> context);
}

final class CpuPortableKernelCandidate {
    record BufferUse(
            PreparationResourceRequirement.Buffer requirement,
            int representationIndex) {}

    record WorkspaceUse(
            PreparationResourceRequirement.Workspace requirement) {}

    CpuPortableKernelCandidate(
            CpuKernelSpecialization specialization,
            CpuFamilyKernelEmitter familyEmitter,
            List<PreparationResourceRequirement> requirements,
            List<BufferUse> bufferUses,
            List<WorkspaceUse> workspaceUses,
            CpuPortableInvocationBinder invocationBinder);

    CpuKernelSpecialization specialization();
    CpuFamilyKernelEmitter familyEmitter();
    List<PreparationResourceRequirement> requirements();
    List<BufferUse> bufferUses();
    List<WorkspaceUse> workspaceUses();
    CpuPortableInvocationBinder invocationBinder();
}

@FunctionalInterface
interface CpuPortableInvocationBinder {
    BoundInvocation bind(
            RunState runState,
            MethodHandle entryPoint,
            CpuKernelSpecialization specialization,
            CpuPreparedParallelConfiguration parallelConfiguration,
            CpuWorkerGroup workerGroup,
            CpuBufferArgument[] bufferArguments,
            CpuNativeWorkspace[] workspaces);
}

final class CpuPortablePreparationPlan implements BackendPreparationPlan {
    CpuPortablePreparationPlan(
            CpuPortableKernelCandidate candidate,
            CpuPreparedParallelConfiguration parallelConfiguration);
    CpuPortableKernelCandidate candidate();
    CpuPreparedParallelConfiguration parallelConfiguration();
}

final class CpuPortablePartitionPreparer implements BackendPartitionPreparer<
        CpuPortableAnalysisInputs, CpuPortablePreparationPlan> {
    CpuPortablePartitionPreparer(CpuPortableCandidateSource candidateSource);
    @Override BackendPartitionAnalysis<CpuPortablePreparationPlan> analyze(
            PrepareContext<CpuPortableAnalysisInputs> context);
}

final class CpuPortablePartitionFinalizer
        implements BackendPartitionFinalizer<CpuPortablePreparationPlan> {
    CpuPortablePartitionFinalizer(Path artifactRoot, CpuWorkerGroup workerGroup);
    @Override BackendId backendId();
    @Override PreparedExecutable finalizePartition(
            BackendPartitionFinalization<CpuPortablePreparationPlan> finalization);
}

final class CpuPortablePreparedExecutable extends CpuPreparedExecutable {
    CpuPortablePreparedExecutable(
            PreparedMemoryPlan memoryPlan,
            List<BufferSelection> bufferSelections,
            List<WorkspaceSelection> workspaceSelections,
            List<BufferAccess> bufferAccesses,
            List<DataType> bufferDataTypes,
            CpuGeneratedKernel generatedKernel,
            CpuPreparedParallelConfiguration parallelConfiguration,
            CpuWorkerGroup workerGroup,
            CpuPortableInvocationBinder invocationBinder);

    CpuGeneratedKernel generatedKernel();
    MethodHandle entryPoint();

    @Override protected BoundInvocation bindCpu(
            RunState runState,
            CpuBufferArgument[] bufferArguments,
            CpuNativeWorkspace[] workspaces);
}
```

`CpuPortablePreparedExecutable` stores the exact `generatedKernel.entryPoint()` in its own final
field and checks that it remains the same reference returned by `generatedKernel.entryPoint()`.
This explicit direct-handle field makes the Runtime recipe's hot invocation dependency visible;
the retained generated kernel supplies the strong hidden-class/lookup/bytes lifetime.

No production `CpuPortableBoundInvocation` is added in this foundation. Generated signatures are
not uniform, so each later family must implement its own typed binder/invocation beside its
semantics. Task tests use nested synthetic signature-specific implementations and do not create a
generic production invoker.

## Analysis, selection, and resource contract

- `CpuPreparedParallelConfiguration` validates positive worker count, then positive minimum range
  size. It contains no executor, worker, route, target discovery, or ownership lifecycle.
- `CpuPortableAnalysisInputs` snapshots its supported Vector shapes in supplied order, rejects
  null/duplicate entries, and retains the exact non-null parallel configuration. An empty species
  list permits scalar candidates only.
- Candidate construction validates specialization, emitter, requirements, buffer uses,
  workspace uses, and binder in declaration order and snapshots every list.
- The emitter lowering fingerprint must exactly equal the specialization lowering fingerprint.
- Buffer uses align one-for-one and in order with `specialization.arguments()`. Each use references
  by exact object identity one declared `PreparationResourceRequirement.Buffer`; its projected
  value's `DataType` equals the aligned specialization argument type. The argument's carrier and
  `BufferAccess` are the complete selected representation/access facts.
- Workspace uses reference by exact object identity declared workspace requirements. Repeated
  uses may address one declaration, but every declared buffer and workspace requirement must be
  used at least once and no use may name an undeclared requirement.
- Requirement declarations remain unique under the existing
  `BackendPartitionAnalysis` contract. Candidate construction performs no allocation, slot
  assignment, artifact lookup, class generation, or invocation.
- Analysis validates `context` first, then exact CPU ownership, candidate-source result and
  entries, candidate-to-context resource/value agreement, and mode eligibility in source order.
- A scalar mode is eligible without a Vector species. A Vector mode is eligible only when its
  exact `VectorShape` occurs in `supportedVectorShapes`. A parallel mode is eligible only when
  the supplied configuration has the positive worker capacity already established by its
  constructor. No mode is inferred from a carrier or installed JDK module.
- The first fully valid eligible candidate is selected. Source order is the family owner's
  deterministic safe-heuristic order; this foundation adds no numeric score or universal route
  priority. An invalid candidate fails the analysis rather than being silently skipped. A valid
  but target-ineligible candidate may be skipped. No eligible candidate fails with
  `IllegalArgumentException("no supported CPU portable candidate")`.
- The returned analysis retains `context.partition()` by exact reference, wraps the selected
  candidate and exact parallel configuration in one immutable plan, and returns the candidate's
  exact ordered requirement references. Analysis does not mutate the source list or candidate.

## Finalization, binding, and hot-path contract

- `CpuPortablePartitionFinalizer` validates its explicit `artifactRoot`, then the borrowed open
  worker group. Construction may create the CPU-0003 store identity because that constructor does
  no filesystem access; only `finalizePartition` may call `loadOrGenerate`.
- `backendId()` returns `CpuCapabilityProvider.CPU_BACKEND_ID` by exact reference.
- Existing `BackendPartitionFinalization` construction remains authoritative for exact analysis,
  assignment order, source identity, plan position, slot identity, and geometry validation.
- The CPU finalizer first validates that the analysis partition remains CPU-owned, then maps every
  selected buffer/workspace use from its exact requirement reference to the already-validated
  assignment. It constructs all Runtime `BufferSelection`, `WorkspaceSelection`, `BufferAccess`,
  and `DataType` lists before filesystem or generator work.
- The retained worker group's count must equal the selected prepared parallel configuration. The
  group must be open before artifact work. The finalizer borrows it and never closes it.
- Only after those checks does finalization call
  `CpuGeneratedKernelArtifactStore.loadOrGenerate(specialization, familyEmitter)` exactly once.
  It neither changes the specialization nor retries outside the completed store contract.
- The resulting executable retains the exact shared `PreparedMemoryPlan` reference and exact
  generated artifact. It owns no closeable resource and performs no allocation, transfer,
  materialization, publication, or schedule assembly.
- Runtime's final `PreparedExecutable.bind` performs plan/open/range lookup and CPU 0001's base
  performs representation, lifetime, size, access, data-type, and alignment checks first.
- `CpuPortablePreparedExecutable.bindCpu` then validates buffer argument count and exact variant
  against each specialization carrier in order. Dynamic primitive-array offsets remain the
  already-classified carrier-relative offsets; exact-segment arguments require the exact
  `Segment` form and zero relative offset. No conversion or copy is permitted.
- For parallel modes, every `CpuBufferArgument.Segment` and workspace segment must satisfy
  `CpuWorkerGroup.isAccessibleByEveryWorker(...)` before the binder is called. Single-thread modes
  require current-thread accessibility already enforced by CPU 0001.
- After all checks, `bindCpu` calls the selected binder once with fresh arrays. The returned
  invocation must be non-null, retain the exact supplied `RunState`, direct handle, direct typed
  carrier/segment fields, and direct worker/workspace fields that its signature needs. It must
  not retain either array, a nominal representation, a slot-selection path, or
  `HostTensorStorage`.
- Synthetic tests inspect the bound invocation's direct fields and Class-File model to prove its
  hot call contains no route/cache/disk/hash/checksum/class-definition/lookup/reflection/storage-
  discovery/argument-classification/slot-lookup path. This recurring invariant belongs in tests,
  not repeated manual `javap` checks.
- `CpuBorrowedBuffer` remains the sole CPU bridge from caller-owned `HostTensorStorage` into
  Runtime's representation lifecycle. It is not an artifact cache, tuning cache, generated
  artifact, executable, or ownership transfer. Its task-0001 no-op close and caller lifetime
  remain unchanged.

## Validation and failure order

Automated tests must lock this order:

1. parallel configuration: worker count, minimum range size;
2. analysis inputs: supported-shape list, indexed entries/duplicates, parallel configuration;
3. candidate: specialization, emitter, lowering fingerprint match, requirements, uses, exact
   declaration-use association, binder;
4. preparer: context, CPU owner, candidate-source result, candidates in encounter order,
   context/resource agreement, target/mode eligibility, then first eligible selection;
5. shared Prepare's existing finalization validation: analysis, plan, assignments, exact
   requirement association, plan positions/slots, geometry;
6. CPU finalizer: CPU owner, complete use-to-assignment mapping, worker-count/open state, then one
   artifact-store call and executable construction;
7. Runtime/CPU cold binding: Runtime plan/open/selection checks, CPU 0001 representation checks,
   specialization carrier checks, parallel worker accessibility, then one typed binder call; and
8. bound execution: the existing state-open guard, then direct signature-specific handle call.

Stable CPU failures must identify indexed candidate/resource/use positions. Existing Prepare and
Runtime messages remain unchanged. Artifact failures retain CPU 0003's exact contextual causes;
the finalizer adds no fallback or wrapper that hides them.

## Affected files

Expected CPU production paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPreparedParallelConfiguration.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPortableAnalysisInputs.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPortableCandidateSource.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPortableKernelCandidate.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPortableInvocationBinder.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPortablePreparationPlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPortablePartitionPreparer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPortablePartitionFinalizer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPortablePreparedExecutable.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/package-info.java`

Expected CPU test paths:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPortablePartitionPreparerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPortablePartitionFinalizerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPortablePreparedExecutableTest.java`

Expected explanatory documentation:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`

Expected planning paths:

- this task
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

Review only, with a reasoned no-change conclusion unless implementation finds a concrete
contradiction:

- `ARCHITECTURE.md`, focused architecture pages, ADRs 0010/0011, and architecture tests;
- completed CPU 0001–0003 production, tests, Javadocs, task records, and CPU Gradle configuration;
- current `BackendAnalysisInputs`, `BackendPartitionPreparer`, `BackendPartitionAnalysis`,
  `PreparationResourceRequirement`, `BackendPartitionFinalizer`,
  `BackendPartitionFinalization`, `PreparationResourceAssignment`, and `PreparedPartition`;
- current Runtime `PreparedMemoryPlan`, `PreparedRepresentationPlan`, `PreparedExecutable`,
  `BoundInvocation`, `BufferRepresentation`, `WorkspaceRepresentation`, `RunState`, and runner;
- Model `HostTensorStorage`, projected graph/descriptor facts, capability contracts, public API
  documentation, glossary terminology, backend-conformance/integration placeholders, Config,
  Engine, other backends, extensions, and tools.

## Maximum scope

At most 18 paths:

- 10 CPU production paths;
- 3 CPU test paths;
- 2 explanatory documentation paths; and
- 3 planning paths.

No existing CPU production path other than `execution/package-info.java` may change. No existing
test, build, shared-module, architecture, ADR, conformance, integration, public API, other backend,
or later task-specification path may change. If implementation needs another path, a public
composition owner, a change to `CpuPreparedExecutable`, a shared contract extension, or a generic
invocation adapter, stop and update planning rather than expanding silently.

## Acceptance criteria

- [x] The exact package-private surface and package placement above are implemented with no new
      public CPU type or package.
- [x] CPU analysis implements the existing typed `BackendPartitionPreparer` contract, accepts
      only an already CPU-owned exact partition, retains that partition, and exposes no Compiler
      aggregate or second graph model.
- [x] Exact immutable CPU inputs and parallel configuration are validated and deeply snapshotted;
      no Config API, discovery, default, global state, or mutable target observation is added.
- [x] The candidate source is one direct typed collaboration, not a registry, service locator,
      reflection mechanism, generic parameter bag, or `Map<String, ?>`.
- [x] Responsibilities remain split among immutable candidate/plan values, the narrow
      analysis/filter/selection preparer, the assignment/artifact/executable finalizer, and the
      cold executable/binder boundary; no god analyzer, finalizer, manager, or broad facade is
      introduced.
- [x] Complete candidates align exact resource declarations and uses with the exact
      `CpuKernelSpecialization` arguments, emitter fingerprint, typed carriers, accesses, and
      binder, with no undeclared or unused requirement.
- [x] Analysis hard-filters owner, context/resource, Vector-species, execution-mode, and parallel
      compatibility in the specified order, selects the first valid eligible candidate
      deterministically, and fails closed when none exists.
- [x] Every selected shared buffer/workspace requirement is exact and returned before slot
      assignment; analysis performs no artifact, generation, allocation, executable, Runtime,
      measurement, tuning, or cache-mutation work.
- [x] Finalization implements only the existing `BackendPartitionFinalizer` handoff, maps all
      assignments before artifact access, and neither changes selection nor adds a requirement.
- [x] The explicit artifact root is caller-supplied to the CPU-private finalizer. No default root,
      home/environment lookup, public Config field, Engine facade, or filesystem access outside
      finalization is introduced.
- [x] CPU 0003's store is called exactly once only after assignment/worker validation; its format,
      key, validation, publication, concurrency, weak retention, security, and failure behavior
      remain unchanged.
- [x] The immutable prepared executable retains the exact shared memory plan, selected resource
      recipes, `CpuGeneratedKernel`, direct entry handle, specialization, parallel configuration,
      worker reference, and typed binder, while owning no closeable physical resource.
- [x] `CpuBorrowedBuffer` remains the non-owning `HostTensorStorage` representation/lifetime
      boundary and is consumed through Runtime `BufferRepresentation` cold binding. No direct
      `HostTensorStorage` argument binding or artifact-cache meaning is introduced.
- [x] Cold binding checks exact heap/segment/mixed carrier compatibility and parallel worker
      accessibility once, then returns a family-owned typed `BoundInvocation` retaining direct
      fields rather than fresh arrays, nominal representations, storage, or slot lookups.
- [x] Synthetic direct invocations exercise all four portable execution modes and call exact
      handles without reflection, adapters, `invokeWithArguments`, generic objects, or argument
      classification in `executeBound()`.
- [x] Runtime hot execution contains no route, cache, disk, hash, checksum, class-file parsing,
      class definition, lookup, reflection, storage discovery, heap-base inspection, argument
      classification, graph inspection, backend discovery, slot lookup, tuning, or allocation.
- [x] Tests are bounded and synthetic. They establish lifecycle mechanics only and make no Model
      numerical result, operation-family coverage, capability, conformance, or integration claim.
- [x] `CpuCapabilityProvider` and all completed CPU 0001–0003 source/contracts remain unchanged;
      support remains false for every non-null query.
- [x] CPU 0001–0003 remain `Complete`; CPU 0004 is synchronized through implementation; CPU
      0005–0016 remain `Draft` without detailed specifications; future Model 0026 remains present.
- [x] Focused development tests and one final `:backends:cpu:test` run after executable Java
      stability pass without an installed native provider or external artifact root.
- [x] New/changed production declarations have complete Javadoc for purpose, immutability,
      ownership, lifetime, threading, inputs, returns, nullability, failures, and hot/cold
      boundaries.
- [x] A separate clean documentation-focused pass finalizes affected Javadocs, CPU guidance,
      glossary impact, planning evidence, links, and status in the same overall change, reusing
      Java evidence unless it changes executable behavior or records a concrete stale-evidence
      risk.
- [x] CPU Javadoc, generated pages, Markdown links/anchors/fences/newlines, exact 18-path scope,
      surface/mechanism/status/later-specification/excluded-path checks, trailing whitespace, and
      `git diff --check` pass.

## Tests / validation

Implementation development may run focused classes:

```bash
./gradlew :backends:cpu:test --tests '*CpuPortablePartitionPreparerTest'
./gradlew :backends:cpu:test --tests '*CpuPortablePartitionFinalizerTest' --tests '*CpuPortablePreparedExecutableTest'
```

The focused matrix must cover:

- constructor nulls, snapshots, duplicates, stable failure order/messages, and exact package-
  private/final surface;
- wrong-owner rejection before candidate-source invocation;
- empty, null, malformed, target-ineligible, first-valid, all-four-mode, and concurrent analysis;
- exact projected values, requirements, repeated uses, representation indices, carrier/access/
  type alignment, Vector species, and parallel configuration;
- proof that analysis performs no filesystem, artifact, generation, definition, allocation,
  binding, worker dispatch, measurement, or cache mutation;
- complete assignment mapping before store use; missing/mismatched assignment failure without
  artifact access; valid hit/miss and strong generated-artifact retention;
- heap array, read-only exact heap segment, native exact segment, and mixed cold binding through
  `CpuBorrowedBuffer`/`CpuNativeBuffer`, including offset, access, liveness, alignment, and worker
  accessibility failures;
- exact direct handle identity and signature-specific synthetic bound fields, repeated and
  concurrent binds to distinct `RunState` instances, closed-state rejection, and all four mode
  executions; and
- source/Class-File assertions excluding public surface, `HostTensorStorage` direct binding,
  Runtime hot-path discovery, reflection/adaptation, generic objects, registries, service
  locators, tuning, vendor routes, and operation/capability claims.

After executable Java stabilizes, run one final CPU module test exactly once:

```bash
./gradlew :backends:cpu:test
```

Implementation pass also runs:

```bash
git diff --check
```

After the implementation pass records the exact final CPU test result, hand the diff and evidence
to a separate clean documentation-focused agent/thread. That pass follows
`docs/developer-guide/documentation-rules.md`, the General, API/Javadoc, Backend Guide, Planning,
and Example profiles as applicable, and runs after final Javadoc edits:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

The documentation pass must validate repository-local Markdown links and heading anchors for
this task, CPU master plan, roadmap, CPU guide, and glossary; balanced fences, final newlines, and
trailing whitespace; generated pages for every new/changed production declaration; exact 18-path
scope; package-private surface; CPU 0001–0004 synchronization; CPU 0005–0016 `Draft` rows and
absence of their detailed specs; preserved Model 0026; and unchanged excluded paths. It reuses the
successful Java evidence and does not rerun Java tests unless it changes executable Java behavior
or records a concrete stale-evidence risk.

Repository-wide Java, architecture, backend-conformance, and integration validation remain
deferred to CPU 0009 or continuous integration because this task changes one backend-private
foundation and no dependency/architecture boundary. If implementation changes a shared contract,
dependency, build rule, public capability, or another module, stop instead of broadening the
validation tier.

## Dependencies

- Complete CPU 0001 capability, non-owning borrowed representation, native representations,
  typed buffer arguments, CPU prepared-executable base, direct binding contract, and worker group.
- Complete CPU 0002 exact specialization, four modes, family emitter, deterministic verified
  bytes, direct typed entry handle, and generated-artifact lifetime.
- Complete CPU 0003 explicit-root artifact store, cold loading, atomic persistence, weak interning,
  and strong-caller lifetime contract.
- Complete Prepare analysis/resource/finalization/orchestration contracts, especially
  `BackendAnalysisInputs`, `BackendPartitionPreparer`, `BackendPartitionAnalysis`,
  `BackendPartitionFinalizer`, and `BackendPartitionFinalization`.
- Complete Runtime prepared memory, representation, `PreparedExecutable`, `BoundInvocation`,
  `RunState`, schedule, and runner contracts.
- Current Model projected static graph/descriptor facts and borrowed `HostTensorStorage`; this task
  changes neither.
- Java 26 Class-File and Vector APIs already configured in the CPU module.

## Follow-up tasks

- CPU 0005 remains Draft and is the first owner of real portable elementwise/pointwise candidate
  generation, family emitters, signature-specific typed binders/invocations, numerical tests, and
  truthful capability additions.
- CPU 0006–0008 remain Draft for later portable families; CPU 0009 remains the portable generated-
  coverage/capability/conformance checkpoint.
- CPU 0010–0015 remain Draft optional native-route work. CPU 0016 remains Draft compatible tuning-
  evidence selection and does not replace the generated-class artifact store.
- A later Engine/CPU composition task must own construction/lifetime of the worker group and
  explicit trusted artifact root before public end-to-end use. This task deliberately adds no
  default or public facade.
- Future Model 0026 remains Draft and is required before any FLOAT16 or mixed-precision backend
  claim.

## Architecture impact

Expected impact: None.

This task implements the existing concrete-backend analysis/finalization, exact-resource,
prepared-recipe, per-run representation, and cold-binding responsibilities. It changes no shared
contract, dependency edge, lifecycle stage, or authority. If implementation requires an
architecture change, stop and report the exact conflict.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md,
docs/planning/backends/cpu/master-plan.md, and
docs/planning/backends/cpu/tasks/0004-typed-portable-analysis-specialization-and-finalization.md.
Read completed CPU tasks 0001–0003 and only the directly relevant current CPU, Prepare, Runtime,
Model storage, architecture, documentation, and Java 26 contracts named by the task.

Implement CPU task 0004 exactly as specified. Preserve CpuBorrowedBuffer as the Runtime
BufferRepresentation lifetime boundary, keep capability fail-closed, and use only bounded
synthetic candidates/tests. Do not implement a Model operation family, public facade, registry,
service locator, default artifact-root Config API, vendor route, tuning/benchmark work, shared
module/build/architecture change, or later CPU task. Stop and report any architecture, package,
typed-invocation, artifact-lifecycle, or maximum-scope conflict instead of inventing a boundary.

After Java implementation and the one final CPU test run, hand the resulting diff and exact test
evidence to a separate documentation-focused agent/thread with clean context. That pass must
follow docs/developer-guide/documentation-rules.md, independently finalize affected Javadocs,
CPU guidance, glossary impact, planning evidence, and documentation validation in the same
overall change, and must not repeat successful Java tests unless executable behavior changes or
a concrete stale-evidence risk is recorded.

Update this task with local decisions, known limitations, validation evidence, implementation
notes, completion summary, and synchronized final status. Do not mark Complete before every
acceptance criterion and the documentation pass succeed.
```

## Local decisions

- Preserve `CpuBorrowedBuffer` exactly as CPU 0001's non-owning representation of caller-owned
  `HostTensorStorage`. It is the required Model-storage-to-Runtime-representation boundary used
  for caller inputs and later cold `CpuBufferArgument` classification. Direct
  `HostTensorStorage`-to-argument binding was rejected because it would reopen completed CPU 0001,
  bypass Runtime's `BufferRepresentation` lifecycle/ownership model, and erase the checked cold
  boundary. `CpuBorrowedBuffer` is not an artifact cache; its first production portable execution
  use begins in this task.
- Keep all new types package-private in `execution`. The current collaboration needs direct access
  to completed CPU internals and does not justify a public backend facade or cross-package API.
- Represent typed portable storage selection with the existing
  `CpuKernelSpecialization.Argument` carrier/access vocabulary rather than adding a duplicate
  storage-kind enum.
- Use one injected typed `CpuPortableCandidateSource` because CPU 0005+ family owners need a
  direct seam and CPU 0004 must test selection without implementing operation semantics. A
  registry, service locator, reflection, and generic parameter bag were rejected.
- Select the first fully valid target-eligible candidate in deterministic source order. This is a
  bounded safe-heuristic seam, not a broad cost model or universal route priority; later family
  tasks own their candidate order.
- Keep invocation construction family-owned. Generated `MethodType` values vary by carrier,
  offset/extent baking, and mode, so a generic production invocation would require forbidden
  object arrays, adaptation, or hot argument switching.
- Supply the artifact root and worker group explicitly to the CPU-private finalizer. The store is
  used only after assignment during finalization, and the worker group is borrowed rather than
  allocated or closed by the prepared recipe. Public composition and defaults remain later work.
- Strongly retain both `CpuGeneratedKernel` and its exact direct handle in the immutable prepared
  executable. The artifact owns hidden-class lifetime; the direct field makes the hot dependency
  explicit without another lookup.
- Preserve the backend-declared resource geometry exactly during CPU analysis. Projected value
  identity and data type constrain a candidate, but task 0004 does not derive dense byte size from
  `TensorDescriptor`, infer a layout, or choose materialization. The existing shared
  `BackendPartitionFinalization` contract remains the sole validation boundary between each exact
  declaration and its assigned `PreparedMemoryPlan` geometry.

## Known limitations

- CPU still advertises and truthfully executes no Model operation after this foundation task.
- Only bounded synthetic sources, emitters, binders, and invocations prove the lifecycle. They
  are test code and do not establish a numerical result or operation-family capability.
- No production family candidate source exists until CPU 0005. Consequently no public or Engine-
  composed end-to-end CPU execution exists at this frontier.
- The explicit artifact root and worker group have no public composition owner yet. This task
  defines only the CPU-private constructor seam and borrows the worker lifetime.
- The foundation selects among portable generated candidates only. Native providers, tuning
  evidence, transition-cost comparison, packing, materialization, and representation-plan
  reconciliation remain later work.
- Prepare still requires fully static Shapes. The generated specialization may carry dynamic
  invocation offsets/extents, but this task adds no new run-dynamic Shape contract.
- Generated signature-specific binders remain family-owned, so later family tasks must test every
  signature they introduce without weakening the direct hot-path rule.
- FLOAT16 and mixed precision remain unavailable pending Model 0026 and later exact backend-route
  evidence.

## Validation evidence

- Implementation compilation: `./gradlew :backends:cpu:compileJava` and
  `./gradlew :backends:cpu:compileTestJava` passed before the final executable runs.
- Final focused implementation evidence:
  `./gradlew :backends:cpu:test --tests io.github.pho001.synaptik.backend.cpu.execution.CpuPortablePartitionPreparerTest --tests io.github.pho001.synaptik.backend.cpu.execution.CpuPortablePartitionFinalizerTest --tests io.github.pho001.synaptik.backend.cpu.execution.CpuPortablePreparedExecutableTest`
  passed with `BUILD SUCCESSFUL`: three suites, 24 tests (6 + 7 + 11), and zero failures,
  errors, or skips. The expanded matrix directly covers exact surface and validation order,
  immutable snapshots, deterministic all-mode selection, the opaque 12-byte declaration against
  projected FLOAT32 `Shape[4]`, analysis side-effect exclusion, assignment-before-artifact
  failures, artifact miss/reuse and strong retention, borrowed/native heap/segment/mixed binding,
  repeated resources, dynamic/baked offsets, worker accessibility, concurrent binds, exact handle
  calls, binder result association, and source/Class-File forbidden-mechanism checks.
- Final implementation module evidence: the sole stabilized
  `./gradlew :backends:cpu:test` run passed with `BUILD SUCCESSFUL`, 13 suites, 72 tests, zero
  failures, errors, or skips, and 21 actionable tasks (one executed and 20 up-to-date).
  Implementation `git diff --check` and forbidden-mechanism/public-surface scans also passed. The
  earlier 58-test module result and one failed provisional dense-byte-geometry development run are
  superseded baseline/remediation evidence, not final results; the provisional geometry change
  was completely rolled back. No executable Java changed after the final 72-test run.
- Mandatory separate clean documentation context `/root/cpu_0004_docs` applied the General,
  API/Javadoc, Backend Guide, Planning, and Example profiles. It independently reviewed all ten
  production paths and three prescribed tests, generated Javadoc, the CPU guide, glossary,
  task/master/roadmap records, completed CPU tasks 0001–0003, the staged Prepare and Runtime
  contracts, Model storage lifetime, fail-closed capability, architecture/ADR boundaries, and
  Java 26 CPU build configuration. It changed documentation and Javadoc only, so it reused the
  final Java evidence and did not rerun Java tests.
- Documentation validation: `./gradlew :backends:cpu:javadoc` passed after final Javadoc edits
  with `BUILD SUCCESSFUL` and 11 actionable tasks (two executed and nine up-to-date); the expected
  Java Vector API incubator warnings and pre-existing task-0001 documentation warnings were
  non-failing. Generated declaration pages were present for all nine new package-private types,
  their nested records, and the changed execution package page.
- Repository-local Markdown links and heading anchors across this task, CPU master plan, roadmap,
  CPU guide, and glossary passed. The same five paths passed balanced-fence and final-newline
  checks; all 18 changed paths passed trailing-whitespace checks. `git diff --check` passed on the
  final combined diff.
- Exact-scope validation found exactly the authorized 18 paths: ten CPU production paths, three
  CPU tests, two explanatory documentation paths, and three planning paths. Source/reflection and
  generated-page checks confirmed every new type remains package-private, the exact task surface
  is present, `CpuBorrowedBuffer` remains the `BufferRepresentation` lifetime bridge, and
  `CpuCapabilityProvider` remains unchanged and fail-closed.
- CPU 0001–0004 status synchronization, CPU 0005–0016 `Draft` rows, absence of detailed task files
  for 0005–0016, and the preserved future Model 0026 dependency passed. Excluded architecture,
  ADR, architecture-test, shared API, Config, Engine, other-backend, extension, tool, Gradle,
  conformance, and integration paths remained unchanged.

Reasoned no-change conclusions:

- `ARCHITECTURE.md`, focused architecture pages, ADRs 0010/0011, and architecture tests already
  define backend-owned analysis, exact declarations, post-assignment finalization, artifact reuse,
  per-run representations, and cold binding. Task 0004 implements those boundaries without a new
  dependency edge or rule.
- Shared Model, Planning, Prepare, Runtime, Backend Contract, Trace, Compiler, and public API
  contracts remain accurate. In particular, `HostTensorStorage` remains caller-owned,
  `CpuBorrowedBuffer` remains its non-owning Runtime representation, and shared Prepare—not CPU
  analysis—validates declared byte geometry against assigned plan geometry.
- Capability/public documentation, Config, Engine, OpenBLAS and other backends, extensions, tools,
  and CPU tasks 0001–0003 need no change: task 0004 adds no public composition, operation family,
  capability, vendor route, default artifact root, tuning/benchmark behavior, or completed-contract
  revision.
- Backend-conformance and integration tests remain unchanged because the implementation proves
  only package-private synthetic lifecycle mechanics and advertises no executable Model semantic.
  CPU and root Gradle configuration remain unchanged because task 0002 already supplies the Java
  26 Class-File/Vector API setup required here.

## Implementation notes

- Added immutable CPU-private analysis inputs, prepared parallel configuration, complete candidate
  and opaque-plan values, one direct candidate source, and one family-owned typed binder seam.
- Added deterministic CPU analysis that preserves exact backend declarations, selects the first
  valid target-eligible candidate, and performs no artifact, allocation, generation, binding,
  measurement, tuning, or Runtime work.
- Added CPU finalization that resolves every assigned use before one artifact-store call and
  constructs an immutable generated executable retaining the exact artifact, handle, prepared
  recipes, borrowed worker group, and typed binder.
- Extended the existing CPU cold boundary without changing it: heap arrays, exact segments,
  `CpuBorrowedBuffer`, `CpuNativeBuffer`, workspaces, offsets, access, liveness, alignment, and
  parallel accessibility are validated before one signature-specific binder call.
- Finalized all affected production/package Javadocs, added the typed preparation lifecycle and
  synthetic-proof boundary to the CPU guide, added reusable glossary definitions, and synchronized
  planning evidence and status.

## Completion summary

- Completed changes: CPU-private typed portable analysis, exact pre-assignment declarations,
  post-assignment artifact finalization, immutable strong generated-artifact retention, and direct
  per-run cold binding through the preserved Runtime representation boundary.
- Files changed or created: exactly the ten listed CPU production paths, three prescribed CPU
  tests, CPU backend guide, glossary, this task, CPU master plan, and roadmap.
- Tests and validation: the implementation pass recorded the final focused 3-suite/24-test pass
  and sole stabilized CPU 13-suite/72-test pass with zero failures, errors, or skips; the clean
  documentation context reused those results and passed CPU Javadoc, generated-page, Markdown,
  exact-scope, surface, status, excluded-path, and whitespace validation.
- Documentation-agent review: `/root/cpu_0004_docs` independently finalized the combined change
  using the General, API/Javadoc, Backend Guide, Planning, and Example profiles without changing
  executable behavior.
- Documentation impact: the CPU guide and glossary now explain the staged typed lifecycle,
  explicit root/borrowed-worker ownership, exact declaration-versus-assignment boundary,
  `CpuBorrowedBuffer` lifetime role, direct hot call, and synthetic-only limitation.
- Javadoc review: every new production declaration, nested record, constructor, method, package
  boundary, parameter, result, expected failure, ownership/lifetime rule, and concurrency boundary
  was reviewed and final Javadoc generation passed.
- Glossary impact: added current CPU portable preparation-plan and prepared-executable terms and
  synchronized the generic preparer/finalization implementation status.
- Architecture, dependency, build, shared API, ADR, architecture-test, capability, conformance,
  and integration impact: none; reasoned no-change conclusions are recorded above.
- Unresolved issues: None within CPU 0004 scope.
- Follow-up required: None for task 0004. CPU 0005 remains Draft and owns the first real portable
  operation-family implementation and truthful capability addition.

Status: Complete
