# Task 0004: Typed Portable Analysis, [Specialization](../../../../glossary.md#term-specialization), and Finalization

## Reader context

The participants here are CPU analysis, shared Prepare, CPU finalization, and Runtime—not four
interchangeable execution stages. Analysis returned candidate descriptions and exact
buffer/workspace requirements without allocating Tensor storage. Shared Prepare assigned
positions. CPU finalization then loaded or generated the selected code and returned an immutable
executable recipe. Runtime supplied current run representations to `PreparedExecutable.bind`,
which constructed a signature-specific invocation.

The prepared recipe strongly retained its generated artifact but only borrowed the supplied
worker group. Application input storage remained borrowed through `CpuBorrowedBuffer`; internal
storage remained run-owned. The candidates and emitters used to prove this checkpoint were
synthetic, so its completed lifecycle was not evidence of real operation support. The old
package-private surface below is preserved as a superseded design.

Current reading alongside this historical record:

- [CPU analysis and finalization](../../../../explanation/packages/cpu.md#iogithubpho001synaptikbackendcpuinternalprepare)
  — explains declaration before assignment and realization afterward.
- [CPU binding, validation, and ranges](../../../../explanation/packages/cpu.md#iogithubpho001synaptikbackendcpuinternalexecutable)
  — identifies current-value validation, direct invocation state, and worker/resource lifetimes.

## Status

Superseded

## Goal

Connect the completed [CPU](../../../../glossary.md#term-cpu) representation, generated-[kernel](../../../../glossary.md#kernel), and durable-[artifact](../../../../glossary.md#term-artifact) foundations to
the existing staged [Prepare](../../../../glossary.md#prepare) and [Runtime](../../../../glossary.md#term-runtime) contracts without claiming a real [Model](../../../../glossary.md#term-model) [operation](../../../../glossary.md#operation):

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

This task delivers the CPU-private [lifecycle](../../../../glossary.md#lifecycle) foundation used by later portable operation-family
tasks. It proves the complete analysis-to-bound-invocation path only with bounded synthetic
candidates and emitters. `CpuCapabilityProvider.supports(...)` remains unconditionally false;
CPU 0005 and later tasks must add truthful family computation and capability coverage before any
Model operation is advertised.

## [Scope](../../../../glossary.md#term-scope)

- Add one package-private [immutable](../../../../glossary.md#term-immutable) [CPU](../../../../glossary.md#term-cpu) implementation of `BackendAnalysisInputs` carrying only
  the exact supported Vector species snapshot and prepared parallel configuration supplied by a
  later CPU composition owner.<span data-glossary-navigation="true"> ([Vector species glossary](../../../../glossary.md#term-vector-species); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- Add one package-private immutable prepared parallel configuration with positive worker count,
  positive minimum range size, and explicit deterministic/non-deterministic range intent.
- Add one package-private direct typed candidate-source collaboration. It receives the complete
  `PrepareContext<CpuPortableAnalysisInputs>` and returns candidates in deterministic preference
  order. The task supplies no production operation-family implementation; tests supply bounded
  synthetic sources only.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- Add one immutable complete portable candidate that binds together exactly one
  `CpuKernelSpecialization`, its matching `CpuFamilyKernelEmitter`, its ordered exact buffer and
  workspace declarations, its ordered buffer/workspace uses, and one family-owned typed
  invocation binder.<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer))</span>
- Treat each `CpuKernelSpecialization.Argument` as the exact selected typed portable
  representation and access fact for its aligned buffer use. The carrier is selected before
  shared slot assignment and is never rediscovered from storage during execution.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Analyze only a [partition](../../../../glossary.md#partition) whose `PlannedPartition.owner()` equals
  `CpuCapabilityProvider.CPU_BACKEND_ID`; reject another [owner](../../../../glossary.md#term-owner) before candidate generation.
- Validate every candidate against the exact [`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext): all resource values are projected,
  buffer uses align one-for-one with specialization arguments, every use refers by exact
  reference to a declared requirement, every declaration is used, data type/access/carrier facts
  agree, and no declaration or use introduces an unprojected value or undeclared workspace.<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization); [data type glossary](../../../../glossary.md#data-type--datatype))</span>
- Filter [Vector](../../../../glossary.md#term-vector) candidates against the exact supported `VectorShape` snapshot and filter parallel
  candidates against the prepared worker configuration. Select the first valid candidate in the
  source's deterministic preference order. Fail closed when the source is empty, contains an
  invalid result, or has no eligible candidate.<span data-glossary-navigation="true"> ([Fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation))</span>
- Return `BackendPartitionAnalysis<CpuPortablePreparationPlan>` retaining the exact context
  partition, selected immutable plan, and the selected candidate's exact ordered declarations.
- Add one CPU-private `BackendPartitionFinalizer<CpuPortablePreparationPlan>` with exact `cpu`
  identity. Its constructor receives an explicit artifact root and an already-owned<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [artifact glossary](../../../../glossary.md#term-artifact))</span>
  [`CpuWorkerGroup`](../../../../glossary.md#cpu-worker-group--cpuworkergroup); it invents no default root, Config API, environment lookup, home-directory
  convention, worker owner, registry, or service locator.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [lookup glossary](../../../../glossary.md#term-lookup); [registry glossary](../../../../glossary.md#term-registry); [service locator glossary](../../../../glossary.md#term-service-locator))</span>
- During finalization, consume only the existing validated
  `BackendPartitionFinalization`/`PreparationResourceAssignment` handoff. Resolve every buffer and
  workspace use to its assigned dense plan position before any artifact-store operation.<span data-glossary-navigation="true"> ([handoff glossary](../../../../glossary.md#term-handoff))</span>
- Consult `CpuGeneratedKernelArtifactStore` only after all assignments, worker configuration, and
  selected-plan facts validate. A compatible hit or verified miss follows the completed CPU 0003
  contract unchanged.
- Construct an immutable `CpuPortablePreparedExecutable` that strongly retains the exact
  `CpuGeneratedKernel`, the exact direct [`MethodHandle`](../../../../glossary.md#native-numerical-library-and-instruction-architecture) returned by that artifact, the exact
  specialization, prepared parallel configuration, borrowed worker-group reference, selected
  resource recipes, and family-owned typed invocation binder.<span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease))</span>
- Extend the existing `CpuPreparedExecutable` cold boundary. Validate the actual
  `CpuBufferArgument` variant against the selected specialization carrier, validate dynamic array
  offsets and exact segment form without copying, and prove every segment used by a parallel mode
  is accessible to every retained worker before invoking the binder.
- Require the family-owned binder to copy the exact handle, direct array/segment carrier fields,
  primitive offsets/extents/range configuration, and direct workspace fields needed by its
  signature into a typed [`BoundInvocation`](../../../../glossary.md#bound-invocation--boundinvocation). It must not retain the fresh cold-[binding](../../../../glossary.md#term-binding) arrays.
- For the bounded synthetic proof, use [signature](../../../../glossary.md#term-signature)-specific test-only `BoundInvocation` subclasses
  whose `executeBound()` calls the exact handle without `invokeWithArguments`, spreaders,
  collectors, reflection, `Object[]`, or [argument](../../../../glossary.md#term-argument)-kind switches.
- Exercise all four existing `CpuPortableExecutionMode` values, heap, exact-segment, and mixed
  signatures, single-thread and worker-dispatched parallel paths, shared assignment, artifact
  reuse, repeated/concurrent binds, failure order, and direct hot calls with synthetic work only.<span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap))</span>
- Preserve `CpuBorrowedBuffer` as the completed CPU 0001 non-owning
  `HostTensorStorage` representation/lifetime boundary. Production portable cold binding in this
  task consumes it through Runtime's <span data-glossary-navigation="true"> ([lifetime glossary](../../../../glossary.md#term-lifetime); [Runtime glossary](../../../../glossary.md#term-runtime))</span>`BufferRepresentation`/[`RunState`](../../../../glossary.md#run-state--runstate) path; no executable or
  binder accepts `HostTensorStorage` directly.
- Finalize affected Javadocs, CPU guidance, glossary impact, and planning evidence through a
  separate clean documentation-focused handoff after Java stabilizes.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- any truthful Model operation-family computation, numerical algorithm, Tensor result,
  operation-kind lowering, fusion, capability advertisement, backend-conformance claim, or
  integration claim;<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [operation glossary](../../../../glossary.md#operation); [Tensor glossary](../../../../glossary.md#tensor); [lowering glossary](../../../../glossary.md#lowering); [fusion glossary](../../../../glossary.md#term-fusion); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- changing `CpuCapabilityProvider`, including returning `true` for a synthetic, metadata-only,
  zero-element, view-like, or otherwise bounded test occurrence;<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [element glossary](../../../../glossary.md#term-element); [view glossary](../../../../glossary.md#term-view-meanings); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- a public CPU backend, public facade, public preparer/finalizer, public route/candidate/config
  API, registry, service locator, generic manager, reflective discovery, or default artifact-root
  Config API without an existing composition owner;<span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [registry glossary](../../../../glossary.md#term-registry); [service locator glossary](../../../../glossary.md#term-service-locator); [artifact glossary](../../../../glossary.md#term-artifact); [owner glossary](../../../../glossary.md#term-owner))</span>
- replacing `CpuBorrowedBuffer` with direct `HostTensorStorage`-to-`CpuBufferArgument` binding,
  bypassing <span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>`BufferRepresentation`, changing [borrowed](../../../../glossary.md#ownership-borrowing-and-lease) [lifetime](../../../../glossary.md#term-lifetime), or reopening [CPU](../../../../glossary.md#term-cpu) 0001 storage;
- changing the CPU 0001 buffer/workspace allocation, representation, ownership, close, or worker
  lifecycle contracts;<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- changing the CPU 0002 schema, specialization equality/fingerprint, four-mode vocabulary,
  generated entry shape, emitter ownership, Class-File API mechanism, or Vector build flags;<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [specialization glossary](../../../../glossary.md#term-specialization); [fingerprint glossary](../../../../glossary.md#term-fingerprint); [Class-File API glossary](../../../../glossary.md#term-class-file-api); [Vector glossary](../../../../glossary.md#term-vector))</span>
- changing the CPU 0003 artifact format, key, root trust boundary, publication, validation,
  single-flight, weak interning, corruption, or lifetime policy;<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [validation glossary](../../../../glossary.md#term-validation); [single-flight glossary](../../../../glossary.md#single-flight-and-least-recently-used-caching); [weak interning glossary](../../../../glossary.md#weak-reference-and-weak-interning))</span>
- a tuning cache, compatible tuning-decision input, measurements, benchmarks, autotuning,
  broad cost model, transition-cost model, performance threshold, vendor priority, or mutation of
  any persistent evidence;<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [autotuning glossary](../../../../glossary.md#autotuning--model-autotuning); [cost model glossary](../../../../glossary.md#compiler-transformations-and-dominance))</span>
- [OpenBLAS](../../../../glossary.md#term-openblas), [oneMKL](../../../../glossary.md#term-onemkl), [oneDNN](../../../../glossary.md#term-onednn), Accelerate, [AOCL](../../../../glossary.md#term-aocl), [ZenDNN](../../../../glossary.md#term-zendnn), native vendor, [Metal](../../../../glossary.md#term-metal), or [CUDA](../../../../glossary.md#term-cuda) routes;
- operation-family kernels assigned to CPU 0005–0008, the CPU 0009 capability checkpoint, or any
  CPU 0010–0016 behavior or detailed specification;<span data-glossary-navigation="true"> ([kernels glossary](../../../../glossary.md#kernel); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- FLOAT16 or mixed-precision semantics/capability; future Model 0026 remains the prerequisite for
  any FLOAT16 claim;<span data-glossary-navigation="true"> ([FLOAT16 glossary](../../../../glossary.md#term-float16); [precision glossary](../../../../glossary.md#term-precision))</span>
- a new schedule assembler, public composition owner, PreparedExecution owner/close lifecycle,
  prepared-resource serialization, worker pooling, allocation pooling, materialization policy,
  transfer policy, publication behavior, or output access;<span data-glossary-navigation="true"> ([PreparedExecution glossary](../../../../glossary.md#prepared-execution--preparedexecution); [close lifecycle glossary](../../../../glossary.md#ownership-borrowing-and-lease); [serialization glossary](../../../../glossary.md#term-serialization); [pooling glossary](../../../../glossary.md#term-pooling); [materialization glossary](../../../../glossary.md#materialization))</span>
- changes to Model, Config, Planning, Prepare, Runtime, Backend Contract, Trace, Compiler, Engine,
  OpenBLAS provider, other backends, extensions, tools, Gradle, dependencies, architecture,
  focused architecture documentation, ADRs, architecture tests, backend conformance, or
  integration tests; and<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Trace glossary](../../../../glossary.md#trace); [Compiler glossary](../../../../glossary.md#term-compiler); [Engine glossary](../../../../glossary.md#term-engine); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- [commits](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings) or pushes.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
  - Core invariants
  - `modules/runtime`
  - `modules/prepare`
  - [Concrete backend](../../../../glossary.md#concrete-backend) [modules](../../../../glossary.md#term-module)
  - [CPU](../../../../glossary.md#term-cpu) [backend](../../../../glossary.md#backend) routes
  - [Prepare](../../../../glossary.md#prepare) [lifecycle](../../../../glossary.md#lifecycle)
  - [Run](../../../../glossary.md#term-run) lifecycle
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Runtime, Prepare, and Backend Boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime))</span>
- [ADR 0010: Staged backend preparation](../../../../design/decisions/0010-staged-backend-preparation.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- [ADR 0011: Per-run Runtime resource ownership and cold binding](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)<span data-glossary-navigation="true"> ([resource ownership glossary](../../../../glossary.md#term-resource-ownership))</span><span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership); [binding glossary](../../../../glossary.md#term-binding))</span>
- [CPU backend guide](../../../../backend-guide/cpu-backend.md)
- [Task 0001 CPU capability, representation, binding, and parallel foundation](0001-cpu-capability-representation-binding-and-parallel-foundation.md)
- [Task 0002 Portable Class-File API generator foundation](0002-portable-class-file-api-generator-foundation.md)<span data-glossary-navigation="true"> ([Class-File API glossary](../../../../glossary.md#term-class-file-api))</span>
- [Task 0003 Durable generated-kernel artifact store and cold loading](0003-bounded-generated-artifact-cache-and-cold-finalization.md)<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel); [artifact glossary](../../../../glossary.md#term-artifact))</span>

## Architecture constraints

- Planning has already selected CPU ownership. CPU analysis may reject that partition or select
  its private portable route; it must not reconsider backend ownership.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [CPU glossary](../../../../glossary.md#term-cpu); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [partition glossary](../../../../glossary.md#partition); [backend ownership glossary](../../../../glossary.md#backend-ownership))</span>
- [`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext) is the complete concrete-backend-facing semantic/planning projection. CPU must
  not accept or retain <span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>`CompileArtifacts`, [Compiler](../../../../glossary.md#term-compiler) aggregates, a [Tensor](../../../../glossary.md#tensor), or a second [graph](../../../../glossary.md#graph) [model](../../../../glossary.md#term-model).
- Analysis is deterministic from the exact context, explicit CPU inputs, and the direct candidate
  source. It performs no filesystem access, artifact lookup, class generation/definition,
  allocation, measurement, cache mutation, executable construction, or Runtime work.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [lookup glossary](../../../../glossary.md#term-lookup); [cache glossary](../../../../glossary.md#term-cache); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- The selected `CpuPortablePreparationPlan` is immutable opaque backend state. It retains the
  selected lowering/emitter, specialization, resource-use mapping, binder, and parallel recipe;
  it is not a generated artifact, executable, physical resource, slot assignment, or per-run
  object.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [opaque glossary](../../../../glossary.md#term-opaque); [lowering glossary](../../../../glossary.md#lowering); [specialization glossary](../../../../glossary.md#term-specialization); [run glossary](../../../../glossary.md#term-run))</span>
- Exact buffer/workspace requirements are returned before shared assignment. Finalization must
  use the same declarations and cannot change the selected mode, carrier, specialization,
  parallel configuration, route, or resource set.<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- CPU 0003 artifact lookup/generation/definition occurs only in backend finalization after the
  complete assignment mapping succeeds. Runtime performs none of that work.
- Finalization constructs immutable Java recipe state only. It borrows the externally owned
  worker group and acquires no closeable resource. The generated artifact has no close lifecycle
  and is retained strongly by the prepared executable.<span data-glossary-navigation="true"> ([close lifecycle glossary](../../../../glossary.md#ownership-borrowing-and-lease); [prepared executable glossary](../../../../glossary.md#prepared-executable--preparedexecutable))</span>
- `PreparedExecutable.bind(RunState)` remains the only [heterogeneous](../../../../glossary.md#term-heterogeneous) compatibility boundary.
  `CpuBorrowedBuffer` and `CpuNativeBuffer` remain Runtime-visible nominal representations until
  the CPU cold boundary classifies them to exact <span data-glossary-navigation="true"> ([nominal glossary](../../../../glossary.md#term-nominal))</span>`CpuBufferArgument` values.
- A family-owned typed binder is required because the generated entry `MethodType` varies with
  exact carriers, dynamic offsets/extents, and execution mode. One generic `Object[]` or adapted
  handle invoker would violate the direct hot-path contract.
- The [bound invocation](../../../../glossary.md#bound-invocation--boundinvocation) retains the exact [`RunState`](../../../../glossary.md#run-state--runstate) through [`BoundInvocation`](../../../../glossary.md#bound-invocation--boundinvocation), performs only its
  final state-open guard, then invokes direct typed fields. Runtime execution performs no route,
  cache, disk, hash, checksum, class-file, lookup, reflection, storage discovery, heap-base
  discovery, argument classification, graph inspection, or slot lookup.<span data-glossary-navigation="true"> ([guard glossary](../../../../glossary.md#term-guard); [hash glossary](../../../../glossary.md#term-hash); [checksum glossary](../../../../glossary.md#term-checksum); [heap glossary](../../../../glossary.md#term-heap); [argument glossary](../../../../glossary.md#term-argument))</span>
- No public/shared contract, dependency direction, module boundary, Gradle rule, or architecture
  decision changes. Stop if implementation needs one.<span data-glossary-navigation="true"> ([dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [module glossary](../../../../glossary.md#term-module); [Gradle glossary](../../../../glossary.md#term-gradle))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu` — unchanged [fail-closed](../../../../glossary.md#term-fail-closed) public capability [identity](../../../../glossary.md#term-identity-meanings).
- `io.github.pho001.synaptik.backend.cpu.execution` — extended with the package-private typed
  analysis/finalization/recipe foundation beside the completed representation, worker,
  specialization, generator, and artifact machinery.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact))</span>

Packages added or changed:

- No Java package is added. All production types remain package-private in `execution`; this lets
  them collaborate with `CpuBorrowedBuffer`, `CpuBufferArgument`, [`CpuWorkerGroup`](../../../../glossary.md#cpu-worker-group--cpuworkergroup),
  `CpuKernelSpecialization`, and `CpuGeneratedKernelArtifactStore` without widening completed
  internals or creating a facade.<span data-glossary-navigation="true"> ([widening glossary](../../../../glossary.md#term-widening); [facade glossary](../../../../glossary.md#term-facade))</span>

Type placement:

- `CpuPortableAnalysisInputs` — exact [immutable](../../../../glossary.md#term-immutable) [CPU](../../../../glossary.md#term-cpu) [target](../../../../glossary.md#term-target)/parallel inputs implementing
  `BackendAnalysisInputs`.
- `CpuPreparedParallelConfiguration` — selected primitive parallel recipe, not a worker [owner](../../../../glossary.md#term-owner).
- `CpuPortableCandidateSource` — one direct typed candidate-generation collaboration; no
  registry or discovery.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry))</span>
- `CpuPortableKernelCandidate` — complete typed candidate, declarations, use mapping, emitter,
  specialization, and binder.
- `CpuPortableInvocationBinder` — cold family-owned construction seam for a signature-specific
  direct <span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature))</span>[`BoundInvocation`](../../../../glossary.md#bound-invocation--boundinvocation).
- `CpuPortablePreparationPlan` — immutable selected [opaque](../../../../glossary.md#term-opaque) plan implementing
  `BackendPreparationPlan`.
- `CpuPortablePartitionPreparer` — small analysis/filter/selection owner implementing the current
  [`BackendPartitionPreparer`](../../../../glossary.md#backend-partition-preparer--backendpartitionpreparer) contract.
- `CpuPortablePartitionFinalizer` — small assignment/artifact/executable owner implementing the
  current `BackendPartitionFinalizer` contract.
- `CpuPortablePreparedExecutable` — immutable CPU [Runtime](../../../../glossary.md#term-runtime) recipe and cold [specialization](../../../../glossary.md#term-specialization) check.

## Exact package-private surface

The task must implement the following package-private source-level surface. Explicit compact
constructors/accessors may be added only to validate, [snapshot](../../../../glossary.md#term-snapshot), or document these exact members.
A [signature](../../../../glossary.md#term-signature) or package change requires updating this task before implementation continues.

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
This explicit direct-handle field makes the [Runtime](../../../../glossary.md#term-runtime) recipe's hot invocation [dependency](../../../../glossary.md#term-dependency) visible;
the retained generated [kernel](../../../../glossary.md#kernel) supplies the strong hidden-class/[lookup](../../../../glossary.md#term-method-handle-lookup)/bytes [lifetime](../../../../glossary.md#term-lifetime).

No production `CpuPortableBoundInvocation` is added in this foundation. Generated signatures are
not uniform, so each later family must implement its own typed binder/invocation beside its
semantics. Task tests use nested synthetic signature-specific implementations and do not create a
generic production invoker.

## Analysis, selection, and resource contract

- `CpuPreparedParallelConfiguration` validates positive worker count, then positive minimum range
  size. It contains no executor, worker, route, target discovery, or ownership lifecycle.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- `CpuPortableAnalysisInputs` snapshots its supported Vector shapes in supplied order, rejects
  null/duplicate entries, and retains the exact non-null parallel configuration. An empty species
  list permits scalar candidates only.<span data-glossary-navigation="true"> ([Vector glossary](../../../../glossary.md#term-vector); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Candidate construction validates specialization, emitter, requirements, buffer uses,
  workspace uses, and binder in declaration order and snapshots every list.
- The emitter [lowering](../../../../glossary.md#lowering) [fingerprint](../../../../glossary.md#term-fingerprint) must exactly equal the [specialization](../../../../glossary.md#term-specialization) lowering fingerprint.
- [Buffer](../../../../glossary.md#term-buffer) uses align one-for-one and in order with `specialization.arguments()`. Each use references
  by exact object identity one declared <span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-reference-identity))</span>`PreparationResourceRequirement.Buffer`; its projected
  value's [`DataType`](../../../../glossary.md#data-type--datatype) equals the aligned specialization [argument](../../../../glossary.md#term-argument) type. The argument's [carrier](../../../../glossary.md#term-storage-carrier) and
  `BufferAccess` are the complete selected representation/access facts.
- Workspace uses reference by exact object identity declared workspace requirements. Repeated
  uses may address one declaration, but every declared buffer and workspace requirement must be
  used at least once and no use may name an undeclared requirement.
- Requirement declarations remain unique under the existing
  [`BackendPartitionAnalysis`](../../../../glossary.md#backend-partition-analysis-backendpartitionanalysis) contract. Candidate construction performs no allocation, slot
  assignment, artifact lookup, class generation, or invocation.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- Analysis validates `context` first, then exact CPU ownership, candidate-source result and
  entries, candidate-to-context resource/value agreement, and mode eligibility in source order.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- A scalar mode is eligible without a Vector species. A Vector mode is eligible only when its
  exact <span data-glossary-navigation="true"> ([Vector species glossary](../../../../glossary.md#term-vector-species))</span>`VectorShape` occurs in `supportedVectorShapes`. A parallel mode is eligible only when
  the supplied configuration has the positive worker capacity already established by its
  constructor. No mode is inferred from a carrier or installed JDK module.<span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [module glossary](../../../../glossary.md#term-module))</span>
- The first fully valid eligible candidate is selected. Source order is the family owner's
  deterministic safe-heuristic order; this foundation adds no numeric score or universal route
  priority. An invalid candidate fails the analysis rather than being silently skipped. A valid
  but target-ineligible candidate may be skipped. No eligible candidate fails with<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [heuristic glossary](../../../../glossary.md#term-heuristic))</span>
  `IllegalArgumentException("no supported CPU portable candidate")`.
- The returned analysis retains `context.partition()` by exact reference, wraps the selected
  candidate and exact parallel configuration in one immutable plan, and returns the candidate's
  exact ordered requirement references. Analysis does not mutate the source list or candidate.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>

## Finalization, [binding](../../../../glossary.md#term-binding), and hot-path contract

- `CpuPortablePartitionFinalizer` validates its explicit `artifactRoot`, then the borrowed open
  worker group. Construction may create the CPU-0003 store identity because that constructor does
  no filesystem access; only <span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>`finalizePartition` may call `loadOrGenerate`.
- `backendId()` returns `CpuCapabilityProvider.CPU_BACKEND_ID` by exact reference.
- Existing `BackendPartitionFinalization` construction remains authoritative for exact analysis,
  assignment order, source identity, plan position, slot identity, and geometry validation.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- The CPU finalizer first validates that the analysis partition remains CPU-owned, then maps every
  selected buffer/workspace use from its exact requirement reference to the already-validated
  assignment. It constructs all Runtime <span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [buffer glossary](../../../../glossary.md#term-buffer))</span>`BufferSelection`, `WorkspaceSelection`, `BufferAccess`,
  and [`DataType`](../../../../glossary.md#data-type--datatype) lists before filesystem or generator work.
- The retained worker group's count must equal the selected prepared parallel configuration. The
  group must be open before artifact work. The finalizer borrows it and never closes it.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact))</span>
- Only after those checks does finalization call
  `CpuGeneratedKernelArtifactStore.loadOrGenerate(specialization, familyEmitter)` exactly once.
  It neither changes the specialization nor retries outside the completed store contract.<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization))</span>
- The resulting executable retains the exact shared [`PreparedMemoryPlan`](../../../../glossary.md#prepared-memory-plan--preparedmemoryplan) reference and exact
  generated artifact. It owns no closeable resource and performs no allocation, transfer,
  materialization, publication, or schedule assembly.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [publication glossary](../../../../glossary.md#term-publication))</span>
- [Runtime](../../../../glossary.md#term-runtime)'s final `PreparedExecutable.bind` performs plan/open/range lookup and CPU 0001's base
  performs representation, lifetime, size, access, data-type, and alignment checks first.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup); [lifetime glossary](../../../../glossary.md#term-lifetime); [alignment glossary](../../../../glossary.md#term-alignment))</span>
- `CpuPortablePreparedExecutable.bindCpu` then validates buffer argument count and exact variant
  against each specialization carrier in order. Dynamic primitive-array offsets remain the
  already-classified carrier-relative offsets; exact-segment arguments require the exact<span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
  `Segment` form and zero relative offset. No conversion or copy is permitted.
- For parallel modes, every `CpuBufferArgument.Segment` and workspace segment must satisfy
  `CpuWorkerGroup.isAccessibleByEveryWorker(...)` before the binder is called. Single-thread modes
  require current-thread accessibility already enforced by CPU 0001.
- After all checks, `bindCpu` calls the selected binder once with fresh arrays. The returned
  invocation must be non-null, retain the exact supplied [`RunState`](../../../../glossary.md#run-state--runstate), direct handle, direct typed
  carrier/segment fields, and direct worker/workspace fields that its signature needs. It must
  not retain either array, a nominal representation, a slot-selection path, or<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [nominal glossary](../../../../glossary.md#term-nominal))</span>
  `HostTensorStorage`.
- Synthetic tests inspect the bound invocation's direct fields and Class-File model to prove its
  hot call contains no route/cache/disk/hash/checksum/class-definition/lookup/reflection/storage-
  discovery/argument-classification/slot-lookup path. This recurring invariant belongs in tests,
  not repeated manual <span data-glossary-navigation="true"> ([bound invocation glossary](../../../../glossary.md#bound-invocation--boundinvocation); [model glossary](../../../../glossary.md#term-model); [cache glossary](../../../../glossary.md#term-cache); [hash glossary](../../../../glossary.md#term-hash); [checksum glossary](../../../../glossary.md#term-checksum); [invariant glossary](../../../../glossary.md#term-invariant))</span>`javap` checks.
- `CpuBorrowedBuffer` remains the sole [CPU](../../../../glossary.md#term-cpu) bridge from caller-owned `HostTensorStorage` into
  Runtime's representation lifecycle. It is not an artifact cache, tuning cache, generated
  artifact, executable, or ownership transfer. Its task-0001 no-op close and caller lifetime
  remain unchanged.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>

## [Validation](../../../../glossary.md#term-validation) and failure order

Automated tests must lock this order:

1. parallel configuration: worker count, minimum range size;
2. analysis inputs: supported-shape list, indexed entries/duplicates, parallel configuration;
3. candidate: specialization, emitter, lowering fingerprint match, requirements, uses, exact
   declaration-use association, binder;<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization); [lowering glossary](../../../../glossary.md#lowering); [fingerprint glossary](../../../../glossary.md#term-fingerprint))</span>
4. preparer: context, CPU owner, candidate-source result, candidates in encounter order,
   context/resource agreement, target/mode eligibility, then first eligible selection;<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [target glossary](../../../../glossary.md#term-target))</span>
5. shared Prepare's existing finalization validation: analysis, plan, assignments, exact
   requirement association, plan positions/slots, geometry;
6. CPU finalizer: CPU owner, complete use-to-assignment mapping, worker-count/open state, then one
   artifact-store call and executable construction;
7. Runtime/CPU cold binding: Runtime plan/open/selection checks, CPU 0001 representation checks,
   specialization carrier checks, parallel worker accessibility, then one typed binder call; and<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
8. bound execution: the existing state-open [guard](../../../../glossary.md#term-guard), then direct [signature](../../../../glossary.md#term-signature)-specific handle call.

Stable [CPU](../../../../glossary.md#term-cpu) failures must identify indexed candidate/resource/use positions. Existing [Prepare](../../../../glossary.md#prepare) and
[Runtime](../../../../glossary.md#term-runtime) messages remain unchanged. [Artifact](../../../../glossary.md#term-artifact) failures retain CPU 0003's exact contextual causes;
the finalizer adds no fallback or wrapper that hides them.

## Affected files

Expected [CPU](../../../../glossary.md#term-cpu) production paths:

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

Expected [planning](../../../../glossary.md#term-planning) paths:

- this task
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

Review only, with a reasoned no-change conclusion unless implementation finds a concrete
contradiction:

- `ARCHITECTURE.md`, focused architecture pages, ADRs 0010/0011, and architecture tests;
- completed CPU 0001–0003 production, tests, Javadocs, task records, and CPU [Gradle](../../../../glossary.md#term-gradle) configuration;
- current `BackendAnalysisInputs`, [`BackendPartitionPreparer`](../../../../glossary.md#backend-partition-preparer--backendpartitionpreparer), [`BackendPartitionAnalysis`](../../../../glossary.md#backend-partition-analysis-backendpartitionanalysis),
  `PreparationResourceRequirement`, `BackendPartitionFinalizer`,
  `BackendPartitionFinalization`, `PreparationResourceAssignment`, and [`PreparedPartition`](../../../../glossary.md#prepared-partition--preparedpartition);
- current [Runtime](../../../../glossary.md#term-runtime) [`PreparedMemoryPlan`](../../../../glossary.md#prepared-memory-plan--preparedmemoryplan), [`PreparedRepresentationPlan`](../../../../glossary.md#prepared-representation-plan--preparedrepresentationplan), [`PreparedExecutable`](../../../../glossary.md#prepared-executable--preparedexecutable),
  [`BoundInvocation`](../../../../glossary.md#bound-invocation--boundinvocation), `BufferRepresentation`, `WorkspaceRepresentation`, [`RunState`](../../../../glossary.md#run-state--runstate), and runner;
- [Model](../../../../glossary.md#term-model) `HostTensorStorage`, projected graph/descriptor facts, capability contracts, public API
  documentation, glossary terminology, backend-conformance/integration placeholders, Config,
  Engine, other backends, extensions, and tools.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [descriptor glossary](../../../../glossary.md#term-descriptor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine))</span>

## Maximum [scope](../../../../glossary.md#term-scope)

At most 18 paths:

- 10 [CPU](../../../../glossary.md#term-cpu) production paths;
- 3 CPU test paths;
- 2 explanatory documentation paths; and
- 3 [planning](../../../../glossary.md#term-planning) paths.

No existing CPU production path other than `execution/package-info.java` may change. No existing
test, build, shared-[module](../../../../glossary.md#term-module), architecture, [ADR](../../../../glossary.md#term-architecture-decision-record-adr), [conformance](../../../../glossary.md#term-conformance), integration, public [API](../../../../glossary.md#term-application-programming-interface-api), other [backend](../../../../glossary.md#backend),
or later task-specification path may change. If implementation needs another path, a public
composition [owner](../../../../glossary.md#term-owner), a change to `CpuPreparedExecutable`, a shared contract extension, or a generic
invocation adapter, stop and update planning rather than expanding silently.

## Acceptance criteria

- [x] The exact package-private surface and package placement above are implemented with no new
      public CPU type or package.
- [x] [CPU](../../../../glossary.md#term-cpu) analysis implements the existing typed [`BackendPartitionPreparer`](../../../../glossary.md#backend-partition-preparer--backendpartitionpreparer) contract, accepts
      only an already CPU-owned exact partition, retains that partition, and exposes no Compiler
      aggregate or second graph model.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [Compiler glossary](../../../../glossary.md#term-compiler); [graph glossary](../../../../glossary.md#graph))</span>
- [x] Exact immutable CPU inputs and parallel configuration are validated and deeply snapshotted;
      no Config API, discovery, default, global state, or mutable target observation is added.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [mutable glossary](../../../../glossary.md#term-mutable); [target glossary](../../../../glossary.md#term-target))</span>
- [x] The candidate source is one direct typed collaboration, not a registry, service locator,
      reflection mechanism, generic parameter bag, or <span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [service locator glossary](../../../../glossary.md#term-service-locator); [parameter glossary](../../../../glossary.md#term-parameter))</span>`Map<String, ?>`.
- [x] Responsibilities remain split among immutable candidate/plan values, the narrow
      analysis/filter/selection preparer, the assignment/artifact/executable finalizer, and the
      cold executable/binder boundary; no god analyzer, finalizer, manager, or broad facade is
      introduced.<span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade))</span>
- [x] Complete candidates align exact resource declarations and uses with the exact
      `CpuKernelSpecialization` arguments, emitter fingerprint, typed carriers, accesses, and
      binder, with no undeclared or unused requirement.<span data-glossary-navigation="true"> ([fingerprint glossary](../../../../glossary.md#term-fingerprint))</span>
- [x] Analysis hard-filters owner, context/resource, Vector-species, execution-mode, and parallel
      compatibility in the specified order, selects the first valid eligible candidate
      deterministically, and fails closed when none exists.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [Vector glossary](../../../../glossary.md#term-vector))</span>
- [x] Every selected shared buffer/workspace requirement is exact and returned before slot
      assignment; analysis performs no artifact, generation, allocation, executable, Runtime,
      measurement, tuning, or cache-mutation work.<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- [x] Finalization implements only the existing `BackendPartitionFinalizer` handoff, maps all
      assignments before artifact access, and neither changes selection nor adds a requirement.<span data-glossary-navigation="true"> ([handoff glossary](../../../../glossary.md#term-handoff))</span>
- [x] The explicit artifact root is caller-supplied to the CPU-private finalizer. No default root,
      home/environment lookup, public Config field, Engine facade, or filesystem access outside
      finalization is introduced.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup); [Engine glossary](../../../../glossary.md#term-engine))</span>
- [x] CPU 0003's store is called exactly once only after assignment/worker validation; its format,
      key, validation, publication, concurrency, weak retention, security, and failure behavior
      remain unchanged.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [publication glossary](../../../../glossary.md#term-publication); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>
- [x] The immutable prepared executable retains the exact shared memory plan, selected resource
      recipes, <span data-glossary-navigation="true"> ([prepared executable glossary](../../../../glossary.md#prepared-executable--preparedexecutable))</span>`CpuGeneratedKernel`, direct entry handle, specialization, parallel configuration,
      worker reference, and typed binder, while owning no closeable physical resource.<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization))</span>
- [x] `CpuBorrowedBuffer` remains the non-owning `HostTensorStorage` representation/lifetime
      boundary and is consumed through Runtime <span data-glossary-navigation="true"> ([lifetime glossary](../../../../glossary.md#term-lifetime))</span>`BufferRepresentation` cold [binding](../../../../glossary.md#term-binding). No direct
      `HostTensorStorage` [argument](../../../../glossary.md#term-argument) binding or [artifact](../../../../glossary.md#term-artifact)-[cache](../../../../glossary.md#term-cache) meaning is introduced.
- [x] Cold binding checks exact heap/segment/mixed carrier compatibility and parallel worker
      accessibility once, then returns a family-owned typed <span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>[`BoundInvocation`](../../../../glossary.md#bound-invocation--boundinvocation) retaining direct
      fields rather than fresh arrays, nominal representations, storage, or slot lookups.<span data-glossary-navigation="true"> ([nominal glossary](../../../../glossary.md#term-nominal))</span>
- [x] Synthetic direct invocations exercise all four portable execution modes and call exact
      handles without reflection, adapters, `invokeWithArguments`, generic objects, or argument
      classification in `executeBound()`.
- [x] Runtime hot execution contains no route, cache, disk, hash, checksum, class-file parsing,
      class definition, lookup, reflection, storage discovery, heap-base inspection, argument
      classification, graph inspection, backend discovery, slot lookup, tuning, or allocation.<span data-glossary-navigation="true"> ([hash glossary](../../../../glossary.md#term-hash); [checksum glossary](../../../../glossary.md#term-checksum); [backend glossary](../../../../glossary.md#backend))</span>
- [x] Tests are bounded and synthetic. They establish lifecycle mechanics only and make no Model
      numerical result, operation-family coverage, capability, conformance, or integration claim.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [operation glossary](../../../../glossary.md#operation); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- [x] `CpuCapabilityProvider` and all completed CPU 0001–0003 source/contracts remain unchanged;
      support remains false for every non-null query.
- [x] CPU 0001–0003 remain `Complete`; CPU 0004 is synchronized through implementation; CPU
      0005–0016 remain `Draft` without detailed specifications; future [Model](../../../../glossary.md#term-model) 0026 remains present.
- [x] Focused development tests and one final `:backends:cpu:test` run after executable Java
      stability pass without an installed native provider or external artifact root.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
- [x] New/changed production declarations have complete Javadoc for purpose, immutability,
      ownership, lifetime, threading, inputs, returns, nullability, failures, and hot/cold
      boundaries.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [nullability glossary](../../../../glossary.md#term-nullability))</span>
- [x] A separate clean documentation-focused pass finalizes affected Javadocs, CPU guidance,
      glossary impact, planning evidence, links, and status in the same overall change, reusing
      Java evidence unless it changes executable behavior or records a concrete stale-evidence
      risk.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- [x] CPU Javadoc, generated pages, Markdown links/anchors/fences/newlines, exact 18-path scope,
      surface/mechanism/status/later-specification/excluded-path checks, trailing whitespace, and<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>
      `git diff --check` pass.

## Tests / [validation](../../../../glossary.md#term-validation)

Implementation development may [run](../../../../glossary.md#term-run) focused classes:

```bash
./gradlew :backends:cpu:test --tests '*CpuPortablePartitionPreparerTest'
./gradlew :backends:cpu:test --tests '*CpuPortablePartitionFinalizerTest' --tests '*CpuPortablePreparedExecutableTest'
```

The focused [matrix](../../../../glossary.md#term-matrix-meanings) must cover:

- constructor nulls, snapshots, duplicates, stable failure order/messages, and exact package-
  private/final surface;
- wrong-[owner](../../../../glossary.md#term-owner) rejection before candidate-source invocation;
- empty, null, malformed, [target](../../../../glossary.md#term-target)-ineligible, first-valid, all-four-mode, and concurrent analysis;
- exact projected values, requirements, repeated uses, representation indices, carrier/access/
  type alignment, Vector species, and parallel configuration;<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [alignment glossary](../../../../glossary.md#term-alignment); [Vector species glossary](../../../../glossary.md#term-vector-species))</span>
- proof that analysis performs no filesystem, artifact, generation, definition, allocation,
  binding, worker dispatch, measurement, or cache mutation;<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [dispatch glossary](../../../../glossary.md#term-dispatch); [cache glossary](../../../../glossary.md#term-cache))</span>
- complete assignment mapping before store use; missing/mismatched assignment failure without
  artifact access; valid hit/miss and strong generated-artifact retention;
- [heap](../../../../glossary.md#term-heap) array, read-only exact heap segment, native exact segment, and mixed cold [binding](../../../../glossary.md#term-binding) through
  `CpuBorrowedBuffer`/`CpuNativeBuffer`, including offset, access, liveness, alignment, and worker
  accessibility failures;<span data-glossary-navigation="true"> ([liveness glossary](../../../../glossary.md#term-liveness))</span>
- exact direct handle identity and signature-specific synthetic bound fields, repeated and
  concurrent binds to distinct <span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [signature glossary](../../../../glossary.md#term-signature))</span>[`RunState`](../../../../glossary.md#run-state--runstate) instances, closed-state rejection, and all four mode
  executions; and
- source/Class-File assertions excluding public surface, `HostTensorStorage` direct binding,
  Runtime hot-path discovery, reflection/adaptation, generic objects, registries, service
  locators, tuning, vendor routes, and operation/capability claims.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [operation glossary](../../../../glossary.md#operation))</span>

After executable Java stabilizes, run one final [CPU](../../../../glossary.md#term-cpu) [module](../../../../glossary.md#term-module) test exactly once:

```bash
./gradlew :backends:cpu:test
```

Implementation pass also runs:

```bash
git diff --check
```

After the implementation pass records the exact final CPU test result, hand the diff and evidence
to a separate clean documentation-focused agent/thread. That pass follows
`docs/developer-guide/documentation-rules.md`, the General, [API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc), [Backend](../../../../glossary.md#backend) Guide, [Planning](../../../../glossary.md#term-planning),
and Example profiles as applicable, and runs after final Javadoc edits:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

The documentation pass must validate repository-local Markdown links and heading anchors for
this task, CPU master plan, roadmap, CPU guide, and glossary; balanced fences, final newlines, and
trailing whitespace; generated pages for every new/changed production declaration; exact 18-path
[scope](../../../../glossary.md#term-scope); package-private surface; CPU 0001–0004 [synchronization](../../../../glossary.md#term-synchronization); CPU 0005–0016 `Draft` rows and
absence of their detailed specs; preserved [Model](../../../../glossary.md#term-model) 0026; and unchanged excluded paths. It reuses the
successful Java evidence and does not rerun Java tests unless it changes executable Java behavior
or records a concrete stale-evidence risk.

Repository-wide Java, architecture, backend-[conformance](../../../../glossary.md#term-conformance), and integration validation remain
deferred to CPU 0009 or [continuous integration](../../../../glossary.md#term-continuous-integration-ci) because this task changes one backend-private
foundation and no [dependency](../../../../glossary.md#term-dependency)/architecture boundary. If implementation changes a shared contract,
dependency, build rule, public capability, or another module, stop instead of broadening the
validation tier.

## Dependencies

- Complete CPU 0001 capability, non-owning borrowed representation, native representations,
  typed buffer arguments, CPU prepared-executable base, direct binding contract, and worker group.<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer); [arguments glossary](../../../../glossary.md#term-argument); [binding glossary](../../../../glossary.md#term-binding))</span>
- Complete CPU 0002 exact specialization, four modes, family emitter, deterministic verified
  bytes, direct typed entry handle, and generated-artifact lifetime.<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization); [artifact glossary](../../../../glossary.md#term-artifact); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- Complete CPU 0003 explicit-root artifact store, cold loading, atomic persistence, weak interning,
  and strong-caller lifetime contract.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [persistence glossary](../../../../glossary.md#term-persistence); [weak interning glossary](../../../../glossary.md#weak-reference-and-weak-interning))</span>
- Complete [Prepare](../../../../glossary.md#prepare) analysis/resource/finalization/orchestration contracts, especially
  `BackendAnalysisInputs`, [`BackendPartitionPreparer`](../../../../glossary.md#backend-partition-preparer--backendpartitionpreparer), [`BackendPartitionAnalysis`](../../../../glossary.md#backend-partition-analysis-backendpartitionanalysis),
  `BackendPartitionFinalizer`, and `BackendPartitionFinalization`.
- Complete [Runtime](../../../../glossary.md#term-runtime) prepared memory, representation, [`PreparedExecutable`](../../../../glossary.md#prepared-executable--preparedexecutable), [`BoundInvocation`](../../../../glossary.md#bound-invocation--boundinvocation),
  [`RunState`](../../../../glossary.md#run-state--runstate), schedule, and runner contracts.
- Current [Model](../../../../glossary.md#term-model) projected static [graph](../../../../glossary.md#graph)/[descriptor](../../../../glossary.md#term-descriptor) facts and [borrowed](../../../../glossary.md#ownership-borrowing-and-lease) `HostTensorStorage`; this task
  changes neither.
- Java 26 Class-File and [Vector](../../../../glossary.md#term-vector) APIs already configured in the [CPU](../../../../glossary.md#term-cpu) [module](../../../../glossary.md#term-module).

## Follow-up tasks

- CPU 0005 remains Draft and is the first owner of real portable elementwise/pointwise candidate
  generation, family emitters, signature-specific typed binders/invocations, numerical tests, and
  truthful capability additions.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [owner glossary](../../../../glossary.md#term-owner); [elementwise glossary](../../../../glossary.md#term-elementwise); [pointwise glossary](../../../../glossary.md#term-pointwise); [signature glossary](../../../../glossary.md#term-signature))</span>
- CPU 0006–0008 remain Draft for later portable families; CPU 0009 remains the portable generated-
  coverage/capability/conformance checkpoint.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- CPU 0010–0015 remain Draft optional native-route work. CPU 0016 remains Draft compatible tuning-
  evidence selection and does not replace the generated-class artifact store.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact))</span>
- A later Engine/CPU composition task must own construction/lifetime of the worker group and
  explicit trusted artifact root before public end-to-end use. This task deliberately adds no
  default or public facade.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [lifetime glossary](../../../../glossary.md#term-lifetime); [facade glossary](../../../../glossary.md#term-facade))</span>
- Future Model 0026 remains Draft and is required before any FLOAT16 or mixed-precision backend
  claim.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [FLOAT16 glossary](../../../../glossary.md#term-float16); [precision glossary](../../../../glossary.md#term-precision); [backend glossary](../../../../glossary.md#backend))</span>

## Architecture impact

Expected impact: None.

This task implements the existing concrete-[backend](../../../../glossary.md#backend) analysis/finalization, exact-resource,
prepared-recipe, per-[run](../../../../glossary.md#term-run) representation, and cold-[binding](../../../../glossary.md#term-binding) responsibilities. It changes no shared
contract, [dependency edge](../../../../glossary.md#term-dependency-edge), [lifecycle](../../../../glossary.md#lifecycle) stage, or authority. If implementation requires an
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

- Preserve `CpuBorrowedBuffer` exactly as [CPU](../../../../glossary.md#term-cpu) 0001's non-owning representation of caller-owned
  `HostTensorStorage`. It is the required Model-storage-to-Runtime-representation boundary used
  for caller inputs and later cold <span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Runtime glossary](../../../../glossary.md#term-runtime))</span>`CpuBufferArgument` classification. Direct
  `HostTensorStorage`-to-argument binding was rejected because it would reopen completed CPU 0001,
  bypass Runtime's <span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument); [binding glossary](../../../../glossary.md#term-binding))</span>`BufferRepresentation` lifecycle/ownership model, and erase the checked cold
  boundary. <span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>`CpuBorrowedBuffer` is not an artifact cache; its first production portable execution
  use begins in this task.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [cache glossary](../../../../glossary.md#term-cache))</span>
- Keep all new types package-private in `execution`. The current collaboration needs direct access
  to completed CPU internals and does not justify a public backend facade or cross-package API.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [facade glossary](../../../../glossary.md#term-facade); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Represent typed portable storage selection with the existing
  `CpuKernelSpecialization.Argument` carrier/access vocabulary rather than adding a duplicate
  storage-kind enum.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Use one injected typed `CpuPortableCandidateSource` because CPU 0005+ family owners need a
  direct seam and CPU 0004 must test selection without implementing operation semantics. A
  registry, service locator, reflection, and generic parameter bag were rejected.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [registry glossary](../../../../glossary.md#term-registry); [service locator glossary](../../../../glossary.md#term-service-locator); [parameter glossary](../../../../glossary.md#term-parameter))</span>
- Select the first fully valid target-eligible candidate in deterministic source order. This is a
  bounded safe-heuristic seam, not a broad cost model or universal route priority; later family
  tasks own their candidate order.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [heuristic glossary](../../../../glossary.md#term-heuristic); [cost model glossary](../../../../glossary.md#compiler-transformations-and-dominance))</span>
- Keep invocation construction family-owned. Generated `MethodType` values vary by carrier,
  offset/extent baking, and mode, so a generic production invocation would require forbidden
  object arrays, adaptation, or hot argument switching.<span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent))</span>
- Supply the artifact root and worker group explicitly to the CPU-private finalizer. The store is
  used only after assignment during finalization, and the worker group is borrowed rather than
  allocated or closed by the prepared recipe. Public composition and defaults remain later work.<span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease))</span>
- Strongly retain both `CpuGeneratedKernel` and its exact direct handle in the immutable prepared
  executable. The artifact owns hidden-class lifetime; the direct field makes the hot dependency
  explicit without another lookup.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-method-handle-lookup))</span><span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [prepared executable glossary](../../../../glossary.md#prepared-executable--preparedexecutable); [lifetime glossary](../../../../glossary.md#term-lifetime); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Preserve the backend-declared resource geometry exactly during CPU analysis. Projected value
  identity and data type constrain a candidate, but task 0004 does not derive dense byte size from<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [data type glossary](../../../../glossary.md#data-type--datatype))</span>
  `TensorDescriptor`, infer a [layout](../../../../glossary.md#layout), or choose [materialization](../../../../glossary.md#materialization). The existing shared
  `BackendPartitionFinalization` contract remains the sole validation boundary between each exact
  declaration and its assigned <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>[`PreparedMemoryPlan`](../../../../glossary.md#prepared-memory-plan--preparedmemoryplan) geometry.

## Known limitations

- [CPU](../../../../glossary.md#term-cpu) still advertises and truthfully executes no [Model](../../../../glossary.md#term-model) [operation](../../../../glossary.md#operation) after this foundation task.
- Only bounded synthetic sources, emitters, binders, and invocations prove the lifecycle. They
  are test code and do not establish a numerical result or operation-family capability.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- No production family candidate source exists until CPU 0005. Consequently no public or Engine-
  composed end-to-end CPU execution exists at this frontier.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span>
- The explicit artifact root and worker group have no public composition owner yet. This task
  defines only the CPU-private constructor seam and borrows the worker lifetime.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [owner glossary](../../../../glossary.md#term-owner); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- The foundation selects among portable generated candidates only. Native providers, tuning
  evidence, transition-cost comparison, packing, materialization, and representation-plan
  reconciliation remain later work.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization))</span>
- Prepare still requires fully static Shapes. The generated specialization may carry dynamic
  invocation offsets/extents, but this task adds no new run-dynamic Shape contract.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [Shapes glossary](../../../../glossary.md#shape); [specialization glossary](../../../../glossary.md#term-specialization); [carry glossary](../../../../glossary.md#term-carry); [run glossary](../../../../glossary.md#term-run))</span>
- Generated signature-specific binders remain family-owned, so later family tasks must test every
  signature they introduce without weakening the direct hot-path rule.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature))</span>
- FLOAT16 and mixed precision remain unavailable pending Model 0026 and later exact backend-route
  evidence.<span data-glossary-navigation="true"> ([FLOAT16 glossary](../../../../glossary.md#term-float16); [mixed precision glossary](../../../../glossary.md#term-mixed-precision); [backend glossary](../../../../glossary.md#backend))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names): `./gradlew :backends:cpu:compileJava` and
  `./gradlew :backends:cpu:compileTestJava` passed before the final executable runs.
- Final focused implementation evidence:
  `./gradlew :backends:cpu:test --tests io.github.pho001.synaptik.backend.cpu.execution.CpuPortablePartitionPreparerTest --tests io.github.pho001.synaptik.backend.cpu.execution.CpuPortablePartitionFinalizerTest --tests io.github.pho001.synaptik.backend.cpu.execution.CpuPortablePreparedExecutableTest`
  passed with `BUILD SUCCESSFUL`: three suites, 24 tests (6 + 7 + 11), and zero failures,
  errors, or skips. The expanded matrix directly covers exact surface and validation order,
  immutable snapshots, deterministic all-mode selection, the opaque 12-byte declaration against
  projected FLOAT32 <span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings); [immutable glossary](../../../../glossary.md#term-immutable); [opaque glossary](../../../../glossary.md#term-opaque); [FLOAT32 glossary](../../../../glossary.md#term-float32))</span>`Shape[4]`, analysis side-effect exclusion, assignment-before-artifact
  failures, artifact miss/reuse and strong retention, borrowed/native heap/segment/mixed binding,
  repeated resources, dynamic/baked offsets, worker accessibility, concurrent binds, exact handle
  calls, binder result association, and source/Class-File forbidden-mechanism checks.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [heap glossary](../../../../glossary.md#term-heap); [binding glossary](../../../../glossary.md#term-binding))</span>
- Final implementation [module](../../../../glossary.md#term-module) evidence: the sole stabilized
  `./gradlew :backends:cpu:test` [run](../../../../glossary.md#term-run) passed with `BUILD SUCCESSFUL`, 13 suites, 72 tests, zero
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
  final Java evidence and did not rerun Java tests.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc); [Backend glossary](../../../../glossary.md#backend); [Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Model glossary](../../../../glossary.md#term-model); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- Documentation validation: `./gradlew :backends:cpu:javadoc` passed after final Javadoc edits
  with `BUILD SUCCESSFUL` and 11 actionable tasks (two executed and nine up-to-date); the expected
  Java Vector API incubator warnings and pre-existing task-0001 documentation warnings were
  non-failing. Generated declaration pages were present for all nine new package-private types,
  their nested records, and the changed execution package page.<span data-glossary-navigation="true"> ([Vector API glossary](../../../../glossary.md#term-vector-api))</span>
- Repository-local Markdown links and heading anchors across this task, CPU master plan, roadmap,
  CPU guide, and glossary passed. The same five paths passed balanced-fence and final-newline
  checks; all 18 changed paths passed trailing-whitespace checks. `git diff --check` passed on the
  final combined diff.
- Exact-scope validation found exactly the authorized 18 paths: ten CPU production paths, three
  CPU tests, two explanatory documentation paths, and three planning paths. Source/reflection and
  generated-page checks confirmed every new type remains package-private, the exact task surface
  is present, <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>`CpuBorrowedBuffer` remains the `BufferRepresentation` [lifetime](../../../../glossary.md#term-lifetime) bridge, and
  `CpuCapabilityProvider` remains unchanged and [fail-closed](../../../../glossary.md#term-fail-closed).
- [CPU](../../../../glossary.md#term-cpu) 0001–0004 status [synchronization](../../../../glossary.md#term-synchronization), CPU 0005–0016 `Draft` rows, absence of detailed task files
  for 0005–0016, and the preserved future Model 0026 dependency passed. Excluded architecture,
  ADR, architecture-test, shared API, Config, Engine, other-backend, extension, tool, Gradle,
  conformance, and integration paths remained unchanged.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [Engine glossary](../../../../glossary.md#term-engine); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance))</span>

Reasoned no-change conclusions:

- `ARCHITECTURE.md`, focused architecture pages, ADRs 0010/0011, and architecture tests already
  define backend-owned analysis, exact declarations, post-assignment finalization, artifact reuse,
  per-run representations, and cold binding. Task 0004 implements those boundaries without a new
  dependency edge or rule.<span data-glossary-navigation="true"> ([dependency edge glossary](../../../../glossary.md#term-dependency-edge))</span>
- Shared Model, Planning, Prepare, Runtime, Backend Contract, Trace, Compiler, and public API
  contracts remain accurate. In particular, <span data-glossary-navigation="true"> ([Trace glossary](../../../../glossary.md#trace); [Compiler glossary](../../../../glossary.md#term-compiler))</span>`HostTensorStorage` remains caller-owned,
  `CpuBorrowedBuffer` remains its non-owning Runtime representation, and shared Prepare—not CPU
  analysis—validates declared byte geometry against assigned plan geometry.
- Capability/public documentation, Config, Engine, OpenBLAS and other backends, extensions, tools,
  and CPU tasks 0001–0003 need no change: task 0004 adds no public composition, operation family,
  capability, vendor route, default artifact root, tuning/benchmark behavior, or completed-contract
  revision.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas); [operation glossary](../../../../glossary.md#operation))</span>
- Backend-conformance and integration tests remain unchanged because the implementation proves
  only package-private synthetic lifecycle mechanics and advertises no executable Model semantic.
  CPU and root Gradle configuration remain unchanged because task 0002 already supplies the Java
  26 Class-File/Vector API setup required here.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>

## Implementation notes

- Added immutable CPU-private analysis inputs, prepared parallel configuration, complete candidate
  and opaque-plan values, one direct candidate source, and one family-owned typed binder seam.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [opaque glossary](../../../../glossary.md#term-opaque))</span>
- Added deterministic CPU analysis that preserves exact backend declarations, selects the first
  valid target-eligible candidate, and performs no artifact, allocation, generation, binding,
  measurement, tuning, or Runtime work.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [target glossary](../../../../glossary.md#term-target); [artifact glossary](../../../../glossary.md#term-artifact); [binding glossary](../../../../glossary.md#term-binding); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Added CPU finalization that resolves every assigned use before one artifact-store call and
  constructs an immutable generated executable retaining the exact artifact, handle, prepared
  recipes, borrowed worker group, and typed binder.<span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease))</span>
- Extended the existing [CPU](../../../../glossary.md#term-cpu) cold boundary without changing it: [heap](../../../../glossary.md#term-heap) arrays, exact segments,
  `CpuBorrowedBuffer`, `CpuNativeBuffer`, workspaces, offsets, access, liveness, alignment, and
  parallel accessibility are validated before one signature-specific binder call.<span data-glossary-navigation="true"> ([liveness glossary](../../../../glossary.md#term-liveness); [alignment glossary](../../../../glossary.md#term-alignment); [signature glossary](../../../../glossary.md#term-signature))</span>
- Finalized all affected production/package Javadocs, added the typed preparation lifecycle and
  synthetic-proof boundary to the CPU guide, added reusable glossary definitions, and synchronized
  planning evidence and status.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [planning glossary](../../../../glossary.md#term-planning))</span>

## Completion summary

- Completed changes: CPU-private typed portable analysis, exact pre-assignment declarations,
  post-assignment artifact finalization, immutable strong generated-artifact retention, and direct
  per-run cold binding through the preserved Runtime representation boundary.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [immutable glossary](../../../../glossary.md#term-immutable); [run glossary](../../../../glossary.md#term-run); [binding glossary](../../../../glossary.md#term-binding); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Files changed or created: exactly the ten listed CPU production paths, three prescribed CPU
  tests, CPU backend guide, glossary, this task, CPU master plan, and roadmap.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- Tests and validation: the implementation pass recorded the final focused 3-suite/24-test pass
  and sole stabilized CPU 13-suite/72-test pass with zero failures, errors, or skips; the clean
  documentation context reused those results and passed CPU Javadoc, generated-page, Markdown,
  exact-scope, surface, status, excluded-path, and whitespace validation.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Documentation-agent review: `/root/cpu_0004_docs` independently finalized the combined change
  using the General, API/Javadoc, Backend Guide, Planning, and Example profiles without changing
  executable behavior.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: the CPU guide and glossary now explain the staged typed lifecycle,
  explicit root/borrowed-worker ownership, exact declaration-versus-assignment boundary,<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
  `CpuBorrowedBuffer` [lifetime](../../../../glossary.md#term-lifetime) role, direct hot call, and synthetic-only limitation.
- Javadoc review: every new production declaration, nested record, constructor, method, package
  boundary, parameter, result, expected failure, ownership/lifetime rule, and concurrency boundary
  was reviewed and final Javadoc generation passed.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>
- Glossary impact: added current CPU portable preparation-plan and prepared-executable terms and
  synchronized the generic preparer/finalization implementation status.
- Architecture, dependency, build, shared API, ADR, architecture-test, capability, conformance,
  and integration impact: none; reasoned no-change conclusions are recorded above.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Unresolved issues: None within [CPU](../../../../glossary.md#term-cpu) 0004 [scope](../../../../glossary.md#term-scope).
- Follow-up required: None for task 0004. CPU 0005 remains Draft and owns the first real portable
  operation-family implementation and truthful capability addition.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>

Status: Complete
