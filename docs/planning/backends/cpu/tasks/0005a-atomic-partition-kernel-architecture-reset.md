# Task 0005A: Atomic Partition-Kernel Architecture Reset

## Status

Complete

## Goal

Replace the provisional one-graph-node-to-one-lowering/emitter/invocation/kernel CPU pipeline in
one atomic change before further operation-family expansion. CPU analysis lowers one complete
CPU-owned `PlannedPartition`, forms computation-oriented execution units, performs legal fusion
before declaring resources, builds one route-independent canonical `CpuKernelIr` per unit, selects
the final route and execution strategy, and finalizes one partition-level `PreparedExecutable`.

The portable CPU route is the bytecode-first Java 26 Class-File API plus Vector API baseline and
the always-available semantic fallback for every occurrence it supports. CPU 0005A realizes only
its scalar strategy; later portable tasks add the Vector API strategies without changing the route
boundary. A metadata-only unit may emit no class. OpenBLAS is a narrow BLAS-compatible native
fallback, while vendor/platform-specialized libraries are peer routes selected by exact capability
and whole-plan cost. The scalar reference realization is for conformance and fail-closed checking,
not an `Operation` or IR interpreter in the Runtime hot path.

The proving slice is one fully static, canonical-dense FLOAT64 partition:

```text
a ----\
       ADD -> sum -> GELU -> activated ----\
b ----/                                     MUL -> output
c -----------------------------------------/

one PlannedPartition
  -> one CPU execution unit
  -> one canonical CpuKernelIr
  -> one generated class/artifact
  -> one partition-level BoundInvocation
```

`sum` and `activated` remain graph and canonical-IR values, but receive no buffer declaration,
slot, representation, or invocation argument. The generated loop accepts primitive `start` and
`end` bounds. The exact same generated class bytes and loaded compatibility identity must serve at
least two compatible element extents for the same ADD -> GELU -> MUL topology.

This task is an atomic working-tree replacement, not deletion of Git history. CPU tasks 0001
through 0005 remain historical `Complete` records during implementation. They become
`Superseded` only in the final synchronized status edit after implementation, focused and final
validation, the mandatory independent documentation pass, and the reset checkpoint all pass.

## Scope

- Adopt the structured CPU-internal package layout defined below. `CpuCapabilityProvider` remains
  the sole supported public CPU API; technically public cross-package contracts live only below
  `io.github.pho001.synaptik.backend.cpu.internal` and are documented as unsupported internal API.
- Establish one route-neutral selected-plan seam plus one `internal.route.portable` realization
  contract for the proving slice. Do not add native/vendor adapters or placeholder packages.
- Replace every per-node portable candidate, pointwise-ADD lowerer, finalizer, executable, and
  obsolete parallel foundation with one coherent partition pipeline. Leave no old/new parallel
  pipelines, aliases, bridges, adapters, registries, service locators, or dead production types.
- Lower the complete CPU-owned partition, form computation-oriented units, fuse before resource
  declaration, and emit exact declarations only for materialized unit and partition boundaries.
- Prove exactly one legal ADD -> exact GELU -> MUL fusion, one `CpuKernelIr`, one generated class,
  one partition-level `BoundInvocation`, and no physical resources for its two internal values.
- Admit only parameterless FLOAT64 `ADD`, exact `GELU`, and parameterless FLOAT64 `MUL` for the
  proving slice. Shapes are fully static, compatible, scalar or zero-element, and layouts are
  resolved canonical-dense, zero-offset native `MemorySegment` layouts. Other occurrences fail
  closed as complete occurrences before artifact access.
- Generate universal primitive `start`/`end` loops whose class bytes do not bake a concrete
  element count or compatible extents. Bind checked element count and extents once on the cold
  path. Lock same-byte and same-loaded-compatibility-identity reuse for at least two non-zero
  compatible extents, plus scalar and zero-element behavior.
- Preserve exact GELU semantics as the Model target
  `0.5 * x * (1 + erf(x / sqrt(2)))`. Preserve operation order and prohibit reassociation,
  contraction, fast math, or substitution of `GELU_TANH_APPROXIMATION`. Use a CPU-private scalar
  `erf` implementation shared by generated and reference realizations and retain the existing
  planned oracle/error requirements.
- Keep the final 0005A execution strategy scalar and single-threaded while representing execution
  strategy as the orthogonal compute/orchestration axes defined below. Do not retain unused worker
  or vector production foundations merely for later tasks.
- Make generated-class persistence an optional cold-path policy. Without a configured trusted
  artifact root, deterministically emit, verify, and define in memory. With a root, a compatible
  persistent hit may avoid emission but must still be verified and defined. Persistence is never
  correctness-critical, Runtime state, or preservation of JIT machine code or profiling.
- Preserve deterministic structural keys, process-local compatible loaded-artifact reuse, and
  strong ownership by the prepared execution. The default persistent policy remains disabled and
  carries no performance claim until a later evidence gate. Add no migration reader.
- Harden shared `GraphPreparation` narrowly: each logical value crossing two or more planned
  partitions must be declared by its producer partition, when present, and every distinct external
  consumer partition. Do not infer CPU fusion, virtuality, or materialization in shared Prepare.
- Independently finalize affected Javadocs, the CPU guide, glossary, task/master/roadmap records,
  and status synchronization in the mandatory clean documentation context after Java stabilizes.

## Out of scope

- Implementing general broadcast, offset/strided access, heap or mixed carriers, vector execution,
  parallel execution, CPU-internal materialization, reductions, scans, matrix multiplication,
  convolution, semantic specialized kernels, vendor routes, tuning, or benchmarks in 0005A.
- Creating `route.nativeblas`, `route.nativeops`, or provider-specific packages before their
  concrete Draft implementation tasks; adding an OpenBLAS, Accelerate, oneMKL, oneDNN, AOCL, or
  ZenDNN adapter in this reset.
- Fixed-shape or unrolled generated variants. They are permitted only in a later evidence-selected
  specialization with an explicit specialization budget.
- Interpreting `Operation`, `CompiledNode`, or `CpuKernelIr` in Runtime; class-per-operation
  generation; hot cursor objects; per-element division/modulo; hot lookup, allocation, route
  dispatch, storage classification, reflection, or registry access.
- Changing Model, Compiler, Planning, or Runtime semantics; putting gradient policy into CPU
  broadcasting; altering `LogicalMemoryPlan`; or changing shared Prepare beyond the narrow
  cross-partition declaration-completeness check.
- Editing `ARCHITECTURE.md`, ADRs, architecture tests, Gradle, module dependencies, public API,
  backend-conformance, integration behavior, or another backend.
- Creating a detailed specification for any later CPU task. Later work is represented only by the
  ordered Draft rows in the CPU master plan and roadmap.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Runtime, Prepare, and Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Performance evidence and tuning](../../../../architecture/performance-evidence-and-tuning.md)
- [CPU backend guide](../../../../backend-guide/cpu-backend.md)
- [Planning guide](../../../planning-guide.md)
- [CPU master plan](../master-plan.md)

## Architecture constraints

- Planning selects ownership and maximal same-owner partitions. CPU analysis owns whole-partition
  lowering, unit formation, fusion, route and execution-strategy selection, and exact resource
  declarations before shared assignment.
- Planning selects the single `BackendId("cpu")`; it never selects OpenBLAS or a vendor library as
  another backend. CPU Prepare compares the portable route, eligible native fallbacks, and
  eligible vendor/platform routes as peer candidates with no fixed vendor priority.
- Partition lowering, fusion legality and profitability, canonical IR, access planning,
  materialization accounting, numerical/determinism checks, and representation planning are
  shared and route-independent. Provider adapters may consume the selected route plan but may not
  reinterpret the graph or duplicate those responsibilities or resource lifetime ownership.
- CPU 0005A remains an exact/default ADD -> exact GELU -> MUL proving slice. Common analysis owns
  numerical eligibility and the selected numerical mode; emitters consume that decision, and no
  route may infer relaxed permission from hardware, provider availability, workload size,
  objectives, tuning, or benchmarks. The numerical mode participates in specialization/cache
  compatibility and the cold lowering manifest, with no hot-path policy lookup.
- The same exact/default policy applies to forward and compiler-generated gradient operations.
  This task creates no gradient-specific numerical policy and no `POW` strength reduction; those
  remain concise later master-plan rows.
- `LogicalMemoryPlan` remains complete for graph values. A physical Runtime slot exists only when
  backend analysis emits a `PreparationResourceRequirement.Buffer`.
- Analysis fixes fusion, access regime, materialization boundary, route, execution strategy, and
  declarations. Finalization after assignment may reuse or emit the selected artifact but may not
  change those decisions.
- Runtime receives one partition executable and a cold-bound invocation. It receives no graph
  semantics, operation objects, canonical IR, cache policy, or materialization decision.
- CPU consumes current Model shape and layout contracts. It does not replace symbolic dimension
  identity, singleton expansion, zero-extent rules, checked long spans, base offsets, or long
  strides with legacy CPU rules.
- Any need to alter an authoritative architecture rule, dependency direction, or module boundary
  is a stop condition rather than an implied part of this task.

## Permanent CPU lowering model

The reset establishes the boundary and data model; 0005A implements only the proving slice.
Eventually, CPU analysis must support all of the following through that same boundary:

- pointwise and broadcast fusion;
- folding of view and metadata operations when their semantics permit it;
- reduction and scan units, including numerically stable multi-pass units;
- matrix-multiplication and convolution epilogues; and
- explicit semantic specialized kernels.

Legal fusion and profitable fusion are separate decisions. Legality preserves node order,
numerical semantics, aliases and overlap, state, random-number semantics, publication, fan-out,
and cross-partition boundaries. Profitability may reject legal fusion because of code size,
register pressure, reuse, materialization, vendor eligibility, or measured cost. A decomposed
softmax sequence must not silently become a stable `SOFTMAX` unit unless an explicit semantic and
numerical equivalence policy authorizes that recognition.

The reset does not create one class per Model operation. One generated class realizes one selected
computation unit and may contain multiple ordered operations. Metadata-only units may need no
class. The portable generated route is the baseline and supported semantic fallback. OpenBLAS is
only a narrow BLAS-compatible native fallback, not a universal or preferred route. Accelerate,
oneMKL, oneDNN, AOCL, and optional ZenDNN are later peer routes, not fallback dispatch inside a
generated loop.

All routes consume common CPU analysis. Native provider adapters own only ABI calls and their own
provider-specific compatibility/lifetime mechanics. They do not own graph interpretation,
broadcast planning, fusion, canonical IR construction, materialization decisions,
numerical/determinism policy, representation planning, or Runtime resource lifetime.

Route families remain distinct: Apple CPU may select Accelerate BLAS, vDSP, or vForce; Intel may
select oneMKL BLAS/VML or the separate oneDNN family; AMD may select AOCL-BLAS/AOCL-LibM or the
separate optional ZenDNN family. ARM is capability-first, not tied to one library: Apple Silicon
may use Accelerate, while other ARM targets use portable code generation unless a later task adds
an explicitly verified provider.

## Canonical IR, access plans, and identity

There is one shared route-independent `CpuKernelIr`. It records:

- typed boundary values and typed virtual values;
- ordered computation topology and exact operation semantics;
- the structural form of each normalized input access plan;
- a universal primitive `start`/`end` loop model; and
- ordered output stores.

It does not record a selected route, thread count, vector species, trusted artifact root, Runtime
slot, segment instance or address, graph identity, concrete invocation binding, generator version,
or JIT state. The default canonical identity does not include concrete compatible extents or the
element count.

The eventual normalized `CpuAccessPlan` is the single access system for elementwise, `WHERE`, and
fused operands. Each input has aligned rank, cold-bound concrete extents, base offset, effective
long strides, and zero strides on broadcast axes. It must cover scalars, rank expansion, singleton
and multi-axis broadcast, zero extents, multi-input masks and parameters, different fused-operand
patterns, offset-contiguous and positive-strided layouts, zero-stride views, heap arrays,
`MemorySegment`, mixed carriers, and dynamic or symbolic dimensions once Prepare has bound them
exactly. The IR contains the access-plan form; the invocation binding supplies instance extents,
offsets, carriers, and addresses.

The specialization and artifact identity compose canonical IR with the structural access regime,
fusion form, data types, and separately selected route, execution strategy, route configuration,
and CPU/JDK/class-file/generator compatibility fingerprints. Instance bindings remain excluded.
Fixed-shape or unrolled variants may add exact extents only as explicit evidence-selected
specializations, subject to a bounded specialization budget that prevents class explosion.

## Broadcasting and layout requirements for follow-up work

Full right-aligned broadcasting must meet or exceed the selected legacy capability while using
current `ShapeBroadcast` and `LayoutDescriptor` semantics. Current Model contracts are stronger
than the legacy implementation: `Dimension` equality is symbolic/canonical, singleton expansion
is explicit, zero extents are valid, offsets and strides are `long`, zero-stride views are
representable, and spans use checked arithmetic. CPU must consume those contracts.

CPU must not copy the legacy `int` arithmetic, `Math.max` zero-dimension bug, duplicated
elementwise/`WHERE`/fusion planners, per-operation storage-method matrices, Runtime fallback,
hot cursor objects, or incomplete native/vector broadcast eligibility. Broadcast gradients remain
the Compiler/Model `SUM_TO_SHAPE` responsibility and later CPU reduction coverage; CPU access
planning contains no gradient policy.

The planned performance tiers, from simplest to most general, are:

1. dense linear access;
2. scalar or all-zero-stride broadcast;
3. last-axis bias broadcast;
4. block/outer broadcast with a contiguous inner loop;
5. general positive-strided odometer initialized once from `start`;
6. scalar fallback;
7. cost-gated vector gather; and
8. optional contiguous materialization.

Generated bytecode performs offset and carry arithmetic directly. It creates no hot cursor object,
makes no per-element virtual call, and uses no per-element division or modulo. General gather is
optional where scalar execution or materialization wins. Complete semantic coverage does not imply
that every layout is vectorized.

## Materialization analysis and shared assignment

Before shared assignment, CPU analysis eventually compares direct strided/broadcast execution with
CPU-internal contiguous materialization. The comparison includes copy cost, kernel benefit,
reuse/fan-out, vendor eligibility, memory cost, and expected repeated runs. If materialization is
selected, CPU analysis declares the exact internal resource before assignment and lowers a copy
unit plus consumer unit without changing the Model graph. Shared Prepare remains CPU-blind.

For 0005A, apply these exact rules:

1. Retain every graph value and its logical requirement in `LogicalMemoryPlan`.
2. Declare buffers for `a`, `b`, `c`, and `output` in deterministic encounter order.
3. Represent `sum` and `activated` as virtual `CpuKernelIr` values with no declaration or slot.
4. Reject publication, extra fan-out, cross-partition use, alias ambiguity, or other facts that
   would require either intermediate to be materialized.
5. Use checked `long` element and byte arithmetic, including scalar and zero-element cases.

Shared Prepare checks only partition boundaries. For a logical value produced in `P` and consumed
in distinct partitions `Q` and `R`, analyses for `P`, `Q`, and `R` must each declare the exact
`ValueId`. A value confined to `P` may remain undeclared. Existing bindable-input, constant,
initialized-representation, publication, and geometry checks remain unchanged.

## Execution strategies

The final portable strategy vocabulary is exactly the Cartesian product of two axes:

| Compute axis | Orchestration axis | Strategy |
|---|---|---|
| scalar | single-thread | scalar |
| vector | single-thread | vector |
| scalar | parallel | parallel-scalar |
| vector | parallel | parallel-vector |

Generated kernels always accept primitive `start` and `end`. Parallel workers dispatch chunks
outside the generated inner loop. Vector chunks and scalar or masked tails must be correct for
arbitrary legal bounds. CPU 0005A implements only scalar/single-thread; ordered Draft follow-ups
deliver vector, parallel-scalar, and parallel-vector without changing this vocabulary.

## Optional generated-class persistence

Deterministic structural identity and class-byte verification are mandatory; filesystem
persistence is not. The cold path has two policies:

- no trusted artifact root: deterministically emit, structurally verify, define, and process-locally
  intern compatible generated class bytes in memory;
- configured trusted artifact root: attempt compatible lookup, then verify and define a hit, or
  deterministically emit, verify, define, and optionally publish a miss.

A persistent hit can avoid byte emission only. It cannot preserve JVM JIT machine code, profile,
or hidden-class state. Corrupt, incompatible, absent, or unusable storage must safely fall back to
deterministic in-memory emission. Persistence is disabled and makes no default performance claim
until the ordered evidence-gate task justifies a policy change.

The reset defines one current schema and accepts only that schema. It has no reader, converter,
alias, or migration path for earlier entries. `CpuPreparedExecutable` strongly owns the selected
loaded artifact and direct handle for its lifetime; persistent storage and process-local interning
do not weaken that ownership.

## Structured internal packages and visibility

The reset adopts this package structure immediately:

```text
io.github.pho001.synaptik.backend.cpu
  CpuCapabilityProvider                 sole supported public CPU API

io.github.pho001.synaptik.backend.cpu.internal
  memory                                representations and cold binding
  prepare                               analysis/finalization lifecycle
  lowering                              partition-to-unit lowering and fusion
  ir                                    canonical IR and normalized access plans
  codegen.emit                           portable Class-File generation and loop emission
  route.portable                         portable selection/realization plan
  cache                                 structural identity and optional persistence
  executable                            prepared partition execution
  reference                             conformance/fail-closed scalar reference
```

CPU 0005A creates only the packages above. The permanent target map reserves these later locations
without creating them now:

```text
internal.route.nativeblas
  openblas                              narrow BLAS-compatible native fallback
  accelerate                            Apple Accelerate BLAS
  mkl                                   Intel oneMKL BLAS
  aocl                                  AMD AOCL-BLAS

internal.route.nativeops
  accelerate                            Apple vDSP/vForce
  mkl                                   Intel oneMKL VML
  onednn                                Intel oneDNN partition operations
  aocl                                  AMD AOCL-LibM
  zendnn                                optional AMD ZenDNN partition operations
```

Those are route-family/provider leaves, not separate backends. A concrete later task may refine a
leaf when verified provider boundaries require it, but may not move common lowering, access,
fusion, materialization, numerical, representation, or lifecycle ownership into that leaf.

Java subpackages are not friends. The exact technically public, unsupported cross-package types are
`CpuBorrowedBuffer`, `CpuBufferArgument`, `CpuBufferRepresentation`, `CpuNativeBuffer`,
`CpuPartitionAnalysisInputs`, `CpuPartitionPreparationPlan`, `CpuPartitionPreparer`,
`CpuPartitionFinalizer`, `CpuPartitionLowering`, `CpuKernelIr`, `CpuAccessPlan`,
`CpuClassFileKernelGenerator`, `CpuGeneratedKernel`, `CpuPortableRoutePlan`,
`CpuGeneratedKernelArtifactStore`,
`CpuGeneratorSchema`, `CpuKernelSpecialization`, `CpuLoweringFingerprint`,
`CpuPreparedExecutable`, and `CpuScalarReferenceKernel`. Each is required by a named adjacent
package or by the sole supported provider and remains unsupported internal API by namespace and
Javadoc. `CpuCarrierEmitter`, `CpuLoopEmitter`, and `CpuScalarEmitter` remain package-private to
`codegen.emit`; implementation-only nested types are no more visible than their exact consumer
requires. Add no JPMS export, service locator, registry, broad facade, or public compatibility
bridge. Focused inventory tests enforce the sole supported public surface, this visibility list,
and the exact internal package/type inventory.

## Exact production path and type disposition

Every current production path is listed. Destination shorthands such as
`internal/memory/CpuBorrowedBuffer.java` expand beneath
`backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/`. `Move/rewrite` deletes the old
path and creates only the named destination. `Remove` has no replacement alias. Nested types
follow their enclosing type.

| Current path | Action and exact destination |
|---|---|
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProvider.java` | Rewrite in place; sole supported public entry and complete-occurrence fail-closed reporting |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/package-info.java` | Rewrite in place; supported surface and internal-package boundary |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuBorrowedBuffer.java` | Move/rewrite to `internal/memory/CpuBorrowedBuffer.java` |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuBufferArgument.java` | Move/rewrite to `internal/memory/CpuBufferArgument.java` |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuBufferRepresentation.java` | Move/rewrite to `internal/memory/CpuBufferRepresentation.java` |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuNativeBuffer.java` | Move/rewrite to `internal/memory/CpuNativeBuffer.java` |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuCarrierEmitter.java` | Move/rewrite to `internal/codegen/emit/CpuCarrierEmitter.java` |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuClassFileKernelGenerator.java` | Move/rewrite to `internal/codegen/emit/CpuClassFileKernelGenerator.java` |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuGeneratedKernel.java` | Move/rewrite to `internal/codegen/emit/CpuGeneratedKernel.java` |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuLoopEmitter.java` | Move/rewrite to `internal/codegen/emit/CpuLoopEmitter.java` |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuScalarEmitter.java` | Move/rewrite to `internal/codegen/emit/CpuScalarEmitter.java` |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuGeneratedKernelArtifactStore.java` | Move/rewrite to `internal/cache/CpuGeneratedKernelArtifactStore.java`; optional trusted-root policy |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuGeneratorSchema.java` | Move/rewrite to `internal/cache/CpuGeneratorSchema.java`; one current schema, no migration reader |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuKernelSpecialization.java` | Move/rewrite to `internal/cache/CpuKernelSpecialization.java`; structural specialization without instance extents by default |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuLoweringFingerprint.java` | Move/rewrite to `internal/cache/CpuLoweringFingerprint.java` |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPreparedExecutable.java` | Move/rewrite to `internal/executable/CpuPreparedExecutable.java`; partition-level executable and strong artifact owner |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuFamilyKernelEmitter.java` | Remove; canonical-IR emission replaces family callback shape |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuNativeWorkspace.java` | Remove; no 0005A workspace consumer |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuParallelExecutionException.java` | Remove; parallel strategy returns in ordered follow-up |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPointwiseAddCandidateSource.java` | Remove; replaced by whole-partition lowering |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPointwiseAddInvocation.java` | Remove; replaced by partition-level cold binding |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPointwiseAddKernelEmitter.java` | Remove; replaced by canonical-IR scalar emission |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPointwiseAddLowering.java` | Remove; replaced by `internal/lowering/CpuPartitionLowering.java` |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPortableAnalysisInputs.java` | Remove; replaced by `internal/prepare/CpuPartitionAnalysisInputs.java` |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPortableCandidateSource.java` | Remove; no candidate registry/source layer |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPortableExecutionMode.java` | Remove; strategy is explicit plan data, not emitter dispatch |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPortableInvocationBinder.java` | Remove; binding belongs to the selected preparation plan/executable |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPortableKernelCandidate.java` | Remove; replaced by canonical IR plus selected plan facts |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPortablePartitionCandidate.java` | Remove; replaced by `internal/prepare/CpuPartitionPreparationPlan.java` |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPortablePartitionFinalizer.java` | Remove; replaced by `internal/prepare/CpuPartitionFinalizer.java` |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPortablePartitionPreparer.java` | Remove; replaced by `internal/prepare/CpuPartitionPreparer.java` |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPortablePreparationPlan.java` | Remove; replaced by `internal/prepare/CpuPartitionPreparationPlan.java` |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPortablePreparedExecutable.java` | Remove; replaced by `internal/executable/CpuPreparedExecutable.java` |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPreparedParallelConfiguration.java` | Remove; no 0005A parallel strategy implementation |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuRangeBody.java` | Remove; generated primitive bounds replace hot range callbacks |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuReductionEmitter.java` | Remove; no current consumer and reductions return in a later unit task |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuVectorEmitter.java` | Remove; vector emission returns with implemented vector strategy |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuWorkerGroup.java` | Remove; parallel orchestration returns in ordered follow-up |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/package-info.java` | Remove; replaced by exact structured package documentation below |

The new top-level production types are exactly:

| New path | Type and role |
|---|---|
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionAnalysisInputs.java` | Checked complete-partition analysis inputs |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparationPlan.java` | Immutable units, selected route/strategy, bindings, and declarations; exact nested types `ExecutionUnitPlan`, `Route`, and `ExecutionStrategy` |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java` | Whole-partition analysis entry |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizer.java` | Post-assignment verification, artifact realization, and executable finalization |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionLowering.java` | Unit formation, exact fusion legality, and canonical lowering |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuKernelIr.java` | Route-independent typed computation and universal loop IR; exact nested value types `Value`, `Instruction`, `Loop`, and `Store` |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuAccessPlan.java` | One normalized per-input access-plan form and cold binding contract; exact nested value types `Regime` and `Binding` |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/portable/CpuPortableRoutePlan.java` | Immutable portable Class-File realization facts beneath the route-neutral partition plan; no graph interpretation, scalar-reference ownership, or native-provider policy |
| `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuScalarReferenceKernel.java` | Conformance/fail-closed scalar reference, never a Runtime IR interpreter |

Add exactly these ten package documentation paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/memory/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/portable/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/package-info.java`

They state unsupported-internal status, ownership, hot/cold boundaries, and permitted
collaboration.

## Exact test path disposition

| Current path | Action and exact destination |
|---|---|
| `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/.gitkeep` | Remove; real root-package tests remain |
| `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProviderPublicShapeTest.java` | Rewrite in place for sole supported public type and internal namespace exclusion |
| `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProviderTest.java` | Rewrite in place for exact complete-occurrence truth |
| `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/execution/CpuBufferBindingTest.java` | Move/rewrite to `internal/memory/CpuBufferBindingTest.java` |
| `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/execution/CpuClassFileKernelGeneratorTest.java` | Move/rewrite to `internal/codegen/emit/CpuClassFileKernelGeneratorTest.java` |
| `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/execution/CpuGeneratedKernelArtifactStoreTest.java` | Move/rewrite to `internal/cache/CpuGeneratedKernelArtifactStoreTest.java` |
| `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/execution/CpuGeneratedKernelShapeTest.java` | Move/rewrite to `internal/codegen/emit/CpuGeneratedKernelShapeTest.java` |
| `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/execution/CpuKernelSpecializationTest.java` | Move/rewrite to `internal/cache/CpuKernelSpecializationTest.java` |
| `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/execution/CpuNativeRepresentationTest.java` | Move/rewrite to `internal/memory/CpuNativeRepresentationTest.java` |
| `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPreparedExecutableTest.java` | Move/rewrite to `internal/executable/CpuPreparedExecutableTest.java` |
| `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPointwiseAddCandidateSourceTest.java` | Remove; obsolete candidate architecture |
| `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPointwiseAddExecutionTest.java` | Remove; replaced by fused generated/reference coverage |
| `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPortablePartitionFinalizerTest.java` | Remove; replaced by partition finalizer coverage |
| `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPortablePartitionPreparerTest.java` | Remove; replaced by partition preparer coverage |
| `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPortablePreparedExecutableTest.java` | Remove; replaced by partition executable coverage |
| `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/execution/CpuWorkerGroupTest.java` | Remove; parallel coverage returns with parallel strategies |

Add exactly these seven new test paths:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuInternalPackageInventoryTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuKernelIrTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuFusedGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuShapePolymorphicArtifactTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuReferenceDifferentialTest.java`

## Atomic implementation path ceiling

The implementation task may touch at most 114 paths:

| Category | Maximum | Arithmetic |
|---|---:|---|
| CPU production | 72 | 39 current paths plus 14 moved destinations, 9 new type paths, and 10 new package-info paths; the two in-place paths are already among the 39 |
| CPU tests | 30 | 16 current paths, including `.gitkeep`, plus 7 moved destinations and 7 new test paths |
| Shared Prepare | 2 | `GraphPreparation.java` and `GraphPreparationTest.java` only |
| Explanatory documentation | 2 | CPU backend guide and glossary |
| Planning/status | 8 | this task, CPU master plan, roadmap, and the five historical CPU task status files |
| **Total** | **114** | **72 + 30 + 2 + 2 + 8** |

All source destination paths are enumerated above. No Java source may remain under the old
`backend.cpu.execution` package. No test may remain there. A necessary path outside this ceiling is
a planning stop and correction, not implied authorization.

## Acceptance criteria

- The old per-node pipeline and flat `execution` package are absent; every production and test
  path has the exact disposition above, and no dead or compatibility production code remains.
- Only `route.portable` exists after 0005A. No `route.nativeblas`, `route.nativeops`, OpenBLAS, or
  vendor leaf package is created before its concrete Draft task.
- `CpuCapabilityProvider` is the sole supported public CPU API. Minimal cross-subpackage contracts
  are technically public only below `.internal`, documented unsupported, and enforced by focused
  public-shape and inventory tests.
- One complete eligible partition lowers to one execution unit, one canonical IR, one generated
  class, one partition executable, and one `BoundInvocation`; its two internal values have no
  declaration or physical slot.
- The same generated class bytes and loaded compatibility identity execute at least two compatible
  non-zero extents for the identical ADD -> GELU -> MUL topology. Concrete extents and element
  count are cold-bound and excluded from default class/cache identity.
- Generated loops use primitive `start` and `end`; scalar/single-thread execution handles arbitrary
  bounds, scalar shape, zero elements, checked long arithmetic, and correct overlap/alias rejection.
- Canonical IR is route-independent and excludes every selected-route/configuration, Runtime,
  graph-identity, generator-version, and instance-binding fact named above.
- Portable Class-File generation is the supported semantic baseline/fallback; route-neutral
  preparation admits later peer routes without a bridge layer, provider-owned lowering, another
  `BackendId`, or a fixed vendor priority.
- No-root artifact realization works entirely in memory. Configured-root hits are fully verified
  and defined; corrupt/missing storage safely emits in memory. No migration reader exists and no
  test claims persistence preserves JIT code or is faster.
- Capability is truthful for complete occurrences and fails closed for unsupported type, shape,
  layout, parameter, alias, state, fan-out, publication, or partition facts.
- Exact GELU, NaN/infinity/signed-zero classifications, fixed operation order, generated/reference
  differential results, vector-tail invariants at the IR contract boundary, determinism, and
  compatibility fingerprints have focused tests. Every finite result has absolute error at most
  `2e-7 * max(1, abs(idealGelu))` against precomputed high-precision oracle vectors spanning tails,
  `[-12, 12]`, threshold neighborhoods, subnormals, and ordinary values.
- The proving slice uses exact/default eligibility only; its selected numerical mode is present in
  specialization/cache compatibility and the cold manifest, and neither emitters nor the hot path
  can introduce or look up an independent relaxed policy.
- Shared Prepare rejects missing cross-partition producer/consumer declarations while permitting an
  undeclared same-partition value. It learns no CPU virtuality or materialization policy.
- Trace/debug support can emit a cold lowering manifest containing unit boundaries, fusion reasons,
  access regime, selected route/strategy, structural key, and resource declarations without
  exposing segments, addresses, or hot-path lookup.
- Prepared execution strongly owns artifacts and bound resources for the documented lifetime and
  performs no hot lookup, allocation, reflection, storage classification, or semantic dispatch.
- Focused CPU, Model/Planning/Prepare/Runtime contract checks, CPU Javadoc, module validation, and
  the reset checkpoint pass before statuses change. Repository-wide validation is required because
  shared Prepare and the complete CPU package surface change.

## Validation

Run the implementation-time commands required by the planning guide and record exact suites,
counts, and results. At minimum:

```bash
./gradlew :backends:cpu:test
./gradlew :modules:prepare:test
./gradlew :backends:cpu:javadoc
./gradlew test
git diff --check
```

Also validate:

- same topology/two extents -> identical deterministic class bytes, one compatibility identity,
  different cold-bound start/end and correct results;
- exactly four buffer declarations and no slots for the fused intermediate values;
- no-root, compatible-root hit, corrupt-root fallback, and no-migration behavior;
- exact public and internal package/type inventory, only the implemented `route.portable` leaf,
  and no `backend.cpu.execution` paths/imports or native/vendor placeholder packages;
- no `Operation`, `CompiledNode`, IR interpretation, cursor object, per-element division/modulo,
  lookup, allocation, reflection, registry, or service locator in the generated/hot path;
- checked long arithmetic, zero/scalar behavior, alias rejection, numerical/determinism rules,
  vector chunk/tail contract, fingerprints, and generated/reference differential vectors;
- local Markdown links, anchors, fences, final newlines, and trailing whitespace;
- 0005A is the sole detailed Ready CPU task and every immediate follow-up is Draft with no spec;
- CPU 0001–0005 remain Complete until every final gate passes, then change atomically to
  Superseded while 0005A changes to Complete.

## Dependencies

- CPU tasks 0001 through 0005 are completed historical evidence.
- Current Model `Shape`, `Dimension`, `ShapeBroadcast`, and `LayoutDescriptor` contracts.
- Current Planning `LogicalMemoryPlan` and `PlannedPartition` contracts.
- Current Prepare staged analysis/assignment/finalization and Runtime `PreparedExecutable` /
  `BoundInvocation` contracts.

## Ordered follow-up gates

The CPU master plan contains the only authorized follow-up records. In order they deliver:

1. universal right-aligned access plans and complete broadcast/layout semantics;
2. vector, parallel-scalar, and parallel-vector execution strategies;
3. CPU-internal materialization analysis and the optional persistence/specialization evidence gate;
4. broader portable pointwise types, carriers, and semantic families; then reduction/scan,
   matrix/convolution, vendor, and tuning milestones.

No follow-up has a detailed task specification while 0005A is Ready.

## Implementation prompt

Implement this task in a clean coding context. Read the required architecture, planning,
documentation, completed CPU-task, current CPU/Prepare/Runtime/Model-contract, and selected legacy
evidence files. Treat legacy as behavioral evidence only. Respect the exact path map and 114-path
ceiling. Do not retain both pipelines, invent architecture, create future detailed specs, commit,
or push. After Java and tests stabilize, use a distinct clean documentation context to finalize
Javadocs and the authorized Markdown paths. Mark historical tasks Superseded and this task Complete
only after every implementation, validation, documentation, and checkpoint gate passes.

## Local decisions

- The current route set contains exactly `internal.route.portable`; no native or vendor leaf is
  present before its ordered concrete task.
- CPU 0005A selects only exact/default numerical mode and scalar/single-thread execution. The
  selected mode and strategy participate in compatibility identity rather than hot-path policy.
- Optional persistence accepts one current schema only. Absence, incompatibility, corruption, or
  publication failure returns to verified in-memory generation and does not affect correctness.
- Shared Prepare changed only its backend-neutral cross-partition declaration-completeness check.
  It gained no knowledge of CPU units, fusion, virtuality, or materialization.

## Known limitations

- The implemented semantic slice is exactly parameterless canonical-dense FLOAT64
  `ADD -> GELU -> MUL` with fully static equal shapes and exact GELU semantics.
- Only native `MemorySegment` buffers, canonical-dense zero-offset access, and the portable scalar
  strategy are executable. Broadcasting, views, heap or mixed carriers, vector and parallel
  execution, native routes, CPU-internal materialization, and broader semantic families remain
  ordered Draft work.
- The task makes no backend-conformance, integration, performance, relaxed-math, tuning, or
  persistence-speed claim.

## Validation evidence

Implementation context completed the executable validation before the independent documentation
pass:

- `./gradlew :backends:cpu:test` passed 16 suites and 18 tests with zero failures, errors, or
  skips. The final replacement run recompiled CPU production because concurrent Javadoc edits were
  visible and still passed.
- `./gradlew :modules:prepare:test` passed 10 suites and 36 tests with zero failures, errors, or
  skips.
- `./gradlew test` passed with 56 actionable tasks: 1 executed and 55 up-to-date.
- `git diff --check` passed before the documentation pass.

The clean documentation-focused context independently reviewed the final CPU and shared Prepare
source and tests against the API/Javadoc, backend-guide, planning, example, and general profiles.
It finalized every affected CPU contract Javadoc, all ten structured internal package summaries,
the CPU guide, glossary, and synchronized planning/status records. It reused the successful Java
test evidence because it changed no executable behavior. Its final evidence is:

- `./gradlew :backends:cpu:javadoc` passed after the final Javadoc edits.
- Targeted local Markdown target/anchor, fence, final-newline, and trailing-whitespace checks
  passed for the authorized documentation paths.
- Exact production/test inventory, route-leaf, removed-package/import, later-task, affected-path,
  and 114-path-ceiling checks passed; the two pre-existing compiler/config planning edits were
  identified separately and preserved unchanged.
- `git diff --check` passed after all documentation and status edits.

## Implementation notes

The flat `backend.cpu.execution` production and test trees were removed atomically. The replacement
contains 35 CPU production Java paths and 16 CPU test Java paths, with one `route.portable` leaf
and no native placeholders. One eligible partition lowers to one fused unit, one canonical IR,
one artifact, one executable, and one bound invocation. Its four declarations cover `a`, `b`,
`c`, and output; `sum` and `activated` remain virtual with no physical slot. Focused tests lock
universal primitive bounds, two compatible extents sharing deterministic bytes and loaded class
identity, exact generated/reference results, persistence fallback, preserved graph/logical-memory
facts, and narrow cross-partition declaration hardening.

No architecture contract, focused architecture explanation, ADR, architecture test, backend
conformance test, integration test, Gradle configuration, module dependency, Model, Compiler,
Planning, Runtime, Config, Trace, or other backend change was required. CPU-private ownership and
the existing staged Prepare/Runtime boundary remain unchanged; only the authorized shared Prepare
validation was hardened.

## Completion summary

Implemented and documented the atomic CPU partition-kernel reset. Historical CPU tasks 0001
through 0005 are preserved as Superseded records, CPU 0005A is Complete, and every later CPU task
remains Draft without a detailed task specification.

Status: Complete
