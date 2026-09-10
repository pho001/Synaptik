# Task 0010: Narrow OpenBLAS BLAS-Compatible Native Route

## Status

Complete

## Goal

Add the first CPU-native peer route at
`io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas`. The route consumes the
existing common MATMUL lowering and invokes the completed low-level OpenBLAS provider only for one
small, exactly defined dense SGEMM/DGEMM subset. CPU analysis retains the portable generated
candidate, filters the native candidate exactly, accounts for any one-input carrier/layout
materialization, and selects OpenBLAS only when caller-supplied immutable cost facts prove the
complete native plan cheaper by both configured thresholds.

OpenBLAS remains an optional route within `BackendId("cpu")`, never another backend, a universal
MATMUL implementation, a preferred route, a runtime fallback, or a provider-owned graph/lowering
pipeline. Default analysis remains portable-only.

## Scope

- Extend CPU-private analysis inputs with one disabled-by-default typed OpenBLAS route
  configuration and immutable expected boundary-storage facts. These are availability,
  qualification, thread-configuration, alignment, and cost snapshots only; they contain no
  `OpenBlasLibrary`, segment, address, runtime slot, mutable resource, or graph semantics.
- Generate the OpenBLAS candidate only after the current partition DAG, MATMUL unit, fusion,
  access, portable realization, and representation facts have been derived by common CPU
  lowering. The OpenBLAS leaf consumes those facts and performs no graph interpretation.
- Admit exactly one bare, one-node, one-unit rank-two MATMUL with positive static `m`, `n`, and
  `k`; same-type `FLOAT32` or same-type `FLOAT64` inputs/result; exact/default numerical mode;
  canonical dense row-major zero-offset output; and dimensions representable by the provider's
  32-bit `blasint` contract.
- Use one provider call with `alpha = 1` and `beta = 0`: `sgemm` for `FLOAT32`, `dgemm` for
  `FLOAT64`. The selected CPU route plan carries the exact type, `m`, `n`, `k`, boundary
  positions, selected thread count, representation choice, and whole-plan cost decision. It
  carries no live provider or runtime representation.
- Permit either direct native `A`, `B`, and `C`, or exactly one existing CPU affine
  materialization of `A` or `B` into a run-owned aligned native contiguous workspace. One copy may
  jointly resolve a source's non-canonical layout and non-native carrier. The output is never
  copied, two-input materialization remains rejected, and no second copy/packing framework is
  added.
- Extend the route-neutral partition plan only enough to represent `PORTABLE` or `OPENBLAS` and
  the exact optional OpenBLAS plan. Preserve the selected portable route plan as the supported
  semantic alternative in the immutable analysis result. Finalization and Runtime never
  reselect between them.
- Add one narrow CPU-private invocation seam that borrows and strongly retains the current
  `OpenBlasLibrary` for prepared use, exposes only the exact SGEMM/DGEMM and thread-count checks
  needed by this route, and enables deterministic native-free tests. It does not load, discover,
  configure, close, restore, or otherwise own the provider handle.
- Extend finalization and prepared execution so a direct OpenBLAS plan produces one immutable
  native executable, while a one-copy plan uses the existing partition composite to execute the
  generated affine copy once before the native child. Generalize the composite child type only as
  far as needed to retain portable and native prepared children.
- Validate the route with native-free fake invocation tests and one explicit test-source native
  checkpoint using a caller-supplied absolute compatible OpenBLAS path. Ordinary tests perform no
  native lookup, skip, discovery, or environment/property-based activation.
- Replace the backend-conformance placeholder with one narrow ordinary conformance test that
  constructs the existing one-node MATMUL preparation projection, selects the direct OpenBLAS
  route from explicit immutable facts, finalizes it with the fake invocation seam, cold-binds
  arena-backed `MemorySegmentStorage` through `CpuBorrowedBuffer` in an exact `RunState`, and
  executes the prepared recipe. This test uses no Engine and performs no native-library lookup or
  call.
- Finalize the CPU backend guide, every changed Java/Javadoc contract, native route package
  documentation, glossary impact, this task, CPU master plan, and roadmap through a distinct
  clean documentation-focused context after executable Java stabilizes.

## Exact initial eligibility

Every row is mandatory. Failure or uncertainty in any row removes only the OpenBLAS candidate and
leaves the already-valid portable plan selected during analysis.

| Dimension | Required OpenBLAS fact |
|---|---|
| Partition/unit | Exactly one partition node and one execution unit; the unit is MATMUL only, with no fused bias, terminal, virtual suffix, sibling unit, or publication-sensitive internal value. |
| Shape | Resolved rank-two `A[m,k]`, `B[k,n]`, and `C[m,n]`; `m`, `n`, and `k` are each in `[1, Integer.MAX_VALUE]`; all checked element/span arithmetic succeeds. |
| Type | `A`, `B`, and `C` are all `FLOAT32`, or all `FLOAT64`; no promotion or output conversion. |
| Numerical mode | Existing `EXACT_DEFAULT`; the MATMUL contract's permitted reassociation/FMA applies, but the supplied provider binary must have passed this task's CPU-native conformance checkpoint. |
| Operation scalars | Fixed `alpha = 1` and `beta = 0`; the route does not read an existing logical output value. |
| Output | Canonical dense row-major, zero logical offset, injective, direct expected native segment, writable at binding, type-width aligned, and large enough for exactly `m * n` elements. |
| Direct input | Canonical dense row-major, zero logical offset, expected native segment, type-width aligned, and large enough for its complete matrix. |
| Copied input | At most one of `A` or `B`; any current admitted non-negative affine layout/carrier that the existing affine-copy plan can read completely and copy into one canonical native workspace. |
| Aliasing | `A` and `B` may overlap. Binding rejects any overlap between `C` and either effective input, or between a materialization workspace and any selected buffer/workspace, before a copy or GEMM can write. |
| Execution | One OpenBLAS call on the invoking thread with the sole initial `SINGLE_THREAD` OpenBLAS configuration, whose provider count is exactly one. CPU portable worker ranges/species do not leak into the native plan. |
| Availability | The typed route input is enabled, marks the exact supplied provider as CPU-route-qualified, and selects `SINGLE_THREAD`. Finalization has a matching open borrowed invocation, and cold binding observes provider thread count one. |
| Profitability | The complete native candidate passes the deterministic whole-plan comparison below. Equality, insufficient benefit, overflow, or missing cost facts retains portable. |

The provider still revalidates dimensions, native segments, alignment, spans, output mutability,
and overlap. CPU binding performs its own route-specific checks first so all selected resources
fail before any materialization or output mutation.

## Whole-plan candidate and cost contract

`CpuPartitionAnalysisInputs.OpenBlasRouteConfig` is a closed, typed configuration with a
fail-closed `DISABLED` value. An enabled value contains:

- one explicit `QUALIFIED` availability value meaning that composition has associated this input
  with a compatible 32-bit-`blasint` OpenBLAS binary that passed the CPU checkpoint;
- the sole initial `SINGLE_THREAD` OpenBLAS configuration, meaning provider thread count exactly
  one, already installed and externally coordinated by the caller for the complete prepared-use
  lifetime;
- non-negative portable fixed, per-output, and per-multiply-accumulate cost units;
- non-negative OpenBLAS fixed, per-output, and per-multiply-accumulate cost units; and
- non-negative absolute benefit and relative basis-point thresholds, with basis points in
  `[0, 10_000]`.

The values are dimensionless immutable heuristic inputs, not measurements, benchmark claims,
tuning-cache entries, provider capabilities, or Planning-module scoring. The OpenBLAS fixed term
must include the native transition/provider-call cost and the supplied thread configuration's
known fixed cost. A selected copy reuses the current `MaterializationPolicy` expected-run count,
copy fixed/per-element terms, byte ceiling, and exact `CpuMaterializationPlan`; its fixed term is
documented to include run-owned workspace allocation/binding and one affine-copy invocation. No
copy candidate exists unless materialization policy is enabled and its exact bytes fit the
existing ceiling.

For `W = m * n * k`, `O = m * n`, expected runs `R`, and copied logical elements `E` (zero for a
direct candidate), compare checked non-negative costs:

```text
portable = R * (portableFixed + portablePerOutput * O + portablePerMac * W)

openblas = R * (openblasFixed + openblasPerOutput * O + openblasPerMac * W
                + copyFixed(if E > 0) + copyPerElement * E)
```

The candidate wins only when `openblas < portable`, `portable - openblas` meets the absolute
threshold, and its benefit meets the relative basis-point threshold against `portable`. All
arithmetic is exact and checked. Overflow, a zero portable denominator with a nonzero relative
threshold, inconsistent expected-run facts, or any incomplete term rejects OpenBLAS rather than
saturating or guessing.

Candidate ordering is stable: direct, copy `A`, copy `B`. Cost wins first; exact ties prefer fewer
copies, then fewer workspace bytes, then that stable order. Selection covers the entire one-unit
plan—provider call, computation, native transition, copy, workspace, and expected reuse—not an
isolated GEMM timing. No preparation-time measurement, benchmark lookup, tuning-cache access,
machine probing, or fixed “OpenBLAS first” priority is permitted.

## Provider ownership, thread state, and failures

- Composition explicitly opens `OpenBlasLibrary` by exact name or absolute path and remains its
  lifetime owner. CPU receives only an explicitly borrowed route invocation at finalization.
- CPU analysis selects the typed `SINGLE_THREAD` configuration from immutable inputs. The caller
  must install provider thread count one before finalization, exclude competing OpenBLAS
  setters/native users for the complete
  prepared-use lifetime, keep the handle open, and perform any desired restoration after all
  prepared work is quiescent. This task adds no discovery, global registry, hidden singleton,
  per-call setter, automatic restoration, lock, active-call counter, or claim of isolation across
  other handles/class loaders/native consumers.
- Finalization and each cold bind fail if a selected native route has no invocation, the borrowed
  provider is closed, or the observed thread count is not one. These checks
  do not make a query atomic with later native work; the external no-race contract remains
  mandatory.
- A selected native plan never falls back during finalization, binding, or execution. Closure,
  thread-state drift, inaccessible storage, wrong type/layout/provenance, misalignment, overlap,
  or provider failure propagates as a stable CPU binding/execution failure before writes where
  possible. Runtime performs no route lookup or reselection.
- Prepared recipes strongly retain the invocation adapter and therefore the provider reference,
  but they do not prevent the caller from closing it incorrectly and never close it themselves.

## Numerical conformance boundary

The Model permits floating MATMUL reassociation and fused multiply-add, so the OpenBLAS route need
not be bitwise identical to the sequential portable reference. It must nevertheless preserve the
same FLOAT32/FLOAT64 result type, full contraction, IEEE operation-class behavior, and positive
zero for cases that mathematically require an unambiguous positive zero. This initial route
excludes `k == 0`, so the provider's positive-output/empty-contraction behavior is not consumed.

Native checkpoint comparisons use an independent higher-precision sum-of-products oracle. For a
finite output, let `u` be `2^-24` for FLOAT32 or `2^-53` for FLOAT64, let
`gamma = (2 * k * u) / (1 - 2 * k * u)`, and let `sumAbs` be the higher-precision sum of the
absolute exact input products. Checkpoint cases must keep `2 * k * u < 1`. The accepted absolute
error is:

```text
max(gamma * sumAbs, 4 * ulp(target-rounded higher-precision result))
```

The four-ULP term uses the target type's ULP, including its minimum positive value at zero. The
same formula applies to direct and one-copy results. Separate cases require NaN when exactly one
contributing product is NaN, positive or negative infinity when that is the sole infinite sign in
the sum, and positive zero for an all-positive-zero-times-positive input. They do not compare NaN
payloads or impose an evaluation order forbidden by Model. The test and CPU guide must name this
formula and its bounded-checkpoint role and fail rather than silently disabling assertions. Do
not claim general OpenBLAS accuracy, determinism, performance, or cross-build compatibility from
the bounded checkpoint.

## Out of scope

- Vector-vector, matrix-vector, vector-matrix, batched, right-broadcast-batched, rank greater than
  two, dynamic, unresolved, transposed, strided-batched, sparse, quantized, or complex MATMUL.
- `BFLOAT16`, mixed floating types, `INT32`, `INT64`, any conversion, or any operation other than
  bare MATMUL.
- Bias, activation, clamp, `linear`, attention, convolution lowering, higher-level BLAS pattern
  recognition, or any fused native epilogue.
- `k == 0`, `m == 0`, `n == 0`, output materialization, two-input copies, packing panels,
  transpose/reorder kernels, persistent packed weights, workspace pooling, or provider-side
  allocation.
- OpenBLAS discovery, platform filename selection, download, packaging, version probing, ILP64,
  additional symbols, provider API/source/test changes, or a default installed-library
  assumption.
- Autotuning, tuning-cache persistence, workload cache, benchmarking, profiling, performance
  claims, relaxed numerical mode, or CPU tasks 0011 through 0017.
- Another `BackendId`, Planning ownership/cost changes, shared Prepare/Runtime/Engine API changes,
  runtime route choice, dynamic fallback, service loader, registry, broad native facade, or common
  vendor abstraction.
- Model, Tensor, Compiler, Training, Config public surface, capability reporting, or generated
  portable MATMUL semantics/schema/inventory changes.
- Architecture contract, ADR, production-module dependency, shared production build, shared test
  harness, or integration-test changes. The only build/dependency change is the exact test-scoped
  backend-conformance dependency surface listed below, guarded by one focused architecture test.
  If conformance requires an Engine dependency, a shared API, or another dependency, stop and
  revise the plan instead of hiding it in this task.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially CPU routes, OpenBLAS provider,
  preparation/finalization ownership, resources, performance, and testing boundaries
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Runtime, Prepare, and backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Performance evidence and tuning](../../../../architecture/performance-evidence-and-tuning.md)
- [Partition scoring](../../../../architecture/partition-scoring.md)
- [ADR 0002: Backend-owned lowering](../../../../design/decisions/0002-backend-owned-lowering.md)
- [ADR 0008: Performance evidence and tuning boundaries](../../../../design/decisions/0008-performance-evidence-and-tuning-boundaries.md)
- [ADR 0010: Staged preparation](../../../../design/decisions/0010-staged-backend-preparation.md)
- [ADR 0011: Per-run resources](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)
- [Model MATMUL semantics](../../../modules/model/tasks/0019-matmul-semantics-and-tensor-expression.md)
- [CPU 0005A architecture reset](0005a-atomic-partition-kernel-architecture-reset.md)
- [CPU 0008F portable MATMUL](0008f-portable-matmul-execution-and-bounded-linear-epilogues.md)
- [CPU 0009F1 MATMUL route decision](0009f1-matmul-and-convolution-route-decisions.md)
- [CPU 0009G checkpoint](0009g-final-support-correctness-hygiene-and-inventory-checkpoint.md)
- [CPU 0009G1 evidence correction](0009g1-scalar-strategy-evidence-correction.md)
- [OpenBLAS provider master plan](../../openblas-provider/master-plan.md)
- [Provider 0001 loading/binding](../../openblas-provider/tasks/0001-library-loading-and-required-symbol-binding.md)
- [Provider 0002 SGEMM/DGEMM](../../openblas-provider/tasks/0002-float32-float64-row-major-gemm-invocation.md)
- [Provider 0003 thread control/checkpoint](../../openblas-provider/tasks/0003-thread-control-and-native-provider-checkpoint.md)

## Architecture constraints

- Planning continues to select only CPU ownership. CPU preparation owns complete route candidate
  construction, exact filtering, cost comparison, representation/materialization choice, selected
  route plan, and resource declarations.
- Common CPU lowering, MATMUL geometry/access facts, fusion decisions, numerical mode, portable
  realization, and representation framework remain authoritative. The OpenBLAS leaf consumes
  them and owns only provider-specific eligibility, route configuration, invocation adaptation,
  and prepared native execution.
- `backends/cpu -> backends/openblas-provider` is the existing allowed dependency. The provider
  remains a zero-project-dependency leaf and never imports CPU, Model, Planning, Prepare, Runtime,
  Tensor, graph, route, or configuration contracts.
- `testing/backend-conformance` may depend outward on `backends/cpu` in test scope and on the
  exact shared Model, Planning, Prepare, Runtime, and Backend Contract types used to construct and
  execute the existing staged boundary. This testing edge changes no production dependency
  direction. The borrowed-invocation seam must not expose an OpenBLAS-provider type in the ABI
  consumed by this test: backend conformance supplies only a fake and adds no direct dependency
  on either the provider or Engine.
- Analysis uses only complete immutable context and performs no native query or measurement.
  Finalization verifies resources but cannot change route, thread configuration, copies, or
  declarations. Runtime invokes already-bound work and sees no route vocabulary.
- Run-owned buffers/workspaces retain current CPU ownership. The OpenBLAS handle and its mutable
  process/library thread state are externally coordinated borrowed state, not a run-owned buffer,
  workspace, provider-owned lease, or hidden global.
- Portable generation remains the correctness baseline. Ineligibility selects portable during
  analysis; a failure after OpenBLAS has been selected is an error, not a late fallback.
- Any need to change authoritative architecture, dependency direction, shared lifecycle/API,
  Model semantics, provider surface, or production-module build structure is a stop condition.
  The exact backend-conformance test build and architecture-test changes listed in this task are
  required scope, not stop conditions.

## Package impact

Existing packages changed:

- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — adds typed storage/config inputs,
  route-plan invariants, cold selection handoff, and provider-aware finalization.
- `io.github.pho001.synaptik.backend.cpu.internal.lowering` — reuses the bounded representation
  candidates and permits one native carrier/layout transition through the existing affine-copy
  plan.
- `io.github.pho001.synaptik.backend.cpu.internal.executable` — permits the fixed partition
  composite to retain either a portable or native prepared child.

Package added:

- `io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas` — the reserved
  OpenBLAS leaf for exact route filtering/configuration, immutable plan, borrowed provider
  invocation, and prepared SGEMM/DGEMM execution. It owns no graph lowering or common
  representation policy.

Expected type placement:

- `CpuPartitionAnalysisInputs.OpenBlasRouteConfig` and `BoundaryStorageFact` — immutable
  route-availability/cost/thread and expected storage/alignment snapshots because they are inputs
  to CPU analysis, not provider state.
- `CpuOpenBlasRouteSelector` — field-free cold candidate filtering and complete cost comparison
  over common unit/representation facts.
- `CpuOpenBlasRoutePlan` — exact immutable selected SGEMM/DGEMM and representation facts.
- `CpuOpenBlasInvocation` — minimal technically public but unsupported `.internal` cross-package
  invocation interface/test seam whose consumed ABI names no provider type. Its CPU-private
  production adapter strongly retains one borrowed `OpenBlasLibrary`, while the backend-
  conformance test can supply a deterministic fake without loading native code; no general
  native-provider abstraction.
- `CpuOpenBlasPreparedExecutable` — route-specific cold binding and one-call execution.

Do not create `nativeblas` umbrella policy types, a vendor registry/factory, a new supported public
CPU type, a provider-side CPU adapter, or another native package.

## Affected files

Expected CPU production paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionAnalysisInputs.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparationPlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuRepresentationPlanner.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuMaterializationPlan.java`, only if the existing identity needs one exact native-transition reason
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedPartitionExecutable.java`
- new files only under
  `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/`:
  `package-info.java`, `CpuOpenBlasRouteSelector.java`, `CpuOpenBlasRoutePlan.java`,
  `CpuOpenBlasInvocation.java`, and `CpuOpenBlasPreparedExecutable.java`

Expected CPU test/checkpoint paths:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProviderPublicShapeTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuInternalPackageInventoryTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuRepresentationPlannerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedPartitionExecutableTest.java`
- new matching route-leaf tests `CpuOpenBlasRouteSelectorTest.java`,
  `CpuOpenBlasPreparedExecutableTest.java`, and non-JUnit
  `CpuOpenBlasNativeCheckpoint.java`

Backend-conformance and dependency-enforcement paths:

- `testing/backend-conformance/build.gradle.kts`
- `testing/backend-conformance/src/test/java/io/github/pho001/synaptik/testing/conformance/CpuOpenBlasRouteConformanceTest.java`
- `testing/architecture-tests/src/test/java/io/github/pho001/synaptik/testing/architecture/BackendConformanceDependencyContractTest.java`

The backend-conformance build keeps its existing
`implementation(project(":modules:backend-contract"))` edge and adds exactly these test
dependencies, in this order:

```kotlin
testImplementation(project(":modules:model"))
testImplementation(project(":modules:planning"))
testImplementation(project(":modules:runtime"))
testImplementation(project(":modules:prepare"))
testImplementation(project(":backends:cpu"))
```

Its Java compile and test tasks add `jdk.incubator.vector`, matching the CPU module because the
conformance test crosses CPU's existing internal preparation types. The new architecture test
locks this exact project-dependency surface, requires the CPU edge to remain test-scoped, and
rejects any direct OpenBLAS-provider, Engine, or other dependency drift. Keeping provider types
out of the fake seam's consumed ABI lets the ordinary conformance source remain provider-agnostic
and native-free; the CPU-local explicit checkpoint separately owns real-provider composition.

Documentation/planning paths:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`, only if current backend-route/OpenBLAS entries cannot express the final
  CPU-route/thread-lifecycle distinction accurately
- this task
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

Review-only paths include current MATMUL lowerer/IR/reference/generated tests, buffer/workspace
representations, provider source/tests/tasks, CPU/provider Gradle files, Model/Compiler/Training/
Config/Prepare/Runtime APIs, architecture/ADRs and all other architecture-test sources, all other
backend-conformance sources, integration tests, and portable evidence resources. They remain
byte-for-byte unchanged unless this task is first returned to planning.

## Maximum scope

At most 29 paths:

- 12 CPU production paths: seven existing paths (including the conditional materialization path)
  and five new OpenBLAS-leaf paths;
- nine CPU test/checkpoint paths;
- three backend-conformance/build/dependency-enforcement paths;
- at most two explanatory documentation paths; and
- three planning paths.

If `CpuMaterializationPlan.java` or `docs/glossary.md` does not need an edit, the final scope is
smaller. Compatibility constructors belong in the listed preparation inputs/plan so existing
tests do not require mechanical edits. A need for another production/test path, shared module,
provider path, Gradle file, architecture/conformance/integration source, or more than 29 total
paths is a planning stop, not implied scope.

## Acceptance criteria

1. Default CPU analysis and every ineligible, unqualified, missing-cost, overflow, tied, or
   insufficient-benefit case select the unchanged portable plan and declare no OpenBLAS-only
   resource.
2. Exact filtering admits only the tabled rank-two, positive, same-type FLOAT32/FLOAT64 bare
   MATMUL subset. Tests independently reject every excluded rank, batch/broadcast, type/promotion,
   empty dimension, oversized dimension, epilogue/fusion, layout, output carrier, alignment,
   copy-count, numerical-mode, and provider-qualification boundary.
3. Direct, copy-`A`, and copy-`B` candidates reuse common lowering and the existing typed
   materialization/resource framework. Selection is deterministic; two-input and output copies
   are absent; workspaces are exact-sized, aligned, run-owned, deduplicated, and included in the
   plan and cost.
4. The immutable plan retains both the common selected portable realization and exactly one
   optional OpenBLAS plan, with constructor invariants that make route/configuration/
   representation/resource disagreements unrepresentable. Finalization cannot change the
   selection.
5. Whole-plan arithmetic follows the specified formula and thresholds with checked values.
   Table-driven tests cover direct and copied wins/losses, expected-run reuse, absolute and
   relative thresholds, ties, candidate tie-breaking, byte ceilings, and every overflow/fail-
   closed branch. No measurement or fixed vendor priority occurs in preparation.
6. Finalization requires a matching open borrowed invocation and selected thread count, generates
   only selected copy artifacts, and creates the direct native executable or existing copy-first
   composite. Missing/closed/drifted provider state fails deterministically and never produces a
   portable fallback after selection.
7. Cold binding validates exact FLOAT32/FLOAT64 native effective segments, accessibility,
   alignment, spans, output writability, workspaces, and all write overlaps before mutation. Hot
   execution performs exactly one typed provider GEMM call and no graph/IR interpretation, route
   lookup, type/carrier/layout dispatch, allocation, boxing, reflection, map/string dispatch,
   thread setter/query, synchronization, or copy.
8. Native-free tests prove exact `m`, `n`, `k`, `alpha`, `beta`, segment identity, call count,
   failure propagation, no call after failed bind/copy validation, repeated-run reuse, and direct
   versus one-copy semantic parity for both types.
9. `CpuOpenBlasRouteConformanceTest` exercises one representative direct native-storage MATMUL
   for both FLOAT32 and FLOAT64 through `PrepareContext` -> `CpuPartitionPreparer.analyze` ->
   `BackendPartitionFinalization` -> `CpuPartitionFinalizer.finalizePartition` -> cold binding ->
   bound execution. It builds the exact `PreparedMemoryPlan` and assignments from the analyzed
   requirements, then binds arena-backed `MemorySegmentStorage` through `CpuBorrowedBuffer` in a
   `RunState`. It supplies explicit qualifying storage/cost/thread facts, proves the selected route
   is `OPENBLAS`, injects a fake open single-thread invocation, checks the exact one-call
   dimensions/scalars/segment identities, and verifies the output against independently computed
   row-major MATMUL results. It uses no Engine, installed library, native lookup, native call,
   skip, environment variable, or system-property activation.
10. The backend-conformance build has only the listed project dependencies and CPU remains a
    test-scoped edge. `BackendConformanceDependencyContractTest` fails closed for an Engine,
    any OpenBLAS-provider edge, a production-scoped CPU edge, missing shared test dependency,
    reordered dependency, or otherwise unlisted project edge.
11. The explicit native checkpoint consumes exactly one caller-supplied absolute compatible
   library path, installs/restores the selected thread count in a `finally` path under the stated
   exclusive-use precondition, drives CPU analysis through prepared execution for FLOAT32 and
   FLOAT64 direct and one-copy cases, confirms OpenBLAS selection, and passes the documented
   finite/special numerical checks. It never discovers, downloads, silently skips, or converts a
   missing/incompatible binary into a passing result.
12. `CpuCapabilityProvider` remains the sole supported CPU API; `BackendId("cpu")` and capability
    reporting are unchanged; the provider remains unchanged; public/internal package inventory
    records exactly the new unsupported leaf types.
13. A distinct clean documentation-focused context reviews the final diff, finalizes every
    affected Javadoc and package summary under the General and API/Javadoc profiles, updates the
    CPU guide under the Backend Guide profile, resolves glossary impact, and synchronizes this
    task/master/roadmap under the Planning profile. It records reasoned no-change conclusions for
    public Tensor/Compile/Training/Config APIs, Model semantics, provider API, shared Prepare/
    Runtime, architecture/ADRs other than the named dependency test, all other backend-
    conformance sources, integration, other Gradle builds, and other modules.
14. CPU and backend-conformance tests, the explicit native checkpoint, the architecture suite,
    repository-wide tests, CPU Javadoc/render inspection, Markdown/scope/status checks, and
    whitespace validation pass before the task becomes `Complete`.

## Tests / validation

During implementation, use the focused affected-module command as needed:

```bash
./gradlew :backends:cpu:test :testing:backend-conformance:test \
  :testing:architecture-tests:test
```

The CPU suite must include selector/config/plan invariants, preparation/finalization,
representation and exact resource declarations, composite sequencing, binding/failure-before-
write, fake provider invocation, lifecycle/thread drift, inventory/public shape, and unchanged
portable fallback coverage. The backend-conformance suite must include the two-type direct-route
preparation/finalization/binding/execution case above. The architecture suite must enforce the
exact new test-only dependency surface. All ordinary tests require no native access or installed
OpenBLAS.

Compile the explicit checkpoint and run it separately with the caller-supplied binary:

```bash
./gradlew :backends:cpu:testClasses
java --enable-native-access=ALL-UNNAMED --add-modules jdk.incubator.vector \
  -cp backends/cpu/build/classes/java/test:backends/cpu/build/classes/java/main:backends/openblas-provider/build/classes/java/main:modules/model/build/classes/java/main:modules/config/build/classes/java/main:modules/planning/build/classes/java/main:modules/runtime/build/classes/java/main:modules/prepare/build/classes/java/main:modules/backend-contract/build/classes/java/main:modules/trace/build/classes/java/main \
  io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas.CpuOpenBlasNativeCheckpoint \
  <ABSOLUTE_COMPATIBLE_OPENBLAS_LIBRARY>
```

On Windows, use `;` as the class-path separator and pass an absolute compatible DLL path. The
checkpoint command is mandatory evidence for completion; an absent supplied library leaves task
0010 incomplete.

Because this task adds a project dependency, changes a Gradle build, and spans CPU plus two
testing modules, the implementation context runs one final repository-wide suite after executable
code, build files, and tests stabilize:

```bash
./gradlew test
```

This final repository command may serve as the final CPU, backend-conformance, and architecture
evidence instead of repeating an unchanged successful focused command. Record task/test counts
for those three projects from that run. The separate explicit real-native checkpoint remains
required because it is intentionally outside JUnit.

The documentation-focused context reuses the successful repository-wide tests and native
checkpoint unless it changes executable Java, build logic, or tests, then runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
git status --short -uall
```

It renders and inspects the changed generated Javadoc pages; checks local Markdown links and
anchors, heading order/uniqueness, balanced fences, LF/final newlines, terminology, exact path
scope, provider/portable no-change, and public/internal inventory; and confirms CPU 0010 alone was
`Ready` before implementation and CPU 0011 through 0017 remain `Draft`. It also confirms the
backend-conformance test is native-free and Engine-free, the one architecture-test addition is
limited to the new dependency surface, and no successful Java suite is duplicated without stale
evidence. Integration validation remains deferred until Engine provides the public end-to-end
lifecycle.

No benchmark, tuning workflow, provider suite rerun, or generated portable evidence regeneration
is required. The recorded `./gradlew test` run is the task's repository-wide gate; CI remains the
final independent gate.

## Dependencies

- Complete [CPU 0005A](0005a-atomic-partition-kernel-architecture-reset.md), which reserves the
  route leaf, keeps common lowering authoritative, and establishes the portable fallback.
- Complete [CPU 0009G1](0009g1-scalar-strategy-evidence-correction.md), after completed CPU 0009G,
  which closes the current portable checkpoint and task frontier.
- Complete [OpenBLAS provider](../../openblas-provider/master-plan.md), including completed tasks
  [0001](../../openblas-provider/tasks/0001-library-loading-and-required-symbol-binding.md),
  [0002](../../openblas-provider/tasks/0002-float32-float64-row-major-gemm-invocation.md), and
  [0003](../../openblas-provider/tasks/0003-thread-control-and-native-provider-checkpoint.md).
- Current `CpuMatmulLowering`, representation/materialization plans, run-owned native
  buffer/workspace types, staged CPU finalization, portable MATMUL reference/tests, and the
  already-declared `backends/cpu -> backends/openblas-provider` Gradle dependency.
- The existing included `testing/backend-conformance` and `testing/architecture-tests` projects.
  Backend conformance currently has only its marker type and Backend Contract dependency, so this
  task owns the first concrete CPU conformance test and its exact test-scoped dependencies.

All prerequisites are complete and consistent. The provider checkpoint recorded a compatible
32-bit-`blasint` SGEMM/DGEMM/thread-control binary, so the explicit task checkpoint has an
established contract even though ordinary tests remain native-free. The completed provider
master plan's statement that no CPU task was yet Ready records that provider milestone's
completion-time handoff; the CPU master plan and repository roadmap own the current CPU frontier.

## Follow-up tasks

- CPU 0011 remains the next `Draft` row and owns distinct oneMKL BLAS/VML peer routes only after
  task 0010 completes.
- Broader OpenBLAS shapes, batches, transposes, epilogues, two-input/output materialization,
  packing, thread candidates, or persistent tuning evidence are not implied follow-ups. Add an
  ordered bounded task only after a concrete use case and compatible evidence exist.
- CPU 0016 may later consume compatible measured selected-route evidence; this task creates no
  cache or opaque tuning handoff. CPU 0017 later owns explicit relaxed numerical candidates.

## Architecture impact

Expected architecture impact: None. This implements the already-reserved OpenBLAS CPU route using
the existing production dependency and staged CPU-owned analysis/finalization boundary. The new
backend-conformance edges are test-only outward dependencies from a testing project, and the
focused architecture test enforces that they do not introduce Engine, production-scoped
CPU, any direct OpenBLAS-provider edge, or other dependency drift.

If implementation requires an architecture rule, ADR, production-module edge, dependency beyond
the exact listed backend-conformance test surface, provider contract, shared Prepare/Runtime/
Engine surface, global thread-state coordinator, or backend identity change, stop and report the
exact conflict before editing outside this specification.

## Implementation prompt

Use this prompt in a separate agentic task/thread:

```text
You are the clean implementation agent for Synaptik CPU 0010. Do not use GSD, commit, or push.
Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, the CPU master plan, and
docs/planning/backends/cpu/tasks/0010-narrow-openblas-blas-compatible-native-route.md, plus every
architecture, prerequisite task, source, test, and documentation contract directly referenced by
that task. Implement the task exactly within its 29-path ceiling. Stop for any architecture,
provider, shared-lifecycle, or scope conflict instead of inventing a new boundary.

Implement the required native-free backend-conformance test and exact dependency architecture
test without adding Engine. After Java, build, conformance, repository-wide, and required native
validation stabilize, hand the exact diff and recorded evidence to a distinct clean
documentation-focused agent following documentation-rules.md. That agent must finalize affected
Javadocs, package documentation, CPU guide, glossary impact, task, master plan, roadmap, and
documentation validation without repeating successful Java tests unless it changes executable
behavior or records a concrete stale-evidence risk. Do not mark CPU 0010 Complete until the
backend-conformance test, repository-wide suite, native checkpoint, and documentation pass all
succeed.
```

## Local decisions

- Start with bare rank-two same-type FLOAT32/FLOAT64 MATMUL because it is the exact semantic
  intersection of current `CpuMatmulLowering` and the provider's row-major non-transposed
  SGEMM/DGEMM contract. Vector/rank-one/batch/epilogue support would require additional mapping or
  call structure and is deliberately excluded.
- Require positive `m`, `n`, and `k`. Portable execution remains responsible for every empty
  case, avoiding a numerical claim around the provider's `k == 0` beta behavior and pointless
  native transitions for output-empty calls.
- Reuse one existing affine-copy materialization for an input carrier/layout transition. This
  makes native transition cost and resource ownership real without adding packing, output copy,
  or a second representation system.
- Keep an explicit expected native/alignment fact separate from `CarrierAccess.MEMORY_SEGMENT`.
  The latter can represent a read-only heap segment whose heap base is not observable and cannot
  prove provider-native storage.
- Keep the live provider out of analysis. A borrowed invocation enters only finalization, just as
  the current worker group is an explicitly borrowed finalization resource.
- Select only `SINGLE_THREAD` and verify an externally installed provider count of one without
  setting/querying it in the hot invocation. This is the smallest honest initial policy over
  process/library-global mutable state; it does not pretend CPU can coordinate unrelated native
  users or qualify arbitrary thread configurations.
- Retain the selected portable route plan even when OpenBLAS wins, but permit no post-analysis
  fallback. Retention preserves a complete semantic alternative for analysis/evidence; immutable
  prepared execution preserves the Runtime boundary.
- Use deterministic configured cost units and strict benefit thresholds. They are safe heuristic
  selection inputs, not benchmark evidence or a performance claim.
- Put the first concrete CPU backend-conformance case in the existing conformance package and
  exercise the staged internal boundary directly. A test-scoped CPU dependency is the smallest
  honest integration because Engine does not exist; direct provider use would bypass the route,
  and reflection or copied CPU test fixtures would not prove the real preparation/finalization
  path. Guard the exact dependency set in the architecture suite because the build edge changes.

## Known limitations

- Only one bare positive rank-two same-type FLOAT32/FLOAT64 MATMUL can use OpenBLAS.
- At most one input is copied. Output copy, two-input copies, transpose, packing, batching,
  broadcasting, epilogues, and empty dimensions stay portable.
- Qualification and thread-state exclusion are explicit caller/composition responsibilities.
  CPU cannot prevent another handle, class loader, or arbitrary native caller from changing the
  shared OpenBLAS state or closing an aliased owner incorrectly.
- The native checkpoint qualifies only the supplied binary, process, ABI, selected thread count,
  and bounded cases. It establishes no universal OpenBLAS build, platform, determinism, or
  performance guarantee.
- No public configuration or Engine integration exists yet. The route is exercised through the
  current internal staged CPU preparation/finalization contracts in both CPU-local and backend-
  conformance tests.

## Validation evidence

- The implementation handoff recorded
  `./gradlew :backends:cpu:test :testing:backend-conformance:test
  :testing:architecture-tests:test` passing after executable stabilization.
- The final repository gate `./gradlew test` ran exactly once and passed in 2m53s with 63
  actionable tasks. The resulting XML reports were independently inspected by documentation
  context `/root`: CPU passed 942 tests with zero failures/errors and 28 existing opt-in skips;
  backend conformance passed 3 tests with zero failures/errors/skips; architecture passed 7 tests
  with zero failures/errors/skips.
- The required explicit CPU native checkpoint first rejected the stale exact provider-version
  path rather than discovering or substituting another library. The coordinator then supplied
  `/opt/homebrew/Cellar/openblas/0.3.34/lib/libopenblasp-r0.3.34.dylib`; the exact documented
  checkpoint command passed and printed
  `CPU OpenBLAS native checkpoint passed; restored thread count 16`. It exercised FLOAT32 and
  FLOAT64 direct and one-copy prepared routes under the documented bounded gamma/ULP and
  exceptional-class checks.
- `javap -public` confirmed that `CpuOpenBlasInvocation` exposes only `isOpen`, `threadCount`,
  `sgemm`, and `dgemm`, using JDK `MemorySegment` and no provider type. Production-source search
  found `OpenBlasLibrary` only in the finalization adapter in `CpuOpenBlasRouteSelector`; the
  explicit native checkpoint is the only test-source provider reference. Backend conformance is
  provider-free, Engine-free, and native-free.
- Documentation context `/root` ran `./gradlew :backends:cpu:javadoc` after its initial Javadoc
  pass and reran it after the final failure-contract additions; both passed. The final run was
  `BUILD SUCCESSFUL` in 3s with 11 actionable tasks. Its 94 total warnings comprise the expected
  Java 26 incubating-Vector notices and pre-existing missing-parameter warnings on older
  compatibility constructors in `CpuPartitionLowering` and `CpuPartitionPreparationPlan`; none
  names a new OpenBLAS declaration or this task's changed Javadocs.
- The changed generated pages for the OpenBLAS package and its four types, analysis inputs,
  preparation plan/preparer/finalizer, representation planner, and partition composite were
  rendered to plain text with Pandoc and inspected for descriptions, parameter/result/failure
  sections, ownership, route selection, provider-free ABI, copy resources, and late-fallback
  exclusions.
- Documentation context `/root` ran
  `./gradlew :testing:architecture-tests:test`; it passed as up-to-date in 506ms with 3 actionable
  tasks, reusing the still-current successful architecture result because no executable test or
  build file changed during documentation review.
- The final targeted Markdown checker passed for the CPU guide, glossary, task, CPU master plan,
  and roadmap: local files/anchors, heading uniqueness, balanced fences, LF endings, final
  newlines, and trailing whitespace are valid. Final status checks show CPU 0010 `Complete`, CPU
  0011 as the next `Draft` CPU frontier, and CPU 0012 through 0017 still `Draft`.
- Final scope inspection found exactly 24 changed/new paths, within the 29-path ceiling and exact
  task allowlist. `git diff --check` passed. No ordinary Java suite, provider suite, benchmark,
  tuning workflow, generated portable evidence, or integration test was rerun by the
  documentation context.

## Implementation notes

- CPU analysis inputs now carry disabled-by-default expected native-storage facts and a typed,
  immutable qualification/thread/cost snapshot. The selector consumes only the complete common
  portable MATMUL plan and fails closed for every uncertain eligibility or checked-cost branch.
- The selected route plan records exact type, `m`, `n`, `k`, boundary positions, single-thread
  configuration, direct/copy representation, optional existing affine materialization and exact
  workspace, and checked whole-plan cost/benefit facts. It retains no provider or run resource.
- Finalization borrows an open single-thread provider-free invocation. Direct plans create one
  native child; copied plans reuse the existing copy-first atomic composite. Cold binding checks
  provider state, carriers, type, native storage, accessibility, alignment, spans, writability,
  and all write overlaps before mutation. Hot work performs exactly one typed GEMM call and never
  falls back or reselects a route.
- Ordinary tests use fakes only. Backend conformance constructs the existing staged CPU boundary
  without Engine or the provider. The explicit checkpoint alone adapts the caller-supplied real
  provider and owns the bounded installed-library evidence.
- Documentation context `/root` independently read the architecture, named ADRs, provider plans,
  documentation profiles, final implementation/test/build diff, and recorded reports. It
  finalized only Javadocs/package documentation in already changed production files plus the CPU
  guide, glossary, this task, CPU master plan, and roadmap; executable semantics were preserved.
- No-change review: `CpuCapabilityProvider` remains the sole supported CPU API and
  `BackendId("cpu")`/capability reporting are unchanged. Public Tensor, Compile, Training, and
  Config APIs; Model MATMUL semantics; provider API/source/tests/tasks; shared Prepare and Runtime;
  Engine; architecture contracts and ADRs; other backend-conformance sources; integration tests;
  other Gradle builds; other modules; and portable MATMUL/generated schema and evidence remain
  unchanged. The sole architecture addition enforces only the exact backend-conformance
  test-dependency surface.

## Completion summary

- Completed changes: implemented the qualified narrow OpenBLAS CPU MATMUL selection, exact direct
  or one-affine-input-copy resource path, finalization-only borrowed invocation, fail-before-write
  binding, one-call prepared execution, native-free CPU/conformance coverage, exact dependency
  enforcement, explicit native checkpoint, and synchronized Javadoc/guide/glossary/planning.
- Files changed or created: 11 CPU production paths, 5 CPU test/checkpoint paths, the
  backend-conformance build and conformance test, one architecture dependency test, the CPU guide,
  glossary, this task, CPU master plan, and roadmap—24 paths total. Final status/diff inspection
  recorded the exact list, all within [Affected files](#affected-files), with no provider,
  architecture-contract, ADR, integration, shared-module, or unrelated documentation path.
- Tests and validation: focused CPU/conformance/architecture tests passed; the sole final root
  test passed with the recorded 942/3/7 project counts; the caller-supplied OpenBLAS 0.3.34
  checkpoint passed and restored count 16; CPU Javadoc, generated-page inspection, ABI/source
  checks, architecture rerun, Markdown/status/scope checks, and whitespace validation passed.
- Documentation impact: current guide and glossary text now distinguish portable MATMUL, the
  narrow CPU-owned OpenBLAS route, and the provider-only invocation/lifecycle boundary. Every
  affected OpenBLAS/configuration/plan/finalization/execution Javadoc and the native package
  summary records ownership, constraints, resources, failures, and the no-late-fallback rule.
- Unresolved issues: none within task 0010. Integration remains deliberately deferred because no
  public Engine lifecycle exists; broader native shapes, providers, and tuning remain separate
  Draft work.
- Required follow-up: none for task 0010. CPU 0011 remains the next Draft CPU frontier.

Status: Complete
