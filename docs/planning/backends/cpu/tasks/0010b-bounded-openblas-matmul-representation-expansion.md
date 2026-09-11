# Task 0010B: Bounded OpenBLAS MATMUL Representation Expansion

## Status

Ready

## Goal

Expand the existing CPU-owned OpenBLAS route from direct matrices or one copied input to the
complete bounded representation set for one positive, rank-two, same-type FLOAT32/FLOAT64 bare
`MATMUL`. CPU preparation may canonicalize either or both inputs before the provider call and may
compute into one canonical output workspace before copying to the logical result. The provider
continues to receive exactly one dense row-major, non-transposed SGEMM or DGEMM call.

The selected plan must account for the complete per-run transition: workspace allocation and
binding, every input copy, the GEMM call, optional output copy, workspace bytes, and expected run
count. Portable execution remains a valid preparation candidate, and every incomplete,
ineligible, tied, overflowing, or insufficient-benefit native candidate fails closed to that
portable plan during analysis. A selected native recipe never falls back later.

## Scope

- Preserve the exact one-node, one-unit, bare rank-two MATMUL semantic boundary from CPU 0010:
  positive static `m`, `n`, and `k`; same-type FLOAT32 or same-type FLOAT64 inputs/result; exact
  default numerical mode; no rank promotion, batch prefix, broadcast batch, or epilogue.
- Retain direct canonical native matrices where their current storage and layout facts qualify.
- Reuse the existing typed access bindings and generated affine-copy machinery to canonicalize
  zero, one, or both inputs into distinct aligned native run-owned workspaces.
- Add the smallest OpenBLAS-route-owned output-copy plan needed to compute into one canonical
  aligned native workspace and copy the represented result into an admitted logical output
  binding. Do not redefine the general external-read `CpuMaterializationPlan` as an output plan.
- Represent and validate all eight bounded copy masks over left input, right input, and output:
  direct; left; right; left+right; output; left+output; right+output; and left+right+output. Only
  masks required by the actual boundary facts are candidates.
- Extend OpenBLAS finalization and its immutable prepared recipe so cold binding validates every
  buffer, workspace, generated copy, and provider precondition before the first write. Hot
  execution orders all selected input copies, exactly one GEMM, and the optional output copy.
- Extend the CPU-private OpenBLAS cost snapshot with explicit non-negative representation terms
  for per-workspace allocation/binding, workspace bytes, copy-in invocation/elements, and
  copy-out invocation/elements. Keep provider/GEMM route terms separate.
- Preserve stable shared workspace declarations in the immutable prepared plan. Physical
  workspaces remain run-owned and separately allocated for each `RunState`; copied values are not
  persistent prepared resources and are rebuilt on every run.
- Add proportional native-free unit and backend-conformance coverage for the expanded
  representation boundary and extend the explicit real-native checkpoint for the supplied
  OpenBLAS 0.3.34 binary.
- Finalize affected Javadoc, package documentation, the CPU backend guide, glossary impact, this
  task, CPU master plan, and roadmap through a distinct clean documentation-focused context after
  executable Java stabilizes.

## Out of scope

- Any OpenBLAS-provider API, ABI, symbol, implementation, test, package, build, or documentation
  change. Provider calls remain row-major, non-transposed SGEMM/DGEMM with `alpha = 1` and
  `beta = 0`.
- Rank-one, rank greater than two, batch prefixes, batch broadcasting, repeated GEMM scheduling,
  strided-batched calls, dynamic Shapes, unresolved layouts, sparse, quantized, or complex
  MATMUL.
- `BFLOAT16`, mixed floating types, integral types, promotion, result conversion, or any operation
  other than bare MATMUL.
- Bias, activation, clamp, `linear`, attention, convolution, fused epilogues, provider transpose
  flags, provider packing APIs, panel packing, persistent packed weights, or workspace pooling.
- Empty output or contraction geometry. `m == 0`, `n == 0`, or `k == 0` remains on the portable
  plan; this task adds no provider call or copy-only native route for those cases.
- Changing CPU 0010A discovery/loading, treating `LOADED` as `QUALIFIED`, binary qualification or
  fingerprinting, public Config/Engine composition, or provider lifetime ownership.
- Changing the current externally coordinated `SINGLE_THREAD` provider rule, adding thread
  candidates, setting/restoring thread count in prepared execution, or introducing a CPU-wide
  concurrency coordinator. CPU 0010C owns those decisions.
- Model autotuning, tuning-cache lookup or mutation, measurement during preparation, general
  autotuning, persistent representation evidence, benchmarking, or performance claims. CPU 0010E
  and `tools/tuning` own those later capabilities.
- Public Tensor, Config, Compile, Engine, Runtime, Prepare, Training, or CPU APIs; capability
  reporting; another `BackendId`; Planning ownership/cost logic; architecture or dependency
  changes; generated MATMUL implementations; or unrelated generator work.
- Generalizing ordinary CPU 0008E representation selection, its `CO_CONSUMED_PAIR` rejection, or
  its maximum two external-read copies. The expanded copy masks are private to the selected
  OpenBLAS MATMUL route.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially Planning ownership, staged
  preparation, Runtime resource ownership, CPU routes, the OpenBLAS provider, and performance
  evidence boundaries
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Runtime, Prepare, and backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Performance evidence and tuning](../../../../architecture/performance-evidence-and-tuning.md)
- [CPU backend guide](../../../../backend-guide/cpu-backend.md)
- [CPU 0005B access plans](0005b-universal-access-plans-and-right-aligned-broadcasting.md)
- [CPU 0005D materialization evidence](0005d-materialization-specialization-and-persistence-evidence-gate.md)
- [CPU 0006 affine views and boundary materialization](0006-portable-static-affine-views-and-boundary-materialization.md)
- [CPU 0008E bounded multi-input materialization](0008e-bounded-multi-input-materialization-and-representation-reuse.md)
- [CPU 0008E1 shared partition-DAG adoption](0008e1-shared-partition-dag-adoption-and-reconstruction-removal.md)
- [CPU 0008F portable MATMUL](0008f-portable-matmul-execution-and-bounded-linear-epilogues.md)
- [CPU 0009F1 MATMUL route decision](0009f1-matmul-and-convolution-route-decisions.md)
- [CPU 0010 narrow OpenBLAS route](0010-narrow-openblas-blas-compatible-native-route.md)
- [CPU 0010A discovery and composition foundation](0010a-automatic-openblas-discovery-and-internal-composition-foundation.md)
- [OpenBLAS provider master plan](../../openblas-provider/master-plan.md)
- [Provider 0001 loading and binding](../../openblas-provider/tasks/0001-library-loading-and-required-symbol-binding.md)
- [Provider 0002 row-major GEMM](../../openblas-provider/tasks/0002-float32-float64-row-major-gemm-invocation.md)
- [Provider 0003 thread control and checkpoint](../../openblas-provider/tasks/0003-thread-control-and-native-provider-checkpoint.md)

## Architecture constraints

- Planning continues to select only `BackendId("cpu")`. CPU analysis alone constructs, filters,
  costs, and selects the OpenBLAS route and physical representation plan.
- Common CPU MATMUL lowering, exact rank-two geometry, typed access bindings, numerical mode,
  portable realization, and retained portable plan remain authoritative. The OpenBLAS leaf must
  consume those facts without graph reconstruction or semantic reinterpretation.
- Backend analysis selects the complete route, copy mask, costs, and exact shared resource
  requirements before slot assignment. Finalization may load existing affine-copy artifacts and
  build the immutable recipe, but it cannot change the selection or add a resource.
- Runtime allocates and owns each run's declared workspaces and executes only the already-prepared
  sequence. It must not inspect layouts, select copies, select a route, discover a provider, or
  retry with portable execution.
- `backends/openblas-provider` remains a low-level leaf. The existing dependency direction
  `backends/cpu -> backends/openblas-provider` and the current provider surface remain unchanged.
- CPU 0010A discovery remains cold, bounded, and separate from qualification. Loading still does
  not authorize native selection.
- Portable plans remain complete semantic alternatives during analysis. Native ineligibility
  retains portable; provider closure, thread drift, representation mismatch, copy failure, or
  provider failure after native selection is an error.
- Generated affine copies reuse the existing Java 26 Class-File machinery. If implementation
  discovers that an emitter, generated schema, or generated computation loop must change, stop
  and return the task to planning; do not introduce unrelated generator work under this task.
- Any need for a shared Prepare/Runtime contract, provider API, public Config/Engine API, module
  edge, architecture rule, or more general materialization contract is a stop condition.

## Current evidence and baseline

- CPU 0008F supplies complete static portable MATMUL geometry, access bindings, reference
  semantics, and the retained portable plan for every form in this task.
- `CpuMatmulLowering` already accepts resolved non-negative rank-two layouts, computes exact
  offsets/strides/spans, and rejects non-injective output layouts. `CpuAccessPlan.Binding` records
  the complete logical range and accessed element interval.
- `CpuRepresentationPlanner.openBlasInputMaterialization(...)` already builds one exact generated
  affine input copy into a canonical native workspace, but its fixed workspace identity and CPU
  0010 selector currently permit only one input copy.
- `CpuMaterializationPlan` is specifically an external-read source-to-workspace plan with consumer
  coordinates. It cannot honestly represent a workspace-to-logical-output copy.
- `CpuPreparedPartitionExecutable` currently schedules zero through two general representation
  copies before all computation children. It has no post-computation copy phase. An OpenBLAS-
  specific sequence or a narrowly generalized phase contract is therefore required; the
  implementation must not disguise copy-out as an input materialization.
- `CpuOpenBlasRouteSelector` currently admits only canonical zero-offset output, requires the
  other input to be direct before considering one copied input, and ranks direct/copy-left/copy-
  right by complete checked cost.
- `CpuOpenBlasPreparedExecutable` currently consumes zero or one workspace, substitutes only one
  copied input, and performs one non-transposed provider call. Its provider/lifecycle/type/span
  validation is the baseline to preserve.
- `CpuContiguousWorkspace` already provides aligned native run-owned storage suitable for both
  canonical inputs and canonical output. Shared Prepare already assigns stable workspace slots
  from exact backend declarations; no shared resource contract change is evidenced.
- CPU 0010A adds only discovery metadata and an owned provider session. Neither source nor its
  tests participate in representation selection and they remain unchanged.

## Exact semantic and representation contract

### Logical geometry and empty cases

The only native semantic form remains:

```text
A[m,k] MATMUL B[k,n] -> C[m,n]
```

`m`, `n`, and `k` are static, positive, and no greater than `Integer.MAX_VALUE`; all matrix
element counts, byte counts, accessed spans, workspace totals, and cost arithmetic must succeed
exactly. `A`, `B`, and `C` have one exact common FLOAT32 or FLOAT64 type. The occurrence is bare,
rank two, exact/default, and has one output.

If any dimension is zero, if a dimension exceeds the provider integer range, or if checked
geometry overflows, no OpenBLAS candidate exists. Portable execution owns `m == 0`, `n == 0`, and
`k == 0`, including the positive-zero empty-contraction result. No workspace is declared for a
rejected native candidate.

### Accepted affine access geometry

For each boundary, the logical extents must equal its matrix extents, the binding must cover the
complete half-open logical range `[0, elementCount)`, its non-negative base offset and effective
strides must address only the declared referenced element span, and its access kind must be
`READ` for `A`/`B` or `WRITE` for `C`. The output binding must retain the current MATMUL lowerer's
injectivity proof.

In this task, a **canonical matrix** has zero base offset, rank-two extents `[rows, columns]`,
effective element strides `[columns, 1]`, dense-linear access, and an exact referenced span of
`rows * columns`. It is direct only when the expected boundary fact and cold-bound
representation also prove a type-width-aligned native segment, with a writable segment for `C`.

An **affine matrix view** is any other current rank-two binding with the exact complete range and
checked non-negative offset/strides/span above. Zero strides are permitted for read-only inputs
because they repeat a represented input element; they are not permitted for a multi-element
output because the existing injectivity proof rejects repeated writes. A **transpose view** is
not a provider flag or new semantic attribute: it is the affine special case whose logical axes
address a dense underlying matrix in swapped order, such as logical extents `[rows, columns]`
with strides `[1, rows]`. CPU materializes its logical row-major values before the unchanged
provider call.

Any direct-boundary failure caused by heap storage, nonzero offset, noncanonical strides, a larger
referenced span, or missing native/alignment proof may be resolved only by the corresponding
copy. A copied matrix's logical element count must fit the existing affine-copy address-table
limit and its exact bytes must fit the complete representation byte ceiling. Unresolved,
negative-stride, out-of-span, partial-range, non-injective-output, or otherwise unproved geometry
is ineligible rather than guessed.

### Copy masks, workspaces, and execution order

The selector enumerates exactly these stable masks:

| Stable rank | Representation | Prepared execution |
|---|---|---|
| 0 | direct | GEMM |
| 1 | copy left | copy `A`, GEMM |
| 2 | copy right | copy `B`, GEMM |
| 3 | copy output | GEMM to workspace, copy to `C` |
| 4 | copy left + right | copy `A`, copy `B`, GEMM |
| 5 | copy left + output | copy `A`, GEMM to workspace, copy to `C` |
| 6 | copy right + output | copy `B`, GEMM to workspace, copy to `C` |
| 7 | copy left + right + output | copy `A`, copy `B`, GEMM to workspace, copy to `C` |

Each selected copy owns one distinct exact-size, type-width-aligned native workspace declaration.
Input workspaces contain canonical row-major logical values. The optional output workspace is a
canonical row-major `[m,n]` provider destination and is the source of one generated affine copy
whose destination binding is the original logical `C` binding.

All input copies execute in left-then-right order, followed by exactly one SGEMM/DGEMM call, then
the optional output copy. Every generated copy executes exactly once per run. Finalization loads
or generates only selected affine-copy artifacts. Preparation declares no workspace for an
unselected or rejected candidate.

### Cost and deterministic selection

Let `O = m * n`, `W = m * n * k`, `IA = m * k` when left is copied, `IB = k * n` when right is
copied, `OC = m * n` when output is copied, `S` be the number of selected workspaces, `B` their
total bytes, and `R` the positive expected-run count. Extend the existing immutable route facts
with complete non-negative representation terms and compare checked costs:

```text
portable = R * (portableFixed + portablePerOutput * O + portablePerMac * W)

openblas = R * (openblasFixed + openblasPerOutput * O + openblasPerMac * W
                + workspaceAllocationAndBindingFixed * S
                + workspaceCostPerByte * B
                + copyInFixed * inputCopyCount
                + copyInPerElement * (IA + IB)
                + copyOutFixed * outputCopyCount
                + copyOutPerElement * OC)
```

`openblasFixed` continues to include the native transition/provider call and current thread
configuration cost, but it must not double-count a representation term. Every physical workspace
and every copy is per-run, so the entire parenthesized cost is multiplied by `R`; this task adds
no persistent copied representation amortized across runs. The stable prepared declarations and
copy artifacts are reusable recipes, not reusable copied data.

The candidate must be strictly cheaper than portable and meet the configured absolute and
relative basis-point thresholds. Overflow, a missing term, inconsistent expected-run fact, zero
relative denominator when a nonzero threshold applies, equality, or insufficient benefit rejects
the candidate. Candidate comparison uses, in order: lower complete OpenBLAS cost, fewer total
copies, fewer workspace bytes, then the stable mask rank above. No fixed OpenBLAS priority,
measurement, benchmark lookup, or tuning-cache access is permitted.

### Aliasing, failure, lifecycle, and concurrency

- `A` and `B` may overlap each other, including when one or both are copied.
- The logical output buffer `C` must not overlap either logical input buffer, even when the input
  is copied before GEMM. This conservative rule preserves the current functional boundary and
  avoids representation-dependent alias semantics.
- Every selected workspace must be distinct and pairwise disjoint from all selected buffers and
  other workspaces. The output workspace is never reused as an input workspace.
- Cold binding validates provider open/thread state, exact carrier/type/layout roles, segment
  accessibility, alignment, spans, output writability, generated artifact compatibility, and all
  overlap rules before any copy or GEMM writes.
- A selected plan never falls back. Input-copy, provider, or output-copy failure propagates. A
  provider failure may have partially written a direct `C`; with an output workspace, `C` is not
  touched until GEMM succeeds, although an output-copy failure need not be transactional.
- The prepared recipe, route plan, copy plans, artifacts, selections, and workspace-slot
  identities are immutable and reusable. Each run has a distinct `RunState` and distinct physical
  workspaces. Concurrent runs may reuse the same prepared recipe only while the borrowed provider
  remains open and the existing CPU 0010 external single-thread-state exclusion contract is
  satisfied; this task adds no stronger cross-run or cross-owner coordination guarantee.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu.internal.ir`
- `io.github.pho001.synaptik.backend.cpu.internal.lowering`
- `io.github.pho001.synaptik.backend.cpu.internal.prepare`
- `io.github.pho001.synaptik.backend.cpu.internal.executable`
- `io.github.pho001.synaptik.backend.cpu.internal.memory`
- `io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas`

Packages added or changed:

- `io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas` — extends only the
  existing route's representation facts and prepared sequencing.
- Existing lowering, preparation, or executable packages may receive the minimum reusable seam
  needed to build and bind existing affine-copy artifacts; no new package is added.

Type placement:

- `CpuOpenBlasRoutePlan` — owns the exact selected copy mask, ordered input-copy plans, optional
  output-copy plan, workspace declarations, complete costs, and provider geometry.
- `CpuOpenBlasOutputCopyPlan` — proposed route-local immutable workspace-to-logical-output affine
  copy contract because the general `CpuMaterializationPlan` specifically owns external reads and
  consumer workspaces. If implementation finds a smaller equally typed route-local name, update
  this task before creating it.
- `CpuOpenBlasRouteSelector` — continues to own exact eligibility, bounded enumeration, checked
  whole-plan costing, and deterministic selection.
- `CpuOpenBlasPreparedExecutable` — owns the immutable copy-in/GEMM/copy-out recipe or the native
  GEMM child within one narrowly typed OpenBLAS sequence.
- `CpuRepresentationPlanner` — may expose focused constructors for existing input-copy and
  output-copy affine plans; it must not select the OpenBLAS route or generalize ordinary 0008E
  policy.

## Affected files

Expected CPU production paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionAnalysisInputs.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparationPlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuRepresentationPlanner.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuMaterializationPlan.java`, only if an invariant/Javadoc clarification is required without changing its external-read ownership
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedPartitionExecutable.java`, only if a narrow reusable before/after-copy binding seam is smaller than route-local sequencing
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasRoutePlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasRouteSelector.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasPreparedExecutable.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/package-info.java`
- new `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasOutputCopyPlan.java`, only if the output plan is not a nested immutable value in an existing route type

Expected CPU test/checkpoint paths:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuInternalPackageInventoryTest.java`, only if a top-level production type is added
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuRepresentationPlannerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedPartitionExecutableTest.java`, only if its production owner changes
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasRouteSelectorTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasPreparedExecutableTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasNativeCheckpoint.java`

Backend-conformance path:

- `testing/backend-conformance/src/test/java/io/github/pho001/synaptik/testing/conformance/CpuOpenBlasRouteConformanceTest.java`

Documentation and planning paths:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`, only if the current OpenBLAS route, CPU native peer route, materialization,
  or MATMUL entries cannot accurately express the finalized distinction
- this task
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

Review-only paths include `CpuMatmulLowering`, `CpuAccessPlan`, `CpuAffineCopyIr`, the affine-copy
emitter/generator/tests, buffer/workspace implementations, discovery source/tests, provider
source/tests/plans, public APIs, architecture/ADRs, Gradle files, other backend-conformance
sources, integration tests, and other modules. They remain byte-for-byte unchanged unless the
task is returned to planning first.

## Maximum scope

This task may create or modify at most 25 paths:

- at most 11 CPU production paths, including at most one new route-local output-copy type and at
  most one of the two conditional existing paths above;
- at most eight CPU test/checkpoint paths;
- the one existing backend-conformance test path;
- at most two explanatory documentation paths; and
- the three planning paths.

The final implementation should be smaller when conditional files are unnecessary. A need for an
emitter/generator/schema path, provider path, Gradle file, shared module, architecture/conformance
test beyond the listed existing path, integration test, more than one new production type, or more
than 25 total paths is a planning stop rather than implied scope.

## Acceptance criteria

1. Exact filtering retains the CPU 0010 semantic/type/provider/thread boundary and admits only the
   resolved positive rank-two access geometry defined here. Empty, zero-K, batched, broadcast-
   batched, promoted, epilogue, negative-stride, unresolved, partial-range, overflow, and
   unproved-span cases remain portable with no native-only resource.
2. Direct boundaries require exact canonical zero-offset row-major native/alignment facts. Every
   admitted transpose or other affine view is represented only through an existing generated
   affine copy; no provider transpose flag or semantic transpose inference exists.
3. The selector constructs only the eight bounded masks, with zero through two input copies and
   zero or one output copy. Each mask is internally complete, declares exactly one distinct
   workspace per copy, and fits the combined byte ceiling. Tests prove every mask and every
   missing-required-copy rejection.
4. Input materializations reuse `CpuMaterializationPlan` and existing affine-copy identity. Output
   copy-out has an honest typed route-local direction/ownership contract and does not pretend to
   be an external-read consumer materialization or alter ordinary 0008E selection.
5. The exact cost equation, expected runs, thresholds, and tie-break order are implemented with
   checked arithmetic. Table-driven tests cover every term independently, all masks, byte and
   copied-element totals, threshold boundaries, ties, missing terms, overflow, and deterministic
   fail-closed selection.
6. Analysis retains the complete portable plan, selects one immutable OpenBLAS plan at most, and
   declares all selected workspace requirements before assignment. Constructor invariants reject
   copy-mask, binding, resource-ID, byte/alignment, cost, and route disagreements. Finalization
   adds no resource and cannot change selection.
7. Finalization loads only selected input/output affine-copy artifacts and builds one immutable
   copy-in/GEMM/copy-out recipe. No OpenBLAS representation change alters the portable MATMUL
   generator, generated schema, or ordinary representation candidates.
8. Cold binding validates every resource and overlap before the first write. It permits A/B
   overlap, rejects C overlap with either input regardless of copy mask, and rejects every buffer-
   workspace or workspace-workspace overlap. Tests cover heap/native/mixed carriers, offsets,
   transposed/strided/zero-stride input reads, noncanonical injective output writes, alignment,
   read-only output, closed resources, and provider thread drift.
9. Hot execution performs selected input copies left then right, one typed provider call with the
   exact existing dimensions/scalars, and optional output copy. Native-free fakes prove order,
   call count, segment identity, represented results, repeated-run recipe reuse, isolated
   concurrent-run workspaces, and failure propagation with no route lookup or late fallback.
10. The existing backend-conformance route test adds at least one FLOAT32 and one FLOAT64 expanded
    representation case through staged analysis, assignment, finalization, cold binding, and
    execution. Together they cover both input copies and output copy without native lookup,
    provider dependency, Engine, environment activation, or skips; the original direct-route
    coverage remains.
11. The explicit native checkpoint uses the caller-supplied exact OpenBLAS 0.3.34 binary and
    exercises FLOAT32/FLOAT64 direct, left-transpose, right-affine, both-input-copy, output-copy,
    and both-input-plus-output-copy cases against the existing higher-precision numerical oracle.
    It remains an explicit non-JUnit gate, restores thread count in `finally`, and never silently
    skips, discovers a substitute binary, or makes a performance claim.
12. `CpuCapabilityProvider`, `BackendId("cpu")`, capability reporting, 0010A discovery/loading,
    provider public shape/source/tests, public/shared APIs, dependencies, Gradle, generated schema,
    and other modules remain unchanged. Package inventory changes only if one exact new route-
    local type is created.
13. A distinct clean documentation-focused agent reviews the final diff and implementation
    evidence, finalizes affected Javadoc/package documentation under the General and API/Javadoc
    profiles, updates the CPU guide under the Backend Guide profile, resolves glossary impact,
    and synchronizes task/master/roadmap under the Planning profile. It records reasoned no-change
    conclusions for public Tensor/Compile/Training/Config/Engine APIs, Model semantics, provider,
    shared Prepare/Runtime, architecture/ADRs, Gradle/dependencies, generated code/schema,
    discovery, other backend-conformance sources, integration, and other modules.
14. The required Java, native, Javadoc, generated-page, Markdown, exact-scope/status/order, and
    whitespace validation passes before this task becomes `Complete`.

## Tests / validation

During implementation, run focused tests as needed. After executable Java stabilizes, run one
final affected-project validation:

```bash
./gradlew :backends:cpu:test :testing:backend-conformance:test
```

Ordinary tests must require no installed OpenBLAS or native-access permission. They must cover the
selector/config/plan invariants, exact geometry, all eight masks, resource declarations and
assignment, finalization, sequencing, binding/failure-before-write, fake provider invocation,
portable fallback, and the conformance cases above.

Compile and run the expanded explicit checkpoint with the caller-supplied exact compatible
OpenBLAS 0.3.34 binary:

```bash
./gradlew :backends:cpu:testClasses
java --enable-native-access=ALL-UNNAMED --add-modules jdk.incubator.vector \
  -cp backends/cpu/build/classes/java/test:backends/cpu/build/classes/java/main:backends/openblas-provider/build/classes/java/main:modules/model/build/classes/java/main:modules/config/build/classes/java/main:modules/planning/build/classes/java/main:modules/runtime/build/classes/java/main:modules/prepare/build/classes/java/main:modules/backend-contract/build/classes/java/main:modules/trace/build/classes/java/main \
  io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas.CpuOpenBlasNativeCheckpoint \
  <ABSOLUTE_OPENBLAS_0_3_34_LIBRARY>
```

The implementation context then runs one repository-wide gate because behavior and tests span the
CPU and backend-conformance projects:

```bash
./gradlew test
```

The repository-wide result may serve as the final affected-project evidence if it is run after
all executable/test changes; do not repeat the same stable suites merely to duplicate evidence.
The 0010A-0010E capability checkpoint and CI remain later independent gates.

No Class-File decompilation, generated structural inspection, generated/direct performance gate,
or benchmark is required because this task reuses the existing affine-copy generator and changes
no generated-code implementation or schema. If generated code must change, stop and replan; any
later authorized generated implementation must use optimal clean Java as its structural and
performance oracle and add proportionate semantic, Class-File, forbidden-call/allocation, and
performance evidence.

The documentation-focused context receives and reuses the successful Java/native evidence unless
it changes executable behavior. After final Javadoc/documentation edits it runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
git status --short -uall
```

It renders and inspects every changed generated Javadoc page; checks local Markdown links and
anchors, heading order/uniqueness, balanced fences, LF/final newlines, terminology, examples, and
trailing whitespace; validates the exact 25-path ceiling and package inventory; confirms 0010B is
`Complete` only after all evidence, 0010C-0010E remain `Draft`, 0011 remains `Blocked`, and no
detailed 0010C/0010D/0010E file exists. It must not rerun successful Java suites unless it changes
executable behavior or records a concrete stale-evidence risk.

## Dependencies

- Complete [CPU 0010](0010-narrow-openblas-blas-compatible-native-route.md), which supplies the
  qualified one-call route, provider-free invocation seam, checked whole-plan selection, staged
  finalization, native-free conformance case, and native checkpoint.
- Complete [CPU 0010A](0010a-automatic-openblas-discovery-and-internal-composition-foundation.md),
  which supplies bounded discovery and explicit provider-session ownership without changing
  qualification.
- Complete [CPU 0008E](0008e-bounded-multi-input-materialization-and-representation-reuse.md),
  plus 0008E1 and 0008F, which supply bounded input-copy facts, shared partition-DAG adoption,
  and complete portable MATMUL/access geometry.
- Complete OpenBLAS provider tasks 0001-0003 and the current compatible OpenBLAS 0.3.34 checkpoint
  environment.
- Current shared Prepare workspace assignment and Runtime per-run ownership contracts, which are
  sufficient and remain unchanged.

All dependencies are complete and the current source demonstrates no shared-contract or
architecture gap.

## Follow-up tasks

- CPU 0010C remains `Draft` and owns coordinated provider thread-count candidates, explicit
  composition-owned state exclusion/restoration, and the shared CPU concurrency budget.
- CPU 0010D remains `Draft` and owns installed-binary qualification and stable target/binary
  fingerprinting. This task continues to require explicit qualification.
- CPU 0010E remains `Draft` and owns typed OpenBLAS tuning candidates, compatibility signatures,
  and consumption of explicit compatible selection evidence. It must include this task's exact
  copy mask and cost-schema meaning rather than redefining it.
- CPU 0011 remains `Blocked` on the ordered OpenBLAS sequence plus its independent Intel use-case
  and supported oneMKL ABI evidence.
- Batch, broadcast, repeated GEMM, epilogue, packing, and persistent packed-weight work has no
  authorized detailed task and is not an implied follow-up.

## Architecture impact

Expected impact: None. This is a concrete-CPU representation and prepared-recipe expansion under
the existing staged backend boundary. It retains one CPU backend identity, the current provider
leaf and dependency direction, shared Prepare workspace declarations/assignment, Runtime per-run
resource ownership, and immutable prepared execution.

If implementation requires a provider or shared/public API change, architecture update, module or
Gradle edge, general materialization redesign, Runtime route logic, or another file category
outside the exact ceiling, stop and report the conflicting rule and needed decision.

## Implementation prompt

Use this prompt in a separate agentic task/thread with a clean context:

```text
You are the clean implementation agent for Synaptik CPU task 0010B. Work in
/Users/phujka/IdeaProjects/Synaptik. Do not use GSD, commit, or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, the CPU master plan, and
docs/planning/backends/cpu/tasks/0010b-bounded-openblas-matmul-representation-expansion.md in full,
plus every architecture, prerequisite task, source, test, and documentation contract directly
referenced by that task. Implement it exactly within its 25-path ceiling. Stop for any
architecture, provider, shared-contract, generated-code, public-API, or scope conflict instead of
inventing a boundary.

After executable code and Java/native validation stabilize, hand the exact diff and recorded
evidence to a distinct clean documentation-focused agent in the same overall change. That pass
must follow docs/developer-guide/documentation-rules.md, independently inspect the implementation
and tests, finalize affected Javadocs/package documentation, CPU guide, glossary impact, and
planning evidence, and avoid repeating successful Java suites unless executable behavior changes
or a concrete stale-evidence risk is recorded. Do not mark 0010B Complete until that pass and all
specified validation succeed.
```

## Local decisions

- Keep transpose handling in CPU representation planning. A transpose is a resolved affine
  access pattern copied into canonical order, not a provider flag or new MATMUL semantic.
- Permit current proved zero-stride input reads as affine copies because they repeat immutable
  represented values and the existing address plan handles them exactly. Continue to require an
  injective output.
- Use one workspace per copied boundary. Do not alias or recycle the output workspace with an
  input workspace; the bounded extra allocation makes ownership and overlap validation explicit.
- Preserve the conservative C-versus-A/B non-overlap rule even if an input is copied first. Alias
  legality therefore does not change with the selected representation.
- Keep general external-read materialization and route-local output copy as separate typed
  contracts. This avoids weakening `CpuMaterializationPlan` consumer meaning or ordinary 0008E
  invariants.
- Price every run-owned workspace and copy on every expected run. This task introduces no
  persistent materialized data or amortized packing.
- Require one real OpenBLAS 0.3.34 checkpoint because the expanded path changes CPU-side data
  representation surrounding the same provider call. The checkpoint is semantic evidence, not
  qualification automation or a performance claim.

## Known limitations

- Only positive rank-two bare same-type FLOAT32/FLOAT64 MATMUL is expanded. Empty/zero-K,
  promoted, batched, broadcast, and fused forms remain portable.
- Every non-direct boundary is fully copied on every run. There is no packing, partial copy,
  persistent weight representation, workspace reuse within a run, or cross-run copied-data reuse.
- The existing affine-copy implementation eagerly retains address pairs and therefore bounds a
  copied matrix to its current representable element-count limit even when the provider dimensions
  themselves fit 32-bit `blasint`.
- Provider qualification, single-thread installation/exclusion, and lifetime remain explicit
  composition responsibilities. Discovery success alone is insufficient.
- The task supplies no measured profitability calibration. Selection uses explicit dimensionless
  complete-cost facts; CPU 0010E later owns compatible measured candidate evidence.

## Validation evidence

Empty until implemented. Record every command and result, exact test counts, the OpenBLAS 0.3.34
path and checkpoint cases, reused versus rerun evidence, Javadoc/render/Markdown/scope checks, the
clean documentation context ID and result, package placement, glossary conclusion, and all
reasoned no-change conclusions required above.

## Implementation notes

Empty until implemented. Record the finalized type placement, copy-mask representation,
workspace IDs, sequencing owner, cost-fact shape, and any within-scope local decision that differs
from the proposed route-local output-copy type.

## Completion summary

Empty until implemented. Use the Planning Guide completion-summary format and include completed
changes, exact files, tests/validation, documentation-agent review, documentation/Javadoc/glossary
impact, unresolved issues, required follow-up, and final status.
