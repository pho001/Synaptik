# cpu1 Experimental Backend Transfer Plan

This is an isolated plan for evolving `backend.cpu1` from the existing experimental elementwise backend. It intentionally ignores previous planning documents and does not assume `cpu1` is wired into production runtime yet.

## Goal

Build `cpu1` into a prepared-plan CPU backend that keeps the current `cpu1` strengths:

- prepare-time kernel/layout/storage/vector/thread decisions
- minimal execute-time branching
- generated static elementwise hot paths
- clean separation from `Tensor`, autograd, and old production `backend.cpu` internals

The plan transfers only proven concepts and selected implementation pieces from `backend.cpu`. It must not copy the old runtime-heavy architecture wholesale.

## Non-Goals

- Do not wire `cpu1` into default runtime execution yet.
- Do not replace `backend.cpu`.
- Do not port `CpuBackend`, `CpuNodePreparer`, `CpuKernelExecutor`, or `CpuNodeExecutionPlan` directly.
- Do not port the old `CpuStorageAwareKernel` hierarchy.
- Do not add fused execution in this phase.
- Do not introduce generic compatibility adapters just to reuse old code.

## Architecture Direction

`cpu1` should keep this model:

```text
prepare:
  inspect compiled node/descriptors/config
  choose exact executable unit
  choose storage kind
  choose layout kind
  choose scalar/vector path
  choose launch/chunk policy
  allocate or specify workspace needs

execute:
  bind runtime views
  run prepared executable
  publish output residency/trace
```

The old `backend.cpu` package should be treated as a source of algorithms and policies, not as a structure to clone.

## Target Package Shape

```text
backend.cpu1
  Cpu1Backend

backend.cpu1.prepare
  Cpu1NodePreparer
  Cpu1ElementwisePreparer
  Cpu1LayoutPreparer
  Cpu1ReductionPreparer
  Cpu1MatmulPreparer
  Cpu1PrepareConfig

backend.cpu1.exec
  Cpu1ExecutableUnit
  Cpu1RangeExecutableUnit
  Cpu1LayoutExecutableUnit
  Cpu1ReductionExecutableUnit
  Cpu1MatmulExecutableUnit
  Cpu1KernelArgs
  Cpu1TensorView
  Cpu1Workspace
  Cpu1WorkspaceSpec

backend.cpu1.dispatch
  Cpu1DispatchPolicy
  Cpu1DispatchDecision
  Cpu1CostClass

backend.cpu1.kernels
  generated elementwise loops
  reduction loops
  layout loops
  matmul loops/providers

backend.cpu1.provider.matmul
  Cpu1MatmulProvider
  Cpu1JavaMatmulProvider
  Cpu1OpenBlasMatmulProvider
  Cpu1MatmulRoute
```

Do not create these packages until a phase actually needs them.

## Phase 1: Executable Unit Boundary

### Problem

`Cpu1PreparedArtifact` currently wraps a single `Cpu1PreparedUnit` shape that fits elementwise range kernels. Layout ops, reductions, matmul, and provider-backed executables will not fit cleanly into that object.

### Implementation

Introduce a small executable abstraction:

```java
public interface Cpu1ExecutableUnit {
    void run(ExecutionContext context);
}
```

Then adapt existing elementwise execution:

```text
Cpu1PreparedArtifact
  -> Cpu1ExecutableUnit executable

Cpu1RangeExecutableUnit
  -> wraps current Cpu1PreparedUnit behavior
```

Keep current `Cpu1PreparedUnit` for range kernels if that avoids churn, but make it an implementation detail of `Cpu1RangeExecutableUnit`.

### Acceptance Criteria

- Existing `backend.cpu1.*` tests pass.
- `Cpu1PreparedArtifact.execute(context)` delegates to a `Cpu1ExecutableUnit`.
- No production runtime wiring is added.
- No behavior change for existing elementwise kernels.

## Phase 2: Dispatch Policy

### Problem

`cpu1` currently receives vector/parallel decisions directly from config. That is useful for tests, but it does not scale for real execution.

### Transfer From `backend.cpu`

Transfer the idea from:

- `CpuPlanningPolicy`
- `ElementwiseDispatchPlanner`
- `ResolvedDispatchHints`

Do not copy them directly.

### cpu1 Design

Add:

```java
public enum Cpu1CostClass {
    CHEAP_ELEMENTWISE,
    EXPENSIVE_ELEMENTWISE,
    REDUCTION,
    MATMUL
}

public record Cpu1DispatchDecision(
    Cpu1VectorizationKind vectorizationKind,
    Cpu1LaunchConfig launchConfig,
    int scalarChunkSize,
    int vectorChunkSize,
    int plannedWorkers
) {}
```

Add policy:

```java
public final class Cpu1DispatchPolicy {
    Cpu1DispatchDecision decideElementwise(
        Operation.OpType opType,
        DataType computeType,
        long elementCount,
        Cpu1PrepareConfig config
    )
}
```

Initial classification:

- Cheap: `ADD`, `SUB`, `MUL`, `DIV`, `MIN`, `MAX`, `RELU`, `ABS`, `NEG`, `INV`, `CLAMP_MIN`, `CLAMP_MAX`, comparisons, logical ops.
- Expensive: `EXP`, `FAST_EXP`, `LOG`, `TANH`, `FAST_TANH`, `SIGMOID`, `POW`, `POW_TENSOR`, `ERF`, `SQRT`.
- Reduction and matmul are placeholders until those phases exist.

### Rules

- Decision happens in prepare.
- Execute never computes thresholds.
- Config can still force scalar/vector/single/parallel for tests.
- Automatic policy must use benchmark/profile-backed `CpuKernelConfig` values, not local hardcoded threshold defaults.
- While `cpu1` is not wired into production runtime, tests and experiments may pass `Cpu1PrepareConfig.automatic(...)` explicitly.
- When `cpu1` is later wired into the real prepare/runtime path, pass the active `RuntimeConfig.cpuKernelConfig()` into `cpu1` prepare instead of resolving or inventing a separate CPU policy.

### Acceptance Criteria

- Existing explicit config tests still pass.
- New tests cover automatic dispatch for small and large elementwise ops.
- `Cpu1PreparedUnit` stores the resolved launch/vector decision.
- Automatic dispatch decisions are derived from `CpuKernelConfig` threshold/chunk fields.

## Phase 3: Workspace

### Problem

Some kernels need temporary memory or reusable caches. Allocating inside hot path creates avoidable GC pressure.

### Transfer From `backend.cpu`

Transfer the concept from `CpuNodeWorkspace`, not the full class.

Useful old concepts:

- `float[]` temporary continuation/intermediate storage
- `int[]` index workspace
- packed weight cache for linear/matmul

### cpu1 Design

Add:

```java
public record Cpu1WorkspaceSpec(
    int f32Elements,
    int f64Elements,
    int i32Elements,
    boolean needsProviderCache
) {}

public final class Cpu1Workspace {
    float[] requireF32();
    double[] requireF64();
    int[] requireI32();
    Object providerCache();
}
```

Workspace should be allocated or reused by the executable unit, not by kernels directly.

### Initial Uses

- reductions requiring temporary accumulation buffers
- softmax/logsoftmax row workspace
- matmul packed weight cache
- BF16 internal F32 intermediates

### Acceptance Criteria

- No existing elementwise kernel allocates workspace unnecessarily.
- Workspace is optional.
- Workspace is not exposed to public `Tensor`.

## Phase 4: Layout And View Ops

### Problem

Elementwise support is not enough for real graphs. Many graph nodes are layout/view operations.

### Transfer From `backend.cpu`

Transfer behavior and tests from layout kernels where useful:

- `CpuAliasViewKernel`
- `CpuContiguousKernel`
- `CpuReshapeLikeKernel`
- `CpuExpandKernel`
- `CpuPermuteKernel`
- `CpuConcatKernel`
- `CpuPadKernel`
- `CpuTileKernel`

Do not transfer the old storage-aware kernel hierarchy.

### Split Layout Ops

Metadata-only or alias/view:

- `RESHAPE`
- `EXPAND`
- `SELECT`
- `SLICE`
- `PERMUTE`
- `EXPAND_DIMS`
- `SQUEEZE`
- `NOOP`

Copy/materializing:

- `CONTIGUOUS`
- `CONCAT`
- `PAD`
- `TILE`
- later `UNFOLD_AXIS`, `UNFOLD2D`, `FOLD2D`

### cpu1 Design

Add:

```text
Cpu1LayoutPreparer
Cpu1LayoutExecutableUnit
Cpu1LayoutKernelId
```

Metadata-only ops should create runtime views or alias outputs without numeric loops.

Copy ops should use explicit prepared copy kernels.

### Acceptance Criteria

- `cpu1` can execute common elementwise graphs containing reshape/expand/permute/slice.
- Metadata-only ops do not copy storage.
- Copy ops mark output storage modified.
- Tests verify both array and MemorySegment where applicable.

## Phase 5: Reductions

### Problem

Reductions are the next major operation family after elementwise and layout.

### Transfer From `backend.cpu`

Use algorithmic parts from:

- `ReductionTraversal`
- `ReductionStorageAccess`
- `SumLoops`
- `MinMaxReduceLoops`
- `BoolReduceLoops`
- `SoftmaxLikeTraversal`
- `SoftmaxLikeExecutor`

Do not directly import old `StorageAwareReductionKernel`.

### Initial Scope

Start with:

- `SUM`
- `MEAN`
- `REDUCE_MIN`
- `REDUCE_MAX`
- `REDUCE_PROD`

Then:

- `REDUCE_ALL`
- `REDUCE_ANY`
- `ARGMAX`
- `CUMSUM`
- `SOFTMAX`
- `LOG_SOFTMAX`

### cpu1 Design

Add:

```text
Cpu1ReductionPreparer
Cpu1ReductionPlan
Cpu1ReductionExecutableUnit
Cpu1ReductionKernelId
```

Prepare should resolve:

- axis/axes
- keepDims
- input/output shape contract
- contiguous vs strided traversal
- accumulation dtype
- workspace requirements
- parallel strategy

### Acceptance Criteria

- Reductions support F32/F64/BF16 where meaningful.
- Bool reductions support BOOL.
- BF16 reductions accumulate in F32 unless a stronger reason exists.
- No hidden Tensor/autograd dependencies in kernels.

## Phase 6: Matmul And Provider Model

### Problem

Matmul performance depends heavily on size, dtype, layout, batching, and provider overhead.

### Transfer From `backend.cpu`

Analyze and selectively port:

- Java microkernels under `backend.cpu.kernels.linalg.matmul`
- packing logic
- OpenBLAS provider route
- BF16/F32/F64 dispatch logic
- batched matmul route if benchmarked useful

Do not transfer the old `CpuNodeExecutionPlan` or provider factory as-is.

### cpu1 Design

Add:

```java
public enum Cpu1MatmulRoute {
    JAVA_MICROKERNEL,
    OPENBLAS_ARRAY_COPYING,
    OPENBLAS_NATIVE_SEGMENT
}

public record Cpu1MatmulPlan(
    Cpu1MatmulRoute route,
    DataType inputType,
    DataType computeType,
    DataType outputType,
    int batchCount,
    int m,
    int n,
    int k,
    boolean transposeA,
    boolean transposeB,
    Cpu1WorkspaceSpec workspaceSpec
) {}
```

Add:

```text
Cpu1MatmulPreparer
Cpu1MatmulExecutableUnit
Cpu1MatmulProvider
Cpu1JavaMatmulProvider
Cpu1OpenBlasMatmulProvider
```

### Required Benchmark First

Before enabling AUTO route, measure:

- small matrices where Java microkernel may beat BLAS overhead
- medium matrices
- large matrices
- batched matrices
- F32, F64, BF16
- array vs native segment

### Acceptance Criteria

- Manual route selection works before AUTO.
- AUTO is enabled only after benchmarks define thresholds.
- Provider route is visible in trace/events.
- Packed weight cache uses workspace/provider cache, not per-execute allocation.

## Phase 7: BF16 Compute DType Planning

### Problem

Java does not have a native BF16 primitive. BF16 kernels currently load BF16, compute in F32, and store BF16 per op. For chains, this can add unnecessary conversions and precision loss.

### Direction

Use compiled-plan semantics:

```text
logical dtype: BF16
internal compute dtype: F32
boundary materialized dtype: BF16
```

This is cleaner than old node-by-node float continuation.

### Transfer From `backend.cpu`

Transfer the reason for float continuation, not the implementation.

Old float continuation solves:

- producer stores BF16 because node output is BF16
- consumer wants F32 compute value
- side buffer avoids reload BF16 -> F32 loss

In `cpu1`, solve this at plan level instead:

- F32 intermediate buffers inside a compiled plan
- BF16 materialization only at graph boundaries or required observation points

### Acceptance Criteria

- BF16 chain can execute with F32 intermediate storage in a prepared plan.
- Final output still respects logical BF16 dtype.
- Tests compare against current BF16-per-op behavior and document expected numerical improvement.

## Phase 8: Trace Events Outside Backend

### Problem

Trace code should not pollute kernel or backend logic.

### Direction

Create a backend-neutral trace/event package later:

```text
runtime.trace or backend.trace
  TraceEvent
  TraceSink
  KernelSelectedEvent
  DispatchDecisionEvent
  ProviderRouteEvent
  StorageMaterializedEvent
  FallbackEvent
```

`cpu1` should emit events but not own the trace system.

### cpu1 Events

Emit:

- selected kernel id
- storage kind
- layout kind
- vectorization kind
- launch config
- chunk sizes
- materialization decisions
- provider route
- fallback reason

### Acceptance Criteria

- Trace can be disabled with near-zero hot-path overhead.
- Trace data is structured, not string-only.
- Backend code emits concise events through a minimal interface.

## Phase 9: Fusion

Fusion is intentionally deferred.

When ready, evaluate:

- reuse old fused IR concepts
- reuse old ASM generator only after compatibility review
- keep `cpu1` prepare-time dispatch model
- avoid reintroducing runtime-heavy fused execution paths

## Migration Rules

For every transferred component:

1. Identify the old `backend.cpu` source file or algorithm.
2. Decide whether to port behavior, code, or only tests.
3. Keep the `cpu1` public boundary clean.
4. Preserve prepare-time decision making.
5. Avoid per-element virtual calls in hot loops.
6. Avoid hidden materialization.
7. Add focused tests before broad runtime integration.

## Recommended Execution Order

1. Refactor `Cpu1PreparedArtifact` around `Cpu1ExecutableUnit`.
2. Add `Cpu1DispatchPolicy`.
3. Add optional `Cpu1Workspace`.
4. Add metadata-only layout ops.
5. Add copy layout ops.
6. Add basic reductions.
7. Add softmax/logsoftmax.
8. Benchmark matmul providers.
9. Add manual matmul routes.
10. Add AUTO matmul routing after benchmark thresholds.
11. Add BF16 compute dtype planning.
12. Add backend-neutral trace events.
13. Revisit fusion.

## Done Definition

This plan is complete when `cpu1` can execute a representative unfused inference graph containing:

- layout/view ops
- elementwise ops
- reductions
- softmax/logsoftmax
- matmul/linear through Java and OpenBLAS routes
- BF16 inputs with F32 internal compute where beneficial
- structured trace events for dispatch and provider decisions

Production runtime wiring remains a separate decision after this plan.
