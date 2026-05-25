<!-- generated-by: gsd-doc-writer -->
# Backend Planning And Regions

Navigation: [Index](index.md#recommended-reading-paths) | [Graph Optimizer](graph-optimizer.md#graph-optimizer) | [Configuration](configuration.md#compileconfig) | [Metal Backend](metal-backend.md#end-to-end-flow) | [Compute Flow](compute-flow.md#compile)

This document explains compile-time backend ownership planning, execution regions, region optimization, and memory planning. These concepts used to be described as optimizer stages, but they now belong to explicit compile layers.

## Term Map

| Term | Meaning |
|---|---|
| Backend | Runtime family that can execute a node or region, such as CPU, Metal, or CUDA. |
| Backend intent | A compile-time hint on a node that says which backend should own it if legal. |
| Backend planning | Compile-time process that chooses ownership regions before runtime preparation. |
| Ownership region | A group of graph nodes assigned to a backend as one planned region. |
| Accelerator region | An ownership region for Metal, CUDA, or another accelerator backend. |
| CPU natural region | A CPU-owned group of nodes that can later be split into unit kernels and fused units. |
| Region optimization | The phase that turns an ownership region into execution units. |
| Execution unit | A concrete unit inside a region, for example one CPU matmul kernel or one fused elementwise chain. |
| Memory planning | Compile-time lifetime and reusable-buffer planning across graph and regions. |
| Runtime selection | Prepare-time choice of the executable backend path using `RuntimeConfig` and availability checks. |
| CPU native storage region | CPU execution region that keeps selected values in `MemorySegment` storage for provider-backed routes or explicit native diagnostics. |

The distinction between ownership region and execution unit is critical:

```text
ownership region:
  CPU owns nodes [matmul, add, relu, sum]

execution units inside that region:
  unit 1: matmul kernel
  unit 2: fused add+relu loop
  unit 3: sum reduction kernel
```

One region does not imply one runtime kernel.

## CPU Native Storage Regions

CPU native storage is planned as a storage route inside the CPU backend, not as another backend family.
`CpuRegionLowerer` may lower a CPU-owned subregion to a native-storage region when runtime policy allows it.
That region still executes through CPU-owned prepared artifacts and existing operation-family kernels.

The current selection policy is intentionally conservative:

| Runtime policy | Native-storage behavior |
|---|---|
| `CPU_ARRAY` | No native CPU storage region is selected for compute. |
| `CPU_NATIVE` | Supported native-storage routes may be selected for explicit diagnostics/correctness, with fallback controlled by `NativeCpuFailurePolicy`. |
| `AUTO` | Only provider-backed routes, currently OpenBLAS `MemorySegment` GEMM, and metadata-only native view aliases are eligible by default. Slow segment scalar non-BLAS kernels remain rejected unless measured proof promotes that family. |

The native region trace exposes selected/rejected decisions, provider/local/view nodes, storage contracts, physical
kernel families, layout materialization reasons, and `nativeCpuRegionAutoEligible`. This evidence is the replacement
for the removed native CPU plan/facts/parity runtime stack.

## Current Compile Layers

`CompileConfig` is the source-of-truth compile policy:

```java
CompileConfig compile = CompileConfig.training()
        .withBackendPlanning(BackendPlanningConfig.autoAccelerator())
        .withRegionOptimization(RegionOptimizationConfig.trainingDefaults())
        .withMemoryPlanning(MemoryPlanningConfig.trainingDefaults());
```

The layers are:

| Layer | Source | Owns |
|---|---|---|
| Semantic canonicalization | `SemanticCanonicalizationConfig` | Pre-autograd forward graph canonicalization. |
| Graph optimization | `GraphOptimizationConfig` | Backend-neutral `AR`, `CF`, `CSE`, `DCE`, `LOWER`. |
| Backend planning | `BackendPlanningConfig` | CPU-only, explicit accelerator, or automatic accelerator region discovery. |
| Region optimization | `RegionOptimizationConfig` | Fusion and execution-unit planning inside owned regions. |
| Memory planning | `MemoryPlanningConfig` | Lifetimes, reusable runtime slots, region value flow, region bindings, handoff requirements, runtime binding policy, and summary metrics. |

## BackendPlanningConfig

`BackendPlanningConfig` controls compile-time backend ownership. It is not runtime hardware policy.

Source:

- `src/main/java/config/compile/BackendPlanningConfig.java`
- `src/main/java/config/compile/BackendDiscoveryMode.java`
- `src/main/java/config/compile/BackendPlanningFailurePolicy.java`
- `src/main/java/config/compile/BackendTarget.java`
- `src/main/java/config/compile/RegionOwnershipPlannerStrategy.java`
- `src/main/java/config/compile/PartitionSearchConfig.java`

Main fields:

| Field | Meaning |
|---|---|
| `discoveryMode` | How accelerator regions may be discovered: `CPU_ONLY`, `EXPLICIT`, or `AUTO`. |
| `failurePolicy` | Whether missing accelerator plans are optional or errors. |
| `requirementScope` | Whether requirements apply to any target, each target, or all explicit intents. |
| `targets` | Accelerator targets eligible for planning, such as `GPU_METAL` or `GPU_CUDA`. |
| `ownershipPlanner` | Region planner strategy, currently mapped to anchor or scored partition planners. |
| `search` | Search limits and score weights. It does not select backend target by itself. |
| `cpuRegions` | CPU natural-region policy and width limits. |
| `cost` | Planning cost profile, including Metal transfer model for scored accelerator planning. |

### Discovery Modes

`CPU_ONLY` means no accelerator ownership regions are planned. CPU natural regions may still be planned if CPU region policy is enabled.

```java
CompileConfig compile = CompileConfig.training()
        .withBackendPlanning(BackendPlanningConfig.cpuOnly());
```

`EXPLICIT` means only nodes that already carry explicit accelerator backend intent can seed accelerator planning. The planner can still include legal supporting nodes required for a valid closed region when the target backend partition capability allows it.

```java
CompileConfig compile = CompileConfig.training()
        .withBackendPlanning(BackendPlanningConfig.explicitOnly());
```

`AUTO` means the planner may discover accelerator regions from CPU-owned graph structure according to target legality and cost policy.

```java
CompileConfig compile = CompileConfig.trainingAutoAccelerator();
```

Example:

```text
CPU-owned graph:
  x -> matmul -> add -> relu -> sum

EXPLICIT:
  no GPU region unless some node already carries GPU intent

AUTO:
  planner may consider matmul/add/relu as a GPU region if legal and profitable
```

## Failure Policies

Accelerator planning can be optional or required.

| Preset | Meaning |
|---|---|
| `BackendPlanningConfig.cpuOnly()` | No accelerator plan required or created. |
| `BackendPlanningConfig.explicitOnly()` | Explicit accelerator intents are honored when legal, but failure is optional by default. |
| `BackendPlanningConfig.autoAccelerator()` | Auto-discover accelerator regions when legal/profitable, failure optional. |
| `BackendPlanningConfig.requireAnyAcceleratorRegion()` | Require at least one accelerator region from auto planning. |
| `BackendPlanningConfig.requireEachAcceleratorTarget()` | Require a region for each configured accelerator target. |
| `BackendPlanningConfig.requireAllExplicitIntents()` | Every explicit accelerator intent must be represented by a legal planned accelerator region. |

Use required policies in tests and diagnostics. Use optional policies for production paths where CPU fallback is acceptable and should be visible in traces rather than throwing during compile.

## Authoritative Job Resolution

`BackendPlanningJobResolver` is the single resolver that converts `BackendPlanningConfig` plus compiled nodes into planning jobs.

Source:

- `src/main/java/graph/compile/planning/BackendPlanningJobResolver.java`
- `src/main/java/graph/compile/planning/BackendPlanningService.java`

It resolves:

- which accelerator targets get jobs
- whether a job is explicit or automatic
- which source policy is used
- which region planner strategy is used
- whether a CPU natural-region job is added
- which Metal transfer model is passed into scored planning

Example resolution:

```text
config:
  discoveryMode = AUTO
  targets = [GPU_METAL]
  cpuRegions = defaults

graph:
  all nodes currently CPU-owned

jobs:
  1. GPU_METAL accelerator job
     reason = auto-accelerator-discovery
     source policy = CPU_OR_TARGET_BACKEND
  2. CPU natural-region job
     reason = cpu-natural-region
```

For `EXPLICIT`, the GPU job is emitted only if the graph contains explicit `GPU_METAL` or `GPU_CUDA` intent.

## Region Planning Strategies

The public compile strategy enum is `RegionOwnershipPlannerStrategy`.

| Strategy | Practical meaning |
|---|---|
| `ANCHOR` | Build regions from backend-capable anchors and expand through legal supporting structure. |
| `SCORED` | Search candidate regions and score them using structural and transfer-cost weights. |

The backend planning resolver maps those public strategies to internal partition planners. The internal word "partition" still appears in implementation classes such as `PartitionPlan` and `ScoredCandidatePartitionPlanner`, but in architecture language it means "planned ownership region", not "optimizer stage".

## CPU Natural Regions

CPU region planning is not a GPU fallback afterthought. It gives the later region optimizer a coherent CPU-owned graph area.

Source:

- `src/main/java/config/optimizer/CpuRegionConfig.java`
- `src/main/java/graph/compile/planning/partition/CpuNaturalExecutionRegionPlanner.java`
- `src/main/java/graph/compile/planning/region/CpuRegionOptimizationPolicy.java`

`CpuRegionConfig.defaults()` enables natural CPU regions with a maximum region size. Current default `maxRegionNodes` is `64`.

Example graph:

```text
0 x leaf
1 w leaf
2 b leaf
3 matmul(x, w)
4 add(3, b)
5 tanh(4)
6 sum(5)
```

The CPU natural-region planner may group nodes `[3, 4, 5, 6]` into one CPU execution region. That is a planning artifact. The region optimizer may still split it into multiple execution units:

```text
unit 1: UNIT_KERNEL        matmul
unit 2: FUSED_ELEMENTWISE  add + tanh
unit 3: UNIT_KERNEL        sum
```

This is why "one big CPU partition" does not mean "one giant fused kernel".

## Accelerator Regions

Accelerator regions are stricter than CPU natural regions. They must satisfy target legality.

Typical legality questions:

- Are all operations supported by the target backend?
- Are dtypes supported for the role each node plays?
- Are shapes and layouts representable by the backend lowering path?
- Are region inputs and outputs legal for buffer binding or fallback paths?
- Does the region require a supporting producer, such as a `PERMUTE`, to be included?

Example:

```text
a -> transpose -> matmul -> add
```

If `matmul` has explicit Metal intent, the planner may need to include `transpose` as a legal supporting producer. Rejecting it just because it is not itself a GPU seed would violate the explicit backend planning contract.

For Metal details, see [Metal Backend](metal-backend.md#end-to-end-flow).

## Region Optimization

Region optimization happens after ownership is decided.

Source:

- `src/main/java/config/compile/RegionOptimizationConfig.java`
- `src/main/java/graph/compile/planning/region/DefaultRegionOptimizer.java`
- `src/main/java/graph/compile/planning/region/CpuRegionOptimizationPolicy.java`
- `src/main/java/graph/compile/planning/value/GraphValueRef.java`

Region optimization answers:

- Which nodes become standalone unit kernels?
- Which elementwise chains become fused execution units?
- Which values stay virtual inside a fused unit?
- Which values must be materialized as region outputs or handoff values?

Term explanations:

- Materialized value: a value that exists in runtime storage after a unit runs.
- Continuation value: a value that can be forwarded inside a region without publishing to a public tensor.
- Virtual value: a value represented by expression structure inside a fused unit rather than stored as a standalone array.

Example:

```text
raw region:
  add -> mul -> tanh -> sum

optimized region:
  fused unit: add + mul + tanh
  unit kernel: sum
```

The sum is a reduction and usually acts as a boundary. The elementwise prefix is a better fusion candidate.

## Memory Planning

Memory planning is compile-time lifetime analysis. It decides which temporary runtime storage slots may be reused and where region handoff values must be materialized. `MemoryPlanner` is the public entry point; the package-local planners under `graph.compile.planning.memory` own tensor lifetimes, reusable interval filtering, slot allocation, region value flow, region bindings, handoff requirements, runtime binding policy, and summary metrics.

Source:

- `src/main/java/config/compile/MemoryPlanningConfig.java`
- `src/main/java/graph/compile/planning/memory/MemoryPlanner.java`
- `src/main/java/graph/compile/planning/memory/MemoryPlan.java`
- `src/main/java/graph/compile/planning/memory/TensorLifetimePlanner.java`
- `src/main/java/graph/compile/planning/memory/RegionValueFlowPlanner.java`
- `src/main/java/graph/compile/planning/memory/RegionBindingAllocator.java`
- `src/main/java/graph/execution/residency/RuntimeMemoryBinder.java`

Example:

```text
t1 = add(a, b)
t2 = relu(t1)
t3 = sum(t2)
```

If `t1` is consumed only by `t2` and no public tensor needs `t1`, memory planning can avoid treating `t1` as an independently published long-lived value. If a later region boundary needs `t2`, region value flow records the node-id to graph-value mapping, region binding allocation chooses reusable region slots, and handoff planning creates a requirement for `t2`.

Memory planning is not publication. Publication happens after runtime execution and is controlled by `PublicationPolicy`.

## Runtime Is Separate

`RuntimeConfig` owns hardware/runtime policy:

- CPU vector and parallel thresholds
- BLAS provider and minimum work gates
- approximate transcendental policy
- accelerator enablement
- accelerator runtime availability requirements
- accelerator buffer binding mode
- minimum accelerator work thresholds

Example:

```java
ExecutionProfile profile = new ExecutionProfile(
        "training-auto-metal",
        "training-auto-metal",
        DataType.FLOAT32,
        ExecutionMode.FORWARD_BACKWARD,
        CompileConfig.trainingAutoAccelerator(),
        RuntimeConfig.trainingDefaults()
);
```

Here `CompileConfig.trainingAutoAccelerator()` allows compile-time Metal region discovery. `RuntimeConfig.trainingDefaults()` still decides whether the Metal backend is enabled and available at prepare/runtime.

## PublicationPolicy Is Separate

`PublicationPolicy` controls which run-scoped values are copied back to user-visible tensors after execution.

Source:

- `src/main/java/graph/execution/PublicationPolicy.java`
- `src/main/java/graph/execution/PreparedExecution.java`

Policies:

| Policy | Publishes |
|---|---|
| `ALL` | Every forward value plus gradients. |
| `OUTPUT_AND_GRADIENTS` | Only the compiled output and gradients. Default for ordinary execution. |
| `OUTPUT_ONLY` | Only the compiled output. Default for optimizer-step execution. |
| `NONE` | Nothing is copied back to public tensors. Useful for benchmark paths that inspect traces rather than tensor storage. |

Publication is not backend planning. Disabling publication can avoid public tensor synchronization costs, including device-to-CPU copies, but it does not change which backend owns a region.

Example:

```java
PreparedExecution prepared = compiled.prepare(runtime);
prepared.executeTraced(ExecutionMode.FORWARD_BACKWARD, PublicationPolicy.OUTPUT_ONLY);
```

The run computes forward and backward steps, but only the root output is synchronized to the public tensor surface.

## Benchmark Semantics

A benchmark that compares no-opt CPU, calibrated CPU, and GPU paths should vary the right layer:

| Candidate | Compile policy | Runtime policy |
|---|---|---|
| no-opt CPU baseline | `CompileConfig.cpuOnlyBaseline()` or `noGraphOptimizationBaseline()` depending on the question | `RuntimeConfig.noOptNoVecNoPar()` |
| calibrated CPU | `CompileConfig.training()` or workload policy | calibrated `PlatformRuntimeProfile.toRuntimeConfig()` |
| GPU path | explicit or auto accelerator `CompileConfig` | runtime accelerator enabled and available |

Bad comparison:

```text
baseline: no graph optimization and no backend planning
candidate: graph optimization plus auto accelerator discovery
```

That compares too many things at once. A clean comparison changes one ownership layer at a time and reports fallback visibly.

## Autotune Ownership

Graph autotune may vary graph/workload-owned compile policy:

- current graph policy
- CPU region policy
- CPU fusion policy
- backend planning mode
- region planner strategy
- selected research-only planning cost profiles

Platform calibration may vary runtime/hardware policy:

- BLAS threshold
- vector/parallel thresholds
- fused ASM width
- scheduler chunk policy
- materialization thresholds
- explicit accelerator runtime selection when requested

Calibration must not silently change graph ownership policy. Graph autotune must not silently rewrite calibrated runtime thresholds. `ExecutionProfileAssembler` is the reassembly boundary that combines a saved graph policy with the current platform runtime profile.

## Diagnostics Checklist

When a region is missing:

1. Check `CompileConfig.backendPlanning()`.
2. Check whether the graph has explicit backend intent if using `EXPLICIT`.
3. Check `BackendPlanningJobResolver` output through compile trace evidence.
4. Check target legality rejection reasons.
5. Check whether `RuntimeConfig` disabled the backend later at prepare time.
6. Check run trace fallback reason if a planned region executed on CPU.

When a CPU path is slower:

1. Check whether the compile policy changed graph shape or region splits.
2. Check whether runtime thresholds came from calibration or default fallbacks.
3. Check BLAS provider evidence for matmul-heavy workloads.
4. Check materialization and publication counts separately.
5. For BF16, check whether compute is promoted to F32 or F64; see [CPU BF16 Runtime](cpu-bf16.md#cpu-bf16-runtime).

## See Also

- [Graph Optimizer](graph-optimizer.md#graph-optimizer)
- [Configuration](configuration.md#compileconfig)
- [CPU BF16 Runtime](cpu-bf16.md#cpu-bf16-runtime)
- [Metal Backend](metal-backend.md#end-to-end-flow)
