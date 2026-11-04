# [CPU](../../glossary.md#term-cpu) [kernel](../../glossary.md#kernel) strategy

## Purpose and status

This note explains how [CPU](../../glossary.md#term-cpu) execution routes fit behind one [backend](../../glossary.md#backend) [owner](../../glossary.md#term-owner) and records the current
ordinary-production safe [heuristic](../../glossary.md#term-heuristic). It does not define support or performance guarantees.

## Strategy

[Planning](../../glossary.md#term-planning) selects only `owner = CPU`. [CPU](../../glossary.md#term-cpu) [prepare](../../glossary.md#prepare) examines the owned [partition](../../glossary.md#partition) and chooses among
[scalar](../../glossary.md#term-scalar), [Vector API](../../glossary.md#term-vector-api), generated [JVM](../../glossary.md#term-java-virtual-machine-jvm)-[bytecode](../../glossary.md#term-bytecode) CPU computation-[kernel](../../glossary.md#kernel), [OpenBLAS](../../glossary.md#term-openblas), specialized, or
[fused](../../glossary.md#specialization-and-fusion) routes. CPU finalization generates or reuses a compatible selected [artifact](../../glossary.md#term-artifact) after shared
slot assignment. [Runtime](../../glossary.md#term-runtime) invokes the resulting [`PreparedExecutable`](../../glossary.md#prepared-executable--preparedexecutable) without repeating that
choice.

The architecture fixes this [ownership](../../glossary.md#term-ownership-meanings) and [lifecycle](../../glossary.md#lifecycle), not a particular bytecode-generation [API](../../glossary.md#term-application-programming-interface-api).
The CPU master plan records the selected current Java 26 implementation direction.

Scalar code should provide a readable reference path for selected capabilities. Optimized routes must preserve its specified semantics and failure behavior. OpenBLAS stays behind the CPU [backend](../../glossary.md#backend) through the low-level provider; Vector API configuration stays local to the CPU [module](../../glossary.md#term-module).

## Decision factors

[Prepare](../../glossary.md#prepare)-time route selection may use [operation](../../glossary.md#operation) and [layout](../../glossary.md#layout) facts, [data type](../../glossary.md#data-type--datatype), sizes, [alignment](../../glossary.md#term-alignment),
[fusion](../../glossary.md#term-fusion) opportunity, workspace requirements, native availability, a compatible workload-[cache](../../glossary.md#term-cache)
entry, and an explicit selected [model](../../glossary.md#term-model) plan. [CPU](../../glossary.md#term-cpu) route [owners](../../glossary.md#term-owner) later provide typed,
version-controlled candidate generators that derive complete valid configurations from [target](../../glossary.md#term-target)
capabilities, workload facts, and the tuning budget. [Matrix](../../glossary.md#term-matrix-meanings)-multiplication candidates may include
supported [JDK](../../glossary.md#term-java-development-kit-jdk) [Vector API](../../glossary.md#term-vector-api) species/strategy, unroll, tile, parallelism, and [OpenBLAS](../../glossary.md#term-openblas) thread
configurations. [Scalar](../../glossary.md#term-scalar), [vector](../../glossary.md#term-vector), and OpenBLAS choices remain distinct typed configurations rather
than flags in a [parameter](../../glossary.md#term-parameter) map. Physical vector [lanes](../../glossary.md#term-lane) are constrained by hardware and supported
species, so no candidate promises arbitrary lanes. Concrete safe-[heuristic](../../glossary.md#term-heuristic) thresholds remain
[backend](../../glossary.md#backend)-private and are changed only with reproducible evidence.

## Current ordinary-production decision

`CpuBackendComposition` retains [scalar](../../glossary.md#term-scalar) single-thread portable execution: maximum and available
parallelism are one, no worker group is supplied, exact/default numerics remain fixed, and [Runtime](../../glossary.md#term-runtime)
makes no decision. Explicit internal analysis and tuning inputs may still request the existing
typed [Vector API](../../glossary.md#term-vector-api) candidates; `CpuPartitionAnalysisInputs.DEFAULT` and ordinary production do not.

[CPU task 0010M](../../planning/backends/cpu/tasks/0010m-evidence-backed-production-defaults.md)<span data-glossary-navigation="true"> ([CPU glossary](../../glossary.md#term-cpu))</span>
rejected a global production [vector](../../glossary.md#term-vector) preference. The attempted Apple M3 Max comparison measured
only dense [FLOAT32](../../glossary.md#term-float32) ADD and direct [convolution](../../glossary.md#term-channels-first-convolution-layers) among the substantially broader [pointwise](../../glossary.md#term-pointwise), [MATMUL](../../glossary.md#matrix-multiplication--matmul),
and [MSE](../../glossary.md#mean-squared-error--mse) policy it would enable. It also lacked cross-[target](../../glossary.md#term-target) strata, selected-plan facts, randomized
paired execution, and a valid retained-[batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) floor. Those reports and aggregates were invalidated.

The retained [schema](../../glossary.md#term-schema)-3 scalar reports under
`modules/backend/cpu/evidence/cpu-0010m-defaults/` validate the hardened observational harness only. They
are machine-local and explicitly ineligible to authorize a route/default change because public
[Engine](../../glossary.md#term-engine) does not expose [backend](../../glossary.md#backend)-private selected-plan facts. The [manifest](../../glossary.md#term-manifest) retains a minimum future
acceptance framework only. A future reviewed task must seal the exact target strata, workload
cases, process order, aggregation, and uncertainty procedure before measuring either configuration.

Parallelism, [materialization](../../glossary.md#materialization), partial reductions, MATMUL/[OpenBLAS](../../glossary.md#term-openblas) thresholds, tiles,
[fusion](../../glossary.md#term-fusion)/[decomposition](../../glossary.md#term-decomposition) ceilings, and [specialization](../../glossary.md#term-specialization) budgets also remain unchanged. [Benchmarking](../../glossary.md#term-benchmarking)
never mutates any of these settings.

## Risks and [validation](../../glossary.md#term-validation)

- Splitting routes into separate [backends](../../glossary.md#backend) would confuse [ownership](../../glossary.md#term-ownership-meanings) with implementation.
- Choosing in [runtime](../../glossary.md#term-runtime) would add [graph](../../glossary.md#graph) inspection and branching to the [hot path](../../glossary.md#term-hot-path).
- A family-wide [cache key](../../glossary.md#term-cache-key) would reuse one configuration across incompatible [shapes](../../glossary.md#shape) and [layouts](../../glossary.md#layout).
- A generic [parameter](../../glossary.md#term-parameter) map would move [CPU](../../glossary.md#term-cpu) vocabulary into shared orchestration.
- Accepting an optimized route without reference comparisons could hide numerical differences.
- Using benchmarks without fixed environment and inputs would produce weak evidence.

Route changes require unit and backend-[conformance](../../glossary.md#term-conformance) tests, native cleanup tests where applicable,
and reproducible benchmarks. See [CPU backend guide](../../backend-guide/cpu-backend.md),
[Kernel routes](../../backend-guide/kernel-routes.md)<span data-glossary-navigation="true"> ([Kernel glossary](../../glossary.md#kernel))</span>, and the
[CPU master plan](../../planning/backends/cpu/master-plan.md).
