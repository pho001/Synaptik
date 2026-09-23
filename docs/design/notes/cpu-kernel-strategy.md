# CPU kernel strategy

## Purpose and status

This note explains how CPU execution routes fit behind one backend owner and records the current
ordinary-production safe heuristic. It does not define support or performance guarantees.

## Strategy

Planning selects only `owner = CPU`. CPU prepare examines the owned partition and chooses among
scalar, Vector API, generated JVM-bytecode CPU computation-kernel, OpenBLAS, specialized, or
fused routes. CPU finalization generates or reuses a compatible selected artifact after shared
slot assignment. Runtime invokes the resulting `PreparedExecutable` without repeating that
choice.

The architecture fixes this ownership and lifecycle, not a particular bytecode-generation API.
The CPU master plan records the selected current Java 26 implementation direction.

Scalar code should provide a readable reference path for selected capabilities. Optimized routes must preserve its specified semantics and failure behavior. OpenBLAS stays behind the CPU backend through the low-level provider; Vector API configuration stays local to the CPU module.

## Decision factors

Prepare-time route selection may use operation and layout facts, data type, sizes, alignment,
fusion opportunity, workspace requirements, native availability, a compatible workload-cache
entry, and an explicit selected model plan. CPU route owners later provide typed,
version-controlled candidate generators that derive complete valid configurations from target
capabilities, workload facts, and the tuning budget. Matrix-multiplication candidates may include
supported JDK Vector API species/strategy, unroll, tile, parallelism, and OpenBLAS thread
configurations. Scalar, vector, and OpenBLAS choices remain distinct typed configurations rather
than flags in a parameter map. Physical vector lanes are constrained by hardware and supported
species, so no candidate promises arbitrary lanes. Concrete safe-heuristic thresholds remain
backend-private and are changed only with reproducible evidence.

## Current ordinary-production decision

`CpuBackendComposition` retains scalar single-thread portable execution: maximum and available
parallelism are one, no worker group is supplied, exact/default numerics remain fixed, and Runtime
makes no decision. Explicit internal analysis and tuning inputs may still request the existing
typed Vector API candidates; `CpuPartitionAnalysisInputs.DEFAULT` and ordinary production do not.

[CPU task 0010M](../../planning/backends/cpu/tasks/0010m-evidence-backed-production-defaults.md)
rejected a global production vector preference. The attempted Apple M3 Max comparison measured
only dense FLOAT32 ADD and direct convolution among the substantially broader pointwise, MATMUL,
and MSE policy it would enable. It also lacked cross-target strata, selected-plan facts, randomized
paired execution, and a valid retained-batch floor. Those reports and aggregates were invalidated.

The retained schema-3 scalar reports under
`backends/cpu/evidence/cpu-0010m-defaults/` validate the hardened observational harness only. They
are machine-local and explicitly ineligible to authorize a route/default change because public
Engine does not expose backend-private selected-plan facts. The manifest retains a minimum future
acceptance framework only. A future reviewed task must seal the exact target strata, workload
cases, process order, aggregation, and uncertainty procedure before measuring either configuration.

Parallelism, materialization, partial reductions, MATMUL/OpenBLAS thresholds, tiles,
fusion/decomposition ceilings, and specialization budgets also remain unchanged. Benchmarking
never mutates any of these settings.

## Risks and validation

- Splitting routes into separate backends would confuse ownership with implementation.
- Choosing in runtime would add graph inspection and branching to the hot path.
- A family-wide cache key would reuse one configuration across incompatible shapes and layouts.
- A generic parameter map would move CPU vocabulary into shared orchestration.
- Accepting an optimized route without reference comparisons could hide numerical differences.
- Using benchmarks without fixed environment and inputs would produce weak evidence.

Route changes require unit and backend-conformance tests, native cleanup tests where applicable,
and reproducible benchmarks. See [CPU backend guide](../../backend-guide/cpu-backend.md),
[Kernel routes](../../backend-guide/kernel-routes.md), and the
[CPU master plan](../../planning/backends/cpu/master-plan.md).
