# CPU kernel strategy

## Purpose and status

This pre-implementation note explains how CPU execution routes fit behind one backend owner. It does not define thresholds, supported operations, or performance guarantees.

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
species, so no candidate promises arbitrary lanes. For `[64, 128] × [128, 32]`, 262,144 multiply
contributions provide a concrete size fact, but no route threshold is established yet.

## Risks and validation

- Splitting routes into separate backends would confuse ownership with implementation.
- Choosing in runtime would add graph inspection and branching to the hot path.
- A family-wide cache key would reuse one configuration across incompatible shapes and layouts.
- A generic parameter map would move CPU vocabulary into shared orchestration.
- Accepting an optimized route without reference comparisons could hide numerical differences.
- Using benchmarks without fixed environment and inputs would produce weak evidence.

Future tasks require unit and backend-conformance tests for route behavior, native cleanup tests, and reproducible benchmarks. See [CPU backend guide](../../backend-guide/cpu-backend.md), [Kernel routes](../../backend-guide/kernel-routes.md), and the [CPU master plan](../../planning/backends/cpu/master-plan.md).
