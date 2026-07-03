# CPU kernel strategy

## Purpose and status

This pre-implementation note explains how CPU execution routes fit behind one backend owner. It does not define thresholds, supported operations, or performance guarantees.

## Strategy

Planning selects only `owner = CPU`. CPU prepare examines the owned partition and chooses among scalar, Vector API, ASM, OpenBLAS, specialized, or fused routes. Runtime invokes the resulting `PreparedExecutable` without repeating that choice.

Scalar code should provide a readable reference path for selected capabilities. Optimized routes must preserve its specified semantics and failure behavior. OpenBLAS stays behind the CPU backend through the low-level provider; Vector API configuration stays local to the CPU module.

## Decision factors

Prepare-time route selection may use operation and layout facts, data type, sizes, alignment, fusion opportunity, workspace requirements, native availability, and immutable tuning profiles. For `[64, 128] × [128, 32]`, 262,144 multiply contributions provide a concrete size fact, but no route threshold is established yet.

## Risks and validation

- Splitting routes into separate backends would confuse ownership with implementation.
- Choosing in runtime would add graph inspection and branching to the hot path.
- Accepting an optimized route without reference comparisons could hide numerical differences.
- Using benchmarks without fixed environment and inputs would produce weak evidence.

Future tasks require unit and backend-conformance tests for route behavior, native cleanup tests, and reproducible benchmarks. See [CPU backend guide](../../backend-guide/cpu-backend.md), [Kernel routes](../../backend-guide/kernel-routes.md), and the [CPU master plan](../../planning/backends/cpu/master-plan.md).
