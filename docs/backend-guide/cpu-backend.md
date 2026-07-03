# CPU backend (planned implementation)

## Outcome and status

This guide defines the intended CPU integration boundary and helps contributors avoid treating CPU routes as separate backends. `backends/cpu` and `backends/openblas-provider` currently contain build structure and placeholder module markers, not backend contracts or execution behavior.

## Planned scope

The CPU backend will own capability reporting, partition lowering, specialization, fusion, scalar and optimized routes, executable units, host-side backend storage/workspaces, and typed tracing. Scalar, Vector API, ASM, specialized, fused, and OpenBLAS implementations are routes within one CPU owner.

```text
planning: owner = CPU
CPU prepare: choose scalar / Vector API / OpenBLAS / specialized / fused
runtime: invoke prepared CPU executable
```

The low-level OpenBLAS provider owns library loading, symbol binding, GEMM calls, and thread control only. Dependency direction is `backends/cpu -> backends/openblas-provider`.

## Toolchain and resources

The project baseline is JDK 26. The Vector API remains an incubator module and is not enabled globally; a focused CPU task must configure and validate it. OpenBLAS and any native binding require platform-specific setup and explicit library, symbol, threading, and lifetime contracts.

## Limitations and validation

No CPU operation coverage, route threshold, native requirement, concurrency guarantee, or performance result is implemented or promised. Future work must compare optimized routes with a scalar reference through backend-conformance tests and keep benchmarks reproducible.

See the [CPU master plan](../planning/backends/cpu/master-plan.md), [kernel routes](kernel-routes.md), and [CPU kernel strategy](../design/notes/cpu-kernel-strategy.md).
