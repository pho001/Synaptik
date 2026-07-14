# Select backend-local kernel routes (planned contract)

## Outcome and status

This guide explains where a concrete implementation route is selected. No backend or prepare implementation exists yet.

Planning chooses backend ownership. During prepare, the owning backend chooses a kernel or executable route:

```text
planning: owner = CPU
prepare:  scalar | Vector API | OpenBLAS | specialized | fused
run:      invoke the selected PreparedExecutable
```

CPU routes are not separate backends. The same rule applies to MPSGraph versus custom Metal kernels and to multiple CUDA kernels.

## Decision inputs

A backend may consider operation semantics, data type, resolved shape and layout, alignment,
prepare configuration, fusion opportunities, workspace needs, native availability, immutable
compatible workload-cache entries, and an explicit selected model plan. Those inputs use backend-
owned vocabulary and must not leak into planning. Safe backend heuristics remain the correctness
fallback when tuning is disabled or no compatible cache entry exists. The backend must not wait
for runtime to inspect the graph and choose or tune a route.

Each route owns a typed, version-controlled, tested candidate generator that returns complete
valid configurations. Shared prepare and tuning orchestration handles those candidates opaquely;
it does not use a generic parameter map or interpret private route fields. Operation family only
selects the appropriate generator. Reuse is keyed by a canonical workload signature containing
the exact semantic, data, layout, policy, and target-compatibility facts.

For a matrix multiplication `[64, 128] × [128, 32]`, the output has `64 × 32 = 2,048` values and performs `64 × 128 × 32 = 262,144` multiply contributions. Such size facts may help CPU prepare compare routes. No threshold or performance promise is currently defined.

## Failure and diagnostics

If no route can realize a capability that the backend declared, preparation fails with diagnostic context. Runtime does not switch to another backend. Route traces should identify the selected typed route and relevant facts without using an unstructured string map as the primary trace model.

## Validation expectations

Test route predicates and candidate validity at boundaries, compare each optimized route with a
reference implementation, benchmark performance separately from correctness, and verify resource
cleanup for native routes.

See [CPU backend](cpu-backend.md), [Partition preparer](partition-preparer.md), and [Partition scoring](../architecture/partition-scoring.md).
