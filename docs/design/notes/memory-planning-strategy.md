# Memory planning strategy

## Purpose and status

This note separates logical compile-time requirements from prepared physical resources and per-run residency. The relevant planning, prepare, runtime, and backend contracts are not implemented.

## Three levels

```text
compile/planning          prepare                    run
logical requirements -> physical slots/plan -> current residency/state
```

Planning may describe value lifetimes, logical materialization, and estimated transfer needs using graph facts. It must not allocate buffers or inspect current residency.

Prepare creates physical buffer and workspace slots, maps logical needs to resources, and schedules transfers or materialization. Concrete backends provide storage and workspace implementations.

Run tracks which prepared representation is current for one invocation and follows the schedule. It does not redesign the memory plan.

## Reuse example

If `ValueId(4)` dies before `ValueId(9)` is produced, prepare may map both to slot 2. The logical IDs remain different, while the physical storage can be reused because lifetimes do not overlap. This is why graph IDs cannot double as memory-slot IDs.

## Open implementation detail

Allocation algorithms, alignment, alias analysis, workspace pooling, ownership of externally supplied memory, dynamic-shape specialization, concurrency, and out-of-memory diagnostics remain for focused tasks. Any solution must keep physical addresses and mutable residency out of compile artifacts.

See [Lifecycle](../../architecture/lifecycle.md), [Runtime/prepare/backend boundary](../../architecture/runtime-prepare-backend-boundary.md), and the [planning master plan](../../planning/modules/planning/master-plan.md).
