# Memory planning strategy

## Purpose and status

This note separates current logical compile-time requirements from planned prepared physical
resources and per-run residency. Planning now exposes immutable `LogicalMemoryRequirement` and
`LogicalMemoryPlan` records and contains package-private derivation from a closed graph and ordered
complete partitions. Public planning orchestration, compiler integration, physical memory,
prepare, runtime, and concrete backend behavior remain planned.

## Three levels

```text
compile/planning          prepare                    run
logical requirements -> physical slots/plan -> current residency/state
```

Planning may describe value lifetimes, logical materialization, and estimated transfer needs using graph facts. It must not allocate buffers or inspect current residency.

The current plan deliberately stops before lifetimes and transfer estimates. It contains one
requirement per `CompiledGraphModel.values()` entry, in that exact order:

- the graph-local `ValueId` and exact retained `TensorDescriptor`;
- the optional producing `PlannedPartition`, empty for a graph input;
- each distinct consuming partition in partition-list order; and
- whether `CompiledGraphModel.outputs()` declares the value.

From these primitive facts, a later consumer can recognize partition inputs and outputs,
same-owner or cross-owner boundaries, partition-internal values, and graph-output preservation.
The roles overlap, so the plan does not store a separate role enum.

Prepare creates physical buffer and workspace slots, maps logical needs to resources, and schedules transfers or materialization. Concrete backends provide storage and workspace implementations.

Run tracks which prepared representation is current for one invocation and follows the schedule. It does not redesign the memory plan.

## Current logical example

Suppose partition `p0` produces `ValueId(4)`, partition `p1` and nonconsecutive partition `p2`
consume it, and the graph also declares it as an output. Its generated requirement retains
producer `p0`, consumers `[p1, p2]`, and `graphOutput == true`. The requirement says that the
logical value must reach both consuming regions and remain available at the graph boundary. It
does not say whether either availability uses aliasing, copying, transfer, recomputation, or a
backend-native representation.

A dynamic or expression-dimension value retains its full `TensorDescriptor`. Planning does not
replace an unresolved extent with a byte count, and logical `DataType.byteWidth()` does not choose
backend alignment or padding.

## Planned physical reuse example

If `ValueId(4)` dies before `ValueId(9)` is produced, prepare may map both to slot 2. The logical IDs remain different, while the physical storage can be reused because lifetimes do not overlap. This is why graph IDs cannot double as memory-slot IDs.

## Open implementation detail

The current logical plan does not accept `PublicationBinding`; graph outputs provide only a
preservation obligation, while a future compiler-owned `PublicationPlan` will own tensor-to-value
publication context. Allocation algorithms, alignment, alias analysis, workspace pooling,
ownership of externally supplied memory, dynamic-shape specialization, concurrency, and
out-of-memory diagnostics remain for focused tasks. Any solution must keep physical addresses and
mutable residency out of compile artifacts.

See [Lifecycle](../../architecture/lifecycle.md), [Runtime/prepare/backend boundary](../../architecture/runtime-prepare-backend-boundary.md), and the [planning master plan](../../planning/modules/planning/master-plan.md).
