# Memory [planning](../../glossary.md#term-planning) strategy


## Editorial reading context

The implementation-status statements below describe this note's original
frontier. They are not current permission to add physical allocation to Planning.
Use the [live lifecycle explanation](../../architecture/lifecycle.md) for current
prepared and per-run contracts.

A logical value identifies what must be preserved; a slot identifies a planned
place to hold a resource; a backend representation is the actual storage used in
one run. Shared Prepare calls backend analysis for exact requirements and assigns
slots. Runtime later requests per-run allocation through backend implementations.
Two values having disjoint lifetimes can motivate reuse, but the hypothetical slot
example below is not evidence that a particular reuse algorithm shipped. See the
[resource handoff](../../architecture/runtime-prepare-backend-boundary.md) for who
borrows inputs, owns scratch and retains published outputs.

## Purpose and status

This note separates current logical [compile](../../glossary.md#compile)-time requirements from planned prepared physical
resources and per-[run](../../glossary.md#term-run) [residency](../../glossary.md#residency). [Planning](../../glossary.md#term-planning) now exposes [immutable](../../glossary.md#term-immutable) `LogicalMemoryRequirement` and
`LogicalMemoryPlan` records and contains package-private derivation from a closed [graph](../../glossary.md#graph) and ordered
complete partitions. Public planning orchestration, [compiler](../../glossary.md#term-compiler) integration, physical memory,
[prepare](../../glossary.md#prepare), [runtime](../../glossary.md#term-runtime), and [concrete backend](../../glossary.md#concrete-backend) behavior remain planned.

## Three levels

```text
compile/planning          prepare                    run
logical requirements -> physical slots/plan -> current residency/state
```

[Planning](../../glossary.md#term-planning) may describe value lifetimes, logical [materialization](../../glossary.md#materialization), and estimated transfer needs using [graph](../../glossary.md#graph) facts. It must not allocate [buffers](../../glossary.md#term-buffer) or inspect current [residency](../../glossary.md#residency).

The current plan deliberately stops before lifetimes and transfer estimates. It contains one
requirement per `CompiledGraphModel.values()` entry, in that exact order:

- the graph-local [`ValueId`](../../glossary.md#logical-valueid-versus-physical-memory-slot) and exact retained `TensorDescriptor`;
- the optional producing `PlannedPartition`, empty for a graph input;
- each distinct consuming [partition](../../glossary.md#partition) in partition-list order; and
- whether `CompiledGraphModel.outputs()` declares the value.

From these primitive facts, a later consumer can recognize partition inputs and outputs,
same-[owner](../../glossary.md#term-owner) or cross-owner boundaries, partition-internal values, and graph-output preservation.
The roles overlap, so the plan does not store a separate role enum.

[Prepare](../../glossary.md#prepare) creates physical buffer and workspace slots, maps logical needs to resources, and schedules transfers or materialization. Concrete [backends](../../glossary.md#backend) provide storage and workspace implementations.

[Run](../../glossary.md#term-run) tracks which prepared representation is current for one invocation and follows the schedule. It does not redesign the memory plan.

## Current logical example

Suppose [partition](../../glossary.md#partition) `p0` produces `ValueId(4)`, partition `p1` and nonconsecutive partition `p2`
consume it, and the [graph](../../glossary.md#graph) also declares it as an output. Its generated requirement retains
[producer](../../glossary.md#term-producer) `p0`, consumers `[p1, p2]`, and `graphOutput == true`. The requirement says that the
logical value must reach both consuming regions and remain available at the graph boundary. It
does not say whether either availability uses [aliasing](../../glossary.md#term-aliasing), copying, transfer, recomputation, or a
[backend](../../glossary.md#backend)-native representation.

A dynamic or expression-[dimension](../../glossary.md#dimension) value retains its full `TensorDescriptor`. [Planning](../../glossary.md#term-planning) does not
replace an unresolved [extent](../../glossary.md#term-extent) with a byte count, and logical `DataType.byteWidth()` does not choose
backend [alignment](../../glossary.md#term-alignment) or [padding](../../glossary.md#padding).

## Planned physical reuse example

If `ValueId(4)` dies before `ValueId(9)` is produced, [prepare](../../glossary.md#prepare) may map both to slot 2. The logical IDs remain different, while the physical storage can be reused because lifetimes do not overlap. This is why [graph](../../glossary.md#graph) IDs cannot double as memory-slot IDs.

## Open implementation detail

The current logical plan does not accept `ForwardPublicationBinding` or
`GradientPublicationBinding`; [graph](../../glossary.md#graph) outputs provide only a preservation obligation, while the
[compiler](../../glossary.md#term-compiler)-owned `PublicationPlan` owns [tensor](../../glossary.md#tensor)-to-value [publication](../../glossary.md#term-publication) context. Allocation algorithms,
[alignment](../../glossary.md#term-alignment), [alias](../../glossary.md#aliasing-and-arena) analysis, workspace [pooling](../../glossary.md#term-pooling),
[ownership](../../glossary.md#term-ownership-meanings) of externally supplied memory, dynamic-[shape](../../glossary.md#shape) [specialization](../../glossary.md#term-specialization), [concurrency](../../glossary.md#term-concurrency), and
out-of-memory diagnostics remain for focused tasks. Any solution must keep physical addresses and
[mutable](../../glossary.md#term-mutable) [residency](../../glossary.md#residency) out of [compile artifacts](../../glossary.md#compile-artifacts).

See [Lifecycle](../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../glossary.md#lifecycle))</span>, [Runtime/prepare/backend boundary](../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../glossary.md#term-runtime); [prepare glossary](../../glossary.md#prepare); [backend glossary](../../glossary.md#backend))</span>, and the [planning master plan](../../planning/modules/planning/master-plan.md)<span data-glossary-navigation="true"> ([planning glossary](../../glossary.md#term-planning))</span>.
