# Select [backend](../glossary.md#backend)-local [kernel](../glossary.md#kernel) routes (planned contract)

## Outcome and status

This guide explains where a concrete implementation route is selected. No [backend](../glossary.md#backend) or [prepare](../glossary.md#prepare) implementation exists yet.

### Reading the planned description against the implementation

The “no implementation” sentence above records this guide's original planning checkpoint;
it is not the current repository status. The
[CPU route pipeline](../explanation/packages/cpu.md#read-the-pipeline-before-the-packages) and
[Metal preparation stages](../explanation/packages/metal.md#2-lower-once-declare-resources-then-compile)
explain implemented callers and resources. Shared Prepare calls the chosen backend's analysis
with immutable partition descriptions; backend finalization returns executable recipes after
slot assignment. Runtime calls those recipes with each run's bound storage, not with a request
to choose a new route. CUDA remains planned.


[Planning](../glossary.md#term-planning) chooses [backend ownership](../glossary.md#backend-ownership). During prepare, the owning backend chooses a [kernel](../glossary.md#kernel) or executable route:

```text
planning: owner = CPU
prepare:  scalar | Vector API | generated JVM bytecode | OpenBLAS | specialized | fused
run:      invoke the selected PreparedExecutable
```

[CPU](../glossary.md#term-cpu) routes are not separate backends. The same rule applies to [MPSGraph](../glossary.md#term-mpsgraph) versus custom [Metal](../glossary.md#term-metal) [kernels](../glossary.md#kernel) and to multiple [CUDA](../glossary.md#term-cuda) kernels.

A **kernel** is a computation routine; a **route** is the implementation choice that uses such
routines within one backend. “Generated [JVM](../glossary.md#term-java-virtual-machine-jvm) [bytecode](../glossary.md#term-bytecode)” means [Java Virtual Machine](../glossary.md#java-platform-and-build-tools) instructions
created during preparation, not a second backend or a [runtime](../glossary.md#term-runtime) [graph](../glossary.md#graph) interpreter. [SIMD](../glossary.md#simd)/[vector](../glossary.md#term-vector)
work applies operations to several value positions together; parallel work uses multiple threads.
Those are different choices, even when one route combines them.

For generated CPU computation kernels, CPU preparation owns generation and compatible [artifact](../glossary.md#term-artifact)
caching as part of its private [lowering](../glossary.md#lowering) and finalization work. The architecture permits generated
JVM bytecode without prescribing the [API](../glossary.md#term-application-programming-interface-api) used to build it.

## Decision inputs

A **candidate** is a complete valid alternative, including its resource requirements.
**[Qualification](../glossary.md#term-qualification)** establishes bounded compatibility/correctness evidence; **tuning** compares
eligible alternatives using representative measurements. A **[heuristic](../glossary.md#term-heuristic)** estimates a choice
without that measurement. None of these terms makes unsupported arithmetic legal or proves a
universal speed advantage.

A [backend](../glossary.md#backend) may consider [operation](../glossary.md#operation) semantics, [data type](../glossary.md#data-type--datatype), resolved [shape](../glossary.md#shape) and [layout](../glossary.md#layout), [alignment](../glossary.md#term-alignment),
[prepare](../glossary.md#prepare) configuration, [fusion](../glossary.md#term-fusion) opportunities, workspace needs, native availability, [immutable](../glossary.md#term-immutable)
compatible workload-[cache](../glossary.md#term-cache) entries, and an explicit selected [model](../glossary.md#term-model) plan. Those inputs use backend-
owned vocabulary and must not leak into [planning](../glossary.md#term-planning). Safe backend heuristics remain the correctness
fallback when tuning is disabled or no compatible cache entry exists. The backend must not wait
for [runtime](../glossary.md#term-runtime) to inspect the [graph](../glossary.md#graph) and choose or tune a route.

Each route owns a typed, version-controlled, tested [candidate generator](../glossary.md#candidate-generator) that returns complete
valid configurations. Shared prepare and tuning orchestration handles those candidates opaquely;
it does not use a generic [parameter](../glossary.md#term-parameter) map or interpret private route fields. Operation family only
selects the appropriate generator. Reuse is keyed by a [canonical workload signature](../glossary.md#canonical-workload-signature) containing
the exact semantic, data, layout, policy, and [target](../glossary.md#term-target)-compatibility facts.

For a matrix product $A\in\mathbb R^{m\times k}$ and $B\in\mathbb R^{k\times n}$,
$m,n,k$ are non-negative integer row, output-column and contracted-feature counts. There are
$mn$ output cells and $mnk$ multiply contributions. With $(m,k,n)=(64,128,32)$, those counts
are $2048$ cells and $262144$ contributions. This counts mathematical work, not bytes,
instructions or elapsed time; it does not authorize reassociating rounded products/sums.
If $m$ or $n$ is zero there are no output cells; if $k=0$ each output has no multiply
contributors. Actual route admission and empty identities remain family-specific.
These counts may inform cold CPU comparisons but establish no threshold or speed promise.

## Failure and diagnostics

If no route can realize a capability that the [backend](../glossary.md#backend) declared, preparation fails with diagnostic context. [Runtime](../glossary.md#term-runtime) does not switch to another backend. Route traces should identify the selected typed route and relevant facts without using an unstructured string map as the primary [trace](../glossary.md#trace) [model](../glossary.md#term-model).

## [Validation](../glossary.md#term-validation) expectations

Test route predicates and candidate validity at boundaries, compare each optimized route with a
reference implementation, benchmark performance separately from correctness, and verify resource
cleanup for native routes.

See [CPU backend](cpu-backend.md)<span data-glossary-navigation="true"> ([CPU glossary](../glossary.md#term-cpu); [backend glossary](../glossary.md#backend))</span>, [Partition preparer](partition-preparer.md)<span data-glossary-navigation="true"> ([Partition glossary](../glossary.md#partition))</span>, and [Partition scoring](../architecture/partition-scoring.md).
