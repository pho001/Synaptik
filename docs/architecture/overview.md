# Architecture Overview

This document explains the architecture defined by the authoritative [`ARCHITECTURE.md`](../../ARCHITECTURE.md). It summarizes the system without replacing the contract.

The layer descriptions cover implemented and planned responsibilities, not a uniform support
promise. The [current support domains](../backend-guide/production-support.md#status-and-interpretation)
separate CPU/Metal execution evidence from planned CUDA work and from cold persistence support.

## High-level architecture

Synaptik turns a public [tensor](../glossary.md#tensor) expression into an [immutable](../glossary.md#term-immutable) [compile](../glossary.md#compile)-time recipe, prepares that recipe for explicitly registered [backends](../glossary.md#backend), and then executes a [prepared schedule](../glossary.md#prepared-schedule--preparedschedule). The architecture keeps semantic [graph](../glossary.md#graph) work separate from backend implementation choices and per-[run state](../glossary.md#run-state--runstate).

Here **semantic** means what a computation is required to calculate; **physical** means the code,
storage and [device](../glossary.md#term-device) used to calculate it. A graph records computations as **[nodes](../glossary.md#node)** and the data
between them as **values**. A backend is an execution [owner](../glossary.md#term-owner) such as [CPU](../glossary.md#term-cpu) or [Metal](../glossary.md#term-metal), not a particular
[kernel](../glossary.md#kernel). A [partition](../glossary.md#partition) groups consecutive nodes assigned to one owner; it is not necessarily one
[fused](../glossary.md#specialization-and-fusion) executable. The [framework walkthrough](../explanation/framework.md) introduces these
relationships with one small [model](../glossary.md#term-model) before the detailed [module](../glossary.md#term-module) inventory below.

The main layers are:

- **Model** owns tensors, [operation](../glossary.md#operation) semantics, [shape](../glossary.md#shape), [data type](../glossary.md#data-type--datatype), [layout](../glossary.md#layout), [host storage](../glossary.md#host-storage) abstractions, and the immutable graph model. It has no backend or [runtime](../glossary.md#term-runtime) responsibility.
- **[Planning](../glossary.md#term-planning)** makes backend-neutral compile-time decisions: intent, capability, [ownership](../glossary.md#term-ownership-meanings) scoring, partitioning, and logical memory or [materialization](../glossary.md#materialization) requirements.
- **[Compiler](../glossary.md#term-compiler)** [captures](../glossary.md#term-capture-meanings) and transforms the graph, expands [autograd](../glossary.md#autograd) when requested, coordinates planning, and produces immutable `CompileArtifacts`.
- **[Prepare](../glossary.md#prepare)** validates the transition from [compile artifacts](../glossary.md#compile-artifacts) to executable runtime state and defines the shared backend preparation contract.
- **Backend** implementations lower their assigned partitions, fuse and specialize work, choose concrete kernel routes, and provide backend storage and workspace implementations.
- **Runtime** executes prepared schedules and manages dynamic run state, [residency](../glossary.md#residency), transfers, materialization, and [publication](../glossary.md#term-publication).
- **[Engine](../glossary.md#term-engine)** is the [composition root](../glossary.md#term-composition-root) and public [lifecycle](../glossary.md#lifecycle) [facade](../glossary.md#term-facade). It wires the compiler, prepare layer, runtime, and explicitly registered backends.
- **[Trace](../glossary.md#trace)** is a [dependency](../glossary.md#term-dependency)-[leaf](../glossary.md#term-dependency-leaf) module containing typed diagnostic DTOs only.

See [Module Boundaries](module-boundaries.md) for detailed ownership and [Dependency Rules](dependency-rules.md) for allowed [dependency direction](../glossary.md#java-platform-and-build-tools).

## Why [compile](../glossary.md#compile), [prepare](../glossary.md#prepare), and [run](../glossary.md#term-run) are separate

Each stage answers a different kind of question:

1. **Compile** decides what the [graph](../glossary.md#graph) means, how it is optimized, and which [backend](../glossary.md#backend) owns each [node](../glossary.md#node) or segment. Its output is an [immutable](../glossary.md#term-immutable) recipe, not executable backend state.
2. **Prepare** turns the recipe into executable state. Concrete backends lower partitions and choose implementations; shared orchestration creates and validates memory and schedule structures.
3. **Run** executes only the already-[prepared schedule](../glossary.md#prepared-schedule--preparedschedule). It handles inputs and [mutable](../glossary.md#term-mutable) state without repeating [graph optimization](../glossary.md#forward-graph-optimization), backend discovery, [lowering](../glossary.md#lowering), or [kernel](../glossary.md#kernel) selection.

This separation keeps compile-time reasoning independent of physical [buffers](../glossary.md#term-buffer), keeps concrete implementation selection inside backends, and keeps the [runtime](../glossary.md#term-runtime) [hot path](../glossary.md#term-hot-path) free of [`Operation`](../glossary.md#operation) and `CompiledNode`.

### Who calls each stage?

An application calls `Model.compute`, or uses `Engine.compile`, `prepare` and `run` explicitly.
In the Model path, NN makes those Engine calls internally. Engine delegates expression capture
to Compiler, passes immutable compile recipes to shared Prepare, and invokes Runtime with the
prepared schedule and this call's stored inputs. Prepare calls each selected backend's analysis
and finalization; Runtime later invokes the resulting executable, not the semantic graph.

Compilation reads descriptions rather than taking ownership of input memory. Prepared owners
retain reusable executable resources; each `RunResult` protects one run's resources and borrowed
inputs. Explicit materialization creates independent host values. The
[call-by-call ownership table](../explanation/ownership.md#which-call-borrows-or-copies) explains
why these are three different lifetimes, not three copies of the same Tensor.

## [Lifecycle](../glossary.md#lifecycle) summary

```text
Tensor expression
  -> GraphCompiler
  -> CompileArtifacts
     - CompiledGraphModel
     - PlannedPartition[]
     - LogicalMemoryPlan
     - PublicationPlan
  -> prepare
  -> BackendPartitionAnalysis[]
  -> PreparedMemoryPlan with assigned slots
  -> PreparedPartition[]
  -> PreparedSchedule
  -> PreparedExecution
  -> RunState
  -> execute
```

The central state distinction is:

“[Immutable](../glossary.md#term-immutable)” means the recipe's recorded decisions do not change. It does not mean that input
values are frozen or that closeable resources live forever. A reusable recipe and one invocation's
[mutable](../glossary.md#term-mutable) memory have different lifetimes; see [ownership and lifetimes](../explanation/ownership.md)<span data-glossary-navigation="true"> ([ownership glossary](../glossary.md#term-ownership-meanings))</span>.

- `CompileArtifacts` are immutable [compile](../glossary.md#compile)-time output.
- [`PreparedExecution`](../glossary.md#prepared-execution--preparedexecution) is prepared [runtime](../glossary.md#term-runtime) state reusable across runs.
- [`RunState`](../glossary.md#run-state--runstate) is mutable state for an individual [run](../glossary.md#term-run).
- A [`PreparedExecutable`](../glossary.md#prepared-executable--preparedexecutable) computes only the prepared region assigned to it.

The complete stage-by-stage flow is described in [Lifecycle](lifecycle.md). The [handoff](../glossary.md#term-handoff) among [prepare](../glossary.md#prepare), runtime, and concrete [backends](../glossary.md#backend) is described in [Runtime, Prepare, and Backend Boundary](runtime-prepare-backend-boundary.md).

## Boundary scenario

Consider a future [matrix multiplication](../glossary.md#matrix-multiplication--matmul) followed by an [elementwise](../glossary.md#term-elementwise) addition. [Compile](../glossary.md#compile) may assign both [nodes](../glossary.md#node) to [CPU](../glossary.md#term-cpu) and place them in one planned [partition](../glossary.md#partition). CPU [prepare](../glossary.md#prepare) may lower that partition to an [OpenBLAS](../glossary.md#term-openblas) matrix multiplication plus a [scalar](../glossary.md#term-scalar) or vectorized addition. [Run](../glossary.md#term-run) invokes those prepared units with bound inputs. Moving the OpenBLAS choice into [planning](../glossary.md#term-planning) would violate [backend-owned lowering](../glossary.md#backend-owned-lowering); passing the original operations back into [runtime](../glossary.md#term-runtime) would violate the hot-path boundary.

## Repository structure

The repository groups responsibilities by architectural role:

```text
modules/       shared model, configuration, planning, compiler, prepare,
               runtime, engine, generic training/checkpoint, backend contracts, and tracing
modules/backend/ concrete CPU, Metal, and CUDA backends, plus leaf providers
extensions/    NN composition (including retained old checkpoint readers) and ONNX
tools/         tuning, benchmarks, and CLI tooling
modules/backend/metal/macos-arm64/ platform-specific Metal native integration
testing/       architecture, backend-conformance, and integration tests
docs/          explanatory architecture, design, user, backend, and developer docs
```

Production Java packages use the `io.github.pho001.synaptik.*` namespace. [Module](../glossary.md#term-module) directories remain
concise, such as `modules/compiler`, `modules/backend/cpu`, and `modules/training`. Training retains
[Gradle](../glossary.md#term-gradle) [identity](../glossary.md#term-identity-meanings) `:extensions:training` and its Java packages through an explicit settings mapping.
Its generic live [admission](../glossary.md#term-admission) depends only on public [Model](../glossary.md#term-model)/[Engine](../glossary.md#term-engine); [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) supplies a live Module adapter.

Generic [Checkpoint](../glossary.md#term-checkpoint) uses Model publicly and [Compiler](../glossary.md#term-compiler)'s cold source boundary privately; NN consumes
Checkpoint and public Engine for its retained old readers. A [graph](../glossary.md#graph) load creates [heap](../glossary.md#term-heap) state and
symbolic expressions, not a [prepared executable](../glossary.md#prepared-executable--preparedexecutable) or a new Model workflow.
Compiler owns closed cold source admission/reconstruction; Checkpoint owns the selective
v1/v2/v3 wire. The selected default [CNN](../glossary.md#term-convolutional-neural-network-cnn)/static recurrent graphs and fixed flat scans use v3.
A two-[Linear](../glossary.md#term-linear-meanings) graph with RELU, [TANH](../glossary.md#term-tanh) or [SIGMOID](../glossary.md#term-sigmoid) acquires private state through the same
NN path and saves as v2; a graph using only the original five kinds still saves byte-identical v1.
This adds neither an execution pipeline nor a [backend capability](../glossary.md#backend-capability). See the
[graph format compatibility](../reference/checkpoint/graph-formats.md#selective-graph-format-versions).

NN's fluent `Model.builder()` instead creates private layers and retains their ordinary [Tensor
composition](../glossary.md#tensor-composition). Only supplied [parameter](../glossary.md#term-parameter) [leaves](../glossary.md#term-leaf) pass through the zero-[node](../glossary.md#node) raw copier, before any
expression is built. A selected default CNN or statically [unrolled](../glossary.md#backpropagation-through-time-and-recurrent-state) recurrent model saves its
ordinary graph, then computes through the same public Engine preparations after cold loading.
Save rejects operations outside the narrow source slice before touching a file; that failure leaves
execution usable. Fixed scans persist cold only, without [backend](../glossary.md#backend) execution or [BPTT](../glossary.md#term-backpropagation-through-time-bptt). This distinction
prevents [persistence](../glossary.md#term-persistence) limits from becoming execution limits without adding a reconstruction table.
See [ADR 0040](../design/decisions/0040-fluent-model-builder.md)<span data-glossary-navigation="true"> ([ADR glossary](../glossary.md#term-architecture-decision-record-adr))</span>.
