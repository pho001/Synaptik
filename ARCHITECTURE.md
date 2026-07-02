# Synaptik Architecture Contract

This document is the authoritative architecture contract for this repository.

All contributors and agents must preserve these boundaries unless this document is explicitly updated as part of the same change.

For agent working instructions, see:

```text
AGENTS.md
```

For extended rationale and the current planning-level architecture proposal, see:

```text
docs/architecture/current-architecture-plan.md
```

Files under `docs/` are explanatory unless this document explicitly references them as normative.

## Java namespace

All production Java packages must use:

```text
io.github.pho001.synaptik.*
```

Gradle module directories may remain short and do not need the `synaptik-` prefix.

Examples:

```text
modules/compiler
modules/runtime
backends/cpu
backends/metal
extensions/training
```

Example Java packages:

```text
io.github.pho001.synaptik.model
io.github.pho001.synaptik.compiler
io.github.pho001.synaptik.runtime
io.github.pho001.synaptik.backend.cpu
io.github.pho001.synaptik.backend.metal
```

## Repository layout

The intended high-level repository layout is:

```text
ComputationalGraph/
  AGENTS.md
  ARCHITECTURE.md
  README.md
  build-logic/

  docs/
    index.md
    getting-started.md
    architecture/
    design/
    user-guide/
    backend-guide/
    developer-guide/
    api/

  modules/
    trace/
    backend-contract/
    model/
    config/
    planning/
    compiler/
    runtime/
    prepare/
    engine/

  backends/
    openblas-provider/
    cpu/
    metal/
    cuda/

  extensions/
    training/
    onnx/

  tools/
    tuning/
    benchmarks/
    cli/

  native/
    metal-macos-arm64/
    cuda/

  testing/
    architecture-tests/
    backend-conformance/
    integration-tests/
```

## Core lifecycle

The core lifecycle is:

```text
Tensor expression
  -> GraphCompiler
  -> CompileArtifacts
     - CompiledGraphModel
     - PlannedPartition[]
     - LogicalMemoryPlan
     - PublicationPlan
  -> prepare
  -> PreparedPartition[]
  -> PreparedMemoryPlan
  -> PreparedSchedule
  -> PreparedExecution
  -> RunState
  -> execute
```

Public API shape:

```java
CompiledGraph graph =
        CompiledGraph.compile(output, CompileConfig.auto());

PreparedExecution execution =
        graph.prepare(PrepareConfig.defaults());

RunResult result =
        execution.run(inputs, RunOptions.defaults());
```

## Core invariants

The following invariants must remain true:

- `Tensor` is public mutable API state.
- `Tensor` is not an IR node.
- `Operation` owns semantic behavior but never backend support.
- `Operation` must not expose `supportedBackends()`.
- `CompiledGraphModel` is immutable compile-time graph state.
- `CompileArtifacts` are immutable compile-time output.
- `PreparedExecution` is prepared runtime state.
- `RunState` is per-run mutable state.
- `PreparedExecutable` computes only its prepared region.
- Runtime hot path must not see `Operation` or `CompiledNode`.
- Compiler must not allocate physical buffers.
- Planning must not select concrete kernels.
- Planning scoring selects backend ownership, not implementation routes.
- Backend prepare owns backend-specific lowering and kernel selection.
- Runtime executes prepared schedules only.
- Engine is the composition root.
- Concrete backends must not depend on engine.
- Runtime must not depend on concrete backend implementations.

## Module responsibilities

### `modules/trace`

`modules/trace` owns typed diagnostic DTOs only.

Allowed:

- trace event envelopes
- typed compile payloads
- typed prepare payloads
- typed run payloads
- typed backend payloads
- trace-local IDs
- typed trace attributes

Forbidden:

- importing model
- importing planning
- importing compiler
- importing runtime
- importing prepare
- importing engine
- importing concrete backends
- graph traversal
- backend execution
- business logic
- runtime state

Trace must use typed DTOs.

`Map<String,String>` must not be used as the primary trace model.

Backend-specific details may use typed `TraceAttributes` as an escape hatch.

### `modules/backend-contract`

`modules/backend-contract` owns minimal backend identities and declarative requirements.

Allowed:

- `BackendId`
- `BackendDeviceId`
- `BackendAvailabilitySnapshot`
- `BackendRequirement`
- `DeviceClass`

Forbidden:

- kernel registry
- operation support logic
- backend prepare services
- executable units
- runtime storage
- physical buffers
- cost model implementation

Compile-time plans must hold backend identity, not live backend services.

Use `BackendId`, not concrete backend objects, in compile-time ownership and partitioning data.

### `modules/model`

`modules/model` owns the public tensor model, operation semantics, shape/data type/layout model, host storage abstraction, and immutable graph model.

Allowed:

- `Tensor`
- `TensorId`
- `TensorFactory`
- `DataType`
- `Shape`
- `LayoutDescriptor`
- `HostTensorStorage`
- `MemorySegmentStorage`
- `Operation`
- `OperationAttrs`
- `CompiledGraphModel`
- `CompiledNode`
- `GraphValue`
- `NodeId`
- `ValueId`
- `GraphPhase`
- `PublicationBinding`
- `TensorDescriptor`

Forbidden:

- backend support
- `supportedBackends()`
- device residency
- runtime workspaces
- physical device buffers
- kernel selection
- backend-specific storage
- prepared execution
- runtime state

`Tensor` may have public storage and gradient state, but it must not own runtime device residency.

Device storage belongs to runtime/backend layers, not model.

### `modules/config`

`modules/config` owns declarative configuration only.

Allowed:

- `CompileConfig`
- `CompileMode`
- `BackendIntent`
- `GraphOptimizationConfig`
- `PartitionScoringConfig`
- `PartitionScoringPolicy`
- `PrepareConfig`
- `CpuPrepareConfig`
- `AcceleratorPrepareConfig`
- `RunOptions`
- `PublicationPolicy`
- `PlatformProfile`
- `BackendProfile`
- `TuningProfile`

Forbidden:

- live services
- concrete backend classes
- kernel class references
- runtime state
- executable units
- backend-specific implementation logic

Backend-specific interpretation of config belongs to backend prepare.

### `modules/planning`

`modules/planning` owns backend-neutral compile-time planning.

Allowed:

- backend intent propagation
- capability query contracts
- capability matrix construction
- backend-neutral partition scoring
- node ownership decisions
- segment ownership decisions
- maximal same-owner partitioning
- logical materialization requirements
- logical memory requirements

Forbidden:

- fusion implementation
- specialization
- concrete kernel selection
- OpenBLAS route selection
- Vector API route selection
- MPSGraph route selection
- CUDA kernel selection
- physical memory allocation
- runtime residency
- prepared schedules
- prepared executables
- backend-specific DAG construction
- backend-specific lowering
- runtime execution units
- concrete kernel/runtime scoring

Planning answers:

```text
Where should this node or segment run?
```

Planning must not answer:

```text
Which concrete kernel, executable, BLAS route, MPSGraph route, or CUDA implementation should run it?
```

## Partition scoring

Planning includes backend-neutral partition scoring.

Partition scoring may use compile-time information only, such as:

- graph metadata
- op kind
- data type
- shape
- estimated element count
- estimated byte size
- backend capabilities
- backend intent
- graph phase
- producer/consumer ownership candidates
- logical materialization estimates
- transfer estimates
- boundary penalties
- accelerator bonuses
- small-region penalties
- platform profiles

Partition scoring must not use:

- current runtime residency
- current device buffers
- concrete kernel classes
- concrete MPSGraph executables
- concrete CUDA kernels
- concrete OpenBLAS calls
- current `RunState`
- physical buffer addresses
- prepared executables

Partition scoring decides backend ownership at node or segment level before maximal same-owner partitioning.

The output of scoring is ownership, not executable implementation.

Backend-specific lowering and kernel selection belong to backend prepare.

### `modules/compiler`

`modules/compiler` owns graph compilation.

Allowed:

- graph capture
- topological sorting
- producer/use indexing
- canonicalization
- shape inference
- data type inference
- validation
- dead-code elimination
- common subexpression elimination
- constant folding
- algebraic simplification
- autograd expansion
- backward graph construction
- post-autograd optimization
- publication binding
- partition planning orchestration
- logical memory planning orchestration
- compile diagnostics
- `CompileArtifacts`

Forbidden:

- physical buffers
- `PreparedSchedule`
- `PreparedExecution`
- backend executables
- concrete kernel selection
- backend-specific lowering
- runtime workspace state
- runtime residency
- concrete backend dependencies

The compiler produces immutable compile-time artifacts.

It must not construct runtime execution units.

## Compile artifacts

`CompileArtifacts` should contain:

```java
public record CompileArtifacts(
        CompiledGraphModel graph,
        List<PlannedPartition> partitions,
        LogicalMemoryPlan memory,
        PublicationPlan publication,
        CompileDiagnostics diagnostics
) {}
```

`CompileArtifacts` must not contain:

- physical buffers
- prepared executables
- backend executable objects
- concrete kernel routes
- runtime workspaces
- runtime residency state
- mutable run state

## Training graph model

In training compile modes, the compiler may expand the forward graph into a combined forward + backward graph before post-autograd optimization.

The combined graph is still compile-time graph state.

Runtime may expose separate forward and backward schedules, or a single training-step schedule, depending on prepare-time decisions.

Backends must not implement global autograd.

Backends execute prepared regions only.

Optimizer updates are either:

- backend-agnostic runtime/training steps, or
- graph operations generated by the training extension and lowered by backend prepare

Training must not depend on concrete backend modules.

### `modules/runtime`

`modules/runtime` owns prepared execution contracts and dynamic runtime state.

Allowed:

- `PreparedExecution`
- `PreparedUnit`
- `PreparedExecutable`
- `PreparedSchedule`
- `PreparedMemoryPlan`
- `BufferSlot`
- `WorkspaceSlot`
- `RuntimeSlotTable`
- `RunState`
- residency management
- transfer execution
- publication execution
- runtime resources
- prepared execution runner

Forbidden:

- graph optimization
- autograd construction
- compiler passes
- backend discovery
- service lookup for backend implementations
- concrete backend dependencies
- kernel selection
- backend-specific lowering
- `Operation` in hot path
- `CompiledNode` in hot path

Runtime executes prepared schedules.

Runtime does not decide how graph partitions should be lowered.

Runtime does not select kernels.

Runtime does not discover backend plugins.

### `modules/prepare`

`modules/prepare` owns shared prepare contracts and validation.

Allowed:

- `PrepareContext`
- `BackendPartitionPreparer`
- `PreparedPartition`
- partition coverage validation
- prepared memory validation
- prepared schedule validation

Forbidden:

- concrete CPU lowering
- concrete Metal lowering
- concrete CUDA lowering
- concrete kernel selection
- backend-specific executable implementation
- backend-specific storage implementation

Prepare is the bridge between compile artifacts and runtime.

Concrete backend prepare implementations live in concrete backend modules.

## Concrete backend modules

Concrete backend modules own concrete backend implementation.

Examples:

```text
backends/cpu
backends/metal
backends/cuda
```

Allowed:

- capability provider
- backend-owned prepare/lowering
- backend-specific fusion
- backend-specific specialization
- kernel route selection
- executable units
- backend storage
- backend workspace
- backend trace contribution
- native bridge integration

Forbidden:

- public Tensor API ownership
- global graph compiler logic
- engine dependency
- service locator ownership
- runtime plugin discovery ownership
- changing module ownership rules

Concrete backend modules may depend on:

- model
- config
- planning
- runtime
- prepare
- backend-contract
- trace

Concrete backend modules must not depend on engine.

## CPU backend routes

CPU scalar, CPU Vector API, CPU ASM, and OpenBLAS are routes inside the CPU backend.

They are not separate backends.

Planning chooses:

```text
owner = CPU
```

CPU prepare chooses:

```text
scalar route
Vector API route
OpenBLAS route
specialized kernel
fused kernel
```

Do not create separate backend modules such as:

```text
cpu-scalar
cpu-vector
cpu-blas
```

unless this document is updated first.

## Metal backend

Metal backend owns:

- MPSGraph lowering
- MPSGraph executable creation
- custom Metal kernel routes
- Metal storage
- native bridge integration
- Metal-specific materialization
- Metal trace contributions

Metal-specific optimizer execution belongs to Metal backend prepare/kernels, not to training.

Do not add `MetalOptimizerBridge` to `extensions/training`.

## OpenBLAS provider

`backends/openblas-provider` is a low-level leaf provider.

Allowed:

- OpenBLAS library loading
- symbol binding
- GEMM calls
- thread control

Forbidden:

- config interpretation
- planning
- fallback logic
- prepared execution
- Tensor API
- runtime residency
- backend ownership decisions

The dependency direction is:

```text
backends/cpu -> backends/openblas-provider
```

Never the reverse.

### `modules/engine`

`modules/engine` owns public lifecycle orchestration and composition.

Allowed:

- public `CompiledGraph` facade
- explicit backend registration
- compile orchestration
- prepare orchestration
- wiring compiler, runtime, prepare, and concrete backends

Forbidden:

- kernel implementations
- backend internals
- graph optimizer passes
- runtime service locator
- reflective plugin discovery as the core backend mechanism

Engine is the composition root.

Backends are registered explicitly.

Example:

```java
SynaptikEngine engine = SynaptikEngine.builder()
        .addBackend(cpuBackend())
        .addBackend(metalBackend())
        .build();
```

## Runtime service locator

A runtime service locator is forbidden as a core mechanism.

A runtime service locator means runtime dynamically asks for services or backends during execution, for example:

```java
Backend backend = RuntimeServices.get("metal");
KernelRegistry kernels = RuntimeServices.get(KernelRegistry.class);
```

This is forbidden because runtime must execute already-prepared schedules.

Backend selection and executable construction must happen before runtime hot path execution.

## Reflective backend plugin discovery

Reflective backend plugin discovery is forbidden as the core backend mechanism.

Examples include:

- classpath scanning
- annotation scanning
- automatic backend discovery through reflection
- `ServiceLoader` as the default runtime backend mechanism

Backends must be registered explicitly through engine composition.

`ServiceLoader` or plugin discovery may be added later as a convenience layer only if this document is updated first.

It must not become a runtime hot-path mechanism.

### `extensions/training`

`extensions/training` owns training-level concepts and optimizer algorithms.

Allowed:

- `Optimizer`
- `Sgd`
- `Adam`
- `AdamW`
- `Parameter`
- `ParameterGroup`
- `TrainingSession`
- `TrainingStep`

Forbidden:

- dependency on concrete backends
- `MetalOptimizerBridge`
- `CudaOptimizerBridge`
- `CpuOptimizerBridge`
- backend-specific optimizer execution
- backend storage access
- backend kernel selection

Training owns optimizer algorithms, not backend-specific optimizer execution.

Backend-specific optimizer routes, such as fused Adam on Metal, belong to backend prepare/kernels.

### `extensions/onnx`

`extensions/onnx` owns ONNX import/export and mapping.

It must not be part of runtime hot path.

Allowed:

- ONNX import
- ONNX export
- ONNX-to-model mapping
- model-to-ONNX mapping

Forbidden:

- runtime execution
- backend-specific lowering
- kernel selection
- runtime residency

## Documentation

The repository distinguishes between normative architecture and explanatory documentation.

```text
ARCHITECTURE.md
  authoritative architecture contract

docs/
  explanations, guides, design notes, examples, ADRs

AGENTS.md
  agent working instructions
```

Recommended documentation structure:

```text
docs/
  index.md
  getting-started.md

  architecture/
    overview.md
    lifecycle.md
    module-boundaries.md
    dependency-rules.md
    partition-scoring.md
    training-graph.md
    tracing.md
    runtime-prepare-backend-boundary.md

  design/
    decisions/
      0001-layered-architecture.md
      0002-backend-owned-lowering.md
      0003-typed-trace-dtos.md
      0004-partition-scoring.md
      0005-training-combined-forward-backward-graph.md
      0006-no-runtime-service-locator.md
    notes/

  user-guide/
  backend-guide/
  developer-guide/
  api/
```

When an architecture decision changes, update:

1. this document
2. the relevant file under `docs/architecture/`
3. an ADR under `docs/design/decisions/`, if the decision is significant
4. architecture tests, if dependency rules change

## Dependency rules

The intended dependency direction is:

```text
trace
backend-contract
config
model
  -> planning
  -> compiler
```

Runtime/prepare/backend side:

```text
model
config
backend-contract
trace
planning
compiler
  -> runtime
  -> prepare
  -> concrete backends
```

Composition:

```text
compiler
runtime
prepare
backends/cpu
backends/metal
backends/cuda
  -> engine
```

Concrete rules:

- `modules/trace` must not depend on model, planning, compiler, runtime, prepare, engine, or concrete backends.
- `modules/model` must not depend on planning, compiler, runtime, prepare, engine, or concrete backends.
- `modules/config` must not depend on concrete backend implementations.
- `modules/planning` must not depend on concrete backends, runtime, prepare, or engine.
- `modules/compiler` must not depend on runtime, prepare, engine, or concrete backends.
- `modules/runtime` must not depend on concrete backends or engine.
- `modules/prepare` must not depend on concrete backend implementations.
- Concrete backends must not depend on `modules/engine`.
- `backends/openblas-provider` must not depend on compiler, planning, runtime, prepare, engine, or Tensor API.
- `extensions/training` must not depend on concrete backend modules.
- `extensions/onnx` must not depend on runtime hot-path execution internals.

## Compile lifecycle

Compile lifecycle:

```text
Tensor output
  -> GraphCapture
  -> topological sort
  -> producer/use index
  -> canonicalization
  -> shape and data type inference
  -> validation
  -> forward optimization
     - DCE
     - CSE
     - constant folding
     - algebraic simplification
  -> autograd, if CompileMode requires backward
  -> post-autograd optimization
  -> publication binding
  -> backend intent propagation
  -> capability analysis
  -> partition scoring
  -> ownership decision
  -> maximal same-owner partitioning
  -> logical memory/materialization requirements
  -> CompileArtifacts
```

Compile must not create:

- prepared schedules
- prepared units
- prepared executions
- backend executables
- physical buffers
- kernel routes
- runtime workspaces
- backend-specific DAGs

## Prepare lifecycle

Prepare lifecycle:

```text
CompileArtifacts
  -> validate partition coverage
  -> for each PlannedPartition call BackendPartitionPreparer
  -> backend prepare does lowering/specialization/fusion/kernel selection
  -> build PreparedPartition[]
  -> build PreparedMemoryPlan
  -> build PreparedSchedule
  -> validate prepared memory/schedule
  -> PreparedExecution
```

Prepare is where these are created:

- `PreparedPartition`
- `PreparedUnit`
- `PreparedExecutable`
- `PreparedMemoryPlan`
- `PreparedSchedule`
- `PreparedExecution`

Concrete backend lowering occurs in concrete backend modules.

## Run lifecycle

Run lifecycle:

```text
PreparedExecution.run(...)
  -> create/reuse RunState
  -> bind inputs
  -> execute PreparedSchedule
  -> residency/materialization as needed
  -> PreparedExecutable.execute(...)
  -> update residency
  -> publication
  -> RunResult
```

Run must not perform:

- graph optimization
- autograd construction
- backend discovery
- kernel selection
- backend-specific lowering
- compiler passes

## Optimizer/training lifecycle

Initial version:

```text
compile:
  forward graph
  -> autograd
  -> combined forward + backward graph
  -> optimize
  -> CompileArtifacts

run:
  forward/backward prepared execution
  -> publish gradients
  -> optimizer.step()
```

Later version:

```text
compile:
  forward + backward + optimizer update graph
  -> optimize
  -> partition scoring
  -> CompileArtifacts

prepare:
  backend prepare may fuse optimizer update routes

run:
  trainingStep schedule
```

Rules:

- Training owns optimizer algorithms.
- Training does not own backend-specific optimizer execution.
- Concrete backend optimizer routes belong to backend prepare/kernels.
- No training module may depend on backend-metal, backend-cpu, or backend-cuda.

## Explicit non-goals

Do not add these unless this document is updated first:

- compile-time physical schedule
- standalone artifacts/program module
- shared `backend.lowering` module
- runtime service locator
- reflective backend plugin discovery as the core backend mechanism
- separate `cpu-scalar`, `cpu-vector`, or `cpu-blas` backend modules
- `MetalOptimizerBridge` in training
- `Map<String,String>` as the primary trace model
- backend-specific kernel/runtime scoring in planning

## Future extensions allowed only with architecture update

The following may be added later, but only with an explicit update to this document:

- `modules/compiler-api`
- `modules/program`
- `LogicalSchedulePlan`
- source-generated CPU fused kernels
- bytecode-generated CPU fused kernels
- external plugin ecosystem
- `ServiceLoader` as an optional engine-level convenience layer
- more advanced segment-level partition scoring
- profile-guided partition scoring

## Testing requirements

Architecture-sensitive changes must include or update architecture tests under:

```text
testing/architecture-tests/
```

Architecture tests should enforce:

- trace does not depend on other modules
- model does not depend on planning/compiler/runtime/prepare/engine/backend
- config does not depend on concrete backends
- planning does not depend on concrete backend/runtime/prepare/engine
- compiler does not depend on runtime/prepare/engine/concrete backend
- runtime does not depend on concrete backend/engine
- prepare does not depend on concrete backend implementations
- backends do not depend on engine
- openblas-provider does not depend on compiler/planning/runtime/prepare/engine/Tensor API
- `Operation` does not expose `supportedBackends()`
- runtime hot path does not use `Operation` or `CompiledNode`
- planning scoring does not reference concrete kernel classes

Backend behavior changes should include or update backend conformance tests under:

```text
testing/backend-conformance/
```

End-to-end behavior changes should include or update integration tests under:

```text
testing/integration-tests/
```

## Final summary

The architecture is:

```text
model      = clean computational model
planning   = backend-neutral intent, capability, scoring, ownership, logical memory
compiler   = graph transformations, autograd, compile artifacts
prepare    = transition from compile artifacts to executable runtime
backend    = concrete lowering, fusion, kernel selection, storage
runtime    = hot-path execution, residency, publication
engine     = composition root and public lifecycle
trace      = typed diagnostic leaf
```

The most important invariant is:

```text
CompileArtifacts are an immutable recipe.
Planning scoring selects backend ownership, not kernel implementation.
PreparedExecution is prepared runtime state.
RunState is per-run mutable state.
PreparedExecutable computes only its prepared region.
```
