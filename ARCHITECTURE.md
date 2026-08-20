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
extensions/nn
```

Example Java packages:

```text
io.github.pho001.synaptik.model
io.github.pho001.synaptik.compiler
io.github.pho001.synaptik.runtime
io.github.pho001.synaptik.backend.cpu
io.github.pho001.synaptik.backend.metal
io.github.pho001.synaptik.nn
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
    nn/
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
  -> BackendPartitionAnalysis[]
  -> PreparedMemoryPlan with assigned buffer/workspace slots
  -> PreparedPartition[]
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

- `Tensor` is public model state with immutable identity, descriptor, and expression provenance.
- `Tensor` is not an IR node.
- `Tensor` has no gradient field, backward method, or gradient-lifecycle state.
- `Operation` owns semantic behavior but never backend support.
- `Operation` must not expose `supportedBackends()`.
- `CompiledGraphModel` is immutable compile-time graph state.
- `CompileArtifacts` are immutable compile-time output.
- `PreparedExecution`, its prepared memory/schedule/executable recipes, and immutable persistent
  prepared resources are immutable and reusable across runs.
- Each active logical execution of one `PreparedExecution` has exactly one mutable `RunState`
  covering the complete heterogeneous run. Concurrent runs use distinct `RunState` instances and
  isolated mutable state and run-owned resources.
- `PreparedExecutable` computes only its prepared region.
- Runtime hot path must not see `Operation` or `CompiledNode`.
- Compiler must not allocate physical buffers.
- Planning must not select concrete kernels.
- Planning scoring selects backend ownership, not implementation routes.
- Planning may consume backend-neutral cost estimates, but it must not interpret backend route,
  vector, thread, tile, kernel, or other implementation parameters.
- Backend prepare owns backend-specific lowering and kernel selection.
- Backend preparation is staged: backend analysis and exact shared-resource declaration precede
  shared slot assignment, and backend finalization follows slot assignment.
- Model autotuning, when requested, must complete before runtime hot-path execution.
- Runtime profiling is passive observation and must not select or mutate execution settings.
- Runtime executes prepared schedules only.
- Engine is the composition root.
- Concrete backends must not depend on engine.
- Runtime must not depend on concrete backend implementations.

## Fixed recurrent scan without graph regions

The first recurrent operation whose valid sequence lengths can change between runs is one fixed,
first-class Model operation family. It is an ordinary flat, identity-distinct, multi-output
`TensorProducer` occurrence before capture and exactly one ordinary flat `CompiledNode` after
capture. This decision does not authorize a user-defined callback, lambda, Tensor body, nested
`CompiledGraphModel`, callable graph, block, region, loop intermediate representation, captured
subgraph, or other body value.

```text
declarative Model operation
  = immutable fixed transition meaning plus ordered Tensor inputs and outputs

backend-internal runtime control flow
  = one prepared bounded loop implementing that meaning
```

The operation captures no Tensor reference beyond its ordered ordinary inputs and owns no body
input or output, free variable, nested graph-local identity, region identity, cross-graph
ownership rule, or loop condition. `Tensor`, `TensorProducer`, `Operation`, `CompiledNode`, and
`CompiledGraphModel` retain their current flat meanings. A general control-flow or graph-region
system remains forbidden until another explicit architecture update.

<a id="fixed-family-and-planned-model-surface"></a>

### Fixed family and current Model surface

The exact semantic variants are:

```text
RNN_TANH
GRU_RESET_AFTER
LSTM
```

Each occurrence has exactly one immutable `FORWARD` or `REVERSE` direction attribute. The first
family has no bidirectional, stacked, arbitrary-cell, configurable-gate, configurable-activation,
peephole, projection, recurrent-dropout, residual, stateful, sparse, or quantized variant.

The fixed transitions match the current NN cells. `RNN_TANH` uses one tanh transition, separate
input and hidden weights, and one optional shared input-side bias. `GRU_RESET_AFTER` uses reset,
update, and candidate gate order, applies reset after the recurrent candidate projection, computes
`candidate + update * (hidden - candidate)`, and has one optional packed input-side bias. `LSTM`
uses input, forget, candidate, and output gate order, explicit hidden and cell states, the current
fixed sigmoid/tanh equations, and one optional packed input-side bias.

Model publishes exactly one direction enum, a two-output recurrent result, a three-output LSTM
result, and the following six receiver methods in the existing Tensor package. This is current
Java expression-construction API, not a runnable Compiler, Engine, Runtime, or backend API:

```java
RecurrentScanResult rnnScan(
        Tensor validLengths,
        Tensor initialHidden,
        Tensor inputWeight,
        Tensor hiddenWeight,
        RecurrentDirection direction)

RecurrentScanResult rnnScan(
        Tensor validLengths,
        Tensor initialHidden,
        Tensor inputWeight,
        Tensor hiddenWeight,
        Tensor bias,
        RecurrentDirection direction)

RecurrentScanResult gruScan(
        Tensor validLengths,
        Tensor initialHidden,
        Tensor inputWeight,
        Tensor hiddenWeight,
        RecurrentDirection direction)

RecurrentScanResult gruScan(
        Tensor validLengths,
        Tensor initialHidden,
        Tensor inputWeight,
        Tensor hiddenWeight,
        Tensor bias,
        RecurrentDirection direction)

LstmRecurrentScanResult lstmScan(
        Tensor validLengths,
        Tensor initialHidden,
        Tensor initialCell,
        Tensor inputWeight,
        Tensor hiddenWeight,
        RecurrentDirection direction)

LstmRecurrentScanResult lstmScan(
        Tensor validLengths,
        Tensor initialHidden,
        Tensor initialCell,
        Tensor inputWeight,
        Tensor hiddenWeight,
        Tensor bias,
        RecurrentDirection direction)
```

The receiver is the time-major input Tensor. `RecurrentScanResult` exposes exactly `outputs` and
`finalHidden`; `LstmRecurrentScanResult` exposes exactly `outputs`, `finalHidden`, and
`finalCell`. These are canonical wrappers from one exact shared producer in that order.

Operation input order is fixed:

```text
RNN/GRU without bias:
  [input, validLengths, initialHidden, inputWeight, hiddenWeight]
RNN/GRU with bias:
  [input, validLengths, initialHidden, inputWeight, hiddenWeight, bias]
LSTM without bias:
  [input, validLengths, initialHidden, initialCell, inputWeight, hiddenWeight]
LSTM with bias:
  [input, validLengths, initialHidden, initialCell, inputWeight, hiddenWeight, bias]
```

Operation output order is fixed:

```text
RNN/GRU: [outputs, finalHidden]
LSTM:    [outputs, finalHidden, finalCell]
```

### Static Shape and runtime-value boundary

The first executable capability has fully static Shapes while valid-length values remain ordinary
runtime inputs:

- input is rank three `[time, batch, inputSize]` in time-major order;
- valid lengths are a rank-one `INT64` Tensor `[batch]` with `requiresGrad == false`;
- initial hidden and the LSTM initial cell are `[batch, hiddenSize]`;
- input weight is `[gateCount * hiddenSize, inputSize]`;
- hidden weight is `[gateCount * hiddenSize, hiddenSize]`;
- optional bias is `[gateCount * hiddenSize]`;
- `gateCount` is one for RNN, three for GRU, and four for LSTM;
- `inputSize` and `hiddenSize` are positive, while `time` and `batch` may be zero; and
- input, states, weights, and optional bias have one exact common floating data type.

The dense output Shape is `[time, batch, hiddenSize]`, and each final-state Shape is
`[batch, hiddenSize]`. Layout remains unresolved at Model construction. Output gradient
eligibility is the OR of the differentiable floating input, state, and parameter roles; valid
lengths never contribute.

Dynamic or binding-dependent `time`, `batch`, `inputSize`, `hiddenSize`, parameter Shapes, or
output Shapes are outside this first program. Prepared execution must be reusable across
different valid-length values for the same compatible static descriptors, but not across
arbitrary Shapes. Runtime valid lengths must not be used to disguise a dynamic-Shape lifecycle.

### Valid lengths, traversal, and outputs

For each original batch row `b`, execution validates the runtime value `L[b]` in the inclusive
range `[0, time]`. Lengths are never inferred from input contents, padding values, zero, NaN,
token identifiers, labels, or storage.

`FORWARD` applies the fixed transition at original coordinates `0 .. L[b]-1`. `REVERSE` applies
the same transition at original coordinates `L[b]-1 .. 0`; it reverses only the valid prefix and
never traverses the padded suffix. In either direction:

- each valid coordinate stores the next hidden state produced after consuming that coordinate;
- each padded coordinate `t >= L[b]` stores the exact positive zero of the common data type;
- final hidden, and final cell for LSTM, is the state after the row's last executed transition;
- a zero-length row has positive-zero output at every time and returns its exact initial state
  values semantically; and
- when `time` is zero, every length must be zero, the dense output is empty, and the initial states
  are the semantic final states.

Padding is never a recurrent input. Before mutating any output representation, an executable
backend must validate the complete length vector, every bound, and any representation-specific
access precondition. Invalid lengths fail the run without partially written published results.
The future Engine facade owns public exception translation; this architecture does not select its
exception type.

### Purity and lifecycle ownership

The operation is functionally pure. All carried state is explicit in its ordinary inputs and
final outputs. It owns no hidden module, compiler, prepared, or runtime state; RNG; mode; counter;
`Buffer`; `Parameter`; mutation; callback; I/O; or external resource. NN continues to own
parameter wrappers and state paths, while the operation consumes their current Tensor bindings as
ordinary inputs. Different executions use isolated `RunState` instances under the existing
Runtime contract.

NN state dictionaries contain only parameter and persistent-buffer Tensor bindings. Future model
checkpoints may persist the materialized values of those entries. The scan operation, compiler
graph, prepared executable, runtime state, and backend artifacts are rebuilt and are not
serialized by this decision.

Owner boundaries are exact:

- Model owns operation kind and attributes, fixed semantics, descriptor-visible validation,
  result metadata, canonical multi-output provenance, and the current Tensor surface.
- Compiler captures one ordinary flat node, owns inference and final validation, adopts the
  operation in its exact closed inventory, and initially rejects every backward-capable request
  that reaches the family before constructing any gradient Tensor.
- Backpropagation through time (BPTT) is a later explicit Compiler decision. No saved-gate output,
  tape, checkpointing or recomputation policy, backward operation, or derivative formula is
  selected here.
- Planning performs its existing ordinary operation capability query and selects backend
  ownership. It does not interpret recurrence, direction, valid lengths, active rows, loop
  parameters, or routes.
- Shared Prepare uses its existing static-Shape projection, staged backend analysis, exact shared-
  resource declaration, slot assignment, and finalization lifecycle. It gains no loop-body or
  control-flow contract.
- Runtime uses its existing caller-input representation, immutable prepared recipe, cold binding,
  schedule, isolated run state, and bound invocation contracts. It does not inspect valid
  lengths, select a loop count, compact rows, or interpret recurrence.
- Engine owns typed logical caller-input to ordered Runtime-representation binding and typed
  publication mapping. A valid-length value never causes Engine to specialize or rebuild a graph.
- A concrete backend advertises only exact implemented variant, type, Shape, and direction
  combinations. It lowers the occurrence once, declares all shared resources before slot
  assignment, and constructs the reusable executable during finalization.
- NN retains module composition and current parameter/state bindings; a later NN/Data task owns
  runtime-valid-length convenience and schema integration. Data does not own recurrence and must
  not infer lengths from values.
- Training retains optimizer and training-orchestration ownership. It does not implement BPTT,
  recurrent execution, saved-state policy, or a backend route.

The current Planning, Prepare, and Runtime shared contracts are sufficient for this static-Shape
ordinary-node design. A later implementation must stop and provide evidence if a concrete backend
reveals an actual shared-contract gap; it must not add speculative region or loop-body types.

### Performance and migration invariants

One Model scan occurrence remains one compiler graph node regardless of `time` or valid-length
values, so its graph construction and graph size are `O(1)` in `time`. Backend analysis prepares
the transition implementation once for that occurrence and must not construct, compile, or retain
one graph, node, or executable body per time step.

The runtime hot path performs no reflection, string dispatch, scalar-element boxing, graph
inspection, operation dispatch, backend lookup, route selection, or per-step object-graph growth.
Invalid coordinates execute no recurrent dot products, gates, or state update. A bounded branch-
based row/time traversal is sufficient for the first capability. Physical active-row compaction,
sorting, packed buffers, vectorized packed batches, workspace reuse, and claims that length checks
or zero initialization perform no work remain deferred backend route decisions requiring truthful
resource and performance evidence.

Current `RnnSequence`, `GruSequence`, `LstmSequence`, and the three bidirectional containers keep
their snapshotted Java `long[]`, compact-output-list, static-unroll, exact-provenance, and final-
state contracts unchanged. Model and Compiler implementation must not redirect those APIs to the
new operation. A later NN decision must address whether to add runtime-length overloads, migrate,
retain compatibility, or deprecate. It must acknowledge that the new scan returns dense zero-
padded original-time-aligned outputs, while current static containers return compact per-step
output lists. A host `long[]` adapter that builds a different graph for each length pattern is not
an implementation of this runtime-valid-length decision.

Any later bidirectional migration must preserve independent parameter ownership, valid-prefix-
only reverse traversal, forward-first final-axis concatenation, original-time alignment, and
type-specific final hidden and cell states.

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
- `ForwardPublicationBinding`
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

`Tensor` may retain its existing mutable borrowed host-storage association, but it must not own
gradient state or runtime device residency. Its identity, descriptor, and expression provenance
remain immutable.

Every derived expression producer owns one operation occurrence, its ordered input tensors, its
ordered output descriptors, and the canonical exact `Tensor` wrapper for every output slot.
`TensorFactory` constructs those wrappers atomically with their indexed provenance. A producer
must return the retained wrapper for a slot rather than reconstructing an equal wrapper. This
pre-capture model relationship is not graph IR, graph membership, or graph-local identity.

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
- immutable declarative planning-cost inputs, after their planning consumer is stable
- immutable declarative model-autotuning inputs, after their owning contracts are stable

Forbidden:

- live services
- concrete backend classes
- kernel class references
- runtime state
- executable units
- backend-specific implementation logic
- benchmark runners, model-autotuning search, cache mutation, or measurement algorithms
- live platform discovery or mutable measurement evidence

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
- backend-neutral planning cost estimates and profiles

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
- backend route names or route-selection parameters
- vector species, lane counts, unroll factors, thread counts, chunk sizes, or tile sizes

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
- backend-neutral planning cost profiles

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
- backend-local workload-cache entries, route configurations, or other model-autotuning values

Partition scoring decides backend ownership at node or segment level before maximal same-owner partitioning.

The output of scoring is ownership, not executable implementation.

Backend-specific lowering and kernel selection belong to backend prepare.

A planning cost model is not a model-autotuning parameter set. It may estimate ownership
cost from backend-neutral graph and transfer facts, but it must not interpret vector, thread,
tile, route, kernel, or other backend implementation vocabulary.

### `modules/compiler`

`modules/compiler` owns graph compilation.

Allowed:

- graph capture
- fail-closed autograd preflight over Tensor expression occurrences
- compiler-owned gradient-rule dispatch expressed through public Tensor operations
- per-compile identity-based gradient-contribution accumulation
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
- combined forward/backward graph optimization
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

In backward-capable compile modes, the compiler constructs backward Tensor expressions before
graph capture. It then captures the forward outputs and requested gradient roots together exactly
once into a combined forward + backward graph.

The combined graph is immutable compile-time graph state. Public Tensors remain expression model
state and never become graph nodes or values.

Runtime may expose separate forward and backward schedules, or a single training-step schedule, depending on prepare-time decisions.

Backends must not implement global autograd.

Backends execute prepared regions only.

Optimizer updates are either:

- backend-agnostic runtime/training steps, or
- graph operations generated by the training extension and lowered by backend prepare

Training must not depend on concrete backend modules.

## Compiler-owned automatic differentiation

The compiler owns reverse-mode automatic differentiation. The required compile-time flow is:

```text
original forward Tensor expression DAG
  -> fail-closed backward-reachable operation/attribute/policy preflight
  -> compiler-owned reverse traversal and Tensor gradient-expression construction
  -> combined forward + gradient Tensor expression DAG
  -> one phase-aware graph capture
  -> immutable combined compiler graph
  -> inference and validation
  -> canonicalization and exact combined-graph optimization
  -> final validation
  -> publication and planning
```

`FORWARD_ONLY` skips autograd. `FORWARD_AND_BACKWARD` and `TRAINING_STEP` construct the combined
Tensor expression DAG before capture. `TRAINING_STEP` does not add optimizer updates until a later
architecture decision and implementation task explicitly introduces them.

Gradient-rule dispatch belongs to named compiler components such as
`ElementwiseGradientRules`. A rule constructs formulas only by calling existing public Tensor
operations such as `mul`, `add`, `sumToShape`, and `transpose`. The compiler must not add
model-owned derivative rules, a second low-level algebra language, direct generated graph-node
formula construction, a public gradient registry or facade, or Tensor gradient/backward state.

For one compile request, the compiler may use identity-based maps from exact Tensor objects to
ordered gradient contributions and accumulated gradients. This is ephemeral reverse-accumulation
bookkeeping, not public Tensor state and not another graph representation. Multiple contributions
are accumulated with ordinary `Tensor.add`.

Before constructing any backward expression, the compiler must inventory every
backward-reachable operation occurrence and its exact attributes and derivative policies. Any
unsupported or ambiguous occurrence fails closed. Full inference and validation still occur only
after the one combined capture, so a later construction, capture, inference, validation, or
optimization failure may consume temporary model-level `TensorId` values. Tensor IDs are opaque
and are never rolled back or reused.

Seeds and derivative constants are storage-free Tensor leaves or expressions explicitly
registered as compile-time constant splats. Tensor storage, labels, descriptor shape, factory
history, or provenance absence must never silently imply a compile-time constant.

Phase-aware capture receives the ordered forward outputs, requested gradient roots and their
target roles, the identity set of original forward producers, and explicit constant facts. It
assigns every `NodeId` and `ValueId` exactly once. Nodes whose producer identity belongs to the
original forward set have phase `FORWARD`; generated derivative producers have phase `BACKWARD`.
Per-node `GraphPhase` remains authoritative and must not be replaced by only a positional
backward-start index.

Distinct differentiation targets may legitimately resolve to the same captured gradient
`ValueId`. Gradient result roles map each target independently, while the graph's public output
boundary contains each distinct gradient value once. The compiler must not create identity nodes
solely to make those result values distinct.

Optimization operates on the immutable combined graph. The exact arithmetic rules, constant
folding, dead-code elimination, and common-subexpression elimination already established by
compiler tasks 0003, 0003A, and 0003B must be reassessed for both phases and applied only where
their existing semantic guards remain valid. Common-subexpression elimination is phase-local
unless a later architecture update and proof establishes a broader safe rule. Every changed
candidate is revalidated through the compiler's inference-and-validation boundary. This contract
does not authorize new algebraic rewrites.

Generated gradient formulas are ordinary differentiable Tensor expressions. Higher derivatives
are not implemented by the initial autograd task, but the design must preserve that path. A later
task must define an explicit create-graph or derivative-order lifecycle contract, rules for every
operation used by gradient formulas, and graph representation for derivative order in addition to
phase. It must not retrofit mutable gradient lifecycle state onto Tensor.

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
- passive runtime profiling and observation through typed trace contracts

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
- model-autotuning search, tuning-cache lookup or mutation, graph inspection for tuning, or
  selection of execution settings in the hot path

Runtime executes prepared schedules.

Runtime does not decide how graph partitions should be lowered.

Runtime does not select kernels.

Runtime does not discover backend plugins.

Runtime owns each run's logical slot state, resource-lifecycle orchestration, representation
validity and residency needed by the prepared schedule, failure cleanup, and concurrent-run
isolation. Concrete backends own physical buffer and workspace representation implementations and
the allocation, release, transfer, and access mechanics for those representations. Runtime must
not know concrete host, native, Metal, or CUDA storage classes and must not choose a backend.

A buffer slot may have one or more backend/device representations only when the prepared schedule
requires them. Representation creation and transfer are explicit prepared work, not on-demand
kernel or backend discovery. A workspace slot is per-run backend-local implementation scratch and
normally binds one physical representation for its declared use; host staging and device scratch
are separate workspace requirements when both are needed.

Caller inputs are borrowed for one run. Internal buffers and workspaces are run-owned. Published
outputs transfer or lease ownership to `RunResult`, while immutable persistent prepared resources
remain owned by `PreparedExecution` and are not ordinary workspace. Runtime orchestrates cleanup,
but concrete representations perform physical release. Failure cleanup releases only resources
still owned by the run, never borrowed inputs or already transferred outputs.

Before hot-path execution, a cold binding phase validates representation compatibility and creates
backend-owned typed bound invocation objects with direct references. Any heterogeneous Java type
check is explicit, checked, and confined to that boundary. The hot path performs no map lookup,
reflection, string dispatch, graph inspection, backend discovery, kernel selection, or repeated
unsafe cast. Runtime resource contracts must not use raw `Object`, unchecked generic access, a
global registry, a service locator, or a public switch over concrete backend types.

### `modules/prepare`

`modules/prepare` owns shared prepare contracts and validation.

Allowed:

- `PrepareContext`
- `BackendPartitionPreparer`
- `BackendPartitionAnalysis`
- backend-neutral buffer and workspace requirement declarations
- `PreparedPartition`
- shared assignment of stable runtime buffer and workspace slot identities
- partition coverage validation
- prepared memory validation
- prepared schedule validation
- a future narrow orchestration boundary that exposes complete valid preparation candidates
  opaquely to model-autotuning tooling

Forbidden:

- concrete CPU lowering
- concrete Metal lowering
- concrete CUDA lowering
- concrete kernel selection
- backend-specific executable implementation
- backend-specific storage implementation
- interpretation of private backend candidate parameters

Prepare is the bridge between compile artifacts and runtime. It projects the exact stable
semantic and planning facts, resolved prepare-time bindings, target capabilities, configuration,
and compatible cached tuning decisions that backend analysis may consume. That projection must
not expose `CompileArtifacts` or another compiler-owned aggregate to a concrete backend.

For each planned partition, the concrete backend first analyzes and lowers the projected facts,
selects a supported route and configuration, and returns a `BackendPartitionAnalysis`. The
analysis retains the backend's selected lowering and route state opaquely while declaring every
shared buffer and workspace requirement exactly enough for shared preparation to assign stable
runtime slot identities. Shared preparation does not interpret the opaque backend plan or private
route vocabulary.

After shared preparation assigns slots, the same backend finalizes the analysis against those
assignments and constructs the `PreparedExecutable` and `PreparedPartition`. Backend
finalization must not change the selected route or introduce undeclared shared requirements.
Physical allocation and per-run binding remain runtime/backend concerns after preparation.

Backend analysis is deterministic from its explicit facts, configuration, and compatible cache
inputs. An explicitly enabled later model-autotuning workflow may instead supply a selected
compatible decision before analysis; this lifecycle does not authorize prepare-time measurement
or search. Any unresolved fact needed to choose a route or declare an exact resource requirement
must fail preparation unless an explicit prepared contract represents that fact as run-dynamic
without changing route or slot assignment.

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
- typed, version-controlled, tested candidate generators colocated with the routes they configure
- compatible workload-cache lookup during preparation
- safe heuristic selection when model autotuning is disabled or compatible cache entries are
  absent
- deterministic partition analysis and exact shared buffer/workspace requirement declaration
- finalization of an analyzed partition against shared assigned slots

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

Each concrete backend owns the complete typed configuration vocabulary and candidate generator
for each route it implements. A generator derives and prunes complete valid configurations from
target capabilities, canonical workload facts, and the tuning budget. CPU matrix-multiplication
candidates may, for example, include supported Vector API species and strategy, unroll, tile,
parallelism, or OpenBLAS thread configurations. Vector, scalar, and OpenBLAS choices are distinct
route-specific typed configurations, not booleans or entries in a generic parameter bag.

Operation family selects the appropriate backend candidate generator; it is not a cache key for
one universal family-wide configuration. Local tuning measurements use a canonical workload
signature that includes semantics and attributes, data types, shapes, layouts, relevant policies,
and target compatibility. Identical signatures can reuse one result across occurrences and
models. Physical vector lanes remain constrained by hardware and supported JDK Vector API
species; candidate generation must not promise arbitrary lane counts.

Backend candidate discovery must not use `Map<String,Object>`, string dispatch, reflective
annotations, a central knob registry, or a generic configuration language. Shared preparation and
tuning orchestration sees candidates opaquely and does not interpret private backend fields.

## Performance evidence and optimization tooling

`tools/benchmarks` owns observational, report-oriented benchmarking. It runs fixed reproducible
operation, operation-family, model, and end-to-end workloads to compare commits, models, or
environments. A benchmark produces measurement evidence and reports only. It must not select or
mutate production settings.

`tools/tuning` owns one explicit model-autotuning workflow with two coordinated phases. First, it
extracts actual tunable workloads and routes from the model, forms canonical workload signatures,
deduplicates identical signatures while retaining occurrence weight and context, reuses compatible
entries from an explicit persistent workload cache, and measures only cache misses. Second, it
measures a bounded set of complete valid graph, fusion, ownership/partition, layout,
materialization, route, and configuration candidates end to end and selects an explicit prepared
plan or artifact. This second phase does not repeat local route-parameter search.

Compiler, planning, prepare, and concrete backends generate the candidates for decisions they
own. Tuning tooling coordinates measurement and selection without taking over graph semantics,
transformations, ownership rules, lowering, route logic, or private backend vocabulary. The model
author supplies the model, representative input or shape profiles, objective, budget, constraints,
and explicit cache locations; backend authors define backend candidates. Running this same
workflow over a representative model corpus may pre-seed the same workload cache for a target,
but there is no separate platform-calibration subsystem, workflow, or profile.

`modules/config` may store immutable declarative inputs to this workflow after consumers are
stable, but it does not own runners, search algorithms, live discovery, caches, or mutable
evidence. Model autotuning is optional for correctness. Cache-only or heuristic preparation must
remain safe when it is not requested.

Future tuning artifacts are explicit persistent files: a reusable workload tuning cache and a
model-specific plan cache or prepared-plan record. A load reuses only compatible hits; a miss may
be tuned and atomically persisted. Entries carry explicit schema and backend candidate-schema
versions, target and workload or model fingerprints, objective and constraints, and a measurement
summary. Implementations invalidate incompatible entries and safely reject corrupt data. They do
not use hidden global caches, Java object serialization, or executable payload assumptions. Rich
measurement evidence remains separate from compact caches. Physical file formats and prepared
executable serialization remain deferred to their backend and lifecycle owners.

Runtime profiling is passive observation of actual execution. `modules/runtime` owns the observed
execution context and `modules/trace` owns typed diagnostic DTOs; neither profiling nor tracing
selects settings.

## CPU backend routes

CPU scalar, CPU Vector API, generated JVM-bytecode CPU computation kernels, and OpenBLAS are
routes inside the CPU backend.

They are not separate backends.

A generated CPU computation kernel is backend-internal executable code whose JVM bytecode is
produced for a selected CPU lowering and specialization. CPU backend analysis owns the lowering,
specialization, fusion, route choice, and exact shared-resource declarations. CPU backend
finalization may generate and define the selected kernel, or reuse it from a CPU-owned compatible
generated-artifact cache, only after shared Prepare assigns slots. Runtime receives the resulting
prepared executable and neither generates, caches, selects, nor specializes kernels.

This contract does not prescribe a bytecode-generation library or a particular JDK builder API.
Changing that implementation mechanism within the CPU backend does not change module ownership,
dependency direction, or lifecycle placement.

Planning chooses:

```text
owner = CPU
```

CPU prepare chooses:

```text
scalar route
Vector API route
generated JVM-bytecode CPU computation-kernel route
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

### `extensions/nn`

`extensions/nn` owns the stateful neural-network composition layer. It defines model modules,
their trainable parameters and persistent buffers, training/evaluation mode, and the forward
context needed to apply that mode consistently through a module tree.

Allowed:

- `Module`
- `Parameter`
- `Buffer`
- module-owned parameter and buffer traversal
- training/evaluation mode propagation
- forward-context contracts
- neural-network layers, blocks, and functional conveniences composed from model semantics

Forbidden:

- optimizer algorithms
- optimizer update orchestration
- autograd construction
- backend storage access
- backend kernel selection
- concrete backend dependencies

`extensions/nn` depends on `modules/model` for tensor and operation semantics. It must not make
the model depend on neural-network layers or stateful module ownership.

### `extensions/training`

`extensions/training` owns optimizer algorithms and training orchestration over parameters
declared by `extensions/nn` modules.

Allowed:

- `Optimizer`
- `Sgd`
- `Adam`
- `AdamW`
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

Training depends on `extensions/nn`, not the reverse. `train()` and `eval()` mode are module
forward-behavior concerns owned by `extensions/nn`; an optimizer neither selects nor changes
that mode.

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
    performance-evidence-and-tuning.md
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
      0007-neural-network-module-and-training-boundary.md
      0008-performance-evidence-and-tuning-boundaries.md
      0009-compiler-owned-pre-capture-tensor-expression-autograd.md
      0010-staged-backend-preparation.md
      0011-per-run-runtime-resource-ownership.md
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

Neural-network composition and training use this extension direction:

```text
modules/model
  -> extensions/nn
  -> extensions/training
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
- `extensions/nn` may depend on `modules/model` but must not depend on `extensions/training`, compiler, runtime, prepare, engine, or concrete backends.
- `extensions/training` may depend on `extensions/nn` and backend-neutral contracts it requires, but must not make `extensions/nn` depend on training.
- `extensions/training` must not depend on concrete backend modules.
- `extensions/onnx` must not depend on runtime hot-path execution internals.

## Compile lifecycle

Compile lifecycle:

```text
Tensor forward outputs
  -> if backward is requested:
     - fail-closed autograd preflight
     - reverse accumulation through public Tensor operations
     - combined forward + gradient Tensor expression DAG
  -> one phase-aware GraphCapture
  -> topological sort
  -> producer/use index
  -> shape and data type inference
  -> validation
  -> canonicalization
  -> combined-graph optimization
     - DCE
     - phase-local CSE
     - constant folding
     - algebraic simplification
  -> final validation
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
  -> project partition-scoped semantic/planning facts and resolved prepare inputs
  -> for each PlannedPartition call BackendPartitionPreparer analysis
  -> backend analysis does lowering/specialization/fusion/kernel selection
     and declares exact shared buffer/workspace requirements
  -> build BackendPartitionAnalysis[]
  -> assign stable buffer/workspace slots and build PreparedMemoryPlan
  -> finalize each backend analysis against its assigned slots
  -> build PreparedPartition[] and PreparedExecutable[]
  -> build PreparedSchedule
  -> validate prepared memory/schedule
  -> PreparedExecution
```

Prepare is where these are created:

- `BackendPartitionAnalysis`
- `PreparedPartition`
- `PreparedUnit`
- `PreparedExecutable`
- `PreparedMemoryPlan`
- `PreparedSchedule`
- `PreparedExecution`

Concrete backend lowering occurs in concrete backend modules.
Route selection and shared-resource discovery occur during backend analysis. Executable
construction occurs only during backend finalization after shared slot assignment.

## Run lifecycle

Run lifecycle:

```text
PreparedExecution.run(...)
  -> create exactly one RunState for the complete logical run
  -> bind caller inputs as borrowed resources
  -> create run-owned internal buffer/workspace representations through prepared backend work
  -> perform cold checked binding to backend-owned typed invocation objects
  -> execute PreparedSchedule
  -> perform explicit prepared residency/materialization/transfer work as needed
  -> PreparedExecutable.execute(...)
  -> update residency
  -> publication and output ownership transfer/lease
  -> RunResult
  -> release resources still owned by RunState
```

Run must not perform:

- graph optimization
- autograd construction
- backend discovery
- kernel selection
- backend-specific lowering
- compiler passes

The initial runtime resource model introduces no automatic pooling, reuse, aliasing, distributed
sharding, hidden mutation/coherence protocol, or multi-device scheduling. Multiple physical
representations exist only when explicitly required by a prepared schedule, and immutable
functional value semantics do not imply hidden write-back between them.

## Optimizer/training lifecycle

Initial version:

```text
compile:
  forward Tensor expression DAG
  -> compiler-owned autograd as Tensor expressions
  -> one capture of the combined forward + backward DAG
  -> infer, validate, optimize, and revalidate the immutable combined graph
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

- `extensions/nn` owns module-declared trainable parameters, persistent buffers, and train/eval forward behavior.
- `FORWARD_ONLY` performs no autograd.
- `FORWARD_AND_BACKWARD` and the initial `TRAINING_STEP` use the same combined pre-capture
  forward/backward construction; the initial `TRAINING_STEP` adds no optimizer-update graph work.
- Training owns optimizer algorithms.
- Training consumes the parameters declared by `extensions/nn`; it does not own layer behavior or train/eval mode.
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
- benchmark-driven production-setting mutation
- model autotuning or tuning-cache mutation in the runtime hot path
- planning interpretation of backend routes, vector parameters, threads, chunks, or tiles
- `Tensor.gradient`, `Tensor.backward`, mutable Tensor gradient fields, or hidden thread-local
  gradient/compilation scope
- model-owned derivative rules
- placeholder Tensors that stand in for already-captured `ValueId` values
- a `ValueId`-to-Tensor conversion map or a second low-level gradient algebra language
- a public compiler gradient-rule registry or facade

## Future extensions allowed only with architecture update

The following may be added later, but only with an explicit update to this document:

- `modules/compiler-api`
- `modules/program`
- `LogicalSchedulePlan`
- source-generated CPU fused kernels
- external plugin ecosystem
- `ServiceLoader` as an optional engine-level convenience layer
- more advanced segment-level partition scoring

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
- `extensions/nn` does not depend on training or execution/backend layers
- `extensions/training` depends on `extensions/nn` when both modules exist, never in the reverse direction

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
nn         = stateful neural-network composition, parameters, buffers, and train/eval behavior
training   = optimizer algorithms and training orchestration over nn-declared parameters
```

The most important invariant is:

```text
CompileArtifacts are an immutable recipe.
Planning scoring selects backend ownership, not kernel implementation.
Backend analysis selects a route and declares exact shared requirements before slot assignment.
Backend finalization constructs executable state only after slot assignment.
PreparedExecution recipes are immutable and reusable across runs.
Exactly one RunState owns the mutable state of each active complete logical run.
PreparedExecutable computes only its prepared region.
```
