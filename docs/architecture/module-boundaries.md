# Module Boundaries

This document explains the module responsibilities established by [`ARCHITECTURE.md`](../../ARCHITECTURE.md). The contract is authoritative when a summary here is incomplete.

The boundaries apply to both implemented and planned modules. Model, Backend Contract, Planning, Compiler, Runtime, and the first Prepare contracts have substantive implementations; Config and Trace are partial. Engine, concrete backends, and most extensions and tools remain planned or placeholder-only. The [roadmap](../planning/roadmap.md) records exact delivery status.

## Shared modules

### `modules/trace`

Owns typed diagnostic DTOs: the event envelope, compile/prepare/run/backend payloads, trace-local identifiers, and typed trace attributes. It contains no graph traversal, execution, business logic, or mutable runtime state and must not import higher layers. This keeps trace data usable from every producer without creating reverse dependencies.

See [Tracing](tracing.md).

### `modules/backend-contract`

Owns the minimal backend vocabulary shared across layers: `BackendId`, `BackendDeviceId`, availability snapshots, declarative requirements, and device classes. Compile-time plans refer to backend identities rather than live backend services.

It does not own capabilities implementation, operation support logic, kernel registries, prepare services, executables, storage, physical buffers, or cost-model implementation.

### `modules/model`

Owns the public tensor model, operation semantics, shape/data type/layout types, host storage
abstractions, and immutable graph model. A `Tensor` has immutable identity, descriptor, and
expression provenance, plus its existing mutable borrowed host-storage association; it has no
gradient/backward lifecycle state. Each derived `TensorProducer` retains the canonical exact
wrapper for every output position so compiler-owned pre-capture formulas can use hidden auxiliary
outputs without reconstructing wrappers. `CompiledGraphModel`, `CompiledNode`, and graph values
remain distinct compile-time state.

The model does not know backend support, device residency, kernel selection, backend-specific storage, prepared execution, or runtime state. `Operation` expresses semantics and never exposes `supportedBackends()`. Runtime device storage belongs outside this module.

The current Model fixed recurrent scan follows this same flat boundary. Model owns the fixed
`RNN_TANH`, `GRU_RESET_AFTER`, and `LSTM` meanings, one `FORWARD` or `REVERSE` attribute, ordered
ordinary Tensor inputs, fully static descriptor rules, canonical dense output and final-state
wrappers, and exactly six public static biased or bias-free constructors on the final field-free
`model.tensor.RecurrentScan` namespace. The time-major input is an explicit first argument;
`Tensor` has no recurrent receiver alias. This advanced low-level namespace is not an NN layer or
module, execution service, registry, or general scan-body abstraction. It owns no callback body,
nested graph, region, captured free variable, runtime length inspection, hidden state, or
execution loop. Compiler adoption and every executable route remain future work. Current NN
`RnnSequence`, `GruSequence`, and `LstmSequence` containers retain construction-time Java
`long[]` lengths and static unrolling; later NN delegation requires Compiler and backend adoption.

### `modules/config`

Owns immutable declarative compile, prepare, run, publication, planning-cost, and model-autotuning
inputs after their consumers are stable. It may describe backend intent and scoring
policy, but it contains no benchmark runner, search algorithm, live discovery, mutable evidence,
live service, concrete backend class, executable unit, runtime state, or kernel class reference.
A concrete backend interprets its backend-specific prepare inputs inside that backend.

### `modules/planning`

Owns backend-neutral compile-time planning: intent propagation, capability query contracts and matrices, ownership scoring, node or segment ownership, maximal same-owner partitions, and logical memory/materialization requirements.

Planning answers where work should run. It does not implement fusion or specialization, select a concrete kernel or backend route, allocate physical memory, construct backend DAGs, or create prepared schedules and runtime units.

Planning may interpret a backend-neutral cost model to choose `BackendId` ownership. It never
interprets route names, vector species or lanes, unroll factors, thread counts, chunks, tiles, or
other backend parameter vocabulary.

When the fixed recurrent scan becomes executable, it enters Planning through the unchanged
ordinary operation capability query. Planning selects an owner but does not interpret cell
variant, direction, valid-length values, active rows, loop bounds, or a recurrent execution
route. No current capability provider advertises the Model family.

See [Partition Scoring](partition-scoring.md).

### `modules/compiler`

Owns graph capture and compilation: fail-closed pre-capture autograd inventory, named
gradient-rule dispatch, per-compile identity-based contribution accumulation, one phase-aware
combined capture, indexing and ordering, inference and validation, canonicalization and combined
optimization, publication binding, planning orchestration, compile diagnostics, and
`CompileArtifacts`.

Its output is immutable compile-time state. It does not create physical buffers, backend executables, runtime workspaces, prepared schedules, or `PreparedExecution`, and it has no concrete backend dependencies.

Compiler can structurally capture a current Model scan producer as exactly one ordinary flat
node, but current inference rejects `RecurrentScanKind` as unsupported before planning. Its
closed autograd inventory rejects every backward-capable request reaching the family before any
gradient Tensor is constructed. Compiler task 0006A owns forward inference, validation, and
closed-inventory adoption; a later explicit backpropagation-through-time (BPTT) decision owns
derivatives. Compiler must not unroll the recurrence into `time` nodes or create a body graph.

### `modules/runtime`

Owns prepared execution contracts and dynamic execution state, including `PreparedExecution`, `PreparedUnit`, `PreparedExecutable`, `PreparedSchedule`, `PreparedMemoryPlan`, slots, `RunState`, resources, transfers, residency, publication, and the prepared execution runner.

Prepared recipes are immutable and reusable. Each active complete logical run has one isolated
mutable `RunState`, even when its schedule spans multiple backends. Runtime owns logical slot
state, ownership transitions, validity/residency, cleanup orchestration, and run isolation;
concrete backends own physical buffer/workspace representation implementations and the actual
allocation, release, transfer, and access mechanics. Checked heterogeneous binding occurs once
before the hot path and produces backend-owned typed direct-reference invocation objects.

Runtime executes already-prepared schedules. It does not optimize graphs, construct autograd,
discover backends, look up backend services, lower partitions, select kernels, autotune, inspect
the graph for tuning, or access or mutate tuning caches. Runtime profiling is passive observation
translated into typed trace data; it cannot change settings. The hot path does not use
`Operation` or `CompiledNode`.

For a future executable fixed recurrent scan, Runtime receives only ordinary caller-input
representations, one reusable prepared executable recipe, cold-bound direct references, and an
isolated `RunState`. It does not inspect the valid-length value, interpret direction or
recurrence, choose loop bounds, or compact active rows.

### `modules/prepare`

Owns shared preparation orchestration and validation: `PrepareContext`,
`BackendPartitionPreparer`, `BackendPartitionAnalysis`, exact backend-neutral buffer/workspace
declarations, stable slot assignment, `PreparedPartition`, partition coverage validation,
prepared memory validation, and prepared schedule validation.

Prepare bridges compile artifacts to runtime contracts, but shared prepare code does not contain concrete CPU, Metal, or CUDA lowering, kernel selection, executable implementations, or backend-specific storage. Those belong to concrete backend modules.

The future executable fixed recurrent scan needs no shared loop-body contract. Prepare projects one
ordinary static-Shape partition occurrence, accepts the backend's opaque one-time recurrent-loop
analysis and exact resource declarations, assigns slots, and supplies those assignments for
finalization through the existing staged lifecycle.

### `modules/engine`

Owns the public lifecycle facade and composition root. It wires the compiler, prepare validators and orchestration, runtime, trace sinks, and explicitly registered concrete backends.

Engine does not own kernels, backend internals, graph optimization passes, a runtime service locator, or reflective plugin discovery as the core backend mechanism. Concrete backends never depend on engine.

For a future runnable recurrent scan, Engine owns the checked typed mapping from the logical input
Tensors, including `INT64[batch]` valid lengths, to ordered Runtime caller-input representations
and from publication positions to typed outputs. A length value never causes graph capture,
specialization, or preparation to repeat.

## Concrete backend modules

`backends/cpu`, `backends/metal`, and `backends/cuda` are vertical implementations. Each backend
owns its capability provider, deterministic partition analysis, lowering, fusion, specialization,
route selection, opaque analysis plan, executable finalization against assigned slots, executable
units, physical buffer and workspace representations, their allocation/release/transfer/access
mechanics, typed cold-bound invocation objects, trace contributions, and native integration.

Concrete backend logic belongs in the backend that implements it:

- CPU scalar, Vector API, generated JVM-bytecode CPU computation kernels, and OpenBLAS are routes
  inside `backends/cpu`, not separate backends. CPU owns generation and compatible generated-
  artifact caching during preparation; shared Prepare and Runtime do not interpret or select
  those kernels.
- MPSGraph and custom Metal kernels, Metal storage, and native bridges belong to `backends/metal`.
- CUDA lowering, kernels, storage, and native integration belong to `backends/cuda`.

A backend that implements fixed recurrent scan advertises only the exact variant, floating type,
fully static Shape, and direction combinations it executes. Analysis lowers one ordinary node to
one bounded recurrent-loop plan and declares all resources before slot assignment; finalization
constructs the reusable executable. Before output mutation, execution validates every length in
`[0, time]`. Invalid coordinates perform no recurrent dot products, gates, or state update, but
the first capability makes no active-row-compaction or packed-memory promise.

Concrete backends also own typed, version-controlled, tested candidate generators beside their
routes, compatible workload-cache lookup during preparation, and safe heuristic fallback. Shared
prepare and tuning orchestration sees candidates opaquely; backend-specific parameter vocabulary
does not leak into planning or shared parameter bags.

Backends may consume model, config, planning, runtime, prepare, backend-contract, and trace contracts. They must not own public tensor APIs or global compiler logic, and must not depend on engine.

`backends/openblas-provider` is a lower-level leaf used by the CPU backend. It owns loading, symbol binding, GEMM calls, and thread control only. It does not interpret config, decide fallback or ownership, participate in planning, manage residency, or expose the Tensor API. Dependency direction is `backends/cpu -> backends/openblas-provider`.

## Extensions

### `extensions/nn`

Owns stateful neural-network composition: `Module`, `Parameter`, `Buffer`, module-tree traversal,
train/eval mode, forward context, layers, blocks, and neural-network functional conveniences.
A parameter is a module-owned trainable value; a buffer is persistent module state that an
optimizer does not update. `train()` and `eval()` are module forward-behavior modes, so their
propagation belongs here rather than in an optimizer.

`extensions/nn` composes the generic tensor semantics supplied by `modules/model`. It does not
own autograd construction, optimizer algorithms, training-step orchestration, backend storage,
kernel selection, or concrete backend dependencies.

The existing static RNN, GRU, LSTM, and bidirectional containers retain their snapshotted Java
`long[]`, static unroll, compact per-step output, and final-state contracts. NN does not own the
fixed scan operation or its prepared loop, and no existing container is silently redirected to
it. Runtime-length API and compatibility migration remain a later NN decision after the complete
Model, Compiler, Engine, and concrete-backend path is executable.

### `extensions/training`

Owns training concepts and optimizer algorithms such as `Optimizer`, `Sgd`, `Adam`, `AdamW`,
parameter groups, sessions, and training steps. It consumes the trainable parameters declared by
`extensions/nn` modules and describes their mathematical updates; it does not own `Parameter`,
`Buffer`, layer behavior, or train/eval mode.

The dependency direction is `modules/model -> extensions/nn -> extensions/training`. Training
must not depend on concrete backend modules. A fused Adam route on Metal, for example, is a Metal
backend prepare/kernel concern rather than a `MetalOptimizerBridge` in training. See [Training
Graph](training-graph.md).

Training does not own fixed recurrent execution, valid-length handling, or backpropagation through
time. Compiler owns the later BPTT decision and gradient construction; Training remains the
consumer of published gradients and owner of optimizer orchestration.

### `extensions/onnx`

Owns ONNX import, export, and mapping to and from the model. It remains outside the runtime hot path and does not own backend lowering, kernel selection, execution, or residency.

## Tooling

`tools/benchmarks` owns fixed reproducible workloads and observational `BenchmarkReport`
evidence. It compares commits, models, and environments without selecting or mutating production
settings.

`tools/tuning` coordinates one explicit model-autotuning workflow. It reuses or measures canonical
local workload signatures, then measures a bounded set of complete valid graph and plan
candidates end to end. Compiler, planning, prepare, and concrete backends generate candidates for
their own decisions. Tuning owns measurement, explicit cache coordination, and selection, not
semantics, ownership policy, lowering, or route vocabulary.

Future results use an explicit file-backed reusable workload cache plus a model-specific plan
cache or prepared-plan record. A representative model corpus may pre-seed the same workload cache;
there is no separate platform-calibration subsystem or profile.

## Boundary summary

```text
model      = computation semantics and immutable graph model
planning   = backend-neutral ownership and logical requirements
compiler   = graph transformation, autograd, and compile artifacts
prepare    = shared transition contracts and validation
backend    = concrete lowering, implementation choice, and storage
runtime    = prepared execution and dynamic run state
engine     = explicit composition and public lifecycle
trace      = typed diagnostic leaf
nn         = module composition, parameters, buffers, and forward mode
training   = optimizer algorithms and training orchestration
```

The dependency implications of these responsibilities are detailed in [Dependency Rules](dependency-rules.md).
