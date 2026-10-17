# Dependency Rules

This document explains the dependency direction required by [`ARCHITECTURE.md`](../../ARCHITECTURE.md). It is explanatory; the root contract is authoritative.

The direction constrains both current and future code. Focused architecture-test coverage is planned and not yet complete; see the [architecture test guide](../developer-guide/architecture-tests.md).

## Intended direction

The compile-time side builds upward from small shared contracts into planning and compilation:

```text
trace
backend-contract
config
model
  -> planning
  -> compiler
```

The execution side consumes compile-time contracts while moving from shared runtime contracts to concrete backend implementations:

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

The engine depends on the components it composes:

```text
compiler
runtime
prepare
modules/backend/cpu
modules/backend/metal
modules/backend/cuda
tools/tuning
  -> engine
```

These diagrams express architectural direction; individual modules should depend only on the contracts they actually use.

The concrete CPU backend consumes only shared Model, Planning, Prepare, Runtime, and backend
contracts plus CPU-owned configuration. Engine passes Compiler-owned `CompileArtifacts` to
shared Prepare, which projects stable partition and schedule facts before invoking CPU. The CPU
build therefore has no production dependency on Compiler. Its compiler-backed fixtures use a
test-only dependency, while the prohibited `modules/backend/cpu -> modules/engine` edge also remains
absent.

The first advanced Engine lifecycle facade directly names Model-owned `Tensor` and
`HostTensorStorage` in its advanced public surface and Planning-owned
`BackendCapabilityProvider` in its package-private composition seam. Its implementation therefore
declares direct `modules/engine -> modules/model` and
`modules/engine -> modules/planning` dependencies rather than relying on Compiler, Prepare, or a
concrete backend to expose those contracts transitively. These edges realize Engine's existing
outer composition-root role; they do not move Model or Planning behavior into Engine and do not
permit either inward module to depend back on Engine.

The optional tuning operation directly invokes the workflow owned by `tools/tuning`, so Engine
declares `modules/engine -> tools/tuning`. This realizes the existing composition-root rule; it
does not move cache, measurement, or ranking ownership into Engine. The reverse
`tools/tuning -> modules/engine` edge remains forbidden and absent, so this is not a new root
architecture decision.

Neural-network composition and training have a separate extension direction:

```text
modules/model + public modules/engine facade -> modules/training (:extensions:training)
modules/model + modules/training -> extensions/nn
extensions/nn + public modules/engine facade + modules/model -> extensions/checkpoint
modules/training + extensions/checkpoint -> optional future extensions/training-checkpoint
```

`extensions/nn` owns modules, parameters, buffers, and train/eval forward behavior. Generic
Training has exactly Model and Engine `api` edges; a Training-only consumer needs no NN types.
NN has exactly Model and Training `api` edges: `Module.trainingParameters()` adapts its live tree
to Training's ordered admission boundary. That adapter owns no optimizer state or execution.
Training retains its public Engine lifecycle for one compiled/prepared forward/backward session.
This is not permission to import lower execution internals or add a direct NN Engine edge.

NN also owns the implemented
[closed declarative specification/reconstruction boundary](contracts/extensions-training.md#closed-declarative-models).
For example, describing a nested Sequential and constructing an alias-preserving descriptor skeleton
uses only NN plus Model metadata; the existing downstream recovery owner can then load its values.
Putting a file codec, Engine runner or Training lifecycle inside NN would reverse the owner boundary.
That declarative addition changed no edge; Training0007 separately reverses the admission edge.

The [accepted CP0005B whole-model contract](contracts/extensions-training.md#whole-model-definition-and-composition)
keeps persistence downstream: Checkpoint describes via public NN, captures/validates the unchanged
values component via public ModelCheckpoint, and reconstructs/loads a private NN candidate. No
new edge, private component parser or public validation credential is needed. For example, loading
a supported Sequential definition returns heap-backed eval state without opening an Engine;
executing its forward result remains the caller's ordinary Engine responsibility. This is not
implemented API or acceptance of future recurrent/Training resume contracts.

Checkpoint declares exactly NN, public Engine, and Model as `api` project edges because its public
typed inspection and save/load facade expose all three contracts. Its artifact API reads and
publishes bounded opaque files; its model package captures detached bytes through public Engine
compute and delegates whole-tree installation to NN. It has no backend or Training dependency.
The optional Training Checkpoint adapter is not in Gradle settings and must wait for its separate
gate. The [public model checkpoint decision](../design/decisions/0034-public-model-checkpoint.md)
explains the Model edge and target-defined alias policy.

## Concrete forbidden dependencies

- `modules/trace` must not depend on model, planning, compiler, runtime, prepare, engine, or concrete backends.
- `modules/model` must not depend on planning, compiler, runtime, prepare, engine, or concrete backends.
- `modules/config` must not depend on concrete backend implementations.
- `modules/planning` must not depend on concrete backends, runtime, prepare, or engine.
- `modules/compiler` must not depend on runtime, prepare, engine, or concrete backends.
- `modules/runtime` must not depend on concrete backends or engine.
- `modules/prepare` must not depend on concrete backend implementations.
- Concrete backends must not depend on `modules/engine`.
- `modules/backend/openblas-provider` must not depend on compiler, planning, runtime, prepare, engine, or the Tensor API.
- `extensions/nn` exposes only Model and Training directly; it must not directly depend on
  Compiler, Runtime, Prepare, Engine, Checkpoint, or concrete backends.
- `modules/training` exposes only Model and public Engine directly. It has no NN dependency or
  import, including test fixtures, and no direct Compiler, Runtime, Prepare, tuning, Engine
  internals, or concrete backend dependency.
- `extensions/checkpoint` depends directly only on NN, the public Engine facade, and Model; it
  must not depend on Training, Text, lower execution layers, tuning, or concrete backends. Inward
  modules, NN, and Training must not depend on Checkpoint.
- `extensions/onnx` must not depend on runtime hot-path execution internals.

## Why the boundaries matter

### Trace is a dependency leaf

Every phase may emit diagnostics, so trace DTOs must be safe for every phase to reference. If trace imported graph, runtime, or backend types, diagnostic use would introduce reverse dependencies and make the trace schema inherit business logic. Trace-local identifiers and typed payloads let producers describe events without importing one another.

### Model is backend- and runtime-independent

The model defines computation semantics and the public tensor abstraction. Keeping backend
capability, device residency, physical buffers, and prepared execution out of it makes the same
graph meaningful before any backend is chosen. It also prevents the Tensor's mutable borrowed
host-storage association from becoming runtime device state.

The compiler may depend inward on the model's public Tensor operations and producer/provenance
contracts to build gradients before capture. That existing direction does not let model own
derivative rules or depend back on compiler. Compiler-local identity maps are temporary
bookkeeping and create no new package or module edge.

### Runtime is independent of concrete backends

Runtime executes `PreparedExecutable` and schedule contracts. Concrete backend preparation supplies implementations before execution. A runtime dependency on concrete backends would pull backend discovery, lowering, or kernel choice into the hot path and break explicit composition.

### Backends are independent of engine

Engine is the outer composition root: it knows and wires concrete backend modules. If a backend depended on engine, composition would become cyclic and backend implementation would be coupled to the public orchestration layer.

The current public composition builder preserves this direction with concrete Engine-owned
`takeOwnership(...)` overloads. CPU and Metal publish backend-owned integrations; neither
implements an Engine-owned interface. Their appearance in public builder signatures makes the
Engine-to-CPU and Engine-to-Metal dependencies API-visible to Engine consumers, but never adds a
backend-to-Engine edge or exposes backend types through execution handles.

### Generic Training admission is independent of neural-network composition

Layers declare their own Parameter/Buffer state and choose train/eval forward behavior without
selecting an optimizer. NN's narrow Training edge exposes a live membership adapter, not state
declaration or optimizer orchestration. A non-NN owner can supply the same `TrainingParameters`
boundary without importing layers. Frozen lists cannot detect later binding replacement, so the
NN adapter retains the root and re-traverses exact ordered paths/wrappers/Tensors. Training checks
storage separately; copying into native memory therefore preserves that same membership adapter.

### Model-only recovery is downstream of NN, Engine, and Model

NN state paths and values can be meaningful without an optimizer; Engine provides the public
host-materialization boundary for any executed values. A model-only Checkpoint owner can consume
both without teaching NN file I/O or leaking backend representation access into persistence. A
future training-resume artifact adds Training-owned state through a separate adapter. If
Checkpoint instead coordinated Training state now, model-only persistence would acquire optimizer
restore obligations. NN's transitive live-admission dependency grants no such ownership or restore
authority to Checkpoint.

## Dependency scenario

A CPU partition preparer may implement a shared Prepare contract and return a Runtime
`PreparedExecutable`; those dependencies point from the concrete backend toward shared inward
contracts. `CpuBackendIntegration` exposes an artifact-free `PartitionPreparation` and schedule
assembler. Engine obtains the exact stable `PrepareContext` projection through
`GraphPreparation`, supplies it to CPU tuning when needed, and passes the resulting positional
preparation back to `GraphPreparation` for complete orchestration. Engine directly depends on
each public inward contract its lifecycle source names and may depend on CPU to compose the
adapter. If CPU imported Compiler artifacts or Engine to find configuration or register itself,
the backend would bypass the projection boundary or depend back on the composition root.

## Related semantic dependency rules

Source-level dependency tests should also protect the boundaries that type dependencies alone may miss:

- `Operation` must not expose `supportedBackends()`.
- Runtime hot-path code must not use `Operation` or `CompiledNode`.
- Planning scoring must not reference concrete kernel classes or prepared executables.
- Compile-time plans hold `BackendId`, not live backend services.
- CPU routes such as scalar, Vector API, generated JVM-bytecode CPU computation kernels, and
  OpenBLAS remain implementation routes inside the CPU backend.
- `extensions/nn` owns `Parameter`, `Buffer`, train/eval behavior and its live adapter;
  `modules/training` owns generic admission, optimizer algorithms and training orchestration.

## Architecture-test enforcement

Tests under `testing/architecture-tests/` should fail when forbidden module or package edges appear. They should cover every dependency rule above and add focused checks for:

- the trace and model leaf boundaries;
- config independence from concrete backend classes;
- planning and compiler independence from runtime and concrete implementations;
- runtime and prepare independence from concrete backends;
- backend independence from engine;
- Engine's exact direct dependency inventory, including direct Model, Planning, and tuning
  dependencies, plus tuning's absence of an Engine dependency;
- the CPU backend's exact production dependency set, excluding Compiler and Engine while allowing
  compiler-backed test fixtures only;
- the OpenBLAS provider's low-level leaf role;
- absence of backend support APIs on `Operation`;
- absence of compile-time graph types in the runtime hot path; and
- absence of concrete implementation references in partition scoring.
- exact public Model/Engine-to-Training and Model/Training-to-NN edges, physical Training mapping,
  generic signatures, NN-free Training sources and isolated consumer compilation.
- Checkpoint's present exact NN/Engine/Model edges, absent reverse edges, and exclusion of the optional
  Training Checkpoint adapter until its separate gate.

Architecture tests enforce the contract; they do not redefine it. When a dependency rule changes, update [`ARCHITECTURE.md`](../../ARCHITECTURE.md), the relevant explanatory document, an ADR when significant, and the architecture tests in the same change.

## Numerical-semantics dependency direction

Model owns the one family/dtype numerical contract. Planning queries occurrence-specific provider
capability without a graph-wide numerical selector; Compiler keeps guarded transformation
legality, and concrete backends qualify their routes. Config, Prepare, Engine, Runtime, and Trace
carry no numerical-profile identity or production tolerance policy. This preserves the same
dependency direction without making Config a semantics owner.
