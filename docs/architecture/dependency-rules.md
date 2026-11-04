# [Dependency](../glossary.md#term-dependency) Rules

This document explains the [dependency direction](../glossary.md#java-platform-and-build-tools) required by [`ARCHITECTURE.md`](../../ARCHITECTURE.md). It is explanatory; the root contract is authoritative.

The direction constrains both current and future code. Focused architecture-test coverage is planned and not yet complete; see the [architecture test guide](../developer-guide/architecture-tests.md)<span data-glossary-navigation="true"> ([architecture test glossary](../glossary.md#term-architecture-test))</span>.

A **dependency** means one [module](../glossary.md#term-module) needs another module's types or behavior. A dependency is not
the direction in which [tensor](../glossary.md#tensor) values flow during execution. In the layer sketches below, arrows
lead toward the consuming outer layer: `model -> planning` means [Planning](../glossary.md#term-planning) consumes [Model](../glossary.md#term-model), not
that Model imports Planning. Later explicit `modules/... -> modules/...` dependency examples name
the dependent module first, as their surrounding prose states.

An **[API](../glossary.md#term-application-programming-interface-api) export** makes a dependency's public types available to a module's consumers.
An **implementation-only** dependency is used internally. A **transitive** dependency is reached
through another dependency; an explicit direct edge records that the module uses the types itself.
These are build/module relationships, not permissions to take over the other module's decisions.

## Intended direction

The [compile](../glossary.md#compile)-time side builds upward from small shared contracts into [planning](../glossary.md#term-planning) and [compilation](../glossary.md#compiler-engine-and-execution-stage-names):

```text
trace
backend-contract
config
model
  -> planning
  -> compiler
```

The execution side consumes compile-time contracts while moving from shared [runtime](../glossary.md#term-runtime) contracts to [concrete backend](../glossary.md#concrete-backend) implementations:

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

The [engine](../glossary.md#term-engine) depends on the components it composes:

```text
compiler
runtime
prepare
modules/backend/cpu
modules/backend/metal
tools/tuning
  -> engine
```

These diagrams express architectural direction; individual [modules](../glossary.md#term-module) should depend only on the contracts they actually use.

The first-production product closure includes the implemented [CPU](../glossary.md#term-cpu) and [Metal](../glossary.md#term-metal) [backends](../glossary.md#backend), not an
Engine [dependency](../glossary.md#term-dependency) on the unused [CUDA](../glossary.md#term-cuda) marker. CUDA remains registered with its source for future
development but is not a published supported product. Existing public-[signature](../glossary.md#term-signature) types require
[API](../glossary.md#term-application-programming-interface-api) exports; private implementation use such as Engine's Tuning runners does not.

The concrete CPU backend consumes only shared [Model](../glossary.md#term-model), Planning, [Prepare](../glossary.md#prepare), Runtime, and backend
contracts plus CPU-owned configuration. Engine passes [Compiler](../glossary.md#term-compiler)-owned `CompileArtifacts` to
shared Prepare, which projects stable [partition](../glossary.md#partition) and schedule facts before invoking CPU. The CPU
build therefore has no production dependency on Compiler. Its compiler-backed fixtures use a
test-only dependency, while the prohibited `modules/backend/cpu -> modules/engine` edge also remains
absent.

The first advanced Engine [lifecycle](../glossary.md#lifecycle) [facade](../glossary.md#term-facade) directly names Model-owned [`Tensor`](../glossary.md#tensor) and
`HostTensorStorage` in its advanced public surface and Planning-owned
[`BackendCapabilityProvider`](../glossary.md#backend-capability-provider--backendcapabilityprovider) in its package-private composition seam. Its implementation therefore
declares direct `modules/engine -> modules/model` and
`modules/engine -> modules/planning` dependencies rather than relying on Compiler, Prepare, or a
concrete backend to expose those contracts transitively. These edges realize Engine's existing
outer composition-root role; they do not move Model or Planning behavior into Engine and do not
permit either inward module to depend back on Engine.

The optional tuning [operation](../glossary.md#operation) directly invokes the workflow owned by `tools/tuning`, so Engine
declares `modules/engine -> tools/tuning`. This realizes the existing composition-root rule; it
does not move [cache](../glossary.md#term-cache), measurement, or [ranking](../glossary.md#term-ranking) [ownership](../glossary.md#term-ownership-meanings) into Engine. The reverse
`tools/tuning -> modules/engine` edge remains forbidden and absent, so this is not a new root
architecture decision.

Generic [Checkpoint](../glossary.md#term-checkpoint) is a cold [owner](../glossary.md#term-owner), not an Engine execution layer. Arrows below [mean](../glossary.md#term-mean)
“is consumed by”; public signatures use API edges and Compiler remains implementation-only.

```text
Model + Engine -> modules/training (:extensions:training)
Model (API export only) -> extensions/data (:extensions:data)
Data (API export; Model transitive) -> extensions/text (:extensions:text)
Model + Compiler (cold only) -> modules/checkpoint (:extensions:checkpoint)
Model + Training + Checkpoint + Engine -> extensions/nn
Training + Checkpoint -> optional future training-checkpoint (not included)
```

Training retains exactly public Model/Engine dependencies and generic live [admission](../glossary.md#term-admission). [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) has
exactly public Model/Training/Checkpoint/Engine edges: its live tree adapter owns membership,
and owned Models compose public Engine preparations, Training sessions and cold GraphCheckpoint values;
its relocated `nn.checkpoint` APIs retain old Engine-materializing model.v1/whole-model.v1
readers and flexible declarative semantics. They are not a second execution pipeline.

Generic Checkpoint exposes only Model publicly and uses Compiler's existing cold [capture](../glossary.md#term-capture-meanings) and
reconstruction privately. For example, saving a [MATMUL](../glossary.md#matrix-multiplication--matmul)/PERMUTE/ADD [graph](../glossary.md#graph) reads named [host](../glossary.md#term-host) [leaves](../glossary.md#term-leaf),
not [prototype](../glossary.md#term-prototype) input values, without opening an Engine; load returns ordinary symbolic Tensors.
It does not call NN or retain a second Tensor [DAG](../glossary.md#term-directed-acyclic-graph-dag). [Device](../glossary.md#term-device)-only state rejects rather than triggering
execution. The [graph API](../reference/checkpoint/graph.md) explains static signatures; the
[old NN API](../reference/checkpoint/models.md) retains flexible leading [dimensions](../glossary.md#dimension) and old defaults.

## Concrete forbidden dependencies

- `modules/trace` must not depend on [model](../glossary.md#term-model), [planning](../glossary.md#term-planning), [compiler](../glossary.md#term-compiler), [runtime](../glossary.md#term-runtime), [prepare](../glossary.md#prepare), [engine](../glossary.md#term-engine), or concrete [backends](../glossary.md#backend).
- `modules/model` must not depend on planning, compiler, runtime, prepare, engine, or concrete backends.
- `modules/config` must not depend on [concrete backend](../glossary.md#concrete-backend) implementations.
- `modules/planning` must not depend on concrete backends, runtime, prepare, or engine.
- `modules/compiler` must not depend on runtime, prepare, engine, or concrete backends.
- `modules/runtime` must not depend on concrete backends or engine.
- `modules/prepare` must not depend on concrete backend implementations.
- Concrete backends must not depend on `modules/engine`.
- `modules/backend/openblas-provider` must not depend on compiler, planning, runtime, prepare, engine, or the [Tensor](../glossary.md#tensor) [API](../glossary.md#term-application-programming-interface-api).
- `extensions/data` has only `api(project(":modules:model"))`, with no other project edge even
  in tests. Inward modules and existing extensions cannot depend back on Data except the
  separately approved Text edge below. Vision still needs a separate decision. A Data-only consumer
  may compile a Model Tensor declaration through that API export; adding Engine to a Data test
  would violate the boundary rather than establish execution support. See the<span data-glossary-navigation="true"> ([modules glossary](../glossary.md#term-module); [compile glossary](../glossary.md#compile))</span>
  [owning Data contract](contracts/extensions-training.md#extensionsdata) and
  [ADR 0042](../design/decisions/0042-data-extension-boundary.md)<span data-glossary-navigation="true"> ([ADR glossary](../glossary.md#term-architecture-decision-record-adr))</span>.
- `extensions/text` has exactly `api(project(":extensions:data"))` across all
  configurations; Model is transitive, with no direct project edge. Only public Data/Model imports
  are permitted. Text has no NN/Engine/backend or other project edge, and NN has no Data/Text edge.
  Numerical composition belongs to the integration-test project, whose Text edge is test-only.<span data-glossary-navigation="true"> ([NN glossary](../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
  [ADR 0043](../design/decisions/0043-whitespace-text-and-padded-integral-batches.md) records this
  boundary; an isolated Text-only consumer compiles Data/Model declarations through that export.
- `extensions/nn` exposes Model, Training, Checkpoint and public Engine directly; it must not
  directly depend on Compiler, Runtime, Prepare or concrete backends.<span data-glossary-navigation="true"> ([Checkpoint glossary](../glossary.md#term-checkpoint))</span>
- `modules/training` exposes only Model and public Engine directly. It has no NN dependency or
  import, including test fixtures, and no direct Compiler, Runtime, Prepare, tuning, Engine
  internals, or concrete backend dependency.<span data-glossary-navigation="true"> ([dependency glossary](../glossary.md#term-dependency))</span>
- `modules/checkpoint` has exactly api(Model) and implementation(Compiler), cold-source-only.
  It has no NN, Training, Engine, Runtime, Prepare, tuning or backend dependency, including tests.
  Only NN may depend on it among inward composition modules; Compiler never depends back.
- `extensions/onnx` must not depend on runtime hot-path execution internals.

## Why the boundaries matter

### [Trace](../glossary.md#trace) is a [dependency leaf](../glossary.md#term-dependency-leaf)

A [dependency](../glossary.md#term-dependency) **[leaf](../glossary.md#term-leaf)** needs no higher layer. A **[DTO](../glossary.md#data-transfer-object--dto)** is a [data-transfer object](../glossary.md#data-transfer-object--dto): a structured
description of a fact, without the [graph](../glossary.md#graph) or execution [owner](../glossary.md#term-owner) that produced it.

Every phase may emit diagnostics, so trace DTOs must be safe for every phase to reference. If trace imported graph, [runtime](../glossary.md#term-runtime), or [backend](../glossary.md#backend) types, diagnostic use would introduce reverse dependencies and make the trace [schema](../glossary.md#term-schema) inherit business logic. Trace-local identifiers and typed payloads let [producers](../glossary.md#term-producer) describe events without importing one another.

### [Model](../glossary.md#term-model) is [backend](../glossary.md#backend)- and [runtime](../glossary.md#term-runtime)-independent

The model defines computation semantics and the public [tensor](../glossary.md#tensor) abstraction. Keeping [backend
capability](../glossary.md#backend-capability), [device](../glossary.md#term-device) [residency](../glossary.md#residency), physical [buffers](../glossary.md#term-buffer), and [prepared execution](../glossary.md#prepared-execution--preparedexecution) out of it makes the same
[graph](../glossary.md#graph) meaningful before any backend is chosen. It also prevents the Tensor's [mutable](../glossary.md#term-mutable) [borrowed](../glossary.md#ownership-borrowing-and-lease)
[host](../glossary.md#term-host)-storage association from becoming runtime device state.

The [compiler](../glossary.md#term-compiler) may depend inward on the model's public Tensor operations and [producer](../glossary.md#term-producer)/[provenance](../glossary.md#provenance)
contracts to build [gradients](../glossary.md#term-gradient) before [capture](../glossary.md#term-capture-meanings). That existing direction does not let model own
[derivative](../glossary.md#term-derivative) rules or depend back on compiler. Compiler-local [identity](../glossary.md#term-identity-meanings) maps are temporary
bookkeeping and create no new package or [module](../glossary.md#term-module) edge.

### [Runtime](../glossary.md#term-runtime) is independent of concrete [backends](../glossary.md#backend)

Runtime executes [`PreparedExecutable`](../glossary.md#prepared-executable--preparedexecutable) and schedule contracts. [Concrete backend](../glossary.md#concrete-backend) preparation supplies implementations before execution. A runtime [dependency](../glossary.md#term-dependency) on concrete backends would pull backend discovery, [lowering](../glossary.md#lowering), or [kernel](../glossary.md#kernel) choice into the [hot path](../glossary.md#term-hot-path) and break explicit composition.

### [Backends](../glossary.md#backend) are independent of [engine](../glossary.md#term-engine)

Engine is the outer [composition root](../glossary.md#term-composition-root): it knows and wires [concrete backend](../glossary.md#concrete-backend) [modules](../glossary.md#term-module). If a backend depended on engine, composition would become cyclic and backend implementation would be coupled to the public orchestration layer.

The current public composition builder preserves this direction with concrete Engine-owned
`takeOwnership(...)` overloads. [CPU](../glossary.md#term-cpu) and [Metal](../glossary.md#term-metal) publish backend-owned integrations; neither
implements an Engine-owned interface. Their appearance in public builder [signatures](../glossary.md#term-signature) makes the
Engine-to-CPU and Engine-to-Metal dependencies [API](../glossary.md#term-application-programming-interface-api)-visible to Engine consumers, but never adds a
backend-to-Engine edge or exposes backend types through execution handles.

### Generic Training [admission](../glossary.md#term-admission) is independent of neural-network composition

Layers declare their own [Parameter](../glossary.md#term-parameter)/[Buffer](../glossary.md#term-buffer) state and choose train/eval forward behavior without
selecting an [optimizer](../glossary.md#term-optimizer). [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context)'s Training edge exposes live membership and delegates configured [owned
Model](../glossary.md#model-module-buffer-and-scan-distinctions) training to [TrainingSession](../glossary.md#training-session--trainingsession), never implementing optimizer algorithms. A non-NN [owner](../glossary.md#term-owner) can supply the same [`TrainingParameters`](../glossary.md#live-parameter-source--trainingparameters)
boundary without importing layers. Frozen lists cannot detect later [binding](../glossary.md#term-binding) replacement, so the
NN adapter retains the root and re-traverses exact ordered paths/wrappers/[Tensors](../glossary.md#tensor). Training checks
storage separately; copying into [native memory](../glossary.md#aliasing-and-arena) therefore preserves that same membership adapter.

### Generic [persistence](../glossary.md#term-persistence) and [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) recovery have distinct [owners](../glossary.md#term-owner)

Generic [Checkpoint](../glossary.md#term-checkpoint) reads [canonical](../glossary.md#term-canonical-form) [Model](../glossary.md#term-model) [host](../glossary.md#term-host) [leaves](../glossary.md#term-leaf) and delegates source [capture](../glossary.md#term-structural-capture)/reconstruction
to [Compiler](../glossary.md#term-compiler)'s cold [API](../glossary.md#term-application-programming-interface-api). It never asks [Engine](../glossary.md#term-engine) to materialize [device](../glossary.md#term-device) state. Old NN recovery instead
lives in `nn.checkpoint`: it uses public Engine compute and NN's strict whole-tree installation,
retaining the old flexible readers. This split avoids making a generic [graph](../glossary.md#graph) save depend on a
layer tree or an execution [backend](../glossary.md#backend). Neither path restores Training-owned [optimizer state](../glossary.md#term-optimizer-state); a
future separately gated adapter still needs each state [owner](../glossary.md#term-owner)'s [validation](../glossary.md#term-validation) and caller exclusion.

## [Dependency](../glossary.md#term-dependency) scenario

A [CPU](../glossary.md#term-cpu) [partition](../glossary.md#partition) preparer may implement a shared [Prepare](../glossary.md#prepare) contract and return a [Runtime](../glossary.md#term-runtime)
[`PreparedExecutable`](../glossary.md#prepared-executable--preparedexecutable); those dependencies point from the [concrete backend](../glossary.md#concrete-backend) toward shared inward
contracts. `CpuBackendIntegration` exposes an [artifact](../glossary.md#term-artifact)-free `PartitionPreparation` and schedule
assembler. [Engine](../glossary.md#term-engine) obtains the exact stable [`PrepareContext`](../glossary.md#prepare-context-preparecontext) projection through
[`GraphPreparation`](../glossary.md#graph-preparation--graphpreparation), supplies it to CPU tuning when needed, and passes the resulting positional
preparation back to `GraphPreparation` for complete orchestration. Engine directly depends on
each public inward contract its [lifecycle](../glossary.md#lifecycle) source names and may depend on CPU to compose the
adapter. If CPU imported [Compiler](../glossary.md#term-compiler) artifacts or Engine to find configuration or register itself,
the [backend](../glossary.md#backend) would bypass the projection boundary or depend back on the [composition root](../glossary.md#term-composition-root).

## Related semantic [dependency](../glossary.md#term-dependency) rules

Source-level dependency tests should also protect the boundaries that type dependencies alone may miss:

- [`Operation`](../glossary.md#operation) must not expose `supportedBackends()`.
- [Runtime](../glossary.md#term-runtime) hot-path code must not use `Operation` or `CompiledNode`.
- [Planning](../glossary.md#term-planning) scoring must not reference concrete [kernel](../glossary.md#kernel) classes or prepared executables.
- [Compile](../glossary.md#compile)-time plans hold [`BackendId`](../glossary.md#backend-identity--backendid), not live [backend](../glossary.md#backend) services.
- CPU routes such as scalar, Vector API, generated JVM-bytecode CPU computation kernels, and
  OpenBLAS remain implementation routes inside the CPU backend.<span data-glossary-navigation="true"> ([CPU glossary](../glossary.md#term-cpu); [scalar glossary](../glossary.md#term-scalar); [Vector API glossary](../glossary.md#term-vector-api); [JVM glossary](../glossary.md#term-java-virtual-machine-jvm); [bytecode glossary](../glossary.md#term-bytecode); [kernels glossary](../glossary.md#kernel); [OpenBLAS glossary](../glossary.md#term-openblas))</span>
- `extensions/nn` owns [`Parameter`](../glossary.md#term-parameter), [`Buffer`](../glossary.md#term-buffer), train/eval behavior and its live adapter;
  `modules/training` owns generic [admission](../glossary.md#term-admission), [optimizer](../glossary.md#term-optimizer) algorithms and training orchestration.

## Architecture-test enforcement

Tests under `testing/architecture-tests/` should fail when forbidden [module](../glossary.md#term-module) or package edges appear. They should cover every [dependency](../glossary.md#term-dependency) rule above and add focused checks for:

- the [trace](../glossary.md#trace) and [model](../glossary.md#term-model) [leaf](../glossary.md#term-leaf) boundaries;
- config independence from [concrete backend](../glossary.md#concrete-backend) classes;
- [planning](../glossary.md#term-planning) and [compiler](../glossary.md#term-compiler) independence from [runtime](../glossary.md#term-runtime) and concrete implementations;
- runtime and [prepare](../glossary.md#prepare) independence from concrete [backends](../glossary.md#backend);
- backend independence from [engine](../glossary.md#term-engine);
- Engine's exact direct dependency inventory, including direct Model, Planning, and tuning
  dependencies, plus tuning's absence of an Engine dependency;
- the CPU backend's exact production dependency set, excluding Compiler and Engine while allowing
  compiler-backed test fixtures only;<span data-glossary-navigation="true"> ([CPU glossary](../glossary.md#term-cpu))</span>
- the [OpenBLAS](../glossary.md#term-openblas) provider's low-level leaf role;
- Data registration/path, exact Model API edge in every configuration, absent reverse edges
  except authorized Text and absent Vision inclusion, namespace/import fixtures and isolated
  Data-only consumer compilation of Model declarations; Data0001 implements this enforcement,
  not batching or runtime binding;<span data-glossary-navigation="true"> ([API glossary](../glossary.md#term-application-programming-interface-api); [compilation glossary](../glossary.md#compiler-engine-and-execution-stage-names); [binding glossary](../glossary.md#term-binding))</span>
- Text registration/path, sole Data API edge, transitive Model exposure in a Text-only consumer,
  public Data/Model-only imports and forbidden NN-to-Data/Text edges; these checks must be updated
  with Text implementation, using positive and negative fixtures;<span data-glossary-navigation="true"> ([NN glossary](../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- absence of backend support APIs on [`Operation`](../glossary.md#operation);
- absence of [compile](../glossary.md#compile)-time [graph](../glossary.md#graph) types in the runtime [hot path](../glossary.md#term-hot-path); and
- absence of concrete implementation references in [partition](../glossary.md#partition) scoring.
- exact public Model/Engine-to-Training and Model/Training/Checkpoint/Engine-to-NN edges, physical Training mapping,
  generic signatures, NN-free Training sources and isolated consumer compilation.<span data-glossary-navigation="true"> ([Checkpoint glossary](../glossary.md#term-checkpoint); [signatures glossary](../glossary.md#term-signature))</span>
- Checkpoint's exact Model API/cold Compiler implementation edges, isolated consumers, and exclusion of the optional
  Training Checkpoint adapter until its separate gate.

Architecture tests enforce the contract; they do not redefine it. When a dependency rule changes, update [`ARCHITECTURE.md`](../../ARCHITECTURE.md), the relevant explanatory document, an [ADR](../glossary.md#term-architecture-decision-record-adr) when significant, and the architecture tests in the same change.

## Numerical-semantics [dependency direction](../glossary.md#java-platform-and-build-tools)

[Model](../glossary.md#term-model) owns the one family/[dtype](../glossary.md#data-type--datatype) numerical contract. [Planning](../glossary.md#term-planning) queries [occurrence](../glossary.md#term-occurrence)-specific provider
capability without a [graph](../glossary.md#graph)-wide numerical selector; [Compiler](../glossary.md#term-compiler) keeps guarded transformation
legality, and concrete [backends](../glossary.md#backend) qualify their routes. Config, [Prepare](../glossary.md#prepare), [Engine](../glossary.md#term-engine), [Runtime](../glossary.md#term-runtime), and [Trace](../glossary.md#trace)
[carry](../glossary.md#term-carry) no numerical-profile [identity](../glossary.md#term-identity-meanings) or production [tolerance](../glossary.md#term-tolerance) policy. This preserves the same
dependency direction without making Config a semantics [owner](../glossary.md#term-owner).
