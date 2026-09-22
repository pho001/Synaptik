# Synaptik Architecture Contract

This file is the authoritative architecture root and the repository's sole authority index. The
global rules in this root and the six scoped contracts explicitly incorporated below together form
the complete architecture contract.

All contributors and agents must preserve these boundaries unless this root and, when scoped
detail is affected, the one owning scoped contract are explicitly updated as part of the same
change.

For agent working instructions, see [`AGENTS.md`](AGENTS.md). For explanatory architecture
navigation, see [current architecture documentation](docs/architecture/current-architecture-plan.md).

## Authority, incorporation, and precedence

Only the following six files are incorporated as normative, and only within their stated,
non-overlapping scopes:

1. [Foundational module contract](docs/architecture/contracts/foundational-modules.md)
2. [Fixed recurrent-scan contract](docs/architecture/contracts/recurrent-scan.md)
3. [Compiler and automatic-differentiation contract](docs/architecture/contracts/compiler-autograd.md)
4. [Runtime, Prepare, and Engine contract](docs/architecture/contracts/runtime-prepare-engine.md)
5. [Backend execution contract](docs/architecture/contracts/backend-execution.md)
6. [Extensions and training contract](docs/architecture/contracts/extensions-training.md)

All other files under `docs/`, including architecture explanations, architecture decision records
(ADRs), design notes, guides, plans, and task briefs, are explanatory or coordinative evidence.
Source comments and tests are also evidence. None can override the architecture contract.

The global invariants and dependency rules in this root apply everywhere and take precedence over
a scoped contract on conflict. Each scoped contract is authoritative only for its stated scope and
must point back here; no scoped file has independent authority. Any overlap, contradiction,
missing applicable scope, or ambiguity requires contributors and agents to stop and make an
explicit coordinated architecture update. They must not choose an interpretation silently.

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
- `PreparedExecution` is the unique owner of persistent prepared resources. It has an explicit,
  idempotent close lifecycle even though its logical recipe remains immutable.
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

## Scope-indexed normative contracts

Use this table to find the one detailed contract that owns a concern. The linked heading is the
normative destination of the corresponding former root detail.

The following compatibility anchors keep existing focused links routed to this index while task
briefs and maintained navigation adopt the exact scoped destinations:

<a id="fixed-recurrent-scan-without-graph-regions"></a>
<a id="fixed-family-and-planned-model-surface"></a>
<a id="fixed-family-and-current-model-surface"></a>
<a id="static-shape-and-runtime-value-boundary"></a>
<a id="valid-lengths-traversal-and-outputs"></a>
<a id="purity-and-lifecycle-ownership"></a>
<a id="performance-and-migration-invariants"></a>
<a id="module-responsibilities"></a>
<a id="modulestrace"></a>
<a id="modulesbackend-contract"></a>
<a id="modulesmodel"></a>
<a id="modulesconfig"></a>
<a id="modulesplanning"></a>
<a id="partition-scoring"></a>
<a id="modulescompiler"></a>
<a id="compile-artifacts"></a>
<a id="training-graph-model"></a>
<a id="compiler-owned-automatic-differentiation"></a>
<a id="modulesruntime"></a>
<a id="modulesprepare"></a>
<a id="concrete-backend-modules"></a>
<a id="performance-evidence-and-optimization-tooling"></a>
<a id="cpu-backend-routes"></a>
<a id="metal-backend"></a>
<a id="openblas-provider"></a>
<a id="modulesengine"></a>
<a id="runtime-service-locator"></a>
<a id="reflective-backend-plugin-discovery"></a>
<a id="extensionsnn"></a>
<a id="extensionstraining"></a>
<a id="extensionsonnx"></a>
<a id="compile-lifecycle"></a>
<a id="prepare-lifecycle"></a>
<a id="run-lifecycle"></a>
<a id="optimizertraining-lifecycle"></a>

| Concern, module, or boundary | Normative contract and heading |
|---|---|
| Trace | [Foundational modules — `modules/trace`](docs/architecture/contracts/foundational-modules.md#modulestrace) |
| Backend Contract | [Foundational modules — `modules/backend-contract`](docs/architecture/contracts/foundational-modules.md#modulesbackend-contract) |
| Model and Tensor producer ownership | [Foundational modules — `modules/model`](docs/architecture/contracts/foundational-modules.md#modulesmodel) |
| Config | [Foundational modules — `modules/config`](docs/architecture/contracts/foundational-modules.md#modulesconfig) |
| Planning responsibility | [Foundational modules — `modules/planning`](docs/architecture/contracts/foundational-modules.md#modulesplanning) |
| Fixed recurrent scan | [Recurrent scan — fixed recurrent scan without graph regions](docs/architecture/contracts/recurrent-scan.md#fixed-recurrent-scan-without-graph-regions) |
| Partition scoring | [Compiler/autograd — partition scoring](docs/architecture/contracts/compiler-autograd.md#partition-scoring) |
| Compiler responsibility | [Compiler/autograd — `modules/compiler`](docs/architecture/contracts/compiler-autograd.md#modulescompiler) |
| Compile artifacts | [Compiler/autograd — compile artifacts](docs/architecture/contracts/compiler-autograd.md#compile-artifacts) |
| Training graph and compiler-owned autograd | [Compiler/autograd — training graph model](docs/architecture/contracts/compiler-autograd.md#training-graph-model) |
| Compile lifecycle | [Compiler/autograd — compile lifecycle](docs/architecture/contracts/compiler-autograd.md#compile-lifecycle) |
| Runtime | [Runtime/Prepare/Engine — `modules/runtime`](docs/architecture/contracts/runtime-prepare-engine.md#modulesruntime) |
| Prepare | [Runtime/Prepare/Engine — `modules/prepare`](docs/architecture/contracts/runtime-prepare-engine.md#modulesprepare) |
| Engine and explicit composition | [Runtime/Prepare/Engine — `modules/engine`](docs/architecture/contracts/runtime-prepare-engine.md#modulesengine) |
| Prepare and run lifecycles | [Runtime/Prepare/Engine — prepare lifecycle](docs/architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle) |
| Service-locator and plugin-discovery prohibitions | [Runtime/Prepare/Engine — runtime service locator](docs/architecture/contracts/runtime-prepare-engine.md#runtime-service-locator) |
| Concrete CPU, Metal, and CUDA backends | [Backend execution — concrete backend modules](docs/architecture/contracts/backend-execution.md#concrete-backend-modules) |
| Benchmarking and model-autotuning tools | [Backend execution — performance evidence and optimization tooling](docs/architecture/contracts/backend-execution.md#performance-evidence-and-optimization-tooling) |
| CPU routes and generated JVM bytecode | [Backend execution — CPU backend routes](docs/architecture/contracts/backend-execution.md#cpu-backend-routes) |
| Metal | [Backend execution — Metal backend](docs/architecture/contracts/backend-execution.md#metal-backend) |
| OpenBLAS provider | [Backend execution — OpenBLAS provider](docs/architecture/contracts/backend-execution.md#openblas-provider) |
| NN | [Extensions/training — `extensions/nn`](docs/architecture/contracts/extensions-training.md#extensionsnn) |
| Training | [Extensions/training — `extensions/training`](docs/architecture/contracts/extensions-training.md#extensionstraining) |
| ONNX | [Extensions/training — `extensions/onnx`](docs/architecture/contracts/extensions-training.md#extensionsonnx) |
| Optimizer/training lifecycle | [Extensions/training — optimizer/training lifecycle](docs/architecture/contracts/extensions-training.md#optimizertraining-lifecycle) |
| CLI | [Backend execution — other tool scopes](docs/architecture/contracts/backend-execution.md#other-tool-scopes) |
| Native projects | [Backend execution — concrete backend modules](docs/architecture/contracts/backend-execution.md#concrete-backend-modules) |
| Architecture, conformance, and integration testing | [Testing requirements](#testing-requirements) |

## Module-ownership routing

This compact routing summary does not replace the linked details:

| Scope | Owner |
|---|---|
| Model semantics and immutable graph model | Model |
| Backend-neutral capability, ownership, partitioning, and logical requirements | Planning |
| Graph transformations, autograd, publication, and compile artifacts | Compiler |
| Shared transition contracts, slot assignment, and prepared validation | Prepare |
| Concrete lowering, fusion, routes, storage, and backend execution | Each concrete backend |
| Prepared schedules, residency, publication, and per-run mutable state | Runtime |
| Explicit composition and public lifecycle | Engine |
| Typed diagnostics | Trace |
| Stateful modules, parameters, buffers, and train/eval behavior | NN |
| Optimizer algorithms and training orchestration | Training |
| Fixed benchmark evidence | `tools/benchmarks` |
| Model-autotuning measurement, cache coordination, and selection | `tools/tuning` |

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

<a id="documentation"></a>

## Documentation authority and architecture updates

The repository distinguishes authority from explanation and process:

```text
ARCHITECTURE.md
  authoritative root and sole authority index

docs/architecture/contracts/
  the six scoped normative contracts incorporated by the root

docs/
  explanatory architecture, guides, examples, ADRs, and design notes

docs/planning/
  non-authoritative implementation plans

AGENTS.md
  agent working instructions
```

When an architecture decision changes, update in the same change:

1. this root and the one owning scoped contract;
2. the relevant explanatory file under `docs/architecture/`;
3. an ADR under `docs/design/decisions/`, when the decision is significant; and
4. architecture tests under `testing/architecture-tests/`, when dependency rules change.

A change must not create overlapping scoped authority. If ownership would cross two scoped
contracts, update the root index to establish one unambiguous owner before implementation.

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
