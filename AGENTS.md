# Agent Guidance

## Project Context

This repository is a Java tensor and compiled computation graph framework with CPU,
Metal, CUDA, OpenCL, and shared planning/runtime code. Treat package ownership and
dependency direction as part of correctness, not as optional code organization.

These instructions are self-contained and define the durable architecture for new
work. Planning documents may describe the mechanics and progress of an active
migration, but normal implementation work must not depend on reading a planning
document to discover the rules below.

## Source-First Rule

Before proposing or implementing a change:

1. Read the relevant production source, focused tests, package documentation, and
   configuration or tuning code.
2. Trace the actual call path from compile through prepare to execute when the
   change crosses phases.
3. Search for all consumers and tests with `rg`; do not infer ownership, behavior,
   or support from class and package names.
4. Identify whether each affected value is compile-time metadata, prepare-time
   policy, prepared immutable state, runtime dynamic state, or kernel data.
5. Check the current worktree before editing so existing user or agent changes are
   preserved.

Use local planning documents only when they are directly relevant to the task.
Do not require a particular planning framework or external workflow tool to
understand or modify the codebase.

## Durable Package Ownership

### `tensor`

`tensor` owns the logical public tensor API, dtype/shape/stride semantics, tensor
storage objects, and user-facing expression construction.

- Keep public `Tensor` behavior logical and backend-neutral.
- Do not add CPU, Metal, CUDA, OpenCL, provider, residency, or dispatch policy to
  `Tensor`.
- Public lifecycle methods may call `graph.CompiledGraph` and expose the public
  prepared execution contract, but tensor internals must not become a second
  runtime.
- Physical residency, transfers, prepared buffer bindings, and execution lifetime
  belong to `runtime`.

### `operations`

`operations` owns operation semantics and operation-specific metadata.

- Every concrete `Operation` implementation is the source of truth for its arity,
  fusability, semantic family, computational cost, result kind, control behavior,
  and operation-specific attributes.
- Do not duplicate this metadata in backend switches, registries, helper maps, or
  representative-operation heuristics.
- Compile and prepare code may translate an `Operation` into primitive IDs, flags,
  constants, and immutable prepared fields. The `Operation` object must not leak
  into the execution hot path.
- Add an operation property to the concrete operation when it is semantic. Add a
  backend capability to backend planning/prepare when it is implementation-specific.

### `graph`

`graph` owns the immutable compiled graph model and graph compilation workflow.

- `graph.model` contains immutable compiled node/value snapshots. It is a dependency
  leaf and must not import planning, prepare, runtime, trace producers, or concrete
  backends.
- `graph.optimizer` performs backend-neutral graph rewrites over graph/operation
  semantics. It must not contain concrete backend capability checks or runtime
  behavior.
- `graph.compile` coordinates snapshotting, optimization, lowering inputs, and the
  production of compile artifacts. It must not execute work or own prepared state.
- `graph.CompiledGraph` is the only lifecycle facade in `graph`. It may bridge the
  public `compile` and `prepare` lifecycle to planning, prepare orchestration, public
  runtime execution types, and compile trace types.
- Other graph classes must not import `prepare` or `runtime` merely for convenience.

### `planning`

The top-level `planning` package owns backend-neutral compile-time plans:
descriptors, intents, values, partitions, execution plans, costs, memory plans, and
materialization requirements.

- Planning describes what must be executed and what capabilities are required.
- Planning may consume backend-neutral capability contracts or capability snapshots;
  it must not import concrete backend packages.
- Planning must not execute kernels, access runtime buffers, read runtime state, or
  import prepare orchestration.
- Materialization requirements are explicit planning outputs. They are not hidden
  kernel behavior.

### `prepare.context`

`prepare.context` owns shared immutable inputs and indexes needed by backend
preparers. It may expose graph model, planning, configuration, and runtime contract
data required to compile a backend step.

- It must not select a concrete backend.
- It must not import concrete backend preparers.
- It must not import `prepare.orchestration`.

### `prepare.validation`

`prepare.validation` owns backend-neutral validation of planning and prepare
contracts.

- It validates invariants shared by multiple backends.
- It must not contain backend-specific kernel selection.
- It must not import concrete backend preparers or `prepare.orchestration`.

### `prepare.orchestration`

`prepare.orchestration` is the composition root for prepared execution.

- It may select and invoke concrete backend preparers.
- It combines compile artifacts, planning output, shared context/validation, runtime
  contracts, configuration, and trace snapshots.
- It does not implement backend kernel selection itself.
- Concrete backend preparers must never import orchestration, dispatchers, or
  builders from this package.

### Backend-specific prepare

`backend.<name>.prepare` is the compiler for that backend and remains backend-owned.
It converts backend-neutral plans into immutable, directly executable backend state.

Backend prepare is responsible for resolving:

- supported operation and provider route,
- dtype and storage representation,
- shape, layout, strides, offsets, and broadcast strategy,
- kernel and specialization,
- scalar, vector, or native implementation,
- thread count, chunking, and launch policy,
- workspace or scratch-buffer requirements,
- materialization/fallback eligibility and diagnostic snapshots.

Shared context, validation, and orchestration do not move into a concrete backend.
Conversely, backend-specific compilation logic does not move into top-level
`runtime` merely to make backend packages smaller.

### `runtime`

The top-level `runtime` package owns prepared execution contracts and dynamic
execution state: prepared steps, execution context, runner, state, residency,
memory/resource lifetime, device buffer bindings, transfers, publication, and
runtime-owned materialization support.

- Runtime invokes the executable contract attached during prepare.
- Runtime must not select or import a concrete backend.
- Runtime may depend on backend-neutral `backend.contract`, graph model, planning
  output, tensor storage, and trace DTOs where the lifecycle requires it.
- Runtime owns mutable state that exists only for a run. It must not redo static
  prepare decisions.
- Concrete backends implement runtime execution contracts; runtime does not call a
  central switch over concrete backend types.

### `trace`

The top-level `trace` package owns diagnostics DTOs only. Producers remain in their
own layers:

- graph/optimizer/planning produce compile trace snapshots,
- prepare orchestration and backend prepare produce prepare snapshots,
- runtime produces execution/transfer/lifetime snapshots,
- backend trace contributors produce backend execution snapshots.

Trace DTOs must not import their producers, concrete backends, graph objects,
planning objects, or live runtime state. Snapshot data into trace-owned primitives,
strings, enums, and records. Trace collection must not perform capability, config,
dispatch, or fallback decisions.

### `backend.contract`

`backend.contract` owns small backend-neutral identities and capability contracts.
It is a dependency leaf.

- It must not import graph, planning, prepare, runtime, trace, or concrete backends.
- Do not turn it into a generic backend service layer.

### `backend.provider`

`backend.provider` owns low-level integrations with external compute libraries that
can be shared by concrete backends.

- A provider exposes direct capability queries and low-level entry points.
- A provider does not select routes, read application configuration, apply tuning
  thresholds, decide fallback, build prepared artifacts, or produce policy traces.
- Provider packages must remain low-level leaves and must not import graph,
  planning, prepare, runtime, trace, tensor, or concrete backend families.
- Backend-specific adapters around a provider remain owned by that backend.

### Concrete backend families

Concrete backend code lives under `backend.<name>`. Follow the established naming
of that family rather than forcing a repository-wide synonym:

```text
backend.<name>/
  prepare/       backend compiler and dispatch resolution
  exec/          or execution/, according to the existing family convention
  kernels/       small concrete compute implementations
  launch/        backend launch and range mechanics when needed
  storage/       backend-specific storage access plans/adapters when needed
  trace/         backend trace producers when needed
  provider/      backend-owned adapters around shared external providers
```

Do not create empty packages to satisfy this diagram. Create a package only when a
real owned responsibility exists.

Kernels must be small compute implementations. They receive prepared arguments and
perform arithmetic, memory access, or a direct provider call. They must not parse
configuration, inspect graph objects, select a route, query provider availability,
or silently choose a fallback.

### `config`

`config` owns declarative user/runtime configuration and immutable configuration
values. It does not own execution, hardware probing, or calibration algorithms.
Configuration enums are data-only unless parsing is genuinely their semantic
responsibility and is consistently used across config IO.

### `tuning`

`tuning` owns calibration, candidate generation, benchmark-driven threshold
selection, and canonical platform profiles. Concrete backends consume resolved
values from configuration/profiles; they do not embed benchmark policy as hardcoded
magic constants.

## Dependency Direction

Use this high-level direction for new code:

```text
operations/tensor + backend.contract
                -> graph.model
                -> graph.optimizer/graph.compile + planning
                -> prepare.context/validation
                -> backend.<name>.prepare

prepare.orchestration -> concrete backend preparers
backend.<name>.prepare -> prepared runtime contracts + backend execution/kernels
runtime.runner         -> prepared executable contract
concrete backend       -> runtime contracts

producers -> trace DTOs
concrete backend adapters -> backend.provider
```

The following dependencies are forbidden:

- `graph.model` to planning, prepare, runtime, or a concrete backend.
- `graph.optimizer` to prepare, runtime, or a concrete backend.
- `planning` to `graph.CompiledGraph`, graph compile orchestration, prepare,
  runtime, or a concrete backend.
- `prepare.context` or `prepare.validation` to orchestration or concrete preparers.
- `backend.<name>.prepare` to `prepare.orchestration`.
- `runtime` to CPU, cpu1, Metal, CUDA, OpenCL, or another concrete backend.
- `trace` DTOs to their graph/planning/prepare/runtime/backend producers.
- shared providers to concrete backends or higher lifecycle layers.
- kernels to graph optimizer, prepare orchestration, config parsing, or tuning.

When a desired import violates this direction, move the data contract to the layer
that owns it or pass a prepared primitive snapshot. Do not hide the cycle behind a
facade, service locator, reflection, or a generic `Object` payload.

## Compile, Prepare, And Hot Path

The normal lifecycle is:

```text
Tensor expression
  -> graph compilation and optimizer
  -> backend-neutral planning
  -> prepare orchestration
  -> backend-specific prepare
  -> immutable prepared execution
  -> runtime runner
  -> backend executable/kernel
```

Shape, dtype, layout, strides, storage kind, offsets, broadcasting, kernel variant,
vector policy, thread count, chunk sizing, provider route, capability result,
fallback decision, and workspace sizing belong in compile or prepare whenever they
are stable for the prepared plan.

Execution must apply the prepared plan. The hot path must not:

- read configuration or system properties,
- query provider or hardware capabilities,
- inspect `Operation` or graph nodes,
- perform registry lookup or kernel selection,
- recompute static shape/layout/storage classifications,
- choose vector/scalar, thread policy, provider route, or fallback,
- allocate persistent output storage that could have been planned and reused.

Legitimate runtime dynamic state includes actual input/output buffer bindings,
run-specific tensor values, leased scratch buffers, dependency completion, command
handles, task scheduling, error propagation, cancellation, and timing/trace samples.
Dynamic state may apply a prepared decision; it must not reinterpret the policy.

## Storage, Residency, And Materialization

- Public `Tensor` remains logical. Backend residency is not a public tensor mode.
- Runtime owns host/device residency, transfers, resource lifetime, slot bindings,
  and publication of results.
- Planning represents required layout/storage/materialization explicitly from
  backend-neutral capability contracts or capability snapshots.
- Backend prepare validates the selected route against those requirements and
  compiles the concrete copy/materialization step when required.
- A kernel must never silently convert storage, materialize a contiguous copy, move
  data between devices, or fall back to another implementation.
- Every conversion, transfer, materialization, and fallback must be represented in
  the plan/prepared execution and visible in diagnostics and benchmarks.

## BLAS And External Providers

OpenBLAS has one shared low-level home:

```text
backend.provider.blas.openblas
```

This package is JDK/FFM-only. It owns symbol lookup, low-level availability queries,
thread-control entry points, GEMM layout conversion, and direct array/segment GEMM
calls. It must not know route selection, thresholds, debug policy, tuning profiles,
fallback, graph/planning objects, or prepared artifacts.

`config.runtime.BlasProvider` is a data-only provider selection enum. It does not
probe libraries or read system properties.

CPU and cpu1 own their respective BLAS adapters and prepare decisions. Their prepare
code resolves provider selection, shape/work thresholds, availability, debug state,
and thread policy once, then stores the result in backend-owned prepared state.
Execution calls the selected adapter/provider path directly.

Apply the same ownership test to future external providers: shared native bindings
are low-level leaves; route and policy remain in backend prepare; configuration and
tuning remain in their own packages.

## Architecture And Abstraction Rules

- Add an abstraction only when there is a real second implementation/use or a clear
  dependency boundary that cannot be expressed directly.
- Do not add speculative interfaces, registries, factories, extension points, or
  generic wrappers for hypothetical future backends.
- Prefer direct, explicit data flow and focused immutable records over service
  lookup and hidden mutable state.
- Keep one source of truth. Derived prepared fields are allowed; parallel semantic
  classification tables are not.
- Do not create package-level facades that merely forward calls.
- Do not force unrelated backends into identical class structure when their
  execution models differ. They must obey the same ownership boundaries.
- Prefer backend-neutral accelerator contracts over Metal-only, CUDA-only, or
  OpenCL-only shortcuts when the concept is genuinely shared.

Validate data at the boundary that owns the invariant. Public input, compile
artifacts, planning output, and prepared contracts should fail early with useful
errors. Once compile or prepare has established an invariant, do not repeat the same
null, capability, dtype, layout, or route checks in every executable unit, kernel,
or loop iteration. Unsupported cases should fail during compile/prepare rather than
trigger a defensive hot-path fallback.

## Active Migration Versus Durable Ownership

The ownership rules above describe the target architecture and apply to all new
code now. Some existing classes may still live in known legacy packages while an
active migration is incomplete.

- Do not add new responsibilities or new consumers to a known legacy package.
- Do not perform unrelated mass package moves outside the requested scope.
- When a package migration is explicitly approved, update production code, tests,
  docs, and hygiene checks atomically and delete the old path in the same change.
- The project is not yet a production compatibility surface. Do not preserve old
  internal packages through deprecated aliases, dual read/write paths, transitional
  facades, forwarding wrappers, or temporary adapters.
- A compatibility layer is allowed only when the assignment explicitly requires
  external compatibility or a safe staged rollout. Document the concrete risk, why
  an atomic change is impossible, and the exact removal criterion.
- Do not copy one-time move maps, phase tracking, or migration checklists into this
  file. Those belong in the relevant planning document.

## Coding Task Execution

Every repository-changing coding task must run in a dedicated, isolated agent or
thread context.

For each coding task:

1. Give the isolated context one explicit objective and a precise write scope.
2. Tell it that other work may exist in the repository and that it must not revert
   changes outside its scope.
3. Have that same context inspect the relevant source, implement the change, run
   focused validation, and review its own diff against the assignment.
4. The main context may inspect the resulting diff and run additional validation,
   but must not silently broaden the task.
5. Use separate isolated contexts for independent tasks. Do not let multiple agents
   edit the same files concurrently.

Small documentation or configuration edits are still coding tasks when they change
the repository. Analysis and read-only investigation do not require a new isolated
context.

## Editing And Git Rules

- Assume the worktree may be dirty. Existing changes belong to the user or another
  agent unless proven otherwise.
- Never revert unrelated work. If an existing change intersects the task, understand
  and preserve it; ask only when it makes the requested change impossible.
- Search with `rg` or `rg --files` first. Use the next available tool only when `rg`
  is unavailable.
- Use `apply_patch` for manual file edits. Use formatters or mechanical rewrite
  tools only for genuinely mechanical changes.
- Default to ASCII. Introduce Unicode only when the file already uses it and the
  content requires it.
- Add concise comments only where intent is not clear from code. Do not narrate
  obvious statements.
- Do not use destructive Git commands such as `git reset --hard` or
  `git checkout --` unless the user explicitly requests them and the scope is clear.
- Do not commit or push unless requested. When requested, commit by coherent topic.
- Do not commit temporary verification files, local benchmark output, or local
  calibration artifacts. Commit calibration data only when intentionally updating
  a canonical profile or fixture.

## Verification

Verification depth must match risk and blast radius.

- Run focused tests for the touched behavior and package boundaries.
- Add or update contract/parity tests when changing a backend implementation,
  storage route, lowering rule, prepared contract, or cross-backend behavior.
- Run compile checks after package/import changes.
- Use targeted Gradle filters when the full suite is expensive or includes debug
  benchmarks.
- Run broader suites for shared contracts, runtime state, planning, graph rewrites,
  or public API changes.
- If a required test cannot run, state exactly which test was not run and why.

Common commands:

```bash
./gradlew classes
./gradlew test --tests <TestClassOrPattern>
./gradlew metalTest
```

For Metal native work, build or point to the native shim as required by the relevant
tests and documentation.

### Performance validation

Do not claim a performance improvement or regression without measurement.

A meaningful benchmark comparison must:

- compare equivalent semantics, dtype, shapes, layout, storage, and output checks,
- report the exact baseline and changed route,
- include warmup and enough measured iterations or forks for stable results,
- record relevant JDK, hardware, thread count, vector/provider route, configuration,
  and tuning profile,
- distinguish conversion/materialization cost from kernel-only cost,
- report variability or multiple samples rather than one favorable timing,
- explain observed differences using measured evidence, not assumptions.

Keep benchmark/calibration output local unless the task explicitly updates a
canonical report, profile, or fixture.

## Technical Debt Policy

Do not introduce avoidable technical debt:

- unnecessary compatibility or migration layers,
- unused abstractions or speculative extension points,
- duplicated logic or metadata,
- dead code,
- hidden fallback or conversion,
- temporary paths without an explicit removal criterion,
- broad refactors unrelated to the task.

After implementation, review the diff against the original assignment and remove
obsolete code made unnecessary by the change.

## Final Response Contract

The final response for every coding task must use these headings exactly:

### What changed

List the files, components, functions, and behavior changed.

### Why it changed

Explain why the changes were necessary for the assignment and architecture.

### Validation against the assignment

List the checks and tests run and confirm how they cover the request.

### Remaining debt or follow-up work

State limitations, trade-offs, skipped validation, or remaining work. If none
exists, say exactly:

> No known remaining technical debt was introduced by this change.
