# Engine Master Plan

## Goal

Provide the public lifecycle facade and explicit composition root for compiler, prepare, runtime, and concrete backends.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)

## Scope

- public compiled graph facade
- explicit backend registration
- deterministic standard composition of known built-in backends
- compile and prepare orchestration
- composition of runtime, validators, tracing, and backends
- explicit typed caller-input binding and published-result access
- explicit host materialization/publication needed by downstream persistence consumers

## Out of scope

- kernel implementations
- backend internals
- graph optimizer passes
- runtime service locator and core reflective discovery
- Tensor-owned execution, backward methods, gradients, or runtime state

## Module invariants

- Engine is the outer composition root.
- Every usable backend is registered explicitly by Engine construction code. Task 0001 owns one
  explicit CPU adapter because it is the sole supported complete lifecycle adapter. A later
  standard factory may own a fixed ordered inventory of known built-ins only as their adapters and
  complete schedule-composition contracts become real.
- Backend eligibility and partition ownership are resolved before Runtime execution.
- Concrete backends never depend on engine.

## Allowed dependencies

- modules/model
- modules/planning
- modules/compiler
- modules/runtime
- modules/prepare
- modules/config
- modules/trace
- concrete backend modules

Engine declares dependencies directly for every public contract its source names. Task 0001's
advanced API names Model `Tensor` and `HostTensorStorage`, while its package-private composition
seam names Planning `BackendCapabilityProvider`; direct Model and Planning dependencies are
therefore required rather than inherited transitively through Compiler, Prepare, or CPU.

## Forbidden dependencies

- No inward module may depend on engine.

## Package structure

```text
io.github.pho001.synaptik.engine/
  Engine                      public ordinary standard-composition owner
  CompiledGraph               public ordinary owner-bound compile handle and input metadata
  PreparedExecution           public ordinary owner-bound prepared handle
  RunResult                   public ordinary metadata-only publication lease
  AdvancedEngine              public advanced composition and lifecycle owner
  AdvancedCompiledGraph       public opaque owner-bound compile handle
  AdvancedPreparedExecution  public opaque owner-bound prepared handle
  AdvancedRunResult           public lifecycle-only result wrapper
  package-private composition and lifecycle machinery
```

The root package is a deliberate small facade. Task 0001 adds no public `spi` package or generic
backend abstraction: its package-private composition seam has the current CPU realization and a
focused fake test realization only. Task 0002 adds the distinct ordinary `Engine` owner with a
construction-and-close surface only. Complete task 0003 adds the three ordinary handle/result
types and lifecycle methods shown above without exposing the advanced representation-level
surface.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| [0001](tasks/0001-advanced-composition-and-representation-level-lifecycle-foundation.md) | Advanced composition and representation-level lifecycle foundation | Complete | Compiler 0006B3; CPU 0010F; current Prepare 0003–0004 and Runtime 0010 contracts | Own exactly one supported CPU lifecycle adapter and expose opaque owner-bound compile/prepare handles plus a representation-level run seam for integration and tests. Reject zero/pass-through and mixed/multi-partition preparation; do not claim nonexistent schedule composition. |
| [0002](tasks/0002-standard-built-in-composition.md) | Standard built-in composition | Complete | 0001; supported built-in backend adapters | Added a distinct ordinary `Engine.standard()` construction path that owns one fresh CPU-only built-in composition without exposing `CpuBackendIntegration`, advanced handles, or inward integration contracts. Preserved `AdvancedEngine.takeOwnership(...)`; mixed-backend success still requires another concrete adapter and a justified complete schedule-composition contract. |
| [0003](tasks/0003-typed-logical-input-binding-and-published-result-access.md) | Typed logical input binding and published-result access | Complete | 0001–0002; [Compiler 0006B4](../compiler/tasks/0006b4-stable-caller-input-tensor-identity-bindings.md); current publication/Prepare/Runtime contracts | Added the ordinary compile, prepare, TensorId-based host-input binding, synchronous run, and metadata-only publication/result lifecycle without exposing inward identities, coordinates, representations, or advanced handles. |
| 0004 | Explicit host materialization boundary | Draft | 0003; concrete-backend host-transfer routes | Materialize selected current Tensor/state values through prepared execution and publication into bounded caller-owned host payloads; add no backend access to NN, Training, or Checkpoint. |
| 0005 | Engine-owned one-shot forward convenience | Draft | 0002–0004 | Add an Engine-owned one-shot facade from Tensor output(s) and typed inputs through compile, prepare, execute, typed results, materialization, and cleanup. Fix exact names only when this task becomes the planning frontier; add no execution method or runtime dependency to `Tensor`. |
| 0006 | Engine-owned one-shot scalar-objective backward convenience | Draft | 0005; Compiler 0006 functional gradient request contract | Extend an Engine-owned execution request with a simple scalar-objective backward option requiring explicit gradient targets and lowering to Compiler's functional request. Retain 0003's explicit-seed compile overload and the advanced full-request path; infer neither targets nor mutable autograd state and promise no ambiguous no-argument `withBackward()`. |
| 0007 | Optional model-autotuning composition | Draft | 0002; 0005; Config 0006A; tools/tuning 0001; operational representative execution facts | Explicitly map Config's request into tuning's generic collaboration before preparation, retaining deterministic untuned fallback and keeping all tuning out of Runtime. This does not yet implement tools/tuning 0002 graph/plan search. |
| 0008 | Engine lifecycle capability checkpoint | Draft | 0001–0006; Compiler 0006B; CPU 0008A | Validate standard and advanced composition, typed input/output ownership, host materialization, one-shot forward/backward lowering, cleanup, concurrency, architecture tests, documentation, and representative NCW Conv1d plus NCHW Conv2d and NCDHW Conv3d forward execution before persistence adapters or NN convolution integration depend on Engine. |


## Milestones

- Explicit backend composition
- Compile facade
- Prepare and run lifecycle facade

## Current status

Tasks 0001–0003 are Complete. Compiler 0006B3, Prepare 0003–0004, Runtime 0010 and its closure hardening,
and CPU 0010F supply the bounded CPU-only Engine lifecycle without shared-contract changes:
`GraphCompilationPort` supplies complete compile artifacts, `GraphPreparation` accepts explicit
positional preparation plus one complete assembler, `PreparedExecutionRunner` runs
representation-level inputs, and `CpuBackendIntegration` supplies all three CPU collaborations.

The first clean implementation attempt exposed one planning error before any implementation was
retained: the planned API directly names Model types and its package-private composition seam
directly names a Planning type, but the Engine build does not yet declare either project. Task
0001 now requires the exact direct `modules:model` and `modules:planning` dependencies, the
focused Engine dependency inventory test, and repository-wide validation appropriate to a module-
dependency change. This is a realization of the existing outer composition direction, not an
architecture decision change.

The corrected implementation then proved a second planning omission after Engine compilation,
six focused Engine tests, one focused architecture test, and whitespace validation passed. The
integration-test module depended only on Engine, whose deliberate `implementation` dependencies
are not exported. Its test directly names CPU, Config, and Model contracts and javac must also
resolve Compiler and Runtime types exposed by the advanced Engine signatures. Task 0001 therefore
retains the integration module's Engine `implementation` dependency and adds exact ordered
`testImplementation` dependencies on Compiler, Runtime, Config, Model, and CPU. It does not widen
Engine dependencies to `api` or add unrelated integration dependencies. This is test-fixture
dependency closure, not a production architecture decision change.

Task 0001 is deliberately an advanced/integration foundation. It uses the current standalone
`CompileMode`, `GraphOptimizationConfig`, `BackendIntent`, and `PartitionScoringConfig` values and
the existing representation-level Runtime caller input. Missing `CompileConfig`, `PrepareConfig`,
and `RunOptions` aggregates do not block that seam; their final convenience ownership remains in
Config. The normal standard composition is current from Engine 0002, typed binding and result
access are current from 0003, and host materialization remains in 0004. Therefore 0001 is not
documented as the completed end-user experience.

The critical seam audit does not support the earlier broad implication that several registered
backends can already contribute to one schedule. CPU is the only supported concrete lifecycle
adapter. Its assembler deliberately owns the complete schedule for exactly one non-empty maximal
CPU partition; Metal and CUDA contain placeholders only. Task 0001 therefore takes ownership of
exactly one caller-supplied CPU integration, keeps compiled and prepared delegates behind opaque
owner-bound handles, and preserves CPU's zero/pass-through and mixed/multi-partition rejection.
It adds no public hypothetical backend interface. A package-private composition collaboration is
justified by the current CPU realization and lifecycle failure-injection tests.

The current standard composition is compatible with the architecture and required no
authoritative decision task. “Registered explicitly” permits its fixed Engine-owned factory to
construct and register only known built-in adapters in deterministic order. It does not permit
classpath or annotation scanning, `ServiceLoader`, hidden service location, mutable process-global
Engine state, or Runtime backend selection. The compile/prepare lifecycle selects eligible
partition ownership from that fixed composition. Until another concrete adapter and a shared
complete-schedule contribution contract exist, the truthful standard inventory is CPU-only.
Ordinary users will not name or construct a CPU adapter.

The supported future user surface converges on Tensor expressions, standard or advanced Engine
composition, compile/prepare/run, typed input bindings, typed/host results, one-shot convenience,
and optional autotuning. Compiler, Prepare, Runtime, and CPU integration contracts are public SPI
only and must not appear in ordinary user-facade signatures; `.internal` stays private.

One-shot execution remains Engine-owned. Task 0005 may provide ergonomics equivalent to executing
an output directly, but exact type and method names wait for that planning frontier and `Tensor`
gains no `execute`, `backward`, gradient field, Runtime dependency, or hidden lifecycle state.
Task 0006 lowers backward convenience to Compiler's explicit functional gradient request. Its
simple form is restricted to a scalar objective with explicit gradient targets; 0003's ordinary
explicit-seed compile overload and the advanced full-request path remain available. Engine must
not inspect the graph to guess targets or invent a no-argument backward promise whose seed/target
meaning is undefined.

Config 0006A mapping and tools/tuning 0001 integration belong to optional Engine 0007, not Engine
0001 or the standard untuned path. Tools/tuning 0002 remains later bounded graph/plan tuning.

Model/training checkpoint persistence depends on the future task 0004 boundary because the
current Runtime `RunResult` privately retains representations and exposes no value or storage
access. A checkpoint plan must not bypass that gap by reading backend storage from NN or Training.

The dimensional-convolution program adds no Engine operation switch or convolution-specific
binding API. Task 0008 will exercise representative rank-one composition and first-class rank-two
and rank-three convolution through the same typed logical-input, Prepare, Runtime, and publication
mapping established by tasks 0001–0006. This is the execution-readiness gate for the later NN
layer integration checkpoint; it does not move shape inference, lowering, or kernel selection into
Engine.

Detailed task 0001 records the exact advanced API, lifecycle gate, ownership transfer, failure
rollback, file ceiling, validation, and clean documentation handoff. Its final evidence includes
7/7 focused Engine tests, 2/2 integration tests, 1/1 focused architecture test, Engine Javadoc,
the distinct-package public fixture, exact `javap` inspection, and the repository checkpoint of
3,067 tests with zero failures or errors and 28 skipped. It also records the two historical
compile failures that corrected dependency scope; neither is a final failure.

Engine 0002 is Complete. Its implementation adds a distinct ordinary
`Engine` type with `standard()`, `isClosed()`, and `close()` only. The standard factory owns one
fresh CPU integration and delegates its lifetime to the already-proved advanced lifecycle; it
does not expose that delegate. Typed compile/prepare/run signatures were reserved for task 0003 so the
ordinary surface does not inherit Compiler, Runtime representation, or CPU integration types.
The implementation and documentation contexts passed the exact 12/12 Engine, 1/1 integration,
1/1 architecture, Javadoc, public-shape, fixture, bytecode, token, Markdown, scope, status,
unchanged-Gradle/advanced-source, and whitespace gates recorded by the task.

Detailed Engine 0003 is `Complete` after implementation context
`01a0a46c-8a9e-7862-bd81-0d3495092894` and clean documentation context
`01a0a488-b8dd-7bf1-a8fc-947f531f420f`. Complete Compiler 0006B4 exposes the immutable ordered
`TensorId`/final-`ValueId` association for every caller-bindable input. Existing publication facts
preserve forward and gradient Tensor identity, order, duplicates, and aliases; Prepare preserves
caller-input and forward-then-gradient publication occurrence order; Runtime supplies isolated
execution and a metadata-independent result lease; and CPU supplies the host-storage borrow seam.
The ordinary API binds arbitrary-order caller Tensors by ID, snapshots current caller-owned host
storage once per run, preserves every forward/gradient publication occurrence and alias role, and
keeps numerical values outside the metadata-only result. The corrected real CPU examples use
only CONTIGUOUS over resolved FLOAT32 leaves, including `seedLeaf.contiguous()` as the repeated
explicit seed. The exact lifecycle, ownership, validation order, 19-path implementation ceiling,
tests, documentation handoff, and current limitations are recorded in the task. No architecture
or inward-contract change was required. Tasks 0004–0008 remain `Draft`, and no task 0004
specification exists.

## Open questions

- Define the public materialized host payload, ownership, size limits, cancellation/failure, and
  close behavior without exposing concrete backend representation types.
- Decide whether materializing already host-backed leaves can use a proven direct fast path while
  preserving the same public ownership and validation contract.
- Expand the standard built-in inventory only after another supported lifecycle adapter and a
  complete multi-backend schedule-composition contract exist; task 0002 fixes the current
  inventory to CPU only.
- Define mixed-backend schedule contributions only after a second concrete lifecycle adapter
  establishes a non-hypothetical consumer need. The current complete CPU assembler cannot be
  combined with another complete assembler.
- Fix one-shot type/method names and result-close ergonomics only after tasks 0003–0004 establish
  typed binding, publication, and materialization ownership.

## Decisions made

- The implementation must follow the current architecture contract.
- Legacy code is capability evidence only; new implementation is written from scratch.

## Risks

- Becoming a service locator or absorbing backend implementation details.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).
