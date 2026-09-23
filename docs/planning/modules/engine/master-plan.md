# Engine Master Plan

## Goal and authority

Engine is the public lifecycle facade and explicit composition root for compiler, Prepare,
Runtime, tuning, and concrete backends. This plan is a non-authoritative implementation map;
[`ARCHITECTURE.md`](../../../../ARCHITECTURE.md), especially `Core lifecycle`,
`modules/engine`, and the compile/prepare/run lifecycle headings, is authoritative.

Focused explanations and decisions:

- [Lifecycle](../../../architecture/lifecycle.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)
- [Runtime, Prepare, and Backend Boundary](../../../architecture/runtime-prepare-backend-boundary.md)
- [ADR 0013: Prepared-execution persistent-resource lifecycle](../../../design/decisions/0013-prepared-execution-persistent-resource-lifecycle.md)

## Lifecycle position

```text
Tensor outputs -> Engine compile handle -> Engine prepared handle
caller inputs  -> synchronous run -> leased RunResult
publication    -> explicit materialization -> detached HostTensorValue
```

Engine owns composition and outward handles. Prepare constructs the accepted recipe; Runtime
owns the inward `PreparedExecution`, its run lease, and physical prepared-resource lifetime.
One-shot compute/backward and representative tuning use the same boundaries with temporary
handles that Engine closes before returning or publishing another owner.

## Scope and non-goals

Engine owns standard and advanced composition, compilation/preparation orchestration, owner-bound
handles, typed input binding and publication metadata, explicit host materialization, one-shot
compute/backward workflows, and bounded optional model-autotuning composition.

Engine does not implement kernels, lowering, backend storage, graph optimizer passes, Runtime
state, a service locator, reflective discovery, Tensor-owned execution/backward state, or hidden
backend selection during a run.

## Stable invariants and ownership

- Engine is the outer composition root. It explicitly constructs or takes ownership of concrete
  adapters; concrete backends and inward modules never depend on Engine.
- `Engine.standard()` currently owns one fresh fixed CPU composition. `AdvancedEngine` takes one
  supported CPU integration. No current surface assembles a mixed-owner schedule.
- Ordinary public signatures expose Engine/Model values, not Compiler, Prepare, Runtime, or CPU
  SPI identities. The advanced surface remains an explicit lower-level integration boundary.
- Ordinary and advanced prepared handles are explicit closeable owners of exactly one inward
  Runtime execution. Closing a handle unregisters and closes that execution once but does not
  independently close an already-returned result.
- Runtime remains the unique run-lease authority. Engine synchronizes outward admission and
  delegate ownership without adding a second resource lease or waiting close protocol.
- Every one-shot, tuning trial, correctness run, loser, fallback rollback, and rejected
  preparation is closed exactly once. A selected preparation has one published Engine owner.
- Engine shutdown waits for admitted operations, then closes retained results in reverse run-
  publication order, retained preparations in reverse prepare-publication order, and finally the
  backend composition.
- Materialization borrows one authenticated publication only while its result is open and returns
  a detached immutable value. Engine never exposes a physical representation or backend handle.

## Dependencies

Current allowed direct dependencies are `modules/model`, `modules/planning`, `modules/compiler`,
`modules/runtime`, `modules/prepare`, `modules/config`, `modules/trace`, `backends/cpu`,
`backends/metal`, `backends/cuda`, and `tools/tuning`. Each edge exists only for a contract named
by composition. No inward module or concrete backend may depend on `modules/engine`.

## Package map

All types remain in `io.github.pho001.synaptik.engine`; the root is a deliberately small facade,
not a catch-all service registry.

| Surface | Types and role |
|---|---|
| Ordinary public lifecycle | `Engine`, `CompiledGraph`, closeable `PreparedExecution`, closeable `RunResult`, detached `HostTensorValue`, and `ScalarObjectiveBackwardResult`. |
| Public autotuning | `ModelAutotuningRequest` and `ModelAutotuningPreparation`, which publish one retained ordinary prepared handle plus outcome/evidence. |
| Advanced public lifecycle | `AdvancedEngine`, `AdvancedCompiledGraph`, closeable `AdvancedPreparedExecution`, and closeable `AdvancedRunResult`. |
| Package-private composition | `EngineBackendComposition`, the CPU realization, representative execution/correctness machinery, lifecycle registries, and cleanup arbitration. |

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| [0001](tasks/0001-advanced-composition-and-representation-level-lifecycle-foundation.md) | Advanced composition and representation-level lifecycle foundation | Complete | Compiler 0006B3; CPU 0010F; current Prepare 0003–0004 and Runtime 0010 contracts | Own exactly one supported CPU lifecycle adapter and expose opaque owner-bound compile/prepare handles plus a representation-level run seam for integration and tests. Reject zero/pass-through and mixed/multi-partition preparation; do not claim nonexistent schedule composition. |
| [0002](tasks/0002-standard-built-in-composition.md) | Standard built-in composition | Complete | 0001; supported built-in backend adapters | Added a distinct ordinary `Engine.standard()` construction path that owns one fresh CPU-only built-in composition without exposing `CpuBackendIntegration`, advanced handles, or inward integration contracts. Preserved `AdvancedEngine.takeOwnership(...)`; mixed-backend success still requires another concrete adapter and a justified complete schedule-composition contract. |
| [0003](tasks/0003-typed-logical-input-binding-and-published-result-access.md) | Typed logical input binding and published-result access | Complete | 0001–0002; [Compiler 0006B4](../compiler/tasks/0006b4-stable-caller-input-tensor-identity-bindings.md); current publication/Prepare/Runtime contracts | Added the ordinary compile, prepare, TensorId-based host-input binding, synchronous run, and metadata-only publication/result lifecycle without exposing inward identities, coordinates, representations, or advanced handles. |
| [0004](tasks/0004-explicit-host-materialization-boundary.md) | Explicit host materialization boundary | Complete | 0003; Runtime 0015; CPU 0010G | Lazily copies one exact selected publication occurrence into a detached immutable bounded `HostTensorValue` with canonical row-major big-endian bytes. Serializes copying against result/Engine close and exposes no representation, backend storage, arena, `MemorySegment`, `ValueId`, or slot. |
| [0005](tasks/0005-one-shot-forward-convenience.md) | Engine-owned one-shot forward convenience | Complete | 0002–0004; Runtime 0015; CPU 0010G | Added exact single-output and ordered-output `Engine.forward(...)` overloads with explicit logical inputs and one aggregate canonical-byte limit. Each call freshly compiles, prepares, runs, materializes every forward publication in order, and cleans up under one Engine admission; it adds no Tensor execution or cache. |
| [0005A](tasks/0005a-automatic-input-discovery-and-compute-convenience.md) | Automatic input discovery and compute convenience | Complete | 0005; Compiler 0006B4 | Replaced the two explicit-input one-shot `forward(...)` methods with four `compute(...)` overloads. Inventories reachable provenance-free Tensor leaves transiently by object identity, then selects only matching Tensors in authoritative final `CompiledGraph.inputs()` order; adds no Compiler IR traversal, liveness inference, retained Tensor state, or cache. |
| [0006](tasks/0006-one-shot-scalar-objective-backward-convenience.md) | Engine-owned one-shot scalar-objective backward convenience | Complete | 0005A; Compiler 0006B5; Prepare 0005; CPU 0010H | Added one explicit-target `Engine.backward(...)` call returning a detached scalar objective plus immutable target-aligned gradients. Reuses 0005A's transient leaf-inventory/final-binding selection seam with no explicit input list and Compiler's absent scalar unit seed and ERROR policy through the completed source-only constant chain; retains the explicit-seed ordinary compile and advanced full-request paths. |
| [0006A](tasks/0006a-representative-tuning-execution-and-safe-fallback.md) | Representative tuning execution and safe fallback foundation | Complete | 0003; 0006; Runtime 0010/0015; Prepare 0004; CPU 0010I; reviewed Config 0006A and tools/tuning 0001 contracts | Added package-private representative-input binding, synchronous complete trial execution, cleanup, failure isolation, fresh selected preparation, and deterministic strict/allowed fallback. Adds no public API, tuning algorithm, cache work, or tools/tuning dependency. |
| [0007](tasks/0007-optional-model-autotuning-composition.md) | Optional model-autotuning composition | Complete | 0002; 0005; [0006A](tasks/0006a-representative-tuning-execution-and-safe-fallback.md); Config 0006A; tools/tuning 0001; [CPU 0010I](../../backends/cpu/tasks/0010i-supported-cpu-local-workload-tuning-composition-adapter.md) | Added one CPU-only public representative request with caller-defined model identity; maps the sole handoff to occurrence 0/partition 0/weight 1, invokes cache-first tuning, and returns fresh selected preparation plus translated evidence or explicit safe fallback outside Runtime. |
| [0008](tasks/0008-engine-lifecycle-capability-checkpoint.md) | Engine lifecycle capability checkpoint | Complete | 0001–0007; Compiler 0006B3–0006B6; Prepare 0003/0003A/0004/0005; Runtime 0010/0012/0014/0015; CPU 0008/0008A/0010F–0010I; Config 0006A; tools/tuning 0001 | Consolidated repository-wide evidence and current documentation for the standard, advanced, typed, materialized, one-shot forward/backward, bounded tuning/fallback, cleanup, failure, concurrency, dependency, and public dimensional-convolution execution boundaries. |
| [0008A](tasks/0008a-representative-complete-plan-correctness-oracle.md) | Representative complete-plan correctness oracle | Complete | 0004, 0006A–0008; Runtime 0015; CPU 0010G and 0010J | Added the smallest package-private Engine oracle that captures all ordered representative publications as bounded canonical bytes and compares later complete recipes by exact represented-bit equality. It owns no timing, tuning algorithm, cache, public request, selected preparation, or fallback policy. |
| [0009](tasks/0009-public-complete-plan-autotuning-composition.md) | Public complete-plan autotuning composition | Complete | 0006A–0008A; Config 0006B; tools/tuning 0002; CPU 0010I–0010J; Prepare 0004; Runtime 0010/0015 | Composed the authenticated Phase-1 decision into CPU's complete-plan batch, adapted exact correctness and fresh execution to generic Phase 2, freshly prepared the authenticated winner, and extended existing public evidence without a new request, operation, or top-level public type. |
| [0010](tasks/0010-prepared-handle-ownership-and-closure.md) | Prepared-handle ownership and closure | Complete | Runtime 0016; Prepare 0006; 0001–0009 | Made ordinary and advanced prepared handles explicit closeable owners, closed every one-shot/trial/loser/rollback preparation exactly once, and closed retained preparations after results and before backend integration shutdown. |
| [0011](tasks/0011-compile-artifact-projection-boundary-reconciliation.md) | Compile-artifact projection boundary reconciliation | Complete | 0010; Prepare 0003A/0005/0006; CPU 0010F/0010H–0010J | Restored Engine/Prepare ownership of `CompileArtifacts` orchestration, removed CPU's production Compiler edge, and preserved CPU preparation/tuning behavior through stable projections. |
| [0012](tasks/0012-runtime-prepare-engine-authority-reconciliation.md) | Runtime, Prepare, and Engine authority reconciliation | Complete | 0011; Compiler 0006B8; current Engine/Prepare/Runtime APIs | Reconciled and independently reviewed the root, scoped contract, and focused explanations against the current owner-bound CPU Engine lifecycle and ordered Runtime schedule-step model. |
| [0013](tasks/0013-prepared-execution-guide-status-reconciliation.md) | Prepared execution guide status reconciliation | Ready | 0012; Model 0025M; current Engine lifecycle and autotuning APIs | Reconcile the two confirmed explanatory lifecycle drifts without changing authority or executable behavior. |

## Milestones and current frontier

- Explicit CPU composition and advanced lifecycle are Complete through 0002.
- Ordinary typed compile/prepare/run, publication, host materialization, compute, and backward are
  Complete through 0006.
- Representative and complete-plan autotuning composition plus its capability checkpoint are
  Complete through 0009.
- Prepared-handle ownership, closure, and compile-artifact projection reconciliation are Complete
  through 0011.
- [0013](tasks/0013-prepared-execution-guide-status-reconciliation.md) is `Ready` as the sole
  active frontier after 0012 and Model 0025M. It is limited to the confirmed prepared-execution
  guide and focused boundary-explanation drift.

## Live risks and gates

- Expanding standard composition requires another supported lifecycle adapter and a real
  complete mixed-owner schedule-contribution contract. Fixed CPU-only composition must not imply
  generic discovery or mixed-backend execution.
- Preserve one owner for every prepared handle and close all temporary, losing, rollback, and
  retained preparations in the established order. Never duplicate Runtime's lease protocol.
- Keep public materialization detached and explicit; do not expose Runtime representations,
  backend storage, or inward SPI types through ordinary signatures.
- Metal 0004 owns Metal route candidates and cache compatibility. It does not require a new
  Engine task or weaken backend-private route ownership.
- Preserve 0011's repaired boundary: Engine/Prepare own Compiler aggregates and the single
  projection, while concrete CPU production remains Compiler-free.

## Authorized drift-remediation exception

The user's requested repair order authorizes sequential drift remediation ahead of the normal
post-Metal frontier reassessment: Engine 0011 repaired the CPU boundary, Compiler 0006B8 repaired
autograd documentation, and Engine 0012 repaired the first confirmed remaining-drift cluster.
Their implementations and required reviews are complete. Engine 0013 is the next separately owned
confirmed drift repair; later drift remains undetailed and is not `Ready`.

## Status normalization

The task table and linked task status/results are controlling. Engine 0013 is the sole `Ready`
frontier. Metal 0002–0004 are Complete. No later Engine or drift-remediation task is detailed.

## History and update policy

Detailed results, validation commands, context identifiers, audits, and past ordering exceptions
remain in linked task files and Git history. Update this map only when ownership, dependencies,
task order/status, a milestone, or a live gate changes. Put executable scope and evidence in the
task brief and follow the [planning guide](../../planning-guide.md).
