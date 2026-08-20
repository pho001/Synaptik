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
- compile and prepare orchestration
- composition of runtime, validators, tracing, and backends
- explicit typed caller-input binding and published-result access
- explicit host materialization/publication needed by downstream persistence consumers

## Out of scope

- kernel implementations
- backend internals
- graph optimizer passes
- runtime service locator and core reflective discovery

## Module invariants

- Engine is the outer composition root.
- Backends are registered explicitly.
- Concrete backends never depend on engine.

## Allowed dependencies

- modules/compiler
- modules/runtime
- modules/prepare
- modules/config
- modules/trace
- concrete backend modules

## Forbidden dependencies

- No inward module may depend on engine.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | Explicit engine composition and lifecycle facade | Draft | CPU reference backend; Compiler/Prepare/Runtime contracts | Compose explicitly registered backends and expose the compile, prepare, and run lifecycle without discovery or inward dependencies. |
| 0002 | Typed input binding and published output access | Draft | 0001; Runtime publication/result-access extension; Prepare source/publication mapping | Map caller Tensor/host inputs and logical publications to Runtime coordinates, preserve aliases and ownership, and expose typed result values without leaking backend representations. |
| 0003 | Explicit host materialization boundary | Draft | 0002; concrete-backend host-transfer routes | Materialize selected current Tensor/state values through prepared execution and publication into bounded caller-owned host payloads; add no backend access to NN, Training, or Checkpoint. |
| 0004 | Engine lifecycle capability checkpoint | Draft | 0001–0003; Compiler 0006B; CPU 0008E | Validate composition, typed input/output ownership, host materialization, cleanup, concurrency, architecture tests, documentation, and representative NCW Conv1d composition plus NCHW Conv2d and NCDHW Conv3d forward execution before persistence adapters or NN convolution integration depend on Engine. |


## Milestones

- Explicit backend composition
- Compile facade
- Prepare and run lifecycle facade

## Current status

Draft. Model/training checkpoint persistence depends on the future task 0003 boundary because the
current Runtime `RunResult` privately retains representations and exposes no value or storage
access. A checkpoint plan must not bypass that gap by reading backend storage from NN or Training.

The dimensional-convolution program adds no Engine operation switch or convolution-specific
binding API. Task 0004 will exercise representative rank-one composition and first-class rank-two
and rank-three convolution through the same typed logical-input, Prepare, Runtime, and publication
mapping established by tasks 0001–0003. This is the execution-readiness gate for the later NN
layer integration checkpoint; it does not move shape inference, lowering, or kernel selection into
Engine.

This module is not yet planned in detail. Detailed task specifications will be created when it becomes the current or next implementation frontier.

## Open questions

- Define the public materialized host payload, ownership, size limits, cancellation/failure, and
  close behavior without exposing concrete backend representation types.
- Decide whether materializing already host-backed leaves can use a proven direct fast path while
  preserving the same public ownership and validation contract.

## Decisions made

- The implementation must follow the current architecture contract.
- Legacy code is capability evidence only; new implementation is written from scratch.

## Risks

- Becoming a service locator or absorbing backend implementation details.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).
