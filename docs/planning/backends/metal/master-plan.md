# Metal Backend Master Plan

## Goal

Implement Metal capability, MPSGraph and custom-kernel preparation, storage, native integration, and execution.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)

## Scope

- Metal capability provider
- MPSGraph and custom-kernel lowering
- Metal executables, storage, workspace, and materialization
- native bridge and typed tracing
- typed, version-controlled, tested MPSGraph/custom-route candidate generators and compatible
  workload-cache lookup during prepare

## Out of scope

- global autograd
- public Tensor ownership
- engine dependency
- training-owned Metal optimizer bridge

## Module invariants

- Metal prepare owns lowering and route selection.
- Metal-specific optimizer execution remains in backend prepare or kernels.
- Metal backend never depends on engine.
- Metal candidate generators return complete valid route-specific configurations and remain
  opaque to shared tuning orchestration.
- Safe Metal heuristics remain correct without a compatible tuning result.

## Allowed dependencies

- modules/model
- modules/config
- modules/planning
- modules/runtime
- modules/prepare
- modules/backend-contract
- modules/trace

## Forbidden dependencies

- modules/engine
- extensions/training implementation dependencies

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | Metal capability and native foundation | Draft | Stable shared planning, runtime, prepare, backend-contract, and trace contracts | Establish truthful capability and native lifecycle contracts. |
| 0002 | Metal prepared execution and routes | Draft | 0001 | Add validated MPSGraph/custom-route lowering, storage, materialization, and execution. |
| 0003 | Typed Metal route candidate generators and cache compatibility | Draft | 0002, opaque prepare/tuning boundary and artifact versioning | Add colocated typed complete-candidate generation and canonical workload compatibility without exposing Metal knobs to planning or shared parameter bags. |


## Milestones

- Capability and bridge
- MPSGraph preparation
- Custom routes, storage, and conformance

## Current status

Draft.

This backend is not yet planned in detail. Detailed task specifications will be created when it becomes the current or next implementation frontier.

## Open questions

- Exact route-specific configuration records, target fingerprints, and candidate-schema versions
  wait for implemented Metal routes and the shared opaque orchestration consumer.

## Decisions made

- The implementation must follow the current architecture contract.
- Legacy code is capability evidence only; new implementation is written from scratch.
- Operation family selects the appropriate Metal candidate generator but is not a universal cache
  key. Model tuning may compare complete plans while Metal retains route and lowering ownership.

## Risks

- Moving Metal lowering into shared prepare or training.
- Exposing private Metal candidate fields through generic maps, reflection, or shared string
  dispatch.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).
